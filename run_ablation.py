from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cross_market_transformer import (
    CrossMarketTransformerSharedHeadModel,
    HKTransformerOnlyModel,
    Trainer,
    build_multi_company_dataset,
    discover_cleaned_pairs,
    numpy_collate_fn,
)
from cross_market_transformer.trainer import (
    _information_coefficient,
    _peak_trough_metrics_from_event_logits,
)
from minimal_config import (
    DATASET_ROOT,
    HK_LOOKBACK,
    JOINT_TARGET_TASK_TYPE,
    MODEL_CONFIG,
    NORMALIZATION_MODE,
    P_INDEX_GAP_THRESHOLD,
    P_INDEX_MODE,
    ROLLING_NORMALIZATION_WINDOW,
    TARGET_COL,
    USE_US_PREV_NIGHT,
    US_LOOKBACK,
    make_task_model_config,
    make_task_train_config,
)


TRAIN_COMPANY_COUNT = 25


def _make_loader(dataset, train_config):
    return DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=numpy_collate_fn,
    )


def _reindex_company_specs(company_specs):
    reindexed_specs = []
    for new_company_id, spec in enumerate(company_specs):
        updated_spec = dict(spec)
        updated_spec["company_id"] = new_company_id
        reindexed_specs.append(updated_spec)
    return reindexed_specs


def split_company_specs(company_specs, train_company_count: int = TRAIN_COMPANY_COUNT):
    if len(company_specs) <= train_company_count:
        raise ValueError(
            f"Need more than {train_company_count} company pairs for held-out-company testing; "
            f"found {len(company_specs)}."
        )
    train_specs = _reindex_company_specs(company_specs[:train_company_count])
    test_specs = _reindex_company_specs(company_specs[train_company_count:])
    return train_specs, test_specs


def print_company_split(split_name: str, company_specs) -> None:
    print(f"{split_name} companies ({len(company_specs)}):")
    for spec in company_specs:
        print(
            f"  [{spec['company_id']:02d}] {spec['company_name']}: "
            f"HK={spec['hk_path']} | US={spec['us_path']}"
        )
    print()


def make_joint_dataset(company_specs, p_index_mode: str):
    return build_multi_company_dataset(
        company_specs=company_specs,
        hk_lookback=HK_LOOKBACK,
        us_lookback=US_LOOKBACK,
        task_type=JOINT_TARGET_TASK_TYPE,
        target_col=TARGET_COL,
        multiclass_num_classes=MODEL_CONFIG.num_classes,
        use_us_prev_night=USE_US_PREV_NIGHT,
        normalization_mode=NORMALIZATION_MODE,
        rolling_normalization_window=ROLLING_NORMALIZATION_WINDOW,
        p_index_mode=p_index_mode,
        p_index_gap_threshold=P_INDEX_GAP_THRESHOLD,
    )


def build_experiments():
    return [
        {
            "name": "main_shared_head",
            "model_cls": CrossMarketTransformerSharedHeadModel,
            "p_index_mode": P_INDEX_MODE,
        },
        {
            "name": "hk_only_shared_head",
            "model_cls": HKTransformerOnlyModel,
            "p_index_mode": "none",
        },
        {
            "name": "main_no_p_index",
            "model_cls": CrossMarketTransformerSharedHeadModel,
            "p_index_mode": "none",
        },
    ]


def _peak_trough_binary_targets_from_class_labels(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.long().view(-1)
    return torch.stack([(labels == 2).float(), (labels == 0).float()], dim=1)


def _random_walk_joint_metrics(dataset) -> dict[str, float]:
    target = dataset.target.float()
    r1_target = target[:, 0]
    r1_preds = torch.zeros_like(r1_target)
    r1_mse = torch.mean((r1_preds - r1_target) ** 2).item()
    r1_mae = torch.mean(torch.abs(r1_preds - r1_target)).item()

    # Random walk expected-return view predicts zero next-day return and no
    # peak/trough event signal.
    peak_trough_logits = torch.full((target.size(0), 2), -10.0, dtype=torch.float32)
    peak_trough_target = _peak_trough_binary_targets_from_class_labels(target[:, 1])

    train_config = make_task_train_config(JOINT_TARGET_TASK_TYPE)
    pos_weight = None
    if train_config.class_weight is not None:
        pos_weight = torch.tensor(train_config.class_weight, dtype=torch.float32)
    peak_trough_loss = F.binary_cross_entropy_with_logits(
        peak_trough_logits,
        peak_trough_target,
        pos_weight=pos_weight,
    ).item()
    loss = train_config.r1_loss_weight * r1_mse + train_config.peak_trough_loss_weight * peak_trough_loss

    metrics = {
        "loss": loss,
        "r1_mse": r1_mse,
        "r1_rmse": r1_mse ** 0.5,
        "r1_mae": r1_mae,
        "r1_ic": _information_coefficient(r1_preds, r1_target),
        "r1_sign_accuracy": ((r1_preds >= 0) == (r1_target >= 0)).float().mean().item(),
    }
    metrics.update(_peak_trough_metrics_from_event_logits(peak_trough_logits, peak_trough_target))
    return metrics


def evaluate_random_walk_baseline(joint_train_set, joint_test_set):
    print("=" * 100)
    print("Running baseline: random_walk")
    train_metrics = _random_walk_joint_metrics(joint_train_set)
    test_metrics = _random_walk_joint_metrics(joint_test_set)
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics: {test_metrics}")
    return [("random_walk", "train_loss", train_metrics["loss"], test_metrics)]


def run_model_experiment(run_name, model_cls, p_index_mode, train_specs, test_specs):
    print("=" * 100)
    print(f"Running experiment: {run_name} ({JOINT_TARGET_TASK_TYPE}, p_index_mode={p_index_mode})")

    train_set = make_joint_dataset(train_specs, p_index_mode=p_index_mode)
    test_set = make_joint_dataset(test_specs, p_index_mode=p_index_mode)
    train_config = make_task_train_config(JOINT_TARGET_TASK_TYPE)
    train_config.checkpoint_name = f"{run_name}.pt"
    train_config.history_plot_name = f"{run_name}_history.png"
    train_config.threshold_sweep_name = f"{run_name}_threshold_sweep.csv"
    train_config.threshold_sweep_plot_name = f"{run_name}_threshold_sweep.png"

    train_loader = _make_loader(train_set, train_config)
    test_loader = _make_loader(test_set, train_config)

    model_config = make_task_model_config(
        JOINT_TARGET_TASK_TYPE,
        num_classes=3,
        num_companies=len(train_specs),
        hk_input_dim=train_set.x_hk.shape[-1],
        us_input_dim=train_set.x_us.shape[-1],
        p_index_mode=p_index_mode,
        p_index_gap_feature_dim=train_set.p_index_gap_features.shape[-1],
    )

    torch.manual_seed(42)
    model = model_cls(model_config)
    trainer = Trainer(
        model=model,
        train_config=train_config,
        task_type=model_config.task_type,
        num_classes=model_config.num_classes,
    )
    fit_result = trainer.fit(train_loader, val_loader=None, test_loader=test_loader)
    test_metrics = trainer.evaluate(test_loader)

    print(f"Finished experiment: {run_name}")
    print(f"Best {fit_result['best_score_name']}: {fit_result['best_score']:.6f}")
    print(f"Test metrics: {test_metrics}")
    return run_name, fit_result["best_score_name"], fit_result["best_score"], test_metrics


def main() -> None:
    torch.manual_seed(42)
    company_specs = discover_cleaned_pairs(DATASET_ROOT)

    print("Loaded leak-free cleaned company pairs:")
    for spec in company_specs:
        print(
            f"  [{spec['company_id']:02d}] {spec['company_name']}: "
            f"HK={spec['hk_path']} | US={spec['us_path']}"
        )
    print()

    train_specs, test_specs = split_company_specs(company_specs)
    print_company_split("Train", train_specs)
    print_company_split("Test", test_specs)

    base_train_set = make_joint_dataset(train_specs, p_index_mode=P_INDEX_MODE)
    base_test_set = make_joint_dataset(test_specs, p_index_mode=P_INDEX_MODE)
    results = evaluate_random_walk_baseline(base_train_set, base_test_set)

    for experiment in build_experiments():
        results.append(
            run_model_experiment(
                run_name=experiment["name"],
                model_cls=experiment["model_cls"],
                p_index_mode=experiment["p_index_mode"],
                train_specs=train_specs,
                test_specs=test_specs,
            )
        )

    print("=" * 100)
    print("Baseline and ablation summary")
    for exp_name, best_score_name, best_score, test_metrics in results:
        print(f"{exp_name:24s} | {best_score_name}={best_score:.6f} | test={test_metrics}")


if __name__ == "__main__":
    main()
