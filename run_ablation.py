from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cross_market_transformer import (
    HKTransformerOnlyModel,
    HKUSConcatBaseline,
    Trainer,
    build_multi_company_splits,
    discover_cleaned_pairs,
    numpy_collate_fn,
    retarget_regression_peak_trough_dataset,
)
from minimal_config import (
    DATASET_ROOT,
    HK_LOOKBACK,
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
from cross_market_transformer.trainer import (
    _information_coefficient,
    _peak_trough_metrics_from_event_logits,
)

TASK_RUNS = [
    ("r1", "regression"),
    ("peak_trough", "peak_trough_classification"),
]


def _make_loader(dataset, train_config):
    return DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=numpy_collate_fn,
    )


def make_joint_datasets(company_specs):
    train_set, val_set, test_set = build_multi_company_splits(
        company_specs=company_specs,
        hk_lookback=HK_LOOKBACK,
        us_lookback=US_LOOKBACK,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        task_type="regression_peak_trough",
        target_col=TARGET_COL,
        multiclass_num_classes=MODEL_CONFIG.num_classes,
        use_us_prev_night=USE_US_PREV_NIGHT,
        normalization_mode=NORMALIZATION_MODE,
        rolling_normalization_window=ROLLING_NORMALIZATION_WINDOW,
        p_index_mode=P_INDEX_MODE,
        p_index_gap_threshold=P_INDEX_GAP_THRESHOLD,
    )
    return train_set, val_set, test_set


def build_experiments():
    return [
        ("hk_us_concat", HKUSConcatBaseline),
        ("hk_transformer_only", HKTransformerOnlyModel),
    ]


def _random_walk_regression_metrics(dataset) -> dict[str, float]:
    target = dataset.target.float().view(-1)
    preds = torch.zeros_like(target)
    mse = torch.mean((preds - target) ** 2).item()
    mae = torch.mean(torch.abs(preds - target)).item()
    return {
        "loss": mse,
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": mae,
        "ic": _information_coefficient(preds, target),
        "sign_accuracy": ((preds >= 0) == (target >= 0)).float().mean().item(),
    }


def _random_walk_peak_trough_metrics(dataset) -> dict[str, float]:
    target = dataset.target.float()
    logits = torch.full_like(target, -10.0)  # Random walk maps to near-zero peak/trough event probabilities.

    pos_weight = None
    peak_train_config = make_task_train_config("peak_trough_classification")
    if peak_train_config.class_weight is not None:
        if len(peak_train_config.class_weight) == 2:
            pos_weight = torch.tensor(peak_train_config.class_weight, dtype=torch.float32)
        elif len(peak_train_config.class_weight) == 3:
            pos_weight = torch.tensor(
                [peak_train_config.class_weight[2], peak_train_config.class_weight[0]],
                dtype=torch.float32,
            )
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight).item()
    metrics = _peak_trough_metrics_from_event_logits(logits, target)
    return {"loss": loss, **metrics}


def evaluate_random_walk_baseline(joint_train_set, joint_val_set, joint_test_set):
    results = []
    evaluators = {
        "regression": _random_walk_regression_metrics,
        "peak_trough_classification": _random_walk_peak_trough_metrics,
    }
    for task_name, task_type in TASK_RUNS:
        run_name = f"random_walk_{task_name}"
        print("=" * 100)
        print(f"Running baseline: {run_name} ({task_type})")
        train_set = retarget_regression_peak_trough_dataset(joint_train_set, task_type)
        val_set = retarget_regression_peak_trough_dataset(joint_val_set, task_type)
        test_set = retarget_regression_peak_trough_dataset(joint_test_set, task_type)
        evaluate = evaluators[task_type]
        train_metrics = evaluate(train_set)
        val_metrics = evaluate(val_set)
        test_metrics = evaluate(test_set)

        print(f"Train metrics: {train_metrics}")
        print(f"Val metrics: {val_metrics}")
        print(f"Test metrics: {test_metrics}")
        results.append((run_name, "val_loss", val_metrics["loss"], test_metrics))
    return results


def main() -> None:
    torch.manual_seed(42)
    company_specs = discover_cleaned_pairs(DATASET_ROOT)
    joint_train_set, joint_val_set, joint_test_set = make_joint_datasets(company_specs)
    hk_input_dim = joint_train_set.x_hk.shape[-1]
    us_input_dim = joint_train_set.x_us.shape[-1]
    p_index_gap_feature_dim = joint_train_set.p_index_gap_features.shape[-1]

    print("Loaded leak-free cleaned company pairs:")
    for spec in company_specs:
        print(
            f"  [{spec['company_id']:02d}] {spec['company_name']}: "
            f"HK={spec['hk_path']} | US={spec['us_path']}"
        )
    print()

    results = evaluate_random_walk_baseline(joint_train_set, joint_val_set, joint_test_set)
    for exp_name, model_cls in build_experiments():
        for task_name, task_type in TASK_RUNS:
            run_name = f"{exp_name}_{task_name}"
            print("=" * 100)
            print(f"Running experiment: {run_name} ({task_type})")

            train_set = retarget_regression_peak_trough_dataset(joint_train_set, task_type)
            val_set = retarget_regression_peak_trough_dataset(joint_val_set, task_type)
            test_set = retarget_regression_peak_trough_dataset(joint_test_set, task_type)
            train_config = make_task_train_config(task_type)
            train_config.checkpoint_name = f"{run_name}.pt"
            train_config.history_plot_name = f"{run_name}_history.png"
            train_config.threshold_sweep_name = f"{run_name}_threshold_sweep.csv"
            train_config.threshold_sweep_plot_name = f"{run_name}_threshold_sweep.png"

            train_loader = _make_loader(train_set, train_config)
            val_loader = _make_loader(val_set, train_config)
            test_loader = _make_loader(test_set, train_config)

            model_config = make_task_model_config(
                task_type,
                num_classes=3,
                num_companies=len(company_specs),
                hk_input_dim=hk_input_dim,
                us_input_dim=us_input_dim,
                p_index_gap_feature_dim=p_index_gap_feature_dim,
            )

            torch.manual_seed(42)
            model = model_cls(model_config)
            trainer = Trainer(
                model=model,
                train_config=train_config,
                task_type=model_config.task_type,
                num_classes=model_config.num_classes,
            )
            fit_result = trainer.fit(train_loader, val_loader, test_loader=test_loader)
            test_metrics = trainer.evaluate(test_loader)
            results.append((run_name, fit_result["best_score_name"], fit_result["best_score"], test_metrics))

            print(f"Finished experiment: {run_name}")
            print(f"Best {fit_result['best_score_name']}: {fit_result['best_score']:.6f}")
            print(f"Test metrics: {test_metrics}")

    print("=" * 100)
    print("Ablation summary")
    for exp_name, best_score_name, best_score, test_metrics in results:
        print(f"{exp_name:28s} | {best_score_name}={best_score:.6f} | test={test_metrics}")


if __name__ == "__main__":
    main()
