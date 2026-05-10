from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from cross_market_transformer import (
    CrossMarketDataset,
    CrossMarketTransformerSharedHeadModel,
    Trainer,
    build_multi_company_dataset,
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

HELD_OUT_SHARED_HEAD_COMPANIES = {"zai_lab", "noah"}

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


def _with_shared_company_id(dataset: CrossMarketDataset) -> CrossMarketDataset:
    return CrossMarketDataset(
        x_hk=dataset.x_hk,
        x_us=dataset.x_us,
        hk_time_delta=dataset.hk_time_delta,
        us_time_delta=dataset.us_time_delta,
        company_id=torch.zeros_like(dataset.company_id),
        us_open_prev_night=dataset.us_open_prev_night,
        us_sessions_since_last_hk=dataset.us_sessions_since_last_hk,
        latest_us_gap_days=dataset.latest_us_gap_days,
        p_index_gap_features=dataset.p_index_gap_features,
        target=dataset.target,
        hk_padding_mask=dataset.hk_padding_mask,
        us_padding_mask=dataset.us_padding_mask,
    )


def _reindex_company_specs(company_specs):
    remapped = []
    for new_id, spec in enumerate(company_specs):
        remapped.append({**spec, "company_id": new_id})
    return remapped


def split_company_specs(company_specs):
    base_specs = [spec for spec in company_specs if spec["company_name"] not in HELD_OUT_SHARED_HEAD_COMPANIES]
    held_out_specs = [spec for spec in company_specs if spec["company_name"] in HELD_OUT_SHARED_HEAD_COMPANIES]
    return _reindex_company_specs(base_specs), _reindex_company_specs(held_out_specs)


def make_joint_datasets(base_specs, held_out_specs):
    train_set = build_multi_company_dataset(
        company_specs=base_specs,
        hk_lookback=HK_LOOKBACK,
        us_lookback=US_LOOKBACK,
        task_type="regression_peak_trough",
        target_col=TARGET_COL,
        multiclass_num_classes=MODEL_CONFIG.num_classes,
        use_us_prev_night=USE_US_PREV_NIGHT,
        normalization_mode=NORMALIZATION_MODE,
        rolling_normalization_window=ROLLING_NORMALIZATION_WINDOW,
        p_index_mode=P_INDEX_MODE,
        p_index_gap_threshold=P_INDEX_GAP_THRESHOLD,
    )
    test_set = build_multi_company_dataset(
        company_specs=held_out_specs,
        hk_lookback=HK_LOOKBACK,
        us_lookback=US_LOOKBACK,
        task_type="regression_peak_trough",
        target_col=TARGET_COL,
        multiclass_num_classes=MODEL_CONFIG.num_classes,
        use_us_prev_night=USE_US_PREV_NIGHT,
        normalization_mode=NORMALIZATION_MODE,
        rolling_normalization_window=ROLLING_NORMALIZATION_WINDOW,
        p_index_mode=P_INDEX_MODE,
        p_index_gap_threshold=P_INDEX_GAP_THRESHOLD,
    )
    return _with_shared_company_id(train_set), _with_shared_company_id(test_set)


def main() -> None:
    torch.manual_seed(42)
    company_specs = discover_cleaned_pairs(DATASET_ROOT)
    base_specs, held_out_specs = split_company_specs(company_specs)

    if len(held_out_specs) != len(HELD_OUT_SHARED_HEAD_COMPANIES):
        raise ValueError(
            "Held-out shared-head companies were not found exactly as expected: "
            f"{sorted(HELD_OUT_SHARED_HEAD_COMPANIES)}"
        )

    joint_train_set, joint_test_set = make_joint_datasets(base_specs, held_out_specs)

    print("Training companies for shared-head model using cleaned inputs:")
    for spec in base_specs:
        print(
            f"  [{spec['company_id']:02d}] {spec['company_name']}: "
            f"HK={spec['hk_path']} | US={spec['us_path']}"
        )
    print()
    print("Held-out test companies for shared-head model using cleaned inputs:")
    for spec in held_out_specs:
        print(
            f"  [{spec['company_id']:02d}] {spec['company_name']}: "
            f"HK={spec['hk_path']} | US={spec['us_path']}"
        )
    print()

    results = []
    for task_name, task_type in TASK_RUNS:
        print("=" * 100)
        print(f"Running held-out shared-head task: {task_name} ({task_type})")
        train_set = retarget_regression_peak_trough_dataset(joint_train_set, task_type)
        test_set = retarget_regression_peak_trough_dataset(joint_test_set, task_type)

        model_config = make_task_model_config(
            task_type,
            num_classes=3,
            num_companies=1,
            hk_input_dim=train_set.x_hk.shape[-1],
            us_input_dim=train_set.x_us.shape[-1],
            p_index_gap_feature_dim=train_set.p_index_gap_features.shape[-1],
        )
        train_config = make_task_train_config(task_type)
        train_config.checkpoint_name = f"cross_market_shared_head_{task_name}.pt"
        train_config.history_plot_name = f"cross_market_shared_head_{task_name}_history.png"
        train_config.threshold_sweep_name = f"cross_market_shared_head_{task_name}_threshold_sweep.csv"
        train_config.threshold_sweep_plot_name = f"cross_market_shared_head_{task_name}_threshold_sweep.png"

        train_loader = _make_loader(train_set, train_config)
        test_loader = _make_loader(test_set, train_config)

        torch.manual_seed(42)
        model = CrossMarketTransformerSharedHeadModel(model_config)
        trainer = Trainer(
            model=model,
            train_config=train_config,
            task_type=model_config.task_type,
            num_classes=model_config.num_classes,
        )

        fit_result = trainer.fit(train_loader, val_loader=None, test_loader=test_loader)
        test_metrics = trainer.evaluate(test_loader)
        results.append((task_name, fit_result["best_score_name"], fit_result["best_score"], test_metrics))

        print(f"Best {fit_result['best_score_name']}: {fit_result['best_score']:.6f}")
        print(f"Test metrics: {test_metrics}")

    print("=" * 100)
    print("Held-out shared-head summary")
    for task_name, best_score_name, best_score, test_metrics in results:
        print(f"{task_name:12s} | {best_score_name}={best_score:.6f} | test={test_metrics}")


if __name__ == "__main__":
    main()
