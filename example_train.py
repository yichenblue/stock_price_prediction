from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from cross_market_transformer import (
    CrossMarketTransformerSharedHeadModel,
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
    TRAIN_CONFIG,
    USE_US_PREV_NIGHT,
    US_LOOKBACK,
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


def _configure_train(task_name: str, task_type: str):
    train_config = deepcopy(TRAIN_CONFIG)
    train_config.checkpoint_name = f"cross_market_shared_head_pindex_feature_{task_name}.pt"
    train_config.history_plot_name = f"cross_market_shared_head_pindex_feature_{task_name}_history.png"
    train_config.threshold_sweep_name = f"cross_market_shared_head_pindex_feature_{task_name}_threshold_sweep.csv"
    train_config.threshold_sweep_plot_name = f"cross_market_shared_head_pindex_feature_{task_name}_threshold_sweep.png"
    train_config.save_threshold_sweep = task_type == "peak_trough_classification"
    if task_type == "regression":
        train_config.class_weight = None
    return train_config


def _run_task(
    task_name: str,
    task_type: str,
    joint_train_set,
    joint_val_set,
    joint_test_set,
    num_companies: int,
) -> tuple[str, str, float, dict[str, float]]:
    train_set = retarget_regression_peak_trough_dataset(joint_train_set, task_type)
    val_set = retarget_regression_peak_trough_dataset(joint_val_set, task_type)
    test_set = retarget_regression_peak_trough_dataset(joint_test_set, task_type)

    train_config = _configure_train(task_name, task_type)
    train_loader = _make_loader(train_set, train_config)
    val_loader = _make_loader(val_set, train_config)
    test_loader = _make_loader(test_set, train_config)

    model_config = replace(
        MODEL_CONFIG,
        task_type=task_type,
        num_classes=3,
        num_companies=num_companies,
        hk_input_dim=train_set.x_hk.shape[-1],
        us_input_dim=train_set.x_us.shape[-1],
        p_index_gap_feature_dim=train_set.p_index_gap_features.shape[-1],
    )
    torch.manual_seed(42)
    model = CrossMarketTransformerSharedHeadModel(model_config)
    trainer = Trainer(
        model=model,
        train_config=train_config,
        task_type=model_config.task_type,
        num_classes=model_config.num_classes,
    )

    fit_result = trainer.fit(train_loader, val_loader, test_loader=test_loader)
    test_metrics = trainer.evaluate(test_loader)

    print(f"Finished task: {task_name}")
    print(f"Best {fit_result['best_score_name']}:", fit_result["best_score"])
    print(f"Test metrics:", test_metrics)
    return task_name, fit_result["best_score_name"], fit_result["best_score"], test_metrics


def main() -> None:
    company_specs = discover_cleaned_pairs(DATASET_ROOT)
    print("Loaded leak-free cleaned company pairs:")
    for spec in company_specs:
        print(
            f"  [{spec['company_id']:02d}] {spec['company_name']}: "
            f"HK={spec['hk_path']} | US={spec['us_path']}"
        )
    print()

    joint_train_set, joint_val_set, joint_test_set = build_multi_company_splits(
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

    results = []
    for task_name, task_type in TASK_RUNS:
        print("=" * 100)
        print(f"Running main shared-head task: {task_name} ({task_type})")
        results.append(
            _run_task(
                task_name=task_name,
                task_type=task_type,
                joint_train_set=joint_train_set,
                joint_val_set=joint_val_set,
                joint_test_set=joint_test_set,
                num_companies=len(company_specs),
            )
        )

    print("=" * 100)
    print("Main model summary")
    for task_name, best_score_name, best_score, test_metrics in results:
        print(f"{task_name:12s} | {best_score_name}={best_score:.6f} | test={test_metrics}")
    print("Number of company pairs:", len(company_specs))


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
