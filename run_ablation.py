from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch
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

    results = []
    for exp_name, model_cls in build_experiments():
        for task_name, task_type in TASK_RUNS:
            run_name = f"{exp_name}_{task_name}"
            print("=" * 100)
            print(f"Running experiment: {run_name} ({task_type})")

            train_set = retarget_regression_peak_trough_dataset(joint_train_set, task_type)
            val_set = retarget_regression_peak_trough_dataset(joint_val_set, task_type)
            test_set = retarget_regression_peak_trough_dataset(joint_test_set, task_type)
            train_config = deepcopy(TRAIN_CONFIG)
            train_config.checkpoint_name = f"{run_name}.pt"
            train_config.history_plot_name = f"{run_name}_history.png"
            train_config.threshold_sweep_name = f"{run_name}_threshold_sweep.csv"
            train_config.threshold_sweep_plot_name = f"{run_name}_threshold_sweep.png"
            train_config.save_threshold_sweep = task_type == "peak_trough_classification"
            if task_type == "regression":
                train_config.class_weight = None

            train_loader = _make_loader(train_set, train_config)
            val_loader = _make_loader(val_set, train_config)
            test_loader = _make_loader(test_set, train_config)

            model_config = replace(
                deepcopy(MODEL_CONFIG),
                task_type=task_type,
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
