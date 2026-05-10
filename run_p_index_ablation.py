from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

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


P_INDEX_EXPERIMENTS = [
    (
        "p_index_as_feature",
        "feature",
        "A. Current-style baseline: keep P_index as a normal HK/US sequence feature.",
    ),
    (
        "no_p_index",
        "none",
        "B. Remove P_index entirely from sequence features and auxiliary signals.",
    ),
    (
        "p_index_feature_plus_gap",
        "feature_plus_gap",
        "C. Keep P_index as a normal feature and inject soft HK-US discrepancy features into the pre-open query.",
    ),
]


def _make_loader(dataset, train_config):
    return DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=numpy_collate_fn,
    )


def _make_joint_datasets(company_specs, p_index_mode: str):
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
        p_index_mode=p_index_mode,
        p_index_gap_threshold=P_INDEX_GAP_THRESHOLD,
    )
    return train_set, val_set, test_set


def _write_summary(results: list[dict[str, float | str]], path: Path) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved P-index ablation summary to {path}")


def main() -> None:
    company_specs = discover_cleaned_pairs(DATASET_ROOT)
    print("Loaded leak-free cleaned company pairs:")
    for spec in company_specs:
        print(
            f"  [{spec['company_id']:02d}] {spec['company_name']}: "
            f"HK={spec['hk_path']} | US={spec['us_path']}"
        )
    print()

    results = []
    for exp_name, p_index_mode, description in P_INDEX_EXPERIMENTS:
        print("=" * 100)
        print(f"Running P-index experiment: {exp_name}")
        print(description)

        joint_train_set, joint_val_set, joint_test_set = _make_joint_datasets(
            company_specs=company_specs,
            p_index_mode=p_index_mode,
        )
        hk_input_dim = joint_train_set.x_hk.shape[-1]
        us_input_dim = joint_train_set.x_us.shape[-1]
        print(
            f"Input dims: hk_input_dim={hk_input_dim}, us_input_dim={us_input_dim}, "
            f"p_index_gap_features={tuple(joint_train_set.p_index_gap_features.shape[1:])}"
        )

        for task_name, task_type in TASK_RUNS:
            run_name = f"{exp_name}_{task_name}"
            print("-" * 100)
            print(f"Running P-index task: {run_name} ({task_type})")

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

            torch.manual_seed(42)
            model_config = replace(
                deepcopy(MODEL_CONFIG),
                task_type=task_type,
                num_classes=3,
                num_companies=len(company_specs),
                hk_input_dim=hk_input_dim,
                us_input_dim=us_input_dim,
                p_index_mode=p_index_mode,
                p_index_gap_feature_dim=joint_train_set.p_index_gap_features.shape[-1],
            )
            model = CrossMarketTransformerSharedHeadModel(model_config)
            trainer = Trainer(
                model=model,
                train_config=train_config,
                task_type=model_config.task_type,
                num_classes=model_config.num_classes,
            )

            fit_result = trainer.fit(train_loader, val_loader, test_loader=test_loader)
            test_metrics = trainer.evaluate(test_loader)
            row = {
                "experiment": exp_name,
                "task": task_name,
                "p_index_mode": p_index_mode,
                "hk_input_dim": float(hk_input_dim),
                "us_input_dim": float(us_input_dim),
                "best_score_name": fit_result["best_score_name"],
                "best_score": float(fit_result["best_score"]),
            }
            row.update({f"test_{key}": float(value) for key, value in test_metrics.items()})
            results.append(row)

            print(f"Finished P-index task: {run_name}")
            print(f"Best {fit_result['best_score_name']}: {fit_result['best_score']:.6f}")
            print(f"Test metrics: {test_metrics}")

    print("=" * 100)
    print("P-index ablation summary")
    for row in results:
        print(row)
    _write_summary(results, Path(TRAIN_CONFIG.checkpoint_dir) / "p_index_ablation_summary.csv")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
