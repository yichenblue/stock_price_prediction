from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from cross_market_transformer import (
    CrossMarketTransformerModel,
    HKTransformerOnlyModel,
    HKUSConcatBaseline,
    Trainer,
    build_multi_company_splits,
    discover_cleaned_pairs,
    numpy_collate_fn,
)
from minimal_config import (
    DATASET_ROOT,
    HK_LOOKBACK,
    MODEL_CONFIG,
    NORMALIZATION_MODE,
    ROLLING_NORMALIZATION_WINDOW,
    TARGET_COL,
    TRAIN_CONFIG,
    USE_US_PREV_NIGHT,
    US_LOOKBACK,
)


def _make_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=TRAIN_CONFIG.batch_size,
        shuffle=False,
        num_workers=TRAIN_CONFIG.num_workers,
        collate_fn=numpy_collate_fn,
    )


def make_dataloaders(company_specs):
    train_set, val_set, test_set = build_multi_company_splits(
        company_specs=company_specs,
        hk_lookback=HK_LOOKBACK,
        us_lookback=US_LOOKBACK,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        task_type=MODEL_CONFIG.task_type,
        target_col=TARGET_COL,
        multiclass_num_classes=MODEL_CONFIG.num_classes,
        use_us_prev_night=USE_US_PREV_NIGHT,
        normalization_mode=NORMALIZATION_MODE,
        rolling_normalization_window=ROLLING_NORMALIZATION_WINDOW,
    )
    return (
        _make_loader(train_set),
        _make_loader(val_set),
        _make_loader(test_set),
    )


def build_experiments():
    return [
        ("hk_us_concat", HKUSConcatBaseline),
        ("hk_transformer_only", HKTransformerOnlyModel),
        ("cross_market_transformer", CrossMarketTransformerModel),
    ]


def main() -> None:
    torch.manual_seed(42)
    company_specs = discover_cleaned_pairs(DATASET_ROOT)
    train_loader, val_loader, test_loader = make_dataloaders(company_specs)

    print("Loaded leak-free cleaned company pairs:")
    for spec in company_specs:
        print(
            f"  [{spec['company_id']:02d}] {spec['company_name']}: "
            f"HK={spec['hk_path']} | US={spec['us_path']}"
        )
    print()

    results = []
    for exp_name, model_cls in build_experiments():
        print("=" * 100)
        print(f"Running experiment: {exp_name}")

        model_config = replace(deepcopy(MODEL_CONFIG), num_companies=len(company_specs))
        train_config = deepcopy(TRAIN_CONFIG)
        train_config.checkpoint_name = f"{exp_name}.pt"
        train_config.history_plot_name = f"{exp_name}_history.png"
        train_config.threshold_sweep_name = f"{exp_name}_threshold_sweep.csv"
        train_config.threshold_sweep_plot_name = f"{exp_name}_threshold_sweep.png"

        model = model_cls(model_config)
        trainer = Trainer(
            model=model,
            train_config=train_config,
            task_type=model_config.task_type,
            num_classes=model_config.num_classes,
        )
        fit_result = trainer.fit(train_loader, val_loader, test_loader=test_loader)
        test_metrics = trainer.evaluate(test_loader)
        results.append((exp_name, fit_result["best_score_name"], fit_result["best_score"], test_metrics))

        print(f"Finished experiment: {exp_name}")
        print(f"Best {fit_result['best_score_name']}: {fit_result['best_score']:.6f}")
        print(f"Test metrics: {test_metrics}")

    print("=" * 100)
    print("Ablation summary")
    for exp_name, best_score_name, best_score, test_metrics in results:
        print(f"{exp_name:28s} | {best_score_name}={best_score:.6f} | test={test_metrics}")


if __name__ == "__main__":
    main()
