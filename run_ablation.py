from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from cross_market_transformer import (
    CrossMarketDataset,
    CrossMarketTransformerModel,
    CrossMarketTransformerSharedHeadModel,
    HKTransformerOnlyModel,
    HKUSConcatBaseline,
    Trainer,
    build_multi_company_dataset,
    build_multi_company_splits,
    discover_standardized_pairs,
    numpy_collate_fn,
)
from minimal_config import (
    DATASET_ROOT,
    HK_LOOKBACK,
    MODEL_CONFIG,
    TARGET_COL,
    TRAIN_CONFIG,
    USE_US_PREV_NIGHT,
    US_LOOKBACK,
)

HELD_OUT_SHARED_HEAD_COMPANIES = {"zai_lab", "noah"}


def _make_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=TRAIN_CONFIG.batch_size,
        shuffle=False,
        num_workers=TRAIN_CONFIG.num_workers,
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
        target=dataset.target,
        hk_padding_mask=dataset.hk_padding_mask,
        us_padding_mask=dataset.us_padding_mask,
    )


def split_company_specs(company_specs):
    base_specs = [spec for spec in company_specs if spec["company_name"] not in HELD_OUT_SHARED_HEAD_COMPANIES]
    held_out_specs = [spec for spec in company_specs if spec["company_name"] in HELD_OUT_SHARED_HEAD_COMPANIES]
    return base_specs, held_out_specs


def make_standard_dataloaders(company_specs):
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
    )
    return (
        _make_loader(train_set),
        _make_loader(val_set),
        _make_loader(test_set),
    )


def make_shared_head_dataloaders(base_specs, held_out_specs):
    train_set = build_multi_company_dataset(
        company_specs=base_specs,
        hk_lookback=HK_LOOKBACK,
        us_lookback=US_LOOKBACK,
        task_type=MODEL_CONFIG.task_type,
        target_col=TARGET_COL,
        multiclass_num_classes=MODEL_CONFIG.num_classes,
        use_us_prev_night=USE_US_PREV_NIGHT,
    )
    test_set = build_multi_company_dataset(
        company_specs=held_out_specs,
        hk_lookback=HK_LOOKBACK,
        us_lookback=US_LOOKBACK,
        task_type=MODEL_CONFIG.task_type,
        target_col=TARGET_COL,
        multiclass_num_classes=MODEL_CONFIG.num_classes,
        use_us_prev_night=USE_US_PREV_NIGHT,
    )
    return (
        _make_loader(_with_shared_company_id(train_set)),
        None,
        _make_loader(_with_shared_company_id(test_set)),
    )


def build_experiments():
    return [
        ("hk_us_concat", HKUSConcatBaseline),
        ("hk_transformer_only", HKTransformerOnlyModel),
        ("cross_market_shared_head", CrossMarketTransformerSharedHeadModel),
        ("cross_market_transformer", CrossMarketTransformerModel),
    ]


def main() -> None:
    torch.manual_seed(42)
    company_specs = discover_standardized_pairs(DATASET_ROOT)
    base_specs, held_out_specs = split_company_specs(company_specs)

    if len(held_out_specs) != len(HELD_OUT_SHARED_HEAD_COMPANIES):
        raise ValueError(
            "Held-out shared-head companies were not found exactly as expected: "
            f"{sorted(HELD_OUT_SHARED_HEAD_COMPANIES)}"
        )

    standard_train_loader, standard_val_loader, standard_test_loader = make_standard_dataloaders(base_specs)
    shared_train_loader, shared_val_loader, shared_test_loader = make_shared_head_dataloaders(base_specs, held_out_specs)

    print("Training/evaluation company pairs for standard models:")
    for spec in base_specs:
        print(
            f"  [{spec['company_id']:02d}] {spec['company_name']}: "
            f"HK={spec['hk_path']} | US={spec['us_path']}"
        )
    print()
    print("Held-out companies used only for shared-head test:")
    for spec in held_out_specs:
        print(
            f"  [{spec['company_id']:02d}] {spec['company_name']}: "
            f"HK={spec['hk_path']} | US={spec['us_path']}"
        )
    print()

    results = []
    for exp_name, model_cls in build_experiments():
        print("=" * 100)
        print(f"Running experiment: {exp_name}")

        if exp_name == "cross_market_shared_head":
            model_config = replace(deepcopy(MODEL_CONFIG), num_companies=1)
            train_loader = shared_train_loader
            val_loader = shared_val_loader
            test_loader = shared_test_loader
        else:
            model_config = replace(deepcopy(MODEL_CONFIG), num_companies=len(base_specs))
            train_loader = standard_train_loader
            val_loader = standard_val_loader
            test_loader = standard_test_loader
        train_config = deepcopy(TRAIN_CONFIG)
        train_config.checkpoint_name = f"{exp_name}.pt"
        train_config.history_plot_name = f"{exp_name}_history.png"

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
