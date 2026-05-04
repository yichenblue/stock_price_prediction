from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from cross_market_transformer import (
    CrossMarketDataset,
    CrossMarketTransformerSharedHeadModel,
    Trainer,
    build_multi_company_dataset,
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


def _reindex_company_specs(company_specs):
    remapped = []
    for new_id, spec in enumerate(company_specs):
        remapped.append({**spec, "company_id": new_id})
    return remapped


def split_company_specs(company_specs):
    base_specs = [spec for spec in company_specs if spec["company_name"] not in HELD_OUT_SHARED_HEAD_COMPANIES]
    held_out_specs = [spec for spec in company_specs if spec["company_name"] in HELD_OUT_SHARED_HEAD_COMPANIES]
    return _reindex_company_specs(base_specs), _reindex_company_specs(held_out_specs)


def make_dataloaders(base_specs, held_out_specs):
    train_set = build_multi_company_dataset(
        company_specs=base_specs,
        hk_lookback=HK_LOOKBACK,
        us_lookback=US_LOOKBACK,
        task_type=MODEL_CONFIG.task_type,
        target_col=TARGET_COL,
        multiclass_num_classes=MODEL_CONFIG.num_classes,
        use_us_prev_night=USE_US_PREV_NIGHT,
        normalization_mode=NORMALIZATION_MODE,
        rolling_normalization_window=ROLLING_NORMALIZATION_WINDOW,
    )
    test_set = build_multi_company_dataset(
        company_specs=held_out_specs,
        hk_lookback=HK_LOOKBACK,
        us_lookback=US_LOOKBACK,
        task_type=MODEL_CONFIG.task_type,
        target_col=TARGET_COL,
        multiclass_num_classes=MODEL_CONFIG.num_classes,
        use_us_prev_night=USE_US_PREV_NIGHT,
        normalization_mode=NORMALIZATION_MODE,
        rolling_normalization_window=ROLLING_NORMALIZATION_WINDOW,
    )
    return (
        _make_loader(_with_shared_company_id(train_set)),
        _make_loader(_with_shared_company_id(test_set)),
    )


def main() -> None:
    torch.manual_seed(42)
    company_specs = discover_cleaned_pairs(DATASET_ROOT)
    base_specs, held_out_specs = split_company_specs(company_specs)

    if len(held_out_specs) != len(HELD_OUT_SHARED_HEAD_COMPANIES):
        raise ValueError(
            "Held-out shared-head companies were not found exactly as expected: "
            f"{sorted(HELD_OUT_SHARED_HEAD_COMPANIES)}"
        )

    train_loader, test_loader = make_dataloaders(base_specs, held_out_specs)

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

    model_config = replace(deepcopy(MODEL_CONFIG), num_companies=1)
    train_config = deepcopy(TRAIN_CONFIG)
    train_config.checkpoint_name = "cross_market_shared_head.pt"
    train_config.history_plot_name = "cross_market_shared_head_history.png"
    train_config.threshold_sweep_name = "cross_market_shared_head_threshold_sweep.csv"
    train_config.threshold_sweep_plot_name = "cross_market_shared_head_threshold_sweep.png"

    model = CrossMarketTransformerSharedHeadModel(model_config)
    trainer = Trainer(
        model=model,
        train_config=train_config,
        task_type=model_config.task_type,
        num_classes=model_config.num_classes,
    )

    fit_result = trainer.fit(train_loader, val_loader=None, test_loader=test_loader)
    test_metrics = trainer.evaluate(test_loader)

    print(f"Best {fit_result['best_score_name']}: {fit_result['best_score']:.6f}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
