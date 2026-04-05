from __future__ import annotations

from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from cross_market_transformer import (
    CrossMarketTransformerModel,
    Trainer,
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


def main() -> None:
    company_specs = discover_standardized_pairs(DATASET_ROOT)
    print("Loaded company pairs:")
    for spec in company_specs:
        print(
            f"  [{spec['company_id']:02d}] {spec['company_name']}: "
            f"HK={spec['hk_path']} | US={spec['us_path']}"
        )
    print()

    model_config = replace(MODEL_CONFIG, num_companies=len(company_specs))
    train_config = TRAIN_CONFIG

    train_set, val_set, test_set = build_multi_company_splits(
        company_specs=company_specs,
        hk_lookback=HK_LOOKBACK,
        us_lookback=US_LOOKBACK,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        task_type=model_config.task_type,
        target_col=TARGET_COL,
        multiclass_num_classes=model_config.num_classes,
        use_us_prev_night=USE_US_PREV_NIGHT,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=numpy_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=numpy_collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=numpy_collate_fn,
    )

    model = CrossMarketTransformerModel(model_config)
    trainer = Trainer(
        model=model,
        train_config=train_config,
        task_type=model_config.task_type,
        num_classes=model_config.num_classes,
    )

    fit_result = trainer.fit(train_loader, val_loader, test_loader=test_loader)
    test_metrics = trainer.evaluate(test_loader)

    print(f"Best {fit_result['best_score_name']}:", fit_result["best_score"])
    print("Test metrics:", test_metrics)
    print("Number of company pairs:", len(company_specs))


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
