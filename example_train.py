from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from cross_market_transformer import (
    CrossMarketTransformerSharedHeadModel,
    Trainer,
    build_multi_company_dataset,
    discover_cleaned_pairs,
    numpy_collate_fn,
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


RUN_NAME = "cross_market_shared_head_joint_pindex_feature"
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


def _split_company_specs(company_specs, train_company_count: int):
    if len(company_specs) <= train_company_count:
        raise ValueError(
            f"Need more than {train_company_count} company pairs for held-out-company testing; "
            f"found {len(company_specs)}."
        )
    train_specs = _reindex_company_specs(company_specs[:train_company_count])
    test_specs = _reindex_company_specs(company_specs[train_company_count:])
    return train_specs, test_specs


def _print_company_split(split_name: str, company_specs) -> None:
    print(f"{split_name} companies ({len(company_specs)}):")
    for spec in company_specs:
        print(
            f"  [{spec['company_id']:02d}] {spec['company_name']}: "
            f"HK={spec['hk_path']} | US={spec['us_path']}"
        )
    print()


def _make_joint_dataset(company_specs):
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
        p_index_mode=P_INDEX_MODE,
        p_index_gap_threshold=P_INDEX_GAP_THRESHOLD,
    )


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

    train_specs, test_specs = _split_company_specs(company_specs, train_company_count=TRAIN_COMPANY_COUNT)
    _print_company_split("Train", train_specs)
    _print_company_split("Test", test_specs)

    train_set = _make_joint_dataset(train_specs)
    test_set = _make_joint_dataset(test_specs)

    train_config = make_task_train_config(JOINT_TARGET_TASK_TYPE)
    train_config.checkpoint_name = f"{RUN_NAME}.pt"
    train_config.history_plot_name = f"{RUN_NAME}_history.png"
    train_config.threshold_sweep_name = f"{RUN_NAME}_threshold_sweep.csv"
    train_config.threshold_sweep_plot_name = f"{RUN_NAME}_threshold_sweep.png"

    train_loader = _make_loader(train_set, train_config)
    test_loader = _make_loader(test_set, train_config)

    model_config = make_task_model_config(
        JOINT_TARGET_TASK_TYPE,
        num_classes=3,
        num_companies=len(train_specs),
        hk_input_dim=train_set.x_hk.shape[-1],
        us_input_dim=train_set.x_us.shape[-1],
        p_index_gap_feature_dim=train_set.p_index_gap_features.shape[-1],
    )

    model = CrossMarketTransformerSharedHeadModel(model_config)
    trainer = Trainer(
        model=model,
        train_config=train_config,
        task_type=model_config.task_type,
        num_classes=model_config.num_classes,
    )

    print("=" * 100)
    print("Running main shared-head joint model")
    fit_result = trainer.fit(train_loader, val_loader=None, test_loader=test_loader)
    test_metrics = trainer.evaluate(test_loader)

    print("=" * 100)
    print("Main model summary")
    print(f"{RUN_NAME} | {fit_result['best_score_name']}={fit_result['best_score']:.6f} | test={test_metrics}")
    print("Train company pairs:", len(train_specs))
    print("Test company pairs:", len(test_specs))


if __name__ == "__main__":
    main()
