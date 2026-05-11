from __future__ import annotations

import numpy as np

from cross_market_transformer import build_multi_company_dataset, discover_cleaned_pairs
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
)


TRAIN_COMPANY_COUNT = 25


def reindex_company_specs(company_specs):
    reindexed_specs = []
    for new_company_id, spec in enumerate(company_specs):
        updated_spec = dict(spec)
        updated_spec["company_id"] = new_company_id
        reindexed_specs.append(updated_spec)
    return reindexed_specs


def split_company_specs(company_specs, train_company_count: int = TRAIN_COMPANY_COUNT):
    if len(company_specs) <= train_company_count:
        raise ValueError(
            f"Need more than {train_company_count} company pairs for held-out-company testing; "
            f"found {len(company_specs)}."
        )
    train_specs = reindex_company_specs(company_specs[:train_company_count])
    test_specs = reindex_company_specs(company_specs[train_company_count:])
    return train_specs, test_specs


def build_regression_dataset(company_specs):
    return build_multi_company_dataset(
        company_specs=company_specs,
        hk_lookback=HK_LOOKBACK,
        us_lookback=US_LOOKBACK,
        task_type="regression",
        target_col=TARGET_COL,
        multiclass_num_classes=MODEL_CONFIG.num_classes,
        use_us_prev_night=USE_US_PREV_NIGHT,
        normalization_mode=NORMALIZATION_MODE,
        rolling_normalization_window=ROLLING_NORMALIZATION_WINDOW,
        p_index_mode=P_INDEX_MODE,
        p_index_gap_threshold=P_INDEX_GAP_THRESHOLD,
    )


def summarize_split(name: str, values: np.ndarray) -> None:
    mean = float(np.mean(values))
    std = float(np.std(values))
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    positive_ratio = float(np.mean(values > 0.0))
    negative_ratio = float(np.mean(values < 0.0))
    zero_ratio = float(np.mean(values == 0.0))

    abs_values = np.abs(values)
    q50 = float(np.quantile(abs_values, 0.50))
    q75 = float(np.quantile(abs_values, 0.75))
    q90 = float(np.quantile(abs_values, 0.90))
    q95 = float(np.quantile(abs_values, 0.95))
    one_std_tail = float(np.mean(abs_values > std)) if std > 0 else 0.0
    two_std_tail = float(np.mean(abs_values > 2.0 * std)) if std > 0 else 0.0

    print(f"[{name}]")
    print(f"  count            : {len(values)}")
    print(f"  mean             : {mean:.6f}")
    print(f"  std              : {std:.6f}")
    print(f"  min              : {min_value:.6f}")
    print(f"  max              : {max_value:.6f}")
    print(f"  positive_ratio   : {positive_ratio:.4f}")
    print(f"  negative_ratio   : {negative_ratio:.4f}")
    print(f"  zero_ratio       : {zero_ratio:.4f}")
    print(f"  |r1| q50         : {q50:.6f}")
    print(f"  |r1| q75         : {q75:.6f}")
    print(f"  |r1| q90         : {q90:.6f}")
    print(f"  |r1| q95         : {q95:.6f}")
    print(f"  |r1| > 1 std     : {one_std_tail:.4f}")
    print(f"  |r1| > 2 std     : {two_std_tail:.4f}")
    print()


def main() -> None:
    company_specs = discover_cleaned_pairs(DATASET_ROOT)
    train_specs, test_specs = split_company_specs(company_specs)
    train_set = build_regression_dataset(train_specs)
    test_set = build_regression_dataset(test_specs)
    train_targets = train_set.target.numpy().astype(np.float64)
    test_targets = test_set.target.numpy().astype(np.float64)

    print(f"Number of company pairs: {len(company_specs)}")
    print(f"Train company pairs: {len(train_specs)}")
    print(f"Test company pairs : {len(test_specs)}")
    print(f"Train samples: {len(train_set)}")
    print(f"Test samples : {len(test_set)}")
    print()

    summarize_split("train", train_targets)
    summarize_split("test", test_targets)


if __name__ == "__main__":
    main()
