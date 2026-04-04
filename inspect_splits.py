from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from cross_market_transformer import build_multi_company_splits, discover_standardized_pairs
from minimal_config import (
    DATASET_ROOT,
    HK_LOOKBACK,
    MODEL_CONFIG,
    TARGET_COL,
    USE_US_PREV_NIGHT,
    US_LOOKBACK,
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
    company_specs = discover_standardized_pairs(DATASET_ROOT)
    train_set, val_set, test_set = build_multi_company_splits(
        company_specs=company_specs,
        hk_lookback=HK_LOOKBACK,
        us_lookback=US_LOOKBACK,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        task_type="regression",
        target_col=TARGET_COL,
        multiclass_num_classes=MODEL_CONFIG.num_classes,
        use_us_prev_night=USE_US_PREV_NIGHT,
    )
    train_targets = train_set.target.numpy().astype(np.float64)
    val_targets = val_set.target.numpy().astype(np.float64)
    test_targets = test_set.target.numpy().astype(np.float64)

    print(f"Number of company pairs: {len(company_specs)}")
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples  : {len(val_set)}")
    print(f"Test samples : {len(test_set)}")
    print()

    summarize_split("train", train_targets)
    summarize_split("val", val_targets)
    summarize_split("test", test_targets)


if __name__ == "__main__":
    main()
