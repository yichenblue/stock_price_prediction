from __future__ import annotations

import math

import numpy as np

from cross_market_transformer import build_samples_from_excel_pair
from minimal_config import (
    HK_EXCEL_PATH,
    HK_LOOKBACK,
    MODEL_CONFIG,
    TARGET_COL,
    USE_US_PREV_NIGHT,
    US_EXCEL_PATH,
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
    samples = build_samples_from_excel_pair(
        hk_path=HK_EXCEL_PATH,
        us_path=US_EXCEL_PATH,
        hk_lookback=HK_LOOKBACK,
        us_lookback=US_LOOKBACK,
        company_id=0,
        task_type="regression",
        target_col=TARGET_COL,
        multiclass_num_classes=MODEL_CONFIG.num_classes,
        use_us_prev_night=USE_US_PREV_NIGHT,
    )

    targets = np.asarray(samples["target"], dtype=np.float64)
    dates = np.asarray(samples["sample_dates"])
    total = len(targets)

    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.15)

    train_targets = targets[:train_end]
    val_targets = targets[train_end:val_end]
    test_targets = targets[val_end:]

    print("Split date ranges")
    print(f"  train: {dates[0]} -> {dates[train_end - 1]}")
    print(f"  val  : {dates[train_end]} -> {dates[val_end - 1]}")
    print(f"  test : {dates[val_end]} -> {dates[-1]}")
    print()

    summarize_split("train", train_targets)
    summarize_split("val", val_targets)
    summarize_split("test", test_targets)


if __name__ == "__main__":
    main()
