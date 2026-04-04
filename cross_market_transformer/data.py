from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset

XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
XLSX_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
OFFICE_DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


@dataclass
class SampleBatch:
    x_hk: torch.Tensor
    x_us: torch.Tensor
    hk_time_delta: torch.Tensor
    us_time_delta: torch.Tensor
    hk_padding_mask: torch.Tensor
    us_padding_mask: torch.Tensor
    company_id: torch.Tensor
    us_open_prev_night: torch.Tensor
    us_sessions_since_last_hk: torch.Tensor
    latest_us_gap_days: torch.Tensor
    target: torch.Tensor

    def to(self, device: torch.device | str) -> "SampleBatch":
        return SampleBatch(
            x_hk=self.x_hk.to(device),
            x_us=self.x_us.to(device),
            hk_time_delta=self.hk_time_delta.to(device),
            us_time_delta=self.us_time_delta.to(device),
            hk_padding_mask=self.hk_padding_mask.to(device),
            us_padding_mask=self.us_padding_mask.to(device),
            company_id=self.company_id.to(device),
            us_open_prev_night=self.us_open_prev_night.to(device),
            us_sessions_since_last_hk=self.us_sessions_since_last_hk.to(device),
            latest_us_gap_days=self.latest_us_gap_days.to(device),
            target=self.target.to(device),
        )


class CrossMarketDataset(Dataset):
    """Dataset for chronologically aligned cross-market samples."""

    def __init__(
        self,
        x_hk: np.ndarray | torch.Tensor,
        x_us: np.ndarray | torch.Tensor,
        hk_time_delta: np.ndarray | torch.Tensor,
        us_time_delta: np.ndarray | torch.Tensor,
        company_id: np.ndarray | torch.Tensor,
        us_open_prev_night: np.ndarray | torch.Tensor,
        us_sessions_since_last_hk: np.ndarray | torch.Tensor,
        latest_us_gap_days: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
        hk_padding_mask: np.ndarray | torch.Tensor | None = None,
        us_padding_mask: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        self.x_hk = self._to_tensor(x_hk, dtype=torch.float32)
        self.x_us = self._to_tensor(x_us, dtype=torch.float32)
        self.hk_time_delta = self._to_tensor(hk_time_delta, dtype=torch.float32)
        self.us_time_delta = self._to_tensor(us_time_delta, dtype=torch.float32)
        self.company_id = self._to_tensor(company_id, dtype=torch.long)
        self.us_open_prev_night = self._to_tensor(us_open_prev_night, dtype=torch.long)
        self.us_sessions_since_last_hk = self._to_tensor(us_sessions_since_last_hk, dtype=torch.float32)
        self.latest_us_gap_days = self._to_tensor(latest_us_gap_days, dtype=torch.float32)
        self.target = self._to_tensor(target)

        num_samples = self.x_hk.shape[0]
        if hk_padding_mask is None:
            hk_padding_mask = torch.zeros(num_samples, self.x_hk.shape[1], dtype=torch.bool)
        if us_padding_mask is None:
            us_padding_mask = torch.zeros(num_samples, self.x_us.shape[1], dtype=torch.bool)

        self.hk_padding_mask = self._to_tensor(hk_padding_mask, dtype=torch.bool)
        self.us_padding_mask = self._to_tensor(us_padding_mask, dtype=torch.bool)

        self._validate_shapes()

    def _validate_shapes(self) -> None:
        n = self.x_hk.shape[0]
        if self.x_us.shape[0] != n:
            raise ValueError("x_hk and x_us must have the same number of samples.")
        if self.hk_time_delta.shape[:2] != self.x_hk.shape[:2]:
            raise ValueError("hk_time_delta must match x_hk sequence dimensions.")
        if self.us_time_delta.shape[:2] != self.x_us.shape[:2]:
            raise ValueError("us_time_delta must match x_us sequence dimensions.")
        if self.company_id.shape[0] != n or self.us_open_prev_night.shape[0] != n:
            raise ValueError("company_id and us_open_prev_night must match sample count.")
        if self.us_sessions_since_last_hk.shape[0] != n or self.latest_us_gap_days.shape[0] != n:
            raise ValueError("Global timing features must match sample count.")
        if self.target.shape[0] != n:
            raise ValueError("target must match sample count.")
        if self.hk_padding_mask.shape[:2] != self.x_hk.shape[:2]:
            raise ValueError("hk_padding_mask must match x_hk sequence dimensions.")
        if self.us_padding_mask.shape[:2] != self.x_us.shape[:2]:
            raise ValueError("us_padding_mask must match x_us sequence dimensions.")

    @staticmethod
    def _to_tensor(
        value: np.ndarray | torch.Tensor,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype) if dtype is not None else value
        tensor = torch.from_numpy(np.asarray(value))
        return tensor.to(dtype=dtype) if dtype is not None else tensor

    def __len__(self) -> int:
        return self.x_hk.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x_hk": self.x_hk[index],
            "x_us": self.x_us[index],
            "hk_time_delta": self.hk_time_delta[index],
            "us_time_delta": self.us_time_delta[index],
            "hk_padding_mask": self.hk_padding_mask[index],
            "us_padding_mask": self.us_padding_mask[index],
            "company_id": self.company_id[index],
            "us_open_prev_night": self.us_open_prev_night[index],
            "us_sessions_since_last_hk": self.us_sessions_since_last_hk[index],
            "latest_us_gap_days": self.latest_us_gap_days[index],
            "target": self.target[index],
        }


def numpy_collate_fn(batch: Sequence[dict[str, torch.Tensor]]) -> SampleBatch:
    x_hk = torch.stack([item["x_hk"] for item in batch], dim=0)
    x_us = torch.stack([item["x_us"] for item in batch], dim=0)
    hk_time_delta = torch.stack([item["hk_time_delta"] for item in batch], dim=0)
    us_time_delta = torch.stack([item["us_time_delta"] for item in batch], dim=0)
    hk_padding_mask = torch.stack([item["hk_padding_mask"] for item in batch], dim=0)
    us_padding_mask = torch.stack([item["us_padding_mask"] for item in batch], dim=0)
    company_id = torch.stack([item["company_id"] for item in batch], dim=0)
    us_open_prev_night = torch.stack([item["us_open_prev_night"] for item in batch], dim=0)
    us_sessions_since_last_hk = torch.stack([item["us_sessions_since_last_hk"] for item in batch], dim=0)
    latest_us_gap_days = torch.stack([item["latest_us_gap_days"] for item in batch], dim=0)
    target = torch.stack([item["target"] for item in batch], dim=0)
    return SampleBatch(
        x_hk=x_hk,
        x_us=x_us,
        hk_time_delta=hk_time_delta,
        us_time_delta=us_time_delta,
        hk_padding_mask=hk_padding_mask,
        us_padding_mask=us_padding_mask,
        company_id=company_id,
        us_open_prev_night=us_open_prev_night,
        us_sessions_since_last_hk=us_sessions_since_last_hk,
        latest_us_gap_days=latest_us_gap_days,
        target=target,
    )


def chronological_split(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]:
    total = len(dataset)
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0.")
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    indices = np.arange(total)
    train_subset = torch.utils.data.Subset(dataset, indices[:train_end].tolist())
    val_subset = torch.utils.data.Subset(dataset, indices[train_end:val_end].tolist())
    test_subset = torch.utils.data.Subset(dataset, indices[val_end:].tolist())
    return train_subset, val_subset, test_subset


def load_factor_xlsx(
    path: str | Path,
    date_col: str = "Date",
) -> dict[str, np.ndarray | list[str]]:
    """
    Read the standardized factor Excel file without pandas/openpyxl.

    Returns:
        {
            "dates": np.ndarray[datetime64[D]],
            "features": np.ndarray[float32],   # [num_rows, num_features]
            "feature_names": list[str],
        }
    """
    path = Path(path)
    rows = _read_first_sheet_rows(path)
    if not rows:
        raise ValueError(f"No rows found in Excel file: {path}")

    header = rows[0]
    if date_col not in header:
        raise ValueError(f"Missing date column '{date_col}' in {path}.")

    date_idx = header.index(date_col)
    feature_names = [name for idx, name in enumerate(header) if idx != date_idx]

    dates = []
    features = []
    for raw_row in rows[1:]:
        row = raw_row + [""] * (len(header) - len(raw_row))
        dates.append(_excel_serial_to_date(row[date_idx]))
        feature_row = [float(row[idx]) for idx in range(len(header)) if idx != date_idx]
        features.append(feature_row)

    return {
        "dates": np.asarray(dates, dtype="datetime64[D]"),
        "features": np.asarray(features, dtype=np.float32),
        "feature_names": feature_names,
    }


def build_samples_from_excel_pair(
    hk_path: str | Path,
    us_path: str | Path,
    hk_lookback: int,
    us_lookback: int,
    company_id: int = 0,
    task_type: str = "binary_classification",
    target_col: str = "r1",
    multiclass_num_classes: int = 3,
    multiclass_thresholds: Sequence[float] | None = None,
    use_us_prev_night: bool = True,
) -> dict[str, np.ndarray]:
    """
    Build aligned samples for one HK/US listed company pair.

    For HK trading day t:
    - x_hk uses HK rows ending at t-1
    - if use_us_prev_night is True:
      x_us uses the latest completed US row with us_date < hk_date
    - if use_us_prev_night is False:
      x_us ends at the US row immediately before the latest completed US row
    - target uses the HK row at date t
    """
    hk_data = load_factor_xlsx(hk_path)
    us_data = load_factor_xlsx(us_path)

    hk_feature_names = hk_data["feature_names"]
    us_feature_names = us_data["feature_names"]
    if hk_feature_names != us_feature_names:
        raise ValueError("HK and US files must have the same feature columns in the same order.")

    if target_col not in hk_feature_names:
        raise ValueError(f"target_col '{target_col}' not found in HK features.")
    target_idx = hk_feature_names.index(target_col)

    hk_dates = hk_data["dates"]
    hk_features = hk_data["features"]
    us_dates = us_data["dates"]
    us_features = us_data["features"]

    x_hk_list = []
    x_us_list = []
    hk_time_delta_list = []
    us_time_delta_list = []
    company_ids = []
    session_flags = []
    us_sessions_since_last_hk_list = []
    latest_us_gap_days_list = []
    targets = []
    sample_dates = []

    for hk_idx in range(hk_lookback, len(hk_dates)):
        hk_date = hk_dates[hk_idx]
        us_latest_idx = _find_latest_us_index_before_date(us_dates, hk_date)
        if us_latest_idx is None:
            continue

        effective_us_latest_idx = us_latest_idx if use_us_prev_night else us_latest_idx - 1
        if effective_us_latest_idx < 0 or effective_us_latest_idx + 1 < us_lookback:
            continue

        hk_window = hk_features[hk_idx - hk_lookback : hk_idx]
        hk_window_dates = hk_dates[hk_idx - hk_lookback : hk_idx]
        us_window = us_features[effective_us_latest_idx - us_lookback + 1 : effective_us_latest_idx + 1]
        us_window_dates = us_dates[effective_us_latest_idx - us_lookback + 1 : effective_us_latest_idx + 1]
        target_value = float(hk_features[hk_idx, target_idx])
        last_hk_obs_date = hk_window_dates[-1]
        hk_time_delta = _compute_time_deltas(hk_date, hk_window_dates)
        us_time_delta = _compute_time_deltas(hk_date, us_window_dates)
        us_sessions_since_last_hk = _count_us_sessions_between(
            us_dates,
            last_hk_obs_date,
            hk_date,
            include_latest=use_us_prev_night,
        )
        latest_us_gap_days = int((hk_date - us_dates[effective_us_latest_idx]).astype("timedelta64[D]").astype(int))

        x_hk_list.append(hk_window.astype(np.float32))
        x_us_list.append(us_window.astype(np.float32))
        hk_time_delta_list.append(hk_time_delta.astype(np.float32))
        us_time_delta_list.append(us_time_delta.astype(np.float32))
        company_ids.append(company_id)
        session_flags.append(_was_us_open_prev_night(hk_date, us_dates) if use_us_prev_night else 0)
        us_sessions_since_last_hk_list.append(float(us_sessions_since_last_hk))
        latest_us_gap_days_list.append(float(latest_us_gap_days))
        targets.append(
            _build_target(
                target_value=target_value,
                task_type=task_type,
                multiclass_num_classes=multiclass_num_classes,
                multiclass_thresholds=multiclass_thresholds,
            )
        )
        sample_dates.append(hk_date)

    if not x_hk_list:
        raise ValueError("No valid aligned samples were created. Check lookback lengths and date overlap.")

    target_dtype = np.float32 if task_type in {"regression", "binary_classification"} else np.int64
    return {
        "x_hk": np.asarray(x_hk_list, dtype=np.float32),
        "x_us": np.asarray(x_us_list, dtype=np.float32),
        "hk_time_delta": np.asarray(hk_time_delta_list, dtype=np.float32),
        "us_time_delta": np.asarray(us_time_delta_list, dtype=np.float32),
        "company_id": np.asarray(company_ids, dtype=np.int64),
        "us_open_prev_night": np.asarray(session_flags, dtype=np.int64),
        "us_sessions_since_last_hk": np.asarray(us_sessions_since_last_hk_list, dtype=np.float32),
        "latest_us_gap_days": np.asarray(latest_us_gap_days_list, dtype=np.float32),
        "target": np.asarray(targets, dtype=target_dtype),
        "sample_dates": np.asarray(sample_dates, dtype="datetime64[D]"),
        "feature_names": hk_feature_names,
    }


def build_multi_company_dataset(
    company_specs: Sequence[dict],
    hk_lookback: int,
    us_lookback: int,
    task_type: str = "binary_classification",
    target_col: str = "r1",
    multiclass_num_classes: int = 3,
    multiclass_thresholds: Sequence[float] | None = None,
    use_us_prev_night: bool = True,
) -> CrossMarketDataset:
    x_hk_parts = []
    x_us_parts = []
    company_parts = []
    session_parts = []
    us_sessions_parts = []
    latest_us_gap_parts = []
    target_parts = []
    hk_time_delta_parts = []
    us_time_delta_parts = []

    for spec in company_specs:
        samples = build_samples_from_excel_pair(
            hk_path=spec["hk_path"],
            us_path=spec["us_path"],
            hk_lookback=hk_lookback,
            us_lookback=us_lookback,
            company_id=spec["company_id"],
            task_type=task_type,
            target_col=target_col,
            multiclass_num_classes=multiclass_num_classes,
            multiclass_thresholds=multiclass_thresholds,
            use_us_prev_night=use_us_prev_night,
        )
        x_hk_parts.append(samples["x_hk"])
        x_us_parts.append(samples["x_us"])
        hk_time_delta_parts.append(samples["hk_time_delta"])
        us_time_delta_parts.append(samples["us_time_delta"])
        company_parts.append(samples["company_id"])
        session_parts.append(samples["us_open_prev_night"])
        us_sessions_parts.append(samples["us_sessions_since_last_hk"])
        latest_us_gap_parts.append(samples["latest_us_gap_days"])
        target_parts.append(samples["target"])

    x_hk = np.concatenate(x_hk_parts, axis=0)
    x_us = np.concatenate(x_us_parts, axis=0)
    hk_time_delta = np.concatenate(hk_time_delta_parts, axis=0)
    us_time_delta = np.concatenate(us_time_delta_parts, axis=0)
    company_id = np.concatenate(company_parts, axis=0)
    us_open_prev_night = np.concatenate(session_parts, axis=0)
    us_sessions_since_last_hk = np.concatenate(us_sessions_parts, axis=0)
    latest_us_gap_days = np.concatenate(latest_us_gap_parts, axis=0)
    target = np.concatenate(target_parts, axis=0)

    return CrossMarketDataset(
        x_hk=x_hk,
        x_us=x_us,
        hk_time_delta=hk_time_delta,
        us_time_delta=us_time_delta,
        company_id=company_id,
        us_open_prev_night=us_open_prev_night,
        us_sessions_since_last_hk=us_sessions_since_last_hk,
        latest_us_gap_days=latest_us_gap_days,
        target=target,
    )


def _build_target(
    target_value: float,
    task_type: str,
    multiclass_num_classes: int,
    multiclass_thresholds: Sequence[float] | None,
):
    if task_type == "regression":
        return target_value
    if task_type == "binary_classification":
        return 1.0 if target_value > 0.0 else 0.0
    if task_type == "multiclass_classification":
        thresholds = list(multiclass_thresholds) if multiclass_thresholds is not None else _default_multiclass_thresholds(multiclass_num_classes)
        return int(np.digitize(target_value, thresholds))
    raise ValueError(f"Unsupported task_type: {task_type}")


def _default_multiclass_thresholds(num_classes: int) -> list[float]:
    if num_classes == 3:
        return [-0.5, 0.5]
    raise ValueError(
        "Provide multiclass_thresholds explicitly when num_classes is not 3."
    )


def _find_latest_us_index_before_date(us_dates: np.ndarray, hk_date: np.datetime64) -> int | None:
    idx = np.searchsorted(us_dates, hk_date, side="left") - 1
    if idx < 0:
        return None
    return int(idx)


def _was_us_open_prev_night(hk_date: np.datetime64, us_dates: np.ndarray) -> int:
    prev_day = hk_date.astype("datetime64[D]") - np.timedelta64(1, "D")
    idx = np.searchsorted(us_dates, prev_day, side="left")
    return int(idx < len(us_dates) and us_dates[idx] == prev_day)


def _compute_time_deltas(target_date: np.datetime64, window_dates: np.ndarray) -> np.ndarray:
    return (target_date - window_dates).astype("timedelta64[D]").astype(np.int64)


def _count_us_sessions_between(
    us_dates: np.ndarray,
    last_hk_obs_date: np.datetime64,
    hk_date: np.datetime64,
    include_latest: bool = True,
) -> int:
    left = np.searchsorted(us_dates, last_hk_obs_date, side="right")
    right = np.searchsorted(us_dates, hk_date, side="left")
    count = max(0, int(right - left))
    if not include_latest and count > 0:
        count -= 1
    return count


def _excel_serial_to_date(value: str) -> datetime:
    base = datetime(1899, 12, 30)
    serial = float(value)
    return base + timedelta(days=serial)


def _read_first_sheet_rows(path: Path) -> list[list[str]]:
    with ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
        rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: _normalize_target(rel.attrib["Target"]) for rel in rel_root.findall(f"{XLSX_REL_NS}Relationship")}

        sheets = workbook_root.find(f"{XLSX_NS}sheets")
        first_sheet = sheets.findall(f"{XLSX_NS}sheet")[0]
        rel_id = first_sheet.attrib[f"{{{OFFICE_DOC_REL_NS}}}id"]
        sheet_target = rel_map[rel_id]
        sheet_root = ET.fromstring(zf.read(sheet_target))
        sheet_data = sheet_root.find(f"{XLSX_NS}sheetData")

        rows = []
        for row in sheet_data.findall(f"{XLSX_NS}row"):
            values = {}
            for cell in row.findall(f"{XLSX_NS}c"):
                ref = cell.attrib.get("r", "A1")
                col_idx = _col_to_idx(re.match(r"[A-Z]+", ref).group(0))
                values[col_idx] = _parse_cell_value(cell, shared_strings)
            if values:
                max_idx = max(values)
                rows.append([values.get(i, "") for i in range(max_idx + 1)])
        return rows


def _read_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values = []
    for si in root.findall(f"{XLSX_NS}si"):
        texts = [t.text or "" for t in si.iter(f"{XLSX_NS}t")]
        values.append("".join(texts))
    return values


def _parse_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find(f"{XLSX_NS}v")
    if value_node is not None:
        if cell_type == "s":
            return shared_strings[int(value_node.text)]
        return value_node.text or ""
    inline_node = cell.find(f"{XLSX_NS}is")
    if inline_node is not None:
        return "".join(text.text or "" for text in inline_node.iter(f"{XLSX_NS}t"))
    return ""


def _normalize_target(target: str) -> str:
    target = target.lstrip("/")
    return target if target.startswith("xl/") else f"xl/{target}"


def _col_to_idx(col: str) -> int:
    idx = 0
    for char in col:
        idx = idx * 26 + (ord(char.upper()) - 64)
    return idx - 1
