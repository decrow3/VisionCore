"""Shared helpers for the non-circular FEM information analyses.

The analyses in ``Non_circular_FEM_information_tests_prescription.md`` should
stay scientifically separate, but they all need the same production-run
provenance, paired-condition bookkeeping, and small statistical utilities.
This module keeps that plumbing in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TWININFO_RUN_DIR = ROOT / "outputs" / "twininfo" / "active-sensing-all-images-1crop-2fix2ms-16units-gpu"
DEFAULT_STACK_OUT_DIR = ROOT / "outputs" / "active_sensing_movie_information"
PRIMARY_FINAL_METRIC = "final_cumulative_spatial_ssi_bits_per_spike"
PRIMARY_SERIES_METRIC = "cumulative_spatial_ssi_bits_per_spike"


@dataclass(frozen=True)
class SeriesTable:
    """Cumulative-information arrays plus matching per-row metadata."""

    records: list[dict[str, str]]
    arrays: dict[str, np.ndarray]


def canonical_condition(condition: str) -> str:
    """Normalize historical condition aliases."""
    if str(condition) == "phase_order_shuffle":
        return "trajectory_order_shuffle"
    return str(condition)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV as dictionaries, returning an empty list if missing/empty."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV with stable field ordering by first use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    """Convert numpy/path objects to JSON-friendly values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def parse_float(value: Any, default: float = float("nan")) -> float:
    """Parse a float from CSV-ish input."""
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_int(value: Any, default: int = 0) -> int:
    """Parse an int from CSV-ish input."""
    if value is None or value == "":
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def parse_csv_list(text: str | None) -> list[str]:
    if text is None:
        return []
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_float_list(text: str | None) -> list[float]:
    return [float(part) for part in parse_csv_list(text)]


def mean_sem(values: Iterable[float]) -> tuple[float, float, int]:
    """Return finite mean, SEM, and count."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    if arr.size == 1:
        return float(arr[0]), 0.0, 1
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(arr.size)), int(arr.size)


def percentile_ci(values: Iterable[float], lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    """Percentile interval over finite values."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def paired_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    """Key one trace/image/crop sample independent of condition."""
    return (
        str(row.get("example_id", "")),
        str(row.get("kind", "")),
        parse_int(row.get("image_index", 0)),
        parse_int(row.get("crop_rank", 0)),
    )


def paired_condition_table(
    rows: list[dict[str, str]],
    metric: str,
) -> dict[tuple[str, str, int, int], dict[str, float]]:
    """Map sample keys to condition -> metric."""
    table: dict[tuple[str, str, int, int], dict[str, float]] = {}
    for row in rows:
        if metric not in row:
            continue
        table.setdefault(paired_key(row), {})[canonical_condition(str(row.get("condition", "")))] = parse_float(row[metric])
    return table


def paired_contrast_rows(
    rows: list[dict[str, str]],
    *,
    metric: str,
    contrasts: Iterable[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    """Return one paired-delta row per available sample/contrast."""
    table = paired_condition_table(rows, metric)
    out: list[dict[str, Any]] = []
    for key, conds in table.items():
        example_id, kind, image_index, crop_rank = key
        for contrast, condition, baseline in contrasts:
            if condition not in conds or baseline not in conds:
                continue
            value = conds[condition]
            base = conds[baseline]
            out.append(
                {
                    "contrast": contrast,
                    "example_id": example_id,
                    "kind": kind,
                    "image_index": image_index,
                    "crop_rank": crop_rank,
                    "condition": condition,
                    "baseline_condition": baseline,
                    "metric": metric,
                    "condition_value": value,
                    "baseline_value": base,
                    "delta": value - base,
                }
            )
    return out


def summarize_groups(rows: list[dict[str, Any]], group_keys: tuple[str, ...], value_key: str = "delta") -> list[dict[str, Any]]:
    """Mean/SEM summary for grouped numeric rows."""
    groups: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        groups.setdefault(key, []).append(parse_float(row.get(value_key)))
    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        mean, sem, n = mean_sem(groups[key])
        ci_low, ci_high = percentile_ci(groups[key])
        item = {k: v for k, v in zip(group_keys, key, strict=True)}
        item.update({"mean": mean, "sem": sem, "ci_low": ci_low, "ci_high": ci_high, "n": n})
        out.append(item)
    return out


def load_summary_rows(run_dir: Path) -> list[dict[str, str]]:
    return read_csv_rows(Path(run_dir) / "metadata" / "05_lagcube_information_summary.csv")


def load_series(run_dir: Path) -> SeriesTable:
    """Load cumulative-information arrays from a production twininfo run."""
    run_dir = Path(run_dir)
    records = read_csv_rows(run_dir / "metadata" / "05_information_series_records.csv")
    with np.load(run_dir / "cache" / "cumulative_information_series.npz") as npz:
        arrays = {key: np.asarray(npz[key]) for key in npz.files}
    if not records and "record_example_id" in arrays:
        records = records_from_series_arrays(arrays)
    return SeriesTable(records=records, arrays=arrays)


def records_from_series_arrays(arrays: dict[str, np.ndarray]) -> list[dict[str, str]]:
    """Build records from metadata arrays saved in cumulative_information_series.npz."""
    n = int(np.asarray(arrays["record_condition"]).shape[0])
    records: list[dict[str, str]] = []
    for i in range(n):
        records.append(
            {
                "example_id": str(arrays["record_example_id"][i]),
                "kind": str(arrays["record_kind"][i]),
                "condition": canonical_condition(str(arrays["record_condition"][i])),
                "image_index": str(int(arrays["record_image_index"][i])),
                "crop_rank": str(int(arrays["record_crop_rank"][i])),
            }
        )
    return records


def load_run_config(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "metadata" / "run_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_crop_rows(run_dir: Path) -> list[dict[str, str]]:
    return read_csv_rows(Path(run_dir) / "metadata" / "02_image_crop_hotspots.csv")


def load_trace_rows(run_dir: Path) -> list[dict[str, str]]:
    used = read_csv_rows(Path(run_dir) / "metadata" / "01_trace_examples_used.csv")
    return used or read_csv_rows(Path(run_dir) / "metadata" / "01_trace_examples.csv")


def load_selected_traces(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Reconstruct selected eye traces from production metadata.

    This uses the same raw fixRSVP eye-trace source as ``jake.twininfo``.  It is
    intentionally not called by summary-only scripts.
    """
    from jake.twininfo.common import extract_fixrsvp_eye_traces, load_digital_twin

    config = load_run_config(run_dir)
    t_max = parse_int(config.get("t_max", 128), 128)
    model, _model_info, _device = load_digital_twin()
    eye_traces, _durations = extract_fixrsvp_eye_traces(model, min_fix_dur=t_max)
    all_rows = {str(row["example_id"]): row for row in read_csv_rows(Path(run_dir) / "metadata" / "01_trace_examples.csv")}
    out: dict[str, dict[str, Any]] = {}
    for row in load_trace_rows(run_dir):
        full = dict(all_rows.get(str(row["example_id"]), {}))
        full.update(row)
        source_idx = parse_int(full.get("source_trace_index"))
        start = parse_int(full.get("window_start"))
        stop = parse_int(full.get("window_stop"), start + t_max)
        trace = np.asarray(eye_traces[source_idx, start:stop], dtype=np.float32)
        if trace.shape[0] < t_max:
            raise ValueError(f"Trace {full['example_id']} has {trace.shape[0]} samples; expected {t_max}.")
        full["trace"] = trace[:t_max].astype(np.float32)
        out[str(full["example_id"])] = full
    return out


def scale_trace(trace: np.ndarray, scale: float) -> np.ndarray:
    """Mean-center an eye trace and multiply displacement by ``scale``."""
    tr = np.asarray(trace, dtype=np.float32)
    center = np.mean(tr, axis=0, keepdims=True).astype(np.float32)
    return (center + float(scale) * (tr - center)).astype(np.float32)


def stable_trace(trace: np.ndarray) -> np.ndarray:
    tr = np.asarray(trace, dtype=np.float32)
    return np.repeat(np.mean(tr, axis=0, keepdims=True), tr.shape[0], axis=0).astype(np.float32)


def robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Simple endpoint-safe least-squares slope over finite samples."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 2:
        return float("nan")
    x = x - float(np.mean(x))
    denom = float(np.sum(x * x))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * (y - float(np.mean(y)))) / denom)


def time_to_fraction(time_s: np.ndarray, y: np.ndarray, fraction: float) -> float:
    """First time at which a cumulative series reaches fraction of final value."""
    t = np.asarray(time_s, dtype=np.float64)
    arr = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(t) & np.isfinite(arr)
    t = t[keep]
    arr = arr[keep]
    if t.size == 0:
        return float("nan")
    final = float(arr[-1])
    if not np.isfinite(final) or final <= 0:
        return float("nan")
    idx = np.where(arr >= float(fraction) * final)[0]
    if idx.size == 0:
        return float("nan")
    return float(t[int(idx[0])])
