"""Summarize Figure 5 additional-check readiness for a twininfo run.

This utility is intentionally light-weight: it audits condition/pair coverage,
checks the primary Figure 5 metric columns, and writes paired final/time-window
delta summaries for the contrasts named in the Figure 5 prep note.

Example
-------
.venv/bin/python declan/active_sensing_movie_information/summarize_figure5_additional_checks.py \
  --run-dir outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RUN_DIR = Path("outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu")
DEFAULT_RETINAL_QC_PATH = Path(
    "outputs/active_sensing_movie_information/active_sensing_movie_information_figure/retinal_movie_transform_qc.csv"
)
PRIMARY_METRIC = "final_cumulative_spatial_ssi_bits_per_spike"
SERIES_METRIC = "cumulative_spatial_ssi_bits_per_spike"
FINAL_METRICS = (
    PRIMARY_METRIC,
    "final_cumulative_spatial_ssi_bits",
    "final_cumulative_spatial_ssi_bits_per_second",
    "final_cumulative_expected_spikes",
    "final_cumulative_fisher_pattern",
    "final_cumulative_fisher_pattern_per_spike",
)
COMPANION_METRICS = tuple(metric for metric in FINAL_METRICS if metric != PRIMARY_METRIC)
TRAJECTORY_QC_METRICS = (
    "path_length_relative_error",
    "rms_relative_error",
    "step_rms_relative_error",
    "step_cov_relative_error",
    "pos_cov_relative_error",
    "mean_position_error_deg",
    "control_path_length_deg",
    "real_path_length_deg",
    "control_rms_displacement_deg",
    "real_rms_displacement_deg",
    "control_step_rms_deg",
    "real_step_rms_deg",
)
IMAGE_CONTROL_QC_METRICS = (
    "band_energy_fraction",
    "raw_std",
    "clipped_std",
    "original_roi_std",
    "clipping_fraction",
    "complex_coeff_magnitude_relative_error",
    "pyramid_reconstruction_relative_error",
    "outside_roi_changed_fraction",
)
RETINAL_TRANSFORM_METRICS = (
    "gradient_magnitude_mean",
    "gradient_magnitude_p95",
    "motion_power_vs_matched_stabilized_mean",
    "movie_power_mean",
    "temporal_contrast_rms_mean",
    "temporal_contrast_rms_p95",
    "temporal_power_0p5_4hz",
    "temporal_power_4_15hz",
    "temporal_power_15_60hz",
)
TRAJECTORY_CONTROL_CONDITIONS = (
    "random_amp",
    "random_amp_cloud_matched",
    "random_cov",
    "trajectory_order_shuffle",
)
RANDOM_AMP_CLOUD_MATCH_THRESHOLDS = {
    "path_length_relative_error": 0.25,
    "rms_relative_error": 0.20,
    "step_rms_relative_error": 0.30,
    "pos_cov_relative_error": 0.30,
    "mean_position_error_deg": 1e-5,
}

CONTRASTS = (
    ("real_minus_stabilized", "real", "stabilized"),
    ("real_minus_random_amp", "real", "random_amp"),
    ("real_minus_random_amp_cloud_matched", "real", "random_amp_cloud_matched"),
    ("real_minus_random_cov", "real", "random_cov"),
    ("real_minus_trajectory_order_shuffle", "real", "trajectory_order_shuffle"),
    ("sf_low_minus_stabilized_sf_low", "sf_low", "stabilized_sf_low"),
    ("sf_mid_low_minus_stabilized_sf_mid_low", "sf_mid_low", "stabilized_sf_mid_low"),
    ("sf_mid_high_minus_stabilized_sf_mid_high", "sf_mid_high", "stabilized_sf_mid_high"),
    ("sf_high_minus_stabilized_sf_high", "sf_high", "stabilized_sf_high"),
    (
        "pyramid_phase_scrambled_minus_stabilized_pyramid_phase_scrambled",
        "pyramid_phase_scrambled",
        "stabilized_pyramid_phase_scrambled",
    ),
)

REQUIRED_FILES = (
    "metadata/run_config.json",
    "metadata/01_trace_examples_used.csv",
    "metadata/03_trajectory_control_qc.csv",
    "metadata/05_lagcube_information_summary.csv",
    "metadata/05_information_series_records.csv",
    "cache/cumulative_information_series.npz",
)


@dataclass(frozen=True)
class SeriesTable:
    records: list[dict[str, Any]]
    arrays: dict[str, np.ndarray]


def canonical_condition(condition: str) -> str:
    if str(condition) == "phase_order_shuffle":
        return "trajectory_order_shuffle"
    return str(condition)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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


def pair_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["example_id"]),
        str(row.get("kind", "")),
        str(row["image_index"]),
        str(row.get("crop_rank", "0")),
    )


def load_series(run_dir: Path) -> SeriesTable:
    record_path = run_dir / "metadata" / "05_information_series_records.csv"
    records = read_csv_rows(record_path)
    with np.load(run_dir / "cache" / "cumulative_information_series.npz") as npz:
        arrays = {key: np.asarray(npz[key]) for key in npz.files}
    return SeriesTable(records=records, arrays=arrays)


def mean_sem(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def parse_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def pearsonr(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[keep]
    y_arr = y_arr[keep]
    if x_arr.size < 3:
        return float("nan")
    x_arr = x_arr - float(np.mean(x_arr))
    y_arr = y_arr - float(np.mean(y_arr))
    denom = float(np.sqrt(np.sum(x_arr * x_arr) * np.sum(y_arr * y_arr)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x_arr * y_arr) / denom)


def linear_fit(x: list[float], y: list[float]) -> dict[str, float]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[keep]
    y_arr = y_arr[keep]
    if x_arr.size < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    x_mean = float(np.mean(x_arr))
    y_mean = float(np.mean(y_arr))
    x_centered = x_arr - x_mean
    y_centered = y_arr - y_mean
    denom = float(np.sum(x_centered * x_centered))
    if denom <= 0:
        return {"slope": float("nan"), "intercept": y_mean, "r2": float("nan")}
    slope = float(np.sum(x_centered * y_centered) / denom)
    intercept = float(y_mean - slope * x_mean)
    pred = intercept + slope * x_arr
    ss_res = float(np.sum((y_arr - pred) ** 2))
    ss_tot = float(np.sum((y_arr - y_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"slope": slope, "intercept": intercept, "r2": r2}


def clipped_text(value: str, max_chars: int = 160) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def summarize_final_deltas(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    metric_by_key: dict[tuple[str, str], float] = {}
    pair_keys: set[tuple[str, str, str, str]] = set()
    for row in rows:
        condition = canonical_condition(row["condition"])
        key = pair_key(row)
        pair_keys.add(key)
        if PRIMARY_METRIC not in row or row[PRIMARY_METRIC] == "":
            continue
        metric_by_key[(key, condition)] = float(row[PRIMARY_METRIC])

    out: list[dict[str, Any]] = []
    for contrast, lhs, rhs in CONTRASTS:
        deltas = []
        for key in sorted(pair_keys):
            lhs_value = metric_by_key.get((key, lhs))
            rhs_value = metric_by_key.get((key, rhs))
            if lhs_value is not None and rhs_value is not None:
                deltas.append(lhs_value - rhs_value)
        arr = np.asarray(deltas, dtype=np.float64)
        mean, sem = mean_sem(arr)
        out.append(
            {
                "summary_type": "final",
                "contrast": contrast,
                "lhs": lhs,
                "rhs": rhs,
                "window": "final",
                "sample_index": "",
                "time_s": "",
                "n_pairs": int(arr.size),
                "mean_delta": mean,
                "sem_delta": sem,
            }
        )
    return out


def summarize_series_deltas(series: SeriesTable) -> list[dict[str, Any]]:
    y = np.asarray(series.arrays[SERIES_METRIC], dtype=np.float64)
    time_s = np.asarray(series.arrays.get("time_s", np.arange(y.shape[1])), dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"{SERIES_METRIC} must be a 2-D array, got shape {y.shape}")
    if len(series.records) != y.shape[0]:
        raise ValueError(
            f"series record count {len(series.records)} does not match {SERIES_METRIC} rows {y.shape[0]}"
        )

    by_key: dict[tuple[tuple[str, str, str, str], str], int] = {}
    for idx, row in enumerate(series.records):
        by_key[(pair_key(row), canonical_condition(row["condition"]))] = idx

    sample_indices = [
        ("early_25pct", max(int(round((y.shape[1] - 1) * 0.25)), 0)),
        ("mid_50pct", max(int(round((y.shape[1] - 1) * 0.50)), 0)),
        ("final", y.shape[1] - 1),
    ]
    out: list[dict[str, Any]] = []
    pair_keys = sorted({key for key, _condition in by_key})
    for contrast, lhs, rhs in CONTRASTS:
        for window, sample_idx in sample_indices:
            deltas = []
            for key in pair_keys:
                lhs_idx = by_key.get((key, lhs))
                rhs_idx = by_key.get((key, rhs))
                if lhs_idx is None or rhs_idx is None:
                    continue
                deltas.append(float(y[lhs_idx, sample_idx] - y[rhs_idx, sample_idx]))
            arr = np.asarray(deltas, dtype=np.float64)
            mean, sem = mean_sem(arr)
            out.append(
                {
                    "summary_type": "time_series",
                    "contrast": contrast,
                    "lhs": lhs,
                    "rhs": rhs,
                    "window": window,
                    "sample_index": int(sample_idx),
                    "time_s": float(time_s[sample_idx]) if sample_idx < time_s.size else "",
                    "n_pairs": int(arr.size),
                    "mean_delta": mean,
                    "sem_delta": sem,
                }
            )
    return out


def summarize_metric_deltas(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    metric_by_key: dict[tuple[tuple[str, str, str, str], str], dict[str, float]] = {}
    pair_kind: dict[tuple[str, str, str, str], str] = {}
    pair_keys: set[tuple[str, str, str, str]] = set()
    columns = set(rows[0].keys()) if rows else set()
    available_metrics = [metric for metric in FINAL_METRICS if metric in columns]
    for row in rows:
        key = pair_key(row)
        condition = canonical_condition(row["condition"])
        pair_keys.add(key)
        pair_kind[key] = str(row.get("kind", ""))
        metric_by_key[(key, condition)] = {
            metric: parse_float(row.get(metric))
            for metric in available_metrics
        }

    groups = [("all", sorted(pair_keys))]
    for kind in sorted({kind for kind in pair_kind.values() if kind}):
        groups.append((kind, sorted(key for key in pair_keys if pair_kind.get(key) == kind)))

    out: list[dict[str, Any]] = []
    for group, keys in groups:
        for contrast, lhs, rhs in CONTRASTS:
            for metric in available_metrics:
                deltas = []
                lhs_values = []
                rhs_values = []
                for key in keys:
                    lhs_metrics = metric_by_key.get((key, lhs))
                    rhs_metrics = metric_by_key.get((key, rhs))
                    if lhs_metrics is None or rhs_metrics is None:
                        continue
                    lhs_value = lhs_metrics.get(metric, float("nan"))
                    rhs_value = rhs_metrics.get(metric, float("nan"))
                    if np.isfinite(lhs_value) and np.isfinite(rhs_value):
                        lhs_values.append(lhs_value)
                        rhs_values.append(rhs_value)
                        deltas.append(lhs_value - rhs_value)
                delta_arr = np.asarray(deltas, dtype=np.float64)
                lhs_mean, lhs_sem = mean_sem(np.asarray(lhs_values, dtype=np.float64))
                rhs_mean, rhs_sem = mean_sem(np.asarray(rhs_values, dtype=np.float64))
                delta_mean, delta_sem = mean_sem(delta_arr)
                out.append(
                    {
                        "group": group,
                        "contrast": contrast,
                        "lhs": lhs,
                        "rhs": rhs,
                        "metric": metric,
                        "n_pairs": int(delta_arr.size),
                        "lhs_mean": lhs_mean,
                        "lhs_sem": lhs_sem,
                        "rhs_mean": rhs_mean,
                        "rhs_sem": rhs_sem,
                        "mean_delta": delta_mean,
                        "sem_delta": delta_sem,
                    }
                )
    return out


def summarize_trajectory_fairness(qc_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[dict[str, str]]]] = []
    conditions = sorted({canonical_condition(row.get("condition", "")) for row in qc_rows})
    for condition in conditions:
        condition_rows = [row for row in qc_rows if canonical_condition(row.get("condition", "")) == condition]
        if condition_rows:
            groups.append((condition, "all", condition_rows))
            for kind in sorted({str(row.get("kind", "")) for row in condition_rows if row.get("kind", "")}):
                groups.append((condition, kind, [row for row in condition_rows if str(row.get("kind", "")) == kind]))

    out: list[dict[str, Any]] = []
    for condition, group, rows in groups:
        control_descriptions = sorted({str(row.get("control_description", "")) for row in rows})
        fallback_count = sum("fallback" in str(row.get("control_description", "")) for row in rows)
        random_candidate_count = sum(
            "random_directions" in str(row.get("control_description", ""))
            and "fallback" not in str(row.get("control_description", ""))
            for row in rows
        )
        for metric in TRAJECTORY_QC_METRICS:
            values = np.asarray([parse_float(row.get(metric)) for row in rows], dtype=np.float64)
            mean, sem = mean_sem(values)
            finite = values[np.isfinite(values)]
            threshold = RANDOM_AMP_CLOUD_MATCH_THRESHOLDS.get(metric)
            status = ""
            if condition == "random_amp_cloud_matched" and group == "all" and threshold is not None:
                status = "pass" if finite.size and float(np.percentile(finite, 95.0)) <= threshold else "warn"
            out.append(
                {
                    "condition": condition,
                    "group": group,
                    "metric": metric,
                    "status": status,
                    "threshold": threshold if threshold is not None else "",
                    "n": int(finite.size),
                    "mean": mean,
                    "sem": sem,
                    "median": float(np.median(finite)) if finite.size else float("nan"),
                    "p95": float(np.percentile(finite, 95.0)) if finite.size else float("nan"),
                    "fallback_count": int(fallback_count),
                    "random_candidate_count": int(random_candidate_count),
                    "control_description_examples": clipped_text("|".join(control_descriptions[:3])),
                }
            )
    return out


def summarize_image_control_qc(audit_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[dict[str, str]]]] = []
    controls = sorted({str(row.get("control", "")) for row in audit_rows if row.get("control", "")})
    for control in controls:
        control_rows = [row for row in audit_rows if str(row.get("control", "")) == control]
        groups.append((control, "all", control_rows))
        for kind in sorted({str(row.get("kind", "")) for row in control_rows if row.get("kind", "")}):
            groups.append((control, kind, [row for row in control_rows if str(row.get("kind", "")) == kind]))

    out: list[dict[str, Any]] = []
    for control, group, rows in groups:
        for metric in IMAGE_CONTROL_QC_METRICS:
            values = np.asarray([parse_float(row.get(metric)) for row in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            mean, sem = mean_sem(values)
            out.append(
                {
                    "control": control,
                    "group": group,
                    "metric": metric,
                    "n": int(finite.size),
                    "mean": mean,
                    "sem": sem,
                    "median": float(np.median(finite)) if finite.size else float("nan"),
                    "p95": float(np.percentile(finite, 95.0)) if finite.size else float("nan"),
                }
            )
    return out


def summarize_trajectory_qc_gain_correlations(
    summary_rows: list[dict[str, str]],
    qc_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    metric_by_key: dict[tuple[tuple[str, str, str, str], str], float] = {}
    kind_by_key: dict[tuple[str, str, str, str], str] = {}
    for row in summary_rows:
        key = pair_key(row)
        condition = canonical_condition(row["condition"])
        metric_by_key[(key, condition)] = parse_float(row.get(PRIMARY_METRIC))
        kind_by_key[key] = str(row.get("kind", ""))

    joined: list[dict[str, Any]] = []
    for row in qc_rows:
        condition = canonical_condition(row.get("condition", ""))
        if condition not in TRAJECTORY_CONTROL_CONDITIONS:
            continue
        key = pair_key(row)
        condition_value = metric_by_key.get((key, condition), float("nan"))
        real_value = metric_by_key.get((key, "real"), float("nan"))
        stabilized_value = metric_by_key.get((key, "stabilized"), float("nan"))
        if not (np.isfinite(condition_value) and np.isfinite(real_value) and np.isfinite(stabilized_value)):
            continue
        joined.append(
            {
                "condition": condition,
                "kind": kind_by_key.get(key, ""),
                "control_minus_real": condition_value - real_value,
                "control_minus_stabilized": condition_value - stabilized_value,
                **{metric: parse_float(row.get(metric)) for metric in TRAJECTORY_QC_METRICS},
            }
        )

    out: list[dict[str, Any]] = []
    for condition in sorted({row["condition"] for row in joined}):
        condition_rows = [row for row in joined if row["condition"] == condition]
        groups = [("all", condition_rows)]
        for kind in sorted({row["kind"] for row in condition_rows if row["kind"]}):
            groups.append((kind, [row for row in condition_rows if row["kind"] == kind]))
        for group, rows in groups:
            for target in ("control_minus_real", "control_minus_stabilized"):
                y = [float(row[target]) for row in rows]
                target_mean, target_sem = mean_sem(np.asarray(y, dtype=np.float64))
                for metric in TRAJECTORY_QC_METRICS:
                    x = [float(row.get(metric, float("nan"))) for row in rows]
                    out.append(
                        {
                            "condition": condition,
                            "group": group,
                            "target": target,
                            "qc_metric": metric,
                            "n": int(np.sum(np.isfinite(np.asarray(x, dtype=np.float64)) & np.isfinite(np.asarray(y, dtype=np.float64)))),
                            "target_mean": target_mean,
                            "target_sem": target_sem,
                            "pearson_r": pearsonr(x, y),
                        }
                    )
    return out


def summarize_retinal_transform_qc(retinal_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[dict[str, str]]]] = []
    conditions = sorted({canonical_condition(row.get("condition", "")) for row in retinal_rows})
    for condition in conditions:
        condition_rows = [
            row for row in retinal_rows
            if canonical_condition(row.get("condition", "")) == condition
        ]
        groups.append((condition, "all", condition_rows))
        for kind in sorted({str(row.get("kind", "")) for row in condition_rows if row.get("kind", "")}):
            groups.append((condition, kind, [row for row in condition_rows if str(row.get("kind", "")) == kind]))

    out: list[dict[str, Any]] = []
    for condition, group, rows in groups:
        for metric in RETINAL_TRANSFORM_METRICS:
            values = np.asarray([parse_float(row.get(metric)) for row in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            mean, sem = mean_sem(values)
            out.append(
                {
                    "condition": condition,
                    "group": group,
                    "metric": metric,
                    "n": int(finite.size),
                    "mean": mean,
                    "sem": sem,
                    "median": float(np.median(finite)) if finite.size else float("nan"),
                    "p95": float(np.percentile(finite, 95.0)) if finite.size else float("nan"),
                }
            )
    return out


def summarize_retinal_transform_gain_regression(
    summary_rows: list[dict[str, str]],
    retinal_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    metric_by_key: dict[tuple[tuple[str, str, str, str], str], float] = {}
    kind_by_key: dict[tuple[str, str, str, str], str] = {}
    for row in summary_rows:
        key = pair_key(row)
        condition = canonical_condition(row["condition"])
        metric_by_key[(key, condition)] = parse_float(row.get(PRIMARY_METRIC))
        kind_by_key[key] = str(row.get("kind", ""))

    retinal_by_key: dict[tuple[tuple[str, str, str, str], str], dict[str, float]] = {}
    for row in retinal_rows:
        key = pair_key(row)
        condition = canonical_condition(row.get("condition", ""))
        retinal_by_key[(key, condition)] = {
            metric: parse_float(row.get(metric))
            for metric in RETINAL_TRANSFORM_METRICS
        }
        kind_by_key.setdefault(key, str(row.get("kind", "")))

    pair_keys = sorted({key for key, _condition in metric_by_key} & {key for key, _condition in retinal_by_key})
    groups = [("all", pair_keys)]
    for kind in sorted({kind for kind in kind_by_key.values() if kind}):
        groups.append((kind, [key for key in pair_keys if kind_by_key.get(key) == kind]))

    out: list[dict[str, Any]] = []
    for group, keys in groups:
        for contrast, lhs, rhs in CONTRASTS:
            for transform_metric in RETINAL_TRANSFORM_METRICS:
                gain_delta: list[float] = []
                transform_delta: list[float] = []
                lhs_transform_abs: list[float] = []
                for key in keys:
                    lhs_gain = metric_by_key.get((key, lhs), float("nan"))
                    rhs_gain = metric_by_key.get((key, rhs), float("nan"))
                    lhs_transform = retinal_by_key.get((key, lhs), {}).get(transform_metric, float("nan"))
                    rhs_transform = retinal_by_key.get((key, rhs), {}).get(transform_metric, float("nan"))
                    if not (np.isfinite(lhs_gain) and np.isfinite(rhs_gain)):
                        continue
                    if np.isfinite(lhs_transform):
                        lhs_transform_abs.append(lhs_transform)
                    if np.isfinite(lhs_transform) and np.isfinite(rhs_transform):
                        gain_delta.append(lhs_gain - rhs_gain)
                        transform_delta.append(lhs_transform - rhs_transform)

                gain_mean, gain_sem = mean_sem(np.asarray(gain_delta, dtype=np.float64))
                transform_mean, transform_sem = mean_sem(np.asarray(transform_delta, dtype=np.float64))
                fit = linear_fit(transform_delta, gain_delta)
                out.append(
                    {
                        "group": group,
                        "contrast": contrast,
                        "lhs": lhs,
                        "rhs": rhs,
                        "transform_metric": transform_metric,
                        "n_delta_pairs": int(np.sum(np.isfinite(np.asarray(transform_delta, dtype=np.float64)) & np.isfinite(np.asarray(gain_delta, dtype=np.float64)))),
                        "gain_delta_mean": gain_mean,
                        "gain_delta_sem": gain_sem,
                        "transform_delta_mean": transform_mean,
                        "transform_delta_sem": transform_sem,
                        "delta_pearson_r": pearsonr(transform_delta, gain_delta),
                        "delta_slope": fit["slope"],
                        "delta_intercept": fit["intercept"],
                        "delta_r2": fit["r2"],
                        "lhs_abs_n": int(np.sum(np.isfinite(np.asarray(lhs_transform_abs, dtype=np.float64)))),
                        "lhs_abs_transform_mean": mean_sem(np.asarray(lhs_transform_abs, dtype=np.float64))[0],
                    }
                )
    return out


def audit_run(run_dir: Path, rows: list[dict[str, str]], series: SeriesTable) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    for rel in REQUIRED_FILES:
        path = run_dir / rel
        audit_rows.append(
            {
                "scope": "required_file",
                "item": rel,
                "status": "pass" if path.exists() else "fail",
                "value": str(path),
                "notes": "",
            }
        )

    conditions = sorted({canonical_condition(row["condition"]) for row in rows})
    pair_count_by_condition = {
        condition: len({pair_key(row) for row in rows if canonical_condition(row["condition"]) == condition})
        for condition in conditions
    }
    audit_rows.append(
        {
            "scope": "coverage",
            "item": "conditions",
            "status": "pass" if conditions else "fail",
            "value": ";".join(conditions),
            "notes": "",
        }
    )
    for condition, n_pairs in pair_count_by_condition.items():
        audit_rows.append(
            {
                "scope": "coverage",
                "item": f"n_pairs:{condition}",
                "status": "pass" if n_pairs > 0 else "fail",
                "value": n_pairs,
                "notes": "",
            }
        )

    columns = set(rows[0].keys()) if rows else set()
    for metric in (PRIMARY_METRIC, *COMPANION_METRICS):
        audit_rows.append(
            {
                "scope": "metric_column",
                "item": metric,
                "status": "pass" if metric in columns else "fail",
                "value": metric in columns,
                "notes": "",
            }
        )

    summary_keys = {(pair_key(row), canonical_condition(row["condition"])) for row in rows}
    series_keys = {(pair_key(row), canonical_condition(row["condition"])) for row in series.records}
    missing_in_series = summary_keys - series_keys
    missing_in_summary = series_keys - summary_keys
    audit_rows.append(
        {
            "scope": "series_match",
            "item": "summary_rows_missing_in_series",
            "status": "pass" if not missing_in_series else "fail",
            "value": len(missing_in_series),
            "notes": "",
        }
    )
    audit_rows.append(
        {
            "scope": "series_match",
            "item": "series_rows_missing_in_summary",
            "status": "pass" if not missing_in_summary else "fail",
            "value": len(missing_in_summary),
            "notes": "",
        }
    )

    qc_rows = read_csv_rows(run_dir / "metadata" / "03_trajectory_control_qc.csv")
    cloud_rows = [
        row for row in qc_rows
        if canonical_condition(row.get("condition", "")) == "random_amp_cloud_matched"
    ]
    random_candidate = [
        row for row in cloud_rows
        if "random_directions" in str(row.get("control_description", ""))
        and "fallback" not in str(row.get("control_description", ""))
    ]
    audit_rows.append(
        {
            "scope": "trajectory_qc",
            "item": "random_amp_cloud_matched_random_candidate_fraction",
            "status": "pass" if cloud_rows and len(random_candidate) / max(len(cloud_rows), 1) >= 0.5 else "warn",
            "value": f"{len(random_candidate)}/{len(cloud_rows)}",
            "notes": "fallback rows are acceptable but weaken the matched-random interpretation",
        }
    )
    return audit_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--retinal-qc-path",
        type=Path,
        default=DEFAULT_RETINAL_QC_PATH,
        help="Optional retinal movie transform QC CSV aligned by example/image/crop/kind/condition.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    summary_path = run_dir / "metadata" / "05_lagcube_information_summary.csv"
    rows = read_csv_rows(summary_path)
    if not rows:
        raise FileNotFoundError(f"No summary rows found at {summary_path}")
    series = load_series(run_dir)
    trajectory_qc_rows = read_csv_rows(run_dir / "metadata" / "03_trajectory_control_qc.csv")
    image_control_rows = read_csv_rows(run_dir / "metadata" / "02_pyramid_image_control_audit.csv")
    retinal_rows = read_csv_rows(args.retinal_qc_path)

    audit_rows = audit_run(run_dir, rows, series)
    audit_rows.append(
        {
            "scope": "retinal_transform_qc",
            "item": "retinal_qc_path",
            "status": "pass" if retinal_rows else "warn",
            "value": str(args.retinal_qc_path),
            "notes": "transform-to-gain checks are skipped if this table is missing",
        }
    )
    delta_rows = summarize_final_deltas(rows) + summarize_series_deltas(series)
    companion_rows = summarize_metric_deltas(rows)
    trajectory_fairness_rows = summarize_trajectory_fairness(trajectory_qc_rows)
    image_control_qc_rows = summarize_image_control_qc(image_control_rows)
    trajectory_gain_corr_rows = summarize_trajectory_qc_gain_correlations(rows, trajectory_qc_rows)
    retinal_transform_rows = summarize_retinal_transform_qc(retinal_rows) if retinal_rows else []
    retinal_gain_rows = (
        summarize_retinal_transform_gain_regression(rows, retinal_rows)
        if retinal_rows else []
    )

    audit_path = run_dir / "metadata" / "figure5_additional_checks_audit.csv"
    delta_path = run_dir / "metadata" / "figure5_delta_curve_summary.csv"
    companion_path = run_dir / "metadata" / "figure5_companion_metric_delta_summary.csv"
    trajectory_fairness_path = run_dir / "metadata" / "figure5_trajectory_fairness_summary.csv"
    image_control_qc_path = run_dir / "metadata" / "figure5_image_control_qc_summary.csv"
    trajectory_gain_corr_path = run_dir / "metadata" / "figure5_trajectory_qc_gain_correlations.csv"
    retinal_transform_path = run_dir / "metadata" / "figure5_retinal_transform_qc_summary.csv"
    retinal_gain_path = run_dir / "metadata" / "figure5_retinal_transform_gain_regression.csv"
    write_csv_rows(audit_path, audit_rows)
    write_csv_rows(delta_path, delta_rows)
    write_csv_rows(companion_path, companion_rows)
    write_csv_rows(trajectory_fairness_path, trajectory_fairness_rows)
    write_csv_rows(image_control_qc_path, image_control_qc_rows)
    write_csv_rows(trajectory_gain_corr_path, trajectory_gain_corr_rows)
    write_csv_rows(retinal_transform_path, retinal_transform_rows)
    write_csv_rows(retinal_gain_path, retinal_gain_rows)

    failed = [row for row in audit_rows if row["status"] == "fail"]
    warned = [row for row in audit_rows if row["status"] == "warn"]
    print("Figure 5 additional-check summary complete")
    print(f"  audit: {audit_path}")
    print(f"  deltas: {delta_path}")
    print(f"  companion metrics: {companion_path}")
    print(f"  trajectory fairness: {trajectory_fairness_path}")
    print(f"  image-control QC: {image_control_qc_path}")
    print(f"  trajectory QC/gain correlations: {trajectory_gain_corr_path}")
    print(f"  retinal transform QC: {retinal_transform_path}")
    print(f"  retinal transform/gain regression: {retinal_gain_path}")
    print(f"  audit rows: {len(audit_rows)}")
    print(f"  delta rows: {len(delta_rows)}")
    print(f"  companion rows: {len(companion_rows)}")
    print(f"  trajectory fairness rows: {len(trajectory_fairness_rows)}")
    print(f"  image-control QC rows: {len(image_control_qc_rows)}")
    print(f"  trajectory QC/gain correlation rows: {len(trajectory_gain_corr_rows)}")
    print(f"  retinal transform rows: {len(retinal_transform_rows)}")
    print(f"  retinal transform/gain rows: {len(retinal_gain_rows)}")
    print(f"  failures: {len(failed)}")
    print(f"  warnings: {len(warned)}")
    if failed:
        for row in failed:
            print(f"    FAIL {row['scope']}:{row['item']} value={row['value']}")
    if warned:
        for row in warned:
            print(f"    WARN {row['scope']}:{row['item']} value={row['value']}")


if __name__ == "__main__":
    main()
