#!/usr/bin/env python3
"""Unit-first BackImage RR100 SF and contour-alignment summaries.

This is a post-hoc analysis over the completed original/rot90 contour-axis
caches.  The primary estimators deliberately average within unit before
averaging units, so fixed SSI differences between response families cannot
silently dominate the SF motion-gain or contour-alignment conclusions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.analyze_backimage_rr100_rotation_crossover import (
    DEFAULT_ORIGINAL_RUN_DIR,
    DEFAULT_ROT90_RUN_DIR,
    DEFAULT_SF_GROUPS_CSV,
    EPS,
    angle_180_distance,
    bootstrap_ci,
    condition_frame,
    contour_axis_to_image_frame,
    fit_harmonic,
    load_cache,
    load_movie_frame,
    metric_arrays,
    orientation_axis_180,
    parse_csv_list,
    selection_masks,
    write_json,
)
from declan.redundancy_resolved_v1_population import load_population_view


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATIC_SF_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "sf_group_ssi_modulation_static_log_gaussian_nearest_threshold_low0p4_high1_v1/"
    "static_log_gaussian_nearest_sf_tuning_unit_groups.csv"
)
DEFAULT_OUT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_rot90_v1/"
    "unit_first_primary_results"
)
RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)
SF_ORDER = ["low_sf", "middle_sf", "high_sf"]
SF_LABEL = {"low_sf": "low SF", "middle_sf": "middle SF", "high_sf": "high SF"}
SF_COLORS = {"low_sf": "#1f77b4", "middle_sf": "#8a8a8a", "high_sf": "#d62728"}
RUN_STYLE = {"original": "-", "rot90": "--"}


@dataclass(frozen=True)
class ViewSpec:
    name: str
    title: str
    xcol: str
    fixed_col: str
    fixed_value: float


VIEW_SPECS = [
    ViewSpec("across_along0", "across, along=0", "across_scale", "along_scale", 0.0),
    ViewSpec("across_along1", "across, along=1", "across_scale", "along_scale", 1.0),
    ViewSpec("along_across0", "along, across=0", "along_scale", "across_scale", 0.0),
    ViewSpec("along_across1", "along, across=1", "along_scale", "across_scale", 1.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run-dir", type=Path, default=DEFAULT_ORIGINAL_RUN_DIR)
    parser.add_argument("--rot90-run-dir", type=Path, default=DEFAULT_ROT90_RUN_DIR)
    parser.add_argument("--primary-sf-csv", type=Path, default=DEFAULT_SF_GROUPS_CSV)
    parser.add_argument("--static-sf-csv", type=Path, default=DEFAULT_STATIC_SF_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metric", choices=("time_resolved", "mean_map"), default="time_resolved")
    parser.add_argument("--alignment-angle-deg", type=float, default=22.5)
    parser.add_argument("--min-orientation-selectivity", type=float, default=0.05)
    parser.add_argument("--axis-coordinate-frame", choices=("gaze", "image"), default="gaze")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=29)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--primary-grouping-label", default="dynamic_log_gaussian_marginal_low0p05_high0p5")
    parser.add_argument("--static-grouping-label", default="static_log_gaussian_nearest_low0p4_high1")
    parser.add_argument("--static-sf-column", default="static_log_gaussian_nearest_sf_cpd")
    parser.add_argument("--static-low-threshold", type=float, default=0.4)
    parser.add_argument("--static-high-threshold", type=float, default=1.0)
    parser.add_argument("--rr100-version", default=RR100_VERSION)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(path, index=False)


def load_grouped_units(
    path: Path,
    *,
    n_units: int,
    grouping_label: str,
    derive_from_column: str | None = None,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
) -> pd.DataFrame:
    units = pd.read_csv(path).copy()
    required = {
        "unit_index",
        "unit_label",
        "prior_preferred_orientation_deg",
        "prior_orientation_selectivity_index",
    }
    missing = sorted(required.difference(units.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    units["unit_index"] = pd.to_numeric(units["unit_index"], errors="coerce").astype(int)
    units = units[(units["unit_index"] >= 0) & (units["unit_index"] < int(n_units))].copy()

    if derive_from_column:
        if derive_from_column not in units.columns:
            raise ValueError(f"{path} does not contain {derive_from_column!r}.")
        sf = pd.to_numeric(units[derive_from_column], errors="coerce")
        lo = float(low_threshold)
        hi = float(high_threshold)
        units["sf_split_metric"] = sf
        units["sf_group"] = np.where(sf <= lo, "low_sf", np.where(sf >= hi, "high_sf", "middle_sf"))
        units["sf_group_definition"] = f"derived: low_sf <= {lo:g} cpd; high_sf >= {hi:g} cpd; middle otherwise"
        units["sf_split_metric_name"] = grouping_label
        units["sf_split_metric_column"] = derive_from_column
    else:
        if "sf_group" not in units.columns:
            raise ValueError(f"{path} has no sf_group column and no derive_from_column was supplied.")
        if "sf_split_metric" not in units.columns:
            metric_col = units.get("sf_split_metric_column", pd.Series([""] * len(units))).astype(str).iloc[0]
            if metric_col and metric_col in units.columns:
                units["sf_split_metric"] = pd.to_numeric(units[metric_col], errors="coerce")
            else:
                units["sf_split_metric"] = np.nan

    units["preferred_orientation_image_deg"] = orientation_axis_180(
        pd.to_numeric(units["prior_preferred_orientation_deg"], errors="coerce").to_numpy(dtype=float)
    )
    units["prior_orientation_selectivity_index"] = pd.to_numeric(
        units["prior_orientation_selectivity_index"], errors="coerce"
    )
    units["grouping_label"] = str(grouping_label)
    units["sf_group"] = units["sf_group"].astype(str)
    units["sf_rank_low_to_high"] = pd.to_numeric(
        units.get("sf_rank_low_to_high", pd.Series(np.arange(len(units)) + 1)),
        errors="coerce",
    )
    return units.sort_values(["sf_group", "sf_rank_low_to_high", "unit_index"]).reset_index(drop=True)


def group_unit_indices(units: pd.DataFrame, sf_group: str) -> np.ndarray:
    return units.loc[units["sf_group"].astype(str) == str(sf_group), "unit_index"].to_numpy(dtype=int)


def static_condition_index(conditions: pd.DataFrame) -> int:
    idx = np.flatnonzero(conditions["is_static_baseline"].to_numpy(dtype=bool))
    if idx.size != 1:
        raise ValueError(f"Expected one static baseline, found {idx.size}.")
    return int(idx[0])


def view_condition_indices(conditions: pd.DataFrame, view: ViewSpec) -> np.ndarray:
    mask = np.isclose(conditions[view.fixed_col].to_numpy(dtype=float), float(view.fixed_value))
    idx = np.flatnonzero(mask)
    order = np.argsort(conditions.iloc[idx][view.xcol].to_numpy(dtype=float))
    return idx[order]


def weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> float:
    arr = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(arr)
    if weights is None:
        return float(np.nanmean(arr[ok])) if np.any(ok) else float("nan")
    w = np.asarray(weights, dtype=np.float64)
    ok &= np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return float("nan")
    return float(np.sum(arr[ok] * w[ok]) / max(np.sum(w[ok]), EPS))


def nanmean_axis0(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(values)
    counts = np.sum(finite, axis=0)
    sums = np.where(finite, values, 0.0).sum(axis=0)
    out = np.full(counts.shape, np.nan, dtype=np.float64)
    ok = counts > 0
    out[ok] = sums[ok] / counts[ok]
    return out


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(ok) < 2:
        return float("nan")
    coef = np.polyfit(x[ok], y[ok], deg=1)
    return float(coef[0])


def bootstrap_unit_first_mean(
    values_mxu: np.ndarray,
    unit_indices: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    *,
    unit_weights: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    values = np.asarray(values_mxu, dtype=np.float64)[:, unit_indices]
    per_unit = np.nanmean(values, axis=0)
    weights = None if unit_weights is None else np.asarray(unit_weights, dtype=np.float64)[unit_indices]
    mean = weighted_mean(per_unit, weights)
    sem = float(np.nanstd(per_unit, ddof=1) / math.sqrt(np.isfinite(per_unit).sum())) if np.isfinite(per_unit).sum() > 1 else 0.0
    if int(n_boot) <= 0 or values.shape[0] <= 1 or unit_indices.size == 0:
        return mean, sem, float("nan"), float("nan")
    boot = np.full(int(n_boot), np.nan, dtype=np.float64)
    for bi in range(int(n_boot)):
        sample = rng.integers(0, values.shape[0], size=values.shape[0])
        boot[bi] = weighted_mean(np.nanmean(values[sample, :], axis=0), weights)
    ok = np.isfinite(boot)
    if not np.any(ok):
        return mean, sem, float("nan"), float("nan")
    lo, hi = np.percentile(boot[ok], [2.5, 97.5])
    return mean, sem, float(lo), float(hi)


def summarize_unit_first_motion_gain(
    *,
    run_label: str,
    bits: np.ndarray,
    conditions: pd.DataFrame,
    units: pd.DataFrame,
    grouping_label: str,
    n_boot: int,
    rng: np.random.Generator,
    unit_weights: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    static_idx = static_condition_index(conditions)
    gain = np.asarray(bits, dtype=np.float64) - np.asarray(bits[static_idx], dtype=np.float64)[None, :, :]

    condition_rows: list[dict[str, Any]] = []
    per_unit_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []

    for sf_group in SF_ORDER:
        uidx = group_unit_indices(units, sf_group)
        if uidx.size == 0:
            continue
        for ci, cond in conditions.iterrows():
            mean, sem, lo, hi = bootstrap_unit_first_mean(
                gain[int(ci)],
                uidx,
                rng,
                n_boot,
                unit_weights=unit_weights,
            )
            per_unit = np.nanmean(gain[int(ci)][:, uidx], axis=0)
            condition_rows.append(
                {
                    "grouping_label": grouping_label,
                    "run": run_label,
                    "condition_index": int(ci),
                    "condition_id": str(cond.condition_id),
                    "along_scale": float(cond.along_scale),
                    "across_scale": float(cond.across_scale),
                    "motion_scale": float(cond.motion_scale),
                    "sf_group": sf_group,
                    "n_units": int(uidx.size),
                    "unit_first_gain_mean": mean,
                    "unit_first_gain_sem_across_units": sem,
                    "unit_first_gain_ci_low_window_boot": lo,
                    "unit_first_gain_ci_high_window_boot": hi,
                    "median_per_unit_gain": float(np.nanmedian(per_unit)),
                    "fraction_units_positive_gain": float(np.mean(per_unit > 0.0)),
                }
            )
            for unit_index, value in zip(uidx, per_unit, strict=False):
                per_unit_rows.append(
                    {
                        "grouping_label": grouping_label,
                        "run": run_label,
                        "condition_index": int(ci),
                        "condition_id": str(cond.condition_id),
                        "along_scale": float(cond.along_scale),
                        "across_scale": float(cond.across_scale),
                        "motion_scale": float(cond.motion_scale),
                        "sf_group": sf_group,
                        "unit_index": int(unit_index),
                        "unit_label": f"u{int(unit_index):03d}",
                        "unit_first_gain": float(value),
                    }
                )

        for view in VIEW_SPECS:
            idx = view_condition_indices(conditions, view)
            x = conditions.iloc[idx][view.xcol].to_numpy(dtype=float)
            endpoint_ci = int(idx[np.nanargmax(x)])
            endpoint_gain = np.nanmean(gain[endpoint_ci][:, uidx], axis=0)
            per_unit_curve = np.asarray([np.nanmean(gain[int(ci)][:, uidx], axis=0) for ci in idx])
            slopes = np.asarray([linear_slope(x, per_unit_curve[:, j]) for j in range(uidx.size)], dtype=float)
            positive = endpoint_gain[endpoint_gain > 0.0]
            top5_positive = np.sort(positive)[-5:] if positive.size else np.asarray([], dtype=float)
            positive_share = float(np.sum(top5_positive) / max(np.sum(positive), EPS)) if positive.size else float("nan")
            signed_total = float(np.nansum(endpoint_gain))
            signed_share = float(np.nansum(np.sort(endpoint_gain)[-5:]) / signed_total) if abs(signed_total) > EPS else float("nan")
            endpoint_rows.append(
                {
                    "grouping_label": grouping_label,
                    "run": run_label,
                    "view": view.name,
                    "view_title": view.title,
                    "x_column": view.xcol,
                    "endpoint_condition_index": endpoint_ci,
                    "endpoint_condition_id": str(conditions.iloc[endpoint_ci].condition_id),
                    "endpoint_scale": float(np.nanmax(x)),
                    "sf_group": sf_group,
                    "n_units": int(uidx.size),
                    "fraction_units_positive_endpoint_gain": float(np.mean(endpoint_gain > 0.0)),
                    "median_endpoint_gain": float(np.nanmedian(endpoint_gain)),
                    "mean_endpoint_gain": float(np.nanmean(endpoint_gain)),
                    "median_linear_slope": float(np.nanmedian(slopes)),
                    "fraction_units_positive_slope": float(np.mean(slopes > 0.0)),
                    "top5_positive_endpoint_gain_share": positive_share,
                    "top5_signed_endpoint_gain_share": signed_share,
                }
            )

    return pd.DataFrame(condition_rows), pd.DataFrame(per_unit_rows), pd.DataFrame(endpoint_rows)


def unit_first_alignment_values(
    *,
    condition_index: int,
    orig_bits: np.ndarray,
    rot_bits: np.ndarray,
    masks: dict[str, dict[str, list[np.ndarray]]],
    sf_group: str,
    units: pd.DataFrame,
    unit_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if unit_indices is None:
        unit_indices = group_unit_indices(units, sf_group)
    keep = set(int(u) for u in unit_indices)
    n_movies = int(orig_bits.shape[1])
    n_units = int(orig_bits.shape[2])
    values = np.full((n_movies, n_units), np.nan, dtype=np.float64)
    for movie_idx in range(n_movies):
        aligned = [int(u) for u in masks[sf_group]["aligned"][movie_idx] if int(u) in keep]
        orthogonal = [int(u) for u in masks[sf_group]["orthogonal"][movie_idx] if int(u) in keep]
        if aligned:
            a = np.asarray(aligned, dtype=int)
            values[movie_idx, a] = orig_bits[condition_index, movie_idx, a] - rot_bits[condition_index, movie_idx, a]
        if orthogonal:
            o = np.asarray(orthogonal, dtype=int)
            values[movie_idx, o] = rot_bits[condition_index, movie_idx, o] - orig_bits[condition_index, movie_idx, o]
    return values[:, unit_indices], unit_indices


def bootstrap_alignment_mean(
    values_mxu: np.ndarray,
    static_mxu: np.ndarray | None,
    rng: np.random.Generator,
    n_boot: int,
    *,
    unit_weights: np.ndarray | None = None,
    unit_indices: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    values = np.asarray(values_mxu, dtype=np.float64)
    if static_mxu is not None:
        values = values - np.asarray(static_mxu, dtype=np.float64)
    per_unit = nanmean_axis0(values)
    weights = None
    if unit_weights is not None and unit_indices is not None:
        weights = np.asarray(unit_weights, dtype=np.float64)[unit_indices]
    mean = weighted_mean(per_unit, weights)
    sem = float(np.nanstd(per_unit, ddof=1) / math.sqrt(np.isfinite(per_unit).sum())) if np.isfinite(per_unit).sum() > 1 else 0.0
    if int(n_boot) <= 0 or values.shape[0] <= 1:
        return mean, sem, float("nan"), float("nan")
    boot = np.full(int(n_boot), np.nan, dtype=np.float64)
    for bi in range(int(n_boot)):
        sample = rng.integers(0, values.shape[0], size=values.shape[0])
        boot[bi] = weighted_mean(nanmean_axis0(values[sample, :]), weights)
    ok = np.isfinite(boot)
    if not np.any(ok):
        return mean, sem, float("nan"), float("nan")
    lo, hi = np.percentile(boot[ok], [2.5, 97.5])
    return mean, sem, float(lo), float(hi)


def summarize_unit_first_alignment_crossover(
    *,
    conditions: pd.DataFrame,
    units: pd.DataFrame,
    grouping_label: str,
    orig_bits: np.ndarray,
    rot_bits: np.ndarray,
    masks: dict[str, dict[str, list[np.ndarray]]],
    n_boot: int,
    rng: np.random.Generator,
    unit_weights: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    per_unit_rows: list[dict[str, Any]] = []
    static_idx = static_condition_index(conditions)
    static_by_group: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sf_group in SF_ORDER:
        uidx = group_unit_indices(units, sf_group)
        if uidx.size:
            static_by_group[sf_group] = unit_first_alignment_values(
                condition_index=static_idx,
                orig_bits=orig_bits,
                rot_bits=rot_bits,
                masks=masks,
                sf_group=sf_group,
                units=units,
                unit_indices=uidx,
            )

    for sf_group in SF_ORDER:
        uidx = group_unit_indices(units, sf_group)
        if uidx.size == 0:
            continue
        static_values, _ = static_by_group[sf_group]
        static_per_unit = nanmean_axis0(static_values)
        for ci, cond in conditions.iterrows():
            values, _ = unit_first_alignment_values(
                condition_index=int(ci),
                orig_bits=orig_bits,
                rot_bits=rot_bits,
                masks=masks,
                sf_group=sf_group,
                units=units,
                unit_indices=uidx,
            )
            c_mean, c_sem, c_lo, c_hi = bootstrap_alignment_mean(
                values,
                None,
                rng,
                n_boot,
                unit_weights=unit_weights,
                unit_indices=uidx,
            )
            d_mean, d_sem, d_lo, d_hi = bootstrap_alignment_mean(
                values,
                static_values,
                rng,
                n_boot,
                unit_weights=unit_weights,
                unit_indices=uidx,
            )
            per_unit = nanmean_axis0(values)
            per_unit_delta = per_unit - static_per_unit
            valid_counts = np.sum(np.isfinite(values), axis=0)
            rows.append(
                {
                    "grouping_label": grouping_label,
                    "condition_index": int(ci),
                    "condition_id": str(cond.condition_id),
                    "along_scale": float(cond.along_scale),
                    "across_scale": float(cond.across_scale),
                    "motion_scale": float(cond.motion_scale),
                    "sf_group": sf_group,
                    "n_units": int(uidx.size),
                    "n_units_with_valid_windows": int(np.count_nonzero(valid_counts > 0)),
                    "median_valid_windows_per_unit": float(np.nanmedian(valid_counts)),
                    "crossover_c_mean": c_mean,
                    "crossover_c_sem_across_units": c_sem,
                    "crossover_c_ci_low_window_boot": c_lo,
                    "crossover_c_ci_high_window_boot": c_hi,
                    "delta_c_mean": d_mean,
                    "delta_c_sem_across_units": d_sem,
                    "delta_c_ci_low_window_boot": d_lo,
                    "delta_c_ci_high_window_boot": d_hi,
                    "median_per_unit_c": float(np.nanmedian(per_unit)),
                    "median_per_unit_delta_c": float(np.nanmedian(per_unit_delta)),
                    "fraction_units_positive_c": float(np.mean(per_unit > 0.0)),
                    "fraction_units_positive_delta_c": float(np.mean(per_unit_delta > 0.0)),
                }
            )
            for unit_index, value, delta, n_valid in zip(uidx, per_unit, per_unit_delta, valid_counts, strict=False):
                per_unit_rows.append(
                    {
                        "grouping_label": grouping_label,
                        "condition_index": int(ci),
                        "condition_id": str(cond.condition_id),
                        "along_scale": float(cond.along_scale),
                        "across_scale": float(cond.across_scale),
                        "motion_scale": float(cond.motion_scale),
                        "sf_group": sf_group,
                        "unit_index": int(unit_index),
                        "unit_label": f"u{int(unit_index):03d}",
                        "n_valid_fixations": int(n_valid),
                        "unit_first_c": float(value),
                        "unit_first_delta_c": float(delta),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(per_unit_rows)


def direct_alignment_samples(
    *,
    condition_index: int,
    bits: np.ndarray,
    masks: dict[str, dict[str, list[np.ndarray]]],
    sf_group: str,
    units: pd.DataFrame,
    unit_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if unit_indices is None:
        unit_indices = group_unit_indices(units, sf_group)
    unit_indices = np.asarray(unit_indices, dtype=int)
    col_by_unit = {int(unit_index): col for col, unit_index in enumerate(unit_indices)}
    aligned = np.full((bits.shape[1], unit_indices.size), np.nan, dtype=np.float64)
    orthogonal = np.full((bits.shape[1], unit_indices.size), np.nan, dtype=np.float64)
    for movie_idx in range(bits.shape[1]):
        for unit_index in masks[sf_group]["aligned"][movie_idx]:
            col = col_by_unit.get(int(unit_index))
            if col is not None:
                aligned[movie_idx, col] = float(bits[condition_index, movie_idx, int(unit_index)])
        for unit_index in masks[sf_group]["orthogonal"][movie_idx]:
            col = col_by_unit.get(int(unit_index))
            if col is not None:
                orthogonal[movie_idx, col] = float(bits[condition_index, movie_idx, int(unit_index)])
    return aligned, orthogonal, unit_indices


def direct_alignment_per_unit(
    aligned_runs: list[np.ndarray],
    orthogonal_runs: list[np.ndarray],
    sample: np.ndarray | None = None,
) -> np.ndarray:
    if sample is None:
        aligned = np.vstack(aligned_runs)
        orthogonal = np.vstack(orthogonal_runs)
    else:
        aligned = np.vstack([arr[sample, :] for arr in aligned_runs])
        orthogonal = np.vstack([arr[sample, :] for arr in orthogonal_runs])
    return nanmean_axis0(aligned) - nanmean_axis0(orthogonal)


def direct_alignment_counts(aligned_runs: list[np.ndarray], orthogonal_runs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    aligned_counts = np.sum([np.sum(np.isfinite(arr), axis=0) for arr in aligned_runs], axis=0)
    orthogonal_counts = np.sum([np.sum(np.isfinite(arr), axis=0) for arr in orthogonal_runs], axis=0)
    return np.asarray(aligned_counts, dtype=int), np.asarray(orthogonal_counts, dtype=int)


def bootstrap_direct_alignment_mean(
    aligned_runs: list[np.ndarray],
    orthogonal_runs: list[np.ndarray],
    static_aligned_runs: list[np.ndarray] | None,
    static_orthogonal_runs: list[np.ndarray] | None,
    rng: np.random.Generator,
    n_boot: int,
    *,
    unit_weights: np.ndarray | None = None,
    unit_indices: np.ndarray | None = None,
) -> tuple[float, float, float, float, np.ndarray]:
    per_unit = direct_alignment_per_unit(aligned_runs, orthogonal_runs)
    if static_aligned_runs is not None and static_orthogonal_runs is not None:
        per_unit = per_unit - direct_alignment_per_unit(static_aligned_runs, static_orthogonal_runs)
    weights = None
    if unit_weights is not None and unit_indices is not None:
        weights = np.asarray(unit_weights, dtype=np.float64)[unit_indices]
    mean = weighted_mean(per_unit, weights)
    sem = float(np.nanstd(per_unit, ddof=1) / math.sqrt(np.isfinite(per_unit).sum())) if np.isfinite(per_unit).sum() > 1 else 0.0
    n_windows = int(aligned_runs[0].shape[0]) if aligned_runs else 0
    if int(n_boot) <= 0 or n_windows <= 1:
        return mean, sem, float("nan"), float("nan"), per_unit
    boot = np.full(int(n_boot), np.nan, dtype=np.float64)
    for bi in range(int(n_boot)):
        sample = rng.integers(0, n_windows, size=n_windows)
        boot_per_unit = direct_alignment_per_unit(aligned_runs, orthogonal_runs, sample=sample)
        if static_aligned_runs is not None and static_orthogonal_runs is not None:
            boot_per_unit = boot_per_unit - direct_alignment_per_unit(
                static_aligned_runs,
                static_orthogonal_runs,
                sample=sample,
            )
        boot[bi] = weighted_mean(boot_per_unit, weights)
    ok = np.isfinite(boot)
    if not np.any(ok):
        return mean, sem, float("nan"), float("nan"), per_unit
    lo, hi = np.percentile(boot[ok], [2.5, 97.5])
    return mean, sem, float(lo), float(hi), per_unit


def summarize_unit_first_direct_alignment(
    *,
    conditions: pd.DataFrame,
    units: pd.DataFrame,
    grouping_label: str,
    orig_bits: np.ndarray,
    rot_bits: np.ndarray,
    orig_masks: dict[str, dict[str, list[np.ndarray]]],
    rot_masks: dict[str, dict[str, list[np.ndarray]]],
    n_boot: int,
    rng: np.random.Generator,
    unit_weights: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    per_unit_rows: list[dict[str, Any]] = []
    static_idx = static_condition_index(conditions)
    run_modes = ["original", "rot90", "original_plus_rot90"]

    static_samples: dict[tuple[str, str], tuple[list[np.ndarray], list[np.ndarray], np.ndarray]] = {}
    for sf_group in SF_ORDER:
        uidx = group_unit_indices(units, sf_group)
        if uidx.size == 0:
            continue
        orig_a, orig_o, _ = direct_alignment_samples(
            condition_index=static_idx,
            bits=orig_bits,
            masks=orig_masks,
            sf_group=sf_group,
            units=units,
            unit_indices=uidx,
        )
        rot_a, rot_o, _ = direct_alignment_samples(
            condition_index=static_idx,
            bits=rot_bits,
            masks=rot_masks,
            sf_group=sf_group,
            units=units,
            unit_indices=uidx,
        )
        static_samples[(sf_group, "original")] = ([orig_a], [orig_o], uidx)
        static_samples[(sf_group, "rot90")] = ([rot_a], [rot_o], uidx)
        static_samples[(sf_group, "original_plus_rot90")] = ([orig_a, rot_a], [orig_o, rot_o], uidx)

    for sf_group in SF_ORDER:
        uidx = group_unit_indices(units, sf_group)
        if uidx.size == 0:
            continue
        for ci, cond in conditions.iterrows():
            orig_a, orig_o, _ = direct_alignment_samples(
                condition_index=int(ci),
                bits=orig_bits,
                masks=orig_masks,
                sf_group=sf_group,
                units=units,
                unit_indices=uidx,
            )
            rot_a, rot_o, _ = direct_alignment_samples(
                condition_index=int(ci),
                bits=rot_bits,
                masks=rot_masks,
                sf_group=sf_group,
                units=units,
                unit_indices=uidx,
            )
            current_samples = {
                "original": ([orig_a], [orig_o]),
                "rot90": ([rot_a], [rot_o]),
                "original_plus_rot90": ([orig_a, rot_a], [orig_o, rot_o]),
            }
            for run_mode in run_modes:
                aligned_runs, orthogonal_runs = current_samples[run_mode]
                static_aligned_runs, static_orthogonal_runs, _ = static_samples[(sf_group, run_mode)]
                d_mean, d_sem, d_lo, d_hi, per_unit_d = bootstrap_direct_alignment_mean(
                    aligned_runs,
                    orthogonal_runs,
                    None,
                    None,
                    rng,
                    n_boot,
                    unit_weights=unit_weights,
                    unit_indices=uidx,
                )
                delta_mean, delta_sem, delta_lo, delta_hi, per_unit_delta = bootstrap_direct_alignment_mean(
                    aligned_runs,
                    orthogonal_runs,
                    static_aligned_runs,
                    static_orthogonal_runs,
                    rng,
                    n_boot,
                    unit_weights=unit_weights,
                    unit_indices=uidx,
                )
                aligned_counts, orthogonal_counts = direct_alignment_counts(aligned_runs, orthogonal_runs)
                valid = np.isfinite(per_unit_d)
                rows.append(
                    {
                        "grouping_label": grouping_label,
                        "run_mode": run_mode,
                        "condition_index": int(ci),
                        "condition_id": str(cond.condition_id),
                        "along_scale": float(cond.along_scale),
                        "across_scale": float(cond.across_scale),
                        "motion_scale": float(cond.motion_scale),
                        "sf_group": sf_group,
                        "n_units": int(uidx.size),
                        "n_units_with_direct_alignment": int(np.count_nonzero(valid)),
                        "median_aligned_windows_per_unit": float(np.nanmedian(aligned_counts)),
                        "median_orthogonal_windows_per_unit": float(np.nanmedian(orthogonal_counts)),
                        "direct_alignment_d_mean": d_mean,
                        "direct_alignment_d_sem_across_units": d_sem,
                        "direct_alignment_d_ci_low_window_boot": d_lo,
                        "direct_alignment_d_ci_high_window_boot": d_hi,
                        "direct_delta_d_mean": delta_mean,
                        "direct_delta_d_sem_across_units": delta_sem,
                        "direct_delta_d_ci_low_window_boot": delta_lo,
                        "direct_delta_d_ci_high_window_boot": delta_hi,
                        "median_per_unit_d": float(np.nanmedian(per_unit_d)),
                        "median_per_unit_delta_d": float(np.nanmedian(per_unit_delta)),
                        "fraction_units_positive_d": float(np.mean(per_unit_d[valid] > 0.0)) if np.any(valid) else float("nan"),
                        "fraction_units_positive_delta_d": float(np.mean(per_unit_delta[np.isfinite(per_unit_delta)] > 0.0))
                        if np.any(np.isfinite(per_unit_delta))
                        else float("nan"),
                    }
                )
                for unit_index, d_value, delta_value, n_a, n_o in zip(
                    uidx,
                    per_unit_d,
                    per_unit_delta,
                    aligned_counts,
                    orthogonal_counts,
                    strict=False,
                ):
                    per_unit_rows.append(
                        {
                            "grouping_label": grouping_label,
                            "run_mode": run_mode,
                            "condition_index": int(ci),
                            "condition_id": str(cond.condition_id),
                            "along_scale": float(cond.along_scale),
                            "across_scale": float(cond.across_scale),
                            "motion_scale": float(cond.motion_scale),
                            "sf_group": sf_group,
                            "unit_index": int(unit_index),
                            "unit_label": f"u{int(unit_index):03d}",
                            "n_aligned_windows": int(n_a),
                            "n_orthogonal_windows": int(n_o),
                            "unit_first_direct_d": float(d_value),
                            "unit_first_direct_delta_d": float(delta_value),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(per_unit_rows)


def weighted_group_gap_for_fixations(
    *,
    condition_index: int,
    bits: np.ndarray,
    masks: dict[str, dict[str, list[np.ndarray]]],
    sf_group: str,
    unit_keep: Iterable[int],
    unit_weights: np.ndarray | None = None,
) -> np.ndarray:
    keep = set(int(u) for u in unit_keep)
    gaps = np.full(bits.shape[1], np.nan, dtype=np.float64)
    for movie_idx in range(bits.shape[1]):
        a = np.asarray([int(u) for u in masks[sf_group]["aligned"][movie_idx] if int(u) in keep], dtype=int)
        o = np.asarray([int(u) for u in masks[sf_group]["orthogonal"][movie_idx] if int(u) in keep], dtype=int)
        if a.size == 0 or o.size == 0:
            continue
        aw = None if unit_weights is None else unit_weights[a]
        ow = None if unit_weights is None else unit_weights[o]
        gaps[movie_idx] = weighted_mean(bits[condition_index, movie_idx, a], aw) - weighted_mean(
            bits[condition_index, movie_idx, o],
            ow,
        )
    return gaps


def harmonic_amplitude_for_mode(
    *,
    condition_index: int,
    bits: np.ndarray,
    movies: pd.DataFrame,
    masks: dict[str, dict[str, list[np.ndarray]]],
    units: pd.DataFrame,
    sf_group: str,
    unit_weights: np.ndarray | None = None,
    exclude_units: set[int] | None = None,
) -> dict[str, float]:
    uidx = group_unit_indices(units, sf_group)
    if exclude_units:
        uidx = np.asarray([int(u) for u in uidx if int(u) not in exclude_units], dtype=int)
    gaps = weighted_group_gap_for_fixations(
        condition_index=condition_index,
        bits=bits,
        masks=masks,
        sf_group=sf_group,
        unit_keep=uidx,
        unit_weights=unit_weights,
    )
    return fit_harmonic(movies["contour_axis_image_deg"].to_numpy(dtype=float), gaps)


def key_estimates_for_mode(
    *,
    mode_label: str,
    grouping_label: str,
    conditions: pd.DataFrame,
    movies: pd.DataFrame,
    units: pd.DataFrame,
    orig_bits: np.ndarray,
    rot_bits: np.ndarray,
    masks: dict[str, dict[str, list[np.ndarray]]],
    unit_weights: np.ndarray | None = None,
    exclude_units: set[int] | None = None,
) -> dict[str, Any]:
    static_idx = static_condition_index(conditions)
    low_units = group_unit_indices(units, "low_sf")
    high_units = group_unit_indices(units, "high_sf")
    if exclude_units:
        low_units = np.asarray([u for u in low_units if int(u) not in exclude_units], dtype=int)
        high_units = np.asarray([u for u in high_units if int(u) not in exclude_units], dtype=int)

    across_along1 = next(view for view in VIEW_SPECS if view.name == "across_along1")
    idx = view_condition_indices(conditions, across_along1)
    x = conditions.iloc[idx][across_along1.xcol].to_numpy(dtype=float)
    endpoint_ci = int(idx[np.nanargmax(x)])
    gain_orig = orig_bits[endpoint_ci] - orig_bits[static_idx]
    gain_rot = rot_bits[endpoint_ci] - rot_bits[static_idx]
    low_orig_units = np.nanmean(gain_orig[:, low_units], axis=0) if low_units.size else np.asarray([])
    low_rot_units = np.nanmean(gain_rot[:, low_units], axis=0) if low_units.size else np.asarray([])
    low_weights = None if unit_weights is None else unit_weights[low_units]
    low_gain_orig = weighted_mean(low_orig_units, low_weights)
    low_gain_rot = weighted_mean(low_rot_units, low_weights)
    low_gain_pair_mean = float(np.nanmean([low_gain_orig, low_gain_rot]))

    static_values, _ = unit_first_alignment_values(
        condition_index=static_idx,
        orig_bits=orig_bits,
        rot_bits=rot_bits,
        masks=masks,
        sf_group="high_sf",
        units=units,
        unit_indices=high_units,
    )
    high_weights = None if unit_weights is None else unit_weights[high_units]
    high_c0 = weighted_mean(nanmean_axis0(static_values), high_weights)

    pure_across_1_idx = int(
        conditions.index[np.isclose(conditions["along_scale"], 0.0) & np.isclose(conditions["across_scale"], 1.0)][0]
    )
    c1_values, _ = unit_first_alignment_values(
        condition_index=pure_across_1_idx,
        orig_bits=orig_bits,
        rot_bits=rot_bits,
        masks=masks,
        sf_group="high_sf",
        units=units,
        unit_indices=high_units,
    )
    high_delta_c1 = weighted_mean(nanmean_axis0(c1_values - static_values), high_weights)

    harmonic = harmonic_amplitude_for_mode(
        condition_index=static_idx,
        bits=orig_bits,
        movies=movies,
        masks=masks,
        units=units,
        sf_group="high_sf",
        unit_weights=unit_weights,
        exclude_units=exclude_units,
    )
    return {
        "grouping_label": grouping_label,
        "mode": mode_label,
        "n_low_units": int(low_units.size),
        "n_high_units": int(high_units.size),
        "low_sf_across_along1_endpoint_condition": str(conditions.iloc[endpoint_ci].condition_id),
        "low_sf_unit_first_gain_orig": low_gain_orig,
        "low_sf_unit_first_gain_rot90": low_gain_rot,
        "low_sf_unit_first_gain_orig_rot90_mean": low_gain_pair_mean,
        "high_sf_static_c0": high_c0,
        "high_sf_delta_c_pure_across_1x": high_delta_c1,
        "high_sf_static_harmonic_amplitude_original": harmonic["anisotropy_amplitude_bits_per_spike"],
        "high_sf_static_harmonic_gamma_original": harmonic["gamma_bits_per_spike"],
        "high_sf_static_harmonic_preferred_gap_axis_deg_original": harmonic["preferred_gap_axis_deg"],
    }


def build_robustness_tables(
    *,
    grouping_label: str,
    conditions: pd.DataFrame,
    movies: pd.DataFrame,
    units: pd.DataFrame,
    orig_bits: np.ndarray,
    rot_bits: np.ndarray,
    masks: dict[str, dict[str, list[np.ndarray]]],
    cluster_weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    base_modes = [
        ("RR100 equal-unit", None, set()),
        ("without u054", None, {54}),
        ("without u018", None, {18}),
        ("without u054+u018", None, {54, 18}),
        ("RR100 cluster-size weighted", cluster_weights, set()),
    ]
    for label, weights, exclude in base_modes:
        rows.append(
            key_estimates_for_mode(
                mode_label=label,
                grouping_label=grouping_label,
                conditions=conditions,
                movies=movies,
                units=units,
                orig_bits=orig_bits,
                rot_bits=rot_bits,
                masks=masks,
                unit_weights=weights,
                exclude_units=exclude,
            )
        )

    all_unit_indices = sorted(int(u) for u in units["unit_index"].to_numpy(dtype=int))
    for unit_index in all_unit_indices:
        row = key_estimates_for_mode(
            mode_label=f"leave-one-out u{unit_index:03d}",
            grouping_label=grouping_label,
            conditions=conditions,
            movies=movies,
            units=units,
            orig_bits=orig_bits,
            rot_bits=rot_bits,
            masks=masks,
            unit_weights=None,
            exclude_units={unit_index},
        )
        row["left_out_unit_index"] = int(unit_index)
        loo_rows.append(row)

    full_row = {key: np.nan for key in rows[0].keys()}
    full_row.update(
        {
            "grouping_label": grouping_label,
            "mode": "full756 direct",
            "note": "not available from RR100-only contour SSI caches",
        }
    )
    rows.append(full_row)

    return pd.DataFrame(rows), pd.DataFrame(loo_rows)


def absolute_orientation_table(
    *,
    grouping_label: str,
    units: pd.DataFrame,
    orig_bits: np.ndarray,
    orig_spikes: np.ndarray,
    rot_bits: np.ndarray,
    rot_spikes: np.ndarray,
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    static_idx = static_condition_index(conditions)
    rows: list[dict[str, Any]] = []
    high_units = group_unit_indices(units, "high_sf")
    total_num = {}
    for run_label, bits, spikes in [
        ("original", orig_bits, orig_spikes),
        ("rot90", rot_bits, rot_spikes),
    ]:
        per_unit_num = np.nansum(bits[static_idx][:, high_units] * spikes[static_idx][:, high_units], axis=0)
        total_num[run_label] = float(np.nansum(per_unit_num))
        for unit_index, numerator in zip(high_units, per_unit_num, strict=False):
            unit = units.loc[units["unit_index"] == int(unit_index)].iloc[0]
            rows.append(
                {
                    "grouping_label": grouping_label,
                    "run": run_label,
                    "unit_index": int(unit_index),
                    "unit_label": f"u{int(unit_index):03d}",
                    "preferred_orientation_deg": float(unit["preferred_orientation_image_deg"]),
                    "orientation_selectivity": float(unit["prior_orientation_selectivity_index"]),
                    "sf_metric": float(unit["sf_split_metric"]),
                    "static_unit_ssi": float(np.nanmean(bits[static_idx, :, int(unit_index)])),
                    "static_expected_spikes": float(np.nansum(spikes[static_idx, :, int(unit_index)])),
                    "static_information_numerator": float(numerator),
                    "static_information_numerator_share": float(numerator / max(total_num[run_label], EPS)),
                }
            )
    return pd.DataFrame(rows)


def plot_motion_gain(summary: pd.DataFrame, out_dir: Path, dpi: int, grouping_label: str) -> None:
    df = summary[summary["grouping_label"] == grouping_label].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True, constrained_layout=True)
    for ax, view in zip(axes.flat, VIEW_SPECS, strict=False):
        sub = df[np.isclose(df[view.fixed_col], view.fixed_value)].copy()
        for sf_group in SF_ORDER:
            for run_label in ["original", "rot90"]:
                line = sub[(sub["sf_group"] == sf_group) & (sub["run"] == run_label)].sort_values(view.xcol)
                if line.empty:
                    continue
                x = line[view.xcol].to_numpy(dtype=float)
                y = line["unit_first_gain_mean"].to_numpy(dtype=float)
                lo = line["unit_first_gain_ci_low_window_boot"].to_numpy(dtype=float)
                hi = line["unit_first_gain_ci_high_window_boot"].to_numpy(dtype=float)
                label = f"{SF_LABEL[sf_group]}, {run_label}" if view.name == "across_along0" else None
                ax.plot(
                    x,
                    y,
                    RUN_STYLE[run_label],
                    marker="o",
                    color=SF_COLORS[sf_group],
                    linewidth=2.0 if run_label == "original" else 1.8,
                    label=label,
                )
                ax.fill_between(x, lo, hi, color=SF_COLORS[sf_group], alpha=0.10, linewidth=0)
        ax.axhline(0, color="0.55", lw=1, ls="--")
        ax.set_title(view.title)
        ax.set_xlabel(view.xcol.replace("_", " "))
        ax.grid(True, color="0.9", linewidth=0.8)
    axes[0, 0].set_ylabel("G(s): unit-first SSI gain")
    axes[1, 0].set_ylabel("G(s): unit-first SSI gain")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Unit-first motion gain by SF: original solid, rot90 dashed", fontsize=14)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"backimage_rr100_unit_first_motion_gain_{grouping_label}.{ext}", dpi=dpi)
    plt.close(fig)


def plot_absolute_orientation(table: pd.DataFrame, out_dir: Path, dpi: int, grouping_label: str) -> None:
    df = table[table["grouping_label"] == grouping_label].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    pivot = df.pivot(index="unit_index", columns="run", values="static_unit_ssi")
    orient = df.drop_duplicates("unit_index").set_index("unit_index")["preferred_orientation_deg"]
    share = df[df["run"] == "original"].set_index("unit_index")["static_information_numerator_share"]
    for unit_index in pivot.index:
        if "original" in pivot.columns and "rot90" in pivot.columns:
            ax.plot(
                [orient.loc[unit_index], orient.loc[unit_index]],
                [pivot.loc[unit_index, "original"], pivot.loc[unit_index, "rot90"]],
                color="0.72",
                lw=0.9,
                zorder=1,
            )
    for run_label, marker, color in [("original", "o", "#333333"), ("rot90", "^", "#d95f02")]:
        sub = df[df["run"] == run_label]
        sizes = 35 + 900 * np.sqrt(np.maximum(sub["static_information_numerator_share"].to_numpy(dtype=float), 0.0))
        ax.scatter(
            sub["preferred_orientation_deg"],
            sub["static_unit_ssi"],
            s=sizes,
            marker=marker,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
            label=run_label,
            zorder=2,
        )

    original = df[df["run"] == "original"].copy()
    theta = np.deg2rad(2.0 * original["preferred_orientation_deg"].to_numpy(dtype=float))
    y = original["static_unit_ssi"].to_numpy(dtype=float)
    ok = np.isfinite(theta) & np.isfinite(y)
    if np.count_nonzero(ok) >= 3:
        xmat = np.column_stack([np.ones(np.count_nonzero(ok)), np.cos(theta[ok]), np.sin(theta[ok])])
        coef, *_ = np.linalg.lstsq(xmat, y[ok], rcond=None)
        grid = np.linspace(0, 180, 361)
        pred = coef[0] + coef[1] * np.cos(np.deg2rad(2 * grid)) + coef[2] * np.sin(np.deg2rad(2 * grid))
        ax.plot(grid, pred, color="#111111", lw=1.5, alpha=0.75, label="original cos2 fit")

    for unit_index in [54, 18]:
        sub = df[(df["unit_index"] == unit_index) & (df["run"] == "original")]
        if not sub.empty:
            row = sub.iloc[0]
            ax.annotate(
                f"u{unit_index:03d}",
                (float(row["preferred_orientation_deg"]), float(row["static_unit_ssi"])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="#111111",
            )
    ax.set_xlim(-4, 184)
    ax.set_xlabel("preferred absolute orientation (deg)")
    ax.set_ylabel("static per-unit SSI (bits/spike)")
    ax.set_title("High-SF static SSI by absolute unit orientation")
    ax.grid(True, color="0.9", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"backimage_rr100_high_sf_absolute_orientation_{grouping_label}.{ext}", dpi=dpi)
    plt.close(fig)


def plot_alignment_crossover(summary: pd.DataFrame, out_dir: Path, dpi: int, grouping_label: str) -> None:
    df = summary[(summary["grouping_label"] == grouping_label) & (summary["sf_group"].isin(["low_sf", "high_sf"]))].copy()
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey="row", constrained_layout=True)
    for col, view in enumerate(VIEW_SPECS):
        sub = df[np.isclose(df[view.fixed_col], view.fixed_value)].copy()
        for row_idx, (metric, lo_col, hi_col, ylabel) in enumerate(
            [
                ("crossover_c_mean", "crossover_c_ci_low_window_boot", "crossover_c_ci_high_window_boot", "C(s)"),
                ("delta_c_mean", "delta_c_ci_low_window_boot", "delta_c_ci_high_window_boot", "C(s) - C(0)"),
            ]
        ):
            ax = axes[row_idx, col]
            for sf_group in ["high_sf", "low_sf"]:
                line = sub[sub["sf_group"] == sf_group].sort_values(view.xcol)
                if line.empty:
                    continue
                x = line[view.xcol].to_numpy(dtype=float)
                y = line[metric].to_numpy(dtype=float)
                lo = line[lo_col].to_numpy(dtype=float)
                hi = line[hi_col].to_numpy(dtype=float)
                ax.plot(x, y, marker="o", color=SF_COLORS[sf_group], label=SF_LABEL[sf_group] if col == 0 and row_idx == 0 else None)
                ax.fill_between(x, lo, hi, color=SF_COLORS[sf_group], alpha=0.15, linewidth=0)
            ax.axhline(0, color="0.55", lw=1, ls="--")
            ax.set_title(view.title)
            ax.set_xlabel(view.xcol.replace("_", " "))
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.grid(True, color="0.9", linewidth=0.8)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Paired rot90 label-swap contour-alignment crossover", fontsize=14)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"backimage_rr100_unit_first_alignment_crossover_{grouping_label}.{ext}", dpi=dpi)
    plt.close(fig)


def plot_direct_alignment_collapsed(summary: pd.DataFrame, out_dir: Path, dpi: int, grouping_label: str) -> None:
    df = summary[
        (summary["grouping_label"] == grouping_label)
        & (summary["run_mode"] == "original_plus_rot90")
        & (summary["sf_group"].isin(["low_sf", "high_sf"]))
    ].copy()
    if df.empty:
        return
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey="row", constrained_layout=True)
    for col, view in enumerate(VIEW_SPECS):
        sub = df[np.isclose(df[view.fixed_col], view.fixed_value)].copy()
        for row_idx, (metric, lo_col, hi_col, ylabel) in enumerate(
            [
                (
                    "direct_alignment_d_mean",
                    "direct_alignment_d_ci_low_window_boot",
                    "direct_alignment_d_ci_high_window_boot",
                    "D(s)",
                ),
                (
                    "direct_delta_d_mean",
                    "direct_delta_d_ci_low_window_boot",
                    "direct_delta_d_ci_high_window_boot",
                    "D(s) - D(0)",
                ),
            ]
        ):
            ax = axes[row_idx, col]
            for sf_group in ["high_sf", "low_sf"]:
                line = sub[sub["sf_group"] == sf_group].sort_values(view.xcol)
                if line.empty:
                    continue
                x = line[view.xcol].to_numpy(dtype=float)
                y = line[metric].to_numpy(dtype=float)
                lo = line[lo_col].to_numpy(dtype=float)
                hi = line[hi_col].to_numpy(dtype=float)
                ax.plot(
                    x,
                    y,
                    marker="o",
                    color=SF_COLORS[sf_group],
                    label=SF_LABEL[sf_group] if col == 0 and row_idx == 0 else None,
                )
                ax.fill_between(x, lo, hi, color=SF_COLORS[sf_group], alpha=0.15, linewidth=0)
            ax.axhline(0, color="0.55", lw=1, ls="--")
            ax.set_title(view.title)
            ax.set_xlabel(view.xcol.replace("_", " "))
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.grid(True, color="0.9", linewidth=0.8)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Direct unit-first alignment: original+rot90 collapsed", fontsize=14)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"backimage_rr100_unit_first_direct_alignment_collapsed_{grouping_label}.{ext}", dpi=dpi)
    plt.close(fig)


def plot_direct_alignment_by_run(summary: pd.DataFrame, out_dir: Path, dpi: int, grouping_label: str) -> None:
    df = summary[
        (summary["grouping_label"] == grouping_label)
        & (summary["sf_group"].isin(["low_sf", "high_sf"]))
        & (summary["run_mode"].isin(["original", "rot90", "original_plus_rot90"]))
    ].copy()
    if df.empty:
        return
    styles = {
        "original": ("-", 1.6, "original"),
        "rot90": ("--", 1.6, "rot90"),
        "original_plus_rot90": ("-", 2.8, "collapsed"),
    }
    alpha = {"original": 0.55, "rot90": 0.55, "original_plus_rot90": 1.0}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey="row", constrained_layout=True)
    for col, view in enumerate(VIEW_SPECS):
        sub = df[np.isclose(df[view.fixed_col], view.fixed_value)].copy()
        for row_idx, (metric, ylabel) in enumerate(
            [
                ("direct_alignment_d_mean", "D(s)"),
                ("direct_delta_d_mean", "D(s) - D(0)"),
            ]
        ):
            ax = axes[row_idx, col]
            for sf_group in ["high_sf", "low_sf"]:
                for run_mode, (linestyle, linewidth, run_label) in styles.items():
                    line = sub[(sub["sf_group"] == sf_group) & (sub["run_mode"] == run_mode)].sort_values(view.xcol)
                    if line.empty:
                        continue
                    x = line[view.xcol].to_numpy(dtype=float)
                    y = line[metric].to_numpy(dtype=float)
                    label = f"{SF_LABEL[sf_group]}, {run_label}" if col == 0 and row_idx == 0 else None
                    ax.plot(
                        x,
                        y,
                        linestyle=linestyle,
                        marker="o" if run_mode == "original_plus_rot90" else None,
                        color=SF_COLORS[sf_group],
                        alpha=alpha[run_mode],
                        linewidth=linewidth,
                        label=label,
                    )
            ax.axhline(0, color="0.55", lw=1, ls="--")
            ax.set_title(view.title)
            ax.set_xlabel(view.xcol.replace("_", " "))
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.grid(True, color="0.9", linewidth=0.8)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Direct unit-first alignment by run, with collapsed estimate", fontsize=14)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"backimage_rr100_unit_first_direct_alignment_by_run_{grouping_label}.{ext}", dpi=dpi)
    plt.close(fig)


def plot_robustness(robust: pd.DataFrame, loo: pd.DataFrame, out_dir: Path, dpi: int, grouping_label: str) -> None:
    df = robust[robust["grouping_label"] == grouping_label].copy()
    loo_df = loo[loo["grouping_label"] == grouping_label].copy()
    metrics = [
        ("low_sf_unit_first_gain_orig_rot90_mean", "low SF gain\n3x-0x"),
        ("high_sf_static_c0", "high SF\nC(0)"),
        ("high_sf_delta_c_pure_across_1x", "high SF\nDelta C 1x"),
        ("high_sf_static_harmonic_amplitude_original", "absolute-axis\nharmonic amp."),
    ]
    modes = [
        "RR100 equal-unit",
        "leave-one-out range",
        "without u054",
        "without u018",
        "without u054+u018",
        "RR100 cluster-size weighted",
        "full756 direct",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), sharey=False, constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics, strict=False):
        y_positions = np.arange(len(modes))
        ax.axvline(0, color="0.75", lw=1)
        for yi, mode in enumerate(modes):
            if mode == "leave-one-out range":
                vals = loo_df[metric].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    mid = float(np.nanmedian(vals))
                    lo = float(np.nanmin(vals))
                    hi = float(np.nanmax(vals))
                    ax.hlines(yi, lo, hi, color="#555555", lw=2)
                    ax.plot(mid, yi, "o", color="#555555")
                continue
            row = df[df["mode"] == mode]
            if row.empty:
                continue
            value = pd.to_numeric(row.iloc[0][metric], errors="coerce")
            if np.isfinite(value):
                ax.plot(float(value), yi, "o", color="#222222")
            else:
                ax.text(0.02, yi, "n/a", transform=ax.get_yaxis_transform(), fontsize=8, va="center", color="0.45")
        ax.set_title(title)
        ax.set_xlabel("bits/spike")
        ax.grid(True, axis="x", color="0.9", linewidth=0.8)
        ax.set_yticks(y_positions, labels=modes if ax is axes[0] else [])
        ax.invert_yaxis()
    fig.suptitle("Robustness of key estimates", fontsize=14)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"backimage_rr100_unit_first_robustness_{grouping_label}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_stats = load_cache(Path(args.original_run_dir))
    rot_stats = load_cache(Path(args.rot90_run_dir))
    if not np.array_equal(orig_stats["movie_source_row"], rot_stats["movie_source_row"]):
        raise ValueError("Original and rot90 movie_source_row orders do not match.")
    if not np.array_equal(orig_stats["condition_id"].astype(str), rot_stats["condition_id"].astype(str)):
        raise ValueError("Original and rot90 condition orders do not match.")

    conditions = condition_frame(orig_stats)
    orig_movies = load_movie_frame(Path(args.original_run_dir), orig_stats, str(args.axis_coordinate_frame))
    rot_movies = load_movie_frame(Path(args.rot90_run_dir), rot_stats, str(args.axis_coordinate_frame))
    orig_bits, orig_spikes = metric_arrays(orig_stats, str(args.metric))
    rot_bits, rot_spikes = metric_arrays(rot_stats, str(args.metric))

    n_units = int(orig_bits.shape[2])
    primary_units = load_grouped_units(
        Path(args.primary_sf_csv),
        n_units=n_units,
        grouping_label=str(args.primary_grouping_label),
    )
    static_units = load_grouped_units(
        Path(args.static_sf_csv),
        n_units=n_units,
        grouping_label=str(args.static_grouping_label),
        derive_from_column=str(args.static_sf_column),
        low_threshold=float(args.static_low_threshold),
        high_threshold=float(args.static_high_threshold),
    )
    grouping_specs = [(str(args.primary_grouping_label), primary_units), (str(args.static_grouping_label), static_units)]

    view = load_population_view(version_name=str(args.rr100_version))
    cluster_weights = (np.asarray(view.cluster_membership) > 0).sum(axis=1).astype(np.float64)
    if cluster_weights.shape[0] != n_units:
        raise ValueError(f"RR100 cluster weights shape {cluster_weights.shape} does not match n_units={n_units}.")

    rng = np.random.default_rng(int(args.bootstrap_seed))
    all_motion: list[pd.DataFrame] = []
    all_motion_units: list[pd.DataFrame] = []
    all_motion_endpoints: list[pd.DataFrame] = []
    all_align: list[pd.DataFrame] = []
    all_align_units: list[pd.DataFrame] = []
    all_direct_align: list[pd.DataFrame] = []
    all_direct_align_units: list[pd.DataFrame] = []
    all_abs_orientation: list[pd.DataFrame] = []
    all_robust: list[pd.DataFrame] = []
    all_loo: list[pd.DataFrame] = []
    group_count_rows: list[dict[str, Any]] = []

    for grouping_label, units in grouping_specs:
        masks = selection_masks(
            orig_movies,
            units,
            SF_ORDER,
            alignment_angle_deg=float(args.alignment_angle_deg),
            min_orientation_selectivity=float(args.min_orientation_selectivity),
        )
        rot_masks = selection_masks(
            rot_movies,
            units,
            SF_ORDER,
            alignment_angle_deg=float(args.alignment_angle_deg),
            min_orientation_selectivity=float(args.min_orientation_selectivity),
        )
        for sf_group in SF_ORDER:
            sub = units[units["sf_group"] == sf_group]
            group_count_rows.append(
                {
                    "grouping_label": grouping_label,
                    "sf_group": sf_group,
                    "n_units": int(sub.shape[0]),
                    "sf_min": float(pd.to_numeric(sub["sf_split_metric"], errors="coerce").min()) if not sub.empty else float("nan"),
                    "sf_median": float(pd.to_numeric(sub["sf_split_metric"], errors="coerce").median()) if not sub.empty else float("nan"),
                    "sf_max": float(pd.to_numeric(sub["sf_split_metric"], errors="coerce").max()) if not sub.empty else float("nan"),
                }
            )
        for run_label, bits in [("original", orig_bits), ("rot90", rot_bits)]:
            motion, motion_units, endpoints = summarize_unit_first_motion_gain(
                run_label=run_label,
                bits=bits,
                conditions=conditions,
                units=units,
                grouping_label=grouping_label,
                n_boot=int(args.n_bootstrap),
                rng=rng,
            )
            all_motion.append(motion)
            all_motion_units.append(motion_units)
            all_motion_endpoints.append(endpoints)
        align, align_units = summarize_unit_first_alignment_crossover(
            conditions=conditions,
            units=units,
            grouping_label=grouping_label,
            orig_bits=orig_bits,
            rot_bits=rot_bits,
            masks=masks,
            n_boot=int(args.n_bootstrap),
            rng=rng,
        )
        all_align.append(align)
        all_align_units.append(align_units)
        direct_align, direct_align_units = summarize_unit_first_direct_alignment(
            conditions=conditions,
            units=units,
            grouping_label=grouping_label,
            orig_bits=orig_bits,
            rot_bits=rot_bits,
            orig_masks=masks,
            rot_masks=rot_masks,
            n_boot=int(args.n_bootstrap),
            rng=rng,
        )
        all_direct_align.append(direct_align)
        all_direct_align_units.append(direct_align_units)
        all_abs_orientation.append(
            absolute_orientation_table(
                grouping_label=grouping_label,
                units=units,
                orig_bits=orig_bits,
                orig_spikes=orig_spikes,
                rot_bits=rot_bits,
                rot_spikes=rot_spikes,
                conditions=conditions,
            )
        )
        robust, loo = build_robustness_tables(
            grouping_label=grouping_label,
            conditions=conditions,
            movies=orig_movies,
            units=units,
            orig_bits=orig_bits,
            rot_bits=rot_bits,
            masks=masks,
            cluster_weights=cluster_weights,
        )
        all_robust.append(robust)
        all_loo.append(loo)

    motion_df = pd.concat(all_motion, ignore_index=True)
    motion_units_df = pd.concat(all_motion_units, ignore_index=True)
    endpoints_df = pd.concat(all_motion_endpoints, ignore_index=True)
    align_df = pd.concat(all_align, ignore_index=True)
    align_units_df = pd.concat(all_align_units, ignore_index=True)
    direct_align_df = pd.concat(all_direct_align, ignore_index=True)
    direct_align_units_df = pd.concat(all_direct_align_units, ignore_index=True)
    abs_df = pd.concat(all_abs_orientation, ignore_index=True)
    robust_df = pd.concat(all_robust, ignore_index=True)
    loo_df = pd.concat(all_loo, ignore_index=True)
    group_counts_df = pd.DataFrame(group_count_rows)

    write_csv(out_dir / "sf_group_counts.csv", group_counts_df)
    write_csv(out_dir / "unit_first_motion_gain_summary.csv", motion_df)
    write_csv(out_dir / "unit_first_motion_gain_per_unit.csv", motion_units_df)
    write_csv(out_dir / "unit_first_motion_gain_endpoint_diagnostics.csv", endpoints_df)
    write_csv(out_dir / "unit_first_alignment_crossover_summary.csv", align_df)
    write_csv(out_dir / "unit_first_alignment_crossover_per_unit.csv", align_units_df)
    write_csv(out_dir / "unit_first_direct_alignment_summary.csv", direct_align_df)
    write_csv(out_dir / "unit_first_direct_alignment_per_unit.csv", direct_align_units_df)
    write_csv(out_dir / "high_sf_absolute_orientation_static_ssi.csv", abs_df)
    write_csv(out_dir / "key_estimate_robustness_summary.csv", robust_df)
    write_csv(out_dir / "key_estimate_leave_one_out.csv", loo_df)

    for grouping_label, _ in grouping_specs:
        plot_motion_gain(motion_df, out_dir, int(args.dpi), grouping_label)
        plot_absolute_orientation(abs_df, out_dir, int(args.dpi), grouping_label)
        plot_alignment_crossover(align_df, out_dir, int(args.dpi), grouping_label)
        plot_direct_alignment_collapsed(direct_align_df, out_dir, int(args.dpi), grouping_label)
        plot_direct_alignment_by_run(direct_align_df, out_dir, int(args.dpi), grouping_label)
        plot_robustness(robust_df, loo_df, out_dir, int(args.dpi), grouping_label)

    write_json(
        out_dir / "summary.json",
        {
            "analysis": "backimage_rr100_unit_first_primary_results",
            "original_run_dir": Path(args.original_run_dir),
            "rot90_run_dir": Path(args.rot90_run_dir),
            "primary_sf_csv": Path(args.primary_sf_csv),
            "static_sf_csv": Path(args.static_sf_csv),
            "metric": str(args.metric),
            "alignment_angle_deg": float(args.alignment_angle_deg),
            "min_orientation_selectivity": float(args.min_orientation_selectivity),
            "axis_coordinate_frame": str(args.axis_coordinate_frame),
            "n_bootstrap": int(args.n_bootstrap),
            "bootstrap_seed": int(args.bootstrap_seed),
            "n_conditions": int(conditions.shape[0]),
            "n_fixations": int(orig_bits.shape[1]),
            "n_units": n_units,
            "sf_group_counts": group_counts_df.to_dict(orient="records"),
            "outputs": {
                "sf_group_counts": out_dir / "sf_group_counts.csv",
                "unit_first_motion_gain_summary": out_dir / "unit_first_motion_gain_summary.csv",
                "unit_first_motion_gain_endpoint_diagnostics": out_dir / "unit_first_motion_gain_endpoint_diagnostics.csv",
                "unit_first_alignment_crossover_summary": out_dir / "unit_first_alignment_crossover_summary.csv",
                "unit_first_direct_alignment_summary": out_dir / "unit_first_direct_alignment_summary.csv",
                "high_sf_absolute_orientation_static_ssi": out_dir / "high_sf_absolute_orientation_static_ssi.csv",
                "key_estimate_robustness_summary": out_dir / "key_estimate_robustness_summary.csv",
                "key_estimate_leave_one_out": out_dir / "key_estimate_leave_one_out.csv",
            },
            "contracts": {
                "motion_gain": "g_{u,f}(s)=I_{u,f}(s)-I_{u,f}(0); average fixations inside unit, then average units.",
                "alignment_crossover": (
                    "For units aligned to the original contour, d=I_orig-I_rot90; for units orthogonal to the "
                    "original contour, d=I_rot90-I_orig. C(s) averages d within unit before averaging units; "
                    "Delta C subtracts each unit's static d before averaging."
                ),
                "direct_alignment": (
                    "D(s) is computed within each run by averaging SSI over fixations where a unit is contour-aligned "
                    "minus fixations where the same unit is contour-orthogonal, then averaging units. The "
                    "original_plus_rot90 mode collapses original and rot90 samples before taking the per-unit difference."
                ),
                "bootstrap": "Confidence intervals resample paired source windows/fixations with replacement.",
                "full756": "Direct full756 rows are placeholders because the completed contour caches are RR100 only.",
            },
        },
    )
    print(f"Wrote unit-first primary results to {out_dir}")


if __name__ == "__main__":
    main()
