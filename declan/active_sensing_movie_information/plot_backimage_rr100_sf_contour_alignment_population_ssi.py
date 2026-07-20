#!/usr/bin/env python3
"""Post hoc SF x contour-alignment RR100 population SSI summary.

This script reuses cached BackImage contour-axis RR100 SSI runs.  It does not
rerun the twin.  For each fixation, low/high spatial-frequency groups are split
again by whether each unit's orientation preference is aligned with the local
contour axis or its orthogonal axis.  Population information is then accumulated
across fixations even though the selected units may differ from fixation to
fixation.  The default metric is time-resolved SSI, which is the cached
spike-weighted average of per-frame spatial SSI values.  Mean-map SSI is kept
only as an explicit diagnostic because it computes SSI after averaging activation
maps over the trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTOUR_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_spatial_ssi_mean_map_primary_n128_across_sweep_v1"
)
DEFAULT_SF_GROUPS_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
    "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)
DEFAULT_OUT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_sf_contour_alignment_population_ssi_dynamic_log_gaussian_marginal_low0p05_high0p5_v1"
)
EPS = 1e-12
ORIENTATION_POOLS = ["all_sf_units", "target_oriented", "off_axis_tuned", "low_osi"]
ALIGNMENT_GROUPS = ["contour_aligned", "contour_orthogonal"]
MOVIE_FEATURE_COLUMNS = [
    "balanced_manifest_index",
    "axis_balance_deg",
    "axis_balance_bin",
    "axis_balance_bin_start_deg",
    "axis_balance_bin_stop_deg",
    "energy_balance_column",
    "energy_balance_value",
    "energy_balance_bin",
    "energy_balance_quantile_bins",
    "image_patch_rms_contrast",
    "image_patch_std",
    "image_gradient_energy",
    "image_orientation_coherence",
    "image_oriented_gradient_energy",
    "image_multi_orientation_energy",
    "image_edge_density",
    "image_spectrum_anisotropy",
    "image_abs_8plus_power_proxy",
    "image_oriented_8plus_power_proxy",
    "image_high_freq_power_fraction",
    "image_power_8plus_cpd_fraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contour-run-dir", type=Path, default=DEFAULT_CONTOUR_RUN_DIR)
    parser.add_argument("--sf-groups-csv", type=Path, default=DEFAULT_SF_GROUPS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--ssi-metric",
        choices=("time_resolved", "mean_map"),
        default="time_resolved",
        help=(
            "Which cached unit SSI array to aggregate. time_resolved is the promotable cached metric "
            "here: per-frame spatial SSI spike-weighted over trajectory frames. mean_map is a diagnostic "
            "that computes SSI after trajectory-averaging activation maps."
        ),
    )
    parser.add_argument(
        "--sf-groups",
        type=str,
        default="low_sf,high_sf",
        help="Comma-separated SF groups to include from the tuning table.",
    )
    parser.add_argument(
        "--high-sf-min-cpd",
        type=float,
        default=None,
        help=(
            "Optional override for the high_sf group: units with sf_split_metric at or above this cpd value "
            "are labeled high_sf, while original high_sf units below threshold are excluded from high_sf."
        ),
    )
    parser.add_argument(
        "--alignment-angle-deg",
        type=float,
        default=22.5,
        help="Maximum axial orientation distance from the contour or orthogonal axis.",
    )
    parser.add_argument(
        "--min-orientation-selectivity",
        type=float,
        default=0.05,
        help="Minimum prior_orientation_selectivity_index for a unit to enter either alignment pool.",
    )
    parser.add_argument(
        "--min-units-per-fixation-group",
        type=int,
        default=3,
        help="Per-fixation group rows below this size are retained but excluded from aggregate summaries.",
    )
    parser.add_argument(
        "--axis-coordinate-frame",
        choices=("gaze", "image"),
        default="gaze",
        help="Frame for movie_condition_inventory.axis_deg. image_edge_axis_deg uses gaze.",
    )
    parser.add_argument(
        "--sweep-axis",
        choices=("auto", "across", "along"),
        default="auto",
        help="Which condition scale to use as the plot x-axis. auto preserves the cached sweep behavior.",
    )
    parser.add_argument(
        "--fixed-along-scale",
        type=float,
        default=None,
        help="Keep only conditions with this along-contour scale before summarizing.",
    )
    parser.add_argument(
        "--fixed-across-scale",
        type=float,
        default=None,
        help="Keep only conditions with this across-contour scale before summarizing.",
    )
    parser.add_argument("--reference-condition-id", type=str, default="")
    parser.add_argument("--endpoint-condition-id", type=str, default="")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def movie_feature_payload(movie: Any) -> dict[str, Any]:
    data = movie._asdict() if hasattr(movie, "_asdict") else dict(movie)
    out: dict[str, Any] = {}
    for key in MOVIE_FEATURE_COLUMNS:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            out[key] = value if math.isfinite(value) else None
        else:
            out[key] = value
    return out


def sem(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def weighted_mean(values: np.ndarray | pd.Series, weights: np.ndarray | pd.Series) -> float:
    value_arr = np.asarray(values, dtype=np.float64)
    weight_arr = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(value_arr) & np.isfinite(weight_arr) & (weight_arr > 0.0)
    if not ok.any():
        return float("nan")
    return float(np.average(value_arr[ok], weights=weight_arr[ok]))


def effective_n_from_weights(weights: np.ndarray | pd.Series) -> float:
    arr = np.asarray(weights, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return 0.0
    numerator = float(np.sum(arr) ** 2)
    denominator = float(np.sum(arr * arr))
    return numerator / max(denominator, EPS)


def orientation_axis_180(angle_deg: float | np.ndarray) -> np.ndarray:
    return np.asarray(angle_deg, dtype=np.float64) % 180.0


def angle_180_distance(a_deg: float | np.ndarray, b_deg: float | np.ndarray) -> np.ndarray:
    return np.abs(((np.asarray(a_deg, dtype=np.float64) - np.asarray(b_deg, dtype=np.float64) + 90.0) % 180.0) - 90.0)


def axial_alignment_score(preferred_deg: np.ndarray, contour_axis_deg: float) -> np.ndarray:
    """+1 means bar axis matches contour, -1 means bar axis is orthogonal."""
    delta = angle_180_distance(preferred_deg, contour_axis_deg)
    return np.cos(np.deg2rad(2.0 * delta))


def contour_axis_to_image_frame(axis_deg: float | np.ndarray, coordinate_frame: str) -> np.ndarray:
    axis = np.asarray(axis_deg, dtype=np.float64)
    if str(coordinate_frame) == "gaze":
        axis = -axis
    return orientation_axis_180(axis)


def load_npz_without_identity(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files if key != "cache_identity_json"}


def cache_path(run_dir: Path) -> Path:
    return Path(run_dir) / "cache" / "backimage_contour_axis_rr100_spatial_ssi_cache.npz"


def metric_array(stats: dict[str, np.ndarray], metric: str) -> tuple[np.ndarray, str]:
    if metric == "mean_map":
        return np.asarray(stats["unit_mean_map_bits_per_movie"], dtype=np.float64), "unit_mean_map_bits_per_movie"
    if metric == "time_resolved":
        return np.asarray(stats["unit_time_resolved_bits_per_movie"], dtype=np.float64), "unit_time_resolved_bits_per_movie"
    raise ValueError(f"Unknown metric {metric!r}")


def metric_label(metric: str) -> str:
    return {
        "time_resolved": "spike-weighted time-resolved SSI",
        "mean_map": "mean-map SSI diagnostic",
    }.get(str(metric), str(metric))


def metric_contract(metric: str) -> str:
    if str(metric) == "time_resolved":
        return (
            "For each unit/movie/condition, spatial SSI is computed per trajectory frame and averaged with "
            "expected spikes as frame weights. Population pools then sum unit information numerators and "
            "expected-spike denominators before taking bits/spike."
        )
    if str(metric) == "mean_map":
        return (
            "Diagnostic only: SSI is computed after averaging activation maps over trajectory frames. "
            "This should not be used as the promoted motion-scale SSI metric."
        )
    return ""


def condition_frame(stats: dict[str, np.ndarray]) -> pd.DataFrame:
    condition_id = np.asarray(stats["condition_id"]).astype(str)
    n_conditions = int(condition_id.size)
    along = np.asarray(stats["along_scale"], dtype=np.float64)
    across = np.asarray(stats["across_scale"], dtype=np.float64)
    motion = np.asarray(stats.get("motion_scale", across), dtype=np.float64)
    motion_fallback = np.maximum(np.abs(along), np.abs(across))
    motion = np.where(np.isfinite(motion), motion, motion_fallback)
    sweep_mode = np.asarray(stats.get("sweep_mode", np.asarray(["across"] * n_conditions))).astype(str)
    return pd.DataFrame(
        {
            "condition_index": np.arange(n_conditions, dtype=int),
            "condition_id": condition_id,
            "condition_label": np.asarray(stats["condition_label"]).astype(str),
            "along_scale": along,
            "across_scale": across,
            "motion_scale": motion,
            "sweep_mode": sweep_mode,
            "is_static_baseline": np.asarray(stats["is_static_baseline"], dtype=bool),
            "is_across_sweep": np.asarray(stats["is_across_sweep"], dtype=bool),
        }
    )


def load_movie_frame(run_dir: Path, stats: dict[str, np.ndarray], axis_coordinate_frame: str) -> pd.DataFrame:
    inventory_path = Path(run_dir) / "movie_condition_inventory.csv"
    if not inventory_path.exists():
        raise FileNotFoundError(inventory_path)
    inventory = pd.read_csv(inventory_path)
    required = {"movie_index", "trial_id", "source_row", "session", "trial_idx", "axis_deg"}
    missing = sorted(required.difference(inventory.columns))
    if missing:
        raise ValueError(f"Missing required columns in {inventory_path}: {missing}")
    movies = inventory.sort_values(["movie_index", "condition_index"]).drop_duplicates("movie_index", keep="first").copy()
    n_movies = int(np.asarray(stats["movie_source_row"]).size)
    movies = movies.set_index("movie_index", drop=False).reindex(np.arange(n_movies))
    if movies["source_row"].isna().any():
        raise ValueError("movie_condition_inventory.csv is missing at least one cache movie_index.")
    cache_source = np.asarray(stats["movie_source_row"], dtype=int)
    inv_source = movies["source_row"].to_numpy(dtype=int)
    if not np.array_equal(cache_source, inv_source):
        raise ValueError("Cache movie_source_row order does not match movie_condition_inventory.csv.")
    movies["axis_deg"] = pd.to_numeric(movies["axis_deg"], errors="coerce")
    movies["contour_axis_image_deg"] = contour_axis_to_image_frame(
        movies["axis_deg"].to_numpy(dtype=float),
        axis_coordinate_frame,
    )
    movies["orthogonal_axis_image_deg"] = orientation_axis_180(movies["contour_axis_image_deg"].to_numpy(dtype=float) + 90.0)
    movies["axis_coordinate_frame"] = axis_coordinate_frame
    return movies.reset_index(drop=True)


def apply_sf_threshold_overrides(units: pd.DataFrame, *, high_sf_min_cpd: float | None) -> pd.DataFrame:
    units = units.copy()
    units["original_sf_group"] = units["sf_group"].astype(str)
    units["sf_group_definition_mode"] = "source_table"
    if high_sf_min_cpd is None or not np.isfinite(float(high_sf_min_cpd)):
        return units
    threshold = float(high_sf_min_cpd)
    metric = pd.to_numeric(units["sf_split_metric"], errors="coerce")
    high_mask = metric >= threshold
    original_high = units["original_sf_group"].astype(str) == "high_sf"
    units.loc[high_mask, "sf_group"] = "high_sf"
    units.loc[original_high & ~high_mask, "sf_group"] = "high_sf_below_threshold"
    units["sf_group_definition_mode"] = f"high_sf_min_cpd_{threshold:g}"
    if "sf_group_label" in units.columns:
        n_high = int(high_mask.sum())
        units.loc[high_mask, "sf_group_label"] = f"high SF >= {threshold:g} cpd (n={n_high})"
        units.loc[original_high & ~high_mask, "sf_group_label"] = f"original high SF below {threshold:g} cpd"
    return units


def load_unit_table(path: Path, sf_groups: list[str], n_units: int, *, high_sf_min_cpd: float | None = None) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    units = pd.read_csv(path)
    required = {
        "unit_index",
        "unit_label",
        "sf_group",
        "sf_rank_low_to_high",
        "sf_split_metric",
        "prior_preferred_orientation_deg",
        "prior_orientation_selectivity_index",
    }
    missing = sorted(required.difference(units.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    units = units.copy()
    units["unit_index"] = pd.to_numeric(units["unit_index"], errors="coerce").astype(int)
    units = units[(units["unit_index"] >= 0) & (units["unit_index"] < int(n_units))]
    units["sf_split_metric"] = pd.to_numeric(units["sf_split_metric"], errors="coerce")
    units = apply_sf_threshold_overrides(units, high_sf_min_cpd=high_sf_min_cpd)
    units = units[units["sf_group"].astype(str).isin(sf_groups)].copy()
    if units.empty:
        raise ValueError(f"No units remain after filtering for sf_groups={sf_groups}.")
    units["preferred_orientation_image_deg"] = orientation_axis_180(
        pd.to_numeric(units["prior_preferred_orientation_deg"], errors="coerce").to_numpy(dtype=float)
    )
    units["prior_orientation_selectivity_index"] = pd.to_numeric(
        units["prior_orientation_selectivity_index"],
        errors="coerce",
    )
    return units.sort_values(["sf_group", "sf_rank_low_to_high", "unit_index"]).reset_index(drop=True)


def sf_group_definition_rows(path: Path, n_units: int, *, high_sf_min_cpd: float | None = None) -> list[dict[str, Any]]:
    units = pd.read_csv(path)
    required = {"unit_index", "sf_group", "sf_rank_low_to_high", "sf_split_metric", "sf_split_metric_name", "sf_split_metric_column"}
    missing = sorted(required.difference(units.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    units = units.copy()
    units["unit_index"] = pd.to_numeric(units["unit_index"], errors="coerce").astype(int)
    units = units[(units["unit_index"] >= 0) & (units["unit_index"] < int(n_units))]
    units["sf_rank_low_to_high"] = pd.to_numeric(units["sf_rank_low_to_high"], errors="coerce")
    units["sf_split_metric"] = pd.to_numeric(units["sf_split_metric"], errors="coerce")
    units = units[np.isfinite(units["sf_split_metric"].to_numpy(dtype=float))].copy()
    units = apply_sf_threshold_overrides(units, high_sf_min_cpd=high_sf_min_cpd)
    sorted_units = units.sort_values(["sf_split_metric", "unit_index"], ascending=[True, True]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for sf_group, sub in sorted_units.groupby("sf_group", sort=True):
        sub = sub.sort_values(["sf_split_metric", "unit_index"], ascending=[True, True])
        first_pos = int(sub.index.min())
        last_pos = int(sub.index.max())
        below = sorted_units.iloc[first_pos - 1] if first_pos > 0 else None
        above = sorted_units.iloc[last_pos + 1] if last_pos + 1 < len(sorted_units) else None
        threshold_from_below = (
            float((float(below["sf_split_metric"]) + float(sub.iloc[0]["sf_split_metric"])) / 2.0)
            if below is not None
            else float("nan")
        )
        threshold_to_above = (
            float((float(sub.iloc[-1]["sf_split_metric"]) + float(above["sf_split_metric"])) / 2.0)
            if above is not None
            else float("nan")
        )
        rows.append(
            {
                "sf_group": str(sf_group),
                "n_units": int(sub.shape[0]),
                "rank_min_low_to_high": int(np.nanmin(sub["sf_rank_low_to_high"].to_numpy(dtype=float))),
                "rank_max_low_to_high": int(np.nanmax(sub["sf_rank_low_to_high"].to_numpy(dtype=float))),
                "sf_split_metric_name": str(sub["sf_split_metric_name"].iloc[0]),
                "sf_split_metric_column": str(sub["sf_split_metric_column"].iloc[0]),
                "sf_group_definition_mode": str(sub["sf_group_definition_mode"].iloc[0]),
                "sf_split_metric_min": float(np.nanmin(sub["sf_split_metric"].to_numpy(dtype=float))),
                "sf_split_metric_max": float(np.nanmax(sub["sf_split_metric"].to_numpy(dtype=float))),
                "sf_split_metric_median": float(np.nanmedian(sub["sf_split_metric"].to_numpy(dtype=float))),
                "effective_lower_boundary_midpoint": threshold_from_below,
                "effective_upper_boundary_midpoint": threshold_to_above,
                "neighbor_below_sf_group": None if below is None else str(below["sf_group"]),
                "neighbor_below_sf_split_metric": None if below is None else float(below["sf_split_metric"]),
                "neighbor_above_sf_group": None if above is None else str(above["sf_group"]),
                "neighbor_above_sf_split_metric": None if above is None else float(above["sf_split_metric"]),
            }
        )
    return rows


def unit_mask_for_movie(
    units: pd.DataFrame,
    *,
    movie: pd.Series,
    sf_group: str,
    alignment_group: str,
    alignment_angle_deg: float,
    min_orientation_selectivity: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    sub = units[units["sf_group"].astype(str) == str(sf_group)].copy()
    if sub.empty:
        return np.zeros(0, dtype=int), sub
    pref = sub["preferred_orientation_image_deg"].to_numpy(dtype=float)
    contour = float(movie["contour_axis_image_deg"])
    orthogonal = float(movie["orthogonal_axis_image_deg"])
    sub["orientation_delta_from_contour_deg"] = angle_180_distance(pref, contour)
    sub["orientation_delta_from_orthogonal_deg"] = angle_180_distance(pref, orthogonal)
    valid = (
        np.isfinite(sub["preferred_orientation_image_deg"].to_numpy(dtype=float))
        & np.isfinite(sub["prior_orientation_selectivity_index"].to_numpy(dtype=float))
        & (sub["prior_orientation_selectivity_index"].to_numpy(dtype=float) >= float(min_orientation_selectivity))
    )
    if alignment_group == "contour_aligned":
        keep = valid & (sub["orientation_delta_from_contour_deg"].to_numpy(dtype=float) <= float(alignment_angle_deg))
    elif alignment_group == "contour_orthogonal":
        keep = valid & (sub["orientation_delta_from_orthogonal_deg"].to_numpy(dtype=float) <= float(alignment_angle_deg))
    else:
        raise ValueError(f"Unknown alignment_group {alignment_group!r}")
    kept = sub.loc[keep].copy()
    return kept["unit_index"].to_numpy(dtype=int), kept


def orientation_pool_masks_for_movie(
    units: pd.DataFrame,
    *,
    movie: pd.Series,
    sf_group: str,
    alignment_angle_deg: float,
    min_orientation_selectivity: float,
) -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
    sub = units[units["sf_group"].astype(str) == str(sf_group)].copy()
    if sub.empty:
        return {pool: (np.zeros(0, dtype=int), sub.copy()) for pool in ORIENTATION_POOLS}
    pref = sub["preferred_orientation_image_deg"].to_numpy(dtype=float)
    contour = float(movie["contour_axis_image_deg"])
    orthogonal = float(movie["orthogonal_axis_image_deg"])
    sub["orientation_delta_from_contour_deg"] = angle_180_distance(pref, contour)
    sub["orientation_delta_from_orthogonal_deg"] = angle_180_distance(pref, orthogonal)
    sub["orientation_delta_from_nearest_target_deg"] = np.minimum(
        sub["orientation_delta_from_contour_deg"].to_numpy(dtype=float),
        sub["orientation_delta_from_orthogonal_deg"].to_numpy(dtype=float),
    )
    osi = sub["prior_orientation_selectivity_index"].to_numpy(dtype=float)
    pref_finite = np.isfinite(sub["preferred_orientation_image_deg"].to_numpy(dtype=float))
    osi_finite = np.isfinite(osi)
    target_geometry = (
        (sub["orientation_delta_from_contour_deg"].to_numpy(dtype=float) <= float(alignment_angle_deg))
        | (sub["orientation_delta_from_orthogonal_deg"].to_numpy(dtype=float) <= float(alignment_angle_deg))
    )
    orientation_tuned = pref_finite & osi_finite & (osi >= float(min_orientation_selectivity))
    target_oriented = orientation_tuned & target_geometry
    low_osi = osi_finite & (osi < float(min_orientation_selectivity))
    pool_masks = {
        "all_sf_units": np.ones(len(sub), dtype=bool),
        "target_oriented": target_oriented,
        "off_axis_tuned": orientation_tuned & ~target_geometry,
        "low_osi": low_osi,
    }
    return {
        pool: (sub.loc[mask, "unit_index"].to_numpy(dtype=int), sub.loc[mask].copy())
        for pool, mask in pool_masks.items()
    }


def unit_list_text(values: np.ndarray) -> str:
    return " ".join(f"u{int(v):03d}" for v in np.asarray(values, dtype=int))


def weighted_unit_list_text(unit_indices: np.ndarray, weights: np.ndarray) -> str:
    if len(unit_indices) == 0:
        return ""
    return " ".join(
        f"u{int(unit_idx):03d}:{float(weight):.3f}"
        for unit_idx, weight in zip(np.asarray(unit_indices, dtype=int), np.asarray(weights, dtype=float), strict=True)
    )


def weighted_alignment_pools_for_movie(
    units: pd.DataFrame,
    *,
    movie: pd.Series,
    sf_group: str,
    min_orientation_selectivity: float,
    min_units_per_fixation_group: int,
) -> dict[str, dict[str, Any]]:
    sub = units[units["sf_group"].astype(str) == str(sf_group)].copy()
    empty: dict[str, dict[str, Any]] = {}
    if sub.empty:
        for alignment_group in ALIGNMENT_GROUPS:
            empty[alignment_group] = {
                "unit_indices": np.zeros(0, dtype=int),
                "weights": np.zeros(0, dtype=np.float64),
                "kept": sub.copy(),
                "n_sf_units_total": 0,
                "n_orientation_tuned_units": 0,
                "usable_by_min_units": False,
            }
        return empty

    pref = sub["preferred_orientation_image_deg"].to_numpy(dtype=float)
    contour = float(movie["contour_axis_image_deg"])
    orthogonal = float(movie["orthogonal_axis_image_deg"])
    sub["orientation_delta_from_contour_deg"] = angle_180_distance(pref, contour)
    sub["orientation_delta_from_orthogonal_deg"] = angle_180_distance(pref, orthogonal)
    osi = sub["prior_orientation_selectivity_index"].to_numpy(dtype=float)
    valid = (
        np.isfinite(pref)
        & np.isfinite(osi)
        & (osi >= float(min_orientation_selectivity))
    )
    signed_alignment = np.zeros(len(sub), dtype=np.float64)
    signed_alignment[valid] = axial_alignment_score(pref[valid], contour)
    pool_weights = {
        "contour_aligned": np.maximum(signed_alignment, 0.0),
        "contour_orthogonal": np.maximum(-signed_alignment, 0.0),
    }
    out: dict[str, dict[str, Any]] = {}
    n_tuned = int(valid.sum())
    usable_base = bool(n_tuned >= int(min_units_per_fixation_group))
    for alignment_group in ALIGNMENT_GROUPS:
        weights = pool_weights[alignment_group]
        positive = np.isfinite(weights) & (weights > EPS)
        kept = sub.loc[positive].copy()
        kept["alignment_weight"] = weights[positive]
        unit_indices = kept["unit_index"].to_numpy(dtype=int)
        kept_weights = kept["alignment_weight"].to_numpy(dtype=np.float64)
        out[alignment_group] = {
            "unit_indices": unit_indices,
            "weights": kept_weights,
            "kept": kept,
            "n_sf_units_total": int(len(sub)),
            "n_orientation_tuned_units": n_tuned,
            "usable_by_min_units": bool(usable_base and float(np.sum(kept_weights)) > EPS),
        }
    return out


def build_selection_table(
    units: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    sf_groups: list[str],
    alignment_angle_deg: float,
    min_orientation_selectivity: float,
    min_units_per_fixation_group: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str, str], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    masks: dict[tuple[int, str, str], np.ndarray] = {}
    for movie in movies.itertuples(index=False):
        movie_series = pd.Series(movie._asdict())
        for sf_group in sf_groups:
            for alignment_group in ["contour_aligned", "contour_orthogonal"]:
                unit_indices, kept = unit_mask_for_movie(
                    units,
                    movie=movie_series,
                    sf_group=sf_group,
                    alignment_group=alignment_group,
                    alignment_angle_deg=alignment_angle_deg,
                    min_orientation_selectivity=min_orientation_selectivity,
                )
                masks[(int(movie.movie_index), str(sf_group), str(alignment_group))] = unit_indices
                delta_col = (
                    "orientation_delta_from_contour_deg"
                    if alignment_group == "contour_aligned"
                    else "orientation_delta_from_orthogonal_deg"
                )
                rows.append(
                    {
                        "movie_index": int(movie.movie_index),
                        "trial_id": int(movie.trial_id),
                        "source_row": int(movie.source_row),
                        "session": str(movie.session),
                        "trial_idx": int(movie.trial_idx),
                        "axis_deg": float(movie.axis_deg),
                        "axis_coordinate_frame": str(movie.axis_coordinate_frame),
                        "contour_axis_image_deg": float(movie.contour_axis_image_deg),
                        "orthogonal_axis_image_deg": float(movie.orthogonal_axis_image_deg),
                        "sf_group": str(sf_group),
                        "alignment_group": str(alignment_group),
                        "n_units": int(unit_indices.size),
                        "usable_by_min_units": bool(unit_indices.size >= int(min_units_per_fixation_group)),
                        "mean_orientation_target_delta_deg": (
                            float(np.nanmean(kept[delta_col].to_numpy(dtype=float))) if not kept.empty else float("nan")
                        ),
                        "median_orientation_target_delta_deg": (
                            float(np.nanmedian(kept[delta_col].to_numpy(dtype=float))) if not kept.empty else float("nan")
                        ),
                        "mean_orientation_selectivity_index": (
                            float(np.nanmean(kept["prior_orientation_selectivity_index"].to_numpy(dtype=float)))
                            if not kept.empty
                            else float("nan")
                        ),
                        "mean_sf_split_metric": (
                            float(np.nanmean(kept["sf_split_metric"].to_numpy(dtype=float))) if not kept.empty else float("nan")
                        ),
                        "unit_indices": unit_list_text(unit_indices),
                    }
                )
    return rows, masks


def build_weighted_alignment_table(
    units: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    sf_groups: list[str],
    min_orientation_selectivity: float,
    min_units_per_fixation_group: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pools_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for movie in movies.itertuples(index=False):
        movie_series = pd.Series(movie._asdict())
        for sf_group in sf_groups:
            pools = weighted_alignment_pools_for_movie(
                units,
                movie=movie_series,
                sf_group=sf_group,
                min_orientation_selectivity=min_orientation_selectivity,
                min_units_per_fixation_group=min_units_per_fixation_group,
            )
            for alignment_group in ALIGNMENT_GROUPS:
                pool = pools[alignment_group]
                pools_by_key[(int(movie.movie_index), str(sf_group), str(alignment_group))] = pool
                kept = pool["kept"]
                weights = np.asarray(pool["weights"], dtype=np.float64)
                delta_col = (
                    "orientation_delta_from_contour_deg"
                    if alignment_group == "contour_aligned"
                    else "orientation_delta_from_orthogonal_deg"
                )
                rows.append(
                    {
                        "movie_index": int(movie.movie_index),
                        "trial_id": int(movie.trial_id),
                        "source_row": int(movie.source_row),
                        "session": str(movie.session),
                        "trial_idx": int(movie.trial_idx),
                        "axis_deg": float(movie.axis_deg),
                        "axis_coordinate_frame": str(movie.axis_coordinate_frame),
                        "contour_axis_image_deg": float(movie.contour_axis_image_deg),
                        "orthogonal_axis_image_deg": float(movie.orthogonal_axis_image_deg),
                        "sf_group": str(sf_group),
                        "alignment_group": str(alignment_group),
                        "alignment_weight_scheme": "positive_halfwave_cos2_pref_minus_contour",
                        "n_sf_units_total": int(pool["n_sf_units_total"]),
                        "n_orientation_tuned_units": int(pool["n_orientation_tuned_units"]),
                        "n_units": int(len(pool["unit_indices"])),
                        "weight_sum": float(np.sum(weights)),
                        "effective_n_units": effective_n_from_weights(weights),
                        "usable_by_min_units": bool(pool["usable_by_min_units"]),
                        "mean_orientation_target_delta_deg": (
                            weighted_mean(kept[delta_col].to_numpy(dtype=float), weights) if not kept.empty else float("nan")
                        ),
                        "median_orientation_target_delta_deg": (
                            float(np.nanmedian(kept[delta_col].to_numpy(dtype=float))) if not kept.empty else float("nan")
                        ),
                        "mean_orientation_selectivity_index": (
                            weighted_mean(kept["prior_orientation_selectivity_index"].to_numpy(dtype=float), weights)
                            if not kept.empty
                            else float("nan")
                        ),
                        "mean_sf_split_metric": (
                            weighted_mean(kept["sf_split_metric"].to_numpy(dtype=float), weights)
                            if not kept.empty
                            else float("nan")
                        ),
                        "unit_indices": unit_list_text(np.asarray(pool["unit_indices"], dtype=int)),
                        "unit_weights": weighted_unit_list_text(
                            np.asarray(pool["unit_indices"], dtype=int),
                            weights,
                        ),
                    }
                )
    return rows, pools_by_key


def build_orientation_pool_selection_table(
    units: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    sf_groups: list[str],
    alignment_angle_deg: float,
    min_orientation_selectivity: float,
    min_units_per_fixation_group: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str, str], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    masks: dict[tuple[int, str, str], np.ndarray] = {}
    for movie in movies.itertuples(index=False):
        movie_series = pd.Series(movie._asdict())
        for sf_group in sf_groups:
            pool_masks = orientation_pool_masks_for_movie(
                units,
                movie=movie_series,
                sf_group=sf_group,
                alignment_angle_deg=alignment_angle_deg,
                min_orientation_selectivity=min_orientation_selectivity,
            )
            for orientation_pool in ORIENTATION_POOLS:
                unit_indices, kept = pool_masks[orientation_pool]
                masks[(int(movie.movie_index), str(sf_group), str(orientation_pool))] = unit_indices
                rows.append(
                    {
                        "movie_index": int(movie.movie_index),
                        "trial_id": int(movie.trial_id),
                        "source_row": int(movie.source_row),
                        "session": str(movie.session),
                        "trial_idx": int(movie.trial_idx),
                        "axis_deg": float(movie.axis_deg),
                        "axis_coordinate_frame": str(movie.axis_coordinate_frame),
                        "contour_axis_image_deg": float(movie.contour_axis_image_deg),
                        "orthogonal_axis_image_deg": float(movie.orthogonal_axis_image_deg),
                        "sf_group": str(sf_group),
                        "orientation_pool": str(orientation_pool),
                        "n_units": int(unit_indices.size),
                        "usable_by_min_units": bool(unit_indices.size >= int(min_units_per_fixation_group)),
                        "mean_orientation_nearest_target_delta_deg": (
                            float(np.nanmean(kept["orientation_delta_from_nearest_target_deg"].to_numpy(dtype=float)))
                            if not kept.empty
                            else float("nan")
                        ),
                        "median_orientation_nearest_target_delta_deg": (
                            float(np.nanmedian(kept["orientation_delta_from_nearest_target_deg"].to_numpy(dtype=float)))
                            if not kept.empty
                            else float("nan")
                        ),
                        "mean_orientation_selectivity_index": (
                            float(np.nanmean(kept["prior_orientation_selectivity_index"].to_numpy(dtype=float)))
                            if not kept.empty
                            else float("nan")
                        ),
                        "mean_sf_split_metric": (
                            float(np.nanmean(kept["sf_split_metric"].to_numpy(dtype=float))) if not kept.empty else float("nan")
                        ),
                        "unit_indices": unit_list_text(unit_indices),
                    }
                )
    return rows, masks


def per_fixation_group_rows(
    stats: dict[str, np.ndarray],
    movies: pd.DataFrame,
    conditions: pd.DataFrame,
    masks: dict[tuple[int, str, str], np.ndarray],
    *,
    sf_groups: list[str],
    ssi_metric: str,
    min_units_per_fixation_group: int,
) -> list[dict[str, Any]]:
    bits, metric_key = metric_array(stats, ssi_metric)
    spikes = np.asarray(stats["unit_expected_spikes_per_movie"], dtype=np.float64)
    rates = np.asarray(stats["unit_mean_rate_per_movie"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for condition in conditions.itertuples(index=False):
        cidx = int(condition.condition_index)
        for movie in movies.itertuples(index=False):
            midx = int(movie.movie_index)
            for sf_group in sf_groups:
                for alignment_group in ["contour_aligned", "contour_orthogonal"]:
                    unit_idx = masks[(midx, str(sf_group), str(alignment_group))]
                    n_units = int(unit_idx.size)
                    usable = bool(n_units >= int(min_units_per_fixation_group))
                    if n_units:
                        unit_spikes = spikes[cidx, midx, unit_idx]
                        unit_bits = bits[cidx, midx, unit_idx]
                        unit_rates = rates[cidx, midx, unit_idx]
                        numerator = float(np.nansum(unit_bits * unit_spikes))
                        denominator = float(np.nansum(unit_spikes))
                        pop_bits = numerator / max(denominator, EPS) if denominator > EPS else float("nan")
                        mean_bits = float(np.nanmean(unit_bits))
                        mean_rate = float(np.nanmean(unit_rates))
                    else:
                        numerator = 0.0
                        denominator = 0.0
                        pop_bits = float("nan")
                        mean_bits = float("nan")
                        mean_rate = float("nan")
                    rows.append(
                        {
                            "condition_index": cidx,
                            "condition_id": str(condition.condition_id),
                            "condition_label": str(condition.condition_label),
                            "along_scale": float(condition.along_scale),
                            "across_scale": float(condition.across_scale),
                            "motion_scale": float(condition.motion_scale),
                            "sweep_mode": str(condition.sweep_mode),
                            "is_static_baseline": bool(condition.is_static_baseline),
                            "is_across_sweep": bool(condition.is_across_sweep),
                            "movie_index": midx,
                            "trial_id": int(movie.trial_id),
                            "source_row": int(movie.source_row),
                            "session": str(movie.session),
                            "trial_idx": int(movie.trial_idx),
                            "axis_deg": float(movie.axis_deg),
                            "axis_coordinate_frame": str(movie.axis_coordinate_frame),
                            "contour_axis_image_deg": float(movie.contour_axis_image_deg),
                            **movie_feature_payload(movie),
                            "sf_group": str(sf_group),
                            "alignment_group": str(alignment_group),
                            "n_units": n_units,
                            "usable_by_min_units": usable,
                            "ssi_metric": str(ssi_metric),
                            "ssi_metric_cache_key": metric_key,
                            "information_numerator_bits_arbitrary_dt": numerator,
                            "expected_spikes_arbitrary_dt": denominator,
                            "population_bits_per_spike": pop_bits,
                            "mean_unit_bits_per_spike": mean_bits,
                            "mean_unit_rate": mean_rate,
                            "unit_indices": unit_list_text(unit_idx),
                        }
                    )
    return rows


def per_fixation_weighted_alignment_rows(
    stats: dict[str, np.ndarray],
    movies: pd.DataFrame,
    conditions: pd.DataFrame,
    pools: dict[tuple[int, str, str], dict[str, Any]],
    *,
    sf_groups: list[str],
    ssi_metric: str,
) -> list[dict[str, Any]]:
    bits, metric_key = metric_array(stats, ssi_metric)
    spikes = np.asarray(stats["unit_expected_spikes_per_movie"], dtype=np.float64)
    rates = np.asarray(stats["unit_mean_rate_per_movie"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for condition in conditions.itertuples(index=False):
        cidx = int(condition.condition_index)
        for movie in movies.itertuples(index=False):
            midx = int(movie.movie_index)
            for sf_group in sf_groups:
                for alignment_group in ALIGNMENT_GROUPS:
                    pool = pools[(midx, str(sf_group), str(alignment_group))]
                    unit_idx = np.asarray(pool["unit_indices"], dtype=int)
                    weights = np.asarray(pool["weights"], dtype=np.float64)
                    n_units = int(unit_idx.size)
                    usable = bool(pool["usable_by_min_units"])
                    weight_sum = float(np.sum(weights))
                    effective_n = effective_n_from_weights(weights)
                    if n_units and weight_sum > EPS:
                        unit_spikes = spikes[cidx, midx, unit_idx]
                        unit_bits = bits[cidx, midx, unit_idx]
                        unit_rates = rates[cidx, midx, unit_idx]
                        weighted_spikes = unit_spikes * weights
                        numerator = float(np.nansum(unit_bits * weighted_spikes))
                        denominator = float(np.nansum(weighted_spikes))
                        pop_bits = numerator / max(denominator, EPS) if denominator > EPS else float("nan")
                        mean_bits = weighted_mean(unit_bits, weights)
                        mean_rate = weighted_mean(unit_rates, weights)
                    else:
                        numerator = 0.0
                        denominator = 0.0
                        pop_bits = float("nan")
                        mean_bits = float("nan")
                        mean_rate = float("nan")
                    rows.append(
                        {
                            "condition_index": cidx,
                            "condition_id": str(condition.condition_id),
                            "condition_label": str(condition.condition_label),
                            "along_scale": float(condition.along_scale),
                            "across_scale": float(condition.across_scale),
                            "motion_scale": float(condition.motion_scale),
                            "sweep_mode": str(condition.sweep_mode),
                            "is_static_baseline": bool(condition.is_static_baseline),
                            "is_across_sweep": bool(condition.is_across_sweep),
                            "movie_index": midx,
                            "trial_id": int(movie.trial_id),
                            "source_row": int(movie.source_row),
                            "session": str(movie.session),
                            "trial_idx": int(movie.trial_idx),
                            "axis_deg": float(movie.axis_deg),
                            "axis_coordinate_frame": str(movie.axis_coordinate_frame),
                            "contour_axis_image_deg": float(movie.contour_axis_image_deg),
                            **movie_feature_payload(movie),
                            "sf_group": str(sf_group),
                            "alignment_group": str(alignment_group),
                            "alignment_weight_scheme": "positive_halfwave_cos2_pref_minus_contour",
                            "n_sf_units_total": int(pool["n_sf_units_total"]),
                            "n_orientation_tuned_units": int(pool["n_orientation_tuned_units"]),
                            "n_units": n_units,
                            "weight_sum": weight_sum,
                            "effective_n_units": effective_n,
                            "usable_by_min_units": usable,
                            "ssi_metric": str(ssi_metric),
                            "ssi_metric_cache_key": metric_key,
                            "information_numerator_bits_arbitrary_dt": numerator,
                            "expected_spikes_arbitrary_dt": denominator,
                            "population_bits_per_spike": pop_bits,
                            "mean_unit_bits_per_spike": mean_bits,
                            "mean_unit_rate": mean_rate,
                            "unit_indices": unit_list_text(unit_idx),
                            "unit_weights": weighted_unit_list_text(unit_idx, weights),
                        }
                    )
    return rows


def per_fixation_orientation_pool_rows(
    stats: dict[str, np.ndarray],
    movies: pd.DataFrame,
    conditions: pd.DataFrame,
    masks: dict[tuple[int, str, str], np.ndarray],
    *,
    sf_groups: list[str],
    ssi_metric: str,
    min_units_per_fixation_group: int,
) -> list[dict[str, Any]]:
    bits, metric_key = metric_array(stats, ssi_metric)
    spikes = np.asarray(stats["unit_expected_spikes_per_movie"], dtype=np.float64)
    rates = np.asarray(stats["unit_mean_rate_per_movie"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for condition in conditions.itertuples(index=False):
        cidx = int(condition.condition_index)
        for movie in movies.itertuples(index=False):
            midx = int(movie.movie_index)
            for sf_group in sf_groups:
                for orientation_pool in ORIENTATION_POOLS:
                    unit_idx = masks[(midx, str(sf_group), str(orientation_pool))]
                    n_units = int(unit_idx.size)
                    usable = bool(n_units >= int(min_units_per_fixation_group))
                    if n_units:
                        unit_spikes = spikes[cidx, midx, unit_idx]
                        unit_bits = bits[cidx, midx, unit_idx]
                        unit_rates = rates[cidx, midx, unit_idx]
                        numerator = float(np.nansum(unit_bits * unit_spikes))
                        denominator = float(np.nansum(unit_spikes))
                        pop_bits = numerator / max(denominator, EPS) if denominator > EPS else float("nan")
                        mean_bits = float(np.nanmean(unit_bits))
                        mean_rate = float(np.nanmean(unit_rates))
                    else:
                        numerator = 0.0
                        denominator = 0.0
                        pop_bits = float("nan")
                        mean_bits = float("nan")
                        mean_rate = float("nan")
                    rows.append(
                        {
                            "condition_index": cidx,
                            "condition_id": str(condition.condition_id),
                            "condition_label": str(condition.condition_label),
                            "along_scale": float(condition.along_scale),
                            "across_scale": float(condition.across_scale),
                            "motion_scale": float(condition.motion_scale),
                            "sweep_mode": str(condition.sweep_mode),
                            "is_static_baseline": bool(condition.is_static_baseline),
                            "is_across_sweep": bool(condition.is_across_sweep),
                            "movie_index": midx,
                            "trial_id": int(movie.trial_id),
                            "source_row": int(movie.source_row),
                            "session": str(movie.session),
                            "trial_idx": int(movie.trial_idx),
                            "axis_deg": float(movie.axis_deg),
                            "axis_coordinate_frame": str(movie.axis_coordinate_frame),
                            "contour_axis_image_deg": float(movie.contour_axis_image_deg),
                            **movie_feature_payload(movie),
                            "sf_group": str(sf_group),
                            "orientation_pool": str(orientation_pool),
                            "n_units": n_units,
                            "usable_by_min_units": usable,
                            "ssi_metric": str(ssi_metric),
                            "ssi_metric_cache_key": metric_key,
                            "information_numerator_bits_arbitrary_dt": numerator,
                            "expected_spikes_arbitrary_dt": denominator,
                            "population_bits_per_spike": pop_bits,
                            "mean_unit_bits_per_spike": mean_bits,
                            "mean_unit_rate": mean_rate,
                            "unit_indices": unit_list_text(unit_idx),
                        }
                    )
    return rows


def bootstrap_ci(
    sub: pd.DataFrame,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if int(n_bootstrap) <= 0 or sub.empty:
        return float("nan"), float("nan")
    per_movie = (
        sub.groupby("movie_index", sort=True)[
            ["information_numerator_bits_arbitrary_dt", "expected_spikes_arbitrary_dt"]
        ]
        .sum()
        .reset_index()
    )
    numer = per_movie["information_numerator_bits_arbitrary_dt"].to_numpy(dtype=np.float64)
    denom = per_movie["expected_spikes_arbitrary_dt"].to_numpy(dtype=np.float64)
    n = int(numer.size)
    if n <= 1:
        return float("nan"), float("nan")
    sample_idx = rng.integers(0, n, size=(int(n_bootstrap), n))
    values = np.nansum(numer[sample_idx], axis=1) / np.maximum(np.nansum(denom[sample_idx], axis=1), EPS)
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_paired_delta_ci(
    condition_sub: pd.DataFrame,
    reference_sub: pd.DataFrame,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if int(n_bootstrap) <= 0 or condition_sub.empty or reference_sub.empty:
        return float("nan"), float("nan")
    condition = (
        condition_sub.groupby("movie_index", sort=True)[
            ["information_numerator_bits_arbitrary_dt", "expected_spikes_arbitrary_dt"]
        ]
        .sum()
        .rename(
            columns={
                "information_numerator_bits_arbitrary_dt": "condition_numerator",
                "expected_spikes_arbitrary_dt": "condition_denominator",
            }
        )
    )
    reference = (
        reference_sub.groupby("movie_index", sort=True)[
            ["information_numerator_bits_arbitrary_dt", "expected_spikes_arbitrary_dt"]
        ]
        .sum()
        .rename(
            columns={
                "information_numerator_bits_arbitrary_dt": "reference_numerator",
                "expected_spikes_arbitrary_dt": "reference_denominator",
            }
        )
    )
    paired = condition.join(reference, how="inner")
    n = int(paired.shape[0])
    if n <= 1:
        return float("nan"), float("nan")
    cond_num = paired["condition_numerator"].to_numpy(dtype=np.float64)
    cond_den = paired["condition_denominator"].to_numpy(dtype=np.float64)
    ref_num = paired["reference_numerator"].to_numpy(dtype=np.float64)
    ref_den = paired["reference_denominator"].to_numpy(dtype=np.float64)
    sample_idx = rng.integers(0, n, size=(int(n_bootstrap), n))
    cond_values = np.nansum(cond_num[sample_idx], axis=1) / np.maximum(np.nansum(cond_den[sample_idx], axis=1), EPS)
    ref_values = np.nansum(ref_num[sample_idx], axis=1) / np.maximum(np.nansum(ref_den[sample_idx], axis=1), EPS)
    lo, hi = np.nanpercentile(cond_values - ref_values, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_groups(
    per_fixation: pd.DataFrame,
    *,
    n_movies: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "condition_index",
        "condition_id",
        "condition_label",
        "along_scale",
        "across_scale",
        "motion_scale",
        "sweep_mode",
        "is_static_baseline",
        "is_across_sweep",
        "sf_group",
        "alignment_group",
    ]
    usable = per_fixation[per_fixation["usable_by_min_units"].astype(bool)].copy()
    for keys, sub in usable.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys, strict=True))
        values = sub["population_bits_per_spike"].to_numpy(dtype=np.float64)
        numerator = float(np.nansum(sub["information_numerator_bits_arbitrary_dt"].to_numpy(dtype=np.float64)))
        denominator = float(np.nansum(sub["expected_spikes_arbitrary_dt"].to_numpy(dtype=np.float64)))
        ci_lo, ci_hi = bootstrap_ci(sub, n_bootstrap=n_bootstrap, rng=rng)
        row.update(
            {
                "n_fixations_total": int(n_movies),
                "n_fixations_usable": int(sub["movie_index"].nunique()),
                "fraction_fixations_usable": float(sub["movie_index"].nunique() / max(int(n_movies), 1)),
                "mean_n_units_per_fixation": float(np.nanmean(sub["n_units"].to_numpy(dtype=float))),
                "min_n_units_per_fixation": int(np.nanmin(sub["n_units"].to_numpy(dtype=float))),
                "max_n_units_per_fixation": int(np.nanmax(sub["n_units"].to_numpy(dtype=float))),
                "accumulated_bits_per_spike": numerator / max(denominator, EPS),
                "accumulated_bits_per_spike_boot_ci_low": ci_lo,
                "accumulated_bits_per_spike_boot_ci_high": ci_hi,
                "mean_fixation_bits_per_spike": float(np.nanmean(values)),
                "sem_fixation_bits_per_spike": sem(values),
                "median_fixation_bits_per_spike": float(np.nanmedian(values)),
                "expected_spikes_sum_arbitrary_dt": denominator,
                "information_numerator_sum_bits_arbitrary_dt": numerator,
            }
        )
        if "effective_n_units" in sub.columns:
            row.update(
                {
                    "mean_effective_n_units_per_fixation": float(np.nanmean(sub["effective_n_units"].to_numpy(dtype=float))),
                    "min_effective_n_units_per_fixation": float(np.nanmin(sub["effective_n_units"].to_numpy(dtype=float))),
                    "max_effective_n_units_per_fixation": float(np.nanmax(sub["effective_n_units"].to_numpy(dtype=float))),
                }
            )
        if "weight_sum" in sub.columns:
            row.update(
                {
                    "mean_alignment_weight_sum_per_fixation": float(np.nanmean(sub["weight_sum"].to_numpy(dtype=float))),
                    "min_alignment_weight_sum_per_fixation": float(np.nanmin(sub["weight_sum"].to_numpy(dtype=float))),
                    "max_alignment_weight_sum_per_fixation": float(np.nanmax(sub["weight_sum"].to_numpy(dtype=float))),
                }
            )
        if "n_sf_units_total" in sub.columns:
            row["mean_n_sf_units_total"] = float(np.nanmean(sub["n_sf_units_total"].to_numpy(dtype=float)))
        if "n_orientation_tuned_units" in sub.columns:
            row["mean_n_orientation_tuned_units"] = float(
                np.nanmean(sub["n_orientation_tuned_units"].to_numpy(dtype=float))
            )
        rows.append(row)
    return rows


def summarize_alignment_contrasts(per_fixation: pd.DataFrame) -> list[dict[str, Any]]:
    usable = per_fixation[per_fixation["usable_by_min_units"].astype(bool)].copy()
    index_cols = [
        "condition_index",
        "condition_id",
        "condition_label",
        "along_scale",
        "across_scale",
        "motion_scale",
        "sweep_mode",
        "is_static_baseline",
        "is_across_sweep",
        "movie_index",
        "sf_group",
    ]
    pivot = usable.pivot_table(
        index=index_cols,
        columns="alignment_group",
        values=[
            "population_bits_per_spike",
            "information_numerator_bits_arbitrary_dt",
            "expected_spikes_arbitrary_dt",
            "n_units",
        ],
        aggfunc="first",
    )
    pivot.columns = [f"{value}_{group}" for value, group in pivot.columns]
    pivot = pivot.reset_index()
    required = {
        "population_bits_per_spike_contour_aligned",
        "population_bits_per_spike_contour_orthogonal",
        "information_numerator_bits_arbitrary_dt_contour_aligned",
        "information_numerator_bits_arbitrary_dt_contour_orthogonal",
        "expected_spikes_arbitrary_dt_contour_aligned",
        "expected_spikes_arbitrary_dt_contour_orthogonal",
    }
    missing = required.difference(pivot.columns)
    if missing:
        return []
    pivot["aligned_minus_orthogonal_bits_per_spike"] = (
        pivot["population_bits_per_spike_contour_aligned"]
        - pivot["population_bits_per_spike_contour_orthogonal"]
    )
    rows: list[dict[str, Any]] = []
    group_cols = [
        "condition_index",
        "condition_id",
        "condition_label",
        "along_scale",
        "across_scale",
        "motion_scale",
        "sweep_mode",
        "is_static_baseline",
        "is_across_sweep",
        "sf_group",
    ]
    for keys, sub in pivot.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys, strict=True))
        aligned_numer = float(np.nansum(sub["information_numerator_bits_arbitrary_dt_contour_aligned"]))
        aligned_denom = float(np.nansum(sub["expected_spikes_arbitrary_dt_contour_aligned"]))
        orth_numer = float(np.nansum(sub["information_numerator_bits_arbitrary_dt_contour_orthogonal"]))
        orth_denom = float(np.nansum(sub["expected_spikes_arbitrary_dt_contour_orthogonal"]))
        aligned_acc = aligned_numer / max(aligned_denom, EPS)
        orth_acc = orth_numer / max(orth_denom, EPS)
        values = sub["aligned_minus_orthogonal_bits_per_spike"].to_numpy(dtype=np.float64)
        row.update(
            {
                "n_fixations_with_both_groups": int(sub["movie_index"].nunique()),
                "accumulated_aligned_bits_per_spike": aligned_acc,
                "accumulated_orthogonal_bits_per_spike": orth_acc,
                "accumulated_aligned_minus_orthogonal_bits_per_spike": float(aligned_acc - orth_acc),
                "mean_fixation_aligned_minus_orthogonal_bits_per_spike": float(np.nanmean(values)),
                "sem_fixation_aligned_minus_orthogonal_bits_per_spike": sem(values),
                "median_fixation_aligned_minus_orthogonal_bits_per_spike": float(np.nanmedian(values)),
                "fraction_fixations_aligned_gt_orthogonal": float(np.nanmean(values > 0.0)),
                "mean_n_aligned_units": float(
                    np.nanmean(sub.get("n_units_contour_aligned", pd.Series(dtype=float)).to_numpy(dtype=float))
                ),
                "mean_n_orthogonal_units": float(
                    np.nanmean(sub.get("n_units_contour_orthogonal", pd.Series(dtype=float)).to_numpy(dtype=float))
                ),
            }
        )
        rows.append(row)
    return rows


def combined_per_fixation_frame(per_fixation: pd.DataFrame, *, min_units_per_fixation_group: int) -> pd.DataFrame:
    group_cols = [
        "condition_index",
        "condition_id",
        "condition_label",
        "along_scale",
        "across_scale",
        "motion_scale",
        "sweep_mode",
        "is_static_baseline",
        "is_across_sweep",
        "movie_index",
        "sf_group",
    ]
    combined = (
        per_fixation.groupby(group_cols, sort=True)[
            ["information_numerator_bits_arbitrary_dt", "expected_spikes_arbitrary_dt", "n_units"]
        ]
        .sum()
        .reset_index()
    )
    combined = combined[combined["n_units"].to_numpy(dtype=float) >= float(min_units_per_fixation_group)].copy()
    combined["alignment_group"] = "aligned_plus_orthogonal"
    combined["usable_by_min_units"] = True
    combined["population_bits_per_spike"] = combined["information_numerator_bits_arbitrary_dt"].to_numpy(
        dtype=np.float64
    ) / np.maximum(combined["expected_spikes_arbitrary_dt"].to_numpy(dtype=np.float64), EPS)
    return combined


def summarize_alignment_combined(
    per_fixation: pd.DataFrame,
    *,
    n_movies: int,
    min_units_per_fixation_group: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    source = per_fixation.copy()
    group_cols = [
        "condition_index",
        "condition_id",
        "condition_label",
        "along_scale",
        "across_scale",
        "motion_scale",
        "sweep_mode",
        "is_static_baseline",
        "is_across_sweep",
        "sf_group",
    ]
    rows: list[dict[str, Any]] = []
    for keys, source_sub in source.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys, strict=True))
        per_movie = (
            source_sub.groupby("movie_index", sort=True)[
                ["information_numerator_bits_arbitrary_dt", "expected_spikes_arbitrary_dt", "n_units"]
            ]
            .sum()
            .reset_index()
        )
        per_movie = per_movie[per_movie["n_units"].to_numpy(dtype=float) >= float(min_units_per_fixation_group)].copy()
        sub = source_sub[source_sub["movie_index"].isin(per_movie["movie_index"])].copy()
        if sub.empty or per_movie.empty:
            continue
        combined_values = per_movie["information_numerator_bits_arbitrary_dt"].to_numpy(dtype=np.float64) / np.maximum(
            per_movie["expected_spikes_arbitrary_dt"].to_numpy(dtype=np.float64),
            EPS,
        )
        numerator = float(np.nansum(sub["information_numerator_bits_arbitrary_dt"].to_numpy(dtype=np.float64)))
        denominator = float(np.nansum(sub["expected_spikes_arbitrary_dt"].to_numpy(dtype=np.float64)))
        ci_lo, ci_hi = bootstrap_ci(sub, n_bootstrap=n_bootstrap, rng=rng)
        unit_pivot = source_sub[source_sub["movie_index"].isin(per_movie["movie_index"])].pivot_table(
            index="movie_index",
            columns="alignment_group",
            values="n_units",
            aggfunc="first",
        )
        n_with_both = (
            int(unit_pivot[["contour_aligned", "contour_orthogonal"]].notna().all(axis=1).sum())
            if {"contour_aligned", "contour_orthogonal"}.issubset(unit_pivot.columns)
            else 0
        )
        row.update(
            {
                "combined_group": "aligned_plus_orthogonal",
                "combined_group_contract": (
                    "Combined pool is computed by summing information numerator and expected-spike denominator "
                    "across contour-aligned and contour-orthogonal selected units, then taking the bits/spike ratio."
                ),
                "n_fixations_total": int(n_movies),
                "n_fixations_with_any_pool": int(per_movie["movie_index"].nunique()),
                "n_fixations_with_both_pools": n_with_both,
                "accumulated_combined_bits_per_spike": numerator / max(denominator, EPS),
                "accumulated_combined_bits_per_spike_boot_ci_low": ci_lo,
                "accumulated_combined_bits_per_spike_boot_ci_high": ci_hi,
                "mean_fixation_combined_bits_per_spike": float(np.nanmean(combined_values)),
                "sem_fixation_combined_bits_per_spike": sem(combined_values),
                "median_fixation_combined_bits_per_spike": float(np.nanmedian(combined_values)),
                "mean_n_units_combined": float(np.nanmean(per_movie["n_units"].to_numpy(dtype=float))),
                "mean_n_aligned_units": (
                    float(np.nanmean(unit_pivot["contour_aligned"].to_numpy(dtype=float)))
                    if "contour_aligned" in unit_pivot
                    else float("nan")
                ),
                "mean_n_orthogonal_units": (
                    float(np.nanmean(unit_pivot["contour_orthogonal"].to_numpy(dtype=float)))
                    if "contour_orthogonal" in unit_pivot
                    else float("nan")
                ),
                "expected_spikes_sum_arbitrary_dt": denominator,
                "information_numerator_sum_bits_arbitrary_dt": numerator,
            }
        )
        if "effective_n_units" in source_sub.columns:
            effective_per_movie = (
                source_sub.groupby("movie_index", sort=True)["effective_n_units"]
                .sum()
                .reindex(per_movie["movie_index"])
                .to_numpy(dtype=float)
            )
            row.update(
                {
                    "mean_effective_n_units_combined": float(np.nanmean(effective_per_movie)),
                    "min_effective_n_units_combined": float(np.nanmin(effective_per_movie)),
                    "max_effective_n_units_combined": float(np.nanmax(effective_per_movie)),
                }
            )
        if "weight_sum" in source_sub.columns:
            weight_per_movie = (
                source_sub.groupby("movie_index", sort=True)["weight_sum"]
                .sum()
                .reindex(per_movie["movie_index"])
                .to_numpy(dtype=float)
            )
            row.update(
                {
                    "mean_alignment_weight_sum_combined": float(np.nanmean(weight_per_movie)),
                    "min_alignment_weight_sum_combined": float(np.nanmin(weight_per_movie)),
                    "max_alignment_weight_sum_combined": float(np.nanmax(weight_per_movie)),
                }
            )
        rows.append(row)
    return rows


def summarize_delta_from_reference(
    per_fixation: pd.DataFrame,
    *,
    reference_condition_id: str,
    n_movies: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    usable = per_fixation[per_fixation["usable_by_min_units"].astype(bool)].copy()
    reference = usable[usable["condition_id"].astype(str) == str(reference_condition_id)].copy()
    if reference.empty:
        return []
    rows: list[dict[str, Any]] = []
    condition_cols = [
        "condition_index",
        "condition_id",
        "condition_label",
        "along_scale",
        "across_scale",
        "motion_scale",
        "sweep_mode",
        "is_static_baseline",
        "is_across_sweep",
        "sf_group",
        "alignment_group",
    ]
    for keys, condition_sub in usable.groupby(condition_cols, sort=True):
        row = dict(zip(condition_cols, keys, strict=True))
        ref_sub = reference[
            (reference["sf_group"].astype(str) == str(row["sf_group"]))
            & (reference["alignment_group"].astype(str) == str(row["alignment_group"]))
        ].copy()
        if ref_sub.empty:
            continue
        paired = condition_sub[["movie_index", "population_bits_per_spike"]].merge(
            ref_sub[["movie_index", "population_bits_per_spike"]],
            on="movie_index",
            how="inner",
            suffixes=("_condition", "_reference"),
        )
        if paired.empty:
            continue
        paired_movie_index = set(int(v) for v in paired["movie_index"].to_numpy(dtype=int))
        condition_paired = condition_sub[condition_sub["movie_index"].astype(int).isin(paired_movie_index)].copy()
        reference_paired = ref_sub[ref_sub["movie_index"].astype(int).isin(paired_movie_index)].copy()
        condition_numerator = float(np.nansum(condition_paired["information_numerator_bits_arbitrary_dt"].to_numpy(dtype=float)))
        condition_denominator = float(np.nansum(condition_paired["expected_spikes_arbitrary_dt"].to_numpy(dtype=float)))
        reference_numerator = float(np.nansum(reference_paired["information_numerator_bits_arbitrary_dt"].to_numpy(dtype=float)))
        reference_denominator = float(np.nansum(reference_paired["expected_spikes_arbitrary_dt"].to_numpy(dtype=float)))
        condition_bits = condition_numerator / max(condition_denominator, EPS)
        reference_bits = reference_numerator / max(reference_denominator, EPS)
        values = (
            paired["population_bits_per_spike_condition"].to_numpy(dtype=float)
            - paired["population_bits_per_spike_reference"].to_numpy(dtype=float)
        )
        ci_lo, ci_hi = bootstrap_paired_delta_ci(
            condition_paired,
            reference_paired,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        row.update(
            {
                "reference_condition_id": str(reference_condition_id),
                "n_fixations_total": int(n_movies),
                "n_fixations_with_reference_and_condition": int(paired["movie_index"].nunique()),
                "accumulated_reference_bits_per_spike": reference_bits,
                "accumulated_condition_bits_per_spike": condition_bits,
                "accumulated_delta_vs_reference_bits_per_spike": float(condition_bits - reference_bits),
                "accumulated_delta_vs_reference_boot_ci_low": ci_lo,
                "accumulated_delta_vs_reference_boot_ci_high": ci_hi,
                "mean_fixation_delta_vs_reference_bits_per_spike": float(np.nanmean(values)),
                "sem_fixation_delta_vs_reference_bits_per_spike": sem(values),
                "median_fixation_delta_vs_reference_bits_per_spike": float(np.nanmedian(values)),
                "fraction_fixations_delta_positive": float(np.nanmean(values > 0.0)),
            }
        )
        rows.append(row)
    return rows


def summarize_orientation_pools(
    per_fixation: pd.DataFrame,
    *,
    n_movies: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "condition_index",
        "condition_id",
        "condition_label",
        "along_scale",
        "across_scale",
        "motion_scale",
        "sweep_mode",
        "is_static_baseline",
        "is_across_sweep",
        "sf_group",
        "orientation_pool",
    ]
    usable = per_fixation[per_fixation["usable_by_min_units"].astype(bool)].copy()
    for keys, sub in usable.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys, strict=True))
        values = sub["population_bits_per_spike"].to_numpy(dtype=np.float64)
        numerator = float(np.nansum(sub["information_numerator_bits_arbitrary_dt"].to_numpy(dtype=np.float64)))
        denominator = float(np.nansum(sub["expected_spikes_arbitrary_dt"].to_numpy(dtype=np.float64)))
        ci_lo, ci_hi = bootstrap_ci(sub, n_bootstrap=n_bootstrap, rng=rng)
        row.update(
            {
                "n_fixations_total": int(n_movies),
                "n_fixations_usable": int(sub["movie_index"].nunique()),
                "fraction_fixations_usable": float(sub["movie_index"].nunique() / max(int(n_movies), 1)),
                "mean_n_units_per_fixation": float(np.nanmean(sub["n_units"].to_numpy(dtype=float))),
                "min_n_units_per_fixation": int(np.nanmin(sub["n_units"].to_numpy(dtype=float))),
                "max_n_units_per_fixation": int(np.nanmax(sub["n_units"].to_numpy(dtype=float))),
                "accumulated_bits_per_spike": numerator / max(denominator, EPS),
                "accumulated_bits_per_spike_boot_ci_low": ci_lo,
                "accumulated_bits_per_spike_boot_ci_high": ci_hi,
                "mean_fixation_bits_per_spike": float(np.nanmean(values)),
                "sem_fixation_bits_per_spike": sem(values),
                "median_fixation_bits_per_spike": float(np.nanmedian(values)),
                "expected_spikes_sum_arbitrary_dt": denominator,
                "information_numerator_sum_bits_arbitrary_dt": numerator,
            }
        )
        rows.append(row)
    return rows


def choose_reference_and_endpoint(
    conditions: pd.DataFrame,
    *,
    reference_condition_id: str,
    endpoint_condition_id: str,
) -> tuple[str, str]:
    condition_ids = set(conditions["condition_id"].astype(str))
    if reference_condition_id:
        if reference_condition_id not in condition_ids:
            raise ValueError(f"reference condition {reference_condition_id!r} not found.")
        ref = str(reference_condition_id)
    else:
        ref_candidates = conditions[
            np.isclose(conditions["along_scale"].to_numpy(dtype=float), 1.0)
            & np.isclose(conditions["across_scale"].to_numpy(dtype=float), 1.0)
        ]
        if not ref_candidates.empty:
            ref = str(ref_candidates.iloc[0]["condition_id"])
        else:
            static = conditions[conditions["is_static_baseline"].astype(bool)]
            if not static.empty:
                ref = str(static.iloc[0]["condition_id"])
            else:
                non_static = conditions[~conditions["is_static_baseline"].astype(bool)]
                ref = str(non_static.iloc[0]["condition_id"] if not non_static.empty else conditions.iloc[0]["condition_id"])

    if endpoint_condition_id:
        if endpoint_condition_id not in condition_ids:
            raise ValueError(f"endpoint condition {endpoint_condition_id!r} not found.")
        end = str(endpoint_condition_id)
    else:
        sweep = conditions[conditions["is_across_sweep"].astype(bool)].copy()
        if sweep.empty:
            sweep = conditions.copy()
        sweep["_sort_scale"] = np.where(
            np.isfinite(sweep["motion_scale"].to_numpy(dtype=float)),
            sweep["motion_scale"].to_numpy(dtype=float),
            np.maximum(np.abs(sweep["along_scale"].to_numpy(dtype=float)), np.abs(sweep["across_scale"].to_numpy(dtype=float))),
        )
        sweep = sweep.sort_values(["_sort_scale", "condition_index"], ascending=[False, False])
        end = str(sweep.iloc[0]["condition_id"])
    return ref, end


def endpoint_delta_rows(per_fixation: pd.DataFrame, ref_condition_id: str, end_condition_id: str) -> list[dict[str, Any]]:
    usable = per_fixation[per_fixation["usable_by_min_units"].astype(bool)].copy()
    keep = usable[usable["condition_id"].astype(str).isin([str(ref_condition_id), str(end_condition_id)])].copy()
    pivot = keep.pivot_table(
        index=["movie_index", "sf_group", "alignment_group"],
        columns="condition_id",
        values="population_bits_per_spike",
        aggfunc="first",
    ).reset_index()
    if ref_condition_id not in pivot.columns or end_condition_id not in pivot.columns:
        return []
    pivot["endpoint_delta_bits_per_spike"] = pivot[str(end_condition_id)] - pivot[str(ref_condition_id)]
    rows: list[dict[str, Any]] = []
    accumulated_delta: dict[tuple[str, str], float] = {}
    for (sf_group, alignment_group), sub in pivot.groupby(["sf_group", "alignment_group"], sort=True):
        values = sub["endpoint_delta_bits_per_spike"].to_numpy(dtype=np.float64)
        source = keep[
            (keep["sf_group"].astype(str) == str(sf_group))
            & (keep["alignment_group"].astype(str) == str(alignment_group))
        ]
        ref = source[source["condition_id"].astype(str) == str(ref_condition_id)]
        end = source[source["condition_id"].astype(str) == str(end_condition_id)]
        ref_bits = float(np.nansum(ref["information_numerator_bits_arbitrary_dt"])) / max(
            float(np.nansum(ref["expected_spikes_arbitrary_dt"])),
            EPS,
        )
        end_bits = float(np.nansum(end["information_numerator_bits_arbitrary_dt"])) / max(
            float(np.nansum(end["expected_spikes_arbitrary_dt"])),
            EPS,
        )
        accum_delta = float(end_bits - ref_bits)
        accumulated_delta[(str(sf_group), str(alignment_group))] = accum_delta
        rows.append(
            {
                "sf_group": str(sf_group),
                "alignment_group": str(alignment_group),
                "reference_condition_id": str(ref_condition_id),
                "endpoint_condition_id": str(end_condition_id),
                "accumulated_reference_bits_per_spike": ref_bits,
                "accumulated_endpoint_bits_per_spike": end_bits,
                "accumulated_endpoint_delta_bits_per_spike": accum_delta,
                "n_fixations": int(np.isfinite(values).sum()),
                "mean_endpoint_delta_bits_per_spike": float(np.nanmean(values)),
                "sem_endpoint_delta_bits_per_spike": sem(values),
                "median_endpoint_delta_bits_per_spike": float(np.nanmedian(values)),
                "fraction_endpoint_delta_positive": float(np.nanmean(values > 0.0)),
            }
        )
    contrast_pivot = pivot.pivot_table(
        index=["movie_index", "sf_group"],
        columns="alignment_group",
        values="endpoint_delta_bits_per_spike",
        aggfunc="first",
    ).reset_index()
    if {"contour_aligned", "contour_orthogonal"}.issubset(contrast_pivot.columns):
        contrast_pivot["endpoint_delta_aligned_minus_orthogonal"] = (
            contrast_pivot["contour_aligned"] - contrast_pivot["contour_orthogonal"]
        )
        for sf_group, sub in contrast_pivot.groupby("sf_group", sort=True):
            values = sub["endpoint_delta_aligned_minus_orthogonal"].to_numpy(dtype=np.float64)
            accum_delta = accumulated_delta.get((str(sf_group), "contour_aligned"), float("nan")) - accumulated_delta.get(
                (str(sf_group), "contour_orthogonal"),
                float("nan"),
            )
            rows.append(
                {
                    "sf_group": str(sf_group),
                    "alignment_group": "aligned_minus_orthogonal",
                    "reference_condition_id": str(ref_condition_id),
                    "endpoint_condition_id": str(end_condition_id),
                    "accumulated_reference_bits_per_spike": float("nan"),
                    "accumulated_endpoint_bits_per_spike": float("nan"),
                    "accumulated_endpoint_delta_bits_per_spike": float(accum_delta),
                    "n_fixations": int(np.isfinite(values).sum()),
                    "mean_endpoint_delta_bits_per_spike": float(np.nanmean(values)),
                    "sem_endpoint_delta_bits_per_spike": sem(values),
                    "median_endpoint_delta_bits_per_spike": float(np.nanmedian(values)),
                    "fraction_endpoint_delta_positive": float(np.nanmean(values > 0.0)),
                }
            )
    return rows


def sf_label(sf_group: str) -> str:
    return {"low_sf": "low SF", "middle_sf": "middle SF", "high_sf": "high SF"}.get(str(sf_group), str(sf_group))


def alignment_label(alignment_group: str) -> str:
    return {
        "contour_aligned": "contour-aligned",
        "contour_orthogonal": "orthogonal",
    }.get(str(alignment_group), str(alignment_group))


def alignment_color(alignment_group: str) -> str:
    return {"contour_aligned": "#168a96", "contour_orthogonal": "#c06b2d"}.get(str(alignment_group), "0.35")


def sf_color(sf_group: str) -> str:
    return {"low_sf": "#2673a6", "middle_sf": "0.55", "high_sf": "#c74343"}.get(str(sf_group), "0.25")


def orientation_pool_label(orientation_pool: str) -> str:
    return {
        "all_sf_units": "all SF units",
        "target_oriented": "aligned + orthogonal",
        "off_axis_tuned": "off-axis tuned",
        "low_osi": "low OSI",
    }.get(str(orientation_pool), str(orientation_pool))


def orientation_pool_color(orientation_pool: str) -> str:
    return {
        "all_sf_units": "#222222",
        "target_oriented": "#666666",
        "off_axis_tuned": "#4f8f5b",
        "low_osi": "#b15a86",
    }.get(str(orientation_pool), "0.35")


def filter_conditions_for_sweep(
    conditions: pd.DataFrame,
    *,
    fixed_along_scale: float | None,
    fixed_across_scale: float | None,
) -> pd.DataFrame:
    keep = np.ones(int(conditions.shape[0]), dtype=bool)
    if fixed_along_scale is not None and np.isfinite(float(fixed_along_scale)):
        keep &= np.isclose(conditions["along_scale"].to_numpy(dtype=float), float(fixed_along_scale))
    if fixed_across_scale is not None and np.isfinite(float(fixed_across_scale)):
        keep &= np.isclose(conditions["across_scale"].to_numpy(dtype=float), float(fixed_across_scale))
    out = conditions.loc[keep].copy()
    if out.empty:
        raise ValueError(
            f"No conditions remain for fixed_along_scale={fixed_along_scale} "
            f"fixed_across_scale={fixed_across_scale}."
        )
    return out.reset_index(drop=True)


def x_values_for_sweep(frame: pd.DataFrame, sweep_axis: str) -> np.ndarray:
    if str(sweep_axis) == "across":
        return frame["across_scale"].to_numpy(dtype=float)
    if str(sweep_axis) == "along":
        return frame["along_scale"].to_numpy(dtype=float)
    return frame["motion_scale"].to_numpy(dtype=float)


def x_label_for_sweep(sweep_axis: str, *, fixed_along_scale: float | None, fixed_across_scale: float | None) -> str:
    if str(sweep_axis) == "across":
        if fixed_along_scale is not None and np.isfinite(float(fixed_along_scale)):
            return f"across-contour scale (along={float(fixed_along_scale):g})"
        return "across-contour scale"
    if str(sweep_axis) == "along":
        if fixed_across_scale is not None and np.isfinite(float(fixed_across_scale)):
            return f"along-contour scale (across={float(fixed_across_scale):g})"
        return "along-contour scale"
    return "motion scale"


def sweep_description(sweep_axis: str, *, fixed_along_scale: float | None, fixed_across_scale: float | None) -> str:
    if str(sweep_axis) == "across":
        return (
            f"x varies the across-contour component with along-contour scale fixed at "
            f"{float(fixed_along_scale):g}"
            if fixed_along_scale is not None and np.isfinite(float(fixed_along_scale))
            else "x varies the across-contour component"
        )
    if str(sweep_axis) == "along":
        return (
            f"x varies the along-contour component with across-contour scale fixed at "
            f"{float(fixed_across_scale):g}"
            if fixed_across_scale is not None and np.isfinite(float(fixed_across_scale))
            else "x varies the along-contour component"
        )
    return "cached motion scale"


def sorted_sweep(summary: pd.DataFrame, *, sweep_axis: str = "auto") -> pd.DataFrame:
    if str(sweep_axis) == "auto":
        sweep = summary[summary["is_across_sweep"].astype(bool)].copy()
        if sweep.empty:
            sweep = summary.copy()
    else:
        sweep = summary.copy()
    sweep["_x"] = x_values_for_sweep(sweep, str(sweep_axis))
    return sweep.sort_values(["_x", "condition_index"], kind="mergesort")


def fixation_support_label(sub: pd.DataFrame, usable_col: str, total_col: str = "n_fixations_total") -> tuple[str, float]:
    if sub.empty or usable_col not in sub.columns:
        return "", float("nan")
    usable = pd.to_numeric(sub[usable_col], errors="coerce").to_numpy(dtype=float)
    usable = usable[np.isfinite(usable)]
    if usable.size == 0:
        return "", float("nan")
    n_usable = int(np.nanmedian(usable))
    if total_col in sub.columns:
        total = pd.to_numeric(sub[total_col], errors="coerce").to_numpy(dtype=float)
        total = total[np.isfinite(total)]
        n_total = int(np.nanmedian(total)) if total.size else 0
    else:
        n_total = 0
    if n_total > 0:
        return f"{n_usable}/{n_total} fix", float(n_usable / max(n_total, 1))
    return f"{n_usable} fix", float("nan")


def plot_population_curves(
    out_dir: Path,
    group_summary: pd.DataFrame,
    combined_summary: pd.DataFrame,
    sf_groups: list[str],
    *,
    metric_name: str,
    sweep_axis: str,
    x_label: str,
    sweep_text: str,
    dpi: int,
    output_stem: str = "backimage_rr100_sf_contour_alignment_population_curves",
    title_prefix: str = "BackImage RR100 SF groups split per fixation by contour alignment",
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(len(sf_groups), 2, figsize=(11.4, 3.45 * len(sf_groups)), squeeze=False, constrained_layout=True)
    for row_idx, sf_group in enumerate(sf_groups):
        ax_raw = axes[row_idx, 0]
        ax_combined = axes[row_idx, 1]
        sf_summary = sorted_sweep(group_summary[group_summary["sf_group"].astype(str) == str(sf_group)], sweep_axis=sweep_axis)
        for alignment_group in ["contour_aligned", "contour_orthogonal"]:
            sub = sf_summary[sf_summary["alignment_group"].astype(str) == alignment_group]
            if sub.empty:
                continue
            x = sub["_x"].to_numpy(dtype=float)
            y = sub["accumulated_bits_per_spike"].to_numpy(dtype=float)
            lo = sub["accumulated_bits_per_spike_boot_ci_low"].to_numpy(dtype=float)
            hi = sub["accumulated_bits_per_spike_boot_ci_high"].to_numpy(dtype=float)
            color = alignment_color(alignment_group)
            support_text, support_fraction = fixation_support_label(sub, "n_fixations_usable")
            linestyle = "--" if np.isfinite(support_fraction) and support_fraction < 0.5 else "-"
            finite_ci = np.isfinite(lo) & np.isfinite(hi)
            if finite_ci.any():
                ax_raw.fill_between(
                    x[finite_ci],
                    lo[finite_ci],
                    hi[finite_ci],
                    color=color,
                    alpha=0.09 if linestyle == "--" else 0.14,
                    linewidth=0.0,
                )
            label = alignment_label(alignment_group)
            if support_text:
                label = f"{label} ({support_text})"
            ax_raw.plot(
                x,
                y,
                marker="o",
                linewidth=2.0,
                markersize=4.2,
                color=color,
                linestyle=linestyle,
                label=label,
            )
        ax_raw.axvline(1.0, color="0.65", linestyle=":", linewidth=1.0)
        ax_raw.grid(True, color="0.9", linewidth=0.7)
        ax_raw.set_title(f"{sf_label(sf_group)} accumulated population SSI", fontsize=10)
        ax_raw.set_xlabel(x_label)
        ax_raw.set_ylabel("bits/spike")
        ax_raw.legend(frameon=False, fontsize=8)

        sf_combined = sorted_sweep(
            combined_summary[combined_summary["sf_group"].astype(str) == str(sf_group)],
            sweep_axis=sweep_axis,
        )
        combined_support_text = ""
        if not sf_combined.empty:
            x = sf_combined["_x"].to_numpy(dtype=float)
            y = sf_combined["accumulated_combined_bits_per_spike"].to_numpy(dtype=float)
            lo = sf_combined["accumulated_combined_bits_per_spike_boot_ci_low"].to_numpy(dtype=float)
            hi = sf_combined["accumulated_combined_bits_per_spike_boot_ci_high"].to_numpy(dtype=float)
            color = "#4c4c4c"
            combined_support_text, _ = fixation_support_label(sf_combined, "n_fixations_with_any_pool")
            finite_ci = np.isfinite(lo) & np.isfinite(hi)
            if finite_ci.any():
                ax_combined.fill_between(x[finite_ci], lo[finite_ci], hi[finite_ci], color=color, alpha=0.14, linewidth=0.0)
            ax_combined.plot(x, y, marker="o", linewidth=2.0, markersize=4.2, color=color)
        ax_combined.axvline(1.0, color="0.65", linestyle=":", linewidth=1.0)
        ax_combined.grid(True, color="0.9", linewidth=0.7)
        combined_title = f"{sf_label(sf_group)} aligned + orthogonal pooled"
        if combined_support_text:
            combined_title = f"{combined_title}\n({combined_support_text}; min n after pooling)"
        ax_combined.set_title(combined_title, fontsize=10)
        ax_combined.set_xlabel(x_label)
        ax_combined.set_ylabel("bits/spike")
    fig.suptitle(
        f"{title_prefix}\n"
        f"{metric_name}; {sweep_text}",
        fontsize=12,
    )
    png = out_dir / f"{output_stem}.png"
    pdf = out_dir / f"{output_stem}.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_delta_from_reference_curves(
    out_dir: Path,
    group_delta_summary: pd.DataFrame,
    combined_delta_summary: pd.DataFrame,
    sf_groups: list[str],
    *,
    reference_condition_id: str,
    metric_name: str,
    sweep_axis: str,
    x_label: str,
    sweep_text: str,
    dpi: int,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(len(sf_groups), 2, figsize=(11.4, 3.45 * len(sf_groups)), squeeze=False, constrained_layout=True)
    y_values: list[float] = []
    for row_idx, sf_group in enumerate(sf_groups):
        ax_raw = axes[row_idx, 0]
        ax_combined = axes[row_idx, 1]
        sf_summary = sorted_sweep(
            group_delta_summary[group_delta_summary["sf_group"].astype(str) == str(sf_group)],
            sweep_axis=sweep_axis,
        )
        for alignment_group in ["contour_aligned", "contour_orthogonal"]:
            sub = sf_summary[sf_summary["alignment_group"].astype(str) == alignment_group]
            if sub.empty:
                continue
            x = sub["_x"].to_numpy(dtype=float)
            y = sub["accumulated_delta_vs_reference_bits_per_spike"].to_numpy(dtype=float)
            lo = sub["accumulated_delta_vs_reference_boot_ci_low"].to_numpy(dtype=float)
            hi = sub["accumulated_delta_vs_reference_boot_ci_high"].to_numpy(dtype=float)
            color = alignment_color(alignment_group)
            finite_ci = np.isfinite(lo) & np.isfinite(hi)
            if finite_ci.any():
                ax_raw.fill_between(x[finite_ci], lo[finite_ci], hi[finite_ci], color=color, alpha=0.13, linewidth=0.0)
                y_values.extend(lo[finite_ci].tolist())
                y_values.extend(hi[finite_ci].tolist())
            ax_raw.plot(
                x,
                y,
                marker="o",
                linewidth=2.1,
                markersize=4.4,
                color=color,
                label=alignment_label(alignment_group),
            )
            y_values.extend(y[np.isfinite(y)].tolist())
        ax_raw.axhline(0.0, color="0.45", linestyle="--", linewidth=0.9)
        ax_raw.axvline(1.0, color="0.65", linestyle=":", linewidth=1.0)
        ax_raw.grid(True, color="0.9", linewidth=0.7)
        ax_raw.set_title(f"{sf_label(sf_group)} change from reference", fontsize=10)
        ax_raw.set_xlabel(x_label)
        ax_raw.set_ylabel("SSI - reference SSI (bits/spike)")
        ax_raw.legend(frameon=False, fontsize=8)

        sf_combined = sorted_sweep(
            combined_delta_summary[combined_delta_summary["sf_group"].astype(str) == str(sf_group)],
            sweep_axis=sweep_axis,
        )
        if not sf_combined.empty:
            x = sf_combined["_x"].to_numpy(dtype=float)
            y = sf_combined["accumulated_delta_vs_reference_bits_per_spike"].to_numpy(dtype=float)
            lo = sf_combined["accumulated_delta_vs_reference_boot_ci_low"].to_numpy(dtype=float)
            hi = sf_combined["accumulated_delta_vs_reference_boot_ci_high"].to_numpy(dtype=float)
            color = "#4c4c4c"
            finite_ci = np.isfinite(lo) & np.isfinite(hi)
            if finite_ci.any():
                ax_combined.fill_between(x[finite_ci], lo[finite_ci], hi[finite_ci], color=color, alpha=0.14, linewidth=0.0)
                y_values.extend(lo[finite_ci].tolist())
                y_values.extend(hi[finite_ci].tolist())
            ax_combined.plot(x, y, marker="o", linewidth=2.1, markersize=4.4, color=color)
            y_values.extend(y[np.isfinite(y)].tolist())
        ax_combined.axhline(0.0, color="0.45", linestyle="--", linewidth=0.9)
        ax_combined.axvline(1.0, color="0.65", linestyle=":", linewidth=1.0)
        ax_combined.grid(True, color="0.9", linewidth=0.7)
        ax_combined.set_title(f"{sf_label(sf_group)} aligned + orthogonal pooled\nchange from reference", fontsize=10)
        ax_combined.set_xlabel(x_label)
        ax_combined.set_ylabel("SSI - reference SSI (bits/spike)")
    finite_y = np.asarray(y_values, dtype=float)
    finite_y = finite_y[np.isfinite(finite_y)]
    if finite_y.size:
        pad = max(0.08 * float(np.ptp(finite_y)), 0.004)
        lo = min(float(np.nanmin(finite_y) - pad), -pad)
        hi = max(float(np.nanmax(finite_y) + pad), pad)
        for ax in axes.ravel():
            ax.set_ylim(lo, hi)
    fig.suptitle(
        "BackImage RR100 SF groups split per fixation: SSI change from reference\n"
        f"{metric_name}; reference={reference_condition_id}; {sweep_text}",
        fontsize=12,
    )
    png = out_dir / "backimage_rr100_sf_contour_alignment_delta_vs_reference_curves.png"
    pdf = out_dir / "backimage_rr100_sf_contour_alignment_delta_vs_reference_curves.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_alignment_separate_population_curves(
    out_dir: Path,
    group_summary: pd.DataFrame,
    sf_groups: list[str],
    *,
    metric_name: str,
    sweep_axis: str,
    x_label: str,
    sweep_text: str,
    dpi: int,
    output_stem: str = "backimage_rr100_sf_contour_alignment_separate_population_curves",
    title_prefix: str = "BackImage RR100 accumulated SSI by local contour-alignment pool",
    alignment_title_suffix: str = "selected units",
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), sharey=True, constrained_layout=True)
    alignment_groups = ["contour_aligned", "contour_orthogonal"]
    y_values: list[float] = []
    for ax, alignment_group in zip(axes, alignment_groups, strict=True):
        align_summary = sorted_sweep(
            group_summary[group_summary["alignment_group"].astype(str) == alignment_group],
            sweep_axis=sweep_axis,
        )
        for sf_group in sf_groups:
            sub = align_summary[align_summary["sf_group"].astype(str) == str(sf_group)]
            if sub.empty:
                continue
            x = sub["_x"].to_numpy(dtype=float)
            y = sub["accumulated_bits_per_spike"].to_numpy(dtype=float)
            lo = sub["accumulated_bits_per_spike_boot_ci_low"].to_numpy(dtype=float)
            hi = sub["accumulated_bits_per_spike_boot_ci_high"].to_numpy(dtype=float)
            color = sf_color(sf_group)
            finite_ci = np.isfinite(lo) & np.isfinite(hi)
            if finite_ci.any():
                ax.fill_between(x[finite_ci], lo[finite_ci], hi[finite_ci], color=color, alpha=0.13, linewidth=0.0)
                y_values.extend(lo[finite_ci].tolist())
                y_values.extend(hi[finite_ci].tolist())
            ax.plot(x, y, marker="o", linewidth=2.1, markersize=4.4, color=color, label=sf_label(sf_group))
            y_values.extend(y[np.isfinite(y)].tolist())
        ax.axvline(1.0, color="0.65", linestyle=":", linewidth=1.0)
        ax.grid(True, color="0.9", linewidth=0.7)
        ax.set_title(f"{alignment_label(alignment_group)} {alignment_title_suffix}", fontsize=10.5)
        ax.set_xlabel(x_label)
        ax.legend(frameon=False, fontsize=8, loc="best")
    axes[0].set_ylabel("accumulated population SSI (bits/spike)")
    finite_y = np.asarray(y_values, dtype=float)
    finite_y = finite_y[np.isfinite(finite_y)]
    if finite_y.size:
        pad = max(0.01 * float(np.ptp(finite_y)), 0.004)
        for ax in axes:
            ax.set_ylim(float(np.nanmin(finite_y) - pad), float(np.nanmax(finite_y) + pad))
    fig.suptitle(
        f"{title_prefix}\n"
        f"{metric_name}; {sweep_text}",
        fontsize=12,
    )
    png = out_dir / f"{output_stem}.png"
    pdf = out_dir / f"{output_stem}.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_selection_counts(
    out_dir: Path,
    selection: pd.DataFrame,
    sf_groups: list[str],
    *,
    dpi: int,
    output_stem: str = "backimage_rr100_sf_contour_alignment_selection_counts",
    title_prefix: str = "Selection",
    y_label: str = "selected units per fixation",
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    labels: list[str] = []
    values: list[np.ndarray] = []
    colors: list[str] = []
    for sf_group in sf_groups:
        for alignment_group in ["contour_aligned", "contour_orthogonal"]:
            sub = selection[
                (selection["sf_group"].astype(str) == str(sf_group))
                & (selection["alignment_group"].astype(str) == str(alignment_group))
            ]
            labels.append(f"{sf_label(sf_group)}\n{alignment_label(alignment_group)}")
            values.append(sub["n_units"].to_numpy(dtype=float))
            colors.append(alignment_color(alignment_group))
    box = axes[0].boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    for median in box["medians"]:
        median.set_color("0.15")
        median.set_linewidth(1.2)
    axes[0].set_ylabel(y_label)
    axes[0].set_title(f"{title_prefix}: per-fixation group size")
    axes[0].tick_params(axis="x", labelsize=7)
    axes[0].grid(True, axis="y", color="0.9", linewidth=0.7)

    target_delta = selection["mean_orientation_target_delta_deg"].to_numpy(dtype=float)
    osi = selection["mean_orientation_selectivity_index"].to_numpy(dtype=float)
    n_units = selection["n_units"].to_numpy(dtype=float)
    ok = np.isfinite(target_delta) & np.isfinite(osi) & (n_units > 0)
    axes[1].scatter(target_delta[ok], osi[ok], s=18 + 4 * n_units[ok], color="#5c6f7c", alpha=0.55, linewidths=0.0)
    axes[1].set_xlabel("mean target-axis distance (deg)")
    axes[1].set_ylabel("mean orientation selectivity")
    axes[1].set_title(f"{title_prefix}: quality by fixation/group")
    axes[1].grid(True, color="0.9", linewidth=0.7)
    png = out_dir / f"{output_stem}.png"
    pdf = out_dir / f"{output_stem}.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_endpoint_distributions(
    out_dir: Path,
    per_fixation: pd.DataFrame,
    sf_groups: list[str],
    ref_condition_id: str,
    end_condition_id: str,
    *,
    dpi: int,
) -> tuple[Path, Path]:
    usable = per_fixation[per_fixation["usable_by_min_units"].astype(bool)].copy()
    endpoint = usable[usable["condition_id"].astype(str).isin([ref_condition_id, end_condition_id])].copy()
    endpoint_pivot = endpoint.pivot_table(
        index=["movie_index", "sf_group", "alignment_group"],
        columns="condition_id",
        values="population_bits_per_spike",
        aggfunc="first",
    ).reset_index()
    if ref_condition_id in endpoint_pivot.columns and end_condition_id in endpoint_pivot.columns:
        endpoint_pivot["endpoint_delta"] = endpoint_pivot[end_condition_id] - endpoint_pivot[ref_condition_id]
    else:
        endpoint_pivot["endpoint_delta"] = np.nan
    contrast = endpoint_pivot.pivot_table(
        index=["movie_index", "sf_group"],
        columns="alignment_group",
        values="endpoint_delta",
        aggfunc="first",
    ).reset_index()
    if {"contour_aligned", "contour_orthogonal"}.issubset(contrast.columns):
        contrast["aligned_minus_orthogonal_endpoint_delta"] = contrast["contour_aligned"] - contrast["contour_orthogonal"]
    else:
        contrast["aligned_minus_orthogonal_endpoint_delta"] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    rng = np.random.default_rng(0)
    positions: list[float] = []
    values: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []
    pos = 0
    for sf_group in sf_groups:
        for alignment_group in ["contour_aligned", "contour_orthogonal"]:
            sub = endpoint_pivot[
                (endpoint_pivot["sf_group"].astype(str) == str(sf_group))
                & (endpoint_pivot["alignment_group"].astype(str) == alignment_group)
            ]
            y = sub["endpoint_delta"].to_numpy(dtype=float)
            y = y[np.isfinite(y)]
            positions.append(float(pos))
            values.append(y)
            labels.append(f"{sf_label(sf_group)}\n{alignment_label(alignment_group)}")
            colors.append(alignment_color(alignment_group))
            jitter = rng.normal(0.0, 0.04, size=y.size)
            axes[0].scatter(np.full(y.size, pos) + jitter, y, s=9, color=alignment_color(alignment_group), alpha=0.35, linewidths=0)
            pos += 1
    box = axes[0].boxplot(values, positions=positions, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.20)
        patch.set_edgecolor(color)
    axes[0].axhline(0.0, color="0.45", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel(f"{end_condition_id} minus {ref_condition_id} bits/spike")
    axes[0].set_title("Endpoint delta by selected pool")
    axes[0].tick_params(axis="x", labelsize=7)
    axes[0].grid(True, axis="y", color="0.9", linewidth=0.7)

    contrast_values: list[np.ndarray] = []
    contrast_labels: list[str] = []
    for idx, sf_group in enumerate(sf_groups):
        sub = contrast[contrast["sf_group"].astype(str) == str(sf_group)]
        y = sub["aligned_minus_orthogonal_endpoint_delta"].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        contrast_values.append(y)
        contrast_labels.append(sf_label(sf_group))
        jitter = rng.normal(0.0, 0.04, size=y.size)
        axes[1].scatter(np.full(y.size, idx) + jitter, y, s=11, color="#4c4c4c", alpha=0.38, linewidths=0)
    axes[1].boxplot(contrast_values, tick_labels=contrast_labels, patch_artist=True, showfliers=False)
    axes[1].axhline(0.0, color="0.45", linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("aligned-minus-orthogonal endpoint delta")
    axes[1].set_title("Interaction contrast")
    axes[1].grid(True, axis="y", color="0.9", linewidth=0.7)

    png = out_dir / "backimage_rr100_sf_contour_alignment_endpoint_distributions.png"
    pdf = out_dir / "backimage_rr100_sf_contour_alignment_endpoint_distributions.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_orientation_pool_control_curves(
    out_dir: Path,
    orientation_summary: pd.DataFrame,
    sf_groups: list[str],
    *,
    metric_name: str,
    sweep_axis: str,
    x_label: str,
    sweep_text: str,
    dpi: int,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, len(sf_groups), figsize=(5.7 * len(sf_groups), 4.25), squeeze=False, constrained_layout=True)
    for ax, sf_group in zip(axes[0], sf_groups, strict=True):
        sf_summary = sorted_sweep(
            orientation_summary[orientation_summary["sf_group"].astype(str) == str(sf_group)],
            sweep_axis=sweep_axis,
        )
        y_values: list[float] = []
        for orientation_pool in ORIENTATION_POOLS:
            sub = sf_summary[sf_summary["orientation_pool"].astype(str) == orientation_pool]
            if sub.empty:
                continue
            x = sub["_x"].to_numpy(dtype=float)
            y = sub["accumulated_bits_per_spike"].to_numpy(dtype=float)
            lo = sub["accumulated_bits_per_spike_boot_ci_low"].to_numpy(dtype=float)
            hi = sub["accumulated_bits_per_spike_boot_ci_high"].to_numpy(dtype=float)
            color = orientation_pool_color(orientation_pool)
            finite_ci = np.isfinite(lo) & np.isfinite(hi)
            if finite_ci.any():
                ax.fill_between(x[finite_ci], lo[finite_ci], hi[finite_ci], color=color, alpha=0.11, linewidth=0.0)
                y_values.extend(lo[finite_ci].tolist())
                y_values.extend(hi[finite_ci].tolist())
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.0 if orientation_pool != "all_sf_units" else 2.3,
                markersize=4.0,
                color=color,
                label=orientation_pool_label(orientation_pool),
            )
            y_values.extend(y[np.isfinite(y)].tolist())
        ax.axvline(1.0, color="0.65", linestyle=":", linewidth=1.0)
        ax.grid(True, color="0.9", linewidth=0.7)
        ax.set_title(f"{sf_label(sf_group)} orientation-control pools", fontsize=10.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel("bits/spike")
        ax.legend(frameon=False, fontsize=7.5, loc="best")
        finite_y = np.asarray(y_values, dtype=float)
        finite_y = finite_y[np.isfinite(finite_y)]
        if finite_y.size:
            pad = max(0.01 * float(np.ptp(finite_y)), 0.003)
            ax.set_ylim(float(np.nanmin(finite_y) - pad), float(np.nanmax(finite_y) + pad))
    fig.suptitle(
        "BackImage RR100 SF groups by orientation-control pool\n"
        f"{metric_name}; {sweep_text}",
        fontsize=12,
    )
    png = out_dir / "backimage_rr100_sf_orientation_pool_control_curves.png"
    pdf = out_dir / "backimage_rr100_sf_orientation_pool_control_curves.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def best_row(rows: list[dict[str, Any]], sf_group: str, alignment_group: str) -> dict[str, Any] | None:
    matches = [
        row for row in rows if str(row.get("sf_group")) == str(sf_group) and str(row.get("alignment_group")) == str(alignment_group)
    ]
    if not matches:
        return None
    return matches[0]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = load_npz_without_identity(cache_path(Path(args.contour_run_dir)))
    bits, metric_key = metric_array(stats, str(args.ssi_metric))
    n_conditions, n_movies, n_units = bits.shape
    conditions = filter_conditions_for_sweep(
        condition_frame(stats),
        fixed_along_scale=args.fixed_along_scale,
        fixed_across_scale=args.fixed_across_scale,
    )
    movies = load_movie_frame(Path(args.contour_run_dir), stats, str(args.axis_coordinate_frame))
    sf_groups = parse_csv_list(str(args.sf_groups))
    high_sf_min_cpd = None if args.high_sf_min_cpd is None else float(args.high_sf_min_cpd)
    units = load_unit_table(Path(args.sf_groups_csv), sf_groups, int(n_units), high_sf_min_cpd=high_sf_min_cpd)
    sf_definition_rows = sf_group_definition_rows(Path(args.sf_groups_csv), int(n_units), high_sf_min_cpd=high_sf_min_cpd)
    sweep_axis = str(args.sweep_axis)
    x_label = x_label_for_sweep(
        sweep_axis,
        fixed_along_scale=args.fixed_along_scale,
        fixed_across_scale=args.fixed_across_scale,
    )
    sweep_text = sweep_description(
        sweep_axis,
        fixed_along_scale=args.fixed_along_scale,
        fixed_across_scale=args.fixed_across_scale,
    )

    selection_rows, masks = build_selection_table(
        units,
        movies,
        sf_groups=sf_groups,
        alignment_angle_deg=float(args.alignment_angle_deg),
        min_orientation_selectivity=float(args.min_orientation_selectivity),
        min_units_per_fixation_group=int(args.min_units_per_fixation_group),
    )
    weighted_selection_rows, weighted_pools = build_weighted_alignment_table(
        units,
        movies,
        sf_groups=sf_groups,
        min_orientation_selectivity=float(args.min_orientation_selectivity),
        min_units_per_fixation_group=int(args.min_units_per_fixation_group),
    )
    orientation_pool_selection_rows, orientation_pool_masks = build_orientation_pool_selection_table(
        units,
        movies,
        sf_groups=sf_groups,
        alignment_angle_deg=float(args.alignment_angle_deg),
        min_orientation_selectivity=float(args.min_orientation_selectivity),
        min_units_per_fixation_group=int(args.min_units_per_fixation_group),
    )
    per_fix_rows = per_fixation_group_rows(
        stats,
        movies,
        conditions,
        masks,
        sf_groups=sf_groups,
        ssi_metric=str(args.ssi_metric),
        min_units_per_fixation_group=int(args.min_units_per_fixation_group),
    )
    weighted_per_fix_rows = per_fixation_weighted_alignment_rows(
        stats,
        movies,
        conditions,
        weighted_pools,
        sf_groups=sf_groups,
        ssi_metric=str(args.ssi_metric),
    )
    orientation_pool_per_fix_rows = per_fixation_orientation_pool_rows(
        stats,
        movies,
        conditions,
        orientation_pool_masks,
        sf_groups=sf_groups,
        ssi_metric=str(args.ssi_metric),
        min_units_per_fixation_group=int(args.min_units_per_fixation_group),
    )

    rng = np.random.default_rng(int(args.bootstrap_seed))
    per_fix_df = pd.DataFrame(per_fix_rows)
    weighted_per_fix_df = pd.DataFrame(weighted_per_fix_rows)
    orientation_pool_per_fix_df = pd.DataFrame(orientation_pool_per_fix_rows)
    group_rows = summarize_groups(
        per_fix_df,
        n_movies=int(n_movies),
        n_bootstrap=int(args.n_bootstrap),
        rng=rng,
    )
    contrast_rows = summarize_alignment_contrasts(per_fix_df)
    combined_rows = summarize_alignment_combined(
        per_fix_df,
        n_movies=int(n_movies),
        min_units_per_fixation_group=int(args.min_units_per_fixation_group),
        n_bootstrap=int(args.n_bootstrap),
        rng=rng,
    )
    orientation_pool_rows = summarize_orientation_pools(
        orientation_pool_per_fix_df,
        n_movies=int(n_movies),
        n_bootstrap=int(args.n_bootstrap),
        rng=rng,
    )
    weighted_rng = np.random.default_rng(int(args.bootstrap_seed) + 10)
    weighted_group_rows = summarize_groups(
        weighted_per_fix_df,
        n_movies=int(n_movies),
        n_bootstrap=int(args.n_bootstrap),
        rng=weighted_rng,
    )
    weighted_contrast_rows = summarize_alignment_contrasts(weighted_per_fix_df)
    weighted_combined_rows = summarize_alignment_combined(
        weighted_per_fix_df,
        n_movies=int(n_movies),
        min_units_per_fixation_group=int(args.min_units_per_fixation_group),
        n_bootstrap=int(args.n_bootstrap),
        rng=np.random.default_rng(int(args.bootstrap_seed) + 11),
    )
    ref_condition_id, end_condition_id = choose_reference_and_endpoint(
        conditions,
        reference_condition_id=str(args.reference_condition_id).strip(),
        endpoint_condition_id=str(args.endpoint_condition_id).strip(),
    )
    endpoint_rows = endpoint_delta_rows(per_fix_df, ref_condition_id, end_condition_id)
    delta_rng = np.random.default_rng(int(args.bootstrap_seed) + 1)
    group_delta_rows = summarize_delta_from_reference(
        per_fix_df,
        reference_condition_id=ref_condition_id,
        n_movies=int(n_movies),
        n_bootstrap=int(args.n_bootstrap),
        rng=delta_rng,
    )
    combined_per_fix_df = combined_per_fixation_frame(
        per_fix_df,
        min_units_per_fixation_group=int(args.min_units_per_fixation_group),
    )
    combined_delta_rng = np.random.default_rng(int(args.bootstrap_seed) + 2)
    combined_delta_rows = summarize_delta_from_reference(
        combined_per_fix_df,
        reference_condition_id=ref_condition_id,
        n_movies=int(n_movies),
        n_bootstrap=int(args.n_bootstrap),
        rng=combined_delta_rng,
    )

    selection_csv = out_dir / "per_fixation_unit_selection.csv"
    weighted_selection_csv = out_dir / "per_fixation_weighted_alignment_selection.csv"
    per_fix_csv = out_dir / "per_fixation_group_population_ssi.csv"
    weighted_per_fix_csv = out_dir / "per_fixation_weighted_alignment_population_ssi.csv"
    group_csv = out_dir / "condition_group_population_ssi_summary.csv"
    weighted_group_csv = out_dir / "condition_weighted_alignment_population_ssi_summary.csv"
    contrast_csv = out_dir / "condition_alignment_contrast_summary.csv"
    weighted_contrast_csv = out_dir / "condition_weighted_alignment_contrast_summary.csv"
    combined_csv = out_dir / "condition_alignment_combined_summary.csv"
    weighted_combined_csv = out_dir / "condition_weighted_alignment_combined_summary.csv"
    group_delta_csv = out_dir / "condition_group_delta_vs_reference_summary.csv"
    combined_delta_csv = out_dir / "condition_alignment_combined_delta_vs_reference_summary.csv"
    orientation_pool_selection_csv = out_dir / "per_fixation_orientation_pool_selection.csv"
    orientation_pool_per_fix_csv = out_dir / "per_fixation_orientation_pool_population_ssi.csv"
    orientation_pool_csv = out_dir / "condition_orientation_pool_population_ssi_summary.csv"
    endpoint_csv = out_dir / "endpoint_delta_summary.csv"
    units_csv = out_dir / "analysis_unit_table.csv"
    sf_definition_csv = out_dir / "sf_group_definition_summary.csv"
    write_csv(selection_csv, selection_rows)
    write_csv(weighted_selection_csv, weighted_selection_rows)
    write_csv(per_fix_csv, per_fix_rows)
    write_csv(weighted_per_fix_csv, weighted_per_fix_rows)
    write_csv(group_csv, group_rows)
    write_csv(weighted_group_csv, weighted_group_rows)
    write_csv(contrast_csv, contrast_rows)
    write_csv(weighted_contrast_csv, weighted_contrast_rows)
    write_csv(combined_csv, combined_rows)
    write_csv(weighted_combined_csv, weighted_combined_rows)
    write_csv(group_delta_csv, group_delta_rows)
    write_csv(combined_delta_csv, combined_delta_rows)
    write_csv(orientation_pool_selection_csv, orientation_pool_selection_rows)
    write_csv(orientation_pool_per_fix_csv, orientation_pool_per_fix_rows)
    write_csv(orientation_pool_csv, orientation_pool_rows)
    write_csv(endpoint_csv, endpoint_rows)
    write_csv(units_csv, units.to_dict(orient="records"))
    write_csv(sf_definition_csv, sf_definition_rows)

    group_summary = pd.DataFrame(group_rows)
    combined_summary = pd.DataFrame(combined_rows)
    weighted_group_summary = pd.DataFrame(weighted_group_rows)
    weighted_combined_summary = pd.DataFrame(weighted_combined_rows)
    group_delta_summary = pd.DataFrame(group_delta_rows)
    combined_delta_summary = pd.DataFrame(combined_delta_rows)
    orientation_pool_summary = pd.DataFrame(orientation_pool_rows)
    selection = pd.DataFrame(selection_rows)
    weighted_selection = pd.DataFrame(weighted_selection_rows)
    metric_name = metric_label(str(args.ssi_metric))
    if high_sf_min_cpd is not None:
        metric_name = f"{metric_name}; high SF >= {high_sf_min_cpd:g} cpd"
    separate_png, separate_pdf = plot_alignment_separate_population_curves(
        out_dir,
        group_summary,
        sf_groups,
        metric_name=metric_name,
        sweep_axis=sweep_axis,
        x_label=x_label,
        sweep_text=sweep_text,
        dpi=int(args.dpi),
    )
    curve_png, curve_pdf = plot_population_curves(
        out_dir,
        group_summary,
        combined_summary,
        sf_groups,
        metric_name=metric_name,
        sweep_axis=sweep_axis,
        x_label=x_label,
        sweep_text=sweep_text,
        dpi=int(args.dpi),
    )
    weighted_separate_png, weighted_separate_pdf = plot_alignment_separate_population_curves(
        out_dir,
        weighted_group_summary,
        sf_groups,
        metric_name=metric_name,
        sweep_axis=sweep_axis,
        x_label=x_label,
        sweep_text=sweep_text,
        dpi=int(args.dpi),
        output_stem="backimage_rr100_sf_contour_alignment_weighted_separate_population_curves",
        title_prefix="BackImage RR100 accumulated SSI by continuous local contour-alignment weights",
        alignment_title_suffix="weighted pool",
    )
    weighted_curve_png, weighted_curve_pdf = plot_population_curves(
        out_dir,
        weighted_group_summary,
        weighted_combined_summary,
        sf_groups,
        metric_name=metric_name,
        sweep_axis=sweep_axis,
        x_label=x_label,
        sweep_text=sweep_text,
        dpi=int(args.dpi),
        output_stem="backimage_rr100_sf_contour_alignment_weighted_population_curves",
        title_prefix="BackImage RR100 SF groups with continuous per-fixation contour-alignment weights",
    )
    delta_png = delta_pdf = None
    if not group_delta_summary.empty and not combined_delta_summary.empty:
        delta_png, delta_pdf = plot_delta_from_reference_curves(
            out_dir,
            group_delta_summary,
            combined_delta_summary,
            sf_groups,
            reference_condition_id=ref_condition_id,
            metric_name=metric_name,
            sweep_axis=sweep_axis,
            x_label=x_label,
            sweep_text=sweep_text,
            dpi=int(args.dpi),
        )
    counts_png, counts_pdf = plot_selection_counts(out_dir, selection, sf_groups, dpi=int(args.dpi))
    weighted_counts_png, weighted_counts_pdf = plot_selection_counts(
        out_dir,
        weighted_selection,
        sf_groups,
        dpi=int(args.dpi),
        output_stem="backimage_rr100_sf_contour_alignment_weighted_selection_counts",
        title_prefix="Continuous-weight alignment",
        y_label="positive-weight units per fixation",
    )
    endpoint_png, endpoint_pdf = plot_endpoint_distributions(
        out_dir,
        per_fix_df,
        sf_groups,
        ref_condition_id,
        end_condition_id,
        dpi=int(args.dpi),
    )
    orientation_pool_png, orientation_pool_pdf = plot_orientation_pool_control_curves(
        out_dir,
        orientation_pool_summary,
        sf_groups,
        metric_name=metric_name,
        sweep_axis=sweep_axis,
        x_label=x_label,
        sweep_text=sweep_text,
        dpi=int(args.dpi),
    )

    endpoint_summary = {}
    for sf_group in sf_groups:
        endpoint_summary[str(sf_group)] = {
            "aligned": best_row(endpoint_rows, sf_group, "contour_aligned"),
            "orthogonal": best_row(endpoint_rows, sf_group, "contour_orthogonal"),
            "aligned_minus_orthogonal": best_row(endpoint_rows, sf_group, "aligned_minus_orthogonal"),
        }

    write_json(
        out_dir / "summary.json",
        {
            "analysis": "backimage_rr100_sf_contour_alignment_population_ssi",
            "contour_run_dir": Path(args.contour_run_dir),
            "sf_groups_csv": Path(args.sf_groups_csv),
            "out_dir": out_dir,
            "cache_path": cache_path(Path(args.contour_run_dir)),
            "n_conditions": int(n_conditions),
            "n_conditions_after_sweep_filter": int(conditions.shape[0]),
            "n_movies": int(n_movies),
            "n_units": int(n_units),
            "ssi_metric": str(args.ssi_metric),
            "ssi_metric_cache_key": metric_key,
            "ssi_metric_label": metric_name,
            "ssi_metric_contract": metric_contract(str(args.ssi_metric)),
            "sf_groups": sf_groups,
            "high_sf_min_cpd_override": high_sf_min_cpd,
            "sf_group_definition_contract": (
                "Source table SF groups are rank-based tertiles of sf_split_metric unless high_sf_min_cpd_override "
                "is set. With the override, high_sf contains units with sf_split_metric >= threshold; original "
                "high_sf units below threshold are labeled high_sf_below_threshold and are excluded unless requested."
            ),
            "sf_group_definitions": sf_definition_rows,
            "alignment_groups": ALIGNMENT_GROUPS,
            "weighted_alignment_groups": ALIGNMENT_GROUPS,
            "weighted_alignment_contract": (
                "Continuous alignment pools keep the SF-defined unit universe fixed and assign per-fixation "
                "fractional weights from cos(2 * (preferred_orientation - contour_axis)). contour_aligned uses "
                "max(score, 0); contour_orthogonal uses max(-score, 0). Units below the orientation-selectivity "
                "threshold or without finite preferred orientation get zero weight. Population SSI sums "
                "unit_bits * expected_spikes * weight and expected_spikes * weight, then takes bits/spike."
            ),
            "orientation_control_pools": ORIENTATION_POOLS,
            "orientation_control_pool_contract": (
                "all_sf_units keeps every unit in the SF group; target_oriented pools contour-aligned and "
                "contour-orthogonal orientation-tuned units; off_axis_tuned keeps orientation-tuned units outside "
                "those target windows; low_osi keeps units below the orientation-selectivity threshold."
            ),
            "primary_plot_contract": (
                "Primary 2x2 figure shows separate aligned/orthogonal accumulated population curves on the left "
                "and an aligned-plus-orthogonal pooled population on the right. Left curves apply the minimum-unit "
                "support check to each alignment pool separately, so their legends report fixation support and "
                "low-support curves are dashed. The pooled curve first sums information numerators, expected-spike "
                "denominators, and selected units across alignment pools, then applies the minimum-unit support check "
                "and takes the bits/spike ratio."
            ),
            "alignment_angle_deg": float(args.alignment_angle_deg),
            "min_orientation_selectivity": float(args.min_orientation_selectivity),
            "min_units_per_fixation_group": int(args.min_units_per_fixation_group),
            "axis_coordinate_frame": str(args.axis_coordinate_frame),
            "sweep_axis": sweep_axis,
            "fixed_along_scale": None if args.fixed_along_scale is None else float(args.fixed_along_scale),
            "fixed_across_scale": None if args.fixed_across_scale is None else float(args.fixed_across_scale),
            "x_axis_label": x_label,
            "sweep_description": sweep_text,
            "axis_frame_contract": (
                "If axis_coordinate_frame is gaze, axis_deg is converted to image-array coordinates as -axis_deg mod 180 "
                "before comparison to prior_preferred_orientation_deg."
            ),
            "reference_condition_id": ref_condition_id,
            "endpoint_condition_id": end_condition_id,
            "n_bootstrap": int(args.n_bootstrap),
            "bootstrap_seed": int(args.bootstrap_seed),
            "csv_outputs": {
                "selection": selection_csv,
                "weighted_alignment_selection": weighted_selection_csv,
                "per_fixation": per_fix_csv,
                "weighted_alignment_per_fixation": weighted_per_fix_csv,
                "group_summary": group_csv,
                "weighted_alignment_group_summary": weighted_group_csv,
                "alignment_contrast": contrast_csv,
                "weighted_alignment_contrast": weighted_contrast_csv,
                "alignment_combined": combined_csv,
                "weighted_alignment_combined": weighted_combined_csv,
                "group_delta_vs_reference": group_delta_csv,
                "alignment_combined_delta_vs_reference": combined_delta_csv,
                "orientation_pool_selection": orientation_pool_selection_csv,
                "orientation_pool_per_fixation": orientation_pool_per_fix_csv,
                "orientation_pool_summary": orientation_pool_csv,
                "endpoint_delta": endpoint_csv,
                "unit_table": units_csv,
                "sf_group_definition_summary": sf_definition_csv,
            },
            "figure_outputs": {
                "separate_population_curves_png": separate_png,
                "separate_population_curves_pdf": separate_pdf,
                "population_curves_png": curve_png,
                "population_curves_pdf": curve_pdf,
                "delta_vs_reference_curves_png": delta_png,
                "delta_vs_reference_curves_pdf": delta_pdf,
                "selection_counts_png": counts_png,
                "selection_counts_pdf": counts_pdf,
                "weighted_separate_population_curves_png": weighted_separate_png,
                "weighted_separate_population_curves_pdf": weighted_separate_pdf,
                "weighted_population_curves_png": weighted_curve_png,
                "weighted_population_curves_pdf": weighted_curve_pdf,
                "weighted_selection_counts_png": weighted_counts_png,
                "weighted_selection_counts_pdf": weighted_counts_pdf,
                "endpoint_distributions_png": endpoint_png,
                "endpoint_distributions_pdf": endpoint_pdf,
                "orientation_pool_control_png": orientation_pool_png,
                "orientation_pool_control_pdf": orientation_pool_pdf,
            },
            "endpoint_summary": endpoint_summary,
        },
    )
    print(f"Wrote {selection_csv}")
    print(f"Wrote {weighted_selection_csv}")
    print(f"Wrote {per_fix_csv}")
    print(f"Wrote {weighted_per_fix_csv}")
    print(f"Wrote {group_csv}")
    print(f"Wrote {weighted_group_csv}")
    print(f"Wrote {contrast_csv}")
    print(f"Wrote {weighted_contrast_csv}")
    print(f"Wrote {combined_csv}")
    print(f"Wrote {weighted_combined_csv}")
    print(f"Wrote {group_delta_csv}")
    print(f"Wrote {combined_delta_csv}")
    print(f"Wrote {orientation_pool_selection_csv}")
    print(f"Wrote {orientation_pool_per_fix_csv}")
    print(f"Wrote {orientation_pool_csv}")
    print(f"Wrote {endpoint_csv}")
    print(f"Wrote {sf_definition_csv}")
    print(f"Wrote {separate_png}")
    print(f"Wrote {curve_png}")
    print(f"Wrote {weighted_separate_png}")
    print(f"Wrote {weighted_curve_png}")
    if delta_png is not None:
        print(f"Wrote {delta_png}")
    print(f"Wrote {counts_png}")
    print(f"Wrote {weighted_counts_png}")
    print(f"Wrote {endpoint_png}")
    print(f"Wrote {orientation_pool_png}")


if __name__ == "__main__":
    main()
