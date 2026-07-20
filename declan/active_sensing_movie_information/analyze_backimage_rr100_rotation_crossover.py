#!/usr/bin/env python3
"""Paired original/rot90 crossover diagnostics for BackImage RR100 contour SSI.

The population alignment plots label units as contour-aligned or orthogonal on
each fixation.  A 90-degree whole-movie rotation swaps those labels for a fixed
unit while preserving the contour-relative motion geometry.  This script uses
that paired structure to separate fixed absolute-orientation pool anisotropy
from a possible contour-relative alignment component.
"""

from __future__ import annotations

import argparse
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
DEFAULT_ORIGINAL_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_v1/"
    "contour_rr100_spatial_ssi_pairs27"
)
DEFAULT_ROT90_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_rot90_v1/"
    "contour_rr100_spatial_ssi_pairs27"
)
DEFAULT_SF_GROUPS_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
    "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)
DEFAULT_OUT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_rot90_v1/"
    "rotation_crossover_diagnostics"
)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run-dir", type=Path, default=DEFAULT_ORIGINAL_RUN_DIR)
    parser.add_argument("--rot90-run-dir", type=Path, default=DEFAULT_ROT90_RUN_DIR)
    parser.add_argument("--sf-groups-csv", type=Path, default=DEFAULT_SF_GROUPS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sf-groups", type=str, default="low_sf,high_sf")
    parser.add_argument("--alignment-angle-deg", type=float, default=22.5)
    parser.add_argument("--min-orientation-selectivity", type=float, default=0.05)
    parser.add_argument("--axis-coordinate-frame", choices=("gaze", "image"), default="gaze")
    parser.add_argument("--metric", choices=("time_resolved", "mean_map"), default="time_resolved")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=19)
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
    pd.DataFrame(rows).to_csv(path, index=False)


def orientation_axis_180(angle_deg: float | np.ndarray) -> np.ndarray:
    return np.asarray(angle_deg, dtype=np.float64) % 180.0


def angle_180_distance(a_deg: float | np.ndarray, b_deg: float | np.ndarray) -> np.ndarray:
    return np.abs(((np.asarray(a_deg, dtype=np.float64) - np.asarray(b_deg, dtype=np.float64) + 90.0) % 180.0) - 90.0)


def contour_axis_to_image_frame(axis_deg: float | np.ndarray, coordinate_frame: str) -> np.ndarray:
    axis = np.asarray(axis_deg, dtype=np.float64)
    if str(coordinate_frame) == "gaze":
        axis = -axis
    return orientation_axis_180(axis)


def cache_path(run_dir: Path) -> Path:
    return Path(run_dir) / "cache" / "backimage_contour_axis_rr100_spatial_ssi_cache.npz"


def load_cache(run_dir: Path) -> dict[str, np.ndarray]:
    path = cache_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files if key != "cache_identity_json"}


def metric_arrays(stats: dict[str, np.ndarray], metric: str) -> tuple[np.ndarray, np.ndarray]:
    if metric == "time_resolved":
        bits = np.asarray(stats["unit_time_resolved_bits_per_movie"], dtype=np.float64)
    elif metric == "mean_map":
        bits = np.asarray(stats["unit_mean_map_bits_per_movie"], dtype=np.float64)
    else:
        raise ValueError(metric)
    spikes = np.asarray(stats["unit_expected_spikes_per_movie"], dtype=np.float64)
    return bits, spikes


def condition_frame(stats: dict[str, np.ndarray]) -> pd.DataFrame:
    n = int(np.asarray(stats["condition_id"]).size)
    along = np.asarray(stats["along_scale"], dtype=np.float64)
    across = np.asarray(stats["across_scale"], dtype=np.float64)
    motion = np.asarray(stats.get("motion_scale", across), dtype=np.float64)
    motion = np.where(np.isfinite(motion), motion, np.maximum(np.abs(along), np.abs(across)))
    return pd.DataFrame(
        {
            "condition_index": np.arange(n, dtype=int),
            "condition_id": np.asarray(stats["condition_id"]).astype(str),
            "condition_label": np.asarray(stats["condition_label"]).astype(str),
            "along_scale": along,
            "across_scale": across,
            "motion_scale": motion,
            "sweep_mode": np.asarray(stats.get("sweep_mode", np.asarray(["pairs"] * n))).astype(str),
            "is_static_baseline": np.asarray(stats["is_static_baseline"], dtype=bool),
            "is_across_sweep": np.asarray(stats["is_across_sweep"], dtype=bool),
        }
    )


def load_movie_frame(run_dir: Path, stats: dict[str, np.ndarray], axis_coordinate_frame: str) -> pd.DataFrame:
    inventory_path = Path(run_dir) / "movie_condition_inventory.csv"
    inventory = pd.read_csv(inventory_path)
    movies = inventory.sort_values(["movie_index", "condition_index"]).drop_duplicates("movie_index", keep="first").copy()
    n_movies = int(np.asarray(stats["movie_source_row"]).size)
    movies = movies.set_index("movie_index", drop=False).reindex(np.arange(n_movies))
    if movies["source_row"].isna().any():
        raise ValueError(f"{inventory_path} is missing one or more movie_index rows.")
    if not np.array_equal(np.asarray(stats["movie_source_row"], dtype=int), movies["source_row"].to_numpy(dtype=int)):
        raise ValueError(f"Cache source-row order does not match {inventory_path}.")
    movies["axis_deg"] = pd.to_numeric(movies["axis_deg"], errors="coerce")
    movies["contour_axis_image_deg"] = contour_axis_to_image_frame(
        movies["axis_deg"].to_numpy(dtype=np.float64),
        axis_coordinate_frame,
    )
    movies["orthogonal_axis_image_deg"] = orientation_axis_180(movies["contour_axis_image_deg"].to_numpy(dtype=float) + 90.0)
    return movies.reset_index(drop=True)


def load_units(path: Path, sf_groups: list[str], n_units: int) -> pd.DataFrame:
    units = pd.read_csv(path).copy()
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
    units["unit_index"] = pd.to_numeric(units["unit_index"], errors="coerce").astype(int)
    units = units[(units["unit_index"] >= 0) & (units["unit_index"] < int(n_units))]
    units = units[units["sf_group"].astype(str).isin(sf_groups)].copy()
    units["preferred_orientation_image_deg"] = orientation_axis_180(
        pd.to_numeric(units["prior_preferred_orientation_deg"], errors="coerce").to_numpy(dtype=float)
    )
    units["prior_orientation_selectivity_index"] = pd.to_numeric(units["prior_orientation_selectivity_index"], errors="coerce")
    units["sf_split_metric"] = pd.to_numeric(units["sf_split_metric"], errors="coerce")
    return units.sort_values(["sf_group", "sf_rank_low_to_high", "unit_index"]).reset_index(drop=True)


def selection_masks(
    movies: pd.DataFrame,
    units: pd.DataFrame,
    sf_groups: list[str],
    *,
    alignment_angle_deg: float,
    min_orientation_selectivity: float,
) -> dict[str, dict[str, list[np.ndarray]]]:
    masks: dict[str, dict[str, list[np.ndarray]]] = {
        sf: {"aligned": [], "orthogonal": []} for sf in sf_groups
    }
    for _, movie in movies.iterrows():
        contour = float(movie["contour_axis_image_deg"])
        orthogonal = float(movie["orthogonal_axis_image_deg"])
        for sf_group in sf_groups:
            sub = units[units["sf_group"].astype(str) == str(sf_group)]
            pref = sub["preferred_orientation_image_deg"].to_numpy(dtype=np.float64)
            osi = sub["prior_orientation_selectivity_index"].to_numpy(dtype=np.float64)
            valid = np.isfinite(pref) & np.isfinite(osi) & (osi >= float(min_orientation_selectivity))
            aligned = valid & (angle_180_distance(pref, contour) <= float(alignment_angle_deg))
            orth = valid & (angle_180_distance(pref, orthogonal) <= float(alignment_angle_deg))
            masks[sf_group]["aligned"].append(sub.loc[aligned, "unit_index"].to_numpy(dtype=int))
            masks[sf_group]["orthogonal"].append(sub.loc[orth, "unit_index"].to_numpy(dtype=int))
    return masks


def nanmean_or_nan(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    ok = np.isfinite(arr)
    if not np.any(ok):
        return float("nan")
    return float(np.nanmean(arr[ok]))


def weighted_mean_or_nan(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(ok):
        return float("nan")
    return float(np.sum(values[ok] * weights[ok]) / max(np.sum(weights[ok]), EPS))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    if int(n_boot) <= 0 or arr.size <= 1:
        return mean, sem, float("nan"), float("nan")
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    boot = np.mean(arr[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, sem, float(lo), float(hi)


def fit_harmonic(theta_deg: np.ndarray, gap: np.ndarray) -> dict[str, float]:
    theta = np.asarray(theta_deg, dtype=np.float64)
    y = np.asarray(gap, dtype=np.float64)
    ok = np.isfinite(theta) & np.isfinite(y)
    theta = theta[ok]
    y = y[ok]
    if y.size < 3:
        return {
            "n_fixations": int(y.size),
            "gamma_bits_per_spike": float("nan"),
            "cos2_coeff": float("nan"),
            "sin2_coeff": float("nan"),
            "anisotropy_amplitude_bits_per_spike": float("nan"),
            "preferred_gap_axis_deg": float("nan"),
            "r2": float("nan"),
        }
    rad2 = np.deg2rad(2.0 * theta)
    x = np.column_stack([np.ones_like(theta), np.cos(rad2), np.sin(rad2)])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    amp = float(math.hypot(float(coef[1]), float(coef[2])))
    phase = float((0.5 * np.rad2deg(math.atan2(float(coef[2]), float(coef[1])))) % 180.0)
    return {
        "n_fixations": int(y.size),
        "gamma_bits_per_spike": float(coef[0]),
        "cos2_coeff": float(coef[1]),
        "sin2_coeff": float(coef[2]),
        "anisotropy_amplitude_bits_per_spike": amp,
        "preferred_gap_axis_deg": phase,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > EPS else float("nan"),
    }


def harmonic_bootstrap(
    theta: np.ndarray,
    gap: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, float]:
    base = fit_harmonic(theta, gap)
    ok = np.isfinite(theta) & np.isfinite(gap)
    theta_ok = np.asarray(theta, dtype=np.float64)[ok]
    gap_ok = np.asarray(gap, dtype=np.float64)[ok]
    if int(n_boot) <= 0 or gap_ok.size <= 3:
        return base
    gamma = []
    amp = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, gap_ok.size, size=gap_ok.size)
        fit = fit_harmonic(theta_ok[idx], gap_ok[idx])
        gamma.append(fit["gamma_bits_per_spike"])
        amp.append(fit["anisotropy_amplitude_bits_per_spike"])
    for key, vals in [
        ("gamma", gamma),
        ("anisotropy_amplitude", amp),
    ]:
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            lo, hi = np.percentile(arr, [2.5, 97.5])
            base[f"{key}_ci_low"] = float(lo)
            base[f"{key}_ci_high"] = float(hi)
    return base


def build_crossover_rows(
    conditions: pd.DataFrame,
    movies: pd.DataFrame,
    units: pd.DataFrame,
    orig_bits: np.ndarray,
    orig_spikes: np.ndarray,
    rot_bits: np.ndarray,
    rot_spikes: np.ndarray,
    masks: dict[str, dict[str, list[np.ndarray]]],
    sf_groups: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_conditions, n_movies, _ = orig_bits.shape
    for ci in range(n_conditions):
        cond = conditions.iloc[ci]
        for movie_idx in range(n_movies):
            movie = movies.iloc[movie_idx]
            for sf_group in sf_groups:
                a_idx = masks[sf_group]["aligned"][movie_idx]
                o_idx = masks[sf_group]["orthogonal"][movie_idx]
                if a_idx.size == 0 or o_idx.size == 0:
                    continue
                ob_a = orig_bits[ci, movie_idx, a_idx]
                rb_a = rot_bits[ci, movie_idx, a_idx]
                os_a = orig_spikes[ci, movie_idx, a_idx]
                rs_a = rot_spikes[ci, movie_idx, a_idx]
                ob_o = orig_bits[ci, movie_idx, o_idx]
                rb_o = rot_bits[ci, movie_idx, o_idx]
                os_o = orig_spikes[ci, movie_idx, o_idx]
                rs_o = rot_spikes[ci, movie_idx, o_idx]

                mean_orig_a = nanmean_or_nan(ob_a)
                mean_rot_a = nanmean_or_nan(rb_a)
                mean_orig_o = nanmean_or_nan(ob_o)
                mean_rot_o = nanmean_or_nan(rb_o)
                spike_orig_a = weighted_mean_or_nan(ob_a, os_a)
                spike_rot_a = weighted_mean_or_nan(rb_a, rs_a)
                spike_orig_o = weighted_mean_or_nan(ob_o, os_o)
                spike_rot_o = weighted_mean_or_nan(rb_o, rs_o)
                rows.append(
                    {
                        "condition_index": int(ci),
                        "condition_id": str(cond.condition_id),
                        "condition_label": str(cond.condition_label),
                        "along_scale": float(cond.along_scale),
                        "across_scale": float(cond.across_scale),
                        "motion_scale": float(cond.motion_scale),
                        "is_static_baseline": bool(cond.is_static_baseline),
                        "movie_index": int(movie_idx),
                        "source_row": int(movie.source_row),
                        "session": str(movie.session),
                        "trial_idx": int(movie.trial_idx),
                        "contour_axis_image_deg_original": float(movie.contour_axis_image_deg),
                        "sf_group": str(sf_group),
                        "n_aligned_units_original_label": int(a_idx.size),
                        "n_orthogonal_units_original_label": int(o_idx.size),
                        "mean_orig_aligned_label": mean_orig_a,
                        "mean_rot_aligned_label": mean_rot_a,
                        "mean_orig_orthogonal_label": mean_orig_o,
                        "mean_rot_orthogonal_label": mean_rot_o,
                        "spike_weighted_orig_aligned_label": spike_orig_a,
                        "spike_weighted_rot_aligned_label": spike_rot_a,
                        "spike_weighted_orig_orthogonal_label": spike_orig_o,
                        "spike_weighted_rot_orthogonal_label": spike_rot_o,
                        "expected_spikes_orig_aligned_label": float(np.nansum(os_a)),
                        "expected_spikes_rot_aligned_label": float(np.nansum(rs_a)),
                        "expected_spikes_orig_orthogonal_label": float(np.nansum(os_o)),
                        "expected_spikes_rot_orthogonal_label": float(np.nansum(rs_o)),
                        "crossover_equal_unit_bits_per_spike": 0.5
                        * ((mean_orig_a - mean_rot_a) + (mean_rot_o - mean_orig_o)),
                        "crossover_spike_weighted_within_fix_bits_per_spike": 0.5
                        * ((spike_orig_a - spike_rot_a) + (spike_rot_o - spike_orig_o)),
                    }
                )
    return rows


def summarize_crossover(per_fix: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    static = (
        per_fix[per_fix["is_static_baseline"]]
        .set_index(["movie_index", "sf_group"])[
            ["crossover_equal_unit_bits_per_spike", "crossover_spike_weighted_within_fix_bits_per_spike"]
        ]
        .rename(
            columns={
                "crossover_equal_unit_bits_per_spike": "static_equal",
                "crossover_spike_weighted_within_fix_bits_per_spike": "static_spike",
            }
        )
    )
    rows: list[dict[str, Any]] = []
    for keys, sub in per_fix.groupby(["condition_index", "condition_id", "along_scale", "across_scale", "motion_scale", "sf_group"], sort=True):
        ci, condition_id, along, across, motion, sf_group = keys
        merged = sub.join(static, on=["movie_index", "sf_group"], how="left")
        equal = merged["crossover_equal_unit_bits_per_spike"].to_numpy(dtype=np.float64)
        spike = merged["crossover_spike_weighted_within_fix_bits_per_spike"].to_numpy(dtype=np.float64)
        delta_equal = equal - merged["static_equal"].to_numpy(dtype=np.float64)
        delta_spike = spike - merged["static_spike"].to_numpy(dtype=np.float64)
        mean, sem, lo, hi = bootstrap_ci(equal, rng, n_boot)
        smean, ssem, slo, shi = bootstrap_ci(spike, rng, n_boot)
        dmean, dsem, dlo, dhi = bootstrap_ci(delta_equal, rng, n_boot)
        dsmean, dssem, dslo, dshi = bootstrap_ci(delta_spike, rng, n_boot)
        rows.append(
            {
                "condition_index": int(ci),
                "condition_id": str(condition_id),
                "along_scale": float(along),
                "across_scale": float(across),
                "motion_scale": float(motion),
                "sf_group": str(sf_group),
                "n_fixations": int(np.isfinite(equal).sum()),
                "crossover_equal_unit_mean": mean,
                "crossover_equal_unit_sem": sem,
                "crossover_equal_unit_ci_low": lo,
                "crossover_equal_unit_ci_high": hi,
                "crossover_spike_weighted_mean": smean,
                "crossover_spike_weighted_sem": ssem,
                "crossover_spike_weighted_ci_low": slo,
                "crossover_spike_weighted_ci_high": shi,
                "delta_vs_static_equal_unit_mean": dmean,
                "delta_vs_static_equal_unit_sem": dsem,
                "delta_vs_static_equal_unit_ci_low": dlo,
                "delta_vs_static_equal_unit_ci_high": dhi,
                "delta_vs_static_spike_weighted_mean": dsmean,
                "delta_vs_static_spike_weighted_sem": dssem,
                "delta_vs_static_spike_weighted_ci_low": dslo,
                "delta_vs_static_spike_weighted_ci_high": dshi,
            }
        )
    return pd.DataFrame(rows).sort_values(["sf_group", "condition_index"]).reset_index(drop=True)


def build_pool_decomposition(
    run_label: str,
    conditions: pd.DataFrame,
    movies: pd.DataFrame,
    bits: np.ndarray,
    spikes: np.ndarray,
    masks: dict[str, dict[str, list[np.ndarray]]],
    sf_groups: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ci, cond in conditions.iterrows():
        for sf_group in sf_groups:
            for alignment_label, key in [("contour_aligned", "aligned"), ("contour_orthogonal", "orthogonal")]:
                numer = 0.0
                denom = 0.0
                equal_fix_values: list[float] = []
                spike_fix_values: list[float] = []
                n_units: list[int] = []
                for movie_idx in range(bits.shape[1]):
                    idx = masks[sf_group][key][movie_idx]
                    if idx.size == 0:
                        continue
                    b = bits[int(ci), movie_idx, idx]
                    s = spikes[int(ci), movie_idx, idx]
                    ok = np.isfinite(b) & np.isfinite(s) & (s > 0)
                    if not np.any(ok):
                        continue
                    numer += float(np.sum(b[ok] * s[ok]))
                    denom += float(np.sum(s[ok]))
                    equal_fix_values.append(nanmean_or_nan(b[ok]))
                    spike_fix_values.append(weighted_mean_or_nan(b[ok], s[ok]))
                    n_units.append(int(idx.size))
                equal_arr = np.asarray(equal_fix_values, dtype=np.float64)
                spike_arr = np.asarray(spike_fix_values, dtype=np.float64)
                rows.append(
                    {
                        "run": run_label,
                        "condition_index": int(ci),
                        "condition_id": str(cond.condition_id),
                        "along_scale": float(cond.along_scale),
                        "across_scale": float(cond.across_scale),
                        "motion_scale": float(cond.motion_scale),
                        "sf_group": str(sf_group),
                        "alignment_group": alignment_label,
                        "n_fixations": int(equal_arr.size),
                        "mean_n_units": float(np.mean(n_units)) if n_units else float("nan"),
                        "pooled_information_numerator": numer,
                        "pooled_expected_spikes_denominator": denom,
                        "pooled_bits_per_spike": numer / max(denom, EPS) if denom > 0 else float("nan"),
                        "equal_fixation_unit_mean_bits_per_spike": float(np.nanmean(equal_arr)) if equal_arr.size else float("nan"),
                        "spike_weighted_within_fix_mean_bits_per_spike": float(np.nanmean(spike_arr)) if spike_arr.size else float("nan"),
                    }
                )
    return rows


def build_harmonic_rows(
    run_label: str,
    conditions: pd.DataFrame,
    movies: pd.DataFrame,
    bits: np.ndarray,
    spikes: np.ndarray,
    masks: dict[str, dict[str, list[np.ndarray]]],
    sf_groups: list[str],
    n_boot: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    per_fix_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    theta = movies["contour_axis_image_deg"].to_numpy(dtype=np.float64)
    for ci, cond in conditions.iterrows():
        for sf_group in sf_groups:
            equal_gap = np.full(bits.shape[1], np.nan, dtype=np.float64)
            spike_gap = np.full(bits.shape[1], np.nan, dtype=np.float64)
            for movie_idx in range(bits.shape[1]):
                a_idx = masks[sf_group]["aligned"][movie_idx]
                o_idx = masks[sf_group]["orthogonal"][movie_idx]
                if a_idx.size == 0 or o_idx.size == 0:
                    continue
                b_a = bits[int(ci), movie_idx, a_idx]
                b_o = bits[int(ci), movie_idx, o_idx]
                s_a = spikes[int(ci), movie_idx, a_idx]
                s_o = spikes[int(ci), movie_idx, o_idx]
                equal_gap[movie_idx] = nanmean_or_nan(b_a) - nanmean_or_nan(b_o)
                spike_gap[movie_idx] = weighted_mean_or_nan(b_a, s_a) - weighted_mean_or_nan(b_o, s_o)
                per_fix_rows.append(
                    {
                        "run": run_label,
                        "condition_index": int(ci),
                        "condition_id": str(cond.condition_id),
                        "along_scale": float(cond.along_scale),
                        "across_scale": float(cond.across_scale),
                        "motion_scale": float(cond.motion_scale),
                        "sf_group": str(sf_group),
                        "movie_index": int(movie_idx),
                        "source_row": int(movies.iloc[movie_idx].source_row),
                        "contour_axis_image_deg": float(theta[movie_idx]),
                        "equal_unit_gap_bits_per_spike": float(equal_gap[movie_idx]),
                        "spike_weighted_gap_bits_per_spike": float(spike_gap[movie_idx]),
                    }
                )
            for gap_name, gap_values in [
                ("equal_unit", equal_gap),
                ("spike_weighted", spike_gap),
            ]:
                fit = harmonic_bootstrap(theta, gap_values, rng, n_boot)
                fit.update(
                    {
                        "run": run_label,
                        "condition_index": int(ci),
                        "condition_id": str(cond.condition_id),
                        "along_scale": float(cond.along_scale),
                        "across_scale": float(cond.across_scale),
                        "motion_scale": float(cond.motion_scale),
                        "sf_group": str(sf_group),
                        "gap_estimator": gap_name,
                    }
                )
                fit_rows.append(fit)
    return per_fix_rows, fit_rows


def plot_crossover(summary: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    view_specs = [
        ("across, along=0", lambda d: np.isclose(d["along_scale"], 0.0), "across_scale"),
        ("across, along=1", lambda d: np.isclose(d["along_scale"], 1.0), "across_scale"),
        ("along, across=0", lambda d: np.isclose(d["across_scale"], 0.0), "along_scale"),
        ("along, across=1", lambda d: np.isclose(d["across_scale"], 1.0), "along_scale"),
    ]
    colors = {"low_sf": "#1f77b4", "high_sf": "#d62728"}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey="row", constrained_layout=True)
    for col, (title, mask_fn, xcol) in enumerate(view_specs):
        for row, ycol in enumerate(["crossover_equal_unit_mean", "delta_vs_static_equal_unit_mean"]):
            ax = axes[row, col]
            view = summary[mask_fn(summary)].copy()
            for sf_group, sub in view.groupby("sf_group", sort=True):
                sub = sub.sort_values(xcol)
                x = sub[xcol].to_numpy(dtype=float)
                y = sub[ycol].to_numpy(dtype=float)
                if row == 0:
                    lo = sub["crossover_equal_unit_ci_low"].to_numpy(dtype=float)
                    hi = sub["crossover_equal_unit_ci_high"].to_numpy(dtype=float)
                else:
                    lo = sub["delta_vs_static_equal_unit_ci_low"].to_numpy(dtype=float)
                    hi = sub["delta_vs_static_equal_unit_ci_high"].to_numpy(dtype=float)
                ax.plot(x, y, marker="o", label=sf_group.replace("_", " "), color=colors.get(sf_group, None))
                ax.fill_between(x, lo, hi, color=colors.get(sf_group, "0.4"), alpha=0.15, linewidth=0)
            ax.axhline(0, color="0.5", lw=1, ls="--")
            ax.set_title(title)
            ax.set_xlabel(xcol.replace("_", " "))
            if col == 0:
                ax.set_ylabel("C(s)" if row == 0 else "C(s) - C(0)")
            if row == 0 and col == 0:
                ax.legend(frameon=False)
    fig.suptitle("Original/rot90 within-unit crossover: equal-unit means", fontsize=14)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"backimage_rr100_rotation_crossover_equal_unit.{ext}", dpi=dpi)
    plt.close(fig)


def plot_harmonic(fits: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    static = fits[
        (fits["condition_index"] == 0)
        & (fits["gap_estimator"] == "equal_unit")
        & (fits["sf_group"].isin(["low_sf", "high_sf"]))
    ].copy()
    if static.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric, label in [
        (axes[0], "gamma_bits_per_spike", "axis-balanced gap gamma"),
        (axes[1], "anisotropy_amplitude_bits_per_spike", "cos2 anisotropy amplitude"),
    ]:
        pivot = static.pivot(index="sf_group", columns="run", values=metric)
        pivot.plot(kind="bar", ax=ax, color=["#777777", "#d95f02"])
        ax.axhline(0, color="0.5", lw=1)
        ax.set_ylabel("bits/spike")
        ax.set_xlabel("")
        ax.set_title(label)
        ax.legend(frameon=False)
    fig.suptitle("Static harmonic fit: G(theta)=gamma+a cos2theta+b sin2theta", fontsize=12)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"backimage_rr100_rotation_harmonic_static_summary.{ext}", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf_groups = parse_csv_list(args.sf_groups)

    orig_stats = load_cache(Path(args.original_run_dir))
    rot_stats = load_cache(Path(args.rot90_run_dir))
    if not np.array_equal(orig_stats["movie_source_row"], rot_stats["movie_source_row"]):
        raise ValueError("Original and rot90 movie_source_row orders do not match.")
    if not np.array_equal(orig_stats["condition_id"].astype(str), rot_stats["condition_id"].astype(str)):
        raise ValueError("Original and rot90 condition orders do not match.")

    conditions = condition_frame(orig_stats)
    orig_movies = load_movie_frame(Path(args.original_run_dir), orig_stats, str(args.axis_coordinate_frame))
    rot_movies = load_movie_frame(Path(args.rot90_run_dir), rot_stats, str(args.axis_coordinate_frame))
    units = load_units(Path(args.sf_groups_csv), sf_groups, int(orig_stats["unit_bits_per_movie"].shape[2]))
    orig_bits, orig_spikes = metric_arrays(orig_stats, str(args.metric))
    rot_bits, rot_spikes = metric_arrays(rot_stats, str(args.metric))

    original_masks = selection_masks(
        orig_movies,
        units,
        sf_groups,
        alignment_angle_deg=float(args.alignment_angle_deg),
        min_orientation_selectivity=float(args.min_orientation_selectivity),
    )
    rot_masks = selection_masks(
        rot_movies,
        units,
        sf_groups,
        alignment_angle_deg=float(args.alignment_angle_deg),
        min_orientation_selectivity=float(args.min_orientation_selectivity),
    )

    crossover_rows = build_crossover_rows(
        conditions,
        orig_movies,
        units,
        orig_bits,
        orig_spikes,
        rot_bits,
        rot_spikes,
        original_masks,
        sf_groups,
    )
    per_fix = pd.DataFrame(crossover_rows)
    per_fix.to_csv(out_dir / "per_fixation_rotation_crossover.csv", index=False)
    crossover_summary = summarize_crossover(per_fix, int(args.n_bootstrap), int(args.bootstrap_seed))
    crossover_summary.to_csv(out_dir / "condition_rotation_crossover_summary.csv", index=False)

    pool_rows = []
    pool_rows.extend(build_pool_decomposition("original", conditions, orig_movies, orig_bits, orig_spikes, original_masks, sf_groups))
    pool_rows.extend(build_pool_decomposition("rot90", conditions, rot_movies, rot_bits, rot_spikes, rot_masks, sf_groups))
    pd.DataFrame(pool_rows).to_csv(out_dir / "condition_pool_decomposition.csv", index=False)

    harmonic_per_fix: list[dict[str, Any]] = []
    harmonic_fit_rows: list[dict[str, Any]] = []
    rows, fits = build_harmonic_rows(
        "original",
        conditions,
        orig_movies,
        orig_bits,
        orig_spikes,
        original_masks,
        sf_groups,
        int(args.n_bootstrap),
        int(args.bootstrap_seed) + 101,
    )
    harmonic_per_fix.extend(rows)
    harmonic_fit_rows.extend(fits)
    rows, fits = build_harmonic_rows(
        "rot90",
        conditions,
        rot_movies,
        rot_bits,
        rot_spikes,
        rot_masks,
        sf_groups,
        int(args.n_bootstrap),
        int(args.bootstrap_seed) + 202,
    )
    harmonic_per_fix.extend(rows)
    harmonic_fit_rows.extend(fits)
    pd.DataFrame(harmonic_per_fix).to_csv(out_dir / "per_fixation_harmonic_alignment_gaps.csv", index=False)
    harmonic_fits = pd.DataFrame(harmonic_fit_rows)
    harmonic_fits.to_csv(out_dir / "condition_harmonic_alignment_gap_fits.csv", index=False)

    plot_crossover(crossover_summary, out_dir, int(args.dpi))
    plot_harmonic(harmonic_fits, out_dir, int(args.dpi))

    write_json(
        out_dir / "summary.json",
        {
            "analysis": "backimage_rr100_rotation_crossover",
            "original_run_dir": Path(args.original_run_dir),
            "rot90_run_dir": Path(args.rot90_run_dir),
            "sf_groups_csv": Path(args.sf_groups_csv),
            "sf_groups": sf_groups,
            "metric": str(args.metric),
            "alignment_angle_deg": float(args.alignment_angle_deg),
            "min_orientation_selectivity": float(args.min_orientation_selectivity),
            "axis_coordinate_frame": str(args.axis_coordinate_frame),
            "n_bootstrap": int(args.n_bootstrap),
            "bootstrap_seed": int(args.bootstrap_seed),
            "n_conditions": int(conditions.shape[0]),
            "n_fixations": int(orig_bits.shape[1]),
            "n_units": int(orig_bits.shape[2]),
            "outputs": {
                "per_fixation_rotation_crossover": out_dir / "per_fixation_rotation_crossover.csv",
                "condition_rotation_crossover_summary": out_dir / "condition_rotation_crossover_summary.csv",
                "condition_pool_decomposition": out_dir / "condition_pool_decomposition.csv",
                "per_fixation_harmonic_alignment_gaps": out_dir / "per_fixation_harmonic_alignment_gaps.csv",
                "condition_harmonic_alignment_gap_fits": out_dir / "condition_harmonic_alignment_gap_fits.csv",
                "crossover_plot": out_dir / "backimage_rr100_rotation_crossover_equal_unit.png",
                "harmonic_plot": out_dir / "backimage_rr100_rotation_harmonic_static_summary.png",
            },
            "contract": (
                "C_f(s)=0.5*((orig-rot) over units aligned to the original contour + "
                "(rot-orig) over units orthogonal to the original contour). The original and "
                "rot90 fixations are bootstrapped as paired source windows."
            ),
        },
    )
    print(f"Wrote rotation crossover diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
