#!/usr/bin/env python3
"""Generate a random BackImage patch x eye-trajectory RR100 SSI matrix.

This runner implements the collaborator-facing data object:

    movies x units SSI matrix
    movies x conditioning-feature table
    units x population-metadata table

The movie grid is a Cartesian product of high-contrast BackImage patches and
native, center-cropped eye-trace snippets.  It deliberately avoids scaling or
temporal compression of the measured traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    RR100_MOVIE_MEDOID_VERSION,
    build_native_snippet_trace_bank,
    trace_bank_metadata_row,
    unit_spatial_ssi_for_movie,
)
from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import _session_dataset_cache
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
    _align_response_to_trace,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _standardize_uint_like,
    _trace_xy_to_twin_helper_order,
)
from declan.redundancy_resolved_v1_population import (
    apply_population_view,
    full_population_view,
    load_population_view,
)


DEFAULT_WINDOWS_CSV = (
    ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_structure_reviewed_v2_screenfiltered_yfix"
    / "backimage_image_fem_windows.csv"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "active_sensing_movie_information"
    / "backimage_random_highcontrast_patch_trace_rr100_ssi_matrix_n100_p1000_t32_v1"
)
DEFAULT_UNIT_TUNING_CSV = (
    ROOT
    / "outputs"
    / "active_sensing_movie_information"
    / "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1"
    / "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1"
    / "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)
EPS = 1e-8
IMAGE_FEATURE_COLUMNS = [
    "image_patch_fraction_inside_image",
    "image_patch_fraction_background",
    "image_patch_distance_to_image_border_px",
    "image_patch_mean",
    "image_patch_std",
    "image_patch_rms_contrast",
    "image_gradient_energy",
    "image_edge_density",
    "image_orientation_coherence",
    "image_gradient_axis_deg",
    "image_edge_axis_deg",
    "image_gradient_orientation_deg",
    "image_edge_orientation_deg",
    "image_dominant_orientation_deg",
    "image_spectrum_anisotropy",
    "image_spectrum_orientation_deg",
    "image_high_freq_power_fraction",
    "image_power_0_2_cpd_fraction",
    "image_power_2_4_cpd_fraction",
    "image_power_4_8_cpd_fraction",
    "image_power_8plus_cpd_fraction",
]
PATCH_ID_COLUMNS = [
    "source_row",
    "session",
    "stimulus",
    "regime",
    "trial_idx",
    "global_start",
    "global_stop",
    "local_start",
    "local_stop",
    "duration_s",
    "mean_x_deg",
    "mean_y_deg",
]
TRACE_MANIFEST_NUMERIC_COLUMNS = [
    "snippet_n_samples",
    "snippet_duration_s",
    "observed_rms_arcmin",
    "rendered_rms_radius_arcmin",
    "rendered_max_radius_deg",
    "rendered_path_length_arcmin",
    "rendered_path_speed_arcmin_s",
    "rendered_speed_mean_arcmin_s",
    "rendered_speed_median_arcmin_s",
    "rendered_speed_p95_arcmin_s",
    "rendered_diffusion_constant_arcmin2_s",
    "rendered_position_autocorr_lag1",
    "rendered_velocity_autocorr_lag1",
    "source_cov_major_sd_arcmin",
    "source_cov_minor_sd_arcmin",
    "source_cov_axis_ratio",
    "source_cov_orientation_deg",
    "source_bcea68_arcmin2",
    "rendered_cov_major_sd_arcmin",
    "rendered_cov_minor_sd_arcmin",
    "rendered_cov_axis_ratio",
    "rendered_cov_orientation_deg",
    "rendered_bcea68_arcmin2",
    "trace_cov_anisotropy",
    "n_microsaccade_events",
    "fraction_microsaccade_samples",
    "peak_microsaccade_speed_dps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-csv", type=Path, default=DEFAULT_WINDOWS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-patches", type=int, default=100)
    parser.add_argument("--n-traces", type=int, default=1000)
    parser.add_argument("--n-timepoints", type=int, default=32)
    parser.add_argument("--max-movies", type=int, default=0, help="0 means the full patch x trace grid.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--bin-seconds", type=float, default=None)
    parser.add_argument("--high-contrast-metric", type=str, default="image_patch_rms_contrast")
    parser.add_argument("--high-contrast-quantile", type=float, default=0.75)
    parser.add_argument("--min-patch-fraction-inside-image", type=float, default=0.95)
    parser.add_argument("--max-patch-fraction-background", type=float, default=0.05)
    parser.add_argument("--min-patch-distance-to-border-px", type=float, default=0.0)
    parser.add_argument("--max-trace-source-windows", type=int, default=0, help="0 means all windows.")
    parser.add_argument("--trace-max-path-length-arcmin", type=float, default=350.0)
    parser.add_argument("--trace-max-rms-arcmin", type=float, default=0.0)
    parser.add_argument("--trace-max-radius-arcmin", type=float, default=0.0)
    parser.add_argument("--trace-max-speed-p95-arcmin-s", type=float, default=0.0)
    parser.add_argument("--trace-max-microsaccade-events", type=int, default=-1)
    parser.add_argument("--microsaccade-speed-threshold-dps", type=float, default=None)
    parser.add_argument("--microsaccade-threshold-z", type=float, default=6.0)
    parser.add_argument("--microsaccade-pad-frames", type=int, default=1)
    parser.add_argument("--population-mode", choices=("rr100", "full"), default="rr100")
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument(
        "--unit-tuning-csv",
        type=Path,
        default=DEFAULT_UNIT_TUNING_CSV,
        help="Optional one-row-per-unit tuning table to join into unit_metadata.csv and copy as unit_tuning_matrix.csv.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trace-batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write manifests only; do not load the twin.")
    return parser.parse_args()


def _progress(message: str) -> None:
    print(f"[backimage-random-ssi] {message}", flush=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _add_source_row(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "source_row" not in out.columns:
        out["source_row"] = np.arange(out.shape[0], dtype=np.int64)
    return out


def _axis_delta_deg(a_deg: float, b_deg: float) -> float:
    if not (math.isfinite(float(a_deg)) and math.isfinite(float(b_deg))):
        return float("nan")
    return float(0.5 * np.degrees(np.angle(np.exp(2j * np.radians(float(a_deg) - float(b_deg))))))


def _cos2_delta(a_deg: float, b_deg: float) -> float:
    delta = _axis_delta_deg(a_deg, b_deg)
    return float(np.cos(2.0 * np.radians(delta))) if math.isfinite(delta) else float("nan")


def _direction_deg(vec: np.ndarray) -> float:
    arr = np.asarray(vec, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(arr)) or float(np.linalg.norm(arr)) <= 1e-12:
        return float("nan")
    return float((np.degrees(np.arctan2(arr[1], arr[0])) + 360.0) % 360.0)


def _passes_upper_limit(value: Any, limit: float) -> bool:
    limit = float(limit)
    if limit <= 0.0:
        return True
    val = _finite_float(value)
    return math.isfinite(val) and val <= limit


def _sample_patch_rows(windows: pd.DataFrame, args: argparse.Namespace, rng: np.random.Generator) -> pd.DataFrame:
    work = windows.copy()
    if "image_feature_ok" in work.columns:
        work = work[work["image_feature_ok"].astype(bool)]
    if "image_patch_fraction_inside_image" in work.columns:
        work = work[
            pd.to_numeric(work["image_patch_fraction_inside_image"], errors="coerce")
            >= float(args.min_patch_fraction_inside_image)
        ]
    if "image_patch_fraction_background" in work.columns:
        work = work[
            pd.to_numeric(work["image_patch_fraction_background"], errors="coerce")
            <= float(args.max_patch_fraction_background)
        ]
    if "image_patch_distance_to_image_border_px" in work.columns and float(args.min_patch_distance_to_border_px) > 0.0:
        work = work[
            pd.to_numeric(work["image_patch_distance_to_image_border_px"], errors="coerce")
            >= float(args.min_patch_distance_to_border_px)
        ]
    metric = str(args.high_contrast_metric)
    if metric not in work.columns:
        raise ValueError(f"High-contrast metric {metric!r} is not present in {args.windows_csv}")
    values = pd.to_numeric(work[metric], errors="coerce")
    work = work[np.isfinite(values)]
    values = pd.to_numeric(work[metric], errors="coerce")
    cutoff = float(values.quantile(float(args.high_contrast_quantile)))
    eligible = work[values >= cutoff].copy()
    if eligible.shape[0] < int(args.n_patches):
        raise ValueError(
            f"Only {eligible.shape[0]} high-contrast patches remain after filtering; "
            f"need {int(args.n_patches)}."
        )
    sampled = eligible.sample(n=int(args.n_patches), replace=False, random_state=int(rng.integers(0, 2**32 - 1)))
    sampled = sampled.sort_values(["session", "trial_idx", "source_row"]).reset_index(drop=True)
    sampled["patch_index"] = np.arange(sampled.shape[0], dtype=np.int32)
    _progress(
        f"sampled {sampled.shape[0]} patches from top {100.0 * (1.0 - float(args.high_contrast_quantile)):.1f}% "
        f"of {metric} (cutoff={cutoff:.6g})"
    )
    return sampled


def _filter_trace_items(items: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    rejected = {
        "trace_rejected_path_length": 0,
        "trace_rejected_rms": 0,
        "trace_rejected_radius": 0,
        "trace_rejected_speed_p95": 0,
        "trace_rejected_microsaccade_events": 0,
    }
    max_events = int(args.trace_max_microsaccade_events)
    for item in items:
        path_arcmin = item.get("rendered_path_length_arcmin", _finite_float(item.get("path_length_deg")) * 60.0)
        if not _passes_upper_limit(path_arcmin, float(args.trace_max_path_length_arcmin)):
            rejected["trace_rejected_path_length"] += 1
            continue
        rms_arcmin = item.get("rendered_rms_radius_arcmin", np.nan)
        if not math.isfinite(_finite_float(rms_arcmin)) and "observed_rms_deg" in item:
            rms_arcmin = _finite_float(item.get("observed_rms_deg")) * 60.0
        if not _passes_upper_limit(rms_arcmin, float(args.trace_max_rms_arcmin)):
            rejected["trace_rejected_rms"] += 1
            continue
        radius_arcmin = _finite_float(item.get("rendered_max_radius_deg", np.nan)) * 60.0
        if not _passes_upper_limit(radius_arcmin, float(args.trace_max_radius_arcmin)):
            rejected["trace_rejected_radius"] += 1
            continue
        speed_p95 = item.get("rendered_speed_p95_arcmin_s", _finite_float(item.get("rendered_speed_p95_deg_s")) * 60.0)
        if not _passes_upper_limit(speed_p95, float(args.trace_max_speed_p95_arcmin_s)):
            rejected["trace_rejected_speed_p95"] += 1
            continue
        if max_events >= 0 and int(item.get("n_microsaccade_events", 0)) > max_events:
            rejected["trace_rejected_microsaccade_events"] += 1
            continue
        kept.append(item)
    return kept, rejected


def _trace_endpoint_payload(trace: np.ndarray) -> dict[str, float]:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
        return {}
    delta = arr[-1] - arr[0]
    return {
        "trace_endpoint_dx_deg": float(delta[0]),
        "trace_endpoint_dy_deg": float(delta[1]),
        "trace_endpoint_amplitude_arcmin": float(np.linalg.norm(delta) * 60.0),
        "trace_endpoint_direction_deg": _direction_deg(delta),
    }


def _sample_trace_items(
    windows: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = windows.copy()
    if int(args.max_trace_source_windows) > 0 and source.shape[0] > int(args.max_trace_source_windows):
        source = source.sample(
            n=int(args.max_trace_source_windows),
            replace=False,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )
    source = source.sort_values(["session", "trial_idx", "source_row"]).reset_index(drop=True)
    _progress(f"building native {int(args.n_timepoints)}-sample trace bank from {source.shape[0]} source windows")
    eyepos_by_session = _session_dataset_cache(source["session"].astype(str).to_list())
    trace_bank, bank_meta = build_native_snippet_trace_bank(
        source,
        eyepos_by_session,
        int(args.n_timepoints),
        dt=float(args.dt),
        microsaccade_speed_threshold_dps=args.microsaccade_speed_threshold_dps,
        microsaccade_threshold_z=float(args.microsaccade_threshold_z),
        microsaccade_pad_frames=int(args.microsaccade_pad_frames),
    )
    filtered, rejected = _filter_trace_items(trace_bank, args)
    if len(filtered) < int(args.n_traces):
        raise ValueError(f"Only {len(filtered)} trace snippets remain after filtering; need {int(args.n_traces)}.")
    choices = rng.choice(len(filtered), size=int(args.n_traces), replace=False)
    selected = [filtered[int(idx)] for idx in choices]
    rows: list[dict[str, Any]] = []
    for trace_index, item in enumerate(selected):
        row = trace_bank_metadata_row(
            item,
            trace_index,
            n_timepoints=int(args.n_timepoints),
            scale_metric="rendered_diffusion_constant_arcmin2_s",
        )
        row["trace_index"] = int(trace_index)
        row["trace_source_row"] = int(row.pop("source_row"))
        row["trace_source_session"] = str(row.pop("session"))
        row["trace_source_trial_idx"] = int(row.pop("trial_idx"))
        row.update(_trace_endpoint_payload(np.asarray(item["trace"], dtype=np.float32)))
        rows.append(row)
    meta = {
        **bank_meta,
        **rejected,
        "n_trace_bank_items_before_filter": int(len(trace_bank)),
        "n_trace_bank_items_after_filter": int(len(filtered)),
        "n_trace_items_sampled": int(len(selected)),
        "trace_filter_max_path_length_arcmin": float(args.trace_max_path_length_arcmin),
        "trace_filter_max_microsaccade_events": int(args.trace_max_microsaccade_events),
    }
    _progress(f"sampled {len(selected)} traces after filtering ({len(filtered)} eligible)")
    return selected, rows, meta


def _patch_manifest_rows(patches: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in patches.iterrows():
        out: dict[str, Any] = {"patch_index": int(row["patch_index"])}
        for key in PATCH_ID_COLUMNS + IMAGE_FEATURE_COLUMNS:
            if key in row.index:
                value = row[key]
                out[key] = value.item() if isinstance(value, np.generic) else value
        rows.append(out)
    return rows


def _prefix_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in TRACE_MANIFEST_NUMERIC_COLUMNS:
        if key in row:
            out[f"trace_{key}"] = row[key]
    for key in (
        "trace_source_row",
        "trace_source_session",
        "trace_source_trial_idx",
        "trace_hash",
        "trace_endpoint_dx_deg",
        "trace_endpoint_dy_deg",
        "trace_endpoint_amplitude_arcmin",
        "trace_endpoint_direction_deg",
    ):
        if key in row:
            out[key] = row[key]
    return out


def _pair_direction_payload(patch_row: dict[str, Any], trace_row: dict[str, Any]) -> dict[str, float | bool]:
    edge_axis = _finite_float(patch_row.get("image_edge_axis_deg"))
    grad_axis = _finite_float(patch_row.get("image_gradient_axis_deg"))
    spectrum_axis = _finite_float(patch_row.get("image_spectrum_orientation_deg"))
    cov_axis = _finite_float(
        trace_row.get("rendered_cov_orientation_deg", trace_row.get("source_cov_orientation_deg", np.nan))
    )
    endpoint_dir = _finite_float(trace_row.get("trace_endpoint_direction_deg"))
    dx = _finite_float(trace_row.get("trace_endpoint_dx_deg"), 0.0)
    dy = _finite_float(trace_row.get("trace_endpoint_dy_deg"), 0.0)
    payload: dict[str, float | bool] = {
        "same_source_row": int(patch_row.get("source_row", -1)) == int(trace_row.get("trace_source_row", -2)),
        "trace_cov_edge_delta_deg": _axis_delta_deg(cov_axis, edge_axis),
        "trace_cov_edge_cos2": _cos2_delta(cov_axis, edge_axis),
        "trace_cov_gradient_delta_deg": _axis_delta_deg(cov_axis, grad_axis),
        "trace_cov_gradient_cos2": _cos2_delta(cov_axis, grad_axis),
        "trace_cov_spectrum_delta_deg": _axis_delta_deg(cov_axis, spectrum_axis),
        "trace_cov_spectrum_cos2": _cos2_delta(cov_axis, spectrum_axis),
        "trace_endpoint_edge_delta_deg": _axis_delta_deg(endpoint_dir, edge_axis),
        "trace_endpoint_edge_cos2": _cos2_delta(endpoint_dir, edge_axis),
    }
    if math.isfinite(edge_axis):
        theta = math.radians(edge_axis)
        along = dx * math.cos(theta) + dy * math.sin(theta)
        across = -dx * math.sin(theta) + dy * math.cos(theta)
        payload["trace_endpoint_edge_along_arcmin"] = float(along * 60.0)
        payload["trace_endpoint_edge_across_arcmin"] = float(across * 60.0)
    else:
        payload["trace_endpoint_edge_along_arcmin"] = float("nan")
        payload["trace_endpoint_edge_across_arcmin"] = float("nan")
    return payload


def _build_movie_manifest(
    patch_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_full = len(patch_rows) * len(trace_rows)
    pairs = np.asarray(
        [(patch_idx, trace_idx) for patch_idx in range(len(patch_rows)) for trace_idx in range(len(trace_rows))],
        dtype=np.int32,
    )
    if int(args.max_movies) > 0 and int(args.max_movies) < n_full:
        keep = rng.choice(n_full, size=int(args.max_movies), replace=False)
        pairs = pairs[np.sort(keep)]
    # Sort by patch to reduce canvas/model setup churn during scoring.
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    rows: list[dict[str, Any]] = []
    for matrix_row, (patch_idx, trace_idx) in enumerate(pairs):
        patch_row = patch_rows[int(patch_idx)]
        trace_row = trace_rows[int(trace_idx)]
        out: dict[str, Any] = {
            "matrix_row": int(matrix_row),
            "patch_index": int(patch_idx),
            "trace_index": int(trace_idx),
            "patch_source_row": int(patch_row.get("source_row", -1)),
            "trace_source_row": int(trace_row.get("trace_source_row", -1)),
        }
        for key in PATCH_ID_COLUMNS + IMAGE_FEATURE_COLUMNS:
            if key in patch_row:
                out[key] = patch_row[key]
        out.update(_prefix_trace_row(trace_row))
        out.update(_pair_direction_payload(patch_row, trace_row))
        rows.append(out)
    _progress(f"built movie manifest with {len(rows)} rows ({len(patch_rows)} patches x {len(trace_rows)} traces source grid)")
    return pd.DataFrame(rows)


def _population_unit_rows(population_view: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reps = population_view.meta.get("representatives", []) if isinstance(population_view.meta, dict) else []
    by_idx = {int(rep.get("rep_idx", idx)): rep for idx, rep in enumerate(reps)} if reps else {}
    membership = population_view.membership
    for unit_idx in range(int(population_view.n_units)):
        rep = by_idx.get(unit_idx, {})
        row: dict[str, Any] = {
            "unit_index": int(unit_idx),
            "population_name": str(population_view.name),
            "population_n_units": int(population_view.n_units),
            "population_input_channels": int(population_view.input_channels),
        }
        if membership is not None:
            weights = np.asarray(membership[unit_idx], dtype=np.float64)
            nonzero = np.flatnonzero(np.abs(weights) > 1e-12)
            row["membership_n_nonzero"] = int(nonzero.size)
            row["membership_selected_channel"] = int(nonzero[np.argmax(np.abs(weights[nonzero]))]) if nonzero.size else -1
        for key, value in rep.items():
            if key == "members":
                members = [int(v) for v in value]
                row["member_count"] = int(len(members))
                row["members"] = ";".join(str(v) for v in members)
            elif isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
                row[f"rep_{key}"] = value.item() if isinstance(value, np.generic) else value
        rows.append(row)
    return rows


def _unit_tuning_rows(path: Path | None, n_units: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        return [], {"unit_tuning_csv": None, "unit_tuning_status": "disabled"}
    path = Path(path)
    if not path.exists():
        return [], {"unit_tuning_csv": str(path), "unit_tuning_status": "missing"}
    frame = pd.read_csv(path)
    if "unit_index" not in frame.columns:
        raise ValueError(f"Unit tuning table {path} does not contain a unit_index column.")
    frame["unit_index"] = pd.to_numeric(frame["unit_index"], errors="raise").astype(int)
    duplicate_count = int(frame["unit_index"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Unit tuning table {path} has {duplicate_count} duplicate unit_index rows.")
    by_unit = {int(row["unit_index"]): row for row in frame.to_dict(orient="records")}
    rows: list[dict[str, Any]] = []
    missing = 0
    for unit_idx in range(int(n_units)):
        raw = by_unit.get(unit_idx)
        if raw is None:
            rows.append({"unit_index": int(unit_idx), "tuning_row_present": False})
            missing += 1
            continue
        row = {"unit_index": int(unit_idx), "tuning_row_present": True}
        for key, value in raw.items():
            if key == "unit_index":
                continue
            row[str(key)] = value
        rows.append(row)
    meta = {
        "unit_tuning_csv": str(path),
        "unit_tuning_status": "loaded",
        "unit_tuning_source_rows": int(frame.shape[0]),
        "unit_tuning_rows_written": int(len(rows)),
        "unit_tuning_missing_units": int(missing),
    }
    return rows, meta


def _join_unit_tuning(unit_rows: list[dict[str, Any]], tuning_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tuning_rows:
        return unit_rows
    tuning_by_unit = {int(row["unit_index"]): row for row in tuning_rows}
    joined: list[dict[str, Any]] = []
    for row in unit_rows:
        out = dict(row)
        tuning = tuning_by_unit.get(int(row["unit_index"]), {})
        for key, value in tuning.items():
            if key == "unit_index":
                continue
            out[f"tuning_{key}"] = value
        joined.append(out)
    return joined


def _iter_reduced_rate_maps_for_traces(
    scorer: CanonicalTwinScorer,
    patch: np.ndarray,
    traces: list[np.ndarray],
    *,
    trace_batch_size: int,
    population_view: Any,
):
    image = _standardize_uint_like(patch)
    trace_batch_size = max(1, int(trace_batch_size))
    for start in range(0, len(traces), trace_batch_size):
        trace_chunk = traces[start : start + trace_batch_size]
        stims = []
        lengths: list[int] = []
        for trace in trace_chunk:
            trace = np.asarray(trace, dtype=np.float32)
            if trace.shape[0] < int(scorer.common.N_LAGS):
                raise ValueError(
                    f"Trace has {trace.shape[0]} samples, but the twin helper requires at least "
                    f"{int(scorer.common.N_LAGS)} samples."
                )
            full_stack = np.broadcast_to(
                image[None, :, :],
                (trace.shape[0] + scorer.common.N_LAGS + 1, *image.shape),
            ).copy()
            eye = scorer.torch.from_numpy(_trace_xy_to_twin_helper_order(trace))
            stim = scorer.common.make_counterfactual_stim(
                full_stack,
                eye,
                ppd=scorer.common.PPD,
                scale_factor=1.0,
                n_lags=scorer.common.N_LAGS,
                out_size=scorer.common.OUT_SIZE,
            )
            stims.append((stim - 127.0) / 255.0)
            lengths.append(int(stim.shape[0]))
        rate_map = scorer._compute_rate_map_batched(scorer.torch.cat(stims, dim=0))
        rate_np = rate_map.detach().cpu().numpy().astype(np.float32, copy=False)
        reduced_np = apply_population_view(rate_np, population_view).astype(np.float32, copy=False)
        offset = 0
        for length in lengths:
            yield reduced_np[offset : offset + length], length
            offset += length
        del stims, rate_map, rate_np, reduced_np


def _extract_patch_arrays(
    patch_df: pd.DataFrame,
    *,
    patch_size_px: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    patches: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    for _, row in tqdm(list(patch_df.iterrows()), desc="extract patches"):
        patch, meta = _extract_patch(row, canvas_cache=canvas_cache, patch_size_px=int(patch_size_px))
        patches.append(np.asarray(patch, dtype=np.float32))
        meta_rows.append({k: (v.item() if isinstance(v, np.generic) else v) for k, v in meta.items()})
    return patches, meta_rows


def _score_manifest(
    args: argparse.Namespace,
    movie_df: pd.DataFrame,
    patch_arrays: list[np.ndarray],
    trace_items: list[dict[str, Any]],
    population_view: Any,
) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    ssi_path = out_dir / "ssi_matrix.npy"
    spikes_path = out_dir / "expected_spikes_matrix.npy"
    mean_rate_path = out_dir / "mean_rate_matrix.npy"
    pop_path = out_dir / "population_bits_per_spike.npy"
    total_spikes_path = out_dir / "population_expected_spikes.npy"
    if not bool(args.force):
        for path in (ssi_path, spikes_path, mean_rate_path, pop_path, total_spikes_path):
            if path.exists():
                raise FileExistsError(f"{path} exists; pass --force to overwrite.")
    n_movies = int(movie_df.shape[0])
    n_units = int(population_view.n_units)
    ssi = np.lib.format.open_memmap(ssi_path, mode="w+", dtype=np.float32, shape=(n_movies, n_units))
    spikes = np.lib.format.open_memmap(spikes_path, mode="w+", dtype=np.float32, shape=(n_movies, n_units))
    mean_rate = np.lib.format.open_memmap(mean_rate_path, mode="w+", dtype=np.float32, shape=(n_movies, n_units))
    pop_bits = np.lib.format.open_memmap(pop_path, mode="w+", dtype=np.float32, shape=(n_movies,))
    total_spikes = np.lib.format.open_memmap(total_spikes_path, mode="w+", dtype=np.float32, shape=(n_movies,))
    scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(args.batch_size))
    bin_seconds = float(args.bin_seconds) if args.bin_seconds is not None else float(args.dt)
    grouped = movie_df.groupby("patch_index", sort=True)
    for patch_index, group in tqdm(grouped, total=len(grouped), desc="score patch x traces"):
        patch = patch_arrays[int(patch_index)]
        traces = [np.asarray(trace_items[int(idx)]["trace"], dtype=np.float32) for idx in group["trace_index"].to_list()]
        rate_maps = _iter_reduced_rate_maps_for_traces(
            scorer,
            patch,
            traces,
            trace_batch_size=int(args.trace_batch_size),
            population_view=population_view,
        )
        for matrix_row, (full_map, _length) in zip(group["matrix_row"].to_list(), rate_maps, strict=True):
            aligned = _align_response_to_trace(full_map, int(args.n_timepoints))
            ssi_payload = unit_spatial_ssi_for_movie(aligned, bin_seconds=bin_seconds)
            unit_bits = np.asarray(ssi_payload["unit_bits_per_spike"], dtype=np.float32)
            unit_spikes = np.asarray(ssi_payload["unit_expected_spikes"], dtype=np.float32)
            unit_rate = np.asarray(ssi_payload["unit_mean_rate"], dtype=np.float32)
            row = int(matrix_row)
            ssi[row] = unit_bits
            spikes[row] = unit_spikes
            mean_rate[row] = unit_rate
            pop_bits[row] = float(ssi_payload["population_bits_per_spike"])
            total_spikes[row] = float(np.sum(unit_spikes))
            del aligned
    for arr in (ssi, spikes, mean_rate, pop_bits, total_spikes):
        arr.flush()
    return {
        "ssi_matrix": str(ssi_path),
        "expected_spikes_matrix": str(spikes_path),
        "mean_rate_matrix": str(mean_rate_path),
        "population_bits_per_spike": str(pop_path),
        "population_expected_spikes": str(total_spikes_path),
        "n_movies_scored": int(n_movies),
        "n_units": int(n_units),
        "bin_seconds": float(bin_seconds),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = _add_source_row(pd.read_csv(args.windows_csv))
    patch_df = _sample_patch_rows(windows, args, rng)
    trace_items, trace_rows, trace_meta = _sample_trace_items(windows, args, rng)
    patch_rows = _patch_manifest_rows(patch_df)
    patch_arrays, patch_extract_meta = _extract_patch_arrays(patch_df, patch_size_px=int(args.patch_size_px))
    for row, meta in zip(patch_rows, patch_extract_meta, strict=True):
        row.update(meta)
    movie_df = _build_movie_manifest(patch_rows, trace_rows, args, rng)

    _write_csv(out_dir / "patch_table.csv", patch_rows)
    _write_csv(out_dir / "trace_table.csv", trace_rows)
    movie_df.to_csv(out_dir / "movie_feature_matrix.csv", index=False)

    score_meta: dict[str, Any] = {"dry_run": bool(args.dry_run)}
    unit_tuning_meta: dict[str, Any] = {"unit_tuning_status": "not_loaded"}
    if bool(args.dry_run):
        if str(args.population_mode) == "rr100":
            population_view = load_population_view(version_name=str(args.rr100_version))
            unit_rows = _population_unit_rows(population_view)
            unit_tuning, unit_tuning_meta = _unit_tuning_rows(args.unit_tuning_csv, int(population_view.n_units))
            if unit_tuning:
                _write_csv(out_dir / "unit_tuning_matrix.csv", unit_tuning)
                unit_rows = _join_unit_tuning(unit_rows, unit_tuning)
        else:
            unit_rows = []
        _progress("dry run requested; skipped model loading and SSI matrix generation")
    else:
        if str(args.population_mode) == "rr100":
            population_view = load_population_view(version_name=str(args.rr100_version))
        else:
            temp_scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(args.batch_size))
            population_view = full_population_view(temp_scorer.n_units, name="full_canonical_shared_readout")
            del temp_scorer
        unit_rows = _population_unit_rows(population_view)
        unit_tuning, unit_tuning_meta = _unit_tuning_rows(args.unit_tuning_csv, int(population_view.n_units))
        if unit_tuning:
            _write_csv(out_dir / "unit_tuning_matrix.csv", unit_tuning)
            unit_rows = _join_unit_tuning(unit_rows, unit_tuning)
        _write_csv(out_dir / "unit_metadata.csv", unit_rows)
        score_meta = _score_manifest(args, movie_df, patch_arrays, trace_items, population_view)

    run_meta = {
        "windows_csv": str(args.windows_csv),
        "out_dir": str(out_dir),
        "n_patches_requested": int(args.n_patches),
        "n_traces_requested": int(args.n_traces),
        "n_timepoints": int(args.n_timepoints),
        "n_movies": int(movie_df.shape[0]),
        "max_movies": int(args.max_movies),
        "seed": int(args.seed),
        "patch_size_px": int(args.patch_size_px),
        "dt": float(args.dt),
        "bin_seconds": float(args.bin_seconds) if args.bin_seconds is not None else float(args.dt),
        "patch_sampling": {
            "high_contrast_metric": str(args.high_contrast_metric),
            "high_contrast_quantile": float(args.high_contrast_quantile),
            "min_patch_fraction_inside_image": float(args.min_patch_fraction_inside_image),
            "max_patch_fraction_background": float(args.max_patch_fraction_background),
            "min_patch_distance_to_border_px": float(args.min_patch_distance_to_border_px),
        },
        "trace_sampling": trace_meta,
        "population_mode": str(args.population_mode),
        "rr100_version": str(args.rr100_version) if str(args.population_mode) == "rr100" else None,
        "unit_tuning": unit_tuning_meta,
        "matrix_contract": (
            "Rows are movie_feature_matrix.matrix_row; columns are unit_metadata.unit_index. "
            "Each movie is one high-contrast BackImage patch rendered with one native center-cropped "
            "eye trace. SSI is time-resolved spatial bits/spike, expected-spike weighted over frames."
        ),
        "score_outputs": score_meta,
    }
    _write_json(out_dir / "run_metadata.json", run_meta)
    if bool(args.dry_run):
        _write_csv(out_dir / "unit_metadata.csv", unit_rows)
    _progress(f"wrote manifests and metadata to {out_dir}")


if __name__ == "__main__":
    main()
