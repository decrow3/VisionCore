#!/usr/bin/env python3
"""Render instantaneous BackImage RR100 unit-map sheets for one contour window."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    DEFAULT_AXIS_RUN_DIR,
    RR100_MOVIE_MEDOID_VERSION,
    STIMULUS_NORMALIZATION,
    combined_axis_trace,
    identity_text,
    parse_float_list,
    rate_map_for_trace,
    scale_token,
    select_source_trials,
    trace_rms,
    unit_spatial_ssi_for_movie,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
    _align_response_to_trace,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


DEFAULT_SOURCE_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_spatial_ssi_n128_across_sweep_v1"
)
DEFAULT_OUT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_instantaneous_unit_maps_latest_v1"
)
DEFAULT_SCALES = "0,0.125,0.25,0.5,0.75,1,1.5,2,3"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-run-dir", type=Path, default=DEFAULT_AXIS_RUN_DIR)
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--scales", type=str, default=DEFAULT_SCALES)
    parser.add_argument("--units", type=str, default="", help="Optional comma-separated RR100 unit indices.")
    parser.add_argument("--movie-index", type=int, default=None)
    parser.add_argument("--source-row", type=int, default=None)
    parser.add_argument("--source-trace-scale", type=float, default=1.0)
    parser.add_argument("--source-trace-prior-family", type=str, default="axis_edge_parallel")
    parser.add_argument("--axis-column", type=str, default="image_edge_axis_deg")
    parser.add_argument(
        "--sweep-fixed-scale",
        type=float,
        default=1.0,
        help="Scale for the non-varied axis in the along/across one-dimensional sweeps.",
    )
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--compact-frame", type=int, default=None)
    parser.add_argument("--map-vmin-percentile", type=float, default=1.0)
    parser.add_argument("--map-vmax-percentile", type=float, default=99.0)
    parser.add_argument("--orientation-probe-deg", type=str, default="0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170")
    parser.add_argument("--orientation-probe-cycles-per-patch", type=float, default=16.0)
    parser.add_argument("--orientation-probe-contrast", type=float, default=0.8)
    parser.add_argument("--orientation-probe-window-sigma-frac", type=float, default=0.22)
    parser.add_argument(
        "--annotate-frame-ssi-units",
        type=str,
        default="",
        help="Optional comma-separated RR100 units for a separate all-timepoint PDF with per-frame SSI labels.",
    )
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def safe_slug(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text).strip("_") or "unnamed"


def cache_path(out_dir: Path) -> Path:
    return out_dir / "cache" / "backimage_rr100_instantaneous_unit_maps.npz"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_cache(path: Path, identity: dict[str, Any]) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            observed = str(np.asarray(data["cache_identity_json"]).ravel()[0])
            candidate_identities = [identity]
            if np.isclose(float(identity.get("sweep_fixed_scale", float("nan"))), 1.0):
                legacy_identity = dict(identity)
                legacy_identity.pop("sweep_fixed_scale", None)
                candidate_identities.append(legacy_identity)
            if observed not in {identity_text(candidate) for candidate in candidate_identities}:
                return None
            return {key: np.asarray(data[key]) for key in data.files if key != "cache_identity_json"}
    except Exception:
        return None


def save_cache(path: Path, payload: dict[str, Any], identity: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload, cache_identity_json=np.asarray([identity_text(identity)]))


def choose_movie(trials: pd.DataFrame, args: argparse.Namespace) -> tuple[int, pd.Series]:
    if args.source_row is not None:
        matches = np.flatnonzero(trials["source_row"].astype(int).to_numpy() == int(args.source_row))
        if matches.size != 1:
            raise ValueError(f"Expected exactly one trial with source_row={args.source_row}, found {matches.size}.")
        idx = int(matches[0])
        return idx, trials.iloc[idx]
    if args.movie_index is not None:
        idx = int(args.movie_index)
        if idx < 0 or idx >= int(trials.shape[0]):
            raise IndexError(f"movie_index {idx} outside 0..{int(trials.shape[0]) - 1}")
        return idx, trials.iloc[idx]

    coherence = pd.to_numeric(trials.get("image_orientation_coherence", 0.0), errors="coerce").fillna(0.0)
    gradient = pd.to_numeric(trials.get("image_gradient_energy", 0.0), errors="coerce").fillna(0.0)
    background = pd.to_numeric(trials.get("image_patch_fraction_background", 0.0), errors="coerce").fillna(0.0)
    inside = pd.to_numeric(trials.get("image_patch_fraction_inside_image", 1.0), errors="coerce").fillna(1.0)
    ok = trials.get("image_feature_ok", True)
    ok_arr = np.asarray(ok, dtype=bool) if not isinstance(ok, bool) else np.full(trials.shape[0], ok)
    score = coherence.to_numpy(dtype=float) * np.sqrt(np.maximum(gradient.to_numpy(dtype=float), 0.0))
    score = score * np.clip(1.0 - background.to_numpy(dtype=float), 0.0, 1.0) * np.clip(inside.to_numpy(dtype=float), 0.0, 1.0)
    score[~ok_arr] = -np.inf
    if not np.isfinite(score).any():
        raise ValueError("No finite image-feature scores available for movie selection.")
    idx = int(np.nanargmax(score))
    return idx, trials.iloc[idx]


def condition_rows(scales: list[float], *, fixed_scale: float = 1.0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], int] = {}

    def add_condition(axis_mode: str, scale: float, along: float, across: float) -> None:
        key = (scale_token(along), scale_token(across))
        if key not in seen:
            seen[key] = len(specs)
            specs.append(
                {
                    "condition_index": len(specs),
                    "condition_id": f"along{scale_token(along)}_across{scale_token(across)}",
                    "condition_label": f"a{float(along):g}/c{float(across):g}",
                    "along_scale": float(along),
                    "across_scale": float(across),
                    "is_static_baseline": bool(np.isclose(along, 0.0) and np.isclose(across, 0.0)),
                }
            )
        refs.append(
            {
                "axis_mode": axis_mode,
                "display_scale": float(scale),
                "condition_index": int(seen[key]),
                "along_scale": float(along),
                "across_scale": float(across),
            }
        )

    fixed = float(fixed_scale)
    for scale in scales:
        add_condition("along_sweep", float(scale), float(scale), fixed)
    for scale in scales:
        add_condition("across_sweep", float(scale), fixed, float(scale))
    return specs, refs


def render_maps(args: argparse.Namespace, trial: pd.Series, specs: list[dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray]:
    view = load_population_view(version_name=str(args.rr100_version))
    scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(args.batch_size), empty_cache_every_batch=True)
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    patch, patch_meta = _extract_patch(trial, canvas_cache=canvas_cache, patch_size_px=int(args.patch_size_px))
    source_trace = np.asarray(trial["source_trace"], dtype=np.float32)
    axis_deg = float(trial[str(args.axis_column)])
    maps: list[np.ndarray] = []
    trace_rows: list[dict[str, Any]] = []
    condition_traces: list[np.ndarray] = []
    for idx, spec in enumerate(specs, start=1):
        if bool(spec["is_static_baseline"]):
            trace = np.zeros_like(source_trace, dtype=np.float32)
            trace_meta = {
                "source_trace_rms_deg": trace_rms(source_trace),
                "along_component_rms_deg": 0.0,
                "across_component_rms_deg": 0.0,
                "output_trace_rms_deg": 0.0,
            }
        else:
            trace, trace_meta = combined_axis_trace(
                source_trace,
                axis_deg=axis_deg,
                along_scale=float(spec["along_scale"]),
                across_scale=float(spec["across_scale"]),
            )
        print(
            f"[backimage-instant-unit-maps] {idx}/{len(specs)} "
            f"source_row={int(trial['source_row'])} condition={spec['condition_id']}",
            flush=True,
        )
        full_map = rate_map_for_trace(scorer, patch, trace)
        full_map = _align_response_to_trace(full_map, int(args.n_timepoints))
        rr100_map = apply_population_view(full_map, view).astype(np.float32, copy=False)
        maps.append(rr100_map)
        condition_traces.append(np.asarray(trace, dtype=np.float32))
        trace_rows.append(
            {
                "condition_index": int(spec["condition_index"]),
                "condition_id": str(spec["condition_id"]),
                "along_scale": float(spec["along_scale"]),
                "across_scale": float(spec["across_scale"]),
                **trace_meta,
            }
        )
        del full_map, rr100_map
    return (
        {
            "maps": np.stack(maps, axis=0).astype(np.float32),
            "condition_traces": np.stack(condition_traces, axis=0).astype(np.float32),
            "condition_id": np.asarray([str(spec["condition_id"]) for spec in specs]),
            "condition_label": np.asarray([str(spec["condition_label"]) for spec in specs]),
            "condition_along_scale": np.asarray([float(spec["along_scale"]) for spec in specs], dtype=np.float32),
            "condition_across_scale": np.asarray([float(spec["across_scale"]) for spec in specs], dtype=np.float32),
            "source_trace": source_trace.astype(np.float32),
            "axis_deg": np.asarray([axis_deg], dtype=np.float32),
            "rr100_version": np.asarray([str(view.name)]),
            "stimulus_normalization": np.asarray([STIMULUS_NORMALIZATION]),
            "patch_meta_json": np.asarray([json.dumps(json_ready(patch_meta), sort_keys=True)]),
            "trace_rows_json": np.asarray([json.dumps(json_ready(trace_rows), sort_keys=True)]),
        },
        patch,
    )


def image_scale(images: list[np.ndarray], vmin_percentile: float, vmax_percentile: float) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(img, dtype=np.float32).ravel() for img in images])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, float(vmin_percentile)))
    vmax = float(np.nanpercentile(finite, float(vmax_percentile)))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def angle_diff_deg(a: float, b: float) -> float:
    return float(abs(((float(a) - float(b) + 90.0) % 180.0) - 90.0))


def gaze_axis_deg_to_image_axis_deg(axis_deg: float) -> float:
    """Convert a gaze-space axis angle (+y up) to imshow/image-array angle (+row down)."""
    return float(-float(axis_deg))


def axis_vector_image(axis_deg: float) -> tuple[float, float]:
    theta = math.radians(float(axis_deg))
    return math.cos(theta), math.sin(theta)


def map_axis_deg(image: np.ndarray) -> tuple[float, float, float, float]:
    arr = np.asarray(image, dtype=np.float64)
    baseline = float(np.nanpercentile(arr, 20.0))
    w = np.clip(arr - baseline, 0.0, None)
    total = float(np.nansum(w))
    if not np.isfinite(total) or total <= EPS:
        return float("nan"), 0.0, float(np.nanmean(arr)), 0.0
    yy, xx = np.mgrid[: arr.shape[0], : arr.shape[1]]
    cx = float(np.nansum(w * xx) / total)
    cy = float(np.nansum(w * yy) / total)
    dx = xx - cx
    dy = yy - cy
    cov_xx = float(np.nansum(w * dx * dx) / total)
    cov_yy = float(np.nansum(w * dy * dy) / total)
    cov_xy = float(np.nansum(w * dx * dy) / total)
    theta = 0.5 * math.degrees(math.atan2(2.0 * cov_xy, cov_xx - cov_yy))
    vals = np.linalg.eigvalsh(np.asarray([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64))
    denom = float(np.sum(vals))
    anisotropy = float((vals[-1] - vals[0]) / denom) if denom > EPS else 0.0
    mean_rate = float(np.nanmean(arr))
    return theta, anisotropy, mean_rate, spatial_ssi_single_map(arr)


def spatial_ssi_single_map(image: np.ndarray) -> float:
    rate = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    mean_rate = float(np.nanmean(rate))
    if not np.isfinite(mean_rate) or mean_rate <= EPS:
        return 0.0
    gain = rate / mean_rate
    return float(np.nanmean(gain * np.log2(gain + EPS)))


def choose_compact_frame(source_trace: np.ndarray, requested: int | None) -> int:
    trace = np.asarray(source_trace, dtype=np.float64)
    if requested is not None:
        return int(np.clip(int(requested), 0, trace.shape[0] - 1))
    radius = np.linalg.norm(trace - np.mean(trace, axis=0, keepdims=True), axis=1)
    if not np.isfinite(radius).any():
        return trace.shape[0] // 2
    return int(np.nanargmax(radius))


def unit_metric_rows(
    args: argparse.Namespace,
    payload: dict[str, np.ndarray],
    *,
    contour_axis_deg: float,
    compact_frame: int,
) -> list[dict[str, Any]]:
    maps = np.asarray(payload["maps"], dtype=np.float32)
    along = np.asarray(payload["condition_along_scale"], dtype=float)
    across = np.asarray(payload["condition_across_scale"], dtype=float)
    ref_candidates = np.flatnonzero(np.isclose(along, 1.0) & np.isclose(across, 1.0))
    ref_idx = int(ref_candidates[0]) if ref_candidates.size else 0
    ref_maps = maps[ref_idx, int(compact_frame)]
    contour_axis_image_deg = gaze_axis_deg_to_image_axis_deg(float(contour_axis_deg))
    rows: list[dict[str, Any]] = []
    for unit in range(ref_maps.shape[0]):
        axis, anis, mean_rate, map_ssi = map_axis_deg(ref_maps[unit])
        diff = angle_diff_deg(axis, contour_axis_image_deg) if np.isfinite(axis) else float("nan")
        rows.append(
            {
                "unit_index": int(unit),
                "unit_label": f"u{int(unit):03d}",
                "selection_reference_condition_index": int(ref_idx),
                "selection_reference_condition": str(np.asarray(payload["condition_id"]).astype(str)[ref_idx]),
                "selection_reference_frame": int(compact_frame),
                "activation_axis_deg": float(axis),
                "activation_axis_coordinate_frame": "image_array_x_right_y_down",
                "contour_axis_gaze_deg": float(contour_axis_deg),
                "contour_axis_image_deg": float(contour_axis_image_deg),
                "activation_axis_anisotropy": float(anis),
                "activation_axis_abs_delta_from_contour_deg": float(diff),
                "activation_axis_abs_delta_from_orthogonal_deg": float(abs(diff - 90.0)) if np.isfinite(diff) else float("nan"),
                "reference_map_mean_rate": float(mean_rate),
                "reference_map_ssi_bits_per_spike": float(map_ssi),
                "quality_score": float(max(map_ssi, 0.0) * max(anis, 0.0) * math.sqrt(max(mean_rate, 0.0) + EPS)),
            }
        )
    return rows


def choose_dynamic_contrast_unit(args: argparse.Namespace, metrics: pd.DataFrame, chosen: set[int]) -> int | None:
    z_path = Path(args.source_run_dir) / "backimage_contour_axis_rr100_unit_zscore_curves.csv"
    if z_path.exists():
        z = pd.read_csv(z_path)
        if {"unit_index", "absolute_dynamic_range"}.issubset(z.columns):
            z["abs_dynamic"] = pd.to_numeric(z["absolute_dynamic_range"], errors="coerce").abs()
            for row in z.sort_values("abs_dynamic", ascending=False).itertuples(index=False):
                unit = int(row.unit_index)
                if unit not in chosen and unit in set(metrics["unit_index"].astype(int).to_list()):
                    return unit
    rest = metrics[~metrics["unit_index"].astype(int).isin(chosen)].sort_values("quality_score", ascending=False)
    if rest.empty:
        return None
    return int(rest.iloc[0]["unit_index"])


def selected_rows_from_roles(
    metric_rows: list[dict[str, Any]],
    unit_roles: list[tuple[int, str]],
    *,
    orientation_summary_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    metrics = pd.DataFrame(metric_rows)
    if orientation_summary_rows is not None:
        summary = pd.DataFrame(orientation_summary_rows)
        metrics = metrics.merge(summary, on=["unit_index", "unit_label"], how="left")
    role_by_unit = {int(unit): str(role) for unit, role in unit_roles}
    selected_units = [int(unit) for unit, _role in unit_roles]
    out_rows = []
    for unit in selected_units:
        if unit < 0 or unit >= int(metrics["unit_index"].max()) + 1:
            raise IndexError(f"unit {unit} outside available RR100 unit range")
        match = metrics[metrics["unit_index"].astype(int) == int(unit)]
        if match.empty:
            raise IndexError(f"unit {unit} not found in unit metrics")
        row = match.iloc[0].to_dict()
        row["selection_role"] = role_by_unit[int(unit)]
        out_rows.append(row)
    return selected_units, out_rows


def select_activation_axis_units(
    args: argparse.Namespace,
    metric_rows: list[dict[str, Any]],
) -> tuple[list[int], list[dict[str, Any]]]:
    requested = parse_int_list(str(args.units))
    metrics = pd.DataFrame(metric_rows)
    if requested:
        unit_roles = []
        max_unit = int(metrics["unit_index"].max())
        for pos, unit in enumerate(requested):
            if unit < 0 or unit > max_unit:
                raise IndexError(f"unit {unit} outside 0..{max_unit}")
            role = ["requested_1", "requested_2", "requested_3"][pos] if pos < 3 else f"requested_{pos + 1}"
            unit_roles.append((int(unit), role))
    else:
        aligned = metrics.sort_values(
            by=["activation_axis_abs_delta_from_contour_deg", "quality_score"],
            ascending=[True, False],
        ).iloc[0]
        orth_pool = metrics[metrics["unit_index"].astype(int) != int(aligned["unit_index"])].copy()
        orthogonal = orth_pool.sort_values(
            by=["activation_axis_abs_delta_from_orthogonal_deg", "quality_score"],
            ascending=[True, False],
        ).iloc[0]
        chosen = {int(aligned["unit_index"]), int(orthogonal["unit_index"])}
        third_unit = choose_dynamic_contrast_unit(args, metrics, chosen)
        if third_unit is None:
            raise ValueError("Could not select a third unit.")
        unit_roles = [
            (int(aligned["unit_index"]), "activation_axis_aligned_with_contour"),
            (int(orthogonal["unit_index"]), "activation_axis_orthogonal_to_contour"),
            (int(third_unit), "strong_time_resolved_curve_contrast"),
        ]
    return selected_rows_from_roles(metric_rows, unit_roles)


def condition_lookup(payload: dict[str, np.ndarray]) -> dict[tuple[str, str], int]:
    along = np.asarray(payload["condition_along_scale"], dtype=float)
    across = np.asarray(payload["condition_across_scale"], dtype=float)
    return {(scale_token(a), scale_token(c)): int(i) for i, (a, c) in enumerate(zip(along, across, strict=True))}


def condition_for_ref(ref: dict[str, Any], lookup: dict[tuple[str, str], int]) -> int:
    key = (scale_token(float(ref["along_scale"])), scale_token(float(ref["across_scale"])))
    if key not in lookup:
        raise KeyError(f"Missing condition for along={ref['along_scale']} across={ref['across_scale']}")
    return int(lookup[key])


def display_label_axis_mode(axis_mode: str, *, fixed_scale: float | None = None) -> str:
    fixed = 1.0 if fixed_scale is None else float(fixed_scale)
    if axis_mode == "along_sweep":
        return f"scale along; across={fixed:g}"
    if axis_mode == "across_sweep":
        return f"scale across; along={fixed:g}"
    return axis_mode


def fixed_scale_from_refs(axis_mode: str, refs: list[dict[str, Any]]) -> float:
    if not refs:
        return 1.0
    key = "across_scale" if axis_mode == "along_sweep" else "along_scale"
    values = np.asarray([float(ref[key]) for ref in refs], dtype=float)
    return float(np.nanmedian(values))


def fixed_scale_from_frame(axis_mode: str, frame: pd.DataFrame) -> float:
    if frame.empty:
        return 1.0
    key = "across_scale" if axis_mode == "along_sweep" else "along_scale"
    values = frame[key].to_numpy(dtype=float)
    return float(np.nanmedian(values))


def normalized_contact_sheet(stack: np.ndarray, *, vmin: float, vmax: float, pad: int = 2) -> np.ndarray:
    arr = np.asarray(stack, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected stack with shape row x col x H x W, got {arr.shape}")
    n_rows, n_cols, height, width = arr.shape
    denom = max(float(vmax) - float(vmin), EPS)
    norm = np.clip((arr - float(vmin)) / denom, 0.0, 1.0)
    sheet = np.ones((n_rows * height + (n_rows - 1) * pad, n_cols * width + (n_cols - 1) * pad), dtype=np.float32)
    for r in range(n_rows):
        for c in range(n_cols):
            y0 = r * (height + pad)
            x0 = c * (width + pad)
            sheet[y0 : y0 + height, x0 : x0 + width] = norm[r, c]
    return sheet


def tile_centers(n: int, tile: int, pad: int) -> np.ndarray:
    return np.asarray([i * (tile + pad) + 0.5 * (tile - 1) for i in range(n)], dtype=float)


def plot_compact(
    out_dir: Path,
    payload: dict[str, np.ndarray],
    patch: np.ndarray,
    selected_units: list[int],
    selected_rows: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    scales: list[float],
    trial: pd.Series,
    *,
    compact_frame: int,
    dpi: int,
    vmin_percentile: float,
    vmax_percentile: float,
) -> tuple[Path, Path]:
    maps = np.asarray(payload["maps"], dtype=np.float32)
    lookup = condition_lookup(payload)
    axis_deg = float(np.asarray(payload["axis_deg"]).ravel()[0])
    axis_image_deg = gaze_axis_deg_to_image_axis_deg(axis_deg)
    source_trace = np.asarray(payload["source_trace"], dtype=np.float32)
    fig = plt.figure(figsize=(1.35 * len(scales) + 3.6, 2.0 * len(selected_units) + 3.1))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 2.0 * len(selected_units)], hspace=0.30)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.2, 1.3, 1.7], wspace=0.28)
    ax_patch = fig.add_subplot(top[0, 0])
    patch_img = np.asarray(patch, dtype=np.float32)
    pvmin, pvmax = image_scale([patch_img], 1.0, 99.0)
    ax_patch.imshow(patch_img, cmap="gray", vmin=pvmin, vmax=pvmax)
    h, w = patch_img.shape
    length = 0.32 * min(h, w)
    axis_dx, axis_dy = axis_vector_image(axis_image_deg)
    dx = axis_dx * length
    dy = axis_dy * length
    ax_patch.arrow(
        w / 2 - dx / 2,
        h / 2 - dy / 2,
        dx,
        dy,
        width=2.2,
        head_width=14.0,
        head_length=16.0,
        color="#1f9fb5",
        length_includes_head=True,
    )
    ax_patch.set_title(
        f"source row {int(trial['source_row'])}\n"
        f"contour axis {axis_deg:.1f} deg gaze / {axis_image_deg:.1f} deg image",
        fontsize=8,
    )
    ax_patch.set_xticks([])
    ax_patch.set_yticks([])

    ax_trace = fig.add_subplot(top[0, 1])
    centered = source_trace - np.mean(source_trace, axis=0, keepdims=True)
    theta_gaze = np.radians(axis_deg)
    along_u = np.asarray([np.cos(theta_gaze), np.sin(theta_gaze)], dtype=np.float32)
    across_u = np.asarray([-np.sin(theta_gaze), np.cos(theta_gaze)], dtype=np.float32)
    ax_trace.plot(centered @ along_u, label="along", color="#2c6db2", linewidth=1.4)
    ax_trace.plot(centered @ across_u, label="across", color="#b24f2c", linewidth=1.4)
    ax_trace.axvline(int(compact_frame), color="0.35", linestyle=":", linewidth=1.0)
    ax_trace.set_title("source trace components", fontsize=8)
    ax_trace.set_xlabel("frame")
    ax_trace.set_ylabel("deg")
    ax_trace.grid(True, color="0.9", linewidth=0.6)
    ax_trace.legend(frameon=False, fontsize=7)

    ax_units = fig.add_subplot(top[0, 2])
    ax_units.axis("off")
    unit_text = []
    for row in selected_rows:
        unit_text.append(
            f"{row['unit_label']}  {row['selection_role']}\n"
            f"axis {float(row['activation_axis_deg']):.1f} deg, "
            f"delta {float(row['activation_axis_abs_delta_from_contour_deg']):.1f} deg, "
            f"SSI {float(row['reference_map_ssi_bits_per_spike']):.3f}"
        )
    ax_units.text(0.0, 1.0, "\n".join(unit_text), ha="left", va="top", fontsize=7.2)

    map_grid = outer[1].subgridspec(len(selected_units) * 2, len(scales), hspace=0.08, wspace=0.04)
    refs_by_axis = {
        axis: [ref for ref in refs if ref["axis_mode"] == axis]
        for axis in ("across_sweep", "along_sweep")
    }
    fixed_by_axis = {axis: fixed_scale_from_refs(axis, axis_refs) for axis, axis_refs in refs_by_axis.items()}
    row_lookup = {int(row["unit_index"]): row for row in selected_rows}
    for unit_pos, unit in enumerate(selected_units):
        row_maps = []
        for axis_mode in ("across_sweep", "along_sweep"):
            for ref in refs_by_axis[axis_mode]:
                row_maps.append(maps[condition_for_ref(ref, lookup), :, int(unit)])
        vmin, vmax = image_scale(
            [img for stack in row_maps for img in stack],
            float(vmin_percentile),
            float(vmax_percentile),
        )
        for local_row, axis_mode in enumerate(("across_sweep", "along_sweep")):
            global_row = unit_pos * 2 + local_row
            for col, ref in enumerate(refs_by_axis[axis_mode]):
                ax = fig.add_subplot(map_grid[global_row, col])
                image = maps[condition_for_ref(ref, lookup), int(compact_frame), int(unit)]
                ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if global_row == 0:
                    ax.set_title(f"{float(ref['display_scale']):g}x", fontsize=7)
                if col == 0:
                    meta = row_lookup[int(unit)]
                    ax.set_ylabel(
                        f"{meta['unit_label']}\n{display_label_axis_mode(axis_mode, fixed_scale=fixed_by_axis[axis_mode])}",
                        fontsize=6.8,
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=46,
                    )
                for spine in ax.spines.values():
                    spine.set_linewidth(0.45)
                    spine.set_edgecolor("0.45")

    fig.suptitle(
        "BackImage RR100 instantaneous activation maps for one contour window\n"
        f"compact view uses single frame t={int(compact_frame)}; no trajectory-averaged maps",
        fontsize=11,
        y=0.995,
    )
    png = out_dir / "backimage_rr100_instantaneous_unit_maps_compact.png"
    pdf = out_dir / "backimage_rr100_instantaneous_unit_maps_compact.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_full_timepoint_pdf(
    out_dir: Path,
    payload: dict[str, np.ndarray],
    selected_units: list[int],
    selected_rows: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    scales: list[float],
    trial: pd.Series,
    *,
    vmin_percentile: float,
    vmax_percentile: float,
) -> Path:
    maps = np.asarray(payload["maps"], dtype=np.float32)
    lookup = condition_lookup(payload)
    axis_deg = float(np.asarray(payload["axis_deg"]).ravel()[0])
    refs_by_axis = {
        axis: [ref for ref in refs if ref["axis_mode"] == axis]
        for axis in ("across_sweep", "along_sweep")
    }
    fixed_by_axis = {axis: fixed_scale_from_refs(axis, axis_refs) for axis, axis_refs in refs_by_axis.items()}
    row_lookup = {int(row["unit_index"]): row for row in selected_rows}
    path = out_dir / "backimage_rr100_instantaneous_unit_maps_all_timepoints.pdf"
    with PdfPages(path) as pdf:
        for unit in selected_units:
            meta = row_lookup[int(unit)]
            for axis_mode in ("across_sweep", "along_sweep"):
                condition_indices = [condition_for_ref(ref, lookup) for ref in refs_by_axis[axis_mode]]
                stack = maps[condition_indices, :, int(unit)]
                vmin, vmax = image_scale(
                    [stack[row, col] for row in range(stack.shape[0]) for col in range(stack.shape[1])],
                    float(vmin_percentile),
                    float(vmax_percentile),
                )
                sheet = normalized_contact_sheet(stack, vmin=vmin, vmax=vmax, pad=2)
                fig, ax = plt.subplots(figsize=(18.8, 6.5), constrained_layout=True)
                ax.imshow(sheet, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                centers_x = tile_centers(stack.shape[1], stack.shape[-1], 2)
                centers_y = tile_centers(stack.shape[0], stack.shape[-2], 2)
                xticks = list(range(0, stack.shape[1], 4))
                if (stack.shape[1] - 1) not in xticks:
                    xticks.append(stack.shape[1] - 1)
                ax.set_xticks(centers_x[xticks], [str(v) for v in xticks], fontsize=6)
                ax.set_yticks(centers_y, [f"{float(scale):g}x" for scale in scales], fontsize=7)
                ax.set_xlabel("time index")
                axis_label = display_label_axis_mode(axis_mode, fixed_scale=fixed_by_axis[axis_mode])
                ax.set_ylabel(axis_label)
                ax.set_title(
                    f"BackImage RR100 {meta['unit_label']} ({meta['selection_role']}), "
                    f"{axis_label}\n"
                    f"source row {int(trial['source_row'])}, contour axis {axis_deg:.1f} deg gaze; "
                    "each tile is one instantaneous activation map",
                    fontsize=10,
                )
                for spine in ax.spines.values():
                    spine.set_visible(False)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    return path


def plot_frame_ssi_annotated_unit_pdf(
    out_dir: Path,
    payload: dict[str, np.ndarray],
    annotate_units: list[int],
    annotate_rows: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    scales: list[float],
    trial: pd.Series,
    *,
    vmin_percentile: float,
    vmax_percentile: float,
    dpi: int,
) -> tuple[Path, list[Path]]:
    maps = np.asarray(payload["maps"], dtype=np.float32)
    lookup = condition_lookup(payload)
    axis_deg = float(np.asarray(payload["axis_deg"]).ravel()[0])
    refs_by_axis = {
        axis: [ref for ref in refs if ref["axis_mode"] == axis]
        for axis in ("across_sweep", "along_sweep")
    }
    fixed_by_axis = {axis: fixed_scale_from_refs(axis, axis_refs) for axis, axis_refs in refs_by_axis.items()}
    row_lookup = {int(row["unit_index"]): row for row in annotate_rows}
    unit_slug = "_".join(f"u{int(unit):03d}" for unit in annotate_units)
    path = out_dir / f"backimage_rr100_{unit_slug}_instantaneous_unit_maps_frame_ssi_annotated.pdf"
    png_paths: list[Path] = []
    with PdfPages(path) as pdf:
        for unit in annotate_units:
            meta = row_lookup[int(unit)]
            for axis_mode in ("across_sweep", "along_sweep"):
                condition_indices = [condition_for_ref(ref, lookup) for ref in refs_by_axis[axis_mode]]
                stack = maps[condition_indices, :, int(unit)]
                bits = instantaneous_bits(stack[:, :, None, :, :].reshape(-1, 1, stack.shape[-2], stack.shape[-1]))
                bits = bits.reshape(stack.shape[0], stack.shape[1])
                vmin, vmax = image_scale(
                    [stack[row, col] for row in range(stack.shape[0]) for col in range(stack.shape[1])],
                    float(vmin_percentile),
                    float(vmax_percentile),
                )
                pad = 2
                sheet = normalized_contact_sheet(stack, vmin=vmin, vmax=vmax, pad=pad)
                fig, ax = plt.subplots(figsize=(19.5, 7.2), constrained_layout=True)
                ax.imshow(sheet, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                tile_h = stack.shape[-2]
                tile_w = stack.shape[-1]
                centers_x = tile_centers(stack.shape[1], tile_w, pad)
                centers_y = tile_centers(stack.shape[0], tile_h, pad)
                xticks = list(range(0, stack.shape[1], 4))
                if (stack.shape[1] - 1) not in xticks:
                    xticks.append(stack.shape[1] - 1)
                ax.set_xticks(centers_x[xticks], [str(v) for v in xticks], fontsize=6)
                ax.set_yticks(centers_y, [f"{float(scale):g}x" for scale in scales], fontsize=7)
                ax.set_xlabel("time index")
                axis_label = display_label_axis_mode(axis_mode, fixed_scale=fixed_by_axis[axis_mode])
                ax.set_ylabel(axis_label)
                for row in range(stack.shape[0]):
                    for col in range(stack.shape[1]):
                        x = col * (tile_w + pad) + 1.5
                        y = row * (tile_h + pad) + 4.4
                        label = f"{float(bits[row, col]):.3f}"
                        text = ax.text(
                            x,
                            y,
                            label,
                            ha="left",
                            va="top",
                            color="white",
                            fontsize=3.5,
                            family="monospace",
                        )
                        text.set_path_effects(
                            [
                                path_effects.Stroke(linewidth=0.75, foreground="black"),
                                path_effects.Normal(),
                            ]
                        )
                ax.set_title(
                    f"BackImage RR100 {meta['unit_label']} ({meta['selection_role']}), "
                    f"{axis_label}\n"
                    f"source row {int(trial['source_row'])}, contour axis {axis_deg:.1f} deg gaze; "
                    "tile labels are instantaneous spatial SSI bits/spike",
                    fontsize=10,
                )
                for spine in ax.spines.values():
                    spine.set_visible(False)
                pdf.savefig(fig, bbox_inches="tight")
                png = out_dir / (
                    f"backimage_rr100_{meta['unit_label']}_instantaneous_unit_maps_"
                    f"{safe_slug(axis_mode)}_frame_ssi_annotated.png"
                )
                fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
                png_paths.append(png)
                plt.close(fig)
    return path, png_paths


def frame_ssi_timecourse_rows(
    payload: dict[str, np.ndarray],
    annotate_units: list[int],
    refs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    maps = np.asarray(payload["maps"], dtype=np.float32)
    lookup = condition_lookup(payload)
    rows: list[dict[str, Any]] = []
    for unit in annotate_units:
        for ref in refs:
            condition_idx = condition_for_ref(ref, lookup)
            movie = maps[condition_idx, :, int(unit)][:, None, :, :]
            bits = instantaneous_bits(movie)
            mean_rate = np.mean(np.maximum(movie[:, 0], 0.0), axis=(-2, -1))
            for frame_idx, (bit, rate) in enumerate(zip(bits, mean_rate, strict=True)):
                sharpness = activation_map_sharpness_metrics(movie[frame_idx, 0])
                rows.append(
                    {
                        "unit_index": int(unit),
                        "unit_label": f"u{int(unit):03d}",
                        "axis_mode": str(ref["axis_mode"]),
                        "display_scale": float(ref["display_scale"]),
                        "condition_index": int(condition_idx),
                        "condition_id": str(np.asarray(payload["condition_id"]).astype(str)[condition_idx]),
                        "along_scale": float(ref["along_scale"]),
                        "across_scale": float(ref["across_scale"]),
                        "frame_index": int(frame_idx),
                        "instantaneous_spatial_ssi_bits_per_spike": float(bit),
                        "instantaneous_mean_rate": float(rate),
                        **sharpness,
                    }
                )
    return rows


def activation_map_sharpness_metrics(image: np.ndarray) -> dict[str, float]:
    arr = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    if not np.isfinite(mean) or mean <= EPS:
        return {
            "activation_map_cv": 0.0,
            "mean_normalized_gradient_rms": 0.0,
            "mean_normalized_total_variation": 0.0,
            "mean_normalized_laplacian_variance": 0.0,
            "sharpness_metric_contract": "nonnegative activation map; metrics computed after per-frame mean normalization",
        }
    norm = arr / mean
    dx = np.diff(norm, axis=1)
    dy = np.diff(norm, axis=0)
    grad_rms = math.sqrt(max(0.0, 0.5 * (float(np.nanmean(dx * dx)) + float(np.nanmean(dy * dy)))))
    total_variation = 0.5 * (float(np.nanmean(np.abs(dx))) + float(np.nanmean(np.abs(dy))))
    lap = (
        -4.0 * norm[1:-1, 1:-1]
        + norm[:-2, 1:-1]
        + norm[2:, 1:-1]
        + norm[1:-1, :-2]
        + norm[1:-1, 2:]
    )
    return {
        "activation_map_cv": float(std / mean),
        "mean_normalized_gradient_rms": float(grad_rms),
        "mean_normalized_total_variation": float(total_variation),
        "mean_normalized_laplacian_variance": float(np.nanvar(lap)) if lap.size else 0.0,
        "sharpness_metric_contract": "nonnegative activation map; metrics computed after per-frame mean normalization",
    }


def sharpness_metric_specs() -> list[tuple[str, str]]:
    return [
        ("instantaneous_spatial_ssi_bits_per_spike", "SSI bits/spike"),
        ("mean_normalized_gradient_rms", "mean-normalized gradient RMS"),
        ("mean_normalized_laplacian_variance", "mean-normalized Laplacian variance"),
    ]


def plot_frame_ssi_timecourse_overlay(
    out_dir: Path,
    frame_rows: list[dict[str, Any]],
    annotate_units: list[int],
    annotate_rows: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    trial: pd.Series,
    *,
    dpi: int,
) -> tuple[Path, list[Path]]:
    df = pd.DataFrame(frame_rows)
    refs_by_axis = {
        axis: [ref for ref in refs if ref["axis_mode"] == axis]
        for axis in ("across_sweep", "along_sweep")
    }
    fixed_by_axis = {axis: fixed_scale_from_refs(axis, axis_refs) for axis, axis_refs in refs_by_axis.items()}
    row_lookup = {int(row["unit_index"]): row for row in annotate_rows}
    unit_slug = "_".join(f"u{int(unit):03d}" for unit in annotate_units)
    path = out_dir / f"backimage_rr100_{unit_slug}_frame_ssi_timecourse_rows_colored.pdf"
    png_paths: list[Path] = []
    with PdfPages(path) as pdf:
        for unit in annotate_units:
            meta = row_lookup[int(unit)]
            unit_df = df[df["unit_index"].astype(int) == int(unit)]
            fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), sharey=True, constrained_layout=True)
            for ax, axis_mode in zip(axes, ("across_sweep", "along_sweep"), strict=True):
                sub = unit_df[unit_df["axis_mode"].astype(str) == axis_mode].copy()
                scales = np.asarray(sorted(sub["display_scale"].unique()), dtype=float)
                cmap = plt.get_cmap("viridis")
                norm = plt.Normalize(vmin=float(np.nanmin(scales)), vmax=float(np.nanmax(scales)))
                for scale in scales:
                    scale_sub = sub[np.isclose(sub["display_scale"].astype(float), float(scale))].sort_values("frame_index")
                    ax.plot(
                        scale_sub["frame_index"].to_numpy(dtype=int),
                        scale_sub["instantaneous_spatial_ssi_bits_per_spike"].to_numpy(dtype=float),
                        color=cmap(norm(float(scale))),
                        linewidth=1.5,
                        alpha=0.95,
                        label=f"{float(scale):g}x",
                    )
                axis_label = display_label_axis_mode(axis_mode, fixed_scale=fixed_by_axis[axis_mode])
                ax.set_title(axis_label, fontsize=10)
                ax.set_xlabel("frame")
                ax.grid(True, color="0.9", linewidth=0.7)
                ax.axvline(0, color="0.55", linestyle=":", linewidth=0.8)
            axes[0].set_ylabel("instantaneous spatial SSI bits/spike")
            handles, labels = axes[1].get_legend_handles_labels()
            fig.legend(handles, labels, title="scale row", frameon=False, loc="center right", bbox_to_anchor=(1.05, 0.5))
            fig.suptitle(
                f"BackImage RR100 {meta['unit_label']} frame-SSI timecourses ({meta['selection_role']})\n"
                f"source row {int(trial['source_row'])}; each colored line is one scale row",
                fontsize=11,
            )
            pdf.savefig(fig, bbox_inches="tight")
            png = out_dir / f"backimage_rr100_{meta['unit_label']}_frame_ssi_timecourse_rows_colored.png"
            fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
            png_paths.append(png)
            plt.close(fig)
    return path, png_paths


def plot_frame_sharpness_method_comparison(
    out_dir: Path,
    frame_rows: list[dict[str, Any]],
    annotate_units: list[int],
    annotate_rows: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    trial: pd.Series,
    *,
    dpi: int,
) -> tuple[Path, list[Path]]:
    df = pd.DataFrame(frame_rows)
    refs_by_axis = {
        axis: [ref for ref in refs if ref["axis_mode"] == axis]
        for axis in ("across_sweep", "along_sweep")
    }
    fixed_by_axis = {axis: fixed_scale_from_refs(axis, axis_refs) for axis, axis_refs in refs_by_axis.items()}
    row_lookup = {int(row["unit_index"]): row for row in annotate_rows}
    metrics = sharpness_metric_specs()
    unit_slug = "_".join(f"u{int(unit):03d}" for unit in annotate_units)
    path = out_dir / f"backimage_rr100_{unit_slug}_frame_sharpness_method_timecourses.pdf"
    png_paths: list[Path] = []
    with PdfPages(path) as pdf:
        for unit in annotate_units:
            meta = row_lookup[int(unit)]
            unit_df = df[df["unit_index"].astype(int) == int(unit)]
            fig, axes = plt.subplots(
                len(metrics),
                2,
                figsize=(12.2, 2.85 * len(metrics)),
                sharex=True,
                constrained_layout=True,
            )
            if len(metrics) == 1:
                axes = np.asarray([axes])
            legend_handles = None
            legend_labels = None
            for metric_idx, (metric_col, metric_label) in enumerate(metrics):
                for col_idx, axis_mode in enumerate(("across_sweep", "along_sweep")):
                    ax = axes[metric_idx, col_idx]
                    sub = unit_df[unit_df["axis_mode"].astype(str) == axis_mode].copy()
                    scales = np.asarray(sorted(sub["display_scale"].unique()), dtype=float)
                    cmap = plt.get_cmap("viridis")
                    norm = plt.Normalize(vmin=float(np.nanmin(scales)), vmax=float(np.nanmax(scales)))
                    for scale in scales:
                        scale_sub = sub[np.isclose(sub["display_scale"].astype(float), float(scale))].sort_values("frame_index")
                        ax.plot(
                            scale_sub["frame_index"].to_numpy(dtype=int),
                            scale_sub[metric_col].to_numpy(dtype=float),
                            color=cmap(norm(float(scale))),
                            linewidth=1.35,
                            alpha=0.95,
                            label=f"{float(scale):g}x",
                        )
                    ax.grid(True, color="0.9", linewidth=0.7)
                    ax.axvline(0, color="0.55", linestyle=":", linewidth=0.8)
                    if metric_idx == 0:
                        axis_label = display_label_axis_mode(axis_mode, fixed_scale=fixed_by_axis[axis_mode])
                        ax.set_title(axis_label, fontsize=10)
                    if col_idx == 0:
                        ax.set_ylabel(metric_label)
                    if metric_idx == len(metrics) - 1:
                        ax.set_xlabel("frame")
                    if metric_idx == 0 and col_idx == 1:
                        legend_handles, legend_labels = ax.get_legend_handles_labels()
                row_values = unit_df[metrics[metric_idx][0]].to_numpy(dtype=float)
                finite = row_values[np.isfinite(row_values)]
                if finite.size:
                    ymin = float(np.nanmin(finite))
                    ymax = float(np.nanmax(finite))
                    pad = 0.05 * max(ymax - ymin, EPS)
                    for ax in axes[metric_idx]:
                        ax.set_ylim(ymin - pad, ymax + pad)
            if legend_handles is not None and legend_labels is not None:
                fig.legend(
                    legend_handles,
                    legend_labels,
                    title="scale row",
                    frameon=False,
                    loc="center right",
                    bbox_to_anchor=(1.06, 0.5),
                )
            fig.suptitle(
                f"BackImage RR100 {meta['unit_label']} per-frame sharpness metrics ({meta['selection_role']})\n"
                "all metrics are calculated directly from instantaneous activation maps",
                fontsize=11,
            )
            pdf.savefig(fig, bbox_inches="tight")
            png = out_dir / f"backimage_rr100_{meta['unit_label']}_frame_sharpness_method_timecourses.png"
            fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
            png_paths.append(png)
            plt.close(fig)
    return path, png_paths


def compute_displayed_ssi(
    payload: dict[str, np.ndarray],
    selected_units: list[int],
    refs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    maps = np.asarray(payload["maps"], dtype=np.float32)
    lookup = condition_lookup(payload)
    rows: list[dict[str, Any]] = []
    for unit in selected_units:
        for ref in refs:
            condition_idx = condition_for_ref(ref, lookup)
            movie = maps[condition_idx, :, int(unit)][:, None, :, :]
            ssi = unit_spatial_ssi_for_movie(movie, bin_seconds=1.0)
            unit_bits_t = instantaneous_bits(movie[:, 0])
            rows.append(
                {
                    "unit_index": int(unit),
                    "unit_label": f"u{int(unit):03d}",
                    "axis_mode": str(ref["axis_mode"]),
                    "display_scale": float(ref["display_scale"]),
                    "condition_index": int(condition_idx),
                    "condition_id": str(np.asarray(payload["condition_id"]).astype(str)[condition_idx]),
                    "along_scale": float(ref["along_scale"]),
                    "across_scale": float(ref["across_scale"]),
                    "displayed_movie_time_resolved_ssi_bits_per_spike": float(
                        np.asarray(ssi["unit_bits_per_spike"], dtype=float)[0]
                    ),
                    "displayed_movie_mean_rate": float(np.asarray(ssi["unit_mean_rate"], dtype=float)[0]),
                    "displayed_movie_expected_spikes_arbitrary_dt": float(
                        np.asarray(ssi["unit_expected_spikes"], dtype=float)[0]
                    ),
                    "instantaneous_ssi_mean_unweighted": float(np.nanmean(unit_bits_t)),
                    "instantaneous_ssi_max": float(np.nanmax(unit_bits_t)),
                }
            )
    return rows


def instantaneous_bits(movie: np.ndarray) -> np.ndarray:
    y = np.maximum(np.asarray(movie, dtype=np.float64), 0.0)
    flat = y.reshape(y.shape[0], -1)
    rbar = np.mean(flat, axis=1)
    gain = flat / (rbar[:, None] + EPS)
    return np.mean(gain * np.log2(gain + EPS), axis=1)


def plot_displayed_ssi_curves(
    out_dir: Path,
    ssi_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    *,
    dpi: int,
) -> tuple[Path, Path]:
    df = pd.DataFrame(ssi_rows)
    selected = pd.DataFrame(selected_rows)
    units = [int(v) for v in selected["unit_index"].to_list()]
    fig, axes = plt.subplots(
        len(units),
        2,
        figsize=(9.5, 2.2 * len(units)),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    if len(units) == 1:
        axes = np.asarray([axes])
    for row_idx, unit in enumerate(units):
        meta = selected[selected["unit_index"].astype(int) == int(unit)].iloc[0]
        for col_idx, axis_mode in enumerate(("across_sweep", "along_sweep")):
            ax = axes[row_idx, col_idx]
            sub = df[(df["unit_index"].astype(int) == int(unit)) & (df["axis_mode"].astype(str) == axis_mode)]
            sub = sub.sort_values("display_scale")
            axis_label = display_label_axis_mode(axis_mode, fixed_scale=fixed_scale_from_frame(axis_mode, sub))
            ax.plot(
                sub["display_scale"].to_numpy(dtype=float),
                sub["displayed_movie_time_resolved_ssi_bits_per_spike"].to_numpy(dtype=float),
                marker="o",
                linewidth=1.5,
                color="#222222",
            )
            ax.axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
            ax.grid(True, color="0.9", linewidth=0.7)
            ax.set_title(axis_label, fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"{meta['unit_label']}\nSSI", rotation=0, ha="right", va="center", labelpad=30)
    axes[-1, 0].set_xlabel("scale")
    axes[-1, 1].set_xlabel("scale")
    fig.suptitle("Displayed movie SSI: averaged over instantaneous maps, not mean-map SSI", fontsize=11)
    png = out_dir / "backimage_rr100_displayed_movie_time_resolved_ssi_curves.png"
    pdf = out_dir / "backimage_rr100_displayed_movie_time_resolved_ssi_curves.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def classify_orientation_tuning_groups(
    orientation_summary_rows: list[dict[str, Any]],
    *,
    response_margin_threshold: float = 0.10,
    target_delta_threshold_deg: float = 30.0,
) -> list[dict[str, Any]]:
    """Assign every available RR100 unit to a coarse contour/across/off-axis group."""
    rows: list[dict[str, Any]] = []
    for row in orientation_summary_rows:
        contour_norm = float(row.get("orientation_probe_contour_norm_response", float("nan")))
        across_norm = float(row.get("orientation_probe_across_norm_response", float("nan")))
        delta_contour = float(row.get("preferred_delta_from_contour_deg", float("nan")))
        delta_across = float(row.get("preferred_delta_from_across_deg", float("nan")))
        target_margin = contour_norm - across_norm
        nearest_target_delta = min(delta_contour, delta_across)
        if (
            not np.isfinite(target_margin)
            or not np.isfinite(nearest_target_delta)
            or abs(target_margin) < float(response_margin_threshold)
            or nearest_target_delta >= float(target_delta_threshold_deg)
        ):
            group = "off_axis_or_mixed"
            group_label = "off-axis / mixed"
            group_rank = 2
        elif target_margin > 0.0:
            group = "contour_biased"
            group_label = "contour-biased"
            group_rank = 0
        else:
            group = "across_biased"
            group_label = "across-biased"
            group_rank = 1
        out = dict(row)
        out.update(
            {
                "orientation_group": group,
                "orientation_group_label": group_label,
                "orientation_group_rank": int(group_rank),
                "orientation_group_contract": (
                    "contour/across groups require abs(contour_norm-across_norm) >= "
                    f"{float(response_margin_threshold):g} and preferred orientation within "
                    f"{float(target_delta_threshold_deg):g} deg of that axis; remaining units are off-axis/mixed"
                ),
                "orientation_target_margin_contour_minus_across": float(target_margin),
                "orientation_nearest_target_delta_deg": float(nearest_target_delta),
            }
        )
        rows.append(out)
    return rows


def zscore_orientation_group_ssi_rows(
    ssi_rows: list[dict[str, Any]],
    orientation_group_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    df = pd.DataFrame(ssi_rows)
    groups = pd.DataFrame(orientation_group_rows)[
        [
            "unit_index",
            "unit_label",
            "orientation_group",
            "orientation_group_label",
            "orientation_group_rank",
            "orientation_target_margin_contour_minus_across",
            "orientation_nearest_target_delta_deg",
            "preferred_orientation_deg",
            "orientation_selectivity_index",
            "orientation_probe_contour_norm_response",
            "orientation_probe_across_norm_response",
            "preferred_delta_from_contour_deg",
            "preferred_delta_from_across_deg",
        ]
    ].copy()
    merged = df.merge(groups, on=["unit_index", "unit_label"], how="left")
    value_col = "displayed_movie_time_resolved_ssi_bits_per_spike"
    z_rows: list[dict[str, Any]] = []
    for (_unit, axis_mode), sub in merged.groupby(["unit_index", "axis_mode"], sort=False):
        values = sub[value_col].to_numpy(dtype=float)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values, ddof=0))
        usable = bool(np.isfinite(std) and std > 1e-9)
        for record in sub.sort_values("display_scale").to_dict("records"):
            raw = float(record[value_col])
            record["ssi_zscore_axis_mode"] = float((raw - mean) / std) if usable and np.isfinite(raw) else float("nan")
            record["ssi_zscore_mean_axis_mode"] = mean
            record["ssi_zscore_std_axis_mode"] = std
            record["ssi_zscore_usable"] = usable
            record["ssi_zscore_contract"] = "per-unit z-score across display scales within each axis_mode"
            z_rows.append(record)

    zdf = pd.DataFrame(z_rows)
    summary_rows: list[dict[str, Any]] = []
    for (axis_mode, group), sub in zdf.groupby(["axis_mode", "orientation_group"], sort=False):
        label = str(sub["orientation_group_label"].iloc[0])
        rank = int(sub["orientation_group_rank"].iloc[0])
        usable_units = int(sub[sub["ssi_zscore_usable"].astype(bool)]["unit_index"].nunique())
        total_units = int(sub["unit_index"].nunique())
        for scale, scale_sub in sub.groupby("display_scale", sort=True):
            y = scale_sub["ssi_zscore_axis_mode"].to_numpy(dtype=float)
            y = y[np.isfinite(y)]
            n = int(y.size)
            summary_rows.append(
                {
                    "axis_mode": str(axis_mode),
                    "display_scale": float(scale),
                    "orientation_group": str(group),
                    "orientation_group_label": label,
                    "orientation_group_rank": rank,
                    "n_units_total": total_units,
                    "n_units_zscore_usable": usable_units,
                    "n_values": n,
                    "mean_zscore": float(np.nanmean(y)) if n else float("nan"),
                    "sem_zscore": float(np.nanstd(y, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
                    "median_zscore": float(np.nanmedian(y)) if n else float("nan"),
                    "mean_raw_ssi_bits_per_spike": float(np.nanmean(scale_sub[value_col].to_numpy(dtype=float))),
                    "zscore_contract": "mean/SEM of per-unit within-axis-mode z-scored displayed-movie instantaneous SSI",
                }
            )
    return z_rows, summary_rows


def plot_orientation_group_zscored_ssi_curves(
    out_dir: Path,
    z_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    dpi: int,
) -> tuple[Path, Path]:
    zdf = pd.DataFrame(z_rows)
    summary = pd.DataFrame(summary_rows)
    colors = {
        "contour_biased": "#18a6b8",
        "across_biased": "#d95f02",
        "off_axis_or_mixed": "#5b8a2f",
    }
    group_order = ["contour_biased", "across_biased", "off_axis_or_mixed"]
    axis_modes = ["across_sweep", "along_sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True, constrained_layout=True)
    for ax, axis_mode in zip(axes, axis_modes, strict=True):
        axis_z = zdf[(zdf["axis_mode"].astype(str) == axis_mode) & zdf["ssi_zscore_usable"].astype(bool)]
        axis_summary = summary[summary["axis_mode"].astype(str) == axis_mode]
        for group in group_order:
            group_summary = axis_summary[axis_summary["orientation_group"].astype(str) == str(group)]
            if group_summary.empty:
                continue
            group_summary = group_summary.sort_values("display_scale")
            rank = int(group_summary["orientation_group_rank"].iloc[0])
            label = str(group_summary["orientation_group_label"].iloc[0])
            n_units = int(group_summary["n_units_zscore_usable"].iloc[0])
            color = colors.get(str(group), "0.35")
            group_units = axis_z[axis_z["orientation_group"].astype(str) == str(group)]
            for unit, unit_sub in group_units.groupby("unit_index", sort=False):
                unit_sub = unit_sub.sort_values("display_scale")
                ax.plot(
                    unit_sub["display_scale"].to_numpy(dtype=float),
                    unit_sub["ssi_zscore_axis_mode"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.12,
                    linewidth=0.65,
                    zorder=1 + rank,
                )
            x = group_summary["display_scale"].to_numpy(dtype=float)
            y = group_summary["mean_zscore"].to_numpy(dtype=float)
            sem = group_summary["sem_zscore"].to_numpy(dtype=float)
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.16, linewidth=0.0, zorder=5 + rank)
            ax.plot(
                x,
                y,
                color=color,
                marker="o",
                linewidth=2.0,
                markersize=4.0,
                label=f"{label} (n={n_units})",
                zorder=10 + rank,
            )
        ax.axhline(0.0, color="0.55", linestyle="--", linewidth=0.8)
        ax.axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
        ax.grid(True, color="0.9", linewidth=0.7)
        ax.set_title(display_label_axis_mode(axis_mode, fixed_scale=fixed_scale_from_frame(axis_mode, axis_z)), fontsize=10)
        ax.set_xlabel("scale")
    axes[0].set_ylabel("SSI z-score within unit and axis mode")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "BackImage RR100 displayed-movie instantaneous SSI grouped by orientation tuning\n"
        "thin lines are units; thick lines are group mean +/- SEM; no mean-map SSI",
        fontsize=11,
    )
    png = out_dir / "backimage_rr100_orientation_group_zscored_ssi_curves.png"
    pdf = out_dir / "backimage_rr100_orientation_group_zscored_ssi_curves.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def orientation_cache_path(out_dir: Path) -> Path:
    return out_dir / "cache" / "backimage_rr100_orientation_probe_tuning.npz"


def orientation_axis_180(angle_deg: float) -> float:
    return float(float(angle_deg) % 180.0)


def angle_180_distance(a_deg: float, b_deg: float) -> float:
    return float(abs(((float(a_deg) - float(b_deg) + 90.0) % 180.0) - 90.0))


def make_windowed_grating_patch(
    size_px: int,
    *,
    orientation_deg: float,
    cycles_per_patch: float,
    contrast: float,
    sigma_frac: float,
) -> np.ndarray:
    size = int(size_px)
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    x = (xx - 0.5 * (size - 1)) / float(size)
    y = (yy - 0.5 * (size - 1)) / float(size)
    theta = math.radians(float(orientation_deg))
    # orientation_deg is the bar/edge axis; contrast varies along the orthogonal normal.
    normal_coord = -math.sin(theta) * x + math.cos(theta) * y
    carrier = np.cos(2.0 * math.pi * float(cycles_per_patch) * normal_coord)
    sigma = max(float(sigma_frac), 1e-3)
    window = np.exp(-0.5 * (x * x + y * y) / (sigma * sigma))
    image = 127.5 + 127.5 * float(contrast) * carrier * window
    return np.clip(image, 0.0, 255.0).astype(np.float32)


def render_orientation_probe(
    args: argparse.Namespace,
    *,
    selected_units: list[int],
    orientations: list[float],
    identity: dict[str, Any],
) -> dict[str, np.ndarray]:
    cache_file = orientation_cache_path(Path(args.out_dir))
    observed = load_cache(cache_file, identity) if not bool(args.force) else None
    if observed is not None:
        print(f"Loaded orientation-probe cache: {cache_file}")
        return observed

    view = load_population_view(version_name=str(args.rr100_version))
    scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(args.batch_size), empty_cache_every_batch=True)
    zero_trace = np.zeros((int(args.n_timepoints), 2), dtype=np.float32)
    unit_mean = np.zeros((len(orientations), len(selected_units)), dtype=np.float32)
    unit_max = np.zeros_like(unit_mean)
    unit_center = np.zeros_like(unit_mean)
    example_images: list[np.ndarray] = []
    for ori_idx, orientation in enumerate(orientations):
        patch = make_windowed_grating_patch(
            int(args.patch_size_px),
            orientation_deg=float(orientation),
            cycles_per_patch=float(args.orientation_probe_cycles_per_patch),
            contrast=float(args.orientation_probe_contrast),
            sigma_frac=float(args.orientation_probe_window_sigma_frac),
        )
        if ori_idx in {0, len(orientations) // 4, len(orientations) // 2}:
            example_images.append(patch)
        print(
            f"[backimage-orientation-probe] {ori_idx + 1}/{len(orientations)} "
            f"orientation={float(orientation):g}",
            flush=True,
        )
        full_map = rate_map_for_trace(scorer, patch, zero_trace)
        full_map = _align_response_to_trace(full_map, int(args.n_timepoints))
        rr100_map = apply_population_view(full_map, view).astype(np.float32, copy=False)
        mean_map = np.mean(rr100_map[:, selected_units], axis=0)
        unit_mean[ori_idx] = np.mean(mean_map, axis=(-2, -1))
        unit_max[ori_idx] = np.max(mean_map, axis=(-2, -1))
        cy = mean_map.shape[-2] // 2
        cx = mean_map.shape[-1] // 2
        unit_center[ori_idx] = mean_map[:, cy, cx]
        del full_map, rr100_map, mean_map
    payload = {
        "orientations_deg": np.asarray(orientations, dtype=np.float32),
        "selected_units": np.asarray(selected_units, dtype=np.int32),
        "unit_mean_rate": unit_mean,
        "unit_max_rate": unit_max,
        "unit_center_rate": unit_center,
        "example_gratings": np.stack(example_images, axis=0).astype(np.float32) if example_images else np.zeros((0, 1, 1), dtype=np.float32),
    }
    save_cache(cache_file, payload, identity)
    print(f"Saved orientation-probe cache: {cache_file}")
    return payload


def interpolate_orientation_curve(orientations_deg: np.ndarray, response: np.ndarray, target_deg: float) -> float:
    orientations = np.asarray(orientations_deg, dtype=float)
    y = np.asarray(response, dtype=float)
    target = float(target_deg) % 180.0
    x_closed = np.r_[orientations, orientations[0] + 180.0]
    y_closed = np.r_[y, y[0]]
    return float(np.interp(target, x_closed, y_closed))


def orientation_unit_summary_rows(
    payload: dict[str, np.ndarray],
    *,
    contour_axis_deg: float,
) -> list[dict[str, Any]]:
    orientations = np.asarray(payload["orientations_deg"], dtype=float)
    units = [int(v) for v in np.asarray(payload["selected_units"], dtype=int)]
    unit_mean = np.asarray(payload["unit_mean_rate"], dtype=float)
    contour_axis_gaze = orientation_axis_180(float(contour_axis_deg))
    across_axis_gaze = orientation_axis_180(float(contour_axis_deg) + 90.0)
    contour_axis_image = orientation_axis_180(gaze_axis_deg_to_image_axis_deg(float(contour_axis_deg)))
    across_axis_image = orientation_axis_180(gaze_axis_deg_to_image_axis_deg(float(contour_axis_deg)) + 90.0)
    rows: list[dict[str, Any]] = []
    for unit_pos, unit in enumerate(units):
        y = np.asarray(unit_mean[:, unit_pos], dtype=float)
        response = np.maximum(y, 0.0)
        denom = max(float(np.sum(response)), EPS)
        vector = np.sum(response * np.exp(2j * np.radians(orientations))) / denom
        preferred = orientation_axis_180(0.5 * math.degrees(math.atan2(float(vector.imag), float(vector.real))))
        osi = float(abs(vector))
        peak = float(np.nanmax(y)) if np.isfinite(y).any() else float("nan")
        trough = float(np.nanmin(y)) if np.isfinite(y).any() else float("nan")
        mean = float(np.nanmean(y)) if np.isfinite(y).any() else float("nan")
        dynamic = float(peak - trough) if np.isfinite(peak) and np.isfinite(trough) else float("nan")
        y_norm = y / max(peak, EPS) if np.isfinite(peak) and peak > EPS else np.zeros_like(y, dtype=float)
        contour_norm = interpolate_orientation_curve(orientations, y_norm, contour_axis_image)
        across_norm = interpolate_orientation_curve(orientations, y_norm, across_axis_image)
        contour_margin = float(contour_norm - across_norm)
        across_margin = float(across_norm - contour_norm)
        contour_display_delta = abs(float(preferred) - float(contour_axis_image))
        across_display_delta = abs(float(preferred) - float(across_axis_image))
        target_scale = math.sqrt(max(peak, 0.0)) * max(dynamic, 0.0)
        rows.append(
            {
                "unit_index": int(unit),
                "unit_label": f"u{int(unit):03d}",
                "preferred_orientation_deg": float(preferred),
                "probe_orientation_coordinate_frame": "image_array_x_right_y_down",
                "orientation_selectivity_index": osi,
                "orientation_probe_mean_rate": mean,
                "orientation_probe_peak_rate": peak,
                "orientation_probe_dynamic_range": dynamic,
                "orientation_probe_contour_norm_response": contour_norm,
                "orientation_probe_across_norm_response": across_norm,
                "orientation_probe_contour_minus_across_norm": contour_margin,
                "orientation_probe_across_minus_contour_norm": across_margin,
                "preferred_display_delta_from_contour_deg": contour_display_delta,
                "preferred_display_delta_from_across_deg": across_display_delta,
                "contour_selectivity_score": contour_margin * float(osi) * target_scale / (1.0 + contour_display_delta / 20.0),
                "across_selectivity_score": across_margin * float(osi) * target_scale / (1.0 + across_display_delta / 20.0),
                "off_axis_orientation_score": min(
                    angle_180_distance(preferred, contour_axis_image),
                    angle_180_distance(preferred, across_axis_image),
                )
                * float(osi)
                * target_scale,
                "contour_axis_image_deg_0_180": contour_axis_image,
                "across_axis_image_deg_0_180": across_axis_image,
                "contour_axis_gaze_deg_0_180": contour_axis_gaze,
                "across_axis_gaze_deg_0_180": across_axis_gaze,
                "preferred_delta_from_contour_deg": angle_180_distance(preferred, contour_axis_image),
                "preferred_delta_from_across_deg": angle_180_distance(preferred, across_axis_image),
            }
        )
    return rows


def orientation_candidate_pool(summary: pd.DataFrame) -> pd.DataFrame:
    pool = summary[
        np.isfinite(summary["preferred_orientation_deg"].astype(float))
        & np.isfinite(summary["orientation_selectivity_index"].astype(float))
        & np.isfinite(summary["orientation_probe_dynamic_range"].astype(float))
        & np.isfinite(summary["orientation_probe_peak_rate"].astype(float))
        & np.isfinite(summary["orientation_probe_contour_norm_response"].astype(float))
        & np.isfinite(summary["orientation_probe_across_norm_response"].astype(float))
        & (summary["orientation_probe_peak_rate"].astype(float) > 0.0)
    ].copy()
    if pool.empty:
        return summary.copy()
    min_osi = max(0.05, float(np.nanpercentile(pool["orientation_selectivity_index"].to_numpy(dtype=float), 25.0)))
    min_dynamic = float(np.nanpercentile(pool["orientation_probe_dynamic_range"].to_numpy(dtype=float), 25.0))
    min_peak = float(np.nanpercentile(pool["orientation_probe_peak_rate"].to_numpy(dtype=float), 20.0))
    filtered = pool[
        (pool["orientation_selectivity_index"].astype(float) >= min_osi)
        & (pool["orientation_probe_dynamic_range"].astype(float) >= min_dynamic)
        & (pool["orientation_probe_peak_rate"].astype(float) >= min_peak)
    ].copy()
    return filtered if not filtered.empty else pool


def select_orientation_tuning_units(
    args: argparse.Namespace,
    metric_rows: list[dict[str, Any]],
    orientation_payload: dict[str, np.ndarray],
    *,
    contour_axis_deg: float,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    requested = parse_int_list(str(args.units))
    summary_rows = orientation_unit_summary_rows(orientation_payload, contour_axis_deg=float(contour_axis_deg))
    if requested:
        unit_roles = []
        for pos, unit in enumerate(requested):
            role = ["requested_1", "requested_2", "requested_3"][pos] if pos < 3 else f"requested_{pos + 1}"
            unit_roles.append((int(unit), role))
        selected_units, selected_rows = selected_rows_from_roles(
            metric_rows,
            unit_roles,
            orientation_summary_rows=summary_rows,
        )
        return selected_units, selected_rows, summary_rows

    metrics = pd.DataFrame(metric_rows)
    summary = pd.DataFrame(summary_rows)
    merged = metrics.merge(summary, on=["unit_index", "unit_label"], how="inner")
    pool = orientation_candidate_pool(merged)
    contour_pool = pool[
        (pool["preferred_delta_from_contour_deg"].astype(float) <= 20.0)
        & (pool["preferred_display_delta_from_contour_deg"].astype(float) <= 30.0)
        & (pool["orientation_probe_contour_norm_response"].astype(float) >= 0.70)
        & (pool["orientation_probe_across_norm_response"].astype(float) <= 0.55)
        & (pool["orientation_probe_contour_minus_across_norm"].astype(float) > 0.0)
    ].copy()
    if contour_pool.empty:
        contour_pool = pool[pool["orientation_probe_contour_minus_across_norm"].astype(float) > 0.0].copy()
    aligned = contour_pool.sort_values(
        by=[
            "orientation_probe_contour_minus_across_norm",
            "contour_selectivity_score",
            "preferred_display_delta_from_contour_deg",
        ],
        ascending=[False, False, True],
    ).iloc[0]
    orth_pool = pool[pool["unit_index"].astype(int) != int(aligned["unit_index"])].copy()
    across_pool = orth_pool[
        (orth_pool["preferred_delta_from_across_deg"].astype(float) <= 20.0)
        & (orth_pool["preferred_display_delta_from_across_deg"].astype(float) <= 30.0)
        & (orth_pool["orientation_probe_across_norm_response"].astype(float) >= 0.70)
        & (orth_pool["orientation_probe_contour_norm_response"].astype(float) <= 0.45)
        & (orth_pool["orientation_probe_across_minus_contour_norm"].astype(float) > 0.0)
    ].copy()
    if across_pool.empty:
        across_pool = orth_pool[orth_pool["orientation_probe_across_minus_contour_norm"].astype(float) > 0.0].copy()
    orthogonal = across_pool.sort_values(
        by=[
            "across_selectivity_score",
            "preferred_display_delta_from_across_deg",
            "orientation_probe_across_minus_contour_norm",
        ],
        ascending=[False, True, False],
    ).iloc[0]
    chosen = {int(aligned["unit_index"]), int(orthogonal["unit_index"])}
    off_axis_pool = pool[
        (~pool["unit_index"].astype(int).isin(chosen))
        & (pool["preferred_delta_from_contour_deg"].astype(float) >= 25.0)
        & (pool["preferred_delta_from_across_deg"].astype(float) >= 25.0)
    ].copy()
    if off_axis_pool.empty:
        off_axis_pool = pool[~pool["unit_index"].astype(int).isin(chosen)].copy()
    third_unit = int(
        off_axis_pool.sort_values(
            by=["off_axis_orientation_score", "orientation_selectivity_index", "orientation_probe_dynamic_range"],
            ascending=[False, False, False],
        ).iloc[0]["unit_index"]
    )
    unit_roles = [
        (int(aligned["unit_index"]), "orientation_tuning_aligned_with_contour"),
        (int(orthogonal["unit_index"]), "orientation_tuning_orthogonal_to_contour"),
        (int(third_unit), "off_axis_orientation_control"),
    ]
    selected_units, selected_rows = selected_rows_from_roles(
        metric_rows,
        unit_roles,
        orientation_summary_rows=summary_rows,
    )
    return selected_units, selected_rows, summary_rows


def subset_orientation_payload(payload: dict[str, np.ndarray], selected_units: list[int]) -> dict[str, np.ndarray]:
    available = [int(v) for v in np.asarray(payload["selected_units"], dtype=int)]
    position_by_unit = {unit: pos for pos, unit in enumerate(available)}
    positions = [position_by_unit[int(unit)] for unit in selected_units]
    out: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        arr = np.asarray(value)
        if key in {"unit_mean_rate", "unit_max_rate", "unit_center_rate"}:
            out[key] = arr[:, positions]
        elif key == "selected_units":
            out[key] = np.asarray(selected_units, dtype=np.int32)
        else:
            out[key] = arr
    return out


def orientation_tuning_rows(
    payload: dict[str, np.ndarray],
    selected_rows: list[dict[str, Any]],
    *,
    contour_axis_deg: float,
) -> list[dict[str, Any]]:
    orientations = np.asarray(payload["orientations_deg"], dtype=float)
    units = [int(v) for v in np.asarray(payload["selected_units"], dtype=int)]
    unit_mean = np.asarray(payload["unit_mean_rate"], dtype=float)
    unit_max = np.asarray(payload["unit_max_rate"], dtype=float)
    unit_center = np.asarray(payload["unit_center_rate"], dtype=float)
    rows: list[dict[str, Any]] = []
    contour_axis_gaze = orientation_axis_180(float(contour_axis_deg))
    across_axis_gaze = orientation_axis_180(float(contour_axis_deg) + 90.0)
    contour_axis_image = orientation_axis_180(gaze_axis_deg_to_image_axis_deg(float(contour_axis_deg)))
    across_axis_image = orientation_axis_180(gaze_axis_deg_to_image_axis_deg(float(contour_axis_deg)) + 90.0)
    selected_lookup = {int(row["unit_index"]): row for row in selected_rows}
    for unit_pos, unit in enumerate(units):
        response = np.maximum(unit_mean[:, unit_pos], 0.0)
        denom = max(float(np.sum(response)), EPS)
        vector = np.sum(response * np.exp(2j * np.radians(orientations))) / denom
        preferred = orientation_axis_180(0.5 * math.degrees(math.atan2(float(vector.imag), float(vector.real))))
        osi = float(abs(vector))
        for ori_idx, orientation in enumerate(orientations):
            rows.append(
                {
                    "unit_index": int(unit),
                    "unit_label": f"u{int(unit):03d}",
                    "selection_role": str(selected_lookup[int(unit)].get("selection_role", "")),
                    "probe_orientation_deg": float(orientation),
                    "probe_unit_mean_rate": float(unit_mean[ori_idx, unit_pos]),
                    "probe_unit_max_rate": float(unit_max[ori_idx, unit_pos]),
                    "probe_unit_center_rate": float(unit_center[ori_idx, unit_pos]),
                    "preferred_orientation_deg": float(preferred),
                    "probe_orientation_coordinate_frame": "image_array_x_right_y_down",
                    "orientation_selectivity_index": osi,
                    "contour_axis_image_deg_0_180": contour_axis_image,
                    "across_axis_image_deg_0_180": across_axis_image,
                    "contour_axis_gaze_deg_0_180": contour_axis_gaze,
                    "across_axis_gaze_deg_0_180": across_axis_gaze,
                    "preferred_delta_from_contour_deg": angle_180_distance(preferred, contour_axis_image),
                    "preferred_delta_from_across_deg": angle_180_distance(preferred, across_axis_image),
                    "probe_contract": "centered windowed sinusoidal grating; static trace; RR100 spatial mean rate",
                }
            )
    return rows


def plot_orientation_axes_reference(
    out_dir: Path,
    patch: np.ndarray,
    orientation_payload: dict[str, np.ndarray],
    selected_rows: list[dict[str, Any]],
    trial: pd.Series,
    *,
    contour_axis_deg: float,
    dpi: int,
) -> tuple[Path, Path]:
    orientations = np.asarray(orientation_payload["orientations_deg"], dtype=float)
    units = [int(v) for v in np.asarray(orientation_payload["selected_units"], dtype=int)]
    unit_mean = np.asarray(orientation_payload["unit_mean_rate"], dtype=float)
    selected = pd.DataFrame(selected_rows)
    contour_axis_gaze = orientation_axis_180(float(contour_axis_deg))
    contour_axis_image = orientation_axis_180(gaze_axis_deg_to_image_axis_deg(float(contour_axis_deg)))
    across_axis_image = orientation_axis_180(gaze_axis_deg_to_image_axis_deg(float(contour_axis_deg)) + 90.0)

    fig = plt.figure(figsize=(11.5, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(len(units), 2, width_ratios=[1.05, 1.45])
    ax_image = fig.add_subplot(gs[:, 0])
    patch_img = np.asarray(patch, dtype=np.float32)
    vmin, vmax = image_scale([patch_img], 1.0, 99.0)
    ax_image.imshow(patch_img, cmap="gray", vmin=vmin, vmax=vmax)
    h, w = patch_img.shape
    cx = 0.5 * (w - 1)
    cy = 0.5 * (h - 1)
    length = 0.38 * min(h, w)
    line_specs = [
        ("along contour", contour_axis_image, "#18a6b8", 4.0),
        ("across contour", across_axis_image, "#d95f02", 3.2),
    ]
    for label, angle, color, linewidth in line_specs:
        unit_dx, unit_dy = axis_vector_image(float(angle))
        dx = unit_dx * length
        dy = unit_dy * length
        ax_image.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=color, linewidth=linewidth, solid_capstyle="round")
        ax_image.annotate(
            label,
            xy=(cx + 0.58 * dx, cy + 0.58 * dy),
            xytext=(cx + 0.72 * dx, cy + 0.72 * dy),
            color=color,
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.2},
        )
    ax_image.scatter([cx], [cy], s=20, color="white", edgecolor="black", linewidth=0.8, zorder=4)
    ax_image.set_title(
        f"source row {int(trial['source_row'])}; contour axis "
        f"{float(contour_axis_deg):.1f} deg gaze / {contour_axis_image:.1f} deg image",
        fontsize=10,
    )
    ax_image.set_xticks([])
    ax_image.set_yticks([])

    colors = ["#2c6db2", "#b24f2c", "#5b8a2f", "#6b4c9a"]
    for row_idx, unit in enumerate(units):
        ax = fig.add_subplot(gs[row_idx, 1])
        y = unit_mean[:, row_idx]
        y_norm = y / max(float(np.nanmax(y)), EPS)
        row = selected[selected["unit_index"].astype(int) == int(unit)].iloc[0]
        response = np.maximum(y, 0.0)
        denom = max(float(np.sum(response)), EPS)
        vector = np.sum(response * np.exp(2j * np.radians(orientations))) / denom
        preferred = orientation_axis_180(0.5 * math.degrees(math.atan2(float(vector.imag), float(vector.real))))
        osi = float(abs(vector))
        closed_x = np.r_[orientations, orientations[0] + 180.0]
        closed_y = np.r_[y_norm, y_norm[0]]
        ax.plot(closed_x, closed_y, marker="o", markersize=3.0, color=colors[row_idx % len(colors)], linewidth=1.5)
        for x, label, color, linestyle in [
            (contour_axis_image, "contour", "#18a6b8", "-"),
            (across_axis_image, "across", "#d95f02", "-"),
            (preferred, "pref", "0.15", ":"),
        ]:
            ax.axvline(x, color=color, linestyle=linestyle, linewidth=1.1)
            ax.text(x, 1.03, label, color=color, fontsize=7, ha="center", va="bottom", rotation=90)
        ax.set_xlim(0.0, 180.0)
        ax.set_ylim(0.0, 1.12)
        ax.set_ylabel(f"{row['unit_label']}\nnorm rate", rotation=0, ha="right", va="center", labelpad=34)
        ax.set_title(
            f"{row['selection_role']} | pref {preferred:.1f} deg, OSI {osi:.2f}",
            fontsize=8,
        )
        ax.grid(True, color="0.9", linewidth=0.7)
        if row_idx == len(units) - 1:
            ax.set_xlabel("probe grating bar orientation (image-array deg)")
        else:
            ax.set_xticklabels([])
    fig.suptitle(
        "BackImage contour axes and RR100 unit orientation-tuning reference\n"
        f"tuning probe uses centered windowed gratings with static trace; contour markers use image-array axes "
        f"(stored gaze contour {float(contour_axis_deg):.1f} deg; axial {contour_axis_gaze:.1f} deg)",
        fontsize=11,
    )
    png = out_dir / "backimage_rr100_orientation_tuning_and_contour_axes_reference.png"
    pdf = out_dir / "backimage_rr100_orientation_tuning_and_contour_axes_reference.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = parse_float_list(str(args.scales))
    specs, refs = condition_rows(scales, fixed_scale=float(args.sweep_fixed_scale))

    select_args = argparse.Namespace(
        axis_run_dir=Path(args.axis_run_dir),
        source_trace_scale=float(args.source_trace_scale),
        source_trace_prior_family=str(args.source_trace_prior_family),
        axis_column=str(args.axis_column),
        max_trials=0,
        trial_start=0,
        n_timepoints=int(args.n_timepoints),
    )
    trials, source_meta = select_source_trials(select_args)
    production_cache = load_npz(Path(args.source_run_dir) / "cache" / "backimage_contour_axis_rr100_spatial_ssi_cache.npz")
    expected = np.asarray(production_cache["movie_trial_id"], dtype=int)
    observed = trials["trial_id"].to_numpy(dtype=int)
    if expected.shape != observed.shape or not np.array_equal(expected, observed):
        raise ValueError("Reconstructed trial order does not match source-run movie_trial_id.")
    movie_index, trial = choose_movie(trials, args)
    compact_frame = choose_compact_frame(np.asarray(trial["source_trace"], dtype=np.float32), args.compact_frame)
    identity = {
        "analysis": "backimage_rr100_instantaneous_unit_maps",
        "source_run_dir": Path(args.source_run_dir).resolve(),
        "axis_run_dir": Path(args.axis_run_dir).resolve(),
        "rr100_version": str(args.rr100_version),
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "movie_index": int(movie_index),
        "trial_id": int(trial["trial_id"]),
        "source_row": int(trial["source_row"]),
        "axis_column": str(args.axis_column),
        "source_trace_scale": float(args.source_trace_scale),
        "source_trace_prior_family": str(args.source_trace_prior_family),
        "sweep_fixed_scale": float(args.sweep_fixed_scale),
        "n_timepoints": int(args.n_timepoints),
        "patch_size_px": int(args.patch_size_px),
        "condition_specs": specs,
    }
    write_json(out_dir / "request_identity.json", identity)
    write_csv(out_dir / "condition_specs.csv", specs)
    write_csv(out_dir / "condition_display_refs.csv", refs)

    if bool(args.dry_run):
        print(json.dumps(json_ready(identity), indent=2, sort_keys=True))
        return

    path = cache_path(out_dir)
    cached = None if bool(args.force) else load_cache(path, identity)
    patch_cache = out_dir / "cache" / "selected_patch.npy"
    if cached is None:
        payload, patch = render_maps(args, trial, specs)
        save_cache(path, payload, identity)
        patch_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(patch_cache, np.asarray(patch, dtype=np.float32))
        print(f"Saved instantaneous map cache: {path}")
    else:
        payload = cached
        if not patch_cache.exists():
            patch, _patch_meta = _extract_patch(trial, canvas_cache={}, patch_size_px=int(args.patch_size_px))
            np.save(patch_cache, np.asarray(patch, dtype=np.float32))
        patch = np.load(patch_cache)
        print(f"Loaded instantaneous map cache: {path}")

    metric_rows = unit_metric_rows(
        args,
        payload,
        contour_axis_deg=float(trial[str(args.axis_column)]),
        compact_frame=int(compact_frame),
    )
    orientation_probe_degrees = parse_float_list(str(args.orientation_probe_deg))
    requested_units = parse_int_list(str(args.units))
    orientation_probe_units = requested_units or list(range(int(np.asarray(payload["maps"]).shape[2])))
    orientation_identity = {
        "analysis": "backimage_rr100_orientation_probe_tuning",
        "rr100_version": str(args.rr100_version),
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "selected_units": orientation_probe_units,
        "orientation_probe_degrees": orientation_probe_degrees,
        "orientation_probe_cycles_per_patch": float(args.orientation_probe_cycles_per_patch),
        "orientation_probe_contrast": float(args.orientation_probe_contrast),
        "orientation_probe_window_sigma_frac": float(args.orientation_probe_window_sigma_frac),
        "orientation_probe_coordinate_frame": "image_array_x_right_y_down",
        "n_timepoints": int(args.n_timepoints),
        "patch_size_px": int(args.patch_size_px),
    }
    orientation_payload_all = render_orientation_probe(
        args,
        selected_units=orientation_probe_units,
        orientations=orientation_probe_degrees,
        identity=orientation_identity,
    )
    selected_units, selected_rows, orientation_summary_rows_all = select_orientation_tuning_units(
        args,
        metric_rows,
        orientation_payload_all,
        contour_axis_deg=float(trial[str(args.axis_column)]),
    )
    orientation_summary_csv = out_dir / "orientation_probe_unit_summary.csv"
    write_csv(orientation_summary_csv, orientation_summary_rows_all)
    orientation_payload = subset_orientation_payload(orientation_payload_all, selected_units)
    write_csv(out_dir / "selected_units.csv", selected_rows)
    annotate_frame_ssi_units = parse_int_list(str(args.annotate_frame_ssi_units))
    annotate_frame_ssi_rows: list[dict[str, Any]] = []
    annotate_frame_ssi_units_csv: Path | None = None
    if annotate_frame_ssi_units:
        _, annotate_frame_ssi_rows = selected_rows_from_roles(
            metric_rows,
            [(int(unit), "frame_ssi_annotated") for unit in annotate_frame_ssi_units],
            orientation_summary_rows=orientation_summary_rows_all,
        )
        annotate_frame_ssi_units_csv = out_dir / "frame_ssi_annotated_units.csv"
        write_csv(annotate_frame_ssi_units_csv, annotate_frame_ssi_rows)

    all_orientation_units = [int(v) for v in np.asarray(orientation_payload_all["selected_units"], dtype=int)]
    all_ssi_rows = compute_displayed_ssi(payload, all_orientation_units, refs)
    all_ssi_csv = out_dir / "displayed_movie_instantaneous_ssi_all_units.csv"
    write_csv(all_ssi_csv, all_ssi_rows)
    selected_unit_order = {int(unit): pos for pos, unit in enumerate(selected_units)}
    axis_order = {"across_sweep": 0, "along_sweep": 1}
    ssi_rows = [row for row in all_ssi_rows if int(row["unit_index"]) in selected_unit_order]
    ssi_rows.sort(
        key=lambda row: (
            selected_unit_order[int(row["unit_index"])],
            axis_order.get(str(row["axis_mode"]), 99),
            float(row["display_scale"]),
        )
    )
    write_csv(out_dir / "displayed_movie_instantaneous_ssi.csv", ssi_rows)
    orientation_group_rows = classify_orientation_tuning_groups(orientation_summary_rows_all)
    orientation_groups_csv = out_dir / "orientation_tuning_groups.csv"
    write_csv(orientation_groups_csv, orientation_group_rows)
    orientation_group_z_rows, orientation_group_summary_rows = zscore_orientation_group_ssi_rows(
        all_ssi_rows,
        orientation_group_rows,
    )
    orientation_group_z_csv = out_dir / "orientation_group_zscored_ssi_curves.csv"
    orientation_group_summary_csv = out_dir / "orientation_group_zscored_ssi_summary.csv"
    write_csv(orientation_group_z_csv, orientation_group_z_rows)
    write_csv(orientation_group_summary_csv, orientation_group_summary_rows)
    orientation_group_png, orientation_group_pdf = plot_orientation_group_zscored_ssi_curves(
        out_dir,
        orientation_group_z_rows,
        orientation_group_summary_rows,
        dpi=int(args.dpi),
    )
    compact_png, compact_pdf = plot_compact(
        out_dir,
        payload,
        np.asarray(patch, dtype=np.float32),
        selected_units,
        selected_rows,
        refs,
        scales,
        trial,
        compact_frame=int(compact_frame),
        dpi=int(args.dpi),
        vmin_percentile=float(args.map_vmin_percentile),
        vmax_percentile=float(args.map_vmax_percentile),
    )
    full_pdf = plot_full_timepoint_pdf(
        out_dir,
        payload,
        selected_units,
        selected_rows,
        refs,
        scales,
        trial,
        vmin_percentile=float(args.map_vmin_percentile),
        vmax_percentile=float(args.map_vmax_percentile),
    )
    annotated_frame_ssi_pdf: Path | None = None
    annotated_frame_ssi_pngs: list[Path] = []
    frame_ssi_timecourse_csv: Path | None = None
    frame_ssi_timecourse_pdf: Path | None = None
    frame_ssi_timecourse_pngs: list[Path] = []
    frame_sharpness_method_pdf: Path | None = None
    frame_sharpness_method_pngs: list[Path] = []
    if annotate_frame_ssi_units:
        annotated_frame_ssi_pdf, annotated_frame_ssi_pngs = plot_frame_ssi_annotated_unit_pdf(
            out_dir,
            payload,
            annotate_frame_ssi_units,
            annotate_frame_ssi_rows,
            refs,
            scales,
            trial,
            vmin_percentile=float(args.map_vmin_percentile),
            vmax_percentile=float(args.map_vmax_percentile),
            dpi=int(args.dpi),
        )
        frame_ssi_timecourse_rows_out = frame_ssi_timecourse_rows(payload, annotate_frame_ssi_units, refs)
        frame_ssi_timecourse_csv = out_dir / "frame_ssi_timecourses.csv"
        write_csv(frame_ssi_timecourse_csv, frame_ssi_timecourse_rows_out)
        frame_ssi_timecourse_pdf, frame_ssi_timecourse_pngs = plot_frame_ssi_timecourse_overlay(
            out_dir,
            frame_ssi_timecourse_rows_out,
            annotate_frame_ssi_units,
            annotate_frame_ssi_rows,
            refs,
            trial,
            dpi=int(args.dpi),
        )
        frame_sharpness_method_pdf, frame_sharpness_method_pngs = plot_frame_sharpness_method_comparison(
            out_dir,
            frame_ssi_timecourse_rows_out,
            annotate_frame_ssi_units,
            annotate_frame_ssi_rows,
            refs,
            trial,
            dpi=int(args.dpi),
        )
    curves_png, curves_pdf = plot_displayed_ssi_curves(out_dir, ssi_rows, selected_rows, dpi=int(args.dpi))
    orientation_rows = orientation_tuning_rows(
        orientation_payload,
        selected_rows,
        contour_axis_deg=float(trial[str(args.axis_column)]),
    )
    orientation_csv = out_dir / "orientation_probe_tuning.csv"
    write_csv(orientation_csv, orientation_rows)
    orientation_png, orientation_pdf = plot_orientation_axes_reference(
        out_dir,
        np.asarray(patch, dtype=np.float32),
        orientation_payload,
        selected_rows,
        trial,
        contour_axis_deg=float(trial[str(args.axis_column)]),
        dpi=int(args.dpi),
    )
    summary = {
        "source_meta": source_meta,
        "movie_index": int(movie_index),
        "trial_id": int(trial["trial_id"]),
        "source_row": int(trial["source_row"]),
        "session": str(trial["session"]),
        "trial_idx": int(trial["trial_idx"]),
        "axis_deg": float(trial[str(args.axis_column)]),
        "compact_frame": int(compact_frame),
        "selected_units": selected_rows,
        "outputs": {
            "cache": path,
            "selected_units_csv": out_dir / "selected_units.csv",
            "displayed_movie_instantaneous_ssi_csv": out_dir / "displayed_movie_instantaneous_ssi.csv",
            "displayed_movie_instantaneous_ssi_all_units_csv": all_ssi_csv,
            "compact_png": compact_png,
            "compact_pdf": compact_pdf,
            "all_timepoints_pdf": full_pdf,
            "displayed_movie_ssi_curves_png": curves_png,
            "displayed_movie_ssi_curves_pdf": curves_pdf,
            "orientation_probe_cache": orientation_cache_path(out_dir),
            "orientation_probe_unit_summary_csv": orientation_summary_csv,
            "orientation_probe_tuning_csv": orientation_csv,
            "orientation_tuning_axes_png": orientation_png,
            "orientation_tuning_axes_pdf": orientation_pdf,
            "orientation_tuning_groups_csv": orientation_groups_csv,
            "orientation_group_zscored_ssi_csv": orientation_group_z_csv,
            "orientation_group_zscored_ssi_summary_csv": orientation_group_summary_csv,
            "orientation_group_zscored_ssi_curves_png": orientation_group_png,
            "orientation_group_zscored_ssi_curves_pdf": orientation_group_pdf,
            "frame_ssi_annotated_units_csv": annotate_frame_ssi_units_csv,
            "frame_ssi_annotated_pdf": annotated_frame_ssi_pdf,
            "frame_ssi_annotated_pngs": annotated_frame_ssi_pngs,
            "frame_ssi_timecourses_csv": frame_ssi_timecourse_csv,
            "frame_ssi_timecourses_pdf": frame_ssi_timecourse_pdf,
            "frame_ssi_timecourse_pngs": frame_ssi_timecourse_pngs,
            "frame_sharpness_method_comparison_pdf": frame_sharpness_method_pdf,
            "frame_sharpness_method_comparison_pngs": frame_sharpness_method_pngs,
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(f"Selected movie index {movie_index}, source_row {int(trial['source_row'])}, frame {compact_frame}")
    print("Selected units: " + ", ".join(f"u{unit:03d}" for unit in selected_units))
    print(f"Wrote compact PDF: {compact_pdf}")
    print(f"Wrote all-timepoints PDF: {full_pdf}")
    print(f"Wrote displayed-movie SSI curves: {curves_pdf}")
    print(f"Wrote orientation tuning/axes reference: {orientation_pdf}")
    print(f"Wrote orientation-group z-scored SSI curves: {orientation_group_pdf}")
    if annotated_frame_ssi_pdf is not None:
        print(f"Wrote frame-SSI annotated map PDF: {annotated_frame_ssi_pdf}")
    if frame_ssi_timecourse_pdf is not None:
        print(f"Wrote frame-SSI timecourse PDF: {frame_ssi_timecourse_pdf}")
    if frame_sharpness_method_pdf is not None:
        print(f"Wrote frame-sharpness method comparison PDF: {frame_sharpness_method_pdf}")


if __name__ == "__main__":
    main()
