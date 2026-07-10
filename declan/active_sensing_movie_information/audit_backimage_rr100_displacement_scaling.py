#!/usr/bin/env python3
"""Audit BackImage RR100 trace scaling and model-pixel displacement contracts.

This script checks the coordinate bookkeeping behind the BackImage RR100
contour-axis SSI analyses without re-running the V1 model.  It verifies three
contracts:

1. cached condition traces are exactly the source trace decomposed into the
   local contour-axis basis and recombined with the saved along/across scales;
2. the population cache condition specs imply the requested 0x/0.5x/1x/2x/3x
   component scaling over all source windows;
3. the shared counterfactual-stimulus helper maps degrees to pixels with the
   expected PPD, sign, and x/y ordering used by the BackImage scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    DEFAULT_AXIS_RUN_DIR,
    combined_axis_trace,
    scale_token,
    select_source_trials,
    trace_path_length,
    trace_rms,
)
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _trace_xy_to_twin_helper_order,
)
from jake.twininfo.common import N_LAGS, OUT_SIZE, PPD, make_counterfactual_stim


DEFAULT_UNIT_MAP_CACHE = (
    ROOT
    / "outputs"
    / "active_sensing_movie_information"
    / "backimage_rr100_instantaneous_unit_maps_latest_v1"
    / "cache"
    / "backimage_rr100_instantaneous_unit_maps.npz"
)
DEFAULT_POP_CACHE = (
    ROOT
    / "outputs"
    / "active_sensing_movie_information"
    / "backimage_contour_axis_rr100_spatial_ssi_n128_across_sweep_v1"
    / "cache"
    / "backimage_contour_axis_rr100_spatial_ssi_cache.npz"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "active_sensing_movie_information"
    / "backimage_rr100_displacement_scaling_audit"
)
TRACE_TOL_DEG = 5e-7
PIXEL_TOL = 2e-3
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-map-cache", type=Path, default=DEFAULT_UNIT_MAP_CACHE)
    parser.add_argument("--population-cache", type=Path, default=DEFAULT_POP_CACHE)
    parser.add_argument("--axis-run-dir", type=Path, default=DEFAULT_AXIS_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--source-trace-scale", type=float, default=1.0)
    parser.add_argument("--source-trace-prior-family", type=str, default="axis_edge_parallel")
    parser.add_argument("--axis-column", type=str, default="image_edge_axis_deg")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def cache_identity(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    if "cache_identity_json" not in payload:
        return {}
    return json.loads(str(np.asarray(payload["cache_identity_json"]).ravel()[0]))


def axis_vectors(axis_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.radians(float(axis_deg))
    along = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)
    across = np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    return along, across


def centered_trace(trace: np.ndarray) -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float64)
    return arr - np.mean(arr, axis=0, keepdims=True)


def projection_rms(trace: np.ndarray, unit_vector: np.ndarray) -> float:
    proj = centered_trace(trace) @ np.asarray(unit_vector, dtype=np.float64)
    return float(np.sqrt(np.mean(proj * proj))) if proj.size else 0.0


def expected_combined_rms(
    base_along_rms: float,
    base_across_rms: float,
    *,
    along_scale: float,
    across_scale: float,
) -> float:
    return float(
        np.sqrt(
            (float(along_scale) * float(base_along_rms)) ** 2
            + (float(across_scale) * float(base_across_rms)) ** 2
        )
    )


def audit_unit_map_cache(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = load_npz(path)
    source_trace = np.asarray(payload["source_trace"], dtype=np.float32)
    traces = np.asarray(payload["condition_traces"], dtype=np.float32)
    axis_deg = float(np.asarray(payload["axis_deg"]).ravel()[0])
    condition_id = np.asarray(payload["condition_id"]).astype(str)
    along_scales = np.asarray(payload["condition_along_scale"], dtype=np.float64)
    across_scales = np.asarray(payload["condition_across_scale"], dtype=np.float64)
    along_u, across_u = axis_vectors(axis_deg)
    base_along_rms = projection_rms(source_trace, along_u)
    base_across_rms = projection_rms(source_trace, across_u)

    rows: list[dict[str, Any]] = []
    max_trace_error = 0.0
    max_component_error = 0.0
    for idx, (cond, along_scale, across_scale) in enumerate(
        zip(condition_id, along_scales, across_scales, strict=True)
    ):
        expected, meta = combined_axis_trace(
            source_trace,
            axis_deg=axis_deg,
            along_scale=float(along_scale),
            across_scale=float(across_scale),
        )
        actual = np.asarray(traces[idx], dtype=np.float64)
        diff = actual - np.asarray(expected, dtype=np.float64)
        actual_along_rms = projection_rms(actual, along_u)
        actual_across_rms = projection_rms(actual, across_u)
        expected_along_rms = abs(float(along_scale)) * base_along_rms
        expected_across_rms = abs(float(across_scale)) * base_across_rms
        component_error = max(
            abs(actual_along_rms - expected_along_rms),
            abs(actual_across_rms - expected_across_rms),
        )
        trace_error = float(np.max(np.abs(diff))) if diff.size else 0.0
        max_trace_error = max(max_trace_error, trace_error)
        max_component_error = max(max_component_error, component_error)
        rows.append(
            {
                "cache": "unit_map_latest",
                "condition_index": int(idx),
                "condition_id": str(cond),
                "along_scale": float(along_scale),
                "across_scale": float(across_scale),
                "source_trace_rms_deg": trace_rms(source_trace),
                "base_along_component_rms_deg": float(base_along_rms),
                "base_across_component_rms_deg": float(base_across_rms),
                "expected_along_component_rms_deg": float(expected_along_rms),
                "actual_along_component_rms_deg": float(actual_along_rms),
                "expected_across_component_rms_deg": float(expected_across_rms),
                "actual_across_component_rms_deg": float(actual_across_rms),
                "expected_output_rms_deg": expected_combined_rms(
                    base_along_rms,
                    base_across_rms,
                    along_scale=float(along_scale),
                    across_scale=float(across_scale),
                ),
                "actual_output_rms_deg": trace_rms(actual),
                "actual_output_rms_px": trace_rms(actual) * float(PPD),
                "actual_path_length_deg": trace_path_length(actual),
                "recomputed_output_rms_deg": float(meta["output_trace_rms_deg"]),
                "max_abs_cached_minus_recomputed_deg": trace_error,
                "max_abs_component_rms_error_deg": component_error,
                "status": "PASS" if trace_error <= TRACE_TOL_DEG and component_error <= TRACE_TOL_DEG else "FAIL",
            }
        )
    summary = {
        "path": path,
        "axis_deg": axis_deg,
        "source_trace_rms_deg": trace_rms(source_trace),
        "source_trace_rms_px": trace_rms(source_trace) * float(PPD),
        "source_trace_path_length_deg": trace_path_length(source_trace),
        "base_along_component_rms_deg": base_along_rms,
        "base_across_component_rms_deg": base_across_rms,
        "base_along_component_rms_px": base_along_rms * float(PPD),
        "base_across_component_rms_px": base_across_rms * float(PPD),
        "max_abs_cached_minus_recomputed_deg": max_trace_error,
        "max_abs_component_rms_error_deg": max_component_error,
        "n_conditions": int(condition_id.size),
        "status": "PASS" if max_trace_error <= TRACE_TOL_DEG and max_component_error <= TRACE_TOL_DEG else "FAIL",
    }
    return rows, summary


def condition_specs_from_cache(payload: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    ident = cache_identity(payload)
    specs = ident.get("condition_specs", [])
    if specs:
        return list(specs)
    condition_id = np.asarray(payload["condition_id"]).astype(str)
    along = np.asarray(payload["along_scale"], dtype=np.float64)
    across = np.asarray(payload["across_scale"], dtype=np.float64)
    is_static = np.asarray(payload["is_static_baseline"], dtype=bool)
    return [
        {
            "condition_id": str(condition_id[i]),
            "along_scale": float(along[i]),
            "across_scale": float(across[i]),
            "is_static_baseline": bool(is_static[i]),
        }
        for i in range(condition_id.size)
    ]


def source_trials_for_population_cache(args: argparse.Namespace, pop_payload: dict[str, np.ndarray]) -> tuple[Any, dict[str, Any]]:
    ident = cache_identity(pop_payload)
    ns = SimpleNamespace(
        axis_run_dir=Path(ident.get("axis_run_dir", args.axis_run_dir)),
        source_trace_scale=float(ident.get("source_trace_scale", args.source_trace_scale)),
        source_trace_prior_family=str(ident.get("source_trace_prior_family", args.source_trace_prior_family)),
        axis_column=str(ident.get("axis_column", args.axis_column)),
        n_timepoints=int(ident.get("n_timepoints", args.n_timepoints)),
        max_trials=0,
        trial_start=0,
    )
    trials, meta = select_source_trials(ns)
    wanted_rows = [int(v) for v in np.asarray(pop_payload.get("movie_source_row", []), dtype=np.int64).tolist()]
    if wanted_rows:
        by_source = trials.drop_duplicates("source_row").set_index("source_row", drop=False)
        missing = sorted(set(wanted_rows).difference(set(int(v) for v in by_source.index.to_list())))
        if missing:
            raise ValueError(f"Could not reconstruct source rows from population cache: {missing[:10]}")
        trials = by_source.loc[wanted_rows].reset_index(drop=True)
    return trials, meta


def audit_population_cache(args: argparse.Namespace, path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payload = load_npz(path)
    ident = cache_identity(payload)
    axis_column = str(ident.get("axis_column", args.axis_column))
    specs = condition_specs_from_cache(payload)
    trials, meta = source_trials_for_population_cache(args, payload)
    movie_rows = np.asarray(payload.get("movie_source_row", []), dtype=np.int64)
    reconstructed_rows = trials["source_row"].to_numpy(dtype=np.int64)
    source_row_match = bool(movie_rows.size == 0 or np.array_equal(movie_rows, reconstructed_rows))

    detail_rows: list[dict[str, Any]] = []
    for movie_idx, (_, trial) in enumerate(trials.iterrows()):
        source_trace = np.asarray(trial["source_trace"], dtype=np.float32)
        axis_deg = float(trial[axis_column])
        along_u, across_u = axis_vectors(axis_deg)
        base_along_rms = projection_rms(source_trace, along_u)
        base_across_rms = projection_rms(source_trace, across_u)
        for condition_idx, spec in enumerate(specs):
            along_scale = float(spec["along_scale"])
            across_scale = float(spec["across_scale"])
            if bool(spec.get("is_static_baseline", False)):
                trace = np.zeros_like(source_trace, dtype=np.float32)
            else:
                trace, _meta = combined_axis_trace(
                    source_trace,
                    axis_deg=axis_deg,
                    along_scale=along_scale,
                    across_scale=across_scale,
                )
            actual_along_rms = projection_rms(trace, along_u)
            actual_across_rms = projection_rms(trace, across_u)
            expected_along_rms = 0.0 if bool(spec.get("is_static_baseline", False)) else abs(along_scale) * base_along_rms
            expected_across_rms = 0.0 if bool(spec.get("is_static_baseline", False)) else abs(across_scale) * base_across_rms
            expected_rms = 0.0 if bool(spec.get("is_static_baseline", False)) else expected_combined_rms(
                base_along_rms,
                base_across_rms,
                along_scale=along_scale,
                across_scale=across_scale,
            )
            detail_rows.append(
                {
                    "cache": "population_n128",
                    "movie_index": int(movie_idx),
                    "source_row": int(trial["source_row"]),
                    "trial_id": int(trial["trial_id"]),
                    "condition_index": int(condition_idx),
                    "condition_id": str(spec["condition_id"]),
                    "is_static_baseline": bool(spec.get("is_static_baseline", False)),
                    "along_scale": along_scale,
                    "across_scale": across_scale,
                    "source_trace_rms_deg": trace_rms(source_trace),
                    "source_trace_rms_px": trace_rms(source_trace) * float(PPD),
                    "base_along_component_rms_deg": float(base_along_rms),
                    "base_across_component_rms_deg": float(base_across_rms),
                    "expected_along_component_rms_deg": float(expected_along_rms),
                    "actual_along_component_rms_deg": float(actual_along_rms),
                    "expected_across_component_rms_deg": float(expected_across_rms),
                    "actual_across_component_rms_deg": float(actual_across_rms),
                    "expected_output_rms_deg": float(expected_rms),
                    "actual_output_rms_deg": trace_rms(trace),
                    "actual_output_rms_px": trace_rms(trace) * float(PPD),
                    "actual_path_length_deg": trace_path_length(trace),
                    "along_component_rms_error_deg": float(actual_along_rms - expected_along_rms),
                    "across_component_rms_error_deg": float(actual_across_rms - expected_across_rms),
                    "output_rms_error_deg": float(trace_rms(trace) - expected_rms),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for condition_idx, spec in enumerate(specs):
        rows = [row for row in detail_rows if int(row["condition_index"]) == int(condition_idx)]
        for key in ("along_component_rms_error_deg", "across_component_rms_error_deg", "output_rms_error_deg"):
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
            if key == "along_component_rms_error_deg":
                prefix = "along"
            elif key == "across_component_rms_error_deg":
                prefix = "across"
            else:
                prefix = "output"
            if not any(int(r["condition_index"]) == int(condition_idx) for r in summary_rows):
                summary_rows.append(
                    {
                        "condition_index": int(condition_idx),
                        "condition_id": str(spec["condition_id"]),
                        "is_static_baseline": bool(spec.get("is_static_baseline", False)),
                        "along_scale": float(spec["along_scale"]),
                        "across_scale": float(spec["across_scale"]),
                    }
                )
            summary_rows[-1][f"{prefix}_rms_error_abs_max_deg"] = float(np.max(np.abs(values))) if values.size else 0.0
            summary_rows[-1][f"{prefix}_rms_error_abs_max_px"] = (
                float(np.max(np.abs(values)) * float(PPD)) if values.size else 0.0
            )
        actual_px = np.asarray([float(row["actual_output_rms_px"]) for row in rows], dtype=np.float64)
        source_px = np.asarray([float(row["source_trace_rms_px"]) for row in rows], dtype=np.float64)
        summary_rows[-1].update(
            {
                "actual_output_rms_px_mean": float(np.mean(actual_px)) if actual_px.size else float("nan"),
                "actual_output_rms_px_median": float(np.median(actual_px)) if actual_px.size else float("nan"),
                "actual_output_rms_px_p95": float(np.percentile(actual_px, 95.0)) if actual_px.size else float("nan"),
                "source_trace_rms_px_mean": float(np.mean(source_px)) if source_px.size else float("nan"),
                "n_movies": int(len(rows)),
            }
        )

    max_abs_error = max(
        (
            abs(float(row["along_component_rms_error_deg"]))
            for row in detail_rows
        ),
        default=0.0,
    )
    max_abs_error = max(
        max_abs_error,
        max((abs(float(row["across_component_rms_error_deg"])) for row in detail_rows), default=0.0),
        max((abs(float(row["output_rms_error_deg"])) for row in detail_rows), default=0.0),
    )
    summary = {
        "path": path,
        "n_conditions": int(len(specs)),
        "n_movies": int(trials.shape[0]),
        "source_row_order_matches_cache": source_row_match,
        "source_trace_contract": meta.get("source_trace_contract", ""),
        "max_abs_rms_error_deg": float(max_abs_error),
        "max_abs_rms_error_px": float(max_abs_error * float(PPD)),
        "status": "PASS" if source_row_match and max_abs_error <= TRACE_TOL_DEG else "FAIL",
    }
    return detail_rows, summary_rows, summary


def current_lag_movie(stim: Any, t_max: int) -> np.ndarray:
    arr = stim.detach().cpu().numpy().astype(np.float32, copy=False)
    arr = np.squeeze(arr)
    if arr.ndim == 4:
        cubes = arr
    elif arr.ndim == 5:
        cubes = arr[:, 0]
    else:
        raise ValueError(f"Unexpected stimulus tensor shape after squeeze: {arr.shape}")
    movie = cubes[1 : int(t_max) + 1, 0]
    if movie.shape[0] != int(t_max):
        raise ValueError(f"Expected {t_max} aligned frames, got {movie.shape[0]}")
    return movie


def render_coordinate_movie(trace_xy: np.ndarray, coordinate: str) -> np.ndarray:
    import torch

    trace_xy = np.asarray(trace_xy, dtype=np.float32)
    height = width = 540
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    image = xx if coordinate == "x" else yy
    full_stack = np.broadcast_to(
        image[None, :, :],
        (trace_xy.shape[0] + N_LAGS + 1, height, width),
    ).copy()
    eye = torch.from_numpy(_trace_xy_to_twin_helper_order(trace_xy)).float()
    stim = make_counterfactual_stim(
        full_stack,
        eye,
        ppd=PPD,
        scale_factor=1.0,
        n_lags=N_LAGS,
        out_size=OUT_SIZE,
    )
    return current_lag_movie(stim, trace_xy.shape[0])


def audit_synthetic_pixel_mapping() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    test_points = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [-0.1, 0.0],
            [0.0, 0.1],
            [0.0, -0.1],
            [0.1, 0.1],
            [-0.1, -0.1],
            [1.0 / 60.0, -2.0 / 60.0],
        ],
        dtype=np.float32,
    )
    trace = np.zeros((40, 2), dtype=np.float32)
    trace[: test_points.shape[0]] = test_points
    movie_x = render_coordinate_movie(trace, "x")
    movie_y = render_coordinate_movie(trace, "y")
    center_y = int(OUT_SIZE[0]) // 2
    center_x = int(OUT_SIZE[1]) // 2
    actual_x = movie_x[:, center_y, center_x].astype(np.float64)
    actual_y = movie_y[:, center_y, center_x].astype(np.float64)
    image_center_x = (540 - 1) / 2.0
    image_center_y = (540 - 1) / 2.0
    expected_x = image_center_x - trace[:, 0].astype(np.float64) * float(PPD)
    expected_y = image_center_y + trace[:, 1].astype(np.float64) * float(PPD)
    rows: list[dict[str, Any]] = []
    for idx in range(test_points.shape[0]):
        rows.append(
            {
                "sample": int(idx),
                "trace_x_deg": float(trace[idx, 0]),
                "trace_y_deg": float(trace[idx, 1]),
                "expected_source_x_px": float(expected_x[idx]),
                "actual_source_x_px": float(actual_x[idx]),
                "x_error_px": float(actual_x[idx] - expected_x[idx]),
                "expected_source_y_px": float(expected_y[idx]),
                "actual_source_y_px": float(actual_y[idx]),
                "y_error_px": float(actual_y[idx] - expected_y[idx]),
                "expected_delta_x_px_from_zero": float(expected_x[idx] - expected_x[0]),
                "actual_delta_x_px_from_zero": float(actual_x[idx] - actual_x[0]),
                "expected_delta_y_px_from_zero": float(expected_y[idx] - expected_y[0]),
                "actual_delta_y_px_from_zero": float(actual_y[idx] - actual_y[0]),
            }
        )
    max_error = max(
        max(abs(float(row["x_error_px"])) for row in rows),
        max(abs(float(row["y_error_px"])) for row in rows),
    )
    summary = {
        "ppd": float(PPD),
        "out_size": [int(v) for v in OUT_SIZE],
        "formula": "source_x_px = image_center_x - trace_x_deg * PPD; source_y_px = image_center_y + trace_y_deg * PPD",
        "max_abs_pixel_error": float(max_error),
        "status": "PASS" if max_error <= PIXEL_TOL else "FAIL",
    }
    return rows, summary


def plot_audit(
    out_dir: Path,
    latest_rows: list[dict[str, Any]],
    population_summary_rows: list[dict[str, Any]],
    synthetic_rows: list[dict[str, Any]],
    *,
    dpi: int,
) -> tuple[Path, Path]:
    colors = {
        "blue": "#0072B2",
        "orange": "#E69F00",
        "purple": "#CC79A7",
        "gray": "#666666",
    }
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)

    ax = axs[0]
    expected_x = np.asarray([float(row["expected_delta_x_px_from_zero"]) for row in synthetic_rows])
    actual_x = np.asarray([float(row["actual_delta_x_px_from_zero"]) for row in synthetic_rows])
    expected_y = np.asarray([float(row["expected_delta_y_px_from_zero"]) for row in synthetic_rows])
    actual_y = np.asarray([float(row["actual_delta_y_px_from_zero"]) for row in synthetic_rows])
    lim = float(max(np.max(np.abs(expected_x)), np.max(np.abs(actual_x)), np.max(np.abs(expected_y)), np.max(np.abs(actual_y)), 1.0))
    ax.plot([-lim, lim], [-lim, lim], color="0.65", linewidth=1.0, linestyle=":")
    ax.scatter(expected_x, actual_x, color=colors["blue"], label="x offset", s=38)
    ax.scatter(expected_y, actual_y, color=colors["orange"], label="y offset", s=38, marker="s")
    ax.set_xlim(-lim * 1.08, lim * 1.08)
    ax.set_ylim(-lim * 1.08, lim * 1.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("synthetic movie mapping")
    ax.set_xlabel("expected pixel delta")
    ax.set_ylabel("actual pixel delta")
    ax.grid(True, color="0.9", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)

    ax = axs[1]
    x = np.arange(len(latest_rows))
    labels = [str(row["condition_id"]).replace("along", "a").replace("_across", "/c") for row in latest_rows]
    expected = np.asarray([float(row["expected_output_rms_deg"]) * float(PPD) for row in latest_rows])
    actual = np.asarray([float(row["actual_output_rms_px"]) for row in latest_rows])
    ax.plot(x, expected, color=colors["gray"], linestyle=":", linewidth=1.2, label="expected")
    ax.scatter(x, actual, color=colors["blue"], s=28, label="cached")
    ax.set_title("latest unit-map cache traces")
    ax.set_ylabel("output trace RMS (px)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.grid(True, axis="y", color="0.9", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)

    ax = axs[2]
    labels = [str(row["condition_id"]).replace("static_", "static\n").replace("along", "a").replace("_across", "/c") for row in population_summary_rows]
    x = np.arange(len(population_summary_rows))
    med = np.asarray([float(row["actual_output_rms_px_median"]) for row in population_summary_rows])
    p95 = np.asarray([float(row["actual_output_rms_px_p95"]) for row in population_summary_rows])
    ax.plot(x, med, color=colors["purple"], linewidth=1.8, marker="o", label="median")
    ax.plot(x, p95, color=colors["orange"], linewidth=1.4, marker="s", linestyle="--", label="p95")
    ax.set_title("n=128 population trace amplitudes")
    ax.set_ylabel("output trace RMS (px)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.grid(True, axis="y", color="0.9", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("BackImage RR100 displacement and scale audit", fontsize=13)
    png = out_dir / "backimage_rr100_displacement_scaling_audit.png"
    pdf = out_dir / "backimage_rr100_displacement_scaling_audit.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_rows, latest_summary = audit_unit_map_cache(Path(args.unit_map_cache))
    population_detail_rows, population_summary_rows, population_summary = audit_population_cache(
        args,
        Path(args.population_cache),
    )
    synthetic_rows, synthetic_summary = audit_synthetic_pixel_mapping()
    png, pdf = plot_audit(
        out_dir,
        latest_rows,
        population_summary_rows,
        synthetic_rows,
        dpi=int(args.dpi),
    )

    write_csv_rows(out_dir / "latest_unit_map_condition_trace_audit.csv", latest_rows)
    write_csv_rows(out_dir / "population_condition_trace_audit.csv", population_detail_rows)
    write_csv_rows(out_dir / "population_condition_trace_summary.csv", population_summary_rows)
    write_csv_rows(out_dir / "synthetic_pixel_mapping_audit.csv", synthetic_rows)
    summary = {
        "unit_map_cache": latest_summary,
        "population_cache": population_summary,
        "synthetic_pixel_mapping": synthetic_summary,
        "figure_png": png,
        "figure_pdf": pdf,
        "overall_status": (
            "PASS"
            if latest_summary["status"] == "PASS"
            and population_summary["status"] == "PASS"
            and synthetic_summary["status"] == "PASS"
            else "FAIL"
        ),
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
