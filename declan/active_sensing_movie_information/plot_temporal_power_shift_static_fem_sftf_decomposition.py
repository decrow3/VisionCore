#!/usr/bin/env python3
"""Show static and FEM-driven spatiotemporal power for one real image/trace.

This checkpoint intentionally keeps the decomposition concrete:

    full FEM movie = temporal mean image + frame-by-frame residual

The temporal mean image is the TF=0 component. The residual contains the
positive temporal-frequency power created by eye movements.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import shift as nd_shift

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import (
    DEFAULT_RUN_DIR,
    one_trace_from_source,
    source_row_by_id,
)
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import (
    DEFAULT_SOURCE_CSV,
    load_source_rows,
)
from declan.active_sensing_movie_information.temporal_remapping import MODEL_RATE_HZ
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch


DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_11_static_fem_sftf_decomposition_v1"
OKABE_BLUE = "#0072B2"
OKABE_ORANGE = "#E69F00"
OKABE_GREEN = "#009E73"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--image-index", type=int, default=9)
    parser.add_argument("--trace-index", type=int, default=31)
    parser.add_argument("--n-timepoints", type=int, default=32)
    parser.add_argument("--frame-rate-hz", type=float, default=MODEL_RATE_HZ)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--spectrum-size-px", type=int, default=256)
    parser.add_argument("--n-spatial-bins", type=int, default=38)
    parser.add_argument("--db-floor", type=float, default=-45.0)
    parser.add_argument("--dpi", type=int, default=180)
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


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def center_crop(arr: np.ndarray, size_px: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D patch, got shape {arr.shape}.")
    size = int(size_px)
    if size > arr.shape[0] or size > arr.shape[1]:
        raise ValueError(f"Crop size {size} exceeds patch shape {arr.shape}.")
    y0 = (arr.shape[0] - size) // 2
    x0 = (arr.shape[1] - size) // 2
    return np.asarray(arr[y0 : y0 + size, x0 : x0 + size], dtype=np.float64)


def normalize_contrast(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return (arr - float(np.nanmean(arr))) / (float(np.nanstd(arr)) + 1.0e-8)


def build_shifted_movie(patch: np.ndarray, trace_deg: np.ndarray, ppd: float) -> tuple[np.ndarray, np.ndarray]:
    trace = np.asarray(trace_deg, dtype=np.float64)
    centered = trace - np.nanmean(trace, axis=0, keepdims=True)
    # The sign convention only changes movie phase, not the power magnitude.
    shifts_px = np.column_stack([centered[:, 1] * float(ppd), -centered[:, 0] * float(ppd)])
    frames = []
    for dy_px, dx_px in shifts_px:
        frames.append(
            nd_shift(
                np.asarray(patch, dtype=np.float64),
                shift=(float(dy_px), float(dx_px)),
                order=1,
                mode="nearest",
                prefilter=False,
            )
        )
    return np.stack(frames, axis=0), shifts_px


def abs_temporal_groups(n_time: int, frame_rate_hz: float) -> tuple[np.ndarray, list[np.ndarray]]:
    freqs = np.fft.fftfreq(int(n_time), d=1.0 / float(frame_rate_hz))
    rounded = np.unique(np.round(np.abs(freqs), 10))
    rounded = np.sort(rounded)
    groups = [np.where(np.isclose(np.round(np.abs(freqs), 10), freq))[0] for freq in rounded]
    return rounded.astype(np.float64), groups


def center_edges(centers: np.ndarray, *, first_edge: float | None = None) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 0:
        raise ValueError("Cannot build edges from empty centers.")
    if centers.size == 1:
        width = max(abs(float(centers[0])), 1.0)
        lo = max(0.0, float(centers[0]) - width / 2.0) if first_edge is None else float(first_edge)
        return np.asarray([lo, float(centers[0]) + width / 2.0], dtype=np.float64)
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = float(first_edge) if first_edge is not None else max(0.0, float(centers[0]) - (float(mids[0]) - float(centers[0])))
    last = float(centers[-1]) + (float(centers[-1]) - float(mids[-1]))
    return np.concatenate([[first], mids, [last]]).astype(np.float64)


def spatial_frequency_grid(size_px: int, ppd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq = np.fft.fftfreq(int(size_px), d=1.0 / float(ppd))
    fy, fx = np.meshgrid(freq, freq, indexing="ij")
    rr = np.sqrt(fx * fx + fy * fy)
    return fx, fy, rr


def spatial_edges(rr: np.ndarray, size_px: int, ppd: float, n_bins: int) -> np.ndarray:
    min_positive = 1.0 / (float(size_px) / float(ppd))
    max_freq = float(np.nanmax(rr))
    low = max(0.25, min_positive)
    high = min(max_freq, float(ppd) / 2.0)
    return np.geomspace(low, high, int(n_bins) + 1).astype(np.float64)


def sftf_power(
    movie: np.ndarray,
    *,
    ppd: float,
    frame_rate_hz: float,
    n_spatial_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(movie, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected movie with shape (time,y,x), got {arr.shape}.")
    n_time, height, width = arr.shape
    if height != width:
        raise ValueError(f"Expected square frames, got {arr.shape}.")
    spatial_window = np.outer(np.hanning(height), np.hanning(width))
    weighted = arr * spatial_window[None, :, :]
    spec = np.fft.fftn(weighted, axes=(0, 1, 2), norm="ortho")
    power = np.abs(spec) ** 2
    _, _, rr = spatial_frequency_grid(height, float(ppd))
    sf_edges = spatial_edges(rr, height, float(ppd), int(n_spatial_bins))
    sf_centers = np.sqrt(sf_edges[:-1] * sf_edges[1:])
    tf_centers, tf_groups = abs_temporal_groups(n_time, float(frame_rate_hz))
    out = np.full((tf_centers.size, sf_centers.size), np.nan, dtype=np.float64)
    for tf_i, indices in enumerate(tf_groups):
        tf_power = np.sum(power[indices, :, :], axis=0)
        for sf_i, (lo, hi) in enumerate(zip(sf_edges[:-1], sf_edges[1:], strict=True)):
            mask = (rr >= lo) & (rr < hi)
            if np.any(mask):
                out[tf_i, sf_i] = float(np.nanmean(tf_power[mask]))
    return sf_centers, sf_edges, tf_centers, out


def db_relative(power: np.ndarray, reference: float, floor_db: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(np.maximum(np.asarray(power, dtype=np.float64), 1.0e-300) / float(reference))
    out = np.maximum(out, float(floor_db))
    return np.nan_to_num(out, nan=float(floor_db), posinf=0.0, neginf=float(floor_db))


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        0.0,
        1.08,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )


def add_image(ax: plt.Axes, image: np.ndarray, title: str, *, cmap: str = "gray", diverging: bool = False) -> None:
    arr = np.asarray(image, dtype=np.float64)
    if diverging:
        vmax = float(np.nanpercentile(np.abs(arr), 99.0))
        ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        lo, hi = np.nanpercentile(arr, [1.0, 99.0])
        ax.imshow(arr, cmap=cmap, vmin=float(lo), vmax=float(hi))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def save_power_table(
    out_path: Path,
    *,
    sf_centers: np.ndarray,
    tf_centers: np.ndarray,
    components: dict[str, np.ndarray],
    reference_power: float,
    db_floor: float,
) -> None:
    rows: list[dict[str, Any]] = []
    for component, power in components.items():
        power_db = db_relative(power, reference_power, float(db_floor))
        for tf_i, tf_hz in enumerate(tf_centers):
            for sf_i, sf_cpd in enumerate(sf_centers):
                rows.append(
                    {
                        "component": component,
                        "spatial_frequency_cpd": float(sf_cpd),
                        "temporal_frequency_hz": float(tf_hz),
                        "power": float(power[tf_i, sf_i]),
                        "power_db_relative_to_static_peak": float(power_db[tf_i, sf_i]),
                    }
                )
    write_csv(out_path, rows)


def save_curve_table(
    out_path: Path,
    *,
    sf_centers: np.ndarray,
    tf_centers: np.ndarray,
    static_power: np.ndarray,
    fem_dynamic_power: np.ndarray,
    reference_power: float,
    db_floor: float,
) -> None:
    rows: list[dict[str, Any]] = []
    static_sf = static_power[0, :]
    dynamic_sf = np.nansum(fem_dynamic_power[tf_centers > 0.0, :], axis=0)
    static_tf = np.nansum(static_power, axis=1)
    dynamic_tf = np.nansum(fem_dynamic_power, axis=1)
    for sf, static_value, dynamic_value in zip(sf_centers, static_sf, dynamic_sf, strict=True):
        rows.append(
            {
                "curve": "spatial",
                "x_value": float(sf),
                "x_unit": "cycles/degree",
                "static_tf0_power_db": float(db_relative(np.asarray([static_value]), reference_power, db_floor)[0]),
                "fem_dynamic_tfpos_power_db": float(db_relative(np.asarray([dynamic_value]), reference_power, db_floor)[0]),
            }
        )
    for tf, static_value, dynamic_value in zip(tf_centers, static_tf, dynamic_tf, strict=True):
        rows.append(
            {
                "curve": "temporal",
                "x_value": float(tf),
                "x_unit": "Hz",
                "static_tf0_power_db": float(db_relative(np.asarray([static_value]), reference_power, db_floor)[0]),
                "fem_dynamic_tfpos_power_db": float(db_relative(np.asarray([dynamic_value]), reference_power, db_floor)[0]),
            }
        )
    write_csv(out_path, rows)


def plot_decomposition(
    *,
    out_dir: Path,
    patch: np.ndarray,
    trace: np.ndarray,
    ppd: float,
    image_index: int,
    image_source_row: int,
    trace_index: int,
    trace_source_row: int,
    frame_rate_hz: float,
    n_spatial_bins: int,
    db_floor: float,
    dpi: int,
) -> tuple[Path, Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    static_movie = np.repeat(patch[None, :, :], trace.shape[0], axis=0)
    fem_movie, shifts_px = build_shifted_movie(patch, trace, float(ppd))
    fem_mean = np.mean(fem_movie, axis=0)
    fem_dynamic = fem_movie - fem_mean[None, :, :]

    sf_centers, sf_edges, tf_centers, static_power = sftf_power(
        static_movie,
        ppd=float(ppd),
        frame_rate_hz=float(frame_rate_hz),
        n_spatial_bins=int(n_spatial_bins),
    )
    _, _, _, fem_mean_power = sftf_power(
        np.repeat(fem_mean[None, :, :], trace.shape[0], axis=0),
        ppd=float(ppd),
        frame_rate_hz=float(frame_rate_hz),
        n_spatial_bins=int(n_spatial_bins),
    )
    _, _, _, fem_dynamic_power = sftf_power(
        fem_dynamic,
        ppd=float(ppd),
        frame_rate_hz=float(frame_rate_hz),
        n_spatial_bins=int(n_spatial_bins),
    )
    reference_power = float(np.nanmax(static_power))
    power_components = {
        "stabilized_static_tf0": static_power,
        "fem_temporal_mean_tf0": fem_mean_power,
        "fem_dynamic_residual_tfpos": fem_dynamic_power,
    }

    power_csv = out_dir / "checkpoint_11_static_fem_sftf_power_table.csv"
    curve_csv = out_dir / "checkpoint_11_static_fem_sftf_curve_table.csv"
    save_power_table(
        power_csv,
        sf_centers=sf_centers,
        tf_centers=tf_centers,
        components=power_components,
        reference_power=reference_power,
        db_floor=float(db_floor),
    )
    save_curve_table(
        curve_csv,
        sf_centers=sf_centers,
        tf_centers=tf_centers,
        static_power=static_power,
        fem_dynamic_power=fem_dynamic_power,
        reference_power=reference_power,
        db_floor=float(db_floor),
    )

    centered_trace = trace - np.nanmean(trace, axis=0, keepdims=True)
    time_ms = np.arange(trace.shape[0], dtype=float) * 1000.0 / float(frame_rate_hz)
    speed_deg_s = np.zeros((trace.shape[0],), dtype=np.float64)
    if trace.shape[0] > 1:
        speed_deg_s[1:] = np.linalg.norm(np.diff(trace, axis=0), axis=1) * float(frame_rate_hz)
    temporal_edges = center_edges(tf_centers, first_edge=0.0)
    vmax_resid = float(np.nanpercentile(np.abs(fem_dynamic), 99.5))
    show_frames = [0, min(8, trace.shape[0] - 1), min(16, trace.shape[0] - 1), trace.shape[0] - 1]

    fig = plt.figure(figsize=(15.8, 12.2), constrained_layout=False)
    gs = fig.add_gridspec(
        4,
        12,
        height_ratios=[0.92, 1.1, 1.2, 1.05],
        left=0.055,
        right=0.985,
        top=0.92,
        bottom=0.07,
        hspace=0.58,
        wspace=0.48,
    )

    ax_patch = fig.add_subplot(gs[0, 0:3])
    add_image(ax_patch, patch, "Image patch")
    panel_letter(ax_patch, "A")
    xy_px = np.column_stack(
        [
            patch.shape[1] / 2.0 + centered_trace[:, 0] * float(ppd),
            patch.shape[0] / 2.0 - centered_trace[:, 1] * float(ppd),
        ]
    )
    ax_patch.plot(xy_px[:, 0], xy_px[:, 1], color=OKABE_ORANGE, lw=1.6)
    ax_patch.scatter(xy_px[0, 0], xy_px[0, 1], s=32, color=OKABE_BLUE, edgecolor="white", linewidth=0.7)
    ax_patch.scatter(xy_px[-1, 0], xy_px[-1, 1], s=32, color=OKABE_ORANGE, edgecolor="white", linewidth=0.7)
    ax_patch.text(
        0.02,
        0.04,
        "orange path = eye-position samples",
        transform=ax_patch.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="0.15",
        bbox={"facecolor": "white", "edgecolor": "0.82", "pad": 3.0, "alpha": 0.9},
    )

    ax_trace = fig.add_subplot(gs[0, 3:6])
    ax_trace.plot(time_ms, centered_trace[:, 0] * 60.0, color=OKABE_BLUE, lw=1.5, label="horizontal")
    ax_trace.plot(time_ms, centered_trace[:, 1] * 60.0, color=OKABE_GREEN, lw=1.5, label="vertical")
    ax_trace.set_title("Eye movement over 32 frames", fontsize=10)
    ax_trace.set_xlabel("time (ms)")
    ax_trace.set_ylabel("position (arcmin)")
    ax_trace.grid(True, color="#e8e8e8", lw=0.7)
    ax_trace.legend(frameon=False, fontsize=8.5, loc="best")

    ax_speed = fig.add_subplot(gs[0, 6:9])
    ax_speed.plot(time_ms, speed_deg_s, color=OKABE_ORANGE, lw=1.8)
    ax_speed.set_title("Retinal image speed", fontsize=10)
    ax_speed.set_xlabel("time (ms)")
    ax_speed.set_ylabel("speed (deg/s)")
    ax_speed.grid(True, color="#e8e8e8", lw=0.7)

    ax_rule = fig.add_subplot(gs[0, 9:12])
    ax_rule.axis("off")
    ax_rule.text(
        0.0,
        0.94,
        "Power split used here",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    ax_rule.text(
        0.0,
        0.78,
        "Stimulus power only\n(not model response)\n\nStatic repeated frame -> TF=0\nFEM mean image -> TF=0\nFEM residual -> TF>0",
        ha="left",
        va="top",
        fontsize=9.4,
        linespacing=1.35,
    )

    ax_static = fig.add_subplot(gs[1, 0:2])
    add_image(ax_static, patch, "Static frame")
    panel_letter(ax_static, "B")
    ax_equal = fig.add_subplot(gs[1, 2])
    ax_equal.axis("off")
    ax_equal.text(0.5, 0.5, "vs", ha="center", va="center", fontsize=16, color="0.25")
    for col_i, frame_i in enumerate(show_frames):
        ax_frame = fig.add_subplot(gs[1, 3 + col_i : 4 + col_i])
        add_image(ax_frame, fem_movie[frame_i], f"FEM frame {frame_i}")
    ax_mean = fig.add_subplot(gs[1, 8:10])
    add_image(ax_mean, fem_mean, "FEM mean\nTF=0")
    ax_resid = fig.add_subplot(gs[1, 10:12])
    add_image(ax_resid, fem_dynamic[show_frames[-1]], "FEM residual\nTF>0", cmap="coolwarm", diverging=True)

    static_db = db_relative(static_power, reference_power, float(db_floor))
    fem_mean_db = db_relative(fem_mean_power, reference_power, float(db_floor))
    dynamic_db = db_relative(fem_dynamic_power, reference_power, float(db_floor))
    heatmap_kwargs = {
        "shading": "auto",
        "cmap": "magma",
        "vmin": float(db_floor),
        "vmax": 0.0,
    }
    ax_h_static = fig.add_subplot(gs[2, 0:4])
    mesh_static = ax_h_static.pcolormesh(sf_edges, temporal_edges, static_db, **heatmap_kwargs)
    panel_letter(ax_h_static, "C")
    ax_h_static.set_title("Static power sits at temporal frequency 0", fontsize=11)
    ax_h_static.set_xscale("log")
    ax_h_static.set_xlabel("spatial frequency (cycles/degree)")
    ax_h_static.set_ylabel("temporal frequency (Hz)")
    ax_h_static.set_yticks([0.0, 7.5, 15.0, 30.0, 60.0])
    ax_h_static.set_ylim(0.0, float(temporal_edges[-1]))
    ax_h_static.grid(False)

    ax_h_mean = fig.add_subplot(gs[2, 4:8])
    ax_h_mean.pcolormesh(sf_edges, temporal_edges, fem_mean_db, **heatmap_kwargs)
    panel_letter(ax_h_mean, "D")
    ax_h_mean.set_title("FEM mean image is also TF=0", fontsize=11)
    ax_h_mean.set_xscale("log")
    ax_h_mean.set_xlabel("spatial frequency (cycles/degree)")
    ax_h_mean.set_yticks([0.0, 7.5, 15.0, 30.0, 60.0])
    ax_h_mean.set_yticklabels([])
    ax_h_mean.set_ylim(0.0, float(temporal_edges[-1]))

    ax_h_dyn = fig.add_subplot(gs[2, 8:12])
    ax_h_dyn.pcolormesh(sf_edges, temporal_edges, dynamic_db, **heatmap_kwargs)
    panel_letter(ax_h_dyn, "E")
    ax_h_dyn.set_title("FEM residual creates TF>0 power", fontsize=11)
    ax_h_dyn.set_xscale("log")
    ax_h_dyn.set_xlabel("spatial frequency (cycles/degree)")
    ax_h_dyn.set_yticks([0.0, 7.5, 15.0, 30.0, 60.0])
    ax_h_dyn.set_yticklabels([])
    ax_h_dyn.set_ylim(0.0, float(temporal_edges[-1]))
    cbar = fig.colorbar(mesh_static, ax=[ax_h_static, ax_h_mean, ax_h_dyn], fraction=0.018, pad=0.012)
    cbar.set_label("power (dB relative to static peak)")

    ax_spatial = fig.add_subplot(gs[3, 0:6])
    static_sf_power = static_power[0, :]
    dynamic_sf_power = np.nansum(fem_dynamic_power[tf_centers > 0.0, :], axis=0)
    valid_sf_curve = np.isfinite(static_sf_power) & np.isfinite(dynamic_sf_power)
    static_sf_db = db_relative(static_sf_power, reference_power, float(db_floor))
    dynamic_sf_db = db_relative(dynamic_sf_power, reference_power, float(db_floor))
    panel_letter(ax_spatial, "F")
    ax_spatial.plot(sf_centers[valid_sf_curve], static_sf_db[valid_sf_curve], color="black", lw=2.0, label="static, TF=0")
    ax_spatial.plot(
        sf_centers[valid_sf_curve],
        dynamic_sf_db[valid_sf_curve],
        color=OKABE_ORANGE,
        lw=2.0,
        label="FEM-driven, TF>0",
    )
    ax_spatial.set_xscale("log")
    ax_spatial.set_ylim(float(db_floor), 1.0)
    ax_spatial.set_title("Same spatial image power, moved into time", fontsize=11)
    ax_spatial.set_xlabel("spatial frequency (cycles/degree)")
    ax_spatial.set_ylabel("power (dB relative to static peak)")
    ax_spatial.grid(True, color="#e8e8e8", lw=0.7, which="both")
    ax_spatial.legend(frameon=False, fontsize=9)

    ax_temporal = fig.add_subplot(gs[3, 6:12])
    static_tf_power = np.nansum(static_power, axis=1)
    dynamic_tf_power = np.nansum(fem_dynamic_power, axis=1)
    temporal_curve_reference = max(float(static_tf_power[0]), 1.0e-300)
    static_tf_db = db_relative(static_tf_power, temporal_curve_reference, float(db_floor))
    dynamic_tf_db = db_relative(dynamic_tf_power, temporal_curve_reference, float(db_floor))
    panel_letter(ax_temporal, "G")
    ax_temporal.scatter(tf_centers, static_tf_db, color="black", s=36, label="static")
    ax_temporal.plot(tf_centers, dynamic_tf_db, color=OKABE_ORANGE, lw=2.0, marker="o", ms=4, label="FEM-driven")
    ax_temporal.axvline(0.0, color="0.35", lw=1.0, linestyle=":")
    ax_temporal.set_ylim(float(db_floor), 1.0)
    ax_temporal.set_title("Static is TF=0; FEM spreads power across TF", fontsize=11)
    ax_temporal.set_xlabel("temporal frequency (Hz)")
    ax_temporal.set_ylabel("power (dB relative to static TF=0 total)")
    ax_temporal.grid(True, color="#e8e8e8", lw=0.7)
    ax_temporal.legend(frameon=False, fontsize=9)

    for ax in [ax_h_static, ax_h_mean, ax_h_dyn, ax_spatial]:
        ax.set_xticks([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
        ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16"])

    fig.suptitle(
        "Static image power and eye-movement-driven temporal power",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.952,
        f"image {image_index} / source row {image_source_row}; trace {trace_index} / source row {trace_source_row}; {trace.shape[0]} frames at {frame_rate_hz:.0f} Hz",
        ha="left",
        va="top",
        fontsize=10,
        color="0.30",
    )

    png = out_dir / "checkpoint_11_static_fem_sftf_decomposition.png"
    pdf = out_dir / "checkpoint_11_static_fem_sftf_decomposition.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    trace_csv = out_dir / "checkpoint_11_trace_frame_values.csv"
    trace_rows = []
    for frame_i, row in enumerate(centered_trace):
        trace_rows.append(
            {
                "frame_index": int(frame_i),
                "time_ms": float(time_ms[frame_i]),
                "horizontal_position_arcmin": float(row[0] * 60.0),
                "vertical_position_arcmin": float(row[1] * 60.0),
                "speed_deg_s": float(speed_deg_s[frame_i]),
                "shift_y_px": float(shifts_px[frame_i, 0]),
                "shift_x_px": float(shifts_px[frame_i, 1]),
            }
        )
    write_csv(trace_csv, trace_rows)

    write_json(
        out_dir / "checkpoint_11_metadata.json",
        {
            "analysis": "static_fem_sftf_decomposition",
            "image_index": int(image_index),
            "image_source_row": int(image_source_row),
            "trace_index": int(trace_index),
            "trace_source_row": int(trace_source_row),
            "frame_rate_hz": float(frame_rate_hz),
            "n_timepoints": int(trace.shape[0]),
            "patch_shape": list(map(int, patch.shape)),
            "patch_ppd": float(ppd),
            "db_floor": float(db_floor),
            "static_interpretation": "A repeated static frame has temporal power at TF=0. Positive TF bins are zero apart from numerical/windowing noise.",
            "fem_interpretation": "The FEM movie is split into temporal mean image (TF=0) and residual movie (TF>0 power created by frame-to-frame shifts).",
            "sign_note": "The image-shift sign convention changes phase but not the power magnitude summarized here.",
            "outputs": {
                "figure_png": png,
                "figure_pdf": pdf,
                "power_table_csv": power_csv,
                "curve_table_csv": curve_csv,
                "trace_frame_values_csv": trace_csv,
            },
        },
    )
    return png, pdf, power_csv, curve_csv


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    image_table = pd.read_csv(run_dir / "image_feature_table.csv")
    trace_table = pd.read_csv(run_dir / "trace_feature_table.csv")
    image_matches = image_table[pd.to_numeric(image_table["image_index"], errors="coerce").astype("Int64") == int(args.image_index)]
    if image_matches.empty:
        raise ValueError(f"image_index={args.image_index} was not found in {run_dir / 'image_feature_table.csv'}")
    trace_matches = trace_table[pd.to_numeric(trace_table["trace_index"], errors="coerce").astype("Int64") == int(args.trace_index)]
    if trace_matches.empty:
        raise ValueError(f"trace_index={args.trace_index} was not found in {run_dir / 'trace_feature_table.csv'}")
    image_row = image_matches.iloc[0]
    trace_row = trace_matches.iloc[0]
    source_rows = load_source_rows(Path(args.source_csv))
    patch_row = source_row_by_id(source_rows, int(image_row["source_row"]))
    patch, patch_meta = _extract_patch(
        patch_row,
        canvas_cache={},
        patch_size_px=int(args.patch_size_px),
    )
    crop = normalize_contrast(center_crop(np.asarray(patch), int(args.spectrum_size_px)))
    trace = one_trace_from_source(
        source_rows,
        int(trace_row["trace_source_row"]),
        n_timepoints=int(args.n_timepoints),
        bin_seconds=1.0 / float(args.frame_rate_hz),
    )
    png, pdf, power_csv, curve_csv = plot_decomposition(
        out_dir=out_dir,
        patch=crop,
        trace=trace,
        ppd=float(patch_meta["patch_ppd"]),
        image_index=int(image_row["image_index"]),
        image_source_row=int(image_row["source_row"]),
        trace_index=int(trace_row["trace_index"]),
        trace_source_row=int(trace_row["trace_source_row"]),
        frame_rate_hz=float(args.frame_rate_hz),
        n_spatial_bins=int(args.n_spatial_bins),
        db_floor=float(args.db_floor),
        dpi=int(args.dpi),
    )
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {power_csv}")
    print(f"Wrote {curve_csv}")


if __name__ == "__main__":
    main()
