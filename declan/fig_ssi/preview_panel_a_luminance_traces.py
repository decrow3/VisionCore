#!/usr/bin/env python3
"""Quick preview for Panel A grating/luminance trace design."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

try:
    import pandas as pd
except Exception:
    pd = None


ROOT = Path(__file__).resolve().parents[2]
OUT_BASE = ROOT / "outputs" / "fig_ssi" / "panel_a_luminance_trace_preview"
REAL_TRACE_BANK_DIR = (
    ROOT
    / "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
)
TRACE_XY_NPY = REAL_TRACE_BANK_DIR / "trace_xy.npy"
TRACE_COMPONENT_METRICS_CSV = (
    REAL_TRACE_BANK_DIR
    / "phase1_phase2_conditioning_v1/trace_component_conditioning_v1/"
    "phase2_contour_relative_trace_component_movie_metrics.csv"
)
REAL_TRACE_IMAGE_INDEX = 86
SMALL_REAL_TRACE_INDEX = 127
LARGE_REAL_TRACE_INDEX = 833
SMALL_REAL_TRACE_ROTATION_DEG = 0.0
LARGE_REAL_TRACE_ROTATION_DEG = 90.0

RED = "#c51f27"
BLUE = "#1e4ed8"
CYAN = "#00bcd4"
GRAY = "#5f6368"
INK = "#111111"
EPS = 1e-12

N_FRAMES = 180
FRAME_RATE_HZ = 120.0
INTEGRATION_FRAMES = 3
MODEL_PPD = 37.50476617
FEM_RANDOM_SEED = 11
FEM_BANK_MEDIAN_STEP_ARCMIN_FALLBACK = 2.7506821400214396
SMALL_STEP_SCALE = 0.5
LARGE_STEP_SCALE = 2.0
FEM_RANDOM_DAMPING = 0.0
FEM_RANDOM_PULL = 1.1
GRATING_ANGLE_DEG = -9.0
HIGH_SF_CPD = 8.0
LOW_SF_CPD = 2.0


def arcmin_to_px(value_arcmin: float) -> float:
    return float(value_arcmin) / 60.0 * MODEL_PPD


def sf_cpd_to_wavelength_px(sf_cpd: float) -> float:
    return MODEL_PPD / float(sf_cpd)


def bank_median_2d_step_arcmin() -> float:
    if not TRACE_XY_NPY.exists():
        return FEM_BANK_MEDIAN_STEP_ARCMIN_FALLBACK
    try:
        trace_xy = np.load(TRACE_XY_NPY, mmap_mode="r")
    except Exception:
        return FEM_BANK_MEDIAN_STEP_ARCMIN_FALLBACK

    trace_indices = None
    if TRACE_COMPONENT_METRICS_CSV.exists() and pd is not None:
        try:
            metrics = pd.read_csv(
                TRACE_COMPONENT_METRICS_CSV,
                usecols=["trace_index", "has_microsaccade"],
            )
            drift_only = metrics[~metrics["has_microsaccade"].astype(bool)]
            trace_indices = drift_only["trace_index"].drop_duplicates().to_numpy(dtype=np.int64)
            trace_indices = trace_indices[(trace_indices >= 0) & (trace_indices < trace_xy.shape[0])]
        except Exception:
            trace_indices = None

    try:
        if trace_indices is not None and trace_indices.size:
            traces = np.asarray(trace_xy[trace_indices], dtype=np.float64)
        else:
            traces = np.asarray(trace_xy, dtype=np.float64)
    except Exception:
        return FEM_BANK_MEDIAN_STEP_ARCMIN_FALLBACK

    steps_arcmin = np.linalg.norm(np.diff(traces, axis=1), axis=2) * 60.0
    median_step = float(np.nanmedian(steps_arcmin))
    if not np.isfinite(median_step) or median_step <= EPS:
        return FEM_BANK_MEDIAN_STEP_ARCMIN_FALLBACK
    return median_step


def grating_normal(angle_deg: float = GRATING_ANGLE_DEG) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    return np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.float64)


def trace_to_image_px(trace_xy_deg: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace_xy_deg, dtype=np.float64)
    centered = trace - np.nanmean(trace, axis=0, keepdims=True)
    # Retinal-image displacement convention used by the model-input rendering.
    return np.column_stack([-centered[:, 1], centered[:, 0]]) * MODEL_PPD


def rotate_trace_xy_px(trace_xy_px: np.ndarray, angle_deg: float) -> np.ndarray:
    trace = np.asarray(trace_xy_px, dtype=np.float64)
    theta = np.deg2rad(float(angle_deg))
    rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float64,
    )
    return trace @ rotation.T


def trace_bank_metrics(trace_index: int) -> dict[str, float]:
    if pd is None or not TRACE_COMPONENT_METRICS_CSV.exists():
        return {}
    try:
        metrics = pd.read_csv(
            TRACE_COMPONENT_METRICS_CSV,
            usecols=[
                "image_index",
                "trace_index",
                "has_microsaccade",
                "rendered_path_length_arcmin",
                "across_path_arcmin",
                "along_path_arcmin",
            ],
        )
    except Exception:
        return {}
    rows = metrics[
        metrics["trace_index"].eq(int(trace_index))
        & metrics["image_index"].eq(int(REAL_TRACE_IMAGE_INDEX))
    ].copy()
    if rows.empty:
        rows = metrics[metrics["trace_index"].eq(int(trace_index))].copy()
    if rows.empty:
        return {}
    drift_rows = rows[~rows["has_microsaccade"].astype(bool)]
    if not drift_rows.empty:
        rows = drift_rows
    row = rows.iloc[0]
    return {
        "rendered_path_length_arcmin": float(row["rendered_path_length_arcmin"]),
        "across_path_arcmin": float(row["across_path_arcmin"]),
        "along_path_arcmin": float(row["along_path_arcmin"]),
    }


def real_trace_path_length_arcmin(trace_xy_deg: np.ndarray) -> float:
    trace = np.asarray(trace_xy_deg, dtype=np.float64)
    return float(np.nansum(np.linalg.norm(np.diff(trace, axis=0), axis=1)) * 60.0)


def scale_walk_to_median_step(walk_xy: np.ndarray, target_step_px: float) -> np.ndarray:
    walk = np.asarray(walk_xy, dtype=np.float64)
    step_norm = np.linalg.norm(np.diff(walk, axis=0), axis=1)
    median_step = float(np.nanmedian(step_norm))
    if not np.isfinite(median_step) or median_step <= EPS:
        return np.zeros_like(walk)
    return walk * (float(target_step_px) / median_step)


def fem_like_random_walk_2d(n_frames: int = N_FRAMES, seed: int = FEM_RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    position = np.zeros(2, dtype=np.float64)
    velocity = np.zeros(2, dtype=np.float64)
    walk = np.empty((int(n_frames), 2), dtype=np.float64)
    for frame in range(int(n_frames)):
        velocity = (
            FEM_RANDOM_DAMPING * velocity
            - FEM_RANDOM_PULL * position
            + rng.normal(size=2)
        )
        position = position + velocity
        walk[frame] = position

    walk -= np.mean(walk, axis=0, keepdims=True)
    return walk


def synthetic_motion_traces(n_frames: int = N_FRAMES) -> dict[str, np.ndarray]:
    t = np.linspace(0.0, 1.0, int(n_frames), dtype=np.float64)
    median_step_arcmin = bank_median_2d_step_arcmin()
    base_walk = fem_like_random_walk_2d(n_frames)
    small_xy = scale_walk_to_median_step(
        base_walk,
        arcmin_to_px(SMALL_STEP_SCALE * median_step_arcmin),
    )
    large_xy = scale_walk_to_median_step(
        base_walk,
        arcmin_to_px(LARGE_STEP_SCALE * median_step_arcmin),
    )
    normal = grating_normal()
    small_across = small_xy @ normal
    large_across = large_xy @ normal
    return {
        "t": t,
        "small_xy_px": small_xy,
        "large_xy_px": large_xy,
        "small_px": small_across,
        "large_px": large_across,
        "bank_median_step_arcmin": median_step_arcmin,
    }


def selected_real_motion_traces() -> dict[str, np.ndarray]:
    if not TRACE_XY_NPY.exists():
        return synthetic_motion_traces()

    trace_xy = np.load(TRACE_XY_NPY, mmap_mode="r")
    trace_count = int(trace_xy.shape[0])
    if SMALL_REAL_TRACE_INDEX >= trace_count or LARGE_REAL_TRACE_INDEX >= trace_count:
        return synthetic_motion_traces()

    small_deg = np.asarray(trace_xy[SMALL_REAL_TRACE_INDEX], dtype=np.float64)
    large_deg = np.asarray(trace_xy[LARGE_REAL_TRACE_INDEX], dtype=np.float64)
    small_xy = rotate_trace_xy_px(trace_to_image_px(small_deg), SMALL_REAL_TRACE_ROTATION_DEG)
    large_xy = rotate_trace_xy_px(trace_to_image_px(large_deg), LARGE_REAL_TRACE_ROTATION_DEG)
    normal = grating_normal()
    small_metrics = trace_bank_metrics(SMALL_REAL_TRACE_INDEX)
    large_metrics = trace_bank_metrics(LARGE_REAL_TRACE_INDEX)
    small_path_arcmin = small_metrics.get("rendered_path_length_arcmin", real_trace_path_length_arcmin(small_deg))
    large_path_arcmin = large_metrics.get("rendered_path_length_arcmin", real_trace_path_length_arcmin(large_deg))
    return {
        "t": np.linspace(0.0, 1.0, small_xy.shape[0], dtype=np.float64),
        "small_xy_px": small_xy,
        "large_xy_px": large_xy,
        "small_px": small_xy @ normal,
        "large_px": large_xy @ normal,
        "small_trace_index": SMALL_REAL_TRACE_INDEX,
        "large_trace_index": LARGE_REAL_TRACE_INDEX,
        "small_path_arcmin": float(small_path_arcmin),
        "large_path_arcmin": float(large_path_arcmin),
    }


def grating_patch(size: int, wavelength_px: float, *, angle_deg: float = -9.0) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    cx = cy = 0.5 * (size - 1)
    x = xx - cx
    y = yy - cy
    theta = np.deg2rad(float(angle_deg))
    normal_coord = -np.sin(theta) * x + np.cos(theta) * y
    envelope = np.exp(-0.5 * ((x / (0.58 * size)) ** 2 + (y / (0.58 * size)) ** 2))
    image = 0.50 + 0.46 * np.sin(2.0 * np.pi * normal_coord / float(wavelength_px)) * envelope
    return np.clip(image, 0.0, 1.0)


def sampled_luminance_trace(motion_px: np.ndarray, wavelength_px: float) -> np.ndarray:
    motion = np.asarray(motion_px, dtype=np.float64)
    luminance = 0.50 + 0.46 * np.sin(2.0 * np.pi * motion / float(wavelength_px))
    return np.clip(luminance, 0.0, 1.0)


def temporal_integrate_trace(values: np.ndarray, n_frames: int = INTEGRATION_FRAMES) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = max(1, int(n_frames))
    if n <= 1:
        return values.copy()
    kernel = np.ones(n, dtype=np.float64) / float(n)
    padded = np.pad(values, (n - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def add_trace_path(ax, center: np.ndarray, trace_xy_px: np.ndarray, color: str, *, lw: float, zorder: float) -> None:
    trace = np.asarray(trace_xy_px, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 2 or trace.shape[0] < 2:
        return
    points = center[None, :] + trace
    ax.plot(
        points[:, 0],
        points[:, 1],
        color="white",
        lw=float(lw) + 1.7,
        alpha=0.88,
        solid_capstyle="round",
        zorder=zorder - 0.1,
    )
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        lw=float(lw),
        alpha=0.96,
        solid_capstyle="round",
        zorder=zorder,
    )
    ax.plot(points[-1, 0], points[-1, 1], marker="o", ms=3.2, mfc=color, mec="white", mew=0.7, zorder=zorder + 0.2)


def add_grating_panel(
    ax,
    *,
    title: str,
    wavelength_px: float,
    small_trace_xy_px: np.ndarray | None = None,
    large_trace_xy_px: np.ndarray | None = None,
    angle_deg: float = GRATING_ANGLE_DEG,
) -> None:
    size = 150
    image = grating_patch(size, wavelength_px, angle_deg=angle_deg)
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="bicubic")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#444")

    theta = np.deg2rad(float(angle_deg))
    tangent = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)
    center = np.asarray([0.5 * (size - 1), 0.5 * (size - 1)], dtype=np.float64)
    trace_arrays = [
        np.asarray(trace, dtype=np.float64)
        for trace in [small_trace_xy_px, large_trace_xy_px]
        if trace is not None and np.asarray(trace).ndim == 2 and np.asarray(trace).shape[1] == 2
    ]
    if trace_arrays:
        trace_extent = max(float(np.nanmax(np.abs(trace))) for trace in trace_arrays)
    else:
        trace_extent = 0.0
    view_half = min(0.48 * size, max(18.0, trace_extent + 7.0))
    line_half = 0.92 * view_half
    ax.plot(
        [center[0] - tangent[0] * line_half, center[0] + tangent[0] * line_half],
        [center[1] - tangent[1] * line_half, center[1] + tangent[1] * line_half],
        color=CYAN,
        lw=2.2,
        alpha=0.85,
        linestyle=(0, (4.2, 3.2)),
        solid_capstyle="round",
    )
    ax.plot(center[0], center[1], marker="o", ms=4.5, mfc="white", mec=INK, mew=1.0, zorder=4.0)
    add_trace_path(ax, center, large_trace_xy_px, BLUE, lw=2.2, zorder=5.0)
    add_trace_path(ax, center, small_trace_xy_px, RED, lw=2.4, zorder=6.0)
    ax.set_xlim(center[0] - view_half, center[0] + view_half)
    ax.set_ylim(center[1] + view_half, center[1] - view_half)
    ax.set_title(title, fontsize=10, pad=4)


def add_motion_trace_lane(ax, t: np.ndarray, values: np.ndarray, color: str, label: str, *, ylim: float) -> None:
    ax.axhline(0.0, color="#d6d8dc", lw=0.8, zorder=0)
    ax.plot(t, values, color=color, lw=1.9)
    ax.text(1.01, 0.50, label, transform=ax.transAxes, fontsize=8.2, color=color, ha="left", va="center", clip_on=False)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-float(ylim), float(ylim))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def add_luminance_lane(ax, t: np.ndarray, values: np.ndarray, color: str, label: str) -> None:
    ax.axhline(0.50, color="#d6d8dc", lw=0.8, zorder=0)
    ax.plot(t, values, color=color, lw=2.0)
    ax.text(-0.02, 0.50, label, transform=ax.transAxes, fontsize=8.2, color=color, ha="right", va="center", clip_on=False)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def add_time_arrow(ax, label: str = "time") -> None:
    ax.annotate(
        "",
        xy=(1.02, -0.12),
        xytext=(0.0, -0.12),
        xycoords=("axes fraction", "axes fraction"),
        arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK),
        clip_on=False,
    )
    ax.text(0.50, -0.38, label, transform=ax.transAxes, fontsize=8.0, ha="center", clip_on=False)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    traces = selected_real_motion_traces()
    t = traces["t"]
    small = traces["small_px"]
    large = traces["large_px"]
    small_xy = traces["small_xy_px"]
    large_xy = traces["large_xy_px"]
    small_trace_index = int(traces.get("small_trace_index", -1))
    large_trace_index = int(traces.get("large_trace_index", -1))
    small_path_arcmin = float(traces.get("small_path_arcmin", np.nan))
    large_path_arcmin = float(traces.get("large_path_arcmin", np.nan))
    motion_ylim = 1.15 * float(np.nanmax(np.abs(np.concatenate([small, large]))))
    if not np.isfinite(motion_ylim) or motion_ylim <= EPS:
        motion_ylim = 1.0

    specs = [
        (f"High-SF\n{HIGH_SF_CPD:g} cpd", sf_cpd_to_wavelength_px(HIGH_SF_CPD), 0.45),
        (f"Low-SF\n{LOW_SF_CPD:g} cpd", sf_cpd_to_wavelength_px(LOW_SF_CPD), 0.12),
    ]

    fig = plt.figure(figsize=(10.2, 5.1), facecolor="white")
    fig.text(0.035, 0.955, "Panel A design preview: motion through luminance gradients", fontsize=14, weight="bold", ha="left")

    fig.text(
        0.060,
        0.850,
        f"real drift traces ({FRAME_RATE_HZ:.0f} Hz; path {small_path_arcmin:.0f}' / {large_path_arcmin:.0f}')",
        fontsize=10.0,
        color=INK,
        ha="left",
        va="bottom",
    )
    motion_red_ax = fig.add_axes([0.060, 0.790, 0.200, 0.044])
    motion_blue_ax = fig.add_axes([0.060, 0.720, 0.200, 0.044])
    add_motion_trace_lane(motion_red_ax, t, small, RED, f"trace {small_trace_index}", ylim=motion_ylim)
    add_motion_trace_lane(motion_blue_ax, t, large, BLUE, f"trace {large_trace_index}", ylim=motion_ylim)
    add_time_arrow(motion_blue_ax, label="frame")

    integration_ms = 1000.0 * INTEGRATION_FRAMES / FRAME_RATE_HZ
    fig.text(
        0.690,
        0.850,
        f"sampled luminance, {INTEGRATION_FRAMES}-frame integration ({integration_ms:.0f} ms)",
        fontsize=10.5,
        color=GRAY,
        ha="left",
        va="bottom",
    )
    y_rows = [0.360, 0.075]
    for (row_label, wavelength, label_y), y0 in zip(specs, y_rows):
        fig.text(0.060, y0 + 0.130, row_label, fontsize=12.0, weight="bold", ha="left", va="center")
        grating_ax = fig.add_axes([0.190, y0, 0.135, 0.135 * fig.get_figwidth() / fig.get_figheight()])
        add_grating_panel(
            grating_ax,
            title="",
            wavelength_px=wavelength,
            small_trace_xy_px=small_xy,
            large_trace_xy_px=large_xy,
        )

        red_lum = temporal_integrate_trace(sampled_luminance_trace(small, wavelength))
        blue_lum = temporal_integrate_trace(sampled_luminance_trace(large, wavelength))
        trace_left = 0.405
        trace_w = 0.285
        lane_h = 0.060
        red_ax = fig.add_axes([trace_left, y0 + 0.112, trace_w, lane_h])
        blue_ax = fig.add_axes([trace_left, y0 + 0.018, trace_w, lane_h])
        add_luminance_lane(red_ax, t, red_lum, RED, "small")
        add_luminance_lane(blue_ax, t, blue_lum, BLUE, "large")
        add_time_arrow(blue_ax)

        fig.patches.append(
            patches.FancyArrowPatch(
                (0.340, y0 + 0.105),
                (0.390, y0 + 0.105),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.0,
                color=GRAY,
            )
        )

    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_BASE}.png", bbox_inches="tight", pad_inches=0.04, dpi=220)
    fig.savefig(f"{OUT_BASE}.pdf", bbox_inches="tight", pad_inches=0.04, dpi=300)
    fig.savefig(f"{OUT_BASE}.svg", bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close(fig)
    print(f"{OUT_BASE}.png")


if __name__ == "__main__":
    main()
