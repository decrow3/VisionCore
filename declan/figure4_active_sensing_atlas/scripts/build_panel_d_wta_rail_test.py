#!/usr/bin/env python3
"""Stress-test the current WTA axis estimator on the row-17/18 rail crop."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from scipy import ndimage

try:  # pragma: no cover
    from declan.figure4_active_sensing_atlas.scripts import build_panel_d_story_options as story
    from declan.figure4_active_sensing_atlas.scripts.run_panel_d_wta_behavior_diagnostic import _wta_axis_from_patch
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from declan.figure4_active_sensing_atlas.scripts import build_panel_d_story_options as story
    from declan.figure4_active_sensing_atlas.scripts.run_panel_d_wta_behavior_diagnostic import _wta_axis_from_patch


OUT_DIR = story.OUT_DIR
PNG = OUT_DIR / "4D_wta_rail_thumbnail_stress_test.png"
PDF = OUT_DIR / "4D_wta_rail_thumbnail_stress_test.pdf"
CSV = OUT_DIR / "4D_wta_rail_thumbnail_stress_test_values.csv"

INK = "#20262c"
MUTED = "#68727d"
AVG = "#244f7a"
WTA = "#c15b44"
HOUGH = "#2f8f6a"
GRID = "#dfe4e9"


def _axis_delta_deg(a_deg: float, b_deg: float) -> float:
    return float(0.5 * np.degrees(np.angle(np.exp(2j * np.radians(float(a_deg) - float(b_deg))))))


def _axis_vector(axis_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(axis_deg))
    return np.asarray([np.cos(theta), np.sin(theta)], dtype=float)


def _add_axis(ax: plt.Axes, axis_deg: float, color: str, *, scale: float, lw: float = 2.0) -> None:
    vec = _axis_vector(axis_deg)
    center = np.asarray([0.5, 0.5])
    ax.add_patch(
        FancyArrowPatch(
            tuple(center - vec * scale),
            tuple(center + vec * scale),
            arrowstyle="<|-|>",
            mutation_scale=9,
            linewidth=lw,
            color=color,
            transform=ax.transAxes,
            alpha=0.95,
        )
    )


def _norm_image(image: np.ndarray) -> np.ndarray:
    return story._norm_image(np.asarray(image, dtype=np.float64))


def _recover_row(raw_index: int) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    raw = pd.read_csv(story.WINDOWS_CSV)
    row = raw.iloc[int(raw_index)]
    canvas, ppd, screen_shape = story._backimage_canvas(str(row["session"]), int(row["trial_idx"]))
    center = story.gaze_deg_to_screen_px(
        np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
        ppd=float(ppd),
        screen_shape=screen_shape,
    )
    thumbnail_patch = story._crop_centered(canvas, (float(center[0]), float(center[1])), 190)
    radius = int(round(float(row["image_patch_radius_px"])))
    analysis_patch = canvas[
        max(0, int(round(float(center[1]))) - radius) : min(canvas.shape[0], int(round(float(center[1]))) + radius + 1),
        max(0, int(round(float(center[0]))) - radius) : min(canvas.shape[1], int(round(float(center[0]))) + radius + 1),
    ]
    return np.asarray(thumbnail_patch, dtype=np.float64), np.asarray(analysis_patch, dtype=np.float64), row


def _hough_axis_deg(patch: np.ndarray) -> float:
    img = (_norm_image(patch) * 255.0).astype("uint8")
    edges = cv2.Canny(img, 80, 180)
    min_line = max(12, int(round(min(patch.shape) * 0.24)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=min_line, maxLineGap=15)
    if lines is None:
        return float("nan")
    segments = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        array_axis = float(((np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 90.0) % 180.0) - 90.0)
        display_axis = -array_axis
        if 20.0 <= array_axis <= 55.0 and length >= min_line:
            segments.append((length, array_axis, display_axis))
    if not segments:
        return float("nan")
    weights = np.asarray([s[0] for s in segments], dtype=np.float64)
    angles = np.radians([s[1] for s in segments])
    mean_array = 0.5 * np.arctan2(
        float(np.sum(weights * np.sin(2.0 * angles))),
        float(np.sum(weights * np.cos(2.0 * angles))),
    )
    return -float(np.degrees(mean_array))


def _orientation_histogram(patch: np.ndarray, *, n_bins: int = 36, energy_quantile: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    gx = ndimage.sobel(np.asarray(patch, dtype=np.float64), axis=1, mode="nearest")
    gy = ndimage.sobel(np.asarray(patch, dtype=np.float64), axis=0, mode="nearest")
    energy = gx * gx + gy * gy
    finite = np.isfinite(energy)
    threshold = float(np.nanquantile(energy[finite], float(energy_quantile)))
    mask = finite & (energy >= threshold) & (energy > 0)
    gradient_axis_array = np.degrees(np.arctan2(gy[mask], gx[mask]))
    edge_axis_array = ((gradient_axis_array + 180.0) % 180.0) - 90.0
    edge_axis_display = -edge_axis_array
    weights = energy[mask].astype(np.float64)
    edges = np.linspace(-90.0, 90.0, int(n_bins) + 1)
    hist, edges = np.histogram(edge_axis_display, bins=edges, weights=weights)
    if float(np.sum(hist)) > 0:
        hist = hist / float(np.sum(hist))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def _plot_image(ax: plt.Axes, patch: np.ndarray, avg_axis: float, wta_axis: float, hough_axis: float, title: str) -> None:
    ax.imshow(_norm_image(patch), cmap="gray", vmin=0, vmax=1)
    _add_axis(ax, avg_axis, AVG, scale=0.36, lw=2.0)
    _add_axis(ax, wta_axis, WTA, scale=0.28, lw=2.0)
    if np.isfinite(hough_axis):
        _add_axis(ax, hough_axis, HOUGH, scale=0.20, lw=2.1)
    ax.set_title(title, fontsize=8.2, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_hist(ax: plt.Axes, patch: np.ndarray, avg_axis: float, wta_axis: float, hough_axis: float, title: str) -> None:
    centers, hist = _orientation_histogram(patch)
    ax.bar(centers, hist, width=4.5, color="#87919a", alpha=0.88)
    for axis, color, label in [(avg_axis, AVG, "average"), (wta_axis, WTA, "WTA"), (hough_axis, HOUGH, "rail fit")]:
        if np.isfinite(axis):
            ax.axvline(axis, color=color, lw=1.8, label=label)
    ax.set_xlim(-90, 90)
    ax.set_xlabel("display edge axis (deg)")
    ax.set_ylabel("energy fraction")
    ax.set_title(title, fontsize=8.2, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=6.7, loc="upper left")


def build() -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    fig = plt.figure(figsize=(11.4, 6.6), constrained_layout=False)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.92], hspace=0.44, wspace=0.28)

    for col, raw_index in enumerate([17, 18]):
        thumbnail, analysis, row = _recover_row(raw_index)
        avg_axis = float(row["image_edge_axis_deg"])
        hough_thumbnail = _hough_axis_deg(thumbnail)
        hough_analysis = _hough_axis_deg(analysis)
        wta_thumbnail = float(_wta_axis_from_patch(thumbnail, n_bins=36, energy_quantile=0.75)["wta_edge_axis_deg"])
        wta_analysis = float(_wta_axis_from_patch(analysis, n_bins=36, energy_quantile=0.75)["wta_edge_axis_deg"])

        _plot_image(
            fig.add_subplot(gs[0, col * 2]),
            thumbnail,
            avg_axis,
            wta_thumbnail,
            hough_thumbnail,
            f"row {raw_index}: 190 px thumbnail\navg {avg_axis:+.1f}, WTA {wta_thumbnail:+.1f}, rail {hough_thumbnail:+.1f}",
        )
        _plot_image(
            fig.add_subplot(gs[0, col * 2 + 1]),
            analysis,
            avg_axis,
            wta_analysis,
            hough_analysis,
            f"row {raw_index}: local analysis patch\navg {avg_axis:+.1f}, WTA {wta_analysis:+.1f}, rail {hough_analysis:+.1f}",
        )
        _plot_hist(
            fig.add_subplot(gs[1, col * 2]),
            thumbnail,
            avg_axis,
            wta_thumbnail,
            hough_thumbnail,
            "thumbnail orientation-energy WTA",
        )
        _plot_hist(
            fig.add_subplot(gs[1, col * 2 + 1]),
            analysis,
            avg_axis,
            wta_analysis,
            hough_analysis,
            "analysis-patch orientation-energy WTA",
        )
        records.extend(
            [
                {
                    "raw_window_row": int(raw_index),
                    "patch": "thumbnail_190px",
                    "session": str(row["session"]),
                    "trial_idx": int(row["trial_idx"]),
                    "stored_average_axis_deg": avg_axis,
                    "current_wta_axis_deg": wta_thumbnail,
                    "hough_visible_rail_axis_deg": hough_thumbnail,
                    "wta_minus_hough_abs_delta_deg": abs(_axis_delta_deg(wta_thumbnail, hough_thumbnail)),
                    "average_minus_hough_abs_delta_deg": abs(_axis_delta_deg(avg_axis, hough_thumbnail)),
                    "image_orientation_coherence": float(row["image_orientation_coherence"]),
                },
                {
                    "raw_window_row": int(raw_index),
                    "patch": "analysis_patch",
                    "session": str(row["session"]),
                    "trial_idx": int(row["trial_idx"]),
                    "stored_average_axis_deg": avg_axis,
                    "current_wta_axis_deg": wta_analysis,
                    "hough_visible_rail_axis_deg": hough_analysis,
                    "wta_minus_hough_abs_delta_deg": abs(_axis_delta_deg(wta_analysis, hough_analysis)),
                    "average_minus_hough_abs_delta_deg": abs(_axis_delta_deg(avg_axis, hough_analysis)),
                    "image_orientation_coherence": float(row["image_orientation_coherence"]),
                },
            ]
        )

    fig.suptitle(
        "Current WTA estimator stress test on the diagonal rail thumbnail",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=14.0,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.94,
        "Blue = stored patch-average orientation. Orange = current orientation-energy WTA. Green = visible rail/Hough fit. If orange misses green here, this WTA rule is not a valid prominent-contour estimator.",
        ha="left",
        va="top",
        fontsize=8.9,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.08)
    fig.savefig(PNG, dpi=240, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(records).to_csv(CSV, index=False)
    return [PNG, PDF, CSV]


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()
