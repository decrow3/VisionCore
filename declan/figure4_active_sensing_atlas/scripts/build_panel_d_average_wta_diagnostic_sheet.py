#!/usr/bin/env python3
"""Build quick diagnostic plots for average-vs-WTA Figure 4D reruns."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

try:
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE = REPO_ROOT / "outputs" / "fixation_statistics_by_stimulus_all_sessions_after_review"
ATLAS = REPO_ROOT / "declan" / "figure4_active_sensing_atlas"
OUT_DIR = ATLAS / "figures" / "panel_D" / "diagnostics"

AVERAGE_FEATURE = BASE / "backimage_axis_conditioned_matched_static_feature_posterior_gabor_pyramid_k4_8_average_axis_wta_comparison_v1"
WTA_FEATURE = BASE / "backimage_axis_conditioned_matched_static_feature_posterior_gabor_pyramid_k4_8_wta_axis_wta_comparison_v1"
WTA_VALUES = BASE / "backimage_wta_orientation_axis_input_v1" / "selected_wta_axis_values.csv"
WTA_INPUT = BASE / "backimage_wta_orientation_axis_input_v1" / "backimage_image_fem_windows_wta_axis.csv"

PNG = OUT_DIR / "4D_average_vs_wta_diagnostic_sheet.png"
PDF = OUT_DIR / "4D_average_vs_wta_diagnostic_sheet.pdf"
VALUES = OUT_DIR / "4D_average_vs_wta_diagnostic_values.csv"

INK = "#20262c"
MUTED = "#68727d"
GRID = "#dfe4e9"
AVG = "#244f7a"
WTA = "#c15b44"


def _axis_delta_deg(a_deg: np.ndarray | float, b_deg: np.ndarray | float) -> np.ndarray:
    return 0.5 * np.degrees(np.angle(np.exp(2j * np.radians(np.asarray(a_deg) - np.asarray(b_deg)))))


def _norm_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    lo, hi = np.nanpercentile(arr, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _crop_centered(image: np.ndarray, center_xy: tuple[float, float], size: int) -> np.ndarray:
    height, width = image.shape[:2]
    cx, cy = center_xy
    half = int(size) // 2
    x0 = max(0, min(width - int(size), int(round(cx)) - half))
    y0 = max(0, min(height - int(size), int(round(cy)) - half))
    return image[y0 : y0 + int(size), x0 : x0 + int(size)]


def _axis_vector(axis_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(axis_deg))
    return np.asarray([np.cos(theta), np.sin(theta)], dtype=float)


def _add_axis(ax: plt.Axes, axis_deg: float, color: str, *, scale: float, lw: float) -> None:
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
            alpha=0.95,
            transform=ax.transAxes,
        )
    )


def _load_contrasts() -> pd.DataFrame:
    rows = []
    for axis_estimator, path in [("average", AVERAGE_FEATURE), ("wta", WTA_FEATURE)]:
        df = pd.read_csv(path / "feature_axis_contrasts.csv")
        df.insert(0, "axis_estimator", axis_estimator)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["latent_label"] = out["latent"].map(
        {
            "gabor_local_field": "gabor",
            "pyramid_local_field": "pyramid",
        }
    ).fillna(out["latent"].astype(str))
    out["feature_label"] = out["latent_label"].astype(str) + " k=" + out["requested_k"].astype(str)
    return out


def _plot_contrast_bars(ax: plt.Axes, contrasts: pd.DataFrame) -> None:
    order = ["gabor k=4", "gabor k=8", "pyramid k=4", "pyramid k=8"]
    x = np.arange(len(order), dtype=float)
    width = 0.34
    for offset, axis_estimator, color, label in [(-width / 2, "average", AVG, "average axis"), (width / 2, "wta", WTA, "WTA axis")]:
        block = contrasts[contrasts["axis_estimator"].eq(axis_estimator)].set_index("feature_label")
        y = np.asarray([float(block.loc[key, "mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal"]) for key in order])
        lo = np.asarray([float(block.loc[key, "mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal_ci_low"]) for key in order])
        hi = np.asarray([float(block.loc[key, "mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal_ci_high"]) for key in order])
        yerr = np.vstack([y - lo, hi - y])
        ax.bar(x + offset, y, width=width, color=color, label=label)
        ax.errorbar(x + offset, y, yerr=yerr, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylabel("feature gain: parallel - orthogonal")
    ax.set_title("Axis-conditioned decoding")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_delta_bars(ax: plt.Axes, contrasts: pd.DataFrame) -> None:
    pivot = contrasts.pivot(index="feature_label", columns="axis_estimator", values="mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal")
    order = ["gabor k=4", "gabor k=8", "pyramid k=4", "pyramid k=8"]
    delta = np.asarray([float(pivot.loc[key, "wta"] - pivot.loc[key, "average"]) for key in order])
    colors = [WTA if value >= 0 else "#8d96a0" for value in delta]
    ax.barh(np.arange(len(order)), delta, color=colors)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("WTA - average")
    ax.set_title("Estimator sensitivity")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_axis_hist(ax: plt.Axes, values: pd.DataFrame) -> None:
    ax.hist(values["wta_average_axis_delta_deg"], bins=np.linspace(0, 30, 16), color="#7d8790", alpha=0.9)
    ax.axvline(float(values["wta_average_axis_delta_deg"].median()), color=INK, lw=1.1, label="median")
    ax.set_xlabel("|WTA axis - average axis| (deg)")
    ax.set_ylabel("windows")
    ax.set_title("Axis disagreement across paired windows")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_thumbnail(ax: plt.Axes, row: pd.Series, full_windows: pd.DataFrame) -> dict[str, object]:
    source_row = int(row["source_row"])
    source = full_windows.loc[full_windows["source_row"].astype(int).eq(source_row)].iloc[0]
    canvas, ppd, screen_shape = _backimage_canvas(str(source["session"]), int(source["trial_idx"]))
    center = gaze_deg_to_screen_px(
        np.asarray([float(source["mean_x_deg"]), float(source["mean_y_deg"])]),
        ppd=float(ppd),
        screen_shape=screen_shape,
    )
    patch = _crop_centered(canvas, (float(center[0]), float(center[1])), 190)
    ax.imshow(_norm_image(patch), cmap="gray", vmin=0, vmax=1)
    _add_axis(ax, float(row["image_edge_axis_deg"]), AVG, scale=0.35, lw=2.0)
    _add_axis(ax, float(row["wta_edge_axis_deg"]), WTA, scale=0.27, lw=2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"src {source_row} | Δ {float(row['wta_average_axis_delta_deg']):.1f}°\n"
        f"avg {float(row['image_edge_axis_deg']):+.0f}°, WTA {float(row['wta_edge_axis_deg']):+.0f}°",
        fontsize=7.3,
        color=INK,
        pad=4,
    )
    return {
        "source_row": source_row,
        "session": str(source["session"]),
        "trial_idx": int(source["trial_idx"]),
        "image_edge_axis_deg": float(row["image_edge_axis_deg"]),
        "wta_edge_axis_deg": float(row["wta_edge_axis_deg"]),
        "wta_peak_fraction": float(row["wta_peak_fraction"]),
        "wta_average_axis_delta_deg": float(row["wta_average_axis_delta_deg"]),
    }


def build() -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    contrasts = _load_contrasts()
    values = pd.read_csv(WTA_VALUES)
    full_windows = pd.read_csv(WTA_INPUT)
    values["wta_average_axis_delta_deg"] = np.abs(_axis_delta_deg(values["wta_edge_axis_deg"], values["image_edge_axis_deg"]))
    thumb_rows = values.sort_values(["wta_average_axis_delta_deg", "wta_peak_fraction"], ascending=False).head(8)

    fig = plt.figure(figsize=(11.8, 8.8), constrained_layout=False)
    gs = GridSpec(4, 4, figure=fig, height_ratios=[0.90, 0.80, 1.0, 1.0], hspace=0.58, wspace=0.34)
    ax_bar = fig.add_subplot(gs[0:2, 0:2])
    ax_delta = fig.add_subplot(gs[0, 2:4])
    ax_hist = fig.add_subplot(gs[1, 2:4])
    _plot_contrast_bars(ax_bar, contrasts)
    _plot_delta_bars(ax_delta, contrasts)
    _plot_axis_hist(ax_hist, values)

    thumb_records = []
    for i, (_, row) in enumerate(thumb_rows.iterrows()):
        ax = fig.add_subplot(gs[2 + i // 4, i % 4])
        thumb_records.append(_plot_thumbnail(ax, row, full_windows))

    fig.suptitle(
        "Figure 4D average-orientation vs WTA-orientation diagnostics",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=15.0,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.945,
        "Blue axis = stored patch-average orientation energy. Orange axis = image-only winner-take-all local orientation mode. Thumbnails show largest estimator disagreements in the paired n=64 rerun.",
        ha="left",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.89, bottom=0.055)
    fig.savefig(PNG, dpi=230, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    plt.close(fig)

    contrast_values = contrasts[
        [
            "axis_estimator",
            "latent",
            "requested_k",
            "mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal",
            "mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal_ci_low",
            "mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal_ci_high",
            "mean_motion_delta_parallel_minus_orthogonal",
            "mean_motion_delta_parallel_minus_orthogonal_ci_low",
            "mean_motion_delta_parallel_minus_orthogonal_ci_high",
        ]
    ].copy()
    contrast_values["record_type"] = "contrast"
    thumb_values = pd.DataFrame(thumb_records)
    thumb_values["record_type"] = "thumbnail"
    pd.concat([contrast_values, thumb_values], ignore_index=True, sort=False).to_csv(VALUES, index=False)
    return [PNG, PDF, VALUES]


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()
