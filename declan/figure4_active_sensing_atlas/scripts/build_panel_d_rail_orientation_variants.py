"""Render angle variants for the recovered row-17 rail thumbnail."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

try:  # pragma: no cover
    from . import build_panel_d_story_options as story
except ImportError:  # pragma: no cover
    import build_panel_d_story_options as story


OUT_DIR = story.OUT_DIR
PNG = OUT_DIR / "4D_row17_rail_orientation_variants.png"
PDF = OUT_DIR / "4D_row17_rail_orientation_variants.pdf"
CSV = OUT_DIR / "4D_row17_rail_orientation_variants_values.csv"
SELECTED_PNG = OUT_DIR / "4D_row17_row18_visible_rail_fit_orientation.png"
SELECTED_PDF = OUT_DIR / "4D_row17_row18_visible_rail_fit_orientation.pdf"
SELECTED_CSV = OUT_DIR / "4D_row17_row18_visible_rail_fit_orientation_values.csv"

INK = "#20262c"
MUTED = "#68727d"
GREEN = "#2f8f6a"
PURPLE = "#8064a2"


def _axis_vectors(axis_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(axis_deg)
    along = np.asarray([np.cos(theta), np.sin(theta)], dtype=float)
    across = np.asarray([np.cos(theta + np.pi / 2.0), np.sin(theta + np.pi / 2.0)], dtype=float)
    return along, across


def _add_arrows(ax: plt.Axes, axis_deg: float, *, scale: float = 0.32) -> None:
    along, across = _axis_vectors(axis_deg)
    center = np.asarray([0.50, 0.50])
    for vector, color in [(along, GREEN), (across, PURPLE)]:
        ax.add_patch(
            FancyArrowPatch(
                tuple(center - vector * scale),
                tuple(center + vector * scale),
                arrowstyle="<|-|>",
                mutation_scale=12,
                linewidth=2.4,
                color=color,
                transform=ax.transAxes,
            )
        )


def _recover_raw_row_patch(raw_index: int = 17) -> tuple[np.ndarray, pd.Series]:
    raw = pd.read_csv(story.WINDOWS_CSV)
    row = raw.iloc[int(raw_index)]
    canvas, ppd, screen_shape = story._backimage_canvas(str(row["session"]), int(row["trial_idx"]))
    center = story.gaze_deg_to_screen_px(
        np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
        ppd=float(ppd),
        screen_shape=screen_shape,
    )
    patch = story._crop_centered(canvas, (float(center[0]), float(center[1])), 190)
    return patch, row


def _hough_rail_angle_deg(patch: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    img = (story._norm_image(patch) * 255.0).astype("uint8")
    edges = cv2.Canny(img, 80, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=45, maxLineGap=15)
    if lines is None:
        return float("nan"), []
    segments: list[dict[str, float]] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        array_angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        array_axis = float(((array_angle + 90.0) % 180.0) - 90.0)
        display_axis = -array_axis
        segments.append(
            {
                "length": length,
                "array_axis_deg": array_axis,
                "display_axis_deg": display_axis,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    diagonal = [s for s in segments if 20.0 <= s["array_axis_deg"] <= 55.0 and s["length"] > 50.0]
    if not diagonal:
        return float("nan"), segments
    weights = np.asarray([s["length"] for s in diagonal], dtype=float)
    angles = np.deg2rad([s["array_axis_deg"] for s in diagonal])
    mean_array = 0.5 * np.arctan2(
        float(np.sum(weights * np.sin(2.0 * angles))),
        float(np.sum(weights * np.cos(2.0 * angles))),
    )
    return -float(np.degrees(mean_array)), segments


def _plot_selected_rail_fit() -> list[str]:
    raw = pd.read_csv(story.WINDOWS_CSV)
    rows = [17, 18]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.55), constrained_layout=False)
    fig.patch.set_facecolor("white")
    records: list[dict[str, object]] = []

    for ax, raw_index in zip(np.ravel(axes), rows, strict=True):
        patch, row = _recover_raw_row_patch(raw_index)
        hough_axis, _segments = _hough_rail_angle_deg(patch)
        aggregate_axis = float(row["image_edge_axis_deg"])
        ax.imshow(story._norm_image(patch), cmap="gray", vmin=0, vmax=1)
        _add_arrows(ax, hough_axis)
        ax.set_title(
            f"raw row {raw_index} | rail fit {hough_axis:+.1f} deg\n"
            f"stored aggregate {aggregate_axis:+.1f} deg | coh {float(row['image_orientation_coherence']):.2f}",
            fontsize=8.0,
            color=INK,
            pad=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        records.append(
            {
                "raw_window_row": int(raw_index),
                "session": str(row["session"]),
                "trial_idx": int(row["trial_idx"]),
                "mean_x_deg": float(row["mean_x_deg"]),
                "mean_y_deg": float(row["mean_y_deg"]),
                "visible_rail_fit_axis_deg": float(hough_axis),
                "stored_image_edge_axis_deg": aggregate_axis,
                "image_edge_axis_array_deg": float(row.get("image_edge_axis_array_deg", np.nan)),
                "image_gradient_axis_deg": float(row.get("image_gradient_axis_deg", np.nan)),
                "image_orientation_coherence": float(row["image_orientation_coherence"]),
            }
        )

    fig.suptitle(
        "Recovered rail crops with visible-contour fit",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=12.4,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.915,
        "Green is along the fitted bright rail; purple is across it. CSV retains the stored aggregate local image axis.",
        ha="left",
        va="top",
        fontsize=8.0,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.77, bottom=0.04, wspace=0.11)
    fig.savefig(SELECTED_PNG, dpi=260, bbox_inches="tight")
    fig.savefig(SELECTED_PDF, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(records).to_csv(SELECTED_CSV, index=False)
    return [str(SELECTED_PNG), str(SELECTED_PDF), str(SELECTED_CSV)]


def build() -> list[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patch, row = _recover_raw_row_patch(17)
    aggregate_axis = float(row["image_edge_axis_deg"])
    hough_axis, segments = _hough_rail_angle_deg(patch)

    variants = [
        ("stored aggregate", aggregate_axis),
        ("Hough rail fit", hough_axis),
        ("rail -40 deg", -40.0),
        ("rail -42 deg", -42.0),
        ("rail -45 deg", -45.0),
    ]
    variants = [(label, angle) for label, angle in variants if np.isfinite(angle)]

    fig, axes = plt.subplots(1, len(variants), figsize=(2.15 * len(variants), 3.05), constrained_layout=False)
    fig.patch.set_facecolor("white")
    axes_arr = np.ravel(axes)
    image = story._norm_image(patch)
    for ax, (label, angle) in zip(axes_arr, variants, strict=True):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        _add_arrows(ax, angle)
        ax.set_title(f"{label}\n{angle:+.1f} deg", fontsize=8.2, color=INK, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "Row-17 rail orientation variants",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=12.0,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.915,
        f"Raw row 17, {row['session']} trial {int(row['trial_idx'])}. Green is along; purple is across.",
        ha="left",
        va="top",
        fontsize=8.4,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.78, bottom=0.02, wspace=0.05)
    fig.savefig(PNG, dpi=260, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    plt.close(fig)

    records = [
        {
            "variant": label,
            "display_axis_deg": angle,
            "raw_window_row": 17,
            "session": str(row["session"]),
            "trial_idx": int(row["trial_idx"]),
            "stored_image_edge_axis_deg": aggregate_axis,
            "hough_rail_fit_axis_deg": hough_axis,
        }
        for label, angle in variants
    ]
    pd.DataFrame(records).to_csv(CSV, index=False)
    pd.DataFrame(segments).to_csv(OUT_DIR / "4D_row17_rail_hough_segments.csv", index=False)
    return [
        str(PNG),
        str(PDF),
        str(CSV),
        str(OUT_DIR / "4D_row17_rail_hough_segments.csv"),
        *_plot_selected_rail_fit(),
    ]


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()
