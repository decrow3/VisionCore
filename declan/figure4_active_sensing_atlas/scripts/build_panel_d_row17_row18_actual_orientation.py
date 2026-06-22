"""Render the old row-17/18 rail thumbnails with their actual image axes.

The exploratory thumbnail sheet mixed a raw-window visual lookup with a later
stability-row label. This script intentionally recovers the raw-window crops
that appeared in that sheet, then draws the local edge axis in screen/gaze
coordinates. Because the arrows are drawn in Matplotlib axes coordinates
(`y` upward), the correct display angle is the stored `image_edge_axis_deg`
itself, not its sign-flipped array-coordinate angle.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

try:  # pragma: no cover - script-mode import fallback
    from . import build_panel_d_story_options as story
except ImportError:  # pragma: no cover
    import build_panel_d_story_options as story


OUT_DIR = story.OUT_DIR
PNG = OUT_DIR / "4D_row17_row18_actual_orientation.png"
PDF = OUT_DIR / "4D_row17_row18_actual_orientation.pdf"
CSV = OUT_DIR / "4D_row17_row18_actual_orientation_values.csv"

INK = "#20262c"
MUTED = "#68727d"
GREEN = "#2f8f6a"
PURPLE = "#8064a2"


def _axis_vectors(axis_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(axis_deg)
    along = np.asarray([np.cos(theta), np.sin(theta)], dtype=float)
    across = np.asarray([np.cos(theta + np.pi / 2.0), np.sin(theta + np.pi / 2.0)], dtype=float)
    return along, across


def _add_axis_arrows(ax: plt.Axes, axis_deg: float, *, scale: float = 0.31) -> None:
    along, across = _axis_vectors(axis_deg)
    center = np.asarray([0.50, 0.50])
    for vector, color, label_xy, label in [
        (along, GREEN, (0.56, 0.28), "along edge"),
        (across, PURPLE, (0.08, 0.78), "across edge"),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                tuple(center - vector * scale),
                tuple(center + vector * scale),
                arrowstyle="<|-|>",
                mutation_scale=13,
                linewidth=2.5,
                color=color,
                transform=ax.transAxes,
            )
        )
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )


def _plot_row(ax: plt.Axes, row: pd.Series, raw_index: int) -> dict[str, object]:
    canvas, ppd, screen_shape = story._backimage_canvas(str(row["session"]), int(row["trial_idx"]))
    center = story.gaze_deg_to_screen_px(
        np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
        ppd=float(ppd),
        screen_shape=screen_shape,
    )
    patch = story._crop_centered(canvas, (float(center[0]), float(center[1])), 190)
    ax.imshow(story._norm_image(patch), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    axis_deg = float(row["image_edge_axis_deg"])
    _add_axis_arrows(ax, axis_deg)
    ax.set_title(
        f"raw row {raw_index} | edge {axis_deg:+.1f} deg\n"
        f"{row['session']} trial {int(row['trial_idx'])} | coh {float(row['image_orientation_coherence']):.2f}",
        color=INK,
        fontsize=8.5,
        pad=8,
    )
    return {
        "raw_window_row": int(raw_index),
        "session": str(row["session"]),
        "trial_idx": int(row["trial_idx"]),
        "mean_x_deg": float(row["mean_x_deg"]),
        "mean_y_deg": float(row["mean_y_deg"]),
        "image_edge_axis_deg": axis_deg,
        "image_edge_axis_array_deg": float(row.get("image_edge_axis_array_deg", np.nan)),
        "image_gradient_axis_deg": float(row.get("image_gradient_axis_deg", np.nan)),
        "image_orientation_coherence": float(row["image_orientation_coherence"]),
    }


def build() -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(story.WINDOWS_CSV)
    rows = [17, 18]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), constrained_layout=False)
    fig.patch.set_facecolor("white")
    records = []
    for ax, idx in zip(np.ravel(axes), rows, strict=True):
        records.append(_plot_row(ax, raw.iloc[int(idx)], int(idx)))
    fig.suptitle(
        "Recovered row-17/18 rail crops with actual local orientations",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=12.8,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.915,
        "Green is along the local edge; purple is across it. Axes use screen/gaze coordinates, so no display sign flip is applied.",
        ha="left",
        va="top",
        fontsize=8.2,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.78, bottom=0.04, wspace=0.12)
    fig.savefig(PNG, dpi=260, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(records).to_csv(CSV, index=False)
    return [PNG, PDF, CSV]


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()
