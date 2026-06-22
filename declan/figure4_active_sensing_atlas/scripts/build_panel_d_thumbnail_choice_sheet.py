"""Build a corrected real-BackImage thumbnail choice sheet for Figure 4D."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

try:  # pragma: no cover - script-mode import path fallback
    from . import build_panel_d_story_options as story
except ImportError:  # pragma: no cover
    import build_panel_d_story_options as story


OUT_DIR = story.OUT_DIR
SHEET = OUT_DIR / "4D_thumbnail_choice_sheet_corrected.png"
VALUES = OUT_DIR / "4D_thumbnail_choice_values.csv"

GREEN = "#2f8f6a"
PURPLE = "#8064a2"
INK = "#20262c"
MUTED = "#68727d"


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
    half = size // 2
    x0 = max(0, min(width - size, int(round(cx)) - half))
    y0 = max(0, min(height - size, int(round(cy)) - half))
    return image[y0 : y0 + size, x0 : x0 + size]


def _axis_vectors(axis_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(axis_deg)
    along = np.asarray([np.cos(theta), np.sin(theta)], dtype=float)
    across = np.asarray([np.cos(theta + np.pi / 2.0), np.sin(theta + np.pi / 2.0)], dtype=float)
    return along, across


def _score_candidates(stability: pd.DataFrame, source_windows: pd.DataFrame) -> pd.DataFrame:
    work = stability[
        (stability["twin_stability_advantage"].astype(float) > 0)
        & (stability["image_orientation_coherence"].astype(float) >= 0.45)
    ].copy()
    if work.empty:
        raise ValueError("No positive model-preservation candidates passed the thumbnail filter.")

    features = []
    for row in work.itertuples(index=False):
        source = source_windows.iloc[int(row.window_row)]
        features.append(
            {
                "source_rms_contrast": float(source["image_patch_rms_contrast"]),
                "source_gradient_energy": float(source["image_gradient_energy"]),
                "source_edge_density": float(source["image_edge_density"]),
                "source_fraction_background": float(source["image_patch_fraction_background"]),
            }
        )
    feat = pd.DataFrame(features, index=work.index)
    work = pd.concat([work, feat], axis=1)
    work = work[work["source_fraction_background"].astype(float) <= 0.02].copy()

    score_cols = [
        ("image_orientation_coherence", 0.30),
        ("source_rms_contrast", 0.25),
        ("source_gradient_energy", 0.25),
        ("source_edge_density", 0.10),
        ("twin_stability_advantage", 0.10),
    ]
    work["visual_score"] = 0.0
    for column, weight in score_cols:
        values = work[column].astype(float)
        lo, hi = values.quantile(0.05), values.quantile(0.95)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            scaled = np.zeros(work.shape[0], dtype=float)
        else:
            scaled = ((values - lo) / (hi - lo)).clip(0.0, 1.0)
        work["visual_score"] += float(weight) * scaled
    return work.sort_values("visual_score", ascending=False)


def _selected_rows(stability: pd.DataFrame, source_windows: pd.DataFrame, n: int) -> pd.DataFrame:
    ranked = _score_candidates(stability, source_windows)
    preferred = [17, 34, 26, 29, 60]
    rows = []
    used = set()
    for row_id in preferred:
        hit = ranked[ranked["window_row"].astype(int).eq(row_id)]
        if not hit.empty:
            rows.append(hit.iloc[0])
            used.add(row_id)
    for row in ranked.itertuples(index=False):
        row_id = int(row.window_row)
        if row_id not in used:
            rows.append(pd.Series(row._asdict()))
            used.add(row_id)
        if len(rows) >= n:
            break
    return pd.DataFrame(rows).reset_index(drop=True)


def _plot_thumbnail(ax: plt.Axes, row: pd.Series, source_windows: pd.DataFrame) -> dict[str, object]:
    source = source_windows.iloc[int(row["window_row"])]
    canvas, ppd, screen_shape = story._backimage_canvas(str(source["session"]), int(source["trial_idx"]))
    center = story.gaze_deg_to_screen_px(
        np.asarray([float(source["mean_x_deg"]), float(source["mean_y_deg"])]),
        ppd=float(ppd),
        screen_shape=screen_shape,
    )
    patch = _crop_centered(canvas, (float(center[0]), float(center[1])), 190)
    ax.imshow(_norm_image(patch), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    display_axis_deg = float(row["edge_axis_deg"])
    along, across = _axis_vectors(display_axis_deg)
    center_axes = np.asarray([0.50, 0.50])
    scale = 0.34
    for vector, color in [(along, GREEN), (across, PURPLE)]:
        ax.add_patch(
            FancyArrowPatch(
                tuple(center_axes - vector * scale),
                tuple(center_axes + vector * scale),
                arrowstyle="<|-|>",
                mutation_scale=9,
                linewidth=1.7,
                color=color,
                transform=ax.transAxes,
            )
        )

    ax.set_title(
        f"row {int(row['window_row'])} | coh {float(row['image_orientation_coherence']):.2f}\n"
        f"model Δ {float(row['twin_stability_advantage']) * 1e4:.1f} | axis {display_axis_deg:+.0f}°",
        fontsize=8.0,
        color=INK,
    )
    return {
        "window_row": int(row["window_row"]),
        "source_window_id": int(source["window_id"]),
        "session": str(source["session"]),
        "trial_idx": int(source["trial_idx"]),
        "display_axis_deg": float(display_axis_deg),
        "stability_edge_axis_gaze_deg": float(row["edge_axis_deg"]),
        "image_orientation_coherence": float(row["image_orientation_coherence"]),
        "source_rms_contrast": float(source["image_patch_rms_contrast"]),
        "source_gradient_energy": float(source["image_gradient_energy"]),
        "source_edge_density": float(source["image_edge_density"]),
        "twin_parallel_cost_x1e4": float(row["twin_parallel_cost"]) * 1e4,
        "twin_orthogonal_cost_x1e4": float(row["twin_orthogonal_cost"]) * 1e4,
        "twin_stability_advantage_x1e4": float(row["twin_stability_advantage"]) * 1e4,
        "pixel_stability_advantage": float(row["pixel_stability_advantage"]),
        "visual_score": float(row.get("visual_score", np.nan)),
    }


def _rows_by_id(stability: pd.DataFrame, row_ids: list[int]) -> pd.DataFrame:
    rows = []
    for row_id in row_ids:
        hit = stability[stability["window_row"].astype(int).eq(int(row_id))]
        if hit.empty:
            raise ValueError(f"Requested window_row {row_id} is not present in the stability table.")
        rows.append(hit.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def build(n: int = 18, row_ids: list[int] | None = None, session: str = "") -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_windows = story._stability_source_windows()
    stability = pd.read_csv(story.STABILITY_DIR / "edge_parallel_stability_by_window.csv")
    if session:
        allowed = set(source_windows[source_windows["session"].astype(str).eq(session)].index.astype(int))
        stability = stability[stability["window_row"].astype(int).isin(allowed)].copy()
        if stability.empty:
            raise ValueError(f"No stability rows found for session {session!r}.")
    selected = _rows_by_id(stability, row_ids) if row_ids else _selected_rows(stability, source_windows, n=n)

    n_cols = 6
    n_rows = int(np.ceil(selected.shape[0] / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.5, 2.35 * n_rows + 0.85), constrained_layout=False)
    axes_arr = np.asarray(axes).reshape(-1)
    records = []
    for ax, (_, row) in zip(axes_arr, selected.iterrows(), strict=False):
        records.append(_plot_thumbnail(ax, row, source_windows))
    for ax in axes_arr[selected.shape[0] :]:
        ax.axis("off")
    fig.suptitle(
        "Figure 4D Thumbnail Choices: corrected BackImage windows and edge axes",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.945,
        "Green is along the local edge; purple is across it. Model Δ is orthogonal-minus-parallel response cost, x1e4.",
        ha="left",
        va="top",
        fontsize=9,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.02, right=0.985, top=0.84, bottom=0.04, hspace=0.42, wspace=0.05)
    fig.savefig(SHEET, dpi=220, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(records).to_csv(VALUES, index=False)
    return [SHEET, VALUES]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=18)
    parser.add_argument(
        "--rows",
        default="",
        help="Comma-separated stability window_row ids to render in order. Overrides --n.",
    )
    parser.add_argument("--session", default="", help="Restrict automatically selected rows to one session.")
    args = parser.parse_args()
    row_ids = [int(v) for v in str(args.rows).split(",") if v.strip()] or None
    for path in build(n=int(args.n), row_ids=row_ids, session=str(args.session)):
        print(path)


if __name__ == "__main__":
    main()
