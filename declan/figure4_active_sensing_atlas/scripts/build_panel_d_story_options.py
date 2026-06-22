"""Build story-first option sheets for Figure 4D.

The older D promotion sheet chose among analysis modules. This sheet asks a
slightly different design question: what should Panel D communicate between
Panel C's compact-subspace decoder result and Panel E's measured behavior?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont

try:
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px


REPO_ROOT = Path(__file__).resolve().parents[3]
ATLAS = REPO_ROOT / "declan" / "figure4_active_sensing_atlas"
OUT_DIR = ATLAS / "figures" / "panel_D" / "story_options"
BASE = REPO_ROOT / "outputs" / "fixation_statistics_by_stimulus_all_sessions_after_review"
WINDOWS_CSV = BASE / "backimage_image_structure_reviewed_v2_screenfiltered_yfix" / "backimage_image_fem_windows.csv"
STABILITY_DIR = BASE / "backimage_edge_parallel_stability_screen_yfix_n256_pop256"
AUDIT_DIR = BASE / "backimage_twin_stability_metric_audit"
OBJECTIVE_DIR = BASE / "backimage_conditional_fixation_objectives_twin_axis_only_n256"

INK = "#20262c"
MUTED = "#68727d"
GRID = "#dfe4e9"
BLUE = "#244f7a"
GREEN = "#2f8f6a"
PURPLE = "#8064a2"
ORANGE = "#d07a22"
GRAY = "#747a80"
LIGHT = "#eef2f4"


@dataclass(frozen=True)
class StoryOption:
    slug: str
    title: str
    file_name: str
    read: str
    concern: str


OPTIONS = (
    StoryOption(
        slug="D1",
        title="Model-response preservation",
        file_name="4D_story_option_1_axes_plus_preservation.png",
        read="Best candidate: first shows the local edge-defined motion axes, then shows absolute model-response disruption for along-edge versus across-edge shifts.",
        concern="Pixel preservation is omitted from the main panel and should remain a sanity-check sentence or supplement.",
    ),
    StoryOption(
        slug="D2",
        title="Model costs, not contrasts",
        file_name="4D_story_option_2_along_across_costs.png",
        read="Makes the comparison concrete by showing the absolute along-edge and across-edge model-response costs directly.",
        concern="Nearly all the story sits in the bar chart; needs the thumbnail to keep the direction comparison intuitive.",
    ),
    StoryOption(
        slug="D3",
        title="Metric robustness",
        file_name="4D_story_option_3_metric_robustness.png",
        read="Answers the audit question: the preservation sign is positive across response-normalized and whitened model metrics.",
        concern="Too diagnostic for the main figure unless reviewers are already worried about metric choice.",
    ),
    StoryOption(
        slug="D4",
        title="Bridge to behavior",
        file_name="4D_story_option_4_axes_to_behavior_bridge.png",
        read="Frames D as the hinge between mechanism and behavior: image axes define a low-disruption model-response direction, then real drift follows clear edge axes.",
        concern="Adds density and overlaps Panel E; useful as a story sketch, not necessarily as the final main panel.",
    ),
    StoryOption(
        slug="D5",
        title="Policy guardrail",
        file_name="4D_story_option_5_objective_guardrail.png",
        read="Keeps the story honest by showing response-objective axes do not yet beat raw edge geometry.",
        concern="Negative/control-forward; it would interrupt the main figure's positive story.",
    ),
    StoryOption(
        slug="D6",
        title="Minimal mechanism",
        file_name="4D_story_option_6_minimal_mechanism.png",
        read="Very clear conceptual grammar for a narrow slot: local edge -> along/across matched shifts -> lower disruption along the edge.",
        concern="Less data-rich; probably better as an inset style than as the whole panel.",
    ),
)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _norm_image(image: np.ndarray, *, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    lo, hi = np.nanpercentile(arr, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _synthetic_edge_patch(size: int = 180) -> np.ndarray:
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    edge = 0.45 + 0.28 * np.tanh(7.5 * (0.55 * x + 0.83 * y + 0.02))
    texture = 0.06 * np.sin(18.0 * (0.86 * x - 0.33 * y))
    texture += 0.04 * np.sin(26.0 * (0.12 * x + 0.98 * y))
    return np.clip(edge + texture, 0.0, 1.0)


def _crop_centered(image: np.ndarray, center_xy: tuple[float, float], size: int) -> np.ndarray:
    height, width = image.shape[:2]
    cx, cy = center_xy
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x0 = max(0, min(width - size, x0))
    y0 = max(0, min(height - size, y0))
    return image[y0 : y0 + size, x0 : x0 + size]


def _stability_source_windows() -> pd.DataFrame:
    metadata = json.loads((STABILITY_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    cfg = metadata["config"]
    df = pd.read_csv(REPO_ROOT / cfg["input"])
    duration = df["duration_s"] if "duration_s" in df.columns else df.get("epoch_duration_s", np.nan)
    keep = (
        np.isfinite(df["drift_orientation_deg"].astype(float))
        & np.isfinite(df["image_edge_axis_deg"].astype(float))
        & (df["anisotropy"].astype(float) >= float(cfg["reliable_drift_anisotropy_min"]))
        & (df["image_orientation_coherence"].astype(float) >= float(cfg["reliable_image_coherence_min"]))
        & (duration.astype(float) >= float(cfg["min_duration_s"]))
        & (df["image_patch_distance_to_image_border_px"].astype(float) >= float(cfg["min_patch_image_margin_px"]))
    )
    work = df.loc[keep].copy()
    work["window_id"] = np.arange(work.shape[0], dtype=int)
    max_windows = int(cfg["max_windows"])
    if max_windows > 0 and work.shape[0] > max_windows:
        work = work.sample(n=max_windows, replace=False, random_state=int(cfg["seed"])).sort_values(
            ["session", "trial_idx", "window_id"]
        )
    return work.reset_index(drop=True)


def _load_example_patch() -> tuple[np.ndarray, float, dict[str, object]]:
    source_windows = _stability_source_windows()
    stability = pd.read_csv(STABILITY_DIR / "edge_parallel_stability_by_window.csv")
    preferred_window_rows = [17, 34, 26, 29, 60]
    stability = stability[
        (stability["twin_stability_advantage"].astype(float) > 0)
        & (stability["image_orientation_coherence"].astype(float) > 0.45)
    ].copy()
    if stability.empty:
        return _synthetic_edge_patch(), -32.0, {"source": "synthetic"}
    visual_rows = stability[stability["window_row"].isin(preferred_window_rows)].copy()
    if not visual_rows.empty:
        visual_rows["visual_order"] = visual_rows["window_row"].map(
            {window_row: idx for idx, window_row in enumerate(preferred_window_rows)}
        )
        row = visual_rows.sort_values("visual_order").iloc[0]
        selection_note = "preferred corrected source-row exemplar"
    else:
        stability["score"] = (
            stability["image_orientation_coherence"].astype(float)
            * np.sqrt(np.maximum(stability["twin_stability_advantage"].astype(float), 0.0))
        )
        row = stability.sort_values("score", ascending=False).iloc[0]
        selection_note = "fallback high-coherence positive-model-preservation score"
    win = source_windows.iloc[int(row["window_row"])]
    try:
        canvas, ppd, screen_shape = _backimage_canvas(str(win["session"]), int(win["trial_idx"]))
        center_xy_px = gaze_deg_to_screen_px(
            np.asarray([float(win["mean_x_deg"]), float(win["mean_y_deg"])]),
            ppd=float(ppd),
            screen_shape=screen_shape,
        )
        center = (float(center_xy_px[0]), float(center_xy_px[1]))
        patch = _crop_centered(canvas, center, 190)
        axis = float(row["edge_axis_deg"])
        meta = {
            "source": "real_backimage",
            "session": str(win["session"]),
            "trial_idx": int(win["trial_idx"]),
            "window_row": int(row["window_row"]),
            "source_window_id": int(win["window_id"]),
            "selection_note": selection_note,
            "plot_axis_screen_deg": float(axis),
            "stability_edge_axis_gaze_deg": float(row["edge_axis_deg"]),
            "image_orientation_coherence": float(row["image_orientation_coherence"]),
            "pixel_stability_advantage": float(row["pixel_stability_advantage"]),
            "twin_stability_advantage": float(row["twin_stability_advantage"]),
        }
        return patch, axis, meta
    except Exception as exc:  # pragma: no cover - data availability fallback
        return _synthetic_edge_patch(), -32.0, {"source": "synthetic", "error": str(exc)}


def _axis_vectors(axis_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(axis_deg)
    along = np.asarray([np.cos(theta), np.sin(theta)], dtype=float)
    across = np.asarray([np.cos(theta + np.pi / 2.0), np.sin(theta + np.pi / 2.0)], dtype=float)
    return along, across


def _add_axis_arrows(ax: plt.Axes, axis_deg: float, *, label: bool = True, scale: float = 0.32) -> None:
    along, across = _axis_vectors(axis_deg)
    center = np.asarray([0.50, 0.50])

    def arrow(vec: np.ndarray, color: str) -> None:
        start = center - vec * scale
        end = center + vec * scale
        ax.add_patch(
            FancyArrowPatch(
                tuple(start),
                tuple(end),
                arrowstyle="<|-|>",
                mutation_scale=11,
                linewidth=2.1,
                color=color,
                transform=ax.transAxes,
            )
        )

    arrow(along, GREEN)
    arrow(across, PURPLE)
    if label:
        ax.text(0.58, 0.18, "along edge", color=GREEN, fontsize=8.0, fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.74, "across edge", color=PURPLE, fontsize=8.0, fontweight="bold", transform=ax.transAxes)


def _plot_patch(ax: plt.Axes, patch: np.ndarray, axis_deg: float, *, title: str = "matched directions") -> None:
    ax.imshow(_norm_image(patch), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    _add_axis_arrows(ax, axis_deg)
    ax.set_title(title, pad=5)


def _stability_summary() -> pd.DataFrame:
    rows = pd.read_csv(STABILITY_DIR / "stability_summary.csv").copy()
    rows["screen_label"] = rows["screen"].map({"pixel": "image", "twin": "model"})
    rows["plot_mean"] = rows["mean_advantage_session_mean"].astype(float)
    rows["plot_low"] = rows["ci95_low_session_mean"].astype(float)
    rows["plot_high"] = rows["ci95_high_session_mean"].astype(float)
    rows.loc[rows["screen"].eq("twin"), ["plot_mean", "plot_low", "plot_high"]] *= 1e4
    rows["plot_units"] = np.where(rows["screen"].eq("twin"), "model x1e4", "pixels")
    return rows


def _plot_model_cost(ax: plt.Axes, *, title: str = "model responses") -> None:
    rows = pd.read_csv(STABILITY_DIR / "stability_summary.csv").copy()
    row = rows[rows["screen"].eq("twin")].iloc[0]
    vals = np.asarray(
        [float(row["mean_parallel_cost_window"]), float(row["mean_orthogonal_cost_window"])],
        dtype=float,
    ) * 1e4
    ax.bar([0, 1], vals, color=[GREEN, PURPLE], width=0.62)
    ax.set_xticks([0, 1], ["along\nedge", "across\nedge"])
    ax.set_title(title, pad=4)
    ax.set_ylabel("response change from baseline\n(mean squared, x1e4)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    _clean_axis(ax)
    for idx, value in enumerate(vals):
        ax.text(
            idx,
            value * 1.02,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=INK,
        )
    ax.set_ylim(0.0, float(vals.max()) * 1.28)
    ax.text(
        0.50,
        0.94,
        f"across > along in {int(row['n_sessions_positive_advantage'])}/{int(row['n_sessions'])} sessions",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color=INK,
    )


def _metric_robustness_rows() -> pd.DataFrame:
    rows = pd.read_csv(AUDIT_DIR / "cheap_synthesis" / "first_order_signed_stability_advantage_session_ci.csv")
    keep = [
        "raw_mse",
        "response_norm_mse",
        "per_rate_mse",
        "diag_whitened_mse",
        "full_cov_whitened_mse",
        "top_modulated_units_raw_mse",
        "other_units_raw_mse",
    ]
    labels = {
        "raw_mse": "raw model",
        "response_norm_mse": "response-normalized",
        "per_rate_mse": "per-rate model",
        "diag_whitened_mse": "diag whitened",
        "full_cov_whitened_mse": "full-cov whitened",
        "top_modulated_units_raw_mse": "top units",
        "other_units_raw_mse": "other units",
    }
    block = rows[rows["metric"].isin(keep)].copy()
    block["order"] = block["metric"].map({name: idx for idx, name in enumerate(keep)})
    block = block.sort_values("order")
    block["label"] = block["metric"].map(labels)
    half = (block["ci_high"].astype(float) - block["ci_low"].astype(float)) / 2.0
    se = half / 1.96
    block["effect_over_se"] = block["mean_session"].astype(float) / se.replace(0.0, np.nan)
    block["low_over_se"] = block["ci_low"].astype(float) / se.replace(0.0, np.nan)
    block["high_over_se"] = block["ci_high"].astype(float) / se.replace(0.0, np.nan)
    return block


def _plot_metric_robustness(ax: plt.Axes) -> None:
    block = _metric_robustness_rows()
    y = np.arange(block.shape[0])
    ax.errorbar(
        block["effect_over_se"],
        y,
        xerr=np.vstack([block["effect_over_se"] - block["low_over_se"], block["high_over_se"] - block["effect_over_se"]]),
        color=BLUE,
        marker="o",
        markersize=3.7,
        lw=1.0,
        capsize=0,
        linestyle="none",
    )
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, block["label"])
    ax.invert_yaxis()
    ax.set_xlabel("signed preservation / SE")
    ax.set_title("sign survives metric choice", pad=4)
    ax.grid(axis="x", color=GRID, lw=0.8)
    _clean_axis(ax)


def _objective_rows() -> pd.DataFrame:
    path = OBJECTIVE_DIR / "conditional_residual_summary" / "key_paired_deltas_vs_raw_edge.csv"
    rows = pd.read_csv(path)
    order = [
        "optimized_response_stability",
        "optimized_response_refresh_lambda_0.25",
        "optimized_PA",
        "optimized_PB",
        "optimized_pixel_isophote",
    ]
    labels = {
        "optimized_response_stability": "response\nstability",
        "optimized_response_refresh_lambda_0.25": "response\nrefresh",
        "optimized_PA": "pose-aware\nresponse",
        "optimized_PB": "pose-blind\nresponse",
        "optimized_pixel_isophote": "pixel\nisophote",
    }
    block = rows[rows["objective"].isin(order)].copy()
    block["order"] = block["objective"].map({name: idx for idx, name in enumerate(order)})
    block = block.sort_values("order")
    block["label"] = block["objective"].map(labels)
    return block


def _plot_objective_guardrail(ax: plt.Axes) -> None:
    block = _objective_rows()
    x = np.arange(block.shape[0])
    y = block["mean_delta_cos2_session"].astype(float).to_numpy()
    lo = block["ci95_low"].astype(float).to_numpy()
    hi = block["ci95_high"].astype(float).to_numpy()
    colors = [PURPLE, PURPLE, PURPLE, PURPLE, ORANGE]
    ax.bar(x, y, color=colors, width=0.65)
    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), color=INK, lw=1.0, capsize=0, linestyle="none")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, block["label"], rotation=22, ha="right")
    ax.set_ylabel("alignment vs raw edge")
    ax.set_title("model objective remains a guardrail", pad=4)
    ax.grid(axis="y", color=GRID, lw=0.8)
    _clean_axis(ax)


def _behavior_bins() -> pd.DataFrame:
    windows = pd.read_csv(WINDOWS_CSV)
    work = windows[windows["image_feature_ok"].astype(bool)].copy()
    work = work[np.isfinite(work["image_orientation_coherence"]) & np.isfinite(work["drift_edge_cos2"])]
    bins = np.linspace(0.0, 1.0, 7)
    work["bin"] = pd.cut(work["image_orientation_coherence"], bins=bins, include_lowest=True)
    rows = []
    for interval, block in work.groupby("bin", observed=True):
        if block.empty:
            continue
        rows.append(
            {
                "center": float((interval.left + interval.right) / 2.0),
                "mean": float(block["drift_edge_cos2"].mean()),
                "n": int(block.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _plot_behavior_mini(ax: plt.Axes) -> None:
    bins = _behavior_bins()
    ax.plot(bins["center"], bins["mean"], color=BLUE, marker="o", lw=1.7, markersize=3.5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("edge coherence")
    ax.set_ylabel("drift alignment")
    ax.set_title("real drift follows clear edges", pad=4)
    ax.set_ylim(-0.06, 0.36)
    ax.grid(axis="y", color=GRID, lw=0.8)
    _clean_axis(ax)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _option_axes_plus_preservation(patch: np.ndarray, axis_deg: float, path: Path) -> None:
    fig = plt.figure(figsize=(5.7, 2.35), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.08, 1.22])
    ax_patch = fig.add_subplot(gs[0, 0])
    ax_model = fig.add_subplot(gs[0, 1])
    _plot_patch(ax_patch, patch, axis_deg, title="local edge defines matched shifts")
    _plot_model_cost(ax_model, title="across-edge shifts change model more")
    fig.suptitle("Along-edge motion preserves model responses", x=0.02, ha="left", fontsize=10.5, fontweight="bold")
    _save(fig, path)


def _option_costs(patch: np.ndarray, axis_deg: float, path: Path) -> None:
    fig = plt.figure(figsize=(5.5, 2.45), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.92, 1.25])
    ax_patch = fig.add_subplot(gs[0, 0])
    ax_model = fig.add_subplot(gs[0, 1])
    _plot_patch(ax_patch, patch, axis_deg, title="same displacement")
    _plot_model_cost(ax_model, title="absolute response change")
    fig.suptitle("Matched across-edge shifts change model responses more", x=0.02, ha="left", fontsize=10.5, fontweight="bold")
    _save(fig, path)


def _option_metric_robustness(patch: np.ndarray, axis_deg: float, path: Path) -> None:
    fig = plt.figure(figsize=(5.7, 2.55), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.82, 1.55])
    ax_patch = fig.add_subplot(gs[0, 0])
    ax_metrics = fig.add_subplot(gs[0, 1])
    _plot_patch(ax_patch, patch, axis_deg, title="tested direction pair")
    _plot_metric_robustness(ax_metrics)
    fig.suptitle("Along-edge preservation is not a metric artifact", x=0.02, ha="left", fontsize=10.5, fontweight="bold")
    _save(fig, path)


def _option_bridge(patch: np.ndarray, axis_deg: float, path: Path) -> None:
    fig = plt.figure(figsize=(6.0, 2.55), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.90, 1.10, 1.0])
    ax_patch = fig.add_subplot(gs[0, 0])
    ax_model = fig.add_subplot(gs[0, 1])
    ax_beh = fig.add_subplot(gs[0, 2])
    _plot_patch(ax_patch, patch, axis_deg, title="image axes")
    _plot_model_cost(ax_model, title="model response cost")
    _plot_behavior_mini(ax_beh)
    fig.suptitle("Local image axes connect model utility to measured drift", x=0.02, ha="left", fontsize=10.5, fontweight="bold")
    _save(fig, path)


def _option_guardrail(path: Path) -> None:
    fig = plt.figure(figsize=(5.5, 2.55), constrained_layout=True)
    ax = fig.add_subplot(111)
    _plot_objective_guardrail(ax)
    fig.suptitle("Do not claim the tested model objective is the policy", x=0.02, ha="left", fontsize=10.5, fontweight="bold")
    _save(fig, path)


def _option_minimal(patch: np.ndarray, axis_deg: float, path: Path) -> None:
    fig = plt.figure(figsize=(5.2, 2.20), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 0.25, 1.2])
    ax_patch = fig.add_subplot(gs[0, 0])
    ax_arrow = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[0, 2])
    _plot_patch(ax_patch, patch, axis_deg, title="local edge axis")
    ax_arrow.axis("off")
    ax_arrow.text(0.5, 0.50, "->", ha="center", va="center", fontsize=18, color=MUTED)
    ax_text.axis("off")
    lines = [
        ("along edge", "less image/model change", GREEN),
        ("across edge", "more image/model change", PURPLE),
    ]
    for i, (label, text, color) in enumerate(lines):
        y = 0.68 - i * 0.34
        ax_text.plot([0.02, 0.18], [y, y], color=color, lw=3, transform=ax_text.transAxes)
        ax_text.text(0.22, y + 0.04, label, transform=ax_text.transAxes, color=color, fontweight="bold", fontsize=9.0)
        ax_text.text(0.22, y - 0.08, text, transform=ax_text.transAxes, color=INK, fontsize=8.0)
    ax_text.text(
        0.02,
        0.05,
        "Panel claim: the image supplies\nmotion axes with different costs.",
        transform=ax_text.transAxes,
        color=MUTED,
        fontsize=7.7,
    )
    fig.suptitle("Image-defined motion axes have different consequences", x=0.02, ha="left", fontsize=10.5, fontweight="bold")
    _save(fig, path)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _contain(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    scale = min(box[0] / image.width, box[1] / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resample = getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", 3))
    return image.resize(size, resample)


def _wrap(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, max_width: int, *, fill: tuple[int, int, int]) -> int:
    x, y = xy
    line = ""
    line_h = int(getattr(font, "size", 18) * 1.22)
    for word in text.split():
        candidate = word if not line else f"{line} {word}"
        width = font.getbbox(candidate)[2] - font.getbbox(candidate)[0]
        if width <= max_width:
            line = candidate
        else:
            if line:
                draw.text((x, y), line, font=font, fill=fill)
                y += line_h
            line = word
    if line:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h
    return y


def _make_sheet(paths: dict[str, Path], out_path: Path) -> None:
    width, height = 2600, 2180
    margin, gap = 62, 38
    thumb_w, thumb_h = 790, 410
    sheet = Image.new("RGB", (width, height), (250, 251, 252))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 44), "Figure 4D Story Options", font=_font(48, True), fill=(20, 26, 32))
    draw.text(
        (margin, 112),
        "Question: how should this panel explain the bridge from compact-subspace decoding to contour-following behavior?",
        font=_font(25),
        fill=(73, 80, 88),
    )
    draw.line((margin, 170, width - margin, 170), fill=(183, 190, 198), width=2)
    draw.text((margin, 202), "Working main story", font=_font(25, True), fill=(20, 26, 32))
    story = (
        "Local image edges define matched along-edge and across-edge motion directions. "
        "At the tested displacement, moving along the edge preserves V1-twin responses better. "
        "That makes image geometry a plausible useful coordinate system, without proving the animal optimizes the tested model objective."
    )
    _wrap(draw, (margin, 238), story, _font(22), width - 2 * margin, fill=(45, 49, 54))

    start_y = 360
    for i, option in enumerate(OPTIONS):
        row = i // 3
        col = i % 3
        x = margin + col * (thumb_w + gap)
        y = start_y + row * 860
        draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline=(191, 199, 207), fill="white")
        image = Image.open(paths[option.slug]).convert("RGBA")
        image = _contain(image, (thumb_w - 28, thumb_h - 28))
        sheet.paste(image, (x + 14 + (thumb_w - 28 - image.width) // 2, y + 14), image)
        draw.text((x, y + thumb_h + 24), f"{option.slug}. {option.title}", font=_font(25, True), fill=(20, 26, 32))
        y2 = _wrap(draw, (x, y + thumb_h + 64), option.read, _font(20), thumb_w, fill=(45, 49, 54))
        _wrap(draw, (x, y2 + 12), f"Concern: {option.concern}", _font(18), thumb_w, fill=(92, 100, 108))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, optimize=True)


def _write_readme(paths: dict[str, Path], metadata: dict[str, object]) -> None:
    lines = [
        "# Figure 4D Story Options",
        "",
        "Status: exploratory option sheet for revising Panel D.",
        "",
        "![Story option sheet](/home/declan/VisionCore/declan/figure4_active_sensing_atlas/figures/panel_D/story_options/4D_story_option_sheet.png)",
        "",
        "## Main Story",
        "",
        "Panel D should explain why local image geometry is a useful coordinate system for the active-sensing story. The clean main-figure claim is: local edges define matched along-edge and across-edge directions, and along-edge shifts preserve V1-twin responses better at the tested displacement. Pixel preservation remains a sanity check, but the promoted panel should use model responses because that is the link to Panels B/C.",
        "",
        "## Current Recommendation",
        "",
        "Option D1 is the best starting point for the main composite. It makes the along/across comparison visually legible before showing the absolute model-response disruption costs. D3 is useful if we need to foreground robustness to model-response metric choice. D5 should remain a guardrail or supplement unless the figure needs to emphasize what we are not claiming.",
        "",
        "## Axis Estimator Caveat",
        "",
        "The local image axis may depend on the estimator. A patch-level average orientation-energy estimate can differ from a prominent orientation feature that a winner-take-all readout might select. The row-17/18 rail crop is the current reference example: raw BackImage rows 17/18 from `Allen_2022-02-16`, trial `184`, have stored aggregate `image_edge_axis_deg = -31.4 deg`, while a visible bright-rail fit gives `-37.6 deg`.",
        "",
        "![Row-17/18 visible rail fit](/home/declan/VisionCore/declan/figure4_active_sensing_atlas/figures/panel_D/story_options/4D_row17_row18_visible_rail_fit_orientation.png)",
        "",
        "Use this as a provenance example for the open axis-estimator question. It is not a correction to the quantitative Panel D readout.",
        "",
        "## Real-Patch Provenance",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Files", ""])
    for option in OPTIONS:
        lines.append(f"- `{option.file_name}`")
        lines.append(f"- `{Path(option.file_name).with_suffix('.pdf').name}`")
    lines.extend(
        [
            "- `4D_story_option_sheet.png`",
            "- `4D_story_option_values.csv`",
            "- `4D_row17_row18_visible_rail_fit_orientation.png`",
            "- `4D_row17_row18_visible_rail_fit_orientation_values.csv`",
            "",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build() -> list[Path]:
    _configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patch, axis_deg, metadata = _load_example_patch()

    paths: dict[str, Path] = {}
    for option in OPTIONS:
        paths[option.slug] = OUT_DIR / option.file_name

    _option_axes_plus_preservation(patch, axis_deg, paths["D1"])
    _option_costs(patch, axis_deg, paths["D2"])
    _option_metric_robustness(patch, axis_deg, paths["D3"])
    _option_bridge(patch, axis_deg, paths["D4"])
    _option_guardrail(paths["D5"])
    _option_minimal(patch, axis_deg, paths["D6"])
    _make_sheet(paths, OUT_DIR / "4D_story_option_sheet.png")

    values = []
    for option in OPTIONS:
        values.append(
            {
                "slug": option.slug,
                "title": option.title,
                "file": option.file_name,
                "read": option.read,
                "concern": option.concern,
            }
        )
    pd.DataFrame(values).to_csv(OUT_DIR / "4D_story_option_values.csv", index=False)
    _write_readme(paths, metadata)

    out = [paths[option.slug] for option in OPTIONS]
    out.extend([OUT_DIR / "4D_story_option_sheet.png", OUT_DIR / "4D_story_option_values.csv", OUT_DIR / "README.md"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for path in build():
        print(path)


if __name__ == "__main__":
    main()
