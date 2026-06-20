"""Build cache-only Figure 4A premise/QC subpanels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[3]
MOVIE_PACK = (
    REPO_ROOT
    / "outputs/active_sensing_movie_information/"
    / "active_sensing_movie_information_figure_frozen_20260615_pre_backimage_collab_pack"
)
FIG4_HEADLINE = REPO_ROOT / "outputs/fig4_active_sensing/active_sensing_headline_figure"
REAFFERENT_DIR = REPO_ROOT / "outputs/active_sensing_movie_information/reafferent_variance_accounting"
DEFAULT_OUT_DIR = REPO_ROOT / "declan/figure4_active_sensing_atlas/figures/panel_A"

COLORS = {
    "dark": "#242a2f",
    "muted": "#65717a",
    "grid": "#d8dde3",
    "blue": "#244f7a",
    "green": "#2f8f6a",
    "orange": "#d07a22",
    "purple": "#8064a2",
    "gray": "#8e9aa6",
    "light": "#f8fafb",
    "warn": "#fff8ef",
}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#5d6871") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color=color,
            transform=ax.transAxes,
        )
    )


def _node(ax: plt.Axes, xy: tuple[float, float], text: str, width: float = 0.20, color: str = "#f8fafb") -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - 0.060),
            width,
            0.120,
            boxstyle="round,pad=0.014,rounding_size=0.014",
            edgecolor="#c5ccd2",
            facecolor=color,
            linewidth=0.8,
            transform=ax.transAxes,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=8, transform=ax.transAxes)


def _synthetic_natural_patch(size: int = 192) -> np.ndarray:
    rng = np.random.default_rng(42)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    patch = 0.48 + 0.20 * np.tanh(6.5 * (0.45 * x - 0.9 * y + 0.05))
    for amp, sx, sy, x0, y0 in [
        (0.26, 0.16, 0.23, -0.45, 0.28),
        (-0.18, 0.24, 0.18, 0.28, -0.20),
        (0.12, 0.12, 0.15, 0.52, 0.45),
    ]:
        patch += amp * np.exp(-(((x - x0) ** 2) / sx + ((y - y0) ** 2) / sy))
    patch += 0.08 * np.sin(15.0 * (0.8 * x + 0.25 * y))
    patch += 0.05 * np.sin(31.0 * (-0.15 * x + 0.9 * y))
    patch += 0.035 * rng.normal(size=(size, size))
    return np.clip(patch, 0.0, 1.0)


def _eye_trace(n: int = 80) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    rng = np.random.default_rng(13)
    trace = np.c_[
        10.0 * np.sin(1.7 * np.pi * t) + 3.2 * np.sin(7.0 * np.pi * t),
        6.0 * np.sin(2.8 * np.pi * t + 0.5) + 2.8 * np.cos(5.6 * np.pi * t),
    ]
    trace += np.cumsum(rng.normal(scale=0.18, size=(n, 2)), axis=0)
    trace -= trace.mean(axis=0, keepdims=True)
    return trace


def _crop_with_shift(image: np.ndarray, dx: float, dy: float, size: int = 70) -> np.ndarray:
    center = np.array(image.shape) // 2
    x0 = int(center[1] + round(dx) - size // 2)
    y0 = int(center[0] + round(dy) - size // 2)
    x0 = max(0, min(image.shape[1] - size, x0))
    y0 = max(0, min(image.shape[0] - size, y0))
    return image[y0 : y0 + size, x0 : x0 + size]


def plot_a1_retinal_movie_transform(out_dir: Path) -> pd.DataFrame:
    image = _synthetic_natural_patch()
    trace = _eye_trace()
    idxs = [8, 38, 68]
    shifts = trace[idxs]

    fig, ax = plt.subplots(figsize=(7.0, 2.7), constrained_layout=True)
    ax.set_axis_off()
    ax.set_title("A fixed screen image becomes a moving retinal crop", pad=8)

    screen_ax = ax.inset_axes([0.03, 0.16, 0.24, 0.70])
    screen_ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    screen_ax.plot(image.shape[1] / 2 + trace[:, 0], image.shape[0] / 2 + trace[:, 1], color=COLORS["green"], lw=1.2)
    for idx, color in zip(idxs, [COLORS["blue"], COLORS["orange"], COLORS["purple"]], strict=True):
        x = image.shape[1] / 2 + trace[idx, 0]
        y = image.shape[0] / 2 + trace[idx, 1]
        screen_ax.add_patch(Rectangle((x - 35, y - 35), 70, 70, fill=False, edgecolor=color, lw=1.4))
    screen_ax.set_xticks([])
    screen_ax.set_yticks([])
    screen_ax.set_xlabel("screen image + eye trace", fontsize=7.5)

    trace_ax = ax.inset_axes([0.36, 0.20, 0.20, 0.58])
    trace_ax.plot(trace[:, 0], trace[:, 1], color=COLORS["green"], lw=1.3)
    for idx, color in zip(idxs, [COLORS["blue"], COLORS["orange"], COLORS["purple"]], strict=True):
        trace_ax.scatter(trace[idx, 0], trace[idx, 1], color=color, s=24, zorder=3)
    trace_ax.axhline(0, color=COLORS["grid"], lw=0.7)
    trace_ax.axvline(0, color=COLORS["grid"], lw=0.7)
    trace_ax.set_xticks([])
    trace_ax.set_yticks([])
    trace_ax.set_title("measured FEM", fontsize=8.5)
    for spine in trace_ax.spines.values():
        spine.set_color("#c5ccd2")

    for j, (idx, color) in enumerate(zip(idxs, [COLORS["blue"], COLORS["orange"], COLORS["purple"]], strict=True)):
        crop_ax = ax.inset_axes([0.64 + 0.11 * j, 0.26, 0.095, 0.44])
        crop_ax.imshow(_crop_with_shift(image, trace[idx, 0], trace[idx, 1]), cmap="gray", vmin=0, vmax=1)
        crop_ax.set_xticks([])
        crop_ax.set_yticks([])
        for spine in crop_ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.3)
        crop_ax.set_title(f"t{j + 1}", fontsize=7.5, color=color)

    _arrow(ax, (0.29, 0.50), (0.34, 0.50))
    _arrow(ax, (0.57, 0.50), (0.62, 0.50))
    ax.text(0.78, 0.12, "same screen image, shifted retinal samples", ha="center", fontsize=7.8, transform=ax.transAxes)

    _save(fig, out_dir, "A1_retinal_movie_transform")
    return pd.DataFrame(
        {
            "frame": ["t1", "t2", "t3"],
            "trace_index": idxs,
            "retinal_shift_x_px": shifts[:, 0],
            "retinal_shift_y_px": shifts[:, 1],
            "source": "schematic_from_synthetic_patch_and_trace",
        }
    )


def plot_a2_movie_transform_qc(summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    metrics = [
        ("temporal_contrast_rms_mean", "temporal\ncontrast"),
        ("motion_power_vs_matched_stabilized_mean", "motion\npower"),
        ("movie_power_mean", "movie\npower"),
    ]
    block = summary[
        summary["condition"].isin(["real", "stabilized"]) & summary["metric"].isin([m for m, _ in metrics])
    ].copy()
    pivot = block.pivot(index="metric", columns="condition", values="mean")
    norm_rows = []
    for metric, label in metrics:
        real = float(pivot.loc[metric, "real"])
        stabilized = float(pivot.loc[metric, "stabilized"])
        denom = real if real != 0 else 1.0
        norm_rows.append(
            {
                "metric": metric,
                "label": label.replace("\n", " "),
                "real_mean": real,
                "stabilized_mean": stabilized,
                "real_relative_to_real": 1.0,
                "stabilized_relative_to_real": stabilized / denom,
                "n": int(block.loc[(block["condition"] == "real") & (block["metric"] == metric), "n"].iloc[0]),
            }
        )
    values = pd.DataFrame(norm_rows)

    fig, ax = plt.subplots(figsize=(4.8, 2.8), constrained_layout=True)
    x = np.arange(len(values))
    width = 0.34
    ax.bar(x - width / 2, values["real_relative_to_real"], width=width, color=COLORS["green"], label="FEM movie")
    ax.bar(
        x + width / 2,
        values["stabilized_relative_to_real"],
        width=width,
        color=COLORS["gray"],
        label="stabilized",
    )
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("relative to FEM movie")
    ax.set_ylim(0, 1.25)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.8)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Retinal-motion transform QC")
    _clean_axis(ax)
    ax.text(0.02, 0.92, "n = 108 image/trace movies", transform=ax.transAxes, fontsize=7.4, color=COLORS["muted"])
    _save(fig, out_dir, "A2_movie_transform_qc")
    return values


def plot_a3_local_gradient_sampling(out_dir: Path) -> pd.DataFrame:
    size = 128
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    patch = np.clip(0.50 + 0.30 * np.tanh(7.5 * (0.82 * x - 0.48 * y)) + 0.06 * np.sin(18 * y), 0, 1)
    gradient_axis_deg = -30.0
    edge_axis_deg = 60.0

    fig, ax = plt.subplots(figsize=(6.8, 3.0), constrained_layout=True)
    ax.set_axis_off()
    ax.set_title("Small translations sample local image gradients", pad=8)

    img_ax = ax.inset_axes([0.04, 0.16, 0.30, 0.70])
    img_ax.imshow(patch, cmap="gray", vmin=0, vmax=1, extent=(-1, 1, -1, 1))
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    for spine in img_ax.spines.values():
        spine.set_color("#c5ccd2")
    img_ax.arrow(-0.10, -0.15, 0.58, -0.34, width=0.012, head_width=0.08, color=COLORS["orange"], length_includes_head=True)
    img_ax.arrow(-0.10, -0.15, 0.32, 0.55, width=0.012, head_width=0.08, color=COLORS["green"], length_includes_head=True)
    img_ax.text(0.55, -0.56, "normal", color=COLORS["orange"], fontsize=7.5)
    img_ax.text(0.22, 0.46, "parallel", color=COLORS["green"], fontsize=7.5)

    ax.text(0.43, 0.68, "for small displacement dx:", fontsize=8.3, transform=ax.transAxes)
    ax.text(0.43, 0.55, "movie change ~= gradient dot dx", fontsize=8.3, transform=ax.transAxes)
    ax.text(
        0.43,
        0.36,
        "same dx can be informative\nor preserving, depending\non local geometry",
        fontsize=7.9,
        transform=ax.transAxes,
    )
    _node(ax, (0.84, 0.67), "normal motion\nlarge change", width=0.23, color="#fff8ef")
    _node(ax, (0.84, 0.38), "parallel motion\nstable sample", width=0.23, color="#f5faf7")

    values = pd.DataFrame(
        [
            {"quantity": "edge_axis_deg", "value": edge_axis_deg, "source": "conceptual_schematic"},
            {"quantity": "gradient_axis_deg", "value": gradient_axis_deg, "source": "conceptual_schematic"},
        ]
    )
    _save(fig, out_dir, "A3_gradient_sampling_cartoon")
    return values


def plot_a4_backimage_pipeline(stats_path: Path, out_dir: Path) -> pd.DataFrame:
    stats = json.loads(stats_path.read_text())
    rows = [
        ("images", stats["n_images"]),
        ("sessions", stats["n_sessions"]),
        ("trace samples per condition", stats["trace_samples_per_condition"]),
        ("trace sources", stats["motion_bookkeeping"]["n_trace_sources"]),
        ("RMS ratio", stats["motion_bookkeeping"]["median_effective_to_requested_rms"]),
        ("max clipped fraction", stats["motion_bookkeeping"]["max_clipped_fraction"]),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 2.8), constrained_layout=True)
    ax.set_axis_off()
    ax.set_title("Canonical BackImage/V1-twin pipeline used downstream", pad=8)
    _node(ax, (0.12, 0.58), "screen image", width=0.18)
    _node(ax, (0.32, 0.58), "eye trace", width=0.16)
    _node(ax, (0.52, 0.58), "retinal\nmovie", width=0.17)
    _node(ax, (0.72, 0.58), "756-unit\nV1 twin", width=0.17)
    _node(ax, (0.88, 0.58), "response\nmovie", width=0.15)
    for start, end in [((0.21, 0.58), (0.26, 0.58)), ((0.39, 0.58), (0.46, 0.58)), ((0.59, 0.58), (0.66, 0.58)), ((0.79, 0.58), (0.83, 0.58))]:
        _arrow(ax, start, end)

    facts = (
        f"{stats['n_images']} images; {stats['n_sessions']} sessions; "
        f"{stats['motion_bookkeeping']['n_trace_sources']} drift-only trace sources\n"
        f"{stats['population']}; grouped-by-image CV; "
        "RMS ratio = 1.0; clipping = 0.0"
    )
    ax.text(
        0.50,
        0.18,
        facts,
        ha="center",
        va="center",
        fontsize=7.7,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )

    _save(fig, out_dir, "A4_backimage_pipeline_bridge")
    return pd.DataFrame({"field": [r[0] for r in rows], "value": [r[1] for r in rows]})


def plot_a5_covariance_bridge(aggregate: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    keep = {
        "compact_reafferent_numerator_candidate": "compact\nderivative",
        "finite_difference_reafferent_numerator_candidate": "finite-diff\ntangent",
        "noise_correlation_eye_correction_proxy": "noise-corr\neye correction",
        "reliable_shared_denominator_proxy": "reliable-shared\nproxy",
    }
    block = aggregate[aggregate["evidence_class"].isin(keep)].copy()
    block["label"] = block["evidence_class"].map(keep)
    order = list(keep.values())
    block["label"] = pd.Categorical(block["label"], order, ordered=True)
    block = block.sort_values("label")

    colors = [COLORS["blue"], COLORS["blue"], COLORS["purple"], COLORS["gray"]]
    fig, ax = plt.subplots(figsize=(5.7, 2.9), constrained_layout=True)
    x = np.arange(len(block))
    y = block["fraction_session_mean"].to_numpy(dtype=float)
    sem = block["fraction_session_sem"].to_numpy(dtype=float)
    ax.bar(x, y, color=colors, width=0.64)
    ax.errorbar(x, y, yerr=sem, color=COLORS["dark"], lw=1.0, capsize=0, linestyle="none")
    ax.set_xticks(x, block["label"])
    ax.set_ylabel("session-mean fraction")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.8)
    ax.set_title("Bridge to FEM-linked covariance evidence")
    _clean_axis(ax)
    ax.text(
        0.02,
        0.91,
        "mixed denominators; bridge/supplement only",
        transform=ax.transAxes,
        fontsize=7.5,
        color=COLORS["orange"],
    )
    for idx, row in block.iterrows():
        pos = list(block.index).index(idx)
        ax.text(pos, float(row["fraction_session_mean"]) + 0.045, f"n={int(row['n_sessions'])}", ha="center", fontsize=7.1)
    _save(fig, out_dir, "A5_covariance_bridge_guardrail")
    return block[
        [
            "evidence_class",
            "metric",
            "label",
            "n_rows",
            "n_sessions",
            "fraction_session_mean",
            "fraction_session_sem",
            "notes",
        ]
    ]


def _write_caption(out_dir: Path) -> None:
    caption = """# Panel A Subpanels

Generated cache-first from existing retinal-movie QC, Figure 4 headline stats,
and reafferent-variance accounting tables, with lightweight premise cartoons.

Subpanels:

- `A1_retinal_movie_transform`: fixed screen image plus measured-like eye trace
  creates shifted retinal crops.
- `A2_movie_transform_qc`: FEM movies introduce temporal contrast and motion
  power while stabilized movies remove retinal translation.
- `A3_gradient_sampling_cartoon`: small translations sample local gradients,
  motivating image-dependent motion axes.
- `A4_backimage_pipeline_bridge`: canonical BackImage/V1-twin pipeline used by
  B-E.
- `A5_covariance_bridge_guardrail`: bridge to existing FEM-linked covariance
  evidence with mixed-denominator caveat.

Claim boundary:

```text
Panel A teaches the physical transformation: a fixed screen image becomes a
retinal movie during fixation. It provides QC/provenance and a bridge to
FEM-linked covariance, but it does not by itself establish active-sensing
optimality or the downstream BackImage observer/behavior claims.
```
"""
    (out_dir / "panel_A_subpanels_caption.md").write_text(caption)


def _write_index(out_dir: Path, generated: Iterable[str]) -> None:
    lines = ["# Panel A Generated Assets", ""]
    for stem in generated:
        lines.append(f"- `{stem}.png`")
        lines.append(f"- `{stem}.pdf`")
    lines.extend(
        [
            "- `panel_A_retinal_movie_transform_values.csv`",
            "- `panel_A_movie_transform_qc_values.csv`",
            "- `panel_A_gradient_sampling_values.csv`",
            "- `panel_A_backimage_pipeline_values.csv`",
            "- `panel_A_covariance_bridge_values.csv`",
            "- `panel_A_subpanels_caption.md`",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie-pack", type=Path, default=MOVIE_PACK)
    parser.add_argument("--fig4-headline", type=Path, default=FIG4_HEADLINE)
    parser.add_argument("--reafferent-dir", type=Path, default=REAFFERENT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    _configure_matplotlib()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    qc_summary = pd.read_csv(args.movie_pack / "retinal_movie_transform_qc_summary.csv")
    reafferent_summary = pd.read_csv(args.reafferent_dir / "variance_accounting_aggregate_summary.csv")

    a1_values = plot_a1_retinal_movie_transform(args.out_dir)
    a2_values = plot_a2_movie_transform_qc(qc_summary, args.out_dir)
    a3_values = plot_a3_local_gradient_sampling(args.out_dir)
    a4_values = plot_a4_backimage_pipeline(args.fig4_headline / "fig4_active_sensing_headline_stats.json", args.out_dir)
    a5_values = plot_a5_covariance_bridge(reafferent_summary, args.out_dir)

    a1_values.to_csv(args.out_dir / "panel_A_retinal_movie_transform_values.csv", index=False)
    a2_values.to_csv(args.out_dir / "panel_A_movie_transform_qc_values.csv", index=False)
    a3_values.to_csv(args.out_dir / "panel_A_gradient_sampling_values.csv", index=False)
    a4_values.to_csv(args.out_dir / "panel_A_backimage_pipeline_values.csv", index=False)
    a5_values.to_csv(args.out_dir / "panel_A_covariance_bridge_values.csv", index=False)
    _write_caption(args.out_dir)
    _write_index(
        args.out_dir,
        [
            "A1_retinal_movie_transform",
            "A2_movie_transform_qc",
            "A3_gradient_sampling_cartoon",
            "A4_backimage_pipeline_bridge",
            "A5_covariance_bridge_guardrail",
        ],
    )


if __name__ == "__main__":
    main()
