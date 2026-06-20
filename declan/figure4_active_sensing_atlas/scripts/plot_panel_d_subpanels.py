"""Build cache-only Figure 4D subpanels from axis and edge-geometry outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKIMAGE_BASE = REPO_ROOT / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
DEFAULT_MATCHED_AXIS_DIR = (
    BACKIMAGE_BASE / "backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1"
)
DEFAULT_HARDNEG_AXIS_DIR = (
    BACKIMAGE_BASE / "backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1"
)
DEFAULT_STABILITY_DIR = BACKIMAGE_BASE / "backimage_edge_parallel_stability_screen_yfix_n256_pop256"
DEFAULT_OBJECTIVE_DIR = BACKIMAGE_BASE / "backimage_conditional_fixation_objectives_twin_axis_only_n256"
DEFAULT_OUT_DIR = REPO_ROOT / "declan/figure4_active_sensing_atlas/figures/panel_D"

COLORS = {
    "known": "#242a2f",
    "zero": "#8e9aa6",
    "parallel": "#2f8f6a",
    "orthogonal": "#8064a2",
    "pixel": "#2f8f6a",
    "twin": "#3366aa",
    "raw": "#242a2f",
    "response": "#8064a2",
    "pixel_objective": "#d07a22",
    "muted": "#6f7a83",
    "grid": "#d8dde3",
}
AXIS_LABELS = {
    "axis_edge_parallel": "edge-parallel",
    "axis_edge_orthogonal": "edge-orthogonal",
}
OBJECTIVE_LABELS = {
    "optimized_response_stability": "response stability",
    "optimized_response_refresh_lambda_0.25": "response refresh",
    "optimized_PA": "pose-aware response",
    "optimized_PB": "pose-blind response",
    "optimized_pixel_isophote": "pixel isophote",
    "optimized_refresh_only": "pixel refresh control",
}
OBJECTIVE_PLOT_LABELS = {
    "optimized_response_stability": "response\nstability",
    "optimized_response_refresh_lambda_0.25": "response\nrefresh",
    "optimized_PA": "pose-aware\nresponse",
    "optimized_PB": "pose-blind\nresponse",
    "optimized_pixel_isophote": "pixel\nisophote",
    "optimized_refresh_only": "pixel refresh\ncontrol",
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


def _synthetic_edge_patch(size: int = 128) -> np.ndarray:
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    edge = 0.45 + 0.26 * np.tanh(8.0 * (0.58 * x + 0.82 * y + 0.05))
    texture = 0.08 * np.sin(18.0 * (0.86 * x - 0.33 * y))
    texture += 0.05 * np.sin(27.0 * (0.12 * x + 0.98 * y))
    return np.clip(edge + texture, 0.0, 1.0)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=2.0,
            color=color,
            transform=ax.transAxes,
        )
    )


def _load_axis_rows(matched_dir: Path, hardneg_dir: Path) -> pd.DataFrame:
    matched = pd.read_csv(matched_dir / "observer_summary.csv")
    hardneg = pd.read_csv(hardneg_dir / "observer_summary.csv")
    rows = []

    def add_rows(df: pd.DataFrame, condition_label: str, condition_order: int) -> None:
        work = df[df["likelihood_scale"].astype(float) == 1.0].copy()
        for scale, scale_rows in work.groupby(work["observation_scale"].astype(float)):
            zero = float(scale_rows["zero_eye_accuracy"].iloc[0])
            known = float(scale_rows["known_eye_accuracy"].iloc[0])
            axis_values = {}
            for axis in ["axis_edge_parallel", "axis_edge_orthogonal"]:
                axis_row = scale_rows[scale_rows["prior_family"] == axis]
                if axis_row.empty:
                    raise ValueError(f"Missing {axis} for {condition_label} scale {scale}")
                axis_values[axis] = float(axis_row["joint_eye_accuracy"].iloc[0])
            rows.append(
                {
                    "condition": condition_label,
                    "condition_order": condition_order,
                    "scale": scale,
                    "zero_eye_accuracy": zero,
                    "known_eye_accuracy": known,
                    "parallel_joint_accuracy": axis_values["axis_edge_parallel"],
                    "orthogonal_joint_accuracy": axis_values["axis_edge_orthogonal"],
                    "parallel_minus_orthogonal": axis_values["axis_edge_parallel"]
                    - axis_values["axis_edge_orthogonal"],
                }
            )

    add_rows(matched, "matched static 0.5x", 0)
    add_rows(hardneg[hardneg["observation_scale"].astype(float) == 0.5], "hard negatives 0.5x", 1)
    add_rows(hardneg[hardneg["observation_scale"].astype(float) == 1.0], "hard negatives 1.0x", 2)
    add_rows(hardneg[hardneg["observation_scale"].astype(float) == 2.0], "hard negatives 2.0x", 3)

    out = pd.DataFrame(rows)
    out = out.sort_values("condition_order").drop_duplicates("condition_order")
    return out


def plot_d1_axis_schematic(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    ax.imshow(_synthetic_edge_patch(), cmap="gray", vmin=0, vmax=1, extent=(0, 1, 0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    center = (0.52, 0.50)
    _arrow(ax, center, (0.82, 0.70), COLORS["parallel"])
    _arrow(ax, center, (0.33, 0.80), COLORS["orthogonal"])
    _arrow(ax, center, (0.70, 0.20), COLORS["orthogonal"])
    ax.plot([0.19, 0.86], [0.28, 0.75], color="white", lw=3.2, alpha=0.75, transform=ax.transAxes)
    ax.plot([0.19, 0.86], [0.28, 0.75], color="#242a2f", lw=1.2, alpha=0.85, transform=ax.transAxes)
    ax.text(0.74, 0.77, "edge-parallel", color=COLORS["parallel"], fontsize=8, transform=ax.transAxes)
    ax.text(0.10, 0.83, "edge-normal /\northogonal", color=COLORS["orthogonal"], fontsize=8, transform=ax.transAxes)
    ax.text(0.50, 0.06, "Useful motion direction depends on the image and objective", ha="center", fontsize=8.5, transform=ax.transAxes)
    ax.set_title("Local image axes for trajectory priors")
    _save(fig, out_dir, "D1_local_axis_schematic")


def plot_d2_axis_observer(axis_values: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(6.2, 3.0), constrained_layout=True)
    x = np.arange(len(axis_values))
    ax.plot(x, axis_values["known_eye_accuracy"], color=COLORS["known"], marker="o", lw=1.8, label="known eye")
    ax.plot(x, axis_values["zero_eye_accuracy"], color=COLORS["zero"], marker="o", lw=1.8, label="zero eye")
    ax.plot(
        x,
        axis_values["parallel_joint_accuracy"],
        color=COLORS["parallel"],
        marker="o",
        lw=1.9,
        label="edge-parallel prior",
    )
    ax.plot(
        x,
        axis_values["orthogonal_joint_accuracy"],
        color=COLORS["orthogonal"],
        marker="s",
        lw=1.9,
        label="edge-orthogonal prior",
    )
    ax.set_xticks(x, axis_values["condition"], rotation=18, ha="right")
    ax.set_ylim(0.25, 1.04)
    ax.set_ylabel("image-identification accuracy")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.8)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    ax.set_title("Axis-conditioned priors rescue image identity")
    _clean_axis(ax)
    _save(fig, out_dir, "D2_axis_conditioned_accuracy")
    return axis_values.copy()


def plot_d3_axis_preference(axis_values: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    block = axis_values[["condition", "condition_order", "parallel_minus_orthogonal"]].copy()
    fig, ax = plt.subplots(figsize=(4.8, 2.8), constrained_layout=True)
    x = np.arange(len(block))
    vals = block["parallel_minus_orthogonal"].to_numpy(dtype=float)
    colors = [COLORS["parallel"] if val >= 0 else COLORS["orthogonal"] for val in vals]
    ax.bar(x, vals, color=colors, width=0.65)
    ax.axhline(0.0, color="#242a2f", lw=0.8)
    ax.set_xticks(x, block["condition"], rotation=20, ha="right")
    ax.set_ylabel("parallel minus orthogonal\njoint accuracy")
    ax.set_ylim(-0.08, 0.06)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.8)
    ax.set_title("Axis preference changes with condition")
    _clean_axis(ax)
    for idx, val in enumerate(vals):
        va = "bottom" if val >= 0 else "top"
        y = val + (0.004 if val >= 0 else -0.004)
        ax.text(idx, y, f"{val:+.3f}", ha="center", va=va, fontsize=7.5)
    _save(fig, out_dir, "D3_axis_preference_guardrail")
    return block


def plot_d4_edge_stability(stability_dir: Path, out_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(stability_dir / "stability_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.8), constrained_layout=True)
    for ax, screen in zip(axes, ["pixel", "twin"], strict=True):
        row = summary[summary["screen"] == screen].iloc[0]
        mean = float(row["mean_advantage_session_mean"])
        lo = float(row["ci95_low_session_mean"])
        hi = float(row["ci95_high_session_mean"])
        ax.bar([0], [mean], color=COLORS[screen], width=0.55)
        ax.errorbar([0], [mean], yerr=[[mean - lo], [hi - mean]], color="#242a2f", lw=1.2, capsize=0)
        ax.axhline(0.0, color="#242a2f", lw=0.8)
        ax.set_xticks([0], [screen])
        ax.grid(axis="y", color=COLORS["grid"], lw=0.8)
        ax.set_title(f"{screen} stability")
        _clean_axis(ax)
        ax.text(
            0,
            mean + (hi - lo) * 0.12,
            f"{int(row['n_sessions_positive_advantage'])}/{int(row['n_sessions'])} sessions",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    axes[0].set_ylabel("orthogonal cost minus\nparallel cost")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.suptitle("Edge-parallel displacement preserves local structure")
    _save(fig, out_dir, "D4_edge_parallel_stability")
    return summary.copy()


def plot_d5_objective_guardrail(objective_dir: Path, out_dir: Path) -> pd.DataFrame:
    deltas = pd.read_csv(objective_dir / "paired_session_deltas_vs_raw_edge.csv")
    selected = [
        "optimized_response_stability",
        "optimized_response_refresh_lambda_0.25",
        "optimized_PA",
        "optimized_PB",
        "optimized_pixel_isophote",
        "optimized_refresh_only",
    ]
    block = deltas[deltas["objective"].isin(selected)].copy()
    order = {name: idx for idx, name in enumerate(selected)}
    block["order"] = block["objective"].map(order)
    block = block.sort_values("order")
    block["label"] = block["objective"].map(OBJECTIVE_LABELS)
    block["objective_class"] = np.where(block["objective"].str.contains("pixel|refresh_only"), "pixel_objective", "response")
    block.loc[block["objective"].eq("optimized_refresh_only"), "objective_class"] = "pixel_objective"

    fig, ax = plt.subplots(figsize=(5.7, 3.0), constrained_layout=True)
    x = np.arange(len(block))
    y = block["mean_delta_cos2_session"].to_numpy(dtype=float)
    lo = block["ci95_low"].to_numpy(dtype=float)
    hi = block["ci95_high"].to_numpy(dtype=float)
    colors = [COLORS[key] for key in block["objective_class"]]
    ax.bar(x, y, color=colors, width=0.68)
    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), color="#242a2f", lw=1.0, capsize=0, linestyle="none")
    ax.axhline(0.0, color="#242a2f", lw=0.8)
    plot_labels = block["objective"].map(OBJECTIVE_PLOT_LABELS)
    ax.set_xticks(x, plot_labels, rotation=25, ha="right")
    ax.set_ylabel("delta cos2 versus raw edge")
    ax.set_ylim(-0.46, 0.38)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.8)
    ax.set_title("Response objectives do not yet beat raw edge")
    _clean_axis(ax)
    ax.text(
        0.02,
        0.95,
        "positive = more aligned than raw edge",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#303840",
    )
    _save(fig, out_dir, "D5_objective_alignment_guardrail")
    return block


def _write_caption(out_dir: Path) -> None:
    caption = """# Panel D Subpanels

Generated cache-only from BackImage axis-conditioned observer, edge-parallel
stability, and objective-alignment outputs.

Subpanels:

- `D1_local_axis_schematic`: local edge-parallel and edge-orthogonal axes.
- `D2_axis_conditioned_accuracy`: axis-conditioned priors rescue image
  identity above zero-eye.
- `D3_axis_preference_guardrail`: parallel-vs-orthogonal preference changes
  with candidate set and scale.
- `D4_edge_parallel_stability`: edge-parallel displacement has lower pixel and
  V1-twin cost than matched orthogonal displacement.
- `D5_objective_alignment_guardrail`: current response objectives do not yet
  beat raw edge alignment; pixel-only controls are separate.

Claim boundary:

```text
Panel D supports image-conditioned useful motion axes and a clean local
edge-parallel preservation result. It does not support a universal
edge-parallel policy or a settled V1-twin objective that beats raw image
geometry.
```
"""
    (out_dir / "panel_D_subpanels_caption.md").write_text(caption)


def _write_index(out_dir: Path, generated: Iterable[str]) -> None:
    lines = ["# Panel D Generated Assets", ""]
    for stem in generated:
        lines.append(f"- `{stem}.png`")
        lines.append(f"- `{stem}.pdf`")
    lines.extend(
        [
            "- `panel_D_axis_conditioned_values.csv`",
            "- `panel_D_axis_preference_values.csv`",
            "- `panel_D_edge_stability_values.csv`",
            "- `panel_D_objective_guardrail_values.csv`",
            "- `panel_D_subpanels_caption.md`",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-axis-dir", type=Path, default=DEFAULT_MATCHED_AXIS_DIR)
    parser.add_argument("--hardneg-axis-dir", type=Path, default=DEFAULT_HARDNEG_AXIS_DIR)
    parser.add_argument("--stability-dir", type=Path, default=DEFAULT_STABILITY_DIR)
    parser.add_argument("--objective-dir", type=Path, default=DEFAULT_OBJECTIVE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    _configure_matplotlib()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    axis_values = _load_axis_rows(args.matched_axis_dir, args.hardneg_axis_dir)
    plot_d1_axis_schematic(args.out_dir)
    axis_panel_values = plot_d2_axis_observer(axis_values, args.out_dir)
    preference_values = plot_d3_axis_preference(axis_values, args.out_dir)
    stability_values = plot_d4_edge_stability(args.stability_dir, args.out_dir)
    objective_values = plot_d5_objective_guardrail(args.objective_dir, args.out_dir)

    axis_panel_values.to_csv(args.out_dir / "panel_D_axis_conditioned_values.csv", index=False)
    preference_values.to_csv(args.out_dir / "panel_D_axis_preference_values.csv", index=False)
    stability_values.to_csv(args.out_dir / "panel_D_edge_stability_values.csv", index=False)
    objective_values.to_csv(args.out_dir / "panel_D_objective_guardrail_values.csv", index=False)
    _write_caption(args.out_dir)
    _write_index(
        args.out_dir,
        [
            "D1_local_axis_schematic",
            "D2_axis_conditioned_accuracy",
            "D3_axis_preference_guardrail",
            "D4_edge_parallel_stability",
            "D5_objective_alignment_guardrail",
        ],
    )


if __name__ == "__main__":
    main()
