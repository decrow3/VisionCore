"""Build cache-only Figure 4B subpanels from aggregate FEM-information outputs."""

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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKIMAGE_BASE = REPO_ROOT / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
DEFAULT_AGGREGATE_DIR = (
    BACKIMAGE_BASE
    / "backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched"
)
DEFAULT_INCREMENTAL_DIR = DEFAULT_AGGREGATE_DIR / "incremental_static_plus_motion_relids"
DEFAULT_OUT_DIR = REPO_ROOT / "declan/figure4_active_sensing_atlas/figures/panel_B"

MOTION_ORDER = ["empirical", "ou", "brownian", "rotated"]
COLORS = {
    "empirical": "#244f7a",
    "ou": "#d07a22",
    "brownian": "#707070",
    "rotated": "#8064a2",
    "gabor": "#244f7a",
    "pyramid": "#2f8f6a",
    "dark": "#242a2f",
    "muted": "#7b8792",
    "light": "#eef2f4",
}
MOTION_LABELS = {
    "empirical": "empirical drift",
    "ou": "OU-like",
    "brownian": "Brownian",
    "rotated": "rotated",
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


def _scale_value(scale_id: str) -> float:
    return float(str(scale_id).replace("rel_", "").replace("p", ".").replace("x", ""))


def _scale_label(value: float) -> str:
    return f"{value:g}x"


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _errbar(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    y_col: str,
    lo_col: str,
    hi_col: str,
    label: str,
    color: str,
    marker: str = "o",
    linestyle: str = "-",
) -> None:
    block = df.sort_values("scale")
    x = block["scale"].to_numpy(dtype=float)
    y = block[y_col].to_numpy(dtype=float)
    lo = block[lo_col].to_numpy(dtype=float)
    hi = block[hi_col].to_numpy(dtype=float)
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([y - lo, hi - y]),
        color=color,
        marker=marker,
        markersize=4,
        linewidth=1.8,
        linestyle=linestyle,
        capsize=0,
        label=label,
    )


def _node(ax: plt.Axes, xy: tuple[float, float], text: str, width: float = 0.19) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x - width / 2, y - 0.060),
        width,
        0.120,
        boxstyle="round,pad=0.014,rounding_size=0.014",
        edgecolor="#c5ccd2",
        facecolor="#f8fafb",
        linewidth=0.8,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=8, transform=ax.transAxes)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#5d6871",
            transform=ax.transAxes,
        )
    )


def _synthetic_patch(size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(17)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    edge = 0.25 * np.tanh(8.0 * (0.5 * x + 0.9 * y + 0.1))
    texture = 0.12 * np.sin(16.0 * (0.8 * x - 0.35 * y))
    texture += 0.08 * np.sin(32.0 * (0.2 * x + 0.95 * y))
    blob = 0.22 * np.exp(-((x + 0.36) ** 2 / 0.18 + (y - 0.16) ** 2 / 0.22))
    patch = 0.50 + edge + texture + blob + 0.04 * rng.normal(size=(size, size))
    return np.clip(patch, 0.0, 1.0)


def _trace(kind: str, n: int = 72) -> np.ndarray:
    rng = np.random.default_rng({"empirical": 3, "ou": 5, "brownian": 7, "rotated": 11}[kind])
    if kind == "ou":
        trace = np.zeros((n, 2))
        for i in range(1, n):
            trace[i] = 0.84 * trace[i - 1] + rng.normal(scale=0.040, size=2)
    elif kind == "brownian":
        trace = np.cumsum(rng.normal(scale=0.026, size=(n, 2)), axis=0)
    else:
        t = np.linspace(0, 1, n)
        trace = np.c_[
            0.16 * np.sin(2.0 * np.pi * t) + 0.04 * rng.normal(size=n),
            0.07 * np.sin(5.4 * np.pi * t + 0.3) + 0.025 * rng.normal(size=n),
        ]
        if kind == "rotated":
            trace = trace @ np.array([[0.0, -1.0], [1.0, 0.0]]).T
    trace -= trace.mean(axis=0, keepdims=True)
    denom = np.max(np.abs(trace)) or 1.0
    return trace / denom


def plot_b1_task_schematic(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.2), constrained_layout=True)
    ax.set_axis_off()

    patch_ax = ax.inset_axes([0.03, 0.22, 0.16, 0.58])
    patch_ax.imshow(_synthetic_patch(), cmap="gray", vmin=0, vmax=1)
    patch_ax.set_xticks([])
    patch_ax.set_yticks([])
    for spine in patch_ax.spines.values():
        spine.set_color("#c5ccd2")
        spine.set_linewidth(0.8)
    ax.text(0.11, 0.12, "natural patch", ha="center", fontsize=7.5, transform=ax.transAxes)

    trace_ax = ax.inset_axes([0.25, 0.18, 0.22, 0.65])
    trace_ax.set_facecolor("#fbfcfd")
    offsets = {"empirical": 0.27, "ou": 0.09, "brownian": -0.09, "rotated": -0.27}
    for family in MOTION_ORDER:
        tr = _trace(family)
        trace_ax.plot(0.13 * tr[:, 0], 0.13 * tr[:, 1] + offsets[family], color=COLORS[family], lw=1.2)
        trace_ax.text(0.20, offsets[family], MOTION_LABELS[family], color=COLORS[family], va="center", fontsize=7.2)
    trace_ax.set_xlim(-0.18, 0.44)
    trace_ax.set_ylim(-0.43, 0.43)
    trace_ax.set_xticks([])
    trace_ax.set_yticks([])
    for spine in trace_ax.spines.values():
        spine.set_visible(False)
    ax.text(0.36, 0.08, "motion families", ha="center", fontsize=7.5, transform=ax.transAxes)

    _node(ax, (0.59, 0.63), "V1-twin\nresponse movie", width=0.20)
    _node(ax, (0.59, 0.35), "temporal-PC\nsummary", width=0.20)
    _node(ax, (0.82, 0.50), "decode image\nfeatures", width=0.18)
    _arrow(ax, (0.20, 0.50), (0.25, 0.50))
    _arrow(ax, (0.47, 0.50), (0.50, 0.58))
    _arrow(ax, (0.59, 0.57), (0.59, 0.42))
    _arrow(ax, (0.69, 0.50), (0.73, 0.50))
    ax.text(
        0.50,
        0.94,
        "Panel B question: can the response movie tell us more about image features than the static response?",
        ha="center",
        va="top",
        fontsize=9.5,
        color=COLORS["dark"],
        transform=ax.transAxes,
    )
    _save(fig, out_dir, "B1_task_schematic")


def plot_b2_motion_qc(motion_summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    nonstatic = motion_summary[motion_summary["family"].isin(MOTION_ORDER)].copy()
    nonstatic["scale"] = nonstatic["scale_id"].map(_scale_value)
    nonstatic = nonstatic.sort_values(["family", "scale"])

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45), constrained_layout=True)
    ax = axes[0]
    for family in MOTION_ORDER:
        sub = nonstatic[nonstatic["family"] == family]
        ax.plot(
            sub["scale"],
            sub["median_effective_to_requested_rms"],
            marker="o",
            color=COLORS[family],
            lw=1.7,
            label=MOTION_LABELS[family],
        )
    ax.axhline(1.0, color="#222222", lw=0.8, linestyle="--")
    ax.set_xlabel("motion scale")
    ax.set_ylabel("effective / requested RMS")
    ax.set_xticks(sorted(nonstatic["scale"].unique()), [_scale_label(v) for v in sorted(nonstatic["scale"].unique())])
    ax.set_ylim(0.94, 1.06)
    ax.grid(axis="y", color="#d8dde3", lw=0.8)
    _clean_axis(ax)

    ax = axes[1]
    for family in MOTION_ORDER:
        sub = nonstatic[nonstatic["family"] == family]
        ax.plot(
            sub["scale"],
            sub["median_path_length_deg"],
            marker="o",
            color=COLORS[family],
            lw=1.7,
            label=MOTION_LABELS[family],
        )
    ax.set_xlabel("motion scale")
    ax.set_ylabel("median path length (deg)")
    ax.set_xticks(sorted(nonstatic["scale"].unique()), [_scale_label(v) for v in sorted(nonstatic["scale"].unique())])
    ax.grid(axis="y", color="#d8dde3", lw=0.8)
    _clean_axis(ax)
    axes[0].legend(loc="upper left", frameon=False, ncol=1)
    fig.suptitle("Motion controls are RMS matched and unclipped", fontsize=10.5)
    _save(fig, out_dir, "B2_motion_family_qc")
    return nonstatic


def plot_b3_gain_vs_static(gain_vs_static: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    block = gain_vs_static[
        (gain_vs_static["motion_summary"] == "temporal_pca")
        & (gain_vs_static["family"] == "empirical")
        & (
            ((gain_vs_static["latent"] == "gabor_local_field") & (gain_vs_static["k"] == 4))
            | ((gain_vs_static["latent"] == "pyramid_local_field") & (gain_vs_static["k"] == 8))
        )
    ].copy()
    block["scale"] = block["scale_id"].map(_scale_value)
    block = block.sort_values(["latent", "scale"])

    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    _errbar(
        ax,
        block[(block["latent"] == "gabor_local_field") & (block["k"] == 4)],
        y_col="incremental_gain_neg_mse",
        lo_col="ci95_low",
        hi_col="ci95_high",
        label="Gabor k=4",
        color=COLORS["gabor"],
        marker="o",
    )
    _errbar(
        ax,
        block[(block["latent"] == "pyramid_local_field") & (block["k"] == 8)],
        y_col="incremental_gain_neg_mse",
        lo_col="ci95_low",
        hi_col="ci95_high",
        label="Pyramid k=8",
        color=COLORS["pyramid"],
        marker="s",
    )
    ax.axhline(0.0, color="#222222", lw=0.8)
    ax.set_xlabel("motion scale")
    ax.set_ylabel("incremental feature decoding gain\n(static + motion minus static, -MSE)")
    ax.set_xticks(sorted(block["scale"].unique()), [_scale_label(v) for v in sorted(block["scale"].unique())])
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color="#d8dde3", lw=0.8)
    ax.set_title("Empirical drift-like motion adds decodable feature structure")
    _clean_axis(ax)
    _save(fig, out_dir, "B3_empirical_gain_vs_static")
    return block


def plot_b4_control_contrasts(contrasts: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    block = contrasts[
        (contrasts["motion_summary"] == "temporal_pca")
        & (contrasts["lhs_family"] == "empirical")
        & (contrasts["latent"] == "gabor_local_field")
        & (contrasts["k"] == 4)
        & (contrasts["rhs_family"].isin(["ou", "brownian", "rotated"]))
    ].copy()
    block["scale"] = block["scale_id"].map(_scale_value)
    block = block.sort_values(["rhs_family", "scale"])

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for rhs in ["ou", "brownian", "rotated"]:
        _errbar(
            ax,
            block[block["rhs_family"] == rhs],
            y_col="incremental_gain_delta_neg_mse",
            lo_col="ci95_low",
            hi_col="ci95_high",
            label=f"empirical - {MOTION_LABELS[rhs]}",
            color=COLORS[rhs],
            marker={"ou": "o", "brownian": "s", "rotated": "^"}[rhs],
        )
    ax.axhline(0.0, color="#222222", lw=0.8)
    ax.set_xlabel("motion scale")
    ax.set_ylabel("incremental gain contrast (-MSE)")
    ax.set_xticks(sorted(block["scale"].unique()), [_scale_label(v) for v in sorted(block["scale"].unique())])
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color="#d8dde3", lw=0.8)
    ax.set_title("Empirical advantage is strongest at small scales")
    _clean_axis(ax)
    _save(fig, out_dir, "B4_empirical_minus_controls")
    return block


def plot_b5_absolute_gain_guardrail(gain_vs_static: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    block = gain_vs_static[
        (gain_vs_static["motion_summary"] == "temporal_pca")
        & (gain_vs_static["latent"] == "gabor_local_field")
        & (gain_vs_static["k"] == 4)
        & (gain_vs_static["family"].isin(MOTION_ORDER))
    ].copy()
    block["scale"] = block["scale_id"].map(_scale_value)
    block = block.sort_values(["family", "scale"])

    fig, ax = plt.subplots(figsize=(4.4, 3.0), constrained_layout=True)
    for family in MOTION_ORDER:
        _errbar(
            ax,
            block[block["family"] == family],
            y_col="incremental_gain_neg_mse",
            lo_col="ci95_low",
            hi_col="ci95_high",
            label=MOTION_LABELS[family],
            color=COLORS[family],
            marker={"empirical": "o", "ou": "o", "brownian": "s", "rotated": "^"}[family],
        )
    ax.axhline(0.0, color="#222222", lw=0.8)
    ax.set_xlabel("motion scale")
    ax.set_ylabel("incremental gain over static (-MSE)")
    ax.set_xticks(sorted(block["scale"].unique()), [_scale_label(v) for v in sorted(block["scale"].unique())])
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    ax.grid(axis="y", color="#d8dde3", lw=0.8)
    ax.set_title("Guardrail: generic motion can catch up")
    _clean_axis(ax)
    _save(fig, out_dir, "B5_absolute_gain_guardrail")
    return block


def _write_caption(out_dir: Path) -> None:
    caption = """# Panel B Subpanels

Generated cache-only from the cleaned BackImage aggregate FEM-information run:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
```

Subpanels:

- `B1_task_schematic`: analysis schematic for image features decoded from
  static-plus-motion response summaries.
- `B2_motion_family_qc`: RMS matching and path-length summaries for empirical,
  OU-like, Brownian, and rotated motion families.
- `B3_empirical_gain_vs_static`: empirical temporal-PCA feature-decoding gain
  over the static-only response.
- `B4_empirical_minus_controls`: empirical-minus-control incremental gain
  contrasts for Gabor k=4 temporal-PCA summaries.
- `B5_absolute_gain_guardrail`: absolute gains for all motion families, showing
  why Brownian/rotated caveats matter at larger scales.

Claim boundary:

```text
This is deterministic V1-twin feature-decoding gain in -MSE units, not literal
mutual information. The strongest control-specific claim is small-scale:
empirical beats OU robustly and beats Brownian/rotated most cleanly at 0.25x
to 0.5x.
```
"""
    (out_dir / "panel_B_subpanels_caption.md").write_text(caption)


def _write_index(out_dir: Path, generated: Iterable[str]) -> None:
    lines = ["# Panel B Generated Assets", ""]
    for stem in generated:
        lines.append(f"- `{stem}.png`")
        lines.append(f"- `{stem}.pdf`")
    lines.extend(
        [
            "- `panel_B_motion_qc_values.csv`",
            "- `panel_B_gain_vs_static_values.csv`",
            "- `panel_B_control_contrast_values.csv`",
            "- `panel_B_absolute_gain_guardrail_values.csv`",
            "- `panel_B_subpanels_caption.md`",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, default=DEFAULT_AGGREGATE_DIR)
    parser.add_argument("--incremental-dir", type=Path, default=DEFAULT_INCREMENTAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    _configure_matplotlib()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    motion_summary = pd.read_csv(args.aggregate_dir / "aggregate_motion_summary.csv")
    gain_vs_static = pd.read_csv(args.incremental_dir / "incremental_gain_vs_static.csv")
    contrasts = pd.read_csv(args.incremental_dir / "incremental_gain_contrasts.csv")

    plot_b1_task_schematic(args.out_dir)
    qc_values = plot_b2_motion_qc(motion_summary, args.out_dir)
    gain_values = plot_b3_gain_vs_static(gain_vs_static, args.out_dir)
    contrast_values = plot_b4_control_contrasts(contrasts, args.out_dir)
    guardrail_values = plot_b5_absolute_gain_guardrail(gain_vs_static, args.out_dir)

    qc_values.to_csv(args.out_dir / "panel_B_motion_qc_values.csv", index=False)
    gain_values.to_csv(args.out_dir / "panel_B_gain_vs_static_values.csv", index=False)
    contrast_values.to_csv(args.out_dir / "panel_B_control_contrast_values.csv", index=False)
    guardrail_values.to_csv(args.out_dir / "panel_B_absolute_gain_guardrail_values.csv", index=False)
    _write_caption(args.out_dir)
    _write_index(
        args.out_dir,
        [
            "B1_task_schematic",
            "B2_motion_family_qc",
            "B3_empirical_gain_vs_static",
            "B4_empirical_minus_controls",
            "B5_absolute_gain_guardrail",
        ],
    )


if __name__ == "__main__":
    main()
