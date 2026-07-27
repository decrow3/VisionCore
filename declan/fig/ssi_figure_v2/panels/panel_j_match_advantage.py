#!/usr/bin/env python3
"""Panel J (promoted): trace-contour match advantage by local edge coherence.

This is the single-panel distillation of the behavior-model bridge: for each
population, how much more (or less) predicted SSI the real trace-contour
matching gives you than a randomly rotated trajectory, as a function of local
edge coherence. It reuses the same data, palette, and population styling as
the standalone explainer figure's Panel C
(behavior_model_bridge/plot_bridge_explainer_figure.py), rendered at this
slot's real footprint (originally Panel I's; the whole figure's G/H/I shifted
to H/I/J once a new panel G was inserted between F and the old G -- see
generate_ssi_figure_v2.py's EF_INSET_* constants and draw_contour_components_panel).

This replaced panel_i_edge_alignment.py's descriptive drift-cloud/edge
alignment plot in the main ssi_figure_v2 slot: that panel showed behavior
correlates with coherence, but never showed that the correlation is
model-beneficial relative to chance -- this panel closes that loop. The
original is left in place, unwired, for reference/comparison.

Only three of the five populations from the explainer figure are shown here
(aligned high-SF, all high-SF, all low-SF): oblique and orthogonal high-SF
sit between aligned and all-high-SF and don't add a distinguishable line at
this panel's size -- all-high-SF already carries that "partial or no
alignment" middle ground. The five-population version remains the one to use
in the explainer/option-sheet context where there's room for it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.fig.ssi_figure_v2.behavior_model_bridge import plot_bridge_explainer_figure as explainer
from declan.fig.ssi_figure_v2.behavior_model_bridge import run_behavior_model_bridge as bridge

OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels"
COHERENCE_SUMMARY_CSV = explainer.COHERENCE_SUMMARY_CSV
METRIC_FAMILY = "component_rms"
# The real ssi_figure_v2 gs[2, 2] cell (MAIN_GRID_KWARGS at FIGURE_SIZE_IN =
# (8.5, 11.0)) -- panel I's actual footprint, not its ~2.35x2.25 standalone
# preview approximation.
FIGSIZE = (1.955, 2.432)


def configure_matplotlib() -> None:
    explainer.configure_matplotlib()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_values(csv_path: Path = COHERENCE_SUMMARY_CSV) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame = frame[frame["score_type"].astype(str).eq("component_mean_marginal") & frame["metric_family"].astype(str).eq(METRIC_FAMILY)].copy()
    frame["coherence_bin"] = pd.Categorical(frame["coherence_bin"], categories=bridge.COHERENCE_ORDER, ordered=True)
    return frame


PLOT_POPULATION_ORDER = ("high_sf_aligned", "high_sf_all", "low_sf_all")
SHORT_POPULATION_LABELS = {
    "high_sf_aligned": "Aligned high-SF",
    "high_sf_all": "All high-SF",
    "low_sf_all": "Low-SF",
}


TITLE = "Real (vs. randomly rotated)\nFEM trajectories preferentially\nbenefit aligned high-SF units"


def draw_panel(
    ax: plt.Axes,
    *,
    label: str = "J",
    title: str = TITLE,
    values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Draw the coherence-resolved match-advantage panel on ``ax``."""
    values = load_values() if values is None else values.copy()
    x = np.arange(len(bridge.COHERENCE_ORDER), dtype=float)

    ax.axhspan(0.0, 0.25, color=explainer.ORANGE, alpha=0.07, lw=0)
    ax.axhspan(-0.25, 0.0, color=explainer.GRAY, alpha=0.08, lw=0)
    ax.axhline(0.0, color=explainer.INK, lw=0.9, ls=":", alpha=0.6)

    for population_key in PLOT_POPULATION_ORDER:
        sub = values[values["population_key"].astype(str).eq(population_key)].sort_values("coherence_bin")
        y = sub["observed_minus_rotated"].to_numpy(dtype=float)
        lo = sub["observed_minus_rotated_ci95_low"].to_numpy(dtype=float)
        hi = sub["observed_minus_rotated_ci95_high"].to_numpy(dtype=float)
        is_aligned = population_key == "high_sf_aligned"
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - lo, hi - y]),
            color=explainer.POPULATION_COLORS[population_key],
            marker=explainer.POPULATION_MARKERS[population_key],
            markersize=3.4,
            markerfacecolor="white",
            markeredgewidth=1.0,
            lw=2.0 if is_aligned else 1.5,
            capsize=1.8,
            zorder=4 if is_aligned else 3,
            label=SHORT_POPULATION_LABELS[population_key],
        )

    ax.set_xlim(-0.45, len(bridge.COHERENCE_ORDER) - 0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(bridge.COHERENCE_ORDER, fontsize=5.6, rotation=38, ha="right")
    ax.set_xlabel("local edge coherence", labelpad=1.5)
    ax.set_ylabel("observed − random rotated\n(pp SSI, RMS excursion)")
    ax.grid(axis="y", color=explainer.PALE_GRID, lw=0.75)
    ax.set_axisbelow(True)
    ax.set_title(
        f"{label}  {title}", loc="left", fontsize=7.4, fontweight="bold", pad=5, color=explainer.INK, linespacing=1.25
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=5.8,
        loc="lower left",
        handlelength=1.2,
        labelspacing=0.3,
        borderaxespad=0.2,
        handletextpad=0.4,
    )

    return values


def build_panel(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    values = load_values()
    values.to_csv(out_dir / "panel_j_match_advantage_values.csv", index=False)

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=False)
    draw_panel(ax, values=values)
    fig.tight_layout(pad=0.55)
    paths = {
        "png": out_dir / "panel_j_match_advantage.png",
        "pdf": out_dir / "panel_j_match_advantage.pdf",
        "svg": out_dir / "panel_j_match_advantage.svg",
    }
    fig.savefig(paths["png"], dpi=220, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    paths = build_panel()
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
