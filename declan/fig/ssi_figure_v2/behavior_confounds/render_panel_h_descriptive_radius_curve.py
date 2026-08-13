#!/usr/bin/env python3
"""Render a clean descriptive Figure 4H radius curve.

This presentation panel retains the original Figure 4H estimand: the slope of
fixation-centered trajectory--contour alignment versus local orientation
coherence (coherence > 0.3) at each patch radius.  It uses the audited
hierarchical session/trial estimates but intentionally does not display the
trajectory-reassignment or same-image offset controls.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
SOURCE_DIR = (
    ROOT
    / "outputs"
    / "fig"
    / "ssi_figure_v2"
    / "behavior_confounds_map_first_v1"
    / "panel_h_pairing_locality_radius_population_v1"
)
SOURCE_CSV = SOURCE_DIR / "panel_h_hierarchical_slope_curves.csv"
OUT_DIR = SOURCE_DIR / "descriptive_panel_h_v1"

INK = "#202124"
CURVE = "#4D5C68"
INTERVAL = "#AAB5BD"
GRID = "#E5E7E9"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_curve() -> pd.DataFrame:
    values = pd.read_csv(SOURCE_CSV)
    mask = values["scope"].eq("grand_equal_subject") & values["response"].eq(
        "observed_local_cos2"
    )
    curve = values.loc[
        mask,
        [
            "patch_radius_deg",
            "slope_vs_coherence",
            "ci95_low",
            "ci95_high",
            "coherence_min",
        ],
    ].copy()
    curve = curve.sort_values("patch_radius_deg").reset_index(drop=True)
    if curve.empty:
        raise ValueError("No grand observed-local slope curve found")
    if curve["patch_radius_deg"].duplicated().any():
        raise ValueError("Expected one pooled estimate per patch radius")
    return curve


def draw_panel(ax: plt.Axes, curve: pd.DataFrame) -> None:
    x = curve["patch_radius_deg"].to_numpy(dtype=float)
    y = curve["slope_vs_coherence"].to_numpy(dtype=float)
    lo = curve["ci95_low"].to_numpy(dtype=float)
    hi = curve["ci95_high"].to_numpy(dtype=float)

    ax.errorbar(
        x,
        y,
        yerr=np.vstack([y - lo, hi - y]),
        fmt="none",
        ecolor=INTERVAL,
        elinewidth=1.15,
        capsize=0,
        zorder=1,
    )
    ax.plot(x, y, color=CURVE, lw=1.65, zorder=2)
    ax.scatter(
        x,
        y,
        s=25,
        facecolor=CURVE,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )
    ax.axhline(0, color="#777777", lw=0.8, ls=":", alpha=0.8, zorder=0)

    ax.set_xlim(0.15, 3.1)
    ax.set_ylim(min(-0.18, float(np.nanmin(lo)) - 0.04), float(np.nanmax(hi)) + 0.06)
    ax.set_xticks([0.25, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_xticklabels(["0.25", "0.5", "1", "1.5", "2", "2.5", "3"])
    ax.set_xlabel("Fixation-centered patch radius (deg)")
    ax.set_ylabel("Alignment/coherence slope")
    ax.set_title(
        "Alignment strengthens near\nfoveal scale",
        loc="left",
        color=INK,
        fontweight="semibold",
        linespacing=1.15,
        pad=8,
    )
    ax.text(
        0.02,
        0.98,
        "coherence > 0.3",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#6B6F75",
    )
    ax.text(
        0.98,
        0.02,
        "95% hierarchical bootstrap CI",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.0,
        color="#6B6F75",
    )
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    configure_matplotlib()
    curve = load_curve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    values_path = OUT_DIR / "panel_h_descriptive_radius_curve_values.csv"
    curve.to_csv(values_path, index=False)

    fig, ax = plt.subplots(figsize=(3.15, 3.0), constrained_layout=True)
    draw_panel(ax, curve)
    outputs = {}
    for suffix, kwargs in (
        ("png", {"dpi": 300}),
        ("pdf", {}),
        ("svg", {}),
    ):
        path = OUT_DIR / f"panel_h_descriptive_radius_curve.{suffix}"
        fig.savefig(path, transparent=True, **kwargs)
        outputs[suffix] = str(path.relative_to(ROOT))
    plt.close(fig)

    provenance = {
        "panel": "Figure 4H descriptive candidate",
        "source_table": str(SOURCE_CSV.relative_to(ROOT)),
        "values_table": str(values_path.relative_to(ROOT)),
        "displayed_response": "observed_local_cos2",
        "displayed_scope": "grand_equal_subject",
        "estimand": (
            "Slope of real fixation-centered trajectory-contour axial alignment "
            "versus local orientation coherence, fit for coherence > 0.3"
        ),
        "uncertainty": (
            "95% hierarchical bootstrap interval over sessions and trials, "
            "with Allen and Logan equally weighted"
        ),
        "controls_intentionally_not_displayed": [
            "matched trajectory reassignment",
            "same-image offset patches",
        ],
        "outputs": outputs,
    }
    provenance_path = OUT_DIR / "panel_h_descriptive_radius_curve_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for path in outputs.values():
        print(ROOT / path)
    print(values_path)
    print(provenance_path)


if __name__ == "__main__":
    main()
