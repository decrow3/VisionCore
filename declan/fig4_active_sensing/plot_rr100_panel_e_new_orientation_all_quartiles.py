#!/usr/bin/env python3
"""Plot Figure-4E component curves for every SF quartile using new orientations."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2 import compose_ssi_figure_v4_corrected_sf_quartiles as figure
from declan.fig4_active_sensing.update_rr100_panel_e_new_orientation import (
    ASSEMBLED,
    ORIENTATION_AUDIT,
    component_stats,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fig4_active_sensing/rr100_panel_e_new_orientation_all_quartiles_clean32_v1"
GROUPS = ("sf_q1", "sf_q2", "sf_q3", "sf_q4")
LABELS = {"sf_q1": "Q1 · lowest SF", "sf_q2": "Q2", "sf_q3": "Q3", "sf_q4": "Q4 · highest SF"}
COLORS = {"sf_q1": "#0072B2", "sf_q2": "#009E73", "sf_q3": "#E69F00", "sf_q4": "#D55E00"}


def new_orientation_data() -> tuple[dict[str, object], pd.DataFrame]:
    figure.ASSEMBLED = ASSEMBLED
    data = figure.build_inputs()
    orientation = pd.read_csv(ORIENTATION_AUDIT)
    replacement = orientation[[
        "rr100_index", "new_preferred_orientation_image_deg", "new_orientation_selectivity_recomputed"
    ]]
    units = data["units"].drop(
        columns=["prior_preferred_orientation_deg", "prior_orientation_selectivity_index"]
    ).merge(replacement, on="rr100_index", validate="one_to_one")
    units = units.rename(columns={
        "new_preferred_orientation_image_deg": "prior_preferred_orientation_deg",
        "new_orientation_selectivity_recomputed": "prior_orientation_selectivity_index",
    })
    data["units"] = units
    return data, orientation


def draw(tables: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), constrained_layout=True, sharex=True, sharey=True)
    for panel, (ax, group) in enumerate(zip(axes.flat, GROUPS), start=1):
        color = COLORS[group]
        ax.scatter([0], [0], marker="o", s=25, facecolor="white", edgecolor=color, lw=1.2, zorder=5)
        for component, linestyle, marker in (("across", "-", "o"), ("along", "--", "s")):
            sub = tables[tables.sf_quartile.eq(group) & tables.component.eq(component)].sort_values("bin")
            yerr = np.vstack([sub.delta_percent - sub.ci95_low, sub.ci95_high - sub.delta_percent])
            ax.errorbar(
                figure.component_broken_log(sub.x_median), sub.delta_percent, yerr=yerr,
                color=color, ls=linestyle, marker=marker, mfc="white", mec=color,
                ms=4.2, lw=1.55, capsize=1.8, label=component,
            )
        ax.axhline(0, color="0.45", lw=0.7, ls=":")
        figure.format_broken_component_axis(ax)
        ax.set_title(f"{chr(64 + panel)}  {LABELS[group]}", loc="left", weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="best")
    for ax in axes[:, 0]:
        ax.set_ylabel("SSI change (%) vs matched stabilized")
    for ax in axes[1]:
        ax.set_xlabel("component RMS excursion (arcmin)")
    fig.suptitle(
        "Contour-component excursion curves across preferred-SF quartiles\n"
        "new grating-based orientation preference and selectivity · same 32-round clean-history responses",
        fontsize=13.5, weight="bold",
    )
    stem = "panel_e_new_orientation_all_sf_quartiles"
    fig.savefig(OUT / f"{stem}.png", dpi=230, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    data, orientation = new_orientation_data()
    _, matched = figure.make_unit_masks(data)
    tables = []
    selection_rows = []
    image_axis = data["images"].corrected_reconstruction_contour_axis_deg.to_numpy(float)
    coherence = data["images"].corrected_reconstruction_orientation_coherence.to_numpy(float)
    strong = np.isfinite(image_axis) & np.isfinite(coherence) & (coherence >= figure.COHERENCE_MIN)
    orientation_index = orientation.set_index("rr100_index")
    for group in GROUPS:
        table = figure.compute_component_table(data, component_stats(data, matched[group]))
        table["sf_quartile"] = group
        tables.append(table)
        members = data["units"][data["units"].sf_quartile.eq(group)]
        for row in members.itertuples(index=False):
            unit = int(row.rr100_index)
            source = orientation_index.loc[unit]
            selected = np.flatnonzero(matched[group][:, int(row.unit_index)])
            selection_rows.append({
                "sf_quartile": group, "rr100_index": unit,
                "preferred_orientation_image_deg": float(row.prior_preferred_orientation_deg),
                "orientation_selectivity": float(row.prior_orientation_selectivity_index),
                "passes_osi_gate": bool(row.prior_orientation_selectivity_index >= figure.OSI_MIN),
                "n_matched_strong_images": int(len(selected)),
                "matched_strong_images": ";".join(map(str, selected)),
                "legacy_to_new_orientation_change_deg": float(source.converted_new_vs_legacy_delta_deg),
            })
    values = pd.concat(tables, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    values.to_csv(OUT / "panel_e_all_quartile_component_values.csv", index=False)
    selections.to_csv(OUT / "panel_e_all_quartile_orientation_selections.csv", index=False)
    summary = selections.groupby("sf_quartile", sort=False).agg(
        n_units=("rr100_index", "size"), n_units_passing_osi=("passes_osi_gate", "sum"),
        median_orientation_change_deg=("legacy_to_new_orientation_change_deg", "median"),
        total_matched_image_unit_pairs=("n_matched_strong_images", "sum"),
        median_matched_images_per_unit=("n_matched_strong_images", "median"),
    ).reindex(GROUPS)
    summary.to_csv(OUT / "panel_e_all_quartile_selection_summary.csv")
    draw(values)
    stem = "panel_e_new_orientation_all_sf_quartiles"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "all_quartile_new_orientation_component_checkpoint_complete",
        "assembled_snapshot": str(ASSEMBLED.resolve()), "n_conditions": int(len(data["condition"])),
        "orientation_source": str(ORIENTATION_AUDIT.resolve()),
        "gate": {
            "image_coherence_min": figure.COHERENCE_MIN,
            "orientation_selectivity_min": figure.OSI_MIN,
            "maximum_axial_orientation_delta_deg": figure.ORIENTATION_MATCH_MAX_DEG,
        },
        "outputs": {
            "png": str((OUT / f"{stem}.png").resolve()), "pdf": str((OUT / f"{stem}.pdf").resolve()),
            "values": str((OUT / "panel_e_all_quartile_component_values.csv").resolve()),
            "selections": str((OUT / "panel_e_all_quartile_orientation_selections.csv").resolve()),
            "selection_summary": str((OUT / "panel_e_all_quartile_selection_summary.csv").resolve()),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
