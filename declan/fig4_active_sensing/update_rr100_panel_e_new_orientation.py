#!/usr/bin/env python3
"""Regenerate Figure 4E with the new grating-based preferred orientations."""

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


ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "assembled/rounds_000_031_n032_clean_history_snapshot_v1"
)
ORIENTATION_AUDIT = ROOT / (
    "outputs/fig4_active_sensing/rr100_new_vs_legacy_orientation_audit_v1/"
    "unit_orientation_comparison.csv"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_panel_e_new_orientation_clean32_v1"


def component_stats(data: dict[str, object], mask: np.ndarray) -> dict[str, np.ndarray]:
    image_ids = data["condition"].image_index.to_numpy(int)
    values = figure.condition_sufficient_statistics(
        data["moving_info"], data["moving_spikes"], data["baseline_info"],
        data["baseline_spikes"], image_ids, mask,
    )
    return dict(zip(("info", "spikes", "base_info", "base_spikes", "valid"), values))


def selection_audit(
    data: dict[str, object], old_mask: np.ndarray, new_mask: np.ndarray, orientation: pd.DataFrame
) -> pd.DataFrame:
    units = data["units"]
    q4 = units[units.sf_quartile.eq(figure.COMPONENT_GROUP)][["rr100_index"]].merge(
        orientation, on="rr100_index", validate="one_to_one"
    )
    rows = []
    for row in q4.itertuples(index=False):
        unit = int(row.rr100_index)
        old_images = np.flatnonzero(old_mask[:, unit])
        new_images = np.flatnonzero(new_mask[:, unit])
        union = np.union1d(old_images, new_images)
        rows.append({
            "rr100_index": unit,
            "legacy_preferred_orientation_image_deg": row.legacy_preferred_orientation_image_deg,
            "new_preferred_orientation_image_deg": row.new_preferred_orientation_image_deg,
            "orientation_change_deg": row.converted_new_vs_legacy_delta_deg,
            "legacy_orientation_selectivity": row.legacy_orientation_selectivity,
            "new_orientation_selectivity": row.new_orientation_selectivity_recomputed,
            "legacy_passes_osi_gate": bool(row.legacy_orientation_selectivity >= figure.OSI_MIN),
            "new_passes_osi_gate": bool(row.new_orientation_selectivity_recomputed >= figure.OSI_MIN),
            "legacy_n_matched_images": len(old_images), "new_n_matched_images": len(new_images),
            "matched_image_jaccard": float(len(np.intersect1d(old_images, new_images)) / len(union)) if len(union) else np.nan,
            "legacy_matched_images": ";".join(map(str, old_images)),
            "new_matched_images": ";".join(map(str, new_images)),
        })
    return pd.DataFrame(rows)


def draw_comparison(old: pd.DataFrame, new: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True, sharey=True)
    styles = {"across": ("#D55E00", "-", "o"), "along": ("#D55E00", "--", "s")}
    for ax, table, title in zip(
        axes, (old, new),
        ("A  Legacy preferred-orientation gate", "B  New grating-based orientation gate"),
    ):
        ax.scatter([0], [0], s=22, facecolor="white", edgecolor="#D55E00", zorder=4)
        for component in ("across", "along"):
            sub = table[table.component.eq(component)].sort_values("bin")
            color, linestyle, marker = styles[component]
            yerr = np.vstack([sub.delta_percent - sub.ci95_low, sub.ci95_high - sub.delta_percent])
            ax.errorbar(
                figure.component_broken_log(sub.x_median), sub.delta_percent, yerr=yerr,
                color=color, ls=linestyle, marker=marker, mfc="white", mec=color,
                ms=4, lw=1.5, capsize=1.8, label=component,
            )
        ax.axhline(0, color="0.45", lw=0.7, ls=":")
        figure.format_broken_component_axis(ax)
        ax.set_xlabel("component RMS excursion (arcmin)")
        ax.set_title(title, loc="left", weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    axes[0].set_ylabel("SSI change (%) vs matched stabilized")
    fig.suptitle(
        "Figure 4E orientation-gate update · highest-SF quartile\n"
        "same 32-round clean-history responses; only preferred-orientation method and matched selectivity change",
        fontsize=12.5, weight="bold",
    )
    fig.savefig(path, dpi=220, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    figure.ASSEMBLED = ASSEMBLED
    data = figure.build_inputs()
    orientation = pd.read_csv(ORIENTATION_AUDIT)

    _, old_matched = figure.make_unit_masks(data)
    old_mask = old_matched[figure.COMPONENT_GROUP]
    old_table = figure.compute_component_table(data, component_stats(data, old_mask))
    old_table["orientation_method"] = "legacy"

    replacement = orientation[[
        "rr100_index", "new_preferred_orientation_image_deg", "new_orientation_selectivity_recomputed"
    ]]
    new_units = data["units"].drop(
        columns=["prior_preferred_orientation_deg", "prior_orientation_selectivity_index"]
    ).merge(replacement, on="rr100_index", validate="one_to_one")
    new_units = new_units.rename(columns={
        "new_preferred_orientation_image_deg": "prior_preferred_orientation_deg",
        "new_orientation_selectivity_recomputed": "prior_orientation_selectivity_index",
    })
    new_data = dict(data)
    new_data["units"] = new_units
    _, new_matched = figure.make_unit_masks(new_data)
    new_mask = new_matched[figure.COMPONENT_GROUP]
    new_table = figure.compute_component_table(new_data, component_stats(new_data, new_mask))
    new_table["orientation_method"] = "new_grating_based"

    values = pd.concat([old_table, new_table], ignore_index=True)
    values.to_csv(OUT / "panel_e_old_vs_new_orientation_values.csv", index=False)
    audit = selection_audit(data, old_mask, new_mask, orientation)
    audit.to_csv(OUT / "panel_e_q4_orientation_selection_audit.csv", index=False)

    # Figure-ready replacement panel and an explicit old/new diagnostic.
    figure.draw_component_panel(
        new_table, out_path=OUT / "panel_e_new_orientation.pdf", figsize=(3.20, 2.48)
    )
    draw_comparison(old_table, new_table, OUT / "panel_e_old_vs_new_orientation_comparison.png")

    paired = old_table.merge(
        new_table, on=["component", "bin", "context"], suffixes=("_old", "_new")
    )
    paired["delta_percent_change"] = paired.delta_percent_new - paired.delta_percent_old
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "panel_e_new_orientation_complete",
        "assembled_snapshot": str(ASSEMBLED.resolve()), "n_conditions": int(len(data["condition"])),
        "orientation_source": str(ORIENTATION_AUDIT.resolve()),
        "orientation_coordinate_conversion": "theta_image = (-theta_native) mod 180 degrees",
        "gate": {
            "sf_group": figure.COMPONENT_GROUP, "image_coherence_min": figure.COHERENCE_MIN,
            "orientation_selectivity_min": figure.OSI_MIN,
            "maximum_axial_orientation_delta_deg": figure.ORIENTATION_MATCH_MAX_DEG,
        },
        "q4_units": int(len(audit)),
        "q4_units_passing_osi_legacy": int(audit.legacy_passes_osi_gate.sum()),
        "q4_units_passing_osi_new": int(audit.new_passes_osi_gate.sum()),
        "median_orientation_change_deg": float(audit.orientation_change_deg.median()),
        "median_matched_image_jaccard": float(audit.matched_image_jaccard.median()),
        "median_absolute_curve_change_percentage_points": float(paired.delta_percent_change.abs().median()),
        "maximum_absolute_curve_change_percentage_points": float(paired.delta_percent_change.abs().max()),
        "outputs": {
            "replacement_panel_pdf": str((OUT / "panel_e_new_orientation.pdf").resolve()),
            "comparison_png": str((OUT / "panel_e_old_vs_new_orientation_comparison.png").resolve()),
            "comparison_pdf": str((OUT / "panel_e_old_vs_new_orientation_comparison.pdf").resolve()),
            "values": str((OUT / "panel_e_old_vs_new_orientation_values.csv").resolve()),
            "selection_audit": str((OUT / "panel_e_q4_orientation_selection_audit.csv").resolve()),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
