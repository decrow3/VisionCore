#!/usr/bin/env python3
"""Plot the new parametric preferred-SF distribution and validated median split."""
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


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "outputs/fig/ssi_figure_v2/corrected_sf_halves_clean_history_rounds000_022_v1/"
    "ssi_figure_v4_corrected_cache_sf_halves_clean_history_no_bottom_row_rounds000_022_v1_unit_assignments.csv"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_new_preferred_sf_distribution_v2"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#9AA0A6"
INK = "#171717"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(SOURCE)
    valid_model = frame[frame.model_valid.astype(bool)].copy()
    validated = valid_model[valid_model.recorded_validation_pass.astype(bool)].copy()
    rejected = valid_model[~valid_model.recorded_validation_pass.astype(bool)].copy()
    low = validated[validated.sf_quartile.eq("sf_low_half")].sort_values("preferred_sf_cpd")
    high = validated[validated.sf_quartile.eq("sf_high_half")].sort_values("preferred_sf_cpd")
    if (len(low), len(high)) != (31, 30):
        raise ValueError(f"Unexpected half sizes: {len(low)}/{len(high)}")
    low_max = float(low.preferred_sf_cpd.max())
    high_min = float(high.preferred_sf_cpd.min())
    if not low_max < high_min:
        raise ValueError("Median split crosses a preferred-SF tie")
    boundary = float(np.sqrt(low_max * high_min))
    bins = np.geomspace(
        float(valid_model.preferred_sf_cpd.min()) * 0.96,
        float(valid_model.preferred_sf_cpd.max()) * 1.04,
        15,
    )

    plotted = frame[[
        "rr100_index", "model_valid", "recorded_validation_pass", "preferred_sf_cpd",
        "preferred_tf_hz", "sf_quartile", "recorded_sf_curve_r_full_support",
        "joint_parametric_surface_r2",
    ]].copy()
    plotted["display_group"] = np.select(
        [
            ~plotted.model_valid.astype(bool),
            plotted.model_valid.astype(bool) & ~plotted.recorded_validation_pass.astype(bool),
            plotted.sf_quartile.eq("sf_low_half"),
            plotted.sf_quartile.eq("sf_high_half"),
        ],
        ["invalid_parametric_model", "failed_recorded_validation", "validated_low_half", "validated_high_half"],
        default="unclassified",
    )
    plotted.to_csv(OUT / "preferred_sf_distribution_values.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 7.2), constrained_layout=True, height_ratios=[1.0, 1.2])
    ax = axes[0]
    ax.hist(
        rejected.preferred_sf_cpd,
        bins=bins,
        histtype="step",
        color=GRAY,
        lw=1.7,
        hatch="///",
        label=f"model-valid, failed recorded gate (n={len(rejected)})",
    )
    ax.hist(
        [low.preferred_sf_cpd, high.preferred_sf_cpd],
        bins=bins,
        stacked=True,
        color=[BLUE, ORANGE],
        alpha=0.74,
        edgecolor="white",
        linewidth=0.7,
        label=[f"validated low half (n={len(low)})", f"validated high half (n={len(high)})"],
    )
    ax.axvline(boundary, color=INK, lw=1.3, ls="--", label=f"half boundary: {low_max:.2f} / {high_min:.2f} cpd")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 1.5, 2, 3, 4, 6, 8, 12], ["1", "1.5", "2", "3", "4", "6", "8", "12"])
    ax.set_ylabel("units")
    ax.set_title("A  Distribution and recorded-validation gate", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ordered = validated.sort_values(["preferred_sf_cpd", "rr100_index"]).reset_index(drop=True)
    ranks = np.arange(1, len(ordered) + 1)
    colors = ordered.sf_quartile.map({"sf_low_half": BLUE, "sf_high_half": ORANGE}).to_numpy()
    ax.hlines(ranks, 0.90, ordered.preferred_sf_cpd, color=colors, alpha=0.28, lw=0.8)
    ax.scatter(ordered.preferred_sf_cpd, ranks, c=colors, s=34, edgecolor="white", linewidth=0.55, zorder=3)
    ax.axvline(boundary, color=INK, lw=1.3, ls="--")
    boundary_rows = ordered.iloc[[len(low) - 1, len(low)]]
    annotation_styles = [
        {"xytext": (-8, -11), "ha": "right", "va": "top"},
        {"xytext": (8, 8), "ha": "left", "va": "bottom"},
    ]
    for rank, row, style in zip(
        [len(low), len(low) + 1], boundary_rows.itertuples(index=False), annotation_styles, strict=True
    ):
        ax.annotate(
            f"u{int(row.rr100_index):03d}: {row.preferred_sf_cpd:.2f}",
            (row.preferred_sf_cpd, rank),
            xytext=style["xytext"],
            textcoords="offset points",
            ha=style["ha"],
            va=style["va"],
            fontsize=8,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 1.5, 2, 3, 4, 6], ["1", "1.5", "2", "3", "4", "6"])
    ax.set_xlabel("new parametric preferred spatial frequency (cycles/degree)")
    ax.set_ylabel("validated-unit rank")
    ax.set_title("B  Recorded-validated units sorted by preferred SF", loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "RR100 new preferred-SF distribution\n"
        "85 model-valid fits · 61 pass recorded SF-curve r ≥ 0.5 · median halves 31/30",
        fontsize=14,
        weight="bold",
    )
    stem = "rr100_new_preferred_sf_distribution_recorded_validated_halves_v2"
    fig.savefig(OUT / f"{stem}.png", dpi=220, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", facecolor="white")
    plt.close(fig)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preferred_sf_distribution_complete",
        "source": str(SOURCE.resolve()),
        "n_total_units": len(frame),
        "n_model_valid": len(valid_model),
        "n_failed_recorded_gate": len(rejected),
        "n_recorded_validated": len(validated),
        "n_low_half": len(low),
        "n_high_half": len(high),
        "validated_range_cpd": [float(validated.preferred_sf_cpd.min()), float(validated.preferred_sf_cpd.max())],
        "validated_median_cpd": float(validated.preferred_sf_cpd.median()),
        "low_half_max_cpd": low_max,
        "high_half_min_cpd": high_min,
        "display_boundary_cpd_geometric_midpoint": boundary,
        "outputs": {
            "png": str((OUT / f"{stem}.png").resolve()),
            "pdf": str((OUT / f"{stem}.pdf").resolve()),
            "values": str((OUT / "preferred_sf_distribution_values.csv").resolve()),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
