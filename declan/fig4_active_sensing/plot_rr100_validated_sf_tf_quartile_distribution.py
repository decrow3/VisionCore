#!/usr/bin/env python3
"""Plot validated RR100 preferred-SF/TF distributions by SF quartile."""
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
ASSIGNMENTS = ROOT / (
    "outputs/fig/ssi_figure_v2/corrected_sf_quartiles_clean_history_rounds000_022_v4/"
    "ssi_figure_v4_corrected_cache_sf_quartiles_clean_history_no_bottom_row_rounds000_022_v4_unit_assignments.csv"
)
MODELS = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/rr100_sf_tf_parametric_models.csv"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_validated_sf_tf_quartile_distribution_v1"
GROUPS = ("sf_q1", "sf_q2", "sf_q3", "sf_q4")
LABELS = {"sf_q1": "Q1 · lowest SF", "sf_q2": "Q2", "sf_q3": "Q3", "sf_q4": "Q4 · highest SF"}
COLORS = {"sf_q1": "#0072B2", "sf_q2": "#009E73", "sf_q3": "#E69F00", "sf_q4": "#CC79A7"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    assignments = pd.read_csv(ASSIGNMENTS)
    models = pd.read_csv(MODELS)
    extra = [
        "rr100_index", "sf_fit_support_min_cpd", "sf_fit_support_max_cpd",
        "tf_fit_support_min_hz", "tf_fit_support_max_hz", "tf_fit_r2",
        "tf_sampled_preferred_hz", "sf_sampled_preferred_cpd",
    ]
    frame = assignments.merge(models[extra], on="rr100_index", validate="one_to_one")
    frame = frame[frame.recorded_validation_pass.astype(bool) & frame.sf_quartile.isin(GROUPS)].copy()
    frame["sf_edge_preference"] = np.isclose(frame.preferred_sf_cpd, frame.sf_fit_support_min_cpd) | np.isclose(
        frame.preferred_sf_cpd, frame.sf_fit_support_max_cpd
    )
    frame["tf_edge_preference"] = np.isclose(frame.preferred_tf_hz, frame.tf_fit_support_min_hz) | np.isclose(
        frame.preferred_tf_hz, frame.tf_fit_support_max_hz
    )
    frame = frame.sort_values(["preferred_sf_cpd", "rr100_index"]).reset_index(drop=True)
    frame["validated_sf_rank"] = np.arange(1, len(frame) + 1)
    frame.to_csv(OUT / "validated_sf_tf_quartile_values.csv", index=False)

    summary = frame.groupby("sf_quartile", sort=False).agg(
        n=("rr100_index", "size"),
        sf_min_cpd=("preferred_sf_cpd", "min"),
        sf_median_cpd=("preferred_sf_cpd", "median"),
        sf_max_cpd=("preferred_sf_cpd", "max"),
        tf_min_hz=("preferred_tf_hz", "min"),
        tf_median_hz=("preferred_tf_hz", "median"),
        tf_max_hz=("preferred_tf_hz", "max"),
        sf_fit_r2_median=("sf_fit_r2", "median"),
        tf_fit_r2_median=("tf_fit_r2", "median"),
        joint_r2_median=("joint_parametric_surface_r2", "median"),
        recorded_sf_r_median=("recorded_sf_curve_r_full_support", "median"),
        n_sf_edge=("sf_edge_preference", "sum"),
        n_tf_edge=("tf_edge_preference", "sum"),
    ).reindex(GROUPS)
    summary.to_csv(OUT / "validated_sf_tf_quartile_summary.csv")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True)

    ax = axes[0, 0]
    for group in GROUPS:
        sub = frame[frame.sf_quartile.eq(group)]
        ax.scatter(
            sub.preferred_sf_cpd, sub.validated_sf_rank, s=34, color=COLORS[group],
            edgecolor="white", linewidth=0.5, label=f"{LABELS[group]} (n={len(sub)})", zorder=3,
        )
    for left, right in zip(GROUPS[:-1], GROUPS[1:]):
        boundary = np.sqrt(
            frame.loc[frame.sf_quartile.eq(left), "preferred_sf_cpd"].max()
            * frame.loc[frame.sf_quartile.eq(right), "preferred_sf_cpd"].min()
        )
        ax.axvline(boundary, color="0.35", lw=0.8, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 1.5, 2, 3, 4, 6], ["1", "1.5", "2", "3", "4", "6"])
    ax.set(xlabel="preferred SF (cycles/degree)", ylabel="validated-unit SF rank")
    ax.set_title("A  Preferred-SF ranks and quartile boundaries", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")

    ax = axes[0, 1]
    rng = np.random.default_rng(20260814)
    positions = np.arange(1, 5)
    values = [frame.loc[frame.sf_quartile.eq(group), "preferred_tf_hz"].to_numpy(float) for group in GROUPS]
    box = ax.boxplot(values, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch, group in zip(box["boxes"], GROUPS):
        patch.set(facecolor=COLORS[group], alpha=0.18, edgecolor=COLORS[group], linewidth=1.3)
    for item in box["medians"]:
        item.set(color="0.15", linewidth=1.3)
    for group, position, vals in zip(GROUPS, positions, values):
        jitter = rng.uniform(-0.17, 0.17, len(vals))
        sub = frame[frame.sf_quartile.eq(group)]
        edge = np.where(sub.tf_edge_preference.to_numpy(bool), "black", "white")
        ax.scatter(position + jitter, vals, s=31, color=COLORS[group], edgecolor=edge, linewidth=0.65, zorder=3)
    ax.set_yscale("log", base=2)
    ax.set_yticks([0.5, 1, 2, 4, 8, 16, 32], ["0.5", "1", "2", "4", "8", "16", "32"])
    ax.set_xticks(positions, ["Q1", "Q2", "Q3", "Q4"])
    ax.set(xlabel="preferred-SF quartile", ylabel="preferred TF (Hz)")
    ax.set_title("B  Preferred-TF distribution within SF quartiles", loc="left", weight="bold")
    ax.text(0.02, 0.03, "black edge: TF preference at fit-support boundary", transform=ax.transAxes, fontsize=7)

    ax = axes[1, 0]
    for group in GROUPS:
        sub = frame[frame.sf_quartile.eq(group)]
        ax.scatter(
            sub.preferred_sf_cpd, sub.preferred_tf_hz, s=45,
            facecolor=COLORS[group], edgecolor=np.where(sub.tf_edge_preference, "black", "white"),
            linewidth=0.7, alpha=0.92,
        )
        for row in sub.itertuples(index=False):
            ax.annotate(
                f"u{int(row.rr100_index):03d}", (row.preferred_sf_cpd, row.preferred_tf_hz),
                xytext=(3, 2), textcoords="offset points", fontsize=5.3, color="0.22",
            )
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks([1, 1.5, 2, 3, 4, 6], ["1", "1.5", "2", "3", "4", "6"])
    ax.set_yticks([0.5, 1, 2, 4, 8, 16, 32], ["0.5", "1", "2", "4", "8", "16", "32"])
    ax.set(xlabel="preferred SF (cycles/degree)", ylabel="preferred TF (Hz)")
    ax.set_title("C  Joint preferred-SF / preferred-TF distribution", loc="left", weight="bold")

    ax = axes[1, 1]
    for group in GROUPS:
        sub = frame[frame.sf_quartile.eq(group)]
        marker_sizes = 25 + 55 * np.clip(sub.joint_parametric_surface_r2.to_numpy(float), 0, 1)
        ax.scatter(
            sub.preferred_tf_hz, sub.tf_fit_r2, s=marker_sizes, color=COLORS[group],
            edgecolor=np.where(sub.tf_edge_preference, "black", "white"), linewidth=0.7,
            alpha=0.85, label=LABELS[group],
        )
    ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32], ["0.5", "1", "2", "4", "8", "16", "32"])
    ax.set(xlabel="preferred TF (Hz)", ylabel="TF factor fit $R^2$")
    ax.set_title("D  TF preference and fit quality", loc="left", weight="bold")
    ax.text(0.02, 0.04, "size: joint SF×TF surface $R^2$", transform=ax.transAxes, fontsize=7)

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="0.92", lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(
        "RR100 recorded-validated tuning preferences by preferred-SF quartile\n"
        "61 units · quartiles assigned only after recorded SF-curve validation",
        fontsize=14, weight="bold",
    )
    stem = "rr100_validated_sf_tf_quartile_distribution"
    fig.savefig(OUT / f"{stem}.png", dpi=220, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "validated_sf_tf_quartile_distribution_complete",
        "n_units": int(len(frame)),
        "sources": {"assignments": str(ASSIGNMENTS.resolve()), "models": str(MODELS.resolve())},
        "outputs": {
            "png": str((OUT / f"{stem}.png").resolve()),
            "pdf": str((OUT / f"{stem}.pdf").resolve()),
            "values": str((OUT / "validated_sf_tf_quartile_values.csv").resolve()),
            "summary": str((OUT / "validated_sf_tf_quartile_summary.csv").resolve()),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
