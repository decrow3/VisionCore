#!/usr/bin/env python3
"""Checkpoint 1 for unpacking Figure 4 B/D: inputs, gates, and weights."""
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

from declan.fig.ssi_figure_v2.compose_ssi_figure_v4_corrected_sf_quartiles import (
    COHERENCE_MIN,
    GROUPS,
    LABELS,
    COLORS,
    ORIENTATION_MATCH_MAX_DEG,
    OSI_MIN,
    axis_delta_deg,
)


ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "assembled/rounds_000_022_n023_clean_history_snapshot_v1"
)
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
ASSIGNMENTS = ROOT / (
    "outputs/fig/ssi_figure_v2/corrected_sf_quartiles_clean_history_rounds000_022_v2/"
    "ssi_figure_v4_corrected_cache_sf_quartiles_clean_history_no_bottom_row_rounds000_022_v2_unit_assignments.csv"
)
UNIT_TABLE = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged/unit_feature_table.csv"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_figure4_bd_estimand_anatomy_clean23_v1"
INK = "#161616"


def effective_count(weights: np.ndarray) -> float:
    values = np.asarray(weights, float)
    total = float(values.sum())
    return total * total / float(np.square(values).sum()) if total > 0 else float("nan")


def make_masks(images: pd.DataFrame, units: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    coherence = images.corrected_reconstruction_orientation_coherence.to_numpy(float)
    axes = images.corrected_reconstruction_contour_axis_deg.to_numpy(float)
    strong = np.isfinite(axes) & np.isfinite(coherence) & (coherence >= COHERENCE_MIN)
    output: dict[str, dict[str, np.ndarray]] = {"B_strong_contours": {}, "D_contour_matched": {}}
    for group in GROUPS:
        membership = units.sf_quartile.eq(group).to_numpy()
        b = np.zeros((100, 100), bool)
        d = np.zeros((100, 100), bool)
        b[strong] = membership
        for row in units[membership].itertuples(index=False):
            if (
                np.isfinite(row.prior_preferred_orientation_deg)
                and np.isfinite(row.prior_orientation_selectivity_index)
                and row.prior_orientation_selectivity_index >= OSI_MIN
            ):
                matched = strong & (
                    axis_delta_deg(axes, row.prior_preferred_orientation_deg)
                    <= ORIENTATION_MATCH_MAX_DEG
                )
                d[matched, int(row.unit_index)] = True
        output["B_strong_contours"][group] = b
        output["D_contour_matched"][group] = d
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ASSEMBLED / "manifest.json").read_text())
    condition = pd.read_csv(ASSEMBLED / "condition_index.csv")
    traces = pd.read_csv(COHORT / "corrected1000_traces.csv")
    condition = condition.merge(
        traces[["trace_index", "corrected_dpi_crop120_path_length_arcmin", "corrected_events_in_trial"]],
        on="trace_index",
        validate="many_to_one",
    )
    condition["context"] = np.where(condition.corrected_events_in_trial.gt(0), "microsaccade", "drift_only")
    images = pd.read_csv(COHORT / "corrected100_images.csv").sort_values("image_index").reset_index(drop=True)
    assignment = pd.read_csv(ASSIGNMENTS)
    unit_meta = pd.read_csv(UNIT_TABLE)[
        ["unit_index", "prior_preferred_orientation_deg", "prior_orientation_selectivity_index"]
    ]
    units = unit_meta.merge(
        assignment[["rr100_index", "preferred_sf_cpd", "sf_quartile", "sf_quartile_label"]],
        left_on="unit_index",
        right_on="rr100_index",
        validate="one_to_one",
    )
    masks = make_masks(images, units)
    moving_spikes = np.load(ASSEMBLED / "moving_expected_spikes.npy", mmap_mode="r")
    with np.load(ASSEMBLED / "stabilized_by_image_sufficient_statistics.npz") as data:
        baseline_spikes = np.asarray(data["expected_spikes"], float)
        baseline_ssi = np.asarray(data["movie_ssi_bits_per_spike"], float)

    anatomy_rows: list[dict[str, object]] = []
    for relation, relation_masks in masks.items():
        for group in GROUPS:
            mask = relation_masks[group]
            for image_index in range(100):
                ids = np.flatnonzero(mask[image_index])
                if len(ids) == 0:
                    continue
                base_w = baseline_spikes[image_index, ids]
                rows = np.flatnonzero(condition.image_index.to_numpy(int) == image_index)
                moving_w = np.asarray(moving_spikes[np.ix_(rows, ids)], float).mean(axis=0)
                combined = base_w + moving_w
                anatomy_rows.append(
                    {
                        "relation": relation,
                        "sf_quartile": group,
                        "image_index": image_index,
                        "n_eligible_units": len(ids),
                        "eligible_units": ";".join(map(str, ids)),
                        "baseline_expected_spikes": float(base_w.sum()),
                        "baseline_population_ssi": float(np.sum(base_w * baseline_ssi[image_index, ids]) / np.maximum(base_w.sum(), 1e-12)),
                        "effective_units_baseline_weights": effective_count(base_w),
                        "effective_units_combined_weights": effective_count(combined),
                        "largest_unit_combined_spike_share": float(combined.max() / np.maximum(combined.sum(), 1e-12)),
                        "n_conditions_for_image": len(rows),
                    }
                )
    anatomy = pd.DataFrame(anatomy_rows)

    balance_rows: list[dict[str, object]] = []
    x = condition.corrected_dpi_crop120_path_length_arcmin.to_numpy(float)
    image_ids = condition.image_index.to_numpy(int)
    context = condition.context.to_numpy(str)
    for relation, relation_masks in masks.items():
        for group in GROUPS:
            valid_images = relation_masks[group].any(axis=1)
            for context_name, n_bins in (("drift_only", 7), ("microsaccade", 3)):
                use = (context == context_name) & valid_images[image_ids] & np.isfinite(x)
                selected = np.flatnonzero(use)
                labels = pd.qcut(pd.Series(x[use]).rank(method="first"), n_bins, labels=False).to_numpy(int)
                for bin_index in range(n_bins):
                    rows = selected[labels == bin_index]
                    counts = pd.Series(image_ids[rows]).value_counts().sort_index()
                    shares = counts.to_numpy(float) / counts.sum()
                    balance_rows.append(
                        {
                            "relation": relation,
                            "sf_quartile": group,
                            "context": context_name,
                            "bin": bin_index,
                            "path_median_arcmin": float(np.median(x[rows])),
                            "n_conditions": len(rows),
                            "n_images": len(counts),
                            "effective_images_by_condition_count": float(1.0 / np.square(shares).sum()),
                            "largest_image_condition_share": float(shares.max()),
                            "min_conditions_per_image": int(counts.min()),
                            "max_conditions_per_image": int(counts.max()),
                        }
                    )
    balance = pd.DataFrame(balance_rows)
    anatomy.to_csv(OUT / "panel_bd_image_unit_weight_anatomy.csv", index=False)
    balance.to_csv(OUT / "panel_bd_bin_image_balance.csv", index=False)
    condition[["matrix_row_index", "round_index", "image_index", "trace_index", "context", "corrected_dpi_crop120_path_length_arcmin"]].to_csv(
        OUT / "panel_bd_condition_inputs.csv", index=False
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(13.2, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(grid[0, 0])
    ax.axis("off")
    ax.text(0, 1, "A  What changes from B to D", transform=ax.transAxes, va="top", weight="bold", fontsize=11)
    ax.text(
        0.01,
        0.84,
        "Panel B\nstrong-contour images\n+ every validated unit in each SF quartile\n\n"
        "Panel D\nsame image gate\n+ OSI ≥ 0.05\n+ preferred orientation within 22.5° of image contour\n\n"
        "Both panels\npool information and expected spikes first\nthen express moving SSI relative to the matched-image baseline",
        transform=ax.transAxes,
        va="top",
        linespacing=1.45,
    )

    def box_panel(axis: plt.Axes, column: str, title: str, ylabel: str) -> None:
        positions = np.arange(4)
        width = 0.32
        for offset, relation, hatch in ((-width / 2, "B_strong_contours", ""), (width / 2, "D_contour_matched", "///")):
            values = [
                anatomy.loc[(anatomy.relation == relation) & (anatomy.sf_quartile == group), column].dropna().to_numpy(float)
                for group in GROUPS
            ]
            bp = axis.boxplot(values, positions=positions + offset, widths=width * 0.82, patch_artist=True, showfliers=False)
            for patch, group in zip(bp["boxes"], GROUPS, strict=True):
                patch.set(facecolor=COLORS[group], alpha=0.22 if relation.startswith("B") else 0.58, hatch=hatch, edgecolor=COLORS[group])
            for element in ("whiskers", "caps", "medians"):
                for line in bp[element]:
                    line.set(color=INK, linewidth=0.8)
        axis.set_xticks(positions, ["Q1", "Q2", "Q3", "Q4"])
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", weight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(0.02, 0.98, "pale: B   hatched/darker: D", transform=axis.transAxes, va="top", fontsize=7)

    box_panel(fig.add_subplot(grid[0, 1]), "n_eligible_units", "B  Units available per image", "eligible units")
    box_panel(fig.add_subplot(grid[0, 2]), "effective_units_combined_weights", "C  Effective units after spike weighting", "effective unit count")
    box_panel(fig.add_subplot(grid[1, 0]), "largest_unit_combined_spike_share", "D  Largest unit's spike share", "largest share")
    box_panel(fig.add_subplot(grid[1, 1]), "baseline_expected_spikes", "E  Baseline denominator", "expected spikes per image")

    ax = fig.add_subplot(grid[1, 2])
    for relation, marker, linestyle in (("B_strong_contours", "o", "-"), ("D_contour_matched", "s", "--")):
        for group in GROUPS:
            sub = balance[(balance.relation == relation) & (balance.sf_quartile == group)]
            ax.scatter(
                sub.effective_images_by_condition_count,
                sub.largest_image_condition_share * 100,
                color=COLORS[group], marker=marker, facecolor="white" if relation.startswith("B") else COLORS[group],
                alpha=0.8, s=28, label=f"{relation[0]} {LABELS[group]}" if relation.endswith("contours") else None,
            )
    ax.set_xlabel("effective images represented in a plotted bin")
    ax.set_ylabel("largest image share of bin (%)")
    ax.set_title("F  Image balance of plotted bins", loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.98, "circles/open: B   squares/filled: D", transform=ax.transAxes, va="top", fontsize=7)
    fig.suptitle(
        "Figure 4 B/D estimand anatomy before inspecting response maps\n"
        "23 clean-history rounds · 13,271 conditions · 61 recorded-validated units",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(OUT / "figure4_bd_estimand_anatomy.png", dpi=210, facecolor="white")
    fig.savefig(OUT / "figure4_bd_estimand_anatomy.pdf", facecolor="white")
    plt.close(fig)

    summary = anatomy.groupby(["relation", "sf_quartile"]).agg(
        n_images=("image_index", "nunique"),
        median_eligible_units=("n_eligible_units", "median"),
        min_eligible_units=("n_eligible_units", "min"),
        max_eligible_units=("n_eligible_units", "max"),
        median_effective_units=("effective_units_combined_weights", "median"),
        median_largest_unit_share=("largest_unit_combined_spike_share", "median"),
        median_baseline_expected_spikes=("baseline_expected_spikes", "median"),
    ).reset_index()
    summary.to_csv(OUT / "panel_bd_estimand_anatomy_summary.csv", index=False)
    output_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "map_first_input_estimand_checkpoint_complete",
        "stage": "input_and_manipulation_before_raw_maps",
        "hypothesis": "D reflects orientation-conditioned within-unit response changes",
        "evidence_against_hypothesis": "D is dominated by changing unit/image composition, unstable denominators, or very low effective unit counts",
        "source_snapshot": str(ASSEMBLED),
        "n_conditions": int(manifest["n_conditions"]),
        "outputs": {
            "figure": str((OUT / "figure4_bd_estimand_anatomy.png").resolve()),
            "image_unit_weights": str((OUT / "panel_bd_image_unit_weight_anatomy.csv").resolve()),
            "bin_balance": str((OUT / "panel_bd_bin_image_balance.csv").resolve()),
            "summary": str((OUT / "panel_bd_estimand_anatomy_summary.csv").resolve()),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(output_manifest, indent=2))


if __name__ == "__main__":
    main()
