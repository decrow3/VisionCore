#!/usr/bin/env python3
"""Regenerate key BackImage population results using low/high SF halves."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from declan.active_sensing_movie_information import (
    plot_backimage_real_trace_unit_first_and_population_schematics as schematic,
)
from declan.fig4_active_sensing.rerun_backimage_all_images_component_sf_quartiles import (
    build_component_summary,
    prepare_component_inputs,
)
from declan.fig4_active_sensing.rerun_backimage_all_images_population_sf_quartiles import (
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    N_DRIFT_BINS,
    N_MICROSACCADE_BINS,
)
from declan.fig4_active_sensing.rerun_backimage_real_trace_contour_matched_sf_quartiles import (
    DEFAULT_ASSIGNMENTS,
    DEFAULT_MATRIX_DIR,
    ROOT,
    configure_matplotlib,
    file_identity,
    save_figure,
)
from declan.fig4_active_sensing.rerun_backimage_strong_contours_population_sf_quartiles import (
    build_summary,
)


DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/backimage_real_trace_sf_half_checks_v1"
)
GROUPS = ("sf_low_half", "sf_high_half")
LABELS = {
    "sf_low_half": "Low SF half",
    "sf_high_half": "High SF half",
}
COLORS = {
    "sf_low_half": "#3366A8",
    "sf_high_half": "#D55E00",
}
RELATIONS = (
    ("all_images_no_osi", "all image windows; no OSI gate", "All images"),
    ("strong_contours_no_osi", "strong contour image windows; no OSI gate", "Strong contours"),
    ("contour_matched", "strong contour image windows; orientation aligned", "Aligned"),
    ("contour_intermediate", "strong contour image windows; orientation intermediate", "Intermediate"),
    ("contour_orthogonal", "strong contour image windows; orientation orthogonal", "Orthogonal"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--assignments-csv", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    return parser.parse_args()


def prepare_half_units(
    unit: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    columns = [
        "rr100_index", "model_valid", "preferred_sf_cpd", "preferred_tf_hz",
        "joint_parametric_surface_r2",
    ]
    assigned = assignments[columns].copy()
    valid = assigned[assigned["model_valid"].fillna(False).astype(bool)].copy()
    if len(valid) != 85:
        raise ValueError(f"Expected 85 valid parametric fits, found {len(valid)}")
    threshold = float(valid["preferred_sf_cpd"].median())
    assigned["sf_half"] = "invalid_model"
    valid_mask = assigned["model_valid"].fillna(False).astype(bool)
    assigned.loc[valid_mask & (assigned["preferred_sf_cpd"] <= threshold), "sf_half"] = "sf_low_half"
    assigned.loc[valid_mask & (assigned["preferred_sf_cpd"] > threshold), "sf_half"] = "sf_high_half"
    assigned["sf_half_label"] = assigned["sf_half"].map(LABELS).fillna("invalid model")

    boundary = valid.sort_values(["preferred_sf_cpd", "rr100_index"]).iloc[40:46].copy()
    low_max = float(valid.loc[valid["preferred_sf_cpd"] <= threshold, "preferred_sf_cpd"].max())
    high_min = float(valid.loc[valid["preferred_sf_cpd"] > threshold, "preferred_sf_cpd"].min())
    if not low_max < high_min:
        raise ValueError("Median boundary would split an exact preferred-SF tie")

    out = unit.merge(assigned, left_on="unit_index", right_on="rr100_index", how="left", validate="one_to_one")
    if out["rr100_index"].isna().any():
        raise ValueError("Some matrix units have no parametric-fit assignment")
    out = out[out["sf_half"].isin(GROUPS)].copy()
    out["historical_sf_group"] = out["sf_group"]
    out["sf_group"] = out["sf_half"]
    out["sf_group_label"] = out["sf_half_label"]
    out["sf_group_definition"] = "halves of valid joint-parametric preferred SF; median retained in low half"
    out["sf_split_metric"] = out["preferred_sf_cpd"]

    audit = assigned.copy()
    audit["median_threshold_cpd"] = threshold
    audit["low_half_max_cpd"] = low_max
    audit["high_half_min_cpd"] = high_min
    audit["boundary_audit"] = audit["rr100_index"].isin(boundary["rr100_index"])
    return out, audit, threshold


def _style_axes(ax: plt.Axes) -> None:
    ax.axhline(0, color="0.45", ls=":", lw=0.9)
    ax.grid(True, color="0.92", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_population_context_grid(
    summaries: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(17.5, 7.2), sharex=True, sharey=True)
    all_y = pd.concat(list(summaries.values()), ignore_index=True)["population_ssi_delta_vs_stabilized"]
    pad = max(0.004, 0.08 * float(all_y.max() - all_y.min()))
    ylim = (float(all_y.min()) - pad, float(all_y.max()) + pad)
    for row_idx, group in enumerate(GROUPS):
        color = COLORS[group]
        for col_idx, (relation, _, short_label) in enumerate(RELATIONS):
            ax = axes[row_idx, col_idx]
            data = summaries[relation]
            for context, filled in (("drift_only", False), ("microsaccade", True)):
                sub = data[data["sf_group"].eq(group) & data["context"].eq(context)].sort_values("path_median_arcmin")
                x = sub["path_median_arcmin"].to_numpy(float)
                y = sub["population_ssi_delta_vs_stabilized"].to_numpy(float)
                lo = sub["population_delta_ci95_low_image_boot"].to_numpy(float)
                hi = sub["population_delta_ci95_high_image_boot"].to_numpy(float)
                ax.fill_between(x, lo, hi, color=color, alpha=0.07, linewidth=0)
                ax.plot(
                    x, y, color=color, lw=2.0, marker="o", ms=4.5,
                    markerfacecolor=color if filled else "white", markeredgewidth=1.3,
                )
            _style_axes(ax)
            ax.set_ylim(*ylim)
            if row_idx == 0:
                ax.set_title(short_label, fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{LABELS[group]}\nSSI minus stabilized\n(bits/spike)")
            if row_idx == 1:
                ax.set_xlabel("trace path (arcmin)")
    legend = [
        Line2D([0], [0], color="0.25", marker="o", markerfacecolor="white", label="drift-only"),
        Line2D([0], [0], color="0.25", marker="o", markerfacecolor="0.25", label=">=1 microsaccade"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        "Population movement modulation across image and unit-contour contexts",
        fontsize=14, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.955,
        "Low/high halves of 85 valid parametric SF fits; ribbons are 95% image-bootstrap intervals",
        ha="center", va="top", fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.935), w_pad=1.0, h_pad=1.2)
    save_figure(fig, out_dir, "key_01_population_contexts_low_high_sf_halves")


def plot_component_context_grid(
    summaries: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(17.5, 7.2), sharex=True, sharey=True)
    all_y = pd.concat(list(summaries.values()), ignore_index=True)["population_ssi_delta_vs_stabilized"]
    pad = max(0.004, 0.08 * float(all_y.max() - all_y.min()))
    ylim = (float(all_y.min()) - pad, float(all_y.max()) + pad)
    styles = {
        "across_path_arcmin": ("-", "o"),
        "along_path_arcmin": ("--", "s"),
    }
    for row_idx, group in enumerate(GROUPS):
        color = COLORS[group]
        for col_idx, (relation, _, short_label) in enumerate(RELATIONS):
            ax = axes[row_idx, col_idx]
            data = summaries[relation]
            for metric, (linestyle, marker) in styles.items():
                for context, filled in (("drift_only", False), ("microsaccade", True)):
                    sub = data[
                        data["sf_group"].eq(group)
                        & data["component_metric"].eq(metric)
                        & data["context"].eq(context)
                    ].sort_values("component_median_arcmin")
                    x = sub["component_median_arcmin"].to_numpy(float)
                    y = sub["population_ssi_delta_vs_stabilized"].to_numpy(float)
                    ax.plot(
                        x, y, color=color, ls=linestyle, marker=marker, lw=1.9, ms=4.2,
                        markerfacecolor=color if filled else "white", markeredgewidth=1.2,
                    )
            _style_axes(ax)
            ax.set_ylim(*ylim)
            if row_idx == 0:
                ax.set_title(short_label, fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{LABELS[group]}\nSSI minus stabilized\n(bits/spike)")
            if row_idx == 1:
                ax.set_xlabel("component path (arcmin)")
    legend = [
        Line2D([0], [0], color="0.25", ls="-", marker="o", markerfacecolor="white", label="across, drift-only"),
        Line2D([0], [0], color="0.25", ls="--", marker="s", markerfacecolor="white", label="along, drift-only"),
        Line2D([0], [0], color="0.25", ls="-", marker="o", markerfacecolor="0.25", label="across, microsaccade"),
        Line2D([0], [0], color="0.25", ls="--", marker="s", markerfacecolor="0.25", label="along, microsaccade"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.005), fontsize=8.5)
    fig.suptitle(
        "Across- and along-contour component modulation across contexts",
        fontsize=14, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.955,
        "Low/high halves of 85 valid parametric SF fits; components are defined relative to each image contour",
        ha="center", va="top", fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.935), w_pad=1.0, h_pad=1.2)
    save_figure(fig, out_dir, "key_02_component_contexts_low_high_sf_halves")


def plot_weighting_grid(
    populations: dict[str, pd.DataFrame],
    unit_summaries: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(17.5, 7.2), sharex=True, sharey=True)
    values = []
    for relation in populations:
        values.extend(populations[relation]["population_ssi_delta_vs_stabilized"].tolist())
        values.extend(unit_summaries[relation]["mean_unit_ssi_delta_vs_stabilized"].tolist())
    pad = max(0.004, 0.08 * (max(values) - min(values)))
    ylim = (min(values) - pad, max(values) + pad)
    for row_idx, group in enumerate(GROUPS):
        color = COLORS[group]
        for col_idx, (relation, _, short_label) in enumerate(RELATIONS):
            ax = axes[row_idx, col_idx]
            for context, filled in (("drift_only", False), ("microsaccade", True)):
                pop = populations[relation]
                pop = pop[pop["sf_group"].eq(group) & pop["context"].eq(context)].sort_values("path_median_arcmin")
                unit = unit_summaries[relation]
                unit = unit[unit["sf_group"].eq(group) & unit["context"].eq(context)].sort_values("path_median_arcmin")
                ax.plot(
                    pop["path_median_arcmin"], pop["population_ssi_delta_vs_stabilized"],
                    color=color, lw=2.0, marker="o", ms=4.2,
                    markerfacecolor=color if filled else "white",
                )
                ax.plot(
                    unit["path_median_arcmin"], unit["mean_unit_ssi_delta_vs_stabilized"],
                    color="0.25", ls="--", lw=1.6, marker="s", ms=3.8,
                    markerfacecolor="0.25" if filled else "white",
                )
            _style_axes(ax)
            ax.set_ylim(*ylim)
            if row_idx == 0:
                ax.set_title(short_label, fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{LABELS[group]}\nSSI minus stabilized\n(bits/spike)")
            if row_idx == 1:
                ax.set_xlabel("trace path (arcmin)")
    legend = [
        Line2D([0], [0], color="#777777", lw=2, marker="o", label="spike-weighted population"),
        Line2D([0], [0], color="0.25", ls="--", marker="s", label="equal-unit mean"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Population-weighting audit across key contexts", fontsize=14, fontweight="bold", y=0.99)
    fig.text(
        0.5, 0.955,
        "Group-colored circles: spike-weighted population; charcoal dashed squares: equal-unit mean",
        ha="center", va="top", fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.935), w_pad=1.0, h_pad=1.2)
    save_figure(fig, out_dir, "key_03_population_weighting_low_high_sf_halves")


def endpoint_table(
    populations: dict[str, pd.DataFrame],
    components: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for relation, _, relation_label in RELATIONS:
        pop = populations[relation]
        for group in GROUPS:
            for context, path_bin in (("drift_only", "drift_only_q01"), ("microsaccade", "microsaccade_q01")):
                row = pop[pop["sf_group"].eq(group) & pop["path_bin"].eq(path_bin)].iloc[0]
                rows.append({
                    "relation": relation, "relation_label": relation_label, "sf_half": group,
                    "summary_kind": "total_path", "context": context, "component": "total",
                    "path_median_arcmin": float(row["path_median_arcmin"]),
                    "delta_vs_stabilized": float(row["population_ssi_delta_vs_stabilized"]),
                    "ci95_low": float(row["population_delta_ci95_low_image_boot"]),
                    "ci95_high": float(row["population_delta_ci95_high_image_boot"]),
                })
            comp = components[relation]
            for metric in ("across_path_arcmin", "along_path_arcmin"):
                row = comp[
                    comp["sf_group"].eq(group)
                    & comp["component_metric"].eq(metric)
                    & comp["component_bin"].eq("drift_only_q01")
                ].iloc[0]
                rows.append({
                    "relation": relation, "relation_label": relation_label, "sf_half": group,
                    "summary_kind": "component_path", "context": "drift_only", "component": metric,
                    "path_median_arcmin": float(row["component_median_arcmin"]),
                    "delta_vs_stabilized": float(row["population_ssi_delta_vs_stabilized"]),
                    "ci95_low": float(row["population_delta_ci95_low_image_boot"]),
                    "ci95_high": float(row["population_delta_ci95_high_image_boot"]),
                })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    matrix_dir = Path(args.matrix_dir)
    data = schematic.load_dataset(matrix_dir)
    assignments = pd.read_csv(args.assignments_csv)
    unit_half, assignment_audit, threshold = prepare_half_units(data["unit"], assignments)
    counts = unit_half["sf_group"].value_counts().reindex(GROUPS, fill_value=0)
    if counts.to_dict() != {"sf_low_half": 43, "sf_high_half": 42}:
        raise ValueError(f"Unexpected half counts: {counts.to_dict()}")

    trace, trace_bins = schematic.add_equal_count_trace_bins(
        data["trace"], n_drift_bins=N_DRIFT_BINS,
        n_microsaccade_bins=N_MICROSACCADE_BINS,
    )
    component_metrics, bins_by_metric, component_bins = prepare_component_inputs(data)
    row_grid = schematic.build_movie_row_grid(data["movie"])
    baseline_lookup = schematic.baseline_rows_by_image(data["image"], data["baseline_table"])

    schematic.SF_COLORS.update(COLORS)
    schematic.SF_LABELS.update(LABELS)
    populations: dict[str, pd.DataFrame] = {}
    unit_summaries: dict[str, pd.DataFrame] = {}
    components: dict[str, pd.DataFrame] = {}
    selections = []
    component_selections = []
    unit_curves_all = []

    for relation, relation_label, short_label in RELATIONS:
        relation_dir = args.out_dir / relation
        relation_dir.mkdir(parents=True, exist_ok=True)
        selection, unit_curves, unit_summary, population = build_summary(
            data, unit_half, list(GROUPS), trace, trace_bins, row_grid,
            baseline_lookup, n_bootstrap=int(args.n_bootstrap), relation=relation,
            relation_label=relation_label,
        )
        component_selection, component_summary = build_component_summary(
            data, unit_half, list(GROUPS), component_metrics, bins_by_metric,
            row_grid, baseline_lookup, n_bootstrap=int(args.n_bootstrap),
            relation=relation, relation_label=relation_label,
        )
        populations[relation] = population
        unit_summaries[relation] = unit_summary
        components[relation] = component_summary
        selections.append(selection)
        component_selections.append(component_selection.assign(relation=relation))
        unit_curves_all.append(unit_curves)

        schematic.plot_sf_rows_population_panel(
            population, relation_dir,
            stem=f"{relation}_low_high_sf_halves_population_absolute_delta",
            title=f"Spike-weighted population SSI - {short_label}",
            subtitle=(
                f"low/high halves of valid parametric SF fits (n=43/42; split {threshold:.4f} cpd); "
                f"{N_DRIFT_BINS} drift-only / {N_MICROSACCADE_BINS} microsaccade bins"
            ),
            sf_groups=list(GROUPS), dpi=220,
        )
        schematic.plot_component_population_12_panel(
            component_summary, relation_dir,
            stem=f"{relation}_low_high_sf_halves_across_along_components",
            title=f"Spike-weighted population SSI components - {short_label}",
            sf_groups=list(GROUPS), dpi=220,
        )
        population.to_csv(relation_dir / "spike_weighted_population_summary.csv", index=False)
        unit_summary.to_csv(relation_dir / "unit_first_summary.csv", index=False)
        component_summary.to_csv(relation_dir / "spike_weighted_population_component_summary.csv", index=False)
        selection.to_csv(relation_dir / "unit_image_selection.csv", index=False)

    plot_population_context_grid(populations, args.out_dir)
    plot_component_context_grid(components, args.out_dir)
    plot_weighting_grid(populations, unit_summaries, args.out_dir)
    schematic.plot_sf_rows_population_panel(
        populations["contour_matched"], args.out_dir,
        stem="key_04_mixed_context_low_high_sf_halves",
        title="Low/high SF halves: spike-weighted population SSI",
        subtitle=(
            "left: all image windows, no OSI gate; right: strong-contour, orientation-aligned unit-image pairs; "
            f"median split {threshold:.4f} cpd"
        ),
        sf_groups=list(GROUPS), dpi=220,
        absolute_summary=populations["all_images_no_osi"],
        modulation_summary=populations["contour_matched"],
        absolute_column_title="Absolute SSI\nall images, no OSI gate",
        modulation_column_title="Movement modulation\nstrong contours, orientation-aligned",
    )

    assignment_audit.to_csv(args.out_dir / "sf_half_unit_assignments.csv", index=False)
    trace_bins.to_csv(args.out_dir / "trace_path_bin_definitions.csv", index=False)
    component_bins.to_csv(args.out_dir / "component_path_bin_definitions.csv", index=False)
    pd.concat(selections, ignore_index=True).to_csv(args.out_dir / "all_relation_unit_image_selection.csv", index=False)
    pd.concat(component_selections, ignore_index=True).to_csv(
        args.out_dir / "all_relation_component_unit_image_selection.csv", index=False,
    )
    pd.concat(unit_curves_all, ignore_index=True).to_csv(args.out_dir / "all_relation_unit_first_curves.csv", index=False)
    endpoints = endpoint_table(populations, components)
    endpoints.to_csv(args.out_dir / "key_endpoint_audit.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "question": "key BackImage population and component results under low/high halves of the new preferred-SF estimates",
        "sf_half_contract": {
            "metric": "preferred_sf_cpd from the joint parametric SF/TF model",
            "n_valid": 85,
            "median_threshold_cpd": threshold,
            "rule": "low <= median; high > median",
            "tie_check": "boundary does not split an exact preferred-SF tie",
            "counts": {str(k): int(v) for k, v in counts.items()},
        },
        "analysis_contract": {
            "baseline": "trial-mean stabilized reference",
            "aggregation": "spike-weighted population SSI with equal-unit audit",
            "n_drift_bins": N_DRIFT_BINS,
            "n_microsaccade_bins": N_MICROSACCADE_BINS,
            "n_image_bootstraps": int(args.n_bootstrap),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "relations": [relation for relation, _, _ in RELATIONS],
        },
        "visual_contract": {
            "family": "ordered line small multiples",
            "palette": COLORS,
            "non_color_encodings": "open/filled markers for context; solid-circle/dashed-square for across/along components",
        },
        "sources": {
            "assignments": file_identity(Path(args.assignments_csv)),
            "parametric_arrays": file_identity(
                ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/rr100_sf_tf_parametric_model_arrays.npz"
            ),
            "ssi_matrix": file_identity(matrix_dir / "ssi_matrix.npy"),
            "expected_spikes_matrix": file_identity(matrix_dir / "expected_spikes_matrix.npy"),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        f"""# BackImage key results with low/high SF halves

The 85 valid parametric SF fits are split at the observed median preferred SF
of {threshold:.6f} cpd. The low half contains 43 units (including the median)
and the high half contains 42 units. The boundary does not split an exact tie.

Key figures:

- `key_01_population_contexts_low_high_sf_halves.png`
- `key_02_component_contexts_low_high_sf_halves.png`
- `key_03_population_weighting_low_high_sf_halves.png`
- `key_04_mixed_context_low_high_sf_halves.png`

All historical baseline, selection, binning, aggregation, and bootstrap contracts
are retained. Per-relation figures and tables are saved in relation subfolders.
""",
        encoding="utf-8",
    )
    print(f"Wrote {args.out_dir.resolve()}")
    print(f"Median preferred-SF split: {threshold:.9f} cpd; counts {counts.to_dict()}")
    print(endpoints.to_string(index=False))


if __name__ == "__main__":
    main()
