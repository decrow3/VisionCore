#!/usr/bin/env python3
"""Checkpoint 8: page-17 all-images across/along component SSI with new SF quartiles."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information import (
    plot_backimage_real_trace_unit_first_and_population_schematics as schematic,
)
from declan.fig4_active_sensing.rerun_backimage_all_images_population_sf_quartiles import (
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    N_DRIFT_BINS,
    N_MICROSACCADE_BINS,
    RELATION,
    RELATION_LABEL,
    prepare_quartile_units,
)
from declan.fig4_active_sensing.rerun_backimage_real_trace_contour_matched_sf_quartiles import (
    COLORS,
    DEFAULT_ASSIGNMENTS,
    DEFAULT_MATRIX_DIR,
    GROUPS,
    LABELS,
    ROOT,
    configure_matplotlib,
    file_identity,
    save_figure,
)


DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/"
    "checkpoint_08_all_images_component_sf_quartiles"
)
COMPONENT_LABELS = {
    "across_path_arcmin": "across contour",
    "along_path_arcmin": "along contour",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--assignments-csv", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    return parser.parse_args()


def prepare_component_inputs(data: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    metrics = schematic.compute_component_movie_metrics(
        data["movie"], data["trace"], data["trace_xy"], image_axis_col="image_edge_axis_deg",
    )
    bins_by_metric = {}
    tables = []
    for metric_col, metric_label in schematic.COMPONENT_PATH_SPECS:
        metrics, bins = schematic.add_equal_count_component_bins(
            metrics, metric_col=metric_col, metric_label=metric_label,
            n_drift_bins=N_DRIFT_BINS, n_microsaccade_bins=N_MICROSACCADE_BINS,
        )
        bins_by_metric[metric_col] = bins
        tables.append(bins)
    return metrics, bins_by_metric, pd.concat(tables, ignore_index=True)


def build_component_summary(
    data: dict,
    unit: pd.DataFrame,
    groups: list[str],
    component_metrics: pd.DataFrame,
    bins_by_metric: dict[str, pd.DataFrame],
    row_grid: np.ndarray,
    baseline_lookup: dict[int, int],
    *,
    n_bootstrap: int,
    relation: str = RELATION,
    relation_label: str = RELATION_LABEL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selections = schematic.unit_image_selection(
        unit, data["image"], relation=relation, sf_groups=groups,
        min_osi=0.05, match_max_deg=22.5, orthogonal_min_deg=67.5,
        image_axis_col="image_edge_axis_deg",
    )
    summary = schematic.build_component_population_summary_for_relation(
        relation=relation, relation_label=relation_label,
        selections=selections, sf_groups=groups,
        component_metrics=component_metrics, component_bins_by_metric=bins_by_metric,
        ssi=data["ssi"], expected=data["expected"],
        stabilized_ssi=data["stabilized_ssi"], stabilized_expected=data["stabilized_expected"],
        row_grid=row_grid, baseline_lookup=baseline_lookup,
        rng=np.random.default_rng(BOOTSTRAP_SEED), n_bootstrap=int(n_bootstrap),
    )
    selection_rows = []
    metadata = unit.set_index("unit_index")
    for group, unit_to_images in selections.items():
        for unit_index, images in unit_to_images.items():
            row = metadata.loc[int(unit_index)]
            selection_rows.append(
                {
                    "sf_group": group, "unit_index": int(unit_index),
                    "unit_label": str(row["unit_label"]), "n_selected_images": int(len(images)),
                }
            )
    return pd.DataFrame(selection_rows), summary


def validate_historical(
    data: dict,
    component_metrics: pd.DataFrame,
    bins_by_metric: dict[str, pd.DataFrame],
    row_grid: np.ndarray,
    baseline_lookup: dict[int, int],
    matrix_dir: Path,
    *,
    n_bootstrap: int,
    relation: str = RELATION,
    relation_label: str = RELATION_LABEL,
) -> dict:
    _, reconstructed = build_component_summary(
        data, data["unit"], ["low_sf", "middle_sf", "high_sf"],
        component_metrics, bins_by_metric, row_grid, baseline_lookup,
        n_bootstrap=n_bootstrap,
        relation=relation, relation_label=relation_label,
    )
    saved_path = matrix_dir / (
        "phase1_phase2_conditioning_v1/schematic_pathlength_summary_v1/unit_first_and_population_v1/"
        "spike_weighted_population_component_summary.csv"
    )
    saved = pd.read_csv(saved_path)
    saved = saved[saved["relation"].eq(relation)].copy()
    keys = ["sf_group", "component_metric", "context", "component_bin"]
    columns = [
        "population_ssi_bits_per_spike", "population_ssi_delta_vs_stabilized",
        "information_numerator_bits", "expected_spikes",
    ]
    compare = reconstructed.merge(
        saved[keys + columns], on=keys, suffixes=("_new", "_saved"), validate="one_to_one",
    )
    max_diff = max(
        float(np.max(np.abs(compare[f"{column}_new"] - compare[f"{column}_saved"])))
        for column in columns
    )
    passed = len(compare) == len(saved) == len(reconstructed) and max_diff < 1e-8
    return {
        "saved_component_summary": file_identity(saved_path),
        "n_rows_reconstructed": int(len(reconstructed)),
        "n_rows_saved": int(len(saved)),
        "n_rows_matched": int(len(compare)),
        "max_absolute_core_difference": max_diff,
        "bootstrap_ci_note": "core values validated; bootstrap intervals are seed- and call-order-dependent",
        "passed": bool(passed),
    }


def endpoint_audit(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        for metric in COMPONENT_LABELS:
            for endpoint, component_bin in (
                ("smallest_drift", "drift_only_q01"),
                ("largest_drift", "drift_only_q08"),
                ("smallest_microsaccade", "microsaccade_q01"),
                ("largest_microsaccade", "microsaccade_q05"),
            ):
                row = summary[
                    summary["sf_group"].eq(group)
                    & summary["component_metric"].eq(metric)
                    & summary["component_bin"].eq(component_bin)
                ].iloc[0]
                rows.append(
                    {
                        "sf_quartile": group, "component": metric,
                        "component_label": COMPONENT_LABELS[metric], "endpoint": endpoint,
                        "component_bin_internal": component_bin,
                        "component_median_arcmin": float(row["component_median_arcmin"]),
                        "population_delta_vs_stabilized": float(row["population_ssi_delta_vs_stabilized"]),
                        "delta_ci95_low": float(row["population_delta_ci95_low_image_boot"]),
                        "delta_ci95_high": float(row["population_delta_ci95_high_image_boot"]),
                        "delta_p_bootstrap_sign": float(row["population_delta_p_image_bootstrap_sign"]),
                    }
                )
    audit = pd.DataFrame(rows)
    pivot = audit.pivot(
        index=["sf_quartile", "endpoint"], columns="component",
        values="population_delta_vs_stabilized",
    ).reset_index()
    pivot["across_minus_along_delta"] = pivot["across_path_arcmin"] - pivot["along_path_arcmin"]
    return audit.merge(
        pivot[["sf_quartile", "endpoint", "across_minus_along_delta"]],
        on=["sf_quartile", "endpoint"], validate="many_to_one",
    )


def make_direct_overlay(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    stem: str = "checkpoint_08_across_vs_along_direct_overlay",
    title: str = "Across- versus along-contour population modulation",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.9))
    styles = {
        "across_path_arcmin": ("-", "o"),
        "along_path_arcmin": ("--", "s"),
    }
    for ax, group in zip(axes.ravel(), GROUPS, strict=True):
        for metric, (linestyle, marker) in styles.items():
            for context, face in (("drift_only", "white"), ("microsaccade", COLORS[group])):
                sub = summary[
                    summary["sf_group"].eq(group)
                    & summary["component_metric"].eq(metric)
                    & summary["context"].eq(context)
                ].sort_values("component_median_arcmin")
                ax.plot(
                    sub["component_median_arcmin"], sub["population_ssi_delta_vs_stabilized"],
                    color=COLORS[group], ls=linestyle, marker=marker,
                    markerfacecolor=face, lw=2.0, ms=4.3,
                    label=COMPONENT_LABELS[metric] if context == "drift_only" else None,
                )
        ax.axhline(0, color="0.45", ls=":", lw=0.9)
        ax.set_title(f"{LABELS[group]} (n={int(summary[summary['sf_group'].eq(group)]['n_units'].iloc[0])})", loc="left", fontweight="bold")
        ax.set(xlabel="component path length (arcmin)", ylabel="population SSI minus stabilized")
        ax.grid(True, color="0.92", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        title,
        x=0.02, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.2, w_pad=2.0)
    save_figure(fig, out_dir, stem)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    matrix_dir = Path(args.matrix_dir)
    data = schematic.load_dataset(matrix_dir)
    component_metrics, bins_by_metric, bins_table = prepare_component_inputs(data)
    row_grid = schematic.build_movie_row_grid(data["movie"])
    baseline_lookup = schematic.baseline_rows_by_image(data["image"], data["baseline_table"])
    validation = validate_historical(
        data, component_metrics, bins_by_metric, row_grid, baseline_lookup, matrix_dir,
        n_bootstrap=int(args.n_bootstrap),
    )
    if not validation["passed"]:
        raise ValueError(f"Historical page-17 reconstruction failed: {validation}")

    assignments = pd.read_csv(args.assignments_csv)
    unit_new = prepare_quartile_units(data["unit"], assignments)
    selection, summary = build_component_summary(
        data, unit_new, list(GROUPS), component_metrics, bins_by_metric,
        row_grid, baseline_lookup, n_bootstrap=int(args.n_bootstrap),
    )
    endpoint = endpoint_audit(summary)

    schematic.SF_COLORS.update(COLORS)
    schematic.SF_LABELS.update(LABELS)
    schematic.plot_component_population_12_panel(
        summary, args.out_dir,
        stem="017_all_images_valid_fit_sf_quartiles_across_along_components",
        title="Spike-weighted population SSI - all images: across and along contour components",
        sf_groups=list(GROUPS), dpi=220,
    )
    make_direct_overlay(summary, args.out_dir)

    bins_table.to_csv(args.out_dir / "component_path_bin_definitions.csv", index=False)
    selection.to_csv(args.out_dir / "unit_image_selection.csv", index=False)
    summary.to_csv(args.out_dir / "spike_weighted_population_component_summary.csv", index=False)
    endpoint.to_csv(args.out_dir / "component_endpoint_audit.csv", index=False)
    (args.out_dir / "historical_contract_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    counts = selection.groupby("sf_group")["unit_index"].nunique().reindex(GROUPS, fill_value=0).to_dict()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "checkpoint_08_complete",
        "source_test": "PDF page 17: all-images across/along component-path population SSI",
        "replacement_scope": "all 85 model-valid units in tie-aware preferred-SF quartiles; no OSI gate",
        "component_contract": "trace steps projected onto each image contour axis; equal-count image-by-trace movie bins formed separately for across and along path",
        "aggregation_contract": "spike-weighted population SSI with trial-mean stabilized baseline",
        "bootstrap": {"n_image_resamples": int(args.n_bootstrap), "seed": BOOTSTRAP_SEED},
        "unit_counts": counts,
        "sources": {
            "assignments": file_identity(Path(args.assignments_csv)),
            "trace_xy": file_identity(matrix_dir / "trace_xy.npy"),
            "ssi_matrix": file_identity(matrix_dir / "ssi_matrix.npy"),
            "expected_spikes_matrix": file_identity(matrix_dir / "expected_spikes_matrix.npy"),
        },
        "historical_contract_validation": validation,
        "not_run": "No strong-contour population summary (page 18) or later test was regenerated.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    first = endpoint[endpoint["endpoint"].eq("smallest_drift")]
    lines = []
    for group in GROUPS:
        sub = first[first["sf_quartile"].eq(group)].set_index("component")
        lines.append(
            f"- {LABELS[group]}: across {sub.loc['across_path_arcmin', 'population_delta_vs_stabilized']:+.4f}; "
            f"along {sub.loc['along_path_arcmin', 'population_delta_vs_stabilized']:+.4f}; "
            f"across-minus-along {sub['across_minus_along_delta'].iloc[0]:+.4f} bits/spike."
        )
    readme = f"""# Checkpoint 8: all-images across/along component SF quartiles

This regenerates page 17 with the new preferred-SF quartiles. Across and along
components are defined relative to each image's contour axis, so their bins are
equal-count image-by-trace movie bins rather than trace-only bins.

## Smallest drift-only component bins

{chr(10).join(lines)}

The historical low/middle/high core values are exactly reproduced; see
`historical_contract_validation.json`. No page-18 strong-contour analysis was
run here.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {args.out_dir.resolve()}")
    print(first.to_string(index=False))
    print(endpoint.to_string(index=False))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
