#!/usr/bin/env python3
"""Checkpoint 7: page-16 all-images population SSI with new SF quartiles."""

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
    "checkpoint_07_all_images_population_sf_quartiles"
)
RELATION = "all_images_no_osi"
RELATION_LABEL = "all image windows; no OSI gate"
N_DRIFT_BINS = 8
N_MICROSACCADE_BINS = 5
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 47


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--assignments-csv", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    return parser.parse_args()


def prepare_quartile_units(unit: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rr100_index", "model_valid", "sf_quartile", "sf_quartile_label",
        "preferred_sf_cpd", "preferred_tf_hz", "joint_parametric_surface_r2",
    ]
    out = unit.merge(
        assignments[columns], left_on="unit_index", right_on="rr100_index",
        how="left", validate="one_to_one",
    )
    out = out[out["model_valid"].fillna(False).astype(bool) & out["sf_quartile"].isin(GROUPS)].copy()
    out["historical_sf_group"] = out["sf_group"]
    out["sf_group"] = out["sf_quartile"]
    out["sf_group_label"] = out["sf_quartile_label"]
    return out


def build_summary(
    data: dict,
    unit: pd.DataFrame,
    sf_groups: list[str],
    trace: pd.DataFrame,
    trace_bins: pd.DataFrame,
    row_grid: np.ndarray,
    baseline_lookup: dict[int, int],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections = schematic.unit_image_selection(
        unit, data["image"], relation=RELATION, sf_groups=sf_groups,
        min_osi=0.05, match_max_deg=22.5, orthogonal_min_deg=67.5,
        image_axis_col="image_edge_axis_deg",
    )
    return schematic.build_curves_for_relation(
        relation=RELATION, relation_label=RELATION_LABEL,
        selections=selections, sf_groups=sf_groups,
        trace=trace, trace_bins=trace_bins, row_grid=row_grid,
        baseline_lookup=baseline_lookup, ssi=data["ssi"], expected=data["expected"],
        stabilized_ssi=data["stabilized_ssi"], stabilized_expected=data["stabilized_expected"],
        unit=unit, rng=np.random.default_rng(seed), n_bootstrap=int(n_bootstrap),
    )


def validate_historical(
    data: dict,
    trace: pd.DataFrame,
    trace_bins: pd.DataFrame,
    row_grid: np.ndarray,
    baseline_lookup: dict[int, int],
    matrix_dir: Path,
    *,
    n_bootstrap: int,
) -> dict:
    _, _, unit_summary, population_summary = build_summary(
        data, data["unit"], ["low_sf", "middle_sf", "high_sf"],
        trace, trace_bins, row_grid, baseline_lookup,
        n_bootstrap=n_bootstrap, seed=BOOTSTRAP_SEED,
    )
    old_root = matrix_dir / (
        "phase1_phase2_conditioning_v1/schematic_pathlength_summary_v1/unit_first_and_population_v1"
    )
    saved_population_path = old_root / "spike_weighted_population_summary.csv"
    saved_unit_path = old_root / "unit_first_summary.csv"
    saved_population = pd.read_csv(saved_population_path)
    saved_population = saved_population[saved_population["relation"].eq(RELATION)].copy()
    saved_unit = pd.read_csv(saved_unit_path)
    saved_unit = saved_unit[saved_unit["relation"].eq(RELATION)].copy()
    keys = ["sf_group", "context", "path_bin"]
    pop_columns = [
        "population_ssi_bits_per_spike", "population_ssi_delta_vs_stabilized",
        "information_numerator_bits", "expected_spikes",
    ]
    unit_columns = [
        "mean_unit_ssi_bits_per_spike", "sem_unit_ssi_bits_per_spike",
        "mean_unit_ssi_delta_vs_stabilized", "sem_unit_ssi_delta_vs_stabilized",
    ]
    pop_compare = population_summary.merge(
        saved_population[keys + pop_columns], on=keys, suffixes=("_new", "_saved"), validate="one_to_one",
    )
    unit_compare = unit_summary.merge(
        saved_unit[keys + unit_columns], on=keys, suffixes=("_new", "_saved"), validate="one_to_one",
    )
    pop_diff = max(
        float(np.max(np.abs(pop_compare[f"{column}_new"] - pop_compare[f"{column}_saved"])))
        for column in pop_columns
    )
    unit_diff = max(
        float(np.max(np.abs(unit_compare[f"{column}_new"] - unit_compare[f"{column}_saved"])))
        for column in unit_columns
    )
    passed = (
        len(pop_compare) == len(saved_population) == len(population_summary)
        and len(unit_compare) == len(saved_unit) == len(unit_summary)
        and pop_diff < 1e-8 and unit_diff < 1e-8
    )
    return {
        "saved_population_summary": file_identity(saved_population_path),
        "saved_unit_first_summary": file_identity(saved_unit_path),
        "n_population_rows_matched": int(len(pop_compare)),
        "n_unit_first_rows_matched": int(len(unit_compare)),
        "max_absolute_population_core_difference": pop_diff,
        "max_absolute_unit_first_difference": unit_diff,
        "bootstrap_ci_note": "core values validated; bootstrap intervals are seed- and call-order-dependent",
        "passed": bool(passed),
    }


def endpoint_audit(population: pd.DataFrame, unit_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        for context, path_bin in (
            ("smallest_drift", "drift_only_q01"),
            ("largest_drift", "drift_only_q08"),
            ("smallest_microsaccade", "microsaccade_q01"),
            ("largest_microsaccade", "microsaccade_q05"),
        ):
            pop = population[population["sf_group"].eq(group) & population["path_bin"].eq(path_bin)].iloc[0]
            unit = unit_summary[unit_summary["sf_group"].eq(group) & unit_summary["path_bin"].eq(path_bin)].iloc[0]
            rows.append(
                {
                    "sf_quartile": group, "endpoint": context, "path_bin_internal": path_bin,
                    "path_median_arcmin": float(pop["path_median_arcmin"]),
                    "population_delta_vs_stabilized": float(pop["population_ssi_delta_vs_stabilized"]),
                    "population_delta_ci95_low": float(pop["population_delta_ci95_low_image_boot"]),
                    "population_delta_ci95_high": float(pop["population_delta_ci95_high_image_boot"]),
                    "population_delta_p_bootstrap_sign": float(pop["population_delta_p_image_bootstrap_sign"]),
                    "mean_unit_delta_vs_stabilized": float(unit["mean_unit_ssi_delta_vs_stabilized"]),
                    "sem_unit_delta_vs_stabilized": float(unit["sem_unit_ssi_delta_vs_stabilized"]),
                    "population_minus_unit_first_delta": float(
                        pop["population_ssi_delta_vs_stabilized"] - unit["mean_unit_ssi_delta_vs_stabilized"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_unit_weighting_figure(
    population: pd.DataFrame,
    unit_summary: pd.DataFrame,
    out_dir: Path,
    *,
    title: str = "Population weighting audit: all images and valid-fit units",
    stem: str = "checkpoint_07_population_vs_equal_unit_audit",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8), sharex=True)
    for ax, group in zip(axes.ravel(), GROUPS, strict=True):
        pop = population[population["sf_group"].eq(group)].copy()
        unit = unit_summary[unit_summary["sf_group"].eq(group)].copy()
        for context, marker_face in (("drift_only", "white"), ("microsaccade", COLORS[group])):
            p = pop[pop["context"].eq(context)].sort_values("path_median_arcmin")
            u = unit[unit["context"].eq(context)].sort_values("path_median_arcmin")
            ax.plot(
                p["path_median_arcmin"], p["population_ssi_delta_vs_stabilized"],
                color=COLORS[group], marker="o", markerfacecolor=marker_face,
                lw=2.2, ms=4.5, label="spike-weighted population" if context == "drift_only" else None,
            )
            ax.fill_between(
                p["path_median_arcmin"].to_numpy(float),
                p["population_delta_ci95_low_image_boot"].to_numpy(float),
                p["population_delta_ci95_high_image_boot"].to_numpy(float),
                color=COLORS[group], alpha=0.10, lw=0,
            )
            ax.plot(
                u["path_median_arcmin"], u["mean_unit_ssi_delta_vs_stabilized"],
                color="0.25", marker="s", markerfacecolor="white" if context == "drift_only" else "0.25",
                ls="--", lw=1.5, ms=3.8, label="equal-unit mean" if context == "drift_only" else None,
            )
        ax.axhline(0, color="0.45", ls=":", lw=0.9)
        ax.set_title(f"{LABELS[group]} (n={int(pop['n_units'].iloc[0])})", loc="left", fontweight="bold")
        ax.set_ylabel("SSI minus stabilized (bits/spike)")
        ax.grid(True, color="0.92", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("trace path length (arcmin)")
    axes[1, 1].set_xlabel("trace path length (arcmin)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        title,
        x=0.02, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.2, w_pad=2.0)
    save_figure(fig, out_dir, stem)


def smallest_drift_unit_contributions(
    data: dict,
    unit: pd.DataFrame,
    trace: pd.DataFrame,
    row_grid: np.ndarray,
    baseline_lookup: dict[int, int],
) -> pd.DataFrame:
    traces = trace.loc[trace["path_bin"].eq("drift_only_q01"), "trace_bank_index"].astype(int).to_numpy()
    images = data["image"]["image_index"].astype(int).to_numpy()
    rows = row_grid[np.ix_(images, traces)]
    baseline_rows = np.asarray([baseline_lookup[int(image)] for image in images], dtype=int)
    records = []
    for group in GROUPS:
        group_units = unit.loc[unit["sf_group"].eq(group), "unit_index"].astype(int).to_numpy()
        pieces = []
        for unit_index in group_units:
            condition_ssi = np.asarray(data["ssi"][rows, unit_index], dtype=float)
            condition_weight = np.asarray(data["expected"][rows, unit_index], dtype=float)
            baseline_ssi = np.asarray(data["stabilized_ssi"][baseline_rows, unit_index], dtype=float)
            baseline_weight = np.asarray(data["stabilized_expected"][baseline_rows, unit_index], dtype=float)
            pieces.append(
                {
                    "unit_index": int(unit_index),
                    "condition_num": float(np.nansum(condition_ssi * condition_weight)),
                    "condition_den": float(np.nansum(condition_weight)),
                    "baseline_num": float(np.nansum(baseline_ssi * baseline_weight)),
                    "baseline_den": float(np.nansum(baseline_weight)),
                }
            )
        frame = pd.DataFrame(pieces)
        totals = frame[["condition_num", "condition_den", "baseline_num", "baseline_den"]].sum()
        full_delta = (
            totals["condition_num"] / totals["condition_den"]
            - totals["baseline_num"] / totals["baseline_den"]
        )
        metadata = unit.set_index("unit_index")
        for row in frame.itertuples(index=False):
            leave_out_delta = (
                (totals["condition_num"] - row.condition_num) / (totals["condition_den"] - row.condition_den)
                - (totals["baseline_num"] - row.baseline_num) / (totals["baseline_den"] - row.baseline_den)
            )
            meta = metadata.loc[int(row.unit_index)]
            records.append(
                {
                    "sf_quartile": group, "unit_index": int(row.unit_index),
                    "unit_label": str(meta["unit_label"]),
                    "preferred_sf_cpd": float(meta["preferred_sf_cpd"]),
                    "preferred_tf_hz": float(meta["preferred_tf_hz"]),
                    "condition_expected_spike_share": float(row.condition_den / totals["condition_den"]),
                    "baseline_expected_spike_share": float(row.baseline_den / totals["baseline_den"]),
                    "unit_own_condition_ssi": float(row.condition_num / row.condition_den),
                    "unit_own_baseline_ssi": float(row.baseline_num / row.baseline_den),
                    "unit_own_delta": float(row.condition_num / row.condition_den - row.baseline_num / row.baseline_den),
                    "full_population_delta": float(full_delta),
                    "population_delta_without_unit": float(leave_out_delta),
                    "leave_one_out_change": float(leave_out_delta - full_delta),
                }
            )
    return pd.DataFrame(records)


def make_contribution_figure(contributions: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8))
    for ax, group in zip(axes.ravel(), GROUPS, strict=True):
        sub = contributions[contributions["sf_quartile"].eq(group)].copy()
        ax.scatter(
            100 * sub["condition_expected_spike_share"], sub["unit_own_delta"],
            s=24, color=COLORS[group], alpha=0.68, edgecolor="white", lw=0.4,
        )
        influential = sub.reindex(sub["leave_one_out_change"].abs().sort_values(ascending=False).index).head(3)
        for row in influential.itertuples(index=False):
            ax.annotate(
                row.unit_label,
                (100 * row.condition_expected_spike_share, row.unit_own_delta),
                xytext=(4, 4), textcoords="offset points", fontsize=7,
            )
        ax.axhline(0, color="0.45", ls=":", lw=0.9)
        ax.set_title(LABELS[group], loc="left", fontweight="bold")
        ax.set(xlabel="smallest-drift expected-spike share (%)", ylabel="unit SSI minus its stabilized SSI")
        ax.grid(True, color="0.92", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Smallest-drift population contribution audit",
        x=0.02, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.2, w_pad=2.0)
    save_figure(fig, out_dir, "checkpoint_07_smallest_drift_unit_contributions")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    matrix_dir = Path(args.matrix_dir)
    data = schematic.load_dataset(matrix_dir)
    trace, trace_bins = schematic.add_equal_count_trace_bins(
        data["trace"], n_drift_bins=N_DRIFT_BINS, n_microsaccade_bins=N_MICROSACCADE_BINS,
    )
    row_grid = schematic.build_movie_row_grid(data["movie"])
    baseline_lookup = schematic.baseline_rows_by_image(data["image"], data["baseline_table"])
    validation = validate_historical(
        data, trace, trace_bins, row_grid, baseline_lookup, matrix_dir,
        n_bootstrap=int(args.n_bootstrap),
    )
    if not validation["passed"]:
        raise ValueError(f"Historical page-16 reconstruction failed: {validation}")

    assignments = pd.read_csv(args.assignments_csv)
    unit_new = prepare_quartile_units(data["unit"], assignments)
    selection, unit_curves, unit_summary, population_summary = build_summary(
        data, unit_new, list(GROUPS), trace, trace_bins, row_grid, baseline_lookup,
        n_bootstrap=int(args.n_bootstrap), seed=BOOTSTRAP_SEED,
    )
    endpoint = endpoint_audit(population_summary, unit_summary)
    contributions = smallest_drift_unit_contributions(data, unit_new, trace, row_grid, baseline_lookup)

    schematic.SF_COLORS.update(COLORS)
    schematic.SF_LABELS.update(LABELS)
    schematic.plot_sf_rows_population_panel(
        population_summary, args.out_dir,
        stem="016_all_images_valid_fit_sf_quartiles_population_absolute_delta",
        title="Spike-weighted population SSI - all image windows, valid-fit units",
        subtitle=(
            f"new SF quartiles; path bins split as {N_DRIFT_BINS} drift-only / "
            f"{N_MICROSACCADE_BINS} microsaccade"
        ),
        sf_groups=list(GROUPS), dpi=220,
    )
    make_unit_weighting_figure(population_summary, unit_summary, args.out_dir)
    make_contribution_figure(contributions, args.out_dir)

    trace_bins.to_csv(args.out_dir / "trace_path_bin_definitions.csv", index=False)
    selection.to_csv(args.out_dir / "unit_image_selection.csv", index=False)
    unit_curves.to_csv(args.out_dir / "unit_first_curves.csv", index=False)
    unit_summary.to_csv(args.out_dir / "unit_first_summary.csv", index=False)
    population_summary.to_csv(args.out_dir / "spike_weighted_population_summary.csv", index=False)
    endpoint.to_csv(args.out_dir / "population_vs_unit_first_endpoint_audit.csv", index=False)
    contributions.to_csv(args.out_dir / "smallest_drift_unit_weight_contributions.csv", index=False)
    (args.out_dir / "historical_contract_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    counts = selection.groupby("sf_group")["unit_index"].nunique().reindex(GROUPS, fill_value=0).to_dict()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "checkpoint_07_complete",
        "source_test": "PDF page 16: all-images, all-units spike-weighted population SSI",
        "replacement_scope": "all model-valid units divided into preferred-SF quartiles; no OSI gate",
        "aggregation_contract": "sum(SSI bits/spike * expected spikes) / sum(expected spikes) over units, all images, and traces within bin",
        "baseline_contract": "trial-mean stabilized SSI and expected spikes, accumulated over the same units and images",
        "trace_bins": f"equal-count {N_DRIFT_BINS} drift-only and {N_MICROSACCADE_BINS} microsaccade bins",
        "bootstrap": {"n_image_resamples": int(args.n_bootstrap), "seed": BOOTSTRAP_SEED},
        "unit_counts": counts,
        "sources": {
            "assignments": file_identity(Path(args.assignments_csv)),
            "ssi_matrix": file_identity(matrix_dir / "ssi_matrix.npy"),
            "expected_spikes_matrix": file_identity(matrix_dir / "expected_spikes_matrix.npy"),
        },
        "historical_contract_validation": validation,
        "not_run": "No across/along component decomposition (page 17) or later strong-contour test was regenerated.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    first = endpoint[endpoint["endpoint"].eq("smallest_drift")]
    lines = []
    for row in first.itertuples(index=False):
        lines.append(
            f"- {LABELS[row.sf_quartile]}: population {row.population_delta_vs_stabilized:+.4f} "
            f"[{row.population_delta_ci95_low:+.4f}, {row.population_delta_ci95_high:+.4f}]; "
            f"equal-unit {row.mean_unit_delta_vs_stabilized:+.4f} +/- {row.sem_unit_delta_vs_stabilized:.4f}."
        )
    readme = f"""# Checkpoint 7: all-images population SF quartiles

This regenerates page 16 using the new parametric preferred-SF quartiles. The
85 valid fits form tie-aware groups of 22/21/21/21 units; all 100 images are
used and there is no OSI gate. Absolute population SSI is spike-weighted, while
the companion audit uses an equal-unit mean.

## Smallest drift-only bin versus stabilization

{chr(10).join(lines)}

The historical low/middle/high implementation is exactly reproduced for core
population and unit-first values; see `historical_contract_validation.json`.
No page-17 component decomposition was run here.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {args.out_dir.resolve()}")
    print(first.to_string(index=False))
    print(endpoint.to_string(index=False))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
