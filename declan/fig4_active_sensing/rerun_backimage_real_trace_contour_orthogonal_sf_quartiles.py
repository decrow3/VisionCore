#!/usr/bin/env python3
"""Checkpoint 4: rerun page-13 contour-orthogonal curves with new SF quartiles."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.analyze_backimage_real_trace_ssi_matrix_phase1_phase2 import (
    build_real_trace_sf_contour_matched_unit_curves,
    load_matrix_dataset,
    load_stabilized_baseline,
    trace_microsaccade_path_context_from_frame,
)
from declan.fig4_active_sensing.rerun_backimage_real_trace_contour_matched_sf_quartiles import (
    COLORS,
    DEFAULT_ASSIGNMENTS,
    DEFAULT_CONTEXT,
    DEFAULT_MATRIX_DIR,
    EXTREMES,
    GROUPS,
    LABELS,
    MATCH_MAX_DEG,
    MIN_MATCHED_IMAGES,
    MIN_OSI,
    ORTHOGONAL_MIN_DEG,
    ROOT,
    add_trace_bins,
    calculate_unit_changes,
    configure_matplotlib,
    file_identity,
    leave_one_out_sensitivity,
    plot_summary_panel,
    prepare_new_unit_table,
    save_figure,
    select_example_roles,
    summarize_changes,
)


DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/"
    "checkpoint_04_contour_orthogonal_sf_quartiles"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--assignments-csv", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--trace-context-csv", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def make_figures(
    selection: pd.DataFrame,
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    changes: pd.DataFrame,
    context: pd.DataFrame,
    out_dir: Path,
) -> None:
    absolute = "unit_contour_matched_ssi_bits_per_spike"
    delta = "unit_contour_matched_ssi_delta_vs_stabilized"

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharex=True)
    plot_summary_panel(
        axes[0], summary, EXTREMES, absolute,
        title="Absolute SSI on contour-orthogonal windows",
        ylabel="SSI (bits/spike)", context=context,
    )
    plot_summary_panel(
        axes[1], summary, EXTREMES, delta,
        title="Movement modulation",
        ylabel="SSI minus stabilized baseline (bits/spike)", context=context,
    )
    fig.suptitle(
        "Contour-orthogonal unit-window pairs: real trace scale\n"
        "new SF extremes; orthogonality >= 67.5 deg; OSI >= 0.05; min images/unit = 1",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, out_dir, "013_phase2_contour_orthogonal_sf_q1_q4_curves")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharex=True)
    plot_summary_panel(
        axes[0], summary, GROUPS, absolute,
        title="A. All quartiles: absolute SSI", ylabel="SSI (bits/spike)", context=context,
    )
    plot_summary_panel(
        axes[1], summary, GROUPS, delta,
        title="B. All quartiles: moving minus stabilized",
        ylabel="SSI difference (bits/spike)", context=context,
    )
    fig.suptitle(
        "Checkpoint 4: contour-orthogonal quartile audit",
        x=0.02, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.2)
    save_figure(fig, out_dir, "checkpoint_04_contour_orthogonal_all_quartiles")

    bins = sorted(curves["trace_path_length_bin"].astype(str).unique())
    xmap = {key: idx for idx, key in enumerate(bins)}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), sharex=True)
    for ax, group in zip(axes.ravel(), GROUPS, strict=True):
        sub = curves[curves["sf_group"].astype(str).eq(group)]
        for _, unit_curve in sub.groupby("unit_index", sort=True):
            unit_curve = unit_curve.sort_values("trace_path_length_bin")
            ax.plot(
                unit_curve["trace_path_length_bin"].map(xmap),
                unit_curve[delta], color=COLORS[group], alpha=0.22, lw=0.8,
            )
        mean_sub = summary[
            summary["sf_group"].astype(str).eq(group)
            & summary["value_name"].astype(str).eq(delta)
        ].sort_values("trace_path_length_bin")
        ax.plot(
            mean_sub["trace_path_length_bin"].map(xmap), mean_sub["mean"],
            color=COLORS[group], marker="o", lw=2.4, ms=4.5,
        )
        ax.axhline(0, color="0.45", ls=":", lw=0.9)
        ax.set_title(
            f"{LABELS[group]}: individual units (n={sub['unit_index'].nunique()})",
            loc="left", fontweight="bold",
        )
        ax.set_ylabel("SSI minus stabilized")
        ax.set_xticks(
            range(len(bins)),
            ["short" if b == "q01" else "long" if b == "q06" else f"bin {int(b[1:])}" for b in bins],
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("trace path bin (endpoint labels shown)")
    axes[1, 1].set_xlabel("trace path bin (endpoint labels shown)")
    fig.suptitle(
        "Contour-orthogonal unit-level heterogeneity",
        x=0.02, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.2, w_pad=2.0)
    save_figure(fig, out_dir, "checkpoint_04_contour_orthogonal_unit_heterogeneity")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    input_counts = selection.groupby("sf_group").size().reindex(GROUPS, fill_value=0)
    osi_counts = selection[selection["passes_orientation_selectivity"].astype(bool)].groupby("sf_group").size().reindex(GROUPS, fill_value=0)
    selected_counts = curves.groupby("sf_group")["unit_index"].nunique().reindex(GROUPS, fill_value=0)
    x = np.arange(len(GROUPS)); width = 0.25
    axes[0].bar(x - width, input_counts, width, color="0.78", label="valid-fit input")
    axes[0].bar(x, osi_counts, width, color="0.48", label="OSI pass")
    axes[0].bar(x + width, selected_counts, width, color=[COLORS[g] for g in GROUPS], label="orthogonal support")
    axes[0].set_xticks(x, [g.replace("sf_", "").upper() for g in GROUPS])
    axes[0].set(ylabel="units", title="A. Selection support")
    axes[0].legend(frameon=False, fontsize=7)
    for idx, group in enumerate(GROUPS):
        vals = changes.loc[changes["sf_group"].eq(group), "last_minus_first_ssi"].to_numpy(float)
        jitter = np.linspace(-0.10, 0.10, len(vals)) if len(vals) else np.array([])
        axes[1].scatter(idx + jitter, vals, color=COLORS[group], s=22, alpha=0.72, edgecolor="white", lw=0.3)
        axes[1].plot([idx - 0.20, idx + 0.20], [np.median(vals), np.median(vals)], color="black", lw=1.5)
    axes[1].axhline(0, color="0.45", ls=":", lw=0.9)
    axes[1].set_xticks(x, [g.replace("sf_", "").upper() for g in GROUPS])
    axes[1].set(ylabel="long minus short SSI (bits/spike)", title="B. Unit-level trace-scale change")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Contour-orthogonal selection and change audit",
        x=0.02, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.4)
    save_figure(fig, out_dir, "checkpoint_04_contour_orthogonal_selection_audit")


def validate_historical_contract(
    movie: pd.DataFrame,
    ssi: np.ndarray,
    unit: pd.DataFrame,
    baseline: dict,
    matrix_dir: Path,
) -> dict:
    _, _, reconstructed = build_real_trace_sf_contour_matched_unit_curves(
        movie, ssi, unit,
        image_axis_col="image_edge_axis_deg", sf_groups=["low_sf", "high_sf"],
        match_max_deg=MATCH_MAX_DEG, orthogonal_min_deg=ORTHOGONAL_MIN_DEG,
        min_orientation_selectivity=MIN_OSI,
        min_matched_images_per_unit=MIN_MATCHED_IMAGES,
        contour_relation="orthogonal", stabilized_baseline=baseline,
    )
    saved_path = matrix_dir / "phase1_phase2_conditioning_v1/phase2_real_trace_sf_contour_orthogonal_summary.csv"
    saved = pd.read_csv(saved_path)
    keys = ["sf_group", "trace_path_length_bin", "value_name"]
    compare = reconstructed.merge(saved, on=keys, suffixes=("_new", "_saved"), validate="one_to_one")
    mean_diff = float(np.max(np.abs(compare["mean_new"] - compare["mean_saved"])))
    sem_diff = float(np.max(np.abs(compare["sem_new"] - compare["sem_saved"])))
    passed = len(compare) == len(saved) == len(reconstructed) and mean_diff < 1e-10 and sem_diff < 1e-10
    return {
        "saved_summary": file_identity(saved_path),
        "n_rows_reconstructed": int(len(reconstructed)),
        "n_rows_saved": int(len(saved)),
        "n_rows_matched": int(len(compare)),
        "max_absolute_mean_difference_bits_per_spike": mean_diff,
        "max_absolute_sem_difference_bits_per_spike": sem_diff,
        "passed": bool(passed),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    matrix_dir = Path(args.matrix_dir)
    ssi, _expected, _population, movie, image, _trace, raw_unit = load_matrix_dataset(matrix_dir)
    enriched_path = matrix_dir / "phase1_phase2_conditioning_v1/phase1_unit_summary_with_ssi.csv"
    unit = pd.read_csv(enriched_path)
    if len(unit) != len(raw_unit) or not unit["unit_index"].equals(raw_unit["unit_index"]):
        raise ValueError("Enriched phase-1 unit table is not row-aligned with the SSI matrix unit table")
    movie = add_trace_bins(movie)
    baseline = load_stabilized_baseline(matrix_dir, image, unit)
    if baseline is None:
        raise ValueError("The page-13 contract requires the matched stabilized baseline")
    assignments = pd.read_csv(args.assignments_csv)
    unit_new = prepare_new_unit_table(unit, assignments)
    selection, curves, summary = build_real_trace_sf_contour_matched_unit_curves(
        movie, ssi, unit_new,
        image_axis_col="image_edge_axis_deg", sf_groups=list(GROUPS),
        match_max_deg=MATCH_MAX_DEG, orthogonal_min_deg=ORTHOGONAL_MIN_DEG,
        min_orientation_selectivity=MIN_OSI,
        min_matched_images_per_unit=MIN_MATCHED_IMAGES,
        contour_relation="orthogonal", stabilized_baseline=baseline,
    )
    changes = calculate_unit_changes(curves, unit_new)
    change_summary = summarize_changes(changes)
    sensitivity = leave_one_out_sensitivity(changes)
    examples = select_example_roles(changes)
    context_source = pd.read_csv(args.trace_context_csv)
    context = trace_microsaccade_path_context_from_frame(
        context_source,
        source_label="large_fixation_sample_pathle350arcmin",
        source_path=args.trace_context_csv,
    )
    make_figures(selection, curves, summary, changes, context, args.out_dir)
    validation = validate_historical_contract(movie, ssi, unit, baseline, matrix_dir)
    if not validation["passed"]:
        raise ValueError(f"Historical contour-orthogonal reconstruction failed: {validation}")

    outputs = {
        "contour_orthogonal_unit_selection.csv": selection,
        "contour_orthogonal_unit_curves.csv": curves,
        "contour_orthogonal_group_summary.csv": summary,
        "contour_orthogonal_unit_changes.csv": changes,
        "contour_orthogonal_change_summary.csv": change_summary,
        "contour_orthogonal_leave_one_out_sensitivity.csv": sensitivity,
        "contour_orthogonal_algorithmic_example_units.csv": examples,
        "trace_path_context_windows.csv": context,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / name, index=False)
    (args.out_dir / "historical_contract_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    selected_counts = curves.groupby("sf_group")["unit_index"].nunique().reindex(GROUPS, fill_value=0).to_dict()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "checkpoint_04_complete",
        "source_test": "PDF page 13: contour-orthogonal low/high-SF unit-first curves",
        "exact_replacement": "Q1 versus Q4; all quartiles retained in audit artifacts",
        "estimand": "within unit and trace-path bin, mean SSI over images orthogonal to the unit preference; then equal-unit group mean and SEM",
        "selection_contract": {
            "image_axis": "image_edge_axis_deg",
            "orthogonality_min_deg": ORTHOGONAL_MIN_DEG,
            "minimum_orientation_selectivity_index": MIN_OSI,
            "minimum_orthogonal_images_per_unit": MIN_MATCHED_IMAGES,
        },
        "matrix_dir": str(matrix_dir.resolve()),
        "assignments": file_identity(Path(args.assignments_csv)),
        "selected_unit_counts": selected_counts,
        "historical_contract_validation": validation,
        "artifacts": {
            "exact_page_13_analog": "013_phase2_contour_orthogonal_sf_q1_q4_curves.{png,pdf,svg}",
            "all_quartile_summary": "checkpoint_04_contour_orthogonal_all_quartiles.{png,pdf,svg}",
            "unit_heterogeneity": "checkpoint_04_contour_orthogonal_unit_heterogeneity.{png,pdf,svg}",
            "selection_audit": "checkpoint_04_contour_orthogonal_selection_audit.{png,pdf,svg}",
        },
        "not_run": "No aligned-versus-orthogonal overlay (page 14) or later Figure 4 test was regenerated.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = []
    for row in change_summary.itertuples(index=False):
        lines.append(
            f"- {LABELS[row.sf_quartile]}: n={row.n_units}; mean long-minus-short={row.mean_last_minus_first_ssi:+.4f} "
            f"bits/spike; median={row.median_last_minus_first_ssi:+.4f}; positive fraction={row.fraction_units_positive:.2f}."
        )
    sensitivity_lines = []
    for row in sensitivity.itertuples(index=False):
        sensitivity_lines.append(
            f"- {LABELS[row.sf_quartile]}: most influential is {row.most_influential_unit_label} "
            f"({row.most_influential_unit_change:+.4f}); mean without it is "
            f"{row.mean_excluding_most_influential_unit:+.4f} bits/spike."
        )
    readme = f"""# Checkpoint 4: contour-orthogonal SF quartiles

This reproduces the old page-13 unit-first contour-orthogonal test. The exact
analog uses Q1 versus Q4; companion artifacts retain Q2 and Q3 and expose
individual-unit heterogeneity and selection support.

## Unit-level trace-scale changes

{chr(10).join(lines)}

## Leave-one-unit-out sensitivity

{chr(10).join(sensitivity_lines)}

Units require OSI >= {MIN_OSI:g} and at least {MIN_MATCHED_IMAGES} image contour
at least {ORTHOGONAL_MIN_DEG:g} degrees from their preferred orientation. The
stabilized reference is the existing trial-mean baseline matched by image.

The implementation exactly reproduces the saved historical page-13 summary;
see `historical_contract_validation.json`. No aligned-versus-orthogonal overlay
or later test was run here.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {args.out_dir.resolve()}")
    print(change_summary.to_string(index=False))
    print(sensitivity.to_string(index=False))
    print(f"Selected counts: {selected_counts}")


if __name__ == "__main__":
    main()
