#!/usr/bin/env python3
"""Checkpoint 3: rerun page-12 contour-matched curves with new SF quartiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.analyze_backimage_real_trace_ssi_matrix_phase1_phase2 import (
    add_trace_path_context_bands,
    build_real_trace_sf_contour_matched_unit_curves,
    load_matrix_dataset,
    load_stabilized_baseline,
    trace_microsaccade_path_context_from_frame,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
)
DEFAULT_ASSIGNMENTS = ROOT / (
    "outputs/fig4_active_sensing/rr100_sf_quartile_iteration_checks_v1/"
    "sf_quartile_unit_assignments.csv"
)
DEFAULT_CONTEXT = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_trace_bank_diffusion_large_fixation_sample_n5000_n40_v1/"
    "filtered_path_length_le350arcmin/trace_bank_metadata_filtered.csv"
)
DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/"
    "checkpoint_03_contour_matched_sf_quartiles"
)
GROUPS = ("sf_q1", "sf_q2", "sf_q3", "sf_q4")
EXTREMES = ("sf_q1", "sf_q4")
LABELS = {
    "sf_q1": "SF Q1 (lowest)",
    "sf_q2": "SF Q2",
    "sf_q3": "SF Q3",
    "sf_q4": "SF Q4 (highest)",
}
COLORS = {
    "sf_q1": "#46327E",
    "sf_q2": "#2A788E",
    "sf_q3": "#2FB47C",
    "sf_q4": "#BDDF26",
}
MATCH_MAX_DEG = 22.5
ORTHOGONAL_MIN_DEG = 67.5
MIN_OSI = 0.05
MIN_MATCHED_IMAGES = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--assignments-csv", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--trace-context-csv", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


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


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 220} if suffix == "png" else {}
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def add_trace_bins(movie: pd.DataFrame) -> pd.DataFrame:
    out = movie.copy()
    trace_paths = out.drop_duplicates("trace_index").set_index("trace_index")["rendered_path_length_arcmin"]
    trace_bins = pd.qcut(trace_paths, q=6, labels=[f"q{i:02d}" for i in range(1, 7)]).astype(str)
    out["trace_path_length_bin"] = trace_bins.reindex(out["trace_index"]).to_numpy()
    return out


def prepare_new_unit_table(unit: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rr100_index", "model_valid", "sf_quartile", "sf_quartile_label",
        "preferred_sf_cpd", "preferred_tf_hz", "joint_parametric_surface_r2",
    ]
    out = unit.merge(
        assignments[columns], left_on="unit_index", right_on="rr100_index",
        how="left", validate="one_to_one",
    )
    if out["rr100_index"].isna().any():
        raise ValueError("Some matrix units have no new parametric-fit assignment")
    out["historical_sf_group"] = out["sf_group"]
    out["historical_sf_group_label"] = out["sf_group_label"]
    out["sf_group"] = out["sf_quartile"].where(out["model_valid"].astype(bool), "invalid_model")
    out["sf_group_label"] = out["sf_quartile_label"].fillna("invalid parametric fit")
    out["sf_group_definition"] = "quartiles of valid joint-parametric preferred SF"
    out["sf_split_metric"] = out["preferred_sf_cpd"]
    return out


def calculate_unit_changes(curves: pd.DataFrame, unit: pd.DataFrame) -> pd.DataFrame:
    absolute = curves.pivot(
        index=["unit_index", "unit_label", "sf_group"],
        columns="trace_path_length_bin",
        values="unit_contour_matched_ssi_bits_per_spike",
    ).reset_index()
    absolute["last_minus_first_ssi"] = absolute["q06"] - absolute["q01"]
    support = curves.groupby("unit_index", sort=False).agg(
        n_matched_images=("n_matched_images", "first"),
        preferred_orientation_deg=("preferred_orientation_deg", "first"),
        prior_orientation_selectivity_index=("prior_orientation_selectivity_index", "first"),
    ).reset_index()
    metadata = unit[
        ["unit_index", "preferred_sf_cpd", "preferred_tf_hz", "joint_parametric_surface_r2"]
    ].copy()
    return absolute.merge(support, on="unit_index", validate="one_to_one").merge(
        metadata, on="unit_index", validate="one_to_one"
    )


def summarize_changes(changes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        values = changes.loc[changes["sf_group"].eq(group), "last_minus_first_ssi"].dropna().to_numpy(float)
        rows.append(
            {
                "sf_quartile": group,
                "n_units": int(values.size),
                "mean_last_minus_first_ssi": float(np.mean(values)),
                "sem_last_minus_first_ssi": float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0,
                "median_last_minus_first_ssi": float(np.median(values)),
                "fraction_units_positive": float(np.mean(values > 0)),
            }
        )
    return pd.DataFrame(rows)


def leave_one_out_sensitivity(changes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        sub = changes.loc[
            changes["sf_group"].eq(group), ["unit_index", "unit_label", "last_minus_first_ssi"]
        ].dropna().copy()
        full_mean = float(sub["last_minus_first_ssi"].mean())
        candidates = []
        for row in sub.itertuples(index=False):
            leave_out_mean = float(sub.loc[sub["unit_index"].ne(row.unit_index), "last_minus_first_ssi"].mean())
            candidates.append((abs(leave_out_mean - full_mean), row, leave_out_mean))
        _, influential, leave_out_mean = max(candidates, key=lambda item: item[0])
        rows.append(
            {
                "sf_quartile": group,
                "n_units": int(len(sub)),
                "full_mean_last_minus_first_ssi": full_mean,
                "most_influential_unit_index": int(influential.unit_index),
                "most_influential_unit_label": str(influential.unit_label),
                "most_influential_unit_change": float(influential.last_minus_first_ssi),
                "mean_excluding_most_influential_unit": leave_out_mean,
                "change_in_group_mean_when_excluded": leave_out_mean - full_mean,
            }
        )
    return pd.DataFrame(rows)


def select_example_roles(changes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        sub = changes[changes["sf_group"].eq(group)].dropna(subset=["last_minus_first_ssi"]).copy()
        median = float(sub["last_minus_first_ssi"].median())
        choices = [
            ("largest_increase", sub["last_minus_first_ssi"].idxmax(), "maximum q06-minus-q01 SSI"),
            ("largest_decrease", sub["last_minus_first_ssi"].idxmin(), "minimum q06-minus-q01 SSI"),
            ("median_change", (sub["last_minus_first_ssi"] - median).abs().idxmin(), "closest to group-median q06-minus-q01 SSI"),
        ]
        for role, index, criterion in choices:
            row = sub.loc[index]
            rows.append(
                {
                    "selection_method": "algorithmic",
                    "selection_role": role,
                    "criterion": criterion,
                    "sf_quartile": group,
                    "unit_index": int(row["unit_index"]),
                    "unit_label": str(row["unit_label"]),
                    "criterion_value_bits_per_spike": float(row["last_minus_first_ssi"]),
                    "preferred_sf_cpd": float(row["preferred_sf_cpd"]),
                    "preferred_tf_hz": float(row["preferred_tf_hz"]),
                    "preferred_orientation_deg": float(row["preferred_orientation_deg"]),
                    "orientation_selectivity_index": float(row["prior_orientation_selectivity_index"]),
                    "n_matched_images": int(row["n_matched_images"]),
                }
            )
    return pd.DataFrame(rows)


def plot_summary_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    groups: tuple[str, ...],
    value_name: str,
    *,
    title: str,
    ylabel: str,
    context: pd.DataFrame | None = None,
) -> None:
    if context is not None:
        add_trace_path_context_bands(ax, context, include_legend=False)
    for group in groups:
        sub = summary[
            summary["sf_group"].astype(str).eq(group)
            & summary["value_name"].astype(str).eq(value_name)
        ].sort_values("trace_path_length_bin_median_arcmin")
        if sub.empty:
            continue
        x = sub["trace_path_length_bin_median_arcmin"].to_numpy(float)
        y = sub["mean"].to_numpy(float)
        e = sub["sem"].to_numpy(float)
        label = f"{LABELS[group]} (n={int(sub['n_units'].iloc[0])})"
        ax.plot(x, y, marker="o", lw=2.2, ms=4.5, color=COLORS[group], label=label, zorder=4)
        ax.fill_between(x, y - e, y + e, color=COLORS[group], alpha=0.17, lw=0, zorder=2)
    if "delta" in value_name:
        ax.axhline(0, color="0.4", ls=":", lw=1)
    ax.set(xlabel="trace path length bin median (arcmin)", ylabel=ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, color="0.92", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7)


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
        title="Absolute SSI on contour-matched windows", ylabel="SSI (bits/spike)", context=context,
    )
    plot_summary_panel(
        axes[1], summary, EXTREMES, delta,
        title="Movement modulation", ylabel="SSI minus stabilized baseline (bits/spike)", context=context,
    )
    fig.suptitle(
        "Contour-matched unit-window pairs: real trace scale\n"
        "new SF extremes; alignment <= 22.5 deg; OSI >= 0.05; min images/unit = 1",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, out_dir, "012_phase2_contour_matched_sf_q1_q4_curves")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharex=True)
    plot_summary_panel(
        axes[0], summary, GROUPS, absolute,
        title="A. All quartiles: absolute SSI", ylabel="SSI (bits/spike)", context=context,
    )
    plot_summary_panel(
        axes[1], summary, GROUPS, delta,
        title="B. All quartiles: moving minus stabilized", ylabel="SSI difference (bits/spike)", context=context,
    )
    fig.suptitle("Checkpoint 3: contour-matched quartile audit", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.2)
    save_figure(fig, out_dir, "checkpoint_03_contour_matched_all_quartiles")

    bins = sorted(curves["trace_path_length_bin"].astype(str).unique())
    xmap = {key: idx for idx, key in enumerate(bins)}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), sharex=True)
    for ax, group in zip(axes.ravel(), GROUPS, strict=True):
        sub = curves[curves["sf_group"].astype(str).eq(group)]
        for _, unit_curve in sub.groupby("unit_index", sort=True):
            unit_curve = unit_curve.sort_values("trace_path_length_bin")
            x = unit_curve["trace_path_length_bin"].map(xmap).to_numpy(float)
            y = unit_curve["unit_contour_matched_ssi_delta_vs_stabilized"].to_numpy(float)
            ax.plot(x, y, color=COLORS[group], alpha=0.22, lw=0.8)
        mean_sub = summary[
            summary["sf_group"].astype(str).eq(group)
            & summary["value_name"].astype(str).eq(delta)
        ].sort_values("trace_path_length_bin")
        ax.plot(
            mean_sub["trace_path_length_bin"].map(xmap), mean_sub["mean"],
            color=COLORS[group], marker="o", lw=2.4, ms=4.5,
        )
        ax.axhline(0, color="0.45", ls=":", lw=0.9)
        n = sub["unit_index"].nunique()
        ax.set_title(f"{LABELS[group]}: individual units (n={n})", loc="left", fontweight="bold")
        ax.set_ylabel("SSI minus stabilized")
        ax.set_xticks(range(len(bins)), bins)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("trace path bin")
    axes[1, 1].set_xlabel("trace path bin")
    fig.suptitle("Contour-matched unit-level heterogeneity", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.2, w_pad=2.0)
    save_figure(fig, out_dir, "checkpoint_03_contour_matched_unit_heterogeneity")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    input_counts = selection.groupby("sf_group").size().reindex(GROUPS, fill_value=0)
    osi_counts = selection[selection["passes_orientation_selectivity"].astype(bool)].groupby("sf_group").size().reindex(GROUPS, fill_value=0)
    selected_counts = curves.groupby("sf_group")["unit_index"].nunique().reindex(GROUPS, fill_value=0)
    x = np.arange(len(GROUPS))
    width = 0.25
    axes[0].bar(x - width, input_counts, width, color="0.78", label="valid-fit input")
    axes[0].bar(x, osi_counts, width, color="0.48", label="OSI pass")
    axes[0].bar(x + width, selected_counts, width, color=[COLORS[g] for g in GROUPS], label="matched support")
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
    axes[1].set(ylabel="q06 minus q01 SSI (bits/spike)", title="B. Unit-level trace-scale change")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Contour-match selection and change audit", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.4)
    save_figure(fig, out_dir, "checkpoint_03_contour_matched_selection_audit")


def validate_historical_contract(
    movie: pd.DataFrame,
    ssi: np.ndarray,
    unit: pd.DataFrame,
    baseline: dict[str, Any],
    matrix_dir: Path,
) -> dict[str, Any]:
    _, _, reconstructed = build_real_trace_sf_contour_matched_unit_curves(
        movie, ssi, unit,
        image_axis_col="image_edge_axis_deg", sf_groups=["low_sf", "high_sf"],
        match_max_deg=MATCH_MAX_DEG, orthogonal_min_deg=ORTHOGONAL_MIN_DEG,
        min_orientation_selectivity=MIN_OSI, min_matched_images_per_unit=MIN_MATCHED_IMAGES,
        contour_relation="matched", stabilized_baseline=baseline,
    )
    saved_path = matrix_dir / "phase1_phase2_conditioning_v1/phase2_real_trace_sf_contour_matched_summary.csv"
    saved = pd.read_csv(saved_path)
    keys = ["sf_group", "trace_path_length_bin", "value_name"]
    compare = reconstructed.merge(saved, on=keys, suffixes=("_new", "_saved"), validate="one_to_one")
    max_diff = float(np.max(np.abs(compare["mean_new"] - compare["mean_saved"])))
    max_sem_diff = float(np.max(np.abs(compare["sem_new"] - compare["sem_saved"])))
    return {
        "saved_summary": file_identity(saved_path),
        "n_rows_reconstructed": int(len(reconstructed)),
        "n_rows_saved": int(len(saved)),
        "n_rows_matched": int(len(compare)),
        "max_absolute_mean_difference_bits_per_spike": max_diff,
        "max_absolute_sem_difference_bits_per_spike": max_sem_diff,
        "passed": bool(len(compare) == len(saved) == len(reconstructed) and max_diff < 1e-10 and max_sem_diff < 1e-10),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    matrix_dir = Path(args.matrix_dir)
    ssi, _expected, _population, movie, image, _trace, raw_unit = load_matrix_dataset(matrix_dir)
    enriched_unit_path = matrix_dir / "phase1_phase2_conditioning_v1/phase1_unit_summary_with_ssi.csv"
    unit = pd.read_csv(enriched_unit_path)
    if len(unit) != len(raw_unit) or not unit["unit_index"].equals(raw_unit["unit_index"]):
        raise ValueError("Enriched phase-1 unit table is not row-aligned with the SSI matrix unit table")
    movie = add_trace_bins(movie)
    baseline = load_stabilized_baseline(matrix_dir, image, unit)
    if baseline is None:
        raise ValueError("The page-12 contract requires the matched stabilized baseline")
    assignments = pd.read_csv(args.assignments_csv)
    unit_new = prepare_new_unit_table(unit, assignments)
    selection, curves, summary = build_real_trace_sf_contour_matched_unit_curves(
        movie, ssi, unit_new,
        image_axis_col="image_edge_axis_deg", sf_groups=list(GROUPS),
        match_max_deg=MATCH_MAX_DEG, orthogonal_min_deg=ORTHOGONAL_MIN_DEG,
        min_orientation_selectivity=MIN_OSI, min_matched_images_per_unit=MIN_MATCHED_IMAGES,
        contour_relation="matched", stabilized_baseline=baseline,
    )
    changes = calculate_unit_changes(curves, unit_new)
    change_summary = summarize_changes(changes)
    sensitivity = leave_one_out_sensitivity(changes)
    examples = select_example_roles(changes)
    context_source = pd.read_csv(args.trace_context_csv)
    context = trace_microsaccade_path_context_from_frame(
        context_source, source_label="large_fixation_sample_pathle350arcmin", source_path=args.trace_context_csv
    )
    make_figures(selection, curves, summary, changes, context, args.out_dir)
    validation = validate_historical_contract(movie, ssi, unit, baseline, matrix_dir)
    if not validation["passed"]:
        raise ValueError(f"Historical contour-matched reconstruction failed: {validation}")

    selection.to_csv(args.out_dir / "contour_matched_unit_selection.csv", index=False)
    curves.to_csv(args.out_dir / "contour_matched_unit_curves.csv", index=False)
    summary.to_csv(args.out_dir / "contour_matched_group_summary.csv", index=False)
    changes.to_csv(args.out_dir / "contour_matched_unit_changes.csv", index=False)
    change_summary.to_csv(args.out_dir / "contour_matched_change_summary.csv", index=False)
    sensitivity.to_csv(args.out_dir / "contour_matched_leave_one_out_sensitivity.csv", index=False)
    examples.to_csv(args.out_dir / "contour_matched_algorithmic_example_units.csv", index=False)
    context.to_csv(args.out_dir / "trace_path_context_windows.csv", index=False)
    (args.out_dir / "historical_contract_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    selected_counts = curves.groupby("sf_group")["unit_index"].nunique().reindex(GROUPS, fill_value=0).to_dict()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "checkpoint_03_complete",
        "source_test": "PDF page 12: contour-matched low/high-SF unit-first curves",
        "exact_replacement": "Q1 versus Q4; all quartiles retained in audit artifacts",
        "estimand": "for each selected unit and trace-path bin, mean instantaneous SSI over all movie rows whose image contour is matched to the unit; then equal-unit group mean and SEM",
        "selection_contract": {
            "image_axis": "image_edge_axis_deg",
            "alignment_max_deg": MATCH_MAX_DEG,
            "minimum_orientation_selectivity_index": MIN_OSI,
            "minimum_matched_images_per_unit": MIN_MATCHED_IMAGES,
        },
        "stabilized_baseline_contract": "existing trial-mean stabilized SSI, matched by image; not a deterministic static-center oracle",
        "matrix_dir": str(matrix_dir.resolve()),
        "assignments": file_identity(Path(args.assignments_csv)),
        "ssi_matrix": file_identity(matrix_dir / "ssi_matrix.npy"),
        "stabilized_ssi": file_identity(matrix_dir / "stabilized_ssi_by_image.npy"),
        "enriched_phase1_unit_table": file_identity(enriched_unit_path),
        "selected_unit_counts": selected_counts,
        "historical_contract_validation": validation,
        "artifacts": {
            "exact_page_12_analog": "012_phase2_contour_matched_sf_q1_q4_curves.{png,pdf,svg}",
            "all_quartile_summary": "checkpoint_03_contour_matched_all_quartiles.{png,pdf,svg}",
            "unit_heterogeneity": "checkpoint_03_contour_matched_unit_heterogeneity.{png,pdf,svg}",
            "selection_audit": "checkpoint_03_contour_matched_selection_audit.{png,pdf,svg}",
            "selection_table": "contour_matched_unit_selection.csv",
            "unit_curves": "contour_matched_unit_curves.csv",
            "algorithmic_examples": "contour_matched_algorithmic_example_units.csv",
            "leave_one_out_sensitivity": "contour_matched_leave_one_out_sensitivity.csv",
        },
        "not_run": "No contour-orthogonal, aligned-versus-orthogonal, component-path, or final Figure 4 test was regenerated.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = []
    for row in change_summary.itertuples(index=False):
        lines.append(
            f"- {LABELS[row.sf_quartile]}: n={row.n_units}; mean q06-q01={row.mean_last_minus_first_ssi:+.4f} "
            f"bits/spike; median={row.median_last_minus_first_ssi:+.4f}; positive fraction={row.fraction_units_positive:.2f}."
        )
    sensitivity_lines = []
    for row in sensitivity.itertuples(index=False):
        sensitivity_lines.append(
            f"- {LABELS[row.sf_quartile]}: most influential is {row.most_influential_unit_label} "
            f"({row.most_influential_unit_change:+.4f}); group mean without it is "
            f"{row.mean_excluding_most_influential_unit:+.4f} bits/spike."
        )
    readme = f"""# Checkpoint 3: contour-matched SF quartiles

This reproduces the old page-12 unit-first contour-matched test. The exact analog
uses Q1 versus Q4; companion artifacts retain Q2 and Q3 and expose individual-unit
heterogeneity and selection support.

## Unit-level trace-scale changes

{chr(10).join(lines)}

## Leave-one-unit-out sensitivity

{chr(10).join(sensitivity_lines)}

Units require OSI >= {MIN_OSI:g} and at least {MIN_MATCHED_IMAGES} of the 100 image
contours within {MATCH_MAX_DEG:g} degrees of their preferred orientation. The
stabilized reference is the existing trial-mean baseline matched by image.

The algorithmic example table records the largest increase, largest decrease,
and median-like unit in every quartile for a possible map-level drill-down.
The Q1-versus-Q4 exact analog does not retain a high-SF decline: Q1 increases,
while Q4 is nearly flat/slightly positive. Q3 decreases on average, but its
magnitude is sensitive to the large negative change in u054; the median Q3
change remains slightly negative.
The implementation exactly reproduces the saved historical page-12 summary;
see `historical_contract_validation.json`.

No contour-orthogonal or aligned-versus-orthogonal test was run here.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {args.out_dir.resolve()}")
    print(change_summary.to_string(index=False))
    print(f"Selected counts: {selected_counts}")


if __name__ == "__main__":
    main()
