#!/usr/bin/env python3
"""Checkpoint 2: rerun the page-7 phase-2 SF plot with new SF quartiles."""

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
    "checkpoint_02_phase2_sf_by_trace_path"
)
GROUPS = ("sf_q1", "sf_q2", "sf_q3", "sf_q4")
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


def sem(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    return 0.0 if arr.size <= 1 else float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def weighted_ssi(ssi: np.ndarray, expected: np.ndarray, ids: np.ndarray) -> np.ndarray:
    s = np.asarray(ssi[:, ids], dtype=np.float64)
    e = np.asarray(expected[:, ids], dtype=np.float64)
    return np.divide(np.sum(s * e, axis=1), np.maximum(np.sum(e, axis=1), 1e-12))


def load_and_calculate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix_dir = Path(args.matrix_dir)
    ssi = np.load(matrix_dir / "ssi_matrix.npy", mmap_mode="r")
    expected = np.load(matrix_dir / "expected_spikes_matrix.npy", mmap_mode="r")
    baseline_ssi = np.load(matrix_dir / "stabilized_ssi_by_image.npy", mmap_mode="r")
    baseline_expected = np.load(matrix_dir / "stabilized_expected_spikes_by_image.npy", mmap_mode="r")
    movie_source = pd.read_csv(matrix_dir / "movie_feature_table.csv")
    assignments = pd.read_csv(args.assignments_csv)
    required_movie = {"movie_index", "image_index", "trace_index", "rendered_path_length_arcmin"}
    required_assignment = {"rr100_index", "model_valid", "sf_quartile"}
    if missing := required_movie.difference(movie_source.columns):
        raise ValueError(f"Movie table missing {sorted(missing)}")
    if missing := required_assignment.difference(assignments.columns):
        raise ValueError(f"Assignment table missing {sorted(missing)}")
    if ssi.shape != expected.shape or ssi.shape[0] != len(movie_source):
        raise ValueError("Movie matrix shapes do not agree")
    if baseline_ssi.shape != baseline_expected.shape or baseline_ssi.shape[1] != ssi.shape[1]:
        raise ValueError("Stabilized matrix shapes do not agree")
    if assignments["rr100_index"].duplicated().any() or len(assignments) != ssi.shape[1]:
        raise ValueError("Assignments must contain one row per matrix unit")

    movie = movie_source[
        ["movie_index", "image_index", "trace_index", "rendered_path_length_arcmin"]
    ].copy()
    movie["trace_path_length_bin"] = pd.qcut(
        movie_source.drop_duplicates("trace_index").set_index("trace_index")["rendered_path_length_arcmin"],
        q=6,
        labels=[f"q{i:02d}" for i in range(1, 7)],
    ).astype(str).reindex(movie["trace_index"]).to_numpy()
    image_ids = pd.to_numeric(movie["image_index"], errors="raise").astype(int).to_numpy()

    valid_assignments = assignments[assignments["model_valid"].astype(bool)].copy()
    for group in GROUPS:
        ids = valid_assignments.loc[valid_assignments["sf_quartile"].eq(group), "rr100_index"].astype(int).to_numpy()
        if ids.size == 0:
            raise ValueError(f"No valid units in {group}")
        moving_weighted = weighted_ssi(ssi, expected, ids)
        moving_mean = np.nanmean(np.asarray(ssi[:, ids], dtype=np.float64), axis=1)
        baseline_weighted_by_image = weighted_ssi(baseline_ssi, baseline_expected, ids)
        baseline_mean_by_image = np.nanmean(np.asarray(baseline_ssi[:, ids], dtype=np.float64), axis=1)
        movie[f"{group}_weighted_ssi"] = moving_weighted
        movie[f"{group}_mean_ssi"] = moving_mean
        movie[f"{group}_weighted_stabilized_ssi"] = baseline_weighted_by_image[image_ids]
        movie[f"{group}_mean_stabilized_ssi"] = baseline_mean_by_image[image_ids]
        movie[f"{group}_weighted_delta"] = moving_weighted - baseline_weighted_by_image[image_ids]
        movie[f"{group}_mean_delta"] = moving_mean - baseline_mean_by_image[image_ids]
        movie[f"{group}_weighting_gap"] = moving_weighted - moving_mean
        movie[f"{group}_expected_spikes"] = np.sum(
            np.asarray(expected[:, ids], dtype=np.float64), axis=1
        )

    trace_value_cols = [col for col in movie.columns if col.startswith("sf_q")]
    trace_values = (
        movie.groupby(["trace_index", "trace_path_length_bin"], observed=True, sort=True)
        .agg(rendered_path_length_arcmin=("rendered_path_length_arcmin", "first"), **{col: (col, "mean") for col in trace_value_cols})
        .reset_index()
    )
    return movie, trace_values, assignments


def summarize(movie: pd.DataFrame, trace_values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    metric_suffixes = ("weighted_ssi", "mean_ssi", "weighted_delta", "mean_delta", "weighting_gap", "expected_spikes")
    for level, frame, n_label in (("movie", movie, "n_movies"), ("trace", trace_values, "n_traces")):
        for path_bin, sub in frame.groupby("trace_path_length_bin", observed=True, sort=True):
            x = float(sub["rendered_path_length_arcmin"].median())
            for group in GROUPS:
                for suffix in metric_suffixes:
                    col = f"{group}_{suffix}"
                    rows.append(
                        {
                            "sampling_level": level,
                            "trace_path_length_bin": str(path_bin),
                            "path_median_arcmin": x,
                            "sf_quartile": group,
                            "metric": suffix,
                            "mean": float(sub[col].mean()),
                            "sem": sem(sub[col]),
                            n_label: int(sub[col].notna().sum()),
                        }
                    )
    summary = pd.DataFrame(rows)

    trend_rows = []
    for group in GROUPS:
        for suffix in metric_suffixes:
            col = f"{group}_{suffix}"
            valid = trace_values[["rendered_path_length_arcmin", col]].dropna()
            by_bin = summary[
                summary["sampling_level"].eq("trace")
                & summary["sf_quartile"].eq(group)
                & summary["metric"].eq(suffix)
            ].sort_values("trace_path_length_bin")
            trend_rows.append(
                {
                    "sf_quartile": group,
                    "metric": suffix,
                    "n_traces": int(len(valid)),
                    "spearman_path_vs_metric": float(valid.corr(method="spearman").iloc[0, 1]),
                    "first_bin_mean": float(by_bin["mean"].iloc[0]),
                    "last_bin_mean": float(by_bin["mean"].iloc[-1]),
                    "last_minus_first_bin_mean": float(by_bin["mean"].iloc[-1] - by_bin["mean"].iloc[0]),
                }
            )
    return summary, pd.DataFrame(trend_rows)


def validate_historical_contract(matrix_dir: Path, movie: pd.DataFrame) -> dict[str, Any]:
    """Reconstruct the three historical curves and compare them with the saved phase-2 table."""
    unit = pd.read_csv(matrix_dir / "unit_feature_table.csv")
    saved_path = matrix_dir / "phase1_phase2_conditioning_v1/phase1_movie_analysis_table.csv"
    saved_cols = [
        "trace_path_length_bin",
        "sf_low_sf_weighted_ssi",
        "sf_middle_sf_weighted_ssi",
        "sf_high_sf_weighted_ssi",
    ]
    saved = pd.read_csv(saved_path, usecols=saved_cols)
    ssi = np.load(matrix_dir / "ssi_matrix.npy", mmap_mode="r")
    expected = np.load(matrix_dir / "expected_spikes_matrix.npy", mmap_mode="r")
    differences: dict[str, float] = {}
    counts: dict[str, int] = {}
    for group in ("low_sf", "middle_sf", "high_sf"):
        ids = unit.loc[unit["sf_group"].astype(str).eq(group), "unit_index"].astype(int).to_numpy()
        reconstructed = weighted_ssi(ssi, expected, ids)
        stored = pd.to_numeric(saved[f"sf_{group}_weighted_ssi"], errors="raise").to_numpy(float)
        differences[group] = float(np.max(np.abs(reconstructed - stored)))
        counts[group] = int(ids.size)
    return {
        "saved_phase2_table": file_identity(saved_path),
        "old_group_counts": counts,
        "max_absolute_difference_bits_per_spike": differences,
        "trace_path_bin_fraction_exactly_matching": float(
            np.mean(saved["trace_path_length_bin"].astype(str).to_numpy() == movie["trace_path_length_bin"].astype(str).to_numpy())
        ),
        "passed": bool(
            max(differences.values()) < 1e-6
            and np.all(saved["trace_path_length_bin"].astype(str).to_numpy() == movie["trace_path_length_bin"].astype(str).to_numpy())
        ),
    }


def plot_curves(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    sampling_level: str,
    context: pd.DataFrame | None = None,
) -> None:
    if context is not None:
        add_trace_path_context_bands(ax, context, include_legend=False)
    for group in GROUPS:
        sub = summary[
            summary["sampling_level"].eq(sampling_level)
            & summary["sf_quartile"].eq(group)
            & summary["metric"].eq(metric)
        ].sort_values("trace_path_length_bin")
        ax.errorbar(
            sub["path_median_arcmin"], sub["mean"], yerr=sub["sem"],
            color=COLORS[group], marker="o", ms=4, lw=1.8, capsize=2,
            label=LABELS[group],
        )
    ax.set(xlabel="trace path length bin median (arcmin)", ylabel=ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.axhline(0, color="0.7", lw=0.8, zorder=0) if "delta" in metric or "gap" in metric else None
    ax.spines[["top", "right"]].set_visible(False)


def make_figures(summary: pd.DataFrame, context: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    plot_curves(
        ax, summary, "weighted_ssi", title="SF quartile SSI across real trace scale",
        ylabel="group spike-weighted SSI (bits/spike)", sampling_level="movie", context=context,
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir, "007_phase2_sf_quartiles_by_trace_path")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    plot_curves(
        axes[0, 0], summary, "weighted_ssi", title="A. Page-7 estimand: spike-weighted absolute SSI",
        ylabel="weighted SSI (bits/spike)", sampling_level="trace", context=context,
    )
    plot_curves(
        axes[0, 1], summary, "mean_ssi", title="B. Equal-unit absolute SSI",
        ylabel="mean unit SSI (bits/spike)", sampling_level="trace",
    )
    plot_curves(
        axes[1, 0], summary, "weighted_delta", title="C. Moving minus stabilized, spike weighted",
        ylabel="SSI difference (bits/spike)", sampling_level="trace",
    )
    plot_curves(
        axes[1, 1], summary, "weighting_gap", title="D. Spike-weighting effect",
        ylabel="weighted minus equal-unit SSI", sampling_level="trace",
    )
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("BackImage real-trace phase-2 SF regrouping checkpoint", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(
        0.02, 0.935,
        "Error bars in this audit are SEM across traces after averaging the 100 images per trace; the standalone page-7 analog retains the historical movie-row SEM.",
        fontsize=8.2, color="0.35",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.4, w_pad=2.2)
    save_figure(fig, out_dir, "checkpoint_02_phase2_sf_quartile_audit")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    movie, trace_values, assignments = load_and_calculate(args)
    summary, trends = summarize(movie, trace_values)
    historical_validation = validate_historical_contract(Path(args.matrix_dir), movie)
    if not historical_validation["passed"]:
        raise ValueError(f"Historical page-7 reconstruction failed: {historical_validation}")
    context_source = pd.read_csv(args.trace_context_csv)
    context = trace_microsaccade_path_context_from_frame(
        context_source, source_label="large_fixation_sample_pathle350arcmin", source_path=args.trace_context_csv
    )
    make_figures(summary, context, args.out_dir)

    assignments.to_csv(args.out_dir / "unit_assignments_used.csv", index=False)
    movie.to_csv(args.out_dir / "phase2_sf_quartile_movie_values.csv.gz", index=False, compression="gzip")
    trace_values.to_csv(args.out_dir / "phase2_sf_quartile_trace_values.csv", index=False)
    summary.to_csv(args.out_dir / "phase2_sf_quartile_bin_summary.csv", index=False)
    trends.to_csv(args.out_dir / "phase2_sf_quartile_trends.csv", index=False)
    context.to_csv(args.out_dir / "trace_path_context_windows.csv", index=False)
    (args.out_dir / "historical_contract_validation.json").write_text(
        json.dumps(historical_validation, indent=2) + "\n", encoding="utf-8"
    )

    key_trends = trends[trends["metric"].isin(["weighted_ssi", "mean_ssi", "weighted_delta", "weighting_gap"])].copy()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "checkpoint_02_complete",
        "source_test": "PDF page 7: phase2 SF-group spike-weighted absolute SSI across six trace-path bins",
        "exact_estimand": "per movie, expected-spike-weighted mean unit SSI within each valid new SF quartile; then mean and SEM across movie rows in each trace-path bin",
        "audit_estimand": "the same metrics averaged across images within trace before bin-level mean and SEM across traces",
        "stabilized_baseline_contract": "trial-mean stabilized SSI matched by image_index; used only in audit delta panels, not the exact page-7 analog",
        "matrix_dir": str(Path(args.matrix_dir).resolve()),
        "sources": {
            name: file_identity(Path(args.matrix_dir) / filename)
            for name, filename in {
                "ssi": "ssi_matrix.npy",
                "expected_spikes": "expected_spikes_matrix.npy",
                "stabilized_ssi": "stabilized_ssi_by_image.npy",
                "stabilized_expected_spikes": "stabilized_expected_spikes_by_image.npy",
                "movie_table": "movie_feature_table.csv",
            }.items()
        },
        "assignments": file_identity(Path(args.assignments_csv)),
        "trace_context": file_identity(Path(args.trace_context_csv)),
        "historical_contract_validation": historical_validation,
        "n_movies": int(len(movie)),
        "n_traces": int(len(trace_values)),
        "n_valid_units": int(assignments["model_valid"].astype(bool).sum()),
        "n_invalid_units_excluded": int((~assignments["model_valid"].astype(bool)).sum()),
        "quartile_counts": assignments.loc[assignments["model_valid"].astype(bool), "sf_quartile"].value_counts().sort_index().to_dict(),
        "artifacts": {
            "exact_page_7_analog": "007_phase2_sf_quartiles_by_trace_path.{png,pdf,svg}",
            "audit_sheet": "checkpoint_02_phase2_sf_quartile_audit.{png,pdf,svg}",
            "movie_values": "phase2_sf_quartile_movie_values.csv.gz",
            "trace_values": "phase2_sf_quartile_trace_values.csv",
            "bin_summary": "phase2_sf_quartile_bin_summary.csv",
            "trends": "phase2_sf_quartile_trends.csv",
            "historical_validation": "historical_contract_validation.json",
        },
        "not_run": "No contour-conditioned, unit-image orientation, across/along component-path, or final Figure 4 tests were regenerated.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = []
    for row in key_trends.itertuples(index=False):
        lines.append(
            f"- {LABELS[row.sf_quartile]}, `{row.metric}`: rho={row.spearman_path_vs_metric:+.3f}; "
            f"last-first={row.last_minus_first_bin_mean:+.4f} bits/spike."
        )
    readme = f"""# Checkpoint 2: page-7 SF quartiles across trace scale

The standalone figure reproduces the page-7 contract with the new parametric-fit
SF quartiles: expected-spike-weighted absolute SSI, averaged over movie rows in
each of the original six trace-path bins. The audit sheet uses trace-first error
bars and separates equal-unit SSI, the moving-minus-stabilized difference, and
the effect of spike weighting.

## Trace-level trends

{chr(10).join(lines)}

The stabilized reference is the existing trial-mean baseline matched by image;
it is not a deterministic static-center oracle. The exact page-7 analog does
not subtract it. Because every trace contains the same 100 images, subtracting
that reference shifts each quartile curve vertically but does not change its
trace-path trend. Fifteen invalid parametric fits are excluded without imputation.

The implementation was also run with the historical low/middle/high assignments
and reproduces every saved page-7 movie value to within 1e-6 bits/spike, with
100% agreement in trace-path-bin labels. See `historical_contract_validation.json`.

No contour-conditioned, orientation-matched, component-path, or final Figure 4
analysis was run at this checkpoint. See `manifest.json` for source identities.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {args.out_dir.resolve()}")
    print(key_trends.to_string(index=False))


if __name__ == "__main__":
    main()
