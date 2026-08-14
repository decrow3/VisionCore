#!/usr/bin/env python3
"""Make an interim corrected-cache Figure 4 population plot using SF quartiles.

The recorded SF-curve validation gate is applied before rank-balanced
quartiles are assigned.  Only complete balanced production-cache rounds are
analyzed; the input assembly is therefore immutable even if scoring resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSEMBLED = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "assembled/rounds_000_011_n012_quartile_snapshot_v1"
)
DEFAULT_COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
DEFAULT_ASSIGNMENTS = ROOT / (
    "outputs/fig4_active_sensing/backimage_real_trace_sf_halves_recorded_validated_r0p5_v1/"
    "sf_half_recorded_validated_unit_assignments.csv"
)
DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected_quartile_population_"
    "rounds000_011_checkpoint_v2"
)
GROUPS = ("sf_q1", "sf_q2", "sf_q3", "sf_q4")
LABELS = {
    "sf_q1": "Q1 · lowest SF",
    "sf_q2": "Q2",
    "sf_q3": "Q3",
    "sf_q4": "Q4 · highest SF",
}
# Deliberately non-sequential hues keep the two middle quartiles legible.
COLORS = {
    "sf_q1": "#0072B2",
    "sf_q2": "#009E73",
    "sf_q3": "#E69F00",
    "sf_q4": "#CC79A7",
}
SEED = 20260813


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembled-dir", type=Path, default=DEFAULT_ASSEMBLED)
    parser.add_argument("--cohort-dir", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--assignments-csv", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=4000)
    parser.add_argument("--n-path-bins", type=int, default=7)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "sha256": digest.hexdigest()}


def population_ssi(info: np.ndarray, spikes: np.ndarray, units: np.ndarray) -> np.ndarray:
    return info[:, units].sum(axis=1) / np.maximum(spikes[:, units].sum(axis=1), 1e-12)


def residualize(values: np.ndarray, image_ids: np.ndarray) -> np.ndarray:
    return values - pd.Series(values).groupby(image_ids).transform("mean").to_numpy()


def within_image_slope(x: np.ndarray, y: np.ndarray, image_ids: np.ndarray) -> float:
    xx = residualize(x, image_ids)
    yy = residualize(y, image_ids)
    denominator = float(np.dot(xx, xx))
    return float(np.dot(xx, yy) / denominator) if denominator > 0 else float("nan")


def bootstrap_summary(
    x: np.ndarray,
    y: np.ndarray,
    image_ids: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    images = np.unique(image_ids)
    image_means = pd.DataFrame({"image": image_ids, "y": y}).groupby("image").y.mean()
    numerators = np.empty(len(images), dtype=float)
    denominators = np.empty(len(images), dtype=float)
    mean_values = image_means.reindex(images).to_numpy(float)
    for ordinal, image in enumerate(images):
        use = image_ids == image
        xx = x[use] - x[use].mean()
        yy = y[use] - y[use].mean()
        numerators[ordinal] = float(np.dot(xx, yy))
        denominators[ordinal] = float(np.dot(xx, xx))
    sampled_ordinals = rng.integers(0, len(images), size=(n_bootstrap, len(images)))
    slopes = numerators[sampled_ordinals].sum(axis=1) / denominators[sampled_ordinals].sum(axis=1)
    means = mean_values[sampled_ordinals].mean(axis=1)
    slope_ci = np.quantile(slopes, [0.025, 0.5, 0.975])
    mean_ci = np.quantile(means, [0.025, 0.5, 0.975])
    return {
        "path_slope": within_image_slope(x, y, image_ids),
        "path_slope_bootstrap_median": float(slope_ci[1]),
        "path_slope_ci_low": float(slope_ci[0]),
        "path_slope_ci_high": float(slope_ci[2]),
        "path_slope_probability_gt_zero": float(np.mean(slopes > 0)),
        "mean_delta_ssi": float(y.mean()),
        "mean_delta_bootstrap_median": float(mean_ci[1]),
        "mean_delta_ci_low": float(mean_ci[0]),
        "mean_delta_ci_high": float(mean_ci[2]),
        "mean_delta_probability_gt_zero": float(np.mean(means > 0)),
    }


def assign_validated_quartiles(assignments: pd.DataFrame) -> pd.DataFrame:
    audit = assignments.copy()
    required = {"rr100_index", "preferred_sf_cpd", "recorded_validation_pass"}
    missing = required.difference(audit.columns)
    if missing:
        raise ValueError(f"Assignment table lacks {sorted(missing)}")
    valid = audit[audit.recorded_validation_pass.astype(bool)].sort_values(
        ["preferred_sf_cpd", "rr100_index"]
    )
    if len(valid) != 61:
        raise ValueError(f"Expected 61 recorded-validated units, found {len(valid)}")
    audit["sf_quartile"] = "excluded"
    for group, indices in zip(GROUPS, np.array_split(valid.index.to_numpy(), 4), strict=True):
        audit.loc[indices, "sf_quartile"] = group
    audit["sf_quartile_label"] = audit.sf_quartile.map(LABELS).fillna("excluded by fit gate")
    audit["quartile_definition"] = (
        "rank-balanced quartiles after model-valid and recorded SF-curve r >= 0.5 gate; "
        "sorted by preferred_sf_cpd then rr100_index"
    )
    return audit


def image_adjusted_bins(
    x: np.ndarray,
    y: np.ndarray,
    image_ids: np.ndarray,
    *,
    n_bins: int,
    group: str,
) -> pd.DataFrame:
    adjusted_x = residualize(x, image_ids) + float(x.mean())
    adjusted_y = residualize(y, image_ids) + float(y.mean())
    bin_id = pd.qcut(pd.Series(adjusted_x).rank(method="first"), n_bins, labels=False).to_numpy(int)
    rows = []
    for path_bin in range(n_bins):
        use = bin_id == path_bin
        image_means = pd.DataFrame(
            {"image": image_ids[use], "x": adjusted_x[use], "y": adjusted_y[use]}
        ).groupby("image", as_index=False).mean()
        rows.append(
            {
                "sf_quartile": group,
                "path_bin": path_bin,
                "path_median_arcmin": float(np.median(adjusted_x[use])),
                "delta_ssi_mean": float(adjusted_y[use].mean()),
                "delta_ssi_sem_across_images": float(image_means.y.sem()),
                "n_conditions": int(use.sum()),
                "n_images": int(image_means.image.nunique()),
            }
        )
    return pd.DataFrame(rows)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def main() -> None:
    args = parse_args()
    assembled = args.assembled_dir.resolve()
    cohort = args.cohort_dir.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)

    assembly_manifest = json.loads((assembled / "manifest.json").read_text(encoding="utf-8"))
    condition = pd.read_csv(assembled / "condition_index.csv")
    expected_rows = int(assembly_manifest["n_complete_rounds"]) * 1000
    if len(condition) != expected_rows:
        raise ValueError(f"Expected {expected_rows} balanced conditions, found {len(condition)}")
    for round_index, frame in condition.groupby("round_index"):
        if len(frame) != 1000 or frame.image_index.nunique() != 100 or frame.trace_index.nunique() != 1000:
            raise ValueError(f"Round {round_index} is not a complete balanced round")

    traces = pd.read_csv(cohort / "corrected1000_traces.csv")
    path_column = "corrected_dpi_crop120_path_length_arcmin"
    condition = condition.merge(
        traces[["trace_index", path_column]], on="trace_index", how="left", validate="many_to_one"
    )
    if condition[path_column].isna().any():
        raise ValueError("Some conditions lack corrected dpi_pix path lengths")

    assignment_audit = assign_validated_quartiles(pd.read_csv(args.assignments_csv))
    assignment_audit.to_csv(out / "recorded_validated_sf_quartile_assignments.csv", index=False)
    groups = {
        group: assignment_audit.loc[assignment_audit.sf_quartile.eq(group), "rr100_index"].to_numpy(int)
        for group in GROUPS
    }

    moving_info = np.load(assembled / "moving_information_numerator_bits_spikes.npy", mmap_mode="r")
    moving_spikes = np.load(assembled / "moving_expected_spikes.npy", mmap_mode="r")
    with np.load(assembled / "stabilized_by_image_sufficient_statistics.npz") as baseline:
        baseline_spikes = np.asarray(baseline["expected_spikes"], dtype=float)
        baseline_info = np.asarray(baseline["movie_ssi_bits_per_spike"], dtype=float) * baseline_spikes

    image_ids = condition.image_index.to_numpy(int)
    round_ids = condition.round_index.to_numpy(int)
    paths = condition[path_column].to_numpy(float)
    rng = np.random.default_rng(SEED)
    estimate_rows: list[dict[str, Any]] = []
    bin_tables: list[pd.DataFrame] = []
    convergence_rows: list[dict[str, Any]] = []
    response_by_group: dict[str, np.ndarray] = {}
    for group, units in groups.items():
        moving = population_ssi(moving_info, moving_spikes, units)
        stabilized = population_ssi(baseline_info, baseline_spikes, units)[image_ids]
        delta = moving - stabilized
        response_by_group[group] = delta
        summary = bootstrap_summary(
            paths, delta, image_ids, n_bootstrap=int(args.n_bootstrap), rng=rng
        )
        estimate_rows.append(
            {
                "sf_quartile": group,
                "sf_quartile_label": LABELS[group],
                "n_units": int(len(units)),
                "preferred_sf_min_cpd": float(assignment_audit.loc[assignment_audit.sf_quartile.eq(group), "preferred_sf_cpd"].min()),
                "preferred_sf_max_cpd": float(assignment_audit.loc[assignment_audit.sf_quartile.eq(group), "preferred_sf_cpd"].max()),
                "n_conditions": int(len(condition)),
                "n_complete_rounds": int(assembly_manifest["n_complete_rounds"]),
                **summary,
            }
        )
        bin_tables.append(
            image_adjusted_bins(paths, delta, image_ids, n_bins=int(args.n_path_bins), group=group)
        )
        for cumulative_rounds in range(1, int(assembly_manifest["n_complete_rounds"]) + 1):
            use = round_ids < cumulative_rounds
            convergence_rows.append(
                {
                    "sf_quartile": group,
                    "n_complete_rounds": cumulative_rounds,
                    "n_conditions": int(use.sum()),
                    "within_image_path_slope": within_image_slope(paths[use], delta[use], image_ids[use]),
                }
            )

    estimates = pd.DataFrame(estimate_rows)
    bins = pd.concat(bin_tables, ignore_index=True)
    convergence = pd.DataFrame(convergence_rows)
    estimates.to_csv(out / "quartile_population_estimates.csv", index=False)
    bins.to_csv(out / "quartile_path_bin_curves.csv", index=False)
    convergence.to_csv(out / "quartile_cumulative_round_convergence.csv", index=False)

    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.1), constrained_layout=True)
    ax = axes[0, 0]
    for group in GROUPS:
        sub = bins[bins.sf_quartile.eq(group)]
        n_units = len(groups[group])
        ax.errorbar(
            sub.path_median_arcmin,
            sub.delta_ssi_mean,
            yerr=sub.delta_ssi_sem_across_images,
            color=COLORS[group],
            marker="o",
            ms=4.5,
            lw=2.0,
            capsize=2,
            label=f"{LABELS[group]} (n={n_units})",
        )
    ax.axhline(0, color="0.55", lw=0.8)
    ax.set(
        title="A  Corrected path-length response",
        xlabel="within-image-adjusted retinal path length (arcmin)",
        ylabel="FEM − stabilized population SSI (bits/spike)",
    )
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[0, 1]
    positions = np.arange(4)
    for position, group in enumerate(GROUPS):
        row = estimates[estimates.sf_quartile.eq(group)].iloc[0]
        ax.errorbar(
            position,
            row.path_slope,
            yerr=[[row.path_slope - row.path_slope_ci_low], [row.path_slope_ci_high - row.path_slope]],
            fmt="o",
            ms=7,
            color=COLORS[group],
            capsize=4,
        )
    ax.axhline(0, color="0.55", lw=0.8)
    ax.set_xticks(positions, ["Q1", "Q2", "Q3", "Q4"])
    ax.set(
        title="B  Within-image path slopes",
        ylabel="ΔSSI slope (bits/spike/arcmin)",
        xlabel="preferred-SF quartile after recorded-fit gate",
    )

    ax = axes[1, 0]
    for position, group in enumerate(GROUPS):
        row = estimates[estimates.sf_quartile.eq(group)].iloc[0]
        ax.errorbar(
            position,
            row.mean_delta_ssi,
            yerr=[[row.mean_delta_ssi - row.mean_delta_ci_low], [row.mean_delta_ci_high - row.mean_delta_ssi]],
            fmt="o",
            ms=7,
            color=COLORS[group],
            capsize=4,
        )
    ax.axhline(0, color="0.55", lw=0.8)
    ax.set_xticks(positions, ["Q1", "Q2", "Q3", "Q4"])
    ax.set(
        title="C  Mean FEM benefit",
        ylabel="mean FEM − stabilized population SSI (bits/spike)",
        xlabel="preferred-SF quartile after recorded-fit gate",
    )

    ax = axes[1, 1]
    for group in GROUPS:
        sub = convergence[convergence.sf_quartile.eq(group)]
        ax.plot(
            sub.n_complete_rounds,
            sub.within_image_path_slope,
            color=COLORS[group],
            marker="o",
            ms=3.5,
            lw=1.8,
            label=LABELS[group],
        )
    ax.axhline(0, color="0.55", lw=0.8)
    ax.set(
        title="D  Interim convergence",
        xlabel="complete balanced rounds included",
        ylabel="cumulative within-image path slope",
        xticks=[1, 3, 6, 9, 12],
    )

    ranges = "; ".join(
        f"{group[-2:].upper()}: {row.preferred_sf_min_cpd:.2f}–{row.preferred_sf_max_cpd:.2f} c/deg"
        for group, row in ((g, estimates[estimates.sf_quartile.eq(g)].iloc[0]) for g in GROUPS)
    )
    fig.suptitle(
        "Corrected Figure 4 population result by preferred-SF quartile\n"
        "12 complete balanced rounds · 12,000 movies · explicit recorded history · corrected dpi_pix trajectories",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        f"Recorded-fit gate first (r ≥ 0.5; 61 units). {ranges}. Error bars: 95% image-cluster bootstrap in B/C; SEM across image means in A.",
        ha="center",
        va="top",
        fontsize=8,
        color="0.3",
    )
    stem = "ssi_figure_v4_corrected_cache_sf_quartiles_rounds000_011_interim_v2"
    for suffix in ("pdf", "png", "svg"):
        kwargs = {"dpi": 220} if suffix == "png" else {}
        fig.savefig(out / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)

    result_summary = {
        row.sf_quartile: {
            "n_units": int(row.n_units),
            "preferred_sf_range_cpd": [float(row.preferred_sf_min_cpd), float(row.preferred_sf_max_cpd)],
            "mean_delta_ssi": float(row.mean_delta_ssi),
            "mean_delta_ci95": [float(row.mean_delta_ci_low), float(row.mean_delta_ci_high)],
            "path_slope": float(row.path_slope),
            "path_slope_ci95": [float(row.path_slope_ci_low), float(row.path_slope_ci_high)],
            "path_slope_probability_gt_zero": float(row.path_slope_probability_gt_zero),
        }
        for row in estimates.itertuples(index=False)
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "interim_corrected_quartile_figure_complete",
        "scope": (
            "12 complete balanced rounds (12% of the intended 100x1000 cache). "
            "This is an interim corrected-cache result, not the complete half-bank or full-bank result."
        ),
        "estimand": (
            "expected-spike-weighted population spatial SSI for moving explicit-history movies minus "
            "the matched-image stabilized baseline"
        ),
        "split": (
            "rank-balanced preferred-SF quartiles among the 61 model-valid units with "
            "recorded_sf_curve_r_full_support >= 0.5"
        ),
        "uncertainty": {"cluster": "image", "n_bootstrap": int(args.n_bootstrap), "seed": SEED},
        "source_assembly_manifest": file_identity(assembled / "manifest.json"),
        "source_assignments": file_identity(args.assignments_csv),
        "source_trace_table": file_identity(cohort / "corrected1000_traces.csv"),
        "n_complete_rounds": int(assembly_manifest["n_complete_rounds"]),
        "n_conditions": int(len(condition)),
        "n_images": int(condition.image_index.nunique()),
        "n_unique_traces": int(condition.trace_index.nunique()),
        "results": result_summary,
        "outputs": {
            "figure_pdf": str((out / f"{stem}.pdf").resolve()),
            "figure_png": str((out / f"{stem}.png").resolve()),
            "figure_svg": str((out / f"{stem}.svg").resolve()),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Corrected Figure 4 SF-quartile population checkpoint\n\n"
        "This non-overwriting figure uses the first 12 complete balanced rounds of the corrected "
        "100 × 1,000 production cache. The recorded-fit gate is applied before the preferred-SF "
        "quartile split. See `manifest.json` for the frozen input identities and interim-scope caveat.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
