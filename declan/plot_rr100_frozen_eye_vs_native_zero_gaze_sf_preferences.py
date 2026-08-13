#!/usr/bin/env python3
"""Compare earlier frozen-eye SF fits with current native zero-gaze SF factors."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = ROOT / "outputs/fig/ssi_figure_v2/panels/previous_sf_tuning_groups/previous_sf_tuning_unit_summary.csv"
DEFAULT_NATIVE = ROOT / (
    "outputs/redundancy_resolved_v1_twin/"
    "rr100_zero_gaze_separable_sf_tf_factorization_v1/separable_fit_unit_summary.csv"
)
DEFAULT_OUT = ROOT / (
    "outputs/redundancy_resolved_v1_twin/"
    "rr100_zero_gaze_separable_sf_tf_factorization_v1/frozen_eye_fit_comparison_v1"
)

FROZEN_COLUMN = "dynamic_log_gaussian_marginal_sf_cpd"
NATIVE_SAMPLED_COLUMN = "preferred_sf_cpd_sampled"
NATIVE_WEIGHTED_COLUMN = "weighted_geometric_sf_cpd"
GROUP_COLORS = {"low_sf": "#0072B2", "middle_sf": "#009E73", "high_sf": "#D55E00"}
GROUP_LABELS = {"low_sf": "earlier low-SF fit", "middle_sf": "earlier middle-SF fit", "high_sf": "earlier high-SF fit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-unit-summary", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--native-fit-summary", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def identity(path: Path) -> dict[str, Any]:
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frozen = pd.read_csv(args.frozen_unit_summary)
    native = pd.read_csv(args.native_fit_summary)
    native = native[native["surface_definition"].eq("preferred_orientation_abs_tf")].copy()
    joined = frozen.merge(
        native,
        left_on="unit_index",
        right_on="rr100_index",
        how="inner",
        validate="one_to_one",
        suffixes=("_frozen", "_native"),
    )
    if len(joined) != 100:
        raise ValueError(f"Expected 100 matched RR units, found {len(joined)}")
    joined["paired_comparison_usable"] = (
        joined["dynamic_log_gaussian_marginal_fit_ok"].astype(bool)
        & joined["responsive_max_f1_flag"].astype(bool)
        & np.isfinite(joined[FROZEN_COLUMN])
        & np.isfinite(joined[NATIVE_SAMPLED_COLUMN])
    )
    paired = joined[joined["paired_comparison_usable"]].copy()
    excluded = joined[~joined["paired_comparison_usable"]].copy()

    frozen_values = paired[FROZEN_COLUMN].to_numpy(dtype=float)
    native_sampled = paired[NATIVE_SAMPLED_COLUMN].to_numpy(dtype=float)
    native_weighted = paired[NATIVE_WEIGHTED_COLUMN].to_numpy(dtype=float)
    plot_min = 0.0125
    plot_max = 16.0
    log_edges = np.geomspace(plot_min, plot_max, 19)
    frozen_probe = np.asarray([0.0125, 0.05, 0.2, 0.8, 3.2, 12.8])
    native_probe = np.asarray([1.0, np.sqrt(2), 2.0, 2 * np.sqrt(2), 4.0, 4 * np.sqrt(2), 8.0, 8 * np.sqrt(2)])

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), layout="constrained")
    ax = axes[0]
    ax.hist(frozen_values, bins=log_edges, color="#6A3D9A", alpha=0.82, edgecolor="white")
    for value in frozen_probe:
        ax.axvline(value, color="0.75", lw=0.55, zorder=0)
    ax.axvline(0.3713343185, color="0.2", ls="--", lw=1.1, label="1 cycle / 101-px window")
    ax.set(
        xscale="log",
        xlim=(plot_min, plot_max),
        xlabel="fit-derived preferred SF (cpd)",
        ylabel="matched responsive RR100 units",
        title="A. Earlier frozen-eye log-Gaussian fits",
    )
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.text(
        0.02,
        0.02,
        "phase-RMS; averaged over TF and orientation\n101-px center-map readout; n=91 matched units",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        color="0.25",
    )

    ax = axes[1]
    unique, counts = np.unique(native_sampled, return_counts=True)
    widths = unique * 0.18
    ax.bar(unique, counts, width=widths, color="#0072B2", alpha=0.82, edgecolor="white", label="sampled factor peak")
    ax.scatter(native_weighted, np.full_like(native_weighted, -0.8), marker="|", s=75, color="#D55E00", alpha=0.65, label="factor-weighted center")
    for value in native_probe:
        ax.axvline(value, color="0.82", lw=0.55, zorder=0)
    ax.set(
        xscale="log",
        xlim=(plot_min, plot_max),
        xlabel="separable SF-factor preference (cpd)",
        ylabel="matched responsive RR100 units",
        title="B. Current native zero-gaze SF factors",
    )
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.text(
        0.02,
        0.02,
        "native fitted-unit readout; preferred orientation\n|TF|-folded rank-one factor; n=91 matched units",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        color="0.25",
    )

    ax = axes[2]
    for group in ["low_sf", "middle_sf", "high_sf"]:
        subset = paired[paired["sf_group"].eq(group)]
        ax.scatter(
            subset[FROZEN_COLUMN],
            subset[NATIVE_SAMPLED_COLUMN],
            s=31,
            color=GROUP_COLORS[group],
            alpha=0.78,
            label=f"{GROUP_LABELS[group]} (n={len(subset)})",
        )
    diagonal = np.geomspace(plot_min, plot_max, 100)
    ax.plot(diagonal, diagonal, color="0.25", ls="--", lw=1.0, label="identity")
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(plot_min, plot_max),
        ylim=(plot_min, plot_max),
        xlabel="earlier frozen-eye fitted SF (cpd)",
        ylabel="current native sampled SF-factor peak (cpd)",
        title="C. Same-unit comparison",
    )
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    for axis in axes:
        axis.grid(True, which="major", color="0.91", lw=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    figure_title = "RR100 SF preference: earlier frozen-eye fits versus current native zero-gaze factors"
    fig.suptitle(figure_title, fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.035,
        "The panels compare unit identities, not equivalent measurement contracts. Nine current near-silent units are excluded from both plotted marginals.",
        ha="center",
        fontsize=8,
        color="0.25",
    )
    png = args.out_dir / "rr100_frozen_eye_vs_native_zero_gaze_sf_preference.png"
    fig.savefig(png, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    joined.to_csv(args.out_dir / "paired_unit_sf_preferences.csv", index=False)
    excluded.to_csv(args.out_dir / "excluded_units.csv", index=False)
    distribution_rows = []
    for source, values in [
        ("earlier_frozen_eye_log_gaussian_fit", frozen_values),
        ("current_native_zero_gaze_sampled_factor_peak", native_sampled),
        ("current_native_zero_gaze_factor_weighted_center", native_weighted),
    ]:
        distribution_rows.append(
            {
                "source": source,
                "n": len(values),
                "minimum_cpd": float(np.min(values)),
                "q25_cpd": float(np.quantile(values, 0.25)),
                "median_cpd": float(np.median(values)),
                "q75_cpd": float(np.quantile(values, 0.75)),
                "maximum_cpd": float(np.max(values)),
            }
        )
    pd.DataFrame(distribution_rows).to_csv(args.out_dir / "sf_preference_distribution_summary.csv", index=False)

    log_frozen = np.log2(frozen_values)
    log_native_sampled = np.log2(native_sampled)
    log_native_weighted = np.log2(native_weighted)
    manifest = {
        "analysis": "rr100_frozen_eye_vs_native_zero_gaze_sf_preference",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "paired_metric_contract_diagnostic_not_interchangeable_tuning_estimates",
        "n_rr100_joined": len(joined),
        "n_paired_usable": len(paired),
        "n_excluded": len(excluded),
        "frozen_source": identity(args.frozen_unit_summary),
        "native_source": identity(args.native_fit_summary),
        "frozen_contract": "dynamic phase-RMS averaged over positive TF and orientation at each SF; baseline + amplitude * Gaussian(log2 SF); 101-px center-map readout",
        "native_contract": "dynamic F1; preferred orientation; signed directions folded to |TF|; rank-one SF factor; 51-px session-native fitted-unit readout",
        "paired_log2_correlation_frozen_vs_native_sampled": float(np.corrcoef(log_frozen, log_native_sampled)[0, 1]),
        "paired_log2_correlation_frozen_vs_native_weighted": float(np.corrcoef(log_frozen, log_native_weighted)[0, 1]),
        "artifacts": {
            "figure_png": png.name,
            "figure_pdf": png.with_suffix(".pdf").name,
            "paired_units": "paired_unit_sf_preferences.csv",
            "excluded_units": "excluded_units.csv",
            "distribution_summary": "sf_preference_distribution_summary.csv",
        },
    }
    (args.out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
