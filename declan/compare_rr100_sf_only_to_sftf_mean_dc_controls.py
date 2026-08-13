#!/usr/bin/env python3
"""Test whether SFxTF mean-rate/DC controls recover current SF-only tuning."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
SF_ONLY_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_original_sequence_dense_sf_native_readout_v1"
SFTF_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_factorization_v1"
OUT_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_sf_only_vs_sftf_mean_dc_controls_v1"
F1_SURFACE = "preferred_orientation_abs_tf"
COMPARATORS = ("dynamic_f1", "dynamic_mean_rate", "static_dc")
LABELS = {
    "sf_only": "SF-only native sequence mean rate",
    "dynamic_f1": "SFxTF dynamic F1 factor",
    "dynamic_mean_rate": "SFxTF dynamic mean rate",
    "static_dc": "SFxTF static phase-averaged DC",
}
COLORS = {
    "sf_only": "#0072B2",
    "dynamic_f1": "#D55E00",
    "dynamic_mean_rate": "#009E73",
    "static_dc": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sf-only-dir", type=Path, default=SF_ONLY_DIR)
    parser.add_argument("--sftf-dir", type=Path, default=SFTF_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def normalize_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    lo = frame.min(axis=1)
    span = frame.max(axis=1) - lo
    valid = span > 1e-12
    return frame.sub(lo, axis=0).div(span.where(valid), axis=0), valid


def centroid(frame: pd.DataFrame) -> pd.Series:
    positive = frame.clip(lower=0.0)
    weights = positive.div(positive.sum(axis=1).replace(0.0, np.nan), axis=0)
    return np.exp2(weights @ np.log2(np.asarray(frame.columns, dtype=float)))


def correlations(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.Series:
    result: dict[int, float] = {}
    for idx in reference.index:
        x = reference.loc[idx].to_numpy(dtype=float)
        y = candidate.loc[idx].to_numpy(dtype=float)
        result[int(idx)] = (
            float(pearsonr(x, y).statistic)
            if np.isfinite(x).all() and np.isfinite(y).all() and np.ptp(x) > 1e-12 and np.ptp(y) > 1e-12
            else np.nan
        )
    return pd.Series(result)


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    samples = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    return tuple(float(x) for x in np.nanpercentile(np.nanmedian(values[samples], axis=1), [2.5, 97.5]))


def prepare(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Path]]:
    paths = {
        "sf_only_curves": args.sf_only_dir / "rr100_native_dense_sf_curves_long.csv",
        "sf_only_metrics": args.sf_only_dir / "rr100_native_dense_sf_unit_metrics.csv",
        "sftf_fit_summary": args.sftf_dir / "separable_fit_unit_summary.csv",
        "sftf_factors": args.sftf_dir / "separable_factor_points.csv",
        "preferred_orientation": args.sftf_dir / "preferred_orientation_by_unit.csv",
        "dynamic_mean_control": args.sftf_dir / "dynamic_mean_rate_control.csv",
        "static_dc_control": args.sftf_dir / "static_phase_averaged_dc_control.csv",
        "sf_only_manifest": args.sf_only_dir / "manifest.json",
        "sftf_manifest": args.sftf_dir / "analysis_manifest.json",
    }
    sf_long = pd.read_csv(paths["sf_only_curves"])
    sf_metrics = pd.read_csv(paths["sf_only_metrics"]).set_index("rr100_index")
    fit = pd.read_csv(paths["sftf_fit_summary"])
    factors = pd.read_csv(paths["sftf_factors"])
    pref = pd.read_csv(paths["preferred_orientation"])[["rr100_index", "session", "preferred_orientation_deg"]]
    dynamic = pd.read_csv(paths["dynamic_mean_control"])
    static = pd.read_csv(paths["static_dc_control"])

    sf_only = (
        sf_long.loc[sf_long["response_state"].eq("all_model_valid_bins")]
        .pivot(index="rr100_index", columns="target_sf_cpd", values="mean_rate_hz")
        .sort_index(axis=1)
    )
    sf_only = sf_only.loc[:, sf_only.columns <= 8.0 * np.sqrt(2.0) + 1e-8]

    f1 = factors.loc[
        factors["surface_definition"].eq(F1_SURFACE) & factors["axis"].eq("spatial_frequency")
    ].pivot(index="rr100_index", columns="frequency", values="normalized_factor").sort_index(axis=1)

    dynamic = dynamic.merge(pref, on=["rr100_index", "session"], how="inner", validate="many_to_one")
    dynamic = dynamic.loc[np.isclose(dynamic["orientation_deg"], dynamic["preferred_orientation_deg"])]
    dynamic_mean = dynamic.groupby(["rr100_index", "spatial_cpd"], as_index=False)["mean_rate_above_blank_hz"].mean()
    dynamic_mean = dynamic_mean.pivot(index="rr100_index", columns="spatial_cpd", values="mean_rate_above_blank_hz")

    static = static.merge(pref, on=["rr100_index", "session"], how="inner", validate="many_to_one")
    static = static.loc[np.isclose(static["orientation_deg"], static["preferred_orientation_deg"])]
    static_dc = static.pivot(index="rr100_index", columns="spatial_cpd", values="static_mean_rate_above_blank_hz")

    raw = {"sf_only": sf_only, "dynamic_f1": f1, "dynamic_mean_rate": dynamic_mean, "static_dc": static_dc}
    support = sf_only.columns.to_numpy(dtype=float)
    for key, frame in raw.items():
        frame.index = frame.index.astype(int)
        frame.columns = frame.columns.astype(float)
        frame.sort_index(inplace=True)
        frame.sort_index(axis=1, inplace=True)
        if frame.shape != (100, 8) or frame.isna().any().any() or not np.allclose(frame.columns, support):
            raise ValueError(f"{key}: expected finite 100 x 8 common-support matrix, got {frame.shape}")

    normalized: dict[str, pd.DataFrame] = {}
    valid: dict[str, pd.Series] = {}
    for key, frame in raw.items():
        normalized[key], valid[key] = normalize_rows(frame)

    unit = sf_metrics[["session", "canonical_channel", "robust_modulation_fraction"]].copy()
    response = fit.loc[fit["surface_definition"].eq(F1_SURFACE)].set_index("rr100_index")
    unit["f1_responsive"] = response["responsive_max_f1_flag"].astype(bool)
    unit["f1_maximum_hz"] = response["surface_maximum_f1_hz"]
    for key, frame in raw.items():
        unit[f"{key}_normalizable"] = valid[key]
        unit[f"{key}_peak_cpd"] = frame.idxmax(axis=1).astype(float)
        unit[f"{key}_centroid_cpd"] = centroid(normalized[key])
        unit[f"{key}_range"] = frame.max(axis=1) - frame.min(axis=1)
    for key in COMPARATORS:
        unit[f"{key}_curve_pearson_r"] = correlations(normalized["sf_only"], normalized[key])
        unit[f"{key}_peak_difference_octaves"] = np.log2(unit[f"{key}_peak_cpd"] / unit["sf_only_peak_cpd"])
        unit[f"{key}_centroid_difference_octaves"] = np.log2(
            unit[f"{key}_centroid_cpd"] / unit["sf_only_centroid_cpd"]
        )
    unit["dynamic_mean_rescue_over_f1_r"] = unit["dynamic_mean_rate_curve_pearson_r"] - unit["dynamic_f1_curve_pearson_r"]
    unit["static_dc_rescue_over_f1_r"] = unit["static_dc_curve_pearson_r"] - unit["dynamic_f1_curve_pearson_r"]
    unit.index.name = "rr100_index"
    return normalized, unit.reset_index(), paths


def summarize(unit: pd.DataFrame, rng: np.random.Generator, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset in ("f1_responsive", "all"):
        d = unit.loc[unit["f1_responsive"]] if subset == "f1_responsive" else unit
        for key in COMPARATORS:
            peak_a = np.log2(d["sf_only_peak_cpd"].to_numpy(dtype=float))
            peak_b = np.log2(d[f"{key}_peak_cpd"].to_numpy(dtype=float))
            cent_a = np.log2(d["sf_only_centroid_cpd"].to_numpy(dtype=float))
            cent_b = np.log2(d[f"{key}_centroid_cpd"].to_numpy(dtype=float))
            delta = d[f"{key}_peak_difference_octaves"].to_numpy(dtype=float)
            curve_r = d[f"{key}_curve_pearson_r"].to_numpy(dtype=float)
            peak_test = spearmanr(peak_a, peak_b)
            centroid_test = spearmanr(cent_a, cent_b)
            ci = bootstrap_median_ci(curve_r, rng, n_bootstrap)
            sf_values = np.sort(unit["sf_only_peak_cpd"].unique())
            hist_a = np.asarray([np.count_nonzero(np.isclose(d["sf_only_peak_cpd"], sf)) for sf in sf_values], dtype=float)
            hist_b = np.asarray([np.count_nonzero(np.isclose(d[f"{key}_peak_cpd"], sf)) for sf in sf_values], dtype=float)
            rows.append({
                "unit_subset": subset,
                "comparator": key,
                "n_units": len(d),
                "n_exact_peaks": int(np.count_nonzero(np.isclose(delta, 0.0))),
                "exact_peak_fraction": float(np.mean(np.isclose(delta, 0.0))),
                "n_within_half_octave": int(np.count_nonzero(np.abs(delta) <= 0.5 + 1e-8)),
                "within_half_octave_fraction": float(np.mean(np.abs(delta) <= 0.5 + 1e-8)),
                "n_within_one_octave": int(np.count_nonzero(np.abs(delta) <= 1.0 + 1e-8)),
                "within_one_octave_fraction": float(np.mean(np.abs(delta) <= 1.0 + 1e-8)),
                "median_signed_peak_difference_octaves": float(np.median(delta)),
                "peak_spearman_rho": float(peak_test.statistic),
                "peak_spearman_p": float(peak_test.pvalue),
                "centroid_spearman_rho": float(centroid_test.statistic),
                "centroid_spearman_p": float(centroid_test.pvalue),
                "median_curve_pearson_r": float(np.nanmedian(curve_r)),
                "median_curve_pearson_r_ci_low": ci[0],
                "median_curve_pearson_r_ci_high": ci[1],
                "fraction_curve_r_gt_0p5": float(np.mean(curve_r > 0.5)),
                "fraction_curve_r_gt_0p8": float(np.mean(curve_r > 0.8)),
                "peak_distribution_total_variation": float(
                    0.5 * np.abs(hist_a / hist_a.sum() - hist_b / hist_b.sum()).sum()
                ),
            })
    return pd.DataFrame(rows)


def paired_rescue_summary(unit: pd.DataFrame, rng: np.random.Generator, n_bootstrap: int) -> pd.DataFrame:
    d = unit.loc[unit["f1_responsive"]].copy()
    rows: list[dict[str, object]] = []
    f1_peak_error = d["dynamic_f1_peak_difference_octaves"].abs().to_numpy(dtype=float)
    f1_centroid_error = d["dynamic_f1_centroid_difference_octaves"].abs().to_numpy(dtype=float)
    for key in ("dynamic_mean_rate", "static_dc"):
        delta_r = (d[f"{key}_curve_pearson_r"] - d["dynamic_f1_curve_pearson_r"]).to_numpy(dtype=float)
        ci = bootstrap_median_ci(delta_r, rng, n_bootstrap)
        peak_error = d[f"{key}_peak_difference_octaves"].abs().to_numpy(dtype=float)
        centroid_error = d[f"{key}_centroid_difference_octaves"].abs().to_numpy(dtype=float)
        rows.append({
            "replacement_for_dynamic_f1": key,
            "n_units": len(d),
            "median_paired_curve_r_change": float(np.median(delta_r)),
            "median_paired_curve_r_change_ci_low": ci[0],
            "median_paired_curve_r_change_ci_high": ci[1],
            "mean_paired_curve_r_change": float(np.mean(delta_r)),
            "n_curve_r_improved": int(np.count_nonzero(delta_r > 0)),
            "fraction_curve_r_improved": float(np.mean(delta_r > 0)),
            "paired_wilcoxon_p": float(wilcoxon(delta_r).pvalue),
            "dynamic_f1_median_absolute_peak_error_octaves": float(np.median(f1_peak_error)),
            "replacement_median_absolute_peak_error_octaves": float(np.median(peak_error)),
            "n_peak_error_improved": int(np.count_nonzero(peak_error < f1_peak_error)),
            "n_peak_error_worsened": int(np.count_nonzero(peak_error > f1_peak_error)),
            "dynamic_f1_median_absolute_centroid_error_octaves": float(np.median(f1_centroid_error)),
            "replacement_median_absolute_centroid_error_octaves": float(np.median(centroid_error)),
        })
    return pd.DataFrame(rows)


def select_examples(unit: pd.DataFrame) -> pd.DataFrame:
    responsive = unit.loc[unit["f1_responsive"]].copy()
    roles: list[tuple[str, int, str, float]] = []

    def add(role: str, criterion: str, maximize: bool) -> None:
        used = {idx for _, idx, _, _ in roles}
        pool = responsive.loc[~responsive["rr100_index"].isin(used) & responsive[criterion].notna()]
        row = pool.loc[pool[criterion].idxmax() if maximize else pool[criterion].idxmin()]
        roles.append((role, int(row["rr100_index"]), criterion, float(row[criterion])))

    add("largest dynamic-mean rescue", "dynamic_mean_rescue_over_f1_r", True)
    add("largest static-DC rescue", "static_dc_rescue_over_f1_r", True)
    add("F1 already matches", "dynamic_f1_curve_pearson_r", True)
    responsive["best_control_curve_r"] = responsive[["dynamic_mean_rate_curve_pearson_r", "static_dc_curve_pearson_r"]].max(axis=1)
    add("no construction matches", "best_control_curve_r", False)
    selected = pd.DataFrame(roles, columns=["selection_role", "rr100_index", "criterion", "criterion_value"])
    columns = [
        "rr100_index", "session", "canonical_channel", "sf_only_peak_cpd", "dynamic_f1_peak_cpd",
        "dynamic_mean_rate_peak_cpd", "static_dc_peak_cpd", "dynamic_f1_curve_pearson_r",
        "dynamic_mean_rate_curve_pearson_r", "static_dc_curve_pearson_r",
        "dynamic_mean_rescue_over_f1_r", "static_dc_rescue_over_f1_r", "f1_maximum_hz",
    ]
    return selected.merge(responsive[columns], on="rr100_index", how="left", validate="one_to_one")


def plot_summary(unit: pd.DataFrame, summary: pd.DataFrame, out: Path, dpi: int) -> pd.DataFrame:
    d = unit.loc[unit["f1_responsive"]].copy()
    sf_values = np.sort(d["sf_only_peak_cpd"].unique())
    x = np.arange(len(sf_values))
    width = 0.19
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.7), constrained_layout=True)

    counts_rows = []
    for offset, key in zip((-1.5, -0.5, 0.5, 1.5), ("sf_only",) + COMPARATORS):
        counts = [int(np.count_nonzero(np.isclose(d[f"{key}_peak_cpd"], sf))) for sf in sf_values]
        axes[0, 0].bar(x + offset * width, counts, width, color=COLORS[key], label=LABELS[key])
        counts_rows.extend({"curve_source": key, "sf_cpd": sf, "n_units": n} for sf, n in zip(sf_values, counts))
    axes[0, 0].set_xticks(x, [f"{v:g}" for v in sf_values], rotation=35, ha="right")
    axes[0, 0].set_xlabel("sampled preferred SF (cpd)")
    axes[0, 0].set_ylabel("units")
    axes[0, 0].set_title("A  Preferred-SF distributions", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8)

    values = [d[f"{key}_curve_pearson_r"].to_numpy(dtype=float) for key in COMPARATORS]
    box = axes[0, 1].boxplot(values, tick_labels=["dynamic F1", "dynamic\nmean rate", "static DC"], patch_artist=True,
                             showfliers=False, medianprops={"color": "black", "linewidth": 1.5})
    for patch, key in zip(box["boxes"], COMPARATORS):
        patch.set_facecolor(COLORS[key]); patch.set_alpha(0.75)
    rng = np.random.default_rng(20260811)
    for j, (vals, key) in enumerate(zip(values, COMPARATORS), start=1):
        axes[0, 1].scatter(j + rng.uniform(-0.11, 0.11, len(vals)), vals, s=9, alpha=0.28, color=COLORS[key])
    axes[0, 1].axhline(0, color="0.55", lw=0.8)
    axes[0, 1].set_ylim(-1.05, 1.05)
    axes[0, 1].set_ylabel("within-unit SF-curve Pearson r\nversus SF-only")
    axes[0, 1].set_title("B  Does changing response statistic rescue curve shape?", loc="left", fontweight="bold")

    for ax, key, panel in [
        (axes[1, 0], "dynamic_mean_rate", "C  Dynamic mean-rate centroids"),
        (axes[1, 1], "static_dc", "D  Static-DC centroids"),
    ]:
        xx = np.log2(d["sf_only_centroid_cpd"])
        yy = np.log2(d[f"{key}_centroid_cpd"])
        ax.scatter(xx, yy, s=35, alpha=0.75, color=COLORS[key], edgecolor="white", linewidth=0.35)
        lo, hi = np.log2(sf_values[[0, -1]])
        ax.plot([lo, hi], [lo, hi], "--", color="0.3", lw=1)
        ticks = np.log2(sf_values)
        labels = [f"{v:g}" for v in sf_values]
        ax.set_xticks(ticks, labels, rotation=35, ha="right")
        ax.set_yticks(ticks, labels)
        ax.set_xlim(lo - 0.15, hi + 0.15); ax.set_ylim(lo - 0.15, hi + 0.15)
        ax.set_xlabel("SF-only centroid (cpd)")
        ax.set_ylabel(f"{LABELS[key]} centroid (cpd)")
        rho = spearmanr(xx, yy).statistic
        row = summary.loc[(summary["unit_subset"].eq("f1_responsive")) & summary["comparator"].eq(key)].iloc[0]
        ax.text(0.03, 0.97, f"centroid rho={rho:.2f}\nmedian curve r={row['median_curve_pearson_r']:.2f}",
                transform=ax.transAxes, va="top")
        ax.set_title(panel, loc="left", fontweight="bold")
    fig.suptitle(
        "Can SFxTF mean-rate or DC controls recover native-sequence SF tuning?\n"
        "Same preferred orientation, common 1–11.3 cpd support, same 91 dynamic-F1-responsive units",
        fontsize=14,
    )
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(counts_rows)


def plot_examples(selected: pd.DataFrame, curves: dict[str, pd.DataFrame], out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 2.65 * len(selected)), sharex=True, constrained_layout=True)
    sf_values = curves["sf_only"].columns.to_numpy(dtype=float)
    for ax, (_, row) in zip(axes, selected.iterrows()):
        idx = int(row["rr100_index"])
        for key in ("sf_only",) + COMPARATORS:
            ax.plot(sf_values, curves[key].loc[idx], marker="o", ms=4.5, lw=1.8, color=COLORS[key], label=LABELS[key])
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.08, 1.08)
        ax.set_ylabel("range-normalized\nSF tuning")
        ax.grid(axis="y", color="0.9")
        ax.set_title(
            f"{row['selection_role']} — RR100 {idx}; curve r: F1={row['dynamic_f1_curve_pearson_r']:.2f}, "
            f"dynamic mean={row['dynamic_mean_rate_curve_pearson_r']:.2f}, DC={row['static_dc_curve_pearson_r']:.2f}",
            loc="left", fontsize=10,
        )
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[-1].set_xticks(sf_values, [f"{v:g}" for v in sf_values])
    axes[-1].set_xlabel("spatial frequency (cycles/degree)")
    fig.suptitle("Algorithmically selected unit-level rescue and failure cases", fontsize=13)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    curves, unit, paths = prepare(args)
    summary = summarize(unit, rng, args.n_bootstrap)
    rescue = paired_rescue_summary(unit, rng, args.n_bootstrap)
    selected = select_examples(unit)
    summary_png = args.out_dir / "sf_only_vs_sftf_mean_dc_control_summary.png"
    examples_png = args.out_dir / "sf_only_vs_sftf_mean_dc_selected_units.png"
    counts = plot_summary(unit, summary, summary_png, args.dpi)
    plot_examples(selected, curves, examples_png, args.dpi)

    unit.to_csv(args.out_dir / "control_comparison_unit_table.csv", index=False)
    summary.to_csv(args.out_dir / "control_comparison_summary.csv", index=False)
    rescue.to_csv(args.out_dir / "paired_response_statistic_rescue_summary.csv", index=False)
    selected.to_csv(args.out_dir / "selected_unit_examples.csv", index=False)
    counts.to_csv(args.out_dir / "preferred_sf_distribution_counts.csv", index=False)
    pd.concat(
        [frame.rename_axis(index="rr100_index", columns="sf_cpd").stack().rename("range_normalized_tuning").reset_index().assign(curve_source=key)
         for key, frame in curves.items()], ignore_index=True
    ).to_csv(args.out_dir / "normalized_control_curves.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "SF-only versus SFxTF dynamic-F1, dynamic-mean-rate, and static-DC diagnostic",
        "status": "response-statistic diagnostic checkpoint",
        "primary_unit_subset": "same 91 preferred-orientation dynamic-F1-responsive units",
        "sf_support_cpd": [float(v) for v in curves["sf_only"].columns],
        "aggregation": {
            "dynamic_mean_rate": "preferred F1 orientation; unweighted mean across signed TF conditions after phase collapse",
            "static_dc": "preferred F1 orientation; phase-averaged static mean rate above blank",
            "normalization": "independent per-unit min-max normalization over common SF support",
        },
        "inputs": {key: file_identity(path) for key, path in paths.items()},
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
    }
    with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(summary.to_string(index=False))
    print("\nPaired response-statistic rescue:\n" + rescue.to_string(index=False))
    print("\nSelected examples:\n" + selected.to_string(index=False))
    print(f"\nWrote {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
