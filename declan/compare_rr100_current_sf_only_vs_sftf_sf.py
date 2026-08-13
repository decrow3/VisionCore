#!/usr/bin/env python3
"""Compare SF tuning in the current SF-only and separable SF-by-TF RR100 runs.

The SF-only response is the mean native fitted rate over the reconstructed
recorded sequence.  The SF-by-TF response is the zero-gaze dynamic-F1 rank-one
SF factor.  They are therefore compared as independently range-normalized
tuning shapes on their identical resolution-robust SF support, not as response
amplitudes.
"""

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
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SF_ONLY = ROOT / (
    "outputs/redundancy_resolved_v1_twin/"
    "rr100_original_sequence_dense_sf_native_readout_v1"
)
DEFAULT_SFTF = ROOT / (
    "outputs/redundancy_resolved_v1_twin/"
    "rr100_zero_gaze_separable_sf_tf_factorization_v1"
)
DEFAULT_OUT = ROOT / (
    "outputs/redundancy_resolved_v1_twin/"
    "rr100_current_sf_only_vs_sftf_sf_comparison_v1"
)
PRIMARY_SURFACE = "preferred_orientation_abs_tf"
SENSITIVITY_SURFACE = "orientation_marginal_abs_tf"
SURFACES = (PRIMARY_SURFACE, SENSITIVITY_SURFACE)
COLORS = {"sf_only": "#0072B2", "sf_tf": "#D55E00"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sf-only-dir", type=Path, default=DEFAULT_SF_ONLY)
    parser.add_argument("--sftf-dir", type=Path, default=DEFAULT_SFTF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    stat = path.resolve().stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def range_normalize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    lo = frame.min(axis=1)
    span = frame.max(axis=1) - lo
    valid = span > 1e-12
    normalized = frame.sub(lo, axis=0).div(span.where(valid), axis=0)
    return normalized, valid


def geometric_centroid(normalized: pd.DataFrame) -> pd.Series:
    values = normalized.clip(lower=0.0)
    denom = values.sum(axis=1)
    weights = values.div(denom.where(denom > 1e-12), axis=0)
    log_sf = np.log2(np.asarray(values.columns, dtype=float))
    return np.exp2(weights @ log_sf)


def row_correlations(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    pearson: dict[int, float] = {}
    spearman: dict[int, float] = {}
    for idx in a.index:
        av = a.loc[idx].to_numpy(dtype=float)
        bv = b.loc[idx].to_numpy(dtype=float)
        if np.isfinite(av).all() and np.isfinite(bv).all() and np.ptp(av) > 1e-12 and np.ptp(bv) > 1e-12:
            pearson[int(idx)] = float(pearsonr(av, bv).statistic)
            spearman[int(idx)] = float(spearmanr(av, bv).statistic)
        else:
            pearson[int(idx)] = np.nan
            spearman[int(idx)] = np.nan
    return pd.Series(pearson), pd.Series(spearman)


def bootstrap_ci(
    values: np.ndarray,
    statistic,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    values = np.asarray(values)
    draws = np.empty(n_bootstrap, dtype=float)
    for start in range(0, n_bootstrap, 500):
        stop = min(start + 500, n_bootstrap)
        samples = rng.integers(0, len(values), size=(stop - start, len(values)))
        draws[start:stop] = statistic(values[samples], axis=1)
    return tuple(float(x) for x in np.nanpercentile(draws, [2.5, 97.5]))


def prepare(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, dict[str, Path]]:
    paths = {
        "sf_only_curves": args.sf_only_dir / "rr100_native_dense_sf_curves_long.csv",
        "sf_only_metrics": args.sf_only_dir / "rr100_native_dense_sf_unit_metrics.csv",
        "sftf_summary": args.sftf_dir / "separable_fit_unit_summary.csv",
        "sftf_factors": args.sftf_dir / "separable_factor_points.csv",
        "sf_only_manifest": args.sf_only_dir / "manifest.json",
        "sftf_manifest": args.sftf_dir / "analysis_manifest.json",
    }
    curves = pd.read_csv(paths["sf_only_curves"])
    sf_metrics = pd.read_csv(paths["sf_only_metrics"])
    sftf_summary = pd.read_csv(paths["sftf_summary"])
    factors = pd.read_csv(paths["sftf_factors"])

    sf_curve = (
        curves.loc[curves["response_state"].eq("all_model_valid_bins")]
        .pivot(index="rr100_index", columns="target_sf_cpd", values="mean_rate_hz")
        .sort_index(axis=1)
    )
    sf_curve = sf_curve.loc[:, sf_curve.columns <= 8.0 * np.sqrt(2.0) + 1e-8]
    if sf_curve.shape != (100, 8) or sf_curve.isna().any().any():
        raise ValueError(f"Expected finite 100 x 8 robust SF-only curves; got {sf_curve.shape}")
    sf_curve.columns = sf_curve.columns.astype(float)
    sf_norm, sf_valid = range_normalize(sf_curve)

    base = sf_metrics.copy().set_index("rr100_index")
    base["sf_only_curve_normalizable"] = sf_valid
    base["sf_only_centroid_cpd_common_norm"] = geometric_centroid(sf_norm)
    base["sf_only_peak_cpd"] = base["preferred_sf_cpd_resolution_robust"]

    normalized: dict[str, pd.DataFrame] = {"sf_only": sf_norm}
    for surface in SURFACES:
        factor = factors.loc[
            factors["surface_definition"].eq(surface) & factors["axis"].eq("spatial_frequency")
        ]
        factor = factor.pivot(index="rr100_index", columns="frequency", values="normalized_factor").sort_index(axis=1)
        factor.columns = factor.columns.astype(float)
        if factor.shape != (100, 8) or not np.allclose(factor.columns, sf_curve.columns):
            raise ValueError(f"{surface}: factor support {factor.shape} does not match SF-only support")
        factor_norm, factor_valid = range_normalize(factor)
        normalized[surface] = factor_norm

        summary = sftf_summary.loc[sftf_summary["surface_definition"].eq(surface)].set_index("rr100_index")
        if len(summary) != 100:
            raise ValueError(f"{surface}: expected 100 unit summaries, found {len(summary)}")
        prefix = "sftf_prefori" if surface == PRIMARY_SURFACE else "sftf_orimarg"
        base[f"{prefix}_peak_cpd"] = summary["preferred_sf_cpd_sampled"]
        base[f"{prefix}_centroid_cpd_common_norm"] = geometric_centroid(factor_norm)
        base[f"{prefix}_responsive"] = summary["responsive_max_f1_flag"].astype(bool)
        base[f"{prefix}_maximum_f1_hz"] = summary["surface_maximum_f1_hz"]
        base[f"{prefix}_rank1_centered_r2"] = summary["rank1_centered_r2"]
        base[f"{prefix}_rank1_energy_fraction"] = summary["rank1_energy_fraction"]
        base[f"{prefix}_curve_normalizable"] = factor_valid

        pear, spear = row_correlations(sf_norm, factor_norm)
        base[f"{prefix}_curve_pearson_r"] = pear
        base[f"{prefix}_curve_spearman_rho"] = spear
        base[f"{prefix}_curve_rmse"] = np.sqrt(((sf_norm - factor_norm) ** 2).mean(axis=1))
        base[f"{prefix}_signed_peak_difference_octaves"] = np.log2(
            base[f"{prefix}_peak_cpd"] / base["sf_only_peak_cpd"]
        )
        base[f"{prefix}_absolute_peak_difference_octaves"] = base[
            f"{prefix}_signed_peak_difference_octaves"
        ].abs()
        base[f"{prefix}_signed_centroid_difference_octaves"] = np.log2(
            base[f"{prefix}_centroid_cpd_common_norm"] / base["sf_only_centroid_cpd_common_norm"]
        )

    base.index = base.index.astype(int)
    base.index.name = "rr100_index"
    return base.reset_index(), normalized, sf_curve, paths


def summarize_variant(
    unit_table: pd.DataFrame,
    surface: str,
    subset: str,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, object]:
    prefix = "sftf_prefori" if surface == PRIMARY_SURFACE else "sftf_orimarg"
    keep = np.ones(len(unit_table), dtype=bool)
    if subset == "responsive":
        keep = unit_table[f"{prefix}_responsive"].to_numpy(dtype=bool)
    elif subset == "responsive_and_sf_only_modulated":
        keep = (
            unit_table[f"{prefix}_responsive"].to_numpy(dtype=bool)
            & (unit_table["robust_modulation_fraction"].to_numpy(dtype=float) >= 0.05)
        )
    d = unit_table.loc[keep].copy()
    delta = d[f"{prefix}_signed_peak_difference_octaves"].to_numpy(dtype=float)
    pear = d[f"{prefix}_curve_pearson_r"].to_numpy(dtype=float)
    peaks_a = d["sf_only_peak_cpd"].to_numpy(dtype=float)
    peaks_b = d[f"{prefix}_peak_cpd"].to_numpy(dtype=float)
    cent_a = np.log2(d["sf_only_centroid_cpd_common_norm"].to_numpy(dtype=float))
    cent_b = np.log2(d[f"{prefix}_centroid_cpd_common_norm"].to_numpy(dtype=float))
    sf_values = np.sort(unit_table["sf_only_peak_cpd"].unique())
    hist_a = np.asarray([np.count_nonzero(np.isclose(peaks_a, sf)) for sf in sf_values], dtype=float)
    hist_b = np.asarray([np.count_nonzero(np.isclose(peaks_b, sf)) for sf in sf_values], dtype=float)
    median_pear_ci = bootstrap_ci(pear, np.nanmedian, rng, n_bootstrap)
    exact_ci = bootstrap_ci(np.isclose(delta, 0.0).astype(float), np.mean, rng, n_bootstrap)
    half_ci = bootstrap_ci((np.abs(delta) <= 0.5 + 1e-8).astype(float), np.mean, rng, n_bootstrap)
    peak_test = spearmanr(np.log2(peaks_a), np.log2(peaks_b))
    centroid_test = spearmanr(cent_a, cent_b)
    return {
        "surface_definition": surface,
        "unit_subset": subset,
        "n_units": len(d),
        "n_curve_correlations": int(np.isfinite(pear).sum()),
        "n_exact_peaks": int(np.count_nonzero(np.isclose(delta, 0.0))),
        "exact_peak_fraction": float(np.mean(np.isclose(delta, 0.0))),
        "exact_peak_fraction_ci_low": exact_ci[0],
        "exact_peak_fraction_ci_high": exact_ci[1],
        "n_within_half_octave": int(np.count_nonzero(np.abs(delta) <= 0.5 + 1e-8)),
        "within_half_octave_fraction": float(np.mean(np.abs(delta) <= 0.5 + 1e-8)),
        "within_half_octave_fraction_ci_low": half_ci[0],
        "within_half_octave_fraction_ci_high": half_ci[1],
        "n_within_one_octave": int(np.count_nonzero(np.abs(delta) <= 1.0 + 1e-8)),
        "within_one_octave_fraction": float(np.mean(np.abs(delta) <= 1.0 + 1e-8)),
        "median_signed_peak_difference_octaves": float(np.median(delta)),
        "median_absolute_peak_difference_octaves": float(np.median(np.abs(delta))),
        "peak_spearman_rho": float(peak_test.statistic),
        "peak_spearman_p": float(peak_test.pvalue),
        "centroid_spearman_rho": float(centroid_test.statistic),
        "centroid_spearman_p": float(centroid_test.pvalue),
        "median_curve_pearson_r": float(np.nanmedian(pear)),
        "median_curve_pearson_r_ci_low": median_pear_ci[0],
        "median_curve_pearson_r_ci_high": median_pear_ci[1],
        "fraction_curve_pearson_gt_0p5": float(np.nanmean(pear > 0.5)),
        "fraction_curve_pearson_gt_0p8": float(np.nanmean(pear > 0.8)),
        "median_curve_spearman_rho": float(np.nanmedian(d[f"{prefix}_curve_spearman_rho"])),
        "median_curve_rmse": float(np.nanmedian(d[f"{prefix}_curve_rmse"])),
        "peak_distribution_total_variation": float(0.5 * np.abs(hist_a / hist_a.sum() - hist_b / hist_b.sum()).sum()),
        "peak_distribution_jensen_shannon_distance": float(jensenshannon(hist_a / hist_a.sum(), hist_b / hist_b.sum(), base=2.0)),
    }


def select_examples(unit_table: pd.DataFrame) -> pd.DataFrame:
    d = unit_table.loc[unit_table["sftf_prefori_responsive"]].copy()
    strong = d.loc[d["robust_modulation_fraction"] >= 0.05].copy()
    roles: list[tuple[str, int, str, float]] = []

    def add(role: str, pool: pd.DataFrame, criterion: str, maximize: bool) -> None:
        used = {idx for _, idx, _, _ in roles}
        eligible = pool.loc[~pool["rr100_index"].isin(used) & pool[criterion].notna()]
        if eligible.empty:
            return
        row = eligible.loc[eligible[criterion].idxmax() if maximize else eligible[criterion].idxmin()]
        roles.append((role, int(row["rr100_index"]), criterion, float(row[criterion])))

    add("best whole-curve agreement", strong, "sftf_prefori_curve_pearson_r", True)
    add("largest sampled-peak disagreement", strong, "sftf_prefori_absolute_peak_difference_octaves", True)
    exact = strong.loc[np.isclose(strong["sftf_prefori_signed_peak_difference_octaves"], 0.0)]
    add("same peak, weakest curve agreement", exact, "sftf_prefori_curve_pearson_r", False)
    high_to_low = strong.loc[
        (strong["sf_only_peak_cpd"] >= 4.0) & (strong["sftf_prefori_peak_cpd"] <= 2.0)
    ]
    add("SF-only high to SFxTF low", high_to_low, "sftf_prefori_signed_peak_difference_octaves", False)

    weak = unit_table.loc[~unit_table["sftf_prefori_responsive"]].copy()
    add("near-silent SFxTF control", weak, "sftf_prefori_maximum_f1_hz", False)
    selected = pd.DataFrame(roles, columns=["selection_role", "rr100_index", "criterion", "criterion_value"])
    keep_cols = [
        "rr100_index",
        "session",
        "canonical_channel",
        "sf_only_peak_cpd",
        "sftf_prefori_peak_cpd",
        "sftf_prefori_curve_pearson_r",
        "sftf_prefori_curve_rmse",
        "sftf_prefori_maximum_f1_hz",
        "robust_modulation_fraction",
    ]
    return selected.merge(unit_table[keep_cols], on="rr100_index", how="left", validate="one_to_one")


def plot_main(unit_table: pd.DataFrame, out_path: Path, dpi: int) -> pd.DataFrame:
    d = unit_table.loc[unit_table["sftf_prefori_responsive"]].copy()
    sf_values = np.sort(unit_table["sf_only_peak_cpd"].unique())
    labels = [f"{x:g}" for x in sf_values]
    counts_a = np.asarray([np.count_nonzero(np.isclose(d["sf_only_peak_cpd"], sf)) for sf in sf_values])
    counts_b = np.asarray([np.count_nonzero(np.isclose(d["sftf_prefori_peak_cpd"], sf)) for sf in sf_values])
    transition = pd.crosstab(d["sf_only_peak_cpd"], d["sftf_prefori_peak_cpd"]).reindex(
        index=sf_values, columns=sf_values, fill_value=0
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.7), constrained_layout=True)
    ax = axes[0, 0]
    x = np.arange(len(sf_values))
    width = 0.38
    ax.bar(x - width / 2, counts_a, width, color=COLORS["sf_only"], label="SF-only: native mean rate")
    ax.bar(x + width / 2, counts_b, width, color=COLORS["sf_tf"], label="SFxTF: zero-gaze F1 factor")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_xlabel("sampled preferred SF (cpd)")
    ax.set_ylabel("responsive units")
    ax.set_title(f"A  Preferred-SF distributions (same n={len(d)} units)", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    im = ax.imshow(transition.to_numpy(), origin="lower", cmap="Blues", vmin=0)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_yticks(x, labels)
    ax.set_xlabel("SFxTF preferred SF (cpd)")
    ax.set_ylabel("SF-only preferred SF (cpd)")
    ax.set_title("B  Paired peak transitions", loc="left", fontweight="bold")
    for i in range(len(sf_values)):
        for j in range(len(sf_values)):
            value = int(transition.iloc[i, j])
            if value:
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8,
                        color="white" if value > transition.to_numpy().max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, label="unit count", shrink=0.82)

    ax = axes[1, 0]
    ax.scatter(
        np.log2(d["sf_only_centroid_cpd_common_norm"]),
        np.log2(d["sftf_prefori_centroid_cpd_common_norm"]),
        c=d["sftf_prefori_curve_pearson_r"], cmap="coolwarm", vmin=-1, vmax=1,
        s=35, alpha=0.82, edgecolor="white", linewidth=0.35,
    )
    lo, hi = np.log2(sf_values[[0, -1]])
    ax.plot([lo, hi], [lo, hi], color="0.25", lw=1, ls="--")
    ticks = np.log2(sf_values)
    ax.set_xticks(ticks, labels, rotation=35, ha="right")
    ax.set_yticks(ticks, labels)
    ax.set_xlim(lo - 0.15, hi + 0.15)
    ax.set_ylim(lo - 0.15, hi + 0.15)
    ax.set_xlabel("SF-only tuning centroid (cpd)")
    ax.set_ylabel("SFxTF tuning centroid (cpd)")
    rho = spearmanr(np.log2(d["sf_only_centroid_cpd_common_norm"]), np.log2(d["sftf_prefori_centroid_cpd_common_norm"])).statistic
    ax.text(0.03, 0.97, f"Spearman rho = {rho:.2f}", transform=ax.transAxes, va="top")
    ax.set_title("C  Continuous common-support centroids", loc="left", fontweight="bold")

    ax = axes[1, 1]
    vals = d["sftf_prefori_curve_pearson_r"].dropna().to_numpy()
    ax.hist(vals, bins=np.linspace(-1, 1, 17), color="#7A5195", edgecolor="white")
    median = float(np.median(vals))
    ax.axvline(median, color="black", lw=1.5, ls="--", label=f"median r = {median:.2f}")
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Pearson r between normalized 8-point SF curves")
    ax.set_ylabel("units")
    ax.set_title("D  Whole-curve agreement", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    fig.suptitle(
        "Current RR100 SF-only vs separable SFxTF: SF agreement\n"
        "Primary comparison excludes 9 near-silent dynamic-F1 units; both curves range-normalized independently",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return transition.rename_axis(index="sf_only_peak_cpd", columns="sftf_peak_cpd").stack().rename("n_units").reset_index()


def plot_examples(
    selected: pd.DataFrame,
    normalized: dict[str, pd.DataFrame],
    out_path: Path,
    dpi: int,
) -> None:
    n = len(selected)
    fig, axes = plt.subplots(n, 1, figsize=(9.3, 2.35 * n), sharex=True, constrained_layout=True)
    if n == 1:
        axes = [axes]
    sf_values = normalized["sf_only"].columns.to_numpy(dtype=float)
    for ax, (_, row) in zip(axes, selected.iterrows()):
        idx = int(row["rr100_index"])
        ax.plot(sf_values, normalized["sf_only"].loc[idx], "o-", lw=2, color=COLORS["sf_only"],
                label="SF-only native mean rate")
        ax.plot(sf_values, normalized[PRIMARY_SURFACE].loc[idx], "s-", lw=2, color=COLORS["sf_tf"],
                label="SFxTF zero-gaze F1 factor")
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.08, 1.08)
        ax.set_ylabel("range-normalized\nSF tuning")
        ax.set_title(
            f"{row['selection_role']} — RR100 {idx}; peaks {row['sf_only_peak_cpd']:g} vs "
            f"{row['sftf_prefori_peak_cpd']:g} cpd; curve r={row['sftf_prefori_curve_pearson_r']:.2f}; "
            f"max F1={row['sftf_prefori_maximum_f1_hz']:.2g} Hz",
            loc="left", fontsize=10,
        )
        ax.grid(axis="y", color="0.9")
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    axes[-1].set_xticks(sf_values, [f"{x:g}" for x in sf_values])
    axes[-1].set_xlabel("spatial frequency (cycles/degree)")
    fig.suptitle("Auditable unit-level examples: same SF support, different response constructions", fontsize=13)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    unit_table, normalized, sf_curve, paths = prepare(args)
    summaries = []
    for surface in SURFACES:
        for subset in ("responsive", "responsive_and_sf_only_modulated", "all"):
            summaries.append(summarize_variant(unit_table, surface, subset, rng, args.n_bootstrap))
    summary = pd.DataFrame(summaries)
    selected = select_examples(unit_table)

    main_png = args.out_dir / "rr100_current_sf_only_vs_sftf_sf_agreement.png"
    examples_png = args.out_dir / "rr100_current_sf_only_vs_sftf_selected_units.png"
    transition = plot_main(unit_table, main_png, args.dpi)
    plot_examples(selected, normalized, examples_png, args.dpi)

    unit_table.to_csv(args.out_dir / "comparison_unit_table.csv", index=False)
    summary.to_csv(args.out_dir / "comparison_summary.csv", index=False)
    selected.to_csv(args.out_dir / "selected_unit_examples.csv", index=False)
    transition.to_csv(args.out_dir / "preferred_sf_transition_counts.csv", index=False)
    curves_out = []
    for source, frame in normalized.items():
        long = frame.rename_axis(index="rr100_index", columns="sf_cpd").stack().rename("range_normalized_tuning").reset_index()
        long.insert(1, "curve_source", source)
        curves_out.append(long)
    pd.concat(curves_out, ignore_index=True).to_csv(args.out_dir / "common_support_normalized_curves.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "current RR100 SF-only versus separable SFxTF SF agreement",
        "primary_surface_definition": PRIMARY_SURFACE,
        "sensitivity_surface_definition": SENSITIVITY_SURFACE,
        "primary_unit_subset": "responsive_max_f1_flag in preferred_orientation_abs_tf (91 expected)",
        "common_sf_support_cpd": [float(x) for x in sf_curve.columns],
        "comparison_contract": {
            "sf_only": "mean native fitted rate over all model-valid bins in reconstructed recorded sequence",
            "sftf": "zero-gaze dynamic-F1 rank-one SF factor at preferred orientation, |TF| folded",
            "normalization": "each unit and response construction independently min-max normalized over common SF support",
            "amplitudes_comparable": False,
        },
        "inputs": {key: file_identity(path) for key, path in paths.items()},
        "outputs": [
            str(main_png.resolve()), str(examples_png.resolve()),
            str((args.out_dir / "comparison_unit_table.csv").resolve()),
            str((args.out_dir / "comparison_summary.csv").resolve()),
            str((args.out_dir / "selected_unit_examples.csv").resolve()),
            str((args.out_dir / "preferred_sf_transition_counts.csv").resolve()),
            str((args.out_dir / "common_support_normalized_curves.csv").resolve()),
        ],
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
    }
    with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    primary = summary.loc[
        summary["surface_definition"].eq(PRIMARY_SURFACE) & summary["unit_subset"].eq("responsive")
    ].iloc[0]
    print(summary.to_string(index=False))
    print(
        f"\nPrimary: n={int(primary['n_units'])}, exact={primary['exact_peak_fraction']:.3f}, "
        f"within 0.5 octave={primary['within_half_octave_fraction']:.3f}, "
        f"median curve r={primary['median_curve_pearson_r']:.3f}, "
        f"centroid rho={primary['centroid_spearman_rho']:.3f}"
    )
    print(f"Wrote {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
