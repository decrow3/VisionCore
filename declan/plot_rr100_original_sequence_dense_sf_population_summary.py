#!/usr/bin/env python3
"""Population comparison of dense native-sequence SF tuning to prior RR100 SF groups.

The primary tuning support is 1--11.3137 cpd.  The 16 cpd substitution is
retained as a displayed sampling-edge control, but it cannot define preferred
SF, tuning centroids, or group assignments.
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
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / (
    "outputs/redundancy_resolved_v1_twin/"
    "rr100_original_sequence_dense_sf_native_readout_v1"
)
DEFAULT_PREVIOUS = ROOT / "outputs/fig/ssi_figure_v2/panels/previous_sf_tuning_groups"
DEFAULT_OUT = DEFAULT_INPUT / "population_summary_robust_v1"

ROBUST_MAX_SF = float(8.0 * np.sqrt(2.0))
EDGE_SF = 16.0
GROUPS = ("low_sf", "middle_sf", "high_sf")
GROUP_LABELS = {"low_sf": "previous low", "middle_sf": "previous middle", "high_sf": "previous high"}
GROUP_COLORS = {"low_sf": "#0072B2", "middle_sf": "#559F76", "high_sf": "#D55E00"}
NEW_BANDS = ("low_1_to_2", "mid_2p8_to_5p7", "high_8_to_11p3")
NEW_BAND_LABELS = {
    "low_1_to_2": "new low\n(1--2)",
    "mid_2p8_to_5p7": "new mid\n(2.8--5.7)",
    "high_8_to_11p3": "new high\n(8--11.3)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--previous-dir", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--n-permutations", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    stat = path.resolve().stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def new_band(sf: float) -> str:
    if sf <= 2.0:
        return "low_1_to_2"
    if sf >= 8.0:
        return "high_8_to_11p3"
    return "mid_2p8_to_5p7"


def normalize_rows_robust(matrix: pd.DataFrame, robust_columns: list[float]) -> tuple[pd.DataFrame, pd.Series]:
    robust = matrix[robust_columns]
    lo = robust.min(axis=1)
    span = robust.max(axis=1) - lo
    valid = span > 1e-12
    normalized = matrix.sub(lo, axis=0).div(span.where(valid), axis=0)
    return normalized, valid


def tuning_centroid(matrix: pd.DataFrame, robust_columns: list[float]) -> pd.Series:
    robust = matrix[robust_columns]
    shifted = robust.sub(robust.min(axis=1), axis=0).clip(lower=0.0)
    denom = shifted.sum(axis=1)
    weights = shifted.div(denom.where(denom > 1e-12), axis=0)
    return np.exp2(weights @ np.log2(np.asarray(robust_columns, dtype=float))).rename("robust_sf_centroid_cpd")


def bootstrap_curve_summary(
    normalized: pd.DataFrame,
    joined: pd.DataFrame,
    sf_columns: list[float],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in GROUPS:
        ids = joined.loc[joined["sf_group"].eq(group), "rr100_index"].to_numpy(dtype=int)
        values = normalized.loc[ids, sf_columns].to_numpy(dtype=float)
        observed = np.nanmedian(values, axis=0)
        draws = np.empty((n_bootstrap, len(sf_columns)), dtype=np.float64)
        for start in range(0, n_bootstrap, 500):
            stop = min(start + 500, n_bootstrap)
            sampled = rng.integers(0, len(ids), size=(stop - start, len(ids)))
            draws[start:stop] = np.nanmedian(values[sampled], axis=1)
        lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
        for sf, med, low, high in zip(sf_columns, observed, lo, hi):
            rows.append(
                {
                    "sf_group": group,
                    "sf_group_label": GROUP_LABELS[group],
                    "target_sf_cpd": sf,
                    "is_resolution_robust": bool(sf <= ROBUST_MAX_SF + 1e-8),
                    "median_robust_range_normalized_response": med,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "n_units": len(ids),
                }
            )
    return pd.DataFrame(rows)


def permutation_effect(
    values: pd.Series,
    groups: pd.Series,
    group_a: str,
    group_b: str,
    rng: np.random.Generator,
    n_permutations: int,
    n_bootstrap: int,
) -> dict[str, float | int | str]:
    keep = groups.isin([group_a, group_b]) & values.notna()
    vals = values[keep].to_numpy(dtype=float)
    labels = groups[keep].to_numpy(dtype=str)
    a = vals[labels == group_a]
    b = vals[labels == group_b]
    observed = float(np.median(b) - np.median(a))
    pooled = vals.copy()
    n_a = len(a)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm = rng.permutation(pooled)
        null[i] = np.median(perm[n_a:]) - np.median(perm[:n_a])
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        boot[i] = np.median(bb) - np.median(aa)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    return {
        "comparison": f"{group_b}_minus_{group_a}",
        "n_a": len(a),
        "n_b": len(b),
        "median_difference_log2_cpd": observed,
        "median_ratio_cpd": float(2.0**observed),
        "bootstrap_ci_low_log2_cpd": float(ci_low),
        "bootstrap_ci_high_log2_cpd": float(ci_high),
        "permutation_p_two_sided": float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (n_permutations + 1)),
    }


def prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    paths = {
        "curves": args.input_dir / "rr100_native_dense_sf_curves_long.csv",
        "metrics": args.input_dir / "rr100_native_dense_sf_unit_metrics.csv",
        "previous_units": args.previous_dir / "previous_sf_tuning_unit_summary.csv",
        "previous_curves": args.previous_dir / "previous_sf_tuning_marginal_curves.csv",
    }
    curves = pd.read_csv(paths["curves"])
    metrics = pd.read_csv(paths["metrics"])
    previous = pd.read_csv(paths["previous_units"]).rename(columns={"unit_index": "rr100_index"})
    old_curves = pd.read_csv(paths["previous_curves"])

    current = curves[curves["response_state"].eq("all_model_valid_bins")].copy()
    matrix = current.pivot(index="rr100_index", columns="target_sf_cpd", values="mean_rate_hz").sort_index(axis=1)
    if matrix.shape != (100, 9) or matrix.isna().any().any():
        raise ValueError(f"Expected a finite 100 x 9 current curve matrix, got {matrix.shape}")
    robust_columns = [float(v) for v in matrix.columns if float(v) <= ROBUST_MAX_SF + 1e-8]
    all_columns = [float(v) for v in matrix.columns]
    normalized, valid_norm = normalize_rows_robust(matrix, robust_columns)
    centroids = tuning_centroid(matrix, robust_columns)

    keep_previous = previous[
        [
            "rr100_index",
            "unit_label",
            "sf_group",
            "sf_group_label",
            "final_sf_group",
            "dynamic_log_gaussian_marginal_sf_cpd",
            "dynamic_log_gaussian_marginal_r2",
            "dynamic_log_gaussian_marginal_fit_ok",
            "dynamic_sf_probe_one_cycle_cpd",
            "dynamic_log_gaussian_marginal_low_subcycle_amp_share",
        ]
    ].copy()
    joined = metrics.merge(keep_previous, on="rr100_index", how="inner", validate="one_to_one")
    if len(joined) != 100 or set(joined["sf_group"]) != set(GROUPS):
        raise ValueError("Historical SF-group join did not preserve all 100 RR100 units and three groups")
    joined["robust_sf_centroid_cpd"] = joined["rr100_index"].map(centroids)
    joined["robust_curve_normalizable"] = joined["rr100_index"].map(valid_norm).astype(bool)
    joined["new_robust_peak_band"] = joined["preferred_sf_cpd_resolution_robust"].map(new_band)
    joined["all_grid_peak_is_16_edge"] = np.isclose(joined["preferred_sf_cpd_all"], EDGE_SF)
    joined["shape_resolved_modulation_fraction_ge_0p05"] = joined["robust_modulation_fraction"] >= 0.05
    joined["previous_log2_preferred_sf"] = np.log2(joined["dynamic_log_gaussian_marginal_sf_cpd"])
    joined["current_robust_log2_peak_sf"] = np.log2(joined["preferred_sf_cpd_resolution_robust"])
    joined["current_robust_log2_centroid_sf"] = np.log2(joined["robust_sf_centroid_cpd"])

    normalized.index = normalized.index.astype(int)
    normalized.columns = normalized.columns.astype(float)
    return joined, matrix, normalized, old_curves, paths


def statistical_summary(
    joined: pd.DataFrame, rng: np.random.Generator, n_permutations: int, n_bootstrap: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, label in [
        ("current_robust_log2_peak_sf", "robust_discrete_peak"),
        ("current_robust_log2_centroid_sf", "robust_response_centroid"),
    ]:
        x = joined[metric]
        rho, rho_p = spearmanr(joined["previous_log2_preferred_sf"], x)
        arrays = [joined.loc[joined["sf_group"].eq(g), metric].to_numpy(dtype=float) for g in GROUPS]
        kw_h, kw_p = kruskal(*arrays)
        low, high = arrays[0], arrays[2]
        mw_u, mw_p = mannwhitneyu(low, high, alternative="two-sided")
        effect = permutation_effect(
            x,
            joined["sf_group"],
            "low_sf",
            "high_sf",
            rng,
            n_permutations,
            n_bootstrap,
        )
        rows.append(
            {
                "metric": label,
                "n_units": len(joined),
                "previous_continuous_fit_spearman_rho": rho,
                "previous_continuous_fit_spearman_p": rho_p,
                "historical_three_group_kruskal_h": kw_h,
                "historical_three_group_kruskal_p": kw_p,
                "previous_low_vs_high_mannwhitney_u": mw_u,
                "previous_low_vs_high_mannwhitney_p": mw_p,
                **effect,
            }
        )
    return pd.DataFrame(rows)


def sensitivity_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold in (0.0, 0.01, 0.05, 0.10):
        sub = joined[joined["robust_modulation_fraction"] >= threshold]
        for metric in ("current_robust_log2_peak_sf", "current_robust_log2_centroid_sf"):
            rho, p = spearmanr(sub["previous_log2_preferred_sf"], sub[metric])
            rows.append(
                {
                    "minimum_robust_modulation_fraction": threshold,
                    "metric": metric,
                    "n_units": len(sub),
                    "spearman_rho": rho,
                    "spearman_p": p,
                }
            )
    return pd.DataFrame(rows)


def categorical_summary(joined: pd.DataFrame) -> pd.DataFrame:
    low_high = joined[joined["sf_group"].isin(["low_sf", "high_sf"])]
    table = (
        pd.crosstab(low_high["sf_group"], low_high["new_robust_peak_band"])
        .reindex(index=["low_sf", "high_sf"], columns=NEW_BANDS, fill_value=0)
    )
    chi2, chi2_p, dof, _expected = chi2_contingency(table.to_numpy(dtype=int))
    low_band_table = np.asarray(
        [
            [int(table.loc["low_sf", "low_1_to_2"]), int(table.loc["low_sf", NEW_BANDS[1:]].sum())],
            [int(table.loc["high_sf", "low_1_to_2"]), int(table.loc["high_sf", NEW_BANDS[1:]].sum())],
        ]
    )
    odds_ratio, fisher_p = fisher_exact(low_band_table, alternative="two-sided")
    return pd.DataFrame(
        [
            {
                "comparison": "previous_low_vs_high_by_three_new_peak_bands",
                "test": "chi_square",
                "statistic": chi2,
                "degrees_of_freedom": dof,
                "p_two_sided": chi2_p,
                "odds_ratio": np.nan,
            },
            {
                "comparison": "previous_low_vs_high_by_new_low_band_occupancy",
                "test": "fisher_exact",
                "statistic": np.nan,
                "degrees_of_freedom": np.nan,
                "p_two_sided": fisher_p,
                "odds_ratio": odds_ratio,
            },
        ]
    )


def collapsed_statistics(
    joined: pd.DataFrame, rng: np.random.Generator, n_permutations: int, n_bootstrap: int
) -> pd.DataFrame:
    labels = pd.Series(
        np.where(joined["sf_group"].eq("high_sf"), "high_sf", "low_middle_sf"),
        index=joined.index,
    )
    rows: list[dict[str, object]] = []
    for metric, label in [
        ("current_robust_log2_peak_sf", "robust_discrete_peak"),
        ("current_robust_log2_centroid_sf", "robust_response_centroid"),
    ]:
        low_mid = joined.loc[labels.eq("low_middle_sf"), metric].to_numpy(dtype=float)
        high = joined.loc[labels.eq("high_sf"), metric].to_numpy(dtype=float)
        u, p = mannwhitneyu(low_mid, high, alternative="two-sided")
        effect = permutation_effect(
            joined[metric], labels, "low_middle_sf", "high_sf", rng, n_permutations, n_bootstrap
        )
        rows.append({"metric": label, "mannwhitney_u": u, "mannwhitney_p": p, **effect})
    return pd.DataFrame(rows)


def historical_curve_summary(old_curves: pd.DataFrame) -> pd.DataFrame:
    return (
        old_curves.groupby(["sf_group", "spatial_cpd"], sort=True)["dynamic_marginal_response_norm"]
        .agg(median="median", q25=lambda x: np.nanpercentile(x, 25), q75=lambda x: np.nanpercentile(x, 75), n_units="count")
        .reset_index()
    )


def transition_table(joined: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(joined["sf_group"], joined["new_robust_peak_band"]).reindex(index=GROUPS, columns=NEW_BANDS, fill_value=0)
    long = counts.rename_axis(index="previous_sf_group", columns="new_robust_peak_band").stack().rename("n_units").reset_index()
    long["fraction_within_previous_group"] = long["n_units"] / long["previous_sf_group"].map(joined["sf_group"].value_counts())
    return long


def plot_main(
    args: argparse.Namespace,
    joined: pd.DataFrame,
    normalized: pd.DataFrame,
    curve_summary: pd.DataFrame,
    old_summary: pd.DataFrame,
    transitions: pd.DataFrame,
    stats: pd.DataFrame,
    categorical: pd.DataFrame,
) -> Path:
    robust_sfs = sorted(curve_summary.loc[curve_summary["is_resolution_robust"], "target_sf_cpd"].unique())
    all_sfs = sorted(curve_summary["target_sf_cpd"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.9))
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.08, top=0.88, wspace=0.31, hspace=0.38)

    ax = axes[0, 0]
    for group in GROUPS:
        sub = old_summary[old_summary["sf_group"].eq(group)].sort_values("spatial_cpd")
        ax.plot(sub["spatial_cpd"], sub["median"], color=GROUP_COLORS[group], lw=2.1, label=f"{GROUP_LABELS[group]} (n={int(sub['n_units'].max())})")
        ax.fill_between(sub["spatial_cpd"].to_numpy(), sub["q25"].to_numpy(), sub["q75"].to_numpy(), color=GROUP_COLORS[group], alpha=0.13, linewidth=0)
    ax.axvspan(0.0125, 0.371334, color="#F2C14E", alpha=0.13, label="<1 cycle/window")
    ax.axvline(0.05, color="0.4", ls=":", lw=1)
    ax.axvline(0.5, color="0.4", ls="--", lw=1)
    ax.set(xscale="log", xlabel="historical probe SF (cpd)", ylabel="within-unit normalized response", title="A. Historical curves that defined the labels")
    ax.set_xticks([0.0125, 0.05, 0.2, 0.8, 3.2, 12.8], [".0125", ".05", ".2", ".8", "3.2", "12.8"])
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    ax = axes[0, 1]
    for group in GROUPS:
        sub = curve_summary[curve_summary["sf_group"].eq(group)].sort_values("target_sf_cpd")
        robust = sub[sub["is_resolution_robust"]]
        ax.plot(robust["target_sf_cpd"], robust["median_robust_range_normalized_response"], color=GROUP_COLORS[group], marker="o", lw=2.2, ms=4, label=GROUP_LABELS[group])
        ax.fill_between(robust["target_sf_cpd"].to_numpy(), robust["bootstrap_ci_low"].to_numpy(), robust["bootstrap_ci_high"].to_numpy(), color=GROUP_COLORS[group], alpha=0.16, linewidth=0)
        edge = sub[np.isclose(sub["target_sf_cpd"], EDGE_SF)]
        ax.plot([robust_sfs[-1], EDGE_SF], [robust["median_robust_range_normalized_response"].iloc[-1], edge["median_robust_range_normalized_response"].iloc[0]], color=GROUP_COLORS[group], ls=":", lw=1.2)
        ax.scatter(edge["target_sf_cpd"], edge["median_robust_range_normalized_response"], facecolor="white", edgecolor=GROUP_COLORS[group], s=36, zorder=4)
    ax.axvspan(ROBUST_MAX_SF * 1.03, 17.2, color="0.88", alpha=0.8)
    ax.axvline(ROBUST_MAX_SF, color="0.45", ls="--", lw=1)
    ax.set(xscale="log", xlabel="substituted SF (cpd)", ylabel="robust-range normalized native response", title="B. Same units, original sequence + dense SF substitution")
    ax.set_xticks(all_sfs, [f"{v:g}" for v in all_sfs])
    ax.text(15.7, 0.04, "edge\ncontrol", ha="right", va="bottom", fontsize=7, color="0.35")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 2]
    rng = np.random.default_rng(args.seed + 99)
    for pos, group in enumerate(GROUPS):
        vals = joined.loc[joined["sf_group"].eq(group), "preferred_sf_cpd_resolution_robust"].to_numpy(dtype=float)
        x = pos + rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(x, vals, s=24, alpha=0.67, color=GROUP_COLORS[group], edgecolor="white", linewidth=0.35)
        med = float(np.median(vals))
        ax.plot([pos - 0.22, pos + 0.22], [med, med], color="black", lw=2.2)
    fisher_row = categorical[categorical["test"].eq("fisher_exact")].iloc[0]
    low_fraction = float((joined.loc[joined["sf_group"].eq("low_sf"), "new_robust_peak_band"] == "low_1_to_2").mean())
    high_fraction = float((joined.loc[joined["sf_group"].eq("high_sf"), "new_robust_peak_band"] == "low_1_to_2").mean())
    ax.set(yscale="log", ylabel="new robust discrete peak (cpd)", title="C. Robust peaks under the correct construction")
    ax.set_xticks(range(3), ["previous\nlow", "previous\nmiddle", "previous\nhigh"])
    ax.set_yticks(robust_sfs, [f"{v:g}" for v in robust_sfs])
    ax.text(0.03, 0.97, f"new-low band: {low_fraction:.0%} vs {high_fraction:.0%}\nFisher p={fisher_row['p_two_sided']:.3g}", transform=ax.transAxes, va="top", fontsize=7.5)

    ax = axes[1, 0]
    count_matrix = transitions.pivot(index="previous_sf_group", columns="new_robust_peak_band", values="n_units").reindex(index=GROUPS, columns=NEW_BANDS)
    fraction_matrix = transitions.pivot(index="previous_sf_group", columns="new_robust_peak_band", values="fraction_within_previous_group").reindex(index=GROUPS, columns=NEW_BANDS)
    im = ax.imshow(fraction_matrix.to_numpy(dtype=float), vmin=0, vmax=0.7, cmap="Blues", aspect="auto")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(count_matrix.iloc[i, j])}\n({fraction_matrix.iloc[i, j]:.0%})", ha="center", va="center", color="white" if fraction_matrix.iloc[i, j] > 0.43 else "black", fontsize=8)
    ax.set_xticks(range(3), [NEW_BAND_LABELS[g] for g in NEW_BANDS])
    ax.set_yticks(range(3), [GROUP_LABELS[g] for g in GROUPS])
    ax.set(xlabel="new peak band (cpd)", ylabel="historical group", title="D. Label transfer is diffuse, not one-to-one")

    ax = axes[1, 1]
    for group in GROUPS:
        sub = joined[joined["sf_group"].eq(group)]
        ax.scatter(sub["dynamic_log_gaussian_marginal_sf_cpd"], sub["robust_sf_centroid_cpd"], s=29, alpha=0.72, color=GROUP_COLORS[group], edgecolor="white", linewidth=0.35, label=GROUP_LABELS[group])
    centroid_stat = stats[stats["metric"].eq("robust_response_centroid")].iloc[0]
    ax.set(xscale="log", yscale="log", xlabel="historical fitted preference (cpd)", ylabel="new robust SF centroid (cpd)", title="E. Continuous preference transfer is weak")
    ax.axvspan(0.0125, 0.371334, color="#F2C14E", alpha=0.13)
    ax.text(0.03, 0.97, f"Spearman ρ={centroid_stat['previous_continuous_fit_spearman_rho']:.2f}\np={centroid_stat['previous_continuous_fit_spearman_p']:.3g}", transform=ax.transAxes, va="top", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    ax = axes[1, 2]
    for pos, group in enumerate(GROUPS):
        sub = joined[joined["sf_group"].eq(group)]
        vals = sub["edge_16_minus_11p3_hz"].to_numpy(dtype=float)
        x = pos + rng.uniform(-0.18, 0.18, len(vals))
        edge_winner = sub["all_grid_peak_is_16_edge"].to_numpy(dtype=bool)
        ax.scatter(x[~edge_winner], vals[~edge_winner], s=22, facecolor="none", edgecolor=GROUP_COLORS[group], alpha=0.55, linewidth=0.8)
        ax.scatter(x[edge_winner], vals[edge_winner], s=28, color=GROUP_COLORS[group], alpha=0.8, edgecolor="white", linewidth=0.35)
    ax.axhline(0, color="0.4", lw=1)
    ax.set_xticks(range(3), ["previous\nlow", "previous\nmiddle", "previous\nhigh"])
    ax.set(ylabel="16 minus 11.3 cpd response (Hz)", title="F. Why 16 cpd is only an edge diagnostic")
    n_edge = int(joined["all_grid_peak_is_16_edge"].sum())
    ax.text(0.03, 0.97, f"filled: 16 cpd wins all-grid peak\n{n_edge}/100 units", transform=ax.transAxes, va="top", fontsize=8)

    for ax in axes.ravel():
        ax.grid(True, color="0.92", linewidth=0.65, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("RR100 SF tuning: historical fit labels versus native original-sequence substitution", x=0.02, y=0.975, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.925, "Primary inference uses 1--11.3 cpd. The 16 cpd point is shown but excluded from peaks, centroids, and new bands.", fontsize=9, color="0.3")
    png = args.out_dir / "rr100_dense_sf_population_vs_previous_groups.png"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return png


def plot_collapsed(args: argparse.Namespace, joined: pd.DataFrame, curve_summary: pd.DataFrame) -> Path:
    collapse = {"low_sf": "previous low+middle", "middle_sf": "previous low+middle", "high_sf": "previous high"}
    colors = {"previous low+middle": "#0072B2", "previous high": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.9))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.78, wspace=0.34)
    # Recompute the displayed collapsed curve directly from unit-level normalized responses.
    curves = pd.read_csv(args.input_dir / "rr100_native_dense_sf_curves_long.csv")
    matrix = curves[curves["response_state"].eq("all_model_valid_bins")].pivot(index="rr100_index", columns="target_sf_cpd", values="mean_rate_hz").sort_index(axis=1)
    robust_columns = [float(v) for v in matrix.columns if float(v) <= ROBUST_MAX_SF + 1e-8]
    normalized, _ = normalize_rows_robust(matrix, robust_columns)
    all_sfs = [float(v) for v in matrix.columns]
    joined = joined.copy()
    joined["collapsed_group"] = joined["sf_group"].map(collapse)
    rng = np.random.default_rng(args.seed + 301)
    for group in ("previous low+middle", "previous high"):
        ids = joined.loc[joined["collapsed_group"].eq(group), "rr100_index"].to_numpy(dtype=int)
        vals = normalized.loc[ids, all_sfs].to_numpy(dtype=float)
        med = np.nanmedian(vals, axis=0)
        lo = np.empty(len(all_sfs)); hi = np.empty(len(all_sfs))
        draws = np.empty((args.n_bootstrap, len(all_sfs)))
        for start in range(0, args.n_bootstrap, 500):
            stop = min(start + 500, args.n_bootstrap)
            sampled = rng.integers(0, len(ids), size=(stop - start, len(ids)))
            draws[start:stop] = np.nanmedian(vals[sampled], axis=1)
        lo[:], hi[:] = np.nanpercentile(draws, [2.5, 97.5], axis=0)
        axes[0].plot(all_sfs[:-1], med[:-1], color=colors[group], marker="o", lw=2.2, label=f"{group} (n={len(ids)})")
        axes[0].fill_between(all_sfs[:-1], lo[:-1], hi[:-1], color=colors[group], alpha=0.16, linewidth=0)
        axes[0].plot(all_sfs[-2:], med[-2:], color=colors[group], ls=":", lw=1.2)
        axes[0].scatter([all_sfs[-1]], [med[-1]], facecolor="white", edgecolor=colors[group], s=38, zorder=4)
        peak = joined.loc[joined["collapsed_group"].eq(group), "preferred_sf_cpd_resolution_robust"].to_numpy(dtype=float)
        axes[1].hist(np.log2(peak), bins=np.arange(-0.25, 3.76, 0.5), alpha=0.48, color=colors[group], label=group)
    axes[0].axvspan(ROBUST_MAX_SF * 1.03, 17.2, color="0.88", alpha=0.8)
    axes[0].set(xscale="log", xlabel="substituted SF (cpd)", ylabel="robust-range normalized response", title="A. Final historical collapse")
    axes[0].set_xticks(all_sfs, [f"{v:g}" for v in all_sfs])
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set(xlabel="new robust peak (cpd)", ylabel="units", title="B. Peak distributions remain highly overlapping")
    axes[1].set_xticks(np.arange(0, 3.51, 0.5), [f"{2**v:g}" for v in np.arange(0, 3.51, 0.5)])
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.grid(True, color="0.92", lw=0.65)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Previously used low+middle versus high collapse", x=0.02, y=0.96, ha="left", fontsize=12, fontweight="bold")
    path = args.out_dir / "rr100_dense_sf_previous_final_collapse.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def write_readme(args: argparse.Namespace, joined: pd.DataFrame, stats: pd.DataFrame, categorical: pd.DataFrame) -> None:
    centroid = stats[stats["metric"].eq("robust_response_centroid")].iloc[0]
    fisher = categorical[categorical["test"].eq("fisher_exact")].iloc[0]
    low_band = joined["new_robust_peak_band"].eq("low_1_to_2")
    low_fraction = float(low_band[joined["sf_group"].eq("low_sf")].mean())
    high_fraction = float(low_band[joined["sf_group"].eq("high_sf")].mean())
    low_high_band = float(
        joined.loc[joined["sf_group"].eq("low_sf"), "new_robust_peak_band"].eq("high_8_to_11p3").mean()
    )
    high_high_band = float(
        joined.loc[joined["sf_group"].eq("high_sf"), "new_robust_peak_band"].eq("high_8_to_11p3").mean()
    )
    lines = [
        "# RR100 dense-SF population comparison",
        "",
        "The native fitted-unit readouts were evaluated on the exact original grating sequence after substituting a dense SF grid. Primary tuning inference is restricted to 1--11.3137 cpd; 16 cpd is displayed only as a sampling-edge diagnostic.",
        "",
        "## Main result",
        "",
        "The historical low/high labels preserve a modest relative shift, but they do not transfer as clean low- versus high-SF populations under the corrected construction.",
        "",
        f"- Historical low versus high robust-centroid median ratio: {centroid['median_ratio_cpd']:.3f}x (permutation p={centroid['permutation_p_two_sided']:.4f}).",
        f"- Continuous historical fit versus corrected centroid: Spearman rho={centroid['previous_continuous_fit_spearman_rho']:.3f}, p={centroid['previous_continuous_fit_spearman_p']:.4f}.",
        f"- Corrected low-band occupancy: historical low={low_fraction:.1%}, historical high={high_fraction:.1%} (Fisher p={fisher['p_two_sided']:.4f}).",
        f"- Corrected high-band occupancy is nearly identical: historical low={low_high_band:.1%}, historical high={high_high_band:.1%}.",
        f"- If 16 cpd were allowed to define preference, it would win for {int(joined['all_grid_peak_is_16_edge'].sum())}/100 units.",
        "",
        "The categorical difference is therefore mostly a shift from the corrected low band into the corrected middle band, not recovery of a distinct high-SF population.",
        "",
        "## Historical-label caveat",
        "",
        "The previous labels came from dynamic marginal log-Gaussian fits on a 101-pixel probe. One cycle across that window is about 0.371 cpd, while the historical low threshold was 0.05 cpd. Thus much of the old low group was defined in the sub-cycle regime and partly reflected global ramp/flicker structure rather than conventional multi-cycle grating tuning.",
    ]
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    joined, matrix, normalized, old_curves, paths = prepare(args)
    sf_columns = [float(v) for v in matrix.columns]
    curve_summary = bootstrap_curve_summary(normalized, joined, sf_columns, rng, args.n_bootstrap)
    old_summary = historical_curve_summary(old_curves)
    transitions = transition_table(joined)
    stats = statistical_summary(joined, rng, args.n_permutations, args.n_bootstrap)
    sensitivity = sensitivity_summary(joined)
    categorical = categorical_summary(joined)
    collapsed_stats = collapsed_statistics(joined, rng, args.n_permutations, args.n_bootstrap)

    main_png = plot_main(args, joined, normalized, curve_summary, old_summary, transitions, stats, categorical)
    collapse_png = plot_collapsed(args, joined, curve_summary)
    write_readme(args, joined, stats, categorical)

    joined.to_csv(args.out_dir / "rr100_dense_sf_population_unit_join.csv", index=False)
    curve_summary.to_csv(args.out_dir / "rr100_dense_sf_previous_group_curve_summary.csv", index=False)
    transitions.to_csv(args.out_dir / "rr100_dense_sf_previous_to_new_band_transition.csv", index=False)
    stats.to_csv(args.out_dir / "rr100_dense_sf_population_statistics.csv", index=False)
    categorical.to_csv(args.out_dir / "rr100_dense_sf_categorical_statistics.csv", index=False)
    collapsed_stats.to_csv(args.out_dir / "rr100_dense_sf_previous_final_collapse_statistics.csv", index=False)
    sensitivity.to_csv(args.out_dir / "rr100_dense_sf_modulation_sensitivity.csv", index=False)

    counts = joined["sf_group"].value_counts().reindex(GROUPS).to_dict()
    peak_counts = joined["preferred_sf_cpd_resolution_robust"].value_counts().sort_index().to_dict()
    edge_by_group = joined[joined["all_grid_peak_is_16_edge"]]["sf_group"].value_counts().reindex(GROUPS, fill_value=0).to_dict()
    manifest = {
        "analysis": "rr100_original_sequence_dense_sf_population_vs_previous_groups",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_inference_support_cpd": [1.0, ROBUST_MAX_SF],
        "edge_control_cpd": EDGE_SF,
        "edge_control_excluded_from": ["preferred_sf", "sf_centroid", "new_peak_band"],
        "historical_group_definition": {
            "metric": "dynamic marginal log-Gaussian fitted SF preference",
            "low_sf": "<= 0.05 cpd",
            "middle_sf": "> 0.05 and < 0.5 cpd",
            "high_sf": ">= 0.5 cpd",
            "caution": "historical probe has one cycle/window at 0.371334 cpd; much of low group is sub-cycle",
        },
        "historical_group_counts": {k: int(v) for k, v in counts.items()},
        "robust_discrete_peak_counts": {f"{float(k):g}": int(v) for k, v in peak_counts.items()},
        "n_all_grid_peak_at_16_edge": int(joined["all_grid_peak_is_16_edge"].sum()),
        "n_all_grid_peak_at_16_edge_by_previous_group": {k: int(v) for k, v in edge_by_group.items()},
        "n_shape_resolved_modulation_fraction_ge_0p05": int(joined["shape_resolved_modulation_fraction_ge_0p05"].sum()),
        "random_seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "inputs": {name: file_identity(path) for name, path in paths.items()},
        "outputs": {
            "main_figure": str(main_png.resolve()),
            "collapsed_figure": str(collapse_png.resolve()),
        },
    }
    (args.out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(stats.to_string(index=False))
    print(categorical.to_string(index=False))
    print(collapsed_stats.to_string(index=False))


if __name__ == "__main__":
    main()
