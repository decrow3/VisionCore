#!/usr/bin/env python3
"""Primary joint SF-by-TF tuning from dynamic F0 for the current RR100 grid.

The observed response is the phase-averaged, direction-folded mean rate above
blank at the F0-preferred orientation. Signed surfaces are preserved. A
nonnegative rank-one fit is applied only to the excitatory component max(F0,0).
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
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_native_production_v1"
SF_ONLY_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_original_sequence_dense_sf_native_readout_v1"
OUT_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_f0_factorization_v1"
RESPONSE_THRESHOLD_HZ = 0.5
TF_SUPPORT = np.power(2.0, np.arange(-1.0, 5.0 + 0.5, 0.5)).astype(float)
SF_COLOR = "#0072B2"
F0_COLOR = "#009E73"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--sf-only-dir", type=Path, default=SF_ONLY_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--responsive-max-positive-f0-hz", type=float, default=RESPONSE_THRESHOLD_HZ)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def complete_matrix(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    matrix = frame.pivot(index="spatial_cpd", columns="temporal_hz", values=value).sort_index().sort_index(axis=1)
    if matrix.isna().any().any():
        raise ValueError(f"Incomplete matrix for {value}: {matrix.shape}")
    return matrix


def range_normalize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    lo = frame.min(axis=1)
    span = frame.max(axis=1) - lo
    valid = span > 1e-12
    return frame.sub(lo, axis=0).div(span.where(valid), axis=0), valid


def centroid(frame: pd.DataFrame) -> pd.Series:
    positive = frame.clip(lower=0.0)
    weights = positive.div(positive.sum(axis=1).replace(0.0, np.nan), axis=0)
    return np.exp2(weights @ np.log2(np.asarray(frame.columns, dtype=float)))


def fit_nonnegative_rank_one(matrix: pd.DataFrame) -> dict[str, object]:
    observed = matrix.to_numpy(dtype=float)
    left, singular, right_t = np.linalg.svd(observed, full_matrices=False)
    sf_raw = left[:, 0].copy()
    tf_raw = right_t[0].copy()
    if np.sum(sf_raw) < 0:
        sf_raw *= -1
        tf_raw *= -1
    sf_raw = np.maximum(sf_raw, 0.0)
    tf_raw = np.maximum(tf_raw, 0.0)
    sf_factor = sf_raw / max(float(sf_raw.max()), 1e-15)
    tf_factor = tf_raw / max(float(tf_raw.max()), 1e-15)
    gain = float(singular[0] * sf_raw.max() * tf_raw.max())
    reconstruction = gain * np.outer(sf_factor, tf_factor)
    residual = observed - reconstruction
    total_energy = float(np.sum(observed**2))
    centered_energy = float(np.sum((observed - observed.mean()) ** 2))
    residual_energy = float(np.sum(residual**2))
    return {
        "sf_factor": sf_factor,
        "tf_factor": tf_factor,
        "gain_hz": gain,
        "reconstruction": reconstruction,
        "residual": residual,
        "energy_fraction": 1.0 - residual_energy / max(total_energy, 1e-30),
        "centered_r2": 1.0 - residual_energy / centered_energy if centered_energy > 1e-30 else np.nan,
        "relative_rmse": float(np.sqrt(residual_energy / max(total_energy, 1e-30))),
    }


def prepare_surfaces(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    paths = {
        "condition_summary": args.source_dir / "native_condition_unit_summary.csv",
        "source_manifest": args.source_dir / "analysis_manifest.json",
        "sf_only_curves": args.sf_only_dir / "rr100_native_dense_sf_curves_long.csv",
        "sf_only_metrics": args.sf_only_dir / "rr100_native_dense_sf_unit_metrics.csv",
        "sf_only_manifest": args.sf_only_dir / "manifest.json",
    }
    data = pd.read_csv(paths["condition_summary"])
    dynamic = data.loc[data["condition_kind"].eq("drifting_grating") & data["primary_fit_support"].astype(bool)].copy()
    phase = dynamic.groupby(
        ["rr100_index", "session", "orientation_deg", "spatial_cpd", "signed_temporal_hz"], as_index=False
    ).agg(
        signed_f0_hz=("mean_rate_above_blank_hz", "mean"),
        absolute_f0_hz=("mean_rate_hz", "mean"),
        n_carrier_phases=("phase_index", "nunique"),
    )
    phase["temporal_hz"] = phase["signed_temporal_hz"].abs()
    folded = phase.groupby(
        ["rr100_index", "session", "orientation_deg", "spatial_cpd", "temporal_hz"], as_index=False
    ).agg(
        signed_f0_hz=("signed_f0_hz", "mean"),
        absolute_f0_hz=("absolute_f0_hz", "mean"),
        direction_difference_f0_hz=("signed_f0_hz", lambda x: float(np.max(x) - np.min(x))),
    )
    folded["positive_f0_hz"] = folded["signed_f0_hz"].clip(lower=0.0)
    folded["suppressive_f0_hz"] = (-folded["signed_f0_hz"]).clip(lower=0.0)

    orientation_scores = folded.groupby(["rr100_index", "session", "orientation_deg"], as_index=False).agg(
        mean_positive_f0_hz=("positive_f0_hz", "mean"),
        maximum_positive_f0_hz=("positive_f0_hz", "max"),
        mean_signed_f0_hz=("signed_f0_hz", "mean"),
    )
    preferred = orientation_scores.sort_values(
        ["rr100_index", "mean_positive_f0_hz", "maximum_positive_f0_hz", "orientation_deg"],
        ascending=[True, False, False, True],
    ).drop_duplicates("rr100_index")
    preferred = preferred.rename(columns={"orientation_deg": "preferred_orientation_deg"})
    selected = folded.merge(
        preferred[["rr100_index", "preferred_orientation_deg"]], on="rr100_index", how="inner", validate="many_to_one"
    )
    selected = selected.loc[np.isclose(selected["orientation_deg"], selected["preferred_orientation_deg"])].copy()
    if selected.groupby("rr100_index").size().nunique() != 1 or selected.groupby("rr100_index").size().iloc[0] != 104:
        raise ValueError("Expected 8 SF x 13 TF points per unit after F0 orientation selection")
    return selected, preferred, orientation_scores, phase, paths


def factorize(
    selected: pd.DataFrame, preferred: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    factors: list[dict[str, object]] = []
    points: list[dict[str, object]] = []
    for unit, frame in selected.groupby("rr100_index", sort=True):
        session = str(frame["session"].iloc[0])
        orientation = float(frame["preferred_orientation_deg"].iloc[0])
        signed = complete_matrix(frame, "signed_f0_hz")
        positive = complete_matrix(frame, "positive_f0_hz")
        fit = fit_nonnegative_rank_one(positive)
        sf = positive.index.to_numpy(dtype=float)
        tf = positive.columns.to_numpy(dtype=float)
        sf_factor = np.asarray(fit["sf_factor"], dtype=float)
        tf_factor = np.asarray(fit["tf_factor"], dtype=float)
        maximum = float(positive.to_numpy().max())
        responsive = maximum >= threshold
        direct_sf = positive.mean(axis=1).to_numpy(dtype=float)
        direct_tf = positive.mean(axis=0).to_numpy(dtype=float)
        summaries.append({
            "rr100_index": int(unit),
            "session": session,
            "preferred_orientation_deg": orientation,
            "surface_maximum_positive_f0_hz": maximum,
            "surface_minimum_signed_f0_hz": float(signed.to_numpy().min()),
            "surface_mean_signed_f0_hz": float(signed.to_numpy().mean()),
            "surface_positive_fraction": float(np.mean(signed.to_numpy() > 0)),
            "maximum_direction_difference_f0_hz": float(frame["direction_difference_f0_hz"].max()),
            "rank1_gain_f0_hz": float(fit["gain_hz"]),
            "rank1_energy_fraction": float(fit["energy_fraction"]),
            "rank1_centered_r2": float(fit["centered_r2"]),
            "rank1_relative_rmse": float(fit["relative_rmse"]),
            "responsive_positive_f0_flag": bool(responsive),
            "preferred_sf_cpd_factor": float(sf[np.argmax(sf_factor)]) if responsive else np.nan,
            "preferred_tf_hz_factor": float(tf[np.argmax(tf_factor)]) if responsive else np.nan,
            "preferred_sf_cpd_direct_marginal": float(sf[np.argmax(direct_sf)]) if responsive else np.nan,
            "preferred_tf_hz_direct_marginal": float(tf[np.argmax(direct_tf)]) if responsive else np.nan,
        })
        for axis, frequencies, factor_values, marginal_values in (
            ("spatial_frequency", sf, sf_factor, direct_sf),
            ("temporal_frequency", tf, tf_factor, direct_tf),
        ):
            marginal_norm = marginal_values / max(float(np.max(marginal_values)), 1e-15)
            for frequency, factor_value, marginal_value, marginal_normalized in zip(
                frequencies, factor_values, marginal_values, marginal_norm
            ):
                factors.append({
                    "rr100_index": int(unit), "session": session, "preferred_orientation_deg": orientation,
                    "axis": axis, "frequency": float(frequency), "normalized_factor": float(factor_value),
                    "direct_positive_f0_marginal_hz": float(marginal_value),
                    "direct_positive_f0_marginal_normalized": float(marginal_normalized),
                    "responsive_positive_f0_flag": bool(responsive),
                })
        reconstruction = np.asarray(fit["reconstruction"], dtype=float)
        excitatory_residual = np.asarray(fit["residual"], dtype=float)
        for i, spatial in enumerate(sf):
            for j, temporal in enumerate(tf):
                observed_signed = float(signed.iloc[i, j])
                points.append({
                    "rr100_index": int(unit), "session": session, "preferred_orientation_deg": orientation,
                    "spatial_cpd": float(spatial), "temporal_hz": float(temporal),
                    "observed_signed_f0_hz": observed_signed,
                    "observed_positive_f0_hz": float(positive.iloc[i, j]),
                    "observed_suppressive_f0_hz": max(-observed_signed, 0.0),
                    "rank1_reconstructed_positive_f0_hz": float(reconstruction[i, j]),
                    "excitatory_interaction_residual_f0_hz": float(excitatory_residual[i, j]),
                    "full_signed_residual_to_excitatory_fit_hz": observed_signed - float(reconstruction[i, j]),
                })
    return pd.DataFrame(summaries), pd.DataFrame(factors), pd.DataFrame(points)


def compare_sf_only(
    args: argparse.Namespace, summary: pd.DataFrame, factors: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    sf_long = pd.read_csv(args.sf_only_dir / "rr100_native_dense_sf_curves_long.csv")
    seq = sf_long.loc[sf_long["response_state"].eq("all_model_valid_bins")].pivot(
        index="rr100_index", columns="target_sf_cpd", values="mean_rate_hz"
    ).sort_index(axis=1)
    seq = seq.loc[:, seq.columns <= 8.0 * np.sqrt(2.0) + 1e-8]
    f0 = factors.loc[factors["axis"].eq("spatial_frequency")].pivot(
        index="rr100_index", columns="frequency", values="normalized_factor"
    ).sort_index(axis=1)
    if seq.shape != (100, 8) or f0.shape != (100, 8) or not np.allclose(seq.columns, f0.columns):
        raise ValueError("SF-only and F0-factor supports do not match")
    seq_norm, _ = range_normalize(seq)
    f0_norm, _ = range_normalize(f0)
    seq_centroid = centroid(seq_norm)
    f0_centroid = centroid(f0_norm)
    table = summary.copy().set_index("rr100_index")
    table["sf_only_peak_cpd"] = seq.idxmax(axis=1)
    table["sf_only_centroid_cpd"] = seq_centroid
    table["joint_f0_centroid_cpd"] = f0_centroid
    correlations = []
    for unit in table.index:
        correlations.append(float(pearsonr(seq_norm.loc[unit], f0_norm.loc[unit]).statistic))
    table["sf_curve_pearson_r"] = correlations
    table["signed_peak_difference_octaves"] = np.log2(table["preferred_sf_cpd_factor"] / table["sf_only_peak_cpd"])
    table["signed_centroid_difference_octaves"] = np.log2(table["joint_f0_centroid_cpd"] / table["sf_only_centroid_cpd"])
    responsive = table["responsive_positive_f0_flag"]
    d = table.loc[responsive].copy()
    delta = d["signed_peak_difference_octaves"].to_numpy(dtype=float)
    peak_test = spearmanr(np.log2(d["sf_only_peak_cpd"]), np.log2(d["preferred_sf_cpd_factor"]))
    centroid_test = spearmanr(np.log2(d["sf_only_centroid_cpd"]), np.log2(d["joint_f0_centroid_cpd"]))
    sf_values = np.sort(seq.columns.to_numpy(dtype=float))
    sequence_hist = np.asarray([np.count_nonzero(np.isclose(d["sf_only_peak_cpd"], sf)) for sf in sf_values], dtype=float)
    joint_hist = np.asarray([np.count_nonzero(np.isclose(d["preferred_sf_cpd_factor"], sf)) for sf in sf_values], dtype=float)
    stats: dict[str, float | int] = {
        "n_f0_responsive_units": int(len(d)),
        "n_f0_nonresponsive_controls": int((~responsive).sum()),
        "n_exact_sf_peaks": int(np.count_nonzero(np.isclose(delta, 0))),
        "exact_sf_peak_fraction": float(np.mean(np.isclose(delta, 0))),
        "n_within_half_octave": int(np.count_nonzero(np.abs(delta) <= 0.5 + 1e-8)),
        "within_half_octave_fraction": float(np.mean(np.abs(delta) <= 0.5 + 1e-8)),
        "n_within_one_octave": int(np.count_nonzero(np.abs(delta) <= 1.0 + 1e-8)),
        "within_one_octave_fraction": float(np.mean(np.abs(delta) <= 1.0 + 1e-8)),
        "median_signed_peak_difference_octaves": float(np.median(delta)),
        "peak_spearman_rho": float(peak_test.statistic),
        "peak_spearman_p": float(peak_test.pvalue),
        "centroid_spearman_rho": float(centroid_test.statistic),
        "centroid_spearman_p": float(centroid_test.pvalue),
        "median_sf_curve_pearson_r": float(d["sf_curve_pearson_r"].median()),
        "sf_peak_distribution_total_variation": float(
            0.5 * np.abs(sequence_hist / sequence_hist.sum() - joint_hist / joint_hist.sum()).sum()
        ),
        "factor_vs_direct_sf_peak_exact_fraction": float(np.mean(np.isclose(d["preferred_sf_cpd_factor"], d["preferred_sf_cpd_direct_marginal"]))),
        "factor_vs_direct_tf_peak_exact_fraction": float(np.mean(np.isclose(d["preferred_tf_hz_factor"], d["preferred_tf_hz_direct_marginal"]))),
        "median_rank1_centered_r2": float(d["rank1_centered_r2"].median()),
        "n_rank1_centered_r2_ge_0p8": int((d["rank1_centered_r2"] >= 0.8).sum()),
        "n_tf_peaks_at_low_edge_0p5_hz": int(np.isclose(d["preferred_tf_hz_factor"], 0.5).sum()),
        "n_tf_peaks_at_high_edge_32_hz": int(np.isclose(d["preferred_tf_hz_factor"], 32.0).sum()),
    }
    curves = pd.concat([
        seq_norm.rename_axis(index="rr100_index", columns="sf_cpd").stack().rename("range_normalized_tuning").reset_index().assign(curve_source="sf_only_native_sequence"),
        f0_norm.rename_axis(index="rr100_index", columns="sf_cpd").stack().rename("range_normalized_tuning").reset_index().assign(curve_source="joint_dynamic_f0_factor"),
    ], ignore_index=True)
    return table.reset_index(), curves, stats


def choose_examples(unit_table: pd.DataFrame) -> pd.DataFrame:
    responsive = unit_table.loc[unit_table["responsive_positive_f0_flag"]].copy()
    strong = responsive.loc[responsive["surface_maximum_positive_f0_hz"] >= 5.0].copy()
    controls = unit_table.loc[~unit_table["responsive_positive_f0_flag"]].copy()
    roles: list[tuple[str, int, str, float]] = []

    def add(role: str, pool: pd.DataFrame, criterion: str, maximize: bool) -> None:
        used = {u for _, u, _, _ in roles}
        p = pool.loc[~pool["rr100_index"].isin(used) & pool[criterion].notna()]
        row = p.loc[p[criterion].idxmax() if maximize else p[criterion].idxmin()]
        roles.append((role, int(row["rr100_index"]), criterion, float(row[criterion])))

    add("strongest positive F0", responsive, "surface_maximum_positive_f0_hz", True)
    add("strong separable F0", strong, "rank1_centered_r2", True)
    add("largest F0 interaction", strong, "rank1_centered_r2", False)
    add("strongest suppressive component", responsive, "surface_minimum_signed_f0_hz", False)
    add("weak or suppressive-only control", controls, "surface_maximum_positive_f0_hz", False)
    selected = pd.DataFrame(roles, columns=["selection_role", "rr100_index", "criterion", "criterion_value"])
    return selected.merge(unit_table, on="rr100_index", how="left", validate="one_to_one")


def heatmap(ax, matrix: pd.DataFrame, title: str, *, signed: bool, vmax: float) -> None:
    if signed:
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="coolwarm", norm=TwoSlopeNorm(0, -vmax, vmax))
    else:
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(matrix.shape[1]), [f"{x:g}" for x in matrix.columns], rotation=55, ha="right", fontsize=7)
    ax.set_yticks(np.arange(matrix.shape[0]), [f"{x:g}" for x in matrix.index], fontsize=7)
    ax.set_xlabel("TF (Hz)"); ax.set_ylabel("SF (cpd)"); ax.set_title(title, fontsize=9)
    plt.colorbar(image, ax=ax, shrink=0.78, label="F0 above blank (Hz)")


def plot_selected(points: pd.DataFrame, selected: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(len(selected), 4, figsize=(16, 3.0 * len(selected)), constrained_layout=True)
    for row_i, (_, choice) in enumerate(selected.iterrows()):
        unit = int(choice["rr100_index"])
        p = points.loc[points["rr100_index"].eq(unit)]
        signed = complete_matrix(p, "observed_signed_f0_hz")
        positive = complete_matrix(p, "observed_positive_f0_hz")
        recon = complete_matrix(p, "rank1_reconstructed_positive_f0_hz")
        residual = complete_matrix(p, "full_signed_residual_to_excitatory_fit_hz")
        signed_vmax = max(float(np.abs(signed.to_numpy()).max()), 1e-6)
        positive_vmax = max(float(positive.to_numpy().max()), 1e-6)
        residual_vmax = max(float(np.abs(residual.to_numpy()).max()), 1e-6)
        heatmap(axes[row_i, 0], signed, "signed observed F0", signed=True, vmax=signed_vmax)
        heatmap(axes[row_i, 1], positive, "excitatory max(F0,0)", signed=False, vmax=positive_vmax)
        heatmap(axes[row_i, 2], recon, "rank-one excitatory fit", signed=False, vmax=positive_vmax)
        heatmap(axes[row_i, 3], residual, "signed residual to fit", signed=True, vmax=residual_vmax)
        axes[row_i, 0].text(
            -0.46, 0.5,
            f"{choice['selection_role']}\nRR100 {unit}\nori {choice['preferred_orientation_deg']:g}°\n"
            f"max +F0 {choice['surface_maximum_positive_f0_hz']:.2f} Hz\nR² {choice['rank1_centered_r2']:.2f}",
            transform=axes[row_i, 0].transAxes, ha="right", va="center", fontsize=9,
        )
    fig.suptitle("Dynamic-F0 joint SF×TF surfaces: observed sign, fitted excitation, and failures", fontsize=14)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_distributions(unit_table: pd.DataFrame, out: Path, dpi: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = unit_table.loc[unit_table["responsive_positive_f0_flag"]].copy()
    sf_values = np.sort(d["sf_only_peak_cpd"].unique())
    tf_values = TF_SUPPORT.copy()
    sf_counts = []
    for source, column in (("sf_only_native_sequence", "sf_only_peak_cpd"), ("joint_dynamic_f0_factor", "preferred_sf_cpd_factor")):
        sf_counts.extend({"curve_source": source, "sf_cpd": float(sf), "n_units": int(np.count_nonzero(np.isclose(d[column], sf)))} for sf in sf_values)
    sf_counts_frame = pd.DataFrame(sf_counts)
    tf_counts = pd.DataFrame({
        "tf_hz": tf_values,
        "n_units": [int(np.count_nonzero(np.isclose(d["preferred_tf_hz_factor"], tf))) for tf in tf_values],
    })

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.6), constrained_layout=True)
    x = np.arange(len(sf_values)); width = 0.38
    a = sf_counts_frame.loc[sf_counts_frame["curve_source"].eq("sf_only_native_sequence"), "n_units"].to_numpy()
    b = sf_counts_frame.loc[sf_counts_frame["curve_source"].eq("joint_dynamic_f0_factor"), "n_units"].to_numpy()
    axes[0, 0].bar(x-width/2, a, width, color=SF_COLOR, label="original-sequence mean rate (F0)")
    axes[0, 0].bar(x+width/2, b, width, color=F0_COLOR, label="joint dynamic-F0 factor")
    axes[0, 0].set_xticks(x, [f"{v:g}" for v in sf_values], rotation=35, ha="right")
    axes[0, 0].set(xlabel="preferred SF (cpd)", ylabel="responsive units")
    axes[0, 0].set_title("A  SF preference distributions", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)

    transition = pd.crosstab(d["sf_only_peak_cpd"], d["preferred_sf_cpd_factor"]).reindex(
        index=sf_values, columns=sf_values, fill_value=0
    )
    im = axes[0, 1].imshow(transition, origin="lower", cmap="Greens")
    axes[0, 1].set_xticks(x, [f"{v:g}" for v in sf_values], rotation=35, ha="right")
    axes[0, 1].set_yticks(x, [f"{v:g}" for v in sf_values])
    axes[0, 1].set(xlabel="joint F0 preferred SF (cpd)", ylabel="sequence preferred SF (cpd)")
    axes[0, 1].set_title("B  Paired SF peak transitions", loc="left", fontweight="bold")
    for i in range(len(sf_values)):
        for j in range(len(sf_values)):
            n = int(transition.iloc[i, j])
            if n:
                axes[0, 1].text(j, i, str(n), ha="center", va="center", fontsize=8,
                                color="white" if n > transition.to_numpy().max()*0.55 else "black")
    fig.colorbar(im, ax=axes[0, 1], label="unit count", shrink=0.82)

    tx = np.arange(len(tf_values))
    axes[1, 0].bar(tx, tf_counts["n_units"], color="#7A5195")
    axes[1, 0].set_xticks(tx, [f"{v:g}" for v in tf_values], rotation=40, ha="right")
    axes[1, 0].set(xlabel="preferred TF (Hz)", ylabel="responsive units")
    axes[1, 0].set_title("C  Joint dynamic-F0 TF preferences", loc="left", fontweight="bold")

    xx = np.log2(d["sf_only_centroid_cpd"]); yy = np.log2(d["joint_f0_centroid_cpd"])
    scatter = axes[1, 1].scatter(xx, yy, c=d["sf_curve_pearson_r"], cmap="coolwarm", vmin=-1, vmax=1,
                                 s=38, alpha=0.8, edgecolor="white", linewidth=0.35)
    lo, hi = np.log2(sf_values[[0,-1]])
    axes[1, 1].plot([lo,hi],[lo,hi],"--",color="0.3")
    ticks=np.log2(sf_values); labels=[f"{v:g}" for v in sf_values]
    axes[1, 1].set_xticks(ticks,labels,rotation=35,ha="right"); axes[1, 1].set_yticks(ticks,labels)
    axes[1, 1].set_xlim(lo-.15,hi+.15); axes[1, 1].set_ylim(lo-.15,hi+.15)
    axes[1, 1].set(xlabel="sequence F0 SF centroid (cpd)",ylabel="joint F0 SF centroid (cpd)")
    rho=spearmanr(xx,yy).statistic
    axes[1, 1].text(.03,.97,f"Spearman rho={rho:.2f}\nmedian curve r={d['sf_curve_pearson_r'].median():.2f}",
                    transform=axes[1, 1].transAxes,va="top")
    axes[1, 1].set_title("D  Same-unit continuous SF agreement",loc="left",fontweight="bold")
    fig.colorbar(scatter,ax=axes[1,1],label="8-point SF curve r",shrink=.82)
    fig.suptitle(
        f"RR100 joint SF×TF tuning from dynamic F0 (n={len(d)} responsive; {len(unit_table)-len(d)} controls excluded)\n"
        "Signed F0 preserved; preferences from rank-one fit to max(F0,0)", fontsize=14,
    )
    fig.savefig(out,dpi=dpi,bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)
    transitions = transition.rename_axis(index="sf_only_peak_cpd",columns="joint_f0_peak_cpd").stack().rename("n_units").reset_index()
    return pd.concat([sf_counts_frame, tf_counts.assign(curve_source="joint_dynamic_f0_factor")],ignore_index=True,sort=False), transitions


def main() -> None:
    args = parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    selected_surface, preferred, orientation_scores, phase, paths = prepare_surfaces(args)
    summary, factors, points = factorize(selected_surface, preferred, float(args.responsive_max_positive_f0_hz))
    unit_table, comparison_curves, comparison_stats = compare_sf_only(args, summary, factors)
    examples = choose_examples(unit_table)

    surface_png=args.out_dir/"selected_dynamic_f0_sf_tf_surfaces.png"
    distributions_png=args.out_dir/"dynamic_f0_sf_tf_preference_distributions.png"
    plot_selected(points,examples,surface_png,int(args.dpi))
    distribution_counts,transitions=plot_distributions(unit_table,distributions_png,int(args.dpi))

    preferred.to_csv(args.out_dir/"f0_preferred_orientation_by_unit.csv",index=False)
    orientation_scores.to_csv(args.out_dir/"f0_orientation_scores.csv",index=False)
    selected_surface.to_csv(args.out_dir/"direction_folded_signed_f0_points.csv",index=False)
    summary.to_csv(args.out_dir/"f0_separable_fit_unit_summary.csv",index=False)
    factors.to_csv(args.out_dir/"f0_separable_factor_points.csv",index=False)
    points.to_csv(args.out_dir/"f0_surface_fit_and_residual_points.csv",index=False)
    unit_table.to_csv(args.out_dir/"f0_vs_sequence_sf_unit_comparison.csv",index=False)
    comparison_curves.to_csv(args.out_dir/"f0_vs_sequence_normalized_sf_curves.csv",index=False)
    examples.to_csv(args.out_dir/"selected_unit_examples.csv",index=False)
    distribution_counts.to_csv(args.out_dir/"preference_distribution_counts.csv",index=False)
    transitions.to_csv(args.out_dir/"sf_peak_transition_counts.csv",index=False)
    pd.DataFrame([comparison_stats]).to_csv(args.out_dir/"f0_vs_sequence_sf_comparison_summary.csv",index=False)
    census=[]
    for threshold in (0.1,0.25,0.5,1.0,2.0,5.0):
        census.append({"maximum_positive_f0_threshold_hz":threshold,"n_units_passing":int((summary["surface_maximum_positive_f0_hz"]>=threshold).sum())})
    pd.DataFrame(census).to_csv(args.out_dir/"f0_response_threshold_census.csv",index=False)

    manifest={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "analysis":"primary RR100 joint SFxTF tuning from dynamic F0",
        "status":"F0 surface, selected examples, and preference-distribution checkpoint",
        "response_contract":"mean rate above gray blank; phase averaged; directions folded at matched |TF|",
        "orientation_contract":"orientation maximizing mean max(direction-folded signed F0,0) over primary SFxTF support",
        "fit_contract":"signed F0 preserved; rank-one nonnegative SVD only on max(direction-folded F0,0)",
        "response_threshold_hz":float(args.responsive_max_positive_f0_hz),
        "n_responsive":int(summary["responsive_positive_f0_flag"].sum()),
        "n_controls":int((~summary["responsive_positive_f0_flag"]).sum()),
        "sf_support_cpd":sorted(float(x) for x in selected_surface["spatial_cpd"].unique()),
        "tf_support_hz":sorted(float(x) for x in selected_surface["temporal_hz"].unique()),
        "inputs":{key:file_identity(path) for key,path in paths.items()},
        "comparison_stats":comparison_stats,
    }
    with (args.out_dir/"manifest.json").open("w",encoding="utf-8") as handle: json.dump(manifest,handle,indent=2)
    print(json.dumps(comparison_stats,indent=2))
    print("\nSF factor peaks:\n"+summary.loc[summary.responsive_positive_f0_flag,"preferred_sf_cpd_factor"].value_counts().sort_index().to_string())
    print("\nTF factor peaks:\n"+summary.loc[summary.responsive_positive_f0_flag,"preferred_tf_hz_factor"].value_counts().sort_index().to_string())
    print("\nSelected examples:\n"+examples[["selection_role","rr100_index","surface_maximum_positive_f0_hz","surface_minimum_signed_f0_hz","rank1_centered_r2"]].to_string(index=False))
    print(f"\nWrote {args.out_dir.resolve()}")


if __name__ == "__main__": main()
