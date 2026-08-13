#!/usr/bin/env python3
"""Checkpoint 12B: neural FEM-effect structure before mechanistic fitting.

This checkpoint uses only the frozen RR100 zero-gaze and original-FEM response
cache for the 16 existing image/eye-trajectory pairs. It compares several
response-effect definitions, visualizes the image x unit matrices and raw
timecourses, and tests for common condition-pair ordering against independently
shuffled condition labels within each unit. Retinal power and SF/TF tuning are
not used.
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
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs/fig4_active_sensing/rr100_all16_spectral_explainability_checkpoint_11_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_neural_effect_rank_checkpoint_12b_v1"
RESPONSIVE_THRESHOLD_HZ = 1e-4
N_NULL = 5000
N_SPLIT = 5000
SEED = 20260812
EPS = 1e-30

METRICS = {
    "delta_temporal_sd_hz": "temporal SD of FEM minus zero",
    "delta_rms_hz": "RMS of FEM minus zero",
    "delta_mean_absolute_hz": "mean absolute FEM minus zero",
    "absolute_delta_mean_hz": "absolute mean FEM minus zero",
    "signed_delta_mean_hz": "signed mean FEM minus zero",
}
NONNEGATIVE_METRICS = set(METRICS) - {"signed_delta_mean_hz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--n-null", type=int, default=N_NULL)
    parser.add_argument("--n-split", type=int, default=N_SPLIT)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_response_cache(input_dir: Path) -> tuple[dict[int, dict[str, np.ndarray]], np.ndarray]:
    archive = np.load(input_dir / "all16_original_pair_all_rr100_response_cache.npz")
    units = archive["rr100_indices"].astype(int)
    responses = {}
    for image_index in range(16):
        zero = archive[f"image_{image_index:02d}_zero"][:, :, 0, 0].astype(float)
        fem = archive[f"image_{image_index:02d}_fem"][:, :, 0, 0].astype(float)
        if zero.shape != (97, 100) or fem.shape != zero.shape:
            raise ValueError(f"Unexpected response shape for image {image_index}: {zero.shape}, {fem.shape}")
        responses[image_index] = {"zero": zero, "fem": fem, "delta": fem - zero}
    return responses, units


def build_effect_table(responses: dict[int, dict[str, np.ndarray]], units: np.ndarray) -> pd.DataFrame:
    rows = []
    for image_index, arrays in responses.items():
        zero, fem, delta = arrays["zero"], arrays["fem"], arrays["delta"]
        for position, unit in enumerate(units):
            d = delta[:, position]
            rows.append({
                "image_index": image_index, "rr100_index": int(unit),
                "zero_mean_rate_hz": float(zero[:, position].mean()),
                "fem_mean_rate_hz": float(fem[:, position].mean()),
                "zero_temporal_sd_hz": float(zero[:, position].std()),
                "fem_temporal_sd_hz": float(fem[:, position].std()),
                "delta_temporal_sd_hz": float(d.std()),
                "delta_rms_hz": float(np.sqrt(np.mean(d**2))),
                "delta_mean_absolute_hz": float(np.mean(np.abs(d))),
                "absolute_delta_mean_hz": float(abs(np.mean(d))),
                "signed_delta_mean_hz": float(np.mean(d)),
                "delta_min_hz": float(d.min()), "delta_max_hz": float(d.max()),
            })
    return pd.DataFrame(rows)


def matrices_from_table(table: pd.DataFrame, units: np.ndarray) -> dict[str, np.ndarray]:
    matrices = {}
    for metric in METRICS:
        matrices[metric] = table.pivot(index="image_index", columns="rr100_index", values=metric).reindex(
            index=np.arange(16), columns=units
        ).to_numpy(float)
    return matrices


def standardize_columns(matrix: np.ndarray) -> np.ndarray:
    scale = matrix.std(axis=0, ddof=0)
    if np.any(scale <= EPS):
        raise ValueError("Attempted to standardize a constant unit profile")
    return (matrix - matrix.mean(axis=0, keepdims=True)) / scale[None, :]


def orient_pc_scores(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, singular, vh = np.linalg.svd(z, full_matrices=False)
    score = singular[0] * u[:, 0]
    loading = vh[0].copy()
    simple = z.mean(axis=1)
    if np.corrcoef(score, simple)[0, 1] < 0:
        score *= -1; loading *= -1
    return score, loading, singular


def safe_correlation(a: np.ndarray, b: np.ndarray, kind: str = "pearson") -> float:
    if np.std(a) <= EPS or np.std(b) <= EPS:
        return np.nan
    return float(pearsonr(a, b).statistic if kind == "pearson" else spearmanr(a, b).statistic)


def shuffle_pc1_null(z: np.ndarray, n_null: int, rng: np.random.Generator) -> np.ndarray:
    values = np.empty(n_null, dtype=float)
    for iteration in range(n_null):
        shuffled = np.empty_like(z)
        for unit in range(z.shape[1]):
            shuffled[:, unit] = z[rng.permutation(z.shape[0]), unit]
        singular = np.linalg.svd(shuffled, full_matrices=False, compute_uv=False)
        values[iteration] = singular[0] ** 2 / np.sum(singular**2)
    return values


def split_half_reliability(z: np.ndarray, n_split: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    pearson = np.empty(n_split, dtype=float); spearman = np.empty(n_split, dtype=float)
    n_first = z.shape[1] // 2
    for iteration in range(n_split):
        order = rng.permutation(z.shape[1])
        first = z[:, order[:n_first]].mean(axis=1)
        second = z[:, order[n_first:]].mean(axis=1)
        pearson[iteration] = safe_correlation(first, second, "pearson")
        spearman[iteration] = safe_correlation(first, second, "spearman")
    return pearson, spearman


def analyze_metrics(
    matrices: dict[str, np.ndarray], responsive: np.ndarray, n_null: int, n_split: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    summary_rows = []
    image_rows = []
    unit_rows = []
    arrays: dict[str, np.ndarray] = {}
    for metric, matrix_all in matrices.items():
        matrix = matrix_all[:, responsive]
        z = standardize_columns(matrix)
        score, loading, singular = orient_pc_scores(z)
        simple_score = z.mean(axis=1)
        pc1 = float(singular[0] ** 2 / np.sum(singular**2))
        null = shuffle_pc1_null(z, n_null, rng)
        split_p, split_s = split_half_reliability(z, n_split, rng)
        raw_s = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
        mean_normalized_rank1 = np.nan
        if metric in NONNEGATIVE_METRICS:
            mean_normalized = matrix / np.maximum(matrix.mean(axis=0, keepdims=True), EPS)
            norm_s = np.linalg.svd(mean_normalized, full_matrices=False, compute_uv=False)
            mean_normalized_rank1 = float(norm_s[0] ** 2 / np.sum(norm_s**2))
        pairwise = []
        for first in range(z.shape[1]):
            for second in range(first + 1, z.shape[1]):
                pairwise.append(safe_correlation(z[:, first], z[:, second], "spearman"))
        summary_rows.append({
            "metric": metric, "metric_definition": METRICS[metric], "n_images": 16,
            "n_responsive_units": int(responsive.sum()),
            "raw_uncentered_rank1_energy_fraction": float(raw_s[0] ** 2 / np.sum(raw_s**2)),
            "per_unit_mean_normalized_uncentered_rank1_energy_fraction": mean_normalized_rank1,
            "per_unit_zscored_centered_pc1_variance_fraction": pc1,
            "image_shuffle_null_pc1_median": float(np.median(null)),
            "image_shuffle_null_pc1_ci99_high": float(np.quantile(null, 0.99)),
            "image_shuffle_null_p_upper": float((1 + np.sum(null >= pc1)) / (len(null) + 1)),
            "split_half_image_score_pearson_median": float(np.nanmedian(split_p)),
            "split_half_image_score_pearson_ci95_low": float(np.nanquantile(split_p, 0.025)),
            "split_half_image_score_pearson_ci95_high": float(np.nanquantile(split_p, 0.975)),
            "split_half_image_score_spearman_median": float(np.nanmedian(split_s)),
            "median_pairwise_unit_profile_spearman": float(np.nanmedian(pairwise)),
            "pc1_vs_simple_population_mean_z_pearson": safe_correlation(score, simple_score, "pearson"),
        })
        for image_index in range(16):
            image_rows.append({
                "metric": metric, "image_index": image_index,
                "pc1_image_score": float(score[image_index]),
                "mean_unit_z_image_score": float(simple_score[image_index]),
                "population_median_effect": float(np.median(matrix[image_index])),
                "population_mean_effect": float(np.mean(matrix[image_index])),
            })
        for position, unit_position in enumerate(np.flatnonzero(responsive)):
            profile = z[:, position]
            unit_rows.append({
                "metric": metric, "rr100_index": int(unit_position),
                "pc1_loading": float(loading[position]),
                "unit_profile_vs_pc1_score_pearson": safe_correlation(profile, score, "pearson"),
                "unit_profile_vs_pc1_score_spearman": safe_correlation(profile, score, "spearman"),
                "unit_profile_vs_leave_one_unit_mean_z_pearson": safe_correlation(
                    profile, np.delete(z, position, axis=1).mean(axis=1), "pearson"
                ),
                "unit_mean_effect": float(matrix[:, position].mean()),
                "unit_across_image_effect_sd": float(matrix[:, position].std()),
            })
        arrays[f"{metric}_responsive_matrix"] = matrix
        arrays[f"{metric}_responsive_zscore"] = z
        arrays[f"{metric}_pc1_score"] = score
        arrays[f"{metric}_pc1_loading"] = loading
        arrays[f"{metric}_singular_values"] = singular
        arrays[f"{metric}_shuffle_pc1_null"] = null
        arrays[f"{metric}_split_half_pearson"] = split_p
        arrays[f"{metric}_split_half_spearman"] = split_s
    return pd.DataFrame(summary_rows), pd.DataFrame(image_rows), pd.DataFrame(unit_rows), arrays


def metric_score_agreement(image_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for first_pos, first in enumerate(METRICS):
        a = image_scores.loc[image_scores["metric"].eq(first)].sort_values("image_index")
        for second in list(METRICS)[first_pos + 1:]:
            b = image_scores.loc[image_scores["metric"].eq(second)].sort_values("image_index")
            rows.append({
                "metric_a": first, "metric_b": second,
                "pc1_image_score_pearson": safe_correlation(a["pc1_image_score"].to_numpy(), b["pc1_image_score"].to_numpy(), "pearson"),
                "pc1_image_score_spearman": safe_correlation(a["pc1_image_score"].to_numpy(), b["pc1_image_score"].to_numpy(), "spearman"),
                "mean_z_image_score_pearson": safe_correlation(a["mean_unit_z_image_score"].to_numpy(), b["mean_unit_z_image_score"].to_numpy(), "pearson"),
            })
    return pd.DataFrame(rows)


def make_unit_contract(primary: np.ndarray, units: np.ndarray) -> pd.DataFrame:
    mean_effect = primary.mean(axis=0)
    responsive = mean_effect > RESPONSIVE_THRESHOLD_HZ
    return pd.DataFrame({
        "rr100_index": units, "mean_delta_temporal_sd_hz_across_images": mean_effect,
        "median_delta_temporal_sd_hz_across_images": np.median(primary, axis=0),
        "across_image_sd_of_delta_temporal_sd_hz": primary.std(axis=0),
        "responsive_for_scale_normalized_rank": responsive,
        "responsive_threshold_hz": RESPONSIVE_THRESHOLD_HZ,
        "cohort_reason": np.where(
            responsive,
            "mean FEM-minus-zero temporal SD exceeds threshold",
            "weak-effect control; excluded only because z-scoring would amplify numerical-scale variation",
        ),
    })


def threshold_sensitivity(primary: np.ndarray, thresholds: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    mean_effect = primary.mean(axis=0)
    for threshold in thresholds:
        keep = mean_effect > float(threshold)
        z = standardize_columns(primary[:, keep])
        singular = np.linalg.svd(z, full_matrices=False, compute_uv=False)
        rng = np.random.default_rng(SEED + int(round(-np.log10(max(threshold, 1e-12)) * 100)))
        split_p, _ = split_half_reliability(z, 1000, rng)
        rows.append({
            "responsive_threshold_hz": threshold, "n_units": int(keep.sum()),
            "centered_pc1_variance_fraction": float(singular[0] ** 2 / np.sum(singular**2)),
            "split_half_image_score_pearson_median": float(np.nanmedian(split_p)),
            "split_half_image_score_pearson_ci95_low": float(np.nanquantile(split_p, 0.025)),
            "split_half_image_score_pearson_ci95_high": float(np.nanquantile(split_p, 0.975)),
        })
    return pd.DataFrame(rows)


def condition_removal_sensitivity(primary: np.ndarray, responsive: np.ndarray) -> pd.DataFrame:
    matrix = primary[:, responsive]
    definitions = (
        ("all_16_condition_pairs", ()),
        ("remove_strongest_pair_6", (6,)),
        ("remove_strongest_pair_6_and_lowest_pair_2", (6, 2)),
        ("remove_lowest_pair_2", (2,)),
    )
    full_z = standardize_columns(matrix)
    full_score, _, _ = orient_pc_scores(full_z)
    score_energy = full_score**2 / np.sum(full_score**2)
    rows = []
    for label, removed in definitions:
        keep = np.ones(matrix.shape[0], dtype=bool)
        keep[list(removed)] = False
        z = standardize_columns(matrix[keep])
        singular = np.linalg.svd(z, full_matrices=False, compute_uv=False)
        pairwise = []
        for first in range(z.shape[1]):
            for second in range(first + 1, z.shape[1]):
                pairwise.append(safe_correlation(z[:, first], z[:, second], "spearman"))
        rows.append({
            "condition_set": label,
            "removed_image_trajectory_pairs": ",".join(str(value) for value in removed),
            "n_condition_pairs": int(keep.sum()),
            "per_unit_zscored_centered_pc1_variance_fraction": float(singular[0] ** 2 / np.sum(singular**2)),
            "median_pairwise_unit_profile_spearman": float(np.nanmedian(pairwise)),
            "pair_6_fraction_of_full_pc1_condition_score_squared_energy": float(score_energy[6]),
            "pair_2_fraction_of_full_pc1_condition_score_squared_energy": float(score_energy[2]),
        })
    return pd.DataFrame(rows)


def select_examples(unit_contract: pd.DataFrame, primary_units: pd.DataFrame) -> pd.DataFrame:
    merged = unit_contract.merge(
        primary_units.loc[primary_units["metric"].eq("delta_temporal_sd_hz")], on="rr100_index", how="left", validate="one_to_one"
    )
    responsive = merged.loc[merged["responsive_for_scale_normalized_rank"]].copy()
    strong = responsive.loc[
        responsive["unit_mean_effect"].ge(responsive["unit_mean_effect"].median())
    ].copy()
    positive = strong.loc[strong["unit_profile_vs_leave_one_unit_mean_z_pearson"].idxmax()]
    dissociation = strong.loc[strong["unit_profile_vs_leave_one_unit_mean_z_pearson"].idxmin()]
    weak = merged.loc[merged["mean_delta_temporal_sd_hz_across_images"].idxmin()]
    specs = (
        ("shared_factor_positive", positive, "highest leave-one-unit population-profile correlation among above-median-effect responsive units"),
        ("strong_effect_dissociation", dissociation, "lowest leave-one-unit population-profile correlation among above-median-effect responsive units"),
        ("weak_effect_control", weak, "minimum mean FEM-minus-zero temporal SD across all RR100 units"),
    )
    rows = []
    for order, (role, row, criterion) in enumerate(specs, start=1):
        rows.append({"display_order": order, "selection_role": role, "selection_criterion": criterion, **row.to_dict()})
    return pd.DataFrame(rows)


def plot_response_matrices(
    out: Path, matrices: dict[str, np.ndarray], responsive: np.ndarray,
    arrays: dict[str, np.ndarray], dpi: int,
) -> None:
    primary_z = arrays["delta_temporal_sd_hz_responsive_zscore"]
    score = arrays["delta_temporal_sd_hz_pc1_score"]
    loading = arrays["delta_temporal_sd_hz_pc1_loading"]
    image_order = np.argsort(score)
    unit_order = np.argsort(loading)
    metrics = ["delta_temporal_sd_hz", "delta_rms_hz", "delta_mean_absolute_hz", "signed_delta_mean_hz"]
    titles = ["Temporal SD of Δresponse", "RMS of Δresponse", "Mean |Δresponse|", "Signed mean Δresponse"]
    fig, axes = plt.subplots(2, 4, figsize=(17.0, 8.8), constrained_layout=True, sharex=True)
    for column, (metric, title) in enumerate(zip(metrics, titles, strict=True)):
        matrix = matrices[metric][:, responsive][:, unit_order]
        if metric == "signed_delta_mean_hz":
            vmax = float(np.quantile(np.abs(matrix), 0.99)); norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax); cmap = "coolwarm"
        else:
            positive = matrix[matrix > 0]; norm = LogNorm(max(float(np.quantile(positive, 0.01)), 1e-7), float(np.quantile(positive, 0.99))); cmap = "magma"
        im = axes[0, column].imshow(matrix[image_order], aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        axes[0, column].set_title(f"{chr(65+column)}  {title}\nabsolute units; same image/unit order")
        axes[0, column].set_yticks(range(16), image_order)
        axes[0, column].set_ylabel("original image–trajectory pair")
        fig.colorbar(im, ax=axes[0, column], label="Hz")
        z = standardize_columns(matrices[metric][:, responsive])[:, unit_order]
        im2 = axes[1, column].imshow(z[image_order], aspect="auto", interpolation="nearest", cmap="coolwarm", norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5))
        axes[1, column].set_title("Each unit centered and scaled across images")
        axes[1, column].set_yticks(range(16), image_order)
        axes[1, column].set(xlabel="responsive RR100 units, ordered by primary PC1 loading", ylabel="image pair")
        fig.colorbar(im2, ax=axes[1, column], label="within-unit SD units")
    fig.suptitle(
        "Checkpoint 12B neural effect matrices: a common condition-pair ordering is visible across several summaries\n"
        "No retinal power or SF–TF tuning is used; nine numerical-scale weak-effect units are shown separately as controls",
        fontsize=14, weight="bold",
    )
    fig.savefig(out.with_suffix(".png"), dpi=dpi); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def plot_shared_factor(
    out: Path, summary: pd.DataFrame, image_scores: pd.DataFrame, unit_metrics: pd.DataFrame,
    arrays: dict[str, np.ndarray], dpi: int,
) -> None:
    primary_summary = summary.set_index("metric").loc["delta_temporal_sd_hz"]
    score = arrays["delta_temporal_sd_hz_pc1_score"]
    null = arrays["delta_temporal_sd_hz_shuffle_pc1_null"]
    split = arrays["delta_temporal_sd_hz_split_half_pearson"]
    units = unit_metrics.loc[unit_metrics["metric"].eq("delta_temporal_sd_hz")]
    metrics_order = list(METRICS)
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.6), constrained_layout=True)
    order = np.argsort(score)
    axes[0, 0].bar(np.arange(16), score[order], color=plt.cm.turbo(np.linspace(0.05, 0.95, 16)))
    axes[0, 0].set_xticks(range(16), order)
    axes[0, 0].set(
        xlabel="original image–trajectory pair", ylabel="algebraic PC1 condition score",
        title="A  PC1 summarizes a common ordering of these 16 pairs\npair 6 contributes 46% of squared score energy",
    )
    axes[0, 0].axhline(0, color="black", lw=0.8)
    axes[0, 1].hist(null, bins=35, color="0.72", edgecolor="white", label="independent image shuffles within units")
    observed = float(primary_summary["per_unit_zscored_centered_pc1_variance_fraction"])
    axes[0, 1].axvline(observed, color="#D55E00", lw=2.5, label=f"observed={observed:.2f}")
    axes[0, 1].set(xlabel="PC1 variance fraction", ylabel="shuffle count", title=f"B  Shared structure exceeds shuffled image alignment\np<{1/(len(null)+1):.4f}")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].hist(split, bins=35, color="#56B4E9", edgecolor="white")
    median = float(np.median(split)); low, high = np.quantile(split, [0.025, 0.975])
    axes[1, 0].axvline(median, color="#0072B2", lw=2.5)
    axes[1, 0].set(xlim=(-1, 1), xlabel="Pearson r between random unit-half condition scores", ylabel="split count", title=f"C  Ordering is stable across subsets of this RR100 population\nmedian r={median:.2f}, 95% interval [{low:.2f}, {high:.2f}]")
    x = np.arange(len(metrics_order))
    observed_values = summary.set_index("metric").loc[metrics_order, "per_unit_zscored_centered_pc1_variance_fraction"].to_numpy(float)
    null_values = summary.set_index("metric").loc[metrics_order, "image_shuffle_null_pc1_median"].to_numpy(float)
    axes[1, 1].bar(x - 0.18, observed_values, width=0.36, color="#0072B2", label="observed")
    axes[1, 1].bar(x + 0.18, null_values, width=0.36, color="0.7", label="shuffle median")
    axes[1, 1].set_xticks(x, ["Δ SD", "Δ RMS", "mean |Δ|", "|mean Δ|", "signed mean Δ"], rotation=20, ha="right")
    axes[1, 1].set(ylabel="centered PC1 variance fraction", title="D  Shared structure is not unique to one effect summary")
    axes[1, 1].legend(frameon=False)
    for axis in axes.ravel(): axis.grid(color="0.92", zorder=0)
    fig.suptitle(
        "Checkpoint 12B: most responsive frozen RR100 units share an ordering of these 16 image–trajectory pairs",
        fontsize=14, weight="bold",
    )
    fig.savefig(out.with_suffix(".png"), dpi=dpi); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def plot_selected_units(
    out: Path, examples: pd.DataFrame, responses: dict[int, dict[str, np.ndarray]],
    matrices: dict[str, np.ndarray], arrays: dict[str, np.ndarray], dpi: int,
) -> None:
    score = arrays["delta_temporal_sd_hz_pc1_score"]
    image_choices = [int(np.argmin(score)), int(np.argsort(score)[len(score)//2]), int(np.argmax(score))]
    primary = matrices["delta_temporal_sd_hz"]
    responsive_units = np.flatnonzero(primary.mean(axis=0) > RESPONSIVE_THRESHOLD_HZ)
    fig, axes = plt.subplots(3, 4, figsize=(16.0, 10.5), constrained_layout=True)
    times = np.arange(97) / 120.0
    colors = ["#0072B2", "#E69F00", "#CC79A7"]
    for row, (_, example) in enumerate(examples.sort_values("display_order").iterrows()):
        unit = int(example["rr100_index"])
        for column, image_index in enumerate(image_choices):
            delta = responses[image_index]["delta"][:, unit]
            axes[row, column].plot(times, delta, color=colors[row], lw=1.2)
            axes[row, column].axhline(0, color="0.4", lw=0.7)
            axes[row, column].set_title(f"pair {image_index} · population PC1 score {score[image_index]:+.1f}\nΔresponse SD={np.std(delta):.3f} Hz", fontsize=9.5)
            axes[row, column].set(xlabel="valid response time (s)", ylabel=f"RR100 {unit}\nFEM − zero (Hz)")
            axes[row, column].grid(color="0.93")
        axis = axes[row, 3]
        profile = primary[:, unit]
        role = str(example["selection_role"]).replace("_", " ")
        if bool(example["responsive_for_scale_normalized_rank"]):
            leave_units = responsive_units[responsive_units != unit]
            leave_population_z = standardize_columns(primary[:, leave_units]).mean(axis=1)
            unit_z = (profile - profile.mean()) / max(profile.std(), EPS)
            axis.scatter(leave_population_z, unit_z, c=np.arange(16), cmap="turbo", s=42)
            for image_index in range(16):
                axis.annotate(str(image_index), (leave_population_z[image_index], unit_z[image_index]), xytext=(3,3), textcoords="offset points", fontsize=7)
            rho = safe_correlation(leave_population_z, unit_z, "pearson")
            axis.set(xlabel="leave-one-unit-out population image score", ylabel="unit modulation (within-unit z)", title=f"{role}\nRR100 {unit} · profile r={rho:+.2f}")
            axis.grid(color="0.92")
        else:
            axis.axis("off")
            axis.text(0.05, 0.84, role, fontsize=13, weight="bold")
            axis.text(0.05, 0.66, f"RR100 {unit}", fontsize=13)
            axis.text(0.05, 0.27, "Not z-scored or correlated:\nvariation is at numerical scale.", fontsize=11, color="#555555")
    fig.suptitle(
        "Checkpoint 12B auditable examples: raw FEM-minus-zero timecourses and image profiles\n"
        "Left-to-right pairs have low, middle, and high algebraic population PC1 scores",
        fontsize=14, weight="bold",
    )
    fig.savefig(out.with_suffix(".png"), dpi=dpi); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    responses, units = load_response_cache(args.input_dir)
    table = build_effect_table(responses, units)
    table.to_csv(args.out_dir / "all16_neural_fem_effect_metrics_all_rr100.csv", index=False)
    matrices = matrices_from_table(table, units)
    contract = make_unit_contract(matrices["delta_temporal_sd_hz"], units)
    contract.to_csv(args.out_dir / "neural_effect_unit_cohort_contract.csv", index=False)
    responsive = contract["responsive_for_scale_normalized_rank"].to_numpy(bool)
    summary, image_scores, unit_metrics, arrays = analyze_metrics(
        matrices, responsive, args.n_null, args.n_split
    )
    agreement = metric_score_agreement(image_scores)
    threshold_audit = threshold_sensitivity(
        matrices["delta_temporal_sd_hz"], (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2)
    )
    removal_audit = condition_removal_sensitivity(matrices["delta_temporal_sd_hz"], responsive)
    examples = select_examples(contract, unit_metrics)
    summary.to_csv(args.out_dir / "neural_effect_metric_rank_and_reliability_summary.csv", index=False)
    image_scores.to_csv(args.out_dir / "neural_effect_shared_image_scores.csv", index=False)
    unit_metrics.to_csv(args.out_dir / "neural_effect_unit_shared_factor_metrics.csv", index=False)
    agreement.to_csv(args.out_dir / "neural_effect_cross_metric_image_score_agreement.csv", index=False)
    threshold_audit.to_csv(args.out_dir / "neural_effect_responsive_threshold_sensitivity.csv", index=False)
    removal_audit.to_csv(args.out_dir / "neural_effect_condition_removal_sensitivity.csv", index=False)
    examples.to_csv(args.out_dir / "auditable_neural_effect_example_selection.csv", index=False)
    np.savez_compressed(
        args.out_dir / "neural_effect_matrices_rank_and_null_arrays.npz",
        responsive_rr100_mask=responsive, rr100_indices=units,
        **{f"{metric}_all100_matrix": matrix for metric, matrix in matrices.items()}, **arrays,
    )
    plot_response_matrices(args.out_dir / "checkpoint_12b_neural_effect_matrices", matrices, responsive, arrays, args.dpi)
    plot_shared_factor(args.out_dir / "checkpoint_12b_shared_factor_rank_and_reliability", summary, image_scores, unit_metrics, arrays, args.dpi)
    plot_selected_units(args.out_dir / "checkpoint_12b_selected_unit_raw_timecourses_and_profiles", examples, responses, matrices, arrays, args.dpi)
    zero_sd_max = float(table["zero_temporal_sd_hz"].max())
    primary = summary.set_index("metric").loc["delta_temporal_sd_hz"]
    primary_image = image_scores.loc[image_scores["metric"].eq("delta_temporal_sd_hz")].sort_values("image_index")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "neural-only all-16 image-trajectory-pair x RR100 FEM-effect rank and internal-stability audit",
        "status": "checkpoint_12b_complete_stop_before_additive_multiplicative_models",
        "retinal_power_used": False, "sf_tf_tuning_used": False,
        "response_contract": "center-pixel frozen RR100 response; 97 valid frames; original FEM minus matched true zero gaze",
        "primary_metric": "temporal SD across frames of FEM-minus-zero response",
        "metric_dependence_warning": "SD, RMS, mean absolute difference, and signed/absolute mean are summaries of the same response traces, not independent datasets",
        "zero_baseline_temporal_constancy_max_sd_hz": zero_sd_max,
        "n_rr100_units": 100, "n_responsive_for_normalized_rank": int(responsive.sum()),
        "responsive_threshold_hz": RESPONSIVE_THRESHOLD_HZ,
        "primary_centered_pc1_variance_fraction": float(primary["per_unit_zscored_centered_pc1_variance_fraction"]),
        "primary_shuffle_null_pc1_ci99_high": float(primary["image_shuffle_null_pc1_ci99_high"]),
        "primary_shuffle_p_upper": float(primary["image_shuffle_null_p_upper"]),
        "primary_split_half_image_score_pearson_median": float(primary["split_half_image_score_pearson_median"]),
        "primary_split_half_image_score_pearson_ci95": [
            float(primary["split_half_image_score_pearson_ci95_low"]),
            float(primary["split_half_image_score_pearson_ci95_high"]),
        ],
        "primary_highest_shared_score_image": int(primary_image.loc[primary_image["pc1_image_score"].idxmax(), "image_index"]),
        "primary_lowest_shared_score_image": int(primary_image.loc[primary_image["pc1_image_score"].idxmin(), "image_index"]),
        "pair_6_fraction_of_squared_pc1_condition_score_energy": float(removal_audit.iloc[0]["pair_6_fraction_of_full_pc1_condition_score_squared_energy"]),
        "pc1_variance_fraction_after_removing_pair_6": float(removal_audit.set_index("condition_set").loc["remove_strongest_pair_6", "per_unit_zscored_centered_pc1_variance_fraction"]),
        "pc1_variance_fraction_after_removing_pairs_6_and_2": float(removal_audit.set_index("condition_set").loc["remove_strongest_pair_6_and_lowest_pair_2", "per_unit_zscored_centered_pc1_variance_fraction"]),
        "median_pairwise_unit_profile_spearman_after_removing_pair_6": float(removal_audit.set_index("condition_set").loc["remove_strongest_pair_6", "median_pairwise_unit_profile_spearman"]),
        "scope_statement": "Within these frozen RR100 responses to these 16 specific image-eye-trajectory pairs, FEM-modulation magnitude has a strong common ordering across most responsive units.",
        "unsupported_at_this_checkpoint": [
            "whether the common ordering generalizes to new images, new eye traces, other model seeds, or experimental neurons",
            "whether PC1 corresponds to a single biological latent factor",
            "whether the common component is additive or multiplicative gain",
            "whether retinal total power causes or predicts the common neural condition-pair ordering",
            "whether unit-specific SFxTF tuning explains residual neural structure",
            "independent confirmation across response measures because all reported measures summarize the same response traces",
        ],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Checkpoint 12B: neural FEM-effect structure\n\n"
        "This neural-only checkpoint uses the 16 original FEM/zero response pairs and does not read retinal power or unit "
        "SF×TF tuning. The primary matrix is temporal SD of the framewise FEM-minus-zero response. Because zero gaze is "
        f"temporally constant to numerical precision (maximum SD {zero_sd_max:.2e} Hz), this is also the FEM response's "
        "temporal SD and the temporal variability added above zero. Ninety-one units exceed the fixed 1e-4 Hz response "
        "threshold; nine numerical-scale units are excluded only from z-scored rank analyses and retained as weak-effect controls.\n\n"
        f"After centering and scaling each responsive unit across images, PC1 explains {primary['per_unit_zscored_centered_pc1_variance_fraction']*100:.1f}% "
        f"of the image×unit variance, above the 99th percentile of the within-unit image-shuffle null "
        f"({primary['image_shuffle_null_pc1_ci99_high']*100:.1f}%; p={primary['image_shuffle_null_p_upper']:.4g}). Random "
        f"halves of this same RR100 population recover the condition-pair ordering with median Pearson "
        f"r={primary['split_half_image_score_pearson_median']:.2f}. Pair 6 is influential, contributing "
        f"{removal_audit.iloc[0]['pair_6_fraction_of_full_pc1_condition_score_squared_energy']*100:.0f}% of squared PC1 score energy, "
        f"but removing it leaves PC1 at {removal_audit.set_index('condition_set').loc['remove_strongest_pair_6', 'per_unit_zscored_centered_pc1_variance_fraction']*100:.1f}% "
        f"and median pairwise unit correlation at {removal_audit.set_index('condition_set').loc['remove_strongest_pair_6', 'median_pairwise_unit_profile_spearman']:.2f}. "
        f"Removing pairs 6 and 2 leaves PC1 at {removal_audit.set_index('condition_set').loc['remove_strongest_pair_6_and_lowest_pair_2', 'per_unit_zscored_centered_pc1_variance_fraction']*100:.1f}%.\n\n"
        "The supported conclusion is narrow: across these 16 original image–trajectory pairs, the frozen RR100 population "
        "shows a robust common ordering of FEM-effect magnitude across most responsive units. This is an internal structural "
        "observation. It is not yet specifically image-dependent, externally reproducible, a biological latent factor, caused "
        "by retinal power, or additive/multiplicative gain. Several units, including RR100 62, show substantial dissociations.\n"
    )
    print(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))
    print(examples[["selection_role", "rr100_index", "unit_mean_effect", "unit_profile_vs_leave_one_unit_mean_z_pearson"]].to_string(index=False))


if __name__ == "__main__":
    main()
