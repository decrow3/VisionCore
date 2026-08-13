#!/usr/bin/env python3
"""Checkpoint 12A: is all-16 FEM spectral variation mostly amplitude-like?

This is an input-only checkpoint. It uses the exact supported SF x TF power
maps saved by checkpoint 11 and does not read any neural response or unit-tuning
data. Raw maps, total-normalized shapes, common-template residuals, and rank
diagnostics are saved before any gain interpretation is attempted.
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
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs/fig4_active_sensing/rr100_all16_spectral_explainability_checkpoint_11_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_retinal_spectral_rank_checkpoint_12a_v1"
EPS = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_maps(input_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    archive = np.load(input_dir / "all16_original_pair_supported_sf_tf_power.npz")
    sf = archive["sf_centers_cpd"].astype(float)
    tf = archive["tf_centers_hz"].astype(float)
    maps = np.stack([archive[f"image_{idx:02d}_supported_power_sf_tf"].astype(float) for idx in range(16)])
    images = pd.read_csv(input_dir / "all16_original_pair_image_contract.csv").sort_values("image_index")
    if maps.shape != (16, len(sf), len(tf)) or not np.all(maps > 0):
        raise ValueError(f"Unexpected power maps: shape={maps.shape}, min={maps.min()}")
    return maps, sf, tf, images


def rank_metrics(matrix: np.ndarray, prefix: str) -> dict[str, float]:
    singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    centered_singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    return {
        f"{prefix}_uncentered_rank1_energy_fraction": float(singular[0] ** 2 / np.sum(singular**2)),
        f"{prefix}_uncentered_rank2_cumulative_energy_fraction": float(np.sum(singular[:2] ** 2) / np.sum(singular**2)),
        f"{prefix}_centered_pc1_variance_fraction": float(centered_singular[0] ** 2 / np.sum(centered_singular**2)),
        f"{prefix}_centered_pc2_cumulative_variance_fraction": float(np.sum(centered_singular[:2] ** 2) / np.sum(centered_singular**2)),
    }


def analyze_maps(maps: np.ndarray, images: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, np.ndarray]]:
    flat = maps.reshape(len(maps), -1)
    totals = flat.sum(axis=1)
    normalized = flat / totals[:, None]
    template = normalized.mean(axis=0)
    template /= template.sum()
    amplitude_reconstruction = totals[:, None] * template[None, :]
    normalized_residual = normalized - template[None, :]
    log2_ratio = np.log2(np.maximum(normalized, EPS) / np.maximum(template[None, :], EPS))

    u, singular, vh = np.linalg.svd(flat, full_matrices=False)
    best_rank1 = singular[0] * np.outer(u[:, 0], vh[0])
    # SVD signs are arbitrary. Orient the spatial vector to have positive sum so
    # the corresponding image score reads as a positive amplitude.
    sign = 1.0 if float(vh[0].sum()) >= 0 else -1.0
    rank1_scores = sign * singular[0] * u[:, 0]
    rank1_template = sign * vh[0]
    rank1_template = rank1_template / max(rank1_template.sum(), EPS)

    rows = []
    for position, (_, image) in enumerate(images.iterrows()):
        q = normalized[position]
        cosine = float(np.dot(q, template) / (np.linalg.norm(q) * np.linalg.norm(template)))
        js = float(jensenshannon(q, template, base=2.0))
        tv = float(0.5 * np.sum(np.abs(q - template)))
        raw_relative_l2 = float(np.linalg.norm(flat[position] - amplitude_reconstruction[position]) / np.linalg.norm(flat[position]))
        rank1_relative_l2 = float(np.linalg.norm(flat[position] - best_rank1[position]) / np.linalg.norm(flat[position]))
        rows.append({
            "image_index": int(image["image_index"]), "source_row": int(image["source_row"]), "session": str(image["session"]),
            "total_supported_dynamic_power": float(totals[position]),
            "sqrt_total_supported_dynamic_power": float(np.sqrt(totals[position])),
            "best_rank1_amplitude_score": float(rank1_scores[position]),
            "normalized_shape_cosine_similarity_to_mean_template": cosine,
            "normalized_shape_jensen_shannon_distance_bits_sqrt": js,
            "normalized_shape_total_variation_distance": tv,
            "amplitude_only_relative_l2_residual": raw_relative_l2,
            "best_svd_rank1_relative_l2_residual": rank1_relative_l2,
            "normalized_log2_ratio_rms_unweighted": float(np.sqrt(np.mean(log2_ratio[position] ** 2))),
        })
    image_metrics = pd.DataFrame(rows)

    pair_rows = []
    for first in range(16):
        for second in range(first + 1, 16):
            a, b = normalized[first], normalized[second]
            pair_rows.append({
                "image_index_a": first, "image_index_b": second,
                "normalized_shape_cosine_similarity": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
                "normalized_shape_jensen_shannon_distance_bits_sqrt": float(jensenshannon(a, b, base=2.0)),
                "normalized_shape_total_variation_distance": float(0.5 * np.sum(np.abs(a - b))),
            })
    pairwise = pd.DataFrame(pair_rows)

    raw_metrics = rank_metrics(flat, "raw_linear_power_unequal_image_weight")
    shape_metrics = rank_metrics(normalized, "total_normalized_linear_shape_equal_image_weight")
    sqrt_metrics = rank_metrics(np.sqrt(flat), "sqrt_power_amplitude_space_unequal_image_weight")
    summary: dict[str, object] = {
        **raw_metrics, **shape_metrics, **sqrt_metrics,
        "amplitude_template_global_fraction_squared_error": float(
            np.sum((flat - amplitude_reconstruction) ** 2) / np.sum(flat**2)
        ),
        "best_svd_rank1_global_fraction_squared_error": float(np.sum((flat - best_rank1) ** 2) / np.sum(flat**2)),
        "spearman_total_power_vs_best_rank1_score": float(spearmanr(totals, rank1_scores).statistic),
        "pearson_total_power_vs_best_rank1_score": float(pearsonr(totals, rank1_scores).statistic),
        "median_pairwise_normalized_shape_cosine_similarity": float(pairwise["normalized_shape_cosine_similarity"].median()),
        "minimum_pairwise_normalized_shape_cosine_similarity": float(pairwise["normalized_shape_cosine_similarity"].min()),
        "median_pairwise_normalized_shape_jensen_shannon_distance_bits_sqrt": float(pairwise["normalized_shape_jensen_shannon_distance_bits_sqrt"].median()),
        "maximum_pairwise_normalized_shape_jensen_shannon_distance_bits_sqrt": float(pairwise["normalized_shape_jensen_shannon_distance_bits_sqrt"].max()),
        "total_power_max_to_min_ratio": float(totals.max() / totals.min()),
        "total_power_median": float(np.median(totals)),
    }
    arrays = {
        "normalized_maps": normalized.reshape(maps.shape),
        "mean_normalized_template": template.reshape(maps.shape[1:]),
        "normalized_log2_ratio_to_template": log2_ratio.reshape(maps.shape),
        "amplitude_only_reconstruction": amplitude_reconstruction.reshape(maps.shape),
        "best_svd_rank1_reconstruction": best_rank1.reshape(maps.shape),
        "best_svd_rank1_normalized_template": rank1_template.reshape(maps.shape[1:]),
        "raw_singular_values": singular,
    }
    return image_metrics, pairwise, summary, arrays


def select_examples(metrics: pd.DataFrame) -> pd.DataFrame:
    definitions = (
        ("lowest_total_power", metrics["total_supported_dynamic_power"].idxmin(), "minimum total supported dynamic power"),
        ("median_total_power", (np.log10(metrics["total_supported_dynamic_power"]) - np.log10(metrics["total_supported_dynamic_power"]).median()).abs().idxmin(), "closest to median log10 total power"),
        ("highest_total_power", metrics["total_supported_dynamic_power"].idxmax(), "maximum total supported dynamic power"),
        ("most_common_spectral_shape", metrics["normalized_shape_jensen_shannon_distance_bits_sqrt"].idxmin(), "minimum Jensen-Shannon distance from mean normalized template"),
        ("most_distinct_spectral_shape", metrics["normalized_shape_jensen_shannon_distance_bits_sqrt"].idxmax(), "maximum Jensen-Shannon distance from mean normalized template"),
    )
    rows = []
    for order, (role, index, criterion) in enumerate(definitions, start=1):
        row = metrics.loc[index]
        rows.append({"display_order": order, "selection_role": role, "selection_criterion": criterion, **row.to_dict()})
    return pd.DataFrame(rows)


def format_power(value: float) -> str:
    return f"{value:.1e}"


def plot_absolute_maps(out: Path, maps: np.ndarray, sf: np.ndarray, tf: np.ndarray, metrics: pd.DataFrame, dpi: int) -> None:
    positive = maps[maps > 0]
    norm = LogNorm(vmin=float(np.quantile(positive, 0.01)), vmax=float(np.quantile(positive, 0.995)))
    fig, axes = plt.subplots(4, 4, figsize=(14.5, 12.0), constrained_layout=True, sharex=True, sharey=True)
    for image_index, axis in enumerate(axes.ravel()):
        im = axis.pcolormesh(sf, tf, maps[image_index].T, cmap="turbo", norm=norm, shading="auto")
        total = metrics.set_index("image_index").loc[image_index, "total_supported_dynamic_power"]
        axis.set_title(f"image {image_index} · total={format_power(total)}", fontsize=10)
        axis.set_xscale("log"); axis.set_yscale("log")
        if image_index // 4 == 3: axis.set_xlabel("SF (cycles/deg)")
        if image_index % 4 == 0: axis.set_ylabel("TF (Hz)")
    fig.colorbar(im, ax=axes, label="dynamic power (a.u.; one shared log scale)", shrink=0.82)
    fig.suptitle(
        "Checkpoint 12A input maps: FEM-created SF–TF power varies enormously across the 16 original image–trajectory pairs\n"
        "No neural responses or unit tuning are used",
        fontsize=14, weight="bold",
    )
    fig.savefig(out.with_suffix(".png"), dpi=dpi); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def plot_normalized_maps(
    out: Path, arrays: dict[str, np.ndarray], sf: np.ndarray, tf: np.ndarray,
    metrics: pd.DataFrame, dpi: int,
) -> None:
    normalized = arrays["normalized_maps"]
    positive = normalized[normalized > 0]
    norm = LogNorm(vmin=float(np.quantile(positive, 0.01)), vmax=float(np.quantile(positive, 0.995)))
    fig, axes = plt.subplots(4, 4, figsize=(14.5, 12.0), constrained_layout=True, sharex=True, sharey=True)
    indexed = metrics.set_index("image_index")
    for image_index, axis in enumerate(axes.ravel()):
        im = axis.pcolormesh(sf, tf, normalized[image_index].T, cmap="turbo", norm=norm, shading="auto")
        js = indexed.loc[image_index, "normalized_shape_jensen_shannon_distance_bits_sqrt"]
        cosine = indexed.loc[image_index, "normalized_shape_cosine_similarity_to_mean_template"]
        axis.set_title(f"image {image_index} · shape JS={js:.2f}, cos={cosine:.2f}", fontsize=9.2)
        axis.set_xscale("log"); axis.set_yscale("log")
        if image_index // 4 == 3: axis.set_xlabel("SF (cycles/deg)")
        if image_index % 4 == 0: axis.set_ylabel("TF (Hz)")
    fig.colorbar(im, ax=axes, label="fraction of each image’s supported power (shared log scale)", shrink=0.82)
    fig.suptitle(
        "After removing total power, the FEM spectra retain visibly different SF–TF shapes\n"
        "Each map sums to one; remaining differences cannot be explained by a scalar amplitude",
        fontsize=14, weight="bold",
    )
    fig.savefig(out.with_suffix(".png"), dpi=dpi); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def plot_template_residuals(
    out: Path, arrays: dict[str, np.ndarray], sf: np.ndarray, tf: np.ndarray,
    metrics: pd.DataFrame, examples: pd.DataFrame, summary: dict[str, object], dpi: int,
) -> None:
    template = arrays["mean_normalized_template"]
    ratios = arrays["normalized_log2_ratio_to_template"]
    roles = ["most_common_spectral_shape", "median_total_power", "most_distinct_spectral_shape"]
    selected = examples.set_index("selection_role").loc[roles]
    indices = selected["image_index"].astype(int).tolist()
    fig, axes = plt.subplots(1, 5, figsize=(17.0, 5.2), constrained_layout=True)
    ax = axes[0]
    positive = template[template > 0]
    im = ax.pcolormesh(sf, tf, template.T, cmap="turbo", norm=LogNorm(positive.min(), positive.max()), shading="auto")
    ax.set(xscale="log", yscale="log", xlabel="SF (cycles/deg)", ylabel="TF (Hz)", title="A  Mean normalized\nspectral template")
    fig.colorbar(im, ax=ax, label="power fraction")
    norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    residual_axes = []
    for row, image_index in enumerate(indices):
        ax = axes[1 + row]
        residual_axes.append(ax)
        im2 = ax.pcolormesh(sf, tf, np.clip(ratios[image_index].T, -4, 4), cmap="coolwarm", norm=norm, shading="auto")
        role = roles[row].replace("_spectral_shape", " shape").replace("_", " ")
        js = metrics.set_index("image_index").loc[image_index, "normalized_shape_jensen_shannon_distance_bits_sqrt"]
        ax.set(xscale="log", yscale="log", xlabel="SF (cycles/deg)", ylabel="TF (Hz)", title=f"{chr(66+row)}  image {image_index}\n{role} · JS={js:.2f}")
    fig.colorbar(im2, ax=residual_axes, label="log₂(normalized image power / template), clipped ±4", shrink=0.78)
    text_axis = axes[4]; text_axis.axis("off")
    text_axis.text(0.02, 0.94, "Amplitude-only hypothesis", fontsize=13, weight="bold", va="top")
    text_axis.text(0.02, 0.76, "Pᵢ(SF,TF) ≈ Aᵢ × common template", fontsize=13, va="top")
    text_axis.text(0.02, 0.56, f"Equal-image normalized rank-1 energy: {summary['total_normalized_linear_shape_equal_image_weight_uncentered_rank1_energy_fraction']*100:.1f}%", fontsize=12)
    text_axis.text(0.02, 0.42, f"Median pairwise shape cosine: {summary['median_pairwise_normalized_shape_cosine_similarity']:.2f}", fontsize=12)
    text_axis.text(0.02, 0.28, f"Worst pairwise shape cosine: {summary['minimum_pairwise_normalized_shape_cosine_similarity']:.2f}", fontsize=12)
    text_axis.text(0.02, 0.10, "Residual maps show where the scalar approximation fails.", fontsize=11, color="#555555")
    fig.suptitle("Checkpoint 12A common-template audit: amplitude is important, but spectral shape is not fixed", fontsize=14, weight="bold")
    fig.savefig(out.with_suffix(".png"), dpi=dpi); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def plot_rank_diagnostics(out: Path, maps: np.ndarray, arrays: dict[str, np.ndarray], metrics: pd.DataFrame, summary: dict[str, object], dpi: int) -> None:
    flat = maps.reshape(16, -1)
    normalized = arrays["normalized_maps"].reshape(16, -1)
    raw_s = np.linalg.svd(flat, full_matrices=False, compute_uv=False)
    shape_s = np.linalg.svd(normalized, full_matrices=False, compute_uv=False)
    raw_cumulative = np.cumsum(raw_s**2) / np.sum(raw_s**2)
    shape_cumulative = np.cumsum(shape_s**2) / np.sum(shape_s**2)
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), constrained_layout=True)
    axes[0].scatter(metrics["total_supported_dynamic_power"], metrics["best_rank1_amplitude_score"], c=metrics["image_index"], cmap="turbo", s=55)
    for _, row in metrics.iterrows(): axes[0].annotate(str(int(row["image_index"])), (row["total_supported_dynamic_power"], row["best_rank1_amplitude_score"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[0].set(xscale="log", yscale="log", xlabel="total supported dynamic power", ylabel="best rank-1 amplitude score", title=f"A  Raw rank-1 score mostly tracks total power\nSpearman ρ={summary['spearman_total_power_vs_best_rank1_score']:.2f}")
    ranks = np.arange(1, 17)
    axes[1].plot(ranks, raw_cumulative, marker="o", label="raw power (amplitude weighted)")
    axes[1].plot(ranks, shape_cumulative, marker="o", label="each image normalized to total=1")
    axes[1].axhline(0.9, color="0.5", ls="--", lw=1)
    axes[1].set(xlabel="number of uncentered SVD components", ylabel="cumulative energy fraction", ylim=(0, 1.02), title="B  Rank depends on whether high-power images dominate")
    axes[1].legend(frameon=False, fontsize=8); axes[1].grid(color="0.92")
    axes[2].scatter(metrics["total_supported_dynamic_power"], metrics["normalized_shape_jensen_shannon_distance_bits_sqrt"], c=metrics["image_index"], cmap="turbo", s=55)
    for _, row in metrics.iterrows(): axes[2].annotate(str(int(row["image_index"])), (row["total_supported_dynamic_power"], row["normalized_shape_jensen_shannon_distance_bits_sqrt"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
    rho = spearmanr(metrics["total_supported_dynamic_power"], metrics["normalized_shape_jensen_shannon_distance_bits_sqrt"]).statistic
    axes[2].set(xscale="log", xlabel="total supported dynamic power", ylabel="shape distance from mean template", title=f"C  Amplitude and shape are distinct properties\nSpearman ρ={rho:+.2f}")
    axes[2].grid(color="0.92")
    fig.suptitle("Checkpoint 12A quantitative audit: do not infer a common spectrum from raw rank alone", fontsize=14, weight="bold")
    fig.savefig(out.with_suffix(".png"), dpi=dpi); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    maps, sf, tf, images = load_maps(args.input_dir)
    metrics, pairwise, summary, arrays = analyze_maps(maps, images)
    examples = select_examples(metrics)
    metrics.to_csv(args.out_dir / "per_image_retinal_spectral_amplitude_and_shape_metrics.csv", index=False)
    pairwise.to_csv(args.out_dir / "pairwise_normalized_spectral_shape_distances.csv", index=False)
    examples.to_csv(args.out_dir / "auditable_retinal_spectral_example_selection.csv", index=False)
    np.savez_compressed(args.out_dir / "retinal_spectral_template_and_residual_maps.npz", sf_centers_cpd=sf, tf_centers_hz=tf, **arrays)
    plot_absolute_maps(args.out_dir / "checkpoint_12a_absolute_sftf_power_all16", maps, sf, tf, metrics, args.dpi)
    plot_normalized_maps(args.out_dir / "checkpoint_12a_total_normalized_spectral_shapes_all16", arrays, sf, tf, metrics, args.dpi)
    plot_template_residuals(args.out_dir / "checkpoint_12a_common_template_and_shape_residuals", arrays, sf, tf, metrics, examples, summary, args.dpi)
    plot_rank_diagnostics(args.out_dir / "checkpoint_12a_rank_and_shape_diagnostics", maps, arrays, metrics, summary, args.dpi)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "retinal-input-only audit of amplitude-like versus shape-changing variation in all-16 FEM SFxTF maps",
        "status": "checkpoint_12a_complete_stop_before_neural_analysis",
        "input_contract": "16 original image/measured-eye-trajectory pairs; positive TF and fitted SF/TF support from checkpoint 11",
        "neural_data_used": False, "unit_tuning_used": False,
        "primary_hypothesis": "P_i(SF,TF) is approximately an image-specific scalar amplitude times a common spectral template",
        "primary_shape_test": "uncentered SVD of total-normalized linear power maps, giving every image equal total weight",
        "warning": "raw-power SVD is amplitude weighted and can look low-rank because the highest-power images dominate",
        "summary": summary,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Checkpoint 12A: retinal spectral amplitude versus shape\n\n"
        "This input-only checkpoint asks whether the 16 exact FEM-created SF×TF power maps are adequately described by "
        "an image-specific scalar times one common spectral template. Raw-power SVD is reported but not treated as the "
        "primary shape test because high-power images dominate it. The primary equal-image test first normalizes every map "
        "to total power one and then measures rank, cosine similarity, Jensen–Shannon distance, and direct residual maps. "
        "No neural responses or unit tuning are used. The workflow stops here before testing neural low rank or additive "
        "versus multiplicative models.\n\n"
        f"Raw linear power is {summary['raw_linear_power_unequal_image_weight_uncentered_rank1_energy_fraction']*100:.1f}% rank-1, but total power spans "
        f"{summary['total_power_max_to_min_ratio']:.0f}-fold and therefore lets high-power images dominate. After normalizing every image to equal total "
        f"power, rank-1 energy falls to {summary['total_normalized_linear_shape_equal_image_weight_uncentered_rank1_energy_fraction']*100:.1f}%. The median "
        f"pairwise normalized-shape cosine is {summary['median_pairwise_normalized_shape_cosine_similarity']:.2f}, and the worst is "
        f"{summary['minimum_pairwise_normalized_shape_cosine_similarity']:.2f}. Thus FEM-created power has a very large image-dependent amplitude component, "
        "but the retinal spectra are not adequately described as exact scalar multiples of one fixed SF×TF template. This does not invalidate a shared "
        "neural drive or gain-like story; it prevents using raw spectral rank alone as its mechanism.\n"
    )
    print(json.dumps(summary, indent=2))
    print(examples[["selection_role", "image_index", "total_supported_dynamic_power", "normalized_shape_jensen_shannon_distance_bits_sqrt"]].to_string(index=False))


if __name__ == "__main__":
    main()
