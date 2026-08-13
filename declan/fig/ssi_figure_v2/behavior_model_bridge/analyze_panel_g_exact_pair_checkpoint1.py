#!/usr/bin/env python3
"""Checkpoint 1 inspection of the full Panel G exact-pair production run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import _extract_patch
from declan.fig.ssi_figure_v2.behavior_confounds.build_checkpoint1_reference_frame_examples import axis_vector


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUN_DIR = ROOT / (
    "outputs/fig/ssi_figure_v2/behavior_model_bridge/"
    "panel_g_exact_pair_fig4_trace_bank_n1000_v1"
)
DEFAULT_SHARD_DIR = DEFAULT_RUN_DIR / "shards/pairs_000000_001000"
DEFAULT_SOURCE = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
DEFAULT_TRACE_XY = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/trace_xy.npy"
)
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "checkpoint1_production_readout"
COHERENCE_EDGES = (-np.inf, 0.2, 0.5, 0.8, np.inf)
COHERENCE_LABELS = ("0–0.2", "0.2–0.5", "0.5–0.8", "0.8–1")
POPULATION_COLORS = {
    "high_sf_aligned": "#7351a3",
    "high_sf_orthogonal": "#d47a25",
    "high_sf_all": "#2678a8",
    "low_sf_all": "#3b8a61",
}
POPULATION_LABELS = {
    "high_sf_aligned": "aligned high-SF",
    "high_sf_orthogonal": "orthogonal high-SF",
    "high_sf_all": "all high-SF",
    "low_sf_all": "all low-SF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--source-windows", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--trace-xy", type=Path, default=DEFAULT_TRACE_XY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _behavior_metrics(trace: np.ndarray, edge_axis_deg: float) -> dict[str, float]:
    centered = np.asarray(trace, dtype=np.float64) - np.mean(trace, axis=0, keepdims=True)
    parallel_axis = axis_vector(float(edge_axis_deg))
    normal_axis = np.asarray([-parallel_axis[1], parallel_axis[0]])
    parallel = centered @ parallel_axis
    normal = centered @ normal_axis
    parallel_rms = float(np.sqrt(np.mean(parallel**2)) * 60.0)
    normal_rms = float(np.sqrt(np.mean(normal**2)) * 60.0)
    covariance = np.cov(centered.T, ddof=0)
    eigenvalues = np.linalg.eigvalsh(covariance)
    anisotropy = float((eigenvalues[-1] - eigenvalues[0]) / max(eigenvalues.sum(), 1e-12))
    return {
        "parallel_rms_arcmin": parallel_rms,
        "normal_rms_arcmin": normal_rms,
        "parallel_minus_normal_rms_arcmin": parallel_rms - normal_rms,
        "trace_covariance_anisotropy": anisotropy,
    }


def _extended_aligned(contrasts: pd.DataFrame, cohort: pd.DataFrame, trace_xy: np.ndarray) -> pd.DataFrame:
    aligned = contrasts[contrasts["population"].astype(str).eq("high_sf_aligned")].copy()
    aligned = aligned.merge(cohort, on="pair_index", suffixes=("", "_cohort"), validate="one_to_one")
    metrics = [
        {"pair_index": int(row.pair_index), **_behavior_metrics(trace_xy[int(row.pair_index)], float(row.image_edge_axis_deg))}
        for row in aligned.itertuples(index=False)
    ]
    aligned = aligned.merge(pd.DataFrame(metrics), on="pair_index", validate="one_to_one")
    aligned["coherence_bin"] = pd.cut(
        aligned["image_orientation_coherence"], COHERENCE_EDGES,
        labels=COHERENCE_LABELS, right=False,
    )
    return aligned


def _select_pairs(aligned: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    metric = "real_minus_rotation_bits_per_spike"
    info = "real_minus_rotation_information_bits_per_sample"
    fraction = "fraction_rotations_below_real_bits_per_spike"
    pivot = contrasts.pivot(index="pair_index", columns="population", values=metric)
    rows: list[dict[str, object]] = []

    def add(role: str, criterion: str, pair_index: int) -> None:
        row = aligned[aligned["pair_index"].astype(int).eq(int(pair_index))].iloc[0]
        rows.append(
            {
                "selection_role": role,
                "selection_criterion": criterion,
                "pair_index": int(pair_index),
                "source_row": int(row["source_row"]),
                "session": str(row["session"]),
                "subject": str(row["subject"]),
                "trial_idx": int(row["trial_idx"]),
                "phase": str(row["phase"]),
                "image_orientation_coherence": float(row["image_orientation_coherence"]),
                "n_aligned_units": int(row["n_units"]),
                "aligned_bits_per_spike_effect": float(row[metric]),
                "aligned_information_per_sample_effect": float(row[info]),
                "aligned_expected_spikes_per_sample_effect": float(row["real_minus_rotation_expected_spikes_per_sample"]),
                "fraction_rotations_below_real_aligned_ssi": float(row[fraction]),
                "orthogonal_bits_per_spike_effect": float(pivot.loc[int(pair_index), "high_sf_orthogonal"]),
                "parallel_minus_normal_rms_arcmin": float(row["parallel_minus_normal_rms_arcmin"]),
            }
        )

    supported_positive = aligned[
        (aligned[info] > 0) & (aligned[fraction] >= 1.0)
    ].nlargest(1, metric).iloc[0]
    add(
        "information-supported positive",
        "largest aligned bits/spike gain among pairs with positive information/sample and real SSI above all 8 rotations",
        int(supported_positive["pair_index"]),
    )
    supported_negative = aligned[
        (aligned[info] < 0) & (aligned[fraction] <= 0.0)
    ].nsmallest(1, metric).iloc[0]
    add(
        "information-supported negative",
        "most negative aligned bits/spike effect among pairs with negative information/sample and real SSI below all 8 rotations",
        int(supported_negative["pair_index"]),
    )
    normalization = aligned[(aligned[metric] > 0) & (aligned[info] <= 0)].nlargest(1, metric).iloc[0]
    add(
        "normalization dissociation",
        "largest positive aligned bits/spike effect with nonpositive information/sample effect",
        int(normalization["pair_index"]),
    )
    opposing = pivot[(pivot["high_sf_aligned"] > 0) & (pivot["high_sf_orthogonal"] < 0)].copy()
    opposing["aligned_minus_orthogonal"] = opposing["high_sf_aligned"] - opposing["high_sf_orthogonal"]
    population_pair = int(opposing["aligned_minus_orthogonal"].idxmax())
    add(
        "population dissociation",
        "largest aligned-minus-orthogonal effect among pairs with aligned positive and orthogonal negative",
        population_pair,
    )
    high_coherence = aligned[aligned["image_orientation_coherence"] >= 0.8].copy()
    null_pair = int((high_coherence[metric].abs()).idxmin())
    add(
        "high-coherence near-null control",
        "smallest absolute aligned bits/spike effect among coherence >= 0.8 pairs",
        int(aligned.loc[null_pair, "pair_index"]),
    )
    high_positive = high_coherence[high_coherence[info] > 0].nlargest(1, metric).iloc[0]
    add(
        "high-coherence positive",
        "largest aligned bits/spike gain among coherence >= 0.8 pairs with positive information/sample",
        int(high_positive["pair_index"]),
    )
    selected = pd.DataFrame(rows)
    if selected["pair_index"].duplicated().any():
        raise RuntimeError("Selection roles unexpectedly resolved to duplicate pairs")
    return selected


def _plot_overview(aligned: pd.DataFrame, contrasts: pd.DataFrame, selected: pd.DataFrame, out_dir: Path) -> None:
    metric = "real_minus_rotation_bits_per_spike"
    info = "real_minus_rotation_information_bits_per_sample"
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))
    ax = axes[0, 0]
    ax.scatter(aligned["image_orientation_coherence"], aligned[metric], s=11, alpha=0.24, color="#7351a3")
    grouped = aligned.groupby("coherence_bin", observed=True)
    centers = np.asarray([0.1, 0.35, 0.65, 0.9])
    ax.plot(centers, grouped[metric].median().reindex(COHERENCE_LABELS), color="black", marker="o", label="bin median")
    ax.plot(centers, grouped[metric].mean().reindex(COHERENCE_LABELS), color="#c23b38", marker="s", linestyle="--", label="bin mean")
    ax.axhline(0, color="0.55", linewidth=0.8)
    ax.set_xlabel("local orientation coherence")
    ax.set_ylabel("real − rotation mean SSI (bits/spike)")
    ax.set_title("Aligned high-SF: raw pairs")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    display = contrasts.copy()
    low, high = display[metric].quantile([0.01, 0.99])
    bins = np.linspace(float(low), float(high), 70)
    for population, color in POPULATION_COLORS.items():
        values = display[display["population"].astype(str).eq(population)][metric]
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.6, color=color, label=POPULATION_LABELS[population])
    ax.axvline(0, color="0.55", linewidth=0.8)
    ax.set_xlabel("real − rotation mean SSI (bits/spike)\n1st–99th percentile display range")
    ax.set_ylabel("density")
    ax.set_title("Population controls")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    sign_match = np.sign(aligned[metric]) == np.sign(aligned[info])
    ax.scatter(aligned.loc[sign_match, metric], aligned.loc[sign_match, info], s=12, alpha=0.25, color="#2678a8", label="same sign")
    ax.scatter(aligned.loc[~sign_match, metric], aligned.loc[~sign_match, info], s=16, alpha=0.45, color="#d44a3a", label="sign mismatch")
    ax.axhline(0, color="0.55", linewidth=0.8)
    ax.axvline(0, color="0.55", linewidth=0.8)
    ax.set_xlabel("SSI effect (bits/spike)")
    ax.set_ylabel("information effect (bits/sample)")
    ax.set_title(f"Normalization diagnostic: {(~sign_match).mean():.1%} sign mismatch")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    scatter = ax.scatter(
        aligned["parallel_minus_normal_rms_arcmin"], aligned[metric],
        c=aligned["image_orientation_coherence"], cmap="viridis", s=13, alpha=0.45,
    )
    ax.axhline(0, color="0.55", linewidth=0.8)
    ax.axvline(0, color="0.55", linewidth=0.8)
    for row in selected.itertuples(index=False):
        point = aligned[aligned["pair_index"].astype(int).eq(int(row.pair_index))].iloc[0]
        ax.annotate(str(int(row.pair_index)), (point["parallel_minus_normal_rms_arcmin"], point[metric]), fontsize=7)
    ax.set_xlabel("parallel − normal positional RMS (arcmin)")
    ax.set_ylabel("real − rotation mean SSI (bits/spike)")
    ax.set_title("Behavioral alignment versus direct SSI effect")
    fig.colorbar(scatter, ax=ax, label="orientation coherence", fraction=0.046)

    fig.suptitle(
        "Panel G production checkpoint: 1,000 native image–trajectory pairs\n"
        "descriptive pair distributions; no dose-curve interpolation",
        fontsize=13, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "checkpoint1_production_overview.png", dpi=220)
    fig.savefig(out_dir / "checkpoint1_production_overview.pdf")
    plt.close(fig)


def _plot_selected(
    selected: pd.DataFrame,
    population_metrics: pd.DataFrame,
    cohort: pd.DataFrame,
    source: pd.DataFrame,
    trace_xy: np.ndarray,
    out_dir: Path,
) -> None:
    source_by_id = source.set_index(source["source_row"].astype(int), drop=False)
    cohort_by_pair = cohort.set_index("pair_index", drop=False)
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    fig, axes = plt.subplots(len(selected), 4, figsize=(13.8, 2.45 * len(selected)), squeeze=False)
    for row_index, selection in enumerate(selected.itertuples(index=False)):
        pair_index = int(selection.pair_index)
        pair = cohort_by_pair.loc[pair_index]
        source_row = source_by_id.loc[int(pair.source_row)]
        patch, patch_meta = _extract_patch(source_row, canvas_cache=canvas_cache, patch_size_px=540)
        trace = np.asarray(trace_xy[pair_index], dtype=float)
        centered = trace - np.mean(trace, axis=0, keepdims=True)
        edge_axis = float(pair.image_edge_axis_deg)
        parallel_axis = axis_vector(edge_axis)
        normal_axis = np.asarray([-parallel_axis[1], parallel_axis[0]])
        # Stored image axes and eye traces are in gaze coordinates (+y up),
        # whereas the displayed image array uses +row down. Reflect y only for
        # the pixel overlay; contour-coordinate projections remain in gaze space.
        screen_edge_axis = axis_vector(-edge_axis)

        ax = axes[row_index, 0]
        radius = 55
        center = np.asarray(patch.shape[::-1], dtype=float) / 2.0
        x0, y0 = int(center[0] - radius), int(center[1] - radius)
        ax.imshow(patch[y0 : y0 + 2 * radius, x0 : x0 + 2 * radius], cmap="gray", origin="upper")
        xy = centered * float(patch_meta["patch_ppd"])
        xy[:, 1] *= -1.0
        ax.plot(xy[:, 0] + radius, xy[:, 1] + radius, color="#00e5ff", linewidth=1.5)
        ax.scatter([xy[0, 0] + radius], [xy[0, 1] + radius], color="#ffe75b", s=16, zorder=3)
        axis_length = 34
        ax.arrow(radius, radius, axis_length * screen_edge_axis[0], axis_length * screen_edge_axis[1], color="#ff4b4b", width=0.8, head_width=4, length_includes_head=True)
        ax.set_xlim(0, 2 * radius)
        ax.set_ylim(2 * radius, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"{selection.selection_role}\npair {pair_index}", fontsize=8, weight="bold")
        if row_index == 0:
            ax.set_title("native patch + trace\nred: local contour axis (screen coordinates)", fontsize=9)

        ax = axes[row_index, 1]
        t = np.arange(len(trace)) / 120.0
        ax.plot(t, centered @ parallel_axis * 60.0, color="#d84a3f", label="parallel")
        ax.plot(t, centered @ normal_axis * 60.0, color="#3169a8", label="normal")
        ax.axhline(0, color="0.7", linewidth=0.6)
        ax.set_ylabel("position (arcmin)", fontsize=8)
        if row_index == len(selected) - 1:
            ax.set_xlabel("time (s)")
        if row_index == 0:
            ax.set_title("contour-coordinate trajectory", fontsize=9)
            ax.legend(frameon=False, fontsize=7)

        ax = axes[row_index, 2]
        pair_values = population_metrics[population_metrics["pair_index"].astype(int).eq(pair_index)]
        for population, linestyle in (("high_sf_aligned", "-"), ("high_sf_orthogonal", "--")):
            group = pair_values[pair_values["population"].astype(str).eq(population)]
            real = float(group[group["condition_kind"].astype(str).eq("real")]["bits_per_spike"].iloc[0])
            rotations = group[group["condition_kind"].astype(str).eq("rotation")].sort_values("rotation_angle_deg")
            ax.plot(rotations["rotation_angle_deg"], rotations["bits_per_spike"], marker="o", markersize=3, linewidth=1.2, linestyle=linestyle, color=POPULATION_COLORS[population], label=POPULATION_LABELS[population])
            ax.axhline(real, color=POPULATION_COLORS[population], linestyle=":", linewidth=1.0)
        ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_ylabel("direct SSI (bits/spike)", fontsize=8)
        if row_index == len(selected) - 1:
            ax.set_xlabel("trajectory rotation (deg)")
        if row_index == 0:
            ax.set_title("fresh angle evaluations\ndotted: recorded condition", fontsize=9)
            ax.legend(frameon=False, fontsize=7)

        ax = axes[row_index, 3]
        ax.axis("off")
        text = (
            f"coherence = {selection.image_orientation_coherence:.3f}\n"
            f"aligned units = {selection.n_aligned_units}\n"
            f"parallel − normal RMS = {selection.parallel_minus_normal_rms_arcmin:+.2f} arcmin\n\n"
            f"aligned ΔSSI = {selection.aligned_bits_per_spike_effect:+.5f} bits/sp\n"
            f"aligned Δinfo = {selection.aligned_information_per_sample_effect:+.2e} bits/sample\n"
            f"aligned Δspikes = {selection.aligned_expected_spikes_per_sample_effect:+.2e}/sample\n"
            f"orthogonal ΔSSI = {selection.orthogonal_bits_per_spike_effect:+.5f}\n"
            f"real > {selection.fraction_rotations_below_real_aligned_ssi:.0%} of rotations"
        )
        ax.text(0.02, 0.94, text, va="top", fontsize=8.2, linespacing=1.35)
        if row_index == 0:
            ax.set_title("pair-level audit values", fontsize=9)
    fig.suptitle(
        "Auditable production examples selected by predeclared response roles\n"
        "input and angle curves only; activation-map rerender is the next checkpoint",
        fontsize=12, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_dir / "checkpoint1_selected_pair_inputs_and_angle_curves.png", dpi=220)
    fig.savefig(out_dir / "checkpoint1_selected_pair_inputs_and_angle_curves.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    contrasts = pd.read_csv(args.shard_dir / "direct_pair_rotation_contrasts.csv")
    population_metrics = pd.read_csv(args.shard_dir / "direct_pair_population_metrics.csv")
    cohort = pd.read_csv(args.run_dir / "exact_pair_cohort_manifest.csv")
    trace_xy = np.load(args.trace_xy)
    source = pd.read_csv(args.source_windows)
    if "source_row" not in source.columns:
        source = source.copy()
        source["source_row"] = np.arange(len(source), dtype=int)
    aligned = _extended_aligned(contrasts, cohort, trace_xy)
    selected = _select_pairs(aligned, contrasts)
    aligned.to_csv(out_dir / "checkpoint1_aligned_pair_metrics.csv", index=False)
    selected.to_csv(out_dir / "checkpoint1_selected_pairs.csv", index=False)
    _plot_overview(aligned, contrasts, selected, out_dir)
    _plot_selected(selected, population_metrics, cohort, source, trace_xy, out_dir)
    metric = "real_minus_rotation_bits_per_spike"
    summary = {
        "analysis": "panel_g_exact_pair_production_checkpoint1",
        "artifact_type": "descriptive_pair_distribution_and_auditable_example_selection",
        "n_pairs": int(aligned["pair_index"].nunique()),
        "population_inference_performed": False,
        "aligned_effect": {
            "mean_bits_per_spike": float(aligned[metric].mean()),
            "median_bits_per_spike": float(aligned[metric].median()),
            "fraction_positive": float((aligned[metric] > 0).mean()),
            "pearson_coherence": float(aligned["image_orientation_coherence"].corr(aligned[metric])),
            "spearman_coherence": float(aligned["image_orientation_coherence"].corr(aligned[metric], method="spearman")),
            "bits_information_sign_mismatch_fraction": float(
                np.mean(np.sign(aligned[metric]) != np.sign(aligned["real_minus_rotation_information_bits_per_sample"]))
            ),
            "pearson_behavior_alignment": float(aligned["parallel_minus_normal_rms_arcmin"].corr(aligned[metric])),
        },
        "selection_roles": selected[["selection_role", "pair_index", "selection_criterion"]].to_dict("records"),
        "next_checkpoint": "fresh targeted activation-map rerender for the six frozen pair roles",
    }
    (out_dir / "checkpoint1_metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["aligned_effect"], indent=2))
    print(selected[["selection_role", "pair_index", "aligned_bits_per_spike_effect", "aligned_information_per_sample_effect"]].to_string(index=False))
    print(f"[panel-g-checkpoint1] wrote {out_dir}")


if __name__ == "__main__":
    main()
