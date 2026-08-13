#!/usr/bin/env python3
"""Checkpoint 13A: map-first RR100 FEM modulation in the 384-window cache.

This is deliberately a visual checkpoint before population inference.  It
extracts the exact RR100 medoid channels from a production cache containing
each image window's paired empirical trace and eight matched traces from other
trials.  The response measure is the RMS represented by the first four
zero-mean temporal DCT coefficients of FEM-minus-static response.
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

from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.redundancy_resolved_v1_population import load_population_view


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_local_pairing_Iz_conditional_n384_k8_rel1_seed23_v1"
)
PILOT = ROOT / (
    "outputs/fig4_active_sensing/rr100_all16_spectral_explainability_checkpoint_11_v1/"
    "all16_original_pair_all_rr100_response_cache.npz"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_fullbank_map_checkpoint_13a_v1"
N_DCT = 4
EPS = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--pilot-cache", type=Path, default=PILOT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def fixed_dct(n_timepoints: int, n_components: int = N_DCT) -> np.ndarray:
    t = np.arange(n_timepoints, dtype=float)
    basis = []
    for k in range(1, n_components + 1):
        vec = np.cos(np.pi * (t + 0.5) * k / n_timepoints)
        vec -= vec.mean()
        vec /= np.sqrt(np.sum(vec * vec)) + EPS
        basis.append(vec)
    return np.column_stack(basis)


def rr100_columns() -> tuple[np.ndarray, str]:
    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    if view.membership is None or view.membership.shape != (100, 756):
        raise ValueError(f"Unexpected RR100 membership: {None if view.membership is None else view.membership.shape}")
    nonzero = np.count_nonzero(view.membership, axis=1)
    if not np.all(nonzero == 1) or not np.allclose(view.membership.sum(axis=1), 1):
        raise ValueError("This checkpoint requires the one-hot movie-medoid RR100 view")
    return np.argmax(view.membership, axis=1).astype(int), view.name


def dct_rms_from_flat(flat: np.ndarray, columns: np.ndarray, n_timepoints: int) -> np.ndarray:
    arr = np.asarray(flat, dtype=float)
    if arr.shape[-1] != N_DCT * 756:
        raise ValueError(f"Unexpected flattened DCT response shape {arr.shape}")
    shaped = arr.reshape(*arr.shape[:-1], N_DCT, 756)[..., columns]
    return np.sqrt(np.sum(shaped * shaped, axis=-2) / float(n_timepoints))


def load_bank(cache_dir: Path, columns: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, int]:
    metadata = pd.read_csv(cache_dir / "analysis_images.csv").sort_values("image_index").reset_index(drop=True)
    archive = np.load(cache_dir / "response_summary_arrays.npz")
    n_timepoints = int(archive["temporal_dct_basis"].shape[0])
    paired = dct_rms_from_flat(
        archive["temporal_dct_delta__actual_paired_empirical__rel_1x"], columns, n_timepoints
    )
    samples = np.stack(
        [
            dct_rms_from_flat(
                archive[f"temporal_dct_delta__matched_unpaired_empirical_sample{sample}__rel_1x"],
                columns,
                n_timepoints,
            )
            for sample in range(8)
        ],
        axis=1,
    )
    matched = samples.mean(axis=1)
    if paired.shape != (384, 100) or matched.shape != paired.shape or samples.shape != (384, 8, 100):
        raise ValueError(f"Unexpected bank shapes: paired={paired.shape}, matched={matched.shape}, samples={samples.shape}")
    if not np.array_equal(metadata["image_index"].to_numpy(), np.arange(384)):
        raise ValueError("analysis_images.csv is not aligned to cached response rows")
    return metadata, paired, matched, samples, n_timepoints


def validate_proxy(pilot_cache: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    archive = np.load(pilot_cache)
    rows = []
    variance_fractions = []
    for image_index in range(16):
        delta = (
            archive[f"image_{image_index:02d}_fem"][:, :, 0, 0]
            - archive[f"image_{image_index:02d}_zero"][:, :, 0, 0]
        ).astype(float)
        basis = fixed_dct(delta.shape[0])
        coefficients = basis.T @ delta
        proxy = np.sqrt(np.sum(coefficients * coefficients, axis=0) / delta.shape[0])
        exact = delta.std(axis=0)
        variance_fractions.extend(((proxy / np.maximum(exact, EPS)) ** 2).tolist())
        for unit in range(100):
            rows.append(
                {
                    "image_index": image_index,
                    "rr100_index": unit,
                    "exact_delta_temporal_sd_hz": exact[unit],
                    "four_dct_delta_rms_hz": proxy[unit],
                }
            )
    table = pd.DataFrame(rows)
    image_means = table.groupby("image_index")[["exact_delta_temporal_sd_hz", "four_dct_delta_rms_hz"]].mean()
    per_unit_rho = table.groupby("rr100_index").apply(
        lambda x: spearmanr(x["exact_delta_temporal_sd_hz"], x["four_dct_delta_rms_hz"]).statistic,
        include_groups=False,
    )
    summary = {
        "matrix_pearson_r": float(pearsonr(table["exact_delta_temporal_sd_hz"], table["four_dct_delta_rms_hz"]).statistic),
        "matrix_spearman_rho": float(spearmanr(table["exact_delta_temporal_sd_hz"], table["four_dct_delta_rms_hz"]).statistic),
        "image_mean_pearson_r": float(pearsonr(image_means.iloc[:, 0], image_means.iloc[:, 1]).statistic),
        "image_mean_spearman_rho": float(spearmanr(image_means.iloc[:, 0], image_means.iloc[:, 1]).statistic),
        "median_per_unit_spearman_rho": float(per_unit_rho.median()),
        "median_temporal_variance_fraction_captured": float(np.median(variance_fractions)),
    }
    return table, summary


def standardize_units(matrix: np.ndarray) -> np.ndarray:
    scale = matrix.std(axis=0)
    return (matrix - matrix.mean(axis=0)) / np.maximum(scale, EPS)


def select_windows(metadata: pd.DataFrame, paired: np.ndarray, matched: np.ndarray) -> pd.DataFrame:
    paired_mean = paired.mean(axis=1)
    matched_mean = matched.mean(axis=1)
    shared = 0.5 * (paired_mean + matched_mean)
    difference = paired_mean - matched_mean
    candidates = {
        "shared_high": np.argsort(shared)[::-1],
        "shared_low": np.argsort(shared),
        "paired_enhanced": np.argsort(difference)[::-1],
        "paired_suppressed": np.argsort(difference),
    }
    criterion = {
        "shared_high": "maximum mean of paired and matched population modulation",
        "shared_low": "minimum mean of paired and matched population modulation",
        "paired_enhanced": "maximum paired-minus-matched population modulation",
        "paired_suppressed": "minimum paired-minus-matched population modulation",
    }
    selected = []
    used: set[int] = set()
    for role, order in candidates.items():
        image_index = next(int(i) for i in order if int(i) not in used)
        used.add(image_index)
        row = metadata.loc[metadata["image_index"].eq(image_index)].iloc[0]
        selected.append(
            {
                "selection_role": role,
                "selection_criterion": criterion[role],
                "image_index": image_index,
                "source_row": int(row["source_row"]),
                "session": row["session"],
                "trial_idx": int(row["trial_idx"]),
                "phase": row["phase"],
                "paired_population_mean_hz": paired_mean[image_index],
                "matched_population_mean_hz": matched_mean[image_index],
                "paired_minus_matched_population_mean_hz": difference[image_index],
                "shared_population_mean_hz": shared[image_index],
                "actual_observed_rms_deg": float(row["actual_observed_rms_deg"]),
                "actual_path_length_deg": float(row["actual_path_length_deg"]),
            }
        )
    return pd.DataFrame(selected)


def make_metric_figure(validation: pd.DataFrame, summary: dict[str, float], out_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    ax = axes[0]
    x = validation["exact_delta_temporal_sd_hz"].to_numpy()
    y = validation["four_dct_delta_rms_hz"].to_numpy()
    ax.hexbin(x, y, gridsize=48, bins="log", mincnt=1, cmap="magma")
    lim = max(np.quantile(x, 0.995), np.quantile(y, 0.995))
    ax.plot([0, lim], [0, lim], color="white", lw=1.2, ls="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Exact temporal SD from full response trace (Hz)")
    ax.set_ylabel("RMS in first four temporal DCT components (Hz)")
    ax.set_title("A  Low-frequency proxy preserves unit × image ordering", loc="left", fontweight="bold")
    ax.text(
        0.03,
        0.96,
        f"Spearman ρ = {summary['matrix_spearman_rho']:.2f}\nPearson r = {summary['matrix_pearson_r']:.2f}",
        transform=ax.transAxes,
        va="top",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 4},
    )

    ax = axes[1]
    by_image = validation.groupby("image_index")[["exact_delta_temporal_sd_hz", "four_dct_delta_rms_hz"]].mean()
    ax.scatter(by_image.iloc[:, 0], by_image.iloc[:, 1], s=38, color="#2a9d8f", edgecolor="white", linewidth=0.5)
    for image_index, row in by_image.iterrows():
        ax.annotate(str(image_index), tuple(row), xytext=(3, 2), textcoords="offset points", fontsize=7)
    lim = max(by_image.to_numpy().max() * 1.08, EPS)
    ax.plot([0, lim], [0, lim], color="0.35", lw=1.0, ls="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Exact population-mean temporal SD (Hz)")
    ax.set_ylabel("Four-DCT population-mean RMS (Hz)")
    ax.set_title("B  Original 16 conditions retain the same ordering", loc="left", fontweight="bold")
    ax.text(
        0.03,
        0.96,
        f"Image-mean Spearman ρ = {summary['image_mean_spearman_rho']:.2f}\n"
        f"Median variance captured = {100 * summary['median_temporal_variance_fraction_captured']:.0f}%",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8", "pad": 4},
    )
    fig.suptitle(
        "Metric checkpoint: the cache measures low-frequency FEM modulation, not total temporal modulation",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_map_figure(
    metadata: pd.DataFrame,
    paired: np.ndarray,
    matched: np.ndarray,
    selected: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    shared_condition = 0.5 * (paired.mean(axis=1) + matched.mean(axis=1))
    order = np.argsort(shared_condition)[::-1]
    unit_order = np.argsort(0.5 * (paired.mean(axis=0) + matched.mean(axis=0)))[::-1]
    zp = standardize_units(paired)[order][:, unit_order]
    zu = standardize_units(matched)[order][:, unit_order]
    residual_scale = np.std(paired - matched, axis=0)
    zd = ((paired - matched) / np.maximum(residual_scale, EPS))[order][:, unit_order]

    fig = plt.figure(figsize=(17.5, 14.0))
    gs = fig.add_gridspec(
        3, 3, height_ratios=[1.45, 0.78, 1.05],
        left=0.055, right=0.94, bottom=0.055, top=0.91, hspace=0.28, wspace=0.20,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cmap = "coolwarm"
    for ax, matrix, title in zip(
        axes,
        [zp, zu, zd],
        [
            "A  Own eye trajectory",
            "B  Mean over 8 matched other-trial trajectories",
            "C  Own minus matched trajectories",
        ],
    ):
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-2.5, vmax=2.5)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("RR100 units (ordered by overall modulation)", labelpad=3)
        ax.set_ylabel("384 windows, ordered by shared modulation")
        ax.set_xticks([0, 24, 49, 74, 99])
        ax.set_xticklabels([0, 25, 50, 75, 100])
    cb = fig.colorbar(im, ax=axes, location="right", shrink=0.78, pad=0.015)
    cb.set_label("Within-unit standardized modulation")

    ax = fig.add_subplot(gs[1, :2])
    x = matched.mean(axis=1)
    y = paired.mean(axis=1)
    ax.scatter(x, y, s=19, alpha=0.55, color="#355070", edgecolors="none")
    lim = max(x.max(), y.max()) * 1.06
    ax.plot([0, lim], [0, lim], ls="--", color="0.45", lw=1)
    colors = {
        "shared_high": "#d62828",
        "shared_low": "#457b9d",
        "paired_enhanced": "#f77f00",
        "paired_suppressed": "#6a4c93",
    }
    for row in selected.itertuples():
        idx = int(row.image_index)
        ax.scatter(x[idx], y[idx], s=85, color=colors[row.selection_role], edgecolor="black", linewidth=0.6, zorder=4)
        ax.annotate(row.selection_role.replace("_", " "), (x[idx], y[idx]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Mean modulation under matched other-trial trajectories (Hz)")
    ax.set_ylabel("Mean modulation under own trajectory (Hz)")
    ax.set_title("D  Window-level modulation is shared across trajectory constructions", loc="left", fontweight="bold")
    ax.text(0.02, 0.94, f"Pearson r = {pearsonr(x, y).statistic:.2f}\nSpearman ρ = {spearmanr(x, y).statistic:.2f}", transform=ax.transAxes, va="top")

    ax = fig.add_subplot(gs[1, 2])
    differences = y - x
    ax.hist(differences, bins=35, color="#6c757d", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(np.median(differences), color="#d62828", lw=1.5, ls="--", label=f"median = {np.median(differences):.4f} Hz")
    ax.set_xlabel("Own minus matched population mean (Hz)")
    ax.set_ylabel("Windows")
    ax.set_title("E  Pairing-specific deviations are bidirectional", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    example_gs = gs[2, :].subgridspec(2, 2, hspace=0.48, wspace=0.22)
    for axis_index, row in enumerate(selected.itertuples()):
        ax = fig.add_subplot(example_gs[axis_index // 2, axis_index % 2])
        idx = int(row.image_index)
        ax.plot(np.arange(100), matched[idx], color="#457b9d", lw=1.2, label="matched other-trial mean")
        ax.plot(np.arange(100), paired[idx], color="#d62828", lw=1.2, label="own trajectory")
        ax.fill_between(np.arange(100), matched[idx], paired[idx], color="#adb5bd", alpha=0.25)
        ax.set_xlabel("RR100 unit")
        ax.set_ylabel("Low-frequency\nmodulation RMS (Hz)")
        ax.set_title(
            f"{chr(70 + axis_index)}  {row.selection_role.replace('_', ' ').title()} — "
            f"window {idx}, {row.session}, trial {row.trial_idx}",
            loc="left",
            fontsize=10,
            fontweight="bold",
        )
        if axis_index == 0:
            ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        "Full cached movie bank: shared window ordering is visible before fitting a latent model",
        y=0.975,
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    columns, view_name = rr100_columns()
    metadata, paired, matched, samples, n_timepoints = load_bank(args.cache_dir, columns)
    validation, validation_summary = validate_proxy(args.pilot_cache)
    selected = select_windows(metadata, paired, matched)

    trial_group = metadata["session"].astype(str) + "__trial" + metadata["trial_idx"].astype(str)
    window_summary = metadata.copy()
    window_summary["source_trial_group"] = trial_group
    window_summary["paired_population_mean_hz"] = paired.mean(axis=1)
    window_summary["matched_population_mean_hz"] = matched.mean(axis=1)
    window_summary["paired_minus_matched_population_mean_hz"] = paired.mean(axis=1) - matched.mean(axis=1)
    window_summary["matched_across_trace_sd_population_mean_hz"] = samples.std(axis=1).mean(axis=1)
    unit_summary = pd.DataFrame(
        {
            "rr100_index": np.arange(100),
            "canonical_channel": columns,
            "paired_mean_hz": paired.mean(axis=0),
            "matched_mean_hz": matched.mean(axis=0),
            "paired_minus_matched_mean_hz": (paired - matched).mean(axis=0),
            "paired_vs_matched_window_pearson_r": [pearsonr(paired[:, u], matched[:, u]).statistic for u in range(100)],
            "paired_vs_matched_window_spearman_rho": [spearmanr(paired[:, u], matched[:, u]).statistic for u in range(100)],
        }
    )

    validation.to_csv(args.out_dir / "pilot_four_dct_vs_exact_temporal_sd.csv", index=False)
    selected.to_csv(args.out_dir / "selected_windows.csv", index=False)
    window_summary.to_csv(args.out_dir / "window_effect_summary.csv", index=False)
    unit_summary.to_csv(args.out_dir / "rr100_unit_effect_summary.csv", index=False)
    np.savez_compressed(
        args.out_dir / "fullbank_rr100_effect_matrices.npz",
        paired_four_dct_rms_hz=paired.astype(np.float32),
        matched_unpaired_four_dct_rms_hz=matched.astype(np.float32),
        matched_unpaired_sample_four_dct_rms_hz=samples.astype(np.float32),
        rr100_canonical_channels=columns,
        image_index=metadata["image_index"].to_numpy(int),
    )
    make_metric_figure(validation, validation_summary, args.out_dir / "metric_validation.png", args.dpi)
    make_map_figure(metadata, paired, matched, selected, args.out_dir / "paired_vs_matched_maps.png", args.dpi)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_stage": "map-first visual checkpoint; no inferential population claim",
        "cache_dir": str(args.cache_dir.resolve()),
        "pilot_cache": str(args.pilot_cache.resolve()),
        "rr100_population_view": view_name,
        "rr100_canonical_channels": columns.tolist(),
        "n_windows": 384,
        "n_unique_source_trials": int(trial_group.nunique()),
        "n_sessions": int(metadata["session"].nunique()),
        "n_matched_other_trial_traces_per_window": 8,
        "n_timepoints": n_timepoints,
        "n_temporal_dct_components": N_DCT,
        "metric": "sqrt(sum(first four zero-mean temporal DCT coefficients of FEM-minus-static response squared) / n_timepoints)",
        "metric_limitation": "low-frequency temporal modulation proxy, not full-trace temporal SD",
        "validation_against_original_16": validation_summary,
        "paired_vs_matched_all_entries_pearson_r": float(pearsonr(paired.ravel(), matched.ravel()).statistic),
        "paired_vs_matched_window_population_mean_pearson_r": float(pearsonr(paired.mean(axis=1), matched.mean(axis=1)).statistic),
        "paired_vs_matched_window_population_mean_spearman_rho": float(spearmanr(paired.mean(axis=1), matched.mean(axis=1)).statistic),
        "paired_vs_matched_unit_grand_mean_pearson_r": float(pearsonr(paired.mean(axis=0), matched.mean(axis=0)).statistic),
        "selected_windows": selected.to_dict(orient="records"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# RR100 full-bank map checkpoint 13A\n\n"
        "This is a map-first visual checkpoint using the 384-window local-pairing cache. "
        "It does not yet make a population-level generalization claim.\n\n"
        "The response measure is the RMS represented by the first four zero-mean temporal DCT "
        "components of FEM-minus-static response. It is a low-frequency proxy, not the exact "
        "full-trace temporal SD used in checkpoint 12B. `metric_validation.png` calibrates that "
        "substitution on the original 16 conditions.\n\n"
        "`paired_vs_matched_maps.png` compares each image window's own empirical trajectory with "
        "the mean modulation magnitude under eight matched trajectories from other trials. "
        "Rows are windows, not independent experimental replicates: 384 windows come from 221 "
        "session-by-trial groups.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
