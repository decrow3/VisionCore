#!/usr/bin/env python3
"""Directional audit of corrected balanced round 0 against the legacy 100x1000 cache."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/assembled/rounds_000_000_n001"
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
LEGACY = ROOT / "outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
ASSIGNMENTS = ROOT / "outputs/fig4_active_sensing/backimage_real_trace_sf_halves_recorded_validated_r0p5_v1/sf_half_recorded_validated_unit_assignments.csv"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_corrected_round0_vs_legacy_directional_checkpoint_38_v1"
SEED = 20260813


def population_ssi(info: np.ndarray, spikes: np.ndarray, units: np.ndarray) -> np.ndarray:
    return info[:, units].sum(1) / np.maximum(spikes[:, units].sum(1), 1e-12)


def within_image_slope(x: np.ndarray, y: np.ndarray, image: np.ndarray) -> float:
    xr = x - pd.Series(x).groupby(image).transform("mean").to_numpy()
    yr = y - pd.Series(y).groupby(image).transform("mean").to_numpy()
    return float(np.dot(xr, yr) / np.dot(xr, xr))


def image_cluster_bootstrap(
    x: np.ndarray, y: np.ndarray, image: np.ndarray, *, n_boot: int = 2000
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(SEED)
    images = np.unique(image)
    values = np.empty(n_boot, dtype=float)
    for iteration in range(n_boot):
        sampled = rng.choice(images, len(images), replace=True)
        numerator = 0.0
        denominator = 0.0
        for identity in sampled:
            use = image == identity
            xx = x[use] - x[use].mean()
            yy = y[use] - y[use].mean()
            numerator += float(np.dot(xx, yy))
            denominator += float(np.dot(xx, xx))
        values[iteration] = numerator / denominator
    low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
    return float(low), float(median), float(high), float(np.mean(values > 0))


def binned_curve(x: np.ndarray, y: np.ndarray, bins: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = pd.qcut(pd.Series(x).rank(method="first"), bins, labels=False)
    table = pd.DataFrame({"x": x, "y": y, "bin": labels}).groupby("bin", sort=True)
    return table.x.median().to_numpy(), table.y.mean().to_numpy(), table.y.sem().to_numpy()


def image_adjusted_curve(
    x: np.ndarray, y: np.ndarray, image: np.ndarray, bins: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Display the same within-image contrast used by the reported slope."""
    x_adjusted = x - pd.Series(x).groupby(image).transform("mean").to_numpy() + x.mean()
    y_adjusted = y - pd.Series(y).groupby(image).transform("mean").to_numpy() + y.mean()
    return binned_curve(x_adjusted, y_adjusted, bins=bins)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    condition = pd.read_csv(ASSEMBLED / "condition_index.csv")
    images = pd.read_csv(COHORT / "corrected100_images.csv")
    traces = pd.read_csv(COHORT / "corrected1000_traces.csv")
    legacy_traces = pd.read_csv(LEGACY / "trace_feature_table.csv")
    condition = condition.merge(
        images[["image_index", "cohort_role"]].rename(columns={"cohort_role": "image_role"})
    ).merge(
        traces[
            [
                "trace_index",
                "cohort_role",
                "corrected_dpi_crop120_path_length_arcmin",
                "legacy_path_rank",
                "corrected_path_rank",
            ]
        ].rename(columns={"cohort_role": "trace_role"})
    ).merge(
        legacy_traces[["trace_bank_index", "rendered_path_length_arcmin"]].rename(
            columns={"trace_bank_index": "trace_index"}
        )
    )
    return condition, images, traces


def select_input_examples(images: pd.DataFrame, traces: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    retained_images = images[images.cohort_role.eq("retained_valid_legacy_image")].copy()
    values = retained_images.legacy_corrected_patch_pixel_r.astype(float)
    selections = [
        retained_images.loc[[values.idxmax()]].assign(selection_role="closest_image_match"),
        retained_images.loc[[(values - values.median()).abs().idxmin()]].assign(selection_role="typical_image_match"),
        retained_images.loc[[values.idxmin()]].assign(selection_role="largest_image_change"),
    ]
    image_examples = pd.concat(selections).drop_duplicates("image_index")
    retained_traces = traces[traces.cohort_role.eq("retained_explicit_history_valid_legacy_trace")].copy()
    retained_traces["absolute_path_rank_change"] = (
        retained_traces.corrected_path_rank - retained_traces.legacy_path_rank
    ).abs()
    trace_examples = pd.concat(
        [
            retained_traces.nsmallest(1, "absolute_path_rank_change").assign(selection_role="stable_path_rank"),
            retained_traces.nlargest(1, "corrected_path_rank").assign(selection_role="long_corrected_path"),
            retained_traces.nlargest(1, "absolute_path_rank_change").assign(selection_role="largest_path_rank_change"),
        ]
    ).drop_duplicates("trace_index")
    return image_examples, trace_examples


def plot_inputs(images: pd.DataFrame, traces: pd.DataFrame, image_examples: pd.DataFrame, trace_examples: pd.DataFrame) -> None:
    legacy_xy = np.load(LEGACY / "trace_xy.npy")
    corrected_xy = np.load(
        ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/input_cache/corrected_trace_segments.npz"
    )["score_xy_deg"]
    fig, axes = plt.subplots(2, 6, figsize=(15, 7), constrained_layout=True)
    for column, row in enumerate(image_examples.itertuples(index=False)):
        with np.load(row.corrected_patch_npz, allow_pickle=False) as data:
            old = np.asarray(data["legacy_patch"])
            new = np.asarray(data["corrected_patch"])
        axes[0, 2 * column].imshow(old, cmap="gray", origin="upper")
        axes[0, 2 * column + 1].imshow(new, cmap="gray", origin="upper")
        axes[0, 2 * column].set_title(f"{row.selection_role}\nlegacy patch", fontsize=8)
        axes[0, 2 * column + 1].set_title(
            f"corrected patch\npixel r={row.legacy_corrected_patch_pixel_r:.2f}", fontsize=8
        )
        axes[0, 2 * column].axis("off")
        axes[0, 2 * column + 1].axis("off")
    for column, row in enumerate(trace_examples.itertuples(index=False)):
        old = legacy_xy[int(row.trace_index)] - legacy_xy[int(row.trace_index)].mean(0)
        new = corrected_xy[int(row.trace_index)] - corrected_xy[int(row.trace_index)].mean(0)
        scale = 60.0
        ax_old, ax_new = axes[1, 2 * column], axes[1, 2 * column + 1]
        ax_old.plot(old[:, 0] * scale, old[:, 1] * scale, "-o", ms=2, lw=1)
        ax_new.plot(new[:, 0] * scale, new[:, 1] * scale, "-o", ms=2, lw=1, color="#D55E00")
        lim = max(np.abs(old * scale).max(), np.abs(new * scale).max()) * 1.08
        for ax in (ax_old, ax_new):
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.axhline(0, color="0.85", lw=0.5)
            ax.axvline(0, color="0.85", lw=0.5)
        ax_old.set_title(f"{row.selection_role}\nlegacy rendered trace", fontsize=8)
        ax_new.set_title(
            f"corrected dpi_pix trace\nrank change={row.corrected_path_rank-row.legacy_path_rank:+.2f}", fontsize=8
        )
        ax_old.set_xlabel("x (arcmin)")
        ax_new.set_xlabel("x (arcmin)")
        ax_old.set_ylabel("y (arcmin)")
    fig.suptitle(
        "Matched-input audit: the corrected run changes crop geometry and eye trajectories\n"
        "Examples selected algorithmically; images shown with photographic origin (upper)",
        fontsize=13,
        weight="bold",
    )
    fig.savefig(OUT / "matched_input_examples.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "matched_input_examples.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    condition, images, traces = load_tables()
    image_examples, trace_examples = select_input_examples(images, traces)
    image_examples.to_csv(OUT / "selected_image_examples.csv", index=False)
    trace_examples.to_csv(OUT / "selected_trace_examples.csv", index=False)
    plot_inputs(images, traces, image_examples, trace_examples)

    image_id = condition.image_index.to_numpy(int)
    trace_id = condition.trace_index.to_numpy(int)
    matched = (
        condition.image_role.eq("retained_valid_legacy_image")
        & condition.trace_role.eq("retained_explicit_history_valid_legacy_trace")
    ).to_numpy()
    corrected_path = condition.corrected_dpi_crop120_path_length_arcmin.to_numpy(float)
    legacy_path = condition.rendered_path_length_arcmin.to_numpy(float)

    moving_info = np.load(ASSEMBLED / "moving_information_numerator_bits_spikes.npy")
    moving_spikes = np.load(ASSEMBLED / "moving_expected_spikes.npy")
    corrected_rate = np.load(ASSEMBLED / "moving_mean_rate_hz.npy")
    corrected_unit_ssi = np.load(ASSEMBLED / "moving_movie_ssi_bits_per_spike.npy")
    baseline = np.load(ASSEMBLED / "stabilized_by_image_sufficient_statistics.npz")
    legacy_rate = np.load(LEGACY / "mean_rate_matrix.npy", mmap_mode="r").reshape(100, 1000, 100)[image_id, trace_id]
    legacy_unit_ssi = np.load(LEGACY / "ssi_matrix.npy", mmap_mode="r").reshape(100, 1000, 100)[image_id, trace_id]
    legacy_spikes = np.load(LEGACY / "expected_spikes_matrix.npy", mmap_mode="r").reshape(100, 1000, 100)[image_id, trace_id]
    legacy_baseline_ssi = np.load(LEGACY / "stabilized_ssi_by_image.npy")[image_id]
    legacy_baseline_spikes = np.load(LEGACY / "stabilized_expected_spikes_by_image.npy")[image_id]

    assignments = pd.read_csv(ASSIGNMENTS)
    groups = {
        "all RR100": np.arange(100),
        "low SF": assignments.loc[assignments.sf_outer_third.eq("sf_low_half"), "rr100_index"].to_numpy(int),
        "high SF": assignments.loc[assignments.sf_outer_third.eq("sf_high_half"), "rr100_index"].to_numpy(int),
    }
    responses: dict[str, dict[str, np.ndarray]] = {}
    slope_rows: list[dict[str, object]] = []
    for group, units in groups.items():
        corrected_absolute = population_ssi(moving_info, moving_spikes, units)
        corrected_baseline = population_ssi(
            baseline["movie_ssi_bits_per_spike"] * baseline["expected_spikes"],
            baseline["expected_spikes"],
            units,
        )[image_id]
        legacy_absolute = population_ssi(legacy_unit_ssi * legacy_spikes, legacy_spikes, units)
        legacy_baseline = population_ssi(
            legacy_baseline_ssi * legacy_baseline_spikes, legacy_baseline_spikes, units
        )
        responses[group] = {
            "corrected_absolute": corrected_absolute,
            "legacy_absolute": legacy_absolute,
            "corrected_delta": corrected_absolute - corrected_baseline,
            "legacy_delta": legacy_absolute - legacy_baseline,
        }
        variants = [
            ("corrected all 1,000", corrected_path, responses[group]["corrected_delta"], np.ones(len(condition), bool)),
            ("corrected matched 479", corrected_path, responses[group]["corrected_delta"], matched),
            ("legacy matched; corrected path", corrected_path, responses[group]["legacy_delta"], matched),
            ("legacy matched; legacy path", legacy_path, responses[group]["legacy_delta"], matched),
        ]
        for source, x, y, use in variants:
            estimate = within_image_slope(x[use], y[use], image_id[use])
            low, median, high, probability = image_cluster_bootstrap(x[use], y[use], image_id[use])
            slope_rows.append(
                {
                    "sf_group": group,
                    "source": source,
                    "n_conditions": int(use.sum()),
                    "n_images": int(np.unique(image_id[use]).size),
                    "estimate_bits_per_spike_per_arcmin": estimate,
                    "bootstrap_median": median,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_probability_gt_zero": probability,
                }
            )
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(OUT / "path_slope_comparison.csv", index=False)

    all_response = responses["all RR100"]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "directional_pilot_complete_not_full_validation",
        "corrected_conditions": int(len(condition)),
        "matched_retained_conditions": int(matched.sum()),
        "matched_images": int(np.unique(image_id[matched]).size),
        "absolute_population_ssi_spearman": float(
            spearmanr(all_response["corrected_absolute"][matched], all_response["legacy_absolute"][matched]).statistic
        ),
        "delta_population_ssi_spearman": float(
            spearmanr(all_response["corrected_delta"][matched], all_response["legacy_delta"][matched]).statistic
        ),
        "matched_delta_sign_agreement_fraction": float(
            np.mean(
                np.sign(all_response["corrected_delta"][matched])
                == np.sign(all_response["legacy_delta"][matched])
            )
        ),
        "matched_corrected_delta_mean": float(all_response["corrected_delta"][matched].mean()),
        "matched_legacy_delta_mean": float(all_response["legacy_delta"][matched].mean()),
        "condition_mean_rate_spearman": float(
            spearmanr(corrected_rate[matched].mean(1), legacy_rate[matched].mean(1)).statistic
        ),
        "median_per_unit_rate_spearman": float(
            np.nanmedian([spearmanr(corrected_rate[matched, unit], legacy_rate[matched, unit]).statistic for unit in range(100)])
        ),
        "median_per_unit_ssi_spearman": float(
            np.nanmedian(
                [spearmanr(corrected_unit_ssi[matched, unit], legacy_unit_ssi[matched, unit]).statistic for unit in range(100)]
            )
        ),
        "interpretation": {
            "supported": "absolute response organization, positive FEM-minus-stabilized SSI, and positive low-SF path dependence",
            "refined": "corrected high-SF path dependence is clearly negative; the legacy cache was weak and path-definition dependent",
            "unresolved": "full-bank convergence, image/trace variance decomposition, and tuning-weighted spectral prediction",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(all_response["legacy_absolute"][matched], all_response["corrected_absolute"][matched], s=10, alpha=0.5)
    ax.set(xlabel="legacy population SSI", ylabel="corrected population SSI", title=f"Absolute organization transfers\nSpearman ρ={summary['absolute_population_ssi_spearman']:.2f}")
    ax = axes[0, 1]
    ax.scatter(all_response["legacy_delta"][matched], all_response["corrected_delta"][matched], s=10, alpha=0.5)
    ax.axhline(0, color="0.5", lw=0.7); ax.axvline(0, color="0.5", lw=0.7)
    ax.set(xlabel="legacy FEM−stabilized SSI", ylabel="corrected FEM−stabilized SSI", title=f"FEM-effect ordering largely transfers\nSpearman ρ={summary['delta_population_ssi_spearman']:.2f}")
    ax = axes[0, 2]
    ax.scatter(legacy_path[matched], corrected_path[matched], s=10, alpha=0.5, color="#009E73")
    ax.set(xlabel="legacy rendered path (arcmin)", ylabel="corrected dpi_pix path (arcmin)", title=f"Trace ranks partly transfer\nSpearman ρ={spearmanr(legacy_path[matched], corrected_path[matched]).statistic:.2f}")

    colors = {"corrected all 1,000": "#D55E00", "corrected matched 479": "#E69F00", "legacy matched; corrected path": "#0072B2"}
    for column, group in enumerate(("low SF", "high SF")):
        ax = axes[1, column]
        for source, x, y, use in [
            ("corrected all 1,000", corrected_path, responses[group]["corrected_delta"], np.ones(len(condition), bool)),
            ("corrected matched 479", corrected_path, responses[group]["corrected_delta"], matched),
            ("legacy matched; corrected path", corrected_path, responses[group]["legacy_delta"], matched),
        ]:
            bx, by, be = image_adjusted_curve(x[use], y[use], image_id[use])
            ax.errorbar(bx, by, yerr=be, marker="o", lw=1.5, label=source, color=colors[source])
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set(xlabel="image-adjusted corrected path (arcmin)", ylabel="image-adjusted FEM−stabilized SSI", title=f"{group} units")
        ax.legend(fontsize=7)
    ax = axes[1, 2]
    show = slopes[
        slopes.sf_group.isin(["low SF", "high SF"])
        & slopes.source.isin(["corrected all 1,000", "corrected matched 479", "legacy matched; corrected path"])
    ].copy()
    positions = []
    labels = []
    for group_index, group in enumerate(("low SF", "high SF")):
        for source_index, source in enumerate(colors):
            row = show[(show.sf_group == group) & (show.source == source)].iloc[0]
            position = group_index * 4 + source_index
            positions.append(position); labels.append(source.replace("corrected ", "corr. ").replace(" matched", "\nmatched"))
            ax.errorbar(position, row.estimate_bits_per_spike_per_arcmin, yerr=[[row.estimate_bits_per_spike_per_arcmin-row.ci_low], [row.ci_high-row.estimate_bits_per_spike_per_arcmin]], fmt="o", color=colors[source], capsize=3)
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xticks(positions, labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("within-image SSI slope\n(bits/spike)/arcmin")
    ax.set_title("Direction of path effect")
    fig.suptitle(
        "Corrected 1,000-condition pilot versus flawed legacy cache\n"
        "Matched comparisons use 479 retained image–trace identities; replacements are summarized separately",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(OUT / "directional_results_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "directional_results_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
