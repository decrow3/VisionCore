#!/usr/bin/env python3
"""Convergence audit for the first three balanced corrected RR100 rounds.

Every round contains all 100 images and all 1,000 traces exactly once, with
10 traces per image.  The round is therefore the natural independent block
for asking whether the interim Figure 4 conclusions stabilize as conditions
accumulate.  This script deliberately uses only complete balanced rounds.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED = (
    ROOT
    / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/assembled/rounds_000_002_n003"
)
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
ASSIGNMENTS = (
    ROOT
    / "outputs/fig4_active_sensing/backimage_real_trace_sf_halves_recorded_validated_r0p5_v1/"
    "sf_half_recorded_validated_unit_assignments.csv"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_corrected_three_round_convergence_checkpoint_39_v1"
SEED = 20260813
N_BOOT = 4000


def population_ssi(info: np.ndarray, spikes: np.ndarray, units: np.ndarray) -> np.ndarray:
    return info[:, units].sum(axis=1) / np.maximum(spikes[:, units].sum(axis=1), 1e-12)


def residualize_by_image(values: np.ndarray, image_ids: np.ndarray) -> np.ndarray:
    return values - pd.Series(values).groupby(image_ids).transform("mean").to_numpy()


def within_image_slope(x: np.ndarray, y: np.ndarray, image_ids: np.ndarray) -> float:
    xx = residualize_by_image(x, image_ids)
    yy = residualize_by_image(y, image_ids)
    denominator = float(np.dot(xx, xx))
    return float(np.dot(xx, yy) / denominator) if denominator > 0 else float("nan")


def summarize_subset(
    x: np.ndarray,
    y: np.ndarray,
    image_ids: np.ndarray,
    *,
    rng: np.random.Generator,
) -> dict[str, float]:
    images = np.unique(image_ids)
    slope_samples = np.empty(N_BOOT, dtype=float)
    mean_samples = np.empty(N_BOOT, dtype=float)
    image_means = pd.DataFrame({"image": image_ids, "y": y}).groupby("image").y.mean()
    for iteration in range(N_BOOT):
        sampled = rng.choice(images, len(images), replace=True)
        numerator = 0.0
        denominator = 0.0
        for identity in sampled:
            use = image_ids == identity
            xx = x[use] - x[use].mean()
            yy = y[use] - y[use].mean()
            numerator += float(np.dot(xx, yy))
            denominator += float(np.dot(xx, xx))
        slope_samples[iteration] = numerator / denominator
        mean_samples[iteration] = float(image_means.loc[sampled].mean())
    slope_ci = np.quantile(slope_samples, [0.025, 0.5, 0.975])
    mean_ci = np.quantile(mean_samples, [0.025, 0.5, 0.975])
    return {
        "path_slope": within_image_slope(x, y, image_ids),
        "path_slope_bootstrap_median": float(slope_ci[1]),
        "path_slope_ci_low": float(slope_ci[0]),
        "path_slope_ci_high": float(slope_ci[2]),
        "path_slope_probability_gt_zero": float(np.mean(slope_samples > 0)),
        "mean_delta_ssi": float(y.mean()),
        "mean_delta_bootstrap_median": float(mean_ci[1]),
        "mean_delta_ci_low": float(mean_ci[0]),
        "mean_delta_ci_high": float(mean_ci[2]),
        "mean_delta_probability_gt_zero": float(np.mean(mean_samples > 0)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    condition = pd.read_csv(ASSEMBLED / "condition_index.csv")
    if condition.groupby("round_index").size().to_dict() != {0: 1000, 1: 1000, 2: 1000}:
        raise RuntimeError("Expected exactly three complete 1,000-condition rounds")
    for round_index, frame in condition.groupby("round_index"):
        if frame.image_index.nunique() != 100 or frame.trace_index.nunique() != 1000:
            raise RuntimeError(f"Round {round_index} is not balanced")

    traces = pd.read_csv(COHORT / "corrected1000_traces.csv")
    condition = condition.merge(
        traces[["trace_index", "corrected_dpi_crop120_path_length_arcmin"]],
        on="trace_index",
        how="left",
        validate="many_to_one",
    )
    if condition.corrected_dpi_crop120_path_length_arcmin.isna().any():
        raise RuntimeError("Missing corrected path length")

    moving_info = np.load(ASSEMBLED / "moving_information_numerator_bits_spikes.npy")
    moving_spikes = np.load(ASSEMBLED / "moving_expected_spikes.npy")
    baseline = np.load(ASSEMBLED / "stabilized_by_image_sufficient_statistics.npz")
    baseline_info = baseline["movie_ssi_bits_per_spike"] * baseline["expected_spikes"]
    baseline_spikes = baseline["expected_spikes"]

    assignments = pd.read_csv(ASSIGNMENTS)
    groups = {
        "all RR100": np.arange(100, dtype=int),
        "low SF": assignments.loc[
            assignments.sf_outer_third.eq("sf_low_half"), "rr100_index"
        ].to_numpy(int),
        "high SF": assignments.loc[
            assignments.sf_outer_third.eq("sf_high_half"), "rr100_index"
        ].to_numpy(int),
    }

    image_ids = condition.image_index.to_numpy(int)
    round_ids = condition.round_index.to_numpy(int)
    paths = condition.corrected_dpi_crop120_path_length_arcmin.to_numpy(float)
    rng = np.random.default_rng(SEED)
    estimate_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    deltas: dict[str, np.ndarray] = {}

    subsets: list[tuple[str, np.ndarray]] = []
    for round_index in range(3):
        subsets.append((f"round {round_index + 1} only", round_ids == round_index))
    for last_round in range(3):
        subsets.append((f"cumulative {last_round + 1}", round_ids <= last_round))
    for omitted in range(3):
        subsets.append((f"leave round {omitted + 1} out", round_ids != omitted))

    for group_name, units in groups.items():
        moving = population_ssi(moving_info, moving_spikes, units)
        stabilized_by_image = population_ssi(baseline_info, baseline_spikes, units)
        delta = moving - stabilized_by_image[image_ids]
        deltas[group_name] = delta
        for subset_name, use in subsets:
            stats = summarize_subset(paths[use], delta[use], image_ids[use], rng=rng)
            estimate_rows.append(
                {
                    "sf_group": group_name,
                    "subset": subset_name,
                    "n_conditions": int(use.sum()),
                    "n_rounds": int(np.unique(round_ids[use]).size),
                    "n_images": int(np.unique(image_ids[use]).size),
                    "n_units": int(len(units)),
                    **stats,
                }
            )
        for round_index in range(3):
            use = round_ids == round_index
            table = pd.DataFrame({"image_index": image_ids[use], "delta": delta[use]}).groupby(
                "image_index", as_index=False
            ).delta.mean()
            for row in table.itertuples(index=False):
                profile_rows.append(
                    {
                        "sf_group": group_name,
                        "round_index": round_index,
                        "image_index": int(row.image_index),
                        "mean_delta_ssi": float(row.delta),
                    }
                )

    estimates = pd.DataFrame(estimate_rows)
    estimates.to_csv(OUT / "round_and_cumulative_estimates.csv", index=False)
    profiles = pd.DataFrame(profile_rows)
    profiles.to_csv(OUT / "per_round_image_profiles.csv", index=False)

    correlation_rows: list[dict[str, object]] = []
    for group_name in groups:
        wide = profiles[profiles.sf_group.eq(group_name)].pivot(
            index="image_index", columns="round_index", values="mean_delta_ssi"
        )
        for first, second in ((0, 1), (0, 2), (1, 2)):
            result = spearmanr(wide[first], wide[second])
            correlation_rows.append(
                {
                    "sf_group": group_name,
                    "round_a": first + 1,
                    "round_b": second + 1,
                    "image_profile_spearman_rho": float(result.statistic),
                    "p_value": float(result.pvalue),
                }
            )
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(OUT / "independent_round_image_profile_correlations.csv", index=False)

    colors = {"all RR100": "#333333", "low SF": "#0072B2", "high SF": "#D55E00"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    cumulative = estimates[estimates.subset.str.startswith("cumulative")].copy()
    cumulative["rounds"] = cumulative.subset.str.extract(r"(\d+)$").astype(int)
    individual = estimates[estimates.subset.str.contains("only")].copy()
    individual["rounds"] = individual.subset.str.extract(r"round (\d+)").astype(int)
    for group_name, color in colors.items():
        sub = cumulative[cumulative.sf_group.eq(group_name)].sort_values("rounds")
        axes[0, 0].errorbar(
            sub.rounds,
            sub.mean_delta_ssi,
            yerr=[sub.mean_delta_ssi - sub.mean_delta_ci_low, sub.mean_delta_ci_high - sub.mean_delta_ssi],
            marker="o",
            color=color,
            label=group_name,
            capsize=3,
        )
        axes[0, 1].errorbar(
            sub.rounds,
            sub.path_slope,
            yerr=[sub.path_slope - sub.path_slope_ci_low, sub.path_slope_ci_high - sub.path_slope],
            marker="o",
            color=color,
            label=group_name,
            capsize=3,
        )
    axes[0, 0].axhline(0, color="0.7", lw=0.8)
    axes[0, 0].set(title="FEM effect stabilizes as rounds accumulate", xlabel="complete balanced rounds", ylabel="mean FEM − stabilized SSI\n(bits/spike)", xticks=[1, 2, 3])
    axes[0, 1].axhline(0, color="0.7", lw=0.8)
    axes[0, 1].set(title="Path-length effect across cumulative samples", xlabel="complete balanced rounds", ylabel="within-image slope\n(bits/spike)/arcmin", xticks=[1, 2, 3])
    axes[0, 1].legend(frameon=False, fontsize=8)

    positions = np.arange(3)
    offsets = {"all RR100": -0.2, "low SF": 0.0, "high SF": 0.2}
    for group_name, color in colors.items():
        sub = individual[individual.sf_group.eq(group_name)].sort_values("rounds")
        axes[0, 2].errorbar(
            positions + offsets[group_name],
            sub.path_slope,
            yerr=[sub.path_slope - sub.path_slope_ci_low, sub.path_slope_ci_high - sub.path_slope],
            fmt="o",
            color=color,
            capsize=3,
            label=group_name,
        )
    axes[0, 2].axhline(0, color="0.7", lw=0.8)
    axes[0, 2].set(title="Each round separately", ylabel="within-image path slope", xticks=positions, xticklabels=["round 1", "round 2", "round 3"])

    for column, group_name in enumerate(groups):
        wide = profiles[profiles.sf_group.eq(group_name)].pivot(index="image_index", columns="round_index", values="mean_delta_ssi")
        ax = axes[1, column]
        ax.scatter(wide[0], wide[1], s=18, alpha=0.65, color=colors[group_name], label="round 2")
        ax.scatter(wide[0], wide[2], s=18, alpha=0.65, color="#009E73", marker="x", label="round 3")
        rho12 = spearmanr(wide[0], wide[1]).statistic
        rho13 = spearmanr(wide[0], wide[2]).statistic
        ax.axhline(0, color="0.8", lw=0.7)
        ax.axvline(0, color="0.8", lw=0.7)
        ax.set(
            title=f"{group_name}: image ordering across rounds\nρ₁₂={rho12:.2f}; ρ₁₃={rho13:.2f}",
            xlabel="round 1 image-mean ΔSSI",
            ylabel="later-round image-mean ΔSSI",
        )
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Corrected Figure 4 cache: convergence across three balanced rounds\n"
        "Each round spans all 100 images and all 1,000 traces once; uncertainty resamples images",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(OUT / "three_round_convergence.png", dpi=190, bbox_inches="tight")
    fig.savefig(OUT / "three_round_convergence.pdf", bbox_inches="tight")
    plt.close(fig)

    final_rows = estimates[estimates.subset.eq("cumulative 3")].set_index("sf_group")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "three_balanced_round_interim_convergence_complete",
        "source": str(ASSEMBLED),
        "n_conditions": 3000,
        "n_complete_rounds": 3,
        "n_images": 100,
        "n_unique_traces_per_round": 1000,
        "bootstrap": {"unit": "image", "n": N_BOOT, "seed": SEED},
        "population_groups": {name: int(len(units)) for name, units in groups.items()},
        "cumulative_three_round_results": {
            name: {
                "mean_delta_ssi": float(final_rows.loc[name, "mean_delta_ssi"]),
                "mean_delta_ci": [
                    float(final_rows.loc[name, "mean_delta_ci_low"]),
                    float(final_rows.loc[name, "mean_delta_ci_high"]),
                ],
                "path_slope": float(final_rows.loc[name, "path_slope"]),
                "path_slope_ci": [
                    float(final_rows.loc[name, "path_slope_ci_low"]),
                    float(final_rows.loc[name, "path_slope_ci_high"]),
                ],
            }
            for name in groups
        },
        "scope": (
            "Interim convergence evidence from 3% of the predeclared 100x1000 crossing. "
            "It is not a complete half-bank or full-bank result."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Corrected RR100 three-round convergence\n\n"
        "This checkpoint uses only complete balanced rounds 0–2. Each round contains all 100 images and all "
        "1,000 traces exactly once, with ten traces per image. It tests convergence of the FEM-minus-stabilized "
        "population SSI and its within-image association with corrected DPI path length. It is an interim 3,000-condition "
        "analysis, not a substitute for either predeclared 50,000-condition half.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
