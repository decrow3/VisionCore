"""Build simple Figure 4C continuous-joint estimator diagnostics.

This script summarizes the cache-only continuous latent-eye estimator runs
against the finite trajectory-table observer baselines. It intentionally keeps
the plots plain: the goal is to check whether the continuous estimator is
tractable and where it fails, not to make a promoted panel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1"
)
OUT_DIR = (
    REPO_ROOT
    / "declan"
    / "figure4_active_sensing_atlas"
    / "figures"
    / "panel_C"
    / "diagnostics"
    / "continuous_joint"
)
HEADLINE_RUN_SLUG = "noanchor_knownstart_linear_ar1_var0p01"


@dataclass(frozen=True)
class RunSpec:
    label: str
    slug: str
    path: Path
    full_cache: bool


RUNS = [
    RunSpec(
        label="Kalman marginal, k=50",
        slug="kalman_k50",
        path=SOURCE_ROOT / "continuous_joint_kalman_image_disjoint_basis50_v1",
        full_cache=True,
    ),
    RunSpec(
        label="Linear Poisson profile, k=5",
        slug="poisson_k5",
        path=SOURCE_ROOT / "continuous_joint_linear_poisson_profile_k5_v1",
        full_cache=True,
    ),
    RunSpec(
        label="Linear Poisson profile, k=10",
        slug="poisson_k10",
        path=SOURCE_ROOT / "continuous_joint_linear_poisson_profile_k10_v1",
        full_cache=True,
    ),
    RunSpec(
        label="Linear Poisson profile, k=10, A(t)",
        slug="poisson_k10_timevary",
        path=SOURCE_ROOT / "continuous_joint_linear_poisson_compact_k10_timevary_full",
        full_cache=True,
    ),
    RunSpec(
        label="Linear Poisson profile, k=10, smooth A(t)",
        slug="poisson_k10_timevary_smooth",
        path=SOURCE_ROOT / "continuous_joint_linear_poisson_compact_k10_smoothAt_a0p70_q1em02_full",
        full_cache=True,
    ),
    RunSpec(
        label="Linear Poisson profile, k=10, matched Brownian prior",
        slug="poisson_k10_matched_brownian",
        path=SOURCE_ROOT / "continuous_joint_linear_poisson_compact_k10_matched_brownian_scale1_full",
        full_cache=True,
    ),
    RunSpec(
        label="No-anchor known-start AR(1), k=10",
        slug=HEADLINE_RUN_SLUG,
        path=SOURCE_ROOT / "continuous_joint_noanchor_knownstart_linear_ar1_var0p01_full",
        full_cache=True,
    ),
    RunSpec(
        label="No-anchor known-start AR(1), axis-interleaved k=10",
        slug="noanchor_knownstart_linear_ar1_axis_interleaved_basis_k10",
        path=SOURCE_ROOT / "continuous_joint_noanchor_knownstart_linear_ar1_axis_interleaved_basis_k10_full",
        full_cache=True,
    ),
    RunSpec(
        label="No-anchor residual CTF known-start DCT, k=10",
        slug="noanchor_residual_c2f_knownstart_dct_ar1_var0p01",
        path=SOURCE_ROOT / "continuous_joint_noanchor_residual_c2f_knownstart_dct_ar1_var0p01_full",
        full_cache=True,
    ),
    RunSpec(
        label="No-anchor DCT coarse-to-fine, k=10, AR(1)",
        slug="noanchor_dct_c2f_k10_ar1_basis2_var0p05",
        path=SOURCE_ROOT / "continuous_joint_noanchor_dct_c2f_k10_ar1_basis2_var0p05_full",
        full_cache=True,
    ),
    RunSpec(
        label="No-anchor DCT coarse-to-fine, k=10, matched Brownian",
        slug="noanchor_dct_c2f_k10_matched_brownian_basis12_var0p05",
        path=SOURCE_ROOT / "continuous_joint_noanchor_dct_c2f_k10_matched_brownian_basis12_var0p05_full",
        full_cache=True,
    ),
    RunSpec(
        label="Catalog residual profile, k=10, smooth A(t)",
        slug="catalog_residual_k10_timevary_smooth",
        path=SOURCE_ROOT / "continuous_joint_catalog_residual_compact_k10_smoothAt_a0p70_q1em02_full",
        full_cache=True,
    ),
    RunSpec(
        label="Catalog residual profile, k=10, smooth A(t), top-2 anchors",
        slug="catalog_residual_k10_timevary_smooth_topk2",
        path=SOURCE_ROOT / "continuous_joint_catalog_residual_compact_k10_smoothAt_a0p70_q1em02_topk2_full",
        full_cache=True,
    ),
    RunSpec(
        label="Catalog residual profile, k=10, smooth A(t), top-2 anchors, tuned AR(1)",
        slug="catalog_residual_k10_timevary_smooth_topk2_a0p85",
        path=SOURCE_ROOT / "continuous_joint_catalog_residual_compact_k10_smoothAt_a0p85_q1em02_topk2_full",
        full_cache=True,
    ),
    RunSpec(
        label="Catalog residual profile, k=10, smooth A(t), shrunk top-2 anchors",
        slug="catalog_residual_k10_timevary_smooth_topk2_shrink0p2",
        path=SOURCE_ROOT / "continuous_joint_catalog_residual_compact_k10_smoothAt_a0p70_q1em02_topk2_shrink0p2_full",
        full_cache=True,
    ),
    RunSpec(
        label="Catalog residual profile, k=10, smooth A(t), shrunk top-2 anchors, fine scan",
        slug="catalog_residual_k10_timevary_smooth_topk2_shrink0p19",
        path=SOURCE_ROOT / "continuous_joint_catalog_residual_compact_k10_smoothAt_a0p70_q1em02_topk2_shrink0p19_full",
        full_cache=True,
    ),
    RunSpec(
        label="Catalog residual profile, k=10, coarse-to-fine anchors, keep 16",
        slug="catalog_residual_c2f_k10_sched6_0_keep16_topk2_shrink0p19",
        path=SOURCE_ROOT / "continuous_joint_catalog_residual_c2f_k10_sched6-0_keep16_topk2_shrink0p19_full",
        full_cache=True,
    ),
    RunSpec(
        label="Catalog residual profile, k=10, coarse-to-fine anchors, keep 8",
        slug="catalog_residual_c2f_k10_sched6_0_keep8_topk2_shrink0p19",
        path=SOURCE_ROOT / "continuous_joint_catalog_residual_c2f_k10_sched6-0_keep8_topk2_shrink0p19_full",
        full_cache=True,
    ),
    RunSpec(
        label="Catalog residual profile, k=10, coarse-to-fine anchors, keep 4",
        slug="catalog_residual_c2f_k10_sched6_0_keep4_topk2_shrink0p19",
        path=SOURCE_ROOT / "continuous_joint_catalog_residual_c2f_k10_sched6-0_keep4_topk2_shrink0p19_full",
        full_cache=True,
    ),
    RunSpec(
        label="Catalog residual profile, k=10, smoothed anchors, sigma=6",
        slug="catalog_residual_k10_topk2_shrink0p19_anchor_smooth6",
        path=SOURCE_ROOT / "continuous_joint_catalog_residual_k10_topk2_shrink0p19_anchor_smooth6_full",
        full_cache=True,
    ),
    RunSpec(
        label="Linear Poisson profile, k=20",
        slug="poisson_k20",
        path=SOURCE_ROOT / "continuous_joint_linear_poisson_profile_k20_v1",
        full_cache=True,
    ),
    RunSpec(
        label="Linear Poisson profile, k=20, A(t)",
        slug="poisson_k20_timevary",
        path=SOURCE_ROOT / "continuous_joint_linear_poisson_compact_k20_timevary_full",
        full_cache=True,
    ),
    RunSpec(
        label="Linear Poisson profile, k=50 subset",
        slug="poisson_k50_subset96",
        path=SOURCE_ROOT / "continuous_joint_linear_poisson_profile_k50_subset96",
        full_cache=False,
    ),
    RunSpec(
        label="Subset: known-start AR(1), k=10",
        slug="subset_knownstart_ar1_k10",
        path=SOURCE_ROOT / "continuous_joint_noanchor_knownstart_linear_ar1_var0p01_subset96",
        full_cache=False,
    ),
    RunSpec(
        label="Subset: axis-interleaved known-start AR(1)",
        slug="subset_axis_interleaved_ar1_k10",
        path=SOURCE_ROOT / "continuous_joint_noanchor_knownstart_linear_ar1_axis_interleaved_basis_k10_subset96",
        full_cache=False,
    ),
    RunSpec(
        label="Subset: full-unit known-start AR(1)",
        slug="subset_knownstart_ar1_fullunits",
        path=SOURCE_ROOT / "continuous_joint_noanchor_knownstart_linear_ar1_identity_fullunits_subset96",
        full_cache=False,
    ),
    RunSpec(
        label="Subset: catalog Gaussian prior",
        slug="subset_catalog_gaussian_prior",
        path=SOURCE_ROOT / "continuous_joint_noanchor_knownstart_catalog_gaussian_s0_shrink0_subset96",
        full_cache=False,
    ),
    RunSpec(
        label="Subset: catalog Gaussian CTF",
        slug="subset_catalog_gaussian_ctf",
        path=SOURCE_ROOT / "continuous_joint_noanchor_residual_c2f_knownstart_catalog_gaussian_s0_shrink0_subset96",
        full_cache=False,
    ),
]

COLORS = {
    "known": "#111827",
    "zero": "#6b7280",
    "joint": "#235789",
    "best_single_tau": "#b35c2e",
    "continuous": "#2f8f6a",
    "rmse": "#8a5ca8",
    "corr": "#2f8f6a",
}
SCALE_POS = {0.5: 0.0, 1.0: 1.0, 2.0: 2.0}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9dee5", lw=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _read_run(spec: RunSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = spec.path / "continuous_joint_summary.csv"
    recovery_path = spec.path / "continuous_joint_trajectory_recovery.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not recovery_path.exists():
        raise FileNotFoundError(recovery_path)

    summary = pd.read_csv(summary_path)
    recovery = pd.read_csv(recovery_path)
    for frame in (summary, recovery):
        frame["run_label"] = spec.label
        frame["run_slug"] = spec.slug
        frame["full_cache"] = spec.full_cache
    return summary, recovery


def _load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    recoveries = []
    for spec in RUNS:
        summary, recovery = _read_run(spec)
        summaries.append(summary)
        recoveries.append(recovery)
    return pd.concat(summaries, ignore_index=True), pd.concat(recoveries, ignore_index=True)


def _overall_accuracy(summary: pd.DataFrame) -> pd.DataFrame:
    rows = summary[(summary["full_cache"]) & (summary["likelihood_scale"].astype(float) == 1.0)].copy()
    melted = rows.melt(
        id_vars=["run_label", "run_slug", "n_trials"],
        value_vars=[
            "known_accuracy",
            "zero_accuracy",
            "joint_accuracy",
            "best_single_tau_accuracy",
            "continuous_joint_accuracy",
        ],
        var_name="observer",
        value_name="accuracy",
    )
    overall = (
        melted.assign(weight=lambda df: df["n_trials"].astype(float))
        .groupby(["run_label", "run_slug", "observer"], as_index=False)
        .apply(lambda g: pd.Series({"accuracy": np.average(g["accuracy"], weights=g["weight"])}), include_groups=False)
    )
    label_map = {
        "known_accuracy": "known eye",
        "zero_accuracy": "zero eye",
        "joint_accuracy": "finite joint",
        "best_single_tau_accuracy": "best catalog trajectory",
        "continuous_joint_accuracy": "continuous joint",
    }
    overall["observer_label"] = overall["observer"].map(label_map)
    return overall


def _headline_by_scale(summary: pd.DataFrame) -> pd.DataFrame:
    rows = summary[
        (summary["run_slug"] == HEADLINE_RUN_SLUG)
        & (summary["likelihood_scale"].astype(float) == 1.0)
    ].copy()
    rows["prior_scale"] = rows["prior_scale"].astype(float)
    return (
        rows.groupby("prior_scale", as_index=False)
        .agg(
            known_accuracy=("known_accuracy", "mean"),
            zero_accuracy=("zero_accuracy", "mean"),
            finite_joint_accuracy=("joint_accuracy", "mean"),
            best_single_tau_accuracy=("best_single_tau_accuracy", "mean"),
            continuous_joint_accuracy=("continuous_joint_accuracy", "mean"),
            median_trajectory_rmse=("median_trajectory_rmse", "median"),
            median_trajectory_corr_mean=("median_trajectory_corr_mean", "median"),
            n_trials=("n_trials", "sum"),
        )
        .sort_values("prior_scale")
    )


def _headline_recovery(recovery: pd.DataFrame) -> pd.DataFrame:
    rows = recovery[
        (recovery["run_slug"] == HEADLINE_RUN_SLUG)
        & (recovery["likelihood_scale"].astype(float) == 1.0)
    ].copy()
    rows["prior_scale"] = rows["prior_scale"].astype(float)
    return (
        rows.groupby("prior_scale", as_index=False)
        .agg(
            median_rmse=("trajectory_rmse", "median"),
            q25_rmse=("trajectory_rmse", lambda x: np.quantile(x, 0.25)),
            q75_rmse=("trajectory_rmse", lambda x: np.quantile(x, 0.75)),
            median_corr=("trajectory_corr_mean", "median"),
            q25_corr=("trajectory_corr_mean", lambda x: np.quantile(x, 0.25)),
            q75_corr=("trajectory_corr_mean", lambda x: np.quantile(x, 0.75)),
            median_nearest_distance=("nearest_trajectory_distance", "median"),
            n_rows=("trajectory_rmse", "size"),
        )
        .sort_values("prior_scale")
    )


def _headline_conditioning() -> pd.DataFrame:
    spec = next(run for run in RUNS if run.slug == HEADLINE_RUN_SLUG)
    qc_path = spec.path / "continuous_joint_qc.csv"
    if not qc_path.exists():
        return pd.DataFrame()
    qc = pd.read_csv(qc_path)
    needed = {
        "prior_scale",
        "qc_type",
        "A_singular2_median",
        "A_anisotropy_median",
        "A_rank1_fraction_anisotropy_lt_0p2",
    }
    if not needed.issubset(set(qc.columns)):
        return pd.DataFrame()
    rows = qc[qc["qc_type"].eq("A_I_fit")].copy()
    rows["prior_scale"] = rows["prior_scale"].astype(float)
    out = (
        rows.groupby("prior_scale", as_index=False)
        .agg(
            median_singular1=("A_singular1_median", "median"),
            median_singular2=("A_singular2_median", "median"),
            median_singular2_p10=("A_singular2_p10", "median"),
            median_anisotropy=("A_anisotropy_median", "median"),
            median_log10_condition=("A_log10_condition_median", "median"),
            mean_rank1_fraction_lt_0p2=("A_rank1_fraction_anisotropy_lt_0p2", "mean"),
            n_rows=("A_singular2_median", "size"),
        )
        .sort_values("prior_scale")
    )
    return out


def _subset_prior_comparison(summary: pd.DataFrame, recovery: pd.DataFrame) -> pd.DataFrame:
    run_order = [
        "subset_knownstart_ar1_k10",
        "subset_axis_interleaved_ar1_k10",
        "subset_knownstart_ar1_fullunits",
        "subset_catalog_gaussian_prior",
        "subset_catalog_gaussian_ctf",
    ]
    rows = summary[
        summary["run_slug"].isin(run_order)
        & (summary["likelihood_scale"].astype(float) == 1.0)
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    metrics = ["zero_accuracy", "joint_accuracy", "best_single_tau_accuracy", "continuous_joint_accuracy"]
    recovery_rows = recovery[
        recovery["run_slug"].isin(run_order)
        & (recovery["likelihood_scale"].astype(float) == 1.0)
    ].copy()
    out_rows = []
    for slug, group in rows.groupby("run_slug", sort=False):
        weights = group["n_trials"].astype(float).to_numpy()
        row = {
            "run_slug": slug,
            "run_label": str(group["run_label"].iloc[0]),
            "n_trials": int(np.sum(group["n_trials"].astype(int))),
        }
        for metric in metrics:
            values = group[metric].astype(float).to_numpy()
            row[metric] = float(np.average(values, weights=weights))
        rec_group = recovery_rows[recovery_rows["run_slug"].eq(slug)]
        row["median_trajectory_rmse"] = float(rec_group["trajectory_rmse"].median()) if not rec_group.empty else float("nan")
        row["median_trajectory_corr_mean"] = (
            float(rec_group["trajectory_corr_mean"].median()) if not rec_group.empty else float("nan")
        )
        out_rows.append(row)
    out = pd.DataFrame(out_rows)
    out["run_order"] = out["run_slug"].map({slug: idx for idx, slug in enumerate(run_order)})
    return out.sort_values("run_order").drop(columns=["run_order"]).reset_index(drop=True)


def _plot_overall_accuracy(overall: pd.DataFrame) -> None:
    run_order = [
        "poisson_k10",
        "poisson_k10_timevary_smooth",
        HEADLINE_RUN_SLUG,
        "noanchor_knownstart_linear_ar1_axis_interleaved_basis_k10",
        "noanchor_residual_c2f_knownstart_dct_ar1_var0p01",
        "poisson_k10_matched_brownian",
        "noanchor_dct_c2f_k10_ar1_basis2_var0p05",
        "noanchor_dct_c2f_k10_matched_brownian_basis12_var0p05",
        "catalog_residual_k10_timevary_smooth",
        "catalog_residual_k10_timevary_smooth_topk2_shrink0p19",
    ]
    observer_order = [
        ("zero_accuracy", "zero", "zero eye"),
        ("joint_accuracy", "joint", "finite joint"),
        ("best_single_tau_accuracy", "best_single_tau", "best catalog"),
        ("continuous_joint_accuracy", "continuous", "continuous"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.2), constrained_layout=True)
    x = np.arange(len(run_order), dtype=float)
    width = 0.18
    for idx, (observer, color_key, label) in enumerate(observer_order):
        block = overall.set_index(["run_slug", "observer"]).reindex([(run, observer) for run in run_order])
        ax.bar(
            x + (idx - 1.5) * width,
            block["accuracy"].to_numpy(),
            width=width,
            color=COLORS[color_key],
            label=label,
        )
    known_ceiling = overall[overall["observer"].eq("known_accuracy")]["accuracy"].max()
    if np.isfinite(known_ceiling):
        ax.axhline(known_ceiling, color=COLORS["known"], lw=1.3, linestyle="--", label="known-eye ceiling")
    ax.set_title("Continuous joint observer: full-cache accuracy")
    ax.set_ylabel("image identification accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Poisson\nk=10",
            "Poisson\nk=10 smooth A(t)",
            "Known-start\nAR(1)",
            "Axis-interleaved\nAR(1)",
            "Known-start\nresidual CTF",
            "No-anchor\nBrownian",
            "No-anchor DCT\nAR(1)",
            "No-anchor DCT\nBrownian",
            "Catalog residual\nall anchors",
            "Catalog residual\nshrunk top-2",
        ],
        rotation=18,
        ha="right",
    )
    ax.set_ylim(0.25, 1.05)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    _clean_axis(ax)
    fig.savefig(OUT_DIR / "continuous_joint_overall_accuracy.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_overall_accuracy.pdf")
    plt.close(fig)


def _plot_subset_prior_comparison(comparison: pd.DataFrame) -> None:
    if comparison.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), constrained_layout=True)
    x = np.arange(comparison.shape[0], dtype=float)
    labels = [
        "AR(1)\nk=10",
        "axis-interleaved\nAR(1)",
        "AR(1)\nall units",
        "catalog\nGaussian",
        "Gaussian\nCTF",
    ]
    width = 0.22
    for offset, column, color_key, label in [
        (-1.0, "zero_accuracy", "zero", "zero"),
        (0.0, "joint_accuracy", "joint", "finite joint"),
        (1.0, "continuous_joint_accuracy", "continuous", "continuous"),
    ]:
        axes[0].bar(
            x + offset * width,
            comparison[column],
            width=width,
            color=COLORS[color_key],
            label=label,
        )
    axes[0].set_title("96-table image accuracy")
    axes[0].set_ylabel("accuracy")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.25, 1.0)
    axes[0].legend(frameon=False, loc="upper right")
    _clean_axis(axes[0])

    axes[1].bar(
        x - width / 2,
        comparison["median_trajectory_rmse"],
        width=width,
        color=COLORS["rmse"],
        label="RMSE",
    )
    axes[1].bar(
        x + width / 2,
        comparison["median_trajectory_corr_mean"],
        width=width,
        color=COLORS["corr"],
        label="corr",
    )
    axes[1].axhline(0.0, color="#9ca3af", lw=0.9)
    axes[1].set_title("96-table trace diagnostics")
    axes[1].set_xticks(x, labels)
    axes[1].legend(frameon=False, loc="upper right")
    _clean_axis(axes[1])
    fig.suptitle("No-anchor trajectory-prior checks")
    fig.savefig(OUT_DIR / "continuous_joint_subset_prior_comparison.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_subset_prior_comparison.pdf")
    plt.close(fig)


def _plot_headline_by_scale(by_scale: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    x = by_scale["prior_scale"].map(SCALE_POS).to_numpy()
    series = [
        ("zero_accuracy", "zero", "zero eye", "-"),
        ("finite_joint_accuracy", "joint", "finite joint", "-"),
        ("best_single_tau_accuracy", "best_single_tau", "best catalog", ":"),
        ("continuous_joint_accuracy", "continuous", "continuous", "-"),
    ]
    for column, color_key, label, linestyle in series:
        ax.plot(
            x,
            by_scale[column],
            marker="o",
            lw=2.0 if color_key in {"continuous", "joint"} else 1.5,
            color=COLORS[color_key],
            linestyle=linestyle,
            label=label,
        )
    ax.plot(
        x,
        by_scale["known_accuracy"],
        marker="o",
        lw=1.3,
        color=COLORS["known"],
        linestyle="--",
        label="known-eye ceiling",
    )
    ax.set_title("Headline no-anchor continuous estimator by motion scale")
    ax.set_ylabel("image identification accuracy")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.set_ylim(0.25, 1.05)
    ax.legend(frameon=False, loc="lower left")
    _clean_axis(ax)
    fig.savefig(OUT_DIR / "continuous_joint_accuracy_by_scale.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_accuracy_by_scale.pdf")
    plt.close(fig)


def _plot_recovery(recovery: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    x = recovery["prior_scale"].map(SCALE_POS).to_numpy()

    axes[0].plot(x, recovery["median_rmse"], marker="o", lw=2.0, color=COLORS["rmse"])
    axes[0].fill_between(x, recovery["q25_rmse"], recovery["q75_rmse"], color=COLORS["rmse"], alpha=0.18, lw=0)
    axes[0].set_title("Trajectory error")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    _clean_axis(axes[0])

    axes[1].axhline(0, color="#9ca3af", lw=0.9)
    axes[1].plot(x, recovery["median_corr"], marker="o", lw=2.0, color=COLORS["corr"])
    axes[1].fill_between(x, recovery["q25_corr"], recovery["q75_corr"], color=COLORS["corr"], alpha=0.18, lw=0)
    axes[1].set_title("Trajectory correlation")
    axes[1].set_ylabel("mean x/y correlation")
    axes[1].set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    axes[1].set_ylim(-0.35, 0.45)
    _clean_axis(axes[1])

    fig.suptitle("Continuous trajectory recovery (no-anchor known-start AR(1))")
    fig.savefig(OUT_DIR / "continuous_joint_trajectory_recovery.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_trajectory_recovery.pdf")
    plt.close(fig)


def _plot_conditioning(conditioning: pd.DataFrame) -> None:
    if conditioning.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    x = conditioning["prior_scale"].map(SCALE_POS).to_numpy()
    axes[0].plot(
        x,
        conditioning["median_anisotropy"],
        marker="o",
        lw=2.0,
        color=COLORS["continuous"],
    )
    axes[0].set_title("Displacement observability")
    axes[0].set_ylabel("median s2 / s1")
    axes[0].set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    axes[0].set_ylim(0.0, max(0.55, float(conditioning["median_anisotropy"].max()) * 1.2))
    _clean_axis(axes[0])

    axes[1].plot(
        x,
        conditioning["mean_rank1_fraction_lt_0p2"],
        marker="o",
        lw=2.0,
        color=COLORS["rmse"],
    )
    axes[1].set_title("Near rank-1 time bins")
    axes[1].set_ylabel("fraction with s2/s1 < 0.2")
    axes[1].set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    axes[1].set_ylim(0.0, 1.0)
    _clean_axis(axes[1])
    fig.suptitle("Local eye-response encoder conditioning")
    fig.savefig(OUT_DIR / "continuous_joint_A_conditioning.png", dpi=220)
    fig.savefig(OUT_DIR / "continuous_joint_A_conditioning.pdf")
    plt.close(fig)


def _write_readme(
    overall: pd.DataFrame,
    by_scale: pd.DataFrame,
    recovery: pd.DataFrame,
    conditioning: pd.DataFrame,
    subset_comparison: pd.DataFrame,
) -> None:
    headline = overall[
        (overall["run_slug"] == HEADLINE_RUN_SLUG)
        & (overall["observer"] == "continuous_joint_accuracy")
    ]["accuracy"].iloc[0]
    known = overall[
        (overall["run_slug"] == HEADLINE_RUN_SLUG)
        & (overall["observer"] == "known_accuracy")
    ]["accuracy"].iloc[0]
    zero = overall[
        (overall["run_slug"] == HEADLINE_RUN_SLUG)
        & (overall["observer"] == "zero_accuracy")
    ]["accuracy"].iloc[0]
    joint = overall[
        (overall["run_slug"] == HEADLINE_RUN_SLUG)
        & (overall["observer"] == "joint_accuracy")
    ]["accuracy"].iloc[0]
    text = f"""# Continuous Joint Diagnostics

Generated by `declan/figure4_active_sensing_atlas/scripts/build_panel_c_continuous_joint_checks.py`.

Inputs are the full-cache continuous-joint analyzer outputs under:

`{SOURCE_ROOT.relative_to(REPO_ROOT)}`

Headline full-cache result at likelihood scale 1.0:

```text
known-eye ceiling:                      {known:.3f}
zero-eye accuracy:                       {zero:.3f}
finite catalog joint:                    {joint:.3f}
active no-anchor continuous, k=10:       {headline:.3f}
remaining gap to known-eye ceiling:      {known - headline:.3f}
```

The active headline is the no-anchor known-start AR(1) profile. Catalog-residual
runs are retained as diagnostics because they show the benefit of local response
manifold support, but they are not the promoted no-anchor model.
"""
    if not conditioning.empty:
        text += "\nLocal encoder conditioning by scale:\n\n```text\n"
        text += conditioning.to_string(index=False)
        text += "\n```\n"
    if not subset_comparison.empty:
        text += "\nNo-anchor subset prior checks:\n\n```text\n"
        text += subset_comparison.to_string(index=False)
        text += "\n```\n"
    (OUT_DIR / "README.md").write_text(text)


def main() -> None:
    _configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, recovery_trials = _load_all()
    overall = _overall_accuracy(summary)
    by_scale = _headline_by_scale(summary)
    recovery = _headline_recovery(recovery_trials)
    conditioning = _headline_conditioning()
    subset_comparison = _subset_prior_comparison(summary, recovery_trials)

    summary.to_csv(OUT_DIR / "continuous_joint_all_summary_rows.csv", index=False)
    overall.to_csv(OUT_DIR / "continuous_joint_overall_accuracy.csv", index=False)
    by_scale.to_csv(OUT_DIR / "continuous_joint_accuracy_by_scale.csv", index=False)
    recovery.to_csv(OUT_DIR / "continuous_joint_trajectory_recovery_by_scale.csv", index=False)
    if not conditioning.empty:
        conditioning.to_csv(OUT_DIR / "continuous_joint_A_conditioning.csv", index=False)
    if not subset_comparison.empty:
        subset_comparison.to_csv(OUT_DIR / "continuous_joint_subset_prior_comparison.csv", index=False)

    _plot_overall_accuracy(overall)
    _plot_subset_prior_comparison(subset_comparison)
    _plot_headline_by_scale(by_scale)
    _plot_recovery(recovery)
    _plot_conditioning(conditioning)
    _write_readme(overall, by_scale, recovery, conditioning, subset_comparison)


if __name__ == "__main__":
    main()
