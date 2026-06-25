"""Diagnose what catalog-residual anchor smoothing changes for Figure 4C."""

from __future__ import annotations

import os
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

FULL_RUNS = {
    "unsmoothed_top2_shrink": "continuous_joint_catalog_residual_compact_k10_smoothAt_a0p70_q1em02_topk2_shrink0p19_full",
    "smooth6_top2_shrink": "continuous_joint_catalog_residual_k10_topk2_shrink0p19_anchor_smooth6_full",
    "c2f_6_0_keep16": "continuous_joint_catalog_residual_c2f_k10_sched6-0_keep16_topk2_shrink0p19_full",
    "c2f_6_0_keep8": "continuous_joint_catalog_residual_c2f_k10_sched6-0_keep8_topk2_shrink0p19_full",
    "c2f_6_0_keep4": "continuous_joint_catalog_residual_c2f_k10_sched6-0_keep4_topk2_shrink0p19_full",
}
SIGMA_RUNS = {
    0.0: "continuous_joint_catalog_residual_k10_topk2_shrink0p19_anchor_smooth0_subset96",
    0.75: "continuous_joint_catalog_residual_k10_topk2_shrink0p19_anchor_smooth0p75_subset96",
    1.5: "continuous_joint_catalog_residual_k10_topk2_shrink0p19_anchor_smooth1p5_subset96",
    3.0: "continuous_joint_catalog_residual_k10_topk2_shrink0p19_anchor_smooth3_subset96",
    6.0: "continuous_joint_catalog_residual_k10_topk2_shrink0p19_anchor_smooth6_subset96",
}
KEY = ["trial_id", "prior_family", "prior_scale", "response_cache_path"]


def _read(slug: str) -> pd.DataFrame:
    path = SOURCE_ROOT / slug / "continuous_joint_trials.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _summary_rows() -> pd.DataFrame:
    rows = []
    for label, slug in FULL_RUNS.items():
        df = _read(slug)
        rows.append(
            {
                "run": label,
                "n": int(df.shape[0]),
                "continuous_accuracy": float(df["continuous_joint_correct"].mean()),
                "zero_accuracy": float(df["zero_correct"].mean()),
                "finite_joint_accuracy": float(df["joint_correct"].mean()),
                "best_single_tau_accuracy": float(df["best_single_tau_correct"].mean()),
                "median_rmse": float(df["trajectory_rmse"].median()),
                "median_corr": float(df["trajectory_corr_mean"].median()),
                "median_anchor_score_gap": float(df["catalog_residual_true_anchor_score_gap"].median()),
                "median_score_corr_with_zero": float(df["continuous_joint_score_corr_with_zero"].median()),
            }
        )
    return pd.DataFrame(rows)


def _sigma_rows() -> pd.DataFrame:
    rows = []
    for sigma, slug in sorted(SIGMA_RUNS.items()):
        df = _read(slug)
        rows.append(
            {
                "sigma": float(sigma),
                "n": int(df.shape[0]),
                "continuous_accuracy": float(df["continuous_joint_correct"].mean()),
                "median_rmse": float(df["trajectory_rmse"].median()),
                "median_corr": float(df["trajectory_corr_mean"].median()),
                "median_anchor_score_gap": float(df["catalog_residual_true_anchor_score_gap"].median()),
                "median_true_margin": float(df["continuous_joint_true_margin"].median()),
            }
        )
    return pd.DataFrame(rows)


def _transition_rows() -> pd.DataFrame:
    base = _read(FULL_RUNS["unsmoothed_top2_shrink"])
    smooth = _read(FULL_RUNS["smooth6_top2_shrink"])
    cols = KEY + [
        "true_image_id",
        "continuous_joint_correct",
        "continuous_joint_pred_image_id",
        "continuous_joint_true_margin",
        "catalog_residual_true_best_anchor_index",
        "catalog_residual_true_anchor_score_gap",
        "trajectory_rmse",
        "trajectory_corr_mean",
        "nearest_trajectory_distance",
    ]
    merged = base[cols].merge(smooth[cols], on=KEY, suffixes=("_unsmoothed", "_smooth6"))
    merged["prediction_changed"] = merged["continuous_joint_pred_image_id_unsmoothed"].ne(
        merged["continuous_joint_pred_image_id_smooth6"]
    )
    merged["anchor_changed"] = merged["catalog_residual_true_best_anchor_index_unsmoothed"].ne(
        merged["catalog_residual_true_best_anchor_index_smooth6"]
    )
    merged["rmse_improved"] = merged["trajectory_rmse_smooth6"].lt(merged["trajectory_rmse_unsmoothed"])
    merged["corr_improved"] = merged["trajectory_corr_mean_smooth6"].gt(merged["trajectory_corr_mean_unsmoothed"])
    return merged


def _plot_summary(summary: pd.DataFrame, sigma: pd.DataFrame, transitions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), constrained_layout=True)
    order = ["unsmoothed_top2_shrink", "smooth6_top2_shrink", "c2f_6_0_keep16", "c2f_6_0_keep8", "c2f_6_0_keep4"]
    block = summary.set_index("run").loc[order].reset_index()
    x = np.arange(block.shape[0], dtype=float)
    labels = ["unsmoothed", "smooth6", "CTF keep16", "CTF keep8", "CTF keep4"]
    axes[0].bar(x - 0.17, block["continuous_accuracy"], width=0.34, color="#2f8f6a", label="accuracy")
    axes[0].bar(x + 0.17, block["median_rmse"], width=0.34, color="#8a5ca8", label="median RMSE")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylim(0.0, 0.95)
    axes[0].set_title("Full-cache anchor variants")
    axes[0].legend(frameon=False)

    axes[1].plot(sigma["sigma"], sigma["median_rmse"], marker="o", color="#8a5ca8", label="median RMSE")
    axes[1].plot(sigma["sigma"], sigma["continuous_accuracy"], marker="o", color="#2f8f6a", label="accuracy")
    axes[1].set_xlabel("anchor smoothing sigma")
    axes[1].set_title("Subset smoothing sweep")
    axes[1].legend(frameon=False)

    axes[2].scatter(
        transitions["trajectory_rmse_unsmoothed"],
        transitions["trajectory_rmse_smooth6"],
        s=12,
        color="#235789",
        alpha=0.45,
        linewidths=0,
    )
    lim = float(np.nanpercentile(transitions[["trajectory_rmse_unsmoothed", "trajectory_rmse_smooth6"]], 99))
    axes[2].plot([0, lim], [0, lim], color="#111827", lw=1.0, linestyle="--")
    axes[2].set_xlim(0, lim)
    axes[2].set_ylim(0, lim)
    axes[2].set_xlabel("unsmoothed RMSE")
    axes[2].set_ylabel("smooth6 RMSE")
    axes[2].set_title("Smoothing improves trace RMSE")
    for ax in axes:
        ax.grid(axis="y", color="#d9dee5", lw=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Catalog-residual anchor smoothing diagnostics")
    fig.savefig(OUT_DIR / "catalog_residual_anchor_smoothing_diagnostics.png", dpi=220)
    fig.savefig(OUT_DIR / "catalog_residual_anchor_smoothing_diagnostics.pdf")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = _summary_rows()
    sigma = _sigma_rows()
    transitions = _transition_rows()
    summary.to_csv(OUT_DIR / "catalog_residual_anchor_smoothing_summary.csv", index=False)
    sigma.to_csv(OUT_DIR / "catalog_residual_anchor_smoothing_sigma_sweep_subset96.csv", index=False)
    transitions.to_csv(OUT_DIR / "catalog_residual_anchor_smoothing_trial_transitions.csv", index=False)
    transitions[transitions["prediction_changed"]].to_csv(
        OUT_DIR / "catalog_residual_anchor_smoothing_changed_predictions.csv",
        index=False,
    )
    _plot_summary(summary, sigma, transitions)
    readme = [
        "# Catalog-Residual Anchor Smoothing Diagnostics",
        "",
        "Anchor smoothing mostly improves recovered trajectory RMSE while leaving image decisions nearly unchanged.",
        f"Smooth6 changes the predicted image on {int(transitions['prediction_changed'].sum())}/{transitions.shape[0]} tables.",
        f"The true-candidate best anchor changes on {int(transitions['anchor_changed'].sum())}/{transitions.shape[0]} tables.",
        f"RMSE improves on {100.0 * float(transitions['rmse_improved'].mean()):.1f}% of tables.",
        "",
        "Primary files:",
        "- `catalog_residual_anchor_smoothing_summary.csv`",
        "- `catalog_residual_anchor_smoothing_sigma_sweep_subset96.csv`",
        "- `catalog_residual_anchor_smoothing_trial_transitions.csv`",
        "- `catalog_residual_anchor_smoothing_changed_predictions.csv`",
        "- `catalog_residual_anchor_smoothing_diagnostics.png`",
    ]
    (OUT_DIR / "catalog_residual_anchor_smoothing_README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
