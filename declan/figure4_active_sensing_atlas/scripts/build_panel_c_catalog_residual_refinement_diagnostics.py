"""Diagnose why catalog-residual scoring beats finite trajectory scoring."""

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
CATALOG_RUNS = {
    "all_anchor_logmean": "continuous_joint_catalog_residual_compact_k10_smoothAt_a0p70_q1em02_full",
    "top2_logmean": "continuous_joint_catalog_residual_compact_k10_smoothAt_a0p70_q1em02_topk2_full",
    "top2_shrink0p19": "continuous_joint_catalog_residual_compact_k10_smoothAt_a0p70_q1em02_topk2_shrink0p19_full",
    "top2_a0p85": "continuous_joint_catalog_residual_compact_k10_smoothAt_a0p85_q1em02_topk2_full",
}
HEADLINE_RUN = CATALOG_RUNS["top2_shrink0p19"]
KEY = ["trial_id", "prior_family", "prior_scale", "response_cache_path"]


def _read(slug: str, name: str = "continuous_joint_trials.csv") -> pd.DataFrame:
    path = SOURCE_ROOT / slug / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _headline_lift_trials() -> pd.DataFrame:
    trials = _read(HEADLINE_RUN)
    features = _read(HEADLINE_RUN, "continuous_joint_feature_posterior.csv")
    wide = (
        features[features["observer_mode"].isin(["best_single_tau", "continuous_joint"])]
        .pivot_table(
            index=["table_index", "response_cache_path", "candidate_index", "candidate_id", "is_true_candidate"],
            columns="observer_mode",
            values="candidate_score",
            aggfunc="first",
        )
        .reset_index()
    )
    wide["residual_lift"] = wide["continuous_joint"] - wide["best_single_tau"]
    rows = []
    for (table_index, path), group in wide.groupby(["table_index", "response_cache_path"], sort=False):
        true = group[group["is_true_candidate"].astype(bool)].iloc[0]
        competitors = group[~group["is_true_candidate"].astype(bool)]
        best_comp = competitors.loc[competitors["continuous_joint"].idxmax()]
        rows.append(
            {
                "table_index": int(table_index),
                "response_cache_path": str(path),
                "true_residual_lift": float(true["residual_lift"]),
                "continuous_top_comp_residual_lift": float(best_comp["residual_lift"]),
                "true_lift_minus_top_comp_lift": float(true["residual_lift"] - best_comp["residual_lift"]),
            }
        )
    lift = pd.DataFrame(rows)
    merged = trials.merge(lift, on=["table_index", "response_cache_path"], how="left")
    merged["transition_best_to_residual"] = np.select(
        [
            merged["best_single_tau_correct"] & merged["continuous_joint_correct"],
            merged["best_single_tau_correct"] & ~merged["continuous_joint_correct"],
            ~merged["best_single_tau_correct"] & merged["continuous_joint_correct"],
            ~merged["best_single_tau_correct"] & ~merged["continuous_joint_correct"],
        ],
        ["both_correct", "lost_by_residual", "rescued_by_residual", "both_wrong"],
        default="unknown",
    )
    merged["margin_gain_vs_best_single"] = merged["continuous_joint_true_margin"] - merged["best_single_tau_true_margin"]
    return merged


def _aggregation_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    wide: pd.DataFrame | None = None
    base_cols = [
        "continuous_joint_correct",
        "continuous_joint_pred_image_id",
        "continuous_joint_true_margin",
        "continuous_joint_score_corr_with_zero",
        "trajectory_rmse",
        "nearest_trajectory_distance",
    ]
    optional = ["catalog_residual_true_anchor_score_gap"]
    for label, slug in CATALOG_RUNS.items():
        df = _read(slug)
        for col in optional:
            if col not in df.columns:
                df[col] = np.nan
        df = df.copy()
        df["run"] = label
        frames.append(df)
        sub = df[KEY + base_cols + optional].rename(columns={col: f"{col}_{label}" for col in base_cols + optional})
        wide = sub if wide is None else wide.merge(sub, on=KEY, how="inner")
    all_rows = pd.concat(frames, ignore_index=True)
    summary = (
        all_rows.groupby("run", sort=False)
        .agg(
            n=("continuous_joint_correct", "size"),
            accuracy=("continuous_joint_correct", "mean"),
            median_true_margin=("continuous_joint_true_margin", "median"),
            median_score_corr_with_zero=("continuous_joint_score_corr_with_zero", "median"),
            median_anchor_score_gap=("catalog_residual_true_anchor_score_gap", "median"),
            median_rmse=("trajectory_rmse", "median"),
        )
        .reset_index()
    )
    assert wide is not None
    return summary, wide


def _transition_counts(frame: pd.DataFrame, a: str, b: str) -> dict[str, int]:
    a_col = f"continuous_joint_correct_{a}"
    b_col = f"continuous_joint_correct_{b}"
    return {
        "both_correct": int((frame[a_col] & frame[b_col]).sum()),
        "lost": int((frame[a_col] & ~frame[b_col]).sum()),
        "rescued": int((~frame[a_col] & frame[b_col]).sum()),
        "both_wrong": int((~frame[a_col] & ~frame[b_col]).sum()),
    }


def _plot(lift: pd.DataFrame, agg_summary: pd.DataFrame, agg_transitions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), constrained_layout=True)

    transition_order = ["both_correct", "rescued_by_residual", "lost_by_residual", "both_wrong"]
    transition_labels = ["both\ncorrect", "rescued", "lost", "both\nwrong"]
    counts = lift["transition_best_to_residual"].value_counts().reindex(transition_order).fillna(0)
    axes[0].bar(np.arange(len(counts)), counts.to_numpy(), color=["#235789", "#2f8f6a", "#b35c2e", "#6b7280"])
    axes[0].set_xticks(np.arange(len(counts)), transition_labels)
    axes[0].set_ylabel("tables")
    axes[0].set_title("Finite best -> residual")

    order = ["all_anchor_logmean", "top2_logmean", "top2_shrink0p19", "top2_a0p85"]
    block = agg_summary.set_index("run").loc[order].reset_index()
    axes[1].bar(np.arange(block.shape[0]), block["accuracy"], color="#2f8f6a")
    axes[1].set_xticks(
        np.arange(block.shape[0]),
        ["all anchors", "top-2", "top-2\nshrink", "top-2\nAR retune"],
        rotation=18,
        ha="right",
    )
    axes[1].set_ylim(0.72, 0.88)
    axes[1].set_title("Anchor aggregation")
    axes[1].set_ylabel("accuracy")

    data = [
        lift.loc[lift["transition_best_to_residual"].eq(name), "nearest_trajectory_distance"].dropna()
        for name in transition_order
    ]
    axes[2].boxplot(data, tick_labels=transition_labels, showfliers=False, patch_artist=True)
    for patch, color in zip(axes[2].artists, ["#235789", "#2f8f6a", "#b35c2e", "#6b7280"], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    axes[2].set_title("Rescues occur farther from catalog")
    axes[2].set_ylabel("nearest trace distance")

    for ax in axes:
        ax.grid(axis="y", color="#d9dee5", lw=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Catalog-residual refinement diagnostics")
    fig.savefig(OUT_DIR / "catalog_residual_refinement_diagnostics.png", dpi=220)
    fig.savefig(OUT_DIR / "catalog_residual_refinement_diagnostics.pdf")
    plt.close(fig)

    trans_rows = []
    for a, b in [
        ("all_anchor_logmean", "top2_logmean"),
        ("top2_logmean", "top2_shrink0p19"),
        ("all_anchor_logmean", "top2_shrink0p19"),
    ]:
        row = {"comparison": f"{a}_to_{b}"}
        row.update(_transition_counts(agg_transitions, a, b))
        trans_rows.append(row)
    pd.DataFrame(trans_rows).to_csv(OUT_DIR / "catalog_residual_aggregation_transition_counts.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lift = _headline_lift_trials()
    agg_summary, agg_transitions = _aggregation_summary()
    lift.to_csv(OUT_DIR / "catalog_residual_refinement_lift_trials.csv", index=False)
    agg_summary.to_csv(OUT_DIR / "catalog_residual_aggregation_calibration_summary.csv", index=False)
    agg_transitions.to_csv(OUT_DIR / "catalog_residual_aggregation_calibration_transitions.csv", index=False)
    _plot(lift, agg_summary, agg_transitions)

    counts = lift["transition_best_to_residual"].value_counts()
    readme = [
        "# Catalog-Residual Refinement Diagnostics",
        "",
        "Catalog-residual scoring improves over the finite best-trajectory observer mostly by rescuing finite-catalog misses.",
        f"Best-single -> catalog-residual rescued: {int(counts.get('rescued_by_residual', 0))}",
        f"Best-single -> catalog-residual lost: {int(counts.get('lost_by_residual', 0))}",
        "",
        "Aggregation/calibration also matters: all-anchor log-mean is weaker than top-2, and a small shrink toward all-anchor improves full-cache accuracy.",
        "",
        "Primary files:",
        "- `catalog_residual_refinement_lift_trials.csv`",
        "- `catalog_residual_aggregation_calibration_summary.csv`",
        "- `catalog_residual_aggregation_transition_counts.csv`",
        "- `catalog_residual_refinement_diagnostics.png`",
    ]
    (OUT_DIR / "catalog_residual_refinement_README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
