"""Build the feature-posterior compact-mechanism panel for Figure 4.

This renders the cache-first hard-negative n128 scale sweep for
``pyramid_local_field`` PCA ``k=8`` after applying the compact-subspace
intervention in the same feature-posterior metric used by Panel C.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
ATLAS = REPO_ROOT / "declan" / "figure4_active_sensing_atlas"
OUT_DIR = ATLAS / "figures" / "panel_C" / "promotion_candidates"
SOURCE_DIR = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1"
)
SUMMARY_CSV = SOURCE_DIR / "feature_compact_mechanism_summary.csv"

COLORS = {
    "parallel": "#2f8f6a",
    "orthogonal": "#8063a6",
    "zero": "#30363d",
    "grid": "#d8dde3",
}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.6,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _selected_rows() -> pd.DataFrame:
    rows = pd.read_csv(SUMMARY_CSV)
    selected = rows[
        (rows["latent"] == "pyramid_local_field")
        & (rows["requested_k"].astype(int) == 8)
        & (rows["k_dim"].astype(int) == 10)
        & (rows["candidate_set_mode"] == "hard_negative_structure")
        & (
            rows["response_variant"].isin(
                ["zero_static", "compact_only", "compact_removed", "known_eye", "full_exact", "compact_addback"]
            )
        )
    ].copy()
    selected["scale_label"] = selected["observation_scale"].map({0.5: "0.5x", 1.0: "1x", 2.0: "2x"})
    selected["axis_label"] = selected["prior_family"].map(
        {
            "axis_edge_parallel": "along local edge",
            "axis_edge_orthogonal": "across local edge",
        }
    )
    selected = selected.sort_values(["prior_family", "observation_scale", "response_variant"])
    if len(selected) != 36:
        raise ValueError(f"Expected 36 primary feature-compact rows, found {len(selected)}")
    return selected


def build() -> list[Path]:
    _configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _selected_rows()

    summary = (
        rows.groupby(["observation_scale", "response_variant"], as_index=False)
        .agg(mean_feature_cosine=("mean_feature_cosine", "mean"))
        .pivot(index="observation_scale", columns="response_variant", values="mean_feature_cosine")
        .reset_index()
        .rename(
            columns={
                "zero_static": "zero_mean_cosine",
                "compact_only": "compact_subspace_mean_cosine",
                "compact_removed": "compact_removed_mean_cosine",
                "known_eye": "known_mean_cosine",
                "full_exact": "full_joint_mean_cosine",
                "compact_addback": "compact_addback_mean_cosine",
            }
        )
        .sort_values("observation_scale")
    )

    fig, ax = plt.subplots(figsize=(4.35, 3.15), constrained_layout=True)
    x_map = {0.5: 0, 1.0: 1, 2.0: 2}
    x = summary["observation_scale"].map(x_map).astype(float).to_numpy()
    zero = summary["zero_mean_cosine"].to_numpy(dtype=float)
    compact = summary["compact_subspace_mean_cosine"].to_numpy(dtype=float)
    removed = summary["compact_removed_mean_cosine"].to_numpy(dtype=float)
    known = summary["known_mean_cosine"].to_numpy(dtype=float)

    ax.plot(x, zero, color=COLORS["zero"], marker="o", lw=2.0, label="zero eye")
    ax.plot(x, compact, color=COLORS["parallel"], marker="o", lw=2.2, label="compact subspace")
    ax.plot(x, removed, color=COLORS["orthogonal"], marker="o", lw=2.0, label="compact removed")
    ax.plot(x, known, color="#1f252b", lw=1.4, linestyle=":", label="known eye")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.set_xlabel("motion scale")
    ax.set_ylabel("feature recovery (cosine)")
    ax.set_ylim(0.48, 0.98)
    ax.set_title("Compact subspace carries feature recovery")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    out = OUT_DIR / "4C_candidate_5_joint_feature_posterior_recovery.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    values = summary.assign(
        scale_label=summary["observation_scale"].map({0.5: "0.5x", 1.0: "1x", 2.0: "2x"}),
        latent="pyramid_local_field",
        requested_k=8,
        k_dim=10,
        candidate_set_mode="hard_negative_structure",
        selected_option="4C_option_6_feature_space_compact_removed",
    )
    values.to_csv(OUT_DIR / "4C_candidate_5_joint_feature_posterior_recovery_values.csv", index=False)
    return [out, out.with_suffix(".pdf"), OUT_DIR / "4C_candidate_5_joint_feature_posterior_recovery_values.csv"]


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()
