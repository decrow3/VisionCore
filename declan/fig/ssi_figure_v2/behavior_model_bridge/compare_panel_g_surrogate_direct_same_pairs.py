#!/usr/bin/env python3
"""Decompose the old-to-new Panel G change into cohort and estimator effects.

The old result is a one-dimensional dose-curve surrogate.  This script applies
that exact surrogate to the source rows used by the 1,000-pair direct run,
using both the original 256 random rotations and the eight fixed rotation
angles used by the direct evaluator.  It then joins the surrogate and direct
effects pair by pair.  Magnitudes are not commensurate: the surrogate is
percent change from a stabilized baseline, while the direct effect is absolute
bits/spike (real trajectory minus rotated-trajectory mean).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2.behavior_model_bridge import run_behavior_model_bridge as bridge
from declan.fig.ssi_figure_v2.behavior_model_bridge import run_random_rotation_match_null as rotation_null
from declan.fig.ssi_figure_v2.behavior_model_bridge import run_random_rotation_prediction_by_coherence as old_panel


RUN_ROOT = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_model_bridge"
    / "panel_g_exact_pair_fig4_trace_bank_n1000_v1"
)
OUT_DIR = RUN_ROOT / "old_surrogate_vs_direct_audit"
COHORT_CSV = RUN_ROOT / "exact_pair_cohort_manifest.csv"
DIRECT_CSV = RUN_ROOT / "checkpoint1_production_readout/checkpoint1_aligned_pair_metrics.csv"
OLD_SUMMARY_CSV = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_model_bridge"
    / "behavior_model_bridge_random_rotation_prediction_by_coherence_summary.csv"
)

COHERENCE_ORDER = list(bridge.COHERENCE_ORDER)
COHERENCE_EDGES = [-np.inf, 0.2, 0.5, 0.8, np.inf]
DIRECT_ANGLES_RAD = np.radians([22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5])


def _surrogate_pair(
    row: pd.Series,
    *,
    angles_rad: np.ndarray,
    curves: dict[str, pd.DataFrame],
) -> tuple[float, float, float]:
    coords = rotation_null._axis_coordinates(row)
    if coords is None:
        return math.nan, math.nan, math.nan
    pos_along, pos_across, step_along, step_across = coords
    observed = rotation_null._component_metrics_for_angles(
        pos_along, pos_across, step_along, step_across, np.zeros(1, dtype=float)
    )
    rotated = rotation_null._component_metrics_for_angles(
        pos_along, pos_across, step_along, step_across, angles_rad
    )
    component_effects = []
    for component in ("along", "across"):
        curve = curves[component]
        obs, _ = bridge._interpolate_curve(observed[("component_rms", component)], curve)
        rot, _ = bridge._interpolate_curve(rotated[("component_rms", component)], curve)
        if obs.size and np.isfinite(obs[0]) and np.isfinite(rot).any():
            component_effects.append(float(obs[0] - np.nanmean(rot)))
        else:
            component_effects.append(math.nan)
    return (
        float(np.nanmean(component_effects)) if np.isfinite(component_effects).any() else math.nan,
        component_effects[0],
        component_effects[1],
    )


def _bin_summary(pair_table: pd.DataFrame, old_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    old_sel = old_summary[
        old_summary["score_type"].astype(str).eq("component_mean_marginal")
        & old_summary["population_key"].astype(str).eq("high_sf_aligned")
        & old_summary["metric_family"].astype(str).eq("component_rms")
    ].set_index("coherence_bin")
    for label in COHERENCE_ORDER:
        sub = pair_table[pair_table["coherence_bin"].astype(str).eq(label)]
        rows.append(
            {
                "coherence_bin": label,
                "n_full_old_cohort": int(old_sel.loc[label, "n_windows"]),
                "old_surrogate_full_cohort_percent_points": float(
                    old_sel.loc[label, "observed_minus_rotated"]
                ),
                "n_exact_pair_cohort": int(len(sub)),
                "old_surrogate_exact_cohort_256rot_percent_points": float(
                    sub["surrogate_effect_256rot_percent_points"].mean()
                ),
                "old_surrogate_exact_cohort_8matched_percent_points": float(
                    sub["surrogate_effect_8matched_percent_points"].mean()
                ),
                "direct_exact_cohort_bits_per_spike": float(
                    sub["direct_effect_bits_per_spike"].mean()
                ),
                "direct_exact_cohort_median_bits_per_spike": float(
                    sub["direct_effect_bits_per_spike"].median()
                ),
                "direct_fraction_positive": float(
                    (sub["direct_effect_bits_per_spike"] > 0).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(bridge.BEHAVIOR_WINDOWS_CSV)
    cohort = pd.read_csv(COHORT_CSV)
    direct = pd.read_csv(DIRECT_CSV)
    old_summary = pd.read_csv(OLD_SUMMARY_CSV)
    model_values = pd.read_csv(bridge.MODEL_VALUES_CSV)

    curves = {
        component: bridge._curve_for(
            model_values,
            population_key="high_sf_aligned",
            metric_family="component_rms",
            component=component,
        )
        for component in ("along", "across")
    }

    # Recreate the original random-angle array so each selected source row gets
    # precisely the angles it had in the historical full-cohort calculation.
    rng = np.random.default_rng(old_panel.SEED)
    original_angles = rng.uniform(0.0, np.pi, size=(len(windows), old_panel.DEFAULT_N_ROTATIONS))

    rows = []
    for item in cohort.itertuples(index=False):
        source_row = int(item.source_row)
        behavior_row = windows.iloc[source_row]
        effect_256, along_256, across_256 = _surrogate_pair(
            behavior_row,
            angles_rad=original_angles[source_row],
            curves=curves,
        )
        effect_8, along_8, across_8 = _surrogate_pair(
            behavior_row,
            angles_rad=DIRECT_ANGLES_RAD,
            curves=curves,
        )
        rows.append(
            {
                "pair_index": int(item.pair_index),
                "source_row": source_row,
                "session": str(item.session),
                "image_orientation_coherence": float(item.image_orientation_coherence),
                "surrogate_effect_256rot_percent_points": effect_256,
                "surrogate_along_effect_256rot_percent_points": along_256,
                "surrogate_across_effect_256rot_percent_points": across_256,
                "surrogate_effect_8matched_percent_points": effect_8,
                "surrogate_along_effect_8matched_percent_points": along_8,
                "surrogate_across_effect_8matched_percent_points": across_8,
            }
        )
    pairs = pd.DataFrame(rows)
    direct_cols = [
        "pair_index",
        "real_minus_rotation_bits_per_spike",
        "real_minus_rotation_information_bits_per_sample",
    ]
    geometry_cols = [
        "pair_index",
        "parallel_rms_arcmin",
        "normal_rms_arcmin",
        "parallel_minus_normal_rms_arcmin",
        "trace_covariance_anisotropy",
    ]
    pairs = pairs.merge(
        direct[list(dict.fromkeys(direct_cols + geometry_cols))],
        on="pair_index",
        validate="one_to_one",
    )
    pairs = pairs.rename(
        columns={
            "real_minus_rotation_bits_per_spike": "direct_effect_bits_per_spike",
            "real_minus_rotation_information_bits_per_sample": "direct_effect_information_bits_per_sample",
        }
    )
    pairs["coherence_bin"] = pd.cut(
        pairs["image_orientation_coherence"],
        bins=COHERENCE_EDGES,
        labels=COHERENCE_ORDER,
        right=False,
    )
    finite_comparison = (
        pairs["surrogate_effect_8matched_percent_points"].notna()
        & pairs["direct_effect_bits_per_spike"].notna()
    )
    pairs["surrogate_direct_sign_match"] = pd.Series(pd.NA, index=pairs.index, dtype="boolean")
    pairs.loc[finite_comparison, "surrogate_direct_sign_match"] = (
        np.sign(pairs.loc[finite_comparison, "surrogate_effect_8matched_percent_points"])
        == np.sign(pairs.loc[finite_comparison, "direct_effect_bits_per_spike"])
    )

    bins = _bin_summary(pairs, old_summary)
    metrics = {
        "n_pairs": int(len(pairs)),
        "n_surrogate_finite_256": int(pairs["surrogate_effect_256rot_percent_points"].notna().sum()),
        "surrogate_256_vs_8_pearson": float(
            pairs["surrogate_effect_256rot_percent_points"].corr(
                pairs["surrogate_effect_8matched_percent_points"]
            )
        ),
        "surrogate_256_vs_8_spearman": float(
            pairs["surrogate_effect_256rot_percent_points"].corr(
                pairs["surrogate_effect_8matched_percent_points"], method="spearman"
            )
        ),
        "surrogate_vs_direct_pearson": float(
            pairs["surrogate_effect_8matched_percent_points"].corr(
                pairs["direct_effect_bits_per_spike"]
            )
        ),
        "surrogate_vs_direct_spearman": float(
            pairs["surrogate_effect_8matched_percent_points"].corr(
                pairs["direct_effect_bits_per_spike"], method="spearman"
            )
        ),
        "surrogate_direct_sign_match_fraction": float(pairs["surrogate_direct_sign_match"].mean()),
        "surrogate_vs_parallel_minus_normal_rms_spearman": float(
            pairs["surrogate_effect_8matched_percent_points"].corr(
                pairs["parallel_minus_normal_rms_arcmin"], method="spearman"
            )
        ),
        "direct_vs_parallel_minus_normal_rms_spearman": float(
            pairs["direct_effect_bits_per_spike"].corr(
                pairs["parallel_minus_normal_rms_arcmin"], method="spearman"
            )
        ),
        "surrogate_vs_coherence_spearman": float(
            pairs["surrogate_effect_8matched_percent_points"].corr(
                pairs["image_orientation_coherence"], method="spearman"
            )
        ),
        "direct_vs_coherence_spearman": float(
            pairs["direct_effect_bits_per_spike"].corr(
                pairs["image_orientation_coherence"], method="spearman"
            )
        ),
    }

    pairs.to_csv(OUT_DIR / "old_surrogate_vs_direct_pair_table.csv", index=False)
    bins.to_csv(OUT_DIR / "old_surrogate_vs_direct_coherence_bins.csv", index=False)
    (OUT_DIR / "old_surrogate_vs_direct_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7), constrained_layout=True)
    x = np.arange(len(bins))
    axes[0].plot(x, bins["old_surrogate_full_cohort_percent_points"], "o-", label="old cohort (n=11,749)")
    axes[0].plot(x, bins["old_surrogate_exact_cohort_256rot_percent_points"], "s-", label="same 1,000 rows")
    axes[0].axhline(0, color="0.5", lw=0.8)
    axes[0].set_xticks(x, bins["coherence_bin"], rotation=25)
    axes[0].set_ylabel("old surrogate match advantage (percentage points)")
    axes[0].set_title("Cohort change, estimator held fixed")
    axes[0].legend(frameon=False, fontsize=7)

    axes[1].scatter(
        pairs["surrogate_effect_8matched_percent_points"],
        pairs["direct_effect_bits_per_spike"],
        c=pairs["image_orientation_coherence"], cmap="viridis", s=12, alpha=0.45,
    )
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].axvline(0, color="0.5", lw=0.8)
    axes[1].set_xlabel("old surrogate, same 8 rotations (percentage points)")
    axes[1].set_ylabel("fresh direct effect (bits/spike)")
    axes[1].set_title(f"Estimator change on identical pairs\nSpearman r={metrics['surrogate_vs_direct_spearman']:.3f}")

    axes[2].plot(x, bins["direct_exact_cohort_bits_per_spike"], "o-", color="#7351a3")
    axes[2].axhline(0, color="0.5", lw=0.8)
    axes[2].set_xticks(x, bins["coherence_bin"], rotation=25)
    axes[2].set_ylabel("fresh direct match effect (bits/spike)")
    axes[2].set_title("Exact image-trajectory evaluation")
    fig.savefig(OUT_DIR / "old_surrogate_vs_direct_decomposition.png", dpi=220)
    fig.savefig(OUT_DIR / "old_surrogate_vs_direct_decomposition.pdf")
    plt.close(fig)

    print(bins.to_string(index=False))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
