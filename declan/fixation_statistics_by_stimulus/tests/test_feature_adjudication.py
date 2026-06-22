from __future__ import annotations

import pandas as pd

from declan.fixation_statistics_by_stimulus.analyze_backimage_feature_decomposition_adjudication import (
    build_branch_metrics,
)


def test_branch_metrics_preserve_same_cell_from_distinct_source_files() -> None:
    inventory = pd.DataFrame(
        [
            {
                "branch": "local_Iz",
                "table_role": "incremental_gain_contrasts",
                "latent": "pyramid_local_field",
                "k": 16,
                "response_summary": "delta_mean",
                "scale": "rel_0p25x",
                "control_contrast": "actual_paired_empirical-brownian_matched_actual",
                "estimate": 2.0,
                "ci_low": 0.5,
                "ci_high": 3.5,
                "p_value": float("nan"),
                "source_file": "seed7/incremental_gain_contrasts.csv",
                "feature_space": "selected_windows_zscore_pca",
            },
            {
                "branch": "local_Iz",
                "table_role": "incremental_gain_contrasts",
                "latent": "pyramid_local_field",
                "k": 16,
                "response_summary": "delta_mean",
                "scale": "rel_0p25x",
                "control_contrast": "actual_paired_empirical-brownian_matched_actual",
                "estimate": 4.0,
                "ci_low": 1.0,
                "ci_high": 7.0,
                "p_value": float("nan"),
                "source_file": "seed11/incremental_gain_contrasts.csv",
                "feature_space": "selected_windows_zscore_pca",
            },
            {
                "branch": "local_Iz",
                "table_role": "incremental_gain_contrasts",
                "latent": "pyramid_local_field",
                "k": 16,
                "response_summary": "delta_mean",
                "scale": "rel_0p25x",
                "control_contrast": "actual_paired_empirical-brownian_matched_actual",
                "estimate": 4.0,
                "ci_low": 1.0,
                "ci_high": 7.0,
                "p_value": float("nan"),
                "source_file": "seed11/incremental_gain_contrasts.csv",
                "feature_space": "selected_windows_zscore_pca",
            },
        ]
    )

    metrics = build_branch_metrics(
        inventory,
        requested_latents=["pyramid_local_field"],
        requested_k=[16],
        requested_summaries=["delta_mean"],
        primary_scales=["rel_0p25x"],
        sentinel_scales=[],
    )

    assert len(metrics) == 2
    assert set(metrics["source_file"]) == {
        "seed7/incremental_gain_contrasts.csv",
        "seed11/incremental_gain_contrasts.csv",
    }


def test_branch_metrics_collapse_duplicate_joint_metrics_across_source_tables() -> None:
    inventory = pd.DataFrame(
        [
            {
                "branch": "joint_posterior",
                "table_role": "feature_axis_contrasts",
                "latent": "pyramid_local_field",
                "k": 16,
                "response_summary": "joint_parallel_minus_orthogonal",
                "scale": "rel_0p5x",
                "control_contrast": "joint_parallel_minus_orthogonal",
                "estimate": 0.7,
                "ci_low": -1.0,
                "ci_high": 2.0,
                "p_value": 0.5,
                "source_file": "feature_axis_contrasts.csv",
                "feature_space": "selected_windows_zscore_pca",
            },
            {
                "branch": "joint_posterior",
                "table_role": "feature_axis_contrasts",
                "latent": "pyramid_local_field",
                "k": 16,
                "response_summary": "joint_parallel_minus_orthogonal",
                "scale": "rel_0p5x",
                "control_contrast": "joint_parallel_minus_orthogonal",
                "estimate": 0.7,
                "ci_low": -1.0,
                "ci_high": 2.0,
                "p_value": 0.5,
                "source_file": "feature_posterior_uncertainty.csv",
                "feature_space": "selected_windows_zscore_pca",
            },
        ]
    )

    metrics = build_branch_metrics(
        inventory,
        requested_latents=["pyramid_local_field"],
        requested_k=[16],
        requested_summaries=["delta_mean"],
        primary_scales=["rel_0p5x"],
        sentinel_scales=[],
    )

    assert len(metrics) == 1
    assert metrics.iloc[0]["metric"] == "joint_parallel_minus_orthogonal"
