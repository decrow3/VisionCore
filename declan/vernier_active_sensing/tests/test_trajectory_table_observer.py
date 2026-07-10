"""Focused tests for the exact cached-trajectory Vernier observer."""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from declan.vernier_active_sensing.run_rr100_noisy_trajectory_observer import (
    _cache_tables,
    _score_condition_poisson,
    _static_baseline_mask,
)
from declan.vernier_active_sensing.run_rr100_heldout_trajectory_observer import (
    _trajectory_log_weights_for_observations,
    split_trace_indices,
)

from declan.vernier_active_sensing.trajectory_table_observer import (
    score_trajectory_table_vernier_observer_trial,
    summarize_trajectory_table_rows,
    trajectory_gaussian_log_weights,
)


def _fixture() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    plus = np.asarray(
        [
            [[5.0, 1.0], [6.0, 1.0]],
            [[4.0, 1.0], [5.0, 1.0]],
        ],
        dtype=np.float64,
    )
    minus = np.asarray(
        [
            [[1.0, 5.0], [1.0, 6.0]],
            [[1.0, 4.0], [1.0, 5.0]],
        ],
        dtype=np.float64,
    )
    zero = {
        "plus": np.asarray([[3.0, 3.0], [3.0, 3.0]], dtype=np.float64),
        "minus": np.asarray([[3.0, 3.0], [3.0, 3.0]], dtype=np.float64),
    }
    return {"plus": plus, "minus": minus}, zero


def test_heldout_split_keeps_observations_out_of_prior_pool() -> None:
    obs, prior = split_trace_indices(
        20,
        n_observation_traces=5,
        n_prior_traces=8,
        seed=11,
    )
    assert obs.shape == (5,)
    assert prior.shape == (8,)
    assert not set(obs.tolist()).intersection(prior.tolist())


def test_heldout_sigma_zero_uses_nearest_retained_prior_trace() -> None:
    observed = np.asarray([[[0.1, 0.0], [0.1, 0.2]]], dtype=np.float64)
    prior = np.asarray(
        [
            [[2.0, 0.0], [2.0, 0.2]],
            [[0.0, 0.0], [0.0, 0.2]],
            [[5.0, 0.0], [5.0, 0.2]],
        ],
        dtype=np.float64,
    )
    logw, dist2 = _trajectory_log_weights_for_observations(
        observed,
        prior,
        sigma_arcmin=0.0,
    )
    assert int(np.argmax(logw[0])) == 1
    assert np.isfinite(logw[0, 1])
    assert np.isneginf(logw[0, 0])
    assert np.isneginf(logw[0, 2])
    assert int(np.nanargmin(dist2[0])) == 1


def test_trajectory_table_marginal_llr_classifies_with_self_lookup() -> None:
    table, zero = _fixture()
    result = score_trajectory_table_vernier_observer_trial(
        table["plus"][0],
        "plus",
        table,
        true_trace_index=0,
        zero_counts_by_theta=zero,
        include_self=True,
        phi=1.0,
    )
    assert result["readout"] == "trajectory_table_marginal_vernier_llr"
    assert result["decision_rule"] == "marginal_vernier_llr"
    assert result["joint_score_family"] == "poisson_log_likelihood"
    assert result["pred_joint"] == "plus"
    assert result["joint_correct"] is True
    assert result["known_correct"] is True
    assert result["best_trajectory_correct"] is True
    assert result["n_joint_trajectories"] == 2
    assert result["true_trajectory_rank_true"] == 1.0
    assert 1.0 <= result["posterior_neff_true"] <= 2.0
    assert math.isfinite(result["gap_closure_vs_zero_known"])
    assert math.isfinite(result["margin_gap_closure_vs_zero_known"])


def test_trajectory_table_leave_one_out_removes_true_trace_from_joint_catalog() -> None:
    table, zero = _fixture()
    result = score_trajectory_table_vernier_observer_trial(
        table["plus"][0],
        "plus",
        table,
        true_trace_index=0,
        zero_counts_by_theta=zero,
        include_self=False,
        phi=1.0,
    )
    assert result["n_catalog_trajectories"] == 2
    assert result["n_joint_trajectories"] == 1
    assert result["trajectory_table_leave_one_out"] is True
    assert result["pred_known"] == "plus"
    assert result["best_trajectory_correct"] is True


def test_trajectory_weight_sigma_inf_matches_uniform_marginal() -> None:
    table, zero = _fixture()
    poses = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.1]],
            [[1.0, 0.0], [1.0, 0.1]],
        ],
        dtype=np.float64,
    )
    logw, dist2 = trajectory_gaussian_log_weights(
        poses[0],
        poses,
        sigma_arcmin=float("inf"),
        anchor_index=0,
    )
    uniform = score_trajectory_table_vernier_observer_trial(
        table["plus"][0],
        "plus",
        table,
        true_trace_index=0,
        zero_counts_by_theta=zero,
        include_self=True,
        phi=1.0,
    )
    weighted = score_trajectory_table_vernier_observer_trial(
        table["plus"][0],
        "plus",
        table,
        true_trace_index=0,
        zero_counts_by_theta=zero,
        joint_log_trajectory_weights=logw,
        trajectory_prior_label="gaussian_noisy_retinal_trajectory_sigma_infarcmin",
        trajectory_weight_sigma_arcmin=float("inf"),
        trajectory_mean_dist2_arcmin2=dist2,
        include_self=True,
        phi=1.0,
    )
    assert math.isclose(weighted["joint_log_evidence_plus"], uniform["joint_log_evidence_plus"])
    assert math.isclose(weighted["joint_log_evidence_minus"], uniform["joint_log_evidence_minus"])
    assert weighted["trajectory_weight_neff"] == 2.0


def test_trajectory_weight_sigma_zero_collapses_to_known_trace() -> None:
    table, zero = _fixture()
    poses = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.1]],
            [[1.0, 0.0], [1.0, 0.1]],
        ],
        dtype=np.float64,
    )
    logw, dist2 = trajectory_gaussian_log_weights(
        poses[1],
        poses,
        sigma_arcmin=0.0,
        anchor_index=1,
    )
    result = score_trajectory_table_vernier_observer_trial(
        table["minus"][1],
        "minus",
        table,
        true_trace_index=1,
        zero_counts_by_theta=zero,
        joint_log_trajectory_weights=logw,
        trajectory_prior_label="gaussian_noisy_retinal_trajectory_sigma_0arcmin",
        trajectory_weight_sigma_arcmin=0.0,
        trajectory_mean_dist2_arcmin2=dist2,
        include_self=True,
        phi=1.0,
    )
    assert math.isclose(result["joint_log_evidence_plus"], result["known_log_evidence_plus"])
    assert math.isclose(result["joint_log_evidence_minus"], result["known_log_evidence_minus"])
    assert result["trajectory_weight_neff"] == 1.0
    assert result["trajectory_weight_true"] == 1.0


def test_weighted_best_trajectory_uses_noisy_trajectory_prior() -> None:
    table = {
        "plus": np.asarray([[[4.0]], [[1.0]]], dtype=np.float64),
        "minus": np.asarray([[[1.0]], [[5.0]]], dtype=np.float64),
    }
    observed = np.asarray([[5.0]], dtype=np.float64)
    logw = np.asarray([0.0, -np.inf], dtype=np.float64)
    result = score_trajectory_table_vernier_observer_trial(
        observed,
        "plus",
        table,
        true_trace_index=0,
        joint_log_trajectory_weights=logw,
        trajectory_prior_label="one_hot_trace0",
        trajectory_weight_sigma_arcmin=0.0,
        include_self=True,
        phi=1.0,
    )
    assert result["pred_best_trajectory"] == "plus"
    assert math.isclose(result["best_trajectory_log_evidence_plus"], result["known_log_evidence_plus"])
    assert math.isclose(result["best_trajectory_log_evidence_minus"], result["known_log_evidence_minus"])


def test_trajectory_table_residual_mode_is_not_labeled_as_llr() -> None:
    table, zero = _fixture()
    result = score_trajectory_table_vernier_observer_trial(
        table["plus"][0],
        "plus",
        table,
        true_trace_index=0,
        zero_counts_by_theta=zero,
        include_self=True,
        phi=1.0,
        likelihood_normalization="residual",
    )
    assert result["readout"] == "trajectory_table_marginal_residual_score"
    assert result["decision_rule"] == "marginal_mahalanobis_residual_score"
    assert result["joint_score_family"] == "mahalanobis_residual_score"
    assert result["joint_evidence_is_normalized_log_probability"] is False
    assert math.isnan(result["gap_closure_vs_zero_known"])


def test_trajectory_table_summary_reports_observer_accuracies() -> None:
    table, zero = _fixture()
    rows = []
    for label in ("plus", "minus"):
        rows.append(
            {
                "condition": "synthetic",
                "fd_step_arcmin": 0.25,
                "inference_mode": "framewise",
                "zero_eye_reference_condition": "static_center",
                **score_trajectory_table_vernier_observer_trial(
                    table[label][0],
                    label,
                    table,
                    true_trace_index=0,
                    zero_counts_by_theta=zero,
                    include_self=True,
                    phi=1.0,
                ),
            }
        )
    summary = summarize_trajectory_table_rows(rows)
    assert len(summary) == 1
    assert summary[0]["joint_accuracy"] == 1.0
    assert summary[0]["known_accuracy"] == 1.0
    assert summary[0]["best_trajectory_accuracy"] == 1.0
    assert math.isfinite(summary[0]["mean_posterior_neff_true"])


def test_trajectory_table_leave_one_out_requires_retained_trajectory() -> None:
    table, zero = _fixture()
    one_trace = {label: arr[:1] for label, arr in table.items()}
    try:
        score_trajectory_table_vernier_observer_trial(
            one_trace["plus"][0],
            "plus",
            one_trace,
            true_trace_index=0,
            zero_counts_by_theta=zero,
            include_self=False,
            phi=1.0,
        )
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected leave-one-out with one trajectory to fail")


def test_trajectory_table_rejects_nonfinite_counts() -> None:
    table, zero = _fixture()
    bad = {label: arr.copy() for label, arr in table.items()}
    bad["plus"][0, 0, 0] = np.nan
    try:
        score_trajectory_table_vernier_observer_trial(
            table["plus"][0],
            "plus",
            bad,
            true_trace_index=0,
            zero_counts_by_theta=zero,
            include_self=True,
            phi=1.0,
        )
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("Expected non-finite count table to fail")


def test_trajectory_table_known_eye_can_use_observation_table_separate_from_prior() -> None:
    table, zero = _fixture()
    misleading_prior = {"plus": table["minus"].copy(), "minus": table["plus"].copy()}
    result = score_trajectory_table_vernier_observer_trial(
        table["plus"][0],
        "plus",
        misleading_prior,
        true_trace_index=0,
        known_counts_by_theta=table,
        zero_counts_by_theta=zero,
        include_self=True,
        phi=1.0,
    )
    assert result["pred_known"] == "plus"
    assert result["known_correct"] is True
    assert result["pred_joint"] == "minus"


def test_trajectory_table_summary_groups_prior_condition() -> None:
    table, zero = _fixture()
    rows = []
    for prior_condition in ("prior_a", "prior_b"):
        rows.append(
            {
                "condition": "synthetic",
                "prior_condition": prior_condition,
                "fd_step_arcmin": 0.25,
                "inference_mode": "framewise",
                "zero_eye_reference_condition": "static_center",
                **score_trajectory_table_vernier_observer_trial(
                    table["plus"][0],
                    "plus",
                    table,
                    true_trace_index=0,
                    known_counts_by_theta=table,
                    zero_counts_by_theta=zero,
                    include_self=True,
                    phi=1.0,
                ),
            }
        )
    summary = summarize_trajectory_table_rows(rows)
    assert [row["prior_condition"] for row in summary] == ["prior_a", "prior_b"]


def test_trajectory_table_summary_keeps_trajectory_sigmas_separate() -> None:
    table, zero = _fixture()
    rows = []
    for sigma in (0.0, float("inf")):
        poses = np.asarray(
            [
                [[0.0, 0.0], [0.0, 0.1]],
                [[1.0, 0.0], [1.0, 0.1]],
            ],
            dtype=np.float64,
        )
        logw, dist2 = trajectory_gaussian_log_weights(poses[0], poses, sigma_arcmin=sigma, anchor_index=0)
        rows.append(
            {
                "condition": "synthetic",
                "fd_step_arcmin": 0.25,
                "inference_mode": "framewise",
                "zero_eye_reference_condition": "static_center",
                **score_trajectory_table_vernier_observer_trial(
                    table["plus"][0],
                    "plus",
                    table,
                    true_trace_index=0,
                    zero_counts_by_theta=zero,
                    joint_log_trajectory_weights=logw,
                    trajectory_prior_label=f"gaussian_sigma_{sigma}",
                    trajectory_weight_sigma_arcmin=sigma,
                    trajectory_mean_dist2_arcmin2=dist2,
                    include_self=True,
                    phi=1.0,
                ),
            }
        )
    summary = summarize_trajectory_table_rows(rows)
    assert len(summary) == 2
    assert sorted(row["mean_trajectory_weight_neff"] for row in summary) == [1.0, 2.0]


def test_rr100_fast_poisson_path_matches_generic_observer() -> None:
    table, zero = _fixture()
    poses = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.1]],
            [[1.0, 0.0], [1.0, 0.1]],
        ],
        dtype=np.float64,
    )
    args = argparse.Namespace(
        include_self=True,
        trajectory_sigmas_arcmin=[0.0, float("inf")],
        likelihood_scale=1.0,
        likelihood_normalization="poisson",
        reference_condition="static_center",
    )
    rows = _score_condition_poisson(
        args,
        condition="synthetic",
        cache={"fd_step_arcmin": 0.25, "path": "synthetic_cache.npz"},
        table=table,
        poses_arcmin=poses,
        zero=zero,
        metadata={},
        n_timebins=2,
    )
    indexed = {
        (row["trajectory_weight_sigma_arcmin"], row["true_label"], row["trace_index"]): row
        for row in rows
    }
    for sigma in args.trajectory_sigmas_arcmin:
        for true_label in ("plus", "minus"):
            for trace_idx in range(2):
                logw, dist2 = trajectory_gaussian_log_weights(
                    poses[trace_idx],
                    poses,
                    sigma_arcmin=sigma,
                    anchor_index=trace_idx,
                )
                expected = score_trajectory_table_vernier_observer_trial(
                    table[true_label][trace_idx],
                    true_label,
                    table,
                    true_trace_index=trace_idx,
                    zero_counts_by_theta=zero,
                    joint_log_trajectory_weights=logw,
                    trajectory_prior_label="fast_path_test",
                    trajectory_weight_sigma_arcmin=sigma,
                    trajectory_mean_dist2_arcmin2=dist2,
                    include_self=True,
                    phi=1.0,
                )
                actual = indexed[(sigma, true_label, trace_idx)]
                for key in (
                    "joint_log_evidence_plus",
                    "joint_log_evidence_minus",
                    "known_log_evidence_plus",
                    "known_log_evidence_minus",
                    "zero_log_evidence_plus",
                    "zero_log_evidence_minus",
                    "best_trajectory_log_evidence_plus",
                    "best_trajectory_log_evidence_minus",
                    "joint_score",
                    "best_trajectory_score",
                    "posterior_neff_true",
                    "trajectory_weight_neff",
                    "true_trajectory_rank_true",
                ):
                    assert math.isclose(actual[key], expected[key])


def test_static_baseline_mask_handles_missing_and_string_booleans() -> None:
    missing = pd.DataFrame({"condition": ["static_center", "real_aniso_across_1_along_1"]})
    assert _static_baseline_mask(missing).tolist() == [True, False]

    stringy = pd.DataFrame(
        {
            "condition": ["static_center", "real_aniso_across_1_along_1"],
            "is_static_baseline": ["False", "True"],
        }
    )
    assert _static_baseline_mask(stringy).tolist() == [False, True]


def test_rr100_cache_validation_rejects_non_3d_arrays_with_value_error() -> None:
    bad_cache = {
        "plus_rates": np.asarray([[1.0, 2.0]], dtype=np.float64),
        "minus_rates": np.asarray([[[1.0], [2.0]]], dtype=np.float64),
        "pose_traces_deg": np.asarray([[[0.0, 0.0], [0.0, 0.0]]], dtype=np.float64),
    }
    try:
        _cache_tables(bad_cache, bin_seconds=1.0)
    except ValueError as exc:
        assert "3D" in str(exc)
    else:
        raise AssertionError("Expected malformed RR100 cache arrays to fail with ValueError")
