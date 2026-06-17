"""Focused tests for the exact cached-trajectory Vernier observer."""

from __future__ import annotations

import math

import numpy as np

from declan.vernier_active_sensing.trajectory_table_observer import (
    score_trajectory_table_vernier_observer_trial,
    summarize_trajectory_table_rows,
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
