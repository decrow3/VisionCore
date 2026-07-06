"""Tests for the second-pass Vernier joint-decoding diagnostic."""

from __future__ import annotations

from declan.vernier_active_sensing.run_second_pass_joint_diagnostic import (
    _sweep_likelihood_scale_trial,
    build_scale_policy_summaries,
    selection_scope_keys,
    summarize_static_margin_comparison,
)
from declan.vernier_active_sensing.tests.test_trajectory_table_observer import _fixture
from declan.vernier_active_sensing.trajectory_table_observer import score_trajectory_table_vernier_observer_trial


def _row(
    *,
    split: int,
    scale: float,
    correct: bool,
    condition: str = "real_fem",
    prior_condition: str = "real_fem",
) -> dict[str, object]:
    return {
        "catalog_mode": "leave_one_out",
        "condition": condition,
        "prior_condition": prior_condition,
        "condition_matches_prior": condition == prior_condition,
        "fd_step_arcmin": 0.5,
        "inference_mode": "framewise",
        "calibration_split": split,
        "trace_index": split,
        "likelihood_scale": scale,
        "joint_correct": correct,
        "known_correct": True,
        "zero_correct": False,
        "best_trajectory_correct": correct,
        "joint_score": 1.0 if correct else -1.0,
        "known_eye_score": 2.0,
        "zero_eye_score": 0.0,
        "best_trajectory_score": 1.0 if correct else -1.0,
        "posterior_neff_true": 2.0,
        "true_trajectory_rank_true": 1.0,
        "gap_closure_vs_zero_known": 0.5,
        "margin_gap_closure_vs_zero_known": 0.5 if correct else 0.0,
        "n_joint_trajectories": 15,
    }


def test_selection_scope_keys_are_explicit() -> None:
    assert selection_scope_keys("global_by_fd_and_mode") == ["catalog_mode", "fd_step_arcmin"]
    assert selection_scope_keys("condition_by_fd_and_mode") == ["catalog_mode", "condition", "fd_step_arcmin"]
    assert selection_scope_keys("condition_prior_by_fd_and_mode") == [
        "catalog_mode",
        "condition",
        "prior_condition",
        "fd_step_arcmin",
    ]


def test_heldout_scale_policy_uses_calibration_traces_for_selection() -> None:
    rows = [
        _row(split=0, scale=0.5, correct=False),
        _row(split=0, scale=1.0, correct=False),
        _row(split=1, scale=0.5, correct=True),
        _row(split=1, scale=1.0, correct=False),
    ]
    selection_rows, heldout_summary, pair_summary = build_scale_policy_summaries(
        rows,
        selection_keys=["catalog_mode", "fd_step_arcmin"],
        baseline_likelihood_scale=1.0,
    )

    heldout0_candidates = [
        row
        for row in selection_rows
        if row["heldout_split"] == 0 and row["selection_role"] == "calibration_candidate"
    ]
    assert len(heldout0_candidates) == 2
    selected_for_heldout0 = [row for row in heldout0_candidates if row["selected_by_calibration"]]
    assert len(selected_for_heldout0) == 1
    assert selected_for_heldout0[0]["likelihood_scale"] == 0.5

    selected_eval = [
        row
        for row in heldout_summary
        if row["heldout_split"] == 0 and row["scale_policy"] == "selected_by_calibration"
    ]
    baseline_eval = [
        row
        for row in heldout_summary
        if row["heldout_split"] == 0 and row["scale_policy"] == "baseline_scale_1"
    ]
    assert selected_eval[0]["joint_accuracy"] == 0.0
    assert baseline_eval[0]["joint_accuracy"] == 0.0
    assert pair_summary


def test_fast_scale_sweep_matches_single_scale_observer() -> None:
    table, zero = _fixture()
    sweep_rows = _sweep_likelihood_scale_trial(
        table["plus"][0],
        "plus",
        table,
        true_trace_index=0,
        known_counts_by_theta=table,
        zero_counts_by_theta=zero,
        include_self=True,
        phi=1.0,
        likelihood_normalization="poisson",
        likelihood_scales=[0.25, 1.0],
    )
    by_scale = {row["likelihood_scale"]: row for row in sweep_rows}
    for scale in (0.25, 1.0):
        baseline = score_trajectory_table_vernier_observer_trial(
            table["plus"][0],
            "plus",
            table,
            true_trace_index=0,
            known_counts_by_theta=table,
            zero_counts_by_theta=zero,
            include_self=True,
            phi=1.0,
            likelihood_normalization="poisson",
            likelihood_scale=scale,
        )
        assert by_scale[scale]["pred_joint"] == baseline["pred_joint"]
        assert by_scale[scale]["joint_correct"] == baseline["joint_correct"]
        assert by_scale[scale]["joint_score"] == baseline["joint_score"]
        assert by_scale[scale]["posterior_neff_true"] == baseline["posterior_neff_true"]


def test_static_margin_comparison_tracks_known_gain_and_joint_gap() -> None:
    base = {
        "catalog_mode": "leave_one_out",
        "fd_step_arcmin": 0.5,
        "inference_mode": "framewise",
        "observation_mode": "expected_counts",
        "likelihood_scale": 1.0,
        "joint_likelihood_normalization": "poisson",
        "joint_score_family": "poisson_log_likelihood",
        "zero_eye_reference_condition": "static_center",
        "n": 32,
        "joint_accuracy": 1.0,
        "known_accuracy": 1.0,
        "zero_accuracy": 1.0,
    }
    rows = [
        {
            **base,
            "condition": "static_center",
            "prior_condition": "static_center",
            "condition_matches_prior": True,
            "mean_joint_score": 3.0,
            "mean_known_eye_score": 3.0,
            "mean_zero_eye_score": 3.0,
        },
        {
            **base,
            "condition": "real_fem",
            "prior_condition": "real_fem",
            "condition_matches_prior": True,
            "joint_accuracy": 0.6,
            "zero_accuracy": 0.5,
            "mean_joint_score": 1.5,
            "mean_known_eye_score": 6.0,
            "mean_zero_eye_score": -1.0,
        },
    ]

    comparison = summarize_static_margin_comparison(rows)
    fem = [row for row in comparison if row["condition"] == "real_fem"][0]

    assert fem["known_margin_ratio_vs_static"] == 2.0
    assert fem["joint_margin_ratio_vs_static"] == 0.5
    assert fem["joint_accuracy_delta_vs_zero"] == 0.09999999999999998
    assert fem["joint_fraction_of_known_static_gain"] == -0.5
