"""Focused tests for the Vernier joint-geometry observer."""

from __future__ import annotations

import math

import numpy as np

from declan.vernier_active_sensing.joint_observer import (
    build_discrete_gaussian_step_prior,
    joint_geometry_vernier_observer_trial,
    score_joint_eye_evidence_enumerated,
)


def _synthetic_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = 3
    n_units = 2
    mu0 = np.zeros((2, t, n_units), dtype=np.float64)
    mu0[0, :, :] = 20.0
    mu0[1, :, :] = np.asarray([23.0, 20.0])
    jac = np.zeros((2, t, n_units, 2), dtype=np.float64)
    jac[0, :, :, :] = np.eye(2)
    jac[1, :, :, :] = 1.5 * np.eye(2)
    pose = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    observed = mu0[0] + np.einsum("tud,td->tu", jac[0], pose)
    u = np.eye(2, dtype=np.float64)
    return observed, mu0, jac, pose, u


def test_step_prior_normalization_and_symmetry() -> None:
    prior = build_discrete_gaussian_step_prior(max_step_arcmin=1.0, sigma_arcmin=1.0, step_arcmin=1.0)
    steps = prior["steps"]
    probs = np.exp(prior["log_probs"])
    assert np.all(probs >= 0.0)
    assert np.isclose(np.sum(probs), 1.0)
    zero_idx = np.where(np.all(np.isclose(steps, 0.0), axis=1))[0][0]
    assert probs[zero_idx] == np.max(probs)
    plus_idx = np.where(np.all(np.isclose(steps, [1.0, 0.0]), axis=1))[0][0]
    minus_idx = np.where(np.all(np.isclose(steps, [-1.0, 0.0]), axis=1))[0][0]
    assert np.isclose(probs[plus_idx], probs[minus_idx])


def test_known_eye_log_evidence_prefers_true_hypothesis() -> None:
    observed, mu0, jac, pose, u = _synthetic_fixture()
    prior = build_discrete_gaussian_step_prior(max_step_arcmin=1.0, sigma_arcmin=1.0, step_arcmin=1.0)
    result = joint_geometry_vernier_observer_trial(
        observed,
        "plus",
        mu0,
        jac,
        u,
        control="correct_chart",
        amplitude_lambda=0.01,
        smoothness_lambda=0.01,
        phi=0.01,
        true_pose_arcmin=pose,
        known_u_trans=u,
        observer_mode="enumerated",
        step_prior=prior,
        max_particles=100,
    )
    assert result["known_log_evidence_plus"] > result["known_log_evidence_minus"]
    assert result["pred_known"] == "plus"


def test_residual_normalization_prefers_zero_residual_known_eye() -> None:
    t = 1
    mu0 = np.asarray([[[1000.0, 1000.0]], [[1.0, 1.0]]], dtype=np.float64)
    jac = np.zeros((2, t, 2, 2), dtype=np.float64)
    pose = np.zeros((t, 2), dtype=np.float64)
    observed = mu0[0].copy()
    u = np.eye(2, dtype=np.float64)
    prior = build_discrete_gaussian_step_prior(max_step_arcmin=0.0, sigma_arcmin=1.0, step_arcmin=1.0)
    result = joint_geometry_vernier_observer_trial(
        observed,
        "plus",
        mu0,
        jac,
        u,
        control="correct_chart",
        amplitude_lambda=0.01,
        smoothness_lambda=0.01,
        phi=1.0,
        true_pose_arcmin=pose,
        known_u_trans=u,
        observer_mode="enumerated",
        step_prior=prior,
        max_particles=10,
        likelihood_normalization="residual",
    )
    assert result["pred_known"] == "plus"
    assert result["known_correct"] is True
    assert result["known_eye_score"] > 0.0


def test_exact_known_candidate_counts_override_local_linear_chart() -> None:
    t = 2
    mu0 = np.zeros((2, t, 2), dtype=np.float64)
    jac = np.zeros((2, t, 2, 2), dtype=np.float64)
    pose = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    known_counts = np.zeros((2, t, 2), dtype=np.float64)
    known_counts[0, :, :] = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    known_counts[1, :, :] = np.asarray([[-2.0, 0.0], [0.0, -2.0]], dtype=np.float64)
    observed = known_counts[0].copy()
    u = np.eye(2, dtype=np.float64)
    prior = build_discrete_gaussian_step_prior(max_step_arcmin=0.0, sigma_arcmin=1.0, step_arcmin=1.0)
    result = joint_geometry_vernier_observer_trial(
        observed,
        "plus",
        mu0,
        jac,
        u,
        control="correct_chart",
        amplitude_lambda=0.01,
        smoothness_lambda=0.01,
        phi=1.0,
        true_pose_arcmin=pose,
        known_u_trans=u,
        known_candidate_counts=known_counts,
        observer_mode="enumerated",
        step_prior=prior,
        max_particles=10,
        likelihood_normalization="residual",
    )
    assert result["known_eye_reference"] == "exact_candidate_counts"
    assert result["known_eye_covariance_reference"] == "candidate_known_counts"
    assert result["joint_geometry_mode"] == "instantaneous_local_chart"
    assert result["joint_score_family"] == "mahalanobis_residual_score"
    assert result["joint_evidence_is_normalized_log_probability"] is False
    assert result["decision_rule"] == "joint_mahalanobis_residual_score"
    assert result["pred_known"] == "plus"
    assert result["known_eye_score"] > 0.0


def test_known_candidate_counts_require_matching_time_and_units() -> None:
    observed, mu0, jac, pose, u = _synthetic_fixture()
    prior = build_discrete_gaussian_step_prior(max_step_arcmin=1.0, sigma_arcmin=1.0, step_arcmin=1.0)
    too_short = np.zeros((2, observed.shape[0] - 1, observed.shape[1]), dtype=np.float64)
    try:
        joint_geometry_vernier_observer_trial(
            observed,
            "plus",
            mu0,
            jac,
            u,
            control="correct_chart",
            amplitude_lambda=0.01,
            smoothness_lambda=0.01,
            phi=1.0,
            true_pose_arcmin=pose,
            known_u_trans=u,
            known_candidate_counts=too_short,
            observer_mode="enumerated",
            step_prior=prior,
            max_particles=10,
        )
    except ValueError as exc:
        assert "time bins" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for short known_candidate_counts")

    wrong_units = np.zeros((2, observed.shape[0], observed.shape[1] + 1), dtype=np.float64)
    try:
        joint_geometry_vernier_observer_trial(
            observed,
            "plus",
            mu0,
            jac,
            u,
            control="correct_chart",
            amplitude_lambda=0.01,
            smoothness_lambda=0.01,
            phi=1.0,
            true_pose_arcmin=pose,
            known_u_trans=u,
            known_candidate_counts=wrong_units,
            observer_mode="enumerated",
            step_prior=prior,
            max_particles=10,
        )
    except ValueError as exc:
        assert "units" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for unit-mismatched known_candidate_counts")


def test_joint_marginalization_beats_zero_eye_for_true_hypothesis() -> None:
    observed, mu0, jac, pose, u = _synthetic_fixture()
    prior = build_discrete_gaussian_step_prior(max_step_arcmin=1.0, sigma_arcmin=1.0, step_arcmin=1.0)
    result = joint_geometry_vernier_observer_trial(
        observed,
        "plus",
        mu0,
        jac,
        u,
        control="correct_chart",
        amplitude_lambda=0.01,
        smoothness_lambda=0.01,
        phi=0.01,
        true_pose_arcmin=pose,
        known_u_trans=u,
        observer_mode="enumerated",
        step_prior=prior,
        max_particles=100,
    )
    assert result["joint_log_evidence_true"] > result["zero_log_evidence_true"]
    assert result["chosen_theta"] == "plus"


def test_residual_mode_suppresses_raw_gap_and_reports_margin_closure() -> None:
    observed, mu0, jac, pose, u = _synthetic_fixture()
    prior = build_discrete_gaussian_step_prior(max_step_arcmin=1.0, sigma_arcmin=1.0, step_arcmin=1.0)
    result = joint_geometry_vernier_observer_trial(
        observed,
        "plus",
        mu0,
        jac,
        u,
        control="correct_chart",
        amplitude_lambda=0.01,
        smoothness_lambda=0.01,
        phi=0.01,
        true_pose_arcmin=pose,
        known_u_trans=u,
        observer_mode="enumerated",
        step_prior=prior,
        max_particles=100,
    )
    expected = (
        (result["joint_score"] - result["zero_eye_score"])
        / (result["known_eye_score"] - result["zero_eye_score"])
    )
    assert math.isnan(result["gap_closure_vs_zero_known"])
    assert math.isfinite(result["margin_gap_closure_vs_zero_known"])
    assert np.isclose(result["margin_gap_closure_vs_zero_known"], expected)


def test_full_likelihood_gap_closure_uses_log_evidence_orientation() -> None:
    observed, mu0, jac, pose, u = _synthetic_fixture()
    prior = build_discrete_gaussian_step_prior(max_step_arcmin=1.0, sigma_arcmin=1.0, step_arcmin=1.0)
    result = joint_geometry_vernier_observer_trial(
        observed,
        "plus",
        mu0,
        jac,
        u,
        control="correct_chart",
        amplitude_lambda=0.01,
        smoothness_lambda=0.01,
        phi=0.01,
        true_pose_arcmin=pose,
        known_u_trans=u,
        observer_mode="enumerated",
        step_prior=prior,
        max_particles=100,
        likelihood_normalization="full",
    )
    expected = (
        (result["joint_log_evidence_true"] - result["zero_log_evidence_true"])
        / (result["known_log_evidence_true"] - result["zero_log_evidence_true"])
    )
    assert math.isfinite(result["gap_closure_vs_zero_known"])
    assert np.isclose(result["gap_closure_vs_zero_known"], expected)


def test_wrong_geometry_has_separate_control_known_score() -> None:
    observed, mu0, jac, pose, u = _synthetic_fixture()
    prior = build_discrete_gaussian_step_prior(max_step_arcmin=1.0, sigma_arcmin=1.0, step_arcmin=1.0)
    result = joint_geometry_vernier_observer_trial(
        observed,
        "plus",
        mu0,
        jac,
        u,
        control="wrong_chart",
        amplitude_lambda=0.01,
        smoothness_lambda=0.01,
        phi=0.01,
        true_pose_arcmin=pose,
        known_u_trans=u,
        observer_mode="enumerated",
        step_prior=prior,
        max_particles=100,
    )
    assert "known_log_evidence_plus_control_chart" in result
    assert "known_log_evidence_plus" in result
    assert not np.isclose(result["known_log_evidence_plus"], result["known_log_evidence_plus_control_chart"])


def test_pruned_beam_evidence_charges_discarded_mass() -> None:
    z = np.zeros((2, 1), dtype=np.float64)
    chart = np.zeros((2, 1, 2), dtype=np.float64)
    cov = np.ones((2, 1, 1), dtype=np.float64)
    prior = build_discrete_gaussian_step_prior(max_step_arcmin=1.0, sigma_arcmin=1.0, step_arcmin=1.0)
    full = score_joint_eye_evidence_enumerated(
        z,
        chart,
        cov,
        prior,
        max_particles=1000,
        likelihood_scale=1.0,
        epsilon=1e-8,
    )
    pruned = score_joint_eye_evidence_enumerated(
        z,
        chart,
        cov,
        prior,
        max_particles=1,
        likelihood_scale=1.0,
        epsilon=1e-8,
    )
    assert pruned["retained_mass_by_t"][0] < 1.0
    assert pruned["log_evidence"] <= full["log_evidence"] + 1e-9


def test_known_eye_missing_pose_is_not_arbitrary_minus() -> None:
    observed, mu0, jac, _pose, u = _synthetic_fixture()
    prior = build_discrete_gaussian_step_prior(max_step_arcmin=1.0, sigma_arcmin=1.0, step_arcmin=1.0)
    result = joint_geometry_vernier_observer_trial(
        observed,
        "plus",
        mu0,
        jac,
        u,
        control="correct_chart",
        amplitude_lambda=0.01,
        smoothness_lambda=0.01,
        phi=0.01,
        true_pose_arcmin=None,
        known_u_trans=u,
        observer_mode="enumerated",
        step_prior=prior,
        max_particles=100,
    )
    assert result["pred_known"] == ""
    assert math.isnan(result["known_correct"])
    assert math.isnan(result["known_log_evidence_true"])


def test_noncorrect_known_upper_requires_explicit_correct_basis() -> None:
    observed, mu0, jac, pose, u = _synthetic_fixture()
    prior = build_discrete_gaussian_step_prior(max_step_arcmin=1.0, sigma_arcmin=1.0, step_arcmin=1.0)
    result = joint_geometry_vernier_observer_trial(
        observed,
        "plus",
        mu0,
        jac,
        u,
        control="wrong_chart",
        amplitude_lambda=0.01,
        smoothness_lambda=0.01,
        phi=0.01,
        true_pose_arcmin=pose,
        known_u_trans=None,
        observer_mode="enumerated",
        step_prior=prior,
        max_particles=100,
    )
    assert result["pred_known"] == ""
    assert math.isnan(result["known_log_evidence_true"])
    assert math.isnan(result["gap_closure_vs_zero_known"])
