"""Tests for the continuous quadratic Vernier joint observer."""

from __future__ import annotations

import numpy as np

from declan.vernier_active_sensing.joint_observer import THETA_MINUS, THETA_PLUS
from declan.vernier_active_sensing.run_continuous_quadratic_joint_observer import (
    FeatureConfig,
    _energy_and_grad,
    fixed_path_score,
    fit_brownian_prior,
    fit_confined_step_prior,
    fit_quadratic_map,
    profile_quadratic_score,
)


def _synthetic_counts() -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    traces = np.asarray(
        [
            [[-1.0, 0.0], [-0.5, 0.1], [0.0, 0.0]],
            [[0.0, 0.0], [0.5, -0.1], [1.0, 0.0]],
            [[1.0, 0.0], [0.5, 0.2], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    zero = {
        THETA_PLUS: np.full((3, 2), 10.0, dtype=np.float64),
        THETA_MINUS: np.full((3, 2), 10.0, dtype=np.float64),
    }
    x = traces[:, :, 0]
    y = traces[:, :, 1]
    plus_resid = np.stack([1.5 * x + 0.25 * x * x, 0.5 * y], axis=-1)
    minus_resid = np.stack([-1.5 * x + 0.25 * x * x, -0.5 * y], axis=-1)
    counts = {
        THETA_PLUS: zero[THETA_PLUS][None, :, :] + plus_resid,
        THETA_MINUS: zero[THETA_MINUS][None, :, :] + minus_resid,
    }
    return counts, traces, zero


def test_energy_gradient_matches_finite_difference() -> None:
    z_obs = np.asarray([[0.2, -0.1], [0.4, 0.1]], dtype=np.float64)
    coef = np.asarray(
        [
            [0.3, -0.2, 0.1, 0.05, -0.01],
            [-0.4, 0.1, 0.02, -0.03, 0.04],
        ],
        dtype=np.float64,
    )
    poses = np.asarray([[0.1, -0.2], [0.3, 0.05]], dtype=np.float64)
    prior = fit_brownian_prior(
        np.asarray([poses, poses + 0.1], dtype=np.float64),
        np.asarray([0, 1]),
        covariance_floor_arcmin2=1e-3,
    )
    energy, grad, _obs, _prior = _energy_and_grad(
        poses.reshape(-1),
        z_obs=z_obs,
        coef=coef,
        prior=prior,
        residual_var=0.7,
        observation_weight=1.3,
    )
    eps = 1e-6
    numeric = np.zeros_like(grad)
    flat = poses.reshape(-1)
    for idx in range(flat.size):
        plus = flat.copy()
        minus = flat.copy()
        plus[idx] += eps
        minus[idx] -= eps
        e_plus = _energy_and_grad(
            plus,
            z_obs=z_obs,
            coef=coef,
            prior=prior,
            residual_var=0.7,
            observation_weight=1.3,
        )[0]
        e_minus = _energy_and_grad(
            minus,
            z_obs=z_obs,
            coef=coef,
            prior=prior,
            residual_var=0.7,
            observation_weight=1.3,
        )[0]
        numeric[idx] = (e_plus - e_minus) / (2.0 * eps)
    assert np.isfinite(energy)
    assert np.allclose(grad, numeric, rtol=1e-5, atol=1e-5)


def test_confined_prior_energy_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(123)
    source = rng.normal(scale=0.2, size=(8, 5, 2)).cumsum(axis=1)
    prior = fit_confined_step_prior(source, covariance_floor_arcmin2=1e-3)
    poses = source[0] + 0.03
    z_obs = np.zeros((poses.shape[0], 2), dtype=np.float64)
    coef = np.zeros((2, 5), dtype=np.float64)
    energy, grad, _obs, _prior = _energy_and_grad(
        poses.reshape(-1),
        z_obs=z_obs,
        coef=coef,
        prior=prior,
        residual_var=0.7,
        observation_weight=1.3,
    )
    eps = 1e-6
    numeric = np.zeros_like(grad)
    flat = poses.reshape(-1)
    for idx in range(flat.size):
        plus = flat.copy()
        minus = flat.copy()
        plus[idx] += eps
        minus[idx] -= eps
        e_plus = _energy_and_grad(
            plus,
            z_obs=z_obs,
            coef=coef,
            prior=prior,
            residual_var=0.7,
            observation_weight=1.3,
        )[0]
        e_minus = _energy_and_grad(
            minus,
            z_obs=z_obs,
            coef=coef,
            prior=prior,
            residual_var=0.7,
            observation_weight=1.3,
        )[0]
        numeric[idx] = (e_plus - e_minus) / (2.0 * eps)
    assert np.isfinite(energy)
    assert np.allclose(grad, numeric, rtol=1e-5, atol=1e-5)


def test_quadratic_profile_recovers_better_score_than_zero_path() -> None:
    counts, poses, zero = _synthetic_counts()
    train = np.asarray([0, 1], dtype=np.int64)
    qmap = fit_quadratic_map(counts, poses, zero, train, basis_dim=2, ridge=1e-6)
    prior = fit_brownian_prior(poses, train, covariance_floor_arcmin2=1e-3)
    true_trace = poses[2]
    observed = counts[THETA_PLUS][2]
    z_obs = (observed - zero[THETA_PLUS]) @ qmap.basis
    starts = [np.zeros_like(true_trace), np.mean(poses[train], axis=0), true_trace]
    profiled = profile_quadratic_score(
        z_obs,
        qmap.coef_by_label[THETA_PLUS],
        prior,
        starts,
        residual_var=qmap.residual_var_by_label[THETA_PLUS],
        observation_weight=1.0,
        max_iter=50,
        true_pose_arcmin=true_trace,
    )
    proposal_only = profile_quadratic_score(
        z_obs,
        qmap.coef_by_label[THETA_PLUS],
        prior,
        starts,
        residual_var=qmap.residual_var_by_label[THETA_PLUS],
        observation_weight=1.0,
        max_iter=0,
        true_pose_arcmin=true_trace,
    )
    zero_score = fixed_path_score(
        z_obs,
        qmap.coef_by_label[THETA_PLUS],
        prior,
        np.zeros_like(true_trace),
        residual_var=qmap.residual_var_by_label[THETA_PLUS],
        observation_weight=1.0,
    )[0]
    assert profiled.score >= zero_score
    assert proposal_only.score >= zero_score
    assert profiled.rmse_arcmin < 2.0


def test_history_features_support_lagged_known_trace_signal() -> None:
    traces = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[0.0, 0.0], [-1.0, 0.0], [-2.0, 0.0], [-3.0, 0.0]],
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]],
        ],
        dtype=np.float64,
    )
    zero = {
        THETA_PLUS: np.full((4, 2), 10.0, dtype=np.float64),
        THETA_MINUS: np.full((4, 2), 10.0, dtype=np.float64),
    }
    lagged_x = np.concatenate([traces[:, :1, 0], traces[:, :-1, 0]], axis=1)
    counts = {
        THETA_PLUS: zero[THETA_PLUS][None, :, :] + np.stack([lagged_x, np.zeros_like(lagged_x)], axis=-1),
        THETA_MINUS: zero[THETA_MINUS][None, :, :] - np.stack([lagged_x, np.zeros_like(lagged_x)], axis=-1),
    }
    train = np.asarray([0, 1], dtype=np.int64)
    cfg = FeatureConfig(mode="history_quadratic", history_lags=(0, 1))
    qmap = fit_quadratic_map(counts, traces, zero, train, basis_dim=2, ridge=1e-6, feature_config=cfg)
    prior = fit_brownian_prior(traces, train, covariance_floor_arcmin2=1e-3)
    observed = counts[THETA_PLUS][2]
    known_scores = {}
    for label in (THETA_PLUS, THETA_MINUS):
        z_obs = (observed - zero[label]) @ qmap.basis
        known_scores[label] = fixed_path_score(
            z_obs,
            qmap.coef_by_label[label],
            prior,
            traces[2],
            residual_var=qmap.residual_var_by_label[label],
            observation_weight=1.0,
            feature_config=qmap.feature_config,
        )[0]
    assert known_scores[THETA_PLUS] > known_scores[THETA_MINUS]
    assert qmap.n_pose_features == 10
