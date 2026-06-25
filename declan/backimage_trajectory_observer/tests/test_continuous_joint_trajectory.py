"""Tests for the continuous joint trajectory observer MVP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from declan.backimage_trajectory_observer.analyze_continuous_joint_trajectory import (
    analyze,
    ar1_profile_log_score,
    build_parser,
    catalog_gaussian_profile_log_score,
    fit_time_constant_observation_matrices,
    kalman_filter_log_likelihood,
    quadratic_profile_log_score,
    score_continuous_joint_score_vectors,
)


def _synthetic_tables() -> dict[str, np.ndarray]:
    t = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
    trajectory_xy = np.asarray(
        [
            np.stack([0.45 * t, 0.25 * np.sin(np.pi * t)], axis=1),
            np.stack([0.50 * t, np.zeros_like(t)], axis=1),
            np.stack([np.zeros_like(t), 0.50 * t], axis=1),
            np.stack([0.35 * t, -0.40 * t], axis=1),
        ],
        dtype=np.float64,
    )
    a_true = np.asarray(
        [
            [5.0, 0.3],
            [0.2, 4.0],
            [2.5, -1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    a_other = np.asarray(
        [
            [-5.0, -0.3],
            [-0.2, -4.0],
            [-2.5, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    a_by_candidate = np.stack([a_true, a_other], axis=0)
    n_candidates, n_traj, n_time, n_units = 2, trajectory_xy.shape[0], trajectory_xy.shape[1], 4
    zero = np.full((n_candidates, n_time, n_units), 30.0, dtype=np.float64)
    zero[1, :, 3] = 31.0
    prior = np.empty((n_candidates, n_traj, n_time, n_units), dtype=np.float64)
    for candidate_index in range(n_candidates):
        for trajectory_index in range(n_traj):
            prior[candidate_index, trajectory_index] = (
                zero[candidate_index] + trajectory_xy[trajectory_index] @ a_by_candidate[candidate_index].T
            )
    known = np.empty((n_candidates, n_time, n_units), dtype=np.float64)
    for candidate_index in range(n_candidates):
        known[candidate_index] = zero[candidate_index] + trajectory_xy[0] @ a_by_candidate[candidate_index].T
    return {
        "trajectory_xy": trajectory_xy,
        "prior_lambda_counts": prior,
        "known_lambda_counts": known,
        "zero_lambda_counts": zero,
        "y_obs_counts": known[0],
        "basis": np.eye(n_units, dtype=np.float64),
    }


def test_kalman_likelihood_prefers_matching_observation_model() -> None:
    fixture = _synthetic_tables()
    tau = fixture["trajectory_xy"][0]
    z = tau @ np.asarray([[5.0, 0.3], [0.2, 4.0], [2.5, -1.0], [0.0, 0.0]], dtype=np.float64).T
    good = kalman_filter_log_likelihood(
        z,
        np.asarray([[5.0, 0.3], [0.2, 4.0], [2.5, -1.0], [0.0, 0.0]], dtype=np.float64),
        alpha=0.90,
        process_var=0.2,
        observation_var=1e-4,
    )
    bad = kalman_filter_log_likelihood(
        z,
        np.asarray([[0.2, 4.0], [5.0, 0.3], [-1.0, 2.5], [0.0, 0.0]], dtype=np.float64),
        alpha=0.90,
        process_var=0.2,
        observation_var=1e-4,
    )
    assert float(good["log_likelihood"]) > float(bad["log_likelihood"])
    assert np.asarray(good["filtered_means"]).shape == tau.shape


def test_continuous_joint_scores_true_image_and_recovers_heldout_trajectory() -> None:
    fixture = _synthetic_tables()
    result = score_continuous_joint_score_vectors(
        y_obs_counts=fixture["y_obs_counts"],
        prior_lambda_counts=fixture["prior_lambda_counts"],
        known_lambda_counts=fixture["known_lambda_counts"],
        zero_lambda_counts=fixture["zero_lambda_counts"],
        trajectory_xy=fixture["trajectory_xy"],
        true_candidate_index=0,
        candidate_ids=["true", "other"],
        basis=fixture["basis"],
        true_trajectory_index=0,
        alpha=0.90,
        process_var=0.2,
        observation_var_floor=1e-5,
        ridge=1e-9,
    )
    assert int(result["continuous_joint_pred_candidate_index"]) == 0
    assert float(result["continuous_joint_true_margin"]) > 0.0
    assert result["fit_rows"][0]["excluded_heldout_trajectory"] is True
    recovery = result["trajectory_recovery"]
    assert float(recovery["trajectory_rmse"]) < 0.08
    assert float(recovery["trajectory_corr_mean"]) > 0.95


def test_time_varying_observation_model_scores_true_image() -> None:
    fixture = _synthetic_tables()
    time_scale = np.linspace(0.55, 1.45, fixture["trajectory_xy"].shape[1], dtype=np.float64)
    prior = fixture["prior_lambda_counts"].copy()
    known = fixture["known_lambda_counts"].copy()
    zero = fixture["zero_lambda_counts"]
    for candidate_index in range(prior.shape[0]):
        delta = prior[candidate_index] - zero[candidate_index][None, :, :]
        prior[candidate_index] = zero[candidate_index][None, :, :] + delta * time_scale[None, :, None]
        known_delta = known[candidate_index] - zero[candidate_index]
        known[candidate_index] = zero[candidate_index] + known_delta * time_scale[:, None]
    result = score_continuous_joint_score_vectors(
        y_obs_counts=known[0],
        prior_lambda_counts=prior,
        known_lambda_counts=known,
        zero_lambda_counts=zero,
        trajectory_xy=fixture["trajectory_xy"],
        true_candidate_index=0,
        candidate_ids=["true", "other"],
        basis=fixture["basis"],
        true_trajectory_index=0,
        alpha=0.90,
        process_var=0.2,
        observation_var_floor=1e-5,
        ridge=1e-9,
        observation_model="time_varying",
        continuous_score_mode="linear_poisson_profile",
    )
    assert int(result["continuous_joint_pred_candidate_index"]) == 0
    assert result["fit_rows"][0]["observation_model"] == "time_varying"
    assert np.asarray(result["A_matrices"]).shape == (2, 9, 4, 2)


def test_quadratic_poisson_profile_scores_curved_true_image() -> None:
    fixture = _synthetic_tables()
    trajectory_xy = fixture["trajectory_xy"]
    x = trajectory_xy[:, :, 0]
    y = trajectory_xy[:, :, 1]
    phi = np.stack([x, y, x * x, x * y, y * y], axis=2)
    coef_true = np.asarray(
        [
            [0.2, 0.1, 5.0, 0.4, 0.2],
            [0.1, 0.2, 0.3, -0.5, 4.5],
            [0.3, -0.1, 2.0, 2.0, -1.5],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    coef_other = -coef_true
    coefficients = np.stack([coef_true, coef_other], axis=0)
    zero = fixture["zero_lambda_counts"].copy()
    prior = np.empty_like(fixture["prior_lambda_counts"])
    known = np.empty_like(fixture["known_lambda_counts"])
    for candidate_index in range(2):
        delta = phi @ coefficients[candidate_index].T
        prior[candidate_index] = zero[candidate_index][None, :, :] + delta
        known[candidate_index] = zero[candidate_index] + delta[0]

    result = score_continuous_joint_score_vectors(
        y_obs_counts=known[0],
        prior_lambda_counts=prior,
        known_lambda_counts=known,
        zero_lambda_counts=zero,
        trajectory_xy=trajectory_xy,
        true_candidate_index=0,
        candidate_ids=["true", "other"],
        basis=fixture["basis"],
        true_trajectory_index=0,
        alpha=0.90,
        process_var=0.2,
        observation_var_floor=1e-5,
        ridge=1e-6,
        continuous_score_mode="quadratic_poisson_profile",
    )
    assert int(result["continuous_joint_pred_candidate_index"]) == 0
    assert float(result["continuous_joint_true_margin"]) > 0.0
    assert any(row.get("observation_model") == "quadratic_time_constant" for row in result["fit_rows"])
    assert any(row.get("qc_type") == "quadratic_profile_optimizer" for row in result["fit_rows"])


def test_quadratic_affine_poisson_profile_scores_offset_curved_true_image() -> None:
    fixture = _synthetic_tables()
    trajectory_xy = fixture["trajectory_xy"]
    x = trajectory_xy[:, :, 0]
    y = trajectory_xy[:, :, 1]
    phi = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=2)
    coef_true = np.asarray(
        [
            [0.7, 0.2, 0.1, 5.0, 0.4, 0.2],
            [-0.4, 0.1, 0.2, 0.3, -0.5, 4.5],
            [0.3, 0.3, -0.1, 2.0, 2.0, -1.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    coef_other = -coef_true
    coefficients = np.stack([coef_true, coef_other], axis=0)
    zero = fixture["zero_lambda_counts"].copy()
    prior = np.empty_like(fixture["prior_lambda_counts"])
    known = np.empty_like(fixture["known_lambda_counts"])
    for candidate_index in range(2):
        delta = phi @ coefficients[candidate_index].T
        prior[candidate_index] = zero[candidate_index][None, :, :] + delta
        known[candidate_index] = zero[candidate_index] + delta[0]

    result = score_continuous_joint_score_vectors(
        y_obs_counts=known[0],
        prior_lambda_counts=prior,
        known_lambda_counts=known,
        zero_lambda_counts=zero,
        trajectory_xy=trajectory_xy,
        true_candidate_index=0,
        candidate_ids=["true", "other"],
        basis=fixture["basis"],
        true_trajectory_index=0,
        alpha=0.90,
        process_var=0.2,
        observation_var_floor=1e-5,
        ridge=1e-6,
        continuous_score_mode="quadratic_affine_poisson_profile",
        quadratic_intercept_ridge_multiplier=4.0,
    )
    assert int(result["continuous_joint_pred_candidate_index"]) == 0
    assert any(row.get("observation_model") == "quadratic_affine_time_constant" for row in result["fit_rows"])
    assert any(row.get("quadratic_include_intercept") is True for row in result["fit_rows"])
    affine_rows = [
        row
        for row in result["fit_rows"]
        if row.get("quadratic_include_intercept") is True and "B_intercept_norm_fraction" in row
    ]
    assert affine_rows
    assert max(float(row.get("B_intercept_norm", 0.0)) for row in affine_rows) > 0.0
    assert all(0.0 <= float(row.get("B_intercept_norm_fraction", -1.0)) <= 1.0 for row in affine_rows)
    assert all(float(row.get("quadratic_intercept_ridge_multiplier", 0.0)) == 4.0 for row in affine_rows)

    ablated = score_continuous_joint_score_vectors(
        y_obs_counts=known[0],
        prior_lambda_counts=prior,
        known_lambda_counts=known,
        zero_lambda_counts=zero,
        trajectory_xy=trajectory_xy,
        true_candidate_index=0,
        candidate_ids=["true", "other"],
        basis=fixture["basis"],
        true_trajectory_index=0,
        alpha=0.90,
        process_var=0.2,
        observation_var_floor=1e-5,
        ridge=1e-6,
        continuous_score_mode="quadratic_affine_poisson_profile",
        quadratic_intercept_ridge_multiplier=4.0,
        quadratic_affine_intercept_scale=0.0,
    )
    assert float(ablated["continuous_joint_true_margin"]) < float(result["continuous_joint_true_margin"])
    optimizer_rows = [
        row
        for row in ablated["fit_rows"]
        if row.get("qc_type") == "quadratic_profile_optimizer"
    ]
    assert optimizer_rows
    assert all(float(row.get("quadratic_affine_intercept_scale", -1.0)) == 0.0 for row in optimizer_rows)

    centered = score_continuous_joint_score_vectors(
        y_obs_counts=known[0],
        prior_lambda_counts=prior,
        known_lambda_counts=known,
        zero_lambda_counts=zero,
        trajectory_xy=trajectory_xy,
        true_candidate_index=0,
        candidate_ids=["true", "other"],
        basis=fixture["basis"],
        true_trajectory_index=0,
        alpha=0.90,
        process_var=0.2,
        observation_var_floor=1e-5,
        ridge=1e-6,
        continuous_score_mode="quadratic_prior_mean_poisson_profile",
    )
    assert int(centered["continuous_joint_pred_candidate_index"]) == 0
    centered_rows = [
        row
        for row in centered["fit_rows"]
        if row.get("quadratic_intercept_strategy") == "prior_mean"
    ]
    assert centered_rows
    assert any(row.get("observation_model") == "quadratic_prior_mean_affine_time_constant" for row in centered_rows)


def test_time_varying_observation_model_records_regularization() -> None:
    fixture = _synthetic_tables()
    fit = fit_time_constant_observation_matrices(
        prior_lambda_counts=fixture["prior_lambda_counts"],
        zero_lambda_counts=fixture["zero_lambda_counts"],
        trajectory_xy=fixture["trajectory_xy"],
        basis=fixture["basis"],
        heldout_trajectory_index=0,
        observation_model="time_varying",
        time_smoothing_sigma=1.0,
        time_shrinkage=0.25,
        ridge=1e-9,
    )
    assert np.asarray(fit["A_matrices"]).shape == (2, 9, 4, 2)
    assert fit["time_smoothing_sigma"] == 1.0
    assert fit["time_shrinkage"] == 0.25
    first_row = fit["fit_rows"][0]
    assert first_row["time_smoothing_sigma"] == 1.0
    assert first_row["time_shrinkage"] == 0.25
    assert first_row["excluded_heldout_trajectory"] is True


def test_ar1_profile_accepts_nonzero_prior_mean_path() -> None:
    t = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
    mean_path = np.stack([0.25 + 0.05 * t, -0.10 + 0.03 * t], axis=1)
    h = np.asarray([[4.0, 0.2], [0.3, 3.0], [1.5, -0.5]], dtype=np.float64)
    z = mean_path @ h.T

    zero_prior = ar1_profile_log_score(
        z,
        h,
        alpha=0.90,
        process_var=0.01,
        observation_var=0.2,
    )
    mean_prior = ar1_profile_log_score(
        z,
        h,
        alpha=0.90,
        process_var=0.01,
        observation_var=0.2,
        prior_mean=mean_path,
    )

    zero_rmse = float(np.sqrt(np.mean((np.asarray(zero_prior["map_means"]) - mean_path) ** 2)))
    mean_rmse = float(np.sqrt(np.mean((np.asarray(mean_prior["map_means"]) - mean_path) ** 2)))
    assert mean_rmse < zero_rmse
    assert float(mean_prior["profile_score"]) > float(zero_prior["profile_score"])


def test_ar1_profile_accepts_known_initial_position_prior() -> None:
    t = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
    tau = np.stack([0.25 + 0.05 * t, -0.15 + 0.03 * t], axis=1)
    h = np.asarray([[2.0, 0.1], [0.1, 1.5]], dtype=np.float64)
    z = tau @ h.T
    unconstrained = ar1_profile_log_score(
        z,
        h,
        alpha=0.90,
        process_var=0.02,
        observation_var=1.0,
    )
    known_start = ar1_profile_log_score(
        z,
        h,
        alpha=0.90,
        process_var=0.02,
        observation_var=1.0,
        initial_mean=tau[0],
        initial_cov=np.eye(2, dtype=np.float64) * 1e-4,
    )

    unconstrained_path = np.asarray(unconstrained["map_means"], dtype=np.float64)
    known_start_path = np.asarray(known_start["map_means"], dtype=np.float64)
    assert float(np.linalg.norm(known_start_path[0] - tau[0])) < float(
        np.linalg.norm(unconstrained_path[0] - tau[0])
    )
    assert float(np.linalg.norm(known_start_path[0] - tau[0])) < 0.01


def test_ar1_profile_supports_matched_brownian_covariance() -> None:
    t = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
    tau = np.stack([0.12 * t, -0.04 * t + 0.02 * np.sin(np.pi * t)], axis=1)
    h = np.asarray([[4.0, 0.2], [0.3, 3.0], [1.5, -0.5]], dtype=np.float64)
    z = tau @ h.T
    process_cov = np.asarray([[0.02, 0.004], [0.004, 0.01]], dtype=np.float64)
    initial_cov = np.eye(2, dtype=np.float64) * 0.05

    result = ar1_profile_log_score(
        z,
        h,
        alpha=1.0,
        process_cov=process_cov,
        initial_cov=initial_cov,
        observation_var=0.05,
    )

    recovered = np.asarray(result["map_means"], dtype=np.float64)
    assert recovered.shape == tau.shape
    assert float(np.sqrt(np.mean((recovered - tau) ** 2))) < 0.04


def test_quadratic_profile_uses_matched_brownian_covariance() -> None:
    t = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    tau = np.stack([np.zeros_like(t), 0.35 * t], axis=1)
    coefficients = np.asarray([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    z = tau[:, 1:2]
    start = np.zeros_like(tau)

    low_y_cov = np.diag([0.05, 1e-4]).astype(np.float64)
    high_y_cov = np.diag([0.05, 0.05]).astype(np.float64)
    low_y = quadratic_profile_log_score(
        z,
        coefficients,
        starts=[start],
        observation_var=0.2,
        alpha=1.0,
        process_cov=low_y_cov,
        initial_mean=np.zeros(2, dtype=np.float64),
        initial_cov=np.eye(2, dtype=np.float64) * 1e-4,
    )
    high_y = quadratic_profile_log_score(
        z,
        coefficients,
        starts=[start],
        observation_var=0.2,
        alpha=1.0,
        process_cov=high_y_cov,
        initial_mean=np.zeros(2, dtype=np.float64),
        initial_cov=np.eye(2, dtype=np.float64) * 1e-4,
    )

    low_path = np.asarray(low_y["map_means"], dtype=np.float64)
    high_path = np.asarray(high_y["map_means"], dtype=np.float64)
    assert high_path.shape == tau.shape
    assert float(high_path[-1, 1]) > float(low_path[-1, 1] + 0.05)
    assert float(high_y["profile_score"]) > float(low_y["profile_score"])


def test_catalog_gaussian_profile_uses_no_anchor_trace_statistics() -> None:
    t = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
    tau = np.stack([0.18 * t + 0.03 * np.sin(np.pi * t), -0.11 * t], axis=1)
    samples = np.stack(
        [
            tau + np.stack([0.01 * np.sin(np.pi * t + phase), -0.01 * np.cos(np.pi * t + phase)], axis=1)
            for phase in np.linspace(0.0, np.pi, 5)
        ],
        axis=0,
    )
    h = np.asarray([[3.0, 0.2], [0.1, 2.5], [1.2, -0.4]], dtype=np.float64)
    z = tau @ h.T

    result = catalog_gaussian_profile_log_score(
        z,
        h,
        samples,
        observation_var=0.05,
        cov_floor=1e-5,
        shrinkage=0.2,
    )

    recovered = np.asarray(result["map_means"], dtype=np.float64)
    assert recovered.shape == tau.shape
    assert float(np.sqrt(np.mean((recovered - tau) ** 2))) < 0.03


def test_catalog_residual_profile_scores_true_image_from_anchor_residual() -> None:
    fixture = _synthetic_tables()
    anchor = fixture["trajectory_xy"][1]
    t = np.linspace(-1.0, 1.0, anchor.shape[0], dtype=np.float64)
    residual = np.stack([0.08 * np.sin(np.pi * t), -0.05 * t], axis=1)
    observed_xy = anchor + residual
    a_true = np.asarray(
        [
            [5.0, 0.3],
            [0.2, 4.0],
            [2.5, -1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    y_obs = fixture["zero_lambda_counts"][0] + observed_xy @ a_true.T
    result = score_continuous_joint_score_vectors(
        y_obs_counts=y_obs,
        prior_lambda_counts=fixture["prior_lambda_counts"],
        known_lambda_counts=fixture["known_lambda_counts"],
        zero_lambda_counts=fixture["zero_lambda_counts"],
        trajectory_xy=fixture["trajectory_xy"],
        observed_trajectory_xy=observed_xy,
        true_candidate_index=0,
        candidate_ids=["true", "other"],
        basis=fixture["basis"],
        true_trajectory_index=-1,
        alpha=0.90,
        process_var=0.1,
        observation_var_floor=1e-5,
        ridge=1e-9,
        continuous_score_mode="catalog_residual_profile",
    )
    assert int(result["continuous_joint_pred_candidate_index"]) == 0
    assert int(result["catalog_residual_best_anchor_indices"][0]) == 1
    assert np.isfinite(float(result["catalog_residual_best_anchor_scores"][0]))
    assert float(result["catalog_residual_anchor_score_gaps"][0]) >= 0.0
    assert float(result["trajectory_recovery"]["trajectory_rmse"]) < 0.10

    smoothed = score_continuous_joint_score_vectors(
        y_obs_counts=y_obs,
        prior_lambda_counts=fixture["prior_lambda_counts"],
        known_lambda_counts=fixture["known_lambda_counts"],
        zero_lambda_counts=fixture["zero_lambda_counts"],
        trajectory_xy=fixture["trajectory_xy"],
        observed_trajectory_xy=observed_xy,
        true_candidate_index=0,
        candidate_ids=["true", "other"],
        basis=fixture["basis"],
        true_trajectory_index=-1,
        alpha=0.90,
        process_var=0.1,
        observation_var_floor=1e-5,
        ridge=1e-9,
        continuous_score_mode="catalog_residual_profile",
        catalog_residual_anchor_smoothing_sigma=1.0,
    )
    assert int(smoothed["continuous_joint_pred_candidate_index"]) == 0
    assert smoothed["catalog_residual_anchor_smoothing_sigma"] == 1.0


def test_coarse_to_fine_profile_scores_true_image_without_anchor_selection() -> None:
    fixture = _synthetic_tables()
    true_tau = fixture["trajectory_xy"][2]
    a_true = np.asarray(
        [
            [5.0, 0.3],
            [0.2, 4.0],
            [2.5, -1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    y_obs = fixture["zero_lambda_counts"][0] + true_tau @ a_true.T
    result = score_continuous_joint_score_vectors(
        y_obs_counts=y_obs,
        prior_lambda_counts=fixture["prior_lambda_counts"],
        known_lambda_counts=fixture["known_lambda_counts"],
        zero_lambda_counts=fixture["zero_lambda_counts"],
        trajectory_xy=fixture["trajectory_xy"],
        observed_trajectory_xy=true_tau,
        true_candidate_index=0,
        candidate_ids=["true", "other"],
        basis=fixture["basis"],
        true_trajectory_index=-1,
        alpha=0.90,
        process_var=0.05,
        observation_var_floor=1e-5,
        ridge=1e-9,
        continuous_score_mode="coarse_to_fine_profile",
        trajectory_basis_components=3,
        trajectory_basis_smoothing_sigma=1.0,
        trajectory_basis_coeff_prior_var=10.0,
    )

    assert int(result["continuous_joint_pred_candidate_index"]) == 0
    assert result["continuous_score_mode"] == "coarse_to_fine_profile"
    assert result["trajectory_basis_components"] == 3
    assert float(result["trajectory_recovery"]["trajectory_rmse"]) < 0.08


def test_coarse_to_fine_profile_supports_catalog_gaussian_coarse_prior() -> None:
    fixture = _synthetic_tables()
    true_tau = fixture["trajectory_xy"][0]
    a_true = np.asarray(
        [
            [5.0, 0.3],
            [0.2, 4.0],
            [2.5, -1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    y_obs = fixture["zero_lambda_counts"][0] + true_tau @ a_true.T
    result = score_continuous_joint_score_vectors(
        y_obs_counts=y_obs,
        prior_lambda_counts=fixture["prior_lambda_counts"],
        known_lambda_counts=fixture["known_lambda_counts"],
        zero_lambda_counts=fixture["zero_lambda_counts"],
        trajectory_xy=fixture["trajectory_xy"],
        observed_trajectory_xy=true_tau,
        true_candidate_index=0,
        candidate_ids=["true", "other"],
        basis=fixture["basis"],
        true_trajectory_index=0,
        alpha=0.90,
        process_var=0.05,
        observation_var_floor=1e-5,
        ridge=1e-9,
        continuous_score_mode="coarse_to_fine_profile",
        trajectory_basis_family="catalog_gaussian",
        catalog_gaussian_cov_floor=1e-5,
        catalog_gaussian_shrinkage=0.2,
    )

    assert int(result["continuous_joint_pred_candidate_index"]) == 0
    assert result["trajectory_basis_family"] == "catalog_gaussian"
    assert float(result["trajectory_recovery"]["trajectory_rmse"]) < 0.12


def test_linear_poisson_profile_supports_catalog_gaussian_prior() -> None:
    fixture = _synthetic_tables()
    true_tau = fixture["trajectory_xy"][0]
    a_true = np.asarray(
        [
            [5.0, 0.3],
            [0.2, 4.0],
            [2.5, -1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    y_obs = fixture["zero_lambda_counts"][0] + true_tau @ a_true.T
    result = score_continuous_joint_score_vectors(
        y_obs_counts=y_obs,
        prior_lambda_counts=fixture["prior_lambda_counts"],
        known_lambda_counts=fixture["known_lambda_counts"],
        zero_lambda_counts=fixture["zero_lambda_counts"],
        trajectory_xy=fixture["trajectory_xy"],
        observed_trajectory_xy=true_tau,
        true_candidate_index=0,
        candidate_ids=["true", "other"],
        basis=fixture["basis"],
        true_trajectory_index=0,
        alpha=0.90,
        process_var=0.05,
        observation_var_floor=1e-5,
        ridge=1e-9,
        continuous_score_mode="linear_poisson_profile",
        trajectory_process_model="catalog_gaussian",
        catalog_gaussian_cov_floor=1e-5,
        catalog_gaussian_shrinkage=0.2,
    )

    assert int(result["continuous_joint_pred_candidate_index"]) == 0
    assert result["trajectory_process_model"] == "catalog_gaussian"
    assert float(result["trajectory_recovery"]["trajectory_rmse"]) < 0.12


def test_catalog_residual_profile_supports_max_anchor_aggregation() -> None:
    fixture = _synthetic_tables()
    anchor = fixture["trajectory_xy"][1]
    t = np.linspace(-1.0, 1.0, anchor.shape[0], dtype=np.float64)
    residual = np.stack([0.08 * np.sin(np.pi * t), -0.05 * t], axis=1)
    observed_xy = anchor + residual
    a_true = np.asarray(
        [
            [5.0, 0.3],
            [0.2, 4.0],
            [2.5, -1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    y_obs = fixture["zero_lambda_counts"][0] + observed_xy @ a_true.T
    common_kwargs = dict(
        y_obs_counts=y_obs,
        prior_lambda_counts=fixture["prior_lambda_counts"],
        known_lambda_counts=fixture["known_lambda_counts"],
        zero_lambda_counts=fixture["zero_lambda_counts"],
        trajectory_xy=fixture["trajectory_xy"],
        observed_trajectory_xy=observed_xy,
        true_candidate_index=0,
        candidate_ids=["true", "other"],
        basis=fixture["basis"],
        true_trajectory_index=-1,
        alpha=0.90,
        process_var=0.1,
        observation_var_floor=1e-5,
        ridge=1e-9,
        continuous_score_mode="catalog_residual_profile",
    )
    marginal = score_continuous_joint_score_vectors(**common_kwargs)
    profile = score_continuous_joint_score_vectors(
        **common_kwargs,
        catalog_residual_aggregation="max",
    )
    top2 = score_continuous_joint_score_vectors(
        **common_kwargs,
        catalog_residual_aggregation="topk_logmeanexp",
        catalog_residual_top_k=2,
    )
    shrunk_top2 = score_continuous_joint_score_vectors(
        **common_kwargs,
        catalog_residual_aggregation="topk_logmeanexp",
        catalog_residual_top_k=2,
        catalog_residual_all_anchor_shrinkage=0.25,
    )

    true_idx = 0
    assert profile["catalog_residual_aggregation"] == "max"
    assert marginal["catalog_residual_aggregation"] == "logmeanexp"
    assert top2["catalog_residual_aggregation"] == "topk_logmeanexp"
    assert shrunk_top2["catalog_residual_aggregation"] == "topk_logmeanexp"
    assert shrunk_top2["catalog_residual_all_anchor_shrinkage"] == 0.25
    assert int(top2["catalog_residual_anchor_aggregate_counts"][true_idx]) == 2
    assert float(profile["continuous_joint_scores"][true_idx]) >= float(
        marginal["continuous_joint_scores"][true_idx]
    )
    assert float(profile["continuous_joint_scores"][true_idx]) >= float(
        top2["continuous_joint_scores"][true_idx]
    )
    assert float(top2["continuous_joint_scores"][true_idx]) >= float(
        marginal["continuous_joint_scores"][true_idx]
    )
    assert float(top2["continuous_joint_scores"][true_idx]) >= float(
        shrunk_top2["continuous_joint_scores"][true_idx]
    )
    assert float(shrunk_top2["continuous_joint_scores"][true_idx]) >= float(
        marginal["continuous_joint_scores"][true_idx]
    )
    assert np.isclose(
        float(profile["continuous_joint_scores"][true_idx]),
        float(profile["catalog_residual_best_anchor_scores"][true_idx]),
    )
    assert np.isclose(
        float(marginal["continuous_joint_scores"][true_idx]),
        float(marginal["catalog_residual_anchor_logmean_scores"][true_idx]),
    )


def test_observation_matrix_fit_rejects_holdout_that_leaves_no_training_trajectory() -> None:
    zero = np.ones((1, 3, 2), dtype=np.float64)
    prior = zero[:, None, :, :] + 0.1
    trajectory_xy = np.zeros((1, 3, 2), dtype=np.float64)
    try:
        fit_time_constant_observation_matrices(
            prior_lambda_counts=prior,
            zero_lambda_counts=zero,
            trajectory_xy=trajectory_xy,
            heldout_trajectory_index=0,
        )
    except ValueError as exc:
        assert "excludes the only available trajectory" in str(exc)
    else:
        raise AssertionError("Expected degenerate held-out fit to fail")


def test_analyzer_reads_cached_trajectory_arrays_and_writes_outputs(tmp_path: Path) -> None:
    fixture = _synthetic_tables()
    run_dir = tmp_path / "run"
    response_dir = run_dir / "response_tables"
    response_dir.mkdir(parents=True)
    table_rel = "response_tables/tiny_continuous.npz"
    np.savez_compressed(
        run_dir / table_rel,
        prior_lambda_counts=fixture["prior_lambda_counts"].astype(np.float32),
        known_lambda_counts=fixture["known_lambda_counts"].astype(np.float32),
        zero_lambda_counts=fixture["zero_lambda_counts"].astype(np.float32),
        y_obs_counts=fixture["y_obs_counts"].astype(np.float32),
        prior_trajectory_xy=fixture["trajectory_xy"].astype(np.float32),
        observed_trajectory_xy=fixture["trajectory_xy"][0].astype(np.float32),
        candidate_ids=np.asarray(["true", "other"]),
        prior_trajectory_ids=np.asarray(["tau0", "tau1", "tau2", "tau3"]),
        true_candidate_index=np.asarray([0], dtype=np.int64),
        true_trajectory_index=np.asarray([0], dtype=np.int64),
        nearest_trajectory_index=np.asarray([0], dtype=np.int64),
        nearest_trajectory_distance=np.asarray([0.0], dtype=np.float32),
    )
    pd.DataFrame(
        [
            {
                "trial_id": 0,
                "candidate_set_mode": "synthetic",
                "observation_family": "synthetic",
                "prior_family": "synthetic",
                "scale": 1.0,
                "axis_catalog_mode": "shared",
                "response_cache_path": table_rel,
                "has_prior_trajectory_xy": True,
                "has_observed_trajectory_xy": True,
            }
        ]
    ).to_csv(run_dir / "response_cache_manifest.csv", index=False)

    out_dir = tmp_path / "continuous"
    args = build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--alpha",
            "0.90",
            "--process-var",
            "0.2",
            "--observation-var-floor",
            "1e-5",
        ]
    )
    analyze(args)

    trials = pd.read_csv(out_dir / "continuous_joint_trials.csv")
    assert trials.shape[0] == 1
    assert bool(trials.loc[0, "continuous_joint_correct"]) is True
    assert float(trials.loc[0, "trajectory_rmse"]) < 0.08

    summary = pd.read_csv(out_dir / "continuous_joint_summary.csv")
    assert summary.shape[0] == 1
    assert float(summary.loc[0, "continuous_joint_accuracy"]) == 1.0

    posterior = pd.read_csv(out_dir / "continuous_joint_feature_posterior.csv")
    assert set(posterior["observer_mode"].astype(str)) == {
        "best_single_tau",
        "continuous_joint",
        "joint",
        "known",
        "zero",
    }
    continuous_posterior = posterior[posterior["observer_mode"].eq("continuous_joint")]
    assert np.isclose(float(continuous_posterior["candidate_posterior"].sum()), 1.0)

    qc = pd.read_csv(out_dir / "continuous_joint_qc.csv")
    assert set(qc["qc_type"].astype(str)) == {"A_I_fit", "signal_nuisance_collapse"}
    fit_qc = qc[qc["qc_type"].eq("A_I_fit")]
    assert fit_qc["excluded_heldout_trajectory"].astype(bool).all()
    collapse_qc = qc[qc["qc_type"].eq("signal_nuisance_collapse")]
    assert not collapse_qc["flat_trajectory_hat"].astype(bool).any()


def test_analyzer_applies_scale_conditioned_basis_and_ridge(tmp_path: Path) -> None:
    fixture = _synthetic_tables()
    run_dir = tmp_path / "run"
    response_dir = run_dir / "response_tables"
    response_dir.mkdir(parents=True)

    manifest_rows = []
    for table_index, scale in enumerate([0.5, 1.0]):
        table_rel = f"response_tables/tiny_scale_{scale:g}.npz"
        np.savez_compressed(
            run_dir / table_rel,
            prior_lambda_counts=fixture["prior_lambda_counts"].astype(np.float32),
            known_lambda_counts=fixture["known_lambda_counts"].astype(np.float32),
            zero_lambda_counts=fixture["zero_lambda_counts"].astype(np.float32),
            y_obs_counts=fixture["y_obs_counts"].astype(np.float32),
            prior_trajectory_xy=fixture["trajectory_xy"].astype(np.float32),
            observed_trajectory_xy=fixture["trajectory_xy"][0].astype(np.float32),
            candidate_ids=np.asarray(["true", "other"]),
            prior_trajectory_ids=np.asarray(["tau0", "tau1", "tau2", "tau3"]),
            true_candidate_index=np.asarray([0], dtype=np.int64),
            true_trajectory_index=np.asarray([0], dtype=np.int64),
        )
        manifest_rows.append(
            {
                "trial_id": table_index,
                "candidate_set_mode": "synthetic",
                "observation_family": "synthetic",
                "prior_family": "synthetic",
                "scale": scale,
                "axis_catalog_mode": "shared",
                "response_cache_path": table_rel,
                "has_prior_trajectory_xy": True,
                "has_observed_trajectory_xy": True,
            }
        )
    pd.DataFrame(manifest_rows).to_csv(run_dir / "response_cache_manifest.csv", index=False)

    out_dir = tmp_path / "continuous_scale_conditioned"
    args = build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--basis-max-dim",
            "1",
            "--basis-max-dim-by-scale",
            "0.5:2,1.0:4",
            "--ridge",
            "0.001",
            "--ridge-by-scale",
            "0.5:0.01,1.0:0.2",
            "--continuous-posterior-temperature",
            "1.5",
            "--continuous-posterior-temperature-by-scale",
            "0.5:0.5,1.0:2.0",
            "--alpha",
            "0.90",
            "--process-var",
            "0.2",
            "--observation-var-floor",
            "1e-5",
        ]
    )
    analyze(args)

    trials = pd.read_csv(out_dir / "continuous_joint_trials.csv").sort_values("prior_scale")
    assert trials["prior_scale"].tolist() == [0.5, 1.0]
    assert trials["basis_max_dim_requested"].tolist() == [2, 4]
    assert trials["basis_dim"].tolist() == [2, 4]
    assert np.allclose(trials["ridge"].to_numpy(dtype=float), [0.01, 0.2])

    summary = pd.read_csv(out_dir / "continuous_joint_summary.csv").sort_values("prior_scale")
    assert summary["basis_max_dim_requested"].tolist() == [2, 4]
    assert summary["basis_dim"].tolist() == [2, 4]
    assert np.allclose(summary["ridge"].to_numpy(dtype=float), [0.01, 0.2])
    assert np.allclose(summary["continuous_posterior_temperature"].to_numpy(dtype=float), [0.5, 2.0])

    posterior = pd.read_csv(out_dir / "continuous_joint_feature_posterior.csv")
    continuous = posterior[posterior["observer_mode"].eq("continuous_joint")].copy()
    assert np.allclose(
        continuous.sort_values("prior_scale")["posterior_temperature"].drop_duplicates().to_numpy(dtype=float),
        [0.5, 2.0],
    )
    assert np.allclose(
        continuous["candidate_score"].to_numpy(dtype=float),
        continuous["candidate_score_raw"].to_numpy(dtype=float)
        / continuous["posterior_temperature"].to_numpy(dtype=float),
    )
    zero = posterior[posterior["observer_mode"].eq("zero")].copy()
    assert np.allclose(zero["posterior_temperature"].to_numpy(dtype=float), 1.0)
    assert np.allclose(zero["candidate_score"].to_numpy(dtype=float), zero["candidate_score_raw"].to_numpy(dtype=float))
