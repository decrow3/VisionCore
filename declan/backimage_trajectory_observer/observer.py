"""Finite image-candidate trajectory-table observer."""

from __future__ import annotations

from typing import Any

import numpy as np

from .likelihood import (
    effective_count,
    entropy,
    logsumexp,
    normalized_log_weights,
    poisson_expected_count_loglik,
    posterior_from_log_scores,
    rank_desc,
    true_margin,
)


def _as_prior_table(arr: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 4:
        raise ValueError(f"{name} must be (candidate, trajectory, time, unit), got {out.shape}")
    if out.shape[0] <= 0 or out.shape[1] <= 0:
        raise ValueError(f"{name} must contain at least one candidate and one trajectory")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values")
    return out


def _as_candidate_table(arr: np.ndarray, name: str, expected_shape: tuple[int, int, int]) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 3:
        raise ValueError(f"{name} must be (candidate, time, unit), got {out.shape}")
    if out.shape != expected_shape:
        raise ValueError(f"{name} shape {out.shape} does not match expected {expected_shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values")
    return out


def _prediction(scores: np.ndarray) -> int:
    vals = np.asarray(scores, dtype=np.float64)
    if vals.size == 0 or not np.isfinite(vals).any():
        return -1
    return int(np.nanargmax(vals))


def _score_summary(prefix: str, scores: np.ndarray, true_candidate_index: int, candidate_ids: list[str]) -> dict[str, Any]:
    pred = _prediction(scores)
    true_idx = int(true_candidate_index)
    return {
        f"{prefix}_pred_candidate_index": pred,
        f"{prefix}_pred_image_id": candidate_ids[pred] if 0 <= pred < len(candidate_ids) else "",
        f"{prefix}_correct": bool(pred == true_idx) if pred >= 0 else False,
        f"{prefix}_true_rank": rank_desc(scores, true_idx),
        f"{prefix}_true_margin": true_margin(scores, true_idx),
        f"{prefix}_true_score": float(scores[true_idx]) if 0 <= true_idx < len(scores) else float("nan"),
    }


def score_image_identity_score_vectors(
    *,
    y_obs_counts: np.ndarray,
    prior_lambda_counts: np.ndarray,
    known_lambda_counts: np.ndarray,
    zero_lambda_counts: np.ndarray,
    true_candidate_index: int,
    candidate_ids: list[str] | None = None,
    log_trajectory_prior: np.ndarray | None = None,
    eps: float = 1e-8,
    likelihood_scale: float = 1.0,
) -> dict[str, Any]:
    """Return full candidate score vectors for finite image/trajectory tables."""
    prior = _as_prior_table(prior_lambda_counts, "prior_lambda_counts")
    n_candidates, n_traj, n_time, n_units = prior.shape
    expected_candidate_shape = (n_candidates, n_time, n_units)
    known = _as_candidate_table(known_lambda_counts, "known_lambda_counts", expected_candidate_shape)
    zero = _as_candidate_table(zero_lambda_counts, "zero_lambda_counts", expected_candidate_shape)
    obs = np.asarray(y_obs_counts, dtype=np.float64)
    if obs.shape != (n_time, n_units):
        raise ValueError(f"y_obs_counts shape {obs.shape} does not match expected {(n_time, n_units)}")
    if not np.isfinite(obs).all():
        raise ValueError("y_obs_counts contains non-finite values")
    true_idx = int(true_candidate_index)
    if true_idx < 0 or true_idx >= n_candidates:
        raise ValueError(f"true_candidate_index {true_idx} outside candidate table size {n_candidates}")
    ids = [str(i) for i in range(n_candidates)] if candidate_ids is None else [str(v) for v in candidate_ids]
    if len(ids) != n_candidates:
        raise ValueError(f"candidate_ids length {len(ids)} does not match n_candidates={n_candidates}")

    prior_ll = poisson_expected_count_loglik(
        obs,
        prior,
        eps=float(eps),
        likelihood_scale=float(likelihood_scale),
    )
    known_scores = poisson_expected_count_loglik(
        obs,
        known,
        eps=float(eps),
        likelihood_scale=float(likelihood_scale),
    )
    zero_scores = poisson_expected_count_loglik(
        obs,
        zero,
        eps=float(eps),
        likelihood_scale=float(likelihood_scale),
    )
    log_prior = normalized_log_weights(log_trajectory_prior, n_traj, n_rows=n_candidates)
    log_prior_for_joint = log_prior[None, :] if log_prior.ndim == 1 else log_prior
    joint_scores = logsumexp(prior_ll + log_prior_for_joint, axis=1)
    best_single_tau_scores = np.max(prior_ll, axis=1)
    return {
        "prior_log_likelihood": prior_ll,
        "log_trajectory_prior": log_prior,
        "known_scores": known_scores,
        "zero_scores": zero_scores,
        "joint_scores": joint_scores,
        "best_single_tau_scores": best_single_tau_scores,
        "candidate_ids": ids,
        "n_candidates": int(n_candidates),
        "n_trajectories": int(n_traj),
        "n_timebins": int(n_time),
        "n_units": int(n_units),
        "true_candidate_index": int(true_idx),
    }


def posterior_weighted_feature(
    scores: np.ndarray,
    candidate_features: np.ndarray,
    *,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a feature vector by posterior averaging over candidate scores."""
    vals = np.asarray(scores, dtype=np.float64)
    features = np.asarray(candidate_features, dtype=np.float64)
    if vals.ndim != 1:
        raise ValueError(f"scores must be 1D, got {vals.shape}")
    if features.ndim != 2:
        raise ValueError(f"candidate_features must be 2D, got {features.shape}")
    if features.shape[0] != vals.shape[0]:
        raise ValueError(
            f"candidate_features has {features.shape[0]} candidates, but scores has {vals.shape[0]}"
        )
    if not np.isfinite(features).all():
        raise ValueError("candidate_features contains non-finite values")
    temp = float(temperature)
    if temp <= 0.0 or not np.isfinite(temp):
        raise ValueError("temperature must be positive and finite")
    posterior = posterior_from_log_scores(vals / temp)
    if not np.isfinite(posterior).all():
        return np.full(features.shape[1], np.nan, dtype=np.float64), posterior
    return posterior @ features, posterior


def feature_recovery_metrics(z_hat: np.ndarray, z_true: np.ndarray) -> dict[str, float]:
    """Return compact feature-recovery metrics where larger neg-MSE is better."""
    pred = np.asarray(z_hat, dtype=np.float64)
    true = np.asarray(z_true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"z_hat shape {pred.shape} does not match z_true shape {true.shape}")
    if pred.ndim != 1:
        raise ValueError(f"feature vectors must be 1D, got {pred.shape}")
    if pred.size == 0:
        raise ValueError("feature vectors must contain at least one value")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        return {
            "feature_mse": float("nan"),
            "feature_neg_mse": float("nan"),
            "feature_rmse": float("nan"),
            "feature_l2_error": float("nan"),
            "feature_cosine": float("nan"),
            "feature_true_norm": float("nan"),
            "feature_pred_norm": float("nan"),
        }
    diff = pred - true
    mse = float(np.mean(diff * diff))
    l2 = float(np.sqrt(np.sum(diff * diff)))
    pred_norm = float(np.sqrt(np.sum(pred * pred)))
    true_norm = float(np.sqrt(np.sum(true * true)))
    denom = pred_norm * true_norm
    cosine = float(np.sum(pred * true) / denom) if denom > 1e-12 else float("nan")
    return {
        "feature_mse": mse,
        "feature_neg_mse": -mse,
        "feature_rmse": float(np.sqrt(mse)),
        "feature_l2_error": l2,
        "feature_cosine": cosine,
        "feature_true_norm": true_norm,
        "feature_pred_norm": pred_norm,
    }


def score_image_identity_table(
    *,
    y_obs_counts: np.ndarray,
    prior_lambda_counts: np.ndarray,
    known_lambda_counts: np.ndarray,
    zero_lambda_counts: np.ndarray,
    true_candidate_index: int,
    candidate_ids: list[str] | None = None,
    log_trajectory_prior: np.ndarray | None = None,
    true_trajectory_index: int | None = None,
    nearest_trajectory_index: int | None = None,
    nearest_trajectory_distance: float | None = None,
    eps: float = 1e-8,
    likelihood_scale: float = 1.0,
) -> dict[str, Any]:
    """Score one observed response against finite image and trajectory tables.

    The same observed counts are scored by all modes. The prior table is used
    only by the latent-eye joint and best-single-trajectory diagnostic.
    Known-eye and zero-eye are separate candidate tables so that leave-one-out
    trajectory priors can still report a non-leaky known-eye upper bound while
    the zero-eye baseline remains a static-eye model assumption, not a static
    input.
    """
    vectors = score_image_identity_score_vectors(
        y_obs_counts=y_obs_counts,
        prior_lambda_counts=prior_lambda_counts,
        known_lambda_counts=known_lambda_counts,
        zero_lambda_counts=zero_lambda_counts,
        true_candidate_index=true_candidate_index,
        candidate_ids=candidate_ids,
        log_trajectory_prior=log_trajectory_prior,
        eps=eps,
        likelihood_scale=likelihood_scale,
    )
    prior_ll = np.asarray(vectors["prior_log_likelihood"], dtype=np.float64)
    log_prior = np.asarray(vectors["log_trajectory_prior"], dtype=np.float64)
    known_scores = np.asarray(vectors["known_scores"], dtype=np.float64)
    zero_scores = np.asarray(vectors["zero_scores"], dtype=np.float64)
    joint_scores = np.asarray(vectors["joint_scores"], dtype=np.float64)
    best_single_tau_scores = np.asarray(vectors["best_single_tau_scores"], dtype=np.float64)
    ids = list(vectors["candidate_ids"])
    n_candidates = int(vectors["n_candidates"])
    n_traj = int(vectors["n_trajectories"])
    n_time = int(vectors["n_timebins"])
    n_units = int(vectors["n_units"])
    true_idx = int(vectors["true_candidate_index"])

    true_log_prior = log_prior if log_prior.ndim == 1 else log_prior[true_idx]
    true_log_posterior_unnorm = prior_ll[true_idx] + true_log_prior
    posterior = posterior_from_log_scores(true_log_posterior_unnorm)
    neff = effective_count(posterior)
    max_posterior = float(np.nanmax(posterior)) if posterior.size else float("nan")
    best_tau_index = int(np.nanargmax(posterior)) if posterior.size and np.isfinite(posterior).any() else -1
    true_tau = -1 if true_trajectory_index is None else int(true_trajectory_index)
    nearest_tau = -1 if nearest_trajectory_index is None else int(nearest_trajectory_index)

    out: dict[str, Any] = {
        "observer": "backimage_trajectory_table_image_identity",
        "likelihood_family": "poisson_expected_count",
        "eps": float(eps),
        "likelihood_scale": float(likelihood_scale),
        "n_candidates": int(n_candidates),
        "n_trajectories": int(n_traj),
        "n_timebins": int(n_time),
        "n_units": int(n_units),
        "true_candidate_index": int(true_idx),
        "true_image_id": ids[true_idx],
        "posterior_N_eff_true_image": float(neff),
        "N_eff_true_image": float(neff),
        "N_eff_true_image_fraction": float(neff / n_traj) if np.isfinite(neff) else float("nan"),
        "posterior_entropy_true_image": entropy(posterior),
        "max_tau_posterior_true_image": max_posterior,
        "best_tau_posterior_index": best_tau_index,
        "true_tau_rank": rank_desc(true_log_posterior_unnorm, true_tau) if 0 <= true_tau < n_traj else float("nan"),
        "nearest_tau_rank": (
            rank_desc(true_log_posterior_unnorm, nearest_tau) if 0 <= nearest_tau < n_traj else float("nan")
        ),
        "nearest_tau_distance": float(nearest_trajectory_distance) if nearest_trajectory_distance is not None else float("nan"),
        "best_single_tau_score": float(best_single_tau_scores[true_idx]),
        "joint_vs_best_single_tau_gap": float(best_single_tau_scores[true_idx] - joint_scores[true_idx]),
        "joint_minus_zero_true_score": float(joint_scores[true_idx] - zero_scores[true_idx]),
        "known_minus_zero_true_score": float(known_scores[true_idx] - zero_scores[true_idx]),
        "best_single_tau_minus_joint_true_score": float(best_single_tau_scores[true_idx] - joint_scores[true_idx]),
        # Backward-compatible names retained for existing result readers. These
        # are not strict oracle upper bounds on accuracy; they are single-tau
        # maximization diagnostics.
        "best_tau_oracle_score": float(best_single_tau_scores[true_idx]),
        "joint_vs_best_dilution_gap": float(best_single_tau_scores[true_idx] - joint_scores[true_idx]),
        "best_oracle_minus_joint_true_score": float(best_single_tau_scores[true_idx] - joint_scores[true_idx]),
    }
    out.update(_score_summary("known", known_scores, true_idx, ids))
    out.update(_score_summary("zero", zero_scores, true_idx, ids))
    out.update(_score_summary("joint", joint_scores, true_idx, ids))
    out.update(_score_summary("best_single_tau", best_single_tau_scores, true_idx, ids))
    out.update(_score_summary("best_trajectory_oracle", best_single_tau_scores, true_idx, ids))
    return out


def summarize_observer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate trial rows into compact summary rows."""
    if not rows:
        return []
    group_keys = [
        "candidate_set_mode",
        "observation_condition",
        "observation_family",
        "observation_scale",
        "prior_condition",
        "prior_family",
        "prior_scale",
        "axis_catalog_mode",
        "trajectory_prior_mode",
        "likelihood_scale",
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        groups.setdefault(key, []).append(row)

    def mean_bool(rs: list[dict[str, Any]], key: str) -> float:
        vals = [bool(r.get(key, False)) for r in rs]
        return float(np.mean(vals)) if vals else float("nan")

    def median_float(rs: list[dict[str, Any]], key: str) -> float:
        vals = np.asarray([float(r.get(key, np.nan)) for r in rs], dtype=np.float64)
        return float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")

    out = []
    for key, rs in sorted(groups.items(), key=lambda item: item[0]):
        row = {k: v for k, v in zip(group_keys, key, strict=True)}
        row.update(
            {
                "n_trials": int(len(rs)),
                "known_eye_accuracy": mean_bool(rs, "known_correct"),
                "zero_eye_accuracy": mean_bool(rs, "zero_correct"),
                "joint_eye_accuracy": mean_bool(rs, "joint_correct"),
                "best_single_tau_accuracy": mean_bool(rs, "best_single_tau_correct"),
                "best_trajectory_oracle_accuracy": mean_bool(rs, "best_trajectory_oracle_correct"),
                "joint_minus_zero_accuracy": mean_bool(rs, "joint_correct") - mean_bool(rs, "zero_correct"),
                "known_minus_zero_accuracy": mean_bool(rs, "known_correct") - mean_bool(rs, "zero_correct"),
                "best_single_tau_minus_joint_accuracy": mean_bool(rs, "best_single_tau_correct")
                - mean_bool(rs, "joint_correct"),
                "best_oracle_minus_joint_accuracy": mean_bool(rs, "best_trajectory_oracle_correct")
                - mean_bool(rs, "joint_correct"),
                "median_known_rank": median_float(rs, "known_true_rank"),
                "median_zero_rank": median_float(rs, "zero_true_rank"),
                "median_joint_rank": median_float(rs, "joint_true_rank"),
                "median_best_single_tau_rank": median_float(rs, "best_single_tau_true_rank"),
                "median_best_trajectory_oracle_rank": median_float(rs, "best_trajectory_oracle_true_rank"),
                "median_N_eff_fraction": median_float(rs, "N_eff_true_image_fraction"),
                "median_nearest_tau_rank": median_float(rs, "nearest_tau_rank"),
                "median_joint_vs_best_single_tau_gap": median_float(rs, "joint_vs_best_single_tau_gap"),
                "median_joint_vs_best_dilution_gap": median_float(rs, "joint_vs_best_dilution_gap"),
                "median_joint_true_margin": median_float(rs, "joint_true_margin"),
                "median_known_true_margin": median_float(rs, "known_true_margin"),
                "median_zero_true_margin": median_float(rs, "zero_true_margin"),
            }
        )
        out.append(row)
    return out
