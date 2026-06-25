"""Quadratic no-anchor joint feature-recovery diagnostic for Figure 4C.

The linear no-anchor observer is a hard endpoint: trajectory recovery can fail
even when the posterior is near the right image feature. This script tests the
first principled nonlinear extension suggested by the polynomial observation
diagnostic: an origin-constrained quadratic local response map

    z_t ~= B_I [x_t, y_t, x_t^2, x_t y_t, y_t^2].

For each image candidate, B_I is fit from trajectory-held-out response-cache
rows. The observed compact residual is then scored by a small MAP/profile solve
over a 2D AR(1) path, with no selected anchor trajectory. The endpoint is
feature recovery cosine, not exact latent trace recovery.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from declan.backimage_trajectory_observer.analyze_continuous_joint_trajectory import (
    _load_npz,
    _trajectory_xy_by_candidate,
    ar1_profile_log_score,
    project_response_delta,
)
from declan.backimage_trajectory_observer.likelihood import (
    effective_count,
    entropy,
    posterior_from_log_scores,
    rank_desc,
    true_margin,
)
from declan.backimage_trajectory_observer.observer import (
    feature_recovery_metrics,
    posterior_weighted_feature,
    score_image_identity_score_vectors,
)
from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_joint_feature_recovery import (
    FEATURE_NPZ,
    PRIMARY_LATENT,
    _load_feature_tables,
    _source_row_from_candidate_id,
)
from declan.figure4_active_sensing_atlas.scripts.build_panel_c_lagged_observation_diagnostic import (
    MANIFEST,
    OUT_DIR,
    SIDECAR_ROOT,
    SOURCE_ROOT,
    _clean_axis,
    _configure_matplotlib,
    _energy_r2,
    _load_basis,
    _ridge_fit,
)
from declan.figure4_active_sensing_atlas.scripts.build_panel_c_polynomial_observation_diagnostic import (
    PolySpec,
    _poly_design,
)


QUADRATIC_SPEC = PolySpec("quadratic", 2, False)
OBSERVER_ORDER = [
    "zero",
    "joint",
    "best_single_tau",
    "linear_continuous",
    "quadratic_profile",
    "quadratic_poisson",
    "known",
]
OBSERVER_LABELS = {
    "zero": "zero",
    "joint": "finite joint",
    "best_single_tau": "best catalog",
    "linear_continuous": "linear profile",
    "quadratic_profile": "quad profile",
    "quadratic_poisson": "quad Poisson",
    "known": "known",
}
COLORS = {
    "zero": "#6b7280",
    "joint": "#235789",
    "best_single_tau": "#b35c2e",
    "linear_continuous": "#2f8f6a",
    "quadratic_profile": "#8a5ca8",
    "quadratic_poisson": "#d62728",
    "known": "#111827",
}


@dataclass(frozen=True)
class QuadraticFit:
    coef: np.ndarray
    residual_var: float
    train_r2: float
    n_fit_trajectories: int
    n_fit_samples: int


def _candidate_ids(table: dict[str, np.ndarray], n_candidates: int) -> list[str]:
    if "candidate_ids" not in table:
        return [str(i) for i in range(int(n_candidates))]
    return [str(v) for v in np.asarray(table["candidate_ids"]).tolist()]


def _scalar_int(table: dict[str, np.ndarray], key: str, default: int = -1) -> int:
    if key not in table:
        return int(default)
    arr = np.asarray(table[key]).reshape(-1)
    return int(arr[0]) if arr.size else int(default)


def _parse_float_schedule(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("schedule must contain at least one value")
    if not np.isfinite(values).all():
        raise ValueError("schedule values must be finite")
    return values


def _parse_csv_values(text: str) -> set[str]:
    return {part.strip() for part in str(text).split(",") if part.strip()}


def _observed_xy(table: dict[str, np.ndarray], true_idx: int, true_tau_idx: int, xy: np.ndarray) -> np.ndarray | None:
    for key in ["observed_trajectory_xy", "true_trajectory_xy", "y_obs_trajectory_xy"]:
        if key in table:
            arr = np.asarray(table[key], dtype=np.float64)
            return arr.reshape(arr.shape[-2], 2) if arr.ndim >= 2 else arr
    if 0 <= int(true_tau_idx) < xy.shape[1]:
        return np.asarray(xy[int(true_idx), int(true_tau_idx)], dtype=np.float64)
    return None


def _fit_quadratic_map(
    z_candidate: np.ndarray,
    xy_candidate: np.ndarray,
    *,
    heldout_trajectory_index: int,
    ridge: float,
) -> QuadraticFit:
    z = np.asarray(z_candidate, dtype=np.float64)
    xy = np.asarray(xy_candidate, dtype=np.float64)
    keep = np.ones(xy.shape[0], dtype=bool)
    if 0 <= int(heldout_trajectory_index) < keep.shape[0] and keep.shape[0] > 1:
        keep[int(heldout_trajectory_index)] = False
    design = _poly_design(xy[keep], QUADRATIC_SPEC)
    x_flat = design.reshape(-1, design.shape[2])
    y_flat = z[keep].reshape(-1, z.shape[2])
    good = np.isfinite(x_flat).all(axis=1) & np.isfinite(y_flat).all(axis=1)
    if int(np.sum(good)) < x_flat.shape[1]:
        raise ValueError("too few finite samples to fit quadratic local map")
    x_fit = x_flat[good]
    y_fit = y_flat[good]
    coef = _ridge_fit(x_fit, y_fit, float(ridge))
    pred = x_fit @ coef
    resid = y_fit - pred
    return QuadraticFit(
        coef=coef.T,
        residual_var=float(np.mean(resid * resid)),
        train_r2=float(_energy_r2(y_fit, pred)),
        n_fit_trajectories=int(np.sum(keep)),
        n_fit_samples=int(x_fit.shape[0]),
    )


def _fit_linear_map(
    z_candidate: np.ndarray,
    xy_candidate: np.ndarray,
    *,
    heldout_trajectory_index: int,
    ridge: float,
) -> QuadraticFit:
    z = np.asarray(z_candidate, dtype=np.float64)
    xy = np.asarray(xy_candidate, dtype=np.float64)
    keep = np.ones(xy.shape[0], dtype=bool)
    if 0 <= int(heldout_trajectory_index) < keep.shape[0] and keep.shape[0] > 1:
        keep[int(heldout_trajectory_index)] = False
    design = _poly_design(xy[keep], PolySpec("linear", 1, False))
    x_flat = design.reshape(-1, design.shape[2])
    y_flat = z[keep].reshape(-1, z.shape[2])
    good = np.isfinite(x_flat).all(axis=1) & np.isfinite(y_flat).all(axis=1)
    if int(np.sum(good)) < x_flat.shape[1]:
        raise ValueError("too few finite samples to fit linear local map")
    x_fit = x_flat[good]
    y_fit = y_flat[good]
    coef = _ridge_fit(x_fit, y_fit, float(ridge))
    pred = x_fit @ coef
    resid = y_fit - pred
    return QuadraticFit(
        coef=coef.T,
        residual_var=float(np.mean(resid * resid)),
        train_r2=float(_energy_r2(y_fit, pred)),
        n_fit_trajectories=int(np.sum(keep)),
        n_fit_samples=int(x_fit.shape[0]),
    )


def _quadratic_compact_delta(path: np.ndarray, coef: np.ndarray) -> np.ndarray:
    tau = np.asarray(path, dtype=np.float64)
    if tau.ndim != 2 or tau.shape[1] != 2:
        raise ValueError(f"path must be (time,2), got {tau.shape}")
    phi = _poly_design(tau[None, :, :], QUADRATIC_SPEC)[0]
    return phi @ np.asarray(coef, dtype=np.float64).T


def _quadratic_objective_and_grad(
    flat_path: np.ndarray,
    z_obs: np.ndarray,
    coef: np.ndarray,
    *,
    quadratic_scale: float,
    observation_var: float,
    alpha: float,
    process_var: float,
    initial_mean: np.ndarray | None,
    initial_var: float,
) -> tuple[float, np.ndarray]:
    tau = np.asarray(flat_path, dtype=np.float64).reshape(-1, 2)
    z = np.asarray(z_obs, dtype=np.float64)
    b = np.asarray(coef, dtype=np.float64)
    x = tau[:, 0]
    y = tau[:, 1]
    q_scale = float(quadratic_scale)
    phi = np.stack([x, y, q_scale * x * x, q_scale * x * y, q_scale * y * y], axis=1)
    pred = phi @ b.T
    err = pred - z
    r = max(float(observation_var), 1e-12)
    value = 0.5 * float(np.sum(err * err) / r)

    d_pred_dx = b[:, 0][None, :] + q_scale * (
        2.0 * x[:, None] * b[:, 2][None, :] + y[:, None] * b[:, 3][None, :]
    )
    d_pred_dy = b[:, 1][None, :] + q_scale * (
        x[:, None] * b[:, 3][None, :] + 2.0 * y[:, None] * b[:, 4][None, :]
    )
    grad = np.empty_like(tau)
    grad[:, 0] = np.sum(err * d_pred_dx, axis=1) / r
    grad[:, 1] = np.sum(err * d_pred_dy, axis=1) / r

    p0 = max(float(initial_var), 1e-12)
    if initial_mean is None:
        initial_target = np.zeros(2, dtype=np.float64)
    else:
        initial_target = np.asarray(initial_mean, dtype=np.float64)
    diff0 = tau[0] - initial_target
    value += 0.5 * float(np.sum(diff0 * diff0) / p0)
    grad[0] += diff0 / p0

    q = max(float(process_var), 1e-12)
    a = float(alpha)
    if tau.shape[0] > 1:
        step = tau[1:] - a * tau[:-1]
        value += 0.5 * float(np.sum(step * step) / q)
        grad[1:] += step / q
        grad[:-1] += -a * step / q
    return value, grad.reshape(-1)


def _profile_quadratic_path(
    z_obs: np.ndarray,
    coef: np.ndarray,
    *,
    starts: list[np.ndarray],
    observation_var: float,
    alpha: float,
    process_var: float,
    initial_mean: np.ndarray | None,
    initial_var: float,
    max_iter: int,
    quadratic_scales: list[float],
    observation_scales: list[float],
) -> dict[str, Any]:
    q_scales = [float(value) for value in quadratic_scales]
    obs_scales = [float(value) for value in observation_scales]
    if not q_scales:
        q_scales = [1.0]
    if len(obs_scales) == 1 and len(q_scales) > 1:
        obs_scales = obs_scales * len(q_scales)
    if len(q_scales) != len(obs_scales):
        raise ValueError("quadratic_scales and observation_scales must have equal length, or one observation scale")
    best: dict[str, Any] | None = None
    for start_index, start in enumerate(starts):
        current = np.asarray(start, dtype=np.float64).reshape(-1)
        stage_results = []
        for stage_index, (q_scale, obs_scale) in enumerate(zip(q_scales, obs_scales)):
            result = minimize(
                lambda flat: _quadratic_objective_and_grad(
                    flat,
                    z_obs,
                    coef,
                    quadratic_scale=float(q_scale),
                    observation_var=float(observation_var) * float(obs_scale),
                    alpha=float(alpha),
                    process_var=float(process_var),
                    initial_mean=initial_mean,
                    initial_var=float(initial_var),
                ),
                current,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": int(max_iter), "ftol": 1e-8, "gtol": 1e-5, "maxls": 30},
            )
            current = np.asarray(result.x, dtype=np.float64)
            stage_results.append(result)
        energy, _grad = _quadratic_objective_and_grad(
            current,
            z_obs,
            coef,
            quadratic_scale=1.0,
            observation_var=float(observation_var),
            alpha=float(alpha),
            process_var=float(process_var),
            initial_mean=initial_mean,
            initial_var=float(initial_var),
        )
        final_result = stage_results[-1]
        total_iterations = int(sum(int(result.nit) for result in stage_results))
        if best is None or energy < float(best["energy"]):
            best = {
                "energy": float(energy),
                "profile_score": -float(energy),
                "path": current.reshape(np.asarray(start, dtype=np.float64).shape),
                "start_index": int(start_index),
                "optimizer_success": bool(final_result.success),
                "optimizer_iterations": total_iterations,
                "optimizer_final_iterations": int(final_result.nit),
                "optimizer_stage_success_fraction": float(np.mean([bool(result.success) for result in stage_results])),
                "optimizer_message": str(final_result.message),
            }
    if best is None:
        raise RuntimeError("quadratic path optimizer did not run")
    return best


def _feature_matrix(candidate_ids: list[str], feature_table: dict[int, np.ndarray]) -> np.ndarray:
    return np.stack([feature_table[_source_row_from_candidate_id(candidate_id)] for candidate_id in candidate_ids], axis=0)


def _feature_rows_for_scores(
    *,
    base: dict[str, Any],
    observer_mode: str,
    scores: np.ndarray,
    candidate_ids: list[str],
    true_idx: int,
    features: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vals = np.asarray(scores, dtype=np.float64)
    posterior = posterior_from_log_scores(vals)
    pred_idx = int(np.nanargmax(vals)) if vals.size else -1
    true_features = features[int(true_idx)]
    z_hat, posterior_for_feature = posterior_weighted_feature(vals, features)
    metrics = feature_recovery_metrics(z_hat, true_features)
    map_metrics = (
        feature_recovery_metrics(features[pred_idx], true_features)
        if 0 <= pred_idx < features.shape[0]
        else {key: float("nan") for key in metrics}
    )
    summary = {
        **base,
        "observer_mode": str(observer_mode),
        "latent": PRIMARY_LATENT,
        "image_correct": bool(pred_idx == int(true_idx)),
        "true_rank": rank_desc(vals, int(true_idx)),
        "true_margin": true_margin(vals, int(true_idx)),
        "candidate_posterior_true_mass": float(posterior_for_feature[int(true_idx)]),
        "candidate_posterior_entropy": entropy(posterior_for_feature),
        "candidate_posterior_N_eff": effective_count(posterior_for_feature),
        "candidate_posterior_N_eff_fraction": float(effective_count(posterior_for_feature) / vals.size),
        "pred_candidate_index_local": int(pred_idx),
        "true_candidate_index_local": int(true_idx),
        "pred_candidate_id": str(candidate_ids[pred_idx]) if 0 <= pred_idx < len(candidate_ids) else "",
        "true_candidate_id": str(candidate_ids[int(true_idx)]),
        **metrics,
        "map_feature_cosine": map_metrics["feature_cosine"],
        "map_feature_mse": map_metrics["feature_mse"],
        "map_feature_rmse": map_metrics["feature_rmse"],
    }
    rows = []
    for candidate_index, candidate_id in enumerate(candidate_ids):
        rows.append(
            {
                **base,
                "observer_mode": str(observer_mode),
                "candidate_index": int(candidate_index),
                "candidate_id": str(candidate_id),
                "is_true_candidate": bool(candidate_index == int(true_idx)),
                "candidate_score": float(vals[candidate_index]),
                "candidate_posterior": float(posterior[candidate_index]),
            }
        )
    return rows, summary


def _score_table(
    *,
    table: dict[str, np.ndarray],
    manifest_row: pd.Series,
    table_index: int,
    basis: np.ndarray,
    ridge: float,
    alpha: float,
    process_var: float,
    observation_var_floor: float,
    initial_position: str,
    initial_position_var: float,
    max_iter: int,
    quadratic_scales: list[float],
    observation_scales: list[float],
    feature_table: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prior = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
    zero = np.asarray(table["zero_lambda_counts"], dtype=np.float64)
    known = np.asarray(table["known_lambda_counts"], dtype=np.float64)
    obs = np.asarray(table["y_obs_counts"], dtype=np.float64)
    n_candidates, n_traj, n_time, n_units = prior.shape
    true_idx = _scalar_int(table, "true_candidate_index", 0)
    true_tau_idx = _scalar_int(table, "true_trajectory_index", -1)
    candidate_ids = _candidate_ids(table, n_candidates)
    xy = _trajectory_xy_by_candidate(
        np.asarray(table["prior_trajectory_xy"], dtype=np.float64),
        n_candidates=n_candidates,
        n_trajectories=n_traj,
        n_time=n_time,
    )
    observed_xy = _observed_xy(table, true_idx, true_tau_idx, xy)
    features = _feature_matrix(candidate_ids, feature_table)
    z_catalog = project_response_delta(prior - zero[:, None, :, :], basis)

    finite = score_image_identity_score_vectors(
        y_obs_counts=obs,
        prior_lambda_counts=prior,
        known_lambda_counts=known,
        zero_lambda_counts=zero,
        true_candidate_index=true_idx,
        candidate_ids=candidate_ids,
    )

    quad_profile_scores = np.full(n_candidates, np.nan, dtype=np.float64)
    quad_poisson_scores = np.full(n_candidates, np.nan, dtype=np.float64)
    linear_profile_scores = np.full(n_candidates, np.nan, dtype=np.float64)
    qc_rows: list[dict[str, Any]] = []
    for candidate_index in range(n_candidates):
        fit = _fit_quadratic_map(
            z_catalog[candidate_index],
            xy[candidate_index],
            heldout_trajectory_index=true_tau_idx,
            ridge=float(ridge),
        )
        z_obs = project_response_delta(obs - zero[candidate_index], basis)
        r_var = max(float(observation_var_floor), float(fit.residual_var))
        init_mean = None
        if str(initial_position) == "known_start":
            if observed_xy is None:
                raise ValueError("known_start requested but observed trajectory is unavailable")
            init_mean = np.asarray(observed_xy[0], dtype=np.float64)
        linear_fit = _fit_linear_map(
            z_catalog[candidate_index],
            xy[candidate_index],
            heldout_trajectory_index=true_tau_idx,
            ridge=float(ridge),
        )
        linear_r_var = max(float(observation_var_floor), float(linear_fit.residual_var))
        linear_out = ar1_profile_log_score(
            z_obs,
            linear_fit.coef,
            alpha=float(alpha),
            process_var=float(process_var),
            observation_var=linear_r_var,
            initial_mean=init_mean,
            initial_var=float(initial_position_var),
        )
        linear_profile_scores[candidate_index] = float(linear_out["profile_score"])
        linear_path = np.asarray(linear_out["map_means"], dtype=np.float64)
        starts = [
            linear_path,
            np.zeros((n_time, 2), dtype=np.float64),
            np.mean(xy[candidate_index], axis=0),
        ]
        if str(initial_position) == "known_start":
            starts = [start.copy() for start in starts]
            for start in starts:
                start[0] = init_mean
        opt = _profile_quadratic_path(
            z_obs,
            fit.coef,
            starts=starts,
            observation_var=r_var,
            alpha=float(alpha),
            process_var=float(process_var),
            initial_mean=init_mean,
            initial_var=float(initial_position_var),
            max_iter=int(max_iter),
            quadratic_scales=quadratic_scales,
            observation_scales=observation_scales,
        )
        tau_hat = np.asarray(opt["path"], dtype=np.float64)
        compact_delta = _quadratic_compact_delta(tau_hat, fit.coef)
        full_delta = compact_delta @ basis.T
        pred_counts = np.maximum(zero[candidate_index] + full_delta, 1e-8)
        quad_profile_scores[candidate_index] = float(opt["profile_score"])
        quad_poisson_scores[candidate_index] = float(np.sum(obs * np.log(pred_counts) - pred_counts))

        qc_rows.append(
            {
                "table_index": int(table_index),
                "trial_id": int(manifest_row.get("trial_id", table_index)),
                "response_cache_path": str(manifest_row["response_cache_path"]),
                "prior_scale": float(manifest_row.get("scale", np.nan)),
                "candidate_index": int(candidate_index),
                "true_candidate_index": int(true_idx),
                "n_fit_trajectories": int(fit.n_fit_trajectories),
                "n_fit_samples": int(fit.n_fit_samples),
                "linear_train_r2": float(linear_fit.train_r2),
                "linear_residual_var": float(linear_fit.residual_var),
                "quadratic_train_r2": float(fit.train_r2),
                "quadratic_residual_var": float(fit.residual_var),
                "profile_energy": float(opt["energy"]),
                "profile_start_index": int(opt["start_index"]),
                "optimizer_success": bool(opt["optimizer_success"]),
                "optimizer_iterations": int(opt["optimizer_iterations"]),
                "optimizer_final_iterations": int(opt["optimizer_final_iterations"]),
                "optimizer_stage_success_fraction": float(opt["optimizer_stage_success_fraction"]),
                "quadratic_continuation_scales": ",".join(f"{value:g}" for value in quadratic_scales),
                "observation_continuation_scales": ",".join(f"{value:g}" for value in observation_scales),
                "trajectory_hat_rms": float(np.sqrt(np.mean(tau_hat * tau_hat))),
                "trajectory_hat_temporal_std_mean": float(np.mean(np.std(tau_hat, axis=0))),
            }
        )

    base = {
        "table_index": int(table_index),
        "trial_id": int(manifest_row.get("trial_id", table_index)),
        "response_cache_path": str(manifest_row["response_cache_path"]),
        "candidate_set_mode": str(manifest_row.get("candidate_set_mode", "")),
        "observation_family": str(manifest_row.get("observation_family", "")),
        "prior_family": str(manifest_row.get("prior_family", "")),
        "prior_scale": float(manifest_row.get("scale", np.nan)),
        "n_candidates": int(n_candidates),
        "n_trajectories": int(n_traj),
        "n_timebins": int(n_time),
        "n_units": int(n_units),
        "basis_dim": int(basis.shape[1]),
        "ridge": float(ridge),
        "alpha": float(alpha),
        "process_var": float(process_var),
        "observation_var_floor": float(observation_var_floor),
        "trajectory_initial_position": str(initial_position),
        "trajectory_initial_position_var": float(initial_position_var),
        "optimizer_max_iter": int(max_iter),
        "quadratic_continuation_scales": ",".join(f"{value:g}" for value in quadratic_scales),
        "observation_continuation_scales": ",".join(f"{value:g}" for value in observation_scales),
    }
    score_vectors = {
        "zero": np.asarray(finite["zero_scores"], dtype=np.float64),
        "joint": np.asarray(finite["joint_scores"], dtype=np.float64),
        "best_single_tau": np.asarray(finite["best_single_tau_scores"], dtype=np.float64),
        "linear_continuous": linear_profile_scores,
        "quadratic_profile": quad_profile_scores,
        "quadratic_poisson": quad_poisson_scores,
        "known": np.asarray(finite["known_scores"], dtype=np.float64),
    }
    posterior_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for observer_mode, scores in score_vectors.items():
        rows, summary = _feature_rows_for_scores(
            base=base,
            observer_mode=observer_mode,
            scores=scores,
            candidate_ids=candidate_ids,
            true_idx=true_idx,
            features=features,
        )
        posterior_rows.extend(rows)
        metric_rows.append(summary)
    return posterior_rows, metric_rows, qc_rows


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["observer_mode", "prior_scale"], as_index=False, sort=False)
        .agg(
            n=("feature_cosine", "size"),
            image_accuracy=("image_correct", "mean"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_map_feature_cosine=("map_feature_cosine", "mean"),
            mean_true_mass=("candidate_posterior_true_mass", "mean"),
            median_N_eff_fraction=("candidate_posterior_N_eff_fraction", "median"),
        )
    )


def _overall(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for observer_mode, group in summary.groupby("observer_mode", sort=False):
        weights = group["n"].astype(float).to_numpy()
        rows.append(
            {
                "observer_mode": str(observer_mode),
                "prior_scale": "all",
                "n": int(group["n"].sum()),
                "image_accuracy": float(np.average(group["image_accuracy"], weights=weights)),
                "mean_feature_cosine": float(np.average(group["mean_feature_cosine"], weights=weights)),
                "median_feature_cosine": float(group["median_feature_cosine"].median()),
                "mean_map_feature_cosine": float(np.average(group["mean_map_feature_cosine"], weights=weights)),
                "mean_true_mass": float(np.average(group["mean_true_mass"], weights=weights)),
                "median_N_eff_fraction": float(group["median_N_eff_fraction"].median()),
            }
        )
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, overall: pd.DataFrame, suffix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    order = [mode for mode in OBSERVER_ORDER if mode in set(overall["observer_mode"])]
    x = np.arange(len(order), dtype=float)
    block = overall.set_index("observer_mode").reindex(order)
    axes[0].bar(
        x,
        block["mean_feature_cosine"],
        color=[COLORS.get(mode, "#4b5563") for mode in order],
    )
    axes[0].set_title("Posterior feature recovery")
    axes[0].set_ylabel("mean feature cosine")
    axes[0].set_xticks(x, [OBSERVER_LABELS.get(mode, mode) for mode in order], rotation=25, ha="right")
    axes[0].set_ylim(0.0, 1.02)
    _clean_axis(axes[0])

    for mode in ["linear_continuous", "quadratic_profile", "quadratic_poisson", "joint", "known"]:
        block = summary[summary["observer_mode"].eq(mode)].sort_values("prior_scale")
        if block.empty:
            continue
        axes[1].plot(
            block["prior_scale"].astype(float),
            block["mean_feature_cosine"],
            marker="o",
            lw=1.6,
            color=COLORS.get(mode, "#4b5563"),
            label=OBSERVER_LABELS.get(mode, mode),
        )
    axes[1].set_title("By motion scale")
    axes[1].set_xlabel("scale")
    axes[1].set_ylabel("mean feature cosine")
    axes[1].set_xticks([0.5, 1.0, 2.0], ["0.5x", "1x", "2x"])
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend(frameon=False)
    _clean_axis(axes[1])
    fig.suptitle("Quadratic no-anchor joint feature diagnostic")
    fig.savefig(OUT_DIR / f"continuous_joint_quadratic_feature_diagnostic{suffix}.png", dpi=220)
    fig.savefig(OUT_DIR / f"continuous_joint_quadratic_feature_diagnostic{suffix}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tables", type=int, default=96, help="0 means all manifest rows.")
    parser.add_argument("--skip-tables", type=int, default=0)
    parser.add_argument(
        "--prior-family-filter",
        default="",
        help="Optional comma-separated prior_family values to keep before skip/max table slicing.",
    )
    parser.add_argument(
        "--scale-filter",
        default="",
        help="Optional comma-separated scale values to keep before skip/max table slicing.",
    )
    parser.add_argument("--basis-max-dim", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=0.92)
    parser.add_argument("--process-var", type=float, default=1e-3)
    parser.add_argument("--observation-var-floor", type=float, default=1e-6)
    parser.add_argument(
        "--trajectory-initial-position",
        choices=("inferred", "known_start"),
        default="inferred",
    )
    parser.add_argument("--trajectory-initial-position-var", type=float, default=1e-4)
    parser.add_argument("--optimizer-max-iter", type=int, default=80)
    parser.add_argument(
        "--quadratic-continuation-scales",
        default="1",
        help="Comma-separated schedule for turning on quadratic terms during path profiling.",
    )
    parser.add_argument(
        "--observation-continuation-scales",
        default="1",
        help="Comma-separated multipliers on observation variance for each continuation stage.",
    )
    parser.add_argument("--suffix", default="subset96")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest[manifest["response_cache_path"].astype(str).str.len() > 0].copy()
    prior_family_filter = _parse_csv_values(str(args.prior_family_filter))
    if prior_family_filter:
        manifest = manifest[manifest["prior_family"].astype(str).isin(prior_family_filter)].copy()
    scale_filter = _parse_csv_values(str(args.scale_filter))
    if scale_filter:
        scale_values = {float(value) for value in scale_filter}
        manifest = manifest[manifest["scale"].astype(float).isin(scale_values)].copy()
    skip_tables = max(0, int(args.skip_tables))
    if skip_tables:
        manifest = manifest.iloc[skip_tables:].copy()
    if int(args.max_tables) > 0:
        manifest = manifest.head(int(args.max_tables)).copy()
    quadratic_scales = _parse_float_schedule(str(args.quadratic_continuation_scales))
    observation_scales = _parse_float_schedule(str(args.observation_continuation_scales))
    if len(observation_scales) not in {1, len(quadratic_scales)}:
        raise ValueError("--observation-continuation-scales must contain one value or match --quadratic-continuation-scales")
    if any(value < 0.0 for value in quadratic_scales):
        raise ValueError("--quadratic-continuation-scales values must be non-negative")
    if any(value <= 0.0 for value in observation_scales):
        raise ValueError("--observation-continuation-scales values must be positive")
    feature_tables = _load_feature_tables()
    feature_table = feature_tables[PRIMARY_LATENT]

    posterior_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    basis_meta: dict[str, Any] | None = None
    progress_every = max(1, int(args.progress_every))
    for table_index, (_idx, row) in enumerate(manifest.iterrows()):
        table_path = SOURCE_ROOT / str(row["response_cache_path"])
        sidecar_path = SIDECAR_ROOT / str(row["response_cache_path"])
        table = {**_load_npz(table_path), **_load_npz(sidecar_path)}
        n_units = int(np.asarray(table["prior_lambda_counts"]).shape[-1])
        basis, basis_meta = _load_basis(n_units, int(args.basis_max_dim))
        post, metrics, qc = _score_table(
            table=table,
            manifest_row=row,
            table_index=table_index,
            basis=basis,
            ridge=float(args.ridge),
            alpha=float(args.alpha),
            process_var=float(args.process_var),
            observation_var_floor=float(args.observation_var_floor),
            initial_position=str(args.trajectory_initial_position),
            initial_position_var=float(args.trajectory_initial_position_var),
            max_iter=int(args.optimizer_max_iter),
            quadratic_scales=quadratic_scales,
            observation_scales=observation_scales,
            feature_table=feature_table,
        )
        posterior_rows.extend(post)
        metric_rows.extend(metrics)
        qc_rows.extend(qc)
        if table_index == 0 or (table_index + 1) % progress_every == 0 or table_index + 1 == len(manifest):
            print(f"[quadratic-feature] scored {table_index + 1}/{len(manifest)} tables", flush=True)

    suffix = str(args.suffix)
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    posterior_df = pd.DataFrame(posterior_rows)
    metrics_df = pd.DataFrame(metric_rows)
    qc_df = pd.DataFrame(qc_rows)
    summary = _summarize(metrics_df)
    overall = _overall(summary)
    posterior_df.to_csv(OUT_DIR / f"continuous_joint_quadratic_feature_posterior{suffix}.csv", index=False)
    metrics_df.to_csv(OUT_DIR / f"continuous_joint_quadratic_feature_trials{suffix}.csv", index=False)
    qc_df.to_csv(OUT_DIR / f"continuous_joint_quadratic_feature_qc{suffix}.csv", index=False)
    summary.to_csv(OUT_DIR / f"continuous_joint_quadratic_feature_summary{suffix}.csv", index=False)
    overall.to_csv(OUT_DIR / f"continuous_joint_quadratic_feature_overall{suffix}.csv", index=False)
    _plot(summary, overall, suffix)

    readme = [
        "# Quadratic Joint Feature Diagnostic",
        "",
        "No-anchor diagnostic using an origin-constrained quadratic compact response map and feature cosine endpoint.",
        "",
        f"Feature source: `{FEATURE_NPZ}`",
        f"Basis: `{(basis_meta or {}).get('basis_source', '')}`",
        f"Basis dim: {(basis_meta or {}).get('basis_dim', '')}",
        f"Manifest rows: {len(manifest)}",
        f"Skip tables: {skip_tables}",
        f"Prior family filter: `{','.join(sorted(prior_family_filter))}`",
        f"Scale filter: `{','.join(sorted(scale_filter))}`",
        f"Ridge: {float(args.ridge):g}",
        f"Initial position mode: `{args.trajectory_initial_position}`",
        f"Quadratic continuation scales: `{','.join(f'{value:g}' for value in quadratic_scales)}`",
        f"Observation continuation scales: `{','.join(f'{value:g}' for value in observation_scales)}`",
        "",
        "Overall:",
        "",
        overall.to_string(index=False),
        "",
    ]
    (OUT_DIR / f"continuous_joint_quadratic_feature_README{suffix}.md").write_text(
        "\n".join(readme),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
