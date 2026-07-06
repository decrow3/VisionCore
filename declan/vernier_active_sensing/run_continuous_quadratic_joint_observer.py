#!/usr/bin/env python3
"""Continuous quadratic Vernier joint-decoding diagnostic.

This is a second-pass joint observer for the Vernier cache that avoids an MLP.
It fits an interpretable local map from eye position to a compact response
basis,

    z_t ~= B_theta [x_t, y_t, x_t^2, x_t y_t, y_t^2],

then decodes Vernier sign by profiling over a continuous Brownian trajectory.
The model uses cached ConvGRU responses only; optional synthetic Brownian walks
are used as optimization starts, not as additional rendered response samples.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .joint_observer import THETA_LABELS, THETA_MINUS, THETA_PLUS
from .metrics import expected_counts
from .run_second_pass_joint_diagnostic import (
    build_scale_policy_summaries,
    selection_scope_keys,
    summarize_rows,
)
from .run_trajectory_table_observer import (
    parse_csv_float,
    parse_csv_str,
    write_csv,
    write_json,
)
from .synthetic_trajectory_priors import (
    SyntheticTrajectoryPriorConfig,
    generate_synthetic_trajectory_prior,
    recommended_empirical_confined_config,
)

POSE_UNITS_PER_DEGREE = 60.0
PROCESS_MODELS = ("matched_brownian", "synthetic_empirical_confined")
CATALOG_MODES = ("include_self", "leave_one_out")
FEATURE_MODES = ("current_quadratic", "history_quadratic")
BASIS_MODES = ("pca", "sign_delta", "pca_plus_sign_delta")
RESIDUAL_VARIANCE_MODES = ("label", "pooled")
DEFAULT_COV_FLOOR_ARCMIN2 = 1e-4


@dataclass(frozen=True)
class FeatureConfig:
    mode: str = "current_quadratic"
    history_lags: tuple[int, ...] = (0,)
    include_velocity: bool = False
    velocity_lags: tuple[int, ...] = (0,)
    filter_alphas: tuple[float, ...] = ()


@dataclass(frozen=True)
class RatePoseCache:
    path: Path
    condition: str
    fd_step_arcmin: float
    inference_mode: str
    plus_rates: list[np.ndarray]
    minus_rates: list[np.ndarray]
    poses_arcmin: list[np.ndarray]


@dataclass(frozen=True)
class BrownianPrior:
    model: str
    init_mean: np.ndarray
    step_mean: np.ndarray
    init_cov: np.ndarray
    step_cov: np.ndarray
    init_inv: np.ndarray
    step_inv: np.ndarray
    init_logdet: float
    step_logdet: float
    process_cov_scale: float
    init_step_mean: np.ndarray | None = None
    init_step_cov: np.ndarray | None = None
    init_step_inv: np.ndarray | None = None
    init_step_logdet: float | None = None
    transition_beta: float | None = None
    position_spring_kappa: float | None = None
    innovation_mean: np.ndarray | None = None
    innovation_cov: np.ndarray | None = None
    innovation_inv: np.ndarray | None = None
    innovation_logdet: float | None = None


@dataclass(frozen=True)
class QuadraticMap:
    basis: np.ndarray
    coef_by_label: dict[str, np.ndarray]
    residual_var_by_label: dict[str, float]
    zero_counts_by_label: dict[str, np.ndarray]
    feature_config: FeatureConfig
    basis_mode: str
    residual_variance_mode: str
    n_pose_features: int
    train_indices: np.ndarray
    basis_dim_effective: int
    ridge: float


@dataclass(frozen=True)
class ProfileScore:
    score: float
    energy: float
    obs_energy: float
    prior_energy: float
    best_start_index: int
    n_starts: int
    n_iter: int
    success: bool
    rmse_arcmin: float
    trajectory: np.ndarray


@dataclass(frozen=True)
class PreparedQuadraticProfile:
    qmap: QuadraticMap
    prior: BrownianPrior
    starts: list[np.ndarray]
    train_indices: np.ndarray
    n_catalog_trajectories: int
    basis_dim_requested: int
    covariance_floor_arcmin2: float
    n_catalog_starts: int
    n_brownian_starts: int
    n_synthetic_prior_samples: int
    synthetic_prior_metadata: dict[str, Any]
    shared_basis: bool
    include_self: bool


def parse_catalog_modes(text: str) -> list[str]:
    modes = parse_csv_str(text)
    if not modes:
        raise ValueError("At least one catalog mode is required")
    bad = [mode for mode in modes if mode not in CATALOG_MODES]
    if bad:
        raise ValueError(f"Unsupported catalog modes {bad}; expected {CATALOG_MODES}")
    return modes


def parse_csv_int(text: str) -> list[int]:
    return [int(part) for part in parse_csv_str(text)]


def _positive_unique_ints(values: list[int], *, include_zero: bool = True) -> tuple[int, ...]:
    out = sorted({int(value) for value in values if int(value) >= 0})
    if include_zero and 0 not in out:
        out.insert(0, 0)
    return tuple(out)


def make_feature_config(
    *,
    mode: str,
    history_lags: list[int],
    include_velocity: bool,
    velocity_lags: list[int],
    filter_alphas: list[float],
) -> FeatureConfig:
    if str(mode) not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature mode {mode!r}; expected {FEATURE_MODES}")
    alphas = tuple(float(alpha) for alpha in filter_alphas)
    if any(alpha <= 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("--filter-alphas must be in the interval (0, 1]")
    if str(mode) == "current_quadratic":
        return FeatureConfig(mode="current_quadratic")
    return FeatureConfig(
        mode="history_quadratic",
        history_lags=_positive_unique_ints(history_lags or [0], include_zero=True),
        include_velocity=bool(include_velocity),
        velocity_lags=_positive_unique_ints(velocity_lags or [0], include_zero=True),
        filter_alphas=alphas,
    )


def _condition_from_cache_path(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("rates_") or "_fd" not in stem:
        raise ValueError(f"Unexpected rate cache filename: {path.name}")
    return stem[len("rates_") : stem.rindex("_fd")]


def _unpadded(arr: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(arr[i, : int(lengths[i])], dtype=np.float64) for i in range(arr.shape[0])]


def _load_rate_pose_caches(source_dir: Path) -> dict[tuple[str, float, str], RatePoseCache]:
    caches: dict[tuple[str, float, str], RatePoseCache] = {}
    for path in sorted((source_dir / "cache").glob("rates_*_fd*arcmin.npz")):
        with np.load(path, allow_pickle=True) as npz:
            if "poses" not in npz:
                continue
            condition = str(npz["condition"][0]) if "condition" in npz else _condition_from_cache_path(path)
            fd_step = float(np.asarray(npz["fd_step_arcmin"])[0])
            inference_mode = str(npz["inference_mode"][0]) if "inference_mode" in npz else "framewise"
            lengths = np.asarray(npz["lengths"], dtype=np.int32)
            plus_rates = _unpadded(np.asarray(npz["plus"], dtype=np.float64), lengths)
            minus_rates = _unpadded(np.asarray(npz["minus"], dtype=np.float64), lengths)
            poses_arcmin = [
                pose * POSE_UNITS_PER_DEGREE
                for pose in _unpadded(np.asarray(npz["poses"], dtype=np.float64), lengths)
            ]
        caches[(condition, fd_step, inference_mode)] = RatePoseCache(
            path=path,
            condition=condition,
            fd_step_arcmin=fd_step,
            inference_mode=inference_mode,
            plus_rates=plus_rates,
            minus_rates=minus_rates,
            poses_arcmin=poses_arcmin,
        )
    if not caches:
        raise FileNotFoundError(f"No pose-bearing rate caches found under {source_dir / 'cache'}")
    return caches


def _select_caches(
    caches: dict[tuple[str, float, str], RatePoseCache],
    *,
    conditions: list[str],
    fd_steps: list[float],
) -> list[RatePoseCache]:
    selected: list[RatePoseCache] = []
    condition_filter = set(conditions)
    fd_filter = {round(float(fd), 10) for fd in fd_steps}
    for (_condition, fd_step, _inference), cache in sorted(caches.items()):
        if condition_filter and cache.condition not in condition_filter:
            continue
        if fd_filter and round(float(fd_step), 10) not in fd_filter:
            continue
        selected.append(cache)
    return selected


def _stack_counts(rates: list[np.ndarray], *, bin_seconds: float, max_timebins: int) -> np.ndarray:
    if not rates:
        raise ValueError("Cannot stack an empty rate list")
    t = min(arr.shape[0] for arr in rates)
    if int(max_timebins) > 0:
        t = min(t, int(max_timebins))
    if t <= 0:
        raise ValueError("No time bins available after truncation")
    units = {int(arr.shape[1]) for arr in rates}
    if len(units) != 1:
        raise ValueError(f"Rate arrays must have one unit count, got {sorted(units)}")
    stacked = np.stack([arr[:t] for arr in rates], axis=0)
    if not np.isfinite(stacked).all():
        raise ValueError("Rate cache contains non-finite values")
    return expected_counts(stacked, float(bin_seconds))


def _stack_poses(poses: list[np.ndarray], *, max_timebins: int, target_timebins: int | None = None) -> np.ndarray:
    if not poses:
        raise ValueError("Cannot stack an empty pose list")
    t = min(arr.shape[0] for arr in poses)
    if int(max_timebins) > 0:
        t = min(t, int(max_timebins))
    if target_timebins is not None:
        t = min(t, int(target_timebins))
    stacked = np.stack([arr[:t] for arr in poses], axis=0)
    if stacked.shape[-1] != 2:
        raise ValueError(f"Expected poses shaped (..., 2), got {stacked.shape}")
    if not np.isfinite(stacked).all():
        raise ValueError("Pose cache contains non-finite values")
    return stacked


def _cache_counts(cache: RatePoseCache, *, bin_seconds: float, max_timebins: int) -> dict[str, np.ndarray]:
    return {
        THETA_PLUS: _stack_counts(cache.plus_rates, bin_seconds=bin_seconds, max_timebins=max_timebins),
        THETA_MINUS: _stack_counts(cache.minus_rates, bin_seconds=bin_seconds, max_timebins=max_timebins),
    }


def _mean_reference_counts(
    cache: RatePoseCache,
    *,
    bin_seconds: float,
    max_timebins: int,
    target_timebins: int,
) -> dict[str, np.ndarray]:
    counts = _cache_counts(cache, bin_seconds=bin_seconds, max_timebins=max_timebins)
    t = min(int(target_timebins), counts[THETA_PLUS].shape[1], counts[THETA_MINUS].shape[1])
    return {
        THETA_PLUS: np.mean(counts[THETA_PLUS][:, :t], axis=0),
        THETA_MINUS: np.mean(counts[THETA_MINUS][:, :t], axis=0),
    }


def _truncate_counts_and_poses(
    observed_counts: dict[str, np.ndarray],
    prior_counts: dict[str, np.ndarray],
    zero_counts: dict[str, np.ndarray],
    observed_poses: np.ndarray,
    prior_poses: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, int]:
    t = min(
        observed_counts[THETA_PLUS].shape[1],
        observed_counts[THETA_MINUS].shape[1],
        prior_counts[THETA_PLUS].shape[1],
        prior_counts[THETA_MINUS].shape[1],
        zero_counts[THETA_PLUS].shape[0],
        zero_counts[THETA_MINUS].shape[0],
        observed_poses.shape[1],
        prior_poses.shape[1],
    )
    obs = {label: arr[:, :t] for label, arr in observed_counts.items()}
    prior = {label: arr[:, :t] for label, arr in prior_counts.items()}
    zero = {label: arr[:t] for label, arr in zero_counts.items()}
    return obs, prior, zero, observed_poses[:, :t], prior_poses[:, :t], int(t)


def _quadratic_design(poses_arcmin: np.ndarray) -> np.ndarray:
    pose = np.asarray(poses_arcmin, dtype=np.float64)
    x = pose[..., 0]
    y = pose[..., 1]
    return np.stack([x, y, x * x, x * y, y * y], axis=-1)


def _lag_time(values: np.ndarray, lag: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"Expected an array with a time axis, got shape {arr.shape}")
    lag_i = max(0, int(lag))
    t = arr.shape[-2]
    indices = np.maximum(np.arange(t) - lag_i, 0)
    return arr[..., indices, :]


def _velocity(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.diff(arr, axis=-2, prepend=arr[..., :1, :])


def _exp_filter_time(values: np.ndarray, alpha: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.empty_like(arr)
    a = float(alpha)
    out[..., 0, :] = arr[..., 0, :]
    for t in range(1, arr.shape[-2]):
        out[..., t, :] = a * arr[..., t, :] + (1.0 - a) * out[..., t - 1, :]
    return out


def _pose_design(poses_arcmin: np.ndarray, config: FeatureConfig | None = None) -> np.ndarray:
    cfg = config or FeatureConfig()
    pose = np.asarray(poses_arcmin, dtype=np.float64)
    if cfg.mode == "current_quadratic":
        return _quadratic_design(pose)
    if cfg.mode != "history_quadratic":
        raise ValueError(f"Unsupported feature mode {cfg.mode!r}")

    blocks: list[np.ndarray] = []
    for lag in cfg.history_lags:
        blocks.append(_quadratic_design(_lag_time(pose, lag)))
    if cfg.include_velocity:
        step = _velocity(pose)
        for lag in cfg.velocity_lags:
            blocks.append(_quadratic_design(_lag_time(step, lag)))
    for alpha in cfg.filter_alphas:
        blocks.append(_quadratic_design(_exp_filter_time(pose, alpha)))
    if not blocks:
        raise ValueError("History feature design has no feature blocks")
    return np.concatenate(blocks, axis=-1)


def _build_response_basis(
    counts_by_label: dict[str, np.ndarray],
    zero_counts_by_label: dict[str, np.ndarray],
    train_indices: np.ndarray,
    *,
    basis_dim: int,
    basis_mode: str = "pca",
    sign_basis_weight: float = 1.0,
) -> tuple[np.ndarray, int]:
    if str(basis_mode) not in BASIS_MODES:
        raise ValueError(f"Unsupported basis mode {basis_mode!r}; expected {BASIS_MODES}")
    blocks: list[np.ndarray] = []
    if str(basis_mode) in {"pca", "pca_plus_sign_delta"}:
        for label in THETA_LABELS:
            residual = counts_by_label[label][train_indices] - zero_counts_by_label[label][None, :, :]
            blocks.append(residual.reshape(-1, residual.shape[-1]))
    if str(basis_mode) in {"sign_delta", "pca_plus_sign_delta"}:
        delta = counts_by_label[THETA_PLUS][train_indices] - counts_by_label[THETA_MINUS][train_indices]
        blocks.append(float(sign_basis_weight) * delta.reshape(-1, delta.shape[-1]))
    matrix = np.concatenate(blocks, axis=0)
    if matrix.size == 0:
        raise ValueError("Cannot build a response basis from an empty training set")
    if matrix.shape[0] >= matrix.shape[1]:
        gram = matrix.T @ matrix
        gram = 0.5 * (gram + gram.T)
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        tol = max(matrix.shape) * np.finfo(np.float64).eps * max(float(eigvals[0]), 1.0)
        rank = int(np.sum(eigvals > tol))
        k = max(1, min(int(basis_dim), eigvecs.shape[1], rank if rank > 0 else eigvecs.shape[1]))
        return np.asarray(eigvecs[:, :k], dtype=np.float64), k
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular_values > max(matrix.shape) * np.finfo(np.float64).eps * singular_values[0]))
    k = max(1, min(int(basis_dim), vt.shape[0], rank if rank > 0 else vt.shape[0]))
    return np.asarray(vt[:k].T, dtype=np.float64), k


def fit_quadratic_map(
    counts_by_label: dict[str, np.ndarray],
    poses_arcmin: np.ndarray,
    zero_counts_by_label: dict[str, np.ndarray],
    train_indices: np.ndarray,
    *,
    basis_dim: int = 20,
    ridge: float = 1e-3,
    residual_floor: float = 1e-8,
    basis: np.ndarray | None = None,
    feature_config: FeatureConfig | None = None,
    basis_mode: str = "pca",
    sign_basis_weight: float = 1.0,
    residual_variance_mode: str = "label",
) -> QuadraticMap:
    cfg = feature_config or FeatureConfig()
    if str(residual_variance_mode) not in RESIDUAL_VARIANCE_MODES:
        raise ValueError(
            f"Unsupported residual variance mode {residual_variance_mode!r}; expected {RESIDUAL_VARIANCE_MODES}"
        )
    train = np.asarray(train_indices, dtype=np.int64)
    if train.size <= 0:
        raise ValueError("At least one training trajectory is required")
    if basis is None:
        basis, k_eff = _build_response_basis(
            counts_by_label,
            zero_counts_by_label,
            train,
            basis_dim=int(basis_dim),
            basis_mode=str(basis_mode),
            sign_basis_weight=float(sign_basis_weight),
        )
    else:
        basis = np.asarray(basis, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[0] != counts_by_label[THETA_PLUS].shape[-1]:
            raise ValueError(
                f"Shared basis must be shaped (units, k), got {basis.shape} "
                f"for {counts_by_label[THETA_PLUS].shape[-1]} units"
            )
        k_eff = int(basis.shape[1])
    design_full = _pose_design(poses_arcmin[train], cfg)
    n_pose_features = int(design_full.shape[-1])
    design = design_full.reshape(-1, n_pose_features)
    xtx = design.T @ design
    regularized = xtx + float(ridge) * np.eye(xtx.shape[0], dtype=np.float64)
    coef_by_label: dict[str, np.ndarray] = {}
    residual_var_by_label: dict[str, float] = {}
    for label in THETA_LABELS:
        residual = counts_by_label[label][train] - zero_counts_by_label[label][None, :, :]
        z = residual.reshape(-1, residual.shape[-1]) @ basis
        coef_t = np.linalg.solve(regularized, design.T @ z)
        coef = coef_t.T
        pred = design @ coef_t
        err = pred - z
        residual_var_by_label[label] = max(float(np.mean(err * err)), float(residual_floor))
        coef_by_label[label] = coef
    if str(residual_variance_mode) == "pooled":
        pooled = max(float(np.mean(list(residual_var_by_label.values()))), float(residual_floor))
        residual_var_by_label = {label: pooled for label in THETA_LABELS}
    return QuadraticMap(
        basis=basis,
        coef_by_label=coef_by_label,
        residual_var_by_label=residual_var_by_label,
        zero_counts_by_label={label: np.asarray(value, dtype=np.float64) for label, value in zero_counts_by_label.items()},
        feature_config=cfg,
        basis_mode=str(basis_mode),
        residual_variance_mode=str(residual_variance_mode),
        n_pose_features=n_pose_features,
        train_indices=train,
        basis_dim_effective=k_eff,
        ridge=float(ridge),
    )


def _regularized_cov(samples: np.ndarray, *, floor: float) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] <= 1:
        cov = np.eye(arr.shape[1], dtype=np.float64) * float(floor)
    else:
        cov = np.cov(arr, rowvar=False)
        cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    cov = 0.5 * (cov + cov.T)
    eigvals = np.linalg.eigvalsh(cov)
    min_eig = float(np.min(eigvals)) if eigvals.size else 0.0
    if not np.isfinite(min_eig) or min_eig < float(floor):
        cov = cov + (float(floor) - min_eig if np.isfinite(min_eig) else float(floor)) * np.eye(cov.shape[0])
    return cov


def fit_brownian_prior(
    poses_arcmin: np.ndarray,
    train_indices: np.ndarray,
    *,
    process_cov_scale: float = 1.0,
    covariance_floor_arcmin2: float = DEFAULT_COV_FLOOR_ARCMIN2,
) -> BrownianPrior:
    train = np.asarray(train_indices, dtype=np.int64)
    if train.size <= 0:
        raise ValueError("At least one training trajectory is required")
    pose = np.asarray(poses_arcmin[train], dtype=np.float64)
    init = pose[:, 0, :]
    steps = np.diff(pose, axis=1).reshape(-1, 2) if pose.shape[1] > 1 else np.zeros((pose.shape[0], 2))
    init_cov = _regularized_cov(init, floor=float(covariance_floor_arcmin2))
    step_cov = _regularized_cov(steps, floor=float(covariance_floor_arcmin2))
    step_cov = step_cov * float(process_cov_scale)
    init_cov = init_cov * float(process_cov_scale)
    init_sign, init_logdet = np.linalg.slogdet(init_cov)
    step_sign, step_logdet = np.linalg.slogdet(step_cov)
    if init_sign <= 0 or step_sign <= 0:
        raise ValueError("Regularized Brownian covariance must be positive definite")
    return BrownianPrior(
        model="matched_brownian",
        init_mean=np.mean(init, axis=0),
        step_mean=np.mean(steps, axis=0),
        init_cov=init_cov,
        step_cov=step_cov,
        init_inv=np.linalg.inv(init_cov),
        step_inv=np.linalg.inv(step_cov),
        init_logdet=float(init_logdet),
        step_logdet=float(step_logdet),
        process_cov_scale=float(process_cov_scale),
    )


def fit_confined_step_prior(
    poses_arcmin: np.ndarray,
    *,
    process_cov_scale: float = 1.0,
    covariance_floor_arcmin2: float = DEFAULT_COV_FLOOR_ARCMIN2,
) -> BrownianPrior:
    """Fit a Gaussian confined-step process prior in arcmin units."""
    pose = np.asarray(poses_arcmin, dtype=np.float64)
    if pose.ndim != 3 or pose.shape[-1] != 2:
        raise ValueError(f"poses_arcmin must be (trajectory, time, 2), got {pose.shape}")
    if pose.shape[0] <= 0:
        raise ValueError("At least one synthetic trajectory is required")
    init = pose[:, 0, :]
    steps_by_trace = np.diff(pose, axis=1)
    steps = steps_by_trace.reshape(-1, 2) if steps_by_trace.size else np.zeros((pose.shape[0], 2))
    if pose.shape[1] >= 3:
        prev_step = steps_by_trace[:, :-1, :]
        next_step = steps_by_trace[:, 1:, :]
        prev_pose = pose[:, 1:-1, :]
        design = np.column_stack([prev_step.reshape(-1), -prev_pose.reshape(-1)])
        target = next_step.reshape(-1)
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        beta = float(np.clip(coef[0], -0.98, 0.98))
        kappa = float(max(coef[1], 0.0))
        residual = next_step - (beta * prev_step - kappa * prev_pose)
        innovation = residual.reshape(-1, 2)
    else:
        beta = 0.0
        kappa = 0.0
        innovation = steps
    init_cov = _regularized_cov(init, floor=float(covariance_floor_arcmin2)) * float(process_cov_scale)
    step_cov = _regularized_cov(steps, floor=float(covariance_floor_arcmin2)) * float(process_cov_scale)
    init_step = steps_by_trace[:, 0, :] if steps_by_trace.shape[1] > 0 else np.zeros((pose.shape[0], 2))
    init_step_cov = _regularized_cov(init_step, floor=float(covariance_floor_arcmin2)) * float(process_cov_scale)
    innovation_cov = _regularized_cov(innovation, floor=float(covariance_floor_arcmin2)) * float(process_cov_scale)
    init_sign, init_logdet = np.linalg.slogdet(init_cov)
    step_sign, step_logdet = np.linalg.slogdet(step_cov)
    init_step_sign, init_step_logdet = np.linalg.slogdet(init_step_cov)
    innovation_sign, innovation_logdet = np.linalg.slogdet(innovation_cov)
    if min(init_sign, step_sign, init_step_sign, innovation_sign) <= 0:
        raise ValueError("Regularized confined-step covariances must be positive definite")
    return BrownianPrior(
        model="synthetic_empirical_confined",
        init_mean=np.mean(init, axis=0),
        step_mean=np.mean(steps, axis=0),
        init_cov=init_cov,
        step_cov=step_cov,
        init_inv=np.linalg.inv(init_cov),
        step_inv=np.linalg.inv(step_cov),
        init_logdet=float(init_logdet),
        step_logdet=float(step_logdet),
        process_cov_scale=float(process_cov_scale),
        init_step_mean=np.mean(init_step, axis=0),
        init_step_cov=init_step_cov,
        init_step_inv=np.linalg.inv(init_step_cov),
        init_step_logdet=float(init_step_logdet),
        transition_beta=float(beta),
        position_spring_kappa=float(kappa),
        innovation_mean=np.mean(innovation, axis=0),
        innovation_cov=innovation_cov,
        innovation_inv=np.linalg.inv(innovation_cov),
        innovation_logdet=float(innovation_logdet),
    )


def _predict_from_pose_features(
    poses_arcmin: np.ndarray,
    coef: np.ndarray,
    feature_config: FeatureConfig | None = None,
) -> np.ndarray:
    design = _pose_design(poses_arcmin, feature_config)
    return design @ np.asarray(coef, dtype=np.float64).T


def _quadratic_pred_and_jacobian(
    poses_arcmin: np.ndarray,
    coef: np.ndarray,
    feature_config: FeatureConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = feature_config or FeatureConfig()
    if cfg.mode != "current_quadratic":
        raise ValueError("Analytic pose gradients are only implemented for current_quadratic features")
    pose = np.asarray(poses_arcmin, dtype=np.float64)
    x = pose[:, 0]
    y = pose[:, 1]
    design = _quadratic_design(pose)
    pred = design @ coef.T
    dx = coef[:, 0][None, :] + 2.0 * x[:, None] * coef[:, 2][None, :] + y[:, None] * coef[:, 3][None, :]
    dy = coef[:, 1][None, :] + x[:, None] * coef[:, 3][None, :] + 2.0 * y[:, None] * coef[:, 4][None, :]
    return pred, dx, dy


def _prior_energy_and_grad(poses: np.ndarray, prior: BrownianPrior) -> tuple[float, np.ndarray]:
    pose = np.asarray(poses, dtype=np.float64)
    grad = np.zeros_like(pose)
    init_resid = pose[0] - prior.init_mean
    init_q = prior.init_inv @ init_resid
    prior_energy = 0.5 * float(init_resid @ init_q)
    grad[0] += init_q
    if str(prior.model) == "matched_brownian":
        if pose.shape[0] > 1:
            step_resid = pose[1:] - pose[:-1] - prior.step_mean[None, :]
            step_q = step_resid @ prior.step_inv.T
            prior_energy += 0.5 * float(np.sum(step_resid * step_q))
            grad[1:] += step_q
            grad[:-1] -= step_q
        return prior_energy, grad
    if str(prior.model) == "synthetic_empirical_confined":
        if pose.shape[0] <= 1:
            return prior_energy, grad
        if (
            prior.init_step_mean is None
            or prior.init_step_inv is None
            or prior.transition_beta is None
            or prior.position_spring_kappa is None
            or prior.innovation_mean is None
            or prior.innovation_inv is None
        ):
            raise ValueError("Confined-step prior is missing required parameters")
        steps = pose[1:] - pose[:-1]
        init_step_resid = steps[0] - prior.init_step_mean
        init_step_q = prior.init_step_inv @ init_step_resid
        prior_energy += 0.5 * float(init_step_resid @ init_step_q)
        grad[1] += init_step_q
        grad[0] -= init_step_q
        if pose.shape[0] > 2:
            beta = float(prior.transition_beta)
            kappa = float(prior.position_spring_kappa)
            resid = steps[1:] - (beta * steps[:-1] - kappa * pose[1:-1]) - prior.innovation_mean[None, :]
            q = resid @ prior.innovation_inv.T
            prior_energy += 0.5 * float(np.sum(resid * q))
            middle_coeff = -1.0 - beta + kappa
            grad[2:] += q
            grad[1:-1] += middle_coeff * q
            grad[:-2] += beta * q
        return prior_energy, grad
    raise ValueError(f"Unsupported process prior model {prior.model!r}")


def _brownian_prior_energy_and_grad(poses: np.ndarray, prior: BrownianPrior) -> tuple[float, np.ndarray]:
    pose = np.asarray(poses, dtype=np.float64)
    grad = np.zeros_like(pose)
    init_resid = pose[0] - prior.init_mean
    init_q = prior.init_inv @ init_resid
    prior_energy = 0.5 * float(init_resid @ init_q)
    grad[0] += init_q
    if pose.shape[0] > 1:
        step_resid = pose[1:] - pose[:-1] - prior.step_mean[None, :]
        step_q = step_resid @ prior.step_inv.T
        prior_energy += 0.5 * float(np.sum(step_resid * step_q))
        grad[1:] += step_q
        grad[:-1] -= step_q
    return prior_energy, grad


def _energy_value(
    flat_poses: np.ndarray,
    *,
    z_obs: np.ndarray,
    coef: np.ndarray,
    prior: BrownianPrior,
    residual_var: float,
    observation_weight: float,
    feature_config: FeatureConfig | None = None,
) -> tuple[float, float, float]:
    poses = np.asarray(flat_poses, dtype=np.float64).reshape(z_obs.shape[0], 2)
    pred = _predict_from_pose_features(poses, coef, feature_config)
    err = pred - z_obs
    inv_var = float(observation_weight) / max(float(residual_var), 1e-12)
    obs_energy = 0.5 * inv_var * float(np.sum(err * err))
    prior_energy, _grad = _prior_energy_and_grad(poses, prior)
    energy = obs_energy + prior_energy
    return energy, obs_energy, prior_energy


def _energy_and_grad(
    flat_poses: np.ndarray,
    *,
    z_obs: np.ndarray,
    coef: np.ndarray,
    prior: BrownianPrior,
    residual_var: float,
    observation_weight: float,
    feature_config: FeatureConfig | None = None,
) -> tuple[float, np.ndarray, float, float]:
    cfg = feature_config or FeatureConfig()
    if cfg.mode != "current_quadratic":
        raise ValueError("Analytic pose gradients are only implemented for current_quadratic features")
    poses = np.asarray(flat_poses, dtype=np.float64).reshape(z_obs.shape[0], 2)
    pred, dx, dy = _quadratic_pred_and_jacobian(poses, coef, cfg)
    err = pred - z_obs
    inv_var = float(observation_weight) / max(float(residual_var), 1e-12)
    obs_energy = 0.5 * inv_var * float(np.sum(err * err))
    grad = np.zeros_like(poses)
    grad[:, 0] += inv_var * np.sum(err * dx, axis=1)
    grad[:, 1] += inv_var * np.sum(err * dy, axis=1)

    prior_energy, prior_grad = _prior_energy_and_grad(poses, prior)
    grad += prior_grad
    energy = obs_energy + prior_energy
    return energy, grad.reshape(-1), obs_energy, prior_energy


def fixed_path_score(
    z_obs: np.ndarray,
    coef: np.ndarray,
    prior: BrownianPrior,
    poses_arcmin: np.ndarray,
    *,
    residual_var: float,
    observation_weight: float,
    feature_config: FeatureConfig | None = None,
) -> tuple[float, float, float, float]:
    energy, obs_energy, prior_energy = _energy_value(
        np.asarray(poses_arcmin, dtype=np.float64).reshape(-1),
        z_obs=z_obs,
        coef=coef,
        prior=prior,
        residual_var=float(residual_var),
        observation_weight=float(observation_weight),
        feature_config=feature_config,
    )
    return -float(energy), float(energy), float(obs_energy), float(prior_energy)


def _sample_brownian_paths(
    prior: BrownianPrior,
    *,
    n_paths: int,
    n_timebins: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    paths: list[np.ndarray] = []
    for _ in range(int(n_paths)):
        path = np.zeros((int(n_timebins), 2), dtype=np.float64)
        path[0] = rng.multivariate_normal(prior.init_mean, prior.init_cov)
        for t in range(1, int(n_timebins)):
            step = rng.multivariate_normal(prior.step_mean, prior.step_cov)
            path[t] = path[t - 1] + step
        paths.append(path)
    return paths


def _sample_confined_paths(
    prior: BrownianPrior,
    *,
    n_paths: int,
    n_timebins: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if (
        prior.init_step_mean is None
        or prior.init_step_cov is None
        or prior.transition_beta is None
        or prior.position_spring_kappa is None
        or prior.innovation_mean is None
        or prior.innovation_cov is None
    ):
        raise ValueError("Confined-step prior is missing required sampling parameters")
    paths: list[np.ndarray] = []
    beta = float(prior.transition_beta)
    kappa = float(prior.position_spring_kappa)
    for _ in range(int(n_paths)):
        path = np.zeros((int(n_timebins), 2), dtype=np.float64)
        path[0] = rng.multivariate_normal(prior.init_mean, prior.init_cov)
        if int(n_timebins) > 1:
            step = rng.multivariate_normal(prior.init_step_mean, prior.init_step_cov)
            path[1] = path[0] + step
            for t in range(2, int(n_timebins)):
                eps = rng.multivariate_normal(prior.innovation_mean, prior.innovation_cov)
                step = beta * step - kappa * path[t - 1] + eps
                path[t] = path[t - 1] + step
        paths.append(path)
    return paths


def _sample_prior_paths(
    prior: BrownianPrior,
    *,
    n_paths: int,
    n_timebins: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if int(n_paths) <= 0:
        return []
    if str(prior.model) == "matched_brownian":
        return _sample_brownian_paths(prior, n_paths=int(n_paths), n_timebins=int(n_timebins), rng=rng)
    if str(prior.model) == "synthetic_empirical_confined":
        return _sample_confined_paths(prior, n_paths=int(n_paths), n_timebins=int(n_timebins), rng=rng)
    raise ValueError(f"Unsupported process prior model {prior.model!r}")


def build_optimization_starts(
    train_poses_arcmin: np.ndarray,
    prior: BrownianPrior,
    *,
    n_catalog_starts: int,
    n_brownian_starts: int,
    seed: int,
) -> list[np.ndarray]:
    poses = np.asarray(train_poses_arcmin, dtype=np.float64)
    t = poses.shape[1]
    starts: list[np.ndarray] = [
        np.mean(poses, axis=0),
        np.zeros((t, 2), dtype=np.float64),
    ]
    for idx in range(min(int(n_catalog_starts), poses.shape[0])):
        starts.append(np.array(poses[idx], copy=True))
    if int(n_brownian_starts) > 0:
        rng = np.random.default_rng(int(seed))
        starts.extend(_sample_prior_paths(prior, n_paths=int(n_brownian_starts), n_timebins=t, rng=rng))
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for start in starts:
        key = np.round(start, decimals=6).tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(start)
    return unique


def profile_quadratic_score(
    z_obs: np.ndarray,
    coef: np.ndarray,
    prior: BrownianPrior,
    starts: list[np.ndarray],
    *,
    residual_var: float,
    observation_weight: float,
    max_iter: int,
    feature_config: FeatureConfig | None = None,
    true_pose_arcmin: np.ndarray | None = None,
) -> ProfileScore:
    if not starts:
        raise ValueError("At least one optimization start is required")
    if int(max_iter) <= 0:
        best_energy = float("inf")
        best_obs_energy = float("nan")
        best_prior_energy = float("nan")
        best_start = -1
        best_trajectory: np.ndarray | None = None
        for start_index, start in enumerate(starts):
            start_arr = np.asarray(start, dtype=np.float64)
            if start_arr.shape != (z_obs.shape[0], 2):
                raise ValueError(f"Start has shape {start_arr.shape}, expected {(z_obs.shape[0], 2)}")
            energy, obs_energy, prior_energy = _energy_value(
                start_arr.reshape(-1),
                z_obs=z_obs,
                coef=coef,
                prior=prior,
                residual_var=float(residual_var),
                observation_weight=float(observation_weight),
                feature_config=feature_config,
            )
            if float(energy) < best_energy:
                best_energy = float(energy)
                best_obs_energy = float(obs_energy)
                best_prior_energy = float(prior_energy)
                best_start = int(start_index)
                best_trajectory = np.array(start_arr, copy=True)
        if best_trajectory is None:
            raise RuntimeError("Path proposal profiling did not produce a result")
        if true_pose_arcmin is None:
            rmse = float("nan")
        else:
            diff = best_trajectory - np.asarray(true_pose_arcmin, dtype=np.float64)
            rmse = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        return ProfileScore(
            score=-best_energy,
            energy=best_energy,
            obs_energy=best_obs_energy,
            prior_energy=best_prior_energy,
            best_start_index=best_start,
            n_starts=len(starts),
            n_iter=0,
            success=True,
            rmse_arcmin=rmse,
            trajectory=best_trajectory,
        )
    best_result: Any | None = None
    best_energy = float("inf")
    best_obs_energy = float("nan")
    best_prior_energy = float("nan")
    best_start = -1
    for start_index, start in enumerate(starts):
        start_arr = np.asarray(start, dtype=np.float64)
        if start_arr.shape != (z_obs.shape[0], 2):
            raise ValueError(f"Start has shape {start_arr.shape}, expected {(z_obs.shape[0], 2)}")

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            energy, grad, _obs_energy, _prior_energy = _energy_and_grad(
                flat,
                z_obs=z_obs,
                coef=coef,
                prior=prior,
                residual_var=float(residual_var),
                observation_weight=float(observation_weight),
                feature_config=feature_config,
            )
            return energy, grad

        result = minimize(
            objective,
            start_arr.reshape(-1),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": int(max_iter), "ftol": 1e-9, "gtol": 1e-6, "maxls": 50},
        )
        energy, _grad, obs_energy, prior_energy = _energy_and_grad(
            result.x,
            z_obs=z_obs,
            coef=coef,
            prior=prior,
            residual_var=float(residual_var),
            observation_weight=float(observation_weight),
            feature_config=feature_config,
        )
        if float(energy) < best_energy:
            best_energy = float(energy)
            best_obs_energy = float(obs_energy)
            best_prior_energy = float(prior_energy)
            best_result = result
            best_start = int(start_index)
    if best_result is None:
        raise RuntimeError("Trajectory profiling did not produce a result")
    trajectory = np.asarray(best_result.x, dtype=np.float64).reshape(z_obs.shape[0], 2)
    if true_pose_arcmin is None:
        rmse = float("nan")
    else:
        diff = trajectory - np.asarray(true_pose_arcmin, dtype=np.float64)
        rmse = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    return ProfileScore(
        score=-best_energy,
        energy=best_energy,
        obs_energy=best_obs_energy,
        prior_energy=best_prior_energy,
        best_start_index=best_start,
        n_starts=len(starts),
        n_iter=int(getattr(best_result, "nit", -1)),
        success=bool(getattr(best_result, "success", False)),
        rmse_arcmin=rmse,
        trajectory=trajectory,
    )


def _predict(score_by_label: dict[str, float]) -> str:
    vals = np.asarray([score_by_label[THETA_PLUS], score_by_label[THETA_MINUS]], dtype=np.float64)
    if not np.isfinite(vals).all():
        return ""
    return THETA_PLUS if vals[0] >= vals[1] else THETA_MINUS


def _row_closure(joint_margin: float, known_margin: float, zero_margin: float) -> float:
    denom = float(known_margin) - float(zero_margin)
    if not np.isfinite(denom) or abs(denom) <= 1e-12:
        return float("nan")
    return (float(joint_margin) - float(zero_margin)) / denom


def prepare_quadratic_profile(
    prior_counts_by_label: dict[str, np.ndarray],
    zero_counts_by_label: dict[str, np.ndarray],
    prior_poses_arcmin: np.ndarray,
    *,
    true_trace_index: int,
    include_self: bool,
    basis_dim: int,
    ridge: float,
    process_model: str,
    process_cov_scale: float,
    covariance_floor_arcmin2: float,
    synthetic_prior_samples: int,
    synthetic_prior_kappa_weight_power: float,
    max_iter: int,
    n_catalog_starts: int,
    n_brownian_starts: int,
    seed: int,
    shared_basis: np.ndarray | None = None,
    feature_config: FeatureConfig | None = None,
    basis_mode: str = "pca",
    sign_basis_weight: float = 1.0,
    residual_variance_mode: str = "label",
) -> PreparedQuadraticProfile:
    del max_iter
    n_prior = int(prior_counts_by_label[THETA_PLUS].shape[0])
    trace_idx = int(true_trace_index)
    train_mask = np.ones(n_prior, dtype=bool)
    if not bool(include_self) and trace_idx < n_prior:
        train_mask[trace_idx] = False
    train_indices = np.flatnonzero(train_mask)
    if train_indices.size <= 0:
        raise ValueError("No prior trajectories remain after leave-one-out masking")

    qmap = fit_quadratic_map(
        prior_counts_by_label,
        prior_poses_arcmin,
        zero_counts_by_label,
        train_indices,
        basis_dim=int(basis_dim),
        ridge=float(ridge),
        basis=shared_basis,
        feature_config=feature_config,
        basis_mode=str(basis_mode),
        sign_basis_weight=float(sign_basis_weight),
        residual_variance_mode=str(residual_variance_mode),
    )
    prior = fit_brownian_prior(
        prior_poses_arcmin,
        train_indices,
        process_cov_scale=float(process_cov_scale),
        covariance_floor_arcmin2=float(covariance_floor_arcmin2),
    )
    synthetic_prior_metadata: dict[str, Any] = {}
    n_synthetic_prior_samples = 0
    process = str(process_model)
    if process == "matched_brownian":
        pass
    elif process == "synthetic_empirical_confined":
        n_synthetic_prior_samples = int(synthetic_prior_samples)
        if n_synthetic_prior_samples <= 0:
            raise ValueError("--synthetic-prior-samples must be positive for synthetic_empirical_confined")
        source_deg = np.asarray(prior_poses_arcmin[train_indices], dtype=np.float64) / POSE_UNITS_PER_DEGREE
        cfg: SyntheticTrajectoryPriorConfig = recommended_empirical_confined_config(
            kappa_weight_power=float(synthetic_prior_kappa_weight_power),
            covariance_mode="full_empirical",
            center_mode="zero_mean",
        )
        synthetic_result = generate_synthetic_trajectory_prior(
            source_deg,
            n_traces=n_synthetic_prior_samples,
            n_frames=prior_poses_arcmin.shape[1],
            seed=int(seed) + 271_828 * int(true_trace_index),
            config=cfg,
        )
        synthetic_poses_arcmin = np.asarray(synthetic_result.traces_deg, dtype=np.float64) * POSE_UNITS_PER_DEGREE
        prior = fit_confined_step_prior(
            synthetic_poses_arcmin,
            process_cov_scale=float(process_cov_scale),
            covariance_floor_arcmin2=float(covariance_floor_arcmin2),
        )
        synthetic_prior_metadata = {
            "synthetic_prior_source_model": str(synthetic_result.metadata.get("process_model", "")),
            "synthetic_prior_kappa_weight_power": float(synthetic_prior_kappa_weight_power),
            "synthetic_prior_empirical_param_count": int(
                synthetic_result.metadata.get("empirical_confined_param_count", 0)
            ),
            "synthetic_prior_sampled_kappa_median": float(
                synthetic_result.metadata.get("sampled_confined_kappa_median", float("nan"))
            ),
            "synthetic_prior_sampled_beta_median": float(
                synthetic_result.metadata.get("sampled_confined_beta_median", float("nan"))
            ),
            "synthetic_prior_fit_beta": float(prior.transition_beta)
            if prior.transition_beta is not None
            else float("nan"),
            "synthetic_prior_fit_kappa": float(prior.position_spring_kappa)
            if prior.position_spring_kappa is not None
            else float("nan"),
        }
    else:
        raise ValueError(f"Unsupported process_model={process!r}; expected {PROCESS_MODELS}")
    starts = build_optimization_starts(
        prior_poses_arcmin[train_indices],
        prior,
        n_catalog_starts=int(n_catalog_starts),
        n_brownian_starts=int(n_brownian_starts),
        seed=int(seed) + 104_729 * int(trace_idx),
    )
    return PreparedQuadraticProfile(
        qmap=qmap,
        prior=prior,
        starts=starts,
        train_indices=train_indices,
        n_catalog_trajectories=n_prior,
        basis_dim_requested=int(basis_dim),
        covariance_floor_arcmin2=float(covariance_floor_arcmin2),
        n_catalog_starts=int(n_catalog_starts),
        n_brownian_starts=int(n_brownian_starts),
        n_synthetic_prior_samples=int(n_synthetic_prior_samples),
        synthetic_prior_metadata=synthetic_prior_metadata,
        shared_basis=shared_basis is not None,
        include_self=bool(include_self),
    )


def score_prepared_quadratic_trial(
    observed: np.ndarray,
    true_label: str,
    true_trace_index: int,
    zero_counts_by_label: dict[str, np.ndarray],
    observed_poses_arcmin: np.ndarray,
    prepared: PreparedQuadraticProfile,
    *,
    observation_weight: float,
    max_iter: int,
) -> dict[str, Any]:
    true = str(true_label)
    other = THETA_MINUS if true == THETA_PLUS else THETA_PLUS
    trace_idx = int(true_trace_index)
    qmap = prepared.qmap
    prior = prepared.prior
    starts = prepared.starts
    feature_config = qmap.feature_config

    true_pose = observed_poses_arcmin[trace_idx]
    profile_by_label: dict[str, ProfileScore] = {}
    known_scores: dict[str, float] = {}
    zero_scores: dict[str, float] = {}
    for label in THETA_LABELS:
        z_obs = (np.asarray(observed, dtype=np.float64) - zero_counts_by_label[label]) @ qmap.basis
        coef = qmap.coef_by_label[label]
        residual_var = qmap.residual_var_by_label[label]
        profile_by_label[label] = profile_quadratic_score(
            z_obs,
            coef,
            prior,
            starts,
            residual_var=residual_var,
            observation_weight=float(observation_weight),
            max_iter=int(max_iter),
            feature_config=feature_config,
            true_pose_arcmin=true_pose if label == true else None,
        )
        known_scores[label], _known_energy, _known_obs, _known_prior = fixed_path_score(
            z_obs,
            coef,
            prior,
            true_pose,
            residual_var=residual_var,
            observation_weight=float(observation_weight),
            feature_config=feature_config,
        )
        zero_path = np.zeros_like(true_pose)
        zero_scores[label], _zero_energy, _zero_obs, _zero_prior = fixed_path_score(
            z_obs,
            coef,
            prior,
            zero_path,
            residual_var=residual_var,
            observation_weight=float(observation_weight),
            feature_config=feature_config,
        )

    joint_scores = {label: profile_by_label[label].score for label in THETA_LABELS}
    pred_joint = _predict(joint_scores)
    pred_known = _predict(known_scores)
    pred_zero = _predict(zero_scores)
    joint_margin = float(joint_scores[true] - joint_scores[other])
    known_margin = float(known_scores[true] - known_scores[other])
    zero_margin = float(zero_scores[true] - zero_scores[other])
    true_profile = profile_by_label[true]
    other_profile = profile_by_label[other]
    return {
        "readout": f"continuous_{feature_config.mode}_profile_vernier_llr",
        "trajectory_table_mode": f"continuous_{feature_config.mode}_pose_response_profile",
        "trajectory_prior": (
            f"{prior.model}_leave_one_out"
            if not bool(prepared.include_self)
            else f"{prior.model}_include_self"
        ),
        "observer_interpretation": (
            "Vernier likelihood-ratio diagnostic with a quadratic response basis and continuous pose profiling"
        ),
        "trajectory_table_include_self": bool(prepared.include_self),
        "trajectory_table_leave_one_out": not bool(prepared.include_self),
        "n_catalog_trajectories": int(prepared.n_catalog_trajectories),
        "n_joint_trajectories": int(prepared.train_indices.size),
        "true_trace_index": trace_idx,
        "true_label": true,
        "pred_joint": pred_joint,
        "pred_known": pred_known,
        "pred_zero": pred_zero,
        "pred_best_trajectory": "",
        "joint_correct": bool(pred_joint == true) if pred_joint else float("nan"),
        "known_correct": bool(pred_known == true) if pred_known else float("nan"),
        "zero_correct": bool(pred_zero == true) if pred_zero else float("nan"),
        "best_trajectory_correct": float("nan"),
        "decision_rule": "profiled_continuous_vernier_llr",
        "joint_likelihood_normalization": "quadratic_basis_gaussian_profile_energy",
        "joint_score_family": "continuous_quadratic_profile_energy",
        "likelihood_scale": float(observation_weight),
        "joint_evidence_is_normalized_log_probability": False,
        "joint_log_evidence_plus": float(joint_scores[THETA_PLUS]),
        "joint_log_evidence_minus": float(joint_scores[THETA_MINUS]),
        "known_log_evidence_plus": float(known_scores[THETA_PLUS]),
        "known_log_evidence_minus": float(known_scores[THETA_MINUS]),
        "zero_log_evidence_plus": float(zero_scores[THETA_PLUS]),
        "zero_log_evidence_minus": float(zero_scores[THETA_MINUS]),
        "joint_log_evidence_true": float(joint_scores[true]),
        "known_log_evidence_true": float(known_scores[true]),
        "zero_log_evidence_true": float(zero_scores[true]),
        "best_trajectory_log_evidence_plus": float("nan"),
        "best_trajectory_log_evidence_minus": float("nan"),
        "best_trajectory_log_evidence_true": float("nan"),
        "joint_score": joint_margin,
        "known_eye_score": known_margin,
        "zero_eye_score": zero_margin,
        "best_trajectory_score": float("nan"),
        "posterior_neff_true": float("nan"),
        "posterior_neff_plus": float("nan"),
        "posterior_neff_minus": float("nan"),
        "true_trajectory_rank_true": float("nan"),
        "true_trajectory_rank_plus": float("nan"),
        "true_trajectory_rank_minus": float("nan"),
        "gap_closure_vs_zero_known": float("nan"),
        "margin_gap_closure_vs_zero_known": _row_closure(joint_margin, known_margin, zero_margin),
        "basis_dim_requested": int(prepared.basis_dim_requested),
        "basis_dim_effective": int(qmap.basis_dim_effective),
        "shared_basis": bool(prepared.shared_basis),
        "basis_mode": str(qmap.basis_mode),
        "residual_variance_mode": str(qmap.residual_variance_mode),
        "feature_mode": str(feature_config.mode),
        "history_lags": ",".join(str(v) for v in feature_config.history_lags),
        "include_velocity": bool(feature_config.include_velocity),
        "velocity_lags": ",".join(str(v) for v in feature_config.velocity_lags),
        "filter_alphas": ",".join(f"{v:g}" for v in feature_config.filter_alphas),
        "n_pose_features": int(qmap.n_pose_features),
        "quadratic_ridge": float(qmap.ridge),
        "process_model": str(prior.model),
        "process_cov_scale": float(prior.process_cov_scale),
        "process_transition_beta": float(prior.transition_beta)
        if prior.transition_beta is not None
        else float("nan"),
        "process_position_spring_kappa": float(prior.position_spring_kappa)
        if prior.position_spring_kappa is not None
        else float("nan"),
        "covariance_floor_arcmin2": float(prepared.covariance_floor_arcmin2),
        "observation_weight": float(observation_weight),
        "residual_var_plus": float(qmap.residual_var_by_label[THETA_PLUS]),
        "residual_var_minus": float(qmap.residual_var_by_label[THETA_MINUS]),
        "n_optimization_starts": int(true_profile.n_starts),
        "n_catalog_starts": int(prepared.n_catalog_starts),
        "n_brownian_starts": int(prepared.n_brownian_starts),
        "n_synthetic_prior_samples": int(prepared.n_synthetic_prior_samples),
        **prepared.synthetic_prior_metadata,
        "profile_success_true": bool(true_profile.success),
        "profile_success_other": bool(other_profile.success),
        "profile_n_iter_true": int(true_profile.n_iter),
        "profile_n_iter_other": int(other_profile.n_iter),
        "profile_best_start_true": int(true_profile.best_start_index),
        "profile_best_start_other": int(other_profile.best_start_index),
        "profile_energy_true": float(true_profile.energy),
        "profile_energy_other": float(other_profile.energy),
        "profile_obs_energy_true": float(true_profile.obs_energy),
        "profile_obs_energy_other": float(other_profile.obs_energy),
        "profile_prior_energy_true": float(true_profile.prior_energy),
        "profile_prior_energy_other": float(other_profile.prior_energy),
        "profile_pose_rmse_arcmin_true": float(true_profile.rmse_arcmin),
    }


def score_continuous_quadratic_trial(
    observed: np.ndarray,
    true_label: str,
    true_trace_index: int,
    observed_counts_by_label: dict[str, np.ndarray],
    prior_counts_by_label: dict[str, np.ndarray],
    zero_counts_by_label: dict[str, np.ndarray],
    observed_poses_arcmin: np.ndarray,
    prior_poses_arcmin: np.ndarray,
    *,
    include_self: bool,
    basis_dim: int,
    ridge: float,
    process_model: str,
    process_cov_scale: float,
    covariance_floor_arcmin2: float,
    synthetic_prior_samples: int,
    synthetic_prior_kappa_weight_power: float,
    observation_weight: float,
    max_iter: int,
    n_catalog_starts: int,
    n_brownian_starts: int,
    seed: int,
    shared_basis: np.ndarray | None = None,
    feature_config: FeatureConfig | None = None,
    basis_mode: str = "pca",
    sign_basis_weight: float = 1.0,
    residual_variance_mode: str = "label",
) -> dict[str, Any]:
    del observed_counts_by_label
    prepared = prepare_quadratic_profile(
        prior_counts_by_label,
        zero_counts_by_label,
        prior_poses_arcmin,
        true_trace_index=int(true_trace_index),
        include_self=bool(include_self),
        basis_dim=int(basis_dim),
        ridge=float(ridge),
        process_model=str(process_model),
        process_cov_scale=float(process_cov_scale),
        covariance_floor_arcmin2=float(covariance_floor_arcmin2),
        synthetic_prior_samples=int(synthetic_prior_samples),
        synthetic_prior_kappa_weight_power=float(synthetic_prior_kappa_weight_power),
        max_iter=int(max_iter),
        n_catalog_starts=int(n_catalog_starts),
        n_brownian_starts=int(n_brownian_starts),
        seed=int(seed),
        shared_basis=shared_basis,
        feature_config=feature_config,
        basis_mode=str(basis_mode),
        sign_basis_weight=float(sign_basis_weight),
        residual_variance_mode=str(residual_variance_mode),
    )
    row = score_prepared_quadratic_trial(
        observed,
        true_label,
        true_trace_index,
        zero_counts_by_label,
        observed_poses_arcmin,
        prepared,
        observation_weight=float(observation_weight),
        max_iter=int(max_iter),
    )
    row["covariance_floor_arcmin2"] = float(covariance_floor_arcmin2)
    return row


def _score_trial_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_dir = Path(args.source_dir)
    caches = _load_rate_pose_caches(source_dir)
    selected = _select_caches(caches, conditions=args.conditions, fd_steps=args.fd_steps_arcmin)
    if not selected:
        raise ValueError("No selected pose-bearing rate caches matched the requested conditions/fd steps")
    feature_config = make_feature_config(
        mode=str(args.feature_mode),
        history_lags=list(args.history_lags),
        include_velocity=bool(args.include_velocity),
        velocity_lags=list(args.velocity_lags),
        filter_alphas=list(args.filter_alphas),
    )
    if feature_config.mode != "current_quadratic" and int(args.max_iter) > 0:
        raise ValueError("History-aware feature modes currently require --max-iter 0 proposal/known-trace scoring")

    prior_conditions = list(args.prior_conditions)
    trial_rows: list[dict[str, Any]] = []
    for cache in selected:
        condition = str(cache.condition)
        fd_step = float(cache.fd_step_arcmin)
        inference_mode = str(cache.inference_mode)
        observed_counts = _cache_counts(cache, bin_seconds=float(args.bin_seconds), max_timebins=int(args.max_timebins))
        observed_poses = _stack_poses(
            cache.poses_arcmin,
            max_timebins=int(args.max_timebins),
            target_timebins=observed_counts[THETA_PLUS].shape[1],
        )
        zero_key = (str(args.reference_condition), fd_step, inference_mode)
        if zero_key not in caches and not bool(args.allow_missing_reference):
            raise FileNotFoundError(
                f"Missing zero-eye reference cache for condition={args.reference_condition!r}, "
                f"fd_step={fd_step:g}, inference_mode={inference_mode!r}"
            )
        if zero_key in caches:
            zero_counts = _mean_reference_counts(
                caches[zero_key],
                bin_seconds=float(args.bin_seconds),
                max_timebins=int(args.max_timebins),
                target_timebins=observed_counts[THETA_PLUS].shape[1],
            )
            zero_ref_available = True
        else:
            zero_counts = {
                THETA_PLUS: np.mean(observed_counts[THETA_PLUS], axis=0),
                THETA_MINUS: np.mean(observed_counts[THETA_MINUS], axis=0),
            }
            zero_ref_available = False

        effective_prior_conditions = prior_conditions or [condition]
        for prior_condition in effective_prior_conditions:
            prior_key = (str(prior_condition), fd_step, inference_mode)
            if prior_key not in caches:
                raise FileNotFoundError(
                    f"Missing prior-condition cache for condition={prior_condition!r}, "
                    f"fd_step={fd_step:g}, inference_mode={inference_mode!r}"
                )
            prior_cache = caches[prior_key]
            prior_counts = _cache_counts(
                prior_cache,
                bin_seconds=float(args.bin_seconds),
                max_timebins=int(args.max_timebins),
            )
            prior_poses = _stack_poses(
                prior_cache.poses_arcmin,
                max_timebins=int(args.max_timebins),
                target_timebins=prior_counts[THETA_PLUS].shape[1],
            )
            obs_table, prior_table, zero_table, obs_pose_table, prior_pose_table, t = _truncate_counts_and_poses(
                observed_counts,
                prior_counts,
                zero_counts,
                observed_poses,
                prior_poses,
            )
            condition_matches_prior = condition == str(prior_condition)
            shared_basis = None
            if bool(args.shared_basis):
                shared_basis, _shared_k = _build_response_basis(
                    prior_table,
                    zero_table,
                    np.arange(prior_table[THETA_PLUS].shape[0], dtype=np.int64),
                    basis_dim=int(args.basis_dim),
                    basis_mode=str(args.basis_mode),
                    sign_basis_weight=float(args.sign_basis_weight),
                )
            requested_trace_indices = list(args.trace_indices)
            if requested_trace_indices:
                trace_indices = requested_trace_indices
            else:
                trace_indices = list(range(obs_table[THETA_PLUS].shape[0]))
            if int(args.max_traces) > 0:
                trace_indices = trace_indices[: int(args.max_traces)]
            for trace_idx in trace_indices:
                if int(trace_idx) < 0 or int(trace_idx) >= obs_table[THETA_PLUS].shape[0]:
                    raise IndexError(
                        f"trace index {trace_idx} outside condition {condition!r} "
                        f"with {obs_table[THETA_PLUS].shape[0]} traces"
                    )
            for catalog_mode in args.catalog_modes:
                include_self = catalog_mode == "include_self"
                for trace_idx in trace_indices:
                    calibration_split = int(trace_idx) % int(args.n_splits)
                    for process_model in args.process_models:
                        for process_cov_scale in args.process_cov_scales:
                            prepared = prepare_quadratic_profile(
                                prior_table,
                                zero_table,
                                prior_pose_table,
                                true_trace_index=int(trace_idx),
                                include_self=include_self,
                                basis_dim=int(args.basis_dim),
                                ridge=float(args.ridge),
                                process_model=str(process_model),
                                process_cov_scale=float(process_cov_scale),
                                covariance_floor_arcmin2=float(args.covariance_floor_arcmin2),
                                synthetic_prior_samples=int(args.synthetic_prior_samples),
                                synthetic_prior_kappa_weight_power=float(args.synthetic_prior_kappa_weight_power),
                                max_iter=int(args.max_iter),
                                n_catalog_starts=int(args.n_catalog_starts),
                                n_brownian_starts=int(args.n_brownian_starts),
                                seed=int(args.seed),
                                shared_basis=shared_basis,
                                feature_config=feature_config,
                                basis_mode=str(args.basis_mode),
                                sign_basis_weight=float(args.sign_basis_weight),
                                residual_variance_mode=str(args.residual_variance_mode),
                            )
                            for true_label in THETA_LABELS:
                                observed = obs_table[true_label][trace_idx]
                                for observation_weight in args.observation_weights:
                                    result = score_prepared_quadratic_trial(
                                        observed,
                                        true_label,
                                        trace_idx,
                                        zero_table,
                                        obs_pose_table,
                                        prepared,
                                        observation_weight=float(observation_weight),
                                        max_iter=int(args.max_iter),
                                    )
                                    trial_rows.append(
                                        {
                                            "condition": condition,
                                            "prior_condition": str(prior_condition),
                                            "condition_matches_prior": bool(condition_matches_prior),
                                            "fd_step_arcmin": fd_step,
                                            "inference_mode": inference_mode,
                                            "trace_index": int(trace_idx),
                                            "calibration_split": calibration_split,
                                            "true_label": true_label,
                                            "n_timebins": int(t),
                                            "n_units": int(observed.shape[1]),
                                            "source_cache": str(cache.path),
                                            "prior_cache": str(prior_cache.path),
                                            "zero_eye_reference_condition": str(args.reference_condition),
                                            "zero_eye_reference_available": bool(zero_ref_available),
                                            "catalog_mode": catalog_mode,
                                            "posterior_trace_diagnostics_interpretable": bool(condition_matches_prior),
                                            "leave_one_out_interpretation": (
                                                "true_trace_removed"
                                                if condition_matches_prior and not include_self
                                                else "same_index_removed_from_cross_prior_catalog"
                                                if not include_self
                                                else "full_catalog"
                                            ),
                                            **result,
                                        }
                                    )
    return trial_rows


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_rows = _score_trial_rows(args)
    selection_keys = selection_scope_keys(str(args.selection_scope))

    summary_by_scale = summarize_rows(
        trial_rows,
        [
            "catalog_mode",
            "condition",
            "prior_condition",
            "condition_matches_prior",
            "fd_step_arcmin",
            "inference_mode",
            "process_model",
            "process_cov_scale",
            "basis_mode",
            "residual_variance_mode",
            "feature_mode",
            "likelihood_scale",
            "joint_likelihood_normalization",
            "joint_score_family",
            "zero_eye_reference_condition",
        ],
    )
    selection_rows, heldout_summary, heldout_summary_by_pair = build_scale_policy_summaries(
        trial_rows,
        selection_keys=[*selection_keys, "process_model", "process_cov_scale"],
        baseline_likelihood_scale=float(args.baseline_observation_weight),
    )

    write_csv(out_dir / "continuous_quadratic_joint_trials.csv", trial_rows)
    write_csv(out_dir / "continuous_quadratic_summary_by_scale.csv", summary_by_scale)
    write_csv(out_dir / "continuous_quadratic_calibration_selection.csv", selection_rows)
    write_csv(out_dir / "continuous_quadratic_heldout_summary.csv", heldout_summary)
    write_csv(out_dir / "continuous_quadratic_heldout_summary_by_pair.csv", heldout_summary_by_pair)
    write_json(
        out_dir / "continuous_quadratic_manifest.json",
        {
            "source_dir": Path(args.source_dir),
            "out_dir": out_dir,
            "conditions": args.conditions,
            "prior_conditions": args.prior_conditions,
            "effective_prior_policy": "explicit" if args.prior_conditions else "same_condition",
            "fd_steps_arcmin": args.fd_steps_arcmin,
            "catalog_modes": args.catalog_modes,
            "basis_dim": int(args.basis_dim),
            "shared_basis": bool(args.shared_basis),
            "basis_mode": str(args.basis_mode),
            "sign_basis_weight": float(args.sign_basis_weight),
            "residual_variance_mode": str(args.residual_variance_mode),
            "feature_mode": str(args.feature_mode),
            "history_lags": args.history_lags,
            "include_velocity": bool(args.include_velocity),
            "velocity_lags": args.velocity_lags,
            "filter_alphas": args.filter_alphas,
            "ridge": float(args.ridge),
            "process_models": args.process_models,
            "process_cov_scales": args.process_cov_scales,
            "synthetic_prior_samples": int(args.synthetic_prior_samples),
            "synthetic_prior_kappa_weight_power": float(args.synthetic_prior_kappa_weight_power),
            "observation_weights": args.observation_weights,
            "baseline_observation_weight": float(args.baseline_observation_weight),
            "covariance_floor_arcmin2": float(args.covariance_floor_arcmin2),
            "n_catalog_starts": int(args.n_catalog_starts),
            "n_brownian_starts": int(args.n_brownian_starts),
            "max_iter": int(args.max_iter),
            "selection_scope": str(args.selection_scope),
            "selection_keys": [*selection_keys, "process_model", "process_cov_scale"],
            "n_splits": int(args.n_splits),
            "split_rule": "trace_index modulo n_splits",
            "trace_indices": args.trace_indices,
            "max_traces": int(args.max_traces),
            "reference_condition": str(args.reference_condition),
            "bin_seconds": float(args.bin_seconds),
            "max_timebins": int(args.max_timebins),
            "n_trial_rows": len(trial_rows),
            "n_summary_by_scale_rows": len(summary_by_scale),
            "n_calibration_selection_rows": len(selection_rows),
            "n_heldout_summary_rows": len(heldout_summary),
            "n_heldout_summary_by_pair_rows": len(heldout_summary_by_pair),
            "brownian_trace_use": (
                "Sampled process-prior paths, when requested, are optimization starts from the fitted prior; "
                "they are not new rendered ConvGRU response cache entries."
            ),
            "interpretation_guardrail": (
                "Static-center accuracy is a baseline, not an information ceiling. The known-trace score is a "
                "diagnostic of the fitted quadratic map at the true trajectory, not the full pose-aware Fisher upper bound."
            ),
            "implementation_provenance": "Implemented independently from specification; no GPL-covered source code copied.",
        },
    )
    return trial_rows, heldout_summary, heldout_summary_by_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--conditions", type=str, default="")
    parser.add_argument("--prior-conditions", type=str, default="")
    parser.add_argument("--fd-steps-arcmin", type=str, default="")
    parser.add_argument("--reference-condition", type=str, default="static_center")
    parser.add_argument("--allow-missing-reference", action="store_true")
    parser.add_argument("--catalog-modes", type=str, default="leave_one_out")
    parser.add_argument("--basis-dim", type=int, default=20)
    parser.add_argument("--shared-basis", action="store_true")
    parser.add_argument("--basis-mode", choices=BASIS_MODES, default="pca")
    parser.add_argument("--sign-basis-weight", type=float, default=1.0)
    parser.add_argument("--residual-variance-mode", choices=RESIDUAL_VARIANCE_MODES, default="label")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="current_quadratic")
    parser.add_argument("--history-lags", type=str, default="0,1,2,4")
    parser.add_argument("--include-velocity", action="store_true")
    parser.add_argument("--velocity-lags", type=str, default="0,1")
    parser.add_argument("--filter-alphas", type=str, default="0.25,0.5")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--process-models", type=str, default="matched_brownian")
    parser.add_argument("--process-cov-scales", type=str, default="0.25,0.5,1,2,4")
    parser.add_argument("--synthetic-prior-samples", type=int, default=512)
    parser.add_argument("--synthetic-prior-kappa-weight-power", type=float, default=0.5)
    parser.add_argument("--observation-weights", type=str, default="0.25,0.5,1,2,4,8,16,32")
    parser.add_argument("--baseline-observation-weight", type=float, default=1.0)
    parser.add_argument("--covariance-floor-arcmin2", type=float, default=DEFAULT_COV_FLOOR_ARCMIN2)
    parser.add_argument("--n-catalog-starts", type=int, default=4)
    parser.add_argument("--n-brownian-starts", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--selection-scope", choices=("global_by_fd_and_mode", "condition_by_fd_and_mode", "condition_prior_by_fd_and_mode"), default="condition_by_fd_and_mode")
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--trace-indices", type=str, default="")
    parser.add_argument("--max-traces", type=int, default=0)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument("--max-timebins", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.conditions = parse_csv_str(args.conditions)
    args.prior_conditions = parse_csv_str(args.prior_conditions)
    args.fd_steps_arcmin = parse_csv_float(args.fd_steps_arcmin)
    args.catalog_modes = parse_catalog_modes(args.catalog_modes)
    args.process_cov_scales = parse_csv_float(args.process_cov_scales)
    args.observation_weights = parse_csv_float(args.observation_weights)
    args.trace_indices = parse_csv_int(args.trace_indices)
    args.history_lags = parse_csv_int(args.history_lags)
    args.velocity_lags = parse_csv_int(args.velocity_lags)
    args.filter_alphas = parse_csv_float(args.filter_alphas)
    args.process_models = parse_csv_str(args.process_models)
    if int(args.n_splits) < 2:
        raise ValueError("--n-splits must be at least 2")
    if int(args.basis_dim) <= 0:
        raise ValueError("--basis-dim must be positive")
    bad_process_models = [model for model in args.process_models if model not in PROCESS_MODELS]
    if bad_process_models:
        raise ValueError(f"Unsupported process models {bad_process_models}; expected {PROCESS_MODELS}")
    if int(args.synthetic_prior_samples) <= 0:
        raise ValueError("--synthetic-prior-samples must be positive")
    if any(scale <= 0.0 for scale in args.process_cov_scales):
        raise ValueError("--process-cov-scales must all be positive")
    if any(weight <= 0.0 for weight in args.observation_weights):
        raise ValueError("--observation-weights must all be positive")
    trial_rows, heldout_summary, heldout_summary_by_pair = run(args)
    print(
        f"Wrote {len(trial_rows)} continuous quadratic trials, "
        f"{len(heldout_summary)} heldout summaries, "
        f"and {len(heldout_summary_by_pair)} pair summaries",
        flush=True,
    )


if __name__ == "__main__":
    main()
