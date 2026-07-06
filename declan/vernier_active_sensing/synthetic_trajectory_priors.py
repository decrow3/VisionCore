"""Reusable synthetic FEM trajectory priors.

The functions here intentionally know nothing about Vernier stimuli or decoder
scores. They take source eye traces in degrees and return newly sampled traces
plus metadata, so the same artificial trajectory priors can be used in Vernier,
backimage, and future observer controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .generate_brownian_eye_traces import CENTER_MODES, COVARIANCE_MODES, generate_brownian_traces
from .trajectories import FRAME_RATE_HZ

SYNTHETIC_TRAJECTORY_PRIOR_MODELS = (
    "brownian",
    "ou",
    "step_ar1",
    "anti_step_ar1",
    "scale_mixture_anti_step_ar1",
    "confined_step_ar1",
    "scale_mixture_confined_step_ar1",
    "scale_mixture_empirical_confined_step_ar1",
    "ar2",
)
SCALE_MIXTURE_FEATURES = ("step_rms_arcmin", "rms_radius_arcmin", "path_length_arcmin")
SCALE_MIXTURE_DISTRIBUTIONS = ("lognormal_mean_std", "lognormal_median_iqr", "empirical")


@dataclass(frozen=True)
class SyntheticTrajectoryPriorConfig:
    """Configuration for sampling artificial FEM-like trajectories."""

    process_model: str = "scale_mixture_empirical_confined_step_ar1"
    covariance_mode: str = "full_empirical"
    center_mode: str = "zero_mean"
    global_scale: float = 1.0
    frame_rate_hz: float = FRAME_RATE_HZ
    brownian_step_cov_deg2_per_frame: np.ndarray | None = None
    brownian_center_deg: np.ndarray | None = None
    ou_max_abs_eigenvalue: float = 0.999
    anti_step_beta: float | None = None
    position_spring_kappa: float = 0.0
    scale_mixture_feature: str = "step_rms_arcmin"
    scale_mixture_distribution: str = "empirical"
    confined_param_beta_min: float = -0.95
    confined_param_beta_max: float = 0.50
    confined_param_kappa_min: float = 0.0
    confined_param_kappa_max: float = 1.50
    confined_param_max_abs_eigenvalue: float = 0.995
    confined_param_kappa_weight_power: float = 0.0


@dataclass(frozen=True)
class SyntheticTrajectoryPriorResult:
    """Synthetic trajectories and provenance for a prior draw."""

    traces_deg: np.ndarray
    metadata: dict[str, Any]
    base_traces_deg: np.ndarray | None = None
    chosen_confined_params: np.ndarray | None = None


def recommended_empirical_confined_config(
    *,
    kappa_weight_power: float = 0.5,
    covariance_mode: str = "full_empirical",
    center_mode: str = "zero_mean",
) -> SyntheticTrajectoryPriorConfig:
    """Recommended broad artificial-FEM comparison prior from the 2026-07-01 refinement."""
    return SyntheticTrajectoryPriorConfig(
        process_model="scale_mixture_empirical_confined_step_ar1",
        covariance_mode=str(covariance_mode),
        center_mode=str(center_mode),
        scale_mixture_feature="step_rms_arcmin",
        scale_mixture_distribution="empirical",
        confined_param_kappa_weight_power=float(kappa_weight_power),
    )


def recenter_traces(traces_deg: np.ndarray, center_mode: str) -> np.ndarray:
    traces = np.asarray(traces_deg, dtype=np.float64).copy()
    mode = str(center_mode)
    if mode == "zero_mean":
        return traces - np.mean(traces, axis=1, keepdims=True)
    if mode == "start_zero":
        return traces - traces[:, :1, :]
    if mode == "source_grand_mean":
        source_mean = np.mean(traces.reshape(-1, 2), axis=0)
        return traces - np.mean(traces, axis=1, keepdims=True) + source_mean[None, None, :]
    raise ValueError(f"Unsupported center_mode={center_mode!r}; expected {CENTER_MODES}")


def trace_features(traces_deg: np.ndarray, frame_rate_hz: float = FRAME_RATE_HZ) -> pd.DataFrame:
    traces_arcmin = np.asarray(traces_deg, dtype=np.float64) * 60.0
    steps_arcmin = np.diff(traces_arcmin, axis=1)
    radius = np.linalg.norm(traces_arcmin, axis=2)
    step_norm = np.linalg.norm(steps_arcmin, axis=2)
    prev = steps_arcmin[:, :-1, :]
    nxt = steps_arcmin[:, 1:, :]
    denom = np.linalg.norm(prev, axis=2) * np.linalg.norm(nxt, axis=2)
    valid = denom > 1e-12
    lag1 = np.full(traces_arcmin.shape[0], np.nan, dtype=np.float64)
    if prev.shape[1] > 0:
        cos = np.full(valid.shape, np.nan, dtype=np.float64)
        cos[valid] = np.sum(prev[valid] * nxt[valid], axis=1) / denom[valid]
        lag1 = np.nanmean(cos, axis=1)

    return pd.DataFrame(
        {
            "rms_radius_arcmin": np.sqrt(np.mean(radius * radius, axis=1)),
            "std_x_arcmin": np.std(traces_arcmin[:, :, 0], axis=1),
            "std_y_arcmin": np.std(traces_arcmin[:, :, 1], axis=1),
            "path_length_arcmin": np.sum(step_norm, axis=1),
            "step_rms_arcmin": np.sqrt(np.mean(step_norm * step_norm, axis=1)),
            "step_rms_arcmin_per_s": np.sqrt(np.mean((step_norm * float(frame_rate_hz)) ** 2, axis=1)),
            "max_radius_arcmin": np.max(radius, axis=1),
            "endpoint_radius_arcmin": radius[:, -1],
            "step_lag1_cosine": lag1,
        }
    )


def fit_step_covariance_from_snippets(
    traces_deg: np.ndarray,
    *,
    covariance_mode: str,
    global_scale: float,
) -> np.ndarray:
    steps = np.diff(np.asarray(traces_deg, dtype=np.float64), axis=1).reshape(-1, 2)
    raw_cov = np.cov(steps.T)
    return covariance_by_mode(raw_cov, covariance_mode=str(covariance_mode), global_scale=float(global_scale))


def regularized_cov(samples: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    cov = np.cov(np.asarray(samples, dtype=np.float64).T)
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    cov = 0.5 * (cov + cov.T)
    eig = np.linalg.eigvalsh(cov)
    min_eig = float(np.min(eig))
    if min_eig < float(floor):
        cov = cov + (float(floor) - min_eig) * np.eye(cov.shape[0])
    return cov


def covariance_by_mode(raw_cov: np.ndarray, covariance_mode: str, global_scale: float = 1.0) -> np.ndarray:
    raw = np.asarray(raw_cov, dtype=np.float64)
    if str(covariance_mode) == "diagonal_empirical":
        cov = np.diag(np.diag(raw))
    elif str(covariance_mode) == "full_empirical":
        cov = raw
    elif str(covariance_mode) == "isotropic_scalar":
        cov = np.eye(raw.shape[0], dtype=np.float64) * float(np.mean(np.diag(raw)))
    else:
        raise ValueError(f"Unsupported covariance_mode={covariance_mode!r}; expected {COVARIANCE_MODES}")
    return np.asarray(cov, dtype=np.float64) * float(global_scale)


def stabilize_transition(matrix: np.ndarray, max_abs_eigenvalue: float) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    eig = np.linalg.eigvals(mat)
    radius = float(np.max(np.abs(eig))) if eig.size else 0.0
    if radius <= float(max_abs_eigenvalue):
        return mat
    return mat * (float(max_abs_eigenvalue) / radius)


def fit_ou_prior_from_snippets(
    traces_deg: np.ndarray,
    *,
    covariance_mode: str,
    global_scale: float,
    max_abs_eigenvalue: float,
) -> dict[str, np.ndarray]:
    traces = np.asarray(traces_deg, dtype=np.float64)
    x_prev = traces[:, :-1, :].reshape(-1, 2)
    x_next = traces[:, 1:, :].reshape(-1, 2)
    coef, *_ = np.linalg.lstsq(x_prev, x_next, rcond=None)
    transition = stabilize_transition(coef.T, max_abs_eigenvalue=float(max_abs_eigenvalue))
    residual = x_next - x_prev @ transition.T
    raw_innovation_cov = regularized_cov(residual)
    innovation_cov = covariance_by_mode(
        raw_innovation_cov,
        covariance_mode=str(covariance_mode),
        global_scale=float(global_scale),
    )
    return {
        "transition_matrix": transition,
        "raw_innovation_cov_deg2": raw_innovation_cov,
        "innovation_cov_deg2": innovation_cov,
        "init_mean_deg": np.mean(traces[:, 0, :], axis=0),
        "init_cov_deg2": regularized_cov(traces[:, 0, :]) * float(global_scale),
        "transition_eigenvalues": np.linalg.eigvals(transition),
    }


def generate_ou_traces(
    *,
    transition_matrix: np.ndarray,
    innovation_cov_deg2: np.ndarray,
    init_mean_deg: np.ndarray,
    init_cov_deg2: np.ndarray,
    n_traces: int,
    n_frames: int,
    seed: int,
    center_mode: str,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    transition = np.asarray(transition_matrix, dtype=np.float64)
    traces = np.zeros((int(n_traces), int(n_frames), 2), dtype=np.float64)
    traces[:, 0, :] = rng.multivariate_normal(
        np.asarray(init_mean_deg, dtype=np.float64),
        np.asarray(init_cov_deg2, dtype=np.float64),
        size=int(n_traces),
    )
    for idx in range(1, int(n_frames)):
        eps = rng.multivariate_normal(
            np.zeros(2, dtype=np.float64),
            np.asarray(innovation_cov_deg2, dtype=np.float64),
            size=int(n_traces),
        )
        traces[:, idx, :] = traces[:, idx - 1, :] @ transition.T + eps
    return recenter_traces(traces, str(center_mode)).astype(np.float32)


def fit_step_ar1_prior_from_snippets(
    traces_deg: np.ndarray,
    *,
    covariance_mode: str,
    global_scale: float,
    max_abs_eigenvalue: float,
) -> dict[str, np.ndarray]:
    traces = np.asarray(traces_deg, dtype=np.float64)
    steps = np.diff(traces, axis=1)
    prev = steps[:, :-1, :].reshape(-1, 2)
    nxt = steps[:, 1:, :].reshape(-1, 2)
    coef, *_ = np.linalg.lstsq(prev, nxt, rcond=None)
    transition = stabilize_transition(coef.T, max_abs_eigenvalue=float(max_abs_eigenvalue))
    residual = nxt - prev @ transition.T
    raw_innovation_cov = regularized_cov(residual)
    innovation_cov = covariance_by_mode(
        raw_innovation_cov,
        covariance_mode=str(covariance_mode),
        global_scale=float(global_scale),
    )
    return {
        "step_transition_matrix": transition,
        "raw_innovation_cov_deg2": raw_innovation_cov,
        "innovation_cov_deg2": innovation_cov,
        "init_mean_deg": np.mean(traces[:, 0, :], axis=0),
        "init_cov_deg2": regularized_cov(traces[:, 0, :]) * float(global_scale),
        "init_step_mean_deg": np.mean(steps[:, 0, :], axis=0),
        "init_step_cov_deg2": regularized_cov(steps[:, 0, :]) * float(global_scale),
        "step_transition_eigenvalues": np.linalg.eigvals(transition),
    }


def median_step_lag1_cosine(traces_deg: np.ndarray) -> float:
    return float(trace_features(np.asarray(traces_deg, dtype=np.float64), FRAME_RATE_HZ)["step_lag1_cosine"].median())


def fit_anti_step_ar1_prior_from_snippets(
    traces_deg: np.ndarray,
    *,
    covariance_mode: str,
    global_scale: float,
    beta: float | None,
) -> dict[str, np.ndarray | float]:
    traces = np.asarray(traces_deg, dtype=np.float64)
    steps_by_trace = np.diff(traces, axis=1)
    steps = steps_by_trace.reshape(-1, 2)
    step_cov = covariance_by_mode(
        regularized_cov(steps),
        covariance_mode=str(covariance_mode),
        global_scale=float(global_scale),
    )
    beta_value = median_step_lag1_cosine(traces) if beta is None or not np.isfinite(float(beta)) else float(beta)
    beta_value = float(np.clip(beta_value, -0.98, 0.98))
    transition = np.eye(2, dtype=np.float64) * beta_value
    innovation_cov = step_cov * max(1.0 - beta_value * beta_value, 1e-6)
    return {
        "step_transition_matrix": transition,
        "target_step_cov_deg2": step_cov,
        "innovation_cov_deg2": innovation_cov,
        "anti_step_beta": beta_value,
        "init_mean_deg": np.mean(traces[:, 0, :], axis=0),
        "init_cov_deg2": regularized_cov(traces[:, 0, :]) * float(global_scale),
        "init_step_mean_deg": np.mean(steps, axis=0),
        "init_step_cov_deg2": step_cov,
        "step_transition_eigenvalues": np.linalg.eigvals(transition),
    }


def confined_step_companion(beta: float, kappa: float) -> np.ndarray:
    return np.array(
        [
            [1.0 - float(kappa), float(beta)],
            [-float(kappa), float(beta)],
        ],
        dtype=np.float64,
    )


def fit_confined_step_ar1_prior_from_snippets(
    traces_deg: np.ndarray,
    *,
    covariance_mode: str,
    global_scale: float,
    beta: float | None,
    kappa: float,
) -> dict[str, np.ndarray | float]:
    traces = np.asarray(traces_deg, dtype=np.float64)
    steps_by_trace = np.diff(traces, axis=1)
    steps = steps_by_trace.reshape(-1, 2)
    step_cov = covariance_by_mode(
        regularized_cov(steps),
        covariance_mode=str(covariance_mode),
        global_scale=float(global_scale),
    )
    beta_value = median_step_lag1_cosine(traces) if beta is None or not np.isfinite(float(beta)) else float(beta)
    beta_value = float(np.clip(beta_value, -0.98, 0.98))
    kappa_value = float(kappa)
    if kappa_value < 0.0:
        raise ValueError(f"position_spring_kappa must be nonnegative, got {kappa_value}")
    companion = confined_step_companion(beta_value, kappa_value)
    companion_eigenvalues = np.linalg.eigvals(companion)
    max_abs_eigenvalue = float(np.max(np.abs(companion_eigenvalues)))
    if max_abs_eigenvalue >= 1.0:
        raise ValueError(
            "confined_step_ar1 is unstable for "
            f"beta={beta_value:.6g}, kappa={kappa_value:.6g}; "
            f"max |eigenvalue|={max_abs_eigenvalue:.6g}"
        )
    innovation_cov = step_cov * max(1.0 - beta_value * beta_value, 0.05)
    return {
        "step_transition_beta": beta_value,
        "position_spring_kappa": kappa_value,
        "target_step_cov_deg2": step_cov,
        "innovation_cov_deg2": innovation_cov,
        "init_mean_deg": np.mean(traces[:, 0, :], axis=0),
        "init_cov_deg2": regularized_cov(traces[:, 0, :]) * float(global_scale),
        "init_step_mean_deg": np.mean(steps, axis=0),
        "init_step_cov_deg2": step_cov,
        "confined_step_companion_eigenvalues": companion_eigenvalues,
        "confined_step_companion_max_abs_eigenvalue": max_abs_eigenvalue,
    }


def fit_empirical_confined_step_parameters(
    traces_deg: np.ndarray,
    *,
    beta_min: float,
    beta_max: float,
    kappa_min: float,
    kappa_max: float,
    max_abs_eigenvalue: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    traces = np.asarray(traces_deg, dtype=np.float64)
    rows = []
    for trace_idx in range(traces.shape[0]):
        trace = traces[trace_idx]
        steps = np.diff(trace, axis=0)
        if steps.shape[0] < 2:
            continue
        design = np.column_stack([steps[:-1].reshape(-1), -trace[1:-1].reshape(-1)])
        target = steps[1:].reshape(-1)
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        beta_value = float(coef[0])
        kappa_value = float(coef[1])
        if not (float(beta_min) <= beta_value <= float(beta_max)):
            continue
        if not (float(kappa_min) <= kappa_value <= float(kappa_max)):
            continue
        eig = np.linalg.eigvals(confined_step_companion(beta_value, kappa_value))
        eig_max = float(np.max(np.abs(eig)))
        if eig_max >= float(max_abs_eigenvalue):
            continue
        rows.append((beta_value, kappa_value, eig_max))
    if not rows:
        raise ValueError("No empirical confined-step parameters passed the stability/plausibility filters")
    params = np.asarray(rows, dtype=np.float64)
    meta: dict[str, float | int] = {
        "empirical_confined_param_count": int(params.shape[0]),
        "empirical_confined_param_fraction_kept": float(params.shape[0] / max(traces.shape[0], 1)),
    }
    for col_idx, name in enumerate(["beta", "kappa", "eigmax"]):
        vals = params[:, col_idx]
        meta[f"empirical_confined_{name}_q05"] = float(np.quantile(vals, 0.05))
        meta[f"empirical_confined_{name}_q25"] = float(np.quantile(vals, 0.25))
        meta[f"empirical_confined_{name}_median"] = float(np.median(vals))
        meta[f"empirical_confined_{name}_q75"] = float(np.quantile(vals, 0.75))
        meta[f"empirical_confined_{name}_q95"] = float(np.quantile(vals, 0.95))
    return params[:, :2], meta


def sample_scale_mixture_targets(
    values: np.ndarray,
    *,
    n: int,
    rng: np.random.Generator,
    distribution: str,
) -> tuple[np.ndarray, dict[str, float]]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size == 0:
        raise ValueError("Scale-mixture target feature has no positive finite values")
    mode = str(distribution)
    if mode == "empirical":
        out = rng.choice(vals, size=int(n), replace=True)
        return out, {
            "scale_mixture_empirical_target_mean": float(np.mean(vals)),
            "scale_mixture_empirical_target_median": float(np.median(vals)),
        }
    log_vals = np.log(vals)
    if mode == "lognormal_mean_std":
        mu = float(np.mean(log_vals))
        sigma = float(np.std(log_vals))
    elif mode == "lognormal_median_iqr":
        mu = float(np.median(log_vals))
        sigma = float((np.quantile(log_vals, 0.75) - np.quantile(log_vals, 0.25)) / 1.349)
    else:
        raise ValueError(
            f"Unsupported scale_mixture_distribution={distribution!r}; expected {SCALE_MIXTURE_DISTRIBUTIONS}"
        )
    sigma = max(sigma, 1e-6)
    return np.exp(rng.normal(mu, sigma, size=int(n))), {
        "scale_mixture_lognormal_mu": mu,
        "scale_mixture_lognormal_sigma": sigma,
        "scale_mixture_target_median": float(np.median(vals)),
    }


def apply_trace_scale_mixture(
    traces_deg: np.ndarray,
    *,
    target_values: np.ndarray,
    feature_name: str,
    frame_rate_hz: float,
    center_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    features = trace_features(np.asarray(traces_deg, dtype=np.float64), float(frame_rate_hz))
    base = np.asarray(features[str(feature_name)], dtype=np.float64)
    scale = np.asarray(target_values, dtype=np.float64) / np.clip(base, 1e-9, None)
    scaled = np.asarray(traces_deg, dtype=np.float64) * scale[:, None, None]
    return recenter_traces(scaled, str(center_mode)).astype(np.float32), scale


def generate_step_ar1_traces(
    *,
    step_transition_matrix: np.ndarray,
    innovation_cov_deg2: np.ndarray,
    init_mean_deg: np.ndarray,
    init_cov_deg2: np.ndarray,
    init_step_mean_deg: np.ndarray,
    init_step_cov_deg2: np.ndarray,
    n_traces: int,
    n_frames: int,
    seed: int,
    center_mode: str,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    transition = np.asarray(step_transition_matrix, dtype=np.float64)
    traces = np.zeros((int(n_traces), int(n_frames), 2), dtype=np.float64)
    traces[:, 0, :] = rng.multivariate_normal(
        np.asarray(init_mean_deg, dtype=np.float64),
        np.asarray(init_cov_deg2, dtype=np.float64),
        size=int(n_traces),
    )
    if int(n_frames) <= 1:
        return recenter_traces(traces, str(center_mode)).astype(np.float32)
    step = rng.multivariate_normal(
        np.asarray(init_step_mean_deg, dtype=np.float64),
        np.asarray(init_step_cov_deg2, dtype=np.float64),
        size=int(n_traces),
    )
    traces[:, 1, :] = traces[:, 0, :] + step
    for idx in range(2, int(n_frames)):
        eps = rng.multivariate_normal(
            np.zeros(2, dtype=np.float64),
            np.asarray(innovation_cov_deg2, dtype=np.float64),
            size=int(n_traces),
        )
        step = step @ transition.T + eps
        traces[:, idx, :] = traces[:, idx - 1, :] + step
    return recenter_traces(traces, str(center_mode)).astype(np.float32)


def generate_confined_step_ar1_traces(
    *,
    step_transition_beta: float,
    position_spring_kappa: float,
    innovation_cov_deg2: np.ndarray,
    init_mean_deg: np.ndarray,
    init_cov_deg2: np.ndarray,
    init_step_mean_deg: np.ndarray,
    init_step_cov_deg2: np.ndarray,
    n_traces: int,
    n_frames: int,
    seed: int,
    center_mode: str,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    beta = float(step_transition_beta)
    kappa = float(position_spring_kappa)
    traces = np.zeros((int(n_traces), int(n_frames), 2), dtype=np.float64)
    traces[:, 0, :] = rng.multivariate_normal(
        np.asarray(init_mean_deg, dtype=np.float64),
        np.asarray(init_cov_deg2, dtype=np.float64),
        size=int(n_traces),
    )
    if int(n_frames) <= 1:
        return recenter_traces(traces, str(center_mode)).astype(np.float32)
    step = rng.multivariate_normal(
        np.asarray(init_step_mean_deg, dtype=np.float64),
        np.asarray(init_step_cov_deg2, dtype=np.float64),
        size=int(n_traces),
    )
    traces[:, 1, :] = traces[:, 0, :] + step
    for idx in range(2, int(n_frames)):
        eps = rng.multivariate_normal(
            np.zeros(2, dtype=np.float64),
            np.asarray(innovation_cov_deg2, dtype=np.float64),
            size=int(n_traces),
        )
        step = beta * step - kappa * traces[:, idx - 1, :] + eps
        traces[:, idx, :] = traces[:, idx - 1, :] + step
    return recenter_traces(traces, str(center_mode)).astype(np.float32)


def generate_empirical_confined_step_ar1_traces(
    *,
    beta_kappa_samples: np.ndarray,
    sample_weights: np.ndarray | None,
    target_step_cov_deg2: np.ndarray,
    init_mean_deg: np.ndarray,
    init_cov_deg2: np.ndarray,
    init_step_mean_deg: np.ndarray,
    init_step_cov_deg2: np.ndarray,
    n_traces: int,
    n_frames: int,
    seed: int,
    center_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    params = np.asarray(beta_kappa_samples, dtype=np.float64)
    weights = None
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=np.float64)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("Empirical confined-step sample weights must have positive finite mass")
        weights = weights / total
    chosen = params[rng.choice(params.shape[0], size=int(n_traces), replace=True, p=weights)]
    beta = chosen[:, 0]
    kappa = chosen[:, 1]
    traces = np.zeros((int(n_traces), int(n_frames), 2), dtype=np.float64)
    traces[:, 0, :] = rng.multivariate_normal(
        np.asarray(init_mean_deg, dtype=np.float64),
        np.asarray(init_cov_deg2, dtype=np.float64),
        size=int(n_traces),
    )
    if int(n_frames) <= 1:
        return recenter_traces(traces, str(center_mode)).astype(np.float32), chosen
    step = rng.multivariate_normal(
        np.asarray(init_step_mean_deg, dtype=np.float64),
        np.asarray(init_step_cov_deg2, dtype=np.float64),
        size=int(n_traces),
    )
    traces[:, 1, :] = traces[:, 0, :] + step
    chol = np.linalg.cholesky(np.asarray(target_step_cov_deg2, dtype=np.float64))
    eps_scale = np.sqrt(np.maximum(1.0 - beta * beta, 0.05))
    for idx in range(2, int(n_frames)):
        eps = (rng.normal(size=(int(n_traces), 2)) @ chol.T) * eps_scale[:, None]
        step = beta[:, None] * step - kappa[:, None] * traces[:, idx - 1, :] + eps
        traces[:, idx, :] = traces[:, idx - 1, :] + step
    return recenter_traces(traces, str(center_mode)).astype(np.float32), chosen


def ar2_companion(coef_2x4: np.ndarray) -> np.ndarray:
    coef = np.asarray(coef_2x4, dtype=np.float64)
    companion = np.zeros((4, 4), dtype=np.float64)
    companion[:2, :] = coef
    companion[2:, :2] = np.eye(2, dtype=np.float64)
    return companion


def stabilize_ar2(coef_2x4: np.ndarray, max_abs_eigenvalue: float) -> np.ndarray:
    coef = np.asarray(coef_2x4, dtype=np.float64)
    radius = float(np.max(np.abs(np.linalg.eigvals(ar2_companion(coef)))))
    if radius <= float(max_abs_eigenvalue):
        return coef
    return coef * (float(max_abs_eigenvalue) / radius)


def fit_ar2_prior_from_snippets(
    traces_deg: np.ndarray,
    *,
    covariance_mode: str,
    global_scale: float,
    max_abs_eigenvalue: float,
) -> dict[str, np.ndarray]:
    traces = np.asarray(traces_deg, dtype=np.float64)
    design = np.concatenate([traces[:, 1:-1, :], traces[:, :-2, :]], axis=2).reshape(-1, 4)
    target = traces[:, 2:, :].reshape(-1, 2)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    coef_2x4 = stabilize_ar2(coef.T, max_abs_eigenvalue=float(max_abs_eigenvalue))
    residual = target - design @ coef_2x4.T
    raw_innovation_cov = regularized_cov(residual)
    innovation_cov = covariance_by_mode(
        raw_innovation_cov,
        covariance_mode=str(covariance_mode),
        global_scale=float(global_scale),
    )
    init_state = np.concatenate([traces[:, 1, :], traces[:, 0, :]], axis=1)
    return {
        "ar2_coef_2x4": coef_2x4,
        "raw_innovation_cov_deg2": raw_innovation_cov,
        "innovation_cov_deg2": innovation_cov,
        "init_state_mean_deg": np.mean(init_state, axis=0),
        "init_state_cov_deg2": regularized_cov(init_state) * float(global_scale),
        "ar2_companion_eigenvalues": np.linalg.eigvals(ar2_companion(coef_2x4)),
    }


def generate_ar2_traces(
    *,
    ar2_coef_2x4: np.ndarray,
    innovation_cov_deg2: np.ndarray,
    init_state_mean_deg: np.ndarray,
    init_state_cov_deg2: np.ndarray,
    n_traces: int,
    n_frames: int,
    seed: int,
    center_mode: str,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    coef = np.asarray(ar2_coef_2x4, dtype=np.float64)
    traces = np.zeros((int(n_traces), int(n_frames), 2), dtype=np.float64)
    init_state = rng.multivariate_normal(
        np.asarray(init_state_mean_deg, dtype=np.float64),
        np.asarray(init_state_cov_deg2, dtype=np.float64),
        size=int(n_traces),
    )
    traces[:, 0, :] = init_state[:, 2:]
    if int(n_frames) > 1:
        traces[:, 1, :] = init_state[:, :2]
    for idx in range(2, int(n_frames)):
        design = np.concatenate([traces[:, idx - 1, :], traces[:, idx - 2, :]], axis=1)
        eps = rng.multivariate_normal(
            np.zeros(2, dtype=np.float64),
            np.asarray(innovation_cov_deg2, dtype=np.float64),
            size=int(n_traces),
        )
        traces[:, idx, :] = design @ coef.T + eps
    return recenter_traces(traces, str(center_mode)).astype(np.float32)


def generate_synthetic_trajectory_prior(
    source_traces_deg: np.ndarray,
    *,
    n_traces: int,
    n_frames: int | None = None,
    seed: int,
    config: SyntheticTrajectoryPriorConfig | None = None,
) -> SyntheticTrajectoryPriorResult:
    """Sample artificial FEM trajectories from a fitted source-trace prior.

    Parameters
    ----------
    source_traces_deg:
        Source traces with shape ``(trace, frame, xy)`` in degrees.
    n_traces:
        Number of new traces to sample.
    n_frames:
        Number of frames per output trace. Defaults to the source trace length.
    seed:
        Random seed for all generated traces and scale-mixture draws.
    config:
        Prior family and fitting options.
    """
    cfg = config or SyntheticTrajectoryPriorConfig()
    process_model = str(cfg.process_model)
    if process_model not in SYNTHETIC_TRAJECTORY_PRIOR_MODELS:
        raise ValueError(f"Unsupported process_model={process_model!r}; expected {SYNTHETIC_TRAJECTORY_PRIOR_MODELS}")
    if str(cfg.scale_mixture_feature) not in SCALE_MIXTURE_FEATURES:
        raise ValueError(
            f"Unsupported scale_mixture_feature={cfg.scale_mixture_feature!r}; expected {SCALE_MIXTURE_FEATURES}"
        )
    if str(cfg.scale_mixture_distribution) not in SCALE_MIXTURE_DISTRIBUTIONS:
        raise ValueError(
            "Unsupported scale_mixture_distribution="
            f"{cfg.scale_mixture_distribution!r}; expected {SCALE_MIXTURE_DISTRIBUTIONS}"
        )
    source = np.asarray(source_traces_deg, dtype=np.float64)
    if source.ndim != 3 or source.shape[2] != 2:
        raise ValueError("source_traces_deg must have shape (trace, frame, 2)")
    frames = int(source.shape[1] if n_frames is None else n_frames)
    if frames < 2:
        raise ValueError("n_frames must be at least 2")
    if frames > source.shape[1]:
        raise ValueError(f"n_frames={frames} exceeds source trace length {source.shape[1]}")
    source_raw = source[:, :frames, :]
    source_centered = recenter_traces(source_raw, str(cfg.center_mode))
    n_total = int(n_traces)
    if n_total <= 0:
        raise ValueError("n_traces must be positive")

    prior: np.ndarray
    base_prior: np.ndarray | None = None
    chosen_params: np.ndarray | None = None
    metadata: dict[str, Any]

    if process_model == "brownian":
        step_cov = (
            np.asarray(cfg.brownian_step_cov_deg2_per_frame, dtype=np.float64)
            if cfg.brownian_step_cov_deg2_per_frame is not None
            else fit_step_covariance_from_snippets(
                source_raw,
                covariance_mode=str(cfg.covariance_mode),
                global_scale=float(cfg.global_scale),
            )
        )
        prior = generate_brownian_traces(
            step_cov_deg2_per_frame=step_cov,
            n_traces=n_total,
            n_frames=frames,
            seed=int(seed),
            center_mode=str(cfg.center_mode),
            center_deg=(
                np.asarray(cfg.brownian_center_deg, dtype=np.float64)
                if cfg.brownian_center_deg is not None
                else np.mean(source_raw.reshape(-1, 2), axis=0)
            ),
        ).astype(np.float64)
        metadata = {"step_cov_deg2_per_frame": step_cov}
    elif process_model == "ou":
        metadata = fit_ou_prior_from_snippets(
            source_centered,
            covariance_mode=str(cfg.covariance_mode),
            global_scale=float(cfg.global_scale),
            max_abs_eigenvalue=float(cfg.ou_max_abs_eigenvalue),
        )
        prior = generate_ou_traces(
            transition_matrix=np.asarray(metadata["transition_matrix"], dtype=np.float64),
            innovation_cov_deg2=np.asarray(metadata["innovation_cov_deg2"], dtype=np.float64),
            init_mean_deg=np.asarray(metadata["init_mean_deg"], dtype=np.float64),
            init_cov_deg2=np.asarray(metadata["init_cov_deg2"], dtype=np.float64),
            n_traces=n_total,
            n_frames=frames,
            seed=int(seed),
            center_mode=str(cfg.center_mode),
        ).astype(np.float64)
    elif process_model == "step_ar1":
        metadata = fit_step_ar1_prior_from_snippets(
            source_centered,
            covariance_mode=str(cfg.covariance_mode),
            global_scale=float(cfg.global_scale),
            max_abs_eigenvalue=float(cfg.ou_max_abs_eigenvalue),
        )
        prior = generate_step_ar1_traces(
            step_transition_matrix=np.asarray(metadata["step_transition_matrix"], dtype=np.float64),
            innovation_cov_deg2=np.asarray(metadata["innovation_cov_deg2"], dtype=np.float64),
            init_mean_deg=np.asarray(metadata["init_mean_deg"], dtype=np.float64),
            init_cov_deg2=np.asarray(metadata["init_cov_deg2"], dtype=np.float64),
            init_step_mean_deg=np.asarray(metadata["init_step_mean_deg"], dtype=np.float64),
            init_step_cov_deg2=np.asarray(metadata["init_step_cov_deg2"], dtype=np.float64),
            n_traces=n_total,
            n_frames=frames,
            seed=int(seed),
            center_mode=str(cfg.center_mode),
        ).astype(np.float64)
    elif process_model in {"anti_step_ar1", "scale_mixture_anti_step_ar1"}:
        fit_scale = 1.0 if process_model.startswith("scale_mixture") else float(cfg.global_scale)
        metadata = fit_anti_step_ar1_prior_from_snippets(
            source_centered,
            covariance_mode=str(cfg.covariance_mode),
            global_scale=fit_scale,
            beta=cfg.anti_step_beta,
        )
        base_prior = generate_step_ar1_traces(
            step_transition_matrix=np.asarray(metadata["step_transition_matrix"], dtype=np.float64),
            innovation_cov_deg2=np.asarray(metadata["innovation_cov_deg2"], dtype=np.float64),
            init_mean_deg=np.asarray(metadata["init_mean_deg"], dtype=np.float64),
            init_cov_deg2=np.asarray(metadata["init_cov_deg2"], dtype=np.float64),
            init_step_mean_deg=np.asarray(metadata["init_step_mean_deg"], dtype=np.float64),
            init_step_cov_deg2=np.asarray(metadata["init_step_cov_deg2"], dtype=np.float64),
            n_traces=n_total,
            n_frames=frames,
            seed=int(seed),
            center_mode=str(cfg.center_mode),
        ).astype(np.float64)
        prior = base_prior
    elif process_model in {"confined_step_ar1", "scale_mixture_confined_step_ar1"}:
        fit_scale = 1.0 if process_model.startswith("scale_mixture") else float(cfg.global_scale)
        metadata = fit_confined_step_ar1_prior_from_snippets(
            source_centered,
            covariance_mode=str(cfg.covariance_mode),
            global_scale=fit_scale,
            beta=cfg.anti_step_beta,
            kappa=float(cfg.position_spring_kappa),
        )
        base_prior = generate_confined_step_ar1_traces(
            step_transition_beta=float(metadata["step_transition_beta"]),
            position_spring_kappa=float(metadata["position_spring_kappa"]),
            innovation_cov_deg2=np.asarray(metadata["innovation_cov_deg2"], dtype=np.float64),
            init_mean_deg=np.asarray(metadata["init_mean_deg"], dtype=np.float64),
            init_cov_deg2=np.asarray(metadata["init_cov_deg2"], dtype=np.float64),
            init_step_mean_deg=np.asarray(metadata["init_step_mean_deg"], dtype=np.float64),
            init_step_cov_deg2=np.asarray(metadata["init_step_cov_deg2"], dtype=np.float64),
            n_traces=n_total,
            n_frames=frames,
            seed=int(seed),
            center_mode=str(cfg.center_mode),
        ).astype(np.float64)
        prior = base_prior
    elif process_model == "scale_mixture_empirical_confined_step_ar1":
        step_cov = fit_step_covariance_from_snippets(
            source_centered,
            covariance_mode=str(cfg.covariance_mode),
            global_scale=1.0,
        )
        param_samples, param_meta = fit_empirical_confined_step_parameters(
            source_centered,
            beta_min=float(cfg.confined_param_beta_min),
            beta_max=float(cfg.confined_param_beta_max),
            kappa_min=float(cfg.confined_param_kappa_min),
            kappa_max=float(cfg.confined_param_kappa_max),
            max_abs_eigenvalue=float(cfg.confined_param_max_abs_eigenvalue),
        )
        steps = np.diff(source_centered, axis=1).reshape(-1, 2)
        sample_weights = None
        if float(cfg.confined_param_kappa_weight_power) != 0.0:
            sample_weights = np.maximum(param_samples[:, 1], 1e-6) ** float(cfg.confined_param_kappa_weight_power)
        base_prior, chosen_params = generate_empirical_confined_step_ar1_traces(
            beta_kappa_samples=param_samples,
            sample_weights=sample_weights,
            target_step_cov_deg2=step_cov,
            init_mean_deg=np.mean(source_centered[:, 0, :], axis=0),
            init_cov_deg2=regularized_cov(source_centered[:, 0, :]),
            init_step_mean_deg=np.mean(steps, axis=0),
            init_step_cov_deg2=step_cov,
            n_traces=n_total,
            n_frames=frames,
            seed=int(seed),
            center_mode=str(cfg.center_mode),
        )
        prior = base_prior.astype(np.float64)
        metadata = {
            **param_meta,
            "target_step_cov_deg2": step_cov,
            "confined_param_beta_min": float(cfg.confined_param_beta_min),
            "confined_param_beta_max": float(cfg.confined_param_beta_max),
            "confined_param_kappa_min": float(cfg.confined_param_kappa_min),
            "confined_param_kappa_max": float(cfg.confined_param_kappa_max),
            "confined_param_max_abs_eigenvalue": float(cfg.confined_param_max_abs_eigenvalue),
            "confined_param_kappa_weight_power": float(cfg.confined_param_kappa_weight_power),
            "sampled_confined_beta_median": float(np.median(chosen_params[:, 0])),
            "sampled_confined_kappa_median": float(np.median(chosen_params[:, 1])),
        }
    elif process_model == "ar2":
        metadata = fit_ar2_prior_from_snippets(
            source_centered,
            covariance_mode=str(cfg.covariance_mode),
            global_scale=float(cfg.global_scale),
            max_abs_eigenvalue=float(cfg.ou_max_abs_eigenvalue),
        )
        prior = generate_ar2_traces(
            ar2_coef_2x4=np.asarray(metadata["ar2_coef_2x4"], dtype=np.float64),
            innovation_cov_deg2=np.asarray(metadata["innovation_cov_deg2"], dtype=np.float64),
            init_state_mean_deg=np.asarray(metadata["init_state_mean_deg"], dtype=np.float64),
            init_state_cov_deg2=np.asarray(metadata["init_state_cov_deg2"], dtype=np.float64),
            n_traces=n_total,
            n_frames=frames,
            seed=int(seed),
            center_mode=str(cfg.center_mode),
        ).astype(np.float64)
    else:
        raise ValueError(f"Unsupported process_model={process_model!r}; expected {SYNTHETIC_TRAJECTORY_PRIOR_MODELS}")

    if process_model.startswith("scale_mixture"):
        rng = np.random.default_rng(int(seed) + 104729)
        real_targets = trace_features(source_centered, float(cfg.frame_rate_hz))[str(cfg.scale_mixture_feature)].to_numpy(
            dtype=np.float64
        )
        target_values, target_meta = sample_scale_mixture_targets(
            real_targets,
            n=n_total,
            rng=rng,
            distribution=str(cfg.scale_mixture_distribution),
        )
        prior, realized_scale = apply_trace_scale_mixture(
            prior,
            target_values=target_values,
            feature_name=str(cfg.scale_mixture_feature),
            frame_rate_hz=float(cfg.frame_rate_hz),
            center_mode=str(cfg.center_mode),
        )
        metadata = {
            **metadata,
            **target_meta,
            "scale_mixture_feature": str(cfg.scale_mixture_feature),
            "scale_mixture_distribution": str(cfg.scale_mixture_distribution),
            "scale_mixture_realized_scale_mean": float(np.mean(realized_scale)),
            "scale_mixture_realized_scale_median": float(np.median(realized_scale)),
            "scale_mixture_realized_scale_q10": float(np.quantile(realized_scale, 0.10)),
            "scale_mixture_realized_scale_q90": float(np.quantile(realized_scale, 0.90)),
        }

    metadata = {
        "process_model": process_model,
        "n_traces": int(n_total),
        "n_frames": int(frames),
        "seed": int(seed),
        "covariance_mode": str(cfg.covariance_mode),
        "center_mode": str(cfg.center_mode),
        "global_scale": float(cfg.global_scale),
        **metadata,
    }
    return SyntheticTrajectoryPriorResult(
        traces_deg=np.asarray(prior, dtype=np.float32),
        metadata=metadata,
        base_traces_deg=None if base_prior is None else np.asarray(base_prior, dtype=np.float32),
        chosen_confined_params=chosen_params,
    )
