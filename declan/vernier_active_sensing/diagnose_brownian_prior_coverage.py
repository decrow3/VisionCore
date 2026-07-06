#!/usr/bin/env python3
"""Diagnose whether a Brownian FEM prior covers real eye traces.

This is a prior-support diagnostic, not a decoder.  It checks whether traces
drawn from the Brownian generator used in the Vernier demos occupy the same
trace space as held-out real FEM snippets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import synthetic_trajectory_priors as trajectory_priors
from .generate_brownian_eye_traces import COVARIANCE_MODES, CENTER_MODES, estimate_source_covariance
from .trajectories import DEFAULT_EYE_TRACES_PATH, FRAME_RATE_HZ

PROCESS_MODELS = trajectory_priors.SYNTHETIC_TRAJECTORY_PRIOR_MODELS
SCALE_MIXTURE_FEATURES = trajectory_priors.SCALE_MIXTURE_FEATURES
SCALE_MIXTURE_DISTRIBUTIONS = trajectory_priors.SCALE_MIXTURE_DISTRIBUTIONS


FEATURE_NAMES = [
    "rms_radius_arcmin",
    "std_x_arcmin",
    "std_y_arcmin",
    "path_length_arcmin",
    "step_rms_arcmin",
    "max_radius_arcmin",
    "endpoint_radius_arcmin",
    "step_lag1_cosine",
]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.generic):
        item = value.item()
        return _json_ready(item)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(val) for val in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_trace_file(path: Path, n_frames: int) -> np.ndarray:
    with np.load(path, allow_pickle=True) as npz:
        traces = np.asarray(npz["traces"], dtype=np.float64)
        durations = np.asarray(npz["durations"], dtype=np.int64)
    ok = durations >= int(n_frames)
    if not np.any(ok):
        raise ValueError(f"No traces in {path} have at least {n_frames} frames")
    return traces[ok, : int(n_frames), :].astype(np.float64)


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


def trace_features(traces_deg: np.ndarray, frame_rate_hz: float) -> pd.DataFrame:
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


def step_chi2_per_step(traces_deg: np.ndarray, step_cov_deg2: np.ndarray) -> np.ndarray:
    steps = np.diff(np.asarray(traces_deg, dtype=np.float64), axis=1)
    inv = np.linalg.inv(np.asarray(step_cov_deg2, dtype=np.float64))
    chi2 = np.einsum("nti,ij,ntj->nt", steps, inv, steps)
    return np.mean(chi2, axis=1)


def transition_chi2_per_step(
    traces_deg: np.ndarray,
    *,
    transition_matrix: np.ndarray,
    innovation_cov_deg2: np.ndarray,
) -> np.ndarray:
    traces = np.asarray(traces_deg, dtype=np.float64)
    pred = traces[:, :-1, :] @ np.asarray(transition_matrix, dtype=np.float64).T
    resid = traces[:, 1:, :] - pred
    inv = np.linalg.inv(np.asarray(innovation_cov_deg2, dtype=np.float64))
    chi2 = np.einsum("nti,ij,ntj->nt", resid, inv, resid)
    return np.mean(chi2, axis=1)


def fit_step_covariance_from_snippets(
    traces_deg: np.ndarray,
    *,
    covariance_mode: str,
    global_scale: float,
) -> np.ndarray:
    steps = np.diff(np.asarray(traces_deg, dtype=np.float64), axis=1).reshape(-1, 2)
    raw_cov = np.cov(steps.T)
    if str(covariance_mode) == "diagonal_empirical":
        cov = np.diag(np.diag(raw_cov))
    elif str(covariance_mode) == "full_empirical":
        cov = raw_cov
    elif str(covariance_mode) == "isotropic_scalar":
        cov = np.eye(2, dtype=np.float64) * float(np.mean(np.diag(raw_cov)))
    else:
        raise ValueError(f"Unsupported covariance_mode={covariance_mode!r}; expected {COVARIANCE_MODES}")
    return np.asarray(cov, dtype=np.float64) * float(global_scale)


def _regularized_cov(samples: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    cov = np.cov(np.asarray(samples, dtype=np.float64).T)
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    cov = 0.5 * (cov + cov.T)
    eig = np.linalg.eigvalsh(cov)
    min_eig = float(np.min(eig))
    if min_eig < float(floor):
        cov = cov + (float(floor) - min_eig) * np.eye(cov.shape[0])
    return cov


def _covariance_by_mode(raw_cov: np.ndarray, covariance_mode: str, global_scale: float) -> np.ndarray:
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


def _stabilize_transition(matrix: np.ndarray, max_abs_eigenvalue: float) -> np.ndarray:
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
    """Fit a discrete OU / AR(1) process x[t+1] = A x[t] + eps."""
    traces = np.asarray(traces_deg, dtype=np.float64)
    x_prev = traces[:, :-1, :].reshape(-1, 2)
    x_next = traces[:, 1:, :].reshape(-1, 2)
    coef, *_ = np.linalg.lstsq(x_prev, x_next, rcond=None)
    transition = _stabilize_transition(coef.T, max_abs_eigenvalue=float(max_abs_eigenvalue))
    residual = x_next - x_prev @ transition.T
    raw_innovation_cov = _regularized_cov(residual)
    innovation_cov = _covariance_by_mode(
        raw_innovation_cov,
        covariance_mode=str(covariance_mode),
        global_scale=float(global_scale),
    )
    return {
        "transition_matrix": transition,
        "raw_innovation_cov_deg2": raw_innovation_cov,
        "innovation_cov_deg2": innovation_cov,
        "init_mean_deg": np.mean(traces[:, 0, :], axis=0),
        "init_cov_deg2": _regularized_cov(traces[:, 0, :]) * float(global_scale),
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
    """Fit step dynamics d[t+1] = B d[t] + eps, then integrate positions."""
    traces = np.asarray(traces_deg, dtype=np.float64)
    steps = np.diff(traces, axis=1)
    prev = steps[:, :-1, :].reshape(-1, 2)
    nxt = steps[:, 1:, :].reshape(-1, 2)
    coef, *_ = np.linalg.lstsq(prev, nxt, rcond=None)
    transition = _stabilize_transition(coef.T, max_abs_eigenvalue=float(max_abs_eigenvalue))
    residual = nxt - prev @ transition.T
    raw_innovation_cov = _regularized_cov(residual)
    innovation_cov = _covariance_by_mode(
        raw_innovation_cov,
        covariance_mode=str(covariance_mode),
        global_scale=float(global_scale),
    )
    return {
        "step_transition_matrix": transition,
        "raw_innovation_cov_deg2": raw_innovation_cov,
        "innovation_cov_deg2": innovation_cov,
        "init_mean_deg": np.mean(traces[:, 0, :], axis=0),
        "init_cov_deg2": _regularized_cov(traces[:, 0, :]) * float(global_scale),
        "init_step_mean_deg": np.mean(steps[:, 0, :], axis=0),
        "init_step_cov_deg2": _regularized_cov(steps[:, 0, :]) * float(global_scale),
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
) -> dict[str, np.ndarray]:
    """Fit a moment-matched anti-persistent step process.

    The ordinary least-squares step-AR(1) fit underestimates the directional
    reversal in these traces.  This model instead sets a scalar step
    coefficient from the observed lag-1 cosine and chooses innovations so the
    stationary step covariance matches the empirical step covariance.
    """
    traces = np.asarray(traces_deg, dtype=np.float64)
    steps = np.diff(traces, axis=1).reshape(-1, 2)
    step_cov = _covariance_by_mode(
        _regularized_cov(steps),
        covariance_mode=str(covariance_mode),
        global_scale=float(global_scale),
    )
    if beta is None or not np.isfinite(float(beta)):
        beta_value = median_step_lag1_cosine(traces)
    else:
        beta_value = float(beta)
    beta_value = float(np.clip(beta_value, -0.98, 0.98))
    transition = np.eye(2, dtype=np.float64) * beta_value
    innovation_cov = step_cov * max(1.0 - beta_value * beta_value, 1e-6)
    return {
        "step_transition_matrix": transition,
        "target_step_cov_deg2": step_cov,
        "innovation_cov_deg2": innovation_cov,
        "anti_step_beta": beta_value,
        "init_mean_deg": np.mean(traces[:, 0, :], axis=0),
        "init_cov_deg2": _regularized_cov(traces[:, 0, :]) * float(global_scale),
        "init_step_mean_deg": np.mean(steps, axis=0),
        "init_step_cov_deg2": step_cov,
        "step_transition_eigenvalues": np.linalg.eigvals(transition),
    }


def confined_step_companion(beta: float, kappa: float) -> np.ndarray:
    """State transition for x[t], step[t] with step[t+1] = beta step[t] - kappa x[t] + eps."""
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
    """Fit a confined anti-persistent step process.

    The process is:

        step[t+1] = beta * step[t] - kappa * position[t] + eps[t]
        position[t+1] = position[t] + step[t+1]

    The spring term supplies longer-timescale confinement while the beta term
    controls one-step reversal. This is useful when real traces have large
    frame-to-frame steps but remain spatially compact.
    """
    traces = np.asarray(traces_deg, dtype=np.float64)
    steps = np.diff(traces, axis=1).reshape(-1, 2)
    step_cov = _covariance_by_mode(
        _regularized_cov(steps),
        covariance_mode=str(covariance_mode),
        global_scale=float(global_scale),
    )
    if beta is None or not np.isfinite(float(beta)):
        beta_value = median_step_lag1_cosine(traces)
    else:
        beta_value = float(beta)
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
        "init_cov_deg2": _regularized_cov(traces[:, 0, :]) * float(global_scale),
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
) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate a plausible per-trace distribution over confined-step dynamics."""
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
    meta: dict[str, float] = {
        "empirical_confined_param_count": float(params.shape[0]),
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


def step_ar1_chi2_per_step(
    traces_deg: np.ndarray,
    *,
    step_transition_matrix: np.ndarray,
    innovation_cov_deg2: np.ndarray,
) -> np.ndarray:
    steps = np.diff(np.asarray(traces_deg, dtype=np.float64), axis=1)
    pred = steps[:, :-1, :] @ np.asarray(step_transition_matrix, dtype=np.float64).T
    resid = steps[:, 1:, :] - pred
    inv = np.linalg.inv(np.asarray(innovation_cov_deg2, dtype=np.float64))
    chi2 = np.einsum("nti,ij,ntj->nt", resid, inv, resid)
    return np.mean(chi2, axis=1)


def confined_step_ar1_chi2_per_step(
    traces_deg: np.ndarray,
    *,
    step_transition_beta: float,
    position_spring_kappa: float,
    innovation_cov_deg2: np.ndarray,
) -> np.ndarray:
    traces = np.asarray(traces_deg, dtype=np.float64)
    steps = np.diff(traces, axis=1)
    pred = float(step_transition_beta) * steps[:, :-1, :] - float(position_spring_kappa) * traces[:, 1:-1, :]
    resid = steps[:, 1:, :] - pred
    inv = np.linalg.inv(np.asarray(innovation_cov_deg2, dtype=np.float64))
    chi2 = np.einsum("nti,ij,ntj->nt", resid, inv, resid)
    return np.mean(chi2, axis=1)


def _ar2_companion(coef_2x4: np.ndarray) -> np.ndarray:
    coef = np.asarray(coef_2x4, dtype=np.float64)
    companion = np.zeros((4, 4), dtype=np.float64)
    companion[:2, :] = coef
    companion[2:, :2] = np.eye(2, dtype=np.float64)
    return companion


def _stabilize_ar2(coef_2x4: np.ndarray, max_abs_eigenvalue: float) -> np.ndarray:
    coef = np.asarray(coef_2x4, dtype=np.float64)
    radius = float(np.max(np.abs(np.linalg.eigvals(_ar2_companion(coef)))))
    if radius <= float(max_abs_eigenvalue):
        return coef
    # Shrink the predictive coefficients conservatively. This is not a refit;
    # it just prevents explosive synthetic samples from a noisy finite fit.
    return coef * (float(max_abs_eigenvalue) / radius)


def fit_ar2_prior_from_snippets(
    traces_deg: np.ndarray,
    *,
    covariance_mode: str,
    global_scale: float,
    max_abs_eigenvalue: float,
) -> dict[str, np.ndarray]:
    """Fit x[t+1] = C [x[t], x[t-1]] + eps."""
    traces = np.asarray(traces_deg, dtype=np.float64)
    design = np.concatenate([traces[:, 1:-1, :], traces[:, :-2, :]], axis=2).reshape(-1, 4)
    target = traces[:, 2:, :].reshape(-1, 2)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    coef_2x4 = _stabilize_ar2(coef.T, max_abs_eigenvalue=float(max_abs_eigenvalue))
    residual = target - design @ coef_2x4.T
    raw_innovation_cov = _regularized_cov(residual)
    innovation_cov = _covariance_by_mode(
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
        "init_state_cov_deg2": _regularized_cov(init_state) * float(global_scale),
        "ar2_companion_eigenvalues": np.linalg.eigvals(_ar2_companion(coef_2x4)),
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


def ar2_chi2_per_step(
    traces_deg: np.ndarray,
    *,
    ar2_coef_2x4: np.ndarray,
    innovation_cov_deg2: np.ndarray,
) -> np.ndarray:
    traces = np.asarray(traces_deg, dtype=np.float64)
    design = np.concatenate([traces[:, 1:-1, :], traces[:, :-2, :]], axis=2)
    pred = design @ np.asarray(ar2_coef_2x4, dtype=np.float64).T
    resid = traces[:, 2:, :] - pred
    inv = np.linalg.inv(np.asarray(innovation_cov_deg2, dtype=np.float64))
    chi2 = np.einsum("nti,ij,ntj->nt", resid, inv, resid)
    return np.mean(chi2, axis=1)


def zscore_features(
    features: pd.DataFrame,
    reference: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    ref = reference[feature_names].to_numpy(dtype=np.float64)
    mu = np.nanmean(ref, axis=0)
    sd = np.nanstd(ref, axis=0)
    sd[sd < 1e-9] = 1.0
    arr = features[feature_names].to_numpy(dtype=np.float64)
    z = (arr - mu[None, :]) / sd[None, :]
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def zscore_paths(traces_deg: np.ndarray, reference_deg: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference_deg, dtype=np.float64).reshape(reference_deg.shape[0], -1)
    mu = np.mean(ref, axis=0)
    sd = np.std(ref, axis=0)
    sd[sd < 1e-9] = 1.0
    arr = np.asarray(traces_deg, dtype=np.float64).reshape(traces_deg.shape[0], -1)
    return (arr - mu[None, :]) / sd[None, :]


def nearest_neighbor(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    best_dist = np.full(q.shape[0], np.inf, dtype=np.float64)
    best_idx = np.full(q.shape[0], -1, dtype=np.int64)
    r_norm = np.sum(r * r, axis=1)
    for start in range(0, q.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), q.shape[0])
        qc = q[start:stop]
        dist2 = np.sum(qc * qc, axis=1)[:, None] + r_norm[None, :] - 2.0 * qc @ r.T
        idx = np.argmin(dist2, axis=1)
        best_idx[start:stop] = idx
        best_dist[start:stop] = np.sqrt(np.maximum(dist2[np.arange(stop - start), idx], 0.0))
    return best_dist, best_idx


def summarize_distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_q10": float("nan"),
            f"{prefix}_q25": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_q75": float("nan"),
            f"{prefix}_q90": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_q10": float(np.quantile(arr, 0.10)),
        f"{prefix}_q25": float(np.quantile(arr, 0.25)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_q75": float(np.quantile(arr, 0.75)),
        f"{prefix}_q90": float(np.quantile(arr, 0.90)),
    }


def percentile_against_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    finite_ref = np.sort(ref[np.isfinite(ref)])
    out = np.full(vals.shape, np.nan, dtype=np.float64)
    finite_vals = np.isfinite(vals)
    if finite_ref.size == 0:
        return out
    out[finite_vals] = np.searchsorted(finite_ref, vals[finite_vals], side="right") / finite_ref.size
    return out


def fraction_below_percentile(percentiles: np.ndarray, cutoff: float) -> float:
    arr = np.asarray(percentiles, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr <= float(cutoff)))


def median_finite(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def write_plots(
    out_dir: Path,
    *,
    feature_table: pd.DataFrame,
    nn_table: pd.DataFrame,
    real_traces: np.ndarray,
    prior_train: np.ndarray,
    real_path_nn_idx: np.ndarray,
    max_examples: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_plot_names = [
        "rms_radius_arcmin",
        "path_length_arcmin",
        "step_rms_arcmin",
        "step_lag1_cosine",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, name in zip(axes.ravel(), feature_plot_names, strict=True):
        for source, color in [("real", "#222222"), ("prior_sample", "#1f77b4"), ("existing_synthetic", "#ff7f0e")]:
            vals = feature_table.loc[feature_table["source"].eq(source), name].to_numpy(dtype=np.float64)
            if vals.size == 0:
                continue
            ax.hist(vals, bins=32, density=True, histtype="step", lw=1.8, color=color, label=source)
        ax.set_title(name)
        ax.set_ylabel("density")
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "brownian_prior_feature_distributions.png", dpi=200)
    fig.savefig(out_dir / "brownian_prior_feature_distributions.pdf")
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric, title in [
        (axes2[0], "feature_nn_zdist", "nearest-neighbor distance in feature space"),
        (axes2[1], "path_nn_zdist", "nearest-neighbor distance in trace space"),
    ]:
        for source, color in [("real_to_prior", "#222222"), ("prior_holdout_to_prior", "#1f77b4"), ("existing_synthetic_to_prior", "#ff7f0e")]:
            vals = nn_table.loc[nn_table["comparison"].eq(source), metric].to_numpy(dtype=np.float64)
            if vals.size == 0:
                continue
            ax.hist(vals, bins=32, density=True, histtype="step", lw=1.8, color=color, label=source)
        ax.set_title(title)
        ax.set_xlabel("z-scored Euclidean NN distance")
        ax.set_ylabel("density")
    axes2[0].legend(frameon=False, fontsize=8)
    fig2.savefig(out_dir / "brownian_prior_nearest_neighbor_distances.png", dpi=200)
    fig2.savefig(out_dir / "brownian_prior_nearest_neighbor_distances.pdf")
    plt.close(fig2)

    n = min(int(max_examples), real_traces.shape[0])
    if n <= 0:
        return
    real_subset = real_traces[:n] * 60.0
    synth_subset = prior_train[real_path_nn_idx[:n]] * 60.0
    fig3, axes3 = plt.subplots(n, 2, figsize=(7, 2.2 * n), constrained_layout=True, squeeze=False)
    for idx in range(n):
        for col, (trace, title) in enumerate(
            [
                (real_subset[idx], f"real {idx}"),
                (synth_subset[idx], f"nearest prior {int(real_path_nn_idx[idx])}"),
            ]
        ):
            ax = axes3[idx, col]
            ax.plot(trace[:, 0], trace[:, 1], lw=1.2)
            ax.scatter(trace[0, 0], trace[0, 1], s=14, color="#2ca02c")
            ax.scatter(trace[-1, 0], trace[-1, 1], s=14, color="#d62728")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title)
            ax.set_xlabel("x arcmin")
            ax.set_ylabel("y arcmin")
    fig3.savefig(out_dir / "real_trace_nearest_prior_examples.png", dpi=200)
    fig3.savefig(out_dir / "real_trace_nearest_prior_examples.pdf")
    plt.close(fig3)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_raw = load_trace_file(Path(args.real_eye_traces), int(args.n_frames))
    real = recenter_traces(real_raw, str(args.center_mode))
    stats = estimate_source_covariance(
        Path(args.real_eye_traces),
        covariance_mode=str(args.covariance_mode),
        global_scale=float(args.global_scale),
    )
    process_model = str(args.process_model)
    brownian_step_cov = None
    if process_model == "brownian" and not bool(args.fit_covariance_from_window):
        brownian_step_cov = np.asarray(stats["synthetic_step_cov_deg2_per_frame"], dtype=np.float64)
    prior_config = trajectory_priors.SyntheticTrajectoryPriorConfig(
        process_model=process_model,
        covariance_mode=str(args.covariance_mode),
        center_mode=str(args.center_mode),
        global_scale=float(args.global_scale),
        frame_rate_hz=float(args.frame_rate_hz),
        brownian_step_cov_deg2_per_frame=brownian_step_cov,
        brownian_center_deg=np.asarray(stats["source_sample_mean_deg"], dtype=np.float64),
        ou_max_abs_eigenvalue=float(args.ou_max_abs_eigenvalue),
        anti_step_beta=None if not np.isfinite(float(args.anti_step_beta)) else float(args.anti_step_beta),
        position_spring_kappa=float(args.position_spring_kappa),
        scale_mixture_feature=str(args.scale_mixture_feature),
        scale_mixture_distribution=str(args.scale_mixture_distribution),
        confined_param_beta_min=float(args.confined_param_beta_min),
        confined_param_beta_max=float(args.confined_param_beta_max),
        confined_param_kappa_min=float(args.confined_param_kappa_min),
        confined_param_kappa_max=float(args.confined_param_kappa_max),
        confined_param_max_abs_eigenvalue=float(args.confined_param_max_abs_eigenvalue),
        confined_param_kappa_weight_power=float(args.confined_param_kappa_weight_power),
    )
    prior_result = trajectory_priors.generate_synthetic_trajectory_prior(
        real_raw,
        n_traces=int(args.n_prior_samples) + int(args.n_prior_holdout),
        n_frames=int(args.n_frames),
        seed=int(args.seed),
        config=prior_config,
    )
    prior = np.asarray(prior_result.traces_deg, dtype=np.float64)
    model_payload = dict(prior_result.metadata)
    if process_model == "brownian":
        model_payload["source_step_cov_deg2_per_frame"] = stats["source_step_cov_deg2_per_frame"]
    prior_train = prior[: int(args.n_prior_samples)]
    prior_holdout = prior[int(args.n_prior_samples) :]

    existing = None
    if args.existing_synthetic_traces:
        existing = load_trace_file(Path(args.existing_synthetic_traces), int(args.n_frames))
        existing = recenter_traces(existing, str(args.center_mode))

    real_features = trace_features(real, float(args.frame_rate_hz))
    train_features = trace_features(prior_train, float(args.frame_rate_hz))
    holdout_features = trace_features(prior_holdout, float(args.frame_rate_hz))
    real_features.insert(0, "source", "real")
    train_features.insert(0, "source", "prior_sample")
    holdout_features.insert(0, "source", "prior_holdout")
    frames = [real_features, train_features, holdout_features]
    if existing is not None:
        existing_features = trace_features(existing, float(args.frame_rate_hz))
        existing_features.insert(0, "source", "existing_synthetic")
        frames.append(existing_features)
    feature_table = pd.concat(frames, ignore_index=True)

    real_z = zscore_features(real_features, train_features, FEATURE_NAMES)
    holdout_z = zscore_features(holdout_features, train_features, FEATURE_NAMES)
    train_z = zscore_features(train_features, train_features, FEATURE_NAMES)
    real_feature_nn, real_feature_idx = nearest_neighbor(real_z, train_z)
    holdout_feature_nn, _holdout_feature_idx = nearest_neighbor(holdout_z, train_z)

    real_path_z = zscore_paths(real, prior_train)
    holdout_path_z = zscore_paths(prior_holdout, prior_train)
    train_path_z = zscore_paths(prior_train, prior_train)
    real_path_nn, real_path_idx = nearest_neighbor(real_path_z, train_path_z)
    holdout_path_nn, _holdout_path_idx = nearest_neighbor(holdout_path_z, train_path_z)

    nn_rows = []
    for label, feature_nn, path_nn in [
        ("real_to_prior", real_feature_nn, real_path_nn),
        ("prior_holdout_to_prior", holdout_feature_nn, holdout_path_nn),
    ]:
        for idx, (f_dist, p_dist) in enumerate(zip(feature_nn, path_nn, strict=True)):
            nn_rows.append(
                {
                    "comparison": label,
                    "trace_index": idx,
                    "feature_nn_zdist": float(f_dist),
                    "path_nn_zdist": float(p_dist),
                }
            )
    existing_feature_nn = np.array([], dtype=np.float64)
    existing_path_nn = np.array([], dtype=np.float64)
    if existing is not None:
        existing_features_source = feature_table[feature_table["source"].eq("existing_synthetic")].reset_index(drop=True)
        existing_z = zscore_features(existing_features_source, train_features, FEATURE_NAMES)
        existing_feature_nn, _existing_feature_idx = nearest_neighbor(existing_z, train_z)
        existing_path_z = zscore_paths(existing, prior_train)
        existing_path_nn, _existing_path_idx = nearest_neighbor(existing_path_z, train_path_z)
        for idx, (f_dist, p_dist) in enumerate(zip(existing_feature_nn, existing_path_nn, strict=True)):
            nn_rows.append(
                {
                    "comparison": "existing_synthetic_to_prior",
                    "trace_index": idx,
                    "feature_nn_zdist": float(f_dist),
                    "path_nn_zdist": float(p_dist),
                }
            )
    nn_table = pd.DataFrame(nn_rows)

    if process_model == "brownian":
        step_cov = np.asarray(model_payload["step_cov_deg2_per_frame"], dtype=np.float64)
        real_chi2 = step_chi2_per_step(real, step_cov)
        train_chi2 = step_chi2_per_step(prior_train, step_cov)
        holdout_chi2 = step_chi2_per_step(prior_holdout, step_cov)
    elif process_model == "ou":
        transition = np.asarray(model_payload["transition_matrix"], dtype=np.float64)
        innovation_cov = np.asarray(model_payload["innovation_cov_deg2"], dtype=np.float64)
        real_chi2 = transition_chi2_per_step(
            real,
            transition_matrix=transition,
            innovation_cov_deg2=innovation_cov,
        )
        train_chi2 = transition_chi2_per_step(
            prior_train,
            transition_matrix=transition,
            innovation_cov_deg2=innovation_cov,
        )
        holdout_chi2 = transition_chi2_per_step(
            prior_holdout,
            transition_matrix=transition,
            innovation_cov_deg2=innovation_cov,
        )
    elif process_model in {"step_ar1", "anti_step_ar1"}:
        transition = np.asarray(model_payload["step_transition_matrix"], dtype=np.float64)
        innovation_cov = np.asarray(model_payload["innovation_cov_deg2"], dtype=np.float64)
        real_chi2 = step_ar1_chi2_per_step(
            real,
            step_transition_matrix=transition,
            innovation_cov_deg2=innovation_cov,
        )
        train_chi2 = step_ar1_chi2_per_step(
            prior_train,
            step_transition_matrix=transition,
            innovation_cov_deg2=innovation_cov,
        )
        holdout_chi2 = step_ar1_chi2_per_step(
            prior_holdout,
            step_transition_matrix=transition,
            innovation_cov_deg2=innovation_cov,
        )
    elif process_model == "scale_mixture_anti_step_ar1":
        real_chi2 = np.full(real.shape[0], np.nan, dtype=np.float64)
        train_chi2 = np.full(prior_train.shape[0], np.nan, dtype=np.float64)
        holdout_chi2 = np.full(prior_holdout.shape[0], np.nan, dtype=np.float64)
    elif process_model == "confined_step_ar1":
        real_chi2 = confined_step_ar1_chi2_per_step(
            real,
            step_transition_beta=float(model_payload["step_transition_beta"]),
            position_spring_kappa=float(model_payload["position_spring_kappa"]),
            innovation_cov_deg2=np.asarray(model_payload["innovation_cov_deg2"], dtype=np.float64),
        )
        train_chi2 = confined_step_ar1_chi2_per_step(
            prior_train,
            step_transition_beta=float(model_payload["step_transition_beta"]),
            position_spring_kappa=float(model_payload["position_spring_kappa"]),
            innovation_cov_deg2=np.asarray(model_payload["innovation_cov_deg2"], dtype=np.float64),
        )
        holdout_chi2 = confined_step_ar1_chi2_per_step(
            prior_holdout,
            step_transition_beta=float(model_payload["step_transition_beta"]),
            position_spring_kappa=float(model_payload["position_spring_kappa"]),
            innovation_cov_deg2=np.asarray(model_payload["innovation_cov_deg2"], dtype=np.float64),
        )
    elif process_model == "scale_mixture_confined_step_ar1":
        real_chi2 = np.full(real.shape[0], np.nan, dtype=np.float64)
        train_chi2 = np.full(prior_train.shape[0], np.nan, dtype=np.float64)
        holdout_chi2 = np.full(prior_holdout.shape[0], np.nan, dtype=np.float64)
    elif process_model == "scale_mixture_empirical_confined_step_ar1":
        real_chi2 = np.full(real.shape[0], np.nan, dtype=np.float64)
        train_chi2 = np.full(prior_train.shape[0], np.nan, dtype=np.float64)
        holdout_chi2 = np.full(prior_holdout.shape[0], np.nan, dtype=np.float64)
    elif process_model == "ar2":
        coef = np.asarray(model_payload["ar2_coef_2x4"], dtype=np.float64)
        innovation_cov = np.asarray(model_payload["innovation_cov_deg2"], dtype=np.float64)
        real_chi2 = ar2_chi2_per_step(
            real,
            ar2_coef_2x4=coef,
            innovation_cov_deg2=innovation_cov,
        )
        train_chi2 = ar2_chi2_per_step(
            prior_train,
            ar2_coef_2x4=coef,
            innovation_cov_deg2=innovation_cov,
        )
        holdout_chi2 = ar2_chi2_per_step(
            prior_holdout,
            ar2_coef_2x4=coef,
            innovation_cov_deg2=innovation_cov,
        )
    else:
        raise ValueError(f"Unsupported process_model={process_model!r}; expected {PROCESS_MODELS}")
    feature_percentile = percentile_against_reference(real_feature_nn, holdout_feature_nn)
    path_percentile = percentile_against_reference(real_path_nn, holdout_path_nn)
    chi2_percentile = (
        percentile_against_reference(real_chi2, holdout_chi2)
        if np.any(np.isfinite(real_chi2)) and np.any(np.isfinite(holdout_chi2))
        else np.full(real.shape[0], np.nan, dtype=np.float64)
    )

    summary = {
        "real_eye_traces": Path(args.real_eye_traces),
        "existing_synthetic_traces": Path(args.existing_synthetic_traces) if args.existing_synthetic_traces else None,
        "out_dir": out_dir,
        "n_frames": int(args.n_frames),
        "n_real_traces": int(real.shape[0]),
        "n_prior_samples": int(prior_train.shape[0]),
        "n_prior_holdout": int(prior_holdout.shape[0]),
        "center_mode": str(args.center_mode),
        "covariance_mode": str(args.covariance_mode),
        "process_model": process_model,
        "global_scale": float(args.global_scale),
        "fit_covariance_from_window": bool(args.fit_covariance_from_window),
        "ou_max_abs_eigenvalue": float(args.ou_max_abs_eigenvalue),
        "anti_step_beta_arg": float(args.anti_step_beta),
        "position_spring_kappa_arg": float(args.position_spring_kappa),
        "confined_param_beta_min_arg": float(args.confined_param_beta_min),
        "confined_param_beta_max_arg": float(args.confined_param_beta_max),
        "confined_param_kappa_min_arg": float(args.confined_param_kappa_min),
        "confined_param_kappa_max_arg": float(args.confined_param_kappa_max),
        "confined_param_max_abs_eigenvalue_arg": float(args.confined_param_max_abs_eigenvalue),
        "confined_param_kappa_weight_power_arg": float(args.confined_param_kappa_weight_power),
        "scale_mixture_feature_arg": str(args.scale_mixture_feature),
        "scale_mixture_distribution_arg": str(args.scale_mixture_distribution),
        "seed": int(args.seed),
        **model_payload,
        "real_feature_nn_percent_below_prior_holdout_q90": fraction_below_percentile(feature_percentile, 0.90),
        "real_path_nn_percent_below_prior_holdout_q90": fraction_below_percentile(path_percentile, 0.90),
        "real_step_chi2_percent_below_prior_holdout_q90": fraction_below_percentile(chi2_percentile, 0.90),
        "real_feature_nn_median_percentile_vs_prior_holdout": median_finite(feature_percentile),
        "real_path_nn_median_percentile_vs_prior_holdout": median_finite(path_percentile),
        "real_step_chi2_median_percentile_vs_prior_holdout": median_finite(chi2_percentile),
        **summarize_distribution(real_feature_nn, "real_feature_nn_zdist"),
        **summarize_distribution(holdout_feature_nn, "prior_holdout_feature_nn_zdist"),
        **summarize_distribution(real_path_nn, "real_path_nn_zdist"),
        **summarize_distribution(holdout_path_nn, "prior_holdout_path_nn_zdist"),
        **summarize_distribution(real_chi2, "real_step_chi2_per_step"),
        **summarize_distribution(holdout_chi2, "prior_holdout_step_chi2_per_step"),
        **summarize_distribution(train_chi2, "prior_train_step_chi2_per_step"),
        **summarize_distribution(real_chi2, "real_model_chi2_per_step"),
        **summarize_distribution(holdout_chi2, "prior_holdout_model_chi2_per_step"),
        **summarize_distribution(train_chi2, "prior_train_model_chi2_per_step"),
    }

    feature_table.to_csv(out_dir / "brownian_prior_trace_features.csv", index=False)
    nn_table.to_csv(out_dir / "brownian_prior_nearest_neighbor_distances.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "brownian_prior_coverage_summary.csv", index=False)
    write_json(out_dir / "brownian_prior_coverage_manifest.json", summary)
    write_plots(
        out_dir,
        feature_table=feature_table,
        nn_table=nn_table,
        real_traces=real,
        prior_train=prior_train,
        real_path_nn_idx=real_path_idx,
        max_examples=int(args.max_examples),
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-eye-traces", type=Path, default=DEFAULT_EYE_TRACES_PATH)
    parser.add_argument("--existing-synthetic-traces", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=60)
    parser.add_argument("--n-prior-samples", type=int, default=4096)
    parser.add_argument("--n-prior-holdout", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--frame-rate-hz", type=float, default=FRAME_RATE_HZ)
    parser.add_argument("--covariance-mode", choices=COVARIANCE_MODES, default="diagonal_empirical")
    parser.add_argument("--center-mode", choices=CENTER_MODES, default="zero_mean")
    parser.add_argument("--process-model", choices=PROCESS_MODELS, default="brownian")
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--fit-covariance-from-window", action="store_true")
    parser.add_argument("--ou-max-abs-eigenvalue", type=float, default=0.999)
    parser.add_argument("--anti-step-beta", type=float, default=float("nan"))
    parser.add_argument("--position-spring-kappa", type=float, default=0.0)
    parser.add_argument("--confined-param-beta-min", type=float, default=-0.95)
    parser.add_argument("--confined-param-beta-max", type=float, default=0.50)
    parser.add_argument("--confined-param-kappa-min", type=float, default=0.0)
    parser.add_argument("--confined-param-kappa-max", type=float, default=1.50)
    parser.add_argument("--confined-param-max-abs-eigenvalue", type=float, default=0.995)
    parser.add_argument("--confined-param-kappa-weight-power", type=float, default=0.0)
    parser.add_argument("--scale-mixture-feature", choices=SCALE_MIXTURE_FEATURES, default="step_rms_arcmin")
    parser.add_argument(
        "--scale-mixture-distribution",
        choices=SCALE_MIXTURE_DISTRIBUTIONS,
        default="lognormal_median_iqr",
    )
    parser.add_argument("--max-examples", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(
        "{} coverage: feature q90={:.3f}, path q90={:.3f}, model-chi2 q90={:.3f}; wrote {}".format(
            str(summary["process_model"]),
            float(summary["real_feature_nn_percent_below_prior_holdout_q90"]),
            float(summary["real_path_nn_percent_below_prior_holdout_q90"]),
            float(summary["real_step_chi2_percent_below_prior_holdout_q90"]),
            summary["out_dir"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
