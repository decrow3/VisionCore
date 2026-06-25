"""Cache-only continuous joint image/trajectory observer.

This analyzer is the first, deliberately small implementation of the Figure 4C
"true joint estimator" plan.  It reuses response-table caches, fits a local
linear eye-displacement observation model in a chosen response basis, and scores
each image candidate with a two-dimensional AR(1) Kalman observer.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from .likelihood import logmeanexp, poisson_expected_count_loglik, posterior_from_log_scores, rank_desc, true_margin
    from .observer import score_image_identity_score_vectors
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.backimage_trajectory_observer.likelihood import (
        logmeanexp,
        poisson_expected_count_loglik,
        posterior_from_log_scores,
        rank_desc,
        true_margin,
    )
    from declan.backimage_trajectory_observer.observer import score_image_identity_score_vectors


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _parse_csv_values(text: str) -> set[str]:
    return {part.strip() for part in str(text).split(",") if part.strip()}


def _parse_float_schedule(text: str | None) -> list[float]:
    if text is None or str(text).strip() == "":
        return []
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    for value in values:
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("smoothing schedule entries must be finite and non-negative")
    return values


def _parse_scale_value_map(text: str | None, *, value_type: type, name: str) -> dict[float, int | float]:
    if text is None or str(text).strip() == "":
        return {}
    out: dict[float, int | float] = {}
    for part in str(text).split(","):
        piece = part.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"{name} entries must be scale:value pairs, got {piece!r}")
        scale_text, value_text = piece.split(":", 1)
        scale = float(scale_text.strip())
        if not np.isfinite(scale):
            raise ValueError(f"{name} scale must be finite, got {scale_text!r}")
        value = value_type(value_text.strip())
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{name} value must be finite, got {value_text!r}")
        out[float(scale)] = value
    return out


def _scale_value(mapping: dict[float, int | float], scale: float, default: int | float) -> int | float:
    scale_value = float(scale)
    if np.isfinite(scale_value):
        for key, value in mapping.items():
            if np.isclose(scale_value, float(key), rtol=0.0, atol=1e-9):
                return value
    return default


def _scale_string_value(mapping: dict[float, int | float], scale: float, default: str) -> str:
    scale_value = float(scale)
    if np.isfinite(scale_value):
        for key, value in mapping.items():
            if np.isclose(scale_value, float(key), rtol=0.0, atol=1e-9):
                return str(value)
    return str(default)


def _validate_positive_float(value: float, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return out


def _scalar_int(table: dict[str, np.ndarray], key: str, default: int = -1) -> int:
    if key not in table:
        return int(default)
    arr = np.asarray(table[key]).reshape(-1)
    return int(arr[0]) if arr.size else int(default)


def _candidate_ids(table: dict[str, np.ndarray], n_candidates: int) -> list[str]:
    if "candidate_ids" not in table:
        return [str(i) for i in range(int(n_candidates))]
    return [str(v) for v in np.asarray(table["candidate_ids"]).tolist()]


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


def _as_basis(basis: np.ndarray | None, n_units: int) -> np.ndarray:
    if basis is None:
        return np.eye(int(n_units), dtype=np.float64)
    u = np.asarray(basis, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError(f"basis must be (unit, k), got {u.shape}")
    if u.shape[0] != int(n_units):
        raise ValueError(f"basis unit count {u.shape[0]} does not match n_units={n_units}")
    if u.shape[1] <= 0:
        raise ValueError("basis must contain at least one component")
    if not np.isfinite(u).all():
        raise ValueError("basis contains non-finite values")
    gram = u.T @ u
    if float(np.linalg.norm(gram - np.eye(gram.shape[0]), ord="fro")) > 1e-5:
        u, _r = np.linalg.qr(u)
    return u[:, : min(u.shape[1], int(n_units))]


def _trajectory_xy_by_candidate(
    trajectory_xy: np.ndarray,
    *,
    n_candidates: int,
    n_trajectories: int,
    n_time: int,
) -> np.ndarray:
    xy = np.asarray(trajectory_xy, dtype=np.float64)
    if xy.shape == (n_trajectories, n_time, 2):
        xy = np.broadcast_to(xy[None, ...], (n_candidates, n_trajectories, n_time, 2)).copy()
    elif xy.shape == (n_candidates, n_trajectories, n_time, 2):
        xy = xy.copy()
    else:
        raise ValueError(
            "trajectory_xy must be (trajectory, time, 2) or "
            f"(candidate, trajectory, time, 2), got {xy.shape}"
        )
    if not np.isfinite(xy).all():
        raise ValueError("trajectory_xy contains non-finite values")
    return xy


def project_response_delta(delta_counts: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project response deltas into basis coordinates, preserving leading axes."""
    delta = np.asarray(delta_counts, dtype=np.float64)
    u = np.asarray(basis, dtype=np.float64)
    if delta.shape[-1] != u.shape[0]:
        raise ValueError(f"delta unit count {delta.shape[-1]} does not match basis {u.shape}")
    if not np.isfinite(delta).all():
        raise ValueError("delta_counts contains non-finite values")
    return np.tensordot(delta, u, axes=([-1], [0]))


def _smooth_time_axis(values: np.ndarray, sigma: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    sig = float(sigma)
    if sig <= 0.0:
        return arr.copy()
    if not np.isfinite(sig):
        raise ValueError("time_smoothing_sigma must be finite and non-negative")
    radius = max(1, int(np.ceil(3.0 * sig)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sig) ** 2)
    kernel /= float(np.sum(kernel))
    padded = np.pad(arr, [(radius, radius), *[(0, 0) for _ in arr.shape[1:]]], mode="edge")
    out = np.empty_like(arr)
    for time_index in range(arr.shape[0]):
        window = padded[time_index : time_index + 2 * radius + 1]
        out[time_index] = np.tensordot(kernel, window, axes=([0], [0]))
    return out


def _time_varying_residuals(z_i: np.ndarray, xy_i: np.ndarray, keep: np.ndarray, h_i: np.ndarray) -> np.ndarray:
    pred = np.einsum("tkd,rtd->rtk", np.asarray(h_i, dtype=np.float64), np.asarray(xy_i[keep], dtype=np.float64))
    return np.asarray(z_i[keep], dtype=np.float64) - pred


def _observation_matrix_condition_metrics(matrix: np.ndarray) -> dict[str, float]:
    h = np.asarray(matrix, dtype=np.float64)
    if h.ndim == 2:
        h_by_t = h[None, :, :]
    elif h.ndim == 3:
        h_by_t = h
    else:
        raise ValueError(f"observation matrix must be (k,2) or (time,k,2), got {h.shape}")
    singular_values = np.linalg.svd(h_by_t, compute_uv=False)
    if singular_values.ndim != 2 or singular_values.shape[1] < 2:
        raise ValueError(f"expected at least two singular values per time, got {singular_values.shape}")
    s1 = singular_values[:, 0]
    s2 = singular_values[:, 1]
    denom = np.maximum(s1, 1e-12)
    anisotropy = s2 / denom
    condition = s1 / np.maximum(s2, 1e-12)
    area = s1 * s2
    return {
        "A_singular1_median": float(np.median(s1)),
        "A_singular2_median": float(np.median(s2)),
        "A_singular2_p10": float(np.quantile(s2, 0.10)),
        "A_anisotropy_median": float(np.median(anisotropy)),
        "A_condition_median": float(np.median(condition)),
        "A_log10_condition_median": float(np.median(np.log10(np.maximum(condition, 1.0)))),
        "A_area_median": float(np.median(area)),
        "A_rank1_fraction_anisotropy_lt_0p1": float(np.mean(anisotropy < 0.10)),
        "A_rank1_fraction_anisotropy_lt_0p2": float(np.mean(anisotropy < 0.20)),
    }


def fit_time_constant_observation_matrices(
    *,
    prior_lambda_counts: np.ndarray,
    zero_lambda_counts: np.ndarray,
    trajectory_xy: np.ndarray,
    basis: np.ndarray | None = None,
    heldout_trajectory_index: int | None = None,
    observation_model: str = "time_constant",
    time_smoothing_sigma: float = 0.0,
    time_shrinkage: float = 0.0,
    ridge: float = 1e-6,
) -> dict[str, Any]:
    """Fit candidate-specific ``z_t = A_I tau_t + eps_t`` matrices.

    If ``heldout_trajectory_index`` is a valid catalog index, that trajectory is
    excluded from every candidate fit.  This is the anti-circularity guard for
    include-self response tables.  Leave-one-out tables usually encode the true
    trajectory as ``-1``; in that case the retained catalog is used as-is.
    """
    prior = _as_prior_table(prior_lambda_counts, "prior_lambda_counts")
    n_candidates, n_traj, n_time, n_units = prior.shape
    zero = _as_candidate_table(zero_lambda_counts, "zero_lambda_counts", (n_candidates, n_time, n_units))
    u = _as_basis(basis, n_units)
    xy = _trajectory_xy_by_candidate(
        trajectory_xy,
        n_candidates=n_candidates,
        n_trajectories=n_traj,
        n_time=n_time,
    )
    ridge_val = float(ridge)
    if ridge_val < 0.0 or not np.isfinite(ridge_val):
        raise ValueError("ridge must be finite and non-negative")
    model = str(observation_model)
    if model not in {"time_constant", "time_varying"}:
        raise ValueError("observation_model must be 'time_constant' or 'time_varying'")
    smooth_sigma = float(time_smoothing_sigma)
    if smooth_sigma < 0.0 or not np.isfinite(smooth_sigma):
        raise ValueError("time_smoothing_sigma must be finite and non-negative")
    shrink = float(time_shrinkage)
    if shrink < 0.0 or shrink > 1.0 or not np.isfinite(shrink):
        raise ValueError("time_shrinkage must be finite and in [0, 1]")

    z = project_response_delta(prior - zero[:, None, :, :], u)
    k_dim = int(u.shape[1])
    if model == "time_constant":
        matrices = np.empty((n_candidates, k_dim, 2), dtype=np.float64)
    else:
        matrices = np.empty((n_candidates, n_time, k_dim, 2), dtype=np.float64)
    residual_variance = np.empty(n_candidates, dtype=np.float64)
    fit_rows: list[dict[str, Any]] = []
    heldout = -1 if heldout_trajectory_index is None else int(heldout_trajectory_index)

    for candidate_index in range(n_candidates):
        keep = np.ones(n_traj, dtype=bool)
        excluded = False
        if 0 <= heldout < n_traj:
            keep[heldout] = False
            excluded = True
        if not bool(np.any(keep)):
            raise ValueError(
                "heldout_trajectory_index excludes the only available trajectory; "
                "fit A_I from a leave-one-out catalog or a finite-difference Jacobian instead"
            )
        residual_chunks: list[np.ndarray] = []
        n_fit_samples = 0
        if model == "time_constant":
            x = xy[candidate_index, keep].reshape(-1, 2)
            y = z[candidate_index, keep].reshape(-1, k_dim)
            good = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
            if int(np.sum(good)) < 2:
                raise ValueError(f"candidate {candidate_index} has too few finite trajectory samples to fit A_I")
            x = x[good]
            y = y[good]
            normal = x.T @ x + ridge_val * np.eye(2, dtype=np.float64)
            coef = np.linalg.solve(normal, x.T @ y)
            pred = x @ coef
            residual_chunks.append(y - pred)
            n_fit_samples = int(x.shape[0])
            matrices[candidate_index] = coef.T
        else:
            x_all = xy[candidate_index, keep].reshape(-1, 2)
            y_all = z[candidate_index, keep].reshape(-1, k_dim)
            good_all = np.isfinite(x_all).all(axis=1) & np.isfinite(y_all).all(axis=1)
            if int(np.sum(good_all)) < 2:
                raise ValueError(f"candidate {candidate_index} has too few finite trajectory samples to fit shrinkage A_I")
            x_all = x_all[good_all]
            y_all = y_all[good_all]
            normal_all = x_all.T @ x_all + ridge_val * np.eye(2, dtype=np.float64)
            const_coef = np.linalg.solve(normal_all, x_all.T @ y_all).T
            for time_index in range(n_time):
                x = xy[candidate_index, keep, time_index]
                y = z[candidate_index, keep, time_index]
                good = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
                if int(np.sum(good)) < 2:
                    raise ValueError(
                        f"candidate {candidate_index} time {time_index} has too few finite trajectory samples to fit A_I(t)"
                    )
                x = x[good]
                y = y[good]
                normal = x.T @ x + ridge_val * np.eye(2, dtype=np.float64)
                coef = np.linalg.solve(normal, x.T @ y)
                n_fit_samples += int(x.shape[0])
                matrices[candidate_index, time_index] = coef.T
            if smooth_sigma > 0.0:
                matrices[candidate_index] = _smooth_time_axis(matrices[candidate_index], smooth_sigma)
            if shrink > 0.0:
                matrices[candidate_index] = (1.0 - shrink) * matrices[candidate_index] + shrink * const_coef[None, :, :]
            residual_chunks.append(_time_varying_residuals(z[candidate_index], xy[candidate_index], keep, matrices[candidate_index]).reshape(-1, k_dim))
        resid = np.concatenate(residual_chunks, axis=0) if residual_chunks else np.empty((0, k_dim))
        residual_variance[candidate_index] = float(np.mean(resid * resid)) if resid.size else float("nan")
        condition_metrics = _observation_matrix_condition_metrics(matrices[candidate_index])
        fit_rows.append(
            {
                "candidate_index": int(candidate_index),
                "n_fit_trajectories": int(np.sum(keep)),
                "n_fit_samples": int(n_fit_samples),
                "heldout_trajectory_index": int(heldout),
                "excluded_heldout_trajectory": bool(excluded),
                "observation_model": str(model),
                "time_smoothing_sigma": float(smooth_sigma),
                "time_shrinkage": float(shrink),
                "ridge": float(ridge_val),
                "basis_dim": int(k_dim),
                "residual_variance": float(residual_variance[candidate_index]),
                "A_fro_norm": float(np.sqrt(np.sum(matrices[candidate_index] * matrices[candidate_index]))),
                **condition_metrics,
            }
        )
    return {
        "A_matrices": matrices,
        "basis": u,
        "projected_prior_delta": z,
        "residual_variance": residual_variance,
        "fit_rows": fit_rows,
        "observation_model": str(model),
        "time_smoothing_sigma": float(smooth_sigma),
        "time_shrinkage": float(shrink),
    }


def kalman_filter_log_likelihood(
    z_obs: np.ndarray,
    observation_matrix: np.ndarray,
    *,
    alpha: float = 0.92,
    process_var: float = 1e-3,
    observation_var: float | np.ndarray = 1.0,
    initial_mean: np.ndarray | None = None,
    initial_cov: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Return Kalman marginal log-likelihood and filtered 2D trajectory means."""
    z = np.asarray(z_obs, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"z_obs must be (time, k), got {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError("z_obs contains non-finite values")
    t_count, k_dim = z.shape
    h = np.asarray(observation_matrix, dtype=np.float64)
    if h.shape == (k_dim, 2):
        h_by_t = np.broadcast_to(h[None, :, :], (t_count, k_dim, 2))
    elif h.shape == (t_count, k_dim, 2):
        h_by_t = h
    else:
        raise ValueError(f"observation_matrix must be (k, 2) or (time, k, 2), got {h.shape}")
    a = float(alpha)
    q = float(process_var)
    if not np.isfinite(a) or abs(a) >= 1.0:
        raise ValueError("alpha must be finite with abs(alpha) < 1")
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError("process_var must be positive and finite")
    obs_var = np.asarray(observation_var, dtype=np.float64)
    if obs_var.ndim == 0:
        if float(obs_var) <= 0.0:
            raise ValueError("observation_var must be positive")
        r = np.eye(k_dim, dtype=np.float64) * float(obs_var)
    elif obs_var.shape == (k_dim,):
        if np.any(obs_var <= 0.0):
            raise ValueError("observation_var entries must be positive")
        r = np.diag(obs_var)
    elif obs_var.shape == (k_dim, k_dim):
        r = obs_var
    else:
        raise ValueError(f"observation_var must be scalar, (k,), or (k,k), got {obs_var.shape}")
    if not np.isfinite(r).all():
        raise ValueError("observation_var contains non-finite values")

    mean = np.zeros(2, dtype=np.float64) if initial_mean is None else np.asarray(initial_mean, dtype=np.float64)
    if mean.shape != (2,):
        raise ValueError(f"initial_mean must be (2,), got {mean.shape}")
    if initial_cov is None:
        cov_scale = q / max(1e-9, 1.0 - a * a)
        cov = np.eye(2, dtype=np.float64) * cov_scale
    else:
        cov = np.asarray(initial_cov, dtype=np.float64)
        if cov.shape != (2, 2):
            raise ValueError(f"initial_cov must be (2,2), got {cov.shape}")

    means = np.empty((t_count, 2), dtype=np.float64)
    covs = np.empty((t_count, 2, 2), dtype=np.float64)
    loglik = 0.0
    eye2 = np.eye(2, dtype=np.float64)
    for time_index in range(t_count):
        mean_pred = a * mean
        cov_pred = (a * a) * cov + q * eye2
        h_t = h_by_t[time_index]
        innov = z[time_index] - h_t @ mean_pred
        s_mat = h_t @ cov_pred @ h_t.T + r
        sign, logdet = np.linalg.slogdet(s_mat)
        if sign <= 0:
            raise ValueError("Kalman innovation covariance is not positive definite")
        solved = np.linalg.solve(s_mat, innov)
        loglik += -0.5 * (k_dim * np.log(2.0 * np.pi) + float(logdet) + float(innov @ solved))
        gain = cov_pred @ h_t.T @ np.linalg.inv(s_mat)
        mean = mean_pred + gain @ innov
        cov = (eye2 - gain @ h_t) @ cov_pred
        cov = 0.5 * (cov + cov.T)
        means[time_index] = mean
        covs[time_index] = cov
    return {
        "log_likelihood": float(loglik),
        "filtered_means": means,
        "filtered_covariances": covs,
    }


def ar1_profile_log_score(
    z_obs: np.ndarray,
    observation_matrix: np.ndarray,
    *,
    alpha: float = 0.92,
    process_var: float = 1e-3,
    process_cov: np.ndarray | None = None,
    observation_var: float | np.ndarray = 1.0,
    initial_var: float | None = None,
    initial_cov: np.ndarray | None = None,
    initial_mean: np.ndarray | None = None,
    prior_mean: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Return profile score after optimizing the full latent AR(1) path.

    This is a MAP/profile objective, not a normalized marginal likelihood.  It
    intentionally omits the candidate-dependent log-determinant/evidence term,
    which makes it a diagnostic for whether the local linear model contains a
    useful continuous analogue of the finite best-trajectory score.
    """
    z = np.asarray(z_obs, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"z_obs must be (time, k), got {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError("z_obs contains non-finite values")
    t_count, k_dim = z.shape
    h = np.asarray(observation_matrix, dtype=np.float64)
    if h.shape == (k_dim, 2):
        h_by_t = np.broadcast_to(h[None, :, :], (t_count, k_dim, 2))
    elif h.shape == (t_count, k_dim, 2):
        h_by_t = h
    else:
        raise ValueError(f"observation_matrix must be (k, 2) or (time, k, 2), got {h.shape}")
    a = float(alpha)
    if not np.isfinite(a) or abs(a) > 1.0:
        raise ValueError("alpha must be finite with abs(alpha) <= 1")
    q = float(process_var)
    if process_cov is None:
        if not np.isfinite(q) or q <= 0.0:
            raise ValueError("process_var must be positive and finite")
        q_cov = np.eye(2, dtype=np.float64) * q
    else:
        q_cov = np.asarray(process_cov, dtype=np.float64)
        if q_cov.shape != (2, 2):
            raise ValueError(f"process_cov must be (2,2), got {q_cov.shape}")
        q_cov = 0.5 * (q_cov + q_cov.T)
        if not np.isfinite(q_cov).all():
            raise ValueError("process_cov contains non-finite values")
        if np.min(np.linalg.eigvalsh(q_cov)) <= 0.0:
            raise ValueError("process_cov must be positive definite")
    q_inv_mat = np.linalg.inv(q_cov)
    obs_var = np.asarray(observation_var, dtype=np.float64)
    if obs_var.ndim == 0:
        if float(obs_var) <= 0.0:
            raise ValueError("observation_var must be positive")
        r_inv = np.eye(k_dim, dtype=np.float64) / float(obs_var)
    elif obs_var.shape == (k_dim,):
        if np.any(obs_var <= 0.0):
            raise ValueError("observation_var entries must be positive")
        r_inv = np.diag(1.0 / obs_var)
    elif obs_var.shape == (k_dim, k_dim):
        r_inv = np.linalg.inv(obs_var)
    else:
        raise ValueError(f"observation_var must be scalar, (k,), or (k,k), got {obs_var.shape}")
    if not np.isfinite(r_inv).all():
        raise ValueError("observation_var contains non-finite values")
    if prior_mean is None:
        mean_path = np.zeros((t_count, 2), dtype=np.float64)
    else:
        mean_path = np.asarray(prior_mean, dtype=np.float64)
        if mean_path.shape != (t_count, 2):
            raise ValueError(f"prior_mean must be (time, 2), got {mean_path.shape}")
        if not np.isfinite(mean_path).all():
            raise ValueError("prior_mean contains non-finite values")
    if initial_mean is None:
        initial_prior_mean = mean_path[0]
    else:
        initial_prior_mean = np.asarray(initial_mean, dtype=np.float64)
        if initial_prior_mean.shape != (2,):
            raise ValueError(f"initial_mean must be (2,), got {initial_prior_mean.shape}")
        if not np.isfinite(initial_prior_mean).all():
            raise ValueError("initial_mean contains non-finite values")

    state_dim = 2 * t_count
    precision = np.zeros((state_dim, state_dim), dtype=np.float64)
    linear = np.zeros(state_dim, dtype=np.float64)
    const = 0.0
    for time_index in range(t_count):
        sl = slice(2 * time_index, 2 * time_index + 2)
        h_t = h_by_t[time_index]
        precision[sl, sl] += h_t.T @ r_inv @ h_t
        linear[sl] += h_t.T @ r_inv @ z[time_index]
        const += float(z[time_index] @ r_inv @ z[time_index])

    if initial_cov is not None:
        p0_cov = np.asarray(initial_cov, dtype=np.float64)
        if p0_cov.shape != (2, 2):
            raise ValueError(f"initial_cov must be (2,2), got {p0_cov.shape}")
        p0_cov = 0.5 * (p0_cov + p0_cov.T)
        if not np.isfinite(p0_cov).all() or np.min(np.linalg.eigvalsh(p0_cov)) <= 0.0:
            raise ValueError("initial_cov must be positive definite")
    else:
        p0 = (
            float(initial_var)
            if initial_var is not None
            else q / max(1e-9, 1.0 - a * a)
        )
        if not np.isfinite(p0) or p0 <= 0.0:
            raise ValueError("initial_var must be positive and finite")
        p0_cov = np.eye(2, dtype=np.float64) * p0
    p0_inv = np.linalg.inv(p0_cov)
    eye2 = np.eye(2, dtype=np.float64)
    precision[0:2, 0:2] += p0_inv
    linear[0:2] += p0_inv @ initial_prior_mean
    const += float(initial_prior_mean @ p0_inv @ initial_prior_mean)
    for time_index in range(1, t_count):
        prev = slice(2 * (time_index - 1), 2 * (time_index - 1) + 2)
        cur = slice(2 * time_index, 2 * time_index + 2)
        prior_step_mean = mean_path[time_index] - a * mean_path[time_index - 1]
        precision[prev, prev] += (a * a) * q_inv_mat
        precision[cur, cur] += q_inv_mat
        precision[prev, cur] += -a * q_inv_mat
        precision[cur, prev] += -a * q_inv_mat
        linear[prev] += -a * (q_inv_mat @ prior_step_mean)
        linear[cur] += q_inv_mat @ prior_step_mean
        const += float(prior_step_mean @ q_inv_mat @ prior_step_mean)

    ridge = 1e-10 * np.eye(state_dim, dtype=np.float64)
    state = np.linalg.solve(precision + ridge, linear)
    energy = const - float(linear @ state)
    return {
        "profile_score": float(-0.5 * energy),
        "map_means": state.reshape(t_count, 2),
        "profile_energy": float(energy),
    }


def trajectory_recovery_metrics(tau_hat: np.ndarray, tau_true: np.ndarray) -> dict[str, float]:
    """Return trajectory recovery diagnostics for an estimated 2D path."""
    pred = np.asarray(tau_hat, dtype=np.float64)
    true = np.asarray(tau_true, dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError(f"trajectory arrays must both be (time, 2), got {pred.shape} and {true.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        return {
            "trajectory_rmse": float("nan"),
            "trajectory_corr_x": float("nan"),
            "trajectory_corr_y": float("nan"),
            "trajectory_corr_mean": float("nan"),
            "trajectory_r2": float("nan"),
            "trajectory_hat_rms": float("nan"),
            "trajectory_true_rms": float("nan"),
        }
    diff = pred - true
    rmse = float(np.sqrt(np.mean(diff * diff)))
    corr_vals = []
    for dim in range(2):
        if float(np.std(pred[:, dim])) <= 1e-12 or float(np.std(true[:, dim])) <= 1e-12:
            corr_vals.append(float("nan"))
        else:
            corr_vals.append(float(np.corrcoef(pred[:, dim], true[:, dim])[0, 1]))
    denom = float(np.sum((true - np.mean(true, axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - float(np.sum(diff * diff)) / denom if denom > 1e-12 else float("nan")
    return {
        "trajectory_rmse": rmse,
        "trajectory_corr_x": corr_vals[0],
        "trajectory_corr_y": corr_vals[1],
        "trajectory_corr_mean": float(np.nanmean(corr_vals)) if np.isfinite(corr_vals).any() else float("nan"),
        "trajectory_r2": float(r2),
        "trajectory_hat_rms": float(np.sqrt(np.mean(pred * pred))),
        "trajectory_true_rms": float(np.sqrt(np.mean(true * true))),
    }


def _compact_delta_from_path(tau_hat: np.ndarray, observation_matrix: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau_hat, dtype=np.float64)
    h = np.asarray(observation_matrix, dtype=np.float64)
    if tau.ndim != 2 or tau.shape[1] != 2:
        raise ValueError(f"tau_hat must be (time, 2), got {tau.shape}")
    if h.ndim == 2:
        if h.shape[1] != 2:
            raise ValueError(f"observation_matrix must end with displacement dim 2, got {h.shape}")
        return tau @ h.T
    if h.ndim == 3:
        if h.shape[0] != tau.shape[0] or h.shape[2] != 2:
            raise ValueError(f"time-varying observation_matrix shape {h.shape} is incompatible with tau {tau.shape}")
        return np.einsum("tkd,td->tk", h, tau)
    raise ValueError(f"observation_matrix must be (k, 2) or (time, k, 2), got {h.shape}")


def _quadratic_design_from_path(
    tau_hat: np.ndarray,
    *,
    quadratic_scale: float = 1.0,
    include_intercept: bool = False,
) -> np.ndarray:
    tau = np.asarray(tau_hat, dtype=np.float64)
    if tau.ndim != 2 or tau.shape[1] != 2:
        raise ValueError(f"tau_hat must be (time, 2), got {tau.shape}")
    x = tau[:, 0]
    y = tau[:, 1]
    q_scale = float(quadratic_scale)
    terms = []
    if include_intercept:
        terms.append(np.ones_like(x))
    terms.extend([x, y, q_scale * x * x, q_scale * x * y, q_scale * y * y])
    return np.stack(terms, axis=1)


def _quadratic_compact_delta_from_path(
    tau_hat: np.ndarray,
    observation_coefficients: np.ndarray,
    *,
    quadratic_scale: float = 1.0,
    include_intercept: bool | None = None,
    intercept_scale: float = 1.0,
) -> np.ndarray:
    coef = np.asarray(observation_coefficients, dtype=np.float64)
    if coef.ndim != 2 or coef.shape[1] not in {5, 6}:
        raise ValueError(f"quadratic observation coefficients must be (k,5) or (k,6), got {coef.shape}")
    use_intercept = bool(coef.shape[1] == 6) if include_intercept is None else bool(include_intercept)
    expected_dim = 6 if use_intercept else 5
    if coef.shape[1] != expected_dim:
        raise ValueError(f"coefficient shape {coef.shape} does not match include_intercept={use_intercept}")
    scale = float(intercept_scale)
    if not np.isfinite(scale):
        raise ValueError("intercept_scale must be finite")
    if use_intercept and scale != 1.0:
        coef = coef.copy()
        coef[:, 0] *= scale
    return (
        _quadratic_design_from_path(
            tau_hat,
            quadratic_scale=float(quadratic_scale),
            include_intercept=use_intercept,
        )
        @ coef.T
    )


def fit_quadratic_observation_maps(
    *,
    prior_lambda_counts: np.ndarray,
    zero_lambda_counts: np.ndarray,
    trajectory_xy: np.ndarray,
    basis: np.ndarray | None = None,
    heldout_trajectory_index: int | None = None,
    ridge: float = 1e-2,
    include_intercept: bool = False,
    intercept_ridge_multiplier: float = 1.0,
    intercept_strategy: str = "free",
) -> dict[str, Any]:
    """Fit compact quadratic ``z_t = B_I phi(tau_t) + eps_t`` maps."""
    prior = _as_prior_table(prior_lambda_counts, "prior_lambda_counts")
    n_candidates, n_traj, n_time, n_units = prior.shape
    zero = _as_candidate_table(zero_lambda_counts, "zero_lambda_counts", (n_candidates, n_time, n_units))
    u = _as_basis(basis, n_units)
    xy = _trajectory_xy_by_candidate(
        trajectory_xy,
        n_candidates=n_candidates,
        n_trajectories=n_traj,
        n_time=n_time,
    )
    ridge_val = float(ridge)
    if ridge_val < 0.0 or not np.isfinite(ridge_val):
        raise ValueError("ridge must be finite and non-negative")
    intercept_mult = float(intercept_ridge_multiplier)
    if not np.isfinite(intercept_mult) or intercept_mult <= 0.0:
        raise ValueError("intercept_ridge_multiplier must be positive and finite")
    strategy = str(intercept_strategy)
    if strategy not in {"none", "free", "prior_mean"}:
        raise ValueError("intercept_strategy must be 'none', 'free', or 'prior_mean'")
    if not bool(include_intercept):
        strategy = "none"
    z = project_response_delta(prior - zero[:, None, :, :], u)
    k_dim = int(u.shape[1])
    design_dim = 6 if bool(include_intercept) else 5
    coefficients = np.empty((n_candidates, k_dim, design_dim), dtype=np.float64)
    residual_variance = np.empty(n_candidates, dtype=np.float64)
    fit_rows: list[dict[str, Any]] = []
    heldout = -1 if heldout_trajectory_index is None else int(heldout_trajectory_index)

    for candidate_index in range(n_candidates):
        keep = np.ones(n_traj, dtype=bool)
        excluded = False
        if 0 <= heldout < n_traj and n_traj > 1:
            keep[heldout] = False
            excluded = True
        if not bool(np.any(keep)):
            raise ValueError("heldout_trajectory_index excludes all trajectories")
        design = np.stack(
            [
                _quadratic_design_from_path(
                    xy[candidate_index, trajectory_index],
                    include_intercept=bool(include_intercept),
                )
                for trajectory_index in np.flatnonzero(keep)
            ],
            axis=0,
        )
        x = design.reshape(-1, design_dim)
        y = z[candidate_index, keep].reshape(-1, k_dim)
        good = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        if int(np.sum(good)) < design_dim:
            raise ValueError(f"candidate {candidate_index} has too few finite trajectory samples to fit B_I")
        x = x[good]
        y = y[good]
        if strategy == "prior_mean":
            x_dyn = x[:, 1:]
            phi_mean = np.mean(x_dyn, axis=0)
            z_mean = np.mean(y, axis=0)
            x_centered = x_dyn - phi_mean[None, :]
            y_centered = y - z_mean[None, :]
            ridge_diag = ridge_val * np.ones(5, dtype=np.float64)
            normal = x_centered.T @ x_centered + np.diag(ridge_diag)
            coef_dyn = np.linalg.solve(normal, x_centered.T @ y_centered)
            coef = np.empty((design_dim, k_dim), dtype=np.float64)
            coef[1:] = coef_dyn
            coef[0] = z_mean - phi_mean @ coef_dyn
            pred = z_mean[None, :] + x_centered @ coef_dyn
        else:
            ridge_diag = ridge_val * np.ones(design_dim, dtype=np.float64)
            if include_intercept:
                ridge_diag[0] *= intercept_mult
            normal = x.T @ x + np.diag(ridge_diag)
            coef = np.linalg.solve(normal, x.T @ y)
            pred = x @ coef
        resid = y - pred
        coefficients[candidate_index] = coef.T
        residual_variance[candidate_index] = float(np.mean(resid * resid))
        candidate_coef = coefficients[candidate_index]
        singular = np.linalg.svd(candidate_coef, compute_uv=False)
        total_norm = float(np.sqrt(np.sum(candidate_coef * candidate_coef)))
        if include_intercept:
            intercept_norm = float(np.linalg.norm(candidate_coef[:, 0]))
            dynamic_norm = float(np.linalg.norm(candidate_coef[:, 1:]))
        else:
            intercept_norm = 0.0
            dynamic_norm = total_norm
        fit_rows.append(
            {
                "candidate_index": int(candidate_index),
                "n_fit_trajectories": int(np.sum(keep)),
                "n_fit_samples": int(x.shape[0]),
                "heldout_trajectory_index": int(heldout),
                "excluded_heldout_trajectory": bool(excluded),
                "observation_model": (
                    "quadratic_prior_mean_affine_time_constant"
                    if strategy == "prior_mean"
                    else "quadratic_affine_time_constant"
                    if include_intercept
                    else "quadratic_time_constant"
                ),
                "quadratic_include_intercept": bool(include_intercept),
                "quadratic_intercept_strategy": str(strategy),
                "quadratic_intercept_ridge_multiplier": float(intercept_mult),
                "ridge": float(ridge_val),
                "basis_dim": int(k_dim),
                "residual_variance": float(residual_variance[candidate_index]),
                "B_fro_norm": total_norm,
                "B_intercept_norm": intercept_norm,
                "B_dynamic_fro_norm": dynamic_norm,
                "B_intercept_norm_fraction": float(intercept_norm / max(total_norm, 1e-12)),
                "B_train_r2_energy": float(1.0 - np.sum(resid * resid) / max(float(np.sum(y * y)), 1e-12)),
                "B_singular1": float(singular[0]) if singular.size > 0 else float("nan"),
                "B_singular2": float(singular[1]) if singular.size > 1 else float("nan"),
                "B_singular3": float(singular[2]) if singular.size > 2 else float("nan"),
            }
        )
    return {
        "B_coefficients": coefficients,
        "basis": u,
        "projected_prior_delta": z,
        "residual_variance": residual_variance,
        "fit_rows": fit_rows,
        "observation_model": (
            "quadratic_prior_mean_affine_time_constant"
            if strategy == "prior_mean"
            else "quadratic_affine_time_constant"
            if include_intercept
            else "quadratic_time_constant"
        ),
    }


def _quadratic_profile_objective_and_grad(
    flat_path: np.ndarray,
    z_obs: np.ndarray,
    observation_coefficients: np.ndarray,
    *,
    quadratic_scale: float,
    observation_var: float,
    alpha: float,
    process_var: float,
    process_cov: np.ndarray | None = None,
    initial_mean: np.ndarray | None,
    initial_var: float,
    initial_cov: np.ndarray | None = None,
    intercept_scale: float = 1.0,
) -> tuple[float, np.ndarray]:
    tau = np.asarray(flat_path, dtype=np.float64).reshape(-1, 2)
    z = np.asarray(z_obs, dtype=np.float64)
    coef = np.asarray(observation_coefficients, dtype=np.float64)
    include_intercept = bool(coef.shape[1] == 6)
    offset = 1 if include_intercept else 0
    x = tau[:, 0]
    y = tau[:, 1]
    q_scale = float(quadratic_scale)
    pred = _quadratic_compact_delta_from_path(tau, coef, quadratic_scale=q_scale, intercept_scale=float(intercept_scale))
    err = pred - z
    r = max(float(observation_var), 1e-12)
    value = 0.5 * float(np.sum(err * err) / r)

    d_pred_dx = coef[:, offset + 0][None, :] + q_scale * (
        2.0 * x[:, None] * coef[:, offset + 2][None, :] + y[:, None] * coef[:, offset + 3][None, :]
    )
    d_pred_dy = coef[:, offset + 1][None, :] + q_scale * (
        x[:, None] * coef[:, offset + 3][None, :] + 2.0 * y[:, None] * coef[:, offset + 4][None, :]
    )
    grad = np.empty_like(tau)
    grad[:, 0] = np.sum(err * d_pred_dx, axis=1) / r
    grad[:, 1] = np.sum(err * d_pred_dy, axis=1) / r

    initial_target = np.zeros(2, dtype=np.float64) if initial_mean is None else np.asarray(initial_mean, dtype=np.float64)
    diff0 = tau[0] - initial_target
    if initial_cov is None:
        p0 = max(float(initial_var), 1e-12)
        p0_inv = np.eye(2, dtype=np.float64) / p0
    else:
        p0_cov = np.asarray(initial_cov, dtype=np.float64)
        if p0_cov.shape != (2, 2):
            raise ValueError(f"initial_cov must be (2,2), got {p0_cov.shape}")
        p0_cov = 0.5 * (p0_cov + p0_cov.T)
        if not np.isfinite(p0_cov).all() or np.min(np.linalg.eigvalsh(p0_cov)) <= 0.0:
            raise ValueError("initial_cov must be positive definite")
        p0_inv = np.linalg.inv(p0_cov)
    value += 0.5 * float(diff0 @ p0_inv @ diff0)
    grad[0] += p0_inv @ diff0

    if process_cov is None:
        q = max(float(process_var), 1e-12)
        q_inv = np.eye(2, dtype=np.float64) / q
    else:
        q_cov = np.asarray(process_cov, dtype=np.float64)
        if q_cov.shape != (2, 2):
            raise ValueError(f"process_cov must be (2,2), got {q_cov.shape}")
        q_cov = 0.5 * (q_cov + q_cov.T)
        if not np.isfinite(q_cov).all():
            raise ValueError("process_cov contains non-finite values")
        if np.min(np.linalg.eigvalsh(q_cov)) <= 0.0:
            raise ValueError("process_cov must be positive definite")
        q_inv = np.linalg.inv(q_cov)
    a = float(alpha)
    if tau.shape[0] > 1:
        step = tau[1:] - a * tau[:-1]
        q_step = step @ q_inv
        value += 0.5 * float(np.sum(step * q_step))
        grad[1:] += q_step
        grad[:-1] += -a * q_step
    return value, grad.reshape(-1)


def quadratic_profile_log_score(
    z_obs: np.ndarray,
    observation_coefficients: np.ndarray,
    *,
    starts: list[np.ndarray],
    observation_var: float = 1.0,
    alpha: float = 0.92,
    process_var: float = 1e-3,
    process_cov: np.ndarray | None = None,
    initial_mean: np.ndarray | None = None,
    initial_var: float = 1e-4,
    initial_cov: np.ndarray | None = None,
    max_iter: int = 80,
    quadratic_scales: list[float] | None = None,
    observation_scales: list[float] | None = None,
    intercept_scale: float = 1.0,
) -> dict[str, Any]:
    """Profile a 2D path under a quadratic compact observation map."""
    q_scales = [1.0] if quadratic_scales is None or not quadratic_scales else [float(v) for v in quadratic_scales]
    obs_scales = [1.0] if observation_scales is None or not observation_scales else [float(v) for v in observation_scales]
    if len(obs_scales) == 1 and len(q_scales) > 1:
        obs_scales = obs_scales * len(q_scales)
    if len(q_scales) != len(obs_scales):
        raise ValueError("quadratic_scales and observation_scales must have equal length, or one observation scale")
    best: dict[str, Any] | None = None
    for start_index, start in enumerate(starts):
        start_shape = np.asarray(start, dtype=np.float64).shape
        current = np.asarray(start, dtype=np.float64).reshape(-1)
        stage_results = []
        for q_scale, obs_scale in zip(q_scales, obs_scales):
            result = minimize(
                lambda flat: _quadratic_profile_objective_and_grad(
                    flat,
                    z_obs,
                    observation_coefficients,
                    quadratic_scale=float(q_scale),
                    observation_var=float(observation_var) * float(obs_scale),
                    alpha=float(alpha),
                    process_var=float(process_var),
                    process_cov=process_cov,
                    initial_mean=initial_mean,
                    initial_var=float(initial_var),
                    initial_cov=initial_cov,
                    intercept_scale=float(intercept_scale),
                ),
                current,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": int(max_iter), "ftol": 1e-8, "gtol": 1e-5, "maxls": 30},
            )
            current = np.asarray(result.x, dtype=np.float64)
            stage_results.append(result)
        energy, _grad = _quadratic_profile_objective_and_grad(
            current,
            z_obs,
            observation_coefficients,
            quadratic_scale=1.0,
            observation_var=float(observation_var),
            alpha=float(alpha),
            process_var=float(process_var),
            process_cov=process_cov,
            initial_mean=initial_mean,
            initial_var=float(initial_var),
            initial_cov=initial_cov,
            intercept_scale=float(intercept_scale),
        )
        final_result = stage_results[-1]
        if best is None or float(energy) < float(best["profile_energy"]):
            best = {
                "profile_score": -float(energy),
                "map_means": current.reshape(start_shape),
                "profile_energy": float(energy),
                "optimizer_success": bool(final_result.success),
                "optimizer_iterations": int(sum(int(result.nit) for result in stage_results)),
                "optimizer_final_iterations": int(final_result.nit),
                "optimizer_stage_success_fraction": float(np.mean([bool(result.success) for result in stage_results])),
                "optimizer_start_index": int(start_index),
                "optimizer_message": str(final_result.message),
            }
    if best is None:
        raise RuntimeError("quadratic profile optimizer did not run")
    return best


def trajectory_basis_profile_log_score(
    z_obs: np.ndarray,
    observation_matrix: np.ndarray,
    trajectory_samples: np.ndarray,
    *,
    n_components: int = 4,
    coeff_prior_var: float = 1.0,
    observation_var: float | np.ndarray = 1.0,
    initial_mean: np.ndarray | None = None,
    initial_cov: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Infer a continuous path in the PCA subspace of the trajectory catalog."""
    z = np.asarray(z_obs, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"z_obs must be (time, k), got {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError("z_obs contains non-finite values")
    t_count, k_dim = z.shape
    h = np.asarray(observation_matrix, dtype=np.float64)
    if h.shape == (k_dim, 2):
        h_by_t = np.broadcast_to(h[None, :, :], (t_count, k_dim, 2))
    elif h.shape == (t_count, k_dim, 2):
        h_by_t = h
    else:
        raise ValueError(f"observation_matrix must be (k, 2) or (time, k, 2), got {h.shape}")
    samples = np.asarray(trajectory_samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[1:] != (t_count, 2):
        raise ValueError(f"trajectory_samples must be (n, time, 2), got {samples.shape}")
    if not np.isfinite(samples).all():
        raise ValueError("trajectory_samples contains non-finite values")
    coeff_var = float(coeff_prior_var)
    if not np.isfinite(coeff_var) or coeff_var <= 0.0:
        raise ValueError("coeff_prior_var must be positive and finite")

    mean_path = np.mean(samples, axis=0)
    centered = samples - mean_path[None, :, :]
    flat = centered.reshape(samples.shape[0], t_count * 2)
    max_components = min(int(n_components), flat.shape[0], flat.shape[1])
    if max_components <= 0 or float(np.max(np.std(flat, axis=0))) <= 1e-12:
        basis_paths = np.empty((0, t_count, 2), dtype=np.float64)
    else:
        _u_svd, _s_svd, vt = np.linalg.svd(flat, full_matrices=False)
        basis_paths = vt[:max_components].reshape(max_components, t_count, 2)

    mean_z = _compact_delta_from_path(mean_path, h_by_t)
    y = z - mean_z
    n_basis = int(basis_paths.shape[0])
    if n_basis == 0:
        tau_hat = mean_path
        compact_delta = mean_z
        obs_energy = _weighted_quadratic(y, observation_var)
        return {
            "profile_score": float(-0.5 * obs_energy),
            "map_means": tau_hat,
            "coefficients": np.empty(0, dtype=np.float64),
            "trajectory_basis_dim": 0,
        }

    design = np.empty((t_count, k_dim, n_basis), dtype=np.float64)
    for basis_index in range(n_basis):
        design[:, :, basis_index] = _compact_delta_from_path(basis_paths[basis_index], h_by_t)
    precision = np.eye(n_basis, dtype=np.float64) / coeff_var
    linear = np.zeros(n_basis, dtype=np.float64)
    const = _weighted_quadratic(y, observation_var)
    obs_var = np.asarray(observation_var, dtype=np.float64)
    if obs_var.ndim == 0:
        weight = 1.0 / float(obs_var)
        flat_design = design.reshape(t_count * k_dim, n_basis)
        flat_y = y.reshape(t_count * k_dim)
        precision += weight * (flat_design.T @ flat_design)
        linear += weight * (flat_design.T @ flat_y)
    elif obs_var.shape == (k_dim,):
        weights = 1.0 / obs_var
        for time_index in range(t_count):
            d_t = design[time_index]
            precision += d_t.T @ (weights[:, None] * d_t)
            linear += d_t.T @ (weights * y[time_index])
    elif obs_var.shape == (k_dim, k_dim):
        r_inv = np.linalg.inv(obs_var)
        for time_index in range(t_count):
            d_t = design[time_index]
            precision += d_t.T @ r_inv @ d_t
            linear += d_t.T @ r_inv @ y[time_index]
    else:
        raise ValueError(f"observation_var must be scalar, (k,), or (k,k), got {obs_var.shape}")
    if initial_mean is not None:
        initial_target = np.asarray(initial_mean, dtype=np.float64) - mean_path[0]
        precision, linear, const = _add_initial_basis_prior(
            precision,
            linear,
            const,
            basis_paths,
            initial_target,
            initial_cov,
        )
    coeffs = np.linalg.solve(precision + 1e-10 * np.eye(n_basis, dtype=np.float64), linear)
    tau_hat = mean_path + np.einsum("b,btd->td", coeffs, basis_paths)
    energy = const - float(linear @ coeffs)
    return {
        "profile_score": float(-0.5 * energy),
        "map_means": tau_hat,
        "coefficients": coeffs,
        "trajectory_basis_dim": n_basis,
    }


def _add_initial_basis_prior(
    precision: np.ndarray,
    linear: np.ndarray,
    const: float,
    basis_paths: np.ndarray,
    initial_target: np.ndarray,
    initial_cov: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    target = np.asarray(initial_target, dtype=np.float64)
    if target.shape != (2,):
        raise ValueError(f"initial_mean must be (2,), got {target.shape}")
    if not np.isfinite(target).all():
        raise ValueError("initial_mean contains non-finite values")
    if initial_cov is None:
        cov = np.eye(2, dtype=np.float64) * 1e-4
    else:
        cov = np.asarray(initial_cov, dtype=np.float64)
        if cov.shape != (2, 2):
            raise ValueError(f"initial_cov must be (2,2), got {cov.shape}")
    cov = 0.5 * (cov + cov.T)
    if not np.isfinite(cov).all() or np.min(np.linalg.eigvalsh(cov)) <= 0.0:
        raise ValueError("initial_cov must be positive definite")
    weight = np.linalg.inv(cov)
    start_design = np.asarray(basis_paths[:, 0, :], dtype=np.float64).T
    precision += start_design.T @ weight @ start_design
    linear += start_design.T @ weight @ target
    const += float(target @ weight @ target)
    return precision, linear, const


def _dct_trajectory_basis_paths(t_count: int, n_components: int) -> np.ndarray:
    n_time = int(t_count)
    n_freq = max(0, min(int(n_components), n_time))
    if n_freq <= 0:
        return np.empty((0, n_time, 2), dtype=np.float64)
    time = np.arange(n_time, dtype=np.float64)
    basis_1d = []
    for freq in range(n_freq):
        if freq == 0:
            phi = np.ones(n_time, dtype=np.float64) / np.sqrt(float(n_time))
        else:
            phi = np.sqrt(2.0 / float(n_time)) * np.cos(np.pi * (time + 0.5) * float(freq) / float(n_time))
        basis_1d.append(phi)
    paths = np.zeros((2 * n_freq, n_time, 2), dtype=np.float64)
    for freq, phi in enumerate(basis_1d):
        paths[2 * freq, :, 0] = phi
        paths[2 * freq + 1, :, 1] = phi
    return paths


def temporal_basis_profile_log_score(
    z_obs: np.ndarray,
    observation_matrix: np.ndarray,
    *,
    n_components: int = 4,
    coeff_prior_var: float = 1.0,
    observation_var: float | np.ndarray = 1.0,
    initial_mean: np.ndarray | None = None,
    initial_cov: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Infer a coarse continuous path in a low-frequency DCT basis.

    This is intentionally anchor-free: the basis contains only generic temporal
    modes for x/y displacement, not catalog trajectory exemplars.
    """
    z = np.asarray(z_obs, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"z_obs must be (time, k), got {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError("z_obs contains non-finite values")
    t_count, k_dim = z.shape
    h = np.asarray(observation_matrix, dtype=np.float64)
    if h.shape == (k_dim, 2):
        h_by_t = np.broadcast_to(h[None, :, :], (t_count, k_dim, 2))
    elif h.shape == (t_count, k_dim, 2):
        h_by_t = h
    else:
        raise ValueError(f"observation_matrix must be (k, 2) or (time, k, 2), got {h.shape}")
    coeff_var = float(coeff_prior_var)
    if not np.isfinite(coeff_var) or coeff_var <= 0.0:
        raise ValueError("coeff_prior_var must be positive and finite")

    basis_paths = _dct_trajectory_basis_paths(t_count, int(n_components))
    n_basis = int(basis_paths.shape[0])
    if n_basis == 0:
        return {
            "profile_score": float(-0.5 * _weighted_quadratic(z, observation_var)),
            "map_means": np.zeros((t_count, 2), dtype=np.float64),
            "coefficients": np.empty(0, dtype=np.float64),
            "trajectory_basis_dim": 0,
        }

    design = np.empty((t_count, k_dim, n_basis), dtype=np.float64)
    for basis_index in range(n_basis):
        design[:, :, basis_index] = _compact_delta_from_path(basis_paths[basis_index], h_by_t)
    precision = np.eye(n_basis, dtype=np.float64) / coeff_var
    linear = np.zeros(n_basis, dtype=np.float64)
    const = _weighted_quadratic(z, observation_var)
    obs_var = np.asarray(observation_var, dtype=np.float64)
    if obs_var.ndim == 0:
        weight = 1.0 / float(obs_var)
        flat_design = design.reshape(t_count * k_dim, n_basis)
        flat_y = z.reshape(t_count * k_dim)
        precision += weight * (flat_design.T @ flat_design)
        linear += weight * (flat_design.T @ flat_y)
    elif obs_var.shape == (k_dim,):
        weights = 1.0 / obs_var
        for time_index in range(t_count):
            d_t = design[time_index]
            precision += d_t.T @ (weights[:, None] * d_t)
            linear += d_t.T @ (weights * z[time_index])
    elif obs_var.shape == (k_dim, k_dim):
        r_inv = np.linalg.inv(obs_var)
        for time_index in range(t_count):
            d_t = design[time_index]
            precision += d_t.T @ r_inv @ d_t
            linear += d_t.T @ r_inv @ z[time_index]
    else:
        raise ValueError(f"observation_var must be scalar, (k,), or (k,k), got {obs_var.shape}")
    if initial_mean is not None:
        precision, linear, const = _add_initial_basis_prior(
            precision,
            linear,
            const,
            basis_paths,
            np.asarray(initial_mean, dtype=np.float64),
            initial_cov,
        )
    coeffs = np.linalg.solve(precision + 1e-10 * np.eye(n_basis, dtype=np.float64), linear)
    tau_hat = np.einsum("b,btd->td", coeffs, basis_paths)
    energy = const - float(linear @ coeffs)
    return {
        "profile_score": float(-0.5 * energy),
        "map_means": tau_hat,
        "coefficients": coeffs,
        "trajectory_basis_dim": n_basis,
    }


def _weighted_quadratic(values: np.ndarray, observation_var: float | np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    obs_var = np.asarray(observation_var, dtype=np.float64)
    if obs_var.ndim == 0:
        return float(np.sum(vals * vals) / float(obs_var))
    if obs_var.shape == (vals.shape[1],):
        return float(np.sum((vals * vals) / obs_var[None, :]))
    if obs_var.shape == (vals.shape[1], vals.shape[1]):
        r_inv = np.linalg.inv(obs_var)
        return float(sum(row @ r_inv @ row for row in vals))
    raise ValueError(f"observation_var must be scalar, (k,), or (k,k), got {obs_var.shape}")


def _matched_brownian_covariances(trajectory_samples: np.ndarray, *, floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(trajectory_samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[2] != 2:
        raise ValueError(f"trajectory_samples must be (n, time, 2), got {samples.shape}")
    if samples.shape[0] < 2 or samples.shape[1] < 2:
        eye = np.eye(2, dtype=np.float64) * max(float(floor), 1e-6)
        return eye.copy(), eye.copy()
    increments = np.diff(samples, axis=1).reshape(-1, 2)
    starts = samples[:, 0, :]
    process_cov = np.cov(increments, rowvar=False)
    initial_cov = np.cov(starts, rowvar=False)
    if process_cov.shape == ():
        process_cov = np.eye(2, dtype=np.float64) * float(process_cov)
    if initial_cov.shape == ():
        initial_cov = np.eye(2, dtype=np.float64) * float(initial_cov)
    process_cov = np.asarray(process_cov, dtype=np.float64).reshape(2, 2)
    initial_cov = np.asarray(initial_cov, dtype=np.float64).reshape(2, 2)
    process_cov = 0.5 * (process_cov + process_cov.T)
    initial_cov = 0.5 * (initial_cov + initial_cov.T)
    floor_val = max(float(floor), 1e-12)
    process_cov += np.eye(2, dtype=np.float64) * floor_val
    initial_cov += np.eye(2, dtype=np.float64) * floor_val
    return process_cov, initial_cov


def _trajectory_sample_covariance(
    samples: np.ndarray,
    *,
    floor: float = 1e-6,
    shrinkage: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"trajectory_samples must be (n, time, 2), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("trajectory_samples contains non-finite values")
    n_samples, t_count, _dim = arr.shape
    flat = arr.reshape(n_samples, t_count * 2)
    mean = np.mean(flat, axis=0)
    if n_samples < 2:
        centered_var = np.full(flat.shape[1], max(float(floor), 1e-6), dtype=np.float64)
        cov = np.diag(centered_var)
    else:
        cov = np.cov(flat, rowvar=False)
        cov = np.asarray(cov, dtype=np.float64).reshape(flat.shape[1], flat.shape[1])
    cov = 0.5 * (cov + cov.T)
    shrink = float(shrinkage)
    if not np.isfinite(shrink) or not 0.0 <= shrink <= 1.0:
        raise ValueError("catalog_gaussian_shrinkage must be finite and in [0, 1]")
    if shrink > 0.0:
        diag = np.maximum(np.diag(cov), 0.0)
        cov = (1.0 - shrink) * cov + shrink * np.diag(diag)
    floor_val = max(float(floor), 1e-12)
    if not np.isfinite(floor_val) or floor_val <= 0.0:
        raise ValueError("catalog_gaussian_cov_floor must be positive and finite")
    cov += np.eye(cov.shape[0], dtype=np.float64) * floor_val
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, floor_val)
    cov = (eigvecs * eigvals[None, :]) @ eigvecs.T
    cov = 0.5 * (cov + cov.T)
    return mean.reshape(t_count, 2), cov


def _add_path_initial_prior(
    precision: np.ndarray,
    linear: np.ndarray,
    const: float,
    initial_mean: np.ndarray,
    initial_cov: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    target = np.asarray(initial_mean, dtype=np.float64)
    if target.shape != (2,):
        raise ValueError(f"initial_mean must be (2,), got {target.shape}")
    if not np.isfinite(target).all():
        raise ValueError("initial_mean contains non-finite values")
    if initial_cov is None:
        cov = np.eye(2, dtype=np.float64) * 1e-4
    else:
        cov = np.asarray(initial_cov, dtype=np.float64)
        if cov.shape != (2, 2):
            raise ValueError(f"initial_cov must be (2,2), got {cov.shape}")
    cov = 0.5 * (cov + cov.T)
    if not np.isfinite(cov).all() or np.min(np.linalg.eigvalsh(cov)) <= 0.0:
        raise ValueError("initial_cov must be positive definite")
    weight = np.linalg.inv(cov)
    precision[0:2, 0:2] += weight
    linear[0:2] += weight @ target
    const += float(target @ weight @ target)
    return precision, linear, const


def catalog_gaussian_profile_log_score(
    z_obs: np.ndarray,
    observation_matrix: np.ndarray,
    trajectory_samples: np.ndarray,
    *,
    observation_var: float | np.ndarray = 1.0,
    smoothing_sigma: float = 0.0,
    cov_floor: float = 1e-6,
    shrinkage: float = 0.25,
    initial_mean: np.ndarray | None = None,
    initial_cov: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Profile the latent path under an empirical no-anchor Gaussian prior."""
    z = np.asarray(z_obs, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"z_obs must be (time, k), got {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError("z_obs contains non-finite values")
    t_count, k_dim = z.shape
    h = np.asarray(observation_matrix, dtype=np.float64)
    if h.shape == (k_dim, 2):
        h_by_t = np.broadcast_to(h[None, :, :], (t_count, k_dim, 2))
    elif h.shape == (t_count, k_dim, 2):
        h_by_t = h
    else:
        raise ValueError(f"observation_matrix must be (k, 2) or (time, k, 2), got {h.shape}")

    samples = _smooth_trajectory_samples(trajectory_samples, float(smoothing_sigma))
    if samples.shape[1:] != (t_count, 2):
        raise ValueError(f"trajectory_samples must have time/displacement shape {(t_count, 2)}, got {samples.shape}")
    mean_path, cov = _trajectory_sample_covariance(samples, floor=float(cov_floor), shrinkage=float(shrinkage))
    state_dim = 2 * t_count
    prior_precision = np.linalg.inv(cov)
    prior_mean = mean_path.reshape(state_dim)
    precision = prior_precision.copy()
    linear = prior_precision @ prior_mean
    const = float(prior_mean @ prior_precision @ prior_mean)

    obs_var = np.asarray(observation_var, dtype=np.float64)
    if obs_var.ndim == 0:
        if float(obs_var) <= 0.0:
            raise ValueError("observation_var must be positive")
        r_inv = np.eye(k_dim, dtype=np.float64) / float(obs_var)
    elif obs_var.shape == (k_dim,):
        if np.any(obs_var <= 0.0):
            raise ValueError("observation_var entries must be positive")
        r_inv = np.diag(1.0 / obs_var)
    elif obs_var.shape == (k_dim, k_dim):
        r_inv = np.linalg.inv(obs_var)
    else:
        raise ValueError(f"observation_var must be scalar, (k,), or (k,k), got {obs_var.shape}")
    if not np.isfinite(r_inv).all():
        raise ValueError("observation_var contains non-finite values")

    const += _weighted_quadratic(z, observation_var)
    for time_index in range(t_count):
        sl = slice(2 * time_index, 2 * time_index + 2)
        h_t = h_by_t[time_index]
        precision[sl, sl] += h_t.T @ r_inv @ h_t
        linear[sl] += h_t.T @ r_inv @ z[time_index]

    if initial_mean is not None:
        precision, linear, const = _add_path_initial_prior(precision, linear, const, initial_mean, initial_cov)

    state = np.linalg.solve(precision + 1e-10 * np.eye(state_dim, dtype=np.float64), linear)
    energy = const - float(linear @ state)
    return {
        "profile_score": float(-0.5 * energy),
        "map_means": state.reshape(t_count, 2),
        "profile_energy": float(energy),
        "catalog_gaussian_prior_mean": mean_path,
    }


def _smooth_trajectory_samples(samples: np.ndarray, sigma: float) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"trajectory samples must be (n, time, 2), got {arr.shape}")
    sig = float(sigma)
    if sig <= 0.0:
        return arr.copy()
    if not np.isfinite(sig):
        raise ValueError("trajectory smoothing sigma must be finite and non-negative")
    out = np.empty_like(arr)
    for index in range(arr.shape[0]):
        out[index] = _smooth_time_axis(arr[index], sig)
    return out


def _heldout_catalog_samples(samples: np.ndarray, heldout_index: int | None) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    heldout = -1 if heldout_index is None else int(heldout_index)
    if 0 <= heldout < arr.shape[0] and arr.shape[0] > 1:
        return np.delete(arr, heldout, axis=0)
    return arr


def score_continuous_joint_score_vectors(
    *,
    y_obs_counts: np.ndarray,
    prior_lambda_counts: np.ndarray,
    known_lambda_counts: np.ndarray,
    zero_lambda_counts: np.ndarray,
    trajectory_xy: np.ndarray,
    fit_prior_lambda_counts: np.ndarray | None = None,
    fit_trajectory_xy: np.ndarray | None = None,
    true_candidate_index: int,
    candidate_ids: list[str] | None = None,
    basis: np.ndarray | None = None,
    true_trajectory_index: int | None = None,
    observed_trajectory_xy: np.ndarray | None = None,
    alpha: float = 0.92,
    process_var: float = 1e-3,
    observation_var: float | None = None,
    observation_var_floor: float = 1e-6,
    ridge: float = 1e-6,
    observation_model: str = "time_constant",
    time_smoothing_sigma: float = 0.0,
    time_shrinkage: float = 0.0,
    continuous_score_mode: str = "kalman_marginal",
    trajectory_prior_mean: str = "zero",
    trajectory_initial_position: str = "inferred",
    trajectory_initial_position_var: float = 1e-4,
    trajectory_process_model: str = "ar1",
    brownian_cov_floor: float = 1e-6,
    brownian_cov_scale: float = 1.0,
    catalog_gaussian_smoothing_sigma: float = 0.0,
    catalog_gaussian_cov_floor: float = 1e-6,
    catalog_gaussian_shrinkage: float = 0.25,
    trajectory_basis_family: str = "dct",
    trajectory_basis_components: int = 4,
    trajectory_basis_smoothing_sigma: float = 6.0,
    trajectory_basis_coeff_prior_var: float = 1.0,
    catalog_residual_aggregation: str = "logmeanexp",
    catalog_residual_top_k: int = 0,
    catalog_residual_all_anchor_shrinkage: float = 0.0,
    catalog_residual_anchor_smoothing_sigma: float = 0.0,
    catalog_residual_anchor_smoothing_schedule: str | None = None,
    catalog_residual_refine_top_k: int = 0,
    quadratic_optimizer_max_iter: int = 80,
    quadratic_continuation_scales: str | None = "1",
    quadratic_observation_scales: str | None = "1",
    quadratic_intercept_ridge_multiplier: float = 1.0,
    quadratic_affine_intercept_scale: float = 1.0,
    eps: float = 1e-8,
    likelihood_scale: float = 1.0,
) -> dict[str, Any]:
    """Score image identity with the continuous Kalman joint estimator."""
    prior = _as_prior_table(prior_lambda_counts, "prior_lambda_counts")
    n_candidates, n_traj, n_time, n_units = prior.shape
    known = _as_candidate_table(known_lambda_counts, "known_lambda_counts", (n_candidates, n_time, n_units))
    zero = _as_candidate_table(zero_lambda_counts, "zero_lambda_counts", (n_candidates, n_time, n_units))
    obs = np.asarray(y_obs_counts, dtype=np.float64)
    if obs.shape != (n_time, n_units):
        raise ValueError(f"y_obs_counts shape {obs.shape} does not match expected {(n_time, n_units)}")
    true_idx = int(true_candidate_index)
    if true_idx < 0 or true_idx >= n_candidates:
        raise ValueError(f"true_candidate_index {true_idx} outside candidate table size {n_candidates}")
    ids = [str(i) for i in range(n_candidates)] if candidate_ids is None else [str(v) for v in candidate_ids]
    if len(ids) != n_candidates:
        raise ValueError(f"candidate_ids length {len(ids)} does not match n_candidates={n_candidates}")

    fit_prior = prior if fit_prior_lambda_counts is None else _as_prior_table(fit_prior_lambda_counts, "fit_prior_lambda_counts")
    if fit_prior.shape[0] != n_candidates or fit_prior.shape[2:] != (n_time, n_units):
        raise ValueError(
            "fit_prior_lambda_counts must match candidate/time/unit dimensions "
            f"{(n_candidates, n_time, n_units)}, got {fit_prior.shape}"
        )
    fit_xy_source = trajectory_xy if fit_trajectory_xy is None else fit_trajectory_xy
    fit_xy = _trajectory_xy_by_candidate(
        fit_xy_source,
        n_candidates=n_candidates,
        n_trajectories=int(fit_prior.shape[1]),
        n_time=n_time,
    )
    fit_heldout = true_trajectory_index if fit_prior_lambda_counts is None and fit_trajectory_xy is None else -1
    fit = fit_time_constant_observation_matrices(
        prior_lambda_counts=fit_prior,
        zero_lambda_counts=zero,
        trajectory_xy=fit_xy,
        basis=basis,
        heldout_trajectory_index=fit_heldout,
        observation_model=str(observation_model),
        time_smoothing_sigma=float(time_smoothing_sigma),
        time_shrinkage=float(time_shrinkage),
        ridge=float(ridge),
    )
    u = np.asarray(fit["basis"], dtype=np.float64)
    a_mats = np.asarray(fit["A_matrices"], dtype=np.float64)
    resid_var = np.asarray(fit["residual_variance"], dtype=np.float64)

    scores = np.empty(n_candidates, dtype=np.float64)
    filtered = np.empty((n_candidates, n_time, 2), dtype=np.float64)
    obs_vars = np.empty(n_candidates, dtype=np.float64)
    projected_obs = np.empty((n_candidates, n_time, u.shape[1]), dtype=np.float64)
    mode = str(continuous_score_mode)
    if mode not in {
        "kalman_marginal",
        "ar1_profile",
        "linear_poisson_profile",
        "quadratic_poisson_profile",
        "quadratic_affine_poisson_profile",
        "quadratic_prior_mean_poisson_profile",
        "catalog_residual_profile",
        "coarse_to_fine_profile",
    }:
        raise ValueError(
            "continuous_score_mode must be 'kalman_marginal', 'ar1_profile', "
            "'linear_poisson_profile', 'quadratic_poisson_profile', "
            "'quadratic_affine_poisson_profile', 'quadratic_prior_mean_poisson_profile', "
            "'catalog_residual_profile', or 'coarse_to_fine_profile'"
        )
    prior_mean_mode = str(trajectory_prior_mean)
    if prior_mean_mode not in {"zero", "catalog_mean"}:
        raise ValueError("trajectory_prior_mean must be 'zero' or 'catalog_mean'")
    initial_position_mode = str(trajectory_initial_position)
    if initial_position_mode not in {"inferred", "known_start"}:
        raise ValueError("trajectory_initial_position must be 'inferred' or 'known_start'")
    initial_position_var = float(trajectory_initial_position_var)
    if not np.isfinite(initial_position_var) or initial_position_var <= 0.0:
        raise ValueError("trajectory_initial_position_var must be positive and finite")
    process_model = str(trajectory_process_model)
    if process_model not in {"ar1", "matched_brownian", "catalog_gaussian"}:
        raise ValueError("trajectory_process_model must be 'ar1', 'matched_brownian', or 'catalog_gaussian'")
    brownian_scale = float(brownian_cov_scale)
    if not np.isfinite(brownian_scale) or brownian_scale <= 0.0:
        raise ValueError("brownian_cov_scale must be positive and finite")
    catalog_gaussian_sigma = float(catalog_gaussian_smoothing_sigma)
    if not np.isfinite(catalog_gaussian_sigma) or catalog_gaussian_sigma < 0.0:
        raise ValueError("catalog_gaussian_smoothing_sigma must be finite and non-negative")
    catalog_gaussian_floor = float(catalog_gaussian_cov_floor)
    if not np.isfinite(catalog_gaussian_floor) or catalog_gaussian_floor <= 0.0:
        raise ValueError("catalog_gaussian_cov_floor must be positive and finite")
    catalog_gaussian_shrink = float(catalog_gaussian_shrinkage)
    if not np.isfinite(catalog_gaussian_shrink) or not 0.0 <= catalog_gaussian_shrink <= 1.0:
        raise ValueError("catalog_gaussian_shrinkage must be finite and in [0, 1]")
    trajectory_basis_family_name = str(trajectory_basis_family)
    if trajectory_basis_family_name not in {"dct", "catalog_pca", "catalog_gaussian"}:
        raise ValueError("trajectory_basis_family must be 'dct', 'catalog_pca', or 'catalog_gaussian'")
    trajectory_basis_k = max(0, int(trajectory_basis_components))
    trajectory_basis_sigma = float(trajectory_basis_smoothing_sigma)
    if not np.isfinite(trajectory_basis_sigma) or trajectory_basis_sigma < 0.0:
        raise ValueError("trajectory_basis_smoothing_sigma must be finite and non-negative")
    trajectory_basis_var = float(trajectory_basis_coeff_prior_var)
    if not np.isfinite(trajectory_basis_var) or trajectory_basis_var <= 0.0:
        raise ValueError("trajectory_basis_coeff_prior_var must be positive and finite")
    anchor_aggregation = str(catalog_residual_aggregation)
    if anchor_aggregation not in {"logmeanexp", "max", "topk_logmeanexp"}:
        raise ValueError("catalog_residual_aggregation must be 'logmeanexp', 'max', or 'topk_logmeanexp'")
    anchor_top_k = max(0, int(catalog_residual_top_k))
    all_anchor_shrinkage = float(catalog_residual_all_anchor_shrinkage)
    if not 0.0 <= all_anchor_shrinkage <= 1.0:
        raise ValueError("catalog_residual_all_anchor_shrinkage must be in [0, 1]")
    anchor_smoothing_sigma = float(catalog_residual_anchor_smoothing_sigma)
    if not np.isfinite(anchor_smoothing_sigma) or anchor_smoothing_sigma < 0.0:
        raise ValueError("catalog_residual_anchor_smoothing_sigma must be finite and non-negative")
    anchor_smoothing_schedule = _parse_float_schedule(catalog_residual_anchor_smoothing_schedule)
    if not anchor_smoothing_schedule:
        anchor_smoothing_schedule = [anchor_smoothing_sigma]
    anchor_refine_top_k = max(0, int(catalog_residual_refine_top_k))
    quad_max_iter = max(1, int(quadratic_optimizer_max_iter))
    quad_scales = _parse_float_schedule(quadratic_continuation_scales)
    if not quad_scales:
        quad_scales = [1.0]
    quad_obs_scales = _parse_float_schedule(quadratic_observation_scales)
    if not quad_obs_scales:
        quad_obs_scales = [1.0]
    if len(quad_obs_scales) not in {1, len(quad_scales)}:
        raise ValueError("quadratic_observation_scales must contain one value or match quadratic_continuation_scales")
    if any(value < 0.0 for value in quad_scales):
        raise ValueError("quadratic_continuation_scales values must be non-negative")
    if any(value <= 0.0 for value in quad_obs_scales):
        raise ValueError("quadratic_observation_scales values must be positive")
    quad_intercept_mult = float(quadratic_intercept_ridge_multiplier)
    if not np.isfinite(quad_intercept_mult) or quad_intercept_mult <= 0.0:
        raise ValueError("quadratic_intercept_ridge_multiplier must be positive and finite")
    quad_affine_intercept_scale = float(quadratic_affine_intercept_scale)
    if not np.isfinite(quad_affine_intercept_scale) or quad_affine_intercept_scale < 0.0:
        raise ValueError("quadratic_affine_intercept_scale must be finite and non-negative")
    quadratic_fit = None
    quadratic_coefficients = None
    quadratic_residual_variance = None
    quadratic_include_intercept = mode in {"quadratic_affine_poisson_profile", "quadratic_prior_mean_poisson_profile"}
    quadratic_intercept_strategy = "prior_mean" if mode == "quadratic_prior_mean_poisson_profile" else "free"
    if mode in {"quadratic_poisson_profile", "quadratic_affine_poisson_profile", "quadratic_prior_mean_poisson_profile"}:
        quadratic_fit = fit_quadratic_observation_maps(
            prior_lambda_counts=fit_prior,
            zero_lambda_counts=zero,
            trajectory_xy=fit_xy,
            basis=u,
            heldout_trajectory_index=fit_heldout,
            ridge=float(ridge),
            include_intercept=quadratic_include_intercept,
            intercept_ridge_multiplier=quad_intercept_mult,
            intercept_strategy=quadratic_intercept_strategy,
        )
        quadratic_coefficients = np.asarray(quadratic_fit["B_coefficients"], dtype=np.float64)
        quadratic_residual_variance = np.asarray(quadratic_fit["residual_variance"], dtype=np.float64)
    best_anchor_indices = np.full(n_candidates, -1, dtype=np.int64)
    best_anchor_scores = np.full(n_candidates, np.nan, dtype=np.float64)
    anchor_logmean_scores = np.full(n_candidates, np.nan, dtype=np.float64)
    anchor_score_gaps = np.full(n_candidates, np.nan, dtype=np.float64)
    anchor_aggregate_counts = np.zeros(n_candidates, dtype=np.int64)
    extra_fit_rows: list[dict[str, Any]] = []
    xy = _trajectory_xy_by_candidate(trajectory_xy, n_candidates=n_candidates, n_trajectories=n_traj, n_time=n_time)
    for candidate_index in range(n_candidates):
        r_var = float(observation_var) if observation_var is not None else float(resid_var[candidate_index])
        r_var = max(float(observation_var_floor), r_var)
        obs_vars[candidate_index] = r_var
        if mode == "catalog_residual_profile":
            active_anchor_indices = np.arange(n_traj, dtype=np.int64)
            anchor_scores = np.full(n_traj, np.nan, dtype=np.float64)
            anchor_residual_paths = np.full((n_traj, n_time, 2), np.nan, dtype=np.float64)
            anchor_base_paths = np.full((n_traj, n_time, 2), np.nan, dtype=np.float64)
            anchor_count_paths = np.full((n_traj, n_time, n_units), np.nan, dtype=np.float64)
            for stage_index, stage_sigma in enumerate(anchor_smoothing_schedule):
                stage_scores = np.full(n_traj, np.nan, dtype=np.float64)
                stage_residual_paths = np.full((n_traj, n_time, 2), np.nan, dtype=np.float64)
                stage_base_paths = np.full((n_traj, n_time, 2), np.nan, dtype=np.float64)
                stage_count_paths = np.full((n_traj, n_time, n_units), np.nan, dtype=np.float64)
                for trajectory_index in active_anchor_indices:
                    anchor_path = xy[candidate_index, trajectory_index]
                    if stage_sigma > 0.0:
                        anchor_path = _smooth_time_axis(anchor_path, stage_sigma)
                        anchor_shift = anchor_path - xy[candidate_index, trajectory_index]
                        anchor_compact_shift = _compact_delta_from_path(anchor_shift, a_mats[candidate_index])
                        anchor_full_shift = anchor_compact_shift @ u.T
                        anchor_counts = prior[candidate_index, trajectory_index] + anchor_full_shift
                    else:
                        anchor_counts = prior[candidate_index, trajectory_index]
                    stage_base_paths[trajectory_index] = anchor_path
                    stage_count_paths[trajectory_index] = anchor_counts
                    z_anchor = project_response_delta(obs - anchor_counts, u)
                    p_out = ar1_profile_log_score(
                        z_anchor,
                        a_mats[candidate_index],
                        alpha=float(alpha),
                        process_var=float(process_var),
                        observation_var=r_var,
                    )
                    tau_delta = np.asarray(p_out["map_means"], dtype=np.float64)
                    stage_residual_paths[trajectory_index] = tau_delta
                    compact_delta = _compact_delta_from_path(tau_delta, a_mats[candidate_index])
                    full_delta = compact_delta @ u.T
                    pred_counts = np.maximum(anchor_counts + full_delta, float(eps))
                    stage_scores[trajectory_index] = float(
                        poisson_expected_count_loglik(
                            obs,
                            pred_counts,
                            eps=float(eps),
                            likelihood_scale=float(likelihood_scale),
                        )
                    )
                is_final_stage = stage_index == len(anchor_smoothing_schedule) - 1
                if is_final_stage:
                    anchor_scores = stage_scores
                    anchor_residual_paths = stage_residual_paths
                    anchor_base_paths = stage_base_paths
                    anchor_count_paths = stage_count_paths
                elif anchor_refine_top_k > 0:
                    finite_active = active_anchor_indices[np.isfinite(stage_scores[active_anchor_indices])]
                    if finite_active.size:
                        keep_count = min(anchor_refine_top_k, int(finite_active.size))
                        kept = finite_active[np.argsort(stage_scores[finite_active])[-keep_count:]]
                        active_anchor_indices = np.sort(kept.astype(np.int64))
                    else:
                        active_anchor_indices = np.empty(0, dtype=np.int64)
            finite_anchor_indices = np.flatnonzero(np.isfinite(anchor_scores))
            best_anchor = (
                int(finite_anchor_indices[np.argmax(anchor_scores[finite_anchor_indices])])
                if finite_anchor_indices.size
                else 0
            )
            best_score = float(anchor_scores[best_anchor]) if finite_anchor_indices.size else float("nan")
            logmean_score = float(logmeanexp(anchor_scores[finite_anchor_indices])) if finite_anchor_indices.size else float("nan")
            best_anchor_indices[candidate_index] = best_anchor
            best_anchor_scores[candidate_index] = best_score
            anchor_logmean_scores[candidate_index] = logmean_score
            anchor_score_gaps[candidate_index] = float(best_score - logmean_score)
            if anchor_aggregation == "max":
                aggregate_score = best_score
                anchor_aggregate_counts[candidate_index] = 1
            elif anchor_aggregation == "topk_logmeanexp":
                finite_anchor_scores = anchor_scores[finite_anchor_indices]
                if finite_anchor_scores.size:
                    k = finite_anchor_scores.size if anchor_top_k <= 0 else min(anchor_top_k, finite_anchor_scores.size)
                    top_scores = np.sort(finite_anchor_scores)[-k:]
                    aggregate_score = float(logmeanexp(top_scores))
                    anchor_aggregate_counts[candidate_index] = int(k)
                else:
                    aggregate_score = float("nan")
                    anchor_aggregate_counts[candidate_index] = 0
            else:
                aggregate_score = logmean_score
                anchor_aggregate_counts[candidate_index] = int(np.isfinite(anchor_scores).sum())
            scores[candidate_index] = float(
                (1.0 - all_anchor_shrinkage) * aggregate_score + all_anchor_shrinkage * logmean_score
            )
            filtered[candidate_index] = anchor_base_paths[best_anchor] + anchor_residual_paths[best_anchor]
            best_anchor_counts = (
                anchor_count_paths[best_anchor]
                if finite_anchor_indices.size
                else prior[candidate_index, best_anchor]
            )
            projected_obs[candidate_index] = project_response_delta(obs - best_anchor_counts, u)
            continue

        z_obs = project_response_delta(obs - zero[candidate_index], u)
        projected_obs[candidate_index] = z_obs
        candidate_prior_mean = (
            np.mean(xy[candidate_index], axis=0)
            if prior_mean_mode == "catalog_mean"
            else None
        )
        candidate_initial_mean = None
        candidate_initial_cov_override = None
        if initial_position_mode == "known_start":
            if observed_trajectory_xy is None:
                raise ValueError("trajectory_initial_position='known_start' requires observed_trajectory_xy")
            observed_xy = np.asarray(observed_trajectory_xy, dtype=np.float64)
            if observed_xy.shape != (n_time, 2):
                raise ValueError(f"observed_trajectory_xy must be (time, 2), got {observed_xy.shape}")
            candidate_initial_mean = observed_xy[0]
            candidate_initial_cov_override = np.eye(2, dtype=np.float64) * initial_position_var
        candidate_alpha = float(alpha)
        candidate_process_cov = None
        candidate_initial_cov = None
        candidate_prior_samples = _heldout_catalog_samples(xy[candidate_index], true_trajectory_index)
        if process_model == "matched_brownian":
            candidate_alpha = 1.0
            candidate_process_cov, candidate_initial_cov = _matched_brownian_covariances(
                candidate_prior_samples,
                floor=float(brownian_cov_floor),
            )
            candidate_process_cov = candidate_process_cov * brownian_scale
            candidate_initial_cov = candidate_initial_cov * brownian_scale
        if candidate_initial_cov_override is not None:
            candidate_initial_cov = candidate_initial_cov_override
        if mode == "kalman_marginal":
            k_out = kalman_filter_log_likelihood(
                z_obs,
                a_mats[candidate_index],
                alpha=float(alpha),
                process_var=float(process_var),
                observation_var=r_var,
                initial_mean=candidate_initial_mean,
                initial_cov=candidate_initial_cov,
            )
            scores[candidate_index] = float(k_out["log_likelihood"]) * float(likelihood_scale)
            filtered[candidate_index] = np.asarray(k_out["filtered_means"], dtype=np.float64)
        elif mode == "ar1_profile":
            if process_model == "catalog_gaussian":
                p_out = catalog_gaussian_profile_log_score(
                    z_obs,
                    a_mats[candidate_index],
                    candidate_prior_samples,
                    observation_var=r_var,
                    smoothing_sigma=catalog_gaussian_sigma,
                    cov_floor=catalog_gaussian_floor,
                    shrinkage=catalog_gaussian_shrink,
                    initial_mean=candidate_initial_mean,
                    initial_cov=candidate_initial_cov,
                )
            else:
                p_out = ar1_profile_log_score(
                    z_obs,
                    a_mats[candidate_index],
                    alpha=candidate_alpha,
                    process_var=float(process_var),
                    process_cov=candidate_process_cov,
                    observation_var=r_var,
                    initial_cov=candidate_initial_cov,
                    initial_mean=candidate_initial_mean,
                    prior_mean=candidate_prior_mean,
                )
            scores[candidate_index] = float(p_out["profile_score"]) * float(likelihood_scale)
            filtered[candidate_index] = np.asarray(p_out["map_means"], dtype=np.float64)
        elif mode == "linear_poisson_profile":
            if process_model == "catalog_gaussian":
                p_out = catalog_gaussian_profile_log_score(
                    z_obs,
                    a_mats[candidate_index],
                    candidate_prior_samples,
                    observation_var=r_var,
                    smoothing_sigma=catalog_gaussian_sigma,
                    cov_floor=catalog_gaussian_floor,
                    shrinkage=catalog_gaussian_shrink,
                    initial_mean=candidate_initial_mean,
                    initial_cov=candidate_initial_cov,
                )
            else:
                p_out = ar1_profile_log_score(
                    z_obs,
                    a_mats[candidate_index],
                    alpha=candidate_alpha,
                    process_var=float(process_var),
                    process_cov=candidate_process_cov,
                    observation_var=r_var,
                    initial_cov=candidate_initial_cov,
                    initial_mean=candidate_initial_mean,
                    prior_mean=candidate_prior_mean,
                )
            tau_hat = np.asarray(p_out["map_means"], dtype=np.float64)
            filtered[candidate_index] = tau_hat
            compact_delta = _compact_delta_from_path(tau_hat, a_mats[candidate_index])
            full_delta = compact_delta @ u.T
            pred_counts = np.maximum(zero[candidate_index] + full_delta, float(eps))
            scores[candidate_index] = float(
                poisson_expected_count_loglik(
                    obs,
                    pred_counts,
                    eps=float(eps),
                    likelihood_scale=float(likelihood_scale),
                )
            )
        elif mode in {"quadratic_poisson_profile", "quadratic_affine_poisson_profile", "quadratic_prior_mean_poisson_profile"}:
            if quadratic_coefficients is None or quadratic_residual_variance is None:
                raise RuntimeError("quadratic observation maps were not fitted")
            q_r_var = float(observation_var) if observation_var is not None else float(quadratic_residual_variance[candidate_index])
            q_r_var = max(float(observation_var_floor), q_r_var)
            linear_out = ar1_profile_log_score(
                z_obs,
                a_mats[candidate_index],
                alpha=candidate_alpha,
                process_var=float(process_var),
                process_cov=candidate_process_cov,
                observation_var=r_var,
                initial_cov=candidate_initial_cov,
                initial_mean=candidate_initial_mean,
                prior_mean=candidate_prior_mean,
            )
            starts = [
                np.asarray(linear_out["map_means"], dtype=np.float64),
                np.zeros((n_time, 2), dtype=np.float64),
                np.mean(xy[candidate_index], axis=0),
            ]
            if candidate_initial_mean is not None:
                starts = [start.copy() for start in starts]
                for start in starts:
                    start[0] = candidate_initial_mean
            q_out = quadratic_profile_log_score(
                z_obs,
                quadratic_coefficients[candidate_index],
                starts=starts,
                observation_var=q_r_var,
                alpha=candidate_alpha,
                process_var=float(process_var),
                process_cov=candidate_process_cov,
                initial_mean=candidate_initial_mean,
                initial_var=initial_position_var,
                initial_cov=candidate_initial_cov,
                max_iter=quad_max_iter,
                quadratic_scales=quad_scales,
                observation_scales=quad_obs_scales,
                intercept_scale=quad_affine_intercept_scale if quadratic_include_intercept else 1.0,
            )
            tau_hat = np.asarray(q_out["map_means"], dtype=np.float64)
            filtered[candidate_index] = tau_hat
            compact_delta = _quadratic_compact_delta_from_path(
                tau_hat,
                quadratic_coefficients[candidate_index],
                intercept_scale=quad_affine_intercept_scale if quadratic_include_intercept else 1.0,
            )
            full_delta = compact_delta @ u.T
            pred_counts = np.maximum(zero[candidate_index] + full_delta, float(eps))
            scores[candidate_index] = float(
                poisson_expected_count_loglik(
                    obs,
                    pred_counts,
                    eps=float(eps),
                    likelihood_scale=float(likelihood_scale),
                )
            )
            extra_fit_rows.append(
                {
                    "qc_type": "quadratic_profile_optimizer",
                    "candidate_index": int(candidate_index),
                    "observation_model": "quadratic_affine_time_constant"
                    if mode == "quadratic_affine_poisson_profile"
                    else "quadratic_prior_mean_affine_time_constant"
                    if mode == "quadratic_prior_mean_poisson_profile"
                    else "quadratic_time_constant",
                    "continuous_score_mode": str(mode),
                    "quadratic_include_intercept": bool(quadratic_include_intercept),
                    "quadratic_intercept_strategy": str(quadratic_intercept_strategy if quadratic_include_intercept else "none"),
                    "quadratic_intercept_ridge_multiplier": float(quad_intercept_mult),
                    "quadratic_affine_intercept_scale": float(quad_affine_intercept_scale),
                    "basis_dim": int(u.shape[1]),
                    "residual_variance": float(q_r_var),
                    "optimizer_success": bool(q_out["optimizer_success"]),
                    "optimizer_iterations": int(q_out["optimizer_iterations"]),
                    "optimizer_final_iterations": int(q_out["optimizer_final_iterations"]),
                    "optimizer_stage_success_fraction": float(q_out["optimizer_stage_success_fraction"]),
                    "optimizer_start_index": int(q_out["optimizer_start_index"]),
                    "quadratic_optimizer_max_iter": int(quad_max_iter),
                    "quadratic_continuation_scales": ",".join(f"{value:g}" for value in quad_scales),
                    "quadratic_observation_scales": ",".join(f"{value:g}" for value in quad_obs_scales),
                    "profile_energy": float(q_out["profile_energy"]),
                }
            )
        else:
            if trajectory_basis_family_name == "catalog_gaussian":
                coarse_out = catalog_gaussian_profile_log_score(
                    z_obs,
                    a_mats[candidate_index],
                    candidate_prior_samples,
                    observation_var=r_var,
                    smoothing_sigma=catalog_gaussian_sigma,
                    cov_floor=catalog_gaussian_floor,
                    shrinkage=catalog_gaussian_shrink,
                    initial_mean=candidate_initial_mean,
                    initial_cov=candidate_initial_cov,
                )
            elif trajectory_basis_family_name == "catalog_pca":
                coarse_samples = _smooth_trajectory_samples(candidate_prior_samples, trajectory_basis_sigma)
                coarse_out = trajectory_basis_profile_log_score(
                    z_obs,
                    a_mats[candidate_index],
                    coarse_samples,
                    n_components=trajectory_basis_k,
                    coeff_prior_var=trajectory_basis_var,
                    observation_var=r_var,
                    initial_mean=candidate_initial_mean,
                    initial_cov=candidate_initial_cov,
                )
            else:
                coarse_out = temporal_basis_profile_log_score(
                    z_obs,
                    a_mats[candidate_index],
                    n_components=trajectory_basis_k,
                    coeff_prior_var=trajectory_basis_var,
                    observation_var=r_var,
                    initial_mean=candidate_initial_mean,
                    initial_cov=candidate_initial_cov,
                )
            coarse_tau = np.asarray(coarse_out["map_means"], dtype=np.float64)
            coarse_compact = _compact_delta_from_path(coarse_tau, a_mats[candidate_index])
            residual_z = z_obs - coarse_compact
            residual_initial_mean = (
                candidate_initial_mean - coarse_tau[0]
                if candidate_initial_mean is not None
                else None
            )
            refined_out = ar1_profile_log_score(
                residual_z,
                a_mats[candidate_index],
                alpha=candidate_alpha,
                process_var=float(process_var),
                process_cov=candidate_process_cov,
                observation_var=r_var,
                initial_cov=candidate_initial_cov,
                initial_mean=residual_initial_mean,
            )
            residual_tau = np.asarray(refined_out["map_means"], dtype=np.float64)
            tau_hat = coarse_tau + residual_tau
            filtered[candidate_index] = tau_hat
            compact_delta = _compact_delta_from_path(tau_hat, a_mats[candidate_index])
            full_delta = compact_delta @ u.T
            pred_counts = np.maximum(zero[candidate_index] + full_delta, float(eps))
            scores[candidate_index] = float(
                poisson_expected_count_loglik(
                    obs,
                    pred_counts,
                    eps=float(eps),
                    likelihood_scale=float(likelihood_scale),
                )
            )

    finite = score_image_identity_score_vectors(
        y_obs_counts=obs,
        prior_lambda_counts=prior,
        known_lambda_counts=known,
        zero_lambda_counts=zero,
        true_candidate_index=true_idx,
        candidate_ids=ids,
        eps=float(eps),
        likelihood_scale=float(likelihood_scale),
    )
    true_tau = None
    if observed_trajectory_xy is not None:
        true_tau = np.asarray(observed_trajectory_xy, dtype=np.float64)
    else:
        tau_idx = -1 if true_trajectory_index is None else int(true_trajectory_index)
        if 0 <= tau_idx < n_traj:
            true_tau = xy[true_idx, tau_idx]
    recovery = (
        trajectory_recovery_metrics(filtered[true_idx], true_tau)
        if true_tau is not None
        else {
            "trajectory_rmse": float("nan"),
            "trajectory_corr_x": float("nan"),
            "trajectory_corr_y": float("nan"),
            "trajectory_corr_mean": float("nan"),
            "trajectory_r2": float("nan"),
            "trajectory_hat_rms": float(np.sqrt(np.mean(filtered[true_idx] * filtered[true_idx]))),
            "trajectory_true_rms": float("nan"),
        }
    )
    zero_scores = np.asarray(finite["zero_scores"], dtype=np.float64)
    return {
        **finite,
        "continuous_joint_scores": scores,
        "continuous_joint_pred_candidate_index": int(np.nanargmax(scores)) if scores.size else -1,
        "continuous_joint_true_rank": rank_desc(scores, true_idx),
        "continuous_joint_true_margin": true_margin(scores, true_idx),
        "continuous_joint_true_score": float(scores[true_idx]),
        "continuous_joint_minus_zero_true_score": float(scores[true_idx] - zero_scores[true_idx]),
        "continuous_joint_score_corr_with_zero": (
            float(np.corrcoef(scores, zero_scores)[0, 1])
            if scores.size > 1 and np.isfinite(scores).all() and np.isfinite(zero_scores).all()
            else float("nan")
        ),
        "A_matrices": a_mats,
        "filtered_state_means": filtered,
        "projected_obs_by_candidate": projected_obs,
        "kalman_observation_variance": obs_vars,
        "fit_rows": [
            *fit["fit_rows"],
            *([] if quadratic_fit is None else quadratic_fit["fit_rows"]),
            *extra_fit_rows,
        ],
        "trajectory_recovery": recovery,
        "kalman_alpha": float(alpha),
        "kalman_process_var": float(process_var),
        "observation_model": str(observation_model),
        "time_smoothing_sigma": float(time_smoothing_sigma),
        "time_shrinkage": float(time_shrinkage),
        "continuous_score_mode": mode,
        "trajectory_prior_mean": prior_mean_mode,
        "trajectory_initial_position": initial_position_mode,
        "trajectory_initial_position_var": initial_position_var,
        "trajectory_process_model": process_model,
        "brownian_cov_floor": float(brownian_cov_floor),
        "brownian_cov_scale": brownian_scale,
        "catalog_gaussian_smoothing_sigma": catalog_gaussian_sigma,
        "catalog_gaussian_cov_floor": catalog_gaussian_floor,
        "catalog_gaussian_shrinkage": catalog_gaussian_shrink,
        "trajectory_basis_family": trajectory_basis_family_name,
        "trajectory_basis_components": trajectory_basis_k,
        "trajectory_basis_smoothing_sigma": trajectory_basis_sigma,
        "trajectory_basis_coeff_prior_var": trajectory_basis_var,
        "catalog_residual_aggregation": anchor_aggregation,
        "catalog_residual_top_k": anchor_top_k,
        "catalog_residual_all_anchor_shrinkage": all_anchor_shrinkage,
        "catalog_residual_anchor_smoothing_sigma": anchor_smoothing_sigma,
        "catalog_residual_anchor_smoothing_schedule": ",".join(f"{v:g}" for v in anchor_smoothing_schedule),
        "catalog_residual_refine_top_k": anchor_refine_top_k,
        "quadratic_optimizer_max_iter": quad_max_iter,
        "quadratic_continuation_scales": ",".join(f"{value:g}" for value in quad_scales),
        "quadratic_observation_scales": ",".join(f"{value:g}" for value in quad_obs_scales),
        "quadratic_intercept_ridge_multiplier": float(quad_intercept_mult),
        "quadratic_affine_intercept_scale": float(quad_affine_intercept_scale),
        "catalog_residual_best_anchor_indices": best_anchor_indices,
        "catalog_residual_best_anchor_scores": best_anchor_scores,
        "catalog_residual_anchor_logmean_scores": anchor_logmean_scores,
        "catalog_residual_anchor_score_gaps": anchor_score_gaps,
        "catalog_residual_anchor_aggregate_counts": anchor_aggregate_counts,
        "basis_dim": int(u.shape[1]),
    }


def _score_summary(prefix: str, scores: np.ndarray, true_idx: int, ids: list[str]) -> dict[str, Any]:
    vals = np.asarray(scores, dtype=np.float64)
    pred = int(np.nanargmax(vals)) if vals.size and np.isfinite(vals).any() else -1
    return {
        f"{prefix}_pred_candidate_index": pred,
        f"{prefix}_pred_image_id": ids[pred] if 0 <= pred < len(ids) else "",
        f"{prefix}_correct": bool(pred == int(true_idx)) if pred >= 0 else False,
        f"{prefix}_true_rank": rank_desc(vals, int(true_idx)),
        f"{prefix}_true_margin": true_margin(vals, int(true_idx)),
        f"{prefix}_true_score": float(vals[int(true_idx)]) if 0 <= int(true_idx) < vals.shape[0] else float("nan"),
    }


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    group_cols = [
        "candidate_set_mode",
        "observation_family",
        "prior_family",
        "prior_scale",
        "axis_catalog_mode",
        "basis_source",
        "basis_max_dim_requested",
        "basis_dim",
        "ridge",
        "continuous_posterior_temperature",
        "kalman_alpha",
        "kalman_process_var",
        "observation_model",
        "time_smoothing_sigma",
        "time_shrinkage",
        "continuous_score_mode",
        "trajectory_prior_mean",
        "trajectory_initial_position",
        "trajectory_initial_position_var",
        "trajectory_process_model",
        "brownian_cov_floor",
        "brownian_cov_scale",
        "catalog_gaussian_smoothing_sigma",
        "catalog_gaussian_cov_floor",
        "catalog_gaussian_shrinkage",
        "trajectory_basis_family",
        "trajectory_basis_components",
        "trajectory_basis_smoothing_sigma",
        "trajectory_basis_coeff_prior_var",
        "catalog_residual_aggregation",
        "catalog_residual_top_k",
        "catalog_residual_all_anchor_shrinkage",
        "catalog_residual_anchor_smoothing_sigma",
        "catalog_residual_anchor_smoothing_schedule",
        "catalog_residual_refine_top_k",
        "quadratic_optimizer_max_iter",
        "quadratic_continuation_scales",
        "quadratic_observation_scales",
        "likelihood_scale",
    ]
    frame = pd.DataFrame(rows)
    for col in group_cols:
        if col not in frame.columns:
            frame[col] = ""
    out: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        row = {col: value for col, value in zip(group_cols, key)}
        row["n_trials"] = int(group.shape[0])
        for mode in ["known", "zero", "joint", "best_single_tau", "continuous_joint"]:
            correct = group.get(f"{mode}_correct")
            rank = pd.to_numeric(group.get(f"{mode}_true_rank"), errors="coerce")
            margin = pd.to_numeric(group.get(f"{mode}_true_margin"), errors="coerce")
            row[f"{mode}_accuracy"] = float(np.mean(correct.astype(bool))) if correct is not None else float("nan")
            row[f"{mode}_median_true_rank"] = float(np.nanmedian(rank)) if np.isfinite(rank).any() else float("nan")
            row[f"{mode}_median_true_margin"] = float(np.nanmedian(margin)) if np.isfinite(margin).any() else float("nan")
        row["continuous_minus_zero_accuracy"] = row["continuous_joint_accuracy"] - row["zero_accuracy"]
        row["joint_minus_zero_accuracy"] = row["joint_accuracy"] - row["zero_accuracy"]
        rmse = pd.to_numeric(group.get("trajectory_rmse"), errors="coerce")
        corr = pd.to_numeric(group.get("trajectory_corr_mean"), errors="coerce")
        row["median_trajectory_rmse"] = float(np.nanmedian(rmse)) if np.isfinite(rmse).any() else float("nan")
        row["median_trajectory_corr_mean"] = float(np.nanmedian(corr)) if np.isfinite(corr).any() else float("nan")
        out.append(row)
    return out


def _load_basis(
    path: Path | None,
    *,
    n_units: int,
    basis_key: str,
    basis_max_dim: int | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if path is None:
        k_dim = int(n_units) if basis_max_dim is None or int(basis_max_dim) <= 0 else min(int(n_units), int(basis_max_dim))
        return np.eye(int(n_units), k_dim, dtype=np.float64), {"basis_source": "identity_units", "basis_dim": int(k_dim)}
    with np.load(path, allow_pickle=True) as data:
        if basis_key == "auto":
            preferred = ["U", "basis", "basis_delta_0p25", "basis_uncentered", "basis_centered_across_tangents_per_unit"]
            chosen = next((key for key in preferred if key in data.files), None)
            if chosen is None:
                chosen = next((key for key in data.files if str(key).startswith("basis")), None)
            if chosen is None:
                raise ValueError(f"No basis-like key found in {path}; available keys={list(data.files)}")
        else:
            chosen = str(basis_key)
            if chosen not in data.files:
                raise ValueError(f"basis_key={chosen!r} not found in {path}; available keys={list(data.files)}")
        basis = np.asarray(data[chosen], dtype=np.float64)
    basis = _as_basis(basis, n_units)
    if basis_max_dim is not None and int(basis_max_dim) > 0:
        basis = basis[:, : min(int(basis_max_dim), basis.shape[1])]
    return basis, {"basis_source": str(path), "basis_key": chosen, "basis_dim": int(basis.shape[1])}


def _trajectory_from_table_or_npz(
    *,
    table: dict[str, np.ndarray],
    trajectory_npz: dict[str, np.ndarray] | None,
    trajectory_key: str,
) -> np.ndarray:
    for key in ["prior_trajectory_xy", "trajectory_xy", "prior_trajectories_xy"]:
        if key in table:
            return np.asarray(table[key], dtype=np.float64)
    if trajectory_npz is not None and trajectory_key in trajectory_npz:
        return np.asarray(trajectory_npz[trajectory_key], dtype=np.float64)
    available = sorted([key for key in table if "trajectory" in key.lower()])
    raise ValueError(
        "No trajectory coordinates found. Expected table key prior_trajectory_xy/trajectory_xy "
        f"or --trajectory-npz key {trajectory_key!r}. Table trajectory-like keys={available}"
    )


def _observed_trajectory_from_table_or_npz(
    *,
    table: dict[str, np.ndarray],
    trajectory_npz: dict[str, np.ndarray] | None,
    observed_trajectory_key: str,
) -> np.ndarray | None:
    for key in ["observed_trajectory_xy", "true_trajectory_xy", "y_obs_trajectory_xy"]:
        if key in table:
            arr = np.asarray(table[key], dtype=np.float64)
            return arr.reshape(arr.shape[-2], 2) if arr.ndim >= 2 else arr
    if trajectory_npz is not None and observed_trajectory_key and observed_trajectory_key in trajectory_npz:
        arr = np.asarray(trajectory_npz[observed_trajectory_key], dtype=np.float64)
        return arr.reshape(arr.shape[-2], 2) if arr.ndim >= 2 else arr
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--response-manifest", type=Path, default=None)
    parser.add_argument("--compact-basis-path", type=Path, default=None)
    parser.add_argument("--basis-key", default="auto")
    parser.add_argument("--basis-max-dim", type=int, default=0)
    parser.add_argument(
        "--basis-max-dim-by-scale",
        default="",
        help="Optional comma-separated scale:dim overrides, e.g. 0.5:10,1.0:20,2.0:20.",
    )
    parser.add_argument("--trajectory-npz", type=Path, default=None)
    parser.add_argument(
        "--trajectory-sidecar-dir",
        type=Path,
        default=None,
        help="Directory mirroring response_cache_path entries with npz files containing trajectory coordinates.",
    )
    parser.add_argument("--trajectory-key", default="trajectory_xy")
    parser.add_argument("--observed-trajectory-key", default="observed_trajectory_xy")
    parser.add_argument("--likelihood-scales", default="1.0")
    parser.add_argument(
        "--continuous-posterior-temperature",
        type=float,
        default=1.0,
        help=(
            "Positive temperature applied only when writing continuous_joint candidate posterior scores. "
            "This does not change MAP image accuracy or trajectory optimization."
        ),
    )
    parser.add_argument(
        "--continuous-posterior-temperature-by-scale",
        default="",
        help="Optional comma-separated scale:temperature overrides for continuous_joint posterior scores.",
    )
    parser.add_argument("--alpha", type=float, default=0.92)
    parser.add_argument("--process-var", type=float, default=1e-3)
    parser.add_argument("--observation-var", type=float, default=None)
    parser.add_argument("--observation-var-floor", type=float, default=1e-6)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument(
        "--ridge-by-scale",
        default="",
        help="Optional comma-separated scale:ridge overrides, e.g. 0.5:0.01,1.0:0.1,2.0:0.1.",
    )
    parser.add_argument(
        "--observation-model",
        choices=("time_constant", "time_varying"),
        default="time_constant",
    )
    parser.add_argument("--time-smoothing-sigma", type=float, default=0.0)
    parser.add_argument("--time-shrinkage", type=float, default=0.0)
    parser.add_argument(
        "--continuous-score-mode",
        choices=(
            "kalman_marginal",
            "ar1_profile",
            "linear_poisson_profile",
            "catalog_residual_profile",
            "coarse_to_fine_profile",
            "quadratic_poisson_profile",
            "quadratic_affine_poisson_profile",
            "quadratic_prior_mean_poisson_profile",
        ),
        default="kalman_marginal",
    )
    parser.add_argument(
        "--trajectory-prior-mean",
        choices=("zero", "catalog_mean"),
        default="zero",
        help="Mean path for no-anchor AR(1) profile inference; catalog_mean averages the trajectory catalog without selecting anchors.",
    )
    parser.add_argument(
        "--trajectory-initial-position",
        choices=("inferred", "known_start"),
        default="inferred",
        help="Initial-position prior for no-anchor trajectory inference; known_start uses observed_trajectory_xy[0] only.",
    )
    parser.add_argument(
        "--trajectory-initial-position-var",
        type=float,
        default=1e-4,
        help="Soft prior variance for known_start initial position; use a small positive value for a near-hard start.",
    )
    parser.add_argument(
        "--trajectory-process-model",
        choices=("ar1", "matched_brownian", "catalog_gaussian"),
        default="ar1",
        help="No-anchor trajectory process prior for profile modes.",
    )
    parser.add_argument(
        "--trajectory-process-model-by-scale",
        default="",
        help=(
            "Optional comma-separated scale:model overrides for no-anchor trajectory process priors, "
            "e.g. 0.5:ar1,1.0:ar1,2.0:matched_brownian."
        ),
    )
    parser.add_argument("--brownian-cov-floor", type=float, default=1e-6)
    parser.add_argument("--brownian-cov-scale", type=float, default=1.0)
    parser.add_argument(
        "--brownian-cov-scale-by-scale",
        default="",
        help="Optional comma-separated scale:multiplier overrides for matched-Brownian covariance scale.",
    )
    parser.add_argument(
        "--catalog-gaussian-smoothing-sigma",
        type=float,
        default=0.0,
        help="Optional time smoothing for traces before estimating the no-anchor catalog Gaussian prior.",
    )
    parser.add_argument(
        "--catalog-gaussian-cov-floor",
        type=float,
        default=1e-6,
        help="Eigenvalue floor for the no-anchor catalog Gaussian trajectory covariance.",
    )
    parser.add_argument(
        "--catalog-gaussian-shrinkage",
        type=float,
        default=0.25,
        help="Shrink empirical catalog trajectory covariance toward its diagonal before inversion.",
    )
    parser.add_argument(
        "--trajectory-basis-family",
        choices=("dct", "catalog_pca", "catalog_gaussian"),
        default="dct",
        help="Coarse prior for coarse_to_fine_profile; dct is generic, catalog_gaussian uses trace statistics without selecting anchors, catalog_pca is diagnostic.",
    )
    parser.add_argument("--trajectory-basis-components", type=int, default=4)
    parser.add_argument("--trajectory-basis-smoothing-sigma", type=float, default=6.0)
    parser.add_argument("--trajectory-basis-coeff-prior-var", type=float, default=1.0)
    parser.add_argument(
        "--catalog-residual-aggregation",
        choices=("logmeanexp", "max", "topk_logmeanexp"),
        default="logmeanexp",
        help="Anchor aggregation for catalog_residual_profile; logmeanexp preserves the current marginal default.",
    )
    parser.add_argument(
        "--catalog-residual-top-k",
        type=int,
        default=0,
        help="Number of anchors retained by topk_logmeanexp; <=0 uses all finite anchors.",
    )
    parser.add_argument(
        "--catalog-residual-all-anchor-shrinkage",
        type=float,
        default=0.0,
        help="Shrink profiled catalog-residual anchor score toward the all-anchor logmeanexp score.",
    )
    parser.add_argument(
        "--catalog-residual-anchor-smoothing-sigma",
        type=float,
        default=0.0,
        help="Gaussian time smoothing sigma for catalog-residual anchor trajectories before local-linear response adjustment.",
    )
    parser.add_argument(
        "--catalog-residual-anchor-smoothing-schedule",
        default="",
        help="Comma-separated smoothing sigmas for coarse-to-fine catalog-residual anchor search; empty uses --catalog-residual-anchor-smoothing-sigma.",
    )
    parser.add_argument(
        "--catalog-residual-refine-top-k",
        type=int,
        default=0,
        help="If positive, keep only this many anchors after each non-final smoothing stage.",
    )
    parser.add_argument(
        "--quadratic-optimizer-max-iter",
        type=int,
        default=80,
        help="L-BFGS-B iterations per continuation stage for quadratic_poisson_profile.",
    )
    parser.add_argument(
        "--quadratic-continuation-scales",
        default="1",
        help="Comma-separated schedule for turning on quadratic observation terms.",
    )
    parser.add_argument(
        "--quadratic-observation-scales",
        default="1",
        help="Comma-separated observation-variance multipliers for quadratic continuation stages.",
    )
    parser.add_argument(
        "--quadratic-intercept-ridge-multiplier",
        type=float,
        default=1.0,
        help=(
            "Positive multiplier on the ridge penalty for the affine quadratic intercept term. "
            "Only affects quadratic_affine_poisson_profile."
        ),
    )
    parser.add_argument(
        "--quadratic-affine-intercept-scale",
        type=float,
        default=1.0,
        help=(
            "Non-negative multiplier applied to the fitted affine quadratic intercept during "
            "trajectory profiling and Poisson scoring. Use 0 for an intercept-ablation control."
        ),
    )
    parser.add_argument("--eps", type=float, default=1e-8)
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
    parser.add_argument("--skip-tables", type=int, default=0)
    parser.add_argument("--max-tables", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def analyze(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.response_manifest) if args.response_manifest else run_dir / "response_cache_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"response manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
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
    likelihood_scales = [float(part.strip()) for part in str(args.likelihood_scales).split(",") if part.strip()]
    basis_max_dim_by_scale = _parse_scale_value_map(
        str(args.basis_max_dim_by_scale),
        value_type=int,
        name="basis_max_dim_by_scale",
    )
    ridge_by_scale = _parse_scale_value_map(str(args.ridge_by_scale), value_type=float, name="ridge_by_scale")
    posterior_temperature = _validate_positive_float(
        float(args.continuous_posterior_temperature),
        name="continuous_posterior_temperature",
    )
    posterior_temperature_by_scale = _parse_scale_value_map(
        str(args.continuous_posterior_temperature_by_scale),
        value_type=float,
        name="continuous_posterior_temperature_by_scale",
    )
    trajectory_process_model_by_scale = _parse_scale_value_map(
        str(args.trajectory_process_model_by_scale),
        value_type=str,
        name="trajectory_process_model_by_scale",
    )
    brownian_cov_scale_by_scale = _parse_scale_value_map(
        str(args.brownian_cov_scale_by_scale),
        value_type=float,
        name="brownian_cov_scale_by_scale",
    )
    for scale, value in basis_max_dim_by_scale.items():
        if int(value) < 0:
            raise ValueError(f"basis_max_dim_by_scale for scale {scale:g} must be non-negative")
    for scale, value in ridge_by_scale.items():
        if float(value) < 0.0 or not np.isfinite(float(value)):
            raise ValueError(f"ridge_by_scale for scale {scale:g} must be finite and non-negative")
    for scale, value in posterior_temperature_by_scale.items():
        posterior_temperature_by_scale[scale] = _validate_positive_float(
            float(value),
            name=f"continuous_posterior_temperature_by_scale[{scale:g}]",
        )
    allowed_process_models = {"ar1", "matched_brownian", "catalog_gaussian"}
    for scale, value in trajectory_process_model_by_scale.items():
        if str(value) not in allowed_process_models:
            raise ValueError(
                f"trajectory_process_model_by_scale for scale {scale:g} must be one of "
                f"{sorted(allowed_process_models)}, got {value!r}"
            )
    for scale, value in brownian_cov_scale_by_scale.items():
        brownian_cov_scale_by_scale[scale] = _validate_positive_float(
            float(value),
            name=f"brownian_cov_scale_by_scale[{scale:g}]",
        )
    trajectory_npz = _load_npz(Path(args.trajectory_npz)) if args.trajectory_npz is not None else None

    trial_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    basis_meta: dict[str, Any] | None = None
    progress_every = max(1, int(args.progress_every))

    for table_index, (_idx, man_row) in enumerate(manifest.iterrows(), start=1):
        table_path = run_dir / str(man_row["response_cache_path"])
        table = _load_npz(table_path)
        if args.trajectory_sidecar_dir is not None and "prior_trajectory_xy" not in table:
            sidecar_path = Path(args.trajectory_sidecar_dir) / str(man_row["response_cache_path"])
            if not sidecar_path.exists():
                raise FileNotFoundError(f"trajectory sidecar not found for {man_row['response_cache_path']}: {sidecar_path}")
            sidecar = _load_npz(sidecar_path)
            table = {**table, **sidecar}
        prior = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
        known = np.asarray(table["known_lambda_counts"], dtype=np.float64)
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float64)
        y_obs = np.asarray(table["y_obs_counts"], dtype=np.float64)
        n_candidates, n_traj, n_time, n_units = prior.shape
        prior_scale = float(man_row.get("scale", np.nan))
        basis_max_dim_for_table = int(
            _scale_value(basis_max_dim_by_scale, prior_scale, int(args.basis_max_dim))
        )
        ridge_for_table = float(_scale_value(ridge_by_scale, prior_scale, float(args.ridge)))
        posterior_temperature_for_table = float(
            _scale_value(posterior_temperature_by_scale, prior_scale, posterior_temperature)
        )
        trajectory_process_model_for_table = _scale_string_value(
            trajectory_process_model_by_scale,
            prior_scale,
            str(args.trajectory_process_model),
        )
        brownian_cov_scale_for_table = float(
            _scale_value(brownian_cov_scale_by_scale, prior_scale, float(args.brownian_cov_scale))
        )
        basis, loaded_basis_meta = _load_basis(
            Path(args.compact_basis_path) if args.compact_basis_path is not None else None,
            n_units=n_units,
            basis_key=str(args.basis_key),
            basis_max_dim=basis_max_dim_for_table,
        )
        basis_meta = loaded_basis_meta
        trajectory_xy = _trajectory_from_table_or_npz(
            table=table,
            trajectory_npz=trajectory_npz,
            trajectory_key=str(args.trajectory_key),
        )
        observed_xy = _observed_trajectory_from_table_or_npz(
            table=table,
            trajectory_npz=trajectory_npz,
            observed_trajectory_key=str(args.observed_trajectory_key),
        )
        true_idx = _scalar_int(table, "true_candidate_index", 0)
        true_tau_idx = _scalar_int(table, "true_trajectory_index", -1)
        nearest_tau_idx = _scalar_int(table, "nearest_trajectory_index", -1)
        nearest_tau_dist = float(np.asarray(table.get("nearest_trajectory_distance", [np.nan])).reshape(-1)[0])
        ids = _candidate_ids(table, n_candidates)

        for likelihood_scale in likelihood_scales:
            vectors = score_continuous_joint_score_vectors(
                y_obs_counts=y_obs,
                prior_lambda_counts=prior,
                known_lambda_counts=known,
                zero_lambda_counts=zero,
                trajectory_xy=trajectory_xy,
                true_candidate_index=true_idx,
                candidate_ids=ids,
                basis=basis,
                true_trajectory_index=true_tau_idx,
                observed_trajectory_xy=observed_xy,
                alpha=float(args.alpha),
                process_var=float(args.process_var),
                observation_var=args.observation_var,
                observation_var_floor=float(args.observation_var_floor),
                ridge=float(ridge_for_table),
                observation_model=str(args.observation_model),
                time_smoothing_sigma=float(args.time_smoothing_sigma),
                time_shrinkage=float(args.time_shrinkage),
                continuous_score_mode=str(args.continuous_score_mode),
                trajectory_prior_mean=str(args.trajectory_prior_mean),
                trajectory_initial_position=str(args.trajectory_initial_position),
                trajectory_initial_position_var=float(args.trajectory_initial_position_var),
                trajectory_process_model=str(trajectory_process_model_for_table),
                brownian_cov_floor=float(args.brownian_cov_floor),
                brownian_cov_scale=float(brownian_cov_scale_for_table),
                catalog_gaussian_smoothing_sigma=float(args.catalog_gaussian_smoothing_sigma),
                catalog_gaussian_cov_floor=float(args.catalog_gaussian_cov_floor),
                catalog_gaussian_shrinkage=float(args.catalog_gaussian_shrinkage),
                trajectory_basis_family=str(args.trajectory_basis_family),
                trajectory_basis_components=int(args.trajectory_basis_components),
                trajectory_basis_smoothing_sigma=float(args.trajectory_basis_smoothing_sigma),
                trajectory_basis_coeff_prior_var=float(args.trajectory_basis_coeff_prior_var),
                catalog_residual_aggregation=str(args.catalog_residual_aggregation),
                catalog_residual_top_k=int(args.catalog_residual_top_k),
                catalog_residual_all_anchor_shrinkage=float(args.catalog_residual_all_anchor_shrinkage),
                catalog_residual_anchor_smoothing_sigma=float(args.catalog_residual_anchor_smoothing_sigma),
                catalog_residual_anchor_smoothing_schedule=str(args.catalog_residual_anchor_smoothing_schedule),
                catalog_residual_refine_top_k=int(args.catalog_residual_refine_top_k),
                quadratic_optimizer_max_iter=int(args.quadratic_optimizer_max_iter),
                quadratic_continuation_scales=str(args.quadratic_continuation_scales),
                quadratic_observation_scales=str(args.quadratic_observation_scales),
                quadratic_intercept_ridge_multiplier=float(args.quadratic_intercept_ridge_multiplier),
                quadratic_affine_intercept_scale=float(args.quadratic_affine_intercept_scale),
                eps=float(args.eps),
                likelihood_scale=float(likelihood_scale),
            )
            base = {
                "table_index": int(table_index - 1),
                "manifest_table_index": int(_idx),
                "trial_id": int(man_row.get("trial_id", table_index - 1)),
                "response_cache_path": str(man_row["response_cache_path"]),
                "candidate_set_mode": str(man_row.get("candidate_set_mode", "")),
                "observation_family": str(man_row.get("observation_family", "")),
                "prior_family": str(man_row.get("prior_family", "")),
                "prior_scale": float(prior_scale),
                "axis_catalog_mode": str(man_row.get("axis_catalog_mode", "shared")),
                "likelihood_scale": float(likelihood_scale),
                "n_candidates": int(n_candidates),
                "n_trajectories": int(n_traj),
                "n_timebins": int(n_time),
                "n_units": int(n_units),
                "true_candidate_index": int(true_idx),
                "true_image_id": ids[int(true_idx)],
                "true_trajectory_index": int(true_tau_idx),
                "nearest_trajectory_index": int(nearest_tau_idx),
                "nearest_trajectory_distance": float(nearest_tau_dist),
                "kalman_alpha": float(args.alpha),
                "kalman_process_var": float(args.process_var),
                "observation_model": str(args.observation_model),
                "time_smoothing_sigma": float(args.time_smoothing_sigma),
                "time_shrinkage": float(args.time_shrinkage),
                "continuous_score_mode": str(args.continuous_score_mode),
                "trajectory_prior_mean": str(args.trajectory_prior_mean),
                "trajectory_initial_position": str(args.trajectory_initial_position),
                "trajectory_initial_position_var": float(args.trajectory_initial_position_var),
                "trajectory_process_model": str(trajectory_process_model_for_table),
                "trajectory_process_model_default": str(args.trajectory_process_model),
                "trajectory_process_model_by_scale": str(args.trajectory_process_model_by_scale),
                "brownian_cov_floor": float(args.brownian_cov_floor),
                "brownian_cov_scale": float(brownian_cov_scale_for_table),
                "brownian_cov_scale_default": float(args.brownian_cov_scale),
                "brownian_cov_scale_by_scale": str(args.brownian_cov_scale_by_scale),
                "catalog_gaussian_smoothing_sigma": float(args.catalog_gaussian_smoothing_sigma),
                "catalog_gaussian_cov_floor": float(args.catalog_gaussian_cov_floor),
                "catalog_gaussian_shrinkage": float(args.catalog_gaussian_shrinkage),
                "trajectory_basis_family": str(args.trajectory_basis_family),
                "trajectory_basis_components": int(args.trajectory_basis_components),
                "trajectory_basis_smoothing_sigma": float(args.trajectory_basis_smoothing_sigma),
                "trajectory_basis_coeff_prior_var": float(args.trajectory_basis_coeff_prior_var),
                "catalog_residual_aggregation": str(args.catalog_residual_aggregation),
                "catalog_residual_top_k": int(args.catalog_residual_top_k),
                "catalog_residual_all_anchor_shrinkage": float(args.catalog_residual_all_anchor_shrinkage),
                "catalog_residual_anchor_smoothing_sigma": float(args.catalog_residual_anchor_smoothing_sigma),
                "catalog_residual_anchor_smoothing_schedule": str(args.catalog_residual_anchor_smoothing_schedule),
                "catalog_residual_refine_top_k": int(args.catalog_residual_refine_top_k),
                "quadratic_optimizer_max_iter": int(args.quadratic_optimizer_max_iter),
                "quadratic_continuation_scales": str(args.quadratic_continuation_scales),
                "quadratic_observation_scales": str(args.quadratic_observation_scales),
                "basis_source": str(loaded_basis_meta["basis_source"]),
                "basis_max_dim_requested": int(basis_max_dim_for_table),
                "basis_dim": int(loaded_basis_meta["basis_dim"]),
                "ridge": float(ridge_for_table),
                "continuous_posterior_temperature": float(posterior_temperature_for_table),
            }
            row = dict(base)
            for mode, scores in [
                ("known", vectors["known_scores"]),
                ("zero", vectors["zero_scores"]),
                ("joint", vectors["joint_scores"]),
                ("best_single_tau", vectors["best_single_tau_scores"]),
                ("continuous_joint", vectors["continuous_joint_scores"]),
            ]:
                row.update(_score_summary(mode, np.asarray(scores, dtype=np.float64), true_idx, ids))
            row["continuous_joint_minus_zero_true_score"] = float(vectors["continuous_joint_minus_zero_true_score"])
            row["continuous_joint_score_corr_with_zero"] = float(vectors["continuous_joint_score_corr_with_zero"])
            true_anchor_idx = int(true_idx)
            if str(args.continuous_score_mode) == "catalog_residual_profile":
                row["catalog_residual_true_best_anchor_index"] = int(
                    np.asarray(vectors["catalog_residual_best_anchor_indices"], dtype=np.int64)[true_anchor_idx]
                )
                row["catalog_residual_true_best_anchor_score"] = float(
                    np.asarray(vectors["catalog_residual_best_anchor_scores"], dtype=np.float64)[true_anchor_idx]
                )
                row["catalog_residual_true_anchor_logmean_score"] = float(
                    np.asarray(vectors["catalog_residual_anchor_logmean_scores"], dtype=np.float64)[true_anchor_idx]
                )
                row["catalog_residual_true_anchor_score_gap"] = float(
                    np.asarray(vectors["catalog_residual_anchor_score_gaps"], dtype=np.float64)[true_anchor_idx]
                )
                row["catalog_residual_true_anchor_aggregate_count"] = int(
                    np.asarray(vectors["catalog_residual_anchor_aggregate_counts"], dtype=np.int64)[true_anchor_idx]
                )
            row.update(vectors["trajectory_recovery"])
            trial_rows.append(row)

            for mode, scores_raw in [
                ("known", vectors["known_scores"]),
                ("zero", vectors["zero_scores"]),
                ("joint", vectors["joint_scores"]),
                ("best_single_tau", vectors["best_single_tau_scores"]),
                ("continuous_joint", vectors["continuous_joint_scores"]),
            ]:
                scores = np.asarray(scores_raw, dtype=np.float64)
                mode_temperature = float(posterior_temperature_for_table) if mode == "continuous_joint" else 1.0
                effective_scores = scores / mode_temperature
                posterior = posterior_from_log_scores(effective_scores)
                for candidate_index, candidate_id in enumerate(ids):
                    feature_rows.append(
                        {
                            **base,
                            "observer_mode": str(mode),
                            "posterior_temperature": float(mode_temperature),
                            "candidate_index": int(candidate_index),
                            "candidate_id": str(candidate_id),
                            "is_true_candidate": bool(int(candidate_index) == int(true_idx)),
                            "candidate_score": float(effective_scores[int(candidate_index)]),
                            "candidate_score_raw": float(scores[int(candidate_index)]),
                            "candidate_posterior": float(posterior[int(candidate_index)]),
                            "feature_source": "not_provided_candidate_posterior_only",
                            "catalog_residual_best_anchor_index": (
                                int(np.asarray(vectors["catalog_residual_best_anchor_indices"], dtype=np.int64)[candidate_index])
                                if mode == "continuous_joint"
                                and str(args.continuous_score_mode) == "catalog_residual_profile"
                                else -1
                            ),
                            "catalog_residual_best_anchor_score": (
                                float(
                                    np.asarray(vectors["catalog_residual_best_anchor_scores"], dtype=np.float64)[
                                        candidate_index
                                    ]
                                )
                                if mode == "continuous_joint"
                                and str(args.continuous_score_mode) == "catalog_residual_profile"
                                else float("nan")
                            ),
                            "catalog_residual_anchor_logmean_score": (
                                float(
                                    np.asarray(vectors["catalog_residual_anchor_logmean_scores"], dtype=np.float64)[
                                        candidate_index
                                    ]
                                )
                                if mode == "continuous_joint"
                                and str(args.continuous_score_mode) == "catalog_residual_profile"
                                else float("nan")
                            ),
                            "catalog_residual_anchor_score_gap": (
                                float(
                                    np.asarray(vectors["catalog_residual_anchor_score_gaps"], dtype=np.float64)[
                                        candidate_index
                                    ]
                                )
                                if mode == "continuous_joint"
                                and str(args.continuous_score_mode) == "catalog_residual_profile"
                                else float("nan")
                            ),
                            "catalog_residual_anchor_aggregate_count": (
                                int(
                                    np.asarray(vectors["catalog_residual_anchor_aggregate_counts"], dtype=np.int64)[
                                        candidate_index
                                    ]
                                )
                                if mode == "continuous_joint"
                                and str(args.continuous_score_mode) == "catalog_residual_profile"
                                else 0
                            ),
                        }
                    )

            rec = dict(base)
            rec.update(vectors["trajectory_recovery"])
            recovery_rows.append(rec)

            for fit_row in vectors["fit_rows"]:
                qrow = dict(base)
                qrow["qc_type"] = "A_I_fit"
                qrow.update(fit_row)
                qc_rows.append(qrow)
            tau_hat = np.asarray(vectors["filtered_state_means"], dtype=np.float64)[int(true_idx)]
            temporal_std = float(np.mean(np.std(tau_hat, axis=0))) if tau_hat.size else float("nan")
            collapse_row = dict(base)
            collapse_row.update(
                {
                    "qc_type": "signal_nuisance_collapse",
                    "trajectory_hat_rms": float(np.sqrt(np.mean(tau_hat * tau_hat))) if tau_hat.size else float("nan"),
                    "trajectory_hat_temporal_std_mean": temporal_std,
                    "continuous_joint_score_corr_with_zero": float(vectors["continuous_joint_score_corr_with_zero"]),
                    "continuous_joint_minus_zero_true_score": float(vectors["continuous_joint_minus_zero_true_score"]),
                    "flat_trajectory_hat": bool(np.isfinite(temporal_std) and temporal_std < 1e-6),
                    "zero_like_score_vector": bool(
                        np.isfinite(float(vectors["continuous_joint_score_corr_with_zero"]))
                        and float(vectors["continuous_joint_score_corr_with_zero"]) > 0.99
                    ),
                }
            )
            qc_rows.append(collapse_row)
        if table_index == 1 or table_index == manifest.shape[0] or table_index % progress_every == 0:
            print(f"[continuous-joint] scored {table_index}/{manifest.shape[0]} response tables", flush=True)

    summary_rows = _summary_rows(trial_rows)
    _write_csv(out_dir / "continuous_joint_trials.csv", trial_rows)
    _write_csv(out_dir / "continuous_joint_summary.csv", summary_rows)
    _write_csv(out_dir / "continuous_joint_feature_posterior.csv", feature_rows)
    _write_csv(out_dir / "continuous_joint_trajectory_recovery.csv", recovery_rows)
    _write_csv(out_dir / "continuous_joint_qc.csv", qc_rows)
    _write_json(
        out_dir / "continuous_joint_metadata.json",
        {
            "run_dir": run_dir,
            "response_manifest": manifest_path,
            "prior_family_filter": sorted(prior_family_filter),
            "scale_filter": sorted(scale_filter),
            "skip_tables": int(skip_tables),
            "n_response_tables": int(manifest.shape[0]),
            "likelihood_scales": likelihood_scales,
            "continuous_posterior_temperature": float(posterior_temperature),
            "continuous_posterior_temperature_by_scale": posterior_temperature_by_scale,
            "basis": basis_meta or {},
            "basis_max_dim": int(args.basis_max_dim),
            "basis_max_dim_by_scale": basis_max_dim_by_scale,
            "ridge": float(args.ridge),
            "ridge_by_scale": ridge_by_scale,
            "observation_model": str(args.observation_model),
            "time_smoothing_sigma": float(args.time_smoothing_sigma),
            "time_shrinkage": float(args.time_shrinkage),
            "continuous_score_mode": str(args.continuous_score_mode),
            "trajectory_prior_mean": str(args.trajectory_prior_mean),
            "trajectory_initial_position": str(args.trajectory_initial_position),
            "trajectory_initial_position_var": float(args.trajectory_initial_position_var),
            "trajectory_process_model": str(args.trajectory_process_model),
            "trajectory_process_model_by_scale": trajectory_process_model_by_scale,
            "brownian_cov_floor": float(args.brownian_cov_floor),
            "brownian_cov_scale": float(args.brownian_cov_scale),
            "brownian_cov_scale_by_scale": brownian_cov_scale_by_scale,
            "catalog_gaussian_smoothing_sigma": float(args.catalog_gaussian_smoothing_sigma),
            "catalog_gaussian_cov_floor": float(args.catalog_gaussian_cov_floor),
            "catalog_gaussian_shrinkage": float(args.catalog_gaussian_shrinkage),
            "trajectory_basis_family": str(args.trajectory_basis_family),
            "trajectory_basis_components": int(args.trajectory_basis_components),
            "trajectory_basis_smoothing_sigma": float(args.trajectory_basis_smoothing_sigma),
            "trajectory_basis_coeff_prior_var": float(args.trajectory_basis_coeff_prior_var),
            "catalog_residual_aggregation": str(args.catalog_residual_aggregation),
            "catalog_residual_top_k": int(args.catalog_residual_top_k),
            "catalog_residual_all_anchor_shrinkage": float(args.catalog_residual_all_anchor_shrinkage),
            "catalog_residual_anchor_smoothing_sigma": float(args.catalog_residual_anchor_smoothing_sigma),
            "catalog_residual_anchor_smoothing_schedule": str(args.catalog_residual_anchor_smoothing_schedule),
            "catalog_residual_refine_top_k": int(args.catalog_residual_refine_top_k),
            "quadratic_optimizer_max_iter": int(args.quadratic_optimizer_max_iter),
            "quadratic_continuation_scales": str(args.quadratic_continuation_scales),
            "quadratic_observation_scales": str(args.quadratic_observation_scales),
            "quadratic_intercept_ridge_multiplier": float(args.quadratic_intercept_ridge_multiplier),
            "quadratic_affine_intercept_scale": float(args.quadratic_affine_intercept_scale),
            "trajectory_source": str(args.trajectory_npz) if args.trajectory_npz is not None else "response_table_keys",
            "trajectory_sidecar_dir": str(args.trajectory_sidecar_dir) if args.trajectory_sidecar_dir is not None else "",
            "outputs": [
                "continuous_joint_trials.csv",
                "continuous_joint_summary.csv",
                "continuous_joint_feature_posterior.csv",
                "continuous_joint_trajectory_recovery.csv",
                "continuous_joint_qc.csv",
                "continuous_joint_metadata.json",
                "continuous_joint_report.md",
            ],
        },
    )
    report = [
        "# Continuous Joint Trajectory Observer",
        "",
        f"- Response tables scored: {manifest.shape[0]}",
        f"- Basis source: {(basis_meta or {}).get('basis_source', 'identity_units')}",
        f"- Basis dim: {(basis_meta or {}).get('basis_dim', 'unknown')}",
        f"- Basis max dim: {int(args.basis_max_dim)}",
        f"- Basis max dim by scale: {basis_max_dim_by_scale if basis_max_dim_by_scale else 'none'}",
        f"- Ridge: {float(args.ridge):.6g}",
        f"- Ridge by scale: {ridge_by_scale if ridge_by_scale else 'none'}",
        f"- Continuous posterior temperature: {float(posterior_temperature):.6g}",
        f"- Continuous posterior temperature by scale: {posterior_temperature_by_scale if posterior_temperature_by_scale else 'none'}",
        f"- Kalman alpha: {float(args.alpha):.6g}",
        f"- Kalman process variance: {float(args.process_var):.6g}",
        f"- Prior family filter: {','.join(sorted(prior_family_filter)) if prior_family_filter else 'none'}",
        f"- Scale filter: {','.join(sorted(scale_filter)) if scale_filter else 'none'}",
        f"- Observation model: {str(args.observation_model)}",
        f"- Time smoothing sigma: {float(args.time_smoothing_sigma):.6g}",
        f"- Time shrinkage: {float(args.time_shrinkage):.6g}",
        f"- Continuous score mode: {str(args.continuous_score_mode)}",
        f"- Trajectory prior mean: {str(args.trajectory_prior_mean)}",
        f"- Trajectory initial position: {str(args.trajectory_initial_position)}",
        f"- Trajectory initial position variance: {float(args.trajectory_initial_position_var):.6g}",
        f"- Trajectory process model: {str(args.trajectory_process_model)}",
        f"- Trajectory process model by scale: {trajectory_process_model_by_scale if trajectory_process_model_by_scale else 'none'}",
        f"- Brownian covariance floor: {float(args.brownian_cov_floor):.6g}",
        f"- Brownian covariance scale: {float(args.brownian_cov_scale):.6g}",
        f"- Brownian covariance scale by scale: {brownian_cov_scale_by_scale if brownian_cov_scale_by_scale else 'none'}",
        f"- Catalog Gaussian smoothing sigma: {float(args.catalog_gaussian_smoothing_sigma):.6g}",
        f"- Catalog Gaussian covariance floor: {float(args.catalog_gaussian_cov_floor):.6g}",
        f"- Catalog Gaussian covariance shrinkage: {float(args.catalog_gaussian_shrinkage):.6g}",
        f"- Trajectory basis family: {str(args.trajectory_basis_family)}",
        f"- Trajectory basis components: {int(args.trajectory_basis_components)}",
        f"- Trajectory basis smoothing sigma: {float(args.trajectory_basis_smoothing_sigma):.6g}",
        f"- Trajectory basis coefficient prior variance: {float(args.trajectory_basis_coeff_prior_var):.6g}",
        f"- Catalog residual aggregation: {str(args.catalog_residual_aggregation)}",
        f"- Catalog residual top-k: {int(args.catalog_residual_top_k)}",
        f"- Catalog residual all-anchor shrinkage: {float(args.catalog_residual_all_anchor_shrinkage):.6g}",
        f"- Catalog residual anchor smoothing sigma: {float(args.catalog_residual_anchor_smoothing_sigma):.6g}",
        f"- Catalog residual anchor smoothing schedule: {str(args.catalog_residual_anchor_smoothing_schedule) or 'single'}",
        f"- Catalog residual refine top-k: {int(args.catalog_residual_refine_top_k)}",
        f"- Quadratic optimizer max iterations: {int(args.quadratic_optimizer_max_iter)}",
        f"- Quadratic continuation scales: {str(args.quadratic_continuation_scales)}",
        f"- Quadratic observation scales: {str(args.quadratic_observation_scales)}",
        f"- Quadratic intercept ridge multiplier: {float(args.quadratic_intercept_ridge_multiplier):.6g}",
        f"- Quadratic affine intercept score scale: {float(args.quadratic_affine_intercept_scale):.6g}",
        "",
        "Primary files:",
        "- `continuous_joint_trials.csv`",
        "- `continuous_joint_summary.csv`",
        "- `continuous_joint_feature_posterior.csv`",
        "- `continuous_joint_trajectory_recovery.csv`",
        "- `continuous_joint_qc.csv`",
        "- `continuous_joint_metadata.json`",
    ]
    (out_dir / "continuous_joint_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return out_dir


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":  # pragma: no cover
    main()
