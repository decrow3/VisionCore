"""Information metrics for Vernier finite-difference responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InformationResult:
    fisher_per_bin: np.ndarray
    cumulative_fisher: np.ndarray
    dprime2_per_bin: np.ndarray
    cumulative_dprime2: np.ndarray
    threshold_proxy: np.ndarray
    spike_count: np.ndarray


def finite_difference_derivative(plus: np.ndarray, minus: np.ndarray, step_arcmin: float) -> np.ndarray:
    return (np.asarray(plus, dtype=np.float64) - np.asarray(minus, dtype=np.float64)) / (2.0 * float(step_arcmin))


def expected_counts(rates: np.ndarray, bin_seconds: float) -> np.ndarray:
    return np.asarray(rates, dtype=np.float64) * float(bin_seconds)


def poisson_fisher_counts(
    plus_counts: np.ndarray,
    minus_counts: np.ndarray,
    *,
    step_arcmin: float,
    epsilon: float = 1e-8,
    phi: float = 1.0,
    fixed_diag: np.ndarray | None = None,
) -> InformationResult:
    """Pose-aware additive Fisher under a diagonal count-noise model.

    ``plus_counts`` and ``minus_counts`` are shaped ``(T, units)``.  The
    cumulative arrays sum Fisher across time bins, matching a block-diagonal
    pose-aware observer.
    """
    plus = np.asarray(plus_counts, dtype=np.float64)
    minus = np.asarray(minus_counts, dtype=np.float64)
    if plus.shape != minus.shape or plus.ndim != 2:
        raise ValueError(f"plus/minus must both be (T, units), got {plus.shape} and {minus.shape}")
    deriv = finite_difference_derivative(plus, minus, step_arcmin)
    delta = plus - minus
    mean_counts = np.maximum((plus + minus) / 2.0, 0.0)
    if fixed_diag is None:
        diag = np.maximum(mean_counts, 0.0) + float(epsilon)
    else:
        fd = np.asarray(fixed_diag, dtype=np.float64)
        diag = np.broadcast_to(fd, plus.shape).copy() + float(epsilon)
    diag = np.maximum(float(phi) * diag, float(epsilon))
    fisher_per_bin = np.sum((deriv * deriv) / diag, axis=1)
    dprime2_per_bin = np.sum((delta * delta) / diag, axis=1)
    cumulative_fisher = np.cumsum(fisher_per_bin)
    cumulative_dprime2 = np.cumsum(dprime2_per_bin)
    return InformationResult(
        fisher_per_bin=fisher_per_bin,
        cumulative_fisher=cumulative_fisher,
        dprime2_per_bin=dprime2_per_bin,
        cumulative_dprime2=cumulative_dprime2,
        threshold_proxy=1.0 / np.sqrt(cumulative_fisher + float(epsilon)),
        spike_count=np.cumsum(np.sum(mean_counts, axis=1)),
    )


def pose_blind_diagonal_fisher(
    plus_trials: list[np.ndarray],
    minus_trials: list[np.ndarray],
    *,
    step_arcmin: float,
    bin_seconds: float,
    epsilon: float = 1e-8,
    phi: float = 1.0,
) -> dict[str, Any]:
    """Pose-blind diagonal Fisher using count noise plus marginal covariance.

    The observer receives pooled responses without explicit phase labels.  For a
    first-pass robust diagnostic, this uses an additive diagonal approximation:
    expected Poisson count noise plus across-trial/trajectory response variance
    at each time bin.  The count-noise term prevents deterministic no-motion
    controls from getting artificially huge Fisher values solely because their
    across-trace marginal variance is near zero.
    """
    t_min = min(min(arr.shape[0] for arr in plus_trials), min(arr.shape[0] for arr in minus_trials))
    plus = np.stack([expected_counts(arr[:t_min], bin_seconds) for arr in plus_trials], axis=0)
    minus = np.stack([expected_counts(arr[:t_min], bin_seconds) for arr in minus_trials], axis=0)
    mean_plus = np.mean(plus, axis=0)
    mean_minus = np.mean(minus, axis=0)
    deriv = finite_difference_derivative(mean_plus, mean_minus, step_arcmin)
    delta = mean_plus - mean_minus
    pooled = np.concatenate([plus, minus], axis=0)
    marginal_var = np.var(pooled, axis=0, ddof=1) if pooled.shape[0] > 1 else 0.0
    count_noise = np.maximum((mean_plus + mean_minus) / 2.0, 0.0)
    diag = np.maximum(float(phi) * (count_noise + marginal_var), float(epsilon))
    fisher_per_bin = np.sum((deriv * deriv) / diag, axis=1)
    dprime2_per_bin = np.sum((delta * delta) / diag, axis=1)
    return {
        "fisher_per_bin": fisher_per_bin,
        "cumulative_fisher": np.cumsum(fisher_per_bin),
        "dprime2_per_bin": dprime2_per_bin,
        "cumulative_dprime2": np.cumsum(dprime2_per_bin),
        "threshold_proxy": 1.0 / np.sqrt(np.cumsum(fisher_per_bin) + float(epsilon)),
    }


def pose_uncertain_diagonal_fisher(
    plus_trials: list[np.ndarray],
    minus_trials: list[np.ndarray],
    pose_trials: list[np.ndarray],
    *,
    step_arcmin: float,
    bin_seconds: float,
    sigma_pose_arcmin: float,
    epsilon: float = 1e-8,
    phi: float = 1.0,
) -> dict[str, Any]:
    """Diagonal Fisher with Gaussian uncertainty over retinal pose.

    ``sigma_pose_arcmin=0`` approximates the pose-aware per-trace average.
    Large ``sigma_pose_arcmin`` approaches the pose-blind diagonal marginal
    calculation because all trace poses receive nearly uniform weight.
    """
    t_min = min(
        min(arr.shape[0] for arr in plus_trials),
        min(arr.shape[0] for arr in minus_trials),
        min(arr.shape[0] for arr in pose_trials),
    )
    plus = np.stack([expected_counts(arr[:t_min], bin_seconds) for arr in plus_trials], axis=0)
    minus = np.stack([expected_counts(arr[:t_min], bin_seconds) for arr in minus_trials], axis=0)
    poses_arcmin = np.stack([np.asarray(arr[:t_min], dtype=np.float64) * 60.0 for arr in pose_trials], axis=0)
    deriv = finite_difference_derivative(plus, minus, step_arcmin)
    delta = plus - minus
    mean_counts = np.maximum((plus + minus) / 2.0, 0.0)
    n, t, _u = plus.shape
    fisher_per_bin = np.zeros(t, dtype=np.float64)
    dprime2_per_bin = np.zeros(t, dtype=np.float64)
    sigma = float(sigma_pose_arcmin)
    for ti in range(t):
        if sigma <= 0.0:
            weights = np.eye(n, dtype=np.float64)
        elif not np.isfinite(sigma) or sigma > 1e5:
            weights = np.full((n, n), 1.0 / max(n, 1), dtype=np.float64)
        else:
            diff = poses_arcmin[:, ti, :][:, None, :] - poses_arcmin[:, ti, :][None, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            weights = np.exp(-0.5 * dist2 / max(sigma * sigma, 1e-12))
            weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)
        dbar = weights @ deriv[:, ti, :]
        deltabar = weights @ delta[:, ti, :]
        count = weights @ mean_counts[:, ti, :]
        mean_plus = weights @ plus[:, ti, :]
        mean_minus = weights @ minus[:, ti, :]
        var_plus = weights @ (plus[:, ti, :] * plus[:, ti, :]) - mean_plus * mean_plus
        var_minus = weights @ (minus[:, ti, :] * minus[:, ti, :]) - mean_minus * mean_minus
        nuisance = np.maximum(0.5 * (var_plus + var_minus), 0.0)
        diag = np.maximum(float(phi) * (count + nuisance), float(epsilon))
        fisher_per_sample = np.sum((dbar * dbar) / diag, axis=1)
        dprime2_per_sample = np.sum((deltabar * deltabar) / diag, axis=1)
        fisher_per_bin[ti] = float(np.mean(fisher_per_sample))
        dprime2_per_bin[ti] = float(np.mean(dprime2_per_sample))
    cumulative = np.cumsum(fisher_per_bin)
    return {
        "fisher_per_bin": fisher_per_bin,
        "cumulative_fisher": cumulative,
        "dprime2_per_bin": dprime2_per_bin,
        "cumulative_dprime2": np.cumsum(dprime2_per_bin),
        "threshold_proxy": 1.0 / np.sqrt(cumulative + float(epsilon)),
    }


def pose_blind_full_covariance_fisher(
    plus_trials: list[np.ndarray],
    minus_trials: list[np.ndarray],
    *,
    step_arcmin: float,
    bin_seconds: float,
    epsilon: float = 1e-8,
    shrinkage: float = 0.1,
) -> dict[str, Any]:
    """Pose-blind full-covariance Fisher with diagonal count-noise floor."""
    t_min = min(min(arr.shape[0] for arr in plus_trials), min(arr.shape[0] for arr in minus_trials))
    plus = np.stack([expected_counts(arr[:t_min], bin_seconds) for arr in plus_trials], axis=0)
    minus = np.stack([expected_counts(arr[:t_min], bin_seconds) for arr in minus_trials], axis=0)
    fisher, dprime = _full_covariance_curves(
        plus,
        minus,
        step_arcmin=float(step_arcmin),
        epsilon=float(epsilon),
        shrinkage=float(shrinkage),
    )
    return {
        "fisher_per_bin": fisher,
        "cumulative_fisher": np.cumsum(fisher),
        "dprime2_per_bin": dprime,
        "cumulative_dprime2": np.cumsum(dprime),
        "threshold_proxy": 1.0 / np.sqrt(np.cumsum(fisher) + float(epsilon)),
    }


def compact_aware_pose_blind_fisher(
    plus_trials: list[np.ndarray],
    minus_trials: list[np.ndarray],
    *,
    step_arcmin: float,
    bin_seconds: float,
    k_list: list[int],
    alpha_list: list[float],
    subspace_sources: list[str],
    epsilon: float = 1e-8,
    shrinkage: float = 0.1,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Hard-projection and soft-discount pose-blind Fisher diagnostics."""
    t_min = min(min(arr.shape[0] for arr in plus_trials), min(arr.shape[0] for arr in minus_trials))
    plus = np.stack([expected_counts(arr[:t_min], bin_seconds) for arr in plus_trials], axis=0)
    minus = np.stack([expected_counts(arr[:t_min], bin_seconds) for arr in minus_trials], axis=0)
    rows: list[dict[str, Any]] = []
    for source in subspace_sources:
        source = str(source)
        if source == "natural_translation_covariance":
            continue
        for k in k_list:
            k = int(k)
            hard_fisher, hard_dprime = _compact_curves(
                plus,
                minus,
                step_arcmin=float(step_arcmin),
                source=source,
                k=k,
                mode="project",
                alpha=1.0,
                epsilon=float(epsilon),
                shrinkage=float(shrinkage),
                seed=int(seed),
            )
            rows.append(
                {
                    "readout": f"pose_blind_compact_project_k{k}",
                    "compact_mode": "hard_project",
                    "subspace_source": source,
                    "compact_k": k,
                    "compact_alpha": float("nan"),
                    "fisher_per_bin": hard_fisher,
                    "cumulative_fisher": np.cumsum(hard_fisher),
                    "dprime2_per_bin": hard_dprime,
                    "cumulative_dprime2": np.cumsum(hard_dprime),
                    "threshold_proxy": 1.0 / np.sqrt(np.cumsum(hard_fisher) + float(epsilon)),
                }
            )
            for alpha in alpha_list:
                alpha = float(alpha)
                soft_fisher, soft_dprime = _compact_curves(
                    plus,
                    minus,
                    step_arcmin=float(step_arcmin),
                    source=source,
                    k=k,
                    mode="discount",
                    alpha=alpha,
                    epsilon=float(epsilon),
                    shrinkage=float(shrinkage),
                    seed=int(seed),
                )
                rows.append(
                    {
                        "readout": f"pose_blind_compact_discount_k{k}_alpha{alpha:g}",
                        "compact_mode": "soft_discount",
                        "subspace_source": source,
                        "compact_k": k,
                        "compact_alpha": alpha,
                        "fisher_per_bin": soft_fisher,
                        "cumulative_fisher": np.cumsum(soft_fisher),
                        "dprime2_per_bin": soft_dprime,
                        "cumulative_dprime2": np.cumsum(soft_dprime),
                        "threshold_proxy": 1.0 / np.sqrt(np.cumsum(soft_fisher) + float(epsilon)),
                    }
                )
    return rows


def _covariance_with_count_noise(
    plus_t: np.ndarray,
    minus_t: np.ndarray,
    *,
    shrinkage: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_plus = np.mean(plus_t, axis=0)
    mean_minus = np.mean(minus_t, axis=0)
    pooled = np.concatenate([plus_t, minus_t], axis=0)
    cov = np.cov(pooled, rowvar=False, ddof=1) if pooled.shape[0] > 1 else np.zeros((pooled.shape[1], pooled.shape[1]))
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    count = np.maximum((mean_plus + mean_minus) / 2.0, 0.0)
    cov = cov + np.diag(count + float(epsilon))
    diag_mean = float(np.mean(np.diag(cov))) if cov.size else 1.0
    lam = float(np.clip(shrinkage, 0.0, 1.0))
    cov = (1.0 - lam) * cov + lam * diag_mean * np.eye(cov.shape[0], dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    return cov, mean_plus, mean_minus


def _full_covariance_curves(
    plus: np.ndarray,
    minus: np.ndarray,
    *,
    step_arcmin: float,
    epsilon: float,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    t = plus.shape[1]
    fisher = np.zeros(t, dtype=np.float64)
    dprime = np.zeros(t, dtype=np.float64)
    for ti in range(t):
        cov, mean_plus, mean_minus = _covariance_with_count_noise(
            plus[:, ti, :],
            minus[:, ti, :],
            shrinkage=shrinkage,
            epsilon=epsilon,
        )
        deriv = finite_difference_derivative(mean_plus, mean_minus, step_arcmin)
        delta = mean_plus - mean_minus
        fisher[ti] = _quadratic_form(cov, deriv)
        dprime[ti] = _quadratic_form(cov, delta)
    return fisher, dprime


def _compact_curves(
    plus: np.ndarray,
    minus: np.ndarray,
    *,
    step_arcmin: float,
    source: str,
    k: int,
    mode: str,
    alpha: float,
    epsilon: float,
    shrinkage: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    t = plus.shape[1]
    fisher = np.zeros(t, dtype=np.float64)
    dprime = np.zeros(t, dtype=np.float64)
    rng = np.random.default_rng(int(seed) + int(k) * 1009 + sum(ord(c) for c in str(source)))
    for ti in range(t):
        cov, mean_plus, mean_minus = _covariance_with_count_noise(
            plus[:, ti, :],
            minus[:, ti, :],
            shrinkage=shrinkage,
            epsilon=epsilon,
        )
        deriv = finite_difference_derivative(mean_plus, mean_minus, step_arcmin)
        delta = mean_plus - mean_minus
        u = _nuisance_basis(plus[:, ti, :], minus[:, ti, :], source=source, k=int(k), rng=rng)
        if u.shape[1] == 0:
            fisher[ti] = _quadratic_form(cov, deriv)
            dprime[ti] = _quadratic_form(cov, delta)
            continue
        if mode == "project":
            q = _orthogonal_complement(u, cov.shape[0])
            if q.shape[1] == 0:
                fisher[ti] = 0.0
                dprime[ti] = 0.0
                continue
            cov_p = q.T @ cov @ q
            fisher[ti] = _quadratic_form(cov_p, q.T @ deriv)
            dprime[ti] = _quadratic_form(cov_p, q.T @ delta)
        else:
            fisher[ti] = _discounted_precision_quadratic_form(cov, u, deriv, alpha=float(alpha), epsilon=float(epsilon))
            dprime[ti] = _discounted_precision_quadratic_form(cov, u, delta, alpha=float(alpha), epsilon=float(epsilon))
    return fisher, dprime


def _nuisance_basis(plus_t: np.ndarray, minus_t: np.ndarray, *, source: str, k: int, rng: np.random.Generator) -> np.ndarray:
    u_dim = plus_t.shape[1]
    k = min(max(int(k), 0), u_dim)
    if k == 0:
        return np.zeros((u_dim, 0), dtype=np.float64)
    if source == "random_orthonormal":
        q, _ = np.linalg.qr(rng.standard_normal((u_dim, k)))
        return q[:, :k]
    if source == "real_fem_trajectory_covariance":
        mat = 0.5 * (np.asarray(plus_t, dtype=np.float64) + np.asarray(minus_t, dtype=np.float64))
    else:
        mat = np.concatenate([plus_t, minus_t], axis=0)
    mat = mat - np.mean(mat, axis=0, keepdims=True)
    if mat.shape[0] < 2:
        return np.zeros((u_dim, 0), dtype=np.float64)
    cov = np.cov(mat, rowvar=False, ddof=1)
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    rank = int(np.sum(vals > max(float(vals[0]) if vals.size else 0.0, 1.0) * 1e-10))
    return vecs[:, : min(k, rank, vecs.shape[1])]


def _orthogonal_complement(u: np.ndarray, n: int) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    if u.size == 0:
        return np.eye(int(n), dtype=np.float64)
    q, _ = np.linalg.qr(u, mode="complete")
    return q[:, u.shape[1] :]


def _discounted_precision_quadratic_form(
    cov: np.ndarray,
    u: np.ndarray,
    vec: np.ndarray,
    *,
    alpha: float,
    epsilon: float,
) -> float:
    """Quadratic form after attenuating precision in the nuisance subspace.

    ``alpha=1`` is the ordinary full-covariance observer. ``alpha=0`` removes
    the positive-semidefinite precision component assigned to ``span(u)``, so
    nuisance modes cannot gain precision by being variance-shrunk.
    """
    vec = np.asarray(vec, dtype=np.float64).ravel()
    keep = np.isfinite(vec)
    if int(np.sum(keep)) == 0:
        return float("nan")
    cov = np.asarray(cov, dtype=np.float64)[np.ix_(keep, keep)]
    u = np.asarray(u, dtype=np.float64)
    if u.size == 0:
        return _quadratic_form(cov, vec[keep])
    u = u[keep, :]
    alpha = float(np.clip(alpha, 0.0, 1.0))
    q, r = np.linalg.qr(u)
    diag = np.abs(np.diag(r)) if r.ndim == 2 else np.zeros(0, dtype=np.float64)
    rank = int(np.sum(diag > max(float(np.max(diag)) if diag.size else 0.0, 1.0) * 1e-10))
    if rank == 0:
        return _quadratic_form(cov, vec[keep])
    q = q[:, :rank]
    v = vec[keep]
    try:
        precision = np.linalg.solve(cov, np.eye(cov.shape[0], dtype=np.float64))
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(cov)
    precision = 0.5 * (precision + precision.T)
    nuisance_precision = precision @ q
    middle = q.T @ nuisance_precision
    middle = 0.5 * (middle + middle.T)
    try:
        middle_inv = np.linalg.solve(middle, np.eye(middle.shape[0], dtype=np.float64))
    except np.linalg.LinAlgError:
        middle_inv = np.linalg.pinv(middle)
    precision_discounted = precision - (1.0 - alpha) * (nuisance_precision @ middle_inv @ nuisance_precision.T)
    precision_discounted = 0.5 * (precision_discounted + precision_discounted.T)
    return max(float(v @ precision_discounted @ v), 0.0)


def _quadratic_form(cov: np.ndarray, vec: np.ndarray) -> float:
    vec = np.asarray(vec, dtype=np.float64).ravel()
    keep = np.isfinite(vec)
    if int(np.sum(keep)) == 0:
        return float("nan")
    cov = np.asarray(cov, dtype=np.float64)[np.ix_(keep, keep)]
    v = vec[keep]
    try:
        sol = np.linalg.solve(cov, v)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(cov) @ v
    return float(np.dot(v, sol))


def summarize_information(result: InformationResult | dict[str, Any], *, prefix: str = "") -> dict[str, float]:
    """Compact summary row for CSV output."""
    get = (lambda key: getattr(result, key)) if isinstance(result, InformationResult) else (lambda key: result[key])
    cumulative = np.asarray(get("cumulative_fisher"), dtype=np.float64)
    dprime = np.asarray(get("cumulative_dprime2"), dtype=np.float64)
    thresh = np.asarray(get("threshold_proxy"), dtype=np.float64)
    final = float(cumulative[-1]) if cumulative.size else float("nan")
    half_idx = int(np.searchsorted(cumulative, 0.5 * final, side="left")) if np.isfinite(final) and final > 0 else -1
    return {
        f"{prefix}final_fisher": final,
        f"{prefix}final_dprime2": float(dprime[-1]) if dprime.size else float("nan"),
        f"{prefix}final_threshold_proxy": float(thresh[-1]) if thresh.size else float("nan"),
        f"{prefix}timebin_to_half_final": half_idx,
        f"{prefix}early_slope_first5": float(np.mean(np.diff(cumulative[:6]))) if cumulative.size >= 6 else float("nan"),
    }
