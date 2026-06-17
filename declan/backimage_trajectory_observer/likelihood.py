"""Likelihood utilities for BackImage trajectory-table observers."""

from __future__ import annotations

import numpy as np


def logsumexp(values: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray | float:
    """Numerically stable log-sum-exp."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        if axis is None:
            return float("-inf")
        return np.full(np.sum(arr, axis=axis, keepdims=keepdims).shape, -np.inf, dtype=np.float64)
    maxv = np.max(arr, axis=axis, keepdims=True)
    finite = np.isfinite(maxv)
    shifted = np.where(finite, arr - maxv, -np.inf)
    out = maxv + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    out = np.where(finite, out, -np.inf)
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    if axis is None and not keepdims:
        return float(np.asarray(out).squeeze())
    return out


def logmeanexp(values: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray | float:
    """Numerically stable log-mean-exp."""
    arr = np.asarray(values, dtype=np.float64)
    if axis is None:
        n = int(arr.size)
    else:
        n = int(arr.shape[int(axis)])
    if n <= 0:
        raise ValueError("logmeanexp requires at least one value")
    return logsumexp(arr, axis=axis, keepdims=keepdims) - float(np.log(n))


def normalized_log_weights(log_weights: np.ndarray | None, n: int) -> np.ndarray:
    """Return normalized log weights of length ``n``."""
    n = int(n)
    if n <= 0:
        raise ValueError("trajectory prior requires at least one trajectory")
    if log_weights is None:
        return np.full(n, -float(np.log(n)), dtype=np.float64)
    weights = np.asarray(log_weights, dtype=np.float64)
    if weights.shape != (n,):
        raise ValueError(f"log_weights must have shape {(n,)}, got {weights.shape}")
    if not np.isfinite(weights).any():
        raise ValueError("log_weights must contain at least one finite entry")
    return weights - float(logsumexp(weights))


def poisson_expected_count_loglik(
    y_obs_counts: np.ndarray,
    lambda_counts: np.ndarray,
    *,
    eps: float = 1e-8,
    likelihood_scale: float = 1.0,
) -> np.ndarray:
    """Poisson expected-count score up to the observation-only constant.

    ``y_obs_counts`` must be ``(time, units)``. ``lambda_counts`` can have any
    leading dimensions, but its trailing dimensions must match
    ``y_obs_counts``. Predictions are clipped to ``eps`` before taking logs.
    """
    obs = np.asarray(y_obs_counts, dtype=np.float64)
    pred = np.asarray(lambda_counts, dtype=np.float64)
    if obs.ndim != 2:
        raise ValueError(f"y_obs_counts must be (time, units), got {obs.shape}")
    if pred.ndim < 2:
        raise ValueError(f"lambda_counts must have at least 2 dimensions, got {pred.shape}")
    if pred.shape[-2:] != obs.shape:
        raise ValueError(f"lambda_counts trailing shape {pred.shape[-2:]} does not match y_obs_counts {obs.shape}")
    if not np.isfinite(obs).all():
        raise ValueError("y_obs_counts contains non-finite values")
    if not np.isfinite(pred).all():
        raise ValueError("lambda_counts contains non-finite values")
    if np.any(obs < 0.0):
        raise ValueError("y_obs_counts contains negative values")
    if np.any(pred < 0.0):
        raise ValueError("lambda_counts contains negative values")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")
    if float(likelihood_scale) <= 0.0:
        raise ValueError("likelihood_scale must be positive")
    safe = np.maximum(pred, float(eps))
    score = np.sum(obs * np.log(safe) - safe, axis=(-2, -1))
    return float(likelihood_scale) * score


def posterior_from_log_scores(log_scores: np.ndarray) -> np.ndarray:
    """Normalize log scores into posterior probabilities."""
    scores = np.asarray(log_scores, dtype=np.float64)
    norm = logsumexp(scores)
    if not np.isfinite(norm):
        return np.full(scores.shape, np.nan, dtype=np.float64)
    return np.exp(scores - float(norm))


def effective_count(probabilities: np.ndarray) -> float:
    """Inverse participation ratio for a probability vector."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.size == 0 or not np.isfinite(probs).all():
        return float("nan")
    denom = float(np.sum(probs * probs))
    return 1.0 / denom if denom > 0.0 else float("nan")


def entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy with zero-probability entries ignored."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.size == 0 or not np.isfinite(probs).all():
        return float("nan")
    pos = probs > 0.0
    return float(-np.sum(probs[pos] * np.log(probs[pos])))


def rank_desc(values: np.ndarray, index: int) -> float:
    """One-based descending rank of ``index`` in ``values``."""
    vals = np.asarray(values, dtype=np.float64)
    idx = int(index)
    if idx < 0 or idx >= vals.shape[0] or not np.isfinite(vals[idx]):
        return float("nan")
    return float(1 + np.sum(vals > vals[idx]))


def true_margin(scores: np.ndarray, true_index: int) -> float:
    """Score margin of true candidate over best non-true candidate."""
    vals = np.asarray(scores, dtype=np.float64)
    idx = int(true_index)
    if idx < 0 or idx >= vals.shape[0] or not np.isfinite(vals[idx]):
        return float("nan")
    if vals.shape[0] <= 1:
        return float("nan")
    mask = np.ones(vals.shape[0], dtype=bool)
    mask[idx] = False
    other = vals[mask]
    if not np.isfinite(other).any():
        return float("nan")
    return float(vals[idx] - np.max(other))
