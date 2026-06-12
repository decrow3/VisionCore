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
