"""Exact cached-trajectory Vernier observer.

This observer treats the cached ConvGRU response for each empirical eye
trajectory as an emission table.  The Vernier sign is the target latent, and
the trajectory index is marginalized as nuisance state.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .joint_observer import THETA_LABELS, THETA_MINUS, THETA_PLUS, logsumexp

SUPPORTED_TABLE_LIKELIHOODS = ("poisson", "residual", "gaussian_full")


def table_score_family(likelihood_model: str) -> str:
    """Human-readable score family for cached trajectory-table outputs."""
    model = str(likelihood_model)
    if model == "poisson":
        return "poisson_log_likelihood"
    if model == "residual":
        return "mahalanobis_residual_score"
    if model == "gaussian_full":
        return "gaussian_log_likelihood"
    raise ValueError(f"Unsupported trajectory-table likelihood {likelihood_model!r}; expected {SUPPORTED_TABLE_LIKELIHOODS}")


def diagonal_count_log_likelihood(
    observed_counts: np.ndarray,
    predicted_counts: np.ndarray,
    *,
    phi: float,
    normalization: str = "poisson",
    likelihood_scale: float = 1.0,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Score observed counts under diagonal Gaussian count noise.

    ``normalization="poisson"`` returns the Poisson count log likelihood up to
    the observation-only log-factorial constant. ``residual`` returns a
    Mahalanobis residual score, and ``gaussian_full`` returns a diagonal
    Gaussian log likelihood.
    """
    score_family = table_score_family(str(normalization))
    obs = np.asarray(observed_counts, dtype=np.float64)
    pred = np.asarray(predicted_counts, dtype=np.float64)
    if obs.ndim != 2:
        raise ValueError(f"observed_counts must be (time, units), got {obs.shape}")
    if pred.ndim == 2:
        pred = pred[None, :, :]
    if pred.ndim != 3:
        raise ValueError(f"predicted_counts must be (n, time, units) or (time, units), got {pred.shape}")
    if pred.shape[1:] != obs.shape:
        raise ValueError(f"predicted_counts has trailing shape {pred.shape[1:]}, expected {obs.shape}")
    if not np.isfinite(obs).all():
        raise ValueError("observed_counts contains non-finite values")
    if not np.isfinite(pred).all():
        raise ValueError("predicted_counts contains non-finite values")
    if score_family == "poisson_log_likelihood":
        rate = np.maximum(pred, float(epsilon))
        return float(likelihood_scale) * np.sum(obs[None, :, :] * np.log(rate) - rate, axis=(1, 2))
    variance = np.maximum(float(phi) * np.maximum(pred, 0.0), float(epsilon))
    resid = obs[None, :, :] - pred
    quad = (resid * resid) / variance
    logp = -0.5 * np.sum(quad, axis=(1, 2))
    if score_family == "gaussian_log_likelihood":
        logp -= 0.5 * np.sum(np.log(variance), axis=(1, 2))
    return float(likelihood_scale) * logp


def _as_label_table(counts_by_theta: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for label in THETA_LABELS:
        if label not in counts_by_theta:
            raise ValueError(f"Missing theta label {label!r}")
        arr = np.asarray(counts_by_theta[label], dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(f"counts_by_theta[{label!r}] must be (trajectory, time, units), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"counts_by_theta[{label!r}] contains non-finite values")
        out[label] = arr
    if out[THETA_PLUS].shape != out[THETA_MINUS].shape:
        raise ValueError(
            f"plus/minus trajectory tables must match, got {out[THETA_PLUS].shape} and {out[THETA_MINUS].shape}"
        )
    return out


def _as_zero_table(zero_counts_by_theta: dict[str, np.ndarray] | None) -> dict[str, np.ndarray] | None:
    if zero_counts_by_theta is None:
        return None
    out: dict[str, np.ndarray] = {}
    for label in THETA_LABELS:
        if label not in zero_counts_by_theta:
            raise ValueError(f"Missing zero-eye theta label {label!r}")
        arr = np.asarray(zero_counts_by_theta[label], dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"zero_counts_by_theta[{label!r}] must be (time, units), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"zero_counts_by_theta[{label!r}] contains non-finite values")
        out[label] = arr
    if out[THETA_PLUS].shape != out[THETA_MINUS].shape:
        raise ValueError(
            f"plus/minus zero-eye counts must match, got {out[THETA_PLUS].shape} and {out[THETA_MINUS].shape}"
        )
    return out


def _prediction(score_by_label: dict[str, float]) -> str:
    vals = np.asarray([score_by_label[THETA_PLUS], score_by_label[THETA_MINUS]], dtype=np.float64)
    if not np.isfinite(vals).all():
        return ""
    return THETA_PLUS if vals[0] >= vals[1] else THETA_MINUS


def _posterior_from_log_likelihood(log_likelihood: np.ndarray) -> np.ndarray:
    ll = np.asarray(log_likelihood, dtype=np.float64)
    norm = logsumexp(ll)
    if not np.isfinite(norm):
        return np.full(ll.shape, np.nan, dtype=np.float64)
    return np.exp(ll - norm)


def _rank_desc(values: np.ndarray, index: int) -> float:
    vals = np.asarray(values, dtype=np.float64)
    idx = int(index)
    if idx < 0 or idx >= vals.shape[0] or not np.isfinite(vals[idx]):
        return float("nan")
    return float(1 + np.sum(vals > vals[idx]))


def _effective_count(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.size == 0 or not np.isfinite(probs).all():
        return float("nan")
    denom = float(np.sum(probs * probs))
    return 1.0 / denom if denom > 0.0 else float("nan")


def score_trajectory_table_vernier_observer_trial(
    observed_counts: np.ndarray,
    true_label: str,
    counts_by_theta: dict[str, np.ndarray],
    *,
    true_trace_index: int,
    known_counts_by_theta: dict[str, np.ndarray] | None = None,
    zero_counts_by_theta: dict[str, np.ndarray] | None = None,
    include_self: bool = True,
    phi: float = 1.0,
    likelihood_normalization: str = "poisson",
    likelihood_scale: float = 1.0,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Score one pseudo-observation with an empirical trajectory table.

    In Poisson-likelihood mode, the joint-eye score is
    ``log mean_w p(response | theta, w)`` over the retained empirical
    trajectory catalog.  Leave-one-out mode is available as a diagnostic, but
    the default includes the true trace and therefore matches the empirical
    trajectory prior literally.
    """
    true = str(true_label)
    if true not in THETA_LABELS:
        raise ValueError(f"true_label must be one of {THETA_LABELS}, got {true_label!r}")
    table = _as_label_table(counts_by_theta)
    known_table = _as_label_table(known_counts_by_theta) if known_counts_by_theta is not None else table
    zero_table = _as_zero_table(zero_counts_by_theta)
    obs = np.asarray(observed_counts, dtype=np.float64)
    if obs.shape != table[THETA_PLUS].shape[1:]:
        raise ValueError(f"observed_counts shape {obs.shape} does not match table bins {table[THETA_PLUS].shape[1:]}")
    if obs.shape != known_table[THETA_PLUS].shape[1:]:
        raise ValueError(
            f"observed_counts shape {obs.shape} does not match known-eye table bins {known_table[THETA_PLUS].shape[1:]}"
        )
    if not np.isfinite(obs).all():
        raise ValueError("observed_counts contains non-finite values")
    n_traj = int(table[THETA_PLUS].shape[0])
    n_known_traj = int(known_table[THETA_PLUS].shape[0])
    true_idx = int(true_trace_index)
    if true_idx < 0 or true_idx >= n_known_traj:
        raise ValueError(f"true_trace_index {true_idx} outside known-eye trajectory table of size {n_known_traj}")
    if true_idx >= n_traj and not bool(include_self):
        raise ValueError(f"true_trace_index {true_idx} outside prior trajectory table of size {n_traj}")

    mask = np.ones(n_traj, dtype=bool)
    if not bool(include_self):
        mask[true_idx] = False
    n_joint = int(np.sum(mask))
    if n_joint <= 0:
        raise ValueError("Trajectory catalog is empty after leave-one-out masking")

    joint_log_evidence: dict[str, float] = {}
    known_log_evidence: dict[str, float] = {}
    zero_log_evidence: dict[str, float] = {}
    trajectory_ll: dict[str, np.ndarray] = {}
    posterior_neff: dict[str, float] = {}
    true_rank: dict[str, float] = {}
    for label in THETA_LABELS:
        ll = diagonal_count_log_likelihood(
            obs,
            table[label],
            phi=float(phi),
            normalization=str(likelihood_normalization),
            likelihood_scale=float(likelihood_scale),
            epsilon=float(epsilon),
        )
        trajectory_ll[label] = ll
        joint_log_evidence[label] = logsumexp(ll[mask]) - float(np.log(n_joint)) if n_joint > 0 else float("nan")
        posterior = _posterior_from_log_likelihood(ll[mask])
        posterior_neff[label] = _effective_count(posterior)
        true_rank[label] = _rank_desc(ll, true_idx) if true_idx < ll.shape[0] else float("nan")
        known_ll = diagonal_count_log_likelihood(
            obs,
            known_table[label],
            phi=float(phi),
            normalization=str(likelihood_normalization),
            likelihood_scale=float(likelihood_scale),
            epsilon=float(epsilon),
        )
        known_log_evidence[label] = float(known_ll[true_idx])
        if zero_table is None:
            zero_log_evidence[label] = float("nan")
        else:
            zero_log_evidence[label] = float(
                diagonal_count_log_likelihood(
                    obs,
                    zero_table[label],
                    phi=float(phi),
                    normalization=str(likelihood_normalization),
                    likelihood_scale=float(likelihood_scale),
                    epsilon=float(epsilon),
                )[0]
            )

    other = THETA_MINUS if true == THETA_PLUS else THETA_PLUS
    pred_joint = _prediction(joint_log_evidence)
    pred_known = _prediction(known_log_evidence)
    pred_zero = _prediction(zero_log_evidence)
    best_traj_log_evidence = {label: float(np.max(trajectory_ll[label][mask])) for label in THETA_LABELS}
    pred_best_traj = _prediction(best_traj_log_evidence)
    joint_margin = float(joint_log_evidence[true] - joint_log_evidence[other])
    known_margin = float(known_log_evidence[true] - known_log_evidence[other])
    zero_margin = float(zero_log_evidence[true] - zero_log_evidence[other])
    best_traj_margin = float(best_traj_log_evidence[true] - best_traj_log_evidence[other])
    score_family = table_score_family(str(likelihood_normalization))
    score_is_llr = score_family in {"poisson_log_likelihood", "gaussian_log_likelihood"}
    decision_rule = "marginal_vernier_llr" if score_is_llr else "marginal_mahalanobis_residual_score"
    readout = (
        "trajectory_table_marginal_vernier_llr"
        if score_is_llr
        else "trajectory_table_marginal_residual_score"
    )
    prior_label = (
        "uniform_empirical_condition_catalog"
        if bool(include_self)
        else "leave_one_out_uniform_empirical_condition_catalog"
    )

    raw_denom = known_log_evidence[true] - zero_log_evidence[true]
    raw_closure = (
        (joint_log_evidence[true] - zero_log_evidence[true]) / raw_denom
        if score_is_llr and np.isfinite(raw_denom) and abs(raw_denom) > 1e-12
        else float("nan")
    )
    margin_denom = known_margin - zero_margin
    margin_closure = (
        (joint_margin - zero_margin) / margin_denom
        if np.isfinite(margin_denom) and abs(margin_denom) > 1e-12
        else float("nan")
    )

    return {
        "readout": readout,
        "trajectory_table_mode": "exact_cached_convgru_response_table",
        "trajectory_prior": prior_label,
        "observer_interpretation": (
            "Vernier likelihood ratio with empirical trajectory nuisance marginalization"
            if score_family == "poisson_log_likelihood"
            else "Gaussian Vernier likelihood-ratio diagnostic over empirical trajectory nuisance catalog"
            if score_family == "gaussian_log_likelihood"
            else "Residual-energy diagnostic over empirical trajectory nuisance catalog"
        ),
        "trajectory_table_include_self": bool(include_self),
        "trajectory_table_leave_one_out": not bool(include_self),
        "n_catalog_trajectories": n_traj,
        "n_known_trajectories": n_known_traj,
        "n_joint_trajectories": n_joint,
        "true_trace_index": true_idx,
        "true_label": true,
        "pred_joint": pred_joint,
        "pred_known": pred_known,
        "pred_zero": pred_zero,
        "pred_best_trajectory": pred_best_traj,
        "joint_correct": bool(pred_joint == true) if pred_joint else float("nan"),
        "known_correct": bool(pred_known == true) if pred_known else float("nan"),
        "zero_correct": bool(pred_zero == true) if pred_zero else float("nan"),
        "best_trajectory_correct": bool(pred_best_traj == true) if pred_best_traj else float("nan"),
        "decision_rule": decision_rule,
        "joint_likelihood_normalization": str(likelihood_normalization),
        "joint_score_family": score_family,
        "joint_evidence_is_normalized_log_probability": bool(score_is_llr),
        "joint_log_evidence_plus": float(joint_log_evidence[THETA_PLUS]),
        "joint_log_evidence_minus": float(joint_log_evidence[THETA_MINUS]),
        "known_log_evidence_plus": float(known_log_evidence[THETA_PLUS]),
        "known_log_evidence_minus": float(known_log_evidence[THETA_MINUS]),
        "zero_log_evidence_plus": float(zero_log_evidence[THETA_PLUS]),
        "zero_log_evidence_minus": float(zero_log_evidence[THETA_MINUS]),
        "joint_log_evidence_true": float(joint_log_evidence[true]),
        "known_log_evidence_true": float(known_log_evidence[true]),
        "zero_log_evidence_true": float(zero_log_evidence[true]),
        "best_trajectory_log_evidence_plus": float(best_traj_log_evidence[THETA_PLUS]),
        "best_trajectory_log_evidence_minus": float(best_traj_log_evidence[THETA_MINUS]),
        "best_trajectory_log_evidence_true": float(best_traj_log_evidence[true]),
        "joint_score": joint_margin,
        "known_eye_score": known_margin,
        "zero_eye_score": zero_margin,
        "best_trajectory_score": best_traj_margin,
        "posterior_neff_true": float(posterior_neff[true]),
        "posterior_neff_plus": float(posterior_neff[THETA_PLUS]),
        "posterior_neff_minus": float(posterior_neff[THETA_MINUS]),
        "true_trajectory_rank_true": float(true_rank[true]),
        "true_trajectory_rank_plus": float(true_rank[THETA_PLUS]),
        "true_trajectory_rank_minus": float(true_rank[THETA_MINUS]),
        "gap_closure_vs_zero_known": float(raw_closure),
        "margin_gap_closure_vs_zero_known": float(margin_closure),
    }


def _finite(values: list[Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _mean_or_nan(values: list[Any]) -> float:
    vals = _finite(values)
    return float(np.mean(vals)) if vals.size else float("nan")


def _accuracy(values: list[Any]) -> float:
    vals = [value for value in values if isinstance(value, (bool, np.bool_))]
    return float(np.mean(vals)) if vals else float("nan")


def summarize_trajectory_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate trajectory-table observer trials."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("readout", ""),
            row.get("condition", ""),
            row.get("prior_condition", row.get("condition", "")),
            row.get("fd_step_arcmin", ""),
            row.get("inference_mode", ""),
            row.get("trajectory_table_mode", ""),
            row.get("trajectory_table_include_self", ""),
            row.get("joint_likelihood_normalization", ""),
            row.get("joint_score_family", ""),
            row.get("zero_eye_reference_condition", ""),
            row.get("zero_eye_reference_available", ""),
        )
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        (
            readout,
            condition,
            prior_condition,
            fd_step,
            inference_mode,
            table_mode,
            include_self,
            normalization,
            score_family,
            zero_ref,
            zero_ref_available,
        ) = key
        grp = groups[key]
        out.append(
            {
                "readout": readout,
                "condition": condition,
                "prior_condition": prior_condition,
                "fd_step_arcmin": fd_step,
                "inference_mode": inference_mode,
                "trajectory_table_mode": table_mode,
                "trajectory_table_include_self": include_self,
                "trajectory_table_leave_one_out": not bool(include_self),
                "joint_likelihood_normalization": normalization,
                "joint_score_family": score_family,
                "zero_eye_reference_condition": zero_ref,
                "zero_eye_reference_available": zero_ref_available,
                "n": len(grp),
                "joint_accuracy": _accuracy([row.get("joint_correct") for row in grp]),
                "known_accuracy": _accuracy([row.get("known_correct") for row in grp]),
                "zero_accuracy": _accuracy([row.get("zero_correct") for row in grp]),
                "best_trajectory_accuracy": _accuracy([row.get("best_trajectory_correct") for row in grp]),
                "mean_joint_score": _mean_or_nan([row.get("joint_score") for row in grp]),
                "mean_known_eye_score": _mean_or_nan([row.get("known_eye_score") for row in grp]),
                "mean_zero_eye_score": _mean_or_nan([row.get("zero_eye_score") for row in grp]),
                "mean_best_trajectory_score": _mean_or_nan([row.get("best_trajectory_score") for row in grp]),
                "mean_posterior_neff_true": _mean_or_nan([row.get("posterior_neff_true") for row in grp]),
                "median_posterior_neff_true": float(np.median(_finite([row.get("posterior_neff_true") for row in grp])))
                if _finite([row.get("posterior_neff_true") for row in grp]).size
                else float("nan"),
                "mean_true_trajectory_rank_true": _mean_or_nan([row.get("true_trajectory_rank_true") for row in grp]),
                "median_true_trajectory_rank_true": float(
                    np.median(_finite([row.get("true_trajectory_rank_true") for row in grp]))
                )
                if _finite([row.get("true_trajectory_rank_true") for row in grp]).size
                else float("nan"),
                "mean_gap_closure_vs_zero_known": _mean_or_nan(
                    [row.get("gap_closure_vs_zero_known") for row in grp]
                ),
                "mean_margin_gap_closure_vs_zero_known": _mean_or_nan(
                    [row.get("margin_gap_closure_vs_zero_known") for row in grp]
                ),
                "median_margin_gap_closure_vs_zero_known": float(
                    np.median(_finite([row.get("margin_gap_closure_vs_zero_known") for row in grp]))
                )
                if _finite([row.get("margin_gap_closure_vs_zero_known") for row in grp]).size
                else float("nan"),
                "mean_n_joint_trajectories": _mean_or_nan([row.get("n_joint_trajectories") for row in grp]),
            }
        )
    return out
