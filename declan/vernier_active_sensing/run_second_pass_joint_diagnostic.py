#!/usr/bin/env python3
"""Second-pass cached Vernier joint-decoding diagnostic.

This script wraps the exact cached-trajectory observer with a stricter
calibration/evaluation layer.  It does not render new model responses.  It
uses existing Vernier rate caches to ask whether trajectory-table joint
decoding improves when likelihood scale is calibrated on one subset of traces
and evaluated on heldout traces.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .joint_observer import THETA_LABELS, THETA_MINUS, THETA_PLUS, logsumexp
from .run_trajectory_table_observer import (
    _cache_counts,
    _load_rate_caches,
    _mean_reference_counts,
    _select_caches,
    _truncate_label_tables,
    parse_csv_float,
    parse_csv_str,
    write_csv,
    write_json,
)
from .trajectory_table_observer import (
    SUPPORTED_TABLE_LIKELIHOODS,
    diagonal_count_log_likelihood,
    table_score_family,
)

CATALOG_MODES = ("include_self", "leave_one_out")
SELECTION_SCOPES = (
    "global_by_fd_and_mode",
    "condition_by_fd_and_mode",
    "condition_prior_by_fd_and_mode",
)
PRIOR_POLICIES = ("selected_conditions", "same_condition", "explicit")


def parse_catalog_modes(text: str) -> list[str]:
    modes = parse_csv_str(text)
    if not modes:
        raise ValueError("At least one catalog mode is required")
    bad = [mode for mode in modes if mode not in CATALOG_MODES]
    if bad:
        raise ValueError(f"Unsupported catalog modes {bad}; expected {CATALOG_MODES}")
    return modes


def selection_scope_keys(scope: str) -> list[str]:
    if scope == "global_by_fd_and_mode":
        return ["catalog_mode", "fd_step_arcmin"]
    if scope == "condition_by_fd_and_mode":
        return ["catalog_mode", "condition", "fd_step_arcmin"]
    if scope == "condition_prior_by_fd_and_mode":
        return ["catalog_mode", "condition", "prior_condition", "fd_step_arcmin"]
    raise ValueError(f"Unsupported selection scope {scope!r}; expected {SELECTION_SCOPES}")


def _finite(values: list[Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _mean_or_nan(values: list[Any]) -> float:
    vals = _finite(values)
    return float(np.mean(vals)) if vals.size else float("nan")


def _median_or_nan(values: list[Any]) -> float:
    vals = _finite(values)
    return float(np.median(vals)) if vals.size else float("nan")


def _accuracy(values: list[Any]) -> float:
    vals = [value for value in values if isinstance(value, (bool, np.bool_))]
    return float(np.mean(vals)) if vals else float("nan")


def _unique_sorted(values: list[Any]) -> list[Any]:
    return sorted(set(values), key=lambda value: (str(type(value)), value))


def _group_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(name, "") for name in keys)
        groups.setdefault(key, []).append(row)
    return groups


def summarize_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    """Aggregate decoder diagnostics with caller-selected grouping keys."""
    summaries: list[dict[str, Any]] = []
    for key, grp in sorted(_group_rows(rows, group_keys).items()):
        out = {name: value for name, value in zip(group_keys, key, strict=True)}
        out.update(
            {
                "n": len(grp),
                "n_conditions": len({row.get("condition") for row in grp}),
                "n_prior_conditions": len({row.get("prior_condition") for row in grp}),
                "n_traces": len({row.get("trace_index") for row in grp}),
                "joint_accuracy": _accuracy([row.get("joint_correct") for row in grp]),
                "known_accuracy": _accuracy([row.get("known_correct") for row in grp]),
                "zero_accuracy": _accuracy([row.get("zero_correct") for row in grp]),
                "best_trajectory_accuracy": _accuracy([row.get("best_trajectory_correct") for row in grp]),
                "mean_joint_score": _mean_or_nan([row.get("joint_score") for row in grp]),
                "mean_known_eye_score": _mean_or_nan([row.get("known_eye_score") for row in grp]),
                "mean_zero_eye_score": _mean_or_nan([row.get("zero_eye_score") for row in grp]),
                "mean_best_trajectory_score": _mean_or_nan([row.get("best_trajectory_score") for row in grp]),
                "mean_posterior_neff_true": _mean_or_nan([row.get("posterior_neff_true") for row in grp]),
                "median_posterior_neff_true": _median_or_nan([row.get("posterior_neff_true") for row in grp]),
                "mean_true_trajectory_rank_true": _mean_or_nan(
                    [row.get("true_trajectory_rank_true") for row in grp]
                ),
                "median_true_trajectory_rank_true": _median_or_nan(
                    [row.get("true_trajectory_rank_true") for row in grp]
                ),
                "mean_gap_closure_vs_zero_known": _mean_or_nan(
                    [row.get("gap_closure_vs_zero_known") for row in grp]
                ),
                "mean_margin_gap_closure_vs_zero_known": _mean_or_nan(
                    [row.get("margin_gap_closure_vs_zero_known") for row in grp]
                ),
                "median_margin_gap_closure_vs_zero_known": _median_or_nan(
                    [row.get("margin_gap_closure_vs_zero_known") for row in grp]
                ),
                "mean_n_joint_trajectories": _mean_or_nan([row.get("n_joint_trajectories") for row in grp]),
                "same_condition_prior_fraction": _mean_or_nan(
                    [float(row.get("condition_matches_prior", False)) for row in grp]
                ),
            }
        )
        summaries.append(out)
    return summaries


def _scale_closeness_to_one(scale: Any) -> float:
    value = float(scale)
    if value <= 0.0 or not np.isfinite(value):
        return float("inf")
    return abs(float(np.log2(value)))


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    accuracy = float(row.get("joint_accuracy", float("nan")))
    closure = float(row.get("mean_margin_gap_closure_vs_zero_known", float("nan")))
    score = float(row.get("mean_joint_score", float("nan")))
    return (
        accuracy if np.isfinite(accuracy) else -float("inf"),
        closure if np.isfinite(closure) else -float("inf"),
        score if np.isfinite(score) else -float("inf"),
        -_scale_closeness_to_one(row.get("likelihood_scale", float("nan"))),
    )


def _best_candidate(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not summary_rows:
        raise ValueError("Cannot select a likelihood scale from an empty candidate list")
    return max(summary_rows, key=_candidate_sort_key)


def _safe_divide(numerator: Any, denominator: Any) -> float:
    num = float(numerator)
    den = float(denominator)
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= 1e-12:
        return float("nan")
    return num / den


def _safe_difference(a: Any, b: Any) -> float:
    aval = float(a)
    bval = float(b)
    if not np.isfinite(aval) or not np.isfinite(bval):
        return float("nan")
    return aval - bval


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def summarize_static_margin_comparison(
    summary_rows: list[dict[str, Any]],
    *,
    static_condition: str = "static_center",
) -> list[dict[str, Any]]:
    """Compare each deterministic observer margin to the static-center margin.

    The expected-count Vernier tables often saturate accuracy for the static
    centered stimulus.  These rows therefore keep the more sensitive benchmark
    explicit: known-trace FEM can exceed static by log-likelihood margin, and
    the joint observer is judged by how much of that deterministic margin it
    recovers.
    """
    baseline_keys = [
        "catalog_mode",
        "fd_step_arcmin",
        "inference_mode",
        "observation_mode",
        "likelihood_scale",
        "joint_likelihood_normalization",
        "joint_score_family",
        "zero_eye_reference_condition",
    ]
    baselines: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in summary_rows:
        if str(row.get("condition")) != str(static_condition):
            continue
        if not _as_bool(row.get("condition_matches_prior")):
            continue
        key = tuple(row.get(name, "") for name in baseline_keys)
        baselines[key] = row

    comparison_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        key = tuple(row.get(name, "") for name in baseline_keys)
        baseline = baselines.get(key)
        if baseline is None:
            continue

        joint_margin = float(row.get("mean_joint_score", float("nan")))
        known_margin = float(row.get("mean_known_eye_score", float("nan")))
        zero_margin = float(row.get("mean_zero_eye_score", float("nan")))
        static_margin = float(baseline.get("mean_known_eye_score", float("nan")))
        static_joint_margin = float(baseline.get("mean_joint_score", float("nan")))
        known_gain = _safe_difference(known_margin, static_margin)
        joint_gain = _safe_difference(joint_margin, static_margin)

        comparison_rows.append(
            {
                "catalog_mode": row.get("catalog_mode", ""),
                "condition": row.get("condition", ""),
                "prior_condition": row.get("prior_condition", ""),
                "condition_matches_prior": row.get("condition_matches_prior", ""),
                "fd_step_arcmin": row.get("fd_step_arcmin", ""),
                "inference_mode": row.get("inference_mode", ""),
                "observation_mode": row.get("observation_mode", ""),
                "likelihood_scale": row.get("likelihood_scale", ""),
                "joint_likelihood_normalization": row.get("joint_likelihood_normalization", ""),
                "joint_score_family": row.get("joint_score_family", ""),
                "static_condition": str(static_condition),
                "static_condition_margin_source": "mean_known_eye_score",
                "joint_accuracy": row.get("joint_accuracy", float("nan")),
                "known_accuracy": row.get("known_accuracy", float("nan")),
                "zero_accuracy": row.get("zero_accuracy", float("nan")),
                "static_joint_accuracy": baseline.get("joint_accuracy", float("nan")),
                "static_known_accuracy": baseline.get("known_accuracy", float("nan")),
                "joint_accuracy_delta_vs_zero": _safe_difference(
                    row.get("joint_accuracy", float("nan")),
                    row.get("zero_accuracy", float("nan")),
                ),
                "joint_accuracy_delta_vs_static": _safe_difference(
                    row.get("joint_accuracy", float("nan")),
                    baseline.get("joint_accuracy", float("nan")),
                ),
                "joint_margin": joint_margin,
                "known_margin": known_margin,
                "zero_margin": zero_margin,
                "static_margin": static_margin,
                "static_joint_margin": static_joint_margin,
                "joint_margin_ratio_vs_static": _safe_divide(joint_margin, static_margin),
                "known_margin_ratio_vs_static": _safe_divide(known_margin, static_margin),
                "zero_margin_ratio_vs_static": _safe_divide(zero_margin, static_margin),
                "joint_fraction_of_known_margin": _safe_divide(joint_margin, known_margin),
                "joint_margin_gain_vs_static": joint_gain,
                "known_margin_gain_vs_static": known_gain,
                "joint_fraction_of_known_static_gain": _safe_divide(joint_gain, known_gain),
                "joint_margin_delta_vs_zero": _safe_difference(joint_margin, zero_margin),
                "known_margin_delta_vs_zero": _safe_difference(known_margin, zero_margin),
                "joint_fraction_of_known_zero_gain": _safe_divide(
                    _safe_difference(joint_margin, zero_margin),
                    _safe_difference(known_margin, zero_margin),
                ),
                "static_baseline_n": baseline.get("n", ""),
                "n": row.get("n", ""),
            }
        )
    return comparison_rows


def _nearest_available_scale(scales: list[float], target: float) -> float:
    if not scales:
        raise ValueError("No likelihood scales are available")
    return min(scales, key=lambda value: (abs(np.log2(float(value) / float(target))), abs(float(value) - float(target))))


def _rows_for_scale(rows: list[dict[str, Any]], scale: float) -> list[dict[str, Any]]:
    return [row for row in rows if np.isclose(float(row.get("likelihood_scale")), float(scale))]


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


def _sweep_likelihood_scale_trial(
    observed_counts: np.ndarray,
    true_label: str,
    counts_by_theta: dict[str, np.ndarray],
    *,
    true_trace_index: int,
    known_counts_by_theta: dict[str, np.ndarray],
    zero_counts_by_theta: dict[str, np.ndarray] | None,
    include_self: bool,
    phi: float,
    likelihood_normalization: str,
    likelihood_scales: list[float],
    epsilon: float = 1e-8,
) -> list[dict[str, Any]]:
    """Score one observation across likelihood scales without recomputing LLs."""
    true = str(true_label)
    other = THETA_MINUS if true == THETA_PLUS else THETA_PLUS
    obs = np.asarray(observed_counts, dtype=np.float64)
    n_traj = int(counts_by_theta[THETA_PLUS].shape[0])
    n_known_traj = int(known_counts_by_theta[THETA_PLUS].shape[0])
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

    base_trajectory_ll: dict[str, np.ndarray] = {}
    base_known_ll: dict[str, np.ndarray] = {}
    base_zero_ll: dict[str, np.ndarray] = {}
    for label in THETA_LABELS:
        base_trajectory_ll[label] = diagonal_count_log_likelihood(
            obs,
            counts_by_theta[label],
            phi=float(phi),
            normalization=str(likelihood_normalization),
            likelihood_scale=1.0,
            epsilon=float(epsilon),
        )
        base_known_ll[label] = diagonal_count_log_likelihood(
            obs,
            known_counts_by_theta[label],
            phi=float(phi),
            normalization=str(likelihood_normalization),
            likelihood_scale=1.0,
            epsilon=float(epsilon),
        )
        if zero_counts_by_theta is not None:
            base_zero_ll[label] = diagonal_count_log_likelihood(
                obs,
                zero_counts_by_theta[label],
                phi=float(phi),
                normalization=str(likelihood_normalization),
                likelihood_scale=1.0,
                epsilon=float(epsilon),
            )

    score_family = table_score_family(str(likelihood_normalization))
    score_is_llr = score_family in {"poisson_log_likelihood", "gaussian_log_likelihood"}
    decision_rule = "marginal_vernier_llr" if score_is_llr else "marginal_mahalanobis_residual_score"
    readout = "trajectory_table_marginal_vernier_llr" if score_is_llr else "trajectory_table_marginal_residual_score"
    prior_label = (
        "uniform_empirical_condition_catalog"
        if bool(include_self)
        else "leave_one_out_uniform_empirical_condition_catalog"
    )
    true_rank = {
        label: _rank_desc(base_trajectory_ll[label], true_idx)
        if true_idx < base_trajectory_ll[label].shape[0]
        else float("nan")
        for label in THETA_LABELS
    }

    rows: list[dict[str, Any]] = []
    for likelihood_scale in likelihood_scales:
        scale = float(likelihood_scale)
        joint_log_evidence: dict[str, float] = {}
        known_log_evidence: dict[str, float] = {}
        zero_log_evidence: dict[str, float] = {}
        posterior_neff: dict[str, float] = {}
        best_traj_log_evidence: dict[str, float] = {}
        for label in THETA_LABELS:
            scaled_ll = scale * base_trajectory_ll[label]
            joint_log_evidence[label] = logsumexp(scaled_ll[mask]) - float(np.log(n_joint))
            posterior = _posterior_from_log_likelihood(scaled_ll[mask])
            posterior_neff[label] = _effective_count(posterior)
            known_log_evidence[label] = float(scale * base_known_ll[label][true_idx])
            zero_log_evidence[label] = (
                float(scale * base_zero_ll[label][0]) if zero_counts_by_theta is not None else float("nan")
            )
            best_traj_log_evidence[label] = float(np.max(scaled_ll[mask]))

        pred_joint = _prediction(joint_log_evidence)
        pred_known = _prediction(known_log_evidence)
        pred_zero = _prediction(zero_log_evidence)
        pred_best_traj = _prediction(best_traj_log_evidence)
        joint_margin = float(joint_log_evidence[true] - joint_log_evidence[other])
        known_margin = float(known_log_evidence[true] - known_log_evidence[other])
        zero_margin = float(zero_log_evidence[true] - zero_log_evidence[other])
        best_traj_margin = float(best_traj_log_evidence[true] - best_traj_log_evidence[other])
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
        rows.append(
            {
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
                "likelihood_scale": scale,
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
        )
    return rows


def build_scale_policy_summaries(
    rows: list[dict[str, Any]],
    *,
    selection_keys: list[str],
    baseline_likelihood_scale: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Select likelihood scales on calibration traces and evaluate heldout traces.

    Each split value is used once as the heldout set.  The selected scale is
    chosen only from the remaining traces within each selection-key group.
    """
    selection_rows: list[dict[str, Any]] = []
    pooled_policy_rows: list[dict[str, Any]] = []
    pair_policy_rows: list[dict[str, Any]] = []
    split_values = _unique_sorted([row.get("calibration_split") for row in rows])

    for key, grp in sorted(_group_rows(rows, selection_keys).items()):
        key_payload = {name: value for name, value in zip(selection_keys, key, strict=True)}
        available_scales = _unique_sorted([float(row["likelihood_scale"]) for row in grp])
        baseline_scale = _nearest_available_scale(available_scales, float(baseline_likelihood_scale))
        for heldout_split in split_values:
            calibration = [row for row in grp if row.get("calibration_split") != heldout_split]
            heldout = [row for row in grp if row.get("calibration_split") == heldout_split]
            if not calibration or not heldout:
                continue

            candidate_summaries = summarize_rows(calibration, ["likelihood_scale"])
            selected = _best_candidate(candidate_summaries)
            selected_scale = float(selected["likelihood_scale"])

            heldout_candidates = summarize_rows(heldout, ["likelihood_scale"])
            oracle = _best_candidate(heldout_candidates)
            oracle_scale = float(oracle["likelihood_scale"])

            for candidate in candidate_summaries:
                selection_rows.append(
                    {
                        **key_payload,
                        "heldout_split": heldout_split,
                        "selection_role": "calibration_candidate",
                        "likelihood_scale": float(candidate["likelihood_scale"]),
                        "selected_by_calibration": bool(
                            np.isclose(float(candidate["likelihood_scale"]), selected_scale)
                        ),
                        **{
                            f"calibration_{name}": value
                            for name, value in candidate.items()
                            if name != "likelihood_scale"
                        },
                    }
                )

            policies = [
                ("selected_by_calibration", selected_scale, "heldout"),
                ("baseline_scale_1", baseline_scale, "heldout"),
                ("oracle_heldout", oracle_scale, "heldout_oracle_not_for_claims"),
            ]
            for policy_name, scale, role in policies:
                policy_rows = [
                    {
                        **row,
                        "heldout_split": heldout_split,
                        "scale_policy": policy_name,
                        "evaluation_role": role,
                        "selected_likelihood_scale": selected_scale,
                        "baseline_likelihood_scale": baseline_scale,
                        "oracle_likelihood_scale": oracle_scale,
                    }
                    for row in _rows_for_scale(heldout, scale)
                ]
                pooled_policy_rows.extend(
                    summarize_rows(policy_rows, [*selection_keys, "heldout_split", "scale_policy", "evaluation_role"])
                )
                pair_keys = [*selection_keys]
                for extra_key in ("condition", "prior_condition"):
                    if extra_key not in pair_keys:
                        pair_keys.append(extra_key)
                pair_policy_rows.extend(
                    summarize_rows(
                        policy_rows,
                        [*pair_keys, "heldout_split", "scale_policy", "evaluation_role"],
                    )
                )

    return selection_rows, pooled_policy_rows, pair_policy_rows


def _score_trial_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_dir = Path(args.source_dir)
    caches = _load_rate_caches(source_dir)
    selected = _select_caches(caches, conditions=args.conditions, fd_steps=args.fd_steps_arcmin)
    if not selected:
        raise ValueError("No selected rate caches matched the requested conditions/fd steps")
    rng = np.random.default_rng(int(args.noise_seed))

    explicit_prior_conditions = list(args.prior_conditions)
    if explicit_prior_conditions:
        prior_policy = "explicit"
    else:
        prior_policy = str(args.prior_policy)
        if prior_policy == "explicit":
            raise ValueError("--prior-policy explicit requires --prior-conditions")
    trial_rows: list[dict[str, Any]] = []
    for cache in selected:
        condition = str(cache["condition"])
        fd_step = float(cache["fd_step_arcmin"])
        inference_mode = str(cache["inference_mode"])
        observed_counts = _cache_counts(
            cache,
            bin_seconds=float(args.bin_seconds),
            max_timebins=int(args.max_timebins),
        )
        zero_key = (str(args.reference_condition), fd_step, inference_mode)
        zero_counts = None
        zero_ref_available = zero_key in caches
        if not zero_ref_available and not bool(args.allow_missing_reference):
            raise FileNotFoundError(
                f"Missing zero-eye reference cache for condition={args.reference_condition!r}, "
                f"fd_step={fd_step:g}, inference_mode={inference_mode!r}"
            )
        if zero_ref_available:
            zero_counts = _mean_reference_counts(
                caches[zero_key],
                bin_seconds=float(args.bin_seconds),
                max_timebins=int(args.max_timebins),
                target_timebins=observed_counts[THETA_PLUS].shape[1],
            )

        if prior_policy == "same_condition":
            effective_prior_conditions = [condition]
        elif prior_policy == "selected_conditions":
            effective_prior_conditions = sorted({str(cache["condition"]) for cache in selected})
        else:
            effective_prior_conditions = explicit_prior_conditions
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
            (obs_table, prior_table, zero_table), t = _truncate_label_tables([observed_counts, prior_counts, zero_counts])
            condition_matches_prior = condition == str(prior_condition)

            for catalog_mode in args.catalog_modes:
                include_self = catalog_mode == "include_self"
                for trace_idx in range(obs_table[THETA_PLUS].shape[0]):
                    calibration_split = int(trace_idx) % int(args.n_splits)
                    for true_label in THETA_LABELS:
                        expected_observed = obs_table[true_label][trace_idx]
                        if int(args.n_poisson_repeats) > 0:
                            observations = [
                                (
                                    int(noise_repeat),
                                    rng.poisson(np.maximum(expected_observed, 0.0)).astype(np.float64),
                                    "poisson_sample",
                                )
                                for noise_repeat in range(int(args.n_poisson_repeats))
                            ]
                        else:
                            observations = [(-1, expected_observed, "expected_counts")]
                        for noise_repeat, observed, observation_mode in observations:
                            scale_results = _sweep_likelihood_scale_trial(
                                observed,
                                true_label,
                                prior_table,
                                true_trace_index=trace_idx,
                                known_counts_by_theta=obs_table,
                                zero_counts_by_theta=zero_table,
                                include_self=include_self,
                                phi=float(args.phi),
                                likelihood_normalization=str(args.likelihood_normalization),
                                likelihood_scales=list(args.likelihood_scales),
                            )
                            for result in scale_results:
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
                                        "observation_mode": observation_mode,
                                        "noise_repeat": int(noise_repeat),
                                        "n_poisson_repeats": int(args.n_poisson_repeats),
                                        "n_timebins": int(t),
                                        "n_units": int(observed.shape[1]),
                                        "source_cache": str(cache["path"]),
                                        "prior_cache": str(prior_cache["path"]),
                                        "zero_eye_reference_condition": str(args.reference_condition),
                                        "zero_eye_reference_available": bool(zero_ref_available),
                                        "catalog_mode": catalog_mode,
                                        "likelihood_scale": float(result["likelihood_scale"])
                                        if "likelihood_scale" in result
                                        else float("nan"),
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
            "observation_mode",
            "likelihood_scale",
            "joint_likelihood_normalization",
            "joint_score_family",
            "zero_eye_reference_condition",
        ],
    )
    static_comparison_by_scale = summarize_static_margin_comparison(
        summary_by_scale,
        static_condition=str(args.reference_condition),
    )
    selection_rows, heldout_summary, heldout_summary_by_pair = build_scale_policy_summaries(
        trial_rows,
        selection_keys=selection_keys,
        baseline_likelihood_scale=float(args.baseline_likelihood_scale),
    )

    write_csv(out_dir / "second_pass_joint_trials.csv", trial_rows)
    write_csv(out_dir / "second_pass_summary_by_scale.csv", summary_by_scale)
    write_csv(out_dir / "second_pass_static_comparison_by_scale.csv", static_comparison_by_scale)
    write_csv(out_dir / "second_pass_calibration_selection.csv", selection_rows)
    write_csv(out_dir / "second_pass_heldout_summary.csv", heldout_summary)
    write_csv(out_dir / "second_pass_heldout_summary_by_pair.csv", heldout_summary_by_pair)
    write_json(
        out_dir / "second_pass_manifest.json",
        {
            "source_dir": Path(args.source_dir),
            "out_dir": out_dir,
            "conditions": args.conditions,
            "prior_conditions": args.prior_conditions,
            "effective_prior_policy": "explicit" if args.prior_conditions else str(args.prior_policy),
            "fd_steps_arcmin": args.fd_steps_arcmin,
            "catalog_modes": args.catalog_modes,
            "likelihood_normalization": str(args.likelihood_normalization),
            "joint_score_family": table_score_family(str(args.likelihood_normalization)),
            "likelihood_scales": args.likelihood_scales,
            "baseline_likelihood_scale": float(args.baseline_likelihood_scale),
            "selection_scope": str(args.selection_scope),
            "selection_keys": selection_keys,
            "n_splits": int(args.n_splits),
            "split_rule": "trace_index modulo n_splits",
            "n_poisson_repeats": int(args.n_poisson_repeats),
            "noise_seed": int(args.noise_seed),
            "reference_condition": str(args.reference_condition),
            "phi": float(args.phi),
            "bin_seconds": float(args.bin_seconds),
            "max_timebins": int(args.max_timebins),
            "n_trial_rows": len(trial_rows),
            "n_summary_by_scale_rows": len(summary_by_scale),
            "n_static_comparison_by_scale_rows": len(static_comparison_by_scale),
            "n_calibration_selection_rows": len(selection_rows),
            "n_heldout_summary_rows": len(heldout_summary),
            "n_heldout_summary_by_pair_rows": len(heldout_summary_by_pair),
            "interpretation_guardrail": (
                "Cross-prior posterior trajectory ranks refer to same catalog index, not necessarily the same "
                "physical eye trace. Heldout selected_by_calibration rows are the primary second-pass diagnostic; "
                "oracle_heldout rows are diagnostic only."
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
    parser.add_argument("--prior-policy", choices=PRIOR_POLICIES, default="selected_conditions")
    parser.add_argument("--fd-steps-arcmin", type=str, default="")
    parser.add_argument("--reference-condition", type=str, default="static_center")
    parser.add_argument("--allow-missing-reference", action="store_true")
    parser.add_argument("--catalog-modes", type=str, default="include_self,leave_one_out")
    parser.add_argument("--likelihood-normalization", choices=SUPPORTED_TABLE_LIKELIHOODS, default="poisson")
    parser.add_argument("--likelihood-scales", type=str, default="0.03125,0.0625,0.125,0.25,0.5,1,2,4")
    parser.add_argument("--baseline-likelihood-scale", type=float, default=1.0)
    parser.add_argument("--selection-scope", choices=SELECTION_SCOPES, default="global_by_fd_and_mode")
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--n-poisson-repeats", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--phi", type=float, default=1.0)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument("--max-timebins", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.conditions = parse_csv_str(args.conditions)
    args.prior_conditions = parse_csv_str(args.prior_conditions)
    args.fd_steps_arcmin = parse_csv_float(args.fd_steps_arcmin)
    args.catalog_modes = parse_catalog_modes(args.catalog_modes)
    args.likelihood_scales = parse_csv_float(args.likelihood_scales)
    if int(args.n_splits) < 2:
        raise ValueError("--n-splits must be at least 2")
    if not args.likelihood_scales:
        raise ValueError("--likelihood-scales must contain at least one value")
    if any(scale <= 0.0 for scale in args.likelihood_scales):
        raise ValueError("--likelihood-scales must all be positive")
    if int(args.n_poisson_repeats) < 0:
        raise ValueError("--n-poisson-repeats must be non-negative")
    trial_rows, heldout_summary, heldout_summary_by_pair = run(args)
    print(
        f"Wrote {len(trial_rows)} second-pass trials, "
        f"{len(heldout_summary)} heldout summaries, "
        f"and {len(heldout_summary_by_pair)} pair summaries",
        flush=True,
    )


if __name__ == "__main__":
    main()
