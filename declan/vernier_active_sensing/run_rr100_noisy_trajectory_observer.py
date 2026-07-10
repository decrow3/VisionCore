#!/usr/bin/env python3
"""Cache-only RR100 noisy-retinal-trajectory Vernier observer.

This post-processes ``run_rr100_real_trace_scale_grid`` caches.  For each
anisotropically scaled real-FEM condition, the observer receives a noisy cue to
the retinal trajectory and marginalizes over the saved empirical trajectory
table with Gaussian path-distance weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .joint_observer import THETA_LABELS, THETA_MINUS, THETA_PLUS
from .metrics import expected_counts
from .trajectory_table_observer import (
    score_trajectory_table_vernier_observer_trial,
    summarize_trajectory_table_rows,
    trajectory_gaussian_log_weights,
)


DEFAULT_SOURCE_DIR = Path("outputs/notebook_vernier_walkthrough/rr100_real_trace_scale_grid")
DEFAULT_SIGMAS = "0,0.125,0.25,0.5,1,2,inf"


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return bool(default)
    try:
        if bool(pd.isna(value)):
            return bool(default)
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "t", "yes", "y"}:
            return True
        if token in {"0", "false", "f", "no", "n", ""}:
            return False
        return bool(default)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return bool(default)


def _bool_series(values: pd.Series, *, default: bool = False) -> pd.Series:
    return values.map(lambda value: _parse_bool(value, default=default)).astype(bool)


def parse_csv_float(text: str) -> list[float]:
    values: list[float] = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        values.append(float("inf") if token.lower() in {"inf", "infinity"} else float(token))
    if not values:
        raise ValueError("At least one sigma is required")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else "inf"
    return value


def _condition_from_cache_path(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("rr100_rates_") or "_fd" not in stem:
        raise ValueError(f"Unexpected RR100 cache filename: {path.name}")
    return stem[len("rr100_rates_") : stem.rindex("_fd")]


def _sigma_label(sigma: float) -> str:
    if not np.isfinite(float(sigma)):
        return "gaussian_noisy_retinal_trajectory_sigma_infarcmin"
    return f"gaussian_noisy_retinal_trajectory_sigma_{float(sigma):g}arcmin"


def _load_rr100_caches(source_dir: Path) -> dict[str, dict[str, Any]]:
    caches: dict[str, dict[str, Any]] = {}
    for path in sorted((source_dir / "cache").glob("rr100_rates_*_fd*arcmin.npz")):
        with np.load(path, allow_pickle=True) as npz:
            condition = str(npz["condition"][0]) if "condition" in npz else _condition_from_cache_path(path)
            if "pose_traces" not in npz:
                raise ValueError(f"RR100 cache lacks pose_traces: {path}")
            caches[condition] = {
                "path": path,
                "condition": condition,
                "fd_step_arcmin": float(np.asarray(npz["fd_step_arcmin"])[0]),
                "bin_seconds": float(np.asarray(npz["bin_seconds"])[0]) if "bin_seconds" in npz else float("nan"),
                "plus_rates": np.asarray(npz["plus_rates"], dtype=np.float64),
                "minus_rates": np.asarray(npz["minus_rates"], dtype=np.float64),
                "pose_traces_deg": np.asarray(npz["pose_traces"], dtype=np.float64),
            }
    if not caches:
        raise FileNotFoundError(f"No RR100 rate caches found under {source_dir / 'cache'}")
    return caches


def _condition_metadata(source_dir: Path) -> dict[str, dict[str, Any]]:
    path = source_dir / "rr100_real_trace_scale_grid_summary.csv"
    if not path.exists():
        return {}
    rows = pd.read_csv(path)
    meta: dict[str, dict[str, Any]] = {}
    for row in rows.to_dict("records"):
        condition = str(row.get("condition", ""))
        if condition:
            meta[condition] = {
                "label": row.get("label", condition),
                "across_scale": row.get("across_scale", float("nan")),
                "along_scale": row.get("along_scale", float("nan")),
                "is_static_baseline": _parse_bool(row.get("is_static_baseline", False)),
            }
    return meta


def _cache_tables(cache: dict[str, Any], *, bin_seconds: float, max_timebins: int = 0) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    plus = expected_counts(np.asarray(cache["plus_rates"], dtype=np.float64), float(bin_seconds))
    minus = expected_counts(np.asarray(cache["minus_rates"], dtype=np.float64), float(bin_seconds))
    poses_arcmin = np.asarray(cache["pose_traces_deg"], dtype=np.float64) * 60.0
    if plus.ndim != 3 or minus.ndim != 3 or poses_arcmin.ndim != 3:
        raise ValueError("RR100 plus/minus/pose caches must be 3D arrays")
    if plus.shape[0] != minus.shape[0] or plus.shape[0] != poses_arcmin.shape[0]:
        raise ValueError("RR100 plus/minus/pose caches have inconsistent trajectory counts")
    if plus.shape[2] != minus.shape[2] or poses_arcmin.shape[2] != 2:
        raise ValueError("RR100 cache has inconsistent unit or pose dimensions")
    t = min(plus.shape[1], minus.shape[1], poses_arcmin.shape[1])
    if int(max_timebins) > 0:
        t = min(t, int(max_timebins))
    return {THETA_PLUS: plus[:, :t], THETA_MINUS: minus[:, :t]}, poses_arcmin[:, :t], int(t)


def _zero_table(cache: dict[str, Any], *, bin_seconds: float, max_timebins: int, target_timebins: int) -> dict[str, np.ndarray]:
    table, _poses, t = _cache_tables(cache, bin_seconds=bin_seconds, max_timebins=max_timebins)
    t = min(int(target_timebins), int(t))
    return {
        THETA_PLUS: np.mean(table[THETA_PLUS][:, :t], axis=0),
        THETA_MINUS: np.mean(table[THETA_MINUS][:, :t], axis=0),
    }


def _truncate_tables(
    table: dict[str, np.ndarray],
    poses_arcmin: np.ndarray,
    zero: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], int]:
    t = min(table[THETA_PLUS].shape[1], table[THETA_MINUS].shape[1], poses_arcmin.shape[1], zero[THETA_PLUS].shape[0], zero[THETA_MINUS].shape[0])
    return (
        {THETA_PLUS: table[THETA_PLUS][:, :t], THETA_MINUS: table[THETA_MINUS][:, :t]},
        poses_arcmin[:, :t],
        {THETA_PLUS: zero[THETA_PLUS][:t], THETA_MINUS: zero[THETA_MINUS][:t]},
        int(t),
    )


def _logsumexp_axis(values: np.ndarray, axis: int) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    vmax = np.max(vals, axis=axis, keepdims=True)
    finite = np.isfinite(vmax)
    shifted = np.where(finite, vals - vmax, -np.inf)
    summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    out = vmax + np.log(summed)
    return np.squeeze(out, axis=axis)


def _softmax_neff(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return float("nan")
    vmax = float(np.max(finite))
    probs = np.exp(finite - vmax)
    probs = probs / max(float(np.sum(probs)), 1e-12)
    denom = float(np.sum(probs * probs))
    return 1.0 / denom if denom > 0.0 else float("nan")


def _rank_desc(values: np.ndarray, index: int) -> float:
    vals = np.asarray(values, dtype=np.float64)
    idx = int(index)
    if idx < 0 or idx >= vals.shape[0] or not np.isfinite(vals[idx]):
        return float("nan")
    return float(1 + np.sum(vals > vals[idx]))


def _predict(plus_score: float, minus_score: float) -> str:
    if not np.isfinite(plus_score) or not np.isfinite(minus_score):
        return ""
    return THETA_PLUS if float(plus_score) >= float(minus_score) else THETA_MINUS


def _poisson_ll_matrix(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    likelihood_scale: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    obs = np.asarray(observed, dtype=np.float64)
    pred = np.maximum(np.asarray(predicted, dtype=np.float64), float(epsilon))
    obs_flat = obs.reshape(obs.shape[0], -1)
    pred_flat = pred.reshape(pred.shape[0], -1)
    return float(likelihood_scale) * (obs_flat @ np.log(pred_flat).T - np.sum(pred_flat, axis=1)[None, :])


def _poisson_zero_ll(
    observed: np.ndarray,
    zero_counts: np.ndarray,
    *,
    likelihood_scale: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    obs = np.asarray(observed, dtype=np.float64).reshape(observed.shape[0], -1)
    pred = np.maximum(np.asarray(zero_counts, dtype=np.float64).reshape(-1), float(epsilon))
    return float(likelihood_scale) * (obs @ np.log(pred) - float(np.sum(pred)))


def _score_condition_poisson(
    args: argparse.Namespace,
    *,
    condition: str,
    cache: dict[str, Any],
    table: dict[str, np.ndarray],
    poses_arcmin: np.ndarray,
    zero: dict[str, np.ndarray],
    metadata: dict[str, Any],
    n_timebins: int,
) -> list[dict[str, Any]]:
    n_traj = int(table[THETA_PLUS].shape[0])
    n_joint = n_traj if bool(args.include_self) else n_traj - 1
    score_family = "poisson_log_likelihood"
    rows: list[dict[str, Any]] = []
    ll = {
        (obs_label, score_label): _poisson_ll_matrix(
            table[obs_label],
            table[score_label],
            likelihood_scale=float(args.likelihood_scale),
        )
        for obs_label in THETA_LABELS
        for score_label in THETA_LABELS
    }
    zero_ll = {
        (obs_label, score_label): _poisson_zero_ll(
            table[obs_label],
            zero[score_label],
            likelihood_scale=float(args.likelihood_scale),
        )
        for obs_label in THETA_LABELS
        for score_label in THETA_LABELS
    }
    for sigma in args.trajectory_sigmas_arcmin:
        logw = np.full((n_traj, n_traj), -np.inf, dtype=np.float64)
        dist2 = np.full((n_traj, n_traj), np.nan, dtype=np.float64)
        for trace_idx in range(n_traj):
            mask = np.ones(n_traj, dtype=bool)
            if not bool(args.include_self):
                mask[trace_idx] = False
            logw[trace_idx], dist2[trace_idx] = trajectory_gaussian_log_weights(
                poses_arcmin[trace_idx],
                poses_arcmin,
                sigma_arcmin=float(sigma),
                mask=mask,
                anchor_index=trace_idx,
            )
        trajectory_prior = _sigma_label(float(sigma))
        for true_label in THETA_LABELS:
            other = THETA_MINUS if true_label == THETA_PLUS else THETA_PLUS
            joint = {
                label: _logsumexp_axis(ll[(true_label, label)] + logw, axis=1)
                for label in THETA_LABELS
            }
            known = {label: np.diag(ll[(true_label, label)]) for label in THETA_LABELS}
            zero_scores = {label: zero_ll[(true_label, label)] for label in THETA_LABELS}
            best = {label: np.max(ll[(true_label, label)] + logw, axis=1) for label in THETA_LABELS}
            for trace_idx in range(n_traj):
                pred_joint = _predict(joint[THETA_PLUS][trace_idx], joint[THETA_MINUS][trace_idx])
                pred_known = _predict(known[THETA_PLUS][trace_idx], known[THETA_MINUS][trace_idx])
                pred_zero = _predict(zero_scores[THETA_PLUS][trace_idx], zero_scores[THETA_MINUS][trace_idx])
                pred_best = _predict(best[THETA_PLUS][trace_idx], best[THETA_MINUS][trace_idx])
                joint_margin = float(joint[true_label][trace_idx] - joint[other][trace_idx])
                known_margin = float(known[true_label][trace_idx] - known[other][trace_idx])
                zero_margin = float(zero_scores[true_label][trace_idx] - zero_scores[other][trace_idx])
                best_margin = float(best[true_label][trace_idx] - best[other][trace_idx])
                raw_denom = float(known[true_label][trace_idx] - zero_scores[true_label][trace_idx])
                raw_closure = (
                    float((joint[true_label][trace_idx] - zero_scores[true_label][trace_idx]) / raw_denom)
                    if np.isfinite(raw_denom) and abs(raw_denom) > 1e-12
                    else float("nan")
                )
                margin_denom = float(known_margin - zero_margin)
                margin_closure = (
                    float((joint_margin - zero_margin) / margin_denom)
                    if np.isfinite(margin_denom) and abs(margin_denom) > 1e-12
                    else float("nan")
                )
                prior_probs = np.exp(logw[trace_idx][np.isfinite(logw[trace_idx])])
                weight_neff = 1.0 / float(np.sum(prior_probs * prior_probs)) if prior_probs.size else float("nan")
                true_weight = (
                    float(np.exp(logw[trace_idx, trace_idx]))
                    if np.isfinite(logw[trace_idx, trace_idx])
                    else 0.0
                )
                weighted_true_ll = ll[(true_label, true_label)][trace_idx] + logw[trace_idx]
                rows.append(
                    {
                        "condition": condition,
                        "prior_condition": condition,
                        "fd_step_arcmin": float(cache["fd_step_arcmin"]),
                        "inference_mode": "rr100_cache",
                        "trace_index": int(trace_idx),
                        "n_timebins": int(n_timebins),
                        "n_units": int(table[true_label].shape[2]),
                        "source_cache": str(cache["path"]),
                        "prior_cache": str(cache["path"]),
                        "zero_eye_reference_condition": str(args.reference_condition),
                        "zero_eye_reference_available": True,
                        "axis_convention": "vertical_vernier_across_x_along_y",
                        "using_real_scaled_trajectories": True,
                        **metadata,
                        "readout": "trajectory_table_marginal_vernier_llr",
                        "trajectory_table_mode": "exact_cached_rr100_response_table",
                        "trajectory_prior": trajectory_prior,
                        "observer_interpretation": "Vernier likelihood ratio with noisy-retinal-trajectory nuisance marginalization",
                        "trajectory_table_include_self": bool(args.include_self),
                        "trajectory_table_leave_one_out": not bool(args.include_self),
                        "trajectory_weight_sigma_arcmin": float(sigma),
                        "trajectory_weight_neff": float(weight_neff),
                        "trajectory_weight_neff_fraction": float(weight_neff / max(n_joint, 1)),
                        "trajectory_weight_true": float(true_weight),
                        "trajectory_weight_max": float(np.max(prior_probs)) if prior_probs.size else float("nan"),
                        "trajectory_weight_min_mean_dist2_arcmin2": float(np.nanmin(dist2[trace_idx][np.isfinite(logw[trace_idx])])),
                        "trajectory_weight_true_mean_dist2_arcmin2": float(dist2[trace_idx, trace_idx]),
                        "n_catalog_trajectories": n_traj,
                        "n_known_trajectories": n_traj,
                        "n_joint_trajectories": n_joint,
                        "true_trace_index": int(trace_idx),
                        "true_label": true_label,
                        "pred_joint": pred_joint,
                        "pred_known": pred_known,
                        "pred_zero": pred_zero,
                        "pred_best_trajectory": pred_best,
                        "joint_correct": bool(pred_joint == true_label) if pred_joint else float("nan"),
                        "known_correct": bool(pred_known == true_label) if pred_known else float("nan"),
                        "zero_correct": bool(pred_zero == true_label) if pred_zero else float("nan"),
                        "best_trajectory_correct": bool(pred_best == true_label) if pred_best else float("nan"),
                        "decision_rule": "marginal_vernier_llr",
                        "joint_likelihood_normalization": str(args.likelihood_normalization),
                        "joint_score_family": score_family,
                        "joint_evidence_is_normalized_log_probability": True,
                        "joint_log_evidence_plus": float(joint[THETA_PLUS][trace_idx]),
                        "joint_log_evidence_minus": float(joint[THETA_MINUS][trace_idx]),
                        "known_log_evidence_plus": float(known[THETA_PLUS][trace_idx]),
                        "known_log_evidence_minus": float(known[THETA_MINUS][trace_idx]),
                        "zero_log_evidence_plus": float(zero_scores[THETA_PLUS][trace_idx]),
                        "zero_log_evidence_minus": float(zero_scores[THETA_MINUS][trace_idx]),
                        "joint_log_evidence_true": float(joint[true_label][trace_idx]),
                        "known_log_evidence_true": float(known[true_label][trace_idx]),
                        "zero_log_evidence_true": float(zero_scores[true_label][trace_idx]),
                        "best_trajectory_log_evidence_plus": float(best[THETA_PLUS][trace_idx]),
                        "best_trajectory_log_evidence_minus": float(best[THETA_MINUS][trace_idx]),
                        "best_trajectory_log_evidence_true": float(best[true_label][trace_idx]),
                        "joint_score": joint_margin,
                        "known_eye_score": known_margin,
                        "zero_eye_score": zero_margin,
                        "best_trajectory_score": best_margin,
                        "posterior_neff_true": _softmax_neff(weighted_true_ll),
                        "posterior_neff_plus": _softmax_neff(ll[(true_label, THETA_PLUS)][trace_idx] + logw[trace_idx]),
                        "posterior_neff_minus": _softmax_neff(ll[(true_label, THETA_MINUS)][trace_idx] + logw[trace_idx]),
                        "true_trajectory_rank_true": _rank_desc(weighted_true_ll, trace_idx),
                        "true_trajectory_rank_plus": _rank_desc(ll[(true_label, THETA_PLUS)][trace_idx] + logw[trace_idx], trace_idx),
                        "true_trajectory_rank_minus": _rank_desc(ll[(true_label, THETA_MINUS)][trace_idx] + logw[trace_idx], trace_idx),
                        "gap_closure_vs_zero_known": raw_closure,
                        "margin_gap_closure_vs_zero_known": margin_closure,
                    }
                )
    return rows


def _add_metadata(rows: list[dict[str, Any]], meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        condition = str(row.get("condition", ""))
        enriched = dict(row)
        enriched.update(meta.get(condition, {}))
        out.append(enriched)
    return out


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        sigma = row.get("trajectory_weight_sigma_arcmin", float("nan"))
        sigma_sort = float(sigma) if sigma != "" else float("nan")
        across = pd.to_numeric(row.get("across_scale", float("nan")), errors="coerce")
        along = pd.to_numeric(row.get("along_scale", float("nan")), errors="coerce")
        return (
            sigma_sort,
            _parse_bool(row.get("is_static_baseline", False)),
            float(along) if np.isfinite(along) else -1.0,
            float(across) if np.isfinite(across) else -1.0,
            str(row.get("condition", "")),
        )

    return sorted(rows, key=key)


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else source_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    caches = _load_rr100_caches(source_dir)
    meta = _condition_metadata(source_dir)
    if str(args.reference_condition) not in caches:
        raise FileNotFoundError(f"Missing reference condition cache: {args.reference_condition}")
    bin_seconds = (
        float(args.bin_seconds)
        if float(args.bin_seconds) > 0.0
        else float(caches[str(args.reference_condition)]["bin_seconds"])
    )

    trial_rows: list[dict[str, Any]] = []
    for condition, cache in sorted(caches.items()):
        if args.conditions and condition not in args.conditions:
            continue
        table, poses_arcmin, t = _cache_tables(cache, bin_seconds=bin_seconds, max_timebins=int(args.max_timebins))
        zero = _zero_table(
            caches[str(args.reference_condition)],
            bin_seconds=bin_seconds,
            max_timebins=int(args.max_timebins),
            target_timebins=t,
        )
        table, poses_arcmin, zero, t = _truncate_tables(table, poses_arcmin, zero)
        if str(args.likelihood_normalization) == "poisson":
            trial_rows.extend(
                _score_condition_poisson(
                    args,
                    condition=condition,
                    cache=cache,
                    table=table,
                    poses_arcmin=poses_arcmin,
                    zero=zero,
                    metadata=meta.get(condition, {}),
                    n_timebins=t,
                )
            )
            continue
        n_traj = int(table[THETA_PLUS].shape[0])
        for sigma in args.trajectory_sigmas_arcmin:
            for trace_idx in range(n_traj):
                mask = np.ones(n_traj, dtype=bool)
                if not bool(args.include_self):
                    mask[trace_idx] = False
                logw, dist2 = trajectory_gaussian_log_weights(
                    poses_arcmin[trace_idx],
                    poses_arcmin,
                    sigma_arcmin=float(sigma),
                    mask=mask,
                    anchor_index=trace_idx,
                )
                for true_label in THETA_LABELS:
                    observed = table[true_label][trace_idx]
                    result = score_trajectory_table_vernier_observer_trial(
                        observed,
                        true_label,
                        table,
                        true_trace_index=trace_idx,
                        known_counts_by_theta=table,
                        zero_counts_by_theta=zero,
                        joint_log_trajectory_weights=logw,
                        trajectory_prior_label=_sigma_label(float(sigma)),
                        trajectory_weight_sigma_arcmin=float(sigma),
                        trajectory_mean_dist2_arcmin2=dist2,
                        include_self=bool(args.include_self),
                        phi=float(args.phi),
                        likelihood_normalization=str(args.likelihood_normalization),
                        likelihood_scale=float(args.likelihood_scale),
                    )
                    trial_rows.append(
                        {
                            "condition": condition,
                            "prior_condition": condition,
                            "fd_step_arcmin": float(cache["fd_step_arcmin"]),
                            "inference_mode": "rr100_cache",
                            "trace_index": int(trace_idx),
                            "n_timebins": int(t),
                            "n_units": int(observed.shape[1]),
                            "source_cache": str(cache["path"]),
                            "prior_cache": str(cache["path"]),
                            "zero_eye_reference_condition": str(args.reference_condition),
                            "zero_eye_reference_available": True,
                            "axis_convention": "vertical_vernier_across_x_along_y",
                            "using_real_scaled_trajectories": True,
                            **meta.get(condition, {}),
                            **result,
                        }
                    )

    summary_rows = _add_metadata(summarize_trajectory_table_rows(trial_rows), meta)
    trial_rows = _sort_rows(trial_rows)
    summary_rows = _sort_rows(summary_rows)
    write_csv(out_dir / "rr100_noisy_trajectory_observer_trials.csv", trial_rows)
    write_csv(out_dir / "rr100_noisy_trajectory_observer_summary.csv", summary_rows)
    write_json_payload = {
        "source_dir": source_dir,
        "out_dir": out_dir,
        "reference_condition": str(args.reference_condition),
        "trajectory_sigmas_arcmin": args.trajectory_sigmas_arcmin,
        "include_self": bool(args.include_self),
        "likelihood_normalization": str(args.likelihood_normalization),
        "likelihood_scale": float(args.likelihood_scale),
        "phi": float(args.phi),
        "bin_seconds": float(bin_seconds),
        "n_trial_rows": len(trial_rows),
        "n_summary_rows": len(summary_rows),
        "observer_interpretation": (
            "RR100 Vernier likelihood ratio with a Gaussian noisy-retinal-trajectory prior "
            "over the saved real scaled trajectory catalog"
        ),
    }
    (out_dir / "rr100_noisy_trajectory_observer_manifest.json").write_text(
        json.dumps(json_ready(write_json_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_plots(out_dir, pd.DataFrame(summary_rows))
    return trial_rows, summary_rows


def _finite_float(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _display_sigma(value: float) -> str:
    return "inf" if not np.isfinite(float(value)) else f"{float(value):g}"


def _static_baseline_mask(summary: pd.DataFrame) -> pd.Series:
    if "is_static_baseline" in summary:
        return _bool_series(summary["is_static_baseline"], default=False)
    if "condition" in summary:
        return summary["condition"].astype(str).eq("static_center")
    return pd.Series(False, index=summary.index, dtype=bool)


def write_plots(out_dir: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    sigma_col = pd.to_numeric(summary["trajectory_weight_sigma_arcmin"], errors="coerce")
    finite_sigmas = sorted(float(v) for v in sigma_col[np.isfinite(sigma_col)].unique())
    has_inf = summary["trajectory_weight_sigma_arcmin"].astype(str).str.lower().isin({"inf", "infinity"}).any()
    sigmas = finite_sigmas + ([float("inf")] if has_inf else [])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=180, constrained_layout=True)
    for condition in [
        "real_aniso_across_0_along_1",
        "real_aniso_across_0p25_along_1",
        "real_aniso_across_1_along_1",
        "real_aniso_across_2_along_1",
    ]:
        sub = summary[summary["condition"].eq(condition)].copy()
        if sub.empty:
            continue
        x = []
        y = []
        for idx, row in sub.iterrows():
            sigma = row["trajectory_weight_sigma_arcmin"]
            val = pd.to_numeric(pd.Series([sigma]), errors="coerce").iloc[0]
            x.append(math.log10(float(val)) if np.isfinite(val) and float(val) > 0 else (-2.0 if float(val) == 0.0 else 1.0))
            y.append(float(row["joint_accuracy"]))
        label = str(sub.iloc[0].get("label", condition))
        axes[0].plot(x, y, marker="o", label=label)
        axes[1].plot(x, sub["mean_trajectory_weight_neff"], marker="o", label=label)
    for ax in axes:
        ax.set_xticks([-2, math.log10(0.125), math.log10(0.25), math.log10(0.5), 0, math.log10(2), 1])
        ax.set_xticklabels(["0", "0.125", "0.25", "0.5", "1", "2", "inf"])
        ax.set_xlabel("trajectory uncertainty sigma (arcmin)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(0.5, color="#333333", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Vernier sign accuracy")
    axes[0].set_title("Noisy trajectory marginal")
    axes[1].set_ylabel("prior effective trajectory count")
    axes[1].set_title("Cue width over the trajectory catalog")
    axes[1].legend(frameon=False, fontsize=7)
    fig.savefig(out_dir / "rr100_noisy_trajectory_observer_sigma_sweep.png", bbox_inches="tight")
    plt.close(fig)

    grid_rows = summary[~_static_baseline_mask(summary)].copy()
    scales = sorted(set(float(v) for v in _finite_float(grid_rows["across_scale"])))
    if not scales:
        return
    selected_sigmas = [s for s in sigmas if s in {0.0, 0.25, 0.5, 1.0, 2.0, float("inf")}]
    if not selected_sigmas:
        selected_sigmas = sigmas[: min(6, len(sigmas))]
    fig2, axes = plt.subplots(1, len(selected_sigmas), figsize=(3.0 * len(selected_sigmas), 3.2), dpi=180, constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, sigma in zip(axes_arr, selected_sigmas, strict=True):
        if np.isfinite(sigma):
            sub = grid_rows[np.isclose(pd.to_numeric(grid_rows["trajectory_weight_sigma_arcmin"], errors="coerce"), sigma)]
        else:
            sub = grid_rows[grid_rows["trajectory_weight_sigma_arcmin"].astype(str).str.lower().isin({"inf", "infinity"})]
        values = np.full((len(scales), len(scales)), np.nan, dtype=float)
        for y, along in enumerate(scales):
            for x, across in enumerate(scales):
                cell = sub[
                    np.isclose(pd.to_numeric(sub["across_scale"], errors="coerce"), across)
                    & np.isclose(pd.to_numeric(sub["along_scale"], errors="coerce"), along)
                ]
                if not cell.empty:
                    values[y, x] = float(cell.iloc[0]["mean_margin_gap_closure_vs_zero_known"])
        im = ax.imshow(values, origin="lower", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"sigma={_display_sigma(sigma)}")
        ax.set_xticks(np.arange(len(scales)))
        ax.set_yticks(np.arange(len(scales)))
        ax.set_xticklabels([f"{s:g}" for s in scales], rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels([f"{s:g}" for s in scales], fontsize=6)
        ax.set_xlabel("across scale")
        for yy in range(values.shape[0]):
            for xx in range(values.shape[1]):
                if np.isfinite(values[yy, xx]):
                    ax.text(xx, yy, f"{values[yy, xx]:.2g}", ha="center", va="center", fontsize=4.8, color="white")
    axes_arr[0].set_ylabel("along scale")
    fig2.colorbar(im, ax=axes_arr.tolist(), fraction=0.025, pad=0.02)
    fig2.suptitle("Mean margin closure for noisy-retinal-trajectory marginal observer", y=1.04)
    fig2.savefig(out_dir / "rr100_noisy_trajectory_observer_closure_heatmaps.png", bbox_inches="tight")
    plt.close(fig2)

    static_rows = summary[_static_baseline_mask(summary)].copy()
    if static_rows.empty or "mean_joint_score" not in summary:
        return
    static_by_sigma: dict[float, float] = {}
    for sigma in sigmas:
        if np.isfinite(sigma):
            sub = static_rows[np.isclose(pd.to_numeric(static_rows["trajectory_weight_sigma_arcmin"], errors="coerce"), sigma)]
        else:
            sub = static_rows[
                static_rows["trajectory_weight_sigma_arcmin"].astype(str).str.lower().isin({"inf", "infinity"})
            ]
        if sub.empty:
            continue
        score = pd.to_numeric(sub.iloc[0].get("mean_joint_score", float("nan")), errors="coerce")
        if np.isfinite(score) and abs(float(score)) > 1e-12:
            static_by_sigma[float(sigma)] = float(score)
    selected_static_sigmas = [sigma for sigma in selected_sigmas if float(sigma) in static_by_sigma]
    if not selected_static_sigmas:
        return
    fig3, axes3 = plt.subplots(
        1,
        len(selected_static_sigmas),
        figsize=(3.0 * len(selected_static_sigmas), 3.2),
        dpi=180,
        constrained_layout=True,
    )
    axes3_arr = np.asarray(axes3).reshape(-1)
    for ax, sigma in zip(axes3_arr, selected_static_sigmas, strict=True):
        if np.isfinite(sigma):
            sub = grid_rows[np.isclose(pd.to_numeric(grid_rows["trajectory_weight_sigma_arcmin"], errors="coerce"), sigma)]
        else:
            sub = grid_rows[grid_rows["trajectory_weight_sigma_arcmin"].astype(str).str.lower().isin({"inf", "infinity"})]
        values = np.full((len(scales), len(scales)), np.nan, dtype=float)
        static_score = static_by_sigma[float(sigma)]
        for y, along in enumerate(scales):
            for x, across in enumerate(scales):
                cell = sub[
                    np.isclose(pd.to_numeric(sub["across_scale"], errors="coerce"), across)
                    & np.isclose(pd.to_numeric(sub["along_scale"], errors="coerce"), along)
                ]
                if not cell.empty:
                    score = pd.to_numeric(cell.iloc[0].get("mean_joint_score", float("nan")), errors="coerce")
                    values[y, x] = float(score) / static_score if np.isfinite(score) else float("nan")
        im3 = ax.imshow(values, origin="lower", interpolation="nearest", cmap="magma", vmin=0.0, vmax=1.5)
        ax.set_title(f"sigma={_display_sigma(sigma)}")
        ax.set_xticks(np.arange(len(scales)))
        ax.set_yticks(np.arange(len(scales)))
        ax.set_xticklabels([f"{s:g}" for s in scales], rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels([f"{s:g}" for s in scales], fontsize=6)
        ax.set_xlabel("across scale")
        for yy in range(values.shape[0]):
            for xx in range(values.shape[1]):
                if np.isfinite(values[yy, xx]):
                    ax.text(xx, yy, f"{values[yy, xx]:.2g}", ha="center", va="center", fontsize=4.8, color="white")
    axes3_arr[0].set_ylabel("along scale")
    fig3.colorbar(im3, ax=axes3_arr.tolist(), fraction=0.025, pad=0.02)
    fig3.suptitle("Mean Vernier LLR margin relative to static", y=1.04)
    fig3.savefig(out_dir / "rr100_noisy_trajectory_observer_static_relative_heatmaps.png", bbox_inches="tight")
    plt.close(fig3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--conditions", type=str, default="")
    parser.add_argument("--reference-condition", type=str, default="static_center")
    parser.add_argument("--trajectory-sigmas-arcmin", type=str, default=DEFAULT_SIGMAS)
    parser.add_argument("--include-self", dest="include_self", action="store_true", default=True)
    parser.add_argument("--leave-one-out", dest="include_self", action="store_false")
    parser.add_argument("--likelihood-normalization", choices=("poisson", "residual", "gaussian_full"), default="poisson")
    parser.add_argument("--likelihood-scale", type=float, default=1.0)
    parser.add_argument("--phi", type=float, default=1.0)
    parser.add_argument("--bin-seconds", type=float, default=0.0, help="Defaults to the cache value when <=0.")
    parser.add_argument("--max-timebins", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.conditions = [part.strip() for part in str(args.conditions).split(",") if part.strip()]
    args.trajectory_sigmas_arcmin = parse_csv_float(args.trajectory_sigmas_arcmin)
    trial_rows, summary_rows = run(args)
    print(
        f"Wrote {len(trial_rows)} RR100 noisy-trajectory trials and {len(summary_rows)} summary rows",
        flush=True,
    )


if __name__ == "__main__":
    main()
