#!/usr/bin/env python3
"""Held-out empirical Monte Carlo trajectory observer for the RR100 Vernier slice.

This post-processes ``run_rr100_real_trace_scale_grid`` caches.  Unlike the
small catalog diagnostic, observed traces and nuisance-prior traces are disjoint.
The trajectory-unknown observer is therefore a Monte Carlo marginal over a
held-out empirical catalog of real scaled eye trajectories.
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

from .joint_observer import THETA_LABELS, THETA_MINUS, THETA_PLUS, logsumexp
from .run_rr100_noisy_trajectory_observer import (
    _cache_tables,
    _condition_metadata,
    _display_sigma,
    _load_rr100_caches,
    _parse_bool,
    _poisson_ll_matrix,
    _poisson_zero_ll,
    _predict,
    _sigma_label,
    _softmax_neff,
    json_ready,
    parse_csv_float,
)
from .trajectory_table_observer import trajectory_gaussian_log_weights


DEFAULT_SOURCE_DIR = Path("outputs/notebook_vernier_walkthrough/rr100_real_trace_along1_mc")
DEFAULT_OUT_DIR = Path("outputs/notebook_vernier_walkthrough/rr100_heldout_trajectory_observer_along1")
DEFAULT_SIGMAS = "0,0.25,0.5,1,2,4,8,inf"
DEFAULT_PRIOR_K = "32,64,128"


def parse_csv_int(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one integer value is required")
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


def split_trace_indices(
    n_traces: int,
    *,
    n_observation_traces: int,
    n_prior_traces: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(n_traces)
    n_obs = int(n_observation_traces)
    n_prior = int(n_prior_traces)
    if n_obs <= 0 or n_prior <= 0:
        raise ValueError("n_observation_traces and n_prior_traces must be positive")
    if n_obs + n_prior > n:
        raise ValueError(
            f"Need at least n_observation_traces + n_prior_traces traces, got {n_obs}+{n_prior}>{n}"
        )
    perm = np.random.default_rng(int(seed)).permutation(n)
    return perm[:n_obs].astype(int), perm[n_obs : n_obs + n_prior].astype(int)


def _logsumexp_axis(values: np.ndarray, axis: int) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    vmax = np.max(vals, axis=axis, keepdims=True)
    finite = np.isfinite(vmax)
    shifted = np.where(finite, vals - vmax, -np.inf)
    summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    return np.squeeze(vmax + np.log(summed), axis=axis)


def _accuracy(values: list[Any]) -> float:
    vals = [value for value in values if isinstance(value, (bool, np.bool_))]
    return float(np.mean(vals)) if vals else float("nan")


def _mean(values: list[Any]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _median(values: list[Any]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _trajectory_log_weights_for_observations(
    observed_poses_arcmin: np.ndarray,
    prior_poses_arcmin: np.ndarray,
    *,
    sigma_arcmin: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_obs = int(observed_poses_arcmin.shape[0])
    n_prior = int(prior_poses_arcmin.shape[0])
    logw = np.full((n_obs, n_prior), -np.inf, dtype=np.float64)
    dist2 = np.full((n_obs, n_prior), np.nan, dtype=np.float64)
    mask = np.ones(n_prior, dtype=bool)
    for obs_idx in range(n_obs):
        logw[obs_idx], dist2[obs_idx] = trajectory_gaussian_log_weights(
            observed_poses_arcmin[obs_idx],
            prior_poses_arcmin,
            sigma_arcmin=float(sigma_arcmin),
            mask=mask,
            anchor_index=None,
        )
    return logw, dist2


def _score_condition(
    args: argparse.Namespace,
    *,
    condition: str,
    cache: dict[str, Any],
    reference_cache: dict[str, Any],
    metadata: dict[str, Any],
    obs_indices: np.ndarray,
    prior_indices_full: np.ndarray,
    prior_k: int,
    bin_seconds: float,
) -> list[dict[str, Any]]:
    table, poses_arcmin, t = _cache_tables(cache, bin_seconds=bin_seconds, max_timebins=int(args.max_timebins))
    ref_table, _ref_poses, ref_t = _cache_tables(
        reference_cache,
        bin_seconds=bin_seconds,
        max_timebins=int(args.max_timebins),
    )
    t = min(t, ref_t)
    table = {label: table[label][:, :t] for label in THETA_LABELS}
    ref_table = {label: ref_table[label][:, :t] for label in THETA_LABELS}
    poses_arcmin = poses_arcmin[:, :t]

    prior_indices = np.asarray(prior_indices_full[: int(prior_k)], dtype=int)
    obs_indices = np.asarray(obs_indices, dtype=int)
    if np.intersect1d(obs_indices, prior_indices).size:
        raise ValueError("Observation and prior indices must be disjoint")
    if obs_indices.max(initial=-1) >= table[THETA_PLUS].shape[0] or prior_indices.max(initial=-1) >= table[THETA_PLUS].shape[0]:
        raise ValueError(f"Split indices exceed trajectory count for condition {condition!r}")

    obs_table = {label: table[label][obs_indices] for label in THETA_LABELS}
    known_table = {label: table[label][obs_indices] for label in THETA_LABELS}
    prior_table = {label: table[label][prior_indices] for label in THETA_LABELS}
    prior_poses = poses_arcmin[prior_indices]
    observed_poses = poses_arcmin[obs_indices]
    zero = {label: np.mean(ref_table[label][prior_indices], axis=0) for label in THETA_LABELS}

    known_ll = {
        (obs_label, score_label): np.diag(
            _poisson_ll_matrix(
                obs_table[obs_label],
                known_table[score_label],
                likelihood_scale=float(args.likelihood_scale),
            )
        )
        for obs_label in THETA_LABELS
        for score_label in THETA_LABELS
    }
    prior_ll = {
        (obs_label, score_label): _poisson_ll_matrix(
            obs_table[obs_label],
            prior_table[score_label],
            likelihood_scale=float(args.likelihood_scale),
        )
        for obs_label in THETA_LABELS
        for score_label in THETA_LABELS
    }
    zero_ll = {
        (obs_label, score_label): _poisson_zero_ll(
            obs_table[obs_label],
            zero[score_label],
            likelihood_scale=float(args.likelihood_scale),
        )
        for obs_label in THETA_LABELS
        for score_label in THETA_LABELS
    }

    rows: list[dict[str, Any]] = []
    for sigma in args.trajectory_sigmas_arcmin:
        logw, dist2 = _trajectory_log_weights_for_observations(
            observed_poses,
            prior_poses,
            sigma_arcmin=float(sigma),
        )
        trajectory_prior = (
            "heldout_nearest_empirical_prior"
            if float(sigma) <= 0.0
            else "heldout_uniform_empirical_prior"
            if not np.isfinite(float(sigma))
            else _sigma_label(float(sigma)).replace("gaussian_", "heldout_gaussian_")
        )
        prior_probs = np.exp(logw)
        weight_neff = 1.0 / np.sum(prior_probs * prior_probs, axis=1)
        min_dist2 = np.nanmin(dist2, axis=1)
        mean_dist2 = np.sum(prior_probs * dist2, axis=1)

        for true_label in THETA_LABELS:
            other = THETA_MINUS if true_label == THETA_PLUS else THETA_PLUS
            joint = {
                label: _logsumexp_axis(prior_ll[(true_label, label)] + logw, axis=1)
                for label in THETA_LABELS
            }
            best = {label: np.max(prior_ll[(true_label, label)] + logw, axis=1) for label in THETA_LABELS}
            for local_idx, trace_idx in enumerate(obs_indices):
                pred_joint = _predict(joint[THETA_PLUS][local_idx], joint[THETA_MINUS][local_idx])
                pred_known = _predict(
                    known_ll[(true_label, THETA_PLUS)][local_idx],
                    known_ll[(true_label, THETA_MINUS)][local_idx],
                )
                pred_zero = _predict(
                    zero_ll[(true_label, THETA_PLUS)][local_idx],
                    zero_ll[(true_label, THETA_MINUS)][local_idx],
                )
                pred_best = _predict(best[THETA_PLUS][local_idx], best[THETA_MINUS][local_idx])
                joint_margin = float(joint[true_label][local_idx] - joint[other][local_idx])
                known_margin = float(
                    known_ll[(true_label, true_label)][local_idx]
                    - known_ll[(true_label, other)][local_idx]
                )
                zero_margin = float(
                    zero_ll[(true_label, true_label)][local_idx]
                    - zero_ll[(true_label, other)][local_idx]
                )
                weighted_true = prior_ll[(true_label, true_label)][local_idx] + logw[local_idx]
                rows.append(
                    {
                        "condition": condition,
                        "prior_condition": condition,
                        "fd_step_arcmin": float(cache["fd_step_arcmin"]),
                        "inference_mode": "rr100_heldout_empirical_monte_carlo",
                        "trace_index": int(trace_idx),
                        "local_observation_index": int(local_idx),
                        "n_timebins": int(t),
                        "n_units": int(table[true_label].shape[2]),
                        "source_cache": str(cache["path"]),
                        "prior_cache": str(cache["path"]),
                        "zero_eye_reference_condition": str(args.reference_condition),
                        "zero_eye_reference_available": True,
                        "axis_convention": "vertical_vernier_across_x_along_y",
                        "using_real_scaled_trajectories": True,
                        "heldout_prior": True,
                        "split_seed": int(args.split_seed),
                        **metadata,
                        "readout": "heldout_trajectory_table_marginal_vernier_llr",
                        "trajectory_table_mode": "exact_cached_rr100_response_table",
                        "trajectory_prior": trajectory_prior,
                        "observer_interpretation": (
                            "Vernier likelihood ratio with leave-one-trajectory-out "
                            "empirical catalog trajectory nuisance marginalization"
                        ),
                        "trajectory_table_include_self": False,
                        "trajectory_table_leave_one_out": True,
                        "trajectory_weight_sigma_arcmin": float(sigma),
                        "trajectory_weight_neff": float(weight_neff[local_idx]),
                        "trajectory_weight_neff_fraction": float(weight_neff[local_idx] / max(int(prior_k), 1)),
                        "trajectory_weight_true": 0.0,
                        "trajectory_weight_max": float(np.max(prior_probs[local_idx])),
                        "trajectory_weight_min_mean_dist2_arcmin2": float(min_dist2[local_idx]),
                        "trajectory_weight_mean_dist2_arcmin2": float(mean_dist2[local_idx]),
                        "trajectory_weight_min_rms_dist_arcmin": float(math.sqrt(max(min_dist2[local_idx], 0.0))),
                        "n_catalog_trajectories": int(prior_k),
                        "n_observation_trajectories": int(obs_indices.size),
                        "n_known_trajectories": int(obs_indices.size),
                        "n_joint_trajectories": int(prior_k),
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
                        "joint_likelihood_normalization": "poisson",
                        "joint_score_family": "poisson_log_likelihood",
                        "joint_evidence_is_normalized_log_probability": True,
                        "joint_log_evidence_plus": float(joint[THETA_PLUS][local_idx]),
                        "joint_log_evidence_minus": float(joint[THETA_MINUS][local_idx]),
                        "known_log_evidence_plus": float(known_ll[(true_label, THETA_PLUS)][local_idx]),
                        "known_log_evidence_minus": float(known_ll[(true_label, THETA_MINUS)][local_idx]),
                        "zero_log_evidence_plus": float(zero_ll[(true_label, THETA_PLUS)][local_idx]),
                        "zero_log_evidence_minus": float(zero_ll[(true_label, THETA_MINUS)][local_idx]),
                        "joint_log_evidence_true": float(joint[true_label][local_idx]),
                        "known_log_evidence_true": float(known_ll[(true_label, true_label)][local_idx]),
                        "zero_log_evidence_true": float(zero_ll[(true_label, true_label)][local_idx]),
                        "best_trajectory_log_evidence_plus": float(best[THETA_PLUS][local_idx]),
                        "best_trajectory_log_evidence_minus": float(best[THETA_MINUS][local_idx]),
                        "best_trajectory_log_evidence_true": float(best[true_label][local_idx]),
                        "joint_score": joint_margin,
                        "known_eye_score": known_margin,
                        "zero_eye_score": zero_margin,
                        "best_trajectory_score": float(best[true_label][local_idx] - best[other][local_idx]),
                        "posterior_neff_true": _softmax_neff(weighted_true),
                        "posterior_neff_plus": _softmax_neff(
                            prior_ll[(true_label, THETA_PLUS)][local_idx] + logw[local_idx]
                        ),
                        "posterior_neff_minus": _softmax_neff(
                            prior_ll[(true_label, THETA_MINUS)][local_idx] + logw[local_idx]
                        ),
                        "true_trajectory_rank_true": float("nan"),
                        "true_trajectory_rank_plus": float("nan"),
                        "true_trajectory_rank_minus": float("nan"),
                    }
                )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_cols = [
        "readout",
        "condition",
        "prior_condition",
        "fd_step_arcmin",
        "inference_mode",
        "trajectory_table_mode",
        "trajectory_prior",
        "trajectory_weight_sigma_arcmin",
        "n_catalog_trajectories",
        "split_seed",
        "zero_eye_reference_condition",
        "zero_eye_reference_available",
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(col, "") for col in group_cols)
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        grp = groups[key]
        first = grp[0]
        out.append(
            {
                **{col: first.get(col, "") for col in group_cols},
                "label": first.get("label", first.get("condition", "")),
                "across_scale": first.get("across_scale", float("nan")),
                "along_scale": first.get("along_scale", float("nan")),
                "is_static_baseline": _parse_bool(first.get("is_static_baseline", False)),
                "n": len(grp),
                "n_observation_trajectories": first.get("n_observation_trajectories", ""),
                "mean_n_joint_trajectories": _mean([row.get("n_joint_trajectories") for row in grp]),
                "joint_accuracy": _accuracy([row.get("joint_correct") for row in grp]),
                "known_accuracy": _accuracy([row.get("known_correct") for row in grp]),
                "zero_accuracy": _accuracy([row.get("zero_correct") for row in grp]),
                "best_trajectory_accuracy": _accuracy([row.get("best_trajectory_correct") for row in grp]),
                "mean_joint_score": _mean([row.get("joint_score") for row in grp]),
                "mean_known_eye_score": _mean([row.get("known_eye_score") for row in grp]),
                "mean_zero_eye_score": _mean([row.get("zero_eye_score") for row in grp]),
                "mean_best_trajectory_score": _mean([row.get("best_trajectory_score") for row in grp]),
                "mean_trajectory_weight_neff": _mean([row.get("trajectory_weight_neff") for row in grp]),
                "mean_trajectory_weight_neff_fraction": _mean(
                    [row.get("trajectory_weight_neff_fraction") for row in grp]
                ),
                "mean_trajectory_weight_max": _mean([row.get("trajectory_weight_max") for row in grp]),
                "mean_min_rms_dist_arcmin": _mean([row.get("trajectory_weight_min_rms_dist_arcmin") for row in grp]),
                "median_min_rms_dist_arcmin": _median(
                    [row.get("trajectory_weight_min_rms_dist_arcmin") for row in grp]
                ),
                "mean_posterior_neff_true": _mean([row.get("posterior_neff_true") for row in grp]),
                "median_posterior_neff_true": _median([row.get("posterior_neff_true") for row in grp]),
            }
        )
    return out


def add_static_ratios(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if out.empty:
        return out
    out["trajectory_weight_sigma_arcmin"] = pd.to_numeric(out["trajectory_weight_sigma_arcmin"], errors="coerce")
    out["n_catalog_trajectories"] = pd.to_numeric(out["n_catalog_trajectories"], errors="coerce")
    static_mask = out["is_static_baseline"].map(lambda value: _parse_bool(value, default=False))
    for idx, row in out.iterrows():
        static = out[
            static_mask
            & np.isclose(out["trajectory_weight_sigma_arcmin"], float(row["trajectory_weight_sigma_arcmin"]))
            & np.isclose(out["n_catalog_trajectories"], float(row["n_catalog_trajectories"]))
        ]
        if static.empty:
            continue
        ref = static.iloc[0]
        for col, out_col in [
            ("mean_joint_score", "mean_joint_score_vs_static"),
            ("mean_known_eye_score", "mean_known_eye_score_vs_static"),
            ("mean_zero_eye_score", "mean_zero_eye_score_vs_static"),
        ]:
            denom = float(ref[col])
            out.at[idx, out_col] = float(row[col]) / denom if np.isfinite(denom) and abs(denom) > 1e-12 else np.nan
    return out


def _sigma_values(summary: pd.DataFrame) -> list[float]:
    vals = pd.to_numeric(summary["trajectory_weight_sigma_arcmin"], errors="coerce")
    finite = sorted(float(v) for v in vals[np.isfinite(vals)].unique())
    has_inf = summary["trajectory_weight_sigma_arcmin"].astype(str).str.lower().isin({"inf", "infinity"}).any()
    if np.isinf(vals).any():
        has_inf = True
    return finite + ([float("inf")] if has_inf and float("inf") not in finite else [])


def write_plots(out_dir: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for col in [
        "trajectory_weight_sigma_arcmin",
        "n_catalog_trajectories",
        "across_scale",
        "along_scale",
        "joint_accuracy",
        "mean_joint_score_vs_static",
        "mean_trajectory_weight_neff",
        "mean_min_rms_dist_arcmin",
    ]:
        if col in summary:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    grid = summary[~summary["is_static_baseline"].map(lambda value: _parse_bool(value, default=False))].copy()
    if grid.empty:
        return
    max_k = int(np.nanmax(grid["n_catalog_trajectories"]))
    along_rows = grid[np.isclose(grid["along_scale"], 1.0) & np.isclose(grid["n_catalog_trajectories"], max_k)].copy()
    sigmas = _sigma_values(along_rows)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(sigmas), 1)))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), dpi=180, constrained_layout=True)
    for color, sigma in zip(colors, sigmas, strict=True):
        if np.isfinite(sigma):
            sub = along_rows[np.isclose(along_rows["trajectory_weight_sigma_arcmin"], sigma)].copy()
            label = f"{sigma:g}"
        else:
            sub = along_rows[np.isinf(along_rows["trajectory_weight_sigma_arcmin"])].copy()
            label = "inf"
        if sub.empty:
            continue
        sub = sub.sort_values("across_scale")
        axes[0].plot(sub["across_scale"], sub["mean_joint_score_vs_static"], marker="o", color=color, label=label)
        axes[1].plot(sub["across_scale"], sub["joint_accuracy"], marker="o", color=color, label=label)
        axes[2].plot(sub["across_scale"], sub["mean_trajectory_weight_neff"], marker="o", color=color, label=label)
    axes[0].axhline(1.0, color="#333333", linestyle="--", linewidth=0.9)
    axes[1].axhline(0.5, color="#333333", linestyle="--", linewidth=0.9)
    if "mean_known_eye_score_vs_static" in along_rows:
        known = (
            along_rows[np.isclose(along_rows["n_catalog_trajectories"], max_k)]
            .drop_duplicates("across_scale")
            .sort_values("across_scale")
        )
        if not known.empty:
            axes[0].plot(
                known["across_scale"],
                known["mean_known_eye_score_vs_static"],
                color="#111111",
                linestyle="--",
                linewidth=1.15,
                label="pose-aware",
            )
            axes[1].plot(
                known["across_scale"],
                known["known_accuracy"],
                color="#111111",
                linestyle="--",
                linewidth=1.15,
                label="pose-aware",
            )
    axes[0].set_ylabel("mean Vernier LLR margin / static")
    axes[1].set_ylabel("Vernier sign accuracy")
    axes[2].set_ylabel("trajectory prior N_eff")
    for ax in axes:
        ax.set_xlabel("across scale, along fixed at 1x")
        ax.set_xscale("symlog", linthresh=0.125, linscale=0.7)
        ax.set_xticks([0, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3])
        ax.set_xticklabels(["0", "0.125", "0.25", "0.5", "0.75", "1", "1.5", "2", "3"], rotation=35, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title(f"Evidence vs static, K={max_k}")
    axes[1].set_title("Accuracy")
    axes[2].set_title("Cue/prior width")
    axes[2].legend(title="sigma arcmin", frameon=False, fontsize=7, title_fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Leave-one-trajectory-out catalog observer: along-scale fixed at 1x", y=1.05)
    fig.savefig(out_dir / "rr100_heldout_along1_across_by_sigma.png", bbox_inches="tight")
    plt.close(fig)

    uniform = grid[np.isinf(grid["trajectory_weight_sigma_arcmin"]) & np.isclose(grid["along_scale"], 1.0)].copy()
    if uniform.empty:
        return
    fig2, axes2 = plt.subplots(1, 2, figsize=(9.4, 3.8), dpi=180, constrained_layout=True)
    for k, sub in uniform.groupby("n_catalog_trajectories"):
        sub = sub.sort_values("across_scale")
        label = f"K={int(k)}"
        axes2[0].plot(sub["across_scale"], sub["mean_joint_score_vs_static"], marker="o", label=label)
        axes2[1].plot(sub["across_scale"], sub["joint_accuracy"], marker="o", label=label)
    axes2[0].axhline(1.0, color="#333333", linestyle="--", linewidth=0.9)
    axes2[1].axhline(0.5, color="#333333", linestyle="--", linewidth=0.9)
    axes2[0].set_ylabel("uniform marginal LLR / static")
    axes2[1].set_ylabel("uniform marginal accuracy")
    for ax in axes2:
        ax.set_xlabel("across scale, along fixed at 1x")
        ax.set_xscale("symlog", linthresh=0.125, linscale=0.7)
        ax.set_xticks([0, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3])
        ax.set_xticklabels(["0", "0.125", "0.25", "0.5", "0.75", "1", "1.5", "2", "3"], rotation=35, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    axes2[1].legend(frameon=False, fontsize=8)
    fig2.suptitle("Leave-one-trajectory-out uniform catalog convergence", y=1.04)
    fig2.savefig(out_dir / "rr100_heldout_uniform_k_convergence.png", bbox_inches="tight")
    plt.close(fig2)


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caches = _load_rr100_caches(source_dir)
    meta = _condition_metadata(source_dir)
    if str(args.reference_condition) not in caches:
        raise FileNotFoundError(f"Missing reference condition cache: {args.reference_condition}")
    reference_cache = caches[str(args.reference_condition)]
    bin_seconds = (
        float(args.bin_seconds)
        if float(args.bin_seconds) > 0.0
        else float(reference_cache["bin_seconds"])
    )

    any_cache = next(iter(caches.values()))
    n_traces = int(np.asarray(any_cache["plus_rates"]).shape[0])
    max_k = max(int(k) for k in args.prior_k_list)
    n_prior_total = int(args.n_prior_traces)
    obs_indices, prior_indices = split_trace_indices(
        n_traces,
        n_observation_traces=int(args.n_observation_traces),
        n_prior_traces=n_prior_total,
        seed=int(args.split_seed),
    )

    conditions = args.conditions or sorted(caches)
    trial_rows: list[dict[str, Any]] = []
    for condition in conditions:
        if condition not in caches:
            raise FileNotFoundError(f"Missing condition cache: {condition}")
        for prior_k in args.prior_k_list:
            trial_rows.extend(
                _score_condition(
                    args,
                    condition=condition,
                    cache=caches[condition],
                    reference_cache=reference_cache,
                    metadata=meta.get(condition, {}),
                    obs_indices=obs_indices,
                    prior_indices_full=prior_indices,
                    prior_k=int(prior_k),
                    bin_seconds=bin_seconds,
                )
            )

    summary_df = add_static_ratios(pd.DataFrame(summarize_rows(trial_rows)))
    summary_rows = summary_df.to_dict("records")
    write_csv(out_dir / "rr100_heldout_trajectory_observer_trials.csv", trial_rows)
    summary_df.to_csv(out_dir / "rr100_heldout_trajectory_observer_summary.csv", index=False)
    manifest = {
        "source_dir": source_dir,
        "out_dir": out_dir,
        "reference_condition": str(args.reference_condition),
        "conditions": conditions,
        "trajectory_sigmas_arcmin": args.trajectory_sigmas_arcmin,
        "prior_k_list": args.prior_k_list,
        "n_observation_traces": int(args.n_observation_traces),
        "n_prior_traces_total": int(n_prior_total),
        "n_prior_traces_max_evaluated": int(max_k),
        "split_seed": int(args.split_seed),
        "observation_indices": obs_indices,
        "prior_indices": prior_indices,
        "bin_seconds": float(bin_seconds),
        "likelihood_scale": float(args.likelihood_scale),
        "observer_interpretation": (
            "Leave-one-trajectory-out empirical catalog approximation to Vernier trajectory "
            "nuisance marginalization. Sigma=0 is nearest retained catalog trajectory, not "
            "the pose-aware endpoint."
        ),
    }
    (out_dir / "rr100_heldout_trajectory_observer_manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_plots(out_dir, summary_df)
    return trial_rows, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--conditions", type=str, default="")
    parser.add_argument("--reference-condition", type=str, default="static_center")
    parser.add_argument("--trajectory-sigmas-arcmin", type=str, default=DEFAULT_SIGMAS)
    parser.add_argument("--prior-k-list", type=str, default=DEFAULT_PRIOR_K)
    parser.add_argument("--n-observation-traces", type=int, default=32)
    parser.add_argument("--n-prior-traces", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--likelihood-scale", type=float, default=1.0)
    parser.add_argument("--bin-seconds", type=float, default=0.0, help="Defaults to the cache value when <=0.")
    parser.add_argument("--max-timebins", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.conditions = [part.strip() for part in str(args.conditions).split(",") if part.strip()]
    args.trajectory_sigmas_arcmin = parse_csv_float(args.trajectory_sigmas_arcmin)
    args.prior_k_list = sorted(set(parse_csv_int(args.prior_k_list)))
    if max(args.prior_k_list) > int(args.n_prior_traces):
        raise ValueError("--n-prior-traces must be at least max(--prior-k-list)")
    trial_rows, summary_rows = run(args)
    print(
        f"Wrote {len(trial_rows)} held-out trajectory trials and {len(summary_rows)} summary rows",
        flush=True,
    )


if __name__ == "__main__":
    main()
