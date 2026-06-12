#!/usr/bin/env python3
"""Recorded V1 pose-aware response prediction from cached fixRSVP data.

This is the cache-first recorded-data bridge for the active-sensing figure. It
compares trial-disjoint held-out Poisson prediction for a small model ladder:

M0: time-bin PSTH GLM only
M_eye_only: eye features only
M1: PSTH + additive eye features
M2: PSTH + one scalar eye-state factor
M3: PSTH + additive eye features + coarse time-by-eye interactions

The analysis deliberately starts from ``outputs/cache/fig3_digitaltwin.pkl`` so
the first-pass recorded result does not require loading the digital twin. Live
finite-difference/tangent machinery should be added only as a geometry-linked
second stage if this smoke-test ladder is positive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dill
import numpy as np
from scipy.special import gammaln
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor

from declan.direct_recorded_derivative_twin_alignment.run_direct_recorded_derivative_alignment import (
    bootstrap_mean_ci,
    sign_test_p_two_sided,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIG3_CACHE = ROOT / "outputs" / "cache" / "fig3_digitaltwin.pkl"
DEFAULT_OUT_DIR = ROOT / "outputs" / "active_sensing_movie_information" / "recorded_pose_aware_prediction"
EPS_RATE = 1e-9


@dataclass
class PredictionConfig:
    fig3_cache: str
    output_root: str
    sessions: list[str]
    n_folds: int
    n_nulls: int
    n_bootstrap: int
    poisson_alpha: float
    augmented_alpha: float
    max_iter: int
    interaction_bin_size: int
    dfs_threshold: float
    min_train_samples: int
    min_test_samples: int
    min_units: int
    max_trials: int
    max_units: int
    seed: int
    include_trial_drift: bool


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def stable_seed(text: str) -> int:
    total = 0
    for idx, char in enumerate(str(text)):
        total = (total + (idx + 1) * ord(char)) % 1_000_003
    return int(total)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not keys:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return dill.load(handle)


def _standardize_from_train(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float64)
    mask = np.asarray(train_mask, dtype=bool)
    mean = np.nanmean(arr[mask], axis=0, keepdims=True)
    std = np.nanstd(arr[mask], axis=0, keepdims=True)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where((std > 1e-12) & np.isfinite(std), std, 1.0)
    z = (arr - mean) / std
    z = np.where(np.isfinite(z), z, 0.0)
    return z, mean.ravel(), std.ravel()


def _time_one_hot(time_ids: np.ndarray, n_time: int) -> np.ndarray:
    t = np.asarray(time_ids, dtype=np.int64)
    out = np.zeros((t.size, int(n_time)), dtype=np.float64)
    ok = (t >= 0) & (t < int(n_time))
    out[np.flatnonzero(ok), t[ok]] = 1.0
    return out


def _interaction_design(eye_z: np.ndarray, time_ids: np.ndarray, n_time: int, bin_size: int) -> np.ndarray:
    eye = np.asarray(eye_z, dtype=np.float64)
    bins = np.asarray(time_ids, dtype=np.int64) // max(int(bin_size), 1)
    n_bins = int(math.ceil(float(n_time) / max(int(bin_size), 1)))
    out = np.zeros((eye.shape[0], n_bins * eye.shape[1]), dtype=np.float64)
    for b in range(n_bins):
        rows = bins == b
        if np.any(rows):
            out[np.ix_(rows, np.arange(b * eye.shape[1], (b + 1) * eye.shape[1]))] = eye[rows]
    return out


def _trial_drift_features(trial_ids: np.ndarray) -> np.ndarray:
    x = np.asarray(trial_ids, dtype=np.float64)
    if x.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    span = max(float(np.max(x) - np.min(x)), 1.0)
    z = (x - float(np.min(x))) / span
    z = 2.0 * z - 1.0
    return np.stack([z, z * z], axis=1)


def _eye_features(eyepos: np.ndarray, valid_mask: np.ndarray, include_trial_drift: bool = False) -> np.ndarray:
    eye = np.asarray(eyepos, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(eye).all(axis=2)
    clean = np.where(valid[:, :, None], eye, np.nan)
    vel = np.full_like(clean, np.nan, dtype=np.float64)
    vel[:, 1:, :] = clean[:, 1:, :] - clean[:, :-1, :]
    vel[:, 0, :] = 0.0
    radius = np.linalg.norm(clean, axis=2, keepdims=True)
    speed = np.linalg.norm(vel, axis=2, keepdims=True)
    feats = [clean, vel, radius, speed]
    flat = np.concatenate(feats, axis=2).reshape(-1, 6)
    if include_trial_drift:
        n_trials, n_time = eye.shape[:2]
        trial_ids = np.repeat(np.arange(n_trials, dtype=np.int64), n_time)
        flat = np.concatenate([flat, _trial_drift_features(trial_ids)], axis=1)
    return flat


def _shuffle_eye_traces(
    eyepos: np.ndarray,
    valid_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eye = np.asarray(eyepos, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(eye).all(axis=2)
    if eye.shape[0] < 2:
        return eye.copy(), valid.copy(), np.arange(eye.shape[0], dtype=np.int64)
    donor_ids = np.arange(eye.shape[0], dtype=np.int64)
    shuffled = np.empty_like(eye)
    shuffled_valid = np.zeros_like(valid, dtype=bool)
    for trial in range(eye.shape[0]):
        needed = valid[trial]
        candidates = []
        for donor in range(eye.shape[0]):
            if donor == trial:
                continue
            if np.all(valid[donor, needed]):
                candidates.append(donor)
        if candidates:
            donor = int(rng.choice(np.asarray(candidates, dtype=np.int64)))
        else:
            donor = int(trial)
        donor_ids[trial] = donor
        shuffled[trial] = eye[donor]
        shuffled_valid[trial] = valid[donor]
    return shuffled, shuffled_valid, donor_ids


def _folds_from_trials(n_trials: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    trials = np.arange(int(n_trials), dtype=np.int64)
    rng.shuffle(trials)
    return [fold.astype(np.int64) for fold in np.array_split(trials, min(int(n_folds), int(n_trials))) if fold.size]


def _fit_predict_poisson(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    alpha: float,
    max_iter: int,
) -> tuple[np.ndarray, str, int]:
    if y_train.size == 0 or float(np.sum(y_train)) <= 0.0:
        lam = np.full(x_test.shape[0], max(float(np.mean(y_train)) if y_train.size else EPS_RATE, EPS_RATE), dtype=np.float64)
        return lam, "constant_no_train_spikes", 0
    model = PoissonRegressor(alpha=float(alpha), fit_intercept=False, max_iter=int(max_iter))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(np.asarray(x_train, dtype=np.float64), np.asarray(y_train, dtype=np.float64))
        pred = np.asarray(model.predict(np.asarray(x_test, dtype=np.float64)), dtype=np.float64)
        status = "ok" if int(getattr(model, "n_iter_", 0)) < int(max_iter) else "max_iter"
        return np.clip(pred, EPS_RATE, 1e9), status, int(getattr(model, "n_iter_", 0))
    except Exception as exc:
        lam = np.full(x_test.shape[0], max(float(np.mean(y_train)), EPS_RATE), dtype=np.float64)
        return lam, f"fit_failed:{type(exc).__name__}", 0


def _poisson_ll(y: np.ndarray, lam: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.float64)
    rr = np.clip(np.asarray(lam, dtype=np.float64), EPS_RATE, 1e9)
    return float(np.sum(yy * np.log(rr) - rr - gammaln(yy + 1.0)))


def _scalar_eye_factor(eye_z: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    x = np.asarray(eye_z, dtype=np.float64)
    train = np.asarray(train_mask, dtype=bool)
    if x.shape[1] == 0 or np.sum(train) < 3:
        return np.zeros((x.shape[0], 1), dtype=np.float64)
    u, _s, vh = np.linalg.svd(x[train], full_matrices=False)
    if vh.size == 0:
        return np.zeros((x.shape[0], 1), dtype=np.float64)
    score = x @ vh[0, :].reshape(-1, 1)
    std = float(np.std(score[train]))
    if std > 1e-12 and np.isfinite(std):
        score = (score - float(np.mean(score[train]))) / std
    return np.where(np.isfinite(score), score, 0.0)


def _model_designs(
    *,
    eye_feat: np.ndarray,
    time_ids: np.ndarray,
    train_mask: np.ndarray,
    n_time: int,
    interaction_bin_size: int,
) -> dict[str, np.ndarray]:
    time_x = _time_one_hot(time_ids, n_time)
    eye_z, _mean, _std = _standardize_from_train(eye_feat, train_mask)
    scalar = _scalar_eye_factor(eye_z, train_mask)
    interactions = _interaction_design(eye_z, time_ids, n_time, interaction_bin_size)
    intercept = np.ones((time_ids.size, 1), dtype=np.float64)
    return {
        "M0_psth_glm": time_x,
        "M_eye_only": np.concatenate([intercept, eye_z], axis=1),
        "M1_additive_eye": np.concatenate([time_x, eye_z], axis=1),
        "M2_scalar_eye_gain": np.concatenate([time_x, scalar], axis=1),
        "M3_time_by_eye_interaction": np.concatenate([time_x, eye_z, interactions], axis=1),
    }


def _evaluate_design(
    *,
    design: np.ndarray,
    y_all: np.ndarray,
    dfs_all: np.ndarray,
    time_ids: np.ndarray,
    base_sample_train: np.ndarray,
    base_sample_test: np.ndarray,
    n_time: int,
    alpha: float,
    max_iter: int,
    dfs_threshold: float,
    min_train_samples: int,
    min_test_samples: int,
) -> dict[str, Any]:
    robs = np.asarray(y_all, dtype=np.float64)
    dfs = np.asarray(dfs_all, dtype=np.float64)
    n_units = robs.shape[1]
    ll_sum = 0.0
    total_spikes = 0.0
    n_eval_obs = 0
    n_units_fit = 0
    n_fit_failures = 0
    n_constant_units = 0
    n_max_iter = 0
    iter_sum = 0
    for unit in range(n_units):
        finite_unit = np.isfinite(robs[:, unit]) & np.isfinite(dfs[:, unit]) & (dfs[:, unit] > float(dfs_threshold))
        train = np.asarray(base_sample_train, dtype=bool) & finite_unit
        test = np.asarray(base_sample_test, dtype=bool) & finite_unit
        if int(np.sum(train)) < int(min_train_samples) or int(np.sum(test)) < int(min_test_samples):
            continue
        pred, status, n_iter = _fit_predict_poisson(
            design[train],
            robs[train, unit],
            design[test],
            alpha=float(alpha),
            max_iter=int(max_iter),
        )
        y_test = robs[test, unit]
        ll_sum += _poisson_ll(y_test, pred)
        total_spikes += float(np.sum(y_test))
        n_eval_obs += int(y_test.size)
        n_units_fit += 1
        iter_sum += int(n_iter)
        if status.startswith("fit_failed"):
            n_fit_failures += 1
        if status.startswith("constant"):
            n_constant_units += 1
        if status == "max_iter":
            n_max_iter += 1
    return {
        "ll_sum": ll_sum,
        "total_spikes": total_spikes,
        "n_eval_observations": int(n_eval_obs),
        "n_units_fit": int(n_units_fit),
        "n_fit_failures": int(n_fit_failures),
        "n_constant_units": int(n_constant_units),
        "n_max_iter": int(n_max_iter),
        "mean_n_iter": float(iter_sum / max(n_units_fit, 1)),
    }


def _add_baseline_deltas(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    delta_ll = float(out.get("ll_sum", 0.0)) - float(baseline.get("ll_sum", 0.0))
    total_spikes = max(float(out.get("total_spikes", 0.0)), EPS_RATE)
    out["baseline_model_name"] = "M0_psth_glm"
    out["baseline_ll_sum"] = float(baseline.get("ll_sum", 0.0))
    out["delta_ll_vs_psth"] = delta_ll
    out["bits_per_spike_delta_vs_psth"] = delta_ll / total_spikes / math.log(2.0)
    return out


def _session_rows(
    *,
    session_row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    session = str(session_row["session"])
    subject = str(session_row.get("subject", ""))
    robs = np.asarray(session_row["robs_used"], dtype=np.float64)
    dfs = np.asarray(session_row.get("dfs_used", np.ones_like(robs)), dtype=np.float64)
    eyepos = np.asarray(session_row["eyepos_used"], dtype=np.float64)
    valid_mask = np.asarray(session_row["valid_mask"], dtype=bool)
    if int(args.max_trials) > 0:
        robs = robs[: int(args.max_trials)]
        dfs = dfs[: int(args.max_trials)]
        eyepos = eyepos[: int(args.max_trials)]
        valid_mask = valid_mask[: int(args.max_trials)]
    if int(args.max_units) > 0:
        robs = robs[:, :, : int(args.max_units)]
        dfs = dfs[:, :, : int(args.max_units)]
    n_trials, n_time, n_units = robs.shape
    trial_ids = np.repeat(np.arange(n_trials, dtype=np.int64), n_time)
    time_ids = np.tile(np.arange(n_time, dtype=np.int64), n_trials)
    robs_flat = robs.reshape(n_trials * n_time, n_units)
    dfs_flat = dfs.reshape(n_trials * n_time, n_units)
    base_valid = valid_mask.reshape(-1) & np.isfinite(eyepos.reshape(-1, 2)).all(axis=1)
    eye_feat = _eye_features(eyepos, valid_mask, include_trial_drift=bool(args.include_trial_drift))
    session_seed = stable_seed(session)
    folds = _folds_from_trials(n_trials, int(args.n_folds), int(args.seed) + session_seed)

    metric_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    for fold_id, test_trials in enumerate(folds):
        train_trials = np.setdiff1d(np.arange(n_trials, dtype=np.int64), test_trials)
        base_train = base_valid & np.isin(trial_ids, train_trials)
        base_test = base_valid & np.isin(trial_ids, test_trials)
        shared_trials = sorted(set(int(v) for v in train_trials).intersection(int(v) for v in test_trials))
        leakage_rows.append(
            {
                "session": session,
                "subject": subject,
                "fold_id": int(fold_id),
                "n_train_trials": int(train_trials.size),
                "n_test_trials": int(test_trials.size),
                "n_shared_trials": int(len(shared_trials)),
                "shared_trials": ";".join(str(v) for v in shared_trials[:50]),
                "n_train_samples_base": int(np.sum(base_train)),
                "n_test_samples_base": int(np.sum(base_test)),
                "status": "pass" if not shared_trials else "fail",
            }
        )
        if int(np.sum(base_train)) < int(args.min_train_samples) or int(np.sum(base_test)) < int(args.min_test_samples):
            continue

        base = {
            "session": session,
            "subject": subject,
            "fold_id": int(fold_id),
            "n_trials": int(n_trials),
            "n_time": int(n_time),
            "n_units_available": int(n_units),
            "n_train_trials": int(train_trials.size),
            "n_test_trials": int(test_trials.size),
            "poisson_alpha": float(args.poisson_alpha),
            "augmented_alpha": float(args.augmented_alpha),
            "interaction_bin_size": int(args.interaction_bin_size),
        }
        designs = _model_designs(
            eye_feat=eye_feat,
            time_ids=time_ids,
            train_mask=base_train,
            n_time=n_time,
            interaction_bin_size=int(args.interaction_bin_size),
        )
        fold_metrics: dict[str, dict[str, Any]] = {}
        for model_name, design in designs.items():
            fold_metrics[model_name] = _evaluate_design(
                design=design,
                y_all=robs_flat,
                dfs_all=dfs_flat,
                time_ids=time_ids,
                base_sample_train=base_train,
                base_sample_test=base_test,
                n_time=n_time,
                alpha=float(args.poisson_alpha if model_name == "M0_psth_glm" else args.augmented_alpha),
                max_iter=int(args.max_iter),
                dfs_threshold=float(args.dfs_threshold),
                min_train_samples=int(args.min_train_samples),
                min_test_samples=int(args.min_test_samples),
            )
        baseline = fold_metrics["M0_psth_glm"]
        for model_name, metrics in fold_metrics.items():
            metric_rows.append(
                {
                    **base,
                    "model_name": model_name,
                    "effective_alpha": float(args.poisson_alpha if model_name == "M0_psth_glm" else args.augmented_alpha),
                    "null_type": "observed",
                    "null_draw": -1,
                    "n_shuffle_self_donors": 0,
                    **_add_baseline_deltas(metrics, baseline),
                }
            )

        for draw in range(int(args.n_nulls)):
            rng = np.random.default_rng(int(args.seed) + int(fold_id) * 1009 + draw * 9173 + session_seed)
            shuffled_eye, _shuffled_valid, donor_ids = _shuffle_eye_traces(eyepos, valid_mask, rng)
            shuffled_feat = _eye_features(shuffled_eye, valid_mask, include_trial_drift=bool(args.include_trial_drift))
            shuffled_designs = _model_designs(
                eye_feat=shuffled_feat,
                time_ids=time_ids,
                train_mask=base_train,
                n_time=n_time,
                interaction_bin_size=int(args.interaction_bin_size),
            )
            for model_name in ("M1_additive_eye", "M3_time_by_eye_interaction"):
                metrics = _evaluate_design(
                    design=shuffled_designs[model_name],
                    y_all=robs_flat,
                    dfs_all=dfs_flat,
                    time_ids=time_ids,
                    base_sample_train=base_train,
                    base_sample_test=base_test,
                    n_time=n_time,
                    alpha=float(args.augmented_alpha),
                    max_iter=int(args.max_iter),
                    dfs_threshold=float(args.dfs_threshold),
                    min_train_samples=int(args.min_train_samples),
                    min_test_samples=int(args.min_test_samples),
                )
                null_rows.append(
                    {
                        **base,
                        "model_name": model_name,
                        "effective_alpha": float(args.augmented_alpha),
                        "null_type": "trial_trace_eye_shuffle",
                        "null_draw": int(draw),
                        "n_shuffle_self_donors": int(np.sum(donor_ids == np.arange(donor_ids.size))),
                        **_add_baseline_deltas(metrics, baseline),
                    }
                )

    observed_fit_rows = [r for r in metric_rows if str(r.get("null_type")) == "observed" and int(r.get("n_units_fit", 0)) > 0]
    session_summary = [
        {
            "session": session,
            "subject": subject,
            "status": "ok" if observed_fit_rows else "no_evaluable_units",
            "n_trials": int(n_trials),
            "n_time": int(n_time),
            "n_units": int(n_units),
            "n_folds_requested": int(args.n_folds),
            "n_folds_evaluable": int(len({int(r["fold_id"]) for r in observed_fit_rows})),
            "valid_sample_fraction": float(np.mean(base_valid)),
            "dfs_gt_threshold_fraction": float(np.mean(dfs_flat > float(args.dfs_threshold))),
            "include_trial_drift": bool(args.include_trial_drift),
        }
    ]
    return session_summary, metric_rows + null_rows, leakage_rows


def _session_model_means(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, float]]:
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        if float(row.get("total_spikes", 0.0)) <= 0.0:
            continue
        key = (str(row["session"]), str(row["model_name"]), str(row["null_type"]))
        groups.setdefault(key, []).append(float(row["bits_per_spike_delta_vs_psth"]))
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, vals in groups.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out[key] = {"mean": float(np.mean(arr)), "median": float(np.median(arr)), "n": int(arr.size)}
    return out


def _summarize_models(rows: list[dict[str, Any]], *, seed: int, n_bootstrap: int) -> list[dict[str, Any]]:
    means = _session_model_means(rows)
    model_names = sorted({key[1] for key in means})
    null_types = sorted({key[2] for key in means})
    rng = np.random.default_rng(int(seed))
    out: list[dict[str, Any]] = []
    for model_name in model_names:
        for null_type in null_types:
            by_session = [v["mean"] for (session, m, nt), v in means.items() if m == model_name and nt == null_type]
            vals = np.asarray(by_session, dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            mean, lo, hi = bootstrap_mean_ci(vals, rng=rng, n_bootstrap=int(n_bootstrap))
            out.append(
                {
                    "model_name": model_name,
                    "null_type": null_type,
                    "metric_name": "bits_per_spike_delta_vs_psth",
                    "n_sessions": int(vals.size),
                    "mean": mean,
                    "boot_ci_low": lo,
                    "boot_ci_high": hi,
                    "n_positive_sessions": int(np.sum(vals > 0.0)),
                    "sign_test_p_two_sided": sign_test_p_two_sided(int(np.sum(vals > 0.0)), int(vals.size)),
                }
            )
    return out


def _comparison_summary(rows: list[dict[str, Any]], *, seed: int, n_bootstrap: int) -> list[dict[str, Any]]:
    means = _session_model_means(rows)
    sessions = sorted({key[0] for key in means})
    comparisons = [
        ("M1_minus_eye_shuffle", "M1_additive_eye", "observed", "M1_additive_eye", "trial_trace_eye_shuffle"),
        ("M3_minus_eye_shuffle", "M3_time_by_eye_interaction", "observed", "M3_time_by_eye_interaction", "trial_trace_eye_shuffle"),
        ("M3_minus_M1", "M3_time_by_eye_interaction", "observed", "M1_additive_eye", "observed"),
        ("M3_minus_M2", "M3_time_by_eye_interaction", "observed", "M2_scalar_eye_gain", "observed"),
        ("M1_minus_eye_only", "M1_additive_eye", "observed", "M_eye_only", "observed"),
    ]
    rng = np.random.default_rng(int(seed) + 17)
    out: list[dict[str, Any]] = []
    for name, a_model, a_null, b_model, b_null in comparisons:
        vals = []
        a_stat = "mean"
        b_stat = "median" if b_null != "observed" else "mean"
        for session in sessions:
            a = means.get((session, a_model, a_null))
            b = means.get((session, b_model, b_null))
            if a is None or b is None:
                continue
            vals.append(float(a[a_stat]) - float(b[b_stat]))
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        mean, lo, hi = bootstrap_mean_ci(arr, rng=rng, n_bootstrap=int(n_bootstrap))
        out.append(
            {
                "comparison": name,
                "metric_name": "bits_per_spike_delta_difference",
                "left_model": a_model,
                "left_null_type": a_null,
                "left_session_stat": a_stat,
                "right_model": b_model,
                "right_null_type": b_null,
                "right_session_stat": b_stat,
                "n_sessions": int(arr.size),
                "mean": mean,
                "boot_ci_low": lo,
                "boot_ci_high": hi,
                "n_positive_sessions": int(np.sum(arr > 0.0)),
                "sign_test_p_two_sided": sign_test_p_two_sided(int(np.sum(arr > 0.0)), int(arr.size)),
            }
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Recorded pose-aware held-out Poisson response prediction")
    p.add_argument("--fig3-cache", type=Path, default=DEFAULT_FIG3_CACHE)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--sessions", type=str, default="Allen_2022-02-16", help="Comma-separated sessions, or all")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-nulls", type=int, default=5)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--poisson-alpha", type=float, default=1e-4, help="L2 penalty for the PSTH-only baseline GLM")
    p.add_argument("--augmented-alpha", type=float, default=1e-2, help="L2 penalty for eye and interaction models")
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--interaction-bin-size", type=int, default=10)
    p.add_argument("--dfs-threshold", type=float, default=0.5)
    p.add_argument("--min-train-samples", type=int, default=20)
    p.add_argument("--min-test-samples", type=int, default=5)
    p.add_argument("--min-units", type=int, default=10)
    p.add_argument("--max-trials", type=int, default=0, help="Smoke-test cap; 0 uses all trials")
    p.add_argument("--max-units", type=int, default=0, help="Smoke-test cap; 0 uses all units")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-trial-drift", action="store_true")
    p.add_argument("--init-only", action="store_true")
    return p


def run_analysis(args: argparse.Namespace) -> None:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    fig3_rows = _load_pickle(Path(args.fig3_cache))
    by_session = {str(row["session"]): row for row in fig3_rows}
    requested_sessions = parse_csv(args.sessions)
    if len(requested_sessions) == 1 and requested_sessions[0].lower() == "all":
        requested_sessions = [str(row["session"]) for row in fig3_rows if str(row.get("subject", "")) in {"Allen", "Logan"}]

    config = PredictionConfig(
        fig3_cache=str(Path(args.fig3_cache).resolve()),
        output_root=str(out.resolve()),
        sessions=requested_sessions,
        n_folds=int(args.n_folds),
        n_nulls=int(args.n_nulls),
        n_bootstrap=int(args.n_bootstrap),
        poisson_alpha=float(args.poisson_alpha),
        augmented_alpha=float(args.augmented_alpha),
        max_iter=int(args.max_iter),
        interaction_bin_size=int(args.interaction_bin_size),
        dfs_threshold=float(args.dfs_threshold),
        min_train_samples=int(args.min_train_samples),
        min_test_samples=int(args.min_test_samples),
        min_units=int(args.min_units),
        max_trials=int(args.max_trials),
        max_units=int(args.max_units),
        seed=int(args.seed),
        include_trial_drift=bool(args.include_trial_drift),
    )
    manifest_base = {
        "analysis": "recorded_pose_aware_response_prediction",
        "status": "initialized_not_run" if bool(args.init_only) else "running",
        "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "claim_guardrail": (
            "This cache-first runner tests whether measured eye state improves held-out recorded response "
            "prediction. Treat additive-eye gains as pose-linked predictability; translation-specific claims "
            "require M3 to beat additive/gain/shuffled controls and a later geometry enrichment analysis."
        ),
    }
    write_json(out / "recorded_pose_aware_prediction_manifest.json", manifest_base)
    if bool(args.init_only):
        (out / "README.md").write_text(
            "# Recorded Pose-Aware Prediction Outputs\n\nStatus: initialized, not run.\n",
            encoding="utf-8",
        )
        return

    session_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    for session in requested_sessions:
        print(f"[pose-aware-prediction] session {session}: starting", flush=True)
        sr = by_session.get(session)
        if sr is None:
            session_rows.append({"session": session, "status": "missing_fig3_cache"})
            continue
        n_units_available = int(np.asarray(sr["robs_used"]).shape[2])
        if n_units_available < int(args.min_units):
            session_rows.append({"session": session, "status": "too_few_units", "n_units": n_units_available})
            continue
        s_rows, m_rows, l_rows = _session_rows(session_row=sr, args=args)
        session_rows.extend(s_rows)
        metric_rows.extend(m_rows)
        leakage_rows.extend(l_rows)
        print(f"[pose-aware-prediction] session {session}: finished ({len(m_rows)} metric/null rows)", flush=True)

    summary_rows = _summarize_models(metric_rows, seed=int(args.seed), n_bootstrap=int(args.n_bootstrap))
    comparison_rows = _comparison_summary(metric_rows, seed=int(args.seed), n_bootstrap=int(args.n_bootstrap))
    write_csv(out / "session_summary.csv", session_rows)
    write_csv(out / "fold_model_metrics.csv", [r for r in metric_rows if str(r.get("null_type")) == "observed"])
    write_csv(out / "fold_eye_shuffle_nulls.csv", [r for r in metric_rows if str(r.get("null_type")) != "observed"])
    write_csv(out / "leakage_audit.csv", leakage_rows)
    write_csv(out / "model_bootstrap_summary.csv", summary_rows)
    write_csv(out / "model_comparison_summary.csv", comparison_rows)
    (out / "README.md").write_text(
        "# Recorded Pose-Aware Prediction Outputs\n\n"
        "Primary tables:\n\n"
        "- `fold_model_metrics.csv`: held-out Poisson LL and bits/spike deltas versus PSTH.\n"
        "- `fold_eye_shuffle_nulls.csv`: trial-trace shuffled-eye controls for M1 and M3.\n"
        "- `model_bootstrap_summary.csv`: session-bootstrap model summaries.\n"
        "- `model_comparison_summary.csv`: real-eye/shuffle and model-ladder contrasts.\n"
        "- `leakage_audit.csv`: trial-disjoint split audit.\n\n"
        "Penalty audit: `poisson_alpha` is used for the fitted M0 PSTH GLM; "
        "`augmented_alpha` is used for the eye and interaction models, with "
        "`effective_alpha` written per row. Because augmented models include "
        "PSTH columns but use `augmented_alpha`, M0-relative deltas are not a "
        "pure nested eye-covariate test when the two penalties differ; prefer "
        "real-eye versus shuffled-eye contrasts for the main smoke-test null.\n\n"
        "Interpretation guardrail: additive-eye gains are pose-linked predictability, not by themselves "
        "translation-specific active-sensing evidence.\n",
        encoding="utf-8",
    )
    write_json(
        out / "recorded_pose_aware_prediction_manifest.json",
        {
            **manifest_base,
            "status": "ok",
            "n_sessions_requested": int(len(requested_sessions)),
            "n_sessions_ok": int(sum(1 for r in session_rows if r.get("status") == "ok")),
            "n_metric_rows": int(len(metric_rows)),
            "n_leakage_rows": int(len(leakage_rows)),
        },
    )
    print(str(out / "model_comparison_summary.csv"))


def main() -> None:
    run_analysis(build_parser().parse_args())


if __name__ == "__main__":
    main()
