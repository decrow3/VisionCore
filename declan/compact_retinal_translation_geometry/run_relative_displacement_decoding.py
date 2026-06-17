#!/usr/bin/env python3
"""Matched-context relative displacement decoding in recorded V1.

This runner implements the recorded-data "readability" bridge for the compact
retinal-translation geometry analysis. It pairs repeats from a matched context,
decodes the relative eye-position difference from the relative recorded
response, and compares compact twin-tangent features with full-population,
orthogonal, random, unit-shuffled, RF/readout-preserving, and global/PC
controls.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from declan.direct_recorded_derivative_twin_alignment.run_direct_recorded_derivative_alignment import (
    bootstrap_mean_ci,
    fixed_within_bin_permutation,
    orth,
    parse_int_list,
    parse_str_list,
    sign_test_p_two_sided,
    tangent_matrix_from_samples,
)
from declan.matched_twin_covariance_closure.run_cache_closure import (
    DEFAULT_FIG2_CACHE,
    DEFAULT_FIG3_CACHE,
    _fig2_by_session,
    _load_pickle,
    _projection_complement,
    _projection_modes,
)
from declan.matched_twin_covariance_closure.run_finite_difference_closure import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_OUT as DEFAULT_FD_OUT,
    _collect_samples,
    _compute_jacobians,
    _fit_rescale_gains,
    _load_twin_model,
    _rf_null_metadata_for_session,
    _target_for_session,
)


DEFAULT_OUTPUT_ROOT = (
    Path("outputs") / "compact_retinal_translation_geometry" / "relative_displacement_decoding"
)


@dataclass
class DecodeConfig:
    output_root: str
    sessions: list[str]
    projection_controls: list[str]
    target_variant: str
    k_list: list[int]
    primary_k: int
    context_mode: str
    context_bin_size: int
    min_repeats_per_condition: int
    max_pairs_per_condition: int
    split_mode: str
    n_folds: int
    n_nulls: int
    n_bootstrap: int
    seed: int


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


def context_labels(samples: Any, mode: str, bin_size: int) -> np.ndarray:
    if mode == "time_bin":
        return np.asarray(samples.time_indices, dtype=np.int64)
    if mode == "time_window":
        return np.asarray(samples.time_indices, dtype=np.int64) // max(int(bin_size), 1)
    if mode == "trial":
        return np.asarray(samples.trial_ids, dtype=np.int64)
    raise ValueError(f"Unsupported context mode: {mode}")


def _condition_keys(
    *,
    image_ids: np.ndarray,
    time_indices: np.ndarray,
    mode: str,
    bin_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    image_ids = np.asarray(image_ids, dtype=np.int64)
    time_indices = np.asarray(time_indices, dtype=np.int64)
    if str(mode) == "image_only":
        time_ctx = np.zeros_like(time_indices)
    elif str(mode) == "image_time_window":
        time_ctx = time_indices // max(int(bin_size), 1)
    elif str(mode) == "image_time_bin":
        time_ctx = time_indices
    else:
        raise ValueError(f"Unsupported image-aware context mode: {mode}")

    labels = np.full(image_ids.shape, -1, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    mapping: dict[tuple[int, int], int] = {}
    for i, (img, tt) in enumerate(zip(image_ids.tolist(), time_ctx.tolist(), strict=True)):
        if int(img) < 0:
            continue
        key = (int(img), int(tt))
        if key not in mapping:
            mapping[key] = len(mapping)
        labels[i] = mapping[key]
    for (img, tt), cid in sorted(mapping.items(), key=lambda kv: kv[1]):
        rows.append(
            {
                "condition_id": int(cid),
                "image_id": int(img),
                "time_context": int(tt),
                "condition_label": (
                    f"image_{int(img)}"
                    if str(mode) == "image_only"
                    else f"image_{int(img)}_time_{int(tt)}"
                ),
            }
        )
    return labels, rows


def _image_ids_for_samples(dset: Any, samples: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct fixRSVP image id at each sampled source row."""
    try:
        from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
        from DataYatesV1.utils.general import get_clock_functions
    except Exception as exc:
        return (
            np.full(samples.source_indices.size, -1, dtype=np.int64),
            {"image_id_status": f"unavailable_import_{type(exc).__name__}"},
        )

    trial_inds = np.asarray(dset.covariates["trial_inds"]).ravel()
    t_bins = np.asarray(dset.covariates["t_bins"]).ravel()
    sess = dset.metadata["sess"]
    ptb2ephys, _ = get_clock_functions(sess.exp)
    trial_cache: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    out = np.full(samples.source_indices.size, -1, dtype=np.int64)
    failures = 0
    for row_i, src in enumerate(np.asarray(samples.source_indices, dtype=np.int64)):
        trial_id = int(trial_inds[int(src)])
        try:
            if trial_id not in trial_cache:
                trial = FixRsvpTrial(sess.exp["D"][trial_id], sess.exp["S"])
                start = np.where(trial.image_ids == 2)[0]
                if start.size == 0:
                    failures += 1
                    continue
                start_idx = int(start[0])
                trial_cache[trial_id] = (
                    np.asarray(trial.image_ids, dtype=np.int64),
                    np.asarray(ptb2ephys(trial.flip_times[start_idx:]), dtype=np.float64),
                    start_idx,
                )
            trial_image_ids, flip_times, start_idx = trial_cache[trial_id]
            hist_idx = int(np.searchsorted(flip_times, float(t_bins[int(src)]), side="right") - 1 + start_idx)
            hist_idx = int(np.clip(hist_idx, 0, trial_image_ids.size - 1))
            out[row_i] = int(trial_image_ids[hist_idx]) - 1
        except Exception:
            failures += 1
    valid = out >= 0
    meta = {
        "image_id_status": "ok" if np.any(valid) else "no_valid_image_ids",
        "n_image_id_valid_samples": int(np.sum(valid)),
        "n_image_id_failures": int(failures),
        "n_unique_image_ids": int(np.unique(out[valid]).size) if np.any(valid) else 0,
    }
    return out, meta


def pair_condition_labels(
    *,
    dset: Any,
    samples: Any,
    mode: str,
    bin_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    if str(mode) in {"time_bin", "time_window", "trial"}:
        labels = context_labels(samples, str(mode), int(bin_size))
        rows: list[dict[str, Any]] = []
        for cid in sorted(int(v) for v in np.unique(labels)):
            rows.append(
                {
                    "condition_id": int(cid),
                    "image_id": -1,
                    "time_context": int(cid),
                    "condition_label": _label_text(str(mode), int(cid), int(bin_size)),
                }
            )
        return labels, rows, {"image_id_status": "not_requested"}
    image_ids, image_meta = _image_ids_for_samples(dset, samples)
    labels, rows = _condition_keys(
        image_ids=image_ids,
        time_indices=np.asarray(samples.time_indices, dtype=np.int64),
        mode=str(mode),
        bin_size=int(bin_size),
    )
    return labels, rows, image_meta


def _label_text(mode: str, condition_id: int, bin_size: int) -> str:
    if mode == "time_window":
        lo = int(condition_id) * int(bin_size)
        return f"time_window_{lo}_{lo + int(bin_size) - 1}"
    return f"{mode}_{int(condition_id)}"


def _condition_folds(condition_ids: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    ids = np.asarray(condition_ids, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    return [fold.astype(np.int64) for fold in np.array_split(shuffled, min(int(n_folds), ids.size)) if fold.size]


def _trial_pair_keys(trial_a: np.ndarray, trial_b: np.ndarray, mask: np.ndarray) -> set[tuple[int, int]]:
    a = np.asarray(trial_a, dtype=np.int64)[mask]
    b = np.asarray(trial_b, dtype=np.int64)[mask]
    return set((int(min(x, y)), int(max(x, y))) for x, y in zip(a, b, strict=False))


def _trial_set(trial_a: np.ndarray, trial_b: np.ndarray, mask: np.ndarray) -> set[int]:
    a = np.asarray(trial_a, dtype=np.int64)[mask]
    b = np.asarray(trial_b, dtype=np.int64)[mask]
    return set(int(v) for v in np.concatenate([a, b])) if a.size or b.size else set()


def projection_target_covariance(
    *,
    j: np.ndarray,
    train_sample_mask: np.ndarray,
    fallback_target_cov: np.ndarray,
) -> np.ndarray:
    mat = tangent_matrix_from_samples(j, train_sample_mask, projection=None).T
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] < 3 or mat.shape[1] != fallback_target_cov.shape[0]:
        return np.asarray(fallback_target_cov, dtype=np.float64)
    mat = mat[np.all(np.isfinite(mat), axis=1)]
    if mat.shape[0] < 3:
        return np.asarray(fallback_target_cov, dtype=np.float64)
    mat = mat - np.mean(mat, axis=0, keepdims=True)
    cov = (mat.T @ mat) / max(mat.shape[0] - 1, 1)
    cov = 0.5 * (cov + cov.T)
    if not np.all(np.isfinite(cov)):
        return np.asarray(fallback_target_cov, dtype=np.float64)
    return cov


def _decode_splits(pairs: dict[str, np.ndarray], n_folds: int, seed: int, split_mode: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if split_mode == "condition_disjoint":
        unique_conditions = np.unique(pairs["condition_id"])
        out = []
        for fold in _condition_folds(unique_conditions, int(n_folds), int(seed)):
            test_mask = np.isin(pairs["condition_id"], fold)
            train_mask = ~test_mask
            out.append((fold, train_mask, test_mask))
        return out
    if split_mode == "trial_disjoint":
        trials = np.unique(np.concatenate([pairs["trial_a"], pairs["trial_b"]]))
        out = []
        for fold in _condition_folds(trials, int(n_folds), int(seed)):
            a_test = np.isin(pairs["trial_a"], fold)
            b_test = np.isin(pairs["trial_b"], fold)
            test_mask = a_test & b_test
            train_mask = (~a_test) & (~b_test)
            if np.sum(train_mask) >= 5 and np.sum(test_mask) >= 3:
                out.append((fold, train_mask, test_mask))
        return out
    raise ValueError(f"Unsupported split mode: {split_mode}")


def build_pair_dataset(
    *,
    samples: Any,
    eye_px: np.ndarray,
    labels: np.ndarray,
    condition_rows: list[dict[str, Any]],
    context_mode: str,
    context_bin_size: int,
    min_repeats_per_condition: int,
    max_pairs_per_condition: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    dy_rows: list[np.ndarray] = []
    de_rows: list[np.ndarray] = []
    pair_condition_ids: list[int] = []
    trial_a_rows: list[int] = []
    trial_b_rows: list[int] = []
    sample_a_rows: list[int] = []
    sample_b_rows: list[int] = []
    inventory: list[dict[str, Any]] = []
    robs = np.asarray(samples.robs, dtype=np.float64)
    eye = np.asarray(eye_px, dtype=np.float64)
    valid = np.isfinite(robs).all(axis=1) & np.isfinite(eye).all(axis=1)
    condition_meta = {int(row["condition_id"]): row for row in condition_rows}
    for condition_id in np.unique(labels):
        idx = np.flatnonzero((labels == int(condition_id)) & valid)
        status = "ok"
        if idx.size < int(min_repeats_per_condition):
            status = "too_few_repeats"
        pairs: list[tuple[int, int]] = []
        if status == "ok":
            for pos_i in range(idx.size):
                for pos_j in range(pos_i + 1, idx.size):
                    a = int(idx[pos_i])
                    b = int(idx[pos_j])
                    if int(samples.trial_ids[a]) == int(samples.trial_ids[b]):
                        continue
                    pairs.append((a, b))
            if not pairs:
                status = "no_cross_trial_pairs"
        n_all_pairs = len(pairs)
        if pairs and int(max_pairs_per_condition) > 0 and len(pairs) > int(max_pairs_per_condition):
            keep = np.sort(rng.choice(len(pairs), size=int(max_pairs_per_condition), replace=False))
            pairs = [pairs[int(k)] for k in keep]
        for a, b in pairs:
            dy_rows.append(robs[a] - robs[b])
            de_rows.append(eye[a] - eye[b])
            pair_condition_ids.append(int(condition_id))
            trial_a_rows.append(int(samples.trial_ids[a]))
            trial_b_rows.append(int(samples.trial_ids[b]))
            sample_a_rows.append(int(a))
            sample_b_rows.append(int(b))
        inventory.append(
            {
                "condition_id": int(condition_id),
                "condition_label": str(condition_meta.get(int(condition_id), {}).get("condition_label", _label_text(context_mode, int(condition_id), int(context_bin_size)))),
                "image_id": int(condition_meta.get(int(condition_id), {}).get("image_id", -1)),
                "time_context": int(condition_meta.get(int(condition_id), {}).get("time_context", -1)),
                "n_repeats": int(idx.size),
                "n_trials": int(np.unique(samples.trial_ids[idx]).size) if idx.size else 0,
                "n_all_cross_trial_pairs": int(n_all_pairs),
                "n_pairs_used": int(len(pairs)),
                "time_min": int(np.min(samples.time_indices[idx])) if idx.size else -1,
                "time_max": int(np.max(samples.time_indices[idx])) if idx.size else -1,
                "status": status,
            }
        )
    if not dy_rows:
        empty = {
            "delta_y": np.zeros((0, robs.shape[1]), dtype=np.float64),
            "delta_e": np.zeros((0, 2), dtype=np.float64),
            "condition_id": np.zeros(0, dtype=np.int64),
            "trial_a": np.zeros(0, dtype=np.int64),
            "trial_b": np.zeros(0, dtype=np.int64),
            "sample_a": np.zeros(0, dtype=np.int64),
            "sample_b": np.zeros(0, dtype=np.int64),
        }
        return empty, inventory
    return (
        {
            "delta_y": np.stack(dy_rows, axis=0).astype(np.float64),
            "delta_e": np.stack(de_rows, axis=0).astype(np.float64),
            "condition_id": np.asarray(pair_condition_ids, dtype=np.int64),
            "trial_a": np.asarray(trial_a_rows, dtype=np.int64),
            "trial_b": np.asarray(trial_b_rows, dtype=np.int64),
            "sample_a": np.asarray(sample_a_rows, dtype=np.int64),
            "sample_b": np.asarray(sample_b_rows, dtype=np.int64),
        },
        inventory,
    )


def _ridge_lambda_grid(x_train: np.ndarray) -> np.ndarray:
    x = np.asarray(x_train, dtype=np.float64)
    n_features = max(int(x.shape[1]), 1)
    scale = float(np.trace(x.T @ x) / n_features)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return np.logspace(-4, 4, 17, dtype=np.float64) * scale


def _standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    std = np.where(std > 1e-12, std, 1.0)
    return (x_train - mean) / std, (x_test - mean) / std


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, lam: float) -> np.ndarray:
    xtx = x_train.T @ x_train
    return np.linalg.solve(xtx + float(lam) * np.eye(xtx.shape[0], dtype=np.float64), x_train.T @ y_train)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    out: dict[str, float] = {}
    ss_res = np.sum((y - p) ** 2, axis=0)
    ss_tot = np.sum((y - np.mean(y, axis=0, keepdims=True)) ** 2, axis=0)
    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, np.nan)
    out["R2_x"] = float(r2[0])
    out["R2_y"] = float(r2[1])
    out["R2_mean"] = float(np.nanmean(r2))
    out["RMSE_x"] = float(np.sqrt(np.nanmean((y[:, 0] - p[:, 0]) ** 2)))
    out["RMSE_y"] = float(np.sqrt(np.nanmean((y[:, 1] - p[:, 1]) ** 2)))
    for idx, axis in enumerate(("x", "y")):
        if np.std(y[:, idx]) > 1e-12 and np.std(p[:, idx]) > 1e-12:
            out[f"Pearson_r_{axis}"] = float(np.corrcoef(y[:, idx], p[:, idx])[0, 1])
        else:
            out[f"Pearson_r_{axis}"] = float("nan")
    yy = y.ravel()
    pp = p.ravel()
    out["2D_vector_correlation"] = float(np.corrcoef(yy, pp)[0, 1]) if np.std(yy) > 1e-12 and np.std(pp) > 1e-12 else float("nan")
    out["sign_accuracy_x"] = float(np.mean(np.sign(y[:, 0]) == np.sign(p[:, 0])))
    out["sign_accuracy_y"] = float(np.mean(np.sign(y[:, 1]) == np.sign(p[:, 1])))
    out["quadrant_accuracy"] = float(np.mean((np.sign(y[:, 0]) == np.sign(p[:, 0])) & (np.sign(y[:, 1]) == np.sign(p[:, 1]))))
    y_norm = np.linalg.norm(y, axis=1)
    p_norm = np.linalg.norm(p, axis=1)
    out["magnitude_correlation"] = float(np.corrcoef(y_norm, p_norm)[0, 1]) if np.std(y_norm) > 1e-12 and np.std(p_norm) > 1e-12 else float("nan")
    dot = np.sum(y * p, axis=1)
    den = np.linalg.norm(y, axis=1) * np.linalg.norm(p, axis=1)
    ok = den > 1e-12
    out["angular_error_rad"] = float(np.nanmean(np.arccos(np.clip(dot[ok] / den[ok], -1.0, 1.0)))) if np.any(ok) else float("nan")
    return out


def _inner_select_lambda(
    x_train: np.ndarray,
    y_train: np.ndarray,
    condition_train: np.ndarray,
    seed: int,
) -> float:
    lambdas = _ridge_lambda_grid(x_train)
    unique_conditions = np.unique(condition_train)
    if unique_conditions.size < 3 or x_train.shape[0] < 10:
        return float(lambdas[len(lambdas) // 2])
    folds = _condition_folds(unique_conditions, min(3, unique_conditions.size), int(seed))
    scores = np.zeros(lambdas.size, dtype=np.float64)
    counts = np.zeros(lambdas.size, dtype=np.int64)
    for held_conditions in folds:
        test = np.isin(condition_train, held_conditions)
        train = ~test
        if np.sum(train) < 5 or np.sum(test) < 3:
            continue
        xtr, xte = _standardize_train_test(x_train[train], x_train[test])
        ytr = y_train[train]
        yte = y_train[test]
        for li, lam in enumerate(lambdas):
            try:
                w = _fit_ridge(xtr, ytr, float(lam))
            except np.linalg.LinAlgError:
                continue
            pred = xte @ w
            scores[li] += _metrics(yte, pred)["R2_mean"]
            counts[li] += 1
    valid = counts > 0
    if not np.any(valid):
        return float(lambdas[len(lambdas) // 2])
    mean_scores = np.full(lambdas.size, -np.inf, dtype=np.float64)
    mean_scores[valid] = scores[valid] / counts[valid]
    return float(lambdas[int(np.nanargmax(mean_scores))])


def _decode_fold(
    x: np.ndarray,
    y: np.ndarray,
    condition_ids: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    seed: int,
    y_train_override: np.ndarray | None = None,
    x_train_override: np.ndarray | None = None,
) -> tuple[dict[str, float], float]:
    x_train = np.asarray(x_train_override if x_train_override is not None else x[train_mask], dtype=np.float64)
    x_test = np.asarray(x[test_mask], dtype=np.float64)
    y_train = np.asarray(y_train_override if y_train_override is not None else y[train_mask], dtype=np.float64)
    y_test = np.asarray(y[test_mask], dtype=np.float64)
    keep_col = np.isfinite(x_train).all(axis=0) & np.isfinite(x_test).all(axis=0)
    if np.sum(keep_col) == 0 or x_train.shape[0] < 5 or x_test.shape[0] < 3:
        return {}, float("nan")
    x_train = x_train[:, keep_col]
    x_test = x_test[:, keep_col]
    x_train_s, x_test_s = _standardize_train_test(x_train, x_test)
    lam = _inner_select_lambda(x_train_s, y_train, condition_ids[train_mask], int(seed))
    try:
        w = _fit_ridge(x_train_s, y_train, lam)
    except np.linalg.LinAlgError:
        return {}, float(lam)
    pred = x_test_s @ w
    return _metrics(y_test, pred), float(lam)


def _basis_from_j(j: np.ndarray, train_sample_mask: np.ndarray, projection: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    mat = tangent_matrix_from_samples(j, train_sample_mask, projection)
    if mat.shape[1] == 0 or not np.isfinite(mat).all():
        return np.zeros((j.shape[1], 0), dtype=np.float64), 0
    vals, vecs = np.linalg.eigh(0.5 * (mat @ mat.T + (mat @ mat.T).T))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    rank = int(np.sum(vals > max(float(vals[0]) if vals.size else 0.0, 1.0) * 1e-10))
    return orth(vecs[:, : min(int(k), rank, vecs.shape[1])]), rank


def _feature_specs(
    *,
    delta_y: np.ndarray,
    compact_basis: np.ndarray,
    projection: np.ndarray,
    modes: np.ndarray,
    k: int,
    rng: np.random.Generator,
    rf_bins: np.ndarray | None,
) -> list[tuple[str, int, str, np.ndarray]]:
    y_proj = delta_y @ projection.T
    specs: list[tuple[str, int, str, np.ndarray]] = [
        ("full_population", 0, "observed", y_proj),
    ]
    u = orth(compact_basis[:, : min(int(k), compact_basis.shape[1])])
    if u.shape[1] > 0:
        specs.append(("compact", int(k), "observed", y_proj @ u))
        residual = projection - u @ u.T
        specs.append(("orthogonal_complement", int(k), "specificity_control", delta_y @ residual.T))
        q, _ = np.linalg.qr(rng.standard_normal((delta_y.shape[1], max(int(k), 1))))
        specs.append(("random_subspace", int(k), "basis_control", y_proj @ q[:, : int(k)]))
        perm = rng.permutation(delta_y.shape[1])
        specs.append(("unit_shuffled_compact", int(k), "basis_control", y_proj @ u[perm, :]))
        if rf_bins is not None:
            rf_perm = fixed_within_bin_permutation(np.asarray(rf_bins, dtype=np.int64), rng)
            specs.append(("rf_readout_permuted_compact", int(k), "basis_control", y_proj @ u[rf_perm, :]))
    if modes.shape[1] > 0:
        specs.append(("global_top_pc_modes", int(modes.shape[1]), "global_pc_control", delta_y @ orth(modes)))
    return specs


def _session_decode_rows(
    *,
    session: str,
    subject: str,
    pairs: dict[str, np.ndarray],
    j: np.ndarray,
    labels: np.ndarray,
    samples: Any,
    target_cov: np.ndarray,
    projection_controls: list[str],
    k_list: list[int],
    primary_k: int,
    rf_bins: np.ndarray | None,
    split_mode: str,
    n_folds: int,
    n_nulls: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    condition_ids = pairs["condition_id"]
    delta_y = pairs["delta_y"]
    delta_e = pairs["delta_e"]
    folds = _decode_splits(pairs, int(n_folds), int(seed), str(split_mode))
    sample_condition_labels = labels
    for fold_idx, (held_ids, train_mask, test_mask) in enumerate(folds):
        train_conditions = set(int(v) for v in np.unique(condition_ids[train_mask]))
        test_condition_set = set(int(v) for v in np.unique(condition_ids[test_mask]))
        shared_conditions = sorted(train_conditions.intersection(test_condition_set))
        train_trials = _trial_set(pairs["trial_a"], pairs["trial_b"], train_mask)
        test_trials = _trial_set(pairs["trial_a"], pairs["trial_b"], test_mask)
        shared_trials = sorted(train_trials.intersection(test_trials))
        train_trial_pairs = _trial_pair_keys(pairs["trial_a"], pairs["trial_b"], train_mask)
        test_trial_pairs = _trial_pair_keys(pairs["trial_a"], pairs["trial_b"], test_mask)
        shared_trial_pairs = sorted(train_trial_pairs.intersection(test_trial_pairs))
        condition_status = "pass" if not shared_conditions else ("warn" if split_mode == "trial_disjoint" else "fail")
        trial_status = "pass" if not shared_trials else "warn"
        trial_pair_status = "pass" if not shared_trial_pairs else "warn"
        status = "fail" if (split_mode == "condition_disjoint" and shared_conditions) or (split_mode == "trial_disjoint" and shared_trials) else "pass"
        leakage_rows.append(
            {
                "session": session,
                "fold_id": int(fold_idx),
                "split_mode": str(split_mode),
                "n_train_conditions": int(len(train_conditions)),
                "n_test_conditions": int(len(test_condition_set)),
                "n_shared_conditions": int(len(shared_conditions)),
                "shared_conditions": ";".join(str(v) for v in shared_conditions),
                "n_train_trials": int(len(train_trials)),
                "n_test_trials": int(len(test_trials)),
                "n_shared_trials": int(len(shared_trials)),
                "shared_trials": ";".join(str(v) for v in shared_trials[:50]),
                "n_shared_trial_pairs": int(len(shared_trial_pairs)),
                "shared_trial_pairs": ";".join(f"{a}:{b}" for a, b in shared_trial_pairs[:50]),
                "n_train_pairs": int(np.sum(train_mask)),
                "n_test_pairs": int(np.sum(test_mask)),
                "condition_status": condition_status,
                "trial_overlap_status": trial_status,
                "trial_pair_overlap_status": trial_pair_status,
                "status": status,
            }
        )
        if split_mode == "trial_disjoint":
            train_sample_mask = ~np.isin(samples.trial_ids, held_ids)
        else:
            train_sample_mask = np.isin(sample_condition_labels, list(train_conditions))
        for projection_control in projection_controls:
            fold_target_cov = projection_target_covariance(
                j=j,
                train_sample_mask=train_sample_mask,
                fallback_target_cov=target_cov,
            )
            modes = _projection_modes(str(projection_control), fold_target_cov)
            projection = _projection_complement(delta_y.shape[1], modes)
            basis_cache: dict[int, tuple[np.ndarray, int]] = {}
            for k in sorted(set([int(primary_k), *[int(v) for v in k_list]])):
                basis_cache[k] = _basis_from_j(j, train_sample_mask, projection, k)
            for k in sorted(set([int(primary_k), *[int(v) for v in k_list]])):
                compact_basis, basis_rank = basis_cache[k]
                rng = np.random.default_rng(int(seed) + int(fold_idx) * 10007 + int(k) * 1009 + len(metric_rows))
                for feature_space, feature_k, feature_role, x in _feature_specs(
                    delta_y=delta_y,
                    compact_basis=compact_basis,
                    projection=projection,
                    modes=modes,
                    k=k,
                    rng=rng,
                    rf_bins=rf_bins,
                ):
                    if k != int(primary_k) and feature_space not in {"compact", "random_subspace"}:
                        continue
                    metrics, lam = _decode_fold(
                        x,
                        delta_e,
                        condition_ids,
                        train_mask,
                        test_mask,
                        seed=int(seed) + int(fold_idx) * 97 + int(k) * 13,
                    )
                    if not metrics:
                        continue
                    base = {
                        "session": session,
                        "subject": subject,
                        "fold_id": int(fold_idx),
                        "feature_space": feature_space,
                        "feature_role": feature_role,
                        "k": int(feature_k),
                        "basis_rank": int(basis_rank),
                        "projection_control": projection_control,
                        "decoder": "ridge",
                        "lambda_selected": lam,
                        "split_mode": str(split_mode),
                        "n_train_conditions": int(len(train_conditions)),
                        "n_test_conditions": int(len(test_condition_set)),
                        "n_train_pairs": int(np.sum(train_mask)),
                        "n_test_pairs": int(np.sum(test_mask)),
                    }
                    for name, value in metrics.items():
                        metric_rows.append({**base, "metric_name": name, "metric_value": value, "null_type": "observed", "null_draw": -1})
                    if feature_space in {"full_population", "compact"}:
                        for null_type in ("eye_label_shuffle", "response_pair_shuffle"):
                            for draw in range(int(n_nulls)):
                                rng_null = np.random.default_rng(
                                    int(seed) + draw * 7919 + int(fold_idx) * 733 + int(k) * 31 + (0 if null_type == "eye_label_shuffle" else 199)
                                )
                                if null_type == "eye_label_shuffle":
                                    y_train_null = delta_e[train_mask].copy()
                                    for cond in np.unique(condition_ids[train_mask]):
                                        local = np.flatnonzero(condition_ids[train_mask] == int(cond))
                                        if local.size > 1:
                                            y_train_null[local] = y_train_null[rng_null.permutation(local)]
                                    null_metrics, null_lam = _decode_fold(
                                        x,
                                        delta_e,
                                        condition_ids,
                                        train_mask,
                                        test_mask,
                                        seed=int(seed) + draw,
                                        y_train_override=y_train_null,
                                    )
                                else:
                                    x_train_null = x[train_mask].copy()
                                    if x_train_null.shape[0] > 1:
                                        x_train_null = x_train_null[rng_null.permutation(x_train_null.shape[0])]
                                    null_metrics, null_lam = _decode_fold(
                                        x,
                                        delta_e,
                                        condition_ids,
                                        train_mask,
                                        test_mask,
                                        seed=int(seed) + draw,
                                        x_train_override=x_train_null,
                                    )
                                for name, value in null_metrics.items():
                                    null_rows.append(
                                        {
                                            **base,
                                            "lambda_selected": null_lam,
                                            "metric_name": name,
                                            "metric_null": value,
                                            "null_type": null_type,
                                            "null_draw": int(draw),
                                        }
                                    )
    return metric_rows, null_rows, leakage_rows


def _summarize_metrics(metrics: list[dict[str, Any]], nulls: list[dict[str, Any]], *, seed: int, n_bootstrap: int) -> list[dict[str, Any]]:
    observed = [r for r in metrics if str(r.get("metric_name")) == "R2_mean" and str(r.get("null_type")) == "observed"]
    if not observed:
        return []
    null_lookup: dict[tuple[str, str, str, int, str], list[float]] = {}
    for r in nulls:
        if str(r.get("metric_name")) != "R2_mean":
            continue
        key = (str(r["session"]), str(r["projection_control"]), str(r["feature_space"]), int(r["k"]), str(r["null_type"]))
        null_lookup.setdefault(key, []).append(float(r["metric_null"]))
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for r in observed:
        key = (str(r["feature_space"]), int(r["k"]), str(r["projection_control"]), str(r["metric_name"]))
        groups.setdefault(key, []).append(r)
    rng = np.random.default_rng(int(seed))
    out: list[dict[str, Any]] = []
    for (feature_space, k, projection_control, metric_name), rows in sorted(groups.items()):
        by_session: dict[str, list[float]] = {}
        for r in rows:
            by_session.setdefault(str(r["session"]), []).append(float(r["metric_value"]))
        sessions = sorted(by_session)
        obs_vals = np.asarray([np.nanmean(by_session[s]) for s in sessions], dtype=np.float64)
        obs_mean, obs_lo, obs_hi = bootstrap_mean_ci(obs_vals, rng=rng, n_bootstrap=int(n_bootstrap))
        row: dict[str, Any] = {
            "feature_space": feature_space,
            "k": int(k),
            "projection_control": projection_control,
            "metric_name": metric_name,
            "n_sessions": int(np.sum(np.isfinite(obs_vals))),
            "observed_mean": obs_mean,
            "observed_boot_ci_low": obs_lo,
            "observed_boot_ci_high": obs_hi,
        }
        for null_type in ("eye_label_shuffle", "response_pair_shuffle"):
            effects = []
            null_medians = []
            for session in sessions:
                vals = null_lookup.get((session, projection_control, feature_space, int(k), null_type), [])
                vals = [v for v in vals if np.isfinite(v)]
                if not vals:
                    continue
                obs = float(np.nanmean(by_session[session]))
                med = float(np.nanmedian(vals))
                null_medians.append(med)
                effects.append(obs - med)
            eff = np.asarray(effects, dtype=np.float64)
            null_arr = np.asarray(null_medians, dtype=np.float64)
            eff_mean, eff_lo, eff_hi = bootstrap_mean_ci(eff, rng=rng, n_bootstrap=int(n_bootstrap))
            row[f"{null_type}_median_mean"] = float(np.nanmean(null_arr)) if null_arr.size else float("nan")
            row[f"effect_minus_{null_type}_mean"] = eff_mean
            row[f"effect_minus_{null_type}_boot_ci_low"] = eff_lo
            row[f"effect_minus_{null_type}_boot_ci_high"] = eff_hi
            finite_eff = eff[np.isfinite(eff)]
            row[f"n_{null_type}_effect_positive"] = int(np.sum(finite_eff > 0.0))
            row[f"sign_test_{null_type}_p_two_sided"] = sign_test_p_two_sided(int(np.sum(finite_eff > 0.0)), int(finite_eff.size))
        out.append(row)
    return out


def _feature_comparison(summary_rows: list[dict[str, Any]], primary_projection: str, primary_k: int) -> list[dict[str, Any]]:
    block = [
        r
        for r in summary_rows
        if str(r.get("projection_control")) == str(primary_projection)
        and str(r.get("metric_name")) == "R2_mean"
        and int(r.get("k", 0)) in {0, int(primary_k), 2}
    ]
    values = {(str(r["feature_space"]), int(r["k"])): float(r["observed_mean"]) for r in block}
    full = values.get(("full_population", 0), float("nan"))
    compact = values.get(("compact", int(primary_k)), float("nan"))
    orth_val = values.get(("orthogonal_complement", int(primary_k)), float("nan"))
    random_val = values.get(("random_subspace", int(primary_k)), float("nan"))
    rf_val = values.get(("rf_readout_permuted_compact", int(primary_k)), float("nan"))
    global_val = values.get(("global_top_pc_modes", 2), float("nan"))
    return [
        {
            "projection_control": primary_projection,
            "primary_k": int(primary_k),
            "full_population_R2_mean": full,
            "compact_R2_mean": compact,
            "compact_fraction_of_full": compact / full if np.isfinite(compact) and np.isfinite(full) and abs(full) > 1e-12 else float("nan"),
            "orthogonal_complement_R2_mean": orth_val,
            "random_subspace_R2_mean": random_val,
            "rf_readout_permuted_compact_R2_mean": rf_val,
            "global_top_pc_modes_R2_mean": global_val,
            "compact_minus_orthogonal": compact - orth_val if np.isfinite(compact) and np.isfinite(orth_val) else float("nan"),
            "compact_minus_random": compact - random_val if np.isfinite(compact) and np.isfinite(random_val) else float("nan"),
            "compact_minus_rf_readout": compact - rf_val if np.isfinite(compact) and np.isfinite(rf_val) else float("nan"),
            "compact_minus_global_pc": compact - global_val if np.isfinite(compact) and np.isfinite(global_val) else float("nan"),
        }
    ]


def _write_summary_figure(out: Path, comparison_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], primary_projection: str, primary_k: int) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    block = [
        r
        for r in summary_rows
        if str(r.get("projection_control")) == str(primary_projection)
        and str(r.get("metric_name")) == "R2_mean"
        and (int(r.get("k", 0)) == 0 or int(r.get("k", 0)) == int(primary_k) or str(r.get("feature_space")) == "global_top_pc_modes")
    ]
    order = [
        "full_population",
        "compact",
        "orthogonal_complement",
        "random_subspace",
        "unit_shuffled_compact",
        "rf_readout_permuted_compact",
        "global_top_pc_modes",
    ]
    vals = []
    labels = []
    for name in order:
        rows = [r for r in block if str(r.get("feature_space")) == name]
        if not rows:
            continue
        vals.append(float(rows[0]["observed_mean"]))
        labels.append(name.replace("_", "\n"))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(np.arange(len(vals)), vals, color="#4c78a8", alpha=0.86)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("R2 mean")
    ax.set_title(f"Relative displacement decoding ({primary_projection}, k={primary_k})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "relative_displacement_decoding_summary.png", dpi=220)
    fig.savefig(fig_dir / "relative_displacement_decoding_summary.pdf")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Matched-context relative displacement decoding")
    p.add_argument("--fig3-cache", type=Path, default=DEFAULT_FIG3_CACHE)
    p.add_argument("--fig2-cache", type=Path, default=DEFAULT_FIG2_CACHE)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    p.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--sessions", type=str, default="Allen_2022-02-16")
    p.add_argument("--window-idx", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=512)
    p.add_argument("--step-px", type=float, default=0.25)
    p.add_argument("--pixels-per-degree-fallback", type=float, default=37.5)
    p.add_argument("--fixation-radius-deg", type=float, default=1.0)
    p.add_argument("--sample-dfs-mode", choices=["all", "any", "none"], default="all")
    p.add_argument("--rescale-mode", choices=["none", "globalgain", "gain", "globalaffine", "affine"], default="affine")
    p.add_argument("--projection-controls", type=str, default="none,global_rate,target_pc1,global_rate+target_pc1")
    p.add_argument("--primary-projection-control", type=str, default="global_rate+target_pc1")
    p.add_argument("--target-variant", choices=["raw", "psd"], default="psd")
    p.add_argument("--k-list", type=str, default="1,2,5,10,20,30")
    p.add_argument("--primary-k", type=int, default=10)
    p.add_argument(
        "--context-mode",
        choices=["image_time_bin", "image_time_window", "image_only", "time_bin", "time_window"],
        default="image_time_bin",
    )
    p.add_argument("--context-bin-size", type=int, default=10)
    p.add_argument("--min-repeats-per-condition", type=int, default=3)
    p.add_argument("--max-pairs-per-condition", type=int, default=100)
    p.add_argument("--split-mode", choices=["condition_disjoint", "trial_disjoint"], default="condition_disjoint")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-nulls", type=int, default=20)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--min-units", type=int, default=50)
    p.add_argument("--enable-rf-readout-null", action="store_true", default=True)
    p.add_argument("--rf-null-min-bin-units", type=int, default=6)
    p.add_argument("--rf-null-bin-features", type=str, default="rf_xy,tangent_norm,mean_rate,ccnorm")
    p.add_argument("--rf-null-session-yaml-dir", type=Path, default=Path("experiments") / "dataset_configs" / "sessions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose-model-load", action="store_true")
    p.add_argument("--write-parent-tables", action="store_true")
    p.add_argument("--init-only", action="store_true")
    return p


def run_analysis(args: argparse.Namespace) -> None:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    projection_controls = parse_str_list(args.projection_controls)
    k_list = parse_int_list(args.k_list)
    fig3_rows = _load_pickle(Path(args.fig3_cache))
    fig2_rows = _load_pickle(Path(args.fig2_cache))
    fig2 = _fig2_by_session(fig2_rows)
    fig3_by_session = {str(row["session"]): row for row in fig3_rows}
    requested_sessions = parse_str_list(args.sessions)
    if len(requested_sessions) == 1 and requested_sessions[0].lower() == "all":
        requested_sessions = []
    if not requested_sessions:
        requested_sessions = [str(row["session"]) for row in fig3_rows if str(row.get("subject", "")) in {"Allen", "Logan"}]
    config = DecodeConfig(
        output_root=str(out),
        sessions=requested_sessions,
        projection_controls=projection_controls,
        target_variant=str(args.target_variant),
        k_list=k_list,
        primary_k=int(args.primary_k),
        context_mode=str(args.context_mode),
        context_bin_size=int(args.context_bin_size),
        min_repeats_per_condition=int(args.min_repeats_per_condition),
        max_pairs_per_condition=int(args.max_pairs_per_condition),
        split_mode=str(args.split_mode),
        n_folds=int(args.n_folds),
        n_nulls=int(args.n_nulls),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    manifest_base = {
        "analysis": "matched_context_relative_displacement_decoding",
        "status": "initialized_not_run" if bool(args.init_only) else "running",
        "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "fig2_cache": str(Path(args.fig2_cache).resolve()),
        "fig3_cache": str(Path(args.fig3_cache).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_config": str(Path(args.model_config).resolve()),
        "dataset_config": str(Path(args.dataset_config).resolve()),
        "fd_closure_default_output": str(DEFAULT_FD_OUT),
        "claim_guardrail": "Primary analysis decodes relative eye-position differences within the requested matched context; same-image language is warranted only for image-aware context modes.",
    }
    write_json(out / "relative_displacement_decoding_manifest.json", manifest_base)
    if bool(args.init_only):
        (out / "README.md").write_text("# Same-Condition Relative Displacement Decoding\n\nStatus: initialized, not run.\n", encoding="utf-8")
        return

    model, model_info = _load_twin_model(args)
    session_rows: list[dict[str, Any]] = []
    pair_inventory_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    rf_rows: list[dict[str, Any]] = []

    for session in requested_sessions:
        print(f"[relative-decoding] session {session}: starting", flush=True)
        if session not in fig3_by_session or session not in fig2:
            session_rows.append({"session": session, "status": "missing_fig2_or_fig3"})
            continue
        if session not in getattr(model, "names", []):
            session_rows.append({"session": session, "status": "missing_model_session"})
            continue
        dataset_idx = int(model.names.index(session))
        sr = fig3_by_session[session]
        common_units, target_raw, target_psd, target_meta = _target_for_session(fig2[session], sr, args)
        if common_units.size < int(args.min_units):
            session_rows.append({"session": session, "status": "too_few_common_units", "n_common_units": int(common_units.size)})
            continue
        print(f"[relative-decoding] session {session}: collecting samples ({common_units.size} units)", flush=True)
        dset, stim_lags, samples = _collect_samples(model=model, dataset_idx=dataset_idx, common_units=common_units, args=args)
        labels, condition_rows, image_meta = pair_condition_labels(
            dset=dset,
            samples=samples,
            mode=str(args.context_mode),
            bin_size=int(args.context_bin_size),
        )
        eye_px = samples.eyepos_deg * float(samples.pixels_per_degree)
        print(f"[relative-decoding] session {session}: building matched-context pairs from {samples.source_indices.size} samples", flush=True)
        pairs, inventory = build_pair_dataset(
            samples=samples,
            eye_px=eye_px,
            labels=labels,
            condition_rows=condition_rows,
            context_mode=str(args.context_mode),
            context_bin_size=int(args.context_bin_size),
            min_repeats_per_condition=int(args.min_repeats_per_condition),
            max_pairs_per_condition=int(args.max_pairs_per_condition),
            seed=int(args.seed) + dataset_idx * 101,
        )
        for row in inventory:
            row.update({"session": session, "subject": sr.get("subject", "")})
        pair_inventory_rows.extend(inventory)
        if pairs["delta_y"].shape[0] < max(20, int(args.n_folds) * 3):
            session_rows.append(
                {
                    "session": session,
                    "subject": sr.get("subject", ""),
                    "status": "too_few_pairs",
                    "n_common_units": int(common_units.size),
                    "n_pairs": int(pairs["delta_y"].shape[0]),
                    "n_pair_conditions": int(np.unique(pairs["condition_id"]).size),
                }
            )
            continue
        print(f"[relative-decoding] session {session}: {pairs['delta_y'].shape[0]} pairs across {np.unique(pairs['condition_id']).size} conditions", flush=True)
        print(f"[relative-decoding] session {session}: fitting response rescale gains", flush=True)
        gains, rescale_status = _fit_rescale_gains(
            model=model,
            dset=dset,
            stim_lags=stim_lags,
            samples=samples,
            common_units=common_units,
            dataset_idx=dataset_idx,
            args=args,
        )
        print(f"[relative-decoding] session {session}: computing finite-difference Jacobians", flush=True)
        j = _compute_jacobians(
            model=model,
            dset=dset,
            stim_lags=stim_lags,
            samples=samples,
            common_units=common_units,
            gains=gains,
            dataset_idx=dataset_idx,
            args=args,
        )
        print(f"[relative-decoding] session {session}: building RF/readout bins and decoding folds", flush=True)
        rf_meta = _rf_null_metadata_for_session(
            session=session,
            subject=str(sr.get("subject", "")),
            common_units=common_units,
            sr=sr,
            samples=samples,
            j=j,
            gains=gains,
            args=args,
        )
        rf_rows.extend(rf_meta.unit_rows)
        target = target_psd if str(args.target_variant) == "psd" else target_raw
        m_rows, n_rows, l_rows = _session_decode_rows(
            session=session,
            subject=str(sr.get("subject", "")),
            pairs=pairs,
            j=j,
            labels=labels,
            samples=samples,
            target_cov=target,
            projection_controls=projection_controls,
            k_list=k_list,
            primary_k=int(args.primary_k),
            rf_bins=rf_meta.bins,
            split_mode=str(args.split_mode),
            n_folds=int(args.n_folds),
            n_nulls=int(args.n_nulls),
            seed=int(args.seed) + dataset_idx * 1009,
        )
        metric_rows.extend(m_rows)
        null_rows.extend(n_rows)
        leakage_rows.extend(l_rows)
        print(f"[relative-decoding] session {session}: finished ({len(m_rows)} metric rows, {len(n_rows)} null rows)", flush=True)
        session_rows.append(
            {
                "session": session,
                "subject": sr.get("subject", ""),
                "status": "ok",
                "dataset_idx": dataset_idx,
                "n_common_units": int(common_units.size),
                "n_samples_used": int(samples.source_indices.size),
                "n_candidate_samples": int(samples.n_candidate_samples),
                "n_pair_conditions": int(np.unique(pairs["condition_id"]).size),
                "n_pairs": int(pairs["delta_y"].shape[0]),
                "pixels_per_degree": float(samples.pixels_per_degree),
                "step_px": float(args.step_px),
                "rescale_status": rescale_status,
                "rf_null_status": rf_meta.status,
                "rf_null_n_bins": int(rf_meta.n_bins),
                "rf_null_largest_bin_fraction": float(rf_meta.largest_bin_fraction),
                "rf_null_bin_features": rf_meta.bin_features,
                "jacobian_abs_median": float(np.median(np.abs(j))),
                **image_meta,
                **target_meta,
            }
        )

    summary_rows = _summarize_metrics(metric_rows, null_rows, seed=int(args.seed), n_bootstrap=int(args.n_bootstrap))
    comparison_rows = _feature_comparison(summary_rows, str(args.primary_projection_control), int(args.primary_k))
    projection_rows = [
        r for r in summary_rows if str(r.get("feature_space")) == "compact" and int(r.get("k", -1)) == int(args.primary_k)
    ]
    k_rows = [
        r
        for r in summary_rows
        if str(r.get("feature_space")) == "compact"
        and str(r.get("projection_control")) == str(args.primary_projection_control)
    ]
    write_csv(out / "session_summary.csv", session_rows)
    write_csv(out / "pair_inventory.csv", pair_inventory_rows)
    write_csv(out / "decoder_metrics.csv", metric_rows)
    write_csv(out / "decoder_nulls.csv", null_rows)
    write_csv(out / "decoder_bootstrap_summary.csv", summary_rows)
    write_csv(out / "feature_space_comparison.csv", comparison_rows)
    write_csv(out / "projection_control_comparison.csv", projection_rows)
    write_csv(out / "k_sweep.csv", k_rows)
    write_csv(out / "split_leakage_audit.csv", leakage_rows)
    write_csv(out / "rf_readout_unit_bins.csv", rf_rows)
    write_csv(out / "decoder_reliability_ceiling.csv", [{"status": "not_run", "reason": "split-half decoding ceiling is not yet implemented for pairwise relative decoder"}])
    write_csv(out / "spectral_bridge.csv", [{"status": "not_run", "reason": "run after primary decoder passes support/null gates"}])
    write_csv(out / "information_gain_bridge.csv", [{"status": "not_run", "reason": "run after primary decoder passes support/null gates"}])
    write_csv(out / "compact_specific_bridge.csv", [{"status": "not_run", "reason": "run after primary decoder passes support/null gates"}])
    _write_summary_figure(out, comparison_rows, summary_rows, str(args.primary_projection_control), int(args.primary_k))

    if bool(args.write_parent_tables) or out.name == DEFAULT_OUTPUT_ROOT.name:
        tables_dir = out.parent / "tables"
        write_csv(tables_dir / "displacement_decoding_metrics.csv", metric_rows)
        write_csv(tables_dir / "displacement_decoding_bootstrap_summary.csv", summary_rows)
        write_csv(tables_dir / "displacement_decoding_nulls.csv", null_rows)
        write_csv(tables_dir / "displacement_decoding_pair_inventory.csv", pair_inventory_rows)
        write_csv(tables_dir / "displacement_decoding_reliability_ceiling.csv", [{"status": "not_run", "reason": "split-half decoding ceiling is not yet implemented for pairwise relative decoder"}])
        write_csv(tables_dir / "panelF_displacement_decoding_metrics.csv", comparison_rows)

    comp = comparison_rows[0] if comparison_rows else {}
    compact_minus_orth = float(comp.get("compact_minus_orthogonal", float("nan"))) if comp else float("nan")
    compact_minus_random = float(comp.get("compact_minus_random", float("nan"))) if comp else float("nan")
    compact_minus_rf = float(comp.get("compact_minus_rf_readout", float("nan"))) if comp else float("nan")
    full_r2 = float(comp.get("full_population_R2_mean", float("nan"))) if comp else float("nan")
    compact_r2 = float(comp.get("compact_R2_mean", float("nan"))) if comp else float("nan")
    compact_summary = [
        r
        for r in summary_rows
        if str(r.get("feature_space")) == "compact"
        and str(r.get("projection_control")) == str(args.primary_projection_control)
        and int(r.get("k", -1)) == int(args.primary_k)
        and str(r.get("metric_name")) == "R2_mean"
    ]
    eye_ci_low = float(compact_summary[0].get("effect_minus_eye_label_shuffle_boot_ci_low", float("nan"))) if compact_summary else float("nan")
    leakage_failures = int(sum(1 for r in leakage_rows if r.get("status") == "fail"))
    status = "ok"
    decision = "diagnostic"
    if (
        np.isfinite(full_r2)
        and np.isfinite(compact_r2)
        and compact_r2 > 0.0
        and np.isfinite(compact_minus_orth)
        and compact_minus_orth > 0.0
        and np.isfinite(compact_minus_random)
        and compact_minus_random > 0.0
        and (not np.isfinite(compact_minus_rf) or compact_minus_rf > 0.0)
        and np.isfinite(eye_ci_low)
        and eye_ci_low > 0.0
        and leakage_failures == 0
    ):
        decision = "candidate_positive"
    audit = {
        "status": status,
        "decision": decision,
        "n_sessions_requested": int(len(requested_sessions)),
        "n_sessions_ok": int(sum(1 for r in session_rows if r.get("status") == "ok")),
        "n_metric_rows": int(len(metric_rows)),
        "n_null_rows": int(len(null_rows)),
        "n_leakage_failures": leakage_failures,
        "primary_projection_control": str(args.primary_projection_control),
        "primary_k": int(args.primary_k),
        "primary_feature_comparison": comp,
        "primary_compact_effect_minus_eye_label_shuffle_ci_low": eye_ci_low,
        "target_pc_projection_scope": "train_estimated_tangent_covariance_with_session_target_fallback",
        "claim_guardrail": "Do not describe this as absolute or image-independent eye-position decoding. Same-image wording is warranted only for image-aware context modes.",
        "model_info": {k: str(v) for k, v in dict(model_info).items()},
    }
    write_json(out / "audit.json", audit)
    write_json(
        out / "relative_displacement_decoding_manifest.json",
        {
            **manifest_base,
            "status": "ok",
            "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
            "model_info": {k: str(v) for k, v in dict(model_info).items()},
        },
    )
    (out / "README.md").write_text(
        "# Matched-Context Relative Displacement Decoding\n\n"
        "Primary tables are `decoder_bootstrap_summary.csv`, `feature_space_comparison.csv`, "
        "and `decoder_nulls.csv`. Promote only if compact k=10 under `global_rate+target_pc1` "
        "beats eye-label shuffle plus compact-specific controls without split leakage.\n",
        encoding="utf-8",
    )


def main() -> None:
    run_analysis(build_parser().parse_args())


if __name__ == "__main__":
    main()
