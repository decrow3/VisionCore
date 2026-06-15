#!/usr/bin/env python3
"""Correct-chart versus wrong-chart pairwise alignment.

This runner implements the A2 analysis from
``declan/content_routed_retinal_registration_analysis_plan.md``.  It pairs
recorded fixRSVP repeats at the same image/time condition, predicts each
recorded response difference from the fitted-twin retinal-translation chart,
and asks whether the correct chart aligns better than matched wrong charts and
compact/gain/null controls.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from declan.compact_retinal_translation_geometry.run_relative_displacement_decoding import (
    _decode_splits,
    _trial_pair_keys,
    _trial_set,
    context_labels,
    fixed_within_bin_permutation,
    orth,
    parse_int_list,
    parse_str_list,
    sign_test_p_two_sided,
)
from declan.direct_recorded_derivative_twin_alignment.run_direct_recorded_derivative_alignment import (
    bootstrap_mean_ci,
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
    _behavior_batch,
    _collect_samples,
    _compute_jacobians,
    _fit_rescale_gains,
    _load_twin_model,
    _predict,
    _rf_null_metadata_for_session,
    _stim_batch,
    _target_for_session,
)


DEFAULT_OUTPUT_ROOT = (
    Path("outputs") / "compact_retinal_translation_geometry" / "correct_chart_swap_alignment"
)


@dataclass
class ChartSwapConfig:
    output_root: str
    sessions: list[str]
    projection_controls: list[str]
    target_variant: str
    k_list: list[int]
    primary_k: int
    context_mode: str
    context_bin_size: int
    split_mode: str
    n_folds: int
    min_repeats_per_condition: int
    max_pairs_per_condition: int
    min_train_samples_per_chart: int
    wrong_chart_pool: str
    wrong_chart_match_features: list[str]
    score_mode: str
    unit_score_subsets: list[str]
    drift_speed_threshold_px: float
    drift_pair_delta_threshold_px: float
    run_pseudo_spike_control: bool
    pseudo_control_modes: list[str]
    pseudo_poisson_scales: list[float]
    pseudo_injection_noise_sds: list[float]
    local_image_radius_px: int
    n_bootstrap: int
    seed: int


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
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
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _condition_keys(
    *,
    image_ids: np.ndarray,
    time_indices: np.ndarray,
    mode: str,
    bin_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    image_ids = np.asarray(image_ids, dtype=np.int64)
    time_indices = np.asarray(time_indices, dtype=np.int64)
    if str(mode) == "image_only":
        time_ctx = np.zeros_like(time_indices)
    elif str(mode) == "time_window":
        time_ctx = time_indices // max(int(bin_size), 1)
    elif str(mode) == "time_bin":
        time_ctx = time_indices
    else:
        raise ValueError(f"Unsupported context mode: {mode}")

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
        rows.append({"condition_id": int(cid), "image_id": int(img), "time_context": int(tt)})
    return labels, image_ids, time_ctx, rows


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
            continue
    valid = out >= 0
    return out, {
        "image_id_status": "ok" if np.any(valid) else "no_valid_image_ids",
        "n_image_id_valid_samples": int(np.sum(valid)),
        "n_image_id_failures": int(failures),
        "n_unique_image_ids": int(np.unique(out[valid]).size) if np.any(valid) else 0,
    }


def _sample_drift_mask(
    *,
    dset: Any,
    samples: Any,
    pixels_per_degree: float,
    speed_threshold_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    trial_inds = np.asarray(dset.covariates["trial_inds"]).ravel()
    eyepos = np.asarray(dset["eyepos"], dtype=np.float64) * float(pixels_per_degree)
    speed = np.full(samples.source_indices.size, np.nan, dtype=np.float64)
    for i, src in enumerate(np.asarray(samples.source_indices, dtype=np.int64)):
        vals: list[float] = []
        if src - 1 >= 0 and int(trial_inds[src - 1]) == int(trial_inds[src]):
            vals.append(float(np.linalg.norm(eyepos[src] - eyepos[src - 1])))
        if src + 1 < eyepos.shape[0] and int(trial_inds[src + 1]) == int(trial_inds[src]):
            vals.append(float(np.linalg.norm(eyepos[src + 1] - eyepos[src])))
        if vals:
            speed[i] = float(max(vals))
    mask = np.isfinite(speed) & (speed <= float(speed_threshold_px))
    return mask, speed


def _image_structure_metrics(frame: np.ndarray) -> dict[str, float]:
    img = np.asarray(frame, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = np.squeeze(img)
    if img.ndim != 2:
        return {
            "image_mean_luminance": float("nan"),
            "image_rms_contrast": float("nan"),
            "image_gradient_rms": float("nan"),
            "image_laplacian_rms": float("nan"),
            "image_structure_score": float("nan"),
        }
    x = img - float(np.mean(img))
    gx = np.diff(x, axis=1, prepend=x[:, :1])
    gy = np.diff(x, axis=0, prepend=x[:1, :])
    lap = (
        -4.0 * x
        + np.roll(x, 1, axis=0)
        + np.roll(x, -1, axis=0)
        + np.roll(x, 1, axis=1)
        + np.roll(x, -1, axis=1)
    )
    gradient_rms = float(np.sqrt(np.mean(gx * gx + gy * gy)))
    laplacian_rms = float(np.sqrt(np.mean(lap * lap)))
    return {
        "image_mean_luminance": float(np.mean(img)),
        "image_rms_contrast": float(np.std(x)),
        "image_gradient_rms": gradient_rms,
        "image_laplacian_rms": laplacian_rms,
        "image_structure_score": float(np.log1p(gradient_rms) + np.log1p(laplacian_rms)),
    }


def _load_fixrsvp_image_metrics() -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    meta: dict[str, Any] = {"image_structure_status": "not_loaded"}
    try:
        from DataYatesV1.exp.support import get_rsvp_fix_stim
    except Exception as exc:
        meta["image_structure_status"] = f"unavailable_import_{type(exc).__name__}"
        return {}, meta
    try:
        images = get_rsvp_fix_stim()
    except Exception as exc:
        meta["image_structure_status"] = f"unavailable_load_{type(exc).__name__}"
        return {}, meta

    pat = re.compile(r"^im(\d+)$")
    out: dict[int, dict[str, float]] = {}
    for key, value in images.items():
        match = pat.match(str(key))
        if match is None:
            continue
        image_id = int(match.group(1)) - 1
        metrics = _image_structure_metrics(np.asarray(value))
        metrics["image_id"] = int(image_id)
        out[image_id] = metrics
    meta.update(
        {
            "image_structure_status": "ok" if out else "no_fixrsvp_images",
            "n_image_structure_images": int(len(out)),
        }
    )
    return out, meta


def _local_image_structure_scores(
    *,
    image_ids: np.ndarray,
    eye_px: np.ndarray,
    image_metrics: dict[int, dict[str, float]],
    radius_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-sample global and local structure proxies.

    Local scores use a small patch centered at image center plus current eye
    position.  If raw images are unavailable, both arrays are NaN.
    """
    n = int(np.asarray(image_ids).size)
    global_score = np.full(n, np.nan, dtype=np.float64)
    local_score = np.full(n, np.nan, dtype=np.float64)
    try:
        from DataYatesV1.exp.support import get_rsvp_fix_stim

        images = get_rsvp_fix_stim()
    except Exception:
        images = {}
    pat = re.compile(r"^im(\d+)$")
    image_arrays: dict[int, np.ndarray] = {}
    for key, value in images.items():
        match = pat.match(str(key))
        if match is None:
            continue
        img = np.asarray(value, dtype=np.float64)
        if img.ndim == 3:
            img = img.mean(axis=2)
        img = np.squeeze(img)
        if img.ndim == 2:
            image_arrays[int(match.group(1)) - 1] = img

    for i, img_id_raw in enumerate(np.asarray(image_ids, dtype=np.int64)):
        img_id = int(img_id_raw)
        if img_id in image_metrics:
            global_score[i] = float(image_metrics[img_id].get("image_structure_score", float("nan")))
        img = image_arrays.get(img_id)
        if img is None:
            continue
        h, w = img.shape
        ex, ey = np.asarray(eye_px[i], dtype=np.float64)
        cx = int(np.clip(round((w - 1) / 2.0 + ex), 0, w - 1))
        cy = int(np.clip(round((h - 1) / 2.0 + ey), 0, h - 1))
        r = max(int(radius_px), 1)
        patch = img[max(0, cy - r) : min(h, cy + r + 1), max(0, cx - r) : min(w, cx + r + 1)]
        local_score[i] = float(_image_structure_metrics(patch).get("image_structure_score", float("nan")))
    return global_score, local_score


def build_chart_pair_dataset(
    *,
    samples: Any,
    eye_px: np.ndarray,
    labels: np.ndarray,
    image_ids: np.ndarray,
    time_contexts: np.ndarray,
    sample_drift_mask: np.ndarray,
    sample_image_structure_score: np.ndarray | None = None,
    sample_local_image_structure_score: np.ndarray | None = None,
    drift_pair_delta_threshold_px: float,
    min_repeats_per_condition: int,
    max_pairs_per_condition: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    robs = np.asarray(samples.robs, dtype=np.float64)
    eye = np.asarray(eye_px, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    image_ids = np.asarray(image_ids, dtype=np.int64)
    time_contexts = np.asarray(time_contexts, dtype=np.int64)
    sample_drift_mask = np.asarray(sample_drift_mask, dtype=bool)
    if sample_image_structure_score is None:
        sample_image_structure_score = np.full(labels.shape, np.nan, dtype=np.float64)
    if sample_local_image_structure_score is None:
        sample_local_image_structure_score = np.full(labels.shape, np.nan, dtype=np.float64)
    sample_image_structure_score = np.asarray(sample_image_structure_score, dtype=np.float64)
    sample_local_image_structure_score = np.asarray(sample_local_image_structure_score, dtype=np.float64)
    valid = np.isfinite(robs).all(axis=1) & np.isfinite(eye).all(axis=1) & (labels >= 0) & (image_ids >= 0)

    dy_rows: list[np.ndarray] = []
    de_rows: list[np.ndarray] = []
    condition_rows: list[int] = []
    image_rows: list[int] = []
    time_rows: list[int] = []
    trial_a_rows: list[int] = []
    trial_b_rows: list[int] = []
    sample_a_rows: list[int] = []
    sample_b_rows: list[int] = []
    drift_rows: list[bool] = []
    image_structure_rows: list[float] = []
    local_image_structure_rows: list[float] = []
    inventory: list[dict[str, Any]] = []

    for condition_id in np.unique(labels[labels >= 0]):
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
            delta_e = eye[a] - eye[b]
            pair_drift = bool(
                sample_drift_mask[a]
                and sample_drift_mask[b]
                and np.linalg.norm(delta_e) <= float(drift_pair_delta_threshold_px)
            )
            dy_rows.append(robs[a] - robs[b])
            de_rows.append(delta_e)
            condition_rows.append(int(condition_id))
            image_rows.append(int(image_ids[a]))
            time_rows.append(int(time_contexts[a]))
            trial_a_rows.append(int(samples.trial_ids[a]))
            trial_b_rows.append(int(samples.trial_ids[b]))
            sample_a_rows.append(int(a))
            sample_b_rows.append(int(b))
            drift_rows.append(pair_drift)
            image_structure_rows.append(float(np.nanmean([sample_image_structure_score[a], sample_image_structure_score[b]])))
            local_image_structure_rows.append(float(np.nanmean([sample_local_image_structure_score[a], sample_local_image_structure_score[b]])))
        inventory.append(
            {
                "condition_id": int(condition_id),
                "image_id": int(image_ids[idx[0]]) if idx.size else -1,
                "time_context": int(time_contexts[idx[0]]) if idx.size else -1,
                "n_repeats": int(idx.size),
                "n_trials": int(np.unique(samples.trial_ids[idx]).size) if idx.size else 0,
                "condition_mean_rate": float(np.nanmean(robs[idx])) if idx.size else float("nan"),
                "condition_response_norm": float(np.linalg.norm(np.nanmean(robs[idx], axis=0))) if idx.size else float("nan"),
                "condition_image_structure_score": float(np.nanmean(sample_image_structure_score[idx])) if idx.size else float("nan"),
                "condition_local_image_structure_score": float(np.nanmean(sample_local_image_structure_score[idx])) if idx.size else float("nan"),
                "n_all_cross_trial_pairs": int(n_all_pairs),
                "n_pairs_used": int(len(pairs)),
                "n_drift_pairs_used": int(sum(drift_rows[-len(pairs) :])) if pairs else 0,
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
            "image_id": np.zeros(0, dtype=np.int64),
            "time_context": np.zeros(0, dtype=np.int64),
            "trial_a": np.zeros(0, dtype=np.int64),
            "trial_b": np.zeros(0, dtype=np.int64),
            "sample_a": np.zeros(0, dtype=np.int64),
            "sample_b": np.zeros(0, dtype=np.int64),
            "drift_mask": np.zeros(0, dtype=bool),
            "image_structure_score": np.zeros(0, dtype=np.float64),
            "local_image_structure_score": np.zeros(0, dtype=np.float64),
        }
        return empty, inventory
    return (
        {
            "delta_y": np.stack(dy_rows, axis=0).astype(np.float64),
            "delta_e": np.stack(de_rows, axis=0).astype(np.float64),
            "condition_id": np.asarray(condition_rows, dtype=np.int64),
            "image_id": np.asarray(image_rows, dtype=np.int64),
            "time_context": np.asarray(time_rows, dtype=np.int64),
            "trial_a": np.asarray(trial_a_rows, dtype=np.int64),
            "trial_b": np.asarray(trial_b_rows, dtype=np.int64),
            "sample_a": np.asarray(sample_a_rows, dtype=np.int64),
            "sample_b": np.asarray(sample_b_rows, dtype=np.int64),
            "drift_mask": np.asarray(drift_rows, dtype=bool),
            "image_structure_score": np.asarray(image_structure_rows, dtype=np.float64),
            "local_image_structure_score": np.asarray(local_image_structure_rows, dtype=np.float64),
        },
        inventory,
    )


def _split_ids_for_folds(ids: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return []
    rng = np.random.default_rng(int(seed))
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    return [fold.astype(np.int64) for fold in np.array_split(shuffled, min(int(n_folds), ids.size)) if fold.size]


def _chart_swap_splits(
    pairs: dict[str, np.ndarray],
    n_folds: int,
    seed: int,
    split_mode: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if str(split_mode) == "condition_disjoint":
        raise ValueError(
            "condition_disjoint is not a valid chart-swap scoring mode because true charts "
            "are condition-local; use trial_disjoint or drift_trial_disjoint"
        )
    if str(split_mode) == "trial_disjoint":
        return _decode_splits(pairs, int(n_folds), int(seed), str(split_mode))
    if str(split_mode) == "trial_disjoint_drift_test":
        trial_a = np.asarray(pairs["trial_a"], dtype=np.int64)
        trial_b = np.asarray(pairs["trial_b"], dtype=np.int64)
        drift_mask = np.asarray(pairs["drift_mask"], dtype=bool)
        trials = np.unique(np.concatenate([trial_a, trial_b])).astype(np.int64)
        out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for held_trials in _split_ids_for_folds(trials, int(n_folds), int(seed)):
            a_test = np.isin(trial_a, held_trials)
            b_test = np.isin(trial_b, held_trials)
            test_mask = a_test & b_test & drift_mask
            train_mask = (~a_test) & (~b_test)
            if np.sum(train_mask) >= 5 and np.sum(test_mask) >= 1:
                out.append((held_trials.astype(np.int64), train_mask, test_mask))
        return out
    if str(split_mode) != "drift_trial_disjoint":
        raise ValueError(f"Unsupported split mode: {split_mode}")

    drift_idx = np.flatnonzero(np.asarray(pairs["drift_mask"], dtype=bool))
    if drift_idx.size == 0:
        return []
    rng = np.random.default_rng(int(seed))
    shuffled = drift_idx.copy()
    rng.shuffle(shuffled)
    folds = [fold.astype(np.int64) for fold in np.array_split(shuffled, min(int(n_folds), drift_idx.size)) if fold.size]
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    trial_a = np.asarray(pairs["trial_a"], dtype=np.int64)
    trial_b = np.asarray(pairs["trial_b"], dtype=np.int64)
    for fold_pair_indices in folds:
        held_trials = np.unique(np.concatenate([trial_a[fold_pair_indices], trial_b[fold_pair_indices]])).astype(np.int64)
        test_mask = np.zeros(trial_a.size, dtype=bool)
        test_mask[fold_pair_indices] = True
        train_mask = (~np.isin(trial_a, held_trials)) & (~np.isin(trial_b, held_trials))
        if np.sum(train_mask) >= 5 and np.sum(test_mask) >= 1:
            out.append((held_trials, train_mask, test_mask))
    return out


def _basis_from_j(j: np.ndarray, train_sample_mask: np.ndarray, projection: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    jj = np.asarray(j, dtype=np.float64)[np.asarray(train_sample_mask, dtype=bool)]
    if jj.shape[0] == 0:
        return np.zeros((j.shape[1], 0), dtype=np.float64), 0
    mat = np.concatenate([jj[:, :, 0].T, jj[:, :, 1].T], axis=1)
    mat = np.asarray(projection, dtype=np.float64) @ mat
    if mat.shape[1] == 0 or not np.isfinite(mat).all():
        return np.zeros((j.shape[1], 0), dtype=np.float64), 0
    vals, vecs = np.linalg.eigh(0.5 * (mat @ mat.T + (mat @ mat.T).T))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    rank = int(np.sum(vals > max(float(vals[0]) if vals.size else 0.0, 1.0) * 1e-10))
    return orth(vecs[:, : min(int(k), rank, vecs.shape[1])]), rank


def _condition_charts(
    *,
    j: np.ndarray,
    labels: np.ndarray,
    train_sample_mask: np.ndarray,
    min_train_samples_per_chart: int,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    charts: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    labels = np.asarray(labels, dtype=np.int64)
    train_sample_mask = np.asarray(train_sample_mask, dtype=bool)
    for condition_id in np.unique(labels[(labels >= 0) & train_sample_mask]):
        idx = np.flatnonzero((labels == int(condition_id)) & train_sample_mask)
        counts[int(condition_id)] = int(idx.size)
        if idx.size >= int(min_train_samples_per_chart):
            charts[int(condition_id)] = np.nanmean(np.asarray(j, dtype=np.float64)[idx], axis=0)
    return charts, counts


def _unit_score_subset_masks(
    j: np.ndarray,
    specs: list[str],
    unit_rows: list[dict[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    n_units = int(np.asarray(j).shape[1])
    out: dict[str, np.ndarray] = {}
    tangent_scores = np.sqrt(np.nanmean(np.sum(np.asarray(j, dtype=np.float64) ** 2, axis=2), axis=0))
    property_scores: dict[str, np.ndarray] = {"fem_tangent": tangent_scores, "tangent": tangent_scores, "fem": tangent_scores}
    if unit_rows:
        sorted_rows = sorted(unit_rows, key=lambda r: int(r.get("unit_position", 0)))
        if len(sorted_rows) == n_units:
            for field, aliases in {
                "gain": ("gain",),
                "mean_rate": ("rate", "mean_rate"),
                "ccnorm": ("ccnorm",),
                "tangent_norm": ("rf_tangent",),
            }.items():
                vals = np.asarray([float(r.get(field, float("nan"))) for r in sorted_rows], dtype=np.float64)
                for alias in aliases:
                    property_scores[alias] = vals
    for spec in specs:
        name = str(spec).strip()
        if not name:
            continue
        if name in {"all", "all_units"}:
            out["all_units"] = np.ones(n_units, dtype=bool)
            continue
        match = re.match(r"^(fem|tangent|fem_tangent|rf_tangent|gain|rate|mean_rate|ccnorm)_(top|bottom)(10|25|50)$", name)
        if match is None:
            continue
        source, tail, pct_raw = match.groups()
        scores = property_scores.get(source)
        if scores is None:
            continue
        scores = np.asarray(scores, dtype=np.float64)
        finite = np.isfinite(scores)
        if not np.any(finite):
            continue
        pct = int(pct_raw)
        if tail == "top":
            threshold = float(np.nanpercentile(scores[finite], 100 - pct))
            mask = finite & (scores >= threshold)
        else:
            threshold = float(np.nanpercentile(scores[finite], pct))
            mask = finite & (scores <= threshold)
        label_source = {"fem": "fem_tangent", "tangent": "fem_tangent", "rate": "mean_rate"}.get(source, source)
        if int(np.sum(mask)) >= 3:
            out[f"{label_source}_{tail}{pct}"] = mask
    if "all_units" not in out:
        out = {"all_units": np.ones(n_units, dtype=bool), **out}
    return out


def _diag_whitener(delta_y: np.ndarray, train_mask: np.ndarray, projection: np.ndarray) -> np.ndarray:
    y = np.asarray(delta_y, dtype=np.float64) @ np.asarray(projection, dtype=np.float64).T
    train = y[np.asarray(train_mask, dtype=bool)]
    if train.shape[0] < 2:
        var = np.nanvar(y, axis=0)
    else:
        var = np.nanvar(train, axis=0)
    med = float(np.nanmedian(var[np.isfinite(var) & (var > 0.0)])) if np.any(np.isfinite(var) & (var > 0.0)) else 1.0
    return np.where(np.isfinite(var) & (var > med * 1e-6), var, med).astype(np.float64)


def _alignment_score(delta_y: np.ndarray, q: np.ndarray, var: np.ndarray, mode: str) -> float:
    y = np.asarray(delta_y, dtype=np.float64).ravel()
    qq = np.asarray(q, dtype=np.float64).ravel()
    keep = np.isfinite(y) & np.isfinite(qq)
    if int(np.sum(keep)) < 3:
        return float("nan")
    y = y[keep]
    qq = qq[keep]
    if str(mode) == "cosine":
        den = float(np.linalg.norm(y) * np.linalg.norm(qq))
        return float(np.dot(y, qq) / den) if den > 1e-12 else float("nan")
    if str(mode) == "unit_dot":
        den = float(np.linalg.norm(qq))
        return float(np.dot(y, qq) / den) if den > 1e-12 else float("nan")
    vv = np.asarray(var, dtype=np.float64).ravel()[keep]
    vv = np.where(vv > 1e-12, vv, 1.0)
    den = float(np.sqrt(np.sum(qq * qq / vv)))
    return float(np.sum(y * qq / vv) / den) if den > 1e-12 else float("nan")


def _project_chart_vector(
    q: np.ndarray,
    *,
    projection: np.ndarray,
    chart_space: str,
    basis: np.ndarray,
    random_basis: np.ndarray | None,
    unit_perm: np.ndarray | None,
    rf_perm: np.ndarray | None,
    gain_axis: np.ndarray,
) -> dict[str, np.ndarray]:
    p = np.asarray(projection, dtype=np.float64)
    q_proj = p @ np.asarray(q, dtype=np.float64)
    out = {"full": q_proj}
    if str(chart_space) == "full":
        return out
    u = orth(basis)
    if u.shape[1] > 0:
        out["compact"] = u @ (u.T @ q_proj)
        if unit_perm is not None:
            us = u[np.asarray(unit_perm, dtype=np.int64), :]
            out["unit_shuffle"] = us @ (us.T @ q_proj)
        if rf_perm is not None:
            urf = u[np.asarray(rf_perm, dtype=np.int64), :]
            out["rf_readout"] = urf @ (urf.T @ q_proj)
    if random_basis is not None and random_basis.shape[1] > 0:
        rb = orth(p @ random_basis)
        out["random"] = rb @ (rb.T @ q_proj)
    gain = np.asarray(gain_axis, dtype=np.float64)
    gain = gain / max(float(np.linalg.norm(gain)), 1e-12)
    out["gain_only"] = gain * float(np.dot(gain, q_proj))
    return out


def _choose_wrong_chart(
    *,
    charts: dict[int, np.ndarray],
    true_condition: int,
    true_image: int,
    true_time: int,
    condition_meta: dict[int, tuple[int, int]],
    condition_features: dict[int, dict[str, float]],
    delta_e: np.ndarray,
    target_norm: float,
    projection: np.ndarray,
    chart_space: str,
    basis: np.ndarray,
    pool: str,
    match_features: list[str],
    match_scales: dict[str, float],
) -> tuple[int, np.ndarray, float, dict[str, float]]:
    best_cond = -1
    best_q = np.full(projection.shape[0], np.nan, dtype=np.float64)
    best_match_score = float("inf")
    best_norm_diff = float("inf")
    best_feature_diffs: dict[str, float] = {}
    true_features = condition_features.get(int(true_condition), {})
    for cond, chart in charts.items():
        if int(cond) == int(true_condition):
            continue
        img, tt = condition_meta.get(int(cond), (-999, -999))
        if pool == "same_time_different_image" and int(tt) != int(true_time):
            continue
        if pool == "same_image_wrong_time" and (int(img) != int(true_image) or int(tt) == int(true_time)):
            continue
        if pool == "different_image" and int(img) == int(true_image):
            continue
        q_raw = np.asarray(chart, dtype=np.float64) @ np.asarray(delta_e, dtype=np.float64)
        q_proj = np.asarray(projection, dtype=np.float64) @ q_raw
        if chart_space == "compact" and basis.shape[1] > 0:
            q_proj = basis @ (basis.T @ q_proj)
        norm = float(np.linalg.norm(q_proj))
        feature_diffs: dict[str, float] = {}
        feature_penalty = 0.0
        for feature in match_features:
            true_val = float(true_features.get(feature, float("nan")))
            cand_val = float(condition_features.get(int(cond), {}).get(feature, float("nan")))
            raw_diff = abs(cand_val - true_val) if np.isfinite(true_val) and np.isfinite(cand_val) else 0.0
            scale = float(match_scales.get(feature, 1.0))
            scale = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
            feature_diffs[f"{feature}_abs_diff"] = raw_diff
            feature_penalty += raw_diff / scale
        norm_diff = abs(norm - float(target_norm))
        match_score = norm_diff + feature_penalty
        if match_score < best_match_score:
            best_cond = int(cond)
            best_q = q_raw
            best_match_score = float(match_score)
            best_norm_diff = float(norm_diff)
            best_feature_diffs = feature_diffs
    return best_cond, best_q, best_norm_diff, best_feature_diffs


def _wrong_chart_match_features(spec: str) -> list[str]:
    aliases = {
        "none": "",
        "rate": "condition_mean_rate",
        "mean_rate": "condition_mean_rate",
        "global_rate": "condition_mean_rate",
        "response_norm": "condition_response_norm",
        "norm": "condition_response_norm",
        "image_structure": "condition_image_structure_score",
        "local_image_structure": "condition_local_image_structure_score",
    }
    out: list[str] = []
    for raw in parse_str_list(str(spec)):
        key = aliases.get(str(raw), str(raw))
        if not key:
            continue
        if key not in out:
            out.append(key)
    return out


def _condition_feature_scales(condition_features: dict[int, dict[str, float]], features: list[str]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for feature in features:
        vals = np.asarray([float(v.get(feature, float("nan"))) for v in condition_features.values()], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        scales[feature] = float(np.nanstd(vals)) if vals.size > 1 and float(np.nanstd(vals)) > 1e-12 else 1.0
    return scales


def _compute_base_rates(
    *,
    model: Any,
    dset: Any,
    stim_lags: np.ndarray,
    samples: Any,
    common_units: np.ndarray,
    gains: np.ndarray,
    dataset_idx: int,
    args: argparse.Namespace,
) -> np.ndarray:
    preds = []
    for start in range(0, samples.source_indices.size, int(args.batch_size)):
        idx = samples.source_indices[start : start + int(args.batch_size)]
        stim = _stim_batch(dset, idx, stim_lags)
        behavior = _behavior_batch(dset, idx)
        pred = _predict(model, stim, behavior, dataset_idx).detach().cpu().numpy()
        preds.append(pred[:, common_units] * gains[None, :])
    return np.concatenate(preds, axis=0).astype(np.float64)


def _parse_float_list(spec: str, *, allow_zero: bool = False) -> list[float]:
    out: list[float] = []
    for raw in parse_str_list(str(spec)):
        try:
            val = float(raw)
        except ValueError:
            continue
        if np.isfinite(val) and (val > 0.0 or (allow_zero and val == 0.0)):
            out.append(val)
    return out


def _chart_injection_delta_y(
    *,
    pairs: dict[str, np.ndarray],
    charts: dict[int, np.ndarray],
    n_units: int,
    noise_sd: float,
    seed: int,
) -> np.ndarray:
    delta_e = np.asarray(pairs["delta_e"], dtype=np.float64)
    condition_ids = np.asarray(pairs["condition_id"], dtype=np.int64)
    out = np.zeros((delta_e.shape[0], int(n_units)), dtype=np.float64)
    for i, cond in enumerate(condition_ids):
        chart = charts.get(int(cond))
        if chart is None:
            out[i] = np.nan
        else:
            out[i] = np.asarray(chart, dtype=np.float64) @ delta_e[i]
    if float(noise_sd) > 0.0:
        rng = np.random.default_rng(int(seed))
        out = out + rng.normal(0.0, float(noise_sd), size=out.shape)
    return out


def _score_session(
    *,
    session: str,
    subject: str,
    pairs: dict[str, np.ndarray],
    labels: np.ndarray,
    condition_meta: dict[int, tuple[int, int]],
    samples: Any,
    j: np.ndarray,
    target_cov: np.ndarray,
    projection_controls: list[str],
    k_list: list[int],
    primary_k: int,
    rf_bins: np.ndarray | None,
    split_mode: str,
    n_folds: int,
    min_train_samples_per_chart: int,
    wrong_chart_pool: str,
    condition_features: dict[int, dict[str, float]],
    wrong_chart_match_features: list[str],
    wrong_chart_match_scales: dict[str, float],
    score_mode: str,
    unit_score_subsets: dict[str, np.ndarray] | None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    folds = _chart_swap_splits(pairs, int(n_folds), int(seed), str(split_mode))
    if not folds:
        leakage_rows.append(
            {
                "session": session,
                "fold": -1,
                "split_mode": str(split_mode),
                "n_train_conditions": 0,
                "n_test_conditions": 0,
                "n_shared_conditions": 0,
                "n_train_trials": 0,
                "n_test_trials": 0,
                "n_shared_trials": 0,
                "n_shared_trial_pairs": 0,
                "n_train_pairs": 0,
                "n_test_pairs": 0,
                "status": "no_valid_folds",
            }
        )
        return pair_rows, leakage_rows

    delta_y = np.asarray(pairs["delta_y"], dtype=np.float64)
    delta_e = np.asarray(pairs["delta_e"], dtype=np.float64)
    condition_ids = np.asarray(pairs["condition_id"], dtype=np.int64)
    gain_axis_raw = np.ones(delta_y.shape[1], dtype=np.float64)
    if unit_score_subsets is None:
        unit_score_subsets = {"all_units": np.ones(delta_y.shape[1], dtype=bool)}
    unit_score_subsets = {
        str(name): np.asarray(mask, dtype=bool)
        for name, mask in unit_score_subsets.items()
        if np.asarray(mask, dtype=bool).shape == (delta_y.shape[1],) and int(np.sum(np.asarray(mask, dtype=bool))) >= 3
    }
    if "all_units" not in unit_score_subsets:
        unit_score_subsets = {"all_units": np.ones(delta_y.shape[1], dtype=bool), **unit_score_subsets}

    for fold_idx, (held_ids, train_mask, test_mask) in enumerate(folds):
        train_conditions = set(int(v) for v in np.unique(condition_ids[train_mask]))
        test_conditions = set(int(v) for v in np.unique(condition_ids[test_mask]))
        train_trials = _trial_set(pairs["trial_a"], pairs["trial_b"], train_mask)
        test_trials = _trial_set(pairs["trial_a"], pairs["trial_b"], test_mask)
        train_trial_pairs = _trial_pair_keys(pairs["trial_a"], pairs["trial_b"], train_mask)
        test_trial_pairs = _trial_pair_keys(pairs["trial_a"], pairs["trial_b"], test_mask)
        shared_conditions = sorted(train_conditions.intersection(test_conditions))
        shared_trials = sorted(train_trials.intersection(test_trials))
        shared_trial_pairs = sorted(train_trial_pairs.intersection(test_trial_pairs))
        leakage_rows.append(
            {
                "session": session,
                "fold": int(fold_idx),
                "split_mode": str(split_mode),
                "n_train_conditions": int(len(train_conditions)),
                "n_test_conditions": int(len(test_conditions)),
                "n_shared_conditions": int(len(shared_conditions)),
                "n_train_trials": int(len(train_trials)),
                "n_test_trials": int(len(test_trials)),
                "n_shared_trials": int(len(shared_trials)),
                "n_shared_trial_pairs": int(len(shared_trial_pairs)),
                "n_train_pairs": int(np.sum(train_mask)),
                "n_test_pairs": int(np.sum(test_mask)),
                "status": (
                    "fail"
                    if (split_mode in {"trial_disjoint", "trial_disjoint_drift_test", "drift_trial_disjoint"} and shared_trials)
                    or shared_trial_pairs
                    else "pass"
                ),
            }
        )
        if str(split_mode) in {"trial_disjoint", "trial_disjoint_drift_test", "drift_trial_disjoint"}:
            train_sample_mask = ~np.isin(samples.trial_ids, held_ids)
        else:
            train_sample_mask = np.isin(labels, list(train_conditions))
        charts, chart_counts = _condition_charts(
            j=j,
            labels=labels,
            train_sample_mask=train_sample_mask,
            min_train_samples_per_chart=int(min_train_samples_per_chart),
        )
        if len(charts) < 2:
            continue
        fold_delta_y = delta_y
        if str(pairs.get("delta_y_mode", "")) == "linear_chart_injection":
            fold_delta_y = _chart_injection_delta_y(
                pairs=pairs,
                charts=charts,
                n_units=delta_y.shape[1],
                noise_sd=float(pairs.get("linear_chart_injection_noise_sd", 0.0)),
                seed=int(seed) + int(fold_idx) * 17011,
            )

        test_indices = np.flatnonzero(test_mask)
        rng_base = np.random.default_rng(int(seed) + int(fold_idx) * 10007)
        shuffled_delta_e = delta_e[test_indices].copy()
        if shuffled_delta_e.shape[0] > 1:
            shuffled_delta_e = shuffled_delta_e[rng_base.permutation(shuffled_delta_e.shape[0])]

        for projection_i, projection_control in enumerate(projection_controls):
            modes = _projection_modes(str(projection_control), target_cov)
            projection = _projection_complement(delta_y.shape[1], modes)
            var = _diag_whitener(fold_delta_y, train_mask, projection)
            gain_axis = projection @ gain_axis_raw
            basis_cache: dict[int, tuple[np.ndarray, int]] = {}
            for k in sorted(set([int(primary_k), *[int(v) for v in k_list]])):
                basis_cache[int(k)] = _basis_from_j(j, train_sample_mask, projection, int(k))

            chart_specs: list[tuple[str, int, np.ndarray, int]] = [("full", 0, np.zeros((delta_y.shape[1], 0)), 0)]
            for k in sorted(set([int(primary_k), *[int(v) for v in k_list]])):
                basis, rank = basis_cache[int(k)]
                if basis.shape[1] > 0:
                    chart_specs.append(("compact", int(k), basis, int(rank)))

            for chart_space, basis_k, basis, basis_rank in chart_specs:
                rng = np.random.default_rng(
                    int(seed) + int(fold_idx) * 7919 + int(basis_k) * 1009 + int(projection_i) * 1543
                )
                random_basis = None
                unit_perm = None
                rf_perm = None
                if chart_space == "compact" and basis_k > 0:
                    qrand, _ = np.linalg.qr(rng.standard_normal((delta_y.shape[1], max(int(basis_k), 1))))
                    random_basis = qrand[:, : int(basis_k)]
                    unit_perm = rng.permutation(delta_y.shape[1])
                    if rf_bins is not None:
                        rf_perm = fixed_within_bin_permutation(np.asarray(rf_bins, dtype=np.int64), rng)

                for local_i, pair_i in enumerate(test_indices):
                    cond = int(condition_ids[pair_i])
                    if cond not in charts:
                        continue
                    de = delta_e[pair_i]
                    dy = projection @ fold_delta_y[pair_i]
                    q_true_raw = charts[cond] @ de
                    true_vecs = _project_chart_vector(
                        q_true_raw,
                        projection=projection,
                        chart_space=chart_space,
                        basis=basis,
                        random_basis=random_basis,
                        unit_perm=unit_perm,
                        rf_perm=rf_perm,
                        gain_axis=gain_axis,
                    )
                    true_key = "full" if chart_space == "full" else "compact"
                    q_true = true_vecs.get(true_key)
                    if q_true is None or not np.isfinite(q_true).all():
                        continue
                    true_norm = float(np.linalg.norm(q_true))
                    wrong_cond, q_wrong_raw, wrong_norm_diff, wrong_feature_diffs = _choose_wrong_chart(
                        charts=charts,
                        true_condition=cond,
                        true_image=int(pairs["image_id"][pair_i]),
                        true_time=int(pairs["time_context"][pair_i]),
                        condition_meta=condition_meta,
                        condition_features=condition_features,
                        delta_e=de,
                        target_norm=true_norm,
                        projection=projection,
                        chart_space=chart_space,
                        basis=basis,
                        pool=str(wrong_chart_pool),
                        match_features=wrong_chart_match_features,
                        match_scales=wrong_chart_match_scales,
                    )
                    if wrong_cond < 0:
                        continue
                    wrong_vecs = _project_chart_vector(
                        q_wrong_raw,
                        projection=projection,
                        chart_space=chart_space,
                        basis=basis,
                        random_basis=random_basis,
                        unit_perm=unit_perm,
                        rf_perm=rf_perm,
                        gain_axis=gain_axis,
                    )
                    q_wrong = wrong_vecs.get(true_key, wrong_vecs.get("full"))
                    q_gain = true_vecs.get("gain_only", np.full_like(q_true, np.nan))
                    q_random = true_vecs.get("random", np.full_like(q_true, np.nan))
                    q_unit = true_vecs.get("unit_shuffle", np.full_like(q_true, np.nan))
                    q_rf = true_vecs.get("rf_readout", np.full_like(q_true, np.nan))
                    q_shuffled_eye_raw = charts[cond] @ shuffled_delta_e[local_i]
                    q_shuffled_eye = _project_chart_vector(
                        q_shuffled_eye_raw,
                        projection=projection,
                        chart_space=chart_space,
                        basis=basis,
                        random_basis=None,
                        unit_perm=None,
                        rf_perm=None,
                        gain_axis=gain_axis,
                    ).get(true_key)
                    q_sign = -q_true
                    q_axis = charts[cond] @ np.asarray([de[1], de[0]], dtype=np.float64)
                    q_axis = _project_chart_vector(
                        q_axis,
                        projection=projection,
                        chart_space=chart_space,
                        basis=basis,
                        random_basis=None,
                        unit_perm=None,
                        rf_perm=None,
                        gain_axis=gain_axis,
                    ).get(true_key)

                    row_common = {
                        "session": session,
                        "subject": subject,
                        "fold": int(fold_idx),
                        "trial_i": int(pairs["trial_a"][pair_i]),
                        "trial_j": int(pairs["trial_b"][pair_i]),
                        "time_i": int(samples.time_indices[int(pairs["sample_a"][pair_i])]),
                        "time_j": int(samples.time_indices[int(pairs["sample_b"][pair_i])]),
                        "condition_id": cond,
                        "wrong_condition_id": int(wrong_cond),
                        "image_id": int(pairs["image_id"][pair_i]),
                        "wrong_image_id": int(condition_meta.get(int(wrong_cond), (-1, -1))[0]),
                        "time_context": int(pairs["time_context"][pair_i]),
                        "wrong_time_context": int(condition_meta.get(int(wrong_cond), (-1, -1))[1]),
                        "drift_mask": bool(pairs["drift_mask"][pair_i]),
                        "delta_eye_x": float(de[0]),
                        "delta_eye_y": float(de[1]),
                        "delta_eye_norm": float(np.linalg.norm(de)),
                        "projection_control": projection_control,
                        "chart_space": chart_space,
                        "basis_k": int(basis_k),
                        "basis_rank": int(basis_rank),
                        "score_mode": score_mode,
                        "wrong_chart_pool": str(wrong_chart_pool),
                        "wrong_chart_match_features": (
                            "+".join(wrong_chart_match_features) if wrong_chart_match_features else "norm_only"
                        ),
                        "prediction_norm_true": true_norm,
                        "prediction_norm_wrong": float(np.linalg.norm(q_wrong)),
                        "image_structure_score": float(
                            pairs.get("image_structure_score", np.full(delta_y.shape[0], np.nan))[pair_i]
                        ),
                        "local_image_structure_score": float(
                            pairs.get("local_image_structure_score", np.full(delta_y.shape[0], np.nan))[pair_i]
                        ),
                        "wrong_chart_norm_abs_diff": float(wrong_norm_diff),
                        "true_chart_train_samples": int(chart_counts.get(cond, 0)),
                        "wrong_chart_train_samples": int(chart_counts.get(int(wrong_cond), 0)),
                        "rate_match_bin": "not_binned",
                    }
                    row_common.update({f"wrong_chart_{key}": float(val) for key, val in wrong_feature_diffs.items()})
                    for subset_name, subset_mask in unit_score_subsets.items():
                        m = np.asarray(subset_mask, dtype=bool)
                        score_true = _alignment_score(dy[m], q_true[m], var[m], score_mode)
                        score_wrong = _alignment_score(dy[m], q_wrong[m], var[m], score_mode)
                        score_gain = _alignment_score(dy[m], q_gain[m], var[m], score_mode)
                        score_random = _alignment_score(dy[m], q_random[m], var[m], score_mode)
                        score_unit = _alignment_score(dy[m], q_unit[m], var[m], score_mode)
                        score_rf = _alignment_score(dy[m], q_rf[m], var[m], score_mode)
                        score_shuffled_eye = _alignment_score(dy[m], q_shuffled_eye[m], var[m], score_mode)
                        score_sign = _alignment_score(dy[m], q_sign[m], var[m], score_mode)
                        score_axis = _alignment_score(dy[m], q_axis[m], var[m], score_mode)
                        row = {
                            **row_common,
                            "unit_score_subset": str(subset_name),
                            "n_unit_score_subset": int(np.sum(m)),
                            "score_true_chart": score_true,
                            "score_wrong_chart": score_wrong,
                            "score_gain_only": score_gain,
                            "score_random": score_random,
                            "score_unit_shuffle": score_unit,
                            "score_rf_readout_null": score_rf,
                            "score_shuffled_eye": score_shuffled_eye,
                            "score_sign_flipped": score_sign,
                            "score_axis_swapped": score_axis,
                            "true_minus_wrong": score_true - score_wrong,
                            "true_minus_gain": score_true - score_gain,
                            "true_minus_random": score_true - score_random,
                            "true_minus_unit_shuffle": score_true - score_unit,
                            "true_minus_rf_readout": score_true - score_rf,
                            "true_minus_shuffled_eye": score_true - score_shuffled_eye,
                        }
                        pair_rows.append(row)
    return pair_rows, leakage_rows


def _summarize_pair_rows(rows: list[dict[str, Any]], *, seed: int, n_bootstrap: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_names = [
        "true_minus_wrong",
        "true_minus_gain",
        "true_minus_random",
        "true_minus_unit_shuffle",
        "true_minus_rf_readout",
        "true_minus_shuffled_eye",
    ]
    session_rows: list[dict[str, Any]] = []
    def median_subset_count(block: list[dict[str, Any]]) -> int:
        vals = np.asarray([float(r.get("n_unit_score_subset", float("nan"))) for r in block], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return int(np.nanmedian(vals)) if vals.size else 0

    groups: dict[tuple[str, str, int, str, str, str, str, str, float, float, bool], list[dict[str, Any]]] = {}
    for row in rows:
        for drift_only in (False, True):
            if drift_only and not bool(row.get("drift_mask", False)):
                continue
            key = (
                str(row["session"]),
                str(row["projection_control"]),
                int(row["basis_k"]),
                str(row["chart_space"]),
                str(row.get("unit_score_subset", "all_units")),
                str(row.get("wrong_chart_pool", "unknown")),
                str(row.get("wrong_chart_match_features", "norm_only")),
                str(row.get("pseudo_control_mode", "recorded")),
                float(row.get("pseudo_control_scale", 1.0)),
                float(row.get("pseudo_injection_noise_sd", 0.0)),
                drift_only,
            )
            groups.setdefault(key, []).append(row)
    for (
        session,
        projection,
        k,
        chart_space,
        unit_subset,
        wrong_pool,
        wrong_match,
        pseudo_mode,
        pseudo_scale,
        pseudo_noise_sd,
        drift_only,
    ), block in sorted(groups.items()):
        base = {
            "session": session,
            "projection_control": projection,
            "basis_k": int(k),
            "chart_space": chart_space,
            "unit_score_subset": unit_subset,
            "wrong_chart_pool": wrong_pool,
            "wrong_chart_match_features": wrong_match,
            "pseudo_control_mode": pseudo_mode,
            "pseudo_control_scale": float(pseudo_scale),
            "pseudo_injection_noise_sd": float(pseudo_noise_sd),
            "n_unit_score_subset": median_subset_count(block),
            "sample_set": "drift_only" if drift_only else "all",
            "n_pairs": int(len(block)),
        }
        for name in metric_names:
            vals = np.asarray([float(r.get(name, np.nan)) for r in block], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            base[f"mean_{name}"] = float(np.mean(vals)) if vals.size else float("nan")
            base[f"median_{name}"] = float(np.median(vals)) if vals.size else float("nan")
            base[f"n_positive_{name}"] = int(np.sum(vals > 0.0)) if vals.size else 0
            base[f"sign_test_{name}_p_two_sided"] = sign_test_p_two_sided(int(np.sum(vals > 0.0)), int(vals.size))
        session_rows.append(base)

    boot_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    summary_groups: dict[tuple[str, int, str, str, str, str, str, float, float, str], list[dict[str, Any]]] = {}
    for row in session_rows:
        key = (
            str(row["projection_control"]),
            int(row["basis_k"]),
            str(row["chart_space"]),
            str(row.get("unit_score_subset", "all_units")),
            str(row.get("wrong_chart_pool", "unknown")),
            str(row.get("wrong_chart_match_features", "norm_only")),
            str(row.get("pseudo_control_mode", "recorded")),
            float(row.get("pseudo_control_scale", 1.0)),
            float(row.get("pseudo_injection_noise_sd", 0.0)),
            str(row["sample_set"]),
        )
        summary_groups.setdefault(key, []).append(row)
    for (
        projection,
        k,
        chart_space,
        unit_subset,
        wrong_pool,
        wrong_match,
        pseudo_mode,
        pseudo_scale,
        pseudo_noise_sd,
        sample_set,
    ), block in sorted(summary_groups.items()):
        for name in metric_names:
            vals = np.asarray([float(r.get(f"mean_{name}", np.nan)) for r in block], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            mean, lo, hi = bootstrap_mean_ci(vals, rng=rng, n_bootstrap=int(n_bootstrap))
            boot_rows.append(
                {
                    "projection_control": projection,
                    "basis_k": int(k),
                    "chart_space": chart_space,
                    "unit_score_subset": unit_subset,
                    "wrong_chart_pool": wrong_pool,
                    "wrong_chart_match_features": wrong_match,
                    "pseudo_control_mode": pseudo_mode,
                    "pseudo_control_scale": float(pseudo_scale),
                    "pseudo_injection_noise_sd": float(pseudo_noise_sd),
                    "n_unit_score_subset": median_subset_count(block),
                    "sample_set": sample_set,
                    "metric": name,
                    "n_sessions": int(vals.size),
                    "session_mean": mean,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "n_positive_sessions": int(np.sum(vals > 0.0)),
                    "sign_test_p_two_sided": sign_test_p_two_sided(int(np.sum(vals > 0.0)), int(vals.size)),
                }
            )
    return session_rows, boot_rows


def _primary_decision(
    bootstrap_rows: list[dict[str, Any]],
    *,
    leakage_failures: int,
    primary_projection_control: str,
    primary_k: int,
    min_sessions: int = 3,
) -> tuple[str, dict[str, Any]]:
    def pick(metric: str) -> dict[str, Any] | None:
        matches = [
            r
            for r in bootstrap_rows
            if str(r.get("projection_control")) == str(primary_projection_control)
            and str(r.get("chart_space")) == "compact"
            and int(r.get("basis_k", -1)) == int(primary_k)
            and str(r.get("sample_set")) == "all"
            and str(r.get("unit_score_subset", "all_units")) == "all_units"
            and str(r.get("metric")) == metric
        ]
        return matches[0] if matches else None

    primary = pick("true_minus_wrong")
    required_metrics = [
        "true_minus_wrong",
        "true_minus_gain",
        "true_minus_random",
        "true_minus_unit_shuffle",
        "true_minus_rf_readout",
    ]
    rows = {metric: pick(metric) for metric in required_metrics}
    checks: dict[str, Any] = {
        "decision_rule": (
            "candidate_positive requires zero leakage failures, at least min_sessions, "
            "positive primary session mean, primary CI low > 0, majority positive sessions, "
            "and all required control CI lows > 0"
        ),
        "min_sessions": int(min_sessions),
        "leakage_failures": int(leakage_failures),
        "required_metrics": required_metrics,
        "missing_metrics": [metric for metric, row in rows.items() if row is None],
    }
    if primary is None:
        checks["failure_reason"] = "missing_primary_row"
        return "diagnostic", checks
    n_sessions = int(primary.get("n_sessions", 0))
    n_positive = int(primary.get("n_positive_sessions", 0))
    primary_mean = float(primary.get("session_mean", float("nan")))
    primary_ci_low = float(primary.get("bootstrap_ci_low", float("nan")))
    checks.update(
        {
            "primary": primary,
            "n_sessions_ok": n_sessions,
            "n_positive_sessions": n_positive,
            "primary_session_mean": primary_mean,
            "primary_ci_low": primary_ci_low,
            "control_ci_lows": {
                metric: float(row.get("bootstrap_ci_low", float("nan"))) if row is not None else float("nan")
                for metric, row in rows.items()
                if metric != "true_minus_wrong"
            },
        }
    )
    controls_pass = all(
        row is not None and float(row.get("bootstrap_ci_low", float("nan"))) > 0.0
        for metric, row in rows.items()
        if metric != "true_minus_wrong"
    )
    positive_session_majority = n_positive > (n_sessions / 2.0)
    candidate = (
        int(leakage_failures) == 0
        and not checks["missing_metrics"]
        and n_sessions >= int(min_sessions)
        and np.isfinite(primary_mean)
        and primary_mean > 0.0
        and np.isfinite(primary_ci_low)
        and primary_ci_low > 0.0
        and positive_session_majority
        and controls_pass
    )
    if not candidate:
        failures: list[str] = []
        if int(leakage_failures) != 0:
            failures.append("leakage_failures")
        if checks["missing_metrics"]:
            failures.append("missing_required_metrics")
        if n_sessions < int(min_sessions):
            failures.append("too_few_primary_sessions")
        if not (np.isfinite(primary_mean) and primary_mean > 0.0):
            failures.append("primary_mean_not_positive")
        if not (np.isfinite(primary_ci_low) and primary_ci_low > 0.0):
            failures.append("primary_ci_crosses_zero")
        if not positive_session_majority:
            failures.append("positive_session_majority_failed")
        if not controls_pass:
            failures.append("required_control_ci_failed")
        checks["failure_reason"] = ",".join(failures)
    return ("candidate_positive" if candidate else "diagnostic"), checks


def _write_figures(out: Path, boot_rows: list[dict[str, Any]], primary_projection: str, primary_k: int) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    primary = [
        r
        for r in boot_rows
        if str(r.get("projection_control")) == str(primary_projection)
        and int(r.get("basis_k", -1)) in {0, int(primary_k)}
        and str(r.get("sample_set")) == "all"
        and str(r.get("unit_score_subset", "all_units")) == "all_units"
        and str(r.get("metric")) in {"true_minus_wrong", "true_minus_gain", "true_minus_random", "true_minus_unit_shuffle"}
    ]
    if primary:
        labels = [f"{r['chart_space']} k={r['basis_k']}\n{str(r['metric']).replace('true_minus_', '')}" for r in primary]
        vals = [float(r["session_mean"]) for r in primary]
        fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(vals)), 3.6))
        ax.bar(np.arange(len(vals)), vals, color="#4c78a8", alpha=0.88)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("True minus control score")
        ax.set_title(f"Chart-swap alignment ({primary_projection})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / "chart_swap_summary.pdf")
        fig.savefig(fig_dir / "chart_swap_summary.png", dpi=220)
        plt.close(fig)
    else:
        (fig_dir / "chart_swap_summary.pdf").write_text("No primary rows available.\n", encoding="utf-8")

    drift = [
        r
        for r in boot_rows
        if str(r.get("projection_control")) == str(primary_projection)
        and int(r.get("basis_k", -1)) == int(primary_k)
        and str(r.get("chart_space")) == "compact"
        and str(r.get("unit_score_subset", "all_units")) == "all_units"
        and str(r.get("metric")) == "true_minus_wrong"
    ]
    if drift:
        labels = [str(r["sample_set"]) for r in drift]
        vals = [float(r["session_mean"]) for r in drift]
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.bar(np.arange(len(vals)), vals, color="#59a14f", alpha=0.88)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("True minus wrong score")
        ax.set_title("All vs drift-only")
        fig.tight_layout()
        fig.savefig(fig_dir / "drift_vs_all_summary.pdf")
        fig.savefig(fig_dir / "drift_vs_all_summary.png", dpi=220)
        plt.close(fig)
    else:
        (fig_dir / "drift_vs_all_summary.pdf").write_text("No drift rows available.\n", encoding="utf-8")

    # Placeholders with explicit content keep downstream audits honest.
    for name, reason in {
        "chart_swap_alignment_pairs.pdf": "Pair scatter figure not yet implemented.",
        "latency_history_sweep.pdf": "Latency/history sweep not yet implemented.",
        "subspace_controls.pdf": "Subspace-control figure is summarized in CSV first pass.",
    }.items():
        path = fig_dir / name
        if not path.exists():
            fig, ax = plt.subplots(figsize=(5.2, 2.6))
            ax.text(0.5, 0.5, reason, ha="center", va="center", wrap=True)
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Correct-chart versus wrong-chart alignment")
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
    p.add_argument("--k-list", type=str, default="2,10")
    p.add_argument("--primary-k", type=int, default=10)
    p.add_argument("--context-mode", choices=["time_bin", "time_window", "image_only"], default="time_bin")
    p.add_argument("--context-bin-size", type=int, default=10)
    p.add_argument("--min-repeats-per-condition", type=int, default=3)
    p.add_argument("--max-pairs-per-condition", type=int, default=100)
    p.add_argument(
        "--split-mode",
        choices=["trial_disjoint", "trial_disjoint_drift_test", "drift_trial_disjoint"],
        default="trial_disjoint",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-train-samples-per-chart", type=int, default=2)
    p.add_argument(
        "--wrong-chart-pool",
        choices=["any", "different_image", "same_time_different_image", "same_image_wrong_time"],
        default="different_image",
    )
    p.add_argument(
        "--wrong-chart-match",
        type=str,
        default="none",
        help=(
            "Comma-separated secondary wrong-chart matching features in addition to prediction-norm matching: "
            "none,rate,response_norm,image_structure,local_image_structure."
        ),
    )
    p.add_argument("--score-mode", choices=["whitened", "cosine", "unit_dot"], default="whitened")
    p.add_argument(
        "--unit-score-subsets",
        type=str,
        default="all",
        help=(
            "Comma-separated score masks: all,fem_top50,fem_bottom50,gain_top50,"
            "rate_top50,ccnorm_top50 and top/bottom 10/25/50 variants."
        ),
    )
    p.add_argument("--drift-speed-threshold-px", type=float, default=2.0)
    p.add_argument("--drift-pair-delta-threshold-px", type=float, default=5.0)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--min-units", type=int, default=50)
    p.add_argument("--enable-rf-readout-null", action="store_true", default=True)
    p.add_argument("--rf-null-min-bin-units", type=int, default=6)
    p.add_argument("--rf-null-bin-features", type=str, default="rf_xy,tangent_norm,mean_rate,ccnorm")
    p.add_argument("--rf-null-session-yaml-dir", type=Path, default=Path("experiments") / "dataset_configs" / "sessions")
    p.add_argument("--run-pseudo-spike-control", action="store_true")
    p.add_argument(
        "--pseudo-control-modes",
        type=str,
        default="poisson",
        help="Comma-separated pseudo controls: poisson,poisson_scaled,rate_delta,linear_chart_injection.",
    )
    p.add_argument("--pseudo-poisson-scales", type=str, default="2,5,10")
    p.add_argument(
        "--pseudo-injection-noise-sd",
        type=str,
        default="0",
        help="Single value or comma-separated noise SDs for split-aware linear_chart_injection.",
    )
    p.add_argument("--local-image-radius-px", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose-model-load", action="store_true")
    p.add_argument("--init-only", action="store_true")
    return p


def run_analysis(args: argparse.Namespace) -> None:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    projection_controls = parse_str_list(args.projection_controls)
    k_list = parse_int_list(args.k_list)
    wrong_chart_match_features = _wrong_chart_match_features(str(args.wrong_chart_match))
    pseudo_control_modes = parse_str_list(args.pseudo_control_modes)
    pseudo_poisson_scales = _parse_float_list(str(args.pseudo_poisson_scales))
    pseudo_injection_noise_sds = _parse_float_list(str(args.pseudo_injection_noise_sd), allow_zero=True)
    fig3_rows = _load_pickle(Path(args.fig3_cache))
    fig2_rows = _load_pickle(Path(args.fig2_cache))
    fig2 = _fig2_by_session(fig2_rows)
    fig3_by_session = {str(row["session"]): row for row in fig3_rows}
    requested_sessions = parse_str_list(args.sessions)
    if len(requested_sessions) == 1 and requested_sessions[0].lower() == "all":
        requested_sessions = []
    if not requested_sessions:
        requested_sessions = [str(row["session"]) for row in fig3_rows if str(row.get("subject", "")) in {"Allen", "Logan"}]

    config = ChartSwapConfig(
        output_root=str(out),
        sessions=requested_sessions,
        projection_controls=projection_controls,
        target_variant=str(args.target_variant),
        k_list=k_list,
        primary_k=int(args.primary_k),
        context_mode=str(args.context_mode),
        context_bin_size=int(args.context_bin_size),
        split_mode=str(args.split_mode),
        n_folds=int(args.n_folds),
        min_repeats_per_condition=int(args.min_repeats_per_condition),
        max_pairs_per_condition=int(args.max_pairs_per_condition),
        min_train_samples_per_chart=int(args.min_train_samples_per_chart),
        wrong_chart_pool=str(args.wrong_chart_pool),
        wrong_chart_match_features=wrong_chart_match_features,
        score_mode=str(args.score_mode),
        unit_score_subsets=parse_str_list(args.unit_score_subsets),
        drift_speed_threshold_px=float(args.drift_speed_threshold_px),
        drift_pair_delta_threshold_px=float(args.drift_pair_delta_threshold_px),
        run_pseudo_spike_control=bool(args.run_pseudo_spike_control),
        pseudo_control_modes=pseudo_control_modes,
        pseudo_poisson_scales=pseudo_poisson_scales,
        pseudo_injection_noise_sds=pseudo_injection_noise_sds,
        local_image_radius_px=int(args.local_image_radius_px),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    manifest_base = {
        "analysis": "correct_chart_swap_alignment",
        "status": "initialized_not_run" if bool(args.init_only) else "running",
        "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "fig2_cache": str(Path(args.fig2_cache).resolve()),
        "fig3_cache": str(Path(args.fig3_cache).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_config": str(Path(args.model_config).resolve()),
        "dataset_config": str(Path(args.dataset_config).resolve()),
        "claim_guardrail": "This is a relative same-image/history chart-swap alignment test, not an absolute eye-position decoder.",
        "context_match_guardrail": (
            "time_bin is the strict same-image/time condition; time_window and image_only are support-oriented "
            "coarse history matches and should not be promoted as strict A2 without that caveat."
        ),
    }
    write_json(out / "manifest.json", manifest_base)
    if bool(args.init_only):
        for required in (
            "session_inventory.csv",
            "drift_mask_summary.csv",
            "pair_inventory.csv",
            "fold_leakage_audit.csv",
            "chart_alignment_pair_metrics.csv",
            "chart_alignment_session_summary.csv",
            "chart_alignment_bootstrap_summary.csv",
            "chart_swap_control_summary.csv",
            "gain_control_summary.csv",
            "compact_vs_full_summary.csv",
            "pseudo_spike_positive_control.csv",
            "latency_history_sweep.csv",
        ):
            write_csv(out / required, [{"status": "initialized_not_run"}])
        (out / "README.md").write_text("# Correct Chart Swap Alignment\n\nStatus: initialized, not run.\n", encoding="utf-8")
        return

    model, model_info = _load_twin_model(args)
    session_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    pair_inventory_rows: list[dict[str, Any]] = []
    pair_metric_rows: list[dict[str, Any]] = []
    pseudo_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    rf_rows: list[dict[str, Any]] = []

    for session in requested_sessions:
        print(f"[chart-swap] session {session}: starting", flush=True)
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
        dset, stim_lags, samples = _collect_samples(model=model, dataset_idx=dataset_idx, common_units=common_units, args=args)
        eye_px = samples.eyepos_deg * float(samples.pixels_per_degree)
        image_ids, image_meta = _image_ids_for_samples(dset, samples)
        image_structure_metrics, image_structure_meta = _load_fixrsvp_image_metrics()
        sample_image_structure, sample_local_image_structure = _local_image_structure_scores(
            image_ids=image_ids,
            eye_px=eye_px,
            image_metrics=image_structure_metrics,
            radius_px=int(args.local_image_radius_px),
        )
        labels, sample_image_ids, time_contexts, condition_rows = _condition_keys(
            image_ids=image_ids,
            time_indices=samples.time_indices,
            mode=str(args.context_mode),
            bin_size=int(args.context_bin_size),
        )
        condition_meta = {int(r["condition_id"]): (int(r["image_id"]), int(r["time_context"])) for r in condition_rows}
        sample_drift_mask, sample_speed = _sample_drift_mask(
            dset=dset,
            samples=samples,
            pixels_per_degree=float(samples.pixels_per_degree),
            speed_threshold_px=float(args.drift_speed_threshold_px),
        )
        drift_rows.append(
            {
                "session": session,
                "subject": sr.get("subject", ""),
                "n_samples": int(samples.source_indices.size),
                "n_drift_samples": int(np.sum(sample_drift_mask)),
                "drift_sample_fraction": float(np.mean(sample_drift_mask)) if sample_drift_mask.size else float("nan"),
                "speed_threshold_px": float(args.drift_speed_threshold_px),
                "pair_delta_threshold_px": float(args.drift_pair_delta_threshold_px),
                "sample_speed_px_p50": float(np.nanpercentile(sample_speed, 50)) if np.any(np.isfinite(sample_speed)) else float("nan"),
                "sample_speed_px_p90": float(np.nanpercentile(sample_speed, 90)) if np.any(np.isfinite(sample_speed)) else float("nan"),
                "event_label_source": "adjacent_eye_step_proxy",
            }
        )
        pairs, inventory = build_chart_pair_dataset(
            samples=samples,
            eye_px=eye_px,
            labels=labels,
            image_ids=sample_image_ids,
            time_contexts=time_contexts,
            sample_drift_mask=sample_drift_mask,
            sample_image_structure_score=sample_image_structure,
            sample_local_image_structure_score=sample_local_image_structure,
            drift_pair_delta_threshold_px=float(args.drift_pair_delta_threshold_px),
            min_repeats_per_condition=int(args.min_repeats_per_condition),
            max_pairs_per_condition=int(args.max_pairs_per_condition),
            seed=int(args.seed) + dataset_idx * 101,
        )
        for row in inventory:
            row.update({"session": session, "subject": sr.get("subject", "")})
        pair_inventory_rows.extend(inventory)
        condition_features = {
            int(row["condition_id"]): {
                "condition_mean_rate": float(row.get("condition_mean_rate", float("nan"))),
                "condition_response_norm": float(row.get("condition_response_norm", float("nan"))),
                "condition_image_structure_score": float(row.get("condition_image_structure_score", float("nan"))),
                "condition_local_image_structure_score": float(
                    row.get("condition_local_image_structure_score", float("nan"))
                ),
            }
            for row in inventory
        }
        wrong_chart_match_scales = _condition_feature_scales(condition_features, wrong_chart_match_features)
        if pairs["delta_y"].shape[0] < max(20, int(args.n_folds) * 3):
            session_rows.append(
                {
                    "session": session,
                    "subject": sr.get("subject", ""),
                    "status": "too_few_pairs",
                    "n_common_units": int(common_units.size),
                    "n_pairs": int(pairs["delta_y"].shape[0]),
                    "n_pair_conditions": int(np.unique(pairs["condition_id"]).size) if pairs["condition_id"].size else 0,
                    **image_meta,
                    **image_structure_meta,
                }
            )
            continue
        print(f"[chart-swap] session {session}: fitting gains and finite-difference charts", flush=True)
        gains, rescale_status = _fit_rescale_gains(
            model=model,
            dset=dset,
            stim_lags=stim_lags,
            samples=samples,
            common_units=common_units,
            dataset_idx=dataset_idx,
            args=args,
        )
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
        unit_score_subsets = _unit_score_subset_masks(j, parse_str_list(args.unit_score_subsets), rf_meta.unit_rows)
        rows, leaks = _score_session(
            session=session,
            subject=str(sr.get("subject", "")),
            pairs=pairs,
            labels=labels,
            condition_meta=condition_meta,
            samples=samples,
            j=j,
            target_cov=target,
            projection_controls=projection_controls,
            k_list=k_list,
            primary_k=int(args.primary_k),
            rf_bins=rf_meta.bins,
            split_mode=str(args.split_mode),
            n_folds=int(args.n_folds),
            min_train_samples_per_chart=int(args.min_train_samples_per_chart),
            wrong_chart_pool=str(args.wrong_chart_pool),
            condition_features=condition_features,
            wrong_chart_match_features=wrong_chart_match_features,
            wrong_chart_match_scales=wrong_chart_match_scales,
            score_mode=str(args.score_mode),
            unit_score_subsets=unit_score_subsets,
            seed=int(args.seed) + dataset_idx * 1009,
        )
        pair_metric_rows.extend(rows)
        leakage_rows.extend(leaks)
        if rows:
            score_status = "ok_scored"
        elif any(str(r.get("status")) == "no_valid_folds" for r in leaks):
            score_status = "ok_no_valid_folds"
        else:
            score_status = "ok_no_scored_rows"

        if bool(args.run_pseudo_spike_control):
            print(f"[chart-swap] session {session}: running pseudo positive controls", flush=True)
            rates = _compute_base_rates(
                model=model,
                dset=dset,
                stim_lags=stim_lags,
                samples=samples,
                common_units=common_units,
                gains=gains,
                dataset_idx=dataset_idx,
                args=args,
            )
            rng = np.random.default_rng(int(args.seed) + dataset_idx * 4441)
            pseudo_specs: list[tuple[str, float, float, np.ndarray]] = []
            if "poisson" in pseudo_control_modes:
                pseudo = rng.poisson(np.clip(rates, 0.0, 1e6)).astype(np.float64)
                pseudo_specs.append(("poisson", 1.0, 0.0, pseudo[pairs["sample_a"]] - pseudo[pairs["sample_b"]]))
            if "poisson_scaled" in pseudo_control_modes:
                for scale in pseudo_poisson_scales:
                    pseudo = rng.poisson(np.clip(rates * float(scale), 0.0, 1e6)).astype(np.float64) / float(scale)
                    pseudo_specs.append(
                        (
                            f"poisson_scaled_{float(scale):g}x",
                            float(scale),
                            0.0,
                            pseudo[pairs["sample_a"]] - pseudo[pairs["sample_b"]],
                        )
                    )
            if "rate_delta" in pseudo_control_modes:
                pseudo_specs.append(("rate_delta", 1.0, 0.0, rates[pairs["sample_a"]] - rates[pairs["sample_b"]]))
            if "linear_chart_injection" in pseudo_control_modes:
                for noise_i, noise_sd in enumerate(pseudo_injection_noise_sds):
                    pseudo_specs.append(
                        (
                            "linear_chart_injection",
                            1.0,
                            float(noise_sd),
                            np.asarray(pairs["delta_y"], dtype=np.float64),
                        )
                    )
            for pseudo_i, (pseudo_mode, pseudo_scale, pseudo_noise_sd, pseudo_delta_y) in enumerate(pseudo_specs):
                pseudo_pairs = dict(pairs)
                pseudo_pairs["delta_y"] = np.asarray(pseudo_delta_y, dtype=np.float64)
                if pseudo_mode == "linear_chart_injection":
                    pseudo_pairs["delta_y_mode"] = "linear_chart_injection"
                    pseudo_pairs["linear_chart_injection_noise_sd"] = float(pseudo_noise_sd)
                p_rows, _ = _score_session(
                    session=session,
                    subject=str(sr.get("subject", "")),
                    pairs=pseudo_pairs,
                    labels=labels,
                    condition_meta=condition_meta,
                    samples=samples,
                    j=j,
                    target_cov=target,
                    projection_controls=projection_controls,
                    k_list=[int(args.primary_k)],
                    primary_k=int(args.primary_k),
                    rf_bins=rf_meta.bins,
                    split_mode=str(args.split_mode),
                    n_folds=int(args.n_folds),
                    min_train_samples_per_chart=int(args.min_train_samples_per_chart),
                    wrong_chart_pool=str(args.wrong_chart_pool),
                    condition_features=condition_features,
                    wrong_chart_match_features=wrong_chart_match_features,
                    wrong_chart_match_scales=wrong_chart_match_scales,
                    score_mode=str(args.score_mode),
                    unit_score_subsets=unit_score_subsets,
                    seed=int(args.seed) + dataset_idx * 7703 + pseudo_i * 101,
                )
                for row in p_rows:
                    row["pseudo_control_mode"] = pseudo_mode
                    row["pseudo_control_scale"] = float(pseudo_scale)
                    row["pseudo_injection_noise_sd"] = float(pseudo_noise_sd)
                    pseudo_rows.append(row)
        session_rows.append(
            {
                "session": session,
                "subject": sr.get("subject", ""),
                "status": score_status,
                "dataset_idx": int(dataset_idx),
                "n_common_units": int(common_units.size),
                "n_samples_used": int(samples.source_indices.size),
                "n_candidate_samples": int(samples.n_candidate_samples),
                "n_pair_conditions": int(np.unique(pairs["condition_id"]).size),
                "n_pairs": int(pairs["delta_y"].shape[0]),
                "n_drift_pairs": int(np.sum(pairs["drift_mask"])),
                "n_scored_rows": int(len(rows)),
                "unit_score_subsets": ",".join(unit_score_subsets.keys()),
                "wrong_chart_pool": str(args.wrong_chart_pool),
                "wrong_chart_match_features": (
                    "+".join(wrong_chart_match_features) if wrong_chart_match_features else "norm_only"
                ),
                "wrong_chart_match_scales": json.dumps(_json_ready(wrong_chart_match_scales), sort_keys=True),
                "pixels_per_degree": float(samples.pixels_per_degree),
                "step_px": float(args.step_px),
                "rescale_status": rescale_status,
                "rf_null_status": rf_meta.status,
                "rf_null_n_bins": int(rf_meta.n_bins),
                "rf_null_largest_bin_fraction": float(rf_meta.largest_bin_fraction),
                "rf_null_bin_features": rf_meta.bin_features,
                "jacobian_abs_median": float(np.median(np.abs(j))),
                **target_meta,
                **image_meta,
                **image_structure_meta,
            }
        )
        print(f"[chart-swap] session {session}: finished ({len(rows)} pair-score rows)", flush=True)

    session_summary_rows, bootstrap_rows = _summarize_pair_rows(
        pair_metric_rows,
        seed=int(args.seed),
        n_bootstrap=int(args.n_bootstrap),
    )
    pseudo_summary_rows, _ = _summarize_pair_rows(
        pseudo_rows,
        seed=int(args.seed) + 17,
        n_bootstrap=max(100, min(int(args.n_bootstrap), 1000)),
    ) if pseudo_rows else ([{"status": "not_run", "reason": "pass --run-pseudo-spike-control"}], [])

    control_summary = [
        r
        for r in bootstrap_rows
        if str(r.get("metric")) in {"true_minus_wrong", "true_minus_random", "true_minus_unit_shuffle", "true_minus_rf_readout", "true_minus_shuffled_eye"}
    ]
    gain_summary = [r for r in bootstrap_rows if str(r.get("metric")) == "true_minus_gain"]
    compact_vs_full = [
        r
        for r in bootstrap_rows
        if str(r.get("metric")) == "true_minus_wrong"
        and int(r.get("basis_k", -1)) in {0, int(args.primary_k)}
    ]
    latency_rows = [{"status": "not_run", "reason": "latency/history sweep is intentionally deferred until first chart-swap smoke passes"}]

    write_csv(out / "session_inventory.csv", session_rows)
    write_csv(out / "drift_mask_summary.csv", drift_rows)
    write_csv(out / "pair_inventory.csv", pair_inventory_rows)
    write_csv(out / "fold_leakage_audit.csv", leakage_rows)
    write_csv(out / "chart_alignment_pair_metrics.csv", pair_metric_rows)
    write_csv(out / "chart_alignment_session_summary.csv", session_summary_rows)
    write_csv(out / "chart_alignment_bootstrap_summary.csv", bootstrap_rows)
    write_csv(out / "chart_swap_control_summary.csv", control_summary)
    write_csv(out / "gain_control_summary.csv", gain_summary)
    write_csv(out / "compact_vs_full_summary.csv", compact_vs_full)
    write_csv(out / "pseudo_spike_positive_control.csv", pseudo_summary_rows)
    write_csv(out / "latency_history_sweep.csv", latency_rows)
    write_csv(out / "rf_readout_unit_bins.csv", rf_rows)
    _write_figures(out, bootstrap_rows, str(args.primary_projection_control), int(args.primary_k))

    leakage_failures = int(sum(1 for r in leakage_rows if r.get("status") == "fail"))
    decision, decision_checks = _primary_decision(
        bootstrap_rows,
        leakage_failures=leakage_failures,
        primary_projection_control=str(args.primary_projection_control),
        primary_k=int(args.primary_k),
    )
    audit = {
        "status": "ok",
        "decision": decision,
        "decision_checks": decision_checks,
        "n_sessions_requested": int(len(requested_sessions)),
        "n_sessions_ok": int(sum(1 for r in session_rows if str(r.get("status", "")).startswith("ok"))),
        "n_sessions_scored": int(sum(1 for r in session_rows if r.get("status") == "ok_scored")),
        "n_pair_metric_rows": int(len(pair_metric_rows)),
        "n_pseudo_rows": int(len(pseudo_rows)),
        "n_leakage_failures": leakage_failures,
        "primary_projection_control": str(args.primary_projection_control),
        "primary_k": int(args.primary_k),
        "primary_true_minus_wrong": decision_checks.get("primary", {}),
        "context_match_guardrail": manifest_base["context_match_guardrail"],
        "model_info": {k: str(v) for k, v in dict(model_info).items()},
        "claim_guardrail": "Promote only if correct chart beats wrong chart and controls, especially under global_rate+target_pc1 and drift-only auditing.",
    }
    write_json(out / "audit.json", audit)
    write_json(
        out / "manifest.json",
        {
            **manifest_base,
            "status": "ok",
            "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
            "model_info": {k: str(v) for k, v in dict(model_info).items()},
        },
    )
    (out / "README.md").write_text(
        "# Correct Chart Swap Alignment\n\n"
        "Primary tables are `chart_alignment_pair_metrics.csv`, "
        "`chart_alignment_bootstrap_summary.csv`, and `chart_swap_control_summary.csv`.\n\n"
        "The skeptic-facing row is compact k=primary under `global_rate+target_pc1`, "
        "with `metric=true_minus_wrong`. Treat this as content-routing evidence only if "
        "it also beats gain-only and subspace controls and leakage audits pass.\n",
        encoding="utf-8",
    )


def main() -> None:
    run_analysis(build_parser().parse_args())


if __name__ == "__main__":
    main()
