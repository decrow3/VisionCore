from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from VisionCore.paths import VISIONCORE_ROOT
from eval.fixrsvp import get_fixrsvp_data

try:
    from jake.twininfo.common import N_LAGS as DEFAULT_N_LAGS
    from jake.twininfo.common import PPD as DEFAULT_MODEL_PPD
except Exception:
    DEFAULT_N_LAGS = 32
    DEFAULT_MODEL_PPD = 37.50476617


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _finite_vals(x: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _finite_median(x: np.ndarray | list[float]) -> float:
    v = _finite_vals(x)
    return float(np.median(v)) if v.size else float("nan")


def _finite_iqr(x: np.ndarray | list[float]) -> float:
    v = _finite_vals(x)
    return float(np.percentile(v, 75) - np.percentile(v, 25)) if v.size else float("nan")


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([head, sep] + body)


def _distribution_string(counts: Counter[int]) -> str:
    if not counts:
        return ""
    hist = Counter(int(v) for v in counts.values())
    return ";".join(f"{k}:{hist[k]}" for k in sorted(hist))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _build_fold_splits(n_objects: int, folds: int, seed: int) -> list[np.ndarray]:
    if n_objects <= 0:
        return []
    perm = np.random.default_rng(int(seed) + 20000).permutation(int(n_objects))
    folds_eff = max(2, min(int(folds), int(n_objects)))
    return [np.asarray(split, dtype=np.int64) for split in np.array_split(perm, folds_eff)]


def _history_overlap_fraction(t_test: int, t_train: int, n_lags: int) -> float:
    overlap = max(0, int(n_lags) - abs(int(t_test) - int(t_train)))
    return float(overlap / max(int(n_lags), 1))


def _resolve_primary_delta(delta_arcmins: list[float], primary_delta: float | None) -> float:
    if not delta_arcmins:
        return float("nan")
    if primary_delta is None or not np.isfinite(float(primary_delta)):
        return float(delta_arcmins[0])
    return float(min(delta_arcmins, key=lambda d: abs(float(d) - float(primary_delta))))


def _build_group_fold_splits(group_ids: list[int], folds: int) -> list[np.ndarray]:
    if not group_ids:
        return []
    group_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, gid in enumerate(group_ids):
        group_to_indices[int(gid)].append(int(idx))
    groups = sorted(group_to_indices.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    folds_eff = max(2, min(int(folds), len(groups)))
    fold_bins: list[list[int]] = [[] for _ in range(folds_eff)]
    fold_sizes = [0 for _ in range(folds_eff)]
    for _, idxs in groups:
        fold_id = min(range(folds_eff), key=lambda f: (fold_sizes[f], len(fold_bins[f]), f))
        fold_bins[fold_id].extend(int(i) for i in idxs)
        fold_sizes[fold_id] += len(idxs)
    return [np.asarray(sorted(bin_idxs), dtype=np.int64) for bin_idxs in fold_bins if bin_idxs]


def _load_cached_filtered_tangents(root: Path) -> tuple[list[float], dict[float, dict[str, dict[str, Any]]], list[dict[str, str]], dict[str, Any]]:
    with (root / "tangent_maps" / "twin_tangent_maps.pkl").open("rb") as handle:
        cached = pickle.load(handle)
    delta_arcmins = [float(v) for v in cached["delta_arcmins"]]
    object_payload = cast(dict[float, dict[str, dict[str, Any]]], cached["object_payload"])
    dropped_rows = _read_csv_rows(root / "dropped_objects_union_basis.csv")
    dropped_by_delta: dict[float, set[str]] = defaultdict(set)
    for row in dropped_rows:
        try:
            dropped_by_delta[float(row["delta"])].add(str(row["object_id"]))
        except Exception:
            continue
    filtered_payload: dict[float, dict[str, dict[str, Any]]] = {}
    for d_arcmin in delta_arcmins:
        dropped_ids = dropped_by_delta.get(float(d_arcmin), set())
        filtered_payload[float(d_arcmin)] = {
            str(oid): meta
            for oid, meta in object_payload[float(d_arcmin)].items()
            if str(oid) not in dropped_ids
        }
    metric_rows = _read_csv_rows(root / "tangent_maps" / "twin_tangent_object_metrics.csv")
    summary_path = root / "twin_feature_tangent_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return delta_arcmins, filtered_payload, metric_rows, summary


def _select_object_ids_for_split_mode(
    split_mode: str,
    filtered_payload: dict[float, dict[str, dict[str, Any]]],
    metric_rows: list[dict[str, str]],
    primary_delta: float,
) -> list[str]:
    primary_payload = filtered_payload.get(float(primary_delta), {})
    object_ids = sorted(primary_payload.keys())
    if split_mode != "unique_image":
        return object_ids

    quality_by_object: dict[str, float] = {}
    for row in metric_rows:
        try:
            d = float(row.get("delta_arcmin_requested", "nan"))
        except Exception:
            continue
        if abs(d - float(primary_delta)) > 1e-9:
            continue
        oid = str(row.get("object_id", ""))
        try:
            quality_by_object[oid] = float(row.get("linear_local_r2", "nan"))
        except Exception:
            quality_by_object[oid] = float("nan")

    best_by_image: dict[int, tuple[tuple[object, ...], str]] = {}
    for oid in object_ids:
        meta = primary_payload[oid]
        image_id = int(meta["image_id"])
        trial_index = int(meta["trial_index"])
        time_index = int(meta["time_index"])
        q = quality_by_object.get(oid, float("nan"))
        if np.isfinite(q):
            key = (0, -float(q), trial_index, time_index, str(oid))
        else:
            key = (1, 0.0, trial_index, time_index, str(oid))
        existing = best_by_image.get(image_id)
        if existing is None or key < existing[0]:
            best_by_image[image_id] = (key, str(oid))
    return sorted(choice for _, choice in best_by_image.values())


def _apply_selected_object_ids(
    filtered_payload: dict[float, dict[str, dict[str, Any]]],
    selected_object_ids: list[str],
) -> dict[float, dict[str, dict[str, Any]]]:
    selected_set = set(str(oid) for oid in selected_object_ids)
    return {
        float(d): {str(oid): meta for oid, meta in payload.items() if str(oid) in selected_set}
        for d, payload in filtered_payload.items()
    }


def _build_fold_plans_for_split_mode(
    payload: dict[str, dict[str, Any]],
    split_mode: str,
    folds: int,
    seed: int,
    n_lags: int,
) -> list[dict[str, Any]]:
    object_ids = sorted(payload.keys())
    metas = [payload[oid] for oid in object_ids]
    if split_mode == "object_random":
        test_splits = _build_fold_splits(len(object_ids), folds=folds, seed=seed)
    else:
        test_splits = _build_group_fold_splits([int(meta["image_id"]) for meta in metas], folds=folds)

    plans: list[dict[str, Any]] = []
    for fold_id, test_idx0 in enumerate(test_splits):
        test_idx = np.asarray(test_idx0, dtype=np.int64)
        test_set = set(int(i) for i in test_idx.tolist())
        train_idx = np.asarray([i for i in range(len(object_ids)) if i not in test_set], dtype=np.int64)
        excluded_test_idx: list[int] = []
        status = "ok"

        if split_mode == "image_disjoint_history_gap":
            train_times_by_trial: dict[int, list[int]] = defaultdict(list)
            for idx in train_idx.tolist():
                meta = metas[int(idx)]
                train_times_by_trial[int(meta["trial_index"])].append(int(meta["time_index"]))
            kept_test_idx: list[int] = []
            for idx in test_idx.tolist():
                meta = metas[int(idx)]
                trial = int(meta["trial_index"])
                time_idx = int(meta["time_index"])
                if any(abs(time_idx - tr_time) < int(n_lags) for tr_time in train_times_by_trial.get(trial, [])):
                    excluded_test_idx.append(int(idx))
                else:
                    kept_test_idx.append(int(idx))
            test_idx = np.asarray(sorted(kept_test_idx), dtype=np.int64)

        if train_idx.size < 2 or test_idx.size < 1:
            status = "not_run_insufficient_support"

        plans.append(
            {
                "fold": int(fold_id),
                "train_idx": train_idx,
                "test_idx": test_idx,
                "excluded_test_idx": np.asarray(excluded_test_idx, dtype=np.int64),
                "status": status,
            }
        )
    return plans


def _build_split_mode_audits(
    payload_by_delta: dict[float, dict[str, dict[str, Any]]],
    delta_arcmins: list[float],
    fold_plans_by_delta: dict[float, list[dict[str, Any]]],
    split_mode: str,
    n_lags: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    object_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    for d_arcmin in delta_arcmins:
        payload = payload_by_delta.get(float(d_arcmin), {})
        object_ids = sorted(payload.keys())
        metas = [payload[oid] for oid in object_ids]
        image_counts = Counter(int(meta["image_id"]) for meta in metas)
        trial_counts = Counter(int(meta["trial_index"]) for meta in metas)
        time_ids = [int(meta["time_index"]) for meta in metas]
        object_rows.append(
            {
                "split_mode": split_mode,
                "delta": float(d_arcmin),
                "n_objects_total": int(len(object_ids)),
                "n_unique_image_ids": int(len(image_counts)),
                "objects_per_image_id_distribution": _distribution_string(image_counts),
                "n_image_ids_with_repeats": int(sum(1 for v in image_counts.values() if int(v) > 1)),
                "n_repeated_image_id_objects": int(len(object_ids) - len(image_counts)),
                "n_unique_trials": int(len(trial_counts)),
                "objects_per_trial_distribution": _distribution_string(trial_counts),
                "min_time_index": int(min(time_ids)) if time_ids else -1,
                "max_time_index": int(max(time_ids)) if time_ids else -1,
            }
        )

        plans = fold_plans_by_delta.get(float(d_arcmin), [])
        for plan in plans:
            train_idx = np.asarray(plan["train_idx"], dtype=np.int64)
            test_idx = np.asarray(plan["test_idx"], dtype=np.int64)
            train_metas = [metas[int(i)] for i in train_idx.tolist()]
            test_metas = [metas[int(i)] for i in test_idx.tolist()]
            train_image_set = {int(meta["image_id"]) for meta in train_metas}
            test_image_set = {int(meta["image_id"]) for meta in test_metas}
            test_images = [int(meta["image_id"]) for meta in test_metas]
            shared_image_ids = train_image_set & test_image_set
            fold_rows.append(
                {
                    "split_mode": split_mode,
                    "delta": float(d_arcmin),
                    "fold": int(plan["fold"]),
                    "fold_status": str(plan.get("status", "ok")),
                    "n_train_objects": int(len(train_metas)),
                    "n_test_objects": int(len(test_metas)),
                    "n_excluded_test_objects": int(len(np.asarray(plan.get("excluded_test_idx", []), dtype=np.int64))),
                    "n_train_unique_images": int(len(train_image_set)),
                    "n_test_unique_images": int(len(test_image_set)),
                    "n_shared_image_ids_train_test": int(len(shared_image_ids)),
                    "fraction_test_objects_with_image_seen_in_train": float(np.mean([img in train_image_set for img in test_images])) if test_images else float("nan"),
                }
            )

            train_times_by_trial: dict[int, list[int]] = defaultdict(list)
            for meta in train_metas:
                train_times_by_trial[int(meta["trial_index"])].append(int(meta["time_index"]))
            min_dists: list[float] = []
            overlap_fracs: list[float] = []
            for meta in test_metas:
                trial = int(meta["trial_index"])
                time_idx = int(meta["time_index"])
                train_times = train_times_by_trial.get(trial, [])
                if not train_times:
                    min_dists.append(float("inf"))
                    overlap_fracs.append(0.0)
                    continue
                abs_dists = [abs(time_idx - t_train) for t_train in train_times]
                min_dists.append(float(min(abs_dists)))
                overlap_fracs.append(max(_history_overlap_fraction(time_idx, t_train, n_lags=n_lags) for t_train in train_times))
            min_dists_arr = np.asarray(min_dists, dtype=np.float64)
            overlap_arr = np.asarray(overlap_fracs, dtype=np.float64)
            finite_min_dists = min_dists_arr[np.isfinite(min_dists_arr)]
            history_rows.append(
                {
                    "split_mode": split_mode,
                    "delta": float(d_arcmin),
                    "fold": int(plan["fold"]),
                    "fold_status": str(plan.get("status", "ok")),
                    "n_test_objects": int(len(test_metas)),
                    "n_test_objects_with_same_trial_in_train": int(np.sum(np.isfinite(min_dists_arr))),
                    "min_abs_time_distance_min": float(np.min(finite_min_dists)) if finite_min_dists.size else float("nan"),
                    "min_abs_time_distance_median": float(np.median(finite_min_dists)) if finite_min_dists.size else float("nan"),
                    "min_abs_time_distance_max": float(np.max(finite_min_dists)) if finite_min_dists.size else float("nan"),
                    "fraction_test_objects_with_train_within_32_frames": float(np.mean(min_dists_arr <= 32)) if min_dists_arr.size else float("nan"),
                    "fraction_test_objects_with_train_within_64_frames": float(np.mean(min_dists_arr <= 64)) if min_dists_arr.size else float("nan"),
                    "fraction_test_objects_with_history_overlap_gt_0": float(np.mean(overlap_arr > 0.0)) if overlap_arr.size else float("nan"),
                    "estimated_history_overlap_fraction_mean": float(np.mean(overlap_arr)) if overlap_arr.size else float("nan"),
                    "estimated_history_overlap_fraction_median": float(np.median(overlap_arr)) if overlap_arr.size else float("nan"),
                    "estimated_history_overlap_fraction_max": float(np.max(overlap_arr)) if overlap_arr.size else float("nan"),
                }
            )
    return object_rows, fold_rows, history_rows


def _compute_union_summary_cached(
    payload_by_delta: dict[float, dict[str, dict[str, Any]]],
    delta_arcmins: list[float],
    null_repeats: int,
    seed: int,
    min_objects_support: int,
    split_mode: str,
) -> list[dict[str, object]]:
    union_summary: list[dict[str, object]] = []
    for d_arcmin in delta_arcmins:
        payload = payload_by_delta.get(float(d_arcmin), {})
        object_ids = sorted(payload.keys())
        if len(object_ids) < int(min_objects_support):
            union_summary.append(
                {
                    "split_mode": split_mode,
                    "delta": float(d_arcmin),
                    "space": "raw",
                    "component_set": "combined",
                    "n_objects": int(len(object_ids)),
                    "rank": 0,
                    "participation_ratio": float("nan"),
                    "null_pr_mean": float("nan"),
                    "null_pr_ci_low": float("nan"),
                    "null_pr_ci_high": float("nan"),
                    "compactness_label": "not_run_insufficient_support",
                    "status": "not_run_insufficient_support",
                }
            )
            continue
        bx = np.stack([np.asarray(payload[oid]["bx"], dtype=np.float64) for oid in object_ids], axis=1)
        by = np.stack([np.asarray(payload[oid]["by"], dtype=np.float64) for oid in object_ids], axis=1)
        mat = np.concatenate([bx, by], axis=1)
        evals, _ = _eigh_desc(mat @ mat.T)
        evals = np.maximum(evals, 0.0)
        total_eval = float(np.sum(evals))
        if total_eval <= 1e-12:
            union_summary.append(
                {
                    "split_mode": split_mode,
                    "delta": float(d_arcmin),
                    "space": "raw",
                    "component_set": "combined",
                    "n_objects": int(len(object_ids)),
                    "rank": 0,
                    "participation_ratio": float("nan"),
                    "null_pr_mean": float("nan"),
                    "null_pr_ci_low": float("nan"),
                    "null_pr_ci_high": float("nan"),
                    "compactness_label": "not_run_no_finite_tangents",
                    "status": "not_run_no_finite_tangents",
                }
            )
            continue
        rank = _numerical_rank(evals)
        pr = _participation_ratio(evals)
        null_pr = []
        for nrep in range(int(null_repeats)):
            nrng = np.random.default_rng(int(seed) + 10000 + nrep)
            m_shuf = np.stack([col[nrng.permutation(col.shape[0])] for col in mat.T], axis=1)
            es, _ = _eigh_desc(m_shuf @ m_shuf.T)
            null_pr.append(_participation_ratio(np.maximum(es, 0.0)))
        null_pr_arr = np.asarray(null_pr, dtype=np.float64)
        union_summary.append(
            {
                "split_mode": split_mode,
                "delta": float(d_arcmin),
                "space": "raw",
                "component_set": "combined",
                "n_objects": int(len(object_ids)),
                "rank": int(rank),
                "participation_ratio": float(pr),
                "null_pr_mean": float(np.mean(null_pr_arr)) if null_pr_arr.size else float("nan"),
                "null_pr_ci_low": float(np.percentile(null_pr_arr, 2.5)) if null_pr_arr.size else float("nan"),
                "null_pr_ci_high": float(np.percentile(null_pr_arr, 97.5)) if null_pr_arr.size else float("nan"),
                "compactness_label": "compact" if null_pr_arr.size and np.isfinite(pr) and pr < np.percentile(null_pr_arr, 2.5) else "not_compact",
                "status": "ok",
            }
        )
    return union_summary


def _compute_basis_rows_cached(
    payload_by_delta: dict[float, dict[str, dict[str, Any]]],
    delta_arcmins: list[float],
    fold_plans_by_delta: dict[float, list[dict[str, Any]]],
    null_repeats: int,
    seed: int,
    min_objects_support: int,
    split_mode: str,
) -> list[dict[str, object]]:
    basis_rows: list[dict[str, object]] = []
    k_grid = [2, 5, 10, 20]
    for d_arcmin in delta_arcmins:
        payload = payload_by_delta.get(float(d_arcmin), {})
        object_ids = sorted(payload.keys())
        if len(object_ids) < int(min_objects_support):
            basis_rows.append(
                {
                    "split_mode": split_mode,
                    "delta": float(d_arcmin),
                    "fold": -1,
                    "tangent_set": "combined",
                    "basis_rank_k": -1,
                    "fold_status": "not_run_insufficient_support",
                    "test_variance_captured": float("nan"),
                    "null_mean": float("nan"),
                    "effect_minus_null": float("nan"),
                }
            )
            continue
        bx = np.stack([np.asarray(payload[oid]["bx"], dtype=np.float64) for oid in object_ids], axis=1)
        by = np.stack([np.asarray(payload[oid]["by"], dtype=np.float64) for oid in object_ids], axis=1)
        plans = fold_plans_by_delta.get(float(d_arcmin), [])
        for plan in plans:
            train_idx = np.asarray(plan["train_idx"], dtype=np.int64)
            test_idx = np.asarray(plan["test_idx"], dtype=np.int64)
            if str(plan.get("status", "ok")) != "ok" or train_idx.size < 2 or test_idx.size < 1:
                basis_rows.append(
                    {
                        "split_mode": split_mode,
                        "delta": float(d_arcmin),
                        "fold": int(plan["fold"]),
                        "train_n_objects": int(train_idx.size),
                        "test_n_objects": int(test_idx.size),
                        "tangent_set": "combined",
                        "basis_rank_k": -1,
                        "rank_train": 0,
                        "rank_test": 0,
                        "test_variance_captured": float("nan"),
                        "null_mean": float("nan"),
                        "null_ci_low": float("nan"),
                        "null_ci_high": float("nan"),
                        "effect_minus_null": float("nan"),
                        "fold_status": "not_run_insufficient_support",
                    }
                )
                continue
            m_train = np.concatenate([bx[:, train_idx], by[:, train_idx]], axis=1)
            m_test = np.concatenate([bx[:, test_idx], by[:, test_idx]], axis=1)
            evals_train, u_train = _eigh_desc(m_train @ m_train.T)
            evals_test, _ = _eigh_desc(m_test @ m_test.T)
            rank_train = _numerical_rank(np.maximum(evals_train, 0.0))
            rank_test = _numerical_rank(np.maximum(evals_test, 0.0))
            for k in k_grid:
                if int(k) > int(rank_train):
                    basis_rows.append(
                        {
                            "split_mode": split_mode,
                            "delta": float(d_arcmin),
                            "fold": int(plan["fold"]),
                            "train_n_objects": int(train_idx.size),
                            "test_n_objects": int(test_idx.size),
                            "tangent_set": "combined",
                            "basis_rank_k": int(k),
                            "rank_train": int(rank_train),
                            "rank_test": int(rank_test),
                            "test_variance_captured": float("nan"),
                            "null_mean": float("nan"),
                            "null_ci_low": float("nan"),
                            "null_ci_high": float("nan"),
                            "effect_minus_null": float("nan"),
                            "fold_status": "ok_rank_exceeded",
                        }
                    )
                    continue
                cap = _variance_capture(u_train, m_test, k=int(k))
                null_vals = []
                for nrep in range(int(null_repeats)):
                    nrng = np.random.default_rng(int(seed) + 30000 + int(plan["fold"]) * 100 + nrep)
                    m_train_shuf = np.stack([col[nrng.permutation(col.shape[0])] for col in m_train.T], axis=1)
                    _, u_shuf = _eigh_desc(m_train_shuf @ m_train_shuf.T)
                    null_vals.append(_variance_capture(u_shuf, m_test, k=int(k)))
                n0 = np.asarray(null_vals, dtype=np.float64)
                basis_rows.append(
                    {
                        "split_mode": split_mode,
                        "delta": float(d_arcmin),
                        "fold": int(plan["fold"]),
                        "train_n_objects": int(train_idx.size),
                        "test_n_objects": int(test_idx.size),
                        "tangent_set": "combined",
                        "basis_rank_k": int(k),
                        "rank_train": int(rank_train),
                        "rank_test": int(rank_test),
                        "test_variance_captured": float(cap),
                        "null_mean": float(np.mean(n0)) if n0.size else float("nan"),
                        "null_ci_low": float(np.percentile(n0, 2.5)) if n0.size else float("nan"),
                        "null_ci_high": float(np.percentile(n0, 97.5)) if n0.size else float("nan"),
                        "effect_minus_null": float(cap - np.mean(n0)) if n0.size else float("nan"),
                        "fold_status": "ok",
                    }
                )
    return basis_rows


def _build_split_mode_summary(
    split_mode: str,
    object_rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
    history_rows: list[dict[str, object]],
    union_rows: list[dict[str, object]],
    basis_rows: list[dict[str, object]],
    k_grid: list[int],
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    deltas = sorted({float(cast(Any, r.get("delta", float("nan")))) for r in object_rows})
    for d_arcmin in deltas:
        obj_row = next((r for r in object_rows if float(cast(Any, r.get("delta", float("nan")))) == d_arcmin), None)
        union_row = next((r for r in union_rows if float(cast(Any, r.get("delta", float("nan")))) == d_arcmin), None)
        fold_block = [r for r in fold_rows if float(cast(Any, r.get("delta", float("nan")))) == d_arcmin]
        hist_block = [r for r in history_rows if float(cast(Any, r.get("delta", float("nan")))) == d_arcmin]
        row: dict[str, object] = {
            "split_mode": split_mode,
            "delta": float(d_arcmin),
            "n_objects": int(cast(Any, obj_row.get("n_objects_total", 0))) if obj_row else 0,
            "n_unique_image_ids": int(cast(Any, obj_row.get("n_unique_image_ids", 0))) if obj_row else 0,
            "objects_per_image_distribution": str(obj_row.get("objects_per_image_id_distribution", "")) if obj_row else "",
            "n_folds": int(len(fold_block)),
            "fraction_test_objects_with_image_seen_in_train": _finite_median([float(cast(Any, r.get("fraction_test_objects_with_image_seen_in_train", float("nan")))) for r in fold_block]),
            "fraction_test_objects_with_history_overlap_gt_0": _finite_median([float(cast(Any, r.get("fraction_test_objects_with_history_overlap_gt_0", float("nan")))) for r in hist_block]),
            "union_participation_ratio": float(cast(Any, union_row.get("participation_ratio", float("nan")))) if union_row else float("nan"),
            "union_null_pr_mean": float(cast(Any, union_row.get("null_pr_mean", float("nan")))) if union_row else float("nan"),
            "union_null_pr_ci_low": float(cast(Any, union_row.get("null_pr_ci_low", float("nan")))) if union_row else float("nan"),
            "union_null_pr_ci_high": float(cast(Any, union_row.get("null_pr_ci_high", float("nan")))) if union_row else float("nan"),
            "union_status": str(union_row.get("status", "")) if union_row else "missing",
        }
        for k in k_grid:
            basis_block = [
                r for r in basis_rows
                if float(cast(Any, r.get("delta", float("nan")))) == d_arcmin
                and int(cast(Any, r.get("basis_rank_k", -1))) == int(k)
                and str(r.get("fold_status", "")) == "ok"
            ]
            row[f"k{k}_capture_median"] = _finite_median([float(cast(Any, r.get("test_variance_captured", float("nan")))) for r in basis_block])
            row[f"k{k}_null_median"] = _finite_median([float(cast(Any, r.get("null_mean", float("nan")))) for r in basis_block])
            row[f"k{k}_effect_median"] = _finite_median([float(cast(Any, r.get("effect_minus_null", float("nan")))) for r in basis_block])
        summary_rows.append(row)
    return summary_rows


def _run_cached_split_mode_analyses(args: argparse.Namespace) -> None:
    root = Path(args.reuse_tangents_root)
    delta_arcmins, filtered_payload, metric_rows, cached_summary = _load_cached_filtered_tangents(root)
    folds = int(args.folds) if args.folds is not None else int(cached_summary.get("folds", 5))
    null_repeats = int(args.null_repeats) if args.null_repeats is not None else int(cached_summary.get("null_repeats", 200))
    split_modes = [s.strip() for s in str(args.split_modes).split(",") if s.strip()]
    primary_delta = _resolve_primary_delta(delta_arcmins, args.primary_delta)
    out_base = root / "split_modes"
    k_grid = [2, 5, 10, 20]

    printed_rows: list[dict[str, object]] = []
    for split_mode in split_modes:
        selected_ids = _select_object_ids_for_split_mode(split_mode, filtered_payload, metric_rows, primary_delta=primary_delta)
        split_payload = _apply_selected_object_ids(filtered_payload, selected_ids)
        fold_plans_by_delta = {
            float(d): _build_fold_plans_for_split_mode(
                payload=split_payload.get(float(d), {}),
                split_mode=split_mode,
                folds=folds,
                seed=int(args.seed),
                n_lags=int(args.history_gap_frames),
            )
            for d in delta_arcmins
        }
        object_rows, fold_rows, history_rows = _build_split_mode_audits(
            payload_by_delta=split_payload,
            delta_arcmins=delta_arcmins,
            fold_plans_by_delta=fold_plans_by_delta,
            split_mode=split_mode,
            n_lags=int(args.history_gap_frames),
        )
        union_rows = _compute_union_summary_cached(
            payload_by_delta=split_payload,
            delta_arcmins=delta_arcmins,
            null_repeats=null_repeats,
            seed=int(args.seed),
            min_objects_support=4,
            split_mode=split_mode,
        )
        basis_rows = _compute_basis_rows_cached(
            payload_by_delta=split_payload,
            delta_arcmins=delta_arcmins,
            fold_plans_by_delta=fold_plans_by_delta,
            null_repeats=null_repeats,
            seed=int(args.seed),
            min_objects_support=4,
            split_mode=split_mode,
        )
        summary_rows = _build_split_mode_summary(
            split_mode=split_mode,
            object_rows=object_rows,
            fold_rows=fold_rows,
            history_rows=history_rows,
            union_rows=union_rows,
            basis_rows=basis_rows,
            k_grid=k_grid,
        )
        out_dir = out_base / split_mode
        _write_csv(out_dir / "object_composition_audit.csv", object_rows)
        _write_csv(out_dir / "fold_leakage_audit.csv", fold_rows)
        _write_csv(out_dir / "history_overlap_audit.csv", history_rows)
        _write_csv(out_dir / f"twin_tangent_union_summary_{split_mode}.csv", union_rows)
        _write_csv(out_dir / f"twin_tangent_train_test_basis_{split_mode}.csv", basis_rows)
        _write_csv(out_dir / "split_mode_summary.csv", summary_rows)
        printed_rows.extend(summary_rows)

    print(json.dumps({"reuse_tangents_root": str(root), "primary_delta": primary_delta, "split_modes": split_modes, "summary_rows": printed_rows}, indent=2))


def _build_leakage_audits(
    filtered_object_payload: dict[float, dict[str, dict[str, Any]]],
    delta_arcmins: list[float],
    folds: int,
    seed: int,
    n_lags: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    object_composition_rows: list[dict[str, object]] = []
    fold_leakage_rows: list[dict[str, object]] = []
    history_overlap_rows: list[dict[str, object]] = []

    for d_arcmin in delta_arcmins:
        payload = filtered_object_payload.get(float(d_arcmin), {})
        object_ids = sorted(payload.keys())
        metas = [payload[oid] for oid in object_ids]
        image_ids = [int(meta["image_id"]) for meta in metas]
        trial_ids = [int(meta["trial_index"]) for meta in metas]
        time_ids = [int(meta["time_index"]) for meta in metas]
        image_counts = Counter(image_ids)
        trial_counts = Counter(trial_ids)

        object_composition_rows.append(
            {
                "delta": float(d_arcmin),
                "n_objects_total": int(len(object_ids)),
                "n_unique_image_ids": int(len(image_counts)),
                "objects_per_image_id_distribution": _distribution_string(image_counts),
                "n_image_ids_with_repeats": int(sum(1 for v in image_counts.values() if int(v) > 1)),
                "n_repeated_image_id_objects": int(len(object_ids) - len(image_counts)),
                "n_unique_trials": int(len(trial_counts)),
                "objects_per_trial_distribution": _distribution_string(trial_counts),
                "min_time_index": int(min(time_ids)) if time_ids else -1,
                "max_time_index": int(max(time_ids)) if time_ids else -1,
            }
        )

        fold_splits = _build_fold_splits(len(object_ids), folds=folds, seed=seed)
        for fold_id, test_idx in enumerate(fold_splits):
            test_set = set(int(i) for i in test_idx.tolist())
            train_idx = np.asarray([i for i in range(len(object_ids)) if i not in test_set], dtype=np.int64)

            train_metas = [metas[int(i)] for i in train_idx.tolist()]
            test_metas = [metas[int(i)] for i in test_idx.tolist()]
            train_images = [int(meta["image_id"]) for meta in train_metas]
            test_images = [int(meta["image_id"]) for meta in test_metas]
            train_image_set = set(train_images)
            test_image_set = set(test_images)
            shared_image_ids = train_image_set & test_image_set

            fold_leakage_rows.append(
                {
                    "delta": float(d_arcmin),
                    "fold": int(fold_id),
                    "n_train_objects": int(len(train_metas)),
                    "n_test_objects": int(len(test_metas)),
                    "n_train_unique_images": int(len(train_image_set)),
                    "n_test_unique_images": int(len(test_image_set)),
                    "n_shared_image_ids_train_test": int(len(shared_image_ids)),
                    "fraction_test_objects_with_image_seen_in_train": float(np.mean([img in train_image_set for img in test_images])) if test_images else float("nan"),
                }
            )

            train_times_by_trial: dict[int, list[int]] = defaultdict(list)
            for meta in train_metas:
                train_times_by_trial[int(meta["trial_index"])].append(int(meta["time_index"]))

            min_dists: list[float] = []
            overlap_fracs: list[float] = []
            for meta in test_metas:
                trial = int(meta["trial_index"])
                time_idx = int(meta["time_index"])
                train_times = train_times_by_trial.get(trial, [])
                if not train_times:
                    min_dists.append(float("inf"))
                    overlap_fracs.append(0.0)
                    continue
                abs_dists = [abs(time_idx - t_train) for t_train in train_times]
                min_dist = min(abs_dists)
                min_dists.append(float(min_dist))
                overlap_fracs.append(max(_history_overlap_fraction(time_idx, t_train, n_lags=n_lags) for t_train in train_times))

            min_dists_arr = np.asarray(min_dists, dtype=np.float64)
            finite_min_dists = min_dists_arr[np.isfinite(min_dists_arr)]
            overlap_arr = np.asarray(overlap_fracs, dtype=np.float64)
            history_overlap_rows.append(
                {
                    "delta": float(d_arcmin),
                    "fold": int(fold_id),
                    "n_test_objects": int(len(test_metas)),
                    "n_test_objects_with_same_trial_in_train": int(np.sum(np.isfinite(min_dists_arr))),
                    "min_abs_time_distance_min": float(np.min(finite_min_dists)) if finite_min_dists.size else float("nan"),
                    "min_abs_time_distance_median": float(np.median(finite_min_dists)) if finite_min_dists.size else float("nan"),
                    "min_abs_time_distance_max": float(np.max(finite_min_dists)) if finite_min_dists.size else float("nan"),
                    "fraction_test_objects_with_train_within_32_frames": float(np.mean(min_dists_arr <= 32)) if min_dists_arr.size else float("nan"),
                    "fraction_test_objects_with_train_within_64_frames": float(np.mean(min_dists_arr <= 64)) if min_dists_arr.size else float("nan"),
                    "estimated_history_overlap_fraction_mean": float(np.mean(overlap_arr)) if overlap_arr.size else float("nan"),
                    "estimated_history_overlap_fraction_median": float(np.median(overlap_arr)) if overlap_arr.size else float("nan"),
                    "estimated_history_overlap_fraction_max": float(np.max(overlap_arr)) if overlap_arr.size else float("nan"),
                }
            )

    return object_composition_rows, fold_leakage_rows, history_overlap_rows




def _frame_to_hw(frame: np.ndarray) -> np.ndarray:
    f = np.asarray(frame, dtype=np.float32)
    if f.ndim == 2:
        return f
    if f.ndim == 3 and f.shape[0] == 1:
        return f[0]
    if f.ndim == 3 and f.shape[-1] == 1:
        return f[..., 0]
    raise ValueError(f"Unsupported frame shape: {f.shape}")


def _movie_to_thw(movie: np.ndarray | torch.Tensor) -> torch.Tensor:
    m = torch.as_tensor(movie, dtype=torch.float32)
    if m.ndim == 3:
        return m
    if m.ndim == 4 and m.shape[1] == 1:
        return m[:, 0]
    if m.ndim == 4 and m.shape[-1] == 1:
        return m[..., 0]
    raise ValueError(f"Unsupported movie shape: {tuple(m.shape)}")


def _history_from_stim(stim_trial: np.ndarray, t_idx: int, n_lags: int) -> np.ndarray:
    return np.asarray([_frame_to_hw(stim_trial[t_idx - lag]) for lag in range(n_lags)], dtype=np.float32)


def _frame_stats(frame: np.ndarray) -> dict[str, float]:
    f = np.asarray(_frame_to_hw(frame), dtype=np.float64)
    gx = np.diff(f, axis=1, prepend=f[:, :1])
    gy = np.diff(f, axis=0, prepend=f[:1, :])
    gmag = np.sqrt(gx * gx + gy * gy)
    gxx = float(np.mean(gx * gx))
    gyy = float(np.mean(gy * gy))
    anis = float(abs(gxx - gyy) / (gxx + gyy + 1e-12))
    return {
        "min": float(np.min(f)),
        "max": float(np.max(f)),
        "mean": float(np.mean(f)),
        "std": float(np.std(f)),
        "rms_contrast": float(np.std(f)),
        "gradient_rms": float(np.sqrt(np.mean(gmag * gmag))),
        "gradient_anisotropy": anis,
    }


def _history_stats(history: np.ndarray) -> dict[str, float]:
    h = np.asarray(history, dtype=np.float64)
    return {
        "min": float(np.min(h)),
        "max": float(np.max(h)),
        "mean": float(np.mean(h)),
        "std": float(np.std(h)),
    }


def _save_history_png(path: Path, history: np.ndarray) -> None:
    frame = np.asarray(history)[len(history) // 2]
    if frame.ndim == 3 and frame.shape[0] == 1:
        frame = frame[0]
    elif frame.ndim != 2:
        raise ValueError(f"Unexpected frame shape for PNG export: {frame.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, frame, cmap="gray")


def _harmonize_fixrsvp_arrays(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stim = np.asarray(data["stim"], dtype=np.float32)
    image_ids = np.asarray(data["image_ids"], dtype=np.int64)
    eyepos = np.asarray(data["eyepos"], dtype=np.float64)
    nt = min(int(stim.shape[0]), int(image_ids.shape[0]), int(eyepos.shape[0]))
    tt = min(int(stim.shape[1]), int(image_ids.shape[1]), int(eyepos.shape[1]))
    return stim[:nt, :tt], image_ids[:nt, :tt], eyepos[:nt, :tt]


def _participation_ratio(vals: np.ndarray) -> float:
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v) & (v >= 0)]
    if v.size == 0:
        return float("nan")
    den = float(np.sum(v * v))
    if den <= 0:
        return float("nan")
    num = float(np.sum(v))
    return float((num * num) / den)


def _numerical_rank(evals: np.ndarray, rel_tol: float = 1e-8, abs_tol: float = 1e-12) -> int:
    e = np.asarray(evals, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return 0
    thresh = max(abs_tol, rel_tol * float(np.max(np.abs(e))))
    return int(np.sum(e > thresh))


def _ranks_at_thresholds(frac_var: np.ndarray) -> dict[str, int]:
    c = np.cumsum(np.asarray(frac_var, dtype=np.float64))
    out = {}
    for q in (0.5, 0.75, 0.8, 0.9, 0.95):
        out[f"rank_{int(q * 100)}"] = int(np.searchsorted(c, q, side="left")) + 1 if c.size else 0
    return out


def _safe_svdvals(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float64)
    try:
        return np.linalg.svd(m, compute_uv=False)
    except np.linalg.LinAlgError:
        # Retry after cleaning non-finite values; fall back to zeros if still unstable.
        m2 = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            return np.linalg.svd(m2, compute_uv=False)
        except np.linalg.LinAlgError:
            return np.zeros((min(m2.shape[0], m2.shape[1]),), dtype=np.float64)


def _subspace_overlap(u_a: np.ndarray, u_b: np.ndarray, k: int) -> float:
    kk = min(int(k), u_a.shape[1], u_b.shape[1])
    if kk <= 0:
        return float("nan")
    svals = _safe_svdvals(u_a[:, :kk].T @ u_b[:, :kk])
    return float(np.mean(svals * svals)) if svals.size else float("nan")


def _covariance(x: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2 or xx.shape[0] < 2:
        return np.zeros((xx.shape[1], xx.shape[1]), dtype=np.float64)
    xc = xx - np.mean(xx, axis=0, keepdims=True)
    return (xc.T @ xc) / max(xx.shape[0] - 1, 1)


def _eigh_desc(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    evals, vecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    order = np.argsort(evals)[::-1]
    return evals[order], vecs[:, order]


def _variance_capture(u_train: np.ndarray, b_test: np.ndarray, k: int) -> float:
    kk = min(int(k), u_train.shape[1])
    if kk <= 0:
        return float("nan")
    u = u_train[:, :kk]
    proj = u @ (u.T @ b_test)
    den = float(np.sum(b_test * b_test))
    if den <= 0:
        return float("nan")
    return float(np.sum(proj * proj) / den)


def _shift_movie_subpixel(movie: torch.Tensor, dx_px: float, dy_px: float) -> torch.Tensor:
    t, h, w = movie.shape
    x = movie.unsqueeze(1)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=movie.device, dtype=movie.dtype),
        torch.linspace(-1.0, 1.0, w, device=movie.device, dtype=movie.dtype),
        indexing="ij",
    )
    gx = xx[None, :, :].expand(t, h, w).clone()
    gy = yy[None, :, :].expand(t, h, w).clone()
    gx = gx - (2.0 * dx_px / max(w - 1, 1))
    gy = gy - (2.0 * dy_px / max(h - 1, 1))
    grid = torch.stack([gx, gy], dim=-1)
    y = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return y[:, 0]


@dataclass
class ObjectCandidate:
    object_id: str
    image_id: int
    trial_index: int
    time_index: int
    frame_stats: dict[str, float]


@dataclass
class TwinContext:
    model: Any
    readout: torch.nn.Module
    compute_rate_map_fn: Any
    n_units: int
    n_lags: int
    manifest_rows: list[dict[str, object]]


def _load_twin_context(model_device: str) -> TwinContext:
    scripts_dir = VISIONCORE_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import dill
    from scripts.spatial_info import compute_rate_map, get_spatial_readout
    from scripts.utils import get_model_and_dataset_configs

    model, _ = get_model_and_dataset_configs(mode="standard")
    model = model.to(str(model_device))
    model.model.eval()

    outputs_path = VISIONCORE_ROOT / "scripts" / "mcfarland_outputs_mono.pkl"
    with outputs_path.open("rb") as handle:
        outputs = dill.load(handle)

    sessions = [outputs[i]["sess"] for i in range(len(outputs))]
    model_dataset_idx = [i for i, name in enumerate(model.names) if name in sessions]
    cids2use = [np.where(outputs[sessions.index(model.names[i])]["ccnorm"]["ccnorm"] > 0.5)[0] for i in model_dataset_idx]
    readout = get_spatial_readout(model, outputs).to(str(model_device)).eval()

    manifest_rows: list[dict[str, object]] = []
    canonical_idx = 0
    for ridx, didx in enumerate(model_dataset_idx):
        sess = str(model.names[didx])
        for cid in cids2use[ridx]:
            manifest_rows.append(
                {
                    "canonical_unit_index": int(canonical_idx),
                    "readout_name": "canonical_shared_population_readout",
                    "dataset_idx": int(didx),
                    "readout_idx": int(didx),
                    "session_name": sess,
                    "unit_id": int(cid),
                    "selection_rule": "mcfarland_shared_readout_ccnorm_gt_0.5",
                    "n_units": int(readout.n_units),
                }
            )
            canonical_idx += 1

    return TwinContext(
        model=model,
        readout=readout,
        compute_rate_map_fn=compute_rate_map,
        n_units=int(readout.n_units),
        n_lags=int(DEFAULT_N_LAGS),
        manifest_rows=manifest_rows,
    )


def _predict_rate_from_history(ctx: TwinContext, history: np.ndarray, model_device: str) -> np.ndarray:
    x = _movie_to_thw(history).to(str(model_device)).unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        feats = ctx.model.model.core_forward(x, None)
        y = ctx.readout(feats[:, :, -1])
        rates_spatial = ctx.model.model.activation(y)
        rates = rates_spatial.amax(dim=(-2, -1))[0]
    return rates.detach().cpu().numpy().astype(np.float64, copy=False)


def _canonical_rate_from_history(ctx: TwinContext, history: np.ndarray, model_device: str) -> np.ndarray:
    x = _movie_to_thw(history).to(str(model_device)).unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        rates_map = ctx.compute_rate_map_fn(ctx.model, ctx.readout, x)
        rates = rates_map.amax(dim=(-2, -1))[0]
    return rates.detach().cpu().numpy().astype(np.float64, copy=False)


def _fit_tangent_for_history(
    ctx: TwinContext,
    history: np.ndarray,
    delta_px: float,
    model_device: str,
) -> dict[str, np.ndarray]:
    """Return base response and translated endpoint rates for one history object.

    Keys: ``r0``, ``bx``, ``by``, ``rx_p``, ``rx_m``, ``ry_p``, ``ry_m``.
    Storing the raw endpoints (``rx_p`` etc.) alongside the linearized tangent
    vectors (``bx``, ``by``) allows Panel B to draw either finite-translation
    patches or linearized arrows from ``r0``.
    """
    h = _movie_to_thw(history).to(str(model_device))
    hx_p = _shift_movie_subpixel(h, dx_px=float(delta_px), dy_px=0.0).detach().cpu().numpy()
    hx_m = _shift_movie_subpixel(h, dx_px=-float(delta_px), dy_px=0.0).detach().cpu().numpy()
    hy_p = _shift_movie_subpixel(h, dx_px=0.0, dy_px=float(delta_px)).detach().cpu().numpy()
    hy_m = _shift_movie_subpixel(h, dx_px=0.0, dy_px=-float(delta_px)).detach().cpu().numpy()
    r0   = _predict_rate_from_history(ctx, history, model_device=model_device)
    rx_p = _predict_rate_from_history(ctx, hx_p,    model_device=model_device)
    rx_m = _predict_rate_from_history(ctx, hx_m,    model_device=model_device)
    ry_p = _predict_rate_from_history(ctx, hy_p,    model_device=model_device)
    ry_m = _predict_rate_from_history(ctx, hy_m,    model_device=model_device)
    bx = (rx_p - rx_m) / (2.0 * float(delta_px))
    by = (ry_p - ry_m) / (2.0 * float(delta_px))
    return {"r0": r0, "bx": bx, "by": by, "rx_p": rx_p, "rx_m": rx_m, "ry_p": ry_p, "ry_m": ry_m}


def _local_linear_r2(
    ctx: TwinContext,
    history: np.ndarray,
    r0: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    delta_px: float,
    model_device: str,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(int(seed))
    offsets: list[tuple[float, float]] = []
    for s in (0.5, 1.0):
        d = float(delta_px) * s
        offsets.extend([(d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d), (d, d), (d, -d), (-d, d), (-d, -d)])
    for _ in range(8):
        off = rng.normal(loc=0.0, scale=max(float(delta_px) * 0.5, 1e-4), size=2)
        offsets.append((float(off[0]), float(off[1])))

    true_all: list[np.ndarray] = []
    pred_all: list[np.ndarray] = []
    true_x: list[np.ndarray] = []
    pred_x: list[np.ndarray] = []
    true_y: list[np.ndarray] = []
    pred_y: list[np.ndarray] = []

    h = _movie_to_thw(history).to(str(model_device))
    for dx, dy in offsets:
        hs = _shift_movie_subpixel(h, dx_px=dx, dy_px=dy).detach().cpu().numpy()
        rs = _predict_rate_from_history(ctx, hs, model_device=model_device)
        dr_true = rs - r0
        dr_pred = bx * dx + by * dy
        true_all.append(dr_true)
        pred_all.append(dr_pred)
        if abs(dx) < 1e-12:
            true_y.append(dr_true)
            pred_y.append(dr_pred)
        if abs(dy) < 1e-12:
            true_x.append(dr_true)
            pred_x.append(dr_pred)

    def _r2(true_list: list[np.ndarray], pred_list: list[np.ndarray]) -> float:
        yt = np.concatenate([v.ravel() for v in true_list])
        yp = np.concatenate([v.ravel() for v in pred_list])
        keep = np.isfinite(yt) & np.isfinite(yp)
        if int(np.sum(keep)) < 10:
            return float("nan")
        yt = yt[keep]
        yp = yp[keep]
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        if ss_tot <= 1e-12:
            return float("nan")
        ss_res = float(np.sum((yt - yp) ** 2))
        return float(1.0 - ss_res / ss_tot)

    return _r2(true_all, pred_all), _r2(true_x, pred_x), _r2(true_y, pred_y)


def _select_objects(candidates: list[ObjectCandidate], n_objects: int, seed: int, stratified: bool) -> list[ObjectCandidate]:
    rng = np.random.default_rng(int(seed))
    if len(candidates) <= n_objects:
        return list(candidates)
    if not stratified:
        picks = rng.choice(np.arange(len(candidates)), size=n_objects, replace=False)
        return [candidates[int(i)] for i in sorted(picks.tolist())]

    rows = [c.frame_stats for c in candidates]
    contrast = np.asarray([r["rms_contrast"] for r in rows], dtype=np.float64)
    grad = np.asarray([r["gradient_rms"] for r in rows], dtype=np.float64)
    anis = np.asarray([r["gradient_anisotropy"] for r in rows], dtype=np.float64)
    bins = lambda x: np.digitize(x, np.quantile(x, [1 / 3, 2 / 3]), right=True)
    cbin, gbin, abin = bins(contrast), bins(grad), bins(anis)
    strata: dict[tuple[int, int, int], list[int]] = {}
    for idx in range(len(candidates)):
        strata.setdefault((int(cbin[idx]), int(gbin[idx]), int(abin[idx])), []).append(idx)

    chosen: list[ObjectCandidate] = []
    while len(chosen) < n_objects:
        progressed = False
        for key in sorted(strata.keys()):
            pool = strata[key]
            if not pool:
                continue
            pick_idx = int(rng.integers(0, len(pool)))
            chosen.append(candidates[pool.pop(pick_idx)])
            progressed = True
            if len(chosen) >= n_objects:
                break
        if not progressed:
            break
    if len(chosen) < n_objects:
        remaining = [c for c in candidates if c not in chosen]
        if remaining:
            extra = rng.choice(np.arange(len(remaining)), size=min(n_objects - len(chosen), len(remaining)), replace=False)
            chosen.extend(remaining[int(i)] for i in extra.tolist())
    return chosen[:n_objects]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Twin Feature-Tangent Structure (TFTS)")
    p.add_argument("--subject", type=str, default="Allen")
    p.add_argument("--date", type=str, default="2022-02-16")
    p.add_argument("--dataset-configs-path", type=str, default="experiments/dataset_configs/multi_basic_240_rsvp.yaml")
    p.add_argument("--output-root", type=Path, default=VISIONCORE_ROOT / "outputs" / "twin_feature_tangent_structure")
    p.add_argument("--mode", type=str, choices=("smoke", "production"), default="smoke")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-device", type=str, default="cuda")
    p.add_argument("--use-cached-data", action="store_true", default=True)
    p.add_argument("--n-images", type=int, default=None)
    p.add_argument("--analysis5-n-images", type=int, default=None)
    p.add_argument("--n-objects-per-image", type=int, default=8)
    p.add_argument("--delta-arcmin", type=str, default=None)
    p.add_argument("--cloud-scales", type=str, default=None)
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--null-repeats", type=int, default=None)
    p.add_argument("--force-secondary", action="store_true", default=False)
    p.add_argument("--stratified-sampling", action="store_true", default=False)
    p.add_argument("--reuse-tangents-root", type=Path, default=None)
    p.add_argument(
        "--split-modes",
        type=str,
        default="object_random",
        help="Comma-separated split modes for cached tangent re-analysis.",
    )
    p.add_argument("--primary-delta", type=float, default=None)
    p.add_argument("--history-gap-frames", type=int, default=32)
    p.add_argument("--model-ppd", type=float, default=float(DEFAULT_MODEL_PPD))
    p.add_argument("--max-eye-samples", type=int, default=128)
    p.add_argument(
        "--eye-cloud-mode",
        type=str,
        choices=("synthetic_local_gaussian", "session_pooled_centered", "local_time_window", "trial_full"),
        default="trial_full",
    )
    p.add_argument("--local-window-radius", type=int, default=30)
    return p


def _build_eye_cloud_offsets_px(
    eyepos: np.ndarray,
    tr: int,
    tt: int,
    ppd: float,
    mode: str,
    max_samples: int,
    seed: int,
    local_window_radius: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    all_valid = np.isfinite(eyepos).all(axis=2)
    trial_valid = np.isfinite(eyepos[tr]).all(axis=1)
    eye_center = np.asarray(eyepos[tr, tt], dtype=np.float64)

    def _subsample(x: np.ndarray) -> np.ndarray:
        if x.shape[0] > int(max_samples):
            pick = rng.choice(np.arange(x.shape[0]), size=int(max_samples), replace=False)
            return x[pick]
        return x

    if mode == "trial_full":
        eye_trial = np.asarray(eyepos[tr, trial_valid], dtype=np.float64)
        offs = (eye_trial - eye_center) * float(ppd)
        return _subsample(offs)

    if mode == "session_pooled_centered":
        pooled = np.asarray(eyepos[all_valid], dtype=np.float64)
        pooled = pooled - np.mean(pooled, axis=0, keepdims=True)
        offs = pooled * float(ppd)
        return _subsample(offs)

    if mode == "local_time_window":
        t0 = max(0, int(tt) - int(local_window_radius))
        t1 = min(int(eyepos.shape[1]), int(tt) + int(local_window_radius) + 1)
        window = np.asarray(eyepos[tr, t0:t1], dtype=np.float64)
        keep = np.isfinite(window).all(axis=1)
        window = window[keep]
        if window.shape[0] < 4:
            eye_trial = np.asarray(eyepos[tr, trial_valid], dtype=np.float64)
            window = eye_trial
        offs = (window - eye_center) * float(ppd)
        return _subsample(offs)

    # synthetic_local_gaussian
    t0 = max(0, int(tt) - int(local_window_radius))
    t1 = min(int(eyepos.shape[1]), int(tt) + int(local_window_radius) + 1)
    local = np.asarray(eyepos[tr, t0:t1], dtype=np.float64)
    keep = np.isfinite(local).all(axis=1)
    local = local[keep]
    if local.shape[0] < 4:
        local = np.asarray(eyepos[tr, trial_valid], dtype=np.float64)
    if local.shape[0] < 4:
        local = np.asarray(eyepos[all_valid], dtype=np.float64)
    local_centered = (local - np.mean(local, axis=0, keepdims=True)) * float(ppd)
    cov = _covariance(local_centered)
    cov = cov + np.eye(2, dtype=np.float64) * 1e-8
    n = int(min(max_samples, max(4, local_centered.shape[0])))
    offs = rng.multivariate_normal(mean=np.zeros(2, dtype=np.float64), cov=cov, size=n)
    return np.asarray(offs, dtype=np.float64)


def main() -> None:
    args = build_parser().parse_args()
    if args.reuse_tangents_root is not None:
        _run_cached_split_mode_analyses(args)
        return
    if args.mode == "smoke":
        n_objects = int(args.n_images) if args.n_images is not None else 8
        delta_arcmins = [float(x) for x in (args.delta_arcmin.split(",") if args.delta_arcmin else ["0.125", "0.25", "0.5", "1.0"])]
        cloud_scales = [float(x) for x in (args.cloud_scales.split(",") if args.cloud_scales else ["0.5", "1.0"])]
        folds = int(args.folds) if args.folds is not None else 2
        null_repeats = int(args.null_repeats) if args.null_repeats is not None else 10
        stratified = bool(args.stratified_sampling)
    else:
        n_objects = int(args.n_images) if args.n_images is not None else 64
        delta_arcmins = [float(x) for x in (args.delta_arcmin.split(",") if args.delta_arcmin else ["0.125", "0.25", "0.5", "1.0"])]
        cloud_scales = [float(x) for x in (args.cloud_scales.split(",") if args.cloud_scales else ["0.25", "0.5", "1.0", "2.0"])]
        folds = int(args.folds) if args.folds is not None else 5
        null_repeats = int(args.null_repeats) if args.null_repeats is not None else 200
        stratified = True

    analysis5_n_objects = int(args.analysis5_n_images) if args.analysis5_n_images is not None else min(n_objects, 32 if args.mode == "production" else n_objects)
    out_root = Path(args.output_root)
    for sub in ("tangent_maps", "union_spectrum", "train_test_basis", "metric_law", "covariance_approx", "figures"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    ctx = _load_twin_context(model_device=str(args.model_device))
    _write_csv(out_root / "canonical_unit_manifest.csv", ctx.manifest_rows)

    data = get_fixrsvp_data(
        subject=str(args.subject),
        date=str(args.date),
        dataset_configs_path=str(args.dataset_configs_path),
        use_cached_data=bool(args.use_cached_data),
    )
    stim, image_ids, eyepos = _harmonize_fixrsvp_arrays(data)
    valid = np.isfinite(eyepos).all(axis=2) & np.isfinite(stim).all(axis=tuple(range(2, stim.ndim))) & (image_ids >= 0)
    n_trials, n_time = image_ids.shape

    candidates: list[ObjectCandidate] = []
    for tr in range(n_trials):
        for tt in range(ctx.n_lags - 1, n_time):
            if not bool(valid[tr, tt]):
                continue
            img = int(image_ids[tr, tt])
            if img < 0:
                continue
            candidates.append(
                ObjectCandidate(
                    object_id=f"{img}/{tr}/{tt}",
                    image_id=img,
                    trial_index=tr,
                    time_index=tt,
                    frame_stats=_frame_stats(stim[tr, tt]),
                )
            )

    selected_objects = _select_objects(candidates, n_objects=n_objects, seed=int(args.seed), stratified=stratified)
    sampled_rows = [
        {
            "object_id": obj.object_id,
            "image_id": int(obj.image_id),
            "trial_index": int(obj.trial_index),
            "time_index": int(obj.time_index),
            **obj.frame_stats,
        }
        for obj in selected_objects
    ]
    _write_csv(out_root / "sampled_object_stats.csv", sampled_rows)
    _write_csv(
        out_root / "input_shape_audit.csv",
        [
            {
                "raw_stim_shape": str(tuple(int(v) for v in stim.shape)),
                "image_ids_shape": str(tuple(int(v) for v in image_ids.shape)),
                "eyepos_shape": str(tuple(int(v) for v in eyepos.shape)),
                "history_length_frames": int(ctx.n_lags),
                "canonical_input_shape": str((1, 1, int(ctx.n_lags), int(stim.shape[-2]), int(stim.shape[-1]))),
                "normalization_policy": "none",
            }
        ],
    )
    if selected_objects:
        first_history = _history_from_stim(stim[selected_objects[0].trial_index], selected_objects[0].time_index, ctx.n_lags)
        _save_history_png(out_root / "figures" / "tfts_example_history.png", first_history)

    arcmin_to_model_px = float(args.model_ppd) / 60.0
    object_payload: dict[float, dict[str, dict[str, Any]]] = {float(d): {} for d in delta_arcmins}
    object_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for obj in selected_objects:
        history = _history_from_stim(stim[obj.trial_index], obj.time_index, ctx.n_lags)
        pred_new = _predict_rate_from_history(ctx, history, model_device=str(args.model_device))
        pred_canon = _canonical_rate_from_history(ctx, history, model_device=str(args.model_device))
        diff = np.abs(pred_new - pred_canon)
        prediction_rows.append(
            {
                "object_id": obj.object_id,
                "image_id": int(obj.image_id),
                "trial_index": int(obj.trial_index),
                "time_index": int(obj.time_index),
                "max_abs_diff": float(np.max(diff)),
                "median_abs_diff": float(np.median(diff)),
                "mean_abs_diff": float(np.mean(diff)),
                "corr_across_units": float(np.corrcoef(pred_new, pred_canon)[0, 1]) if pred_new.size > 1 else float("nan"),
            }
        )

        for d_arcmin in delta_arcmins:
            delta_px = float(d_arcmin) * arcmin_to_model_px
            tang = _fit_tangent_for_history(ctx, history, delta_px=delta_px, model_device=str(args.model_device))
            r0, bx, by = tang["r0"], tang["bx"], tang["by"]
            r2, r2x, r2y = _local_linear_r2(
                ctx,
                history,
                r0=r0,
                bx=bx,
                by=by,
                delta_px=delta_px,
                model_device=str(args.model_device),
                seed=int(args.seed) + int(obj.image_id) + int(obj.time_index),
            )
            svals = _safe_svdvals(np.stack([bx, by], axis=1))
            e = svals * svals
            object_payload[float(d_arcmin)][obj.object_id] = {
                "r0":   tang["r0"].astype(np.float32),
                "bx":   tang["bx"].astype(np.float32),
                "by":   tang["by"].astype(np.float32),
                "rx_p": tang["rx_p"].astype(np.float32),
                "rx_m": tang["rx_m"].astype(np.float32),
                "ry_p": tang["ry_p"].astype(np.float32),
                "ry_m": tang["ry_m"].astype(np.float32),
                "history": history,
                "image_id":    int(obj.image_id),
                "trial_index": int(obj.trial_index),
                "time_index":  int(obj.time_index),
                "delta_arcmin":    float(d_arcmin),
                "delta_model_px":  float(delta_px),
            }
            object_rows.append(
                {
                    "object_id": obj.object_id,
                    "image_id": int(obj.image_id),
                    "trial_index": int(obj.trial_index),
                    "time_index": int(obj.time_index),
                    "delta_arcmin_requested": float(d_arcmin),
                    "delta_model_px_used": float(delta_px),
                    "arcmin_to_model_px": float(arcmin_to_model_px),
                    "conversion_status": "provisional_pixel_equivalent",
                    "object_type": "full_history_tensor",
                    "history_length_frames": int(ctx.n_lags),
                    "n_units": int(bx.size),
                    "norm_bx": float(np.linalg.norm(bx)),
                    "norm_by": float(np.linalg.norm(by)),
                    "cos_bx_by": float(np.dot(bx, by) / (np.linalg.norm(bx) * np.linalg.norm(by) + 1e-12)),
                    "rank_J": int(_numerical_rank(e)),
                    "singular_value_1": float(svals[0]) if svals.size > 0 else float("nan"),
                    "singular_value_2": float(svals[1]) if svals.size > 1 else float("nan"),
                    "frac_energy_top1": float(e[0] / np.sum(e)) if e.size else float("nan"),
                    "linear_local_r2": float(r2),
                    "linear_local_r2_x": float(r2x),
                    "linear_local_r2_y": float(r2y),
                }
            )

    _write_csv(out_root / "prediction_path_validation.csv", prediction_rows)
    _write_csv(out_root / "tangent_maps" / "twin_tangent_object_metrics.csv", object_rows)

    delta_sensitivity_rows: list[dict[str, object]] = []
    for obj in selected_objects:
        rows = [r for r in object_rows if r["object_id"] == obj.object_id]
        rows.sort(key=lambda r: float(cast(Any, r["delta_arcmin_requested"])))
        r2_vals = np.asarray([float(cast(Any, r["linear_local_r2"])) for r in rows], dtype=np.float64)
        bx_norm_vals = np.asarray([float(cast(Any, r["norm_bx"])) for r in rows], dtype=np.float64)
        by_norm_vals = np.asarray([float(cast(Any, r["norm_by"])) for r in rows], dtype=np.float64)
        bx_first = np.asarray(object_payload[delta_arcmins[0]][obj.object_id]["bx"], dtype=np.float64)
        by_first = np.asarray(object_payload[delta_arcmins[0]][obj.object_id]["by"], dtype=np.float64)
        bx_adjacent = []
        by_adjacent = []
        bx_vs_first = []
        by_vs_first = []
        for idx in range(1, len(delta_arcmins)):
            prev = object_payload[delta_arcmins[idx - 1]][obj.object_id]
            curr = object_payload[delta_arcmins[idx]][obj.object_id]
            bx_prev = np.asarray(prev["bx"], dtype=np.float64)
            by_prev = np.asarray(prev["by"], dtype=np.float64)
            bx_curr = np.asarray(curr["bx"], dtype=np.float64)
            by_curr = np.asarray(curr["by"], dtype=np.float64)
            bx_adjacent.append(float(np.dot(bx_prev, bx_curr) / (np.linalg.norm(bx_prev) * np.linalg.norm(bx_curr) + 1e-12)))
            by_adjacent.append(float(np.dot(by_prev, by_curr) / (np.linalg.norm(by_prev) * np.linalg.norm(by_curr) + 1e-12)))
            bx_vs_first.append(float(np.dot(bx_first, bx_curr) / (np.linalg.norm(bx_first) * np.linalg.norm(bx_curr) + 1e-12)))
            by_vs_first.append(float(np.dot(by_first, by_curr) / (np.linalg.norm(by_first) * np.linalg.norm(by_curr) + 1e-12)))
        delta_sensitivity_rows.append(
            {
                "object_id": obj.object_id,
                "image_id": int(obj.image_id),
                "trial_index": int(obj.trial_index),
                "time_index": int(obj.time_index),
                "r2_median": float(np.median(r2_vals)) if r2_vals.size else float("nan"),
                "r2_iqr": float(np.percentile(r2_vals, 75) - np.percentile(r2_vals, 25)) if r2_vals.size else float("nan"),
                "r2_fraction_below_0p1": float(np.mean(r2_vals < 0.1)) if r2_vals.size else float("nan"),
                "norm_bx_mean": float(np.mean(bx_norm_vals)) if bx_norm_vals.size else float("nan"),
                "norm_by_mean": float(np.mean(by_norm_vals)) if by_norm_vals.size else float("nan"),
                "bx_adjacent_delta_cos_mean": float(np.mean(bx_adjacent)) if bx_adjacent else float("nan"),
                "by_adjacent_delta_cos_mean": float(np.mean(by_adjacent)) if by_adjacent else float("nan"),
                "bx_cos_vs_first_delta_mean": float(np.mean(bx_vs_first)) if bx_vs_first else float("nan"),
                "by_cos_vs_first_delta_mean": float(np.mean(by_vs_first)) if by_vs_first else float("nan"),
            }
        )
    _write_csv(out_root / "delta_sensitivity_summary.csv", delta_sensitivity_rows)

    # Union/basis finite guard diagnostics and filtering
    min_valid_objects_union_basis = 16 if str(args.mode) == "smoke" else 48
    tangent_norm_eps = 1e-10
    filtered_object_payload: dict[float, dict[str, dict[str, Any]]] = {}
    dropped_union_basis_rows: list[dict[str, object]] = []
    union_basis_diag_rows: list[dict[str, object]] = []

    for d_arcmin in delta_arcmins:
        payload = object_payload[float(d_arcmin)]
        object_ids = sorted(payload.keys())
        n_objects_total = int(len(object_ids))
        if n_objects_total == 0:
            filtered_object_payload[float(d_arcmin)] = {}
            union_basis_diag_rows.append(
                {
                    "delta": float(d_arcmin),
                    "n_objects_total": 0,
                    "n_objects_with_finite_bx_by": 0,
                    "n_objects_dropped_nonfinite": 0,
                    "n_units_total": int(ctx.n_units),
                    "fraction_finite_bx_by_matrix": float("nan"),
                    "norm_bx_min": float("nan"),
                    "norm_bx_median": float("nan"),
                    "norm_bx_max": float("nan"),
                    "norm_by_min": float("nan"),
                    "norm_by_median": float("nan"),
                    "norm_by_max": float("nan"),
                    "zero_norm_bx_count": 0,
                    "zero_norm_by_count": 0,
                    "nonfinite_or_zero_object_ids": "",
                    "n_valid_objects_union_basis": 0,
                    "min_valid_objects_threshold": int(min_valid_objects_union_basis),
                    "union_basis_stage_status": "not_run_no_objects",
                }
            )
            continue

        bx_mat = np.stack([np.asarray(payload[oid]["bx"], dtype=np.float64) for oid in object_ids], axis=1)
        by_mat = np.stack([np.asarray(payload[oid]["by"], dtype=np.float64) for oid in object_ids], axis=1)
        finite_bx = np.isfinite(bx_mat)
        finite_by = np.isfinite(by_mat)
        finite_obj_mask = np.all(finite_bx, axis=0) & np.all(finite_by, axis=0)
        norm_bx = np.linalg.norm(np.nan_to_num(bx_mat, nan=0.0, posinf=0.0, neginf=0.0), axis=0)
        norm_by = np.linalg.norm(np.nan_to_num(by_mat, nan=0.0, posinf=0.0, neginf=0.0), axis=0)
        zero_norm_mask = (norm_bx <= tangent_norm_eps) | (norm_by <= tangent_norm_eps)
        keep_mask = finite_obj_mask & (~zero_norm_mask)

        dropped_ids: list[str] = []
        for idx, oid in enumerate(object_ids):
            if keep_mask[idx]:
                continue
            drop_nonfinite = not bool(finite_obj_mask[idx])
            drop_zero = bool(zero_norm_mask[idx])
            reason = []
            if drop_nonfinite:
                reason.append("non_finite_tangent")
            if drop_zero:
                reason.append("zero_or_near_zero_tangent_norm")
            dropped_ids.append(str(oid))
            pobj = payload[oid]
            dropped_union_basis_rows.append(
                {
                    "delta": float(d_arcmin),
                    "object_id": str(oid),
                    "image_id": int(pobj["image_id"]),
                    "trial_index": int(pobj["trial_index"]),
                    "time_index": int(pobj["time_index"]),
                    "drop_reason": ";".join(reason),
                    "norm_bx": float(norm_bx[idx]),
                    "norm_by": float(norm_by[idx]),
                    "finite_fraction_bx": float(np.mean(finite_bx[:, idx])),
                    "finite_fraction_by": float(np.mean(finite_by[:, idx])),
                }
            )

        kept_ids = [oid for idx, oid in enumerate(object_ids) if keep_mask[idx]]
        filtered_object_payload[float(d_arcmin)] = {oid: payload[oid] for oid in kept_ids}
        n_finite = int(np.sum(finite_obj_mask))
        n_drop_nonfinite = int(np.sum(~finite_obj_mask))
        n_valid = int(len(kept_ids))
        stage_status = "ok" if n_valid >= int(min_valid_objects_union_basis) else "not_run_insufficient_finite_objects"
        union_basis_diag_rows.append(
            {
                "delta": float(d_arcmin),
                "n_objects_total": int(n_objects_total),
                "n_objects_with_finite_bx_by": int(n_finite),
                "n_objects_dropped_nonfinite": int(n_drop_nonfinite),
                "n_units_total": int(ctx.n_units),
                "fraction_finite_bx_by_matrix": float((np.sum(finite_bx) + np.sum(finite_by)) / (2.0 * bx_mat.shape[0] * bx_mat.shape[1])),
                "norm_bx_min": float(np.nanmin(norm_bx)),
                "norm_bx_median": float(np.nanmedian(norm_bx)),
                "norm_bx_max": float(np.nanmax(norm_bx)),
                "norm_by_min": float(np.nanmin(norm_by)),
                "norm_by_median": float(np.nanmedian(norm_by)),
                "norm_by_max": float(np.nanmax(norm_by)),
                "zero_norm_bx_count": int(np.sum(norm_bx <= tangent_norm_eps)),
                "zero_norm_by_count": int(np.sum(norm_by <= tangent_norm_eps)),
                "nonfinite_or_zero_object_ids": ";".join(dropped_ids),
                "n_valid_objects_union_basis": int(n_valid),
                "min_valid_objects_threshold": int(min_valid_objects_union_basis),
                "union_basis_stage_status": stage_status,
            }
        )

    _write_csv(out_root / "union_basis_input_diagnostics.csv", union_basis_diag_rows)
    _write_csv(out_root / "dropped_objects_union_basis.csv", dropped_union_basis_rows)

    object_composition_audit_rows, fold_leakage_audit_rows, history_overlap_audit_rows = _build_leakage_audits(
        filtered_object_payload=filtered_object_payload,
        delta_arcmins=delta_arcmins,
        folds=int(folds),
        seed=int(args.seed),
        n_lags=int(ctx.n_lags),
    )
    _write_csv(out_root / "object_composition_audit.csv", object_composition_audit_rows)
    _write_csv(out_root / "fold_leakage_audit.csv", fold_leakage_audit_rows)
    _write_csv(out_root / "history_overlap_audit.csv", history_overlap_audit_rows)

    image_summary_rows: list[dict[str, object]] = []
    for d_arcmin in delta_arcmins:
        by_image: dict[int, list[dict[str, Any]]] = {}
        for payload in object_payload[float(d_arcmin)].values():
            by_image.setdefault(int(payload["image_id"]), []).append(payload)
        for image_id, payloads in by_image.items():
            bx_m = np.mean(np.stack([np.asarray(p["bx"], dtype=np.float64) for p in payloads], axis=0), axis=0)
            by_m = np.mean(np.stack([np.asarray(p["by"], dtype=np.float64) for p in payloads], axis=0), axis=0)
            image_summary_rows.append(
                {
                    "image_id": int(image_id),
                    "delta_arcmin_requested": float(d_arcmin),
                    "n_objects": int(len(payloads)),
                    "norm_bx": float(np.linalg.norm(bx_m)),
                    "norm_by": float(np.linalg.norm(by_m)),
                    "cos_bx_by": float(np.dot(bx_m, by_m) / (np.linalg.norm(bx_m) * np.linalg.norm(by_m) + 1e-12)),
                    "summary_type": "image_averaged_history_tangent",
                }
            )
    _write_csv(out_root / "tangent_maps" / "twin_tangent_image_metrics.csv", image_summary_rows)

    union_rows: list[dict[str, object]] = []
    union_summary: list[dict[str, object]] = []
    null_spectrum_rows: list[dict[str, object]] = []
    for d_arcmin in delta_arcmins:
        payload = filtered_object_payload[float(d_arcmin)]
        object_ids = sorted(payload.keys())
        if len(object_ids) < int(min_valid_objects_union_basis):
            union_summary.append(
                {
                    "delta": float(d_arcmin),
                    "space": "raw",
                    "component_set": "combined",
                    "n_objects": int(len(object_ids)),
                    "rank": 0,
                    "participation_ratio": float("nan"),
                    "n_dims_50pct": 0,
                    "n_dims_80pct": 0,
                    "n_dims_90pct": 0,
                    "n_dims_95pct": 0,
                    "null_pr_mean": float("nan"),
                    "null_pr_ci_low": float("nan"),
                    "null_pr_ci_high": float("nan"),
                    "compactness_label": "not_run_insufficient_finite_objects",
                    "status": "not_run_insufficient_finite_objects",
                }
            )
            continue
        bx = np.stack([payload[oid]["bx"] for oid in object_ids], axis=1)
        by = np.stack([payload[oid]["by"] for oid in object_ids], axis=1)
        combined = np.concatenate([bx, by], axis=1)
        for space_name, fspace in {
            "raw": lambda m: m,
        }.items():
            for set_name, mat0 in (("combined", combined),):
                mat = fspace(np.asarray(mat0, dtype=np.float64))
                evals, _ = _eigh_desc(mat @ mat.T)
                evals = np.maximum(evals, 0.0)
                total_eval = float(np.sum(evals))
                if total_eval <= 1e-12:
                    union_summary.append(
                        {
                            "delta": float(d_arcmin),
                            "space": space_name,
                            "component_set": set_name,
                            "n_objects": int(len(object_ids)),
                            "rank": 0,
                            "participation_ratio": float("nan"),
                            "n_dims_50pct": 0,
                            "n_dims_80pct": 0,
                            "n_dims_90pct": 0,
                            "n_dims_95pct": 0,
                            "null_pr_mean": float("nan"),
                            "null_pr_ci_low": float("nan"),
                            "null_pr_ci_high": float("nan"),
                            "compactness_label": "not_run_no_finite_tangents",
                            "status": "not_run_no_finite_tangents",
                        }
                    )
                    continue
                frac = evals / (total_eval + 1e-12)
                rank = _numerical_rank(evals)
                pr = _participation_ratio(evals)
                ranks = _ranks_at_thresholds(frac)
                null_pr = []
                for nrep in range(int(null_repeats)):
                    nrng = np.random.default_rng(int(args.seed) + 10000 + nrep)
                    m_shuf = np.stack([col[nrng.permutation(col.shape[0])] for col in mat.T], axis=1)
                    es, _ = _eigh_desc(m_shuf @ m_shuf.T)
                    es = np.maximum(es, 0.0)
                    null_pr_val = _participation_ratio(es)
                    null_pr.append(null_pr_val)
                    # Persist per-component data so Panel C can draw a real null band.
                    null_total = float(np.sum(es))
                    null_frac = es / (null_total + 1e-12)
                    null_cum = np.cumsum(null_frac)
                    for i in range(min(64, int(es.size))):
                        null_spectrum_rows.append({
                            "delta":                       float(d_arcmin),
                            "space":                       space_name,
                            "tangent_set":                 set_name,
                            "null_type":                   "unit_shuffle",
                            "null_repeat":                 int(nrep),
                            "component_index":             int(i + 1),
                            "eigenvalue":                  float(es[i]),
                            "fraction_variance":           float(null_frac[i]),
                            "cumulative_fraction_variance": float(null_cum[i]),
                            "participation_ratio":         float(null_pr_val),
                        })
                null_pr = np.asarray(null_pr, dtype=np.float64)
                union_summary.append(
                    {
                        "delta": float(d_arcmin),
                        "space": space_name,
                        "component_set": set_name,
                        "n_objects": int(len(object_ids)),
                        "rank": int(rank),
                        "participation_ratio": float(pr),
                        "n_dims_50pct": int(ranks["rank_50"]),
                        "n_dims_80pct": int(ranks["rank_80"]),
                        "n_dims_90pct": int(ranks["rank_90"]),
                        "n_dims_95pct": int(ranks["rank_95"]),
                        "null_pr_mean": float(np.mean(null_pr)) if null_pr.size else float("nan"),
                        "null_pr_ci_low": float(np.percentile(null_pr, 2.5)) if null_pr.size else float("nan"),
                        "null_pr_ci_high": float(np.percentile(null_pr, 97.5)) if null_pr.size else float("nan"),
                        "compactness_label": "compact" if null_pr.size and np.isfinite(pr) and pr < np.percentile(null_pr, 2.5) else "not_compact",
                        "status": "ok",
                    }
                )
                for i, ev in enumerate(evals):
                    union_rows.append(
                        {
                            "delta": float(d_arcmin),
                            "space": space_name,
                            "tangent_set": set_name,
                            "component_index": int(i + 1),
                            "eigenvalue": float(ev),
                            "fraction_variance": float(frac[i]) if i < frac.size else float("nan"),
                            "cumulative_fraction_variance": float(np.sum(frac[: i + 1])) if i < frac.size else float("nan"),
                            "participation_ratio": float(pr),
                            "rank": int(rank),
                            "rank_50": int(ranks["rank_50"]),
                            "rank_75": int(ranks["rank_75"]),
                            "rank_90": int(ranks["rank_90"]),
                            "rank_95": int(ranks["rank_95"]),
                        }
                    )
    _write_csv(out_root / "union_spectrum" / "twin_tangent_union_spectrum.csv", union_rows)
    _write_csv(out_root / "union_spectrum" / "twin_tangent_union_summary.csv", union_summary)
    _write_csv(out_root / "union_spectrum" / "twin_tangent_union_null_spectrum.csv", null_spectrum_rows)

    # Summary by (delta, space, tangent_set, component_index) for Panel C fill_between.
    from collections import defaultdict
    _comp_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in null_spectrum_rows:
        key = (row["delta"], row["space"], row["tangent_set"], row["component_index"])
        _comp_groups[key].append(row)
    null_spectrum_summary_rows: list[dict[str, object]] = []
    for (delta, space, tangent_set, comp_idx), rows in sorted(_comp_groups.items()):
        cumvars  = np.asarray([r["cumulative_fraction_variance"] for r in rows], dtype=np.float64)
        fracvars = np.asarray([r["fraction_variance"]           for r in rows], dtype=np.float64)
        null_spectrum_summary_rows.append({
            "delta":           float(delta),
            "space":           str(space),
            "tangent_set":     str(tangent_set),
            "component_index": int(comp_idx),
            "n_null_repeats":  len(rows),
            "cumvar_median":   float(np.median(cumvars)),
            "cumvar_ci_low":   float(np.percentile(cumvars, 2.5)),
            "cumvar_ci_high":  float(np.percentile(cumvars, 97.5)),
            "fracvar_median":  float(np.median(fracvars)),
            "fracvar_ci_low":  float(np.percentile(fracvars, 2.5)),
            "fracvar_ci_high": float(np.percentile(fracvars, 97.5)),
        })
    _write_csv(out_root / "union_spectrum" / "twin_tangent_union_null_spectrum_summary.csv", null_spectrum_summary_rows)

    basis_rows: list[dict[str, object]] = []
    k_grid = [2, 5, 10, 20]
    for d_arcmin in delta_arcmins:
        payload = filtered_object_payload[float(d_arcmin)]
        object_ids = sorted(payload.keys())
        if len(object_ids) < max(4, int(min_valid_objects_union_basis)):
            basis_rows.append(
                {
                    "delta": float(d_arcmin),
                    "fold": -1,
                    "train_n_objects": 0,
                    "test_n_objects": 0,
                    "tangent_set": "combined",
                    "basis_rank_k": -1,
                    "rank_train": 0,
                    "rank_test": 0,
                    "train_finite_fraction": float("nan"),
                    "test_finite_fraction": float("nan"),
                    "train_matrix_norm": float("nan"),
                    "test_matrix_norm": float("nan"),
                    "test_variance_captured": float("nan"),
                    "null_mean": float("nan"),
                    "null_ci_low": float("nan"),
                    "null_ci_high": float("nan"),
                    "effect_minus_null": float("nan"),
                    "random_basis_null_mean": float("nan"),
                    "fold_status": "not_run_insufficient_finite_objects",
                    "interpretation_label": "not_run_insufficient_finite_objects",
                }
            )
            continue
        fold_splits = _build_fold_splits(len(object_ids), folds=int(folds), seed=int(args.seed))
        bx = np.stack([payload[oid]["bx"] for oid in object_ids], axis=1)
        by = np.stack([payload[oid]["by"] for oid in object_ids], axis=1)
        mats = {"combined": np.concatenate([bx, by], axis=1)}

        for fold_id, test_idx in enumerate(fold_splits):
            test_idx = np.asarray(test_idx, dtype=np.int64)
            test_set = set(test_idx.tolist())
            train_idx = np.asarray([i for i in range(len(object_ids)) if i not in test_set], dtype=np.int64)
            if train_idx.size < 2 or test_idx.size < 1:
                continue
            for set_name, m in mats.items():
                if set_name == "combined":
                    m_train = np.concatenate([bx[:, train_idx], by[:, train_idx]], axis=1)
                    m_test = np.concatenate([bx[:, test_idx], by[:, test_idx]], axis=1)
                else:
                    m_train = m[:, train_idx]
                    m_test = m[:, test_idx]
                train_finite_fraction = float(np.mean(np.isfinite(m_train)))
                test_finite_fraction = float(np.mean(np.isfinite(m_test)))
                train_matrix_norm = float(np.linalg.norm(np.nan_to_num(m_train, nan=0.0, posinf=0.0, neginf=0.0)))
                test_matrix_norm = float(np.linalg.norm(np.nan_to_num(m_test, nan=0.0, posinf=0.0, neginf=0.0)))
                fold_valid = (
                    train_finite_fraction == 1.0
                    and test_finite_fraction == 1.0
                    and train_matrix_norm > 1e-12
                    and test_matrix_norm > 1e-12
                )
                if not fold_valid:
                    for k in k_grid:
                        basis_rows.append(
                            {
                                "delta": float(d_arcmin),
                                "fold": int(fold_id),
                                "train_n_objects": int(train_idx.size),
                                "test_n_objects": int(test_idx.size),
                                "tangent_set": set_name,
                                "basis_rank_k": int(k),
                                "rank_train": 0,
                                "rank_test": 0,
                                "train_finite_fraction": train_finite_fraction,
                                "test_finite_fraction": test_finite_fraction,
                                "train_matrix_norm": train_matrix_norm,
                                "test_matrix_norm": test_matrix_norm,
                                "test_variance_captured": float("nan"),
                                "null_mean": float("nan"),
                                "null_ci_low": float("nan"),
                                "null_ci_high": float("nan"),
                                "effect_minus_null": float("nan"),
                                "random_basis_null_mean": float("nan"),
                                "fold_status": "invalid_nonfinite_or_degenerate_matrix",
                                "interpretation_label": "fold_invalid",
                            }
                        )
                    continue
                c_train = m_train @ m_train.T
                c_test = m_test @ m_test.T
                evals_train, u_train = _eigh_desc(c_train)
                evals_test, _ = _eigh_desc(c_test)
                rank_train = _numerical_rank(np.maximum(evals_train, 0.0))
                rank_test = _numerical_rank(np.maximum(evals_test, 0.0))
                max_rank = rank_train
                for k in k_grid:
                    if int(k) > max_rank:
                        basis_rows.append(
                            {
                                "delta": float(d_arcmin),
                                "fold": int(fold_id),
                                "train_n_objects": int(train_idx.size),
                                "test_n_objects": int(test_idx.size),
                                "tangent_set": set_name,
                                "basis_rank_k": int(k),
                                "rank_train": int(rank_train),
                                "rank_test": int(rank_test),
                                "train_finite_fraction": train_finite_fraction,
                                "test_finite_fraction": test_finite_fraction,
                                "train_matrix_norm": train_matrix_norm,
                                "test_matrix_norm": test_matrix_norm,
                                "test_variance_captured": float("nan"),
                                "null_mean": float("nan"),
                                "null_ci_low": float("nan"),
                                "null_ci_high": float("nan"),
                                "effect_minus_null": float("nan"),
                                "random_basis_null_mean": float("nan"),
                                "fold_status": "ok_rank_exceeded",
                                "interpretation_label": "not_interpretable_rank_exceeded",
                            }
                        )
                        continue
                    cap = _variance_capture(u_train, m_test, k=int(k))
                    null_vals = []
                    null_rand = []
                    for nrep in range(int(null_repeats)):
                        nrng = np.random.default_rng(int(args.seed) + 30000 + fold_id * 100 + nrep)
                        m_train_shuf = np.stack([col[nrng.permutation(col.shape[0])] for col in m_train.T], axis=1)
                        _, u_shuf = _eigh_desc(m_train_shuf @ m_train_shuf.T)
                        null_vals.append(_variance_capture(u_shuf, m_test, k=int(k)))
                        q, _ = np.linalg.qr(nrng.normal(size=(m_test.shape[0], int(k))))
                        null_rand.append(_variance_capture(q, m_test, k=int(k)))
                    n0 = np.asarray(null_vals, dtype=np.float64)
                    n1 = np.asarray(null_rand, dtype=np.float64)
                    basis_rows.append(
                        {
                            "delta": float(d_arcmin),
                            "fold": int(fold_id),
                            "train_n_objects": int(train_idx.size),
                            "test_n_objects": int(test_idx.size),
                            "tangent_set": set_name,
                            "basis_rank_k": int(k),
                            "rank_train": int(rank_train),
                            "rank_test": int(rank_test),
                            "train_finite_fraction": train_finite_fraction,
                            "test_finite_fraction": test_finite_fraction,
                            "train_matrix_norm": train_matrix_norm,
                            "test_matrix_norm": test_matrix_norm,
                            "test_variance_captured": float(cap),
                            "null_mean": float(np.mean(n0)) if n0.size else float("nan"),
                            "null_ci_low": float(np.percentile(n0, 2.5)) if n0.size else float("nan"),
                            "null_ci_high": float(np.percentile(n0, 97.5)) if n0.size else float("nan"),
                            "effect_minus_null": float(cap - np.mean(n0)) if n0.size else float("nan"),
                            "random_basis_null_mean": float(np.mean(n1)) if n1.size else float("nan"),
                            "fold_status": "ok",
                            "interpretation_label": "supported" if np.isfinite(cap) and n0.size and (cap > np.percentile(n0, 97.5)) else "not_supported",
                        }
                    )
    _write_csv(out_root / "train_test_basis" / "twin_tangent_train_test_basis.csv", basis_rows)

    cov_rows: list[dict[str, object]] = []
    for d_arcmin in delta_arcmins:
        payload = filtered_object_payload[float(d_arcmin)]
        object_ids = sorted(payload.keys())[:analysis5_n_objects]
        for object_id in object_ids:
            obj = payload[object_id]
            tr = int(obj["trial_index"])
            tt = int(obj["time_index"])
            history = np.asarray(obj["history"], dtype=np.float32)
            bx = np.asarray(obj["bx"], dtype=np.float64)
            by = np.asarray(obj["by"], dtype=np.float64)
            j = np.stack([bx, by], axis=1)

            offsets_px = _build_eye_cloud_offsets_px(
                eyepos=eyepos,
                tr=tr,
                tt=tt,
                ppd=float(args.model_ppd),
                mode=str(args.eye_cloud_mode),
                max_samples=int(args.max_eye_samples),
                seed=int(args.seed) + int(tr) * 1000 + int(tt),
                local_window_radius=int(args.local_window_radius),
            )
            if offsets_px.shape[0] < 4:
                continue

            for cs in cloud_scales:
                offs = offsets_px * float(cs)
                rs = []
                h0 = _movie_to_thw(history).to(str(args.model_device))
                for dx, dy in offs:
                    hs = _shift_movie_subpixel(h0, dx_px=float(dx), dy_px=float(dy)).detach().cpu().numpy()
                    rs.append(_predict_rate_from_history(ctx, hs, model_device=str(args.model_device)))
                rmat = np.asarray(rs, dtype=np.float64)
                c_full = _covariance(rmat)
                s_eye = _covariance(offs)
                c_lin = j @ s_eye @ j.T
                ef, uf = _eigh_desc(c_full)
                el, ul = _eigh_desc(c_lin)
                ef = np.maximum(ef, 0.0)
                el = np.maximum(el, 0.0)
                rank_full = _numerical_rank(ef)
                rank_lin = _numerical_rank(el)
                pr_f = _participation_ratio(ef)
                pr_l = _participation_ratio(el)
                tr_f = float(np.sum(ef))
                tr_l = float(np.sum(el))

                def overlap_or_nan(k: int) -> float:
                    return float("nan") if int(k) > min(rank_full, rank_lin) else _subspace_overlap(uf, ul, k=int(k))

                def frac_full_in_lin(k: int) -> float:
                    if int(k) > min(rank_full, rank_lin) or tr_f <= 1e-12:
                        return float("nan")
                    kk = min(int(k), ul.shape[1])
                    p = ul[:, :kk] @ ul[:, :kk].T
                    return float(np.trace(p @ c_full) / tr_f)

                cov_rows.append(
                    {
                        "object_id": object_id,
                        "image_id": int(obj["image_id"]),
                        "trial_index": tr,
                        "time_index": tt,
                        "delta": float(d_arcmin),
                        "cloud_source": str(args.eye_cloud_mode),
                        "cloud_scale": float(cs),
                        "n_eye_samples": int(offs.shape[0]),
                        "rank_full": int(rank_full),
                        "rank_lin": int(rank_lin),
                        "trace_c_full": tr_f,
                        "trace_c_lin": tr_l,
                        "trace_ratio_lin_full": float(tr_l / (tr_f + 1e-12)),
                        "subspace_overlap_k1": overlap_or_nan(1),
                        "subspace_overlap_k2": overlap_or_nan(2),
                        "subspace_overlap_k3": overlap_or_nan(3),
                        "full_pr": float(pr_f),
                        "lin_pr": float(pr_l),
                        "fraction_full_variance_in_lin_subspace_k1": frac_full_in_lin(1),
                        "fraction_full_variance_in_lin_subspace_k2": frac_full_in_lin(2),
                        "fraction_full_variance_in_lin_subspace_k3": frac_full_in_lin(3),
                        "magnitude_ratio_note": "diagnostic_only",
                        "claim_type": "object_matched_structural_alignment",
                    }
                )
    _write_csv(out_root / "covariance_approx" / "twin_linear_covariance_approx.csv", cov_rows)

    _write_csv(
        out_root / "tangent_maps" / "twin_tangent_image_metrics_summary.csv",
        [
            {
                "summary_type": "image_averaged_history_tangent",
                "status": "secondary_summary",
                "n_images": len({obj.image_id for obj in selected_objects}),
            }
        ],
    )

    with (out_root / "tangent_maps" / "twin_tangent_maps.pkl").open("wb") as handle:
        pickle.dump(
            {
                "delta_arcmins": delta_arcmins,
                "object_payload": object_payload,
                "metadata": {
                    "model_ppd": float(args.model_ppd),
                    "history_length_frames": int(ctx.n_lags),
                    "arcmin_to_model_px": float(arcmin_to_model_px),
                    "tangent_derivative_units": "response_per_model_pixel",
                    "response_reduction": "spatial_amax_over_population_rate_map",
                    "finite_difference_shift_convention": "grid_sample_border_align_corners_true",
                    "source_runner": "declan.twin_feature_tangent_structure.run_twin_feature_tangent_structure",
                },
            },
            handle,
        )

    prediction_abs_diffs = np.asarray([float(cast(Any, r["max_abs_diff"])) for r in prediction_rows], dtype=np.float64)
    prediction_corrs = np.asarray([float(cast(Any, r.get("corr_across_units", float("nan")))) for r in prediction_rows], dtype=np.float64)
    r2_vals = np.asarray([float(cast(Any, r["linear_local_r2"])) for r in object_rows], dtype=np.float64)
    r2_x_vals = np.asarray([float(cast(Any, r.get("linear_local_r2_x", float("nan")))) for r in object_rows], dtype=np.float64)
    r2_y_vals = np.asarray([float(cast(Any, r.get("linear_local_r2_y", float("nan")))) for r in object_rows], dtype=np.float64)

    n_valid_union_basis_by_delta = {
        float(cast(Any, r["delta"])): int(cast(Any, r["n_valid_objects_union_basis"]))
        for r in union_basis_diag_rows
        if "delta" in r and "n_valid_objects_union_basis" in r
    }
    compact_rows = [
        r
        for r in union_summary
        if str(r.get("space")) == "raw" and str(r.get("component_set")) == "combined"
    ]
    compact_ok = bool(compact_rows) and all(str(r.get("status", "")) == "ok" and np.isfinite(float(cast(Any, r.get("participation_ratio", float("nan"))))) for r in compact_rows)

    basis_combined_rows = [r for r in basis_rows if str(r.get("tangent_set")) == "combined"]
    basis_ok = False
    if basis_combined_rows:
        deltas_with_valid_basis = set()
        for r in basis_combined_rows:
            try:
                d = float(cast(Any, r["delta"]))
                k = int(cast(Any, r.get("basis_rank_k", -1)))
                rank_train = int(cast(Any, r.get("rank_train", 0)))
            except Exception:
                continue
            if k not in (2, 5, 10, 20):
                continue
            if k > rank_train:
                continue
            if str(r.get("fold_status", "")) != "ok":
                continue
            if np.isfinite(float(cast(Any, r.get("test_variance_captured", float("nan"))))):
                deltas_with_valid_basis.add(d)
        required_deltas = {float(d) for d in delta_arcmins if n_valid_union_basis_by_delta.get(float(d), 0) >= int(min_valid_objects_union_basis)}
        basis_ok = bool(required_deltas) and required_deltas.issubset(deltas_with_valid_basis)

    analysis5_ok = bool(cov_rows) and any(np.isfinite(float(cast(Any, r.get("subspace_overlap_k2", float("nan"))))) for r in cov_rows)
    core_stop_rule_status = "pass" if (compact_ok and basis_ok and analysis5_ok) else "fail"
    n_valid_objects_union_basis = int(min(n_valid_union_basis_by_delta.values())) if n_valid_union_basis_by_delta else 0
    n_dropped_objects_union_basis = int(len(dropped_union_basis_rows))

    if str(args.mode) == "smoke":
        claim_state = "diagnostic_only"
    else:
        claim_state = "core_structural_result_passed" if core_stop_rule_status == "pass" else "diagnostic_only"

    # Compact union summary per delta for manuscript-facing summary JSON.
    union_compactness_by_delta: list[dict[str, object]] = []
    for d_arcmin in delta_arcmins:
        row = next(
            (
                r
                for r in union_summary
                if float(cast(Any, r.get("delta", float("nan")))) == float(d_arcmin)
                and str(r.get("space")) == "raw"
                and str(r.get("component_set")) == "combined"
            ),
            None,
        )
        if row is None:
            union_compactness_by_delta.append(
                {
                    "delta": float(d_arcmin),
                    "status": "missing",
                    "participation_ratio": float("nan"),
                    "null_pr_mean": float("nan"),
                    "null_pr_ci_low": float("nan"),
                    "null_pr_ci_high": float("nan"),
                    "compactness_label": "missing",
                    "n_objects": 0,
                    "rank": 0,
                }
            )
            continue
        union_compactness_by_delta.append(
            {
                "delta": float(d_arcmin),
                "status": str(row.get("status", "")),
                "participation_ratio": float(cast(Any, row.get("participation_ratio", float("nan")))),
                "null_pr_mean": float(cast(Any, row.get("null_pr_mean", float("nan")))),
                "null_pr_ci_low": float(cast(Any, row.get("null_pr_ci_low", float("nan")))),
                "null_pr_ci_high": float(cast(Any, row.get("null_pr_ci_high", float("nan")))),
                "compactness_label": str(row.get("compactness_label", "")),
                "n_objects": int(cast(Any, row.get("n_objects", 0))),
                "rank": int(cast(Any, row.get("rank", 0))),
            }
        )

    # Aggregate train/test basis capture at fixed k per delta.
    train_test_basis_by_delta: list[dict[str, object]] = []
    for d_arcmin in delta_arcmins:
        for k in (2, 5, 10, 20):
            rows_dk = [
                r
                for r in basis_rows
                if float(cast(Any, r.get("delta", float("nan")))) == float(d_arcmin)
                and int(cast(Any, r.get("basis_rank_k", -1))) == int(k)
                and str(r.get("tangent_set", "")) == "combined"
            ]
            rows_ok = [r for r in rows_dk if str(r.get("fold_status", "")) == "ok"]
            caps = _finite_vals([float(cast(Any, r.get("test_variance_captured", float("nan")))) for r in rows_ok])
            null_means = _finite_vals([float(cast(Any, r.get("null_mean", float("nan")))) for r in rows_ok])
            effs = _finite_vals([float(cast(Any, r.get("effect_minus_null", float("nan")))) for r in rows_ok])
            train_test_basis_by_delta.append(
                {
                    "delta": float(d_arcmin),
                    "basis_rank_k": int(k),
                    "n_folds_total": int(len(rows_dk)),
                    "n_folds_ok": int(len(rows_ok)),
                    "test_variance_captured_median": _finite_median(caps),
                    "test_variance_captured_ci_low": float(np.percentile(caps, 2.5)) if caps.size else float("nan"),
                    "test_variance_captured_ci_high": float(np.percentile(caps, 97.5)) if caps.size else float("nan"),
                    "null_mean_median": _finite_median(null_means),
                    "effect_minus_null_median": _finite_median(effs),
                    "status": "ok" if len(rows_ok) > 0 else "not_run_or_no_valid_folds",
                }
            )

    if str(args.mode) == "smoke":
        run_status = "smoke_completed"
    else:
        # Treat reduced-load production configurations as sanity runs.
        production_sanity = (
            int(n_objects) < 64
            or int(folds) < 5
            or int(null_repeats) < 200
            or int(analysis5_n_objects) < min(int(n_objects), 32)
        )
        run_status = "production_sanity_completed" if production_sanity else "production_completed"

    summary = {
        "analysis_name": "twin_feature_tangent_structure",
        "mode": str(args.mode),
        "status": run_status,
        "claim_state": claim_state,
        "subject": str(args.subject),
        "date": str(args.date),
        "dataset_configs_path": str(args.dataset_configs_path),
        "n_canonical_cells": int(ctx.n_units),
        "n_objects_requested": int(n_objects),
        "n_objects_used": int(len(selected_objects)),
        "delta_values_arcmin": [float(v) for v in delta_arcmins],
        "cloud_scales": [float(v) for v in cloud_scales],
        "eye_cloud_mode": str(args.eye_cloud_mode),
        "folds": int(folds),
        "null_repeats": int(null_repeats),
        "model_ppd": float(args.model_ppd),
        "history_length_frames": int(ctx.n_lags),
        "raw_stim_shape": [int(v) for v in stim.shape],
        "image_ids_shape": [int(v) for v in image_ids.shape],
        "eyepos_shape": [int(v) for v in eyepos.shape],
        "prediction_validation_max_abs_diff_median": _finite_median(prediction_abs_diffs),
        "prediction_validation_max_abs_diff_max": float(np.max(_finite_vals(prediction_abs_diffs))) if _finite_vals(prediction_abs_diffs).size else float("nan"),
        "prediction_validation_corr_median": _finite_median(prediction_corrs),
        "linear_local_r2_median": _finite_median(r2_vals),
        "linear_local_r2_iqr": _finite_iqr(r2_vals),
        "linear_local_r2_fraction_below_0p1": float(np.mean(_finite_vals(r2_vals) < 0.1)) if _finite_vals(r2_vals).size else float("nan"),
        "linear_local_r2_x_median": _finite_median(r2_x_vals),
        "linear_local_r2_y_median": _finite_median(r2_y_vals),
        "compact_ok": bool(compact_ok),
        "basis_ok": bool(basis_ok),
        "analysis5_ok": bool(analysis5_ok),
        "core_stop_rule_status": str(core_stop_rule_status),
        "n_valid_objects_union_basis": int(n_valid_objects_union_basis),
        "n_dropped_objects_union_basis": int(n_dropped_objects_union_basis),
        "union_compactness_by_delta": union_compactness_by_delta,
        "train_test_basis_by_delta": train_test_basis_by_delta,
        "outputs": {
            "canonical_unit_manifest": "canonical_unit_manifest.csv",
            "sampled_object_stats": "sampled_object_stats.csv",
            "input_shape_audit": "input_shape_audit.csv",
            "prediction_path_validation": "prediction_path_validation.csv",
            "delta_sensitivity_summary": "delta_sensitivity_summary.csv",
            "union_basis_input_diagnostics": "union_basis_input_diagnostics.csv",
            "dropped_objects_union_basis": "dropped_objects_union_basis.csv",
            "object_composition_audit": "object_composition_audit.csv",
            "fold_leakage_audit": "fold_leakage_audit.csv",
            "history_overlap_audit": "history_overlap_audit.csv",
            "tangent_object_metrics": "tangent_maps/twin_tangent_object_metrics.csv",
            "tangent_image_metrics": "tangent_maps/twin_tangent_image_metrics.csv",
            "tangent_maps": "tangent_maps/twin_tangent_maps.pkl",
            "union_spectrum": "union_spectrum/twin_tangent_union_spectrum.csv",
            "union_summary": "union_spectrum/twin_tangent_union_summary.csv",
            "train_test_basis": "train_test_basis/twin_tangent_train_test_basis.csv",
            "covariance_approx": "covariance_approx/twin_linear_covariance_approx.csv",
            "example_history_png": "figures/tfts_example_history.png",
            "manuscript_report": "MANUSCRIPT_REPORT.md",
        },
        "guardrails": [
            "Diagnostic only: smoke is not production evidence.",
            "Primary unit is the full history object (trial, time index).",
            "Image-averaged summaries are secondary only.",
            "Analysis 5 is object-matched and rank-filtered.",
            "Subspace overlap uses mean squared singular values.",
        ],
    }
    _save_json(out_root / "twin_feature_tangent_summary.json", summary)

    # Build compact manuscript-facing markdown report.
    object_audit_rows = []
    for r in union_basis_diag_rows:
        object_audit_rows.append(
            [
                f"{float(cast(Any, r.get('delta', float('nan')))):.3f}",
                int(cast(Any, r.get("n_objects_total", 0))),
                int(cast(Any, r.get("n_objects_with_finite_bx_by", 0))),
                int(cast(Any, r.get("n_valid_objects_union_basis", 0))),
                int(cast(Any, r.get("n_objects_dropped_nonfinite", 0))),
                int(cast(Any, r.get("zero_norm_bx_count", 0))) + int(cast(Any, r.get("zero_norm_by_count", 0))),
                str(r.get("union_basis_stage_status", "")),
            ]
        )

    union_rows_md = []
    for r in union_compactness_by_delta:
        union_rows_md.append(
            [
                f"{float(cast(Any, r.get('delta', float('nan')))):.3f}",
                int(cast(Any, r.get("n_objects", 0))),
                int(cast(Any, r.get("rank", 0))),
                f"{float(cast(Any, r.get('participation_ratio', float('nan')))):.4f}",
                f"{float(cast(Any, r.get('null_pr_mean', float('nan')))):.4f}",
                f"[{float(cast(Any, r.get('null_pr_ci_low', float('nan')))):.4f}, {float(cast(Any, r.get('null_pr_ci_high', float('nan')))):.4f}]",
                str(r.get("compactness_label", "")),
                str(r.get("status", "")),
            ]
        )

    basis_rows_md = []
    for r in train_test_basis_by_delta:
        basis_rows_md.append(
            [
                f"{float(cast(Any, r.get('delta', float('nan')))):.3f}",
                int(cast(Any, r.get("basis_rank_k", 0))),
                f"{int(cast(Any, r.get('n_folds_ok', 0)))}/{int(cast(Any, r.get('n_folds_total', 0)))}",
                f"{float(cast(Any, r.get('test_variance_captured_median', float('nan')))):.4f}",
                f"[{float(cast(Any, r.get('test_variance_captured_ci_low', float('nan')))):.4f}, {float(cast(Any, r.get('test_variance_captured_ci_high', float('nan')))):.4f}]",
                f"{float(cast(Any, r.get('null_mean_median', float('nan')))):.4f}",
                f"{float(cast(Any, r.get('effect_minus_null_median', float('nan')))):.4f}",
                str(r.get("status", "")),
            ]
        )

    # Analysis 5 locality summary by (delta, cloud_scale).
    locality_rows_md: list[list[object]] = []
    if cov_rows:
        delta_vals = sorted({float(cast(Any, r.get("delta", float("nan")))) for r in cov_rows})
        scale_vals = sorted({float(cast(Any, r.get("cloud_scale", float("nan")))) for r in cov_rows})
        for d in delta_vals:
            for cs in scale_vals:
                block = [r for r in cov_rows if float(cast(Any, r.get("delta", float("nan")))) == d and float(cast(Any, r.get("cloud_scale", float("nan")))) == cs]
                if not block:
                    continue
                overlap_k2 = _finite_vals([float(cast(Any, r.get("subspace_overlap_k2", float("nan")))) for r in block])
                trace_ratio = _finite_vals([float(cast(Any, r.get("trace_ratio_lin_full", float("nan")))) for r in block])
                locality_rows_md.append(
                    [
                        f"{d:.3f}",
                        f"{cs:.2f}",
                        len(block),
                        f"{_finite_median(overlap_k2):.4f}",
                        f"{_finite_median(trace_ratio):.4f}",
                    ]
                )

    # Local R2 / delta stability summary by delta.
    delta_stability_rows_md: list[list[object]] = []
    for d in delta_arcmins:
        rows_d = [r for r in object_rows if float(cast(Any, r.get("delta_arcmin_requested", float("nan")))) == float(d)]
        r2_d = _finite_vals([float(cast(Any, r.get("linear_local_r2", float("nan")))) for r in rows_d])
        bx_d = _finite_vals([float(cast(Any, r.get("norm_bx", float("nan")))) for r in rows_d])
        by_d = _finite_vals([float(cast(Any, r.get("norm_by", float("nan")))) for r in rows_d])
        delta_stability_rows_md.append(
            [
                f"{float(d):.3f}",
                len(rows_d),
                f"{_finite_median(r2_d):.4f}",
                f"{_finite_iqr(r2_d):.4f}",
                f"{(float(np.mean(r2_d < 0.1)) if r2_d.size else float('nan')):.4f}",
                f"{_finite_median(bx_d):.4f}",
                f"{_finite_median(by_d):.4f}",
            ]
        )

    report_lines = [
        "# Twin Feature Tangent Structure: Compact Report",
        "",
        f"- mode: {summary['mode']}",
        f"- status: {summary['status']}",
        f"- claim_state: {summary['claim_state']}",
        f"- core_stop_rule_status: {summary['core_stop_rule_status']}",
        "",
        "## Object Audit",
        _md_table(
            ["delta", "n_total", "n_finite", "n_valid_union_basis", "n_dropped_nonfinite", "n_zero_norm_flags", "stage_status"],
            object_audit_rows,
        ),
        "",
        "## Union Compactness",
        _md_table(
            ["delta", "n_objects", "rank", "PR", "null_mean", "null_95CI", "compactness", "status"],
            union_rows_md,
        ),
        "",
        "## Train/Test Basis",
        _md_table(
            ["delta", "k", "valid_folds", "capture_median", "capture_95CI", "null_median", "effect_median", "status"],
            basis_rows_md,
        ),
        "",
        "## Analysis 5 Locality",
        _md_table(
            ["delta", "cloud_scale", "n_rows", "overlap_k2_median", "trace_ratio_lin_full_median"],
            locality_rows_md,
        ) if locality_rows_md else "No Analysis 5 rows available.",
        "",
        "## Local R2 and Delta Stability",
        _md_table(
            ["delta", "n_objects", "r2_median", "r2_iqr", "frac_r2_lt_0.1", "norm_bx_median", "norm_by_median"],
            delta_stability_rows_md,
        ),
        "",
    ]
    (out_root / "MANUSCRIPT_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    readme_lines = [
        "# Twin Feature-Tangent Structure (TFTS)",
        "",
        f"status: {summary['status']}",
        f"claim_state: {summary['claim_state']}",
        f"n_canonical_cells: {ctx.n_units}",
        f"n_objects_used: {len(selected_objects)}", 
        f"delta_arcmin: {delta_arcmins}",
        f"cloud_scales: {cloud_scales}",
        f"eye_cloud_mode: {args.eye_cloud_mode}",
        "",
        "Primary outputs:",
        "- canonical_unit_manifest.csv",
        "- sampled_object_stats.csv",
        "- input_shape_audit.csv",
        "- prediction_path_validation.csv",
        "- delta_sensitivity_summary.csv",
        "- tangent_maps/twin_tangent_object_metrics.csv",
        "- tangent_maps/twin_tangent_image_metrics.csv",
        "- tangent_maps/twin_tangent_maps.pkl",
        "- union_spectrum/twin_tangent_union_spectrum.csv",
        "- union_spectrum/twin_tangent_union_summary.csv",
        "- train_test_basis/twin_tangent_train_test_basis.csv",
        "- covariance_approx/twin_linear_covariance_approx.csv",
        "- figures/tfts_example_history.png",
    ]
    (out_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "n_units": int(ctx.n_units),
                "n_objects_used": int(len(selected_objects)),
                "prediction_validation_max_abs_diff_median": summary["prediction_validation_max_abs_diff_median"],
                "linear_local_r2_median": summary["linear_local_r2_median"],
                "status": summary["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
