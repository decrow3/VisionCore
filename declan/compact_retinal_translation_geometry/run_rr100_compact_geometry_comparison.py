#!/usr/bin/env python3
"""Compare compact translation geometry in full V1 twin and RR100 medoid twin.

This is a cache-first analysis.  It reuses the full-model finite-difference
tangent maps, applies the saved RR100 population transform to the cached
``r0``, ``bx``, and ``by`` vectors, then asks two questions:

1. Does the reduced population still have compact translation tangent structure?
2. Is that reduced-population structure the same object as the full-model
   compact geometry after the full basis is passed through the same RR100 map?
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd

from declan.redundancy_resolved_v1_population import load_population_view

from .run_compact_retinal_translation_geometry import (
    DEFAULT_TFTS_ROOT,
    VISIONCORE_ROOT,
    _load_tangent_maps,
    _nearest_delta,
    _pca_basis,
    _write_csv,
    _write_json,
)
from .run_static_pc_adjudication import (
    TangentObject,
    _assign_group_folds,
    _as_vector,
    _capture_tangent_fraction,
    _cluster_bootstrap_mean_ci,
    _group_label,
    _meta_label,
    _parse_k_list,
)


RR100_MOVIE_MEDOID_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)
DEFAULT_OUT_ROOT = (
    VISIONCORE_ROOT
    / "outputs"
    / "compact_retinal_translation_geometry"
    / "rr100_compact_geometry_comparison_v1"
)


@dataclass(frozen=True)
class PairedTangentObject:
    object_id: str
    group_id: str
    image_id: str
    trial_index: str
    time_index: str
    full: TangentObject
    rr100: TangentObject


def _stack_tangents(objects: list[TangentObject]) -> np.ndarray:
    return np.stack(
        [row for obj in objects for row in (np.asarray(obj.bx), np.asarray(obj.by))],
        axis=0,
    )


def _stack_static(objects: list[TangentObject]) -> np.ndarray:
    return np.stack([np.asarray(obj.r0) for obj in objects], axis=0)


def _fit_basis(rows: np.ndarray, *, max_k: int) -> np.ndarray:
    _, basis = _pca_basis(np.asarray(rows, dtype=np.float64), n_components=int(max_k))
    return np.asarray(basis, dtype=np.float64)


def _orthonormalize(mat: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"basis candidate must be 2D, got shape={mat.shape}")
    if mat.size == 0 or mat.shape[1] == 0:
        return np.zeros((mat.shape[0], 0), dtype=np.float64)
    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    if s.size == 0:
        return np.zeros((mat.shape[0], 0), dtype=np.float64)
    keep = s > max(float(tol), float(s[0]) * float(tol))
    return np.asarray(u[:, keep], dtype=np.float64)


def _random_basis(n_units: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.normal(size=(int(n_units), int(n_dims)))
    q, r = np.linalg.qr(mat, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs[None, :]


def _restrict_basis_to_rr100(
    membership: np.ndarray,
    full_basis: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    return _orthonormalize(np.asarray(membership, dtype=np.float64) @ full_basis[:, : int(k)])


def _spectrum_metrics(rows: np.ndarray, *, max_k: int) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.float64)
    rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
    centered = rows - np.mean(rows, axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = s**2
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        normalized = np.zeros_like(eigenvalues)
        participation_ratio = float("nan")
    else:
        normalized = eigenvalues / total
        participation_ratio = float(total**2 / np.sum(eigenvalues**2))
    cumulative = np.cumsum(normalized)

    out: dict[str, Any] = {
        "n_rows": int(rows.shape[0]),
        "n_units": int(rows.shape[1]),
        "rank": int(np.sum(s > max(1e-12, float(s[0]) * 1e-12))) if s.size else 0,
        "total_centered_energy": total,
        "participation_ratio": participation_ratio,
    }
    for threshold in (0.5, 0.8, 0.9, 0.95):
        if cumulative.size and float(cumulative[-1]) >= threshold:
            out[f"dims_for_{int(threshold * 100)}pct"] = int(np.searchsorted(cumulative, threshold) + 1)
        else:
            out[f"dims_for_{int(threshold * 100)}pct"] = 0
    for k in range(1, int(max_k) + 1):
        out[f"cumulative_capture_k{k}"] = (
            float(cumulative[k - 1]) if k <= cumulative.size else float("nan")
        )
    return out


def _principal_cosines(a: np.ndarray, b: np.ndarray, *, k: int) -> tuple[np.ndarray, int]:
    qa = _orthonormalize(np.asarray(a, dtype=np.float64)[:, : int(k)])
    qb = _orthonormalize(np.asarray(b, dtype=np.float64)[:, : int(k)])
    k_eff = int(min(qa.shape[1], qb.shape[1]))
    if k_eff <= 0:
        return np.asarray([], dtype=np.float64), 0
    cosines = np.linalg.svd(qa[:, :k_eff].T @ qb[:, :k_eff], compute_uv=False)
    return np.asarray(cosines, dtype=np.float64), k_eff


def _subspace_overlap_record(
    *,
    comparison: str,
    fold_id: str | int,
    k: int,
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    n_train_objects: int,
    n_test_objects: int,
) -> dict[str, Any]:
    cosines, k_eff = _principal_cosines(basis_a, basis_b, k=int(k))
    squared = cosines**2
    return {
        "comparison": comparison,
        "fold_id": fold_id,
        "k": int(k),
        "k_effective": int(k_eff),
        "mean_principal_cosine": float(np.mean(cosines)) if cosines.size else float("nan"),
        "mean_squared_principal_cosine": float(np.mean(squared)) if squared.size else float("nan"),
        "min_principal_cosine": float(np.min(cosines)) if cosines.size else float("nan"),
        "max_principal_cosine": float(np.max(cosines)) if cosines.size else float("nan"),
        "principal_cosines": ";".join(f"{float(v):.8g}" for v in cosines),
        "n_train_objects": int(n_train_objects),
        "n_test_objects": int(n_test_objects),
    }


def _collect_paired_objects(
    *,
    tfts_root: Path,
    requested_delta: float,
    group_by: str,
    membership: np.ndarray,
) -> tuple[float, list[PairedTangentObject], int, list[dict[str, Any]]]:
    deltas, payload_by_delta = _load_tangent_maps(tfts_root)
    delta = _nearest_delta(deltas, requested_delta)
    payload = payload_by_delta[delta]
    transform = np.asarray(membership, dtype=np.float64)
    if transform.ndim != 2:
        raise ValueError(f"RR100 membership must be 2D, got shape={transform.shape}")

    objects: list[PairedTangentObject] = []
    skipped: list[dict[str, Any]] = []
    n_full_units: int | None = None
    for object_id, meta in sorted(payload.items()):
        try:
            r0 = _as_vector(meta["r0"], name="r0", object_id=object_id)
            bx = _as_vector(meta["bx"], name="bx", object_id=object_id)
            by = _as_vector(meta["by"], name="by", object_id=object_id)
        except ValueError as exc:
            skipped.append(
                {
                    "delta_arcmin": float(delta),
                    "object_id": str(object_id),
                    "skip_reason": str(exc),
                }
            )
            continue
        if not (r0.shape == bx.shape == by.shape):
            skipped.append(
                {
                    "delta_arcmin": float(delta),
                    "object_id": str(object_id),
                    "skip_reason": (
                        "mismatched vector shapes: "
                        f"r0={r0.shape}, bx={bx.shape}, by={by.shape}"
                    ),
                }
            )
            continue
        if n_full_units is None:
            n_full_units = int(r0.size)
            if transform.shape[1] != n_full_units:
                raise ValueError(
                    f"RR100 membership expects {transform.shape[1]} input channels, "
                    f"but tangent cache has {n_full_units}"
                )
        elif int(r0.size) != n_full_units:
            raise ValueError(f"Object {object_id} has {r0.size} units, expected {n_full_units}")
        energy = float(np.dot(bx, bx) + np.dot(by, by))
        if energy <= 0.0:
            skipped.append(
                {
                    "delta_arcmin": float(delta),
                    "object_id": str(object_id),
                    "skip_reason": "zero combined full tangent energy",
                }
            )
            continue

        rr_bx = transform @ bx
        rr_by = transform @ by
        rr_energy = float(np.dot(rr_bx, rr_bx) + np.dot(rr_by, rr_by))
        if rr_energy <= 0.0:
            skipped.append(
                {
                    "delta_arcmin": float(delta),
                    "object_id": str(object_id),
                    "skip_reason": "zero combined RR100 tangent energy",
                }
            )
            continue

        group_id = _group_label(meta, str(object_id), group_by)
        image_id = _meta_label(meta, "image_id", "")
        trial_index = _meta_label(meta, "trial_index", "")
        time_index = _meta_label(meta, "time_index", "")
        full = TangentObject(
            object_id=str(object_id),
            group_id=group_id,
            image_id=image_id,
            trial_index=trial_index,
            time_index=time_index,
            r0=r0,
            bx=bx,
            by=by,
        )
        rr100 = TangentObject(
            object_id=str(object_id),
            group_id=group_id,
            image_id=image_id,
            trial_index=trial_index,
            time_index=time_index,
            r0=transform @ r0,
            bx=rr_bx,
            by=rr_by,
        )
        objects.append(
            PairedTangentObject(
                object_id=str(object_id),
                group_id=group_id,
                image_id=image_id,
                trial_index=trial_index,
                time_index=time_index,
                full=full,
                rr100=rr100,
            )
        )

    if not objects or n_full_units is None:
        raise ValueError(f"No usable tangent-map objects found for delta={delta}.")
    return float(delta), objects, n_full_units, skipped


def _append_capture_rows(
    rows: list[dict[str, Any]],
    *,
    delta: float,
    test_objects: list[TangentObject],
    population: str,
    basis: np.ndarray,
    basis_type: str,
    basis_training_source: str,
    basis_fit_scope: str,
    fold_id: int,
    k_list: list[int],
    random_id: str,
    n_train_objects: int,
    n_test_objects: int,
    n_train_groups: int,
    n_test_groups: int,
) -> None:
    for obj in test_objects:
        energy_bx = float(np.dot(obj.bx, obj.bx))
        energy_by = float(np.dot(obj.by, obj.by))
        for k in k_list:
            capture, capture_bx, capture_by, k_eff = _capture_tangent_fraction(
                obj.bx,
                obj.by,
                basis,
                k=int(k),
            )
            rows.append(
                {
                    "delta_arcmin": float(delta),
                    "population": population,
                    "object_id": obj.object_id,
                    "group_id": obj.group_id,
                    "image_id": obj.image_id,
                    "trial_index": obj.trial_index,
                    "time_index": obj.time_index,
                    "fold_id": int(fold_id),
                    "basis_type": basis_type,
                    "basis_training_source": basis_training_source,
                    "basis_fit_scope": basis_fit_scope,
                    "random_id": random_id,
                    "k": int(k),
                    "k_effective": int(k_eff),
                    "capture_combined": capture,
                    "capture_bx": capture_bx,
                    "capture_by": capture_by,
                    "energy_combined": energy_bx + energy_by,
                    "energy_bx": energy_bx,
                    "energy_by": energy_by,
                    "n_train_objects": int(n_train_objects),
                    "n_test_objects": int(n_test_objects),
                    "n_train_groups": int(n_train_groups),
                    "n_test_groups": int(n_test_groups),
                }
            )


def _score_fold(
    *,
    capture_rows: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
    paired_objects: list[PairedTangentObject],
    group_to_fold: dict[str, int],
    fold_id: int,
    delta: float,
    membership: np.ndarray,
    k_list: list[int],
    n_random_basis: int,
    seed: int,
    group_by: str,
) -> None:
    train = [obj for obj in paired_objects if group_to_fold[obj.group_id] != fold_id]
    test = [obj for obj in paired_objects if group_to_fold[obj.group_id] == fold_id]
    if not train or not test:
        raise ValueError(f"Fold {fold_id} has empty train or test split")

    train_groups = {obj.group_id for obj in train}
    test_groups = {obj.group_id for obj in test}
    full_train = [obj.full for obj in train]
    rr_train = [obj.rr100 for obj in train]
    full_test = [obj.full for obj in test]
    rr_test = [obj.rr100 for obj in test]

    max_k = int(max(k_list))
    full_basis = _fit_basis(_stack_tangents(full_train), max_k=max_k)
    rr_basis = _fit_basis(_stack_tangents(rr_train), max_k=max_k)
    rr_static_basis = _fit_basis(_stack_static(rr_train), max_k=max_k)
    rng = np.random.default_rng(int(seed) + 1009 * int(fold_id))
    basis_fit_scope = f"fold_disjoint_by_{group_by}"

    common = {
        "delta": float(delta),
        "fold_id": int(fold_id),
        "k_list": k_list,
        "n_train_objects": len(train),
        "n_test_objects": len(test),
        "n_train_groups": len(train_groups),
        "n_test_groups": len(test_groups),
        "basis_fit_scope": basis_fit_scope,
    }
    _append_capture_rows(
        capture_rows,
        test_objects=full_test,
        population="full756",
        basis=full_basis,
        basis_type="full_tangent_pc",
        basis_training_source="fold_train_full_bx_by",
        random_id="observed",
        **common,
    )
    _append_capture_rows(
        capture_rows,
        test_objects=rr_test,
        population="rr100",
        basis=rr_basis,
        basis_type="rr100_tangent_pc",
        basis_training_source="fold_train_rr100_bx_by",
        random_id="observed",
        **common,
    )
    for k in k_list:
        _append_capture_rows(
            capture_rows,
            test_objects=rr_test,
            population="rr100",
            basis=_restrict_basis_to_rr100(membership, full_basis, k=int(k)),
            basis_type="full_tangent_pc_restricted_to_rr100",
            basis_training_source="fold_train_full_topk_bx_by_then_rr100_membership",
            random_id="observed",
            **{**common, "k_list": [int(k)]},
        )
    _append_capture_rows(
        capture_rows,
        test_objects=rr_test,
        population="rr100",
        basis=rr_static_basis,
        basis_type="rr100_static_response_pc",
        basis_training_source="fold_train_rr100_r0",
        random_id="observed",
        **common,
    )
    for draw in range(int(n_random_basis)):
        _append_capture_rows(
            capture_rows,
            test_objects=rr_test,
            population="rr100",
            basis=_random_basis(np.asarray(membership).shape[0], max_k, rng),
            basis_type="rr100_random_orthonormal",
            basis_training_source="isotropic_random_rr100_subspace",
            random_id=f"random_{draw:03d}",
            **common,
        )

    for k in k_list:
        full_to_rr_basis = _restrict_basis_to_rr100(membership, full_basis, k=int(k))
        overlap_rows.append(
            _subspace_overlap_record(
                comparison="rr100_native_vs_full_restricted",
                fold_id=int(fold_id),
                k=int(k),
                basis_a=rr_basis,
                basis_b=full_to_rr_basis,
                n_train_objects=len(train),
                n_test_objects=len(test),
            )
        )
        overlap_rows.append(
            _subspace_overlap_record(
                comparison="rr100_native_vs_rr100_static_pc",
                fold_id=int(fold_id),
                k=int(k),
                basis_a=rr_basis,
                basis_b=rr_static_basis,
                n_train_objects=len(train),
                n_test_objects=len(test),
            )
        )


def _summary_rows(object_df: pd.DataFrame, *, n_bootstrap: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_cols = ["population", "basis_type", "k"]
    for (population, basis_type, k), block in object_df.groupby(group_cols, sort=True):
        mean, lo, hi, _ = _cluster_bootstrap_mean_ci(
            block,
            value_col="capture_combined",
            group_col="group_id",
            n_bootstrap=int(n_bootstrap),
            seed=int(seed) + 17 * int(k) + sum(ord(ch) for ch in str(basis_type)),
        )
        rows.append(
            {
                "population": str(population),
                "basis_type": str(basis_type),
                "k": int(k),
                "capture_combined_mean": mean,
                "capture_combined_ci_low": lo,
                "capture_combined_ci_high": hi,
                "n_objects": int(block["object_id"].nunique()),
                "n_groups": int(block["group_id"].nunique()),
                "n_basis_draws_median": float(block["n_basis_draws"].median()),
            }
        )
    return rows


def _draw_averaged_object_frame(capture_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "delta_arcmin",
        "population",
        "object_id",
        "group_id",
        "image_id",
        "trial_index",
        "time_index",
        "fold_id",
        "basis_type",
        "basis_training_source",
        "basis_fit_scope",
        "k",
    ]
    return (
        capture_df.groupby(group_cols, dropna=False)
        .agg(
            k_effective=("k_effective", "min"),
            capture_combined=("capture_combined", "mean"),
            capture_bx=("capture_bx", "mean"),
            capture_by=("capture_by", "mean"),
            energy_combined=("energy_combined", "first"),
            energy_bx=("energy_bx", "first"),
            energy_by=("energy_by", "first"),
            n_basis_draws=("random_id", "nunique"),
            n_train_objects=("n_train_objects", "first"),
            n_test_objects=("n_test_objects", "first"),
            n_train_groups=("n_train_groups", "first"),
            n_test_groups=("n_test_groups", "first"),
        )
        .reset_index()
    )


def _paired_difference_rows(
    object_df: pd.DataFrame,
    *,
    lhs_basis: str,
    rhs_basises: list[str],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lhs = object_df[object_df["basis_type"].astype(str) == lhs_basis].copy()
    merge_cols = ["population", "delta_arcmin", "object_id", "group_id", "fold_id", "k"]
    for rhs_basis in rhs_basises:
        rhs = object_df[object_df["basis_type"].astype(str) == rhs_basis].copy()
        paired = lhs[merge_cols + ["capture_combined"]].merge(
            rhs[merge_cols + ["capture_combined"]],
            on=merge_cols,
            suffixes=("_lhs", "_rhs"),
            how="inner",
        )
        if paired.empty:
            continue
        paired["diff_lhs_minus_rhs"] = (
            paired["capture_combined_lhs"] - paired["capture_combined_rhs"]
        )
        for k, block in paired.groupby("k", sort=True):
            mean, lo, hi, boot = _cluster_bootstrap_mean_ci(
                block,
                value_col="diff_lhs_minus_rhs",
                group_col="group_id",
                n_bootstrap=int(n_bootstrap),
                seed=int(seed)
                + 101 * int(k)
                + sum(ord(ch) for ch in str(rhs_basis))
                + sum(ord(ch) for ch in lhs_basis),
            )
            if boot.size:
                p_two_sided = float(
                    min(1.0, 2.0 * min(np.mean(boot <= 0.0), np.mean(boot >= 0.0)))
                )
            else:
                p_two_sided = float("nan")
            rows.append(
                {
                    "population": "rr100",
                    "lhs_basis_type": lhs_basis,
                    "rhs_basis_type": str(rhs_basis),
                    "k": int(k),
                    "mean_lhs_minus_rhs": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "bootstrap_p_two_sided_about_zero": p_two_sided,
                    "fraction_objects_lhs_gt_rhs": float(np.mean(block["diff_lhs_minus_rhs"] > 0.0)),
                    "n_objects": int(block["object_id"].nunique()),
                    "n_groups": int(block["group_id"].nunique()),
                }
            )
    return rows


def _overlap_summary_rows(
    overlap_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    if overlap_df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (comparison, k), block in overlap_df.groupby(["comparison", "k"], sort=True):
        for value_col in [
            "mean_principal_cosine",
            "mean_squared_principal_cosine",
            "min_principal_cosine",
        ]:
            mean, lo, hi, _ = _cluster_bootstrap_mean_ci(
                block,
                value_col=value_col,
                group_col="fold_id",
                n_bootstrap=int(n_bootstrap),
                seed=int(seed) + int(k) * 211 + sum(ord(ch) for ch in str(comparison) + value_col),
            )
            rows.append(
                {
                    "comparison": str(comparison),
                    "k": int(k),
                    "metric": value_col,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_folds": int(block["fold_id"].nunique()),
                }
            )
    return rows


def _selected_channels_from_membership(membership: np.ndarray, *, tol: float = 1e-8) -> np.ndarray:
    membership = np.asarray(membership, dtype=np.float64)
    if membership.ndim != 2 or not np.all(np.isfinite(membership)):
        raise ValueError("Random-subset comparison expects a finite 2D membership matrix")
    near_zero = np.isclose(membership, 0.0, atol=float(tol), rtol=0.0)
    near_one = np.isclose(membership, 1.0, atol=float(tol), rtol=0.0)
    if (
        np.any(membership < -float(tol))
        or np.any(membership > 1.0 + float(tol))
        or not np.all(near_zero | near_one)
        or not np.all(np.sum(near_one, axis=1) == 1)
    ):
        raise ValueError("Random-subset comparison currently expects one-hot medoid membership")
    selected = np.argmax(membership, axis=1)
    if len(set(int(v) for v in selected)) != selected.size:
        raise ValueError("RR100 membership selected duplicate input channels")
    return selected.astype(int)


def _random_subset_rows(
    *,
    full_objects: list[TangentObject],
    membership: np.ndarray,
    group_to_fold: dict[str, int],
    n_folds: int,
    group_by: str,
    k_list: list[int],
    n_random_subsets: int,
    seed: int,
) -> list[dict[str, Any]]:
    if int(n_random_subsets) <= 0:
        return []
    selected = _selected_channels_from_membership(membership)
    n_units = int(full_objects[0].r0.size)
    subset_size = selected.size
    rng = np.random.default_rng(int(seed) + 700000)
    rows: list[dict[str, Any]] = []
    fold_cache: dict[int, dict[str, Any]] = {}
    for fold_id in range(int(n_folds)):
        train_objects = [obj for obj in full_objects if group_to_fold[obj.group_id] != fold_id]
        test_objects = [obj for obj in full_objects if group_to_fold[obj.group_id] == fold_id]
        train_groups = {obj.group_id for obj in train_objects}
        test_groups = {obj.group_id for obj in test_objects}
        if not train_objects or not test_objects:
            raise ValueError(f"Fold {fold_id} has empty train or test split")
        full_train_rows = _stack_tangents(train_objects)
        fold_cache[int(fold_id)] = {
            "full_train_rows": full_train_rows,
            "full_test_rows": _stack_tangents(test_objects),
            "full_train_basis": _fit_basis(full_train_rows, max_k=max(k_list)),
            "n_train_objects": len(train_objects),
            "n_test_objects": len(test_objects),
            "n_train_groups": len(train_groups),
            "n_test_groups": len(test_groups),
        }

    subset_specs = [("rr100_selected", selected)]
    for draw in range(int(n_random_subsets)):
        subset_specs.append((f"random_{draw:03d}", rng.choice(n_units, size=subset_size, replace=False)))

    def append_subset(label: str, subset: np.ndarray, *, fold_id: int) -> None:
        subset = np.asarray(subset, dtype=int)
        cache = fold_cache[int(fold_id)]
        full_train_rows = cache["full_train_rows"]
        full_train_basis = cache["full_train_basis"]
        subset_train_rows = full_train_rows[:, subset]
        subset_test_rows = cache["full_test_rows"][:, subset]
        subset_basis = _fit_basis(subset_train_rows, max_k=max(k_list))
        metrics = _spectrum_metrics(subset_train_rows, max_k=max(k_list))
        for k in k_list:
            full_restricted = _orthonormalize(full_train_basis[subset, : int(k)])
            cosines, k_eff = _principal_cosines(subset_basis, full_restricted, k=int(k))
            subset_capture = _capture_rows_fraction(subset_test_rows, subset_basis, k=int(k))
            full_restricted_capture = _capture_rows_fraction(
                subset_test_rows,
                full_restricted,
                k=int(k),
            )
            rows.append(
                {
                    "subset_id": label,
                    "fold_id": int(fold_id),
                    "k": int(k),
                    "k_effective_overlap": int(k_eff),
                    "is_rr100_selected_subset": bool(label == "rr100_selected"),
                    "participation_ratio": metrics["participation_ratio"],
                    "dims_for_80pct": metrics["dims_for_80pct"],
                    "dims_for_90pct": metrics["dims_for_90pct"],
                    "subset_native_capture": subset_capture,
                    "full_restricted_capture": full_restricted_capture,
                    "mean_principal_cosine_to_full_restricted": (
                        float(np.mean(cosines)) if cosines.size else float("nan")
                    ),
                    "mean_squared_principal_cosine_to_full_restricted": (
                        float(np.mean(cosines**2)) if cosines.size else float("nan")
                    ),
                    "min_principal_cosine_to_full_restricted": (
                        float(np.min(cosines)) if cosines.size else float("nan")
                    ),
                    "subset_size": int(subset_size),
                    "n_input_units": int(n_units),
                    "n_train_objects": int(cache["n_train_objects"]),
                    "n_test_objects": int(cache["n_test_objects"]),
                    "n_train_groups": int(cache["n_train_groups"]),
                    "n_test_groups": int(cache["n_test_groups"]),
                    "basis_fit_scope": f"fold_disjoint_by_{group_by}",
                }
            )

    for label, subset in subset_specs:
        for fold_id in range(int(n_folds)):
            append_subset(label, subset, fold_id=int(fold_id))
    return rows


def _capture_rows_fraction(rows: np.ndarray, basis: np.ndarray, *, k: int) -> float:
    rows = np.asarray(rows, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    k_eff = int(min(int(k), basis.shape[1]))
    energy = float(np.sum(rows**2))
    if k_eff <= 0 or energy <= 0.0:
        return float("nan")
    projected = rows @ basis[:, :k_eff]
    return float(np.sum(projected**2) / energy)


def _write_report(
    *,
    out_root: Path,
    config: dict[str, Any],
    compactness: pd.DataFrame,
    summary: pd.DataFrame,
    diffs: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    random_subsets: pd.DataFrame,
) -> None:
    k_values = sorted(int(k) for k in summary["k"].unique()) if not summary.empty else [10]
    key_k = 10 if 10 in k_values else k_values[len(k_values) // 2]

    def fmt(value: Any) -> str:
        try:
            val = float(value)
        except Exception:
            return "nan"
        return f"{val:.3f}" if np.isfinite(val) else "nan"

    def summary_line(basis_type: str) -> str:
        row = summary[
            (summary["population"].astype(str) == "rr100")
            & (summary["basis_type"].astype(str) == basis_type)
            & (summary["k"].astype(int) == key_k)
        ]
        if row.empty:
            return f"- `{basis_type}`: unavailable"
        rec = row.iloc[0]
        return (
            f"- `{basis_type}`: {fmt(rec['capture_combined_mean'])} "
            f"[{fmt(rec['capture_combined_ci_low'])}, {fmt(rec['capture_combined_ci_high'])}]"
        )

    overlap_key = overlap_summary[
        (overlap_summary["comparison"].astype(str) == "rr100_native_vs_full_restricted")
        & (overlap_summary["k"].astype(int) == key_k)
        & (overlap_summary["metric"].astype(str) == "mean_principal_cosine")
    ]
    if overlap_key.empty:
        overlap_text = "rr100/native-vs-full-restricted overlap unavailable"
    else:
        rec = overlap_key.iloc[0]
        overlap_text = (
            f"mean principal cosine to restricted full basis at k={key_k}: "
            f"{fmt(rec['mean'])} [{fmt(rec['ci_low'])}, {fmt(rec['ci_high'])}]"
        )

    diff_key = diffs[
        (diffs["rhs_basis_type"].astype(str) == "full_tangent_pc_restricted_to_rr100")
        & (diffs["k"].astype(int) == key_k)
    ]
    if diff_key.empty:
        diff_text = "rr100 native minus restricted-full capture unavailable"
    else:
        rec = diff_key.iloc[0]
        diff_text = (
            f"rr100 native minus restricted-full capture at k={key_k}: "
            f"{fmt(rec['mean_lhs_minus_rhs'])} "
            f"[{fmt(rec['ci_low'])}, {fmt(rec['ci_high'])}]"
        )

    random_text = "random subset control unavailable"
    if not random_subsets.empty:
        key_random = random_subsets[random_subsets["k"].astype(int) == key_k].copy()
        rr = key_random[key_random["subset_id"].astype(str) == "rr100_selected"]
        rnd = key_random[key_random["subset_id"].astype(str) != "rr100_selected"]
        if not rr.empty and not rnd.empty:
            metric = "mean_principal_cosine_to_full_restricted"
            random_text = (
                f"fold-disjoint RR100 selected subset {metric} at k={key_k}: "
                f"{fmt(rr[metric].mean())}; random subset mean: {fmt(rnd[metric].mean())}"
            )

    compact_lines = []
    for population in ("full756", "rr100"):
        row = compactness[compactness["population"].astype(str) == population]
        if row.empty:
            continue
        rec = row.iloc[0]
        compact_lines.append(
            f"- `{population}` tangent PR: {fmt(rec['participation_ratio'])}; "
            f"dims for 90 percent: `{int(rec['dims_for_90pct'])}`"
        )

    text = "\n".join(
        [
            "# RR100 Compact Geometry Comparison",
            "",
            "This run asks whether the RR100 movie-medoid twin preserves the compact retinal-translation tangent geometry found in the full 756-channel twin, and whether the reduced object is the full object after applying the same medoid bottleneck.",
            "",
            "## Configuration",
            "",
            f"- tangent source: `{config['tangent_source']}`",
            f"- RR100 version: `{config['population_version_name']}`",
            f"- finite-difference step: `{config['delta_arcmin']}` arcmin",
            f"- objects: `{config['n_objects']}`",
            f"- split group: `{config['group_by']}`",
            f"- folds: `{config['n_folds']}`",
            f"- random RR100 bases: `{config['n_random_basis']}`",
            f"- fold-disjoint random 100-channel subsets: `{config['n_random_subsets']}`",
            "",
            "## Compactness",
            "",
            *compact_lines,
            "",
            f"## Key Readout at k={key_k}",
            "",
            summary_line("rr100_tangent_pc"),
            summary_line("full_tangent_pc_restricted_to_rr100"),
            summary_line("rr100_static_response_pc"),
            summary_line("rr100_random_orthonormal"),
            "",
            f"- {diff_text}",
            f"- {overlap_text}",
            f"- {random_text}",
            "",
            "## Files",
            "",
            "- `rr100_compactness_summary.csv`: global tangent spectra for full and RR100 populations.",
            "- `rr100_generalization_object_capture.csv`: per-object held-out capture rows.",
            "- `rr100_generalization_object_capture_draw_averaged.csv`: draw-averaged capture rows.",
            "- `rr100_generalization_summary.csv`: clustered-bootstrap capture summaries.",
            "- `rr100_paired_differences.csv`: paired RR100-native minus control contrasts.",
            "- `rr100_subspace_overlap.csv`: fold/global principal-angle rows.",
            "- `rr100_subspace_overlap_summary.csv`: bootstrapped overlap summaries.",
            "- `rr100_random_subset_summary.csv`: fold-disjoint RR100 selected channels versus random 100-channel subsets.",
            "",
        ]
    )
    (out_root / "rr100_compact_geometry_comparison_report.md").write_text(
        text + "\n",
        encoding="utf-8",
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    tfts_root = Path(args.tfts_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    k_list = _parse_k_list(str(args.k_list))
    max_k = max(k_list)

    view = load_population_view(version_name=str(args.population_version_name))
    if view.membership is None:
        raise ValueError(f"Population view {view.name!r} does not include membership")
    membership = np.asarray(view.membership, dtype=np.float64)
    delta, paired_objects, n_full_units, skipped = _collect_paired_objects(
        tfts_root=tfts_root,
        requested_delta=float(args.delta),
        group_by=str(args.group_by),
        membership=membership,
    )
    _write_csv(out_root / "rr100_skipped_objects.csv", skipped)

    full_objects = [obj.full for obj in paired_objects]
    rr_objects = [obj.rr100 for obj in paired_objects]
    compact_rows = []
    for population, objects in [("full756", full_objects), ("rr100", rr_objects)]:
        row = {
            "population": population,
            "delta_arcmin": float(delta),
            "n_objects": int(len(objects)),
            "n_tangent_rows": int(2 * len(objects)),
        }
        row.update(_spectrum_metrics(_stack_tangents(objects), max_k=max_k))
        compact_rows.append(row)
    _write_csv(out_root / "rr100_compactness_summary.csv", compact_rows)

    group_to_fold, fold_rows = _assign_group_folds(
        rr_objects,
        n_folds=int(args.n_folds),
        seed=int(args.seed),
    )
    _write_csv(out_root / "rr100_folds.csv", fold_rows)

    capture_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    full_basis_all = _fit_basis(_stack_tangents(full_objects), max_k=max_k)
    rr_basis_all = _fit_basis(_stack_tangents(rr_objects), max_k=max_k)
    rr_static_basis_all = _fit_basis(_stack_static(rr_objects), max_k=max_k)
    for k in k_list:
        full_to_rr_basis_all = _restrict_basis_to_rr100(membership, full_basis_all, k=int(k))
        overlap_rows.append(
            _subspace_overlap_record(
                comparison="global_rr100_native_vs_full_restricted",
                fold_id="all",
                k=int(k),
                basis_a=rr_basis_all,
                basis_b=full_to_rr_basis_all,
                n_train_objects=len(paired_objects),
                n_test_objects=0,
            )
        )
        overlap_rows.append(
            _subspace_overlap_record(
                comparison="global_rr100_native_vs_rr100_static_pc",
                fold_id="all",
                k=int(k),
                basis_a=rr_basis_all,
                basis_b=rr_static_basis_all,
                n_train_objects=len(paired_objects),
                n_test_objects=0,
            )
        )

    for fold_id in range(int(args.n_folds)):
        _score_fold(
            capture_rows=capture_rows,
            overlap_rows=overlap_rows,
            paired_objects=paired_objects,
            group_to_fold=group_to_fold,
            fold_id=fold_id,
            delta=delta,
            membership=membership,
            k_list=k_list,
            n_random_basis=int(args.n_random_basis),
            seed=int(args.seed),
            group_by=str(args.group_by),
        )
    _write_csv(out_root / "rr100_generalization_object_capture.csv", capture_rows)
    _write_csv(out_root / "rr100_subspace_overlap.csv", overlap_rows)

    capture_df = pd.DataFrame(capture_rows)
    object_df = _draw_averaged_object_frame(capture_df)
    _write_csv(
        out_root / "rr100_generalization_object_capture_draw_averaged.csv",
        object_df.to_dict(orient="records"),
    )
    summary = pd.DataFrame(
        _summary_rows(
            object_df,
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed),
        )
    )
    diffs = pd.DataFrame(
        _paired_difference_rows(
            object_df[object_df["population"].astype(str) == "rr100"].copy(),
            lhs_basis="rr100_tangent_pc",
            rhs_basises=[
                "full_tangent_pc_restricted_to_rr100",
                "rr100_static_response_pc",
                "rr100_random_orthonormal",
            ],
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed) + 50000,
        )
    )
    overlap_summary = pd.DataFrame(
        _overlap_summary_rows(
            pd.DataFrame(overlap_rows),
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed) + 90000,
        )
    )
    random_subset_rows = _random_subset_rows(
        full_objects=full_objects,
        membership=membership,
        group_to_fold=group_to_fold,
        n_folds=int(args.n_folds),
        group_by=str(args.group_by),
        k_list=k_list,
        n_random_subsets=int(args.n_random_subsets),
        seed=int(args.seed),
    )
    random_subsets = pd.DataFrame(random_subset_rows)

    _write_csv(out_root / "rr100_generalization_summary.csv", summary.to_dict(orient="records"))
    _write_csv(out_root / "rr100_paired_differences.csv", diffs.to_dict(orient="records"))
    _write_csv(
        out_root / "rr100_subspace_overlap_summary.csv",
        overlap_summary.to_dict(orient="records"),
    )
    _write_csv(out_root / "rr100_random_subset_summary.csv", random_subset_rows)

    config = {
        "tangent_source": str((tfts_root / "tangent_maps" / "twin_tangent_maps.pkl").resolve()),
        "out_root": str(out_root.resolve()),
        "population_version_name": str(args.population_version_name),
        "population_name": str(view.name),
        "delta_arcmin": float(delta),
        "requested_delta_arcmin": float(args.delta),
        "group_by": str(args.group_by),
        "n_objects": int(len(paired_objects)),
        "n_skipped_objects": int(len(skipped)),
        "n_groups": int(len({obj.group_id for obj in paired_objects})),
        "n_full_units": int(n_full_units),
        "n_rr100_units": int(membership.shape[0]),
        "n_folds": int(args.n_folds),
        "k_list": k_list,
        "n_random_basis": int(args.n_random_basis),
        "n_random_subsets": int(args.n_random_subsets),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
    }
    _write_json(out_root / "rr100_compact_geometry_comparison_config.json", config)
    _write_report(
        out_root=out_root,
        config=config,
        compactness=pd.DataFrame(compact_rows),
        summary=summary,
        diffs=diffs,
        overlap_summary=overlap_summary,
        random_subsets=random_subsets,
    )
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfts-root", type=Path, default=DEFAULT_TFTS_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--population-version-name", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--k-list", type=str, default="2,5,10,20,30")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--group-by",
        choices=["image_id", "trial_index", "object_id"],
        default="image_id",
        help="Disjoint split group and bootstrap cluster.",
    )
    parser.add_argument("--n-random-basis", type=int, default=64)
    parser.add_argument("--n-random-subsets", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = analyze(args)
    out_root = Path(config["out_root"])
    print(f"Wrote RR100 compact-geometry comparison to {out_root}")
    print(
        f"Objects: {config['n_objects']} across {config['n_groups']} "
        f"{config['group_by']} groups; RR units: {config['n_rr100_units']}"
    )


if __name__ == "__main__":
    main()
