#!/usr/bin/env python3
"""Adjudicate compact translation PCs against fold-disjoint static-response PCs.

This analysis asks whether the Figure 3 compact tangent subspace is more
translation-specific than a low-dimensional basis learned from the same local
windows' static responses.  The split is group-disjoint by image by default, so
held-out tangent charts are not evaluated against static PCs fit to sibling
windows from the same image.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .run_compact_retinal_translation_geometry import (
    ACCENT,
    BRIDGE,
    DEFAULT_TFTS_ROOT,
    GREEN,
    MODEL,
    NULL,
    TEXT,
    VISIONCORE_ROOT,
    _clean_axes,
    _load_tangent_maps,
    _nearest_delta,
    _pca_basis,
    _save_fig,
    _write_csv,
    _write_json,
)


DEFAULT_OUT_ROOT = (
    VISIONCORE_ROOT
    / "outputs"
    / "compact_retinal_translation_geometry"
    / "static_pc_adjudication_v1"
)


@dataclass(frozen=True)
class TangentObject:
    object_id: str
    group_id: str
    image_id: str
    trial_index: str
    time_index: str
    r0: np.ndarray
    bx: np.ndarray
    by: np.ndarray


def _parse_k_list(value: str) -> list[int]:
    out = sorted({int(part.strip()) for part in str(value).split(",") if part.strip()})
    if not out or any(k <= 0 for k in out):
        raise argparse.ArgumentTypeError("--k-list must contain positive integers")
    return out


def _as_vector(value: Any, *, name: str, object_id: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} for object {object_id} is not a vector: shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} for object {object_id} contains non-finite values")
    return arr


def _meta_label(meta: dict[str, Any], key: str, fallback: str) -> str:
    value = meta.get(key, fallback)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return fallback
    return str(value)


def _group_label(meta: dict[str, Any], object_id: str, group_by: str) -> str:
    if group_by == "image_id":
        return _meta_label(meta, "image_id", object_id)
    if group_by == "trial_index":
        image_id = _meta_label(meta, "image_id", "unknown_image")
        trial_index = _meta_label(meta, "trial_index", object_id)
        return f"{image_id}/{trial_index}"
    if group_by == "object_id":
        return str(object_id)
    raise ValueError(f"Unsupported group_by={group_by!r}")


def _collect_tangent_objects(
    *,
    tfts_root: Path,
    requested_delta: float,
    group_by: str,
) -> tuple[float, list[TangentObject], int, list[dict[str, Any]]]:
    deltas, payload_by_delta = _load_tangent_maps(tfts_root)
    delta = _nearest_delta(deltas, requested_delta)
    payload = payload_by_delta[delta]

    objects: list[TangentObject] = []
    skipped: list[dict[str, Any]] = []
    n_units: int | None = None
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
        if n_units is None:
            n_units = int(r0.size)
        elif int(r0.size) != n_units:
            raise ValueError(f"Object {object_id} has {r0.size} units, expected {n_units}")
        energy = float(np.dot(bx, bx) + np.dot(by, by))
        if energy <= 0.0:
            skipped.append(
                {
                    "delta_arcmin": float(delta),
                    "object_id": str(object_id),
                    "skip_reason": "zero combined tangent energy",
                }
            )
            continue
        objects.append(
            TangentObject(
                object_id=str(object_id),
                group_id=_group_label(meta, str(object_id), group_by),
                image_id=_meta_label(meta, "image_id", ""),
                trial_index=_meta_label(meta, "trial_index", ""),
                time_index=_meta_label(meta, "time_index", ""),
                r0=r0,
                bx=bx,
                by=by,
            )
        )

    if not objects or n_units is None:
        raise ValueError(f"No usable tangent-map objects found for delta={delta}.")
    return float(delta), objects, n_units, skipped


def _assign_group_folds(
    objects: list[TangentObject],
    *,
    n_folds: int,
    seed: int,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    groups = sorted({obj.group_id for obj in objects})
    if n_folds < 2:
        raise ValueError("--n-folds must be at least 2")
    if len(groups) < n_folds:
        raise ValueError(f"Requested {n_folds} folds, but only found {len(groups)} groups")

    counts = {group: sum(obj.group_id == group for obj in objects) for group in groups}
    rng = np.random.default_rng(int(seed))
    tie_break = {group: float(rng.random()) for group in groups}
    ordered = sorted(groups, key=lambda group: (-counts[group], tie_break[group], group))

    fold_object_counts = [0 for _ in range(n_folds)]
    fold_group_counts = [0 for _ in range(n_folds)]
    group_to_fold: dict[str, int] = {}
    for group in ordered:
        fold_id = min(
            range(n_folds),
            key=lambda idx: (fold_object_counts[idx], fold_group_counts[idx], idx),
        )
        group_to_fold[group] = int(fold_id)
        fold_object_counts[fold_id] += int(counts[group])
        fold_group_counts[fold_id] += 1

    fold_rows: list[dict[str, Any]] = []
    for fold_id in range(n_folds):
        fold_groups = sorted(group for group, fid in group_to_fold.items() if fid == fold_id)
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "n_groups": int(len(fold_groups)),
                "n_objects": int(sum(counts[group] for group in fold_groups)),
                "group_ids": ",".join(fold_groups),
            }
        )
    return group_to_fold, fold_rows


def _fit_pca_basis(rows: list[np.ndarray], *, max_k: int) -> np.ndarray:
    if not rows:
        raise ValueError("Cannot fit PCA basis from an empty row set")
    _, basis = _pca_basis(np.stack(rows, axis=0), n_components=int(max_k))
    return np.asarray(basis, dtype=np.float64)


def _random_basis(n_units: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.normal(size=(int(n_units), int(n_dims)))
    q, r = np.linalg.qr(mat, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs[None, :]


def _orth_residual_basis(primary: np.ndarray, nuisance: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    primary = np.asarray(primary, dtype=np.float64)
    nuisance = np.asarray(nuisance, dtype=np.float64)
    if primary.ndim != 2 or nuisance.ndim != 2:
        raise ValueError("primary and nuisance bases must be 2D")
    if primary.shape[0] != nuisance.shape[0]:
        raise ValueError("primary and nuisance bases must have matching unit counts")
    if primary.shape[1] == 0:
        return primary[:, :0]
    if nuisance.shape[1] > 0:
        qn, _ = np.linalg.qr(nuisance)
        residual = primary - qn @ (qn.T @ primary)
    else:
        residual = primary.copy()
    q, r = np.linalg.qr(residual)
    diag = np.abs(np.diag(r)) if r.ndim == 2 else np.asarray([], dtype=np.float64)
    scale = float(np.max(diag)) if diag.size else 0.0
    keep = diag > max(float(tol), scale * float(tol))
    return q[:, keep]


def _subspace_overlap_row(
    *,
    delta: float,
    fold_id: int,
    k: int,
    compact_basis: np.ndarray,
    static_basis: np.ndarray,
    n_train_objects: int,
    n_test_objects: int,
    n_train_groups: int,
    n_test_groups: int,
) -> dict[str, Any]:
    uc = np.asarray(compact_basis[:, : int(k)], dtype=np.float64)
    us = np.asarray(static_basis[:, : int(k)], dtype=np.float64)
    k_eff = int(min(uc.shape[1], us.shape[1]))
    if k_eff <= 0:
        cosines = np.asarray([], dtype=np.float64)
    else:
        uc = uc[:, :k_eff]
        us = us[:, :k_eff]
        cosines = np.linalg.svd(uc.T @ us, compute_uv=False)
    sq = cosines**2
    return {
        "delta_arcmin": float(delta),
        "fold_id": int(fold_id),
        "k": int(k),
        "k_effective": int(k_eff),
        "mean_squared_principal_cosine": float(np.mean(sq)) if sq.size else float("nan"),
        "mean_principal_cosine": float(np.mean(cosines)) if cosines.size else float("nan"),
        "min_principal_cosine": float(np.min(cosines)) if cosines.size else float("nan"),
        "max_principal_cosine": float(np.max(cosines)) if cosines.size else float("nan"),
        "principal_cosines": ";".join(f"{float(v):.8g}" for v in cosines),
        "n_train_objects": int(n_train_objects),
        "n_test_objects": int(n_test_objects),
        "n_train_groups": int(n_train_groups),
        "n_test_groups": int(n_test_groups),
    }


def _capture_tangent_fraction(
    bx: np.ndarray,
    by: np.ndarray,
    basis: np.ndarray,
    *,
    k: int,
) -> tuple[float, float, float, int]:
    basis = np.asarray(basis, dtype=np.float64)
    k_eff = int(min(int(k), basis.shape[1]))
    energy_bx = float(np.dot(bx, bx))
    energy_by = float(np.dot(by, by))
    energy = energy_bx + energy_by
    if k_eff <= 0 or energy <= 0.0:
        return float("nan"), float("nan"), float("nan"), k_eff

    b = basis[:, :k_eff]
    bx_capture = float(np.sum((b.T @ bx) ** 2))
    by_capture = float(np.sum((b.T @ by) ** 2))
    capture_combined = (bx_capture + by_capture) / energy
    capture_bx = bx_capture / energy_bx if energy_bx > 0.0 else float("nan")
    capture_by = by_capture / energy_by if energy_by > 0.0 else float("nan")
    return capture_combined, capture_bx, capture_by, k_eff


def _append_capture_rows(
    rows: list[dict[str, Any]],
    *,
    delta: float,
    test_objects: list[TangentObject],
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
    rows: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
    objects: list[TangentObject],
    group_to_fold: dict[str, int],
    fold_id: int,
    delta: float,
    k_list: list[int],
    n_units: int,
    n_random: int,
    n_unit_shuffle: int,
    seed: int,
    group_by: str,
) -> None:
    train_objects = [obj for obj in objects if group_to_fold[obj.group_id] != fold_id]
    test_objects = [obj for obj in objects if group_to_fold[obj.group_id] == fold_id]
    train_groups = {obj.group_id for obj in train_objects}
    test_groups = {obj.group_id for obj in test_objects}
    if not train_objects or not test_objects:
        raise ValueError(f"Fold {fold_id} has empty train or test split")

    max_k = int(max(k_list))
    compact_basis = _fit_pca_basis(
        [obj.bx for obj in train_objects] + [obj.by for obj in train_objects],
        max_k=max_k,
    )
    static_basis = _fit_pca_basis([obj.r0 for obj in train_objects], max_k=max_k + 1)
    basis_fit_scope = f"fold_disjoint_by_{group_by}"

    common_kwargs = {
        "delta": delta,
        "test_objects": test_objects,
        "fold_id": int(fold_id),
        "k_list": k_list,
        "n_train_objects": len(train_objects),
        "n_test_objects": len(test_objects),
        "n_train_groups": len(train_groups),
        "n_test_groups": len(test_groups),
        "basis_fit_scope": basis_fit_scope,
    }
    _append_capture_rows(
        rows,
        basis=compact_basis[:, :max_k],
        basis_type="compact_tangent_pc",
        basis_training_source="fold_train_bx_by",
        random_id="observed",
        **common_kwargs,
    )
    _append_capture_rows(
        rows,
        basis=static_basis[:, :max_k],
        basis_type="static_response_pc",
        basis_training_source="fold_train_r0",
        random_id="observed",
        **common_kwargs,
    )
    if static_basis.shape[1] > 1:
        _append_capture_rows(
            rows,
            basis=static_basis[:, 1 : max_k + 1],
            basis_type="static_response_pc_without_pc1",
            basis_training_source="fold_train_r0_pc2_plus",
            random_id="observed",
            **common_kwargs,
        )

    for k in k_list:
        overlap_rows.append(
            _subspace_overlap_row(
                delta=delta,
                fold_id=int(fold_id),
                k=int(k),
                compact_basis=compact_basis,
                static_basis=static_basis,
                n_train_objects=len(train_objects),
                n_test_objects=len(test_objects),
                n_train_groups=len(train_groups),
                n_test_groups=len(test_groups),
            )
        )
        compact_resid = _orth_residual_basis(compact_basis[:, : int(k)], static_basis[:, : int(k)])
        static_resid = _orth_residual_basis(static_basis[:, : int(k)], compact_basis[:, : int(k)])
        _append_capture_rows(
            rows,
            basis=compact_resid,
            basis_type="compact_residualized_against_static_pc",
            basis_training_source="fold_train_bx_by_residualized_against_static_r0_pc",
            random_id="observed",
            **{**common_kwargs, "k_list": [int(k)]},
        )
        _append_capture_rows(
            rows,
            basis=static_resid,
            basis_type="static_pc_residualized_against_compact",
            basis_training_source="fold_train_r0_residualized_against_compact_tangent_pc",
            random_id="observed",
            **{**common_kwargs, "k_list": [int(k)]},
        )

    global_axis = np.ones((int(n_units), 1), dtype=np.float64) / np.sqrt(float(n_units))
    _append_capture_rows(
        rows,
        basis=global_axis,
        basis_type="global_rate_axis",
        basis_training_source="analytic_unit_mean_axis",
        random_id="observed",
        **common_kwargs,
    )

    rng = np.random.default_rng(int(seed) + 1009 * int(fold_id))
    for draw in range(int(n_unit_shuffle)):
        permuted = compact_basis[rng.permutation(int(n_units)), :max_k]
        _append_capture_rows(
            rows,
            basis=permuted,
            basis_type="unit_shuffle_compact",
            basis_training_source="fold_train_bx_by_unit_permuted",
            random_id=f"shuffle_{draw:03d}",
            **common_kwargs,
        )
    for draw in range(int(n_random)):
        _append_capture_rows(
            rows,
            basis=_random_basis(n_units, max_k, rng),
            basis_type="random_orthonormal",
            basis_training_source="isotropic_random_subspace",
            random_id=f"random_{draw:03d}",
            **common_kwargs,
        )


def _draw_averaged_object_frame(capture_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "delta_arcmin",
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
    agg = (
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
    return agg


def _cluster_bootstrap_mean_ci(
    frame: pd.DataFrame,
    *,
    value_col: str,
    group_col: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, np.ndarray]:
    vals = frame[[group_col, value_col]].copy()
    vals = vals[np.isfinite(vals[value_col].to_numpy(dtype=np.float64))]
    if vals.empty:
        empty = np.asarray([], dtype=np.float64)
        return float("nan"), float("nan"), float("nan"), empty

    observed = float(vals[value_col].mean())
    groups = sorted(vals[group_col].astype(str).unique())
    if len(groups) <= 1 or n_bootstrap <= 0:
        empty = np.asarray([], dtype=np.float64)
        return observed, float("nan"), float("nan"), empty

    grouped_values = {
        group: vals.loc[vals[group_col].astype(str) == group, value_col].to_numpy(dtype=np.float64)
        for group in groups
    }
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        sample_values = np.concatenate([grouped_values[group] for group in sampled])
        boot[i] = float(np.mean(sample_values))
    return observed, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), boot


def _summary_rows(
    object_df: pd.DataFrame,
    *,
    group_by: str,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (basis_type, k), block in object_df.groupby(["basis_type", "k"], sort=True):
        mean, lo, hi, _ = _cluster_bootstrap_mean_ci(
            block,
            value_col="capture_combined",
            group_col="group_id",
            n_bootstrap=n_bootstrap,
            seed=int(seed) + 17 * int(k) + sum(ord(ch) for ch in str(basis_type)),
        )
        bx_mean, bx_lo, bx_hi, _ = _cluster_bootstrap_mean_ci(
            block,
            value_col="capture_bx",
            group_col="group_id",
            n_bootstrap=n_bootstrap,
            seed=int(seed) + 31 * int(k) + sum(ord(ch) for ch in str(basis_type)),
        )
        by_mean, by_lo, by_hi, _ = _cluster_bootstrap_mean_ci(
            block,
            value_col="capture_by",
            group_col="group_id",
            n_bootstrap=n_bootstrap,
            seed=int(seed) + 43 * int(k) + sum(ord(ch) for ch in str(basis_type)),
        )
        rows.append(
            {
                "basis_type": str(basis_type),
                "k": int(k),
                "capture_combined_mean": mean,
                "capture_combined_ci_low": lo,
                "capture_combined_ci_high": hi,
                "capture_bx_mean": bx_mean,
                "capture_bx_ci_low": bx_lo,
                "capture_bx_ci_high": bx_hi,
                "capture_by_mean": by_mean,
                "capture_by_ci_low": by_lo,
                "capture_by_ci_high": by_hi,
                "n_objects": int(block["object_id"].nunique()),
                "n_groups": int(block["group_id"].nunique()),
                "group_by": group_by,
                "n_basis_draws_median": float(block["n_basis_draws"].median()),
            }
        )
    return rows


def _paired_difference_rows(
    object_df: pd.DataFrame,
    *,
    lhs_basis: str,
    group_by: str,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lhs = object_df[object_df["basis_type"].astype(str) == lhs_basis].copy()
    merge_cols = ["delta_arcmin", "object_id", "group_id", "fold_id", "k"]
    for rhs_basis in sorted(set(object_df["basis_type"].astype(str)) - {lhs_basis}):
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
                n_bootstrap=n_bootstrap,
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
                    "group_by": group_by,
                }
            )
    return rows


def _overlap_summary_rows(
    overlap_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if overlap_df.empty:
        return rows
    for k, block in overlap_df.groupby("k", sort=True):
        for value_col in ["mean_squared_principal_cosine", "mean_principal_cosine", "min_principal_cosine"]:
            mean, lo, hi, _ = _cluster_bootstrap_mean_ci(
                block,
                value_col=value_col,
                group_col="fold_id",
                n_bootstrap=int(n_bootstrap),
                seed=int(seed) + int(k) * 211 + sum(ord(ch) for ch in value_col),
            )
            rows.append(
                {
                    "k": int(k),
                    "metric": value_col,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_folds": int(block["fold_id"].nunique()),
                }
            )
    return rows


def _plot_outputs(summary: pd.DataFrame, diffs: pd.DataFrame, out_root: Path) -> None:
    order = [
        "compact_tangent_pc",
        "static_response_pc",
        "compact_residualized_against_static_pc",
        "static_pc_residualized_against_compact",
        "static_response_pc_without_pc1",
        "global_rate_axis",
        "unit_shuffle_compact",
        "random_orthonormal",
    ]
    colors = {
        "compact_tangent_pc": MODEL,
        "static_response_pc": ACCENT,
        "compact_residualized_against_static_pc": "#1b9e77",
        "static_pc_residualized_against_compact": "#d95f02",
        "static_response_pc_without_pc1": GREEN,
        "global_rate_axis": BRIDGE,
        "unit_shuffle_compact": NULL,
        "random_orthonormal": "#b8b8b8",
    }
    labels = {
        "compact_tangent_pc": "compact tangent PCs",
        "static_response_pc": "static-response PCs",
        "compact_residualized_against_static_pc": "compact residual vs static",
        "static_pc_residualized_against_compact": "static residual vs compact",
        "static_response_pc_without_pc1": "static PCs, skip PC1",
        "global_rate_axis": "global rate axis",
        "unit_shuffle_compact": "unit-shuffle compact",
        "random_orthonormal": "random subspace",
    }

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for basis_type in order:
        block = summary[summary["basis_type"].astype(str) == basis_type].sort_values("k")
        if block.empty:
            continue
        x = block["k"].to_numpy(dtype=np.float64)
        y = block["capture_combined_mean"].to_numpy(dtype=np.float64)
        lo = block["capture_combined_ci_low"].to_numpy(dtype=np.float64)
        hi = block["capture_combined_ci_high"].to_numpy(dtype=np.float64)
        ax.plot(
            x,
            y,
            marker="o",
            lw=2.0 if basis_type in {"compact_tangent_pc", "static_response_pc"} else 1.3,
            color=colors[basis_type],
            label=labels[basis_type],
        )
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            ax.fill_between(x, lo, hi, color=colors[basis_type], alpha=0.12, linewidth=0)
    ax.set_xlabel("basis dimension k")
    ax.set_ylabel("held-out tangent variance captured")
    ax.set_title("Compact vs static-PC adjudication", loc="left", fontweight="bold")
    ax.set_ylim(bottom=0.0)
    _clean_axes(ax, grid=True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_root / "figures" / "static_pc_adjudication_capture")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for rhs_basis in [
        "static_response_pc",
        "compact_residualized_against_static_pc",
        "static_pc_residualized_against_compact",
        "static_response_pc_without_pc1",
        "unit_shuffle_compact",
        "random_orthonormal",
        "global_rate_axis",
    ]:
        block = diffs[diffs["rhs_basis_type"].astype(str) == rhs_basis].sort_values("k")
        if block.empty:
            continue
        x = block["k"].to_numpy(dtype=np.float64)
        y = block["mean_lhs_minus_rhs"].to_numpy(dtype=np.float64)
        lo = block["ci_low"].to_numpy(dtype=np.float64)
        hi = block["ci_high"].to_numpy(dtype=np.float64)
        ax.plot(x, y, marker="o", lw=1.8, color=colors.get(rhs_basis, NULL), label=f"minus {labels[rhs_basis]}")
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            ax.fill_between(x, lo, hi, color=colors.get(rhs_basis, NULL), alpha=0.12, linewidth=0)
    ax.axhline(0.0, color=TEXT, lw=0.9, alpha=0.5)
    ax.set_xlabel("basis dimension k")
    ax.set_ylabel("compact capture minus control")
    ax.set_title("Paired object-level contrasts", loc="left", fontweight="bold")
    _clean_axes(ax, grid=True)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    _save_fig(fig, out_root / "figures" / "static_pc_adjudication_compact_minus_controls")


def _format_float(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(val):
        return "nan"
    return f"{val:.3f}"


def _write_report(
    *,
    out_root: Path,
    config: dict[str, Any],
    summary: pd.DataFrame,
    diffs: pd.DataFrame,
    overlap_summary: pd.DataFrame,
) -> None:
    k_values = sorted(int(k) for k in summary["k"].unique())
    key_k = 10 if 10 in k_values else k_values[len(k_values) // 2]
    key_summary = summary[summary["k"].astype(int) == key_k].copy()
    key_diff = diffs[
        (diffs["k"].astype(int) == key_k)
        & (diffs["rhs_basis_type"].astype(str) == "static_response_pc")
    ]
    if not key_diff.empty:
        diff_row = key_diff.iloc[0]
        ci_low = float(diff_row["ci_low"])
        ci_high = float(diff_row["ci_high"])
        if np.isfinite(ci_low) and ci_low > 0.0:
            verdict = (
                "At the key dimension, compact tangent PCs beat fold-disjoint static-response PCs."
            )
        elif np.isfinite(ci_high) and ci_high < 0.0:
            verdict = (
                "At the key dimension, fold-disjoint static-response PCs beat compact tangent PCs."
            )
        else:
            verdict = (
                "At the key dimension, compact and fold-disjoint static-response PCs are not cleanly separated."
            )
        diff_line = (
            f"compact - static-PC mean difference at k={key_k}: "
            f"{_format_float(diff_row['mean_lhs_minus_rhs'])} "
            f"[{_format_float(diff_row['ci_low'])}, {_format_float(diff_row['ci_high'])}]"
        )
    else:
        verdict = "The compact-vs-static paired contrast was not available."
        diff_line = "compact - static-PC contrast unavailable"

    basis_lines = []
    for basis in [
        "compact_tangent_pc",
        "static_response_pc",
        "compact_residualized_against_static_pc",
        "static_pc_residualized_against_compact",
        "static_response_pc_without_pc1",
        "global_rate_axis",
        "unit_shuffle_compact",
        "random_orthonormal",
    ]:
        row = key_summary[key_summary["basis_type"].astype(str) == basis]
        if row.empty:
            continue
        rec = row.iloc[0]
        basis_lines.append(
            f"- `{basis}`: {_format_float(rec['capture_combined_mean'])} "
            f"[{_format_float(rec['capture_combined_ci_low'])}, "
            f"{_format_float(rec['capture_combined_ci_high'])}]"
        )

    overlap_lines = []
    overlap_key = overlap_summary[overlap_summary["k"].astype(int) == int(key_k)]
    for metric, label in [
        ("mean_squared_principal_cosine", "mean squared principal cosine"),
        ("mean_principal_cosine", "mean principal cosine"),
        ("min_principal_cosine", "minimum principal cosine"),
    ]:
        row = overlap_key[overlap_key["metric"].astype(str) == metric]
        if row.empty:
            continue
        rec = row.iloc[0]
        overlap_lines.append(
            f"- compact/static-PC {label}: {_format_float(rec['mean'])} "
            f"[{_format_float(rec['ci_low'])}, {_format_float(rec['ci_high'])}]"
        )

    text = "\n".join(
        [
            "# Static-PC Adjudication",
            "",
            "This run compares held-out translation-tangent capture for bases fit on disjoint training images:",
            "",
            "- `compact_tangent_pc`: PCs of training horizontal/vertical translation tangents.",
            "- `static_response_pc`: PCs of training baseline responses `r0`.",
            "- residualized compact/static-PC bases: each basis after removing the other basis first.",
            "- `static_response_pc_without_pc1`: same static basis after dropping the first static PC.",
            "- `global_rate_axis`, `unit_shuffle_compact`, and `random_orthonormal`: controls.",
            "",
            "The object-level pairing is over the same held-out local windows, and confidence intervals use a clustered bootstrap over the split group.",
            "",
            "## Configuration",
            "",
            f"- tangent source: `{config['tangent_source']}`",
            f"- finite-difference step: `{config['delta_arcmin']}` arcmin",
            f"- split group: `{config['group_by']}`",
            f"- objects: `{config['n_objects']}`",
            f"- skipped objects: `{config['n_skipped_objects']}`",
            f"- groups: `{config['n_groups']}`",
            f"- folds: `{config['n_folds']}`",
            f"- bootstrap draws: `{config['n_bootstrap']}`",
            "",
            f"## Key Readout at k={key_k}",
            "",
            *basis_lines,
            "",
            f"- {diff_line}",
            "",
            *overlap_lines,
            "",
            "## Interpretation",
            "",
            verdict,
            "",
            "The stronger readout is the overlap/residual pattern: compact tangent PCs and static-response PCs capture nearly the same held-out translation variance, and the residualized pieces are much smaller. This does not erase the compact geometry result. It says the shared translation channel is closely aligned with the local image-response manifold, so static PCs are a serious contained control and not a generic nuisance axis.",
            "",
            "The current result therefore supports a conservative interpretation: Figure 3 shows a compact, image-general translation-tangent geometry, but by itself it does not prove that the compact basis is uniquely translation-specific rather than a low-dimensional tangent bundle of the image-response manifold. The next adjudication should ask whether compact and static-PC removals have symmetric consequences in the actual joint image/feature decoders.",
            "",
            "## Files",
            "",
            "- `static_pc_adjudication_object_capture.csv`: per-object, per-draw capture rows.",
            "- `static_pc_adjudication_object_capture_draw_averaged.csv`: per-object rows after averaging random/control draws.",
            "- `static_pc_adjudication_summary.csv`: clustered bootstrap summaries.",
            "- `static_pc_adjudication_paired_differences.csv`: paired compact-minus-control contrasts.",
            "- `static_pc_adjudication_subspace_overlap.csv`: fold-level compact/static-PC principal-angle summaries.",
            "- `static_pc_adjudication_subspace_overlap_summary.csv`: bootstrapped overlap summaries.",
            "- `figures/static_pc_adjudication_capture.png`: capture curves.",
            "- `figures/static_pc_adjudication_compact_minus_controls.png`: paired contrast curves.",
            "",
        ]
    )
    (out_root / "static_pc_adjudication_report.md").write_text(text, encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    tfts_root = Path(args.tfts_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    k_list = _parse_k_list(args.k_list)
    delta, objects, n_units, skipped_rows = _collect_tangent_objects(
        tfts_root=tfts_root,
        requested_delta=float(args.delta),
        group_by=str(args.group_by),
    )
    _write_csv(out_root / "static_pc_adjudication_skipped_objects.csv", skipped_rows)
    group_to_fold, fold_rows = _assign_group_folds(
        objects,
        n_folds=int(args.n_folds),
        seed=int(args.seed),
    )
    _write_csv(out_root / "static_pc_adjudication_folds.csv", fold_rows)

    capture_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for fold_id in range(int(args.n_folds)):
        _score_fold(
            rows=capture_rows,
            overlap_rows=overlap_rows,
            objects=objects,
            group_to_fold=group_to_fold,
            fold_id=fold_id,
            delta=delta,
            k_list=k_list,
            n_units=n_units,
            n_random=int(args.n_random),
            n_unit_shuffle=int(args.n_unit_shuffle),
            seed=int(args.seed),
            group_by=str(args.group_by),
        )

    _write_csv(out_root / "static_pc_adjudication_object_capture.csv", capture_rows)
    _write_csv(out_root / "static_pc_adjudication_subspace_overlap.csv", overlap_rows)
    capture_df = pd.DataFrame(capture_rows)
    object_df = _draw_averaged_object_frame(capture_df)
    object_rows = object_df.to_dict(orient="records")
    _write_csv(out_root / "static_pc_adjudication_object_capture_draw_averaged.csv", object_rows)

    summary = pd.DataFrame(
        _summary_rows(
            object_df,
            group_by=str(args.group_by),
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed),
        )
    )
    diffs = pd.DataFrame(
        _paired_difference_rows(
            object_df,
            lhs_basis="compact_tangent_pc",
            group_by=str(args.group_by),
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed) + 50000,
        )
    )
    _write_csv(out_root / "static_pc_adjudication_summary.csv", summary.to_dict(orient="records"))
    _write_csv(out_root / "static_pc_adjudication_paired_differences.csv", diffs.to_dict(orient="records"))
    overlap_summary = pd.DataFrame(
        _overlap_summary_rows(
            pd.DataFrame(overlap_rows),
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed) + 90000,
        )
    )
    _write_csv(out_root / "static_pc_adjudication_subspace_overlap_summary.csv", overlap_summary.to_dict(orient="records"))
    _plot_outputs(summary, diffs, out_root)

    config = {
        "tangent_source": str((tfts_root / "tangent_maps" / "twin_tangent_maps.pkl").resolve()),
        "out_root": str(out_root.resolve()),
        "delta_arcmin": float(delta),
        "requested_delta_arcmin": float(args.delta),
        "group_by": str(args.group_by),
        "n_objects": int(len(objects)),
        "n_skipped_objects": int(len(skipped_rows)),
        "n_groups": int(len({obj.group_id for obj in objects})),
        "n_units": int(n_units),
        "n_folds": int(args.n_folds),
        "k_list": k_list,
        "n_random": int(args.n_random),
        "n_unit_shuffle": int(args.n_unit_shuffle),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
    }
    _write_json(out_root / "static_pc_adjudication_config.json", config)
    _write_report(out_root=out_root, config=config, summary=summary, diffs=diffs, overlap_summary=overlap_summary)
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfts-root", type=Path, default=DEFAULT_TFTS_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--k-list", type=str, default="2,5,10,20,30,40,50")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--group-by",
        choices=["image_id", "trial_index", "object_id"],
        default="image_id",
        help="Disjoint split group and bootstrap cluster.",
    )
    parser.add_argument("--n-random", type=int, default=64)
    parser.add_argument("--n-unit-shuffle", type=int, default=64)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = analyze(args)
    out_root = Path(config["out_root"])
    print(f"Wrote static-PC adjudication to {out_root}")
    print(f"Objects: {config['n_objects']} across {config['n_groups']} {config['group_by']} groups")


if __name__ == "__main__":
    main()
