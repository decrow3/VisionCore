from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import dill
import numpy as np

from declan.matched_twin_covariance_closure.run_cache_closure import (
    DEFAULT_FIG2_CACHE,
    DEFAULT_FIG3_CACHE,
    _fig2_by_session,
    _load_pickle,
    _projection_complement,
    _projection_modes,
    _psd_clip,
    _sym,
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


DEFAULT_OUTPUT_ROOT = Path("outputs") / "direct_recorded_derivative_twin_alignment"


@dataclass
class AnalysisConfig:
    output_root: str
    sessions: list[str]
    projection_controls: list[str]
    target_variants: list[str]
    k_list: list[int]
    ridge_multipliers: list[float]
    min_samples_per_context: int
    min_units: int
    max_eye_condition: float
    reliability_excess_threshold: float
    n_reliability_shuffles: int
    n_nulls: int
    n_bootstrap: int
    seed: int
    context_mode: str
    context_bin_size: int
    compact_crossfit_group_mode: str


@dataclass
class ContextDerivative:
    context_id: int
    context_label: str
    sample_mask: np.ndarray
    sample_indices: np.ndarray
    trial_ids: np.ndarray
    time_indices: np.ndarray
    n_samples: int
    eye_rank: int
    eye_condition: float
    eye_eig_small: float
    eye_eig_large: float
    ridge_lambda: float
    ridge_multiplier: float
    split_half_overlap: float
    split_half_shuffle_median: float
    split_half_excess: float
    reliability_qualified: bool
    b_rec: np.ndarray
    j_mean: np.ndarray
    status: str


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def parse_int_list(raw: str) -> list[int]:
    out = [int(piece.strip()) for piece in str(raw).split(",") if piece.strip()]
    return out or [10]


def parse_float_list(raw: str) -> list[float]:
    out = [float(piece.strip()) for piece in str(raw).split(",") if piece.strip()]
    return out or [0.0]


def parse_str_list(raw: str) -> list[str]:
    return [piece.strip() for piece in str(raw).split(",") if piece.strip()]


def center_design_response(eye_xy: np.ndarray, responses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(eye_xy, dtype=np.float64)
    r = np.asarray(responses, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"eye_xy must have shape [samples, 2], got {x.shape}")
    if r.ndim != 2 or r.shape[0] != x.shape[0]:
        raise ValueError(f"responses must have shape [samples, units], got {r.shape}")
    keep = np.isfinite(x).all(axis=1) & np.isfinite(r).all(axis=1)
    x = x[keep]
    r = r[keep]
    return x - np.mean(x, axis=0, keepdims=True), r - np.mean(r, axis=0, keepdims=True), keep


def eye_design_diagnostics(eye_xy_centered: np.ndarray) -> dict[str, float]:
    x = np.asarray(eye_xy_centered, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 2:
        return {
            "eye_cov_eig_small": float("nan"),
            "eye_cov_eig_large": float("nan"),
            "eye_condition": float("inf"),
            "eye_rank": 0,
        }
    vals = np.sort(np.maximum(np.linalg.eigvalsh(x.T @ x), 0.0))
    small = float(vals[0])
    large = float(vals[-1])
    cond = float(large / small) if small > 0.0 else float("inf")
    return {
        "eye_cov_eig_small": small,
        "eye_cov_eig_large": large,
        "eye_condition": cond,
        "eye_rank": int(np.sum(vals > max(large, 1.0) * 1e-10)),
    }


def ridge_grid_from_eye(eye_xy_centered: np.ndarray, multipliers: list[float]) -> np.ndarray:
    scale = float(np.trace(np.asarray(eye_xy_centered, dtype=np.float64).T @ np.asarray(eye_xy_centered, dtype=np.float64)) / 2.0)
    return np.asarray([float(m) * scale for m in multipliers], dtype=np.float64)


def fit_recorded_derivative(
    eye_xy_centered: np.ndarray,
    responses_centered: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    x = np.asarray(eye_xy_centered, dtype=np.float64)
    r = np.asarray(responses_centered, dtype=np.float64)
    inv = np.linalg.pinv(x.T @ x + float(ridge_lambda) * np.eye(2, dtype=np.float64))
    return r.T @ x @ inv


def orth(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim == 1:
        m = m[:, None]
    if m.ndim != 2 or m.size == 0:
        return np.zeros((m.shape[0] if m.ndim else 0, 0), dtype=np.float64)
    u, s, _vh = np.linalg.svd(m, full_matrices=False)
    if s.size == 0:
        return u[:, :0]
    keep = s > max(float(s[0]), 1.0) * float(eps)
    return u[:, keep]


def basis_from_tangent_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] == 0 or not np.isfinite(m).all():
        return np.array([], dtype=np.float64), np.zeros((m.shape[0] if m.ndim == 2 else 0, 0), dtype=np.float64)
    vals, vecs = np.linalg.eigh(_sym(m @ m.T))
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def numerical_rank(vals: np.ndarray, eps: float = 1e-10) -> int:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0
    vmax = float(np.max(np.abs(vals)))
    if vmax <= 0.0:
        return 0
    return int(np.sum(vals > vmax * eps))


def frobenius_capture(target_matrix: np.ndarray, basis: np.ndarray) -> float:
    b = np.asarray(target_matrix, dtype=np.float64)
    u = orth(np.asarray(basis, dtype=np.float64))
    denom = float(np.sum(b * b))
    if denom <= 0.0 or u.shape[1] == 0:
        return float("nan")
    proj = u.T @ b
    return float(np.sum(proj * proj) / denom)


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    qa = orth(a)
    qb = orth(b)
    rank = min(qa.shape[1], qb.shape[1])
    if rank == 0:
        return float("nan")
    return float(np.sum((qa.T @ qb) ** 2) / rank)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    keep = np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(keep)) < 3:
        return float("nan")
    aa = aa[keep]
    bb = bb[keep]
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 0.0:
        return float("nan")
    return float(np.dot(aa, bb) / den)


def fixed_within_bin_permutation(bins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(bins, dtype=np.int64).ravel()
    perm = np.arange(labels.size, dtype=np.int64)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == int(label))
        if idx.size > 1:
            shuffled = idx.copy()
            rng.shuffle(shuffled)
            perm[idx] = shuffled
    return perm


def bootstrap_mean_ci(values: np.ndarray, *, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        boots[i] = float(np.mean(rng.choice(vals, size=vals.size, replace=True)))
    return float(np.mean(vals)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def sign_test_p_two_sided(n_pos: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    from math import comb

    lo = min(int(n_pos), int(n) - int(n_pos))
    p = sum(comb(int(n), k) for k in range(lo + 1)) / (2 ** int(n))
    return float(min(1.0, 2.0 * p))


def context_labels(samples: Any, mode: str, bin_size: int) -> np.ndarray:
    if str(mode) == "time_bin":
        return np.asarray(samples.time_indices, dtype=np.int64)
    if str(mode) == "time_window":
        return np.asarray(samples.time_indices, dtype=np.int64) // max(int(bin_size), 1)
    if str(mode) == "trial":
        return np.asarray(samples.trial_ids, dtype=np.int64)
    raise ValueError(f"Unsupported context mode: {mode}")


def tangent_matrix_from_samples(j: np.ndarray, mask: np.ndarray, projection: np.ndarray | None = None) -> np.ndarray:
    jj = np.asarray(j, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if jj.shape[0] == 0:
        return np.zeros((j.shape[1], 0), dtype=np.float64)
    mat = np.concatenate([jj[:, :, 0].T, jj[:, :, 1].T], axis=1)
    if projection is not None:
        mat = np.asarray(projection, dtype=np.float64) @ mat
    return mat


def fit_context_derivatives(
    *,
    samples: Any,
    j: np.ndarray,
    eye_px: np.ndarray,
    ridge_multipliers: list[float],
    min_samples: int,
    max_eye_condition: float,
    reliability_excess_threshold: float,
    n_reliability_shuffles: int,
    seed: int,
    context_mode: str,
    context_bin_size: int,
) -> tuple[list[ContextDerivative], list[dict[str, Any]], list[dict[str, Any]]]:
    labels = context_labels(samples, context_mode, int(context_bin_size))
    rows_inventory: list[dict[str, Any]] = []
    rows_reliability: list[dict[str, Any]] = []
    contexts: list[ContextDerivative] = []
    rng = np.random.default_rng(int(seed))
    for context_id in np.unique(labels):
        mask = labels == context_id
        idx = np.flatnonzero(mask)
        label = f"{context_mode}_{int(context_id)}"
        if str(context_mode) == "time_window":
            lo = int(context_id) * int(context_bin_size)
            hi = lo + int(context_bin_size) - 1
            label = f"time_window_{lo}_{hi}"
        if idx.size < int(min_samples):
            rows_inventory.append(
                {
                    "context_id": int(context_id),
                    "context_label": label,
                    "n_samples": int(idx.size),
                    "status": "too_few_samples",
                }
            )
            continue
        x, r, keep_local = center_design_response(eye_px[idx], samples.robs[idx])
        kept_idx = idx[keep_local]
        diag = eye_design_diagnostics(x)
        status = "ok"
        if x.shape[0] < int(min_samples):
            status = "too_few_finite_samples"
        elif int(diag["eye_rank"]) < 2:
            status = "eye_rank_lt_2"
        elif float(diag["eye_condition"]) > float(max_eye_condition):
            status = "eye_condition_too_high"
        rows_inventory.append(
            {
                "context_id": int(context_id),
                "context_label": label,
                "n_samples": int(x.shape[0]),
                "n_trials": int(np.unique(samples.trial_ids[kept_idx]).size) if kept_idx.size else 0,
                "time_min": int(np.min(samples.time_indices[kept_idx])) if kept_idx.size else -1,
                "time_max": int(np.max(samples.time_indices[kept_idx])) if kept_idx.size else -1,
                "eye_rank": int(diag["eye_rank"]),
                "eye_cov_eig_small": float(diag["eye_cov_eig_small"]),
                "eye_cov_eig_large": float(diag["eye_cov_eig_large"]),
                "eye_condition": float(diag["eye_condition"]),
                "status": status,
            }
        )
        if status != "ok":
            continue

        ridge_grid = ridge_grid_from_eye(x, ridge_multipliers)
        order = np.arange(x.shape[0])
        rng.shuffle(order)
        half = max(1, order.size // 2)
        a = order[:half]
        b = order[half:]
        if a.size < 4 or b.size < 4:
            continue

        best: dict[str, Any] | None = None
        for mult, lam in zip(ridge_multipliers, ridge_grid, strict=True):
            ba = fit_recorded_derivative(x[a], r[a], float(lam))
            bb = fit_recorded_derivative(x[b], r[b], float(lam))
            split = subspace_overlap(ba, bb)
            shuf_vals: list[float] = []
            for _ in range(int(n_reliability_shuffles)):
                xa = x[a].copy()
                xb = x[b].copy()
                rng.shuffle(xa, axis=0)
                rng.shuffle(xb, axis=0)
                shuf_vals.append(subspace_overlap(fit_recorded_derivative(xa, r[a], float(lam)), fit_recorded_derivative(xb, r[b], float(lam))))
            shuf = np.asarray(shuf_vals, dtype=np.float64)
            shuf_med = float(np.nanmedian(shuf))
            excess = float(split - shuf_med) if np.isfinite(split) and np.isfinite(shuf_med) else float("nan")
            cand = {
                "ridge_multiplier": float(mult),
                "ridge_lambda": float(lam),
                "split_half_overlap": float(split),
                "split_half_shuffle_median": shuf_med,
                "split_half_excess": excess,
            }
            if best is None or (np.nan_to_num(excess, nan=-np.inf) > np.nan_to_num(best["split_half_excess"], nan=-np.inf)):
                best = cand
        if best is None:
            continue

        b_rec = fit_recorded_derivative(x, r, float(best["ridge_lambda"]))
        j_mean = np.nanmean(np.asarray(j, dtype=np.float64)[kept_idx], axis=0)
        qualified = bool(np.isfinite(best["split_half_excess"]) and float(best["split_half_excess"]) > float(reliability_excess_threshold))
        rows_reliability.append(
            {
                "context_id": int(context_id),
                "context_label": label,
                "n_samples": int(x.shape[0]),
                "ridge_multiplier": best["ridge_multiplier"],
                "ridge_lambda": best["ridge_lambda"],
                "split_half_overlap": best["split_half_overlap"],
                "split_half_shuffle_median": best["split_half_shuffle_median"],
                "split_half_excess": best["split_half_excess"],
                "reliability_qualified": qualified,
            }
        )
        contexts.append(
            ContextDerivative(
                context_id=int(context_id),
                context_label=label,
                sample_mask=mask,
                sample_indices=kept_idx,
                trial_ids=np.asarray(samples.trial_ids[kept_idx], dtype=np.int64),
                time_indices=np.asarray(samples.time_indices[kept_idx], dtype=np.int64),
                n_samples=int(x.shape[0]),
                eye_rank=int(diag["eye_rank"]),
                eye_condition=float(diag["eye_condition"]),
                eye_eig_small=float(diag["eye_cov_eig_small"]),
                eye_eig_large=float(diag["eye_cov_eig_large"]),
                ridge_lambda=float(best["ridge_lambda"]),
                ridge_multiplier=float(best["ridge_multiplier"]),
                split_half_overlap=float(best["split_half_overlap"]),
                split_half_shuffle_median=float(best["split_half_shuffle_median"]),
                split_half_excess=float(best["split_half_excess"]),
                reliability_qualified=qualified,
                b_rec=b_rec,
                j_mean=j_mean,
                status="ok",
            )
        )
    return contexts, rows_inventory, rows_reliability


def null_capture_pack(
    *,
    b_rec: np.ndarray,
    basis: np.ndarray,
    k: int,
    bins: np.ndarray | None,
    rng: np.random.Generator,
    n_nulls: int,
) -> dict[str, float]:
    n = b_rec.shape[0]
    u = orth(basis[:, : int(k)])
    random_vals: list[float] = []
    unit_vals: list[float] = []
    rf_vals: list[float] = []
    for _ in range(int(n_nulls)):
        q, _ = np.linalg.qr(rng.standard_normal((n, max(int(k), 1))))
        random_vals.append(frobenius_capture(b_rec, q[:, : int(k)]))
        perm = rng.permutation(n)
        unit_vals.append(frobenius_capture(b_rec, u[perm, :]))
        if bins is not None:
            rf_perm = fixed_within_bin_permutation(np.asarray(bins, dtype=np.int64), rng)
            rf_vals.append(frobenius_capture(b_rec, u[rf_perm, :]))

    def median(vals: list[float]) -> float:
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.median(arr)) if arr.size else float("nan")

    return {
        "random_subspace_null_median": median(random_vals),
        "unit_shuffle_null_median": median(unit_vals),
        "rf_readout_null_median": median(rf_vals),
    }


def compute_tier_rows(
    *,
    session: str,
    subject: str,
    contexts: list[ContextDerivative],
    j: np.ndarray,
    target_raw: np.ndarray,
    target_psd: np.ndarray,
    projection_controls: list[str],
    target_variants: list[str],
    k_list: list[int],
    rf_bins: np.ndarray | None,
    n_nulls: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tier1_rows: list[dict[str, Any]] = []
    tier2_rows: list[dict[str, Any]] = []
    tier3_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    labels = np.full(j.shape[0], -1, dtype=np.int64)
    for i, ctx in enumerate(contexts):
        labels[ctx.sample_mask] = i
    targets = {"raw": target_raw, "psd": target_psd}

    for target_variant in target_variants:
        target_base = targets[str(target_variant)]
        for projection_control in projection_controls:
            modes = _projection_modes(projection_control, target_base)
            p = _projection_complement(j.shape[1], modes)
            for ctx_i, ctx in enumerate(contexts):
                train_mask = labels != ctx_i
                train_mask &= labels >= 0
                vals, vecs = basis_from_tangent_matrix(tangent_matrix_from_samples(j, train_mask, p))
                rank = numerical_rank(np.maximum(vals, 0.0))
                b_proj = p @ ctx.b_rec
                j_proj = p @ ctx.j_mean
                max_k = min(max(k_list), rank, vecs.shape[1])
                for k in k_list:
                    row_base = {
                        "session": session,
                        "subject": subject,
                        "target_variant": target_variant,
                        "projection_control": projection_control,
                        "context_id": int(ctx.context_id),
                        "context_label": ctx.context_label,
                        "n_samples": int(ctx.n_samples),
                        "basis_rank": int(rank),
                        "k": int(k),
                        "ridge_multiplier": float(ctx.ridge_multiplier),
                        "ridge_lambda": float(ctx.ridge_lambda),
                        "split_half_overlap": float(ctx.split_half_overlap),
                        "split_half_shuffle_median": float(ctx.split_half_shuffle_median),
                        "split_half_excess": float(ctx.split_half_excess),
                        "reliability_qualified": bool(ctx.reliability_qualified),
                    }
                    if int(k) > int(max_k):
                        tier1_rows.append({**row_base, "row_status": "not_evaluable_rank"})
                        continue
                    rng = np.random.default_rng(int(seed) + int(ctx.context_id) * 1009 + int(k) * 9176 + len(tier1_rows))
                    basis = vecs[:, : int(k)]
                    cap = frobenius_capture(b_proj, basis)
                    nulls = null_capture_pack(
                        b_rec=b_proj,
                        basis=basis,
                        k=int(k),
                        bins=rf_bins,
                        rng=rng,
                        n_nulls=int(n_nulls),
                    )
                    row = {
                        **row_base,
                        "capture": cap,
                        **nulls,
                        "effect_minus_random_subspace_median": cap - nulls["random_subspace_null_median"],
                        "effect_minus_unit_shuffle_median": cap - nulls["unit_shuffle_null_median"],
                        "effect_minus_rf_readout_median": cap - nulls["rf_readout_null_median"],
                        "row_status": "ok",
                    }
                    tier1_rows.append(row)
                    for null_name in ("random_subspace", "unit_shuffle", "rf_readout"):
                        null_rows.append(
                            {
                                "session": session,
                                "target_variant": target_variant,
                                "projection_control": projection_control,
                                "context_id": int(ctx.context_id),
                                "k": int(k),
                                "null_type": null_name,
                                "null_median": nulls[f"{null_name}_null_median"],
                                "effect": row[f"effect_minus_{null_name}_median"],
                            }
                        )

                ov = subspace_overlap(b_proj, j_proj)
                rng = np.random.default_rng(int(seed) + int(ctx.context_id) * 1777 + len(tier2_rows))
                unit_vals: list[float] = []
                rf_vals: list[float] = []
                context_vals: list[float] = []
                for _ in range(int(n_nulls)):
                    perm = rng.permutation(j.shape[1])
                    unit_vals.append(subspace_overlap(b_proj, j_proj[perm, :]))
                    if rf_bins is not None:
                        rf_perm = fixed_within_bin_permutation(rf_bins, rng)
                        rf_vals.append(subspace_overlap(b_proj, j_proj[rf_perm, :]))
                    other = rng.choice([i for i in range(len(contexts)) if i != ctx_i]) if len(contexts) > 1 else ctx_i
                    context_vals.append(subspace_overlap(b_proj, p @ contexts[int(other)].j_mean))
                unit_med = float(np.nanmedian(unit_vals)) if unit_vals else float("nan")
                rf_med = float(np.nanmedian(rf_vals)) if rf_vals else float("nan")
                context_med = float(np.nanmedian(context_vals)) if context_vals else float("nan")
                tier2_rows.append(
                    {
                        "session": session,
                        "subject": subject,
                        "target_variant": target_variant,
                        "projection_control": projection_control,
                        "context_id": int(ctx.context_id),
                        "context_label": ctx.context_label,
                        "n_samples": int(ctx.n_samples),
                        "split_half_excess": float(ctx.split_half_excess),
                        "reliability_qualified": bool(ctx.reliability_qualified),
                        "subspace_overlap": ov,
                        "unit_shuffle_null_median": unit_med,
                        "rf_readout_null_median": rf_med,
                        "context_shuffle_null_median": context_med,
                        "effect_minus_unit_shuffle_median": ov - unit_med,
                        "effect_minus_rf_readout_median": ov - rf_med,
                        "effect_minus_context_shuffle_median": ov - context_med,
                    }
                )
                tier3_rows.append(
                    {
                        "session": session,
                        "subject": subject,
                        "target_variant": target_variant,
                        "projection_control": projection_control,
                        "context_id": int(ctx.context_id),
                        "context_label": ctx.context_label,
                        "n_samples": int(ctx.n_samples),
                        "reliability_qualified": bool(ctx.reliability_qualified),
                        "cos_xx": cosine(b_proj[:, 0], j_proj[:, 0]),
                        "cos_xy": cosine(b_proj[:, 0], j_proj[:, 1]),
                        "cos_yx": cosine(b_proj[:, 1], j_proj[:, 0]),
                        "cos_yy": cosine(b_proj[:, 1], j_proj[:, 1]),
                    }
                )
    for row in tier3_rows:
        row["axis_selectivity_x"] = abs(float(row["cos_xx"])) - abs(float(row["cos_xy"]))
        row["axis_selectivity_y"] = abs(float(row["cos_yy"])) - abs(float(row["cos_yx"]))
    return tier1_rows, tier2_rows, tier3_rows, null_rows


def summarize_tier1(tier1_rows: list[dict[str, Any]], *, seed: int, n_bootstrap: int) -> list[dict[str, Any]]:
    rows = [r for r in tier1_rows if r.get("row_status") == "ok"]
    if not rows:
        return []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        for subset in ("all_contexts", "reliability_qualified"):
            if subset == "reliability_qualified" and not bool(r.get("reliability_qualified", False)):
                continue
            key = (r["target_variant"], r["projection_control"], int(r["k"]), subset)
            groups.setdefault(key, []).append(r)

    out: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    for (target_variant, projection_control, k, subset), group_rows in sorted(groups.items()):
        session_effects: dict[str, list[float]] = {}
        session_caps: dict[str, list[float]] = {}
        session_effects_unit: dict[str, list[float]] = {}
        session_effects_random: dict[str, list[float]] = {}
        for r in group_rows:
            session_effects.setdefault(str(r["session"]), []).append(float(r["effect_minus_rf_readout_median"]))
            session_effects_unit.setdefault(str(r["session"]), []).append(float(r["effect_minus_unit_shuffle_median"]))
            session_effects_random.setdefault(str(r["session"]), []).append(float(r["effect_minus_random_subspace_median"]))
            session_caps.setdefault(str(r["session"]), []).append(float(r["capture"]))
        sessions = sorted(session_effects)
        eff = np.asarray([np.nanmean(session_effects[s]) for s in sessions], dtype=np.float64)
        eff_unit = np.asarray([np.nanmean(session_effects_unit[s]) for s in sessions], dtype=np.float64)
        eff_random = np.asarray([np.nanmean(session_effects_random[s]) for s in sessions], dtype=np.float64)
        caps = np.asarray([np.nanmean(session_caps[s]) for s in sessions], dtype=np.float64)
        eff_mean, eff_lo, eff_hi = bootstrap_mean_ci(eff, rng=rng, n_bootstrap=n_bootstrap)
        unit_mean, unit_lo, unit_hi = bootstrap_mean_ci(eff_unit, rng=rng, n_bootstrap=n_bootstrap)
        rand_mean, rand_lo, rand_hi = bootstrap_mean_ci(eff_random, rng=rng, n_bootstrap=n_bootstrap)
        cap_mean, cap_lo, cap_hi = bootstrap_mean_ci(caps, rng=rng, n_bootstrap=n_bootstrap)
        eff_f = eff[np.isfinite(eff)]
        out.append(
            {
                "target_variant": target_variant,
                "projection_control": projection_control,
                "k": int(k),
                "context_subset": subset,
                "n_sessions": int(eff_f.size),
                "n_context_rows": int(len(group_rows)),
                "capture_mean": cap_mean,
                "capture_boot_ci_low": cap_lo,
                "capture_boot_ci_high": cap_hi,
                "effect_rf_readout_mean": eff_mean,
                "effect_rf_readout_boot_ci_low": eff_lo,
                "effect_rf_readout_boot_ci_high": eff_hi,
                "effect_unit_shuffle_mean": unit_mean,
                "effect_unit_shuffle_boot_ci_low": unit_lo,
                "effect_unit_shuffle_boot_ci_high": unit_hi,
                "effect_random_subspace_mean": rand_mean,
                "effect_random_subspace_boot_ci_low": rand_lo,
                "effect_random_subspace_boot_ci_high": rand_hi,
                "n_effect_rf_readout_positive": int(np.sum(eff_f > 0.0)),
                "sign_test_rf_readout_p_two_sided": sign_test_p_two_sided(int(np.sum(eff_f > 0.0)), int(eff_f.size)),
                "effect_rf_readout_min": float(np.min(eff_f)) if eff_f.size else float("nan"),
                "effect_rf_readout_max": float(np.max(eff_f)) if eff_f.size else float("nan"),
            }
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Direct recorded derivative / twin tangent alignment")
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
    p.add_argument("--target-variants", type=str, default="psd")
    p.add_argument("--k-list", type=str, default="2,5,10,20")
    p.add_argument("--ridge-multipliers", type=str, default="0,0.01,0.03,0.1,0.3,1,3,10")
    p.add_argument("--context-mode", choices=["time_bin", "time_window", "trial"], default="time_window")
    p.add_argument("--context-bin-size", type=int, default=10)
    p.add_argument("--min-samples-per-context", type=int, default=20)
    p.add_argument("--min-units", type=int, default=50)
    p.add_argument("--max-eye-condition", type=float, default=100.0)
    p.add_argument("--reliability-excess-threshold", type=float, default=0.0)
    p.add_argument("--n-reliability-shuffles", type=int, default=20)
    p.add_argument("--n-nulls", type=int, default=50)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose-model-load", action="store_true")
    p.add_argument("--enable-rf-readout-null", action="store_true", default=True)
    p.add_argument("--rf-null-min-bin-units", type=int, default=6)
    p.add_argument("--rf-null-bin-features", type=str, default="rf_xy,tangent_norm,mean_rate,ccnorm")
    p.add_argument("--rf-null-session-yaml-dir", type=Path, default=Path("experiments") / "dataset_configs" / "sessions")
    p.add_argument("--init-only", action="store_true", help="Write scaffold manifest/README and exit.")
    return p


def run_analysis(args: argparse.Namespace) -> None:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    projection_controls = parse_str_list(args.projection_controls)
    target_variants = parse_str_list(args.target_variants)
    k_list = parse_int_list(args.k_list)
    ridge_multipliers = parse_float_list(args.ridge_multipliers)
    fig3_rows = _load_pickle(Path(args.fig3_cache))
    fig2_rows = _load_pickle(Path(args.fig2_cache))
    fig2 = _fig2_by_session(fig2_rows)
    fig3_by_session = {str(row["session"]): row for row in fig3_rows}

    requested_sessions = parse_str_list(args.sessions)
    if len(requested_sessions) == 1 and requested_sessions[0].lower() == "all":
        requested_sessions = []
    if not requested_sessions:
        requested_sessions = [str(row["session"]) for row in fig3_rows if str(row.get("subject", "")) in {"Allen", "Logan"}]

    config = AnalysisConfig(
        output_root=str(out),
        sessions=requested_sessions,
        projection_controls=projection_controls,
        target_variants=target_variants,
        k_list=k_list,
        ridge_multipliers=ridge_multipliers,
        min_samples_per_context=int(args.min_samples_per_context),
        min_units=int(args.min_units),
        max_eye_condition=float(args.max_eye_condition),
        reliability_excess_threshold=float(args.reliability_excess_threshold),
        n_reliability_shuffles=int(args.n_reliability_shuffles),
        n_nulls=int(args.n_nulls),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
        context_mode=str(args.context_mode),
        context_bin_size=int(args.context_bin_size),
        compact_crossfit_group_mode=str(args.context_mode),
    )
    write_json(
        out / "recorded_derivative_manifest.json",
        {
            "status": "initialized_not_run" if bool(args.init_only) else "running",
            "analysis": "direct_recorded_derivative_twin_alignment",
            "config": asdict(config),
            "fig2_cache": str(Path(args.fig2_cache).resolve()),
            "fig3_cache": str(Path(args.fig3_cache).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "model_config": str(Path(args.model_config).resolve()),
            "dataset_config": str(Path(args.dataset_config).resolve()),
            "primary_gate": "Tier 1 must survive global_rate+target_pc1 and RF/readout-preserving null.",
        },
    )
    if bool(args.init_only):
        (out / "README.md").write_text("# Direct Recorded Derivative / Twin Tangent Alignment Outputs\n\nStatus: initialized, not yet run.\n", encoding="utf-8")
        return

    model, model_info = _load_twin_model(args)
    session_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    tier1_rows_all: list[dict[str, Any]] = []
    tier2_rows_all: list[dict[str, Any]] = []
    tier3_rows_all: list[dict[str, Any]] = []
    null_rows_all: list[dict[str, Any]] = []
    rf_rows_all: list[dict[str, Any]] = []

    for session in requested_sessions:
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
        eye_px = samples.eyepos_deg * float(samples.pixels_per_degree)
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
        rf_rows_all.extend(rf_meta.unit_rows)
        contexts, inv_rows, rel_rows = fit_context_derivatives(
            samples=samples,
            j=j,
            eye_px=eye_px,
            ridge_multipliers=ridge_multipliers,
            min_samples=int(args.min_samples_per_context),
            max_eye_condition=float(args.max_eye_condition),
            reliability_excess_threshold=float(args.reliability_excess_threshold),
            n_reliability_shuffles=int(args.n_reliability_shuffles),
            seed=int(args.seed) + dataset_idx * 101,
            context_mode=str(args.context_mode),
            context_bin_size=int(args.context_bin_size),
        )
        for row in inv_rows:
            row.update({"session": session, "subject": sr.get("subject", "")})
        for row in rel_rows:
            row.update({"session": session, "subject": sr.get("subject", "")})
        inventory_rows.extend(inv_rows)
        reliability_rows.extend(rel_rows)
        if len(contexts) < 2:
            session_rows.append({"session": session, "status": "too_few_contexts", "n_contexts_ok": int(len(contexts)), "n_common_units": int(common_units.size)})
            continue
        tier1, tier2, tier3, nulls = compute_tier_rows(
            session=session,
            subject=str(sr.get("subject", "")),
            contexts=contexts,
            j=j,
            target_raw=target_raw,
            target_psd=target_psd,
            projection_controls=projection_controls,
            target_variants=target_variants,
            k_list=k_list,
            rf_bins=rf_meta.bins,
            n_nulls=int(args.n_nulls),
            seed=int(args.seed) + dataset_idx * 1009,
        )
        tier1_rows_all.extend(tier1)
        tier2_rows_all.extend(tier2)
        tier3_rows_all.extend(tier3)
        null_rows_all.extend(nulls)
        session_rows.append(
            {
                "session": session,
                "subject": sr.get("subject", ""),
                "status": "ok",
                "dataset_idx": int(dataset_idx),
                "device": str(model.device),
                "n_common_units": int(common_units.size),
                "n_samples_used": int(samples.source_indices.size),
                "n_candidate_samples": int(samples.n_candidate_samples),
                "n_contexts_ok": int(len(contexts)),
                "n_contexts_reliability_qualified": int(sum(c.reliability_qualified for c in contexts)),
                "context_mode": str(args.context_mode),
                "rescale_status": rescale_status,
                "rf_null_status": rf_meta.status,
                "rf_null_n_bins": int(rf_meta.n_bins),
                "rf_null_bin_features": rf_meta.bin_features,
                "jacobian_abs_median": float(np.median(np.abs(j))),
                **target_meta,
            }
        )

    summary_rows = summarize_tier1(tier1_rows_all, seed=int(args.seed), n_bootstrap=int(args.n_bootstrap))
    write_csv(out / "session_summary.csv", session_rows)
    write_csv(out / "context_inventory.csv", inventory_rows)
    write_csv(out / "recorded_derivative_reliability.csv", reliability_rows)
    write_csv(out / "tier1_compact_basis_capture.csv", tier1_rows_all)
    write_csv(out / "tier1_compact_basis_bootstrap_summary.csv", summary_rows)
    write_csv(out / "tier2_matched_derivative_alignment.csv", tier2_rows_all)
    write_csv(out / "tier3_signed_axis_diagnostics.csv", tier3_rows_all)
    write_csv(out / "null_summary.csv", null_rows_all)
    write_csv(out / "rf_readout_unit_bins.csv", rf_rows_all)

    audit = {
        "status": "ok",
        "n_sessions_requested": int(len(requested_sessions)),
        "n_sessions_ok": int(sum(1 for r in session_rows if r.get("status") == "ok")),
        "n_context_rows": int(len(inventory_rows)),
        "n_reliability_rows": int(len(reliability_rows)),
        "n_tier1_rows": int(len(tier1_rows_all)),
        "model_info": {k: str(v) for k, v in dict(model_info).items()},
        "manifest_device": str(getattr(model, "device", "")),
        "notes": [
            "Context mode defaults to time_window because fixRSVP datasets expose trial/time covariates but no explicit image identity here.",
            "Ridge is selected per context by recorded split-half derivative reliability, never by twin alignment.",
            "RF/readout null uses fixed within-bin permutations from the covariance closure binning machinery.",
        ],
    }
    write_json(out / "audit.json", audit)
    write_json(
        out / "recorded_derivative_manifest.json",
        {
            "status": "ok",
            "analysis": "direct_recorded_derivative_twin_alignment",
            "config": asdict(config),
            "fig2_cache": str(Path(args.fig2_cache).resolve()),
            "fig3_cache": str(Path(args.fig3_cache).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "model_config": str(Path(args.model_config).resolve()),
            "dataset_config": str(Path(args.dataset_config).resolve()),
            "model_info": {k: str(v) for k, v in dict(model_info).items()},
            "primary_gate": "Tier 1 must survive global_rate+target_pc1 and RF/readout-preserving null.",
        },
    )
    (out / "README.md").write_text(
        "# Direct Recorded Derivative / Twin Tangent Alignment Outputs\n\n"
        "Primary output is `tier1_compact_basis_bootstrap_summary.csv`. Treat this as promotable only if the "
        "`reliability_qualified`, `psd`, `global_rate+target_pc1`, `k=10` row is positive over the RF/readout null "
        "with a session-bootstrap CI above zero and most sessions positive.\n",
        encoding="utf-8",
    )


def main() -> None:
    run_analysis(build_parser().parse_args())


if __name__ == "__main__":
    main()
