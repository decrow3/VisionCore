from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from eval.eval_stack_utils import rescale_rhat

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
    SessionSamples,
    _behavior_batch,
    _collect_samples,
    _compute_jacobians,
    _load_twin_model,
    _predict,
    _shift_stimulus_batch,
    _stim_batch,
    _target_for_session,
    _tangent_matrix,
)


DEFAULT_OUTPUT_ROOT = Path("outputs") / "forward_twin_reafferent_denoising"


@dataclass
class AnalysisConfig:
    output_root: str
    sessions: list[str]
    window_idx: int
    max_samples: int
    n_folds: int
    n_nulls: int
    n_bootstrap: int
    n_eye_shuffle_nulls: int
    compact_k: int
    fem_rank: int
    projection_controls: list[str]
    target_variants: list[str]
    include_full_forward: bool
    eye_reference: str
    stabilized_behavior: str
    fold_mode: str
    fit_alpha: str
    seed: int


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_str_list(raw: str) -> list[str]:
    return [piece.strip() for piece in str(raw).split(",") if piece.strip()]


def _group_folds(groups_for_rows: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    groups = np.unique(np.asarray(groups_for_rows, dtype=np.int64))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(groups)
    return [part for part in np.array_split(groups, min(int(n_folds), groups.size)) if part.size]


def _fold_groups(samples: SessionSamples, mode: str, n_folds: int, seed: int) -> tuple[list[np.ndarray], np.ndarray, str]:
    if str(mode) == "image_time":
        row_groups = np.asarray(samples.time_indices, dtype=np.int64)
        label = "time_indices"
    else:
        row_groups = np.asarray(samples.trial_ids, dtype=np.int64)
        label = "trial_ids"
    return _group_folds(row_groups, n_folds, seed), row_groups, label


def _crossfit_psth_residuals(
    samples: SessionSamples,
    train_mask: np.ndarray,
    which_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return held-out residuals and the corresponding PSTH prediction.

    PSTH is estimated as the training-fold mean response for the same fixRSVP
    time index. If a time index is absent in training, the training global mean
    is used as a conservative fallback.
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    which_mask = np.asarray(which_mask, dtype=bool)
    robs = np.asarray(samples.robs, dtype=np.float64)
    time = np.asarray(samples.time_indices, dtype=np.int64)
    train_mean = np.nanmean(robs[train_mask], axis=0)
    by_time: dict[int, np.ndarray] = {}
    for t in np.unique(time[train_mask]):
        local = train_mask & (time == int(t))
        if np.any(local):
            by_time[int(t)] = np.nanmean(robs[local], axis=0)
    pred = np.vstack([by_time.get(int(t), train_mean) for t in time[which_mask]])
    return robs[which_mask] - pred, pred


def _orth(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim == 1:
        m = m[:, None]
    if m.ndim != 2 or m.size == 0:
        return np.zeros((m.shape[0] if m.ndim == 2 else 0, 0), dtype=np.float64)
    u, s, _ = np.linalg.svd(m, full_matrices=False)
    if s.size == 0:
        return u[:, :0]
    keep = s > max(float(s[0]), 1.0) * float(eps)
    return u[:, keep]


def _basis_from_train_tangents(j: np.ndarray, train_mask: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    mat = _tangent_matrix(np.asarray(j, dtype=np.float64), np.asarray(train_mask, dtype=bool))
    if mat.shape[1] == 0 or not np.isfinite(mat).all():
        return np.zeros((j.shape[1], 0), dtype=np.float64), 0
    vals, vecs = np.linalg.eigh(_sym(mat @ mat.T))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    rank = int(np.sum(vals > max(float(vals[0]) if vals.size else 0.0, 1.0) * 1e-10))
    use_k = min(int(k), int(rank), int(vecs.shape[1]))
    return vecs[:, :use_k], rank


def _random_basis(n_units: int, k: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((int(n_units), max(int(k), 1))))
    return q[:, : int(k)]


def _project_rows(rows: np.ndarray, basis: np.ndarray) -> np.ndarray:
    u = _orth(basis)
    if u.shape[1] == 0:
        return np.zeros_like(rows, dtype=np.float64)
    return np.asarray(rows, dtype=np.float64) @ u @ u.T


def _global_gain_projection(rows: np.ndarray) -> np.ndarray:
    n_units = int(np.asarray(rows).shape[1])
    q = np.ones((n_units, 1), dtype=np.float64) / np.sqrt(max(n_units, 1))
    return np.asarray(rows, dtype=np.float64) @ q @ q.T


def _fit_scalar_alpha(resid: np.ndarray, delta: np.ndarray, mode: str) -> float:
    if str(mode) == "fixed_1":
        return 1.0
    r = np.asarray(resid, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    keep = np.isfinite(r) & np.isfinite(d)
    denom = float(np.sum(d[keep] * d[keep]))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(r[keep] * d[keep]) / denom)


def _sse(arr: np.ndarray) -> float:
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.sum(x * x))


def _mean_noise_corr(resid: np.ndarray) -> float:
    r = np.asarray(resid, dtype=np.float64)
    keep = np.isfinite(r).all(axis=1)
    r = r[keep]
    if r.shape[0] < 3 or r.shape[1] < 2:
        return float("nan")
    c = np.corrcoef(r, rowvar=False)
    if c.ndim != 2:
        return float("nan")
    iu = np.triu_indices(c.shape[0], k=1)
    vals = c[iu]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


def _cov_rows(rows: np.ndarray) -> np.ndarray:
    x = np.asarray(rows, dtype=np.float64)
    keep = np.isfinite(x).all(axis=1)
    x = x[keep]
    if x.shape[0] < 2:
        return np.full((rows.shape[1], rows.shape[1]), np.nan, dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    return _sym((x.T @ x) / max(x.shape[0] - 1, 1))


def _fem_subspace_ratio(base: np.ndarray, clean: np.ndarray, target: np.ndarray, rank: int) -> float:
    vals, vecs = np.linalg.eigh(_sym(np.asarray(target, dtype=np.float64)))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    keep = vals > max(float(vals[0]) if vals.size else 0.0, 1.0) * 1e-10
    use = min(int(rank), int(np.sum(keep)), int(vecs.shape[1]))
    if use <= 0:
        return float("nan")
    u = vecs[:, :use]
    cov_base = _cov_rows(base)
    cov_clean = _cov_rows(clean)
    denom = float(np.trace(u.T @ cov_base @ u))
    if denom <= 1e-12 or not np.isfinite(denom):
        return float("nan")
    return float(1.0 - np.trace(u.T @ cov_clean @ u) / denom)


def _bootstrap_mean_ci(vals: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float, float]:
    x = np.asarray(vals, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(x))
    if x.size < 2 or int(n_boot) <= 0:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, x.size, size=(int(n_boot), x.size))
    boot = np.mean(x[idx], axis=1)
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _fit_rescale_gains_on_mask(
    *,
    model: Any,
    dset: Any,
    stim_lags: np.ndarray,
    samples: SessionSamples,
    common_units: np.ndarray,
    dataset_idx: int,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    if str(args.rescale_mode) == "none":
        return np.ones(common_units.size, dtype=np.float64), "none"
    mask = np.asarray(mask, dtype=bool)
    source_indices = np.asarray(samples.source_indices[mask], dtype=np.int64)
    if source_indices.size < 3:
        return np.ones(common_units.size, dtype=np.float64), "too_few_train_samples"
    preds = []
    for start in range(0, source_indices.size, int(args.batch_size)):
        idx = source_indices[start : start + int(args.batch_size)]
        stim = _stim_batch(dset, idx, stim_lags)
        behavior = _behavior_batch(dset, idx)
        pred = _predict(model, stim, behavior, dataset_idx).detach().cpu().numpy()
        preds.append(pred[:, common_units])
    rhat = torch.as_tensor(np.concatenate(preds, axis=0), dtype=torch.float32)
    robs = torch.as_tensor(np.asarray(samples.robs[mask], dtype=np.float32))
    dfs = torch.as_tensor(np.asarray(samples.dfs[mask], dtype=np.float32))
    try:
        _, scale_model = rescale_rhat(robs, rhat, dfs, mode=str(args.rescale_mode))
        if hasattr(scale_model, "g"):
            gains = torch.exp(scale_model.g).detach().cpu().numpy()
            if gains.ndim == 0:
                gains = np.full(common_units.size, float(gains), dtype=np.float64)
            return np.asarray(gains, dtype=np.float64), f"{args.rescale_mode}_train_fold"
    except Exception as exc:
        return np.ones(common_units.size, dtype=np.float64), f"failed:{type(exc).__name__}"
    return np.ones(common_units.size, dtype=np.float64), "unavailable_gain"


def _eye_positions_px(samples: SessionSamples) -> np.ndarray:
    return np.asarray(samples.eyepos_deg, dtype=np.float64) * float(samples.pixels_per_degree)


def _eye_displacements_px(eye_px: np.ndarray, args: argparse.Namespace, train_mask: np.ndarray | None) -> np.ndarray:
    eye_px = np.asarray(eye_px, dtype=np.float64)
    if str(args.eye_reference) == "session_mean":
        if train_mask is None:
            raise ValueError("train_mask is required when eye_reference='session_mean'")
        return eye_px - np.mean(eye_px[np.asarray(train_mask, dtype=bool)], axis=0, keepdims=True)
    return eye_px


def _behavior_baseline_from_mask(dset: Any, source_indices: np.ndarray) -> torch.Tensor:
    behavior = _behavior_batch(dset, np.asarray(source_indices, dtype=np.int64))
    return torch.mean(behavior, dim=0, keepdim=True)


def _stabilized_behavior_batch(
    behavior: torch.Tensor,
    mode: str,
    baseline_behavior: torch.Tensor | None,
) -> torch.Tensor:
    if str(mode) == "zero":
        return torch.zeros_like(behavior)
    if str(mode) == "train_mean":
        if baseline_behavior is None:
            raise ValueError("baseline_behavior is required when stabilized_behavior='train_mean'")
        return baseline_behavior.to(device=behavior.device, dtype=behavior.dtype).expand_as(behavior)
    return behavior


def _compute_full_forward_delta(
    *,
    model: Any,
    dset: Any,
    stim_lags: np.ndarray,
    samples: SessionSamples,
    common_units: np.ndarray,
    dataset_idx: int,
    displacements_px: np.ndarray,
    stabilized_behavior: str,
    baseline_behavior: torch.Tensor | None,
    args: argparse.Namespace,
) -> np.ndarray:
    """Compute twin(shifted stimulus) - twin(stabilized stimulus)."""
    displacements_px = np.asarray(displacements_px, dtype=np.float64)
    rows: list[np.ndarray] = []
    for start in range(0, samples.source_indices.size, int(args.batch_size)):
        idx = samples.source_indices[start : start + int(args.batch_size)]
        local_eye = displacements_px[start : start + idx.size]
        stim = _stim_batch(dset, idx, stim_lags).to(model.device)
        behavior = _behavior_batch(dset, idx)
        base_behavior = _stabilized_behavior_batch(behavior, stabilized_behavior, baseline_behavior)
        base = _predict(model, stim, base_behavior, dataset_idx).detach().cpu().numpy()
        shifted = _predict(model, _shift_stimulus_batch(stim, local_eye), behavior, dataset_idx).detach().cpu().numpy()
        delta = (shifted[:, common_units] - base[:, common_units]).astype(np.float64)
        rows.append(delta)
    return np.concatenate(rows, axis=0)


def _compute_full_forward_delta_for_rows(
    *,
    model: Any,
    dset: Any,
    stim_lags: np.ndarray,
    source_indices: np.ndarray,
    common_units: np.ndarray,
    dataset_idx: int,
    displacements_px: np.ndarray,
    stabilized_behavior: str,
    baseline_behavior: torch.Tensor | None,
    args: argparse.Namespace,
) -> np.ndarray:
    displacements_px = np.asarray(displacements_px, dtype=np.float64)
    rows: list[np.ndarray] = []
    for start in range(0, source_indices.size, int(args.batch_size)):
        idx = np.asarray(source_indices[start : start + int(args.batch_size)], dtype=np.int64)
        local_eye = displacements_px[start : start + idx.size]
        stim = _stim_batch(dset, idx, stim_lags).to(model.device)
        behavior = _behavior_batch(dset, idx)
        base_behavior = _stabilized_behavior_batch(behavior, stabilized_behavior, baseline_behavior)
        base = _predict(model, stim, base_behavior, dataset_idx).detach().cpu().numpy()
        shifted = _predict(model, _shift_stimulus_batch(stim, local_eye), behavior, dataset_idx).detach().cpu().numpy()
        rows.append((shifted[:, common_units] - base[:, common_units]).astype(np.float64))
    return np.concatenate(rows, axis=0)


def _linear_delta_from_displacements(j: np.ndarray, displacements_px: np.ndarray) -> np.ndarray:
    return np.einsum("nua,na->nu", np.asarray(j, dtype=np.float64), np.asarray(displacements_px, dtype=np.float64))


def _candidate_deltas_for_fold(
    *,
    base_delta: np.ndarray,
    j: np.ndarray,
    train_mask: np.ndarray,
    compact_k: int,
    n_nulls: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    n_units = int(base_delta.shape[1])
    out: dict[str, np.ndarray] = {
        "forward_full_space": np.asarray(base_delta, dtype=np.float64),
        "gain_only": _global_gain_projection(base_delta),
    }

    compact_basis, compact_rank = _basis_from_train_tangents(j, train_mask, compact_k)
    if compact_basis.shape[1] > 0:
        out[f"compact_k{int(compact_k)}"] = _project_rows(base_delta, compact_basis)
        for i in range(int(n_nulls)):
            perm = rng.permutation(n_units)
            out[f"unit_shuffled_compact_k{int(compact_k)}_null{i:03d}"] = _project_rows(base_delta, compact_basis[perm, :])
    else:
        out[f"compact_k{int(compact_k)}"] = np.zeros_like(base_delta)

    for i in range(int(n_nulls)):
        rb = _random_basis(n_units, min(int(compact_k), n_units), rng)
        out[f"random_k{int(compact_k)}_null{i:03d}"] = _project_rows(base_delta, rb)

    out["_compact_rank_train"] = np.full_like(base_delta, float(compact_rank))
    out["_compact_basis"] = compact_basis
    return out


def _summarize_session_rows(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not fold_rows:
        return out
    keys = ["session", "subject", "target_variant", "projection_control", "base_delta_source", "correction"]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in fold_rows:
        groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        n = np.asarray([float(r["n_test_samples"]) for r in rows], dtype=np.float64)
        weights = n / max(float(np.sum(n)), 1.0)
        def wmean(name: str) -> float:
            vals = np.asarray([float(r.get(name, np.nan)) for r in rows], dtype=np.float64)
            keep = np.isfinite(vals) & np.isfinite(weights)
            return float(np.sum(vals[keep] * weights[keep]) / max(np.sum(weights[keep]), 1e-12)) if np.any(keep) else float("nan")
        rec = dict(zip(keys, key, strict=True))
        base_sse = float(np.sum([float(r.get("base_sse", np.nan)) for r in rows if np.isfinite(float(r.get("base_sse", np.nan)))]))
        clean_sse = float(np.sum([float(r.get("clean_sse", np.nan)) for r in rows if np.isfinite(float(r.get("clean_sse", np.nan)))]))
        rec.update(
            {
                "n_folds": int(len(rows)),
                "n_test_samples": int(np.sum(n)),
                "n_test_trials": int(sum(int(r.get("n_test_trials", 0)) for r in rows)),
                "variance_reduction": float(1.0 - clean_sse / base_sse) if base_sse > 0.0 else float("nan"),
                "noise_corr_reduction": wmean("noise_corr_reduction"),
                "fem_subspace_reduction": wmean("fem_subspace_reduction"),
                "alpha_mean": wmean("alpha"),
                "gain_median": wmean("gain_median"),
                "base_sse": base_sse,
                "clean_sse": clean_sse,
            }
        )
        out.append(rec)
    return out


def _summarize_bootstrap(session_rows: list[dict[str, Any]], *, n_boot: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in session_rows:
        key = (
            str(row["target_variant"]),
            str(row["projection_control"]),
            str(row["base_delta_source"]),
            str(row["correction"]),
            "all_sessions",
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        vals = np.asarray([float(r["variance_reduction"]) for r in rows], dtype=np.float64)
        nc = np.asarray([float(r["noise_corr_reduction"]) for r in rows], dtype=np.float64)
        fem = np.asarray([float(r["fem_subspace_reduction"]) for r in rows], dtype=np.float64)
        vr_mean, vr_lo, vr_hi = _bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
        nc_mean, nc_lo, nc_hi = _bootstrap_mean_ci(nc, n_boot=n_boot, seed=seed + 11)
        fem_mean, fem_lo, fem_hi = _bootstrap_mean_ci(fem, n_boot=n_boot, seed=seed + 23)
        out.append(
            {
                "target_variant": key[0],
                "projection_control": key[1],
                "base_delta_source": key[2],
                "correction": key[3],
                "n_sessions": int(len(rows)),
                "variance_reduction_mean": vr_mean,
                "variance_reduction_boot_ci_low": vr_lo,
                "variance_reduction_boot_ci_high": vr_hi,
                "noise_corr_reduction_mean": nc_mean,
                "noise_corr_reduction_boot_ci_low": nc_lo,
                "noise_corr_reduction_boot_ci_high": nc_hi,
                "fem_subspace_reduction_mean": fem_mean,
                "fem_subspace_reduction_boot_ci_low": fem_lo,
                "fem_subspace_reduction_boot_ci_high": fem_hi,
                "n_vr_positive": int(np.sum(vals[np.isfinite(vals)] > 0.0)),
                "n_vr_nonzero": int(np.sum(np.isfinite(vals))),
            }
        )
    return out


def _correction_family(correction: str) -> str:
    name = str(correction)
    if name.startswith("random_k"):
        return "random_k"
    if name.startswith("unit_shuffled_compact"):
        return "unit_shuffled_compact"
    if name.startswith("shuffled_eye_trace_compact"):
        return "shuffled_eye_trace_compact"
    if name.startswith("shuffled_eye_trace"):
        return "shuffled_eye_trace"
    return name


def _collapse_control_families(session_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["session", "subject", "target_variant", "projection_control", "base_delta_source"]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in session_rows:
        family = _correction_family(str(row["correction"]))
        groups.setdefault(tuple(row.get(k) for k in keys) + (family,), []).append(row)
    out: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        rec = dict(zip(keys + ["correction_family"], key, strict=True))
        for metric in ["variance_reduction", "noise_corr_reduction", "fem_subspace_reduction", "alpha_mean"]:
            vals = np.asarray([float(r.get(metric, np.nan)) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            rec[metric] = float(np.median(vals)) if vals.size else float("nan")
        rec["n_family_rows"] = int(len(rows))
        rec["corrections_collapsed"] = ";".join(sorted(str(r["correction"]) for r in rows))
        out.append(rec)
    return out


def _summarize_family_bootstrap(family_rows: list[dict[str, Any]], *, n_boot: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in family_rows:
        key = (
            str(row["target_variant"]),
            str(row["projection_control"]),
            str(row["base_delta_source"]),
            str(row["correction_family"]),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        rec = {
            "target_variant": key[0],
            "projection_control": key[1],
            "base_delta_source": key[2],
            "correction_family": key[3],
            "n_sessions": int(len(rows)),
        }
        for i, metric in enumerate(["variance_reduction", "noise_corr_reduction", "fem_subspace_reduction"]):
            vals = np.asarray([float(r.get(metric, np.nan)) for r in rows], dtype=np.float64)
            mean, lo, hi = _bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed + 101 * i)
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_boot_ci_low"] = lo
            rec[f"{metric}_boot_ci_high"] = hi
            rec[f"n_{metric}_positive"] = int(np.sum(vals[np.isfinite(vals)] > 0.0))
            rec[f"n_{metric}_nonzero"] = int(np.sum(np.isfinite(vals)))
        out.append(rec)
    return out


def _summarize_compact_excess(family_rows: list[dict[str, Any]], *, compact_k: int, n_boot: int, seed: int) -> list[dict[str, Any]]:
    compact_name = f"compact_k{int(compact_k)}"
    index: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in family_rows:
        key = (
            str(row["session"]),
            str(row["target_variant"]),
            str(row["projection_control"]),
            str(row["base_delta_source"]),
            str(row["correction_family"]),
        )
        index[key] = row
    pair_rows: list[dict[str, Any]] = []
    controls = {
        "gain_only",
        "shuffled_eye_trace",
        "shuffled_eye_trace_compact",
        "random_k",
        "unit_shuffled_compact",
        "forward_full_space",
    }
    for key, compact in index.items():
        session, target_variant, projection_control, base_source, family = key
        if family != compact_name:
            continue
        for control in controls:
            ctrl = index.get((session, target_variant, projection_control, base_source, control))
            if ctrl is None:
                continue
            row = {
                "session": session,
                "target_variant": target_variant,
                "projection_control": projection_control,
                "base_delta_source": base_source,
                "contrast": f"{compact_name}_minus_{control}",
            }
            for metric in ["variance_reduction", "noise_corr_reduction", "fem_subspace_reduction"]:
                row[f"{metric}_excess"] = float(compact[metric]) - float(ctrl[metric])
            pair_rows.append(row)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in pair_rows:
        key = (
            str(row["target_variant"]),
            str(row["projection_control"]),
            str(row["base_delta_source"]),
            str(row["contrast"]),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        rec = {
            "target_variant": key[0],
            "projection_control": key[1],
            "base_delta_source": key[2],
            "contrast": key[3],
            "n_sessions": int(len(rows)),
        }
        for i, metric in enumerate(["variance_reduction_excess", "noise_corr_reduction_excess", "fem_subspace_reduction_excess"]):
            vals = np.asarray([float(r.get(metric, np.nan)) for r in rows], dtype=np.float64)
            mean, lo, hi = _bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed + 211 * i)
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_boot_ci_low"] = lo
            rec[f"{metric}_boot_ci_high"] = hi
            rec[f"n_{metric}_positive"] = int(np.sum(vals[np.isfinite(vals)] > 0.0))
            rec[f"n_{metric}_nonzero"] = int(np.sum(np.isfinite(vals)))
        out.append(rec)
    return out


def run_analysis(args: argparse.Namespace) -> None:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    fig3_rows = _load_pickle(Path(args.fig3_cache))
    fig2_rows = _load_pickle(Path(args.fig2_cache))
    fig2 = _fig2_by_session(fig2_rows)
    fig3_by_session = {str(row["session"]): row for row in fig3_rows}
    requested_sessions = parse_str_list(args.sessions)
    if len(requested_sessions) == 1 and requested_sessions[0].lower() == "all":
        requested_sessions = [
            str(row["session"])
            for row in fig3_rows
            if str(row.get("subject", "")) in {"Allen", "Logan"}
        ]

    config = AnalysisConfig(
        output_root=str(out),
        sessions=requested_sessions,
        window_idx=int(args.window_idx),
        max_samples=int(args.max_samples),
        n_folds=int(args.n_folds),
        n_nulls=int(args.n_nulls),
        n_bootstrap=int(args.n_bootstrap),
        n_eye_shuffle_nulls=int(args.n_eye_shuffle_nulls),
        compact_k=int(args.compact_k),
        fem_rank=int(args.fem_rank),
        projection_controls=parse_str_list(args.projection_controls),
        target_variants=parse_str_list(args.target_variants),
        include_full_forward=bool(args.include_full_forward),
        eye_reference=str(args.eye_reference),
        stabilized_behavior=str(args.stabilized_behavior),
        fold_mode=str(args.fold_mode),
        fit_alpha=str(args.fit_alpha),
        seed=int(args.seed),
    )
    write_json(
        out / "manifest.json",
        {
            "status": "running",
            "analysis": "forward_twin_reafferent_denoising",
            "config": asdict(config),
            "fig2_cache": str(Path(args.fig2_cache).resolve()),
            "fig3_cache": str(Path(args.fig3_cache).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "model_config": str(Path(args.model_config).resolve()),
            "dataset_config": str(Path(args.dataset_config).resolve()),
            "leakage_guardrail": "PSTH, scalar correction amplitude, response gains, and compact/random/unit-shuffle bases are fit on training rows only within each session fold.",
        },
    )

    model, model_info = _load_twin_model(args)
    session_inventory: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    rng_global = np.random.default_rng(int(args.seed))

    for session in requested_sessions:
        if session not in fig3_by_session or session not in fig2:
            session_inventory.append({"session": session, "status": "missing_fig2_or_fig3"})
            continue
        if session not in getattr(model, "names", []):
            session_inventory.append({"session": session, "status": "missing_model_session"})
            continue
        dataset_idx = int(model.names.index(session))
        sr = fig3_by_session[session]
        common_units, target_raw, target_psd, target_meta = _target_for_session(fig2[session], sr, args)
        if common_units.size < int(args.min_units):
            session_inventory.append({"session": session, "status": "too_few_common_units", "n_common_units": int(common_units.size)})
            continue

        dset, stim_lags, samples = _collect_samples(model=model, dataset_idx=dataset_idx, common_units=common_units, args=args)
        unit_gains = np.ones(common_units.size, dtype=np.float64)
        j_unscaled = _compute_jacobians(
            model=model,
            dset=dset,
            stim_lags=stim_lags,
            samples=samples,
            common_units=common_units,
            gains=unit_gains,
            dataset_idx=dataset_idx,
            args=args,
        )
        eye_px = _eye_positions_px(samples)
        fold_invariant_eye_reference = str(args.eye_reference) == "zero"
        displacements_px = _eye_displacements_px(eye_px, args, None) if fold_invariant_eye_reference else None
        linear_delta_unscaled = (
            _linear_delta_from_displacements(j_unscaled, displacements_px)
            if displacements_px is not None
            else None
        )
        delta_sources_unscaled: dict[str, np.ndarray] = {}
        if linear_delta_unscaled is not None:
            delta_sources_unscaled["linear_tangent"] = linear_delta_unscaled
        precomputed_full_forward_unscaled: np.ndarray | None = None
        if (
            bool(args.include_full_forward)
            and str(args.stabilized_behavior) != "train_mean"
            and displacements_px is not None
        ):
            delta_sources_unscaled["full_forward"] = _compute_full_forward_delta(
                model=model,
                dset=dset,
                stim_lags=stim_lags,
                samples=samples,
                common_units=common_units,
                dataset_idx=dataset_idx,
                displacements_px=displacements_px,
                stabilized_behavior=str(args.stabilized_behavior),
                baseline_behavior=None,
                args=args,
            )
            precomputed_full_forward_unscaled = delta_sources_unscaled["full_forward"]

        folds, row_fold_groups, fold_group_label = _fold_groups(
            samples,
            str(args.fold_mode),
            int(args.n_folds),
            int(args.seed) + dataset_idx * 101,
        )
        if len(folds) < 2:
            session_inventory.append({"session": session, "status": "too_few_trial_folds", "n_common_units": int(common_units.size)})
            continue
        targets = {"raw": target_raw, "psd": target_psd}
        session_inventory.append(
            {
                "session": session,
                "subject": sr.get("subject", ""),
                "status": "ok",
                "dataset_idx": int(dataset_idx),
                "device": str(model.device),
                "n_common_units": int(common_units.size),
                "n_samples_used": int(samples.source_indices.size),
                "n_candidate_samples": int(samples.n_candidate_samples),
                "n_trials": int(np.unique(samples.trial_ids).size),
                "fold_mode": str(args.fold_mode),
                "fold_group_label": fold_group_label,
                "n_folds": int(len(folds)),
                "n_eye_shuffle_nulls": int(args.n_eye_shuffle_nulls) if int(args.n_eye_shuffle_nulls) >= 0 else int(args.n_nulls),
                "rescale_status": f"{args.rescale_mode}_crossfit_per_fold",
                "gain_median": float("nan"),
                "jacobian_abs_median_unscaled": float(np.median(np.abs(j_unscaled))),
                "eye_reference": str(args.eye_reference),
                "stabilized_behavior": str(args.stabilized_behavior),
                **target_meta,
            }
        )

        for fold_idx, test_groups in enumerate(folds):
            test_mask = np.isin(row_fold_groups, test_groups)
            train_mask = ~test_mask
            if int(np.sum(test_mask)) < 3 or int(np.sum(train_mask)) < 3:
                continue
            gains, rescale_status = _fit_rescale_gains_on_mask(
                model=model,
                dset=dset,
                stim_lags=stim_lags,
                samples=samples,
                common_units=common_units,
                dataset_idx=dataset_idx,
                mask=train_mask,
                args=args,
            )
            j_fold = j_unscaled * gains[None, :, None]
            baseline_behavior = None
            if str(args.stabilized_behavior) == "train_mean":
                baseline_behavior = _behavior_baseline_from_mask(dset, samples.source_indices[train_mask])
            fold_displacements_px = (
                displacements_px
                if displacements_px is not None
                else _eye_displacements_px(eye_px, args, train_mask)
            )
            linear_delta_unscaled_fold = (
                linear_delta_unscaled
                if linear_delta_unscaled is not None
                else _linear_delta_from_displacements(j_unscaled, fold_displacements_px)
            )
            delta_sources_unscaled_fold = dict(delta_sources_unscaled)
            delta_sources_unscaled_fold["linear_tangent"] = linear_delta_unscaled_fold
            if bool(args.include_full_forward) and precomputed_full_forward_unscaled is None:
                full_forward_fold_unscaled = np.zeros_like(linear_delta_unscaled_fold, dtype=np.float64)
                full_forward_fold_unscaled[train_mask] = _compute_full_forward_delta_for_rows(
                    model=model,
                    dset=dset,
                    stim_lags=stim_lags,
                    source_indices=samples.source_indices[train_mask],
                    common_units=common_units,
                    dataset_idx=dataset_idx,
                    displacements_px=fold_displacements_px[train_mask],
                    stabilized_behavior=str(args.stabilized_behavior),
                    baseline_behavior=baseline_behavior,
                    args=args,
                )
                full_forward_fold_unscaled[test_mask] = _compute_full_forward_delta_for_rows(
                    model=model,
                    dset=dset,
                    stim_lags=stim_lags,
                    source_indices=samples.source_indices[test_mask],
                    common_units=common_units,
                    dataset_idx=dataset_idx,
                    displacements_px=fold_displacements_px[test_mask],
                    stabilized_behavior=str(args.stabilized_behavior),
                    baseline_behavior=baseline_behavior,
                    args=args,
                )
                delta_sources_unscaled_fold["full_forward"] = full_forward_fold_unscaled
            delta_sources = {
                name: delta_unscaled * gains[None, :]
                for name, delta_unscaled in delta_sources_unscaled_fold.items()
            }
            train_resid_raw, _ = _crossfit_psth_residuals(samples, train_mask, train_mask)
            test_resid_raw, _ = _crossfit_psth_residuals(samples, train_mask, test_mask)
            for target_variant in parse_str_list(args.target_variants):
                target_base = targets[target_variant]
                for projection_control in parse_str_list(args.projection_controls):
                    modes = _projection_modes(projection_control, target_base)
                    p = _projection_complement(common_units.size, modes)
                    target_proj = p @ target_base @ p
                    train_resid = train_resid_raw @ p
                    test_resid = test_resid_raw @ p
                    base_sse = _sse(test_resid)
                    base_nc = _mean_noise_corr(test_resid)
                    for source_name, delta_base in delta_sources.items():
                        fold_rng = np.random.default_rng(int(rng_global.integers(0, 2**31 - 1)))
                        candidates = _candidate_deltas_for_fold(
                            base_delta=delta_base,
                            j=j_fold,
                            train_mask=train_mask,
                            compact_k=int(args.compact_k),
                            n_nulls=int(args.n_nulls),
                            rng=fold_rng,
                        )
                        train_rows = np.flatnonzero(train_mask)
                        test_rows = np.flatnonzero(test_mask)
                        compact_basis = candidates.get("_compact_basis", np.zeros((common_units.size, 0), dtype=np.float64))
                        n_eye_shuffle_nulls = int(args.n_eye_shuffle_nulls)
                        if n_eye_shuffle_nulls < 0:
                            n_eye_shuffle_nulls = int(args.n_nulls)
                        for eye_null_idx in range(n_eye_shuffle_nulls):
                            shuffled_all = np.zeros_like(delta_base, dtype=np.float64)
                            shuffled_train_disp = fold_displacements_px[fold_rng.permutation(train_rows)]
                            shuffled_test_disp = fold_displacements_px[fold_rng.permutation(test_rows)]
                            if source_name == "linear_tangent":
                                shuffled_all[train_mask] = _linear_delta_from_displacements(j_fold[train_mask], shuffled_train_disp)
                                shuffled_all[test_mask] = _linear_delta_from_displacements(j_fold[test_mask], shuffled_test_disp)
                            else:
                                shuffled_all[train_mask] = _compute_full_forward_delta_for_rows(
                                    model=model,
                                    dset=dset,
                                    stim_lags=stim_lags,
                                    source_indices=samples.source_indices[train_mask],
                                    common_units=common_units,
                                    dataset_idx=dataset_idx,
                                    displacements_px=shuffled_train_disp,
                                    stabilized_behavior=str(args.stabilized_behavior),
                                    baseline_behavior=baseline_behavior,
                                    args=args,
                                ) * gains[None, :]
                                shuffled_all[test_mask] = _compute_full_forward_delta_for_rows(
                                    model=model,
                                    dset=dset,
                                    stim_lags=stim_lags,
                                    source_indices=samples.source_indices[test_mask],
                                    common_units=common_units,
                                    dataset_idx=dataset_idx,
                                    displacements_px=shuffled_test_disp,
                                    stabilized_behavior=str(args.stabilized_behavior),
                                    baseline_behavior=baseline_behavior,
                                    args=args,
                                ) * gains[None, :]
                            candidates[f"shuffled_eye_trace_null{eye_null_idx:03d}"] = shuffled_all
                            candidates[f"shuffled_eye_trace_compact_k{int(args.compact_k)}_null{eye_null_idx:03d}"] = _project_rows(
                                shuffled_all,
                                compact_basis,
                            )
                        for corr_name, delta_all in candidates.items():
                            if corr_name.startswith("_"):
                                continue
                            delta_train = np.asarray(delta_all[train_mask], dtype=np.float64) @ p
                            delta_test = np.asarray(delta_all[test_mask], dtype=np.float64) @ p
                            alpha = _fit_scalar_alpha(train_resid, delta_train, str(args.fit_alpha))
                            clean = test_resid - float(alpha) * delta_test
                            clean_sse = _sse(clean)
                            clean_nc = _mean_noise_corr(clean)
                            fold_rows.append(
                                {
                                    "session": session,
                                    "subject": sr.get("subject", ""),
                                    "dataset_idx": int(dataset_idx),
                                    "fold": int(fold_idx),
                                    "target_variant": target_variant,
                                    "projection_control": projection_control,
                                    "base_delta_source": source_name,
                                    "correction": corr_name,
                                    "n_train_samples": int(np.sum(train_mask)),
                                    "n_test_samples": int(np.sum(test_mask)),
                                    "n_test_trials": int(np.unique(samples.trial_ids[test_mask]).size),
                                    "n_common_units": int(common_units.size),
                                    "alpha": float(alpha),
                                    "gain_median": float(np.median(gains)),
                                    "rescale_status": rescale_status,
                                    "base_sse": base_sse,
                                    "clean_sse": clean_sse,
                                    "variance_reduction": float(1.0 - clean_sse / base_sse) if base_sse > 0.0 else float("nan"),
                                    "base_noise_corr": base_nc,
                                    "clean_noise_corr": clean_nc,
                                    "noise_corr_reduction": float(base_nc - clean_nc) if np.isfinite(base_nc) and np.isfinite(clean_nc) else float("nan"),
                                    "fem_subspace_reduction": _fem_subspace_ratio(test_resid, clean, target_proj, int(args.fem_rank)),
                                    "compact_rank_train": int(candidates.get("_compact_rank_train", np.zeros_like(delta_base))[0, 0]),
                                    "fit_alpha_mode": str(args.fit_alpha),
                                    "eye_reference": str(args.eye_reference),
                                    "stabilized_behavior": str(args.stabilized_behavior),
                                    "fold_mode": str(args.fold_mode),
                                    "fold_group_label": fold_group_label,
                                    "window_idx": int(args.window_idx),
                                }
                            )

    session_rows = _summarize_session_rows(fold_rows)
    bootstrap_rows = _summarize_bootstrap(session_rows, n_boot=int(args.n_bootstrap), seed=int(args.seed))
    family_rows = _collapse_control_families(session_rows)
    family_bootstrap_rows = _summarize_family_bootstrap(family_rows, n_boot=int(args.n_bootstrap), seed=int(args.seed))
    compact_excess_rows = _summarize_compact_excess(
        family_rows,
        compact_k=int(args.compact_k),
        n_boot=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    write_csv(out / "fold_metrics.csv", fold_rows)
    write_csv(out / "session_summary.csv", session_rows)
    write_csv(out / "bootstrap_summary.csv", bootstrap_rows)
    write_csv(out / "session_family_summary.csv", family_rows)
    write_csv(out / "family_bootstrap_summary.csv", family_bootstrap_rows)
    write_csv(out / "compact_excess_bootstrap_summary.csv", compact_excess_rows)
    write_csv(out / "session_inventory.csv", session_inventory)
    write_json(
        out / "audit.json",
        {
            "status": "ok",
            "n_sessions_requested": int(len(requested_sessions)),
            "n_sessions_ok": int(sum(1 for r in session_inventory if r.get("status") == "ok")),
            "n_fold_rows": int(len(fold_rows)),
            "n_session_summary_rows": int(len(session_rows)),
            "n_session_family_rows": int(len(family_rows)),
            "n_compact_excess_rows": int(len(compact_excess_rows)),
            "model_info": {k: str(v) for k, v in dict(model_info).items()},
            "manifest_device": str(getattr(model, "device", "")),
            "notes": [
                "Matched recorded/twin unit space is used; no flexible recorded-noise map is fit.",
                "PSTH baseline is cross-fit from training trials using fixRSVP time-bin means.",
                "Response gains and scalar alpha are fit on training residuals only and applied to held-out rows.",
                "Compact, random, and unit-shuffled compact bases are built inside each training fold.",
                "Primary full_forward mode is a stabilized retinal-image control with behavior covariates held fixed.",
                "full_forward uses twin(real visual input, real behavior) - twin(stabilized visual input, stabilized_behavior covariates).",
                "By default eye_reference=zero, so displacements are raw eye position in pixels relative to zero stabilized position.",
                "By default stabilized_behavior=same, so behavior covariates are held unchanged in shifted and stabilized calls.",
                "stabilized_behavior=zero or train_mean are sensitivity analyses for behavioral eye-state pathways.",
                "shuffled_eye_trace permutes eye displacements within train/test splits and recomputes the same-row correction.",
                "shuffled_eye_trace_compact projects the shuffled-eye correction into the same train-fold compact tangent basis.",
            ],
        },
    )
    write_json(
        out / "manifest.json",
        {
            "status": "ok",
            "analysis": "forward_twin_reafferent_denoising",
            "config": asdict(config),
            "fig2_cache": str(Path(args.fig2_cache).resolve()),
            "fig3_cache": str(Path(args.fig3_cache).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "model_config": str(Path(args.model_config).resolve()),
            "dataset_config": str(Path(args.dataset_config).resolve()),
            "model_info": {k: str(v) for k, v in dict(model_info).items()},
        },
    )
    (out / "README.md").write_text(
        "# Forward Twin Reafferent Denoising Outputs\n\n"
        "This analysis tests whether model-derived retinal reafference predictions reduce held-out recorded residual structure.\n\n"
        "Primary files:\n"
        "- `fold_metrics.csv`: fold-level held-out denoising metrics.\n"
        "- `session_summary.csv`: fold-aggregated session metrics.\n"
        "- `bootstrap_summary.csv`: session-bootstrap summary by correction condition.\n"
        "- `session_family_summary.csv`: null replicates collapsed into control families.\n"
        "- `family_bootstrap_summary.csv`: bootstrap summary by collapsed control family.\n"
        "- `compact_excess_bootstrap_summary.csv`: compact correction minus matched controls.\n"
        "- `session_inventory.csv`: session inclusion and provenance audit.\n"
        "- `audit.json`, `manifest.json`: configuration and leakage guardrails.\n\n"
        "The main metric is held-out residual variance reduction relative to cross-fit PSTH residuals, "
        "computed from summed fold SSEs at the session level. "
        "Noise-correlation reduction and recorded-FEM-subspace reduction are also reported. "
        "The primary full-forward correction is a stabilized retinal-image control with behavior covariates held fixed "
        "(`stabilized_behavior=same`). Use `stabilized_behavior=zero` or `train_mean` as a sensitivity analysis for "
        "behavioral eye-state pathways. "
        "`shuffled_eye_trace` permutes eye displacements within train/test splits and recomputes the same-row correction. "
        "`shuffled_eye_trace_compact` projects that shuffled-eye correction into the same train-fold compact tangent basis.\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Forward twin reafferent denoising analysis.")
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
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--step-px", type=float, default=0.5)
    p.add_argument("--pixels-per-degree-fallback", type=float, default=37.5)
    p.add_argument("--fixation-radius-deg", type=float, default=1.0)
    p.add_argument("--sample-dfs-mode", choices=["all", "any", "none"], default="all")
    p.add_argument("--rescale-mode", choices=["none", "globalgain", "gain", "globalaffine", "affine"], default="affine")
    p.add_argument("--projection-controls", type=str, default="global_rate+target_pc1")
    p.add_argument("--target-variants", type=str, default="psd")
    p.add_argument("--eye-reference", choices=["zero", "session_mean"], default="zero")
    p.add_argument("--stabilized-behavior", choices=["same", "zero", "train_mean"], default="same")
    p.add_argument("--fold-mode", choices=["trial", "image_time"], default="trial")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-nulls", type=int, default=20)
    p.add_argument("--n-eye-shuffle-nulls", type=int, default=-1)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--compact-k", type=int, default=10)
    p.add_argument("--fem-rank", type=int, default=2)
    p.add_argument("--fit-alpha", choices=["scalar", "fixed_1"], default="scalar")
    p.add_argument("--include-full-forward", action="store_true")
    p.add_argument("--min-units", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose-model-load", action="store_true")
    # Attributes consumed by imported RF/session helpers.
    p.add_argument("--rf-null-session-yaml-dir", type=Path, default=Path("experiments") / "dataset_configs" / "sessions")
    p.add_argument("--rf-null-min-bin-units", type=int, default=6)
    p.add_argument("--rf-null-bin-features", type=str, default="rf_xy,tangent_norm,mean_rate,ccnorm")
    p.set_defaults(enable_rf_readout_null=False)
    return p


def main() -> None:
    run_analysis(build_parser().parse_args())


if __name__ == "__main__":
    main()
