from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import dill
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from eval.sta_ste import load_cached_sta_ste
from eval.eval_stack_multidataset import load_model
from eval.eval_stack_utils import load_single_dataset, rescale_rhat
from models.config_loader import load_config

from .run_cache_closure import (
    DEFAULT_FIG2_CACHE,
    DEFAULT_FIG3_CACHE,
    VISIONCORE_ROOT,
    _apply_projection_to_cov,
    _basis_from_cov_or_matrix,
    _capture,
    _cov_rows,
    _fig2_by_session,
    _load_pickle,
    _null_captures,
    _projection_complement,
    _projection_modes,
    _psd_clip,
    _sym,
    _unit_mask_intersection,
    _write_csv,
    _write_json,
    build_inventory,
    summarize_metrics,
)


DEFAULT_CHECKPOINT = Path("outputs/cache/fig3_digitaltwin_best.ckpt")
DEFAULT_MODEL_CONFIG = Path("outputs/cache/fig3_digitaltwin_model_config.yaml")
DEFAULT_DATASET_CONFIG = Path("outputs/cache/fig3_digitaltwin_multi_basic_120_long.yaml")
DEFAULT_OUT = Path("outputs/matched_twin_covariance_closure_finite_difference")
MIN_FIX_DUR = 20
VALID_TIME_BINS = 120


@dataclass
class SessionSamples:
    source_indices: np.ndarray
    trial_ids: np.ndarray
    time_indices: np.ndarray
    eyepos_deg: np.ndarray
    robs: np.ndarray
    dfs: np.ndarray
    pixels_per_degree: float
    n_candidate_samples: int
    n_good_trials: int
    n_trials_total: int


@dataclass
class RFNullMetadata:
    status: str
    bins: np.ndarray | None
    unit_rows: list[dict[str, Any]]
    n_bins: int
    largest_bin_fraction: float
    bin_features: str


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        from DataYatesV1 import get_free_device

        return str(get_free_device())
    except Exception as exc:
        print(f"Warning: get_free_device() failed ({type(exc).__name__}: {exc}); falling back to torch device check.")
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_twin_model(args: argparse.Namespace):
    model_config = load_config(args.model_config)
    model, model_info = load_model(
        checkpoint_path=Path(args.checkpoint),
        device=_resolve_device(str(args.device)),
        cfg_dir_override=str(args.dataset_config),
        model_config_dict=model_config,
        verbose=bool(args.verbose_model_load),
    )
    model.model.eval()
    if hasattr(model.model, "convnet"):
        model.model.convnet.use_checkpointing = False
    return model, model_info


def _predict(model, stim: torch.Tensor, behavior: torch.Tensor, dataset_idx: int) -> torch.Tensor:
    stim = stim.to(model.device, non_blocking=True)
    behavior = behavior.to(model.device, non_blocking=True)
    with torch.no_grad():
        if hasattr(model.model, "spike_history"):
            out = model.model(stim, dataset_idx, behavior, None)
        else:
            out = model.model(stim, dataset_idx, behavior)
        if getattr(model, "log_input", False):
            out = torch.exp(out)
    return out


def _shift_stimulus_batch(stim: torch.Tensor, displacements_px: np.ndarray) -> torch.Tensor:
    batch, channels, lags, height, width = stim.shape
    merged = stim.reshape(batch, channels * lags, height, width)
    ys = torch.linspace(-1.0, 1.0, height, device=stim.device, dtype=stim.dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=stim.device, dtype=stim.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    displacement_tensor = torch.as_tensor(displacements_px, device=stim.device, dtype=stim.dtype)
    shift_x = 2.0 * displacement_tensor[:, 0] / max(width - 1, 1)
    shift_y = 2.0 * displacement_tensor[:, 1] / max(height - 1, 1)
    base_grid[..., 0] -= shift_x[:, None, None]
    base_grid[..., 1] -= shift_y[:, None, None]
    shifted = F.grid_sample(
        merged,
        base_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return shifted.reshape(batch, channels, lags, height, width)


def _stim_batch(dset: Any, source_indices: np.ndarray, stim_lags: np.ndarray) -> torch.Tensor:
    idx = torch.as_tensor(source_indices[:, None] - stim_lags[None, :], dtype=torch.long)
    stim = dset["stim"][idx]
    if stim.ndim != 5:
        raise ValueError(f"Expected lagged stim shape (B,L,C,H,W), got {tuple(stim.shape)}")
    return stim.permute(0, 2, 1, 3, 4).contiguous()


def _behavior_batch(dset: Any, source_indices: np.ndarray) -> torch.Tensor:
    return dset["behavior"][torch.as_tensor(source_indices, dtype=torch.long)]


def _pixels_per_degree(dset: Any, fallback: float) -> float:
    metadata = getattr(dset, "metadata", {}) or {}
    for key in ("ppd", "pixels_per_degree"):
        val = metadata.get(key)
        if val is not None:
            return float(val)
    return float(fallback)


def _session_yaml_candidates(session: str, args: argparse.Namespace) -> list[Path]:
    candidates: list[Path] = []
    if getattr(args, "rf_null_session_yaml_dir", None):
        candidates.append(Path(args.rf_null_session_yaml_dir) / f"{session}.yaml")
    dataset_config = Path(args.dataset_config)
    try:
        cfg = load_config(dataset_config)
        session_dir = cfg.get("session_dir")
        if session_dir:
            sd = Path(session_dir)
            if not sd.is_absolute():
                candidates.append(dataset_config.parent / sd / f"{session}.yaml")
                candidates.append(VISIONCORE_ROOT / "experiments" / "dataset_configs" / sd / f"{session}.yaml")
    except Exception:
        pass
    candidates.extend(
        [
            VISIONCORE_ROOT / "experiments" / "dataset_configs" / "sessions" / f"{session}.yaml",
            VISIONCORE_ROOT / "experiments" / "dataset_configs" / "sessions_legacy_e482ece" / f"{session}.yaml",
            VISIONCORE_ROOT / "experiments" / "dataset_configs" / "sessions_all_cells" / f"{session}.yaml",
        ]
    )
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _load_session_cids(session: str, args: argparse.Namespace) -> tuple[np.ndarray | None, str, str]:
    for path in _session_yaml_candidates(session, args):
        if not path.exists():
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            return None, f"session_yaml_load_failed_{type(exc).__name__}", str(path)
        cids = payload.get("cids")
        if cids is None:
            return None, "session_yaml_missing_cids", str(path)
        arr = np.asarray(cids, dtype=np.int64).ravel()
        if arr.size == 0:
            return None, "session_yaml_empty_cids", str(path)
        return arr, "ok_session_yaml_cids", str(path)
    return None, "missing_session_yaml", ""


def _rf_centers_from_sta_cache(session: str) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    arrs = load_cached_sta_ste(session)
    if arrs is None:
        return None, None, "missing_sta_cache"
    stes = np.asarray(arrs.get("stes"), dtype=np.float64)
    if stes.ndim != 4:
        return None, None, "invalid_sta_cache_shape"
    n_units, _n_lags, h, w = stes.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rf_x = np.full(n_units, np.nan, dtype=np.float64)
    rf_y = np.full(n_units, np.nan, dtype=np.float64)
    for u in range(n_units):
        ste_u = stes[u]
        if not np.isfinite(ste_u).any():
            continue
        lag = int(np.nanargmax(np.nanstd(ste_u, axis=(1, 2))))
        im = ste_u[lag]
        weights = np.abs(im - np.nanmedian(im))
        weights = np.where(np.isfinite(weights), weights, 0.0)
        mass = float(np.sum(weights))
        if mass <= 1e-12:
            continue
        rf_x[u] = float(np.sum(weights * xx) / mass)
        rf_y[u] = float(np.sum(weights * yy) / mass)
    return rf_x, rf_y, f"ok_sta_cache_pixels_h{h}_w{w}"


def _load_matched_recorded_rfs(
    *,
    session: str,
    common_units: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str, dict[str, Any]]:
    selected_cids, cid_status, cid_path = _load_session_cids(session, args)
    meta: dict[str, Any] = {
        "rf_cid_status": cid_status,
        "rf_cid_source": cid_path,
        "rf_coordinate_source": "",
    }
    if selected_cids is None:
        return None, None, None, cid_status, meta
    common = np.asarray(common_units, dtype=np.int64).ravel()
    if common.size == 0 or int(np.max(common)) >= selected_cids.size or int(np.min(common)) < 0:
        meta["rf_selected_cids_count"] = int(selected_cids.size)
        return None, None, None, "common_units_outside_selected_cids", meta
    matched_cids = selected_cids[common]

    rf_x_all, rf_y_all, rf_status = _rf_centers_from_sta_cache(session)
    meta["rf_coordinate_source"] = rf_status
    if rf_x_all is None or rf_y_all is None:
        return None, None, matched_cids, rf_status, meta
    if matched_cids.size == 0 or int(np.nanmax(matched_cids)) >= rf_x_all.size or int(np.nanmin(matched_cids)) < 0:
        meta["rf_sta_units_count"] = int(rf_x_all.size)
        meta["rf_matched_cid_min"] = int(np.nanmin(matched_cids)) if matched_cids.size else None
        meta["rf_matched_cid_max"] = int(np.nanmax(matched_cids)) if matched_cids.size else None
        return None, None, matched_cids, "matched_cids_outside_sta_cache", meta
    return rf_x_all[matched_cids], rf_y_all[matched_cids], matched_cids, rf_status, meta


def _collect_samples(
    *,
    model: Any,
    dataset_idx: int,
    common_units: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Any, np.ndarray, SessionSamples]:
    train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
    fixrsvp_inds = torch.cat(
        [train_data.get_dataset_inds("fixrsvp"), val_data.get_dataset_inds("fixrsvp")],
        dim=0,
    )
    dset_idx_local = int(fixrsvp_inds[:, 0].unique().item())
    dset = train_data.dsets[dset_idx_local]
    stim_lags = np.asarray(dataset_config["keys_lags"]["stim"], dtype=np.int64)
    max_lag = int(np.max(stim_lags))

    trial_inds = np.asarray(dset.covariates["trial_inds"]).ravel()
    psth_inds = np.asarray(dset.covariates["psth_inds"]).ravel().astype(np.int64)
    eyepos_all = np.asarray(dset["eyepos"], dtype=np.float64)
    robs_all = np.asarray(dset["robs"], dtype=np.float64)
    dfs_all = np.asarray(dset["dfs"], dtype=np.float64)
    fixation = np.hypot(eyepos_all[:, 0], eyepos_all[:, 1]) < float(args.fixation_radius_deg)

    rows: list[tuple[int, int, int]] = []
    good_trials = 0
    for trial in np.unique(trial_inds):
        ix = np.where((trial_inds == trial) & fixation)[0]
        if ix.size <= MIN_FIX_DUR:
            continue
        good_trials += 1
        # Match Ryan's array-fill semantics: if duplicate time bins exist, the
        # later source row wins.
        by_time: dict[int, int] = {}
        for src, t in zip(ix.tolist(), psth_inds[ix].tolist(), strict=True):
            by_time[int(t)] = int(src)
        for t, src in sorted(by_time.items()):
            if t >= VALID_TIME_BINS or src < max_lag:
                continue
            if not np.isfinite(eyepos_all[src]).all():
                continue
            dfs = dfs_all[src, common_units]
            robs = robs_all[src, common_units]
            if args.sample_dfs_mode == "all" and not np.all(dfs > 0.5):
                continue
            if args.sample_dfs_mode == "any" and not np.any(dfs > 0.5):
                continue
            if not np.isfinite(dfs).all() or not np.isfinite(robs).all():
                continue
            rows.append((int(src), int(trial), int(t)))

    n_candidate = len(rows)
    if n_candidate == 0:
        raise RuntimeError("No valid finite-difference samples after filtering")

    rng = np.random.default_rng(int(args.seed) + int(dataset_idx) * 1009)
    if int(args.max_samples) > 0 and n_candidate > int(args.max_samples):
        keep = np.sort(rng.choice(n_candidate, size=int(args.max_samples), replace=False))
        rows = [rows[int(i)] for i in keep]

    source_indices = np.asarray([r[0] for r in rows], dtype=np.int64)
    trial_ids = np.asarray([r[1] for r in rows], dtype=np.int64)
    time_indices = np.asarray([r[2] for r in rows], dtype=np.int64)
    samples = SessionSamples(
        source_indices=source_indices,
        trial_ids=trial_ids,
        time_indices=time_indices,
        eyepos_deg=eyepos_all[source_indices],
        robs=robs_all[np.ix_(source_indices, common_units)],
        dfs=dfs_all[np.ix_(source_indices, common_units)],
        pixels_per_degree=_pixels_per_degree(dset, float(args.pixels_per_degree_fallback)),
        n_candidate_samples=int(n_candidate),
        n_good_trials=int(good_trials),
        n_trials_total=int(np.unique(trial_inds).size),
    )
    return dset, stim_lags, samples


def _fit_rescale_gains(
    *,
    model: Any,
    dset: Any,
    stim_lags: np.ndarray,
    samples: SessionSamples,
    common_units: np.ndarray,
    dataset_idx: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    if str(args.rescale_mode) == "none":
        return np.ones(common_units.size, dtype=np.float64), "none"

    preds = []
    for start in range(0, samples.source_indices.size, int(args.batch_size)):
        idx = samples.source_indices[start : start + int(args.batch_size)]
        stim = _stim_batch(dset, idx, stim_lags)
        behavior = _behavior_batch(dset, idx)
        pred = _predict(model, stim, behavior, dataset_idx).detach().cpu().numpy()
        preds.append(pred[:, common_units])
    rhat = torch.as_tensor(np.concatenate(preds, axis=0), dtype=torch.float32)
    robs = torch.as_tensor(samples.robs, dtype=torch.float32)
    dfs = torch.as_tensor(samples.dfs, dtype=torch.float32)
    try:
        _, scale_model = rescale_rhat(robs, rhat, dfs, mode=str(args.rescale_mode))
        if hasattr(scale_model, "g"):
            gains = torch.exp(scale_model.g).detach().cpu().numpy()
            if gains.ndim == 0:
                gains = np.full(common_units.size, float(gains), dtype=np.float64)
            return np.asarray(gains, dtype=np.float64), str(args.rescale_mode)
    except Exception as exc:
        return np.ones(common_units.size, dtype=np.float64), f"failed:{type(exc).__name__}"
    return np.ones(common_units.size, dtype=np.float64), "unavailable_gain"


def _compute_jacobians(
    *,
    model: Any,
    dset: Any,
    stim_lags: np.ndarray,
    samples: SessionSamples,
    common_units: np.ndarray,
    gains: np.ndarray,
    dataset_idx: int,
    args: argparse.Namespace,
) -> np.ndarray:
    all_j = []
    step = float(args.step_px)
    for start in range(0, samples.source_indices.size, int(args.batch_size)):
        idx = samples.source_indices[start : start + int(args.batch_size)]
        stim = _stim_batch(dset, idx, stim_lags).to(model.device)
        behavior = _behavior_batch(dset, idx)
        n = idx.size
        xp = np.zeros((n, 2), dtype=np.float64)
        xm = np.zeros((n, 2), dtype=np.float64)
        yp = np.zeros((n, 2), dtype=np.float64)
        ym = np.zeros((n, 2), dtype=np.float64)
        xp[:, 0] = step
        xm[:, 0] = -step
        yp[:, 1] = step
        ym[:, 1] = -step
        rxp = _predict(model, _shift_stimulus_batch(stim, xp), behavior, dataset_idx).detach().cpu().numpy()
        rxm = _predict(model, _shift_stimulus_batch(stim, xm), behavior, dataset_idx).detach().cpu().numpy()
        ryp = _predict(model, _shift_stimulus_batch(stim, yp), behavior, dataset_idx).detach().cpu().numpy()
        rym = _predict(model, _shift_stimulus_batch(stim, ym), behavior, dataset_idx).detach().cpu().numpy()
        jx = (rxp[:, common_units] - rxm[:, common_units]) / (2.0 * step)
        jy = (ryp[:, common_units] - rym[:, common_units]) / (2.0 * step)
        j = np.stack([jx, jy], axis=-1).astype(np.float64)
        j *= gains[None, :, None]
        all_j.append(j)
    return np.concatenate(all_j, axis=0)


def _compute_static_responses(
    *,
    model: Any,
    dset: Any,
    stim_lags: np.ndarray,
    samples: SessionSamples,
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


def _source_payloads(j: np.ndarray, eye_px: np.ndarray) -> dict[str, dict[str, Any]]:
    eye_c = eye_px - np.mean(eye_px, axis=0, keepdims=True)
    sigma_eye = np.cov(eye_c.T).astype(np.float64)
    j_mean = np.mean(j, axis=0)
    eye_trace = np.einsum("nua,na->nu", j, eye_c)
    tangent_gram = np.einsum("nua,ab,nvb->uv", j, sigma_eye, j) / max(j.shape[0], 1)
    mean_cov = _sym(j_mean @ sigma_eye @ j_mean.T)
    return {
        "fd_mean_tangent_matrix": {"cov": mean_cov, "mat": j_mean, "status": "ok"},
        "fd_mean_tangent_cov": {"cov": mean_cov, "mat": None, "status": "ok"},
        "fd_sample_eye_trace_cov": {"cov": _cov_rows(eye_trace), "mat": None, "status": "ok"},
        "fd_tangent_gram_cov": {"cov": _sym(tangent_gram), "mat": None, "status": "ok"},
    }


def _numerical_rank(vals: np.ndarray, eps: float = 1e-10) -> int:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0
    vmax = float(np.max(np.abs(vals)))
    if vmax <= 0.0:
        return 0
    return int(np.sum(vals > vmax * eps))


def _compact_source_name(compact_k: int) -> str:
    return f"fd_sample_eye_trace_xfit_compact_k{int(compact_k)}_cov"


def _static_pc_source_name(static_pc_k: int) -> str:
    return f"fd_sample_eye_trace_xfit_static_pc_k{int(static_pc_k)}_cov"


def _tangent_matrix(j: np.ndarray, mask: np.ndarray) -> np.ndarray:
    jj = np.asarray(j, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if jj.shape[0] == 0:
        return np.zeros((j.shape[1], 0), dtype=np.float64)
    return np.concatenate([jj[:, :, 0].T, jj[:, :, 1].T], axis=1)


def _compact_crossfit_payload(
    *,
    j: np.ndarray,
    eye_px: np.ndarray,
    group_ids: np.ndarray,
    compact_k: int,
    n_folds: int,
    seed: int,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    source_name = _compact_source_name(int(compact_k))
    group_ids = np.asarray(group_ids)
    unique_groups = np.unique(group_ids)
    stats: dict[str, Any] = {
        "compact_source": source_name,
        "compact_group_mode": "trial_inds",
        "compact_basis_k": int(compact_k),
        "compact_n_groups": int(unique_groups.size),
        "compact_requested_folds": int(n_folds),
    }
    if unique_groups.size < 2:
        stats.update({"compact_status": "too_few_groups"})
        return source_name, {"cov": None, "mat": None, "status": "too_few_groups"}, stats

    rng = np.random.default_rng(int(seed))
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)
    folds = [x for x in np.array_split(shuffled_groups, min(int(n_folds), unique_groups.size)) if x.size]

    eye_c = np.asarray(eye_px, dtype=np.float64) - np.mean(eye_px, axis=0, keepdims=True)
    eye_trace = np.einsum("nua,na->nu", np.asarray(j, dtype=np.float64), eye_c)
    projected = np.full_like(eye_trace, np.nan, dtype=np.float64)

    rank_train: list[int] = []
    rank_used: list[int] = []
    test_counts: list[int] = []
    train_counts: list[int] = []
    for fold_idx, test_groups in enumerate(folds):
        test_mask = np.isin(group_ids, test_groups)
        train_mask = ~test_mask
        test_counts.append(int(np.sum(test_mask)))
        train_counts.append(int(np.sum(train_mask)))
        m_train = _tangent_matrix(j, train_mask)
        if m_train.shape[1] == 0 or not np.isfinite(m_train).all():
            continue
        vals, vecs = _basis_from_cov_or_matrix("compact_tangent_matrix", None, m_train)
        rank = _numerical_rank(np.maximum(vals, 0.0))
        use_k = min(int(compact_k), int(rank), int(vecs.shape[1]))
        rank_train.append(int(rank))
        rank_used.append(int(use_k))
        if use_k <= 0:
            continue
        basis = vecs[:, :use_k]
        projected[test_mask] = eye_trace[test_mask] @ basis @ basis.T

    row_ok = np.all(np.isfinite(projected), axis=1)
    cov = _cov_rows(projected[row_ok])
    min_rank = int(min(rank_train)) if rank_train else 0
    status = "ok" if row_ok.all() and min_rank >= int(compact_k) and np.isfinite(cov).all() else "invalid_compact_crossfit"
    stats.update(
        {
            "compact_status": status,
            "compact_n_folds": int(len(folds)),
            "compact_min_train_rank": min_rank,
            "compact_min_rank_used": int(min(rank_used)) if rank_used else 0,
            "compact_projected_samples": int(np.sum(row_ok)),
            "compact_test_samples_min": int(min(test_counts)) if test_counts else 0,
            "compact_test_samples_max": int(max(test_counts)) if test_counts else 0,
            "compact_train_samples_min": int(min(train_counts)) if train_counts else 0,
            "compact_train_samples_max": int(max(train_counts)) if train_counts else 0,
        }
    )
    return source_name, {"cov": cov, "mat": None, "status": status}, stats


def _static_pc_crossfit_payload(
    *,
    j: np.ndarray,
    eye_px: np.ndarray,
    static_responses: np.ndarray,
    group_ids: np.ndarray,
    static_pc_k: int,
    n_folds: int,
    seed: int,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    source_name = _static_pc_source_name(int(static_pc_k))
    group_ids = np.asarray(group_ids)
    unique_groups = np.unique(group_ids)
    stats: dict[str, Any] = {
        "static_pc_source": source_name,
        "static_pc_group_mode": "trial_inds",
        "static_pc_basis_k": int(static_pc_k),
        "static_pc_n_groups": int(unique_groups.size),
        "static_pc_requested_folds": int(n_folds),
    }
    if unique_groups.size < 2:
        stats.update({"static_pc_status": "too_few_groups"})
        return source_name, {"cov": None, "mat": None, "status": "too_few_groups"}, stats

    rng = np.random.default_rng(int(seed))
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)
    folds = [x for x in np.array_split(shuffled_groups, min(int(n_folds), unique_groups.size)) if x.size]

    eye_c = np.asarray(eye_px, dtype=np.float64) - np.mean(eye_px, axis=0, keepdims=True)
    eye_trace = np.einsum("nua,na->nu", np.asarray(j, dtype=np.float64), eye_c)
    r0 = np.asarray(static_responses, dtype=np.float64)
    projected = np.full_like(eye_trace, np.nan, dtype=np.float64)

    rank_train: list[int] = []
    rank_used: list[int] = []
    test_counts: list[int] = []
    train_counts: list[int] = []
    for test_groups in folds:
        test_mask = np.isin(group_ids, test_groups)
        train_mask = ~test_mask
        test_counts.append(int(np.sum(test_mask)))
        train_counts.append(int(np.sum(train_mask)))
        train = r0[train_mask]
        if train.shape[0] < 2 or not np.isfinite(train).all():
            continue
        train = train - np.mean(train, axis=0, keepdims=True)
        _u, s, vt = np.linalg.svd(train, full_matrices=False)
        rank = _numerical_rank(s)
        use_k = min(int(static_pc_k), int(rank), int(vt.shape[0]))
        rank_train.append(int(rank))
        rank_used.append(int(use_k))
        if use_k <= 0:
            continue
        basis = vt[:use_k].T
        projected[test_mask] = eye_trace[test_mask] @ basis @ basis.T

    row_ok = np.all(np.isfinite(projected), axis=1)
    cov = _cov_rows(projected[row_ok])
    min_rank = int(min(rank_train)) if rank_train else 0
    status = "ok" if row_ok.all() and min_rank >= int(static_pc_k) and np.isfinite(cov).all() else "invalid_static_pc_crossfit"
    stats.update(
        {
            "static_pc_status": status,
            "static_pc_n_folds": int(len(folds)),
            "static_pc_min_train_rank": min_rank,
            "static_pc_min_rank_used": int(min(rank_used)) if rank_used else 0,
            "static_pc_projected_samples": int(np.sum(row_ok)),
            "static_pc_test_samples_min": int(min(test_counts)) if test_counts else 0,
            "static_pc_test_samples_max": int(max(test_counts)) if test_counts else 0,
            "static_pc_train_samples_min": int(min(train_counts)) if train_counts else 0,
            "static_pc_train_samples_max": int(max(train_counts)) if train_counts else 0,
        }
    )
    return source_name, {"cov": cov, "mat": None, "status": status}, stats


def _quantile_split_indices(values: np.ndarray, indices: np.ndarray, q: int, min_bin_units: int) -> list[np.ndarray] | None:
    idx = np.asarray(indices, dtype=np.int64)
    if int(q) <= 1 or idx.size < int(q) * int(min_bin_units):
        return None
    vals = np.asarray(values, dtype=np.float64)[idx]
    finite = np.isfinite(vals)
    if int(np.sum(finite)) != idx.size:
        return None
    order = idx[np.argsort(vals, kind="mergesort")]
    parts = [np.asarray(part, dtype=np.int64) for part in np.array_split(order, int(q)) if part.size]
    if len(parts) != int(q) or any(part.size < int(min_bin_units) for part in parts):
        return None
    return parts


def _split_bins_by_feature(
    bins: list[np.ndarray],
    values: np.ndarray,
    *,
    q: int,
    min_bin_units: int,
) -> tuple[list[np.ndarray], bool]:
    out: list[np.ndarray] = []
    changed = False
    for idx in bins:
        parts = _quantile_split_indices(values, idx, int(q), int(min_bin_units))
        if parts is None:
            out.append(idx)
        else:
            out.extend(parts)
            changed = True
    return out, changed


def _label_bins(bins: list[np.ndarray], n_units: int) -> np.ndarray:
    labels = np.full(int(n_units), -1, dtype=np.int64)
    for i, idx in enumerate(bins):
        labels[np.asarray(idx, dtype=np.int64)] = int(i)
    return labels


def _make_adaptive_bins(
    *,
    rf_x: np.ndarray | None,
    rf_y: np.ndarray | None,
    tangent_norm: np.ndarray,
    mean_rate: np.ndarray,
    ccnorm: np.ndarray | None,
    min_bin_units: int,
    requested_features: str,
) -> tuple[np.ndarray | None, str, str, float]:
    n_units = int(np.asarray(tangent_norm).size)
    if n_units < max(2, int(min_bin_units)):
        return None, "too_few_units_for_bins", "", float("nan")

    features = {x.strip() for x in str(requested_features).split(",") if x.strip()}
    bins: list[np.ndarray] = [np.arange(n_units, dtype=np.int64)]
    used: list[str] = []

    has_rf = (
        rf_x is not None
        and rf_y is not None
        and np.asarray(rf_x).size == n_units
        and np.asarray(rf_y).size == n_units
        and np.all(np.isfinite(rf_x))
        and np.all(np.isfinite(rf_y))
    )
    if "rf_xy" in features and has_rf:
        if n_units >= 4 * int(min_bin_units):
            q_spatial = 3 if n_units >= 9 * int(min_bin_units) else 2
            bins, changed_x = _split_bins_by_feature(bins, np.asarray(rf_x, dtype=np.float64), q=q_spatial, min_bin_units=int(min_bin_units))
            bins, changed_y = _split_bins_by_feature(bins, np.asarray(rf_y, dtype=np.float64), q=q_spatial, min_bin_units=int(min_bin_units))
            if changed_x or changed_y:
                used.append(f"rf_xy_q{q_spatial}")
        elif n_units >= 2 * int(min_bin_units):
            x_span = float(np.nanpercentile(rf_x, 90) - np.nanpercentile(rf_x, 10))
            y_span = float(np.nanpercentile(rf_y, 90) - np.nanpercentile(rf_y, 10))
            axis_values = np.asarray(rf_x if x_span >= y_span else rf_y, dtype=np.float64)
            bins, changed = _split_bins_by_feature(bins, axis_values, q=2, min_bin_units=int(min_bin_units))
            if changed:
                used.append("rf_xy_1d_q2")

    if "tangent_norm" in features and np.all(np.isfinite(tangent_norm)):
        bins, changed = _split_bins_by_feature(bins, np.asarray(tangent_norm, dtype=np.float64), q=2, min_bin_units=int(min_bin_units))
        if changed:
            used.append("tangent_norm_q2")

    if "mean_rate" in features and np.all(np.isfinite(mean_rate)):
        bins, changed = _split_bins_by_feature(bins, np.asarray(mean_rate, dtype=np.float64), q=2, min_bin_units=int(min_bin_units))
        if changed:
            used.append("mean_rate_q2")

    if "ccnorm" in features and ccnorm is not None and np.asarray(ccnorm).size == n_units and np.all(np.isfinite(ccnorm)):
        bins, changed = _split_bins_by_feature(bins, np.asarray(ccnorm, dtype=np.float64), q=2, min_bin_units=int(min_bin_units))
        if changed:
            used.append("ccnorm_q2")

    labels = _label_bins(bins, n_units)
    if np.any(labels < 0) or any(np.sum(labels == b) < int(min_bin_units) for b in np.unique(labels)):
        return None, "invalid_bins_after_adaptive_split", ",".join(used), float("nan")
    largest = float(max(np.sum(labels == b) for b in np.unique(labels)) / max(n_units, 1))
    status = "ok_rf_bins" if has_rf and any(u.startswith("rf_xy") for u in used) else "ok_nonspatial_bins"
    return labels, status, ",".join(used), largest


def _constrained_permutation(bins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(bins, dtype=np.int64).ravel()
    perm = np.arange(labels.size, dtype=np.int64)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == int(label))
        if idx.size > 1:
            shuffled = idx.copy()
            rng.shuffle(shuffled)
            perm[idx] = shuffled
    return perm


def _rf_null_captures(
    *,
    rng: np.random.Generator,
    target: np.ndarray,
    source_name: str,
    source_cov: np.ndarray | None,
    source_matrix: np.ndarray | None,
    projection: np.ndarray,
    bins: np.ndarray | None,
    k: int,
    n_nulls: int,
) -> dict[str, Any]:
    if bins is None:
        return {
            "rf_fixed_permutation_null_mean": float("nan"),
            "rf_fixed_permutation_null_median": float("nan"),
            "rf_fixed_permutation_null_ci_low": float("nan"),
            "rf_fixed_permutation_null_ci_high": float("nan"),
        }
    vals: list[float] = []
    for _ in range(int(n_nulls)):
        perm = _constrained_permutation(np.asarray(bins, dtype=np.int64), rng)
        if source_name.endswith("_cov") and source_cov is not None:
            cov_null = np.asarray(source_cov, dtype=np.float64)[np.ix_(perm, perm)]
            cov_null = _apply_projection_to_cov(cov_null, projection)
            eigvals, eigvecs = _basis_from_cov_or_matrix(source_name, cov_null, None)
        elif source_matrix is not None:
            mat_null = projection @ np.asarray(source_matrix, dtype=np.float64)[perm, :]
            eigvals, eigvecs = _basis_from_cov_or_matrix(source_name, None, mat_null)
        else:
            continue
        rank = _numerical_rank(np.maximum(eigvals, 0.0))
        if int(k) <= max(rank, 0) and eigvecs.shape[1] >= int(k):
            vals.append(_capture(target, eigvecs[:, : int(k)]))
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "rf_fixed_permutation_null_mean": float("nan"),
            "rf_fixed_permutation_null_median": float("nan"),
            "rf_fixed_permutation_null_ci_low": float("nan"),
            "rf_fixed_permutation_null_ci_high": float("nan"),
        }
    return {
        "rf_fixed_permutation_null_mean": float(np.mean(arr)),
        "rf_fixed_permutation_null_median": float(np.median(arr)),
        "rf_fixed_permutation_null_ci_low": float(np.percentile(arr, 2.5)),
        "rf_fixed_permutation_null_ci_high": float(np.percentile(arr, 97.5)),
    }


def _rf_null_metadata_for_session(
    *,
    session: str,
    subject: str,
    common_units: np.ndarray,
    sr: dict[str, Any],
    samples: SessionSamples,
    j: np.ndarray,
    gains: np.ndarray,
    args: argparse.Namespace,
) -> RFNullMetadata:
    if not bool(args.enable_rf_readout_null):
        return RFNullMetadata("disabled", None, [], 0, float("nan"), "")

    rf_x, rf_y, matched_cids, rf_status, rf_meta = _load_matched_recorded_rfs(
        session=session,
        common_units=common_units,
        args=args,
    )
    tangent_norm = np.sqrt(np.nanmean(np.sum(np.asarray(j, dtype=np.float64) ** 2, axis=2), axis=0))
    mean_rate = np.nanmean(np.asarray(samples.robs, dtype=np.float64), axis=0)
    sr_pos = {int(u): i for i, u in enumerate(np.asarray(sr.get("neuron_mask", []), dtype=np.int64).tolist())}
    idx3 = np.asarray([sr_pos.get(int(u), -1) for u in np.asarray(common_units, dtype=np.int64)], dtype=np.int64)
    ccnorm = None
    if "ccnorm" in sr and np.all(idx3 >= 0):
        raw_ccnorm = np.asarray(sr["ccnorm"], dtype=np.float64).ravel()
        if raw_ccnorm.size > int(np.max(idx3)):
            ccnorm = raw_ccnorm[idx3]

    bins, bin_status, used_features, largest = _make_adaptive_bins(
        rf_x=rf_x,
        rf_y=rf_y,
        tangent_norm=tangent_norm,
        mean_rate=mean_rate,
        ccnorm=ccnorm,
        min_bin_units=int(args.rf_null_min_bin_units),
        requested_features=str(args.rf_null_bin_features),
    )
    status = bin_status if bins is not None else f"{rf_status};{bin_status}"
    rows: list[dict[str, Any]] = []
    n_units = int(common_units.size)
    labels = bins if bins is not None else np.full(n_units, -1, dtype=np.int64)
    for u in range(n_units):
        rows.append(
            {
                "session": session,
                "subject": subject,
                "unit_position": int(u),
                "matched_unit_index": int(common_units[u]),
                "cluster_id": int(matched_cids[u]) if matched_cids is not None and u < matched_cids.size else -1,
                "rf_null_bin": int(labels[u]),
                "rf_x_pixel": float(rf_x[u]) if rf_x is not None and u < len(rf_x) else float("nan"),
                "rf_y_pixel": float(rf_y[u]) if rf_y is not None and u < len(rf_y) else float("nan"),
                "tangent_norm": float(tangent_norm[u]),
                "mean_rate": float(mean_rate[u]),
                "gain": float(gains[u]),
                "ccnorm": float(ccnorm[u]) if ccnorm is not None and np.isfinite(ccnorm[u]) else float("nan"),
                "rf_status": str(rf_status),
                "bin_status": str(status),
                "bin_features": str(used_features),
                "rf_cid_status": str(rf_meta.get("rf_cid_status", "")),
                "rf_cid_source": str(rf_meta.get("rf_cid_source", "")),
                "rf_coordinate_source": str(rf_meta.get("rf_coordinate_source", "")),
            }
        )
    return RFNullMetadata(
        status=status,
        bins=bins,
        unit_rows=rows,
        n_bins=int(np.unique(bins).size) if bins is not None else 0,
        largest_bin_fraction=largest,
        bin_features=used_features,
    )


def _target_for_session(f2: dict[str, Any], sr: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    common, idx2, _ = _unit_mask_intersection(f2["neuron_mask"], sr["neuron_mask"])
    target_full = _sym(np.asarray(f2["mats"][int(args.window_idx)]["FEM"], dtype=np.float64)[np.ix_(idx2, idx2)])
    unit_keep = np.isfinite(np.diag(target_full))
    common = common[unit_keep]
    idx2 = idx2[unit_keep]
    target = _sym(np.asarray(f2["mats"][int(args.window_idx)]["FEM"], dtype=np.float64)[np.ix_(idx2, idx2)])
    if not np.isfinite(target).all():
        finite_rows = np.isfinite(target).all(axis=0) & np.isfinite(target).all(axis=1)
        common = common[finite_rows]
        idx2 = idx2[finite_rows]
        target = _sym(np.asarray(f2["mats"][int(args.window_idx)]["FEM"], dtype=np.float64)[np.ix_(idx2, idx2)])
    target_psd, eigs = _psd_clip(target)
    meta = {
        "n_common_units": int(common.size),
        "target_trace_raw": float(np.trace(target)),
        "target_trace_psd": float(np.trace(target_psd)),
        "target_min_eigenvalue_raw": float(np.min(eigs)) if eigs.size else float("nan"),
        "target_negative_eigenvalue_mass_raw": float(np.sum(np.abs(eigs[eigs < 0.0]))),
    }
    return common.astype(np.int64), target, target_psd, meta


def run_analysis(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig3_rows = _load_pickle(Path(args.fig3_cache))
    fig2_rows = _load_pickle(Path(args.fig2_cache))
    inventory = build_inventory(fig3_rows, fig2_rows, int(args.window_idx))
    _write_csv(out_dir / "session_inventory.csv", inventory)
    fig2 = _fig2_by_session(fig2_rows)
    fig3_by_session = {str(row["session"]): row for row in fig3_rows}

    requested_sessions = [x.strip() for x in str(args.sessions).split(",") if x.strip()]
    if len(requested_sessions) == 1 and requested_sessions[0].lower() == "all":
        requested_sessions = []
    if not requested_sessions:
        requested_sessions = [str(row["session"]) for row in fig3_rows if str(row.get("subject", "")) in {"Allen", "Logan"}]

    model, model_info = _load_twin_model(args)
    k_list = [int(x) for x in str(args.k_list).split(",") if x.strip()]
    basis_sources = [x.strip() for x in str(args.basis_sources).split(",") if x.strip()]
    projection_controls = [x.strip() for x in str(args.projection_controls).split(",") if x.strip()]
    target_variants = [x.strip() for x in str(args.target_variants).split(",") if x.strip()]
    static_pc_source = _static_pc_source_name(int(args.static_pc_basis_k))
    rng = np.random.default_rng(int(args.seed))
    n_rf_nulls = int(args.rf_null_n_nulls) if int(args.rf_null_n_nulls) > 0 else int(args.n_nulls)

    metric_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    rf_unit_bin_rows: list[dict[str, Any]] = []

    for session in requested_sessions:
        if session not in fig3_by_session or session not in fig2:
            continue
        if session not in getattr(model, "names", []):
            session_rows.append({"session": session, "status": "missing_model_session"})
            continue
        dataset_idx = int(model.names.index(session))
        sr = fig3_by_session[session]
        common_units, target_raw, target_psd, target_meta = _target_for_session(fig2[session], sr, args)
        if common_units.size < 3:
            session_rows.append({"session": session, "status": "too_few_common_units"})
            continue

        dset, stim_lags, samples = _collect_samples(
            model=model,
            dataset_idx=dataset_idx,
            common_units=common_units,
            args=args,
        )
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
        payloads = _source_payloads(j, eye_px)
        static_pc_stats: dict[str, Any] = {}
        if static_pc_source in basis_sources:
            static_r0 = _compute_static_responses(
                model=model,
                dset=dset,
                stim_lags=stim_lags,
                samples=samples,
                common_units=common_units,
                gains=gains,
                dataset_idx=dataset_idx,
                args=args,
            )
            static_name, static_payload, static_pc_stats = _static_pc_crossfit_payload(
                j=j,
                eye_px=eye_px,
                static_responses=static_r0,
                group_ids=samples.trial_ids,
                static_pc_k=int(args.static_pc_basis_k),
                n_folds=int(args.static_pc_n_folds),
                seed=int(args.seed) + int(dataset_idx) * 1543,
            )
            payloads[static_name] = static_payload
        rf_null_meta = _rf_null_metadata_for_session(
            session=session,
            subject=str(sr.get("subject", "")),
            common_units=common_units,
            sr=sr,
            samples=samples,
            j=j,
            gains=gains,
            args=args,
        )
        rf_unit_bin_rows.extend(rf_null_meta.unit_rows)
        compact_name, compact_payload, compact_stats = _compact_crossfit_payload(
            j=j,
            eye_px=eye_px,
            group_ids=samples.trial_ids,
            compact_k=int(args.compact_basis_k),
            n_folds=int(args.compact_n_folds),
            seed=int(args.seed) + int(dataset_idx) * 7919,
        )
        payloads[compact_name] = compact_payload

        session_rows.append(
            {
                "session": session,
                "subject": sr.get("subject", ""),
                "status": "ok",
                "dataset_idx": dataset_idx,
                **target_meta,
                "n_candidate_samples": samples.n_candidate_samples,
                "n_samples_used": int(samples.source_indices.size),
                "n_good_trials": samples.n_good_trials,
                "n_trials_total": samples.n_trials_total,
                "pixels_per_degree": samples.pixels_per_degree,
                "step_px": float(args.step_px),
                "device": str(model.device),
                "rescale_mode": str(args.rescale_mode),
                "rescale_status": rescale_status,
                "gain_median": float(np.median(gains)),
                "gain_min": float(np.min(gains)),
                "gain_max": float(np.max(gains)),
                "jacobian_abs_median": float(np.median(np.abs(j))),
                "eye_px_radius_p50": float(np.percentile(np.linalg.norm(eye_px - eye_px.mean(axis=0), axis=1), 50)),
                "eye_px_radius_p90": float(np.percentile(np.linalg.norm(eye_px - eye_px.mean(axis=0), axis=1), 90)),
                "rf_null_status": rf_null_meta.status,
                "rf_null_n_bins": rf_null_meta.n_bins,
                "rf_null_largest_bin_fraction": rf_null_meta.largest_bin_fraction,
                "rf_null_bin_features": rf_null_meta.bin_features,
                **compact_stats,
                **static_pc_stats,
            }
        )

        targets = {"raw": target_raw, "psd": target_psd}
        for target_variant in target_variants:
            target_base = targets.get(target_variant)
            if target_base is None:
                continue
            for projection_control in projection_controls:
                modes = _projection_modes(projection_control, target_base)
                p = _projection_complement(common_units.size, modes)
                target = _apply_projection_to_cov(target_base, p)
                for source in basis_sources:
                    payload = payloads.get(source)
                    if payload is None:
                        continue
                    cov0 = payload["cov"]
                    mat0 = payload["mat"]
                    cov = cov0
                    mat = mat0
                    if cov is not None:
                        cov = _apply_projection_to_cov(cov, p)
                    if mat is not None:
                        mat = p @ np.asarray(mat, dtype=np.float64)
                    if cov is not None and not np.isfinite(cov).all():
                        status = "invalid_source_covariance"
                        vals = np.array([])
                        vecs = np.zeros((common_units.size, 0))
                    elif mat is not None and not np.isfinite(mat).all():
                        status = "invalid_source_matrix"
                        vals = np.array([])
                        vecs = np.zeros((common_units.size, 0))
                    else:
                        vals, vecs = _basis_from_cov_or_matrix(source, cov, mat)
                        status = str(payload["status"])
                    rank = int(np.sum(np.maximum(vals, 0.0) > max(np.max(vals) if vals.size else 0.0, 1.0) * 1e-10))
                    for k in k_list:
                        row = {
                            "session": session,
                            "subject": sr.get("subject", ""),
                            "window_idx": int(args.window_idx),
                            "target_variant": target_variant,
                            "projection_control": projection_control,
                            "basis_source": source,
                            "basis_status": status,
                            "n_common_units": int(common_units.size),
                            "n_samples_used": int(samples.source_indices.size),
                            "basis_rank": rank,
                            "k": int(k),
                            "target_trace": float(np.trace(target)),
                            "target_trace_raw": float(np.trace(target_raw)),
                            "target_trace_psd": float(np.trace(target_psd)),
                            "rf_null_status": rf_null_meta.status,
                            "rf_null_n_bins": rf_null_meta.n_bins,
                            "rf_null_largest_bin_fraction": rf_null_meta.largest_bin_fraction,
                            "rf_null_bin_features": rf_null_meta.bin_features,
                        }
                        if status != "ok" or int(k) > max(rank, 0):
                            row.update(
                                {
                                    "capture": float("nan"),
                                    "effect_minus_unit_shuffle_median": float("nan"),
                                    "effect_minus_random_subspace_median": float("nan"),
                                    "rf_fixed_permutation_null_mean": float("nan"),
                                    "rf_fixed_permutation_null_median": float("nan"),
                                    "rf_fixed_permutation_null_ci_low": float("nan"),
                                    "rf_fixed_permutation_null_ci_high": float("nan"),
                                    "effect_minus_rf_fixed_permutation_median": float("nan"),
                                    "row_status": "not_evaluable",
                                }
                            )
                            metric_rows.append(row)
                            continue
                        cap = _capture(target, vecs[:, : int(k)])
                        nulls = _null_captures(
                            rng=rng,
                            target=target,
                            basis_vecs=vecs,
                            source_matrix=mat,
                            k=int(k),
                            n_nulls=int(args.n_nulls),
                        )
                        row.update(nulls)
                        if bool(args.enable_rf_readout_null):
                            rf_nulls = _rf_null_captures(
                                rng=rng,
                                target=target,
                                source_name=source,
                                source_cov=cov0,
                                source_matrix=mat0,
                                projection=p,
                                bins=rf_null_meta.bins,
                                k=int(k),
                                n_nulls=n_rf_nulls,
                            )
                            row.update(rf_nulls)
                            rf_med = row.get("rf_fixed_permutation_null_median", float("nan"))
                            row["effect_minus_rf_fixed_permutation_median"] = (
                                cap - float(rf_med) if np.isfinite(float(rf_med)) else float("nan")
                            )
                        row.update(
                            {
                                "capture": cap,
                                "effect_minus_unit_shuffle_median": cap - row["unit_shuffle_null_median"],
                                "effect_minus_random_subspace_median": cap - row["random_subspace_null_median"],
                                "row_status": "ok",
                            }
                        )
                        metric_rows.append(row)

    _write_csv(out_dir / "finite_difference_session_summary.csv", session_rows)
    _write_csv(out_dir / "finite_difference_capture_metrics.csv", metric_rows)
    if bool(args.enable_rf_readout_null):
        _write_csv(out_dir / "rf_null_unit_bins.csv", rf_unit_bin_rows)
    _write_csv(out_dir / "finite_difference_metric_summary.csv", summarize_metrics(metric_rows))
    _write_json(
        out_dir / "run_manifest.json",
        {
            "status": "ok",
            "fig3_cache": str(Path(args.fig3_cache).resolve()),
            "fig2_cache": str(Path(args.fig2_cache).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "model_config": str(Path(args.model_config).resolve()),
            "dataset_config": str(Path(args.dataset_config).resolve()),
            "jacobian_provenance": "live_finite_difference_forward_pass_from_checkpoint",
            "jacobian_uses_tangent_maps_cache": False,
            "jacobian_uses_converted_feature_cache": False,
            "model_info": {k: str(v) for k, v in dict(model_info).items()},
            "sessions_requested": requested_sessions,
            "n_sessions_ok": int(sum(1 for row in session_rows if row.get("status") == "ok")),
            "n_metric_rows": len(metric_rows),
            "device": str(model.device),
            "step_px": float(args.step_px),
            "max_samples": int(args.max_samples),
            "n_nulls": int(args.n_nulls),
            "enable_rf_readout_null": bool(args.enable_rf_readout_null),
            "rf_null_n_nulls": n_rf_nulls,
            "rf_null_min_bin_units": int(args.rf_null_min_bin_units),
            "rf_null_bin_features": str(args.rf_null_bin_features),
            "rf_null_session_yaml_dir": str(Path(args.rf_null_session_yaml_dir).resolve()),
            "rescale_mode": str(args.rescale_mode),
            "basis_sources": basis_sources,
            "projection_controls": projection_controls,
            "target_variants": target_variants,
            "k_list": k_list,
            "compact_basis_k": int(args.compact_basis_k),
            "compact_n_folds": int(args.compact_n_folds),
            "compact_group_mode": "trial_inds",
            "static_pc_basis_k": int(args.static_pc_basis_k),
            "static_pc_n_folds": int(args.static_pc_n_folds),
            "static_pc_source_name": static_pc_source,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Finite-difference fitted-twin retinal tangent closure")
    p.add_argument("--fig3-cache", type=Path, default=DEFAULT_FIG3_CACHE)
    p.add_argument("--fig2-cache", type=Path, default=DEFAULT_FIG2_CACHE)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    p.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
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
    p.add_argument("--basis-sources", type=str, default="fd_mean_tangent_matrix,fd_mean_tangent_cov,fd_sample_eye_trace_cov,fd_sample_eye_trace_xfit_compact_k10_cov,fd_sample_eye_trace_xfit_static_pc_k10_cov,fd_tangent_gram_cov")
    p.add_argument("--projection-controls", type=str, default="none,global_rate,target_pc1,global_rate+target_pc1")
    p.add_argument("--target-variants", type=str, default="raw,psd")
    p.add_argument("--k-list", type=str, default="1,2,3,5,10,20")
    p.add_argument("--compact-basis-k", type=int, default=10)
    p.add_argument("--compact-n-folds", type=int, default=5)
    p.add_argument("--static-pc-basis-k", type=int, default=10)
    p.add_argument("--static-pc-n-folds", type=int, default=5)
    p.add_argument("--n-nulls", type=int, default=100)
    p.add_argument("--enable-rf-readout-null", action="store_true")
    p.add_argument("--rf-null-n-nulls", type=int, default=0, help="Constrained null draws; 0 reuses --n-nulls.")
    p.add_argument("--rf-null-min-bin-units", type=int, default=6)
    p.add_argument("--rf-null-bin-features", type=str, default="rf_xy,tangent_norm,mean_rate,ccnorm")
    p.add_argument("--rf-null-session-yaml-dir", type=Path, default=Path("experiments") / "dataset_configs" / "sessions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose-model-load", action="store_true")
    return p


def main() -> None:
    run_analysis(build_parser().parse_args())


if __name__ == "__main__":
    main()
