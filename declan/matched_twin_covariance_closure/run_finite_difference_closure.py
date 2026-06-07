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

from eval.eval_stack_multidataset import load_model
from eval.eval_stack_utils import load_single_dataset, rescale_rhat
from models.config_loader import load_config

from .run_cache_closure import (
    DEFAULT_FIG2_CACHE,
    DEFAULT_FIG3_CACHE,
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


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
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
    rng = np.random.default_rng(int(args.seed))

    metric_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []

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
                **compact_stats,
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
                    cov = payload["cov"]
                    mat = payload["mat"]
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
                        }
                        if status != "ok" or int(k) > max(rank, 0):
                            row.update(
                                {
                                    "capture": float("nan"),
                                    "effect_minus_unit_shuffle_median": float("nan"),
                                    "effect_minus_random_subspace_median": float("nan"),
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
            "model_info": {k: str(v) for k, v in dict(model_info).items()},
            "sessions_requested": requested_sessions,
            "n_sessions_ok": int(sum(1 for row in session_rows if row.get("status") == "ok")),
            "n_metric_rows": len(metric_rows),
            "device": str(model.device),
            "step_px": float(args.step_px),
            "max_samples": int(args.max_samples),
            "n_nulls": int(args.n_nulls),
            "rescale_mode": str(args.rescale_mode),
            "basis_sources": basis_sources,
            "projection_controls": projection_controls,
            "target_variants": target_variants,
            "k_list": k_list,
            "compact_basis_k": int(args.compact_basis_k),
            "compact_n_folds": int(args.compact_n_folds),
            "compact_group_mode": "trial_inds",
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
    p.add_argument("--basis-sources", type=str, default="fd_mean_tangent_matrix,fd_mean_tangent_cov,fd_sample_eye_trace_cov,fd_sample_eye_trace_xfit_compact_k10_cov,fd_tangent_gram_cov")
    p.add_argument("--projection-controls", type=str, default="none,global_rate,target_pc1,global_rate+target_pc1")
    p.add_argument("--target-variants", type=str, default="raw,psd")
    p.add_argument("--k-list", type=str, default="1,2,3,5,10,20")
    p.add_argument("--compact-basis-k", type=int, default=10)
    p.add_argument("--compact-n-folds", type=int, default=5)
    p.add_argument("--n-nulls", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose-model-load", action="store_true")
    return p


def main() -> None:
    run_analysis(build_parser().parse_args())


if __name__ == "__main__":
    main()
