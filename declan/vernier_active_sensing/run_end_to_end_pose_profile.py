#!/usr/bin/env python3
"""End-to-end continuous Vernier pose-profile diagnostic.

This diagnostic optimizes the eye trace itself through the differentiable
Vernier retinal sampler, lag embedding, model core, and readout.  It is meant
as a small trace-level sanity check for continuous hidden-pose inference, not
as a large production sweep.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.temporal_decoding.stimulus_hires import N_LAGS
from scripts.temporal_decoding.rate_computation import _collapse_spatial

from .forward import STIMULUS_NORMALIZATION, load_model_and_readout, renderer_raw_to_model_pixelnorm
from . import synthetic_trajectory_priors as trajectory_priors
from .joint_observer import THETA_LABELS, THETA_MINUS, THETA_PLUS
from .run_trajectory_table_observer import parse_csv_float, parse_csv_str, write_csv, write_json
from .stimulus import RenderGeometry, VernierRetina, VernierSpec, render_world
from .trajectories import DEFAULT_EYE_TRACES_PATH, load_eye_traces

POSE_UNITS_PER_DEGREE = 60.0
START_MODES = ("zero", "mean_train", "nearest_train", "brownian")
PRIOR_PROCESS_MODELS = ("brownian", "ou")
TRAJECTORY_PRIOR_SOURCES = ("cache_train", "eye_traces")
PRIOR_CENTER_MODES = ("none", "zero_mean", "start_zero", "source_grand_mean")


def parse_csv_int(text: str) -> list[int]:
    return [int(part) for part in parse_csv_str(text)]


def _condition_from_cache_path(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("rates_") or "_fd" not in stem:
        raise ValueError(f"Unexpected rate cache filename: {path.name}")
    return stem[len("rates_") : stem.rindex("_fd")]


def _load_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as npz:
        condition = str(npz["condition"][0]) if "condition" in npz else _condition_from_cache_path(path)
        fd_step = float(np.asarray(npz["fd_step_arcmin"])[0])
        inference_mode = str(npz["inference_mode"][0]) if "inference_mode" in npz else "framewise"
        lengths = np.asarray(npz["lengths"], dtype=np.int64)
        t = int(np.min(lengths))
        return {
            "path": path,
            "condition": condition,
            "fd_step_arcmin": fd_step,
            "inference_mode": inference_mode,
            "plus_rates": np.asarray(npz["plus"], dtype=np.float32)[:, :t],
            "minus_rates": np.asarray(npz["minus"], dtype=np.float32)[:, :t],
            "poses_deg": np.asarray(npz["poses"], dtype=np.float32)[:, :t],
            "n_timebins": t,
        }


def _load_caches(source_dir: Path) -> dict[tuple[str, float, str], dict[str, Any]]:
    caches: dict[tuple[str, float, str], dict[str, Any]] = {}
    for path in sorted((source_dir / "cache").glob("rates_*_fd*arcmin.npz")):
        cache = _load_cache(path)
        caches[(str(cache["condition"]), float(cache["fd_step_arcmin"]), str(cache["inference_mode"]))] = cache
    if not caches:
        raise FileNotFoundError(f"No Vernier rate caches found under {source_dir / 'cache'}")
    return caches


def _apply_prior_center_mode(traces_deg: np.ndarray, center_mode: str) -> np.ndarray:
    mode = str(center_mode)
    traces = np.asarray(traces_deg, dtype=np.float64)
    if mode == "none":
        return traces.copy()
    return trajectory_priors.recenter_traces(traces, mode)


@lru_cache(maxsize=16)
def _cached_eye_trace_prior_poses_arcmin(
    eye_trace_path: str,
    n_timebins: int,
    center_mode: str,
    max_prior_source_traces: int,
    seed: int,
) -> np.ndarray:
    trace_set = load_eye_traces(Path(eye_trace_path))
    ok = np.flatnonzero(np.asarray(trace_set.durations, dtype=np.int64) >= int(n_timebins))
    if ok.size == 0:
        raise ValueError(f"No eye traces in {eye_trace_path} have at least {n_timebins} frames")
    if int(max_prior_source_traces) > 0 and int(max_prior_source_traces) < ok.size:
        rng = np.random.default_rng(int(seed))
        ok = np.sort(rng.choice(ok, size=int(max_prior_source_traces), replace=False))
    traces_deg = np.asarray(trace_set.traces[ok, : int(n_timebins), :], dtype=np.float64)
    traces_deg = _apply_prior_center_mode(traces_deg, str(center_mode))
    return traces_deg * POSE_UNITS_PER_DEGREE


def _prior_poses_for_trial(
    *,
    args: argparse.Namespace,
    cache_train_poses_arcmin: np.ndarray,
    n_timebins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = str(args.trajectory_prior_source)
    if source == "cache_train":
        poses = np.asarray(cache_train_poses_arcmin, dtype=np.float64)
        if str(args.prior_center_mode) != "none":
            poses = _apply_prior_center_mode(poses / POSE_UNITS_PER_DEGREE, str(args.prior_center_mode))
            poses = poses * POSE_UNITS_PER_DEGREE
        return poses, {
            "trajectory_prior_source": "cache_train",
            "eye_trace_path": "",
            "n_prior_source_traces": int(poses.shape[0]),
        }
    if source == "eye_traces":
        poses = _cached_eye_trace_prior_poses_arcmin(
            str(Path(args.eye_traces)),
            int(n_timebins),
            str(args.prior_center_mode),
            int(args.max_prior_source_traces),
            int(args.seed),
        )
        return poses, {
            "trajectory_prior_source": "eye_traces",
            "eye_trace_path": str(Path(args.eye_traces)),
            "n_prior_source_traces": int(poses.shape[0]),
        }
    raise ValueError(f"Unsupported trajectory prior source {source!r}; expected {TRAJECTORY_PRIOR_SOURCES}")


def _build_spec(args: argparse.Namespace, offset_arcmin: float) -> VernierSpec:
    return VernierSpec(
        offset_arcmin=float(offset_arcmin),
        bar_width_arcmin=float(args.bar_width_arcmin),
        gap_arcmin=float(args.gap_arcmin),
        bar_length_arcmin=float(args.bar_length_arcmin),
        contrast=float(args.contrast),
        polarity=str(args.polarity),
        orientation_deg=float(args.stimulus_orientation_deg),
    )


def _torch_lag_embed(movie: torch.Tensor, n_lags: int) -> torch.Tensor:
    # Keep the exact lag ordering from scripts.spatial_info.embed_time_lags.
    if movie.dim() == 3:
        movie = movie.unsqueeze(1)
    total_t, channels, height, width = movie.shape
    out_t = total_t - int(n_lags) + 1
    lagged = torch.empty(
        (out_t, channels, int(n_lags), height, width),
        dtype=movie.dtype,
        device=movie.device,
    )
    for lag in range(int(n_lags)):
        lagged[:, :, lag] = movie[int(n_lags) - 1 - lag : total_t - lag]
    return lagged


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def _forward_counts(
    *,
    model: Any,
    readout: torch.nn.Module,
    retina: VernierRetina,
    world: torch.Tensor,
    pose_arcmin: torch.Tensor,
    bin_seconds: float,
    n_lags: int,
    spatial_collapse: str,
    max_units: int,
) -> torch.Tensor:
    pose_deg = pose_arcmin / POSE_UNITS_PER_DEGREE
    pad = pose_deg[:1].expand(max(int(n_lags) - 1, 0), -1)
    padded_pose = torch.cat([pad, pose_deg], dim=0)
    movie_raw = retina(world, padded_pose)[0, 0]
    movie = renderer_raw_to_model_pixelnorm(movie_raw, max_raw=float(retina.geometry.max_raw))
    stim = _torch_lag_embed(movie, int(n_lags))
    feats = model.model.core_forward(stim, None)
    feats_last = feats[:, :, -1]
    y = readout(feats_last)
    rates = _collapse_spatial(model.model.activation(y), method=str(spatial_collapse))
    if int(max_units) > 0:
        rates = rates[:, : int(max_units)]
    return rates * float(bin_seconds)


def _regularized_cov(samples: np.ndarray, floor: float) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    cov = np.cov(arr, rowvar=False) if arr.shape[0] > 1 else np.eye(arr.shape[1]) * float(floor)
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    cov = 0.5 * (cov + cov.T)
    eig = np.linalg.eigvalsh(cov)
    min_eig = float(np.min(eig))
    if min_eig < float(floor):
        cov = cov + (float(floor) - min_eig) * np.eye(cov.shape[0])
    return cov


def _fit_brownian_prior(
    train_poses_arcmin: np.ndarray,
    *,
    floor: float,
    process_cov_scale: float,
) -> dict[str, Any]:
    poses = np.asarray(train_poses_arcmin, dtype=np.float64)
    init = poses[:, 0]
    steps = np.diff(poses, axis=1).reshape(-1, 2)
    init_cov = _regularized_cov(init, float(floor)) * float(process_cov_scale)
    step_cov = _regularized_cov(steps, float(floor)) * float(process_cov_scale)
    return {
        "process_model": "brownian",
        "init_mean": np.mean(init, axis=0),
        "step_mean": np.mean(steps, axis=0),
        "init_inv": np.linalg.inv(init_cov),
        "step_inv": np.linalg.inv(step_cov),
        "init_cov": init_cov,
        "step_cov": step_cov,
    }


def _fit_ou_prior(
    train_poses_arcmin: np.ndarray,
    *,
    floor: float,
    process_cov_scale: float,
    max_abs_eigenvalue: float,
) -> dict[str, Any]:
    poses = np.asarray(train_poses_arcmin, dtype=np.float64)
    init = poses[:, 0]
    prev = poses[:, :-1, :].reshape(-1, 2)
    nxt = poses[:, 1:, :].reshape(-1, 2)
    if prev.shape[0] == 0:
        transition = np.zeros((2, 2), dtype=np.float64)
        residual = np.zeros((1, 2), dtype=np.float64)
    else:
        coef, *_ = np.linalg.lstsq(prev, nxt, rcond=None)
        transition = trajectory_priors.stabilize_transition(
            coef.T,
            max_abs_eigenvalue=float(max_abs_eigenvalue),
        )
        residual = nxt - prev @ transition.T
    init_cov = _regularized_cov(init, float(floor)) * float(process_cov_scale)
    innovation_cov = _regularized_cov(residual, float(floor)) * float(process_cov_scale)
    return {
        "process_model": "ou",
        "init_mean": np.mean(init, axis=0),
        "init_inv": np.linalg.inv(init_cov),
        "init_cov": init_cov,
        "transition_matrix": transition,
        "innovation_inv": np.linalg.inv(innovation_cov),
        "innovation_cov": innovation_cov,
        "transition_eigenvalues": np.linalg.eigvals(transition),
    }


def _fit_pose_prior(
    train_poses_arcmin: np.ndarray,
    *,
    process_model: str,
    floor: float,
    process_cov_scale: float,
    ou_max_abs_eigenvalue: float,
) -> dict[str, Any]:
    if str(process_model) == "brownian":
        return _fit_brownian_prior(
            train_poses_arcmin,
            floor=float(floor),
            process_cov_scale=float(process_cov_scale),
        )
    if str(process_model) == "ou":
        return _fit_ou_prior(
            train_poses_arcmin,
            floor=float(floor),
            process_cov_scale=float(process_cov_scale),
            max_abs_eigenvalue=float(ou_max_abs_eigenvalue),
        )
    raise ValueError(f"Unsupported prior process model {process_model!r}; expected {PRIOR_PROCESS_MODELS}")


def _prior_energy_torch(pose_arcmin: torch.Tensor, prior: dict[str, Any]) -> torch.Tensor:
    process_model = str(prior.get("process_model", "brownian"))
    init_resid = pose_arcmin[0] - prior["init_mean"]
    init_energy = 0.5 * init_resid @ prior["init_inv"] @ init_resid
    if pose_arcmin.shape[0] <= 1:
        return init_energy
    if process_model == "brownian":
        step_resid = pose_arcmin[1:] - pose_arcmin[:-1] - prior["step_mean"][None, :]
        step_energy = 0.5 * torch.sum((step_resid @ prior["step_inv"]) * step_resid)
        return init_energy + step_energy
    if process_model == "ou":
        pred = pose_arcmin[:-1] @ prior["transition_matrix"].T
        innovation_resid = pose_arcmin[1:] - pred
        innovation_energy = 0.5 * torch.sum(
            (innovation_resid @ prior["innovation_inv"]) * innovation_resid
        )
        return init_energy + innovation_energy
    raise ValueError(f"Unsupported prior process model {process_model!r}")


def _as_torch_prior(prior_np: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {"process_model": str(prior_np.get("process_model", "brownian"))}
    for key, value in prior_np.items():
        if isinstance(value, np.ndarray) and np.isrealobj(value):
            out[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
    return out


def _poisson_energy(pred_counts: torch.Tensor, obs_counts: torch.Tensor, likelihood_scale: float) -> torch.Tensor:
    mu = torch.clamp(pred_counts, min=1e-8)
    return -float(likelihood_scale) * torch.sum(obs_counts * torch.log(mu) - mu)


def _score_fixed_pose(
    *,
    pose_arcmin_np: np.ndarray,
    model: Any,
    readout: torch.nn.Module,
    retina: VernierRetina,
    world: torch.Tensor,
    obs_counts: torch.Tensor,
    prior: dict[str, torch.Tensor],
    bin_seconds: float,
    n_lags: int,
    spatial_collapse: str,
    likelihood_scale: float,
    prior_weight: float,
    max_units: int,
) -> tuple[float, float, float]:
    pose = torch.as_tensor(pose_arcmin_np, dtype=torch.float32, device=obs_counts.device)
    with torch.no_grad():
        pred = _forward_counts(
            model=model,
            readout=readout,
            retina=retina,
            world=world,
            pose_arcmin=pose,
            bin_seconds=float(bin_seconds),
            n_lags=int(n_lags),
            spatial_collapse=str(spatial_collapse),
            max_units=int(max_units),
        )
        like_energy = _poisson_energy(pred, obs_counts, float(likelihood_scale))
        prior_energy = _prior_energy_torch(pose, prior) * float(prior_weight)
        energy = like_energy + prior_energy
    return -float(energy.detach().cpu()), float(like_energy.detach().cpu()), float(prior_energy.detach().cpu())


def _nearest_train_pose(train_poses: np.ndarray, target: np.ndarray) -> np.ndarray:
    diff = np.asarray(train_poses, dtype=np.float64) - np.asarray(target, dtype=np.float64)[None, :, :]
    rmse = np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=1))
    return np.asarray(train_poses[int(np.argmin(rmse))], dtype=np.float64)


def _sample_prior_start(prior_np: dict[str, Any], t: int, rng: np.random.Generator) -> np.ndarray:
    path = np.zeros((int(t), 2), dtype=np.float64)
    path[0] = rng.multivariate_normal(prior_np["init_mean"], prior_np["init_cov"])
    process_model = str(prior_np.get("process_model", "brownian"))
    for idx in range(1, int(t)):
        if process_model == "brownian":
            step = rng.multivariate_normal(prior_np["step_mean"], prior_np["step_cov"])
            path[idx] = path[idx - 1] + step
        elif process_model == "ou":
            innovation = rng.multivariate_normal(np.zeros(2, dtype=np.float64), prior_np["innovation_cov"])
            path[idx] = prior_np["transition_matrix"] @ path[idx - 1] + innovation
        else:
            raise ValueError(f"Unsupported prior process model {process_model!r}")
    return path


def _build_starts(
    *,
    start_modes: list[str],
    train_poses: np.ndarray,
    true_pose: np.ndarray,
    prior_np: dict[str, np.ndarray],
    n_brownian_starts: int,
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    starts: list[tuple[str, np.ndarray]] = []
    t = int(true_pose.shape[0])
    if "zero" in start_modes:
        starts.append(("zero", np.zeros((t, 2), dtype=np.float64)))
    if "mean_train" in start_modes:
        starts.append(("mean_train", np.mean(train_poses, axis=0)))
    if "nearest_train" in start_modes:
        starts.append(("nearest_train", _nearest_train_pose(train_poses, true_pose)))
    if "brownian" in start_modes:
        rng = np.random.default_rng(int(seed))
        process_model = str(prior_np.get("process_model", "brownian"))
        for idx in range(int(n_brownian_starts)):
            starts.append((f"{process_model}_sample_{idx}", _sample_prior_start(prior_np, t, rng)))
    if not starts:
        raise ValueError("At least one start mode is required")
    return starts


def _optimize_pose(
    *,
    start_pose_np: np.ndarray,
    model: Any,
    readout: torch.nn.Module,
    retina: VernierRetina,
    world: torch.Tensor,
    obs_counts: torch.Tensor,
    prior: dict[str, torch.Tensor],
    bin_seconds: float,
    n_lags: int,
    spatial_collapse: str,
    likelihood_scale: float,
    prior_weight: float,
    max_units: int,
    max_iter: int,
    optimizer_name: str,
    learning_rate: float,
) -> tuple[float, float, float, np.ndarray, int, bool]:
    pose = torch.as_tensor(start_pose_np, dtype=torch.float32, device=obs_counts.device).detach().requires_grad_(True)
    last: dict[str, torch.Tensor] = {}

    def evaluate() -> torch.Tensor:
        pred = _forward_counts(
            model=model,
            readout=readout,
            retina=retina,
            world=world,
            pose_arcmin=pose,
            bin_seconds=float(bin_seconds),
            n_lags=int(n_lags),
            spatial_collapse=str(spatial_collapse),
            max_units=int(max_units),
        )
        like_energy = _poisson_energy(pred, obs_counts, float(likelihood_scale))
        prior_energy = _prior_energy_torch(pose, prior) * float(prior_weight)
        energy = like_energy + prior_energy
        last["energy"] = energy.detach()
        last["like_energy"] = like_energy.detach()
        last["prior_energy"] = prior_energy.detach()
        return energy

    opt_name = str(optimizer_name).lower()
    if opt_name == "adam":
        success = True
        m = torch.zeros_like(pose)
        v = torch.zeros_like(pose)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        for idx in range(int(max_iter)):
            energy = evaluate()
            (grad,) = torch.autograd.grad(energy, pose)
            with torch.no_grad():
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * grad * grad
                m_hat = m / (1.0 - beta1 ** (idx + 1))
                v_hat = v / (1.0 - beta2 ** (idx + 1))
                updated = pose - float(learning_rate) * m_hat / (torch.sqrt(v_hat) + eps)
            pose = updated.detach().requires_grad_(True)
        evaluate()
        n_iter = int(max_iter)
    elif opt_name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            [pose],
            max_iter=int(max_iter),
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-5,
            tolerance_change=1e-7,
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            energy = evaluate()
            energy.backward()
            return energy

        try:
            optimizer.step(closure)
            success = True
        except RuntimeError:
            success = False
            closure()
        n_iter = int(optimizer.state.get(pose, {}).get("n_iter", -1))
    else:
        raise ValueError(f"Unsupported optimizer={optimizer_name!r}; expected 'adam' or 'lbfgs'")
    energy = float(last["energy"].detach().cpu())
    like_energy = float(last["like_energy"].detach().cpu())
    prior_energy = float(last["prior_energy"].detach().cpu())
    return (
        -energy,
        like_energy,
        prior_energy,
        pose.detach().cpu().numpy().astype(np.float64),
        n_iter,
        bool(success),
    )


def _predict(scores: dict[str, float]) -> str:
    if scores[THETA_PLUS] >= scores[THETA_MINUS]:
        return THETA_PLUS
    return THETA_MINUS


def _best_margin_threshold(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    margins = np.asarray([row[f"{prefix}_margin_plus_minus"] for row in rows], dtype=np.float64)
    is_plus = np.asarray([row["true_label"] == THETA_PLUS for row in rows], dtype=bool)
    finite = np.isfinite(margins)
    if int(np.sum(finite)) == 0:
        return {
            f"{prefix}_best_margin_accuracy": float("nan"),
            f"{prefix}_best_margin_threshold": float("nan"),
            f"{prefix}_best_margin_direction": float("nan"),
        }
    margins = margins[finite]
    is_plus = is_plus[finite]
    uniq = np.unique(margins)
    if uniq.size == 1:
        thresholds = np.asarray([uniq[0]], dtype=np.float64)
    else:
        thresholds = np.concatenate(
            [
                [np.nextafter(uniq[0], -np.inf)],
                0.5 * (uniq[:-1] + uniq[1:]),
                [np.nextafter(uniq[-1], np.inf)],
            ]
        )
    best_acc = -1.0
    best_threshold = float(thresholds[0])
    best_direction = 1.0
    for direction in (1.0, -1.0):
        signed = margins * direction
        for threshold in thresholds * direction:
            pred_plus = signed >= threshold
            acc = float(np.mean(pred_plus == is_plus))
            if acc > best_acc:
                best_acc = acc
                best_threshold = float(threshold / direction)
                best_direction = float(direction)
    return {
        f"{prefix}_best_margin_accuracy": float(best_acc),
        f"{prefix}_best_margin_threshold": float(best_threshold),
        f"{prefix}_best_margin_direction": float(best_direction),
    }


def _score_one_trial(
    *,
    args: argparse.Namespace,
    cache: dict[str, Any],
    model: Any,
    readout: torch.nn.Module,
    retina: VernierRetina,
    geometry: RenderGeometry,
    trace_index: int,
    true_label: str,
    include_self: bool,
) -> dict[str, Any]:
    device = torch.device(str(args.device))
    label_to_rates = {THETA_PLUS: cache["plus_rates"], THETA_MINUS: cache["minus_rates"]}
    t = min(int(cache["n_timebins"]), int(args.max_timebins) if int(args.max_timebins) > 0 else int(cache["n_timebins"]))
    trace_idx = int(trace_index)
    poses_arcmin = np.asarray(cache["poses_deg"][:, :t], dtype=np.float64) * POSE_UNITS_PER_DEGREE
    true_pose = poses_arcmin[trace_idx]
    obs = np.asarray(label_to_rates[str(true_label)][trace_idx, :t], dtype=np.float32) * float(args.bin_seconds)
    if int(args.max_units) > 0:
        obs = obs[:, : int(args.max_units)]
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

    train_mask = np.ones(poses_arcmin.shape[0], dtype=bool)
    if not bool(include_self):
        train_mask[trace_idx] = False
    train_poses = poses_arcmin[train_mask]
    prior_poses, prior_source_meta = _prior_poses_for_trial(
        args=args,
        cache_train_poses_arcmin=train_poses,
        n_timebins=int(t),
    )
    prior_np = _fit_pose_prior(
        prior_poses,
        process_model=str(args.prior_process_model),
        floor=float(args.covariance_floor_arcmin2),
        process_cov_scale=float(args.process_cov_scale),
        ou_max_abs_eigenvalue=float(args.ou_max_abs_eigenvalue),
    )
    prior = _as_torch_prior(prior_np, device)
    starts = _build_starts(
        start_modes=list(args.start_modes),
        train_poses=prior_poses,
        true_pose=true_pose,
        prior_np=prior_np,
        n_brownian_starts=int(args.n_brownian_starts),
        seed=int(args.seed) + 7919 * trace_idx,
    )

    label_worlds = {}
    for label in THETA_LABELS:
        spec = _build_spec(args, float(cache["fd_step_arcmin"]) if label == THETA_PLUS else -float(cache["fd_step_arcmin"]))
        label_worlds[label] = render_world(spec, geometry, device=str(device))

    neutral_spec = _build_spec(args, float(args.neutral_offset_arcmin))
    neutral_world = render_world(neutral_spec, geometry, device=str(device))
    best_neutral: tuple[float, float, float, np.ndarray, int, bool, str] | None = None
    for start_name, start_pose in starts:
        score, like_energy, prior_energy, pose_hat, n_iter, success = _optimize_pose(
            start_pose_np=start_pose,
            model=model,
            readout=readout,
            retina=retina,
            world=neutral_world,
            obs_counts=obs_t,
            prior=prior,
            bin_seconds=float(args.bin_seconds),
            n_lags=int(args.n_lags),
            spatial_collapse=str(args.spatial_collapse),
            likelihood_scale=float(args.likelihood_scale),
            prior_weight=float(args.prior_weight),
            max_units=int(args.max_units),
            max_iter=int(args.max_iter),
            optimizer_name=str(args.optimizer),
            learning_rate=float(args.learning_rate),
        )
        if best_neutral is None or score > best_neutral[0]:
            best_neutral = (score, like_energy, prior_energy, pose_hat, n_iter, success, start_name)
    if best_neutral is None:
        raise RuntimeError("No neutral-pose optimization starts were evaluated")
    neutral_pose = best_neutral[3]

    fixed_scores: dict[str, float] = {}
    zero_scores: dict[str, float] = {}
    neutral_scores: dict[str, float] = {}
    profile_scores: dict[str, float] = {}
    profile_pose_by_label: dict[str, np.ndarray] = {}
    profile_meta: dict[str, dict[str, Any]] = {}
    zero_pose = np.zeros_like(true_pose)
    for label in THETA_LABELS:
        world = label_worlds[label]
        fixed_scores[label], _fixed_like, _fixed_prior = _score_fixed_pose(
            pose_arcmin_np=true_pose,
            model=model,
            readout=readout,
            retina=retina,
            world=world,
            obs_counts=obs_t,
            prior=prior,
            bin_seconds=float(args.bin_seconds),
            n_lags=int(args.n_lags),
            spatial_collapse=str(args.spatial_collapse),
            likelihood_scale=float(args.likelihood_scale),
            prior_weight=float(args.prior_weight),
            max_units=int(args.max_units),
        )
        zero_scores[label], _zero_like, _zero_prior = _score_fixed_pose(
            pose_arcmin_np=zero_pose,
            model=model,
            readout=readout,
            retina=retina,
            world=world,
            obs_counts=obs_t,
            prior=prior,
            bin_seconds=float(args.bin_seconds),
            n_lags=int(args.n_lags),
            spatial_collapse=str(args.spatial_collapse),
            likelihood_scale=float(args.likelihood_scale),
            prior_weight=float(args.prior_weight),
            max_units=int(args.max_units),
        )
        neutral_scores[label], _neutral_like, _neutral_prior = _score_fixed_pose(
            pose_arcmin_np=neutral_pose,
            model=model,
            readout=readout,
            retina=retina,
            world=world,
            obs_counts=obs_t,
            prior=prior,
            bin_seconds=float(args.bin_seconds),
            n_lags=int(args.n_lags),
            spatial_collapse=str(args.spatial_collapse),
            likelihood_scale=float(args.likelihood_scale),
            prior_weight=float(args.prior_weight),
            max_units=int(args.max_units),
        )
        best: tuple[float, float, float, np.ndarray, int, bool, str] | None = None
        for start_name, start_pose in starts:
            score, like_energy, prior_energy, pose_hat, n_iter, success = _optimize_pose(
                start_pose_np=start_pose,
                model=model,
                readout=readout,
                retina=retina,
                world=world,
                obs_counts=obs_t,
                prior=prior,
                bin_seconds=float(args.bin_seconds),
                n_lags=int(args.n_lags),
                spatial_collapse=str(args.spatial_collapse),
                likelihood_scale=float(args.likelihood_scale),
                prior_weight=float(args.prior_weight),
                max_units=int(args.max_units),
                max_iter=int(args.max_iter),
                optimizer_name=str(args.optimizer),
                learning_rate=float(args.learning_rate),
            )
            if best is None or score > best[0]:
                best = (score, like_energy, prior_energy, pose_hat, n_iter, success, start_name)
        if best is None:
            raise RuntimeError("No optimization starts were evaluated")
        profile_scores[label] = float(best[0])
        profile_pose_by_label[label] = best[3]
        profile_meta[label] = {
            "like_energy": float(best[1]),
            "prior_energy": float(best[2]),
            "n_iter": int(best[4]),
            "success": bool(best[5]),
            "start": str(best[6]),
        }

    true = str(true_label)
    other = THETA_MINUS if true == THETA_PLUS else THETA_PLUS
    pred_profile = _predict(profile_scores)
    pred_known = _predict(fixed_scores)
    pred_zero = _predict(zero_scores)
    pred_neutral = _predict(neutral_scores)
    pose_hat_true = profile_pose_by_label[true]
    pose_hat_pred = profile_pose_by_label[pred_profile]
    rmse_true = float(np.sqrt(np.mean(np.sum((pose_hat_true - true_pose) ** 2, axis=1))))
    rmse_pred = float(np.sqrt(np.mean(np.sum((pose_hat_pred - true_pose) ** 2, axis=1))))
    neutral_rmse = float(np.sqrt(np.mean(np.sum((neutral_pose - true_pose) ** 2, axis=1))))
    true_pose_rms = float(np.sqrt(np.mean(np.sum(true_pose * true_pose, axis=1))))
    profile_pose_rms_true = float(np.sqrt(np.mean(np.sum(pose_hat_true * pose_hat_true, axis=1))))
    profile_pose_rms_pred = float(np.sqrt(np.mean(np.sum(pose_hat_pred * pose_hat_pred, axis=1))))
    neutral_pose_rms = float(np.sqrt(np.mean(np.sum(neutral_pose * neutral_pose, axis=1))))
    flat_trace_input = true_pose_rms <= float(args.flat_trace_tolerance_arcmin)
    flat_trace_profile_pass = (
        bool(profile_pose_rms_pred <= float(args.flat_trace_tolerance_arcmin))
        if flat_trace_input
        else float("nan")
    )
    flat_trace_neutral_pass = (
        bool(neutral_pose_rms <= float(args.flat_trace_tolerance_arcmin))
        if flat_trace_input
        else float("nan")
    )
    pose_estimate_path = ""
    if bool(args.save_pose_estimates):
        pose_dir = Path(args.out_dir) / "pose_estimates"
        pose_dir.mkdir(parents=True, exist_ok=True)
        pose_path = pose_dir / (
            f"{cache['condition']}_fd{float(cache['fd_step_arcmin']):0.4f}_"
            f"trace{trace_idx:04d}_{true}.npz"
        )
        np.savez_compressed(
            pose_path,
            true_pose_arcmin=true_pose.astype(np.float32),
            neutral_pose_arcmin=neutral_pose.astype(np.float32),
            profile_pose_plus_arcmin=profile_pose_by_label[THETA_PLUS].astype(np.float32),
            profile_pose_minus_arcmin=profile_pose_by_label[THETA_MINUS].astype(np.float32),
            pred_profile=np.asarray([pred_profile]),
            true_label=np.asarray([true]),
            condition=np.asarray([str(cache["condition"])]),
            fd_step_arcmin=np.asarray([float(cache["fd_step_arcmin"])]),
            trace_index=np.asarray([int(trace_idx)]),
        )
        pose_estimate_path = str(pose_path)
    return {
        "condition": str(cache["condition"]),
        "fd_step_arcmin": float(cache["fd_step_arcmin"]),
        "inference_mode": str(cache["inference_mode"]),
        "trace_index": int(trace_idx),
        "true_label": true,
        "catalog_mode": "include_self" if bool(include_self) else "leave_one_out",
        "trajectory_prior": (
            f"end_to_end_{args.prior_process_model}_{args.trajectory_prior_source}_"
            f"{'include_self' if bool(include_self) else 'leave_one_out'}"
        ),
        "trajectory_prior_source": str(prior_source_meta["trajectory_prior_source"]),
        "prior_process_model": str(args.prior_process_model),
        "prior_center_mode": str(args.prior_center_mode),
        "eye_trace_path": str(prior_source_meta["eye_trace_path"]),
        "n_prior_source_traces": int(prior_source_meta["n_prior_source_traces"]),
        "nearest_train_start_uses_true_pose": bool("nearest_train" in set(args.start_modes)),
        "n_timebins": int(t),
        "n_units": int(obs.shape[1]),
        "n_starts": int(len(starts)),
        "likelihood_scale": float(args.likelihood_scale),
        "prior_weight": float(args.prior_weight),
        "process_cov_scale": float(args.process_cov_scale),
        "pred_profile": pred_profile,
        "pred_known": pred_known,
        "pred_zero": pred_zero,
        "pred_neutral": pred_neutral,
        "profile_correct": bool(pred_profile == true),
        "known_correct": bool(pred_known == true),
        "zero_correct": bool(pred_zero == true),
        "neutral_correct": bool(pred_neutral == true),
        "profile_score": float(profile_scores[true] - profile_scores[other]),
        "known_score": float(fixed_scores[true] - fixed_scores[other]),
        "zero_score": float(zero_scores[true] - zero_scores[other]),
        "neutral_score": float(neutral_scores[true] - neutral_scores[other]),
        "profile_margin_plus_minus": float(profile_scores[THETA_PLUS] - profile_scores[THETA_MINUS]),
        "known_margin_plus_minus": float(fixed_scores[THETA_PLUS] - fixed_scores[THETA_MINUS]),
        "zero_margin_plus_minus": float(zero_scores[THETA_PLUS] - zero_scores[THETA_MINUS]),
        "neutral_margin_plus_minus": float(neutral_scores[THETA_PLUS] - neutral_scores[THETA_MINUS]),
        "profile_score_plus": float(profile_scores[THETA_PLUS]),
        "profile_score_minus": float(profile_scores[THETA_MINUS]),
        "known_score_plus": float(fixed_scores[THETA_PLUS]),
        "known_score_minus": float(fixed_scores[THETA_MINUS]),
        "zero_score_plus": float(zero_scores[THETA_PLUS]),
        "zero_score_minus": float(zero_scores[THETA_MINUS]),
        "neutral_score_plus": float(neutral_scores[THETA_PLUS]),
        "neutral_score_minus": float(neutral_scores[THETA_MINUS]),
        "profile_pose_rmse_arcmin_true": rmse_true,
        "profile_pose_rmse_arcmin_pred": rmse_pred,
        "neutral_pose_rmse_arcmin_true": neutral_rmse,
        "true_pose_rms_arcmin": true_pose_rms,
        "profile_pose_rms_arcmin_true_label": profile_pose_rms_true,
        "profile_pose_rms_arcmin_pred_label": profile_pose_rms_pred,
        "neutral_pose_rms_arcmin": neutral_pose_rms,
        "flat_trace_input": bool(flat_trace_input),
        "flat_trace_tolerance_arcmin": float(args.flat_trace_tolerance_arcmin),
        "profile_pred_flat_trace_guardrail_pass": flat_trace_profile_pass,
        "neutral_flat_trace_guardrail_pass": flat_trace_neutral_pass,
        "pose_estimate_path": pose_estimate_path,
        "profile_start_true": profile_meta[true]["start"],
        "profile_start_other": profile_meta[other]["start"],
        "profile_start_pred": profile_meta[pred_profile]["start"],
        "neutral_start": str(best_neutral[6]),
        "profile_success_true": profile_meta[true]["success"],
        "profile_success_other": profile_meta[other]["success"],
        "neutral_success": bool(best_neutral[5]),
        "profile_n_iter_true": profile_meta[true]["n_iter"],
        "profile_n_iter_other": profile_meta[other]["n_iter"],
        "neutral_n_iter": int(best_neutral[4]),
        "profile_like_energy_true": profile_meta[true]["like_energy"],
        "profile_like_energy_other": profile_meta[other]["like_energy"],
        "profile_prior_energy_true": profile_meta[true]["prior_energy"],
        "profile_prior_energy_other": profile_meta[other]["prior_energy"],
        "neutral_like_energy": float(best_neutral[1]),
        "neutral_prior_energy": float(best_neutral[2]),
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = (
        "catalog_mode",
        "condition",
        "fd_step_arcmin",
        "trajectory_prior_source",
        "prior_process_model",
        "prior_center_mode",
        "likelihood_scale",
        "prior_weight",
        "process_cov_scale",
    )
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for key, grp in sorted(groups.items()):
        profile_correct = np.asarray([row["profile_correct"] for row in grp], dtype=bool)
        known_correct = np.asarray([row["known_correct"] for row in grp], dtype=bool)
        zero_correct = np.asarray([row["zero_correct"] for row in grp], dtype=bool)
        neutral_correct = np.asarray([row["neutral_correct"] for row in grp], dtype=bool)
        flat_profile = [
            row["profile_pred_flat_trace_guardrail_pass"]
            for row in grp
            if isinstance(row.get("profile_pred_flat_trace_guardrail_pass"), (bool, np.bool_))
        ]
        flat_neutral = [
            row["neutral_flat_trace_guardrail_pass"]
            for row in grp
            if isinstance(row.get("neutral_flat_trace_guardrail_pass"), (bool, np.bool_))
        ]
        out.append(
            {
                **{name: value for name, value in zip(keys, key, strict=True)},
                "n": len(grp),
                "profile_accuracy": float(np.mean(profile_correct)),
                "known_accuracy": float(np.mean(known_correct)),
                "zero_accuracy": float(np.mean(zero_correct)),
                "neutral_accuracy": float(np.mean(neutral_correct)),
                "mean_profile_score": float(np.mean([row["profile_score"] for row in grp])),
                "mean_known_score": float(np.mean([row["known_score"] for row in grp])),
                "mean_zero_score": float(np.mean([row["zero_score"] for row in grp])),
                "mean_neutral_score": float(np.mean([row["neutral_score"] for row in grp])),
                "mean_profile_pose_rmse_arcmin_true": float(np.mean([row["profile_pose_rmse_arcmin_true"] for row in grp])),
                "mean_profile_pose_rmse_arcmin_pred": float(np.mean([row["profile_pose_rmse_arcmin_pred"] for row in grp])),
                "mean_neutral_pose_rmse_arcmin_true": float(
                    np.mean([row["neutral_pose_rmse_arcmin_true"] for row in grp])
                ),
                "mean_true_pose_rms_arcmin": float(np.mean([row["true_pose_rms_arcmin"] for row in grp])),
                "mean_profile_pose_rms_arcmin_pred_label": float(
                    np.mean([row["profile_pose_rms_arcmin_pred_label"] for row in grp])
                ),
                "mean_neutral_pose_rms_arcmin": float(np.mean([row["neutral_pose_rms_arcmin"] for row in grp])),
                "flat_trace_input_fraction": float(np.mean([row["flat_trace_input"] for row in grp])),
                "profile_pred_flat_trace_guardrail_rate": float(np.mean(flat_profile)) if flat_profile else float("nan"),
                "neutral_flat_trace_guardrail_rate": float(np.mean(flat_neutral)) if flat_neutral else float("nan"),
                **_best_margin_threshold(grp, "profile"),
                **_best_margin_threshold(grp, "known"),
                **_best_margin_threshold(grp, "zero"),
                **_best_margin_threshold(grp, "neutral"),
            }
        )
    return out


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "end_to_end_pose_profile_trials.csv"
    summary_path = out_dir / "end_to_end_pose_profile_summary.csv"
    caches = _load_caches(Path(args.source_dir))
    device = torch.device(str(args.device))
    geometry = RenderGeometry()
    model, readout = load_model_and_readout(str(device))
    _freeze(model.model)
    _freeze(readout)
    retina = VernierRetina(geometry=geometry).to(device)
    retina.eval()

    rows: list[dict[str, Any]] = []
    condition_filter = set(args.conditions)
    fd_filter = {round(float(fd), 10) for fd in args.fd_steps_arcmin}
    for (_condition, _fd, _mode), cache in sorted(caches.items()):
        if condition_filter and str(cache["condition"]) not in condition_filter:
            continue
        if fd_filter and round(float(cache["fd_step_arcmin"]), 10) not in fd_filter:
            continue
        trace_indices = list(args.trace_indices) or list(range(int(cache["plus_rates"].shape[0])))
        if int(args.max_traces) > 0:
            trace_indices = trace_indices[: int(args.max_traces)]
        for catalog_mode in args.catalog_modes:
            include_self = str(catalog_mode) == "include_self"
            for trace_index in trace_indices:
                for true_label in THETA_LABELS:
                    rows.append(
                        _score_one_trial(
                            args=args,
                            cache=cache,
                            model=model,
                            readout=readout,
                            retina=retina,
                            geometry=geometry,
                            trace_index=int(trace_index),
                            true_label=true_label,
                            include_self=include_self,
                        )
                    )
                    if int(args.checkpoint_every) > 0 and len(rows) % int(args.checkpoint_every) == 0:
                        write_csv(trials_path, rows)
                        write_csv(summary_path, _summarize(rows))
    summary = _summarize(rows)
    write_csv(trials_path, rows)
    write_csv(summary_path, summary)
    write_json(
        out_dir / "end_to_end_pose_profile_manifest.json",
        {
            "args": vars(args),
            "geometry": asdict(geometry),
            "source_dir": Path(args.source_dir),
            "out_dir": out_dir,
            "n_trial_rows": len(rows),
            "n_summary_rows": len(summary),
            "stimulus_normalization": STIMULUS_NORMALIZATION,
            "interpretation": (
                "This profiles continuous eye pose directly through the differentiable Vernier retinal sampler, "
                "lag embedding, model core, and readout. Model/readout parameters are frozen; gradients flow to pose. "
                "The profile columns optimize pose separately under each Vernier label; the neutral columns estimate "
                "one shared trace from a sign-neutral stimulus before decoding plus/minus at that trace. "
                "When profile scores are compared across labels this is a continuous joint-MAP/profiling decoder "
                "over Vernier sign and eye trace, not a trajectory-catalog observer."
            ),
        },
    )
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--conditions", type=str, default="")
    parser.add_argument("--fd-steps-arcmin", type=str, default="")
    parser.add_argument("--catalog-modes", type=str, default="leave_one_out")
    parser.add_argument("--trace-indices", type=str, default="0")
    parser.add_argument("--max-traces", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-lags", type=int, default=N_LAGS)
    parser.add_argument("--max-timebins", type=int, default=0)
    parser.add_argument("--max-units", type=int, default=0)
    parser.add_argument("--spatial-collapse", choices=("max", "mean"), default="max")
    parser.add_argument("--likelihood-scale", type=float, default=64.0)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--process-cov-scale", type=float, default=1.0)
    parser.add_argument("--prior-process-model", choices=PRIOR_PROCESS_MODELS, default="brownian")
    parser.add_argument("--trajectory-prior-source", choices=TRAJECTORY_PRIOR_SOURCES, default="cache_train")
    parser.add_argument("--eye-traces", type=Path, default=DEFAULT_EYE_TRACES_PATH)
    parser.add_argument("--prior-center-mode", choices=PRIOR_CENTER_MODES, default="none")
    parser.add_argument("--max-prior-source-traces", type=int, default=0)
    parser.add_argument("--ou-max-abs-eigenvalue", type=float, default=0.999)
    parser.add_argument("--covariance-floor-arcmin2", type=float, default=1e-4)
    parser.add_argument("--flat-trace-tolerance-arcmin", type=float, default=0.25)
    parser.add_argument("--neutral-offset-arcmin", type=float, default=0.0)
    parser.add_argument("--start-modes", type=str, default="zero,mean_train,nearest_train,brownian")
    parser.add_argument("--n-brownian-starts", type=int, default=2)
    parser.add_argument("--optimizer", choices=("adam", "lbfgs"), default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument("--bar-width-arcmin", type=float, default=2.0)
    parser.add_argument("--gap-arcmin", type=float, default=4.0)
    parser.add_argument("--bar-length-arcmin", type=float, default=12.0)
    parser.add_argument("--contrast", type=float, default=0.5)
    parser.add_argument("--polarity", choices=("bright", "dark"), default="bright")
    parser.add_argument("--stimulus-orientation-deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-pose-estimates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.conditions = parse_csv_str(args.conditions)
    args.fd_steps_arcmin = parse_csv_float(args.fd_steps_arcmin)
    args.catalog_modes = parse_csv_str(args.catalog_modes)
    args.trace_indices = parse_csv_int(args.trace_indices)
    args.start_modes = parse_csv_str(args.start_modes)
    bad_starts = sorted(set(args.start_modes) - set(START_MODES))
    if bad_starts:
        raise ValueError(f"Unsupported --start-modes {bad_starts}; expected {START_MODES}")
    bad_catalogs = sorted(set(args.catalog_modes) - {"include_self", "leave_one_out"})
    if bad_catalogs:
        raise ValueError(f"Unsupported --catalog-modes {bad_catalogs}")
    if int(args.max_prior_source_traces) < 0:
        raise ValueError("--max-prior-source-traces must be nonnegative")
    rows, summary = run(args)
    print(f"Wrote {len(rows)} end-to-end pose-profile rows and {len(summary)} summaries", flush=True)


if __name__ == "__main__":
    main()
