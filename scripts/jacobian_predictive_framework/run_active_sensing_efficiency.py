#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import dill
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from VisionCore.paths import FIGURES_DIR, VISIONCORE_ROOT


ROOT = VISIONCORE_ROOT
SCRIPTS_DIR = ROOT / "scripts"
TD_DIR = SCRIPTS_DIR / "temporal_decoding"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(TD_DIR) not in sys.path:
    sys.path.insert(0, str(TD_DIR))


DEFAULT_CHECKPOINT_DIR = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints"
)
DEFAULT_MCFARLAND_OUTPUTS = ROOT / "scripts" / "mcfarland_outputs.pkl"
DEFAULT_OUT_BASE = ROOT / "outputs" / "jacobian_predictive_framework"
DEFAULT_EYE_TRACES = ROOT / "scripts" / "temporal_decoding" / "data" / "eye_traces.npz"
DEFAULT_TRAJECTORY_CONTROLS = (
    "real_FEM",
    "fixed_center",
    "stabilized",
    "random_amp",
    "random_cov",
    "scaled_FEM_0.5",
    "scaled_FEM_2.0",
)
DEFAULT_READOUTS = ("linear", "energy", "multinomial")
DEFAULT_FRAMES_PER_IM = (2, 6, 12, 30, 60)
DEFAULT_LOGMARS = (-0.30, -0.20, -0.10, 0.00, 0.10)
DEFAULT_ORIENTATIONS = (0, 90, 180, 270)
DEFAULT_SCALED_FEM_VALUES = (0.5, 2.0)
DEFAULT_OUT_SIZE = (151, 151)
DEFAULT_N_LAGS = 32
DEFAULT_RANDOM_SEED = 0
DEFAULT_RANDOM_CONTROL_REPEATS = 5
DEFAULT_HIRES_THRESHOLD = 0.35
DEFAULT_SIGMA_FRAMES = 3.0
DEFAULT_DTS = (1.0, 1.0 / 120.0)
DEFAULT_EOPTOTYPE_MIN_LOGMARS = 5
DEFAULT_EOPTOTYPE_MIN_ORIENTATION_PAIRS = 6
DEFAULT_EOPTOTYPE_MIN_TRIALS_PER_COND_PAIR = 10
DEFAULT_RANDOM_CONTROL_PATH_ERROR_MAX = 0.25
DEFAULT_RANDOM_CONTROL_ACF_ERROR_MAX = 0.25
DEFAULT_RANDOM_CONTROL_RMS_ERROR_MAX = 0.15
DEFAULT_RANDOM_CONTROL_COV_ERROR_MAX = 0.20
DEFAULT_REQUIRED_EOPTOTYPE_PAIRS = ("0_vs_90", "0_vs_180", "0_vs_270", "90_vs_180", "90_vs_270", "180_vs_270")
DEFAULT_EOPTOTYPE_FEATURE_REPRESENTATIONS = (
    "spatial_avg_time_mean",
    "spatial_avg_energy",
    "map_energy",
    "map_lowrank_pca",
)
DEFAULT_D1_INTEGRATION_WINDOWS = (1, 6, 12, 24, 48, 60)
DEFAULT_RECON_PRIMARY_LOGMARS = (-0.20, -0.25, -0.30, -0.35)
DEFAULT_RECON_SATURATION_LOGMAR = -0.40
EPS = 1e-8


@dataclass(frozen=True)
class ModelBundle:
    model: object
    readout: torch.nn.Module
    session_name: str


@dataclass(frozen=True)
class SanityResult:
    name: str
    status: str
    value: float | str
    threshold: str
    detail: str


def _parse_csv_floats(values: Iterable[str] | str) -> tuple[float, ...]:
    if isinstance(values, str):
        parts = [part.strip() for part in values.split(",") if part.strip()]
        return tuple(float(part) for part in parts)
    return tuple(float(value) for value in values)


def _parse_csv_ints(values: Iterable[str] | str) -> tuple[int, ...]:
    if isinstance(values, str):
        parts = [part.strip() for part in values.split(",") if part.strip()]
        return tuple(int(float(part)) for part in parts)
    return tuple(int(value) for value in values)


def _pick_device(device: str) -> str:
    if (device or "auto").lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text))


def _condition_label(condition: str) -> str:
    mapping = {
        "real_FEM": "real",
        "fixed_center": "fixed_center",
        "stabilized": "stabilized",
        "random_amp": "random_amp",
        "random_cov": "random_cov",
    }
    return mapping.get(condition, condition.replace("scaled_FEM_", "scaled_"))


def _is_randomized_condition(condition_label: str) -> bool:
    return condition_label in {"random_amp", "random_cov"}


def _format_logmar(logmar: float) -> str:
    return f"{float(logmar):.2f}".replace("+", "")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if rows:
            ordered: list[str] = []
            seen: set[str] = set()
            for row in rows:
                for key in row.keys():
                    key_str = str(key)
                    if key_str not in seen:
                        ordered.append(key_str)
                        seen.add(key_str)
            fieldnames = ordered
        else:
            fieldnames = []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(values.size))


def _nanmedian_or_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmedian(arr))


def _nanmean_or_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _nanpercentile_or_nan(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanpercentile(arr, q))


def _aggregate_rows_mean(
    rows: list[dict[str, object]],
    group_keys: tuple[str, ...],
    numeric_average_keys: Iterable[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    out_rows: list[dict[str, object]] = []
    numeric_average_keys = tuple(numeric_average_keys)
    for _, group in grouped.items():
        template = dict(group[0])
        for key in numeric_average_keys:
            if key in template:
                template[key] = _nanmean_or_nan(_safe_float(item.get(key)) for item in group)
        if "random_repeat" in template:
            template["random_repeat"] = -1
        template["n_aggregated_repeats"] = len(group)
        out_rows.append(template)
    return out_rows


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value, value
    samples = np.empty(n_boot, dtype=np.float64)
    for boot_idx in range(n_boot):
        draw = rng.choice(values, size=values.size, replace=True)
        samples[boot_idx] = np.mean(draw)
    return float(np.mean(values)), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return float("nan"), float("nan"), float("nan")
    d = a[valid] - b[valid]
    return _bootstrap_ci(d, rng=rng, n_boot=n_boot)


def _gaussian_kernel1d(sigma_frames: float, radius: int | None = None) -> np.ndarray:
    sigma_frames = max(float(sigma_frames), 1e-3)
    if radius is None:
        radius = max(1, int(round(3.0 * sigma_frames)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (xs / sigma_frames) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def _smooth_trace(trace: np.ndarray, sigma_frames: float) -> np.ndarray:
    kernel = _gaussian_kernel1d(sigma_frames)
    out = np.zeros_like(trace, dtype=np.float64)
    for dim in range(trace.shape[1]):
        out[:, dim] = np.convolve(trace[:, dim], kernel, mode="same")
    return out


def _acf(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] <= lag:
        return float("nan")
    x0 = values[:-lag]
    x1 = values[lag:]
    valid = np.isfinite(x0) & np.isfinite(x1)
    if np.sum(valid) < 3:
        return float("nan")
    x0 = x0[valid] - np.mean(x0[valid])
    x1 = x1[valid] - np.mean(x1[valid])
    denom = np.linalg.norm(x0) * np.linalg.norm(x1)
    if denom <= EPS:
        return float("nan")
    return float(np.dot(x0, x1) / denom)


def _trace_metrics_deg(trace: np.ndarray) -> dict[str, float]:
    trace = np.asarray(trace, dtype=np.float64)
    centered = trace - np.mean(trace, axis=0, keepdims=True)
    radii = np.linalg.norm(centered, axis=1)
    steps = np.diff(trace, axis=0) if trace.shape[0] > 1 else np.zeros((0, 2), dtype=np.float64)
    cov = np.cov(centered.T) if trace.shape[0] > 1 else np.zeros((2, 2), dtype=np.float64)
    path_length = float(np.sum(np.linalg.norm(steps, axis=1))) if steps.size else 0.0
    return {
        "mean_eye_x": float(np.mean(trace[:, 0])) if trace.size else float("nan"),
        "mean_eye_y": float(np.mean(trace[:, 1])) if trace.size else float("nan"),
        "eye_rms_deg": float(np.sqrt(np.mean(np.sum(centered * centered, axis=1)))) if trace.size else float("nan"),
        "eye_path_length_deg": path_length,
        "eye_cov_xx": float(cov[0, 0]) if cov.shape == (2, 2) else float("nan"),
        "eye_cov_xy": float(cov[0, 1]) if cov.shape == (2, 2) else float("nan"),
        "eye_cov_yy": float(cov[1, 1]) if cov.shape == (2, 2) else float("nan"),
        "acf_lag1": _acf(centered[:, 0], 1),
        "acf_lag2": _acf(centered[:, 0], 2),
        "acf_lag4": _acf(centered[:, 0], 4),
    }


def _match_random_amp(trace: np.ndarray, rng: np.random.Generator, sigma_frames: float) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    center = np.mean(trace, axis=0, keepdims=True)
    centered = trace - center
    raw = rng.standard_normal(size=centered.shape)
    smooth = _smooth_trace(raw, sigma_frames=sigma_frames)
    smooth -= np.mean(smooth, axis=0, keepdims=True)
    target_rms = np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
    current_rms = np.sqrt(np.mean(np.sum(smooth * smooth, axis=1)))
    if current_rms > EPS:
        smooth *= target_rms / current_rms
    return smooth + center


def _match_random_cov(trace: np.ndarray, rng: np.random.Generator, sigma_frames: float) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    center = np.mean(trace, axis=0, keepdims=True)
    centered = trace - center
    raw = rng.standard_normal(size=centered.shape)
    smooth = _smooth_trace(raw, sigma_frames=sigma_frames)
    smooth -= np.mean(smooth, axis=0, keepdims=True)
    cov_target = np.cov(centered.T) + 1e-6 * np.eye(2)
    cov_source = np.cov(smooth.T) + 1e-6 * np.eye(2)
    evals_t, evecs_t = np.linalg.eigh(cov_target)
    evals_s, evecs_s = np.linalg.eigh(cov_source)
    whitening = evecs_s @ np.diag(1.0 / np.sqrt(np.maximum(evals_s, 1e-6))) @ evecs_s.T
    coloring = evecs_t @ np.diag(np.sqrt(np.maximum(evals_t, 1e-6))) @ evecs_t.T
    recolored = smooth @ whitening.T @ coloring.T
    return recolored + center


def _trajectory_for_condition(
    trace: np.ndarray,
    condition: str,
    rng: np.random.Generator,
    sigma_frames: float,
) -> tuple[np.ndarray, int | None]:
    trace = np.asarray(trace, dtype=np.float64)
    center = np.mean(trace, axis=0, keepdims=True)
    centered = trace - center
    if condition == "real_FEM":
        return trace.copy(), None
    if condition == "fixed_center":
        return np.zeros_like(trace), None
    if condition == "stabilized":
        return np.repeat(center, trace.shape[0], axis=0), None
    if condition == "random_amp":
        return _match_random_amp(trace, rng=rng, sigma_frames=sigma_frames), None
    if condition == "random_cov":
        return _match_random_cov(trace, rng=rng, sigma_frames=sigma_frames), None
    if condition.startswith("scaled_FEM_"):
        scale = float(condition.split("scaled_FEM_", 1)[1])
        return center + centered * scale, None
    raise ValueError(f"Unsupported condition: {condition}")


def _control_repeats(condition: str, n_random_controls: int) -> int:
    if condition in {"random_amp", "random_cov"}:
        return int(n_random_controls)
    return 1


def _crop_counterfactual_to_T(stim: torch.Tensor, T: int) -> torch.Tensor:
    t0 = int(stim.shape[0])
    if t0 < T:
        raise ValueError(f"Stimulus shorter than requested T: {t0} < {T}")
    if t0 >= T + 1:
        return stim[1 : 1 + T]
    return stim[:T]


def _population_spike_metrics(y: torch.Tensor, dt: float) -> dict[str, float | np.ndarray]:
    from scripts.spatial_info import spatial_ssi_population

    y = torch.clamp(y, min=0.0)
    ispike_t, irate_t, I_tn = spatial_ssi_population(y, dt=dt, spike_weighted=True)
    T, N, H, W = y.shape
    r = y.reshape(T, N, H * W)
    rbar = r.mean(dim=2)
    spikes_tn = rbar * dt
    bits_t = (spikes_tn * I_tn).sum(dim=1)
    spikes_t = spikes_tn.sum(dim=1)
    total_bits = float(torch.sum(bits_t).item())
    total_spikes = float(torch.sum(spikes_t).item())
    cumulative_bits_per_expected_spike = total_bits / max(total_spikes, EPS)
    return {
        "mean_spatial_bits_per_expected_spike": float(torch.mean(ispike_t).item()),
        "median_spatial_bits_per_expected_spike": float(torch.median(ispike_t).item()),
        "cumulative_spatial_bits_per_expected_spike": float(cumulative_bits_per_expected_spike),
        "mean_bits_per_sec": float(torch.mean(irate_t).item()),
        "total_bits": total_bits,
        "mean_expected_spikes_per_bin": float(torch.mean(spikes_t).item()),
        "total_expected_spikes": total_spikes,
        "ispike_t": ispike_t.detach().cpu().numpy(),
        "irate_t": irate_t.detach().cpu().numpy(),
    }


def _confusion_mi_bits(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    classes_true = np.unique(y_true)
    classes_pred = np.unique(y_pred)
    table = np.zeros((classes_true.size, classes_pred.size), dtype=np.float64)
    true_index = {value: idx for idx, value in enumerate(classes_true)}
    pred_index = {value: idx for idx, value in enumerate(classes_pred)}
    for truth, pred in zip(y_true, y_pred, strict=False):
        table[true_index[int(truth)], pred_index[int(pred)]] += 1.0
    table /= np.sum(table)
    p_true = np.sum(table, axis=1, keepdims=True)
    p_pred = np.sum(table, axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(table, p_true @ p_pred, out=np.ones_like(table), where=(table > 0) & (p_true @ p_pred > 0))
        contrib = np.where(table > 0, table * np.log2(ratio), 0.0)
    return float(np.sum(contrib))


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    classes = np.unique(y_true)
    recalls = []
    for cls in classes:
        mask = y_true == cls
        if np.sum(mask) == 0:
            continue
        recalls.append(np.mean(y_pred[mask] == cls))
    return float(np.mean(recalls)) if recalls else float("nan")


def _load_model_bundle(args: argparse.Namespace) -> ModelBundle:
    from eval.eval_stack_multidataset import load_model
    from scripts.spatial_info import get_spatial_readout
    from scripts.utils import get_model_and_dataset_configs

    try:
        model, _model_info = load_model(
            model_type=args.model_type,
            model_index=args.model_index,
            checkpoint_dir=str(args.checkpoint_dir),
            cfg_dir_override="experiments/dataset_configs/multi_basic_120_long_legacy.yaml",
            device="cpu",
        )
    except Exception:
        if (
            str(args.model_type) == "resnet_none_convgru"
            and int(args.model_index) == 0
            and Path(args.checkpoint_dir) == DEFAULT_CHECKPOINT_DIR
        ):
            model, _dataset_configs = get_model_and_dataset_configs(mode="standard")
        else:
            raise
    model.model.eval()
    if hasattr(model.model, "convnet"):
        setattr(model.model.convnet, "use_checkpointing", True)
    model = model.to(args.device)

    with Path(args.mcfarland_outputs).open("rb") as handle:
        outputs = dill.load(handle)
    readout = get_spatial_readout(model, outputs).to(args.device).eval()
    session_name = str(model.names[int(args.dataset_idx)])
    return ModelBundle(model=model, readout=readout, session_name=session_name)


def _extract_fixrsvp_eye_traces(model_bundle: ModelBundle, args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    from eval.eval_stack_multidataset import load_single_dataset

    model = model_bundle.model
    train_data, val_data, _dataset_config = load_single_dataset(model, int(args.dataset_idx))
    inds = torch.concatenate([
        train_data.get_dataset_inds("fixrsvp"),
        val_data.get_dataset_inds("fixrsvp"),
    ], dim=0)
    dataset = train_data.shallow_copy()
    dataset.inds = inds

    dset_idx = int(inds[:, 0].unique().item())
    covs = dataset.dsets[dset_idx].covariates
    trial_inds = covs["trial_inds"].numpy().ravel()
    eye_source = dataset.dsets[dset_idx]["eyepos"]
    eye_all = eye_source.detach().cpu().numpy() if isinstance(eye_source, torch.Tensor) else np.asarray(eye_source)
    fixation = np.hypot(eye_all[:, 0], eye_all[:, 1]) < 1.0

    rows: list[dict[str, object]] = []
    for trial_id in np.unique(trial_inds):
        ix = (trial_inds == trial_id) & fixation
        if not np.any(ix):
            continue
        trace = eye_all[ix].astype(np.float32)
        fix_dur = int(trace.shape[0])
        if fix_dur < int(args.min_fix_dur):
            continue
        metrics = _trace_metrics_deg(trace)
        rows.append(
            {
                "trial_index": int(trial_id),
                "fix_dur": fix_dur,
                "n_valid_frames": fix_dur,
                "trace": trace,
                **metrics,
            }
        )

    rows.sort(key=lambda row: (-_safe_int(row["fix_dur"]), _safe_int(row["trial_index"])))
    if args.max_trials is not None:
        rows = rows[: int(args.max_trials)]

    summary = {
        "n_trials": len(rows),
        "n_valid_frames_mean": float(np.mean([_safe_int(row["n_valid_frames"]) for row in rows])) if rows else float("nan"),
        "session": model_bundle.session_name,
        "dataset_idx": int(args.dataset_idx),
    }
    return rows, summary


def _build_fixrsvp_stimulus(frames_per_im: int, n_frames: int) -> np.ndarray:
    from scripts.spatial_info import make_stimulus_stack

    stack = make_stimulus_stack(type="fixrsvp", frame=None, frames_per_im=int(frames_per_im), num_frames=int(n_frames))
    if stack.shape[0] >= n_frames:
        return stack[:n_frames]
    reps = int(math.ceil(n_frames / stack.shape[0]))
    return np.tile(stack, (reps, 1, 1))[:n_frames]


def _compute_rate_maps(model_bundle: ModelBundle, stim: torch.Tensor, clamp_negative: bool = True) -> tuple[torch.Tensor, bool]:
    from scripts.spatial_info import compute_rate_map_batched

    y = compute_rate_map_batched(model_bundle.model, model_bundle.readout, stim, batch_size=16)
    if not torch.isfinite(y).all():
        raise ValueError("Nonfinite model outputs encountered in rate-map computation")
    had_negative = bool(torch.any(y < 0).item())
    if clamp_negative:
        y = torch.clamp(y, min=0.0)
    return y, had_negative


def _resolve_trajectory_controls(args: argparse.Namespace) -> tuple[str, ...]:
    requested = [str(control) for control in args.trajectory_controls]
    non_scaled = [control for control in requested if not control.startswith("scaled_FEM_")]
    scaled_requested = [control for control in requested if control.startswith("scaled_FEM_")]
    if scaled_requested:
        scaled_controls = [f"scaled_FEM_{float(value):g}" for value in args.scaled_fem_values]
        requested = non_scaled + scaled_controls
    deduped: list[str] = []
    seen: set[str] = set()
    for control in requested:
        if control not in seen:
            deduped.append(control)
            seen.add(control)
    return tuple(deduped)


def _trajectory_control_qc_rows(
    fixrsvp_trials: list[dict[str, object]],
    conditions: tuple[str, ...],
    n_random_controls: int,
    sigma_frames: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], dict[tuple[int, str, int], np.ndarray]]:
    rows: list[dict[str, object]] = []
    trace_cache: dict[tuple[int, str, int], np.ndarray] = {}
    for trial in fixrsvp_trials:
        trace = np.asarray(trial["trace"], dtype=np.float64)
        base_metrics = _trace_metrics_deg(trace)
        for condition in conditions:
            repeats = _control_repeats(condition, n_random_controls)
            for repeat in range(repeats):
                conditioned_trace, _ = _trajectory_for_condition(trace, condition, rng=rng, sigma_frames=sigma_frames)
                conditioned_metrics = _trace_metrics_deg(conditioned_trace)
                matched_rms_error = abs(conditioned_metrics["eye_rms_deg"] - base_metrics["eye_rms_deg"]) / max(base_metrics["eye_rms_deg"], EPS)
                cov_base = np.array(
                    [[base_metrics["eye_cov_xx"], base_metrics["eye_cov_xy"]], [base_metrics["eye_cov_xy"], base_metrics["eye_cov_yy"]]],
                    dtype=np.float64,
                )
                cov_conditioned = np.array(
                    [[conditioned_metrics["eye_cov_xx"], conditioned_metrics["eye_cov_xy"]], [conditioned_metrics["eye_cov_xy"], conditioned_metrics["eye_cov_yy"]]],
                    dtype=np.float64,
                )
                matched_cov_error = float(np.linalg.norm(cov_conditioned - cov_base) / max(np.linalg.norm(cov_base), EPS))
                path_length_error = abs(
                    conditioned_metrics["eye_path_length_deg"] - base_metrics["eye_path_length_deg"]
                ) / max(abs(base_metrics["eye_path_length_deg"]), EPS)
                acf_lag1_error = abs(conditioned_metrics["acf_lag1"] - base_metrics["acf_lag1"]) / max(abs(base_metrics["acf_lag1"]), EPS) if np.isfinite(conditioned_metrics["acf_lag1"]) and np.isfinite(base_metrics["acf_lag1"]) else float("nan")
                acf_lag2_error = abs(conditioned_metrics["acf_lag2"] - base_metrics["acf_lag2"]) / max(abs(base_metrics["acf_lag2"]), EPS) if np.isfinite(conditioned_metrics["acf_lag2"]) and np.isfinite(base_metrics["acf_lag2"]) else float("nan")
                acf_lag4_error = abs(conditioned_metrics["acf_lag4"] - base_metrics["acf_lag4"]) / max(abs(base_metrics["acf_lag4"]), EPS) if np.isfinite(conditioned_metrics["acf_lag4"]) and np.isfinite(base_metrics["acf_lag4"]) else float("nan")
                rows.append(
                    {
                        "trial_index": _safe_int(trial["trial_index"]),
                        "condition": condition,
                        "random_repeat": int(repeat),
                        **conditioned_metrics,
                        "matched_rms_error": float(matched_rms_error),
                        "matched_cov_error": float(matched_cov_error),
                        "path_length_error": float(path_length_error),
                        "acf_lag1_error": float(acf_lag1_error),
                        "acf_lag2_error": float(acf_lag2_error),
                        "acf_lag4_error": float(acf_lag4_error),
                    }
                )
                trace_cache[(_safe_int(trial["trial_index"]), condition, int(repeat))] = conditioned_trace.astype(np.float32)
    return rows, trace_cache


def _verify_dt_convention(
    model_bundle: ModelBundle,
    fixrsvp_trials: list[dict[str, object]],
    trace_cache: dict[tuple[int, str, int], np.ndarray],
    out_dir: Path,
    frames_per_im: int,
    n_lags: int,
    out_size: tuple[int, int],
) -> tuple[list[dict[str, object]], float, str]:
    from scripts.spatial_info import make_counterfactual_stim

    if not fixrsvp_trials:
        return [], 1.0 / 120.0, "no_fixrsvp_trials"

    trial = fixrsvp_trials[0]
    trace = trace_cache[(_safe_int(trial["trial_index"]), "real_FEM", 0)]
    T = trace.shape[0]
    full_stack = _build_fixrsvp_stimulus(frames_per_im=frames_per_im, n_frames=T + n_lags + 1)
    stim = make_counterfactual_stim(
        full_stack,
        torch.from_numpy(trace).float(),
        n_lags=n_lags,
        out_size=out_size,
    )
    stim = _crop_counterfactual_to_T(stim, T) / 127.0
    y, _ = _compute_rate_maps(model_bundle, stim)

    rows: list[dict[str, object]] = []
    plausibility: list[tuple[float, str]] = []
    mean_output = float(torch.mean(y).item())
    mean_rate_map = torch.mean(y.reshape(y.shape[0], y.shape[1], -1), dim=2)
    for candidate_dt in DEFAULT_DTS:
        spikes = float(torch.sum(mean_rate_map * float(candidate_dt)).item())
        mean_rate_hz = float(torch.mean(mean_rate_map).item() / max(float(candidate_dt), EPS))
        if 0.1 <= mean_rate_hz <= 200.0:
            status = "plausible"
        else:
            status = "implausible"
        rows.append(
            {
                "candidate_dt": float(candidate_dt),
                "mean_model_output": mean_output,
                "total_expected_spikes": spikes,
                "mean_expected_rate_hz_if_dt_used": mean_rate_hz,
                "plausibility_status": status,
                "selected_dt": "",
                "selection_reason": "",
            }
        )
        plausibility.append((float(candidate_dt), status))

    selected_dt = 1.0 / 120.0
    selection_reason = "default_120hz"
    plausible = [dt for dt, status in plausibility if status == "plausible"]
    if 1.0 / 120.0 in plausible:
        selected_dt = 1.0 / 120.0
        selection_reason = "matches_model_frame_rate"
    elif plausible:
        selected_dt = plausible[0]
        selection_reason = "first_plausible_candidate"
    for row in rows:
        row["selected_dt"] = float(selected_dt)
        row["selection_reason"] = selection_reason
    _write_csv(out_dir / "dt_convention_check.csv", rows)
    return rows, float(selected_dt), selection_reason


def _fixrsvp_trial_metric(
    model_bundle: ModelBundle,
    trial: dict[str, object],
    trace: np.ndarray,
    frames_per_im: int,
    dt: float,
    n_lags: int,
    out_size: tuple[int, int],
) -> tuple[dict[str, object], np.ndarray]:
    from scripts.spatial_info import make_counterfactual_stim

    T = int(trace.shape[0])
    full_stack = _build_fixrsvp_stimulus(frames_per_im=frames_per_im, n_frames=T + n_lags + 1)
    stim = make_counterfactual_stim(
        full_stack,
        torch.from_numpy(trace).float(),
        n_lags=n_lags,
        out_size=out_size,
    )
    stim = _crop_counterfactual_to_T(stim, T) / 127.0
    y, clamped_negative = _compute_rate_maps(model_bundle, stim)
    metrics = _population_spike_metrics(y, dt=dt)
    row = {
        "trial_index": _safe_int(trial["trial_index"]),
        "n_time_bins": int(T),
        "n_units": int(y.shape[1]),
        "clamped_negative_rates": bool(clamped_negative),
        "status": "ok",
        **{key: value for key, value in trial.items() if key != "trace"},
        **{key: value for key, value in metrics.items() if key not in {"ispike_t", "irate_t"}},
    }
    return row, y.reshape(y.shape[0], y.shape[1], -1).mean(dim=2).detach().cpu().numpy()


def _pairwise_orientation_pairs(orientations: tuple[int, ...]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for idx, ori_a in enumerate(orientations):
        for ori_b in orientations[idx + 1 :]:
            pairs.append((int(ori_a), int(ori_b)))
    return pairs


def _build_hires_eoptotype_stim(
    orientation_deg: int,
    logmar: float,
    trace: np.ndarray,
    n_lags: int,
    renderer: torch.nn.Module,
    retina: torch.nn.Module,
) -> torch.Tensor:
    from scripts.spatial_info import embed_time_lags
    with torch.no_grad():
        world = renderer(float(orientation_deg), float(logmar))
        world_gray = 127.0 * (1.0 - world)
        template = getattr(renderer, "template", None)
        device = template.device if isinstance(template, torch.Tensor) else torch.device("cpu")
        ep = torch.from_numpy(trace).float().to(device=device)
        ep_padded = torch.cat([ep[:1].expand(n_lags, -1), ep], dim=0)
        movie = retina(world_gray, ep_padded)[0, 0].cpu()
    stim = embed_time_lags(movie, n_lags=n_lags) / 127.0
    return stim


def _eoptotype_feature_representations(rate_map: np.ndarray) -> dict[str, np.ndarray]:
    time_mean = np.mean(rate_map, axis=0)
    late_mean = np.mean(rate_map[max(rate_map.shape[0] // 2, 0) :, :], axis=0)
    time_concat = rate_map.reshape(-1)
    energy = np.mean(rate_map ** 2, axis=0)
    return {
        "time_mean": time_mean.astype(np.float32),
        "late_mean": late_mean.astype(np.float32),
        "time_concat": time_concat.astype(np.float32),
        "energy": energy.astype(np.float32),
    }


def _decode_pairwise(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    readout_type: str,
    rng_seed: int,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    unique_groups = np.unique(groups)
    n_splits = max(2, min(5, unique_groups.size))
    splitter = GroupKFold(n_splits=n_splits)
    preds = np.full_like(y, fill_value=-1)
    probabilities = np.full((y.size, len(np.unique(y))), np.nan, dtype=np.float64)
    fold_scores: list[float] = []

    C = 1.0
    for train_idx, test_idx in splitter.split(X, y, groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(
            C=C,
            max_iter=2000,
            solver="lbfgs",
            random_state=rng_seed,
        )
        clf.fit(X_train, y[train_idx])
        preds[test_idx] = clf.predict(X_test)
        probabilities[test_idx] = clf.predict_proba(X_test)
        fold_scores.append(float(clf.score(X_test, y[test_idx])))

    accuracy = float(np.mean(preds == y))
    balanced_accuracy = _balanced_accuracy(y, preds)
    mi_bits = _confusion_mi_bits(y, preds)
    return accuracy, balanced_accuracy, mi_bits, preds, probabilities


def _build_eoptotype_decoder_row(
    task: tuple[str, float, int, int, str, str],
    grouped_rows: dict[tuple[str, float, int], list[dict[str, object]]],
    aggregated_feature_store: dict[str, np.ndarray],
    random_seed: int,
    renderer_status_value: str,
) -> tuple[dict[str, object] | None, int]:
    condition, logmar, ori_a, ori_b, readout_type, feature_name = task
    rows_a = grouped_rows.get((condition, float(logmar), int(ori_a)), [])
    rows_b = grouped_rows.get((condition, float(logmar), int(ori_b)), [])
    trial_ids_a = {_safe_int(row["trial_index"]) for row in rows_a}
    trial_ids_b = {_safe_int(row["trial_index"]) for row in rows_b}
    common_trial_ids = sorted(trial_ids_a & trial_ids_b)
    n = len(common_trial_ids)
    if n < 4:
        return None, 1

    row_a_by_trial = {_safe_int(row["trial_index"]): row for row in rows_a}
    row_b_by_trial = {_safe_int(row["trial_index"]): row for row in rows_b}
    rows_a = [row_a_by_trial[trial_id] for trial_id in common_trial_ids]
    rows_b = [row_b_by_trial[trial_id] for trial_id in common_trial_ids]
    Xa = np.stack([aggregated_feature_store[str(row["feature_key"]) + f"_{feature_name}"] for row in rows_a], axis=0)
    Xb = np.stack([aggregated_feature_store[str(row["feature_key"]) + f"_{feature_name}"] for row in rows_b], axis=0)
    if readout_type == "energy":
        Xa = np.sqrt(np.maximum(Xa, 0.0))
        Xb = np.sqrt(np.maximum(Xb, 0.0))
    X = np.concatenate([Xa, Xb], axis=0)
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)], axis=0)
    groups = np.concatenate([np.asarray(common_trial_ids, dtype=np.int64), np.asarray(common_trial_ids, dtype=np.int64)], axis=0)
    accuracy, balanced_accuracy, mi_bits, preds, _proba = _decode_pairwise(
        X=X,
        y=y,
        groups=groups,
        readout_type=readout_type,
        rng_seed=int(random_seed),
    )
    spikes = np.asarray(
        [_safe_float(row["total_expected_spikes"]) for row in rows_a] + [_safe_float(row["total_expected_spikes"]) for row in rows_b],
        dtype=np.float64,
    )
    rA = np.mean(Xa, axis=0)
    rB = np.mean(Xb, axis=0)
    d2 = _poisson_d2(rA, rB)
    mean_total_expected_spikes = float(np.mean(spikes))
    d2_per_expected_spike = float(d2 / max(mean_total_expected_spikes, EPS))
    delta_theta_rad = float(np.deg2rad(abs(int(ori_b) - int(ori_a))))
    if delta_theta_rad <= 0:
        poisson_fi_per_expected_spike = float("nan")
    else:
        poisson_fi_per_expected_spike = float((d2 / max(delta_theta_rad * delta_theta_rad, EPS)) / max(mean_total_expected_spikes, EPS))

    budget_spikes = float(np.mean(spikes))
    poisson_X = _poisson_resample_features_at_budget(X, budget_spikes=budget_spikes, rng_seed=int(random_seed) + 1000 + n)
    poisson_acc, poisson_balanced_acc, poisson_mi_bits, _poisson_preds, _poisson_proba = _decode_pairwise(
        X=poisson_X,
        y=y,
        groups=groups,
        readout_type=readout_type,
        rng_seed=int(random_seed) + 1001,
    )

    rate_normalized_X = X / np.maximum(spikes[:, None], EPS)
    rate_normalized_acc, _, _, _, _ = _decode_pairwise(
        X=rate_normalized_X,
        y=y,
        groups=groups,
        readout_type=readout_type,
        rng_seed=int(random_seed),
    )
    row = {
        "condition": condition,
        "logmar": float(logmar),
        "orientation_pair": f"{ori_a}_vs_{ori_b}",
        "readout_type": readout_type,
        "feature_representation": feature_name,
        "n_train": int((n * 2) - max(2, (n * 2) // min(5, max(2, n)))),
        "n_test": int(max(2, (n * 2) // min(5, max(2, n)))),
        "n_splits": int(max(2, min(5, n))),
        "poisson_d2": d2,
        "d2_per_expected_spike": d2_per_expected_spike,
        "poisson_fi_per_expected_spike": poisson_fi_per_expected_spike,
        "mean_pair_spatial_bits_per_expected_spike": _nanmean_or_nan([_safe_float(r.get("cumulative_spatial_bits_per_expected_spike")) for r in rows_a + rows_b]),
        "poisson_budget_accuracy": poisson_acc,
        "poisson_budget_balanced_accuracy": poisson_balanced_acc,
        "poisson_budget_confusion_mi_bits": poisson_mi_bits,
        "rate_normalized_decoder_accuracy": rate_normalized_acc,
        "deterministic_decoder_accuracy_qc": accuracy,
        "deterministic_decoder_balanced_accuracy_qc": balanced_accuracy,
        "deterministic_confusion_mi_bits_qc": mi_bits,
        "mean_total_expected_spikes": mean_total_expected_spikes,
        "real_minus_fixed_d2_per_expected_spike": float("nan"),
        "real_minus_stabilized_d2_per_expected_spike": float("nan"),
        "real_minus_random_amp_d2_per_expected_spike": float("nan"),
        "real_minus_random_cov_d2_per_expected_spike": float("nan"),
        "real_minus_fixed_poisson_budget_accuracy": float("nan"),
        "real_minus_stabilized_poisson_budget_accuracy": float("nan"),
        "real_minus_random_amp_poisson_budget_accuracy": float("nan"),
        "real_minus_random_cov_poisson_budget_accuracy": float("nan"),
        "real_minus_fixed_rate_normalized_decoder_accuracy": float("nan"),
        "real_minus_stabilized_rate_normalized_decoder_accuracy": float("nan"),
        "real_minus_random_amp_rate_normalized_decoder_accuracy": float("nan"),
        "real_minus_random_cov_rate_normalized_decoder_accuracy": float("nan"),
        "decision_status": "ok",
        "legacy_alignment_capture_status": "not_compared_same_stimuli_traces",
        "renderer_normalization_status": renderer_status_value,
    }
    return row, 0


def _random_control_path_acf_qc_pass(trace_qc_rows: list[dict[str, object]]) -> bool:
    for condition in ("random_amp", "random_cov"):
        cond_rows = [row for row in trace_qc_rows if str(row.get("condition")) == condition]
        if not cond_rows:
            return False
        path_med = _nanmedian_or_nan(_safe_float(row.get("path_length_error")) for row in cond_rows)
        acf1_med = _nanmedian_or_nan(_safe_float(row.get("acf_lag1_error")) for row in cond_rows)
        acf2_med = _nanmedian_or_nan(_safe_float(row.get("acf_lag2_error")) for row in cond_rows)
        acf4_med = _nanmedian_or_nan(_safe_float(row.get("acf_lag4_error")) for row in cond_rows)
        if not (
            np.isfinite(path_med)
            and path_med <= DEFAULT_RANDOM_CONTROL_PATH_ERROR_MAX
            and np.isfinite(acf1_med)
            and acf1_med <= DEFAULT_RANDOM_CONTROL_ACF_ERROR_MAX
            and np.isfinite(acf2_med)
            and acf2_med <= DEFAULT_RANDOM_CONTROL_ACF_ERROR_MAX
            and np.isfinite(acf4_med)
            and acf4_med <= DEFAULT_RANDOM_CONTROL_ACF_ERROR_MAX
        ):
            return False
    return True


def _compute_d1_window_decoder_rows(
    aggregated_trial_rows: list[dict[str, object]],
    aggregated_feature_store: dict[str, np.ndarray],
    conditions_for_decoder: tuple[str, ...],
    logmar_values: tuple[float, ...],
    orientation_pairs: list[tuple[int, int]],
    d1_windows: tuple[int, ...],
    random_seed: int,
) -> list[dict[str, object]]:
    grouped_rows: dict[tuple[str, float, int], list[dict[str, object]]] = defaultdict(list)
    for row in aggregated_trial_rows:
        grouped_rows[(str(row["condition"]), _safe_float(row["logmar"]), _safe_int(row["orientation"]))].append(row)

    d1_rows: list[dict[str, object]] = []
    for condition in conditions_for_decoder:
        for logmar in logmar_values:
            for ori_a, ori_b in orientation_pairs:
                rows_a = grouped_rows.get((condition, float(logmar), int(ori_a)), [])
                rows_b = grouped_rows.get((condition, float(logmar), int(ori_b)), [])
                trial_ids_a = {_safe_int(row["trial_index"]) for row in rows_a}
                trial_ids_b = {_safe_int(row["trial_index"]) for row in rows_b}
                common_trial_ids = sorted(trial_ids_a & trial_ids_b)
                n = len(common_trial_ids)
                if n < 4:
                    continue
                row_a_by_trial = {_safe_int(row["trial_index"]): row for row in rows_a}
                row_b_by_trial = {_safe_int(row["trial_index"]): row for row in rows_b}
                rows_a = [row_a_by_trial[idx] for idx in common_trial_ids]
                rows_b = [row_b_by_trial[idx] for idx in common_trial_ids]
                spikes = np.asarray(
                    [_safe_float(row["total_expected_spikes"]) for row in rows_a] + [_safe_float(row["total_expected_spikes"]) for row in rows_b],
                    dtype=np.float64,
                )
                mean_total_expected_spikes = float(np.mean(spikes))

                for window in d1_windows:
                    feature_name = f"d1_time_mean_w{int(window)}"
                    Xa = np.stack([aggregated_feature_store[str(row["feature_key"]) + f"_{feature_name}"] for row in rows_a], axis=0)
                    Xb = np.stack([aggregated_feature_store[str(row["feature_key"]) + f"_{feature_name}"] for row in rows_b], axis=0)
                    X = np.concatenate([Xa, Xb], axis=0)
                    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)], axis=0)
                    groups = np.concatenate(
                        [np.asarray(common_trial_ids, dtype=np.int64), np.asarray(common_trial_ids, dtype=np.int64)],
                        axis=0,
                    )
                    accuracy, balanced_accuracy, mi_bits, _preds, _proba = _decode_pairwise(
                        X=X,
                        y=y,
                        groups=groups,
                        readout_type="linear",
                        rng_seed=int(random_seed),
                    )
                    d1_rows.append(
                        {
                            "condition": condition,
                            "logmar": float(logmar),
                            "orientation_pair": f"{ori_a}_vs_{ori_b}",
                            "integration_window": int(window),
                            "readout_type": "linear",
                            "feature_representation": feature_name,
                            "d1_time_mean_accuracy": accuracy,
                            "d1_time_mean_balanced_accuracy": balanced_accuracy,
                            "d1_time_mean_confusion_mi_bits": mi_bits,
                            "mean_total_expected_spikes": mean_total_expected_spikes,
                            "real_minus_fixed_d1_time_mean_accuracy": float("nan"),
                            "real_minus_stabilized_d1_time_mean_accuracy": float("nan"),
                            "real_minus_random_amp_d1_time_mean_accuracy": float("nan"),
                            "real_minus_random_cov_d1_time_mean_accuracy": float("nan"),
                        }
                    )

    d1_lookup: dict[tuple[str, float, str, int], dict[str, object]] = {
        (
            str(row["condition"]),
            _safe_float(row["logmar"]),
            str(row["orientation_pair"]),
            _safe_int(row["integration_window"]),
        ): row
        for row in d1_rows
    }
    for row in d1_rows:
        if str(row.get("condition")) != "real":
            continue
        key_base = (
            _safe_float(row.get("logmar")),
            str(row.get("orientation_pair")),
            _safe_int(row.get("integration_window")),
        )
        value_real = _safe_float(row.get("d1_time_mean_accuracy"))
        for control_name, field_name in (
            ("fixed_center", "real_minus_fixed_d1_time_mean_accuracy"),
            ("stabilized", "real_minus_stabilized_d1_time_mean_accuracy"),
            ("random_amp", "real_minus_random_amp_d1_time_mean_accuracy"),
            ("random_cov", "real_minus_random_cov_d1_time_mean_accuracy"),
        ):
            control = d1_lookup.get((control_name, key_base[0], key_base[1], key_base[2]))
            if control is not None:
                row[field_name] = value_real - _safe_float(control.get("d1_time_mean_accuracy"))
    return d1_rows


def _write_eoptotype_reconciliation_bundle(
    out_dir: Path,
    decoder_rows: list[dict[str, object]],
    d1_rows: list[dict[str, object]],
    trace_qc_rows: list[dict[str, object]],
) -> None:
    qc_random_cov_pass = _random_control_path_acf_qc_pass(trace_qc_rows)
    logmar_values = sorted({_safe_float(row.get("logmar")) for row in decoder_rows if np.isfinite(_safe_float(row.get("logmar")))})

    d1_w60_lookup = {
        (str(row["condition"]), _safe_float(row["logmar"]), str(row["orientation_pair"])): row
        for row in d1_rows
        if _safe_int(row.get("integration_window")) == 60
    }
    d1_all_lookup = {
        (
            str(row["condition"]),
            _safe_float(row["logmar"]),
            str(row["orientation_pair"]),
            _safe_int(row["integration_window"]),
        ): row
        for row in d1_rows
    }

    reconciliation_rows: list[dict[str, object]] = []
    for row in decoder_rows:
        key = (str(row["condition"]), _safe_float(row["logmar"]), str(row["orientation_pair"]))
        d1_row = d1_w60_lookup.get(key)
        out_row = {
            "condition": str(row.get("condition")),
            "logmar": _safe_float(row.get("logmar")),
            "orientation_pair": str(row.get("orientation_pair")),
            "readout_type": str(row.get("readout_type")),
            "feature_representation": str(row.get("feature_representation")),
            "d1_time_mean_accuracy_w60": _safe_float(d1_row.get("d1_time_mean_accuracy") if d1_row else float("nan")),
            "d1_real_minus_fixed_w60": _safe_float(d1_row.get("real_minus_fixed_d1_time_mean_accuracy") if d1_row else float("nan")),
            "d1_real_minus_stabilized_w60": _safe_float(d1_row.get("real_minus_stabilized_d1_time_mean_accuracy") if d1_row else float("nan")),
            "d1_real_minus_random_amp_w60": _safe_float(d1_row.get("real_minus_random_amp_d1_time_mean_accuracy") if d1_row else float("nan")),
            "d1_real_minus_random_cov_w60": _safe_float(d1_row.get("real_minus_random_cov_d1_time_mean_accuracy") if d1_row else float("nan")),
            "deterministic_identity_bits_per_expected_spike": _safe_float(row.get("deterministic_identity_bits_per_expected_spike")),
            "d2_per_expected_spike": _safe_float(row.get("d2_per_expected_spike")),
            "mean_total_expected_spikes": _safe_float(row.get("mean_total_expected_spikes")),
            "real_minus_fixed_identity_bits_per_expected_spike": _safe_float(row.get("real_minus_fixed_identity_bits_per_expected_spike")),
            "real_minus_stabilized_identity_bits_per_expected_spike": _safe_float(row.get("real_minus_stabilized_identity_bits_per_expected_spike")),
            "real_minus_random_amp_identity_bits_per_expected_spike": _safe_float(row.get("real_minus_random_amp_identity_bits_per_expected_spike")),
            "real_minus_random_cov_identity_bits_per_expected_spike": _safe_float(row.get("real_minus_random_cov_identity_bits_per_expected_spike")),
            "real_minus_fixed_d2_per_expected_spike": _safe_float(row.get("real_minus_fixed_d2_per_expected_spike")),
            "real_minus_stabilized_d2_per_expected_spike": _safe_float(row.get("real_minus_stabilized_d2_per_expected_spike")),
            "real_minus_random_amp_d2_per_expected_spike": _safe_float(row.get("real_minus_random_amp_d2_per_expected_spike")),
            "real_minus_random_cov_d2_per_expected_spike": _safe_float(row.get("real_minus_random_cov_d2_per_expected_spike")),
            "random_cov_path_acf_qc_pass": int(bool(qc_random_cov_pass)),
        }
        if not qc_random_cov_pass:
            out_row["d1_real_minus_random_cov_w60"] = float("nan")
            out_row["real_minus_random_cov_identity_bits_per_expected_spike"] = float("nan")
            out_row["real_minus_random_cov_d2_per_expected_spike"] = float("nan")
        reconciliation_rows.append(out_row)

    _write_csv(out_dir / "eoptotype_D1_vs_efficiency_reconciliation.csv", reconciliation_rows)

    sweep_rows: list[dict[str, object]] = []
    windows = sorted({_safe_int(row.get("integration_window")) for row in d1_rows})
    for row in decoder_rows:
        cond = str(row.get("condition"))
        lm = _safe_float(row.get("logmar"))
        pair = str(row.get("orientation_pair"))
        for window in windows:
            d1_row = d1_all_lookup.get((cond, lm, pair, int(window)))
            if d1_row is None:
                continue
            sweep_rows.append(
                {
                    "condition": cond,
                    "logmar": lm,
                    "orientation_pair": pair,
                    "readout_type": str(row.get("readout_type")),
                    "feature_representation": str(row.get("feature_representation")),
                    "integration_window": int(window),
                    "d1_time_mean_accuracy": _safe_float(d1_row.get("d1_time_mean_accuracy")),
                    "real_minus_stabilized_d1_time_mean_accuracy": _safe_float(d1_row.get("real_minus_stabilized_d1_time_mean_accuracy")),
                }
            )
    _write_csv(out_dir / "eoptotype_D1_integration_window_sweep.csv", sweep_rows)

    anchor_rows = [
        row
        for row in reconciliation_rows
        if str(row.get("condition")) == "real"
        and str(row.get("readout_type")) == "linear"
        and str(row.get("feature_representation")) == "spatial_avg_time_mean"
    ]

    def _metric_at_logmar(rows: list[dict[str, object]], metric: str, lm: float) -> float:
        return _nanmedian_or_nan(_safe_float(row.get(metric)) for row in rows if _safe_float(row.get("logmar")) == float(lm))

    d1_035 = _metric_at_logmar(anchor_rows, "d1_real_minus_stabilized_w60", -0.35)
    d1_040 = _metric_at_logmar(anchor_rows, "d1_real_minus_stabilized_w60", -0.40)
    bits_035 = _metric_at_logmar(anchor_rows, "real_minus_stabilized_identity_bits_per_expected_spike", -0.35)
    bits_040 = _metric_at_logmar(anchor_rows, "real_minus_stabilized_identity_bits_per_expected_spike", -0.40)
    d2_035 = _metric_at_logmar(anchor_rows, "real_minus_stabilized_d2_per_expected_spike", -0.35)
    d2_040 = _metric_at_logmar(anchor_rows, "real_minus_stabilized_d2_per_expected_spike", -0.40)

    sat_rows: list[dict[str, object]] = []
    for lm in sorted(set(logmar_values + [DEFAULT_RECON_SATURATION_LOGMAR])):
        sat_rows.append(
            {
                "type": "policy",
                "logmar": float(lm),
                "is_primary_range": int(any(abs(lm - p) < 1e-9 for p in DEFAULT_RECON_PRIMARY_LOGMARS)),
                "is_saturation_control": int(abs(lm - DEFAULT_RECON_SATURATION_LOGMAR) < 1e-9),
                "excluded_by_policy": int(lm <= -0.45),
            }
        )

    def _sat_flag(v35: float, v40: float, abs_floor: float) -> tuple[float, float, int]:
        if not (np.isfinite(v35) and np.isfinite(v40)):
            return float("nan"), float("nan"), 0
        delta = abs(v40 - v35)
        tol = max(abs_floor, 0.15 * abs(v35))
        return delta, tol, int(delta <= tol)

    d1_delta, d1_tol, d1_sat = _sat_flag(d1_035, d1_040, 0.02)
    bits_delta, bits_tol, bits_sat = _sat_flag(bits_035, bits_040, 1e-8)
    d2_delta, d2_tol, d2_sat = _sat_flag(d2_035, d2_040, 1e-8)
    sat_rows.extend(
        [
            {
                "type": "metric",
                "metric": "d1_real_minus_stabilized_w60",
                "value_at_minus_0p35": d1_035,
                "value_at_minus_0p40": d1_040,
                "abs_delta": d1_delta,
                "tolerance": d1_tol,
                "saturation_flag": d1_sat,
            },
            {
                "type": "metric",
                "metric": "bits_real_minus_stabilized",
                "value_at_minus_0p35": bits_035,
                "value_at_minus_0p40": bits_040,
                "abs_delta": bits_delta,
                "tolerance": bits_tol,
                "saturation_flag": bits_sat,
            },
            {
                "type": "metric",
                "metric": "d2_per_spike_real_minus_stabilized",
                "value_at_minus_0p35": d2_035,
                "value_at_minus_0p40": d2_040,
                "abs_delta": d2_delta,
                "tolerance": d2_tol,
                "saturation_flag": d2_sat,
            },
            {
                "type": "aggregate",
                "metric": "all_metrics_saturated_at_minus_0p40",
                "saturation_flag": int(bool(d1_sat and bits_sat and d2_sat)),
            },
        ]
    )
    sat_fieldnames = [
        "type",
        "logmar",
        "is_primary_range",
        "is_saturation_control",
        "excluded_by_policy",
        "metric",
        "value_at_minus_0p35",
        "value_at_minus_0p40",
        "abs_delta",
        "tolerance",
        "saturation_flag",
    ]
    _write_csv(out_dir / "eoptotype_logmar_saturation_flags.csv", sat_rows, fieldnames=sat_fieldnames)

    if np.isfinite(d1_040) and d1_040 > 0 and (not np.isfinite(d1_035) or d1_035 <= 0):
        decision_label = "plateau_limited"
    elif not np.isfinite(d1_035) or d1_035 <= 0:
        decision_label = "prior_D1_result_not_reproduced"
    elif not np.isfinite(bits_035) or bits_035 <= 0:
        decision_label = "accumulated_discriminability_supported_efficiency_not_supported"
    else:
        decision_label = "active_sensing_efficiency_supported_model_side"

    decision_rows = [
        {
            "decision_label": decision_label,
            "d1_real_minus_stabilized_w60_at_minus_0p35": d1_035,
            "bits_real_minus_stabilized_at_minus_0p35": bits_035,
            "d2_real_minus_stabilized_at_minus_0p35": d2_035,
            "d1_real_minus_stabilized_w60_at_minus_0p40": d1_040,
            "random_cov_path_acf_qc_pass": int(bool(qc_random_cov_pass)),
            "primary_logmar_policy": "-0.20,-0.25,-0.30,-0.35",
            "saturation_control_logmar": "-0.40",
            "omitted_logmar_policy": "-0.45,-0.50,-0.55",
        }
    ]
    _write_csv(out_dir / "eoptotype_metric_crosswalk_decision_table.csv", decision_rows)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # D1 accuracy at W=60 by condition over logmar.
    w60_rows = [row for row in d1_rows if _safe_int(row.get("integration_window")) == 60]
    if w60_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        for condition, label in (("real", "real"), ("stabilized", "stabilized"), ("fixed_center", "fixed_center")):
            xs = sorted({_safe_float(row.get("logmar")) for row in w60_rows if str(row.get("condition")) == condition})
            ys = [
                _nanmedian_or_nan(
                    _safe_float(row.get("d1_time_mean_accuracy"))
                    for row in w60_rows
                    if str(row.get("condition")) == condition and _safe_float(row.get("logmar")) == x
                )
                for x in xs
            ]
            if xs:
                ax.plot(xs, ys, marker="o", label=label)
        ax.set_xlabel("LogMAR")
        ax.set_ylabel("D1 accuracy (W=60)")
        ax.set_title("D1 Mean-Rate Discriminability vs LogMAR")
        ax.legend(frameon=False)
        ax.set_ylim(0.45, 1.0)
        fig.tight_layout()
        fig.savefig(fig_dir / "D1_accuracy_vs_logmar.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        xs = sorted({_safe_float(row.get("logmar")) for row in w60_rows if str(row.get("condition")) == "real"})
        ys = [
            _nanmedian_or_nan(
                _safe_float(row.get("real_minus_stabilized_d1_time_mean_accuracy"))
                for row in w60_rows
                if str(row.get("condition")) == "real" and _safe_float(row.get("logmar")) == x
            )
            for x in xs
        ]
        if xs:
            ax.plot(xs, ys, marker="o", color="tab:blue")
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
            ax.set_xlabel("LogMAR")
            ax.set_ylabel("real - stabilized D1 accuracy (W=60)")
            ax.set_title("D1 Real-Minus-Stabilized vs LogMAR")
            fig.tight_layout()
            fig.savefig(fig_dir / "D1_real_minus_stabilized_vs_logmar.png", dpi=200)
            plt.close(fig)

    if anchor_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        for condition, label in (("real", "real"), ("stabilized", "stabilized")):
            xs = sorted({_safe_float(row.get("logmar")) for row in anchor_rows if str(row.get("condition")) == condition})
            ys = [
                _nanmedian_or_nan(
                    _safe_float(row.get("d2_per_expected_spike"))
                    for row in anchor_rows
                    if str(row.get("condition")) == condition and _safe_float(row.get("logmar")) == x
                )
                for x in xs
            ]
            if xs:
                ax.plot(xs, ys, marker="o", label=label)
        ax.set_xlabel("LogMAR")
        ax.set_ylabel("d2 per expected spike")
        ax.set_title("d2/Spike vs LogMAR")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "d2_per_spike_vs_logmar.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for condition, label in (("real", "real"), ("stabilized", "stabilized")):
            xs = sorted({_safe_float(row.get("logmar")) for row in anchor_rows if str(row.get("condition")) == condition})
            ys = [
                _nanmedian_or_nan(
                    _safe_float(row.get("deterministic_identity_bits_per_expected_spike"))
                    for row in anchor_rows
                    if str(row.get("condition")) == condition and _safe_float(row.get("logmar")) == x
                )
                for x in xs
            ]
            if xs:
                ax.plot(xs, ys, marker="o", label=label)
        ax.set_xlabel("LogMAR")
        ax.set_ylabel("identity bits / expected spike")
        ax.set_title("Deterministic Identity Bits/Spike vs LogMAR")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "identity_bits_per_spike_vs_logmar.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        for lm in sorted({_safe_float(row.get("logmar")) for row in anchor_rows}):
            x = _metric_at_logmar(anchor_rows, "real_minus_stabilized_identity_bits_per_expected_spike", lm)
            y = _metric_at_logmar(anchor_rows, "d1_real_minus_stabilized_w60", lm)
            if np.isfinite(x) and np.isfinite(y):
                ax.scatter([x], [y], s=50)
                ax.text(x, y, f"{lm:.2f}", fontsize=8)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("real - stabilized identity bits / expected spike")
        ax.set_ylabel("real - stabilized D1 accuracy (W=60)")
        ax.set_title("D1 vs Bits/Spike Crosswalk")
        fig.tight_layout()
        fig.savefig(fig_dir / "D1_vs_bits_per_spike_crosswalk.png", dpi=200)
        plt.close(fig)

    readme = out_dir / "eoptotype_reconciliation_readme.md"
    readme.write_text(
        "\n".join(
            [
                "# E-optotype D1 vs Efficiency Reconciliation",
                "",
                "Primary LogMAR policy: -0.20, -0.25, -0.30, -0.35.",
                "Saturation-control LogMAR: -0.40 (used only as a plateau check).",
                "Omitted by policy due to prior saturation audits: -0.45, -0.50, -0.55.",
                "Random-cov real-minus contrasts are reported only when path-length + ACF QC pass.",
            ]
        )
        + "\n"
    )


def _poisson_d2(rA: np.ndarray, rB: np.ndarray, eps: float = EPS) -> float:
    rA = np.asarray(rA, dtype=np.float64)
    rB = np.asarray(rB, dtype=np.float64)
    denom = 0.5 * (np.maximum(rA, 0.0) + np.maximum(rB, 0.0)) + float(eps)
    return float(np.sum(((rA - rB) ** 2) / denom))


def _poisson_resample_features_at_budget(
    X: np.ndarray,
    budget_spikes: float,
    rng_seed: int,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    budget = max(float(budget_spikes), EPS)
    out = np.zeros_like(X, dtype=np.float64)
    rng = np.random.default_rng(int(rng_seed))
    for idx in range(X.shape[0]):
        vec = np.maximum(X[idx], 0.0)
        s = float(np.sum(vec))
        if s <= EPS:
            lam = np.full(vec.shape, budget / max(1, vec.size), dtype=np.float64)
        else:
            lam = (vec / s) * budget
        out[idx] = rng.poisson(np.maximum(lam, 0.0)).astype(np.float64)
    return out


def _skaggs_identity_bits_per_expected_spike(response_matrix: np.ndarray) -> float:
    response_matrix = np.asarray(response_matrix, dtype=np.float64)
    response_matrix = np.maximum(response_matrix, 0.0)
    if response_matrix.ndim != 2 or response_matrix.shape[0] == 0 or response_matrix.shape[1] == 0:
        return float("nan")
    rbar = np.mean(response_matrix, axis=0)
    total_rate = float(np.sum(rbar))
    if total_rate <= EPS:
        return float("nan")
    denom = rbar + EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = response_matrix / denom[None, :]
        log_term = np.log2((response_matrix + EPS) / denom[None, :])
        i_i = np.mean(ratio * log_term, axis=0)
    return float(np.sum(rbar * i_i) / total_rate)


def _persist_eoptotype_response_vectors(
    out_dir: Path,
    trial_rows: list[dict[str, object]],
    feature_store: dict[str, np.ndarray],
    feature_representations: tuple[str, ...],
) -> tuple[Path, Path]:
    persistence_dir = out_dir / "eoptotype_identity"
    persistence_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = persistence_dir / "eoptotype_response_vectors.npz"
    schema_path = persistence_dir / "eoptotype_response_vectors.schema.json"

    metadata = {
        "trial_index": np.asarray([_safe_int(row["trial_index"]) for row in trial_rows], dtype=np.int32),
        "condition": np.asarray([str(row["condition"]) for row in trial_rows], dtype=np.str_),
        "logmar": np.asarray([_safe_float(row["logmar"]) for row in trial_rows], dtype=np.float32),
        "orientation": np.asarray([_safe_int(row["orientation"]) for row in trial_rows], dtype=np.int16),
        "random_repeat": np.asarray([_safe_int(row.get("random_repeat")) for row in trial_rows], dtype=np.int16),
        "feature_key": np.asarray([str(row["feature_key"]) for row in trial_rows], dtype=np.str_),
        "n_time_bins": np.asarray([_safe_int(row["n_time_bins"]) for row in trial_rows], dtype=np.int16),
        "n_units": np.asarray([_safe_int(row["n_units"]) for row in trial_rows], dtype=np.int16),
    }
    payload: dict[str, np.ndarray] = dict(metadata)
    for feature_name in feature_representations:
        payload[f"vectors__{feature_name}"] = np.stack(
            [np.asarray(feature_store[f"{row['feature_key']}_{feature_name}"], dtype=np.float32) for row in trial_rows],
            axis=0,
        )
    np.savez_compressed(vectors_path, **payload)

    schema = {
        "kind": "trial_level_response_vectors",
        "row_alignment": "all arrays share the same row order as eoptotype_trial_metrics.csv",
        "metadata_fields": list(metadata.keys()),
        "feature_arrays": [f"vectors__{feature_name}" for feature_name in feature_representations],
        "notes": [
            "Rows correspond to trial_index x condition x logmar x orientation x random_repeat.",
            "Feature arrays hold the per-trial response vectors used for post-hoc class-conditional identity information.",
        ],
    }
    _write_json(schema_path, schema)
    return vectors_path, schema_path


def _compute_deterministic_identity_summary_rows(
    trial_rows: list[dict[str, object]],
    feature_store: dict[str, np.ndarray],
    feature_representations: tuple[str, ...],
    readout_types: tuple[str, ...],
    logmar_values: tuple[float, ...],
    orientations: tuple[int, ...],
) -> tuple[list[dict[str, object]], dict[tuple[str, float, str, str, str], dict[str, float]], dict[tuple[str, float, str, str, str], dict[str, np.ndarray]]]:
    vector_cache: dict[str, np.ndarray] = {
        feature_name: np.stack(
            [np.asarray(feature_store[f"{row['feature_key']}_{feature_name}"], dtype=np.float64) for row in trial_rows],
            axis=0,
        )
        for feature_name in feature_representations
    }

    meta_condition = np.asarray([str(row["condition"]) for row in trial_rows], dtype=np.str_)
    meta_logmar = np.asarray([_safe_float(row["logmar"]) for row in trial_rows], dtype=np.float64)
    meta_orientation = np.asarray([_safe_int(row["orientation"]) for row in trial_rows], dtype=np.int32)
    meta_repeat = np.asarray([_safe_int(row.get("random_repeat")) for row in trial_rows], dtype=np.int32)
    meta_trial_index = np.asarray([_safe_int(row["trial_index"]) for row in trial_rows], dtype=np.int32)

    summary_rows: list[dict[str, object]] = []
    summary_lookup: dict[tuple[str, float, str, str, str], dict[str, float]] = {}
    vector_lookup: dict[tuple[str, float, str, str, str], dict[str, np.ndarray]] = {}

    orientation_pairs = _pairwise_orientation_pairs(orientations)
    conditions = tuple(sorted({str(row["condition"]) for row in trial_rows}))

    for readout_type in readout_types:
        for feature_name in feature_representations:
            vectors = vector_cache[feature_name]
            for condition in conditions:
                for logmar in logmar_values:
                    cond_mask = (meta_condition == condition) & np.isclose(meta_logmar, float(logmar))
                    if not np.any(cond_mask):
                        continue

                    subset_vectors = vectors[cond_mask]
                    subset_orientations = meta_orientation[cond_mask]
                    subset_repeats = meta_repeat[cond_mask]
                    subset_trial_index = meta_trial_index[cond_mask]
                    subset_spikes = np.asarray(
                        [
                            _safe_float(row["total_expected_spikes"])
                            for row, keep in zip(trial_rows, cond_mask, strict=False)
                            if keep
                        ],
                        dtype=np.float64,
                    )

                    classes_present = [ori for ori in orientations if np.any(subset_orientations == int(ori))]
                    if len(classes_present) >= 2:
                        class_means = np.stack(
                            [np.mean(subset_vectors[subset_orientations == int(ori)], axis=0) for ori in classes_present],
                            axis=0,
                        )
                        all_bits = _skaggs_identity_bits_per_expected_spike(class_means)
                        all_key = (condition, float(logmar), "all_orientations", readout_type, feature_name)
                        summary_lookup[all_key] = {
                            "deterministic_identity_bits_per_expected_spike": float(all_bits),
                            "pairwise_deterministic_identity_bits_per_expected_spike": float("nan"),
                            "n_trials": int(subset_vectors.shape[0]),
                            "mean_total_expected_spikes": float(np.mean(subset_spikes)) if subset_spikes.size else float("nan"),
                        }
                        vector_lookup[all_key] = {"vectors": class_means.astype(np.float32), "classes": np.asarray(classes_present, dtype=np.int16)}
                        summary_rows.append(
                            {
                                "condition": condition,
                                "logmar": float(logmar),
                                "orientation_pair": "all_orientations",
                                "readout_type": readout_type,
                                "feature_representation": feature_name,
                                "n_trials": int(subset_vectors.shape[0]),
                                "deterministic_identity_bits_per_expected_spike": float(all_bits),
                                "pairwise_deterministic_identity_bits_per_expected_spike": float("nan"),
                                "mean_total_expected_spikes": float(np.mean(subset_spikes)) if subset_spikes.size else float("nan"),
                                "real_minus_fixed_identity_bits_per_expected_spike": float("nan"),
                                "real_minus_stabilized_identity_bits_per_expected_spike": float("nan"),
                                "real_minus_random_amp_identity_bits_per_expected_spike": float("nan"),
                                "real_minus_random_cov_identity_bits_per_expected_spike": float("nan"),
                            }
                        )

                    for ori_a, ori_b in orientation_pairs:
                        pair_mask = cond_mask & np.isin(meta_orientation, np.asarray([ori_a, ori_b], dtype=np.int32))
                        if not np.any(pair_mask):
                            continue
                        pair_vectors = vectors[pair_mask]
                        pair_orientations = meta_orientation[pair_mask]
                        class_a = pair_vectors[pair_orientations == int(ori_a)]
                        class_b = pair_vectors[pair_orientations == int(ori_b)]
                        if class_a.size == 0 or class_b.size == 0:
                            continue
                        pair_class_means = np.stack([np.mean(class_a, axis=0), np.mean(class_b, axis=0)], axis=0)
                        pair_bits = _skaggs_identity_bits_per_expected_spike(pair_class_means)
                        pair_key = (condition, float(logmar), f"{ori_a}_vs_{ori_b}", readout_type, feature_name)
                        summary_lookup[pair_key] = {
                            "deterministic_identity_bits_per_expected_spike": float(pair_bits),
                            "pairwise_deterministic_identity_bits_per_expected_spike": float(pair_bits),
                            "n_trials": int(pair_vectors.shape[0]),
                            "mean_total_expected_spikes": float(np.mean([
                                _safe_float(row["total_expected_spikes"])
                                for row, keep in zip(trial_rows, pair_mask, strict=False)
                                if keep
                            ])),
                        }
                        vector_lookup[pair_key] = {"vectors": pair_class_means.astype(np.float32), "classes": np.asarray([ori_a, ori_b], dtype=np.int16)}
                        summary_rows.append(
                            {
                                "condition": condition,
                                "logmar": float(logmar),
                                "orientation_pair": f"{ori_a}_vs_{ori_b}",
                                "readout_type": readout_type,
                                "feature_representation": feature_name,
                                "n_trials": int(pair_vectors.shape[0]),
                                "deterministic_identity_bits_per_expected_spike": float(pair_bits),
                                "pairwise_deterministic_identity_bits_per_expected_spike": float(pair_bits),
                                "mean_total_expected_spikes": float(np.mean([
                                    _safe_float(row["total_expected_spikes"])
                                    for row, keep in zip(trial_rows, pair_mask, strict=False)
                                    if keep
                                ])),
                                "real_minus_fixed_identity_bits_per_expected_spike": float("nan"),
                                "real_minus_stabilized_identity_bits_per_expected_spike": float("nan"),
                                "real_minus_random_amp_identity_bits_per_expected_spike": float("nan"),
                                "real_minus_random_cov_identity_bits_per_expected_spike": float("nan"),
                            }
                        )

    # Fill real-minus-control contrasts for rows that have matched counterparts.
    summary_index = {
        (str(row["condition"]), _safe_float(row["logmar"]), str(row["orientation_pair"]), str(row["readout_type"]), str(row["feature_representation"])): row
        for row in summary_rows
    }
    for row in summary_rows:
        if str(row["condition"]) != "real":
            continue
        key_base = (
            _safe_float(row["logmar"]),
            str(row["orientation_pair"]),
            str(row["readout_type"]),
            str(row["feature_representation"]),
        )
        real_bits = _safe_float(row["deterministic_identity_bits_per_expected_spike"])
        for control_name, field_name in (
            ("fixed_center", "real_minus_fixed_identity_bits_per_expected_spike"),
            ("stabilized", "real_minus_stabilized_identity_bits_per_expected_spike"),
            ("random_amp", "real_minus_random_amp_identity_bits_per_expected_spike"),
            ("random_cov", "real_minus_random_cov_identity_bits_per_expected_spike"),
        ):
            control_row = summary_index.get((control_name, key_base[0], key_base[1], key_base[2], key_base[3]))
            if control_row is not None:
                row[field_name] = real_bits - _safe_float(control_row["deterministic_identity_bits_per_expected_spike"])

    return summary_rows, summary_lookup, vector_lookup


def _aggregate_eoptotype_trial_rows(
    trial_rows: list[dict[str, object]],
    feature_store: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    if not trial_rows:
        return [], {}

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in trial_rows:
        key = (
            row["condition"],
            row["logmar"],
            row["orientation"],
            row["trial_index"],
        )
        grouped[key].append(row)

    aggregated_rows: list[dict[str, object]] = []
    aggregated_features: dict[str, np.ndarray] = {}
    numeric_average_keys = (
        "total_expected_spikes",
        "mean_expected_spikes_per_bin",
        "n_time_bins",
        "n_units",
    )

    probe_prefix = str(trial_rows[0]["feature_key"]) + "_"
    feature_suffixes = sorted(
        {
            key[len(probe_prefix) :]
            for key in feature_store
            if key.startswith(probe_prefix)
        }
    )

    for agg_index, group in enumerate(grouped.values()):
        template = dict(group[0])
        for key in numeric_average_keys:
            template[key] = _nanmean_or_nan(_safe_float(row.get(key)) for row in group)
        feature_key = f"agg_feat_{agg_index:06d}"
        for suffix in feature_suffixes:
            stacked = np.stack([feature_store[f"{row['feature_key']}_{suffix}"] for row in group], axis=0)
            aggregated_features[f"{feature_key}_{suffix}"] = np.mean(stacked, axis=0).astype(np.float32)
        template["feature_key"] = feature_key
        template["random_repeat"] = -1
        template["n_aggregated_repeats"] = len(group)
        aggregated_rows.append(template)
    return aggregated_rows, aggregated_features


def _build_sanity_rows(
    trace_qc_rows: list[dict[str, object]],
    dt_rows: list[dict[str, object]],
    fixrsvp_trials: list[dict[str, object]],
    decoder_rows: list[dict[str, object]],
    eoptotype_expected: bool,
) -> list[dict[str, object]]:
    results: list[SanityResult] = []
    rms_errors = [_safe_float(row.get("matched_rms_error")) for row in trace_qc_rows if str(row.get("condition")) == "random_amp"]
    cov_errors = [_safe_float(row.get("matched_cov_error")) for row in trace_qc_rows if str(row.get("condition")) == "random_cov"]
    dt_plausible = sum(1 for row in dt_rows if str(row.get("plausibility_status")) == "plausible")

    results.append(
        SanityResult(
            name="fixrsvp_trials_present",
            status="pass" if len(fixrsvp_trials) > 0 else "fail",
            value=len(fixrsvp_trials),
            threshold="> 0",
            detail="At least one fixRSVP trial survives extraction.",
        )
    )
    rms_median = _nanmedian_or_nan(rms_errors)
    results.append(
        SanityResult(
            name="random_amp_rms_error",
            status=("pass" if np.isfinite(rms_median) and rms_median <= 0.15 else ("warn" if rms_errors else "skipped")),
            value=rms_median,
            threshold="<= 0.15",
            detail="Median RMS mismatch for random_amp controls.",
        )
    )
    cov_median = _nanmedian_or_nan(cov_errors)
    results.append(
        SanityResult(
            name="random_cov_cov_error",
            status=("pass" if np.isfinite(cov_median) and cov_median <= 0.20 else ("warn" if cov_errors else "skipped")),
            value=cov_median,
            threshold="<= 0.20",
            detail="Median covariance mismatch for random_cov controls.",
        )
    )
    results.append(
        SanityResult(
            name="dt_convention_plausible",
            status="pass" if dt_plausible > 0 else "fail",
            value=dt_plausible,
            threshold=">= 1 plausible dt candidate",
            detail="At least one dt candidate produced plausible firing rates.",
        )
    )

    if decoder_rows:
        easy_logmar = max(_safe_float(row["logmar"]) for row in decoder_rows)
        easy_candidates = [
            row
            for row in decoder_rows
            if str(row["condition"]) == "real"
            and str(row["readout_type"]) == "linear"
            and _safe_float(row["logmar"]) == easy_logmar
            and str(row["orientation_pair"]) in {"0_vs_180", "0_vs_90"}
        ]
        easy_acc = _nanmedian_or_nan(_safe_float(row.get("poisson_budget_accuracy")) for row in easy_candidates)
        status = "pass" if np.isfinite(easy_acc) and easy_acc >= 0.60 else ("fail" if eoptotype_expected else "skipped")
        results.append(
            SanityResult(
                name="eoptotype_easy_decoder_accuracy",
                status=status,
                value=easy_acc,
                threshold=">= 0.60",
                detail="Easy-condition Poisson-budget decoder should be above chance when eoptotype is requested.",
            )
        )
    else:
        results.append(
            SanityResult(
                name="eoptotype_easy_decoder_accuracy",
                status="fail" if eoptotype_expected else "skipped",
                value=float("nan"),
                threshold=">= 0.60",
                detail="No decoder rows available; this fails sanity unless --skip-eoptotype was used.",
            )
        )

    return [
        {
            "check_name": result.name,
            "status": result.status,
            "value": result.value,
            "threshold": result.threshold,
            "detail": result.detail,
        }
        for result in results
    ]


def _plot_box_or_line(df_rows: list[dict[str, object]], out_path: Path, x_key: str, y_key: str, hue_key: str, title: str, ylabel: str) -> None:
    if not df_rows:
        return
    x_values = sorted({str(row[x_key]) for row in df_rows})
    hues = sorted({str(row[hue_key]) for row in df_rows})
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(hues))
    x_positions = np.arange(len(x_values), dtype=np.float64)
    for hue_idx, hue in enumerate(hues):
        means = []
        sems = []
        for x_value in x_values:
            vals = np.asarray([
                _safe_float(row[y_key])
                for row in df_rows
                if str(row[x_key]) == x_value and str(row[hue_key]) == hue and np.isfinite(_safe_float(row[y_key]))
            ], dtype=np.float64)
            mean, sem = _mean_sem(vals)
            means.append(mean)
            sems.append(sem)
        xs = x_positions + (hue_idx - (len(hues) - 1) / 2.0) * width
        ax.bar(xs, means, width=width, label=hue, alpha=0.8)
        ax.errorbar(xs, means, yerr=sems, fmt="none", color="black", linewidth=1.0, capsize=2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(x) for x in x_values])
    ax.set_title(title)
    ax.set_xlabel(x_key.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _build_decision_label(
    real_minus_fixed: float,
    real_minus_random: float,
    rate_delta: float,
    ci_width: float,
) -> str:
    if np.isfinite(ci_width) and np.isfinite(real_minus_fixed) and abs(real_minus_fixed) > 0 and ci_width / max(abs(real_minus_fixed), EPS) > 2.0:
        return "underpowered"
    if np.isfinite(real_minus_fixed) and real_minus_fixed < 0:
        return "fem_efficiency_cost"
    if np.isfinite(real_minus_fixed) and np.isfinite(real_minus_random):
        if real_minus_fixed > 0 and real_minus_random > 0 and rate_delta <= 0.0:
            return "model_active_sensing_efficiency_supported"
        if real_minus_fixed > 0 and abs(real_minus_random) <= abs(real_minus_fixed) * 0.25:
            return "generic_dither_efficiency_supported"
        if real_minus_fixed > 0 and rate_delta > 0.0:
            return "rate_or_total_information_only"
    if np.isfinite(real_minus_fixed) and abs(real_minus_fixed) < 1e-6:
        return "no_efficiency_benefit"
    return "inconclusive_mixed"


def _compute_eoptotype_closeout_flags(
    decoder_rows: list[dict[str, object]],
    trace_qc_rows: list[dict[str, object]],
) -> dict[str, bool | int]:
    primary_rows = [
        row
        for row in decoder_rows
        if str(row.get("condition")) == "real"
        and str(row.get("feature_representation")) == "spatial_avg_time_mean"
        and str(row.get("readout_type")) == "linear"
    ]
    logmars = sorted({_safe_float(row.get("logmar")) for row in primary_rows if np.isfinite(_safe_float(row.get("logmar")))})
    orientation_pairs = sorted({str(row.get("orientation_pair")) for row in primary_rows if str(row.get("orientation_pair", ""))})
    min_trials_per_condition_pair = min(
        [
            max(1, (_safe_int(row.get("n_train")) + _safe_int(row.get("n_test"))) // 2)
            for row in decoder_rows
            if str(row.get("readout_type")) == "linear"
        ]
        or [0]
    )

    has_logmar_scope = len(logmars) >= DEFAULT_EOPTOTYPE_MIN_LOGMARS
    has_pair_scope = len(orientation_pairs) >= DEFAULT_EOPTOTYPE_MIN_ORIENTATION_PAIRS
    has_required_pairs = set(DEFAULT_REQUIRED_EOPTOTYPE_PAIRS).issubset(set(orientation_pairs))
    has_trial_scope = min_trials_per_condition_pair >= DEFAULT_EOPTOTYPE_MIN_TRIALS_PER_COND_PAIR

    real_rows = [row for row in decoder_rows if str(row.get("condition")) == "real"]
    by_key: dict[tuple[str, str, float, str], dict[str, float]] = {}
    for row in real_rows:
        key = (
            str(row.get("readout_type")),
            str(row.get("orientation_pair")),
            _safe_float(row.get("logmar")),
            str(row.get("feature_representation")),
        )
        by_key[key] = {
            "fix": _safe_float(row.get("real_minus_fixed_d2_per_expected_spike")),
            "cov": _safe_float(row.get("real_minus_random_cov_d2_per_expected_spike")),
        }
    ablation_keys = [
        key for key in by_key if key[3] in {"spatial_avg_time_mean", "map_energy", "spatial_avg_energy"}
    ]
    anchor_keys = [key for key in ablation_keys if key[3] == "spatial_avg_time_mean"]
    sign_matches = []
    for akey in anchor_keys:
        anchor = by_key.get(akey)
        if anchor is None:
            continue
        for alt_feature in ("spatial_avg_energy", "map_energy"):
            alt_key = (akey[0], akey[1], akey[2], alt_feature)
            alt = by_key.get(alt_key)
            if alt is None:
                continue
            if np.isfinite(anchor["fix"]) and np.isfinite(alt["fix"]):
                sign_matches.append(np.sign(anchor["fix"]) == np.sign(alt["fix"]))
            if np.isfinite(anchor["cov"]) and np.isfinite(alt["cov"]):
                sign_matches.append(np.sign(anchor["cov"]) == np.sign(alt["cov"]))
    spatial_ablation_sign_consistent = bool(sign_matches) and bool(np.all(np.asarray(sign_matches, dtype=bool)))

    control_rows = [row for row in trace_qc_rows if str(row.get("condition")) in {"random_amp", "random_cov"}]
    control_rms_ok = _nanmedian_or_nan(_safe_float(row.get("matched_rms_error")) for row in control_rows) <= DEFAULT_RANDOM_CONTROL_RMS_ERROR_MAX
    control_cov_ok = _nanmedian_or_nan(_safe_float(row.get("matched_cov_error")) for row in control_rows) <= DEFAULT_RANDOM_CONTROL_COV_ERROR_MAX
    control_path_ok = _nanmedian_or_nan(_safe_float(row.get("path_length_error")) for row in control_rows) <= DEFAULT_RANDOM_CONTROL_PATH_ERROR_MAX
    control_acf1_ok = _nanmedian_or_nan(_safe_float(row.get("acf_lag1_error")) for row in control_rows) <= DEFAULT_RANDOM_CONTROL_ACF_ERROR_MAX
    control_acf2_ok = _nanmedian_or_nan(_safe_float(row.get("acf_lag2_error")) for row in control_rows) <= DEFAULT_RANDOM_CONTROL_ACF_ERROR_MAX
    control_acf4_ok = _nanmedian_or_nan(_safe_float(row.get("acf_lag4_error")) for row in control_rows) <= DEFAULT_RANDOM_CONTROL_ACF_ERROR_MAX
    controls_pass_full = control_rms_ok and control_cov_ok and control_path_ok and control_acf1_ok and control_acf2_ok and control_acf4_ok

    renderer_normalization_confirmed = any(str(row.get("renderer_normalization_status", "")) == "matched_hyperacuity_pipeline" for row in decoder_rows)
    covered_features = {str(row.get("feature_representation")) for row in decoder_rows}
    has_required_feature_representations = {"spatial_avg_time_mean", "spatial_avg_energy", "map_energy"}.issubset(covered_features)
    readouts_present = {str(row.get("readout_type")) for row in decoder_rows}
    readout_feature_keys = {
        (str(row.get("readout_type")), str(row.get("feature_representation")))
        for row in decoder_rows
    }
    per_readout_feature_coverage = all(
        all((readout, feature) in readout_feature_keys for feature in ("spatial_avg_time_mean", "spatial_avg_energy", "map_energy"))
        for readout in readouts_present
    ) if readouts_present else False

    closeout_scope_pass = has_logmar_scope and has_pair_scope and has_required_pairs and has_trial_scope
    closeout_ready = (
        closeout_scope_pass
        and controls_pass_full
        and spatial_ablation_sign_consistent
        and has_required_feature_representations
        and per_readout_feature_coverage
        and renderer_normalization_confirmed
    )
    return {
        "n_logmar": len(logmars),
        "n_orientation_pairs": len(orientation_pairs),
        "min_trials_per_condition_pair": min_trials_per_condition_pair,
        "scope_pass": closeout_scope_pass,
        "required_pairs_pass": has_required_pairs,
        "controls_pass_full": controls_pass_full,
        "spatial_ablation_sign_consistent": spatial_ablation_sign_consistent,
        "has_required_feature_representations": has_required_feature_representations,
        "per_readout_feature_coverage": per_readout_feature_coverage,
        "renderer_normalization_confirmed": renderer_normalization_confirmed,
        "closeout_ready": closeout_ready,
    }


def _build_identity_decision_label(decoder_rows: list[dict[str, object]], trace_qc_rows: list[dict[str, object]]) -> str:
    primary_rows = [row for row in decoder_rows if str(row.get("condition")) == "real"]
    if not primary_rows:
        return "implementation_failure"

    real_minus_fixed_vals = np.asarray([_safe_float(row.get("real_minus_fixed_d2_per_expected_spike")) for row in primary_rows], dtype=np.float64)
    real_minus_random_vals = np.asarray([_safe_float(row.get("real_minus_random_cov_d2_per_expected_spike")) for row in primary_rows], dtype=np.float64)
    valid = np.isfinite(real_minus_fixed_vals) & np.isfinite(real_minus_random_vals)
    if not np.any(valid):
        return "implementation_failure"
    real_minus_fixed = float(np.nanmedian(real_minus_fixed_vals[valid]))
    real_minus_random = float(np.nanmedian(real_minus_random_vals[valid]))
    frac_pos = float(np.mean((real_minus_fixed_vals[valid] > 0) & (real_minus_random_vals[valid] > 0)))
    frac_neg = float(np.mean((real_minus_fixed_vals[valid] < 0) & (real_minus_random_vals[valid] < 0)))

    closeout_flags = _compute_eoptotype_closeout_flags(decoder_rows=decoder_rows, trace_qc_rows=trace_qc_rows)
    closeout_ready = bool(closeout_flags["closeout_ready"])
    scope_pass = bool(closeout_flags["scope_pass"])
    controls_pass = bool(closeout_flags["controls_pass_full"])
    ablation_pass = bool(closeout_flags["spatial_ablation_sign_consistent"])

    if not scope_pass:
        return "limited_scope_diagnostic"
    if not controls_pass:
        return "limited_by_control_mismatch"
    if not closeout_ready:
        return "limited_scope_diagnostic"

    if frac_pos >= 0.70 and ablation_pass:
        return "model_active_sensing_efficiency_supported"
    if real_minus_fixed > 0 and np.isfinite(real_minus_random) and abs(real_minus_random) <= abs(real_minus_fixed) * 0.25 and ablation_pass:
        return "generic_dither_efficiency_supported"
    if frac_neg >= 0.70 and ablation_pass:
        return "fem_efficiency_cost_fullscope"
    if np.sign(real_minus_fixed) != np.sign(real_minus_random) or (0.20 < frac_pos < 0.70) or (0.20 < frac_neg < 0.70):
        return "mixed_pair_scale_dependent"
    if np.isfinite(real_minus_fixed) and abs(real_minus_fixed) < 1e-6:
        return "no_efficiency_benefit"
    return "mixed_pair_scale_dependent"


def _run_fixrsvp_mode(
    model_bundle: ModelBundle,
    fixrsvp_trials: list[dict[str, object]],
    trace_cache: dict[tuple[int, str, int], np.ndarray],
    out_dir: Path,
    args: argparse.Namespace,
    dt_selected: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    trial_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    pilot_rows: list[dict[str, object]] = []

    pilot_trial_ids = {_safe_int(row["trial_index"]) for row in fixrsvp_trials[: int(args.pilot_trials)]}
    for frames_per_im in args.frames_per_im:
        frame_rows: list[dict[str, object]] = []
        for trial in fixrsvp_trials:
            for condition in args.trajectory_controls:
                repeats = _control_repeats(condition, args.n_random_controls)
                for repeat in range(repeats):
                    trace = trace_cache[(_safe_int(trial["trial_index"]), condition, int(repeat))]
                    row, rate_map = _fixrsvp_trial_metric(
                        model_bundle=model_bundle,
                        trial=trial,
                        trace=trace,
                        frames_per_im=int(frames_per_im),
                        dt=dt_selected,
                        n_lags=int(args.n_lags),
                        out_size=tuple(args.out_size),
                    )
                    condition_label = _condition_label(condition)
                    row.update(
                        {
                            "run_label": args.run_label,
                            "session": model_bundle.session_name,
                            "dataset_idx": int(args.dataset_idx),
                            "condition": condition_label,
                            "random_repeat": int(repeat),
                            "stim_type": "fixrsvp",
                            "frame": -1,
                            "frames_per_im": int(frames_per_im),
                            "n_lags": int(args.n_lags),
                            "out_h": int(args.out_size[0]),
                            "out_w": int(args.out_size[1]),
                            "dt_selected": float(dt_selected),
                        }
                    )
                    trial_rows.append(row)
                    frame_rows.append(row)

        aggregated_frame_rows = _aggregate_rows_mean(
            frame_rows,
            group_keys=("condition", "frames_per_im", "trial_index"),
            numeric_average_keys=(
                "cumulative_spatial_bits_per_expected_spike",
                "mean_spatial_bits_per_expected_spike",
                "median_spatial_bits_per_expected_spike",
                "mean_bits_per_sec",
                "total_bits",
                "mean_expected_spikes_per_bin",
                "total_expected_spikes",
            ),
        )

        for condition in sorted({str(row["condition"]) for row in aggregated_frame_rows if _safe_int(row["frames_per_im"]) == int(frames_per_im)}):
            rows = [row for row in aggregated_frame_rows if str(row["condition"]) == condition and _safe_int(row["frames_per_im"]) == int(frames_per_im)]
            values = np.asarray([_safe_float(row["cumulative_spatial_bits_per_expected_spike"]) for row in rows], dtype=np.float64)
            bits_sec = np.asarray([_safe_float(row["mean_bits_per_sec"]) for row in rows], dtype=np.float64)
            total_bits = np.asarray([_safe_float(row["total_bits"]) for row in rows], dtype=np.float64)
            total_spikes = np.asarray([_safe_float(row["total_expected_spikes"]) for row in rows], dtype=np.float64)
            mean_value, sem_value = _mean_sem(values)
            summary_rows.append(
                {
                    "session": model_bundle.session_name,
                    "dataset_idx": int(args.dataset_idx),
                    "condition": condition,
                    "frames_per_im": int(frames_per_im),
                    "n_trials": int(len(rows)),
                    "median_cumulative_spatial_bits_per_expected_spike": float(np.nanmedian(values)),
                    "mean_cumulative_spatial_bits_per_expected_spike": mean_value,
                    "sem_cumulative_spatial_bits_per_expected_spike": sem_value,
                    "median_bits_per_sec": float(np.nanmedian(bits_sec)),
                    "median_total_bits": float(np.nanmedian(total_bits)),
                    "median_total_expected_spikes": float(np.nanmedian(total_spikes)),
                }
            )

        for contrast_name, control in (
            ("real_FEM - fixed_center", "fixed_center"),
            ("real_FEM - stabilized", "stabilized"),
            ("real_FEM - random_amp", "random_amp"),
            ("real_FEM - random_cov", "random_cov"),
        ):
            real_rows = [row for row in aggregated_frame_rows if str(row["condition"]) == "real" and _safe_int(row["frames_per_im"]) == int(frames_per_im)]
            by_trial_control = {
                _safe_int(row["trial_index"]): _safe_float(row["cumulative_spatial_bits_per_expected_spike"])
                for row in aggregated_frame_rows
                if str(row["condition"]) == control and _safe_int(row["frames_per_im"]) == int(frames_per_im)
            }
            a = []
            b = []
            pilot_a = []
            pilot_b = []
            for row in real_rows:
                trial_index = _safe_int(row["trial_index"])
                if trial_index not in by_trial_control:
                    continue
                a.append(_safe_float(row["cumulative_spatial_bits_per_expected_spike"]))
                b.append(float(by_trial_control[trial_index]))
                if trial_index in pilot_trial_ids:
                    pilot_a.append(_safe_float(row["cumulative_spatial_bits_per_expected_spike"]))
                    pilot_b.append(float(by_trial_control[trial_index]))
            mean_delta, ci_low, ci_high = _paired_bootstrap_ci(np.asarray(a), np.asarray(b), rng=rng)
            contrast_rows.append(
                {
                    "analysis_mode": "fixrsvp_spatial_ssi",
                    "session": model_bundle.session_name,
                    "dataset_idx": int(args.dataset_idx),
                    "stimulus_axis": "frames_per_im",
                    "logmar": float("nan"),
                    "frames_per_im": int(frames_per_im),
                    "orientation_pair": "",
                    "contrast": contrast_name,
                    "metric": "cumulative_spatial_bits_per_expected_spike",
                    "median_delta": float(np.nanmedian(np.asarray(a) - np.asarray(b))) if a else float("nan"),
                    "mean_delta": mean_delta,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "n_trials_or_splits": int(len(a)),
                    "p_sign": float(np.mean((np.asarray(a) - np.asarray(b)) > 0)) if a else float("nan"),
                    "effect_status": "ok" if np.isfinite(mean_delta) else "insufficient_data",
                }
            )
            if pilot_a:
                pilot_mean, pilot_low, pilot_high = _paired_bootstrap_ci(np.asarray(pilot_a), np.asarray(pilot_b), rng=rng)
                effect = abs(pilot_mean)
                ci_width = pilot_high - pilot_low
                if len(pilot_a) < 10:
                    pilot_decision = "smoke_test_only"
                elif effect > 0 and ci_width / max(effect, EPS) > 2.0:
                    pilot_decision = "underpowered_add_trials"
                elif np.isfinite(effect) and effect < 1e-6:
                    pilot_decision = "likely_null_continue_or_stop"
                else:
                    pilot_decision = "proceed_full"
                pilot_rows.append(
                    {
                        "metric": "cumulative_bits_per_expected_spike",
                        "contrast": contrast_name,
                        "n_trials": int(len(pilot_a)),
                        "median_delta": float(np.nanmedian(np.asarray(pilot_a) - np.asarray(pilot_b))),
                        "ci_low": pilot_low,
                        "ci_high": pilot_high,
                        "ci_width": ci_width,
                        "abs_effect_size": effect,
                        "ci_width_over_abs_effect": ci_width / max(effect, EPS),
                        "pilot_decision": pilot_decision,
                    }
                )

    _write_csv(out_dir / "fixrsvp_spatial_ssi" / "fixrsvp_spatial_trial_metrics.csv", trial_rows)
    _write_csv(out_dir / "fixrsvp_spatial_ssi" / "fixrsvp_spatial_summary_by_condition.csv", summary_rows)
    _write_csv(out_dir / "pilot_power_summary.csv", pilot_rows)
    return trial_rows, summary_rows, contrast_rows, pilot_rows


def _legacy_alignment_capture_same_stimuli(
    grouped_rows: dict[tuple[str, float, int], list[dict[str, object]]],
    feature_store: dict[str, np.ndarray],
    logmar_values: tuple[float, ...],
    orientation_pairs: list[tuple[int, int]],
    conditions: list[str],
) -> tuple[list[dict[str, object]], bool]:
    rows: list[dict[str, object]] = []
    for condition in conditions:
        for logmar in logmar_values:
            for ori_a, ori_b in orientation_pairs:
                rows_a = grouped_rows.get((condition, float(logmar), int(ori_a)), [])
                rows_b = grouped_rows.get((condition, float(logmar), int(ori_b)), [])
                trial_ids_a = {_safe_int(row["trial_index"]) for row in rows_a}
                trial_ids_b = {_safe_int(row["trial_index"]) for row in rows_b}
                common_trial_ids = sorted(trial_ids_a & trial_ids_b)
                if len(common_trial_ids) < 4:
                    continue

                row_a_by_trial = {_safe_int(row["trial_index"]): row for row in rows_a}
                row_b_by_trial = {_safe_int(row["trial_index"]): row for row in rows_b}
                Xa = np.stack(
                    [feature_store[f"{row_a_by_trial[trial_id]['feature_key']}_time_mean"] for trial_id in common_trial_ids],
                    axis=0,
                ).astype(np.float64)
                Xb = np.stack(
                    [feature_store[f"{row_b_by_trial[trial_id]['feature_key']}_time_mean"] for trial_id in common_trial_ids],
                    axis=0,
                ).astype(np.float64)

                D = Xb - Xa
                ref = np.mean(D, axis=0)
                ref_norm = float(np.linalg.norm(ref))
                if ref_norm <= EPS:
                    median_signed = float("nan")
                    median_unsigned = float("nan")
                    median_capture = float("nan")
                else:
                    ref_u = ref / ref_norm
                    d_norm = np.linalg.norm(D, axis=1)
                    signed = (D @ ref_u) / np.maximum(d_norm, EPS)
                    unsigned = np.abs(signed)
                    proj = D @ ref_u
                    capture = (proj * proj) / np.maximum(np.sum(D * D, axis=1), EPS)
                    median_signed = float(np.nanmedian(signed))
                    median_unsigned = float(np.nanmedian(unsigned))
                    median_capture = float(np.nanmedian(capture))

                rows.append(
                    {
                        "condition": condition,
                        "logmar": float(logmar),
                        "orientation_pair": f"{ori_a}_vs_{ori_b}",
                        "n_trials": int(len(common_trial_ids)),
                        "median_signed_alignment": median_signed,
                        "median_unsigned_alignment": median_unsigned,
                        "median_capture": median_capture,
                    }
                )

    conds_present = {str(row.get("condition")) for row in rows}
    compared = bool(rows) and ("real" in conds_present) and ("stabilized" in conds_present)
    return rows, compared


def _renderer_parity_same_pipeline(
    args: argparse.Namespace,
    renderer: torch.nn.Module,
    retina: torch.nn.Module,
    traces: np.ndarray,
    durations: np.ndarray,
    usable: list[int],
) -> tuple[bool, dict[str, object]]:
    details: dict[str, object] = {
        "renderer_class": type(renderer).__name__,
        "retina_class": type(retina).__name__,
        "retina_size_requested": tuple(int(v) for v in args.out_size),
    }
    if not usable:
        details["reason"] = "no_usable_traces"
        return False, details

    from scripts.temporal_decoding import stimulus_hires as hires

    trace_idx = int(usable[0])
    T = int(min(durations[trace_idx], args.max_trial_frames))
    trace = traces[trace_idx, :T].astype(np.float32)
    test_logmar = float(args.logmar_values[0]) if args.logmar_values else -0.2
    test_ori = int(args.orientations[0]) if args.orientations else 0

    stim_local = _build_hires_eoptotype_stim(
        orientation_deg=test_ori,
        logmar=test_logmar,
        trace=trace,
        n_lags=int(args.n_lags),
        renderer=renderer,
        retina=retina,
    )
    stim_ref = hires.hires_counterfactual_stim(
        orientation_deg=test_ori,
        logmar=test_logmar,
        eyepos=trace,
        condition="real",
        n_lags=int(args.n_lags),
        retina_size=tuple(int(v) for v in args.out_size),
        device=str(args.device),
    )

    min_t = min(int(stim_local.shape[0]), int(stim_ref.shape[0]))
    diff = torch.abs(stim_local[:min_t] - stim_ref[:min_t]).detach().cpu().numpy().astype(np.float64)
    mad = float(np.mean(diff)) if diff.size else float("nan")
    mx = float(np.max(diff)) if diff.size else float("nan")

    details.update(
        {
            "stim_shape_local": tuple(int(v) for v in stim_local.shape),
            "stim_shape_reference": tuple(int(v) for v in stim_ref.shape),
            "mean_abs_diff": mad,
            "max_abs_diff": mx,
            "world_ppd_reference": float(getattr(hires, "WORLD_PPD", float("nan"))),
            "retina_ppd_reference": float(getattr(hires, "RETINA_PPD", float("nan")),),
            "polarity_reference": "world_gray = 127*(1-world)",
            "eye_sign_reference": "shift_y = -eye_y",
        }
    )
    matched = bool(np.isfinite(mad) and np.isfinite(mx) and (mad <= 1e-6) and (mx <= 1e-4))
    return matched, details


def _run_eoptotype_mode(
    model_bundle: ModelBundle,
    out_dir: Path,
    args: argparse.Namespace,
    dt_selected: float,
    rng: np.random.Generator,
    trace_qc_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], Path]:
    trial_rows: list[dict[str, object]] = []
    decoder_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []

    diagnostics_rows.append(
        {
            "check_name": "skip_eoptotype_passed",
            "status": "no",
            "value": 0,
            "detail": "--skip-eoptotype was not passed for this run.",
        }
    )

    if not Path(args.eye_traces).exists():
        diagnostics_rows.append(
            {
                "check_name": "eye_traces_available",
                "status": "fail",
                "value": 0,
                "detail": f"Eye trace file missing: {args.eye_traces}",
            }
        )
        _write_csv(out_dir / "eoptotype_failure_diagnostics.csv", diagnostics_rows)
        return trial_rows, decoder_rows, contrast_rows, out_dir / "eoptotype_identity" / "eoptotype_identity_features.npz"

    try:
        eye_data = np.load(args.eye_traces, allow_pickle=True)
    except Exception as exc:
        diagnostics_rows.append(
            {
                "check_name": "eye_traces_load",
                "status": "fail",
                "value": 0,
                "detail": f"Failed to load eye trace file: {exc}",
            }
        )
        _write_csv(out_dir / "eoptotype_failure_diagnostics.csv", diagnostics_rows)
        return trial_rows, decoder_rows, contrast_rows, out_dir / "eoptotype_identity" / "eoptotype_identity_features.npz"

    traces = eye_data["traces"].astype(np.float32)
    durations = eye_data["durations"].astype(np.int32)
    diagnostics_rows.append(
        {
            "check_name": "eye_traces_nonempty",
            "status": "pass" if traces.size > 0 else "fail",
            "value": int(traces.shape[0]) if traces.ndim > 0 else 0,
            "detail": "Eye trace array loaded from eye_traces.npz.",
        }
    )
    usable = [idx for idx in range(traces.shape[0]) if int(durations[idx]) >= int(args.min_fix_dur)]
    usable = usable[: int(args.max_trials)]
    diagnostics_rows.append(
        {
            "check_name": "usable_traces_after_min_fix_dur",
            "status": "pass" if len(usable) > 0 else "fail",
            "value": int(len(usable)),
            "detail": "Number of traces surviving min_fix_dur gate.",
        }
    )

    feature_store: dict[str, np.ndarray] = {}
    row_index = 0
    from scripts.temporal_decoding.stimulus_hires import HiResERenderer, HiResRetina

    try:
        renderer = HiResERenderer(device=args.device).to(args.device)
        retina = HiResRetina(retina_size=tuple(args.out_size)).to(args.device)
    except Exception as exc:
        diagnostics_rows.append(
            {
                "check_name": "hires_renderer_retina_init",
                "status": "fail",
                "value": 0,
                "detail": f"HiResERenderer/HiResRetina init failed: {exc}",
            }
        )
        _write_csv(out_dir / "eoptotype_failure_diagnostics.csv", diagnostics_rows)
        return trial_rows, decoder_rows, contrast_rows, out_dir / "eoptotype_identity" / "eoptotype_identity_features.npz"

    diagnostics_rows.append(
        {
            "check_name": "hires_renderer_retina_init",
            "status": "pass",
            "value": 1,
            "detail": "HiResERenderer and HiResRetina initialized successfully.",
        }
    )
    renderer.eval()
    retina.eval()

    renderer_matched, renderer_detail = _renderer_parity_same_pipeline(
        args=args,
        renderer=renderer,
        retina=retina,
        traces=traces,
        durations=durations,
        usable=usable,
    )
    renderer_status_value = "matched_hyperacuity_pipeline" if renderer_matched else "not_confirmed_hyperacuity_match"
    diagnostics_rows.append(
        {
            "check_name": "renderer_normalization_hyperacuity_match",
            "status": "pass" if renderer_matched else "warn",
            "value": 1 if renderer_matched else 0,
            "detail": json.dumps(renderer_detail, sort_keys=True),
        }
    )

    for condition in args.trajectory_controls:
        repeats = _control_repeats(condition, args.n_random_controls)
        for logmar in args.logmar_values:
            for orientation in args.orientations:
                for trace_idx in usable:
                    T = int(min(durations[trace_idx], args.max_trial_frames))
                    base_trace = traces[trace_idx, :T]
                    for repeat in range(repeats):
                        conditioned_trace, _ = _trajectory_for_condition(base_trace, condition, rng=rng, sigma_frames=args.random_sigma_frames)
                        stim = _build_hires_eoptotype_stim(
                            orientation_deg=int(orientation),
                            logmar=float(logmar),
                            trace=conditioned_trace.astype(np.float32),
                            n_lags=int(args.n_lags),
                            renderer=renderer,
                            retina=retina,
                        )
                        y, clamped_negative = _compute_rate_maps(model_bundle, stim)
                        rate_map = y.reshape(y.shape[0], y.shape[1], -1).mean(dim=2).detach().cpu().numpy()
                        # Spatial-map-preserving ablation feature: pooled spatial energy map.
                        spatial_energy_map = torch.mean(y * y, dim=(0, 1), keepdim=True)
                        pooled = F.adaptive_avg_pool2d(spatial_energy_map, output_size=(16, 16))
                        spatial_map = pooled.reshape(-1).detach().cpu().numpy().astype(np.float32)
                        metrics = _population_spike_metrics(y, dt=dt_selected)
                        feature_reps = _eoptotype_feature_representations(rate_map)
                        feature_key = f"feat_{row_index:06d}"
                        feature_store[feature_key + "_time_mean"] = feature_reps["time_mean"]
                        feature_store[feature_key + "_late_mean"] = feature_reps["late_mean"]
                        feature_store[feature_key + "_time_concat"] = feature_reps["time_concat"]
                        feature_store[feature_key + "_energy"] = feature_reps["energy"]
                        feature_store[feature_key + "_spatial_map"] = spatial_map
                        feature_store[feature_key + "_spatial_avg_time_mean"] = feature_reps["time_mean"]
                        feature_store[feature_key + "_spatial_avg_energy"] = feature_reps["energy"]
                        feature_store[feature_key + "_map_energy"] = spatial_map
                        feature_store[feature_key + "_map_lowrank_pca"] = spatial_map
                        for d1_window in DEFAULT_D1_INTEGRATION_WINDOWS:
                            window = max(1, min(int(d1_window), int(rate_map.shape[0])))
                            feature_store[feature_key + f"_d1_time_mean_w{int(d1_window)}"] = np.mean(
                                rate_map[-window:, :],
                                axis=0,
                            ).astype(np.float32)

                        trial_rows.append(
                            {
                                "run_label": args.run_label,
                                "session": model_bundle.session_name,
                                "dataset_idx": int(args.dataset_idx),
                                "trial_index": int(trace_idx),
                                "condition": _condition_label(condition),
                                "random_repeat": int(repeat),
                                "logmar": float(logmar),
                                "orientation": int(orientation),
                                "n_lags": int(args.n_lags),
                                "out_h": int(args.out_size[0]),
                                "out_w": int(args.out_size[1]),
                                "dt_selected": float(dt_selected),
                                "n_time_bins": int(T),
                                "n_units": int(y.shape[1]),
                                "total_expected_spikes": float(metrics["total_expected_spikes"]),
                                "mean_expected_spikes_per_bin": float(metrics["mean_expected_spikes_per_bin"]),
                                "cumulative_spatial_bits_per_expected_spike": float(metrics["cumulative_spatial_bits_per_expected_spike"]),
                                "feature_representation": "time_mean|late_mean|time_concat|energy",
                                "status": "ok",
                                "feature_key": feature_key,
                                "clamped_negative_rates": bool(clamped_negative),
                            }
                        )
                        row_index += 1

    aggregated_trial_rows, aggregated_feature_store = _aggregate_eoptotype_trial_rows(trial_rows, feature_store)

    pairs = _pairwise_orientation_pairs(args.orientations)
    feature_representations = tuple(DEFAULT_EOPTOTYPE_FEATURE_REPRESENTATIONS)
    readouts_for_eval: tuple[str, ...] = tuple(dict.fromkeys([*args.readouts]))
    grouped_rows: dict[tuple[str, float, int], list[dict[str, object]]] = defaultdict(list)
    for row in aggregated_trial_rows:
        grouped_rows[(str(row["condition"]), _safe_float(row["logmar"]), _safe_int(row["orientation"]))].append(row)

    # Fit a compact low-rank basis once on map-energy features for optional spatial-map representation.
    map_keys = [key for key in aggregated_feature_store if key.endswith("_map_energy")]
    if map_keys:
        matrix = np.stack([aggregated_feature_store[key] for key in map_keys], axis=0).astype(np.float64)
        mean_vec = np.mean(matrix, axis=0, keepdims=True)
        centered = matrix - mean_vec
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        rank = int(min(16, vt.shape[0]))
        basis = vt[:rank]
        for key in map_keys:
            base = key[: -len("_map_energy")]
            vec = aggregated_feature_store[key].astype(np.float64) - mean_vec.reshape(-1)
            lowrank = vec @ basis.T
            aggregated_feature_store[base + "_map_lowrank_pca"] = lowrank.astype(np.float32)

    decoder_nlt4_skips = 0
    decoder_tasks: list[tuple[str, float, int, int, str, str]] = []
    conditions_for_decoder = tuple(sorted({_condition_label(cond) for cond in args.trajectory_controls}))
    for logmar in args.logmar_values:
        for ori_a, ori_b in pairs:
            for readout_type in readouts_for_eval:
                for feature_name in feature_representations:
                    for condition in conditions_for_decoder:
                        decoder_tasks.append((condition, float(logmar), int(ori_a), int(ori_b), readout_type, feature_name))

    decoder_builder = partial(
        _build_eoptotype_decoder_row,
        grouped_rows=grouped_rows,
        aggregated_feature_store=aggregated_feature_store,
        random_seed=int(args.random_seed),
        renderer_status_value=renderer_status_value,
    )
    if int(args.decoder_workers) <= 1:
        decoder_results = [decoder_builder(task) for task in decoder_tasks]
    else:
        with ThreadPoolExecutor(max_workers=int(args.decoder_workers)) as executor:
            decoder_results = list(executor.map(decoder_builder, decoder_tasks))

    for row, skipped in decoder_results:
        decoder_nlt4_skips += int(skipped)
        if row is not None:
            decoder_rows.append(row)

    decoder_lookup: dict[tuple[str, float, str, str, str], dict[str, object]] = {
        (
            str(row["condition"]),
            _safe_float(row["logmar"]),
            str(row["orientation_pair"]),
            str(row["readout_type"]),
            str(row.get("feature_representation", "")),
        ): row
        for row in decoder_rows
    }
    for row in decoder_rows:
        if str(row["condition"]) != "real":
            continue
        key_base = (
            _safe_float(row["logmar"]),
            str(row["orientation_pair"]),
            str(row["readout_type"]),
            str(row.get("feature_representation", "")),
        )
        value_real = _safe_float(row["d2_per_expected_spike"])
        for control_name, field_name in (
            ("fixed_center", "real_minus_fixed_d2_per_expected_spike"),
            ("stabilized", "real_minus_stabilized_d2_per_expected_spike"),
            ("random_amp", "real_minus_random_amp_d2_per_expected_spike"),
            ("random_cov", "real_minus_random_cov_d2_per_expected_spike"),
        ):
            control = decoder_lookup.get((control_name, *key_base))
            if control is not None:
                row[field_name] = value_real - _safe_float(control["d2_per_expected_spike"])
        for control_name, field_name in (
            ("fixed_center", "real_minus_fixed_poisson_budget_accuracy"),
            ("stabilized", "real_minus_stabilized_poisson_budget_accuracy"),
            ("random_amp", "real_minus_random_amp_poisson_budget_accuracy"),
            ("random_cov", "real_minus_random_cov_poisson_budget_accuracy"),
        ):
            control = decoder_lookup.get((control_name, *key_base))
            if control is not None:
                row[field_name] = _safe_float(row["poisson_budget_accuracy"]) - _safe_float(control["poisson_budget_accuracy"])
        for control_name, field_name in (
            ("fixed_center", "real_minus_fixed_rate_normalized_decoder_accuracy"),
            ("stabilized", "real_minus_stabilized_rate_normalized_decoder_accuracy"),
            ("random_amp", "real_minus_random_amp_rate_normalized_decoder_accuracy"),
            ("random_cov", "real_minus_random_cov_rate_normalized_decoder_accuracy"),
        ):
            control = decoder_lookup.get((control_name, *key_base))
            if control is not None:
                row[field_name] = _safe_float(row["rate_normalized_decoder_accuracy"]) - _safe_float(control["rate_normalized_decoder_accuracy"])

    deterministic_summary_rows, deterministic_lookup, deterministic_vector_lookup = _compute_deterministic_identity_summary_rows(
        trial_rows=trial_rows,
        feature_store=feature_store,
        feature_representations=feature_representations,
        readout_types=readouts_for_eval,
        logmar_values=tuple(float(v) for v in args.logmar_values),
        orientations=tuple(int(v) for v in args.orientations),
    )
    response_vectors_path, response_vectors_schema_path = _persist_eoptotype_response_vectors(
        out_dir=out_dir,
        trial_rows=trial_rows,
        feature_store=feature_store,
        feature_representations=feature_representations,
    )

    for row in decoder_rows:
        key = (
            str(row["condition"]),
            _safe_float(row["logmar"]),
            str(row["orientation_pair"]),
            str(row["readout_type"]),
            str(row.get("feature_representation", "")),
        )
        det_row = deterministic_lookup.get(key)
        if det_row is None:
            continue
        row["deterministic_identity_bits_per_expected_spike"] = _safe_float(det_row["deterministic_identity_bits_per_expected_spike"])
        row["pairwise_deterministic_identity_bits_per_expected_spike"] = _safe_float(det_row["pairwise_deterministic_identity_bits_per_expected_spike"])
        row["real_minus_fixed_identity_bits_per_expected_spike"] = float("nan")
        row["real_minus_stabilized_identity_bits_per_expected_spike"] = float("nan")
        row["real_minus_random_amp_identity_bits_per_expected_spike"] = float("nan")
        row["real_minus_random_cov_identity_bits_per_expected_spike"] = float("nan")
        if str(row.get("condition")) == "real":
            key_base = (
                _safe_float(row.get("logmar")),
                str(row.get("orientation_pair")),
                str(row.get("readout_type")),
                str(row.get("feature_representation", "")),
            )
            real_bits = _safe_float(row.get("deterministic_identity_bits_per_expected_spike"))
            for control_name, field_name in (
                ("fixed_center", "real_minus_fixed_identity_bits_per_expected_spike"),
                ("stabilized", "real_minus_stabilized_identity_bits_per_expected_spike"),
                ("random_amp", "real_minus_random_amp_identity_bits_per_expected_spike"),
                ("random_cov", "real_minus_random_cov_identity_bits_per_expected_spike"),
            ):
                control_row = deterministic_lookup.get((control_name, key_base[0], key_base[1], key_base[2], key_base[3]))
                if control_row is not None:
                    row[field_name] = real_bits - _safe_float(control_row.get("deterministic_identity_bits_per_expected_spike"))

    d1_rows = _compute_d1_window_decoder_rows(
        aggregated_trial_rows=aggregated_trial_rows,
        aggregated_feature_store=aggregated_feature_store,
        conditions_for_decoder=conditions_for_decoder,
        logmar_values=tuple(float(v) for v in args.logmar_values),
        orientation_pairs=pairs,
        d1_windows=tuple(int(v) for v in DEFAULT_D1_INTEGRATION_WINDOWS),
        random_seed=int(args.random_seed),
    )
    _write_csv(out_dir / "eoptotype_D1_integration_window_sweep_base.csv", d1_rows)

    _write_csv(out_dir / "eoptotype_identity_bits_per_spike_summary.csv", deterministic_summary_rows)

    for logmar in args.logmar_values:
        for readout_type in readouts_for_eval:
            for feature_name in feature_representations:
                for pair in [f"{a}_vs_{b}" for a, b in pairs]:
                    real = decoder_lookup.get(("real", float(logmar), pair, readout_type, feature_name))
                    if real is None:
                        continue
                    for control_name, label in (
                        ("fixed_center", "real_FEM - fixed_center"),
                        ("stabilized", "real_FEM - stabilized"),
                        ("random_amp", "real_FEM - random_amp"),
                        ("random_cov", "real_FEM - random_cov"),
                    ):
                        control = decoder_lookup.get((control_name, float(logmar), pair, readout_type, feature_name))
                        if control is None:
                            continue
                        delta = _safe_float(real["d2_per_expected_spike"]) - _safe_float(control["d2_per_expected_spike"])
                        contrast_rows.append(
                            {
                                "analysis_mode": "eoptotype_identity",
                                "session": model_bundle.session_name,
                                "dataset_idx": int(args.dataset_idx),
                                "stimulus_axis": "logmar",
                                "logmar": float(logmar),
                                "frames_per_im": float("nan"),
                                "orientation_pair": pair,
                                "contrast": label,
                                "metric": f"d2_per_expected_spike:{readout_type}:{feature_name}",
                                "median_delta": delta,
                                "mean_delta": delta,
                                "bootstrap_ci_low": float("nan"),
                                "bootstrap_ci_high": float("nan"),
                                "n_trials_or_splits": _safe_int(real["n_splits"]),
                                "p_sign": float("nan"),
                                "effect_status": "split_level_estimate",
                            }
                        )

    qc_by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in decoder_rows:
        condition = str(row.get("condition"))
        if condition in {"random_amp", "random_cov"}:
            qc_by_condition[condition].append(row)

    for condition in ("random_amp", "random_cov"):
        rows = qc_by_condition[condition]
        rms_med = _nanmedian_or_nan(_safe_float(row.get("matched_rms_error")) for row in rows)
        cov_med = _nanmedian_or_nan(_safe_float(row.get("matched_cov_error")) for row in rows)
        path_med = _nanmedian_or_nan(_safe_float(row.get("path_length_error")) for row in rows)
        acf1_med = _nanmedian_or_nan(_safe_float(row.get("acf_lag1_error")) for row in rows)
        acf2_med = _nanmedian_or_nan(_safe_float(row.get("acf_lag2_error")) for row in rows)
        acf4_med = _nanmedian_or_nan(_safe_float(row.get("acf_lag4_error")) for row in rows)
        rms_cov_ok = np.isfinite(rms_med) and rms_med <= 0.15 and np.isfinite(cov_med) and cov_med <= 0.20
        temporal_ok = (
            np.isfinite(path_med)
            and path_med <= DEFAULT_RANDOM_CONTROL_PATH_ERROR_MAX
            and np.isfinite(acf1_med)
            and acf1_med <= DEFAULT_RANDOM_CONTROL_ACF_ERROR_MAX
            and np.isfinite(acf2_med)
            and acf2_med <= DEFAULT_RANDOM_CONTROL_ACF_ERROR_MAX
            and np.isfinite(acf4_med)
            and acf4_med <= DEFAULT_RANDOM_CONTROL_ACF_ERROR_MAX
        )
        status = "fully_matched" if rms_cov_ok and temporal_ok else ("covariance_matched_only" if rms_cov_ok else "not_matched")
        diagnostics_rows.append(
            {
                "check_name": f"{condition}_control_matching_scope",
                "status": "pass" if status == "fully_matched" else "warn",
                "value": status,
                "detail": (
                    f"rms={rms_med:.6g}, cov={cov_med:.6g}, path={path_med:.6g}, "
                    f"acf1={acf1_med:.6g}, acf2={acf2_med:.6g}, acf4={acf4_med:.6g}"
                ),
            }
        )

    linear_real = [
        row
        for row in decoder_rows
        if str(row.get("condition")) == "real"
        and str(row.get("readout_type")) == "linear"
        and str(row.get("feature_representation")) == "spatial_avg_time_mean"
    ]
    spatial_real = [
        row
        for row in decoder_rows
        if str(row.get("condition")) == "real"
        and str(row.get("readout_type")) == "linear"
        and str(row.get("feature_representation")) == "map_energy"
    ]
    lin_fix = _nanmedian_or_nan(_safe_float(row.get("real_minus_fixed_d2_per_expected_spike")) for row in linear_real)
    lin_cov = _nanmedian_or_nan(_safe_float(row.get("real_minus_random_cov_d2_per_expected_spike")) for row in linear_real)
    spa_fix = _nanmedian_or_nan(_safe_float(row.get("real_minus_fixed_d2_per_expected_spike")) for row in spatial_real)
    spa_cov = _nanmedian_or_nan(_safe_float(row.get("real_minus_random_cov_d2_per_expected_spike")) for row in spatial_real)
    ablation_sign_consistent = (
        np.isfinite(lin_fix)
        and np.isfinite(lin_cov)
        and np.isfinite(spa_fix)
        and np.isfinite(spa_cov)
        and (np.sign(lin_fix) == np.sign(spa_fix))
        and (np.sign(lin_cov) == np.sign(spa_cov))
    )
    diagnostics_rows.append(
        {
            "check_name": "spatial_map_ablation_sign_consistency",
            "status": "pass" if ablation_sign_consistent else "warn",
            "value": int(bool(ablation_sign_consistent)),
            "detail": f"linear: (fixed={lin_fix:.6g}, random_cov={lin_cov:.6g}); spatial_map: (fixed={spa_fix:.6g}, random_cov={spa_cov:.6g})",
        }
    )

    legacy_status_value = "skipped_current_renderer_only"
    for row in decoder_rows:
        row["legacy_alignment_capture_status"] = legacy_status_value

    diagnostics_rows.append(
        {
            "check_name": "legacy_alignment_capture_same_stimuli",
            "status": "info",
            "value": 0,
            "detail": "Skipped for this cleanup pass: current-renderer Poisson d2 analysis only.",
        }
    )

    # Required audit output for condition x logmar x pair x readout x feature.
    audit_fieldnames = [
        "condition",
        "logmar",
        "orientation_pair",
        "readout_type",
        "feature_representation",
        "poisson_d2",
        "mean_total_expected_spikes",
        "d2_per_expected_spike",
        "deterministic_identity_bits_per_expected_spike",
        "pairwise_deterministic_identity_bits_per_expected_spike",
        "poisson_budget_accuracy",
        "rate_normalized_decoder_accuracy",
        "deterministic_decoder_accuracy_qc",
        "real_minus_fixed_d2_per_expected_spike",
        "real_minus_stabilized_d2_per_expected_spike",
        "real_minus_random_amp_d2_per_expected_spike",
        "real_minus_random_cov_d2_per_expected_spike",
        "real_minus_fixed_identity_bits_per_expected_spike",
        "real_minus_stabilized_identity_bits_per_expected_spike",
        "real_minus_random_amp_identity_bits_per_expected_spike",
        "real_minus_random_cov_identity_bits_per_expected_spike",
    ]
    if decoder_rows:
        existing = set(audit_fieldnames)
        for row in decoder_rows:
            for key in row.keys():
                key_str = str(key)
                if key_str not in existing:
                    audit_fieldnames.append(key_str)
                    existing.add(key_str)
    _write_csv(
        out_dir / "eoptotype_identity_efficiency_audit_summary.csv",
        decoder_rows,
        fieldnames=audit_fieldnames,
    )

    # Numerator/denominator decomposition for real-vs-control deltas.
    decomp_rows: list[dict[str, object]] = []
    for row in decoder_rows:
        if str(row.get("condition")) != "real":
            continue
        key = (
            _safe_float(row.get("logmar")),
            str(row.get("orientation_pair")),
            str(row.get("readout_type")),
            str(row.get("feature_representation")),
        )
        controls = {
            str(r.get("condition")): r
            for r in decoder_rows
            if (
                _safe_float(r.get("logmar")) == key[0]
                and str(r.get("orientation_pair")) == key[1]
                and str(r.get("readout_type")) == key[2]
                and str(r.get("feature_representation")) == key[3]
            )
        }
        for control_name in ("fixed_center", "stabilized", "random_amp", "random_cov"):
            control = controls.get(control_name)
            if control is None:
                continue
            decomp_rows.append(
                {
                    "logmar": key[0],
                    "orientation_pair": key[1],
                    "readout_type": key[2],
                    "feature_representation": key[3],
                    "contrast": f"real_vs_{control_name}",
                    "delta_poisson_d2": _safe_float(row.get("poisson_d2")) - _safe_float(control.get("poisson_d2")),
                    "delta_expected_spikes": _safe_float(row.get("mean_total_expected_spikes")) - _safe_float(control.get("mean_total_expected_spikes")),
                    "delta_d2_per_expected_spike": _safe_float(row.get("d2_per_expected_spike")) - _safe_float(control.get("d2_per_expected_spike")),
                }
            )
    _write_csv(out_dir / "eoptotype_identity_numerator_denominator_decomposition.csv", decomp_rows)

    # Feature-ablation summary asks whether sign survives spatial-map-preserving features.
    feature_rows: list[dict[str, object]] = []
    real_rows = [row for row in decoder_rows if str(row.get("condition")) == "real"]
    group_keys = sorted(
        {
            (
                _safe_float(row.get("logmar")),
                str(row.get("orientation_pair")),
                str(row.get("readout_type")),
            )
            for row in real_rows
        }
    )
    for logmar, orientation_pair, readout_type in group_keys:
        by_feature = {
            str(row.get("feature_representation")): row
            for row in real_rows
            if _safe_float(row.get("logmar")) == logmar
            and str(row.get("orientation_pair")) == orientation_pair
            and str(row.get("readout_type")) == readout_type
        }
        anchor = by_feature.get("spatial_avg_time_mean")
        if anchor is None:
            continue
        anchor_fix = _safe_float(anchor.get("real_minus_fixed_d2_per_expected_spike"))
        anchor_cov = _safe_float(anchor.get("real_minus_random_cov_d2_per_expected_spike"))
        for feature_name, row in by_feature.items():
            val_fix = _safe_float(row.get("real_minus_fixed_d2_per_expected_spike"))
            val_cov = _safe_float(row.get("real_minus_random_cov_d2_per_expected_spike"))
            feature_rows.append(
                {
                    "logmar": logmar,
                    "orientation_pair": orientation_pair,
                    "readout_type": readout_type,
                    "feature_representation": feature_name,
                    "real_minus_fixed_d2_per_expected_spike": val_fix,
                    "real_minus_random_cov_d2_per_expected_spike": val_cov,
                    "sign_matches_anchor_fixed": int(np.isfinite(anchor_fix) and np.isfinite(val_fix) and (np.sign(anchor_fix) == np.sign(val_fix))),
                    "sign_matches_anchor_random_cov": int(np.isfinite(anchor_cov) and np.isfinite(val_cov) and (np.sign(anchor_cov) == np.sign(val_cov))),
                }
            )
    _write_csv(out_dir / "eoptotype_identity_feature_ablation_summary.csv", feature_rows)

    # Summarize random-control QC with median and 95% ranges.
    qc_metrics = (
        "matched_rms_error",
        "matched_cov_error",
        "path_length_error",
        "acf_lag1_error",
        "acf_lag2_error",
        "acf_lag4_error",
    )
    qc_rows: list[dict[str, object]] = []
    for condition in ("random_amp", "random_cov"):
        cond_rows = [row for row in trace_qc_rows if str(row.get("condition")) == condition]
        for metric in qc_metrics:
            vals = [_safe_float(row.get(metric)) for row in cond_rows]
            if metric == "matched_rms_error":
                threshold = DEFAULT_RANDOM_CONTROL_RMS_ERROR_MAX
            elif metric == "matched_cov_error":
                threshold = DEFAULT_RANDOM_CONTROL_COV_ERROR_MAX
            elif metric == "path_length_error":
                threshold = DEFAULT_RANDOM_CONTROL_PATH_ERROR_MAX
            else:
                threshold = DEFAULT_RANDOM_CONTROL_ACF_ERROR_MAX
            median = _nanmedian_or_nan(vals)
            q025 = _nanpercentile_or_nan(vals, 2.5)
            q975 = _nanpercentile_or_nan(vals, 97.5)
            qc_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "median": median,
                    "q2p5": q025,
                    "q97p5": q975,
                    "threshold": threshold,
                    "status": "pass" if np.isfinite(median) and median <= threshold else "warn",
                }
            )
    _write_csv(out_dir / "eoptotype_identity_random_control_qc_summary.csv", qc_rows)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Real-minus-control vs LogMAR (primary metric).
    focus_rows = [
        row
        for row in decoder_rows
        if str(row.get("condition")) == "real"
        and str(row.get("readout_type")) == "linear"
        and str(row.get("feature_representation")) == "spatial_avg_time_mean"
    ]
    if focus_rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        for field, label in (
            ("real_minus_fixed_d2_per_expected_spike", "real-fixed"),
            ("real_minus_random_cov_d2_per_expected_spike", "real-random_cov"),
        ):
            xs = sorted({_safe_float(row.get("logmar")) for row in focus_rows})
            ys = [
                _nanmedian_or_nan(
                    _safe_float(row.get(field))
                    for row in focus_rows
                    if _safe_float(row.get("logmar")) == x
                )
                for x in xs
            ]
            ax.plot(xs, ys, marker="o", label=label)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("LogMAR")
        ax.set_ylabel("delta d2 / expected spike")
        ax.set_title("Real-minus-control d2 per expected spike")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "eoptotype_real_minus_controls_vs_logmar.png", dpi=200)
        plt.close(fig)

        # Orientation-pair heatmap for real-random_cov.
        logmars = sorted({_safe_float(row.get("logmar")) for row in focus_rows})
        pairs_h = sorted({str(row.get("orientation_pair")) for row in focus_rows})
        heat = np.full((len(logmars), len(pairs_h)), np.nan, dtype=np.float64)
        for i, lm in enumerate(logmars):
            for j, pair in enumerate(pairs_h):
                vals = [
                    _safe_float(row.get("real_minus_random_cov_d2_per_expected_spike"))
                    for row in focus_rows
                    if _safe_float(row.get("logmar")) == lm and str(row.get("orientation_pair")) == pair
                ]
                heat[i, j] = _nanmedian_or_nan(vals)
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(heat, aspect="auto", cmap="coolwarm", interpolation="nearest")
        ax.set_xticks(np.arange(len(pairs_h)))
        ax.set_xticklabels(pairs_h, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(logmars)))
        ax.set_yticklabels([f"{lm:.2f}" for lm in logmars])
        ax.set_xlabel("Orientation pair")
        ax.set_ylabel("LogMAR")
        ax.set_title("real-random_cov d2/expected spike")
        fig.colorbar(im, ax=ax, label="delta d2/expected spike")
        fig.tight_layout()
        fig.savefig(fig_dir / "eoptotype_real_minus_random_cov_heatmap.png", dpi=200)
        plt.close(fig)

    # Numerator vs denominator decomposition scatter.
    if decomp_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = np.asarray([_safe_float(row.get("delta_expected_spikes")) for row in decomp_rows], dtype=np.float64)
        ys = np.asarray([_safe_float(row.get("delta_poisson_d2")) for row in decomp_rows], dtype=np.float64)
        ax.scatter(xs, ys, s=20, alpha=0.7)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("delta expected spikes (real-control)")
        ax.set_ylabel("delta poisson d2 (real-control)")
        ax.set_title("Numerator vs denominator decomposition")
        fig.tight_layout()
        fig.savefig(fig_dir / "eoptotype_numerator_denominator_scatter.png", dpi=200)
        plt.close(fig)

    # Feature-ablation comparison figure.
    if feature_rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        features = sorted({str(row.get("feature_representation")) for row in feature_rows})
        vals_fix = [
            _nanmedian_or_nan(
                _safe_float(row.get("real_minus_fixed_d2_per_expected_spike"))
                for row in feature_rows
                if str(row.get("feature_representation")) == feature
            )
            for feature in features
        ]
        vals_cov = [
            _nanmedian_or_nan(
                _safe_float(row.get("real_minus_random_cov_d2_per_expected_spike"))
                for row in feature_rows
                if str(row.get("feature_representation")) == feature
            )
            for feature in features
        ]
        x = np.arange(len(features), dtype=np.float64)
        ax.bar(x - 0.18, vals_fix, width=0.36, label="real-fixed")
        ax.bar(x + 0.18, vals_cov, width=0.36, label="real-random_cov")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=20, ha="right")
        ax.set_ylabel("median delta d2 / expected spike")
        ax.set_title("Feature-ablation comparison")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "eoptotype_feature_ablation_comparison.png", dpi=200)
        plt.close(fig)

    # Random-control QC summary figure.
    if qc_rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f"{row['condition']}:{row['metric']}" for row in qc_rows]
        x = np.arange(len(qc_rows), dtype=np.float64)
        med = np.asarray([_safe_float(row.get("median")) for row in qc_rows], dtype=np.float64)
        lo = np.asarray([_safe_float(row.get("q2p5")) for row in qc_rows], dtype=np.float64)
        hi = np.asarray([_safe_float(row.get("q97p5")) for row in qc_rows], dtype=np.float64)
        yerr = np.vstack([np.maximum(med - lo, 0.0), np.maximum(hi - med, 0.0)])
        ax.errorbar(x, med, yerr=yerr, fmt="o", capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right")
        ax.set_ylabel("error metric")
        ax.set_title("Random-control QC median with 95% range")
        fig.tight_layout()
        fig.savefig(fig_dir / "eoptotype_random_control_qc_summary.png", dpi=200)
        plt.close(fig)

    _write_eoptotype_reconciliation_bundle(
        out_dir=out_dir,
        decoder_rows=decoder_rows,
        d1_rows=d1_rows,
        trace_qc_rows=trace_qc_rows,
    )

    features_path = out_dir / "eoptotype_identity" / "eoptotype_identity_features.npz"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(features_path, **aggregated_feature_store)
    _write_csv(out_dir / "eoptotype_identity" / "eoptotype_identity_trial_metrics.csv", trial_rows)
    _write_csv(out_dir / "eoptotype_identity" / "eoptotype_identity_decoder_metrics.csv", decoder_rows)

    diagnostics_rows.append(
        {
            "check_name": "decoder_n_lt_4_group_skips",
            "status": "info",
            "value": int(decoder_nlt4_skips),
            "detail": "Count of decoder group combinations skipped because n < 4.",
        }
    )
    diagnostics_rows.append(
        {
            "check_name": "decoder_rows_generated",
            "status": "pass" if len(decoder_rows) > 0 else "fail",
            "value": int(len(decoder_rows)),
            "detail": "Total eoptotype decoder rows generated.",
        }
    )
    if len(decoder_rows) == 0:
        diagnostics_rows.append(
            {
                "check_name": "likely_zero_decoder_cause",
                "status": "warn",
                "value": int(decoder_nlt4_skips),
                "detail": "No decoder rows generated; inspect usable traces, min_fix_dur gate, and n<4 skip count.",
            }
        )
    _write_csv(out_dir / "eoptotype_failure_diagnostics.csv", diagnostics_rows)
    return trial_rows, decoder_rows, contrast_rows, features_path


def _write_fixrsvp_figures(out_dir: Path, summary_rows: list[dict[str, object]], contrast_rows: list[dict[str, object]]) -> None:
    fig_dir = out_dir / "figures"
    _plot_box_or_line(
        summary_rows,
        fig_dir / "fixrsvp_spatial_efficiency_by_condition.png",
        x_key="condition",
        y_key="median_cumulative_spatial_bits_per_expected_spike",
        hue_key="frames_per_im",
        title="FixRSVP cumulative spatial bits per expected spike",
        ylabel="bits / expected spike",
    )
    _plot_box_or_line(
        [row for row in contrast_rows if row["analysis_mode"] == "fixrsvp_spatial_ssi"],
        fig_dir / "fixrsvp_real_minus_control_contrasts.png",
        x_key="contrast",
        y_key="mean_delta",
        hue_key="frames_per_im",
        title="FixRSVP real-minus-control contrasts",
        ylabel="delta bits / expected spike",
    )


def _write_eoptotype_figures(out_dir: Path, decoder_rows: list[dict[str, object]], deterministic_summary_rows: list[dict[str, object]] | None = None) -> None:
    fig_dir = out_dir / "figures"
    _plot_box_or_line(
        decoder_rows,
        fig_dir / "eoptotype_identity_efficiency_by_logmar.png",
        x_key="logmar",
        y_key="d2_per_expected_spike",
        hue_key="condition",
        title="E-optotype Poisson d2 per expected spike",
        ylabel="d2 / expected spike",
    )
    _plot_box_or_line(
        decoder_rows,
        fig_dir / "eoptotype_identity_vs_expected_spikes.png",
        x_key="condition",
        y_key="mean_total_expected_spikes",
        hue_key="logmar",
        title="Expected spikes by condition and LogMAR",
        ylabel="expected spikes",
    )

    contrast_subset = [row for row in decoder_rows if np.isfinite(_safe_float(row.get("real_minus_random_cov_d2_per_expected_spike", float("nan"))))]
    if contrast_subset:
        fig, ax = plt.subplots(figsize=(10, 5))
        logmars = sorted({_safe_float(row["logmar"]) for row in contrast_subset})
        pairs = sorted({str(row["orientation_pair"]) for row in contrast_subset})
        for pair in pairs:
            ys = []
            for logmar in logmars:
                vals = [_safe_float(row["real_minus_random_cov_d2_per_expected_spike"]) for row in contrast_subset if str(row["orientation_pair"]) == pair and _safe_float(row["logmar"]) == logmar]
                ys.append(float(np.nanmedian(vals)) if vals else float("nan"))
            ax.plot(logmars, ys, marker="o", label=pair)
        ax.set_xlabel("LogMAR")
        ax.set_ylabel("real - random_cov d2 / expected spike")
        ax.set_title("Hypothesis E load-bearing contrast")
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / "eoptotype_real_minus_random_cov.png", dpi=200)
        plt.close(fig)

    if deterministic_summary_rows:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
        features = sorted({str(row.get("feature_representation")) for row in deterministic_summary_rows})
        conditions = sorted({str(row.get("condition")) for row in deterministic_summary_rows})
        axes_flat = axes.reshape(-1)
        for idx, feature in enumerate(features[: len(axes_flat)]):
            ax = axes_flat[idx]
            for condition in conditions:
                rows = [
                    row
                    for row in deterministic_summary_rows
                    if str(row.get("feature_representation")) == feature
                    and str(row.get("condition")) == condition
                    and str(row.get("orientation_pair")) == "all_orientations"
                    and np.isfinite(_safe_float(row.get("deterministic_identity_bits_per_expected_spike")))
                ]
                if not rows:
                    continue
                xs = sorted({_safe_float(row.get("logmar")) for row in rows})
                ys = [
                    _nanmedian_or_nan(
                        _safe_float(row.get("deterministic_identity_bits_per_expected_spike"))
                        for row in rows
                        if _safe_float(row.get("logmar")) == x
                    )
                    for x in xs
                ]
                ax.plot(xs, ys, marker="o", label=condition)
            ax.set_title(feature)
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
            ax.set_xlabel("LogMAR")
            ax.set_ylabel("deterministic bits / expected spike")
            ax.legend(frameon=False, fontsize=8)
        for ax in axes_flat[len(features):]:
            ax.axis("off")
        fig.suptitle("Deterministic identity bits per expected spike by LogMAR")
        fig.tight_layout()
        fig.savefig(fig_dir / "eoptotype_identity_bits_per_spike_by_logmar.png", dpi=200)
        plt.close(fig)


def _write_combined_outputs(
    out_dir: Path,
    model_bundle: ModelBundle,
    fixrsvp_summary_rows: list[dict[str, object]],
    fixrsvp_contrast_rows: list[dict[str, object]],
    decoder_rows: list[dict[str, object]],
    contrast_rows: list[dict[str, object]],
    pilot_rows: list[dict[str, object]],
    trace_qc_rows: list[dict[str, object]],
    response_vectors_path: Path | None = None,
    response_vectors_schema_path: Path | None = None,
    deterministic_summary_rows: list[dict[str, object]] | None = None,
) -> None:
    combined_contrast_rows = fixrsvp_contrast_rows + contrast_rows
    _write_csv(out_dir / "active_sensing_efficiency_contrast_table.csv", combined_contrast_rows)

    fix_real = [row for row in fixrsvp_summary_rows if row["condition"] == "real"]
    fix_fixed = [row for row in fixrsvp_summary_rows if row["condition"] == "fixed_center"]
    fix_random_cov = [row for row in fixrsvp_summary_rows if row["condition"] == "random_cov"]
    real_fix = _nanmedian_or_nan(_safe_float(row["median_cumulative_spatial_bits_per_expected_spike"]) for row in fix_real)
    fixed_fix = _nanmedian_or_nan(_safe_float(row["median_cumulative_spatial_bits_per_expected_spike"]) for row in fix_fixed)
    random_fix = _nanmedian_or_nan(_safe_float(row["median_cumulative_spatial_bits_per_expected_spike"]) for row in fix_random_cov)
    rate_delta = 0.0

    identity_real = [row for row in decoder_rows if str(row["condition"]) == "real"]
    identity_fixed = [row for row in decoder_rows if str(row["condition"]) == "fixed_center"]
    identity_random = [row for row in decoder_rows if str(row["condition"]) == "random_cov"]
    real_identity = _nanmedian_or_nan(_safe_float(row["d2_per_expected_spike"]) for row in identity_real)
    fixed_identity = _nanmedian_or_nan(_safe_float(row["d2_per_expected_spike"]) for row in identity_fixed)
    random_identity = _nanmedian_or_nan(_safe_float(row["d2_per_expected_spike"]) for row in identity_random)
    real_det_identity = _nanmedian_or_nan(_safe_float(row.get("deterministic_identity_bits_per_expected_spike")) for row in identity_real)
    fixed_det_identity = _nanmedian_or_nan(_safe_float(row.get("deterministic_identity_bits_per_expected_spike")) for row in identity_fixed)
    random_det_identity = _nanmedian_or_nan(_safe_float(row.get("deterministic_identity_bits_per_expected_spike")) for row in identity_random)
    pilot_width = _nanmedian_or_nan(_safe_float(row["ci_width"]) for row in pilot_rows)

    fix_label = _build_decision_label(real_fix - fixed_fix, real_fix - random_fix, rate_delta, pilot_width)
    fixrsvp_real_n = max((_safe_int(row.get("n_trials")) for row in fixrsvp_summary_rows if str(row.get("condition")) == "real"), default=0)
    if fixrsvp_real_n > 0 and fixrsvp_real_n < 10:
        fix_label = "smoke_test_only"
    control_rows = [row for row in trace_qc_rows if str(row.get("condition")) in {"random_amp", "random_cov"}]
    rms_med = _nanmedian_or_nan(_safe_float(row.get("matched_rms_error")) for row in control_rows)
    cov_med = _nanmedian_or_nan(_safe_float(row.get("matched_cov_error")) for row in control_rows)
    path_med = _nanmedian_or_nan(_safe_float(row.get("path_length_error")) for row in control_rows)
    acf1_med = _nanmedian_or_nan(_safe_float(row.get("acf_lag1_error")) for row in control_rows)
    acf2_med = _nanmedian_or_nan(_safe_float(row.get("acf_lag2_error")) for row in control_rows)
    acf4_med = _nanmedian_or_nan(_safe_float(row.get("acf_lag4_error")) for row in control_rows)
    rms_cov_ok = np.isfinite(rms_med) and rms_med <= DEFAULT_RANDOM_CONTROL_RMS_ERROR_MAX and np.isfinite(cov_med) and cov_med <= DEFAULT_RANDOM_CONTROL_COV_ERROR_MAX
    temporal_ok = (
        np.isfinite(path_med)
        and path_med <= DEFAULT_RANDOM_CONTROL_PATH_ERROR_MAX
        and np.isfinite(acf1_med)
        and acf1_med <= DEFAULT_RANDOM_CONTROL_ACF_ERROR_MAX
        and np.isfinite(acf2_med)
        and acf2_med <= DEFAULT_RANDOM_CONTROL_ACF_ERROR_MAX
        and np.isfinite(acf4_med)
        and acf4_med <= DEFAULT_RANDOM_CONTROL_ACF_ERROR_MAX
    )
    if rms_cov_ok and temporal_ok:
        random_control_status = "matched_rms_cov_path_acf"
    elif rms_cov_ok:
        random_control_status = "covariance_matched_only"
    else:
        random_control_status = "limited_by_control_mismatch"

    identity_label = _build_identity_decision_label(decoder_rows, trace_qc_rows)
    closeout_flags = _compute_eoptotype_closeout_flags(decoder_rows=decoder_rows, trace_qc_rows=trace_qc_rows)
    n_logmar = int(closeout_flags["n_logmar"])
    n_pairs = int(closeout_flags["n_orientation_pairs"])
    closeout_scope_pass = bool(closeout_flags["scope_pass"])
    closeout_ready = bool(closeout_flags["closeout_ready"])
    if len(decoder_rows) == 0:
        identity_label = "implementation_failure"
        combined_label = identity_label
    else:
        combined_label = identity_label

    if np.isfinite(real_det_identity) and np.isfinite(fixed_det_identity) and np.isfinite(real_identity) and np.isfinite(fixed_identity):
        if np.sign(real_det_identity - fixed_det_identity) != np.sign(real_identity - fixed_identity):
            combined_label = "metric_dependent_mixed"
    if np.isfinite(real_det_identity) and np.isfinite(random_det_identity) and np.isfinite(real_identity) and np.isfinite(random_identity):
        if np.sign(real_det_identity - random_det_identity) != np.sign(real_identity - random_identity):
            combined_label = "metric_dependent_mixed"

    decision_rows = [
        {
            "analysis_mode": "fixrsvp_spatial_ssi",
            "primary_metric": "cumulative_spatial_bits_per_expected_spike",
            "n_trials": int(sum(_safe_int(row["n_trials"]) for row in fixrsvp_summary_rows if row["condition"] == "real")),
            "n_logmar": 0,
            "n_orientation_pairs": 0,
            "real_minus_fixed": real_fix - fixed_fix,
            "real_minus_fixed_ci": "pilot_power_summary.csv",
            "real_minus_stabilized": _nanmedian_or_nan(_safe_float(row["mean_delta"]) for row in fixrsvp_contrast_rows if row["contrast"] == "real_FEM - stabilized"),
            "real_minus_stabilized_ci": "pilot_power_summary.csv",
            "real_minus_random_amp": _nanmedian_or_nan(_safe_float(row["mean_delta"]) for row in fixrsvp_contrast_rows if row["contrast"] == "real_FEM - random_amp"),
            "real_minus_random_amp_ci": "pilot_power_summary.csv",
            "real_minus_random_cov": real_fix - random_fix,
            "real_minus_random_cov_ci": "pilot_power_summary.csv",
            "scale_dependence_status": "frames_per_im_sweep" if len({_safe_int(row["frames_per_im"]) for row in fixrsvp_summary_rows}) > 1 else "single_frame_rate",
            "random_control_status": "matched_rms_and_cov_qc_logged",
            "rate_confound_status": "inspect_expected_spikes_table",
            "decision_label": fix_label,
            "controls_passed": "trajectory_control_qc.csv",
            "manuscript_implication": "supporting_only",
            "next_action": "inspect_full_run_outputs",
        },
        {
            "analysis_mode": "eoptotype_identity",
            "primary_metric": "d2_per_expected_spike",
            "n_trials": int(len(decoder_rows)),
            "n_logmar": n_logmar,
            "n_orientation_pairs": n_pairs,
            "real_minus_fixed": real_identity - fixed_identity,
            "real_minus_fixed_ci": "split_level_estimate",
            "real_minus_stabilized": _nanmedian_or_nan(_safe_float(row["real_minus_stabilized_d2_per_expected_spike"]) for row in decoder_rows),
            "real_minus_stabilized_ci": "split_level_estimate",
            "real_minus_random_amp": _nanmedian_or_nan(_safe_float(row["real_minus_random_amp_d2_per_expected_spike"]) for row in decoder_rows),
            "real_minus_random_amp_ci": "split_level_estimate",
            "real_minus_random_cov": real_identity - random_identity,
            "real_minus_random_cov_ci": "split_level_estimate",
            "scale_dependence_status": "logmar_sweep",
            "random_control_status": random_control_status,
            "rate_confound_status": "poisson_budget_and_rate_normalized_reported",
            "decision_label": identity_label,
            "controls_passed": "trajectory_control_qc.csv",
            "manuscript_implication": "scope_limited_no_figure4_change" if identity_label != "model_active_sensing_efficiency_supported" else ("figure4_update_candidate" if closeout_ready else "scope_limited_no_figure4_change"),
            "next_action": "inspect_load_bearing_logmar_contrast",
            "scope_gate_status": "full_scope" if closeout_scope_pass else "limited_scope_diagnostic",
            "deterministic_identity_bits_per_expected_spike": real_det_identity,
            "deterministic_vs_poisson_consensus": "mixed" if combined_label == "metric_dependent_mixed" else "aligned_or_unresolved",
        },
        {
            "analysis_mode": "combined",
            "primary_metric": "joint_e1_status",
            "n_trials": int(len(fixrsvp_summary_rows) + len(decoder_rows)),
            "n_logmar": n_logmar,
            "n_orientation_pairs": n_pairs,
            "real_minus_fixed": real_identity - fixed_identity,
            "real_minus_fixed_ci": "mixed_sources",
            "real_minus_stabilized": float("nan"),
            "real_minus_stabilized_ci": "mixed_sources",
            "real_minus_random_amp": float("nan"),
            "real_minus_random_amp_ci": "mixed_sources",
            "real_minus_random_cov": real_identity - random_identity,
            "real_minus_random_cov_ci": "mixed_sources",
            "scale_dependence_status": "cross_axis",
            "random_control_status": "see_individual_modes",
            "rate_confound_status": "see_individual_modes",
            "decision_label": combined_label,
            "controls_passed": "see_individual_modes",
            "manuscript_implication": "changes_figure4" if (combined_label == "model_active_sensing_efficiency_supported" and closeout_ready) else "does_not_change_figure4",
            "next_action": "stop_after_summary",
            "scope_gate_status": "full_scope" if closeout_scope_pass else "limited_scope_diagnostic",
            "deterministic_identity_bits_per_expected_spike": real_det_identity,
        },
    ]
    _write_csv(out_dir / "active_sensing_efficiency_decision_table.csv", decision_rows)

    lines = [
        "# Active Sensing Efficiency",
        "",
        "Current pass: Poisson d2 identity-efficiency under current HiResERenderer/HiResRetina pipeline.",
        "",
        "What was tested:",
        "- E-optotype identity efficiency per expected spike across LogMAR x orientation-pair grid.",
        "- Controls: fixed_center, stabilized, random_amp, random_cov, and scaled FEM controls.",
        "- Feature representations including spatial-average and spatial-map-preserving variants.",
        "",
        f"Full-scope gates passed: {'yes' if closeout_scope_pass else 'no'}",
        f"Random controls passed all QC: {'yes' if random_control_status == 'matched_rms_cov_path_acf' else 'no'}",
        "",
        "Numerator/denominator interpretation:",
        "- See eoptotype_identity_numerator_denominator_decomposition.csv for delta_poisson_d2 vs delta_expected_spikes.",
        "",
        "Spatial-map-preserving ablation:",
        "- See eoptotype_identity_feature_ablation_summary.csv and eoptotype_feature_ablation_comparison.png.",
        "",
        "Global vs pair/scale-specific:",
        "- See eoptotype_real_minus_random_cov_heatmap.png and eoptotype_real_minus_controls_vs_logmar.png.",
        "",
        f"Class-conditional response vectors persisted: {'yes' if response_vectors_path and response_vectors_path.exists() else 'no'}",
        f"Persistence file: {response_vectors_path if response_vectors_path else 'not available'}",
        f"Schema file: {response_vectors_schema_path if response_vectors_schema_path else 'not available'}",
        f"Deterministic identity bits/spike computed: {'yes' if np.isfinite(real_det_identity) else 'no'}",
        f"Deterministic sign vs d2 sign: {'same' if np.isfinite(real_det_identity) and np.isfinite(fixed_det_identity) and np.isfinite(real_identity) and np.isfinite(fixed_identity) and np.sign(real_det_identity - fixed_det_identity) == np.sign(real_identity - fixed_identity) else 'different_or_unresolved'}",
        f"Conclusion changed by deterministic bits/spike: {'yes' if combined_label == 'metric_dependent_mixed' else 'no'}",
        "",
        f"Final E1 status:",
        f"- {combined_label}",
        "",
        f"Figure 4 implication: {'changes Figure 4' if (combined_label == 'model_active_sensing_efficiency_supported' and closeout_ready and combined_label != 'metric_dependent_mixed') else 'does not change Figure 4'}",
        "- Hypothesis table and Figure 4 text are not auto-updated by this script.",
    ]
    (out_dir / "active_sensing_efficiency_readme.md").write_text("\n".join(lines))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the active-sensing efficiency batch analysis.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--model-type", type=str, default="resnet_none_convgru")
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--mcfarland-outputs", type=Path, default=DEFAULT_MCFARLAND_OUTPUTS)
    parser.add_argument("--dataset-idx", type=int, default=10)
    parser.add_argument("--stim-modes", nargs="+", default=["fixrsvp", "eoptotype"])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-label", type=str, default="manual")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--pilot-trials", type=int, default=10)
    parser.add_argument("--max-trials", type=int, default=20)
    parser.add_argument("--max-trial-frames", type=int, default=120)
    parser.add_argument("--min-fix-dur", type=int, default=60)
    parser.add_argument("--n-lags", type=int, default=DEFAULT_N_LAGS)
    parser.add_argument("--out-size", nargs=2, type=int, default=list(DEFAULT_OUT_SIZE))
    parser.add_argument("--dt", type=str, default="auto")
    parser.add_argument("--n-random-controls", type=int, default=DEFAULT_RANDOM_CONTROL_REPEATS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--frames-per-im", nargs="+", type=int, default=list(DEFAULT_FRAMES_PER_IM))
    parser.add_argument("--logmar-values", nargs="+", type=float, default=list(DEFAULT_LOGMARS))
    parser.add_argument("--orientations", nargs="+", type=int, default=list(DEFAULT_ORIENTATIONS))
    parser.add_argument("--trajectory-controls", nargs="+", default=list(DEFAULT_TRAJECTORY_CONTROLS))
    parser.add_argument("--scaled-fem-values", nargs="+", type=float, default=list(DEFAULT_SCALED_FEM_VALUES))
    parser.add_argument("--readouts", nargs="+", default=list(DEFAULT_READOUTS))
    parser.add_argument("--decoder-workers", type=int, default=1)
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--skip-fixrsvp", action="store_true")
    parser.add_argument("--skip-eoptotype", action="store_true")
    parser.add_argument("--eye-traces", type=Path, default=DEFAULT_EYE_TRACES)
    parser.add_argument("--random-sigma-frames", type=float, default=DEFAULT_SIGMA_FRAMES)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    args.device = _pick_device(args.device)
    args.out_size = tuple(int(v) for v in args.out_size)
    args.frames_per_im = tuple(int(v) for v in args.frames_per_im)
    args.logmar_values = tuple(float(v) for v in args.logmar_values)
    args.orientations = tuple(int(v) for v in args.orientations)
    args.scaled_fem_values = tuple(float(v) for v in args.scaled_fem_values)
    args.readouts = tuple(str(v) for v in args.readouts)
    args.decoder_workers = max(1, int(args.decoder_workers))
    args.trajectory_controls = _resolve_trajectory_controls(args)
    if args.sanity_check:
        args.max_trials = min(int(args.max_trials), int(args.pilot_trials), 2)
        args.n_random_controls = min(int(args.n_random_controls), 1)
        if ("fixrsvp" in args.stim_modes) and not args.skip_fixrsvp and len(args.frames_per_im) > 1:
            args.frames_per_im = tuple(sorted({min(args.frames_per_im), max(args.frames_per_im)}))
        if ("eoptotype" in args.stim_modes) and not args.skip_eoptotype:
            args.logmar_values = (max(args.logmar_values),)
            args.orientations = tuple(args.orientations[:2]) if len(args.orientations) >= 2 else args.orientations
        args.pilot_only = True
    if args.pilot_only:
        args.max_trials = min(int(args.max_trials), int(args.pilot_trials))
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT_BASE / f"active_sensing_efficiency_{_safe_slug(args.run_label)}"
    out_dir = Path(args.out_dir)
    deterministic_summary_rows: list[dict[str, Any]] | None = None
    response_vectors_path: Path | None = None
    response_vectors_schema_path: Path | None = None
    (out_dir / "fixrsvp_spatial_ssi").mkdir(parents=True, exist_ok=True)
    (out_dir / "eoptotype_identity").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    _write_json(
        out_dir / "logs" / "run_config.json",
        {
            "run_label": args.run_label,
            "device": args.device,
            "dataset_idx": int(args.dataset_idx),
            "stim_modes": list(args.stim_modes),
            "trajectory_controls": list(args.trajectory_controls),
            "frames_per_im": list(args.frames_per_im),
            "logmar_values": list(args.logmar_values),
            "orientations": list(args.orientations),
            "readouts": list(args.readouts),
        },
    )

    rng = np.random.default_rng(int(args.random_seed))
    model_bundle = _load_model_bundle(args)
    fixrsvp_trials, fixrsvp_summary = _extract_fixrsvp_eye_traces(model_bundle, args)

    trace_qc_rows, trace_cache = _trajectory_control_qc_rows(
        fixrsvp_trials=fixrsvp_trials,
        conditions=args.trajectory_controls,
        n_random_controls=int(args.n_random_controls),
        sigma_frames=float(args.random_sigma_frames),
        rng=np.random.default_rng(int(args.random_seed)),
    )
    _write_csv(out_dir / "trajectory_control_qc.csv", trace_qc_rows)

    dt_rows, dt_selected, selection_reason = _verify_dt_convention(
        model_bundle=model_bundle,
        fixrsvp_trials=fixrsvp_trials,
        trace_cache=trace_cache,
        out_dir=out_dir,
        frames_per_im=max(args.frames_per_im),
        n_lags=int(args.n_lags),
        out_size=(int(args.out_size[0]), int(args.out_size[1])),
    )
    if str(args.dt).lower() != "auto":
        dt_selected = float(args.dt)
        selection_reason = "user_override"

    fixrsvp_trial_rows: list[dict[str, object]] = []
    fixrsvp_summary_rows: list[dict[str, object]] = []
    fixrsvp_contrast_rows: list[dict[str, object]] = []
    pilot_rows: list[dict[str, object]] = []
    if ("fixrsvp" in args.stim_modes) and not args.skip_fixrsvp:
        fixrsvp_trial_rows, fixrsvp_summary_rows, fixrsvp_contrast_rows, pilot_rows = _run_fixrsvp_mode(
            model_bundle=model_bundle,
            fixrsvp_trials=fixrsvp_trials[: int(args.max_trials)],
            trace_cache=trace_cache,
            out_dir=out_dir,
            args=args,
            dt_selected=dt_selected,
            rng=np.random.default_rng(int(args.random_seed) + 1),
        )
        _write_fixrsvp_figures(out_dir, fixrsvp_summary_rows, fixrsvp_contrast_rows)

    eoptotype_trial_rows: list[dict[str, object]] = []
    decoder_rows: list[dict[str, object]] = []
    eoptotype_contrast_rows: list[dict[str, object]] = []
    if ("eoptotype" in args.stim_modes) and not args.skip_eoptotype:
        eoptotype_trial_rows, decoder_rows, eoptotype_contrast_rows, _features_path = _run_eoptotype_mode(
            model_bundle=model_bundle,
            out_dir=out_dir,
            args=args,
            dt_selected=dt_selected,
            rng=np.random.default_rng(int(args.random_seed) + 2),
            trace_qc_rows=trace_qc_rows,
        )
        deterministic_summary_path = out_dir / "eoptotype_identity_bits_per_spike_summary.csv"
        if deterministic_summary_path.exists():
            with deterministic_summary_path.open("r", newline="") as handle:
                deterministic_summary_rows = list(csv.DictReader(handle))
        response_vectors_path = out_dir / "eoptotype_identity" / "eoptotype_response_vectors.npz"
        response_vectors_schema_path = out_dir / "eoptotype_identity" / "eoptotype_response_vectors.schema.json"
        _write_eoptotype_figures(out_dir, decoder_rows, deterministic_summary_rows=deterministic_summary_rows)
    else:
        _write_csv(
            out_dir / "eoptotype_failure_diagnostics.csv",
            [
                {
                    "check_name": "skip_eoptotype_passed",
                    "status": "yes" if args.skip_eoptotype else "no",
                    "value": 1 if args.skip_eoptotype else 0,
                    "detail": "Eoptotype branch was skipped explicitly." if args.skip_eoptotype else "Eoptotype mode was not requested.",
                },
                {
                    "check_name": "decoder_rows_generated",
                    "status": "skipped" if args.skip_eoptotype else "fail",
                    "value": 0,
                    "detail": "Decoder generation not attempted because eoptotype branch was skipped.",
                },
            ],
        )

    sanity_rows = _build_sanity_rows(
        trace_qc_rows=trace_qc_rows,
        dt_rows=dt_rows,
        fixrsvp_trials=fixrsvp_trials,
        decoder_rows=decoder_rows,
        eoptotype_expected=(("eoptotype" in args.stim_modes) and (not args.skip_eoptotype)),
    )
    _write_csv(out_dir / "sanity_check_status.csv", sanity_rows)

    _write_combined_outputs(
        out_dir=out_dir,
        model_bundle=model_bundle,
        fixrsvp_summary_rows=fixrsvp_summary_rows,
        fixrsvp_contrast_rows=fixrsvp_contrast_rows,
        decoder_rows=decoder_rows,
        contrast_rows=eoptotype_contrast_rows,
        pilot_rows=pilot_rows,
        trace_qc_rows=trace_qc_rows,
        response_vectors_path=response_vectors_path,
        response_vectors_schema_path=response_vectors_schema_path,
        deterministic_summary_rows=deterministic_summary_rows,
    )

    if args.sanity_check:
        failing = [row for row in sanity_rows if str(row["status"]) == "fail"]
        if failing:
            raise RuntimeError("Sanity checks failed; inspect sanity_check_status.csv before interpreting outputs")
        return


if __name__ == "__main__":
    main()