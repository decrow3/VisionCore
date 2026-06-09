#!/usr/bin/env python3
"""Pilot runner for the active-sensing movie-information figure.

The runner has two deliberately separated layers:

1. pre-model retinal movie diagnostics, which are always computed;
2. optional V1 twin spatial-SSI curves, enabled with ``--run-model``.

Without ``--run-model`` the script still writes a smoke-test cumulative proxy
based on retinal temporal modulation. Those rows are labeled as a proxy and
should not be interpreted as model information.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift as scipy_shift


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


DEFAULT_OUT_ROOT = ROOT / "outputs" / "active_sensing_movie_information"
DEFAULT_CHECKPOINT_DIR = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints"
)
DEFAULT_MCFARLAND_OUTPUTS = ROOT / "scripts" / "mcfarland_outputs.pkl"
DEFAULT_CONDITIONS = ("real_FEM", "stabilized", "random_amp")
DEFAULT_STIM_CONDITIONS = ("intact",)
DEFAULT_BANDS_CPD = ((0.0, 2.0), (2.0, 8.0), (8.0, 18.0))
DEFAULT_TRACE_CACHE = ROOT / "declan" / "fixrsvp_fixation_pool.pkl"
DEFAULT_PRIMARY_MODEL_METRIC = "cumulative_spatial_bits_per_expected_spike"
EPS = 1e-8
RELATIVE_GAIN_MIN_DENOMINATOR = 1e-6


@dataclass(frozen=True)
class RunnerConfig:
    run_label: str
    out_dir: str
    stimulus_source: str
    n_images: int
    n_traces: int
    n_frames: int
    image_size: int
    ppd: float
    frame_rate_hz: float
    conditions: tuple[str, ...]
    stimulus_conditions: tuple[str, ...]
    trace_source: str
    trace_cache: str
    run_model: bool
    model_metric: str
    seed: int
    n_bootstrap: int


@dataclass(frozen=True)
class ModelBundle:
    model: object
    readout: object
    device: str
    n_lags: int
    batch_size: int


def _parse_csv_strs(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def _parse_bands(value: str) -> tuple[tuple[float, float], ...]:
    bands: list[tuple[float, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        lo_s, hi_s = item.split("-")
        lo = float(lo_s)
        hi = float(hi_s)
        if hi <= lo:
            raise ValueError(f"Band upper edge must exceed lower edge: {item}")
        bands.append((lo, hi))
    return tuple(bands)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_slug(text: object) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text))


def _standardize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image = image - float(np.mean(image))
    sd = float(np.std(image))
    if sd > 0:
        image = image / sd
    return image.astype(np.float32)


def _make_synthetic_natural_images(
    n_images: int,
    size: int,
    rng: np.random.Generator,
    alpha: float = 1.4,
) -> np.ndarray:
    fy = np.fft.fftfreq(size)
    fx = np.fft.fftfreq(size)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    amp = 1.0 / np.maximum(rr, 1.0 / size) ** alpha
    amp[0, 0] = 0.0

    images = np.empty((n_images, size, size), dtype=np.float32)
    for i in range(n_images):
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(size, size))
        coeff = amp * (np.cos(phase) + 1j * np.sin(phase))
        image = np.fft.ifft2(coeff).real
        images[i] = _standardize_image(image)
    return images


def _load_image_stack(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    source = args.stimulus_source
    if source == "synthetic":
        return (
            _make_synthetic_natural_images(args.n_images, args.image_size, rng=rng),
            "synthetic_1_over_f_noise",
        )

    try:
        from spatial_info import make_stimulus_stack
    except Exception as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError(f"Could not import scripts/spatial_info.py helpers: {exc}") from exc

    stack_type = "nat" if source == "nat_stack" else "fixrsvp"
    full_stack = make_stimulus_stack(type=stack_type, frames_per_im=1, num_frames=args.n_images)
    full_stack = np.asarray(full_stack, dtype=np.float32)
    if full_stack.ndim != 3:
        raise ValueError(f"Expected image stack with shape (N,H,W), got {full_stack.shape}")
    if full_stack.shape[0] < args.n_images:
        raise ValueError(f"Requested {args.n_images} images but stack has {full_stack.shape[0]}")

    images = np.empty((args.n_images, args.image_size, args.image_size), dtype=np.float32)
    for i in range(args.n_images):
        images[i] = _center_crop_or_pad(full_stack[i], args.image_size)
        images[i] = _standardize_image(images[i])
    return images, source


def _center_crop_or_pad(image: np.ndarray, size: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape
    if h < size or w < size:
        out = np.full((max(h, size), max(w, size)), float(np.mean(image)), dtype=np.float32)
        y0 = (out.shape[0] - h) // 2
        x0 = (out.shape[1] - w) // 2
        out[y0 : y0 + h, x0 : x0 + w] = image
        image = out
        h, w = image.shape
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return image[y0 : y0 + size, x0 : x0 + size].astype(np.float32)


def _phase_scramble_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    image = _standardize_image(image)
    f = np.fft.fft2(image)
    amp = np.abs(f)
    phase = rng.uniform(-np.pi, np.pi, size=image.shape)
    scrambled = np.fft.ifft2(amp * np.exp(1j * phase)).real
    return _standardize_image(scrambled)


def _generate_real_like_trace(
    n_frames: int,
    frame_rate_hz: float,
    rng: np.random.Generator,
    drift_std_arcmin_per_frame: float = 0.08,
    microsaccade_rate_hz: float = 2.0,
    microsaccade_amp_arcmin_mean: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    trace = np.zeros((n_frames, 2), dtype=np.float32)
    state = np.zeros(n_frames, dtype=np.int8)

    drift_std_deg = drift_std_arcmin_per_frame / 60.0
    steps = rng.normal(0.0, drift_std_deg, size=(n_frames - 1, 2)).astype(np.float32)
    trace[1:] = np.cumsum(steps, axis=0)

    p_ms = microsaccade_rate_hz / max(frame_rate_hz, EPS)
    t = 8
    while t < n_frames - 4:
        if rng.random() < p_ms:
            dur = int(rng.integers(2, 5))
            amp_arcmin = float(rng.lognormal(np.log(microsaccade_amp_arcmin_mean), 0.35))
            amp_deg = amp_arcmin / 60.0
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            jump = amp_deg * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            end = min(n_frames, t + dur)
            ramp = np.linspace(0.0, 1.0, end - t, dtype=np.float32)[:, None]
            trace[t:end] += ramp * jump
            trace[end:] += jump
            state[t:end] = 1
            t = end + int(rng.integers(8, 24))
        else:
            t += 1

    trace -= np.mean(trace, axis=0, keepdims=True)
    return trace.astype(np.float32), state


def _load_cached_trace_pool(
    trace_source: str,
    trace_cache: Path,
    n_frames: int,
) -> list[np.ndarray]:
    if trace_source == "synthetic":
        return []

    with Path(trace_cache).open("rb") as handle:
        obj = pickle.load(handle)

    traces: list[np.ndarray] = []
    if trace_source == "fixrsvp_fixation_pool":
        if not isinstance(obj, list):
            raise TypeError(f"Expected fixation-pool cache to be a list, got {type(obj)}")
        candidates = obj
    elif trace_source == "fixrsvp_cached_eyepos":
        if not isinstance(obj, dict) or "eyepos" not in obj:
            raise TypeError("Expected cached eyepos dict with key 'eyepos'")
        candidates = list(np.asarray(obj["eyepos"]))
    else:
        raise ValueError(f"Unknown trace source: {trace_source}")

    for candidate in candidates:
        arr = np.asarray(candidate, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            continue
        finite = np.isfinite(arr).all(axis=1)
        if not np.any(finite):
            continue
        valid = arr[finite]
        if valid.shape[0] < n_frames:
            continue
        traces.append(valid)

    if not traces:
        raise ValueError(f"No cached traces with at least {n_frames} finite frames in {trace_cache}")
    return traces


def _sample_cached_trace(
    trace_pool: list[np.ndarray],
    trace_id: int,
    n_frames: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str]:
    base = trace_pool[int(trace_id) % len(trace_pool)]
    max_start = max(0, base.shape[0] - n_frames)
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    trace = np.asarray(base[start : start + n_frames], dtype=np.float32).copy()
    trace -= np.mean(trace, axis=0, keepdims=True)
    steps_arcmin = np.linalg.norm(np.diff(trace, axis=0), axis=1) * 60.0
    state = np.zeros(trace.shape[0], dtype=np.int8)
    if steps_arcmin.size:
        # Conservative event marker for cached full-trial traces. For fixation-pool
        # traces this usually stays false, as desired.
        event_frames = np.where(steps_arcmin > 3.0)[0] + 1
        state[event_frames] = 1
    detail = f"cached_fixrsvp_trace_index={trace_id % len(trace_pool)};start={start};length={trace.shape[0]}"
    return trace.astype(np.float32), state, detail


def _get_real_trace(
    args: argparse.Namespace,
    trace_pool: list[np.ndarray],
    trace_id: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str]:
    if args.trace_source == "synthetic":
        trace, state = _generate_real_like_trace(
            args.n_frames,
            frame_rate_hz=args.frame_rate_hz,
            rng=rng,
        )
        return trace, state, "synthetic_real_like_trace"
    return _sample_cached_trace(trace_pool, trace_id, args.n_frames, rng)


def _make_control_trace(
    real_trace: np.ndarray,
    condition: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    real_trace = np.asarray(real_trace, dtype=np.float32)
    if condition == "real_FEM":
        return real_trace.copy(), "measured_or_synthetic_real_like_trace"
    if condition == "stabilized":
        center = np.mean(real_trace, axis=0, keepdims=True)
        return np.repeat(center, real_trace.shape[0], axis=0).astype(np.float32), "trial_mean_stabilized"
    if condition == "fixed_center":
        return np.zeros_like(real_trace), "fixed_center"

    steps = np.diff(real_trace, axis=0)
    if condition == "random_amp":
        amps = np.linalg.norm(steps, axis=1)
        angles = rng.uniform(0.0, 2.0 * np.pi, size=amps.shape[0])
        ctrl_steps = np.column_stack([amps * np.cos(angles), amps * np.sin(angles)])
        ctrl = np.vstack([np.zeros((1, 2)), np.cumsum(ctrl_steps, axis=0)])
        ctrl -= np.mean(ctrl, axis=0, keepdims=True)
        return ctrl.astype(np.float32), "step_amplitude_matched_random_directions"

    if condition == "random_cov":
        mu = np.mean(steps, axis=0)
        cov = np.cov(steps.T)
        cov = np.asarray(cov, dtype=np.float64) + np.eye(2) * 1e-10
        ctrl_steps = rng.multivariate_normal(mu, cov, size=steps.shape[0])
        ctrl = np.vstack([np.zeros((1, 2)), np.cumsum(ctrl_steps, axis=0)])
        ctrl -= np.mean(ctrl, axis=0, keepdims=True)
        return ctrl.astype(np.float32), "step_covariance_matched_gaussian"

    if condition in {"trajectory_order_shuffle", "phase_order_shuffle"}:
        order = rng.permutation(real_trace.shape[0])
        ctrl = real_trace[order]
        ctrl -= np.mean(ctrl, axis=0, keepdims=True)
        return ctrl.astype(np.float32), "same_positions_time_order_shuffled"

    raise ValueError(f"Unknown trajectory condition: {condition}")


def _render_movie(
    image: np.ndarray,
    trace_deg: np.ndarray,
    ppd: float,
    out_size: int,
    order: int = 1,
) -> np.ndarray:
    image = _center_crop_or_pad(_standardize_image(image), out_size)
    movie = np.empty((trace_deg.shape[0], out_size, out_size), dtype=np.float32)
    for t, (x_deg, y_deg) in enumerate(trace_deg):
        # Shift the stimulus opposite to eye position to approximate retinal motion.
        shifted = scipy_shift(
            image,
            shift=(-float(y_deg) * ppd, -float(x_deg) * ppd),
            order=order,
            mode="reflect",
            prefilter=False,
        )
        movie[t] = shifted.astype(np.float32)
    return movie


def _radial_bandpass(image: np.ndarray, ppd: float, band_cpd: tuple[float, float]) -> np.ndarray:
    image = _standardize_image(image)
    h, w = image.shape
    fy = np.fft.fftfreq(h, d=1.0 / ppd)
    fx = np.fft.fftfreq(w, d=1.0 / ppd)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    lo, hi = band_cpd
    mask = (rr >= lo) & (rr < hi)
    filtered = np.fft.ifft2(np.fft.fft2(image) * mask).real
    return _standardize_image(filtered)


def _temporal_contrast(movie: np.ndarray) -> np.ndarray:
    if movie.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)
    diff = np.diff(movie, axis=0)
    return np.sqrt(np.mean(diff * diff, axis=(1, 2))).astype(np.float32)


def _temporal_power_summary(movie: np.ndarray, frame_rate_hz: float) -> dict[str, float]:
    x = np.asarray(movie, dtype=np.float32)
    x = x - np.mean(x, axis=0, keepdims=True)
    spec = np.fft.rfft(x, axis=0)
    power = np.mean(np.abs(spec) ** 2, axis=(1, 2))
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / frame_rate_hz)
    if power.shape[0] > 1:
        nonzero = power[1:]
        nonzero_freqs = freqs[1:]
        peak_idx = int(np.argmax(nonzero))
        total_power = float(np.sum(nonzero))
        peak_freq = float(nonzero_freqs[peak_idx])
    else:
        total_power = 0.0
        peak_freq = 0.0
    low = float(np.sum(power[(freqs >= 0.5) & (freqs < 5.0)]))
    mid = float(np.sum(power[(freqs >= 5.0) & (freqs < 20.0)]))
    high = float(np.sum(power[freqs >= 20.0]))
    return {
        "temporal_power_total": total_power,
        "temporal_power_0p5_5hz": low,
        "temporal_power_5_20hz": mid,
        "temporal_power_ge20hz": high,
        "peak_temporal_frequency_hz": peak_freq,
    }


def _compute_retinal_rows(
    image_id: int,
    trace_id: int,
    stimulus_condition: str,
    condition: str,
    movie: np.ndarray,
    stabilized_movie: np.ndarray,
    trace: np.ndarray,
    state: np.ndarray,
    frame_rate_hz: float,
) -> tuple[dict[str, object], dict[str, object]]:
    tc = _temporal_contrast(movie)
    motion = movie - stabilized_movie
    step_arcmin = np.linalg.norm(np.diff(trace, axis=0), axis=1) * 60.0
    freq = _temporal_power_summary(movie, frame_rate_hz=frame_rate_hz)
    micro_mask = state[1:].astype(bool) if state.shape[0] > 1 else np.zeros_like(tc, dtype=bool)
    drift_mask = ~micro_mask
    diag = {
        "image_id": image_id,
        "trace_id": trace_id,
        "stimulus_condition": stimulus_condition,
        "trajectory_condition": condition,
        "temporal_contrast_mean": float(np.mean(tc)) if tc.size else 0.0,
        "temporal_contrast_drift_mean": float(np.mean(tc[drift_mask])) if np.any(drift_mask) else float("nan"),
        "temporal_contrast_microsaccade_mean": float(np.mean(tc[micro_mask])) if np.any(micro_mask) else float("nan"),
        "motion_power_mean": float(np.mean(motion * motion)),
        "movie_power_mean": float(np.mean(movie * movie)),
        "mean_step_arcmin": float(np.mean(step_arcmin)) if step_arcmin.size else 0.0,
        "p95_step_arcmin": float(np.percentile(step_arcmin, 95)) if step_arcmin.size else 0.0,
        "microsaccade_frame_fraction": float(np.mean(state > 0)),
    }
    diag.update(freq)
    freq_row = {
        "image_id": image_id,
        "trace_id": trace_id,
        "stimulus_condition": stimulus_condition,
        "trajectory_condition": condition,
        **freq,
    }
    return diag, freq_row


def _cumulative_proxy_from_movie(movie: np.ndarray, frame_rate_hz: float) -> np.ndarray:
    tc = _temporal_contrast(movie)
    if tc.size == 0:
        return np.zeros((movie.shape[0],), dtype=np.float32)
    energy = tc * tc
    cumulative = np.concatenate([[0.0], np.cumsum(energy / frame_rate_hz)])
    return cumulative.astype(np.float32)


def _load_model_bundle(args: argparse.Namespace) -> ModelBundle:
    import dill
    import torch
    from eval.eval_stack_multidataset import load_model
    from spatial_info import get_spatial_readout

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    load_kwargs = {
        "model_type": args.model_type,
        "model_index": args.model_index,
        "checkpoint_path": None,
        "checkpoint_dir": str(args.checkpoint_dir),
        "device": "cpu",
    }
    if args.cfg_dir_override:
        load_kwargs["cfg_dir_override"] = str(args.cfg_dir_override)
    try:
        model, _model_info = load_model(**load_kwargs)
    except Exception:
        if (
            str(args.model_type) == "resnet_none_convgru"
            and int(args.model_index) == 0
            and Path(args.checkpoint_dir) == DEFAULT_CHECKPOINT_DIR
        ):
            from scripts.utils import get_model_and_dataset_configs

            model, _dataset_configs = get_model_and_dataset_configs(mode="standard")
        else:
            raise
    model.model.eval()
    if hasattr(model.model, "convnet"):
        model.model.convnet.use_checkpointing = True
    model = model.to(device)

    with Path(args.mcfarland_outputs).open("rb") as handle:
        outputs = dill.load(handle)
    readout = get_spatial_readout(model, outputs).to(device)
    readout.eval()

    return ModelBundle(
        model=model,
        readout=readout,
        device=device,
        n_lags=int(args.n_lags),
        batch_size=int(args.batch_size),
    )


def _model_spatial_ssi_series(movie: np.ndarray, bundle: ModelBundle, frame_rate_hz: float) -> dict[str, np.ndarray]:
    import torch
    from spatial_info import compute_rate_map_batched, embed_time_lags, spatial_ssi_population

    dt = 1.0 / frame_rate_hz
    with torch.no_grad():
        movie_t = torch.from_numpy(movie.astype(np.float32))
        stim = embed_time_lags(movie_t, n_lags=bundle.n_lags)
        rates = compute_rate_map_batched(
            bundle.model,
            bundle.readout,
            stim,
            batch_size=bundle.batch_size,
        )
        ispike_t, irate_t, i_tn = spatial_ssi_population(rates, dt=dt)
        r = rates.reshape(rates.shape[0], rates.shape[1], -1)
        rbar = r.mean(dim=2)
        spikes_tn = rbar * dt
        spikes_t = spikes_tn.sum(dim=1)
        bits_t = (spikes_tn * i_tn).sum(dim=1)
        cumulative_bits = torch.cumsum(bits_t, dim=0)
        cumulative_spikes = torch.cumsum(spikes_t, dim=0)
        cumulative_bits_per_spike = cumulative_bits / torch.clamp(cumulative_spikes, min=EPS)
        cumulative_bits_per_sec = cumulative_bits / torch.clamp(
            torch.arange(1, cumulative_bits.numel() + 1, device=cumulative_bits.device, dtype=cumulative_bits.dtype) * dt,
            min=EPS,
        )
    pad = np.full((bundle.n_lags - 1,), np.nan, dtype=np.float32)

    def padded(x: "torch.Tensor") -> np.ndarray:
        return np.concatenate([pad, x.detach().cpu().numpy().astype(np.float32)])

    return {
        "cumulative_spatial_bits": padded(cumulative_bits),
        "cumulative_spatial_bits_per_expected_spike": padded(cumulative_bits_per_spike),
        "cumulative_expected_spikes": padded(cumulative_spikes),
        "mean_spatial_bits_per_sec_to_date": padded(cumulative_bits_per_sec),
        "instantaneous_spatial_bits_per_expected_spike": padded(ispike_t),
        "instantaneous_spatial_bits_per_sec": padded(irate_t),
    }


def _paired_bootstrap(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if values.size == 1:
        val = float(values[0])
        return val, val, val
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        draw = rng.choice(values, size=values.size, replace=True)
        samples[i] = np.mean(draw)
    return float(np.mean(values)), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _summarize_gains(
    info_rows: list[dict[str, object]],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> list[dict[str, object]]:
    final_rows = [row for row in info_rows if bool(row.get("is_final_time", False))]
    by_key: dict[tuple[object, ...], dict[str, dict[str, object]]] = {}
    for row in final_rows:
        key = (
            row["image_id"],
            row["trace_id"],
            row["stimulus_condition"],
            row["metric"],
        )
        by_key.setdefault(key, {})[str(row["trajectory_condition"])] = row

    controls = sorted({str(row["trajectory_condition"]) for row in final_rows if row["trajectory_condition"] != "real_FEM"})
    out: list[dict[str, object]] = []
    for control in controls:
        diffs: list[float] = []
        rels: list[float] = []
        metric_name = ""
        stimulus_conditions = sorted({str(row["stimulus_condition"]) for row in final_rows})
        for stim_cond in stimulus_conditions:
            stim_diffs: list[float] = []
            stim_rels: list[float] = []
            for key, rows in by_key.items():
                if key[2] != stim_cond:
                    continue
                if "real_FEM" not in rows or control not in rows:
                    continue
                real = float(rows["real_FEM"]["cumulative_information"])
                ctrl = float(rows[control]["cumulative_information"])
                metric_name = str(rows["real_FEM"]["metric"])
                stim_diffs.append(real - ctrl)
                stim_rels.append(real / ctrl if abs(ctrl) >= RELATIVE_GAIN_MIN_DENOMINATOR else float("nan"))
            mean, lo, hi = _paired_bootstrap(np.asarray(stim_diffs), rng, n_bootstrap)
            rel_mean, rel_lo, rel_hi = _paired_bootstrap(np.asarray(stim_rels), rng, n_bootstrap)
            out.append(
                {
                    "stimulus_condition": stim_cond,
                    "metric": metric_name,
                    "comparison": f"real_FEM_minus_{control}",
                    "n_pairs": len(stim_diffs),
                    "mean_gain": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "mean_relative_gain": rel_mean,
                    "relative_ci95_low": rel_lo,
                    "relative_ci95_high": rel_hi,
                    "interpretation_status": "positive" if np.isfinite(mean) and mean > 0 else "not_positive",
                }
            )
            diffs.extend(stim_diffs)
            rels.extend(stim_rels)

        mean, lo, hi = _paired_bootstrap(np.asarray(diffs), rng, n_bootstrap)
        rel_mean, rel_lo, rel_hi = _paired_bootstrap(np.asarray(rels), rng, n_bootstrap)
        out.append(
            {
                "stimulus_condition": "all",
                "metric": metric_name,
                "comparison": f"real_FEM_minus_{control}",
                "n_pairs": len(diffs),
                "mean_gain": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "mean_relative_gain": rel_mean,
                "relative_ci95_low": rel_lo,
                "relative_ci95_high": rel_hi,
                "interpretation_status": "positive" if np.isfinite(mean) and mean > 0 else "not_positive",
            }
        )
    return out


def _band_rows_for_movie(
    image: np.ndarray,
    trace: np.ndarray,
    ppd: float,
    out_size: int,
    bands: tuple[tuple[float, float], ...],
    image_id: int,
    trace_id: int,
    stimulus_condition: str,
    trajectory_condition: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lo, hi in bands:
        band_image = _radial_bandpass(image, ppd=ppd, band_cpd=(lo, hi))
        movie = _render_movie(band_image, trace, ppd=ppd, out_size=out_size)
        tc = _temporal_contrast(movie)
        rows.append(
            {
                "image_id": image_id,
                "trace_id": trace_id,
                "stimulus_condition": stimulus_condition,
                "trajectory_condition": trajectory_condition,
                "band_cpd_low": lo,
                "band_cpd_high": hi,
                "band_temporal_contrast_mean": float(np.mean(tc)) if tc.size else 0.0,
                "band_temporal_contrast_energy": float(np.mean(tc * tc)) if tc.size else 0.0,
            }
        )
    return rows


def _mean_or_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _summarize_drift_microsaccades(retinal_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted(
        {
            (str(row["stimulus_condition"]), str(row["trajectory_condition"]))
            for row in retinal_rows
        }
    )
    out: list[dict[str, object]] = []
    for stim_cond, condition in keys:
        group = [
            row
            for row in retinal_rows
            if row["stimulus_condition"] == stim_cond and row["trajectory_condition"] == condition
        ]
        drift = _mean_or_nan(float(row["temporal_contrast_drift_mean"]) for row in group)
        micro = _mean_or_nan(float(row["temporal_contrast_microsaccade_mean"]) for row in group)
        out.append(
            {
                "stimulus_condition": stim_cond,
                "trajectory_condition": condition,
                "n_movies": len(group),
                "mean_temporal_contrast": _mean_or_nan(float(row["temporal_contrast_mean"]) for row in group),
                "mean_drift_temporal_contrast": drift,
                "mean_microsaccade_temporal_contrast": micro,
                "microsaccade_minus_drift": micro - drift if np.isfinite(micro) and np.isfinite(drift) else float("nan"),
                "microsaccade_over_drift": micro / drift if np.isfinite(micro) and np.isfinite(drift) and abs(drift) >= EPS else float("nan"),
            }
        )
    return out


def _summarize_spectrum_controls(band_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted(
        {
            (
                str(row["stimulus_condition"]),
                str(row["trajectory_condition"]),
                float(row["band_cpd_low"]),
                float(row["band_cpd_high"]),
            )
            for row in band_rows
        }
    )
    out: list[dict[str, object]] = []
    for stim_cond, condition, lo, hi in keys:
        group = [
            row
            for row in band_rows
            if row["stimulus_condition"] == stim_cond
            and row["trajectory_condition"] == condition
            and float(row["band_cpd_low"]) == lo
            and float(row["band_cpd_high"]) == hi
        ]
        out.append(
            {
                "stimulus_condition": stim_cond,
                "trajectory_condition": condition,
                "band_cpd_low": lo,
                "band_cpd_high": hi,
                "n_movies": len(group),
                "mean_band_temporal_contrast": _mean_or_nan(float(row["band_temporal_contrast_mean"]) for row in group),
                "mean_band_temporal_contrast_energy": _mean_or_nan(
                    float(row["band_temporal_contrast_energy"]) for row in group
                ),
            }
        )
    return out


def _summarize_phase_controls(
    retinal_rows: list[dict[str, object]],
    info_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if "phase_scrambled" not in {str(row["stimulus_condition"]) for row in retinal_rows}:
        return [
            {
                "comparison": "phase_scrambled_minus_intact",
                "status": "not_run",
                "detail": "Add --stimulus-conditions intact,phase_scrambled to compute this diagnostic.",
            }
        ]

    out: list[dict[str, object]] = []
    retinal_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    for row in retinal_rows:
        key = (row["image_id"], row["trace_id"], row["trajectory_condition"], row["stimulus_condition"])
        retinal_by_key[key] = row

    conditions = sorted({str(row["trajectory_condition"]) for row in retinal_rows})
    for condition in conditions:
        diffs = []
        for key, phase_row in retinal_by_key.items():
            image_id, trace_id, traj_condition, stim_cond = key
            if traj_condition != condition or stim_cond != "phase_scrambled":
                continue
            intact = retinal_by_key.get((image_id, trace_id, traj_condition, "intact"))
            if intact is None:
                continue
            diffs.append(float(phase_row["temporal_contrast_mean"]) - float(intact["temporal_contrast_mean"]))
        out.append(
            {
                "comparison": "phase_scrambled_minus_intact",
                "trajectory_condition": condition,
                "metric": "temporal_contrast_mean",
                "n_pairs": len(diffs),
                "mean_difference": _mean_or_nan(diffs),
            }
        )

    final_info = [row for row in info_rows if bool(row.get("is_final_time", False))]
    info_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    for row in final_info:
        key = (row["image_id"], row["trace_id"], row["trajectory_condition"], row["stimulus_condition"], row["metric"])
        info_by_key[key] = row
    for condition in conditions:
        metric_names = sorted({str(row["metric"]) for row in final_info})
        for metric_name in metric_names:
            diffs = []
            for key, phase_row in info_by_key.items():
                image_id, trace_id, traj_condition, stim_cond, metric = key
                if traj_condition != condition or stim_cond != "phase_scrambled" or metric != metric_name:
                    continue
                intact = info_by_key.get((image_id, trace_id, traj_condition, "intact", metric_name))
                if intact is None:
                    continue
                diffs.append(float(phase_row["cumulative_information"]) - float(intact["cumulative_information"]))
            out.append(
                {
                    "comparison": "phase_scrambled_minus_intact",
                    "trajectory_condition": condition,
                    "metric": metric_name,
                    "n_pairs": len(diffs),
                    "mean_difference": _mean_or_nan(diffs),
                }
            )
    return out


def _write_summary_figure(
    out_path: Path,
    retinal_rows: list[dict[str, object]],
    info_rows: list[dict[str, object]],
    gain_rows: list[dict[str, object]],
) -> None:
    if not retinal_rows or not info_rows:
        return

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    axes = axes.ravel()

    conditions = sorted({str(row["trajectory_condition"]) for row in retinal_rows})
    tc_means = [
        np.nanmean([float(row["temporal_contrast_mean"]) for row in retinal_rows if row["trajectory_condition"] == cond])
        for cond in conditions
    ]
    axes[0].bar(conditions, tc_means, color="#4C78A8")
    axes[0].set_ylabel("Temporal contrast")
    axes[0].set_title("Retinal movie modulation")
    axes[0].tick_params(axis="x", rotation=25)

    grouped: dict[str, dict[int, list[float]]] = {}
    for row in info_rows:
        cond = str(row["trajectory_condition"])
        frame = int(row["time_frame"])
        grouped.setdefault(cond, {}).setdefault(frame, []).append(float(row["cumulative_information"]))
    for cond in conditions:
        if cond not in grouped:
            continue
        frames = sorted(grouped[cond])
        values = [_mean_or_nan(grouped[cond][frame]) for frame in frames]
        axes[1].plot(frames, values, marker="o", ms=2.5, lw=1.5, label=cond)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Cumulative metric")
    axes[1].set_title("Paired movie endpoint")
    axes[1].legend(frameon=False, fontsize=8)

    gain_plot_rows = [row for row in gain_rows if row["stimulus_condition"] == "all"]
    labels = [str(row["comparison"]).replace("real_FEM_minus_", "") for row in gain_plot_rows]
    gains = [float(row["mean_gain"]) for row in gain_plot_rows]
    lo = [float(row["mean_gain"]) - float(row["ci95_low"]) for row in gain_plot_rows]
    hi = [float(row["ci95_high"]) - float(row["mean_gain"]) for row in gain_plot_rows]
    if labels:
        axes[2].bar(labels, gains, color="#F58518")
        axes[2].errorbar(labels, gains, yerr=[lo, hi], fmt="none", ecolor="0.2", capsize=3)
    axes[2].axhline(0, color="0.2", lw=0.8)
    axes[2].set_ylabel("Real minus control")
    axes[2].set_title("Paired final gain")
    axes[2].tick_params(axis="x", rotation=25)

    peak_freqs = [
        float(row["peak_temporal_frequency_hz"])
        for row in retinal_rows
        if str(row["trajectory_condition"]) == "real_FEM"
    ]
    axes[3].hist(peak_freqs, bins=12, color="#54A24B", alpha=0.85)
    axes[3].set_xlabel("Peak temporal frequency (Hz)")
    axes[3].set_ylabel("Movies")
    axes[3].set_title("Real FEM temporal spectrum")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.18, lw=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_root) / args.run_label
    for subdir in ("movies", "responses", "metrics", "figures", "qc", "logs", "summaries"):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    conditions = _parse_csv_strs(args.conditions)
    stimulus_conditions = _parse_csv_strs(args.stimulus_conditions)
    bands = _parse_bands(args.bands_cpd)
    if "real_FEM" not in conditions:
        conditions = ("real_FEM", *conditions)

    config = RunnerConfig(
        run_label=args.run_label,
        out_dir=str(out_dir),
        stimulus_source=args.stimulus_source,
        n_images=args.n_images,
        n_traces=args.n_traces,
        n_frames=args.n_frames,
        image_size=args.image_size,
        ppd=args.ppd,
        frame_rate_hz=args.frame_rate_hz,
        conditions=conditions,
        stimulus_conditions=stimulus_conditions,
        trace_source=args.trace_source,
        trace_cache=str(args.trace_cache),
        run_model=bool(args.run_model),
        model_metric=args.primary_model_metric if args.run_model else "retinal_temporal_power_proxy",
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )
    _write_json(out_dir / "run_config.json", asdict(config))

    images, image_source_detail = _load_image_stack(args, rng=rng)
    image_rows = [
        {
            "image_id": i,
            "stimulus_source": args.stimulus_source,
            "source_detail": image_source_detail,
            "rms_contrast": float(np.std(images[i])),
            "mean_luminance_after_standardization": float(np.mean(images[i])),
        }
        for i in range(images.shape[0])
    ]
    _write_csv(out_dir / "stimulus_condition_manifest.csv", image_rows)

    model_bundle = _load_model_bundle(args) if args.run_model else None
    trace_pool = _load_cached_trace_pool(args.trace_source, args.trace_cache, args.n_frames)

    retinal_rows: list[dict[str, object]] = []
    frequency_rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []
    info_rows: list[dict[str, object]] = []
    model_efficiency_rows: list[dict[str, object]] = []
    movie_manifest_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []

    time_stride = max(1, int(args.time_stride))
    for trace_id in range(args.n_traces):
        real_trace, state, real_trace_detail = _get_real_trace(args, trace_pool, trace_id, rng)
        step_arcmin = np.linalg.norm(np.diff(real_trace, axis=0), axis=1) * 60.0
        trajectory_rows.append(
            {
                "trace_id": trace_id,
                "trace_source": args.trace_source,
                "trace_detail": real_trace_detail,
                "n_frames": args.n_frames,
                "mean_step_arcmin": float(np.mean(step_arcmin)) if step_arcmin.size else 0.0,
                "p95_step_arcmin": float(np.percentile(step_arcmin, 95)) if step_arcmin.size else 0.0,
                "path_length_arcmin": float(np.sum(step_arcmin)) if step_arcmin.size else 0.0,
                "rms_position_arcmin": float(np.sqrt(np.mean(np.sum(real_trace * real_trace, axis=1))) * 60.0),
                "microsaccade_frame_fraction": float(np.mean(state > 0)),
            }
        )

        control_traces = {
            condition: _make_control_trace(real_trace, condition, rng=rng) for condition in conditions
        }
        if "real_FEM" in control_traces:
            control_traces["real_FEM"] = (control_traces["real_FEM"][0], real_trace_detail)

        for image_id, base_image in enumerate(images):
            stim_images: dict[str, np.ndarray] = {}
            for stim_cond in stimulus_conditions:
                if stim_cond == "intact":
                    stim_images[stim_cond] = base_image
                elif stim_cond == "phase_scrambled":
                    stim_images[stim_cond] = _phase_scramble_image(base_image, rng=rng)
                else:
                    raise ValueError(f"Unknown stimulus condition: {stim_cond}")

            for stim_cond, stim_image in stim_images.items():
                stabilized_trace, _ = control_traces.get(
                    "stabilized",
                    _make_control_trace(real_trace, "stabilized", rng=rng),
                )
                stabilized_movie = _render_movie(stim_image, stabilized_trace, ppd=args.ppd, out_size=args.image_size)

                for condition, (trace, trace_detail) in control_traces.items():
                    movie = _render_movie(stim_image, trace, ppd=args.ppd, out_size=args.image_size)
                    diag, freq = _compute_retinal_rows(
                        image_id=image_id,
                        trace_id=trace_id,
                        stimulus_condition=stim_cond,
                        condition=condition,
                        movie=movie,
                        stabilized_movie=stabilized_movie,
                        trace=trace,
                        state=state,
                        frame_rate_hz=args.frame_rate_hz,
                    )
                    retinal_rows.append(diag)
                    frequency_rows.append(freq)
                    if condition in {"real_FEM", "stabilized", "random_amp"}:
                        band_rows.extend(
                            _band_rows_for_movie(
                                stim_image,
                                trace,
                                ppd=args.ppd,
                                out_size=args.image_size,
                                bands=bands,
                                image_id=image_id,
                                trace_id=trace_id,
                                stimulus_condition=stim_cond,
                                trajectory_condition=condition,
                            )
                        )

                    metric = "retinal_temporal_power_proxy"
                    metric_status = "proxy_retinal_not_model"
                    if model_bundle is not None:
                        series = _model_spatial_ssi_series(
                            movie,
                            bundle=model_bundle,
                            frame_rate_hz=args.frame_rate_hz,
                        )
                        if args.primary_model_metric not in series:
                            raise ValueError(
                                f"Unknown primary model metric {args.primary_model_metric!r}; "
                                f"available metrics: {sorted(series)}"
                            )
                        cumulative = series[args.primary_model_metric]
                        metric = args.primary_model_metric
                        metric_status = "model_spatial_ssi_efficiency"
                    else:
                        cumulative = _cumulative_proxy_from_movie(movie, frame_rate_hz=args.frame_rate_hz)
                        series = {metric: cumulative}

                    time_indices = list(range(0, len(cumulative), time_stride))
                    if len(cumulative) - 1 not in time_indices:
                        time_indices.append(len(cumulative) - 1)
                    for time_idx in time_indices:
                        if model_bundle is not None:
                            for series_metric, values in series.items():
                                model_efficiency_rows.append(
                                    {
                                        "image_id": image_id,
                                        "trace_id": trace_id,
                                        "stimulus_condition": stim_cond,
                                        "trajectory_condition": condition,
                                        "metric": series_metric,
                                        "time_frame": int(time_idx),
                                        "time_s": float(time_idx / args.frame_rate_hz),
                                        "value": float(values[time_idx]),
                                        "is_final_time": bool(time_idx == len(cumulative) - 1),
                                    }
                                )
                        info_rows.append(
                            {
                                "image_id": image_id,
                                "trace_id": trace_id,
                                "stimulus_condition": stim_cond,
                                "trajectory_condition": condition,
                                "metric": metric,
                                "metric_status": metric_status,
                                "time_frame": int(time_idx),
                                "time_s": float(time_idx / args.frame_rate_hz),
                                "cumulative_information": float(cumulative[time_idx]),
                                "is_final_time": bool(time_idx == len(cumulative) - 1),
                            }
                        )

                    movie_manifest_rows.append(
                        {
                            "image_id": image_id,
                            "trace_id": trace_id,
                            "stimulus_condition": stim_cond,
                            "trajectory_condition": condition,
                            "trace_detail": trace_detail,
                            "n_frames": args.n_frames,
                            "image_size": args.image_size,
                            "ppd": args.ppd,
                            "frame_rate_hz": args.frame_rate_hz,
                        }
                    )

    gain_rows = _summarize_gains(info_rows, rng=rng, n_bootstrap=args.n_bootstrap)
    drift_micro_rows = _summarize_drift_microsaccades(retinal_rows)
    spectrum_rows = _summarize_spectrum_controls(band_rows)
    phase_rows = _summarize_phase_controls(retinal_rows, info_rows)
    decision_rows = [
        {
            "criterion": "primary_endpoint",
            "status": "computed",
            "detail": "paired real-vs-control cumulative information gain over matched image/trace movies; default model endpoint penalizes spike-rate increases via bits per expected spike",
            "metric": config.model_metric,
        },
        {
            "criterion": "model_endpoint",
            "status": "computed" if args.run_model else "not_run",
            "detail": "Use --run-model to compute digital-twin spatial SSI; default smoke mode writes a retinal proxy.",
            "metric": config.model_metric,
        },
        {
            "criterion": "retinal_movie_transform",
            "status": "computed",
            "detail": "temporal contrast, motion power, temporal spectrum, and band-specific modulation",
            "metric": "retinal_movie_diagnostics",
        },
    ]

    _write_csv(out_dir / "trajectory_qc.csv", trajectory_rows)
    _write_csv(out_dir / "movie_condition_manifest.csv", movie_manifest_rows)
    _write_csv(out_dir / "metrics" / "retinal_movie_diagnostics.csv", retinal_rows)
    _write_csv(out_dir / "metrics" / "retinal_movie_frequency_summary.csv", frequency_rows)
    _write_csv(out_dir / "metrics" / "retinal_movie_band_summary.csv", band_rows)
    _write_csv(out_dir / "metrics" / "cumulative_information_by_movie.csv", info_rows)
    _write_csv(out_dir / "metrics" / "model_efficiency_by_movie.csv", model_efficiency_rows)
    _write_csv(out_dir / "metrics" / "information_gain_summary.csv", gain_rows)
    _write_csv(out_dir / "metrics" / "drift_microsaccade_decomposition.csv", drift_micro_rows)
    _write_csv(out_dir / "metrics" / "spectrum_control_summary.csv", spectrum_rows)
    _write_csv(out_dir / "metrics" / "phase_control_summary.csv", phase_rows)
    _write_csv(out_dir / "metrics" / "bootstrap_summary.csv", gain_rows)
    _write_csv(out_dir / "metrics" / "decision_table.csv", decision_rows)
    _write_json(
        out_dir / "summaries" / "run_summary.json",
        {
            "out_dir": str(out_dir),
            "n_retinal_movies": len(retinal_rows),
            "n_cumulative_rows": len(info_rows),
            "n_gain_rows": len(gain_rows),
            "metric": config.model_metric,
            "run_model": args.run_model,
        },
    )
    _write_summary_figure(out_dir / "figures" / "active_sensing_movie_information_summary.png", retinal_rows, info_rows, gain_rows)
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", default="smoke", help="Output subdirectory label.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--stimulus-source",
        choices=("synthetic", "nat_stack", "fixrsvp_stack"),
        default="synthetic",
        help="Image source. Synthetic is dependency-light and intended for smoke tests.",
    )
    parser.add_argument("--n-images", type=int, default=6)
    parser.add_argument("--n-traces", type=int, default=4)
    parser.add_argument("--n-frames", type=int, default=80)
    parser.add_argument(
        "--trace-source",
        choices=("synthetic", "fixrsvp_fixation_pool", "fixrsvp_cached_eyepos"),
        default="fixrsvp_fixation_pool",
        help="Eye-trace source for real_FEM. Use synthetic only for dependency-light smoke tests.",
    )
    parser.add_argument(
        "--trace-cache",
        type=Path,
        default=DEFAULT_TRACE_CACHE,
        help="Pickle cache for measured traces. Defaults to declan/fixrsvp_fixation_pool.pkl.",
    )
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--ppd", type=float, default=37.50476617)
    parser.add_argument("--frame-rate-hz", type=float, default=120.0)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--stimulus-conditions", default=",".join(DEFAULT_STIM_CONDITIONS))
    parser.add_argument(
        "--bands-cpd",
        default=",".join(f"{lo:g}-{hi:g}" for lo, hi in DEFAULT_BANDS_CPD),
        help="Comma-separated radial spatial-frequency bands in cycles/deg, e.g. '0-2,2-8,8-18'.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--time-stride", type=int, default=4)

    parser.add_argument("--run-model", action="store_true", help="Run the digital twin and compute spatial SSI efficiency metrics.")
    parser.add_argument(
        "--primary-model-metric",
        default=DEFAULT_PRIMARY_MODEL_METRIC,
        choices=(
            "cumulative_spatial_bits_per_expected_spike",
            "cumulative_spatial_bits",
            "mean_spatial_bits_per_sec_to_date",
        ),
        help="Metric used for paired gain summaries when --run-model is set.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--mcfarland-outputs", type=Path, default=DEFAULT_MCFARLAND_OUTPUTS)
    parser.add_argument("--model-type", default="resnet_none_convgru")
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument(
        "--cfg-dir-override",
        default="experiments/dataset_configs/multi_basic_120_long_legacy.yaml",
        help="Dataset config override passed to load_model; use '' to disable.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-lags", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    out_dir = run(args)
    print(f"Wrote active-sensing movie-information pilot outputs to {out_dir}")


if __name__ == "__main__":
    main()
