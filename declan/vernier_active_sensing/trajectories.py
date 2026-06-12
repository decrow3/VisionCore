"""Retinal phase and motion-condition builders for Vernier active sensing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_EYE_TRACES_PATH = Path("scripts") / "temporal_decoding" / "data" / "eye_traces.npz"
NOMINAL_CENTER_PHASE_DEG = np.zeros(2, dtype=np.float32)
FRAME_RATE_HZ = 120.0
DEFAULT_MICROSACCADE_SPEED_THRESHOLD_DPS = 30.0
DEFAULT_MICROSACCADE_PAD_FRAMES = 1


@dataclass(frozen=True)
class TraceSet:
    traces: np.ndarray
    durations: np.ndarray


def load_eye_traces(path: Path = DEFAULT_EYE_TRACES_PATH) -> TraceSet:
    data = np.load(path, allow_pickle=True)
    return TraceSet(traces=np.asarray(data["traces"], dtype=np.float32), durations=np.asarray(data["durations"], dtype=np.int32))


def subsample_traces(trace_set: TraceSet, n_traces: int | None, seed: int) -> TraceSet:
    if n_traces is None or int(n_traces) <= 0 or int(n_traces) >= trace_set.traces.shape[0]:
        return trace_set
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(trace_set.traces.shape[0], size=int(n_traces), replace=False))
    return TraceSet(trace_set.traces[idx], trace_set.durations[idx])


def valid_trace(trace_set: TraceSet, index: int, max_frames: int | None = None) -> np.ndarray:
    t = int(trace_set.durations[index])
    if max_frames is not None and int(max_frames) > 0:
        t = min(t, int(max_frames))
    return np.asarray(trace_set.traces[index, :t], dtype=np.float32)


def grand_mean_phase(trace_set: TraceSet) -> np.ndarray:
    chunks = [trace_set.traces[i, : int(trace_set.durations[i])] for i in range(trace_set.traces.shape[0])]
    return np.mean(np.concatenate(chunks, axis=0), axis=0).astype(np.float32)


def all_phase_samples(trace_set: TraceSet) -> np.ndarray:
    return np.concatenate([trace_set.traces[i, : int(trace_set.durations[i])] for i in range(trace_set.traces.shape[0])], axis=0).astype(np.float32)


def condition_trace(
    trace: np.ndarray,
    *,
    condition: str,
    trace_set: TraceSet | None = None,
    rng: np.random.Generator | None = None,
    scale: float | None = None,
    frame_rate_hz: float = FRAME_RATE_HZ,
    microsaccade_speed_threshold_dps: float = DEFAULT_MICROSACCADE_SPEED_THRESHOLD_DPS,
    microsaccade_pad_frames: int = DEFAULT_MICROSACCADE_PAD_FRAMES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the effective eye trace for one named condition.

    Conditions are intentionally close to the analysis-plan vocabulary.  The
    returned trace is in degrees and has the same length as ``trace``.
    """
    rng = rng or np.random.default_rng(0)
    eye = np.asarray(trace, dtype=np.float32)
    mean = np.mean(eye, axis=0, keepdims=True).astype(np.float32)
    condition = str(condition)

    if condition in {"real_fem", "real", "movie_real"}:
        return eye.copy(), {"condition_family": "real", "scale": 1.0}
    if condition in {"static_center", "fixed_center"}:
        phase = NOMINAL_CENTER_PHASE_DEG
        return np.broadcast_to(phase[None, :], eye.shape).copy(), {"condition_family": "static", "phase_source": "nominal_zero"}
    if condition in {"static_repeated_phase", "stabilized", "trial_mean_stabilized"}:
        return np.broadcast_to(mean, eye.shape).copy(), {"condition_family": "static", "phase_source": "trial_mean"}
    if condition == "static_phase_cloud_single":
        cloud = all_phase_samples(trace_set) if trace_set is not None else eye
        phase = cloud[int(rng.integers(0, cloud.shape[0]))]
        return np.broadcast_to(phase[None, :], eye.shape).copy(), {"condition_family": "static", "phase_source": "cloud_draw"}
    if condition == "static_phase_cloud_matched_positions":
        idx = np.arange(eye.shape[0])
        rng.shuffle(idx)
        return eye[idx].astype(np.float32, copy=True), {
            "condition_family": "cloud",
            "phase_source": "same_trace_positions_shuffled",
            "paired_phase_set": True,
        }
    if condition == "order_shuffled_positions":
        idx = np.arange(eye.shape[0])
        rng.shuffle(idx)
        return eye[idx].astype(np.float32, copy=True), {"condition_family": "trajectory_control", "phase_source": "same_positions_shuffled"}
    if condition == "axis_horizontal":
        out = np.broadcast_to(mean, eye.shape).copy()
        out[:, 0] = eye[:, 0]
        return out.astype(np.float32), {"condition_family": "axis", "axis": "horizontal"}
    if condition == "axis_vertical":
        out = np.broadcast_to(mean, eye.shape).copy()
        out[:, 1] = eye[:, 1]
        return out.astype(np.float32), {"condition_family": "axis", "axis": "vertical"}
    if condition.startswith("scaled_real") or condition.startswith("scaled_"):
        s = _parse_scale(condition) if scale is None else float(scale)
        return (mean + (eye - mean) * s).astype(np.float32), {"condition_family": "scaled", "scale": float(s)}
    component = _component_condition(condition)
    if component is not None:
        drift_scale, microsaccade_scale, family, scale_value = component
        out, meta = component_scaled_trace(
            eye,
            drift_scale=drift_scale,
            microsaccade_scale=microsaccade_scale,
            frame_rate_hz=frame_rate_hz,
            microsaccade_speed_threshold_dps=microsaccade_speed_threshold_dps,
            microsaccade_pad_frames=microsaccade_pad_frames,
        )
        meta.update({"condition_family": family, "scale": float(scale_value)})
        return out, meta
    if condition == "random_amp_matched":
        steps = np.diff(eye, axis=0, prepend=eye[:1])
        norms = np.linalg.norm(steps, axis=1)
        angles = rng.uniform(0.0, 2.0 * np.pi, size=eye.shape[0])
        rand_steps = np.stack([norms * np.cos(angles), norms * np.sin(angles)], axis=1).astype(np.float32)
        out = mean + np.cumsum(rand_steps - np.mean(rand_steps, axis=0, keepdims=True), axis=0)
        return out.astype(np.float32), {"condition_family": "random", "phase_source": "step_amplitude_matched"}
    if condition == "random_cloud_matched":
        cloud = all_phase_samples(trace_set) if trace_set is not None else eye
        idx = rng.integers(0, cloud.shape[0], size=eye.shape[0])
        out = cloud[idx].astype(np.float32)
        return out, {"condition_family": "random", "phase_source": "cloud_iid"}

    raise ValueError(f"Unsupported Vernier motion condition: {condition}")


def _parse_scale(condition: str) -> float:
    if condition == "scaled_real":
        return 1.0
    for prefix in ("scaled_real_", "scaled_"):
        if condition.startswith(prefix):
            return float(condition[len(prefix) :])
    raise ValueError(f"Could not parse scale from {condition!r}")


def _component_condition(condition: str) -> tuple[float, float, str, float] | None:
    if condition == "drift_only":
        return 1.0, 0.0, "drift_only", 1.0
    if condition == "microsaccade_only":
        return 0.0, 1.0, "microsaccade_only", 1.0
    if condition.startswith("drift_only_scaled_"):
        scale = float(condition[len("drift_only_scaled_") :])
        return scale, 0.0, "drift_only", scale
    if condition.startswith("microsaccade_only_scaled_"):
        scale = float(condition[len("microsaccade_only_scaled_") :])
        return 0.0, scale, "microsaccade_only", scale
    if condition.startswith("drift_scaled_"):
        scale = float(condition[len("drift_scaled_") :])
        return scale, 1.0, "drift_scaled_with_real_microsaccades", scale
    if condition.startswith("microsaccade_scaled_"):
        scale = float(condition[len("microsaccade_scaled_") :])
        return 1.0, scale, "microsaccade_scaled_with_real_drift", scale
    return None


def microsaccade_step_mask(
    trace: np.ndarray,
    *,
    frame_rate_hz: float = FRAME_RATE_HZ,
    speed_threshold_dps: float = DEFAULT_MICROSACCADE_SPEED_THRESHOLD_DPS,
    pad_frames: int = DEFAULT_MICROSACCADE_PAD_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect microsaccade-like steps with a transparent speed threshold."""
    eye = np.asarray(trace, dtype=np.float32)
    if eye.shape[0] < 2:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32)
    steps = np.diff(eye, axis=0)
    speeds = np.linalg.norm(steps * float(frame_rate_hz), axis=1).astype(np.float32)
    mask = speeds > float(speed_threshold_dps)
    pad = max(int(pad_frames), 0)
    if pad > 0 and mask.any():
        expanded = mask.copy()
        idx = np.flatnonzero(mask)
        for i in idx:
            lo = max(0, int(i) - pad)
            hi = min(mask.size, int(i) + pad + 1)
            expanded[lo:hi] = True
        mask = expanded
    return mask, speeds


def component_scaled_trace(
    trace: np.ndarray,
    *,
    drift_scale: float,
    microsaccade_scale: float,
    frame_rate_hz: float = FRAME_RATE_HZ,
    microsaccade_speed_threshold_dps: float = DEFAULT_MICROSACCADE_SPEED_THRESHOLD_DPS,
    microsaccade_pad_frames: int = DEFAULT_MICROSACCADE_PAD_FRAMES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale drift and microsaccade-like displacement components separately."""
    eye = np.asarray(trace, dtype=np.float32)
    mean = np.mean(eye, axis=0, keepdims=True).astype(np.float32)
    if eye.shape[0] < 2:
        return eye.copy(), {
            "component_method": "velocity_threshold",
            "microsaccade_speed_threshold_dps": float(microsaccade_speed_threshold_dps),
            "microsaccade_pad_frames": int(microsaccade_pad_frames),
            "n_microsaccade_steps": 0,
            "fraction_microsaccade_steps": 0.0,
            "drift_scale": float(drift_scale),
            "microsaccade_scale": float(microsaccade_scale),
        }

    event_mask, speeds = microsaccade_step_mask(
        eye,
        frame_rate_hz=frame_rate_hz,
        speed_threshold_dps=microsaccade_speed_threshold_dps,
        pad_frames=microsaccade_pad_frames,
    )
    steps = np.diff(eye, axis=0)
    drift_steps = steps.copy()
    microsaccade_steps = np.zeros_like(steps)
    drift_steps[event_mask] = 0.0
    microsaccade_steps[event_mask] = steps[event_mask]

    drift_path = np.vstack([np.zeros((1, 2), dtype=np.float32), np.cumsum(drift_steps, axis=0)])
    microsaccade_path = np.vstack([np.zeros((1, 2), dtype=np.float32), np.cumsum(microsaccade_steps, axis=0)])
    drift_centered = drift_path - np.mean(drift_path, axis=0, keepdims=True)
    microsaccade_centered = microsaccade_path - np.mean(microsaccade_path, axis=0, keepdims=True)
    out = mean + float(drift_scale) * drift_centered + float(microsaccade_scale) * microsaccade_centered

    reconstructed_real = mean + drift_centered + microsaccade_centered
    reconstruction_error = float(np.max(np.abs(reconstructed_real - eye)))
    meta = {
        "component_method": "velocity_threshold",
        "microsaccade_speed_threshold_dps": float(microsaccade_speed_threshold_dps),
        "microsaccade_pad_frames": int(microsaccade_pad_frames),
        "frame_rate_hz": float(frame_rate_hz),
        "n_microsaccade_steps": int(np.sum(event_mask)),
        "fraction_microsaccade_steps": float(np.mean(event_mask)) if event_mask.size else 0.0,
        "max_speed_dps": float(np.max(speeds)) if speeds.size else 0.0,
        "p95_speed_dps": float(np.percentile(speeds, 95)) if speeds.size else 0.0,
        "drift_scale": float(drift_scale),
        "microsaccade_scale": float(microsaccade_scale),
        "component_reconstruction_max_abs_error_deg": reconstruction_error,
    }
    return out.astype(np.float32), meta
