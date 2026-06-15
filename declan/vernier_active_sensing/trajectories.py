"""Retinal phase and motion-condition builders for Vernier active sensing."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_EYE_TRACES_PATH = Path("scripts") / "temporal_decoding" / "data" / "eye_traces.npz"
NOMINAL_CENTER_PHASE_DEG = np.zeros(2, dtype=np.float32)
FRAME_RATE_HZ = 120.0
DEFAULT_MICROSACCADE_SPEED_THRESHOLD_DPS: float | None = None
DEFAULT_MICROSACCADE_THRESHOLD_Z = 6.0
DEFAULT_MICROSACCADE_PAD_FRAMES = 1


@dataclass(frozen=True)
class TraceSet:
    traces: np.ndarray
    durations: np.ndarray


@dataclass(frozen=True)
class MicrosaccadeDetection:
    step_mask: np.ndarray
    sample_mask: np.ndarray
    speeds_dps: np.ndarray
    threshold_dps: float
    events: list[dict[str, int | float]]


@cache
def _jake_eye_control_functions() -> tuple[Any, Any]:
    from jake.twininfo.eye_controls import detect_microsaccade_events, speed_threshold_mad

    return detect_microsaccade_events, speed_threshold_mad


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
    microsaccade_speed_threshold_dps: float | None = DEFAULT_MICROSACCADE_SPEED_THRESHOLD_DPS,
    microsaccade_threshold_z: float = DEFAULT_MICROSACCADE_THRESHOLD_Z,
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
    matched_scale = _scale_matched_condition(condition)
    if matched_scale is not None:
        family, s = matched_scale
        scaled = _scaled_trace(eye, float(s), mean=mean)
        idx = np.arange(scaled.shape[0])
        rng.shuffle(idx)
        if family == "scaled_phase_cloud":
            return scaled[idx].astype(np.float32, copy=True), {
                "condition_family": "cloud",
                "phase_source": "same_trace_scaled_positions_shuffled",
                "paired_phase_set": True,
                "scale": float(s),
                "scale_matched_to": "scaled_real",
            }
        return scaled[idx].astype(np.float32, copy=True), {
            "condition_family": "trajectory_control",
            "phase_source": "same_scaled_positions_shuffled",
            "scale": float(s),
            "scale_matched_to": "scaled_real",
        }
    if condition.startswith("scaled_real") or condition.startswith("scaled_"):
        s = _parse_scale(condition) if scale is None else float(scale)
        return _scaled_trace(eye, float(s), mean=mean), {"condition_family": "scaled", "scale": float(s)}
    component = _component_condition(condition)
    if component is not None:
        drift_scale, microsaccade_scale, family, scale_value = component
        out, meta = component_scaled_trace(
            eye,
            drift_scale=drift_scale,
            microsaccade_scale=microsaccade_scale,
            frame_rate_hz=frame_rate_hz,
            microsaccade_speed_threshold_dps=microsaccade_speed_threshold_dps,
            microsaccade_threshold_z=microsaccade_threshold_z,
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


def _scaled_trace(trace: np.ndarray, scale: float, *, mean: np.ndarray | None = None) -> np.ndarray:
    eye = np.asarray(trace, dtype=np.float32)
    center = np.mean(eye, axis=0, keepdims=True).astype(np.float32) if mean is None else np.asarray(mean, dtype=np.float32)
    return (center + (eye - center) * float(scale)).astype(np.float32)


def _scale_matched_condition(condition: str) -> tuple[str, float] | None:
    prefixes = (
        ("scaled_phase_cloud_matched_positions_", "scaled_phase_cloud"),
        ("static_phase_cloud_matched_scaled_", "scaled_phase_cloud"),
        ("scaled_order_shuffled_positions_", "scaled_order_shuffled"),
        ("order_shuffled_scaled_", "scaled_order_shuffled"),
    )
    for prefix, family in prefixes:
        if condition.startswith(prefix):
            return family, float(condition[len(prefix) :])
    return None


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
    speed_threshold_dps: float | None = DEFAULT_MICROSACCADE_SPEED_THRESHOLD_DPS,
    threshold_z: float = DEFAULT_MICROSACCADE_THRESHOLD_Z,
    pad_frames: int = DEFAULT_MICROSACCADE_PAD_FRAMES,
) -> MicrosaccadeDetection:
    """Detect microsaccade-like steps with Jake's robust event detector."""
    eye = np.asarray(trace, dtype=np.float32)
    empty = np.zeros(0, dtype=np.float32)
    if eye.shape[0] < 2:
        return MicrosaccadeDetection(
            step_mask=np.zeros(0, dtype=bool),
            sample_mask=np.zeros(eye.shape[0], dtype=bool),
            speeds_dps=empty,
            threshold_dps=float("nan"),
            events=[],
        )
    dt = 1.0 / float(frame_rate_hz)
    steps = np.diff(eye, axis=0)
    speeds = np.linalg.norm(steps * float(frame_rate_hz), axis=1).astype(np.float32)
    detect_microsaccade_events, speed_threshold_mad = _jake_eye_control_functions()
    if speed_threshold_dps is None:
        speed_threshold_dps = speed_threshold_mad(eye, dt=dt, z=float(threshold_z))
    events, sample_mask, threshold = detect_microsaccade_events(
        eye,
        dt=dt,
        threshold_deg_s=float(speed_threshold_dps),
        min_samples=1,
        pad_samples=max(int(pad_frames), 0),
    )
    return MicrosaccadeDetection(
        step_mask=np.asarray(sample_mask[1:], dtype=bool),
        sample_mask=np.asarray(sample_mask, dtype=bool),
        speeds_dps=speeds,
        threshold_dps=float(threshold),
        events=events,
    )


def component_scaled_trace(
    trace: np.ndarray,
    *,
    drift_scale: float,
    microsaccade_scale: float,
    frame_rate_hz: float = FRAME_RATE_HZ,
    microsaccade_speed_threshold_dps: float | None = DEFAULT_MICROSACCADE_SPEED_THRESHOLD_DPS,
    microsaccade_threshold_z: float = DEFAULT_MICROSACCADE_THRESHOLD_Z,
    microsaccade_pad_frames: int = DEFAULT_MICROSACCADE_PAD_FRAMES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale drift and microsaccade-like displacement components separately."""
    eye = np.asarray(trace, dtype=np.float32)
    mean = np.mean(eye, axis=0, keepdims=True).astype(np.float32)
    if eye.shape[0] < 2:
        return eye.copy(), {
            "component_method": "jake.twininfo.eye_controls.detect_microsaccade_events",
            "microsaccade_speed_threshold_dps": (
                float(microsaccade_speed_threshold_dps) if microsaccade_speed_threshold_dps is not None else float("nan")
            ),
            "microsaccade_threshold_z": float(microsaccade_threshold_z),
            "microsaccade_pad_frames": int(microsaccade_pad_frames),
            "n_microsaccade_events": 0,
            "n_microsaccade_steps": 0,
            "fraction_microsaccade_steps": 0.0,
            "drift_scale": float(drift_scale),
            "microsaccade_scale": float(microsaccade_scale),
        }

    detection = microsaccade_step_mask(
        eye,
        frame_rate_hz=frame_rate_hz,
        speed_threshold_dps=microsaccade_speed_threshold_dps,
        threshold_z=microsaccade_threshold_z,
        pad_frames=microsaccade_pad_frames,
    )
    event_mask = detection.step_mask
    speeds = detection.speeds_dps
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
        "component_method": "jake.twininfo.eye_controls.detect_microsaccade_events",
        "microsaccade_speed_threshold_dps": float(detection.threshold_dps),
        "microsaccade_threshold_z": float(microsaccade_threshold_z),
        "microsaccade_threshold_source": "fixed" if microsaccade_speed_threshold_dps is not None else "mad",
        "microsaccade_pad_frames": int(microsaccade_pad_frames),
        "frame_rate_hz": float(frame_rate_hz),
        "n_microsaccade_events": int(len(detection.events)),
        "n_microsaccade_steps": int(np.sum(event_mask)),
        "fraction_microsaccade_steps": float(np.mean(event_mask)) if event_mask.size else 0.0,
        "fraction_microsaccade_samples": float(np.mean(detection.sample_mask)) if detection.sample_mask.size else 0.0,
        "max_speed_dps": float(np.max(speeds)) if speeds.size else 0.0,
        "p95_speed_dps": float(np.percentile(speeds, 95)) if speeds.size else 0.0,
        "drift_scale": float(drift_scale),
        "microsaccade_scale": float(microsaccade_scale),
        "component_reconstruction_max_abs_error_deg": reconstruction_error,
    }
    return out.astype(np.float32), meta
