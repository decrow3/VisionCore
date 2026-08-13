"""Counterfactual eye-trajectory retiming utilities.

This module is intentionally model-free. It builds fixed-geometry retimed
histories and records the timing/velocity metrics needed before expensive twin
inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np


EPS = 1e-12
MODEL_RATE_HZ = 120.0
MODEL_NYQUIST_HZ = MODEL_RATE_HZ / 2.0


@dataclass(frozen=True)
class GeometricPath:
    """Arc-length parameterization of a 2-D eye-position path."""

    progress: np.ndarray
    xy: np.ndarray
    path_length_deg: float
    original_n_samples: int
    original_progress: np.ndarray

    @property
    def start(self) -> np.ndarray:
        return self.xy[0]

    @property
    def end(self) -> np.ndarray:
        return self.xy[-1]

    @property
    def is_stationary(self) -> bool:
        return bool(self.path_length_deg <= EPS or self.xy.shape[0] <= 1)


def as_trace_xy(trace: np.ndarray, *, name: str = "trace") -> np.ndarray:
    """Validate and return a finite float64 trace with shape ``(T, 2)``."""
    arr = np.asarray(trace, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (T, 2), got {arr.shape}.")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def cumulative_arc_length(trace: np.ndarray) -> np.ndarray:
    """Return cumulative Euclidean arc length for a trace."""
    arr = as_trace_xy(trace)
    if arr.shape[0] == 1:
        return np.zeros((1,), dtype=np.float64)
    steps = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)]).astype(np.float64)


def geometric_path_from_trace(trace: np.ndarray) -> GeometricPath:
    """Build a monotonic normalized arc-length path, dropping duplicate samples."""
    arr = as_trace_xy(trace)
    arc = cumulative_arc_length(arr)
    total = float(arc[-1])
    if total <= EPS:
        return GeometricPath(
            progress=np.asarray([0.0, 1.0], dtype=np.float64),
            xy=np.stack([arr[0], arr[-1]], axis=0).astype(np.float64),
            path_length_deg=0.0,
            original_n_samples=int(arr.shape[0]),
            original_progress=np.zeros((arr.shape[0],), dtype=np.float64),
        )

    keep = np.concatenate([[True], np.diff(arc) > EPS])
    progress = arc[keep] / total
    xy = arr[keep]
    if progress[-1] < 1.0:
        progress = np.concatenate([progress, [1.0]])
        xy = np.concatenate([xy, arr[-1:]], axis=0)
    else:
        progress[-1] = 1.0
        xy[-1] = arr[-1]
    progress[0] = 0.0
    return GeometricPath(
        progress=progress.astype(np.float64),
        xy=xy.astype(np.float64),
        path_length_deg=total,
        original_n_samples=int(arr.shape[0]),
        original_progress=(arc / total).astype(np.float64),
    )


def interpolate_path(path: GeometricPath, progress: np.ndarray) -> np.ndarray:
    """Sample ``path`` at normalized arc-length positions in ``[0, 1]``."""
    p = np.asarray(progress, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"progress must be one-dimensional, got {p.shape}.")
    p = np.clip(p, 0.0, 1.0)
    if path.is_stationary:
        return np.repeat(path.start[None, :], p.shape[0], axis=0).astype(np.float64)
    x = np.interp(p, path.progress, path.xy[:, 0])
    y = np.interp(p, path.progress, path.xy[:, 1])
    return np.stack([x, y], axis=1).astype(np.float64)


def smoothstep(progress: np.ndarray) -> np.ndarray:
    """Cubic ease-in/ease-out monotonic progress curve."""
    p = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    return p * p * (3.0 - 2.0 * p)


def progress_samples(
    path: GeometricPath,
    traversal_frames: int,
    *,
    profile: str,
) -> np.ndarray:
    """Return path-progress samples for one traversal, including endpoints."""
    n = int(traversal_frames)
    if n < 1:
        raise ValueError("traversal_frames must be positive.")
    if n == 1:
        return np.asarray([1.0], dtype=np.float64)
    base = np.linspace(0.0, 1.0, n, dtype=np.float64)
    profile_key = str(profile).strip().lower()
    if profile_key in {"uniform", "uniform_arc", "uniform_arc_length"}:
        return base
    if profile_key in {"eased", "smoothstep", "ease"}:
        return smoothstep(base)
    if profile_key in {"natural", "natural_speed", "natural_speed_profile"}:
        if path.original_n_samples <= 1:
            return base
        source_t = np.linspace(0.0, 1.0, int(path.original_n_samples), dtype=np.float64)
        return np.interp(base, source_t, np.asarray(path.original_progress, dtype=np.float64))
    raise ValueError(f"Unknown retiming profile {profile!r}.")


def timing_window(total_frames: int, traversal_frames: int, placement: str) -> tuple[int, int]:
    """Return inclusive start/stop sample indices for a traversal."""
    total = int(total_frames)
    n = int(traversal_frames)
    if total < 1:
        raise ValueError("total_frames must be positive.")
    if n < 1 or n > total:
        raise ValueError(f"traversal_frames must be in [1, {total}], got {n}.")
    key = str(placement).strip().lower()
    if key in {"terminal", "terminal_traversal", "final"}:
        start = total - n
    elif key in {"endpoint_hold", "early", "start"}:
        start = 0
    elif key in {"centered", "center"}:
        start = (total - n) // 2
    else:
        raise ValueError(f"Unknown timing placement {placement!r}.")
    stop = start + n - 1
    return int(start), int(stop)


def retime_trace(
    trace: np.ndarray,
    *,
    traversal_frames: int,
    total_frames: int | None = None,
    placement: str = "terminal",
    profile: str = "uniform",
) -> np.ndarray:
    """Traverse the same continuous path with altered timing.

    The discrete convention is endpoint-inclusive: ``traversal_frames`` samples
    are assigned to the motion path, including ``gamma(0)`` and ``gamma(1)``.
    The physical interval between those endpoint samples is therefore
    ``(traversal_frames - 1) / frame_rate``.
    """
    source = as_trace_xy(trace)
    total = int(total_frames) if total_frames is not None else int(source.shape[0])
    path = geometric_path_from_trace(source)
    start, stop = timing_window(total, int(traversal_frames), placement)
    out = np.empty((total, 2), dtype=np.float64)
    out[:] = path.end
    if start > 0:
        out[:start] = path.start
    if stop + 1 < total:
        out[stop + 1 :] = path.end
    p = progress_samples(path, int(traversal_frames), profile=str(profile))
    out[start : stop + 1] = interpolate_path(path, p)
    return out.astype(np.float32)


def scale_trace_about_center(
    trace: np.ndarray,
    beta: float,
    *,
    center: str | Iterable[float] = "centroid",
) -> np.ndarray:
    """Scale a trace around a fixed center without changing that center."""
    arr = as_trace_xy(trace)
    if isinstance(center, str):
        key = center.strip().lower()
        if key in {"centroid", "mean"}:
            c = np.mean(arr, axis=0)
        elif key == "start":
            c = arr[0]
        elif key == "end":
            c = arr[-1]
        else:
            raise ValueError(f"Unknown scaling center {center!r}.")
    else:
        c = np.asarray(list(center), dtype=np.float64)
        if c.shape != (2,):
            raise ValueError(f"center must contain two values, got {c.shape}.")
    return (c[None, :] + float(beta) * (arr - c[None, :])).astype(np.float32)


def contour_basis(contour_orientation_deg: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Return along-contour and across-contour unit vectors."""
    theta = math.radians(float(contour_orientation_deg) if contour_orientation_deg is not None else 0.0)
    along = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
    across = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    return along, across


def trace_path_length(trace: np.ndarray) -> float:
    arr = as_trace_xy(trace)
    if arr.shape[0] <= 1:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def trajectory_metrics(
    trace: np.ndarray,
    *,
    source_trace: np.ndarray | None = None,
    traversal_frames: int | None = None,
    total_frames: int | None = None,
    frame_rate_hz: float = MODEL_RATE_HZ,
    contour_orientation_deg: float | None = None,
    preferred_sf_cpd: float | None = None,
    retiming_profile: str = "",
    timing_placement: str = "",
    condition_name: str = "",
    nyquist_hz: float = MODEL_NYQUIST_HZ,
) -> dict[str, Any]:
    """Compute continuous-path and sampled-history metrics for one trace."""
    arr = as_trace_xy(trace)
    total = int(total_frames) if total_frames is not None else int(arr.shape[0])
    if total != arr.shape[0]:
        raise ValueError(f"total_frames={total} does not match trace length {arr.shape[0]}.")
    src = as_trace_xy(source_trace, name="source_trace") if source_trace is not None else arr
    path = geometric_path_from_trace(src)
    dt = 1.0 / float(frame_rate_hz)
    vel = np.diff(arr, axis=0) / dt if arr.shape[0] > 1 else np.zeros((0, 2), dtype=np.float64)
    accel = np.diff(vel, axis=0) / dt if vel.shape[0] > 1 else np.zeros((0, 2), dtype=np.float64)
    along_u, across_u = contour_basis(contour_orientation_deg)
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    along_pos = centered @ along_u
    across_pos = centered @ across_u
    along_vel = vel @ along_u if vel.size else np.zeros((0,), dtype=np.float64)
    across_vel = vel @ across_u if vel.size else np.zeros((0,), dtype=np.float64)
    sampled_path = trace_path_length(arr)
    source_path = float(path.path_length_deg)
    tf = float("nan")
    margin = float("nan")
    exceeds = False
    sf = float(preferred_sf_cpd) if preferred_sf_cpd is not None else float("nan")
    if math.isfinite(sf):
        tf = sf * rms(across_vel)
        margin = float(nyquist_hz) - tf
        exceeds = bool(tf >= float(nyquist_hz))
    start_frame = -1
    stop_frame = -1
    hold_before = 0
    hold_after = 0
    if traversal_frames is not None and timing_placement:
        try:
            start_frame, stop_frame = timing_window(total, int(traversal_frames), str(timing_placement))
            hold_before = start_frame
            hold_after = total - stop_frame - 1
        except ValueError:
            start_frame = -1
            stop_frame = -1
    n_traversal = int(traversal_frames) if traversal_frames is not None else int(total)
    return {
        "condition_name": str(condition_name),
        "retiming_profile": str(retiming_profile),
        "timing_placement": str(timing_placement),
        "traversal_frames": int(n_traversal),
        "total_frames": int(total),
        "frame_rate_hz": float(frame_rate_hz),
        "traversal_duration_ms": float(1000.0 * n_traversal / float(frame_rate_hz)),
        "sample_endpoint_interval_ms": float(1000.0 * max(n_traversal - 1, 0) / float(frame_rate_hz)),
        "motion_onset_frame": int(start_frame),
        "motion_offset_frame": int(stop_frame),
        "hold_before_frames": int(hold_before),
        "hold_after_frames": int(hold_after),
        "path_length_deg": source_path,
        "path_length_arcmin": source_path * 60.0,
        "model_sampled_path_length_deg": sampled_path,
        "model_sampled_path_length_arcmin": sampled_path * 60.0,
        "mean_path_speed_deg_s": source_path / max(n_traversal / float(frame_rate_hz), EPS),
        "mean_path_speed_arcmin_s": source_path * 60.0 / max(n_traversal / float(frame_rate_hz), EPS),
        "sample_endpoint_mean_path_speed_deg_s": source_path / max((n_traversal - 1) / float(frame_rate_hz), EPS),
        "rms_speed_deg_s": rms(np.linalg.norm(vel, axis=1)) if vel.size else 0.0,
        "rms_across_velocity_deg_s": rms(across_vel),
        "rms_along_velocity_deg_s": rms(along_vel),
        "peak_speed_deg_s": float(np.max(np.linalg.norm(vel, axis=1))) if vel.size else 0.0,
        "peak_across_velocity_deg_s": float(np.max(np.abs(across_vel))) if across_vel.size else 0.0,
        "rms_acceleration_deg_s2": rms(np.linalg.norm(accel, axis=1)) if accel.size else 0.0,
        "rms_position_total": rms(np.linalg.norm(centered, axis=1)),
        "rms_position_across": rms(across_pos),
        "rms_position_along": rms(along_pos),
        "characteristic_motion_tf_hz": tf,
        "nyquist_margin_hz": margin,
        "exceeds_model_nyquist": exceeds,
        "preferred_sf_cpd": sf,
        "contour_orientation_deg": float(contour_orientation_deg) if contour_orientation_deg is not None else float("nan"),
        "start_x": float(path.start[0]),
        "start_y": float(path.start[1]),
        "end_x": float(path.end[0]),
        "end_y": float(path.end[1]),
        "sampled_start_x": float(arr[0, 0]),
        "sampled_start_y": float(arr[0, 1]),
        "sampled_end_x": float(arr[-1, 0]),
        "sampled_end_y": float(arr[-1, 1]),
        "min_x": float(np.min(path.xy[:, 0])),
        "max_x": float(np.max(path.xy[:, 0])),
        "min_y": float(np.min(path.xy[:, 1])),
        "max_y": float(np.max(path.xy[:, 1])),
        "sampled_min_x": float(np.min(arr[:, 0])),
        "sampled_max_x": float(np.max(arr[:, 0])),
        "sampled_min_y": float(np.min(arr[:, 1])),
        "sampled_max_y": float(np.max(arr[:, 1])),
        "n_distinct_sampled_positions": int(np.unique(np.round(arr, decimals=10), axis=0).shape[0]),
    }


def invariance_report(metrics: list[dict[str, Any]], *, atol: float = 1e-6) -> dict[str, Any]:
    """Check continuous-geometry invariants across a retimed family."""
    if not metrics:
        return {"ok": True, "n": 0, "failures": []}
    invariant_keys = [
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "path_length_deg",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
    ]
    ref = metrics[0]
    failures: list[dict[str, Any]] = []
    for idx, row in enumerate(metrics[1:], start=1):
        for key in invariant_keys:
            a = float(ref.get(key, np.nan))
            b = float(row.get(key, np.nan))
            if not (math.isfinite(a) and math.isfinite(b) and abs(a - b) <= float(atol)):
                failures.append({"row_index": int(idx), "key": key, "reference": a, "value": b, "abs_delta": abs(a - b)})
    return {"ok": not failures, "n": int(len(metrics)), "failures": failures}
