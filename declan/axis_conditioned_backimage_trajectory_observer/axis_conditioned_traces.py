"""Trace construction for axis-conditioned BackImage observer catalogs."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

SUPPORTED_RELATIONS = ("parallel", "orthogonal")
SUPPORTED_TEMPLATE_MODES = (
    "same_parallel_projection",
    "same_orthogonal_projection",
    "same_dominant_projection",
    "arclength_signed",
)


def _as_trace(trace: np.ndarray, name: str = "trace") -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (time, 2), got {arr.shape}")
    if arr.shape[0] <= 0:
        raise ValueError(f"{name} must contain at least one time point")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def axis_unit(axis_deg: float) -> np.ndarray:
    """Return the gaze-coordinate unit vector for an undirected axis angle."""
    if not np.isfinite(float(axis_deg)):
        raise ValueError("axis_deg must be finite")
    theta = np.radians(float(axis_deg))
    return np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)


def axis_perp(axis_deg: float) -> np.ndarray:
    """Return the unit vector perpendicular to an undirected axis angle."""
    if not np.isfinite(float(axis_deg)):
        raise ValueError("axis_deg must be finite")
    theta = np.radians(float(axis_deg))
    return np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.float64)


def _center_trace(trace: np.ndarray) -> np.ndarray:
    arr = _as_trace(trace)
    return arr - np.mean(arr, axis=0, keepdims=True)


def _trace_rms(trace: np.ndarray) -> float:
    arr = _as_trace(trace)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.sum(arr * arr, axis=1))))


def _path_length(trace: np.ndarray) -> float:
    arr = _as_trace(trace)
    if arr.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def _lag1_autocorr(trace: np.ndarray) -> float:
    arr = _as_trace(trace)
    if arr.shape[0] < 3:
        return 0.0
    vals = []
    for dim in range(2):
        a = arr[:-1, dim] - np.mean(arr[:-1, dim])
        b = arr[1:, dim] - np.mean(arr[1:, dim])
        den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        if den > 1e-12:
            vals.append(float(np.sum(a * b) / den))
    if not vals:
        return 0.0
    return float(np.clip(np.mean(vals), -0.95, 0.98))


def _trace_hash(trace: np.ndarray) -> str:
    arr = np.asarray(trace, dtype=np.float32)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def _validate_nonnegative(value: float, name: str) -> float:
    val = float(value)
    if not np.isfinite(val):
        raise ValueError(f"{name} must be finite")
    if val < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return val


def trace_metrics(trace: np.ndarray, *, dt: float = 1.0 / 120.0) -> dict[str, float | int]:
    """Compute compact displacement and speed metrics for a trace."""
    arr = _as_trace(trace)
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive")
    if arr.shape[0] < 2:
        speed = np.zeros(0, dtype=np.float64)
    else:
        speed = np.linalg.norm(np.diff(arr, axis=0), axis=1) / float(dt)
    return {
        "n_timebins": int(arr.shape[0]),
        "rms_displacement_deg": _trace_rms(arr),
        "path_length_deg": _path_length(arr),
        "max_radius_deg": float(np.max(np.linalg.norm(arr, axis=1))) if arr.size else 0.0,
        "duration_s": float(max(0, arr.shape[0] - 1) * float(dt)),
        "speed_mean_deg_s": float(np.mean(speed)) if speed.size else 0.0,
        "speed_median_deg_s": float(np.median(speed)) if speed.size else 0.0,
        "speed_p95_deg_s": float(np.percentile(speed, 95.0)) if speed.size else 0.0,
        "generated_lag1_autocorr": _lag1_autocorr(arr),
    }


def _dominant_projection(centered: np.ndarray, u_parallel: np.ndarray, u_orthogonal: np.ndarray) -> tuple[np.ndarray, str]:
    par = centered @ u_parallel
    orth = centered @ u_orthogonal
    par_rms = float(np.sqrt(np.mean(par * par))) if par.size else 0.0
    orth_rms = float(np.sqrt(np.mean(orth * orth))) if orth.size else 0.0
    if orth_rms > par_rms:
        return orth, "orthogonal_projection"
    return par, "parallel_projection"


def _arclength_signed_template(centered: np.ndarray, u_parallel: np.ndarray, u_orthogonal: np.ndarray) -> tuple[np.ndarray, str]:
    if centered.shape[0] <= 1:
        return np.zeros(centered.shape[0], dtype=np.float64), "arclength_signed"
    par, label = _dominant_projection(centered, u_parallel, u_orthogonal)
    steps = np.diff(centered, axis=0)
    direction = u_orthogonal if label == "orthogonal_projection" else u_parallel
    signs = np.sign(steps @ direction)
    last = 1.0
    for i, val in enumerate(signs):
        if val == 0.0:
            signs[i] = last
        else:
            last = float(val)
    signed_steps = np.linalg.norm(steps, axis=1) * signs
    scalar = np.concatenate([[0.0], np.cumsum(signed_steps)])
    scalar -= np.mean(scalar)
    if np.sqrt(np.mean(scalar * scalar)) <= 1e-12 and np.sqrt(np.mean(par * par)) > 1e-12:
        scalar = par - np.mean(par)
    return scalar, "arclength_signed"


def _scalar_template(centered: np.ndarray, axis_deg: float, template_mode: str) -> tuple[np.ndarray, str]:
    mode = str(template_mode)
    if mode not in SUPPORTED_TEMPLATE_MODES:
        raise ValueError(f"Unsupported template_mode={template_mode!r}; expected {SUPPORTED_TEMPLATE_MODES}")
    u_parallel = axis_unit(float(axis_deg))
    u_orthogonal = axis_perp(float(axis_deg))
    if mode == "same_parallel_projection":
        scalar = centered @ u_parallel
        label = "parallel_projection"
    elif mode == "same_orthogonal_projection":
        scalar = centered @ u_orthogonal
        label = "orthogonal_projection"
    elif mode == "same_dominant_projection":
        scalar, label = _dominant_projection(centered, u_parallel, u_orthogonal)
    else:
        scalar, label = _arclength_signed_template(centered, u_parallel, u_orthogonal)
    scalar = np.asarray(scalar, dtype=np.float64)
    scalar -= np.mean(scalar)
    return scalar, label


def _scale_to_target_rms(
    trace: np.ndarray,
    *,
    target_rms_deg: float,
    max_rms_deg: float | None,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    arr = _as_trace(trace)
    requested = _validate_nonnegative(float(target_rms_deg), "target_rms_deg")
    if max_rms_deg is None:
        effective = requested
        clipped = False
    else:
        cap = _validate_nonnegative(float(max_rms_deg), "max_rms_deg")
        effective = min(requested, cap)
        clipped = requested > effective + 1e-12
    base = _trace_rms(arr)
    degenerate_requested_motion = base <= 1e-12 and effective > 1e-12
    if base <= 1e-12 or effective <= 0.0:
        scaled = np.zeros_like(arr)
        scale_factor = 0.0
    else:
        scale_factor = effective / base
        scaled = arr * scale_factor
        scaled -= np.mean(scaled, axis=0, keepdims=True)
    clip_fraction = 0.0 if requested <= 1e-12 else max(0.0, (requested - effective) / requested)
    return scaled.astype(np.float32), {
        "base_rms_deg": float(base),
        "requested_rms_deg": float(requested),
        "effective_rms_deg": _trace_rms(scaled),
        "scale_factor": float(scale_factor),
        "rms_clipped_high": bool(clipped),
        "clipping_fraction": float(clip_fraction),
        "degenerate_requested_motion": bool(degenerate_requested_motion),
    }


def _relation_axis(axis_deg: float, relation: str) -> tuple[np.ndarray, float]:
    rel = str(relation)
    if rel == "parallel":
        return axis_unit(float(axis_deg)), float(axis_deg)
    if rel == "orthogonal":
        return axis_perp(float(axis_deg)), float(axis_deg) + 90.0
    raise ValueError(f"Unsupported relation={relation!r}; expected {SUPPORTED_RELATIONS}")


def axis_conditioned_trace(
    source_trace: np.ndarray,
    *,
    axis_deg: float,
    relation: str,
    template_mode: str = "same_dominant_projection",
    scale: float = 1.0,
    target_rms_deg: float | None = None,
    max_rms_deg: float | None = None,
    dt: float = 1.0 / 120.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project one source trace onto a local image axis relation.

    `axis_deg` is the local edge axis. `relation` chooses whether the output
    trace lies along that axis or the orthogonal axis. `template_mode` chooses
    which scalar time course is shared by the axis-conditioned traces.
    """
    source = _center_trace(source_trace)
    out_axis, output_axis_deg = _relation_axis(float(axis_deg), str(relation))
    scalar, template_source = _scalar_template(source, float(axis_deg), str(template_mode))
    raw = scalar[:, None] * out_axis[None, :]
    scale_value = _validate_nonnegative(float(scale), "scale")
    requested_rms = (
        _validate_nonnegative(float(target_rms_deg), "target_rms_deg")
        if target_rms_deg is not None
        else scale_value * _trace_rms(source)
    )
    scaled, scale_meta = _scale_to_target_rms(raw, target_rms_deg=requested_rms, max_rms_deg=max_rms_deg)
    source_metrics = trace_metrics(source, dt=float(dt))
    rendered_metrics = trace_metrics(scaled, dt=float(dt))
    meta: dict[str, Any] = {
        "axis_conditioned": True,
        "axis_deg": float(axis_deg),
        "axis_relation": str(relation),
        "output_axis_deg": float(output_axis_deg),
        "axis_template_mode": str(template_mode),
        "template_source": template_source,
        "source_rms_displacement_deg": float(source_metrics["rms_displacement_deg"]),
        "source_path_length_deg": float(source_metrics["path_length_deg"]),
        "source_max_radius_deg": float(source_metrics["max_radius_deg"]),
        "source_duration_s": float(source_metrics["duration_s"]),
        "rendered_rms_displacement_deg": float(rendered_metrics["rms_displacement_deg"]),
        "rendered_path_length_deg": float(rendered_metrics["path_length_deg"]),
        "rendered_max_radius_deg": float(rendered_metrics["max_radius_deg"]),
        "rendered_duration_s": float(rendered_metrics["duration_s"]),
        # Compatibility aliases consumed by the existing trajectory observer's
        # _trajectory_spec metadata adapter.
        "path_length_deg": float(rendered_metrics["path_length_deg"]),
        "generated_lag1_autocorr": float(rendered_metrics["generated_lag1_autocorr"]),
        "speed_mean_deg_s": float(rendered_metrics["speed_mean_deg_s"]),
        "speed_median_deg_s": float(rendered_metrics["speed_median_deg_s"]),
        "speed_p95_deg_s": float(rendered_metrics["speed_p95_deg_s"]),
        "axis_match_status": "unchecked_single_trace",
        **scale_meta,
    }
    return scaled, meta


def matched_axis_trace_pair(
    source_trace: np.ndarray,
    *,
    edge_axis_deg: float,
    template_mode: str = "same_dominant_projection",
    scale: float = 1.0,
    target_rms_deg: float | None = None,
    max_rms_deg: float | None = None,
    dt: float = 1.0 / 120.0,
    tolerance: float = 1e-6,
    source_id: str | int | None = None,
) -> dict[str, dict[str, Any]]:
    """Build matched edge-parallel and edge-orthogonal traces."""
    if float(tolerance) < 0.0 or not np.isfinite(float(tolerance)):
        raise ValueError("tolerance must be finite and non-negative")
    source = _as_trace(source_trace, "source_trace")
    source_token = str(source_id) if source_id is not None else f"trace-{_trace_hash(source)}"
    parallel_trace, parallel_meta = axis_conditioned_trace(
        source,
        axis_deg=float(edge_axis_deg),
        relation="parallel",
        template_mode=str(template_mode),
        scale=float(scale),
        target_rms_deg=target_rms_deg,
        max_rms_deg=max_rms_deg,
        dt=float(dt),
    )
    orthogonal_trace, orthogonal_meta = axis_conditioned_trace(
        source,
        axis_deg=float(edge_axis_deg),
        relation="orthogonal",
        template_mode=str(template_mode),
        scale=float(scale),
        target_rms_deg=target_rms_deg,
        max_rms_deg=max_rms_deg,
        dt=float(dt),
    )
    rms_delta = abs(float(parallel_meta["rendered_rms_displacement_deg"]) - float(orthogonal_meta["rendered_rms_displacement_deg"]))
    path_delta = abs(float(parallel_meta["rendered_path_length_deg"]) - float(orthogonal_meta["rendered_path_length_deg"]))
    duration_delta = abs(float(parallel_meta["rendered_duration_s"]) - float(orthogonal_meta["rendered_duration_s"]))
    clip_delta = abs(float(parallel_meta["clipping_fraction"]) - float(orthogonal_meta["clipping_fraction"]))
    degenerate = bool(parallel_meta["degenerate_requested_motion"]) or bool(orthogonal_meta["degenerate_requested_motion"])
    matched = (not degenerate) and (
        rms_delta <= float(tolerance)
        and path_delta <= float(tolerance)
        and duration_delta <= float(tolerance)
        and clip_delta <= float(tolerance)
    )
    if degenerate:
        match_status = "invalid_degenerate"
    elif matched:
        match_status = "matched"
    else:
        match_status = "mismatch"
    requested = parallel_meta["requested_rms_deg"]
    effective = parallel_meta["effective_rms_deg"]
    cap = "none" if max_rms_deg is None else f"{float(max_rms_deg):.9g}"
    pair_meta = {
        "axis_pair_id": (
            f"src-{source_token}:edge-{float(edge_axis_deg):.6f}:template-{template_mode}:"
            f"scale-{float(scale):.9g}:requested-{float(requested):.9g}:"
            f"effective-{float(effective):.9g}:cap-{cap}"
        ),
        "axis_source_id": source_token,
        "axis_match_status": match_status,
        "axis_match_rms_delta_deg": float(rms_delta),
        "axis_match_path_delta_deg": float(path_delta),
        "axis_match_duration_delta_s": float(duration_delta),
        "axis_match_clipping_fraction_delta": float(clip_delta),
        "axis_match_tolerance": float(tolerance),
        "axis_match_degenerate": bool(degenerate),
    }
    parallel_meta.update(pair_meta)
    orthogonal_meta.update(pair_meta)
    return {
        "parallel": {"trace": parallel_trace, "meta": parallel_meta},
        "orthogonal": {"trace": orthogonal_trace, "meta": orthogonal_meta},
    }
