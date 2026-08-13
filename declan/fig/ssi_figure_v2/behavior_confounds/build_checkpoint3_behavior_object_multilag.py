#!/usr/bin/env python3
"""Checkpoint 3: determine which contour-relative behavior object is aligned.

This is an example-level, map-first diagnostic for the six windows selected at
Checkpoint 1.  It compares raw and lightly smoothed position spread, range,
unsigned path, net displacement, velocity covariance, and lagged displacement.
It also recomputes event masks under predeclared aggressive, primary, and
permissive detector settings without changing the selected examples.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jake.twininfo.eye_controls import detect_microsaccade_events

from declan.fig.ssi_figure_v2.behavior_confounds import (
    build_checkpoint1_reference_frame_examples as cp1,
)
from declan.fig.ssi_figure_v2.behavior_confounds import (
    build_checkpoint2b_local_offset_examples as cp2b,
)
from declan.fixation_statistics_by_stimulus.extraction import (
    _as_numpy,
    _load_dict_dataset,
    _speed_threshold_mad_valid_pairs,
)
from declan.fixation_statistics_by_stimulus.plot_backimage_contour_motion_components import (
    _window_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = cp1.DEFAULT_OUT_DIR / "checkpoint3_behavior_object_v1"
SOURCE_WINDOWS = cp1.DEFAULT_INPUT
SELECTED_WINDOWS = cp1.DEFAULT_OUT_DIR / "checkpoint1_selected_windows.csv"
DT_S = cp1.DT_S
LAGS = cp1.LAGS
SMOOTH_KERNEL = np.asarray([0.25, 0.50, 0.25], dtype=np.float64)

DETECTOR_CONFIGS = (
    {
        "detector_setting": "aggressive",
        "speed_z": 4.0,
        "event_pad_samples": 3,
        "interpretation": "lower threshold and wider padding",
    },
    {
        "detector_setting": "primary",
        "speed_z": 6.0,
        "event_pad_samples": 1,
        "interpretation": "reviewed extraction setting",
    },
    {
        "detector_setting": "permissive",
        "speed_z": 8.0,
        "event_pad_samples": 0,
        "interpretation": "higher threshold and no padding",
    },
)

PARALLEL_COLOR = "#1b7f5c"
NORMAL_COLOR = "#7a3b9a"
RAW_COLOR = "#30343b"
SMOOTH_COLOR = "#c05b35"
POSITION_COLOR = "#245c8a"
VELOCITY_COLOR = "#d08022"
DETECTOR_COLORS = {
    "aggressive": "#b4492d",
    "primary": "#245c8a",
    "permissive": "#7c8792",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _smooth_trace(trace: np.ndarray) -> np.ndarray:
    x = np.asarray(trace, dtype=np.float64)
    padded = np.pad(x, ((1, 1), (0, 0)), mode="edge")
    return (
        SMOOTH_KERNEL[0] * padded[:-2]
        + SMOOTH_KERNEL[1] * padded[1:-1]
        + SMOOTH_KERNEL[2] * padded[2:]
    )


def _axis_stats(vectors: np.ndarray, *, covariance: bool) -> dict[str, float]:
    x = np.asarray(vectors, dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < 3:
        return {"axis_deg": float("nan"), "anisotropy": float("nan")}
    if covariance:
        matrix = np.cov(x.T, ddof=0)
    else:
        matrix = (x.T @ x) / len(x)
    values, directions = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    direction = directions[:, order[0]]
    axis_deg = cp1.axial_signed_deg(
        np.degrees(np.arctan2(direction[1], direction[0]))
    )
    denom = float(np.sum(values))
    anisotropy = float((values[0] - values[1]) / denom) if denom > 0 else float("nan")
    return {"axis_deg": float(axis_deg), "anisotropy": anisotropy}


def _step_reversal_fraction(values: np.ndarray, step_indices: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    step_indices = np.asarray(step_indices, dtype=int)
    if len(values) < 2:
        return float("nan")
    adjacent = step_indices[1:] == step_indices[:-1] + 1
    nonzero = (np.sign(values[1:]) != 0) & (np.sign(values[:-1]) != 0)
    keep = adjacent & nonzero
    return (
        float(np.mean(np.sign(values[1:][keep]) != np.sign(values[:-1][keep])))
        if np.any(keep)
        else float("nan")
    )


def _step_autocorr(values: np.ndarray, step_indices: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    step_indices = np.asarray(step_indices, dtype=int)
    if len(values) < 3:
        return float("nan")
    adjacent = step_indices[1:] == step_indices[:-1] + 1
    a, b = values[:-1][adjacent], values[1:][adjacent]
    if len(a) < 2 or np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _component_metrics(
    trace: np.ndarray,
    edge_axis_deg: float,
    *,
    sample_mask: np.ndarray | None = None,
) -> dict[str, float]:
    x = np.asarray(trace, dtype=np.float64)
    if sample_mask is None:
        sample_mask = np.ones(len(x), dtype=bool)
    sample_mask = np.asarray(sample_mask, dtype=bool) & np.isfinite(x).all(axis=1)
    retained = x[sample_mask]
    if len(retained) < 3:
        raise RuntimeError("Fewer than three retained samples")

    parallel_axis = cp1.axis_vector(edge_axis_deg)
    normal_axis = np.asarray([-parallel_axis[1], parallel_axis[0]])
    centered = retained - np.mean(retained, axis=0, keepdims=True)
    parallel_position = centered @ parallel_axis
    normal_position = centered @ normal_axis

    all_steps = np.diff(x, axis=0)
    pair_mask = sample_mask[:-1] & sample_mask[1:] & np.isfinite(all_steps).all(axis=1)
    step_indices = np.flatnonzero(pair_mask)
    steps = all_steps[pair_mask]
    parallel_steps = steps @ parallel_axis
    normal_steps = steps @ normal_axis

    position_axis = _axis_stats(centered, covariance=True)
    velocity_axis = _axis_stats(steps / DT_S, covariance=True)
    start_end = retained[-1] - retained[0]
    start_end_parallel = float(start_end @ parallel_axis * 60.0)
    start_end_normal = float(start_end @ normal_axis * 60.0)

    out: dict[str, float] = {
        "n_retained_samples": int(np.count_nonzero(sample_mask)),
        "n_retained_steps": int(len(steps)),
        "position_parallel_rms_arcmin": float(
            np.sqrt(np.mean(parallel_position**2)) * 60.0
        ),
        "position_normal_rms_arcmin": float(
            np.sqrt(np.mean(normal_position**2)) * 60.0
        ),
        "position_parallel_range_arcmin": float(np.ptp(parallel_position) * 60.0),
        "position_normal_range_arcmin": float(np.ptp(normal_position) * 60.0),
        "unsigned_parallel_path_arcmin": float(np.sum(np.abs(parallel_steps)) * 60.0),
        "unsigned_normal_path_arcmin": float(np.sum(np.abs(normal_steps)) * 60.0),
        "start_end_parallel_arcmin": start_end_parallel,
        "start_end_normal_arcmin": start_end_normal,
        "start_end_magnitude_arcmin": float(np.linalg.norm(start_end) * 60.0),
        "parallel_step_reversal_fraction": _step_reversal_fraction(
            parallel_steps, step_indices
        ),
        "normal_step_reversal_fraction": _step_reversal_fraction(
            normal_steps, step_indices
        ),
        "parallel_step_autocorr_lag1": _step_autocorr(
            parallel_steps, step_indices
        ),
        "normal_step_autocorr_lag1": _step_autocorr(normal_steps, step_indices),
        "position_covariance_axis_deg": position_axis["axis_deg"],
        "position_covariance_anisotropy": position_axis["anisotropy"],
        "velocity_covariance_axis_deg": velocity_axis["axis_deg"],
        "velocity_covariance_anisotropy": velocity_axis["anisotropy"],
    }
    for prefix in ("position",):
        for metric in ("rms", "range"):
            p = out[f"{prefix}_parallel_{metric}_arcmin"]
            n = out[f"{prefix}_normal_{metric}_arcmin"]
            out[f"{prefix}_{metric}_parallel_minus_normal_arcmin"] = p - n
    out["unsigned_path_parallel_minus_normal_arcmin"] = (
        out["unsigned_parallel_path_arcmin"] - out["unsigned_normal_path_arcmin"]
    )
    out["absolute_start_end_parallel_minus_normal_arcmin"] = (
        abs(start_end_parallel) - abs(start_end_normal)
    )
    out["position_axis_edge_delta_deg"] = float(
        cp1.axial_distance_deg(position_axis["axis_deg"], edge_axis_deg)
    )
    out["velocity_axis_edge_delta_deg"] = float(
        cp1.axial_distance_deg(velocity_axis["axis_deg"], edge_axis_deg)
    )
    return out


def _multilag_metrics(
    trace: np.ndarray,
    edge_axis_deg: float,
    *,
    example_role: str,
    processing: str,
) -> list[dict[str, Any]]:
    x = np.asarray(trace, dtype=np.float64)
    parallel_axis = cp1.axis_vector(edge_axis_deg)
    normal_axis = np.asarray([-parallel_axis[1], parallel_axis[0]])
    rows: list[dict[str, Any]] = []
    for lag in LAGS:
        displacement = x[lag:] - x[:-lag]
        parallel = displacement @ parallel_axis
        normal = displacement @ normal_axis
        angles = np.degrees(np.arctan2(displacement[:, 1], displacement[:, 0]))
        z = np.mean(np.exp(2j * np.radians(angles)))
        preferred = cp1.axial_signed_deg(0.5 * np.degrees(np.angle(z)))
        rows.append(
            {
                "example_role": example_role,
                "processing": processing,
                "lag_samples": int(lag),
                "lag_ms": float(lag * DT_S * 1000.0),
                "n_displacements": int(len(displacement)),
                "parallel_displacement_rms_arcmin": float(
                    np.sqrt(np.mean(parallel**2)) * 60.0
                ),
                "normal_displacement_rms_arcmin": float(
                    np.sqrt(np.mean(normal**2)) * 60.0
                ),
                "parallel_displacement_mean_abs_arcmin": float(
                    np.mean(np.abs(parallel)) * 60.0
                ),
                "normal_displacement_mean_abs_arcmin": float(
                    np.mean(np.abs(normal)) * 60.0
                ),
                "mean_displacement_magnitude_arcmin": float(
                    np.mean(np.linalg.norm(displacement, axis=1)) * 60.0
                ),
                "rms_displacement_magnitude_arcmin": float(
                    np.sqrt(np.mean(np.sum(displacement**2, axis=1))) * 60.0
                ),
                "displacement_preferred_axis_deg": float(preferred),
                "displacement_axis_edge_delta_deg": float(
                    cp1.axial_distance_deg(preferred, edge_axis_deg)
                ),
                "displacement_axial_resultant_r": float(np.abs(z)),
            }
        )
    return rows


@lru_cache(maxsize=16)
def _trial_trace_and_valid(
    session_name: str, trial_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    from DataYatesV1 import get_session

    subject, date = session_name.split("_", 1)
    session = get_session(subject, date)
    dset = _load_dict_dataset(Path(session.sess_dir) / "datasets" / "backimage.dset")
    eyepos = _as_numpy(dset["eyepos"]).astype(np.float64)
    trial_inds = _as_numpy(dset.covariates["trial_inds"]).reshape(-1).astype(int)
    if "dpi_valid" in dset.covariates:
        valid = _as_numpy(dset.covariates["dpi_valid"]).reshape(-1).astype(bool)
    elif "dfs" in dset.covariates:
        dfs = _as_numpy(dset.covariates["dfs"])
        valid = np.asarray(dfs).reshape(dfs.shape[0], -1).any(axis=1)
    else:
        valid = np.ones(len(eyepos), dtype=bool)
    valid &= np.isfinite(eyepos).all(axis=1)
    valid &= (np.abs(eyepos[:, 0]) <= 12.0) & (np.abs(eyepos[:, 1]) <= 12.0)
    idx = np.flatnonzero(trial_inds == int(trial_idx))
    return eyepos[idx], valid[idx]


def _longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def _detector_rows(target: pd.Series) -> list[dict[str, Any]]:
    trial_trace, valid = _trial_trace_and_valid(
        str(target["session"]), int(target["trial_idx"])
    )
    local_start, local_stop = int(target["local_start"]), int(target["local_stop"])
    window = trial_trace[local_start:local_stop]
    rows: list[dict[str, Any]] = []
    for order, config in enumerate(DETECTOR_CONFIGS):
        threshold = _speed_threshold_mad_valid_pairs(
            trial_trace, valid, dt=DT_S, z=float(config["speed_z"])
        )
        events, event_mask, _ = detect_microsaccade_events(
            trial_trace,
            dt=DT_S,
            threshold_deg_s=threshold,
            min_samples=1,
            pad_samples=int(config["event_pad_samples"]),
        )
        keep = valid[local_start:local_stop] & ~event_mask[local_start:local_stop]
        metrics = _component_metrics(
            window, float(target["image_edge_axis_deg"]), sample_mask=keep
        )
        overlaps = sum(
            int(event["offset"]) >= local_start and int(event["onset"]) < local_stop
            for event in events
        )
        rows.append(
            {
                "example_role": str(target["example_role"]),
                "session": str(target["session"]),
                "trial_idx": int(target["trial_idx"]),
                "detector_order": int(order),
                **config,
                "threshold_deg_s": float(threshold),
                "reviewed_primary_threshold_deg_s": float(
                    target["event_threshold_deg_s"]
                ),
                "n_trial_events": int(len(events)),
                "n_events_overlapping_window": int(overlaps),
                "n_flagged_window_samples": int(np.count_nonzero(~keep)),
                "retained_window_fraction": float(np.mean(keep)),
                "longest_clean_run_samples": _longest_true_run(keep),
                "window_remains_full_128_sample_clean": bool(np.all(keep)),
                **metrics,
            }
        )
    return rows


def _plot_input_patch(
    ax: plt.Axes, target: pd.Series, trace: np.ndarray
) -> None:
    patch, ppd = cp1.crop_patch(target)
    ax.imshow(patch, cmap="gray", origin="upper", interpolation="nearest")
    center = (np.asarray(patch.shape[::-1], dtype=float) - 1.0) / 2.0
    centered = trace - np.mean(trace, axis=0, keepdims=True)
    ax.plot(
        center[0] + centered[:, 0] * ppd,
        center[1] - centered[:, 1] * ppd,
        color="#f2ca52",
        lw=0.8,
    )
    ax.scatter(
        [center[0] + centered[0, 0] * ppd],
        [center[1] - centered[0, 1] * ppd],
        s=14,
        c="#36a2eb",
    )
    ax.scatter(
        [center[0] + centered[-1, 0] * ppd],
        [center[1] - centered[-1, 1] * ppd],
        s=15,
        c="#df4c4c",
        marker="s",
    )
    theta = math.radians(float(target["image_edge_axis_array_deg"]))
    length = 0.40 * min(patch.shape)
    dx, dy = length * math.cos(theta), length * math.sin(theta)
    ax.plot(
        [center[0] - dx, center[0] + dx],
        [center[1] - dy, center[1] + dy],
        color=PARALLEL_COLOR,
        lw=2.0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("local image + measured path", fontsize=8.4, weight="bold")


def _plot_raw_smooth(
    ax: plt.Axes, trace: np.ndarray, smoothed: np.ndarray, edge_axis_deg: float
) -> None:
    parallel_axis = cp1.axis_vector(edge_axis_deg)
    normal_axis = np.asarray([-parallel_axis[1], parallel_axis[0]])
    raw = trace - np.mean(trace, axis=0, keepdims=True)
    smooth = smoothed - np.mean(smoothed, axis=0, keepdims=True)
    raw_cn = np.column_stack([raw @ parallel_axis, raw @ normal_axis]) * 60.0
    smooth_cn = np.column_stack([smooth @ parallel_axis, smooth @ normal_axis]) * 60.0
    ax.plot(raw_cn[:, 0], raw_cn[:, 1], color=RAW_COLOR, lw=0.65, alpha=0.55)
    ax.plot(smooth_cn[:, 0], smooth_cn[:, 1], color=SMOOTH_COLOR, lw=1.1)
    ax.scatter(raw_cn[0, 0], raw_cn[0, 1], s=14, c="#36a2eb", zorder=4)
    ax.scatter(raw_cn[-1, 0], raw_cn[-1, 1], s=15, c="#df4c4c", marker="s", zorder=4)
    limit = max(float(np.max(np.abs(raw_cn))) * 1.10, 1.0)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.axhline(0, color="#b6bbc1", lw=0.5)
    ax.axvline(0, color="#b6bbc1", lw=0.5)
    ax.set_xlabel("parallel (arcmin)", fontsize=7)
    ax.set_ylabel("normal (arcmin)", fontsize=7)
    ax.tick_params(labelsize=6.2)
    ax.set_title("raw gray; 3-point smooth orange", fontsize=8.4, weight="bold")


def _plot_covariance_axes(
    ax: plt.Axes, trace: np.ndarray, raw_values: pd.Series, edge_axis_deg: float
) -> None:
    centered = trace - np.mean(trace, axis=0, keepdims=True)
    ax.plot(centered[:, 0], centered[:, 1], color="#9299a1", lw=0.65)
    scale = max(float(np.max(np.abs(centered))) * 0.80, 0.03)
    for angle, color, label, width in (
        (edge_axis_deg, PARALLEL_COLOR, "edge", 2.0),
        (float(raw_values["position_covariance_axis_deg"]), POSITION_COLOR, "position", 2.3),
        (float(raw_values["velocity_covariance_axis_deg"]), VELOCITY_COLOR, "velocity", 2.0),
    ):
        direction = cp1.axis_vector(angle) * scale
        ax.plot(
            [-direction[0], direction[0]],
            [-direction[1], direction[1]],
            color=color,
            lw=width,
            label=label,
        )
    limit = max(float(np.max(np.abs(centered))) * 1.10, 0.04)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.axhline(0, color="#b6bbc1", lw=0.5)
    ax.axvline(0, color="#b6bbc1", lw=0.5)
    ax.set_xlabel("screen x (deg)", fontsize=7)
    ax.set_ylabel("screen y (deg)", fontsize=7)
    ax.tick_params(labelsize=6.2)
    ax.set_title(
        "position vs velocity covariance axes\n"
        f"Δedge={float(raw_values['position_axis_edge_delta_deg']):.1f}° / "
        f"{float(raw_values['velocity_axis_edge_delta_deg']):.1f}°",
        fontsize=8.0,
        weight="bold",
    )
    ax.legend(fontsize=5.8, frameon=False, loc="lower right")


def _plot_multilag(ax: plt.Axes, values: pd.DataFrame) -> None:
    for processing, linestyle, alpha in (
        ("raw", "-", 1.0),
        ("smooth3", "--", 0.82),
    ):
        subset = values[values["processing"].eq(processing)].sort_values("lag_ms")
        ax.plot(
            subset["lag_ms"],
            subset["parallel_displacement_rms_arcmin"],
            color=PARALLEL_COLOR,
            ls=linestyle,
            marker="o",
            ms=2.8,
            lw=1.1,
            alpha=alpha,
        )
        ax.plot(
            subset["lag_ms"],
            subset["normal_displacement_rms_arcmin"],
            color=NORMAL_COLOR,
            ls=linestyle,
            marker="o",
            ms=2.8,
            lw=1.1,
            alpha=alpha,
        )
    ax.set_xscale("log")
    ax.set_xticks([8.33, 25, 50, 100, 250], ["8", "25", "50", "100", "250"])
    ax.set_xlabel("displacement lag (ms)", fontsize=7)
    ax.set_ylabel("displacement RMS (arcmin)", fontsize=7)
    ax.tick_params(labelsize=6.2)
    ax.grid(alpha=0.18, lw=0.5)
    raw = values[values["processing"].eq("raw")].sort_values("lag_ms")
    angle_ax = ax.twinx()
    resultants = raw["displacement_axial_resultant_r"].to_numpy(dtype=float)
    angle_ax.plot(
        raw["lag_ms"],
        raw["displacement_axis_edge_delta_deg"],
        color="#70777e",
        ls=":",
        lw=0.8,
        alpha=0.72,
    )
    angle_ax.scatter(
        raw["lag_ms"],
        raw["displacement_axis_edge_delta_deg"],
        color="#70777e",
        marker="x",
        s=(2.5 + 4.0 * resultants) ** 2,
        linewidths=0.8,
        alpha=0.72,
    )
    angle_ax.set_ylim(0, 90)
    angle_ax.set_yticks([0, 45, 90])
    angle_ax.set_ylabel("raw axis Δedge (deg)", fontsize=5.8, color="#70777e")
    angle_ax.tick_params(axis="y", labelsize=5.5, colors="#70777e")
    ax.set_title(
        "multilag RMS: parallel green / normal purple\n"
        "raw solid / smooth dashed; gray × = raw axis Δ",
        fontsize=7.7,
        weight="bold",
    )


def _plot_object_table(
    ax: plt.Axes, raw: pd.Series, smooth: pd.Series
) -> None:
    ax.axis("off")
    lines = ["object              ∥      ⟂      ∥−⟂  (arcmin)"]
    specs = (
        ("position RMS", "position_parallel_rms_arcmin", "position_normal_rms_arcmin"),
        ("position range", "position_parallel_range_arcmin", "position_normal_range_arcmin"),
        ("unsigned path", "unsigned_parallel_path_arcmin", "unsigned_normal_path_arcmin"),
        ("|start→end|", "start_end_parallel_arcmin", "start_end_normal_arcmin"),
    )
    for label, parallel_key, normal_key in specs:
        p = float(raw[parallel_key])
        n = float(raw[normal_key])
        if label == "|start→end|":
            p, n = abs(p), abs(n)
        lines.append(f"{label:<15} {p:6.1f} {n:6.1f} {p-n:+7.1f}")
    lines.append("")
    lines.append("after 3-point smoothing: ∥−⟂")
    for label, parallel_key, normal_key in specs:
        p = float(smooth[parallel_key])
        n = float(smooth[normal_key])
        if label == "|start→end|":
            p, n = abs(p), abs(n)
        lines.append(f"{label:<15} {p-n:+7.1f}")
    lines.append("")
    lines.append(
        "step reversal ∥/⟂  "
        f"{float(raw['parallel_step_reversal_fraction']):.2f}/"
        f"{float(raw['normal_step_reversal_fraction']):.2f}"
    )
    lines.append(
        "lag-1 autocorr ∥/⟂ "
        f"{float(raw['parallel_step_autocorr_lag1']):+.2f}/"
        f"{float(raw['normal_step_autocorr_lag1']):+.2f}"
    )
    ax.text(
        0.01,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.25,
        family="monospace",
    )
    ax.set_title("absolute behavioral objects", fontsize=8.4, weight="bold")


def _plot_detector(ax: plt.Axes, detector: pd.DataFrame) -> None:
    detector = detector.sort_values("detector_order")
    x = np.arange(len(detector))
    y = detector["position_rms_parallel_minus_normal_arcmin"].to_numpy(dtype=float)
    colors = [DETECTOR_COLORS[str(item)] for item in detector["detector_setting"]]
    ax.axhline(0, color="#8b9299", lw=0.7, ls=":")
    ax.plot(x, y, color="#aab0b6", lw=0.8)
    ax.scatter(x, y, c=colors, s=34, zorder=4)
    ax.set_xticks(x, ["aggr.", "primary", "permiss."])
    ax.set_ylabel("position RMS ∥−⟂ (arcmin)", fontsize=7)
    ax.tick_params(labelsize=6.2)
    ax.grid(axis="y", alpha=0.18, lw=0.5)
    flagged = "/".join(str(int(item)) for item in detector["n_flagged_window_samples"])
    path_delta = detector["unsigned_path_parallel_minus_normal_arcmin"].to_numpy(dtype=float)
    ax.set_title(
        f"event detector sensitivity\nflagged samples a/p/l = {flagged}",
        fontsize=7.9,
        weight="bold",
    )
    ax.text(
        0.02,
        0.03,
        f"path ∥−⟂ a/p/l = {path_delta[0]:+.0f}/{path_delta[1]:+.0f}/{path_delta[2]:+.0f}",
        transform=ax.transAxes,
        fontsize=5.8,
        color="#4a4f55",
        va="bottom",
    )


def _render(
    targets: list[pd.Series],
    traces: list[np.ndarray],
    object_values: pd.DataFrame,
    multilag_values: pd.DataFrame,
    detector_values: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        len(targets),
        6,
        figsize=(18.1, 15.8),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.05, 1.15, 1.26, 1.12]},
    )
    for row_index, (target, trace) in enumerate(zip(targets, traces, strict=True)):
        role = str(target["example_role"])
        smoothed = _smooth_trace(trace)
        raw = object_values[
            object_values["example_role"].eq(role)
            & object_values["processing"].eq("raw")
        ].iloc[0]
        smooth = object_values[
            object_values["example_role"].eq(role)
            & object_values["processing"].eq("smooth3")
        ].iloc[0]
        lag = multilag_values[multilag_values["example_role"].eq(role)]
        detector = detector_values[detector_values["example_role"].eq(role)]
        _plot_input_patch(axes[row_index, 0], target, trace)
        axes[row_index, 0].set_ylabel(
            f"{row_index + 1}. {cp1.ROLE_LABEL[role]}\n"
            f"{target['subject']} | tr {int(target['trial_idx'])}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=12,
            fontsize=8.0,
            weight="bold",
        )
        _plot_raw_smooth(
            axes[row_index, 1], trace, smoothed, float(target["image_edge_axis_deg"])
        )
        _plot_covariance_axes(
            axes[row_index, 2], trace, raw, float(target["image_edge_axis_deg"])
        )
        _plot_multilag(axes[row_index, 3], lag)
        _plot_object_table(axes[row_index, 4], raw, smooth)
        _plot_detector(axes[row_index, 5], detector)

    fig.suptitle(
        "Figure 4 behavior audit — Checkpoint 3: what behavioral object is contour-relative?\n"
        "Six preselected windows; observed paths first, absolute component values and detector sensitivity second",
        y=0.997,
        fontsize=12.7,
        weight="bold",
    )
    fig.text(
        0.285,
        0.952,
        "OBSERVED PATH / LIGHT SMOOTHING",
        ha="center",
        fontsize=10.0,
        weight="bold",
    )
    fig.text(
        0.73,
        0.952,
        "DERIVED BEHAVIORAL OBJECTS",
        ha="center",
        fontsize=10.0,
        weight="bold",
    )
    fig.subplots_adjust(
        left=0.125, right=0.993, top=0.925, bottom=0.05, hspace=0.50, wspace=0.39
    )
    boundary = 0.5 * (
        axes[0, 1].get_position().x1 + axes[0, 2].get_position().x0
    )
    fig.add_artist(
        plt.Line2D(
            [boundary, boundary],
            [0.04, 0.95],
            transform=fig.transFigure,
            color="#a6adb5",
            lw=0.9,
        )
    )
    fig.text(
        0.99,
        0.014,
        "Position quantities use the full 128-sample cloud. Path uses consecutive sample-to-sample steps. "
        "Smoothing is zero-phase [0.25, 0.50, 0.25]. Detector settings are fixed before viewing: z/pad=4/3, 6/1, 8/0.",
        ha="right",
        fontsize=6.8,
        color="#4a4f55",
    )
    fig.savefig(out_dir / "checkpoint3_behavior_object_multilag.png", dpi=220)
    fig.savefig(out_dir / "checkpoint3_behavior_object_multilag.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(SELECTED_WINDOWS)
    source = pd.read_csv(SOURCE_WINDOWS)
    targets: list[pd.Series] = []
    traces: list[np.ndarray] = []
    object_rows: list[dict[str, Any]] = []
    multilag_rows: list[dict[str, Any]] = []
    detector_rows: list[dict[str, Any]] = []

    for selected_row in selected.itertuples(index=False):
        target = cp2b._find_target(source, pd.Series(selected_row._asdict()))
        trace = np.asarray(_window_trace(target), dtype=np.float64)
        if trace.shape != (128, 2) or not np.isfinite(trace).all():
            raise RuntimeError(
                f"Expected native 128x2 trace for {target['example_role']}, got {trace.shape}"
            )
        print(
            f"[checkpoint3] {target['example_role']} "
            f"({target['session']} trial {int(target['trial_idx'])})",
            flush=True,
        )
        smoothed = _smooth_trace(trace)
        for processing, processed in (("raw", trace), ("smooth3", smoothed)):
            object_rows.append(
                {
                    "example_role": str(target["example_role"]),
                    "session": str(target["session"]),
                    "subject": str(target["subject"]),
                    "trial_idx": int(target["trial_idx"]),
                    "global_start": int(target["global_start"]),
                    "global_stop": int(target["global_stop"]),
                    "processing": processing,
                    "image_edge_axis_deg": float(target["image_edge_axis_deg"]),
                    "image_orientation_coherence": float(
                        target["image_orientation_coherence"]
                    ),
                    **_component_metrics(
                        processed, float(target["image_edge_axis_deg"])
                    ),
                }
            )
            multilag_rows.extend(
                _multilag_metrics(
                    processed,
                    float(target["image_edge_axis_deg"]),
                    example_role=str(target["example_role"]),
                    processing=processing,
                )
            )
        detector_rows.extend(_detector_rows(target))
        targets.append(target)
        traces.append(trace)

    object_values = pd.DataFrame(object_rows)
    multilag_values = pd.DataFrame(multilag_rows)
    detector_values = pd.DataFrame(detector_rows)

    primary = detector_values[detector_values["detector_setting"].eq("primary")]
    threshold_error = np.max(
        np.abs(
            primary["threshold_deg_s"].to_numpy(dtype=float)
            - primary["reviewed_primary_threshold_deg_s"].to_numpy(dtype=float)
        )
    )
    if threshold_error > 1e-9:
        raise RuntimeError(
            f"Primary event thresholds do not reproduce reviewed extraction: {threshold_error}"
        )
    if not primary["window_remains_full_128_sample_clean"].all():
        raise RuntimeError("A reviewed selected window is not clean under the primary detector")

    object_values.to_csv(out_dir / "checkpoint3_behavior_object_values.csv", index=False)
    multilag_values.to_csv(out_dir / "checkpoint3_multilag_values.csv", index=False)
    detector_values.to_csv(out_dir / "checkpoint3_detector_sensitivity.csv", index=False)
    _render(
        targets,
        traces,
        object_values,
        multilag_values,
        detector_values,
        out_dir,
    )

    metadata = {
        "artifact_type": "map_first_targeted_visualization_checkpoint",
        "checkpoint": 3,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "population_inference_performed": False,
        "input_windows": SOURCE_WINDOWS,
        "checkpoint1_selection": SELECTED_WINDOWS,
        "n_selected_windows": int(len(targets)),
        "trace_contract": "native reviewed 128-sample BackImage trace",
        "coordinate_contract": (
            "parallel is the local Sobel edge axis; normal is its 90-degree rotation; "
            "all component magnitudes are saved in arcmin"
        ),
        "smoothing_contract": {
            "name": "smooth3",
            "kernel": SMOOTH_KERNEL,
            "boundary": "edge replication",
            "phase": "zero-phase symmetric diagnostic",
        },
        "multilag_samples": LAGS,
        "multilag_ms": [lag * DT_S * 1000.0 for lag in LAGS],
        "detector_configs": DETECTOR_CONFIGS,
        "detector_sensitivity_contract": (
            "keep the selected nominal window fixed; recompute whole-trial thresholds and "
            "event masks; position metrics use retained samples and path metrics use only "
            "adjacent retained pairs"
        ),
        "git_revision": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "outputs": [
            "checkpoint3_behavior_object_multilag.png",
            "checkpoint3_behavior_object_multilag.pdf",
            "checkpoint3_behavior_object_values.csv",
            "checkpoint3_multilag_values.csv",
            "checkpoint3_detector_sensitivity.csv",
            "checkpoint3_run_metadata.json",
        ],
    }
    _write_json(out_dir / "checkpoint3_run_metadata.json", metadata)
    print(f"[checkpoint3] wrote artifacts to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
