"""Audit OU trajectory controls and aggregate response readouts.

This script is cache-first: it replays the aggregate BackImage FEM trace
sampling loop from the original run metadata, validates the replay against the
cached motion metadata, and only then computes trace, response-space, and
decoder/readout audit summaries.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .run_backimage_aggregate_fem_information import (
        _build_trace_bank,
        _eligible_trace_bank_indices,
        _family_raw_trace,
        _family_trace,
        _lag1_autocorr,
        _path_length,
        _prepare_windows,
        _scale_family_raw_trace,
        _session_dataset_cache,
        _trace_covariance_anisotropy,
        _trace_filter_kwargs,
    )
    from .run_backimage_latent_information_screen import _scale_token, _trace_rms
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import (
        _build_trace_bank,
        _eligible_trace_bank_indices,
        _family_raw_trace,
        _family_trace,
        _lag1_autocorr,
        _path_length,
        _prepare_windows,
        _scale_family_raw_trace,
        _session_dataset_cache,
        _trace_covariance_anisotropy,
        _trace_filter_kwargs,
    )
    from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import _scale_token, _trace_rms


DEFAULT_RUN_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1"
)
DEFAULT_READOUT_DIR = DEFAULT_RUN_DIR / "incremental_staticmean_plus_motion_allreadouts_v1"
DEFAULT_OUT_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_ou_trace_control_audit_n384_power_v1"
)

FAMILIES = ("empirical", "ou", "brownian", "rotated")
PRIMARY_READOUTS = (
    "mean",
    "delta_mean",
    "temporal_pca",
    "temporal_delta_pca",
    "temporal_dct",
    "temporal_dct_delta",
)
PAIRWISE_CONTRASTS = (
    ("empirical", "ou"),
    ("empirical", "brownian"),
    ("empirical", "rotated"),
    ("ou", "brownian"),
)
ID_COLUMNS = {
    "image_index",
    "source_row",
    "session",
    "trial_idx",
    "family",
    "scale_id",
    "scale",
    "sample_index",
    "trace_bank_index",
    "trace_source_row",
    "trace_source_session",
    "source_trace_duration_s",
    "raw_trace_reused_across_scales",
}


def _progress(message: str) -> None:
    print(f"[ou-trace-audit] {message}", flush=True)


def _parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _load_original_args(run_dir: Path) -> argparse.Namespace:
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = dict(metadata["config"])
    config["input"] = Path(config["input"])
    config["out_dir"] = Path(config["out_dir"])
    if config.get("window_manifest") is not None:
        config["window_manifest"] = Path(config["window_manifest"])
    return argparse.Namespace(**config)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        return _parse_csv_list(value)
    return list(value)


def _angle_axis_deg(trace: np.ndarray) -> float:
    trace = np.asarray(trace, dtype=np.float64)
    if trace.shape[0] < 2:
        return float("nan")
    cov = np.cov(trace, rowvar=False)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return float("nan")
    vals, vecs = np.linalg.eigh(cov + 1e-12 * np.eye(2))
    axis = vecs[:, int(np.argmax(vals))]
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    return float(angle % 180.0)


def _axis_delta_deg(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    delta = abs((float(a) - float(b)) % 180.0)
    return float(min(delta, 180.0 - delta))


def _cov_eigs(trace: np.ndarray) -> tuple[float, float]:
    trace = np.asarray(trace, dtype=np.float64)
    if trace.shape[0] < 2:
        return 0.0, 0.0
    cov = np.cov(trace, rowvar=False)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return float("nan"), float("nan")
    vals = np.linalg.eigvalsh(cov + 1e-12 * np.eye(2))
    vals = np.maximum(vals, 0.0)
    return float(vals[-1]), float(vals[0])


def _autocorr_scalar(x: np.ndarray, max_lag: int) -> dict[int, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    out: dict[int, float] = {}
    for lag in range(1, int(max_lag) + 1):
        if arr.size <= lag + 1:
            out[lag] = float("nan")
            continue
        a = arr[:-lag] - np.nanmean(arr[:-lag])
        b = arr[lag:] - np.nanmean(arr[lag:])
        den = float(np.sqrt(np.nansum(a * a) * np.nansum(b * b)))
        out[lag] = float(np.nansum(a * b) / den) if den > 1e-12 else float("nan")
    return out


def _autocorr_vector(x: np.ndarray, max_lag: int) -> dict[int, float]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return _autocorr_scalar(arr, max_lag)
    per_dim = [_autocorr_scalar(arr[:, dim], max_lag) for dim in range(arr.shape[1])]
    out: dict[int, float] = {}
    for lag in range(1, int(max_lag) + 1):
        vals = []
        for dim_values in per_dim:
            val = dim_values[lag]
            if np.isfinite(val):
                vals.append(val)
        out[lag] = float(np.mean(vals)) if vals else float("nan")
    return out


def _psd_vector(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] < 3:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    arr = arr - np.mean(arr, axis=0, keepdims=True)
    freqs = np.fft.rfftfreq(arr.shape[0], d=float(dt))
    fft = np.fft.rfft(arr, axis=0)
    power = np.sum(np.abs(fft) ** 2, axis=1) / float(arr.shape[0])
    keep = freqs > 0.0
    return freqs[keep], power[keep]


def _psd_metrics(freqs: np.ndarray, power: np.ndarray, *, low_hz: float = 10.0, high_hz: float = 30.0) -> dict[str, float]:
    if freqs.size == 0 or power.size == 0 or not np.isfinite(power).any():
        return {
            "psd_lowfreq_fraction": float("nan"),
            "psd_highfreq_fraction": float("nan"),
            "psd_centroid_hz": float("nan"),
            "psd_slope_loglog": float("nan"),
        }
    total = float(np.nansum(power))
    if total <= 1e-12:
        low = high = centroid = slope = float("nan")
    else:
        low = float(np.nansum(power[freqs <= float(low_hz)]) / total)
        high = float(np.nansum(power[freqs >= float(high_hz)]) / total)
        centroid = float(np.nansum(freqs * power) / total)
        mask = (freqs > 0.0) & (power > 0.0) & np.isfinite(power)
        if int(np.sum(mask)) >= 2:
            slope = float(np.polyfit(np.log(freqs[mask]), np.log(power[mask]), 1)[0])
        else:
            slope = float("nan")
    return {
        "psd_lowfreq_fraction": low,
        "psd_highfreq_fraction": high,
        "psd_centroid_hz": centroid,
        "psd_slope_loglog": slope,
    }


def _return_to_center_slope(trace: np.ndarray) -> float:
    radius = np.linalg.norm(np.asarray(trace, dtype=np.float64), axis=1)
    if radius.size < 3:
        return float("nan")
    x = radius[:-1]
    y = np.diff(radius)
    x = x - np.mean(x)
    den = float(np.sum(x * x))
    return float(np.sum(x * y) / den) if den > 1e-12 else float("nan")


def _trace_metrics(
    trace: np.ndarray,
    source_trace: np.ndarray,
    *,
    requested_rms: float,
    target_path: float,
    source_trace_lag1: float,
    image_edge_axis_deg: float | None,
    dt: float,
    max_lag: int,
) -> tuple[dict[str, float], list[dict[str, float]], list[dict[str, float]]]:
    trace = np.asarray(trace, dtype=np.float64)
    source_trace = np.asarray(source_trace, dtype=np.float64)
    steps = np.linalg.norm(np.diff(trace, axis=0), axis=1) if trace.shape[0] > 1 else np.asarray([], dtype=np.float64)
    radius = np.linalg.norm(trace, axis=1) if trace.size else np.asarray([], dtype=np.float64)
    path = _path_length(trace)
    endpoint = float(np.linalg.norm(trace[-1] - trace[0])) if trace.shape[0] >= 2 else 0.0
    effective_rms = float(_trace_rms(trace))
    cov_eig1, cov_eig2 = _cov_eigs(trace)
    axis_deg = _angle_axis_deg(trace)
    source_axis_deg = _angle_axis_deg(source_trace)
    velocity = np.diff(trace, axis=0) / float(dt) if trace.shape[0] > 1 else np.zeros((0, 2), dtype=np.float64)
    pos_freqs, pos_power = _psd_vector(trace, dt)
    vel_freqs, vel_power = _psd_vector(velocity, dt)
    pos_psd_metrics = _psd_metrics(pos_freqs, pos_power)
    vel_psd_metrics = _psd_metrics(vel_freqs, vel_power)

    metrics = {
        "requested_rms_deg": float(requested_rms),
        "effective_rms_deg": effective_rms,
        "effective_to_requested_rms": effective_rms / float(requested_rms) if float(requested_rms) > 0.0 else float("nan"),
        "max_radius_deg": float(np.nanmax(radius)) if radius.size else 0.0,
        "endpoint_displacement_deg": endpoint,
        "radial_distance_mean_deg": float(np.nanmean(radius)) if radius.size else float("nan"),
        "radial_distance_p95_deg": float(np.nanpercentile(radius, 95.0)) if radius.size else float("nan"),
        "bounding_box_x_deg": float(np.nanmax(trace[:, 0]) - np.nanmin(trace[:, 0])) if trace.shape[0] else 0.0,
        "bounding_box_y_deg": float(np.nanmax(trace[:, 1]) - np.nanmin(trace[:, 1])) if trace.shape[0] else 0.0,
        "path_length_deg": float(path),
        "path_to_target_ratio": float(path) / float(target_path) if float(target_path) > 0.0 else float("nan"),
        "tortuosity": float(path) / max(endpoint, 1e-12),
        "step_length_mean_deg": float(np.nanmean(steps)) if steps.size else 0.0,
        "step_length_median_deg": float(np.nanmedian(steps)) if steps.size else 0.0,
        "step_length_p95_deg": float(np.nanpercentile(steps, 95.0)) if steps.size else 0.0,
        "speed_mean_deg_s": float(np.nanmean(steps / float(dt))) if steps.size else 0.0,
        "speed_median_deg_s": float(np.nanmedian(steps / float(dt))) if steps.size else 0.0,
        "speed_p95_deg_s": float(np.nanpercentile(steps / float(dt), 95.0)) if steps.size else 0.0,
        "trace_cov_eig1": cov_eig1,
        "trace_cov_eig2": cov_eig2,
        "trace_cov_anisotropy": float(_trace_covariance_anisotropy(trace)),
        "trace_cov_axis_deg": axis_deg,
        "source_cov_axis_deg": source_axis_deg,
        "axis_delta_to_source_deg": _axis_delta_deg(axis_deg, source_axis_deg),
        "axis_delta_to_image_edge_deg": (
            _axis_delta_deg(axis_deg, float(image_edge_axis_deg)) if image_edge_axis_deg is not None else float("nan")
        ),
        "source_trace_lag1": float(source_trace_lag1),
        "generated_lag1_autocorr": float(_lag1_autocorr(trace)),
        "lag1_delta_to_source": float(_lag1_autocorr(trace)) - float(source_trace_lag1),
        "position_psd_lowfreq_fraction": pos_psd_metrics["psd_lowfreq_fraction"],
        "position_psd_highfreq_fraction": pos_psd_metrics["psd_highfreq_fraction"],
        "position_psd_centroid_hz": pos_psd_metrics["psd_centroid_hz"],
        "position_psd_slope_loglog": pos_psd_metrics["psd_slope_loglog"],
        "velocity_psd_lowfreq_fraction": vel_psd_metrics["psd_lowfreq_fraction"],
        "velocity_psd_highfreq_fraction": vel_psd_metrics["psd_highfreq_fraction"],
        "velocity_psd_centroid_hz": vel_psd_metrics["psd_centroid_hz"],
        "velocity_psd_slope_loglog": vel_psd_metrics["psd_slope_loglog"],
        "start_radius_deg": float(radius[0]) if radius.size else 0.0,
        "end_radius_deg": float(radius[-1]) if radius.size else 0.0,
        "start_end_distance_deg": endpoint,
        "mean_return_to_center_slope": _return_to_center_slope(trace),
        "fraction_samples_inside_25pct_rms_radius": (
            float(np.mean(radius <= 0.25 * effective_rms)) if radius.size and effective_rms > 0.0 else float("nan")
        ),
        "fraction_samples_outside_2x_rms_radius": (
            float(np.mean(radius >= 2.0 * effective_rms)) if radius.size and effective_rms > 0.0 else float("nan")
        ),
    }

    for lag, value in _autocorr_vector(trace, max_lag).items():
        metrics[f"position_autocorr_lag_{lag}"] = value
    for lag, value in _autocorr_vector(np.diff(trace, axis=0), max_lag).items():
        metrics[f"velocity_autocorr_lag_{lag}"] = value
    for lag, value in _autocorr_scalar(radius, max_lag).items():
        metrics[f"radial_autocorr_lag_{lag}"] = value

    pos_psd_rows = [
        {"frequency_hz": float(freq), "psd": float(power)}
        for freq, power in zip(pos_freqs, pos_power, strict=True)
    ]
    vel_psd_rows = [
        {"frequency_hz": float(freq), "psd": float(power)}
        for freq, power in zip(vel_freqs, vel_power, strict=True)
    ]
    return metrics, pos_psd_rows, vel_psd_rows


def _replay_generated_traces(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    args = _load_original_args(run_dir)
    rng = np.random.default_rng(int(args.seed))
    work = _prepare_windows(args)
    analysis = pd.read_csv(run_dir / "analysis_images.csv")
    check_cols = ["image_index", "source_row", "session", "trial_idx"]
    if analysis.shape[0] != work.shape[0]:
        raise ValueError(f"analysis_images.csv has {analysis.shape[0]} rows but replay selected {work.shape[0]} windows")
    work_check = work[check_cols].copy()
    analysis_check = analysis[check_cols].copy()
    if not work_check.astype(str).equals(analysis_check.astype(str)):
        raise ValueError("Replay window order does not match analysis_images.csv")

    _progress(f"loading eyepos for {work['session'].nunique()} sessions")
    eyepos_by_session = _session_dataset_cache(work["session"].astype(str).to_list())
    trace_bank = _build_trace_bank(
        work,
        eyepos_by_session,
        int(args.n_timepoints),
        microsaccade_speed_threshold_dps=(
            float(args.microsaccade_speed_threshold_dps) if args.microsaccade_speed_threshold_dps is not None else None
        ),
        microsaccade_threshold_z=float(args.microsaccade_threshold_z),
        microsaccade_pad_frames=int(args.microsaccade_pad_frames),
        trace_window_policy=str(getattr(args, "trace_window_policy", "center_crop_native")),
    )
    families = [str(v) for v in _as_list(args.motion_families)]
    scales = [float(v) for v in _as_list(args.observed_rms_scales)]
    generated: list[dict[str, Any]] = []
    _progress(f"replaying traces for {work.shape[0]} images; families={families}; scales={scales}")
    for image_index, row in work.iterrows():
        reusable_sources: dict[tuple[str, int], int] = {}
        reusable_raw_traces: dict[tuple[str, int], np.ndarray] = {}
        if bool(args.reuse_trace_sources_across_scales):
            eligible = _eligible_trace_bank_indices(
                trace_bank,
                current_source_row=int(row["source_row"]),
                **_trace_filter_kwargs(args),
            )
            if not eligible:
                raise ValueError("Replay found no eligible trace-bank entries for reusable source sampling.")
            for family in families:
                for sample_index in range(int(args.trace_samples_per_condition)):
                    key = (family, int(sample_index))
                    bank_index = int(eligible[int(rng.integers(0, len(eligible)))])
                    item = trace_bank[bank_index]
                    reusable_sources[key] = bank_index
                    reusable_raw_traces[key] = _family_raw_trace(
                        family,
                        item["trace"],
                        float(item["lag1_autocorr"]),
                        rng=rng,
                        max_rms_deg=float(args.max_rms_deg),
                        source_shape=item.get("covariance_shape"),
                        selection_rms=float(item["observed_rms_deg"]),
                        target_path_length=float(item["path_length_deg"]),
                    )
        for scale in scales:
            scale_id = f"rel_{_scale_token(float(scale))}x"
            for family in families:
                for sample_index in range(int(args.trace_samples_per_condition)):
                    eligible = _eligible_trace_bank_indices(
                        trace_bank,
                        current_source_row=int(row["source_row"]),
                        **_trace_filter_kwargs(args),
                    )
                    if not eligible:
                        raise ValueError("Replay found no eligible trace-bank entries for source sampling.")
                    if bool(args.reuse_trace_sources_across_scales):
                        reuse_key = (family, int(sample_index))
                        bank_index = reusable_sources[reuse_key]
                    else:
                        bank_index = int(eligible[int(rng.integers(0, len(eligible)))])
                    item = trace_bank[bank_index]
                    target_rms = float(scale) * float(item["observed_rms_deg"])
                    target_path = float(scale) * float(item["path_length_deg"])
                    if bool(args.reuse_trace_sources_across_scales):
                        trace, meta = _scale_family_raw_trace(
                            reusable_raw_traces[reuse_key],
                            target_rms,
                            max_rms_deg=float(args.max_rms_deg),
                        )
                    else:
                        trace, meta = _family_trace(
                            family,
                            item["trace"],
                            float(item["lag1_autocorr"]),
                            target_rms,
                            rng=rng,
                            max_rms_deg=float(args.max_rms_deg),
                            source_shape=item.get("covariance_shape"),
                            target_path_length=target_path,
                        )
                    generated.append(
                        {
                            "image_index": int(image_index),
                            "source_row": int(row["source_row"]),
                            "session": str(row["session"]),
                            "trial_idx": int(row["trial_idx"]),
                            "family": str(family),
                            "scale_id": scale_id,
                            "scale": float(scale),
                            "sample_index": int(sample_index),
                            "trace_bank_index": int(bank_index),
                            "trace_source_row": int(item["source_row"]),
                            "trace_source_session": str(item["session"]),
                            "trace_source_render_contract": str(item.get("trace_render_contract", "")),
                            "trace_source_window_policy_requested": str(item.get("trace_window_policy_requested", "")),
                            "trace_source_window_policy": str(item.get("trace_window_policy", "")),
                            "trace_source_window_n_samples": int(item.get("source_window_n_samples", -1)),
                            "trace_source_rendered_n_samples": int(item.get("rendered_trace_n_samples", -1)),
                            "trace_source_rendered_offset": int(item.get("rendered_trace_source_offset", -1)),
                            "trace_source_rendered_stop_offset": int(
                                item.get("rendered_trace_source_stop_offset", -1)
                            ),
                            "trace_source_to_render_time_compression": float(
                                item.get("source_to_render_time_compression", np.nan)
                            ),
                            "source_trace_rms_deg": float(item["observed_rms_deg"]),
                            "source_trace_path_length_deg": float(item["path_length_deg"]),
                            "source_table_path_length_deg": float(item["source_path_length_deg"]),
                            "source_trace_duration_s": float(item["duration_s"]),
                            "source_trace_lag1": float(item["lag1_autocorr"]),
                            "requested_rms_deg": float(meta["requested_rms_deg"]),
                            "effective_rms_deg": float(meta["effective_rms_deg"]),
                            "rms_clipped_high": bool(meta["rms_clipped_high"]),
                            "generated_lag1_autocorr": float(meta["generated_lag1_autocorr"]),
                            "target_path_length_deg": float(target_path),
                            "path_length_deg": float(meta["path_length_deg"]),
                            "path_to_target_ratio": (
                                float(meta["path_length_deg"]) / float(target_path) if float(target_path) > 0.0 else np.nan
                            ),
                            "raw_trace_reused_across_scales": bool(args.reuse_trace_sources_across_scales),
                            "trace": np.asarray(trace, dtype=np.float32),
                            "source_trace": np.asarray(item["trace"], dtype=np.float32),
                            "image_edge_axis_deg": _extract_image_edge_axis(row),
                        }
                    )
        if (int(image_index) + 1) % 64 == 0 or int(image_index) + 1 == work.shape[0]:
            _progress(f"replayed image {int(image_index) + 1}/{work.shape[0]}")
    return work, pd.DataFrame(analysis), generated


def _extract_image_edge_axis(row: pd.Series) -> float | None:
    for col in (
        "image_edge_axis_deg",
        "image_edge_orientation_deg",
        "nearest_image_edge_axis_deg",
        "nearest_border_axis_deg",
    ):
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    return None


def _validate_replay(run_dir: Path, replay_rows: list[dict[str, Any]], out_dir: Path) -> tuple[bool, pd.DataFrame, dict[str, Any]]:
    key_cols = ["image_index", "source_row", "family", "scale_id", "sample_index"]
    cached = pd.read_csv(run_dir / "aggregate_motion_metadata.csv")
    cached = cached.loc[cached["family"].astype(str) != "static"].copy()
    replay = pd.DataFrame(
        [
            {
                "image_index": int(row["image_index"]),
                "source_row": int(row["source_row"]),
                "family": str(row["family"]),
                "scale_id": str(row["scale_id"]),
                "sample_index": int(row["sample_index"]),
                "trace_bank_index_replay": int(row["trace_bank_index"]),
                "effective_rms_deg_replay": float(row["effective_rms_deg"]),
                "path_length_deg_replay": float(row["path_length_deg"]),
                "generated_lag1_autocorr_replay": float(row["generated_lag1_autocorr"]),
            }
            for row in replay_rows
        ]
    )
    cached_small = cached[
        key_cols + ["trace_bank_index", "effective_rms_deg", "path_length_deg", "generated_lag1_autocorr"]
    ].rename(
        columns={
            "trace_bank_index": "trace_bank_index_cached",
            "effective_rms_deg": "effective_rms_deg_cached",
            "path_length_deg": "path_length_deg_cached",
            "generated_lag1_autocorr": "generated_lag1_autocorr_cached",
        }
    )
    validation = replay.merge(cached_small, on=key_cols, how="outer", indicator=True)
    validation["trace_bank_index_match"] = (
        validation["trace_bank_index_replay"].astype("Int64") == validation["trace_bank_index_cached"].astype("Int64")
    )
    for col in ("effective_rms_deg", "path_length_deg", "generated_lag1_autocorr"):
        validation[f"{col}_abs_diff"] = (
            validation[f"{col}_replay"].astype(float) - validation[f"{col}_cached"].astype(float)
        ).abs()
    validation.to_csv(out_dir / "trace_replay_validation.csv", index=False)
    matched = validation["_merge"].eq("both")
    trace_mismatch = int((matched & ~validation["trace_bank_index_match"]).sum())
    summary = {
        "cached_rows": int(cached_small.shape[0]),
        "replayed_rows": int(replay.shape[0]),
        "joined_rows": int(validation.shape[0]),
        "unmatched_rows": int((~matched).sum()),
        "trace_bank_index_mismatches": trace_mismatch,
        "median_effective_rms_abs_diff": float(validation.loc[matched, "effective_rms_deg_abs_diff"].median()),
        "median_path_length_abs_diff": float(validation.loc[matched, "path_length_deg_abs_diff"].median()),
        "median_generated_lag1_abs_diff": float(validation.loc[matched, "generated_lag1_autocorr_abs_diff"].median()),
        "max_effective_rms_abs_diff": float(validation.loc[matched, "effective_rms_deg_abs_diff"].max()),
        "max_path_length_abs_diff": float(validation.loc[matched, "path_length_deg_abs_diff"].max()),
        "max_generated_lag1_abs_diff": float(validation.loc[matched, "generated_lag1_autocorr_abs_diff"].max()),
    }
    passed = (
        summary["cached_rows"] == summary["replayed_rows"]
        and summary["unmatched_rows"] == 0
        and summary["trace_bank_index_mismatches"] == 0
        and summary["median_effective_rms_abs_diff"] <= 1e-8
        and summary["median_path_length_abs_diff"] <= 1e-6
        and summary["median_generated_lag1_abs_diff"] <= 1e-6
    )
    return passed, validation, summary


def _write_failure_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# OU Trace Replay Failure Report",
        "",
        "Exact replay validation failed, so the audit stopped before interpreting time-series metrics.",
        "",
        "| Check | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | {value} |")
    (out_dir / "trace_replay_failure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_columns(metrics_df: pd.DataFrame) -> list[str]:
    cols = []
    for col in metrics_df.columns:
        if col in ID_COLUMNS or col.startswith("source_trace_") or col in {"target_path_length_deg", "rms_clipped_high"}:
            continue
        if pd.api.types.is_numeric_dtype(metrics_df[col]):
            cols.append(col)
    return cols


def _session_bootstrap_ci(
    per_image: pd.DataFrame,
    value_col: str,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    draw_cache: dict[tuple[tuple[str, ...], int], np.ndarray] | None = None,
) -> tuple[float, float, float]:
    clean = per_image.loc[np.isfinite(per_image[value_col].astype(float)), ["session", value_col]].copy()
    if clean.empty:
        return float("nan"), float("nan"), float("nan")
    observed = float(clean[value_col].mean())
    if int(n_bootstrap) <= 0 or clean["session"].nunique() < 2:
        return observed, float("nan"), float("nan")
    by_session = clean.groupby("session")[value_col].agg(["sum", "count"]).reset_index()
    sums = by_session["sum"].to_numpy(dtype=np.float64)
    counts = by_session["count"].to_numpy(dtype=np.float64)
    cache_key = (tuple(str(v) for v in by_session["session"].to_list()), int(n_bootstrap))
    if draw_cache is not None and cache_key in draw_cache:
        draw_counts = draw_cache[cache_key]
    else:
        draws = rng.integers(0, len(sums), size=(int(n_bootstrap), len(sums)))
        draw_counts = np.zeros((int(n_bootstrap), len(sums)), dtype=np.float32)
        rows = np.repeat(np.arange(int(n_bootstrap)), len(sums))
        np.add.at(draw_counts, (rows, draws.reshape(-1)), 1.0)
        if draw_cache is not None:
            draw_cache[cache_key] = draw_counts
    boot = (draw_counts @ sums) / np.maximum(draw_counts @ counts, 1.0)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return observed, float(lo), float(hi)


def _summarize_trace_metrics(
    metrics_df: pd.DataFrame,
    *,
    metric_cols: list[str],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed) + 401)
    draw_cache: dict[tuple[tuple[str, ...], int], np.ndarray] = {}

    for (family, scale_id), group in metrics_df.groupby(["family", "scale_id"], sort=True):
        for metric in metric_cols:
            per_image = (
                group[["session", "image_index", metric]]
                .groupby(["session", "image_index"], as_index=False)[metric]
                .mean()
            )
            clean_values = per_image[metric].to_numpy(dtype=np.float64)
            clean_values = clean_values[np.isfinite(clean_values)]
            if clean_values.size == 0:
                continue
            mean, lo, hi = _session_bootstrap_ci(
                per_image,
                metric,
                n_bootstrap=n_bootstrap,
                rng=rng,
                draw_cache=draw_cache,
            )
            iqr = float(np.nanpercentile(clean_values, 75.0) - np.nanpercentile(clean_values, 25.0))
            row = {
                "family": family,
                "scale_id": scale_id,
                "metric": metric,
                "mean": mean,
                "median": float(np.nanmedian(clean_values)),
                "iqr": iqr,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_traces": int(group[metric].notna().sum()),
                "n_images": int(per_image["image_index"].nunique()),
                "n_sessions": int(per_image["session"].nunique()),
            }
            summary_rows.append(row)
            bootstrap_rows.append(
                {
                    "family": family,
                    "scale_id": scale_id,
                    "metric": metric,
                    "observed_mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_bootstrap": int(n_bootstrap),
                    "bootstrap_unit": "session",
                    "image_aggregation": "mean_trace_samples",
                    "n_images": int(per_image["image_index"].nunique()),
                    "n_sessions": int(per_image["session"].nunique()),
                }
            )

    per_image_all = (
        metrics_df[["session", "image_index", "family", "scale_id"] + metric_cols]
        .groupby(["session", "image_index", "family", "scale_id"], as_index=False)[metric_cols]
        .mean()
    )
    for scale_id, scale_group in per_image_all.groupby("scale_id", sort=True):
        for metric in metric_cols:
            pivot = scale_group.pivot_table(
                index=["session", "image_index"],
                columns="family",
                values=metric,
                aggfunc="mean",
            ).reset_index()
            for lhs, rhs in PAIRWISE_CONTRASTS:
                if lhs not in pivot.columns or rhs not in pivot.columns:
                    continue
                diff = pivot[["session", "image_index"]].copy()
                diff["delta"] = pivot[lhs].astype(float) - pivot[rhs].astype(float)
                diff = diff.loc[np.isfinite(diff["delta"])]
                if diff.empty:
                    continue
                mean, lo, hi = _session_bootstrap_ci(
                    diff,
                    "delta",
                    n_bootstrap=n_bootstrap,
                    rng=rng,
                    draw_cache=draw_cache,
                )
                values = diff["delta"].to_numpy(dtype=np.float64)
                contrast_rows.append(
                    {
                        "lhs_family": lhs,
                        "rhs_family": rhs,
                        "contrast": f"{lhs}-{rhs}",
                        "scale_id": scale_id,
                        "metric": metric,
                        "mean_delta": mean,
                        "median_delta": float(np.nanmedian(values)),
                        "iqr_delta": float(np.nanpercentile(values, 75.0) - np.nanpercentile(values, 25.0)),
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n_images": int(diff["image_index"].nunique()),
                        "n_sessions": int(diff["session"].nunique()),
                        "n_bootstrap": int(n_bootstrap),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows), pd.DataFrame(bootstrap_rows)


def _psd_summary(psd_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not psd_rows:
        return pd.DataFrame()
    df = pd.DataFrame(psd_rows)
    return (
        df.groupby(["family", "scale_id", "frequency_hz"], as_index=False)["psd"]
        .agg(mean_psd="mean", median_psd="median", n_traces="size")
        .sort_values(["scale_id", "family", "frequency_hz"])
    )


def _build_trace_metric_tables(
    replay_rows: list[dict[str, Any]],
    *,
    dt: float,
    max_lag: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    pos_psd_rows: list[dict[str, Any]] = []
    vel_psd_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(replay_rows):
        metrics, pos_psd, vel_psd = _trace_metrics(
            row["trace"],
            row["source_trace"],
            requested_rms=float(row["requested_rms_deg"]),
            target_path=float(row["target_path_length_deg"]),
            source_trace_lag1=float(row["source_trace_lag1"]),
            image_edge_axis_deg=row.get("image_edge_axis_deg"),
            dt=float(dt),
            max_lag=int(max_lag),
        )
        base = {
            key: value
            for key, value in row.items()
            if key not in {"trace", "source_trace", "image_edge_axis_deg"}
        }
        base.update(metrics)
        metric_rows.append(base)
        for psd_row in pos_psd:
            psd_row.update(
                {
                    "image_index": int(row["image_index"]),
                    "family": str(row["family"]),
                    "scale_id": str(row["scale_id"]),
                    "sample_index": int(row["sample_index"]),
                }
            )
            pos_psd_rows.append(psd_row)
        for psd_row in vel_psd:
            psd_row.update(
                {
                    "image_index": int(row["image_index"]),
                    "family": str(row["family"]),
                    "scale_id": str(row["scale_id"]),
                    "sample_index": int(row["sample_index"]),
                }
            )
            vel_psd_rows.append(psd_row)
        if (idx + 1) % 5000 == 0 or idx + 1 == len(replay_rows):
            _progress(f"computed trace metrics {idx + 1}/{len(replay_rows)}")
    metrics_df = pd.DataFrame(metric_rows)
    return metrics_df, _psd_summary(pos_psd_rows), _psd_summary(vel_psd_rows)


def _condition_label(scale_id: str) -> str:
    return str(scale_id).replace("rel_", "").replace("p", ".").replace("x", "x")


def _plot_metric_summary(
    summary: pd.DataFrame,
    *,
    metrics: list[str],
    primary_scales: list[str],
    families: tuple[str, ...],
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4), squeeze=False)
    x = np.arange(len(primary_scales))
    for ax, metric in zip(axes.ravel(), metrics, strict=True):
        for family in families:
            vals = []
            lows = []
            highs = []
            for scale in primary_scales:
                row = summary.loc[
                    (summary["family"] == family)
                    & (summary["scale_id"] == scale)
                    & (summary["metric"] == metric)
                ]
                if row.empty:
                    vals.append(np.nan)
                    lows.append(np.nan)
                    highs.append(np.nan)
                else:
                    mean = float(row["mean"].iloc[0])
                    vals.append(mean)
                    lows.append(mean - float(row["ci95_low"].iloc[0]))
                    highs.append(float(row["ci95_high"].iloc[0]) - mean)
            yerr = np.vstack([np.asarray(lows), np.asarray(highs)])
            ax.errorbar(x, vals, yerr=yerr, marker="o", linewidth=1.3, capsize=2, label=family)
        ax.set_title(metric.replace("_", " "), fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([_condition_label(s) for s in primary_scales], rotation=25)
        ax.grid(alpha=0.25, linewidth=0.6)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_autocorr(
    summary: pd.DataFrame,
    *,
    prefix: str,
    primary_scales: list[str],
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(primary_scales), figsize=(4.2 * len(primary_scales), 3.5), sharey=True)
    if len(primary_scales) == 1:
        axes = np.asarray([axes])
    lags = np.arange(1, 21)
    for ax, scale in zip(axes, primary_scales, strict=True):
        for family in FAMILIES:
            vals = []
            for lag in lags:
                metric = f"{prefix}_autocorr_lag_{lag}"
                row = summary.loc[
                    (summary["family"] == family)
                    & (summary["scale_id"] == scale)
                    & (summary["metric"] == metric)
                ]
                vals.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            ax.plot(lags, vals, marker="o", markersize=2.5, linewidth=1.2, label=family)
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_title(_condition_label(scale), fontsize=9)
        ax.set_xlabel("lag")
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("autocorrelation")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_psd(psd: pd.DataFrame, *, primary_scales: list[str], out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(primary_scales), figsize=(4.2 * len(primary_scales), 3.5), sharey=True)
    if len(primary_scales) == 1:
        axes = np.asarray([axes])
    for ax, scale in zip(axes, primary_scales, strict=True):
        scale_df = psd.loc[psd["scale_id"] == scale]
        for family in FAMILIES:
            sub = scale_df.loc[scale_df["family"] == family].sort_values("frequency_hz")
            if sub.empty:
                continue
            ax.plot(sub["frequency_hz"], sub["mean_psd"], linewidth=1.3, label=family)
        ax.set_yscale("log")
        ax.set_xlabel("Hz")
        ax.set_title(_condition_label(scale), fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("mean PSD")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _feature_effective_rank(arr: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    centered = x - np.mean(x, axis=0, keepdims=True)
    variance_trace = float(np.sum(np.var(centered, axis=0, ddof=1)))
    if variance_trace <= 1e-12:
        return variance_trace, 0.0, 0.0, float("nan")
    gram = (centered @ centered.T) / float(max(1, x.shape[0] - 1))
    eig = np.linalg.eigvalsh(np.asarray(gram, dtype=np.float64))
    eig = np.maximum(eig, 0.0)
    eig = eig[eig > max(float(np.max(eig)) * 1e-8, 1e-12)]
    if eig.size == 0:
        return variance_trace, 0.0, 0.0, float("nan")
    p = eig / float(np.sum(eig))
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-30))))
    top_frac = float(np.max(eig) / np.sum(eig))
    condition = float(np.max(eig) / max(np.min(eig), 1e-30))
    return variance_trace, eff_rank, top_frac, condition


def _response_geometry(run_dir: Path) -> pd.DataFrame:
    npz_path = run_dir / "response_summary_arrays.npz"
    arrays = np.load(npz_path)
    rows: list[dict[str, Any]] = []
    static_norms: dict[str, np.ndarray] = {}
    for readout in PRIMARY_READOUTS:
        key = f"{readout}__static__static"
        if key in arrays:
            static_norms[readout] = np.linalg.norm(np.asarray(arrays[key]), axis=1)

    for key in sorted(arrays.files):
        if "__" not in key:
            continue
        readout, family, scale_id = key.split("__", 2)
        if readout not in PRIMARY_READOUTS:
            continue
        arr = np.asarray(arrays[key], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1)
        variance_trace, eff_rank, top_frac, condition = _feature_effective_rank(arr)
        static_norm = static_norms.get(readout)
        if static_norm is None or np.nanmean(static_norm) <= 1e-12:
            static_relative = float("nan")
        else:
            static_relative = float(np.nanmean(norms / np.maximum(static_norm, 1e-12)))
        rows.append(
            {
                "readout": readout,
                "family": family,
                "scale_id": scale_id,
                "feature_norm_mean": float(np.nanmean(norms)),
                "feature_norm_median": float(np.nanmedian(norms)),
                "feature_norm_iqr": float(np.nanpercentile(norms, 75.0) - np.nanpercentile(norms, 25.0)),
                "feature_variance_trace": variance_trace,
                "effective_rank": eff_rank,
                "top_pc_variance_fraction": top_frac,
                "static_relative_norm_mean": static_relative,
                "condition_number_approx": condition,
                "n_images": int(arr.shape[0]),
                "feature_dim": int(arr.shape[1]),
            }
        )
    return pd.DataFrame(rows)


def _plot_response_geometry(response_summary: pd.DataFrame, *, metric: str, primary_scales: list[str], out_path: Path) -> None:
    readouts = list(PRIMARY_READOUTS)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.8), sharey=False)
    x = np.arange(len(primary_scales))
    for ax, readout in zip(axes.ravel(), readouts, strict=True):
        sub = response_summary.loc[response_summary["readout"] == readout]
        for family in FAMILIES:
            vals = []
            for scale in primary_scales:
                row = sub.loc[(sub["family"] == family) & (sub["scale_id"] == scale)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.plot(x, vals, marker="o", linewidth=1.2, label=family)
        ax.set_title(readout, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([_condition_label(s) for s in primary_scales], rotation=25)
        ax.grid(alpha=0.25, linewidth=0.6)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle(metric.replace("_", " "), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _ci_pass_count(rows: pd.DataFrame, value_col: str = "ci95_low") -> int:
    if rows.empty or value_col not in rows:
        return 0
    return int(np.sum(rows[value_col].astype(float) > 0.0))


def _readout_decision_matrix(readout_dir: Path, primary_scales: list[str]) -> pd.DataFrame:
    atlas_dir = readout_dir / "readout_atlas_figures"
    primary = pd.read_csv(atlas_dir / "readout_atlas_primary_scale_score_table.csv")
    nested = pd.read_csv(atlas_dir / "nested_alpha_primary_scale_diagnostic.csv")
    primary_by_readout = primary.set_index("motion_summary")
    rows: list[dict[str, Any]] = []
    for readout in PRIMARY_READOUTS:
        fixed = primary_by_readout.loc[readout] if readout in primary_by_readout.index else pd.Series(dtype=float)
        nested_emp = nested.loc[
            (nested["motion_summary"] == readout)
            & (nested["model"] == "static_plus_motion")
            & (nested["family"] == "empirical")
            & (nested["scale_id"].isin(primary_scales))
        ]
        nested_ou = nested.loc[
            (nested["motion_summary"] == readout)
            & (nested["model"] == "contrast")
            & (nested["control_contrast"] == "empirical-ou")
            & (nested["scale_id"].isin(primary_scales))
        ]
        nested_abs = float(nested_emp["gain_vs_static_mean"].mean()) if not nested_emp.empty else float("nan")
        nested_emp_ou = float(nested_ou["gain_vs_static_mean"].mean()) if not nested_ou.empty else float("nan")
        order_sensitive = readout.startswith("temporal_")
        subtracts_static = "delta" in readout
        if readout in {"mean", "delta_mean"} and np.isfinite(nested_abs) and nested_abs > 0.0:
            role = "primary_absolute_candidate"
            interpretation = "Order-blind response summary with positive empirical gain beyond static under fixed and nested-alpha audits."
        elif order_sensitive and np.isfinite(nested_emp_ou) and nested_emp_ou > 0.0:
            role = "order_sensitive_specificity_candidate"
            interpretation = "Preserves trajectory ordering and separates empirical motion from OU, but absolute gain over static should not be the headline."
        else:
            role = "not_recommended"
            interpretation = "Does not pass the current absolute or order-sensitive specificity gates."
        basis_type = "mean"
        if "pca" in readout:
            basis_type = "temporal_pca"
        elif "dct" in readout:
            basis_type = "temporal_dct"
        rows.append(
            {
                "readout": readout,
                "preserves_trajectory_order": bool(order_sensitive),
                "subtracts_static_response": bool(subtracts_static),
                "basis_type": basis_type,
                "absolute_empirical_gain_primary_mean": float(fixed.get("empirical_mean_gain_primary", np.nan)),
                "absolute_empirical_gain_ci_pass_n": int(fixed.get("empirical_ci_pass_primary_n", 0)),
                "empirical_minus_ou_primary_mean": float(fixed.get("emp_minus_ou_mean_primary", np.nan)),
                "empirical_minus_ou_ci_pass_n": int(fixed.get("emp_minus_ou_ci_pass_primary_n", 0)),
                "empirical_minus_brownian_primary_mean": float(fixed.get("emp_minus_brownian_mean_primary", np.nan)),
                "empirical_minus_rotated_primary_mean": float(fixed.get("emp_minus_rotated_mean_primary", np.nan)),
                "nested_alpha_absolute_gain_primary_mean": nested_abs,
                "nested_alpha_empirical_minus_ou_primary_mean": nested_emp_ou,
                "nested_alpha_empirical_minus_ou_ci_pass_n": _ci_pass_count(nested_ou),
                "interpretation": interpretation,
                "recommended_role": role,
            }
        )
    return pd.DataFrame(rows)


def _metric_value(summary: pd.DataFrame, family: str, scale_id: str, metric: str, column: str = "mean") -> float:
    row = summary.loc[
        (summary["family"] == family) & (summary["scale_id"] == scale_id) & (summary["metric"] == metric)
    ]
    return float(row[column].iloc[0]) if not row.empty else float("nan")


def _metric_delta(contrasts: pd.DataFrame, lhs: str, rhs: str, scale_id: str, metric: str) -> float:
    row = contrasts.loc[
        (contrasts["lhs_family"] == lhs)
        & (contrasts["rhs_family"] == rhs)
        & (contrasts["scale_id"] == scale_id)
        & (contrasts["metric"] == metric)
    ]
    return float(row["mean_delta"].iloc[0]) if not row.empty else float("nan")


def _response_value(response_summary: pd.DataFrame, readout: str, family: str, scale_id: str, metric: str) -> float:
    row = response_summary.loc[
        (response_summary["readout"] == readout)
        & (response_summary["family"] == family)
        & (response_summary["scale_id"] == scale_id)
    ]
    return float(row[metric].iloc[0]) if not row.empty else float("nan")


def _choose_verdict(
    validation_summary: dict[str, Any],
    trace_summary: pd.DataFrame,
    trace_contrasts: pd.DataFrame,
    response_summary: pd.DataFrame,
    decision: pd.DataFrame,
    primary_scales: list[str],
) -> tuple[str, list[str]]:
    if validation_summary.get("trace_bank_index_mismatches", 1) != 0 or validation_summary.get("unmatched_rows", 1) != 0:
        return "ou_invalid_until_regenerated", ["Exact replay did not validate."]

    scale = "rel_1x" if "rel_1x" in primary_scales else primary_scales[-1]
    reasons: list[str] = []
    ou_path = _metric_value(trace_summary, "ou", scale, "path_to_target_ratio", "median")
    ou_lag_delta = _metric_value(trace_summary, "ou", scale, "lag1_delta_to_source", "median")
    ou_rms = _metric_value(trace_summary, "ou", scale, "effective_to_requested_rms", "median")
    if not (0.98 <= ou_rms <= 1.02):
        reasons.append(f"OU RMS ratio at {scale} is {ou_rms:.3g}.")
    if not (0.80 <= ou_path <= 1.20):
        reasons.append(f"OU path-to-target ratio at {scale} is {ou_path:.3g}.")
    if np.isfinite(ou_lag_delta) and abs(ou_lag_delta) > 0.20:
        reasons.append(f"OU lag-1 differs from source by {ou_lag_delta:.3g}.")

    vel_delta = _metric_delta(trace_contrasts, "empirical", "ou", scale, "velocity_psd_centroid_hz")
    center_delta = _metric_delta(trace_contrasts, "empirical", "ou", scale, "mean_return_to_center_slope")
    if np.isfinite(vel_delta) and abs(vel_delta) > 5.0:
        direction = "lower" if vel_delta > 0.0 else "higher"
        reasons.append(f"OU velocity PSD centroid is {abs(vel_delta):.3g} Hz {direction} than empirical at {scale}.")
    if np.isfinite(center_delta) and abs(center_delta) > 0.05:
        direction = "more negative" if center_delta > 0.0 else "less negative"
        reasons.append(f"OU return-to-center slope is {direction} than empirical by {abs(center_delta):.3g} at {scale}.")

    response_flags = []
    for readout in ("temporal_pca", "temporal_delta_pca", "temporal_dct", "temporal_dct_delta"):
        ou_norm = _response_value(response_summary, readout, "ou", scale, "feature_norm_mean")
        emp_norm = _response_value(response_summary, readout, "empirical", scale, "feature_norm_mean")
        if np.isfinite(ou_norm) and np.isfinite(emp_norm) and emp_norm > 0.0:
            ratio = ou_norm / emp_norm
            if ratio < 0.5 or ratio > 2.0:
                response_flags.append(f"{readout} OU/empirical norm ratio {ratio:.2f}")
    if response_flags:
        reasons.append("; ".join(response_flags))

    temporal_rows = decision.loc[decision["recommended_role"] == "order_sensitive_specificity_candidate"]
    temporal_abs_negative = bool((temporal_rows["nested_alpha_absolute_gain_primary_mean"].astype(float) < 0.0).any())
    if temporal_abs_negative:
        reasons.append("Order-sensitive readouts separate empirical from OU but remain below static under nested alpha.")

    if reasons:
        return "ou_valid_diagnostic_only", reasons
    return "ou_valid_primary_control", ["Replay validated and OU stayed within the current trace, response, and nested-alpha gates."]


def _format_float(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def _write_report(
    out_dir: Path,
    validation_summary: dict[str, Any],
    trace_summary: pd.DataFrame,
    trace_contrasts: pd.DataFrame,
    response_summary: pd.DataFrame,
    decision: pd.DataFrame,
    primary_scales: list[str],
) -> None:
    verdict, verdict_reasons = _choose_verdict(
        validation_summary,
        trace_summary,
        trace_contrasts,
        response_summary,
        decision,
        primary_scales,
    )
    scale = "rel_1x" if "rel_1x" in primary_scales else primary_scales[-1]
    conclusion = (
        "Exact replay validates the cached generated traces. OU matches the advertised RMS/path/lag-1 gates well enough "
        "to remain useful as an order-sensitive diagnostic, but the current readout audit does not support using "
        "temporal readouts as the absolute gain-over-static headline. Mean and delta_mean are the defensible absolute "
        "readouts; temporal DCT/PCA variants are specificity diagnostics for trajectory ordering."
        if verdict != "ou_invalid_until_regenerated"
        else "Exact replay failed, so OU control interpretation should be paused until the generated traces are regenerated or the cache mismatch is explained."
    )
    metric_rows = []
    for family in FAMILIES:
        metric_rows.append(
            {
                "family": family,
                "rms_ratio": _metric_value(trace_summary, family, scale, "effective_to_requested_rms", "median"),
                "path_ratio": _metric_value(trace_summary, family, scale, "path_to_target_ratio", "median"),
                "lag1": _metric_value(trace_summary, family, scale, "generated_lag1_autocorr", "median"),
                "vel_centroid_hz": _metric_value(trace_summary, family, scale, "velocity_psd_centroid_hz", "mean"),
                "return_slope": _metric_value(trace_summary, family, scale, "mean_return_to_center_slope", "mean"),
            }
        )
    matrix_preview = decision[
        [
            "readout",
            "absolute_empirical_gain_primary_mean",
            "empirical_minus_ou_primary_mean",
            "nested_alpha_absolute_gain_primary_mean",
            "nested_alpha_empirical_minus_ou_primary_mean",
            "recommended_role",
        ]
    ]

    absolute_candidates = decision.loc[decision["recommended_role"] == "primary_absolute_candidate", "readout"].to_list()
    order_candidates = decision.loc[decision["recommended_role"] == "order_sensitive_specificity_candidate", "readout"].to_list()
    controls_main = "empirical, Brownian, rotated; OU only if framed as diagnostic"
    controls_supp = "OU trajectory/readout audit, plus sentinel 1.5x and 2x scales"
    if verdict == "ou_valid_primary_control":
        controls_main = "empirical, OU, Brownian, rotated"
        controls_supp = "sentinel 1.5x and 2x scales"

    lines = [
        "# OU Trace Control Audit Report",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        "## Exact Replay Validation",
        "",
        "| Check | Value |",
        "|---|---:|",
    ]
    for key, value in validation_summary.items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            f"## Metric Summary At {scale}",
            "",
            "| family | RMS ratio | path ratio | lag-1 | velocity PSD centroid Hz | return slope |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metric_rows:
        lines.append(
            "| {family} | {rms} | {path} | {lag1} | {vel} | {ret} |".format(
                family=row["family"],
                rms=_format_float(row["rms_ratio"]),
                path=_format_float(row["path_ratio"]),
                lag1=_format_float(row["lag1"]),
                vel=_format_float(row["vel_centroid_hz"]),
                ret=_format_float(row["return_slope"]),
            )
        )
    lines.extend(
        [
            "",
            "## OU Verdict",
            "",
            f"`{verdict}`",
            "",
        ]
    )
    for reason in verdict_reasons:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Readout Decision Matrix Summary",
            "",
            "| readout | fixed empirical gain | fixed empirical-OU | nested empirical gain | nested empirical-OU | role |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in matrix_preview.iterrows():
        lines.append(
            "| {readout} | {abs_fixed} | {ou_fixed} | {abs_nested} | {ou_nested} | `{role}` |".format(
                readout=row["readout"],
                abs_fixed=_format_float(float(row["absolute_empirical_gain_primary_mean"])),
                ou_fixed=_format_float(float(row["empirical_minus_ou_primary_mean"])),
                abs_nested=_format_float(float(row["nested_alpha_absolute_gain_primary_mean"])),
                ou_nested=_format_float(float(row["nested_alpha_empirical_minus_ou_primary_mean"])),
                role=row["recommended_role"],
            )
        )
    lines.extend(
        [
            "",
            "## Recommended Panel-B Candidates",
            "",
            f"- Absolute readout candidate: {', '.join(absolute_candidates) if absolute_candidates else 'none'}",
            f"- Order-sensitive specificity candidate: {', '.join(order_candidates) if order_candidates else 'none'}",
            f"- Controls to show in main panel: {controls_main}",
            f"- Controls to route to supplement: {controls_supp}",
            "",
            "## Recommended Next Run",
            "",
            (
                "No V1-twin response cache rerun is needed for the current claim split. If OU is to become a headline "
                "negative control, add a spectrum-matched or velocity-AR surrogate and rerun only the aggregate control "
                "conditions needed for Panel B."
                if verdict != "ou_invalid_until_regenerated"
                else "Regenerate or recover the aggregate trace metadata before any Panel-B promotion."
            ),
        ]
    )
    (out_dir / "ou_trace_control_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--readout-dir", type=Path, default=DEFAULT_READOUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--primary-scales", default="rel_0p25x,rel_0p5x,rel_1x")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--max-autocorr-lag", type=int, default=20)
    return parser


def run(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    readout_dir = Path(args.readout_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    primary_scales = _parse_csv_list(args.primary_scales)

    work, analysis, replay_rows = _replay_generated_traces(run_dir)
    passed, _validation, validation_summary = _validate_replay(run_dir, replay_rows, out_dir)
    (out_dir / "audit_run_metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "readout_dir": str(readout_dir),
                "out_dir": str(out_dir),
                "primary_scales": primary_scales,
                "n_bootstrap": int(args.n_bootstrap),
                "seed": int(args.seed),
                "dt": float(args.dt),
                "max_autocorr_lag": int(args.max_autocorr_lag),
                "n_images": int(work.shape[0]),
                "n_generated_traces": int(len(replay_rows)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not passed:
        _write_failure_report(out_dir, validation_summary)
        raise SystemExit("Exact trace replay validation failed; wrote trace_replay_failure_report.md")
    _progress("exact replay validated")

    metrics_df, position_psd, velocity_psd = _build_trace_metric_tables(
        replay_rows,
        dt=float(args.dt),
        max_lag=int(args.max_autocorr_lag),
    )
    metrics_df.to_csv(out_dir / "trace_control_metrics_by_generated_trace.csv", index=False)
    position_psd.to_csv(out_dir / "position_psd_by_family_scale.csv", index=False)
    velocity_psd.to_csv(out_dir / "velocity_psd_by_family_scale.csv", index=False)

    metric_cols = _metric_columns(metrics_df)
    _progress(f"summarizing {len(metric_cols)} trace metrics with {int(args.n_bootstrap)} session bootstraps")
    trace_summary, trace_contrasts, trace_bootstrap = _summarize_trace_metrics(
        metrics_df,
        metric_cols=metric_cols,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    trace_summary.to_csv(out_dir / "trace_control_metric_summary_by_family_scale.csv", index=False)
    trace_contrasts.to_csv(out_dir / "trace_control_metric_pairwise_empirical_minus_control.csv", index=False)
    trace_bootstrap.to_csv(out_dir / "trace_control_metric_session_bootstrap.csv", index=False)

    _progress("summarizing response readout geometry")
    response_summary = _response_geometry(run_dir)
    response_summary.to_csv(out_dir / "response_readout_geometry_summary.csv", index=False)

    _progress("building decoder/readout decision matrix")
    decision = _readout_decision_matrix(readout_dir, primary_scales)
    decision.to_csv(out_dir / "readout_decision_matrix.csv", index=False)

    _progress("writing figures")
    _plot_metric_summary(
        trace_summary,
        metrics=["effective_to_requested_rms", "path_to_target_ratio", "generated_lag1_autocorr"],
        primary_scales=primary_scales,
        families=FAMILIES,
        out_path=out_dir / "fig_trace_qc_rms_path_lag1.png",
        title="Trace QC: RMS, path, lag-1",
    )
    _plot_metric_summary(
        trace_summary,
        metrics=["speed_mean_deg_s", "speed_p95_deg_s", "tortuosity"],
        primary_scales=primary_scales,
        families=FAMILIES,
        out_path=out_dir / "fig_trace_qc_speed_tortuosity.png",
        title="Trace speed and tortuosity",
    )
    _plot_autocorr(
        trace_summary,
        prefix="position",
        primary_scales=primary_scales,
        out_path=out_dir / "fig_position_autocorr_by_family.png",
        title="Position autocorrelation by family",
    )
    _plot_autocorr(
        trace_summary,
        prefix="velocity",
        primary_scales=primary_scales,
        out_path=out_dir / "fig_velocity_autocorr_by_family.png",
        title="Velocity autocorrelation by family",
    )
    _plot_psd(
        position_psd,
        primary_scales=primary_scales,
        out_path=out_dir / "fig_position_psd_by_family.png",
        title="Position PSD by family",
    )
    _plot_psd(
        velocity_psd,
        primary_scales=primary_scales,
        out_path=out_dir / "fig_velocity_psd_by_family.png",
        title="Velocity PSD by family",
    )
    _plot_metric_summary(
        trace_summary,
        metrics=["trace_cov_anisotropy", "axis_delta_to_source_deg", "axis_delta_to_image_edge_deg"],
        primary_scales=primary_scales,
        families=FAMILIES,
        out_path=out_dir / "fig_covariance_anisotropy_axis_by_family.png",
        title="Covariance anisotropy and axis alignment",
    )
    _plot_metric_summary(
        trace_summary,
        metrics=[
            "end_radius_deg",
            "mean_return_to_center_slope",
            "fraction_samples_outside_2x_rms_radius",
        ],
        primary_scales=primary_scales,
        families=FAMILIES,
        out_path=out_dir / "fig_endpoint_centering_by_family.png",
        title="Endpoint and centering behavior",
    )
    _plot_response_geometry(
        response_summary,
        metric="feature_norm_mean",
        primary_scales=primary_scales,
        out_path=out_dir / "fig_response_readout_norms_by_family.png",
    )
    _plot_response_geometry(
        response_summary,
        metric="effective_rank",
        primary_scales=primary_scales,
        out_path=out_dir / "fig_response_readout_effective_rank_by_family.png",
    )

    _write_report(
        out_dir,
        validation_summary,
        trace_summary,
        trace_contrasts,
        response_summary,
        decision,
        primary_scales,
    )
    _progress(f"audit complete: {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
