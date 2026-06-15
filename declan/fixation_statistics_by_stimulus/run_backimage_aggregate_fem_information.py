"""Aggregate BackImage FEM information pilot with empirical trajectory controls.

This runner is deliberately narrower than the fixed-axis local screens.  It
asks whether an ensemble of natural-image patches is better represented under
empirical FEM-like motion distributions than under matched synthetic controls.
The primary controls are OU trajectories matched to empirical RMS/autocorrelation,
Brownian trajectories matched to RMS, and rotated empirical trajectories.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

try:
    from .extraction import _as_numpy, _load_dict_dataset
    from .image_features import _backimage_canvas, gaze_deg_to_screen_px
    from .run_backimage_latent_information_screen import (
        CanonicalTwinScorer,
        HAVE_STEERABLE_PYRAMID,
        _align_response_to_trace,
        _central_crop,
        _clip_patch,
        _cross_validated_decode,
        _dct_features,
        _extract_latents,
        _gabor_features,
        _parse_float_list,
        _parse_int_list,
        _parse_str_list,
        _pyramid_features,
        _scale_token,
        _static_trace,
        _standardize_uint_like,
        _trace_rms,
        _write_json,
    )
    from .run_fixation_statistics_by_stimulus import load_sessions
    from jake.twininfo.eye_controls import detect_microsaccade_events, speed_threshold_mad
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.fixation_statistics_by_stimulus.extraction import _as_numpy, _load_dict_dataset
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
    from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
        CanonicalTwinScorer,
        HAVE_STEERABLE_PYRAMID,
        _align_response_to_trace,
        _central_crop,
        _clip_patch,
        _cross_validated_decode,
        _dct_features,
        _extract_latents,
        _gabor_features,
        _parse_float_list,
        _parse_int_list,
        _parse_str_list,
        _pyramid_features,
        _scale_token,
        _static_trace,
        _standardize_uint_like,
        _trace_rms,
        _write_json,
    )
    from declan.fixation_statistics_by_stimulus.run_fixation_statistics_by_stimulus import load_sessions
    from jake.twininfo.eye_controls import detect_microsaccade_events, speed_threshold_mad


DEFAULT_INPUT = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
DEFAULT_OUT_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_aggregate_fem_information_pilot"
)


@dataclass(frozen=True)
class AggregateConfig:
    input: str
    out_dir: str
    window_manifest: str | None
    max_images: int
    trace_samples_per_condition: int
    motion_families: list[str]
    observed_rms_scales: list[float]
    patch_size_px: int
    latent_crop_px: int
    center_crop_px: int
    local_field_grid: int
    n_timepoints: int
    temporal_pc_components: int
    pca_k_list: list[int]
    latent_names: list[str]
    ridge_alphas: list[float]
    fixed_ridge_alpha: float | None
    outer_folds: int
    inner_folds: int
    decode_group_mode: str
    reliable_image_coherence_min: float
    reliable_drift_anisotropy_min: float
    min_duration_s: float
    min_patch_image_margin_px: float
    max_rms_deg: float
    max_trace_source_rms_deg: float | None
    max_trace_source_radius_deg: float | None
    max_trace_source_path_length_deg: float | None
    max_trace_source_speed_p95_deg_s: float | None
    max_trace_source_microsaccade_events: int | None
    microsaccade_speed_threshold_dps: float | None
    microsaccade_threshold_z: float
    microsaccade_pad_frames: int
    reuse_trace_sources_across_scales: bool
    twin_batch_size: int
    twin_trace_batch_size: int
    device: str
    progress_every: int
    seed: int
    dry_run: bool


def _progress(message: str) -> None:
    print(f"[backimage-aggregate-fem] {message}", flush=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_windows(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    df["source_row"] = np.arange(df.shape[0], dtype=int)
    required = [
        "session",
        "trial_idx",
        "global_start",
        "global_stop",
        "mean_x_deg",
        "mean_y_deg",
        "anisotropy",
        "image_orientation_coherence",
        "image_patch_distance_to_image_border_px",
    ]
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "duration_s" not in df.columns:
        df["duration_s"] = df.get("epoch_duration_s", np.nan)
    margin = float(args.min_patch_image_margin_px) if args.min_patch_image_margin_px is not None else float(args.patch_size_px) / 2.0
    keep = (
        np.isfinite(df["mean_x_deg"].astype(float))
        & np.isfinite(df["mean_y_deg"].astype(float))
        & (df["anisotropy"].astype(float) >= float(args.reliable_drift_anisotropy_min))
        & (df["image_orientation_coherence"].astype(float) >= float(args.reliable_image_coherence_min))
        & (df["duration_s"].astype(float) >= float(args.min_duration_s))
        & (df["image_patch_distance_to_image_border_px"].astype(float) >= margin)
    )
    work = df.loc[keep].copy()
    if args.window_manifest is not None:
        manifest = pd.read_csv(args.window_manifest)
        if "source_row" not in manifest.columns:
            raise ValueError("--window-manifest must contain source_row for this aggregate runner")
        requested = manifest["source_row"].astype(int).drop_duplicates().to_list()
        available = set(work["source_row"].astype(int).to_list())
        missing_ids = sorted(set(requested).difference(available))
        if missing_ids:
            preview = ", ".join(str(v) for v in missing_ids[:10])
            suffix = "..." if len(missing_ids) > 10 else ""
            raise ValueError(f"--window-manifest source_row values do not survive filters: {preview}{suffix}")
        work = work.set_index("source_row", drop=False).loc[requested].reset_index(drop=True)
    elif int(args.max_images) > 0 and work.shape[0] > int(args.max_images):
        work = work.sample(n=int(args.max_images), replace=False, random_state=int(args.seed))
        work = work.sort_values(["session", "trial_idx", "source_row"])
    work["image_index"] = np.arange(work.shape[0], dtype=int)
    return work.reset_index(drop=True)


def _session_dataset_cache(sessions: list[str]) -> dict[str, np.ndarray]:
    session_objects = {str(getattr(s, "name", s)): s for s in load_sessions(",".join(sorted(set(sessions))))}
    out: dict[str, np.ndarray] = {}
    for name in sorted(set(sessions)):
        session = session_objects[name]
        dset_path = Path(session.sess_dir) / "datasets" / "backimage.dset"
        dset = _load_dict_dataset(dset_path)
        out[name] = _as_numpy(dset["eyepos"]).astype(np.float64)
    return out


def _extract_requested_latents(
    patch: np.ndarray,
    *,
    latent_crop_px: int,
    center_crop_px: int,
    local_field_grid: int,
    requested: set[str],
) -> dict[str, np.ndarray]:
    if not requested:
        return _extract_latents(
            patch,
            latent_crop_px=int(latent_crop_px),
            center_crop_px=int(center_crop_px),
            local_field_grid=int(local_field_grid),
        )
    image = _standardize_uint_like(patch)
    out: dict[str, np.ndarray] = {}
    need_field = any(name.endswith("_local_field") for name in requested)
    need_center = any(name.endswith("_center") for name in requested)
    field_crop = _central_crop(image, int(latent_crop_px)) if need_field else None
    center_crop = _central_crop(image, int(center_crop_px)) if need_center else None
    if "dct_center" in requested and center_crop is not None:
        out["dct_center"] = _dct_features(center_crop, n_freq=8)
    if "dct_local_field" in requested and field_crop is not None:
        out["dct_local_field"] = _dct_features(field_crop, n_freq=8)
    if "gabor_center" in requested and center_crop is not None:
        out["gabor_center"] = _gabor_features(center_crop, scope="center", local_grid=int(local_field_grid))
    if "gabor_local_field" in requested and field_crop is not None:
        out["gabor_local_field"] = _gabor_features(field_crop, scope="local_field", local_grid=int(local_field_grid))
    if HAVE_STEERABLE_PYRAMID and "pyramid_center" in requested and center_crop is not None:
        out["pyramid_center"] = _pyramid_features(center_crop, scope="center", local_grid=int(local_field_grid))
    if HAVE_STEERABLE_PYRAMID and "pyramid_local_field" in requested and field_crop is not None:
        out["pyramid_local_field"] = _pyramid_features(field_crop, scope="local_field", local_grid=int(local_field_grid))
    return {key: value for key, value in out.items() if value.size > 0}


def _resample_trace(trace: np.ndarray, n_timepoints: int) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    if trace.shape[0] < 2:
        return np.zeros((int(n_timepoints), 2), dtype=np.float32)
    idx = np.linspace(0, trace.shape[0] - 1, int(n_timepoints))
    lo = np.floor(idx).astype(int)
    hi = np.ceil(idx).astype(int)
    frac = idx - lo
    out = trace[lo] * (1.0 - frac[:, None]) + trace[hi] * frac[:, None]
    finite = np.isfinite(out).all(axis=1)
    if not np.all(finite):
        good = np.flatnonzero(finite)
        if good.size == 0:
            out = np.zeros_like(out)
        else:
            bad = np.flatnonzero(~finite)
            for dim in range(2):
                out[bad, dim] = np.interp(bad, good, out[good, dim])
    out -= np.mean(out, axis=0, keepdims=True)
    return out.astype(np.float32)


def _scale_to_rms(trace: np.ndarray, target_rms: float, *, max_rms_deg: float) -> tuple[np.ndarray, dict[str, Any]]:
    trace = np.asarray(trace, dtype=np.float64)
    base_rms = _trace_rms(trace)
    requested = float(target_rms)
    effective_target = min(max(requested, 0.0), float(max_rms_deg))
    clipped_high = bool(requested > float(max_rms_deg))
    if base_rms <= 1e-12 or effective_target <= 0.0:
        scaled = np.zeros_like(trace)
    else:
        scaled = trace * (effective_target / base_rms)
        scaled -= np.mean(scaled, axis=0, keepdims=True)
    return scaled.astype(np.float32), {
        "base_rms_deg": float(base_rms),
        "requested_rms_deg": float(requested),
        "effective_rms_deg": float(_trace_rms(scaled)),
        "rms_clipped_high": clipped_high,
    }


def _path_length(trace: np.ndarray) -> float:
    trace = np.asarray(trace, dtype=np.float64)
    if trace.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(trace, axis=0), axis=1)))


def _speed_summary(trace: np.ndarray, dt: float) -> dict[str, float]:
    trace = np.asarray(trace, dtype=np.float64)
    if trace.shape[0] < 2:
        return {"speed_mean_deg_s": 0.0, "speed_median_deg_s": 0.0, "speed_p95_deg_s": 0.0}
    speed = np.linalg.norm(np.diff(trace, axis=0), axis=1) / float(dt)
    return {
        "speed_mean_deg_s": float(np.nanmean(speed)),
        "speed_median_deg_s": float(np.nanmedian(speed)),
        "speed_p95_deg_s": float(np.nanpercentile(speed, 95.0)),
    }


def _microsaccade_stats(
    trace: np.ndarray,
    *,
    dt: float,
    threshold_dps: float | None,
    threshold_z: float,
    pad_frames: int,
) -> dict[str, float | int]:
    threshold = (
        float(threshold_dps)
        if threshold_dps is not None
        else speed_threshold_mad(np.asarray(trace, dtype=np.float64), dt=float(dt), z=float(threshold_z))
    )
    events, sample_mask, _threshold = detect_microsaccade_events(
        np.asarray(trace, dtype=np.float64),
        dt=float(dt),
        threshold_deg_s=threshold,
        min_samples=1,
        pad_samples=max(0, int(pad_frames)),
    )
    peak = max((float(event["peak_speed_deg_s"]) for event in events), default=0.0)
    return {
        "microsaccade_threshold_dps": float(threshold),
        "n_microsaccade_events": int(len(events)),
        "fraction_microsaccade_samples": float(np.mean(sample_mask)) if sample_mask.size else 0.0,
        "peak_microsaccade_speed_dps": float(peak),
    }


def _trace_covariance_shape(trace: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    cov = np.cov(trace, rowvar=False) if trace.shape[0] > 1 else np.eye(2)
    if not np.all(np.isfinite(cov)):
        cov = np.eye(2)
    vals, vecs = np.linalg.eigh(cov + 1e-9 * np.eye(2))
    vals = np.maximum(vals, 1e-9)
    shape = vecs @ np.diag(np.sqrt(vals / np.mean(vals))) @ vecs.T
    return shape.astype(np.float64)


def _lag1_autocorr(trace: np.ndarray) -> float:
    x = np.asarray(trace, dtype=np.float64)
    if x.shape[0] < 3:
        return 0.0
    vals = []
    for dim in range(2):
        a = x[:-1, dim] - np.mean(x[:-1, dim])
        b = x[1:, dim] - np.mean(x[1:, dim])
        den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        if den > 1e-12:
            vals.append(float(np.sum(a * b) / den))
    if not vals:
        return 0.0
    return float(np.clip(np.mean(vals), -0.95, 0.98))


def _brownian_trace(n_timepoints: int, rng: np.random.Generator) -> np.ndarray:
    inc = rng.normal(size=(int(n_timepoints), 2))
    trace = np.cumsum(inc, axis=0)
    trace -= np.mean(trace, axis=0, keepdims=True)
    return trace.astype(np.float32)


def _ou_trace(n_timepoints: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    rho = float(np.clip(rho, -0.95, 0.98))
    x = np.zeros((int(n_timepoints), 2), dtype=np.float64)
    sigma = float(np.sqrt(max(1e-6, 1.0 - rho * rho)))
    x[0] = rng.normal(size=2)
    for t in range(1, int(n_timepoints)):
        x[t] = rho * x[t - 1] + sigma * rng.normal(size=2)
    x -= np.mean(x, axis=0, keepdims=True)
    return x.astype(np.float32)


def _rotated_trace(trace: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    rot = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
    out = np.asarray(trace, dtype=np.float64) @ rot.T
    out -= np.mean(out, axis=0, keepdims=True)
    return out.astype(np.float32)


def _build_trace_bank(
    work: pd.DataFrame,
    eyepos_by_session: dict[str, np.ndarray],
    n_timepoints: int,
    *,
    microsaccade_speed_threshold_dps: float | None,
    microsaccade_threshold_z: float,
    microsaccade_pad_frames: int,
) -> list[dict[str, Any]]:
    bank: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        eyepos = eyepos_by_session[str(row["session"])]
        start = int(row["global_start"])
        stop = int(row["global_stop"])
        trace = _resample_trace(eyepos[start:stop], int(n_timepoints))
        ms = _microsaccade_stats(
            trace,
            dt=1.0 / 120.0,
            threshold_dps=microsaccade_speed_threshold_dps,
            threshold_z=float(microsaccade_threshold_z),
            pad_frames=int(microsaccade_pad_frames),
        )
        bank.append(
            {
                "source_row": int(row["source_row"]),
                "session": str(row["session"]),
                "trace": trace,
                "observed_rms_deg": float(_trace_rms(trace)),
                "source_rms_radius_deg": float(row.get("rms_radius_deg", np.nan)),
                "source_max_radius_deg": float(row.get("max_radius_deg", np.nan)),
                "path_length_deg": _path_length(trace),
                "source_path_length_deg": float(row.get("path_length_deg", np.nan)),
                "source_speed_p95_deg_s": float(row.get("speed_p95_deg_s", np.nan)),
                "duration_s": float(row.get("duration_s", np.nan)),
                "lag1_autocorr": _lag1_autocorr(trace),
                "covariance_shape": _trace_covariance_shape(trace),
                **ms,
            }
        )
    return bank


def _eligible_trace_bank_indices(
    trace_bank: list[dict[str, Any]],
    *,
    current_source_row: int,
    max_trace_source_rms_deg: float | None,
    max_trace_source_radius_deg: float | None,
    max_trace_source_path_length_deg: float | None,
    max_trace_source_speed_p95_deg_s: float | None,
    max_trace_source_microsaccade_events: int | None,
) -> list[int]:
    eligible = []
    for j, item in enumerate(trace_bank):
        if int(item["source_row"]) == int(current_source_row):
            continue
        if max_trace_source_rms_deg is not None and float(item["observed_rms_deg"]) > float(max_trace_source_rms_deg):
            continue
        if max_trace_source_radius_deg is not None and float(item["source_max_radius_deg"]) > float(max_trace_source_radius_deg):
            continue
        if max_trace_source_path_length_deg is not None and float(item["path_length_deg"]) > float(max_trace_source_path_length_deg):
            continue
        if max_trace_source_speed_p95_deg_s is not None and float(item["source_speed_p95_deg_s"]) > float(max_trace_source_speed_p95_deg_s):
            continue
        if (
            max_trace_source_microsaccade_events is not None
            and int(item["n_microsaccade_events"]) > int(max_trace_source_microsaccade_events)
        ):
            continue
        eligible.append(j)
    return eligible


def _family_trace(
    family: str,
    source_trace: np.ndarray,
    source_rho: float,
    target_rms: float,
    *,
    rng: np.random.Generator,
    max_rms_deg: float,
    source_shape: np.ndarray | None = None,
    target_path_length: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if family == "empirical":
        raw = np.asarray(source_trace, dtype=np.float32)
    elif family == "rotated":
        raw = _rotated_trace(source_trace, rng)
    elif family == "brownian":
        raw = _brownian_trace(source_trace.shape[0], rng)
    elif family == "ou":
        best_raw = None
        best_loss = float("inf")
        shape = np.eye(2) if source_shape is None else np.asarray(source_shape, dtype=np.float64)
        for _ in range(12):
            candidate = np.asarray(_ou_trace(source_trace.shape[0], source_rho, rng), dtype=np.float64) @ shape.T
            candidate -= np.mean(candidate, axis=0, keepdims=True)
            if target_path_length is None or not np.isfinite(target_path_length):
                loss = 0.0
            else:
                cand_scaled, _ = _scale_to_rms(candidate, target_rms, max_rms_deg=max_rms_deg)
                loss = abs(_path_length(cand_scaled) - float(target_path_length))
            if loss < best_loss:
                best_loss = float(loss)
                best_raw = candidate
        raw = np.asarray(best_raw, dtype=np.float32)
    else:
        raise ValueError(f"Unknown motion family {family!r}")
    trace, meta = _scale_to_rms(raw, target_rms, max_rms_deg=max_rms_deg)
    meta["generated_lag1_autocorr"] = _lag1_autocorr(trace)
    meta["path_length_deg"] = _path_length(trace)
    meta.update(_speed_summary(trace, dt=1.0 / 120.0))
    return trace, meta


def _fit_temporal_basis(responses: list[np.ndarray], n_components: int) -> np.ndarray:
    if not responses:
        raise ValueError("No responses available for temporal basis")
    T = int(responses[0].shape[0])
    cov = np.zeros((T, T), dtype=np.float64)
    count = 0
    for resp in responses:
        arr = np.asarray(resp, dtype=np.float64)
        if arr.shape[0] != T:
            raise ValueError("All responses must have the same time length for temporal PCA")
        arr = arr - np.mean(arr, axis=0, keepdims=True)
        cov += arr @ arr.T
        count += int(arr.shape[1])
    cov /= max(1, count)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    basis = vecs[:, order[: int(min(n_components, T))]]
    for j in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, j])))
        if basis[pivot, j] < 0:
            basis[:, j] *= -1.0
    return basis.astype(np.float32)


def _fixed_dct_basis(n_timepoints: int, n_components: int) -> np.ndarray:
    t = np.arange(int(n_timepoints), dtype=np.float64)
    basis = []
    for k in range(1, int(n_components) + 1):
        vec = np.cos(np.pi * (t + 0.5) * float(k) / float(n_timepoints))
        vec = vec - np.mean(vec)
        vec = vec / (np.sqrt(np.sum(vec * vec)) + 1e-12)
        basis.append(vec)
    return np.column_stack(basis).astype(np.float32)


def _summarize_response(response: np.ndarray, static: np.ndarray, basis: np.ndarray) -> dict[str, np.ndarray]:
    response = np.asarray(response, dtype=np.float32)
    static = np.asarray(static, dtype=np.float32)
    delta = response - static
    return {
        "temporal_pca": (basis.T @ response).reshape(-1).astype(np.float32),
        "temporal_delta_pca": (basis.T @ delta).reshape(-1).astype(np.float32),
        "mean": np.mean(response, axis=0).astype(np.float32),
        "delta_mean": np.mean(delta, axis=0).astype(np.float32),
    }


def _add_temporal_basis_summaries(
    out: dict[str, np.ndarray],
    response: np.ndarray,
    static: np.ndarray,
    basis: np.ndarray,
    *,
    prefix: str,
) -> None:
    response = np.asarray(response, dtype=np.float32)
    static = np.asarray(static, dtype=np.float32)
    delta = response - static
    out[f"{prefix}"] = (basis.T @ response).reshape(-1).astype(np.float32)
    out[f"{prefix}_delta"] = (basis.T @ delta).reshape(-1).astype(np.float32)


def _stack_condition_features(
    records: list[dict[str, Any]],
    summaries: dict[int, dict[str, np.ndarray]],
    summary_name: str,
) -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec["family"] == "static":
            key = ("static", "static")
        else:
            key = (str(rec["family"]), str(rec["scale_id"]))
        grouped[key][int(rec["image_index"])].append(summaries[int(rec["response_id"])][summary_name])
    out: dict[tuple[str, str], np.ndarray] = {}
    for key, by_image in grouped.items():
        image_features = []
        for image_index in sorted(by_image):
            image_features.append(np.mean(np.vstack(by_image[image_index]), axis=0))
        out[key] = np.vstack(image_features).astype(np.float32)
    return out


def _bootstrap_condition_delta(
    per_image_a: np.ndarray,
    per_image_b: np.ndarray,
    sessions: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    delta = np.asarray(per_image_a, dtype=np.float64) - np.asarray(per_image_b, dtype=np.float64)
    obs = float(np.nanmean(delta))
    if int(n_bootstrap) <= 0:
        return obs, float("nan"), float("nan")
    unique_sessions = np.unique(sessions)
    boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for j in range(int(n_bootstrap)):
        sampled_sessions = rng.choice(unique_sessions, size=unique_sessions.size, replace=True)
        idx = np.concatenate([np.flatnonzero(sessions == sess) for sess in sampled_sessions])
        boot[j] = float(np.nanmean(delta[idx]))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return obs, float(lo), float(hi)


def _condition_motion_tensor(
    records: list[dict[str, Any]],
    summaries: dict[int, dict[str, np.ndarray]],
    summary_name: str,
    key: tuple[str, str],
) -> np.ndarray:
    by_image: dict[int, list[np.ndarray]] = defaultdict(list)
    for rec in records:
        rec_key = ("static", "static") if rec["family"] == "static" else (str(rec["family"]), str(rec["scale_id"]))
        if rec_key != key:
            continue
        by_image[int(rec["image_index"])].append(summaries[int(rec["response_id"])][summary_name])
    arrays = [np.vstack(by_image[i]) for i in sorted(by_image)]
    return np.asarray(arrays, dtype=np.float32)


def _subspace_overlap(A: np.ndarray, B: np.ndarray, max_dim: int) -> float:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape[0] < 2 or B.shape[0] < 2:
        return float("nan")
    A -= np.mean(A, axis=0, keepdims=True)
    B -= np.mean(B, axis=0, keepdims=True)
    _, _, vt_a = np.linalg.svd(A, full_matrices=False)
    _, _, vt_b = np.linalg.svd(B, full_matrices=False)
    dim = int(min(max_dim, vt_a.shape[0], vt_b.shape[0]))
    if dim < 1:
        return float("nan")
    overlap = np.linalg.norm(vt_a[:dim] @ vt_b[:dim].T, ord="fro") ** 2 / float(dim)
    return float(overlap)


def _covariance_rows(
    records: list[dict[str, Any]],
    summaries: dict[int, dict[str, np.ndarray]],
    summary_names: list[str],
    condition_keys: list[tuple[str, str]],
    *,
    overlap_dim: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_name in summary_names:
        for family, scale_id in condition_keys:
            tensor = _condition_motion_tensor(records, summaries, summary_name, (family, scale_id))
            if tensor.ndim != 3:
                continue
            mu = np.mean(tensor, axis=1)
            residual = (tensor - mu[:, None, :]).reshape(-1, tensor.shape[-1])
            signal_trace = float(np.sum(np.var(mu, axis=0, ddof=1))) if mu.shape[0] > 1 else float("nan")
            motion_trace = float(np.sum(np.var(residual, axis=0, ddof=1))) if residual.shape[0] > 1 else float("nan")
            rows.append(
                {
                    "summary": summary_name,
                    "family": family,
                    "scale_id": scale_id,
                    "n_images": int(tensor.shape[0]),
                    "n_trace_samples": int(tensor.shape[1]),
                    "feature_dim": int(tensor.shape[2]),
                    "signal_cov_trace": signal_trace,
                    "motion_cov_trace": motion_trace,
                    "signal_motion_trace_ratio": signal_trace / (motion_trace + 1e-12) if np.isfinite(signal_trace) else float("nan"),
                    "signal_motion_subspace_overlap": _subspace_overlap(mu, residual, int(overlap_dim)),
                }
            )
    return rows


def _decode_rows(
    feature_by_condition: dict[tuple[str, str], np.ndarray],
    latent_arrays: dict[str, np.ndarray],
    groups: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, int], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    per_image: dict[tuple[str, str, str, int], np.ndarray] = {}
    alphas = _parse_float_list(args.ridge_alphas)
    fixed_alpha = float(args.fixed_ridge_alpha) if args.fixed_ridge_alpha is not None else float(alphas[len(alphas) // 2])
    for (family, scale_id), X in sorted(feature_by_condition.items()):
        for latent_name, Z in sorted(latent_arrays.items()):
            for k in _parse_int_list(args.pca_k_list):
                result = _cross_validated_decode(
                    X,
                    Z,
                    groups,
                    k=int(k),
                    alphas=alphas,
                    alpha_mode="fixed",
                    fixed_alpha=fixed_alpha,
                    outer_folds=int(args.outer_folds),
                    inner_folds=int(args.inner_folds),
                    seed=int(args.seed),
                )
                key = (family, scale_id, latent_name, int(k))
                per_image[key] = np.asarray(result["per_window_score"], dtype=np.float64)
                rows.append(
                    {
                        "family": family,
                        "scale_id": scale_id,
                        "latent": latent_name,
                        "k": int(k),
                        "mean_neg_mse": float(result["mean_neg_mse"]),
                        "r2": float(result["r2"]),
                        "chosen_alpha_median": float(result["chosen_alpha_median"]),
                        "ridge_alpha_mode": "fixed",
                        "fixed_ridge_alpha": fixed_alpha,
                        "target_dim": int(result["target_dim"]),
                        "n_images": int(X.shape[0]),
                        "decode_group_mode": str(args.decode_group_mode),
                        "n_decode_groups": int(np.unique(groups).size),
                        "feature_dim": int(X.shape[1]),
                    }
                )
    return rows, per_image


def _contrast_rows(
    decode_rows: list[dict[str, Any]],
    per_image: dict[tuple[str, str, str, int], np.ndarray],
    sessions: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed) + 101)
    static_key_by_latent_k = {
        (str(row["latent"]), int(row["k"])): ("static", "static", str(row["latent"]), int(row["k"]))
        for row in decode_rows
        if row["family"] == "static"
    }
    contrasts = [("empirical", "ou"), ("empirical", "brownian"), ("empirical", "rotated"), ("ou", "brownian")]
    row_lookup = {
        (str(row["family"]), str(row["scale_id"]), str(row["latent"]), int(row["k"])): row
        for row in decode_rows
    }
    for row in decode_rows:
        scale_id = str(row["scale_id"])
        latent = str(row["latent"])
        k = int(row["k"])
        if str(row["family"]) == "static":
            continue
        for lhs_family, rhs_family in contrasts:
            if str(row["family"]) != lhs_family:
                continue
            lhs_key = (lhs_family, scale_id, latent, k)
            rhs_key = (rhs_family, scale_id, latent, k)
            if lhs_key not in per_image or rhs_key not in per_image:
                continue
            mean, lo, hi = _bootstrap_condition_delta(
                per_image[lhs_key],
                per_image[rhs_key],
                sessions,
                n_bootstrap=int(args.n_bootstrap),
                rng=rng,
            )
            rows.append(
                {
                    "lhs_family": lhs_family,
                    "rhs_family": rhs_family,
                    "scale_id": scale_id,
                    "latent": latent,
                    "k": k,
                    "lhs_mean_neg_mse": float(row_lookup[lhs_key]["mean_neg_mse"]),
                    "rhs_mean_neg_mse": float(row_lookup[rhs_key]["mean_neg_mse"]),
                    "mean_delta_neg_mse": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_images": int(sessions.size),
                }
            )
        if str(row["family"]) != "empirical":
            continue
        lhs_key = ("empirical", scale_id, latent, k)
        rhs_key = static_key_by_latent_k.get((latent, k))
        if rhs_key is not None:
            if rhs_key not in per_image:
                continue
            mean, lo, hi = _bootstrap_condition_delta(
                per_image[lhs_key],
                per_image[rhs_key],
                sessions,
                n_bootstrap=int(args.n_bootstrap),
                rng=rng,
            )
            rows.append(
                {
                    "lhs_family": "empirical",
                    "rhs_family": "static",
                    "scale_id": scale_id,
                    "latent": latent,
                    "k": k,
                    "lhs_mean_neg_mse": float(row_lookup[lhs_key]["mean_neg_mse"]),
                    "rhs_mean_neg_mse": float(row_lookup[rhs_key]["mean_neg_mse"]),
                    "mean_delta_neg_mse": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_images": int(sessions.size),
                }
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window-manifest", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--trace-samples-per-condition", type=int, default=4)
    parser.add_argument("--motion-families", default="empirical,ou,brownian,rotated")
    parser.add_argument("--observed-rms-scales", default="0.25,0.5,1.0")
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--min-patch-image-margin-px", type=float, default=None)
    parser.add_argument("--latent-crop-px", type=int, default=151)
    parser.add_argument("--center-crop-px", type=int, default=41)
    parser.add_argument("--local-field-grid", type=int, default=8)
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--temporal-pc-components", type=int, default=4)
    parser.add_argument("--latent-names", default="gabor_local_field,pyramid_local_field")
    parser.add_argument("--pca-k-list", default="4,8")
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--fixed-ridge-alpha", type=float, default=None)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument(
        "--decode-group-mode",
        choices=("image", "session"),
        default="session",
        help=(
            "CV grouping for feature decoding. image keeps each image/source row in one fold; "
            "session is stricter across sessions. Response arrays are already image-averaged."
        ),
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--reliable-image-coherence-min", type=float, default=0.20)
    parser.add_argument("--reliable-drift-anisotropy-min", type=float, default=0.20)
    parser.add_argument("--min-duration-s", type=float, default=0.10)
    parser.add_argument("--max-rms-deg", type=float, default=0.12)
    parser.add_argument(
        "--max-trace-source-rms-deg",
        type=float,
        default=None,
        help=(
            "Optional common-unclipped trace-pool restriction. For a max scale S and cap C, "
            "set this to C/S so every sampled empirical source trace remains unclipped at S."
        ),
    )
    parser.add_argument("--max-trace-source-radius-deg", type=float, default=None)
    parser.add_argument("--max-trace-source-path-length-deg", type=float, default=None)
    parser.add_argument("--max-trace-source-speed-p95-deg-s", type=float, default=None)
    parser.add_argument(
        "--max-trace-source-microsaccade-events",
        type=int,
        default=None,
        help="Optional Jake-detector event-count filter for the trace source bank. Use 0 for drift-only.",
    )
    parser.add_argument(
        "--microsaccade-speed-threshold-dps",
        type=float,
        default=None,
        help="Fixed microsaccade speed threshold. Defaults to Jake/MAD threshold per trace.",
    )
    parser.add_argument("--microsaccade-threshold-z", type=float, default=6.0)
    parser.add_argument("--microsaccade-pad-frames", type=int, default=1)
    parser.add_argument(
        "--reuse-trace-sources-across-scales",
        action="store_true",
        help="Sample each family/sample trace source once per image and reuse it across all requested scales.",
    )
    parser.add_argument("--twin-batch-size", type=int, default=48)
    parser.add_argument("--twin-trace-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Prepare images/traces/latents but skip twin evaluation.")
    return parser


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    work = _prepare_windows(args)
    if work.empty:
        raise ValueError("No BackImage windows survived the aggregate filters.")
    families = _parse_str_list(args.motion_families)
    valid_families = {"empirical", "ou", "brownian", "rotated"}
    invalid = sorted(set(families).difference(valid_families))
    if invalid:
        raise ValueError(f"Unknown --motion-families entries: {invalid}")
    scales = _parse_float_list(args.observed_rms_scales)
    latent_filter = set(_parse_str_list(args.latent_names))
    cfg = AggregateConfig(
        input=str(args.input),
        out_dir=str(out_dir),
        window_manifest=str(args.window_manifest) if args.window_manifest is not None else None,
        max_images=int(args.max_images),
        trace_samples_per_condition=int(args.trace_samples_per_condition),
        motion_families=families,
        observed_rms_scales=scales,
        patch_size_px=int(args.patch_size_px),
        latent_crop_px=int(args.latent_crop_px),
        center_crop_px=int(args.center_crop_px),
        local_field_grid=int(args.local_field_grid),
        n_timepoints=int(args.n_timepoints),
        temporal_pc_components=int(args.temporal_pc_components),
        pca_k_list=_parse_int_list(args.pca_k_list),
        latent_names=sorted(latent_filter),
        ridge_alphas=_parse_float_list(args.ridge_alphas),
        fixed_ridge_alpha=float(args.fixed_ridge_alpha) if args.fixed_ridge_alpha is not None else None,
        outer_folds=int(args.outer_folds),
        inner_folds=int(args.inner_folds),
        decode_group_mode=str(args.decode_group_mode),
        reliable_image_coherence_min=float(args.reliable_image_coherence_min),
        reliable_drift_anisotropy_min=float(args.reliable_drift_anisotropy_min),
        min_duration_s=float(args.min_duration_s),
        min_patch_image_margin_px=(
            float(args.min_patch_image_margin_px) if args.min_patch_image_margin_px is not None else float(args.patch_size_px) / 2.0
        ),
        max_rms_deg=float(args.max_rms_deg),
        max_trace_source_rms_deg=float(args.max_trace_source_rms_deg) if args.max_trace_source_rms_deg is not None else None,
        max_trace_source_radius_deg=float(args.max_trace_source_radius_deg) if args.max_trace_source_radius_deg is not None else None,
        max_trace_source_path_length_deg=(
            float(args.max_trace_source_path_length_deg) if args.max_trace_source_path_length_deg is not None else None
        ),
        max_trace_source_speed_p95_deg_s=(
            float(args.max_trace_source_speed_p95_deg_s) if args.max_trace_source_speed_p95_deg_s is not None else None
        ),
        max_trace_source_microsaccade_events=(
            int(args.max_trace_source_microsaccade_events) if args.max_trace_source_microsaccade_events is not None else None
        ),
        microsaccade_speed_threshold_dps=(
            float(args.microsaccade_speed_threshold_dps) if args.microsaccade_speed_threshold_dps is not None else None
        ),
        microsaccade_threshold_z=float(args.microsaccade_threshold_z),
        microsaccade_pad_frames=int(args.microsaccade_pad_frames),
        reuse_trace_sources_across_scales=bool(args.reuse_trace_sources_across_scales),
        twin_batch_size=int(args.twin_batch_size),
        twin_trace_batch_size=int(args.twin_trace_batch_size),
        device=str(args.device),
        progress_every=int(args.progress_every),
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
    )
    _write_json(out_dir / "run_metadata.json", {"config": asdict(cfg), "steerable_pyramid": HAVE_STEERABLE_PYRAMID})
    _progress(
        f"prepared {work.shape[0]} images; families={families}; scales={scales}; "
        f"K={args.trace_samples_per_condition}; dry_run={args.dry_run}; output={out_dir}"
    )

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
    )
    trace_rows = [
        {
            "bank_index": j,
            "source_row": int(item["source_row"]),
            "session": str(item["session"]),
            "observed_rms_deg": float(item["observed_rms_deg"]),
            "source_rms_radius_deg": float(item["source_rms_radius_deg"]),
            "source_max_radius_deg": float(item["source_max_radius_deg"]),
            "path_length_deg": float(item["path_length_deg"]),
            "source_path_length_deg": float(item["source_path_length_deg"]),
            "source_speed_p95_deg_s": float(item["source_speed_p95_deg_s"]),
            "duration_s": float(item["duration_s"]),
            "lag1_autocorr": float(item["lag1_autocorr"]),
            "microsaccade_threshold_dps": float(item["microsaccade_threshold_dps"]),
            "n_microsaccade_events": int(item["n_microsaccade_events"]),
            "fraction_microsaccade_samples": float(item["fraction_microsaccade_samples"]),
            "peak_microsaccade_speed_dps": float(item["peak_microsaccade_speed_dps"]),
        }
        for j, item in enumerate(trace_bank)
    ]
    _write_csv(out_dir / "trace_bank_metadata.csv", trace_rows)
    trace_pool = _eligible_trace_bank_indices(
        trace_bank,
        current_source_row=-1,
        max_trace_source_rms_deg=float(args.max_trace_source_rms_deg) if args.max_trace_source_rms_deg is not None else None,
        max_trace_source_radius_deg=float(args.max_trace_source_radius_deg) if args.max_trace_source_radius_deg is not None else None,
        max_trace_source_path_length_deg=(
            float(args.max_trace_source_path_length_deg) if args.max_trace_source_path_length_deg is not None else None
        ),
        max_trace_source_speed_p95_deg_s=(
            float(args.max_trace_source_speed_p95_deg_s) if args.max_trace_source_speed_p95_deg_s is not None else None
        ),
        max_trace_source_microsaccade_events=(
            int(args.max_trace_source_microsaccade_events) if args.max_trace_source_microsaccade_events is not None else None
        ),
    )
    if any(
        value is not None
        for value in (
            args.max_trace_source_rms_deg,
            args.max_trace_source_radius_deg,
            args.max_trace_source_path_length_deg,
            args.max_trace_source_speed_p95_deg_s,
            args.max_trace_source_microsaccade_events,
        )
    ):
        _progress(
            "trace source filter keeps "
            f"{len(trace_pool)}/{len(trace_bank)} traces "
            f"(rms<={args.max_trace_source_rms_deg}, radius<={args.max_trace_source_radius_deg}, "
            f"path<={args.max_trace_source_path_length_deg}, speed_p95<={args.max_trace_source_speed_p95_deg_s}, "
            f"events<={args.max_trace_source_microsaccade_events})"
        )

    scorer = None if args.dry_run else CanonicalTwinScorer(device=str(args.device), batch_size=int(args.twin_batch_size))
    image_rows: list[dict[str, Any]] = []
    motion_rows: list[dict[str, Any]] = []
    latent_values: dict[str, list[np.ndarray]] = {}
    raw_responses: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}

    for image_index, row in tqdm(work.iterrows(), total=work.shape[0], desc="aggregate FEM responses"):
        canvas_key = (str(row["session"]), int(row["trial_idx"]))
        if canvas_key not in canvas_cache:
            canvas_cache[canvas_key] = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
        canvas, ppd, screen_shape = canvas_cache[canvas_key]
        center_px = gaze_deg_to_screen_px(
            np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
            ppd=ppd,
            screen_shape=screen_shape,
        )
        patch = _clip_patch(canvas, (float(center_px[0]), float(center_px[1])), int(args.patch_size_px))
        latents = _extract_requested_latents(
            patch,
            latent_crop_px=int(args.latent_crop_px),
            center_crop_px=int(args.center_crop_px),
            local_field_grid=int(args.local_field_grid),
            requested=latent_filter,
        )
        if not latents:
            raise ValueError(f"No requested latent features were available for image {image_index}.")
        for name, value in latents.items():
            latent_values.setdefault(name, []).append(value)
        image_rows.append(
            {
                "image_index": int(image_index),
                "source_row": int(row["source_row"]),
                "session": str(row["session"]),
                "trial_idx": int(row["trial_idx"]),
                "phase": str(row.get("phase", "")),
                "image_orientation_coherence": float(row["image_orientation_coherence"]),
                "drift_anisotropy": float(row["anisotropy"]),
                "source_observed_rms_radius_deg": float(row.get("rms_radius_deg", np.nan)),
            }
        )
        if args.dry_run:
            continue
        traces = [_static_trace(int(args.n_timepoints))]
        trace_specs: list[dict[str, Any]] = [
            {
                "family": "static",
                "scale_id": "static",
                "scale": 0.0,
                "sample_index": 0,
                "trace_bank_index": -1,
                "source_trace_rms_deg": 0.0,
                "source_trace_lag1": np.nan,
                "requested_rms_deg": 0.0,
                "effective_rms_deg": 0.0,
                "rms_clipped_high": False,
            }
        ]
        reusable_sources: dict[tuple[str, int], int] = {}
        if bool(args.reuse_trace_sources_across_scales):
            eligible = _eligible_trace_bank_indices(
                trace_bank,
                current_source_row=int(row["source_row"]),
                max_trace_source_rms_deg=(
                    float(args.max_trace_source_rms_deg) if args.max_trace_source_rms_deg is not None else None
                ),
                max_trace_source_radius_deg=(
                    float(args.max_trace_source_radius_deg) if args.max_trace_source_radius_deg is not None else None
                ),
                max_trace_source_path_length_deg=(
                    float(args.max_trace_source_path_length_deg) if args.max_trace_source_path_length_deg is not None else None
                ),
                max_trace_source_speed_p95_deg_s=(
                    float(args.max_trace_source_speed_p95_deg_s) if args.max_trace_source_speed_p95_deg_s is not None else None
                ),
                max_trace_source_microsaccade_events=(
                    int(args.max_trace_source_microsaccade_events) if args.max_trace_source_microsaccade_events is not None else None
                ),
            )
            if not eligible:
                raise ValueError("Unpaired sampling has no eligible trace-bank entries after source-RMS filtering.")
            for family in families:
                for sample_index in range(int(args.trace_samples_per_condition)):
                    reusable_sources[(family, sample_index)] = int(eligible[int(rng.integers(0, len(eligible)))])
        for scale in scales:
            scale_id = f"rel_{_scale_token(scale)}x"
            for family in families:
                for sample_index in range(int(args.trace_samples_per_condition)):
                    eligible = _eligible_trace_bank_indices(
                        trace_bank,
                        current_source_row=int(row["source_row"]),
                        max_trace_source_rms_deg=(
                            float(args.max_trace_source_rms_deg) if args.max_trace_source_rms_deg is not None else None
                        ),
                        max_trace_source_radius_deg=(
                            float(args.max_trace_source_radius_deg) if args.max_trace_source_radius_deg is not None else None
                        ),
                        max_trace_source_path_length_deg=(
                            float(args.max_trace_source_path_length_deg) if args.max_trace_source_path_length_deg is not None else None
                        ),
                        max_trace_source_speed_p95_deg_s=(
                            float(args.max_trace_source_speed_p95_deg_s) if args.max_trace_source_speed_p95_deg_s is not None else None
                        ),
                        max_trace_source_microsaccade_events=(
                            int(args.max_trace_source_microsaccade_events) if args.max_trace_source_microsaccade_events is not None else None
                        ),
                    )
                    if not eligible:
                        raise ValueError("Unpaired sampling has no eligible trace-bank entries after source-RMS filtering.")
                    if bool(args.reuse_trace_sources_across_scales):
                        bank_index = reusable_sources[(family, int(sample_index))]
                    else:
                        bank_index = int(eligible[int(rng.integers(0, len(eligible)))])
                    item = trace_bank[bank_index]
                    target_rms = float(scale) * float(item["observed_rms_deg"])
                    target_path = float(scale) * float(item["path_length_deg"])
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
                    traces.append(trace)
                    trace_specs.append(
                        {
                            "family": family,
                            "pairing_mode": "unpaired_ensemble",
                            "scale_id": scale_id,
                            "scale": float(scale),
                            "sample_index": int(sample_index),
                            "trace_bank_index": bank_index,
                            "trace_source_row": int(item["source_row"]),
                            "trace_source_session": str(item["session"]),
                            "source_trace_rms_deg": float(item["observed_rms_deg"]),
                            "source_trace_path_length_deg": float(item["path_length_deg"]),
                            "source_trace_duration_s": float(item["duration_s"]),
                            "source_trace_lag1": float(item["lag1_autocorr"]),
                            "requested_rms_deg": float(meta["requested_rms_deg"]),
                            "effective_rms_deg": float(meta["effective_rms_deg"]),
                            "effective_to_requested_rms": (
                                float(meta["effective_rms_deg"]) / float(meta["requested_rms_deg"])
                                if float(meta["requested_rms_deg"]) > 0.0
                                else np.nan
                            ),
                            "rms_clipped_high": bool(meta["rms_clipped_high"]),
                            "generated_lag1_autocorr": float(meta["generated_lag1_autocorr"]),
                            "target_path_length_deg": target_path,
                            "path_length_deg": float(meta["path_length_deg"]),
                            "path_to_target_ratio": (
                                float(meta["path_length_deg"]) / target_path
                                if target_path > 0.0
                                else np.nan
                            ),
                            "speed_mean_deg_s": float(meta["speed_mean_deg_s"]),
                            "speed_median_deg_s": float(meta["speed_median_deg_s"]),
                            "speed_p95_deg_s": float(meta["speed_p95_deg_s"]),
                        }
                    )
        responses = scorer.responses(patch, traces, trace_batch_size=int(args.twin_trace_batch_size))
        aligned = [_align_response_to_trace(resp, int(args.n_timepoints)) for resp in responses]
        for spec, resp in zip(trace_specs, aligned, strict=True):
            response_id = len(raw_responses)
            raw_responses.append(resp.astype(np.float32, copy=False))
            records.append(
                {
                    "response_id": response_id,
                    "image_index": int(image_index),
                    "family": str(spec["family"]),
                    "scale_id": str(spec["scale_id"]),
                }
            )
            motion_row = {
                "response_id": response_id,
                "image_index": int(image_index),
                "source_row": int(row["source_row"]),
                **spec,
                "response_frames": int(resp.shape[0]),
                "response_units": int(resp.shape[1]),
            }
            motion_rows.append(motion_row)
        done = int(image_index) + 1
        if done == 1 or done == work.shape[0] or (int(args.progress_every) > 0 and done % int(args.progress_every) == 0):
            _progress(f"images {done}/{work.shape[0]}; responses={len(raw_responses)}")

    image_df = pd.DataFrame(image_rows)
    image_df.to_csv(out_dir / "analysis_images.csv", index=False)
    _write_csv(out_dir / "aggregate_motion_metadata.csv", motion_rows)
    if motion_rows:
        motion_df = pd.DataFrame(motion_rows)
        summary_cols = [
            "family",
            "scale_id",
            "scale",
            "pairing_mode",
            "effective_rms_deg",
            "requested_rms_deg",
            "effective_to_requested_rms",
            "path_length_deg",
            "path_to_target_ratio",
            "speed_mean_deg_s",
            "speed_median_deg_s",
            "speed_p95_deg_s",
            "generated_lag1_autocorr",
            "rms_clipped_high",
        ]
        available = [col for col in summary_cols if col in motion_df.columns]
        grouped = motion_df.loc[motion_df["family"] != "static", available].groupby(["family", "scale_id"], dropna=False)
        motion_summary = grouped.agg(
            n=("effective_rms_deg", "size"),
            median_effective_rms_deg=("effective_rms_deg", "median"),
            iqr_effective_rms_deg=("effective_rms_deg", lambda x: float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))),
            median_effective_to_requested_rms=("effective_to_requested_rms", "median"),
            median_path_length_deg=("path_length_deg", "median"),
            median_path_to_target_ratio=("path_to_target_ratio", "median"),
            median_speed_mean_deg_s=("speed_mean_deg_s", "median"),
            median_generated_lag1_autocorr=("generated_lag1_autocorr", "median"),
            clipped_fraction=("rms_clipped_high", "mean"),
        ).reset_index()
        motion_summary.to_csv(out_dir / "aggregate_motion_summary.csv", index=False)
    latent_arrays = {name: np.vstack(values).astype(np.float32) for name, values in latent_values.items()}
    np.savez_compressed(out_dir / "latent_feature_arrays.npz", **latent_arrays)

    if args.dry_run:
        _progress("dry run complete; skipped twin responses and summaries")
        return out_dir

    _progress("fitting temporal response basis and writing compact response summaries")
    basis = _fit_temporal_basis(raw_responses, int(args.temporal_pc_components))
    dct_basis = _fixed_dct_basis(int(args.n_timepoints), int(args.temporal_pc_components))
    static_by_image = {
        int(rec["image_index"]): raw_responses[int(rec["response_id"])]
        for rec in records
        if rec["family"] == "static"
    }
    response_summaries = {}
    for rec in records:
        response_id = int(rec["response_id"])
        image_index = int(rec["image_index"])
        summaries = _summarize_response(raw_responses[response_id], static_by_image[image_index], basis)
        _add_temporal_basis_summaries(
            summaries,
            raw_responses[response_id],
            static_by_image[image_index],
            dct_basis,
            prefix="temporal_dct",
        )
        response_summaries[response_id] = summaries
    summary_names = ["temporal_pca", "temporal_delta_pca", "temporal_dct", "temporal_dct_delta", "mean", "delta_mean"]
    summary_arrays: dict[str, np.ndarray] = {}
    for summary in summary_names:
        by_condition = _stack_condition_features(records, response_summaries, summary)
        for (family, scale_id), arr in by_condition.items():
            summary_arrays[f"{summary}__{family}__{scale_id}"] = arr
    np.savez_compressed(out_dir / "response_summary_arrays.npz", temporal_basis=basis, temporal_dct_basis=dct_basis, **summary_arrays)

    sessions = image_df["session"].to_numpy()
    decode_groups = (
        image_df["image_index"].to_numpy(dtype=int)
        if str(args.decode_group_mode) == "image"
        else sessions
    )
    all_decode_rows: list[dict[str, Any]] = []
    all_per_image: dict[tuple[str, str, str, str, int], np.ndarray] = {}
    for summary in summary_names:
        by_condition = _stack_condition_features(records, response_summaries, summary)
        rows, per_image = _decode_rows(by_condition, latent_arrays, decode_groups, args)
        for row in rows:
            row["response_summary"] = summary
            all_decode_rows.append(row)
        for key, values in per_image.items():
            all_per_image[(summary, *key)] = values
        _progress(f"decoded summary={summary}; jobs={len(rows)}")
    _write_csv(out_dir / "decode_summary.csv", all_decode_rows)

    contrast_input_rows = []
    contrast_rows: list[dict[str, Any]] = []
    for summary in summary_names:
        rows = [row for row in all_decode_rows if row["response_summary"] == summary]
        per_image = {
            key[1:]: value
            for key, value in all_per_image.items()
            if key[0] == summary
        }
        for crow in _contrast_rows(rows, per_image, sessions, args):
            crow["response_summary"] = summary
            contrast_rows.append(crow)
        contrast_input_rows.extend(rows)
    _write_csv(out_dir / "decode_contrasts.csv", contrast_rows)

    condition_keys = sorted({("static", "static")} | {(str(rec["family"]), str(rec["scale_id"])) for rec in records if rec["family"] != "static"})
    cov_rows = _covariance_rows(records, response_summaries, summary_names, condition_keys, overlap_dim=5)
    _write_csv(out_dir / "covariance_summary.csv", cov_rows)

    report = [
        "# BackImage Aggregate FEM Information Pilot",
        "",
        f"- Images: {work.shape[0]}",
        f"- Trace samples per family/scale/image: {args.trace_samples_per_condition}",
        f"- Families: {', '.join(families)}",
        f"- Scales: {', '.join(str(v) for v in scales)}",
        f"- Temporal basis components: {basis.shape[1]}",
        f"- Latents: {', '.join(latent_arrays)}",
        "",
        "Primary files:",
        "- `decode_summary.csv`",
        "- `decode_contrasts.csv`",
        "- `covariance_summary.csv`",
        "- `response_summary_arrays.npz`",
    ]
    (out_dir / "summary_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _progress(f"complete; wrote summaries to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
