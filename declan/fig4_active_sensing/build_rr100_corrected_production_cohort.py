#!/usr/bin/env python3
"""Build the immutable corrected 100-image x 1,000-trace production cohort.

The script separates outcome-independent candidate definition, session-isolated
input validation, stratified selection, and final cohort assembly.  It never
loads or calls the neural model.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig4_active_sensing.analyze_rr100_corrected_figure4_cache import render_with_common
from declan.fig4_active_sensing.audit_rr100_eye_trace_conditioning_and_nyquist_power import (
    centered,
    corrected_crop_xy_deg,
    load_dset,
    model_aligned_indices,
)
from declan.fig4_active_sensing.audit_rr100_legacy100_image_identities import metrics as image_metrics
from declan.fixation_statistics_by_stimulus.image_features import (
    backimage_trial_geometry,
    gaze_deg_to_screen_px,
)
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import (
    load_source_rows,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _load_twin_common


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
WINDOWS = ROOT / "outputs/fig4_active_sensing/backimage_240hz_timebase_checkpoint_25_v1/backimage_window_features_240hz.csv"
EVENTS = ROOT / "outputs/fig4_active_sensing/backimage_240hz_timebase_checkpoint_25_v1/backimage_event_features_240hz.csv"
LEGACY_IMAGES = ROOT / "outputs/fig4_active_sensing/rr100_legacy100_corrected_image_audit_checkpoint_24_v1/corrected_image_crosswalk.csv"
LEGACY_IMAGE_PARTIALS = ROOT / "outputs/fig4_active_sensing/rr100_legacy100_corrected_image_audit_checkpoint_24_v1/partials"
LEGACY_TRACES = ROOT / "outputs/fig4_active_sensing/rr100_legacy1000_trace_agreement_checkpoint_23_v1/corrected_trace_crosswalk.csv"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"

N_HISTORY = 32
N_SCORE = 40
MODEL_HZ = 120.0
SOURCE_HZ = 240.0
PATCH_SIZE = 540
SEED = 20260813
N_IMAGE_REPLACEMENTS = 51
N_TRACE_REPLACEMENTS = 27
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=SOURCE)
    parser.add_argument("--window-table", type=Path, default=WINDOWS)
    parser.add_argument("--event-table", type=Path, default=EVENTS)
    parser.add_argument("--legacy-images", type=Path, default=LEGACY_IMAGES)
    parser.add_argument("--legacy-traces", type=Path, default=LEGACY_TRACES)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--image-session")
    mode.add_argument("--trace-session")
    mode.add_argument("--assemble", action="store_true")
    mode.add_argument("--all", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "sha256": digest.hexdigest()}


def trace_metrics(trace: np.ndarray) -> dict[str, float]:
    arr = centered(np.asarray(trace, dtype=np.float64))
    steps = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    freq = np.fft.rfftfreq(len(arr), d=1.0 / MODEL_HZ)
    fft = np.fft.rfft(arr * np.hanning(len(arr))[:, None], axis=0)
    power = np.sum(np.abs(fft) ** 2, axis=1)
    positive = freq > 0
    high32 = freq >= 32.0
    total = max(float(power[positive].sum()), EPS)
    cov = np.cov(arr.T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    return {
        "n_frames": float(len(arr)),
        "sample_rate_hz": MODEL_HZ,
        "duration_s": float((len(arr) - 1) / MODEL_HZ),
        "path_length_arcmin": float(steps.sum() * 60.0),
        "rms_radius_arcmin": float(np.sqrt(np.mean(np.sum(arr**2, axis=1))) * 60.0),
        "max_radius_arcmin": float(np.max(np.linalg.norm(arr, axis=1)) * 60.0),
        "median_step_arcmin": float(np.median(steps) * 60.0),
        "mean_speed_dps": float(np.mean(steps) * MODEL_HZ),
        "median_speed_dps": float(np.median(steps) * MODEL_HZ),
        "p95_speed_dps": float(np.percentile(steps, 95) * MODEL_HZ),
        "position_power_total_positive": total,
        "position_power_fraction_15plus_hz": float(power[freq >= 15.0].sum() / total),
        "position_power_fraction_32plus_hz": float(power[high32].sum() / total),
        "position_power_centroid_hz": float(np.sum(freq[positive] * power[positive]) / total),
        "cov_major_sd_arcmin": float(np.sqrt(max(values[0], 0.0)) * 60.0),
        "cov_minor_sd_arcmin": float(np.sqrt(max(values[1], 0.0)) * 60.0),
        "cov_anisotropy": float((values[0] - values[1]) / max(values.sum(), EPS)),
        "cov_orientation_deg": float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])) % 180.0),
    }


def corrected_intervals(global_start: int, global_stop: int) -> tuple[np.ndarray, np.ndarray]:
    center = (int(global_start) + int(global_stop)) // 2
    score_start = center - N_SCORE
    if score_start % 2:
        score_start -= 1
    score = np.arange(score_start, score_start + 2 * N_SCORE, 2, dtype=np.int64)
    history = np.arange(score_start - 2 * N_HISTORY, score_start, 2, dtype=np.int64)
    return history, score


def rejection_text(flags: list[tuple[bool, str]]) -> str:
    reasons = [reason for failed, reason in flags if failed]
    return ";".join(reasons) if reasons else "eligible_for_session_validation"


def prepare_candidates(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = load_source_rows(args.source_csv)
    legacy_images = pd.read_csv(args.legacy_images)
    used_image_rows = set(legacy_images.source_row.astype(int))
    image = source.copy()
    key_cols = ["session", "trial_idx", "global_start", "global_stop"]
    image["candidate_id"] = np.arange(len(image), dtype=np.int64)
    image["is_duplicate_interval"] = image.duplicated(key_cols, keep="first")
    image["is_legacy_image_identity"] = image.source_row.astype(int).isin(used_image_rows)
    image["source_image_feature_ok"] = image.image_feature_ok.fillna(False).astype(bool)
    image["source_patch_inside_pass"] = image.image_patch_fraction_inside_image.ge(0.99)
    image["source_background_pass"] = image.image_patch_fraction_background.le(0.01)
    image["source_border_pass"] = image.image_patch_distance_to_image_border_px.ge(PATCH_SIZE / 2)
    image["source_trial_validity_pass"] = image.valid_fraction_trial.ge(0.999)
    image["prevalidation_rejection_reason"] = [
        rejection_text(
            [
                (bool(row.is_duplicate_interval), "duplicate_session_trial_interval"),
                (bool(row.is_legacy_image_identity), "legacy_image_identity"),
                (not bool(row.source_image_feature_ok), "source_image_features_invalid"),
                (not bool(row.source_patch_inside_pass), "source_patch_not_inside_image"),
                (not bool(row.source_background_pass), "source_patch_background_fraction_gt_0p01"),
                (not bool(row.source_border_pass), "source_patch_border_lt_270px"),
                (not bool(row.source_trial_validity_pass), "source_trial_valid_fraction_lt_0p999"),
            ]
        )
        for row in image.itertuples()
    ]
    image["prevalidation_eligible"] = image.prevalidation_rejection_reason.eq("eligible_for_session_validation")
    image.to_csv(args.out_dir / "image_candidate_pool_prevalidation.csv", index=False)

    windows = pd.read_csv(args.window_table).copy()
    windows["replacement_window_index"] = np.arange(len(windows), dtype=np.int64)
    interval_arrays = [corrected_intervals(start, stop) for start, stop in zip(windows.global_start, windows.global_stop)]
    windows["corrected_history_global_start"] = [int(history[0]) for history, _ in interval_arrays]
    windows["corrected_history_global_stop_exclusive"] = [int(history[-1] + 2) for history, _ in interval_arrays]
    windows["corrected_scored_global_start"] = [int(score[0]) for _, score in interval_arrays]
    windows["corrected_scored_global_stop_exclusive"] = [int(score[-1] + 2) for _, score in interval_arrays]
    legacy_traces = pd.read_csv(args.legacy_traces)
    used_trace_keys = set(
        zip(
            legacy_traces.session.astype(str),
            legacy_traces.trial_idx.astype(int),
            legacy_traces.corrected_scored_global_start.astype(int),
            legacy_traces.corrected_scored_global_stop_exclusive.astype(int),
        )
    )
    candidate_keys = list(
        zip(
            windows.session.astype(str),
            windows.trial_idx.astype(int),
            windows.corrected_scored_global_start.astype(int),
            windows.corrected_scored_global_stop_exclusive.astype(int),
        )
    )
    windows["is_legacy_trace_interval"] = [key in used_trace_keys for key in candidate_keys]
    windows["is_duplicate_corrected_interval"] = pd.Series(candidate_keys).duplicated(keep="first").to_numpy()
    windows["source_trial_validity_pass"] = windows.valid_fraction_trial.ge(0.999)
    windows["prevalidation_rejection_reason"] = [
        rejection_text(
            [
                (bool(row.is_legacy_trace_interval), "legacy_trace_scored_interval"),
                (bool(row.is_duplicate_corrected_interval), "duplicate_corrected_scored_interval"),
                (not bool(row.source_trial_validity_pass), "source_trial_valid_fraction_lt_0p999"),
            ]
        )
        for row in windows.itertuples()
    ]
    windows["prevalidation_eligible"] = windows.prevalidation_rejection_reason.eq("eligible_for_session_validation")
    windows.to_csv(args.out_dir / "trace_candidate_pool_prevalidation.csv", index=False)

    manifest = {
        "created_utc": utc_now(),
        "status": "replacement_candidate_pools_prepared_not_validated",
        "seed": int(args.seed),
        "image_counts": {
            "all": len(image),
            "prevalidation_eligible": int(image.prevalidation_eligible.sum()),
            "sessions": int(image.loc[image.prevalidation_eligible, "session"].nunique()),
        },
        "trace_counts": {
            "all": len(windows),
            "prevalidation_eligible": int(windows.prevalidation_eligible.sum()),
            "sessions": int(windows.loc[windows.prevalidation_eligible, "session"].nunique()),
        },
        "selection_is_outcome_independent": True,
        "neural_model_calls": False,
        "sources": {
            "source": file_identity(args.source_csv),
            "windows": file_identity(args.window_table),
            "events": file_identity(args.event_table),
            "legacy_images": file_identity(args.legacy_images),
            "legacy_traces": file_identity(args.legacy_traces),
        },
    }
    atomic_json(args.out_dir / "candidate_pool_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def image_session_worker(args: argparse.Namespace, session: str) -> None:
    pool = pd.read_csv(args.out_dir / "image_candidate_pool_prevalidation.csv")
    work = pool[pool.prevalidation_eligible.astype(bool) & pool.session.eq(session)].copy()
    partial_dir = args.out_dir / "image_validation_partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    output = partial_dir / f"{session}.csv"
    if output.exists():
        print(f"image validation exists: {output}", flush=True)
        return
    if work.empty:
        raise ValueError(f"No eligible image candidates in {session}")
    dset = load_dset(session, {})
    common = _load_twin_common()
    canvas: dict[Any, Any] = {}
    crop_all = corrected_crop_xy_deg(dset)
    trial_all = np.asarray(dset.covariates["trial_inds"]).reshape(-1)
    valid_all = np.asarray(dset.covariates["dpi_valid"]).reshape(-1).astype(bool)
    roi = np.asarray(dset.metadata["roi_src"], dtype=float)
    center_yx = (roi[:, 0] + roi[:, 1] - 1.0) / 2.0
    rf_offset = np.asarray([center_yx[1], -center_yx[0]], dtype=float) / float(dset.metadata["ppd"])
    records: list[dict[str, Any]] = []
    for ordinal, row in enumerate(work.itertuples(index=False)):
        record = {
            "candidate_id": int(row.candidate_id),
            "source_row": int(row.source_row),
            "session": session,
            "trial_idx": int(row.trial_idx),
            "global_start": int(row.global_start),
            "global_stop": int(row.global_stop),
        }
        try:
            indices = model_aligned_indices(int(row.global_start), int(row.global_stop))
            crop = crop_all[indices]
            center_xy = crop.mean(axis=0) + rf_offset
            corrected_row = pd.Series(row._asdict())
            corrected_row["mean_x_deg"] = float(center_xy[0])
            corrected_row["mean_y_deg"] = float(center_xy[1])
            patch, patch_meta = _extract_patch(corrected_row, canvas_cache=canvas, patch_size_px=PATCH_SIZE)
            reconstruction = render_with_common(
                np.asarray(patch, np.float32),
                -centered(crop),
                ppd=float(patch_meta["patch_ppd"]),
                common=common,
            )
            exact = np.asarray(dset["stim"][indices], dtype=np.float32)
            exact_available = exact.shape == reconstruction.shape and np.isfinite(exact).all()
            geometry = backimage_trial_geometry(session, int(row.trial_idx))
            center_px = gaze_deg_to_screen_px(
                center_xy,
                ppd=float(geometry["ppd"]),
                screen_shape=geometry["screen_shape"],
            )
            border = min(
                center_px[0],
                center_px[1],
                geometry["screen_width_px"] - center_px[0],
                geometry["screen_height_px"] - center_px[1],
            )
            crop_valid = bool(
                np.isfinite(crop).all()
                and np.all(valid_all[indices])
                and np.all(trial_all[indices] == int(row.trial_idx))
                and border >= PATCH_SIZE / 2
                and np.isfinite(reconstruction).all()
            )
            descriptor = image_metrics(patch, float(patch_meta["patch_ppd"]))
            record.update(
                {
                    "exact_saved_stim_available": bool(exact_available),
                    "corrected_crop_valid": crop_valid,
                    "corrected_crop_border_px": float(border),
                    "reconstruction_exact_pixel_r": (
                        float(np.corrcoef(reconstruction.ravel(), exact.ravel())[0, 1])
                        if exact_available
                        else np.nan
                    ),
                    "reconstruction_exact_mae": (
                        float(np.mean(np.abs(reconstruction - exact))) if exact_available else np.nan
                    ),
                    "corrected_reconstruction_rms_contrast": descriptor["rms_contrast"],
                    "corrected_reconstruction_gradient_energy": descriptor["gradient_energy"],
                    "corrected_reconstruction_orientation_coherence": descriptor["orientation_coherence"],
                    "corrected_reconstruction_contour_axis_deg": descriptor["contour_axis_deg"],
                    "corrected_reconstruction_sf_centroid_cpd": descriptor["sf_centroid_cpd"],
                    "corrected_reconstruction_high_sf_fraction": descriptor["high_sf_fraction"],
                    "patch_ppd": float(patch_meta["patch_ppd"]),
                    "validation_pass": bool(crop_valid and exact_available),
                    "validation_rejection_reason": (
                        "validated_corrected_image_candidate"
                        if crop_valid and exact_available
                        else ";".join(
                            reason
                            for failed, reason in (
                                (not crop_valid, "corrected_crop_invalid"),
                                (not exact_available, "exact_saved_stim_unavailable"),
                            )
                            if failed
                        )
                    ),
                }
            )
            if record["validation_pass"]:
                patch_path = partial_dir / "patches" / f"candidate_{int(row.candidate_id):05d}.npz"
                atomic_npz(
                    patch_path,
                    candidate_id=np.asarray(int(row.candidate_id), dtype=np.int64),
                    corrected_patch=np.asarray(patch, dtype=np.float32),
                    patch_ppd=np.asarray(float(patch_meta["patch_ppd"]), dtype=np.float64),
                )
                record["corrected_patch_npz"] = str(patch_path.resolve())
                record["corrected_patch_key"] = "corrected_patch"
        except Exception as exc:
            record.update(
                {
                    "exact_saved_stim_available": False,
                    "corrected_crop_valid": False,
                    "validation_pass": False,
                    "validation_rejection_reason": f"worker_error:{type(exc).__name__}:{exc}",
                }
            )
        records.append(record)
        if (ordinal + 1) % 20 == 0 or ordinal + 1 == len(work):
            print(f"{session}: image candidates {ordinal + 1}/{len(work)}", flush=True)
    temporary = output.with_suffix(".tmp.csv")
    pd.DataFrame(records).to_csv(temporary, index=False)
    os.replace(temporary, output)


def trace_session_worker(args: argparse.Namespace, session: str) -> None:
    pool = pd.read_csv(args.out_dir / "trace_candidate_pool_prevalidation.csv")
    work = pool[pool.prevalidation_eligible.astype(bool) & pool.session.eq(session)].copy()
    partial_dir = args.out_dir / "trace_validation_partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    output = partial_dir / f"{session}.csv"
    if output.exists():
        print(f"trace validation exists: {output}", flush=True)
        return
    if work.empty:
        raise ValueError(f"No eligible trace candidates in {session}")
    dset = load_dset(session, {})
    crop = corrected_crop_xy_deg(dset)
    eyepos = np.asarray(dset["eyepos"], dtype=np.float64)
    trial = np.asarray(dset.covariates["trial_inds"]).reshape(-1)
    valid = np.asarray(dset.covariates["dpi_valid"]).reshape(-1).astype(bool)
    records: list[dict[str, Any]] = []
    for ordinal, row in enumerate(work.itertuples(index=False)):
        history = np.arange(
            int(row.corrected_history_global_start),
            int(row.corrected_history_global_stop_exclusive),
            2,
            dtype=np.int64,
        )
        score = np.arange(
            int(row.corrected_scored_global_start),
            int(row.corrected_scored_global_stop_exclusive),
            2,
            dtype=np.int64,
        )
        all_indices = np.concatenate([history, score])
        in_bounds = bool(all_indices[0] >= 0 and all_indices[-1] < len(crop))
        same_trial = bool(in_bounds and np.all(trial[all_indices] == int(row.trial_idx)))
        all_valid = bool(in_bounds and np.all(valid[all_indices]))
        all_finite = bool(
            in_bounds and np.isfinite(crop[all_indices]).all() and np.isfinite(eyepos[all_indices]).all()
        )
        passed = bool(
            history.shape == (N_HISTORY,)
            and score.shape == (N_SCORE,)
            and in_bounds
            and same_trial
            and all_valid
            and all_finite
        )
        record = {
            "replacement_window_index": int(row.replacement_window_index),
            "source_row": -1,
            "session": session,
            "trial_idx": int(row.trial_idx),
            "corrected_history_global_start": int(history[0]),
            "corrected_history_global_stop_exclusive": int(history[-1] + 2),
            "corrected_scored_global_start": int(score[0]),
            "corrected_scored_global_stop_exclusive": int(score[-1] + 2),
            "global_decimation_parity": "even",
            "history_in_dataset_bounds": in_bounds,
            "history_and_target_same_trial": same_trial,
            "history_and_target_all_dpi_valid": all_valid,
            "history_and_target_all_finite": all_finite,
            "explicit_history_valid": passed,
            "corrected_events_in_trial": int(row.events_in_trial),
            "corrected_phase": str(row.phase),
            "validation_rejection_reason": (
                "validated_corrected_trace_candidate"
                if passed
                else ";".join(
                    reason
                    for failed, reason in (
                        (history.shape != (N_HISTORY,), "history_length_not_32"),
                        (score.shape != (N_SCORE,), "score_length_not_40"),
                        (not in_bounds, "history_or_score_out_of_bounds"),
                        (not same_trial, "history_and_score_cross_trial"),
                        (not all_valid, "history_or_score_dpi_invalid"),
                        (not all_finite, "history_or_score_nonfinite"),
                    )
                    if failed
                )
            ),
        }
        if passed:
            descriptor = trace_metrics(crop[score])
            record.update({f"corrected_dpi_crop120_{key}": value for key, value in descriptor.items()})
        records.append(record)
        if (ordinal + 1) % 100 == 0 or ordinal + 1 == len(work):
            print(f"{session}: trace candidates {ordinal + 1}/{len(work)}", flush=True)
    temporary = output.with_suffix(".tmp.csv")
    pd.DataFrame(records).to_csv(temporary, index=False)
    os.replace(temporary, output)


def quantile_codes(values: pd.Series, bins: int = 3) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=bins, labels=False, duplicates="drop").astype(int)


def allocate_session_quotas(counts: pd.Series, total: int) -> dict[str, int]:
    sessions = counts.index.astype(str).tolist()
    if len(sessions) > total:
        order = counts.sort_values(ascending=False).index.astype(str).tolist()
        return {session: int(session in order[:total]) for session in sessions}
    quota = {session: 1 for session in sessions}
    remaining = total - len(sessions)
    weights = np.sqrt(counts.astype(float))
    raw = remaining * weights / weights.sum()
    floors = np.floor(raw).astype(int)
    for session, value in floors.items():
        quota[str(session)] += int(value)
    leftover = total - sum(quota.values())
    remainders = (raw - floors).sort_values(ascending=False)
    for session in remainders.index[:leftover]:
        quota[str(session)] += 1
    for session in sessions:
        quota[session] = min(quota[session], int(counts.loc[session]))
    while sum(quota.values()) < total:
        candidates = [session for session in sessions if quota[session] < int(counts.loc[session])]
        if not candidates:
            raise RuntimeError("Could not allocate requested session quotas")
        session = max(candidates, key=lambda item: int(counts.loc[item]) - quota[item])
        quota[session] += 1
    return quota


def stratified_image_selection(valid: pd.DataFrame, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    out = valid.copy()
    metrics = [
        "corrected_reconstruction_rms_contrast",
        "corrected_reconstruction_orientation_coherence",
        "corrected_reconstruction_sf_centroid_cpd",
    ]
    labels = ["contrast", "coherence", "sf"]
    for metric, label in zip(metrics, labels, strict=True):
        out[f"{label}_tertile"] = quantile_codes(out[metric])
    out["selection_stratum"] = out[[f"{label}_tertile" for label in labels]].astype(str).agg("-".join, axis=1)
    rng = np.random.default_rng(seed)
    out["selection_random_key"] = rng.random(len(out))
    quotas = allocate_session_quotas(out.session.value_counts(), n)
    selected: list[int] = []
    stratum_counts: dict[str, int] = {}
    session_counts: dict[str, int] = {session: 0 for session in quotas}
    while len(selected) < n:
        candidates = out.loc[~out.index.isin(selected)].copy()
        candidates = candidates[
            [session_counts[str(session)] < quotas[str(session)] for session in candidates.session]
        ]
        if candidates.empty:
            raise RuntimeError("Image stratification exhausted eligible candidates")
        candidates["stratum_count"] = candidates.selection_stratum.map(stratum_counts).fillna(0)
        candidates["session_fraction"] = [
            session_counts[str(session)] / quotas[str(session)] for session in candidates.session
        ]
        row = candidates.sort_values(
            ["stratum_count", "session_fraction", "selection_random_key", "candidate_id"]
        ).iloc[0]
        selected.append(int(row.name))
        session = str(row.session)
        stratum = str(row.selection_stratum)
        session_counts[session] += 1
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
    return out.loc[selected].copy(), quotas


def stratified_trace_selection(valid: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    out = valid.copy()
    metrics = [
        "corrected_dpi_crop120_path_length_arcmin",
        "corrected_dpi_crop120_rms_radius_arcmin",
        "corrected_dpi_crop120_position_power_fraction_32plus_hz",
    ]
    labels = ["path", "radius", "high_tf"]
    for metric, label in zip(metrics, labels, strict=True):
        out[f"{label}_tertile"] = quantile_codes(out[metric])
    out["selection_stratum"] = out[[f"{label}_tertile" for label in labels]].astype(str).agg("-".join, axis=1)
    rng = np.random.default_rng(seed)
    out["selection_random_key"] = rng.random(len(out))
    selected = (
        out.sort_values(["selection_stratum", "selection_random_key", "replacement_window_index"])
        .groupby("selection_stratum", sort=True, as_index=False)
        .head(1)
    )
    if len(selected) > n:
        selected = selected.sort_values("selection_random_key").head(n)
    if len(selected) < n:
        remaining = out.loc[~out.index.isin(selected.index)].copy()
        counts = selected.selection_stratum.value_counts().to_dict()
        while len(selected) < n:
            remaining["stratum_count"] = remaining.selection_stratum.map(counts).fillna(0)
            row = remaining.sort_values(
                ["stratum_count", "selection_random_key", "replacement_window_index"]
            ).iloc[0]
            selected = pd.concat([selected, row.to_frame().T], axis=0)
            counts[str(row.selection_stratum)] = counts.get(str(row.selection_stratum), 0) + 1
            remaining = remaining.drop(index=row.name)
    return selected.copy()


def create_qa_figure(
    image_table: pd.DataFrame,
    trace_table: pd.DataFrame,
    selected_images: pd.DataFrame,
    selected_traces: pd.DataFrame,
    out_dir: Path,
) -> None:
    selected_images = selected_images.copy()
    selected_traces = selected_traces.copy()
    for column in (
        "corrected_reconstruction_rms_contrast",
        "corrected_reconstruction_orientation_coherence",
        "corrected_reconstruction_sf_centroid_cpd",
    ):
        selected_images[column] = pd.to_numeric(selected_images[column], errors="raise")
    for column in (
        "corrected_dpi_crop120_path_length_arcmin",
        "corrected_dpi_crop120_rms_radius_arcmin",
        "corrected_dpi_crop120_position_power_fraction_32plus_hz",
    ):
        selected_traces[column] = pd.to_numeric(selected_traces[column], errors="raise")
    image_examples = pd.concat(
        [
            selected_images.nsmallest(1, "corrected_reconstruction_rms_contrast").assign(selection_role="replacement_low_contrast"),
            selected_images.nlargest(1, "corrected_reconstruction_rms_contrast").assign(selection_role="replacement_high_contrast"),
            selected_images.nlargest(1, "corrected_reconstruction_orientation_coherence").assign(selection_role="replacement_coherent"),
            selected_images.nsmallest(1, "corrected_reconstruction_orientation_coherence").assign(selection_role="replacement_multi_orientation"),
            selected_images.nsmallest(1, "corrected_reconstruction_sf_centroid_cpd").assign(selection_role="replacement_low_sf"),
            selected_images.nlargest(1, "corrected_reconstruction_sf_centroid_cpd").assign(selection_role="replacement_high_sf"),
        ]
    ).drop_duplicates("candidate_id")
    trace_examples = pd.concat(
        [
            selected_traces.nsmallest(1, "corrected_dpi_crop120_path_length_arcmin").assign(selection_role="replacement_short_path"),
            selected_traces.nlargest(1, "corrected_dpi_crop120_path_length_arcmin").assign(selection_role="replacement_long_path"),
            selected_traces.nsmallest(1, "corrected_dpi_crop120_rms_radius_arcmin").assign(selection_role="replacement_small_radius"),
            selected_traces.nlargest(1, "corrected_dpi_crop120_rms_radius_arcmin").assign(selection_role="replacement_large_radius"),
            selected_traces.nsmallest(1, "corrected_dpi_crop120_position_power_fraction_32plus_hz").assign(selection_role="replacement_low_high_tf"),
            selected_traces.nlargest(1, "corrected_dpi_crop120_position_power_fraction_32plus_hz").assign(selection_role="replacement_high_high_tf"),
        ]
    ).drop_duplicates("replacement_window_index")
    image_examples.to_csv(out_dir / "selected_image_qa_examples.csv", index=False)
    trace_examples.to_csv(out_dir / "selected_trace_qa_examples.csv", index=False)

    fig, axes = plt.subplots(4, 6, figsize=(16, 10), constrained_layout=True)
    for column, row in enumerate(image_examples.head(6).itertuples(index=False)):
        with np.load(row.corrected_patch_npz, allow_pickle=False) as data:
            patch = np.asarray(data["corrected_patch"])
        axes[0, column].imshow(patch, cmap="gray", origin="lower")
        axes[0, column].set_title(f"{row.selection_role.replace('replacement_', '')}\nimage slot {row.image_index}", fontsize=8)
        axes[0, column].axis("off")
    for column in range(len(image_examples), 6):
        axes[0, column].axis("off")

    trace_lookup = trace_table.set_index("trace_index")
    for column, row in enumerate(trace_examples.head(6).itertuples(index=False)):
        final = trace_lookup.loc[int(row.trace_index)]
        cache: dict[Any, Any] = {}
        dset = load_dset(str(final.session), cache)
        crop = corrected_crop_xy_deg(dset)
        indices = np.arange(
            int(final.corrected_scored_global_start),
            int(final.corrected_scored_global_stop_exclusive),
            2,
        )
        xy = centered(crop[indices]) * 60.0
        axes[1, column].plot(xy[:, 0], xy[:, 1], color="#0072B2", lw=1.3)
        axes[1, column].scatter(xy[0, 0], xy[0, 1], color="#009E73", s=18)
        axes[1, column].set_aspect("equal", adjustable="datalim")
        axes[1, column].set_title(f"{row.selection_role.replace('replacement_', '')}\ntrace slot {row.trace_index}", fontsize=8)
        axes[1, column].set_xlabel("x (arcmin)", fontsize=7)
        if column == 0:
            axes[1, column].set_ylabel("y (arcmin)", fontsize=7)
        del dset
        cache.clear()
        gc.collect()
    for column in range(len(trace_examples), 6):
        axes[1, column].axis("off")

    image_metrics_plot = [
        ("corrected_reconstruction_rms_contrast", "RMS contrast"),
        ("corrected_reconstruction_orientation_coherence", "orientation coherence"),
        ("corrected_reconstruction_sf_centroid_cpd", "SF centroid (cpd)"),
    ]
    for column, (metric, label) in enumerate(image_metrics_plot):
        kept = image_table[image_table.cohort_role.eq("retained_valid_legacy_image")][metric].dropna()
        replacement = image_table[image_table.cohort_role.eq("corrected_replacement_image")][metric].dropna()
        axes[2, column].hist(kept, bins=14, alpha=0.55, label="retained 49", color="#999999")
        axes[2, column].hist(replacement, bins=14, alpha=0.65, label="replacement 51", color="#D55E00")
        axes[2, column].set_xlabel(label)
        axes[2, column].set_ylabel("images")
        axes[2, column].legend(frameon=False, fontsize=7)
    for column in range(3, 6):
        axes[2, column].axis("off")

    trace_metrics_plot = [
        ("corrected_dpi_crop120_path_length_arcmin", "path length (arcmin)"),
        ("corrected_dpi_crop120_rms_radius_arcmin", "RMS radius (arcmin)"),
        ("corrected_dpi_crop120_position_power_fraction_32plus_hz", "position power ≥32 Hz"),
    ]
    for column, (metric, label) in enumerate(trace_metrics_plot):
        kept = trace_table[trace_table.cohort_role.eq("retained_explicit_history_valid_legacy_trace")][metric].dropna()
        replacement = trace_table[trace_table.cohort_role.eq("corrected_replacement_trace")][metric].dropna()
        axes[3, column].hist(kept, bins=18, alpha=0.55, label="retained 973", color="#999999")
        axes[3, column].hist(replacement, bins=12, alpha=0.65, label="replacement 27", color="#CC79A7")
        axes[3, column].set_xlabel(label)
        axes[3, column].set_ylabel("traces")
        axes[3, column].legend(frameon=False, fontsize=7)
    axes[3, 3].axis("off")
    image_sessions = image_table.session.value_counts().sort_values(ascending=False)
    axes[3, 4].bar(np.arange(len(image_sessions)), image_sessions.values, color="#E69F00")
    axes[3, 4].set(title="Final image sessions", xlabel="session rank", ylabel="images")
    trace_sessions = trace_table.session.value_counts().sort_values(ascending=False)
    axes[3, 5].bar(np.arange(len(trace_sessions)), trace_sessions.values, color="#56B4E9")
    axes[3, 5].set(title="Final trace sessions", xlabel="session rank", ylabel="traces")
    fig.suptitle(
        "Corrected Figure 4 production cohort — input-level replacement audit\n"
        "49 retained + 51 replacement images; 973 retained + 27 replacement traces; no neural outcomes used",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(out_dir / "corrected_production_cohort_input_qa.png", dpi=190)
    fig.savefig(out_dir / "corrected_production_cohort_input_qa.pdf")
    plt.close(fig)


def assemble(args: argparse.Namespace) -> None:
    image_pool = pd.read_csv(args.out_dir / "image_candidate_pool_prevalidation.csv")
    trace_pool = pd.read_csv(args.out_dir / "trace_candidate_pool_prevalidation.csv")
    image_files = sorted((args.out_dir / "image_validation_partials").glob("*.csv"))
    trace_files = sorted((args.out_dir / "trace_validation_partials").glob("*.csv"))
    image_validation = pd.concat([pd.read_csv(path) for path in image_files], ignore_index=True)
    trace_validation = pd.concat([pd.read_csv(path) for path in trace_files], ignore_index=True)
    expected_image_sessions = set(image_pool.loc[image_pool.prevalidation_eligible.astype(bool), "session"])
    expected_trace_sessions = set(trace_pool.loc[trace_pool.prevalidation_eligible.astype(bool), "session"])
    if set(path.stem for path in image_files) != expected_image_sessions:
        raise RuntimeError("Image validation session set is incomplete")
    if set(path.stem for path in trace_files) != expected_trace_sessions:
        raise RuntimeError("Trace validation session set is incomplete")
    if image_validation.candidate_id.duplicated().any():
        raise RuntimeError("Duplicate image validation identity")
    if trace_validation.replacement_window_index.duplicated().any():
        raise RuntimeError("Duplicate trace validation identity")

    image_all = image_pool.merge(image_validation, on=["candidate_id", "source_row", "session", "trial_idx", "global_start", "global_stop"], how="left", suffixes=("", "_validation"), validate="one_to_one")
    image_all["final_rejection_reason"] = image_all.prevalidation_rejection_reason
    validated_mask = image_all.prevalidation_eligible.astype(bool)
    image_all.loc[validated_mask, "final_rejection_reason"] = image_all.loc[validated_mask, "validation_rejection_reason"].fillna("missing_session_validation")
    image_valid = image_all[image_all.validation_pass.fillna(False).astype(bool)].copy()
    if len(image_valid) < N_IMAGE_REPLACEMENTS:
        raise RuntimeError(f"Only {len(image_valid)} validated image replacements are available")
    selected_images, session_quotas = stratified_image_selection(image_valid, N_IMAGE_REPLACEMENTS, int(args.seed))
    image_all.loc[image_all.candidate_id.isin(selected_images.candidate_id), "final_rejection_reason"] = "selected_corrected_replacement_image"
    image_all.to_csv(args.out_dir / "image_candidate_pool_with_rejections.csv", index=False)

    trace_all = trace_pool.merge(trace_validation, on=["replacement_window_index", "session", "trial_idx", "corrected_history_global_start", "corrected_history_global_stop_exclusive", "corrected_scored_global_start", "corrected_scored_global_stop_exclusive"], how="left", suffixes=("", "_validation"), validate="one_to_one")
    trace_all["final_rejection_reason"] = trace_all.prevalidation_rejection_reason
    validated_trace_mask = trace_all.prevalidation_eligible.astype(bool)
    trace_all.loc[validated_trace_mask, "final_rejection_reason"] = trace_all.loc[validated_trace_mask, "validation_rejection_reason"].fillna("missing_session_validation")
    trace_valid = trace_all[trace_all.explicit_history_valid.fillna(False).astype(bool)].copy()
    if len(trace_valid) < N_TRACE_REPLACEMENTS:
        raise RuntimeError(f"Only {len(trace_valid)} validated trace replacements are available")
    selected_traces = stratified_trace_selection(trace_valid, N_TRACE_REPLACEMENTS, int(args.seed) + 1)
    trace_all.loc[trace_all.replacement_window_index.isin(selected_traces.replacement_window_index), "final_rejection_reason"] = "selected_corrected_replacement_trace"
    trace_all.to_csv(args.out_dir / "trace_candidate_pool_with_rejections.csv", index=False)

    legacy_images = pd.read_csv(args.legacy_images)
    retained_images = legacy_images[legacy_images.corrected_crop_valid.astype(bool)].copy()
    invalid_image_slots = np.sort(legacy_images.loc[~legacy_images.corrected_crop_valid.astype(bool), "image_index"].to_numpy(int))
    if len(retained_images) != 49 or len(invalid_image_slots) != N_IMAGE_REPLACEMENTS:
        raise RuntimeError("Expected 49 retained and 51 replacement image slots")
    selected_images = selected_images.sort_values(["session", "candidate_id"]).reset_index(drop=True)
    selected_images["replaces_legacy_image_index"] = invalid_image_slots
    selected_images["image_index"] = invalid_image_slots
    selected_images["cohort_role"] = "corrected_replacement_image"
    retained_images["cohort_role"] = "retained_valid_legacy_image"
    retained_images["replaces_legacy_image_index"] = np.nan
    retained_images["corrected_patch_npz"] = [
        str((LEGACY_IMAGE_PARTIALS / f"image_{int(index):03d}.npz").resolve())
        for index in retained_images.image_index
    ]
    retained_images["corrected_patch_key"] = "corrected_patch"
    final_images = pd.concat([retained_images, selected_images], ignore_index=True, sort=False).sort_values("image_index")
    if len(final_images) != 100 or final_images.image_index.nunique() != 100 or set(final_images.image_index) != set(range(100)):
        raise RuntimeError("Final image identities do not fill exactly 100 slots")

    legacy_traces = pd.read_csv(args.legacy_traces)
    retained_traces = legacy_traces[legacy_traces.explicit_history_valid.astype(bool)].copy()
    invalid_trace_slots = np.sort(legacy_traces.loc[~legacy_traces.explicit_history_valid.astype(bool), "trace_index"].to_numpy(int))
    if len(retained_traces) != 973 or len(invalid_trace_slots) != N_TRACE_REPLACEMENTS:
        raise RuntimeError("Expected 973 retained and 27 replacement trace slots")
    selected_traces = selected_traces.sort_values(["selection_stratum", "selection_random_key"]).reset_index(drop=True)
    selected_traces["replaces_legacy_trace_index"] = invalid_trace_slots
    selected_traces["trace_index"] = invalid_trace_slots
    selected_traces["cohort_role"] = "corrected_replacement_trace"
    selected_traces["inclusion_recommendation"] = "selected for corrected production cohort"
    retained_traces["cohort_role"] = "retained_explicit_history_valid_legacy_trace"
    retained_traces["replaces_legacy_trace_index"] = np.nan
    final_traces = pd.concat([retained_traces, selected_traces], ignore_index=True, sort=False).sort_values("trace_index")
    if len(final_traces) != 1000 or final_traces.trace_index.nunique() != 1000 or set(final_traces.trace_index) != set(range(1000)):
        raise RuntimeError("Final trace identities do not fill exactly 1,000 slots")
    required_validity = [
        "history_in_dataset_bounds",
        "history_and_target_same_trial",
        "history_and_target_all_dpi_valid",
        "history_and_target_all_finite",
        "explicit_history_valid",
    ]
    if not all(final_traces[column].astype(bool).all() for column in required_validity):
        raise RuntimeError("Final trace cohort failed explicit-history validity")
    final_keys = list(
        zip(
            final_traces.session.astype(str),
            final_traces.trial_idx.astype(int),
            final_traces.corrected_scored_global_start.astype(int),
            final_traces.corrected_scored_global_stop_exclusive.astype(int),
        )
    )
    if pd.Series(final_keys).duplicated().any():
        raise RuntimeError("Final trace cohort contains duplicate scored source intervals")
    image_keys = list(
        zip(
            final_images.session.astype(str),
            final_images.trial_idx.astype(int),
            final_images.global_start.fillna(-1).astype(int),
            final_images.global_stop.fillna(-1).astype(int),
        )
    )
    # Retained image rows identify intervals through source_row rather than
    # carrying global_start/global_stop. Their source rows are unique and were
    # already audited; replacement interval uniqueness is enforced here.
    replacement_keys = pd.Series(
        list(
            zip(
                selected_images.session.astype(str),
                selected_images.trial_idx.astype(int),
                selected_images.global_start.astype(int),
                selected_images.global_stop.astype(int),
            )
        )
    )
    if replacement_keys.duplicated().any():
        raise RuntimeError("Selected replacement images contain duplicate intervals")

    final_images.to_csv(args.out_dir / "corrected100_images.csv", index=False)
    final_traces.to_csv(args.out_dir / "corrected1000_traces.csv", index=False)
    selected_images.to_csv(args.out_dir / "selected51_replacement_images.csv", index=False)
    selected_traces.to_csv(args.out_dir / "selected27_replacement_traces.csv", index=False)
    pd.DataFrame(
        [{"session": session, "replacement_image_quota": quota} for session, quota in session_quotas.items()]
    ).to_csv(args.out_dir / "replacement_image_session_quotas.csv", index=False)
    create_qa_figure(final_images, final_traces, selected_images, selected_traces, args.out_dir)

    manifest = {
        "created_utc": utc_now(),
        "status": "corrected_100x1000_production_cohort_frozen",
        "selection_is_outcome_independent": True,
        "neural_model_calls": False,
        "seed": int(args.seed),
        "counts": {
            "images": 100,
            "retained_images": 49,
            "replacement_images": 51,
            "traces": 1000,
            "retained_traces": 973,
            "replacement_traces": 27,
            "cartesian_movies": 100000,
            "validated_image_candidate_pool": int(len(image_valid)),
            "validated_trace_candidate_pool": int(len(trace_valid)),
        },
        "image_selection": {
            "rule": "one per eligible session, remaining quotas proportional to sqrt(valid candidate count); greedy balance across global tertiles of corrected RMS contrast, orientation coherence, and local SF centroid; fixed random tie-break",
            "session_quotas": session_quotas,
            "required_gates": [
                "corrected_crop_valid",
                "exact_saved_stim_available",
                "unique session/trial/scored interval",
            ],
        },
        "trace_selection": {
            "rule": "balance the 3x3x3 tertile grid of corrected dpi_pix path length, RMS radius, and >=32-Hz position-power fraction; fixed random tie-break",
            "event_rule": "no event quota used; corrected event labels are retained descriptively and no legacy microsaccade label is used for selection",
            "history_contract": "32 global-even corrected dpi_pix lead-in frames + 40 global-even scored frames; same trial; all dpi_valid and finite",
        },
        "visual_contract": "corrected dpi_pix crop, session roi_src offset, global-even 120-Hz samples, retinal sign applied later by renderer",
        "outputs": {
            "images": file_identity(args.out_dir / "corrected100_images.csv"),
            "traces": file_identity(args.out_dir / "corrected1000_traces.csv"),
            "image_replacements": file_identity(args.out_dir / "selected51_replacement_images.csv"),
            "trace_replacements": file_identity(args.out_dir / "selected27_replacement_traces.csv"),
            "qa_figure": str((args.out_dir / "corrected_production_cohort_input_qa.png").resolve()),
        },
        "next_gate": "production runner dry-run, input freeze, then bounded one-block GPU preflight",
    }
    atomic_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def run_all(args: argparse.Namespace) -> None:
    command_base = [
        sys.executable,
        "-m",
        "declan.fig4_active_sensing.build_rr100_corrected_production_cohort",
        "--source-csv",
        str(args.source_csv),
        "--window-table",
        str(args.window_table),
        "--event-table",
        str(args.event_table),
        "--legacy-images",
        str(args.legacy_images),
        "--legacy-traces",
        str(args.legacy_traces),
        "--out-dir",
        str(args.out_dir),
        "--seed",
        str(args.seed),
    ]
    subprocess.run(command_base + ["--prepare"], cwd=ROOT, check=True)
    image_pool = pd.read_csv(args.out_dir / "image_candidate_pool_prevalidation.csv")
    trace_pool = pd.read_csv(args.out_dir / "trace_candidate_pool_prevalidation.csv")
    image_sessions = sorted(image_pool.loc[image_pool.prevalidation_eligible.astype(bool), "session"].unique())
    trace_sessions = sorted(trace_pool.loc[trace_pool.prevalidation_eligible.astype(bool), "session"].unique())
    for index, session in enumerate(image_sessions):
        print(f"image session worker {index + 1}/{len(image_sessions)}: {session}", flush=True)
        subprocess.run(command_base + ["--image-session", str(session)], cwd=ROOT, check=True)
    for index, session in enumerate(trace_sessions):
        print(f"trace session worker {index + 1}/{len(trace_sessions)}: {session}", flush=True)
        subprocess.run(command_base + ["--trace-session", str(session)], cwd=ROOT, check=True)
    subprocess.run(command_base + ["--assemble"], cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    if args.prepare:
        prepare_candidates(args)
    elif args.image_session:
        image_session_worker(args, str(args.image_session))
    elif args.trace_session:
        trace_session_worker(args, str(args.trace_session))
    elif args.assemble:
        assemble(args)
    elif args.all:
        run_all(args)


if __name__ == "__main__":
    main()
