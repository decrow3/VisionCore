#!/usr/bin/env python3
"""Production direct SSI for native Figure 4 image--fixation pairs.

The cohort is the 1,000 real fixation traces used by the prior Figure 4 movie
bank, restored to each trace's own reviewed BackImage window.  Every pair is
scored as the recorded trajectory plus a deterministic full-circle rotation
grid.  No image/trace Cartesian product, stabilized baseline, dose curve, or
interpolation enters this analysis.

The runner is resumable at the pair level and supports disjoint pair shards.
Unit populations are selected from each pair's local image axis before any
model response is inspected, then frozen across its real and rotated movies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import (
    _extract_patch,
    score_traces_for_patch,
)
from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.fig.ssi_figure_v2.behavior_model_bridge.run_direct_exact_pair_ssi import (
    ALIGNED_MAX_DEG,
    DT,
    EPS,
    MIN_OSI,
    N_TIMEPOINTS,
    ORTHOGONAL_MIN_DEG,
    POPULATION_LABELS,
    POPULATION_ORDER,
    PREF_COLUMN,
    OSI_COLUMN,
    SF_COLUMN,
    SF_MIN_CPD,
    _axial_delta_deg,
    _hash_array,
    _native_trace,
    _rotate_trace,
    _rotation_angles_deg,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
)
from declan.redundancy_resolved_v1_population import load_population_view


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FIG4_BANK = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
)
DEFAULT_SOURCE_WINDOWS = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
DEFAULT_UNIT_TUNING = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
    "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)
DEFAULT_OUT_DIR = ROOT / (
    "outputs/fig/ssi_figure_v2/behavior_model_bridge/"
    "panel_g_exact_pair_fig4_trace_bank_n1000_v1"
)
PAIR_METADATA_COLUMNS = (
    "session",
    "subject",
    "trial_idx",
    "global_start",
    "global_stop",
    "phase",
    "n_samples",
    "image_feature_ok",
    "image_patch_center_x_px",
    "image_patch_center_y_px",
    "image_patch_fraction_inside_image",
    "image_patch_fraction_background",
    "image_patch_rms_contrast",
    "image_gradient_energy",
    "image_edge_density",
    "image_orientation_coherence",
    "image_edge_axis_deg",
    "image_spectrum_anisotropy",
    "image_high_freq_power_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig4-bank-dir", type=Path, default=DEFAULT_FIG4_BANK)
    parser.add_argument("--source-windows", type=Path, default=DEFAULT_SOURCE_WINDOWS)
    parser.add_argument("--unit-tuning", type=Path, default=DEFAULT_UNIT_TUNING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--n-rotations", type=int, default=8)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=16)
    parser.add_argument("--trace-batch-size", type=int, default=8)
    parser.add_argument("--pair-start", type=int, default=0, help="Inclusive cohort pair index.")
    parser.add_argument("--pair-stop", type=int, default=0, help="Exclusive pair index; 0 means cohort end.")
    parser.add_argument("--force", action="store_true", help="Replace identity-matched pair caches in the selected shard.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and write the production plan without loading RR100.")
    return parser.parse_args()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_text(payload: dict[str, Any]) -> str:
    return json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))


def _population_masks(tuning: pd.DataFrame, edge_axis_deg: float) -> dict[str, np.ndarray]:
    sf = pd.to_numeric(tuning[SF_COLUMN], errors="coerce").to_numpy(dtype=float)
    pref = pd.to_numeric(tuning[PREF_COLUMN], errors="coerce").to_numpy(dtype=float)
    osi = pd.to_numeric(tuning[OSI_COLUMN], errors="coerce").to_numpy(dtype=float)
    delta = _axial_delta_deg(pref, float(edge_axis_deg))
    tuned_high = (
        np.isfinite(sf) & (sf >= SF_MIN_CPD) & np.isfinite(pref)
        & np.isfinite(osi) & (osi >= MIN_OSI)
    )
    masks = {
        "high_sf_aligned": tuned_high & (delta <= ALIGNED_MAX_DEG),
        "high_sf_orthogonal": tuned_high & (delta >= ORTHOGONAL_MIN_DEG),
        "high_sf_all": np.isfinite(sf) & (sf >= SF_MIN_CPD),
        "low_sf_all": np.isfinite(sf) & (sf < SF_MIN_CPD),
    }
    empty = [name for name, mask in masks.items() if not np.any(mask)]
    if empty:
        raise RuntimeError(f"No eligible units for edge axis {edge_axis_deg:.3f}: {empty}")
    return masks


def _build_cohort(
    bank_dir: Path,
    source_path: Path,
    tuning: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    trace_table = pd.read_csv(bank_dir / "trace_feature_table.csv")
    trace_xy = np.load(bank_dir / "trace_xy.npy")
    source = pd.read_csv(source_path)
    if "source_row" not in source.columns:
        source = source.copy()
        source["source_row"] = np.arange(len(source), dtype=int)
    if len(trace_table) != len(trace_xy):
        raise RuntimeError("Figure 4 trace table and trace_xy.npy have different lengths")
    if trace_table["source_row"].astype(int).duplicated().any():
        raise RuntimeError("Figure 4 trace cohort contains repeated source rows")
    source_by_id = source.set_index(source["source_row"].astype(int), drop=False)
    cohort_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    max_trace_error = 0.0
    for pair_index, trace_row in trace_table.iterrows():
        source_row_id = int(trace_row["source_row"])
        if source_row_id not in source_by_id.index:
            raise RuntimeError(f"Missing source row {source_row_id} for trace bank index {pair_index}")
        source_row = source_by_id.loc[source_row_id]
        if isinstance(source_row, pd.DataFrame):
            raise RuntimeError(f"Duplicate source row ID {source_row_id}")
        native = _native_trace(source_row)
        trace_error = float(np.max(np.abs(native - np.asarray(trace_xy[pair_index], dtype=np.float32))))
        max_trace_error = max(max_trace_error, trace_error)
        if trace_error > 1e-6:
            raise RuntimeError(f"Saved Figure 4 trace {pair_index} differs from native reconstruction: {trace_error}")
        if str(source_row["session"]) != str(trace_row["session"]) or int(source_row["trial_idx"]) != int(trace_row["trial_idx"]):
            raise RuntimeError(f"Source identity mismatch for trace bank index {pair_index}")
        edge_axis = float(source_row["image_edge_axis_deg"])
        if not math.isfinite(edge_axis):
            raise RuntimeError(f"Non-finite local edge axis for pair {pair_index}")
        masks = _population_masks(tuning, edge_axis)
        row: dict[str, Any] = {
            "pair_index": int(pair_index),
            "trace_bank_index": int(trace_row["trace_bank_index"]),
            "source_row": source_row_id,
            "trace_hash": _hash_array(native),
            "native_trace_max_abs_error": trace_error,
            "rendered_path_length_arcmin": float(trace_row["rendered_path_length_arcmin"]),
            "rendered_rms_radius_arcmin": float(trace_row["rendered_rms_radius_arcmin"]),
            "rendered_n_microsaccade_events": int(trace_row["rendered_n_microsaccade_events"]),
        }
        for column in PAIR_METADATA_COLUMNS:
            if column in source_row.index:
                row[column] = source_row[column]
        if "subject" not in row:
            row["subject"] = str(source_row["session"]).split("_", maxsplit=1)[0]
        for population, mask in masks.items():
            row[f"n_units_{population}"] = int(np.sum(mask))
            for unit_index in np.flatnonzero(mask):
                membership_rows.append(
                    {
                        "pair_index": int(pair_index),
                        "population": population,
                        "population_label": POPULATION_LABELS[population],
                        "unit_index": int(unit_index),
                        "local_edge_axis_deg": edge_axis,
                        "selection_frozen_before_model_evaluation": True,
                    }
                )
        cohort_rows.append(row)
    cohort = pd.DataFrame(cohort_rows)
    cohort.attrs["max_trace_error"] = max_trace_error
    return cohort, np.asarray(trace_xy, dtype=np.float32), pd.DataFrame(membership_rows)


def _pair_identity(
    pair: pd.Series,
    patch: np.ndarray,
    trace: np.ndarray,
    rotation_angles: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pair_index": int(pair["pair_index"]),
        "source_row": int(pair["source_row"]),
        "session": str(pair["session"]),
        "trial_idx": int(pair["trial_idx"]),
        "patch_center_x_px": float(pair["image_patch_center_x_px"]),
        "patch_center_y_px": float(pair["image_patch_center_y_px"]),
        "patch_size_px": int(args.patch_size_px),
        "patch_hash": _hash_array(np.asarray(patch, dtype=np.float32)),
        "trace_hash": _hash_array(trace),
        "n_timepoints": int(trace.shape[0]),
        "rotation_angles_deg": rotation_angles,
        "rotation_contract": "deterministic full-circle midpoint grid with antipodal pairs; centroid-preserving trace rotation",
        "rr100_version": str(args.rr100_version),
        "stimulus_normalization": "standardize_uint_like_then_minus_127_div_255",
        "ssi_contract": "instantaneous spatial SSI weighted by predicted expected spikes over 40 native samples",
    }


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _assemble_shard(
    shard_pairs: pd.DataFrame,
    membership: pd.DataFrame,
    cache_paths: dict[int, Path],
    rotation_angles: np.ndarray,
    shard_dir: Path,
) -> None:
    condition_ids = ["real", *[f"rotation_{index:02d}" for index in range(len(rotation_angles))]]
    condition_angles = np.asarray([np.nan, *rotation_angles], dtype=np.float32)
    unit_bits: list[np.ndarray] = []
    unit_information: list[np.ndarray] = []
    unit_expected: list[np.ndarray] = []
    unit_rates: list[np.ndarray] = []
    population_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    membership_lookup = {
        (int(pair_index), str(population)): group["unit_index"].astype(int).to_numpy()
        for (pair_index, population), group in membership.groupby(["pair_index", "population"], sort=False)
    }
    for pair in shard_pairs.itertuples(index=False):
        pair_index = int(pair.pair_index)
        cache = _load_cache(cache_paths[pair_index])
        bits = cache["unit_bits_per_spike"]
        information = cache["unit_information_bits"]
        expected = cache["unit_expected_spikes"]
        rates = cache["unit_mean_rate"]
        unit_bits.append(bits)
        unit_information.append(information)
        unit_expected.append(expected)
        unit_rates.append(rates)
        for population in POPULATION_ORDER:
            indices = membership_lookup[(pair_index, population)]
            condition_values: list[dict[str, float]] = []
            for condition_index, (condition_id, angle) in enumerate(zip(condition_ids, condition_angles)):
                pop_information = float(np.sum(information[condition_index, indices]))
                pop_expected = float(np.sum(expected[condition_index, indices]))
                values = {
                    "bits_per_spike": pop_information / max(pop_expected, EPS),
                    "information_bits": pop_information,
                    "information_bits_per_sample": pop_information / N_TIMEPOINTS,
                    "expected_spikes": pop_expected,
                    "expected_spikes_per_sample": pop_expected / N_TIMEPOINTS,
                    "mean_rate": pop_expected / (N_TIMEPOINTS * DT),
                }
                condition_values.append(values)
                population_rows.append(
                    {
                        "pair_index": pair_index,
                        "source_row": int(pair.source_row),
                        "session": str(pair.session),
                        "subject": str(pair.subject),
                        "trial_idx": int(pair.trial_idx),
                        "phase": str(pair.phase),
                        "image_orientation_coherence": float(pair.image_orientation_coherence),
                        "image_edge_axis_deg": float(pair.image_edge_axis_deg),
                        "condition_index": condition_index,
                        "condition_id": condition_id,
                        "condition_kind": "real" if condition_index == 0 else "rotation",
                        "rotation_angle_deg": float(angle),
                        "population": population,
                        "population_label": POPULATION_LABELS[population],
                        "n_units": int(indices.size),
                        "fresh_model_evaluation": True,
                        **values,
                    }
                )
            real = condition_values[0]
            rotations = condition_values[1:]
            contrast: dict[str, Any] = {
                "pair_index": pair_index,
                "source_row": int(pair.source_row),
                "session": str(pair.session),
                "subject": str(pair.subject),
                "trial_idx": int(pair.trial_idx),
                "phase": str(pair.phase),
                "image_orientation_coherence": float(pair.image_orientation_coherence),
                "image_edge_axis_deg": float(pair.image_edge_axis_deg),
                "rendered_path_length_arcmin": float(pair.rendered_path_length_arcmin),
                "rendered_rms_radius_arcmin": float(pair.rendered_rms_radius_arcmin),
                "rendered_n_microsaccade_events": int(pair.rendered_n_microsaccade_events),
                "population": population,
                "population_label": POPULATION_LABELS[population],
                "n_units": int(indices.size),
                "n_rotations": int(len(rotations)),
            }
            for metric in ("bits_per_spike", "information_bits_per_sample", "expected_spikes_per_sample", "mean_rate"):
                rotation_values = np.asarray([row[metric] for row in rotations], dtype=float)
                contrast[f"real_{metric}"] = float(real[metric])
                contrast[f"rotation_mean_{metric}"] = float(np.mean(rotation_values))
                contrast[f"real_minus_rotation_{metric}"] = float(real[metric] - np.mean(rotation_values))
                contrast[f"rotation_min_{metric}"] = float(np.min(rotation_values))
                contrast[f"rotation_max_{metric}"] = float(np.max(rotation_values))
                contrast[f"fraction_rotations_below_real_{metric}"] = float(np.mean(rotation_values < real[metric]))
            contrast_rows.append(contrast)
    np.savez_compressed(
        shard_dir / "direct_pair_unit_metrics.npz",
        pair_index=shard_pairs["pair_index"].astype(np.int32).to_numpy(),
        condition_id=np.asarray(condition_ids),
        rotation_angle_deg=condition_angles,
        unit_bits_per_spike=np.stack(unit_bits).astype(np.float32),
        unit_information_bits=np.stack(unit_information).astype(np.float32),
        unit_expected_spikes=np.stack(unit_expected).astype(np.float32),
        unit_mean_rate=np.stack(unit_rates).astype(np.float32),
    )
    pd.DataFrame(population_rows).to_csv(shard_dir / "direct_pair_population_metrics.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(shard_dir / "direct_pair_rotation_contrasts.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_dir = args.fig4_bank_dir.resolve()
    source_path = args.source_windows.resolve()
    tuning_path = args.unit_tuning.resolve()
    tuning = pd.read_csv(tuning_path).sort_values("unit_index").reset_index(drop=True)
    if not np.array_equal(tuning["unit_index"].astype(int).to_numpy(), np.arange(len(tuning))):
        raise RuntimeError("Unit tuning table must contain contiguous unit_index values")
    rotation_angles = _rotation_angles_deg(int(args.n_rotations))
    print("[panel-g-production] validating Figure 4 trace cohort and native pairing", flush=True)
    cohort, trace_xy, membership = _build_cohort(bank_dir, source_path, tuning)
    cohort_manifest_path = out_dir / "exact_pair_cohort_manifest.csv"
    membership_path = out_dir / "exact_pair_unit_population_membership.csv"
    if cohort_manifest_path.exists():
        existing = pd.read_csv(cohort_manifest_path)
        identity_columns = ["pair_index", "source_row", "trace_hash"]
        if not existing[identity_columns].equals(cohort[identity_columns]):
            raise RuntimeError(f"Existing cohort identity differs from current inputs: {cohort_manifest_path}")
    else:
        cohort.to_csv(cohort_manifest_path, index=False)
    if membership_path.exists():
        existing = pd.read_csv(membership_path)
        identity_columns = ["pair_index", "population", "unit_index"]
        if not existing[identity_columns].equals(membership[identity_columns]):
            raise RuntimeError(f"Existing unit membership differs from current inputs: {membership_path}")
    else:
        membership.to_csv(membership_path, index=False)
    cohort_size = len(cohort)
    pair_start = max(0, int(args.pair_start))
    pair_stop = cohort_size if int(args.pair_stop) <= 0 else min(cohort_size, int(args.pair_stop))
    if pair_start >= pair_stop:
        raise ValueError(f"Empty pair shard [{pair_start}, {pair_stop}) for cohort size {cohort_size}")
    shard_pairs = cohort.iloc[pair_start:pair_stop].copy().reset_index(drop=True)
    shard_name = f"pairs_{pair_start:06d}_{pair_stop:06d}"
    shard_dir = out_dir / "shards" / shard_name
    cache_dir = shard_dir / "cache"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_pairs.to_csv(shard_dir / "exact_pair_shard_manifest.csv", index=False)
    source = pd.read_csv(source_path)
    if "source_row" not in source.columns:
        source = source.copy()
        source["source_row"] = np.arange(len(source), dtype=int)
    source_by_id = source.set_index(source["source_row"].astype(int), drop=False)
    old_summary = json.loads((bank_dir / "summary.json").read_text())
    historical_seconds_per_movie = float(
        np.mean([float(item["pilot"]["seconds_per_movie"]) for item in old_summary["shard_summaries"]])
    )
    plan = {
        "analysis": "panel_g_exact_native_pair_production",
        "artifact_type": "production_preflight" if args.dry_run else "production_shard",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "fig4_bank_dir": bank_dir,
            "source_windows": source_path,
            "unit_tuning": tuning_path,
            "trace_feature_table_sha256": _file_sha256(bank_dir / "trace_feature_table.csv"),
            "trace_xy_sha256": _file_sha256(bank_dir / "trace_xy.npy"),
        },
        "cohort": {
            "contract": "same 1,000 fixation traces as prior Figure 4 bank, each restored to its own source image window",
            "n_pairs_total": cohort_size,
            "n_unique_source_rows": int(cohort["source_row"].nunique()),
            "max_saved_vs_native_trace_abs_error": float(cohort.attrs["max_trace_error"]),
            "n_microsaccade_pairs": int((cohort["rendered_n_microsaccade_events"] > 0).sum()),
        },
        "shard": {
            "pair_start": pair_start,
            "pair_stop": pair_stop,
            "n_pairs": len(shard_pairs),
            "n_conditions_per_pair": 1 + len(rotation_angles),
            "n_fresh_movies": len(shard_pairs) * (1 + len(rotation_angles)),
        },
        "model": {
            "rr100_version": str(args.rr100_version),
            "device": str(args.device),
            "patch_size_px": int(args.patch_size_px),
            "frame_batch_size": int(args.frame_batch_size),
            "trace_batch_size": int(args.trace_batch_size),
        },
        "trace": {
            "n_timepoints": N_TIMEPOINTS,
            "dt_s": DT,
            "contract": "unscaled, mean-centered, native central 40-sample source trace; no temporal compression",
        },
        "rotation": {
            "angles_deg": rotation_angles,
            "contract": "eight-angle deterministic full-circle midpoint grid with antipodal pairs",
        },
        "ssi": {
            "contract": "direct time-resolved spatial SSI with expected-spike weighting",
            "primary_estimand": "within-pair real minus mean of freshly evaluated trajectory rotations",
            "dose_curve_interpolation_used": False,
            "stabilized_baseline_used": False,
        },
        "historical_runtime_reference": {
            "source": bank_dir / "summary.json",
            "seconds_per_movie": historical_seconds_per_movie,
            "estimated_shard_gpu_hours": len(shard_pairs) * (1 + len(rotation_angles)) * historical_seconds_per_movie / 3600.0,
            "caveat": "old matrix batching reused each image across 1,000 traces; exact-pair patch turnover may be slower",
        },
        "git_revision": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
    }
    _write_json(shard_dir / "run_plan.json", plan)
    if args.dry_run:
        print(
            f"[panel-g-production] dry run: {len(shard_pairs)} exact pairs x "
            f"{1 + len(rotation_angles)} conditions = {plan['shard']['n_fresh_movies']} fresh movies; "
            f"historical lower-bound estimate {plan['historical_runtime_reference']['estimated_shard_gpu_hours']:.2f} GPU h",
            flush=True,
        )
        return

    population_view = load_population_view(version_name=str(args.rr100_version))
    scorer = CanonicalTwinScorer(
        device=str(args.device), batch_size=int(args.frame_batch_size), empty_cache_every_batch=False
    )
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    cache_paths: dict[int, Path] = {}
    n_fresh = 0
    n_reused = 0
    started = time.perf_counter()
    for shard_ordinal, pair_tuple in enumerate(shard_pairs.itertuples(index=False), start=1):
        pair = pd.Series(pair_tuple._asdict())
        pair_index = int(pair["pair_index"])
        source_row = source_by_id.loc[int(pair["source_row"])]
        patch, _patch_meta = _extract_patch(
            source_row,
            canvas_cache=canvas_cache,
            patch_size_px=int(args.patch_size_px),
        )
        trace = np.asarray(trace_xy[pair_index], dtype=np.float32)
        identity = _pair_identity(pair, patch, trace, rotation_angles, args)
        identity_text = _identity_text(identity)
        cache_path = cache_dir / f"pair_{pair_index:06d}.npz"
        cache_paths[pair_index] = cache_path
        if cache_path.exists() and not args.force:
            cached = _load_cache(cache_path)
            cached_identity = str(cached.get("cache_identity_json", np.asarray([""]))[0])
            if cached_identity != identity_text:
                raise RuntimeError(f"Cache identity mismatch for {cache_path}; pass --force to replace")
            n_reused += 1
            print(f"[panel-g-production] {shard_ordinal}/{len(shard_pairs)} reuse pair {pair_index}", flush=True)
            continue
        traces = [trace, *[_rotate_trace(trace, angle) for angle in rotation_angles]]
        bits, expected, rates, _population = score_traces_for_patch(
            scorer,
            population_view,
            patch,
            traces,
            trace_batch_size=int(args.trace_batch_size),
            frame_batch_size=int(args.frame_batch_size),
            n_timepoints=N_TIMEPOINTS,
            bin_seconds=DT,
        )
        np.savez_compressed(
            cache_path,
            cache_identity_json=np.asarray([identity_text]),
            condition_id=np.asarray(["real", *[f"rotation_{index:02d}" for index in range(len(rotation_angles))]]),
            rotation_angle_deg=np.asarray([np.nan, *rotation_angles], dtype=np.float32),
            unit_bits_per_spike=bits.astype(np.float32),
            unit_information_bits=(bits * expected).astype(np.float32),
            unit_expected_spikes=expected.astype(np.float32),
            unit_mean_rate=rates.astype(np.float32),
        )
        n_fresh += 1
        elapsed = time.perf_counter() - started
        print(
            f"[panel-g-production] {shard_ordinal}/{len(shard_pairs)} fresh pair {pair_index}; "
            f"elapsed={elapsed / 60.0:.1f} min",
            flush=True,
        )
    print("[panel-g-production] assembling shard tables", flush=True)
    shard_membership = membership[membership["pair_index"].astype(int).isin(shard_pairs["pair_index"].astype(int))]
    _assemble_shard(shard_pairs, shard_membership, cache_paths, rotation_angles, shard_dir)
    elapsed = time.perf_counter() - started
    plan.update(
        {
            "artifact_type": "production_shard_complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "execution": {
                "elapsed_s": elapsed,
                "n_pairs_fresh": n_fresh,
                "n_pairs_reused": n_reused,
                "movies_per_s_fresh": n_fresh * (1 + len(rotation_angles)) / max(elapsed, 1e-12),
            },
        }
    )
    _write_json(shard_dir / "run_metadata.json", plan)
    print(f"[panel-g-production] wrote completed shard to {shard_dir}", flush=True)


if __name__ == "__main__":
    main()
