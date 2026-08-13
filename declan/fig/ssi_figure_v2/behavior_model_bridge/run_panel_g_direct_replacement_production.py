#!/usr/bin/env python3
"""Fresh direct SSI evaluation for the frozen replacement Panel G cohort.

Every retained native image--fixation pair is evaluated as the recorded trace
and a deterministic full-circle rotation grid.  The primary aligned and
orthogonal high-SF populations use the corrected image-array contour axis;
the historical gaze-axis masks are retained only as labeled diagnostics.
No dose curve, interpolation, or stabilized baseline enters the analysis.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
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
    EPS,
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.fig.ssi_figure_v2.behavior_model_bridge.run_direct_exact_pair_ssi import (
    DT,
    N_TIMEPOINTS,
    OSI_COLUMN,
    PREF_COLUMN,
    SF_COLUMN,
    _axial_delta_deg,
    _hash_array,
    _native_trace,
    _rotate_trace,
    _rotation_angles_deg,
)
from declan.fig.ssi_figure_v2.behavior_model_bridge.run_panel_g_original_matrix_pair_rotation_audit import (
    POPULATION_LABELS,
    POPULATION_ORDER,
    PRIMARY_METRICS,
    _population_masks,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
)
from declan.redundancy_resolved_v1_population import load_population_view


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SELECTION = ROOT / (
    "outputs/fig/ssi_figure_v2/behavior_model_bridge/"
    "panel_g_direct_replacement_strong_contour_v1/frozen_confirmation_cohort.csv"
)
DEFAULT_BANK_DIR = ROOT / (
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
    "panel_g_direct_replacement_strong_contour_v1/direct_n32_gpu0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    parser.add_argument("--source-windows", type=Path, default=DEFAULT_SOURCE_WINDOWS)
    parser.add_argument("--unit-tuning", type=Path, default=DEFAULT_UNIT_TUNING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--n-rotations", type=int, default=32)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=16)
    parser.add_argument("--trace-batch-size", type=int, default=8)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-stop", type=int, default=0, help="Exclusive; zero means cohort end")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _identity_text(payload: dict[str, Any]) -> str:
    return json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _validate_selection(
    selection_path: Path,
    bank_dir: Path,
    source_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    selection = pd.read_csv(selection_path)
    required = {
        "frozen_cohort_index", "pair_index", "trace_bank_index", "source_row",
        "session", "trial_idx", "image_edge_axis_deg", "corrected_edge_axis_array_deg",
        "image_orientation_coherence", "stimulus_image_sha256", "local_window_cluster_id",
        "confirmation_eligible", "selection_uses_historical_surrogate",
        "selection_uses_fresh_model_outcome", "selection_frozen_before_model_evaluation",
    }
    missing = sorted(required - set(selection.columns))
    if missing:
        raise RuntimeError(f"Frozen selection is missing required columns: {missing}")
    selection = selection[selection["confirmation_eligible"].astype(bool)].copy()
    selection = selection.sort_values("frozen_cohort_index", kind="stable").reset_index(drop=True)
    selection.insert(0, "confirmation_index", np.arange(len(selection), dtype=int))
    if selection["pair_index"].astype(int).duplicated().any():
        raise RuntimeError("Frozen confirmation cohort contains duplicate pair_index values")
    if selection["selection_uses_historical_surrogate"].astype(bool).any():
        raise RuntimeError("Frozen selection unexpectedly used historical surrogate outcomes")
    if selection["selection_uses_fresh_model_outcome"].astype(bool).any():
        raise RuntimeError("Frozen selection unexpectedly used fresh model outcomes")
    if not selection["selection_frozen_before_model_evaluation"].astype(bool).all():
        raise RuntimeError("Frozen-before-evaluation flag is not true for every retained pair")

    traces = np.load(bank_dir / "trace_xy.npy")
    source = pd.read_csv(source_path).copy()
    if "source_row" not in source.columns:
        source.insert(0, "source_row", np.arange(len(source), dtype=int))
    source_by_id = source.set_index(source["source_row"].astype(int), drop=False)
    max_error = 0.0
    for row in selection.itertuples(index=False):
        trace_index = int(row.trace_bank_index)
        pair_index = int(row.pair_index)
        if trace_index != pair_index:
            raise RuntimeError(f"pair {pair_index} does not match trace bank index {trace_index}")
        if trace_index < 0 or trace_index >= len(traces):
            raise RuntimeError(f"Trace index outside bank for pair {pair_index}: {trace_index}")
        source_row_id = int(row.source_row)
        if source_row_id not in source_by_id.index:
            raise RuntimeError(f"Missing source row {source_row_id} for pair {pair_index}")
        native = _native_trace(source_by_id.loc[source_row_id])
        error = float(np.max(np.abs(native - np.asarray(traces[trace_index], dtype=np.float32))))
        max_error = max(max_error, error)
        if error > 1e-6:
            raise RuntimeError(f"Native trace mismatch for pair {pair_index}: {error}")
        expected_array = (-float(row.image_edge_axis_deg)) % 180.0
        axis_error = float(_axial_delta_deg(expected_array, float(row.corrected_edge_axis_array_deg)))
        if axis_error > 1e-6:
            raise RuntimeError(f"Gaze/array contour-axis mismatch for pair {pair_index}: {axis_error}")
    selection.attrs["max_native_trace_error"] = max_error
    return selection, np.asarray(traces, dtype=np.float32), source_by_id


def _freeze_membership(selection: pd.DataFrame, tuning: pd.DataFrame) -> pd.DataFrame:
    sf = pd.to_numeric(tuning[SF_COLUMN], errors="coerce").to_numpy(dtype=float)
    pref = pd.to_numeric(tuning[PREF_COLUMN], errors="coerce").to_numpy(dtype=float)
    osi = pd.to_numeric(tuning[OSI_COLUMN], errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for pair in selection.itertuples(index=False):
        gaze_axis = float(pair.image_edge_axis_deg)
        array_axis = float(pair.corrected_edge_axis_array_deg)
        masks = _population_masks(tuning, gaze_axis, array_axis)
        for population in POPULATION_ORDER:
            mask_axis = (
                gaze_axis if "historical_gaze_axis" in population
                else array_axis if "corrected_array_axis" in population
                else np.nan
            )
            deltas = _axial_delta_deg(pref, mask_axis) if np.isfinite(mask_axis) else np.full_like(pref, np.nan)
            for unit_index in np.flatnonzero(masks[population]):
                rows.append(
                    {
                        "confirmation_index": int(pair.confirmation_index),
                        "pair_index": int(pair.pair_index),
                        "population": population,
                        "population_label": POPULATION_LABELS[population],
                        "unit_index": int(unit_index),
                        "unit_sf_cpd": float(sf[unit_index]),
                        "unit_preferred_orientation_deg": float(pref[unit_index]),
                        "unit_orientation_selectivity_index": float(osi[unit_index]),
                        "mask_axis_deg": float(mask_axis),
                        "unit_axis_delta_deg": float(deltas[unit_index]),
                        "selection_frozen_before_fresh_model_evaluation": True,
                    }
                )
    return pd.DataFrame(rows)


def _assemble(
    shard_pairs: pd.DataFrame,
    membership: pd.DataFrame,
    cache_paths: dict[int, Path],
    out_dir: Path,
) -> None:
    lookup = {
        (int(pair_index), str(population)): group["unit_index"].astype(int).to_numpy()
        for (pair_index, population), group in membership.groupby(["pair_index", "population"], sort=False)
    }
    unit_bits: list[np.ndarray] = []
    unit_information: list[np.ndarray] = []
    unit_expected: list[np.ndarray] = []
    unit_rates: list[np.ndarray] = []
    condition_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for pair in shard_pairs.itertuples(index=False):
        pair_index = int(pair.pair_index)
        cache = _load_cache(cache_paths[pair_index])
        bits = cache["unit_bits_per_spike"]
        information = cache["unit_information_bits"]
        expected = cache["unit_expected_spikes"]
        rates = cache["unit_mean_rate"]
        unit_bits.append(bits); unit_information.append(information)
        unit_expected.append(expected); unit_rates.append(rates)
        for population in POPULATION_ORDER:
            units = lookup[(pair_index, population)]
            values: list[dict[str, float]] = []
            for condition_index, condition_id in enumerate(cache["condition_id"].astype(str)):
                pop_information = float(np.sum(information[condition_index, units]))
                pop_expected = float(np.sum(expected[condition_index, units]))
                row_values = {
                    "bits_per_spike": pop_information / max(pop_expected, EPS),
                    "information_bits_per_sample": pop_information / N_TIMEPOINTS,
                    "expected_spikes_per_sample": pop_expected / N_TIMEPOINTS,
                    "mean_rate": pop_expected / (N_TIMEPOINTS * DT),
                }
                values.append(row_values)
                condition_rows.append(
                    {
                        "confirmation_index": int(pair.confirmation_index),
                        "pair_index": pair_index,
                        "source_row": int(pair.source_row),
                        "session": str(pair.session),
                        "subject": str(pair.subject),
                        "trial_idx": int(pair.trial_idx),
                        "stimulus_image_sha256": str(pair.stimulus_image_sha256),
                        "local_window_cluster_id": str(pair.local_window_cluster_id),
                        "image_orientation_coherence": float(pair.image_orientation_coherence),
                        "image_edge_axis_gaze_deg": float(pair.image_edge_axis_deg),
                        "image_edge_axis_array_deg": float(pair.corrected_edge_axis_array_deg),
                        "population": population,
                        "population_label": POPULATION_LABELS[population],
                        "n_units": int(units.size),
                        "condition_index": int(condition_index),
                        "condition_id": condition_id,
                        "condition_kind": "real" if condition_index == 0 else "rotation",
                        "rotation_angle_deg": float(cache["rotation_angle_deg"][condition_index]),
                        "fresh_model_evaluation": True,
                        **row_values,
                    }
                )
            real = values[0]
            rotations = values[1:]
            contrast: dict[str, Any] = {
                "confirmation_index": int(pair.confirmation_index),
                "pair_index": pair_index,
                "source_row": int(pair.source_row),
                "session": str(pair.session),
                "subject": str(pair.subject),
                "trial_idx": int(pair.trial_idx),
                "stimulus_image_sha256": str(pair.stimulus_image_sha256),
                "local_window_cluster_id": str(pair.local_window_cluster_id),
                "image_orientation_coherence": float(pair.image_orientation_coherence),
                "image_edge_axis_gaze_deg": float(pair.image_edge_axis_deg),
                "image_edge_axis_array_deg": float(pair.corrected_edge_axis_array_deg),
                "population": population,
                "population_label": POPULATION_LABELS[population],
                "n_units": int(units.size),
                "n_rotations": int(len(rotations)),
                "fresh_model_evaluation": True,
            }
            for metric in PRIMARY_METRICS:
                rotation_values = np.asarray([item[metric] for item in rotations], dtype=float)
                contrast[f"real_{metric}"] = float(real[metric])
                contrast[f"rotation_mean_{metric}"] = float(np.mean(rotation_values))
                contrast[f"real_minus_rotation_{metric}"] = float(real[metric] - np.mean(rotation_values))
                contrast[f"rotation_min_{metric}"] = float(np.min(rotation_values))
                contrast[f"rotation_max_{metric}"] = float(np.max(rotation_values))
                contrast[f"fraction_rotations_below_real_{metric}"] = float(np.mean(rotation_values < real[metric]))
            contrast_rows.append(contrast)

    np.savez_compressed(
        out_dir / "direct_pair_unit_metrics.npz",
        pair_index=shard_pairs["pair_index"].astype(np.int32).to_numpy(),
        confirmation_index=shard_pairs["confirmation_index"].astype(np.int32).to_numpy(),
        condition_id=_load_cache(next(iter(cache_paths.values())))["condition_id"],
        rotation_angle_deg=_load_cache(next(iter(cache_paths.values())))["rotation_angle_deg"],
        unit_bits_per_spike=np.stack(unit_bits).astype(np.float32),
        unit_information_bits=np.stack(unit_information).astype(np.float32),
        unit_expected_spikes=np.stack(unit_expected).astype(np.float32),
        unit_mean_rate=np.stack(unit_rates).astype(np.float32),
    )
    pd.DataFrame(condition_rows).to_csv(out_dir / "direct_pair_population_rotation_curves.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(out_dir / "direct_pair_rotation_contrasts.csv", index=False)


def main() -> None:
    args = parse_args()
    if str(args.device) != "cuda:0":
        raise ValueError("This frozen replacement run is restricted to --device cuda:0")
    selection_path = args.selection.resolve()
    bank_dir = args.bank_dir.resolve()
    source_path = args.source_windows.resolve()
    tuning_path = args.unit_tuning.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selection, trace_xy, source_by_id = _validate_selection(selection_path, bank_dir, source_path)
    tuning = pd.read_csv(tuning_path).sort_values("unit_index").reset_index(drop=True)
    if not np.array_equal(tuning["unit_index"].astype(int).to_numpy(), np.arange(len(tuning))):
        raise RuntimeError("Unit tuning table must contain contiguous unit_index values")
    membership = _freeze_membership(selection, tuning)
    selection.to_csv(out_dir / "frozen_confirmation_cohort_used.csv", index=False)
    membership.to_csv(out_dir / "direct_unit_population_membership.csv", index=False)

    pair_start = max(0, int(args.pair_start))
    pair_stop = len(selection) if int(args.pair_stop) <= 0 else min(len(selection), int(args.pair_stop))
    if pair_start >= pair_stop:
        raise ValueError(f"Empty shard [{pair_start}, {pair_stop}) for cohort size {len(selection)}")
    shard_pairs = selection.iloc[pair_start:pair_stop].copy().reset_index(drop=True)
    shard_dir = out_dir / "shards" / f"frozen_{pair_start:06d}_{pair_stop:06d}"
    cache_dir = shard_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_pairs.to_csv(shard_dir / "frozen_confirmation_shard_manifest.csv", index=False)

    rotation_angles = _rotation_angles_deg(int(args.n_rotations))
    old_summary = json.loads((bank_dir / "summary.json").read_text())
    seconds_per_movie = float(np.mean([
        float(item["pilot"]["seconds_per_movie"]) for item in old_summary["shard_summaries"]
    ]))
    plan = {
        "analysis": "panel_g_direct_replacement_strong_contour_confirmation",
        "artifact_type": "production_preflight" if args.dry_run else "production_shard",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "exact_native_image_fixation_pairs": True,
            "outcome_blind_selection": True,
            "historical_surrogate_used": False,
            "dose_curve_interpolation_used": False,
            "stabilized_baseline_used": False,
            "primary_population": "high_sf_aligned_corrected_array_axis",
            "historical_gaze_axis_population_is_diagnostic_only": True,
        },
        "inputs": {
            "selection": selection_path,
            "selection_sha256": _file_sha256(selection_path),
            "trace_xy": bank_dir / "trace_xy.npy",
            "trace_xy_sha256": _file_sha256(bank_dir / "trace_xy.npy"),
            "source_windows": source_path,
            "source_windows_sha256": _file_sha256(source_path),
            "unit_tuning": tuning_path,
            "unit_tuning_sha256": _file_sha256(tuning_path),
        },
        "cohort": {
            "n_pairs_total": int(len(selection)),
            "n_stimulus_images": int(selection["stimulus_image_sha256"].nunique()),
            "n_local_window_clusters": int(selection["local_window_cluster_id"].nunique()),
            "n_sessions": int(selection["session"].nunique()),
            "n_subjects": int(selection["subject"].nunique()),
            "max_native_trace_error": float(selection.attrs["max_native_trace_error"]),
        },
        "shard": {
            "pair_start": pair_start,
            "pair_stop": pair_stop,
            "n_pairs": int(len(shard_pairs)),
            "n_conditions_per_pair": int(1 + len(rotation_angles)),
            "n_fresh_movies": int(len(shard_pairs) * (1 + len(rotation_angles))),
        },
        "model": {
            "rr100_version": str(args.rr100_version),
            "device": str(args.device),
            "cuda_visible_devices_required": "0",
            "patch_size_px": int(args.patch_size_px),
            "frame_batch_size": int(args.frame_batch_size),
            "trace_batch_size": int(args.trace_batch_size),
        },
        "trace": {
            "n_timepoints": N_TIMEPOINTS,
            "dt_s": DT,
            "contract": "unscaled mean-centered native central 40 samples; no temporal compression",
        },
        "rotation": {
            "n_rotations": int(len(rotation_angles)),
            "angles_deg": rotation_angles,
            "contract": "deterministic full-circle midpoint grid with antipodal pairs; identity excluded",
        },
        "ssi": {
            "contract": "direct instantaneous spatial SSI weighted by predicted expected spikes",
            "primary_estimand": "within-pair real minus mean of 32 freshly evaluated rotations",
            "companion_metrics": ["information_bits_per_sample", "expected_spikes_per_sample", "mean_rate"],
        },
        "historical_runtime_reference": {
            "seconds_per_movie": seconds_per_movie,
            "estimated_gpu_hours": len(shard_pairs) * (1 + len(rotation_angles)) * seconds_per_movie / 3600.0,
        },
        "git_revision": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
    }
    _write_json(shard_dir / "run_plan.json", plan)
    if args.dry_run:
        print(
            f"[panel-g-replacement] dry run: {len(shard_pairs)} pairs x "
            f"{1 + len(rotation_angles)} conditions = {plan['shard']['n_fresh_movies']} fresh movies; "
            f"historical lower-bound {plan['historical_runtime_reference']['estimated_gpu_hours']:.2f} GPU h",
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
    for ordinal, pair_tuple in enumerate(shard_pairs.itertuples(index=False), start=1):
        pair = pd.Series(pair_tuple._asdict())
        pair_index = int(pair["pair_index"])
        source_row = source_by_id.loc[int(pair["source_row"])]
        patch, _patch_meta = _extract_patch(
            source_row, canvas_cache=canvas_cache, patch_size_px=int(args.patch_size_px)
        )
        trace = np.asarray(trace_xy[int(pair["trace_bank_index"])], dtype=np.float32)
        identity = {
            "schema_version": 1,
            "analysis": "panel_g_direct_replacement_strong_contour_confirmation",
            "confirmation_index": int(pair["confirmation_index"]),
            "pair_index": pair_index,
            "source_row": int(pair["source_row"]),
            "stimulus_image_sha256": str(pair["stimulus_image_sha256"]),
            "local_window_cluster_id": str(pair["local_window_cluster_id"]),
            "image_edge_axis_gaze_deg": float(pair["image_edge_axis_deg"]),
            "image_edge_axis_array_deg": float(pair["corrected_edge_axis_array_deg"]),
            "patch_hash": _hash_array(np.asarray(patch, dtype=np.float32)),
            "trace_hash": _hash_array(trace),
            "rotation_angles_deg": rotation_angles,
            "rr100_version": str(args.rr100_version),
            "ssi_contract": "instantaneous spatial SSI weighted by predicted expected spikes",
        }
        identity_text = _identity_text(identity)
        cache_path = cache_dir / f"pair_{pair_index:06d}.npz"
        cache_paths[pair_index] = cache_path
        if cache_path.exists() and not args.force:
            cached = _load_cache(cache_path)
            cached_identity = str(cached.get("cache_identity_json", np.asarray([""]))[0])
            if cached_identity != identity_text:
                raise RuntimeError(f"Cache identity mismatch for {cache_path}; pass --force to replace")
            n_reused += 1
            print(f"[panel-g-replacement] {ordinal}/{len(shard_pairs)} reuse pair {pair_index}", flush=True)
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
            f"[panel-g-replacement] {ordinal}/{len(shard_pairs)} fresh pair {pair_index}; "
            f"elapsed={elapsed / 60.0:.1f} min",
            flush=True,
        )

    print("[panel-g-replacement] assembling direct pair tables", flush=True)
    shard_membership = membership[membership["pair_index"].astype(int).isin(shard_pairs["pair_index"].astype(int))]
    _assemble(shard_pairs, shard_membership, cache_paths, shard_dir)
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
    print(f"[panel-g-replacement] complete: {shard_dir}", flush=True)


if __name__ == "__main__":
    main()
