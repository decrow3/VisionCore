#!/usr/bin/env python3
"""Fresh RR100 rotation audit for frozen original Figure 4 matrix pairs.

This targeted, map-first run evaluates only the pre-registered image--trace
pairs in ``frozen_pair_selection.csv``.  Every selected image is required to
carry the original bank's strong-contour flag.  The recorded trace and a
deterministic 32-angle full-circle rotation grid are rendered and evaluated
fresh; historical dose curves and interpolation do not enter the scores.

Unit membership is frozen before response evaluation for both the historical
gaze-frame axis convention and the corrected image-array-frame convention.
Representative units are selected from tuning metadata alone and their
time-resolved maps are retained for the map-first checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import (
    _extract_patch,
)
from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    EPS,
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.fig.ssi_figure_v2.behavior_model_bridge.run_direct_exact_pair_ssi import (
    ALIGNED_MAX_DEG,
    DT,
    MIN_OSI,
    N_TIMEPOINTS,
    ORTHOGONAL_MIN_DEG,
    OSI_COLUMN,
    PREF_COLUMN,
    SF_COLUMN,
    SF_MIN_CPD,
    _axial_delta_deg,
    _hash_array,
    _identity_text,
    _load_shard,
    _rotation_angles_deg,
    _score_location,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
)
from declan.redundancy_resolved_v1_population import load_population_view


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BANK_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
)
DEFAULT_AUDIT_ROOT = ROOT / (
    "outputs/fig/ssi_figure_v2/behavior_model_bridge/"
    "panel_g_original_matrix_pair_rotation_audit_v1"
)
DEFAULT_SELECTION = DEFAULT_AUDIT_ROOT / "frozen_pair_selection.csv"
DEFAULT_OUT_DIR = DEFAULT_AUDIT_ROOT / "fresh_direct_rotation_n32_gpu0"
DEFAULT_UNIT_TUNING = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
    "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)

POPULATION_ORDER = (
    "high_sf_aligned_historical_gaze_axis",
    "high_sf_aligned_corrected_array_axis",
    "high_sf_orthogonal_historical_gaze_axis",
    "high_sf_orthogonal_corrected_array_axis",
    "high_sf_all",
    "low_sf_all",
)
POPULATION_LABELS = {
    "high_sf_aligned_historical_gaze_axis": "aligned high-SF (historical gaze-axis mask)",
    "high_sf_aligned_corrected_array_axis": "aligned high-SF (corrected array-axis mask)",
    "high_sf_orthogonal_historical_gaze_axis": "orthogonal high-SF (historical gaze-axis mask)",
    "high_sf_orthogonal_corrected_array_axis": "orthogonal high-SF (corrected array-axis mask)",
    "high_sf_all": "all high-SF",
    "low_sf_all": "all low-SF",
}
MAP_ROLE_ORDER = (
    "high_sf_aligned_historical_gaze_axis",
    "high_sf_aligned_corrected_array_axis",
    "high_sf_orthogonal_corrected_array_axis",
    "low_sf_all",
)
PRIMARY_METRICS = (
    "bits_per_spike",
    "information_bits_per_sample",
    "expected_spikes_per_sample",
    "mean_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    parser.add_argument("--unit-tuning", type=Path, default=DEFAULT_UNIT_TUNING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--n-rotations", type=int, default=32)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=16)
    parser.add_argument("--trace-batch-size", type=int, default=8)
    parser.add_argument("--map-rotation-examples", type=int, default=2)
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


def _population_masks(
    tuning: pd.DataFrame,
    historical_gaze_axis_deg: float,
    corrected_array_axis_deg: float,
) -> dict[str, np.ndarray]:
    sf = pd.to_numeric(tuning[SF_COLUMN], errors="coerce").to_numpy(dtype=float)
    pref = pd.to_numeric(tuning[PREF_COLUMN], errors="coerce").to_numpy(dtype=float)
    osi = pd.to_numeric(tuning[OSI_COLUMN], errors="coerce").to_numpy(dtype=float)
    tuned_high = (
        np.isfinite(sf) & (sf >= SF_MIN_CPD) & np.isfinite(pref)
        & np.isfinite(osi) & (osi >= MIN_OSI)
    )
    historical_delta = _axial_delta_deg(pref, historical_gaze_axis_deg)
    corrected_delta = _axial_delta_deg(pref, corrected_array_axis_deg)
    masks = {
        "high_sf_aligned_historical_gaze_axis": tuned_high & (historical_delta <= ALIGNED_MAX_DEG),
        "high_sf_aligned_corrected_array_axis": tuned_high & (corrected_delta <= ALIGNED_MAX_DEG),
        "high_sf_orthogonal_historical_gaze_axis": tuned_high & (historical_delta >= ORTHOGONAL_MIN_DEG),
        "high_sf_orthogonal_corrected_array_axis": tuned_high & (corrected_delta >= ORTHOGONAL_MIN_DEG),
        "high_sf_all": np.isfinite(sf) & (sf >= SF_MIN_CPD),
        "low_sf_all": np.isfinite(sf) & (sf < SF_MIN_CPD),
    }
    empty = [name for name, mask in masks.items() if not np.any(mask)]
    if empty:
        raise RuntimeError(
            f"Empty populations for gaze axis {historical_gaze_axis_deg:.3f}, "
            f"array axis {corrected_array_axis_deg:.3f}: {empty}"
        )
    return masks


def _freeze_units(
    selection: pd.DataFrame,
    tuning: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    sf = pd.to_numeric(tuning[SF_COLUMN], errors="coerce").to_numpy(dtype=float)
    pref = pd.to_numeric(tuning[PREF_COLUMN], errors="coerce").to_numpy(dtype=float)
    osi = pd.to_numeric(tuning[OSI_COLUMN], errors="coerce").to_numpy(dtype=float)
    membership_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    map_units_by_selection: dict[int, np.ndarray] = {}
    for pair in selection.itertuples(index=False):
        selection_index = int(pair.selection_index)
        gaze_axis = float(pair.image_edge_axis_gaze_deg)
        array_axis = float(pair.image_edge_axis_array_deg)
        masks = _population_masks(tuning, gaze_axis, array_axis)
        for population in POPULATION_ORDER:
            axis = (
                gaze_axis if "historical_gaze_axis" in population
                else array_axis if "corrected_array_axis" in population
                else np.nan
            )
            delta = _axial_delta_deg(pref, axis) if np.isfinite(axis) else np.full_like(pref, np.nan)
            for unit_index in np.flatnonzero(masks[population]):
                membership_rows.append(
                    {
                        "selection_index": selection_index,
                        "selection_role": str(pair.selection_role),
                        "population": population,
                        "population_label": POPULATION_LABELS[population],
                        "unit_index": int(unit_index),
                        "unit_sf_cpd": float(sf[unit_index]),
                        "unit_preferred_orientation_deg": float(pref[unit_index]),
                        "unit_orientation_selectivity_index": float(osi[unit_index]),
                        "mask_axis_deg": float(axis),
                        "unit_axis_delta_deg": float(delta[unit_index]),
                        "selection_frozen_before_fresh_model_evaluation": True,
                    }
                )
        chosen: list[int] = []
        for map_role in MAP_ROLE_ORDER:
            eligible = np.flatnonzero(masks[map_role])
            finite_osi = eligible[np.isfinite(osi[eligible])]
            if finite_osi.size:
                unit_index = int(finite_osi[np.argmax(osi[finite_osi])])
            else:
                unit_index = int(eligible[0])
            chosen.append(unit_index)
            map_rows.append(
                {
                    "selection_index": selection_index,
                    "selection_role": str(pair.selection_role),
                    "map_selection_role": map_role,
                    "selection_rule": "highest finite OSI within pre-specified population; first unit if OSI unavailable",
                    "unit_index": unit_index,
                    "unit_sf_cpd": float(sf[unit_index]),
                    "unit_preferred_orientation_deg": float(pref[unit_index]),
                    "unit_orientation_selectivity_index": float(osi[unit_index]),
                    "selection_frozen_before_fresh_model_evaluation": True,
                }
            )
        map_units_by_selection[selection_index] = np.asarray(list(dict.fromkeys(chosen)), dtype=int)
    return pd.DataFrame(membership_rows), pd.DataFrame(map_rows), map_units_by_selection


def _assemble(
    selection: pd.DataFrame,
    membership: pd.DataFrame,
    cache_paths: dict[int, Path],
    rotation_angles: np.ndarray,
    out_dir: Path,
) -> None:
    membership_lookup = {
        (int(index), str(population)): group["unit_index"].astype(int).to_numpy()
        for (index, population), group in membership.groupby(["selection_index", "population"], sort=False)
    }
    condition_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for pair in selection.itertuples(index=False):
        selection_index = int(pair.selection_index)
        shard = _load_shard(cache_paths[selection_index])
        for population in POPULATION_ORDER:
            units = membership_lookup[(selection_index, population)]
            values: list[dict[str, float]] = []
            for condition_index, condition_id in enumerate(shard["condition_id"].astype(str)):
                information = float(np.sum(shard["unit_information_bits"][condition_index, units]))
                spikes = float(np.sum(shard["unit_expected_spikes"][condition_index, units]))
                row_values = {
                    "bits_per_spike": information / max(spikes, EPS),
                    "information_bits_per_sample": information / N_TIMEPOINTS,
                    "expected_spikes_per_sample": spikes / N_TIMEPOINTS,
                    "mean_rate": spikes / (N_TIMEPOINTS * DT),
                }
                values.append(row_values)
                condition_rows.append(
                    {
                        "selection_index": selection_index,
                        "selection_role": str(pair.selection_role),
                        "image_index": int(pair.image_index),
                        "trace_index": int(pair.trace_index),
                        "image_orientation_coherence": float(pair.image_orientation_coherence),
                        "surrogate_match_advantage_percent_points": float(pair.surrogate_match_advantage_percent_points),
                        "surrogate_both_components_rotation_finite_fraction": float(pair.surrogate_both_components_rotation_finite_fraction),
                        "population": population,
                        "population_label": POPULATION_LABELS[population],
                        "n_units": int(units.size),
                        "condition_index": int(condition_index),
                        "condition_id": condition_id,
                        "condition_kind": "real" if condition_index == 0 else "rotation",
                        "rotation_angle_deg": float(shard["rotation_angle_deg"][condition_index]),
                        "fresh_model_evaluation": True,
                        **row_values,
                    }
                )
            real = values[0]
            rotations = values[1:]
            contrast: dict[str, Any] = {
                "selection_index": selection_index,
                "selection_role": str(pair.selection_role),
                "image_index": int(pair.image_index),
                "trace_index": int(pair.trace_index),
                "image_orientation_coherence": float(pair.image_orientation_coherence),
                "surrogate_match_advantage_percent_points": float(pair.surrogate_match_advantage_percent_points),
                "surrogate_both_components_rotation_finite_fraction": float(pair.surrogate_both_components_rotation_finite_fraction),
                "within_image_calibration_spearman": float(pair.within_image_calibration_spearman),
                "population": population,
                "population_label": POPULATION_LABELS[population],
                "n_units": int(units.size),
                "n_rotations": int(len(rotations)),
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
    pd.DataFrame(condition_rows).to_csv(out_dir / "direct_pair_population_rotation_curves.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(out_dir / "direct_pair_rotation_contrasts.csv", index=False)


def main() -> None:
    args = parse_args()
    selection_path = args.selection.resolve()
    bank_dir = args.bank_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    selection = pd.read_csv(selection_path).sort_values("selection_index").reset_index(drop=True)
    images = pd.read_csv(bank_dir / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    traces = np.load(bank_dir / "trace_xy.npy")
    tuning = pd.read_csv(args.unit_tuning).sort_values("unit_index").reset_index(drop=True)
    if not np.array_equal(tuning["unit_index"].astype(int).to_numpy(), np.arange(len(tuning))):
        raise RuntimeError("Unit tuning table must have contiguous unit_index values")
    image_lookup = images.set_index(images["image_index"].astype(int), drop=False)

    selected_image_rows = image_lookup.loc[selection["image_index"].astype(int)].reset_index(drop=True)
    if not selected_image_rows["image_contour_strong"].astype(bool).all():
        failed = selection.loc[~selected_image_rows["image_contour_strong"].astype(bool), "selection_index"].tolist()
        raise RuntimeError(f"Selections are not all strong-contour windows: {failed}")
    if not selection["trace_is_drift_only"].astype(bool).all():
        raise RuntimeError("Frozen targeted audit unexpectedly contains a non-drift trace")
    if len(selection) != selection["selection_index"].nunique():
        raise RuntimeError("selection_index must be unique")

    rotation_angles = _rotation_angles_deg(int(args.n_rotations))
    membership, map_selection, map_units_by_selection = _freeze_units(selection, tuning)
    selection.to_csv(out_dir / "frozen_pair_selection_used.csv", index=False)
    membership.to_csv(out_dir / "direct_unit_population_membership.csv", index=False)
    map_selection.to_csv(out_dir / "direct_selected_map_units.csv", index=False)

    old_summary = json.loads((bank_dir / "summary.json").read_text())
    historical_seconds_per_movie = float(
        np.mean([float(item["pilot"]["seconds_per_movie"]) for item in old_summary["shard_summaries"]])
    )
    plan = {
        "analysis": "panel_g_original_matrix_pair_fresh_rotation_audit",
        "artifact_type": "targeted_map_first_preflight" if args.dry_run else "targeted_map_first_run",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "scientific_question": "can direct model SSI help explain contour-relative behavioral motion anisotropy",
            "targeted_not_population_representative": True,
            "coherence_sweep_performed": False,
            "image_specific_natural_patch_nonlinearity_treated_as_pair_variance": True,
            "selection_frozen_before_fresh_model_outcomes": True,
            "all_windows_original_bank_strong_contour": True,
        },
        "inputs": {
            "selection": selection_path,
            "selection_sha256": _file_sha256(selection_path),
            "bank_dir": bank_dir,
            "image_table_sha256": _file_sha256(bank_dir / "image_feature_table.csv"),
            "trace_xy_sha256": _file_sha256(bank_dir / "trace_xy.npy"),
            "unit_tuning": args.unit_tuning.resolve(),
            "unit_tuning_sha256": _file_sha256(args.unit_tuning.resolve()),
        },
        "cohort": {
            "n_pairs": int(len(selection)),
            "n_unique_images": int(selection["image_index"].nunique()),
            "n_unique_traces": int(selection["trace_index"].nunique()),
            "minimum_orientation_coherence": float(selection["image_orientation_coherence"].min()),
            "maximum_orientation_coherence": float(selection["image_orientation_coherence"].max()),
            "drift_only": True,
        },
        "model": {
            "rr100_version": str(args.rr100_version),
            "device": str(args.device),
            "patch_size_px": int(args.patch_size_px),
            "frame_batch_size": int(args.frame_batch_size),
            "trace_batch_size": int(args.trace_batch_size),
        },
        "rotation": {
            "n_rotations": int(len(rotation_angles)),
            "angles_deg": rotation_angles,
            "contract": "deterministic full-circle midpoint grid with antipodal pairs; centroid-preserving rotation",
            "n_fresh_movies": int(len(selection) * (1 + len(rotation_angles))),
        },
        "ssi": {
            "contract": "direct instantaneous spatial SSI weighted by expected spikes over 40 samples",
            "primary_estimand": "within-pair recorded minus mean freshly evaluated rotations",
            "dose_curve_interpolation_used": False,
            "stabilized_baseline_used": False,
        },
        "population_axis_audit": {
            "historical_mask_axis": "stored gaze-frame image_edge_axis_gaze_deg",
            "corrected_mask_axis": "image-array-frame image_edge_axis_array_deg = -gaze axis mod 180",
            "both_retained": True,
        },
        "historical_runtime_reference": {
            "seconds_per_movie": historical_seconds_per_movie,
            "estimated_gpu_hours": len(selection) * (1 + len(rotation_angles)) * historical_seconds_per_movie / 3600.0,
            "caveat": "matrix batching reused images; targeted patch turnover and map retention can be slower",
        },
        "git_revision": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
    }
    _write_json(out_dir / "run_plan.json", plan)
    if args.dry_run:
        print(
            f"[original-pair-audit] dry run: {len(selection)} strong-contour pairs x "
            f"{1 + len(rotation_angles)} conditions = {plan['rotation']['n_fresh_movies']} fresh movies; "
            f"historical lower-bound {plan['historical_runtime_reference']['estimated_gpu_hours']:.2f} GPU h",
            flush=True,
        )
        return

    population_view = load_population_view(version_name=str(args.rr100_version))
    scorer = CanonicalTwinScorer(
        device=str(args.device), batch_size=int(args.frame_batch_size), empty_cache_every_batch=False
    )
    score_args = SimpleNamespace(
        trace_batch_size=int(args.trace_batch_size),
        map_rotation_examples=int(args.map_rotation_examples),
    )
    cache_paths: dict[int, Path] = {}
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    n_fresh = 0
    n_reused = 0
    started = time.perf_counter()
    for ordinal, pair_tuple in enumerate(selection.itertuples(index=False), start=1):
        pair = pd.Series(pair_tuple._asdict())
        selection_index = int(pair.selection_index)
        image_row = image_lookup.loc[int(pair.image_index)]
        patch, _patch_meta = _extract_patch(
            image_row, canvas_cache=canvas_cache, patch_size_px=int(args.patch_size_px)
        )
        trace = np.asarray(traces[int(pair.trace_index)], dtype=np.float32)
        if trace.shape != (N_TIMEPOINTS, 2) or not np.isfinite(trace).all():
            raise RuntimeError(f"Selection {selection_index} has invalid trace shape/content: {trace.shape}")
        trace = trace - np.mean(trace, axis=0, keepdims=True)
        map_units = map_units_by_selection[selection_index]
        identity = {
            "schema_version": 1,
            "selection_index": selection_index,
            "image_index": int(pair.image_index),
            "trace_index": int(pair.trace_index),
            "selection_role": str(pair.selection_role),
            "image_contour_strong": True,
            "image_orientation_coherence": float(pair.image_orientation_coherence),
            "patch_hash": _hash_array(np.asarray(patch, dtype=np.float32)),
            "trace_hash": _hash_array(trace),
            "map_unit_indices": map_units,
            "rotation_angles_deg": rotation_angles,
            "rr100_version": str(args.rr100_version),
            "ssi_contract": "instantaneous spatial SSI weighted by expected spikes",
        }
        identity_text = _identity_text(identity)
        cache_path = cache_dir / f"selection_{selection_index:02d}.npz"
        cache_paths[selection_index] = cache_path
        if cache_path.exists() and not args.force:
            existing = _load_shard(cache_path)
            cached_identity = str(existing.get("cache_identity_json", np.asarray([""]))[0])
            if cached_identity != identity_text:
                raise RuntimeError(f"Cache identity mismatch for {cache_path}; pass --force to replace")
            n_reused += 1
            print(f"[original-pair-audit] {ordinal}/{len(selection)} reuse selection {selection_index:02d}", flush=True)
            continue
        result = _score_location(
            scorer=scorer,
            population_view=population_view,
            patch=patch,
            real_trace=trace,
            rotation_angles=rotation_angles,
            map_units=map_units,
            save_maps=True,
            args=score_args,
        )
        result["cache_identity_json"] = np.asarray([identity_text])
        result["selected_map_unit_index"] = map_units.astype(np.int32)
        np.savez_compressed(cache_path, **result)
        n_fresh += 1
        elapsed = time.perf_counter() - started
        print(
            f"[original-pair-audit] {ordinal}/{len(selection)} fresh selection {selection_index:02d}; "
            f"elapsed={elapsed / 60.0:.1f} min",
            flush=True,
        )

    _assemble(selection, membership, cache_paths, rotation_angles, out_dir)
    elapsed = time.perf_counter() - started
    plan.update(
        {
            "artifact_type": "targeted_map_first_run_complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "execution": {
                "elapsed_s": elapsed,
                "n_pairs_fresh": n_fresh,
                "n_pairs_reused": n_reused,
                "movies_per_s_fresh": n_fresh * (1 + len(rotation_angles)) / max(elapsed, 1e-12),
            },
        }
    )
    _write_json(out_dir / "run_metadata.json", plan)
    print(f"[original-pair-audit] complete: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
