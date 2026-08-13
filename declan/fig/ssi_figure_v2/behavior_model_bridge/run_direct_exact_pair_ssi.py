#!/usr/bin/env python3
"""Fresh twin evaluation of exact Figure 4 Panel-G image/trajectory pairs.

Every condition in this runner is rendered and scored by the neural model.  No
one-dimensional dose curve or interpolated SSI value enters the computation.
The targeted design crosses the six map-first example windows with their local
and valid pre-specified same-image 5-deg patches, the recorded native central
40-sample trace, and deterministic full-circle rotations of that trace.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    EPS,
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.active_sensing_movie_information.run_backimage_random_patch_trace_ssi_matrix import (
    _iter_reduced_rate_maps_for_traces,
)
from declan.fig.ssi_figure_v2.behavior_confounds import (
    build_checkpoint1_reference_frame_examples as cp1,
)
from declan.fig.ssi_figure_v2.behavior_model_bridge import run_behavior_model_bridge as bridge
from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas
from declan.fixation_statistics_by_stimulus.plot_backimage_contour_motion_components import (
    _window_trace,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
    _align_response_to_trace,
)
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _clip_patch
from declan.redundancy_resolved_v1_population import load_population_view


DEFAULT_OFFSET_MANIFEST = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "checkpoint2b_offset_patch_manifest.csv"
)
DEFAULT_SOURCE_WINDOWS = cp1.DEFAULT_INPUT
DEFAULT_UNIT_TUNING = (
    ROOT
    / "outputs/active_sensing_movie_information"
    / "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1"
    / "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1"
    / "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_model_bridge"
    / "panel_g_direct_exact_pair_ssi_targeted_v1"
)

DT = 1.0 / 120.0
N_TIMEPOINTS = 40
PATCH_SIZE_PX = 540
N_ROTATIONS = 8
SF_MIN_CPD = 0.50
MIN_OSI = 0.05
ALIGNED_MAX_DEG = 15.0
ORTHOGONAL_MIN_DEG = 67.5
SF_COLUMN = "dynamic_log_gaussian_marginal_sf_cpd"
PREF_COLUMN = "prior_preferred_orientation_deg"
OSI_COLUMN = "prior_orientation_selectivity_index"
ROLE_ORDER = cp1.ROLE_ORDER
POPULATION_ORDER = ("high_sf_aligned", "high_sf_orthogonal", "high_sf_all", "low_sf_all")
POPULATION_LABELS = {
    "high_sf_aligned": "locally aligned high-SF",
    "high_sf_orthogonal": "locally orthogonal high-SF",
    "high_sf_all": "all high-SF",
    "low_sf_all": "all low-SF",
}
MAP_UNIT_ROLES = ("high_sf_aligned", "high_sf_orthogonal", "low_sf_all")
MAP_FRAME_INDICES = (9, 19, 29, 39)
PRIMARY_METRICS = (
    "bits_per_spike",
    "information_bits_per_sample",
    "expected_spikes_per_sample",
    "mean_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset-manifest", type=Path, default=DEFAULT_OFFSET_MANIFEST)
    parser.add_argument("--source-windows", type=Path, default=DEFAULT_SOURCE_WINDOWS)
    parser.add_argument("--unit-tuning", type=Path, default=DEFAULT_UNIT_TUNING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--n-rotations", type=int, default=N_ROTATIONS)
    parser.add_argument("--patch-size-px", type=int, default=PATCH_SIZE_PX)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trace-batch-size", type=int, default=2)
    parser.add_argument("--roles", type=str, default="", help="Optional comma-separated example roles.")
    parser.add_argument("--max-roles", type=int, default=0)
    parser.add_argument("--max-locations-per-role", type=int, default=0)
    parser.add_argument(
        "--display-locations-only",
        action="store_true",
        help=(
            "For each retained role, evaluate only the local patch and the preselected "
            "representative offset used in the map sheet."
        ),
    )
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _safe_slug(value: str) -> str:
    return "_".join(part for part in "".join(ch if ch.isalnum() else "_" for ch in value).split("_") if part)


def _hash_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()[:20]


def _axial_delta_deg(a_deg: np.ndarray | float, b_deg: float) -> np.ndarray:
    return np.abs((np.asarray(a_deg, dtype=float) - float(b_deg) + 90.0) % 180.0 - 90.0)


def _rotation_angles_deg(n_rotations: int) -> np.ndarray:
    n = int(n_rotations)
    if n < 2 or n % 2:
        raise ValueError("--n-rotations must be a positive even integer for antipodal full-circle pairs.")
    return (np.arange(n, dtype=np.float64) + 0.5) * (360.0 / float(n))


def _rotate_trace(trace: np.ndarray, angle_deg: float) -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float64)
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    theta = math.radians(float(angle_deg))
    rotation = np.asarray([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return (centered @ rotation.T).astype(np.float32)


def _source_row(source: pd.DataFrame, location: pd.Series) -> pd.Series:
    mask = (
        source["session"].astype(str).eq(str(location["session"]))
        & source["trial_idx"].astype(int).eq(int(location["trial_idx"]))
        & source["global_start"].astype(int).eq(int(location["global_start"]))
        & source["global_stop"].astype(int).eq(int(location["global_stop"]))
    )
    matches = source[mask]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one source row for {location['example_role']}, found {len(matches)}")
    return matches.iloc[0].copy()


def _native_trace(source_row: pd.Series) -> np.ndarray:
    full = np.asarray(_window_trace(source_row), dtype=np.float64)
    snippet = bridge._central_snippet(full, N_TIMEPOINTS)
    if snippet.shape != (N_TIMEPOINTS, 2) or not np.isfinite(snippet).all():
        raise RuntimeError(f"Expected finite native {N_TIMEPOINTS}x2 snippet, got {snippet.shape}")
    snippet -= np.mean(snippet, axis=0, keepdims=True)
    return snippet.astype(np.float32)


def _role_locations(manifest: pd.DataFrame, roles_text: str, max_roles: int, max_locations: int) -> pd.DataFrame:
    work = manifest[manifest["passes_primary_patch_qc"].astype(bool)].copy()
    requested = [part.strip() for part in str(roles_text).split(",") if part.strip()]
    if requested:
        unknown = sorted(set(requested) - set(work["example_role"].astype(str)))
        if unknown:
            raise ValueError(f"Unknown --roles: {unknown}")
        roles = requested
    else:
        present = set(work["example_role"].astype(str))
        roles = [role for role in ROLE_ORDER if role in present]
    if int(max_roles) > 0:
        roles = roles[: int(max_roles)]
    work = work[work["example_role"].astype(str).isin(roles)].copy()
    work["role_order"] = work["example_role"].astype(str).map({role: idx for idx, role in enumerate(roles)})
    work = work.sort_values(["role_order", "location_index"], kind="stable")
    if int(max_locations) > 0:
        work = work.groupby("example_role", sort=False, group_keys=False).head(int(max_locations))
    return work.reset_index(drop=True)


def _unit_selections(tuning: pd.DataFrame, local_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    sf = pd.to_numeric(tuning[SF_COLUMN], errors="coerce").to_numpy(dtype=float)
    pref = pd.to_numeric(tuning[PREF_COLUMN], errors="coerce").to_numpy(dtype=float)
    osi = pd.to_numeric(tuning[OSI_COLUMN], errors="coerce").to_numpy(dtype=float)
    unit_ids = tuning["unit_index"].astype(int).to_numpy()
    for location in local_rows.itertuples(index=False):
        role = str(location.example_role)
        axis = float(location.local_edge_axis_deg)
        delta = _axial_delta_deg(pref, axis)
        tuned_high = np.isfinite(sf) & (sf >= SF_MIN_CPD) & np.isfinite(pref) & np.isfinite(osi) & (osi >= MIN_OSI)
        masks = {
            "high_sf_aligned": tuned_high & (delta <= ALIGNED_MAX_DEG),
            "high_sf_orthogonal": tuned_high & (delta >= ORTHOGONAL_MIN_DEG),
            "high_sf_all": np.isfinite(sf) & (sf >= SF_MIN_CPD),
            "low_sf_all": np.isfinite(sf) & (sf < SF_MIN_CPD),
        }
        for population in POPULATION_ORDER:
            selected = np.flatnonzero(masks[population])
            if selected.size == 0:
                raise RuntimeError(f"No units for {role} population {population}")
            for pos in selected:
                unit_rows.append(
                    {
                        "example_role": role,
                        "population": population,
                        "population_label": POPULATION_LABELS[population],
                        "unit_index": int(unit_ids[pos]),
                        "unit_preferred_orientation_deg": float(pref[pos]),
                        "unit_orientation_selectivity_index": float(osi[pos]),
                        "unit_sf_cpd": float(sf[pos]),
                        "unit_local_axis_delta_deg": float(delta[pos]),
                        "local_edge_axis_deg": axis,
                        "selection_frozen_across_locations_and_rotations": True,
                    }
                )
        for map_role in MAP_UNIT_ROLES:
            selected = np.flatnonzero(masks[map_role])
            best_pos = int(selected[np.nanargmax(osi[selected])])
            example_rows.append(
                {
                    "example_role": role,
                    "selection_role": map_role,
                    "selection_rule": f"highest OSI within pre-specified {map_role} population",
                    "unit_index": int(unit_ids[best_pos]),
                    "unit_preferred_orientation_deg": float(pref[best_pos]),
                    "unit_orientation_selectivity_index": float(osi[best_pos]),
                    "unit_sf_cpd": float(sf[best_pos]),
                    "unit_local_axis_delta_deg": float(delta[best_pos]),
                    "local_edge_axis_deg": axis,
                }
            )
    return pd.DataFrame(unit_rows), pd.DataFrame(example_rows)


def _display_location_ids(locations: pd.DataFrame) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for role, group in locations.groupby("example_role", sort=False):
        ids = {"local"}
        offsets = group[group["location_kind"].astype(str).eq("offset")].copy()
        if not offsets.empty:
            changing = offsets[offsets["local_offset_axis_delta_deg"].astype(float) >= 30.0]
            choice = changing.nlargest(1, "local_offset_axis_delta_deg") if not changing.empty else offsets.nlargest(1, "local_offset_axis_delta_deg")
            ids.add(str(choice.iloc[0]["location_id"]))
        out[str(role)] = ids
    return out


def _direct_ssi(rate_map: np.ndarray, *, bin_seconds: float = DT) -> dict[str, np.ndarray]:
    y = np.maximum(np.asarray(rate_map, dtype=np.float64), 0.0)
    if y.ndim != 4:
        raise ValueError(f"Expected (T,N,H,W) rate maps, got {y.shape}")
    flat = y.reshape(y.shape[0], y.shape[1], -1)
    rbar = np.mean(flat, axis=2)
    gain = flat / (rbar[..., None] + EPS)
    frame_bits = np.mean(gain * np.log2(gain + EPS), axis=2)
    frame_expected = rbar * float(bin_seconds)
    unit_expected = np.sum(frame_expected, axis=0)
    unit_information = np.sum(frame_bits * frame_expected, axis=0)
    unit_bits = unit_information / np.maximum(unit_expected, EPS)
    return {
        "unit_bits_per_spike": unit_bits.astype(np.float32),
        "unit_information_bits": unit_information.astype(np.float32),
        "unit_expected_spikes": unit_expected.astype(np.float32),
        "unit_mean_rate": np.mean(rbar, axis=0).astype(np.float32),
        "unit_frame_bits_per_spike": frame_bits.astype(np.float32),
        "unit_frame_mean_rate": rbar.astype(np.float32),
    }


def _shard_identity(
    location: pd.Series,
    patch: np.ndarray,
    trace: np.ndarray,
    rotation_angles: np.ndarray,
    args: argparse.Namespace,
    map_units: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "example_role": str(location["example_role"]),
        "location_id": str(location["location_id"]),
        "session": str(location["session"]),
        "trial_idx": int(location["trial_idx"]),
        "patch_center_x_px": float(location["image_patch_center_x_px"]),
        "patch_center_y_px": float(location["image_patch_center_y_px"]),
        "patch_size_px": int(args.patch_size_px),
        "patch_hash": _hash_array(np.asarray(patch, dtype=np.float32)),
        "trace_hash": _hash_array(trace),
        "trace_n_timepoints": int(trace.shape[0]),
        "rotation_angles_deg": rotation_angles,
        "rotation_contract": "deterministic full-circle midpoint grid with antipodal pairs; centroid-preserving trace rotation",
        "rr100_version": str(args.rr100_version),
        "stimulus_normalization": "standardize_uint_like_then_minus_127_div_255",
        "map_unit_indices": map_units,
    }


def _identity_text(payload: dict[str, Any]) -> str:
    return json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))


def _score_location(
    *,
    scorer: CanonicalTwinScorer,
    population_view: Any,
    patch: np.ndarray,
    real_trace: np.ndarray,
    rotation_angles: np.ndarray,
    map_units: np.ndarray,
    save_maps: bool,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    traces = [real_trace, *[_rotate_trace(real_trace, angle) for angle in rotation_angles]]
    condition_ids = np.asarray(["real", *[f"rotation_{idx:02d}" for idx in range(len(rotation_angles))]])
    unit_bits: list[np.ndarray] = []
    unit_information: list[np.ndarray] = []
    unit_spikes: list[np.ndarray] = []
    unit_rates: list[np.ndarray] = []
    frame_bits: list[np.ndarray] = []
    frame_rates: list[np.ndarray] = []
    selected_real: np.ndarray | None = None
    selected_rotation_sum: np.ndarray | None = None
    selected_rotation_examples: list[np.ndarray] = []
    for condition_index, (full_map, _length) in enumerate(
        _iter_reduced_rate_maps_for_traces(
            scorer,
            patch,
            traces,
            trace_batch_size=int(args.trace_batch_size),
            population_view=population_view,
        )
    ):
        aligned = _align_response_to_trace(full_map, N_TIMEPOINTS)
        payload = _direct_ssi(aligned)
        unit_bits.append(payload["unit_bits_per_spike"])
        unit_information.append(payload["unit_information_bits"])
        unit_spikes.append(payload["unit_expected_spikes"])
        unit_rates.append(payload["unit_mean_rate"])
        frame_bits.append(payload["unit_frame_bits_per_spike"][:, map_units])
        frame_rates.append(payload["unit_frame_mean_rate"][:, map_units])
        if save_maps:
            selected = aligned[:, map_units].astype(np.float32, copy=True)
            if condition_index == 0:
                selected_real = selected
            else:
                if selected_rotation_sum is None:
                    selected_rotation_sum = np.zeros_like(selected, dtype=np.float64)
                selected_rotation_sum += selected
                if len(selected_rotation_examples) < int(args.map_rotation_examples):
                    selected_rotation_examples.append(selected)
        del aligned, full_map
    result: dict[str, np.ndarray] = {
        "condition_id": condition_ids,
        "rotation_angle_deg": np.asarray([np.nan, *rotation_angles], dtype=np.float32),
        "unit_bits_per_spike": np.stack(unit_bits).astype(np.float32),
        "unit_information_bits": np.stack(unit_information).astype(np.float32),
        "unit_expected_spikes": np.stack(unit_spikes).astype(np.float32),
        "unit_mean_rate": np.stack(unit_rates).astype(np.float32),
        "selected_unit_frame_bits_per_spike": np.stack(frame_bits).astype(np.float32),
        "selected_unit_frame_mean_rate": np.stack(frame_rates).astype(np.float32),
    }
    if save_maps:
        if selected_real is None or selected_rotation_sum is None:
            raise RuntimeError("Map-saving condition did not receive real and rotated maps")
        map_arrays = [selected_real, (selected_rotation_sum / float(len(rotation_angles))).astype(np.float32)]
        map_labels = ["real", "rotation_mean_map"]
        for index, item in enumerate(selected_rotation_examples):
            map_arrays.append(item)
            map_labels.append(f"rotation_{index:02d}")
        result["selected_map"] = np.stack(map_arrays).astype(np.float32)
        result["selected_map_condition"] = np.asarray(map_labels)
    return result


def _load_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _population_lookup(unit_selection: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    return {
        (str(role), str(population)): group["unit_index"].astype(int).to_numpy()
        for (role, population), group in unit_selection.groupby(["example_role", "population"], sort=False)
    }


def _assemble_tables(
    locations: pd.DataFrame,
    unit_selection: pd.DataFrame,
    shard_paths: dict[tuple[str, str], Path],
    rotation_angles: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    population_lookup = _population_lookup(unit_selection)
    condition_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    population_rows: list[dict[str, Any]] = []
    for location in locations.itertuples(index=False):
        role, location_id = str(location.example_role), str(location.location_id)
        shard = _load_shard(shard_paths[(role, location_id)])
        for condition_index, condition_id in enumerate(shard["condition_id"].astype(str)):
            angle = float(shard["rotation_angle_deg"][condition_index])
            base = {
                "example_role": role,
                "location_id": location_id,
                "location_kind": str(location.location_kind),
                "offset_angle_gaze_deg": float(location.offset_angle_gaze_deg),
                "axis_relationship": str(location.axis_relationship),
                "image_edge_axis_deg": float(location.image_edge_axis_deg),
                "image_orientation_coherence": float(location.image_orientation_coherence),
                "local_offset_axis_delta_deg": float(location.local_offset_axis_delta_deg),
                "condition_index": int(condition_index),
                "condition_id": condition_id,
                "condition_kind": "real" if condition_index == 0 else "rotation",
                "rotation_angle_deg": angle,
                "fresh_model_evaluation": True,
            }
            condition_rows.append(base)
            bits = shard["unit_bits_per_spike"][condition_index]
            information = shard["unit_information_bits"][condition_index]
            spikes = shard["unit_expected_spikes"][condition_index]
            rates = shard["unit_mean_rate"][condition_index]
            for unit_index in range(bits.size):
                unit_rows.append(
                    {
                        **base,
                        "unit_index": int(unit_index),
                        "bits_per_spike": float(bits[unit_index]),
                        "information_bits": float(information[unit_index]),
                        "information_bits_per_sample": float(information[unit_index] / N_TIMEPOINTS),
                        "expected_spikes": float(spikes[unit_index]),
                        "expected_spikes_per_sample": float(spikes[unit_index] / N_TIMEPOINTS),
                        "mean_rate": float(rates[unit_index]),
                    }
                )
            for population in POPULATION_ORDER:
                indices = population_lookup[(role, population)]
                pop_information = float(np.sum(information[indices]))
                pop_spikes = float(np.sum(spikes[indices]))
                population_rows.append(
                    {
                        **base,
                        "population": population,
                        "population_label": POPULATION_LABELS[population],
                        "n_units": int(indices.size),
                        "bits_per_spike": pop_information / max(pop_spikes, EPS),
                        "information_bits": pop_information,
                        "information_bits_per_sample": pop_information / N_TIMEPOINTS,
                        "expected_spikes": pop_spikes,
                        "expected_spikes_per_sample": pop_spikes / N_TIMEPOINTS,
                        "mean_rate": pop_spikes / (N_TIMEPOINTS * DT),
                    }
                )
    conditions = pd.DataFrame(condition_rows)
    units = pd.DataFrame(unit_rows)
    populations = pd.DataFrame(population_rows)

    contrast_rows: list[dict[str, Any]] = []
    for (role, location_id, population), group in populations.groupby(
        ["example_role", "location_id", "population"], sort=False
    ):
        real = group[group["condition_kind"].astype(str).eq("real")]
        rotated = group[group["condition_kind"].astype(str).eq("rotation")]
        if len(real) != 1 or len(rotated) != len(rotation_angles):
            raise RuntimeError(f"Incomplete direct conditions for {role}/{location_id}/{population}")
        row: dict[str, Any] = {
            "example_role": role,
            "location_id": location_id,
            "location_kind": str(real.iloc[0]["location_kind"]),
            "offset_angle_gaze_deg": float(real.iloc[0]["offset_angle_gaze_deg"]),
            "axis_relationship": str(real.iloc[0]["axis_relationship"]),
            "image_edge_axis_deg": float(real.iloc[0]["image_edge_axis_deg"]),
            "image_orientation_coherence": float(real.iloc[0]["image_orientation_coherence"]),
            "local_offset_axis_delta_deg": float(real.iloc[0]["local_offset_axis_delta_deg"]),
            "population": population,
            "population_label": str(real.iloc[0]["population_label"]),
            "n_units": int(real.iloc[0]["n_units"]),
            "n_rotations": int(len(rotated)),
        }
        for metric in PRIMARY_METRICS:
            real_value = float(real.iloc[0][metric])
            rotation_values = pd.to_numeric(rotated[metric], errors="coerce").to_numpy(dtype=float)
            rotation_mean = float(np.mean(rotation_values))
            row[f"real_{metric}"] = real_value
            row[f"rotation_mean_{metric}"] = rotation_mean
            row[f"real_minus_rotation_{metric}"] = real_value - rotation_mean
            row[f"rotation_min_{metric}"] = float(np.min(rotation_values))
            row[f"rotation_max_{metric}"] = float(np.max(rotation_values))
        pooled_info = float(rotated["information_bits"].sum())
        pooled_spikes = float(rotated["expected_spikes"].sum())
        row["rotation_pooled_bits_per_spike"] = pooled_info / max(pooled_spikes, EPS)
        row["real_minus_rotation_pooled_bits_per_spike"] = (
            float(real.iloc[0]["bits_per_spike"]) - row["rotation_pooled_bits_per_spike"]
        )
        contrast_rows.append(row)
    contrasts = pd.DataFrame(contrast_rows)

    locality_rows: list[dict[str, Any]] = []
    for (role, population), group in contrasts.groupby(["example_role", "population"], sort=False):
        local = group[group["location_kind"].astype(str).eq("local")]
        offsets = group[group["location_kind"].astype(str).eq("offset")]
        if len(local) != 1 or offsets.empty:
            continue
        row = {
            "example_role": role,
            "population": population,
            "population_label": str(local.iloc[0]["population_label"]),
            "n_units": int(local.iloc[0]["n_units"]),
            "n_valid_offsets": int(len(offsets)),
        }
        for metric in PRIMARY_METRICS:
            column = f"real_minus_rotation_{metric}"
            d_local = float(local.iloc[0][column])
            d_offset = float(pd.to_numeric(offsets[column], errors="coerce").mean())
            row[f"D_local_{metric}"] = d_local
            row[f"D_offset_{metric}"] = d_offset
            row[f"D_locality_{metric}"] = d_local - d_offset
        locality_rows.append(row)
    locality = pd.DataFrame(locality_rows)
    return conditions, units, populations, contrasts, locality


def _map_ssi(rate_map: np.ndarray) -> float:
    flat = np.maximum(np.asarray(rate_map, dtype=np.float64), 0.0).reshape(-1)
    mean = float(np.mean(flat))
    gain = flat / (mean + EPS)
    return float(np.mean(gain * np.log2(gain + EPS)))


def _render_map_sheets(
    locations: pd.DataFrame,
    example_units: pd.DataFrame,
    shards: dict[tuple[str, str], Path],
    contrasts: pd.DataFrame,
    out_dir: Path,
) -> None:
    display_ids = _display_location_ids(locations)
    role_list = [role for role in ROLE_ORDER if role in display_ids]
    compact_png = out_dir / "direct_exact_pair_selected_unit_maps.png"
    pdf_path = out_dir / "direct_exact_pair_selected_unit_maps_four_frames.pdf"

    def build_frame(frame_index: int) -> plt.Figure:
        fig, axes = plt.subplots(len(role_list), 6, figsize=(14.8, 2.45 * len(role_list)))
        if len(role_list) == 1:
            axes = axes[None, :]
        for row_index, role in enumerate(role_list):
            unit_row = example_units[
                example_units["example_role"].astype(str).eq(role)
                & example_units["selection_role"].astype(str).eq("high_sf_aligned")
            ].iloc[0]
            unit_id = int(unit_row["unit_index"])
            role_map_units = example_units[example_units["example_role"].astype(str).eq(role)]["unit_index"].astype(int).to_numpy()
            map_unit_pos = int(np.flatnonzero(role_map_units == unit_id)[0])
            role_locations = locations[locations["example_role"].astype(str).eq(role)]
            local_id = "local"
            offset_id = next(value for value in display_ids[role] if value != "local")
            maps_by_location: dict[str, np.ndarray] = {}
            for location_id in (local_id, offset_id):
                shard = _load_shard(shards[(role, location_id)])
                maps_by_location[location_id] = shard["selected_map"][:, frame_index, map_unit_pos]
            all_absolute = np.concatenate(
                [maps_by_location[local_id][:2].ravel(), maps_by_location[offset_id][:2].ravel()]
            )
            vmin, vmax = np.percentile(all_absolute, [1.0, 99.0])
            differences = np.concatenate(
                [
                    (maps_by_location[local_id][0] - maps_by_location[local_id][1]).ravel(),
                    (maps_by_location[offset_id][0] - maps_by_location[offset_id][1]).ravel(),
                ]
            )
            diff_max = float(np.percentile(np.abs(differences), 99.0))
            for block_index, location_id in enumerate((local_id, offset_id)):
                maps = maps_by_location[location_id]
                columns = (maps[0], maps[1], maps[0] - maps[1])
                for sub_index, image in enumerate(columns):
                    ax = axes[row_index, block_index * 3 + sub_index]
                    if sub_index < 2:
                        ax.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
                    else:
                        ax.imshow(image, cmap="coolwarm", vmin=-diff_max, vmax=diff_max, origin="lower")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if row_index == 0:
                        prefix = "local" if location_id == "local" else "5° offset"
                        suffix = ("real", "rotation mean map", "real − rotation mean")[sub_index]
                        ax.set_title(f"{prefix}\n{suffix}", fontsize=8.0, weight="bold")
                    if sub_index < 2:
                        ax.text(
                            0.02, 0.02, f"map SSI={_map_ssi(image):.3f}", transform=ax.transAxes,
                            color="white", fontsize=6.2,
                            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 1.2},
                        )
            local_contrast = contrasts[
                contrasts["example_role"].astype(str).eq(role)
                & contrasts["location_id"].astype(str).eq(local_id)
                & contrasts["population"].astype(str).eq("high_sf_aligned")
            ].iloc[0]
            offset_contrast = contrasts[
                contrasts["example_role"].astype(str).eq(role)
                & contrasts["location_id"].astype(str).eq(offset_id)
                & contrasts["population"].astype(str).eq("high_sf_aligned")
            ].iloc[0]
            axes[row_index, 0].set_ylabel(
                f"{cp1.ROLE_LABEL[role]}\nu{unit_id:03d}; OSI={float(unit_row['unit_orientation_selectivity_index']):.2f}\n"
                f"aligned-pop ΔSSI local={float(local_contrast['real_minus_rotation_bits_per_spike']):+.4f}\n"
                f"offset={float(offset_contrast['real_minus_rotation_bits_per_spike']):+.4f}",
                rotation=0, ha="right", va="center", labelpad=10, fontsize=7.0,
            )
        fig.suptitle(
            f"Fresh exact-pair twin responses — frame {frame_index + 1}/{N_TIMEPOINTS}\n"
            "Maps use one pre-selected locally aligned high-SF unit per example; rotation-mean maps average direct rotation responses",
            y=0.995, fontsize=10.5, weight="bold",
        )
        top = 0.88 if len(role_list) == 1 else 0.93
        fig.subplots_adjust(left=0.13, right=0.995, top=top, bottom=0.035, hspace=0.30, wspace=0.08)
        return fig

    compact = build_frame(19)
    compact.savefig(compact_png, dpi=220)
    plt.close(compact)
    with PdfPages(pdf_path) as pdf:
        for frame_index in MAP_FRAME_INDICES:
            fig = build_frame(frame_index)
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[direct-panel-g] loading manifests and freezing selections", flush=True)
    manifest = pd.read_csv(args.offset_manifest)
    locations = _role_locations(manifest, args.roles, int(args.max_roles), int(args.max_locations_per_role))
    if args.display_locations_only:
        display_location_ids = _display_location_ids(locations)
        keep = [
            str(row.location_id) in display_location_ids[str(row.example_role)]
            for row in locations.itertuples(index=False)
        ]
        locations = locations.loc[keep].reset_index(drop=True)
    source = pd.read_csv(args.source_windows)
    tuning = pd.read_csv(args.unit_tuning).sort_values("unit_index").reset_index(drop=True)
    local_rows = locations[locations["location_kind"].astype(str).eq("local")]
    if local_rows["example_role"].nunique() != locations["example_role"].nunique():
        raise RuntimeError("Every selected role must retain its local patch")
    unit_selection, example_units = _unit_selections(tuning, local_rows)
    unit_selection.to_csv(out_dir / "direct_unit_selection.csv", index=False)
    example_units.to_csv(out_dir / "direct_selected_example_units.csv", index=False)
    locations.to_csv(out_dir / "direct_location_manifest.csv", index=False)

    rotation_angles = _rotation_angles_deg(int(args.n_rotations))
    display_ids = _display_location_ids(locations)
    map_units_by_role = {
        role: group.set_index("selection_role").loc[list(MAP_UNIT_ROLES), "unit_index"].astype(int).to_numpy()
        for role, group in example_units.groupby("example_role", sort=False)
    }
    condition_manifest_rows: list[dict[str, Any]] = []
    for location in locations.itertuples(index=False):
        for condition_index, angle in enumerate([np.nan, *rotation_angles]):
            condition_manifest_rows.append(
                {
                    "example_role": str(location.example_role),
                    "location_id": str(location.location_id),
                    "location_kind": str(location.location_kind),
                    "condition_index": condition_index,
                    "condition_id": "real" if condition_index == 0 else f"rotation_{condition_index - 1:02d}",
                    "condition_kind": "real" if condition_index == 0 else "rotation",
                    "rotation_angle_deg": angle,
                    "fresh_model_evaluation_required": True,
                }
            )
    _write_csv(out_dir / "direct_condition_manifest.csv", condition_manifest_rows)

    if args.dry_run:
        print(
            f"[direct-panel-g] dry run: {len(locations)} patches x {1 + len(rotation_angles)} conditions = "
            f"{len(locations) * (1 + len(rotation_angles))} fresh movies",
            flush=True,
        )
        return

    population_view = load_population_view(version_name=str(args.rr100_version))
    scorer = CanonicalTwinScorer(
        device=str(args.device), batch_size=int(args.batch_size), empty_cache_every_batch=True
    )
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    trace_cache: dict[str, np.ndarray] = {}
    shard_paths: dict[tuple[str, str], Path] = {}

    for location_index, location_tuple in enumerate(locations.itertuples(index=False), start=1):
        location = pd.Series(location_tuple._asdict())
        role = str(location["example_role"])
        location_id = str(location["location_id"])
        cache_key = (str(location["session"]), int(location["trial_idx"]))
        if cache_key not in canvas_cache:
            canvas_cache[cache_key] = _backimage_canvas(*cache_key)
        canvas, _ppd, _shape = canvas_cache[cache_key]
        patch = _clip_patch(
            canvas,
            (float(location["image_patch_center_x_px"]), float(location["image_patch_center_y_px"])),
            int(args.patch_size_px),
        )
        if role not in trace_cache:
            trace_cache[role] = _native_trace(_source_row(source, location))
        trace = trace_cache[role]
        map_units = map_units_by_role[role]
        save_maps = location_id in display_ids[role]
        identity = _shard_identity(location, patch, trace, rotation_angles, args, map_units)
        identity_text = _identity_text(identity)
        shard_path = cache_dir / f"{_safe_slug(role)}__{_safe_slug(location_id)}.npz"
        shard_paths[(role, location_id)] = shard_path
        if shard_path.exists() and not args.force:
            shard = _load_shard(shard_path)
            cached_identity = str(shard.get("cache_identity_json", np.asarray([""]))[0])
            if cached_identity != identity_text:
                raise RuntimeError(f"Cache identity mismatch for {shard_path}; pass --force to replace it")
            print(f"[direct-panel-g] {location_index}/{len(locations)} reuse {role}/{location_id}", flush=True)
            continue
        print(
            f"[direct-panel-g] {location_index}/{len(locations)} fresh score {role}/{location_id}: "
            f"1 real + {len(rotation_angles)} rotations",
            flush=True,
        )
        result = _score_location(
            scorer=scorer,
            population_view=population_view,
            patch=patch,
            real_trace=trace,
            rotation_angles=rotation_angles,
            map_units=map_units,
            save_maps=save_maps,
            args=args,
        )
        result["cache_identity_json"] = np.asarray([identity_text])
        result["map_unit_index"] = map_units.astype(np.int32)
        np.savez_compressed(shard_path, **result)

    print("[direct-panel-g] assembling direct metrics and contrasts", flush=True)
    conditions, units, populations, contrasts, locality = _assemble_tables(
        locations, unit_selection, shard_paths, rotation_angles
    )
    conditions.to_csv(out_dir / "direct_condition_table.csv", index=False)
    units.to_csv(out_dir / "direct_unit_metrics.csv", index=False)
    populations.to_csv(out_dir / "direct_population_metrics.csv", index=False)
    contrasts.to_csv(out_dir / "direct_location_rotation_contrasts.csv", index=False)
    locality.to_csv(out_dir / "direct_locality_summary.csv", index=False)
    _render_map_sheets(locations, example_units, shard_paths, contrasts, out_dir)

    metadata = {
        "analysis": "panel_g_direct_exact_pair_ssi_targeted",
        "artifact_type": "fresh_model_evaluation_map_first_checkpoint",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "population_inference_performed": False,
        "inputs": {
            "offset_manifest": args.offset_manifest,
            "source_windows": args.source_windows,
            "unit_tuning": args.unit_tuning,
        },
        "model": {
            "rr100_version": str(args.rr100_version),
            "device": str(args.device),
            "patch_size_px": int(args.patch_size_px),
            "stimulus_normalization": "standardize_uint_like_then_minus_127_div_255",
        },
        "trace": {
            "contract": "central 40 native samples from each reviewed 128-sample window, mean centered; no temporal compression",
            "n_timepoints": N_TIMEPOINTS,
            "dt_s": DT,
        },
        "rotation": {
            "angles_deg": rotation_angles,
            "contract": "deterministic full-circle midpoint grid with antipodal pairs; rotate around trace centroid",
        },
        "ssi": {
            "contract": "instantaneous spatial SSI computed from every fresh nonnegative activation map and averaged with expected-spike weights",
            "primary_contrasts": "absolute real minus mean directly evaluated rotations; local minus mean valid-offset difference-in-differences",
            "dose_curve_interpolation_used": False,
            "stabilized_baseline_used": False,
            "saved_companions": ["information_bits_per_sample", "expected_spikes_per_sample", "mean_rate"],
        },
        "counts": {
            "n_roles": int(locations["example_role"].nunique()),
            "n_locations": int(len(locations)),
            "n_conditions_per_location": int(1 + len(rotation_angles)),
            "n_fresh_movies": int(len(locations) * (1 + len(rotation_angles))),
        },
        "git_revision": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
    }
    _write_json(out_dir / "run_metadata.json", metadata)
    shown = locality[locality["population"].astype(str).eq("high_sf_aligned")]
    print(shown.to_string(index=False), flush=True)
    print(f"[direct-panel-g] wrote direct checkpoint to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
