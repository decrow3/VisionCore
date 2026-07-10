"""Build an endpoint-aligned motion-history feature readout diagnostic.

This assay isolates the finite-history question from the current-position
confound.  Each trajectory segment is rendered as a 32-frame history by
default, shifted so the final eye position is zero:

    tau_endpoint[t] = tau[t] - tau[-1]

The feature decoder then sees only the terminal response frame, or an explicitly
requested terminal window.  The target remains the endpoint-centered image
feature embedding, so static and moving histories are compared at the same
final retinal position.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_feature_embedding_reconstruction import (
    COMPACT_BASIS,
    DEFAULT_FEATURE_SPACE_MODES,
    FEATURE_NPZ,
    PRIMARY_LATENT,
    RESPONSE_BASIS_MODES,
    RESPONSE_POPULATION_MODES,
    RR100_MOVIE_MEDOID_VERSION,
    SOURCE_WEIGHTING_MODES,
    FeatureTransform,
    _assign_source_folds,
    _bootstrap_mean,
    _clean_axis,
    _configure_matplotlib,
    _feature_space_config,
    _fit_feature_transform,
    _fit_forward_posterior,
    _json_ready,
    _load_feature_table,
    _load_feature_weights,
    _metrics,
    _parse_scales,
    _parse_str_list,
    _predict_z,
    _project_response,
    _response_basis,
    _response_population,
    _source_balanced_weights,
    _transform_feature_sources,
)
from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import (
    DEFAULT_INPUT,
    _build_trace_bank,
    _prepare_windows,
    _scale_token,
    _session_dataset_cache,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
    _align_response_to_trace,
    _static_trace,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import (
    _child_rng,
    _counts_from_rates,
    _extract_patch,
    _trace_from_item,
)
from declan.redundancy_resolved_v1_population import PopulationView


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = (
    REPO_ROOT
    / "declan"
    / "figure4_active_sensing_atlas"
    / "figures"
    / "panel_C"
    / "diagnostics"
    / "endpoint_history_feature_readout"
)

STATIC_CONDITION = "static_endpoint_history"
STATIC_LABEL = "static endpoint"
DEFAULT_MOTION_FAMILIES = ("empirical",)
VALID_MOTION_FAMILIES = ("empirical", "ou", "brownian", "rotated", "axis_edge_parallel", "axis_edge_orthogonal")
HISTORY_COORDINATE_MODES = ("raw", "fold_pca")


@dataclass(frozen=True)
class EndpointCondition:
    slug: str
    family: str
    label: str
    interpretation: str


@dataclass
class EndpointDataset:
    rows: pd.DataFrame
    x_by_condition: dict[str, np.ndarray]
    tau_by_condition: dict[str, np.ndarray]
    conditions: list[EndpointCondition]
    trace_metrics: pd.DataFrame


@dataclass(frozen=True)
class EndpointObserver:
    slug: str
    label: str
    train_bank: str
    test_input: str
    interpretation: str


@dataclass(frozen=True)
class EndpointBank:
    x: np.ndarray
    source_rows: np.ndarray


@dataclass(frozen=True)
class HistoryGenerativeModel:
    response_mean: np.ndarray
    tau_mean: np.ndarray
    feature_map: np.ndarray
    tau_map: np.ndarray
    tau_cov: np.ndarray
    noise_variance: float
    ridge: float
    n_train: int


@dataclass(frozen=True)
class CorrelatedHistoryGenerativeModel:
    response_mean: np.ndarray
    latent_mean: np.ndarray
    latent_map: np.ndarray
    latent_cov: np.ndarray
    noise_variance: float
    ridge: float
    z_dim: int
    tau_dim: int
    n_train: int


@dataclass(frozen=True)
class HistoryCoordinateProjection:
    mode: str
    mean: np.ndarray
    basis: np.ndarray
    requested_dim: int
    used_dim: int
    variance_fraction: float


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _condition_for_family(family: str) -> EndpointCondition:
    fam = str(family)
    if fam == "static":
        return EndpointCondition(
            slug=STATIC_CONDITION,
            family="static",
            label=STATIC_LABEL,
            interpretation="All 32 history frames are fixed at the endpoint/crop center.",
        )
    return EndpointCondition(
        slug=f"{fam}_endpoint_history",
        family=fam,
        label=f"{fam} endpoint history",
        interpretation=(
            f"{fam} trajectory rendered after subtracting its terminal eye position; "
            "the final frame is endpoint/crop-center aligned."
        ),
    )


def _endpoint_aligned_trace(trace: np.ndarray) -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"trace must have shape (time, 2), got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("trace must contain at least one frame")
    out = arr - arr[-1:, :]
    return out.astype(np.float32, copy=False)


def _terminal_response_counts(
    response: np.ndarray,
    *,
    n_timepoints: int,
    terminal_frames: int,
    bin_seconds: float,
) -> np.ndarray:
    aligned = _align_response_to_trace(response, int(n_timepoints))
    frames = int(terminal_frames)
    if frames <= 0:
        raise ValueError("--terminal-frames must be positive")
    if frames > aligned.shape[0]:
        raise ValueError(f"terminal_frames={frames} exceeds aligned response length {aligned.shape[0]}")
    terminal = aligned[-frames:]
    return _counts_from_rates(terminal, float(bin_seconds))


def _history_vector(trace: np.ndarray) -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"trace must have shape (time, 2), got {arr.shape}")
    if arr.shape[0] < 2:
        return np.empty(0, dtype=np.float32)
    return arr[:-1].reshape(-1).astype(np.float32, copy=False)


def _concat_response_tau(response_x: np.ndarray, tau_x: np.ndarray) -> np.ndarray:
    response = np.asarray(response_x, dtype=np.float32)
    tau = np.asarray(tau_x, dtype=np.float32)
    if response.ndim != 2 or tau.ndim != 2 or response.shape[0] != tau.shape[0]:
        raise ValueError(f"response/tau matrices must share rows, got {response.shape} and {tau.shape}")
    return np.concatenate([response, tau], axis=1).astype(np.float32, copy=False)


def _response_tau_interaction_features(response_x: np.ndarray, tau_x: np.ndarray) -> np.ndarray:
    response = np.asarray(response_x, dtype=np.float32)
    tau = np.asarray(tau_x, dtype=np.float32)
    if response.ndim != 2 or tau.ndim != 2 or response.shape[0] != tau.shape[0]:
        raise ValueError(f"response/tau matrices must share rows, got {response.shape} and {tau.shape}")
    if tau.shape[1] == 0:
        return response.astype(np.float32, copy=False)
    interaction = response[:, :, None] * tau[:, None, :]
    return np.concatenate(
        [
            response,
            tau,
            interaction.reshape(response.shape[0], response.shape[1] * tau.shape[1]),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _fit_history_coordinate_projection(
    *,
    fit_tau: np.ndarray,
    mode: str,
    n_components: int,
) -> HistoryCoordinateProjection:
    values = np.asarray(fit_tau, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"fit_tau must be a 2D history matrix, got {values.shape}")
    coordinate_mode = str(mode)
    if coordinate_mode == "raw":
        return HistoryCoordinateProjection(
            mode="raw",
            mean=np.zeros(values.shape[1], dtype=np.float64),
            basis=np.eye(values.shape[1], dtype=np.float64),
            requested_dim=int(values.shape[1]),
            used_dim=int(values.shape[1]),
            variance_fraction=float("nan"),
        )
    if coordinate_mode != "fold_pca":
        raise ValueError(f"Unknown history coordinate mode {coordinate_mode!r}; valid={list(HISTORY_COORDINATE_MODES)}")
    if values.shape[0] < 2:
        raise ValueError("fold_pca history coordinates require at least two training rows")
    requested = int(n_components)
    if requested <= 0:
        raise ValueError("--history-dim must be positive for fold_pca history coordinates")
    mean = np.mean(values, axis=0)
    centered = values - mean[None, :]
    _u, svals, vt = np.linalg.svd(centered, full_matrices=False)
    used = min(requested, int(vt.shape[0]), int(values.shape[1]))
    if used < 1:
        raise ValueError("Could not construct any fold_pca history coordinates")
    denom = float(np.sum(svals * svals))
    if denom > 1e-12:
        variance_fraction = float(np.sum(svals[:used] * svals[:used]) / denom)
    else:
        variance_fraction = float("nan")
    return HistoryCoordinateProjection(
        mode="fold_pca",
        mean=mean.astype(np.float64, copy=False),
        basis=vt[:used].T.astype(np.float64, copy=False),
        requested_dim=requested,
        used_dim=int(used),
        variance_fraction=variance_fraction,
    )


def _apply_history_coordinate_projection(
    projection: HistoryCoordinateProjection,
    tau: np.ndarray,
) -> np.ndarray:
    values = np.asarray(tau, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"tau must be a 2D history matrix, got {values.shape}")
    if values.shape[1] != projection.mean.shape[0]:
        raise ValueError(
            f"tau has raw history dim {values.shape[1]}, but projection expects {projection.mean.shape[0]}"
        )
    return ((values - projection.mean[None, :]) @ projection.basis).astype(np.float32, copy=False)


def _history_coordinate_stats(projection: HistoryCoordinateProjection) -> dict[str, Any]:
    return {
        "history_coordinate_mode": projection.mode,
        "history_dim_requested": int(projection.requested_dim),
        "history_dim_used": int(projection.used_dim),
        "history_coordinate_variance_fraction": float(projection.variance_fraction),
    }


def _history_coordinate_matrices(
    *,
    tau: np.ndarray,
    train_mask: np.ndarray,
    mode: str,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = np.asarray(tau, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"tau must be a 2D history matrix, got {values.shape}")
    mask = np.asarray(train_mask, dtype=bool)
    if mask.shape != (values.shape[0],):
        raise ValueError(f"train_mask must have shape ({values.shape[0]},), got {mask.shape}")
    projection = _fit_history_coordinate_projection(
        fit_tau=values[mask],
        mode=str(mode),
        n_components=int(n_components),
    )
    return (
        _apply_history_coordinate_projection(projection, values),
        _apply_history_coordinate_projection(projection, np.zeros_like(values)),
        _history_coordinate_stats(projection),
    )


def _trace_metrics(trace: np.ndarray) -> dict[str, float]:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.shape[0] < 2:
        path = 0.0
    else:
        path = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
    return {
        "endpoint_x_deg": float(arr[-1, 0]),
        "endpoint_y_deg": float(arr[-1, 1]),
        "endpoint_norm_deg": float(np.linalg.norm(arr[-1])),
        "history_rms_deg": float(np.sqrt(np.mean(np.sum(arr * arr, axis=1)))),
        "history_path_length_deg": path,
        "history_max_radius_deg": float(np.max(np.linalg.norm(arr, axis=1))),
    }


def _prepare_endpoint_work(args: argparse.Namespace, feature_sources: set[int]) -> pd.DataFrame:
    prepare_args = argparse.Namespace(**vars(args))
    prepare_args.max_images = 0
    work = _prepare_windows(prepare_args)
    work = work[work["source_row"].astype(int).isin(feature_sources)].copy()
    if work.empty:
        raise ValueError("No selected BackImage windows have matching feature rows")
    if int(args.max_images) > 0 and work.shape[0] > int(args.max_images):
        work = work.sample(n=int(args.max_images), replace=False, random_state=int(args.seed))
        work = work.sort_values(["session", "trial_idx", "source_row"])
        work["image_index"] = np.arange(work.shape[0], dtype=int)
    return work.reset_index(drop=True)


def _build_endpoint_dataset(
    *,
    args: argparse.Namespace,
    scorer: CanonicalTwinScorer,
    population: PopulationView,
    basis: np.ndarray,
    feature_sources: set[int],
) -> EndpointDataset:
    scales = sorted(_parse_scales(args.scales))
    if not scales:
        raise ValueError("--scales must list at least one scale")
    motion_families = _parse_str_list(args.motion_families)
    invalid = sorted(set(motion_families).difference(VALID_MOTION_FAMILIES))
    if invalid:
        raise ValueError(f"Unknown motion families {invalid}; valid={list(VALID_MOTION_FAMILIES)}")
    conditions = [_condition_for_family("static")] + [_condition_for_family(family) for family in motion_families]
    condition_slugs = [condition.slug for condition in conditions]
    if len(set(condition_slugs)) != len(condition_slugs):
        raise ValueError(f"Duplicate endpoint-history conditions requested: {condition_slugs}")

    work = _prepare_endpoint_work(args, feature_sources)
    eyepos_by_session = _session_dataset_cache(work["session"].astype(str).to_list())
    trace_bank = _build_trace_bank(
        work,
        eyepos_by_session,
        int(args.n_timepoints),
        microsaccade_speed_threshold_dps=(
            float(args.microsaccade_speed_threshold_dps)
            if args.microsaccade_speed_threshold_dps is not None
            else None
        ),
        microsaccade_threshold_z=float(args.microsaccade_threshold_z),
        microsaccade_pad_frames=int(args.microsaccade_pad_frames),
    )
    carried_trace_columns = [
        str(args.axis_source_column),
        "image_orientation_coherence",
        "image_gradient_axis_deg",
        "image_edge_axis_deg",
        "image_gradient_orientation_deg",
        "image_edge_orientation_deg",
        "drift_gradient_delta_deg",
        "drift_edge_delta_deg",
    ]
    work_by_source = {
        int(row["source_row"]): row
        for _, row in work.iterrows()
    }
    for item in trace_bank:
        source_row = int(item["source_row"])
        row = work_by_source.get(source_row)
        if row is None:
            continue
        for column in carried_trace_columns:
            if column in row:
                item[column] = row[column]
    source_to_item = {int(item["source_row"]): item for item in trace_bank}
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    row_records: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    x_parts = {condition.slug: [] for condition in conditions}
    tau_parts = {condition.slug: [] for condition in conditions}

    sample_index = 0
    iterator = tqdm(list(work.iterrows()), desc="endpoint histories")
    for work_pos, row in iterator:
        source_row = int(row["source_row"])
        if source_row not in source_to_item:
            raise ValueError(f"Missing trace-bank item for source_row={source_row}")
        item = source_to_item[source_row]
        patch, patch_meta = _extract_patch(
            row,
            canvas_cache=canvas_cache,
            patch_size_px=int(args.patch_size_px),
        )
        static_trace = _static_trace(int(args.n_timepoints))
        for scale in scales:
            traces: list[np.ndarray] = []
            per_condition_meta: list[dict[str, Any]] = []
            for condition in conditions:
                if condition.family == "static":
                    raw_trace = static_trace
                    raw_meta = {
                        "requested_rms_deg": 0.0,
                        "effective_rms_deg": 0.0,
                        "path_length_deg": 0.0,
                    }
                else:
                    raw_trace, raw_meta = _trace_from_item(
                        family=condition.family,
                        item=item,
                        scale=float(scale),
                        rng=_child_rng(int(args.seed), "endpoint-history", source_row, condition.family, float(scale)),
                        max_rms_deg=float(args.max_rms_deg),
                        axis_source_column=str(args.axis_source_column),
                        axis_template_mode=str(args.axis_template_mode),
                        axis_match_policy=str(args.axis_match_policy),
                    )
                endpoint_trace = _endpoint_aligned_trace(raw_trace)
                traces.append(endpoint_trace)
                tau_parts[condition.slug].append(_history_vector(endpoint_trace))
                metrics = _trace_metrics(endpoint_trace)
                per_condition_meta.append({"raw_meta": raw_meta, "metrics": metrics})
                trace_records.append(
                    {
                        "sample_index": int(sample_index),
                        "source_row": source_row,
                        "condition": condition.slug,
                        "family": condition.family,
                        "observation_scale": float(scale),
                        "n_timepoints": int(args.n_timepoints),
                        "endpoint_alignment": "trace_minus_final_position",
                        "raw_requested_rms_deg": float(raw_meta.get("requested_rms_deg", np.nan)),
                        "raw_effective_rms_deg": float(raw_meta.get("effective_rms_deg", np.nan)),
                        "raw_path_length_deg": float(raw_meta.get("path_length_deg", np.nan)),
                        "axis_conditioned": bool(raw_meta.get("axis_conditioned", False)),
                        "axis_relation": str(raw_meta.get("axis_relation", "")),
                        "axis_match_status": str(raw_meta.get("axis_match_status", "")),
                        "axis_deg": float(raw_meta.get("axis_deg", np.nan)),
                        "output_axis_deg": float(raw_meta.get("output_axis_deg", np.nan)),
                        "axis_template_mode": str(raw_meta.get("axis_template_mode", "")),
                        **metrics,
                    }
                )

            responses = scorer.responses(patch, traces, trace_batch_size=int(args.twin_trace_batch_size))
            if len(responses) != len(conditions):
                raise ValueError(f"Expected {len(conditions)} responses, got {len(responses)}")
            response_frame_counts = [int(np.asarray(resp).shape[0]) for resp in responses]
            for condition, response in zip(conditions, responses, strict=True):
                terminal_counts = _terminal_response_counts(
                    response,
                    n_timepoints=int(args.n_timepoints),
                    terminal_frames=int(args.terminal_frames),
                    bin_seconds=float(args.bin_seconds),
                )
                x_parts[condition.slug].append(_project_response(terminal_counts, basis, population=population))

            row_records.append(
                {
                    "sample_index": int(sample_index),
                    "source_row": source_row,
                    "true_source_row": source_row,
                    "session": str(row["session"]),
                    "trial_idx": int(row.get("trial_idx", -1)),
                    "image_index": int(row.get("image_index", work_pos)),
                    "observation_scale": float(scale),
                    "n_timepoints": int(args.n_timepoints),
                    "terminal_frames": int(args.terminal_frames),
                    "readout_time_contract": "terminal_response_only",
                    "endpoint_alignment": "trace_minus_final_position",
                    "patch_center_x_px": float(patch_meta["patch_center_x_px"]),
                    "patch_center_y_px": float(patch_meta["patch_center_y_px"]),
                    "patch_ppd": float(patch_meta["patch_ppd"]),
                    "image_edge_axis_deg": float(row.get("image_edge_axis_deg", np.nan)),
                    "image_orientation_coherence": float(row.get("image_orientation_coherence", np.nan)),
                    "drift_edge_delta_deg": float(row.get("drift_edge_delta_deg", np.nan)),
                    "response_frames_before_alignment_min": int(min(response_frame_counts)),
                    "response_frames_before_alignment_max": int(max(response_frame_counts)),
                }
            )
            sample_index += 1

    x_by_condition = {
        condition: np.stack(values, axis=0).astype(np.float32)
        for condition, values in x_parts.items()
    }
    tau_by_condition = {
        condition: np.stack(values, axis=0).astype(np.float32)
        for condition, values in tau_parts.items()
    }
    return EndpointDataset(
        rows=pd.DataFrame(row_records),
        x_by_condition=x_by_condition,
        tau_by_condition=tau_by_condition,
        conditions=conditions,
        trace_metrics=pd.DataFrame(trace_records),
    )


def _save_endpoint_dataset(dataset: EndpointDataset, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    rows_path = out_dir / "endpoint_history_dataset_rows.csv"
    trace_path = out_dir / "endpoint_history_trace_metrics.csv"
    arrays_path = out_dir / "endpoint_history_dataset_arrays.npz"
    conditions_path = out_dir / "endpoint_history_dataset_conditions.json"
    dataset.rows.to_csv(rows_path, index=False)
    dataset.trace_metrics.to_csv(trace_path, index=False)
    arrays: dict[str, np.ndarray] = {
        "condition_slugs": np.asarray([condition.slug for condition in dataset.conditions]),
    }
    for condition in dataset.conditions:
        arrays[f"x__{condition.slug}"] = np.asarray(dataset.x_by_condition[condition.slug], dtype=np.float32)
        arrays[f"tau__{condition.slug}"] = np.asarray(dataset.tau_by_condition[condition.slug], dtype=np.float32)
    np.savez_compressed(arrays_path, **arrays)
    _write_json(
        conditions_path,
        {
            "conditions": [
                {
                    "slug": condition.slug,
                    "family": condition.family,
                    "label": condition.label,
                    "interpretation": condition.interpretation,
                }
                for condition in dataset.conditions
            ]
        },
    )
    return {
        "dataset_rows": rows_path,
        "dataset_arrays": arrays_path,
        "dataset_conditions": conditions_path,
        "trace_metrics": trace_path,
    }


def _load_endpoint_dataset(dataset_dir: Path) -> EndpointDataset:
    dataset_dir = Path(dataset_dir)
    rows_path = dataset_dir / "endpoint_history_dataset_rows.csv"
    trace_path = dataset_dir / "endpoint_history_trace_metrics.csv"
    arrays_path = dataset_dir / "endpoint_history_dataset_arrays.npz"
    conditions_path = dataset_dir / "endpoint_history_dataset_conditions.json"
    missing = [path for path in [rows_path, trace_path, arrays_path, conditions_path] if not path.exists()]
    if missing:
        preview = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Endpoint dataset cache is missing required files: {preview}")
    payload = _read_json(conditions_path)
    conditions = [
        EndpointCondition(
            slug=str(item["slug"]),
            family=str(item["family"]),
            label=str(item["label"]),
            interpretation=str(item["interpretation"]),
        )
        for item in payload.get("conditions", [])
    ]
    if not conditions:
        raise ValueError(f"No conditions found in {conditions_path}")
    with np.load(arrays_path, allow_pickle=False) as loaded:
        x_by_condition = {
            condition.slug: np.asarray(loaded[f"x__{condition.slug}"], dtype=np.float32)
            for condition in conditions
        }
        tau_by_condition = {
            condition.slug: np.asarray(loaded[f"tau__{condition.slug}"], dtype=np.float32)
            for condition in conditions
        }
    return EndpointDataset(
        rows=pd.read_csv(rows_path),
        x_by_condition=x_by_condition,
        tau_by_condition=tau_by_condition,
        conditions=conditions,
        trace_metrics=pd.read_csv(trace_path),
    )


def _loaded_endpoint_dataset_meta(dataset_dir: Path, dataset: EndpointDataset) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "endpoint_history_feature_readout_manifest.json"
    dim = int(next(iter(dataset.x_by_condition.values())).shape[1])
    fallback_population = {
        "population_name": "cached_endpoint_dataset",
        "population_n_units": dim,
        "source": str(dataset_dir),
    }
    fallback_basis = {
        "response_basis_mode": "cached_endpoint_dataset",
        "response_dim": dim,
        "source": str(dataset_dir),
    }
    if not manifest_path.exists():
        return fallback_population, fallback_basis, {}
    manifest = _read_json(manifest_path)
    population = manifest.get("population", fallback_population)
    basis = manifest.get("basis", fallback_basis)
    assay = manifest.get("assay", {})
    return dict(population), dict(basis), dict(assay)


def _validate_endpoint_dataset_contract(
    *,
    dataset: EndpointDataset,
    args: argparse.Namespace,
    primary_condition: str,
    joint_conditions: list[str],
    cached_assay: dict[str, Any] | None = None,
) -> list[float]:
    required = [STATIC_CONDITION, primary_condition, *joint_conditions]
    missing = sorted({condition for condition in required if condition not in dataset.x_by_condition})
    if missing:
        raise ValueError(f"Endpoint dataset is missing required condition(s): {missing}")
    for condition in required:
        if condition not in dataset.tau_by_condition:
            raise ValueError(f"Endpoint dataset is missing tau history for condition {condition!r}")
        if dataset.x_by_condition[condition].shape[0] != dataset.rows.shape[0]:
            raise ValueError(
                f"Condition {condition!r} has {dataset.x_by_condition[condition].shape[0]} response rows "
                f"but dataset rows has {dataset.rows.shape[0]}"
            )
        if dataset.tau_by_condition[condition].shape[0] != dataset.rows.shape[0]:
            raise ValueError(
                f"Condition {condition!r} has {dataset.tau_by_condition[condition].shape[0]} tau rows "
                f"but dataset rows has {dataset.rows.shape[0]}"
            )
    if "true_source_row" not in dataset.rows:
        raise ValueError("Endpoint dataset rows must include true_source_row")
    scales = sorted(float(value) for value in dataset.rows["observation_scale"].unique().tolist())
    if "n_timepoints" in dataset.rows:
        n_timepoints = sorted(int(value) for value in dataset.rows["n_timepoints"].unique().tolist())
        if n_timepoints != [int(args.n_timepoints)]:
            raise ValueError(
                f"Cached endpoint dataset has n_timepoints={n_timepoints}, "
                f"but requested n_timepoints={int(args.n_timepoints)}"
            )
    if "terminal_frames" in dataset.rows:
        terminal_frames = sorted(int(value) for value in dataset.rows["terminal_frames"].unique().tolist())
        if terminal_frames != [int(args.terminal_frames)]:
            raise ValueError(
                f"Cached endpoint dataset has terminal_frames={terminal_frames}, "
                f"but requested terminal_frames={int(args.terminal_frames)}"
            )
    if cached_assay:
        cached_primary = cached_assay.get("primary_condition")
        if cached_primary is not None and str(cached_primary) != str(primary_condition):
            print(
                "[endpoint-history] warning: cached dataset was originally analyzed with "
                f"primary_condition={cached_primary!r}; reusing cached responses with requested "
                f"primary_condition={primary_condition!r}.",
                flush=True,
            )
    return scales


def _observer_specs(*, primary_condition: str, joint_conditions: list[str]) -> list[EndpointObserver]:
    joint_text = ",".join(joint_conditions)
    return [
        EndpointObserver(
            slug="static_history",
            label="static history",
            train_bank="static_response",
            test_input="static_response",
            interpretation="Static endpoint history: train and test on terminal response after zero displacement history.",
        ),
        EndpointObserver(
            slug="known_history",
            label="known history",
            train_bank="primary_response_plus_tau",
            test_input="primary_response_plus_tau",
            interpretation=(
                "Known-history observer: terminal response plus true previous endpoint-aligned path "
                "tau[0:T-1] are available to the readout."
            ),
        ),
        EndpointObserver(
            slug="known_history_interaction",
            label="known history interaction",
            train_bank="primary_response_plus_tau",
            test_input="primary_response_plus_tau",
            interpretation=(
                "Known-history diagnostic with a low-rank gain-field feature expansion: "
                "terminal response, history coordinates, and response-by-history interactions."
            ),
        ),
        EndpointObserver(
            slug="known_history_multi_direct",
            label="known multi-history direct",
            train_bank="joint_prior_response_plus_tau",
            test_input="primary_response_plus_tau",
            interpretation=(
                "Known-history direct readout trained on repeated endpoint histories: "
                "terminal response plus train-fold history coordinates."
            ),
        ),
        EndpointObserver(
            slug="known_history_multi_interaction",
            label="known multi-history interaction",
            train_bank="joint_prior_response_plus_tau",
            test_input="primary_response_plus_tau",
            interpretation=(
                "Known-history gain-field readout trained on repeated endpoint histories: "
                "terminal response, history coordinates, and response-by-history interactions."
            ),
        ),
        EndpointObserver(
            slug="known_history_repeated_adjusted",
            label="known repeated adjusted",
            train_bank="joint_prior_response_plus_tau",
            test_input="primary_response_plus_tau",
            interpretation=(
                "Known-history repeated-measures observer: estimate history response components "
                "from source-centered repeated histories, adjust the terminal response toward "
                "zero-history reference, then decode endpoint features."
            ),
        ),
        EndpointObserver(
            slug="joint_history_response_only",
            label="joint/hidden history",
            train_bank="joint_prior_response",
            test_input="primary_response",
            interpretation=(
                "Response-only hidden-history observer: train on endpoint-history response samples "
                f"from conditions [{joint_text}], then test on the primary endpoint-history response."
            ),
        ),
        EndpointObserver(
            slug="zero_history_on_motion",
            label="zero-history model on motion",
            train_bank="static_response",
            test_input="primary_response",
            interpretation=(
                "Nuisance observer: response is generated by the primary endpoint motion history, "
                "but the feature readout is the static-history model."
            ),
        ),
        EndpointObserver(
            slug="known_history_generative",
            label="known history generative",
            train_bank="joint_prior_response_plus_tau",
            test_input="primary_response_plus_tau",
            interpretation=(
                "Multi-history linear-Gaussian history model with true previous path plugged in: "
                "r_T = A z + B tau_history + noise."
            ),
        ),
        EndpointObserver(
            slug="joint_history_generative",
            label="joint history generative",
            train_bank="joint_prior_response_plus_tau",
            test_input="primary_response",
            interpretation=(
                "Multi-history linear-Gaussian joint observer that treats previous endpoint-aligned path "
                "as a latent Gaussian nuisance/history variable and marginalizes it."
            ),
        ),
        EndpointObserver(
            slug="zero_history_generative_on_motion",
            label="zero-history generative on motion",
            train_bank="joint_prior_response_plus_tau",
            test_input="primary_response",
            interpretation=(
                "Multi-history linear-Gaussian history model applied to motion response while forcing "
                "the previous endpoint-aligned path to zero."
            ),
        ),
        EndpointObserver(
            slug="known_history_correlated_generative",
            label="known correlated generative",
            train_bank="joint_prior_response_plus_tau",
            test_input="primary_response_plus_tau",
            interpretation=(
                "Linear-Gaussian history model with a joint feature-history Gaussian prior; "
                "known previous path uses the conditional prior p(z | tau_true)."
            ),
        ),
        EndpointObserver(
            slug="joint_history_correlated_generative",
            label="joint correlated generative",
            train_bank="joint_prior_response_plus_tau",
            test_input="primary_response",
            interpretation=(
                "Linear-Gaussian joint observer with a correlated feature-history Gaussian prior "
                "and latent previous path marginalized from the terminal response."
            ),
        ),
        EndpointObserver(
            slug="zero_history_correlated_generative_on_motion",
            label="zero correlated generative on motion",
            train_bank="joint_prior_response_plus_tau",
            test_input="primary_response",
            interpretation=(
                "Correlated-prior history model applied to the motion response while forcing "
                "the previous endpoint-aligned path to zero."
            ),
        ),
    ]


def _endpoint_bank(
    dataset: EndpointDataset,
    *,
    bank_name: str,
    primary_condition: str,
    joint_conditions: list[str],
) -> EndpointBank:
    sources = dataset.rows["true_source_row"].to_numpy(dtype=int)
    if bank_name == "static_response":
        return EndpointBank(x=dataset.x_by_condition[STATIC_CONDITION], source_rows=sources)
    if bank_name == "primary_response":
        return EndpointBank(x=dataset.x_by_condition[primary_condition], source_rows=sources)
    if bank_name == "primary_response_plus_tau":
        return EndpointBank(
            x=_concat_response_tau(
                dataset.x_by_condition[primary_condition],
                dataset.tau_by_condition[primary_condition],
            ),
            source_rows=sources,
        )
    if bank_name == "joint_prior_response_plus_tau":
        xs = [
            _concat_response_tau(dataset.x_by_condition[condition], dataset.tau_by_condition[condition])
            for condition in joint_conditions
        ]
        return EndpointBank(
            x=np.concatenate(xs, axis=0).astype(np.float32, copy=False),
            source_rows=np.tile(sources, len(xs)).astype(int, copy=False),
        )
    if bank_name == "joint_prior_response":
        xs = [dataset.x_by_condition[condition] for condition in joint_conditions]
        return EndpointBank(
            x=np.concatenate(xs, axis=0).astype(np.float32, copy=False),
            source_rows=np.tile(sources, len(xs)).astype(int, copy=False),
        )
    raise ValueError(f"Unknown endpoint bank {bank_name!r}")


def _weights_for_sources(source_rows: np.ndarray, *, source_weighting: str) -> np.ndarray:
    if str(source_weighting) == "source_balanced":
        return _source_balanced_weights(source_rows)
    if str(source_weighting) == "row_unweighted":
        return np.ones(np.asarray(source_rows).shape[0], dtype=np.float64)
    valid = ", ".join(SOURCE_WEIGHTING_MODES)
    raise ValueError(f"Unknown source_weighting={source_weighting!r}; valid modes: {valid}")


def _known_history_nested_prediction(
    *,
    z_all: np.ndarray,
    x_response: np.ndarray,
    x_response_tau: np.ndarray,
    source_rows: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    ridge: float,
    noise_floor: float,
    source_weighting: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Known-history residual observer with alpha=0 fallback.

    The base model decodes from terminal response only.  A second model predicts
    the base residual from response plus previous path.  The correction gain is
    selected by inner source-disjoint validation, and alpha=0 is always in the
    grid so the known-history observer can fall back to response-only.
    """
    sources = np.asarray(source_rows, dtype=int)
    train_sources = sources[train_mask]
    unique_train = np.asarray(sorted(set(train_sources.tolist())), dtype=int)
    alpha_grid = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    selected_alpha = 0.0
    inner_rows = 0
    inner_folds_used = 0
    if unique_train.size >= 4:
        n_inner = min(3, int(unique_train.size))
        inner_fold_by_source = _assign_source_folds(unique_train, n_folds=n_inner, seed=int(seed))
        z_val_parts: list[np.ndarray] = []
        base_val_parts: list[np.ndarray] = []
        corr_val_parts: list[np.ndarray] = []
        for inner_fold in sorted(set(inner_fold_by_source.values())):
            inner_val = train_mask & np.asarray(
                [inner_fold_by_source.get(int(source), -1) == int(inner_fold) for source in sources],
                dtype=bool,
            )
            inner_train = train_mask & ~inner_val
            if int(np.sum(inner_val)) == 0 or int(np.sum(inner_train)) <= z_all.shape[1]:
                continue
            weights = _weights_for_sources(sources[inner_train], source_weighting=source_weighting)
            base_model = _fit_forward_posterior(
                z_train=z_all[inner_train],
                x_train=x_response[inner_train],
                ridge=float(ridge),
                noise_floor=float(noise_floor),
                sample_weight=weights,
            )
            base_train = _predict_z(base_model, x_response[inner_train])
            resid_train = z_all[inner_train] - base_train
            corr_model = _fit_forward_posterior(
                z_train=resid_train,
                x_train=x_response_tau[inner_train],
                ridge=float(ridge),
                noise_floor=float(noise_floor),
                sample_weight=weights,
            )
            z_val_parts.append(z_all[inner_val])
            base_val_parts.append(_predict_z(base_model, x_response[inner_val]))
            corr_val_parts.append(_predict_z(corr_model, x_response_tau[inner_val]))
            inner_rows += int(np.sum(inner_val))
            inner_folds_used += 1
        if z_val_parts:
            z_val = np.concatenate(z_val_parts, axis=0)
            base_val = np.concatenate(base_val_parts, axis=0)
            corr_val = np.concatenate(corr_val_parts, axis=0)
            sse_by_alpha = np.asarray(
                [float(np.sum((z_val - (base_val + alpha * corr_val)) ** 2)) for alpha in alpha_grid],
                dtype=np.float64,
            )
            selected_alpha = float(alpha_grid[int(np.argmin(sse_by_alpha))])

    weights = _weights_for_sources(sources[train_mask], source_weighting=source_weighting)
    base_model = _fit_forward_posterior(
        z_train=z_all[train_mask],
        x_train=x_response[train_mask],
        ridge=float(ridge),
        noise_floor=float(noise_floor),
        sample_weight=weights,
    )
    base_train = _predict_z(base_model, x_response[train_mask])
    resid_train = z_all[train_mask] - base_train
    corr_model = _fit_forward_posterior(
        z_train=resid_train,
        x_train=x_response_tau[train_mask],
        ridge=float(ridge),
        noise_floor=float(noise_floor),
        sample_weight=weights,
    )
    base_test = _predict_z(base_model, x_response[test_mask])
    corr_test = _predict_z(corr_model, x_response_tau[test_mask])
    z_hat = base_test + float(selected_alpha) * corr_test
    return z_hat.astype(np.float64, copy=False), {
        "known_history_model": "response_only_base_plus_response_tau_residual",
        "known_history_alpha": float(selected_alpha),
        "known_history_alpha_grid": ",".join(f"{value:g}" for value in alpha_grid),
        "known_history_alpha_selection": "inner_source_disjoint_sse_with_alpha0_fallback",
        "known_history_inner_rows": int(inner_rows),
        "known_history_inner_folds_used": int(inner_folds_used),
        "base_noise_variance": float(base_model.noise_variance),
        "correction_noise_variance": float(corr_model.noise_variance),
        "base_response_map_fro_norm": float(np.linalg.norm(base_model.response_map)),
        "correction_response_map_fro_norm": float(np.linalg.norm(corr_model.response_map)),
    }


def _fit_history_generative_model(
    *,
    z_train: np.ndarray,
    x_train: np.ndarray,
    tau_train: np.ndarray,
    ridge: float,
    noise_floor: float,
    sample_weight: np.ndarray | None,
) -> HistoryGenerativeModel:
    z = np.asarray(z_train, dtype=np.float64)
    x = np.asarray(x_train, dtype=np.float64)
    tau = np.asarray(tau_train, dtype=np.float64)
    if z.ndim != 2 or x.ndim != 2 or tau.ndim != 2:
        raise ValueError(f"Expected 2D z/x/tau matrices, got {z.shape}, {x.shape}, {tau.shape}")
    if z.shape[0] != x.shape[0] or z.shape[0] != tau.shape[0]:
        raise ValueError(f"z/x/tau must share rows, got {z.shape}, {x.shape}, {tau.shape}")
    if sample_weight is None:
        weights = np.ones(z.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != (z.shape[0],):
            raise ValueError(f"sample_weight must have shape ({z.shape[0]},), got {weights.shape}")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("sample_weight must be finite and non-negative")
    weights = weights / (float(np.mean(weights)) + 1e-12)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        raise ValueError("sample_weight sum must be positive")

    response_mean = np.sum(x * weights[:, None], axis=0) / weight_sum
    tau_mean = np.sum(tau * weights[:, None], axis=0) / weight_sum
    tau_centered = tau - tau_mean[None, :]
    y = x - response_mean[None, :]
    design = np.concatenate([z, tau_centered], axis=1)
    ridge_value = float(ridge)
    if not np.isfinite(ridge_value) or ridge_value < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    w_design = design * weights[:, None]
    normal = design.T @ w_design + ridge_value * np.eye(design.shape[1], dtype=np.float64)
    coeff = np.linalg.solve(normal, w_design.T @ y)
    z_dim = int(z.shape[1])
    feature_map = coeff[:z_dim]
    tau_map = coeff[z_dim:]
    residual = y - design @ coeff
    noise_variance = max(float(np.sum((residual * residual) * weights[:, None]) / (weight_sum * residual.shape[1])), float(noise_floor))

    tau_cov = (tau_centered.T * weights[None, :]) @ tau_centered / weight_sum
    mean_tau_var = float(np.mean(np.diag(tau_cov))) if tau_cov.size else 0.0
    cov_floor = max(1e-10, 1e-3 * mean_tau_var)
    tau_cov = tau_cov + cov_floor * np.eye(tau_cov.shape[0], dtype=np.float64)
    return HistoryGenerativeModel(
        response_mean=response_mean,
        tau_mean=tau_mean,
        feature_map=feature_map,
        tau_map=tau_map,
        tau_cov=tau_cov,
        noise_variance=float(noise_variance),
        ridge=ridge_value,
        n_train=int(z.shape[0]),
    )


def _predict_history_known(model: HistoryGenerativeModel, x: np.ndarray, tau: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    tau_values = np.asarray(tau, dtype=np.float64)
    if tau_values.ndim != 2 or tau_values.shape[0] != values.shape[0]:
        raise ValueError(f"tau rows must match x rows, got {tau_values.shape} and {values.shape}")
    tau_centered = tau_values - model.tau_mean[None, :]
    y = values - model.response_mean[None, :] - tau_centered @ model.tau_map
    a = model.feature_map
    sigma = max(float(model.noise_variance), 1e-12)
    precision = np.eye(a.shape[0], dtype=np.float64) + (a @ a.T) / sigma
    h = (y @ a.T) / sigma
    return np.linalg.solve(precision.T, h.T).T.astype(np.float64, copy=False)


def _predict_history_joint(model: HistoryGenerativeModel, x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    y = values - model.response_mean[None, :]
    a = model.feature_map
    b = model.tau_map
    sigma = max(float(model.noise_variance), 1e-12)
    latent_map = np.vstack([a, b])
    z_dim = int(a.shape[0])
    tau_cov_inv = np.linalg.pinv(model.tau_cov)
    prior_precision = np.zeros((latent_map.shape[0], latent_map.shape[0]), dtype=np.float64)
    prior_precision[:z_dim, :z_dim] = np.eye(z_dim, dtype=np.float64)
    prior_precision[z_dim:, z_dim:] = tau_cov_inv
    precision = prior_precision + (latent_map @ latent_map.T) / sigma
    h = (y @ latent_map.T) / sigma
    latent_hat = np.linalg.solve(precision.T, h.T).T
    return latent_hat[:, :z_dim].astype(np.float64, copy=False)


def _history_generative_stats(model: HistoryGenerativeModel, *, mode: str) -> dict[str, Any]:
    return {
        "history_generative_mode": str(mode),
        "history_generative_model": "linear_gaussian_r_equals_Az_plus_Btau_plus_noise",
        "history_generative_tau_prior": "weighted_empirical_gaussian_endpoint_history",
        "history_generative_noise_variance": float(model.noise_variance),
        "history_generative_tau_cov_trace": float(np.trace(model.tau_cov)),
        "history_generative_feature_map_fro_norm": float(np.linalg.norm(model.feature_map)),
        "history_generative_tau_map_fro_norm": float(np.linalg.norm(model.tau_map)),
    }


def _fit_correlated_history_generative_model(
    *,
    z_train: np.ndarray,
    x_train: np.ndarray,
    tau_train: np.ndarray,
    ridge: float,
    noise_floor: float,
    sample_weight: np.ndarray | None,
) -> CorrelatedHistoryGenerativeModel:
    z = np.asarray(z_train, dtype=np.float64)
    x = np.asarray(x_train, dtype=np.float64)
    tau = np.asarray(tau_train, dtype=np.float64)
    if z.ndim != 2 or x.ndim != 2 or tau.ndim != 2:
        raise ValueError(f"Expected 2D z/x/tau matrices, got {z.shape}, {x.shape}, {tau.shape}")
    if z.shape[0] != x.shape[0] or z.shape[0] != tau.shape[0]:
        raise ValueError(f"z/x/tau must share rows, got {z.shape}, {x.shape}, {tau.shape}")
    if sample_weight is None:
        weights = np.ones(z.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != (z.shape[0],):
            raise ValueError(f"sample_weight must have shape ({z.shape[0]},), got {weights.shape}")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("sample_weight must be finite and non-negative")
    weights = weights / (float(np.mean(weights)) + 1e-12)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        raise ValueError("sample_weight sum must be positive")
    latent = np.concatenate([z, tau], axis=1)
    response_mean = np.sum(x * weights[:, None], axis=0) / weight_sum
    latent_mean = np.sum(latent * weights[:, None], axis=0) / weight_sum
    latent_centered = latent - latent_mean[None, :]
    y = x - response_mean[None, :]
    ridge_value = float(ridge)
    if not np.isfinite(ridge_value) or ridge_value < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    w_latent = latent_centered * weights[:, None]
    normal = latent_centered.T @ w_latent + ridge_value * np.eye(latent_centered.shape[1], dtype=np.float64)
    latent_map = np.linalg.solve(normal, w_latent.T @ y)
    residual = y - latent_centered @ latent_map
    noise_variance = max(float(np.sum((residual * residual) * weights[:, None]) / (weight_sum * residual.shape[1])), float(noise_floor))
    latent_cov = (latent_centered.T * weights[None, :]) @ latent_centered / weight_sum
    mean_latent_var = float(np.mean(np.diag(latent_cov))) if latent_cov.size else 0.0
    cov_floor = max(1e-8, 1e-3 * mean_latent_var)
    latent_cov = latent_cov + cov_floor * np.eye(latent_cov.shape[0], dtype=np.float64)
    return CorrelatedHistoryGenerativeModel(
        response_mean=response_mean,
        latent_mean=latent_mean,
        latent_map=latent_map,
        latent_cov=latent_cov,
        noise_variance=float(noise_variance),
        ridge=ridge_value,
        z_dim=int(z.shape[1]),
        tau_dim=int(tau.shape[1]),
        n_train=int(z.shape[0]),
    )


def _predict_history_correlated_joint(model: CorrelatedHistoryGenerativeModel, x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    y = values - model.response_mean[None, :]
    sigma = max(float(model.noise_variance), 1e-12)
    latent_precision = np.linalg.pinv(model.latent_cov)
    precision = latent_precision + (model.latent_map @ model.latent_map.T) / sigma
    h = (y @ model.latent_map.T) / sigma
    latent_centered_hat = np.linalg.solve(precision.T, h.T).T
    z_mean = model.latent_mean[: model.z_dim]
    return (z_mean[None, :] + latent_centered_hat[:, : model.z_dim]).astype(np.float64, copy=False)


def _predict_history_correlated_known(
    model: CorrelatedHistoryGenerativeModel,
    x: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    tau_values = np.asarray(tau, dtype=np.float64)
    if tau_values.ndim != 2 or tau_values.shape[0] != values.shape[0] or tau_values.shape[1] != model.tau_dim:
        raise ValueError(f"tau rows/dim must match x rows/model tau dim, got {tau_values.shape} and {values.shape}")
    z_dim = int(model.z_dim)
    tau_dim = int(model.tau_dim)
    z_mean = model.latent_mean[:z_dim]
    tau_mean = model.latent_mean[z_dim:]
    a = model.latent_map[:z_dim]
    b = model.latent_map[z_dim:]
    cov = model.latent_cov
    cov_zz = cov[:z_dim, :z_dim]
    cov_zt = cov[:z_dim, z_dim : z_dim + tau_dim]
    cov_tz = cov[z_dim : z_dim + tau_dim, :z_dim]
    cov_tt = cov[z_dim : z_dim + tau_dim, z_dim : z_dim + tau_dim]
    cov_tt_inv = np.linalg.pinv(cov_tt)
    conditional_gain = cov_zt @ cov_tt_inv
    conditional_cov = cov_zz - conditional_gain @ cov_tz
    mean_cond_var = float(np.mean(np.diag(conditional_cov))) if conditional_cov.size else 0.0
    conditional_cov = conditional_cov + max(1e-8, 1e-3 * mean_cond_var) * np.eye(z_dim, dtype=np.float64)
    conditional_precision = np.linalg.pinv(conditional_cov)
    tau_centered = tau_values - tau_mean[None, :]
    prior_z_centered_mean = tau_centered @ conditional_gain.T
    y = values - model.response_mean[None, :] - tau_centered @ b
    sigma = max(float(model.noise_variance), 1e-12)
    precision = conditional_precision + (a @ a.T) / sigma
    h = prior_z_centered_mean @ conditional_precision.T + (y @ a.T) / sigma
    z_centered_hat = np.linalg.solve(precision.T, h.T).T
    return (z_mean[None, :] + z_centered_hat).astype(np.float64, copy=False)


def _history_correlated_generative_stats(model: CorrelatedHistoryGenerativeModel, *, mode: str) -> dict[str, Any]:
    z_dim = int(model.z_dim)
    return {
        "history_generative_mode": str(mode),
        "history_generative_model": "linear_gaussian_r_equals_Az_plus_Btau_plus_noise_correlated_feature_history_prior",
        "history_generative_tau_prior": "weighted_empirical_joint_gaussian_feature_endpoint_history",
        "history_generative_noise_variance": float(model.noise_variance),
        "history_generative_tau_cov_trace": float(np.trace(model.latent_cov[z_dim:, z_dim:])),
        "history_generative_feature_map_fro_norm": float(np.linalg.norm(model.latent_map[:z_dim])),
        "history_generative_tau_map_fro_norm": float(np.linalg.norm(model.latent_map[z_dim:])),
        "history_generative_feature_tau_cov_fro_norm": float(np.linalg.norm(model.latent_cov[:z_dim, z_dim:])),
    }


def _fit_source_centered_history_map(
    *,
    x_train: np.ndarray,
    tau_train: np.ndarray,
    source_rows: np.ndarray,
    ridge: float,
    sample_weight: np.ndarray | None,
) -> np.ndarray:
    x = np.asarray(x_train, dtype=np.float64)
    tau = np.asarray(tau_train, dtype=np.float64)
    sources = np.asarray(source_rows, dtype=int)
    if x.ndim != 2 or tau.ndim != 2 or x.shape[0] != tau.shape[0] or x.shape[0] != sources.shape[0]:
        raise ValueError(f"x/tau/source rows must align, got {x.shape}, {tau.shape}, {sources.shape}")
    if tau.shape[1] == 0:
        return np.zeros((0, x.shape[1]), dtype=np.float64)
    if sample_weight is None:
        weights = np.ones(x.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != (x.shape[0],):
            raise ValueError(f"sample_weight must have shape ({x.shape[0]},), got {weights.shape}")
    dx_parts: list[np.ndarray] = []
    dt_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    for source in sorted(set(sources.tolist())):
        mask = sources == int(source)
        if int(np.sum(mask)) < 2:
            continue
        w = weights[mask].astype(np.float64, copy=False)
        w = w / (float(np.sum(w)) + 1e-12)
        x_mean = np.sum(x[mask] * w[:, None], axis=0)
        tau_mean = np.sum(tau[mask] * w[:, None], axis=0)
        dx_parts.append(x[mask] - x_mean[None, :])
        dt_parts.append(tau[mask] - tau_mean[None, :])
        w_parts.append(weights[mask])
    if not dx_parts:
        return np.zeros((tau.shape[1], x.shape[1]), dtype=np.float64)
    dx = np.concatenate(dx_parts, axis=0)
    dt = np.concatenate(dt_parts, axis=0)
    weights_centered = np.concatenate(w_parts, axis=0)
    weights_centered = weights_centered / (float(np.mean(weights_centered)) + 1e-12)
    ridge_value = float(ridge)
    if not np.isfinite(ridge_value) or ridge_value < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    wdt = dt * weights_centered[:, None]
    normal = dt.T @ wdt + ridge_value * np.eye(dt.shape[1], dtype=np.float64)
    return np.linalg.solve(normal, wdt.T @ dx)


def _adjust_response_for_known_history(
    *,
    x: np.ndarray,
    tau: np.ndarray,
    tau_reference: np.ndarray,
    history_map: np.ndarray,
    gamma: float,
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    tau_values = np.asarray(tau, dtype=np.float64)
    reference = np.asarray(tau_reference, dtype=np.float64)
    if values.ndim != 2 or tau_values.ndim != 2 or reference.ndim != 2:
        raise ValueError(f"Expected 2D x/tau/reference matrices, got {values.shape}, {tau_values.shape}, {reference.shape}")
    if tau_values.shape != reference.shape or tau_values.shape[0] != values.shape[0]:
        raise ValueError(f"x/tau/reference rows must align, got {values.shape}, {tau_values.shape}, {reference.shape}")
    return values - float(gamma) * ((tau_values - reference) @ np.asarray(history_map, dtype=np.float64))


def _known_history_repeated_adjusted_prediction(
    *,
    z_all: np.ndarray,
    x_response: np.ndarray,
    tau: np.ndarray,
    tau_reference: np.ndarray,
    source_rows: np.ndarray,
    train_mask: np.ndarray,
    x_test_response: np.ndarray,
    tau_test: np.ndarray,
    tau_test_reference: np.ndarray,
    ridge: float,
    noise_floor: float,
    source_weighting: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Known-history readout with source-centered history nuisance adjustment."""
    z = np.asarray(z_all, dtype=np.float64)
    x = np.asarray(x_response, dtype=np.float64)
    tau_values = np.asarray(tau, dtype=np.float64)
    tau_ref = np.asarray(tau_reference, dtype=np.float64)
    sources = np.asarray(source_rows, dtype=int)
    unique_train = np.asarray(sorted(set(sources[train_mask].tolist())), dtype=int)
    gamma_grid = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    selected_gamma = 0.0
    inner_rows = 0
    inner_folds_used = 0
    if unique_train.size >= 4:
        n_inner = min(3, int(unique_train.size))
        inner_fold_by_source = _assign_source_folds(unique_train, n_folds=n_inner, seed=int(seed))
        sse_by_gamma = np.zeros(gamma_grid.shape[0], dtype=np.float64)
        rows_by_gamma = np.zeros(gamma_grid.shape[0], dtype=np.int64)
        for inner_fold in sorted(set(inner_fold_by_source.values())):
            inner_val = train_mask & np.asarray(
                [inner_fold_by_source.get(int(source), -1) == int(inner_fold) for source in sources],
                dtype=bool,
            )
            inner_train = train_mask & ~inner_val
            if int(np.sum(inner_val)) == 0 or int(np.sum(inner_train)) <= z.shape[1]:
                continue
            weights = _weights_for_sources(sources[inner_train], source_weighting=source_weighting)
            history_map = _fit_source_centered_history_map(
                x_train=x[inner_train],
                tau_train=tau_values[inner_train],
                source_rows=sources[inner_train],
                ridge=float(ridge),
                sample_weight=weights,
            )
            for idx, gamma in enumerate(gamma_grid):
                x_train_adj = _adjust_response_for_known_history(
                    x=x[inner_train],
                    tau=tau_values[inner_train],
                    tau_reference=tau_ref[inner_train],
                    history_map=history_map,
                    gamma=float(gamma),
                )
                x_val_adj = _adjust_response_for_known_history(
                    x=x[inner_val],
                    tau=tau_values[inner_val],
                    tau_reference=tau_ref[inner_val],
                    history_map=history_map,
                    gamma=float(gamma),
                )
                model = _fit_forward_posterior(
                    z_train=z[inner_train],
                    x_train=x_train_adj,
                    ridge=float(ridge),
                    noise_floor=float(noise_floor),
                    sample_weight=weights,
                )
                pred = _predict_z(model, x_val_adj)
                sse_by_gamma[idx] += float(np.sum((z[inner_val] - pred) ** 2))
                rows_by_gamma[idx] += int(np.sum(inner_val))
            inner_rows += int(np.sum(inner_val))
            inner_folds_used += 1
        valid = rows_by_gamma > 0
        if np.any(valid):
            selected_gamma = float(gamma_grid[np.where(valid)[0][int(np.argmin(sse_by_gamma[valid]))]])

    weights = _weights_for_sources(sources[train_mask], source_weighting=source_weighting)
    history_map = _fit_source_centered_history_map(
        x_train=x[train_mask],
        tau_train=tau_values[train_mask],
        source_rows=sources[train_mask],
        ridge=float(ridge),
        sample_weight=weights,
    )
    x_train_adj = _adjust_response_for_known_history(
        x=x[train_mask],
        tau=tau_values[train_mask],
        tau_reference=tau_ref[train_mask],
        history_map=history_map,
        gamma=float(selected_gamma),
    )
    model = _fit_forward_posterior(
        z_train=z[train_mask],
        x_train=x_train_adj,
        ridge=float(ridge),
        noise_floor=float(noise_floor),
        sample_weight=weights,
    )
    x_test_adj = _adjust_response_for_known_history(
        x=x_test_response,
        tau=tau_test,
        tau_reference=tau_test_reference,
        history_map=history_map,
        gamma=float(selected_gamma),
    )
    z_hat = _predict_z(model, x_test_adj)
    return z_hat.astype(np.float64, copy=False), {
        "known_history_feature_model": "source_centered_repeated_history_adjustment_to_zero_reference",
        "known_history_repeated_gamma": float(selected_gamma),
        "known_history_repeated_gamma_grid": ",".join(f"{value:g}" for value in gamma_grid),
        "known_history_repeated_gamma_selection": "inner_source_disjoint_sse_with_gamma0_fallback",
        "known_history_repeated_inner_rows": int(inner_rows),
        "known_history_repeated_inner_folds_used": int(inner_folds_used),
        "known_history_repeated_history_map_fro_norm": float(np.linalg.norm(history_map)),
        "ridge": float(model.ridge),
        "noise_variance": float(model.noise_variance),
        "n_train_samples": int(model.n_train),
        "n_train_sources": int(len(set(sources[train_mask].tolist()))),
        "response_map_fro_norm": float(np.linalg.norm(model.response_map)),
        "posterior_gain_fro_norm": float(np.linalg.norm(model.posterior_gain)),
    }


def _known_history_generative_shrunk_prediction(
    *,
    z_all: np.ndarray,
    x_response: np.ndarray,
    tau: np.ndarray,
    source_rows: np.ndarray,
    train_mask: np.ndarray,
    x_test_response: np.ndarray,
    tau_test: np.ndarray,
    primary_z_all: np.ndarray | None = None,
    primary_x_response: np.ndarray | None = None,
    primary_tau: np.ndarray | None = None,
    primary_source_rows: np.ndarray | None = None,
    primary_train_mask: np.ndarray | None = None,
    ridge: float,
    noise_floor: float,
    source_weighting: str,
    seed: int,
) -> tuple[np.ndarray, HistoryGenerativeModel, dict[str, Any]]:
    """Known-history plug-in correction shrunk toward joint marginalization."""
    sources = np.asarray(source_rows, dtype=int)
    unique_train = np.asarray(sorted(set(sources[train_mask].tolist())), dtype=int)
    beta_grid = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    selected_beta = 0.0
    inner_rows = 0
    inner_folds_used = 0
    validation_contract = "all_training_histories"
    if (
        primary_z_all is not None
        and primary_x_response is not None
        and primary_tau is not None
        and primary_source_rows is not None
        and primary_train_mask is not None
    ):
        val_z_all = np.asarray(primary_z_all, dtype=np.float64)
        val_x_response = np.asarray(primary_x_response, dtype=np.float64)
        val_tau = np.asarray(primary_tau, dtype=np.float64)
        val_sources = np.asarray(primary_source_rows, dtype=int)
        val_train_mask = np.asarray(primary_train_mask, dtype=bool)
        validation_contract = "primary_history_train_sources"
    else:
        val_z_all = np.asarray(z_all, dtype=np.float64)
        val_x_response = np.asarray(x_response, dtype=np.float64)
        val_tau = np.asarray(tau, dtype=np.float64)
        val_sources = sources
        val_train_mask = np.asarray(train_mask, dtype=bool)
    if unique_train.size >= 4:
        n_inner = min(3, int(unique_train.size))
        inner_fold_by_source = _assign_source_folds(unique_train, n_folds=n_inner, seed=int(seed))
        z_val_parts: list[np.ndarray] = []
        joint_val_parts: list[np.ndarray] = []
        known_val_parts: list[np.ndarray] = []
        for inner_fold in sorted(set(inner_fold_by_source.values())):
            inner_train_sources = {
                int(source)
                for source, source_inner_fold in inner_fold_by_source.items()
                if int(source_inner_fold) != int(inner_fold)
            }
            inner_val_sources = {
                int(source)
                for source, source_inner_fold in inner_fold_by_source.items()
                if int(source_inner_fold) == int(inner_fold)
            }
            inner_train = train_mask & np.asarray(
                [inner_fold_by_source.get(int(source), -1) == int(inner_fold) for source in sources],
                dtype=bool,
            )
            inner_train = train_mask & ~inner_train
            inner_val = val_train_mask & np.asarray([int(source) in inner_val_sources for source in val_sources], dtype=bool)
            if int(np.sum(inner_val)) == 0 or int(np.sum(inner_train)) <= z_all.shape[1]:
                continue
            weights = _weights_for_sources(sources[inner_train], source_weighting=source_weighting)
            model = _fit_history_generative_model(
                z_train=z_all[inner_train],
                x_train=x_response[inner_train],
                tau_train=tau[inner_train],
                ridge=float(ridge),
                noise_floor=float(noise_floor),
                sample_weight=weights,
            )
            z_val_parts.append(val_z_all[inner_val])
            joint_val_parts.append(_predict_history_joint(model, val_x_response[inner_val]))
            known_val_parts.append(_predict_history_known(model, val_x_response[inner_val], val_tau[inner_val]))
            inner_rows += int(np.sum(inner_val))
            inner_folds_used += 1
        if z_val_parts:
            z_val = np.concatenate(z_val_parts, axis=0)
            joint_val = np.concatenate(joint_val_parts, axis=0)
            known_val = np.concatenate(known_val_parts, axis=0)
            delta_val = known_val - joint_val
            sse_by_beta = np.asarray(
                [float(np.sum((z_val - (joint_val + beta * delta_val)) ** 2)) for beta in beta_grid],
                dtype=np.float64,
            )
            selected_beta = float(beta_grid[int(np.argmin(sse_by_beta))])

    weights = _weights_for_sources(sources[train_mask], source_weighting=source_weighting)
    model = _fit_history_generative_model(
        z_train=z_all[train_mask],
        x_train=x_response[train_mask],
        tau_train=tau[train_mask],
        ridge=float(ridge),
        noise_floor=float(noise_floor),
        sample_weight=weights,
    )
    joint_test = _predict_history_joint(model, x_test_response)
    known_test = _predict_history_known(model, x_test_response, tau_test)
    z_hat = joint_test + float(selected_beta) * (known_test - joint_test)
    stats = {
        "known_history_generative_beta": float(selected_beta),
        "known_history_generative_beta_grid": ",".join(f"{value:g}" for value in beta_grid),
        "known_history_generative_beta_selection": "inner_source_disjoint_sse_with_beta0_joint_fallback",
        "known_history_generative_beta_validation_contract": validation_contract,
        "known_history_generative_inner_rows": int(inner_rows),
        "known_history_generative_inner_folds_used": int(inner_folds_used),
    }
    return z_hat.astype(np.float64, copy=False), model, stats


def _run_crossfit(
    *,
    dataset: EndpointDataset,
    feature_table: Any,
    feature_weights: np.ndarray | None,
    feature_space_modes: list[str],
    feature_dim: int,
    n_folds: int,
    fold_seed: int,
    ridge: float,
    noise_floor: float,
    source_weighting: str,
    primary_condition: str,
    joint_conditions: list[str],
    history_coordinate_mode: str,
    history_dim: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_by_source = _assign_source_folds(
        dataset.rows["true_source_row"].to_numpy(dtype=int),
        n_folds=int(n_folds),
        seed=int(fold_seed),
    )
    test_folds = dataset.rows["true_source_row"].map(fold_by_source).to_numpy(dtype=int)
    trial_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    canonical_modes = list(dict.fromkeys(_feature_space_config(mode)["canonical_mode"] for mode in feature_space_modes))
    observers = _observer_specs(primary_condition=primary_condition, joint_conditions=joint_conditions)
    train_banks = {
        name: _endpoint_bank(
            dataset,
            bank_name=name,
            primary_condition=primary_condition,
            joint_conditions=joint_conditions,
        )
        for name in sorted({observer.train_bank for observer in observers})
    }
    test_inputs = {
        name: _endpoint_bank(
            dataset,
            bank_name=name,
            primary_condition=primary_condition,
            joint_conditions=joint_conditions,
        )
        for name in sorted({observer.test_input for observer in observers})
    }

    for mode in canonical_modes:
        for fold in sorted(set(test_folds.tolist())):
            test_mask = test_folds == int(fold)
            heldout_sources = {
                int(source)
                for source, source_fold in fold_by_source.items()
                if int(source_fold) == int(fold)
            }
            fold_train_sources = np.asarray(
                [int(source) for source in feature_table.source_rows.tolist() if int(source) not in heldout_sources],
                dtype=int,
            )
            transform: FeatureTransform = _fit_feature_transform(
                feature_table,
                fit_sources=fold_train_sources,
                feature_dim=int(feature_dim),
                feature_space_mode=mode,
                feature_weights=feature_weights,
            )
            baseline_sources = np.asarray(
                [
                    int(source)
                    for source, source_fold in fold_by_source.items()
                    if int(source_fold) != int(fold)
                ],
                dtype=int,
            )
            z_train_baseline = _transform_feature_sources(transform, feature_table, baseline_sources)
            z_train_mean = np.mean(z_train_baseline, axis=0)
            test_sources = dataset.rows.loc[test_mask, "true_source_row"].to_numpy(dtype=int)
            z_true = _transform_feature_sources(transform, feature_table, test_sources)
            base_sources = dataset.rows["true_source_row"].to_numpy(dtype=int)
            z_primary_all = _transform_feature_sources(transform, feature_table, base_sources)
            tau_primary_raw = dataset.tau_by_condition[primary_condition]
            joint_response_raw = np.concatenate(
                [dataset.x_by_condition[condition] for condition in joint_conditions],
                axis=0,
            ).astype(np.float32, copy=False)
            joint_tau_raw = np.concatenate(
                [dataset.tau_by_condition[condition] for condition in joint_conditions],
                axis=0,
            ).astype(np.float32, copy=False)
            joint_source_rows = np.tile(base_sources, len(joint_conditions)).astype(int, copy=False)
            joint_train_mask_for_projection = np.asarray(
                [fold_by_source.get(int(source), -1) != int(fold) for source in joint_source_rows],
                dtype=bool,
            )
            history_projection = _fit_history_coordinate_projection(
                fit_tau=joint_tau_raw[joint_train_mask_for_projection],
                mode=str(history_coordinate_mode),
                n_components=int(history_dim),
            )
            tau_primary = _apply_history_coordinate_projection(history_projection, tau_primary_raw)
            tau_zero_primary = _apply_history_coordinate_projection(history_projection, np.zeros_like(tau_primary_raw))
            tau_joint = _apply_history_coordinate_projection(history_projection, joint_tau_raw)
            tau_zero_joint = np.tile(tau_zero_primary, (len(joint_conditions), 1)).astype(np.float32, copy=False)
            history_coordinate_stats = _history_coordinate_stats(history_projection)
            for observer in observers:
                bank = train_banks[observer.train_bank]
                test_bank = test_inputs[observer.test_input]
                train_mask = np.asarray([fold_by_source.get(int(source), -1) != int(fold) for source in bank.source_rows])
                if int(np.sum(train_mask)) <= transform.feature_dim:
                    raise ValueError(f"Fold {fold} has too few training samples for {observer.slug} / {mode}")
                z_bank = _transform_feature_sources(transform, feature_table, bank.source_rows)
                train_weights = _weights_for_sources(bank.source_rows[train_mask], source_weighting=source_weighting)
                model_response_dim = int(bank.x.shape[1])
                model_test_response_dim = int(test_bank.x.shape[1])
                if observer.slug in {
                    "known_history_generative",
                    "joint_history_generative",
                    "zero_history_generative_on_motion",
                }:
                    response_x = joint_response_raw
                    tau_x = tau_joint
                    zero_tau_x = tau_zero_primary
                    primary_response_x = test_inputs["primary_response"].x
                    model_response_dim = int(response_x.shape[1])
                    model_test_response_dim = int(primary_response_x.shape[1])
                    if observer.slug == "known_history_generative":
                        z_hat, gen_model, known_gen_stats = _known_history_generative_shrunk_prediction(
                            z_all=z_bank,
                            x_response=response_x,
                            tau=tau_x,
                            source_rows=bank.source_rows,
                            train_mask=train_mask,
                            x_test_response=primary_response_x[test_mask],
                            tau_test=tau_primary[test_mask],
                            primary_z_all=z_primary_all,
                            primary_x_response=primary_response_x,
                            primary_tau=tau_primary,
                            primary_source_rows=base_sources,
                            primary_train_mask=~test_mask,
                            ridge=float(ridge),
                            noise_floor=float(noise_floor),
                            source_weighting=str(source_weighting),
                            seed=int(fold_seed) + 2027 * int(fold),
                        )
                        gen_mode = "known_tau_shrunk_toward_joint"
                    else:
                        gen_model = _fit_history_generative_model(
                            z_train=z_bank[train_mask],
                            x_train=response_x[train_mask],
                            tau_train=tau_x[train_mask],
                            ridge=float(ridge),
                            noise_floor=float(noise_floor),
                            sample_weight=train_weights,
                        )
                        known_gen_stats = {}
                        if observer.slug == "zero_history_generative_on_motion":
                            z_hat = _predict_history_known(gen_model, primary_response_x[test_mask], zero_tau_x[test_mask])
                            gen_mode = "zero_tau_plug_in_on_motion_response"
                        else:
                            z_hat = _predict_history_joint(gen_model, primary_response_x[test_mask])
                            gen_mode = "tau_marginalized_joint_posterior"
                    model_stats = {
                        "ridge": float(gen_model.ridge),
                        "noise_variance": float(gen_model.noise_variance),
                        "n_train_samples": int(gen_model.n_train),
                        "n_train_sources": int(len(set(bank.source_rows[train_mask].tolist()))),
                        "response_map_fro_norm": float(np.linalg.norm(gen_model.feature_map)),
                        "posterior_gain_fro_norm": float("nan"),
                        **_history_generative_stats(gen_model, mode=gen_mode),
                        **history_coordinate_stats,
                        **known_gen_stats,
                    }
                elif observer.slug in {
                    "known_history_correlated_generative",
                    "joint_history_correlated_generative",
                    "zero_history_correlated_generative_on_motion",
                }:
                    response_x = joint_response_raw
                    tau_x = tau_joint
                    primary_response_x = test_inputs["primary_response"].x
                    corr_model = _fit_correlated_history_generative_model(
                        z_train=z_bank[train_mask],
                        x_train=response_x[train_mask],
                        tau_train=tau_x[train_mask],
                        ridge=float(ridge),
                        noise_floor=float(noise_floor),
                        sample_weight=train_weights,
                    )
                    if observer.slug == "known_history_correlated_generative":
                        z_hat = _predict_history_correlated_known(
                            corr_model,
                            primary_response_x[test_mask],
                            tau_primary[test_mask],
                        )
                        corr_mode = "known_tau_conditional_feature_history_prior"
                    elif observer.slug == "zero_history_correlated_generative_on_motion":
                        z_hat = _predict_history_correlated_known(
                            corr_model,
                            primary_response_x[test_mask],
                            tau_zero_primary[test_mask],
                        )
                        corr_mode = "zero_tau_conditional_feature_history_prior_on_motion_response"
                    else:
                        z_hat = _predict_history_correlated_joint(corr_model, primary_response_x[test_mask])
                        corr_mode = "tau_marginalized_joint_feature_history_prior"
                    model_response_dim = int(response_x.shape[1])
                    model_test_response_dim = int(primary_response_x.shape[1])
                    model_stats = {
                        "ridge": float(corr_model.ridge),
                        "noise_variance": float(corr_model.noise_variance),
                        "n_train_samples": int(corr_model.n_train),
                        "n_train_sources": int(len(set(bank.source_rows[train_mask].tolist()))),
                        "response_map_fro_norm": float(np.linalg.norm(corr_model.latent_map[: corr_model.z_dim])),
                        "posterior_gain_fro_norm": float("nan"),
                        **_history_correlated_generative_stats(corr_model, mode=corr_mode),
                        **history_coordinate_stats,
                    }
                elif observer.slug in {"known_history", "known_history_interaction"}:
                    if observer.slug == "known_history_interaction":
                        x_response_tau = _response_tau_interaction_features(
                            test_inputs["primary_response"].x,
                            tau_primary,
                        )
                        known_history_feature_model = "response_tau_interaction"
                    else:
                        x_response_tau = _concat_response_tau(test_inputs["primary_response"].x, tau_primary)
                        known_history_feature_model = "response_plus_tau"
                    model_response_dim = int(x_response_tau.shape[1])
                    model_test_response_dim = int(x_response_tau.shape[1])
                    z_hat, special_stats = _known_history_nested_prediction(
                        z_all=z_bank,
                        x_response=test_inputs["primary_response"].x,
                        x_response_tau=x_response_tau,
                        source_rows=bank.source_rows,
                        train_mask=train_mask,
                        test_mask=test_mask,
                        ridge=float(ridge),
                        noise_floor=float(noise_floor),
                        source_weighting=str(source_weighting),
                        seed=int(fold_seed) + 1009 * int(fold),
                    )
                    model_stats = {
                        "ridge": float(ridge),
                        "noise_variance": float("nan"),
                        "n_train_samples": int(np.sum(train_mask)),
                        "n_train_sources": int(len(set(bank.source_rows[train_mask].tolist()))),
                        "response_map_fro_norm": float("nan"),
                        "posterior_gain_fro_norm": float("nan"),
                        "known_history_feature_model": known_history_feature_model,
                        **history_coordinate_stats,
                        **special_stats,
                    }
                elif observer.slug in {"known_history_multi_direct", "known_history_multi_interaction"}:
                    primary_response_x = test_inputs["primary_response"].x
                    if observer.slug == "known_history_multi_interaction":
                        x_train_known = _response_tau_interaction_features(joint_response_raw, tau_joint)
                        x_test_known = _response_tau_interaction_features(primary_response_x, tau_primary)
                        known_history_feature_model = "multi_history_response_tau_interaction"
                    else:
                        x_train_known = _concat_response_tau(joint_response_raw, tau_joint)
                        x_test_known = _concat_response_tau(primary_response_x, tau_primary)
                        known_history_feature_model = "multi_history_response_plus_tau"
                    model = _fit_forward_posterior(
                        z_train=z_bank[train_mask],
                        x_train=x_train_known[train_mask],
                        ridge=float(ridge),
                        noise_floor=float(noise_floor),
                        sample_weight=train_weights,
                    )
                    z_hat = _predict_z(model, x_test_known[test_mask])
                    model_response_dim = int(x_train_known.shape[1])
                    model_test_response_dim = int(x_test_known.shape[1])
                    model_stats = {
                        "ridge": float(model.ridge),
                        "noise_variance": float(model.noise_variance),
                        "n_train_samples": int(model.n_train),
                        "n_train_sources": int(len(set(bank.source_rows[train_mask].tolist()))),
                        "response_map_fro_norm": float(np.linalg.norm(model.response_map)),
                        "posterior_gain_fro_norm": float(np.linalg.norm(model.posterior_gain)),
                        "known_history_feature_model": known_history_feature_model,
                        **history_coordinate_stats,
                    }
                elif observer.slug == "known_history_repeated_adjusted":
                    primary_response_x = test_inputs["primary_response"].x
                    z_hat, repeated_stats = _known_history_repeated_adjusted_prediction(
                        z_all=z_bank,
                        x_response=joint_response_raw,
                        tau=tau_joint,
                        tau_reference=tau_zero_joint,
                        source_rows=bank.source_rows,
                        train_mask=train_mask,
                        x_test_response=primary_response_x[test_mask],
                        tau_test=tau_primary[test_mask],
                        tau_test_reference=tau_zero_primary[test_mask],
                        ridge=float(ridge),
                        noise_floor=float(noise_floor),
                        source_weighting=str(source_weighting),
                        seed=int(fold_seed) + 3037 * int(fold),
                    )
                    model_response_dim = int(primary_response_x.shape[1])
                    model_test_response_dim = int(primary_response_x.shape[1])
                    model_stats = {
                        **repeated_stats,
                        **history_coordinate_stats,
                    }
                else:
                    model = _fit_forward_posterior(
                        z_train=z_bank[train_mask],
                        x_train=bank.x[train_mask],
                        ridge=float(ridge),
                        noise_floor=float(noise_floor),
                        sample_weight=train_weights,
                    )
                    z_hat = _predict_z(model, test_bank.x[test_mask])
                    model_stats = {
                        "ridge": float(model.ridge),
                        "noise_variance": float(model.noise_variance),
                        "n_train_samples": int(model.n_train),
                        "n_train_sources": int(len(set(bank.source_rows[train_mask].tolist()))),
                        "response_map_fro_norm": float(np.linalg.norm(model.response_map)),
                        "posterior_gain_fro_norm": float(np.linalg.norm(model.posterior_gain)),
                        "history_coordinate_mode": "none",
                        "history_dim_requested": 0,
                        "history_dim_used": 0,
                        "history_coordinate_variance_fraction": float("nan"),
                    }
                model_rows.append(
                    {
                        "decoder_mode": "linear_gaussian",
                        "observer_mode": observer.slug,
                        "observer_label": observer.label,
                        "train_bank": observer.train_bank,
                        "test_input": observer.test_input,
                        "primary_condition": primary_condition,
                        "joint_conditions": ",".join(joint_conditions),
                        "latent": transform.latent,
                        "feature_space_mode": transform.feature_space_mode,
                        "fold": int(fold),
                        "n_fit_sources": int(transform.n_fit_sources),
                        "n_test_rows": int(np.sum(test_mask)),
                        "feature_dim": int(transform.feature_dim),
                        "raw_feature_dim": int(transform.raw_feature_dim),
                        "response_dim": model_response_dim,
                        "test_response_dim": model_test_response_dim,
                        "feature_fit_scope": transform.fit_scope,
                        "feature_preprocessing": transform.preprocessing,
                        "feature_whitened": bool(transform.whitened),
                        "feature_weighted": bool(transform.weighted),
                        "feature_variance_fraction": float(transform.explained_variance_sum),
                        "r2_cv_train_baseline": "source_fold_train_feature_mean",
                        "source_weighting": str(source_weighting),
                        "train_weight_min": float(np.min(train_weights)) if train_weights.size else float("nan"),
                        "train_weight_max": float(np.max(train_weights)) if train_weights.size else float("nan"),
                        **model_stats,
                        "interpretation": observer.interpretation,
                    }
                )
                test_meta = dataset.rows.loc[test_mask].reset_index(drop=True)
                for row_index, meta in enumerate(test_meta.to_dict(orient="records")):
                    row = dict(meta)
                    row.update(
                        {
                            "decoder_mode": "linear_gaussian",
                            "observer_mode": observer.slug,
                            "observer_label": observer.label,
                            "train_bank": observer.train_bank,
                            "test_input": observer.test_input,
                            "primary_condition": primary_condition,
                            "joint_conditions": ",".join(joint_conditions),
                            "latent": transform.latent,
                            "feature_space_mode": transform.feature_space_mode,
                            "fold": int(fold),
                            "n_train_samples": int(model_stats["n_train_samples"]),
                            "n_train_sources": int(model_stats["n_train_sources"]),
                            "source_weighting": str(source_weighting),
                            "n_fit_sources": int(transform.n_fit_sources),
                            "feature_fit_scope": transform.fit_scope,
                            "feature_preprocessing": transform.preprocessing,
                            "feature_whitened": bool(transform.whitened),
                            "feature_weighted": bool(transform.weighted),
                            "feature_variance_fraction": float(transform.explained_variance_sum),
                            "r2_cv_train_baseline": "source_fold_train_feature_mean",
                        }
                    )
                    row.update(_metrics(z_hat[row_index], z_true[row_index], train_mean=z_train_mean))
                    trial_rows.append(row)
    return pd.DataFrame(trial_rows), pd.DataFrame(model_rows)


def _summarize(trials: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "decoder_mode",
        "latent",
        "feature_space_mode",
        "observer_mode",
        "observer_label",
        "train_bank",
        "test_input",
        "observation_scale",
    ]
    summary = (
        trials.groupby(group_cols, as_index=False)
        .agg(
            n=("feature_cosine", "size"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_mse=("feature_mse", "mean"),
            median_feature_mse=("feature_mse", "median"),
            feature_sse=("feature_sse", "sum"),
            feature_sst_train_baseline=("feature_sst_train_baseline", "sum"),
            mean_feature_rmse=("feature_rmse", "mean"),
            median_feature_pred_norm=("feature_pred_norm", "median"),
            median_feature_true_norm=("feature_true_norm", "median"),
        )
        .sort_values(["observation_scale", "observer_mode"])
    )
    overall = (
        trials.groupby(
            ["decoder_mode", "latent", "feature_space_mode", "observer_mode", "observer_label", "train_bank", "test_input"],
            as_index=False,
        )
        .agg(
            n=("feature_cosine", "size"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_mse=("feature_mse", "mean"),
            median_feature_mse=("feature_mse", "median"),
            feature_sse=("feature_sse", "sum"),
            feature_sst_train_baseline=("feature_sst_train_baseline", "sum"),
            mean_feature_rmse=("feature_rmse", "mean"),
            median_feature_pred_norm=("feature_pred_norm", "median"),
            median_feature_true_norm=("feature_true_norm", "median"),
        )
        .sort_values("observer_mode")
    )
    summary["R2_cv"] = 1.0 - summary["feature_sse"] / summary["feature_sst_train_baseline"]
    summary.loc[summary["feature_sst_train_baseline"] <= 1e-12, "R2_cv"] = np.nan
    overall["R2_cv"] = 1.0 - overall["feature_sse"] / overall["feature_sst_train_baseline"]
    overall.loc[overall["feature_sst_train_baseline"] <= 1e-12, "R2_cv"] = np.nan
    overall["observation_scale"] = "all"
    return pd.concat([summary, overall[summary.columns]], ignore_index=True)


def _contrasts(trials: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    key_cols = [
        "decoder_mode",
        "latent",
        "feature_space_mode",
        "sample_index",
        "observation_scale",
        "true_source_row",
    ]
    pivot = trials.pivot_table(index=key_cols, columns="observer_mode", values="feature_cosine", aggfunc="first")
    pairs = [
        ("known_history", "joint_history_response_only", "known_minus_joint"),
        ("known_history_interaction", "joint_history_response_only", "known_interaction_minus_joint"),
        ("known_history_multi_direct", "joint_history_response_only", "known_multi_direct_minus_joint_response_only"),
        ("known_history_multi_interaction", "joint_history_response_only", "known_multi_interaction_minus_joint_response_only"),
        ("known_history_repeated_adjusted", "joint_history_response_only", "known_repeated_adjusted_minus_joint_response_only"),
        ("known_history_multi_direct", "joint_history_generative", "known_multi_direct_minus_joint_generative"),
        ("known_history_multi_interaction", "joint_history_generative", "known_multi_interaction_minus_joint_generative"),
        ("known_history_repeated_adjusted", "joint_history_generative", "known_repeated_adjusted_minus_joint_generative"),
        ("joint_history_response_only", "zero_history_on_motion", "joint_minus_zero_history"),
        ("known_history", "zero_history_on_motion", "known_minus_zero_history"),
        ("known_history_interaction", "zero_history_on_motion", "known_interaction_minus_zero_history"),
        ("known_history_multi_direct", "zero_history_generative_on_motion", "known_multi_direct_minus_zero_generative"),
        ("known_history_multi_interaction", "zero_history_generative_on_motion", "known_multi_interaction_minus_zero_generative"),
        ("known_history_repeated_adjusted", "zero_history_generative_on_motion", "known_repeated_adjusted_minus_zero_generative"),
        ("joint_history_response_only", "static_history", "joint_motion_minus_static_history"),
        ("known_history", "static_history", "known_motion_minus_static_history"),
        ("known_history_interaction", "static_history", "known_interaction_motion_minus_static_history"),
        ("known_history_multi_direct", "static_history", "known_multi_direct_motion_minus_static_history"),
        ("known_history_multi_interaction", "static_history", "known_multi_interaction_motion_minus_static_history"),
        ("known_history_repeated_adjusted", "static_history", "known_repeated_adjusted_motion_minus_static_history"),
        ("known_history_generative", "joint_history_generative", "known_generative_minus_joint_generative"),
        ("joint_history_generative", "zero_history_generative_on_motion", "joint_generative_minus_zero_generative"),
        ("known_history_generative", "zero_history_generative_on_motion", "known_generative_minus_zero_generative"),
        ("joint_history_generative", "static_history", "joint_generative_motion_minus_static_history"),
        ("known_history_generative", "static_history", "known_generative_motion_minus_static_history"),
        ("known_history_correlated_generative", "joint_history_correlated_generative", "known_correlated_minus_joint_correlated"),
        (
            "joint_history_correlated_generative",
            "zero_history_correlated_generative_on_motion",
            "joint_correlated_minus_zero_correlated",
        ),
        (
            "known_history_correlated_generative",
            "zero_history_correlated_generative_on_motion",
            "known_correlated_minus_zero_correlated",
        ),
        ("joint_history_correlated_generative", "static_history", "joint_correlated_motion_minus_static_history"),
        ("known_history_correlated_generative", "static_history", "known_correlated_motion_minus_static_history"),
    ]
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for lhs, rhs, contrast in pairs:
        if lhs not in pivot.columns or rhs not in pivot.columns:
            continue
        vals = (pivot[lhs] - pivot[rhs]).rename("delta").reset_index()
        vals = vals[np.isfinite(vals["delta"].to_numpy(dtype=float))]
        for scale_value, scale_rows in vals.groupby("observation_scale", sort=True):
            for (decoder_mode, latent, feature_space_mode), mode_rows in scale_rows.groupby(
                ["decoder_mode", "latent", "feature_space_mode"],
                sort=True,
            ):
                values = mode_rows["delta"].to_numpy(dtype=float)
                mean, lo, hi = _bootstrap_mean(
                    values,
                    rng,
                    int(n_boot),
                    clusters=mode_rows["true_source_row"].to_numpy(dtype=int),
                )
                rows.append(
                    {
                        "decoder_mode": str(decoder_mode),
                        "latent": str(latent),
                        "feature_space_mode": str(feature_space_mode),
                        "contrast": contrast,
                        "lhs": lhs,
                        "rhs": rhs,
                        "observation_scale": float(scale_value),
                        "mean_feature_cosine_delta": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "bootstrap_unit": "true_source_row",
                        "fraction_positive": float(np.mean(values > 0.0)) if values.size else float("nan"),
                        "n": int(values.size),
                        "n_bootstrap_clusters": int(mode_rows["true_source_row"].nunique()),
                    }
                )
        for (decoder_mode, latent, feature_space_mode), mode_rows in vals.groupby(
            ["decoder_mode", "latent", "feature_space_mode"],
            sort=True,
        ):
            values = mode_rows["delta"].to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_mean(
                values,
                rng,
                int(n_boot),
                clusters=mode_rows["true_source_row"].to_numpy(dtype=int),
            )
            rows.append(
                {
                    "decoder_mode": str(decoder_mode),
                    "latent": str(latent),
                    "feature_space_mode": str(feature_space_mode),
                    "contrast": contrast,
                    "lhs": lhs,
                    "rhs": rhs,
                    "observation_scale": "all",
                    "mean_feature_cosine_delta": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "bootstrap_unit": "true_source_row",
                    "fraction_positive": float(np.mean(values > 0.0)) if values.size else float("nan"),
                    "n": int(values.size),
                    "n_bootstrap_clusters": int(mode_rows["true_source_row"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _filter_plot(summary: pd.DataFrame, contrasts: pd.DataFrame, *, latent: str, feature_space_mode: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    available = summary[["latent", "feature_space_mode"]].drop_duplicates()
    if available.empty:
        raise ValueError("No summary rows are available for plotting")
    selected_mode = str(feature_space_mode)
    if selected_mode == "auto":
        match = available[available["latent"].astype(str).eq(str(latent))]
        if match.empty:
            match = available
        selected_mode = str(match.iloc[0]["feature_space_mode"])
    elif selected_mode not in set(available["feature_space_mode"].astype(str)):
        choices = available.to_dict(orient="records")
        raise ValueError(f"Requested plot feature-space mode {selected_mode!r} is absent; available={choices}")
    plot_summary = summary[
        summary["latent"].astype(str).eq(str(latent))
        & summary["feature_space_mode"].astype(str).eq(selected_mode)
    ].copy()
    plot_contrasts = contrasts[
        contrasts["latent"].astype(str).eq(str(latent))
        & contrasts["feature_space_mode"].astype(str).eq(selected_mode)
    ].copy()
    if plot_summary.empty:
        raise ValueError(f"No plot rows for latent={latent!r}, feature_space_mode={selected_mode!r}")
    return plot_summary, plot_contrasts, selected_mode


def _scale_x(values: pd.Series) -> np.ndarray:
    mapping = {0.5: 0.0, 1.0: 1.0, 2.0: 2.0}
    return values.astype(float).map(mapping).fillna(values.astype(float)).to_numpy(dtype=float)


def _plot(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    out_dir: Path,
    *,
    latent: str,
    feature_space_mode: str,
) -> tuple[Path, Path, str]:
    _configure_matplotlib()
    summary, contrasts, selected_mode = _filter_plot(
        summary,
        contrasts,
        latent=str(latent),
        feature_space_mode=str(feature_space_mode),
    )
    scale_summary = summary[summary["observation_scale"].astype(str) != "all"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

    colors = {
        "static_history": "#66717d",
        "known_history": "#111827",
        "known_history_interaction": "#7c2d12",
        "known_history_multi_direct": "#581c87",
        "known_history_multi_interaction": "#be123c",
        "known_history_repeated_adjusted": "#047857",
        "joint_history_response_only": "#235789",
        "zero_history_on_motion": "#8a5ca8",
        "known_history_generative": "#0f172a",
        "joint_history_generative": "#0f766e",
        "zero_history_generative_on_motion": "#b45309",
        "known_history_correlated_generative": "#164e63",
        "joint_history_correlated_generative": "#2563eb",
        "zero_history_correlated_generative_on_motion": "#ca8a04",
        "known_minus_joint": "#111827",
        "known_interaction_minus_joint": "#7c2d12",
        "known_multi_direct_minus_joint_response_only": "#581c87",
        "known_multi_interaction_minus_joint_response_only": "#be123c",
        "known_repeated_adjusted_minus_joint_response_only": "#047857",
        "known_multi_direct_minus_joint_generative": "#6d28d9",
        "known_multi_interaction_minus_joint_generative": "#e11d48",
        "known_repeated_adjusted_minus_joint_generative": "#059669",
        "joint_minus_zero_history": "#235789",
        "known_minus_zero_history": "#2f8f6a",
        "known_interaction_minus_zero_history": "#9a3412",
        "known_multi_direct_minus_zero_generative": "#7e22ce",
        "known_multi_interaction_minus_zero_generative": "#f43f5e",
        "known_repeated_adjusted_minus_zero_generative": "#10b981",
        "joint_motion_minus_static_history": "#4c78a8",
        "known_motion_minus_static_history": "#0f766e",
        "known_interaction_motion_minus_static_history": "#c2410c",
        "known_multi_direct_motion_minus_static_history": "#9333ea",
        "known_multi_interaction_motion_minus_static_history": "#fb7185",
        "known_repeated_adjusted_motion_minus_static_history": "#34d399",
        "known_generative_minus_joint_generative": "#0f172a",
        "joint_generative_minus_zero_generative": "#0f766e",
        "known_generative_minus_zero_generative": "#2f8f6a",
        "joint_generative_motion_minus_static_history": "#14b8a6",
        "known_generative_motion_minus_static_history": "#115e59",
        "known_correlated_minus_joint_correlated": "#164e63",
        "joint_correlated_minus_zero_correlated": "#2563eb",
        "known_correlated_minus_zero_correlated": "#0284c7",
        "joint_correlated_motion_minus_static_history": "#3b82f6",
        "known_correlated_motion_minus_static_history": "#0891b2",
    }
    ax = axes[0]
    for observer, block in scale_summary.groupby("observer_mode", sort=False):
        block = block.sort_values("observation_scale")
        ax.plot(
            _scale_x(block["observation_scale"]),
            block["R2_cv"].to_numpy(dtype=float),
            marker="o",
            lw=2.0 if observer != "static_history" else 1.6,
            linestyle="--" if str(observer).endswith("_generative") or "generative" in str(observer) else (":" if observer == "static_history" else "-"),
            color=colors.get(str(observer), "#111827"),
            label=str(block["observer_label"].iloc[0]),
        )
    ax.axhline(0.0, color="#6b7280", lw=0.8)
    ax.set_title("A. terminal feature recovery")
    ax.set_ylabel("pooled $R^2_{cv}$")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.legend(frameon=False, loc="best")
    _clean_axis(ax)

    ax = axes[1]
    scale_contrasts = contrasts[contrasts["observation_scale"].astype(str) != "all"].copy()
    offsets = np.linspace(-0.12, 0.12, max(1, scale_contrasts["contrast"].nunique()))
    for offset, (contrast, block) in zip(offsets, scale_contrasts.groupby("contrast", sort=False), strict=False):
        block = block.sort_values("observation_scale")
        x = _scale_x(block["observation_scale"]) + float(offset)
        y = block["mean_feature_cosine_delta"].to_numpy(dtype=float)
        yerr = np.vstack([y - block["ci_low"].to_numpy(dtype=float), block["ci_high"].to_numpy(dtype=float) - y])
        label = str(contrast).replace("_", " ")
        ax.errorbar(x, y, yerr=yerr, marker="o", lw=1.5, capsize=2.5, color=colors.get(str(contrast), "#111827"), label=label)
    ax.axhline(0.0, color="#6b7280", lw=0.8)
    ax.set_title("B. paired observer contrasts")
    ax.set_ylabel("feature cosine difference")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.legend(frameon=False, loc="best")
    _clean_axis(ax)

    png = out_dir / "endpoint_history_feature_readout.png"
    pdf = out_dir / "endpoint_history_feature_readout.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf, selected_mode


def _write_readme(
    *,
    out_dir: Path,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    selected_mode = str(manifest["plot_feature_space_mode"])
    plotted = summary[
        (summary["feature_space_mode"].astype(str) == selected_mode)
        & (summary["observation_scale"].astype(str) == "1.0")
    ].copy()
    scale1_lines = ["observer_mode,R2_cv,mean_feature_cosine,n"]
    for row in plotted.sort_values("observer_mode").itertuples(index=False):
        scale1_lines.append(
            f"{row.observer_mode},{float(row.R2_cv):.4f},{float(row.mean_feature_cosine):.4f},{int(row.n)}"
        )
    contrast_lines = ["contrast,observation_scale,mean_cosine_delta,ci_low,ci_high,n"]
    selected_contrasts = contrasts[contrasts["feature_space_mode"].astype(str) == selected_mode].copy()
    for row in selected_contrasts.sort_values(["observation_scale", "contrast"]).itertuples(index=False):
        contrast_lines.append(
            f"{row.contrast},{row.observation_scale},{float(row.mean_feature_cosine_delta):.4f},"
            f"{float(row.ci_low):.4f},{float(row.ci_high):.4f},{int(row.n)}"
        )

    lines = [
        "# Endpoint-History Feature Readout",
        "",
        "This diagnostic is an endpoint-aligned prototype for the joint/history",
        "decoder. It tests whether the model's finite 32-frame history helps",
        "recover endpoint-centered image features when the final retinal position",
        "is matched across conditions.",
        "",
        "Contract:",
        "",
        "```text",
        "tau_endpoint[t] = tau[t] - tau[-1]",
        "tau_endpoint[-1] = 0",
        "response feature = terminal twin response frame only",
        "target = locked endpoint/crop-centered feature embedding",
        "```",
        "",
        f"`n_timepoints = {manifest['assay']['n_timepoints']}` and "
        f"`terminal_frames = {manifest['assay']['terminal_frames']}`.",
        "The static condition uses zero displacement for every history frame.",
        f"History coordinate mode: `{manifest['assay']['history_coordinate_mode']}` "
        f"with requested dimension `{manifest['assay']['history_dim']}`.",
        "",
        f"Plotted feature-space mode: `{selected_mode}`.",
        f"Response population: `{manifest['population']['population_name']}` "
        f"({manifest['population']['population_n_units']} units).",
        f"Response basis: `{manifest['basis']['response_basis_mode']}`.",
        f"Primary motion condition: `{manifest['assay']['primary_condition']}`.",
        f"Joint/history training conditions: `{','.join(manifest['assay']['joint_conditions'])}`.",
        "",
        "Observer modes:",
        "",
        "- `static_history`: train/test on static endpoint history.",
        "- `known_history`: response-only base plus inner-validated path-conditioned residual correction, with alpha=0 fallback.",
        "- `known_history_interaction`: known-history diagnostic with response-by-history interaction terms and the same alpha=0 fallback.",
        "- `known_history_multi_direct`: direct known-history readout trained on repeated endpoint histories.",
        "- `known_history_multi_interaction`: gain-field known-history readout trained on repeated endpoint histories.",
        "- `known_history_repeated_adjusted`: source-centered repeated-measures history adjustment followed by feature readout.",
        "- `joint_history_response_only`: response-only hidden-history readout.",
        "- `zero_history_on_motion`: static-history readout applied to motion-history response.",
        "- `known_history_generative`: linear-Gaussian `r_T = A z + B tau + noise`, with true-tau correction shrunk toward joint by inner validation.",
        "- `joint_history_generative`: same model, with endpoint history treated as a latent Gaussian variable.",
        "- `zero_history_generative_on_motion`: same model, with endpoint history forced to zero on motion response.",
        "- `known_history_correlated_generative`: linear-Gaussian model with a joint feature-history prior, using `p(z | tau_true)`.",
        "- `joint_history_correlated_generative`: same correlated-prior model with latent history marginalized.",
        "- `zero_history_correlated_generative_on_motion`: same correlated-prior model with history forced to zero.",
        "",
        "At the 1x scale:",
        "",
        "```csv",
        *scale1_lines,
        "```",
        "",
        "Paired observer feature-cosine contrasts:",
        "",
        "```csv",
        *contrast_lines,
        "```",
        "",
        "Interpretation boundary: `joint_history_response_only` is a discriminative",
        "response-only hidden/marginal history readout, not yet an explicit posterior",
        "integral over a drift-bridge prior. The assay is nevertheless terminal-state",
        "and endpoint-aligned: same final image, different preceding 32-frame histories,",
        "same feature target and scoring contract.",
        "",
        "Outputs:",
        "",
        "- `endpoint_history_feature_readout_trials.csv`",
        "- `endpoint_history_feature_readout_summary.csv`",
        "- `endpoint_history_feature_readout_contrasts.csv`",
        "- `endpoint_history_feature_readout_models.csv`",
        "- `endpoint_history_trace_metrics.csv`",
        "- `endpoint_history_feature_readout.png`",
        "- `endpoint_history_feature_readout_manifest.json`",
    ]
    (out_dir / "endpoint_history_feature_readout_README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--window-manifest", type=Path, default=None)
    parser.add_argument(
        "--load-endpoint-dataset-dir",
        type=Path,
        default=None,
        help="Reuse endpoint_history_dataset_* artifacts from a previous run and skip twin rendering.",
    )
    parser.add_argument("--feature-npz", type=Path, default=FEATURE_NPZ)
    parser.add_argument("--feature-weights-npz", type=Path, default=None)
    parser.add_argument("--latent", default=PRIMARY_LATENT)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--feature-space-modes", default=",".join(DEFAULT_FEATURE_SPACE_MODES))
    parser.add_argument("--plot-feature-space-mode", default="fold_zscore_whitened_pca")
    parser.add_argument("--motion-families", default=",".join(DEFAULT_MOTION_FAMILIES))
    parser.add_argument("--primary-motion-family", default="empirical")
    parser.add_argument(
        "--joint-prior-families",
        default="",
        help=(
            "Endpoint-history conditions used to train the response-only joint/hidden readout. "
            "Defaults to --motion-families."
        ),
    )
    parser.add_argument("--scales", default="0.5,1.0,2.0")
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--min-patch-image-margin-px", type=float, default=None)
    parser.add_argument("--n-timepoints", type=int, default=32)
    parser.add_argument("--terminal-frames", type=int, default=1)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument(
        "--history-coordinate-mode",
        choices=HISTORY_COORDINATE_MODES,
        default="raw",
        help="Coordinate system for trajectory-history terms in known/generative observers.",
    )
    parser.add_argument(
        "--history-dim",
        type=int,
        default=8,
        help="Number of train-fold PCA history coordinates when --history-coordinate-mode=fold_pca.",
    )
    parser.add_argument("--reliable-image-coherence-min", type=float, default=0.20)
    parser.add_argument("--reliable-drift-anisotropy-min", type=float, default=0.20)
    parser.add_argument("--min-duration-s", type=float, default=0.10)
    parser.add_argument("--max-rms-deg", type=float, default=0.12)
    parser.add_argument("--max-trace-source-rms-deg", type=float, default=None)
    parser.add_argument("--max-trace-source-radius-deg", type=float, default=None)
    parser.add_argument("--max-trace-source-path-length-deg", type=float, default=None)
    parser.add_argument("--max-rendered-trace-path-length-deg", type=float, default=None)
    parser.add_argument("--max-source-trace-path-length-deg", type=float, default=None)
    parser.add_argument("--max-trace-source-speed-p95-deg-s", type=float, default=None)
    parser.add_argument("--max-trace-source-microsaccade-events", type=int, default=None)
    parser.add_argument("--microsaccade-speed-threshold-dps", type=float, default=None)
    parser.add_argument("--microsaccade-threshold-z", type=float, default=6.0)
    parser.add_argument("--microsaccade-pad-frames", type=int, default=1)
    parser.add_argument("--axis-source-column", default="image_edge_axis_deg")
    parser.add_argument("--axis-template-mode", default="same_dominant_projection")
    parser.add_argument("--axis-match-policy", choices=("strict", "allow_invalid"), default="strict")
    parser.add_argument(
        "--response-population-mode",
        choices=RESPONSE_POPULATION_MODES,
        default="rr100",
        help="Use rr100 by default to keep the terminal readout small; rendering still computes the canonical twin response.",
    )
    parser.add_argument("--rr100-version", default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument(
        "--response-basis-mode",
        choices=RESPONSE_BASIS_MODES,
        default="full_units",
        help="Use full_units for the geometry-uncommitted endpoint-history readout.",
    )
    parser.add_argument("--compact-basis-path", type=Path, default=COMPACT_BASIS)
    parser.add_argument("--basis-key", default="basis")
    parser.add_argument("--basis-max-dim", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--noise-floor", type=float, default=1e-8)
    parser.add_argument("--source-weighting", choices=SOURCE_WEIGHTING_MODES, default="source_balanced")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260706)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--twin-batch-size", type=int, default=24)
    parser.add_argument("--twin-trace-batch-size", type=int, default=4)
    parser.add_argument("--cuda-empty-cache-every-batch", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260706)
    return parser


def build(args: argparse.Namespace) -> Path:
    if int(args.n_timepoints) != 32:
        print(
            f"[endpoint-history] warning: n_timepoints={int(args.n_timepoints)}; "
            "the intended assay uses the model's 32-frame history.",
            flush=True,
        )
    if int(args.terminal_frames) != 1:
        print(
            f"[endpoint-history] warning: terminal_frames={int(args.terminal_frames)}; "
            "primary interpretation should use terminal_frames=1.",
            flush=True,
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_table, feature_meta = _load_feature_table(Path(args.feature_npz), latent=str(args.latent))
    feature_weights, feature_weight_meta = _load_feature_weights(
        Path(args.feature_weights_npz) if args.feature_weights_npz is not None else None,
        latent=str(args.latent),
        raw_feature_dim=int(feature_table.features.shape[1]),
    )
    feature_space_modes = _parse_str_list(args.feature_space_modes)
    if not feature_space_modes:
        raise ValueError("--feature-space-modes must list at least one mode")
    global_modes = [
        mode
        for mode in feature_space_modes
        if _feature_space_config(mode)["fit_scope"] == "global"
    ]
    if global_modes:
        raise ValueError(
            "Endpoint-history readout keeps feature transforms source-disjoint; "
            f"global mode(s) are not allowed here: {global_modes}"
        )

    primary_family = str(args.primary_motion_family)
    motion_families = _parse_str_list(args.motion_families)
    if primary_family not in motion_families:
        raise ValueError(
            f"--primary-motion-family={primary_family!r} must be included in --motion-families={args.motion_families!r}"
        )
    primary_condition = _condition_for_family(primary_family).slug
    joint_families = _parse_str_list(args.joint_prior_families) or motion_families
    invalid_joint = sorted(set(joint_families).difference(motion_families))
    if invalid_joint:
        raise ValueError(
            f"--joint-prior-families must be a subset of --motion-families; invalid={invalid_joint}"
        )
    joint_conditions = [_condition_for_family(family).slug for family in joint_families]
    cached_assay: dict[str, Any] = {}
    loaded_dataset_dir = Path(args.load_endpoint_dataset_dir) if args.load_endpoint_dataset_dir is not None else None
    if loaded_dataset_dir is not None:
        dataset = _load_endpoint_dataset(loaded_dataset_dir)
        population_meta, basis_meta, cached_assay = _loaded_endpoint_dataset_meta(loaded_dataset_dir, dataset)
    else:
        scorer = CanonicalTwinScorer(
            device=str(args.device),
            batch_size=int(args.twin_batch_size),
            empty_cache_every_batch=bool(args.cuda_empty_cache_every_batch),
        )
        population, population_meta = _response_population(
            mode=str(args.response_population_mode),
            n_units=int(scorer.n_units),
            rr100_version=str(args.rr100_version),
        )
        if str(args.response_basis_mode) == "compact" and str(args.response_population_mode) != "full756":
            raise ValueError(
                "The compact basis is defined in canonical full-756 space. "
                "Use --response-population-mode full756 with --response-basis-mode compact."
            )
        basis, basis_meta = _response_basis(
            mode=str(args.response_basis_mode),
            path=Path(args.compact_basis_path),
            n_units=int(population.n_units),
            basis_key=str(args.basis_key),
            max_dim=int(args.basis_max_dim),
        )
        dataset = _build_endpoint_dataset(
            args=args,
            scorer=scorer,
            population=population,
            basis=basis,
            feature_sources=set(int(value) for value in feature_table.source_rows.tolist()),
        )
    dataset_scales = _validate_endpoint_dataset_contract(
        dataset=dataset,
        args=args,
        primary_condition=primary_condition,
        joint_conditions=joint_conditions,
        cached_assay=cached_assay,
    )
    available_feature_sources = set(int(value) for value in feature_table.source_rows.tolist())
    missing_feature_sources = sorted(
        set(int(value) for value in dataset.rows["true_source_row"].tolist()).difference(available_feature_sources)
    )
    if missing_feature_sources:
        preview = missing_feature_sources[:10]
        raise ValueError(
            f"Endpoint dataset contains source rows not present in the feature table: {preview}"
            + ("..." if len(missing_feature_sources) > len(preview) else "")
        )
    dataset_cache_paths = _save_endpoint_dataset(dataset, out_dir)
    trials, models = _run_crossfit(
        dataset=dataset,
        feature_table=feature_table,
        feature_weights=feature_weights,
        feature_space_modes=feature_space_modes,
        feature_dim=int(args.feature_dim),
        n_folds=int(args.n_folds),
        fold_seed=int(args.fold_seed),
        ridge=float(args.ridge),
        noise_floor=float(args.noise_floor),
        source_weighting=str(args.source_weighting),
        primary_condition=primary_condition,
        joint_conditions=joint_conditions,
        history_coordinate_mode=str(args.history_coordinate_mode),
        history_dim=int(args.history_dim),
    )
    summary = _summarize(trials)
    contrasts = _contrasts(trials, n_boot=int(args.n_bootstrap), seed=int(args.fold_seed) + 17)

    trials_path = out_dir / "endpoint_history_feature_readout_trials.csv"
    summary_path = out_dir / "endpoint_history_feature_readout_summary.csv"
    contrasts_path = out_dir / "endpoint_history_feature_readout_contrasts.csv"
    models_path = out_dir / "endpoint_history_feature_readout_models.csv"
    trace_path = dataset_cache_paths["trace_metrics"]
    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)
    contrasts.to_csv(contrasts_path, index=False)
    models.to_csv(models_path, index=False)
    png, pdf, plotted_mode = _plot(
        summary,
        contrasts,
        out_dir,
        latent=str(args.latent),
        feature_space_mode=str(args.plot_feature_space_mode),
    )

    manifest = {
        "analysis": "endpoint_history_feature_readout",
        "feature": {
            **feature_meta,
            "feature_dim_requested": int(args.feature_dim),
            "feature_space_modes_requested": feature_space_modes,
            "feature_space_modes_canonical": sorted(set(models["feature_space_mode"].astype(str).tolist())),
            "feature_weights": feature_weight_meta,
        },
        "assay": {
            "n_timepoints": int(args.n_timepoints),
            "terminal_frames": int(args.terminal_frames),
            "endpoint_alignment": "tau_endpoint[t] = tau[t] - tau[-1]",
            "readout_contract": "decode only the terminal response frame/window",
            "target_contract": "endpoint/crop-centered image feature embedding",
            "bin_seconds": float(args.bin_seconds),
            "history_coordinate_mode": str(args.history_coordinate_mode),
            "history_dim": int(args.history_dim),
            "motion_families": motion_families,
            "primary_motion_family": primary_family,
            "primary_condition": primary_condition,
            "joint_prior_families": joint_families,
            "joint_conditions": joint_conditions,
            "scales": dataset_scales,
        },
        "endpoint_dataset": {
            "loaded_from": str(loaded_dataset_dir) if loaded_dataset_dir is not None else None,
            "cached_assay": cached_assay,
            "cache_files": dataset_cache_paths,
        },
        "conditions": [
            {
                "condition": condition.slug,
                "family": condition.family,
                "label": condition.label,
                "interpretation": condition.interpretation,
            }
            for condition in dataset.conditions
        ],
        "population": population_meta,
        "basis": basis_meta,
        "source_weighting": str(args.source_weighting),
        "ridge": float(args.ridge),
        "noise_floor": float(args.noise_floor),
        "n_folds": int(args.n_folds),
        "fold_seed": int(args.fold_seed),
        "n_bootstrap": int(args.n_bootstrap),
        "plot_feature_space_mode": plotted_mode,
        "observer_modes": sorted(set(models["observer_mode"].astype(str).tolist())),
        "n_samples": int(dataset.rows.shape[0]),
        "n_source_rows": int(dataset.rows["true_source_row"].nunique()),
        "outputs": {
            "trials": trials_path,
            "summary": summary_path,
            "contrasts": contrasts_path,
            "models": models_path,
            "trace_metrics": trace_path,
            "dataset_rows": dataset_cache_paths["dataset_rows"],
            "dataset_arrays": dataset_cache_paths["dataset_arrays"],
            "dataset_conditions": dataset_cache_paths["dataset_conditions"],
            "png": png,
            "pdf": pdf,
        },
    }
    _write_json(out_dir / "endpoint_history_feature_readout_manifest.json", manifest)
    _write_readme(out_dir=out_dir, summary=summary, contrasts=contrasts, manifest=manifest)
    return out_dir


def main() -> None:
    args = build_parser().parse_args()
    out_dir = build(args)
    print(f"[endpoint-history] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
