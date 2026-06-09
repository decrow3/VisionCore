"""Production twininfo analysis pipeline.

The pipeline mirrors the analysis outline in the README and intentionally keeps
slow work behind explicit flags:

1. Eye movement selector and visualization.
2. Image selector, pyramid QC, and crop-hotspot selection.
3. Retinal stimulus movies, if requested.
4. Activation-map movies, if requested.
5. Individual cumulative information traces.
6. Spatial-frequency cumulative information traces.
7. Final information-gain summary figures.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import hashlib
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .activation_movies import make_activation_movies
from .common import DEFAULT_CCMAX_THRESHOLD, DT, N_LAGS, OUTPUT_DIR, load_digital_twin, write_json
from .image_selection import PYRAMID_HEIGHT, PYRAMID_ORDER, SF_BANDS_4, crop_rows, select_image_crops
from .io_utils import ensure_run_dirs, write_csv
from .lagcube_information import (
    approximate_unit_fisher_scores,
    block_current_samples,
    block_endpoint_lag_cubes,
    cross_shift_grid,
    cumulative_pattern_fisher,
    cumulative_spatial_ssi,
    final_metric_row,
    finite_difference_shift_set,
    run_shifted_lag_cube_rate_maps,
    square_shift_grid,
    unique_shifts,
)
from .retinal_examples import TraceExample, model_lag_cubes_from_image_trace, pyramid_local_image_controls
from .retinal_movies import CONDITION_LABELS, make_representative_stimulus_movies
from .population import (
    GRID_POSITION_MODES,
    PERFORMANCE_METRICS,
    POPULATION_SELECTION_MODES,
    build_analysis_population,
)
from .trace_selection import run_trace_selection_step


TRAJECTORY_CONTROL_CONDITIONS = ("random_amp", "random_amp_cloud_matched", "random_cov", "trajectory_order_shuffle")
TRAJECTORY_COMPARISON_CONDITIONS = ("real", "stabilized") + TRAJECTORY_CONTROL_CONDITIONS
PHASE_CONDITIONS = ("real", "stabilized", "pyramid_phase_scrambled")
SF_CONDITIONS = ("real",) + SF_BANDS_4
MAIN_CONDITIONS = TRAJECTORY_COMPARISON_CONDITIONS + ("pyramid_phase_scrambled",) + SF_BANDS_4
STABILIZED_VISUAL_CONTROL_CONDITIONS = (
    "stabilized_pyramid_phase_scrambled",
    *(f"stabilized_{condition}" for condition in SF_BANDS_4),
)
AUGMENTED_VISUAL_COMPARISON_CONDITIONS = (
    "real",
    "stabilized",
    "sf_low",
    "stabilized_sf_low",
    "sf_high",
    "stabilized_sf_high",
    "pyramid_phase_scrambled",
    "stabilized_pyramid_phase_scrambled",
)
ALL_CONDITIONS = MAIN_CONDITIONS + STABILIZED_VISUAL_CONTROL_CONDITIONS

CONDITION_COLORS = {
    "real": "#1f77b4",
    "stabilized": "#2ca02c",
    "random_amp": "#7f7f7f",
    "random_amp_cloud_matched": "#525252",
    "random_cov": "#bcbd22",
    "trajectory_order_shuffle": "#17becf",
    "phase_order_shuffle": "#17becf",
    "pyramid_phase_scrambled": "#d62728",
    "sf_low": "#9467bd",
    "sf_mid_low": "#8c564b",
    "sf_mid_high": "#ff7f0e",
    "sf_high": "#4c78a8",
    "stabilized_pyramid_phase_scrambled": "#f2a4a4",
    "stabilized_sf_low": "#c7a9dd",
    "stabilized_sf_mid_low": "#c49c94",
    "stabilized_sf_mid_high": "#ffbb78",
    "stabilized_sf_high": "#9ecae1",
}


@dataclass(frozen=True)
class PipelineConfig:
    run_name: str | None = None
    seed: int = 0
    image_indices: tuple[int, ...] | None = None
    n_crops_per_image: int = 3
    n_examples_per_kind: int = 10
    selected_trace_example_ids: tuple[str, ...] = ()
    t_max: int = 128
    stride: int = 8
    population_size: int = 100
    population_selection: str = "top_performance"
    performance_metric: str = "ccnorm"
    min_performance_score: float | None = None
    population_grid_position_mode: str = "full_grid"
    population_grid_stride: int = 1
    deduplicate_units: bool = True
    dedupe_correlation_threshold: float = 0.95
    dedupe_candidate_multiplier: float = 5.0
    dedupe_battery_frames: int = 96
    ccmax_threshold: float = DEFAULT_CCMAX_THRESHOLD
    batch_size: int = 64
    fisher_step_arcmin: float = 0.5
    shift_grid_max_arcmin: float = 1.0
    shift_grid_step_arcmin: float = 1.0
    shift_grid_mode: str = "square"
    make_stimulus_movies: bool = False
    make_activation_movies: bool = False
    movie_examples_per_kind: int = 1
    movie_fps: int = 30
    conditions: tuple[str, ...] = MAIN_CONDITIONS
    augment_existing: bool = False
    recompute: bool = False


def _jsonable_config(config: PipelineConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["image_indices"] = None if config.image_indices is None else list(config.image_indices)
    payload["selected_trace_example_ids"] = list(config.selected_trace_example_ids)
    payload["analysis_version"] = "production_pyramid_spatial_ssi_v2"
    payload["conditions"] = list(config.conditions)
    payload["default_conditions"] = list(MAIN_CONDITIONS)
    payload["stabilized_visual_control_conditions"] = list(STABILIZED_VISUAL_CONTROL_CONDITIONS)
    payload["pyramid_height"] = PYRAMID_HEIGHT
    payload["pyramid_order"] = PYRAMID_ORDER
    return payload


def _folder_safe(value: str) -> str:
    """Return a short filesystem-safe label that remains readable."""
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    safe = "-".join(part for part in safe.split("-") if part)
    return safe or "analysis"


def run_slug(config: PipelineConfig) -> str:
    """Human-readable analysis folder name.

    Output-generation flags such as MP4 generation are intentionally excluded:
    they change which artifacts are written, but not the analysis definition.
    Use ``--run-name`` when a fixed folder name is preferred.
    """
    if config.run_name:
        return _folder_safe(config.run_name)

    if config.image_indices is None:
        image_label = "all-images"
    else:
        image_label = "images-" + "-".join(str(idx) for idx in config.image_indices)

    trace_label = f"{config.n_examples_per_kind}fix-{config.n_examples_per_kind}ms"
    crop_label = f"{config.n_crops_per_image}crop" if config.n_crops_per_image == 1 else f"{config.n_crops_per_image}crops"
    unit_label = f"{config.population_size}units-per-pixel"
    selection_label = config.performance_metric if config.population_selection == "top_performance" else config.population_selection
    if config.min_performance_score is not None:
        threshold = f"{float(config.min_performance_score):.3g}".replace(".", "p")
        selection_label = f"{selection_label}-ge-{threshold}"
    grid_label = f"{config.shift_grid_mode}-grid"
    pop_grid_label = (
        f"{config.population_grid_position_mode}-s{config.population_grid_stride}"
        if config.population_grid_position_mode == "full_grid"
        else config.population_grid_position_mode
    )
    t_label = f"{config.t_max}frames"
    seed_label = f"seed-{config.seed}"
    return _folder_safe(
        f"{image_label}_{crop_label}_{trace_label}_{unit_label}_{selection_label}_{pop_grid_label}_{grid_label}_{t_label}_{seed_label}"
    )


def _example_seed(seed: int, example_id: str, image_index: int, crop_rank: int) -> int:
    payload = f"{seed}:{example_id}:{image_index}:{crop_rank}".encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest()[:8], 16)


def _filter_trace_examples(config: PipelineConfig, examples: list[TraceExample]) -> list[TraceExample]:
    if not config.selected_trace_example_ids:
        return examples
    by_id = {example.example_id: example for example in examples}
    missing = [example_id for example_id in config.selected_trace_example_ids if example_id not in by_id]
    if missing:
        available = ", ".join(sorted(by_id))
        raise ValueError(f"Missing selected trace IDs {missing}. Available: {available}")
    return [by_id[example_id] for example_id in config.selected_trace_example_ids]


def _condition_seed(seed: int, condition: str) -> int:
    payload = f"{seed}:{condition}".encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest()[:8], 16)


def _canonical_condition_name(condition: str) -> str:
    if condition == "phase_order_shuffle":
        return "trajectory_order_shuffle"
    return condition


def _validate_conditions(conditions: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize requested conditions and reject unknown names early."""
    seen: set[str] = set()
    out: list[str] = []
    valid = set(ALL_CONDITIONS)
    for condition in conditions:
        canonical = _canonical_condition_name(str(condition))
        if canonical not in valid:
            available = ", ".join(ALL_CONDITIONS)
            raise ValueError(f"Unsupported condition {condition!r}. Available: {available}")
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    if not out:
        raise ValueError("At least one condition is required.")
    return tuple(out)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row["example_id"]),
        int(row["image_index"]),
        int(row["crop_rank"]),
        _canonical_condition_name(str(row["condition"])),
    )


def _load_existing_series(path: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]] | None:
    if not path.exists():
        return None
    with np.load(path) as npz:
        y_key = "cumulative_spatial_ssi_bits_per_spike"
        n = int(npz[y_key].shape[0])
        records = [
            {
                "example_id": str(npz["record_example_id"][i]),
                "kind": str(npz["record_kind"][i]),
                "condition": _canonical_condition_name(str(npz["record_condition"][i])),
                "image_index": int(npz["record_image_index"][i]),
                "crop_rank": int(npz["record_crop_rank"][i]),
            }
            for i in range(n)
        ]
        arrays = {
            key: np.asarray(npz[key])
            for key in npz.files
            if not key.startswith("record_")
        }
    return records, arrays


def _config_with_existing_run_definition(config: PipelineConfig, run_config_path: Path) -> PipelineConfig:
    """For augmentation, reuse the original scientific run definition."""
    if not (config.augment_existing and run_config_path.exists()):
        return config
    with open(run_config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    keep_conditions = config.conditions
    kwargs: dict[str, Any] = {}
    tuple_fields = {"image_indices", "selected_trace_example_ids", "conditions"}
    skip = {"run_name", "conditions", "augment_existing", "recompute"}
    for field in fields(PipelineConfig):
        name = field.name
        if name in skip or name not in payload:
            continue
        value = payload[name]
        if name in tuple_fields and value is not None:
            value = tuple(value)
        kwargs[name] = value
    return PipelineConfig(
        **kwargs,
        run_name=config.run_name,
        conditions=keep_conditions,
        augment_existing=config.augment_existing,
        recompute=config.recompute,
    )


def _requested_conditions_complete(summary_csv: Path, requested_conditions: tuple[str, ...]) -> bool:
    rows = _read_csv_rows(summary_csv)
    if not rows:
        return False
    pair_keys = {
        (str(row["example_id"]), int(row["image_index"]), int(row["crop_rank"]))
        for row in rows
    }
    existing = {_row_key(row) for row in rows}
    for example_id, image_index, crop_rank in pair_keys:
        for condition in requested_conditions:
            if (example_id, image_index, crop_rank, condition) not in existing:
                return False
    return True


def _trace_stats_for_qc(trace: np.ndarray) -> dict[str, float]:
    tr = np.asarray(trace, dtype=np.float64)
    if tr.ndim != 2 or tr.shape[1] != 2 or tr.shape[0] == 0:
        return {
            "mean_x_deg": float("nan"),
            "mean_y_deg": float("nan"),
            "rms_displacement_deg": float("nan"),
            "path_length_deg": float("nan"),
            "step_rms_deg": float("nan"),
            "step_mean_deg": float("nan"),
            "step_p95_deg": float("nan"),
            "step_cov_xx": float("nan"),
            "step_cov_xy": float("nan"),
            "step_cov_yy": float("nan"),
            "pos_cov_xx": float("nan"),
            "pos_cov_xy": float("nan"),
            "pos_cov_yy": float("nan"),
        }
    centered = tr - np.mean(tr, axis=0, keepdims=True)
    steps = np.diff(tr, axis=0)
    step_amp = np.linalg.norm(steps, axis=1) if steps.size else np.zeros((0,), dtype=np.float64)
    step_cov = np.cov(steps.T) if steps.shape[0] > 1 else np.zeros((2, 2), dtype=np.float64)
    pos_cov = np.cov(centered.T) if tr.shape[0] > 1 else np.zeros((2, 2), dtype=np.float64)
    return {
        "mean_x_deg": float(np.mean(tr[:, 0])),
        "mean_y_deg": float(np.mean(tr[:, 1])),
        "rms_displacement_deg": float(np.sqrt(np.mean(np.sum(centered * centered, axis=1)))),
        "path_length_deg": float(np.sum(step_amp)) if step_amp.size else 0.0,
        "step_rms_deg": float(np.sqrt(np.mean(step_amp * step_amp))) if step_amp.size else 0.0,
        "step_mean_deg": float(np.mean(step_amp)) if step_amp.size else 0.0,
        "step_p95_deg": float(np.percentile(step_amp, 95.0)) if step_amp.size else 0.0,
        "step_cov_xx": float(step_cov[0, 0]) if step_cov.shape == (2, 2) else float("nan"),
        "step_cov_xy": float(step_cov[0, 1]) if step_cov.shape == (2, 2) else float("nan"),
        "step_cov_yy": float(step_cov[1, 1]) if step_cov.shape == (2, 2) else float("nan"),
        "pos_cov_xx": float(pos_cov[0, 0]) if pos_cov.shape == (2, 2) else float("nan"),
        "pos_cov_xy": float(pos_cov[0, 1]) if pos_cov.shape == (2, 2) else float("nan"),
        "pos_cov_yy": float(pos_cov[1, 1]) if pos_cov.shape == (2, 2) else float("nan"),
    }


def _cov_matrix_from_stats(stats: dict[str, float], prefix: str) -> np.ndarray:
    return np.asarray(
        [
            [stats[f"{prefix}_cov_xx"], stats[f"{prefix}_cov_xy"]],
            [stats[f"{prefix}_cov_xy"], stats[f"{prefix}_cov_yy"]],
        ],
        dtype=np.float64,
    )


def _random_amp_cloud_matched_trace(
    trace: np.ndarray,
    *,
    t_max: int,
    rng: np.random.Generator,
    max_attempts: int = 512,
    tol_path: float = 0.25,
    tol_step_rms: float = 0.30,
    tol_step_p95: float = 0.35,
    tol_rms: float = 0.20,
    tol_cov: float = 0.30,
    tol_eig: float = 0.25,
) -> tuple[np.ndarray, str]:
    """Random-direction path closely matching step scale and occupancy."""
    tr = np.asarray(trace[:t_max], dtype=np.float32)
    center = np.mean(tr, axis=0, keepdims=True).astype(np.float32)
    steps = np.diff(tr, axis=0)
    amp = np.linalg.norm(steps, axis=1).astype(np.float32)
    if amp.size == 0 or not np.any(amp > 0):
        return np.repeat(center, tr.shape[0], axis=0).astype(np.float32), "empty_trace_stabilized_fallback"

    real_stats = _trace_stats_for_qc(tr)
    real_cov = _cov_matrix_from_stats(real_stats, "pos")
    real_cov_norm = max(float(np.linalg.norm(real_cov)), 1e-12)
    real_eig = np.sort(np.linalg.eigvalsh(real_cov))
    real_eig_norm = max(float(np.linalg.norm(real_eig)), 1e-12)
    real_rms = max(float(real_stats["rms_displacement_deg"]), 1e-12)
    real_path = max(float(real_stats["path_length_deg"]), 1e-12)
    real_step_rms = max(float(real_stats["step_rms_deg"]), 1e-12)
    real_step_p95 = max(float(real_stats["step_p95_deg"]), 1e-12)

    def cov_sqrt(mat: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(np.asarray(mat, dtype=np.float64) + np.eye(2) * 1e-12)
        vals = np.maximum(vals, 1e-12)
        return (vecs * np.sqrt(vals)) @ vecs.T

    def cov_invsqrt(mat: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(np.asarray(mat, dtype=np.float64) + np.eye(2) * 1e-12)
        vals = np.maximum(vals, 1e-12)
        return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T

    target_sqrt = cov_sqrt(real_cov)
    alphas = np.linspace(0.0, 1.0, 11, dtype=np.float64)

    def partial_cloud_match_candidates(ctrl_centered: np.ndarray) -> list[np.ndarray]:
        projected = np.asarray(ctrl_centered, dtype=np.float64)
        cand_cov = np.cov(projected.T) if projected.shape[0] > 1 else np.eye(2) * 1e-12
        full_transform = target_sqrt @ cov_invsqrt(cand_cov)
        out = []
        for alpha in alphas:
            transform = (1.0 - float(alpha)) * np.eye(2) + float(alpha) * full_transform
            candidate = projected @ transform.T
            candidate -= np.mean(candidate, axis=0, keepdims=True)
            out.append(candidate.astype(np.float32))
        return out

    best_trace: np.ndarray | None = None
    best_score = float("inf")
    best_desc = ""
    for attempt in range(int(max_attempts)):
        directions = rng.uniform(0.0, 2.0 * np.pi, size=amp.shape[0])
        ctrl_steps = np.column_stack([amp * np.cos(directions), amp * np.sin(directions)]).astype(np.float32)
        ctrl = np.vstack([np.zeros((1, 2), dtype=np.float32), np.cumsum(ctrl_steps, axis=0)])
        ctrl -= np.mean(ctrl, axis=0, keepdims=True)
        for alpha, ctrl_candidate in zip(alphas, partial_cloud_match_candidates(ctrl), strict=True):
            candidate = (ctrl_candidate + center).astype(np.float32)
            stats = _trace_stats_for_qc(candidate)
            cov = _cov_matrix_from_stats(stats, "pos")
            eig = np.sort(np.linalg.eigvalsh(cov))
            path_rel = abs(float(stats["path_length_deg"]) - real_path) / real_path
            step_rms_rel = abs(float(stats["step_rms_deg"]) - real_step_rms) / real_step_rms
            step_p95_rel = abs(float(stats["step_p95_deg"]) - real_step_p95) / real_step_p95
            rms_rel = abs(float(stats["rms_displacement_deg"]) - real_rms) / real_rms
            cov_rel = float(np.linalg.norm(cov - real_cov) / real_cov_norm)
            eig_rel = float(np.linalg.norm(eig - real_eig) / real_eig_norm)
            score = (
                path_rel / max(tol_path, 1e-12)
                + step_rms_rel / max(tol_step_rms, 1e-12)
                + step_p95_rel / max(tol_step_p95, 1e-12)
                + rms_rel / max(tol_rms, 1e-12)
                + cov_rel / max(tol_cov, 1e-12)
                + eig_rel / max(tol_eig, 1e-12)
            )
            if score < best_score:
                best_score = score
                best_trace = candidate
                best_desc = (
                    "step_amplitude_and_cloud_matched_random_directions_best_effort"
                    f"_attempts={attempt + 1}_alpha={float(alpha):.2f}_path_rel={path_rel:.4g}"
                    f"_step_rms_rel={step_rms_rel:.4g}_step_p95_rel={step_p95_rel:.4g}"
                    f"_rms_rel={rms_rel:.4g}_cov_rel={cov_rel:.4g}_eig_rel={eig_rel:.4g}"
                )
            if (
                path_rel <= tol_path
                and step_rms_rel <= tol_step_rms
                and step_p95_rel <= tol_step_p95
                and rms_rel <= tol_rms
                and cov_rel <= tol_cov
                and eig_rel <= tol_eig
            ):
                return candidate, (
                    "step_amplitude_and_cloud_matched_random_directions"
                    f"_attempts={attempt + 1}_alpha={float(alpha):.2f}_path_rel={path_rel:.4g}"
                    f"_step_rms_rel={step_rms_rel:.4g}_step_p95_rel={step_p95_rel:.4g}"
                    f"_rms_rel={rms_rel:.4g}_cov_rel={cov_rel:.4g}_eig_rel={eig_rel:.4g}"
                )
    if best_trace is None:
        raise RuntimeError("Could not generate random_amp_cloud_matched candidate.")
    fallback = (-1.0 * (tr[::-1] - center) + center).astype(np.float32)
    fallback_stats = _trace_stats_for_qc(fallback)
    fallback_cov = _cov_matrix_from_stats(fallback_stats, "pos")
    fallback_eig = np.sort(np.linalg.eigvalsh(fallback_cov))
    fallback_desc = (
        "inverted_time_reversed_exact_scale_cloud_fallback"
        f"_random_best=({best_desc})"
        f"_fallback_path_rel={abs(float(fallback_stats['path_length_deg']) - real_path) / real_path:.4g}"
        f"_fallback_step_rms_rel={abs(float(fallback_stats['step_rms_deg']) - real_step_rms) / real_step_rms:.4g}"
        f"_fallback_step_p95_rel={abs(float(fallback_stats['step_p95_deg']) - real_step_p95) / real_step_p95:.4g}"
        f"_fallback_rms_rel={abs(float(fallback_stats['rms_displacement_deg']) - real_rms) / real_rms:.4g}"
        f"_fallback_cov_rel={float(np.linalg.norm(fallback_cov - real_cov) / real_cov_norm):.4g}"
        f"_fallback_eig_rel={float(np.linalg.norm(fallback_eig - real_eig) / real_eig_norm):.4g}"
    )
    return fallback, fallback_desc


def _trajectory_for_condition(
    trace: np.ndarray,
    condition: str,
    *,
    t_max: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Return the eye trace used by a trajectory condition.

    Random controls are centered on the measured fixation mean so they preserve
    the local image neighborhood while breaking specific temporal statistics.
    """
    tr = np.asarray(trace[:t_max], dtype=np.float32)
    center = np.mean(tr, axis=0, keepdims=True).astype(np.float32)
    if condition == "real":
        return tr.copy(), "measured_trace"
    if condition == "stabilized":
        return np.repeat(center, tr.shape[0], axis=0).astype(np.float32), "trial_mean_stabilized"

    condition_key = _canonical_condition_name(condition)
    rng = np.random.default_rng(_condition_seed(seed, condition_key))
    if condition == "random_amp":
        steps = np.diff(tr, axis=0)
        amp = np.linalg.norm(steps, axis=1)
        angle = rng.uniform(0.0, 2.0 * np.pi, size=amp.shape[0])
        ctrl_steps = np.column_stack([amp * np.cos(angle), amp * np.sin(angle)]).astype(np.float32)
        ctrl = np.vstack([np.zeros((1, 2), dtype=np.float32), np.cumsum(ctrl_steps, axis=0)])
        ctrl -= np.mean(ctrl, axis=0, keepdims=True)
        return (ctrl + center).astype(np.float32), "step_amplitude_matched_random_directions"

    if condition == "random_amp_cloud_matched":
        return _random_amp_cloud_matched_trace(tr, t_max=t_max, rng=rng)

    if condition == "random_cov":
        steps = np.diff(tr, axis=0).astype(np.float64)
        if steps.shape[0] == 0:
            return np.repeat(center, tr.shape[0], axis=0).astype(np.float32), "empty_trace_stabilized_fallback"
        mu = np.mean(steps, axis=0)
        cov = np.cov(steps.T) if steps.shape[0] > 1 else np.eye(2) * 1e-10
        cov = np.asarray(cov, dtype=np.float64) + np.eye(2) * 1e-10
        ctrl_steps = rng.multivariate_normal(mu, cov, size=steps.shape[0]).astype(np.float32)
        ctrl = np.vstack([np.zeros((1, 2), dtype=np.float32), np.cumsum(ctrl_steps, axis=0)])
        ctrl -= np.mean(ctrl, axis=0, keepdims=True)
        return (ctrl + center).astype(np.float32), "step_covariance_matched_gaussian"

    if condition in {"trajectory_order_shuffle", "phase_order_shuffle"}:
        order = rng.permutation(tr.shape[0])
        centered = tr - center
        return (centered[order] + center).astype(np.float32), "same_positions_time_order_shuffled"

    raise ValueError(f"Unsupported trajectory condition: {condition}")


def _trajectory_qc_row(
    *,
    example: TraceExample,
    image_index: int,
    crop_rank: int,
    condition: str,
    control_trace: np.ndarray,
    control_description: str,
) -> dict[str, Any]:
    real_stats = _trace_stats_for_qc(example.trace)
    ctrl_stats = _trace_stats_for_qc(control_trace)
    real_step_cov = _cov_matrix_from_stats(real_stats, "step")
    ctrl_step_cov = _cov_matrix_from_stats(ctrl_stats, "step")
    real_pos_cov = _cov_matrix_from_stats(real_stats, "pos")
    ctrl_pos_cov = _cov_matrix_from_stats(ctrl_stats, "pos")
    cov_denom = max(float(np.linalg.norm(real_step_cov)), 1e-12)
    pos_cov_denom = max(float(np.linalg.norm(real_pos_cov)), 1e-12)
    return {
        "example_id": example.example_id,
        "kind": example.kind,
        "image_index": int(image_index),
        "crop_rank": int(crop_rank),
        "condition": condition,
        "control_description": control_description,
        "real_path_length_deg": real_stats["path_length_deg"],
        "control_path_length_deg": ctrl_stats["path_length_deg"],
        "path_length_relative_error": abs(ctrl_stats["path_length_deg"] - real_stats["path_length_deg"])
        / max(abs(real_stats["path_length_deg"]), 1e-12),
        "real_rms_displacement_deg": real_stats["rms_displacement_deg"],
        "control_rms_displacement_deg": ctrl_stats["rms_displacement_deg"],
        "rms_relative_error": abs(ctrl_stats["rms_displacement_deg"] - real_stats["rms_displacement_deg"])
        / max(abs(real_stats["rms_displacement_deg"]), 1e-12),
        "real_step_rms_deg": real_stats["step_rms_deg"],
        "control_step_rms_deg": ctrl_stats["step_rms_deg"],
        "step_rms_relative_error": abs(ctrl_stats["step_rms_deg"] - real_stats["step_rms_deg"])
        / max(abs(real_stats["step_rms_deg"]), 1e-12),
        "step_cov_relative_error": float(np.linalg.norm(ctrl_step_cov - real_step_cov) / cov_denom),
        "real_mean_x_deg": real_stats["mean_x_deg"],
        "real_mean_y_deg": real_stats["mean_y_deg"],
        "control_mean_x_deg": ctrl_stats["mean_x_deg"],
        "control_mean_y_deg": ctrl_stats["mean_y_deg"],
        "mean_position_error_deg": float(
            np.linalg.norm([
                ctrl_stats["mean_x_deg"] - real_stats["mean_x_deg"],
                ctrl_stats["mean_y_deg"] - real_stats["mean_y_deg"],
            ])
        ),
        "real_pos_cov_xx": real_stats["pos_cov_xx"],
        "real_pos_cov_xy": real_stats["pos_cov_xy"],
        "real_pos_cov_yy": real_stats["pos_cov_yy"],
        "control_pos_cov_xx": ctrl_stats["pos_cov_xx"],
        "control_pos_cov_xy": ctrl_stats["pos_cov_xy"],
        "control_pos_cov_yy": ctrl_stats["pos_cov_yy"],
        "pos_cov_relative_error": float(np.linalg.norm(ctrl_pos_cov - real_pos_cov) / pos_cov_denom),
    }


def _condition_blocks(
    *,
    condition: str,
    image: np.ndarray,
    trace: np.ndarray,
    t_max: int,
    crop_center_offset_px: tuple[float, float],
    real_cubes: np.ndarray,
    control_images: dict[str, np.ndarray],
    condition_seed: int | None = None,
) -> np.ndarray:
    """Return one overlapping model lag cube per aligned movie sample."""
    if condition == "real":
        blocks, _current = block_endpoint_lag_cubes(real_cubes)
        return blocks.astype(np.float32)
    if condition in TRAJECTORY_COMPARISON_CONDITIONS:
        control_trace, _description = _trajectory_for_condition(
            trace,
            condition,
            t_max=t_max,
            seed=0 if condition_seed is None else int(condition_seed),
        )
        cubes = model_lag_cubes_from_image_trace(
            image,
            control_trace,
            t_max=t_max,
            crop_center_offset_px=crop_center_offset_px,
        )
        blocks, _current = block_endpoint_lag_cubes(cubes)
        return blocks.astype(np.float32)
    if condition in control_images:
        cubes = model_lag_cubes_from_image_trace(
            control_images[condition],
            trace,
            t_max=t_max,
            crop_center_offset_px=crop_center_offset_px,
        )
        blocks, _current = block_endpoint_lag_cubes(cubes)
        return blocks.astype(np.float32)
    if condition in STABILIZED_VISUAL_CONTROL_CONDITIONS:
        visual_condition = condition.removeprefix("stabilized_")
        if visual_condition not in control_images:
            raise ValueError(f"Missing visual control image for condition: {condition}")
        stable_trace = np.repeat(np.mean(trace[:t_max], axis=0, keepdims=True), t_max, axis=0).astype(np.float32)
        cubes = model_lag_cubes_from_image_trace(
            control_images[visual_condition],
            stable_trace,
            t_max=t_max,
            crop_center_offset_px=crop_center_offset_px,
        )
        blocks, _current = block_endpoint_lag_cubes(cubes)
        return blocks.astype(np.float32)
    raise ValueError(f"Unsupported condition: {condition}")


def _ci95(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    n = np.sum(np.isfinite(arr), axis=0)
    std = np.nanstd(arr, axis=0, ddof=0)
    correction = np.sqrt(n / np.maximum(n - 1, 1))
    sem = std * correction / np.sqrt(np.maximum(n, 1))
    sem = np.where(n > 1, sem, 0.0)
    delta = 1.96 * sem
    return mean, mean - delta, mean + delta


def _write_series_npz(path: Path, records: list[dict[str, Any]], arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    for key in ("example_id", "kind", "condition"):
        payload[f"record_{key}"] = np.asarray([str(row[key]) for row in records])
    payload["record_image_index"] = np.asarray([int(row["image_index"]) for row in records], dtype=np.int32)
    payload["record_crop_rank"] = np.asarray([int(row["crop_rank"]) for row in records], dtype=np.int32)
    np.savez_compressed(path, **payload)


def _plot_individual_traces(
    records: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    *,
    metric_key: str,
    ylabel: str,
    conditions: tuple[str, ...],
    path: Path,
) -> None:
    time_s = arrays["time_s"]
    fig, axs = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True)
    for ax, kind in zip(axs, ("fixation", "microsaccade"), strict=True):
        for condition in conditions:
            color = CONDITION_COLORS[condition]
            ix = [
                i for i, row in enumerate(records)
                if row["kind"] == kind and row["condition"] == condition
            ]
            for i in ix:
                ax.plot(time_s, arrays[metric_key][i], color=color, alpha=0.25, lw=0.8)
            if ix:
                ax.plot([], [], color=color, lw=2.0, label=CONDITION_LABELS[condition])
        ax.set_title(f"{kind}: individual image/crop/trace movies")
        ax.set_ylabel(ylabel)
        ax.grid(color="0.9", lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axs[-1].set_xlabel("time in movie (s)")
    axs[0].legend(frameon=False, loc="upper left", ncols=min(4, len(conditions)))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_average_overview(
    records: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    *,
    conditions: tuple[str, ...],
    title: str,
    path: Path,
) -> None:
    time_s = arrays["time_s"]
    metrics = (
        ("cumulative_fisher_pattern", "cumulative pattern FI"),
        ("cumulative_fisher_pattern_per_spike", "pattern FI / expected spike"),
        ("cumulative_spatial_ssi_bits", "cumulative spatial SSI (bits)"),
        ("cumulative_spatial_ssi_bits_per_spike", "spatial SSI bits / expected spike"),
    )
    fig, axs = plt.subplots(2, 4, figsize=(19.0, 7.6), sharex=True)
    for r, kind in enumerate(("fixation", "microsaccade")):
        for c, (metric_key, ylabel) in enumerate(metrics):
            ax = axs[r, c]
            for condition in conditions:
                ix = [
                    i for i, row in enumerate(records)
                    if row["kind"] == kind and row["condition"] == condition
                ]
                if not ix:
                    continue
                mean, lo, hi = _ci95(arrays[metric_key][ix])
                color = CONDITION_COLORS[condition]
                ax.plot(time_s, mean, color=color, lw=2.0, label=CONDITION_LABELS[condition])
                ax.fill_between(time_s, lo, hi, color=color, alpha=0.16, linewidth=0)
            if r == 0:
                ax.set_title(ylabel)
            if c == 0:
                ax.set_ylabel(kind)
            if r == 1:
                ax.set_xlabel("time in movie (s)")
            ax.grid(color="0.9", lw=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    axs[0, 0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle(title, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _final_metric_lookup(rows: list[dict[str, Any]], metric: str) -> dict[tuple[str, int, int, str], float]:
    return {
        (str(row["example_id"]), int(row["image_index"]), int(row["crop_rank"]), str(row["condition"])): float(row[metric])
        for row in rows
    }


def _plot_gain_summary(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot condition-minus-stabilized final information gains."""
    metrics = (
        ("final_cumulative_fisher_pattern", "final pattern FI gain"),
        ("final_cumulative_spatial_ssi_bits", "final cumulative spatial SSI gain (bits)"),
        ("final_cumulative_spatial_ssi_bits_per_spike", "final spatial SSI efficiency gain (bits/spike)"),
    )
    gain_conditions = ("real",) + TRAJECTORY_CONTROL_CONDITIONS + ("pyramid_phase_scrambled",) + SF_BANDS_4
    fig, axs = plt.subplots(2, 3, figsize=(17.0, 8.4), sharex=True)
    x = np.arange(len(gain_conditions), dtype=np.float64)
    for r, kind in enumerate(("fixation", "microsaccade")):
        for c, (metric, title) in enumerate(metrics):
            ax = axs[r, c]
            values = _final_metric_lookup(rows, metric)
            means = []
            errs = []
            for condition in gain_conditions:
                deltas = []
                for row in rows:
                    if row["kind"] != kind or row["condition"] != condition:
                        continue
                    key = (str(row["example_id"]), int(row["image_index"]), int(row["crop_rank"]), "stabilized")
                    if key in values:
                        deltas.append(float(row[metric]) - values[key])
                arr = np.asarray(deltas, dtype=np.float64)
                if arr.size:
                    mean, lo, hi = _ci95(arr[:, None])
                    means.append(float(mean[0]))
                    errs.append(float(hi[0] - mean[0]))
                else:
                    means.append(float("nan"))
                    errs.append(0.0)
            colors = [CONDITION_COLORS[condition] for condition in gain_conditions]
            ax.bar(x, means, yerr=errs, capsize=3, color=colors, alpha=0.86)
            ax.axhline(0.0, color="0.15", lw=1.0)
            ax.set_title(f"{kind}: {title}")
            ax.set_xticks(x)
            ax.set_xticklabels([CONDITION_LABELS[cnd] for cnd in gain_conditions], rotation=25, ha="right")
            ax.grid(axis="y", color="0.9", lw=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle("Average information gain relative to stabilized movies; error bars are approximate 95% CI", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_final_metric_summary(rows: list[dict[str, Any]], path: Path) -> None:
    metrics = (
        ("final_cumulative_fisher_pattern", "final pattern FI"),
        ("final_cumulative_fisher_pattern_per_spike", "final pattern FI/spike"),
        ("final_cumulative_spatial_ssi_bits", "final cumulative spatial SSI (bits)"),
        ("final_cumulative_spatial_ssi_bits_per_spike", "final spatial SSI bits/spike"),
    )
    x = np.arange(len(MAIN_CONDITIONS), dtype=np.float64)
    width = 0.34
    fig, axs = plt.subplots(1, 4, figsize=(18.5, 4.3))
    for ax, (metric, title) in zip(axs, metrics, strict=True):
        for k_i, kind in enumerate(("fixation", "microsaccade")):
            means = []
            errs = []
            for condition in MAIN_CONDITIONS:
                vals = np.asarray([
                    float(row[metric]) for row in rows
                    if row["kind"] == kind and row["condition"] == condition
                ], dtype=np.float64)
                mean, lo, hi = _ci95(vals[:, None])
                means.append(float(mean[0]))
                errs.append(float(hi[0] - mean[0]))
            ax.bar(x + (k_i - 0.5) * width, means, width=width, yerr=errs, capsize=3, alpha=0.86, label=kind)
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[cnd] for cnd in MAIN_CONDITIONS], rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", color="0.9", lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axs[0].legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_information_step(
    *,
    config: PipelineConfig,
    dirs: dict[str, Path],
    model: Any,
    population: Any,
    device: Any,
    examples: list[TraceExample],
    image_by_index: dict[int, np.ndarray],
    crops: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int, int, str], np.ndarray]]:
    """Run model responses and cumulative information on all image/crop/trace pairs."""
    requested_conditions = _validate_conditions(config.conditions)
    shift_grid = (
        square_shift_grid(config.shift_grid_max_arcmin, config.shift_grid_step_arcmin)
        if config.shift_grid_mode == "square"
        else cross_shift_grid(config.shift_grid_max_arcmin, config.shift_grid_step_arcmin)
    )
    fisher_shifts = finite_difference_shift_set(config.fisher_step_arcmin)
    all_shifts = unique_shifts(fisher_shifts)

    series_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    pyramid_audit_rows: list[dict[str, Any]] = []
    trajectory_qc_rows: list[dict[str, Any]] = []
    unit_scores: dict[tuple[str, int, int, str], np.ndarray] = {}
    unit_score_rows: list[dict[str, Any]] = []
    series_arrays: dict[str, list[np.ndarray]] = {
        "cumulative_fisher_pattern": [],
        "cumulative_fisher_pattern_per_spike": [],
        "cumulative_expected_spikes": [],
        "cumulative_ssi_total_bits_per_window": [],
        "cumulative_ssi_pattern_bits_per_window": [],
        "prefix_ssi_total_bits_per_spike": [],
        "prefix_ssi_pattern_bits_per_spike": [],
        "cumulative_ssi_bits_per_second": [],
        "spatial_ssi_bits_per_spike": [],
        "spatial_ssi_bits_per_second": [],
        "cumulative_spatial_ssi_bits": [],
        "cumulative_spatial_ssi_bits_per_spike": [],
        "cumulative_spatial_ssi_bits_per_second": [],
        "cumulative_spatial_ssi_expected_spikes": [],
    }
    existing_summary_rows: list[dict[str, Any]] = []
    existing_series_records: list[dict[str, Any]] = []
    existing_series_arrays: dict[str, np.ndarray] | None = None
    existing_keys: set[tuple[str, int, int, str]] = set()
    existing_series_keys: set[tuple[str, int, int, str]] = set()
    if config.augment_existing and not config.recompute:
        existing_summary_rows = _read_csv_rows(dirs["metadata"] / "05_lagcube_information_summary.csv")
        existing_keys = {_row_key(row) for row in existing_summary_rows}
        existing_series = _load_existing_series(dirs["cache"] / "cumulative_information_series.npz")
        if existing_series is not None:
            existing_series_records, existing_series_arrays = existing_series
            existing_series_keys = {_row_key(row) for row in existing_series_records}

    pair_count = len(examples) * len(crops)
    pair_i = 0
    for crop in crops:
        image_index = int(crop["image_index"])
        crop_rank = int(crop["crop_rank"])
        crop_offset = (float(crop["offset_x_px"]), float(crop["offset_y_px"]))
        image = image_by_index[image_index]
        for example in examples:
            pair_i += 1
            print(
                f"Pair {pair_i}/{pair_count}: {example.example_id} image={image_index} crop={crop_rank}",
                flush=True,
            )
            seed = _example_seed(config.seed, example.example_id, image_index, crop_rank)
            real_cubes = model_lag_cubes_from_image_trace(
                image,
                example.trace,
                t_max=config.t_max,
                crop_center_offset_px=crop_offset,
            )
            control_images, audits = pyramid_local_image_controls(
                image,
                example.trace,
                np.random.default_rng(seed),
                crop_center_offset_px=crop_offset,
                height=PYRAMID_HEIGHT,
                order=PYRAMID_ORDER,
                sf_bands=SF_BANDS_4,
            )
            for audit in audits:
                row = dict(audit)
                row.update({
                    "example_id": example.example_id,
                    "kind": example.kind,
                    "image_index": image_index,
                    "crop_rank": crop_rank,
                    "crop_center_offset_x_px": float(crop_offset[0]),
                    "crop_center_offset_y_px": float(crop_offset[1]),
                })
                pyramid_audit_rows.append(row)

            for condition in requested_conditions:
                row_key = (example.example_id, image_index, crop_rank, condition)
                if config.augment_existing and not config.recompute and row_key in existing_keys and row_key in existing_series_keys:
                    print(f"  condition={condition} already present; skipping", flush=True)
                    continue
                print(f"  condition={condition}", flush=True)
                cubes = _condition_blocks(
                    condition=condition,
                    image=image,
                    trace=example.trace,
                    t_max=config.t_max,
                    crop_center_offset_px=crop_offset,
                    real_cubes=real_cubes,
                    control_images=control_images,
                    condition_seed=seed,
                )
                if condition in TRAJECTORY_COMPARISON_CONDITIONS:
                    control_trace, control_description = _trajectory_for_condition(
                        example.trace,
                        condition,
                        t_max=config.t_max,
                        seed=seed,
                    )
                    trajectory_qc_rows.append(
                        _trajectory_qc_row(
                            example=example,
                            image_index=image_index,
                            crop_rank=crop_rank,
                            condition=condition,
                            control_trace=control_trace,
                            control_description=control_description,
                        )
                    )
                rate_maps_by_shift = run_shifted_lag_cube_rate_maps(
                    model,
                    population,
                    device,
                    cubes,
                    all_shifts,
                    batch_size=config.batch_size,
                )
                fisher = cumulative_pattern_fisher(
                    rate_maps_by_shift,
                    fisher_step_arcmin=config.fisher_step_arcmin,
                    dt=DT,
                )
                center_rate_map = rate_maps_by_shift[(0.0, 0.0)]
                ssi = cumulative_spatial_ssi(center_rate_map, dt=DT)
                row = final_metric_row(
                    example_id=example.example_id,
                    kind=example.kind,
                    image_index=image_index,
                    condition=condition,
                    fisher=fisher,
                    ssi=ssi,
                    psd_errors=None,
                )
                row.update({
                    "crop_rank": crop_rank,
                    "crop_center_offset_x_px": float(crop_offset[0]),
                    "crop_center_offset_y_px": float(crop_offset[1]),
                    "n_events_in_window": int(example.n_events_in_window),
                    "event_onset": example.event_onset,
                    "fisher_step_arcmin": float(config.fisher_step_arcmin),
                    "shift_grid_max_arcmin": float(config.shift_grid_max_arcmin),
                    "shift_grid_step_arcmin": float(config.shift_grid_step_arcmin),
                    "shift_grid_mode": config.shift_grid_mode,
                    "n_shift_grid_states": int(shift_grid.shape[0]),
                    "spatial_ssi_uses_shift_grid": False,
                    "rate_map_time_samples": int(center_rate_map.shape[0]),
                    "rate_map_units_per_pixel": int(center_rate_map.shape[1]),
                    "rate_map_spatial_bins": int(center_rate_map.shape[2] * center_rate_map.shape[3]),
                })
                summary_rows.append(row)
                series_records.append({
                    "example_id": example.example_id,
                    "kind": example.kind,
                    "image_index": image_index,
                    "crop_rank": crop_rank,
                    "condition": condition,
                })
                for key in series_arrays:
                    source = fisher if key in fisher else ssi
                    series_arrays[key].append(np.asarray(source[key], dtype=np.float32))
                score = approximate_unit_fisher_scores(fisher["mu0"], fisher["dmu"])
                unit_scores[(example.example_id, image_index, crop_rank, condition)] = score
                top = np.argsort(score)[-min(12, score.size):][::-1]
                unit_score_rows.append({
                    "example_id": example.example_id,
                    "kind": example.kind,
                    "image_index": image_index,
                    "crop_rank": crop_rank,
                    "condition": condition,
                    "top_unit_indices": top.tolist(),
                    "top_unit_scores": score[top].tolist(),
                })

    new_arrays_np: dict[str, np.ndarray] = {}
    if summary_rows:
        new_arrays_np = {key: np.stack(vals, axis=0).astype(np.float32) for key, vals in series_arrays.items()}
        new_arrays_np["analysis_sample_index"] = block_current_samples(config.t_max, n_lags=N_LAGS).astype(np.int32)
        new_arrays_np["time_s"] = (new_arrays_np["analysis_sample_index"].astype(np.float32) * DT).astype(np.float32)

    if config.augment_existing and existing_summary_rows:
        new_keys = {_row_key(row) for row in summary_rows}
        merged_summary_rows = [row for row in existing_summary_rows if _row_key(row) not in new_keys] + summary_rows
    else:
        merged_summary_rows = summary_rows

    if config.augment_existing and existing_series_records and existing_series_arrays is not None:
        new_keys = {_row_key(row) for row in series_records}
        keep = np.asarray([_row_key(row) not in new_keys for row in existing_series_records], dtype=bool)
        merged_series_records = [row for row, keep_row in zip(existing_series_records, keep, strict=True) if bool(keep_row)] + series_records
        arrays_np = {}
        for key in series_arrays:
            old = np.asarray(existing_series_arrays[key])[keep]
            if key in new_arrays_np:
                arrays_np[key] = np.concatenate([old, new_arrays_np[key]], axis=0).astype(np.float32)
            else:
                arrays_np[key] = old.astype(np.float32)
        if "analysis_sample_index" in existing_series_arrays:
            arrays_np["analysis_sample_index"] = np.asarray(existing_series_arrays["analysis_sample_index"], dtype=np.int32)
        elif new_arrays_np:
            arrays_np["analysis_sample_index"] = new_arrays_np["analysis_sample_index"]
        else:
            arrays_np["analysis_sample_index"] = block_current_samples(config.t_max, n_lags=N_LAGS).astype(np.int32)
        if "time_s" in existing_series_arrays:
            arrays_np["time_s"] = np.asarray(existing_series_arrays["time_s"], dtype=np.float32)
        else:
            arrays_np["time_s"] = (arrays_np["analysis_sample_index"].astype(np.float32) * DT).astype(np.float32)
    else:
        merged_series_records = series_records
        arrays_np = new_arrays_np

    if not merged_summary_rows or not merged_series_records:
        raise RuntimeError("No information rows were available or computed for this run.")

    _write_series_npz(dirs["cache"] / "cumulative_information_series.npz", merged_series_records, arrays_np)
    write_csv(merged_summary_rows, dirs["metadata"] / "05_lagcube_information_summary.csv")
    write_csv(merged_series_records, dirs["metadata"] / "05_information_series_records.csv")
    if config.augment_existing:
        existing_unit_score_rows = _read_csv_rows(dirs["metadata"] / "04_unit_score_summary.csv")
        existing_pyramid_rows = _read_csv_rows(dirs["metadata"] / "02_pyramid_image_control_audit.csv")
        existing_trajectory_rows = _read_csv_rows(dirs["metadata"] / "03_trajectory_control_qc.csv")
        write_csv(existing_unit_score_rows + unit_score_rows, dirs["metadata"] / "04_unit_score_summary.csv")
        write_csv(existing_pyramid_rows or pyramid_audit_rows, dirs["metadata"] / "02_pyramid_image_control_audit.csv")
        write_csv(existing_trajectory_rows + trajectory_qc_rows, dirs["metadata"] / "03_trajectory_control_qc.csv")
    else:
        write_csv(unit_score_rows, dirs["metadata"] / "04_unit_score_summary.csv")
        write_csv(pyramid_audit_rows, dirs["metadata"] / "02_pyramid_image_control_audit.csv")
        write_csv(trajectory_qc_rows, dirs["metadata"] / "03_trajectory_control_qc.csv")

    _plot_individual_traces(
        merged_series_records,
        arrays_np,
        metric_key="cumulative_fisher_pattern",
        ylabel="cumulative pattern FI",
        conditions=PHASE_CONDITIONS,
        path=dirs["figures"] / "05_individual_cumulative_pattern_fi_phase.pdf",
    )
    _plot_individual_traces(
        merged_series_records,
        arrays_np,
        metric_key="cumulative_spatial_ssi_bits",
        ylabel="cumulative spatial SSI (bits)",
        conditions=PHASE_CONDITIONS,
        path=dirs["figures"] / "05_individual_cumulative_spatial_ssi_phase.pdf",
    )
    _plot_average_overview(
        merged_series_records,
        arrays_np,
        conditions=PHASE_CONDITIONS,
        title="Phase-control averages; bands are approximate 95% CI",
        path=dirs["figures"] / "05_phase_information_average_overview.pdf",
    )
    _plot_individual_traces(
        merged_series_records,
        arrays_np,
        metric_key="cumulative_spatial_ssi_bits_per_spike",
        ylabel="cumulative spatial SSI (bits/spike)",
        conditions=TRAJECTORY_COMPARISON_CONDITIONS,
        path=dirs["figures"] / "05_individual_cumulative_spatial_ssi_trajectory_controls.pdf",
    )
    _plot_average_overview(
        merged_series_records,
        arrays_np,
        conditions=TRAJECTORY_COMPARISON_CONDITIONS,
        title="Trajectory-control averages; bands are approximate 95% CI",
        path=dirs["figures"] / "05_trajectory_control_information_average_overview.pdf",
    )
    _plot_individual_traces(
        merged_series_records,
        arrays_np,
        metric_key="cumulative_fisher_pattern",
        ylabel="cumulative pattern FI",
        conditions=SF_CONDITIONS,
        path=dirs["figures"] / "06_individual_cumulative_pattern_fi_sf.pdf",
    )
    _plot_average_overview(
        merged_series_records,
        arrays_np,
        conditions=SF_CONDITIONS,
        title="Spatial-frequency averages; bands preserve natural pyramid energy",
        path=dirs["figures"] / "06_sf_information_average_overview.pdf",
    )
    if any(condition in {row["condition"] for row in merged_series_records} for condition in STABILIZED_VISUAL_CONTROL_CONDITIONS):
        _plot_average_overview(
            merged_series_records,
            arrays_np,
            conditions=AUGMENTED_VISUAL_COMPARISON_CONDITIONS,
            title="Direct FEM-vs-stabilized visual-control averages; bands are approximate 95% CI",
            path=dirs["figures"] / "06_stabilized_visual_control_information_average_overview.pdf",
        )
    _plot_final_metric_summary(merged_summary_rows, dirs["figures"] / "07_final_metric_summary.pdf")
    _plot_gain_summary(merged_summary_rows, dirs["figures"] / "07_information_gain_vs_stabilized.pdf")

    write_json(dirs["metadata"] / "05_information_grid.json", {
        "n_trace_examples": len(examples),
        "n_image_crops": len(crops),
        "n_pairs": len(examples) * len(crops),
        "requested_conditions": list(requested_conditions),
        "conditions": sorted({_canonical_condition_name(str(row["condition"])) for row in merged_summary_rows}),
        "default_conditions": list(MAIN_CONDITIONS),
        "stabilized_visual_control_conditions": list(STABILIZED_VISUAL_CONTROL_CONDITIONS),
        "phase_conditions": list(PHASE_CONDITIONS),
        "sf_conditions": list(SF_CONDITIONS),
        "trajectory_conditions": list(TRAJECTORY_COMPARISON_CONDITIONS),
        "shift_grid_deg": shift_grid.tolist(),
        "all_shifts_deg": all_shifts.tolist(),
        "analysis_sample_index": arrays_np["analysis_sample_index"].tolist(),
    })
    return merged_summary_rows, merged_series_records, unit_scores


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    run_dir = OUTPUT_DIR / run_slug(config)
    dirs = ensure_run_dirs(run_dir)
    config = _config_with_existing_run_definition(config, dirs["metadata"] / "run_config.json")
    config = PipelineConfig(
        **{
            **asdict(config),
            "conditions": _validate_conditions(config.conditions),
        }
    )
    summary_path = dirs["metadata"] / "run_summary.json"
    if summary_path.exists() and not config.recompute and not config.augment_existing:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if (
        summary_path.exists()
        and config.augment_existing
        and not config.recompute
        and _requested_conditions_complete(dirs["metadata"] / "05_lagcube_information_summary.csv", config.conditions)
    ):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    write_json(dirs["metadata"] / "run_config.json", _jsonable_config(config))

    model, _model_info, device = load_digital_twin()
    rng = np.random.default_rng(config.seed)
    population, population_rows = build_analysis_population(
        model,
        N=config.population_size,
        rng=rng,
        selection=config.population_selection,
        performance_metric=config.performance_metric,
        min_performance_score=config.min_performance_score,
        ccmax_threshold=config.ccmax_threshold,
        grid_position_mode=config.population_grid_position_mode,
        grid_stride=config.population_grid_stride,
        deduplicate_units=config.deduplicate_units,
        dedupe_correlation_threshold=config.dedupe_correlation_threshold,
        dedupe_candidate_multiplier=config.dedupe_candidate_multiplier,
        dedupe_battery_frames=config.dedupe_battery_frames,
        dedupe_battery_seed=config.seed,
        dedupe_batch_size=config.batch_size,
    )
    write_csv(population_rows, dirs["metadata"] / "00_population_units.csv")
    n_biological_twins = len({int(row["global_unit_idx"]) for row in population_rows}) if population_rows else 0

    examples = run_trace_selection_step(
        figure_dir=dirs["figures"],
        metadata_dir=dirs["metadata"],
        seed=config.seed,
        n_examples_per_kind=config.n_examples_per_kind,
        t_max=config.t_max,
        stride=config.stride,
        model=model,
    )
    examples = _filter_trace_examples(config, examples)
    write_csv(
        [
            {
                "example_id": example.example_id,
                "kind": example.kind,
                "source_trace_index": example.source_trace_index,
                "window_start": example.window_start,
                "window_stop": example.window_stop,
                "n_events_in_window": example.n_events_in_window,
                "event_onset": example.event_onset,
            }
            for example in examples
        ],
        dirs["metadata"] / "01_trace_examples_used.csv",
    )

    image_by_index, image_crops, image_figures = select_image_crops(
        image_indices=config.image_indices,
        n_crops_per_image=config.n_crops_per_image,
        figure_dir=dirs["figures"],
        metadata_path=dirs["metadata"] / "02_image_crop_hotspots.csv",
        seed=config.seed,
        t_max=config.t_max,
        trace_arrays=[example.trace for example in examples],
    )
    crop_table = crop_rows(image_crops)

    stimulus_movies: list[str] = []
    if config.make_stimulus_movies:
        stimulus_movies = make_representative_stimulus_movies(
            examples=examples,
            image_by_index=image_by_index,
            crop_rows=crop_table,
            movie_dir=dirs["movies"],
            figure_dir=dirs["figures"],
            seed=config.seed,
            t_max=config.t_max,
            fps=config.movie_fps,
            max_examples_per_kind=config.movie_examples_per_kind,
        )

    summary_rows, _series_records, unit_scores = run_information_step(
        config=config,
        dirs=dirs,
        model=model,
        population=population,
        device=device,
        examples=examples,
        image_by_index=image_by_index,
        crops=crop_table,
    )

    activation_movies: list[str] = []
    if config.make_activation_movies:
        activation_movies = make_activation_movies(
            model=model,
            population=population,
            device=device,
            examples=examples,
            image_by_index=image_by_index,
            crop_rows=crop_table,
            unit_scores=unit_scores,
            movie_dir=dirs["movies"],
            figure_dir=dirs["figures"],
            seed=config.seed,
            t_max=config.t_max,
            batch_size=config.batch_size,
            fps=config.movie_fps,
            max_examples_per_kind=config.movie_examples_per_kind,
        )

    summary = {
        "run_dir": str(run_dir),
        "n_trace_examples": len(examples),
        "n_images": len(image_by_index),
        "n_image_crops": len(crop_table),
        "n_biological_twins": int(n_biological_twins),
        "n_simulated_neurons": int(population.N),
        "n_summary_rows": len(summary_rows),
        "conditions": sorted({_canonical_condition_name(str(row["condition"])) for row in summary_rows}),
        "requested_conditions": list(config.conditions),
        "phase_conditions": list(PHASE_CONDITIONS),
        "sf_conditions": list(SF_CONDITIONS),
        "stabilized_visual_control_conditions": list(STABILIZED_VISUAL_CONTROL_CONDITIONS),
        "figures": sorted(str(path) for path in dirs["figures"].glob("*.pdf")),
        "image_selection_figures": image_figures,
        "stimulus_movies": stimulus_movies,
        "activation_movies": activation_movies,
        "movies": stimulus_movies + activation_movies,
        "summary_csv": str(dirs["metadata"] / "05_lagcube_information_summary.csv"),
        "series_npz": str(dirs["cache"] / "cumulative_information_series.npz"),
    }
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None,
                        help="Optional readable folder name under outputs/twininfo.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-indices", nargs="+", type=int, default=None,
                        help="Natural image indices. Omit to use every available natural image.")
    parser.add_argument("--n-crops-per-image", type=int, default=3)
    parser.add_argument("--n-examples-per-kind", type=int, default=10)
    parser.add_argument("--selected-trace-example-ids", nargs="+", default=())
    parser.add_argument("--t-max", type=int, default=128)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--population-selection", choices=POPULATION_SELECTION_MODES, default="top_performance",
                        help="How to choose simulated neurons.")
    parser.add_argument("--performance-metric", choices=PERFORMANCE_METRICS, default="ccnorm",
                        help="Fig. 3 metric used when --population-selection=top_performance.")
    parser.add_argument("--min-performance-score", type=float, default=None,
                        help="Optional minimum score for --performance-metric, e.g. ccnorm >= 0.5.")
    parser.add_argument("--population-grid-position-mode", choices=GRID_POSITION_MODES, default="full_grid",
                        help="How to assign one retinotopic grid position to each selected unit.")
    parser.add_argument("--population-grid-stride", type=int, default=1,
                        help="Stride through the model's existing spatial rate map when using full_grid.")
    parser.add_argument("--deduplicate-units", action=argparse.BooleanOptionalAction, default=True,
                        help="Drop top-ranked biological twins whose battery responses are too correlated.")
    parser.add_argument("--dedupe-correlation-threshold", type=float, default=0.95)
    parser.add_argument("--dedupe-candidate-multiplier", type=float, default=5.0)
    parser.add_argument("--dedupe-battery-frames", type=int, default=96)
    parser.add_argument("--ccmax-threshold", type=float, default=DEFAULT_CCMAX_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fisher-step-arcmin", type=float, default=0.5)
    parser.add_argument("--shift-grid-max-arcmin", type=float, default=1.0)
    parser.add_argument("--shift-grid-step-arcmin", type=float, default=1.0)
    parser.add_argument("--shift-grid-mode", choices=("square", "cross"), default="square")
    parser.add_argument("--make-stimulus-movies", action="store_true")
    parser.add_argument("--make-activation-movies", action="store_true")
    parser.add_argument("--movie-examples-per-kind", type=int, default=1)
    parser.add_argument("--movie-fps", type=int, default=30)
    parser.add_argument("--conditions", nargs="+", choices=ALL_CONDITIONS, default=None,
                        help=(
                            "Subset of conditions to compute. Use with --augment-existing to add missing rows "
                            "to an existing run without recomputing standard conditions."
                        ))
    parser.add_argument("--include-stabilized-visual-controls", action="store_true",
                        help="Append all stabilized pyramid/SF visual-control conditions to the requested set.")
    parser.add_argument("--augment-existing", action="store_true",
                        help=(
                            "Merge newly computed condition rows into an existing run. Existing run_config "
                            "settings are reused when present."
                        ))
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = tuple(args.conditions) if args.conditions is not None else MAIN_CONDITIONS
    if bool(args.include_stabilized_visual_controls):
        conditions = tuple(dict.fromkeys((*conditions, *STABILIZED_VISUAL_CONTROL_CONDITIONS)))
    summary = run_pipeline(PipelineConfig(
        run_name=args.run_name,
        seed=args.seed,
        image_indices=None if args.image_indices is None else tuple(int(v) for v in args.image_indices),
        n_crops_per_image=args.n_crops_per_image,
        n_examples_per_kind=args.n_examples_per_kind,
        selected_trace_example_ids=tuple(str(v) for v in args.selected_trace_example_ids),
        t_max=args.t_max,
        stride=args.stride,
        population_size=args.population_size,
        population_selection=args.population_selection,
        performance_metric=args.performance_metric,
        min_performance_score=args.min_performance_score,
        population_grid_position_mode=args.population_grid_position_mode,
        population_grid_stride=args.population_grid_stride,
        deduplicate_units=bool(args.deduplicate_units),
        dedupe_correlation_threshold=args.dedupe_correlation_threshold,
        dedupe_candidate_multiplier=args.dedupe_candidate_multiplier,
        dedupe_battery_frames=args.dedupe_battery_frames,
        ccmax_threshold=args.ccmax_threshold,
        batch_size=args.batch_size,
        fisher_step_arcmin=args.fisher_step_arcmin,
        shift_grid_max_arcmin=args.shift_grid_max_arcmin,
        shift_grid_step_arcmin=args.shift_grid_step_arcmin,
        shift_grid_mode=args.shift_grid_mode,
        make_stimulus_movies=bool(args.make_stimulus_movies),
        make_activation_movies=bool(args.make_activation_movies),
        movie_examples_per_kind=args.movie_examples_per_kind,
        movie_fps=args.movie_fps,
        conditions=conditions,
        augment_existing=bool(args.augment_existing),
        recompute=bool(args.recompute),
    ))
    print("Twininfo production pipeline complete")
    print(f"  run: {summary['run_dir']}")
    print(f"  summary: {summary['summary_csv']}")
    print(f"  figures: {len(summary['figures'])}")
    print(f"  movies: {len(summary['movies'])}")


if __name__ == "__main__":
    main()
