"""Run Figure 5 population checks 5-9 on natural-image twininfo movies.

This replaces the earlier e-optotype cached-rate scaffold for Figure 5
interpretation.  It recomputes only center-location biological-twin response
channels for the natural-image movies described by a ``jake.twininfo`` run,
then treats natural-image identity as the stimulus axis.

The production twininfo run uses a 16-channel biological readout over a 51x51
spatial grid for spatial SSI.  For tractable population-coding checks here we
use the 16 biological channels at the center readout location, not the full
16 x 51 x 51 spatial population.

Example
-------
.venv/bin/python declan/active_sensing_movie_information/run_figure5_natural_image_population_checks_5_to_9.py \
  --run-dir outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu \
  --conditions real,stabilized,random_amp,random_amp_cloud_matched,random_cov,trajectory_order_shuffle
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from jake.twininfo.common import compute_rate_map_batched, load_digital_twin
from jake.twininfo.image_selection import PYRAMID_HEIGHT, PYRAMID_ORDER, SF_BANDS_4, select_image_crops
from jake.twininfo.lagcube_information import lag_cubes_to_stim
from jake.twininfo.pipeline import (
    PipelineConfig,
    STABILIZED_VISUAL_CONTROL_CONDITIONS,
    TRAJECTORY_COMPARISON_CONDITIONS,
    _condition_blocks,
    _example_seed,
    _trace_stats_for_qc,
    _trajectory_for_condition,
    _validate_conditions,
)
from jake.twininfo.population import build_analysis_population
from jake.twininfo.retinal_examples import model_lag_cubes_from_image_trace, pyramid_local_image_controls
from jake.twininfo.trace_selection import run_trace_selection_step


DEFAULT_RUN_DIR = Path("outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu")
DEFAULT_OUT_DIR = Path("outputs/active_sensing_movie_information/figure5_natural_image_population_checks_5_to_9")
DEFAULT_CONDITIONS = (
    "real",
    "stabilized",
    "random_amp",
    "random_amp_cloud_matched",
    "random_cov",
    "trajectory_order_shuffle",
    "sf_low",
    "sf_mid_low",
    "sf_mid_high",
    "sf_high",
    "pyramid_phase_scrambled",
    "stabilized_sf_low",
    "stabilized_sf_mid_low",
    "stabilized_sf_mid_high",
    "stabilized_sf_high",
    "stabilized_pyramid_phase_scrambled",
)


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stable_group_value(value: Any) -> Any:
    if isinstance(value, float) and np.isnan(value):
        return "__nan__"
    return value


def load_pipeline_config(run_dir: Path) -> PipelineConfig:
    path = run_dir / "metadata" / "run_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = set(PipelineConfig.__dataclass_fields__)
    kwargs = {key: value for key, value in payload.items() if key in fields}
    if kwargs.get("image_indices") is not None:
        kwargs["image_indices"] = tuple(int(v) for v in kwargs["image_indices"])
    if kwargs.get("selected_trace_example_ids") is not None:
        kwargs["selected_trace_example_ids"] = tuple(str(v) for v in kwargs["selected_trace_example_ids"])
    if kwargs.get("conditions") is not None:
        kwargs["conditions"] = tuple(str(v) for v in kwargs["conditions"])
    return PipelineConfig(**kwargs)


def infer_available_conditions(run_dir: Path) -> tuple[str, ...]:
    """Use the completed twininfo summary as the default condition provenance."""
    summary_path = run_dir / "metadata" / "05_lagcube_information_summary.csv"
    if not summary_path.exists():
        return DEFAULT_CONDITIONS
    conditions: list[str] = []
    for row in read_csv_rows(summary_path):
        condition = str(row.get("condition", "")).strip()
        if condition and condition not in conditions:
            conditions.append(condition)
    return tuple(conditions) if conditions else DEFAULT_CONDITIONS


def selected_examples(config: PipelineConfig, run_dir: Path, model: Any, scratch_dir: Path):
    used = read_csv_rows(run_dir / "metadata" / "01_trace_examples_used.csv")
    wanted_ids = [row["example_id"] for row in used]
    examples = run_trace_selection_step(
        figure_dir=scratch_dir / "figures",
        metadata_dir=scratch_dir / "metadata",
        seed=int(config.seed),
        n_examples_per_kind=int(config.n_examples_per_kind),
        t_max=int(config.t_max),
        stride=int(config.stride),
        model=model,
    )
    by_id = {example.example_id: example for example in examples}
    missing = [example_id for example_id in wanted_ids if example_id not in by_id]
    if missing:
        raise ValueError(f"Could not reproduce selected trace examples: {missing}")
    return [by_id[example_id] for example_id in wanted_ids]


def crop_rows_from_metadata(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv_rows(run_dir / "metadata" / "02_image_crop_hotspots.csv"):
        rows.append(
            {
                "image_index": int(row["image_index"]),
                "crop_rank": int(row["crop_rank"]),
                "center_x_px": float(row["center_x_px"]),
                "center_y_px": float(row["center_y_px"]),
                "offset_x_px": float(row["offset_x_px"]),
                "offset_y_px": float(row["offset_y_px"]),
            }
        )
    return rows


def center_channel_rates(model: Any, population: Any, cubes: np.ndarray, *, batch_size: int) -> np.ndarray:
    readout = population.readout.to(model.device)
    rate_map = compute_rate_map_batched(
        model,
        readout,
        lag_cubes_to_stim(cubes),
        batch_size=int(batch_size),
    )
    arr = rate_map.detach().cpu().numpy().astype(np.float32)
    center_r = int(arr.shape[2] // 2)
    center_c = int(arr.shape[3] // 2)
    return arr[:, :, center_r, center_c].astype(np.float32)


def build_response_cache(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    run_dir = Path(args.run_dir)
    config = load_pipeline_config(run_dir)
    requested_conditions = tuple(parse_csv(args.conditions)) if args.conditions else infer_available_conditions(run_dir)
    conditions = tuple(_validate_conditions(requested_conditions))
    if args.max_images > 0:
        allowed_images = set(
            sorted({int(row["image_index"]) for row in read_csv_rows(run_dir / "metadata" / "02_image_crop_hotspots.csv")})[
                : int(args.max_images)
            ]
        )
    else:
        allowed_images = None

    model, _model_info, device = load_digital_twin()
    rng = np.random.default_rng(int(config.seed))
    population, _population_rows = build_analysis_population(
        model,
        N=int(config.population_size),
        rng=rng,
        selection=str(config.population_selection),
        performance_metric=str(config.performance_metric),
        min_performance_score=config.min_performance_score,
        ccmax_threshold=float(config.ccmax_threshold),
        grid_position_mode="center",
        grid_stride=int(config.population_grid_stride),
        deduplicate_units=bool(config.deduplicate_units),
        dedupe_correlation_threshold=float(config.dedupe_correlation_threshold),
        dedupe_candidate_multiplier=float(config.dedupe_candidate_multiplier),
        dedupe_battery_frames=int(config.dedupe_battery_frames),
        dedupe_battery_seed=int(config.seed),
        dedupe_batch_size=int(config.batch_size),
    )
    scratch_dir = out_dir / "_scratch_reproduce_selection"
    examples = selected_examples(config, run_dir, model, scratch_dir)
    select_image_result = select_image_crops(
        image_indices=config.image_indices,
        n_crops_per_image=int(config.n_crops_per_image),
        figure_dir=None,
        metadata_path=None,
        seed=int(config.seed),
        t_max=int(config.t_max),
        trace_arrays=[example.trace for example in examples],
    )
    image_by_index = select_image_result[0]
    crops = crop_rows_from_metadata(run_dir)
    if allowed_images is not None:
        crops = [crop for crop in crops if int(crop["image_index"]) in allowed_images]

    records: list[dict[str, Any]] = []
    rate_series: list[np.ndarray] = []
    rate_means: list[np.ndarray] = []
    total = len(crops) * len(examples) * len(conditions)
    done = 0
    for crop in crops:
        image_index = int(crop["image_index"])
        crop_rank = int(crop["crop_rank"])
        crop_offset = (float(crop["offset_x_px"]), float(crop["offset_y_px"]))
        image = image_by_index[image_index]
        for example in examples:
            seed = _example_seed(int(config.seed), example.example_id, image_index, crop_rank)
            real_cubes = model_lag_cubes_from_image_trace(
                image,
                example.trace,
                t_max=int(config.t_max),
                crop_center_offset_px=crop_offset,
            )
            control_images, _audits = pyramid_local_image_controls(
                image,
                example.trace,
                np.random.default_rng(seed),
                crop_center_offset_px=crop_offset,
                height=PYRAMID_HEIGHT,
                order=PYRAMID_ORDER,
                sf_bands=SF_BANDS_4,
            )
            for condition in conditions:
                done += 1
                print(f"{done}/{total} natural-image center rates: image={image_index} {example.example_id} {condition}", flush=True)
                cubes = _condition_blocks(
                    condition=condition,
                    image=image,
                    trace=example.trace,
                    t_max=int(config.t_max),
                    crop_center_offset_px=crop_offset,
                    real_cubes=real_cubes,
                    control_images=control_images,
                    condition_seed=seed,
                )
                rates = center_channel_rates(model, population, cubes, batch_size=int(args.batch_size or config.batch_size))
                records.append(
                    {
                        "source": "twininfo_natural_image_center_rates",
                        "example_id": example.example_id,
                        "kind": example.kind,
                        "image_index": image_index,
                        "crop_rank": crop_rank,
                        "condition": condition,
                    }
                )
                rate_series.append(rates.astype(np.float32))
                rate_means.append(np.mean(rates, axis=0, dtype=np.float64).astype(np.float32))

    rates_np = np.stack(rate_series, axis=0).astype(np.float32)
    means_np = np.stack(rate_means, axis=0).astype(np.float32)
    write_csv_rows(out_dir / "natural_image_center_rate_records.csv", records)
    np.savez_compressed(
        out_dir / "natural_image_center_rates.npz",
        rates=rates_np,
        time_averaged_rates=means_np,
    )
    return records, rates_np, means_np


def trace_for_condition_pose(
    *,
    trace: np.ndarray,
    condition: str,
    t_max: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    if condition in TRAJECTORY_COMPARISON_CONDITIONS:
        return _trajectory_for_condition(trace, condition, t_max=t_max, seed=seed)
    tr = np.asarray(trace[:t_max], dtype=np.float32)
    if condition in STABILIZED_VISUAL_CONTROL_CONDITIONS:
        center = np.mean(tr, axis=0, keepdims=True).astype(np.float32)
        return np.repeat(center, tr.shape[0], axis=0).astype(np.float32), "visual_control_trial_mean_stabilized"
    return tr.copy(), "visual_control_measured_trace"


def pose_frame_rows(
    *,
    record_index: int,
    record: dict[str, Any],
    trace: np.ndarray,
    description: str,
) -> list[dict[str, Any]]:
    tr = np.asarray(trace, dtype=np.float64)
    center = np.mean(tr, axis=0, keepdims=True)
    disp = tr - center
    step = np.diff(tr, axis=0, prepend=tr[:1])
    return [
        {
            "source": "natural_image_condition_pose_covariates",
            "record_index": record_index,
            "example_id": record.get("example_id", ""),
            "kind": record.get("kind", ""),
            "image_index": record.get("image_index", ""),
            "crop_rank": record.get("crop_rank", ""),
            "condition": record.get("condition", ""),
            "trace_description": description,
            "frame_idx": frame_idx,
            "x_deg": float(tr[frame_idx, 0]),
            "y_deg": float(tr[frame_idx, 1]),
            "x_centered_deg": float(disp[frame_idx, 0]),
            "y_centered_deg": float(disp[frame_idx, 1]),
            "radial_displacement_deg": float(np.linalg.norm(disp[frame_idx])),
            "step_x_deg": float(step[frame_idx, 0]),
            "step_y_deg": float(step[frame_idx, 1]),
            "step_norm_deg": float(np.linalg.norm(step[frame_idx])),
        }
        for frame_idx in range(tr.shape[0])
    ]


def export_pose_covariates(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    run_dir = Path(args.run_dir)
    config = load_pipeline_config(run_dir)
    model, _model_info, _device = load_digital_twin()
    scratch_dir = out_dir / "_scratch_reproduce_selection"
    examples = selected_examples(config, run_dir, model, scratch_dir)
    example_by_id = {example.example_id: example for example in examples}

    summary_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        example_id = str(record.get("example_id", ""))
        if example_id not in example_by_id:
            raise ValueError(f"Missing reproduced trace example for {example_id}")
        image_index = int(record["image_index"])
        crop_rank = int(record["crop_rank"])
        condition = str(record["condition"])
        seed = _example_seed(int(config.seed), example_id, image_index, crop_rank)
        pose_trace, description = trace_for_condition_pose(
            trace=example_by_id[example_id].trace,
            condition=condition,
            t_max=int(config.t_max),
            seed=seed,
        )
        stats = _trace_stats_for_qc(pose_trace)
        summary_rows.append(
            {
                "source": "natural_image_condition_pose_covariates",
                "record_index": record_index,
                "example_id": example_id,
                "kind": record.get("kind", ""),
                "image_index": image_index,
                "crop_rank": crop_rank,
                "condition": condition,
                "trace_description": description,
                "n_frames": int(pose_trace.shape[0]),
                **stats,
            }
        )
        frame_rows.extend(
            pose_frame_rows(
                record_index=record_index,
                record=record,
                trace=pose_trace,
                description=description,
            )
        )

    summary_path = out_dir / "natural_image_condition_pose_summary.csv"
    frame_path = out_dir / "natural_image_condition_pose_frames.csv"
    write_csv_rows(summary_path, summary_rows)
    write_csv_rows(frame_path, frame_rows)
    return summary_path, frame_path


def load_response_cache(out_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    records = read_csv_rows(out_dir / "natural_image_center_rate_records.csv")
    with np.load(out_dir / "natural_image_center_rates.npz") as npz:
        means = np.asarray(npz["time_averaged_rates"], dtype=np.float64)
    if len(records) != int(means.shape[0]):
        raise ValueError(
            f"Cache row mismatch: {len(records)} metadata records but "
            f"{int(means.shape[0])} time-averaged rate rows."
        )
    return records, means


def filter_records_and_means(
    records: list[dict[str, Any]],
    means: np.ndarray,
    requested_conditions: tuple[str, ...] | None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if requested_conditions is None:
        return records, means
    wanted = set(_validate_conditions(requested_conditions))
    keep = [idx for idx, row in enumerate(records) if str(row["condition"]) in wanted]
    if not keep:
        available = ", ".join(sorted({str(row["condition"]) for row in records}))
        requested = ", ".join(sorted(wanted))
        raise ValueError(f"No requested conditions found in cache. Requested: {requested}. Available: {available}")
    return [records[idx] for idx in keep], means[np.asarray(keep, dtype=np.int64)]


def arrays_by_class(records: list[dict[str, Any]], means: np.ndarray, condition: str) -> dict[str, np.ndarray]:
    grouped: dict[str, list[tuple[tuple[str, str], np.ndarray]]] = {}
    for idx, row in enumerate(records):
        if str(row["condition"]) != condition:
            continue
        image_key = str(row["image_index"])
        repeat_key = (str(row["kind"]), str(row["example_id"]))
        grouped.setdefault(image_key, []).append((repeat_key, means[idx]))
    out: dict[str, np.ndarray] = {}
    for image_key, vals in grouped.items():
        vals_sorted = [arr for _repeat, arr in sorted(vals, key=lambda item: item[0])]
        out[image_key] = np.stack(vals_sorted, axis=0).astype(np.float64)
    n = min(arr.shape[0] for arr in out.values())
    return {key: arr[:n] for key, arr in out.items()}


def top_subspace_from_samples(samples: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(samples, dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    if x.shape[0] < 2:
        return np.zeros((x.shape[1], 0)), np.zeros((0,))
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    k_eff = max(1, min(int(k), vt.shape[0]))
    eigvals = (s[:k_eff] ** 2) / max(x.shape[0] - 1, 1)
    return vt[:k_eff].T, eigvals


def class_keys(x_by_class: dict[str, np.ndarray]) -> list[str]:
    return sorted(x_by_class, key=lambda value: int(value) if str(value).isdigit() else str(value))


def pooled_residuals(x_by_class: dict[str, np.ndarray]) -> np.ndarray:
    chunks = []
    for key in class_keys(x_by_class):
        x = x_by_class[key]
        chunks.append(x - np.mean(x, axis=0, keepdims=True))
    return np.concatenate(chunks, axis=0)


def signal_means(x_by_class: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([np.mean(x_by_class[key], axis=0) for key in class_keys(x_by_class)], axis=0)


def covariance_spectrum_diagnostics(condition: str, x_by_class: dict[str, np.ndarray], k_list: list[int]) -> list[dict[str, Any]]:
    residuals = pooled_residuals(x_by_class)
    means = signal_means(x_by_class)
    c_signal = covariance_from_samples(means)
    c_reaff = covariance_from_samples(residuals)
    signal_eigs = np.linalg.eigvalsh(c_signal)[::-1]
    reaff_eigs = np.linalg.eigvalsh(c_reaff)[::-1]
    signal_trace = float(np.sum(signal_eigs))
    reaff_trace = float(np.sum(reaff_eigs))

    rows = []
    for k in k_list:
        k_eff = min(int(k), int(signal_eigs.size), int(reaff_eigs.size))
        rows.append(
            {
                "check": "5_natural_image_covariance_spectrum_diagnostic",
                "source": "twininfo_natural_image_center_rates",
                "condition": condition,
                "k": int(k),
                "n_images": len(x_by_class),
                "n_repeats_per_image": int(next(iter(x_by_class.values())).shape[0]),
                "n_residual_samples": int(residuals.shape[0]),
                "n_units": int(residuals.shape[1]),
                "signal_trace": signal_trace,
                "reaff_trace": reaff_trace,
                "reaff_over_signal_trace": reaff_trace / (signal_trace + 1e-12),
                "signal_topk_trace_frac": float(np.sum(signal_eigs[:k_eff]) / (signal_trace + 1e-12)),
                "reaff_topk_trace_frac": float(np.sum(reaff_eigs[:k_eff]) / (reaff_trace + 1e-12)),
                "signal_effective_rank": float((np.sum(signal_eigs) ** 2) / (np.sum(signal_eigs ** 2) + 1e-12)),
                "reaff_effective_rank": float((np.sum(reaff_eigs) ** 2) / (np.sum(reaff_eigs ** 2) + 1e-12)),
                "subspace_ceiling_note": "k is large relative to 16 response dimensions; top-k capture can approach ceiling.",
            }
        )
    return rows


def covariance_from_samples(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr - np.mean(arr, axis=0, keepdims=True)
    c = arr.T @ arr / max(arr.shape[0] - 1, 1)
    return (c + c.T) / 2.0


def ridge_inverse(cov: np.ndarray, ridge_fraction: float) -> np.ndarray:
    c = np.asarray(cov, dtype=np.float64)
    scale = float(np.trace(c) / max(c.shape[0], 1))
    ridge = max(float(ridge_fraction) * scale, 1e-8)
    return np.linalg.pinv(c + np.eye(c.shape[0]) * ridge, hermitian=True)


def principal_angles_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    if u.size == 0 or v.size == 0:
        return np.asarray([], dtype=np.float64)
    qu, _ = np.linalg.qr(u)
    qv, _ = np.linalg.qr(v)
    s = np.linalg.svd(qu.T @ qv, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def pairwise_signal_vectors(means: np.ndarray, keys: list[str]) -> list[tuple[str, np.ndarray]]:
    out = []
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if j <= i:
                continue
            out.append((f"image_{ki}_vs_{kj}", means[i] - means[j]))
    return out


def alignment_metrics(condition: str, x_by_class: dict[str, np.ndarray], k_list: list[int], n_nulls: int, rng: np.random.Generator) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    residuals = pooled_residuals(x_by_class)
    means = signal_means(x_by_class)
    keys = class_keys(x_by_class)
    c_signal = covariance_from_samples(means)
    c_reaff = covariance_from_samples(residuals)
    u_signal_full, signal_eigs = top_subspace_from_samples(means, max(k_list))
    trace_signal = float(np.trace(c_signal))
    n_units = residuals.shape[1]
    rows = []
    pair_rows = []
    for k in k_list:
        u_reaff, reaff_eigs = top_subspace_from_samples(residuals, k)
        u_signal = u_signal_full[:, : min(k, u_signal_full.shape[1])]
        alpha = float(np.trace(u_reaff.T @ c_signal @ u_reaff) / (trace_signal + 1e-12))
        null_alphas = []
        for _ in range(int(n_nulls)):
            q, _ = np.linalg.qr(rng.normal(size=(n_units, u_reaff.shape[1])))
            null_alphas.append(float(np.trace(q.T @ c_signal @ q) / (trace_signal + 1e-12)))
        null_arr = np.asarray(null_alphas, dtype=np.float64)
        angles = principal_angles_deg(u_reaff, u_signal)
        rows.append(
            {
                "check": "5_natural_image_reafference_signal_alignment",
                "source": "twininfo_natural_image_center_rates",
                "condition": condition,
                "k": int(k),
                "n_images": len(keys),
                "n_repeats_per_image": int(next(iter(x_by_class.values())).shape[0]),
                "n_units": int(n_units),
                "alpha": alpha,
                "alpha_null_mean": float(np.mean(null_arr)) if null_arr.size else float("nan"),
                "alpha_null_std": float(np.std(null_arr, ddof=1)) if null_arr.size > 1 else float("nan"),
                "alpha_x_null": alpha / (float(np.mean(null_arr)) + 1e-12) if null_arr.size else float("nan"),
                "principal_angle_mean_deg": float(np.mean(angles)) if angles.size else float("nan"),
                "principal_angle_min_deg": float(np.min(angles)) if angles.size else float("nan"),
                "top_reaff_eig_sum": float(np.sum(reaff_eigs)),
                "top_signal_eig_sum": float(np.sum(signal_eigs[: min(k, signal_eigs.size)])),
            }
        )
        for pair, dmu in pairwise_signal_vectors(means, keys):
            denom = float(dmu @ dmu) + 1e-12
            pair_rows.append(
                {
                    "check": "5_natural_image_pairwise_information_limiting_projection",
                    "source": "twininfo_natural_image_center_rates",
                    "condition": condition,
                    "k": int(k),
                    "pair": pair,
                    "L_ij": float(dmu @ c_reaff @ dmu / denom),
                    "projected_signal_norm_frac": float(np.sum((dmu @ u_reaff) ** 2) / denom),
                }
            )
    return rows, pair_rows


def dprime_metrics(condition: str, x_by_class: dict[str, np.ndarray], ridge_fraction: float) -> list[dict[str, Any]]:
    residuals = pooled_residuals(x_by_class)
    c = covariance_from_samples(residuals)
    inv_full = ridge_inverse(c, ridge_fraction)
    inv_diag = ridge_inverse(np.diag(np.diag(c)), ridge_fraction)
    means = signal_means(x_by_class)
    keys = class_keys(x_by_class)
    rows = []
    for pair, dmu in pairwise_signal_vectors(means, keys):
        d_full = float(dmu @ inv_full @ dmu)
        d_diag = float(dmu @ inv_diag @ dmu)
        rows.append(
            {
                "check": "6_natural_image_constrained_population_coding",
                "source": "twininfo_natural_image_center_rates",
                "condition": condition,
                "pair": pair,
                "dprime2_pop": d_full,
                "dprime2_indep": d_diag,
                "eta_pop_over_indep": d_full / (d_diag + 1e-12),
                "ridge_fraction": float(ridge_fraction),
            }
        )
    return rows


def nearest_centroid_cv(x_by_class: dict[str, np.ndarray], n_splits: int, remove_k: int | None = None) -> tuple[float, float]:
    keys = class_keys(x_by_class)
    n = min(x_by_class[key].shape[0] for key in keys)
    if n < 2:
        return float("nan"), float("nan")
    folds = np.array_split(np.arange(n), min(max(2, int(n_splits)), n))
    accs = []
    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        u_remove = None
        if remove_k is not None and int(remove_k) > 0:
            train_residuals = []
            for key in keys:
                x = x_by_class[key][train_idx]
                train_residuals.append(x - np.mean(x, axis=0, keepdims=True))
            u_remove, _ = top_subspace_from_samples(np.concatenate(train_residuals, axis=0), int(remove_k))
        centroids = []
        for key in keys:
            x = x_by_class[key]
            if u_remove is not None and u_remove.size:
                x = x - (x @ u_remove) @ u_remove.T
            centroids.append(np.mean(x[train_idx], axis=0))
        centroids_arr = np.stack(centroids, axis=0)
        correct = 0
        total = 0
        for label, key in enumerate(keys):
            x = x_by_class[key][test_idx]
            if u_remove is not None and u_remove.size:
                x = x - (x @ u_remove) @ u_remove.T
            dist = np.sum((x[:, None, :] - centroids_arr[None, :, :]) ** 2, axis=2)
            pred = np.argmin(dist, axis=1)
            correct += int(np.sum(pred == label))
            total += int(pred.size)
        accs.append(correct / max(total, 1))
    return float(np.mean(accs)), float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0


def removeout_metrics(condition: str, x_by_class: dict[str, np.ndarray], k_list: list[int], n_splits: int) -> list[dict[str, Any]]:
    base_acc, base_std = nearest_centroid_cv(x_by_class, n_splits=n_splits, remove_k=None)
    rows = []
    for k in k_list:
        clean_acc, clean_std = nearest_centroid_cv(x_by_class, n_splits=n_splits, remove_k=k)
        rows.append(
            {
                "check": "7_natural_image_reafference_aware_removeout",
                "source": "twininfo_natural_image_center_rates",
                "condition": condition,
                "k": int(k),
                "cv_decoder": "nearest_centroid_image_identity",
                "removeout_basis_fit": "training_fold_residual_pca",
                "acc_original": base_acc,
                "acc_original_std": base_std,
                "acc_reaff_removed": clean_acc,
                "acc_reaff_removed_std": clean_std,
                "delta_removed_minus_original": clean_acc - base_acc,
                "n_splits": int(n_splits),
            }
        )
    return rows


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str], value_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(stable_group_value(row.get(k)) for k in group_keys)
        groups.setdefault(key, []).append(row)
    out = []
    for _key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        base = {k: group_rows[0].get(k) for k in group_keys}
        base["n"] = len(group_rows)
        for value_key in value_keys:
            vals = np.asarray([float(row.get(value_key, np.nan)) for row in group_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            base[f"{value_key}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            base[f"{value_key}_sem"] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        out.append(base)
    return out


def run_checks(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "natural_image_center_rates.npz"
    requested_conditions = tuple(parse_csv(args.conditions)) if args.conditions else None
    if bool(args.recompute_cache) or not cache_path.exists():
        records, _rates, means = build_response_cache(args, out_dir)
    else:
        records, means = load_response_cache(out_dir)
        records, means = filter_records_and_means(records, means, requested_conditions)
    conditions = sorted({str(row["condition"]) for row in records})
    k_list = [int(k) for k in parse_csv(args.k_list)]
    rng = np.random.default_rng(int(args.seed))

    inventory_rows = []
    spectrum_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    dprime_rows: list[dict[str, Any]] = []
    removeout_rows: list[dict[str, Any]] = []
    for condition in conditions:
        x_by_class = arrays_by_class(records, means, condition)
        inventory_rows.append(
            {
                "condition": condition,
                "n_images": len(x_by_class),
                "n_repeats_per_image": int(min(arr.shape[0] for arr in x_by_class.values())),
                "n_units": int(next(iter(x_by_class.values())).shape[1]),
            }
        )
        spectrum_rows.extend(covariance_spectrum_diagnostics(condition, x_by_class, k_list))
        a_rows, p_rows = alignment_metrics(condition, x_by_class, k_list, int(args.n_nulls), rng)
        alignment_rows.extend(a_rows)
        pair_rows.extend(p_rows)
        dprime_rows.extend(dprime_metrics(condition, x_by_class, float(args.ridge_fraction)))
        removeout_rows.extend(removeout_metrics(condition, x_by_class, k_list, int(args.n_splits)))

    dprime_summary = aggregate_rows(
        dprime_rows,
        group_keys=["condition"],
        value_keys=["dprime2_pop", "dprime2_indep", "eta_pop_over_indep"],
    )
    removeout_summary = aggregate_rows(
        removeout_rows,
        group_keys=["condition", "k"],
        value_keys=["acc_original", "acc_reaff_removed", "delta_removed_minus_original"],
    )
    alignment_summary = aggregate_rows(
        alignment_rows,
        group_keys=["condition", "k"],
        value_keys=["alpha", "alpha_x_null", "principal_angle_mean_deg"],
    )
    addback_rows = [
        {
            "check": "8_natural_image_compact_addback_removeout",
            "source": "twininfo_natural_image_center_rates",
            "row_status": "skipped_missing_compatible_natural_image_basis",
            "message": "The prior Figure 4 756-unit basis is not compatible with this run's 16 center-channel natural-image responses.",
        }
    ]

    write_csv_rows(out_dir / "natural_image_center_rate_inventory.csv", inventory_rows)
    write_csv_rows(out_dir / "check5_natural_image_covariance_spectrum_diagnostics.csv", spectrum_rows)
    write_csv_rows(out_dir / "check5_natural_image_reafference_signal_alignment.csv", alignment_rows)
    write_csv_rows(out_dir / "check5_natural_image_pairwise_Lij.csv", pair_rows)
    write_csv_rows(out_dir / "check6_natural_image_constrained_dprime.csv", dprime_rows)
    write_csv_rows(out_dir / "check6_natural_image_constrained_dprime_summary.csv", dprime_summary)
    write_csv_rows(out_dir / "check7_natural_image_reafference_removeout.csv", removeout_rows)
    write_csv_rows(out_dir / "check7_natural_image_reafference_removeout_summary.csv", removeout_summary)
    write_csv_rows(out_dir / "check8_natural_image_compact_addback_removeout.csv", addback_rows)
    write_csv_rows(out_dir / "check9_natural_image_condition_sweep_alignment_summary.csv", alignment_summary)

    pose_outputs: list[str] = []
    if bool(args.export_pose_covariates):
        pose_summary_path, pose_frame_path = export_pose_covariates(args=args, out_dir=out_dir, records=records)
        pose_outputs = [str(pose_summary_path), str(pose_frame_path)]

    manifest = {
        "source": "twininfo_natural_image_center_rates",
        "run_dir": str(args.run_dir),
        "conditions_requested": list(requested_conditions) if requested_conditions is not None else "cache_or_run_default",
        "conditions": conditions,
        "k_list": k_list,
        "n_nulls": int(args.n_nulls),
        "n_splits": int(args.n_splits),
        "ridge_fraction": float(args.ridge_fraction),
        "response_cache": str(cache_path),
        "stimulus_axis": "natural_image_identity",
        "response_space": "center readout location, biological twin channels",
        "deprecated_not_used": "scripts/temporal_decoding/data/rates e-optotype cached rates",
        "pose_covariate_outputs": pose_outputs,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Figure 5 natural-image population checks 5-9 complete")
    print(f"  out_dir: {out_dir}")
    print(f"  conditions: {', '.join(conditions)}")
    print(f"  alignment rows: {len(alignment_rows)}")
    print(f"  dprime rows: {len(dprime_rows)}")
    print(f"  removeout rows: {len(removeout_rows)}")
    if pose_outputs:
        print(f"  pose covariate outputs: {len(pose_outputs)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated condition list; defaults to conditions present in the source twininfo summary.",
    )
    p.add_argument("--k-list", type=str, default="2,10")
    p.add_argument("--n-nulls", type=int, default=100)
    p.add_argument("--n-splits", type=int, default=4)
    p.add_argument("--ridge-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--max-images", type=int, default=0, help="Smoke-test limit; 0 means all images from the run.")
    p.add_argument("--recompute-cache", action="store_true")
    p.add_argument("--export-pose-covariates", action="store_true")
    return p


def main() -> None:
    run_checks(build_parser().parse_args())


if __name__ == "__main__":
    main()
