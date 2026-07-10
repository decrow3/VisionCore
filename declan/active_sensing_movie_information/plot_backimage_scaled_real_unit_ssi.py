#!/usr/bin/env python3
"""Plot BackImage unit SSI along a real-eye-motion scale line.

The scale line is centered on each selected trace's mean position:

``scaled_trace = mean(trace) + scale * (trace - mean(trace))``.

Thus ``0x`` is a trial-mean stabilized movie, ``1x`` is the measured real
movie, and larger values amplify the same displacement history. This mirrors
the Vernier along-line SSI diagnostics while keeping the BackImage baseline as
a trial-mean control rather than a deterministic static-center oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jake.twininfo.common import DT, compute_rate_map_batched, load_digital_twin
from jake.twininfo.lagcube_information import block_endpoint_lag_cubes, lag_cubes_to_stim
from jake.twininfo.pipeline import PipelineConfig, _example_seed
from jake.twininfo.population import build_analysis_population
from jake.twininfo.retinal_examples import model_lag_cubes_from_image_trace
from jake.twininfo.stimuli import load_natural_images
from jake.twininfo.trace_selection import run_trace_selection_step


DEFAULT_RUN_DIR = ROOT / "outputs" / "twininfo" / "active-sensing-all-images-1crop-2fix2ms-16units-gpu"
DEFAULT_OUT_DIR = ROOT / "outputs" / "active_sensing_movie_information" / "backimage_spatial_ssi_scale_line"
DEFAULT_SCALES = "0,0.125,0.25,0.5,0.75,1,1.5,2,3"
EPS = 1e-8
CACHE_SCHEMA_VERSION = 1
STIMULUS_NORMALIZATION = "pixelnorm_raw_u8_minus_127_div_255"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scales", type=str, default=DEFAULT_SCALES)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Use only the first N image indices from the source crop table; 0 means all.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Use only the first N image/trace/crop pairs after filtering; 0 means all.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-units", type=int, default=12)
    parser.add_argument("--map-vmin-percentile", type=float, default=0.5)
    parser.add_argument("--map-vmax-percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--line-y-mode",
        choices=("absolute", "log2_ratio"),
        default="absolute",
        help="Plot absolute SSI bits/spike by default; log2_ratio reproduces the old SSI/0x diagnostic.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-individual-unit-sheets", action="store_true")
    return parser.parse_args()


def parse_scales(text: str) -> list[float]:
    vals = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not vals:
        raise ValueError("At least one scale is required.")
    if not any(np.isclose(vals, 0.0)):
        raise ValueError("Scale list must include 0x for the trial-mean baseline.")
    return vals


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_pipeline_config(run_dir: Path) -> PipelineConfig:
    payload = json.loads((run_dir / "metadata" / "run_config.json").read_text(encoding="utf-8"))
    kwargs: dict[str, Any] = {}
    tuple_fields = {"image_indices", "selected_trace_example_ids", "conditions"}
    for field in fields(PipelineConfig):
        if field.name not in payload:
            continue
        value = payload[field.name]
        if field.name in tuple_fields and value is not None:
            value = tuple(value)
        kwargs[field.name] = value
    return PipelineConfig(**kwargs)


def crop_rows_from_metadata(run_dir: Path, *, max_images: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_rows = read_csv_rows(run_dir / "metadata" / "02_image_crop_hotspots.csv")
    if not raw_rows:
        raise FileNotFoundError(f"Missing crop metadata under {run_dir}")
    allowed_images: set[int] | None = None
    if int(max_images) > 0:
        allowed_images = set(sorted({int(row["image_index"]) for row in raw_rows})[: int(max_images)])
    for row in raw_rows:
        image_index = int(row["image_index"])
        if allowed_images is not None and image_index not in allowed_images:
            continue
        rows.append(
            {
                "image_index": image_index,
                "crop_rank": int(row["crop_rank"]),
                "center_x_px": float(row["center_x_px"]),
                "center_y_px": float(row["center_y_px"]),
                "offset_x_px": float(row["offset_x_px"]),
                "offset_y_px": float(row["offset_y_px"]),
            }
        )
    return rows


def selected_examples(config: PipelineConfig, run_dir: Path, model: Any, scratch_dir: Path):
    wanted_ids = selected_example_ids_from_metadata(run_dir)
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


def selected_example_ids_from_metadata(run_dir: Path) -> list[str]:
    used = read_csv_rows(run_dir / "metadata" / "01_trace_examples_used.csv")
    if not used:
        raise FileNotFoundError(f"Missing selected trace metadata under {run_dir}")
    return [str(row["example_id"]) for row in used]


def scaled_trace(trace: np.ndarray, scale: float, t_max: int) -> np.ndarray:
    tr = np.asarray(trace[: int(t_max)], dtype=np.float32)
    center = np.mean(tr, axis=0, keepdims=True)
    return (center + float(scale) * (tr - center)).astype(np.float32)


def unit_spatial_ssi_for_movie(rate_map: np.ndarray) -> dict[str, np.ndarray | float]:
    """Final cumulative unit and population spatial SSI for one rate movie."""
    y = np.asarray(rate_map, dtype=np.float64)
    if y.ndim != 4:
        raise ValueError(f"Expected rate_map with shape (T, N, H, W), got {y.shape}")
    if np.any(y < 0):
        raise ValueError("rate_map must be non-negative")
    t_max, n_units, height, width = y.shape
    flat = y.reshape(t_max, n_units, height * width)
    rbar = np.mean(flat, axis=2)
    gain = flat / (rbar[..., None] + EPS)
    unit_bits_t = np.mean(gain * np.log2(gain + EPS), axis=2)
    unit_expected = np.sum(rbar * DT, axis=0)
    unit_bits = np.sum(unit_bits_t * rbar * DT, axis=0) / np.maximum(unit_expected, EPS)
    population_bits = float(np.sum(unit_bits_t * rbar * DT) / max(float(np.sum(unit_expected)), EPS))
    return {
        "unit_bits_per_spike": unit_bits.astype(np.float32),
        "unit_expected_spikes": unit_expected.astype(np.float32),
        "unit_mean_rate": np.mean(rbar, axis=0).astype(np.float32),
        "population_bits_per_spike": population_bits,
    }


def run_rate_map(model: Any, population: Any, cubes: np.ndarray, *, batch_size: int) -> np.ndarray:
    stim = lag_cubes_to_stim(cubes)
    rate_map = compute_rate_map_batched(
        model,
        population.readout.to(model.device),
        stim,
        batch_size=int(batch_size),
    )
    out = rate_map.detach().cpu().numpy().astype(np.float32)
    del rate_map, stim
    try:
        import torch

        if torch.cuda.is_available() and str(model.device).startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass
    return out


def cache_path(out_dir: Path) -> Path:
    return out_dir / "cache" / "backimage_scaled_real_unit_ssi_cache.npz"


def identity_text(identity: dict[str, Any]) -> str:
    return json.dumps(json_ready(identity), sort_keys=True, separators=(",", ":"))


def build_identity(
    *,
    args: argparse.Namespace,
    config: PipelineConfig,
    scales: list[float],
    crops: list[dict[str, Any]],
    example_ids: list[str],
    n_pairs: int,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "analysis": "backimage_scaled_real_unit_spatial_ssi",
        "run_dir": str(Path(args.run_dir).expanduser().resolve()),
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "scale_contract": "mean(trace)+scale*(trace-mean(trace)); 0x is trial-mean stabilized",
        "readout_time_contract": "all overlapping current-frame samples cumulative SSI",
        "spatial_ssi_ensemble": "convolutional output spatial positions",
        "scales": [float(v) for v in scales],
        "max_images": int(args.max_images),
        "max_pairs": int(args.max_pairs),
        "n_pairs": int(n_pairs),
        "image_crop_keys": [(int(row["image_index"]), int(row["crop_rank"])) for row in crops],
        "example_ids": [str(example_id) for example_id in example_ids],
        "t_max": int(config.t_max),
        "seed": int(config.seed),
        "population_size": int(config.population_size),
        "population_grid_position_mode": str(config.population_grid_position_mode),
        "population_grid_stride": int(config.population_grid_stride),
        "performance_metric": str(config.performance_metric),
        "ccmax_threshold": float(config.ccmax_threshold),
        "deduplicate_units": bool(config.deduplicate_units),
    }


def load_cache(path: Path, expected_identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if "cache_identity_json" not in data:
                return None
            observed = str(np.asarray(data["cache_identity_json"]).ravel()[0])
            if observed != identity_text(expected_identity):
                return None
            return {
                "unit_bits_per_movie": np.asarray(data["unit_bits_per_movie"], dtype=np.float32),
                "unit_expected_spikes_per_movie": np.asarray(data["unit_expected_spikes_per_movie"], dtype=np.float32),
                "unit_mean_rate_per_movie": np.asarray(data["unit_mean_rate_per_movie"], dtype=np.float32),
                "population_bits_per_movie": np.asarray(data["population_bits_per_movie"], dtype=np.float32),
                "mean_rate_map": np.asarray(data["mean_rate_map"], dtype=np.float32),
                "scale_values": np.asarray(data["scale_values"], dtype=np.float32),
                "movie_example_id": np.asarray(data["movie_example_id"]).astype(str),
                "movie_kind": np.asarray(data["movie_kind"]).astype(str),
                "movie_image_index": np.asarray(data["movie_image_index"], dtype=np.int32),
                "movie_crop_rank": np.asarray(data["movie_crop_rank"], dtype=np.int32),
            }
    except Exception:
        return None


def compute_scale_line(
    *,
    args: argparse.Namespace,
    config: PipelineConfig,
    scales: list[float],
    crops: list[dict[str, Any]],
    examples: list[Any],
    model: Any,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(config.seed))
    population, _population_rows = build_analysis_population(
        model,
        N=int(config.population_size),
        rng=rng,
        selection=str(config.population_selection),
        performance_metric=str(config.performance_metric),
        min_performance_score=config.min_performance_score,
        ccmax_threshold=float(config.ccmax_threshold),
        grid_position_mode=str(config.population_grid_position_mode),
        grid_stride=int(config.population_grid_stride),
        deduplicate_units=bool(config.deduplicate_units),
        dedupe_correlation_threshold=float(config.dedupe_correlation_threshold),
        dedupe_candidate_multiplier=float(config.dedupe_candidate_multiplier),
        dedupe_battery_frames=int(config.dedupe_battery_frames),
        dedupe_battery_seed=int(config.seed),
        dedupe_batch_size=int(config.batch_size),
    )

    image_indices = tuple(sorted({int(row["image_index"]) for row in crops}))
    loaded = load_natural_images(max(image_indices) + 1, indices=image_indices)
    image_by_index = {int(spec.image_index): image for spec, image in loaded}

    pair_specs: list[tuple[Any, dict[str, Any]]] = []
    for crop in crops:
        for example in examples:
            pair_specs.append((example, crop))
    if int(args.max_pairs) > 0:
        pair_specs = pair_specs[: int(args.max_pairs)]
    if not pair_specs:
        raise ValueError("No image/trace/crop pairs selected.")

    unit_bits_by_scale: list[list[np.ndarray]] = [[] for _ in scales]
    unit_spikes_by_scale: list[list[np.ndarray]] = [[] for _ in scales]
    unit_rates_by_scale: list[list[np.ndarray]] = [[] for _ in scales]
    population_by_scale: list[list[float]] = [[] for _ in scales]
    map_sum_by_scale: list[np.ndarray | None] = [None for _ in scales]

    movie_example_id: list[str] = []
    movie_kind: list[str] = []
    movie_image_index: list[int] = []
    movie_crop_rank: list[int] = []
    total = len(pair_specs) * len(scales)
    done = 0
    for pair_idx, (example, crop) in enumerate(pair_specs):
        image_index = int(crop["image_index"])
        crop_rank = int(crop["crop_rank"])
        image = image_by_index[image_index]
        crop_offset = (float(crop["offset_x_px"]), float(crop["offset_y_px"]))
        movie_example_id.append(str(example.example_id))
        movie_kind.append(str(example.kind))
        movie_image_index.append(image_index)
        movie_crop_rank.append(crop_rank)
        for scale_idx, scale in enumerate(scales):
            done += 1
            print(
                f"[backimage-ssi-scale] {done}/{total} pair={pair_idx + 1}/{len(pair_specs)} "
                f"example={example.example_id} image={image_index} crop={crop_rank} scale={float(scale):g}",
                flush=True,
            )
            trace = scaled_trace(example.trace, float(scale), int(config.t_max))
            cubes = model_lag_cubes_from_image_trace(
                image,
                trace,
                t_max=int(config.t_max),
                crop_center_offset_px=crop_offset,
            )
            blocks, _current = block_endpoint_lag_cubes(cubes)
            rate_map = run_rate_map(model, population, blocks, batch_size=int(args.batch_size))
            ssi = unit_spatial_ssi_for_movie(rate_map)
            mean_map = np.mean(rate_map, axis=0).astype(np.float32)
            if map_sum_by_scale[scale_idx] is None:
                map_sum_by_scale[scale_idx] = np.zeros_like(mean_map, dtype=np.float64)
            map_sum_by_scale[scale_idx] += mean_map
            unit_bits_by_scale[scale_idx].append(np.asarray(ssi["unit_bits_per_spike"], dtype=np.float32))
            unit_spikes_by_scale[scale_idx].append(np.asarray(ssi["unit_expected_spikes"], dtype=np.float32))
            unit_rates_by_scale[scale_idx].append(np.asarray(ssi["unit_mean_rate"], dtype=np.float32))
            population_by_scale[scale_idx].append(float(ssi["population_bits_per_spike"]))

    unit_bits = np.stack([np.stack(rows, axis=0) for rows in unit_bits_by_scale], axis=0).astype(np.float32)
    unit_spikes = np.stack([np.stack(rows, axis=0) for rows in unit_spikes_by_scale], axis=0).astype(np.float32)
    unit_rates = np.stack([np.stack(rows, axis=0) for rows in unit_rates_by_scale], axis=0).astype(np.float32)
    population = np.stack([np.asarray(rows, dtype=np.float32) for rows in population_by_scale], axis=0)
    n_movies = max(len(pair_specs), 1)
    mean_maps = np.stack([(m / float(n_movies)).astype(np.float32) for m in map_sum_by_scale], axis=0)
    return {
        "unit_bits_per_movie": unit_bits,
        "unit_expected_spikes_per_movie": unit_spikes,
        "unit_mean_rate_per_movie": unit_rates,
        "population_bits_per_movie": population,
        "mean_rate_map": mean_maps,
        "scale_values": np.asarray(scales, dtype=np.float32),
        "movie_example_id": np.asarray(movie_example_id),
        "movie_kind": np.asarray(movie_kind),
        "movie_image_index": np.asarray(movie_image_index, dtype=np.int32),
        "movie_crop_rank": np.asarray(movie_crop_rank, dtype=np.int32),
    }


def save_cache(path: Path, stats: dict[str, Any], identity: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **stats,
        cache_identity_json=np.asarray([identity_text(identity)]),
    )


def mean_sem(arr: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(arr, dtype=np.float64)
    mean = np.nanmean(x, axis=axis)
    n = x.shape[axis]
    sem = np.nanstd(x, axis=axis, ddof=1) / max(math.sqrt(float(n)), 1.0) if n > 1 else np.zeros_like(mean)
    return mean, sem


def summarize(stats: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scales = np.asarray(stats["scale_values"], dtype=np.float64)
    baseline_idx = int(np.nanargmin(np.abs(scales - 0.0)))
    unit_bits = np.asarray(stats["unit_bits_per_movie"], dtype=np.float64)  # scale x movie x unit
    unit_spikes = np.asarray(stats["unit_expected_spikes_per_movie"], dtype=np.float64)
    unit_rates = np.asarray(stats["unit_mean_rate_per_movie"], dtype=np.float64)
    pop = np.asarray(stats["population_bits_per_movie"], dtype=np.float64)
    bits_mean, bits_sem = mean_sem(unit_bits, axis=1)
    rates_mean, rates_sem = mean_sem(unit_rates, axis=1)
    pop_mean, pop_sem = mean_sem(pop, axis=1)

    baseline_bits = bits_mean[baseline_idx]
    baseline_pop = float(pop_mean[baseline_idx])
    unit_ratio = (bits_mean + EPS) / (baseline_bits[None, :] + EPS)
    unit_log2_ratio = np.log2(unit_ratio)
    population_ratio = (pop_mean + EPS) / (baseline_pop + EPS)

    total_expected = np.sum(unit_spikes, axis=2)
    total_bits = np.sum(unit_spikes * unit_bits, axis=2)
    n_scales, _n_movies, n_units = unit_bits.shape
    loo_pop = np.zeros((n_units, n_scales), dtype=np.float64)
    for unit_idx in range(n_units):
        numer = total_bits - unit_spikes[:, :, unit_idx] * unit_bits[:, :, unit_idx]
        denom = np.maximum(total_expected - unit_spikes[:, :, unit_idx], EPS)
        loo_pop[unit_idx] = np.nanmean(numer / denom, axis=1)
    loo_ratio = (loo_pop + EPS) / (loo_pop[:, baseline_idx : baseline_idx + 1] + EPS)
    loo_delta = loo_ratio - population_ratio[None, :]
    loo_abs_delta = loo_pop - pop_mean[None, :]

    unit_rows: list[dict[str, Any]] = []
    for scale_idx, scale in enumerate(scales):
        for unit_idx in range(n_units):
            unit_rows.append(
                {
                    "scale": float(scale),
                    "unit_index": int(unit_idx),
                    "unit_label": f"u{int(unit_idx):03d}",
                    "unit_ssi_bits_per_spike_mean": float(bits_mean[scale_idx, unit_idx]),
                    "unit_ssi_bits_per_spike_sem": float(bits_sem[scale_idx, unit_idx]),
                    "unit_mean_rate_mean": float(rates_mean[scale_idx, unit_idx]),
                    "unit_mean_rate_sem": float(rates_sem[scale_idx, unit_idx]),
                    "unit_ssi_vs_0x": float(unit_ratio[scale_idx, unit_idx]),
                    "unit_log2_ssi_vs_0x": float(unit_log2_ratio[scale_idx, unit_idx]),
                    "population_ssi_bits_per_spike_mean": float(pop_mean[scale_idx]),
                    "population_ssi_bits_per_spike_sem": float(pop_sem[scale_idx]),
                    "population_ssi_vs_0x": float(population_ratio[scale_idx]),
                    "population_log2_ssi_vs_0x": float(np.log2(population_ratio[scale_idx])),
                    "baseline_unit_ssi_bits_per_spike_mean": float(baseline_bits[unit_idx]),
                    "baseline_population_ssi_bits_per_spike_mean": baseline_pop,
                }
            )

    scale_rows = [
        {
            "scale": float(scale),
            "population_ssi_bits_per_spike_mean": float(pop_mean[idx]),
            "population_ssi_bits_per_spike_sem": float(pop_sem[idx]),
            "population_ssi_vs_0x": float(population_ratio[idx]),
            "population_log2_ssi_vs_0x": float(np.log2(population_ratio[idx])),
            "n_movies": int(pop.shape[1]),
            "n_units": int(n_units),
        }
        for idx, scale in enumerate(scales)
    ]

    unit_max_abs_log2 = np.nanmax(np.abs(unit_log2_ratio), axis=0)
    top_rows = []
    for unit_idx in range(n_units):
        top_rows.append(
            {
                "unit_index": int(unit_idx),
                "unit_label": f"u{int(unit_idx):03d}",
                "max_abs_log2_unit_ssi_vs_0x": float(unit_max_abs_log2[unit_idx]),
                "max_abs_leave_one_out_population_ratio_delta": float(np.nanmax(np.abs(loo_delta[unit_idx]))),
                "max_abs_leave_one_out_population_ssi_delta": float(np.nanmax(np.abs(loo_abs_delta[unit_idx]))),
                "max_abs_unit_ssi_delta_vs_0x": float(np.nanmax(np.abs(bits_mean[:, unit_idx] - baseline_bits[unit_idx]))),
                "baseline_unit_ssi_bits_per_spike_mean": float(baseline_bits[unit_idx]),
                "baseline_unit_mean_rate_mean": float(rates_mean[baseline_idx, unit_idx]),
            }
        )
    top_rows.sort(
        key=lambda row: (
            float(row["max_abs_leave_one_out_population_ssi_delta"]),
            float(row["max_abs_unit_ssi_delta_vs_0x"]),
        ),
        reverse=True,
    )
    diagnostics = {
        "scales": scales,
        "baseline_idx": baseline_idx,
        "unit_bits_mean": bits_mean,
        "unit_bits_sem": bits_sem,
        "unit_log2_ratio": unit_log2_ratio,
        "unit_ratio": unit_ratio,
        "population_bits_mean": pop_mean,
        "population_bits_sem": pop_sem,
        "population_ratio": population_ratio,
        "population_log2_ratio": np.log2(population_ratio),
        "leave_one_out_population_bits": loo_pop,
        "leave_one_out_ratio": loo_ratio,
        "leave_one_out_delta": loo_delta,
        "leave_one_out_absolute_delta": loo_abs_delta,
        "top_rows": top_rows,
    }
    return unit_rows, scale_rows, diagnostics


def highlighted_unit_color_map(highlighted_units: list[int]) -> dict[int, Any]:
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(highlighted_units), 1)))
    return {int(unit_index): colors[pos] for pos, unit_index in enumerate(highlighted_units)}


def image_scale(images: list[np.ndarray], vmin_percentile: float, vmax_percentile: float) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(img, dtype=np.float32).ravel() for img in images])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, float(vmin_percentile)))
    vmax = float(np.nanpercentile(finite, float(vmax_percentile)))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def scale_label(scale: float) -> str:
    if np.isclose(float(scale), round(float(scale))):
        return f"{int(round(float(scale)))}x"
    return f"{float(scale):g}x"


def draw_lines_on_axes(
    axes: np.ndarray,
    *,
    diagnostics: dict[str, Any],
    highlighted_units: list[int],
    color_by_unit: dict[int, Any],
    highlight_note: str,
    line_y_mode: str,
) -> None:
    x = np.asarray(diagnostics["scales"], dtype=np.float64)
    if str(line_y_mode) == "log2_ratio":
        unit_y = np.asarray(diagnostics["unit_log2_ratio"], dtype=np.float64)
        population_y = np.asarray(diagnostics["population_log2_ratio"], dtype=np.float64)
        ylabel = "log2 SSI / own 0x SSI"
        zero_line = 0.0
    else:
        unit_y = np.asarray(diagnostics["unit_bits_mean"], dtype=np.float64)
        population_y = np.asarray(diagnostics["population_bits_mean"], dtype=np.float64)
        ylabel = "SSI (bits/spike)"
        zero_line = None

    ax = axes[0]
    for unit_idx in range(unit_y.shape[1]):
        ax.plot(x, unit_y[:, unit_idx], color="#a8a8a8", linewidth=0.65, alpha=0.35, zorder=1)
    for unit_idx in highlighted_units:
        ax.plot(
            x,
            unit_y[:, int(unit_idx)],
            marker="o",
            linewidth=1.35,
            markersize=3.2,
            color=color_by_unit[int(unit_idx)],
            label=f"u{int(unit_idx):03d}",
            zorder=3,
        )
    ax.plot(x, population_y, color="black", marker="o", linewidth=2.0, markersize=4.0, label="population", zorder=4)
    if zero_line is not None:
        ax.axhline(zero_line, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax.set_xlabel("real-trace motion scale around trial mean")
    ax.set_ylabel(ylabel)
    ax.set_title(f"All units; {highlight_note}")
    ax.grid(True, axis="y", color="#e4e4e4", linewidth=0.7)
    ax.legend(frameon=False, fontsize=6.0, ncols=2, loc="best")

    ax = axes[1]
    for unit_idx in highlighted_units:
        ax.plot(
            x,
            unit_y[:, int(unit_idx)],
            marker="o",
            linewidth=1.5,
            markersize=3.4,
            color=color_by_unit[int(unit_idx)],
            label=f"u{int(unit_idx):03d}",
        )
    ax.plot(x, population_y, color="black", marker="o", linewidth=2.1, markersize=4.0, label="population")
    if zero_line is not None:
        ax.axhline(zero_line, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax.set_xlabel("real-trace motion scale around trial mean")
    ax.set_ylabel(ylabel)
    ax.set_title("Highlighted units only")
    ax.grid(True, axis="y", color="#e4e4e4", linewidth=0.7)
    ax.legend(frameon=False, fontsize=6.2, ncols=2, loc="best")


def draw_summary_figure(
    *,
    stats: dict[str, Any],
    diagnostics: dict[str, Any],
    highlighted_units: list[int],
    path: Path,
    dpi: int,
    map_vmin_percentile: float,
    map_vmax_percentile: float,
    title_suffix: str,
    line_y_mode: str,
) -> None:
    scales = np.asarray(stats["scale_values"], dtype=np.float64)
    mean_maps = np.asarray(stats["mean_rate_map"], dtype=np.float32)  # scale x unit x h x w
    color_by_unit = highlighted_unit_color_map(highlighted_units)
    n_units = len(highlighted_units)
    n_cols = len(scales)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_width = max(12.4, 1.12 * (n_cols + 1))
    fig_height = 5.25 + 0.72 * max(n_units, 1)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=int(dpi))
    outer = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[3.8, max(0.62 * max(n_units, 1), 1.4)],
        hspace=0.16,
    )
    top_grid = outer[0].subgridspec(1, 2, wspace=0.12)
    line_axes = np.asarray([fig.add_subplot(top_grid[0, 0]), fig.add_subplot(top_grid[0, 1])])
    draw_lines_on_axes(
        line_axes,
        diagnostics=diagnostics,
        highlighted_units=highlighted_units,
        color_by_unit=color_by_unit,
        highlight_note="largest leave-one-out influences highlighted",
        line_y_mode=str(line_y_mode),
    )

    map_grid = outer[1].subgridspec(
        nrows=n_units + 1,
        ncols=n_cols + 1,
        height_ratios=[0.34, *([1.0] * n_units)],
        width_ratios=[0.82, *([1.0] * n_cols)],
        hspace=0.045,
        wspace=0.035,
    )
    label_header_ax = fig.add_subplot(map_grid[0, 0])
    label_header_ax.axis("off")
    label_header_ax.text(0.98, 0.42, "unit", ha="right", va="center", fontsize=6.7, color="#555555")
    for col_idx, scale in enumerate(scales):
        ax = fig.add_subplot(map_grid[0, col_idx + 1])
        ax.axis("off")
        ax.text(
            0.5,
            0.42,
            scale_label(float(scale)),
            ha="center",
            va="center",
            fontsize=6.7,
            fontweight="bold" if np.isclose(float(scale), 1.0) else "normal",
            color="#333333",
        )

    for unit_pos, unit_idx in enumerate(highlighted_units):
        unit_color = color_by_unit[int(unit_idx)]
        images = [mean_maps[scale_idx, int(unit_idx)] for scale_idx in range(n_cols)]
        vmin, vmax = image_scale(images, float(map_vmin_percentile), float(map_vmax_percentile))
        label_ax = fig.add_subplot(map_grid[unit_pos + 1, 0])
        label_ax.axis("off")
        label_ax.text(
            0.98,
            0.5,
            f"u{int(unit_idx):03d}",
            ha="right",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color=unit_color,
            transform=label_ax.transAxes,
        )
        label_ax.plot(
            [0.18, 0.92],
            [0.2, 0.2],
            color=unit_color,
            linewidth=2.4,
            solid_capstyle="round",
            transform=label_ax.transAxes,
        )
        for col_idx, image in enumerate(images):
            ax = fig.add_subplot(map_grid[unit_pos + 1, col_idx + 1])
            ax.imshow(image, origin="lower", cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine_name, spine in ax.spines.items():
                spine.set_linewidth(0.55 if spine_name != "left" else 1.15)
                spine.set_color(unit_color if spine_name == "left" else "#686868")

    fig.suptitle(
        "BackImage scaled-real unit SSI\n"
        f"{_line_mode_subtitle(str(line_y_mode))}; activation maps below use monotonic grayscale per unit row"
        "\n0x is trial-mean stabilized"
        f"{title_suffix}",
        fontsize=10.9,
        y=0.985,
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.035, top=0.865)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_individual_unit_sheets(
    *,
    stats: dict[str, Any],
    unit_rows: list[dict[str, Any]],
    highlighted_units: list[int],
    out_dir: Path,
    dpi: int,
    map_vmin_percentile: float,
    map_vmax_percentile: float,
    line_y_mode: str,
) -> None:
    by_unit_scale = {
        (int(row["unit_index"]), float(row["scale"])): row
        for row in unit_rows
    }
    scales = np.asarray(stats["scale_values"], dtype=np.float64)
    mean_maps = np.asarray(stats["mean_rate_map"], dtype=np.float32)
    sheet_dir = out_dir / "highlighted_unit_activation_maps" / "unit_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for unit_idx in highlighted_units:
        images = [mean_maps[scale_idx, int(unit_idx)] for scale_idx in range(len(scales))]
        vmin, vmax = image_scale(images, float(map_vmin_percentile), float(map_vmax_percentile))
        fig, axes = plt.subplots(1, len(scales), figsize=(1.65 * len(scales), 2.25), dpi=int(dpi), constrained_layout=True)
        axes_arr = np.asarray(axes).reshape(-1)
        last_im = None
        for scale_idx, scale in enumerate(scales):
            row = by_unit_scale.get((int(unit_idx), float(scale)), {})
            title = f"{scale_label(float(scale))}\nSSI {float(row.get('unit_ssi_bits_per_spike_mean', np.nan)):.4f}"
            if str(line_y_mode) == "log2_ratio":
                title += f"\n{float(row.get('unit_ssi_vs_0x', np.nan)):.2f}x"
            ax = axes_arr[scale_idx]
            last_im = ax.imshow(images[scale_idx], origin="lower", cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=6.0, pad=3)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.45)
                spine.set_color("#777777")
        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes_arr.tolist(), fraction=0.018, pad=0.01)
            cbar.ax.tick_params(labelsize=6.0, length=2)
            cbar.set_label("mean activation", fontsize=6.5)
        fig.suptitle(f"BackImage u{int(unit_idx):03d}: scaled-real mean activation maps", fontsize=10.0, y=1.08)
        fig.savefig(sheet_dir / f"backimage_scaled_real_unit_{int(unit_idx):03d}_activation_maps.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)


def _line_mode_subtitle(line_y_mode: str) -> str:
    if str(line_y_mode) == "log2_ratio":
        return "line panels show log2(SSI / own 0x SSI)"
    return "line panels show absolute SSI bits/spike, no per-unit static division"


def figure_stem(line_y_mode: str) -> str:
    if str(line_y_mode) == "log2_ratio":
        return "backimage_scaled_real_unit_ssi_log2_ratio_with_activation_maps"
    return "backimage_scaled_real_unit_ssi_absolute_with_activation_maps"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = parse_scales(str(args.scales))
    config = load_pipeline_config(run_dir)
    crops = crop_rows_from_metadata(run_dir, max_images=int(args.max_images))
    example_ids = selected_example_ids_from_metadata(run_dir)
    pair_count = len(crops) * len(example_ids)
    if int(args.max_pairs) > 0:
        pair_count = min(pair_count, int(args.max_pairs))
    identity = build_identity(
        args=args,
        config=config,
        scales=scales,
        crops=crops,
        example_ids=example_ids,
        n_pairs=pair_count,
    )
    cpath = cache_path(out_dir)
    stats = None if bool(args.force) else load_cache(cpath, identity)
    if stats is None:
        # Load the model only when a cache miss requires recomputing full
        # spatial rate maps.
        model, _model_info, _device = load_digital_twin()
        examples = selected_examples(config, run_dir, model, out_dir / "_scratch_reproduce_selection")
        stats = compute_scale_line(args=args, config=config, scales=scales, crops=crops, examples=examples, model=model)
        save_cache(cpath, stats, identity)
        print(f"[backimage-ssi-scale] saved cache: {cpath}", flush=True)
    else:
        print(f"[backimage-ssi-scale] loaded cache: {cpath}", flush=True)

    unit_rows, scale_rows, diagnostics = summarize(stats)
    top_units = [int(row["unit_index"]) for row in diagnostics["top_rows"][: max(1, int(args.top_units))]]
    write_csv_rows(out_dir / "backimage_scaled_real_unit_ssi_table.csv", unit_rows)
    write_csv_rows(out_dir / "backimage_scaled_real_scale_summary.csv", scale_rows)
    write_csv_rows(out_dir / "backimage_scaled_real_highlighted_units.csv", diagnostics["top_rows"])
    write_json(
        out_dir / "backimage_scaled_real_unit_ssi_manifest.json",
        {
            **identity,
            "line_y_mode": str(args.line_y_mode),
            "cache_path": cpath,
            "unit_table": out_dir / "backimage_scaled_real_unit_ssi_table.csv",
            "scale_summary": out_dir / "backimage_scaled_real_scale_summary.csv",
            "highlighted_units": top_units,
        },
    )
    title_suffix = f"\nsource movies: n={int(stats['population_bits_per_movie'].shape[1])}; units={int(stats['unit_bits_per_movie'].shape[2])}"
    stem = figure_stem(str(args.line_y_mode))
    figure_path = out_dir / f"{stem}.png"
    draw_summary_figure(
        stats=stats,
        diagnostics=diagnostics,
        highlighted_units=top_units,
        path=figure_path,
        dpi=int(args.dpi),
        map_vmin_percentile=float(args.map_vmin_percentile),
        map_vmax_percentile=float(args.map_vmax_percentile),
        title_suffix=title_suffix,
        line_y_mode=str(args.line_y_mode),
    )
    draw_summary_figure(
        stats=stats,
        diagnostics=diagnostics,
        highlighted_units=top_units,
        path=out_dir / f"{stem}.pdf",
        dpi=int(args.dpi),
        map_vmin_percentile=float(args.map_vmin_percentile),
        map_vmax_percentile=float(args.map_vmax_percentile),
        title_suffix=title_suffix,
        line_y_mode=str(args.line_y_mode),
    )
    if not bool(args.skip_individual_unit_sheets):
        draw_individual_unit_sheets(
            stats=stats,
            unit_rows=unit_rows,
            highlighted_units=top_units,
            out_dir=out_dir,
            dpi=int(args.dpi),
            map_vmin_percentile=float(args.map_vmin_percentile),
            map_vmax_percentile=float(args.map_vmax_percentile),
            line_y_mode=str(args.line_y_mode),
        )
    print(f"[backimage-ssi-scale] wrote {figure_path}", flush=True)


if __name__ == "__main__":
    main()
