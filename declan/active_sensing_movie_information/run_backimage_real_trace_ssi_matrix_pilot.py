#!/usr/bin/env python3
"""Benchmark and pilot a BackImage real-trace x image RR100 SSI matrix.

This runner is deliberately matrix-first.  It samples natural-image patches and
native 40-sample real fixation traces independently, scores every image x trace
movie with the canonical twin, applies the RR100 population view, and writes
flat matrices plus feature tables for downstream conditioning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    RR100_MOVIE_MEDOID_VERSION,
    build_native_snippet_trace_bank,
    trace_bank_metadata_row,
    trace_bank_metric_summary_rows,
    unit_spatial_ssi_for_movie,
)
from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import (
    _session_dataset_cache,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
    _align_response_to_trace,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import (
    _extract_patch,
)
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _standardize_uint_like,
    _trace_xy_to_twin_helper_order,
)
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


DEFAULT_SOURCE_CSV = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
DEFAULT_UNIT_TUNING_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
    "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)
DEFAULT_OUT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_pilot_benchmark_v1"
)
IMAGE_FEATURE_COLUMNS = [
    "image_patch_rms_contrast",
    "image_patch_std",
    "image_gradient_energy",
    "image_oriented_gradient_energy",
    "image_multi_orientation_energy",
    "image_edge_density",
    "image_orientation_coherence",
    "image_gradient_axis_deg",
    "image_edge_axis_deg",
    "image_spectrum_anisotropy",
    "image_edge_spectrum_contour_axis_agreement",
    "image_oriented_8plus_power_proxy",
    "image_spectrum_orientation_deg",
    "image_high_freq_power_fraction",
    "image_power_8plus_cpd_fraction",
    "image_contour_reliable",
    "image_contour_strong",
]
TRACE_FEATURE_COLUMNS = [
    "rendered_path_length_arcmin",
    "rendered_path_speed_arcmin_s",
    "rendered_rms_radius_arcmin",
    "rendered_bcea68_arcmin2",
    "rendered_cov_anisotropy",
    "rendered_cov_axis_ratio",
    "rendered_cov_orientation_deg",
    "rendered_speed_p95_arcmin_s",
    "rendered_diffusion_constant_arcmin2_s",
    "rendered_position_autocorr_lag1",
    "rendered_velocity_autocorr_lag1",
    "rendered_n_microsaccade_events",
    "rendered_fraction_microsaccade_samples",
    "rendered_peak_microsaccade_speed_dps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--unit-tuning-csv", type=Path, default=DEFAULT_UNIT_TUNING_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--n-traces", type=int, default=100)
    parser.add_argument("--benchmark-n-images", type=int, default=2)
    parser.add_argument("--benchmark-n-traces", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--image-contrast-quantile", type=float, default=0.75)
    parser.add_argument("--image-min-orientation-coherence", type=float, default=0.0)
    parser.add_argument("--image-min-drift-anisotropy", type=float, default=0.0)
    parser.add_argument("--min-strong-contour-images", type=int, default=0)
    parser.add_argument("--strong-contour-orientation-coherence-min", type=float, default=0.5)
    parser.add_argument("--max-trace-path-length-arcmin", type=float, default=350.0)
    parser.add_argument("--trace-scale-metric", type=str, default="rendered_path_length_arcmin")
    parser.add_argument("--trace-sampling", choices=("quantile", "random"), default="quantile")
    parser.add_argument(
        "--min-microsaccade-traces",
        type=int,
        default=0,
        help="Require at least this many selected traces with one or more microsaccade events.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--benchmark-frame-batch-sizes", type=str, default="8,32,64")
    parser.add_argument("--benchmark-trace-batch-sizes", type=str, default="1,8,16")
    parser.add_argument("--pilot-frame-batch-size", type=int, default=0)
    parser.add_argument("--pilot-trace-batch-size", type=int, default=0)
    parser.add_argument(
        "--image-shard-start",
        type=int,
        default=0,
        help="Inclusive selected-image index to start scoring. 0 starts at the first selected image.",
    )
    parser.add_argument(
        "--image-shard-stop",
        type=int,
        default=0,
        help="Exclusive selected-image index to stop scoring. 0 scores through --n-images.",
    )
    parser.add_argument("--skip-benchmark", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--benchmark-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[backimage-real-trace-ssi-matrix] {message}", flush=True)


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_source_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    if "source_row" not in rows.columns:
        rows = rows.copy()
        rows["source_row"] = np.arange(rows.shape[0], dtype=int)
    return rows


def circular_axis_delta_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    return 0.5 * np.degrees(np.angle(np.exp(2j * np.radians(np.asarray(a_deg) - np.asarray(b_deg)))))


def add_derived_image_features(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if {"image_gradient_energy", "image_orientation_coherence"}.issubset(out.columns):
        gradient = pd.to_numeric(out["image_gradient_energy"], errors="coerce").astype(float)
        coherence = pd.to_numeric(out["image_orientation_coherence"], errors="coerce").astype(float)
        out["image_oriented_gradient_energy"] = gradient * np.maximum(coherence, 0.0)
        out["image_multi_orientation_energy"] = gradient * np.maximum(1.0 - coherence, 0.0)
    required = {
        "image_power_8plus_cpd_fraction",
        "image_patch_std",
        "image_spectrum_anisotropy",
        "image_spectrum_orientation_deg",
        "image_edge_axis_deg",
    }
    if required.issubset(out.columns):
        abs8 = (
            pd.to_numeric(out["image_power_8plus_cpd_fraction"], errors="coerce").astype(float)
            * pd.to_numeric(out["image_patch_std"], errors="coerce").astype(float)
            * pd.to_numeric(out["image_patch_std"], errors="coerce").astype(float)
        )
        spectrum_contour_axis = pd.to_numeric(out["image_spectrum_orientation_deg"], errors="coerce").to_numpy(float) + 90.0
        edge_axis = pd.to_numeric(out["image_edge_axis_deg"], errors="coerce").to_numpy(float)
        agreement = np.cos(2.0 * np.radians(circular_axis_delta_deg(edge_axis, spectrum_contour_axis)))
        out["image_edge_spectrum_contour_axis_agreement"] = agreement
        out["image_oriented_8plus_power_proxy"] = (
            abs8
            * np.maximum(pd.to_numeric(out["image_spectrum_anisotropy"], errors="coerce").astype(float), 0.0)
            * np.maximum(agreement, 0.0)
        )
    return out


def image_candidate_rows(
    rows: pd.DataFrame,
    *,
    contrast_quantile: float,
    n_timepoints: int,
    min_orientation_coherence: float,
    min_drift_anisotropy: float,
) -> pd.DataFrame:
    work = add_derived_image_features(rows)
    if "image_feature_ok" in work.columns:
        work = work[work["image_feature_ok"].astype(bool)].copy()
    if "n_samples" in work.columns:
        work = work[pd.to_numeric(work["n_samples"], errors="coerce") >= int(n_timepoints)].copy()
    if "image_patch_fraction_inside_image" in work.columns:
        work = work[pd.to_numeric(work["image_patch_fraction_inside_image"], errors="coerce") >= 0.99].copy()
    if "image_patch_rms_contrast" in work.columns and work.shape[0]:
        contrast = pd.to_numeric(work["image_patch_rms_contrast"], errors="coerce")
        threshold = float(contrast.quantile(float(contrast_quantile)))
        work = work[contrast >= threshold].copy()
    if float(min_orientation_coherence) > 0.0:
        if "image_orientation_coherence" not in work.columns:
            raise ValueError("--image-min-orientation-coherence requires image_orientation_coherence.")
        coherence = pd.to_numeric(work["image_orientation_coherence"], errors="coerce")
        work = work[coherence >= float(min_orientation_coherence)].copy()
    if float(min_drift_anisotropy) > 0.0:
        if "anisotropy" not in work.columns:
            raise ValueError("--image-min-drift-anisotropy requires anisotropy.")
        anisotropy = pd.to_numeric(work["anisotropy"], errors="coerce")
        work = work[anisotropy >= float(min_drift_anisotropy)].copy()
    return work.drop_duplicates("source_row").reset_index(drop=True)


def sample_rows_random(work: pd.DataFrame, n_rows: int, *, rng: np.random.Generator) -> pd.DataFrame:
    if work.shape[0] < int(n_rows):
        raise ValueError(f"Requested {n_rows} rows, but only {work.shape[0]} are available.")
    indices = rng.choice(work.index.to_numpy(), size=int(n_rows), replace=False)
    return work.loc[indices].copy().reset_index(drop=True)


def sample_image_rows(
    work: pd.DataFrame,
    n_rows: int,
    *,
    rng: np.random.Generator,
    min_strong_contour_images: int,
    strong_contour_orientation_coherence_min: float,
) -> pd.DataFrame:
    min_strong_contour_images = int(min_strong_contour_images)
    if min_strong_contour_images <= 0:
        return sample_rows_random(work, n_rows, rng=rng)
    if min_strong_contour_images > int(n_rows):
        raise ValueError("--min-strong-contour-images cannot exceed --n-images.")
    if "image_orientation_coherence" not in work.columns:
        raise ValueError("--min-strong-contour-images requires image_orientation_coherence.")
    coherence = pd.to_numeric(work["image_orientation_coherence"], errors="coerce")
    strong = work[coherence >= float(strong_contour_orientation_coherence_min)].copy()
    strong_selected = sample_rows_random(strong, min_strong_contour_images, rng=rng)
    selected_source_rows = set(strong_selected["source_row"].astype(int).to_list())
    remainder_pool = work[~work["source_row"].astype(int).isin(selected_source_rows)].copy()
    remainder = sample_rows_random(remainder_pool, int(n_rows) - min_strong_contour_images, rng=rng)
    selected = pd.concat([strong_selected, remainder], ignore_index=True)
    return selected.iloc[rng.permutation(np.arange(selected.shape[0]))].reset_index(drop=True)


def annotate_selected_image_flags(images: pd.DataFrame, *, reliable_min: float, strong_min: float) -> pd.DataFrame:
    out = images.copy()
    if "image_orientation_coherence" in out.columns:
        coherence = pd.to_numeric(out["image_orientation_coherence"], errors="coerce")
        out["image_contour_reliable"] = coherence >= float(reliable_min)
        out["image_contour_strong"] = coherence >= float(strong_min)
    return out


def microsaccade_event_count(item: dict[str, Any]) -> int:
    for key in ("rendered_n_microsaccade_events", "n_microsaccade_events", "source_n_microsaccade_events"):
        if key not in item:
            continue
        value = pd.to_numeric(pd.Series([item[key]]), errors="coerce").iloc[0]
        if pd.notna(value):
            return max(0, int(value))
    return 0


def _sample_trace_items_unstratified(
    items: list[dict[str, Any]],
    n_traces: int,
    *,
    metric: str,
    sampling: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if len(items) < int(n_traces):
        raise ValueError(f"Requested {n_traces} traces, but only {len(items)} are available.")
    if str(sampling) == "random":
        indices = rng.choice(np.arange(len(items)), size=int(n_traces), replace=False)
        return [items[int(idx)] for idx in indices]
    values = np.asarray([float(item.get(metric, np.nan)) for item in items], dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size < int(n_traces):
        raise ValueError(f"Metric {metric!r} has only {finite.size} finite values for quantile sampling.")
    order = finite[np.argsort(values[finite], kind="mergesort")]
    chunks = np.array_split(order, int(n_traces))
    chosen: list[int] = []
    for chunk in chunks:
        chosen.append(int(rng.choice(chunk)))
    return [items[idx] for idx in chosen]


def sample_trace_items(
    items: list[dict[str, Any]],
    n_traces: int,
    *,
    metric: str,
    sampling: str,
    rng: np.random.Generator,
    min_microsaccade_traces: int = 0,
) -> list[dict[str, Any]]:
    min_microsaccade_traces = int(min_microsaccade_traces)
    if min_microsaccade_traces <= 0:
        return _sample_trace_items_unstratified(
            items,
            n_traces,
            metric=metric,
            sampling=sampling,
            rng=rng,
        )
    if min_microsaccade_traces > int(n_traces):
        raise ValueError("--min-microsaccade-traces cannot exceed --n-traces.")
    microsaccade_items = [item for item in items if microsaccade_event_count(item) > 0]
    drift_items = [item for item in items if microsaccade_event_count(item) <= 0]
    n_drift = int(n_traces) - min_microsaccade_traces
    selected = _sample_trace_items_unstratified(
        microsaccade_items,
        min_microsaccade_traces,
        metric=metric,
        sampling=sampling,
        rng=rng,
    ) + _sample_trace_items_unstratified(
        drift_items,
        n_drift,
        metric=metric,
        sampling=sampling,
        rng=rng,
    )
    return sorted(selected, key=lambda item: float(item.get(metric, np.inf)))


def build_trace_bank(
    rows: pd.DataFrame,
    *,
    n_timepoints: int,
    bin_seconds: float,
    max_path_arcmin: float,
) -> list[dict[str, Any]]:
    trace_rows = rows.drop_duplicates("source_row").copy()
    trace_rows = trace_rows[pd.to_numeric(trace_rows["n_samples"], errors="coerce") >= int(n_timepoints)].copy()
    eyepos_by_session = _session_dataset_cache(trace_rows["session"].astype(str).dropna().unique().tolist())
    bank, _meta = build_native_snippet_trace_bank(
        trace_rows,
        eyepos_by_session,
        int(n_timepoints),
        dt=float(bin_seconds),
        microsaccade_speed_threshold_dps=None,
        microsaccade_threshold_z=6.0,
        microsaccade_pad_frames=1,
    )
    out: list[dict[str, Any]] = []
    for item in bank:
        item["observed_rms_arcmin"] = float(item["observed_rms_deg"]) * 60.0
        item["path_length_arcmin"] = float(item["path_length_deg"]) * 60.0
        path_arcmin = float(item.get("rendered_path_length_arcmin", item["path_length_arcmin"]))
        if path_arcmin <= float(max_path_arcmin):
            out.append(item)
    return out


def rate_maps_for_traces(
    scorer: CanonicalTwinScorer,
    patch: np.ndarray,
    traces: list[np.ndarray],
    *,
    trace_batch_size: int,
) -> list[np.ndarray]:
    if not traces:
        return []
    image = _standardize_uint_like(patch)
    trace_batch_size = max(1, int(trace_batch_size))
    out: list[np.ndarray] = []
    for start in range(0, len(traces), trace_batch_size):
        trace_chunk = traces[start : start + trace_batch_size]
        stims = []
        lengths = []
        for trace in trace_chunk:
            arr = np.asarray(trace, dtype=np.float32)
            full_stack = np.broadcast_to(
                image[None, :, :],
                (arr.shape[0] + scorer.common.N_LAGS + 1, *image.shape),
            ).copy()
            eye = scorer.torch.from_numpy(_trace_xy_to_twin_helper_order(arr))
            stim = scorer.common.make_counterfactual_stim(
                full_stack,
                eye,
                ppd=scorer.common.PPD,
                scale_factor=1.0,
                n_lags=scorer.common.N_LAGS,
                out_size=scorer.common.OUT_SIZE,
            )
            stims.append((stim - 127.0) / 255.0)
            lengths.append(int(stim.shape[0]))
        rate_map = scorer._compute_rate_map_batched(scorer.torch.cat(stims, dim=0))
        offset = 0
        for length in lengths:
            out.append(rate_map[offset : offset + length].detach().cpu().numpy().astype(np.float32, copy=False))
            offset += length
        del stims, rate_map
    return out


def score_traces_for_patch(
    scorer: CanonicalTwinScorer,
    population_view: Any,
    patch: np.ndarray,
    traces: list[np.ndarray],
    *,
    trace_batch_size: int,
    frame_batch_size: int,
    n_timepoints: int,
    bin_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score traces without materializing full spatial rate maps on CPU."""
    if not traces:
        n_units = int(population_view.n_units)
        return (
            np.zeros((0, n_units), dtype=np.float32),
            np.zeros((0, n_units), dtype=np.float32),
            np.zeros((0, n_units), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    scorer.batch_size = int(frame_batch_size)
    device = next(scorer.ctx.model.model.parameters()).device
    scorer.ctx.model.model.eval()
    scorer.ctx.readout.eval()
    image = _standardize_uint_like(patch)
    n_traces = len(traces)
    n_units = int(population_view.n_units)
    unit_expected = np.zeros((n_traces, n_units), dtype=np.float64)
    unit_numer = np.zeros((n_traces, n_units), dtype=np.float64)
    unit_rate_sum = np.zeros((n_traces, n_units), dtype=np.float64)
    unit_frame_count = np.zeros((n_traces,), dtype=np.int64)
    trace_batch_size = max(1, int(trace_batch_size))
    n_timepoints = int(n_timepoints)

    with scorer.torch.no_grad():
        for trace_start in range(0, n_traces, trace_batch_size):
            trace_chunk = traces[trace_start : trace_start + trace_batch_size]
            stims = []
            frame_to_trace: list[int] = []
            for local_idx, trace in enumerate(trace_chunk):
                arr = np.asarray(trace, dtype=np.float32)
                full_stack = np.broadcast_to(
                    image[None, :, :],
                    (arr.shape[0] + scorer.common.N_LAGS + 1, *image.shape),
                ).copy()
                eye = scorer.torch.from_numpy(_trace_xy_to_twin_helper_order(arr))
                stim = scorer.common.make_counterfactual_stim(
                    full_stack,
                    eye,
                    ppd=scorer.common.PPD,
                    scale_factor=1.0,
                    n_lags=scorer.common.N_LAGS,
                    out_size=scorer.common.OUT_SIZE,
                )
                length = int(stim.shape[0])
                if length == n_timepoints:
                    trace_ids = [trace_start + local_idx] * length
                elif length == n_timepoints + 1:
                    trace_ids = [-1] + [trace_start + local_idx] * n_timepoints
                else:
                    raise ValueError(
                        f"Twin response has {length} frames for a {n_timepoints}-sample trace; expected T or T+1."
                    )
                frame_to_trace.extend(trace_ids)
                stims.append((stim - 127.0) / 255.0)
            stim_all = scorer.torch.cat(stims, dim=0)
            frame_to_trace_arr = np.asarray(frame_to_trace, dtype=np.int64)
            for t_start in range(0, int(stim_all.shape[0]), int(frame_batch_size)):
                t_end = min(t_start + int(frame_batch_size), int(stim_all.shape[0]))
                x = stim_all[t_start:t_end].to(device)
                full_map = scorer.compute_rate_map(scorer.ctx.model, scorer.ctx.readout, x)
                rr100_map = apply_population_view(full_map, population_view).clamp_min(0.0).to(scorer.torch.float64)
                flat = rr100_map.reshape(rr100_map.shape[0], rr100_map.shape[1], -1)
                rbar = flat.mean(dim=2)
                gain = flat / (rbar[..., None] + 1e-8)
                unit_bits_t = (gain * (gain + 1e-8).log() / math.log(2.0)).mean(dim=2)
                rbar_cpu = rbar.detach().cpu().numpy()
                bits_cpu = unit_bits_t.detach().cpu().numpy()
                ids = frame_to_trace_arr[t_start:t_end]
                valid_ids = np.unique(ids[ids >= 0])
                for trace_idx in valid_ids:
                    mask = ids == int(trace_idx)
                    rb = rbar_cpu[mask]
                    ub = bits_cpu[mask]
                    weights = rb * float(bin_seconds)
                    unit_expected[int(trace_idx)] += np.sum(weights, axis=0)
                    unit_numer[int(trace_idx)] += np.sum(ub * weights, axis=0)
                    unit_rate_sum[int(trace_idx)] += np.sum(rb, axis=0)
                    unit_frame_count[int(trace_idx)] += int(np.count_nonzero(mask))
                del x, full_map, rr100_map, flat, rbar, gain, unit_bits_t
            del stims, stim_all

    unit_bits = np.divide(unit_numer, np.maximum(unit_expected, 1e-8)).astype(np.float32)
    unit_mean_rate = np.divide(
        unit_rate_sum,
        np.maximum(unit_frame_count[:, None], 1),
    ).astype(np.float32)
    population_numer = np.sum(unit_numer, axis=1)
    population_denom = np.sum(unit_expected, axis=1)
    population_bits = np.divide(population_numer, np.maximum(population_denom, 1e-8)).astype(np.float32)
    return unit_bits, unit_expected.astype(np.float32), unit_mean_rate, population_bits


def score_matrix(
    *,
    scorer: CanonicalTwinScorer,
    population_view: Any,
    image_rows: pd.DataFrame,
    trace_items: list[dict[str, Any]],
    frame_batch_size: int,
    trace_batch_size: int,
    n_timepoints: int,
    bin_seconds: float,
    patch_size_px: int,
    write_outputs: bool,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    scorer.batch_size = int(frame_batch_size)
    traces = [np.asarray(item["trace"], dtype=np.float32) for item in trace_items]
    n_images = int(image_rows.shape[0])
    n_traces = int(len(traces))
    n_movies = n_images * n_traces
    n_units = int(population_view.n_units)
    ssi_matrix = np.zeros((n_movies, n_units), dtype=np.float32)
    expected_matrix = np.zeros((n_movies, n_units), dtype=np.float32)
    mean_rate_matrix = np.zeros((n_movies, n_units), dtype=np.float32)
    population_ssi = np.zeros((n_movies,), dtype=np.float32)
    movie_rows: list[dict[str, Any]] = []
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    started = time.perf_counter()
    for shard_image_ordinal, (_, image_row) in enumerate(image_rows.iterrows()):
        global_image_index = int(image_row["image_index"]) if "image_index" in image_row.index else int(shard_image_ordinal)
        patch, patch_meta = _extract_patch(
            image_row,
            canvas_cache=canvas_cache,
            patch_size_px=int(patch_size_px),
        )
        image_ssi, image_expected, image_mean_rate, image_population = score_traces_for_patch(
            scorer,
            population_view,
            patch,
            traces,
            trace_batch_size=int(trace_batch_size),
            frame_batch_size=int(frame_batch_size),
            n_timepoints=int(n_timepoints),
            bin_seconds=float(bin_seconds),
        )
        for trace_index in range(n_traces):
            matrix_row_index = shard_image_ordinal * n_traces + trace_index
            movie_index = global_image_index * n_traces + trace_index
            ssi_matrix[matrix_row_index] = image_ssi[trace_index]
            expected_matrix[matrix_row_index] = image_expected[trace_index]
            mean_rate_matrix[matrix_row_index] = image_mean_rate[trace_index]
            population_ssi[matrix_row_index] = image_population[trace_index]
            if write_outputs:
                trace_item = trace_items[trace_index]
                row = {
                    "movie_index": int(movie_index),
                    "matrix_row_index": int(matrix_row_index),
                    "image_index": int(global_image_index),
                    "shard_image_ordinal": int(shard_image_ordinal),
                    "trace_index": int(trace_index),
                    "image_source_row": int(image_row["source_row"]),
                    "trace_source_row": int(trace_item["source_row"]),
                    "image_session": str(image_row["session"]),
                    "image_trial_idx": int(image_row["trial_idx"]),
                    "trace_session": str(trace_item["session"]),
                    "trace_trial_idx": int(trace_item.get("trial_idx", -1)),
                    **patch_meta,
                }
                for key in IMAGE_FEATURE_COLUMNS:
                    if key in image_row.index:
                        row[key] = image_row[key]
                for key in TRACE_FEATURE_COLUMNS:
                    if key in trace_item:
                        row[key] = trace_item[key]
                movie_rows.append(row)
        progress(
            f"scored image {shard_image_ordinal + 1}/{n_images} "
            f"(global_image_index={global_image_index}); movies={min((shard_image_ordinal + 1) * n_traces, n_movies)}"
        )
    elapsed = time.perf_counter() - started
    if write_outputs:
        if out_dir is None:
            raise ValueError("out_dir is required when write_outputs=True.")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "ssi_matrix.npy", ssi_matrix)
        np.save(out_dir / "expected_spikes_matrix.npy", expected_matrix)
        np.save(out_dir / "mean_rate_matrix.npy", mean_rate_matrix)
        np.save(out_dir / "population_ssi.npy", population_ssi)
        np.save(out_dir / "trace_xy.npy", np.stack(traces, axis=0).astype(np.float32))
        write_csv(out_dir / "movie_feature_table.csv", movie_rows)
    return {
        "elapsed_s": float(elapsed),
        "n_images": n_images,
        "n_traces": n_traces,
        "n_movies": n_movies,
        "n_units": n_units,
        "movies_per_s": float(n_movies / elapsed) if elapsed > 0.0 else float("nan"),
        "seconds_per_movie": float(elapsed / n_movies) if n_movies > 0 else float("nan"),
    }


def feature_rows_from_items(items: list[dict[str, Any]], *, scale_metric: str, n_timepoints: int) -> list[dict[str, Any]]:
    return [
        trace_bank_metadata_row(item, idx, n_timepoints=int(n_timepoints), scale_metric=str(scale_metric))
        for idx, item in enumerate(items)
    ]


def write_unit_feature_table(path: Path, unit_tuning_csv: Path, n_units: int) -> None:
    base = pd.DataFrame({"unit_index": np.arange(int(n_units), dtype=int), "unit_label": [f"u{idx:03d}" for idx in range(int(n_units))]})
    if unit_tuning_csv.exists():
        tuning = pd.read_csv(unit_tuning_csv)
        if "unit_index" in tuning.columns:
            tuning = tuning.drop_duplicates("unit_index").copy()
            base = base.merge(tuning, on="unit_index", how="left", suffixes=("", "_tuning"))
            if "unit_label_tuning" in base.columns:
                base["unit_label"] = base["unit_label_tuning"].fillna(base["unit_label"])
                base = base.drop(columns=["unit_label_tuning"])
    path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(path, index=False)


def image_sampling_summary(images: pd.DataFrame, *, n_candidates: int, args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_images": int(args.n_images),
        "image_contrast_quantile": float(args.image_contrast_quantile),
        "candidate_rows_after_gates": int(n_candidates),
        "image_min_orientation_coherence": float(args.image_min_orientation_coherence),
        "image_min_drift_anisotropy": float(args.image_min_drift_anisotropy),
        "min_strong_contour_images": int(args.min_strong_contour_images),
        "strong_contour_orientation_coherence_min": float(args.strong_contour_orientation_coherence_min),
    }
    if "image_orientation_coherence" in images.columns:
        coherence = pd.to_numeric(images["image_orientation_coherence"], errors="coerce")
        reliable_min = max(0.2, float(args.image_min_orientation_coherence))
        out.update(
            {
                "selected_reliable_contour_images": int((coherence >= reliable_min).sum()),
                "selected_strong_contour_images": int(
                    (coherence >= float(args.strong_contour_orientation_coherence_min)).sum()
                ),
                "selected_orientation_coherence_min": float(coherence.min()),
                "selected_orientation_coherence_median": float(coherence.median()),
                "selected_orientation_coherence_max": float(coherence.max()),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.force):
        raise FileExistsError(f"{out_dir} already exists and is not empty. Pass --force to append/overwrite pilot files.")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    rows = load_source_rows(Path(args.source_csv))
    image_candidates = image_candidate_rows(
        rows,
        contrast_quantile=float(args.image_contrast_quantile),
        n_timepoints=int(args.n_timepoints),
        min_orientation_coherence=float(args.image_min_orientation_coherence),
        min_drift_anisotropy=float(args.image_min_drift_anisotropy),
    )
    images = sample_image_rows(
        image_candidates,
        int(args.n_images),
        rng=rng,
        min_strong_contour_images=int(args.min_strong_contour_images),
        strong_contour_orientation_coherence_min=float(args.strong_contour_orientation_coherence_min),
    )
    trace_bank = build_trace_bank(
        rows,
        n_timepoints=int(args.n_timepoints),
        bin_seconds=float(args.bin_seconds),
        max_path_arcmin=float(args.max_trace_path_length_arcmin),
    )
    traces = sample_trace_items(
        trace_bank,
        int(args.n_traces),
        metric=str(args.trace_scale_metric),
        sampling=str(args.trace_sampling),
        rng=rng,
        min_microsaccade_traces=int(args.min_microsaccade_traces),
    )
    image_table = annotate_selected_image_flags(
        images.copy().reset_index(drop=True),
        reliable_min=max(0.2, float(args.image_min_orientation_coherence)),
        strong_min=float(args.strong_contour_orientation_coherence_min),
    )
    image_table.insert(0, "image_index", np.arange(image_table.shape[0], dtype=int))
    image_table.to_csv(out_dir / "image_feature_table.csv", index=False)
    shard_start = max(0, int(args.image_shard_start))
    shard_stop = int(args.image_shard_stop) if int(args.image_shard_stop) > 0 else int(image_table.shape[0])
    shard_stop = min(shard_stop, int(image_table.shape[0]))
    if shard_start >= shard_stop:
        raise ValueError(
            f"Empty image shard: start={shard_start}, stop={shard_stop}, n_images={int(image_table.shape[0])}."
        )
    score_images = image_table.iloc[shard_start:shard_stop].copy().reset_index(drop=True)
    score_images.to_csv(out_dir / "scored_image_feature_table.csv", index=False)
    trace_rows = feature_rows_from_items(traces, scale_metric=str(args.trace_scale_metric), n_timepoints=int(args.n_timepoints))
    write_csv(out_dir / "trace_feature_table.csv", trace_rows)
    write_csv(out_dir / "trace_bank_metric_summary.csv", trace_bank_metric_summary_rows(trace_rows))

    population_view = load_population_view(version_name=str(args.rr100_version))
    write_unit_feature_table(out_dir / "unit_feature_table.csv", Path(args.unit_tuning_csv), int(population_view.n_units))

    frame_batches = parse_int_list(str(args.benchmark_frame_batch_sizes))
    trace_batches = parse_int_list(str(args.benchmark_trace_batch_sizes))
    benchmark_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    if not bool(args.skip_benchmark):
        bench_images = score_images.iloc[: min(int(args.benchmark_n_images), score_images.shape[0])].copy().reset_index(drop=True)
        bench_traces = traces[: min(int(args.benchmark_n_traces), len(traces))]
        for frame_batch in frame_batches:
            scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(frame_batch), empty_cache_every_batch=False)
            for trace_batch in trace_batches:
                row: dict[str, Any] = {
                    "frame_batch_size": int(frame_batch),
                    "trace_batch_size": int(trace_batch),
                    "benchmark_n_images": int(bench_images.shape[0]),
                    "benchmark_n_traces": int(len(bench_traces)),
                    "benchmark_n_movies": int(bench_images.shape[0] * len(bench_traces)),
                    "status": "ok",
                }
                try:
                    progress(f"benchmark frame_batch={frame_batch}, trace_batch={trace_batch}")
                    timing = score_matrix(
                        scorer=scorer,
                        population_view=population_view,
                        image_rows=bench_images,
                        trace_items=bench_traces,
                        frame_batch_size=int(frame_batch),
                        trace_batch_size=int(trace_batch),
                        n_timepoints=int(args.n_timepoints),
                        bin_seconds=float(args.bin_seconds),
                        patch_size_px=int(args.patch_size_px),
                        write_outputs=False,
                    )
                    row.update(timing)
                    if best is None or float(row["movies_per_s"]) > float(best["movies_per_s"]):
                        best = dict(row)
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = repr(exc)
                    progress(f"benchmark failed for frame_batch={frame_batch}, trace_batch={trace_batch}: {exc!r}")
                    try:
                        if scorer.torch.cuda.is_available() and str(args.device).startswith("cuda"):
                            scorer.torch.cuda.empty_cache()
                    except Exception:
                        pass
                benchmark_rows.append(row)
            del scorer
        write_csv(out_dir / "benchmark_results.csv", benchmark_rows)

    if bool(args.benchmark_only):
        write_json(
            out_dir / "summary.json",
            {
                "analysis": "backimage_real_trace_ssi_matrix_pilot",
                "mode": "benchmark_only",
                "best_benchmark": best,
                "image_sampling": image_sampling_summary(images, n_candidates=image_candidates.shape[0], args=args),
                "n_images_selected": int(images.shape[0]),
                "n_images_scored": int(score_images.shape[0]),
                "n_traces_selected": int(len(traces)),
                "image_shard": {
                    "start": int(shard_start),
                    "stop": int(shard_stop),
                    "n_total_images": int(image_table.shape[0]),
                    "global_image_indices": score_images["image_index"].astype(int).to_list(),
                },
                "out_dir": out_dir,
            },
        )
        return

    pilot_frame_batch = int(args.pilot_frame_batch_size) if int(args.pilot_frame_batch_size) > 0 else None
    pilot_trace_batch = int(args.pilot_trace_batch_size) if int(args.pilot_trace_batch_size) > 0 else None
    if pilot_frame_batch is None or pilot_trace_batch is None:
        if best is None:
            pilot_frame_batch = frame_batches[-1]
            pilot_trace_batch = trace_batches[-1]
        else:
            pilot_frame_batch = int(best["frame_batch_size"])
            pilot_trace_batch = int(best["trace_batch_size"])
    scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(pilot_frame_batch), empty_cache_every_batch=False)
    progress(f"running pilot matrix with frame_batch={pilot_frame_batch}, trace_batch={pilot_trace_batch}")
    pilot_timing = score_matrix(
        scorer=scorer,
        population_view=population_view,
        image_rows=score_images,
        trace_items=traces,
        frame_batch_size=int(pilot_frame_batch),
        trace_batch_size=int(pilot_trace_batch),
        n_timepoints=int(args.n_timepoints),
        bin_seconds=float(args.bin_seconds),
        patch_size_px=int(args.patch_size_px),
        write_outputs=True,
        out_dir=out_dir,
    )
    write_json(
        out_dir / "summary.json",
        {
            "analysis": "backimage_real_trace_ssi_matrix_pilot",
            "source_csv": Path(args.source_csv),
            "unit_tuning_csv": Path(args.unit_tuning_csv),
            "out_dir": out_dir,
            "rr100_version": str(args.rr100_version),
            "n_timepoints": int(args.n_timepoints),
            "bin_seconds": float(args.bin_seconds),
            "patch_size_px": int(args.patch_size_px),
            "image_sampling": image_sampling_summary(images, n_candidates=image_candidates.shape[0], args=args),
            "image_shard": {
                "start": int(shard_start),
                "stop": int(shard_stop),
                "n_total_images": int(image_table.shape[0]),
                "n_scored_images": int(score_images.shape[0]),
                "global_image_indices": score_images["image_index"].astype(int).to_list(),
            },
            "trace_sampling": {
                "n_traces": int(args.n_traces),
                "trace_sampling": str(args.trace_sampling),
                "trace_scale_metric": str(args.trace_scale_metric),
                "max_trace_path_length_arcmin": float(args.max_trace_path_length_arcmin),
                "trace_bank_eligible_rows": int(len(trace_bank)),
                "min_microsaccade_traces": int(args.min_microsaccade_traces),
                "selected_microsaccade_traces": int(sum(microsaccade_event_count(item) > 0 for item in traces)),
            },
            "benchmark_best": best,
            "pilot": {
                "frame_batch_size": int(pilot_frame_batch),
                "trace_batch_size": int(pilot_trace_batch),
                **pilot_timing,
            },
            "outputs": {
                "ssi_matrix": out_dir / "ssi_matrix.npy",
                "expected_spikes_matrix": out_dir / "expected_spikes_matrix.npy",
                "mean_rate_matrix": out_dir / "mean_rate_matrix.npy",
                "population_ssi": out_dir / "population_ssi.npy",
                "movie_feature_table": out_dir / "movie_feature_table.csv",
                "image_feature_table": out_dir / "image_feature_table.csv",
                "scored_image_feature_table": out_dir / "scored_image_feature_table.csv",
                "trace_feature_table": out_dir / "trace_feature_table.csv",
                "trace_xy": out_dir / "trace_xy.npy",
                "unit_feature_table": out_dir / "unit_feature_table.csv",
                "benchmark_results": out_dir / "benchmark_results.csv",
            },
            "contract": (
                "Rows are image-major image x trace movies. SSI is the corrected time-resolved "
                "spatial SSI computed from full twin rate maps after applying the RR100 population view. "
                "Traces are unscaled center-cropped native real BackImage snippets."
            ),
        },
    )
    progress(f"wrote pilot matrix to {out_dir}")


if __name__ == "__main__":
    main()
