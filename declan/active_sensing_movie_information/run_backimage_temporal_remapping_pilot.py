#!/usr/bin/env python3
"""Run fixed-geometry temporal-remapping counterfactuals for BackImage RR100 SSI.

The default use is a lightweight dry run that builds retimed trajectories,
trajectory metrics, and QC panels. Drop ``--dry-run`` to score the same
conditions through the existing BackImage RR100 twin/SSI path.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    RR100_MOVIE_MEDOID_VERSION,
    trace_bank_metric_summary_rows,
)
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import (
    DEFAULT_SOURCE_CSV,
    DEFAULT_UNIT_TUNING_CSV,
    IMAGE_FEATURE_COLUMNS,
    TRACE_FEATURE_COLUMNS,
    annotate_selected_image_flags,
    build_trace_bank,
    image_candidate_rows,
    load_source_rows,
    microsaccade_event_count,
    sample_image_rows,
    sample_trace_items,
    score_traces_for_patch,
    write_unit_feature_table,
)
from declan.active_sensing_movie_information.temporal_remapping import (
    MODEL_NYQUIST_HZ,
    MODEL_RATE_HZ,
    invariance_report,
    retime_trace,
    scale_trace_about_center,
    timing_window,
    trajectory_metrics,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import CanonicalTwinScorer
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.redundancy_resolved_v1_population import load_population_view


DEFAULT_OUT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/temporal_remapping/"
    "backimage_rr100_retiming_pilot_v1"
)
DEFAULT_FIGURE4_CANDIDATE_SETS_CSV = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_axis_conditioned_matched_static_percandidate_gpu1_n128_c4_k16_scales_0p5_1_2_bconsistent_v1/"
    "candidate_sets.csv"
)
DEFAULT_TRAVERSAL_FRAMES = "8,12,16,24,32"
DEFAULT_TIMING_PLACEMENTS = "terminal,endpoint_hold,centered"
DEFAULT_RETIMING_PROFILES = "uniform,natural_speed_profile"
DEFAULT_AMPLITUDE_SCALES = ""
PREFERRED_SF_COLUMNS = (
    "dynamic_log_gaussian_marginal_sf_cpd",
    "sf_split_metric",
    "dynamic_amp_weighted_sf_cpd",
    "static_rate_weighted_sf_cpd",
    "static_peak_spatial_cpd_by_mean_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--unit-tuning-csv", type=Path, default=DEFAULT_UNIT_TUNING_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--n-traces", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--image-pool",
        choices=("figure4_candidates", "reviewed_windows"),
        default="figure4_candidates",
        help=(
            "Image pool to sample from. figure4_candidates reuses source rows from the saved "
            "Figure-4 matched-static candidate table and applies only basic source-window validity gates."
        ),
    )
    parser.add_argument("--figure4-candidate-sets-csv", type=Path, default=DEFAULT_FIGURE4_CANDIDATE_SETS_CSV)
    parser.add_argument("--n-timepoints", type=int, default=32)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / MODEL_RATE_HZ)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--image-contrast-quantile", type=float, default=0.75)
    parser.add_argument("--image-min-orientation-coherence", type=float, default=0.2)
    parser.add_argument("--image-min-drift-anisotropy", type=float, default=0.0)
    parser.add_argument("--min-strong-contour-images", type=int, default=0)
    parser.add_argument("--strong-contour-orientation-coherence-min", type=float, default=0.5)
    parser.add_argument("--max-trace-path-length-arcmin", type=float, default=350.0)
    parser.add_argument("--max-microsaccade-events", type=int, default=0)
    parser.add_argument("--trace-scale-metric", type=str, default="rendered_path_length_arcmin")
    parser.add_argument("--trace-sampling", choices=("quantile", "random"), default="quantile")
    parser.add_argument("--min-microsaccade-traces", type=int, default=0)
    parser.add_argument("--traversal-frames", type=str, default=DEFAULT_TRAVERSAL_FRAMES)
    parser.add_argument("--timing-placements", type=str, default=DEFAULT_TIMING_PLACEMENTS)
    parser.add_argument("--retiming-profiles", type=str, default=DEFAULT_RETIMING_PROFILES)
    parser.add_argument("--amplitude-scales", type=str, default=DEFAULT_AMPLITUDE_SCALES)
    parser.add_argument("--amplitude-traversal-frames", type=str, default="8,16,32")
    parser.add_argument("--amplitude-placement", type=str, default="terminal")
    parser.add_argument("--amplitude-profile", type=str, default="uniform")
    parser.add_argument("--scaling-center", choices=("centroid", "start", "end"), default="centroid")
    parser.add_argument("--include-static", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-original", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nyquist-hz", type=float, default=MODEL_NYQUIST_HZ)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--trace-batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-unit-observations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verification-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qc-trace-plots", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[backimage-temporal-remapping] {message}", flush=True)


def parse_csv_float_list(text: str | None) -> list[float]:
    if text is None:
        return []
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_csv_int_list(text: str | None) -> list[int]:
    values = [int(float(part.strip())) for part in str(text).split(",") if part.strip()]
    if any(value <= 0 for value in values):
        raise ValueError(f"Expected positive integer values, got {text!r}.")
    return values


def parse_csv_str_list(text: str | None) -> list[str]:
    if text is None:
        return []
    return [part.strip() for part in str(text).split(",") if part.strip()]


def unique_preserving_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def figure4_candidate_source_rows(path: Path) -> list[int]:
    table = pd.read_csv(path)
    rows: list[int] = []
    if "observation_source_row" in table.columns:
        values = pd.to_numeric(table["observation_source_row"], errors="coerce")
        rows.extend(int(value) for value in values.dropna().to_numpy(dtype=np.int64))
    if "candidate_ids" in table.columns:
        for text in table["candidate_ids"].dropna().astype(str):
            for token in text.split(";"):
                token = token.strip()
                if not token.startswith("source_row:"):
                    continue
                try:
                    rows.append(int(token.split(":", 1)[1]))
                except ValueError:
                    continue
    if not rows:
        raise ValueError(f"No source rows found in Figure-4 candidate table: {path}")
    return unique_preserving_order(rows)


def selected_image_quality_summary(images: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
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


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def sem(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else float("nan")
    return float(np.std(arr, ddof=1) / math.sqrt(float(arr.size)))


def condition_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if bool(args.include_static):
        specs.append(
            {
                "condition_index": len(specs),
                "condition_id": "stabilized_static",
                "condition_label": "static",
                "condition_group": "stabilized",
                "is_static_baseline": True,
                "is_original_natural": False,
                "traversal_frames": int(args.n_timepoints),
                "timing_placement": "static",
                "retiming_profile": "static",
                "amplitude_scale": 0.0,
            }
        )
    if bool(args.include_original):
        specs.append(
            {
                "condition_index": len(specs),
                "condition_id": "original_natural_timing",
                "condition_label": "original",
                "condition_group": "original",
                "is_static_baseline": False,
                "is_original_natural": True,
                "traversal_frames": int(args.n_timepoints),
                "timing_placement": "natural",
                "retiming_profile": "natural_source",
                "amplitude_scale": 1.0,
            }
        )
    for profile in parse_csv_str_list(args.retiming_profiles):
        for placement in parse_csv_str_list(args.timing_placements):
            for frames in parse_csv_int_list(args.traversal_frames):
                if int(frames) > int(args.n_timepoints):
                    raise ValueError(f"traversal frame count {frames} exceeds n_timepoints={args.n_timepoints}.")
                specs.append(
                    {
                        "condition_index": len(specs),
                        "condition_id": f"retime_{profile}_{placement}_t{int(frames):02d}",
                        "condition_label": f"{profile}/{placement}/{int(frames)}f",
                        "condition_group": "retiming",
                        "is_static_baseline": False,
                        "is_original_natural": False,
                        "traversal_frames": int(frames),
                        "timing_placement": str(placement),
                        "retiming_profile": str(profile),
                        "amplitude_scale": 1.0,
                    }
                )
    for beta in parse_csv_float_list(args.amplitude_scales):
        for frames in parse_csv_int_list(args.amplitude_traversal_frames):
            if int(frames) > int(args.n_timepoints):
                raise ValueError(f"amplitude traversal frame count {frames} exceeds n_timepoints={args.n_timepoints}.")
            specs.append(
                {
                    "condition_index": len(specs),
                    "condition_id": f"amp_b{scale_token(beta)}_{args.amplitude_profile}_{args.amplitude_placement}_t{int(frames):02d}",
                    "condition_label": f"b{float(beta):g}/{int(frames)}f",
                    "condition_group": "amplitude_duration",
                    "is_static_baseline": bool(np.isclose(float(beta), 0.0)),
                    "is_original_natural": False,
                    "traversal_frames": int(frames),
                    "timing_placement": str(args.amplitude_placement),
                    "retiming_profile": str(args.amplitude_profile),
                    "amplitude_scale": float(beta),
                }
            )
    return specs


def scale_token(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def build_condition_trace(source_trace: np.ndarray, spec: dict[str, Any], *, scaling_center: str) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_trace, dtype=np.float32)
    if bool(spec.get("is_static_baseline", False)) and str(spec.get("condition_group")) == "stabilized":
        static = np.zeros_like(source, dtype=np.float32)
        return static, static
    if bool(spec.get("is_original_natural", False)):
        return source.astype(np.float32, copy=True), source.astype(np.float32, copy=True)
    beta = float(spec.get("amplitude_scale", 1.0))
    base = source
    if not np.isclose(beta, 1.0):
        base = scale_trace_about_center(source, beta, center=str(scaling_center))
    trace = retime_trace(
        base,
        traversal_frames=int(spec["traversal_frames"]),
        total_frames=int(source.shape[0]),
        placement=str(spec["timing_placement"]),
        profile=str(spec["retiming_profile"]),
    )
    return np.asarray(trace, dtype=np.float32), np.asarray(base, dtype=np.float32)


def row_contour_axis_deg(row: pd.Series) -> float:
    for key in ("image_edge_axis_deg", "image_gradient_axis_deg", "image_spectrum_orientation_deg"):
        if key in row.index:
            value = finite_float(row[key])
            if math.isfinite(value):
                if key == "image_spectrum_orientation_deg":
                    return float((value + 90.0) % 180.0)
                return float(value % 180.0)
    return 0.0


def select_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(int(args.seed))
    rows = load_source_rows(Path(args.source_csv))
    if str(args.image_pool) == "figure4_candidates":
        figure4_rows = figure4_candidate_source_rows(Path(args.figure4_candidate_sets_csv))
        base_candidates = image_candidate_rows(
            rows,
            contrast_quantile=0.0,
            n_timepoints=int(args.n_timepoints),
            min_orientation_coherence=0.0,
            min_drift_anisotropy=0.0,
        )
        figure4_set = set(figure4_rows)
        image_candidates = base_candidates[base_candidates["source_row"].astype(int).isin(figure4_set)].copy()
        selected_source_set = set(image_candidates["source_row"].astype(int).to_list())
        missing_after_base_gates = [row for row in figure4_rows if row not in selected_source_set]
        image_meta = {
            "image_pool": "figure4_candidates",
            "figure4_candidate_sets_csv": Path(args.figure4_candidate_sets_csv),
            "figure4_source_rows": int(len(figure4_rows)),
            "candidate_rows_before_figure4_intersection": int(base_candidates.shape[0]),
            "candidate_rows_after_pool": int(image_candidates.shape[0]),
            "figure4_source_rows_missing_after_base_gates": missing_after_base_gates,
            "extra_image_content_gates_applied": False,
        }
    else:
        image_candidates = image_candidate_rows(
            rows,
            contrast_quantile=float(args.image_contrast_quantile),
            n_timepoints=int(args.n_timepoints),
            min_orientation_coherence=float(args.image_min_orientation_coherence),
            min_drift_anisotropy=float(args.image_min_drift_anisotropy),
        )
        image_meta = {
            "image_pool": "reviewed_windows",
            "candidate_rows_after_pool": int(image_candidates.shape[0]),
            "extra_image_content_gates_applied": True,
            "image_contrast_quantile": float(args.image_contrast_quantile),
            "image_min_orientation_coherence": float(args.image_min_orientation_coherence),
            "image_min_drift_anisotropy": float(args.image_min_drift_anisotropy),
        }
    images = sample_image_rows(
        image_candidates,
        int(args.n_images),
        rng=rng,
        min_strong_contour_images=int(args.min_strong_contour_images),
        strong_contour_orientation_coherence_min=float(args.strong_contour_orientation_coherence_min),
    )
    image_table = annotate_selected_image_flags(
        images.copy().reset_index(drop=True),
        reliable_min=max(0.2, float(args.image_min_orientation_coherence)),
        strong_min=float(args.strong_contour_orientation_coherence_min),
    )
    image_table.insert(0, "image_index", np.arange(image_table.shape[0], dtype=int))

    trace_bank = build_trace_bank(
        rows,
        n_timepoints=int(args.n_timepoints),
        bin_seconds=float(args.bin_seconds),
        max_path_arcmin=float(args.max_trace_path_length_arcmin),
    )
    max_events = int(args.max_microsaccade_events)
    if max_events >= 0:
        trace_bank = [item for item in trace_bank if microsaccade_event_count(item) <= max_events]
    traces = sample_trace_items(
        trace_bank,
        int(args.n_traces),
        metric=str(args.trace_scale_metric),
        sampling=str(args.trace_sampling),
        rng=rng,
        min_microsaccade_traces=int(args.min_microsaccade_traces),
    )
    meta = {
        "n_source_rows": int(rows.shape[0]),
        "n_image_candidates": int(image_candidates.shape[0]),
        "n_trace_bank_after_filters": int(len(trace_bank)),
        "image_sampling": {
            "n_images": int(args.n_images),
            "candidate_rows_after_gates": int(image_candidates.shape[0]),
            "min_strong_contour_images": int(args.min_strong_contour_images),
            "strong_contour_orientation_coherence_min": float(args.strong_contour_orientation_coherence_min),
            **image_meta,
            **selected_image_quality_summary(images, args),
        },
        "selected_microsaccade_traces": int(sum(microsaccade_event_count(item) > 0 for item in traces)),
    }
    return image_table, traces, trace_bank, meta


def trace_feature_rows(trace_items: list[dict[str, Any]], *, scale_metric: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_index, item in enumerate(trace_items):
        row = {
            "trace_index": int(trace_index),
            "trace_source_row": int(item["source_row"]),
            "trace_session": str(item["session"]),
            "trace_trial_idx": int(item.get("trial_idx", -1)),
            "trace_has_microsaccade": bool(microsaccade_event_count(item) > 0),
            "trace_microsaccade_event_count": int(microsaccade_event_count(item)),
            "trace_scale_metric": str(scale_metric),
            "trace_scale_metric_value": finite_float(item.get(scale_metric)),
        }
        for key in TRACE_FEATURE_COLUMNS:
            if key in item:
                row[key] = item[key]
        rows.append(row)
    return rows


def build_metric_rows(
    image_table: pd.DataFrame,
    trace_items: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    family_metrics: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for image_pos, image_row in image_table.iterrows():
        image_index = int(image_row["image_index"])
        contour_axis = row_contour_axis_deg(image_row)
        for trace_index, item in enumerate(trace_items):
            source_trace = np.asarray(item["trace"], dtype=np.float32)
            for spec in specs:
                trace, metric_source = build_condition_trace(source_trace, spec, scaling_center=str(args.scaling_center))
                metrics = trajectory_metrics(
                    trace,
                    source_trace=metric_source,
                    traversal_frames=int(spec["traversal_frames"]),
                    total_frames=int(args.n_timepoints),
                    frame_rate_hz=1.0 / float(args.bin_seconds),
                    contour_orientation_deg=contour_axis,
                    preferred_sf_cpd=None,
                    retiming_profile=str(spec["retiming_profile"]),
                    timing_placement=str(spec["timing_placement"]),
                    condition_name=str(spec["condition_id"]),
                    nyquist_hz=float(args.nyquist_hz),
                )
                row = {
                    "image_position": int(image_pos),
                    "image_index": image_index,
                    "image_source_row": int(image_row["source_row"]),
                    "image_session": str(image_row["session"]),
                    "image_trial_idx": int(image_row["trial_idx"]),
                    "image_contour_axis_deg": contour_axis,
                    "trace_index": int(trace_index),
                    "trace_source_row": int(item["source_row"]),
                    "trace_session": str(item["session"]),
                    "trace_trial_idx": int(item.get("trial_idx", -1)),
                    "source_has_microsaccade": bool(microsaccade_event_count(item) > 0),
                    "source_microsaccade_event_count": int(microsaccade_event_count(item)),
                    **spec,
                    **metrics,
                }
                rows.append(row)
                if str(spec.get("condition_group")) == "retiming":
                    family_metrics.setdefault((image_index, trace_index), []).append(metrics)
    reports = [invariance_report(metrics) for metrics in family_metrics.values()]
    qc = {
        "n_retiming_families": int(len(reports)),
        "n_failed_retiming_families": int(sum(not bool(report["ok"]) for report in reports)),
        "failures": [report for report in reports if not bool(report["ok"])],
    }
    return rows, qc


def plot_qc_traces(
    out_dir: Path,
    trace_items: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    n_plots = min(max(0, int(args.qc_trace_plots)), len(trace_items))
    if n_plots <= 0:
        return []
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    retiming_specs = [spec for spec in specs if str(spec.get("condition_group")) == "retiming"]
    if not retiming_specs:
        return []
    paths: list[str] = []
    for trace_index, item in enumerate(trace_items[:n_plots]):
        source = np.asarray(item["trace"], dtype=np.float32)
        subset = [retiming_specs[0]]
        for spec in retiming_specs:
            if spec["condition_id"] not in {s["condition_id"] for s in subset} and len(subset) < 7:
                subset.append(spec)
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))
        ax_xy, ax_pos, ax_speed, ax_across = axes.ravel()
        ax_xy.plot(source[:, 0], source[:, 1], color="#111111", lw=2.0, label="source")
        t = np.arange(source.shape[0])
        ax_pos.plot(t, source[:, 0], color="#111111", lw=1.2, alpha=0.9, label="source x")
        ax_pos.plot(t, source[:, 1], color="#777777", lw=1.2, alpha=0.9, label="source y")
        colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(subset)))
        dt = float(args.bin_seconds)
        for color, spec in zip(colors, subset, strict=True):
            trace, _metric_source = build_condition_trace(source, spec, scaling_center=str(args.scaling_center))
            label = str(spec["condition_label"])
            ax_xy.plot(trace[:, 0], trace[:, 1], color=color, lw=1.0, alpha=0.9, label=label)
            ax_pos.plot(t, trace[:, 0], color=color, lw=0.9, alpha=0.75)
            speed = np.linalg.norm(np.diff(trace, axis=0), axis=1) / dt
            ax_speed.plot(t[1:], speed, color=color, lw=1.0, alpha=0.9, label=label)
            across = np.diff(trace[:, 1], axis=0) / dt
            ax_across.plot(t[1:], across, color=color, lw=1.0, alpha=0.9, label=label)
        ax_xy.set_title("x-y path overlay")
        ax_xy.set_xlabel("x (deg)")
        ax_xy.set_ylabel("y (deg)")
        ax_xy.axis("equal")
        ax_pos.set_title("sampled x/y positions")
        ax_pos.set_xlabel("model frame")
        ax_speed.set_title("speed")
        ax_speed.set_xlabel("model frame")
        ax_speed.set_ylabel("deg/s")
        ax_across.set_title("y-axis velocity proxy")
        ax_across.set_xlabel("model frame")
        ax_across.set_ylabel("deg/s")
        for ax in axes.ravel():
            ax.grid(True, color="#e6e6e6", lw=0.7)
        ax_xy.legend(fontsize=6.0, frameon=False)
        fig.suptitle(f"Retiming QC trace {trace_index} source_row={int(item['source_row'])}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = qc_dir / f"retiming_trace_{trace_index:03d}_source_row_{int(item['source_row'])}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def finite_column(df: pd.DataFrame, key: str) -> np.ndarray:
    if key not in df.columns:
        return np.zeros((0,), dtype=np.float64)
    values = pd.to_numeric(df[key], errors="coerce").to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def save_figure(fig: Any, path: Path, *, dpi: int = 170) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_selected_inputs_overview(
    out_dir: Path,
    image_table: pd.DataFrame,
    trace_items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4))
    ax_img, ax_trace_hist, ax_paths, ax_speed = axes.ravel()

    contrast = finite_column(image_table, "image_patch_rms_contrast")
    coherence = finite_column(image_table, "image_orientation_coherence")
    if contrast.size and coherence.size and contrast.size == coherence.size:
        ax_img.scatter(contrast, coherence, s=32, color="#4c78a8", alpha=0.88)
        ax_img.set_xlabel("patch RMS contrast")
        ax_img.set_ylabel("orientation coherence")
    else:
        ax_img.text(0.5, 0.5, "image contrast/coherence\ncolumns unavailable", ha="center", va="center")
    ax_img.set_title("Selected image patches")

    path_lengths = np.asarray(
        [
            finite_float(item.get("rendered_path_length_arcmin", item.get("path_length_arcmin", np.nan)))
            for item in trace_items
        ],
        dtype=np.float64,
    )
    path_lengths = path_lengths[np.isfinite(path_lengths)]
    if path_lengths.size:
        ax_trace_hist.hist(path_lengths, bins=min(max(path_lengths.size, 4), 20), color="#72b7b2", edgecolor="white")
        ax_trace_hist.axvline(np.nanmedian(path_lengths), color="#333333", lw=1.2, label="median")
        ax_trace_hist.legend(fontsize=7, frameon=False)
    else:
        ax_trace_hist.text(0.5, 0.5, "no finite trace path lengths", ha="center", va="center")
    ax_trace_hist.set_xlabel("selected trace path length (arcmin)")
    ax_trace_hist.set_ylabel("traces")
    ax_trace_hist.set_title("Selected drift traces")

    dt = float(args.bin_seconds)
    for trace_index, item in enumerate(trace_items[: min(len(trace_items), 16)]):
        trace = np.asarray(item["trace"], dtype=np.float64)
        ax_paths.plot(trace[:, 0], trace[:, 1], lw=1.0, alpha=0.72)
        speed = np.linalg.norm(np.diff(trace, axis=0), axis=1) / max(dt, 1e-12)
        if speed.size:
            ax_speed.plot(np.arange(1, trace.shape[0]), speed, lw=0.95, alpha=0.62)
    ax_paths.set_title("Selected source paths")
    ax_paths.set_xlabel("x (deg)")
    ax_paths.set_ylabel("y (deg)")
    ax_paths.axis("equal")
    ax_speed.set_title("Selected source speeds")
    ax_speed.set_xlabel("model frame")
    ax_speed.set_ylabel("deg/s")
    for ax in axes.ravel():
        ax.grid(True, color="#e7e7e7", linewidth=0.7)
    fig.suptitle("Step 1: selected image and trace inputs", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, out_dir / "verification_01_selected_inputs.png")


def plot_condition_timing_design(out_dir: Path, specs: list[dict[str, Any]], args: argparse.Namespace) -> str:
    from matplotlib.patches import Patch

    n_frames = int(args.n_timepoints)
    height = max(4.0, 0.34 * len(specs) + 1.5)
    fig, ax = plt.subplots(figsize=(11.0, height))
    y_ticks: list[float] = []
    y_labels: list[str] = []
    for row_index, spec in enumerate(specs):
        y = float(row_index)
        y_ticks.append(y + 0.4)
        y_labels.append(str(spec["condition_label"])[:42])
        group = str(spec.get("condition_group", ""))
        if group == "stabilized":
            ax.broken_barh([(0, n_frames)], (y, 0.8), facecolors="#c9c9c9")
        elif group == "original":
            ax.broken_barh([(0, n_frames)], (y, 0.8), facecolors="#6f6f6f")
        else:
            start, stop = timing_window(n_frames, int(spec["traversal_frames"]), str(spec["timing_placement"]))
            if start > 0:
                ax.broken_barh([(0, start)], (y, 0.8), facecolors="#d8d8d8")
            ax.broken_barh([(start, stop - start + 1)], (y, 0.8), facecolors="#4c78a8")
            if stop + 1 < n_frames:
                ax.broken_barh([(stop + 1, n_frames - stop - 1)], (y, 0.8), facecolors="#a9a9a9")
    ax.set_xlim(0, n_frames)
    ax.set_ylim(0, len(specs))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("model frame in 32-frame block")
    ax.set_title("Step 2: timing design for counterfactual 32-frame histories")
    ax.grid(True, axis="x", color="#e7e7e7", linewidth=0.7)
    ax.legend(
        handles=[
            Patch(facecolor="#c9c9c9", label="static"),
            Patch(facecolor="#6f6f6f", label="original"),
            Patch(facecolor="#d8d8d8", label="hold start"),
            Patch(facecolor="#4c78a8", label="traverse path"),
            Patch(facecolor="#a9a9a9", label="hold endpoint"),
        ],
        loc="upper right",
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout()
    return save_figure(fig, out_dir / "verification_02_timing_design.png")


def plot_trajectory_metric_summary(out_dir: Path, metric_rows: list[dict[str, Any]]) -> str:
    df = pd.DataFrame(metric_rows)
    work = df[df["condition_group"].isin(["retiming", "amplitude_duration"])].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    if work.empty:
        for ax in axes.ravel():
            ax.axis("off")
        axes.ravel()[0].text(0.5, 0.5, "no retiming/amplitude conditions", ha="center", va="center")
        return save_figure(fig, out_dir / "verification_03_trajectory_metrics.png")

    group_cols = ["condition_group", "retiming_profile", "timing_placement", "amplitude_scale", "traversal_frames", "traversal_duration_ms"]
    summary = (
        work.groupby(group_cols, dropna=False)
        .agg(
            path_length_arcmin=("path_length_arcmin", "mean"),
            sampled_path_length_arcmin=("model_sampled_path_length_arcmin", "mean"),
            rms_across_velocity_deg_s=("rms_across_velocity_deg_s", "mean"),
            peak_speed_deg_s=("peak_speed_deg_s", "mean"),
            n_distinct_sampled_positions=("n_distinct_sampled_positions", "mean"),
            hold_before_frames=("hold_before_frames", "mean"),
            hold_after_frames=("hold_after_frames", "mean"),
        )
        .reset_index()
        .sort_values(["condition_group", "retiming_profile", "timing_placement", "amplitude_scale", "traversal_frames"])
    )
    ax_geom, ax_vel, ax_samples, ax_hold = axes.ravel()
    for key, group in summary.groupby(["condition_group", "retiming_profile", "timing_placement", "amplitude_scale"], dropna=False):
        label = "/".join(str(part) for part in key if str(part) not in {"nan", ""})
        ax_geom.plot(group["traversal_duration_ms"], group["path_length_arcmin"], marker="o", lw=1.0, label=f"continuous {label}")
        ax_geom.plot(group["traversal_duration_ms"], group["sampled_path_length_arcmin"], marker="x", lw=1.0, ls="--", label=f"sampled {label}")
        ax_vel.plot(group["traversal_duration_ms"], group["rms_across_velocity_deg_s"], marker="o", lw=1.1, label=label)
        ax_samples.plot(group["traversal_duration_ms"], group["n_distinct_sampled_positions"], marker="o", lw=1.1, label=label)
        ax_hold.plot(group["traversal_duration_ms"], group["hold_before_frames"], marker="o", lw=1.0, label=f"before {label}")
        ax_hold.plot(group["traversal_duration_ms"], group["hold_after_frames"], marker="x", lw=1.0, ls="--", label=f"after {label}")
    ax_geom.set_title("Continuous vs model-sampled path length")
    ax_geom.set_ylabel("arcmin")
    ax_vel.set_title("Across-contour velocity")
    ax_vel.set_ylabel("deg/s")
    ax_samples.set_title("Distinct sampled positions")
    ax_samples.set_ylabel("positions")
    ax_hold.set_title("Hold frames")
    ax_hold.set_ylabel("frames")
    for ax in axes.ravel():
        ax.set_xlabel("traversal duration (ms)")
        ax.grid(True, color="#e7e7e7", linewidth=0.7)
    ax_vel.legend(fontsize=6.0, frameon=False)
    fig.suptitle("Step 3: trajectory metrics before model scoring", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, out_dir / "verification_03_trajectory_metrics.png")


def plot_geometry_invariance(out_dir: Path, metric_rows: list[dict[str, Any]]) -> str:
    df = pd.DataFrame(metric_rows)
    work = df[df["condition_group"] == "retiming"].copy()
    keys = ["path_length_deg", "start_x", "start_y", "end_x", "end_y", "min_x", "max_x", "min_y", "max_y"]
    deltas = {key: 0.0 for key in keys}
    if not work.empty:
        for _family, group in work.groupby(["image_index", "trace_index"], dropna=False):
            group = group.sort_values("condition_index")
            ref = group.iloc[0]
            for key in keys:
                values = pd.to_numeric(group[key], errors="coerce").to_numpy(float)
                base = finite_float(ref[key])
                if np.isfinite(values).any() and math.isfinite(base):
                    deltas[key] = max(deltas[key], float(np.nanmax(np.abs(values - base))))
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    names = list(deltas)
    values = np.asarray([deltas[name] for name in names], dtype=np.float64)
    ax.bar(np.arange(len(names)), values, color="#59a14f", alpha=0.86)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("max absolute delta")
    if np.nanmax(np.abs(values)) <= 0.0:
        ax.set_ylim(-1e-12, 1e-12)
        ax.text(0.5, 0.72, "all checked invariants exact", transform=ax.transAxes, ha="center", va="center")
    else:
        ax.set_yscale("symlog", linthresh=1e-9)
    ax.set_title("Step 4: continuous-geometry invariance across retimed conditions")
    ax.grid(True, axis="y", color="#e7e7e7", linewidth=0.7)
    fig.tight_layout()
    return save_figure(fig, out_dir / "verification_04_geometry_invariance.png")


def write_verification_figures(
    out_dir: Path,
    image_table: pd.DataFrame,
    trace_items: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    if not bool(args.verification_plots):
        return []
    paths = [
        plot_selected_inputs_overview(out_dir, image_table, trace_items, args),
        plot_condition_timing_design(out_dir, specs, args),
        plot_trajectory_metric_summary(out_dir, metric_rows),
        plot_geometry_invariance(out_dir, metric_rows),
    ]
    return paths


def load_unit_table(unit_tuning_csv: Path, n_units: int) -> pd.DataFrame:
    base = pd.DataFrame({"unit_index": np.arange(int(n_units), dtype=int), "unit_label": [f"u{idx:03d}" for idx in range(int(n_units))]})
    if Path(unit_tuning_csv).exists():
        tuning = pd.read_csv(unit_tuning_csv)
        if "unit_index" in tuning.columns:
            base = base.merge(tuning.drop_duplicates("unit_index"), on="unit_index", how="left", suffixes=("", "_tuning"))
            if "unit_label_tuning" in base.columns:
                base["unit_label"] = base["unit_label_tuning"].fillna(base["unit_label"])
                base = base.drop(columns=["unit_label_tuning"])
    sf_col = next((col for col in PREFERRED_SF_COLUMNS if col in base.columns), "")
    if sf_col:
        base["preferred_sf_cpd"] = pd.to_numeric(base[sf_col], errors="coerce")
        base["preferred_sf_source_column"] = sf_col
    else:
        base["preferred_sf_cpd"] = np.nan
        base["preferred_sf_source_column"] = ""
    return base


def score_counterfactuals(
    image_table: pd.DataFrame,
    trace_items: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    population_view = load_population_view(version_name=str(args.rr100_version))
    requested_device = str(args.device)
    actual_device = requested_device
    if requested_device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                progress(f"requested device {requested_device!r}, but torch reports no CUDA; falling back to CPU")
                actual_device = "cpu"
        except Exception as exc:
            progress(f"could not check CUDA availability ({exc!r}); falling back to CPU")
            actual_device = "cpu"
    scorer = CanonicalTwinScorer(
        device=actual_device,
        batch_size=int(args.frame_batch_size),
        empty_cache_every_batch=False,
    )
    unit_table = load_unit_table(Path(args.unit_tuning_csv), int(population_view.n_units))
    unit_table.to_csv(Path(args.out_dir) / "unit_feature_table.csv", index=False)

    n_images = int(image_table.shape[0])
    n_traces = int(len(trace_items))
    n_conditions = int(len(specs))
    n_units = int(population_view.n_units)
    n_movies = n_images * n_traces * n_conditions
    unit_bits = np.zeros((n_movies, n_units), dtype=np.float32)
    expected = np.zeros((n_movies, n_units), dtype=np.float32)
    mean_rate = np.zeros((n_movies, n_units), dtype=np.float32)
    population = np.zeros((n_movies,), dtype=np.float32)
    observation_rows: list[dict[str, Any]] = []
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    started = time.perf_counter()

    metric_by_key = {
        (
            int(row["image_index"]),
            int(row["trace_index"]),
            int(row["condition_index"]),
        ): row
        for row in pd.read_csv(Path(args.out_dir) / "retimed_trajectory_metrics.csv").to_dict("records")
    }

    row_offset = 0
    for image_pos, (_, image_row) in enumerate(image_table.iterrows()):
        patch, patch_meta = _extract_patch(
            image_row,
            canvas_cache=canvas_cache,
            patch_size_px=int(args.patch_size_px),
        )
        traces_for_patch: list[np.ndarray] = []
        row_keys: list[tuple[int, int, int]] = []
        for trace_index, item in enumerate(trace_items):
            source_trace = np.asarray(item["trace"], dtype=np.float32)
            for spec in specs:
                trace, _metric_source = build_condition_trace(source_trace, spec, scaling_center=str(args.scaling_center))
                traces_for_patch.append(trace)
                row_keys.append((int(image_row["image_index"]), int(trace_index), int(spec["condition_index"])))
        image_ssi, image_expected, image_rate, image_pop = score_traces_for_patch(
            scorer,
            population_view,
            patch,
            traces_for_patch,
            trace_batch_size=int(args.trace_batch_size),
            frame_batch_size=int(args.frame_batch_size),
            n_timepoints=int(args.n_timepoints),
            bin_seconds=float(args.bin_seconds),
        )
        for local_idx, key in enumerate(row_keys):
            matrix_idx = row_offset + local_idx
            unit_bits[matrix_idx] = image_ssi[local_idx]
            expected[matrix_idx] = image_expected[local_idx]
            mean_rate[matrix_idx] = image_rate[local_idx]
            population[matrix_idx] = image_pop[local_idx]
            metric_row = metric_by_key[key]
            row = {
                "observation_index": int(matrix_idx),
                "population_ssi_bits_per_spike": float(image_pop[local_idx]),
                **{k: metric_row[k] for k in metric_row},
                **{f"patch_{k}": v for k, v in patch_meta.items()},
            }
            observation_rows.append(row)
        row_offset += len(row_keys)
        progress(f"scored image {image_pos + 1}/{n_images}; movies={row_offset}/{n_movies}")

    out_dir = Path(args.out_dir)
    np.savez_compressed(
        out_dir / "retiming_ssi.npz",
        unit_bits_per_movie=unit_bits,
        unit_expected_spikes_per_movie=expected,
        unit_mean_rate_per_movie=mean_rate,
        population_bits_per_movie=population,
        condition_id=np.asarray([str(spec["condition_id"]) for spec in specs]),
        condition_group=np.asarray([str(spec["condition_group"]) for spec in specs]),
    )
    obs_df = pd.DataFrame(observation_rows)
    static = obs_df[obs_df["condition_id"] == "stabilized_static"][
        ["image_index", "trace_index", "population_ssi_bits_per_spike"]
    ].rename(columns={"population_ssi_bits_per_spike": "ssi_stabilized"})
    if not static.empty:
        obs_df = obs_df.merge(static, on=["image_index", "trace_index"], how="left")
        obs_df["ssi_delta_absolute"] = obs_df["population_ssi_bits_per_spike"] - obs_df["ssi_stabilized"]
        obs_df["ssi_delta_percent"] = 100.0 * obs_df["ssi_delta_absolute"] / np.maximum(obs_df["ssi_stabilized"], 1e-8)
    obs_df.to_csv(out_dir / "retiming_population_observations.csv", index=False)
    summary = summarize_population_observations(obs_df)
    summary.to_csv(out_dir / "retiming_population_summary.csv", index=False)
    scored_verification_figures: list[str] = []
    for figure_path in (
        plot_population_summary(summary, out_dir),
        plot_population_velocity_summary(summary, out_dir),
    ):
        if figure_path:
            scored_verification_figures.append(str(figure_path))

    if bool(args.write_unit_observations):
        unit_obs_path = out_dir / "retiming_unit_observations.csv"
        write_unit_observations(unit_obs_path, obs_df, unit_bits, expected, mean_rate, unit_table)
        unit_figure = plot_unit_sf_group_summary(unit_obs_path, out_dir)
        if unit_figure:
            scored_verification_figures.append(str(unit_figure))

    elapsed = time.perf_counter() - started
    return {
        "n_movies": int(n_movies),
        "n_units": int(n_units),
        "requested_device": requested_device,
        "actual_device": actual_device,
        "elapsed_s": float(elapsed),
        "movies_per_s": float(n_movies / elapsed) if elapsed > 0.0 else float("nan"),
        "outputs": {
            "retiming_ssi": out_dir / "retiming_ssi.npz",
            "retiming_population_observations": out_dir / "retiming_population_observations.csv",
            "retiming_population_summary": out_dir / "retiming_population_summary.csv",
            "scored_verification_figures": scored_verification_figures,
        },
    }


def summarize_population_observations(obs_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "condition_index",
        "condition_id",
        "condition_label",
        "condition_group",
        "traversal_frames",
        "traversal_duration_ms",
        "timing_placement",
        "retiming_profile",
        "amplitude_scale",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in obs_df.groupby(group_cols, dropna=False, sort=True):
        values = pd.to_numeric(group["population_ssi_bits_per_spike"], errors="coerce").to_numpy(float)
        delta = pd.to_numeric(group.get("ssi_delta_absolute", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        pct = pd.to_numeric(group.get("ssi_delta_percent", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        char_tf = pd.to_numeric(group["characteristic_motion_tf_hz"], errors="coerce").to_numpy(float)
        finite_char_tf = char_tf[np.isfinite(char_tf)]
        row = {col: val for col, val in zip(group_cols, key, strict=True)}
        row.update(
            {
                "n_observations": int(group.shape[0]),
                "population_ssi_bits_per_spike_mean": float(np.nanmean(values)),
                "population_ssi_bits_per_spike_sem": sem(values),
                "ssi_delta_absolute_mean": float(np.nanmean(delta)) if delta.size else float("nan"),
                "ssi_delta_absolute_sem": sem(delta) if delta.size else float("nan"),
                "ssi_delta_percent_mean": float(np.nanmean(pct)) if pct.size else float("nan"),
                "ssi_delta_percent_sem": sem(pct) if pct.size else float("nan"),
                "rms_across_velocity_deg_s_mean": float(np.nanmean(pd.to_numeric(group["rms_across_velocity_deg_s"], errors="coerce"))),
                "characteristic_motion_tf_hz_max_without_unit_sf": (
                    float(np.nanmax(finite_char_tf)) if finite_char_tf.size else float("nan")
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_population_summary(summary: pd.DataFrame, out_dir: Path) -> str | None:
    if summary.empty or "ssi_delta_absolute_mean" not in summary.columns:
        return None
    retiming = summary[summary["condition_group"] == "retiming"].copy()
    if retiming.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for (profile, placement), group in retiming.groupby(["retiming_profile", "timing_placement"], dropna=False):
        group = group.sort_values("traversal_frames")
        ax.errorbar(
            group["traversal_duration_ms"],
            group["ssi_delta_absolute_mean"],
            yerr=group["ssi_delta_absolute_sem"],
            marker="o",
            linewidth=1.2,
            capsize=2,
            label=f"{profile}/{placement}",
        )
    ax.axhline(0.0, color="#555555", lw=0.8)
    ax.set_xlabel("traversal duration (ms)")
    ax.set_ylabel("population SSI delta vs static (bits/spike)")
    ax.set_title("BackImage RR100 fixed-path retiming pilot")
    ax.grid(True, color="#e6e6e6", lw=0.7)
    ax.legend(fontsize=7.0, frameon=False)
    fig.tight_layout()
    png_path = out_dir / "retiming_population_summary.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "retiming_population_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


def plot_population_velocity_summary(summary: pd.DataFrame, out_dir: Path) -> str | None:
    if summary.empty or "ssi_delta_absolute_mean" not in summary.columns:
        return None
    retiming = summary[summary["condition_group"] == "retiming"].copy()
    retiming = retiming[np.isfinite(pd.to_numeric(retiming["rms_across_velocity_deg_s_mean"], errors="coerce"))]
    if retiming.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for (profile, placement), group in retiming.groupby(["retiming_profile", "timing_placement"], dropna=False):
        group = group.sort_values("rms_across_velocity_deg_s_mean")
        ax.errorbar(
            group["rms_across_velocity_deg_s_mean"],
            group["ssi_delta_absolute_mean"],
            yerr=group["ssi_delta_absolute_sem"],
            marker="o",
            linewidth=1.2,
            capsize=2,
            label=f"{profile}/{placement}",
        )
    ax.axhline(0.0, color="#555555", lw=0.8)
    ax.set_xlabel("RMS across-contour velocity (deg/s)")
    ax.set_ylabel("population SSI delta vs static (bits/spike)")
    ax.set_title("Step 5: population SSI vs velocity")
    ax.grid(True, color="#e6e6e6", lw=0.7)
    ax.legend(fontsize=7.0, frameon=False)
    fig.tight_layout()
    return save_figure(fig, out_dir / "verification_05_population_ssi_vs_velocity.png", dpi=180)


def plot_unit_sf_group_summary(unit_obs_path: Path, out_dir: Path) -> str | None:
    if not Path(unit_obs_path).exists():
        return None
    df = pd.read_csv(unit_obs_path)
    if df.empty or "sf_group" not in df.columns:
        return None
    work = df[df["condition_group"] == "retiming"].copy()
    work = work[work["sf_group"].isin(["low_sf", "high_sf"])].copy()
    if work.empty:
        return None
    group_cols = [
        "sf_group",
        "retiming_profile",
        "timing_placement",
        "traversal_frames",
    ]
    summary = (
        work.groupby(group_cols, dropna=False)
        .agg(
            traversal_duration_ms=("traversal_duration_ms", "mean"),
            rms_across_velocity_deg_s=("rms_across_velocity_deg_s", "mean"),
            unit_ssi_delta_absolute=("unit_ssi_delta_absolute", "mean"),
        )
        .reset_index()
    )
    if summary.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    colors = {"low_sf": "#4c78a8", "high_sf": "#c44e52"}
    for sf_group, sf_rows in summary.groupby("sf_group"):
        for (profile, placement), group in sf_rows.groupby(["retiming_profile", "timing_placement"], dropna=False):
            label = f"{sf_group} {profile}/{placement}"
            group = group.sort_values("traversal_frames")
            axes[0].plot(
                group["traversal_duration_ms"],
                group["unit_ssi_delta_absolute"],
                marker="o",
                lw=1.2,
                color=colors.get(str(sf_group), "#777777"),
                alpha=0.88,
                label=label,
            )
            group_v = group.sort_values("rms_across_velocity_deg_s")
            axes[1].plot(
                group_v["rms_across_velocity_deg_s"],
                group_v["unit_ssi_delta_absolute"],
                marker="o",
                lw=1.2,
                color=colors.get(str(sf_group), "#777777"),
                alpha=0.88,
                label=label,
            )
    axes[0].set_xlabel("traversal duration (ms)")
    axes[1].set_xlabel("RMS across-contour velocity (deg/s)")
    axes[0].set_ylabel("mean unit SSI delta vs static")
    axes[0].set_title("Low/high SF vs duration")
    axes[1].set_title("Low/high SF vs velocity")
    for ax in axes:
        ax.axhline(0.0, color="#555555", lw=0.8)
        ax.grid(True, color="#e6e6e6", lw=0.7)
    axes[1].legend(fontsize=6.5, frameon=False)
    fig.suptitle("Step 6: unit-group mechanistic check", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, out_dir / "verification_06_unit_sf_group_curves.png", dpi=180)


def write_unit_observations(
    path: Path,
    obs_df: pd.DataFrame,
    unit_bits: np.ndarray,
    expected: np.ndarray,
    mean_rate: np.ndarray,
    unit_table: pd.DataFrame,
) -> None:
    rows: list[dict[str, Any]] = []
    unit_meta = unit_table.set_index("unit_index").to_dict("index")
    static_lookup: dict[tuple[int, int, int], float] = {}
    for _row_pos, row in obs_df.iterrows():
        if str(row["condition_id"]) != "stabilized_static":
            continue
        obs_idx = int(row["observation_index"])
        for unit_idx in range(unit_bits.shape[1]):
            static_lookup[(int(row["image_index"]), int(row["trace_index"]), int(unit_idx))] = float(unit_bits[obs_idx, unit_idx])
    for _row_pos, row in obs_df.iterrows():
        obs_idx = int(row["observation_index"])
        for unit_idx in range(unit_bits.shape[1]):
            meta = unit_meta.get(int(unit_idx), {})
            sf = finite_float(meta.get("preferred_sf_cpd"))
            across = finite_float(row.get("rms_across_velocity_deg_s"), 0.0)
            char_tf = sf * across if math.isfinite(sf) else float("nan")
            baseline = static_lookup.get((int(row["image_index"]), int(row["trace_index"]), int(unit_idx)), float("nan"))
            value = float(unit_bits[int(obs_idx), unit_idx])
            rows.append(
                {
                    "observation_index": obs_idx,
                    "image_index": int(row["image_index"]),
                    "trace_index": int(row["trace_index"]),
                    "condition_index": int(row["condition_index"]),
                    "condition_id": str(row["condition_id"]),
                    "condition_group": str(row["condition_group"]),
                    "traversal_frames": int(row["traversal_frames"]),
                    "traversal_duration_ms": float(row["traversal_duration_ms"]),
                    "timing_placement": str(row["timing_placement"]),
                    "retiming_profile": str(row["retiming_profile"]),
                    "amplitude_scale": float(row["amplitude_scale"]),
                    "unit_index": int(unit_idx),
                    "unit_label": str(meta.get("unit_label", f"u{unit_idx:03d}")),
                    "sf_group": str(meta.get("sf_group", "")),
                    "preferred_sf_cpd": sf,
                    "preferred_sf_source_column": str(meta.get("preferred_sf_source_column", "")),
                    "rms_across_velocity_deg_s": across,
                    "characteristic_motion_tf_hz": char_tf,
                    "nyquist_margin_hz": 60.0 - char_tf if math.isfinite(char_tf) else float("nan"),
                    "exceeds_model_nyquist": bool(char_tf >= 60.0) if math.isfinite(char_tf) else False,
                    "unit_ssi_bits_per_spike": value,
                    "unit_ssi_stabilized": baseline,
                    "unit_ssi_delta_absolute": value - baseline if math.isfinite(baseline) else float("nan"),
                    "unit_expected_spikes": float(expected[int(obs_idx), unit_idx]),
                    "unit_mean_rate": float(mean_rate[int(obs_idx), unit_idx]),
                }
            )
    write_csv(path, rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.force):
        raise FileExistsError(f"{out_dir} already exists and is not empty. Pass --force to overwrite pilot files.")
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = condition_specs(args)
    image_table, traces, trace_bank, selection_meta = select_inputs(args)
    image_table.to_csv(out_dir / "image_feature_table.csv", index=False)
    write_csv(out_dir / "trace_feature_table.csv", trace_feature_rows(traces, scale_metric=str(args.trace_scale_metric)))
    write_csv(out_dir / "trace_bank_metric_summary.csv", trace_bank_metric_summary_rows(trace_feature_rows(traces, scale_metric=str(args.trace_scale_metric))))
    write_csv(out_dir / "condition_table.csv", specs)

    metric_rows, retiming_qc = build_metric_rows(image_table, traces, specs, args)
    write_csv(out_dir / "retimed_trajectory_metrics.csv", metric_rows)
    verification_figures = write_verification_figures(out_dir, image_table, traces, specs, metric_rows, args)
    qc_plot_paths = plot_qc_traces(out_dir, traces, specs, args)

    scoring_meta: dict[str, Any] | None = None
    if not bool(args.dry_run):
        progress("starting RR100 twin scoring")
        scoring_meta = score_counterfactuals(image_table, traces, specs, args)
    else:
        population_view = load_population_view(version_name=str(args.rr100_version))
        write_unit_feature_table(out_dir / "unit_feature_table.csv", Path(args.unit_tuning_csv), int(population_view.n_units))

    summary = {
        "analysis": "backimage_temporal_remapping_pilot",
        "mode": "dry_run" if bool(args.dry_run) else "scored",
        "source_csv": Path(args.source_csv),
        "unit_tuning_csv": Path(args.unit_tuning_csv),
        "out_dir": out_dir,
        "rr100_version": str(args.rr100_version),
        "n_images": int(image_table.shape[0]),
        "n_traces": int(len(traces)),
        "n_conditions": int(len(specs)),
        "n_trajectory_metric_rows": int(len(metric_rows)),
        "n_timepoints": int(args.n_timepoints),
        "bin_seconds": float(args.bin_seconds),
        "trajectory_qc": retiming_qc,
        "verification_figures": verification_figures,
        "qc_trace_plots": qc_plot_paths,
        "selection": selection_meta,
        "scoring": scoring_meta,
        "outputs": {
            "condition_table": out_dir / "condition_table.csv",
            "image_feature_table": out_dir / "image_feature_table.csv",
            "trace_feature_table": out_dir / "trace_feature_table.csv",
            "unit_feature_table": out_dir / "unit_feature_table.csv",
            "retimed_trajectory_metrics": out_dir / "retimed_trajectory_metrics.csv",
        },
        "contract": (
            "Fixed-geometry retiming counterfactuals use endpoint-inclusive traversal samples. "
            "Continuous path geometry is held fixed within a retiming family; model-sampled path metrics "
            "are reported separately because short traversals sample fewer points at 120 Hz."
        ),
    }
    write_json(out_dir / "summary.json", summary)
    progress(f"wrote temporal-remapping pilot outputs to {out_dir}")


if __name__ == "__main__":
    main()
