#!/usr/bin/env python3
"""Compute RR100 spatial SSI for BackImage contour-axis motion sweeps.

This runner is the BackImage counterpart of the recent Vernier RR100 unit-SSI
plots, but uses the audited BackImage axis-conditioned cache as its source of
window identities and measured traces.  The promoted condition family is not a
pure along-vs-across observer prior; it is a combined retinal trajectory:

    trace = along_scale * along_component + across_scale * across_component

where the components are computed from the saved scale-1 observed trajectory
relative to each selected window's local edge axis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
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

from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
    _align_response_to_trace,
)
from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import (
    _lag1_autocorr,
    _microsaccade_stats,
    _path_length,
    _resample_trace,
    _scale_to_rms,
    _session_dataset_cache,
    _trace_covariance_anisotropy,
    _trace_covariance_shape,
)
from declan.fixation_statistics_by_stimulus.features import fixation_window_features
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _standardize_uint_like,
    _trace_xy_to_twin_helper_order,
)
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view
from jake.twininfo.eye_controls import detect_microsaccade_events, speed_threshold_mad


RR100_MOVIE_MEDOID_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)
DEFAULT_AXIS_RUN_DIR = (
    ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_axis_conditioned_matched_static_percandidate_gpu1_n128_c4_k16_scales_0p5_1_2_bconsistent_v1"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "active_sensing_movie_information"
    / "backimage_contour_axis_rr100_spatial_ssi_smoke"
)
DEFAULT_ACROSS_SCALES = "0,0.5,1"
STIMULUS_NORMALIZATION = "standardize_uint_like_then_minus_127_div_255"
CACHE_SCHEMA_VERSION = 3
EPS = 1e-8
TRACE_BANK_DEFAULT_SCALE_METRIC = "rendered_diffusion_constant_deg2_s"
TRACE_BANK_DEFAULT_BINS = "quantile:4"
TRACE_BANK_SOURCE_METRIC_COLUMNS = [
    "rms_radius_deg",
    "median_radius_deg",
    "max_radius_deg",
    "path_length_deg",
    "path_length_deg_s",
    "step_mean_deg",
    "step_median_deg",
    "step_p95_deg",
    "speed_mean_deg_s",
    "speed_median_deg_s",
    "speed_p95_deg_s",
    "diffusion_constant_deg2_s",
    "msd_lag1_deg2",
    "msd_lag2_deg2",
    "msd_lag4_deg2",
    "msd_lag8_deg2",
    "msd_lag16_deg2",
    "anisotropy",
    "direction_persistence",
    "curvature_rad",
    "return_to_center_strength",
    "position_autocorr_lag1",
    "velocity_autocorr_lag1",
]
TRACE_BANK_METADATA_NUMERIC_COLUMNS = [
    "observed_rms_deg",
    "observed_rms_arcmin",
    "rendered_rms_radius_deg",
    "rendered_rms_radius_arcmin",
    "rendered_max_radius_deg",
    "path_length_deg",
    "path_length_arcmin",
    "rendered_path_length_deg",
    "rendered_path_length_arcmin",
    "rendered_path_length_deg_s",
    "rendered_path_speed_arcmin_s",
    "rendered_speed_mean_deg_s",
    "rendered_speed_mean_arcmin_s",
    "rendered_speed_median_deg_s",
    "rendered_speed_median_arcmin_s",
    "rendered_speed_p95_deg_s",
    "rendered_speed_p95_arcmin_s",
    "rendered_diffusion_constant_deg2_s",
    "rendered_diffusion_constant_arcmin2_s",
    "rendered_position_autocorr_lag1",
    "rendered_velocity_autocorr_lag1",
    "lag1_autocorr",
    "source_trace_observed_rms_deg",
    "source_rms_radius_deg",
    "source_rms_radius_arcmin",
    "source_max_radius_deg",
    "source_path_length_deg",
    "source_path_length_arcmin",
    "source_path_length_deg_s",
    "source_path_speed_arcmin_s",
    "source_speed_mean_deg_s",
    "source_speed_mean_arcmin_s",
    "source_speed_median_deg_s",
    "source_speed_median_arcmin_s",
    "source_speed_p95_deg_s",
    "source_speed_p95_arcmin_s",
    "source_diffusion_constant_deg2_s",
    "source_diffusion_constant_arcmin2_s",
    "source_rendered_diffusion_delta_deg2_s",
    "source_rendered_diffusion_abs_delta_deg2_s",
    "trace_cov_anisotropy",
    "source_trace_cov_anisotropy",
    "source_anisotropy",
    "rendered_anisotropy",
    "source_cov_major_sd_arcmin",
    "source_cov_minor_sd_arcmin",
    "source_cov_axis_ratio",
    "source_cov_orientation_deg",
    "source_bcea68_arcmin2",
    "rendered_cov_major_sd_arcmin",
    "rendered_cov_minor_sd_arcmin",
    "rendered_cov_axis_ratio",
    "rendered_cov_orientation_deg",
    "rendered_bcea68_arcmin2",
    "trace_cov_shape_xx",
    "trace_cov_shape_xy",
    "trace_cov_shape_yy",
    "microsaccade_threshold_dps",
    "n_microsaccade_events",
    "fraction_microsaccade_samples",
    "peak_microsaccade_speed_dps",
    "source_microsaccade_threshold_dps",
    "source_n_microsaccade_events",
    "source_fraction_microsaccade_samples",
    "source_peak_microsaccade_speed_dps",
    "rendered_microsaccade_threshold_dps",
    "rendered_n_microsaccade_events",
    "rendered_fraction_microsaccade_samples",
    "rendered_peak_microsaccade_speed_dps",
]
TRACE_BANK_METRIC_SUMMARY_SPECS = [
    ("path_length_arcmin", "path length", "arcmin"),
    ("rendered_path_speed_arcmin_s", "path speed", "arcmin/s"),
    ("observed_rms_arcmin", "RMS radius", "arcmin"),
    ("rendered_bcea68_arcmin2", "BCEA68", "arcmin^2"),
    ("trace_cov_anisotropy", "covariance anisotropy", "unitless"),
    ("rendered_cov_axis_ratio", "covariance axis ratio", "unitless"),
    ("rendered_speed_p95_arcmin_s", "p95 speed", "arcmin/s"),
    ("rendered_diffusion_constant_arcmin2_s", "MSD diffusion constant", "arcmin^2/s"),
    ("lag1_autocorr", "lag-1 position autocorrelation", "unitless"),
    ("n_microsaccade_events", "microsaccade event count", "events/snippet"),
    ("fraction_microsaccade_samples", "microsaccade sample fraction", "fraction"),
    ("peak_microsaccade_speed_dps", "peak microsaccade speed", "deg/s"),
]
EVENT_SCALED_MICROSACCADE_TRACE_MODES = frozenset(
    {
        "core_scaled_full_snippet",
        "padded_event_scaled_full_snippet",
        "core_scaled_full_snippet_start_anchor",
        "padded_event_scaled_full_snippet_start_anchor",
        "core_scaled_full_snippet_end_anchor",
        "padded_event_scaled_full_snippet_end_anchor",
    }
)
CORE_EVENT_SCALED_MICROSACCADE_TRACE_MODES = frozenset(
    {"core_scaled_full_snippet", "core_scaled_full_snippet_start_anchor", "core_scaled_full_snippet_end_anchor"}
)
PADDED_EVENT_SCALED_MICROSACCADE_TRACE_MODES = frozenset(
    {
        "padded_event_scaled_full_snippet",
        "padded_event_scaled_full_snippet_start_anchor",
        "padded_event_scaled_full_snippet_end_anchor",
    }
)
WINDOW_INVENTORY_COLUMNS = [
    "parent_source_row",
    "source_window_global_start",
    "source_window_global_stop",
    "snippet_global_start",
    "snippet_global_stop",
    "snippet_n_samples",
    "snippet_duration_s",
    "microsaccade_event_index",
    "microsaccade_event_onset_global",
    "microsaccade_event_offset_global",
    "microsaccade_event_peak_global",
    "microsaccade_event_onset_global_padded",
    "microsaccade_event_offset_global_padded",
    "microsaccade_event_onset_frame_in_snippet",
    "microsaccade_event_offset_frame_in_snippet",
    "microsaccade_event_onset_frame_resampled",
    "microsaccade_event_offset_frame_resampled",
    "microsaccade_event_duration_samples",
    "microsaccade_event_padded_duration_samples",
    "microsaccade_trace_mode",
    "microsaccade_trace_keep_onset_frame_in_snippet",
    "microsaccade_trace_keep_offset_frame_in_snippet",
    "microsaccade_trace_keep_onset_frame_resampled",
    "microsaccade_trace_keep_offset_frame_resampled",
    "microsaccade_trace_nonzero_resampled_frames",
    "microsaccade_trace_event_scaled_sample_count",
    "microsaccade_trace_event_scaled_step_count",
    "microsaccade_trace_event_anchor_mode",
    "microsaccade_trace_centering",
    "microsaccade_trace_contract",
    "microsaccade_pre_frames",
    "microsaccade_post_frames",
    "microsaccade_amplitude_deg",
    "microsaccade_amplitude_arcmin",
    "microsaccade_amplitude_start_global",
    "microsaccade_amplitude_stop_global",
    "microsaccade_direction_deg",
    "microsaccade_peak_speed_dps",
    "microsaccade_threshold_dps",
    "microsaccade_detection_threshold_source",
    "microsaccade_dedup_peak_tolerance_frames",
    "microsaccade_dedup_cluster_size",
    "microsaccade_dedup_source_margin_frames",
    "snippet_detected_core_event_count",
    "snippet_detected_padded_event_count",
    "snippet_detected_event_count",
    "snippet_raw_rms_deg",
    "snippet_raw_max_radius_deg",
    "snippet_raw_path_length_deg",
    "balanced_manifest_index",
    "axis_balance_deg",
    "axis_balance_bin",
    "axis_balance_bin_start_deg",
    "axis_balance_bin_stop_deg",
    "energy_balance_column",
    "energy_balance_value",
    "energy_balance_bin",
    "energy_balance_quantile_bins",
    "image_patch_rms_contrast",
    "image_patch_std",
    "image_gradient_energy",
    "image_orientation_coherence",
    "image_oriented_gradient_energy",
    "image_multi_orientation_energy",
    "image_edge_density",
    "image_spectrum_anisotropy",
    "image_abs_8plus_power_proxy",
    "image_oriented_8plus_power_proxy",
    "image_high_freq_power_fraction",
    "image_power_8plus_cpd_fraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-run-dir", type=Path, default=DEFAULT_AXIS_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--across-scales", type=str, default=DEFAULT_ACROSS_SCALES)
    parser.add_argument("--along-scales", type=str, default="")
    parser.add_argument("--along-scale", type=float, default=1.0)
    parser.add_argument(
        "--sweep-mode",
        choices=("across", "isotropic", "grid", "pairs", "trace_bank"),
        default="across",
        help=(
            "across: hold along-scale fixed and sweep across-scale. "
            "isotropic: use --across-scales as total motion scales with "
            "along_scale=across_scale=scale. grid: fully cross --along-scales "
            "with --across-scales. pairs: use explicit --condition-pairs. "
            "trace_bank: sample native real BackImage trace snippets from metric bins; "
            "--across-scales is ignored except for legacy metadata."
        ),
    )
    parser.add_argument(
        "--trace-bank-scale-metric",
        type=str,
        default=TRACE_BANK_DEFAULT_SCALE_METRIC,
        help=(
            "For --sweep-mode trace_bank, metric used to bin native real traces. "
            "Useful values include rendered_diffusion_constant_deg2_s, "
            "rendered_rms_radius_deg, rendered_path_length_deg, "
            "rendered_speed_p95_deg_s, observed_rms_deg, path_length_deg, and "
            "source_diffusion_constant_deg2_s."
        ),
    )
    parser.add_argument(
        "--trace-bank-bins",
        type=str,
        default=TRACE_BANK_DEFAULT_BINS,
        help=(
            "For --sweep-mode trace_bank, either quantile:N for equal-count native "
            "trace bins or comma-separated numeric bin edges in the selected metric."
        ),
    )
    parser.add_argument(
        "--trace-bank-samples-per-bin",
        type=int,
        default=1,
        help="For --sweep-mode trace_bank, number of independently sampled real traces per metric bin.",
    )
    parser.add_argument(
        "--trace-bank-seed",
        type=int,
        default=0,
        help="For --sweep-mode trace_bank, seed for image x metric-bin trace assignments.",
    )
    parser.add_argument(
        "--trace-bank-exclude-same-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For --sweep-mode trace_bank, avoid assigning an image its own recorded source trace.",
    )
    parser.add_argument(
        "--trace-bank-max-source-windows",
        type=int,
        default=0,
        help="For --sweep-mode trace_bank, scan at most this many source rows for the trace bank. 0 means all.",
    )
    parser.add_argument("--trace-bank-max-rms-deg", type=float, default=0.0)
    parser.add_argument("--trace-bank-max-radius-deg", type=float, default=0.0)
    parser.add_argument("--trace-bank-max-path-length-deg", type=float, default=0.0)
    parser.add_argument("--trace-bank-max-speed-p95-deg-s", type=float, default=0.0)
    parser.add_argument(
        "--trace-bank-max-microsaccade-events",
        type=int,
        default=-1,
        help="For --sweep-mode trace_bank, optional max detected event count; -1 disables.",
    )
    parser.add_argument(
        "--condition-pairs",
        type=str,
        default="",
        help="For --sweep-mode pairs, comma-separated along:across scale pairs, e.g. 0:0.25,0.5:1.",
    )
    parser.add_argument("--source-trace-scale", type=float, default=1.0)
    parser.add_argument("--source-trace-prior-family", type=str, default="axis_edge_parallel")
    parser.add_argument("--axis-column", type=str, default="image_edge_axis_deg")
    parser.add_argument(
        "--trial-source-mode",
        choices=("auto", "manifest", "selected_windows", "microsaccade_snippets"),
        default="auto",
        help=(
            "manifest reuses an axis-observer response_cache_manifest/candidate_sets table. "
            "selected_windows reconstructs traces directly from selected_windows.csv, which is "
            "the intended mode for large balanced contour-SSI batches. "
            "microsaccade_snippets detects real microsaccade events in source windows and "
            "uses event-aligned pre/event/post snippets as the source traces."
        ),
    )
    parser.add_argument(
        "--selected-windows-csv",
        type=Path,
        default=None,
        help="Optional selected_windows.csv override for --trial-source-mode selected_windows.",
    )
    parser.add_argument("--max-trials", type=int, default=4, help="0 means all eligible source windows.")
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument(
        "--microsaccade-pre-frames",
        type=int,
        default=8,
        help="For --trial-source-mode microsaccade_snippets, source frames retained before detected event onset.",
    )
    parser.add_argument(
        "--microsaccade-post-frames",
        type=int,
        default=36,
        help="For --trial-source-mode microsaccade_snippets, source frames retained after detected event offset.",
    )
    parser.add_argument(
        "--microsaccade-max-source-windows",
        type=int,
        default=0,
        help="For --trial-source-mode microsaccade_snippets, scan at most this many source rows before event selection. 0 means all.",
    )
    parser.add_argument(
        "--microsaccade-min-amplitude-arcmin",
        type=float,
        default=0.0,
        help="For --trial-source-mode microsaccade_snippets, reject detected events below this amplitude. 0 disables.",
    )
    parser.add_argument(
        "--microsaccade-max-amplitude-arcmin",
        type=float,
        default=60.0,
        help="For --trial-source-mode microsaccade_snippets, reject detected events above this amplitude. 0 disables.",
    )
    parser.add_argument(
        "--microsaccade-amplitude-sd-filter",
        type=float,
        default=0.0,
        help=(
            "For --trial-source-mode microsaccade_snippets, after QC and deduplication keep only events "
            "within mean +/- this many SD of microsaccade_amplitude_arcmin. 0 disables."
        ),
    )
    parser.add_argument(
        "--microsaccade-trace-mode",
        choices=(
            "full_snippet",
            "core_zero_rest",
            "padded_event_zero_rest",
            "core_scaled_full_snippet",
            "padded_event_scaled_full_snippet",
            "core_scaled_full_snippet_start_anchor",
            "padded_event_scaled_full_snippet_start_anchor",
            "core_scaled_full_snippet_end_anchor",
            "padded_event_scaled_full_snippet_end_anchor",
        ),
        default="full_snippet",
        help=(
            "For --trial-source-mode microsaccade_snippets, full_snippet uses the whole pre/event/post "
            "eye trace. *_zero_rest keeps only the detected core or padded event pulse and sets all "
            "pre/post drift samples to zero motion. *_scaled_full_snippet keeps the original full "
            "snippet but applies condition scales only to the detected event interval while retaining "
            "pre/post drift at 1x. *_start_anchor and *_end_anchor are strict event-scaled controls "
            "that preserve the starting or ending local trace position instead of recentering each "
            "scaled condition."
        ),
    )
    parser.add_argument(
        "--microsaccade-require-snippet-within-source-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For --trial-source-mode microsaccade_snippets, require the full pre/event/post "
            "snippet to lie inside the source BackImage window that identified the event."
        ),
    )
    parser.add_argument(
        "--microsaccade-reject-extra-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For --trial-source-mode microsaccade_snippets, reject snippets containing another detected event.",
    )
    parser.add_argument(
        "--microsaccade-max-snippet-rms-deg",
        type=float,
        default=0.0,
        help="For --trial-source-mode microsaccade_snippets, optional centered raw-snippet RMS limit. 0 disables.",
    )
    parser.add_argument(
        "--microsaccade-max-snippet-radius-deg",
        type=float,
        default=0.0,
        help="For --trial-source-mode microsaccade_snippets, optional centered raw-snippet max-radius limit. 0 disables.",
    )
    parser.add_argument(
        "--microsaccade-max-snippet-path-length-deg",
        type=float,
        default=0.0,
        help="For --trial-source-mode microsaccade_snippets, optional raw-snippet path-length limit. 0 disables.",
    )
    parser.add_argument(
        "--microsaccade-speed-threshold-dps",
        type=float,
        default=None,
        help="Fixed microsaccade speed threshold. Defaults to robust MAD threshold per source window.",
    )
    parser.add_argument("--microsaccade-threshold-z", type=float, default=6.0)
    parser.add_argument("--microsaccade-pad-frames", type=int, default=1)
    parser.add_argument(
        "--microsaccade-dedup-tolerance-frames",
        type=int,
        default=3,
        help=(
            "For --trial-source-mode microsaccade_snippets, collapse events from overlapping "
            "source windows when their unpadded peak-speed samples are within this many frames."
        ),
    )
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument(
        "--stimulus-rotation-deg",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help=(
            "Rotate the extracted patch, source trace, and analysis contour axis together by this "
            "counter-clockwise gaze-frame angle before scoring. This is intended for whole-movie "
            "rotational-anisotropy controls."
        ),
    )
    parser.add_argument("--bin-seconds", type=float, default=None)
    parser.add_argument(
        "--primary-ssi-metric",
        choices=("mean_map", "time_resolved"),
        default="time_resolved",
        help=(
            "Primary SSI contract. time_resolved computes per-frame spatial SSI and averages it "
            "with expected-spike weights over the trajectory. mean_map computes SSI after "
            "trajectory-averaging activation maps and should be treated as an activation-map diagnostic."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-units", type=int, default=12)
    parser.add_argument("--map-vmin-percentile", type=float, default=0.5)
    parser.add_argument("--map-vmax-percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--write-zscore-plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--zscore-min-unit-std",
        type=float,
        default=1e-4,
        help="Drop units from the automatic z-scored curve plot below this absolute SSI std.",
    )
    parser.add_argument("--include-static-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write selected trial/condition inventory without loading the twin.")
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one scale is required.")
    return values


def parse_condition_pairs(text: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Condition pair {token!r} must use 'along:across' syntax.")
        along_text, across_text = token.split(":", 1)
        pairs.append((float(along_text.strip()), float(across_text.strip())))
    if not pairs:
        raise ValueError("--condition-pairs must contain at least one along:across pair.")
    return pairs


def scale_token(value: float) -> str:
    text = f"{float(value):.9g}".replace("-", "m").replace(".", "p")
    return text


def safe_slug(text: object) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text))
    return "_".join(part for part in slug.split("_") if part) or "unnamed"


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


def window_inventory_payload(row: pd.Series) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in WINDOW_INVENTORY_COLUMNS:
        if key not in row.index:
            continue
        value = row[key]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            payload[key] = value if math.isfinite(value) else None
        else:
            payload[key] = value
    return payload


def identity_text(identity: dict[str, Any]) -> str:
    return json.dumps(json_ready(identity), sort_keys=True, separators=(",", ":"))


def cache_path(out_dir: Path) -> Path:
    return out_dir / "cache" / "backimage_contour_axis_rr100_spatial_ssi_cache.npz"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    n = x.shape[axis]
    if n <= 1:
        return np.zeros_like(np.nanmean(x, axis=axis), dtype=np.float64)
    return np.nanstd(x, axis=axis, ddof=1) / math.sqrt(float(n))


def trace_rms(trace: np.ndarray) -> float:
    arr = np.asarray(trace, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(arr * arr, axis=1)))) if arr.size else 0.0


def trace_path_length(trace: np.ndarray) -> float:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def trace_hash(trace: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(trace, dtype=np.float32))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()[:20]


def trace_bank_metric_slug(metric: str) -> str:
    return safe_slug(str(metric).replace("_deg2_s", "").replace("_deg_s", "").replace("_deg", ""))


def trace_scale_metrics(trace: np.ndarray, *, dt: float, prefix: str) -> dict[str, float]:
    """Literature-style compact motion metrics for the exact snippet sent to the twin."""
    try:
        metrics = fixation_window_features(np.asarray(trace, dtype=np.float64), dt=float(dt))
    except Exception:
        metrics = {}
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            out[f"{prefix}{key}"] = float(value)
    d_key = f"{prefix}diffusion_constant_deg2_s"
    if d_key in out and math.isfinite(float(out[d_key])):
        out[f"{prefix}diffusion_constant_arcmin2_s"] = float(out[d_key]) * 3600.0
    rms_key = f"{prefix}rms_radius_deg"
    if rms_key in out and math.isfinite(float(out[rms_key])):
        out[f"{prefix}rms_radius_arcmin"] = float(out[rms_key]) * 60.0
    path_key = f"{prefix}path_length_deg"
    if path_key in out and math.isfinite(float(out[path_key])):
        out[f"{prefix}path_length_arcmin"] = float(out[path_key]) * 60.0
    return out


def source_trace_metric_payload(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key in TRACE_BANK_SOURCE_METRIC_COLUMNS:
        value = row.get(key, np.nan) if isinstance(row, dict) else row.get(key, np.nan)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            payload[f"source_{key}"] = number
    if "source_diffusion_constant_deg2_s" in payload:
        payload["source_diffusion_constant_arcmin2_s"] = payload["source_diffusion_constant_deg2_s"] * 3600.0
    if "source_rms_radius_deg" in payload:
        payload["source_rms_radius_arcmin"] = payload["source_rms_radius_deg"] * 60.0
    if "source_path_length_deg" in payload:
        payload["source_path_length_arcmin"] = payload["source_path_length_deg"] * 60.0
    return payload


def build_native_snippet_trace_bank(
    rows: pd.DataFrame,
    eyepos_by_session: dict[str, np.ndarray],
    n_timepoints: int,
    *,
    dt: float,
    microsaccade_speed_threshold_dps: float | None,
    microsaccade_threshold_z: float,
    microsaccade_pad_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build scale-1 trace snippets without temporal compression."""
    bank: list[dict[str, Any]] = []
    n_short = 0
    n_bad = 0
    n_timepoints = int(n_timepoints)
    if n_timepoints < 2:
        raise ValueError("--n-timepoints must be at least 2 for native trace-bank snippets.")

    for _, row in rows.iterrows():
        session = str(row["session"])
        eyepos = np.asarray(eyepos_by_session[session], dtype=np.float64)
        window_start = int(row["global_start"])
        window_stop = int(row["global_stop"])
        window_start = max(0, min(window_start, int(eyepos.shape[0])))
        window_stop = max(0, min(window_stop, int(eyepos.shape[0])))
        n_available = int(window_stop - window_start)
        if n_available < n_timepoints:
            n_short += 1
            continue

        snippet_offset = int((n_available - n_timepoints) // 2)
        snippet_start = int(window_start + snippet_offset)
        snippet_stop = int(snippet_start + n_timepoints)
        raw = np.asarray(eyepos[snippet_start:snippet_stop], dtype=np.float64)
        if raw.ndim != 2 or raw.shape != (n_timepoints, 2):
            n_bad += 1
            continue

        trace = np.asarray(raw, dtype=np.float64).copy()
        finite = np.isfinite(trace).all(axis=1)
        if not np.all(finite):
            good = np.flatnonzero(finite)
            if good.size == 0:
                trace = np.zeros_like(trace)
            else:
                bad = np.flatnonzero(~finite)
                for dim in range(2):
                    trace[bad, dim] = np.interp(bad, good, trace[good, dim])
        trace -= np.mean(trace, axis=0, keepdims=True)
        trace = trace.astype(np.float32)
        ms = _microsaccade_stats(
            trace,
            dt=float(dt),
            threshold_dps=microsaccade_speed_threshold_dps,
            threshold_z=float(microsaccade_threshold_z),
            pad_frames=int(microsaccade_pad_frames),
        )
        metrics = {
            **trace_scale_metrics(trace, dt=float(dt), prefix="source_"),
            **trace_scale_metrics(trace, dt=float(dt), prefix="rendered_"),
        }
        snippet_duration_s = float((n_timepoints - 1) * float(dt))
        source_window_duration_s = float(row.get("duration_s", np.nan))
        if not math.isfinite(source_window_duration_s):
            source_window_duration_s = float(n_available - 1) * float(dt)
        item: dict[str, Any] = {
            "source_row": int(row["source_row"]),
            "session": session,
            "trial_idx": int(row.get("trial_idx", -1)),
            "global_start": int(snippet_start),
            "global_stop": int(snippet_stop),
            "source_window_global_start": int(window_start),
            "source_window_global_stop": int(window_stop),
            "snippet_global_start": int(snippet_start),
            "snippet_global_stop": int(snippet_stop),
            "snippet_n_samples": int(n_timepoints),
            "snippet_duration_s": snippet_duration_s,
            "source_window_n_samples": int(n_available),
            "source_window_duration_s": source_window_duration_s,
            "mean_x_deg": float(np.nanmean(raw[:, 0])),
            "mean_y_deg": float(np.nanmean(raw[:, 1])),
            "trace": trace,
            "observed_rms_deg": float(trace_rms(trace)),
            "source_trace_observed_rms_deg": float(trace_rms(trace)),
            "path_length_deg": _path_length(trace),
            "duration_s": snippet_duration_s,
            "lag1_autocorr": _lag1_autocorr(trace),
            "covariance_shape": _trace_covariance_shape(trace),
            "trace_cov_anisotropy": _trace_covariance_anisotropy(trace),
            "source_trace_cov_anisotropy": _trace_covariance_anisotropy(trace),
            "source_anisotropy": _trace_covariance_anisotropy(trace),
            "trace_bank_snippet_policy": "center_crop_native_n_timepoints",
        }
        item.update(metrics)
        item.update(
            {
                "source_microsaccade_threshold_dps": float(ms["microsaccade_threshold_dps"]),
                "source_n_microsaccade_events": int(ms["n_microsaccade_events"]),
                "source_fraction_microsaccade_samples": float(ms["fraction_microsaccade_samples"]),
                "source_peak_microsaccade_speed_dps": float(ms["peak_microsaccade_speed_dps"]),
                "rendered_microsaccade_threshold_dps": float(ms["microsaccade_threshold_dps"]),
                "rendered_n_microsaccade_events": int(ms["n_microsaccade_events"]),
                "rendered_fraction_microsaccade_samples": float(ms["fraction_microsaccade_samples"]),
                "rendered_peak_microsaccade_speed_dps": float(ms["peak_microsaccade_speed_dps"]),
                "microsaccade_threshold_dps": float(ms["microsaccade_threshold_dps"]),
                "n_microsaccade_events": int(ms["n_microsaccade_events"]),
                "fraction_microsaccade_samples": float(ms["fraction_microsaccade_samples"]),
                "peak_microsaccade_speed_dps": float(ms["peak_microsaccade_speed_dps"]),
            }
        )
        bank.append(item)

    meta = {
        "trace_bank_snippet_policy": "center_crop_native_n_timepoints",
        "trace_bank_native_dt_s": float(dt),
        "trace_bank_native_snippet_n_timepoints": int(n_timepoints),
        "n_trace_bank_source_windows_skipped_short": int(n_short),
        "n_trace_bank_source_windows_skipped_bad_shape": int(n_bad),
    }
    return bank, meta


def trace_metric_value(item: dict[str, Any], metric: str) -> float:
    key = str(metric)
    aliases = {
        "diffusion_constant_deg2_s": "rendered_diffusion_constant_deg2_s",
        "diffusion_constant_arcmin2_s": "rendered_diffusion_constant_arcmin2_s",
        "rms_radius_deg": "rendered_rms_radius_deg",
        "rms_radius_arcmin": "rendered_rms_radius_arcmin",
        "path_length_arcmin": "rendered_path_length_arcmin",
        "speed_p95_deg_s": "rendered_speed_p95_deg_s",
        "observed_rms_arcmin": "observed_rms_arcmin",
    }
    candidates = [key]
    if key in aliases:
        candidates.append(aliases[key])
    if not key.startswith("rendered_"):
        candidates.append(f"rendered_{key}")
    if not key.startswith("source_"):
        candidates.append(f"source_{key}")
    for candidate in candidates:
        if candidate not in item:
            continue
        try:
            value = float(item[candidate])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return float("nan")


def _trace_bank_over_limit(value: object, limit: float) -> bool:
    lim = float(limit)
    if lim <= 0.0:
        return False
    try:
        val = float(value)
    except (TypeError, ValueError):
        return True
    return (not math.isfinite(val)) or val > lim


def filter_trace_bank_items(trace_bank: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[int], dict[str, Any]]:
    kept: list[int] = []
    rejected = {
        "trace_bank_rejected_rms": 0,
        "trace_bank_rejected_radius": 0,
        "trace_bank_rejected_path_length": 0,
        "trace_bank_rejected_speed_p95": 0,
        "trace_bank_rejected_microsaccade_events": 0,
    }
    max_events = int(args.trace_bank_max_microsaccade_events)
    for idx, item in enumerate(trace_bank):
        if _trace_bank_over_limit(item.get("rendered_rms_radius_deg", item.get("observed_rms_deg")), args.trace_bank_max_rms_deg):
            rejected["trace_bank_rejected_rms"] += 1
            continue
        if _trace_bank_over_limit(item.get("rendered_max_radius_deg", item.get("source_max_radius_deg")), args.trace_bank_max_radius_deg):
            rejected["trace_bank_rejected_radius"] += 1
            continue
        if _trace_bank_over_limit(item.get("rendered_path_length_deg", item.get("path_length_deg")), args.trace_bank_max_path_length_deg):
            rejected["trace_bank_rejected_path_length"] += 1
            continue
        if _trace_bank_over_limit(item.get("rendered_speed_p95_deg_s", item.get("source_speed_p95_deg_s")), args.trace_bank_max_speed_p95_deg_s):
            rejected["trace_bank_rejected_speed_p95"] += 1
            continue
        if max_events >= 0 and int(item.get("n_microsaccade_events", 0)) > max_events:
            rejected["trace_bank_rejected_microsaccade_events"] += 1
            continue
        kept.append(int(idx))
    return kept, rejected


def trace_bank_bins(
    trace_bank: list[dict[str, Any]],
    indices: list[int],
    *,
    metric: str,
    bin_spec: str,
) -> list[dict[str, Any]]:
    values = np.asarray([trace_metric_value(trace_bank[idx], metric) for idx in indices], dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError(f"No finite trace-bank values for metric {metric!r}.")
    finite_values = values[finite]
    spec = str(bin_spec).strip().lower()
    if spec.startswith("quantile:"):
        n_bins = max(1, int(spec.split(":", 1)[1]))
        finite_positions = np.flatnonzero(finite)
        order = finite_positions[np.argsort(values[finite_positions], kind="mergesort")]
        chunks = [chunk for chunk in np.array_split(order, min(n_bins, order.size)) if chunk.size]
        out: list[dict[str, Any]] = []
        for chunk in chunks:
            members = [int(indices[pos]) for pos in chunk]
            member_values = values[chunk]
            out.append(
                {
                    "bin_index": int(len(out)),
                    "bin_label": f"q{len(out) + 1:02d}",
                    "metric_low": float(np.nanmin(member_values)),
                    "metric_high": float(np.nanmax(member_values)),
                    "metric_median": float(np.nanmedian(member_values)),
                    "n_trace_bank_members": int(len(members)),
                    "indices": members,
                }
            )
        if not out:
            raise ValueError(f"Trace-bank bin specification {bin_spec!r} produced no nonempty bins.")
        return out
    else:
        edges = np.asarray(parse_float_list(str(bin_spec)), dtype=np.float64)
        if edges.size < 2:
            raise ValueError("--trace-bank-bins must provide at least two numeric edges.")
        edges = np.sort(edges)
    if not np.all(np.isfinite(edges)):
        raise ValueError(f"Non-finite trace-bank bin edges for {metric!r}: {edges}")

    out: list[dict[str, Any]] = []
    for bin_idx in range(max(0, edges.size - 1)):
        lo = float(edges[bin_idx])
        hi = float(edges[bin_idx + 1])
        if bin_idx == edges.size - 2:
            member_mask = finite & (values >= lo) & (values <= hi)
        else:
            member_mask = finite & (values >= lo) & (values < hi)
        members = [int(indices[pos]) for pos in np.flatnonzero(member_mask)]
        if not members and np.isclose(lo, hi):
            member_mask = finite & np.isclose(values, lo)
            members = [int(indices[pos]) for pos in np.flatnonzero(member_mask)]
        if not members:
            continue
        member_values = np.asarray([trace_metric_value(trace_bank[idx], metric) for idx in members], dtype=np.float64)
        label = f"q{len(out) + 1:02d}" if spec.startswith("quantile:") else f"bin{len(out) + 1:02d}"
        out.append(
            {
                "bin_index": int(len(out)),
                "bin_label": label,
                "metric_low": float(np.nanmin(member_values)),
                "metric_high": float(np.nanmax(member_values)),
                "metric_median": float(np.nanmedian(member_values)),
                "n_trace_bank_members": int(len(members)),
                "indices": members,
            }
        )
    if not out:
        raise ValueError(f"Trace-bank bin specification {bin_spec!r} produced no nonempty bins.")
    return out


def trace_bank_condition_specs(args: argparse.Namespace, bins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if bool(args.include_static_baseline):
        specs.append(
            {
                "condition_id": "static_tracebank_reference",
                "condition_label": "static",
                "along_scale": 0.0,
                "across_scale": 0.0,
                "motion_scale": 0.0,
                "sweep_mode": "trace_bank",
                "is_static_baseline": True,
                "is_across_sweep": False,
                "trace_bank_condition": False,
                "trace_bank_scale_metric": str(args.trace_bank_scale_metric),
                "trace_bank_bin_label": "static",
                "trace_bank_metric_low": 0.0,
                "trace_bank_metric_high": 0.0,
                "trace_bank_metric_median": 0.0,
                "trace_bank_n_members": 0,
            }
        )
    n_samples = max(1, int(args.trace_bank_samples_per_bin))
    metric_slug = trace_bank_metric_slug(str(args.trace_bank_scale_metric))
    for bin_info in bins:
        for sample_index in range(n_samples):
            suffix = f"_s{sample_index:02d}" if n_samples > 1 else ""
            label_suffix = f" / s{sample_index + 1}" if n_samples > 1 else ""
            condition_id = f"tracebank_{metric_slug}_{bin_info['bin_label']}{suffix}"
            specs.append(
                {
                    "condition_id": condition_id,
                    "condition_label": (
                        f"{bin_info['bin_label']} "
                        f"{float(bin_info['metric_median']):.3g}{label_suffix}"
                    ),
                    "along_scale": float("nan"),
                    "across_scale": float("nan"),
                    "motion_scale": float(bin_info["metric_median"]),
                    "sweep_mode": "trace_bank",
                    "is_static_baseline": False,
                    "is_across_sweep": True,
                    "trace_bank_condition": True,
                    "trace_bank_scale_metric": str(args.trace_bank_scale_metric),
                    "trace_bank_bin_index": int(bin_info["bin_index"]),
                    "trace_bank_bin_label": str(bin_info["bin_label"]),
                    "trace_bank_metric_low": float(bin_info["metric_low"]),
                    "trace_bank_metric_high": float(bin_info["metric_high"]),
                    "trace_bank_metric_median": float(bin_info["metric_median"]),
                    "trace_bank_n_members": int(bin_info["n_trace_bank_members"]),
                    "trace_bank_sample_index": int(sample_index),
                }
            )
    return specs


def covariance_component_payload(item: dict[str, Any], prefix: str) -> dict[str, float]:
    cov_xx = _finite_float(item.get(f"{prefix}cov_xx_deg2", np.nan))
    cov_xy = _finite_float(item.get(f"{prefix}cov_xy_deg2", np.nan))
    cov_yy = _finite_float(item.get(f"{prefix}cov_yy_deg2", np.nan))
    if not all(math.isfinite(v) for v in (cov_xx, cov_xy, cov_yy)):
        return {}
    cov = np.asarray([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
    if not np.all(np.isfinite(cov)):
        return {}
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    order = np.argsort(vals)
    minor = float(vals[order[0]])
    major = float(vals[order[-1]])
    total = major + minor
    major_vec = vecs[:, order[-1]]
    orientation = float(np.degrees(np.arctan2(float(major_vec[1]), float(major_vec[0]))))
    orientation = float((orientation + 180.0) % 180.0)
    det = max(float(np.linalg.det(cov)), 0.0)
    bcea68_deg2 = 2.0 * (-math.log(1.0 - 0.68)) * math.pi * math.sqrt(det)
    out = {
        f"{prefix}cov_major_var_deg2": major,
        f"{prefix}cov_minor_var_deg2": minor,
        f"{prefix}cov_major_sd_arcmin": math.sqrt(major) * 60.0,
        f"{prefix}cov_minor_sd_arcmin": math.sqrt(minor) * 60.0,
        f"{prefix}cov_axis_ratio": math.sqrt(major / minor) if minor > 0.0 else float("inf"),
        f"{prefix}cov_orientation_deg": orientation,
        f"{prefix}bcea68_deg2": bcea68_deg2,
        f"{prefix}bcea68_arcmin2": bcea68_deg2 * 3600.0,
    }
    if total > 1e-12:
        out[f"{prefix}cov_anisotropy"] = float((major - minor) / total)
    return out


def covariance_shape_payload(item: dict[str, Any]) -> dict[str, float]:
    shape = item.get("covariance_shape")
    if shape is None:
        return {}
    try:
        arr = np.asarray(shape, dtype=np.float64)
    except (TypeError, ValueError):
        return {}
    if arr.shape != (2, 2) or not np.all(np.isfinite(arr)):
        return {}
    return {
        "trace_cov_shape_xx": float(arr[0, 0]),
        "trace_cov_shape_xy": float(arr[0, 1]),
        "trace_cov_shape_yy": float(arr[1, 1]),
    }


def trace_bank_metric_payload(item: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    payload.update(covariance_shape_payload(item))
    payload.update(covariance_component_payload(item, "source_"))
    payload.update(covariance_component_payload(item, "rendered_"))
    for prefix in ("source_", "rendered_"):
        speed_mean = _finite_float(item.get(f"{prefix}speed_mean_deg_s", np.nan))
        speed_median = _finite_float(item.get(f"{prefix}speed_median_deg_s", np.nan))
        speed_p95 = _finite_float(item.get(f"{prefix}speed_p95_deg_s", np.nan))
        path_speed = _finite_float(item.get(f"{prefix}path_length_deg_s", np.nan))
        if math.isfinite(speed_mean):
            payload[f"{prefix}speed_mean_arcmin_s"] = speed_mean * 60.0
        if math.isfinite(speed_median):
            payload[f"{prefix}speed_median_arcmin_s"] = speed_median * 60.0
        if math.isfinite(speed_p95):
            payload[f"{prefix}speed_p95_arcmin_s"] = speed_p95 * 60.0
        if math.isfinite(path_speed):
            payload[f"{prefix}path_speed_arcmin_s"] = path_speed * 60.0
    source_d = _finite_float(item.get("source_diffusion_constant_deg2_s", np.nan))
    rendered_d = _finite_float(item.get("rendered_diffusion_constant_deg2_s", np.nan))
    if math.isfinite(source_d) and math.isfinite(rendered_d):
        payload["source_rendered_diffusion_delta_deg2_s"] = rendered_d - source_d
        payload["source_rendered_diffusion_abs_delta_deg2_s"] = abs(rendered_d - source_d)
    for key in TRACE_BANK_METADATA_NUMERIC_COLUMNS:
        if key in payload:
            continue
        if key not in item:
            continue
        value = item[key]
        if isinstance(value, (int, np.integer)):
            payload[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            payload[key] = float(value)
    return payload


def trace_bank_metadata_row(item: dict[str, Any], idx: int, *, n_timepoints: int, scale_metric: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trace_bank_index": int(idx),
        "source_row": int(item["source_row"]),
        "session": str(item["session"]),
        "trial_idx": int(item.get("trial_idx", -1)),
        "global_start": int(item["global_start"]),
        "global_stop": int(item["global_stop"]),
        "source_window_global_start": int(item.get("source_window_global_start", item["global_start"])),
        "source_window_global_stop": int(item.get("source_window_global_stop", item["global_stop"])),
        "snippet_global_start": int(item.get("snippet_global_start", item["global_start"])),
        "snippet_global_stop": int(item.get("snippet_global_stop", item["global_stop"])),
        "snippet_n_samples": int(item.get("snippet_n_samples", int(n_timepoints))),
        "snippet_duration_s": float(item.get("snippet_duration_s", np.nan)),
        "trace_hash": trace_hash(item["trace"]),
        "scale_metric": str(scale_metric),
        "scale_metric_value": trace_metric_value(item, str(scale_metric)),
    }
    row.update(trace_bank_metric_payload(item))
    return row


def trace_bank_assignment_payload(
    item: dict[str, Any],
    *,
    trace_bank_index: int,
    condition_id: str,
    condition_index: int,
    metric: str,
    bin_info: dict[str, Any],
    sample_index: int,
) -> dict[str, Any]:
    trace = np.asarray(item["trace"], dtype=np.float32)
    metric_value = trace_metric_value(item, metric)
    keys = TRACE_BANK_METADATA_NUMERIC_COLUMNS
    payload: dict[str, Any] = {
        "condition_id": str(condition_id),
        "condition_index": int(condition_index),
        "trace": trace,
        "trace_hash": trace_hash(trace),
        "trace_bank_index": int(trace_bank_index),
        "trace_source_row": int(item["source_row"]),
        "trace_source_session": str(item["session"]),
        "trace_source_trial_idx": int(item.get("trial_idx", -1)),
        "trace_source_global_start": int(item.get("global_start", -1)),
        "trace_source_global_stop": int(item.get("global_stop", -1)),
        "trace_source_window_global_start": int(item.get("source_window_global_start", item.get("global_start", -1))),
        "trace_source_window_global_stop": int(item.get("source_window_global_stop", item.get("global_stop", -1))),
        "trace_source_snippet_global_start": int(item.get("snippet_global_start", item.get("global_start", -1))),
        "trace_source_snippet_global_stop": int(item.get("snippet_global_stop", item.get("global_stop", -1))),
        "trace_source_snippet_n_samples": int(item.get("snippet_n_samples", trace.shape[0])),
        "trace_source_snippet_duration_s": float(item.get("snippet_duration_s", np.nan)),
        "trace_bank_scale_metric": str(metric),
        "trace_bank_scale_metric_value": float(metric_value),
        "trace_bank_bin_index": int(bin_info["bin_index"]),
        "trace_bank_bin_label": str(bin_info["bin_label"]),
        "trace_bank_metric_low": float(bin_info["metric_low"]),
        "trace_bank_metric_high": float(bin_info["metric_high"]),
        "trace_bank_metric_median": float(bin_info["metric_median"]),
        "trace_bank_n_members": int(bin_info["n_trace_bank_members"]),
        "trace_bank_sample_index": int(sample_index),
    }
    payload.update(trace_bank_metric_payload(item))
    for key in keys:
        if key in payload:
            continue
        if key not in item:
            continue
        value = item[key]
        if isinstance(value, (int, float, np.integer, np.floating)):
            payload[key] = float(value) if not isinstance(value, (int, np.integer)) else int(value)
    return payload


def trace_bank_assignment_manifest_rows(trials: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "trace_bank_assignments" not in trials.columns:
        return rows
    for movie_index, (_, trial) in enumerate(trials.iterrows()):
        assignments = trial.get("trace_bank_assignments", {})
        if not isinstance(assignments, dict):
            continue
        for condition_id, assignment in assignments.items():
            rows.append(
                {
                    "movie_index": int(movie_index),
                    "trial_id": int(trial["trial_id"]),
                    "image_source_row": int(trial["source_row"]),
                    "condition_id": str(condition_id),
                    "condition_index": int(assignment["condition_index"]),
                    "trace_hash": str(assignment["trace_hash"]),
                    "trace_bank_index": int(assignment["trace_bank_index"]),
                    "trace_source_row": int(assignment["trace_source_row"]),
                    "trace_source_session": str(assignment["trace_source_session"]),
                    "trace_source_window_global_start": int(assignment["trace_source_window_global_start"]),
                    "trace_source_window_global_stop": int(assignment["trace_source_window_global_stop"]),
                    "trace_source_snippet_global_start": int(assignment["trace_source_snippet_global_start"]),
                    "trace_source_snippet_global_stop": int(assignment["trace_source_snippet_global_stop"]),
                    "trace_source_snippet_n_samples": int(assignment["trace_source_snippet_n_samples"]),
                    "trace_bank_scale_metric": str(assignment["trace_bank_scale_metric"]),
                    "trace_bank_scale_metric_value": float(assignment["trace_bank_scale_metric_value"]),
                    "trace_bank_bin_label": str(assignment["trace_bank_bin_label"]),
                    "trace_bank_sample_index": int(assignment["trace_bank_sample_index"]),
                }
            )
    return rows


def build_trace_bank_assignments(
    trials: pd.DataFrame,
    selected: pd.DataFrame,
    run_metadata: dict[str, Any],
    args: argparse.Namespace,
    *,
    bin_seconds: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    rows = selected.drop_duplicates("source_row").copy()
    sort_cols = [col for col in ["session", "trial_idx", "global_start", "source_row"] if col in rows.columns]
    if sort_cols:
        rows = rows.sort_values(sort_cols)
    if int(args.trace_bank_max_source_windows) > 0:
        rows = rows.iloc[: int(args.trace_bank_max_source_windows)].copy()
    eyepos_by_session = _session_dataset_cache(rows["session"].astype(str).dropna().unique().tolist())
    cfg = run_metadata.get("config", {})
    trace_bank, builder_meta = build_native_snippet_trace_bank(
        rows,
        eyepos_by_session,
        int(args.n_timepoints),
        dt=float(bin_seconds),
        microsaccade_speed_threshold_dps=(
            float(args.microsaccade_speed_threshold_dps)
            if args.microsaccade_speed_threshold_dps is not None
            else (
                float(cfg["microsaccade_speed_threshold_dps"])
                if cfg.get("microsaccade_speed_threshold_dps") is not None
                else None
            )
        ),
        microsaccade_threshold_z=float(getattr(args, "microsaccade_threshold_z", cfg.get("microsaccade_threshold_z", 6.0))),
        microsaccade_pad_frames=int(getattr(args, "microsaccade_pad_frames", cfg.get("microsaccade_pad_frames", 1))),
    )
    for item in trace_bank:
        item["observed_rms_arcmin"] = float(item["observed_rms_deg"]) * 60.0
        item["path_length_arcmin"] = float(item["path_length_deg"]) * 60.0

    eligible, rejected = filter_trace_bank_items(trace_bank, args)
    bins = trace_bank_bins(
        trace_bank,
        eligible,
        metric=str(args.trace_bank_scale_metric),
        bin_spec=str(args.trace_bank_bins),
    )
    specs = trace_bank_condition_specs(args, bins)
    bin_by_index = {int(info["bin_index"]): info for info in bins}
    assignments_by_row: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for movie_index, (_, trial) in enumerate(trials.iterrows()):
        image_source_row = int(trial["source_row"])
        assignment_by_condition: dict[str, Any] = {}
        for condition_index, spec in enumerate(specs):
            if not bool(spec.get("trace_bank_condition", False)):
                continue
            bin_info = bin_by_index[int(spec["trace_bank_bin_index"])]
            members = list(bin_info["indices"])
            if bool(args.trace_bank_exclude_same_source):
                members = [idx for idx in members if int(trace_bank[idx]["source_row"]) != image_source_row]
            if not members:
                raise ValueError(
                    "Trace-bank bin has no eligible source after same-source exclusion: "
                    f"image_source_row={image_source_row}, bin={bin_info['bin_label']}"
                )
            seed = (
                int(args.trace_bank_seed)
                + image_source_row * 1_000_003
                + int(spec["trace_bank_bin_index"]) * 10_007
                + int(spec["trace_bank_sample_index"]) * 1_009
            )
            rng = np.random.default_rng(seed)
            bank_index = int(members[int(rng.integers(0, len(members)))])
            assignment = trace_bank_assignment_payload(
                trace_bank[bank_index],
                trace_bank_index=bank_index,
                condition_id=str(spec["condition_id"]),
                condition_index=int(condition_index),
                metric=str(args.trace_bank_scale_metric),
                bin_info=bin_info,
                sample_index=int(spec["trace_bank_sample_index"]),
            )
            assignment_by_condition[str(spec["condition_id"])] = assignment
        assignments_by_row.append(assignment_by_condition)
        for condition_id, assignment in assignment_by_condition.items():
            manifest_rows.append(
                {
                    "movie_index": int(movie_index),
                    "trial_id": int(trial["trial_id"]),
                    "image_source_row": image_source_row,
                    "condition_id": str(condition_id),
                    "condition_index": int(assignment["condition_index"]),
                    "trace_hash": str(assignment["trace_hash"]),
                    "trace_bank_index": int(assignment["trace_bank_index"]),
                    "trace_source_row": int(assignment["trace_source_row"]),
                    "trace_source_session": str(assignment["trace_source_session"]),
                    "trace_bank_scale_metric": str(assignment["trace_bank_scale_metric"]),
                    "trace_bank_scale_metric_value": float(assignment["trace_bank_scale_metric_value"]),
                    "trace_bank_bin_label": str(assignment["trace_bank_bin_label"]),
                    "trace_bank_sample_index": int(assignment["trace_bank_sample_index"]),
                }
            )
    out = trials.copy()
    out["trace_bank_assignments"] = assignments_by_row
    trace_bank_rows = [
        trace_bank_metadata_row(
            item,
            idx,
            n_timepoints=int(args.n_timepoints),
            scale_metric=str(args.trace_bank_scale_metric),
        )
        for idx, item in enumerate(trace_bank)
    ]
    meta = {
        "trace_bank_source_mode": "native_real_backimage_trace_bank",
        "trace_bank_contract": (
            "Build a bank of center-cropped native BackImage eye-position snippets with n_timepoints samples, "
            "compute predeclared motion-scale metrics, bin native traces by the selected metric, "
            "and assign one unscaled real trace from each bin to every natural-image patch."
        ),
        "trace_bank_metric_literature_note": (
            "The default diffusion metric follows the Brownian/MSD convention used in fixational "
            "drift work: the slope of the 2D mean-squared displacement curve divided by 4."
        ),
        "trace_bank_scale_metric": str(args.trace_bank_scale_metric),
        "trace_bank_bins": str(args.trace_bank_bins),
        "trace_bank_samples_per_bin": int(args.trace_bank_samples_per_bin),
        "trace_bank_seed": int(args.trace_bank_seed),
        "trace_bank_exclude_same_source": bool(args.trace_bank_exclude_same_source),
        "n_trace_bank_source_rows": int(rows.shape[0]),
        "n_trace_bank_total": int(len(trace_bank)),
        "n_trace_bank_eligible": int(len(eligible)),
        **rejected,
        **builder_meta,
        "trace_bank_bin_summary": [
            {
                "bin_index": int(info["bin_index"]),
                "bin_label": str(info["bin_label"]),
                "metric_low": float(info["metric_low"]),
                "metric_high": float(info["metric_high"]),
                "metric_median": float(info["metric_median"]),
                "n_trace_bank_members": int(info["n_trace_bank_members"]),
            }
            for info in bins
        ],
        "trace_bank_condition_specs": specs,
        "trace_bank_assignment_manifest": manifest_rows,
        "trace_bank_rows": trace_bank_rows,
    }
    return out, specs, meta


def decompose_trace(trace: np.ndarray, axis_deg: float) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(trace, dtype=np.float64)
    centered = centered - np.mean(centered, axis=0, keepdims=True)
    theta = np.radians(float(axis_deg))
    along_u = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)
    across_u = np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    along = (centered @ along_u)[:, None] * along_u[None, :]
    across = (centered @ across_u)[:, None] * across_u[None, :]
    return along.astype(np.float32), across.astype(np.float32)


def microsaccade_trace_event_anchor_mode(mode: str) -> str:
    mode_text = str(mode)
    if mode_text.endswith("_start_anchor"):
        return "start"
    if mode_text.endswith("_end_anchor"):
        return "end"
    return "mean_center"


def trial_event_anchor_mode(trial: pd.Series) -> str:
    if "source_trace_event_anchor_mode" not in trial.index:
        return "mean_center"
    value = trial["source_trace_event_anchor_mode"]
    if value is None:
        return "mean_center"
    if isinstance(value, float) and not math.isfinite(value):
        return "mean_center"
    text = str(value)
    return text if text in {"mean_center", "start", "end"} else "mean_center"


def combined_axis_trace(
    source_trace: np.ndarray,
    *,
    axis_deg: float,
    along_scale: float,
    across_scale: float,
    event_scale_mask: np.ndarray | None = None,
    event_anchor_mode: str = "mean_center",
) -> tuple[np.ndarray, dict[str, Any]]:
    along, across = decompose_trace(source_trace, float(axis_deg))
    event_scale_mask_enabled = event_scale_mask is not None
    event_anchor = str(event_anchor_mode)
    if event_anchor not in {"mean_center", "start", "end"}:
        raise ValueError(f"Unknown event anchor mode {event_anchor_mode!r}")
    if event_scale_mask_enabled:
        mask = np.asarray(event_scale_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != along.shape[0]:
            raise ValueError(
                f"Event scale mask length {mask.shape[0]} does not match trace length {along.shape[0]}."
            )
        if along.shape[0] < 2:
            out = np.zeros_like(along, dtype=np.float32)
            step_mask = np.zeros((0,), dtype=bool)
        else:
            # A sample marked as part of the event means the displacement step into
            # that sample is event motion. All other steps retain the original drift.
            step_mask = mask[1:]
            along_step_scale = np.where(step_mask, float(along_scale), 1.0).astype(np.float64)
            across_step_scale = np.where(step_mask, float(across_scale), 1.0).astype(np.float64)
            along_delta = np.diff(np.asarray(along, dtype=np.float64), axis=0)
            across_delta = np.diff(np.asarray(across, dtype=np.float64), axis=0)
            base = np.asarray(along, dtype=np.float64) + np.asarray(across, dtype=np.float64)
            out = np.zeros_like(np.asarray(along, dtype=np.float64))
            if event_anchor == "end":
                out[-1] = base[-1]
                for idx in range(out.shape[0] - 2, -1, -1):
                    out[idx] = (
                        out[idx + 1]
                        - along_step_scale[idx] * along_delta[idx]
                        - across_step_scale[idx] * across_delta[idx]
                    )
            else:
                out[0] = base[0]
                for idx in range(1, out.shape[0]):
                    out[idx] = (
                        out[idx - 1]
                        + along_step_scale[idx - 1] * along_delta[idx - 1]
                        + across_step_scale[idx - 1] * across_delta[idx - 1]
                    )
    else:
        step_mask = np.zeros((max(0, along.shape[0] - 1),), dtype=bool)
        out = float(along_scale) * along + float(across_scale) * across
    final_centering = "mean_center"
    if event_scale_mask_enabled and event_anchor in {"start", "end"}:
        final_centering = "none_after_anchor"
    else:
        out = out - np.mean(out, axis=0, keepdims=True)
    return out.astype(np.float32), {
        "source_trace_rms_deg": trace_rms(source_trace),
        "source_trace_path_length_deg": trace_path_length(source_trace),
        "along_component_rms_deg": trace_rms(along),
        "across_component_rms_deg": trace_rms(across),
        "output_trace_rms_deg": trace_rms(out),
        "output_trace_path_length_deg": trace_path_length(out),
        "output_trace_start_x_deg": float(out[0, 0]) if out.shape[0] else float("nan"),
        "output_trace_start_y_deg": float(out[0, 1]) if out.shape[0] else float("nan"),
        "output_trace_end_x_deg": float(out[-1, 0]) if out.shape[0] else float("nan"),
        "output_trace_end_y_deg": float(out[-1, 1]) if out.shape[0] else float("nan"),
        "output_trace_mean_x_deg": float(np.mean(out[:, 0])) if out.shape[0] else float("nan"),
        "output_trace_mean_y_deg": float(np.mean(out[:, 1])) if out.shape[0] else float("nan"),
        "event_scale_mask_enabled": bool(event_scale_mask_enabled),
        "event_scale_anchor_mode": event_anchor if event_scale_mask_enabled else "none",
        "output_trace_centering": final_centering,
        "event_scale_sample_count": int(np.sum(event_scale_mask)) if event_scale_mask is not None else 0,
        "event_scale_step_count": int(np.sum(step_mask)),
        "outside_event_drift_scale": 1.0 if event_scale_mask_enabled else float("nan"),
        "axis_deg": float(axis_deg),
        "along_scale": float(along_scale),
        "across_scale": float(across_scale),
    }


def rotate_trace_xy(trace: np.ndarray, rotation_deg: float) -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected trace shape (T, 2), got {arr.shape}")
    angle = float(rotation_deg) % 360.0
    if np.isclose(angle, 0.0):
        return arr.astype(np.float32, copy=True)
    theta = np.radians(angle)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    out = np.empty_like(arr, dtype=np.float32)
    out[:, 0] = arr[:, 0] * cos_t - arr[:, 1] * sin_t
    out[:, 1] = arr[:, 0] * sin_t + arr[:, 1] * cos_t
    return out


def rotate_patch_gaze_frame(patch: np.ndarray, rotation_deg: float) -> np.ndarray:
    angle = int(rotation_deg) % 360
    if angle == 0:
        return np.asarray(patch).copy()
    if angle % 90 != 0:
        raise ValueError("stimulus rotation must be a multiple of 90 degrees")
    # Positive gaze-frame rotation maps image rightward structure toward the top
    # of the displayed patch, matching numpy's counter-clockwise rot90 display behavior.
    return np.rot90(np.asarray(patch), k=angle // 90).copy()


def rotated_axis_deg(axis_deg: float, rotation_deg: float) -> float:
    return float((float(axis_deg) + float(rotation_deg)) % 180.0)


def trial_event_scale_mask(trial: pd.Series, n_timepoints: int) -> np.ndarray | None:
    if "source_trace_event_scale_mask" not in trial.index:
        return None
    value = trial["source_trace_event_scale_mask"]
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    mask = np.asarray(value, dtype=bool).reshape(-1)
    if mask.size == 0:
        return None
    if mask.size != int(n_timepoints):
        raise ValueError(f"Expected event scale mask length {int(n_timepoints)}, got {mask.size}.")
    return mask


def unit_spatial_ssi_for_movie(rate_map: np.ndarray, *, bin_seconds: float) -> dict[str, Any]:
    y = np.asarray(rate_map, dtype=np.float64)
    if y.ndim != 4:
        raise ValueError(f"Expected rate map with shape (T,N,H,W), got {y.shape}")
    if np.nanmin(y) < -1e-7:
        raise ValueError(f"rate map contains negative values; min={float(np.nanmin(y)):.6g}")
    y = np.maximum(y, 0.0)
    t_max, n_units, height, width = y.shape
    flat = y.reshape(t_max, n_units, height * width)
    rbar = np.mean(flat, axis=2)
    gain = flat / (rbar[..., None] + EPS)
    unit_bits_t = np.mean(gain * np.log2(gain + EPS), axis=2)
    unit_expected = np.sum(rbar * float(bin_seconds), axis=0)
    unit_bits = np.sum(unit_bits_t * rbar * float(bin_seconds), axis=0) / np.maximum(unit_expected, EPS)
    population_bits = float(np.sum(unit_bits_t * rbar * float(bin_seconds)) / max(float(np.sum(unit_expected)), EPS))
    return {
        "unit_bits_per_spike": unit_bits.astype(np.float32),
        "unit_expected_spikes": unit_expected.astype(np.float32),
        "unit_mean_rate": np.mean(rbar, axis=0).astype(np.float32),
        "population_bits_per_spike": population_bits,
    }


def unit_spatial_ssi_for_mean_map(mean_rate_map: np.ndarray, *, unit_weights: np.ndarray | None = None) -> dict[str, Any]:
    y = np.asarray(mean_rate_map, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError(f"Expected mean rate map with shape (N,H,W), got {y.shape}")
    if np.nanmin(y) < -1e-7:
        raise ValueError(f"mean rate map contains negative values; min={float(np.nanmin(y)):.6g}")
    y = np.maximum(y, 0.0)
    n_units, height, width = y.shape
    flat = y.reshape(n_units, height * width)
    rbar = np.mean(flat, axis=1)
    gain = flat / (rbar[:, None] + EPS)
    unit_bits = np.mean(gain * np.log2(gain + EPS), axis=1)
    if unit_weights is None:
        weights = rbar
    else:
        weights = np.asarray(unit_weights, dtype=np.float64)
        if weights.shape != (n_units,):
            raise ValueError(f"Expected unit_weights shape {(n_units,)}, got {weights.shape}")
    population_bits = float(np.sum(unit_bits * weights) / max(float(np.sum(weights)), EPS))
    return {
        "unit_bits_per_spike": unit_bits.astype(np.float32),
        "unit_mean_rate": rbar.astype(np.float32),
        "population_bits_per_spike": population_bits,
    }


def condition_specs(
    scales: list[float],
    *,
    along_scale: float,
    along_scales: list[float] | None = None,
    condition_pairs: list[tuple[float, float]] | None = None,
    include_static_baseline: bool = True,
    sweep_mode: str = "across",
    zero_motion_is_static_baseline: bool = True,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if str(sweep_mode) == "pairs":
        if not condition_pairs:
            raise ValueError("--sweep-mode pairs requires --condition-pairs.")
        for along_value, across_value in condition_pairs:
            specs.append(
                {
                    "condition_id": f"along{scale_token(along_value)}_across{scale_token(across_value)}",
                    "condition_label": f"a{float(along_value):g}/c{float(across_value):g}",
                    "along_scale": float(along_value),
                    "across_scale": float(across_value),
                    "motion_scale": float("nan"),
                    "sweep_mode": "pairs",
                    "is_static_baseline": bool(
                        zero_motion_is_static_baseline
                        and np.isclose(float(along_value), 0.0)
                        and np.isclose(float(across_value), 0.0)
                    ),
                    "is_across_sweep": True,
                }
            )
        return specs

    if str(sweep_mode) == "grid":
        values = list(along_scales or [])
        if not values:
            raise ValueError("--sweep-mode grid requires --along-scales.")
        for along_value in values:
            for across_value in scales:
                specs.append(
                    {
                        "condition_id": f"along{scale_token(along_value)}_across{scale_token(across_value)}",
                        "condition_label": f"a{float(along_value):g}/c{float(across_value):g}",
                        "along_scale": float(along_value),
                        "across_scale": float(across_value),
                        "motion_scale": float("nan"),
                        "sweep_mode": "grid",
                        "is_static_baseline": bool(
                            zero_motion_is_static_baseline
                            and np.isclose(float(along_value), 0.0)
                            and np.isclose(float(across_value), 0.0)
                        ),
                        "is_across_sweep": True,
                    }
                )
        return specs

    if str(sweep_mode) == "isotropic":
        values = list(scales)
        if bool(include_static_baseline) and not any(np.isclose(float(v), 0.0) for v in values):
            values = [0.0, *values]
        for scale in values:
            specs.append(
                {
                    "condition_id": f"isotropic_scale{scale_token(scale)}",
                    "condition_label": f"{float(scale):g}x",
                    "along_scale": float(scale),
                    "across_scale": float(scale),
                    "motion_scale": float(scale),
                    "sweep_mode": "isotropic",
                    "is_static_baseline": bool(
                        zero_motion_is_static_baseline and np.isclose(float(scale), 0.0)
                    ),
                    "is_across_sweep": True,
                }
            )
        return specs

    if include_static_baseline:
        specs.append(
            {
                "condition_id": "static_along0_across0",
                "condition_label": "static",
                "along_scale": 0.0,
                "across_scale": 0.0,
                "motion_scale": 0.0,
                "sweep_mode": "across",
                "is_static_baseline": True,
                "is_across_sweep": False,
            }
        )
    for scale in scales:
        specs.append(
            {
                "condition_id": f"along{scale_token(along_scale)}_across{scale_token(scale)}",
                "condition_label": f"{float(scale):g}x",
                "along_scale": float(along_scale),
                "across_scale": float(scale),
                "motion_scale": float(scale),
                "sweep_mode": "across",
                "is_static_baseline": False,
                "is_across_sweep": True,
            }
        )
    return specs


def _finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _event_amplitude(
    eyepos: np.ndarray,
    *,
    onset_global: int,
    offset_global: int,
) -> tuple[float, float, float, int, int]:
    arr = np.asarray(eyepos, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        return 0.0, 0.0, float("nan"), int(onset_global), int(offset_global)
    start_idx = max(0, min(arr.shape[0] - 1, int(onset_global) - 1))
    stop_idx = max(0, min(arr.shape[0] - 1, int(offset_global)))
    delta = arr[stop_idx] - arr[start_idx]
    amp_deg = float(np.linalg.norm(delta))
    direction = float(np.degrees(np.arctan2(delta[1], delta[0])) % 360.0) if amp_deg > 0 else float("nan")
    return amp_deg, 60.0 * amp_deg, direction, int(start_idx), int(stop_idx)


def _centered_trace_metrics(trace: np.ndarray) -> dict[str, float]:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        return {
            "snippet_raw_rms_deg": 0.0,
            "snippet_raw_max_radius_deg": 0.0,
            "snippet_raw_path_length_deg": 0.0,
        }
    centered = arr - np.nanmean(arr, axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1)
    path = np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)) if arr.shape[0] > 1 else 0.0
    return {
        "snippet_raw_rms_deg": float(np.sqrt(np.nanmean(radius * radius))) if radius.size else 0.0,
        "snippet_raw_max_radius_deg": float(np.nanmax(radius)) if radius.size else 0.0,
        "snippet_raw_path_length_deg": float(path),
    }


def _resample_trace_no_center(trace: np.ndarray, n_timepoints: int) -> tuple[np.ndarray, np.ndarray]:
    trace = np.asarray(trace, dtype=np.float64)
    n_timepoints = int(n_timepoints)
    if trace.shape[0] < 2:
        idx = np.zeros(max(1, n_timepoints), dtype=np.float64)
        return np.zeros((max(1, n_timepoints), 2), dtype=np.float32), idx
    idx = np.linspace(0.0, float(trace.shape[0] - 1), n_timepoints)
    lo = np.floor(idx).astype(int)
    hi = np.ceil(idx).astype(int)
    frac = idx - lo
    out = trace[lo] * (1.0 - frac[:, None]) + trace[hi] * frac[:, None]
    finite = np.isfinite(out).all(axis=1)
    if not np.all(finite):
        good = np.flatnonzero(finite)
        if good.size == 0:
            out = np.zeros_like(out)
        else:
            bad = np.flatnonzero(~finite)
            for dim in range(2):
                out[bad, dim] = np.interp(bad, good, out[good, dim])
    return out.astype(np.float32), idx


def _microsaccade_source_trace(
    raw_snippet: np.ndarray,
    *,
    mode: str,
    n_timepoints: int,
    core_onset_in_snippet: int,
    core_offset_in_snippet: int,
    padded_onset_in_snippet: int,
    padded_offset_in_snippet: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(raw_snippet, dtype=np.float64)
    n_timepoints = int(n_timepoints)
    mode_text = str(mode)
    if mode_text == "full_snippet":
        source_trace = _resample_trace(raw, n_timepoints)
        return source_trace.astype(np.float32, copy=False), {
            "microsaccade_trace_mode": "full_snippet",
            "microsaccade_trace_keep_onset_frame_in_snippet": int(core_onset_in_snippet),
            "microsaccade_trace_keep_offset_frame_in_snippet": int(core_offset_in_snippet),
            "microsaccade_trace_keep_onset_frame_resampled": float("nan"),
            "microsaccade_trace_keep_offset_frame_resampled": float("nan"),
            "microsaccade_trace_nonzero_resampled_frames": int(source_trace.shape[0]),
            "microsaccade_trace_event_scaled_sample_count": 0,
            "microsaccade_trace_event_scaled_step_count": 0,
            "microsaccade_trace_event_anchor_mode": "none",
            "microsaccade_trace_centering": "resampled_trace_mean_centered",
            "microsaccade_trace_contract": "full pre/event/post snippet, resampled and mean-centered",
        }

    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] != 2:
        return np.zeros((n_timepoints, 2), dtype=np.float32), {
            "microsaccade_trace_mode": mode_text,
            "microsaccade_trace_keep_onset_frame_in_snippet": 0,
            "microsaccade_trace_keep_offset_frame_in_snippet": 0,
            "microsaccade_trace_keep_onset_frame_resampled": float("nan"),
            "microsaccade_trace_keep_offset_frame_resampled": float("nan"),
            "microsaccade_trace_nonzero_resampled_frames": 0,
            "microsaccade_trace_event_scaled_sample_count": 0,
            "microsaccade_trace_event_scaled_step_count": 0,
            "microsaccade_trace_event_anchor_mode": "none",
            "microsaccade_trace_centering": "zero_trace",
            "microsaccade_trace_contract": "invalid raw snippet; zero trace emitted",
        }

    if mode_text == "core_zero_rest" or mode_text in CORE_EVENT_SCALED_MICROSACCADE_TRACE_MODES:
        keep_onset = int(core_onset_in_snippet)
        keep_offset = int(core_offset_in_snippet)
        label = "unpadded detected microsaccade core"
    elif mode_text == "padded_event_zero_rest" or mode_text in PADDED_EVENT_SCALED_MICROSACCADE_TRACE_MODES:
        keep_onset = int(padded_onset_in_snippet)
        keep_offset = int(padded_offset_in_snippet)
        label = "padded detected microsaccade event"
    else:
        raise ValueError(f"Unknown microsaccade trace mode {mode!r}")

    keep_onset = max(0, min(raw.shape[0] - 1, keep_onset))
    keep_offset = max(keep_onset, min(raw.shape[0] - 1, keep_offset))
    keep_onset_resampled = float(
        keep_onset * float(max(1, n_timepoints - 1)) / float(max(1, raw.shape[0] - 1))
    )
    keep_offset_resampled = float(
        keep_offset * float(max(1, n_timepoints - 1)) / float(max(1, raw.shape[0] - 1))
    )
    if mode_text in EVENT_SCALED_MICROSACCADE_TRACE_MODES:
        anchor_mode = microsaccade_trace_event_anchor_mode(mode_text)
        source_trace, raw_idx = _resample_trace_no_center(raw, n_timepoints)
        event_mask = (raw_idx >= float(keep_onset) - 0.5) & (raw_idx <= float(keep_offset) + 0.5)
        source_trace = np.asarray(source_trace, dtype=np.float64)
        source_trace = source_trace - np.mean(source_trace, axis=0, keepdims=True)
        event_sample_count = int(np.sum(event_mask))
        event_step_count = int(np.sum(event_mask[1:])) if event_mask.shape[0] > 1 else 0
        return source_trace.astype(np.float32, copy=False), {
            "microsaccade_trace_mode": mode_text,
            "microsaccade_trace_keep_onset_frame_in_snippet": int(keep_onset),
            "microsaccade_trace_keep_offset_frame_in_snippet": int(keep_offset),
            "microsaccade_trace_keep_onset_frame_resampled": keep_onset_resampled,
            "microsaccade_trace_keep_offset_frame_resampled": keep_offset_resampled,
            "microsaccade_trace_nonzero_resampled_frames": int(source_trace.shape[0]),
            "microsaccade_trace_event_scaled_sample_count": event_sample_count,
            "microsaccade_trace_event_scaled_step_count": event_step_count,
            "microsaccade_trace_event_anchor_mode": anchor_mode,
            "microsaccade_trace_centering": (
                "snippet_mean_centered_source_then_condition_mean_centered"
                if anchor_mode == "mean_center"
                else "snippet_mean_centered_source_no_condition_recentering"
            ),
            "microsaccade_trace_contract": (
                f"full pre/event/post snippet retained; {label} increments are condition-scaled "
                "while outside-event drift increments remain at 1x; "
                + (
                    "each scaled condition is recentered to zero mean"
                    if anchor_mode == "mean_center"
                    else f"strict {anchor_mode}-anchored local trace position is preserved"
                )
            ),
            "source_trace_event_scale_mask": event_mask.astype(bool, copy=False),
            "source_trace_event_anchor_mode": anchor_mode,
        }

    baseline_idx = max(0, keep_onset - 1)
    clipped = np.zeros_like(raw, dtype=np.float64)
    clipped[keep_onset : keep_offset + 1] = raw[keep_onset : keep_offset + 1] - raw[baseline_idx][None, :]
    source_trace, raw_idx = _resample_trace_no_center(clipped, n_timepoints)
    keep_mask = (raw_idx >= float(keep_onset) - 0.5) & (raw_idx <= float(keep_offset) + 0.5)
    source_trace = np.asarray(source_trace, dtype=np.float64)
    source_trace[~keep_mask] = 0.0
    if np.any(keep_mask):
        source_trace[keep_mask] -= np.mean(source_trace[keep_mask], axis=0, keepdims=True)
    source_trace[~keep_mask] = 0.0
    keep_values = source_trace[keep_mask]
    nonzero = (
        int(np.sum(np.linalg.norm(keep_values, axis=1) > 1e-12))
        if keep_values.size
        else 0
    )
    return source_trace.astype(np.float32, copy=False), {
        "microsaccade_trace_mode": mode_text,
        "microsaccade_trace_keep_onset_frame_in_snippet": int(keep_onset),
        "microsaccade_trace_keep_offset_frame_in_snippet": int(keep_offset),
        "microsaccade_trace_keep_onset_frame_resampled": keep_onset_resampled,
        "microsaccade_trace_keep_offset_frame_resampled": keep_offset_resampled,
        "microsaccade_trace_nonzero_resampled_frames": nonzero,
        "microsaccade_trace_event_scaled_sample_count": 0,
        "microsaccade_trace_event_scaled_step_count": 0,
        "microsaccade_trace_event_anchor_mode": "none",
        "microsaccade_trace_centering": "event_pulse_mean_centered_with_zero_rest",
        "microsaccade_trace_contract": (
            f"{label} retained as a zero-mean event pulse; all pre/post drift samples set to zero"
        ),
    }


def _filter_microsaccades_by_amplitude_sd(
    payloads: list[dict[str, Any]],
    *,
    sd_filter: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    k = float(sd_filter)
    if k <= 0.0 or not payloads:
        return payloads, {
            "microsaccade_amplitude_sd_filter": float(k),
            "microsaccade_amplitude_sd_filter_enabled": False,
            "n_rejected_amplitude_sd_filter": 0,
        }
    amplitudes = np.asarray([float(item["microsaccade_amplitude_arcmin"]) for item in payloads], dtype=np.float64)
    finite = np.isfinite(amplitudes)
    if not np.any(finite):
        return [], {
            "microsaccade_amplitude_sd_filter": float(k),
            "microsaccade_amplitude_sd_filter_enabled": True,
            "microsaccade_amplitude_mean_arcmin_before_sd_filter": float("nan"),
            "microsaccade_amplitude_sd_arcmin_before_sd_filter": float("nan"),
            "microsaccade_amplitude_sd_filter_lower_arcmin": float("nan"),
            "microsaccade_amplitude_sd_filter_upper_arcmin": float("nan"),
            "n_rejected_amplitude_sd_filter": int(len(payloads)),
        }
    mean = float(np.nanmean(amplitudes[finite]))
    sd = float(np.nanstd(amplitudes[finite], ddof=0))
    if not math.isfinite(sd) or sd <= 0.0:
        lower = upper = mean
    else:
        lower = mean - k * sd
        upper = mean + k * sd
    kept: list[dict[str, Any]] = []
    for item, amplitude in zip(payloads, amplitudes, strict=True):
        if math.isfinite(float(amplitude)) and lower <= float(amplitude) <= upper:
            kept.append(item)
    for idx, item in enumerate(kept):
        item["source_row"] = int(idx)
        item["trial_id"] = int(idx)
    return kept, {
        "microsaccade_amplitude_sd_filter": float(k),
        "microsaccade_amplitude_sd_filter_enabled": True,
        "microsaccade_amplitude_mean_arcmin_before_sd_filter": mean,
        "microsaccade_amplitude_sd_arcmin_before_sd_filter": sd,
        "microsaccade_amplitude_sd_filter_lower_arcmin": float(lower),
        "microsaccade_amplitude_sd_filter_upper_arcmin": float(upper),
        "n_rejected_amplitude_sd_filter": int(len(payloads) - len(kept)),
    }


def _over_positive_limit(value: float, limit: float) -> bool:
    lim = float(limit)
    if lim <= 0.0:
        return False
    val = float(value)
    return (not math.isfinite(val)) or val > lim


def _trace_speed_deg_s(trace: np.ndarray, *, dt: float) -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != 2:
        return np.asarray([], dtype=np.float64)
    inc = np.diff(arr, axis=0, prepend=arr[:1])
    return np.linalg.norm(inc, axis=1) / max(float(dt), EPS)


def _event_peak_sample(trace: np.ndarray, event: dict[str, Any], *, dt: float) -> int:
    speed = _trace_speed_deg_s(trace, dt=float(dt))
    if speed.size == 0:
        return int(event.get("onset", 0))
    start = max(0, min(speed.size - 1, int(event["onset"])))
    stop = max(start, min(speed.size - 1, int(event["offset"])))
    segment = speed[start : stop + 1]
    finite = np.isfinite(segment)
    if not np.any(finite):
        return int(start)
    local = int(np.nanargmax(np.where(finite, segment, -np.inf)))
    return int(start + local)


def _padded_event_for_core(
    padded_events: list[dict[str, Any]],
    *,
    core_onset: int,
    core_offset: int,
) -> dict[str, Any]:
    for event in padded_events:
        onset = int(event["onset"])
        offset = int(event["offset"])
        if onset <= int(core_onset) and offset >= int(core_offset):
            return event
    return {
        "onset": int(core_onset),
        "offset": int(core_offset),
        "duration_samples": int(core_offset) - int(core_onset) + 1,
    }


def _candidate_quality_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, int]:
    """Lower keys are preferred within one physical microsaccade cluster."""
    return (
        -float(candidate["microsaccade_dedup_source_margin_frames"]),
        float(candidate["snippet_raw_path_length_deg"]),
        float(candidate["snippet_raw_rms_deg"]),
        float(candidate["microsaccade_threshold_dps"]),
        int(candidate["parent_source_row"]),
    )


def _dedupe_microsaccade_candidates(
    candidates: list[dict[str, Any]],
    *,
    tolerance_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    tolerance = max(0, int(tolerance_frames))
    if not candidates:
        return [], {
            "n_candidate_microsaccade_snippets_before_dedup": 0,
            "n_duplicate_events": 0,
            "n_exact_duplicate_events": 0,
            "n_dedup_clusters": 0,
        }

    exact_keys = [
        (
            str(item["session"]),
            int(item["trial_idx"]),
            int(item["microsaccade_event_onset_global"]),
            int(item["microsaccade_event_offset_global"]),
        )
        for item in candidates
    ]
    n_exact_duplicate_rows = len(exact_keys) - len(set(exact_keys))

    selected: list[dict[str, Any]] = []
    n_clusters = 0
    sort_key = lambda item: (
        str(item["session"]),
        int(item["trial_idx"]),
        int(item["microsaccade_event_peak_global"]),
        int(item["microsaccade_event_onset_global"]),
        int(item["microsaccade_event_offset_global"]),
        int(item["parent_source_row"]),
    )

    current_group: tuple[str, int] | None = None
    cluster: list[dict[str, Any]] = []
    cluster_last_peak = -10**12

    def flush_cluster() -> None:
        nonlocal n_clusters
        if not cluster:
            return
        n_clusters += 1
        best = min(cluster, key=_candidate_quality_key)
        best = dict(best)
        best["microsaccade_dedup_cluster_size"] = int(len(cluster))
        selected.append(best)
        cluster.clear()

    for item in sorted(candidates, key=sort_key):
        group = (str(item["session"]), int(item["trial_idx"]))
        peak = int(item["microsaccade_event_peak_global"])
        if current_group is None or group != current_group:
            flush_cluster()
            current_group = group
            cluster_last_peak = peak
        elif peak - cluster_last_peak > tolerance:
            flush_cluster()
            cluster_last_peak = peak
        cluster.append(item)
        cluster_last_peak = max(cluster_last_peak, peak)
    flush_cluster()

    for idx, item in enumerate(selected):
        item["source_row"] = int(idx)
        item["trial_id"] = int(idx)

    return selected, {
        "n_candidate_microsaccade_snippets_before_dedup": int(len(candidates)),
        "n_duplicate_events": int(len(candidates) - len(selected)),
        "n_exact_duplicate_events": int(n_exact_duplicate_rows),
        "n_dedup_clusters": int(n_clusters),
    }


def build_microsaccade_snippet_trials(
    selected: pd.DataFrame,
    run_metadata: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required = {"session", "trial_idx", "global_start", "global_stop", "mean_x_deg", "mean_y_deg", str(args.axis_column)}
    missing = sorted(required.difference(selected.columns))
    if missing:
        raise ValueError(f"Microsaccade snippet source is missing required columns: {missing}")

    rows = selected.copy()
    if "source_row" not in rows.columns:
        rows["source_row"] = np.arange(rows.shape[0], dtype=int)
    sort_cols = [col for col in ["session", "trial_idx", "global_start", "source_row"] if col in rows.columns]
    if sort_cols:
        rows = rows.sort_values(sort_cols)
    if int(args.microsaccade_max_source_windows) > 0:
        rows = rows.iloc[: int(args.microsaccade_max_source_windows)].copy()

    eyepos_by_session = _session_dataset_cache(rows["session"].astype(str).dropna().unique().tolist())
    cfg = run_metadata.get("config", {})
    bin_seconds = _finite_float(cfg.get("bin_seconds", np.nan), 1.0 / 120.0)
    pre_frames = max(0, int(args.microsaccade_pre_frames))
    post_frames = max(0, int(args.microsaccade_post_frames))
    min_amp = max(0.0, float(args.microsaccade_min_amplitude_arcmin))
    max_amp_arg = float(args.microsaccade_max_amplitude_arcmin)
    max_amp = float(max_amp_arg) if max_amp_arg > 0.0 else float("inf")
    fixed_threshold = (
        float(args.microsaccade_speed_threshold_dps)
        if args.microsaccade_speed_threshold_dps is not None
        else None
    )

    candidate_payloads: list[dict[str, Any]] = []
    n_detected = 0
    n_rejected_bounds = 0
    n_rejected_amplitude = 0
    n_rejected_nonfinite_axis = 0
    n_rejected_extra_events = 0
    n_rejected_snippet_qc = 0
    dedup_tolerance = max(0, int(args.microsaccade_dedup_tolerance_frames))
    for _, row in rows.iterrows():
        session = str(row["session"])
        trial_idx = int(row["trial_idx"])
        eyepos = eyepos_by_session[session]
        window_start = int(row["global_start"])
        window_stop = int(row["global_stop"])
        if window_stop <= window_start + 1:
            continue
        window_trace = np.asarray(eyepos[window_start:window_stop], dtype=np.float64)
        if window_trace.ndim != 2 or window_trace.shape[0] < 2 or window_trace.shape[1] != 2:
            continue
        duration_s = _finite_float(row.get("duration_s", row.get("epoch_duration_s", np.nan)), float("nan"))
        dt = (
            duration_s / float(max(1, window_trace.shape[0] - 1))
            if math.isfinite(duration_s) and duration_s > 0.0
            else float(bin_seconds)
        )
        threshold = (
            float(fixed_threshold)
            if fixed_threshold is not None
            else speed_threshold_mad(window_trace, dt=float(dt), z=float(args.microsaccade_threshold_z))
        )
        core_events, _core_mask, threshold = detect_microsaccade_events(
            window_trace,
            dt=float(dt),
            threshold_deg_s=float(threshold),
            min_samples=1,
            pad_samples=0,
        )
        padded_events, _padded_mask, _threshold = detect_microsaccade_events(
            window_trace,
            dt=float(dt),
            threshold_deg_s=float(threshold),
            min_samples=1,
            pad_samples=max(0, int(args.microsaccade_pad_frames)),
        )
        for event_index, event in enumerate(core_events):
            n_detected += 1
            onset_global = int(window_start + int(event["onset"]))
            offset_global = int(window_start + int(event["offset"]))
            peak_local = _event_peak_sample(window_trace, event, dt=float(dt))
            peak_global = int(window_start + peak_local)
            padded_event = _padded_event_for_core(
                padded_events,
                core_onset=int(event["onset"]),
                core_offset=int(event["offset"]),
            )
            padded_onset_global = int(window_start + int(padded_event["onset"]))
            padded_offset_global = int(window_start + int(padded_event["offset"]))
            snippet_start = int(onset_global - pre_frames)
            snippet_stop = int(offset_global + 1 + post_frames)
            if snippet_start < 0 or snippet_stop > int(eyepos.shape[0]) or snippet_stop <= snippet_start + 1:
                n_rejected_bounds += 1
                continue
            if bool(args.microsaccade_require_snippet_within_source_window) and (
                snippet_start < window_start or snippet_stop > window_stop
            ):
                n_rejected_bounds += 1
                continue
            axis_value = _finite_float(row[str(args.axis_column)], float("nan"))
            if not math.isfinite(axis_value):
                n_rejected_nonfinite_axis += 1
                continue
            amp_deg, amp_arcmin, direction_deg, amp_start, amp_stop = _event_amplitude(
                eyepos,
                onset_global=onset_global,
                offset_global=offset_global,
            )
            if amp_arcmin < min_amp or amp_arcmin > max_amp:
                n_rejected_amplitude += 1
                continue
            raw_snippet = np.asarray(eyepos[snippet_start:snippet_stop], dtype=np.float64)
            snippet_core_events, _snippet_core_mask, _snippet_threshold = detect_microsaccade_events(
                raw_snippet,
                dt=float(dt),
                threshold_deg_s=float(threshold),
                min_samples=1,
                pad_samples=0,
            )
            snippet_padded_events, _snippet_padded_mask, _snippet_padded_threshold = detect_microsaccade_events(
                raw_snippet,
                dt=float(dt),
                threshold_deg_s=float(threshold),
                min_samples=1,
                pad_samples=max(0, int(args.microsaccade_pad_frames)),
            )
            if bool(args.microsaccade_reject_extra_events) and len(snippet_core_events) != 1:
                n_rejected_extra_events += 1
                continue
            snippet_metrics = _centered_trace_metrics(raw_snippet)
            if (
                _over_positive_limit(
                    float(snippet_metrics["snippet_raw_rms_deg"]),
                    float(args.microsaccade_max_snippet_rms_deg),
                )
                or _over_positive_limit(
                    float(snippet_metrics["snippet_raw_max_radius_deg"]),
                    float(args.microsaccade_max_snippet_radius_deg),
                )
                or _over_positive_limit(
                    float(snippet_metrics["snippet_raw_path_length_deg"]),
                    float(args.microsaccade_max_snippet_path_length_deg),
                )
            ):
                n_rejected_snippet_qc += 1
                continue
            snippet_mean = np.nanmean(raw_snippet, axis=0)
            snippet_len = int(raw_snippet.shape[0])
            resample_den = float(max(1, snippet_len - 1))
            onset_in_snippet = int(onset_global - snippet_start)
            offset_in_snippet = int(offset_global - snippet_start)
            padded_onset_in_snippet = int(padded_onset_global - snippet_start)
            padded_offset_in_snippet = int(padded_offset_global - snippet_start)
            source_trace, trace_payload = _microsaccade_source_trace(
                raw_snippet,
                mode=str(args.microsaccade_trace_mode),
                n_timepoints=int(args.n_timepoints),
                core_onset_in_snippet=int(onset_in_snippet),
                core_offset_in_snippet=int(offset_in_snippet),
                padded_onset_in_snippet=int(padded_onset_in_snippet),
                padded_offset_in_snippet=int(padded_offset_in_snippet),
            )
            source_margin = min(int(snippet_start - window_start), int(window_stop - snippet_stop))
            payload = dict(row.to_dict())
            payload.update(
                {
                    "source_row": int(len(candidate_payloads)),
                    "parent_source_row": int(row["source_row"]),
                    "trial_id": int(len(candidate_payloads)),
                    "response_cache_path": "",
                    "source_trace_scale": 1.0,
                    "source_trace_prior_family": "real_microsaccade_snippet",
                    "source_trace_contract": str(trace_payload["microsaccade_trace_contract"]),
                    "source_trace": source_trace.astype(np.float32, copy=False),
                    "mean_x_deg": float(snippet_mean[0]),
                    "mean_y_deg": float(snippet_mean[1]),
                    "global_start": int(snippet_start),
                    "global_stop": int(snippet_stop),
                    "source_window_global_start": int(window_start),
                    "source_window_global_stop": int(window_stop),
                    "snippet_global_start": int(snippet_start),
                    "snippet_global_stop": int(snippet_stop),
                    "snippet_n_samples": int(snippet_len),
                    "snippet_duration_s": float((snippet_len - 1) * float(dt)),
                    "duration_s": float((snippet_len - 1) * float(dt)),
                    "n_samples": int(snippet_len),
                    "microsaccade_event_index": int(event_index),
                    "microsaccade_event_onset_global": int(onset_global),
                    "microsaccade_event_offset_global": int(offset_global),
                    "microsaccade_event_peak_global": int(peak_global),
                    "microsaccade_event_onset_global_padded": int(padded_onset_global),
                    "microsaccade_event_offset_global_padded": int(padded_offset_global),
                    "microsaccade_event_onset_frame_in_snippet": int(onset_in_snippet),
                    "microsaccade_event_offset_frame_in_snippet": int(offset_in_snippet),
                    "microsaccade_event_onset_frame_resampled": float(
                        onset_in_snippet * float(max(1, int(args.n_timepoints) - 1)) / resample_den
                    ),
                    "microsaccade_event_offset_frame_resampled": float(
                        offset_in_snippet * float(max(1, int(args.n_timepoints) - 1)) / resample_den
                    ),
                    "microsaccade_event_duration_samples": int(event["duration_samples"]),
                    "microsaccade_event_padded_duration_samples": int(padded_event["duration_samples"]),
                    **trace_payload,
                    "microsaccade_pre_frames": int(pre_frames),
                    "microsaccade_post_frames": int(post_frames),
                    "microsaccade_amplitude_deg": float(amp_deg),
                    "microsaccade_amplitude_arcmin": float(amp_arcmin),
                    "microsaccade_amplitude_start_global": int(amp_start),
                    "microsaccade_amplitude_stop_global": int(amp_stop),
                    "microsaccade_direction_deg": float(direction_deg),
                    "microsaccade_peak_speed_dps": float(event["peak_speed_deg_s"]),
                    "microsaccade_threshold_dps": float(threshold),
                    "microsaccade_detection_threshold_source": (
                        "fixed" if fixed_threshold is not None else "window_mad"
                    ),
                    "microsaccade_dedup_peak_tolerance_frames": int(dedup_tolerance),
                    "microsaccade_dedup_cluster_size": 1,
                    "microsaccade_dedup_source_margin_frames": int(source_margin),
                    "snippet_detected_core_event_count": int(len(snippet_core_events)),
                    "snippet_detected_padded_event_count": int(len(snippet_padded_events)),
                    "snippet_detected_event_count": int(len(snippet_core_events)),
                    **snippet_metrics,
                }
            )
            candidate_payloads.append(payload)

    payloads, dedup_meta = _dedupe_microsaccade_candidates(
        candidate_payloads,
        tolerance_frames=int(dedup_tolerance),
    )
    payloads, amplitude_sd_meta = _filter_microsaccades_by_amplitude_sd(
        payloads,
        sd_filter=float(args.microsaccade_amplitude_sd_filter),
    )

    meta = {
        "microsaccade_source_mode": "microsaccade_snippets",
        "n_source_windows_scanned": int(rows.shape[0]),
        "n_detected_events_before_filters": int(n_detected),
        **dedup_meta,
        **amplitude_sd_meta,
        "n_rejected_bounds": int(n_rejected_bounds),
        "n_rejected_amplitude": int(n_rejected_amplitude),
        "n_rejected_extra_events": int(n_rejected_extra_events),
        "n_rejected_nonfinite_axis": int(n_rejected_nonfinite_axis),
        "n_rejected_snippet_qc": int(n_rejected_snippet_qc),
        "n_available_microsaccade_snippets": int(len(payloads)),
        "microsaccade_trace_mode": str(args.microsaccade_trace_mode),
        "microsaccade_pre_frames": int(pre_frames),
        "microsaccade_post_frames": int(post_frames),
        "microsaccade_min_amplitude_arcmin": float(min_amp),
        "microsaccade_max_amplitude_arcmin": None if not math.isfinite(max_amp) else float(max_amp),
        "microsaccade_speed_threshold_dps": None if fixed_threshold is None else float(fixed_threshold),
        "microsaccade_threshold_z": float(args.microsaccade_threshold_z),
        "microsaccade_pad_frames": int(args.microsaccade_pad_frames),
        "microsaccade_dedup_tolerance_frames": int(dedup_tolerance),
        "microsaccade_require_snippet_within_source_window": bool(
            args.microsaccade_require_snippet_within_source_window
        ),
        "microsaccade_reject_extra_events": bool(args.microsaccade_reject_extra_events),
        "microsaccade_max_snippet_rms_deg": float(args.microsaccade_max_snippet_rms_deg),
        "microsaccade_max_snippet_radius_deg": float(args.microsaccade_max_snippet_radius_deg),
        "microsaccade_max_snippet_path_length_deg": float(args.microsaccade_max_snippet_path_length_deg),
        "microsaccade_detection_contract": (
            "Detect unpadded high-speed event cores in each source BackImage eye trace window, "
            "use padded events only as guardrail masks, deduplicate overlapping source-window "
            "candidates by session/trial and unpadded peak-speed frame, cut pre/event/post snippets "
            "from the original eye trace, center by snippet mean, and pass those real snippets to "
            "the same twin movie renderer used by the drift scale sweeps."
        ),
    }
    return payloads, meta


def select_source_trials(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    axis_run_dir = Path(args.axis_run_dir)
    run_metadata_path = axis_run_dir / "run_metadata.json"
    run_metadata = load_json(run_metadata_path) if run_metadata_path.exists() else {"config": {}}
    requested_mode = str(args.trial_source_mode)
    selected_path = Path(args.selected_windows_csv) if args.selected_windows_csv is not None else axis_run_dir / "selected_windows.csv"
    if args.selected_windows_csv is None and requested_mode == "microsaccade_snippets":
        configured_input = run_metadata.get("config", {}).get("input")
        if configured_input:
            candidate = Path(str(configured_input))
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            if candidate.exists():
                selected_path = candidate
    selected = pd.read_csv(selected_path)
    if "source_row" not in selected.columns:
        selected = selected.copy()
        selected["source_row"] = np.arange(selected.shape[0], dtype=int)

    has_manifest_tables = (axis_run_dir / "candidate_sets.csv").exists() and (axis_run_dir / "response_cache_manifest.csv").exists()
    source_mode = (
        "manifest"
        if requested_mode == "auto" and has_manifest_tables and str(args.sweep_mode) != "trace_bank"
        else requested_mode
    )
    if source_mode == "auto":
        source_mode = "selected_windows"

    source_scale = float(args.source_trace_scale)
    selected_by_source = selected.drop_duplicates("source_row").set_index("source_row", drop=False)
    trace_by_source: dict[int, np.ndarray] | None = None
    trace_source_contract = (
        "response_table_observed_trajectory_xy"
        if source_mode == "manifest"
        else "center_cropped_native_selected_window_trace_n_timepoints"
    )

    def reconstructed_trace(source_row: int) -> np.ndarray:
        nonlocal trace_by_source, trace_source_contract
        if trace_by_source is None:
            cfg = run_metadata.get("config", {})
            trace_dt = _finite_float(cfg.get("bin_seconds", np.nan), 1.0 / 120.0)
            if args.bin_seconds is not None:
                trace_dt = float(args.bin_seconds)
            eyepos_by_session = _session_dataset_cache(selected["session"].astype(str).to_list())
            bank, _bank_meta = build_native_snippet_trace_bank(
                selected,
                eyepos_by_session,
                int(args.n_timepoints),
                dt=float(trace_dt),
                microsaccade_speed_threshold_dps=(
                    float(args.microsaccade_speed_threshold_dps)
                    if args.microsaccade_speed_threshold_dps is not None
                    else (
                    float(cfg["microsaccade_speed_threshold_dps"])
                    if cfg.get("microsaccade_speed_threshold_dps") is not None
                    else None
                    )
                ),
                microsaccade_threshold_z=float(getattr(args, "microsaccade_threshold_z", cfg.get("microsaccade_threshold_z", 6.0))),
                microsaccade_pad_frames=int(getattr(args, "microsaccade_pad_frames", cfg.get("microsaccade_pad_frames", 1))),
            )
            trace_by_source = {int(item["source_row"]): np.asarray(item["trace"], dtype=np.float32) for item in bank}
            trace_source_contract = "center_cropped_native_selected_window_trace_n_timepoints"
        if int(source_row) not in trace_by_source:
            raise ValueError(f"Could not reconstruct trace for source_row={source_row}")
        trace = np.asarray(trace_by_source[int(source_row)], dtype=np.float32)
        if np.isclose(float(args.source_trace_scale), 1.0):
            return trace
        cfg = run_metadata.get("config", {})
        target_rms = float(args.source_trace_scale) * trace_rms(trace)
        max_rms = float(cfg.get("max_rms_deg", max(target_rms, 1.0)))
        scaled, _meta = _scale_to_rms(trace, target_rms, max_rms_deg=max_rms)
        return np.asarray(scaled, dtype=np.float32)

    trial_payloads: list[dict[str, Any]] = []
    start = max(0, int(args.trial_start))
    stop = None if int(args.max_trials) <= 0 else start + int(args.max_trials)

    if source_mode == "manifest":
        candidate_sets = pd.read_csv(axis_run_dir / "candidate_sets.csv")
        manifest = pd.read_csv(axis_run_dir / "response_cache_manifest.csv")
        rows = manifest[
            np.isclose(pd.to_numeric(manifest["scale"], errors="coerce").to_numpy(dtype=np.float64), source_scale)
            & (manifest["prior_family"].astype(str) == str(args.source_trace_prior_family))
        ].copy()
        if rows.empty:
            raise ValueError(
                f"No response-manifest rows for scale={source_scale:g}, "
                f"prior_family={args.source_trace_prior_family!r}"
            )
        rows = rows.drop_duplicates("trial_id", keep="first")
        rows = rows.merge(
            candidate_sets[
                ["trial_id", "observation_source_row", "candidate_set_mode", "candidate_ids", "candidate_indices"]
            ],
            on="trial_id",
            how="left",
            suffixes=("", "_candidate"),
        )
        if rows["observation_source_row"].isna().any():
            raise ValueError("Some manifest trial_id values were missing from candidate_sets.csv")

        for _, row in rows.sort_values("trial_id").iloc[start:stop].iterrows():
            source_row = int(row["observation_source_row"])
            if source_row not in selected_by_source.index:
                raise ValueError(f"source_row={source_row} from candidate_sets.csv not found in selected_windows.csv")
            selected_row = selected_by_source.loc[source_row]
            axis_value = float(selected_row[str(args.axis_column)])
            if not np.isfinite(axis_value):
                raise ValueError(f"source_row={source_row} has non-finite {args.axis_column!r}")
            response_path = axis_run_dir / str(row["response_cache_path"])
            if not response_path.exists():
                raise FileNotFoundError(response_path)
            with np.load(response_path, allow_pickle=False) as data:
                if "observed_trajectory_xy" in data.files:
                    source_trace = np.asarray(data["observed_trajectory_xy"], dtype=np.float32)
                else:
                    source_trace = reconstructed_trace(source_row)
            source_trace = _align_response_to_trace(source_trace, int(args.n_timepoints))
            payload = dict(selected_row.to_dict())
            payload.update(
                {
                    "trial_id": int(row["trial_id"]),
                    "response_cache_path": str(row["response_cache_path"]),
                    "source_trace_scale": source_scale,
                    "source_trace_prior_family": str(args.source_trace_prior_family),
                    "source_trace_contract": trace_source_contract,
                    "source_trace": source_trace,
                }
            )
            trial_payloads.append(payload)
        n_available_trials = int(rows.shape[0])
    elif source_mode == "selected_windows":
        rows = selected.drop_duplicates("source_row").copy()
        sort_cols = [col for col in ["session", "trial_idx", "source_row"] if col in rows.columns]
        if sort_cols:
            rows = rows.sort_values(sort_cols)
        rows = rows.iloc[start:stop].copy()
        for pos, (_, selected_row) in enumerate(rows.iterrows(), start=start):
            source_row = int(selected_row["source_row"])
            axis_value = float(selected_row[str(args.axis_column)])
            if not np.isfinite(axis_value):
                raise ValueError(f"source_row={source_row} has non-finite {args.axis_column!r}")
            source_trace = _align_response_to_trace(reconstructed_trace(source_row), int(args.n_timepoints))
            trial_id = int(selected_row["trial_id"]) if "trial_id" in selected_row.index and pd.notna(selected_row["trial_id"]) else int(pos)
            payload = dict(selected_row.to_dict())
            payload.update(
                {
                    "trial_id": trial_id,
                    "response_cache_path": "",
                    "source_trace_scale": source_scale,
                    "source_trace_prior_family": str(args.source_trace_prior_family),
                    "source_trace_contract": trace_source_contract,
                    "source_trace": source_trace,
                }
            )
            trial_payloads.append(payload)
        n_available_trials = int(selected_by_source.shape[0])
    elif source_mode == "microsaccade_snippets":
        all_payloads, micro_meta = build_microsaccade_snippet_trials(selected, run_metadata, args)
        selected_payloads = all_payloads[start:stop]
        trial_payloads.extend(selected_payloads)
        n_available_trials = int(len(all_payloads))
        trace_source_contract = "event_aligned_microsaccade_snippet_pre_event_post_tail_centered"
    else:
        raise ValueError(f"Unknown trial source mode {source_mode!r}")

    if not trial_payloads:
        raise ValueError("No trials selected.")
    trial_frame = pd.DataFrame(trial_payloads)
    trace_bank_specs: list[dict[str, Any]] | None = None
    trace_bank_meta: dict[str, Any] = {}
    if str(args.sweep_mode) == "trace_bank":
        bin_seconds = _finite_float(run_metadata.get("config", {}).get("bin_seconds", np.nan), 1.0 / 120.0)
        if args.bin_seconds is not None:
            bin_seconds = float(args.bin_seconds)
        trial_frame, trace_bank_specs, trace_bank_meta = build_trace_bank_assignments(
            trial_frame,
            selected,
            run_metadata,
            args,
            bin_seconds=float(bin_seconds),
        )
    meta = {
        "run_metadata_config": run_metadata.get("config", {}),
        "trial_source_mode": source_mode,
        "selected_windows_csv": str(selected_path),
        "n_available_scale1_trials": int(n_available_trials),
        "n_available_source_trials": int(n_available_trials),
        "n_selected_trials": int(trial_frame.shape[0]),
        "source_trace_contract": trace_source_contract,
    }
    if source_mode == "microsaccade_snippets":
        meta.update(micro_meta)
    if trace_bank_specs is not None:
        meta.update(trace_bank_meta)
    return trial_frame, meta


def cache_identity(args: argparse.Namespace, trials: pd.DataFrame, specs: list[dict[str, Any]], rr100_meta: dict[str, Any]) -> dict[str, Any]:
    trace_contracts = (
        sorted({str(v) for v in trials["source_trace_contract"].dropna().to_list()})
        if "source_trace_contract" in trials.columns
        else ["unknown"]
    )
    trace_prior_families = (
        sorted({str(v) for v in trials["source_trace_prior_family"].dropna().to_list()})
        if "source_trace_prior_family" in trials.columns
        else [str(args.source_trace_prior_family)]
    )
    trace_source_scales = (
        sorted({float(v) for v in pd.to_numeric(trials["source_trace_scale"], errors="coerce").dropna().to_list()})
        if "source_trace_scale" in trials.columns
        else [float(args.source_trace_scale)]
    )
    trace_bank_assignment_manifest = trace_bank_assignment_manifest_rows(trials)
    if str(args.sweep_mode) == "trace_bank":
        trace_contract = (
            "native real BackImage trace-bank sweep: each non-static condition is an unscaled "
            "real eye-position snippet sampled from a predeclared motion-metric bin. This mode "
            "does not multiply traces by display scale factors."
        )
        requested_prior_family = None
        requested_source_scale = None
    elif str(args.microsaccade_trace_mode) in EVENT_SCALED_MICROSACCADE_TRACE_MODES:
        event_anchor_modes = (
            sorted({str(v) for v in trials["source_trace_event_anchor_mode"].dropna().to_list()})
            if "source_trace_event_anchor_mode" in trials.columns
            else [microsaccade_trace_event_anchor_mode(str(args.microsaccade_trace_mode))]
        )
        trace_contract = (
            "event-aligned real BackImage microsaccade snippets; full pre/event/post traces are retained. "
            "Conditions scale only the detected microsaccade event increments and leave outside-event "
            "drift increments at 1x. Anchor mode(s): "
            f"{', '.join(event_anchor_modes)}. The 0x condition is a drift-retained event-removal "
            "reference, not a fully static movie."
        )
        requested_prior_family = None
        requested_source_scale = None
    elif any("microsaccade" in value for value in trace_contracts) or any(
        "real_microsaccade_snippet" in value for value in trace_prior_families
    ):
        trace_contract = (
            "event-aligned real BackImage microsaccade snippets; trace_source_contracts specify whether "
            "the full pre/event/post snippet or only a zero-rest event pulse is used. Conditions scale "
            "the resulting snippet directly for isotropic sweeps or decompose it into local contour "
            "along/across components for contour sweeps"
        )
        requested_prior_family = None
        requested_source_scale = None
    else:
        trace_contract = "scale-1 measured BackImage trace decomposed into local edge along/across components"
        requested_prior_family = str(args.source_trace_prior_family)
        requested_source_scale = float(args.source_trace_scale)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "analysis": "backimage_contour_axis_rr100_spatial_ssi",
        "axis_run_dir": str(Path(args.axis_run_dir).expanduser().resolve()),
        "rr100": rr100_meta,
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "trace_source_contracts": trace_contracts,
        "trace_contract": trace_contract,
        "sweep_mode": str(args.sweep_mode),
        "primary_ssi_metric": str(args.primary_ssi_metric),
        "ssi_metric_contracts": {
            "mean_map": (
                "Diagnostic SSI of the trajectory-averaged unit activation map; useful for "
                "activation-map visualization, not as the promoted motion-scale SSI metric."
            ),
            "time_resolved": (
                "Per-frame spatial SSI, rate-weighted over aligned trajectory samples. This is "
                "the promoted cached metric for motion-scale SSI."
            ),
        },
        "readout_contract": "average over aligned trajectory samples; no endpoint-only readout",
        "axis_column": str(args.axis_column),
        "source_trace_scale": trace_source_scales[0] if len(trace_source_scales) == 1 else trace_source_scales,
        "source_trace_prior_family": (
            trace_prior_families[0] if len(trace_prior_families) == 1 else trace_prior_families
        ),
        "source_trace_prior_families": trace_prior_families,
        "requested_source_trace_scale": requested_source_scale,
        "requested_source_trace_prior_family": requested_prior_family,
        "n_timepoints": int(args.n_timepoints),
        "microsaccade_trace_mode": str(args.microsaccade_trace_mode),
        "microsaccade_trace_event_anchor_mode": (
            microsaccade_trace_event_anchor_mode(str(args.microsaccade_trace_mode))
            if str(args.microsaccade_trace_mode) in EVENT_SCALED_MICROSACCADE_TRACE_MODES
            else None
        ),
        "microsaccade_pre_frames": int(args.microsaccade_pre_frames),
        "microsaccade_post_frames": int(args.microsaccade_post_frames),
        "microsaccade_min_amplitude_arcmin": float(args.microsaccade_min_amplitude_arcmin),
        "microsaccade_max_amplitude_arcmin": float(args.microsaccade_max_amplitude_arcmin),
        "microsaccade_amplitude_sd_filter": float(args.microsaccade_amplitude_sd_filter),
        "microsaccade_speed_threshold_dps": (
            None
            if args.microsaccade_speed_threshold_dps is None
            else float(args.microsaccade_speed_threshold_dps)
        ),
        "microsaccade_threshold_z": float(args.microsaccade_threshold_z),
        "microsaccade_pad_frames": int(args.microsaccade_pad_frames),
        "microsaccade_dedup_tolerance_frames": int(args.microsaccade_dedup_tolerance_frames),
        "microsaccade_require_snippet_within_source_window": bool(
            args.microsaccade_require_snippet_within_source_window
        ),
        "microsaccade_reject_extra_events": bool(args.microsaccade_reject_extra_events),
        "microsaccade_max_snippet_rms_deg": float(args.microsaccade_max_snippet_rms_deg),
        "microsaccade_max_snippet_radius_deg": float(args.microsaccade_max_snippet_radius_deg),
        "microsaccade_max_snippet_path_length_deg": float(args.microsaccade_max_snippet_path_length_deg),
        "patch_size_px": int(args.patch_size_px),
        "stimulus_rotation_deg": int(args.stimulus_rotation_deg),
        "stimulus_rotation_contract": (
            "Extracted patch, source trace, and analysis contour axis are rotated together by "
            "--stimulus-rotation-deg in gaze-frame coordinates before trace decomposition and scoring."
        ),
        "bin_seconds": None if args.bin_seconds is None else float(args.bin_seconds),
        "trial_ids": [int(v) for v in trials["trial_id"].to_list()],
        "source_rows": [int(v) for v in trials["source_row"].to_list()],
        "condition_specs": specs,
        "trace_bank_scale_metric": str(args.trace_bank_scale_metric) if str(args.sweep_mode) == "trace_bank" else None,
        "trace_bank_bins": str(args.trace_bank_bins) if str(args.sweep_mode) == "trace_bank" else None,
        "trace_bank_samples_per_bin": (
            int(args.trace_bank_samples_per_bin) if str(args.sweep_mode) == "trace_bank" else None
        ),
        "trace_bank_seed": int(args.trace_bank_seed) if str(args.sweep_mode) == "trace_bank" else None,
        "trace_bank_exclude_same_source": (
            bool(args.trace_bank_exclude_same_source) if str(args.sweep_mode) == "trace_bank" else None
        ),
        "trace_bank_assignment_manifest": trace_bank_assignment_manifest,
    }


def load_cache(path: Path, expected_identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            observed = str(np.asarray(data["cache_identity_json"]).ravel()[0])
            if observed != identity_text(expected_identity):
                return None
            return {key: np.asarray(data[key]) for key in data.files if key != "cache_identity_json"}
    except Exception:
        return None


def rate_map_for_trace(scorer: CanonicalTwinScorer, patch: np.ndarray, trace: np.ndarray) -> np.ndarray:
    image = _standardize_uint_like(patch)
    trace = np.asarray(trace, dtype=np.float32)
    if trace.shape[0] < int(scorer.common.N_LAGS):
        raise ValueError(
            f"Trace has {trace.shape[0]} samples, but the twin stimulus helper requires at least "
            f"{int(scorer.common.N_LAGS)} samples for its lag stack. Increase --n-timepoints."
        )
    full_stack = np.broadcast_to(
        image[None, :, :],
        (trace.shape[0] + scorer.common.N_LAGS + 1, *image.shape),
    ).copy()
    eye = scorer.torch.from_numpy(_trace_xy_to_twin_helper_order(trace))
    stim = scorer.common.make_counterfactual_stim(
        full_stack,
        eye,
        ppd=scorer.common.PPD,
        scale_factor=1.0,
        n_lags=scorer.common.N_LAGS,
        out_size=scorer.common.OUT_SIZE,
    )
    stim = (stim - 127.0) / 255.0
    rate_map = scorer._compute_rate_map_batched(stim)
    out = rate_map.detach().cpu().numpy().astype(np.float32, copy=False)
    del stim, rate_map
    if str(scorer.device).startswith("cuda") and scorer.torch.cuda.is_available():
        scorer.torch.cuda.empty_cache()
    return out


def trace_bank_condition_assignment(trial: pd.Series, spec: dict[str, Any]) -> dict[str, Any]:
    assignments = trial.get("trace_bank_assignments", {})
    if not isinstance(assignments, dict):
        raise ValueError("Trace-bank condition requested, but trial has no trace_bank_assignments dictionary.")
    condition_id = str(spec["condition_id"])
    if condition_id not in assignments:
        raise ValueError(f"Missing trace-bank assignment for condition {condition_id!r}.")
    return assignments[condition_id]


def trace_bank_condition_meta(
    assignment: dict[str, Any],
    trace: np.ndarray,
    image_source_trace: np.ndarray,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "trace_source_contract": "native_real_backimage_trace_bank_unscaled",
        "image_source_trace_rms_deg": trace_rms(image_source_trace),
        "image_source_trace_path_length_deg": trace_path_length(image_source_trace),
        "source_trace_rms_deg": trace_rms(trace),
        "source_trace_path_length_deg": trace_path_length(trace),
        "along_component_rms_deg": float("nan"),
        "across_component_rms_deg": float("nan"),
        "output_trace_rms_deg": trace_rms(trace),
        "output_trace_path_length_deg": trace_path_length(trace),
        "output_trace_start_x_deg": float(trace[0, 0]) if trace.shape[0] else float("nan"),
        "output_trace_start_y_deg": float(trace[0, 1]) if trace.shape[0] else float("nan"),
        "output_trace_end_x_deg": float(trace[-1, 0]) if trace.shape[0] else float("nan"),
        "output_trace_end_y_deg": float(trace[-1, 1]) if trace.shape[0] else float("nan"),
        "output_trace_mean_x_deg": float(np.mean(trace[:, 0])) if trace.shape[0] else float("nan"),
        "output_trace_mean_y_deg": float(np.mean(trace[:, 1])) if trace.shape[0] else float("nan"),
        "event_scale_mask_enabled": False,
        "event_scale_anchor_mode": "none",
        "output_trace_centering": "bank_trace_mean_centered",
        "event_scale_sample_count": 0,
        "event_scale_step_count": 0,
        "outside_event_drift_scale": float("nan"),
        "trace_bank_condition": True,
    }
    for key, value in assignment.items():
        if key == "trace":
            continue
        if isinstance(value, np.ndarray):
            continue
        meta[f"trace_bank_{key}" if not str(key).startswith("trace_bank_") else str(key)] = value
    return meta


def compute_cache(
    args: argparse.Namespace,
    *,
    trials: pd.DataFrame,
    specs: list[dict[str, Any]],
    population_view: Any,
    bin_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(args.batch_size), empty_cache_every_batch=True)
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}

    unit_bits_by_condition: list[list[np.ndarray]] = [[] for _ in specs]
    unit_mean_map_bits_by_condition: list[list[np.ndarray]] = [[] for _ in specs]
    unit_time_resolved_bits_by_condition: list[list[np.ndarray]] = [[] for _ in specs]
    unit_spikes_by_condition: list[list[np.ndarray]] = [[] for _ in specs]
    unit_rates_by_condition: list[list[np.ndarray]] = [[] for _ in specs]
    pop_by_condition: list[list[float]] = [[] for _ in specs]
    pop_mean_map_by_condition: list[list[float]] = [[] for _ in specs]
    pop_time_resolved_by_condition: list[list[float]] = [[] for _ in specs]
    map_sum_by_condition: list[np.ndarray | None] = [None for _ in specs]
    inventory_rows: list[dict[str, Any]] = []

    total = int(trials.shape[0]) * len(specs)
    rotation_deg = int(args.stimulus_rotation_deg)
    done = 0
    for movie_idx, (_, trial) in enumerate(trials.iterrows()):
        patch, patch_meta = _extract_patch(
            trial,
            canvas_cache=canvas_cache,
            patch_size_px=int(args.patch_size_px),
        )
        original_axis_deg = float(trial[str(args.axis_column)])
        source_trace = np.asarray(trial["source_trace"], dtype=np.float32)
        if rotation_deg:
            patch = rotate_patch_gaze_frame(patch, rotation_deg)
            source_trace = rotate_trace_xy(source_trace, rotation_deg)
        event_scale_mask = trial_event_scale_mask(trial, int(source_trace.shape[0]))
        event_anchor_mode = trial_event_anchor_mode(trial)
        axis_deg = rotated_axis_deg(original_axis_deg, rotation_deg)
        for condition_idx, spec in enumerate(specs):
            done += 1
            if bool(spec.get("trace_bank_condition", False)):
                assignment = trace_bank_condition_assignment(trial, spec)
                trace = np.asarray(assignment["trace"], dtype=np.float32)
                if rotation_deg:
                    trace = rotate_trace_xy(trace, rotation_deg)
                trace_meta = trace_bank_condition_meta(assignment, trace, source_trace)
                trace_meta["original_axis_deg"] = original_axis_deg
                trace_meta["axis_deg"] = axis_deg
                trace_meta["stimulus_rotation_deg"] = rotation_deg
                trace_meta["along_scale"] = float("nan")
                trace_meta["across_scale"] = float("nan")
            elif bool(spec["is_static_baseline"]) and event_scale_mask is None:
                trace = np.zeros_like(source_trace, dtype=np.float32)
                trace_meta = {
                    "source_trace_rms_deg": trace_rms(source_trace),
                    "source_trace_path_length_deg": trace_path_length(source_trace),
                    "along_component_rms_deg": 0.0,
                    "across_component_rms_deg": 0.0,
                    "output_trace_rms_deg": 0.0,
                    "output_trace_path_length_deg": 0.0,
                    "original_axis_deg": original_axis_deg,
                    "axis_deg": axis_deg,
                    "stimulus_rotation_deg": rotation_deg,
                    "along_scale": 0.0,
                    "across_scale": 0.0,
                }
            else:
                trace, trace_meta = combined_axis_trace(
                    source_trace,
                    axis_deg=axis_deg,
                    along_scale=float(spec["along_scale"]),
                    across_scale=float(spec["across_scale"]),
                    event_scale_mask=event_scale_mask,
                    event_anchor_mode=event_anchor_mode,
                )
                trace_meta["original_axis_deg"] = original_axis_deg
                trace_meta["stimulus_rotation_deg"] = rotation_deg

            print(
                f"[backimage-contour-ssi] {done}/{total} trial={int(trial['trial_id'])} "
                f"source_row={int(trial['source_row'])} condition={spec['condition_id']}",
                flush=True,
            )
            full_map = rate_map_for_trace(scorer, patch, trace)
            full_map = _align_response_to_trace(full_map, int(args.n_timepoints))
            rr100_map = apply_population_view(full_map, population_view).astype(np.float32, copy=False)
            del full_map
            time_resolved_ssi = unit_spatial_ssi_for_movie(rr100_map, bin_seconds=float(bin_seconds))
            mean_map = np.mean(rr100_map, axis=0).astype(np.float32, copy=False)
            mean_map_ssi = unit_spatial_ssi_for_mean_map(
                mean_map,
                unit_weights=np.asarray(time_resolved_ssi["unit_expected_spikes"], dtype=np.float64),
            )
            primary_ssi = mean_map_ssi if str(args.primary_ssi_metric) == "mean_map" else time_resolved_ssi
            if map_sum_by_condition[condition_idx] is None:
                map_sum_by_condition[condition_idx] = np.zeros_like(mean_map, dtype=np.float64)
            map_sum_by_condition[condition_idx] += mean_map
            unit_bits_by_condition[condition_idx].append(np.asarray(primary_ssi["unit_bits_per_spike"], dtype=np.float32))
            unit_mean_map_bits_by_condition[condition_idx].append(
                np.asarray(mean_map_ssi["unit_bits_per_spike"], dtype=np.float32)
            )
            unit_time_resolved_bits_by_condition[condition_idx].append(
                np.asarray(time_resolved_ssi["unit_bits_per_spike"], dtype=np.float32)
            )
            unit_spikes_by_condition[condition_idx].append(
                np.asarray(time_resolved_ssi["unit_expected_spikes"], dtype=np.float32)
            )
            unit_rates_by_condition[condition_idx].append(np.asarray(time_resolved_ssi["unit_mean_rate"], dtype=np.float32))
            pop_by_condition[condition_idx].append(float(primary_ssi["population_bits_per_spike"]))
            pop_mean_map_by_condition[condition_idx].append(float(mean_map_ssi["population_bits_per_spike"]))
            pop_time_resolved_by_condition[condition_idx].append(float(time_resolved_ssi["population_bits_per_spike"]))
            inventory_rows.append(
                {
                    "movie_index": int(movie_idx),
                    "condition_index": int(condition_idx),
                    "condition_id": str(spec["condition_id"]),
                    "condition_label": str(spec["condition_label"]),
                    "trial_id": int(trial["trial_id"]),
                    "source_row": int(trial["source_row"]),
                    "session": str(trial["session"]),
                    "trial_idx": int(trial["trial_idx"]),
                    "response_cache_path": str(trial["response_cache_path"]),
                    **window_inventory_payload(trial),
                    **{k: v for k, v in patch_meta.items()},
                    **trace_meta,
                }
            )
            del rr100_map, mean_map

    n_movies = max(int(trials.shape[0]), 1)
    stats = {
        "primary_ssi_metric": np.asarray([str(args.primary_ssi_metric)]),
        "unit_bits_per_movie": np.stack([np.stack(rows, axis=0) for rows in unit_bits_by_condition], axis=0).astype(np.float32),
        "unit_mean_map_bits_per_movie": np.stack([np.stack(rows, axis=0) for rows in unit_mean_map_bits_by_condition], axis=0).astype(np.float32),
        "unit_time_resolved_bits_per_movie": np.stack([np.stack(rows, axis=0) for rows in unit_time_resolved_bits_by_condition], axis=0).astype(np.float32),
        "unit_expected_spikes_per_movie": np.stack([np.stack(rows, axis=0) for rows in unit_spikes_by_condition], axis=0).astype(np.float32),
        "unit_mean_rate_per_movie": np.stack([np.stack(rows, axis=0) for rows in unit_rates_by_condition], axis=0).astype(np.float32),
        "population_bits_per_movie": np.stack([np.asarray(rows, dtype=np.float32) for rows in pop_by_condition], axis=0),
        "population_mean_map_bits_per_movie": np.stack([np.asarray(rows, dtype=np.float32) for rows in pop_mean_map_by_condition], axis=0),
        "population_time_resolved_bits_per_movie": np.stack([np.asarray(rows, dtype=np.float32) for rows in pop_time_resolved_by_condition], axis=0),
        "mean_rate_map": np.stack([(m / float(n_movies)).astype(np.float32) for m in map_sum_by_condition], axis=0),
        "condition_id": np.asarray([str(spec["condition_id"]) for spec in specs]),
        "condition_label": np.asarray([str(spec["condition_label"]) for spec in specs]),
        "along_scale": np.asarray([float(spec["along_scale"]) for spec in specs], dtype=np.float32),
        "across_scale": np.asarray([float(spec["across_scale"]) for spec in specs], dtype=np.float32),
        "motion_scale": np.asarray([float(spec.get("motion_scale", spec["across_scale"])) for spec in specs], dtype=np.float32),
        "sweep_mode": np.asarray([str(spec.get("sweep_mode", "across")) for spec in specs]),
        "trace_bank_condition": np.asarray([bool(spec.get("trace_bank_condition", False)) for spec in specs], dtype=bool),
        "trace_bank_scale_metric": np.asarray([str(spec.get("trace_bank_scale_metric", "")) for spec in specs]),
        "trace_bank_bin_label": np.asarray([str(spec.get("trace_bank_bin_label", "")) for spec in specs]),
        "trace_bank_metric_low": np.asarray([float(spec.get("trace_bank_metric_low", np.nan)) for spec in specs], dtype=np.float32),
        "trace_bank_metric_high": np.asarray([float(spec.get("trace_bank_metric_high", np.nan)) for spec in specs], dtype=np.float32),
        "trace_bank_metric_median": np.asarray([float(spec.get("trace_bank_metric_median", np.nan)) for spec in specs], dtype=np.float32),
        "trace_bank_n_members": np.asarray([int(spec.get("trace_bank_n_members", 0)) for spec in specs], dtype=np.int32),
        "trace_bank_sample_index": np.asarray([int(spec.get("trace_bank_sample_index", -1)) for spec in specs], dtype=np.int32),
        "is_static_baseline": np.asarray([bool(spec["is_static_baseline"]) for spec in specs], dtype=bool),
        "is_across_sweep": np.asarray([bool(spec["is_across_sweep"]) for spec in specs], dtype=bool),
        "movie_trial_id": trials["trial_id"].to_numpy(dtype=np.int32),
        "movie_source_row": trials["source_row"].to_numpy(dtype=np.int32),
    }
    return stats, inventory_rows


def save_cache(path: Path, stats: dict[str, Any], identity: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **stats, cache_identity_json=np.asarray([identity_text(identity)]))


def summarize(stats: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    primary_ssi_metric = str(np.asarray(stats.get("primary_ssi_metric", ["time_resolved"])).astype(str).ravel()[0])
    unit_bits = np.asarray(stats["unit_bits_per_movie"], dtype=np.float64)
    unit_mean_map_bits = np.asarray(stats.get("unit_mean_map_bits_per_movie", unit_bits), dtype=np.float64)
    unit_time_resolved_bits = np.asarray(stats.get("unit_time_resolved_bits_per_movie", unit_bits), dtype=np.float64)
    unit_spikes = np.asarray(stats["unit_expected_spikes_per_movie"], dtype=np.float64)
    unit_rates = np.asarray(stats["unit_mean_rate_per_movie"], dtype=np.float64)
    pop = np.asarray(stats["population_bits_per_movie"], dtype=np.float64)
    pop_mean_map = np.asarray(stats.get("population_mean_map_bits_per_movie", pop), dtype=np.float64)
    pop_time_resolved = np.asarray(stats.get("population_time_resolved_bits_per_movie", pop), dtype=np.float64)
    condition_id = np.asarray(stats["condition_id"]).astype(str)
    condition_label = np.asarray(stats["condition_label"]).astype(str)
    along_scale = np.asarray(stats["along_scale"], dtype=np.float64)
    across_scale = np.asarray(stats["across_scale"], dtype=np.float64)
    motion_scale = np.asarray(stats.get("motion_scale", across_scale), dtype=np.float64)
    sweep_mode = np.asarray(stats.get("sweep_mode", np.asarray(["across"] * condition_id.size))).astype(str)
    trace_bank_condition = np.asarray(stats.get("trace_bank_condition", np.zeros(condition_id.size, dtype=bool)), dtype=bool)
    trace_bank_scale_metric = np.asarray(stats.get("trace_bank_scale_metric", np.asarray([""] * condition_id.size))).astype(str)
    trace_bank_bin_label = np.asarray(stats.get("trace_bank_bin_label", np.asarray([""] * condition_id.size))).astype(str)
    trace_bank_metric_low = np.asarray(stats.get("trace_bank_metric_low", np.full(condition_id.size, np.nan)), dtype=np.float64)
    trace_bank_metric_high = np.asarray(stats.get("trace_bank_metric_high", np.full(condition_id.size, np.nan)), dtype=np.float64)
    trace_bank_metric_median = np.asarray(stats.get("trace_bank_metric_median", np.full(condition_id.size, np.nan)), dtype=np.float64)
    trace_bank_n_members = np.asarray(stats.get("trace_bank_n_members", np.zeros(condition_id.size)), dtype=np.int64)
    trace_bank_sample_index = np.asarray(stats.get("trace_bank_sample_index", np.full(condition_id.size, -1)), dtype=np.int64)
    is_static = np.asarray(stats["is_static_baseline"], dtype=bool)
    is_sweep = np.asarray(stats["is_across_sweep"], dtype=bool)

    bits_mean = np.nanmean(unit_bits, axis=1)
    bits_sem = sem(unit_bits, axis=1)
    mean_map_bits_mean = np.nanmean(unit_mean_map_bits, axis=1)
    mean_map_bits_sem = sem(unit_mean_map_bits, axis=1)
    time_resolved_bits_mean = np.nanmean(unit_time_resolved_bits, axis=1)
    time_resolved_bits_sem = sem(unit_time_resolved_bits, axis=1)
    rates_mean = np.nanmean(unit_rates, axis=1)
    rates_sem = sem(unit_rates, axis=1)
    pop_mean = np.nanmean(pop, axis=1)
    pop_sem = sem(pop, axis=1)
    pop_mean_map_mean = np.nanmean(pop_mean_map, axis=1)
    pop_mean_map_sem = sem(pop_mean_map, axis=1)
    pop_time_resolved_mean = np.nanmean(pop_time_resolved, axis=1)
    pop_time_resolved_sem = sem(pop_time_resolved, axis=1)

    static_idx = int(np.flatnonzero(is_static)[0]) if np.any(is_static) else -1
    across1_candidates = np.flatnonzero(is_sweep & np.isclose(along_scale, 1.0) & np.isclose(across_scale, 1.0))
    across1_idx = int(across1_candidates[0]) if across1_candidates.size else -1

    total_expected = np.sum(unit_spikes, axis=2)
    total_bits = np.sum(unit_spikes * unit_bits, axis=2)
    n_conditions, _n_movies, n_units = unit_bits.shape
    loo_pop = np.zeros((n_units, n_conditions), dtype=np.float64)
    for unit_idx in range(n_units):
        numer = total_bits - unit_spikes[:, :, unit_idx] * unit_bits[:, :, unit_idx]
        denom = np.maximum(total_expected - unit_spikes[:, :, unit_idx], EPS)
        loo_pop[unit_idx] = np.nanmean(numer / denom, axis=1)
    loo_abs_delta = loo_pop - pop_mean[None, :]
    max_abs_loo = np.nanmax(np.abs(loo_abs_delta), axis=1)
    max_abs_unit_delta = np.nanmax(
        np.abs(bits_mean - (bits_mean[static_idx][None, :] if static_idx >= 0 else np.nanmean(bits_mean, axis=0)[None, :])),
        axis=0,
    )

    condition_rows: list[dict[str, Any]] = []
    for idx in range(n_conditions):
        condition_rows.append(
            {
                "condition_index": int(idx),
                "condition_id": str(condition_id[idx]),
                "condition_label": str(condition_label[idx]),
                "is_static_baseline": bool(is_static[idx]),
                "is_across_sweep": bool(is_sweep[idx]),
                "along_scale": float(along_scale[idx]),
                "across_scale": float(across_scale[idx]),
                "motion_scale": float(motion_scale[idx]),
                "sweep_mode": str(sweep_mode[idx]),
                "trace_bank_condition": bool(trace_bank_condition[idx]),
                "trace_bank_scale_metric": str(trace_bank_scale_metric[idx]),
                "trace_bank_bin_label": str(trace_bank_bin_label[idx]),
                "trace_bank_metric_low": float(trace_bank_metric_low[idx]),
                "trace_bank_metric_high": float(trace_bank_metric_high[idx]),
                "trace_bank_metric_median": float(trace_bank_metric_median[idx]),
                "trace_bank_n_members": int(trace_bank_n_members[idx]),
                "trace_bank_sample_index": int(trace_bank_sample_index[idx]),
                "primary_ssi_metric": primary_ssi_metric,
                "population_ssi_bits_per_spike_mean": float(pop_mean[idx]),
                "population_ssi_bits_per_spike_sem": float(pop_sem[idx]),
                "population_mean_map_ssi_bits_per_spike_mean": float(pop_mean_map_mean[idx]),
                "population_mean_map_ssi_bits_per_spike_sem": float(pop_mean_map_sem[idx]),
                "population_time_resolved_ssi_bits_per_spike_mean": float(pop_time_resolved_mean[idx]),
                "population_time_resolved_ssi_bits_per_spike_sem": float(pop_time_resolved_sem[idx]),
                "population_delta_vs_static": float(pop_mean[idx] - pop_mean[static_idx]) if static_idx >= 0 else float("nan"),
                "population_delta_vs_across1": float(pop_mean[idx] - pop_mean[across1_idx]) if across1_idx >= 0 else float("nan"),
                "population_mean_map_delta_vs_static": (
                    float(pop_mean_map_mean[idx] - pop_mean_map_mean[static_idx]) if static_idx >= 0 else float("nan")
                ),
                "population_time_resolved_delta_vs_static": (
                    float(pop_time_resolved_mean[idx] - pop_time_resolved_mean[static_idx])
                    if static_idx >= 0
                    else float("nan")
                ),
                "n_movies": int(pop.shape[1]),
                "n_units": int(n_units),
            }
        )

    unit_rows: list[dict[str, Any]] = []
    for idx in range(n_conditions):
        for unit_idx in range(n_units):
            unit_rows.append(
                {
                    "condition_index": int(idx),
                    "condition_id": str(condition_id[idx]),
                    "condition_label": str(condition_label[idx]),
                    "is_static_baseline": bool(is_static[idx]),
                    "is_across_sweep": bool(is_sweep[idx]),
                    "along_scale": float(along_scale[idx]),
                    "across_scale": float(across_scale[idx]),
                    "motion_scale": float(motion_scale[idx]),
                    "sweep_mode": str(sweep_mode[idx]),
                    "trace_bank_condition": bool(trace_bank_condition[idx]),
                    "trace_bank_scale_metric": str(trace_bank_scale_metric[idx]),
                    "trace_bank_bin_label": str(trace_bank_bin_label[idx]),
                    "trace_bank_metric_median": float(trace_bank_metric_median[idx]),
                    "trace_bank_sample_index": int(trace_bank_sample_index[idx]),
                    "unit_index": int(unit_idx),
                    "unit_label": f"u{int(unit_idx):03d}",
                    "primary_ssi_metric": primary_ssi_metric,
                    "unit_ssi_bits_per_spike_mean": float(bits_mean[idx, unit_idx]),
                    "unit_ssi_bits_per_spike_sem": float(bits_sem[idx, unit_idx]),
                    "unit_mean_map_ssi_bits_per_spike_mean": float(mean_map_bits_mean[idx, unit_idx]),
                    "unit_mean_map_ssi_bits_per_spike_sem": float(mean_map_bits_sem[idx, unit_idx]),
                    "unit_time_resolved_ssi_bits_per_spike_mean": float(time_resolved_bits_mean[idx, unit_idx]),
                    "unit_time_resolved_ssi_bits_per_spike_sem": float(time_resolved_bits_sem[idx, unit_idx]),
                    "unit_mean_rate_mean": float(rates_mean[idx, unit_idx]),
                    "unit_mean_rate_sem": float(rates_sem[idx, unit_idx]),
                    "unit_expected_spikes_mean": float(np.nanmean(unit_spikes[idx, :, unit_idx])),
                    "unit_ssi_delta_vs_static": (
                        float(bits_mean[idx, unit_idx] - bits_mean[static_idx, unit_idx])
                        if static_idx >= 0
                        else float("nan")
                    ),
                    "unit_mean_rate_delta_vs_static": (
                        float(rates_mean[idx, unit_idx] - rates_mean[static_idx, unit_idx])
                        if static_idx >= 0
                        else float("nan")
                    ),
                    "unit_mean_map_ssi_delta_vs_static": (
                        float(mean_map_bits_mean[idx, unit_idx] - mean_map_bits_mean[static_idx, unit_idx])
                        if static_idx >= 0
                        else float("nan")
                    ),
                    "unit_time_resolved_ssi_delta_vs_static": (
                        float(time_resolved_bits_mean[idx, unit_idx] - time_resolved_bits_mean[static_idx, unit_idx])
                        if static_idx >= 0
                        else float("nan")
                    ),
                }
            )

    top_rows = [
        {
            "unit_index": int(unit_idx),
            "unit_label": f"u{int(unit_idx):03d}",
            "primary_ssi_metric": primary_ssi_metric,
            "max_abs_leave_one_out_population_ssi_delta": float(max_abs_loo[unit_idx]),
            "max_abs_unit_ssi_delta_vs_static": float(max_abs_unit_delta[unit_idx]),
            "static_unit_ssi_bits_per_spike_mean": (
                float(bits_mean[static_idx, unit_idx]) if static_idx >= 0 else float("nan")
            ),
            "static_unit_mean_map_ssi_bits_per_spike_mean": (
                float(mean_map_bits_mean[static_idx, unit_idx]) if static_idx >= 0 else float("nan")
            ),
            "static_unit_time_resolved_ssi_bits_per_spike_mean": (
                float(time_resolved_bits_mean[static_idx, unit_idx]) if static_idx >= 0 else float("nan")
            ),
            "static_unit_mean_rate_mean": (
                float(rates_mean[static_idx, unit_idx]) if static_idx >= 0 else float("nan")
            ),
        }
        for unit_idx in range(n_units)
    ]
    top_rows.sort(
        key=lambda row: (
            float(row["max_abs_leave_one_out_population_ssi_delta"]),
            float(row["max_abs_unit_ssi_delta_vs_static"]),
        ),
        reverse=True,
    )

    sweep_indices = np.flatnonzero(is_sweep)
    peak_summary: dict[str, Any] = {}
    if sweep_indices.size:
        sweep_pop = pop_mean[sweep_indices]
        best_local = int(np.nanargmax(sweep_pop))
        best_idx = int(sweep_indices[best_local])
        peak_motion_scale = float(motion_scale[best_idx])
        if not np.isfinite(peak_motion_scale):
            peak_motion_scale = float(max(abs(along_scale[best_idx]), abs(across_scale[best_idx])))
        peak_summary = {
            "primary_ssi_metric": primary_ssi_metric,
            "sweep_peak_condition_id": str(condition_id[best_idx]),
            "sweep_mode": str(sweep_mode[best_idx]),
            "sweep_peak_along_scale": float(along_scale[best_idx]),
            "sweep_peak_across_scale": float(across_scale[best_idx]),
            "sweep_peak_motion_scale": peak_motion_scale,
            "sweep_peak_population_ssi_bits_per_spike": float(pop_mean[best_idx]),
            "sweep_peak_is_below_1x": (
                bool(peak_motion_scale < 1.0) if str(sweep_mode[best_idx]) != "trace_bank" else None
            ),
            "trace_bank_scale_metric": (
                str(trace_bank_scale_metric[best_idx]) if str(sweep_mode[best_idx]) == "trace_bank" else ""
            ),
            "trace_bank_bin_label": (
                str(trace_bank_bin_label[best_idx]) if str(sweep_mode[best_idx]) == "trace_bank" else ""
            ),
            "across1_condition_present": bool(across1_idx >= 0),
            "across1_population_ssi_bits_per_spike": float(pop_mean[across1_idx]) if across1_idx >= 0 else float("nan"),
        }

    diagnostics = {
        "primary_ssi_metric": primary_ssi_metric,
        "bits_mean": bits_mean,
        "bits_sem": bits_sem,
        "mean_map_bits_mean": mean_map_bits_mean,
        "time_resolved_bits_mean": time_resolved_bits_mean,
        "rates_mean": rates_mean,
        "pop_mean": pop_mean,
        "pop_sem": pop_sem,
        "pop_mean_map_mean": pop_mean_map_mean,
        "pop_time_resolved_mean": pop_time_resolved_mean,
        "loo_pop": loo_pop,
        "loo_abs_delta": loo_abs_delta,
        "top_rows": top_rows,
        "static_idx": static_idx,
        "across1_idx": across1_idx,
        "peak_summary": peak_summary,
    }
    return condition_rows, unit_rows, top_rows, diagnostics


def _sf_axis_mode(row: dict[str, Any]) -> str:
    sweep = str(row.get("sweep_mode", "across"))
    if sweep == "isotropic":
        return "isotropic_sweep"
    if sweep == "across":
        return "across_sweep"
    if sweep in {"grid", "pairs"}:
        along = _finite_float(row.get("along_scale", np.nan), float("nan"))
        across = _finite_float(row.get("across_scale", np.nan), float("nan"))
        if math.isfinite(along) and np.isclose(along, 0.0):
            return "across_sweep_along0"
        if math.isfinite(along) and np.isclose(along, 1.0):
            return "across_sweep_along1"
        if math.isfinite(across) and np.isclose(across, 0.0):
            return "along_sweep_across0"
        if math.isfinite(across) and np.isclose(across, 1.0):
            return "along_sweep_across1"
    return f"{sweep}_sweep"


def sf_compatible_unit_rows(unit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in unit_rows:
        display_scale = _finite_float(row.get("motion_scale", row.get("across_scale", np.nan)), float("nan"))
        if not math.isfinite(display_scale):
            display_scale = max(
                abs(_finite_float(row.get("along_scale", 0.0), 0.0)),
                abs(_finite_float(row.get("across_scale", 0.0), 0.0)),
            )
        rows.append(
            {
                "unit_index": int(row["unit_index"]),
                "unit_label": str(row["unit_label"]),
                "condition_index": int(row["condition_index"]),
                "condition_id": str(row["condition_id"]),
                "condition_label": str(row["condition_label"]),
                "axis_mode": _sf_axis_mode(row),
                "display_scale": float(display_scale),
                "along_scale": float(row["along_scale"]),
                "across_scale": float(row["across_scale"]),
                "motion_scale": float(display_scale),
                "sweep_mode": str(row.get("sweep_mode", "")),
                "is_static_baseline": bool(row.get("is_static_baseline", False)),
                "trace_bank_condition": bool(row.get("trace_bank_condition", False)),
                "trace_bank_scale_metric": str(row.get("trace_bank_scale_metric", "")),
                "trace_bank_bin_label": str(row.get("trace_bank_bin_label", "")),
                "trace_bank_metric_median": float(row.get("trace_bank_metric_median", np.nan)),
                "trace_bank_sample_index": int(row.get("trace_bank_sample_index", -1)),
                "displayed_movie_time_resolved_ssi_bits_per_spike": float(
                    row["unit_time_resolved_ssi_bits_per_spike_mean"]
                ),
                "displayed_movie_mean_map_ssi_bits_per_spike": float(row["unit_mean_map_ssi_bits_per_spike_mean"]),
                "displayed_movie_primary_ssi_bits_per_spike": float(row["unit_ssi_bits_per_spike_mean"]),
                "displayed_movie_mean_rate": float(row["unit_mean_rate_mean"]),
                "displayed_movie_expected_spikes_arbitrary_dt": float(row["unit_expected_spikes_mean"]),
                "source_table_contract": (
                    "Derived from contour-axis unit_ssi_table.csv; one row per unit and motion-scale "
                    "condition, suitable for SF-group scale-curve summaries."
                ),
            }
        )
    return rows


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


def highlighted_unit_color_map(highlighted_units: list[int]) -> dict[int, Any]:
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(highlighted_units), 1)))
    return {int(unit_index): colors[pos] for pos, unit_index in enumerate(highlighted_units)}


def finite_row_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, np.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def paired_finite_row_values(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        try:
            x_value = float(row.get(x_key, np.nan))
            y_value = float(row.get(y_key, np.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x_value) and math.isfinite(y_value):
            xs.append(x_value)
            ys.append(y_value)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def trace_bank_row_metric_value(row: dict[str, Any], key: str) -> float:
    value = _finite_float(row.get(key, np.nan))
    if math.isfinite(value):
        return value
    if key.endswith("_speed_mean_arcmin_s"):
        return _finite_float(row.get(key.replace("_arcmin_s", "_deg_s"), np.nan)) * 60.0
    if key.endswith("_speed_median_arcmin_s"):
        return _finite_float(row.get(key.replace("_arcmin_s", "_deg_s"), np.nan)) * 60.0
    if key.endswith("_speed_p95_arcmin_s"):
        return _finite_float(row.get(key.replace("_arcmin_s", "_deg_s"), np.nan)) * 60.0
    if key.endswith("_path_speed_arcmin_s"):
        direct = _finite_float(row.get(key.replace("_path_speed_arcmin_s", "_path_length_deg_s"), np.nan))
        if math.isfinite(direct):
            return direct * 60.0
        path = _finite_float(row.get(key.replace("_path_speed_arcmin_s", "_path_length_arcmin"), np.nan))
        duration = _finite_float(row.get("snippet_duration_s", row.get("duration_s", np.nan)))
        if math.isfinite(path) and math.isfinite(duration) and duration > 0.0:
            return path / duration
    if key.endswith("_bcea68_arcmin2") or key.endswith("_cov_axis_ratio") or key.endswith("_cov_orientation_deg"):
        prefix = key.split("_", 1)[0] + "_"
        cov_payload = covariance_component_payload(row, prefix)
        return _finite_float(cov_payload.get(key, np.nan))
    if key == "trace_cov_anisotropy":
        value = _finite_float(row.get("rendered_anisotropy", row.get("source_anisotropy", np.nan)))
        if math.isfinite(value):
            return value
    return float("nan")


def trace_bank_metric_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = [trace_bank_row_metric_value(row, key) for row in rows]
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def trace_bank_microsaccade_mask(rows: list[dict[str, Any]]) -> np.ndarray:
    values: list[bool] = []
    for row in rows:
        n_events = _finite_float(row.get("n_microsaccade_events", row.get("rendered_n_microsaccade_events", 0.0)), 0.0)
        values.append(bool(n_events > 0.0))
    return np.asarray(values, dtype=bool)


def trace_bank_path_length_context_windows(
    trace_bank_rows: list[dict[str, Any]],
    *,
    n_windows: int = 6,
) -> list[dict[str, Any]]:
    if not trace_bank_rows:
        return []
    path = np.asarray([trace_bank_row_metric_value(row, "rendered_path_length_arcmin") for row in trace_bank_rows], dtype=np.float64)
    if not np.isfinite(path).any():
        path = np.asarray([trace_bank_row_metric_value(row, "path_length_arcmin") for row in trace_bank_rows], dtype=np.float64)
    diffusion = np.asarray(
        [trace_bank_row_metric_value(row, "rendered_diffusion_constant_arcmin2_s") for row in trace_bank_rows],
        dtype=np.float64,
    )
    if not np.isfinite(diffusion).any():
        diffusion = np.asarray(
            [trace_bank_row_metric_value(row, "rendered_diffusion_constant_deg2_s") for row in trace_bank_rows],
            dtype=np.float64,
        ) * 3600.0
    finite_path = np.isfinite(path)
    if not finite_path.any():
        return []
    path_series = pd.Series(path[finite_path])
    q = min(int(n_windows), int(path_series.nunique(dropna=True)))
    if q <= 0:
        return []
    if q == 1:
        codes = pd.Series(np.zeros(path_series.shape[0], dtype=int), index=path_series.index)
    else:
        codes = pd.qcut(path_series, q=q, labels=False, duplicates="drop")
    finite_indices = np.flatnonzero(finite_path)
    rows: list[dict[str, Any]] = []
    for code in sorted(pd.Series(codes).dropna().unique()):
        member_indices = finite_indices[np.asarray(codes == code)]
        path_values = path[member_indices]
        diff_values = diffusion[member_indices]
        diff_finite = diff_values[np.isfinite(diff_values)]
        diff_positive = diff_finite[diff_finite > 0.0]
        row: dict[str, Any] = {
            "path_length_window_index": int(code),
            "path_length_window_label": f"q{int(code) + 1:02d}",
            "n_trace_bank_members": int(member_indices.size),
            "path_length_low_arcmin": float(np.nanmin(path_values)),
            "path_length_q25_arcmin": float(np.nanpercentile(path_values, 25.0)),
            "path_length_median_arcmin": float(np.nanmedian(path_values)),
            "path_length_q75_arcmin": float(np.nanpercentile(path_values, 75.0)),
            "path_length_high_arcmin": float(np.nanmax(path_values)),
            "diffusion_finite_n": int(diff_finite.size),
            "diffusion_positive_n": int(diff_positive.size),
            "diffusion_zero_or_negative_fraction": float(np.count_nonzero(diff_finite <= 0.0) / diff_finite.size)
            if diff_finite.size
            else float("nan"),
        }
        if diff_positive.size:
            row.update(
                {
                    "diffusion_positive_q25_arcmin2_s": float(np.nanpercentile(diff_positive, 25.0)),
                    "diffusion_positive_q40_arcmin2_s": float(np.nanpercentile(diff_positive, 40.0)),
                    "diffusion_positive_median_arcmin2_s": float(np.nanmedian(diff_positive)),
                    "diffusion_positive_q60_arcmin2_s": float(np.nanpercentile(diff_positive, 60.0)),
                    "diffusion_positive_q75_arcmin2_s": float(np.nanpercentile(diff_positive, 75.0)),
                }
            )
        rows.append(row)
    return rows


def add_path_length_context_bands(
    ax: Any,
    context_rows: list[dict[str, Any]] | None,
    *,
    x_metric: str,
    alpha: float = 0.11,
) -> None:
    if not context_rows:
        return
    label_used = False
    for row in context_rows:
        if str(x_metric) == "rendered_diffusion_constant_arcmin2_s":
            low = _finite_float(row.get("diffusion_positive_q40_arcmin2_s", row.get("diffusion_positive_q25_arcmin2_s", np.nan)))
            high = _finite_float(row.get("diffusion_positive_q60_arcmin2_s", row.get("diffusion_positive_q75_arcmin2_s", np.nan)))
            center = _finite_float(row.get("diffusion_positive_median_arcmin2_s", np.nan))
        else:
            low = _finite_float(row.get("path_length_q25_arcmin", row.get("path_length_low_arcmin", np.nan)))
            high = _finite_float(row.get("path_length_q75_arcmin", row.get("path_length_high_arcmin", np.nan)))
            center = _finite_float(row.get("path_length_median_arcmin", np.nan))
        if not (math.isfinite(low) and math.isfinite(high) and high > low):
            continue
        if str(x_metric) == "rendered_diffusion_constant_arcmin2_s" and low <= 0.0:
            continue
        ax.axvspan(
            low,
            high,
            color="#8f8f8f",
            alpha=float(alpha),
            linewidth=0,
            zorder=0,
            label="path-length windows" if not label_used else None,
        )
        label_used = True
        if math.isfinite(center) and center > 0.0:
            ax.axvline(center, color="#777777", alpha=0.28, linewidth=0.75, zorder=1)


def trace_bank_metric_summary_rows(trace_bank_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not trace_bank_rows:
        return []
    ms_mask = trace_bank_microsaccade_mask(trace_bank_rows)
    row_groups = [
        ("all", np.ones(len(trace_bank_rows), dtype=bool)),
        ("no_detected_microsaccade", ~ms_mask),
        ("with_detected_microsaccade", ms_mask),
    ]
    out: list[dict[str, Any]] = []
    for metric_key, label, unit in TRACE_BANK_METRIC_SUMMARY_SPECS:
        all_values = np.asarray([trace_bank_row_metric_value(row, metric_key) for row in trace_bank_rows], dtype=np.float64)
        for group_label, group_mask in row_groups:
            values = all_values[group_mask]
            values = values[np.isfinite(values)]
            row: dict[str, Any] = {
                "group": group_label,
                "metric": metric_key,
                "label": label,
                "unit": unit,
                "n_rows": int(np.count_nonzero(group_mask)),
                "finite_n": int(values.size),
            }
            if values.size:
                row.update(
                    {
                        "mean": float(np.nanmean(values)),
                        "std": float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0,
                        "min": float(np.nanmin(values)),
                        "q25": float(np.nanquantile(values, 0.25)),
                        "median": float(np.nanmedian(values)),
                        "q75": float(np.nanquantile(values, 0.75)),
                        "q95": float(np.nanquantile(values, 0.95)),
                        "q99": float(np.nanquantile(values, 0.99)),
                        "max": float(np.nanmax(values)),
                        "zero_or_negative_fraction": float(np.mean(values <= 0.0)),
                    }
                )
            out.append(row)
    return out


def _plot_grouped_hist(
    ax: Any,
    no_ms: np.ndarray,
    with_ms: np.ndarray,
    *,
    title: str,
    xlabel: str,
    log_positive: bool,
    bins: np.ndarray | int | None = None,
    context_rows: list[dict[str, Any]] | None = None,
    context_x_metric: str | None = None,
) -> None:
    no_ms = np.asarray(no_ms, dtype=np.float64)
    with_ms = np.asarray(with_ms, dtype=np.float64)
    no_ms = no_ms[np.isfinite(no_ms)]
    with_ms = with_ms[np.isfinite(with_ms)]
    if no_ms.size == 0 and with_ms.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "no finite values", ha="center", va="center", fontsize=9)
        return
    if log_positive:
        combined = np.concatenate([no_ms[no_ms > 0.0], with_ms[with_ms > 0.0]])
        if combined.size:
            lo = float(np.nanmin(combined))
            hi = float(np.nanmax(combined))
            bins = np.geomspace(lo, hi, 26) if bins is None and lo > 0.0 and hi > lo else bins
        no_hist = no_ms[no_ms > 0.0]
        with_hist = with_ms[with_ms > 0.0]
    else:
        combined = np.concatenate([no_ms, with_ms])
        if bins is None:
            bins, _ = histogram_bins(combined)
        no_hist = no_ms
        with_hist = with_ms
    if bins is None:
        bins = 16
    if context_x_metric:
        add_path_length_context_bands(ax, context_rows, x_metric=str(context_x_metric))
    if no_hist.size:
        ax.hist(
            no_hist,
            bins=bins,
            weights=np.full(no_hist.shape, 1.0 / float(no_hist.size)),
            color="#4c78a8",
            alpha=0.34,
            edgecolor="#4c78a8",
            linewidth=1.0,
            label=(
                f"no ms (n={no_ms.size}; <=0={no_ms.size - no_hist.size})"
                if log_positive
                else f"no ms (n={no_ms.size})"
            ),
        )
    if with_hist.size:
        ax.hist(
            with_hist,
            bins=bins,
            weights=np.full(with_hist.shape, 1.0 / float(with_hist.size)),
            histtype="step",
            color="#c44e52",
            linewidth=1.6,
            label=(
                f">=1 ms (n={with_ms.size}; <=0={with_ms.size - with_hist.size})"
                if log_positive
                else f">=1 ms (n={with_ms.size})"
            ),
        )
    if log_positive and isinstance(bins, np.ndarray):
        ax.set_xscale("log")
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction per bin")
    ax.grid(True, color="#e7e7e7", linewidth=0.7)
    ax.legend(loc="best", fontsize=6.5, frameon=False)


def plot_trace_bank_metric_summary_panel(
    out_dir: Path,
    trace_bank_rows: list[dict[str, Any]],
    *,
    dpi: int,
) -> tuple[Path, Path] | None:
    if not trace_bank_rows:
        return None
    ms_mask = trace_bank_microsaccade_mask(trace_bank_rows)
    context_rows = trace_bank_path_length_context_windows(trace_bank_rows)
    metrics = [
        ("path_length_arcmin", "Path length", "arcmin", True, None),
        ("observed_rms_arcmin", "Spatial spread", "RMS radius (arcmin)", True, None),
        ("rendered_diffusion_constant_arcmin2_s", "MSD diffusion", "arcmin^2/s", True, None),
        ("trace_cov_anisotropy", "Covariance anisotropy", "(major-minor)/(major+minor)", False, np.linspace(0.0, 1.0, 21)),
        ("rendered_bcea68_arcmin2", "BCEA68", "arcmin^2", True, None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2))
    for ax, (metric_key, title, xlabel, log_positive, bins) in zip(axes.ravel()[:5], metrics):
        values = np.asarray([trace_bank_row_metric_value(row, metric_key) for row in trace_bank_rows], dtype=np.float64)
        _plot_grouped_hist(
            ax,
            values[~ms_mask],
            values[ms_mask],
            title=title,
            xlabel=xlabel,
            log_positive=log_positive,
            bins=bins,
            context_rows=context_rows,
            context_x_metric=(
                "path_length_arcmin"
                if metric_key == "path_length_arcmin"
                else "rendered_diffusion_constant_arcmin2_s"
                if metric_key == "rendered_diffusion_constant_arcmin2_s"
                else None
            ),
        )
    ax = axes.ravel()[5]
    events = np.asarray(
        [
            _finite_float(row.get("n_microsaccade_events", row.get("rendered_n_microsaccade_events", 0.0)), 0.0)
            for row in trace_bank_rows
        ],
        dtype=np.float64,
    )
    finite_events = events[np.isfinite(events)]
    if finite_events.size:
        clipped = np.minimum(finite_events.astype(int), 5)
        counts = np.asarray([np.count_nonzero(clipped == idx) for idx in range(6)], dtype=np.float64)
        labels = ["0", "1", "2", "3", "4", ">=5"]
        ax.bar(np.arange(6), counts, color="#777777", alpha=0.82)
        ax.set_xticks(np.arange(6))
        ax.set_xticklabels(labels)
        ax.set_xlabel("detected microsaccades per snippet")
        ax.set_ylabel("snippets")
        ax.set_title("Microsaccade contamination", fontsize=9.5)
        ax.grid(True, axis="y", color="#e7e7e7", linewidth=0.7)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no microsaccade counts", ha="center", va="center", fontsize=9)
    fig.suptitle("BackImage trace-bank scale, shape, and contamination metrics", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png_path = out_dir / "trace_bank_metric_summary_panel.png"
    pdf_path = out_dir / "trace_bank_metric_summary_panel.pdf"
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def histogram_bins(values: np.ndarray) -> tuple[np.ndarray | int, bool]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 10, False
    n_bins = int(np.clip(np.sqrt(float(finite.size)) * 1.8, 10, 48))
    positive = finite[finite > 0.0]
    if positive.size > 1 and positive.size >= max(2, int(0.8 * finite.size)):
        lo = float(np.nanmin(positive))
        hi = float(np.nanmax(positive))
        if lo > 0.0 and hi / lo >= 25.0:
            return np.geomspace(lo, hi, n_bins + 1), True
    return n_bins, False


def plot_trace_bank_diffusion_distribution(
    out_dir: Path,
    trace_bank_rows: list[dict[str, Any]],
    *,
    scale_metric: str,
    bin_summary: list[dict[str, Any]] | None,
    dpi: int,
) -> tuple[Path, Path] | None:
    if not trace_bank_rows:
        return None

    rendered_arcmin = finite_row_values(trace_bank_rows, "rendered_diffusion_constant_arcmin2_s")
    if rendered_arcmin.size == 0:
        rendered_arcmin = finite_row_values(trace_bank_rows, "rendered_diffusion_constant_deg2_s") * 3600.0
    source_arcmin = finite_row_values(trace_bank_rows, "source_diffusion_constant_arcmin2_s")
    if source_arcmin.size == 0:
        source_arcmin = finite_row_values(trace_bank_rows, "source_diffusion_constant_deg2_s") * 3600.0
    if rendered_arcmin.size == 0:
        return None
    context_rows = trace_bank_path_length_context_windows(trace_bank_rows)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    ax_hist, ax_cdf, ax_bins, ax_scatter = axes.ravel()

    hist_values = rendered_arcmin if source_arcmin.size == 0 else np.concatenate([rendered_arcmin, source_arcmin])
    positive_hist_values = hist_values[hist_values > 0.0]
    use_log_x = False
    if positive_hist_values.size:
        lo = float(np.nanmin(positive_hist_values))
        hi = float(np.nanmax(positive_hist_values))
        use_log_x = lo > 0.0 and hi / lo >= 25.0
    if use_log_x:
        bins = np.geomspace(lo, hi, 31)
    else:
        bins, use_log_x = histogram_bins(hist_values)
    rendered_hist = rendered_arcmin[rendered_arcmin > 0.0] if use_log_x else rendered_arcmin
    source_hist = source_arcmin[source_arcmin > 0.0] if use_log_x else source_arcmin
    rendered_zero = int(rendered_arcmin.size - rendered_hist.size) if use_log_x else 0
    source_zero = int(source_arcmin.size - source_hist.size) if use_log_x and source_arcmin.size else 0
    add_path_length_context_bands(ax_hist, context_rows, x_metric="rendered_diffusion_constant_arcmin2_s")
    ax_hist.hist(
        rendered_hist,
        bins=bins,
        color="#3b82b8",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.55,
        label=(
            f"rendered positive (n={rendered_hist.size}; <=0={rendered_zero})"
            if use_log_x
            else f"rendered snippet (n={rendered_hist.size})"
        ),
    )
    if source_hist.size:
        ax_hist.hist(
            source_hist,
            bins=bins,
            histtype="step",
            color="#c44e52",
            linewidth=1.45,
            label=(
                f"source positive (n={source_hist.size}; <=0={source_zero})"
                if use_log_x
                else f"source snippet metric (n={source_hist.size})"
            ),
        )
    if use_log_x:
        ax_hist.set_xscale("log")
    ax_hist.set_title("Diffusion constant distribution", fontsize=10)
    ax_hist.set_xlabel("diffusion constant (arcmin^2/s)")
    ax_hist.set_ylabel("trace snippets")
    ax_hist.grid(True, color="#e7e7e7", linewidth=0.7)
    ax_hist.legend(loc="best", fontsize=7, frameon=False)

    rendered_cdf_values = rendered_hist if use_log_x else rendered_arcmin
    source_cdf_values = source_hist if use_log_x else source_arcmin
    rendered_sorted = np.sort(rendered_cdf_values)
    rendered_cdf = np.arange(1, rendered_sorted.size + 1, dtype=np.float64) / float(rendered_sorted.size)
    add_path_length_context_bands(ax_cdf, context_rows, x_metric="rendered_diffusion_constant_arcmin2_s")
    ax_cdf.plot(rendered_sorted, rendered_cdf, color="#3b82b8", linewidth=1.8, label="rendered")
    if source_cdf_values.size:
        source_sorted = np.sort(source_cdf_values)
        source_cdf = np.arange(1, source_sorted.size + 1, dtype=np.float64) / float(source_sorted.size)
        ax_cdf.plot(source_sorted, source_cdf, color="#c44e52", linewidth=1.3, label="source")
    if use_log_x:
        ax_cdf.set_xscale("log")
    ax_cdf.set_title("Positive empirical CDF" if use_log_x else "Empirical CDF", fontsize=10)
    ax_cdf.set_xlabel("diffusion constant (arcmin^2/s)")
    ax_cdf.set_ylabel("fraction <= x")
    ax_cdf.set_ylim(0.0, 1.02)
    ax_cdf.grid(True, color="#e7e7e7", linewidth=0.7)
    ax_cdf.legend(loc="best", fontsize=7, frameon=False)
    if use_log_x:
        zero_fraction = float(rendered_zero) / float(rendered_arcmin.size) if rendered_arcmin.size else 0.0
        ax_cdf.text(
            0.04,
            0.08,
            f"rendered <=0: {zero_fraction:.1%}",
            transform=ax_cdf.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="#444444",
        )

    bins_payload = list(bin_summary or [])
    if bins_payload:
        labels = [str(row.get("bin_label", f"bin{idx + 1:02d}")) for idx, row in enumerate(bins_payload)]
        counts = np.asarray([float(row.get("n_trace_bank_members", 0)) for row in bins_payload], dtype=np.float64)
        ax_bins.bar(np.arange(len(labels)), counts, color="#4c9f70", alpha=0.82)
        ax_bins.set_xticks(np.arange(len(labels)))
        ax_bins.set_xticklabels(labels, rotation=35, ha="right")
        ax_bins.set_ylabel("eligible snippets")
        ax_bins.set_xlabel(str(scale_metric))
        for idx, row in enumerate(bins_payload):
            try:
                median = float(row.get("metric_median", np.nan))
            except (TypeError, ValueError):
                median = float("nan")
            if math.isfinite(median):
                median_arcmin = median * 3600.0 if str(scale_metric).endswith("_deg2_s") else median
                ax_bins.text(
                    idx,
                    counts[idx],
                    f"{median_arcmin:.2g}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#333333",
                )
        ax_bins.set_title("Trace-bank bins; labels show median in plotted units", fontsize=10)
        ax_bins.grid(True, axis="y", color="#e7e7e7", linewidth=0.7)
    else:
        ax_bins.axis("off")
        ax_bins.text(0.5, 0.5, "no bin summary", ha="center", va="center", fontsize=10)

    source_pair, rendered_pair = paired_finite_row_values(
        trace_bank_rows,
        "source_diffusion_constant_arcmin2_s",
        "rendered_diffusion_constant_arcmin2_s",
    )
    if source_pair.size == 0 or rendered_pair.size == 0:
        source_deg, rendered_deg = paired_finite_row_values(
            trace_bank_rows,
            "source_diffusion_constant_deg2_s",
            "rendered_diffusion_constant_deg2_s",
        )
        source_pair = source_deg * 3600.0
        rendered_pair = rendered_deg * 3600.0
    if source_pair.size:
        ax_scatter.scatter(source_pair, rendered_pair, s=12, alpha=0.48, color="#575757", linewidths=0)
        finite_all = np.concatenate([source_pair[np.isfinite(source_pair)], rendered_pair[np.isfinite(rendered_pair)]])
        positive = finite_all[finite_all > 0.0]
        if positive.size:
            lo = float(np.nanmin(positive))
            hi = float(np.nanmax(positive))
            ax_scatter.plot([lo, hi], [lo, hi], color="#999999", linestyle=":", linewidth=1.0)
            if lo > 0.0 and hi / lo >= 25.0:
                ax_scatter.set_xscale("log")
                ax_scatter.set_yscale("log")
        ax_scatter.set_title("Source vs rendered snippet metric", fontsize=10)
        ax_scatter.set_xlabel("source diffusion constant (arcmin^2/s)")
        ax_scatter.set_ylabel("rendered diffusion constant (arcmin^2/s)")
        ax_scatter.grid(True, color="#e7e7e7", linewidth=0.7)
    else:
        ax_scatter.axis("off")
        ax_scatter.text(0.5, 0.5, "source diffusion constants unavailable", ha="center", va="center", fontsize=10)

    fig.suptitle(
        f"BackImage trace-bank diffusion constants\nscale metric: {scale_metric}; "
        f"rendered snippets are the exact traces sent to the twin",
        fontsize=12,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png_path = out_dir / "trace_bank_diffusion_constant_distribution.png"
    pdf_path = out_dir / "trace_bank_diffusion_constant_distribution.pdf"
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_trace_bank_microsaccade_diffusion_histogram(
    out_dir: Path,
    trace_bank_rows: list[dict[str, Any]],
    *,
    dpi: int,
) -> tuple[Path, Path] | None:
    if not trace_bank_rows:
        return None

    diffusion = finite_row_values(trace_bank_rows, "rendered_diffusion_constant_arcmin2_s")
    if diffusion.size == 0:
        diffusion = finite_row_values(trace_bank_rows, "rendered_diffusion_constant_deg2_s") * 3600.0
    if diffusion.size == 0:
        return None

    events: list[float] = []
    values: list[float] = []
    for row in trace_bank_rows:
        try:
            value = float(row.get("rendered_diffusion_constant_arcmin2_s", np.nan))
        except (TypeError, ValueError):
            try:
                value = float(row.get("rendered_diffusion_constant_deg2_s", np.nan)) * 3600.0
            except (TypeError, ValueError):
                value = float("nan")
        try:
            n_events = float(row.get("n_microsaccade_events", row.get("rendered_n_microsaccade_events", 0)))
        except (TypeError, ValueError):
            n_events = 0.0
        if math.isfinite(value) and math.isfinite(n_events):
            values.append(value)
            events.append(n_events)
    if not values:
        return None

    values_arr = np.asarray(values, dtype=np.float64)
    events_arr = np.asarray(events, dtype=np.float64)
    no_ms = values_arr[events_arr <= 0.0]
    with_ms = values_arr[events_arr > 0.0]
    if no_ms.size == 0 or with_ms.size == 0:
        return None
    context_rows = trace_bank_path_length_context_windows(trace_bank_rows)

    positive_all = values_arr[values_arr > 0.0]
    if positive_all.size == 0:
        return None
    lo = float(np.nanmin(positive_all))
    hi = float(np.nanmax(positive_all))
    use_log_x = lo > 0.0 and hi / lo >= 25.0
    bins: np.ndarray | int
    if use_log_x:
        bins = np.geomspace(lo, hi, 31)
        no_ms_hist = no_ms[no_ms > 0.0]
        with_ms_hist = with_ms[with_ms > 0.0]
    else:
        bins, _ = histogram_bins(values_arr)
        no_ms_hist = no_ms
        with_ms_hist = with_ms

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    add_path_length_context_bands(ax, context_rows, x_metric="rendered_diffusion_constant_arcmin2_s")
    ax.hist(
        no_ms_hist,
        bins=bins,
        weights=np.full(no_ms_hist.shape, 1.0 / float(max(1, no_ms_hist.size))),
        histtype="stepfilled",
        color="#4c78a8",
        alpha=0.34,
        edgecolor="#4c78a8",
        linewidth=1.25,
        label=(
            f"no detected microsaccade (n={no_ms.size}; <=0={no_ms.size - no_ms_hist.size})"
            if use_log_x
            else f"no detected microsaccade (n={no_ms.size})"
        ),
    )
    ax.hist(
        with_ms_hist,
        bins=bins,
        weights=np.full(with_ms_hist.shape, 1.0 / float(max(1, with_ms_hist.size))),
        histtype="step",
        color="#c44e52",
        linewidth=1.8,
        label=(
            f">=1 detected microsaccade (n={with_ms.size}; <=0={with_ms.size - with_ms_hist.size})"
            if use_log_x
            else f">=1 detected microsaccade (n={with_ms.size})"
        ),
    )
    if use_log_x:
        ax.set_xscale("log")
    ax.set_title("Diffusion constants split by detected microsaccades", fontsize=11)
    ax.set_xlabel("rendered diffusion constant (arcmin^2/s)")
    ax.set_ylabel("fraction of positive snippets per bin" if use_log_x else "fraction of snippets per bin")
    ax.grid(True, color="#e7e7e7", linewidth=0.7)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    png_path = out_dir / "trace_bank_diffusion_constant_by_microsaccade_histogram.png"
    pdf_path = out_dir / "trace_bank_diffusion_constant_by_microsaccade_histogram.pdf"
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_summary(
    out_dir: Path,
    stats: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    top_units: int,
    dpi: int,
    map_vmin_percentile: float,
    map_vmax_percentile: float,
) -> tuple[Path, Path]:
    condition_id = np.asarray(stats["condition_id"]).astype(str)
    labels = np.asarray(stats["condition_label"]).astype(str)
    across = np.asarray(stats["across_scale"], dtype=np.float64)
    motion_scale = np.asarray(stats.get("motion_scale", across), dtype=np.float64)
    sweep_mode_values = np.asarray(stats.get("sweep_mode", np.asarray(["across"] * labels.size))).astype(str)
    is_sweep = np.asarray(stats["is_across_sweep"], dtype=bool)
    sweep_idx = np.flatnonzero(is_sweep)
    if sweep_idx.size == 0:
        raise ValueError("No across-sweep conditions to plot.")
    plot_mode = str(sweep_mode_values[sweep_idx[0]])

    primary_ssi_metric = str(diagnostics.get("primary_ssi_metric", "time_resolved"))
    metric_label = "mean-map SSI diagnostic" if primary_ssi_metric == "mean_map" else "spike-weighted time-resolved SSI"
    metric_note = (
        "diagnostic metric: SSI of trajectory-averaged activation maps; spike-weighted time-resolved SSI also cached"
        if primary_ssi_metric == "mean_map"
        else "primary metric: per-frame SSI rate-weighted over trajectory; mean-map diagnostic also cached"
    )
    bits_mean = np.asarray(diagnostics["bits_mean"], dtype=np.float64)
    pop_mean = np.asarray(diagnostics["pop_mean"], dtype=np.float64)
    mean_maps = np.asarray(stats["mean_rate_map"], dtype=np.float32)
    top_rows = diagnostics["top_rows"][: max(1, int(top_units))]
    highlighted = [int(row["unit_index"]) for row in top_rows]
    colors = highlighted_unit_color_map(highlighted)

    n_rows = len(highlighted)
    n_cols = int(sweep_idx.size)
    fig_height = max(8.0, 4.6 + 0.55 * n_rows)
    fig = plt.figure(figsize=(12.6, fig_height))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.0, max(1.2, 0.13 * n_rows * n_cols)],
        hspace=0.34,
        wspace=0.18,
    )
    ax_all = fig.add_subplot(gs[0, 0])
    ax_high = fig.add_subplot(gs[0, 1])
    if plot_mode == "isotropic":
        x = motion_scale[sweep_idx]
        xlabel = "isotropic measured-trace motion scale"
        xtick_labels: list[str] | None = None
    elif plot_mode in {"grid", "pairs"}:
        x = np.arange(sweep_idx.size, dtype=np.float64)
        xlabel = "condition pair (along/across scale)"
        xtick_labels = [str(labels[idx]) for idx in sweep_idx]
    elif plot_mode == "trace_bank":
        x = motion_scale[sweep_idx]
        if not np.all(np.isfinite(x)):
            x = np.arange(sweep_idx.size, dtype=np.float64)
        metric_values = np.asarray(stats.get("trace_bank_scale_metric", np.asarray([""] * labels.size))).astype(str)
        metric = str(metric_values[sweep_idx[0]]) if metric_values.size else ""
        xlabel = f"native trace-bank metric ({metric})" if metric else "native trace-bank metric"
        xtick_labels = [str(labels[idx]) for idx in sweep_idx]
    else:
        x = across[sweep_idx]
        xlabel = "across-contour motion scale, along=1"
        xtick_labels = None

    for unit_idx in range(bits_mean.shape[1]):
        ax_all.plot(x, bits_mean[sweep_idx, unit_idx], color="#a8a8a8", linewidth=0.65, alpha=0.33, zorder=1)
    for unit_idx in highlighted:
        ax_all.plot(
            x,
            bits_mean[sweep_idx, unit_idx],
            marker="o",
            linewidth=1.3,
            markersize=3.2,
            color=colors[int(unit_idx)],
            label=f"u{int(unit_idx):03d}",
            zorder=3,
        )
    ax_all.plot(x, pop_mean[sweep_idx], color="black", marker="o", linewidth=2.0, markersize=4.2, label="population", zorder=4)
    if plot_mode != "trace_bank":
        ax_all.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax_all.set_title("All RR100 units; largest leave-one-out influences highlighted", fontsize=10)
    ax_all.set_xlabel(xlabel)
    ax_all.set_ylabel(f"{metric_label} (bits/spike)")
    if xtick_labels is not None:
        ax_all.set_xticks(x)
        ax_all.set_xticklabels(xtick_labels, rotation=55, ha="right", fontsize=6.2)
    ax_all.grid(True, color="#e6e6e6", linewidth=0.7)
    ax_all.legend(loc="best", fontsize=6.6, ncol=3, frameon=False)

    for unit_idx in highlighted:
        ax_high.plot(
            x,
            bits_mean[sweep_idx, unit_idx],
            marker="o",
            linewidth=1.55,
            markersize=3.6,
            color=colors[int(unit_idx)],
            label=f"u{int(unit_idx):03d}",
        )
    ax_high.plot(x, pop_mean[sweep_idx], color="black", marker="o", linewidth=2.2, markersize=4.4, label="population")
    if plot_mode != "trace_bank":
        ax_high.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax_high.set_title("Highlighted units only", fontsize=10)
    ax_high.set_xlabel(xlabel)
    ax_high.set_ylabel(f"{metric_label} (bits/spike)")
    if xtick_labels is not None:
        ax_high.set_xticks(x)
        ax_high.set_xticklabels(xtick_labels, rotation=55, ha="right", fontsize=6.2)
    ax_high.grid(True, color="#e6e6e6", linewidth=0.7)
    ax_high.legend(loc="best", fontsize=6.6, ncol=3, frameon=False)

    map_gs = gs[1, :].subgridspec(
        nrows=n_rows + 1,
        ncols=n_cols + 1,
        width_ratios=[0.85, *([1.0] * n_cols)],
        height_ratios=[0.23, *([1.0] * n_rows)],
        hspace=0.03,
        wspace=0.04,
    )
    header_ax = fig.add_subplot(map_gs[0, 0])
    header_ax.axis("off")
    header_ax.text(0.98, 0.2, "unit", ha="right", va="center", fontsize=6.5, color="#555555")
    for col, cond_idx in enumerate(sweep_idx, start=1):
        ax = fig.add_subplot(map_gs[0, col])
        ax.axis("off")
        ax.text(0.5, 0.2, str(labels[cond_idx]), ha="center", va="center", fontsize=6.5, color="#555555")

    shown_images = [mean_maps[int(cond_idx), int(unit_idx)] for unit_idx in highlighted for cond_idx in sweep_idx]
    global_vmin, global_vmax = image_scale(shown_images, map_vmin_percentile, map_vmax_percentile)
    del global_vmin, global_vmax
    for row_idx, unit_idx in enumerate(highlighted, start=1):
        label_ax = fig.add_subplot(map_gs[row_idx, 0])
        label_ax.axis("off")
        label_ax.plot([0.08, 0.86], [0.5, 0.5], color=colors[int(unit_idx)], linewidth=2.2, solid_capstyle="round")
        label_ax.text(
            0.9,
            0.5,
            f"u{int(unit_idx):03d}",
            ha="right",
            va="center",
            color=colors[int(unit_idx)],
            fontsize=7,
            fontweight="bold",
        )
        row_maps = [mean_maps[int(cond_idx), int(unit_idx)] for cond_idx in sweep_idx]
        vmin, vmax = image_scale(row_maps, map_vmin_percentile, map_vmax_percentile)
        for col_idx, cond_idx in enumerate(sweep_idx, start=1):
            ax = fig.add_subplot(map_gs[row_idx, col_idx])
            ax.imshow(mean_maps[int(cond_idx), int(unit_idx)], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_edgecolor(colors[int(unit_idx)])
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"BackImage RR100 contour-axis spatial SSI: {plot_mode} sweep\n"
        f"{metric_note}; activation maps use monotonic grayscale per unit row",
        fontsize=11.5,
        y=0.995,
    )
    png_path = out_dir / "backimage_contour_axis_rr100_spatial_ssi_absolute_with_activation_maps.png"
    pdf_path = out_dir / "backimage_contour_axis_rr100_spatial_ssi_absolute_with_activation_maps.pdf"
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def dry_run_inventory(
    trials: pd.DataFrame,
    specs: list[dict[str, Any]],
    axis_column: str,
    *,
    stimulus_rotation_deg: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rotation_deg = int(stimulus_rotation_deg)
    for movie_idx, (_, trial) in enumerate(trials.iterrows()):
        source_trace = np.asarray(trial["source_trace"], dtype=np.float32)
        if rotation_deg:
            source_trace = rotate_trace_xy(source_trace, rotation_deg)
        event_scale_mask = trial_event_scale_mask(trial, int(source_trace.shape[0]))
        event_anchor_mode = trial_event_anchor_mode(trial)
        original_axis_deg = float(trial[str(axis_column)])
        axis_deg = rotated_axis_deg(original_axis_deg, rotation_deg)
        for condition_idx, spec in enumerate(specs):
            if bool(spec.get("trace_bank_condition", False)):
                assignment = trace_bank_condition_assignment(trial, spec)
                trace = np.asarray(assignment["trace"], dtype=np.float32)
                if rotation_deg:
                    trace = rotate_trace_xy(trace, rotation_deg)
                trace_meta = trace_bank_condition_meta(assignment, trace, source_trace)
                trace_meta["original_axis_deg"] = original_axis_deg
                trace_meta["axis_deg"] = axis_deg
                trace_meta["stimulus_rotation_deg"] = rotation_deg
                trace_meta["along_scale"] = float("nan")
                trace_meta["across_scale"] = float("nan")
            elif bool(spec["is_static_baseline"]) and event_scale_mask is None:
                trace_meta = {
                    "source_trace_rms_deg": trace_rms(source_trace),
                    "source_trace_path_length_deg": trace_path_length(source_trace),
                    "along_component_rms_deg": 0.0,
                    "across_component_rms_deg": 0.0,
                    "output_trace_rms_deg": 0.0,
                    "output_trace_path_length_deg": 0.0,
                    "original_axis_deg": original_axis_deg,
                    "axis_deg": axis_deg,
                    "stimulus_rotation_deg": rotation_deg,
                    "along_scale": 0.0,
                    "across_scale": 0.0,
                }
            else:
                _trace, trace_meta = combined_axis_trace(
                    source_trace,
                    axis_deg=axis_deg,
                    along_scale=float(spec["along_scale"]),
                    across_scale=float(spec["across_scale"]),
                    event_scale_mask=event_scale_mask,
                    event_anchor_mode=event_anchor_mode,
                )
                trace_meta["original_axis_deg"] = original_axis_deg
                trace_meta["stimulus_rotation_deg"] = rotation_deg
            rows.append(
                {
                    "movie_index": int(movie_idx),
                    "condition_index": int(condition_idx),
                    "condition_id": str(spec["condition_id"]),
                    "condition_label": str(spec["condition_label"]),
                    "trial_id": int(trial["trial_id"]),
                    "source_row": int(trial["source_row"]),
                    "session": str(trial["session"]),
                    "trial_idx": int(trial["trial_idx"]),
                    "response_cache_path": str(trial["response_cache_path"]),
                    **window_inventory_payload(trial),
                    **trace_meta,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials, source_meta = select_source_trials(args)
    if str(args.sweep_mode) == "trace_bank":
        specs = list(source_meta.get("trace_bank_condition_specs", []))
        if not specs:
            raise ValueError("trace_bank sweep did not produce condition specs.")
    else:
        across_scales = parse_float_list(str(args.across_scales))
        along_scales = parse_float_list(str(args.along_scales)) if str(args.along_scales).strip() else []
        condition_pairs = parse_condition_pairs(str(args.condition_pairs)) if str(args.sweep_mode) == "pairs" else None
        specs = condition_specs(
            across_scales,
            along_scale=float(args.along_scale),
            along_scales=along_scales,
            condition_pairs=condition_pairs,
            include_static_baseline=bool(args.include_static_baseline),
            sweep_mode=str(args.sweep_mode),
            zero_motion_is_static_baseline=not (
                str(args.microsaccade_trace_mode) in EVENT_SCALED_MICROSACCADE_TRACE_MODES
            ),
        )
    run_config = source_meta.get("run_metadata_config", {})
    bin_seconds = float(args.bin_seconds) if args.bin_seconds is not None else float(run_config.get("bin_seconds", 1.0 / 120.0))

    rr100 = load_population_view(version_name=str(args.rr100_version))
    rr100_meta = {
        "version": rr100.name,
        "input_channels": int(rr100.input_channels),
        "n_units": int(rr100.n_units),
        "membership_shape": list(np.asarray(rr100.membership).shape),
        "pooling_mode": str(rr100.meta.get("pooling_mode", "")),
    }
    identity = cache_identity(args, trials, specs, rr100_meta)
    write_json(
        out_dir / "run_metadata.json",
        {
            "identity": identity,
            "source_meta": source_meta,
            "bin_seconds_effective": float(bin_seconds),
            "dry_run": bool(args.dry_run),
        },
    )
    trace_bank_metadata_csv: Path | None = None
    trace_bank_metric_summary_csv: Path | None = None
    trace_bank_path_context_csv: Path | None = None
    trace_bank_assignment_csv: Path | None = None
    trace_bank_diffusion_plot_paths: tuple[Path, Path] | None = None
    trace_bank_microsaccade_plot_paths: tuple[Path, Path] | None = None
    trace_bank_metric_panel_paths: tuple[Path, Path] | None = None
    if "trace_bank_rows" in source_meta:
        trace_bank_metadata_csv = out_dir / "trace_bank_metadata.csv"
        trace_bank_rows = list(source_meta["trace_bank_rows"])
        write_csv_rows(trace_bank_metadata_csv, trace_bank_rows)
        trace_bank_metric_summary_csv = out_dir / "trace_bank_metric_summary.csv"
        write_csv_rows(trace_bank_metric_summary_csv, trace_bank_metric_summary_rows(trace_bank_rows))
        trace_bank_path_context_csv = out_dir / "trace_bank_path_length_context_windows.csv"
        write_csv_rows(trace_bank_path_context_csv, trace_bank_path_length_context_windows(trace_bank_rows))
        trace_bank_metric_panel_paths = plot_trace_bank_metric_summary_panel(
            out_dir,
            trace_bank_rows,
            dpi=int(args.dpi),
        )
        if trace_bank_metric_panel_paths is not None:
            print(f"Wrote trace-bank metric panel: {trace_bank_metric_panel_paths[0]}", flush=True)
            print(f"Wrote trace-bank metric panel: {trace_bank_metric_panel_paths[1]}", flush=True)
        trace_bank_diffusion_plot_paths = plot_trace_bank_diffusion_distribution(
            out_dir,
            trace_bank_rows,
            scale_metric=str(source_meta.get("trace_bank_scale_metric", args.trace_bank_scale_metric)),
            bin_summary=list(source_meta.get("trace_bank_bin_summary", [])),
            dpi=int(args.dpi),
        )
        if trace_bank_diffusion_plot_paths is not None:
            print(f"Wrote trace-bank diffusion plot: {trace_bank_diffusion_plot_paths[0]}", flush=True)
            print(f"Wrote trace-bank diffusion plot: {trace_bank_diffusion_plot_paths[1]}", flush=True)
        trace_bank_microsaccade_plot_paths = plot_trace_bank_microsaccade_diffusion_histogram(
            out_dir,
            trace_bank_rows,
            dpi=int(args.dpi),
        )
        if trace_bank_microsaccade_plot_paths is not None:
            print(
                f"Wrote trace-bank microsaccade diffusion plot: {trace_bank_microsaccade_plot_paths[0]}",
                flush=True,
            )
            print(
                f"Wrote trace-bank microsaccade diffusion plot: {trace_bank_microsaccade_plot_paths[1]}",
                flush=True,
            )
    if "trace_bank_assignment_manifest" in source_meta:
        trace_bank_assignment_csv = out_dir / "trace_bank_assignment_manifest.csv"
        write_csv_rows(trace_bank_assignment_csv, list(source_meta["trace_bank_assignment_manifest"]))

    if bool(args.dry_run):
        rows = dry_run_inventory(
            trials,
            specs,
            str(args.axis_column),
            stimulus_rotation_deg=int(args.stimulus_rotation_deg),
        )
        write_csv_rows(out_dir / "movie_condition_inventory.csv", rows)
        print(f"Dry-run wrote inventory for {len(trials)} trials x {len(specs)} conditions to {out_dir}", flush=True)
        return

    path = cache_path(out_dir)
    stats = None if bool(args.force) else load_cache(path, identity)
    inventory_rows: list[dict[str, Any]] = []
    if stats is None:
        stats, inventory_rows = compute_cache(
            args,
            trials=trials,
            specs=specs,
            population_view=rr100,
            bin_seconds=float(bin_seconds),
        )
        save_cache(path, stats, identity)
        write_csv_rows(out_dir / "movie_condition_inventory.csv", inventory_rows)
        print(f"Saved cache: {path}", flush=True)
    else:
        print(f"Loaded cache: {path}", flush=True)
        inv_path = out_dir / "movie_condition_inventory.csv"
        if inv_path.exists():
            inventory_rows = list(csv.DictReader(inv_path.open(newline="", encoding="utf-8")))

    condition_rows, unit_rows, top_rows, diagnostics = summarize(stats)
    write_csv_rows(out_dir / "condition_summary.csv", condition_rows)
    write_csv_rows(out_dir / "unit_ssi_table.csv", unit_rows)
    sf_rows = sf_compatible_unit_rows(unit_rows)
    write_csv_rows(out_dir / "sf_group_scale_curve_input.csv", sf_rows)
    write_csv_rows(out_dir / "highlighted_units.csv", top_rows[: max(1, int(args.top_units))])
    summary_payload: dict[str, Any] = {
        "primary_ssi_metric": str(diagnostics["primary_ssi_metric"]),
        "condition_summary_csv": out_dir / "condition_summary.csv",
        "unit_ssi_table_csv": out_dir / "unit_ssi_table.csv",
        "sf_group_scale_curve_input_csv": out_dir / "sf_group_scale_curve_input.csv",
        "highlighted_units_csv": out_dir / "highlighted_units.csv",
        "movie_condition_inventory_csv": out_dir / "movie_condition_inventory.csv",
        "cache_npz": path,
        "peak_summary": diagnostics["peak_summary"],
    }
    if trace_bank_metadata_csv is not None:
        summary_payload["trace_bank_metadata_csv"] = trace_bank_metadata_csv
    if trace_bank_metric_summary_csv is not None:
        summary_payload["trace_bank_metric_summary_csv"] = trace_bank_metric_summary_csv
    if trace_bank_path_context_csv is not None:
        summary_payload["trace_bank_path_length_context_windows_csv"] = trace_bank_path_context_csv
    if trace_bank_assignment_csv is not None:
        summary_payload["trace_bank_assignment_manifest_csv"] = trace_bank_assignment_csv
    if trace_bank_metric_panel_paths is not None:
        summary_payload["trace_bank_metric_summary_panel_png"] = trace_bank_metric_panel_paths[0]
        summary_payload["trace_bank_metric_summary_panel_pdf"] = trace_bank_metric_panel_paths[1]
    if trace_bank_diffusion_plot_paths is not None:
        summary_payload["trace_bank_diffusion_distribution_png"] = trace_bank_diffusion_plot_paths[0]
        summary_payload["trace_bank_diffusion_distribution_pdf"] = trace_bank_diffusion_plot_paths[1]
    if trace_bank_microsaccade_plot_paths is not None:
        summary_payload["trace_bank_diffusion_by_microsaccade_png"] = trace_bank_microsaccade_plot_paths[0]
        summary_payload["trace_bank_diffusion_by_microsaccade_pdf"] = trace_bank_microsaccade_plot_paths[1]
    png_path, pdf_path = plot_summary(
        out_dir,
        stats,
        diagnostics,
        top_units=int(args.top_units),
        dpi=int(args.dpi),
        map_vmin_percentile=float(args.map_vmin_percentile),
        map_vmax_percentile=float(args.map_vmax_percentile),
    )
    print(f"Wrote plot: {png_path}", flush=True)
    print(f"Wrote plot: {pdf_path}", flush=True)
    summary_payload["absolute_plot_png"] = png_path
    summary_payload["absolute_plot_pdf"] = pdf_path
    if bool(args.write_zscore_plot):
        from declan.active_sensing_movie_information.plot_backimage_contour_axis_rr100_zscore_curves import (
            plot_zscore_curves,
        )

        zscore_result = plot_zscore_curves(
            out_dir,
            out_dir=out_dir,
            min_unit_std=float(args.zscore_min_unit_std),
            top_units=int(args.top_units),
            dpi=int(args.dpi),
        )
        print(f"Wrote z-score plot: {zscore_result['png']}", flush=True)
        print(f"Wrote z-score plot: {zscore_result['pdf']}", flush=True)
        summary_payload["zscore_plot_png"] = zscore_result["png"]
        summary_payload["zscore_plot_pdf"] = zscore_result["pdf"]
        summary_payload["zscore_curve_csv"] = zscore_result["csv"]
        summary_payload["zscore_retained_units"] = int(zscore_result["retained_units"])
        summary_payload["zscore_total_units"] = int(zscore_result["total_units"])
        summary_payload["zscore_scale_values"] = np.asarray(zscore_result["scale_values"], dtype=float)
        summary_payload["zscore_population_curve"] = np.asarray(zscore_result["population_z"], dtype=float)
        summary_payload["zscore_mean_unit_curve"] = np.asarray(zscore_result["mean_z"], dtype=float)
    write_json(out_dir / "summary.json", summary_payload)
    if diagnostics["peak_summary"]:
        peak = diagnostics["peak_summary"]
        below = peak.get("sweep_peak_is_below_1x")
        below_text = "n/a" if below is None else str(bool(below))
        print(
            "Sweep peak: "
            f"metric={str(peak['primary_ssi_metric'])}, "
            f"across={float(peak['sweep_peak_across_scale']):g}, "
            f"motion_scale={float(peak['sweep_peak_motion_scale']):g}, "
            f"population SSI={float(peak['sweep_peak_population_ssi_bits_per_spike']):.6g}, "
            f"below_1x={below_text}",
            flush=True,
        )


if __name__ == "__main__":
    main()
