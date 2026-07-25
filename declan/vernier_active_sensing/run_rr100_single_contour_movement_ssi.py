#!/usr/bin/env python3
"""Compute RR100 unit SSI for a single contour under different eye movements."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view  # noqa: E402
from declan.vernier_active_sensing.forward import (  # noqa: E402
    PKL_PATH,
    STIMULUS_NORMALIZATION,
    build_vernier_movie,
    load_model_and_readout,
)
from declan.vernier_active_sensing.plot_rr100_endpoint_along0_unit_ssi import (  # noqa: E402
    condition_sequence,
    draw_leave_one_out,
    draw_unit_lines,
    draw_unit_lines_with_activation_rows,
    order_units_by_y_at_x,
    summarize_units,
    unit_ssi_single_frame,
    write_json,
)
from declan.vernier_active_sensing.run_rr100_real_trace_scale_grid import (  # noqa: E402
    DEFAULT_SCALES,
    RR100_VERSION,
    condition_name,
)
from declan.vernier_active_sensing.stimulus import VernierSpec, central_retina_frame  # noqa: E402
from declan.vernier_active_sensing.trajectories import (  # noqa: E402
    DEFAULT_EYE_TRACES_PATH,
    TraceSet,
    condition_trace,
    load_eye_traces,
    subsample_traces,
    valid_trace,
)
from scripts.temporal_decoding.rate_computation import compute_trial_rates  # noqa: E402
from scripts.temporal_decoding.stimulus_hires import N_LAGS as MODEL_HISTORY_FRAMES  # noqa: E402


DEFAULT_OUT_DIR = ROOT / "outputs/notebook_vernier_walkthrough/rr100_single_contour_movement_ssi"
DEFAULT_UNIT_METADATA = (
    ROOT
    / "outputs/active_sensing_movie_information/"
    / "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/unit_feature_table.csv"
)
EPS = 1e-8
CACHE_SCHEMA_VERSION = 2
CACHE_PREFIX = "rr100_single_contour_movement_ssi"
PANEL_C_MOVEMENT_SCALES = "0.25,0.5,0.75,1,1.5,2,3"
PANEL_C_SURFACE_SCALES = PANEL_C_MOVEMENT_SCALES
ANISOTROPIC_CONDITION_RE = re.compile(
    r"^real_aniso_across_([0-9]+(?:[p.][0-9]+)?)_along_([0-9]+(?:[p.][0-9]+)?)$"
)
PANEL_C_SURFACE_FAMILIES = (
    {
        "key": "path",
        "title": "Component path length",
        "across": "across_path_arcmin",
        "along": "along_path_arcmin",
        "description": "sum absolute projected frame-to-frame displacement",
    },
    {
        "key": "rms",
        "title": "Component RMS excursion",
        "across": "across_rms_arcmin",
        "along": "along_rms_arcmin",
        "description": "RMS of centered trace position projected onto each contour-relative axis",
    },
)
PANEL_C_SURFACE_OUTCOMES = (
    (
        "population_ssi_percent_vs_static",
        "SSI bits/spike\n% vs static",
        "SSI efficiency",
        "RdBu_r",
    ),
    (
        "information_bits_per_sample_percent_vs_static",
        "information bits/trace\n% vs static",
        "Information",
        "RdBu_r",
    ),
    (
        "expected_spikes_per_sample_percent_vs_static",
        "expected spikes/trace\n% vs static",
        "Expected spikes",
        "RdBu_r",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--eye-traces-path", type=Path, default=ROOT / DEFAULT_EYE_TRACES_PATH)
    parser.add_argument(
        "--condition-set",
        choices=("along0", "panel_c"),
        default="along0",
        help="along0 keeps the original across-sweep diagnostic; panel_c computes full/across/along conditions.",
    )
    parser.add_argument("--across-scales", type=str, default=DEFAULT_SCALES)
    parser.add_argument("--along-scale", type=float, default=0.0)
    parser.add_argument(
        "--panel-c-scales",
        type=str,
        default=PANEL_C_MOVEMENT_SCALES,
        help="Motion scales for the panel-C full/across/along synthetic-contour plot.",
    )
    parser.add_argument("--n-traces", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, cuda:1, ...")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--contour-width-arcmin", type=float, default=2.0)
    parser.add_argument(
        "--contour-length-arcmin",
        type=float,
        default=28.0,
        help="Total contour length. Internally rendered as two joined zero-gap Vernier halves.",
    )
    parser.add_argument("--contour-contrast", type=float, default=0.5)
    parser.add_argument("--polarity", choices=("bright", "dark"), default="bright")
    parser.add_argument("--orientation-deg", type=float, default=0.0)
    parser.add_argument(
        "--orientation-mode",
        choices=("fixed", "random_per_trace", "random_blocks"),
        default="random_per_trace",
        help=(
            "Use one fixed contour orientation, draw a deterministic random orientation for each selected eye trace, "
            "or draw random orientation blocks with the same selected eye traces repeated inside each block."
        ),
    )
    parser.add_argument("--random-orientation-min-deg", type=float, default=0.0)
    parser.add_argument("--random-orientation-max-deg", type=float, default=180.0)
    parser.add_argument(
        "--n-orientation-blocks",
        type=int,
        default=1,
        help="For --orientation-mode random_blocks, number of random contour orientations to combine.",
    )
    parser.add_argument(
        "--cache-fd-tag-arcmin",
        type=float,
        default=0.0,
        help="Filename compatibility tag for shared postprocessors; not a finite-difference step.",
    )
    parser.add_argument("--top-units", type=int, default=12)
    parser.add_argument("--skip-highlighted-unit-maps", action="store_true")
    parser.add_argument("--map-vmin-percentile", type=float, default=0.5)
    parser.add_argument("--map-vmax-percentile", type=float, default=99.5)
    parser.add_argument("--unit-metadata", type=Path, default=DEFAULT_UNIT_METADATA)
    parser.add_argument("--panel-c-sf-group", default="high_sf")
    parser.add_argument(
        "--panel-c-unit-selection",
        choices=("aligned", "orientation_tuned", "sf_group"),
        default="aligned",
        help=(
            "aligned matches units to a fixed synthetic contour axis; orientation_tuned uses orientation-selective units "
            "within --panel-c-sf-group; sf_group uses every unit in --panel-c-sf-group."
        ),
    )
    parser.add_argument("--panel-c-min-osi", type=float, default=0.05)
    parser.add_argument("--panel-c-match-max-deg", type=float, default=22.5)
    parser.add_argument(
        "--panel-c-contour-axis-deg",
        type=float,
        default=None,
        help="Contour-axis orientation for unit matching. Defaults to vertical axis plus --orientation-deg.",
    )
    parser.add_argument("--panel-c-bootstrap", type=int, default=10000)
    parser.add_argument(
        "--panel-c-output-stem",
        default=None,
        help="Optional output filename stem for panel-C summary/figure files.",
    )
    parser.add_argument(
        "--panel-c-write-orientation-splits",
        action="store_true",
        help=(
            "Also write a three-panel high-SF orientation-relation split using per-trace contour axes: "
            "aligned, oblique, and across-tuned."
        ),
    )
    parser.add_argument(
        "--panel-c-write-component-surfaces",
        action="store_true",
        help=(
            "Also write 2D across x along component surfaces for the per-trace high-SF orientation-relation pools."
        ),
    )
    parser.add_argument(
        "--panel-c-write-arcmin-binned-surfaces",
        action="store_true",
        help=(
            "Also pool generated moving trace-condition samples into measured arcmin bins, matching the "
            "BackImage component-bin surface style."
        ),
    )
    parser.add_argument(
        "--panel-c-surface-scales",
        type=str,
        default=PANEL_C_SURFACE_SCALES,
        help=(
            "Comma-separated component scales for the synthetic surface grid. "
            "Rows vary across-contour scale; columns vary along-contour scale."
        ),
    )
    parser.add_argument(
        "--panel-c-arcmin-surface-bins",
        type=int,
        default=8,
        help="Number of marginal measured-arcmin quantile bins for --panel-c-write-arcmin-binned-surfaces.",
    )
    parser.add_argument(
        "--panel-c-orthogonal-min-deg",
        type=float,
        default=67.5,
        help="Minimum axial distance from the contour axis for the across-tuned split panel.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--preview-only", action="store_true", help="Draw the base single-contour stimulus preview and exit.")
    parser.add_argument("--force", action="store_true", help="Recompute cached contour unit SSI stats.")
    return parser.parse_args()


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
        return [json_ready(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def parse_scale_list(text: str) -> list[float]:
    scales = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not scales:
        raise ValueError("At least one across-contour scale is required.")
    return scales


def canonical_single_contour_spec(args: argparse.Namespace) -> VernierSpec:
    """Represent one contour by joining the two Vernier halves with zero gap."""
    return VernierSpec(
        offset_arcmin=0.0,
        bar_width_arcmin=float(args.contour_width_arcmin),
        gap_arcmin=0.0,
        bar_length_arcmin=0.5 * float(args.contour_length_arcmin),
        contrast=float(args.contour_contrast),
        polarity=str(args.polarity),
        orientation_deg=float(args.orientation_deg),
    )


def validate_orientation_args(args: argparse.Namespace) -> None:
    if int(getattr(args, "n_orientation_blocks", 1)) < 1:
        raise ValueError(f"--n-orientation-blocks must be >= 1, got {args.n_orientation_blocks}.")
    if str(args.orientation_mode) not in {"random_per_trace", "random_blocks"}:
        return
    lo = float(args.random_orientation_min_deg)
    hi = float(args.random_orientation_max_deg)
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError(
            "Random orientation mode requires finite --random-orientation-min-deg "
            f"< --random-orientation-max-deg, got {lo:g} and {hi:g}."
        )


def orientation_block_count(args: argparse.Namespace) -> int:
    return int(args.n_orientation_blocks) if str(args.orientation_mode) == "random_blocks" else 1


def expected_trace_count(args: argparse.Namespace) -> int:
    return int(args.n_traces) * orientation_block_count(args)


def expand_trace_set_for_orientation_blocks(args: argparse.Namespace, trace_set: TraceSet) -> TraceSet:
    n_blocks = orientation_block_count(args)
    if n_blocks <= 1:
        return trace_set
    traces = np.asarray(trace_set.traces)
    durations = np.asarray(trace_set.durations)
    return TraceSet(
        traces=np.concatenate([traces.copy() for _ in range(n_blocks)], axis=0).astype(np.float32, copy=False),
        durations=np.concatenate([durations.copy() for _ in range(n_blocks)], axis=0).astype(np.int32, copy=False),
    )


def trace_orientation_deg(args: argparse.Namespace, trace_idx: int) -> float:
    if str(args.orientation_mode) == "fixed":
        return float(args.orientation_deg)
    if str(args.orientation_mode) == "random_blocks":
        block_idx = int(trace_idx) // max(int(args.n_traces), 1)
        rng = np.random.default_rng(int(args.seed) + 7919 * int(block_idx) + 104729)
    else:
        rng = np.random.default_rng(int(args.seed) + 7919 * int(trace_idx) + 104729)
    return float(rng.uniform(float(args.random_orientation_min_deg), float(args.random_orientation_max_deg)))


def stimulus_orientations_for_traces(args: argparse.Namespace, n_traces: int) -> np.ndarray:
    return np.asarray([trace_orientation_deg(args, trace_idx) for trace_idx in range(int(n_traces))], dtype=np.float32)


def contour_axis_from_orientation_deg(orientation_deg: np.ndarray | float) -> np.ndarray:
    return axis_180_deg(90.0 + np.asarray(orientation_deg, dtype=np.float64))


def single_contour_spec_for_trace(args: argparse.Namespace, trace_idx: int) -> VernierSpec:
    return replace(canonical_single_contour_spec(args), orientation_deg=trace_orientation_deg(args, trace_idx))


def _parse_condition_scale_token(text: str) -> float:
    return float(str(text).replace("p", "."))


def anisotropic_condition_scales(condition: str) -> tuple[float, float] | None:
    match = ANISOTROPIC_CONDITION_RE.match(str(condition))
    if match is None:
        return None
    return _parse_condition_scale_token(match.group(1)), _parse_condition_scale_token(match.group(2))


def contour_basis(orientation_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return unit vectors for across- and along-contour axes in eye-position coordinates."""
    theta = np.deg2rad(float(orientation_deg))
    across = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float32)
    along = np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.float32)
    return across, along


def contour_relative_anisotropic_trace(
    trace: np.ndarray,
    *,
    across_scale: float,
    along_scale: float,
    orientation_deg: float,
) -> np.ndarray:
    eye = np.asarray(trace, dtype=np.float32)
    mean = np.mean(eye, axis=0, keepdims=True).astype(np.float32)
    centered = eye - mean
    across_axis, along_axis = contour_basis(float(orientation_deg))
    across_component = centered @ across_axis
    along_component = centered @ along_axis
    scaled = (
        mean
        + float(across_scale) * across_component[:, None] * across_axis[None, :]
        + float(along_scale) * along_component[:, None] * along_axis[None, :]
    )
    return scaled.astype(np.float32)


def oriented_condition_trace(
    base_trace: np.ndarray,
    *,
    condition: str,
    trace_set: Any,
    rng: np.random.Generator,
    orientation_deg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    scales = anisotropic_condition_scales(condition)
    if scales is None:
        return condition_trace(base_trace, condition=condition, trace_set=trace_set, rng=rng)
    across_scale, along_scale = scales
    trace = contour_relative_anisotropic_trace(
        base_trace,
        across_scale=float(across_scale),
        along_scale=float(along_scale),
        orientation_deg=float(orientation_deg),
    )
    return trace, {
        "condition_family": "anisotropic_scaled",
        "across_scale": float(across_scale),
        "along_scale": float(along_scale),
        "axis_convention": "contour_relative_axes_rotated_with_stimulus",
        "stimulus_orientation_deg": float(orientation_deg),
        "contour_axis_deg": float(contour_axis_from_orientation_deg(float(orientation_deg))),
    }


def stats_cache_path(out_dir: Path, condition: str, cache_fd_tag_arcmin: float, max_frames: int) -> Path:
    return (
        Path(out_dir)
        / "cache"
        / f"{CACHE_PREFIX}_{condition}_frames{int(max_frames)}_fd{float(cache_fd_tag_arcmin):.4f}arcmin.npz"
    )


def _cache_identity(args: argparse.Namespace, *, condition: str) -> dict[str, Any]:
    spec = canonical_single_contour_spec(args)
    condition_set = str(getattr(args, "condition_set", "along0"))
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "analysis": "rr100_single_contour_movement_ssi",
        "condition_set": condition_set,
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "condition": str(condition),
        "eye_traces_path": str(Path(args.eye_traces_path).expanduser().resolve()),
        "across_scales": parse_scale_list(args.across_scales),
        "along_scale": float(args.along_scale),
        "panel_c_scales": parse_scale_list(args.panel_c_scales) if condition_set == "panel_c" else [],
        "n_traces": int(args.n_traces),
        "max_frames": int(args.max_frames),
        "model_history_frames": int(MODEL_HISTORY_FRAMES),
        "seed": int(args.seed),
        "stimulus_orientation_mode": str(args.orientation_mode),
        "fixed_orientation_deg": float(args.orientation_deg),
        "random_orientation_range_deg": [
            float(args.random_orientation_min_deg),
            float(args.random_orientation_max_deg),
        ]
        if str(args.orientation_mode) in {"random_per_trace", "random_blocks"}
        else [],
        "n_orientation_blocks": orientation_block_count(args),
        "n_effective_traces": expected_trace_count(args),
        "motion_axis_convention": "contour_relative_axes_rotated_with_stimulus",
        "population_version": RR100_VERSION,
        "readout_source_pkl": str(PKL_PATH.expanduser().resolve()),
        "single_contour_spec": asdict(spec),
        "stimulus_contract": (
            "single contour represented by zero-offset zero-gap Vernier halves; "
            "orientation is fixed, deterministic-random per selected trace, or deterministic-random by repeated trace blocks"
        ),
        "trajectory_contract": "native real-trace positions, anisotropically scaled around trial mean",
        "readout_time_contract": "all response frames averaged for SSI",
        "ssi_contract": "spatial SSI of fixed-contour response maps; no plus/minus acuity task",
    }


def _identity_text(identity: dict[str, Any]) -> str:
    return json.dumps(json_ready(identity), sort_keys=True, separators=(",", ":"))


def cache_has_required_fields(path: Path, *, require_maps: bool, expected_identity: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    required = {"cache_identity_json", "pose_traces"}
    if require_maps:
        required.add("mean_rate_map")
    try:
        with np.load(path) as data:
            if not required.issubset(set(data.files)):
                return False
            return str(np.asarray(data["cache_identity_json"]).ravel()[0]) == _identity_text(expected_identity)
    except Exception:
        return False


def _rr100_spatial_movie(
    model: Any,
    readout: Any,
    view: Any,
    spec: VernierSpec,
    trace: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    stim = build_vernier_movie(spec, trace, n_lags=int(MODEL_HISTORY_FRAMES), device=device)
    full_spatial = compute_trial_rates(
        model,
        readout,
        stim,
        batch_size=int(batch_size),
        return_spatial=True,
    ).astype(np.float32)
    rr100 = apply_population_view(full_spatial, view).astype(np.float32)
    del stim, full_spatial
    return rr100


def _unit_ssi_movie(rate_movie: np.ndarray) -> dict[str, np.ndarray | float]:
    bits: list[np.ndarray] = []
    rates: list[np.ndarray] = []
    population: list[float] = []
    for frame in np.asarray(rate_movie, dtype=np.float32):
        ssi = unit_ssi_single_frame(frame, eps=EPS)
        bits.append(np.asarray(ssi["unit_bits_per_spike"], dtype=np.float32))
        rates.append(np.asarray(ssi["unit_mean_rate"], dtype=np.float32))
        population.append(float(ssi["population_bits_per_spike"]))
    return {
        "unit_bits_per_spike": np.nanmean(np.asarray(bits, dtype=np.float32), axis=0).astype(np.float32),
        "unit_mean_rate": np.nanmean(np.asarray(rates, dtype=np.float32), axis=0).astype(np.float32),
        "population_bits_per_spike": float(np.nanmean(np.asarray(population, dtype=np.float32))),
    }


def compute_condition_stats(
    args: argparse.Namespace,
    *,
    condition: str,
    trace_set: Any,
    model: Any,
    readout: Any,
    view: Any,
    device: str,
) -> dict[str, Any]:
    unit_bits: list[np.ndarray] = []
    unit_rates: list[np.ndarray] = []
    population_bits: list[float] = []
    pose_traces: list[np.ndarray] = []
    stimulus_orientations: list[float] = []
    contour_axes: list[float] = []
    map_sum: np.ndarray | None = None
    map_sq_sum: np.ndarray | None = None
    map_count = 0

    for trace_idx in range(trace_set.traces.shape[0]):
        base_trace = valid_trace(trace_set, trace_idx, max_frames=int(args.max_frames))
        rng = np.random.default_rng(int(args.seed) + 1009 * trace_idx)
        spec = single_contour_spec_for_trace(args, trace_idx)
        effective_trace, _trace_meta = oriented_condition_trace(
            base_trace,
            condition=condition,
            trace_set=trace_set,
            rng=rng,
            orientation_deg=float(spec.orientation_deg),
        )
        trace = np.asarray(effective_trace[: int(args.max_frames)], dtype=np.float32)
        if trace.shape[0] != int(args.max_frames):
            raise RuntimeError(f"Expected {args.max_frames} frames for {condition}, got {trace.shape[0]}")

        rate_movie = _rr100_spatial_movie(
            model,
            readout,
            view,
            spec,
            trace,
            device=device,
            batch_size=int(args.batch_size),
        )
        ssi = _unit_ssi_movie(rate_movie)
        if map_sum is None:
            map_sum = np.zeros_like(rate_movie[0], dtype=np.float64)
            map_sq_sum = np.zeros_like(rate_movie[0], dtype=np.float64)
        map_sum += np.sum(rate_movie, axis=0, dtype=np.float64)
        map_sq_sum += np.sum(np.square(rate_movie, dtype=np.float64), axis=0)
        map_count += int(rate_movie.shape[0])
        unit_bits.append(np.asarray(ssi["unit_bits_per_spike"], dtype=np.float32))
        unit_rates.append(np.asarray(ssi["unit_mean_rate"], dtype=np.float32))
        population_bits.append(float(ssi["population_bits_per_spike"]))
        pose_traces.append(trace.astype(np.float32, copy=True))
        stimulus_orientations.append(float(spec.orientation_deg))
        contour_axes.append(float(contour_axis_from_orientation_deg(float(spec.orientation_deg))))
        print(
            f"  {condition} trace {trace_idx}: pop SSI={population_bits[-1]:.6g}; "
            f"T={rate_movie.shape[0]}; orientation={float(spec.orientation_deg):.2f} deg",
            flush=True,
        )
        del rate_movie
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if map_sum is None or map_sq_sum is None or map_count <= 0:
        raise RuntimeError(f"No maps were computed for {condition}.")
    mean_map = (map_sum / float(map_count)).astype(np.float32)
    second_moment = map_sq_sum / float(map_count)
    std_map = np.sqrt(np.maximum(second_moment - np.square(mean_map, dtype=np.float64), 0.0)).astype(np.float32)
    return {
        "unit_bits_per_trace": np.asarray(unit_bits, dtype=np.float32),
        "unit_mean_rate_per_trace": np.asarray(unit_rates, dtype=np.float32),
        "population_bits_per_trace": np.asarray(population_bits, dtype=np.float32),
        "pose_traces": np.asarray(pose_traces, dtype=np.float32),
        "stimulus_orientation_deg_per_trace": np.asarray(stimulus_orientations, dtype=np.float32),
        "contour_axis_deg_per_trace": np.asarray(contour_axes, dtype=np.float32),
        "mean_rate_map": mean_map,
        "std_rate_map": std_map,
    }


def load_or_compute_condition_stats(
    args: argparse.Namespace,
    *,
    condition: str,
    trace_set: Any | None,
    model: Any | None,
    readout: Any | None,
    view: Any | None,
    device: str | None,
    require_maps: bool,
) -> dict[str, Any]:
    path = stats_cache_path(Path(args.out_dir), condition, float(args.cache_fd_tag_arcmin), int(args.max_frames))
    expected_identity = _cache_identity(args, condition=condition)
    if path.exists() and not bool(args.force):
        with np.load(path) as data:
            has_required_maps = (not require_maps) or "mean_rate_map" in data
            has_matching_identity = (
                "cache_identity_json" in data
                and str(np.asarray(data["cache_identity_json"]).ravel()[0]) == _identity_text(expected_identity)
            )
            if has_required_maps and has_matching_identity:
                print(f"Loaded single-contour unit SSI cache: {path}", flush=True)
                out = {
                    "unit_bits_per_trace": np.asarray(data["unit_bits_per_trace"], dtype=np.float32),
                    "unit_mean_rate_per_trace": np.asarray(data["unit_mean_rate_per_trace"], dtype=np.float32),
                    "population_bits_per_trace": np.asarray(data["population_bits_per_trace"], dtype=np.float32),
                    "pose_traces": np.asarray(data["pose_traces"], dtype=np.float32),
                }
                if "stimulus_orientation_deg_per_trace" in data:
                    out["stimulus_orientation_deg_per_trace"] = np.asarray(
                        data["stimulus_orientation_deg_per_trace"], dtype=np.float32
                    )
                if "contour_axis_deg_per_trace" in data:
                    out["contour_axis_deg_per_trace"] = np.asarray(data["contour_axis_deg_per_trace"], dtype=np.float32)
                if "mean_rate_map" in data:
                    out["mean_rate_map"] = np.asarray(data["mean_rate_map"], dtype=np.float32)
                if "std_rate_map" in data:
                    out["std_rate_map"] = np.asarray(data["std_rate_map"], dtype=np.float32)
                return out
            print(f"Single-contour cache metadata mismatch or missing maps; recomputing: {path}", flush=True)
    if trace_set is None or model is None or readout is None or view is None or device is None:
        raise RuntimeError("Model/readout/trace resources are required to compute missing caches.")
    print(f"Computing single-contour unit SSI cache: {condition}", flush=True)
    stats = compute_condition_stats(
        args,
        condition=condition,
        trace_set=trace_set,
        model=model,
        readout=readout,
        view=view,
        device=device,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        unit_bits_per_trace=stats["unit_bits_per_trace"],
        unit_mean_rate_per_trace=stats["unit_mean_rate_per_trace"],
        population_bits_per_trace=stats["population_bits_per_trace"],
        pose_traces=stats["pose_traces"],
        stimulus_orientation_deg_per_trace=stats["stimulus_orientation_deg_per_trace"],
        contour_axis_deg_per_trace=stats["contour_axis_deg_per_trace"],
        mean_rate_map=stats["mean_rate_map"],
        std_rate_map=stats["std_rate_map"],
        condition=np.asarray([condition]),
        max_frames=np.asarray([int(args.max_frames)], dtype=np.int32),
        cache_fd_tag_arcmin=np.asarray([float(args.cache_fd_tag_arcmin)], dtype=np.float32),
        trajectory_contract=np.asarray(["native_real_trace_scaled_around_trial_mean_in_contour_relative_axes"]),
        readout_time_contract=np.asarray(["all_response_frames_mean"]),
        ssi_contract=np.asarray(["spatial_ssi_fixed_contour_no_acuity_task"]),
        cache_identity_json=np.asarray([_identity_text(expected_identity)]),
        stimulus_normalization=np.asarray([STIMULUS_NORMALIZATION]),
        stimulus_orientation_mode=np.asarray([str(args.orientation_mode)]),
        motion_axis_convention=np.asarray(["contour_relative_axes_rotated_with_stimulus"]),
    )
    print(f"Saved single-contour unit SSI cache: {path}", flush=True)
    return stats


def draw_contour_preview(spec: VernierSpec, path: Path, dpi: int) -> None:
    image = central_retina_frame(spec, device="cpu")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(2.6, 2.6), dpi=int(dpi), constrained_layout=True)
    ax.imshow(image, origin="lower", cmap="gray", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("single contour stimulus", fontsize=9.0)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def axis_180_deg(values: np.ndarray | float) -> np.ndarray:
    return np.mod(np.asarray(values, dtype=np.float64), 180.0)


def axis_delta_deg(a_deg: np.ndarray | float, b_deg: np.ndarray | float) -> np.ndarray:
    a = np.asarray(a_deg, dtype=np.float64)
    b = np.asarray(b_deg, dtype=np.float64)
    return np.abs(0.5 * np.degrees(np.angle(np.exp(2j * np.radians(a - b)))))


def panel_c_contour_axis_deg(args: argparse.Namespace) -> float:
    if args.panel_c_contour_axis_deg is not None:
        return float(axis_180_deg(float(args.panel_c_contour_axis_deg)))
    # VernierSpec orientation rotates a vertical bar; the contour axis is vertical at orientation_deg=0.
    return float(axis_180_deg(90.0 + float(args.orientation_deg)))


def has_random_trace_orientations(args: argparse.Namespace) -> bool:
    return str(args.orientation_mode) in {"random_per_trace", "random_blocks"}


def effective_panel_c_unit_selection(args: argparse.Namespace) -> str:
    selection = str(args.panel_c_unit_selection)
    if selection == "aligned" and has_random_trace_orientations(args):
        return "orientation_tuned"
    return selection


def panel_c_contour_axis_label(args: argparse.Namespace) -> str:
    if not has_random_trace_orientations(args):
        return f"{panel_c_contour_axis_deg(args):.1f} deg"
    lo = float(axis_180_deg(90.0 + float(args.random_orientation_min_deg)))
    hi = float(axis_180_deg(90.0 + float(args.random_orientation_max_deg)))
    prefix = (
        f"{orientation_block_count(args)} random axes x {int(args.n_traces)} eye traces"
        if str(args.orientation_mode) == "random_blocks"
        else "random axes"
    )
    if np.isclose(abs(float(args.random_orientation_max_deg) - float(args.random_orientation_min_deg)), 180.0):
        return f"{prefix}, uniform over 180 deg"
    return f"{prefix} from {lo:.1f} to {hi:.1f} deg"


def panel_c_condition_rows(scales: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "condition": "static_center",
            "movement_family": "static",
            "movement_label": "static",
            "movement_scale": 0.0,
            "across_scale": 0.0,
            "along_scale": 0.0,
            "is_static_baseline": True,
            "plot_order": 0,
        }
    ]
    specs = [
        ("across", "across contour", lambda scale: (scale, 0.0), 1),
        ("along", "along contour", lambda scale: (0.0, scale), 2),
    ]
    for family, label, scale_fn, order in specs:
        for scale in scales:
            if float(scale) <= 0.0:
                continue
            across, along = scale_fn(float(scale))
            rows.append(
                {
                    "condition": condition_name(float(across), float(along)),
                    "movement_family": family,
                    "movement_label": label,
                    "movement_scale": float(scale),
                    "across_scale": float(across),
                    "along_scale": float(along),
                    "is_static_baseline": False,
                    "plot_order": int(order),
                }
            )
    return rows


def panel_c_sf_label(sf_group: str) -> str:
    return {
        "low_sf": "low-SF",
        "middle_sf": "middle-SF",
        "high_sf": "high-SF",
    }.get(str(sf_group), str(sf_group).replace("_", "-"))


def panel_c_output_stem(args: argparse.Namespace) -> str:
    if args.panel_c_output_stem:
        return str(args.panel_c_output_stem)
    sf_group = str(args.panel_c_sf_group)
    selection = effective_panel_c_unit_selection(args)
    if str(args.orientation_mode) == "random_blocks":
        orientation_tag = f"_random_ori_{int(args.n_orientation_blocks)}blocks"
    else:
        orientation_tag = "_random_ori" if has_random_trace_orientations(args) else ""
    if selection == "sf_group":
        return f"rr100_single_contour_panel_c_all_{sf_group}{orientation_tag}"
    if selection == "orientation_tuned":
        return f"rr100_single_contour_panel_c_orientation_tuned_{sf_group}{orientation_tag}"
    return f"rr100_single_contour_panel_c_aligned_{sf_group}{orientation_tag}"


def panel_c_selection_title(args: argparse.Namespace) -> str:
    label = panel_c_sf_label(str(args.panel_c_sf_group))
    selection = effective_panel_c_unit_selection(args)
    if selection == "sf_group":
        return f"all {label} units"
    if selection == "orientation_tuned":
        return f"orientation-tuned {label} units"
    return f"aligned {label} units"


def load_panel_c_units_table(args: argparse.Namespace, n_units: int) -> pd.DataFrame:
    units = pd.read_csv(Path(args.unit_metadata))
    required = {
        "unit_index",
        "unit_label",
        "sf_group",
        "prior_preferred_orientation_deg",
        "prior_orientation_selectivity_index",
    }
    missing = sorted(required.difference(units.columns))
    if missing:
        raise ValueError(f"{args.unit_metadata} is missing columns: {missing}")
    units = units.copy()
    units["unit_index"] = pd.to_numeric(units["unit_index"], errors="coerce").astype(int)
    units = units[(units["unit_index"] >= 0) & (units["unit_index"] < int(n_units))].copy()
    units["prior_preferred_orientation_deg"] = pd.to_numeric(
        units["prior_preferred_orientation_deg"], errors="coerce"
    )
    units["prior_orientation_selectivity_index"] = pd.to_numeric(
        units["prior_orientation_selectivity_index"], errors="coerce"
    )
    return units


def selected_panel_c_units(args: argparse.Namespace, n_units: int) -> tuple[np.ndarray, pd.DataFrame, float]:
    units = load_panel_c_units_table(args, n_units)
    contour_axis = panel_c_contour_axis_deg(args)
    selection = effective_panel_c_unit_selection(args)
    if has_random_trace_orientations(args):
        units["synthetic_contour_axis_deg"] = np.nan
        units["orientation_delta_from_synthetic_contour_deg"] = np.nan
    else:
        units["synthetic_contour_axis_deg"] = contour_axis
        units["orientation_delta_from_synthetic_contour_deg"] = axis_delta_deg(
            units["prior_preferred_orientation_deg"].to_numpy(dtype=float),
            contour_axis,
        )
    keep = units["sf_group"].astype(str).eq(str(args.panel_c_sf_group))
    if selection == "orientation_tuned":
        keep = (
            keep
            & units["prior_preferred_orientation_deg"].notna()
            & units["prior_orientation_selectivity_index"].ge(float(args.panel_c_min_osi))
        )
    elif selection == "aligned":
        keep = (
            keep
            & units["prior_preferred_orientation_deg"].notna()
            & units["prior_orientation_selectivity_index"].ge(float(args.panel_c_min_osi))
            & units["orientation_delta_from_synthetic_contour_deg"].le(float(args.panel_c_match_max_deg))
        )
    selected = units.loc[keep].sort_values("unit_index").reset_index(drop=True)
    if selected.empty:
        raise ValueError(
            "No units matched the synthetic panel-C selection: "
            f"unit_selection={selection}, sf_group={args.panel_c_sf_group}, contour_axis={contour_axis:g}, "
            f"min_osi={args.panel_c_min_osi:g}, match_max_deg={args.panel_c_match_max_deg:g}."
        )
    return selected["unit_index"].to_numpy(dtype=int), selected, contour_axis


def trace_weighted_parts(stats: dict[str, Any], unit_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bits = np.asarray(stats["unit_bits_per_trace"], dtype=np.float64)[:, unit_indices]
    rates = np.asarray(stats["unit_mean_rate_per_trace"], dtype=np.float64)[:, unit_indices]
    valid = np.isfinite(bits) & np.isfinite(rates) & (rates >= 0.0)
    numer = np.sum(np.where(valid, bits * rates, 0.0), axis=1)
    denom = np.sum(np.where(valid, rates, 0.0), axis=1)
    return numer.astype(np.float64), denom.astype(np.float64)


def panel_c_orientation_relation_specs(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    match_max = float(args.panel_c_match_max_deg)
    orthogonal_min = float(args.panel_c_orthogonal_min_deg)
    if not (0.0 <= match_max < orthogonal_min <= 90.0):
        raise ValueError(
            "Orientation split thresholds must satisfy "
            f"0 <= match_max < orthogonal_min <= 90, got {match_max:g} and {orthogonal_min:g}."
        )
    return (
        {
            "relation": "contour_matched",
            "relation_label": "contour-aligned",
            "relation_title": "Contour-aligned high-SF units",
            "relation_rank": 0,
        },
        {
            "relation": "contour_intermediate",
            "relation_label": "oblique-to-contour",
            "relation_title": "Oblique high-SF units",
            "relation_rank": 1,
        },
        {
            "relation": "contour_orthogonal",
            "relation_label": "contour-orthogonal",
            "relation_title": "Contour-orthogonal high-SF units",
            "relation_rank": 2,
        },
    )


def panel_c_orientation_split_output_stem(args: argparse.Namespace, output_stem: str) -> str:
    if args.panel_c_output_stem:
        return f"{output_stem}_orientation_relation_split"
    if str(args.orientation_mode) == "random_blocks":
        orientation_tag = f"_random_ori_{int(args.n_orientation_blocks)}blocks"
    else:
        orientation_tag = "_random_ori" if has_random_trace_orientations(args) else ""
    return f"rr100_single_contour_panel_c_{args.panel_c_sf_group}_orientation_relation_split{orientation_tag}"


def stats_trace_contour_axes(stats: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    bits = np.asarray(stats["unit_bits_per_trace"])
    n_traces = int(bits.shape[0])
    if "contour_axis_deg_per_trace" in stats:
        axes = np.asarray(stats["contour_axis_deg_per_trace"], dtype=np.float64)
    else:
        axes = contour_axis_from_orientation_deg(stimulus_orientations_for_traces(args, n_traces))
    if axes.size != n_traces:
        raise ValueError(f"Expected {n_traces} contour axes, got {axes.size}.")
    return axis_180_deg(axes)


def orientation_relation_masks_for_traces(
    units: pd.DataFrame,
    contour_axes_deg: np.ndarray,
    *,
    args: argparse.Namespace,
) -> dict[str, list[np.ndarray]]:
    specs = panel_c_orientation_relation_specs(args)
    sf_units = units[units["sf_group"].astype(str).eq(str(args.panel_c_sf_group))].copy()
    pref = sf_units["prior_preferred_orientation_deg"].to_numpy(dtype=np.float64)
    osi = sf_units["prior_orientation_selectivity_index"].to_numpy(dtype=np.float64)
    unit_indices = sf_units["unit_index"].to_numpy(dtype=int)
    orientation_tuned = (
        np.isfinite(pref)
        & np.isfinite(osi)
        & (osi >= float(args.panel_c_min_osi))
    )
    masks: dict[str, list[np.ndarray]] = {str(spec["relation"]): [] for spec in specs}
    for contour_axis in np.asarray(contour_axes_deg, dtype=np.float64):
        delta = axis_delta_deg(pref, float(contour_axis))
        finite = orientation_tuned & np.isfinite(delta)
        relation_masks = {
            "contour_matched": finite & (delta <= float(args.panel_c_match_max_deg)),
            "contour_intermediate": finite
            & (delta > float(args.panel_c_match_max_deg))
            & (delta < float(args.panel_c_orthogonal_min_deg)),
            "contour_orthogonal": finite & (delta >= float(args.panel_c_orthogonal_min_deg)),
        }
        for spec in specs:
            relation = str(spec["relation"])
            masks[relation].append(np.asarray(unit_indices[relation_masks[relation]], dtype=int))
    return masks


def trace_weighted_parts_for_trace_masks(
    stats: dict[str, Any],
    trace_masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bits = np.asarray(stats["unit_bits_per_trace"], dtype=np.float64)
    rates = np.asarray(stats["unit_mean_rate_per_trace"], dtype=np.float64)
    if bits.shape != rates.shape:
        raise ValueError(f"unit_bits_per_trace shape {bits.shape} does not match rate shape {rates.shape}.")
    if len(trace_masks) != int(bits.shape[0]):
        raise ValueError(f"Expected {bits.shape[0]} trace masks, got {len(trace_masks)}.")
    numer = np.zeros(int(bits.shape[0]), dtype=np.float64)
    denom = np.zeros(int(bits.shape[0]), dtype=np.float64)
    counts = np.zeros(int(bits.shape[0]), dtype=np.int64)
    for trace_idx, unit_indices in enumerate(trace_masks):
        idx = np.asarray(unit_indices, dtype=int)
        counts[trace_idx] = int(idx.size)
        if idx.size == 0:
            continue
        trace_bits = bits[trace_idx, idx]
        trace_rates = rates[trace_idx, idx]
        valid = np.isfinite(trace_bits) & np.isfinite(trace_rates) & (trace_rates >= 0.0)
        numer[trace_idx] = float(np.sum(np.where(valid, trace_bits * trace_rates, 0.0)))
        denom[trace_idx] = float(np.sum(np.where(valid, trace_rates, 0.0)))
    return numer, denom, counts


def orientation_split_selection_table(
    units: pd.DataFrame,
    masks_by_relation: dict[str, list[np.ndarray]],
    contour_axes_deg: np.ndarray,
    *,
    args: argparse.Namespace,
) -> pd.DataFrame:
    specs = panel_c_orientation_relation_specs(args)
    unit_labels = units.set_index("unit_index")["unit_label"].astype(str).to_dict()
    pref = units.set_index("unit_index")["prior_preferred_orientation_deg"].astype(float).to_dict()
    rows: list[dict[str, Any]] = []
    for spec in specs:
        relation = str(spec["relation"])
        for trace_idx, unit_indices in enumerate(masks_by_relation[relation]):
            idx = np.asarray(unit_indices, dtype=int)
            deltas = [
                float(axis_delta_deg(float(pref[int(unit_idx)]), float(contour_axes_deg[trace_idx])))
                for unit_idx in idx
                if int(unit_idx) in pref and np.isfinite(float(pref[int(unit_idx)]))
            ]
            rows.append(
                {
                    "trace_index": int(trace_idx),
                    "contour_axis_deg": float(contour_axes_deg[trace_idx]),
                    "relation": relation,
                    "relation_label": str(spec["relation_label"]),
                    "relation_rank": int(spec["relation_rank"]),
                    "sf_group": str(args.panel_c_sf_group),
                    "n_units": int(idx.size),
                    "orientation_delta_mean_deg": float(np.nanmean(deltas)) if deltas else float("nan"),
                    "orientation_delta_min_deg": float(np.nanmin(deltas)) if deltas else float("nan"),
                    "orientation_delta_max_deg": float(np.nanmax(deltas)) if deltas else float("nan"),
                    "unit_indices": " ".join(str(int(unit_idx)) for unit_idx in idx),
                    "unit_labels": " ".join(str(unit_labels.get(int(unit_idx), f"u{int(unit_idx):03d}")) for unit_idx in idx),
                }
            )
    return pd.DataFrame(rows)


def summarize_panel_c_orientation_splits(
    *,
    rows: list[dict[str, Any]],
    stats_by_condition: dict[str, dict[str, Any]],
    units: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = panel_c_orientation_relation_specs(args)
    base_stats = stats_by_condition["static_center"]
    base_axes = stats_trace_contour_axes(base_stats, args)
    base_masks_by_relation = orientation_relation_masks_for_traces(units, base_axes, args=args)
    selection = orientation_split_selection_table(
        units,
        base_masks_by_relation,
        base_axes,
        args=args,
    )
    base_parts_by_relation: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for spec in specs:
        relation = str(spec["relation"])
        base_num, base_den, base_counts = trace_weighted_parts_for_trace_masks(
            base_stats,
            base_masks_by_relation[relation],
        )
        base_parts_by_relation[relation] = (base_num, base_den, base_counts, ratio_from_parts(base_num, base_den))

    out_rows: list[dict[str, Any]] = []
    for relation_idx, spec in enumerate(specs):
        relation = str(spec["relation"])
        base_num, base_den, base_counts, base_point = base_parts_by_relation[relation]
        for row_idx, row in enumerate(rows):
            condition = str(row["condition"])
            stats = stats_by_condition[condition]
            axes = stats_trace_contour_axes(stats, args)
            masks = (
                base_masks_by_relation[relation]
                if condition == "static_center" and np.array_equal(axes, base_axes)
                else orientation_relation_masks_for_traces(units, axes, args=args)[relation]
            )
            numer, denom, counts = trace_weighted_parts_for_trace_masks(stats, masks)
            point = ratio_from_parts(numer, denom)
            n = min(int(numer.size), int(base_num.size))
            rng = np.random.default_rng(int(seed) + 104729 * relation_idx + 9973 * row_idx)
            if n > 0 and int(n_bootstrap) > 0:
                probs = np.full(n, 1.0 / float(n))
                boot_counts = rng.multinomial(n, probs, size=int(n_bootstrap)).astype(np.float64)
                boot = bootstrap_ratio_from_counts(numer[:n], denom[:n], boot_counts)
                base_boot = bootstrap_ratio_from_counts(base_num[:n], base_den[:n], boot_counts)
                delta_boot = boot - base_boot
            else:
                boot = np.asarray([], dtype=np.float64)
                delta_boot = np.asarray([], dtype=np.float64)
            ci_low, ci_high = ci95(boot)
            delta = float(point - base_point) if np.isfinite(point) and np.isfinite(base_point) else float("nan")
            is_static = bool(row["is_static_baseline"])
            delta_ci_low, delta_ci_high = (0.0, 0.0) if is_static else ci95(delta_boot)
            p_value = float("nan") if is_static or n < 2 else bootstrap_p_two_sided(delta_boot)
            denom_sum = float(np.sum(denom))
            out_rows.append(
                {
                    **row,
                    "relation": relation,
                    "relation_label": str(spec["relation_label"]),
                    "relation_title": str(spec["relation_title"]),
                    "relation_rank": int(spec["relation_rank"]),
                    "sf_group": str(args.panel_c_sf_group),
                    "match_max_deg": float(args.panel_c_match_max_deg),
                    "orthogonal_min_deg": float(args.panel_c_orthogonal_min_deg),
                    "path_median_arcmin": 0.0 if is_static else pose_path_median_arcmin(stats),
                    "n_traces": int(numer.size),
                    "n_units_median_per_trace": float(np.nanmedian(counts)) if counts.size else float("nan"),
                    "n_units_min_per_trace": int(np.nanmin(counts)) if counts.size else 0,
                    "n_units_max_per_trace": int(np.nanmax(counts)) if counts.size else 0,
                    "static_n_units_median_per_trace": float(np.nanmedian(base_counts)) if base_counts.size else float("nan"),
                    "information_numerator_bits": float(np.sum(numer)),
                    "expected_spikes": denom_sum,
                    "population_ssi_bits_per_spike": float(point),
                    "population_ssi_delta_vs_static": 0.0 if is_static else float(delta),
                    "population_ssi_percent_vs_static": 0.0
                    if is_static or not np.isfinite(base_point) or abs(base_point) <= EPS
                    else float(100.0 * delta / base_point),
                    "population_ci95_low_trace_boot": float(ci_low),
                    "population_ci95_high_trace_boot": float(ci_high),
                    "population_delta_ci95_low_trace_boot": float(delta_ci_low),
                    "population_delta_ci95_high_trace_boot": float(delta_ci_high),
                    "population_delta_percent_ci95_low_trace_boot": 0.0
                    if is_static or not np.isfinite(base_point) or abs(base_point) <= EPS
                    else float(100.0 * delta_ci_low / base_point),
                    "population_delta_percent_ci95_high_trace_boot": 0.0
                    if is_static or not np.isfinite(base_point) or abs(base_point) <= EPS
                    else float(100.0 * delta_ci_high / base_point),
                    "population_delta_p_trace_bootstrap_sign": p_value,
                    "static_population_ssi_bits_per_spike": float(base_point),
                }
            )
    return pd.DataFrame(out_rows), selection


def ratio_from_parts(numer: np.ndarray, denom: np.ndarray) -> float:
    total_den = float(np.sum(denom))
    if not np.isfinite(total_den) or total_den <= EPS:
        return float("nan")
    return float(np.sum(numer) / total_den)


def bootstrap_ratio_from_counts(numer: np.ndarray, denom: np.ndarray, counts: np.ndarray) -> np.ndarray:
    boot_num = counts @ np.asarray(numer, dtype=np.float64)
    boot_den = counts @ np.asarray(denom, dtype=np.float64)
    return np.divide(boot_num, boot_den, out=np.full_like(boot_num, np.nan), where=boot_den > EPS)


def ci95(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_p_two_sided(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    p = 2.0 * min(float(np.mean(vals <= 0.0)), float(np.mean(vals >= 0.0)))
    return float(min(max(p, 0.0), 1.0))


def pose_path_median_arcmin(stats: dict[str, Any]) -> float:
    poses = np.asarray(stats.get("pose_traces", np.asarray([])), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1] < 2:
        return float("nan")
    lengths = np.sum(np.linalg.norm(np.diff(poses, axis=1), axis=2), axis=1) * 60.0
    return float(np.nanmedian(lengths))


def panel_c_surface_condition_rows(scales: list[float]) -> list[dict[str, Any]]:
    unique_scales: list[float] = []
    for scale in scales:
        value = float(scale)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"Surface scales must be finite and non-negative, got {scale!r}.")
        if not any(np.isclose(value, existing) for existing in unique_scales):
            unique_scales.append(value)
    unique_scales = sorted(unique_scales)
    if not unique_scales:
        raise ValueError("At least one surface scale is required.")

    rows: list[dict[str, Any]] = []
    for across_idx, across_scale in enumerate(unique_scales):
        for along_idx, along_scale in enumerate(unique_scales):
            is_static = bool(np.isclose(across_scale, 0.0) and np.isclose(along_scale, 0.0))
            rows.append(
                {
                    "condition": "static_center" if is_static else condition_name(across_scale, along_scale),
                    "movement_family": "component_surface",
                    "movement_label": "across x along component grid",
                    "movement_scale": float("nan"),
                    "across_scale": float(across_scale),
                    "along_scale": float(along_scale),
                    "is_static_baseline": is_static,
                    "plot_order": 10,
                    "surface_across_index": int(across_idx),
                    "surface_along_index": int(along_idx),
                    "surface_across_label": f"{across_scale:g}x",
                    "surface_along_label": f"{along_scale:g}x",
                }
            )
    return rows


def scalar_percent_delta(value: float, baseline: float) -> float:
    if not (np.isfinite(value) and np.isfinite(baseline) and abs(float(baseline)) > EPS):
        return float("nan")
    return float(100.0 * (float(value) - float(baseline)) / float(baseline))


def total_per_selected_trace(values: np.ndarray, n_selected_trace_samples: int) -> float:
    n = int(n_selected_trace_samples)
    if n <= 0:
        return float("nan")
    total = float(np.sum(np.asarray(values, dtype=np.float64)))
    return float(total / float(n)) if np.isfinite(total) else float("nan")


def total_per_unit_trace(values: np.ndarray, counts: np.ndarray) -> float:
    n = int(np.sum(np.asarray(counts, dtype=np.int64)))
    if n <= 0:
        return float("nan")
    total = float(np.sum(np.asarray(values, dtype=np.float64)))
    return float(total / float(n)) if np.isfinite(total) else float("nan")


def surface_quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot make measured-arcmin bins from an empty array.")
    bins = int(n_bins)
    if bins < 1:
        raise ValueError(f"--panel-c-arcmin-surface-bins must be >= 1, got {n_bins}.")
    edges = np.quantile(finite, np.linspace(0.0, 1.0, bins + 1))
    span = max(float(edges[-1] - edges[0]), 1e-6)
    edges[0] -= 1e-6 * span
    edges[-1] += 1e-6 * span
    return edges.astype(np.float64)


def assign_surface_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, -1, dtype=np.int64)
    ok = np.isfinite(arr)
    out[ok] = np.searchsorted(np.asarray(edges, dtype=np.float64)[1:-1], arr[ok], side="right")
    out[(out < 0) | (out >= len(edges) - 1)] = -1
    return out


def trace_component_metrics(stats: dict[str, Any]) -> dict[str, np.ndarray]:
    poses = np.asarray(stats.get("pose_traces", np.asarray([])), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1] < 2 or poses.shape[2] != 2:
        raise ValueError(f"Expected pose_traces with shape (trace, frame, xy), got {poses.shape}.")
    n_traces = int(poses.shape[0])
    if "stimulus_orientation_deg_per_trace" in stats:
        orientations = np.asarray(stats["stimulus_orientation_deg_per_trace"], dtype=np.float64)
    elif "contour_axis_deg_per_trace" in stats:
        orientations = axis_180_deg(np.asarray(stats["contour_axis_deg_per_trace"], dtype=np.float64) - 90.0)
    else:
        orientations = np.zeros(n_traces, dtype=np.float64)
    if orientations.size != n_traces:
        raise ValueError(f"Expected {n_traces} stimulus orientations, got {orientations.size}.")

    theta = np.deg2rad(orientations)
    across_axis = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    along_axis = np.stack([-np.sin(theta), np.cos(theta)], axis=1)
    steps = np.diff(poses, axis=1)
    centered = poses - np.nanmean(poses, axis=1, keepdims=True)

    across_step = np.einsum("ntd,nd->nt", steps, across_axis)
    along_step = np.einsum("ntd,nd->nt", steps, along_axis)
    across_pos = np.einsum("ntd,nd->nt", centered, across_axis)
    along_pos = np.einsum("ntd,nd->nt", centered, along_axis)
    return {
        "across_path_arcmin": (np.nansum(np.abs(across_step), axis=1) * 60.0).astype(np.float64),
        "along_path_arcmin": (np.nansum(np.abs(along_step), axis=1) * 60.0).astype(np.float64),
        "across_rms_arcmin": (np.sqrt(np.nanmean(across_pos * across_pos, axis=1)) * 60.0).astype(np.float64),
        "along_rms_arcmin": (np.sqrt(np.nanmean(along_pos * along_pos, axis=1)) * 60.0).astype(np.float64),
    }


def summarize_panel_c_component_surfaces(
    *,
    rows: list[dict[str, Any]],
    stats_by_condition: dict[str, dict[str, Any]],
    units: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    specs = panel_c_orientation_relation_specs(args)
    base_stats = stats_by_condition["static_center"]
    base_axes = stats_trace_contour_axes(base_stats, args)
    base_masks_by_relation = orientation_relation_masks_for_traces(units, base_axes, args=args)
    component_metrics_by_condition = {
        condition: trace_component_metrics(stats_by_condition[condition])
        for condition in sorted({str(row["condition"]) for row in rows})
    }

    base_by_relation: dict[str, dict[str, float | np.ndarray]] = {}
    for spec in specs:
        relation = str(spec["relation"])
        base_num, base_den, base_counts = trace_weighted_parts_for_trace_masks(
            base_stats,
            base_masks_by_relation[relation],
        )
        base_n_selected = int(np.count_nonzero(base_counts > 0))
        base_by_relation[relation] = {
            "numer": base_num,
            "denom": base_den,
            "counts": base_counts,
            "point": ratio_from_parts(base_num, base_den),
            "info_per_unit_trace": total_per_unit_trace(base_num, base_counts),
            "spikes_per_unit_trace": total_per_unit_trace(base_den, base_counts),
            "n_selected_trace_samples": base_n_selected,
            "n_unit_trace_samples": int(np.sum(base_counts)),
        }

    out_rows: list[dict[str, Any]] = []
    for spec in specs:
        relation = str(spec["relation"])
        base = base_by_relation[relation]
        base_point = float(base["point"])
        base_info = float(base["info_per_unit_trace"])
        base_spikes = float(base["spikes_per_unit_trace"])
        for row in rows:
            condition = str(row["condition"])
            stats = stats_by_condition[condition]
            axes = stats_trace_contour_axes(stats, args)
            masks = (
                base_masks_by_relation[relation]
                if condition == "static_center" and np.array_equal(axes, base_axes)
                else orientation_relation_masks_for_traces(units, axes, args=args)[relation]
            )
            numer, denom, counts = trace_weighted_parts_for_trace_masks(stats, masks)
            point = ratio_from_parts(numer, denom)
            n_selected = int(np.count_nonzero(counts > 0))
            info = total_per_unit_trace(numer, counts)
            spikes = total_per_unit_trace(denom, counts)
            unit_trace_samples = int(np.sum(counts))
            metrics = component_metrics_by_condition[condition]
            for family in PANEL_C_SURFACE_FAMILIES:
                across_values = np.asarray(metrics[str(family["across"])], dtype=np.float64)
                along_values = np.asarray(metrics[str(family["along"])], dtype=np.float64)
                out_rows.append(
                    {
                        **row,
                        "metric_family": str(family["key"]),
                        "metric_family_title": str(family["title"]),
                        "metric_family_description": str(family["description"]),
                        "relation": relation,
                        "relation_label": str(spec["relation_label"]),
                        "relation_title": str(spec["relation_title"]),
                        "relation_rank": int(spec["relation_rank"]),
                        "sf_group": str(args.panel_c_sf_group),
                        "match_max_deg": float(args.panel_c_match_max_deg),
                        "orthogonal_min_deg": float(args.panel_c_orthogonal_min_deg),
                        "across_median_arcmin": float(np.nanmedian(across_values)),
                        "along_median_arcmin": float(np.nanmedian(along_values)),
                        "n_traces": int(numer.size),
                        "n_selected_trace_samples": n_selected,
                        "n_unit_trace_samples": unit_trace_samples,
                        "n_units_median_per_trace": float(np.nanmedian(counts)) if counts.size else float("nan"),
                        "n_units_min_per_trace": int(np.nanmin(counts)) if counts.size else 0,
                        "n_units_max_per_trace": int(np.nanmax(counts)) if counts.size else 0,
                        "static_n_selected_trace_samples": int(base["n_selected_trace_samples"]),
                        "static_n_unit_trace_samples": int(base["n_unit_trace_samples"]),
                        "information_numerator_bits": float(np.sum(numer)),
                        "expected_spikes": float(np.sum(denom)),
                        "population_ssi_bits_per_spike": float(point),
                        "population_ssi_percent_vs_static": scalar_percent_delta(point, base_point),
                        "information_bits_per_sample": float(info),
                        "information_bits_per_sample_percent_vs_static": scalar_percent_delta(info, base_info),
                        "expected_spikes_per_sample": float(spikes),
                        "expected_spikes_per_sample_percent_vs_static": scalar_percent_delta(spikes, base_spikes),
                        "static_population_ssi_bits_per_spike": base_point,
                        "static_information_bits_per_sample": base_info,
                        "static_expected_spikes_per_sample": base_spikes,
                    }
                )
    return pd.DataFrame(out_rows)


def arcmin_binned_surface_output_stem(args: argparse.Namespace, output_stem: str) -> str:
    bins = int(args.panel_c_arcmin_surface_bins)
    if args.panel_c_output_stem:
        return f"{output_stem}_arcmin_binned_component_surfaces_n{bins}"
    if str(args.orientation_mode) == "random_blocks":
        orientation_tag = f"_random_ori_{int(args.n_orientation_blocks)}blocks"
    else:
        orientation_tag = "_random_ori" if has_random_trace_orientations(args) else ""
    return f"rr100_single_contour_panel_c_{args.panel_c_sf_group}_arcmin_binned_component_surfaces_n{bins}{orientation_tag}"


def summarize_panel_c_arcmin_binned_surfaces(
    *,
    rows: list[dict[str, Any]],
    stats_by_condition: dict[str, dict[str, Any]],
    units: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    moving_rows = [row for row in rows if not bool(row.get("is_static_baseline", False))]
    if not moving_rows:
        raise ValueError("Measured-arcmin binning needs at least one moving surface condition.")

    n_bins = int(args.panel_c_arcmin_surface_bins)
    specs = panel_c_orientation_relation_specs(args)
    base_stats = stats_by_condition["static_center"]
    base_axes = stats_trace_contour_axes(base_stats, args)
    base_masks_by_relation = orientation_relation_masks_for_traces(units, base_axes, args=args)

    sample_parts_by_relation: dict[str, list[dict[str, Any]]] = {
        str(spec["relation"]): [] for spec in specs
    }
    base_by_relation: dict[str, dict[str, float]] = {}
    for spec in specs:
        relation = str(spec["relation"])
        base_num, base_den, base_counts = trace_weighted_parts_for_trace_masks(
            base_stats,
            base_masks_by_relation[relation],
        )
        base_by_relation[relation] = {
            "point": ratio_from_parts(base_num, base_den),
            "info_per_unit_trace": total_per_unit_trace(base_num, base_counts),
            "spikes_per_unit_trace": total_per_unit_trace(base_den, base_counts),
            "n_unit_trace_samples": float(np.sum(base_counts)),
            "n_trace_samples": float(base_num.size),
            "n_selected_trace_samples": float(np.count_nonzero(base_counts > 0)),
        }

    for row in moving_rows:
        condition = str(row["condition"])
        stats = stats_by_condition[condition]
        axes = stats_trace_contour_axes(stats, args)
        metrics = trace_component_metrics(stats)
        masks_by_relation = orientation_relation_masks_for_traces(units, axes, args=args)
        for spec in specs:
            relation = str(spec["relation"])
            numer, denom, counts = trace_weighted_parts_for_trace_masks(stats, masks_by_relation[relation])
            sample_parts_by_relation[relation].append(
                {
                    "row": row,
                    "numer": numer,
                    "denom": denom,
                    "counts": counts,
                    "metrics": metrics,
                }
            )

    out_rows: list[dict[str, Any]] = []
    for spec in specs:
        relation = str(spec["relation"])
        parts = sample_parts_by_relation[relation]
        base = base_by_relation[relation]
        for family in PANEL_C_SURFACE_FAMILIES:
            across_col = str(family["across"])
            along_col = str(family["along"])
            across_values = np.concatenate(
                [np.asarray(part["metrics"][across_col], dtype=np.float64) for part in parts],
                axis=0,
            )
            along_values = np.concatenate(
                [np.asarray(part["metrics"][along_col], dtype=np.float64) for part in parts],
                axis=0,
            )
            across_edges = surface_quantile_edges(across_values, n_bins)
            along_edges = surface_quantile_edges(along_values, n_bins)
            across_bins = assign_surface_bins(across_values, across_edges)
            along_bins = assign_surface_bins(along_values, along_edges)
            numer_all = np.concatenate([np.asarray(part["numer"], dtype=np.float64) for part in parts], axis=0)
            denom_all = np.concatenate([np.asarray(part["denom"], dtype=np.float64) for part in parts], axis=0)
            counts_all = np.concatenate([np.asarray(part["counts"], dtype=np.int64) for part in parts], axis=0)
            across_scales_all = np.concatenate(
                [
                    np.full(np.asarray(part["numer"]).shape, float(part["row"]["across_scale"]), dtype=np.float64)
                    for part in parts
                ],
                axis=0,
            )
            along_scales_all = np.concatenate(
                [
                    np.full(np.asarray(part["numer"]).shape, float(part["row"]["along_scale"]), dtype=np.float64)
                    for part in parts
                ],
                axis=0,
            )
            valid = (across_bins >= 0) & (along_bins >= 0)
            for across_bin in range(n_bins):
                for along_bin in range(n_bins):
                    mask = valid & (across_bins == across_bin) & (along_bins == along_bin)
                    cell_num = numer_all[mask]
                    cell_den = denom_all[mask]
                    cell_counts = counts_all[mask]
                    point = ratio_from_parts(cell_num, cell_den)
                    info = total_per_unit_trace(cell_num, cell_counts)
                    spikes = total_per_unit_trace(cell_den, cell_counts)
                    out_rows.append(
                        {
                            "metric_family": str(family["key"]),
                            "metric_family_title": str(family["title"]),
                            "metric_family_description": str(family["description"]),
                            "relation": relation,
                            "relation_label": str(spec["relation_label"]),
                            "relation_title": str(spec["relation_title"]),
                            "relation_rank": int(spec["relation_rank"]),
                            "sf_group": str(args.panel_c_sf_group),
                            "match_max_deg": float(args.panel_c_match_max_deg),
                            "orthogonal_min_deg": float(args.panel_c_orthogonal_min_deg),
                            "arcmin_bin_count": int(n_bins),
                            "across_bin": int(across_bin + 1),
                            "along_bin": int(along_bin + 1),
                            "across_bin_label": f"Q{across_bin + 1}",
                            "along_bin_label": f"Q{along_bin + 1}",
                            "across_min_arcmin": float(across_edges[across_bin]),
                            "across_max_arcmin": float(across_edges[across_bin + 1]),
                            "along_min_arcmin": float(along_edges[along_bin]),
                            "along_max_arcmin": float(along_edges[along_bin + 1]),
                            "across_median_arcmin": float(np.nanmedian(across_values[mask])) if np.any(mask) else float("nan"),
                            "along_median_arcmin": float(np.nanmedian(along_values[mask])) if np.any(mask) else float("nan"),
                            "across_scale_median": float(np.nanmedian(across_scales_all[mask])) if np.any(mask) else float("nan"),
                            "along_scale_median": float(np.nanmedian(along_scales_all[mask])) if np.any(mask) else float("nan"),
                            "n_trace_condition_samples": int(np.count_nonzero(mask)),
                            "n_selected_trace_samples": int(np.count_nonzero(cell_counts > 0)),
                            "n_unit_trace_samples": int(np.sum(cell_counts)),
                            "information_numerator_bits": float(np.sum(cell_num)),
                            "expected_spikes": float(np.sum(cell_den)),
                            "population_ssi_bits_per_spike": float(point),
                            "population_ssi_percent_vs_static": scalar_percent_delta(point, float(base["point"])),
                            "information_bits_per_sample": float(info),
                            "information_bits_per_sample_percent_vs_static": scalar_percent_delta(
                                info,
                                float(base["info_per_unit_trace"]),
                            ),
                            "expected_spikes_per_sample": float(spikes),
                            "expected_spikes_per_sample_percent_vs_static": scalar_percent_delta(
                                spikes,
                                float(base["spikes_per_unit_trace"]),
                            ),
                            "static_population_ssi_bits_per_spike": float(base["point"]),
                            "static_information_bits_per_sample": float(base["info_per_unit_trace"]),
                            "static_expected_spikes_per_sample": float(base["spikes_per_unit_trace"]),
                            "static_n_trace_samples": int(base["n_trace_samples"]),
                            "static_n_selected_trace_samples": int(base["n_selected_trace_samples"]),
                            "static_n_unit_trace_samples": int(base["n_unit_trace_samples"]),
                            "normalization_note": (
                                "population SSI = sum(unit_bits_per_spike * unit_mean_rate) / sum(unit_mean_rate); "
                                "information/spikes per sample divide summed numerator/spikes by valid unit-trace count"
                            ),
                        }
                    )
    return pd.DataFrame(out_rows)


def surface_output_stem(args: argparse.Namespace, output_stem: str) -> str:
    if args.panel_c_output_stem:
        return f"{output_stem}_component_surfaces"
    if str(args.orientation_mode) == "random_blocks":
        orientation_tag = f"_random_ori_{int(args.n_orientation_blocks)}blocks"
    else:
        orientation_tag = "_random_ori" if has_random_trace_orientations(args) else ""
    return f"rr100_single_contour_panel_c_{args.panel_c_sf_group}_component_surfaces{orientation_tag}"


def surface_grid(surface: pd.DataFrame, value_col: str) -> np.ndarray:
    n_across = int(pd.to_numeric(surface["surface_across_index"], errors="coerce").max()) + 1
    n_along = int(pd.to_numeric(surface["surface_along_index"], errors="coerce").max()) + 1
    arr = np.full((n_across, n_along), np.nan, dtype=np.float64)
    for row in surface.itertuples(index=False):
        arr[int(row.surface_across_index), int(row.surface_along_index)] = float(getattr(row, value_col))
    return arr


def surface_axis_labels(surface: pd.DataFrame, axis: str) -> list[str]:
    index_col = f"surface_{axis}_index"
    scale_col = f"{axis}_scale"
    median_col = f"{axis}_median_arcmin"
    n_bins = int(pd.to_numeric(surface[index_col], errors="coerce").max()) + 1
    labels: list[str] = []
    for idx in range(n_bins):
        sub = surface[pd.to_numeric(surface[index_col], errors="coerce").eq(idx)]
        if sub.empty:
            labels.append("")
            continue
        scale = float(np.nanmedian(pd.to_numeric(sub[scale_col], errors="coerce").to_numpy(dtype=float)))
        median = float(np.nanmedian(pd.to_numeric(sub[median_col], errors="coerce").to_numpy(dtype=float)))
        labels.append(f"{scale:g}x\n{median:.1f}")
    return labels


def surface_symmetric_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    mag = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), 1.0)
    return -1.05 * mag, 1.05 * mag


def surface_color_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if np.isclose(vmin, vmax):
        return vmin - 0.05, vmax + 0.05
    return vmin, vmax


def format_surface_count(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value >= 1000:
        return f"{value / 1000.0:.1f}k"
    return f"{int(round(value))}"


def plot_panel_c_component_surface_family(
    surface: pd.DataFrame,
    *,
    family: dict[str, Any],
    contour_axis_label: str,
    dpi: int,
) -> plt.Figure:
    n_across = int(pd.to_numeric(surface["surface_across_index"], errors="coerce").max()) + 1
    n_along = int(pd.to_numeric(surface["surface_along_index"], errors="coerce").max()) + 1
    fig_width = max(10.8, min(15.0, 3.1 + 1.42 * float(n_along)))
    fig, axes = plt.subplots(1, 4, figsize=(fig_width, 4.75), dpi=int(dpi), constrained_layout=False)
    relation_title = str(surface["relation_title"].iloc[0])
    fig.suptitle(
        f"{family['title']} surface: {relation_title} on zero-gap Vernier contour",
        fontsize=13.0,
        y=0.985,
    )
    fig.text(0.5, 0.925, contour_axis_label, ha="center", va="top", fontsize=7.8, color="0.35")
    xlabels = surface_axis_labels(surface, "along")
    ylabels = surface_axis_labels(surface, "across")

    for idx, (value_col, color_label, title, cmap) in enumerate(PANEL_C_SURFACE_OUTCOMES):
        ax = axes[idx]
        values = surface_grid(surface, value_col)
        vmin, vmax = surface_symmetric_limits(values)
        image = ax.imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10.0)
        ax.set_xticks(np.arange(n_along), xlabels)
        ax.set_yticks(np.arange(n_across), ylabels)
        ax.set_xlabel("along-contour scale; median arcmin", fontsize=8.5)
        if idx == 0:
            ax.set_ylabel("across-contour scale; median arcmin", fontsize=8.5)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7.2)
        threshold = 0.52 * max(abs(vmin), abs(vmax))
        for across_idx in range(n_across):
            for along_idx in range(n_along):
                value = values[across_idx, along_idx]
                if math.isfinite(value):
                    ax.text(
                        along_idx,
                        across_idx,
                        f"{value:+.1f}",
                        ha="center",
                        va="center",
                        fontsize=6.3,
                        color="white" if abs(value) >= threshold else "0.16",
                    )
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_label(color_label, fontsize=7.2)
        cbar.ax.tick_params(labelsize=7.0)

    count_ax = axes[3]
    counts = surface_grid(surface, "n_unit_trace_samples")
    log_counts = np.log10(np.maximum(counts, 1.0))
    vmin, vmax = surface_color_limits(log_counts)
    count_image = count_ax.imshow(log_counts, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    count_ax.set_title("Selected unit-trace samples", fontsize=10.0)
    count_ax.set_xticks(np.arange(n_along), xlabels)
    count_ax.set_yticks(np.arange(n_across), ylabels)
    count_ax.set_yticklabels([])
    count_ax.set_xlabel("along-contour scale; median arcmin", fontsize=8.5)
    count_ax.tick_params(labelsize=7.2)
    count_threshold = float(np.nanmedian(counts)) if np.isfinite(counts).any() else 0.0
    for across_idx in range(n_across):
        for along_idx in range(n_along):
            value = counts[across_idx, along_idx]
            count_ax.text(
                along_idx,
                across_idx,
                format_surface_count(value),
                ha="center",
                va="center",
                fontsize=6.1,
                color="white" if value >= count_threshold else "0.12",
            )
    cbar = fig.colorbar(count_image, ax=count_ax, fraction=0.046, pad=0.025)
    cbar.set_label("log10 unit-trace samples", fontsize=7.2)
    cbar.ax.tick_params(labelsize=7.0)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.text(
        0.5,
        0.025,
        "Rows increase across-contour eye-motion component scale; columns increase along-contour component scale. "
        "Tick labels show scale and measured median projected arcmin; values are spike-weighted population estimates "
        "relative to the static zero-gap contour.",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="0.30",
    )
    fig.subplots_adjust(left=0.058, right=0.992, top=0.84, bottom=0.18, wspace=0.36)
    return fig


def plot_panel_c_component_surfaces(
    summary: pd.DataFrame,
    *,
    contour_axis_label: str,
    out_dir: Path,
    output_stem: str,
    dpi: int,
) -> tuple[Path, list[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{output_stem}_no_gap.pdf"
    png_paths: list[Path] = []
    relation_rows = (
        summary[["relation", "relation_rank"]]
        .drop_duplicates()
        .sort_values("relation_rank")
    )
    with PdfPages(pdf) as pages:
        for relation_row in relation_rows.itertuples(index=False):
            relation = str(relation_row.relation)
            for family in PANEL_C_SURFACE_FAMILIES:
                surface = summary[
                    summary["relation"].astype(str).eq(relation)
                    & summary["metric_family"].astype(str).eq(str(family["key"]))
                ].copy()
                if surface.empty:
                    continue
                fig = plot_panel_c_component_surface_family(
                    surface,
                    family=family,
                    contour_axis_label=contour_axis_label,
                    dpi=int(dpi),
                )
                pages.savefig(fig, bbox_inches="tight")
                relation_slug = relation.replace("contour_", "")
                png = out_dir / f"{output_stem}_{relation_slug}_{family['key']}_no_gap.png"
                fig.savefig(png, bbox_inches="tight", facecolor="white")
                png_paths.append(png)
                plt.close(fig)
    return pdf, png_paths


def arcmin_surface_grid(surface: pd.DataFrame, value_col: str) -> np.ndarray:
    n_bins = int(pd.to_numeric(surface["arcmin_bin_count"], errors="coerce").max())
    arr = np.full((n_bins, n_bins), np.nan, dtype=np.float64)
    for row in surface.itertuples(index=False):
        arr[int(row.across_bin) - 1, int(row.along_bin) - 1] = float(getattr(row, value_col))
    return arr


def arcmin_surface_axis_labels(surface: pd.DataFrame, axis: str) -> list[str]:
    n_bins = int(pd.to_numeric(surface["arcmin_bin_count"], errors="coerce").max())
    labels: list[str] = []
    for idx in range(1, n_bins + 1):
        if axis == "across":
            values = surface[surface["across_bin"].eq(idx)]["across_median_arcmin"].to_numpy(dtype=float)
        else:
            values = surface[surface["along_bin"].eq(idx)]["along_median_arcmin"].to_numpy(dtype=float)
        median = float(np.nanmedian(values)) if values.size else float("nan")
        if not np.isfinite(median):
            labels.append(f"Q{idx}\n")
        elif median >= 10.0:
            labels.append(f"Q{idx}\n{median:.0f}")
        else:
            labels.append(f"Q{idx}\n{median:.1f}")
    return labels


def plot_panel_c_arcmin_binned_surface_family(
    surface: pd.DataFrame,
    *,
    family: dict[str, Any],
    contour_axis_label: str,
    dpi: int,
) -> plt.Figure:
    n_bins = int(pd.to_numeric(surface["arcmin_bin_count"], errors="coerce").max())
    fig_width = max(12.8, min(17.2, 6.0 + 0.72 * float(n_bins)))
    fig, axes = plt.subplots(1, 4, figsize=(fig_width, 5.15), dpi=int(dpi), constrained_layout=False)
    relation_title = str(surface["relation_title"].iloc[0])
    fig.suptitle(
        f"{n_bins}x{n_bins} {family['title'].lower()} binned surface: {relation_title} on zero-gap Vernier contour",
        fontsize=13.0,
        y=0.985,
    )
    fig.text(0.5, 0.932, contour_axis_label, ha="center", va="top", fontsize=7.8, color="0.35")
    xlabels = arcmin_surface_axis_labels(surface, "along")
    ylabels = arcmin_surface_axis_labels(surface, "across")
    label_size = 5.8 if n_bins > 10 else 7.0
    value_size = 4.8 if n_bins > 10 else 6.1

    for idx, (value_col, color_label, title, cmap) in enumerate(PANEL_C_SURFACE_OUTCOMES):
        ax = axes[idx]
        values = arcmin_surface_grid(surface, value_col)
        vmin, vmax = surface_symmetric_limits(values)
        image = ax.imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title.replace("efficiency", "residual"), fontsize=9.5)
        ax.set_xticks(np.arange(n_bins), xlabels)
        ax.set_yticks(np.arange(n_bins), ylabels)
        ax.set_xlabel("along bin; median arcmin", fontsize=8.0)
        if idx == 0:
            ax.set_ylabel("across bin; median arcmin", fontsize=8.0)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=label_size)
        threshold = 0.52 * max(abs(vmin), abs(vmax))
        for across_idx in range(n_bins):
            for along_idx in range(n_bins):
                value = values[across_idx, along_idx]
                if math.isfinite(value):
                    ax.text(
                        along_idx,
                        across_idx,
                        f"{value:+.0f}" if n_bins > 10 else f"{value:+.1f}",
                        ha="center",
                        va="center",
                        fontsize=value_size,
                        color="white" if abs(value) >= threshold else "0.16",
                    )
        cbar = fig.colorbar(image, ax=ax, fraction=0.042, pad=0.02)
        cbar.set_label(color_label, fontsize=6.8)
        cbar.ax.tick_params(labelsize=6.5)

    count_ax = axes[3]
    counts = arcmin_surface_grid(surface, "n_unit_trace_samples")
    log_counts = np.log10(np.maximum(counts, 1.0))
    vmin, vmax = surface_color_limits(log_counts)
    count_image = count_ax.imshow(log_counts, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    count_ax.set_title("Selected unit-trace samples", fontsize=9.5)
    count_ax.set_xticks(np.arange(n_bins), xlabels)
    count_ax.set_yticks(np.arange(n_bins), [])
    count_ax.set_xlabel("along bin; median arcmin", fontsize=8.0)
    count_ax.tick_params(labelsize=label_size)
    median_count = float(np.nanmedian(counts)) if np.isfinite(counts).any() else 0.0
    for across_idx in range(n_bins):
        for along_idx in range(n_bins):
            value = counts[across_idx, along_idx]
            count_ax.text(
                along_idx,
                across_idx,
                format_surface_count(value),
                ha="center",
                va="center",
                fontsize=4.6 if n_bins > 10 else 5.9,
                color="white" if value >= median_count else "0.13",
            )
    cbar = fig.colorbar(count_image, ax=count_ax, fraction=0.042, pad=0.02)
    cbar.set_label("log10 unit-trace samples", fontsize=6.8)
    cbar.ax.tick_params(labelsize=6.5)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.02,
        "Generated moving trace-condition samples are binned by measured contour-relative component arcmin. "
        "Population SSI is spike weighted; information and expected spikes are normalized per valid unit-trace sample.",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="0.30",
    )
    fig.subplots_adjust(left=0.055, right=0.992, top=0.82, bottom=0.18, wspace=0.30)
    return fig


def plot_panel_c_arcmin_binned_surfaces(
    summary: pd.DataFrame,
    *,
    contour_axis_label: str,
    out_dir: Path,
    output_stem: str,
    dpi: int,
) -> tuple[Path, list[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{output_stem}_no_gap.pdf"
    png_paths: list[Path] = []
    relation_rows = (
        summary[["relation", "relation_rank"]]
        .drop_duplicates()
        .sort_values("relation_rank")
    )
    with PdfPages(pdf) as pages:
        for relation_row in relation_rows.itertuples(index=False):
            relation = str(relation_row.relation)
            for family in PANEL_C_SURFACE_FAMILIES:
                surface = summary[
                    summary["relation"].astype(str).eq(relation)
                    & summary["metric_family"].astype(str).eq(str(family["key"]))
                ].copy()
                if surface.empty:
                    continue
                fig = plot_panel_c_arcmin_binned_surface_family(
                    surface,
                    family=family,
                    contour_axis_label=contour_axis_label,
                    dpi=int(dpi),
                )
                pages.savefig(fig, bbox_inches="tight")
                relation_slug = relation.replace("contour_", "")
                png = out_dir / f"{output_stem}_{relation_slug}_{family['key']}_no_gap.png"
                fig.savefig(png, bbox_inches="tight", facecolor="white")
                png_paths.append(png)
                plt.close(fig)
    return pdf, png_paths


def summarize_panel_c(
    *,
    rows: list[dict[str, Any]],
    stats_by_condition: dict[str, dict[str, Any]],
    unit_indices: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    base_stats = stats_by_condition["static_center"]
    base_num, base_den = trace_weighted_parts(base_stats, unit_indices)
    base_point = ratio_from_parts(base_num, base_den)
    out_rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        condition = str(row["condition"])
        stats = stats_by_condition[condition]
        numer, denom = trace_weighted_parts(stats, unit_indices)
        point = ratio_from_parts(numer, denom)
        n = min(int(numer.size), int(base_num.size))
        rng = np.random.default_rng(int(seed) + 9973 * row_idx)
        if n > 0 and int(n_bootstrap) > 0:
            counts = rng.multinomial(n, np.full(n, 1.0 / float(n)), size=int(n_bootstrap)).astype(np.float64)
            boot = bootstrap_ratio_from_counts(numer[:n], denom[:n], counts)
            base_boot = bootstrap_ratio_from_counts(base_num[:n], base_den[:n], counts)
            delta_boot = boot - base_boot
        else:
            boot = np.asarray([], dtype=np.float64)
            delta_boot = np.asarray([], dtype=np.float64)
        ci_low, ci_high = ci95(boot)
        delta = float(point - base_point) if np.isfinite(point) and np.isfinite(base_point) else float("nan")
        delta_ci_low, delta_ci_high = (0.0, 0.0) if bool(row["is_static_baseline"]) else ci95(delta_boot)
        p_value = float("nan") if bool(row["is_static_baseline"]) or n < 2 else bootstrap_p_two_sided(delta_boot)
        out_rows.append(
            {
                **row,
                "path_median_arcmin": 0.0 if bool(row["is_static_baseline"]) else pose_path_median_arcmin(stats),
                "n_traces": int(numer.size),
                "n_units": int(unit_indices.size),
                "information_numerator_bits": float(np.sum(numer)),
                "expected_spikes": float(np.sum(denom)),
                "population_ssi_bits_per_spike": float(point),
                "population_ssi_delta_vs_static": 0.0 if bool(row["is_static_baseline"]) else float(delta),
                "population_ssi_percent_vs_static": 0.0
                if bool(row["is_static_baseline"])
                else float(100.0 * delta / base_point),
                "population_ci95_low_trace_boot": float(ci_low),
                "population_ci95_high_trace_boot": float(ci_high),
                "population_delta_ci95_low_trace_boot": float(delta_ci_low),
                "population_delta_ci95_high_trace_boot": float(delta_ci_high),
                "population_delta_percent_ci95_low_trace_boot": 0.0
                if bool(row["is_static_baseline"])
                else float(100.0 * delta_ci_low / base_point),
                "population_delta_percent_ci95_high_trace_boot": 0.0
                if bool(row["is_static_baseline"])
                else float(100.0 * delta_ci_high / base_point),
                "population_delta_p_trace_bootstrap_sign": p_value,
                "static_population_ssi_bits_per_spike": float(base_point),
            }
        )
    return pd.DataFrame(out_rows)


def x_broken_log(values: np.ndarray | list[float], *, min_pos: float, max_pos: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    mapped = np.zeros_like(x, dtype=np.float64)
    positive = x > 0.0
    if not positive.any():
        return mapped
    if not (np.isfinite(min_pos) and np.isfinite(max_pos) and max_pos > min_pos > 0.0):
        mapped[positive] = x[positive]
        return mapped
    span = 5.1
    mapped[positive] = 1.0 + span * np.log(x[positive] / min_pos) / np.log(max_pos / min_pos)
    return mapped


def p_label(value: float) -> str:
    if not np.isfinite(float(value)):
        return "p=n/a"
    if float(value) < 0.001:
        return "p<0.001"
    return f"p={float(value):.3f}"


def add_bracket(
    ax: plt.Axes,
    *,
    x0: float,
    x1: float,
    y: float,
    text: str,
    color: str,
    linestyle: str | tuple[int, tuple[float, ...]] = "-",
    text_x: float | None = None,
    text_ha: str = "center",
) -> None:
    tick = 0.7
    ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], color=color, lw=1.0, ls=linestyle, zorder=6)
    ax.text(
        0.5 * (x0 + x1) if text_x is None else text_x,
        y + 0.45,
        text,
        ha=text_ha,
        va="bottom",
        color=color,
        fontsize=7.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6},
        zorder=7,
    )


def baseline_projection_arcmin(summary: pd.DataFrame) -> dict[str, float]:
    refs: dict[str, float] = {}
    for family in ("across", "along"):
        sub = summary[
            summary["movement_family"].astype(str).eq(family)
            & np.isclose(pd.to_numeric(summary["movement_scale"], errors="coerce"), 1.0)
        ]
        if sub.empty:
            continue
        value = float(pd.to_numeric(sub["path_median_arcmin"], errors="coerce").iloc[0])
        if np.isfinite(value) and value > 0.0:
            refs[family] = value
    return refs


def draw_baseline_projection_marker(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    min_pos: float,
    max_pos: float,
) -> dict[str, float]:
    refs = baseline_projection_arcmin(summary)
    values = np.asarray(list(refs.values()), dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return refs

    xs = x_broken_log(values, min_pos=min_pos, max_pos=max_pos)
    x0 = float(np.nanmin(xs))
    x1 = float(np.nanmax(xs))
    center = float(np.nanmean(xs))
    min_width = 0.18
    if x1 - x0 < min_width:
        x0 = center - 0.5 * min_width
        x1 = center + 0.5 * min_width

    ax.axvspan(x0, x1, color="#d95f02", alpha=0.055, linewidth=0.0, zorder=0)
    transform = ax.get_xaxis_transform()
    y = 0.035
    tick = 0.022
    ax.plot([x0, x0, x1, x1], [y + tick, y, y, y + tick], color="#9c4a00", lw=1.0, transform=transform, clip_on=False, zorder=6)
    ax.text(
        center,
        y + tick + 0.012,
        "native drift\nprojection (1x)",
        transform=transform,
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#7d3a00",
        linespacing=0.9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.7},
        zorder=7,
    )
    return refs


def panel_c_motion_component_styles() -> dict[str, tuple[str, str, str | tuple[int, tuple[float, ...]], str]]:
    return {
        "across": (
            "across-contour eye motion\nalong component held",
            "#d95f02",
            "-",
            "o",
        ),
        "along": (
            "along-contour eye motion\nacross component held",
            "#d95f02",
            (0, (4.2, 2.0)),
            "s",
        ),
    }


def plot_panel_c(
    summary: pd.DataFrame,
    *,
    selected_units: pd.DataFrame,
    contour_axis_label: str,
    out_dir: Path,
    output_stem: str,
    selection_title: str,
    dpi: int,
) -> tuple[Path, Path]:
    plot_df = summary[~summary["is_static_baseline"].astype(bool)].copy()
    positive_paths = plot_df["path_median_arcmin"].to_numpy(dtype=float)
    positive_paths = positive_paths[np.isfinite(positive_paths) & (positive_paths > 0.0)]
    min_pos = float(np.nanmin(positive_paths)) if positive_paths.size else 1.0
    max_pos = float(np.nanmax(positive_paths)) if positive_paths.size else max(min_pos * 1.1, 2.0)
    if max_pos <= min_pos:
        max_pos = min_pos * 1.1

    styles = panel_c_motion_component_styles()
    fig, ax = plt.subplots(figsize=(5.4, 4.3), dpi=int(dpi))
    y_values: list[float] = [0.0]
    for family, (label, color, linestyle, marker) in styles.items():
        sub = plot_df[plot_df["movement_family"].eq(family)].sort_values("movement_scale")
        if sub.empty:
            continue
        x = x_broken_log(sub["path_median_arcmin"].to_numpy(dtype=float), min_pos=min_pos, max_pos=max_pos)
        y = sub["population_ssi_percent_vs_static"].to_numpy(dtype=float)
        lo = sub["population_delta_percent_ci95_low_trace_boot"].to_numpy(dtype=float)
        hi = sub["population_delta_percent_ci95_high_trace_boot"].to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=2.3, marker=marker, markersize=5.2, label=label, zorder=3)
        ax.errorbar(x, y, yerr=yerr, color=color, linestyle="none", linewidth=1.2, capsize=2.2, zorder=2)
        y_values.extend(y[np.isfinite(y)].tolist())
        y_values.extend(lo[np.isfinite(lo)].tolist())
        y_values.extend(hi[np.isfinite(hi)].tolist())
        ax.scatter([0.0], [0.0], marker=marker, s=33, facecolors="white", edgecolors=color, linewidths=1.3, zorder=5)

    finite_y = np.asarray(y_values, dtype=float)
    finite_y = finite_y[np.isfinite(finite_y)]
    lo = min(-5.0, float(np.nanmin(finite_y)) if finite_y.size else -5.0)
    hi = max(5.0, float(np.nanmax(finite_y)) if finite_y.size else 5.0)
    span = max(hi - lo, 1.0)
    ax.set_ylim(lo - 0.08 * span, hi + 0.32 * span)
    y_lo, y_hi = ax.get_ylim()
    y_span = max(y_hi - y_lo, 1.0)

    across_first = plot_df[plot_df["movement_family"].eq("across")].sort_values("movement_scale").head(1)
    along_first = plot_df[plot_df["movement_family"].eq("along")].sort_values("movement_scale").head(1)
    if not across_first.empty and not along_first.empty:
        first_path = max(
            float(across_first["path_median_arcmin"].iloc[0]),
            float(along_first["path_median_arcmin"].iloc[0]),
        )
        x1 = x_broken_log([first_path], min_pos=min_pos, max_pos=max_pos)[0]
        label = (
            "across "
            + p_label(float(across_first["population_delta_p_trace_bootstrap_sign"].iloc[0]))
            + "\nalong "
            + p_label(float(along_first["population_delta_p_trace_bootstrap_sign"].iloc[0]))
        )
        add_bracket(
            ax,
            x0=0.0,
            x1=float(x1),
            y=y_hi - 0.12 * y_span,
            text=label,
            color="#d95f02",
            text_x=float(x1) + 0.10,
            text_ha="left",
        )

    ticks = [0.0, 25.0, 50.0, 65.0, 90.0, 120.0, 160.0, 240.0, 360.0, 480.0]
    ticks = [tick for tick in ticks if tick == 0.0 or (tick >= 0.75 * min_pos and tick <= 1.05 * max_pos)]
    ax.set_xlim(-0.12, x_broken_log([max(max_pos, max(ticks))], min_pos=min_pos, max_pos=max_pos)[0] + 0.28)
    draw_baseline_projection_marker(ax, summary, min_pos=min_pos, max_pos=max_pos)
    ax.set_xticks(x_broken_log(ticks, min_pos=min_pos, max_pos=max_pos))
    ax.set_xticklabels([str(int(tick)) for tick in ticks])
    ax.text(
        0.52,
        -0.075,
        "//",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        rotation=-20,
        clip_on=False,
    )
    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    ax.grid(True, color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Zero-gap Vernier contour: {selection_title}", fontsize=12.0)
    ax.set_ylabel("SSI change (%)")
    ax.set_xlabel("component path length (arcmin; log scale after break)")
    ax.legend(frameon=False, fontsize=8.3, loc="center left", bbox_to_anchor=(1.01, 0.32))
    n_traces = int(summary["n_traces"].max()) if "n_traces" in summary else 0
    ax.text(
        0.98,
        0.925,
        f"n={int(selected_units.shape[0])} units; {n_traces} traces; {contour_axis_label}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color="0.35",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{output_stem}_no_gap.png"
    pdf = out_dir / f"{output_stem}_no_gap.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def plot_panel_c_orientation_splits(
    summary: pd.DataFrame,
    *,
    contour_axis_label: str,
    out_dir: Path,
    output_stem: str,
    dpi: int,
) -> tuple[Path, Path]:
    plot_df = summary[~summary["is_static_baseline"].astype(bool)].copy()
    positive_paths = plot_df["path_median_arcmin"].to_numpy(dtype=float)
    positive_paths = positive_paths[np.isfinite(positive_paths) & (positive_paths > 0.0)]
    min_pos = float(np.nanmin(positive_paths)) if positive_paths.size else 1.0
    max_pos = float(np.nanmax(positive_paths)) if positive_paths.size else max(min_pos * 1.1, 2.0)
    if max_pos <= min_pos:
        max_pos = min_pos * 1.1

    y_cols = [
        "population_ssi_percent_vs_static",
        "population_delta_percent_ci95_low_trace_boot",
        "population_delta_percent_ci95_high_trace_boot",
    ]
    y_values = []
    for col in y_cols:
        vals = pd.to_numeric(plot_df[col], errors="coerce").to_numpy(dtype=float)
        y_values.extend(vals[np.isfinite(vals)].tolist())
    y_values.append(0.0)
    finite_y = np.asarray(y_values, dtype=float)
    finite_y = finite_y[np.isfinite(finite_y)]
    lo = min(-5.0, float(np.nanmin(finite_y)) if finite_y.size else -5.0)
    hi = max(5.0, float(np.nanmax(finite_y)) if finite_y.size else 5.0)
    span = max(hi - lo, 1.0)
    y_lim = (lo - 0.10 * span, hi + 0.18 * span)

    styles = panel_c_motion_component_styles()
    relation_rows = (
        summary[["relation", "relation_label", "relation_title", "relation_rank"]]
        .drop_duplicates()
        .sort_values("relation_rank")
    )
    fig, axes = plt.subplots(1, len(relation_rows), figsize=(11.2, 4.55), dpi=int(dpi), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.17, top=0.66, wspace=0.25)
    axes = np.atleast_1d(axes)
    ticks = [0.0, 25.0, 50.0, 65.0, 90.0, 120.0, 160.0, 240.0, 360.0, 480.0]
    ticks = [tick for tick in ticks if tick == 0.0 or (tick >= 0.75 * min_pos and tick <= 1.05 * max_pos)]
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for ax_idx, (ax, relation_row) in enumerate(zip(axes, relation_rows.itertuples(index=False))):
        relation = str(relation_row.relation)
        rel_summary = summary[summary["relation"].astype(str).eq(relation)].copy()
        rel_plot = plot_df[plot_df["relation"].astype(str).eq(relation)].copy()
        for family, (label, color, linestyle, marker) in styles.items():
            sub = rel_plot[rel_plot["movement_family"].eq(family)].sort_values("movement_scale")
            if sub.empty:
                continue
            x = x_broken_log(sub["path_median_arcmin"].to_numpy(dtype=float), min_pos=min_pos, max_pos=max_pos)
            y = sub["population_ssi_percent_vs_static"].to_numpy(dtype=float)
            lo_ci = sub["population_delta_percent_ci95_low_trace_boot"].to_numpy(dtype=float)
            hi_ci = sub["population_delta_percent_ci95_high_trace_boot"].to_numpy(dtype=float)
            yerr = np.vstack([np.maximum(y - lo_ci, 0.0), np.maximum(hi_ci - y, 0.0)])
            line = ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=2.1,
                marker=marker,
                markersize=4.9,
                label=label,
                zorder=3,
            )[0]
            if ax_idx == 0:
                legend_handles.append(line)
                legend_labels.append(label)
            ax.errorbar(x, y, yerr=yerr, color=color, linestyle="none", linewidth=1.0, capsize=2.0, zorder=2)
            ax.scatter([0.0], [0.0], marker=marker, s=30, facecolors="white", edgecolors=color, linewidths=1.2, zorder=5)

        ax.set_ylim(*y_lim)
        ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
        ax.grid(True, color="0.90", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.12, x_broken_log([max(max_pos, max(ticks))], min_pos=min_pos, max_pos=max_pos)[0] + 0.28)
        draw_baseline_projection_marker(ax, rel_summary, min_pos=min_pos, max_pos=max_pos)
        ax.set_xticks(x_broken_log(ticks, min_pos=min_pos, max_pos=max_pos))
        ax.set_xticklabels([str(int(tick)) for tick in ticks], fontsize=8)
        ax.text(
            0.52,
            -0.075,
            "//",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            rotation=-20,
            clip_on=False,
        )
        title = str(relation_row.relation_title)
        static = rel_summary[rel_summary["is_static_baseline"].astype(bool)]
        n_text = ""
        if not static.empty:
            n_text = f"\nmedian n={float(static['n_units_median_per_trace'].iloc[0]):.0f} units/trace"
        ax.set_title(title + n_text, fontsize=10.6)
        if ax_idx == 0:
            ax.set_ylabel("SSI change (%)")
        else:
            ax.tick_params(labelleft=False)

    fig.suptitle("Zero-gap Vernier contour: high-SF orientation splits", fontsize=13.2, y=0.975)
    fig.text(0.5, 0.918, contour_axis_label, ha="center", va="top", fontsize=7.8, color="0.35")
    fig.text(
        0.5,
        0.888,
        "Each curve varies one contour-relative eye-motion component; the orthogonal component is held at the trace mean.",
        ha="center",
        va="top",
        fontsize=7.5,
        color="0.35",
    )
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.828),
            ncols=2,
            columnspacing=1.8,
            handlelength=2.7,
            fontsize=8.2,
        )
    fig.text(0.5, 0.035, "component path length (arcmin; log scale after break)", ha="center", va="bottom", fontsize=9.2)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{output_stem}_no_gap.png"
    pdf = out_dir / f"{output_stem}_no_gap.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def run_panel_c(args: argparse.Namespace, *, preview_png: Path) -> None:
    scales = parse_scale_list(args.panel_c_scales)
    rows = panel_c_condition_rows(scales)
    surface_rows: list[dict[str, Any]] = []
    if bool(args.panel_c_write_component_surfaces) or bool(args.panel_c_write_arcmin_binned_surfaces):
        surface_rows = panel_c_surface_condition_rows(parse_scale_list(args.panel_c_surface_scales))
    conditions = list(dict.fromkeys(str(row["condition"]) for row in [*rows, *surface_rows]))
    require_maps = False
    missing = [
        condition
        for condition in conditions
        if bool(args.force)
        or not cache_has_required_fields(
            stats_cache_path(Path(args.out_dir), condition, float(args.cache_fd_tag_arcmin), int(args.max_frames)),
            require_maps=require_maps,
            expected_identity=_cache_identity(args, condition=condition),
        )
    ]
    trace_set = model = readout = view = None
    device: str | None = None
    if missing:
        print(f"Need to compute {len(missing)} single-contour panel-C caches.", flush=True)
        base_trace_set = subsample_traces(load_eye_traces(Path(args.eye_traces_path)), int(args.n_traces), int(args.seed))
        trace_set = expand_trace_set_for_orientation_blocks(args, base_trace_set)
        print(
            f"Using {int(base_trace_set.traces.shape[0])} eye traces x "
            f"{orientation_block_count(args)} orientation block(s) = {int(trace_set.traces.shape[0])} samples.",
            flush=True,
        )
        device_arg = None if str(args.device).lower() == "auto" else str(args.device)
        print("Loading model/readout...", flush=True)
        model, readout = load_model_and_readout(device=device_arg)
        device = str(next(model.model.parameters()).device)
        print(f"Model device: {device}", flush=True)
        view = load_population_view(version_name=RR100_VERSION)
        print(f"Population view: {view.name}; n_units={int(view.n_units)}", flush=True)

    stats_by_condition = {
        condition: load_or_compute_condition_stats(
            args,
            condition=condition,
            trace_set=trace_set,
            model=model,
            readout=readout,
            view=view,
            device=device,
            require_maps=require_maps,
        )
        for condition in conditions
    }
    n_units = int(next(iter(stats_by_condition.values()))["unit_bits_per_trace"].shape[1])
    unit_indices, selected_units, contour_axis = selected_panel_c_units(args, n_units)
    output_stem = panel_c_output_stem(args)
    summary = summarize_panel_c(
        rows=rows,
        stats_by_condition=stats_by_condition,
        unit_indices=unit_indices,
        n_bootstrap=int(args.panel_c_bootstrap),
        seed=int(args.seed),
    )
    summary_csv = Path(args.out_dir) / f"{output_stem}_summary.csv"
    selected_csv = Path(args.out_dir) / f"{output_stem}_selected_units.csv"
    summary.to_csv(summary_csv, index=False)
    selected_units.to_csv(selected_csv, index=False)
    png, pdf = plot_panel_c(
        summary,
        selected_units=selected_units,
        contour_axis_label=panel_c_contour_axis_label(args),
        out_dir=Path(args.out_dir),
        output_stem=output_stem,
        selection_title=panel_c_selection_title(args),
        dpi=int(args.dpi),
    )
    orientation_split_outputs: dict[str, Any] = {}
    if bool(args.panel_c_write_orientation_splits):
        split_units = load_panel_c_units_table(args, n_units)
        split_summary, split_selection = summarize_panel_c_orientation_splits(
            rows=rows,
            stats_by_condition=stats_by_condition,
            units=split_units,
            n_bootstrap=int(args.panel_c_bootstrap),
            seed=int(args.seed),
            args=args,
        )
        split_output_stem = panel_c_orientation_split_output_stem(args, output_stem)
        split_summary_csv = Path(args.out_dir) / f"{split_output_stem}_summary.csv"
        split_selection_csv = Path(args.out_dir) / f"{split_output_stem}_trace_unit_selection.csv"
        split_summary.to_csv(split_summary_csv, index=False)
        split_selection.to_csv(split_selection_csv, index=False)
        split_png, split_pdf = plot_panel_c_orientation_splits(
            split_summary,
            contour_axis_label=panel_c_contour_axis_label(args),
            out_dir=Path(args.out_dir),
            output_stem=split_output_stem,
            dpi=int(args.dpi),
        )
        orientation_split_outputs = {
            "enabled": True,
            "output_stem": split_output_stem,
            "summary_csv": split_summary_csv,
            "trace_unit_selection_csv": split_selection_csv,
            "png": split_png,
            "pdf": split_pdf,
            "relations": [
                {
                    "relation": spec["relation"],
                    "relation_label": spec["relation_label"],
                    "relation_title": spec["relation_title"],
                    "relation_rank": spec["relation_rank"],
                }
                for spec in panel_c_orientation_relation_specs(args)
            ],
            "definition": (
                "per-trace high-SF orientation-tuned unit pools relative to that trace's synthetic contour axis"
            ),
            "match_max_deg": float(args.panel_c_match_max_deg),
            "orthogonal_min_deg": float(args.panel_c_orthogonal_min_deg),
            "min_osi": float(args.panel_c_min_osi),
        }
    else:
        orientation_split_outputs = {"enabled": False}
    component_surface_outputs: dict[str, Any] = {}
    if bool(args.panel_c_write_component_surfaces):
        surface_units = load_panel_c_units_table(args, n_units)
        surface_summary = summarize_panel_c_component_surfaces(
            rows=surface_rows,
            stats_by_condition=stats_by_condition,
            units=surface_units,
            args=args,
        )
        surface_stem = surface_output_stem(args, output_stem)
        surface_summary_csv = Path(args.out_dir) / f"{surface_stem}_summary.csv"
        surface_summary.to_csv(surface_summary_csv, index=False)
        surface_pdf, surface_pngs = plot_panel_c_component_surfaces(
            surface_summary,
            contour_axis_label=panel_c_contour_axis_label(args),
            out_dir=Path(args.out_dir),
            output_stem=surface_stem,
            dpi=int(args.dpi),
        )
        component_surface_outputs = {
            "enabled": True,
            "output_stem": surface_stem,
            "summary_csv": surface_summary_csv,
            "pdf": surface_pdf,
            "pngs": surface_pngs,
            "surface_scales": parse_scale_list(args.panel_c_surface_scales),
            "surface_grid_size": [
                len(sorted({float(row["across_scale"]) for row in surface_rows})),
                len(sorted({float(row["along_scale"]) for row in surface_rows})),
            ],
            "relations": [
                {
                    "relation": spec["relation"],
                    "relation_label": spec["relation_label"],
                    "relation_title": spec["relation_title"],
                    "relation_rank": spec["relation_rank"],
                }
                for spec in panel_c_orientation_relation_specs(args)
            ],
            "metric_families": PANEL_C_SURFACE_FAMILIES,
            "definition": (
                "controlled 2D across x along scaling grid over the same selected real eye traces; "
                "per-trace high-SF unit pools are selected relative to that trace's synthetic contour axis"
            ),
        }
    else:
        component_surface_outputs = {"enabled": False}
    arcmin_binned_surface_outputs: dict[str, Any] = {}
    if bool(args.panel_c_write_arcmin_binned_surfaces):
        arcmin_units = load_panel_c_units_table(args, n_units)
        arcmin_summary = summarize_panel_c_arcmin_binned_surfaces(
            rows=surface_rows,
            stats_by_condition=stats_by_condition,
            units=arcmin_units,
            args=args,
        )
        arcmin_stem = arcmin_binned_surface_output_stem(args, output_stem)
        arcmin_summary_csv = Path(args.out_dir) / f"{arcmin_stem}_summary.csv"
        arcmin_summary.to_csv(arcmin_summary_csv, index=False)
        arcmin_pdf, arcmin_pngs = plot_panel_c_arcmin_binned_surfaces(
            arcmin_summary,
            contour_axis_label=panel_c_contour_axis_label(args),
            out_dir=Path(args.out_dir),
            output_stem=arcmin_stem,
            dpi=int(args.dpi),
        )
        arcmin_binned_surface_outputs = {
            "enabled": True,
            "output_stem": arcmin_stem,
            "summary_csv": arcmin_summary_csv,
            "pdf": arcmin_pdf,
            "pngs": arcmin_pngs,
            "arcmin_bin_count": int(args.panel_c_arcmin_surface_bins),
            "surface_scales": parse_scale_list(args.panel_c_surface_scales),
            "relations": [
                {
                    "relation": spec["relation"],
                    "relation_label": spec["relation_label"],
                    "relation_title": spec["relation_title"],
                    "relation_rank": spec["relation_rank"],
                }
                for spec in panel_c_orientation_relation_specs(args)
            ],
            "metric_families": PANEL_C_SURFACE_FAMILIES,
            "normalization": {
                "population_ssi_bits_per_spike": "sum(unit_bits_per_spike * unit_mean_rate) / sum(unit_mean_rate)",
                "information_bits_per_sample": "sum(unit_bits_per_spike * unit_mean_rate) / valid_unit_trace_samples",
                "expected_spikes_per_sample": "sum(unit_mean_rate) / valid_unit_trace_samples",
                "sample_count": "valid unit-trace contributions, matching the BackImage selected-samples convention",
            },
            "definition": (
                "generated moving trace-condition samples from the controlled scale grid are binned by measured "
                "contour-relative component arcmin before spike-weighted population accumulation"
            ),
        }
    else:
        arcmin_binned_surface_outputs = {"enabled": False}
    manifest_path = Path(args.out_dir) / f"{output_stem}_manifest.json"
    write_json(
        manifest_path,
        {
            "analysis": "rr100_single_contour_panel_c",
            "out_dir": Path(args.out_dir),
            "output_stem": output_stem,
            "conditions": conditions,
            "panel_c_scales": scales,
            "panel_c_unit_selection": effective_panel_c_unit_selection(args),
            "requested_panel_c_unit_selection": str(args.panel_c_unit_selection),
            "n_traces": int(args.n_traces),
            "n_eye_traces_per_orientation": int(args.n_traces),
            "n_orientation_blocks": orientation_block_count(args),
            "n_effective_trace_orientation_samples": expected_trace_count(args),
            "max_frames": int(args.max_frames),
            "stimulus_normalization": STIMULUS_NORMALIZATION,
            "single_contour_spec": asdict(canonical_single_contour_spec(args)),
            "stimulus_contract": "zero-offset zero-gap Vernier halves form one continuous contour; no acuity task",
            "stimulus_orientation_mode": str(args.orientation_mode),
            "stimulus_orientation_deg_per_trace": stimulus_orientations_for_traces(args, expected_trace_count(args)),
            "contour_axis_deg_per_trace": contour_axis_from_orientation_deg(
                stimulus_orientations_for_traces(args, expected_trace_count(args))
            ),
            "random_orientation_range_deg": [
                float(args.random_orientation_min_deg),
                float(args.random_orientation_max_deg),
            ]
            if has_random_trace_orientations(args)
            else [],
            "motion_axis_convention": "contour_relative_axes_rotated_with_stimulus",
            "movement_families": {
                "across": "real_aniso_across_scale_along_0",
                "along": "real_aniso_across_0_along_scale",
            },
            "component_surfaces": component_surface_outputs,
            "arcmin_binned_component_surfaces": arcmin_binned_surface_outputs,
            "baseline_projection_path_median_arcmin": baseline_projection_arcmin(summary),
            "panel_c_sf_group": str(args.panel_c_sf_group),
            "panel_c_min_osi": float(args.panel_c_min_osi),
            "panel_c_match_max_deg": float(args.panel_c_match_max_deg),
            "panel_c_orthogonal_min_deg": float(args.panel_c_orthogonal_min_deg),
            "panel_c_contour_axis_deg": None if has_random_trace_orientations(args) else float(contour_axis),
            "panel_c_contour_axis_label": panel_c_contour_axis_label(args),
            "n_selected_units": int(selected_units.shape[0]),
            "selected_unit_indices": selected_units["unit_index"].astype(int).tolist(),
            "summary_csv": summary_csv,
            "selected_units_csv": selected_csv,
            "orientation_relation_split": orientation_split_outputs,
            "stimulus_preview_png": preview_png,
            "png": png,
            "pdf": pdf,
            "actual_device": device,
            "cache_prefix": CACHE_PREFIX,
            "cache_fd_tag_arcmin": float(args.cache_fd_tag_arcmin),
        },
    )
    print(f"Wrote single-contour panel-C summary: {summary_csv}", flush=True)
    print(f"Wrote selected-unit table: {selected_csv}", flush=True)
    print(f"Wrote single-contour panel-C PNG: {png}", flush=True)
    if orientation_split_outputs.get("enabled"):
        print(
            f"Wrote orientation-split panel-C PNG: {orientation_split_outputs['png']}",
            flush=True,
        )
    if component_surface_outputs.get("enabled"):
        print(f"Wrote component-surface summary: {component_surface_outputs['summary_csv']}", flush=True)
        print(f"Wrote component-surface PDF: {component_surface_outputs['pdf']}", flush=True)
        for surface_png in component_surface_outputs["pngs"]:
            print(f"Wrote component-surface PNG: {surface_png}", flush=True)
    if arcmin_binned_surface_outputs.get("enabled"):
        print(f"Wrote arcmin-binned surface summary: {arcmin_binned_surface_outputs['summary_csv']}", flush=True)
        print(f"Wrote arcmin-binned surface PDF: {arcmin_binned_surface_outputs['pdf']}", flush=True)
        for arcmin_png in arcmin_binned_surface_outputs["pngs"]:
            print(f"Wrote arcmin-binned surface PNG: {arcmin_png}", flush=True)
    print(f"Wrote single-contour panel-C PDF: {pdf}", flush=True)
    print(f"Wrote manifest: {manifest_path}", flush=True)


def main() -> None:
    args = parse_args()
    validate_orientation_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    across_scales = parse_scale_list(args.across_scales)
    rows = condition_sequence(across_scales, float(args.along_scale))
    conditions = [str(row["condition"]) for row in rows]
    require_maps = not bool(args.skip_highlighted_unit_maps)
    preview_png = Path(args.out_dir) / "rr100_single_contour_stimulus_preview.png"
    draw_contour_preview(canonical_single_contour_spec(args), preview_png, int(args.dpi))
    if bool(args.preview_only):
        print(f"Wrote single-contour stimulus preview: {preview_png}", flush=True)
        return
    if str(args.condition_set) == "panel_c":
        run_panel_c(args, preview_png=preview_png)
        return

    missing = [
        condition
        for condition in conditions
        if bool(args.force)
        or not cache_has_required_fields(
            stats_cache_path(Path(args.out_dir), condition, float(args.cache_fd_tag_arcmin), int(args.max_frames)),
            require_maps=require_maps,
            expected_identity=_cache_identity(args, condition=condition),
        )
    ]
    trace_set = model = readout = view = None
    device: str | None = None
    if missing:
        print(f"Need to compute {len(missing)} single-contour unit SSI caches.", flush=True)
        trace_set = subsample_traces(load_eye_traces(Path(args.eye_traces_path)), int(args.n_traces), int(args.seed))
        device_arg = None if str(args.device).lower() == "auto" else str(args.device)
        print("Loading model/readout...", flush=True)
        model, readout = load_model_and_readout(device=device_arg)
        device = str(next(model.model.parameters()).device)
        print(f"Model device: {device}", flush=True)
        view = load_population_view(version_name=RR100_VERSION)
        print(f"Population view: {view.name}; n_units={int(view.n_units)}", flush=True)

    stats_by_condition = {
        condition: load_or_compute_condition_stats(
            args,
            condition=condition,
            trace_set=trace_set,
            model=model,
            readout=readout,
            view=view,
            device=device,
            require_maps=require_maps,
        )
        for condition in conditions
    }

    unit_df, top_df, diagnostics = summarize_units(stats_by_condition, rows)
    unit_csv = Path(args.out_dir) / "rr100_single_contour_movement_ssi_unit_table.csv"
    top_csv = Path(args.out_dir) / "rr100_single_contour_movement_ssi_top_units.csv"
    unit_df.to_csv(unit_csv, index=False)
    top_df.to_csv(top_csv, index=False)

    top_n = max(1, int(args.top_units))
    top_by_unit = (
        unit_df[["unit_index", "unit_max_abs_log2_ssi_vs_static_along0"]]
        .drop_duplicates()
        .sort_values("unit_max_abs_log2_ssi_vs_static_along0", ascending=False, kind="mergesort")
        .head(top_n)["unit_index"]
        .astype(int)
        .tolist()
    )
    top_by_influence = top_df.head(top_n)["unit_index"].astype(int).tolist()
    top_by_unit_for_plot = order_units_by_y_at_x(diagnostics, top_by_unit, x_value=1.0)
    top_by_influence_for_plot = order_units_by_y_at_x(diagnostics, top_by_influence, x_value=1.0)

    figure_title = "RR100 single-contour movement SSI along the along=0 scale line"
    unit_lines_png = Path(args.out_dir) / "rr100_single_contour_movement_ssi_unit_lines.png"
    influence_unit_lines_png = Path(args.out_dir) / "rr100_single_contour_movement_ssi_lines_top_influence.png"
    influence_unit_lines_with_maps_png = (
        Path(args.out_dir) / "rr100_single_contour_movement_ssi_lines_top_influence_with_activation_rows.png"
    )
    loo_png = Path(args.out_dir) / "rr100_single_contour_movement_ssi_leave_one_out.png"

    draw_unit_lines(
        diagnostics,
        top_by_unit_for_plot,
        unit_lines_png,
        int(args.dpi),
        highlight_note="largest unit SSI changes highlighted",
        figure_title=figure_title,
    )
    draw_unit_lines(
        diagnostics,
        top_by_influence_for_plot,
        influence_unit_lines_png,
        int(args.dpi),
        highlight_note="largest leave-one-out influences highlighted",
        figure_title=figure_title,
    )
    if not bool(args.skip_highlighted_unit_maps):
        draw_unit_lines_with_activation_rows(
            diagnostics=diagnostics,
            highlighted_units=top_by_influence_for_plot,
            rows=rows,
            stats_by_condition=stats_by_condition,
            path=influence_unit_lines_with_maps_png,
            dpi=int(args.dpi),
            highlight_note="largest leave-one-out influences highlighted",
            map_vmin_percentile=float(args.map_vmin_percentile),
            map_vmax_percentile=float(args.map_vmax_percentile),
            figure_title=figure_title,
            figure_subtitle=(
                "activation maps are trace-time mean responses to one fixed contour; "
                "SSI is spatial response-map information, not offset discrimination"
            ),
        )
    draw_leave_one_out(
        diagnostics,
        top_by_influence,
        top_df,
        loo_png,
        int(args.dpi),
        figure_title="Does any single RR100 unit drive the single-contour SSI movement line?",
    )

    npz_path = Path(args.out_dir) / "rr100_single_contour_movement_ssi_diagnostics_arrays.npz"
    np.savez_compressed(
        npz_path,
        across_values=np.asarray(diagnostics["across_values"], dtype=np.float32),
        population_ratio=np.asarray(diagnostics["population_ratio"], dtype=np.float32),
        unit_ratio=np.asarray(diagnostics["unit_ratio"], dtype=np.float32),
        unit_log2_ratio=np.asarray(diagnostics["unit_log2_ratio"], dtype=np.float32),
        leave_one_out_ratio=np.asarray(diagnostics["leave_one_out_ratio"], dtype=np.float32),
        leave_one_out_delta=np.asarray(diagnostics["leave_one_out_delta"], dtype=np.float32),
        stimulus_normalization=np.asarray([STIMULUS_NORMALIZATION]),
    )
    manifest_path = Path(args.out_dir) / "rr100_single_contour_movement_ssi_manifest.json"
    write_json(
        manifest_path,
        {
            "analysis": "rr100_single_contour_movement_ssi",
            "out_dir": Path(args.out_dir),
            "conditions": conditions,
            "across_scales": across_scales,
            "along_scale": float(args.along_scale),
            "n_traces": int(args.n_traces),
            "max_frames": int(args.max_frames),
            "stimulus_normalization": STIMULUS_NORMALIZATION,
            "single_contour_spec": asdict(canonical_single_contour_spec(args)),
            "stimulus_orientation_mode": str(args.orientation_mode),
            "stimulus_orientation_deg_per_trace": stimulus_orientations_for_traces(args, int(args.n_traces)),
            "contour_axis_deg_per_trace": contour_axis_from_orientation_deg(
                stimulus_orientations_for_traces(args, int(args.n_traces))
            ),
            "random_orientation_range_deg": [
                float(args.random_orientation_min_deg),
                float(args.random_orientation_max_deg),
            ]
            if has_random_trace_orientations(args)
            else [],
            "seed": int(args.seed),
            "device_arg": str(args.device),
            "actual_device": device,
            "batch_size": int(args.batch_size),
            "population_version": RR100_VERSION,
            "trajectory_contract": "native real trace scaled around trial mean in contour-relative axes",
            "motion_axis_convention": "contour_relative_axes_rotated_with_stimulus",
            "readout_time_contract": "all response frames averaged for SSI",
            "ssi_contract": "spatial SSI of fixed-contour response maps; no plus/minus acuity task",
            "cache_prefix": CACHE_PREFIX,
            "cache_fd_tag_arcmin": float(args.cache_fd_tag_arcmin),
            "unit_table_csv": unit_csv,
            "top_units_csv": top_csv,
            "stimulus_preview_png": preview_png,
            "unit_lines_png": unit_lines_png,
            "influence_unit_lines_png": influence_unit_lines_png,
            "influence_unit_lines_with_activation_rows_png": influence_unit_lines_with_maps_png
            if not bool(args.skip_highlighted_unit_maps)
            else None,
            "leave_one_out_png": loo_png,
            "arrays_npz": npz_path,
            "top_units_by_unit_log2_deviation": top_by_unit,
            "top_units_by_leave_one_out_influence": top_by_influence,
            "top_units_by_unit_log2_deviation_plot_order_at_across1": top_by_unit_for_plot,
            "top_units_by_leave_one_out_influence_plot_order_at_across1": top_by_influence_for_plot,
        },
    )

    print(f"Wrote single-contour unit SSI table: {unit_csv}", flush=True)
    print(f"Wrote single-contour top-unit table: {top_csv}", flush=True)
    print(f"Wrote single-contour stimulus preview: {preview_png}", flush=True)
    print(f"Wrote single-contour unit line plot: {unit_lines_png}", flush=True)
    print(f"Wrote single-contour influence plot: {influence_unit_lines_png}", flush=True)
    if not bool(args.skip_highlighted_unit_maps):
        print(f"Wrote single-contour influence plot with activation rows: {influence_unit_lines_with_maps_png}", flush=True)
    print(f"Wrote single-contour leave-one-out plot: {loo_png}", flush=True)
    print(f"Wrote manifest: {manifest_path}", flush=True)
    print(
        top_df.head(top_n)[
            [
                "unit_index",
                "max_abs_leave_one_out_population_ratio_delta",
                "max_abs_log2_unit_ssi_vs_static",
                "static_unit_ssi_bits_per_spike_mean",
                "static_unit_mean_rate_mean",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6g}"),
        flush=True,
    )


if __name__ == "__main__":
    main()
