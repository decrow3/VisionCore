#!/usr/bin/env python3
"""Probe RR100 unit spatial/temporal-frequency tuning with drifting gratings."""

from __future__ import annotations

import argparse
import csv
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
import matplotlib.ticker
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.plot_backimage_rr100_instantaneous_unit_maps import (  # noqa: E402
    EPS,
    DEFAULT_OUT_DIR as DEFAULT_BACKIMAGE_UNIT_MAP_DIR,
    DEFAULT_SCALES,
    RR100_MOVIE_MEDOID_VERSION,
    STIMULUS_NORMALIZATION,
    angle_180_distance,
    image_scale,
    orientation_axis_180,
    parse_float_list,
    parse_int_list,
    safe_slug,
    write_csv,
    write_json,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (  # noqa: E402
    CanonicalTwinScorer,
)
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view  # noqa: E402


DEFAULT_OUT_DIR = ROOT / "outputs/active_sensing_movie_information/backimage_rr100_frequency_tuning_probe_v1"


def embed_time_lags_local(movie: torch.Tensor, n_lags: int = 32) -> torch.Tensor:
    """Embed a movie as (T_valid, 1, n_lags, H, W), matching scripts.spatial_info."""
    if movie.dim() == 3:
        movie = movie.unsqueeze(1)
    total, channels, height, width = movie.shape
    out_frames = int(total) - int(n_lags) + 1
    if out_frames <= 0:
        raise ValueError(f"movie has {total} frames but n_lags={n_lags}")
    lagged = torch.zeros(out_frames, channels, int(n_lags), height, width, dtype=movie.dtype, device=movie.device)
    for lag in range(int(n_lags)):
        lagged[:, :, lag] = movie[int(n_lags) - 1 - lag : int(total) - lag]
    return lagged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_BACKIMAGE_UNIT_MAP_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--units",
        type=str,
        default="",
        help="Comma-separated RR100 units to highlight in the figure; default uses source selected_units.csv.",
    )
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument(
        "--orientation-deg",
        type=str,
        default="0,45,90,135",
        help="Comma-separated shared grating orientation axes in degrees. All RR100 units are read out for every stimulus.",
    )
    parser.add_argument("--spatial-cpds", type=str, default="0.0125,0.05,0.2,0.8,3.2,12.8")
    parser.add_argument("--temporal-hz", type=str, default="0,0.2,0.8,3.2,12.8,47.2")
    parser.add_argument(
        "--scalar-readout",
        choices=("center_pixel", "spatial_mean"),
        default="center_pixel",
        help=(
            "Collapse each post-activation RR100 rate map to a scalar. center_pixel is the cell-like "
            "single-location readout used in RR100 QC; spatial_mean is retained only as a diagnostic."
        ),
    )
    parser.add_argument("--n-phases", type=int, default=2)
    parser.add_argument(
        "--static-n-phases",
        type=int,
        default=4,
        help="Number of seeded random starting phases for the TF=0 static baseline.",
    )
    parser.add_argument(
        "--phase-seed",
        type=int,
        default=17,
        help="Seed for the shuffled static starting phases.",
    )
    parser.add_argument("--duration-s", type=float, default=1.5)
    parser.add_argument("--frame-rate-hz", type=float, default=120.0)
    parser.add_argument("--n-lags", type=int, default=32)
    parser.add_argument("--discard-frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=101)
    parser.add_argument("--ppd", type=float, default=37.50476617)
    parser.add_argument("--contrast", type=float, default=0.8)
    parser.add_argument("--window-sigma-frac", type=float, default=0.28)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def identity_text(identity: dict[str, Any]) -> str:
    return json.dumps(json_ready(identity), sort_keys=True, separators=(",", ":"))


def stimulus_sampling_summary(*, ppd: float, image_size: int, frame_rate_hz: float) -> dict[str, float]:
    fov_deg = float(image_size) / float(ppd)
    return {
        "ppd": float(ppd),
        "image_size_px": int(image_size),
        "fov_deg": float(fov_deg),
        "spatial_nyquist_cpd": float(0.5 * float(ppd)),
        "one_cycle_across_window_cpd": float(1.0 / max(fov_deg, EPS)),
        "frame_rate_hz": float(frame_rate_hz),
        "temporal_nyquist_hz": float(0.5 * float(frame_rate_hz)),
    }


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_source_summary_and_plot_units(source_dir: Path, requested_units: list[int]) -> tuple[list[int], pd.DataFrame]:
    selected_path = Path(source_dir) / "selected_units.csv"
    summary_path = Path(source_dir) / "orientation_probe_unit_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing orientation summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    if requested_units:
        units = [int(v) for v in requested_units]
    else:
        if not selected_path.exists():
            raise FileNotFoundError(f"Missing selected units and --units was empty: {selected_path}")
        selected = pd.read_csv(selected_path)
        units = [int(v) for v in selected["unit_index"].to_list()]
    available = set(summary["unit_index"].astype(int).to_list())
    missing = [unit for unit in units if int(unit) not in available]
    if missing:
        raise ValueError(f"Requested units absent from orientation summary: {missing}")
    return units, summary


def scalar_readout_traces(rate_maps: np.ndarray, mode: str) -> tuple[np.ndarray, int | None, int | None]:
    """Return T x N scalar traces from post-activation RR100 rate maps."""
    maps = np.asarray(rate_maps, dtype=np.float32)
    if maps.ndim != 4:
        raise ValueError(f"Expected rate maps with shape (T, N, H, W), got {maps.shape}")
    if str(mode) == "center_pixel":
        center_y = int(maps.shape[-2] // 2)
        center_x = int(maps.shape[-1] // 2)
        return maps[:, :, center_y, center_x], center_y, center_x
    if str(mode) == "spatial_mean":
        return maps.mean(axis=(-2, -1)), None, None
    raise ValueError(f"Unknown scalar readout mode {mode!r}")


def make_windowed_drifting_grating_movie(
    *,
    image_size: int,
    orientation_deg: float,
    spatial_cpd: float,
    temporal_hz: float,
    phase_rad: float,
    n_valid_frames: int,
    n_lags: int,
    frame_rate_hz: float,
    ppd: float,
    contrast: float,
    window_sigma_frac: float,
) -> np.ndarray:
    size = int(image_size)
    total_frames = int(n_valid_frames) + int(n_lags) - 1
    frame_idx = np.arange(total_frames, dtype=np.float64) - (int(n_lags) - 1)
    t = frame_idx / float(frame_rate_hz)
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    x = (xx - 0.5 * (size - 1)) / float(ppd)
    y = (yy - 0.5 * (size - 1)) / float(ppd)
    theta = math.radians(float(orientation_deg))
    normal_coord = -math.sin(theta) * x + math.cos(theta) * y
    sigma_px = max(float(window_sigma_frac) * float(size), 1.0)
    r2 = (xx - 0.5 * (size - 1)) ** 2 + (yy - 0.5 * (size - 1)) ** 2
    window = np.exp(-0.5 * r2 / (sigma_px * sigma_px))
    carrier = np.sin(
        2.0 * math.pi * float(spatial_cpd) * normal_coord[None, :, :]
        - 2.0 * math.pi * float(temporal_hz) * t[:, None, None]
        + float(phase_rad)
    )
    movie = 127.5 + 127.5 * float(contrast) * carrier * window[None, :, :]
    return np.clip(movie, 0.0, 255.0).astype(np.float32)


def compute_rr100_movie_maps(
    scorer: CanonicalTwinScorer,
    view: Any,
    movie_uint: np.ndarray,
    *,
    n_lags: int,
) -> np.ndarray:
    movie = (np.asarray(movie_uint, dtype=np.float32) - 127.0) / 255.0
    stim = embed_time_lags_local(torch.from_numpy(movie), n_lags=int(n_lags))
    full_map = scorer._compute_rate_map_batched(stim)
    full_np = full_map.detach().cpu().numpy().astype(np.float32, copy=False)
    rr100 = apply_population_view(full_np, view).astype(np.float32, copy=False)
    del stim, full_map, full_np
    if str(scorer.device).startswith("cuda") and scorer.torch.cuda.is_available():
        scorer.torch.cuda.empty_cache()
    return rr100


def sinusoid_amplitude(values: np.ndarray, *, temporal_hz: float, frame_rate_hz: float, discard_frames: int) -> dict[str, float]:
    y = np.asarray(values, dtype=np.float64)
    if float(temporal_hz) <= 0.0 or y.size <= int(discard_frames) + 4:
        return {
            "n_analysis_frames": max(0, int(y.size) - int(discard_frames)),
            "response_amp": float("nan"),
            "response_amp_sq": float("nan"),
        }
    y = y[int(discard_frames) :]
    t = (np.arange(y.size, dtype=np.float64) + int(discard_frames)) / float(frame_rate_hz)
    omega = 2.0 * math.pi * float(temporal_hz) * t
    design = np.column_stack([np.sin(omega), np.cos(omega), np.ones_like(omega)])
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    amp = float(math.sqrt(float(coeff[0] ** 2 + coeff[1] ** 2)))
    return {
        "n_analysis_frames": int(y.size),
        "response_amp": amp,
        "response_amp_sq": amp * amp,
    }


def phase_schedule_for_temporal_grid(
    temporal_hz: list[float],
    *,
    n_dynamic_phases: int,
    n_static_phases: int,
    phase_seed: int,
) -> dict[float, list[tuple[int, float, str]]]:
    """Build deterministic dynamic phases and shuffled static phases for TF=0."""
    n_dynamic = max(int(n_dynamic_phases), 1)
    dynamic_phases = [
        (int(idx), float(2.0 * math.pi * idx / n_dynamic), "dynamic_uniform_grid")
        for idx in range(n_dynamic)
    ]
    n_static = max(int(n_static_phases), 1)
    rng = np.random.default_rng(int(phase_seed))
    static_values = rng.uniform(0.0, 2.0 * math.pi, size=n_static)
    static_phases = [
        (int(idx), float(phase), "static_seeded_uniform_random")
        for idx, phase in enumerate(static_values)
    ]
    schedule: dict[float, list[tuple[int, float, str]]] = {}
    for tf in temporal_hz:
        tf_value = float(tf)
        schedule[tf_value] = static_phases if np.isclose(tf_value, 0.0) else dynamic_phases
    return schedule


def compute_probe_rows(args: argparse.Namespace, orientation_summary: pd.DataFrame) -> list[dict[str, Any]]:
    orientation_degrees = [orientation_axis_180(v) for v in parse_float_list(str(args.orientation_deg))]
    spatial_cpds = parse_float_list(str(args.spatial_cpds))
    temporal_hz = parse_float_list(str(args.temporal_hz))
    phase_schedule = phase_schedule_for_temporal_grid(
        temporal_hz,
        n_dynamic_phases=int(args.n_phases),
        n_static_phases=int(args.static_n_phases),
        phase_seed=int(args.phase_seed),
    )
    n_valid_frames = max(int(round(float(args.duration_s) * float(args.frame_rate_hz))), int(args.n_lags) + 8)
    discard_frames = min(int(args.discard_frames), max(n_valid_frames - 8, 0))
    view = load_population_view(version_name=str(args.rr100_version))
    scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(args.batch_size), empty_cache_every_batch=True)

    summary_by_unit = {int(row["unit_index"]): row for _, row in orientation_summary.iterrows()}
    rr100_units = list(range(int(view.n_units)))
    rows: list[dict[str, Any]] = []
    total_phases_per_sf = sum(len(phase_schedule[float(tf)]) for tf in temporal_hz)
    total = len(orientation_degrees) * len(spatial_cpds) * total_phases_per_sf
    done = 0
    for orientation_deg in orientation_degrees:
        for sf in spatial_cpds:
            for tf in temporal_hz:
                for phase_idx, phase, phase_policy in phase_schedule[float(tf)]:
                    movie = make_windowed_drifting_grating_movie(
                        image_size=int(args.image_size),
                        orientation_deg=float(orientation_deg),
                        spatial_cpd=float(sf),
                        temporal_hz=float(tf),
                        phase_rad=float(phase),
                        n_valid_frames=n_valid_frames,
                        n_lags=int(args.n_lags),
                        frame_rate_hz=float(args.frame_rate_hz),
                        ppd=float(args.ppd),
                        contrast=float(args.contrast),
                        window_sigma_frac=float(args.window_sigma_frac),
                    )
                    rr100 = compute_rr100_movie_maps(scorer, view, movie, n_lags=int(args.n_lags))
                    scalar_all, center_y, center_x = scalar_readout_traces(rr100, str(args.scalar_readout))
                    scalar_all = np.asarray(scalar_all, dtype=np.float64)
                    analysis = scalar_all[int(discard_frames) :]
                    mean_rate = np.mean(analysis, axis=0)
                    peak_rate = np.max(analysis, axis=0)
                    rate_std = np.std(analysis, axis=0)
                    done += 1
                    for unit in rr100_units:
                        prior_row = summary_by_unit.get(int(unit))
                        if prior_row is None:
                            prior_pref = float("nan")
                            prior_osi = float("nan")
                        else:
                            prior_pref = float(prior_row.get("preferred_orientation_deg", float("nan")))
                            prior_osi = float(prior_row.get("orientation_selectivity_index", float("nan")))
                        scalar = scalar_all[:, int(unit)]
                        amp = sinusoid_amplitude(
                            scalar,
                            temporal_hz=float(tf),
                            frame_rate_hz=float(args.frame_rate_hz),
                            discard_frames=discard_frames,
                        )
                        row = {
                            "unit_index": int(unit),
                            "unit_label": f"u{int(unit):03d}",
                            "probe_orientation_deg": float(orientation_deg),
                            "prior_preferred_orientation_deg": prior_pref,
                            "prior_orientation_selectivity_index": prior_osi,
                            "spatial_cpd": float(sf),
                            "temporal_hz": float(tf),
                            "phase_index": int(phase_idx),
                            "phase_rad": float(phase),
                            "phase_policy": str(phase_policy),
                            "n_valid_frames": int(n_valid_frames),
                            "discard_frames": int(discard_frames),
                            "frame_rate_hz": float(args.frame_rate_hz),
                            "image_size_px": int(args.image_size),
                            "ppd": float(args.ppd),
                            "contrast": float(args.contrast),
                            "window_sigma_frac": float(args.window_sigma_frac),
                            "scalar_readout": str(args.scalar_readout),
                            "center_y": center_y,
                            "center_x": center_x,
                            "mean_rate": float(mean_rate[int(unit)]),
                            "peak_rate": float(peak_rate[int(unit)]),
                            "rate_std": float(rate_std[int(unit)]),
                            "response_amp_per_contrast": float(amp["response_amp"] / max(float(args.contrast), EPS))
                            if np.isfinite(float(amp["response_amp"]))
                            else float("nan"),
                            "response_amp_sq_per_contrast_sq": float(amp["response_amp_sq"] / max(float(args.contrast) ** 2, EPS))
                            if np.isfinite(float(amp["response_amp_sq"]))
                            else float("nan"),
                            "probe_contract": (
                                "shared orientation/SF/TF windowed drifting gratings; scalar response is "
                                "center pixel of post-activation RR100 rate map"
                                if str(args.scalar_readout) == "center_pixel"
                                else "shared orientation/SF/TF windowed drifting gratings; scalar response is RR100 spatial-mean rate"
                            ),
                            **amp,
                        }
                        rows.append(row)
                    watched = [17, 18, 26]
                    watched_text = ", ".join(
                        f"u{unit:03d} mean={float(mean_rate[unit]):.4g}"
                        for unit in watched
                        if unit < mean_rate.shape[0]
                    )
                    print(
                        f"[{done}/{total}] ori={float(orientation_deg):g} deg sf={float(sf):g} cpd "
                        f"tf={float(tf):g} Hz phase={phase_idx}; {watched_text}",
                        flush=True,
                    )
                    del rr100, movie, scalar_all, analysis
    return rows


def aggregate_probe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return [], []
    numeric_cols = [
        "unit_index",
        "probe_orientation_deg",
        "prior_preferred_orientation_deg",
        "prior_orientation_selectivity_index",
        "spatial_cpd",
        "temporal_hz",
        "contrast",
        "mean_rate",
        "response_amp_sq",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped_rows: list[dict[str, Any]] = []
    keys = ["unit_index", "unit_label", "probe_orientation_deg", "spatial_cpd", "temporal_hz", "scalar_readout"]
    for key_values, sub in df.groupby(keys, sort=True):
        rec = dict(zip(keys, key_values, strict=True))
        amp_sq = sub["response_amp_sq"].to_numpy(dtype=float)
        amp = np.sqrt(np.nanmean(amp_sq)) if np.isfinite(amp_sq).any() else float("nan")
        rec.update(
            {
                "n_phases": int(sub.shape[0]),
                "phase_policies": ",".join(sorted(set(str(v) for v in sub["phase_policy"].to_list())))
                if "phase_policy" in sub
                else "",
                "prior_preferred_orientation_deg": float(np.nanmean(sub["prior_preferred_orientation_deg"].to_numpy(dtype=float))),
                "prior_orientation_selectivity_index": float(
                    np.nanmean(sub["prior_orientation_selectivity_index"].to_numpy(dtype=float))
                ),
                "mean_rate": float(np.nanmean(sub["mean_rate"].to_numpy(dtype=float))),
                "sem_mean_rate": float(np.nanstd(sub["mean_rate"].to_numpy(dtype=float), ddof=1) / math.sqrt(sub.shape[0]))
                if sub.shape[0] > 1
                else 0.0,
                "response_amp_rms": float(amp),
                "response_amp_rms_per_contrast": float(amp / max(float(sub["contrast"].iloc[0]), EPS))
                if np.isfinite(amp)
                else float("nan"),
                "response_amp_sq_mean": float(np.nanmean(amp_sq)) if np.isfinite(amp_sq).any() else float("nan"),
                "probe_contract": str(sub["probe_contract"].iloc[0]),
            }
        )
        grouped_rows.append(rec)

    summary_rows: list[dict[str, Any]] = []
    grouped = pd.DataFrame(grouped_rows)
    for col in [
        "unit_index",
        "probe_orientation_deg",
        "prior_preferred_orientation_deg",
        "prior_orientation_selectivity_index",
        "spatial_cpd",
        "temporal_hz",
        "mean_rate",
        "response_amp_rms",
    ]:
        if col in grouped.columns:
            grouped[col] = pd.to_numeric(grouped[col], errors="coerce")
    for unit, sub in grouped.groupby("unit_index", sort=True):
        static = sub[np.isclose(sub["temporal_hz"].astype(float), 0.0)].copy()
        dynamic = sub[~np.isclose(sub["temporal_hz"].astype(float), 0.0)].copy()
        if not static.empty:
            static_best = static.sort_values("mean_rate", ascending=False).iloc[0]
            static_peak_sf = float(static_best["spatial_cpd"])
            static_peak_ori = float(static_best["probe_orientation_deg"])
            static_peak_rate = float(static_best["mean_rate"])
        else:
            static_peak_sf = float("nan")
            static_peak_ori = float("nan")
            static_peak_rate = float("nan")
        if not dynamic.empty:
            dyn_best = dynamic.sort_values("response_amp_rms", ascending=False).iloc[0]
            dyn_peak_ori = float(dyn_best["probe_orientation_deg"])
            dyn_peak_sf = float(dyn_best["spatial_cpd"])
            dyn_peak_tf = float(dyn_best["temporal_hz"])
            dyn_peak_amp = float(dyn_best["response_amp_rms"])
        else:
            dyn_peak_ori = dyn_peak_sf = dyn_peak_tf = dyn_peak_amp = float("nan")
        summary_rows.append(
            {
                "unit_index": int(unit),
                "unit_label": f"u{int(unit):03d}",
                "prior_preferred_orientation_deg": float(sub["prior_preferred_orientation_deg"].iloc[0]),
                "prior_orientation_selectivity_index": float(sub["prior_orientation_selectivity_index"].iloc[0]),
                "static_peak_orientation_deg_by_mean_rate": static_peak_ori,
                "static_peak_spatial_cpd_by_mean_rate": static_peak_sf,
                "static_peak_mean_rate": static_peak_rate,
                "dynamic_peak_orientation_deg_by_amp": dyn_peak_ori,
                "dynamic_peak_spatial_cpd_by_amp": dyn_peak_sf,
                "dynamic_peak_temporal_hz_by_amp": dyn_peak_tf,
                "dynamic_peak_response_amp": dyn_peak_amp,
                "scalar_readout": str(sub["scalar_readout"].iloc[0]),
                "frequency_tuning_contract": (
                    "static orientation/SF from mean rate at TF=0; dynamic orientation/SF/TF from phase-RMS "
                    "response amplitude; scalar response comes from the recorded scalar_readout field"
                ),
            }
        )
    return grouped_rows, summary_rows


def plot_frequency_tuning(
    out_dir: Path,
    grouped_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    plot_units: list[int],
    dpi: int,
) -> tuple[Path, Path]:
    grouped = pd.DataFrame(grouped_rows)
    summary = pd.DataFrame(summary_rows)
    for frame in (grouped, summary):
        for col in [
            "unit_index",
            "probe_orientation_deg",
            "prior_preferred_orientation_deg",
            "prior_orientation_selectivity_index",
            "spatial_cpd",
            "temporal_hz",
            "mean_rate",
            "response_amp_rms",
            "static_peak_orientation_deg_by_mean_rate",
            "static_peak_spatial_cpd_by_mean_rate",
            "dynamic_peak_orientation_deg_by_amp",
            "dynamic_peak_spatial_cpd_by_amp",
            "dynamic_peak_temporal_hz_by_amp",
        ]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
    available = set(int(v) for v in summary["unit_index"].to_list())
    units = [int(v) for v in plot_units if int(v) in available]
    if not units:
        units = [int(v) for v in summary["unit_index"].to_list()[: min(6, len(summary))]]
    fig, axes = plt.subplots(len(units), 3, figsize=(12.0, 3.0 * len(units)), constrained_layout=True)
    if len(units) == 1:
        axes = np.asarray([axes])
    for row_idx, unit in enumerate(units):
        sub = grouped[grouped["unit_index"].astype(int) == int(unit)].copy()
        meta = summary[summary["unit_index"].astype(int) == int(unit)].iloc[0]
        static_ori = float(meta["static_peak_orientation_deg_by_mean_rate"])
        dynamic_ori = float(meta["dynamic_peak_orientation_deg_by_amp"])
        static = sub[
            np.isclose(sub["temporal_hz"].astype(float), 0.0)
            & np.isclose(sub["probe_orientation_deg"].astype(float), static_ori)
        ].sort_values("spatial_cpd")
        dynamic = sub[~np.isclose(sub["temporal_hz"].astype(float), 0.0)].copy()
        heatmap_at_best_ori = sub[np.isclose(sub["probe_orientation_deg"].astype(float), dynamic_ori)].copy()
        dynamic_at_best_ori = dynamic[np.isclose(dynamic["probe_orientation_deg"].astype(float), dynamic_ori)].copy()
        ax = axes[row_idx, 0]
        ax.plot(static["spatial_cpd"], static["mean_rate"], marker="o", color="#222222")
        ax.axvline(float(meta["static_peak_spatial_cpd_by_mean_rate"]), color="0.55", linestyle=":", linewidth=1.0)
        ax.set_xscale("log")
        sf_ticks = sorted(float(v) for v in sub["spatial_cpd"].dropna().unique())
        ax.set_xticks(sf_ticks, [f"{v:g}" for v in sf_ticks], rotation=45, ha="right")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("spatial frequency (cpd)")
        ax.set_ylabel(f"u{int(unit):03d}\nmean rate", rotation=0, ha="right", va="center", labelpad=32)
        ax.set_title(
            f"static SF at best ori {static_ori:.0f} deg\n"
            f"prior pref {float(meta['prior_preferred_orientation_deg']):.0f} deg"
        )
        ax.grid(True, which="both", color="0.9", linewidth=0.7)

        ax = axes[row_idx, 1]
        if not heatmap_at_best_ori.empty:
            heat = heatmap_at_best_ori.pivot_table(
                index="temporal_hz",
                columns="spatial_cpd",
                values="mean_rate",
                aggfunc="mean",
            ).sort_index().sort_index(axis=1)
            heat_arr = heat.to_numpy(dtype=float)
            vmax = float(np.nanmax(heat_arr)) if np.isfinite(heat_arr).any() else 1.0
            im = ax.imshow(
                heat_arr / max(vmax, EPS),
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_xticks(np.arange(heat.shape[1]), [f"{float(v):g}" for v in heat.columns], rotation=45, ha="right")
            ax.set_yticks(np.arange(heat.shape[0]), [f"{float(v):g}" for v in heat.index])
            heat_cols = [float(v) for v in heat.columns]
            heat_index = [float(v) for v in heat.index]
            peak_sf = float(meta["dynamic_peak_spatial_cpd_by_amp"])
            peak_tf = float(meta["dynamic_peak_temporal_hz_by_amp"])
            if peak_sf in heat_cols and peak_tf in heat_index:
                ax.scatter(
                    [heat_cols.index(peak_sf)],
                    [heat_index.index(peak_tf)],
                    marker="x",
                    color="cyan",
                    s=45,
                    linewidths=1.4,
                )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="norm mean rate")
        ax.set_xlabel("spatial frequency (cpd)")
        ax.set_ylabel("temporal frequency (Hz)")
        ax.set_title(f"SF x TF mean rate\nbest dynamic ori {dynamic_ori:.0f} deg")

        ax = axes[row_idx, 2]
        if not dynamic_at_best_ori.empty:
            best_sf = float(meta["dynamic_peak_spatial_cpd_by_amp"])
            best_sf_rows = dynamic_at_best_ori[
                np.isclose(dynamic_at_best_ori["spatial_cpd"].astype(float), best_sf)
            ].sort_values("temporal_hz")
            ax.plot(best_sf_rows["temporal_hz"], best_sf_rows["response_amp_rms"], marker="o", color="#1f7a8c")
            ax.axvline(float(meta["dynamic_peak_temporal_hz_by_amp"]), color="0.55", linestyle=":", linewidth=1.0)
            ax.set_xscale("log")
            tf_ticks = sorted(float(v) for v in best_sf_rows["temporal_hz"].dropna().unique())
            ax.set_xticks(tf_ticks, [f"{v:g}" for v in tf_ticks])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("temporal frequency (Hz)")
        ax.set_ylabel("response amp")
        ax.set_title(f"TF tuning at best SF\n{float(meta['dynamic_peak_spatial_cpd_by_amp']):g} cpd")
        ax.grid(True, which="both", color="0.9", linewidth=0.7)
    scalar_mode = str(summary["scalar_readout"].iloc[0]) if "scalar_readout" in summary.columns and not summary.empty else "unknown"
    fig.suptitle(f"BackImage RR100 grating SF/TF tuning probes ({scalar_mode} readout)", fontsize=12)
    png = out_dir / "backimage_rr100_selected_unit_frequency_tuning.png"
    pdf = out_dir / "backimage_rr100_selected_unit_frequency_tuning.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_population_frequency_summary(
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    *,
    dpi: int,
) -> tuple[Path, Path]:
    summary = pd.DataFrame(summary_rows)
    for col in [
        "unit_index",
        "prior_preferred_orientation_deg",
        "static_peak_orientation_deg_by_mean_rate",
        "static_peak_spatial_cpd_by_mean_rate",
        "dynamic_peak_orientation_deg_by_amp",
        "dynamic_peak_spatial_cpd_by_amp",
        "dynamic_peak_temporal_hz_by_amp",
        "dynamic_peak_response_amp",
    ]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), constrained_layout=True)
    ax = axes[0, 0]
    scatter = ax.scatter(
        summary["prior_preferred_orientation_deg"],
        summary["dynamic_peak_orientation_deg_by_amp"],
        c=summary["dynamic_peak_temporal_hz_by_amp"],
        s=28 + 70 * summary["dynamic_peak_response_amp"] / max(float(summary["dynamic_peak_response_amp"].max()), EPS),
        cmap="viridis",
        alpha=0.8,
        edgecolors="white",
        linewidths=0.35,
    )
    ax.plot([0, 180], [0, 180], color="0.75", linestyle=":", linewidth=1.0)
    ax.set_xlim(-5, 185)
    ax.set_ylim(-5, 185)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel("prior orientation pref (deg)")
    ax.set_ylabel("dynamic peak orientation (deg)")
    ax.set_title("Orientation agreement")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03, label="peak TF (Hz)")
    ax.grid(True, color="0.9", linewidth=0.7)

    ax = axes[0, 1]
    dyn_heat = summary.pivot_table(
        index="dynamic_peak_temporal_hz_by_amp",
        columns="dynamic_peak_spatial_cpd_by_amp",
        values="unit_index",
        aggfunc="count",
        fill_value=0,
    ).sort_index().sort_index(axis=1)
    im = ax.imshow(dyn_heat.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(dyn_heat.shape[1]), [f"{float(v):g}" for v in dyn_heat.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(dyn_heat.shape[0]), [f"{float(v):g}" for v in dyn_heat.index])
    ax.set_xlabel("dynamic peak SF (cpd)")
    ax.set_ylabel("dynamic peak TF (Hz)")
    ax.set_title("Dynamic peak count")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="units")

    ax = axes[1, 0]
    sf_counts = summary["static_peak_spatial_cpd_by_mean_rate"].value_counts().sort_index()
    ax.bar([float(v) for v in sf_counts.index], sf_counts.to_numpy(dtype=float), width=1.8, color="#4c78a8")
    ax.set_xscale("log")
    sf_ticks = sorted(float(v) for v in summary["static_peak_spatial_cpd_by_mean_rate"].dropna().unique())
    ax.set_xticks(sf_ticks, [f"{v:g}" for v in sf_ticks], rotation=45, ha="right")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("static peak SF (cpd)")
    ax.set_ylabel("unit count")
    ax.set_title("Static SF peaks")
    ax.grid(True, axis="y", color="0.9", linewidth=0.7)

    ax = axes[1, 1]
    tf_counts = summary["dynamic_peak_temporal_hz_by_amp"].value_counts().sort_index()
    ax.bar([float(v) for v in tf_counts.index], tf_counts.to_numpy(dtype=float), width=2.8, color="#f58518")
    ax.set_xscale("log")
    tf_ticks = sorted(float(v) for v in summary["dynamic_peak_temporal_hz_by_amp"].dropna().unique())
    ax.set_xticks(tf_ticks, [f"{v:g}" for v in tf_ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("dynamic peak TF (Hz)")
    ax.set_ylabel("unit count")
    ax.set_title("Dynamic TF peaks")
    ax.grid(True, axis="y", color="0.9", linewidth=0.7)

    scalar_mode = str(summary["scalar_readout"].iloc[0]) if "scalar_readout" in summary.columns and not summary.empty else "unknown"
    fig.suptitle(f"BackImage RR100 all-unit SF/TF tuning summary ({scalar_mode} readout)", fontsize=12)
    png = out_dir / "backimage_rr100_all_unit_frequency_tuning_summary.png"
    pdf = out_dir / "backimage_rr100_all_unit_frequency_tuning_summary.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_units, orientation_summary = load_source_summary_and_plot_units(Path(args.source_dir), parse_int_list(str(args.units)))
    orientation_degrees = [orientation_axis_180(v) for v in parse_float_list(str(args.orientation_deg))]
    spatial_cpds = parse_float_list(str(args.spatial_cpds))
    temporal_hz = parse_float_list(str(args.temporal_hz))
    sampling = stimulus_sampling_summary(
        ppd=float(args.ppd),
        image_size=int(args.image_size),
        frame_rate_hz=float(args.frame_rate_hz),
    )
    over_spatial_nyquist = [float(v) for v in spatial_cpds if float(v) > float(sampling["spatial_nyquist_cpd"])]
    over_temporal_nyquist = [float(v) for v in temporal_hz if float(v) > float(sampling["temporal_nyquist_hz"])]
    identity = {
        "analysis": "backimage_rr100_frequency_tuning_probe",
        "source_dir": Path(args.source_dir).resolve(),
        "rr100_version": str(args.rr100_version),
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "computed_units": "all_rr100_units",
        "plot_units": plot_units,
        "orientation_degrees": orientation_degrees,
        "spatial_cpds": spatial_cpds,
        "temporal_hz": temporal_hz,
        "scalar_readout": str(args.scalar_readout),
        "n_phases": int(args.n_phases),
        "static_n_phases": int(args.static_n_phases),
        "phase_seed": int(args.phase_seed),
        "duration_s": float(args.duration_s),
        "frame_rate_hz": float(args.frame_rate_hz),
        "n_lags": int(args.n_lags),
        "discard_frames": int(args.discard_frames),
        "image_size": int(args.image_size),
        "ppd": float(args.ppd),
        "contrast": float(args.contrast),
        "window_sigma_frac": float(args.window_sigma_frac),
        "stimulus_sampling": sampling,
        "over_spatial_nyquist_cpds": over_spatial_nyquist,
        "over_temporal_nyquist_hz": over_temporal_nyquist,
        "probe_contract": (
            "shared orientation/SF/TF windowed drifting gratings; all RR100 units read from each repeated "
            "stimulus; default scalar response is center pixel of the post-activation RR100 rate map; "
            "TF=0 static baseline uses seeded shuffled starting phases"
        ),
    }
    if over_spatial_nyquist:
        print(
            f"WARNING: spatial frequencies above Nyquist ({sampling['spatial_nyquist_cpd']:.3g} cpd): "
            + ", ".join(f"{v:g}" for v in over_spatial_nyquist),
            flush=True,
        )
    if over_temporal_nyquist:
        print(
            f"WARNING: temporal frequencies above Nyquist ({sampling['temporal_nyquist_hz']:.3g} Hz): "
            + ", ".join(f"{v:g}" for v in over_temporal_nyquist),
            flush=True,
        )
    write_json(out_dir / "frequency_tuning_request_identity.json", identity)
    if bool(args.dry_run):
        print(json.dumps(json_ready(identity), indent=2, sort_keys=True))
        return

    manifest = out_dir / "frequency_tuning_manifest.json"
    rows_csv = out_dir / "frequency_tuning_probe_rows.csv"
    grouped_csv = out_dir / "frequency_tuning_grouped.csv"
    summary_csv = out_dir / "frequency_tuning_summary.csv"
    use_cache = False
    if rows_csv.exists() and grouped_csv.exists() and summary_csv.exists() and manifest.exists() and not bool(args.force):
        try:
            observed = json.loads(manifest.read_text(encoding="utf-8")).get("identity_text", "")
            use_cache = str(observed) == identity_text(identity)
        except Exception:
            use_cache = False
    if use_cache:
        probe_rows = read_csv_rows(rows_csv)
        grouped_rows = read_csv_rows(grouped_csv)
        summary_rows = read_csv_rows(summary_csv)
        print(f"Loaded cached frequency-tuning rows from {rows_csv}")
    else:
        probe_rows = compute_probe_rows(args, orientation_summary)
        grouped_rows, summary_rows = aggregate_probe_rows(probe_rows)
        write_csv(rows_csv, probe_rows)
        write_csv(grouped_csv, grouped_rows)
        write_csv(summary_csv, summary_rows)
    png, pdf = plot_frequency_tuning(out_dir, grouped_rows, summary_rows, plot_units=plot_units, dpi=int(args.dpi))
    pop_png, pop_pdf = plot_population_frequency_summary(out_dir, summary_rows, dpi=int(args.dpi))
    write_json(
        manifest,
        {
            "identity": identity,
            "identity_text": identity_text(identity),
            "n_probe_rows": len(probe_rows),
            "n_grouped_rows": len(grouped_rows),
            "n_summary_rows": len(summary_rows),
            "outputs": {
                "probe_rows_csv": rows_csv,
                "grouped_csv": grouped_csv,
                "summary_csv": summary_csv,
                "figure_png": png,
                "figure_pdf": pdf,
                "population_summary_png": pop_png,
                "population_summary_pdf": pop_pdf,
            },
        },
    )
    print(f"Wrote frequency-tuning probe outputs to {out_dir}")
    summary_by_unit = {int(row["unit_index"]): row for row in summary_rows}
    for unit in plot_units:
        if int(unit) not in summary_by_unit:
            continue
        row = summary_by_unit[int(unit)]
        print(
            f"{row['unit_label']}: static peak ori {float(row['static_peak_orientation_deg_by_mean_rate']):g} deg, "
            f"{float(row['static_peak_spatial_cpd_by_mean_rate']):g} cpd; dynamic peak ori "
            f"{float(row['dynamic_peak_orientation_deg_by_amp']):g} deg, "
            f"{float(row['dynamic_peak_spatial_cpd_by_amp']):g} cpd, "
            f"{float(row['dynamic_peak_temporal_hz_by_amp']):g} Hz",
            flush=True,
        )


if __name__ == "__main__":
    main()
