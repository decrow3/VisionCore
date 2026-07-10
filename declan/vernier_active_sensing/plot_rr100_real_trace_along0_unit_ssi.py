#!/usr/bin/env python3
"""Diagnose RR100 unit SSI along the original real-trace along=0 scale line."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view
from declan.vernier_active_sensing.forward import (
    PKL_PATH,
    STIMULUS_NORMALIZATION,
    build_vernier_movie,
    load_model_and_readout,
)
from declan.vernier_active_sensing.plot_rr100_endpoint_along0_unit_ssi import (
    compare_to_summary,
    condition_sequence,
    draw_leave_one_out,
    draw_unit_lines,
    draw_unit_lines_with_activation_rows,
    image_scale,
    order_units_by_y_at_x,
    summarize_units,
    unit_ssi_single_frame,
    write_csv_rows,
    write_json,
)
from declan.vernier_active_sensing.run_rr100_real_trace_scale_grid import (
    DEFAULT_SCALES,
    RR100_VERSION,
    canonical_vernier_spec,
    scale_token,
)
from declan.vernier_active_sensing.trajectories import (
    DEFAULT_EYE_TRACES_PATH,
    condition_trace,
    load_eye_traces,
    subsample_traces,
    valid_trace,
)
from scripts.temporal_decoding.rate_computation import compute_trial_rates
from scripts.temporal_decoding.stimulus_hires import N_LAGS as MODEL_HISTORY_FRAMES


DEFAULT_SCALE_GRID_DIR = ROOT / "outputs/notebook_vernier_walkthrough/rr100_real_trace_scale_grid"
DEFAULT_OUT_DIR = DEFAULT_SCALE_GRID_DIR / "unit_ssi_along0_diagnostics"
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SCALE_GRID_DIR / "rr100_real_trace_scale_grid_summary.csv")
    parser.add_argument("--eye-traces-path", type=Path, default=ROOT / DEFAULT_EYE_TRACES_PATH)
    parser.add_argument("--across-scales", type=str, default=DEFAULT_SCALES)
    parser.add_argument("--along-scale", type=float, default=0.0)
    parser.add_argument("--n-traces", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--fd-step-arcmin", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, cuda:1, ...")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-units", type=int, default=12)
    parser.add_argument(
        "--skip-highlighted-unit-maps",
        action="store_true",
        help="Only write unit SSI line diagnostics; skip highlighted-unit activation-map sheets.",
    )
    parser.add_argument(
        "--skip-individual-unit-maps",
        action="store_true",
        help="Skip one-PNG-per-highlighted-unit-per-scale activation maps.",
    )
    parser.add_argument("--map-vmin-percentile", type=float, default=0.5)
    parser.add_argument("--map-vmax-percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--force", action="store_true", help="Recompute cached unit SSI stats.")
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
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def parse_scale_list(text: str) -> list[float]:
    scales = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not scales:
        raise ValueError("At least one scale is required.")
    return scales


CACHE_SCHEMA_VERSION = 2


def stats_cache_path(out_dir: Path, condition: str, fd_step_arcmin: float, max_frames: int) -> Path:
    return (
        out_dir
        / "cache"
        / f"rr100_real_trace_along0_unit_ssi_{condition}_frames{int(max_frames)}_fd{float(fd_step_arcmin):.4f}arcmin.npz"
    )


def _cache_identity(args: argparse.Namespace, *, condition: str) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "condition": str(condition),
        "eye_traces_path": str(Path(args.eye_traces_path).expanduser().resolve()),
        "across_scales": parse_scale_list(args.across_scales),
        "along_scale": float(args.along_scale),
        "n_traces": int(args.n_traces),
        "max_frames": int(args.max_frames),
        "model_history_frames": int(MODEL_HISTORY_FRAMES),
        "fd_step_arcmin": float(args.fd_step_arcmin),
        "seed": int(args.seed),
        "trajectory_contract": "native real-trace positions, anisotropically scaled around trial mean",
        "readout_time_contract": "all response frames averaged for SSI",
        "population_version": RR100_VERSION,
        "readout_source_pkl": str(PKL_PATH.expanduser().resolve()),
        "plus_spec": asdict(canonical_vernier_spec(+float(args.fd_step_arcmin))),
        "minus_spec": asdict(canonical_vernier_spec(-float(args.fd_step_arcmin))),
        "map_contract": "trace-time mean finite-difference midpoint RR100 spatial maps",
    }


def _identity_text(identity: dict[str, Any]) -> str:
    return json.dumps(json_ready(identity), sort_keys=True, separators=(",", ":"))


def cache_has_required_fields(path: Path, *, require_maps: bool, expected_identity: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    required = {"cache_identity_json"}
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
    spec: Any,
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


def _unit_ssi_movie(midpoint_movie: np.ndarray) -> dict[str, np.ndarray | float]:
    bits: list[np.ndarray] = []
    rates: list[np.ndarray] = []
    population: list[float] = []
    for frame in np.asarray(midpoint_movie, dtype=np.float32):
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
    plus_spec = canonical_vernier_spec(+float(args.fd_step_arcmin))
    minus_spec = canonical_vernier_spec(-float(args.fd_step_arcmin))
    unit_bits: list[np.ndarray] = []
    unit_rates: list[np.ndarray] = []
    population_bits: list[float] = []
    map_sum: np.ndarray | None = None
    map_sq_sum: np.ndarray | None = None
    map_count = 0

    for trace_idx in range(trace_set.traces.shape[0]):
        base_trace = valid_trace(trace_set, trace_idx, max_frames=int(args.max_frames))
        rng = np.random.default_rng(int(args.seed) + 1009 * trace_idx)
        effective_trace, _trace_meta = condition_trace(
            base_trace,
            condition=condition,
            trace_set=trace_set,
            rng=rng,
        )
        trace = np.asarray(effective_trace[: int(args.max_frames)], dtype=np.float32)
        if trace.shape[0] != int(args.max_frames):
            raise RuntimeError(f"Expected {args.max_frames} frames for {condition}, got {trace.shape[0]}")

        plus_movie = _rr100_spatial_movie(
            model,
            readout,
            view,
            plus_spec,
            trace,
            device=device,
            batch_size=int(args.batch_size),
        )
        minus_movie = _rr100_spatial_movie(
            model,
            readout,
            view,
            minus_spec,
            trace,
            device=device,
            batch_size=int(args.batch_size),
        )
        t = min(int(plus_movie.shape[0]), int(minus_movie.shape[0]))
        midpoint_movie = 0.5 * (plus_movie[:t] + minus_movie[:t])
        ssi = _unit_ssi_movie(midpoint_movie)
        if map_sum is None:
            map_sum = np.zeros_like(midpoint_movie[0], dtype=np.float64)
            map_sq_sum = np.zeros_like(midpoint_movie[0], dtype=np.float64)
        map_sum += np.sum(midpoint_movie, axis=0, dtype=np.float64)
        map_sq_sum += np.sum(np.square(midpoint_movie, dtype=np.float64), axis=0)
        map_count += int(t)
        unit_bits.append(np.asarray(ssi["unit_bits_per_spike"], dtype=np.float32))
        unit_rates.append(np.asarray(ssi["unit_mean_rate"], dtype=np.float32))
        population_bits.append(float(ssi["population_bits_per_spike"]))
        print(f"  {condition} trace {trace_idx}: pop SSI={population_bits[-1]:.6g}; T={t}", flush=True)
        del plus_movie, minus_movie, midpoint_movie
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
    path = stats_cache_path(Path(args.out_dir), condition, float(args.fd_step_arcmin), int(args.max_frames))
    expected_identity = _cache_identity(args, condition=condition)
    if path.exists() and not bool(args.force):
        with np.load(path) as data:
            has_required_maps = (not require_maps) or "mean_rate_map" in data
            has_matching_identity = (
                "cache_identity_json" in data
                and str(np.asarray(data["cache_identity_json"]).ravel()[0]) == _identity_text(expected_identity)
            )
            if has_required_maps and has_matching_identity:
                print(f"Loaded real-trace unit SSI cache: {path}", flush=True)
                out = {
                    "unit_bits_per_trace": np.asarray(data["unit_bits_per_trace"], dtype=np.float32),
                    "unit_mean_rate_per_trace": np.asarray(data["unit_mean_rate_per_trace"], dtype=np.float32),
                    "population_bits_per_trace": np.asarray(data["population_bits_per_trace"], dtype=np.float32),
                }
                if "mean_rate_map" in data:
                    out["mean_rate_map"] = np.asarray(data["mean_rate_map"], dtype=np.float32)
                if "std_rate_map" in data:
                    out["std_rate_map"] = np.asarray(data["std_rate_map"], dtype=np.float32)
                return out
            print(f"Real-trace unit SSI cache metadata mismatch or missing maps; recomputing: {path}", flush=True)
    if trace_set is None or model is None or readout is None or view is None or device is None:
        raise RuntimeError("Model/readout/trace resources are required to compute missing caches.")
    print(f"Computing real-trace unit SSI cache: {condition}", flush=True)
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
        mean_rate_map=stats["mean_rate_map"],
        std_rate_map=stats["std_rate_map"],
        condition=np.asarray([condition]),
        fd_step_arcmin=np.asarray([float(args.fd_step_arcmin)], dtype=np.float32),
        max_frames=np.asarray([int(args.max_frames)], dtype=np.int32),
        trajectory_contract=np.asarray(["native_real_trace_scaled_around_trial_mean"]),
        readout_time_contract=np.asarray(["all_response_frames_mean"]),
        cache_identity_json=np.asarray([_identity_text(expected_identity)]),
        stimulus_normalization=np.asarray([STIMULUS_NORMALIZATION]),
    )
    print(f"Saved real-trace unit SSI cache: {path}", flush=True)
    return stats


def _scale_label(row: dict[str, Any]) -> str:
    if bool(row["is_static_baseline"]):
        return "static"
    return f"{float(row['across_scale']):g}x"


def draw_unit_map_sheet(
    *,
    unit_index: int,
    rows: list[dict[str, Any]],
    stats_by_condition: dict[str, dict[str, Any]],
    unit_df: pd.DataFrame,
    vmin: float,
    vmax: float,
    path: Path,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_cols = len(rows)
    fig, axes = plt.subplots(1, n_cols, figsize=(1.65 * n_cols, 2.2), dpi=int(dpi), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    last_im = None
    for pos, row in enumerate(rows):
        condition = str(row["condition"])
        unit_map = np.asarray(stats_by_condition[condition]["mean_rate_map"], dtype=np.float32)[int(unit_index)]
        meta = unit_df[(unit_df["unit_index"].eq(int(unit_index))) & (unit_df["condition"].eq(condition))]
        ssi_text = ""
        if not meta.empty:
            record = meta.iloc[0]
            ssi_text = f"SSI {float(record['unit_ssi_bits_per_spike_mean']):.4f}\nratio {float(record['unit_ssi_vs_static']):.2f}x"
        ax = axes_arr[pos]
        last_im = ax.imshow(unit_map, origin="lower", cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"{_scale_label(row)}\n{ssi_text}", fontsize=6.0, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color("#777777")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_arr.tolist(), fraction=0.018, pad=0.01)
        cbar.ax.tick_params(labelsize=6.0, length=2)
        cbar.set_label("mean activation", fontsize=6.5)
    fig.suptitle(
        f"RR100 u{int(unit_index):03d}: real-trace along=0 finite-difference midpoint maps",
        fontsize=10.0,
        y=1.08,
    )
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_individual_unit_map(
    *,
    unit_index: int,
    row: dict[str, Any],
    stats_by_condition: dict[str, dict[str, Any]],
    unit_df: pd.DataFrame,
    vmin: float,
    vmax: float,
    path: Path,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    condition = str(row["condition"])
    unit_map = np.asarray(stats_by_condition[condition]["mean_rate_map"], dtype=np.float32)[int(unit_index)]
    meta = unit_df[(unit_df["unit_index"].eq(int(unit_index))) & (unit_df["condition"].eq(condition))]
    title = f"RR100 u{int(unit_index):03d}  {_scale_label(row)}"
    subtitle = ""
    if not meta.empty:
        record = meta.iloc[0]
        subtitle = (
            f"SSI {float(record['unit_ssi_bits_per_spike_mean']):.5f} bits/spike  "
            f"ratio {float(record['unit_ssi_vs_static']):.3f}x"
        )
    fig, ax = plt.subplots(figsize=(2.35, 2.65), dpi=int(dpi), constrained_layout=True)
    im = ax.imshow(unit_map, origin="lower", cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(f"{title}\n{subtitle}", fontsize=7.2, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.025)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_highlighted_unit_maps(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    stats_by_condition: dict[str, dict[str, Any]],
    unit_df: pd.DataFrame,
    highlighted_units: list[int],
) -> tuple[Path, list[dict[str, Any]]]:
    map_root = Path(args.out_dir) / "highlighted_unit_activation_maps"
    sheet_dir = map_root / "unit_sheets"
    individual_dir = map_root / "individual"
    manifest_rows: list[dict[str, Any]] = []

    for unit_index in highlighted_units:
        images = [
            np.asarray(stats_by_condition[str(row["condition"])]["mean_rate_map"], dtype=np.float32)[int(unit_index)]
            for row in rows
        ]
        vmin, vmax = image_scale(images, float(args.map_vmin_percentile), float(args.map_vmax_percentile))
        sheet_path = sheet_dir / f"rr100_real_trace_along0_unit_{int(unit_index):03d}_activation_maps.png"
        draw_unit_map_sheet(
            unit_index=int(unit_index),
            rows=rows,
            stats_by_condition=stats_by_condition,
            unit_df=unit_df,
            vmin=vmin,
            vmax=vmax,
            path=sheet_path,
            dpi=int(args.dpi),
        )
        for row in rows:
            condition = str(row["condition"])
            meta = unit_df[(unit_df["unit_index"].eq(int(unit_index))) & (unit_df["condition"].eq(condition))]
            individual_path = ""
            if not bool(args.skip_individual_unit_maps):
                scale_slug = "static" if bool(row["is_static_baseline"]) else f"across_{scale_token(float(row['across_scale']))}"
                individual_png = (
                    individual_dir
                    / f"unit_{int(unit_index):03d}"
                    / f"rr100_real_trace_along0_unit_{int(unit_index):03d}_{scale_slug}_activation_map.png"
                )
                draw_individual_unit_map(
                    unit_index=int(unit_index),
                    row=row,
                    stats_by_condition=stats_by_condition,
                    unit_df=unit_df,
                    vmin=vmin,
                    vmax=vmax,
                    path=individual_png,
                    dpi=int(args.dpi),
                )
                individual_path = str(individual_png)
            record = meta.iloc[0].to_dict() if not meta.empty else {}
            manifest_rows.append(
                {
                    "unit_index": int(unit_index),
                    "condition": condition,
                    "across_scale": row["across_scale"],
                    "along_scale": float(args.along_scale),
                    "is_static_baseline": bool(row["is_static_baseline"]),
                    "map_source": "trace_time_mean_fd_midpoint_map",
                    "n_traces": int(np.asarray(stats_by_condition[condition]["unit_bits_per_trace"]).shape[0]),
                    "unit_ssi_bits_per_spike_mean": record.get("unit_ssi_bits_per_spike_mean", np.nan),
                    "unit_ssi_vs_static": record.get("unit_ssi_vs_static", np.nan),
                    "unit_log2_ssi_vs_static": record.get("unit_log2_ssi_vs_static", np.nan),
                    "unit_mean_rate_mean": record.get("unit_mean_rate_mean", np.nan),
                    "unit_sheet_png": str(sheet_path),
                    "individual_png": individual_path,
                    "unit_sheet_colormap": "gray_monotonic",
                    "unit_sheet_vmin": float(vmin),
                    "unit_sheet_vmax": float(vmax),
                }
            )

    manifest_csv = map_root / "rr100_real_trace_along0_highlighted_unit_map_manifest.csv"
    write_csv_rows(manifest_csv, manifest_rows)
    return manifest_csv, manifest_rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    across_scales = parse_scale_list(args.across_scales)
    rows = condition_sequence(across_scales, float(args.along_scale))
    conditions = [str(row["condition"]) for row in rows]
    require_maps = not bool(args.skip_highlighted_unit_maps)

    missing = [
        condition
        for condition in conditions
        if bool(args.force)
        or not cache_has_required_fields(
            stats_cache_path(Path(args.out_dir), condition, float(args.fd_step_arcmin), int(args.max_frames)),
            require_maps=require_maps,
            expected_identity=_cache_identity(args, condition=condition),
        )
    ]
    trace_set = model = readout = view = None
    device: str | None = None
    if missing:
        print(f"Need to compute {len(missing)} real-trace unit SSI caches.", flush=True)
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
    comparison = compare_to_summary(Path(args.summary_csv), diagnostics)
    if comparison is not None:
        print(
            "Summary cross-check max abs SSI ratio delta: "
            f"{comparison['max_abs_population_ratio_delta']:.6g}",
            flush=True,
        )

    unit_csv = Path(args.out_dir) / "rr100_real_trace_along0_unit_ssi_table.csv"
    top_csv = Path(args.out_dir) / "rr100_real_trace_along0_unit_ssi_top_units.csv"
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
    highlighted_units = sorted(set(top_by_unit).union(top_by_influence))

    figure_title = "RR100 real-trace unit SSI along the along=0 scale line"
    unit_lines_png = Path(args.out_dir) / "rr100_real_trace_along0_unit_ssi_lines.png"
    influence_unit_lines_png = Path(args.out_dir) / "rr100_real_trace_along0_unit_ssi_lines_top_influence.png"
    influence_unit_lines_with_maps_png = (
        Path(args.out_dir) / "rr100_real_trace_along0_unit_ssi_lines_top_influence_with_activation_rows.png"
    )
    loo_png = Path(args.out_dir) / "rr100_real_trace_along0_unit_ssi_leave_one_out.png"
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
    map_manifest_csv = None
    map_manifest_rows: list[dict[str, Any]] = []
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
                "activation maps below use monotonic grayscale per unit row; low is black and high is white; "
                "rows and legend are ordered by y at across=1"
            ),
        )
        map_manifest_csv, map_manifest_rows = write_highlighted_unit_maps(
            args=args,
            rows=rows,
            stats_by_condition=stats_by_condition,
            unit_df=unit_df,
            highlighted_units=highlighted_units,
        )
    draw_leave_one_out(
        diagnostics,
        top_by_influence,
        top_df,
        loo_png,
        int(args.dpi),
        figure_title="Does any single RR100 unit drive the original real-trace along=0 SSI line?",
    )

    npz_path = Path(args.out_dir) / "rr100_real_trace_along0_unit_ssi_diagnostics_arrays.npz"
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
    manifest_path = Path(args.out_dir) / "rr100_real_trace_along0_unit_ssi_manifest.json"
    write_json(
        manifest_path,
        {
            "analysis": "rr100_real_trace_along0_unit_ssi_diagnostic",
            "out_dir": Path(args.out_dir),
            "conditions": conditions,
            "across_scales": across_scales,
            "along_scale": float(args.along_scale),
            "n_traces": int(args.n_traces),
            "max_frames": int(args.max_frames),
            "stimulus_normalization": STIMULUS_NORMALIZATION,
            "fd_step_arcmin": float(args.fd_step_arcmin),
            "seed": int(args.seed),
            "device_arg": str(args.device),
            "actual_device": device,
            "batch_size": int(args.batch_size),
            "population_version": RR100_VERSION,
            "trajectory_contract": "native real trace scaled around trial mean",
            "readout_time_contract": "all response frames averaged for SSI",
            "unit_table_csv": unit_csv,
            "top_units_csv": top_csv,
            "unit_lines_png": unit_lines_png,
            "influence_unit_lines_png": influence_unit_lines_png,
            "influence_unit_lines_with_activation_rows_png": influence_unit_lines_with_maps_png
            if not bool(args.skip_highlighted_unit_maps)
            else None,
            "leave_one_out_png": loo_png,
            "arrays_npz": npz_path,
            "summary_cross_check": comparison,
            "top_units_by_unit_log2_deviation": top_by_unit,
            "top_units_by_leave_one_out_influence": top_by_influence,
            "top_units_by_unit_log2_deviation_plot_order_at_across1": top_by_unit_for_plot,
            "top_units_by_leave_one_out_influence_plot_order_at_across1": top_by_influence_for_plot,
            "highlighted_units_union": highlighted_units,
            "highlighted_unit_map_manifest_csv": map_manifest_csv,
            "highlighted_unit_map_count": len(map_manifest_rows),
        },
    )

    print(f"Wrote unit SSI table: {unit_csv}", flush=True)
    print(f"Wrote top-unit table: {top_csv}", flush=True)
    print(f"Wrote unit line plot: {unit_lines_png}", flush=True)
    print(f"Wrote influence unit line plot: {influence_unit_lines_png}", flush=True)
    if not bool(args.skip_highlighted_unit_maps):
        print(f"Wrote influence unit line plot with activation rows: {influence_unit_lines_with_maps_png}", flush=True)
    print(f"Wrote leave-one-out plot: {loo_png}", flush=True)
    if map_manifest_csv is not None:
        print(f"Wrote highlighted-unit activation-map manifest: {map_manifest_csv}", flush=True)
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
