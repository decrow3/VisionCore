#!/usr/bin/env python3
"""Map-first checkpoint figures for temporal power-shift analysis.

Stage 1 deliberately stops before activation maps. It renders the concrete
normal-vs-stabilized input manipulation for one saved image/trace pair.
"""

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
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import (
    DEFAULT_SOURCE_CSV,
    build_trace_bank,
    load_source_rows,
)
from declan.active_sensing_movie_information.temporal_remapping import MODEL_NYQUIST_HZ, MODEL_RATE_HZ
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch


DEFAULT_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/temporal_remapping/"
    "backimage_rr100_retiming_medium_figure4pool_n16_t32_fullgrid_cuda0_v1"
)
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_01_inputs_v1"
DEFAULT_DENSE_SFTF_CSV = ROOT / (
    "outputs/active_sensing_movie_information/backimage_rr100_dense_sf_tf_speed_pref_groups_v1/"
    "cycle_valid_dense_sf_tf_fit_unit_summary.csv"
)
SF_BANDS = [
    (0.0, 2.0, "0-2 cpd", "image_power_0_2_cpd_fraction"),
    (2.0, 4.0, "2-4 cpd", "image_power_2_4_cpd_fraction"),
    (4.0, 8.0, "4-8 cpd", "image_power_4_8_cpd_fraction"),
    (8.0, math.inf, "8+ cpd", "image_power_8plus_cpd_fraction"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--dense-sftf-csv", type=Path, default=DEFAULT_DENSE_SFTF_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--image-position", type=int, default=0)
    parser.add_argument("--trace-index", type=int, default=0)
    parser.add_argument("--n-timepoints", type=int, default=32)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--frame-rate-hz", type=float, default=MODEL_RATE_HZ)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


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


def source_row_by_id(rows: pd.DataFrame, source_row: int) -> pd.Series:
    matches = rows[pd.to_numeric(rows["source_row"], errors="coerce").astype("Int64") == int(source_row)]
    if matches.empty:
        raise ValueError(f"source_row={source_row} not found.")
    return matches.iloc[0]


def one_trace_from_source(rows: pd.DataFrame, source_row: int, *, n_timepoints: int, bin_seconds: float) -> np.ndarray:
    row = source_row_by_id(rows, int(source_row))
    bank = build_trace_bank(
        pd.DataFrame([row]),
        n_timepoints=int(n_timepoints),
        bin_seconds=float(bin_seconds),
        max_path_arcmin=1.0e9,
    )
    if not bank:
        raise ValueError(f"No trace could be reconstructed for source_row={source_row}.")
    trace = np.asarray(bank[0]["trace"], dtype=np.float32)
    if trace.shape != (int(n_timepoints), 2):
        raise ValueError(f"Expected trace shape {(int(n_timepoints), 2)}, got {trace.shape}.")
    return trace


def image_table_row_by_position(table: pd.DataFrame, image_position: int) -> pd.Series:
    position = int(image_position)
    if "image_position" in table.columns:
        matches = table[pd.to_numeric(table["image_position"], errors="coerce").astype("Int64") == position]
        if not matches.empty:
            return matches.iloc[0]
    if "image_index" in table.columns:
        matches = table[pd.to_numeric(table["image_index"], errors="coerce").astype("Int64") == position]
        if not matches.empty:
            return matches.iloc[0]
    if 0 <= position < table.shape[0]:
        return table.iloc[position]
    raise ValueError(f"image_position={position} not found in image table.")


def radial_power_spectrum(patch: np.ndarray, ppd: float, *, n_bins: int = 80) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(patch, dtype=np.float64)
    arr = arr - float(np.nanmean(arr))
    window_y = np.hanning(arr.shape[0])
    window_x = np.hanning(arr.shape[1])
    tapered = arr * window_y[:, None] * window_x[None, :]
    fft = np.fft.fftshift(np.fft.fft2(tapered))
    power = np.abs(fft) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(arr.shape[0], d=1.0 / float(ppd)))
    fx = np.fft.fftshift(np.fft.fftfreq(arr.shape[1], d=1.0 / float(ppd)))
    rr = np.sqrt(fx[None, :] * fx[None, :] + fy[:, None] * fy[:, None])
    max_freq = float(np.nanmax(rr))
    edges = np.geomspace(max(1.0e-3, 1.0 / (arr.shape[0] / float(ppd))), max_freq, int(n_bins) + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    values = np.full((int(n_bins),), np.nan, dtype=np.float64)
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (rr >= lo) & (rr < hi)
        if np.any(mask):
            values[idx] = float(np.nanmean(power[mask]))
    valid = np.isfinite(values) & (values > 0.0)
    return centers[valid], values[valid]


def choose_representative_units(units: pd.DataFrame, dense_sftf: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        ("low_sf_representative", "low_sf", 0.05),
        ("middle_sf_representative", "middle_sf", 0.25),
        ("high_sf_representative", "high_sf", 1.0),
    ]
    work = units.copy()
    work["preferred_sf_cpd_num"] = pd.to_numeric(work["preferred_sf_cpd"], errors="coerce")
    work["dynamic_peak_temporal_hz_num"] = pd.to_numeric(work.get("dynamic_peak_temporal_hz_by_amp"), errors="coerce")
    dense_lookup: dict[int, dict[str, Any]] = {}
    if dense_sftf is not None and not dense_sftf.empty:
        for dense_row in dense_sftf.itertuples(index=False):
            dense_lookup[int(getattr(dense_row, "unit_index"))] = {
                "fit_pref_tf_hz": finite_float(getattr(dense_row, "fit_pref_tf_hz", float("nan"))),
                "fit_status": str(getattr(dense_row, "fit_status", "")),
            }
    work["dense_fit_pref_tf_hz_num"] = [
        finite_float(dense_lookup.get(int(unit_index), {}).get("fit_pref_tf_hz"))
        for unit_index in work["unit_index"].to_numpy(dtype=int)
    ]
    for role, group, target_sf in specs:
        pool = work[
            work["sf_group"].astype(str).eq(group)
            & np.isfinite(work["preferred_sf_cpd_num"].to_numpy(dtype=float))
        ].copy()
        if pool.empty:
            pool = work[np.isfinite(work["preferred_sf_cpd_num"].to_numpy(dtype=float))].copy()
        pool["distance"] = np.abs(np.log2(pool["preferred_sf_cpd_num"].to_numpy(dtype=float) / float(target_sf)))
        pool["has_dense_tf"] = np.isfinite(pool["dense_fit_pref_tf_hz_num"].to_numpy(dtype=float))
        row = pool.sort_values(["has_dense_tf", "distance", "unit_index"], ascending=[False, True, True], kind="mergesort").iloc[0]
        unit_index = int(row["unit_index"])
        dense_meta = dense_lookup.get(unit_index, {})
        fit_pref_tf = finite_float(dense_meta.get("fit_pref_tf_hz"))
        fit_status = str(dense_meta.get("fit_status", ""))
        rows.append(
            {
                "unit_index": unit_index,
                "unit_label": str(row.get("unit_label", f"u{unit_index:03d}")),
                "selection_role": role,
                "sf_group": str(row.get("sf_group", "")),
                "preferred_sf_cpd": float(row["preferred_sf_cpd_num"]),
                "dense_fit_pref_tf_hz": fit_pref_tf,
                "dense_fit_status": fit_status,
                "legacy_dynamic_peak_temporal_hz_by_amp": finite_float(row.get("dynamic_peak_temporal_hz_num")),
            }
        )
    return rows


def speed_by_frame(trace: np.ndarray, frame_rate_hz: float) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    speed = np.zeros((trace.shape[0],), dtype=np.float64)
    if trace.shape[0] > 1:
        speed[1:] = np.linalg.norm(np.diff(trace, axis=0), axis=1) * float(frame_rate_hz)
    return speed


def plot_checkpoint(
    *,
    out_dir: Path,
    patch: np.ndarray,
    patch_meta: dict[str, Any],
    image_row: pd.Series,
    trace: np.ndarray,
    selected_units: list[dict[str, Any]],
    units: pd.DataFrame,
    population_static: float,
    population_original: float,
    frame_rate_hz: float,
    dpi: int,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ppd = float(patch_meta["patch_ppd"])
    patch_arr = np.asarray(patch, dtype=np.float32)
    speed = speed_by_frame(trace, float(frame_rate_hz))
    frames = np.arange(trace.shape[0], dtype=int)
    dt_ms = 1000.0 / float(frame_rate_hz)
    time_ms = frames * dt_ms
    centered = trace - np.mean(trace, axis=0, keepdims=True)
    sf_x, sf_y = radial_power_spectrum(patch_arr, ppd)

    fig = plt.figure(figsize=(16.6, 10.0), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        6,
        height_ratios=[1.0, 1.08],
        left=0.055,
        right=0.985,
        top=0.88,
        bottom=0.075,
        hspace=0.42,
        wspace=0.42,
    )
    ax_patch = fig.add_subplot(gs[0, 0:2])
    ax_path = fig.add_subplot(gs[0, 2:4])
    ax_speed = fig.add_subplot(gs[0, 4:6])
    ax_power = fig.add_subplot(gs[1, 0:3])
    ax_tf = fig.add_subplot(gs[1, 3:6])

    pvmin, pvmax = np.nanpercentile(patch_arr, [1.0, 99.0])
    ax_patch.imshow(patch_arr, cmap="gray", vmin=pvmin, vmax=pvmax)
    xy_px = np.column_stack(
        [
            patch_arr.shape[1] / 2.0 + centered[:, 0] * ppd,
            patch_arr.shape[0] / 2.0 - centered[:, 1] * ppd,
        ]
    )
    ax_patch.plot(xy_px[:, 0], xy_px[:, 1], color="#e45756", lw=1.6)
    ax_patch.scatter([xy_px[0, 0]], [xy_px[0, 1]], s=34, color="#1f77b4", label="start")
    ax_patch.scatter([xy_px[-1, 0]], [xy_px[-1, 1]], s=34, color="#e45756", label="end")
    ax_patch.set_title(
        f"Stimulus patch with normal eye path\nsource row {int(image_row['source_row'])}",
        fontsize=12,
    )
    ax_patch.set_xticks([])
    ax_patch.set_yticks([])
    ax_patch.legend(frameon=False, fontsize=9, loc="lower right")

    sc = ax_path.scatter(centered[:, 0], centered[:, 1], c=frames, cmap="viridis", s=26)
    ax_path.plot(centered[:, 0], centered[:, 1], color="0.25", lw=0.8, alpha=0.75)
    ax_path.scatter([0.0], [0.0], color="#777777", s=35, marker="+", label="stabilized")
    ax_path.set_title("Eye-position samples sent to the model", fontsize=12)
    ax_path.set_xlabel("horizontal offset from stabilized view (deg)")
    ax_path.set_ylabel("vertical offset from stabilized view (deg)")
    ax_path.axis("equal")
    ax_path.grid(True, color="#e8e8e8", lw=0.7)
    ax_path.legend(frameon=False, fontsize=9, loc="best")
    ax_path.text(
        0.03,
        0.04,
        "dot color: early frames to late frames",
        transform=ax_path.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="0.25",
        bbox={"facecolor": "white", "edgecolor": "0.88", "boxstyle": "round,pad=0.22", "alpha": 0.86},
    )

    ax_speed.plot(time_ms, speed, color="#2c6db2", lw=1.8, label="normal")
    ax_speed.plot(time_ms, np.zeros_like(speed), color="#777777", lw=1.3, label="stabilized")
    ax_speed.set_title("How fast the retinal image moves", fontsize=12)
    ax_speed.set_xlabel("time (ms)")
    ax_speed.set_ylabel("retinal speed (deg/s)")
    ax_speed.grid(True, color="#e8e8e8", lw=0.7)
    ax_speed.legend(frameon=False, fontsize=9)
    peak_idx = int(np.nanargmax(speed)) if speed.size else 0
    peak_speed = float(speed[peak_idx]) if speed.size else 0.0
    if peak_speed > 0.0:
        ax_speed.set_ylim(0.0, peak_speed * 1.24)
    if peak_speed > 0.0:
        ax_speed.annotate(
            f"peak {peak_speed:.1f} deg/s",
            xy=(time_ms[peak_idx], peak_speed),
            xytext=(time_ms[peak_idx], peak_speed * 1.10),
            ha="center",
            va="bottom",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.8},
        )
    delta_ssi = float(population_original - population_static)
    ax_speed.text(
        0.02,
        0.05,
        f"population SSI: normal - static = {delta_ssi:+.4f} bits/spike",
        transform=ax_speed.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "0.82", "boxstyle": "round,pad=0.25", "alpha": 0.92},
    )

    ax_power.plot(sf_x, sf_y / np.nanmax(sf_y), color="#111111", lw=1.8)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    ymax = 1.05
    handles = []
    labels = []
    for color, (lo, hi, label, column) in zip(colors, SF_BANDS, strict=True):
        frac = finite_float(image_row.get(column), 0.0)
        if math.isfinite(hi):
            patch_band = ax_power.axvspan(max(lo, 1.0e-3), hi, color=color, alpha=0.12)
        else:
            patch_band = ax_power.axvspan(lo, max(float(np.nanmax(sf_x)), lo * 1.25), color=color, alpha=0.12)
        handles.append(patch_band)
        labels.append(f"{label}: {frac:.2f}")
    preferred = pd.to_numeric(units["preferred_sf_cpd"], errors="coerce").to_numpy(dtype=float)
    preferred = preferred[np.isfinite(preferred) & (preferred > 0.0)]
    if preferred.size:
        ax_power.scatter(preferred, np.full_like(preferred, 0.04), marker="|", s=42, color="0.25", alpha=0.45)
    ax_power.set_xscale("log")
    ax_power.set_ylim(0.0, ymax)
    ax_power.set_title("What spatial scales are present in the image?", fontsize=12)
    ax_power.set_xlabel("spatial frequency (cycles/degree)")
    ax_power.set_ylabel("relative image power")
    ax_power.grid(True, color="#e8e8e8", lw=0.7, which="both")
    ax_power.legend(
        handles,
        labels,
        title="coarse bands: power share",
        frameon=False,
        fontsize=9,
        title_fontsize=9,
        loc="upper right",
    )
    ax_power.text(
        0.02,
        0.09,
        "gray ticks = unit preferred SFs",
        transform=ax_power.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="0.28",
    )

    unit_colors = ["#4c78a8", "#54a24b", "#e45756"]
    landings: list[np.ndarray] = []
    off_scale_preferences: list[str] = []
    for color, unit in zip(unit_colors, selected_units, strict=True):
        sf = float(unit["preferred_sf_cpd"])
        landing = sf * speed
        landings.append(landing)
        ax_tf.plot(time_ms, landing, color=color, lw=2.0, label=f"{unit['unit_label']} landing, SF={sf:.3g}")
    landing_max = max([float(np.nanmax(v)) for v in landings if v.size] + [1.0])
    relevant_pref_max = landing_max
    for color, unit in zip(unit_colors, selected_units, strict=True):
        tf_pref = finite_float(unit.get("dense_fit_pref_tf_hz"))
        if math.isfinite(tf_pref) and tf_pref > 0.0:
            if tf_pref <= max(landing_max * 1.65, 12.0):
                relevant_pref_max = max(relevant_pref_max, tf_pref)
                ax_tf.axhline(tf_pref, color=color, lw=1.2, alpha=0.55, linestyle=":")
                ax_tf.text(
                    time_ms[-1],
                    tf_pref,
                    f" {unit['unit_label']} TF pref",
                    color=color,
                    ha="left",
                    va="center",
                    fontsize=8.5,
                )
            else:
                off_scale_preferences.append(f"{unit['unit_label']} TF pref {tf_pref:.0f} Hz")
    y_top = max(6.0, relevant_pref_max * 1.18)
    y_top = min(float(MODEL_NYQUIST_HZ), y_top)
    if MODEL_NYQUIST_HZ <= y_top:
        ax_tf.axhline(MODEL_NYQUIST_HZ, color="0.25", lw=1.0, linestyle="--", label="120 Hz Nyquist")
    ax_tf.set_title("Predicted temporal drive from motion", fontsize=12)
    ax_tf.set_xlabel("time (ms)")
    ax_tf.set_ylabel("temporal frequency landing (Hz)")
    ax_tf.set_ylim(0.0, y_top)
    ax_tf.set_xlim(float(time_ms[0]), float(time_ms[-1]) * 1.10)
    ax_tf.grid(True, color="#e8e8e8", lw=0.7)
    ax_tf.legend(frameon=False, fontsize=9, loc="upper left")
    if off_scale_preferences:
        ax_tf.text(
            0.98,
            0.96,
            "; ".join(off_scale_preferences) + " above axis",
            transform=ax_tf.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="0.28",
            bbox={"facecolor": "white", "edgecolor": "0.85", "boxstyle": "round,pad=0.22", "alpha": 0.9},
        )

    fig.suptitle(
        "Input checkpoint: normal eye motion converts image structure into temporal drive",
        fontsize=14,
    )
    png = out_dir / "checkpoint_01_input_power_shift_overview.png"
    pdf = out_dir / "checkpoint_01_input_power_shift_overview.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    image_table = pd.read_csv(run_dir / "image_feature_table.csv")
    trace_table = pd.read_csv(run_dir / "trace_feature_table.csv")
    unit_table = pd.read_csv(run_dir / "unit_feature_table.csv")
    dense_sftf = pd.read_csv(args.dense_sftf_csv) if Path(args.dense_sftf_csv).exists() else pd.DataFrame()
    observations = pd.read_csv(run_dir / "retiming_population_observations.csv")
    image_row = image_table_row_by_position(image_table, int(args.image_position))
    trace_row = trace_table[trace_table["trace_index"].astype(int) == int(args.trace_index)].iloc[0]
    source_rows = load_source_rows(Path(args.source_csv))

    patch_row = source_row_by_id(source_rows, int(image_row["source_row"]))
    patch, patch_meta = _extract_patch(
        patch_row,
        canvas_cache={},
        patch_size_px=int(args.patch_size_px),
    )
    bin_seconds = 1.0 / float(args.frame_rate_hz)
    trace = one_trace_from_source(
        source_rows,
        int(trace_row["trace_source_row"]),
        n_timepoints=int(args.n_timepoints),
        bin_seconds=bin_seconds,
    )
    selected_units = choose_representative_units(unit_table, dense_sftf)

    obs_pair = observations[
        (observations["image_position"].astype(int) == int(args.image_position))
        & (observations["trace_index"].astype(int) == int(args.trace_index))
        & observations["condition_id"].astype(str).isin(["stabilized_static", "original_natural_timing"])
    ]
    by_condition = {
        str(row.condition_id): float(row.population_ssi_bits_per_spike)
        for row in obs_pair.itertuples(index=False)
    }
    population_static = float(by_condition.get("stabilized_static", float("nan")))
    population_original = float(by_condition.get("original_natural_timing", float("nan")))

    png, pdf = plot_checkpoint(
        out_dir=out_dir,
        patch=np.asarray(patch),
        patch_meta=patch_meta,
        image_row=image_row,
        trace=trace,
        selected_units=selected_units,
        units=unit_table,
        population_static=population_static,
        population_original=population_original,
        frame_rate_hz=float(args.frame_rate_hz),
        dpi=int(args.dpi),
    )

    speed = speed_by_frame(trace, float(args.frame_rate_hz))
    frame_rows: list[dict[str, Any]] = []
    for frame_idx, speed_value in enumerate(speed):
        row = {
            "frame_index": int(frame_idx),
            "time_ms": float(frame_idx * 1000.0 / float(args.frame_rate_hz)),
            "normal_speed_deg_s": float(speed_value),
            "stabilized_speed_deg_s": 0.0,
        }
        for unit in selected_units:
            label = str(unit["unit_label"])
            row[f"{label}_preferred_sf_cpd"] = float(unit["preferred_sf_cpd"])
            row[f"{label}_tf_landing_hz"] = float(unit["preferred_sf_cpd"]) * float(speed_value)
            row[f"{label}_dense_fit_pref_tf_hz"] = finite_float(unit.get("dense_fit_pref_tf_hz"))
        frame_rows.append(row)
    values_csv = out_dir / "checkpoint_01_framewise_speed_tf_landing.csv"
    write_csv(values_csv, frame_rows)

    selected_csv = out_dir / "checkpoint_01_representative_units.csv"
    write_csv(selected_csv, selected_units)
    write_json(
        out_dir / "checkpoint_01_metadata.json",
        {
            "analysis": "temporal_power_shift_map_first_checkpoint_01_inputs",
            "run_dir": run_dir,
            "source_csv": Path(args.source_csv),
            "dense_sftf_csv": Path(args.dense_sftf_csv),
            "image_position": int(args.image_position),
            "image_source_row": int(image_row["source_row"]),
            "trace_index": int(args.trace_index),
            "trace_source_row": int(trace_row["trace_source_row"]),
            "n_timepoints": int(args.n_timepoints),
            "frame_rate_hz": float(args.frame_rate_hz),
            "patch_meta": patch_meta,
            "population_static_ssi_bits_per_spike": population_static,
            "population_original_ssi_bits_per_spike": population_original,
            "selected_units": selected_units,
            "outputs": {
                "overview_png": png,
                "overview_pdf": pdf,
                "framewise_values_csv": values_csv,
                "representative_units_csv": selected_csv,
            },
            "checkpoint_policy": "Stop after input/mechanism figure before rendering activation maps.",
        },
    )
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {values_csv}")
    print(f"Wrote {selected_csv}")


if __name__ == "__main__":
    main()
