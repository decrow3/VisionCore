#!/usr/bin/env python3
"""Render non-averaged instantaneous maps for bimodal microsaccade unit groups."""

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

from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    RR100_MOVIE_MEDOID_VERSION,
    combined_axis_trace,
    identity_text,
    image_scale,
    rate_map_for_trace,
    rotated_axis_deg,
    rotate_patch_gaze_frame,
    rotate_trace_xy,
    select_source_trials,
    trial_event_scale_mask,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
    _align_response_to_trace,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


DEFAULT_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_microsaccade_snippet_rr100_spatial_ssi_isotropic_padded_event_scaled_full_amp1sd_n40_v1"
)
DEFAULT_SCALES = "0,0.25,0.5,0.75,1,1.5,2,3"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--selected-units-csv", type=Path, default=None)
    parser.add_argument("--examples-per-group", type=int, default=4)
    parser.add_argument("--scales", type=str, default=DEFAULT_SCALES)
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--map-vmin-percentile", type=float, default=1.0)
    parser.add_argument("--map-vmax-percentile", type=float, default=99.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    out = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not out:
        raise ValueError("At least one scale is required.")
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


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_slug(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text).strip("_") or "unnamed"


def scale_token(value: float) -> str:
    return f"{float(value):.9g}".replace("-", "m").replace(".", "p")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files if key != "cache_identity_json"}


def source_args_from_run(run_dir: Path) -> argparse.Namespace:
    metadata = load_json(run_dir / "run_metadata.json")
    identity = metadata["identity"]
    source_meta = metadata.get("source_meta", {})
    source_config = source_meta.get("run_metadata_config", {})
    return argparse.Namespace(
        axis_run_dir=Path(identity["axis_run_dir"]),
        selected_windows_csv=Path(source_meta["selected_windows_csv"]),
        trial_source_mode="microsaccade_snippets",
        max_trials=0,
        trial_start=0,
        source_trace_scale=float(identity.get("source_trace_scale", 1.0) or 1.0),
        source_trace_prior_family=str(identity.get("source_trace_prior_family", "axis_edge_parallel")),
        axis_column=str(identity.get("axis_column", "image_edge_axis_deg")),
        n_timepoints=int(identity["n_timepoints"]),
        microsaccade_pre_frames=int(identity.get("microsaccade_pre_frames", 8)),
        microsaccade_post_frames=int(identity.get("microsaccade_post_frames", 36)),
        microsaccade_max_source_windows=0,
        microsaccade_min_amplitude_arcmin=float(identity.get("microsaccade_min_amplitude_arcmin", 0.0)),
        microsaccade_max_amplitude_arcmin=float(identity.get("microsaccade_max_amplitude_arcmin", 60.0)),
        microsaccade_amplitude_sd_filter=float(identity.get("microsaccade_amplitude_sd_filter", 0.0)),
        microsaccade_trace_mode=str(identity.get("microsaccade_trace_mode", "full_snippet")),
        microsaccade_require_snippet_within_source_window=bool(
            identity.get("microsaccade_require_snippet_within_source_window", True)
        ),
        microsaccade_reject_extra_events=bool(identity.get("microsaccade_reject_extra_events", True)),
        microsaccade_max_snippet_rms_deg=float(identity.get("microsaccade_max_snippet_rms_deg", 0.0)),
        microsaccade_max_snippet_radius_deg=float(identity.get("microsaccade_max_snippet_radius_deg", 0.0)),
        microsaccade_max_snippet_path_length_deg=float(identity.get("microsaccade_max_snippet_path_length_deg", 0.0)),
        microsaccade_speed_threshold_dps=identity.get("microsaccade_speed_threshold_dps", None),
        microsaccade_threshold_z=float(identity.get("microsaccade_threshold_z", source_config.get("microsaccade_threshold_z", 6.0))),
        microsaccade_pad_frames=int(identity.get("microsaccade_pad_frames", source_config.get("microsaccade_pad_frames", 1))),
        microsaccade_dedup_tolerance_frames=int(identity.get("microsaccade_dedup_tolerance_frames", 3)),
    )


def condition_specs_from_run(run_dir: Path, scales: list[float]) -> list[dict[str, Any]]:
    identity = load_json(run_dir / "run_metadata.json")["identity"]
    wanted = {scale_token(scale) for scale in scales}
    specs = []
    for spec in identity["condition_specs"]:
        motion_scale = float(spec.get("motion_scale", spec.get("across_scale", 0.0)))
        if scale_token(motion_scale) in wanted:
            item = dict(spec)
            item["condition_index"] = len(specs)
            specs.append(item)
    if len(specs) != len(wanted):
        found = {scale_token(float(spec.get("motion_scale", spec.get("across_scale", 0.0)))) for spec in specs}
        missing = sorted(wanted.difference(found))
        raise ValueError(f"Missing requested scales in run metadata: {missing}")
    return sorted(specs, key=lambda row: float(row.get("motion_scale", row.get("across_scale", 0.0))))


def representative_unit_table(args: argparse.Namespace, stats: dict[str, np.ndarray]) -> pd.DataFrame:
    selected_path = (
        Path(args.selected_units_csv)
        if args.selected_units_csv is not None
        else Path(args.run_dir)
        / "bimodal_unit_curve_groups"
        / "example_activation_maps"
        / "selected_example_units.csv"
    )
    if not selected_path.exists():
        selected_path = Path(args.run_dir) / "bimodal_unit_curve_groups" / "bimodal_unit_curve_groups.csv"
    units = pd.read_csv(selected_path)
    if "curve_group" not in units.columns:
        raise ValueError(f"{selected_path} must contain curve_group.")
    if "unit_index" not in units.columns:
        raise ValueError(f"{selected_path} must contain unit_index.")
    if "example_score" not in units.columns:
        units = units.copy()
        units["example_score"] = pd.to_numeric(units.get("absolute_dynamic_range", 0.0), errors="coerce").fillna(0.0)
    rows = []
    for group in ["large_scale_preferring", "small_scale_preferring"]:
        sub = units[units["curve_group"].astype(str) == group].copy()
        sub["example_score"] = pd.to_numeric(sub["example_score"], errors="coerce").fillna(0.0)
        sub = sub.sort_values("example_score", ascending=False).head(max(1, int(args.examples_per_group)))
        rows.append(sub)
    out = pd.concat(rows, axis=0, ignore_index=True)
    motion_scale = np.asarray(stats["motion_scale"], dtype=np.float64)
    bits = np.asarray(stats["unit_time_resolved_bits_per_movie"], dtype=np.float64)
    movie_source_row = np.asarray(stats["movie_source_row"], dtype=int)
    choice_rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        unit = int(row["unit_index"])
        curves = bits[:, :, unit].T
        target = np.nanmean(curves, axis=0)
        target_std = float(np.nanstd(target))
        if not np.isfinite(target_std) or target_std <= EPS:
            target_z = target - float(np.nanmean(target))
        else:
            target_z = (target - float(np.nanmean(target))) / target_std
        scores = []
        for movie_idx, curve in enumerate(curves):
            finite = np.isfinite(curve) & np.isfinite(target_z)
            if int(np.sum(finite)) < 3:
                continue
            curve_std = float(np.nanstd(curve[finite]))
            if not np.isfinite(curve_std) or curve_std <= EPS:
                curve_z = curve[finite] - float(np.nanmean(curve[finite]))
            else:
                curve_z = (curve[finite] - float(np.nanmean(curve[finite]))) / curve_std
            mse = float(np.nanmean((curve_z - target_z[finite]) ** 2))
            dynamic_range = float(np.nanmax(curve[finite]) - np.nanmin(curve[finite]))
            scores.append((mse, -dynamic_range, int(movie_idx), curve))
        if not scores:
            raise ValueError(f"No finite per-movie SSI curves for unit {unit}.")
        mse, neg_range, movie_idx, curve = sorted(scores, key=lambda item: (item[0], item[1], item[2]))[0]
        payload = row.to_dict()
        payload.update(
            {
                "representative_movie_index": int(movie_idx),
                "representative_source_row": int(movie_source_row[movie_idx]),
                "representative_curve_mse_to_mean_z": float(mse),
                "representative_curve_dynamic_range": float(-neg_range),
            }
        )
        for scale, value in zip(motion_scale, curve, strict=True):
            payload[f"representative_movie_ssi_at_scale_{scale_token(float(scale))}"] = float(value)
        choice_rows.append(payload)
    return pd.DataFrame(choice_rows)


def compute_unit_terms(movie: np.ndarray, unit_pos: int, *, bin_seconds: float) -> dict[str, np.ndarray | float]:
    unit_movie = np.maximum(np.asarray(movie[:, int(unit_pos)], dtype=np.float64), 0.0)
    rbar = np.mean(unit_movie, axis=(1, 2))
    gain = unit_movie / (rbar[:, None, None] + EPS)
    bits_t = np.mean(gain * np.log2(gain + EPS), axis=(1, 2))
    numerator_t = bits_t * rbar * float(bin_seconds)
    denominator = float(np.sum(rbar * float(bin_seconds)))
    bits = float(np.sum(numerator_t) / max(denominator, EPS))
    return {
        "instantaneous_bits": bits_t.astype(np.float32),
        "mean_rate": rbar.astype(np.float32),
        "ssi_numerator": numerator_t.astype(np.float32),
        "unit_bits_per_spike": bits,
    }


def render_representative_movies(
    args: argparse.Namespace,
    trials: pd.DataFrame,
    unit_table: pd.DataFrame,
    specs: list[dict[str, Any]],
    *,
    bin_seconds: float,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    selected_units = [int(v) for v in unit_table["unit_index"].to_list()]
    movie_indices = sorted({int(v) for v in unit_table["representative_movie_index"].to_list()})
    view = load_population_view(version_name=str(args.rr100_version))
    scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(args.batch_size), empty_cache_every_batch=True)
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    source_args = source_args_from_run(Path(args.run_dir))
    rotation_deg = int(load_json(Path(args.run_dir) / "run_metadata.json")["identity"].get("stimulus_rotation_deg", 0))
    rendered: dict[int, dict[str, Any]] = {}
    tile_rows: list[dict[str, Any]] = []
    unit_positions = {int(unit): pos for pos, unit in enumerate(selected_units)}
    for movie_order, movie_idx in enumerate(movie_indices, start=1):
        trial = trials.iloc[int(movie_idx)]
        patch, patch_meta = _extract_patch(trial, canvas_cache=canvas_cache, patch_size_px=int(load_json(Path(args.run_dir) / "run_metadata.json")["identity"].get("patch_size_px", 540)))
        source_trace = np.asarray(trial["source_trace"], dtype=np.float32)
        if rotation_deg:
            patch = rotate_patch_gaze_frame(patch, rotation_deg)
            source_trace = rotate_trace_xy(source_trace, rotation_deg)
        event_mask = trial_event_scale_mask(trial, int(source_trace.shape[0]))
        original_axis_deg = float(trial[str(source_args.axis_column)])
        axis_deg = rotated_axis_deg(original_axis_deg, rotation_deg)
        condition_maps: list[np.ndarray] = []
        condition_traces: list[np.ndarray] = []
        condition_terms: list[dict[int, dict[str, np.ndarray | float]]] = []
        for cond_order, spec in enumerate(specs, start=1):
            if bool(spec.get("is_static_baseline", False)) and event_mask is None:
                trace = np.zeros_like(source_trace, dtype=np.float32)
            else:
                trace, _trace_meta = combined_axis_trace(
                    source_trace,
                    axis_deg=axis_deg,
                    along_scale=float(spec["along_scale"]),
                    across_scale=float(spec["across_scale"]),
                    event_scale_mask=event_mask,
                )
            print(
                f"[bimodal-instant-maps] movie {movie_order}/{len(movie_indices)} "
                f"condition {cond_order}/{len(specs)} source_row={int(trial['source_row'])} "
                f"{spec['condition_label']}",
                flush=True,
            )
            full_map = rate_map_for_trace(scorer, patch, trace)
            full_map = _align_response_to_trace(full_map, int(source_args.n_timepoints))
            rr100_map = apply_population_view(full_map, view).astype(np.float32, copy=False)
            unit_map = rr100_map[:, selected_units].astype(np.float32, copy=False)
            terms = {
                int(unit): compute_unit_terms(unit_map, unit_positions[int(unit)], bin_seconds=bin_seconds)
                for unit in selected_units
            }
            condition_maps.append(unit_map)
            condition_traces.append(np.asarray(trace, dtype=np.float32))
            condition_terms.append(terms)
            del full_map, rr100_map, unit_map
        rendered[int(movie_idx)] = {
            "maps": np.stack(condition_maps, axis=0).astype(np.float32),
            "traces": np.stack(condition_traces, axis=0).astype(np.float32),
            "terms": condition_terms,
            "trial": trial,
            "event_mask": event_mask.astype(bool, copy=False) if event_mask is not None else np.zeros((int(source_args.n_timepoints),), dtype=bool),
            "patch_meta": patch_meta,
            "selected_units": selected_units,
        }
        for unit in selected_units:
            reference = np.asarray(condition_terms[0][int(unit)]["ssi_numerator"], dtype=np.float64)
            all_deltas = np.stack(
                [
                    np.asarray(terms[int(unit)]["ssi_numerator"], dtype=np.float64) - reference
                    for terms in condition_terms
                ],
                axis=0,
            )
            reference_peak_frame = int(np.nanargmax(np.nanmax(np.abs(all_deltas), axis=0)))
            for cond_idx, spec in enumerate(specs):
                terms = condition_terms[cond_idx][int(unit)]
                contribution = np.asarray(terms["ssi_numerator"], dtype=np.float64)
                delta = contribution - reference
                if cond_idx == 0:
                    peak_frame = reference_peak_frame
                else:
                    peak_frame = int(np.nanargmax(np.abs(delta))) if np.isfinite(delta).any() else 0
                tile_rows.append(
                    {
                        "unit_index": int(unit),
                        "unit_label": f"u{int(unit):03d}",
                        "movie_index": int(movie_idx),
                        "source_row": int(trial["source_row"]),
                        "condition_index": int(cond_idx),
                        "condition_label": str(spec["condition_label"]),
                        "motion_scale": float(spec.get("motion_scale", spec.get("across_scale", 0.0))),
                        "peak_contribution_frame": int(peak_frame),
                        "peak_frame_selection_contract": (
                            "For non-0x conditions, frame with largest absolute per-frame SSI numerator "
                            "change relative to the 0x event-removed drift reference. For 0x, frame with "
                            "largest across-condition absolute change relative to 0x."
                        ),
                        "peak_instantaneous_ssi_bits_per_spike": float(np.asarray(terms["instantaneous_bits"])[peak_frame]),
                        "peak_mean_rate": float(np.asarray(terms["mean_rate"])[peak_frame]),
                        "peak_ssi_numerator": float(np.asarray(terms["ssi_numerator"])[peak_frame]),
                        "peak_ssi_numerator_delta_vs_0x": float(delta[peak_frame]),
                        "movie_unit_ssi_bits_per_spike": float(terms["unit_bits_per_spike"]),
                    }
                )
    return rendered, tile_rows


def unit_color(group: str) -> str:
    return "#1f77b4" if str(group) == "large_scale_preferring" else "#d62728"


def plot_peak_map_sheet(
    out_dir: Path,
    rendered: dict[int, dict[str, Any]],
    unit_table: pd.DataFrame,
    tile_rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    stats: dict[str, np.ndarray],
    *,
    dpi: int,
    vmin_percentile: float,
    vmax_percentile: float,
) -> tuple[Path, Path]:
    tile_df = pd.DataFrame(tile_rows)
    motion_scale = np.asarray(stats["motion_scale"], dtype=np.float64)
    bits = np.asarray(stats["unit_time_resolved_bits_per_movie"], dtype=np.float64)
    n_rows = int(unit_table.shape[0])
    n_cols = len(specs)
    fig = plt.figure(figsize=(1.55 * n_cols + 3.4, max(6.5, 1.38 * n_rows + 1.6)))
    grid = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1.85, *([1.0] * n_cols)], hspace=0.12, wspace=0.04)
    selected_units = [int(v) for v in unit_table["unit_index"].to_list()]
    unit_positions = {int(unit): pos for pos, unit in enumerate(selected_units)}
    for row_idx, unit_row in unit_table.reset_index(drop=True).iterrows():
        unit = int(unit_row["unit_index"])
        group = str(unit_row["curve_group"])
        color = unit_color(group)
        movie_idx = int(unit_row["representative_movie_index"])
        movie = rendered[movie_idx]
        unit_pos = unit_positions[unit]
        ax_curve = fig.add_subplot(grid[row_idx, 0])
        mean_curve = np.nanmean(bits[:, :, unit], axis=1)
        movie_curve = bits[:, movie_idx, unit]
        ax_curve.plot(motion_scale, mean_curve, color="0.65", linewidth=1.0, label="all snippets")
        ax_curve.plot(motion_scale, movie_curve, color=color, marker="o", markersize=2.7, linewidth=1.25, label="shown snippet")
        ax_curve.axvline(1.0, color="0.55", linestyle=":", linewidth=0.8)
        ax_curve.grid(True, color="0.9", linewidth=0.55)
        ax_curve.set_ylabel(f"u{unit:03d}", color=color, rotation=0, ha="right", va="center", labelpad=24, fontsize=8.0)
        if row_idx == n_rows - 1:
            ax_curve.set_xlabel("scale", fontsize=7.5)
        else:
            ax_curve.set_xticklabels([])
        ax_curve.tick_params(labelsize=6.3)
        title = (
            f"{str(unit_row.get('curve_group_label', group)).replace('_', ' ')} | "
            f"{str(unit_row.get('sf_group_label', '')).split('(')[0].strip()}"
        )
        ax_curve.set_title(title, color=color, fontsize=7.8, pad=2.5)
        if row_idx == 0:
            ax_curve.legend(frameon=False, fontsize=5.7, loc="best")

        row_images: list[np.ndarray] = []
        row_peaks: list[int] = []
        for cond_idx, spec in enumerate(specs):
            peak = int(
                tile_df[
                    (tile_df["unit_index"].astype(int) == unit)
                    & (tile_df["movie_index"].astype(int) == movie_idx)
                    & (tile_df["condition_index"].astype(int) == cond_idx)
                ].iloc[0]["peak_contribution_frame"]
            )
            row_peaks.append(peak)
            row_images.append(movie["maps"][cond_idx, peak, unit_pos])
        vmin, vmax = image_scale(row_images, float(vmin_percentile), float(vmax_percentile))
        for cond_idx, spec in enumerate(specs):
            ax = fig.add_subplot(grid[row_idx, cond_idx + 1])
            image = row_images[cond_idx]
            ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(str(spec["condition_label"]), fontsize=7.7)
            peak_frame = row_peaks[cond_idx]
            event_mask = np.asarray(movie["event_mask"], dtype=bool)
            tag = "event" if peak_frame < event_mask.size and bool(event_mask[peak_frame]) else "outside"
            ax.text(
                0.04,
                0.05,
                f"t={peak_frame}\n{tag}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                color="white",
                fontsize=5.4,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "black", "alpha": 0.45, "linewidth": 0},
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.9)
                spine.set_edgecolor(color)
    fig.suptitle(
        "Bimodal microsaccade groups: instantaneous maps from representative single snippets\n"
        "Each tile is the frame with the largest per-frame SSI-numerator change versus 0x; no trajectory or movie averaged maps",
        fontsize=11.2,
        y=0.996,
    )
    png = out_dir / "example_units_instantaneous_peak_contribution_maps.png"
    pdf = out_dir / "example_units_instantaneous_peak_contribution_maps.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_contribution_timecourses(
    out_dir: Path,
    rendered: dict[int, dict[str, Any]],
    unit_table: pd.DataFrame,
    specs: list[dict[str, Any]],
    *,
    dpi: int,
) -> tuple[Path, Path]:
    selected_units = [int(v) for v in unit_table["unit_index"].to_list()]
    unit_positions = {int(unit): pos for pos, unit in enumerate(selected_units)}
    del unit_positions
    n_rows = int(unit_table.shape[0])
    fig, axes = plt.subplots(n_rows, 1, figsize=(9.2, max(6.5, 1.25 * n_rows)), sharex=True)
    if n_rows == 1:
        axes = np.asarray([axes])
    cmap = plt.get_cmap("viridis")
    scales = np.asarray([float(spec.get("motion_scale", spec.get("across_scale", 0.0))) for spec in specs], dtype=float)
    norm = plt.Normalize(vmin=float(np.nanmin(scales)), vmax=float(np.nanmax(scales)))
    for ax, (_, unit_row) in zip(axes, unit_table.reset_index(drop=True).iterrows(), strict=True):
        unit = int(unit_row["unit_index"])
        group = str(unit_row["curve_group"])
        color = unit_color(group)
        movie_idx = int(unit_row["representative_movie_index"])
        movie = rendered[movie_idx]
        event_mask = np.asarray(movie["event_mask"], dtype=bool)
        event_frames = np.flatnonzero(event_mask)
        for cond_idx, spec in enumerate(specs):
            scale = float(spec.get("motion_scale", spec.get("across_scale", 0.0)))
            terms = movie["terms"][cond_idx][unit]
            contribution = np.asarray(terms["ssi_numerator"], dtype=np.float64)
            ax.plot(
                np.arange(contribution.size),
                contribution,
                color=cmap(norm(scale)),
                linewidth=1.25,
                alpha=0.92,
                label=f"{scale:g}x",
            )
        if event_frames.size:
            ax.axvspan(float(event_frames[0]) - 0.5, float(event_frames[-1]) + 0.5, color=color, alpha=0.10, linewidth=0)
        ax.set_ylabel(f"u{unit:03d}", color=color, rotation=0, ha="right", va="center", labelpad=24, fontsize=8)
        ax.grid(True, color="0.9", linewidth=0.55)
        ax.tick_params(labelsize=6.7)
        ax.set_title(
            f"{str(unit_row.get('curve_group_label', group)).replace('_', ' ')} | representative movie {movie_idx}",
            color=color,
            fontsize=8,
            pad=2.5,
        )
    axes[-1].set_xlabel("time bin")
    fig.text(0.012, 0.5, "per-frame SSI numerator", rotation=90, va="center", ha="center", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="event scale", frameon=False, loc="center right", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    fig.suptitle(
        "Representative-snippet per-frame SSI contribution traces\n"
        "Shaded region is the resampled detected microsaccade event mask; outside-event drift is retained at 1x",
        fontsize=11.2,
        y=0.996,
    )
    png = out_dir / "example_units_per_frame_ssi_contribution_traces.png"
    pdf = out_dir / "example_units_per_frame_ssi_contribution_traces.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def cache_path(out_dir: Path) -> Path:
    return out_dir / "instantaneous_example_maps_cache.npz"


def save_render_cache(path: Path, rendered: dict[int, dict[str, Any]], unit_table: pd.DataFrame, specs: list[dict[str, Any]]) -> None:
    movie_indices = np.asarray(sorted(rendered), dtype=np.int32)
    maps = np.stack([rendered[int(idx)]["maps"] for idx in movie_indices], axis=0).astype(np.float32)
    event_masks = np.stack([rendered[int(idx)]["event_mask"] for idx in movie_indices], axis=0).astype(bool)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        movie_indices=movie_indices,
        maps=maps,
        event_masks=event_masks,
        selected_units=unit_table["unit_index"].to_numpy(dtype=np.int32),
        condition_label=np.asarray([str(spec["condition_label"]) for spec in specs]),
        motion_scale=np.asarray([float(spec.get("motion_scale", spec.get("across_scale", 0.0))) for spec in specs], dtype=np.float32),
        cache_identity_json=np.asarray(
            [
                identity_text(
                    {
                        "analysis": "backimage_microsaccade_bimodal_instantaneous_maps",
                        "movie_indices": movie_indices.tolist(),
                        "selected_units": unit_table["unit_index"].astype(int).to_list(),
                        "condition_specs": specs,
                    }
                )
            ]
        ),
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir / "bimodal_unit_curve_groups" / "example_instantaneous_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = load_npz(run_dir / "cache" / "backimage_contour_axis_rr100_spatial_ssi_cache.npz")
    source_args = source_args_from_run(run_dir)
    trials, source_meta = select_source_trials(source_args)
    scales = parse_float_list(str(args.scales))
    specs = condition_specs_from_run(run_dir, scales)
    unit_table = representative_unit_table(args, stats)
    bin_seconds = float(load_json(run_dir / "run_metadata.json").get("bin_seconds_effective", 1.0 / 120.0))
    selected_csv = out_dir / "selected_example_units_with_representative_movies.csv"
    unit_table.to_csv(selected_csv, index=False)
    rendered, tile_rows = render_representative_movies(args, trials, unit_table, specs, bin_seconds=bin_seconds)
    write_csv_rows(out_dir / "instantaneous_peak_map_tiles.csv", tile_rows)
    save_render_cache(cache_path(out_dir), rendered, unit_table, specs)
    peak_png, peak_pdf = plot_peak_map_sheet(
        out_dir,
        rendered,
        unit_table,
        tile_rows,
        specs,
        stats,
        dpi=int(args.dpi),
        vmin_percentile=float(args.map_vmin_percentile),
        vmax_percentile=float(args.map_vmax_percentile),
    )
    trace_png, trace_pdf = plot_contribution_timecourses(out_dir, rendered, unit_table, specs, dpi=int(args.dpi))
    write_json(
        out_dir / "instantaneous_example_map_metadata.json",
        {
            "analysis": "backimage_microsaccade_bimodal_instantaneous_maps",
            "run_dir": run_dir,
            "source_meta": source_meta,
            "selected_units_csv": selected_csv,
            "tile_csv": out_dir / "instantaneous_peak_map_tiles.csv",
            "peak_map_png": peak_png,
            "peak_map_pdf": peak_pdf,
            "timecourse_png": trace_png,
            "timecourse_pdf": trace_pdf,
            "map_contract": (
                "Each displayed activation tile is a single instantaneous RR100 map from one representative "
                "microsaccade snippet. The frame is selected by the largest absolute per-frame SSI-numerator "
                "change relative to the 0x event-removed drift reference for that unit/condition/movie. "
                "No trajectory-averaged or movie-averaged activation maps are displayed."
            ),
        },
    )
    print(f"Wrote {peak_png}")
    print(f"Wrote {trace_png}")


if __name__ == "__main__":
    main()
