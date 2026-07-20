#!/usr/bin/env python3
"""Plot RR100 population response traces around detected BackImage microsaccades."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    DEFAULT_AXIS_RUN_DIR,
    RR100_MOVIE_MEDOID_VERSION,
    CanonicalTwinScorer,
    _align_response_to_trace,
    _extract_patch,
    apply_population_view,
    combined_axis_trace,
    condition_specs,
    load_population_view,
    rate_map_for_trace,
    select_source_trials,
    trial_event_scale_mask,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SF_GROUPS_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
    "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)
DEFAULT_OUT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_microsaccade_population_response_traces_n40_sample32_v1"
)
GROUP_COLORS = {
    "all_rr100": "#222222",
    "low_sf": "#1f77b4",
    "high_sf": "#d62728",
}
SCALE_COLORS = {
    0.0: "#8c8c8c",
    0.25: "#66c2a5",
    0.5: "#3288bd",
    0.75: "#5e4fa2",
    1.0: "#1f77b4",
    1.5: "#984ea3",
    2.0: "#f28e2b",
    3.0: "#d62728",
}
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-run-dir", type=Path, default=DEFAULT_AXIS_RUN_DIR)
    parser.add_argument("--sf-groups-csv", type=Path, default=DEFAULT_SF_GROUPS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rr100-version", default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--scales", default="0,1,2,3")
    parser.add_argument("--max-snippets", type=int, default=32)
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--axis-column", default="image_edge_axis_deg")
    parser.add_argument("--microsaccade-pre-frames", type=int, default=8)
    parser.add_argument("--microsaccade-post-frames", type=int, default=36)
    parser.add_argument("--microsaccade-max-source-windows", type=int, default=0)
    parser.add_argument("--microsaccade-min-amplitude-arcmin", type=float, default=0.0)
    parser.add_argument("--microsaccade-max-amplitude-arcmin", type=float, default=60.0)
    parser.add_argument("--microsaccade-amplitude-sd-filter", type=float, default=0.0)
    parser.add_argument(
        "--microsaccade-trace-mode",
        choices=(
            "full_snippet",
            "core_zero_rest",
            "padded_event_zero_rest",
            "core_scaled_full_snippet",
            "padded_event_scaled_full_snippet",
        ),
        default="full_snippet",
    )
    parser.add_argument("--microsaccade-speed-threshold-dps", type=float, default=None)
    parser.add_argument("--microsaccade-threshold-z", type=float, default=6.0)
    parser.add_argument("--microsaccade-pad-frames", type=int, default=1)
    parser.add_argument("--microsaccade-dedup-tolerance-frames", type=int, default=3)
    parser.add_argument("--microsaccade-reject-extra-events", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--microsaccade-require-snippet-within-source-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--microsaccade-max-snippet-rms-deg", type=float, default=0.0)
    parser.add_argument("--microsaccade-max-snippet-radius-deg", type=float, default=0.0)
    parser.add_argument("--microsaccade-max-snippet-path-length-deg", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one scale is required.")
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


def sem(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(values)
    n = np.sum(finite, axis=axis)
    sd = np.nanstd(values, axis=axis, ddof=1)
    return sd / np.sqrt(np.maximum(n, 1))


def runner_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        axis_run_dir=Path(args.axis_run_dir),
        selected_windows_csv=None,
        trial_source_mode="microsaccade_snippets",
        source_trace_scale=1.0,
        source_trace_prior_family="axis_edge_parallel",
        axis_column=str(args.axis_column),
        trial_start=0,
        max_trials=int(args.max_snippets),
        n_timepoints=int(args.n_timepoints),
        microsaccade_pre_frames=int(args.microsaccade_pre_frames),
        microsaccade_post_frames=int(args.microsaccade_post_frames),
        microsaccade_max_source_windows=int(args.microsaccade_max_source_windows),
        microsaccade_min_amplitude_arcmin=float(args.microsaccade_min_amplitude_arcmin),
        microsaccade_max_amplitude_arcmin=float(args.microsaccade_max_amplitude_arcmin),
        microsaccade_amplitude_sd_filter=float(args.microsaccade_amplitude_sd_filter),
        microsaccade_trace_mode=str(args.microsaccade_trace_mode),
        microsaccade_speed_threshold_dps=args.microsaccade_speed_threshold_dps,
        microsaccade_threshold_z=float(args.microsaccade_threshold_z),
        microsaccade_pad_frames=int(args.microsaccade_pad_frames),
        microsaccade_dedup_tolerance_frames=int(args.microsaccade_dedup_tolerance_frames),
        microsaccade_reject_extra_events=bool(args.microsaccade_reject_extra_events),
        microsaccade_require_snippet_within_source_window=bool(args.microsaccade_require_snippet_within_source_window),
        microsaccade_max_snippet_rms_deg=float(args.microsaccade_max_snippet_rms_deg),
        microsaccade_max_snippet_radius_deg=float(args.microsaccade_max_snippet_radius_deg),
        microsaccade_max_snippet_path_length_deg=float(args.microsaccade_max_snippet_path_length_deg),
    )


def load_group_masks(path: Path, n_units: int) -> dict[str, np.ndarray]:
    units = pd.read_csv(path)
    required = {"unit_index", "sf_group"}
    missing = sorted(required.difference(units.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    masks: dict[str, np.ndarray] = {"all_rr100": np.arange(n_units, dtype=int)}
    for group in ["low_sf", "high_sf"]:
        idx = pd.to_numeric(units.loc[units["sf_group"].astype(str).eq(group), "unit_index"], errors="coerce")
        idx = idx[np.isfinite(idx)].astype(int).to_numpy()
        idx = idx[(idx >= 0) & (idx < int(n_units))]
        masks[group] = np.asarray(sorted(set(int(v) for v in idx)), dtype=int)
    return masks


def relative_time_ms(row: pd.Series, n_timepoints: int) -> np.ndarray:
    duration_s = float(row["snippet_duration_s"])
    onset = float(row["microsaccade_event_onset_frame_resampled"])
    frame_dt_ms = 1000.0 * duration_s / float(max(1, int(n_timepoints) - 1))
    return (np.arange(int(n_timepoints), dtype=np.float64) - onset) * frame_dt_ms


def event_window_ms(row: pd.Series, n_timepoints: int) -> tuple[float, float, float, float]:
    duration_s = float(row["snippet_duration_s"])
    frame_dt_ms = 1000.0 * duration_s / float(max(1, int(n_timepoints) - 1))
    onset = float(row["microsaccade_event_onset_frame_resampled"])
    offset = float(row["microsaccade_event_offset_frame_resampled"])
    snippet_start = int(row["snippet_global_start"])
    raw_den = float(max(1, int(row["snippet_n_samples"]) - 1))
    padded_onset = (int(row["microsaccade_event_onset_global_padded"]) - snippet_start) * float(n_timepoints - 1) / raw_den
    padded_offset = (int(row["microsaccade_event_offset_global_padded"]) - snippet_start) * float(n_timepoints - 1) / raw_den
    return (
        0.0,
        (offset - onset) * frame_dt_ms,
        (padded_onset - onset) * frame_dt_ms,
        (padded_offset - onset) * frame_dt_ms,
    )


def compute_trace_cache(args: argparse.Namespace, cache_path: Path) -> dict[str, Any]:
    scales = parse_float_list(str(args.scales))
    specs = condition_specs(
        scales,
        along_scale=1.0,
        along_scales=[],
        condition_pairs=None,
        include_static_baseline=True,
        sweep_mode="isotropic",
        zero_motion_is_static_baseline=not str(args.microsaccade_trace_mode).endswith("_scaled_full_snippet"),
    )
    trials, source_meta = select_source_trials(runner_namespace(args))
    trials = trials.reset_index(drop=True)
    rr100 = load_population_view(version_name=str(args.rr100_version))
    group_masks = load_group_masks(Path(args.sf_groups_csv), int(rr100.n_units))
    scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(args.batch_size), empty_cache_every_batch=True)
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}

    trace_rows: list[dict[str, Any]] = []
    eye_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    total = int(trials.shape[0]) * len(specs)
    done = 0
    for movie_index, (_, trial) in enumerate(trials.iterrows()):
        patch, _patch_meta = _extract_patch(trial, canvas_cache=canvas_cache, patch_size_px=int(args.patch_size_px))
        source_trace = np.asarray(trial["source_trace"], dtype=np.float32)
        event_scale_mask = trial_event_scale_mask(trial, int(source_trace.shape[0]))
        axis_deg = float(trial[str(args.axis_column)])
        time_ms = relative_time_ms(trial, int(args.n_timepoints))
        onset_ms, offset_ms, padded_onset_ms, padded_offset_ms = event_window_ms(trial, int(args.n_timepoints))
        event_rows.append(
            {
                "movie_index": int(movie_index),
                "trial_id": int(trial["trial_id"]),
                "source_row": int(trial["source_row"]),
                "session": str(trial["session"]),
                "trial_idx": int(trial["trial_idx"]),
                "event_onset_ms": float(onset_ms),
                "event_offset_ms": float(offset_ms),
                "padded_event_onset_ms": float(padded_onset_ms),
                "padded_event_offset_ms": float(padded_offset_ms),
                "microsaccade_amplitude_arcmin": float(trial["microsaccade_amplitude_arcmin"]),
                "microsaccade_peak_speed_dps": float(trial["microsaccade_peak_speed_dps"]),
                "snippet_duration_s": float(trial["snippet_duration_s"]),
                "snippet_n_samples": int(trial["snippet_n_samples"]),
            }
        )
        onset_idx = int(np.clip(round(float(trial["microsaccade_event_onset_frame_resampled"])), 0, source_trace.shape[0] - 1))
        displacement_arcmin = np.linalg.norm(source_trace - source_trace[onset_idx][None, :], axis=1) * 60.0
        for frame_index, (t_ms, disp) in enumerate(zip(time_ms, displacement_arcmin, strict=True)):
            eye_rows.append(
                {
                    "movie_index": int(movie_index),
                    "frame_index": int(frame_index),
                    "time_ms": float(t_ms),
                    "displacement_from_onset_arcmin": float(disp),
                }
            )
        for spec in specs:
            done += 1
            if bool(spec["is_static_baseline"]) and event_scale_mask is None:
                trace = np.zeros_like(source_trace, dtype=np.float32)
            else:
                trace, _trace_meta = combined_axis_trace(
                    source_trace,
                    axis_deg=axis_deg,
                    along_scale=float(spec["along_scale"]),
                    across_scale=float(spec["across_scale"]),
                    event_scale_mask=event_scale_mask,
                )
            print(
                f"[microsaccade-response-traces] {done}/{total} "
                f"movie={movie_index} condition={spec['condition_id']}",
                flush=True,
            )
            full_map = rate_map_for_trace(scorer, patch, trace)
            full_map = _align_response_to_trace(full_map, int(args.n_timepoints))
            rr100_map = apply_population_view(full_map, rr100)
            rbar = np.mean(rr100_map.reshape(rr100_map.shape[0], rr100_map.shape[1], -1), axis=2)
            for group, unit_idx in group_masks.items():
                if unit_idx.size == 0:
                    continue
                values = np.nanmean(rbar[:, unit_idx], axis=1)
                for frame_index, (t_ms, value) in enumerate(zip(time_ms, values, strict=True)):
                    trace_rows.append(
                        {
                            "movie_index": int(movie_index),
                            "condition_id": str(spec["condition_id"]),
                            "condition_label": str(spec["condition_label"]),
                            "motion_scale": float(spec["motion_scale"]),
                            "is_static_baseline": bool(spec["is_static_baseline"]),
                            "population_group": str(group),
                            "n_units": int(unit_idx.size),
                            "frame_index": int(frame_index),
                            "time_ms": float(t_ms),
                            "mean_rate": float(value),
                        }
                    )
            del full_map, rr100_map, rbar

    traces = pd.DataFrame(trace_rows)
    eye = pd.DataFrame(eye_rows)
    events = pd.DataFrame(event_rows)
    traces.to_csv(cache_path.parent / "population_response_traces_by_snippet.csv", index=False)
    eye.to_csv(cache_path.parent / "microsaccade_eye_displacement_traces.csv", index=False)
    events.to_csv(cache_path.parent / "microsaccade_event_timing.csv", index=False)
    np.savez_compressed(
        cache_path,
        trace_rows=traces.to_records(index=False),
        eye_rows=eye.to_records(index=False),
        event_rows=events.to_records(index=False),
    )
    return {
        "traces": traces,
        "eye": eye,
        "events": events,
        "source_meta": source_meta,
        "specs": specs,
    }


def load_trace_cache(cache_path: Path) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=True) as data:
        return {
            "traces": pd.DataFrame.from_records(data["trace_rows"]),
            "eye": pd.DataFrame.from_records(data["eye_rows"]),
            "events": pd.DataFrame.from_records(data["event_rows"]),
            "source_meta": {},
            "specs": [],
        }


def interpolate_by_movie(df: pd.DataFrame, value_col: str, grid_ms: np.ndarray) -> np.ndarray:
    rows = []
    for _, sub in df.groupby("movie_index", sort=True):
        sub = sub.sort_values("time_ms")
        x = sub["time_ms"].to_numpy(dtype=np.float64)
        y = sub[value_col].to_numpy(dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < 2:
            rows.append(np.full_like(grid_ms, np.nan, dtype=np.float64))
            continue
        rows.append(np.interp(grid_ms, x[valid], y[valid], left=np.nan, right=np.nan))
    return np.vstack(rows) if rows else np.zeros((0, grid_ms.size), dtype=np.float64)


def summarize_traces(traces: pd.DataFrame, eye: pd.DataFrame, grid_ms: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    reference_lookup: dict[tuple[str, int], np.ndarray] = {}
    reference_rows = traces[
        traces["is_static_baseline"].astype(bool)
        | np.isclose(pd.to_numeric(traces["motion_scale"], errors="coerce").to_numpy(dtype=float), 0.0)
    ].copy()
    for group, group_df in reference_rows.groupby("population_group", sort=False):
        for movie_index, sub in group_df.groupby("movie_index", sort=True):
            reference_lookup[(str(group), int(movie_index))] = np.interp(
                grid_ms,
                sub.sort_values("time_ms")["time_ms"].to_numpy(dtype=np.float64),
                sub.sort_values("time_ms")["mean_rate"].to_numpy(dtype=np.float64),
                left=np.nan,
                right=np.nan,
            )
    for (condition_id, label, scale, group), sub in traces.groupby(
        ["condition_id", "condition_label", "motion_scale", "population_group"],
        sort=False,
    ):
        mat = interpolate_by_movie(sub, "mean_rate", grid_ms)
        delta = np.full_like(mat, np.nan)
        for row_idx, movie_index in enumerate(sorted(sub["movie_index"].unique())):
            reference = reference_lookup.get((str(group), int(movie_index)))
            if reference is not None:
                delta[row_idx] = mat[row_idx] - reference
        for i, t_ms in enumerate(grid_ms):
            values = mat[:, i]
            dvalues = delta[:, i]
            rows.append(
                {
                    "condition_id": str(condition_id),
                    "condition_label": str(label),
                    "motion_scale": float(scale),
                    "population_group": str(group),
                    "time_ms": float(t_ms),
                    "mean_rate_mean": float(np.nanmean(values)),
                    "mean_rate_sem": float(sem(values)),
                    "delta_rate_vs_static_mean": float(np.nanmean(dvalues)),
                    "delta_rate_vs_static_sem": float(sem(dvalues)),
                    "n_snippets": int(np.sum(np.isfinite(values))),
                }
            )
    eye_mat = interpolate_by_movie(eye, "displacement_from_onset_arcmin", grid_ms)
    eye_rows = []
    for i, t_ms in enumerate(grid_ms):
        values = eye_mat[:, i]
        eye_rows.append(
            {
                "time_ms": float(t_ms),
                "displacement_from_onset_arcmin_mean": float(np.nanmean(values)),
                "displacement_from_onset_arcmin_sem": float(sem(values)),
                "n_snippets": int(np.sum(np.isfinite(values))),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(eye_rows)


def plot(summary: pd.DataFrame, eye_summary: pd.DataFrame, events: pd.DataFrame, out_dir: Path, *, dpi: int) -> tuple[Path, Path]:
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=True)
    padded_on = float(np.nanmedian(events["padded_event_onset_ms"].to_numpy(dtype=float)))
    padded_off = float(np.nanmedian(events["padded_event_offset_ms"].to_numpy(dtype=float)))
    core_off = float(np.nanmedian(events["event_offset_ms"].to_numpy(dtype=float)))
    for ax in axes:
        ax.axvspan(padded_on, padded_off, color="0.85", alpha=0.45, linewidth=0, label="padded event" if ax is axes[0] else None)
        ax.axvline(0.0, color="0.25", linestyle="--", linewidth=1.0)
        if abs(core_off) > EPS:
            ax.axvline(core_off, color="0.45", linestyle=":", linewidth=0.9)
        ax.grid(True, color="0.9", linewidth=0.8)

    all_rr100 = summary[summary["population_group"].astype(str).eq("all_rr100")].copy()
    for scale, sub in all_rr100.groupby("motion_scale", sort=True):
        sub = sub.sort_values("time_ms")
        color = SCALE_COLORS.get(float(scale), None)
        label = f"{float(scale):g}x"
        axes[0].plot(sub["time_ms"], sub["mean_rate_mean"], color=color, linewidth=1.9, label=label)
        axes[0].fill_between(
            sub["time_ms"].to_numpy(dtype=float),
            (sub["mean_rate_mean"] - sub["mean_rate_sem"]).to_numpy(dtype=float),
            (sub["mean_rate_mean"] + sub["mean_rate_sem"]).to_numpy(dtype=float),
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        if not np.isclose(float(scale), 0.0):
            axes[1].plot(sub["time_ms"], sub["delta_rate_vs_static_mean"], color=color, linewidth=1.9, label=label)
            axes[1].fill_between(
                sub["time_ms"].to_numpy(dtype=float),
                (sub["delta_rate_vs_static_mean"] - sub["delta_rate_vs_static_sem"]).to_numpy(dtype=float),
                (sub["delta_rate_vs_static_mean"] + sub["delta_rate_vs_static_sem"]).to_numpy(dtype=float),
                color=color,
                alpha=0.12,
                linewidth=0,
            )

    sf = summary[
        summary["population_group"].astype(str).isin(["low_sf", "high_sf"])
        & summary["motion_scale"].isin([1.0, 3.0])
    ].copy()
    for (group, scale), sub in sf.groupby(["population_group", "motion_scale"], sort=True):
        sub = sub.sort_values("time_ms")
        color = GROUP_COLORS.get(str(group), None)
        linestyle = "-" if np.isclose(float(scale), 1.0) else "--"
        label = f"{'low SF' if group == 'low_sf' else 'high SF'} {float(scale):g}x"
        axes[2].plot(sub["time_ms"], sub["delta_rate_vs_static_mean"], color=color, linestyle=linestyle, linewidth=1.9, label=label)

    axes[0].set_ylabel("mean rate")
    axes[0].set_title("All RR100 population response")
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    axes[1].axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("rate minus 0x")
    axes[1].set_title("All RR100 motion-evoked response")
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    ax_eye = axes[2].twinx()
    eye_summary = eye_summary.sort_values("time_ms")
    ax_eye.plot(
        eye_summary["time_ms"],
        eye_summary["displacement_from_onset_arcmin_mean"],
        color="0.55",
        linewidth=1.2,
        alpha=0.75,
        label="eye displacement",
    )
    axes[2].axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    axes[2].set_ylabel("SF group rate minus 0x")
    ax_eye.set_ylabel("eye displacement (arcmin)", color="0.45")
    axes[2].set_xlabel("time from microsaccade onset (ms)")
    axes[2].set_title("Low/high SF response timing at 1x and 3x")
    handles, labels = axes[2].get_legend_handles_labels()
    h2, l2 = ax_eye.get_legend_handles_labels()
    axes[2].legend(handles + h2, labels + l2, frameon=False, ncol=3, fontsize=8, loc="upper left")
    n_snippets = int(np.nanmax(summary["n_snippets"].to_numpy(dtype=float))) if not summary.empty else 0
    fig.suptitle(
        f"BackImage microsaccade-triggered RR100 response traces (n={n_snippets} snippets)\n"
        "vertical line: event onset; gray band: median padded detector event",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    png = out_dir / "backimage_microsaccade_population_response_traces.png"
    pdf = out_dir / "backimage_microsaccade_population_response_traces.pdf"
    fig.savefig(png, dpi=int(dpi))
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "population_response_trace_cache.npz"
    if cache_path.exists() and not bool(args.force):
        payload = load_trace_cache(cache_path)
    else:
        payload = compute_trace_cache(args, cache_path)
    traces = payload["traces"]
    eye = payload["eye"]
    events = payload["events"]
    grid_ms = np.linspace(
        float(np.nanpercentile(traces["time_ms"].to_numpy(dtype=float), 1.0)),
        float(np.nanpercentile(traces["time_ms"].to_numpy(dtype=float), 99.0)),
        int(args.n_timepoints),
    )
    summary, eye_summary = summarize_traces(traces, eye, grid_ms)
    summary_csv = args.out_dir / "population_response_trace_summary.csv"
    eye_summary_csv = args.out_dir / "microsaccade_eye_displacement_summary.csv"
    summary.to_csv(summary_csv, index=False)
    eye_summary.to_csv(eye_summary_csv, index=False)
    png, pdf = plot(summary, eye_summary, events, args.out_dir, dpi=int(args.dpi))
    write_json(
        args.out_dir / "summary.json",
        {
            "analysis": "backimage_microsaccade_population_response_traces",
            "axis_run_dir": args.axis_run_dir,
            "sf_groups_csv": args.sf_groups_csv,
            "rr100_version": str(args.rr100_version),
            "n_timepoints": int(args.n_timepoints),
            "max_snippets": int(args.max_snippets),
            "scales": parse_float_list(str(args.scales)),
            "microsaccade_trace_mode": str(args.microsaccade_trace_mode),
            "microsaccade_amplitude_sd_filter": float(args.microsaccade_amplitude_sd_filter),
            "n_snippets": int(events.shape[0]),
            "median_core_event_offset_ms": float(np.nanmedian(events["event_offset_ms"].to_numpy(dtype=float))),
            "median_padded_event_onset_ms": float(np.nanmedian(events["padded_event_onset_ms"].to_numpy(dtype=float))),
            "median_padded_event_offset_ms": float(np.nanmedian(events["padded_event_offset_ms"].to_numpy(dtype=float))),
            "outputs": {
                "trace_cache": cache_path,
                "trace_rows": args.out_dir / "population_response_traces_by_snippet.csv",
                "eye_rows": args.out_dir / "microsaccade_eye_displacement_traces.csv",
                "event_timing": args.out_dir / "microsaccade_event_timing.csv",
                "summary": summary_csv,
                "eye_summary": eye_summary_csv,
                "png": png,
                "pdf": pdf,
            },
        },
    )
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(summary.groupby(["population_group", "motion_scale"])["n_snippets"].max().to_string())


if __name__ == "__main__":
    main()
