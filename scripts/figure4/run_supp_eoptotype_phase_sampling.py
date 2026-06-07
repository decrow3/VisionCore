#!/usr/bin/env python3
"""Supplemental E-optotype phase-sampling analysis.

This script is intentionally cache-only for the main response metrics. It
consumes E-optotype rate caches produced by
`scripts/temporal_decoding/cache_eoptotype_rates.py` and writes a bounded
supplemental figure focused on deterministic model-side response geometry.

Primary comparison:
  real_fem - stationary_phase_jittered

In this repository, `stationary_phase_jittered` is implemented as the existing
trial-mean stationary condition (`stabilized`): each trace contributes one
matched stationary retinal phase, while real FEM samples a within-trial phase
trajectory. `fixed_center` is retained only as a deterministic-oracle diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from VisionCore.paths import VISIONCORE_ROOT


ORIENTATIONS_DEFAULT = (0, 90, 180, 270)
LOGMARS_DEFAULT = (-0.20, -0.25, -0.30, -0.35, -0.40)
CONDITIONS_DEFAULT = ("real_fem", "stationary_phase_jittered", "fixed_center")
HIRES_THRESHOLD_DEFAULT = 0.35
EPS = 1e-12

CONDITION_CACHE_ALIASES = {
    "real_fem": ("real",),
    "real": ("real",),
    "stationary_phase_jittered": ("stationary_phase_jittered", "stabilized"),
    "trial_mean_stabilized": ("trial_mean_stabilized", "stabilized"),
    "stabilized": ("stabilized",),
    "fixed_center": ("fixed_center",),
}

CONDITION_LABELS = {
    "real_fem": "real FEM",
    "real": "real FEM",
    "stationary_phase_jittered": "stationary phase-jittered",
    "trial_mean_stabilized": "trial-mean stabilized",
    "stabilized": "stationary phase-jittered",
    "scaled_0.1": "micro-FEM 0.1x",
    "scaled_0.05": "micro-FEM 0.05x",
    "fixed_center": "fixed center",
}


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in str(text).split(",") if str(x).strip())


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(float(x)) for x in str(text).split(",") if str(x).strip())


def _parse_csv_strings(text: str) -> tuple[str, ...]:
    return tuple(str(x).strip() for x in str(text).split(",") if str(x).strip())


def _format_logmar(logmar: float) -> str:
    return f"{float(logmar):.2f}"


def _normalize_file_tag(tag: str | None) -> str:
    if tag is None:
        return ""
    tag = str(tag).strip()
    if not tag:
        return ""
    if not tag.startswith("_"):
        tag = "_" + tag
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. ")
    if any(ch not in allowed for ch in tag):
        raise ValueError("--file-tag contains unsupported characters")
    return tag.replace(" ", "_")


def _rate_path(rates_dir: Path, logmar: float, orientation: int, condition: str, hires_threshold: float, file_tag: str = "") -> Path:
    prefix = "rates_hires_lm" if float(logmar) < float(hires_threshold) else "rates_lm"
    return rates_dir / f"{prefix}{_format_logmar(logmar)}_ori{int(orientation)}_{condition}{file_tag}.npz"


def _resolve_rate_path(
    rates_dir: Path,
    logmar: float,
    orientation: int,
    condition: str,
    hires_threshold: float,
    file_tag: str = "",
) -> tuple[Path, str]:
    aliases = CONDITION_CACHE_ALIASES.get(condition, (condition,))
    tried = []
    for cache_condition in aliases:
        path = _rate_path(rates_dir, logmar, orientation, cache_condition, hires_threshold, file_tag=file_tag)
        tried.append(str(path))
        if path.exists():
            return path, cache_condition
    raise FileNotFoundError("Missing cached rates. Tried:\n" + "\n".join(tried))


def _load_rates(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    d = np.load(path, allow_pickle=True)
    trace_indices = None
    if "stim_trace_indices" in d.files:
        trace_indices = np.asarray(d["stim_trace_indices"]).reshape(-1).astype(np.int32)
    return np.asarray(d["rates"], dtype=np.float64), np.asarray(d["lengths"], dtype=np.int32), trace_indices


def _window_mean(rates_padded: np.ndarray, lengths: np.ndarray, window: int, n_trials: int) -> np.ndarray:
    out = np.zeros((n_trials, rates_padded.shape[-1]), dtype=np.float64)
    for i in range(n_trials):
        t = max(1, int(lengths[i]))
        if t >= int(window):
            seg = rates_padded[i, t - int(window) : t]
        else:
            first = rates_padded[i, 0:1]
            pad = np.repeat(first, int(window) - t, axis=0)
            seg = np.concatenate([pad, rates_padded[i, :t]], axis=0)
        out[i] = np.nanmean(seg, axis=0)
    return out


def _class_mean_pairwise_separation(features_by_orientation: dict[int, np.ndarray], orientations: tuple[int, ...], idx: np.ndarray) -> tuple[float, dict[tuple[int, int], float]]:
    means = {ori: np.nanmean(features_by_orientation[ori][idx], axis=0) for ori in orientations}
    pair_values: dict[tuple[int, int], float] = {}
    for ori_a, ori_b in combinations(orientations, 2):
        pair_values[(ori_a, ori_b)] = float(np.linalg.norm(means[ori_b] - means[ori_a]))
    return float(np.mean(list(pair_values.values()))), pair_values


def _paired_trace_pairwise_values(
    features_by_orientation: dict[int, np.ndarray],
    orientations: tuple[int, ...],
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    n_trials = min(features_by_orientation[ori].shape[0] for ori in orientations)
    pair_arrays = []
    pair_means: dict[tuple[int, int], float] = {}
    for ori_a, ori_b in combinations(orientations, 2):
        diff = features_by_orientation[ori_b][:n_trials] - features_by_orientation[ori_a][:n_trials]
        vals = np.linalg.norm(diff, axis=1).astype(np.float64)
        pair_arrays.append(vals)
        pair_means[(ori_a, ori_b)] = float(np.nanmean(vals))
    per_trace = np.nanmean(np.stack(pair_arrays, axis=1), axis=1)
    return per_trace, pair_means


def _bootstrap_condition(
    features_by_orientation: dict[int, np.ndarray],
    orientations: tuple[int, ...],
    n_trials: int,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float, dict[tuple[int, int], float], float, dict[tuple[int, int], float]]:
    base_idx = np.arange(n_trials, dtype=np.int32)
    class_point, class_pair_values = _class_mean_pairwise_separation(features_by_orientation, orientations, base_idx)
    trace_values, pair_values = _paired_trace_pairwise_values(features_by_orientation, orientations)
    point = float(np.nanmean(trace_values[:n_trials]))
    if n_trials <= 1:
        return point, point, point, pair_values, class_point, class_pair_values
    boots = np.empty(int(n_bootstrap), dtype=np.float64)
    for b in range(int(n_bootstrap)):
        idx = rng.integers(0, n_trials, size=n_trials)
        boots[b] = float(np.nanmean(trace_values[idx]))
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), pair_values, class_point, class_pair_values


def _bootstrap_gain(
    real_features: dict[int, np.ndarray],
    stationary_features: dict[int, np.ndarray],
    orientations: tuple[int, ...],
    n_trials: int,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float, float]:
    real_values, _ = _paired_trace_pairwise_values(real_features, orientations)
    stat_values, _ = _paired_trace_pairwise_values(stationary_features, orientations)
    paired_delta = real_values[:n_trials] - stat_values[:n_trials]
    point = float(np.nanmean(paired_delta))
    if n_trials <= 1:
        p_nonpositive = 1.0 if point <= 0 else 0.0
        return point, point, point, p_nonpositive
    boots = np.empty(int(n_bootstrap), dtype=np.float64)
    for b in range(int(n_bootstrap)):
        idx = rng.integers(0, n_trials, size=n_trials)
        boots[b] = float(np.nanmean(paired_delta[idx]))
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), float(np.mean(boots <= 0.0))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _load_eye_traces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return d["traces"].astype(np.float32), d["durations"].astype(np.int32)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2 or np.nanstd(a) <= EPS or np.nanstd(b) <= EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _length_alignment_row(
    *,
    requested_condition: str,
    cache_condition: str,
    logmar: float,
    orientation: int,
    path: Path,
    lengths: np.ndarray,
    durations: np.ndarray,
    trace_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    if trace_indices is not None and len(trace_indices) >= len(lengths):
        source_durations = np.asarray(durations[trace_indices[: len(lengths)]], dtype=np.int64)
        trace_index_status = "present"
    else:
        source_durations = np.asarray(durations, dtype=np.int64)
        trace_index_status = "absent"
    n = min(int(len(lengths)), int(len(source_durations)))
    cached = np.asarray(lengths[:n], dtype=np.int64)
    current = np.asarray(source_durations[:n], dtype=np.int64)
    same_duration = float(np.mean(cached == current)) if n else float("nan")
    same_duration_plus_one = float(np.mean(cached == (current + 1))) if n else float("nan")
    corr_duration = _safe_corr(cached.astype(np.float64), current.astype(np.float64)) if n else float("nan")
    median_diff = float(np.median(cached - current)) if n else float("nan")
    status = "ok"
    if n == 0:
        status = "empty"
    elif same_duration < 0.95 and same_duration_plus_one < 0.95 and (not np.isfinite(corr_duration) or corr_duration < 0.95):
        status = "mismatch"
    return {
        "requested_condition": requested_condition,
        "cache_condition": cache_condition,
        "logmar": float(logmar),
        "orientation": int(orientation),
        "path": str(path),
        "trace_indices_status": trace_index_status,
        "n_cached": int(len(lengths)),
        "n_current_eye_traces": int(len(durations)),
        "n_compared": int(n),
        "cached_length_min": int(np.min(cached)) if n else -1,
        "cached_length_median": float(np.median(cached)) if n else float("nan"),
        "cached_length_max": int(np.max(cached)) if n else -1,
        "current_duration_min": int(np.min(current)) if n else -1,
        "current_duration_median": float(np.median(current)) if n else float("nan"),
        "current_duration_max": int(np.max(current)) if n else -1,
        "same_duration_fraction": same_duration,
        "same_duration_plus_one_fraction": same_duration_plus_one,
        "length_duration_corr": corr_duration,
        "median_cached_minus_current_duration": median_diff,
        "alignment_status": status,
    }


def _trace_scale_rows(
    traces: np.ndarray,
    durations: np.ndarray,
    n_trials: int,
    logmars: tuple[float, ...],
    conditions: tuple[str, ...],
    window: int,
    retina_ppd: float,
    world_ppd: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        full_rms_arcmin = []
        window_rms_arcmin = []
        window_step_arcmin = []
        for i in range(min(n_trials, len(durations))):
            dur = int(durations[i])
            if dur <= 0:
                continue
            e_full_raw = np.asarray(traces[i, :dur], dtype=np.float64)
            mean_full = np.mean(e_full_raw, axis=0, keepdims=True)
            if condition in ("stabilized", "stationary_phase_jittered", "trial_mean_stabilized"):
                e_full = np.repeat(mean_full, dur, axis=0)
            elif condition == "fixed_center":
                grand = np.mean(np.concatenate([traces[j, : int(durations[j])] for j in range(min(n_trials, len(durations))) if int(durations[j]) > 0], axis=0), axis=0, keepdims=True)
                e_full = np.repeat(grand, dur, axis=0)
            elif condition.startswith("scaled_"):
                scale = float(condition.split("_", 1)[1])
                e_full = mean_full + (e_full_raw - mean_full) * scale
            else:
                e_full = e_full_raw
            e_win = e_full[max(0, dur - int(window)) : dur]
            c_full = e_full - np.mean(e_full, axis=0, keepdims=True)
            c_win = e_win - np.mean(e_win, axis=0, keepdims=True)
            full_rms_arcmin.append(float(np.sqrt(np.mean(np.sum(c_full * c_full, axis=1))) * 60.0))
            window_rms_arcmin.append(float(np.sqrt(np.mean(np.sum(c_win * c_win, axis=1))) * 60.0))
            if len(e_win) > 1:
                steps = np.linalg.norm(np.diff(e_win, axis=0), axis=1)
                window_step_arcmin.append(float(np.median(steps) * 60.0))

        full_rms = np.asarray(full_rms_arcmin, dtype=np.float64)
        win_rms = np.asarray(window_rms_arcmin, dtype=np.float64)
        win_step = np.asarray(window_step_arcmin, dtype=np.float64)
        for logmar in logmars:
            gap_arcmin = float(10.0 ** float(logmar))
            letter_arcmin = 5.0 * gap_arcmin
            row = {
                "condition": condition,
                "logmar": float(logmar),
                "n_trials": int(len(win_rms)),
                "window_frames": int(window),
                "gap_arcmin": gap_arcmin,
                "letter_arcmin": letter_arcmin,
                "gap_world_px": gap_arcmin / 60.0 * float(world_ppd),
                "gap_retina_px": gap_arcmin / 60.0 * float(retina_ppd),
                "letter_world_px": letter_arcmin / 60.0 * float(world_ppd),
                "full_trial_rms_arcmin_median": float(np.nanmedian(full_rms)) if full_rms.size else float("nan"),
                "window_rms_arcmin_median": float(np.nanmedian(win_rms)) if win_rms.size else float("nan"),
                "window_rms_arcmin_p10": float(np.nanquantile(win_rms, 0.10)) if win_rms.size else float("nan"),
                "window_rms_arcmin_p90": float(np.nanquantile(win_rms, 0.90)) if win_rms.size else float("nan"),
                "window_median_step_arcmin_median": float(np.nanmedian(win_step)) if win_step.size else float("nan"),
                "n_window_rms_lte_gap": int(np.sum(win_rms <= gap_arcmin)) if win_rms.size else 0,
                "n_window_rms_lte_letter": int(np.sum(win_rms <= letter_arcmin)) if win_rms.size else 0,
            }
            row["median_window_rms_over_gap"] = row["window_rms_arcmin_median"] / max(gap_arcmin, EPS)
            row["median_window_rms_over_letter"] = row["window_rms_arcmin_median"] / max(letter_arcmin, EPS)
            rows.append(row)
    return rows


def _eye_overlay(traces: np.ndarray, durations: np.ndarray, max_traces: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(int(max_traces), len(durations))
    valid = [traces[i, : int(durations[i])] for i in range(n) if int(durations[i]) > 0]
    if not valid:
        z = np.zeros((0, 2), dtype=np.float32)
        return z, z, z
    all_pos = np.concatenate(valid, axis=0)
    grand_mean = np.mean(all_pos, axis=0, keepdims=True)
    trial_means = np.stack([np.mean(v, axis=0) for v in valid], axis=0) - grand_mean
    first_trace = valid[0] - grand_mean
    return grand_mean[0], trial_means, first_trace


def _copy_phase_landscape(source_csv: Path | None, output_csv: Path) -> list[dict[str, str]]:
    if source_csv is None or not source_csv.exists():
        _write_csv(
            output_csv,
            [],
            fieldnames=[
                "logmar",
                "offset_x_deg",
                "offset_y_deg",
                "offset_x_arcmin",
                "offset_y_arcmin",
                "mean_pairwise_sep",
                "status",
            ],
        )
        return []
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_csv, output_csv)
    with source_csv.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _plot_figure(
    path_png: Path,
    path_pdf: Path,
    summary_rows: list[dict[str, Any]],
    gain_rows: list[dict[str, Any]],
    phase_rows: list[dict[str, str]],
    primary_logmar: float,
    eye_trial_means: np.ndarray,
    eye_first_trace: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2), constrained_layout=True)
    ax = axes[0, 0]
    if eye_trial_means.size:
        tm = eye_trial_means * 60.0
        tr = eye_first_trace * 60.0
        ax.scatter(tm[:, 0], tm[:, 1], s=12, alpha=0.25, color="#4c78a8", label="stationary samples")
        if tr.size:
            ax.plot(tr[:, 0], tr[:, 1], color="#e45756", linewidth=1.4, alpha=0.9, label="one FEM trajectory")
            ax.scatter(tr[0, 0], tr[0, 1], s=28, color="#e45756")
        ax.scatter([0.0], [0.0], marker="x", s=60, color="#222222", label="fixed center")
    ax.set_title("A  Phase-sampling regimes")
    ax.set_xlabel("x offset (arcmin)")
    ax.set_ylabel("y offset (arcmin)")
    ax.axis("equal")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[0, 1]
    heat_rows = []
    if phase_rows:
        logmars = np.asarray([float(r["logmar"]) for r in phase_rows], dtype=np.float64)
        nearest = float(logmars[np.argmin(np.abs(logmars - float(primary_logmar)))])
        heat_rows = [r for r in phase_rows if abs(float(r["logmar"]) - nearest) < 1e-9 and "mean_pairwise_sep" in r]
    if heat_rows:
        xs = np.asarray(sorted({float(r["offset_x_arcmin"]) for r in heat_rows}), dtype=np.float64)
        ys = np.asarray(sorted({float(r["offset_y_arcmin"]) for r in heat_rows}), dtype=np.float64)
        grid = np.full((len(xs), len(ys)), np.nan, dtype=np.float64)
        x_index = {float(x): i for i, x in enumerate(xs)}
        y_index = {float(y): i for i, y in enumerate(ys)}
        for r in heat_rows:
            grid[x_index[float(r["offset_x_arcmin"])]][y_index[float(r["offset_y_arcmin"])]] = float(r["mean_pairwise_sep"])
        im = ax.imshow(
            grid.T,
            origin="lower",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            aspect="equal",
            cmap="viridis",
        )
        ax.scatter([0.0], [0.0], marker="x", s=60, color="white")
        fig.colorbar(im, ax=ax, shrink=0.8, label="mean pairwise separation")
        ax.set_title(f"B  Phase landscape, LogMAR {nearest:+.2f}")
    else:
        ax.text(0.5, 0.5, "phase landscape\nnot provided", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("B  Phase landscape")
    ax.set_xlabel("x offset (arcmin)")
    ax.set_ylabel("y offset (arcmin)")

    ax = axes[1, 0]
    colors = {
        "real_fem": "#e45756",
        "real": "#e45756",
        "stationary_phase_jittered": "#4c78a8",
        "stabilized": "#4c78a8",
        "scaled_0.1": "#59a14f",
        "scaled_0.05": "#f28e2b",
        "fixed_center": "#222222",
    }
    plot_conditions = []
    for row in summary_rows:
        cond = str(row["condition"])
        if cond not in plot_conditions:
            plot_conditions.append(cond)
    for condition in plot_conditions:
        rows = [r for r in summary_rows if r["condition"] == condition]
        rows.sort(key=lambda r: float(r["logmar"]))
        if not rows:
            continue
        x = np.asarray([float(r["logmar"]) for r in rows])
        y = np.asarray([float(r["mean_pairwise_separation"]) for r in rows])
        lo = np.asarray([float(r["ci_low"]) for r in rows])
        hi = np.asarray([float(r["ci_high"]) for r in rows])
        label = CONDITION_LABELS.get(condition, condition)
        ax.plot(x, y, marker="o", color=colors.get(condition), label=label)
        ax.fill_between(x, lo, hi, color=colors.get(condition), alpha=0.17)
    ax.set_title("C  Separation across size")
    ax.set_xlabel("LogMAR")
    ax.set_ylabel("mean pairwise separation")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    gain_rows = sorted(gain_rows, key=lambda r: float(r["logmar"]))
    if gain_rows:
        x = np.asarray([float(r["logmar"]) for r in gain_rows])
        y = np.asarray([float(r["real_minus_stationary_gain"]) for r in gain_rows])
        lo = np.asarray([float(r["ci_low"]) for r in gain_rows])
        hi = np.asarray([float(r["ci_high"]) for r in gain_rows])
        ax.plot(x, y, marker="o", color="#59a14f")
        ax.fill_between(x, lo, hi, color="#59a14f", alpha=0.22)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    ax.set_title("D  Real minus stationary")
    ax.set_xlabel("LogMAR")
    ax.set_ylabel("separation gain")
    ax.grid(alpha=0.2)

    fig.savefig(path_png, dpi=240)
    fig.savefig(path_pdf)
    plt.close(fig)


def _render_audit_rows(
    logmars: tuple[float, ...],
    orientations: tuple[int, ...],
    traces: np.ndarray,
    durations: np.ndarray,
    n_trials: int,
    retina_ppd: float,
) -> list[dict[str, Any]]:
    if n_trials <= 0:
        return []
    means = np.stack([np.mean(traces[i, : int(durations[i])], axis=0) for i in range(n_trials)], axis=0)
    rows: list[dict[str, Any]] = []
    crop_deg = 101.0 / float(retina_ppd)
    for logmar in logmars:
        letter_deg = 5.0 * (10.0 ** float(logmar) / 60.0)
        rows.append(
            {
                "logmar": float(logmar),
                "n_orientations": int(len(orientations)),
                "n_trials_per_orientation": int(n_trials),
                "orientation_counts_equal": 1,
                "orientation_specific_offset_imbalance_arcmin": 0.0,
                "mean_stationary_offset_x_arcmin": float(np.mean(means[:, 0]) * 60.0),
                "mean_stationary_offset_y_arcmin": float(np.mean(means[:, 1]) * 60.0),
                "rms_stationary_offset_arcmin": float(np.sqrt(np.mean(np.sum((means - np.mean(means, axis=0, keepdims=True)) ** 2, axis=1))) * 60.0),
                "letter_size_deg": letter_deg,
                "retina_crop_deg": crop_deg,
                "letter_inside_crop": int(letter_deg < crop_deg),
                "render_artifact_flag": "saturation_possible_below_-0.40" if float(logmar) <= -0.40 else "none_known",
            }
        )
    return rows


def _write_text_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    gain_rows: list[dict[str, Any]],
    trace_scale_rows: list[dict[str, Any]],
) -> None:
    caption = (
        "In a deterministic V1 digital twin, fine E-optotype responses depend strongly on subpixel retinal phase. "
        "Real FEM trajectories sample multiple phases within a trial, whereas the stationary phase-jittered control "
        "samples one matched trial-mean phase per trace. FEM-related gains in orientation separation therefore indicate "
        "a model-side phase-sampling effect, not a direct behavioral acuity claim. Because the twin lacks an intrinsic "
        "biological noise floor, decoder-like metrics are interpreted as deterministic response geometry."
    )
    (out_dir / "caption.txt").write_text(caption + "\n")

    primary = next((r for r in gain_rows if abs(float(r["logmar"]) - float(args.primary_logmar)) < 1e-9), None)
    primary_line = "not available"
    if primary:
        primary_line = (
            f"{float(primary['real_minus_stationary_gain']):.6g} "
            f"[{float(primary['ci_low']):.6g}, {float(primary['ci_high']):.6g}]"
        )
    alignment_status = ", ".join(metadata.get("cache_eye_trace_alignment_statuses", [])) or "unknown"
    scale_row = next(
        (
            r for r in trace_scale_rows
            if abs(float(r["logmar"]) - float(args.primary_logmar)) < 1e-9
            and str(r.get("condition")) == str(args.gain_condition)
        ),
        None,
    )
    scale_line = "not available"
    if scale_row:
        scale_line = (
            f"median last-window RMS {float(scale_row['window_rms_arcmin_median']):.3g} arcmin; "
            f"gap {float(scale_row['gap_arcmin']):.3g} arcmin; "
            f"letter {float(scale_row['letter_arcmin']):.3g} arcmin"
        )

    readme = [
        "# Supplemental E-optotype phase-sampling analysis",
        "",
        "## Interpretation guardrail",
        "",
        "The V1 twin is treated as a deterministic sensory transducer. Pairwise separation is a model-side response-geometry metric, not a behavioral acuity estimate.",
        "",
        "## Primary comparison",
        "",
        "- real_fem: measured within-trial FEM trajectories.",
        "- stationary_phase_jittered: one stationary trial-mean phase per trace, backed by the existing `stabilized` caches when explicit alias caches are absent.",
        "- fixed_center: deterministic oracle diagnostic only.",
        "",
        "## Settings",
        "",
        f"- rates_dir: `{args.rates_dir}`",
        f"- eye_traces: `{args.eye_traces}`",
        f"- logmars: {', '.join(f'{x:+.2f}' for x in metadata['logmars'])}",
        f"- orientations: {metadata['orientations']}",
        f"- integration_window_frames: {metadata['integration_window_frames']}",
        f"- bootstrap_samples: {metadata['bootstrap_samples']}",
        f"- random_seed: {metadata['random_seed']}",
        f"- model/source: {metadata['model_checkpoint']}",
        f"- retinal offsets: degrees in caches, arcmin in figure overlays and audits",
        f"- deterministic rates: yes",
        f"- noise model: none",
        f"- cache_eye_trace_alignment_statuses: {alignment_status}",
        f"- primary_scale_audit: {scale_line}",
        "",
        "## Primary result",
        "",
        f"- {args.gain_condition} - {args.stationary_condition} at LogMAR {float(args.primary_logmar):+.2f}: {primary_line}",
        "",
        "## Outputs",
        "",
        "- `supp_eoptotype_phase_sampling_summary.csv`",
        "- `bootstrap_summary.csv`",
        "- `phase_landscape_metrics.csv`",
        "- `condition_metadata.json`",
        "- `audit_summary.csv`",
        "- `cache_eye_trace_alignment.csv`",
        "- `fem_scale_audit.csv`",
        "- `supp_eoptotype_phase_sampling.png`",
        "- `supp_eoptotype_phase_sampling.pdf`",
        "- `caption.txt`",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build supplemental E-optotype phase-sampling figure from cached rates.")
    parser.add_argument("--rates-dir", type=Path, default=VISIONCORE_ROOT / "scripts" / "temporal_decoding" / "data" / "rates")
    parser.add_argument("--eye-traces", type=Path, default=VISIONCORE_ROOT / "scripts" / "temporal_decoding" / "data" / "eye_traces.npz")
    parser.add_argument("--phase-landscape-csv", type=Path, default=VISIONCORE_ROOT / "declan" / "results" / "phase_landscape_fine" / "phase_landscape_summary.csv")
    parser.add_argument("--out-dir", type=Path, default=VISIONCORE_ROOT / "outputs" / "supp_eoptotype_phase_sampling")
    parser.add_argument("--logmars", type=str, default=",".join(f"{x:.2f}" for x in LOGMARS_DEFAULT))
    parser.add_argument("--orientations", type=str, default=",".join(str(x) for x in ORIENTATIONS_DEFAULT))
    parser.add_argument("--conditions", type=str, default=",".join(CONDITIONS_DEFAULT))
    parser.add_argument("--file-tag", type=str, default="", help="Optional cache filename tag, e.g. allhires_fresh.")
    parser.add_argument("--gain-condition", type=str, default="real_fem", help="Condition to subtract stationary baseline from.")
    parser.add_argument("--stationary-condition", type=str, default="stationary_phase_jittered", help="Stationary baseline condition for gain.")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--max-traces", type=int, default=471)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--primary-logmar", type=float, default=-0.20)
    parser.add_argument("--hires-threshold", type=float, default=HIRES_THRESHOLD_DEFAULT)
    parser.add_argument("--retina-ppd", type=float, default=37.50476617)
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logmars = _parse_csv_floats(args.logmars)
    orientations = _parse_csv_ints(args.orientations)
    conditions = _parse_csv_strings(args.conditions)
    file_tag = _normalize_file_tag(args.file_tag)
    rng = np.random.default_rng(int(args.random_seed))
    traces, durations = _load_eye_traces(args.eye_traces)

    summary_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    missing_conditions: list[str] = []
    cache_resolution_rows: list[dict[str, Any]] = []
    length_alignment_rows: list[dict[str, Any]] = []
    features: dict[tuple[float, str], dict[int, np.ndarray]] = {}
    n_trials_by_logmar: dict[float, int] = {}

    for logmar in logmars:
        for condition in conditions:
            by_ori: dict[int, np.ndarray] = {}
            cache_names = []
            n_trials_condition: int | None = None
            for ori in orientations:
                path, cache_condition = _resolve_rate_path(
                    args.rates_dir,
                    float(logmar),
                    int(ori),
                    condition,
                    float(args.hires_threshold),
                    file_tag=file_tag,
                )
                cache_names.append(cache_condition)
                rates, lengths, trace_indices = _load_rates(path)
                length_alignment_rows.append(
                    _length_alignment_row(
                        requested_condition=condition,
                        cache_condition=cache_condition,
                        logmar=float(logmar),
                        orientation=int(ori),
                        path=path,
                        lengths=lengths,
                        durations=durations,
                        trace_indices=trace_indices,
                    )
                )
                n_here = min(int(args.max_traces), int(rates.shape[0]), int(lengths.shape[0]), int(traces.shape[0]))
                n_trials_condition = n_here if n_trials_condition is None else min(n_trials_condition, n_here)
                by_ori[int(ori)] = _window_mean(rates, lengths, int(args.window), n_here)
                cache_resolution_rows.append(
                    {
                        "requested_condition": condition,
                        "cache_condition": cache_condition,
                        "logmar": float(logmar),
                        "orientation": int(ori),
                        "path": str(path),
                        "n_trials_available": int(n_here),
                    }
                )
            if n_trials_condition is None or n_trials_condition <= 0:
                missing_conditions.append(f"{condition}:{logmar:+.2f}")
                continue
            by_ori = {ori: arr[:n_trials_condition] for ori, arr in by_ori.items()}
            n_trials_by_logmar[float(logmar)] = min(n_trials_by_logmar.get(float(logmar), n_trials_condition), n_trials_condition)
            features[(float(logmar), condition)] = by_ori

            seed = int(rng.integers(0, 2**31 - 1))
            point, lo, hi, pairs, class_point, class_pairs = _bootstrap_condition(
                by_ori,
                orientations,
                n_trials_condition,
                np.random.default_rng(seed),
                int(args.n_bootstrap),
            )
            summary_rows.append(
                {
                    "condition": condition,
                    "cache_condition_used": ",".join(sorted(set(cache_names))),
                    "logmar": float(logmar),
                    "window": int(args.window),
                    "n_traces": int(n_trials_condition),
                    "n_orientations": int(len(orientations)),
                    "primary_metric": "paired_trace_mean_pairwise_separation",
                    "mean_pairwise_separation": point,
                    "paired_trace_pairwise_separation": point,
                    "class_mean_pairwise_separation": class_point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "bootstrap_seed": seed,
                }
            )
            bootstrap_rows.append(
                {
                    "metric": "paired_trace_mean_pairwise_separation",
                    "condition": condition,
                    "logmar": float(logmar),
                    "estimate": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_bootstrap": int(args.n_bootstrap),
                    "bootstrap_seed": seed,
                }
            )
            for (ori_a, ori_b), value in pairs.items():
                pairwise_rows.append(
                    {
                        "condition": condition,
                        "logmar": float(logmar),
                        "orientation_a": int(ori_a),
                        "orientation_b": int(ori_b),
                        "pairwise_separation": value,
                        "paired_trace_pairwise_separation": value,
                        "class_mean_pairwise_separation": class_pairs[(ori_a, ori_b)],
                    }
                )

    gain_rows: list[dict[str, Any]] = []
    for logmar in logmars:
        gain_condition = str(args.gain_condition)
        stat_condition = str(args.stationary_condition)
        gain_key = (float(logmar), gain_condition)
        if gain_key not in features and gain_condition == "real_fem":
            gain_key = (float(logmar), "real")
        stat_key = (float(logmar), stat_condition)
        if stat_key not in features and stat_condition == "stationary_phase_jittered":
            stat_key = (float(logmar), "stabilized")
        if gain_key not in features or stat_key not in features:
            continue
        n_trials = min(
            min(arr.shape[0] for arr in features[gain_key].values()),
            min(arr.shape[0] for arr in features[stat_key].values()),
        )
        real_features = {ori: arr[:n_trials] for ori, arr in features[gain_key].items()}
        stat_features = {ori: arr[:n_trials] for ori, arr in features[stat_key].items()}
        seed = int(rng.integers(0, 2**31 - 1))
        point, lo, hi, p_nonpositive = _bootstrap_gain(
            real_features,
            stat_features,
            orientations,
            n_trials,
            np.random.default_rng(seed),
            int(args.n_bootstrap),
        )
        gain_row = {
            "metric": f"{gain_key[1]}_minus_{stat_key[1]}_gain",
            "gain_condition": str(gain_key[1]),
            "stationary_condition": str(stat_key[1]),
            "logmar": float(logmar),
            "window": int(args.window),
            "n_traces": int(n_trials),
            "real_minus_stationary_gain": point,
            "ci_low": lo,
            "ci_high": hi,
            "p_bootstrap_nonpositive": p_nonpositive,
            "n_bootstrap": int(args.n_bootstrap),
            "bootstrap_seed": seed,
        }
        gain_rows.append(gain_row)
        bootstrap_rows.append(
            {
                "metric": "real_minus_stationary_gain",
                "condition": f"{gain_key[1]}_minus_{stat_key[1]}",
                "logmar": float(logmar),
                "estimate": point,
                "ci_low": lo,
                "ci_high": hi,
                "n_bootstrap": int(args.n_bootstrap),
                "bootstrap_seed": seed,
            }
        )

    n_audit = min([int(args.max_traces), int(traces.shape[0]), *(n_trials_by_logmar.values() or [int(args.max_traces)])])
    audit_rows = _render_audit_rows(logmars, orientations, traces, durations, n_audit, float(args.retina_ppd))
    trace_scale_rows = _trace_scale_rows(
        traces=traces,
        durations=durations,
        n_trials=n_audit,
        logmars=logmars,
        conditions=conditions,
        window=int(args.window),
        retina_ppd=float(args.retina_ppd),
        world_ppd=120.0,
    )
    phase_rows = _copy_phase_landscape(args.phase_landscape_csv, out_dir / "phase_landscape_metrics.csv")
    _, eye_trial_means, eye_first_trace = _eye_overlay(traces, durations, n_audit)

    _write_csv(out_dir / "supp_eoptotype_phase_sampling_summary.csv", summary_rows)
    _write_csv(out_dir / "bootstrap_summary.csv", bootstrap_rows)
    _write_csv(out_dir / "real_minus_stationary_gain.csv", gain_rows)
    _write_csv(out_dir / "pairwise_separation.csv", pairwise_rows)
    _write_csv(out_dir / "audit_summary.csv", audit_rows)
    _write_csv(out_dir / "cache_eye_trace_alignment.csv", length_alignment_rows)
    _write_csv(out_dir / "fem_scale_audit.csv", trace_scale_rows)
    _write_csv(out_dir / "cache_resolution_manifest.csv", cache_resolution_rows)

    alignment_statuses = sorted({str(row["alignment_status"]) for row in length_alignment_rows})

    metadata = {
        "script": "scripts/figure4/run_supp_eoptotype_phase_sampling.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rates_dir": str(args.rates_dir),
        "file_tag": file_tag,
        "eye_trace_source": str(args.eye_traces),
        "phase_landscape_source": str(args.phase_landscape_csv),
        "conditions": list(conditions),
        "gain_condition": str(args.gain_condition),
        "stationary_condition": str(args.stationary_condition),
        "condition_semantics": {
            "real_fem": "measured within-trial FEM trajectories",
            "stationary_phase_jittered": "trial-mean stationary phase per trace; existing stabilized caches are a valid backing cache",
            "fixed_center": "grand-mean fixed position, deterministic-oracle diagnostic",
        },
        "logmars": list(logmars),
        "orientations": list(orientations),
        "integration_window_frames": int(args.window),
        "max_traces": int(args.max_traces),
        "bootstrap_samples": int(args.n_bootstrap),
        "random_seed": int(args.random_seed),
        "hires_threshold": float(args.hires_threshold),
        "retina_ppd": float(args.retina_ppd),
        "model_checkpoint": "learned_resnet_none_convgru_gaussian epoch 147 via cached temporal_decoding rates",
        "deterministic_rates": True,
        "noise_model": "none",
        "cache_eye_trace_alignment_statuses": alignment_statuses,
        "missing_conditions": missing_conditions,
    }
    (out_dir / "condition_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    _plot_figure(
        out_dir / "supp_eoptotype_phase_sampling.png",
        out_dir / "supp_eoptotype_phase_sampling.pdf",
        summary_rows,
        gain_rows,
        phase_rows,
        float(args.primary_logmar),
        eye_trial_means,
        eye_first_trace,
    )
    _write_text_outputs(out_dir, args, metadata, gain_rows, trace_scale_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
