from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


DEFAULT_SUMMARY_METRICS = (
    "rms_radius_deg",
    "cloud_area_deg2",
    "anisotropy",
    "step_mean_deg",
    "speed_mean_deg_s",
    "diffusion_constant_deg2_s",
    "velocity_autocorr_lag1",
    "direction_persistence",
    "path_length_deg_s",
    "return_to_center_strength",
    "position_psd_slope_1_30hz",
    "position_high_freq_power_fraction_15_60hz",
)


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, np.nan))
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")


def summarize_windows(
    rows: list[dict[str, Any]],
    *,
    metrics: tuple[str, ...] = DEFAULT_SUMMARY_METRICS,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["session"]), str(row["stimulus"]), str(row["regime"]), str(row["phase"]))].append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        session, stimulus, regime, phase = key
        items = groups[key]
        base = {
            "session": session,
            "stimulus": stimulus,
            "regime": regime,
            "phase": phase,
            "n_windows": len(items),
            "n_trials": len({int(r["trial_idx"]) for r in items}),
        }
        for metric in metrics:
            vals = [_float(r, metric) for r in items]
            base[f"{metric}_mean"] = _mean(vals)
            base[f"{metric}_std"] = _std(vals)
        out.append(base)
    return out


def summarize_events(event_rows: list[dict[str, Any]], inventory_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    durations: dict[tuple[str, str], float] = defaultdict(float)
    trials: dict[tuple[str, str], int] = defaultdict(int)
    for row in inventory_rows:
        key = (str(row["session"]), str(row["stimulus"]))
        trials[key] += int(row.get("n_trials", 0) or 0)
        durations[key] += float(row.get("valid_duration_s", 0.0) or 0.0)

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        groups[(str(row["session"]), str(row["stimulus"]), str(row["regime"]))].append(row)

    out: list[dict[str, Any]] = []
    all_keys = {(str(r["session"]), str(r["stimulus"]), str(r["regime"])) for r in inventory_rows}
    all_keys |= set(groups)
    for session, stimulus, regime in sorted(all_keys):
        items = groups.get((session, stimulus, regime), [])
        dur = float(durations.get((session, stimulus), np.nan))
        amp = [_float(r, "event_amplitude_deg") for r in items]
        peak = [_float(r, "event_peak_speed_deg_s") for r in items]
        out.append({
            "session": session,
            "stimulus": stimulus,
            "regime": regime,
            "n_events": len(items),
            "n_trials_inventory": int(trials.get((session, stimulus), 0)),
            "event_valid_duration_s": dur,
            "event_rate_hz": len(items) / dur if dur and np.isfinite(dur) else np.nan,
            "event_amplitude_deg_mean": _mean(amp),
            "event_amplitude_deg_std": _std(amp),
            "event_peak_speed_deg_s_mean": _mean(peak),
        })
    return out


def paired_metric_contrasts(
    session_summary: list[dict[str, Any]],
    *,
    baseline: str = "fixrsvp",
    metrics: tuple[str, ...] = DEFAULT_SUMMARY_METRICS,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    index = {
        (str(r["session"]), str(r["stimulus"]), str(r["phase"])): r
        for r in session_summary
    }
    sessions = sorted({str(r["session"]) for r in session_summary})
    stimuli = sorted({str(r["stimulus"]) for r in session_summary if str(r["stimulus"]) != baseline})
    phases = sorted({str(r["phase"]) for r in session_summary})
    rng = np.random.default_rng(int(seed))
    out: list[dict[str, Any]] = []
    for stim in stimuli:
        for phase in phases:
            for metric in metrics:
                diffs: list[float] = []
                for session in sessions:
                    a = index.get((session, stim, phase))
                    b = index.get((session, baseline, phase))
                    if a is None or b is None:
                        continue
                    diff = _float(a, f"{metric}_mean") - _float(b, f"{metric}_mean")
                    if np.isfinite(diff):
                        diffs.append(diff)
                arr = np.asarray(diffs, dtype=np.float64)
                if arr.size == 0:
                    continue
                if arr.size > 1 and n_bootstrap > 0:
                    boot = np.mean(arr[rng.integers(0, arr.size, size=(int(n_bootstrap), arr.size))], axis=1)
                    lo, hi = np.quantile(boot, [0.025, 0.975])
                else:
                    lo = hi = float(np.mean(arr))
                out.append({
                    "baseline_stimulus": baseline,
                    "comparison_stimulus": stim,
                    "phase": phase,
                    "metric": metric,
                    "n_sessions": int(arr.size),
                    "mean_diff": float(np.mean(arr)),
                    "median_diff": float(np.median(arr)),
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "fraction_positive": float(np.mean(arr > 0)),
                })
    return out
