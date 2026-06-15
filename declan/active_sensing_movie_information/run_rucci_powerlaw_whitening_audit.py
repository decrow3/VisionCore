#!/usr/bin/env python3
"""Audit Rucci-style spatial power-law whitening by retinal drift.

The older input-whitening runner summarizes a pixel-averaged temporal PSD.  That
is useful, but it is not the same as the classic power-law argument: natural
images have excess low-spatial-frequency power, and small fixational motion can
boost high-spatial-frequency temporal modulations through image derivatives.

This runner asks the more direct cache-free question:

    Is frame-to-frame retinal modulation power flatter across spatial frequency
    at biological drift scale than at smaller/larger scales?

It renders retinal movies using the same crop path as the twininfo pipeline, but
does not run the digital twin.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from non_circular_fem_common import (
    DEFAULT_STACK_OUT_DIR,
    DEFAULT_TWININFO_RUN_DIR,
    load_crop_rows,
    load_run_config,
    load_selected_traces,
    parse_float_list,
    parse_int,
    read_csv_rows,
    scale_trace,
    stable_trace,
    summarize_groups,
    write_csv_rows,
    write_json,
)

from jake.twininfo.common import DT, PPD
from jake.twininfo.eye_controls import detect_microsaccade_events
from jake.twininfo.retinal_examples import retinal_movie_from_image_trace
from jake.twininfo.stimuli import load_natural_images


DEFAULT_OUT_DIR = DEFAULT_STACK_OUT_DIR / "rucci_powerlaw_whitening_audit"
DEFAULT_AMPLITUDE_SCALES = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
DEFAULT_FIT_BAND_CPD = (2.0, 16.0)
PRIMARY_METRICS = (
    "modulation_total_power",
    "modulation_power_slope",
    "source_image_power_slope",
    "transfer_slope",
    "transfer_slope_error_to_2",
    "modulation_spatial_flatness",
    "modulation_spatial_entropy",
)


def stable_seed(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest()[:8], 16)


def mean_sem(values: Iterable[float]) -> tuple[float, float, int]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    if arr.size == 1:
        return float(arr[0]), 0.0, 1
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(arr.size)), int(arr.size)


def spectral_flatness(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(vals + 1e-30))) / max(float(np.mean(vals)), 1e-30))


def spectral_entropy(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < 2:
        return float("nan")
    probs = vals / max(float(np.sum(vals)), 1e-30)
    return float(-np.sum(probs * np.log(probs + 1e-30)) / np.log(vals.size))


def loglog_slope(freq: np.ndarray, power: np.ndarray) -> float:
    f = np.asarray(freq, dtype=np.float64)
    p = np.asarray(power, dtype=np.float64)
    keep = np.isfinite(f) & np.isfinite(p) & (f > 0) & (p > 0)
    if int(np.sum(keep)) < 2:
        return float("nan")
    return float(np.polyfit(np.log10(f[keep]), np.log10(p[keep]), 1)[0])


def radial_power(
    frames: np.ndarray,
    *,
    ppd: float,
    fit_low_cpd: float,
    fit_high_cpd: float,
    n_bins: int,
) -> dict[str, np.ndarray | float]:
    """Return radially binned spatial power and summary metrics."""
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    arr = arr - np.mean(arr, axis=(1, 2), keepdims=True)
    h, w = arr.shape[1:]
    fy = np.fft.fftfreq(h, d=1.0 / float(ppd))
    fx = np.fft.fftfreq(w, d=1.0 / float(ppd))
    rr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    nyquist = float(np.nanmax(rr))
    low = max(float(fit_low_cpd) / 2.0, float(ppd) / max(h, w), 1e-6)
    high = min(max(float(fit_high_cpd) * 1.5, low * 1.01), nyquist)
    edges = np.geomspace(low, high, int(n_bins) + 1)
    fft = np.fft.fft2(arr, axes=(1, 2), norm="ortho")
    power2d = np.mean(np.abs(fft) ** 2, axis=0)

    centers: list[float] = []
    powers: list[float] = []
    counts: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (rr >= lo) & (rr < hi)
        centers.append(float(np.sqrt(lo * hi)))
        counts.append(int(np.sum(mask)))
        if np.any(mask):
            powers.append(float(np.mean(power2d[mask])))
        else:
            powers.append(float("nan"))

    center = np.asarray(centers, dtype=np.float64)
    power = np.asarray(powers, dtype=np.float64)
    fit = (center >= float(fit_low_cpd)) & (center <= float(fit_high_cpd))
    return {
        "freq_cpd": center,
        "power": power,
        "n_coefficients": np.asarray(counts, dtype=np.int64),
        "fit_mask": fit,
        "fit_slope": loglog_slope(center[fit], power[fit]),
        "fit_flatness": spectral_flatness(power[fit]),
        "fit_entropy": spectral_entropy(power[fit]),
        "fit_total_power": float(np.nansum(power[fit])),
        "nyquist_cpd": nyquist,
    }


def trace_pool_selected(run_dir: Path, t_max: int, max_traces: int) -> list[dict[str, Any]]:
    rows = [row for row in load_selected_traces(run_dir).values() if str(row.get("kind", "")) == "fixation"]
    rows = sorted(rows, key=lambda row: str(row.get("example_id", "")))
    if int(max_traces) > 0:
        rows = rows[: int(max_traces)]
    return rows


def trace_pool_raw_fixation(
    *,
    t_max: int,
    max_traces: int,
    seed: int,
    stride: int,
    max_source_traces: int,
) -> list[dict[str, Any]]:
    """Sample no-microsaccade fixational windows from the raw fixRSVP pool."""
    from jake.twininfo.common import extract_fixrsvp_eye_traces, load_digital_twin

    model, _model_info, _device = load_digital_twin()
    eye_traces, durations = extract_fixrsvp_eye_traces(model, min_fix_dur=t_max)
    source_indices = list(range(len(durations)))
    if int(max_source_traces) > 0:
        source_indices = source_indices[: int(max_source_traces)]
    candidates: list[dict[str, Any]] = []
    for source_idx in source_indices:
        duration = int(durations[source_idx])
        if duration < t_max:
            continue
        source = np.asarray(eye_traces[source_idx, :duration], dtype=np.float32)
        if np.isnan(source).any():
            continue
        _events, event_mask, threshold = detect_microsaccade_events(source, min_samples=1)
        for start in range(0, duration - t_max + 1, int(stride)):
            stop = start + t_max
            if np.any(event_mask[start:stop]):
                continue
            trace = source[start:stop]
            centered = trace - np.mean(trace, axis=0, keepdims=True)
            rms = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
            path = float(np.sum(np.linalg.norm(np.diff(trace, axis=0), axis=1)))
            candidates.append(
                {
                    "example_id": f"rawfix_src{source_idx:05d}_start{start:05d}",
                    "kind": "fixation",
                    "source_trace_index": source_idx,
                    "window_start": start,
                    "window_stop": stop,
                    "threshold_deg_s": threshold,
                    "rms_displacement_deg": rms,
                    "path_length_deg": path,
                    "trace": trace.astype(np.float32),
                }
            )
    rng = np.random.default_rng(int(seed))
    rng.shuffle(candidates)
    if int(max_traces) > 0:
        candidates = candidates[: int(max_traces)]
    return candidates


def load_trace_pool(args: argparse.Namespace, t_max: int) -> list[dict[str, Any]]:
    if str(args.trace_source) == "selected":
        return trace_pool_selected(Path(args.run_dir), t_max=t_max, max_traces=int(args.max_traces))
    if str(args.trace_source) == "raw_fixation":
        return trace_pool_raw_fixation(
            t_max=t_max,
            max_traces=int(args.max_traces),
            seed=int(args.seed),
            stride=int(args.trace_stride),
            max_source_traces=int(args.max_source_traces),
        )
    raise ValueError(f"Unknown trace source: {args.trace_source}")


def selected_crops(run_dir: Path, *, max_images: int, max_crops: int) -> list[dict[str, str]]:
    crops = load_crop_rows(run_dir)
    if int(max_crops) > 0:
        crops = crops[: int(max_crops)]
    if int(max_images) > 0:
        allowed = {int(row["image_index"]) for row in crops[: int(max_images)]}
        crops = [row for row in crops if int(row["image_index"]) in allowed]
    return crops


def image_lookup(crops: list[dict[str, str]]) -> dict[int, np.ndarray]:
    indices = sorted({int(row["image_index"]) for row in crops})
    loaded = load_natural_images(len(indices), indices=tuple(indices))
    return {int(spec.image_index): image for spec, image in loaded if spec.image_index is not None}


def summarize_scale(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("motion_family", "amplitude_scale", "diffusion_scale")
    out: list[dict[str, Any]] = []
    metrics = (
        "source_image_power_slope",
        "modulation_power_slope",
        "abs_modulation_power_slope",
        "modulation_spatial_flatness",
        "modulation_spatial_entropy",
        "modulation_total_power",
        "transfer_slope",
        "transfer_slope_error_to_2",
    )
    for metric in metrics:
        for row in summarize_groups(rows, keys, metric):
            row["metric"] = metric
            out.append(row)
    return out


def metric_mean_table(rows: list[dict[str, Any]], group_keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    """Average raw movie rows over group keys before a second-stage summary."""
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    group_values: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        group_values.setdefault(key, {k: row.get(k, "") for k in group_keys})
        for metric in metrics:
            value = row.get(metric)
            if value is None:
                continue
            try:
                fval = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fval):
                grouped[key][metric].append(fval)
    out: list[dict[str, Any]] = []
    for key in sorted(grouped):
        item = dict(group_values[key])
        for metric in metrics:
            vals = grouped[key].get(metric, [])
            item[metric] = float(np.mean(vals)) if vals else float("nan")
        out.append(item)
    return out


def summarize_second_stage(rows: list[dict[str, Any]], *, unit_name: str) -> list[dict[str, Any]]:
    keys = ("motion_family", "amplitude_scale", "diffusion_scale")
    out: list[dict[str, Any]] = []
    for metric in PRIMARY_METRICS:
        for row in summarize_groups(rows, keys, metric):
            row["metric"] = metric
            row["summary_unit"] = unit_name
            out.append(row)
    return out


def decision_table(
    scale_rows: list[dict[str, Any]],
    *,
    min_power_by_family: dict[str, float] | None = None,
    decision_scope: str = "all_nonzero",
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scale_rows:
        grouped[(str(row["motion_family"]), str(row["metric"]))].append(row)
    maximize = {
        "modulation_spatial_flatness": True,
        "modulation_spatial_entropy": True,
        "modulation_total_power": True,
        "abs_modulation_power_slope": False,
        "transfer_slope_error_to_2": False,
    }
    out: list[dict[str, Any]] = []
    for (family, metric), rows in sorted(grouped.items()):
        if metric not in maximize:
            continue
        row_by_scale = {float(row["amplitude_scale"]): row for row in rows}
        scales = np.asarray([float(row["amplitude_scale"]) for row in rows], dtype=np.float64)
        means = np.asarray([float(row["mean"]) for row in rows], dtype=np.float64)
        keep = np.isfinite(scales) & np.isfinite(means)
        if decision_scope != "all":
            keep &= scales > 0.0
        if min_power_by_family is not None:
            power_min = float(min_power_by_family.get(family, 0.0))
            power = np.asarray(
                [float(row_by_scale.get(float(scale), {}).get("modulation_total_power_mean", float("nan"))) for scale in scales],
                dtype=np.float64,
            )
            keep &= np.isfinite(power) & (power >= power_min)
        scales = scales[keep]
        means = means[keep]
        if means.size == 0:
            continue
        idx = int(np.nanargmax(means) if maximize[metric] else np.nanargmin(means))
        bio_idx = int(np.nanargmin(np.abs(scales - 1.0)))
        opt = float(means[idx])
        bio = float(means[bio_idx])
        out.append(
            {
                "motion_family": family,
                "metric": metric,
                "optimum_rule": "argmax" if maximize[metric] else "argmin",
                "amplitude_scale_opt": float(scales[idx]),
                "diffusion_scale_opt": float(scales[idx] ** 2),
                "value_at_opt": opt,
                "nearest_biological_amplitude_scale": float(scales[bio_idx]),
                "value_at_biological": bio,
                "biological_fraction_of_peak": bio / opt if maximize[metric] and opt != 0 else float("nan"),
                "distance_from_biological_amplitude_scale": abs(float(scales[idx]) - 1.0),
                "boundary_call": "upper_boundary"
                if np.isclose(scales[idx], np.nanmax(scales))
                else ("lower_boundary" if np.isclose(scales[idx], np.nanmin(scales)) else "interior"),
                "n_scales": int(scales.size),
                "decision_scope": decision_scope,
            }
        )
    return out


def attach_power_means(scale_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy total-power means onto each metric row for power-gated decisions."""
    power_by_key = {
        (row["motion_family"], row["amplitude_scale"], row["diffusion_scale"]): row["mean"]
        for row in scale_rows
        if row.get("metric") == "modulation_total_power"
    }
    out: list[dict[str, Any]] = []
    for row in scale_rows:
        item = dict(row)
        item["modulation_total_power_mean"] = power_by_key.get(
            (row["motion_family"], row["amplitude_scale"], row["diffusion_scale"]),
            float("nan"),
        )
        out.append(item)
    return out


def power_thresholds(scale_rows: list[dict[str, Any]], fraction_of_bio: float) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for row in scale_rows:
        if row.get("metric") != "modulation_total_power":
            continue
        if not np.isclose(float(row.get("amplitude_scale", float("nan"))), 1.0):
            continue
        thresholds[str(row["motion_family"])] = max(0.0, float(row["mean"]) * float(fraction_of_bio))
    return thresholds


def per_scale_readout(scale_rows: list[dict[str, Any]], trace_rows: list[dict[str, Any]], image_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wide table with requested per-scale metrics and trace/image CIs."""
    base: dict[tuple[str, float, float], dict[str, Any]] = {}
    for row in scale_rows:
        key = (str(row["motion_family"]), float(row["amplitude_scale"]), float(row["diffusion_scale"]))
        item = base.setdefault(key, {"motion_family": key[0], "amplitude_scale": key[1], "diffusion_scale": key[2]})
        metric = str(row["metric"])
        item[f"{metric}_mean"] = float(row["mean"])
        item[f"{metric}_ci_low_movie"] = float(row["ci_low"])
        item[f"{metric}_ci_high_movie"] = float(row["ci_high"])
        item[f"{metric}_n_movie"] = int(row["n"])
    for source, suffix in ((trace_rows, "trace"), (image_rows, "image")):
        for row in source:
            key = (str(row["motion_family"]), float(row["amplitude_scale"]), float(row["diffusion_scale"]))
            item = base.setdefault(key, {"motion_family": key[0], "amplitude_scale": key[1], "diffusion_scale": key[2]})
            metric = str(row["metric"])
            item[f"{metric}_mean_by_{suffix}"] = float(row["mean"])
            item[f"{metric}_ci_low_{suffix}"] = float(row["ci_low"])
            item[f"{metric}_ci_high_{suffix}"] = float(row["ci_high"])
            item[f"{metric}_n_{suffix}"] = int(row["n"])
    return [base[key] for key in sorted(base, key=lambda k: (k[0], k[1]))]


def small_motion_sanity_rows(movie_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scales = sorted({float(row["amplitude_scale"]) for row in movie_rows if float(row["amplitude_scale"]) > 0})
    if not scales:
        return []
    scale = scales[0]
    rows = [row for row in movie_rows if np.isclose(float(row["amplitude_scale"]), scale)]
    out: list[dict[str, Any]] = []
    for metric, target in (
        ("transfer_slope", 2.0),
        ("transfer_slope_error_to_2", 0.0),
        ("modulation_power_slope", None),
        ("source_image_power_slope", None),
    ):
        vals = [float(row[metric]) for row in rows if np.isfinite(float(row.get(metric, float("nan"))))]
        mean, sem, n = mean_sem(vals)
        ci_low = float(np.percentile(vals, 2.5)) if vals else float("nan")
        ci_high = float(np.percentile(vals, 97.5)) if vals else float("nan")
        item = {
            "sanity_check": "small_motion_derivative_transfer",
            "amplitude_scale": scale,
            "diffusion_scale": scale ** 2,
            "metric": metric,
            "target": target if target is not None else "",
            "mean": mean,
            "sem": sem,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": n,
        }
        if target is not None and np.isfinite(mean):
            item["mean_minus_target"] = mean - target
        out.append(item)
    return out


def write_figures(out_dir: Path, scale_rows: list[dict[str, Any]]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = (
        "abs_modulation_power_slope",
        "modulation_spatial_flatness",
        "transfer_slope_error_to_2",
        "modulation_total_power",
    )
    fig, axs = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 3.2), squeeze=False)
    for ax, metric in zip(axs[0], metrics, strict=True):
        rows = [row for row in scale_rows if row.get("metric") == metric]
        for family in sorted({str(row["motion_family"]) for row in rows}):
            sub = sorted([row for row in rows if row["motion_family"] == family], key=lambda row: float(row["amplitude_scale"]))
            ax.plot([float(row["amplitude_scale"]) for row in sub], [float(row["mean"]) for row in sub], marker="o", label=family)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("amplitude scale")
        ax.set_title(metric.replace("_", " "))
    axs[0, 0].set_ylabel("mean")
    axs[0, -1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "rucci_powerlaw_scale_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    trace_rows: list[dict[str, Any]],
    crop_rows: list[dict[str, str]],
    decisions: list[dict[str, Any]],
    power_gated_decisions: list[dict[str, Any]],
    readout_rows: list[dict[str, Any]],
    sanity_rows: list[dict[str, Any]],
) -> None:
    primary_metrics = {"abs_modulation_power_slope", "transfer_slope_error_to_2", "modulation_spatial_flatness"}
    source_trace_count = len({int(row.get("source_trace_index", -1)) for row in trace_rows})
    lines = [
        "# Rucci Power-Law Whitening Audit",
        "",
        "This audit measures spatial-frequency power-law flattening of frame-to-frame retinal modulation, not pooled temporal PSD flatness.",
        "",
        "## Run",
        "",
        f"- Trace source: `{args.trace_source}`",
        f"- Fixation traces: `{len(trace_rows)}`",
        f"- Source trace indices: `{source_trace_count}`",
        f"- Image/crops: `{len(crop_rows)}`",
        f"- Amplitude scales: `{','.join(str(v) for v in parse_float_list(args.amplitude_scales))}`",
        f"- Fit band: `{float(args.fit_low_cpd):g}-{float(args.fit_high_cpd):g} cpd`",
        f"- Tiny-power exclusion: modulation total power >= `{float(args.min_power_fraction_of_bio):g}` x biological-scale total power",
        "",
        "## Primary Decisions",
        "",
    ]
    for row in decisions:
        if row.get("metric") not in primary_metrics:
            continue
        lines.append(
            f"- `{row['metric']}`: amplitude_scale_opt=`{float(row['amplitude_scale_opt']):.6g}` "
            f"(diffusion scale `{float(row['diffusion_scale_opt']):.6g}`), "
            f"boundary=`{row['boundary_call']}`, biological value=`{float(row['value_at_biological']):.6g}`."
        )
    lines.extend(["", "## Power-Gated Decisions", ""])
    for row in power_gated_decisions:
        if row.get("metric") not in primary_metrics:
            continue
        lines.append(
            f"- `{row['metric']}`: amplitude_scale_opt=`{float(row['amplitude_scale_opt']):.6g}` "
            f"(diffusion scale `{float(row['diffusion_scale_opt']):.6g}`), "
            f"boundary=`{row['boundary_call']}`, biological value=`{float(row['value_at_biological']):.6g}`."
        )
    lines.extend(["", "## Biological-Scale Readout", ""])
    bio = next((row for row in readout_rows if np.isclose(float(row["amplitude_scale"]), 1.0)), None)
    small = next((row for row in readout_rows if float(row["amplitude_scale"]) > 0.0), None)
    if bio is not None:
        lines.append(
            f"- At 1x biological amplitude: modulation slope `{bio.get('modulation_power_slope_mean', float('nan')):.6g}`, "
            f"transfer slope `{bio.get('transfer_slope_mean', float('nan')):.6g}`, "
            f"flatness `{bio.get('modulation_spatial_flatness_mean', float('nan')):.6g}`, "
            f"total power `{bio.get('modulation_total_power_mean', float('nan')):.6g}`."
        )
    if bio is not None and small is not None:
        power_ratio = float(bio.get("modulation_total_power_mean", float("nan"))) / max(float(small.get("modulation_total_power_mean", float("nan"))), 1e-30)
        lines.append(
            f"- 1x versus smallest nonzero scale (`{small['amplitude_scale']:.6g}`): "
            f"total power ratio `{power_ratio:.6g}`, "
            f"flatness delta `{float(bio.get('modulation_spatial_flatness_mean', float('nan'))) - float(small.get('modulation_spatial_flatness_mean', float('nan'))):.6g}`, "
            f"transfer-error delta `{float(bio.get('transfer_slope_error_to_2_mean', float('nan'))) - float(small.get('transfer_slope_error_to_2_mean', float('nan'))):.6g}`."
        )
    lines.extend(["", "## Small-Motion Sanity Check", ""])
    for row in sanity_rows:
        if row["metric"] not in {"transfer_slope", "transfer_slope_error_to_2"}:
            continue
        target = row.get("target", "")
        target_text = f", target `{float(target):.6g}`" if target != "" else ""
        lines.append(
            f"- `{row['metric']}` at amplitude `{float(row['amplitude_scale']):.6g}`: "
            f"mean `{float(row['mean']):.6g}`{target_text}, CI [`{float(row['ci_low']):.6g}`, `{float(row['ci_high']):.6g}`], n=`{int(row['n'])}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- `amplitude_scale` multiplies eye-position displacement; approximate diffusion/power scale is `amplitude_scale^2`.",
            "- The primary Rucci-style endpoint is spatial flattening of modulation power. Entropy/flatness and total modulation power are diagnostics, not sufficient evidence of efficient whitening.",
            "- Slopes are fit per trace/image/crop movie and then aggregated, so high-power examples do not dominate the headline slope.",
            "- `transfer_slope` is the log-log slope of modulation power divided by source image power; in the small-translation derivative limit, the target is approximately `+2`.",
            "- Runs using `trace_source=selected` inherit the small selected-trace pool from the source twininfo run and should be treated as smoke/provenance checks.",
            "",
        ]
    )
    (out_dir / "rucci_powerlaw_whitening_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_run_config(run_dir)
    t_max = parse_int(config.get("t_max", 128), 128)
    trace_rows = load_trace_pool(args, t_max)
    if not trace_rows:
        raise RuntimeError("No fixation traces available for Rucci whitening audit.")
    crop_rows = selected_crops(run_dir, max_images=int(args.max_images), max_crops=int(args.max_crops))
    if not crop_rows:
        raise RuntimeError("No image/crop rows available for Rucci whitening audit.")
    images = image_lookup(crop_rows)
    amplitude_scales = parse_float_list(args.amplitude_scales)

    movie_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    fit_low = float(args.fit_low_cpd)
    fit_high = float(args.fit_high_cpd)
    for crop in crop_rows:
        image_index = int(crop["image_index"])
        image = images[image_index]
        crop_offset = (float(crop["offset_x_px"]), float(crop["offset_y_px"]))
        for trace_row in trace_rows:
            trace = np.asarray(trace_row["trace"], dtype=np.float32)[:t_max]
            source_movie = retinal_movie_from_image_trace(image, stable_trace(trace), t_max=t_max, crop_center_offset_px=crop_offset)
            source_radial = radial_power(
                source_movie[0],
                ppd=PPD,
                fit_low_cpd=fit_low,
                fit_high_cpd=fit_high,
                n_bins=int(args.n_radial_bins),
            )
            source_power = np.asarray(source_radial["power"], dtype=np.float64)
            source_freq = np.asarray(source_radial["freq_cpd"], dtype=np.float64)
            source_fit = np.asarray(source_radial["fit_mask"], dtype=bool)
            for scale in amplitude_scales:
                if str(args.motion_family) != "scaled_measured_drift":
                    raise ValueError(f"Unsupported motion family for this audit: {args.motion_family}")
                motion_trace = stable_trace(trace) if float(scale) == 0.0 else scale_trace(trace, float(scale))
                movie = retinal_movie_from_image_trace(image, motion_trace, t_max=t_max, crop_center_offset_px=crop_offset)
                temporal_derivative = np.diff(movie, axis=0) / float(DT)
                modulation_radial = radial_power(
                    temporal_derivative,
                    ppd=PPD,
                    fit_low_cpd=fit_low,
                    fit_high_cpd=fit_high,
                    n_bins=int(args.n_radial_bins),
                )
                modulation_power = np.asarray(modulation_radial["power"], dtype=np.float64)
                modulation_freq = np.asarray(modulation_radial["freq_cpd"], dtype=np.float64)
                fit = np.asarray(modulation_radial["fit_mask"], dtype=bool)
                ratio = modulation_power / np.maximum(source_power, 1e-30)
                transfer_slope = loglog_slope(modulation_freq[fit], ratio[fit])
                row = {
                    "movie_id": len(movie_rows),
                    "example_id": str(trace_row.get("example_id", "")),
                    "kind": str(trace_row.get("kind", "")),
                    "source_trace_index": int(trace_row.get("source_trace_index", -1)),
                    "window_start": int(trace_row.get("window_start", -1)),
                    "image_index": image_index,
                    "crop_rank": int(crop["crop_rank"]),
                    "crop_center_offset_x_px": float(crop["offset_x_px"]),
                    "crop_center_offset_y_px": float(crop["offset_y_px"]),
                    "motion_family": "scaled_measured_drift",
                    "amplitude_scale": float(scale),
                    "diffusion_scale": float(scale) ** 2,
                    "fit_low_cpd": fit_low,
                    "fit_high_cpd": fit_high,
                    "source_image_power_slope": float(source_radial["fit_slope"]),
                    "source_image_spatial_flatness": float(source_radial["fit_flatness"]),
                    "source_image_spatial_entropy": float(source_radial["fit_entropy"]),
                    "modulation_power_slope": float(modulation_radial["fit_slope"]),
                    "abs_modulation_power_slope": abs(float(modulation_radial["fit_slope"]))
                    if np.isfinite(float(modulation_radial["fit_slope"]))
                    else float("nan"),
                    "modulation_spatial_flatness": float(modulation_radial["fit_flatness"]),
                    "modulation_spatial_entropy": float(modulation_radial["fit_entropy"]),
                    "modulation_total_power": float(modulation_radial["fit_total_power"]),
                    "transfer_slope": transfer_slope,
                    "transfer_slope_error_to_2": abs(transfer_slope - 2.0) if np.isfinite(transfer_slope) else float("nan"),
                    "nyquist_cpd": float(modulation_radial["nyquist_cpd"]),
                }
                movie_rows.append(row)
                if bool(args.write_radial_rows):
                    for freq, src_p, mod_p, n_coeff, in_fit in zip(
                        source_freq,
                        source_power,
                        modulation_power,
                        np.asarray(modulation_radial["n_coefficients"], dtype=np.int64),
                        source_fit & fit,
                        strict=True,
                    ):
                        radial_rows.append(
                            {
                                **{k: row[k] for k in ("movie_id", "example_id", "image_index", "crop_rank", "motion_family", "amplitude_scale", "diffusion_scale")},
                                "spatial_frequency_cpd": float(freq),
                                "source_image_power": float(src_p),
                                "modulation_power": float(mod_p),
                                "transfer_power_ratio": float(mod_p / max(src_p, 1e-30)),
                                "n_fourier_coefficients": int(n_coeff),
                                "in_fit_band": bool(in_fit),
                            }
                        )

    scale_rows = summarize_scale(movie_rows)
    scale_rows_with_power = attach_power_means(scale_rows)
    trace_mean_rows = metric_mean_table(
        movie_rows,
        ("motion_family", "amplitude_scale", "diffusion_scale", "example_id", "source_trace_index"),
        PRIMARY_METRICS,
    )
    image_mean_rows = metric_mean_table(
        movie_rows,
        ("motion_family", "amplitude_scale", "diffusion_scale", "image_index"),
        PRIMARY_METRICS,
    )
    trace_scale_rows = summarize_second_stage(trace_mean_rows, unit_name="trace")
    image_scale_rows = summarize_second_stage(image_mean_rows, unit_name="image")
    readout_rows = per_scale_readout(scale_rows, trace_scale_rows, image_scale_rows)
    power_min = power_thresholds(scale_rows, float(args.min_power_fraction_of_bio))
    decisions = decision_table(scale_rows_with_power, decision_scope="all_nonzero")
    power_gated_decisions = decision_table(
        scale_rows_with_power,
        min_power_by_family=power_min,
        decision_scope="nonzero_power_gated",
    )
    sanity_rows = small_motion_sanity_rows(movie_rows)
    write_csv_rows(out_dir / "rucci_powerlaw_movie_metrics.csv", movie_rows)
    write_csv_rows(out_dir / "rucci_powerlaw_scale_summary.csv", scale_rows)
    write_csv_rows(out_dir / "rucci_powerlaw_trace_scale_summary.csv", trace_scale_rows)
    write_csv_rows(out_dir / "rucci_powerlaw_image_scale_summary.csv", image_scale_rows)
    write_csv_rows(out_dir / "rucci_powerlaw_per_scale_readout.csv", readout_rows)
    write_csv_rows(out_dir / "rucci_powerlaw_decision_table.csv", decisions)
    write_csv_rows(out_dir / "rucci_powerlaw_power_gated_decision_table.csv", power_gated_decisions)
    write_csv_rows(out_dir / "rucci_powerlaw_sanity_checks.csv", sanity_rows)
    write_csv_rows(out_dir / "rucci_powerlaw_radial_power_by_movie.csv", radial_rows)
    write_figures(out_dir, scale_rows)
    write_summary(
        out_dir,
        args=args,
        trace_rows=trace_rows,
        crop_rows=crop_rows,
        decisions=decisions,
        power_gated_decisions=power_gated_decisions,
        readout_rows=readout_rows,
        sanity_rows=sanity_rows,
    )
    write_json(
        out_dir / "rucci_powerlaw_manifest.json",
        {
            "analysis": "rucci_powerlaw_whitening_audit",
            "run_dir": run_dir,
            "out_dir": out_dir,
            "trace_source": args.trace_source,
            "n_traces": len(trace_rows),
            "n_crops": len(crop_rows),
            "t_max": t_max,
            "dt_s": DT,
            "ppd": PPD,
            "amplitude_scales": amplitude_scales,
            "diffusion_scales": [float(v) ** 2 for v in amplitude_scales],
            "fit_band_cpd": [fit_low, fit_high],
            "n_movie_rows": len(movie_rows),
            "n_radial_rows": len(radial_rows),
            "n_trace_scale_summary_rows": len(trace_scale_rows),
            "n_image_scale_summary_rows": len(image_scale_rows),
            "min_power_fraction_of_bio": float(args.min_power_fraction_of_bio),
            "power_thresholds_by_family": power_min,
            "claim_boundary": "Spatial power-law modulation audit; does not run digital twin responses.",
        },
    )
    print(f"Wrote Rucci power-law whitening audit to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_TWININFO_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trace-source", choices=("selected", "raw_fixation"), default="selected")
    parser.add_argument("--max-traces", type=int, default=0)
    parser.add_argument("--max-source-traces", type=int, default=0)
    parser.add_argument("--trace-stride", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-crops", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amplitude-scales", default=",".join(str(v) for v in DEFAULT_AMPLITUDE_SCALES))
    parser.add_argument("--motion-family", default="scaled_measured_drift")
    parser.add_argument("--fit-low-cpd", type=float, default=DEFAULT_FIT_BAND_CPD[0])
    parser.add_argument("--fit-high-cpd", type=float, default=DEFAULT_FIT_BAND_CPD[1])
    parser.add_argument("--n-radial-bins", type=int, default=14)
    parser.add_argument("--min-power-fraction-of-bio", type=float, default=0.05)
    parser.add_argument("--write-radial-rows", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
