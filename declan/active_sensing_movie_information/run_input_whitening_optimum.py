#!/usr/bin/env python3
"""Run input-level retinal temporal whitening over FEM scale.

This analysis deliberately avoids twin responses.  It uses natural images,
selected crops, and real fixation traces from a production ``jake.twininfo``
run, renders retinal luminance movies, and asks which motion scale whitens the
temporal input spectrum.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

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
from jake.twininfo.retinal_examples import retinal_movie_from_image_trace
from jake.twininfo.stimuli import load_natural_images


DEFAULT_OUT_DIR = DEFAULT_STACK_OUT_DIR / "input_whitening"
DEFAULT_SCALES = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
DEFAULT_TEMPORAL_LOWS = (0.5, 1.0, 2.0)
DEFAULT_TEMPORAL_HIGHS = (20.0, 30.0, 60.0)
DEFAULT_SPATIAL_LOWS = (2.0, 4.0, 8.0)
DEFAULT_SPATIAL_HIGHS = (20.0, 30.0, 40.0, 60.0)
PRIMARY_TEMPORAL_PASSBAND = (1.0, 30.0)
PRIMARY_SPATIAL_PASSBAND = (4.0, 40.0)


def stable_seed(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest()[:8], 16)


def fit_drift_diffusion(
    traces: list[np.ndarray],
    *,
    dt: float,
    lag_min_ms: float,
    lag_max_ms: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Estimate Brownian diffusion from fixation traces via MSD ~= 4 D tau."""
    lag_min = max(1, int(round(float(lag_min_ms) / 1000.0 / float(dt))))
    lag_max = max(lag_min, int(round(float(lag_max_ms) / 1000.0 / float(dt))))
    rows: list[dict[str, Any]] = []
    all_tau: list[float] = []
    all_msd: list[float] = []
    for i, trace in enumerate(traces):
        tr = np.asarray(trace, dtype=np.float64)
        trace_tau: list[float] = []
        trace_msd: list[float] = []
        for lag in range(lag_min, min(lag_max, tr.shape[0] - 1) + 1):
            disp = tr[lag:] - tr[:-lag]
            msd = float(np.mean(np.sum(disp * disp, axis=1)))
            tau = float(lag * dt)
            trace_tau.append(tau)
            trace_msd.append(msd)
            all_tau.append(tau)
            all_msd.append(msd)
        fit = fit_msd_line(np.asarray(trace_tau), np.asarray(trace_msd))
        rows.append(
            {
                "trace_index": i,
                "D_eye_deg2_per_s": fit["D"],
                "D_eye_arcmin2_per_s": fit["D"] * 3600.0,
                "fit_lag_min_ms": lag_min_ms,
                "fit_lag_max_ms": lag_max_ms,
                "fit_r2": fit["r2"],
                "n_lags": len(trace_tau),
            }
        )
    pooled = fit_msd_line(np.asarray(all_tau), np.asarray(all_msd))
    summary = {
        "D_eye_deg2_per_s": pooled["D"],
        "D_eye_arcmin2_per_s": pooled["D"] * 3600.0,
        "fit_lag_min_ms": lag_min_ms,
        "fit_lag_max_ms": lag_max_ms,
        "fit_r2": pooled["r2"],
        "n_trace_windows": len(traces),
        "n_lag_samples": len(all_tau),
    }
    return rows, summary


def fit_msd_line(tau_s: np.ndarray, msd_deg2: np.ndarray) -> dict[str, float]:
    tau = np.asarray(tau_s, dtype=np.float64)
    msd = np.asarray(msd_deg2, dtype=np.float64)
    keep = np.isfinite(tau) & np.isfinite(msd)
    tau = tau[keep]
    msd = msd[keep]
    if tau.size < 2:
        return {"D": float("nan"), "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    x = np.column_stack([tau, np.ones_like(tau)])
    slope, intercept = np.linalg.lstsq(x, msd, rcond=None)[0]
    pred = slope * tau + intercept
    ss_res = float(np.sum((msd - pred) ** 2))
    ss_tot = float(np.sum((msd - np.mean(msd)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"D": float(max(slope / 4.0, 0.0)), "slope": float(slope), "intercept": float(intercept), "r2": r2}


def synthetic_brownian(trace: np.ndarray, D_eye: float, scale: float, *, dt: float, seed: int) -> np.ndarray:
    tr = np.asarray(trace, dtype=np.float32)
    center = np.mean(tr, axis=0, keepdims=True).astype(np.float32)
    D = max(float(D_eye) * float(scale) ** 2, 0.0)
    if D == 0.0:
        return stable_trace(tr)
    rng = np.random.default_rng(int(seed))
    steps = rng.normal(scale=np.sqrt(2.0 * D * float(dt)), size=(tr.shape[0] - 1, 2)).astype(np.float32)
    walk = np.vstack([np.zeros((1, 2), dtype=np.float32), np.cumsum(steps, axis=0)])
    walk -= np.mean(walk, axis=0, keepdims=True)
    return (center + walk).astype(np.float32)


def synthetic_ou(
    trace: np.ndarray,
    D_eye: float,
    scale: float,
    *,
    dt: float,
    seed: int,
    tau_s: float = 0.25,
) -> np.ndarray:
    tr = np.asarray(trace, dtype=np.float32)
    center = np.mean(tr, axis=0, keepdims=True).astype(np.float32)
    D = max(float(D_eye) * float(scale) ** 2, 0.0)
    if D == 0.0:
        return stable_trace(tr)
    rng = np.random.default_rng(int(seed))
    theta = 1.0 / max(float(tau_s), float(dt))
    x = np.zeros_like(tr, dtype=np.float32)
    sigma = np.sqrt(2.0 * D * float(dt))
    for t in range(1, tr.shape[0]):
        x[t] = x[t - 1] * (1.0 - theta * float(dt)) + rng.normal(scale=sigma, size=2)
    x -= np.mean(x, axis=0, keepdims=True)
    return (center + x).astype(np.float32)


def spatial_bandpass(movie: np.ndarray, low_cpd: float, high_cpd: float, *, ppd: float) -> np.ndarray:
    arr = np.asarray(movie, dtype=np.float32)
    if low_cpd <= 0 and not np.isfinite(high_cpd):
        return arr
    h, w = arr.shape[1:]
    fy = np.fft.fftfreq(h, d=1.0 / float(ppd))
    fx = np.fft.fftfreq(w, d=1.0 / float(ppd))
    rr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    mask = (rr >= float(low_cpd)) & (rr <= float(high_cpd))
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = np.fft.ifft2(np.fft.fft2(arr[i]) * mask).real.astype(np.float32)
    return out


def temporal_psd(movie: np.ndarray, *, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frequencies, pixel-averaged temporal PSD, and contrast movie."""
    arr = np.asarray(movie, dtype=np.float32)
    contrast = arr - np.mean(arr, axis=0, keepdims=True)
    window = np.hanning(arr.shape[0]).astype(np.float32)
    norm = float(np.sum(window * window))
    fft = np.fft.rfft(contrast * window[:, None, None], axis=0)
    psd = np.mean(np.abs(fft) ** 2, axis=(1, 2)) / max(norm, 1e-12)
    freq = np.fft.rfftfreq(arr.shape[0], d=float(dt))
    return freq.astype(np.float32), psd.astype(np.float64), contrast.astype(np.float32)


def autocorrelation_time(contrast: np.ndarray, *, dt: float, max_pixels: int = 4096) -> float:
    arr = np.asarray(contrast, dtype=np.float32).reshape(contrast.shape[0], -1)
    if arr.shape[1] > int(max_pixels):
        step = max(1, arr.shape[1] // int(max_pixels))
        arr = arr[:, ::step]
    arr = arr - np.mean(arr, axis=0, keepdims=True)
    denom = np.sum(arr * arr, axis=0)
    good = denom > 1e-12
    if not np.any(good):
        return 0.0
    arr = arr[:, good]
    denom = denom[good]
    n = arr.shape[0]
    ac = np.zeros(n, dtype=np.float64)
    for lag in range(n):
        ac[lag] = float(np.mean(np.sum(arr[: n - lag] * arr[lag:], axis=0) / denom))
    positive = ac > 0
    if not np.any(positive):
        return 0.0
    stop = int(np.argmax(~positive)) if np.any(~positive) else n
    stop = max(stop, 1)
    return float(np.trapz(ac[:stop], dx=float(dt)))


def whitening_metrics(
    freq: np.ndarray,
    psd: np.ndarray,
    contrast: np.ndarray,
    *,
    temporal_low_hz: float,
    temporal_high_hz: float,
    dt: float,
) -> dict[str, float]:
    f = np.asarray(freq, dtype=np.float64)
    p = np.asarray(psd, dtype=np.float64)
    band = (f >= float(temporal_low_hz)) & (f <= float(temporal_high_hz)) & np.isfinite(p) & (p > 0)
    if int(np.sum(band)) < 2:
        slope = float("nan")
        entropy = float("nan")
        flatness = float("nan")
        power = float("nan")
    else:
        x = np.log10(f[band])
        y = np.log10(p[band])
        slope = float(np.polyfit(x, y, 1)[0])
        vals = p[band]
        probs = vals / max(float(np.sum(vals)), 1e-30)
        entropy_raw = -float(np.sum(probs * np.log(probs + 1e-30)))
        entropy = entropy_raw / np.log(vals.size)
        flatness = float(np.exp(np.mean(np.log(vals + 1e-30))) / max(float(np.mean(vals)), 1e-30))
        power = float(np.trapz(vals, f[band]))
    return {
        "loglog_temporal_psd_slope": slope,
        "abs_loglog_temporal_psd_slope": abs(slope) if np.isfinite(slope) else float("nan"),
        "spectral_entropy": entropy,
        "spectral_flatness": flatness,
        "autocorrelation_time_s": autocorrelation_time(contrast, dt=dt),
        "temporal_power_in_passband": power,
    }


def passband_grid(args: argparse.Namespace) -> list[tuple[float, float, float, float, str]]:
    spatial_lows = parse_float_list(args.spatial_lows)
    spatial_highs = parse_float_list(args.spatial_highs)
    temporal_lows = parse_float_list(args.temporal_lows)
    temporal_highs = parse_float_list(args.temporal_highs)
    out: list[tuple[float, float, float, float, str]] = []
    for slo in spatial_lows:
        for shi in spatial_highs:
            if shi <= slo:
                continue
            for tlo in temporal_lows:
                for thi in temporal_highs:
                    if thi <= tlo:
                        continue
                    label = f"spatial_{slo:g}_{shi:g}cpd__temporal_{tlo:g}_{thi:g}hz".replace(".", "p")
                    out.append((slo, shi, tlo, thi, label))
    return out


def build_movie_manifest(
    *,
    args: argparse.Namespace,
    traces: dict[str, dict[str, Any]],
    crops: list[dict[str, str]],
    D_eye: float,
    t_max: int,
) -> list[dict[str, Any]]:
    trace_items = [item for item in traces.values() if str(item.get("kind", "")) == str(args.trace_kind)]
    if int(args.max_traces) > 0:
        trace_items = trace_items[: int(args.max_traces)]
    crop_items = crops
    if int(args.max_crops) > 0:
        crop_items = crop_items[: int(args.max_crops)]
    allowed_images = None
    if int(args.max_images) > 0:
        allowed_images = {int(row["image_index"]) for row in crop_items[: int(args.max_images)]}
    scales = parse_float_list(args.scales)
    families = set(args.families.split(","))
    rows: list[dict[str, Any]] = []
    for crop in crop_items:
        image_index = int(crop["image_index"])
        if allowed_images is not None and image_index not in allowed_images:
            continue
        for trace_row in trace_items:
            trace = np.asarray(trace_row["trace"], dtype=np.float32)[:t_max]
            for family in sorted(families):
                for scale in scales:
                    if family == "scaled_measured_drift_D":
                        motion_trace = stable_trace(trace) if scale == 0.0 else scale_trace(trace, scale)
                    elif family == "synthetic_brownian_D":
                        motion_trace = synthetic_brownian(
                            trace,
                            D_eye,
                            scale,
                            dt=DT,
                            seed=stable_seed(trace_row["example_id"], image_index, crop["crop_rank"], family, scale),
                        )
                    elif family == "synthetic_ou_D":
                        motion_trace = synthetic_ou(
                            trace,
                            D_eye,
                            scale,
                            dt=DT,
                            seed=stable_seed(trace_row["example_id"], image_index, crop["crop_rank"], family, scale),
                        )
                    elif family == "stabilized":
                        if scale != 0.0:
                            continue
                        motion_trace = stable_trace(trace)
                    else:
                        raise ValueError(f"Unknown motion family: {family}")
                    rows.append(
                        {
                            "movie_id": len(rows),
                            "example_id": str(trace_row["example_id"]),
                            "kind": str(trace_row.get("kind", "")),
                            "image_index": image_index,
                            "crop_rank": int(crop["crop_rank"]),
                            "crop_center_offset_x_px": float(crop["offset_x_px"]),
                            "crop_center_offset_y_px": float(crop["offset_y_px"]),
                            "motion_family": family,
                            "D_scale": float(scale),
                            "synthetic_D_eye_deg2_per_s": float(D_eye) * float(scale) ** 2,
                            "trace": motion_trace,
                        }
                    )
    return rows


def image_lookup(movie_rows: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    indices = sorted({int(row["image_index"]) for row in movie_rows})
    loaded = load_natural_images(len(indices), indices=tuple(indices))
    return {int(spec.image_index): image for spec, image in loaded if spec.image_index is not None}


def summarize_scale(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("motion_family", "D_scale", "spatial_low_cpd", "spatial_high_cpd", "temporal_low_hz", "temporal_high_hz", "passband_label")
    out: list[dict[str, Any]] = []
    for metric in (
        "loglog_temporal_psd_slope",
        "abs_loglog_temporal_psd_slope",
        "spectral_entropy",
        "spectral_flatness",
        "autocorrelation_time_s",
        "temporal_power_in_passband",
    ):
        for row in summarize_groups(metrics_rows, keys, metric):
            row["metric"] = metric
            out.append(row)
    return out


def passband_sensitivity(scale_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in scale_summary:
        grouped.setdefault((str(row["motion_family"]), str(row["passband_label"]), str(row["metric"])), []).append(row)
    for (family, label, metric), vals in sorted(grouped.items()):
        if family == "stabilized":
            continue
        if metric == "abs_loglog_temporal_psd_slope":
            best = min(vals, key=lambda row: float(row["mean"]) if np.isfinite(float(row["mean"])) else np.inf)
            optimum = "argmin_abs_slope"
        elif metric in {"spectral_entropy", "spectral_flatness"}:
            best = max(vals, key=lambda row: float(row["mean"]) if np.isfinite(float(row["mean"])) else -np.inf)
            optimum = "argmax"
        else:
            continue
        rows.append(
            {
                "motion_family": family,
                "passband_label": label,
                "metric": metric,
                "optimum_rule": optimum,
                "D_opt": float(best["D_scale"]),
                "mean_at_opt": float(best["mean"]),
                "distance_from_biological_D1": abs(float(best["D_scale"]) - 1.0),
            }
        )
    return rows


def write_figures(out_dir: Path, metrics_rows: list[dict[str, Any]], scale_summary: list[dict[str, Any]]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    primary_label = f"spatial_{PRIMARY_SPATIAL_PASSBAND[0]:g}_{PRIMARY_SPATIAL_PASSBAND[1]:g}cpd__temporal_{PRIMARY_TEMPORAL_PASSBAND[0]:g}_{PRIMARY_TEMPORAL_PASSBAND[1]:g}hz".replace(".", "p")
    rows = [
        row
        for row in scale_summary
        if row.get("passband_label") == primary_label and row.get("metric") in {"abs_loglog_temporal_psd_slope", "spectral_entropy", "spectral_flatness"}
    ]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), squeeze=False)
    for ax, metric in zip(axs[0], ("abs_loglog_temporal_psd_slope", "spectral_entropy", "spectral_flatness"), strict=True):
        for family in sorted({str(row["motion_family"]) for row in rows if row["metric"] == metric}):
            vals = sorted([row for row in rows if row["metric"] == metric and row["motion_family"] == family], key=lambda row: float(row["D_scale"]))
            ax.plot([float(row["D_scale"]) for row in vals], [float(row["mean"]) for row in vals], marker="o", label=family)
        ax.axvline(1.0, color="#444444", linestyle="--", linewidth=0.9)
        ax.set_xlabel("D scale")
        ax.set_title(metric.replace("_", " "))
    axs[0, 0].set_ylabel("mean")
    axs[0, -1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "whitening_scale_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    examples = [row for row in metrics_rows if row.get("passband_label") == primary_label]
    examples = examples[: min(36, len(examples))]
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for row in examples:
        ax.scatter(float(row["D_scale"]), float(row["abs_loglog_temporal_psd_slope"]), s=14, alpha=0.4)
    ax.axvline(1.0, color="#444444", linestyle="--", linewidth=0.9)
    ax.set_xlabel("D scale")
    ax.set_ylabel("abs temporal PSD slope")
    ax.set_title("movie-level whitening examples")
    fig.tight_layout()
    fig.savefig(fig_dir / "retinal_temporal_psd_examples.pdf", bbox_inches="tight")
    plt.close(fig)

    sens_rows = passband_sensitivity(scale_summary)
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    d_vals = [float(row["D_opt"]) for row in sens_rows if row["metric"] == "abs_loglog_temporal_psd_slope"]
    if d_vals:
        ax.hist(d_vals, bins=np.arange(-0.25, 3.51, 0.25), color="#9fb8cc", edgecolor="white")
    ax.axvline(1.0, color="#444444", linestyle="--", linewidth=0.9)
    ax.set_xlabel("D optimum across passbands")
    ax.set_ylabel("count")
    ax.set_title("passband sensitivity")
    fig.tight_layout()
    fig.savefig(fig_dir / "whitening_passband_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def write_bootstrap_placeholder(out_dir: Path) -> None:
    """Document that bootstrap stability remains to be run."""
    path = out_dir / "whitening_paired_bootstrap.md"
    path.write_text(
        "# Whitening Paired Bootstrap\n\n"
        "Status: not computed by this runner yet.\n\n"
        "The whitening smoke/full metric tables are available, but image/crop "
        "bootstrap resampling still needs a dedicated implementation before "
        "`whitening_paired_bootstrap.csv` should be produced.\n",
        encoding="utf-8",
    )


def write_summary_md(out_dir: Path, drift_summary: dict[str, Any], sensitivity_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Input Whitening Summary",
        "",
        f"Estimated biological drift diffusion: D={drift_summary['D_eye_deg2_per_s']:.6g} deg^2/s "
        f"({drift_summary['D_eye_arcmin2_per_s']:.6g} arcmin^2/s), R2={drift_summary['fit_r2']:.3g}.",
        "",
        "## Whitening Optima",
        "",
    ]
    primary_label = f"spatial_{PRIMARY_SPATIAL_PASSBAND[0]:g}_{PRIMARY_SPATIAL_PASSBAND[1]:g}cpd__temporal_{PRIMARY_TEMPORAL_PASSBAND[0]:g}_{PRIMARY_TEMPORAL_PASSBAND[1]:g}hz".replace(".", "p")
    primary = [row for row in sensitivity_rows if row["passband_label"] == primary_label]
    for row in primary:
        lines.append(
            f"- {row['motion_family']} / {row['metric']}: D_opt={row['D_opt']:.6g}, "
            f"distance from biological D=1 is {row['distance_from_biological_D1']:.6g}"
        )
    if not primary:
        lines.append("- Primary passband was not present in this run.")
    lines.extend(
        [
            "",
            "Caveat: whitening is an input-statistics signature, not proof of cortical utility or global optimality.",
            "",
        ]
    )
    (out_dir / "whitening_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_run_config(run_dir)
    t_max = parse_int(config.get("t_max", 128), 128)
    traces = load_selected_traces(run_dir)
    trace_list = [np.asarray(row["trace"], dtype=np.float32)[:t_max] for row in traces.values() if str(row.get("kind", "")) == str(args.trace_kind)]
    if int(args.max_traces) > 0:
        trace_list = trace_list[: int(args.max_traces)]
    drift_rows, drift_summary = fit_drift_diffusion(
        trace_list,
        dt=DT,
        lag_min_ms=float(args.fit_lag_min_ms),
        lag_max_ms=float(args.fit_lag_max_ms),
    )
    crops = load_crop_rows(run_dir)
    movie_rows = build_movie_manifest(args=args, traces=traces, crops=crops, D_eye=float(drift_summary["D_eye_deg2_per_s"]), t_max=t_max)
    image_by_index = image_lookup(movie_rows)
    passbands = passband_grid(args)

    movie_manifest_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    psd_rows: list[dict[str, Any]] = []
    primary_label = f"spatial_{PRIMARY_SPATIAL_PASSBAND[0]:g}_{PRIMARY_SPATIAL_PASSBAND[1]:g}cpd__temporal_{PRIMARY_TEMPORAL_PASSBAND[0]:g}_{PRIMARY_TEMPORAL_PASSBAND[1]:g}hz".replace(".", "p")
    for movie_row in movie_rows:
        image = image_by_index[int(movie_row["image_index"])]
        trace = np.asarray(movie_row.pop("trace"), dtype=np.float32)
        crop_offset = (float(movie_row["crop_center_offset_x_px"]), float(movie_row["crop_center_offset_y_px"]))
        movie = retinal_movie_from_image_trace(image, trace, t_max=t_max, crop_center_offset_px=crop_offset)
        movie_manifest_rows.append(dict(movie_row))
        for spatial_low, spatial_high, temporal_low, temporal_high, label in passbands:
            filtered = spatial_bandpass(movie, spatial_low, spatial_high, ppd=PPD)
            freq, psd, contrast = temporal_psd(filtered, dt=DT)
            metrics = whitening_metrics(
                freq,
                psd,
                contrast,
                temporal_low_hz=temporal_low,
                temporal_high_hz=temporal_high,
                dt=DT,
            )
            metric_row = {
                **movie_row,
                "spatial_low_cpd": spatial_low,
                "spatial_high_cpd": spatial_high,
                "temporal_low_hz": temporal_low,
                "temporal_high_hz": temporal_high,
                "passband_label": label,
                **metrics,
            }
            metrics_rows.append(metric_row)
            if str(args.write_psd_rows).lower() in {"1", "true", "yes"} and label == primary_label:
                for f, p in zip(freq, psd, strict=True):
                    psd_rows.append({**movie_row, "passband_label": label, "temporal_frequency_hz": float(f), "temporal_psd": float(p)})

    scale_summary = summarize_scale(metrics_rows)
    sensitivity = passband_sensitivity(scale_summary)
    write_csv_rows(out_dir / "drift_diffusion_estimates.csv", [{**row, **{"scope": "trace"}} for row in drift_rows] + [{**drift_summary, "scope": "pooled"}])
    write_csv_rows(out_dir / "whitening_movie_manifest.csv", movie_manifest_rows)
    write_csv_rows(out_dir / "whitening_movie_metrics.csv", metrics_rows)
    write_csv_rows(out_dir / "retinal_temporal_psd_by_movie.csv", psd_rows)
    write_csv_rows(out_dir / "whitening_scale_summary.csv", scale_summary)
    write_csv_rows(out_dir / "whitening_passband_sensitivity.csv", sensitivity)
    write_bootstrap_placeholder(out_dir)
    write_figures(out_dir, metrics_rows, scale_summary)
    write_summary_md(out_dir, drift_summary, sensitivity)
    write_json(
        out_dir / "input_whitening_manifest.json",
        {
            "run_dir": run_dir,
            "out_dir": out_dir,
            "t_max": t_max,
            "dt_s": DT,
            "ppd": PPD,
            "scales": parse_float_list(args.scales),
            "families": sorted(set(args.families.split(","))),
            "n_movies": len(movie_manifest_rows),
            "n_metric_rows": len(metrics_rows),
            "n_psd_rows": len(psd_rows),
            "psd_rows_scope": "primary_passband_only" if str(args.write_psd_rows).lower() in {"1", "true", "yes"} else "not_written",
            "drift_summary": drift_summary,
            "bootstrap_status": "not_computed",
            "primary_temporal_passband_hz": PRIMARY_TEMPORAL_PASSBAND,
            "primary_spatial_passband_cpd": PRIMARY_SPATIAL_PASSBAND,
        },
    )
    print(f"Wrote input-whitening outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_TWININFO_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scales", default=",".join(str(v) for v in DEFAULT_SCALES))
    parser.add_argument("--families", default="scaled_measured_drift_D,synthetic_brownian_D,synthetic_ou_D")
    parser.add_argument("--trace-kind", default="fixation")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-crops", type=int, default=0)
    parser.add_argument("--max-traces", type=int, default=0)
    parser.add_argument("--fit-lag-min-ms", type=float, default=8.0)
    parser.add_argument("--fit-lag-max-ms", type=float, default=80.0)
    parser.add_argument("--spatial-lows", default=",".join(str(v) for v in DEFAULT_SPATIAL_LOWS))
    parser.add_argument("--spatial-highs", default=",".join(str(v) for v in DEFAULT_SPATIAL_HIGHS))
    parser.add_argument("--temporal-lows", default=",".join(str(v) for v in DEFAULT_TEMPORAL_LOWS))
    parser.add_argument("--temporal-highs", default=",".join(str(v) for v in DEFAULT_TEMPORAL_HIGHS))
    parser.add_argument("--write-psd-rows", default="false")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
