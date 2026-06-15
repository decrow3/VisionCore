#!/usr/bin/env python3
"""Estimate a digital-twin temporal sensitivity curve from controlled flicker probes.

The whitening analyses can ask whether retinal movie spectra are flat after
weighting temporal frequencies by what the fitted V1-like model can actually
use.  This script measures that weighting curve directly: it drives the twin
with sinusoidal temporal contrast modulations at fixed spatial frequencies,
then estimates the population response amplitude at each temporal frequency.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from non_circular_fem_common import DEFAULT_STACK_OUT_DIR, write_csv_rows, write_json

from declan.vernier_active_sensing.forward import load_model_and_readout
from scripts.spatial_info import embed_time_lags
from scripts.temporal_decoding.rate_computation import OUT_SIZE, PPD, compute_trial_rates
from scripts.temporal_decoding.stimulus_hires import N_LAGS


DEFAULT_OUT_DIR = DEFAULT_STACK_OUT_DIR / "model_temporal_sensitivity"


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def make_probe_movie(
    *,
    temporal_frequency_hz: float,
    spatial_cpd: float,
    phase_rad: float,
    n_valid_frames: int,
    n_lags: int,
    frame_rate_hz: float,
    image_size: tuple[int, int],
    ppd: float,
    base_luminance: float,
    contrast: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    h, w = int(image_size[0]), int(image_size[1])
    total_frames = int(n_valid_frames) + int(n_lags) - 1
    frame_idx = np.arange(total_frames, dtype=np.float64) - (int(n_lags) - 1)
    t = frame_idx / float(frame_rate_hz)

    x_deg = (np.arange(w, dtype=np.float64) - (w - 1) / 2.0) / float(ppd)
    spatial_1d = np.sin(2.0 * np.pi * float(spatial_cpd) * x_deg)
    spatial = np.tile(spatial_1d[None, :], (h, 1))
    spatial = spatial / max(float(np.sqrt(np.mean(spatial**2))), 1e-12)

    temporal = np.sin(2.0 * np.pi * float(temporal_frequency_hz) * t + float(phase_rad))
    movie = float(base_luminance) + float(contrast) * temporal[:, None, None] * spatial[None, :, :]
    clipped_fraction = float(np.mean((movie < 0.0) | (movie > 1.0)))
    movie = np.clip(movie, 0.0, 1.0).astype(np.float32)
    stim = embed_time_lags(torch.from_numpy(movie), n_lags=int(n_lags))
    stats = {
        "stimulus_rms_contrast": float(contrast),
        "clipped_fraction": clipped_fraction,
        "movie_min": float(np.min(movie)),
        "movie_max": float(np.max(movie)),
    }
    return stim, stats


def response_amplitude(
    rates: np.ndarray,
    *,
    temporal_frequency_hz: float,
    frame_rate_hz: float,
    discard_frames: int,
) -> dict[str, float]:
    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim != 2 or rates.shape[0] <= int(discard_frames) + 4:
        return {
            "n_analysis_frames": 0,
            "n_units": int(rates.shape[1]) if rates.ndim == 2 else 0,
            "mean_rate": float("nan"),
            "response_amp_mean": float("nan"),
            "response_amp_rms": float("nan"),
            "response_gain_sq": float("nan"),
        }
    y = rates[int(discard_frames) :]
    t = (np.arange(y.shape[0], dtype=np.float64) + int(discard_frames)) / float(frame_rate_hz)
    omega_t = 2.0 * np.pi * float(temporal_frequency_hz) * t
    design = np.column_stack([np.sin(omega_t), np.cos(omega_t), np.ones_like(omega_t)])
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    amp = np.sqrt(coeff[0] ** 2 + coeff[1] ** 2)
    return {
        "n_analysis_frames": int(y.shape[0]),
        "n_units": int(y.shape[1]),
        "mean_rate": float(np.mean(rates[int(discard_frames) :])),
        "response_amp_mean": float(np.mean(amp)),
        "response_amp_rms": float(np.sqrt(np.mean(amp**2))),
        "response_gain_sq": float(np.mean(amp**2)),
    }


def mean_sem(values: Iterable[float]) -> tuple[float, float, int]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    if arr.size == 1:
        return float(arr[0]), 0.0, 1
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / math.sqrt(arr.size)), int(arr.size)


def aggregate_curve(probe_rows: list[dict[str, Any]], spatial_aggregation: str) -> list[dict[str, Any]]:
    rows_for_temporal_curve = list(probe_rows)
    if spatial_aggregation == "within_spatial_cpd_normalized_mean":
        max_by_spatial: dict[float, float] = {}
        for row in probe_rows:
            cpd = float(row["spatial_cpd"])
            value = float(row["response_gain_sq"])
            if np.isfinite(value):
                max_by_spatial[cpd] = max(max_by_spatial.get(cpd, 0.0), value)
        rows_for_temporal_curve = []
        for row in probe_rows:
            cpd = float(row["spatial_cpd"])
            denom = max_by_spatial.get(cpd, 0.0)
            normalized_row = dict(row)
            normalized_row["temporal_curve_gain_sq"] = float(row["response_gain_sq"]) / denom if denom > 0 else float("nan")
            rows_for_temporal_curve.append(normalized_row)
    elif spatial_aggregation == "raw_gain_sq_mean":
        rows_for_temporal_curve = []
        for row in probe_rows:
            normalized_row = dict(row)
            normalized_row["temporal_curve_gain_sq"] = float(row["response_gain_sq"])
            rows_for_temporal_curve.append(normalized_row)
    else:
        raise ValueError(f"Unsupported spatial aggregation: {spatial_aggregation}")

    by_freq: dict[float, list[dict[str, Any]]] = {}
    for row in rows_for_temporal_curve:
        by_freq.setdefault(float(row["temporal_frequency_hz"]), []).append(row)
    raw_rows: list[dict[str, Any]] = []
    for freq, rows in sorted(by_freq.items()):
        mean_gain, sem_gain, n = mean_sem(float(row["temporal_curve_gain_sq"]) for row in rows)
        mean_amp, sem_amp, _ = mean_sem(float(row["response_amp_rms"]) for row in rows)
        raw_rows.append(
            {
                "weight_name": "model_response_gain_sq",
                "temporal_frequency_hz": freq,
                "mean_response_gain_sq": mean_gain,
                "sem_response_gain_sq": sem_gain,
                "mean_response_amp_rms": mean_amp,
                "sem_response_amp_rms": sem_amp,
                "n": n,
                "spatial_aggregation": spatial_aggregation,
            }
        )
    max_gain = max([float(row["mean_response_gain_sq"]) for row in raw_rows if np.isfinite(float(row["mean_response_gain_sq"]))] or [0.0])
    for row in raw_rows:
        row["normalized_weight"] = float(row["mean_response_gain_sq"]) / max_gain if max_gain > 0 else float("nan")
    return raw_rows


def write_figures(out_dir: Path, curve_rows: list[dict[str, Any]], probe_rows: list[dict[str, Any]]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    freq = np.asarray([float(row["temporal_frequency_hz"]) for row in curve_rows], dtype=np.float64)
    weight = np.asarray([float(row["normalized_weight"]) for row in curve_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(freq, weight, marker="o", color="#3366aa")
    ax.set_xscale("log")
    ax.set_xlabel("Temporal frequency (Hz)")
    ax.set_ylabel("Normalized model weight")
    ax.set_title("Model-derived temporal sensitivity")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "model_temporal_sensitivity_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    spatial_cpds = sorted({float(row["spatial_cpd"]) for row in probe_rows})
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for cpd in spatial_cpds:
        sub = [row for row in probe_rows if np.isclose(float(row["spatial_cpd"]), cpd)]
        by_freq: dict[float, list[float]] = {}
        for row in sub:
            by_freq.setdefault(float(row["temporal_frequency_hz"]), []).append(float(row["response_gain_sq"]))
        xs = np.asarray(sorted(by_freq), dtype=np.float64)
        ys = np.asarray([np.mean(by_freq[float(x)]) for x in xs], dtype=np.float64)
        if np.nanmax(ys) > 0:
            ys = ys / float(np.nanmax(ys))
        ax.plot(xs, ys, marker="o", label=f"{cpd:g} cpd")
    ax.set_xscale("log")
    ax.set_xlabel("Temporal frequency (Hz)")
    ax.set_ylabel("Within-SF normalized gain^2")
    ax.set_title("Probe sensitivity by spatial frequency")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "model_temporal_sensitivity_by_spatial_frequency.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(out_dir: Path, curve_rows: list[dict[str, Any]], probe_rows: list[dict[str, Any]]) -> None:
    if curve_rows:
        best = max(curve_rows, key=lambda row: float(row["normalized_weight"]) if np.isfinite(float(row["normalized_weight"])) else -np.inf)
        peak_line = f"Peak measured weight: {float(best['temporal_frequency_hz']):.6g} Hz"
    else:
        peak_line = "Peak measured weight: unavailable"
    lines = [
        "# Model Temporal Sensitivity Summary",
        "",
        "This is a digital-twin-derived temporal transfer estimate from sinusoidal contrast probes.",
        "",
        f"- probe rows: {len(probe_rows)}",
        f"- temporal-frequency rows: {len(curve_rows)}",
        f"- {peak_line}",
        "",
        "## Use",
        "",
        "Pass `temporal_sensitivity_curve.csv` to `summarize_v1_weighted_whitening.py --external-weight-csv` to weight input-whitening PSDs by this measured curve.",
        "",
        "## Claim Boundary",
        "",
        "The curve is a controlled probe of model response gain, not a recorded-neuron transfer function. Its interpretation depends on the chosen spatial frequencies, contrast, spatial aggregation, readout collapse, and framewise lag-embedded inference path.",
        "",
    ]
    (out_dir / "model_temporal_sensitivity_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frequencies = parse_float_list(args.frequencies)
    spatial_cpds = parse_float_list(args.spatial_cpds)
    phases = [2.0 * np.pi * idx / max(int(args.n_phases), 1) for idx in range(max(int(args.n_phases), 1))]
    n_valid_frames = max(int(round(float(args.duration_s) * float(args.frame_rate_hz))), int(args.n_lags) + 8)
    discard_frames = min(int(args.discard_frames), max(n_valid_frames // 2 - 1, 0))
    image_size = (int(args.image_size), int(args.image_size))
    if n_valid_frames - discard_frames < float(args.frame_rate_hz):
        print(
            "Warning: less than 1 second remains after discard; low-frequency amplitudes may be unstable.",
            flush=True,
        )

    model, readout = load_model_and_readout(device=args.device)
    probe_rows: list[dict[str, Any]] = []
    total = len(frequencies) * len(spatial_cpds) * len(phases)
    done = 0
    for freq in frequencies:
        for cpd in spatial_cpds:
            for phase_idx, phase in enumerate(phases):
                stim, stim_stats = make_probe_movie(
                    temporal_frequency_hz=freq,
                    spatial_cpd=cpd,
                    phase_rad=float(phase),
                    n_valid_frames=n_valid_frames,
                    n_lags=int(args.n_lags),
                    frame_rate_hz=float(args.frame_rate_hz),
                    image_size=image_size,
                    ppd=float(args.ppd),
                    base_luminance=float(args.base_luminance),
                    contrast=float(args.contrast),
                )
                rates = compute_trial_rates(
                    model,
                    readout,
                    stim,
                    batch_size=int(args.batch_size),
                    spatial_collapse=str(args.spatial_collapse),
                )
                amp = response_amplitude(
                    rates,
                    temporal_frequency_hz=freq,
                    frame_rate_hz=float(args.frame_rate_hz),
                    discard_frames=discard_frames,
                )
                done += 1
                row = {
                    "temporal_frequency_hz": float(freq),
                    "spatial_cpd": float(cpd),
                    "phase_index": int(phase_idx),
                    "phase_rad": float(phase),
                    "n_valid_frames": int(n_valid_frames),
                    "discard_frames": int(discard_frames),
                    "frame_rate_hz": float(args.frame_rate_hz),
                    "spatial_collapse": str(args.spatial_collapse),
                    "amplitude_estimator": "sin_cos_lstsq",
                    **stim_stats,
                    **amp,
                }
                row["response_gain_sq_per_contrast"] = float(row["response_gain_sq"]) / max(float(args.contrast) ** 2, 1e-12)
                probe_rows.append(row)
                print(
                    f"[{done}/{total}] f={freq:g} Hz sf={cpd:g} cpd phase={phase_idx} "
                    f"gain_sq={row['response_gain_sq']:.6g}",
                    flush=True,
                )
                del stim, rates
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    curve_rows = aggregate_curve(probe_rows, spatial_aggregation=str(args.spatial_aggregation))
    write_csv_rows(out_dir / "model_temporal_sensitivity_probes.csv", probe_rows)
    write_csv_rows(out_dir / "temporal_sensitivity_curve.csv", curve_rows)
    write_figures(out_dir, curve_rows, probe_rows)
    write_summary(out_dir, curve_rows, probe_rows)
    write_json(
        out_dir / "model_temporal_sensitivity_manifest.json",
        {
            "analysis": "model_temporal_sensitivity",
            "out_dir": out_dir,
            "device": args.device,
            "frequencies": frequencies,
            "spatial_cpds": spatial_cpds,
            "n_phases": int(args.n_phases),
            "duration_s": float(args.duration_s),
            "frame_rate_hz": float(args.frame_rate_hz),
            "n_lags": int(args.n_lags),
            "n_valid_frames": int(n_valid_frames),
            "discard_frames": int(discard_frames),
            "image_size": image_size,
            "ppd": float(args.ppd),
            "base_luminance": float(args.base_luminance),
            "contrast": float(args.contrast),
            "batch_size": int(args.batch_size),
            "spatial_collapse": args.spatial_collapse,
            "spatial_aggregation": args.spatial_aggregation,
            "amplitude_estimator": "sin_cos_lstsq",
            "n_probe_rows": len(probe_rows),
            "n_curve_rows": len(curve_rows),
            "outputs": {
                "probe_csv": out_dir / "model_temporal_sensitivity_probes.csv",
                "curve_csv": out_dir / "temporal_sensitivity_curve.csv",
                "summary": out_dir / "model_temporal_sensitivity_summary.md",
            },
            "claim_boundary": "Digital-twin controlled-probe response-gain curve; not a recorded temporal transfer function.",
        },
    )
    print(f"Wrote model temporal sensitivity outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frequencies", default="1,2,4,8,12,16,24,32,48")
    parser.add_argument("--spatial-cpds", default="4,8,16")
    parser.add_argument("--n-phases", type=int, default=2)
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--frame-rate-hz", type=float, default=120.0)
    parser.add_argument("--n-lags", type=int, default=N_LAGS)
    parser.add_argument("--discard-frames", type=int, default=N_LAGS)
    parser.add_argument("--image-size", type=int, default=OUT_SIZE[0])
    parser.add_argument("--ppd", type=float, default=PPD)
    parser.add_argument("--base-luminance", type=float, default=0.5)
    parser.add_argument("--contrast", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--spatial-collapse", choices=("max", "mean"), default="max")
    parser.add_argument(
        "--spatial-aggregation",
        choices=("within_spatial_cpd_normalized_mean", "raw_gain_sq_mean"),
        default="within_spatial_cpd_normalized_mean",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
