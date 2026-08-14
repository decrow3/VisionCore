#!/usr/bin/env python3
"""Map-first audit of composite RR100 input-space effective receptive fields.

The prior pooling-scale checkpoint measures only the learned 14 x 14 Gaussian
readout masks.  This checkpoint backpropagates center-map RR100 rates through
the complete convolutional/recurrent core and spatial readout to the genuine
32-lag, 151 x 151 production input.  It saves squared-gradient spatial maps,
lag-energy profiles, per-unit scale measurements, and a predeclared readout-
quantile example sheet.  It is an input-design checkpoint; it does not build a
phase surrogate or calculate surrogate neural responses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import torch

from declan.fig4_active_sensing.make_rr100_global_3d_phase_scramble_explicit_history_checkpoint import (
    scored_lag_stack,
)
from declan.redundancy_resolved_v1_population import (
    load_canonical_twin_bundle,
    load_population_view,
)


ROOT = Path(__file__).resolve().parents[2]
RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)
SOURCE = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_global_3d_phase_scramble_explicit_history_checkpoint_45_v4/"
    "data/example_2_image_068.npz"
)
READOUT_AUDIT = ROOT / (
    "outputs/fig4_active_sensing/rr100_spatial_filter_pooling_scale_checkpoint_41_v1/"
    "canonical_spatial_readout_scales.csv"
)
OUT = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_composite_effective_rf_pooling_scale_checkpoint_48_v1"
)
PPD = 37.50476617
N_HISTORY = 32
N_SCORE = 40
FRACTIONS = (0.50, 0.80, 0.90, 0.95)
EXAMPLE_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
EPS = np.finfo(np.float64).tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--readout-audit", type=Path, default=READOUT_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", default="0,19,39")
    parser.add_argument("--max-units", type=int, default=0)
    parser.add_argument("--ppd", type=float, default=PPD)
    parser.add_argument("--crop-size", type=int, default=61)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rr100-version", default=RR100_VERSION)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_frames(text: str) -> list[int]:
    frames = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not frames or len(frames) != len(set(frames)):
        raise ValueError(f"Frames must be a nonempty unique list, got {frames}")
    if min(frames) < 0 or max(frames) >= N_SCORE:
        raise ValueError(f"Scored-frame indices must lie in [0,{N_SCORE - 1}], got {frames}")
    return frames


def select_readout_quantile_examples(readout: pd.DataFrame) -> pd.DataFrame:
    metric = "mask_energy_radius_90_stimulus_px"
    work = readout.sort_values("rr100_index").copy()
    available = work.copy()
    rows: list[dict[str, Any]] = []
    for quantile in EXAMPLE_QUANTILES:
        target = float(np.quantile(work[metric].to_numpy(np.float64), quantile))
        if available.empty:
            break
        index = (available[metric] - target).abs().idxmin()
        selected = available.loc[index].to_dict()
        selected.update(
            {
                "selection_role": f"readout_energy_radius_q{int(round(100 * quantile)):02d}",
                "selection_criterion": "closest RR100 unit to pre-gradient readout energy-radius quantile",
                "selection_metric": metric,
                "selection_target_quantile": quantile,
                "selection_target_value": target,
                "selection_value": float(selected[metric]),
                "selection_is_algorithmic": True,
            }
        )
        rows.append(selected)
        available = available.drop(index=index)
    return pd.DataFrame(rows).sort_values("selection_target_quantile").reset_index(drop=True)


def geometry(weight: np.ndarray, *, prefix: str) -> dict[str, float]:
    values = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    total = float(values.sum())
    if not np.isfinite(total) or total <= EPS:
        raise ValueError(f"Degenerate {prefix} map with total={total}")
    values /= total
    yy, xx = np.mgrid[: values.shape[0], : values.shape[1]]
    cy = float(np.sum(values * yy))
    cx = float(np.sum(values * xx))
    dy = yy - cy
    dx = xx - cx
    covariance = np.asarray(
        [
            [np.sum(values * dy * dy), np.sum(values * dy * dx)],
            [np.sum(values * dy * dx), np.sum(values * dx * dx)],
        ],
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    sigma_major, sigma_minor = np.sqrt(eigenvalues)
    major_y, major_x = eigenvectors[:, 0]
    radius = np.hypot(dy, dx)
    flat_order = np.argsort(radius.ravel())
    sorted_radius = radius.ravel()[flat_order]
    cumulative = np.cumsum(values.ravel()[flat_order])
    edge = int(min(5, values.shape[0] // 4, values.shape[1] // 4))
    edge_fraction = float(
        values[:edge].sum()
        + values[-edge:].sum()
        + values[edge:-edge, :edge].sum()
        + values[edge:-edge, -edge:].sum()
    )
    center_y = (values.shape[0] - 1) / 2.0
    center_x = (values.shape[1] - 1) / 2.0
    result = {
        f"{prefix}_center_y_px": cy,
        f"{prefix}_center_x_px": cx,
        f"{prefix}_center_offset_px": float(np.hypot(cy - center_y, cx - center_x)),
        f"{prefix}_sigma_major_px": float(sigma_major),
        f"{prefix}_sigma_minor_px": float(sigma_minor),
        f"{prefix}_sigma_geomean_px": float(np.sqrt(max(sigma_major * sigma_minor, 0.0))),
        f"{prefix}_axis_ratio": float(sigma_major / max(sigma_minor, EPS)),
        f"{prefix}_principal_angle_deg": float(np.degrees(np.arctan2(major_y, major_x))),
        f"{prefix}_outer_5px_fraction": edge_fraction,
    }
    for fraction in FRACTIONS:
        tag = int(round(100 * fraction))
        index = min(int(np.searchsorted(cumulative, fraction)), sorted_radius.size - 1)
        result[f"{prefix}_radius_{tag}_px"] = float(sorted_radius[index])
    return result


def lag_metrics(lag_energy: np.ndarray) -> dict[str, float]:
    values = np.maximum(np.asarray(lag_energy, dtype=np.float64), 0.0)
    values /= max(float(values.sum()), EPS)
    lag = np.arange(values.size, dtype=np.float64)
    cumulative = np.cumsum(values)
    result = {
        "lag_energy_mean_frames_ago": float(np.sum(values * lag)),
        "lag_energy_sigma_frames": float(np.sqrt(np.sum(values * (lag - np.sum(values * lag)) ** 2))),
        "lag_energy_current_fraction": float(values[0]),
        "lag_energy_recent_4_fraction": float(values[:4].sum()),
        "lag_energy_recent_8_fraction": float(values[:8].sum()),
        "lag_energy_recent_16_fraction": float(values[:16].sum()),
    }
    for fraction in FRACTIONS:
        tag = int(round(100 * fraction))
        result[f"lag_energy_radius_{tag}_frames"] = float(np.searchsorted(cumulative, fraction))
    return result


def add_physical_units(row: dict[str, Any], *, ppd: float) -> None:
    for key in list(row):
        if key.startswith("gradient_energy_") and key.endswith("_px"):
            stem = key[: -len("_px")]
            degrees = float(row[key]) / float(ppd)
            row[f"{stem}_deg"] = degrees
            row[f"{stem}_arcmin"] = 60.0 * degrees


def enclosed_crop_fraction(weight: np.ndarray, crop_size: int) -> float:
    values = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    half = int(crop_size) // 2
    cy, cx = values.shape[0] // 2, values.shape[1] // 2
    crop = values[cy - half : cy + half + 1, cx - half : cx + half + 1]
    return float(crop.sum() / max(float(values.sum()), EPS))


def forward_rate_maps(bundle: Any, stim: torch.Tensor) -> torch.Tensor:
    core = bundle.model.model.core_forward(stim, None)
    logits = bundle.readout(core[:, :, -1])
    return bundle.model.model.activation(logits)


def measure_gradients(
    bundle: Any,
    stim_all: np.ndarray,
    frames: list[int],
    rr_channels: np.ndarray,
    rr_indices: np.ndarray,
    *,
    ppd: float,
    crop_size: int,
    selected_rr: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    device = torch.device(bundle.device)
    dtype = next(bundle.model.model.parameters()).dtype
    rows: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []
    spatial_maps = np.empty((len(frames), len(rr_indices), 151, 151), dtype=np.float32)
    selected_gradients: dict[str, np.ndarray] = {}

    for parameter in bundle.model.model.parameters():
        parameter.requires_grad_(False)
    bundle.model.model.eval()
    bundle.readout.eval()

    for frame_ordinal, frame_index in enumerate(frames):
        x = torch.as_tensor(stim_all[frame_index : frame_index + 1], device=device, dtype=dtype).clone()
        x.requires_grad_(True)
        rates = forward_rate_maps(bundle, x)
        center_y, center_x = int(rates.shape[-2] // 2), int(rates.shape[-1] // 2)
        print(
            f"frame {frame_index}: input={tuple(x.shape)} rate_map={tuple(rates.shape)} "
            f"center=({center_y},{center_x})",
            flush=True,
        )
        for unit_ordinal, rr_index in enumerate(rr_indices):
            canonical_channel = int(rr_channels[int(rr_index)])
            retain = unit_ordinal < len(rr_indices) - 1
            gradient = torch.autograd.grad(
                rates[0, canonical_channel, center_y, center_x],
                x,
                retain_graph=retain,
                create_graph=False,
            )[0][0, 0]
            gradient_np = gradient.detach().cpu().numpy().astype(np.float32)
            energy_lag_xy = np.square(gradient_np, dtype=np.float32)
            energy_xy = energy_lag_xy.sum(axis=0, dtype=np.float64)
            lag_energy = energy_lag_xy.sum(axis=(1, 2), dtype=np.float64)
            spatial_maps[frame_ordinal, unit_ordinal] = energy_xy.astype(np.float32)
            row: dict[str, Any] = {
                "rr100_index": int(rr_index),
                "canonical_channel": canonical_channel,
                "scored_frame_index": int(frame_index),
                "input_current_movie_frame_index": int(N_HISTORY + frame_index),
                "center_output_y": center_y,
                "center_output_x": center_x,
                "center_rate_hz": float(rates[0, canonical_channel, center_y, center_x].detach().cpu()),
                "total_gradient_energy": float(energy_xy.sum()),
                "central_crop_size_px": int(crop_size),
                "central_crop_gradient_energy_fraction": enclosed_crop_fraction(energy_xy, int(crop_size)),
                **geometry(energy_xy, prefix="gradient_energy"),
                **lag_metrics(lag_energy),
            }
            add_physical_units(row, ppd=float(ppd))
            rows.append(row)
            normalized_lag = lag_energy / max(float(lag_energy.sum()), EPS)
            lag_rows.extend(
                {
                    "rr100_index": int(rr_index),
                    "canonical_channel": canonical_channel,
                    "scored_frame_index": int(frame_index),
                    "lag_frames_ago": int(lag),
                    "lag_energy": float(value),
                    "lag_energy_fraction": float(fraction),
                }
                for lag, (value, fraction) in enumerate(zip(lag_energy, normalized_lag, strict=True))
            )
            if int(rr_index) in selected_rr:
                selected_gradients[f"rr{int(rr_index):03d}_frame{int(frame_index):02d}"] = gradient_np
            if (unit_ordinal + 1) % 10 == 0 or unit_ordinal + 1 == len(rr_indices):
                print(f"  completed {unit_ordinal + 1}/{len(rr_indices)} RR100 gradients", flush=True)
        del rates, x
    return pd.DataFrame(rows), pd.DataFrame(lag_rows), spatial_maps, selected_gradients


def summarize_units(measurements: pd.DataFrame, readout: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in measurements.columns
        if column not in {"rr100_index", "canonical_channel", "scored_frame_index"}
        and pd.api.types.is_numeric_dtype(measurements[column])
    ]
    median = measurements.groupby(["rr100_index", "canonical_channel"], as_index=False)[numeric].median()
    variation = (
        measurements.groupby("rr100_index")["gradient_energy_radius_90_px"]
        .agg(composite_radius90_min_px="min", composite_radius90_max_px="max", composite_radius90_std_px="std")
        .reset_index()
    )
    result = median.merge(variation, on="rr100_index", validate="one_to_one")
    keep = [
        "rr100_index",
        "mask_sigma_major_stimulus_px",
        "mask_sigma_minor_stimulus_px",
        "mask_sigma_geomean_stimulus_px",
        "mask_mass_radius_90_stimulus_px",
        "mask_energy_radius_90_stimulus_px",
        "mask_axis_ratio",
        "mask_principal_angle_deg",
        "mask_edge_mass",
    ]
    result = result.merge(readout[keep], on="rr100_index", validate="one_to_one")
    result["composite_to_readout_energy_radius90_ratio"] = (
        result["gradient_energy_radius_90_px"]
        / np.maximum(result["mask_energy_radius_90_stimulus_px"], EPS)
    )
    result["composite_radius90_frame_range_px"] = (
        result["composite_radius90_max_px"] - result["composite_radius90_min_px"]
    )
    return result.sort_values("rr100_index").reset_index(drop=True)


def quantile_table(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "gradient_energy_sigma_major_px",
        "gradient_energy_sigma_minor_px",
        "gradient_energy_sigma_geomean_px",
        "gradient_energy_radius_50_px",
        "gradient_energy_radius_80_px",
        "gradient_energy_radius_90_px",
        "gradient_energy_radius_95_px",
        "gradient_energy_axis_ratio",
        "gradient_energy_outer_5px_fraction",
        "central_crop_gradient_energy_fraction",
        "lag_energy_mean_frames_ago",
        "lag_energy_radius_90_frames",
        "composite_to_readout_energy_radius90_ratio",
        "composite_radius90_frame_range_px",
    ]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = summary[metric].to_numpy(np.float64)
        for quantile in (0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0):
            rows.append(
                {
                    "population": "rr100_movie_medoids",
                    "n_units": int(values.size),
                    "metric": metric,
                    "quantile": quantile,
                    "value": float(np.quantile(values, quantile)),
                }
            )
    return pd.DataFrame(rows)


def crop_center(values: np.ndarray, crop_size: int) -> tuple[np.ndarray, int, int]:
    half = int(crop_size) // 2
    cy, cx = values.shape[-2] // 2, values.shape[-1] // 2
    return values[..., cy - half : cy + half + 1, cx - half : cx + half + 1], cy - half, cx - half


def plot_input_and_distributions(
    movie72: np.ndarray,
    score_trace: np.ndarray,
    frames: list[int],
    summary: pd.DataFrame,
    measurements: pd.DataFrame,
    *,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 8.0), constrained_layout=True)
    for column, frame_index in enumerate(frames[:3]):
        axes[0, column].imshow(movie72[N_HISTORY + frame_index], cmap="gray", vmin=0, vmax=255, origin="lower")
        axes[0, column].set_title(f"scored frame {frame_index}\nmovie frame {N_HISTORY + frame_index}")
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
    trace_arcmin = np.asarray(score_trace, dtype=np.float64) * 60.0
    axes[0, 3].plot(trace_arcmin[:, 0], trace_arcmin[:, 1], color="#333333", lw=1.2)
    for frame_index, color in zip(frames[:3], ("#0072B2", "#D55E00", "#009E73"), strict=False):
        axes[0, 3].scatter(
            trace_arcmin[frame_index, 0], trace_arcmin[frame_index, 1], color=color, s=42, label=f"frame {frame_index}"
        )
    axes[0, 3].set_aspect("equal", adjustable="datalim")
    axes[0, 3].set(xlabel="horizontal (arcmin)", ylabel="vertical (arcmin)", title="same FEM trace")
    axes[0, 3].legend(frameon=False, fontsize=8)

    axes[1, 0].hist(
        summary["mask_energy_radius_90_stimulus_px"], bins=18, alpha=0.65, label="readout only", color="#999999"
    )
    axes[1, 0].hist(
        summary["gradient_energy_radius_90_px"], bins=18, alpha=0.65, label="composite", color="#0072B2"
    )
    axes[1, 0].set(xlabel="90% energy radius (stimulus px)", ylabel="units", title="RR100 scale distribution")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].scatter(
        summary["mask_energy_radius_90_stimulus_px"], summary["gradient_energy_radius_90_px"],
        s=20, alpha=0.7, color="#0072B2"
    )
    limits = [
        0,
        1.05 * float(
            max(summary["mask_energy_radius_90_stimulus_px"].max(), summary["gradient_energy_radius_90_px"].max())
        ),
    ]
    axes[1, 1].plot(limits, limits, color="black", ls="--", lw=1)
    axes[1, 1].set(
        xlim=limits,
        ylim=limits,
        xlabel="readout-only radius90 (px)",
        ylabel="composite radius90 (px)",
        title="core contribution by unit",
    )

    for frame_index, color in zip(frames, ("#0072B2", "#D55E00", "#009E73", "#CC79A7"), strict=False):
        values = measurements.loc[
            measurements.scored_frame_index.eq(frame_index), "gradient_energy_radius_90_px"
        ]
        axes[1, 2].hist(values, bins=16, histtype="step", lw=1.8, color=color, label=f"frame {frame_index}")
    axes[1, 2].set(xlabel="composite radius90 (px)", ylabel="units", title="stimulus/time dependence")
    axes[1, 2].legend(frameon=False, fontsize=8)

    axes[1, 3].scatter(
        summary["gradient_energy_radius_90_px"],
        summary["lag_energy_mean_frames_ago"],
        c=summary["gradient_energy_axis_ratio"],
        cmap="viridis",
        s=24,
        alpha=0.8,
    )
    axes[1, 3].set(
        xlabel="composite spatial radius90 (px)",
        ylabel="mean lag (frames ago)",
        title="spatial and temporal support",
    )
    for ax in axes[1]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.suptitle(
        "Checkpoint 48: composite input-space effective RF scales\n"
        "Squared center-rate gradients through the complete RR100 twin; image 68, trace 561",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_example_maps(
    examples: pd.DataFrame,
    readout: pd.DataFrame,
    measurements: pd.DataFrame,
    lag_profiles: pd.DataFrame,
    spatial_maps: np.ndarray,
    rr_indices: np.ndarray,
    frames: list[int],
    *,
    crop_size: int,
    path: Path,
) -> None:
    n_rows = len(examples)
    fig, axes = plt.subplots(n_rows, 2 + len(frames), figsize=(3.15 * (2 + len(frames)), 3.0 * n_rows), constrained_layout=True)
    rr_lookup = {int(rr): index for index, rr in enumerate(rr_indices)}
    for row_index, selection in examples.iterrows():
        rr = int(selection.rr100_index)
        readout_row = readout.loc[readout.rr100_index.eq(rr)].iloc[0]
        canonical = int(readout_row.canonical_channel)
        # The CSV stores geometry, not mask pixels. Display a Gaussian ellipse
        # reconstructed from its measured covariance only as a labelled readout
        # geometry schematic; the composite panels are observed gradients.
        size = int(crop_size)
        yy, xx = np.mgrid[:size, :size]
        cy = cx = (size - 1) / 2.0
        theta = np.radians(float(readout_row.mask_principal_angle_deg))
        dy, dx = yy - cy, xx - cx
        major = dx * np.cos(theta) + dy * np.sin(theta)
        minor = -dx * np.sin(theta) + dy * np.cos(theta)
        sigma_major = max(float(readout_row.mask_sigma_major_stimulus_px), 1e-3)
        sigma_minor = max(float(readout_row.mask_sigma_minor_stimulus_px), 1e-3)
        schematic = np.exp(-0.5 * ((major / sigma_major) ** 2 + (minor / sigma_minor) ** 2))
        axes[row_index, 0].imshow(schematic, cmap="magma", origin="lower", norm=PowerNorm(0.45))
        axes[row_index, 0].set_title(
            f"RR{rr:02d} / ch{canonical}\nreadout-only schematic; r90={readout_row.mask_energy_radius_90_stimulus_px:.1f}px",
            fontsize=8,
        )
        axes[row_index, 0].set_xticks([])
        axes[row_index, 0].set_yticks([])

        for column, frame_index in enumerate(frames, start=1):
            frame_ordinal = frames.index(frame_index)
            values = spatial_maps[frame_ordinal, rr_lookup[rr]].astype(np.float64)
            values /= max(float(values.sum()), EPS)
            cropped, y0, x0 = crop_center(values, int(crop_size))
            record = measurements.loc[
                measurements.rr100_index.eq(rr) & measurements.scored_frame_index.eq(frame_index)
            ].iloc[0]
            ax = axes[row_index, column]
            ax.imshow(cropped, cmap="magma", origin="lower", norm=PowerNorm(0.35, vmin=0, vmax=float(cropped.max())))
            ax.add_patch(
                Circle(
                    (
                        float(record.gradient_energy_center_x_px) - x0,
                        float(record.gradient_energy_center_y_px) - y0,
                    ),
                    float(record.gradient_energy_radius_90_px),
                    fill=False,
                    ec="cyan",
                    lw=1.0,
                )
            )
            ax.set_title(
                f"frame {frame_index}; composite energy\nr90={record.gradient_energy_radius_90_px:.1f}px; "
                f"crop={100*record.central_crop_gradient_energy_fraction:.1f}%",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        ax = axes[row_index, -1]
        for frame_index, color in zip(frames, ("#0072B2", "#D55E00", "#009E73", "#CC79A7"), strict=False):
            profile = lag_profiles.loc[
                lag_profiles.rr100_index.eq(rr) & lag_profiles.scored_frame_index.eq(frame_index)
            ].sort_values("lag_frames_ago")
            ax.plot(profile.lag_frames_ago, profile.lag_energy_fraction, color=color, lw=1.5, label=f"frame {frame_index}")
        ax.set(xlabel="lag (frames ago)", ylabel="gradient-energy fraction", title="temporal support")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
        if row_index == 0:
            ax.legend(frameon=False, fontsize=7)
        axes[row_index, 0].set_ylabel(str(selection.selection_role), fontsize=8)
    fig.suptitle(
        "Predeclared readout-size quantiles: observed composite RF maps and lag profiles\n"
        "Cyan circles enclose 90% of squared input-gradient energy; each map is normalized independently",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frames = parse_frames(args.frames)
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {out_dir}")
    out_dir.mkdir(parents=True)
    torch.set_num_threads(max(1, int(args.threads)))

    source_path = args.source.resolve()
    with np.load(source_path, allow_pickle=False) as data:
        movie72 = np.asarray(data["movie_intact_full72"], dtype=np.float32)
        history_trace = np.asarray(data["history_xy_deg"], dtype=np.float32)
        score_trace = np.asarray(data["score_xy_deg"], dtype=np.float32)
        image_index = int(data["image_index"].item())
        trace_index = int(data["trace_index"].item())
    if movie72.shape != (N_HISTORY + N_SCORE, 151, 151):
        raise ValueError(f"Expected 72 x 151 x 151 source movie, got {movie72.shape}")
    stim = (scored_lag_stack(movie72) - 127.0) / 255.0

    readout = pd.read_csv(args.readout_audit.resolve())
    readout = readout.loc[readout.is_rr100.astype(bool)].sort_values("rr100_index").reset_index(drop=True)
    if len(readout) != 100 or not np.array_equal(readout.rr100_index.to_numpy(int), np.arange(100)):
        raise ValueError("Readout audit does not contain exactly ordered RR100 units")
    examples = select_readout_quantile_examples(readout)

    view = load_population_view(version_name=str(args.rr100_version))
    if view.membership is None or view.membership.shape != (100, 756):
        raise ValueError(f"Unexpected RR100 membership shape: {None if view.membership is None else view.membership.shape}")
    rr_channels = np.argmax(view.membership, axis=1).astype(int)
    if not np.array_equal(rr_channels, readout.canonical_channel.to_numpy(int)):
        raise ValueError("RR100 mapping differs between population view and readout audit")
    n_units = 100 if int(args.max_units) <= 0 else min(100, int(args.max_units))
    rr_indices = np.arange(n_units, dtype=int)
    examples = examples.loc[examples.rr100_index.isin(rr_indices)].copy()
    if examples.empty:
        examples = select_readout_quantile_examples(readout.iloc[:n_units].copy())
    examples.to_csv(out_dir / "selected_units.csv", index=False)

    bundle = load_canonical_twin_bundle(device=str(args.device), mode="standard")
    measurements, lag_profiles, spatial_maps, selected_gradients = measure_gradients(
        bundle,
        stim,
        frames,
        rr_channels,
        rr_indices,
        ppd=float(args.ppd),
        crop_size=int(args.crop_size),
        selected_rr=set(examples.rr100_index.astype(int)),
    )
    summary = summarize_units(measurements, readout)
    measurements = measurements.merge(
        readout[
            [
                "rr100_index",
                "mask_energy_radius_90_stimulus_px",
                "mask_sigma_major_stimulus_px",
                "mask_sigma_minor_stimulus_px",
                "mask_axis_ratio",
            ]
        ],
        on="rr100_index",
        validate="many_to_one",
    )
    quantiles = quantile_table(summary)

    measurements.to_csv(out_dir / "composite_effective_rf_measurements.csv", index=False)
    lag_profiles.to_csv(out_dir / "composite_effective_rf_lag_profiles.csv", index=False)
    summary.to_csv(out_dir / "composite_effective_rf_unit_summary.csv", index=False)
    quantiles.to_csv(out_dir / "composite_effective_rf_quantiles.csv", index=False)
    np.savez_compressed(
        out_dir / "composite_effective_rf_spatial_energy_maps.npz",
        spatial_gradient_energy=spatial_maps,
        scored_frame_indices=np.asarray(frames, dtype=np.int64),
        rr100_indices=rr_indices.astype(np.int64),
        canonical_channels=rr_channels[rr_indices].astype(np.int64),
    )
    np.savez_compressed(out_dir / "selected_lag_resolved_signed_gradients.npz", **selected_gradients)

    plot_input_and_distributions(
        movie72,
        score_trace,
        frames,
        summary,
        measurements,
        path=out_dir / "composite_effective_rf_input_and_distributions.png",
    )
    plot_example_maps(
        examples,
        readout,
        measurements,
        lag_profiles,
        spatial_maps,
        rr_indices,
        frames,
        crop_size=int(args.crop_size),
        path=out_dir / "composite_effective_rf_example_maps.png",
    )

    radius = summary["gradient_energy_radius_90_px"].to_numpy(np.float64)
    readout_radius = summary["mask_energy_radius_90_stimulus_px"].to_numpy(np.float64)
    frame_range = summary["composite_radius90_frame_range_px"].to_numpy(np.float64)
    crop_fraction = measurements["central_crop_gradient_energy_fraction"].to_numpy(np.float64)
    edge_fraction = measurements["gradient_energy_outer_5px_fraction"].to_numpy(np.float64)
    correlation = float(np.corrcoef(readout_radius, radius)[0, 1]) if len(radius) > 1 else float("nan")
    manifest = {
        "analysis": "rr100_composite_input_space_effective_rf_pooling_scale_checkpoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "input_design_human_checkpoint_no_surrogate_no_surrogate_neural_scoring",
        "contract": (
            "center-map RR100 rate gradients are backpropagated through the complete standard twin "
            "to genuine normalized 32-lag 151x151 inputs; squared gradients define spatial and lag energy"
        ),
        "image_index": image_index,
        "trace_index": trace_index,
        "scored_frame_indices": frames,
        "n_units": int(len(rr_indices)),
        "n_measurements": int(len(measurements)),
        "device": str(args.device),
        "torch_threads": int(args.threads),
        "stimulus_ppd": float(args.ppd),
        "input_shape_per_measurement": [1, 1, N_HISTORY, 151, 151],
        "model_output_spatial_shape": [51, 51],
        "composite_energy_radius90_px_quantiles": {
            str(q): float(np.quantile(radius, q)) for q in EXAMPLE_QUANTILES
        },
        "readout_energy_radius90_px_quantiles": {
            str(q): float(np.quantile(readout_radius, q)) for q in EXAMPLE_QUANTILES
        },
        "composite_to_readout_radius90_ratio_median": float(
            np.median(radius / np.maximum(readout_radius, EPS))
        ),
        "readout_vs_composite_radius90_pearson_r": correlation,
        "median_across_frame_radius90_range_px": float(np.median(frame_range)),
        "minimum_central_crop_energy_fraction": float(np.min(crop_fraction)),
        "maximum_outer_5px_energy_fraction": float(np.max(edge_fraction)),
        "recommended_initial_pooling_parameterization": (
            "use paired composite gradient-energy major/minor sigmas or measured normalized energy kernels "
            "sampled from per-unit median rows; retain the measured lag-energy distribution separately"
        ),
        "scope_limit": (
            "This first checkpoint uses one median-structure image and one trace at three scored frames. "
            "It measures local linear sensitivity at those operating points, not a stimulus-invariant anatomical RF."
        ),
        "inputs": {
            "source_movie": file_identity(source_path),
            "readout_scale_audit": file_identity(args.readout_audit.resolve()),
        },
        "outputs": {
            name: file_identity(out_dir / name)
            for name in (
                "selected_units.csv",
                "composite_effective_rf_measurements.csv",
                "composite_effective_rf_lag_profiles.csv",
                "composite_effective_rf_unit_summary.csv",
                "composite_effective_rf_quantiles.csv",
                "composite_effective_rf_spatial_energy_maps.npz",
                "selected_lag_resolved_signed_gradients.npz",
                "composite_effective_rf_input_and_distributions.png",
                "composite_effective_rf_example_maps.png",
            )
        },
        "next_checkpoint_if_approved": (
            "build one source-image optimizer matching RF-local scored and history power; then save input-only eye-check movies"
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(json_ready(manifest), indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Checkpoint 48: composite input-space effective RF scales\n\n"
        "This is Stage 1 of `RF_LOCAL_PHASE_SURROGATE_ANALYSIS_PLAN.md`. It backpropagates center-map "
        "RR100 rates through the complete digital twin to genuine 32-lag production inputs for image 68, "
        "trace 561, at three scored frames. Squared gradients define the reported spatial and temporal "
        "energy distributions. The selected example units were chosen before gradient inspection from "
        "readout-only size quantiles. No phase surrogate and no surrogate neural response was computed.\n\n"
        "Human checkpoint: inspect the example maps, edge/crop containment, frame dependence, and the "
        "composite-versus-readout scale comparison before accepting a pooling-kernel distribution.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(manifest), indent=2))


if __name__ == "__main__":
    main()
