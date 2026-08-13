#!/usr/bin/env python3
"""Kuang-style checkpoint 1: exact 51x51 retinal input power redistribution.

This targeted, model-free render compares one real natural-image/eye-trace pair
to a true zero-gaze counterfactual.  It stops before weighting the stimulus by
RR100 unit tuning or evaluating twin responses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import (
    DEFAULT_RUN_DIR,
    one_trace_from_source,
    source_row_by_id,
)
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import (
    DEFAULT_SOURCE_CSV,
    load_source_rows,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    _load_twin_common,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import (
    _extract_patch,
)
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _standardize_uint_like,
    _trace_xy_to_twin_helper_order,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "outputs/fig4_active_sensing/rr100_kuang_input_power_checkpoint_01_v1"
FRAME_RATE_HZ = 120.0
FRAME_SIZE_PX = 51
N_LAGS = 32
SF_FIT_MIN_CPD = 1.0
SF_FIT_MAX_CPD = 11.313708498984761
TF_FIT_MIN_HZ = 0.5
TF_FIT_MAX_HZ = 32.0
EPS = 1e-20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--image-position", type=int, default=-1, help="Negative selects by the saved explicit image-content score.")
    parser.add_argument("--n-timepoints", type=int, default=128)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def select_image(image_table: pd.DataFrame, position: int) -> tuple[pd.Series, pd.DataFrame]:
    ranked = image_table.copy()
    contrast = pd.to_numeric(ranked["image_patch_rms_contrast"], errors="coerce").to_numpy(float)
    gradient = pd.to_numeric(ranked["image_gradient_energy"], errors="coerce").to_numpy(float)
    coherence = pd.to_numeric(ranked["image_orientation_coherence"], errors="coerce").to_numpy(float)
    ranked["selection_score"] = contrast * np.sqrt(np.maximum(gradient, 0.0)) * (0.5 + np.maximum(coherence, 0.0))
    valid = np.isfinite(ranked["selection_score"].to_numpy(float))
    if "image_feature_ok" in ranked:
        valid &= ranked["image_feature_ok"].astype(bool).to_numpy()
    if "image_patch_fraction_inside_image" in ranked:
        valid &= pd.to_numeric(ranked["image_patch_fraction_inside_image"], errors="coerce").to_numpy(float) >= 0.99
    ranked["selection_eligible"] = valid
    if int(position) >= 0:
        match = ranked[pd.to_numeric(ranked["image_index"], errors="coerce").astype("Int64") == int(position)]
        if match.shape[0] != 1:
            raise ValueError(f"Expected one image_index={int(position)}, found {match.shape[0]}")
        chosen = match.iloc[0]
        criterion = "user_requested_image_position"
    else:
        eligible = ranked[ranked["selection_eligible"]].copy()
        if eligible.empty:
            raise ValueError("No eligible image is available")
        chosen = eligible.loc[eligible["selection_score"].idxmax()]
        criterion = "maximum_rms_contrast_x_sqrt_gradient_x_0p5_plus_orientation_coherence"
    ranked["selected"] = ranked["image_index"].astype(int).eq(int(chosen["image_index"]))
    ranked["selection_criterion"] = criterion
    return chosen, ranked


def render_retinal_movie(patch: np.ndarray, trace_xy: np.ndarray, *, ppd: float) -> np.ndarray:
    import torch

    common = _load_twin_common()
    image = _standardize_uint_like(patch)
    trace = np.asarray(trace_xy, dtype=np.float32)
    full_stack = np.broadcast_to(
        image[None, :, :],
        (trace.shape[0] + int(N_LAGS) + 1, *image.shape),
    ).copy()
    eye = torch.from_numpy(_trace_xy_to_twin_helper_order(trace))
    stim = common.make_counterfactual_stim(
        full_stack,
        eye,
        ppd=float(ppd),
        scale_factor=1.0,
        n_lags=int(N_LAGS),
        out_size=(int(FRAME_SIZE_PX), int(FRAME_SIZE_PX)),
    )
    lag_zero = stim.detach().cpu().numpy()[:, 0, 0]
    if lag_zero.shape[0] >= trace.shape[0] + 1:
        lag_zero = lag_zero[1 : trace.shape[0] + 1]
    else:
        lag_zero = lag_zero[: trace.shape[0]]
    if lag_zero.shape != (trace.shape[0], FRAME_SIZE_PX, FRAME_SIZE_PX):
        raise ValueError(f"Unexpected retinal movie shape {lag_zero.shape}")
    return lag_zero.astype(np.float32, copy=False)


def spectral_decomposition(
    movie: np.ndarray, *, ppd: float, frame_rate_hz: float, temporal_window: str = "hann"
) -> dict[str, np.ndarray]:
    arr = np.asarray(movie, dtype=np.float64)
    temporal_mean = np.mean(arr, axis=0)
    residual = arr - temporal_mean[None, :, :]
    spatial_window = np.outer(np.hanning(arr.shape[1]), np.hanning(arr.shape[2]))

    dc_image = (temporal_mean - float(np.mean(temporal_mean))) * spatial_window
    dc_fft = np.fft.fftshift(np.fft.fft2(dc_image))
    dc_power = np.abs(dc_fft) ** 2

    if temporal_window == "hann":
        time_weights = np.hanning(arr.shape[0])
    elif temporal_window == "rectangular":
        time_weights = np.ones(arr.shape[0], dtype=float)
    else:
        raise ValueError(f"Unknown temporal window {temporal_window!r}")
    weighted = residual * time_weights[:, None, None] * spatial_window[None, :, :]
    spatial_fft = np.fft.fftshift(np.fft.fft2(weighted, axes=(1, 2)), axes=(1, 2))
    full_fft = np.fft.fft(spatial_fft, axis=0)
    temporal_frequency = np.fft.fftfreq(arr.shape[0], d=1.0 / float(frame_rate_hz))
    keep = temporal_frequency >= 0.0
    temporal_frequency = temporal_frequency[keep]
    dynamic_power = np.abs(full_fft[keep]) ** 2

    spatial_y = np.fft.fftshift(np.fft.fftfreq(arr.shape[1], d=1.0 / float(ppd)))
    spatial_x = np.fft.fftshift(np.fft.fftfreq(arr.shape[2], d=1.0 / float(ppd)))
    radial_sf = np.sqrt(spatial_x[None, :] ** 2 + spatial_y[:, None] ** 2)
    return {
        "temporal_mean_image": temporal_mean,
        "temporal_residual_movie": residual,
        "dc_power_2d": dc_power,
        "dynamic_power_tf_y_x": dynamic_power,
        "temporal_frequency_hz": temporal_frequency,
        "radial_sf_cpd": radial_sf,
    }


def radialize_power(decomp: dict[str, np.ndarray], *, ppd: float, frame_size: int) -> dict[str, np.ndarray]:
    fundamental = float(ppd) / float(frame_size)
    nyquist = float(ppd) / 2.0
    sf_edges = np.geomspace(fundamental, nyquist, 14)
    sf_centers = np.sqrt(sf_edges[:-1] * sf_edges[1:])
    radial = np.asarray(decomp["radial_sf_cpd"], dtype=float)
    dc = np.asarray(decomp["dc_power_2d"], dtype=float)
    dynamic = np.asarray(decomp["dynamic_power_tf_y_x"], dtype=float)
    dc_radial = np.full(sf_centers.shape, np.nan)
    dynamic_radial = np.full((sf_centers.size, dynamic.shape[0]), np.nan)
    mode_counts = np.zeros(sf_centers.shape, dtype=int)
    for idx, (lo, hi) in enumerate(zip(sf_edges[:-1], sf_edges[1:], strict=True)):
        mask = (radial >= lo) & (radial < hi)
        mode_counts[idx] = int(np.sum(mask))
        if np.any(mask):
            dc_radial[idx] = float(np.mean(dc[mask]))
            dynamic_radial[idx] = np.mean(dynamic[:, mask], axis=1)
    return {
        "sf_edges_cpd": sf_edges,
        "sf_centers_cpd": sf_centers,
        "spatial_mode_count": mode_counts,
        "dc_radial_power": dc_radial,
        "dynamic_radial_power": dynamic_radial,
    }


def support_summary(decomp: dict[str, np.ndarray]) -> dict[str, float]:
    power = np.asarray(decomp["dynamic_power_tf_y_x"], dtype=float)
    tf = np.asarray(decomp["temporal_frequency_hz"], dtype=float)
    sf = np.asarray(decomp["radial_sf_cpd"], dtype=float)
    positive = tf > 0.0
    total = float(np.sum(power[positive]))
    sf_ok = (sf >= SF_FIT_MIN_CPD) & (sf <= SF_FIT_MAX_CPD)
    tf_ok = positive & (tf >= TF_FIT_MIN_HZ) & (tf <= TF_FIT_MAX_HZ)
    joint = float(np.sum(power[tf_ok][:, sf_ok]))
    spatial = float(np.sum(power[positive][:, sf_ok]))
    temporal = float(np.sum(power[tf_ok]))
    high_tf = float(np.sum(power[(tf > TF_FIT_MAX_HZ)]))
    high_sf = float(np.sum(power[positive][:, sf > SF_FIT_MAX_CPD]))
    return {
        "total_positive_tf_dynamic_power": total,
        "fraction_dynamic_power_in_joint_fitted_support": joint / max(total, EPS),
        "fraction_dynamic_power_in_sf_fitted_support": spatial / max(total, EPS),
        "fraction_dynamic_power_in_tf_fitted_support": temporal / max(total, EPS),
        "fraction_dynamic_power_above_tf_fit_max": high_tf / max(total, EPS),
        "fraction_dynamic_power_above_sf_fit_max": high_sf / max(total, EPS),
    }


def montage(movie: np.ndarray, indices: np.ndarray) -> np.ndarray:
    tiles = [np.asarray(movie[int(idx)]) for idx in indices]
    separator = np.full((movie.shape[1], 2), np.nan)
    pieces: list[np.ndarray] = []
    for idx, tile in enumerate(tiles):
        if idx:
            pieces.append(separator)
        pieces.append(tile)
    return np.concatenate(pieces, axis=1)


def plot_checkpoint(
    *,
    out_dir: Path,
    patch: np.ndarray,
    patch_meta: dict[str, Any],
    image_row: pd.Series,
    trace: np.ndarray,
    static_movie: np.ndarray,
    fem_movie: np.ndarray,
    static_decomp: dict[str, np.ndarray],
    fem_decomp: dict[str, np.ndarray],
    static_radial: dict[str, np.ndarray],
    fem_radial: dict[str, np.ndarray],
    support: dict[str, float],
    dpi: int,
) -> tuple[Path, Path]:
    ppd = float(patch_meta["patch_ppd"])
    centered_trace = np.asarray(trace, dtype=float)
    speed = np.zeros(trace.shape[0], dtype=float)
    speed[1:] = np.linalg.norm(np.diff(centered_trace, axis=0), axis=1) * FRAME_RATE_HZ
    time_ms = np.arange(trace.shape[0]) * 1000.0 / FRAME_RATE_HZ
    frame_indices = np.asarray([0, trace.shape[0] // 3, 2 * trace.shape[0] // 3, trace.shape[0] - 1], dtype=int)

    fig = plt.figure(figsize=(17.2, 11.2), constrained_layout=False)
    gs = fig.add_gridspec(2, 4, left=0.055, right=0.975, top=0.90, bottom=0.075, hspace=0.34, wspace=0.34)
    axes = [fig.add_subplot(gs[row, col]) for row in range(2) for col in range(4)]

    patch_arr = np.asarray(patch, dtype=float)
    lo, hi = np.nanpercentile(patch_arr, [1, 99])
    axes[0].imshow(patch_arr, cmap="gray", vmin=lo, vmax=hi)
    xy_px = np.column_stack([
        patch_arr.shape[1] / 2.0 + centered_trace[:, 0] * ppd,
        patch_arr.shape[0] / 2.0 - centered_trace[:, 1] * ppd,
    ])
    axes[0].plot(xy_px[:, 0], xy_px[:, 1], color="#D55E00", lw=1.4)
    axes[0].scatter(xy_px[frame_indices, 0], xy_px[frame_indices, 1], c=np.arange(4), cmap="viridis", s=35, edgecolor="white", lw=0.5)
    axes[0].set_title(f"A  Natural-image patch + real FEM\nsource row {int(image_row['source_row'])}", loc="left", fontweight="bold")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    all_frames = np.concatenate([static_movie[frame_indices], fem_movie[frame_indices]], axis=0)
    fvmin, fvmax = np.nanpercentile(all_frames, [1, 99])
    axes[1].imshow(montage(static_movie, frame_indices), cmap="gray", vmin=fvmin, vmax=fvmax)
    axes[1].set_title("B  Zero gaze: identical 51×51 frames", loc="left", fontweight="bold")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[2].imshow(montage(fem_movie, frame_indices), cmap="gray", vmin=fvmin, vmax=fvmax)
    axes[2].set_title("C  Real FEM: translated retinal frames", loc="left", fontweight="bold")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    axes[3].plot(time_ms, speed, color="#0072B2", lw=1.5)
    axes[3].scatter(time_ms[frame_indices], speed[frame_indices], c=np.arange(4), cmap="viridis", s=32, zorder=3)
    axes[3].set_title("D  Retinal-image speed", loc="left", fontweight="bold")
    axes[3].set(xlabel="time (ms)", ylabel="speed (deg/s)")
    axes[3].grid(color="0.9")

    sf = static_radial["sf_centers_cpd"]
    static_dc = static_radial["dc_radial_power"]
    fem_dc = fem_radial["dc_radial_power"]
    norm = max(float(np.nanmax(static_dc)), EPS)
    axes[4].plot(sf, static_dc / norm, "o-", color="black", label="zero gaze")
    axes[4].plot(sf, fem_dc / norm, "o-", color="#D55E00", label="FEM temporal mean")
    axes[4].axvspan(SF_FIT_MIN_CPD, SF_FIT_MAX_CPD, color="#009E73", alpha=0.09, label="SF fit support")
    axes[4].set_xscale("log", base=2)
    axes[4].set_yscale("log")
    axes[4].set_title("E  Temporal-DC spatial power", loc="left", fontweight="bold")
    axes[4].set(xlabel="spatial frequency (cpd)", ylabel="power / zero-gaze maximum")
    axes[4].grid(color="0.9", which="both"); axes[4].legend(frameon=False, fontsize=8)

    tf = static_decomp["temporal_frequency_hz"]
    positive = tf > 0
    static_dyn = static_radial["dynamic_radial_power"][:, positive]
    fem_dyn = fem_radial["dynamic_radial_power"][:, positive]
    shared_max = max(float(np.nanmax(fem_dyn)), EPS)
    static_db = np.clip(10.0 * np.log10(np.maximum(static_dyn, EPS) / shared_max), -60, 0)
    fem_db = np.clip(10.0 * np.log10(np.maximum(fem_dyn, EPS) / shared_max), -60, 0)
    extent = [math.log2(sf[0]), math.log2(sf[-1]), tf[positive][0], tf[positive][-1]]
    cmap = "magma"
    im_static = axes[5].imshow(static_db.T, origin="lower", aspect="auto", extent=extent, vmin=-40, vmax=0, cmap=cmap)
    axes[5].set_title("F  Zero-gaze dynamic power", loc="left", fontweight="bold")
    axes[5].set(xlabel="spatial frequency (cpd)", ylabel="temporal frequency (Hz)")
    axes[5].set_xticks(np.log2([1, 2, 4, 8, 16]), ["1", "2", "4", "8", "16"])

    axes[6].imshow(fem_db.T, origin="lower", aspect="auto", extent=extent, vmin=-40, vmax=0, cmap=cmap)
    axes[6].set_title("G  FEM-created dynamic power", loc="left", fontweight="bold")
    axes[6].set(xlabel="spatial frequency (cpd)", ylabel="temporal frequency (Hz)")
    axes[6].set_xticks(np.log2([1, 2, 4, 8, 16]), ["1", "2", "4", "8", "16"])
    axes[6].axvline(math.log2(SF_FIT_MIN_CPD), color="white", ls="--", lw=0.8)
    axes[6].axvline(math.log2(SF_FIT_MAX_CPD), color="white", ls="--", lw=0.8)
    axes[6].axhline(TF_FIT_MAX_HZ, color="white", ls="--", lw=0.8)
    cbar = fig.colorbar(im_static, ax=[axes[5], axes[6]], fraction=0.025, pad=0.02)
    cbar.set_label("dynamic power (dB relative to FEM maximum)")

    conditional = fem_dyn / np.maximum(np.sum(fem_dyn, axis=1, keepdims=True), EPS)
    conditional_db = np.clip(10.0 * np.log10(np.maximum(conditional, EPS) / np.maximum(np.max(conditional, axis=1, keepdims=True), EPS)), -30, 0)
    axes[7].imshow(conditional_db.T, origin="lower", aspect="auto", extent=extent, vmin=-30, vmax=0, cmap="viridis")
    axes[7].set_title("H  TF redistribution within each SF", loc="left", fontweight="bold")
    axes[7].set(xlabel="spatial frequency (cpd)", ylabel="temporal frequency (Hz)")
    axes[7].set_xticks(np.log2([1, 2, 4, 8, 16]), ["1", "2", "4", "8", "16"])
    axes[7].axvline(math.log2(SF_FIT_MIN_CPD), color="white", ls="--", lw=0.8)
    axes[7].axvline(math.log2(SF_FIT_MAX_CPD), color="white", ls="--", lw=0.8)
    axes[7].axhline(TF_FIT_MAX_HZ, color="white", ls="--", lw=0.8)
    axes[7].text(
        0.02, 0.98,
        f"power inside SF×TF fit support: {support['fraction_dynamic_power_in_joint_fitted_support']:.1%}\n"
        f"above 32 Hz: {support['fraction_dynamic_power_above_tf_fit_max']:.1%}\n"
        f"above 11.31 cpd: {support['fraction_dynamic_power_above_sf_fit_max']:.1%}",
        transform=axes[7].transAxes, ha="left", va="top", fontsize=8.5, color="white",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "boxstyle": "round,pad=0.3"},
    )

    fig.suptitle(
        "RR100 Figure 4 input checkpoint: real FEM redistributes a static natural image into temporal frequencies\n"
        "Observed retinal power only — no unit tuning or neural response weighting",
        fontsize=14,
    )
    png = out_dir / "checkpoint_01_kuang_input_power_redistribution.png"
    pdf = out_dir / "checkpoint_01_kuang_input_power_redistribution.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_table_path = Path(args.run_dir) / "image_feature_table.csv"
    image_table = pd.read_csv(image_table_path)
    image_row, ranking = select_image(image_table, int(args.image_position))
    source_rows = load_source_rows(Path(args.source_csv))
    source_row = source_row_by_id(source_rows, int(image_row["source_row"]))
    patch, patch_meta = _extract_patch(source_row, canvas_cache={}, patch_size_px=int(args.patch_size_px))
    trace = one_trace_from_source(
        source_rows,
        int(image_row["source_row"]),
        n_timepoints=int(args.n_timepoints),
        bin_seconds=1.0 / FRAME_RATE_HZ,
    )
    trace = np.asarray(trace, dtype=np.float32)
    trace = trace - np.mean(trace, axis=0, keepdims=True)
    zero_trace = np.zeros_like(trace)

    # The RR100 grating sweep and this spectral audit share the canonical PPD.
    common = _load_twin_common()
    model_ppd = float(common.PPD)
    static_movie = render_retinal_movie(patch, zero_trace, ppd=model_ppd)
    fem_movie = render_retinal_movie(patch, trace, ppd=model_ppd)
    static_decomp = spectral_decomposition(static_movie, ppd=model_ppd, frame_rate_hz=FRAME_RATE_HZ)
    fem_decomp = spectral_decomposition(fem_movie, ppd=model_ppd, frame_rate_hz=FRAME_RATE_HZ)
    fem_rectangular_decomp = spectral_decomposition(
        fem_movie, ppd=model_ppd, frame_rate_hz=FRAME_RATE_HZ, temporal_window="rectangular"
    )
    static_radial = radialize_power(static_decomp, ppd=model_ppd, frame_size=FRAME_SIZE_PX)
    fem_radial = radialize_power(fem_decomp, ppd=model_ppd, frame_size=FRAME_SIZE_PX)
    support = support_summary(fem_decomp)
    rectangular_support = support_summary(fem_rectangular_decomp)
    static_frame_max_abs_difference = float(np.max(np.abs(static_movie - static_movie[:1])))
    static_dynamic_power = float(np.sum(static_decomp["dynamic_power_tf_y_x"][static_decomp["temporal_frequency_hz"] > 0]))

    png, pdf = plot_checkpoint(
        out_dir=Path(args.out_dir),
        patch=np.asarray(patch),
        patch_meta={**patch_meta, "patch_ppd": model_ppd},
        image_row=image_row,
        trace=trace,
        static_movie=static_movie,
        fem_movie=fem_movie,
        static_decomp=static_decomp,
        fem_decomp=fem_decomp,
        static_radial=static_radial,
        fem_radial=fem_radial,
        support=support,
        dpi=int(args.dpi),
    )

    ranking_columns = [
        "image_index", "source_row", "session", "selection_score", "selection_eligible", "selected",
        "selection_criterion", "image_patch_rms_contrast", "image_gradient_energy", "image_orientation_coherence",
    ]
    ranking[ranking_columns].sort_values("selection_score", ascending=False).to_csv(
        Path(args.out_dir) / "checkpoint_01_image_selection.csv", index=False
    )

    long_rows: list[dict[str, Any]] = []
    tf = static_decomp["temporal_frequency_hz"]
    for condition, radial in (("zero_gaze", static_radial), ("real_fem", fem_radial)):
        dyn = radial["dynamic_radial_power"]
        for sf_idx, sf_value in enumerate(radial["sf_centers_cpd"]):
            denom = float(np.sum(dyn[sf_idx, tf > 0]))
            for tf_idx, tf_value in enumerate(tf):
                long_rows.append({
                    "condition": condition,
                    "sf_bin_center_cpd": float(sf_value),
                    "temporal_frequency_hz": float(tf_value),
                    "dynamic_power": float(dyn[sf_idx, tf_idx]),
                    "dynamic_power_fraction_within_sf": float(dyn[sf_idx, tf_idx] / max(denom, EPS)) if tf_value > 0 else 0.0,
                    "spatial_mode_count": int(radial["spatial_mode_count"][sf_idx]),
                })
    pd.DataFrame(long_rows).to_csv(Path(args.out_dir) / "checkpoint_01_sf_tf_power_long.csv", index=False)
    pd.DataFrame([
        {
            "condition": "zero_gaze",
            "static_frame_max_abs_difference": static_frame_max_abs_difference,
            "positive_tf_dynamic_power": static_dynamic_power,
            **{key: np.nan for key in support},
        },
        {
            "condition": "real_fem",
            "static_frame_max_abs_difference": np.nan,
            "positive_tf_dynamic_power": support["total_positive_tf_dynamic_power"],
            **support,
        },
        {
            "condition": "real_fem_rectangular_window_diagnostic",
            "static_frame_max_abs_difference": np.nan,
            "positive_tf_dynamic_power": rectangular_support["total_positive_tf_dynamic_power"],
            **rectangular_support,
        },
    ]).to_csv(Path(args.out_dir) / "checkpoint_01_power_support_summary.csv", index=False)
    np.savez_compressed(
        Path(args.out_dir) / "checkpoint_01_retinal_movies_and_power.npz",
        zero_gaze_movie=static_movie,
        real_fem_movie=fem_movie,
        centered_real_trace_xy_deg=trace,
        zero_trace_xy_deg=zero_trace,
        sf_bin_centers_cpd=static_radial["sf_centers_cpd"],
        temporal_frequency_hz=tf,
        zero_gaze_dc_radial_power=static_radial["dc_radial_power"],
        real_fem_dc_radial_power=fem_radial["dc_radial_power"],
        zero_gaze_dynamic_radial_power=static_radial["dynamic_radial_power"],
        real_fem_dynamic_radial_power=fem_radial["dynamic_radial_power"],
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "rr100_kuang_style_exact_retinal_input_power_checkpoint_01",
        "status": "input_mechanism_checkpoint_complete_stop_before_unit_weighting",
        "scope": "one audibly selected natural image and its own real 128-sample FEM trace versus true zero gaze",
        "observed_vs_derived": {
            "observed": "exact 51x51 gaze-contingent retinal frames from the canonical renderer",
            "derived": "temporal-DC spatial spectrum and positive-TF SFxTF power after temporal-mean decomposition",
            "not_yet_computed": "RR100 tuning-weighted drive and direct twin response",
        },
        "baseline_contract": "true zero gaze at the mean landing position; not trial-mean stabilization",
        "spectral_contract": "temporal mean is assigned exactly to TF=0; only the residual movie receives a temporal Hann window, preventing artificial nonzero-TF power in the zero-gaze movie",
        "frame_rate_hz": FRAME_RATE_HZ,
        "n_timepoints": int(args.n_timepoints),
        "retinal_frame_size_px": FRAME_SIZE_PX,
        "model_ppd": model_ppd,
        "field_of_view_deg": FRAME_SIZE_PX / model_ppd,
        "one_cycle_spatial_frequency_cpd": model_ppd / FRAME_SIZE_PX,
        "sf_fit_support_cpd": [SF_FIT_MIN_CPD, SF_FIT_MAX_CPD],
        "tf_fit_support_hz": [TF_FIT_MIN_HZ, TF_FIT_MAX_HZ],
        "selected_image": {
            "image_index": int(image_row["image_index"]),
            "source_row": int(image_row["source_row"]),
            "session": str(image_row["session"]),
            "selection_score": float(image_row["selection_score"]),
            "selection_criterion": str(ranking.loc[ranking["selected"], "selection_criterion"].iloc[0]),
        },
        "checks": {
            "zero_gaze_max_frame_difference": static_frame_max_abs_difference,
            "zero_gaze_positive_tf_dynamic_power": static_dynamic_power,
            **support,
            "rectangular_window_fraction_dynamic_power_in_joint_fitted_support": rectangular_support[
                "fraction_dynamic_power_in_joint_fitted_support"
            ],
            "rectangular_window_fraction_dynamic_power_above_tf_fit_max": rectangular_support[
                "fraction_dynamic_power_above_tf_fit_max"
            ],
        },
        "inputs": {
            "image_feature_table": file_identity(image_table_path),
            "source_windows": file_identity(Path(args.source_csv)),
        },
        "artifacts": {
            "figure_png": png.name,
            "figure_pdf": pdf.name,
            "image_selection": "checkpoint_01_image_selection.csv",
            "sf_tf_power_long": "checkpoint_01_sf_tf_power_long.csv",
            "power_support_summary": "checkpoint_01_power_support_summary.csv",
            "retinal_movies_and_power": "checkpoint_01_retinal_movies_and_power.npz",
        },
        "checkpoint_policy": "Stop here for human interpretation before unit-weighted maps or model inference.",
    }
    write_json(Path(args.out_dir) / "manifest.json", manifest)
    print(json.dumps(json_ready(manifest["selected_image"]), indent=2))
    print(json.dumps(json_ready(manifest["checks"]), indent=2))
    print(f"Wrote {png.resolve()}")


if __name__ == "__main__":
    main()
