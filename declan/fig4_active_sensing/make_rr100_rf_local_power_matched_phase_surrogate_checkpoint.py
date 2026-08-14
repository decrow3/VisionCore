#!/usr/bin/env python3
"""Optimize one source-image phase surrogate at the median composite RF scale.

This is Stage 2/3 of RF_LOCAL_PHASE_SURROGATE_ANALYSIS_PLAN.md.  A single
histogram-matched random-phase source image is optimized through the exact
72-frame retinal renderer.  The primary loss matches locally pooled
SF x orientation x TF power at the population-median composite RF scale.  The
other four RF-scale quantiles and an offset spatial grid are held out for audit.

The script is input-only: it writes source images, four-condition retinal
movies, power/phase/contrast diagnostics, and optimization traces.  It does not
load the digital twin or calculate neural responses/SSI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity
import torch

from declan.fig4_active_sensing.make_rr100_global_3d_phase_scramble_checkpoint import (
    histogram_audit,
    rank_histogram_match,
)
from declan.fig4_active_sensing.make_rr100_global_source_phase_scramble_checkpoint import (
    _difference,
    _font,
    _gray,
    _write_mp4,
    file_identity,
)
from declan.fig4_active_sensing.make_rr100_phase_surrogate_input_checkpoint import (
    PPD,
    movie_audit,
    power_audit,
    relative_db,
)
from declan.fig4_active_sensing.run_interim_input_spectral_cache import (
    FRAME_RATE_HZ,
    ORIENTATION_EDGES_DEG,
    SF_EDGES_CPD,
    SF_FIT_MAX_CPD,
    SF_FIT_MIN_CPD,
    TF_CORE_MAX_HZ,
)
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _load_twin_common,
    _trace_xy_to_twin_helper_order,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "outputs/fig4_active_sensing/rr100_global_source_phase_scramble_checkpoint_42_v1/"
    "data/example_2_image_068.npz"
)
ERF = ROOT / (
    "outputs/fig4_active_sensing/rr100_composite_effective_rf_pooling_scale_checkpoint_48_v1/"
    "composite_effective_rf_lag_profiles.csv"
)
OUT = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_rf_local_power_matched_phase_surrogate_checkpoint_49_v1"
)
N_HISTORY = 32
N_SCORE = 40
OUT_SIZE = 151
PATCH_SIZE = 31
MEDIAN_SIGMA_PX = 2.9679
AUDIT_SIGMAS_PX = (2.3077, 2.6080, 2.9679, 3.1809, 3.4436)
HISTORY_FRAMES = (0, 19, 39)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--erf-lag-profiles", type=Path, default=ERF)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2.0)
    parser.add_argument("--train-grid-size", type=int, default=11)
    parser.add_argument("--center-batch-size", type=int, default=25)
    parser.add_argument("--spatial-seed", type=int, default=20260814)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--median-sigma-px", type=float, default=MEDIAN_SIGMA_PX)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--ppd", type=float, default=PPD)
    parser.add_argument("--weight-local", type=float, default=1.0)
    parser.add_argument("--weight-history", type=float, default=0.5)
    parser.add_argument("--weight-global", type=float, default=0.5)
    parser.add_argument("--weight-intensity", type=float, default=0.1)
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_array(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def grid_centers(size: int, n: int, *, offset: bool) -> list[tuple[int, int]]:
    if int(n) < 1:
        raise ValueError("Grid size must be positive")
    coordinates = np.linspace(25, size - 26, int(n))
    if offset:
        coordinates = 0.5 * (coordinates[:-1] + coordinates[1:])
    coordinates = np.rint(coordinates).astype(int)
    return [(int(y), int(x)) for y in coordinates for x in coordinates]


def stochastic_center_batches(
    centers: list[tuple[int, int]],
    *,
    n_batches: int,
    batch_size: int,
    seed: int,
) -> list[list[tuple[int, int]]]:
    """Predeclare cycling shuffled minibatches over the spatial center pool."""
    if not centers:
        raise ValueError("Center pool is empty")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(centers)).tolist()
    cursor = 0
    batches: list[list[tuple[int, int]]] = []
    for _ in range(int(n_batches)):
        indices: list[int] = []
        while len(indices) < int(batch_size):
            remaining = min(int(batch_size) - len(indices), len(order) - cursor)
            indices.extend(order[cursor : cursor + remaining])
            cursor += remaining
            if cursor == len(order):
                order = rng.permutation(len(centers)).tolist()
                cursor = 0
        batches.append([centers[index] for index in indices])
    return batches


def source_roi_slices(trace72: np.ndarray, *, ppd: float, source_shape: tuple[int, int]) -> tuple[slice, slice]:
    max_shift = float(np.max(np.abs(np.asarray(trace72, dtype=np.float64)))) * float(ppd)
    half = int(math.ceil(OUT_SIZE / 2 + max_shift + 15))
    cy, cx = source_shape[0] // 2, source_shape[1] // 2
    if cy - half < 0 or cx - half < 0 or cy + half + 1 > source_shape[0] or cx + half + 1 > source_shape[1]:
        raise ValueError(f"Required differentiable source ROI does not fit: half={half}, shape={source_shape}")
    return slice(cy - half, cy + half + 1), slice(cx - half, cx + half + 1)


def render_source(
    common: Any,
    source: torch.Tensor,
    trace72: torch.Tensor,
    *,
    ppd: float,
) -> torch.Tensor:
    stack = source.unsqueeze(0).expand(int(trace72.shape[0]), -1, -1)
    eye = torch.as_tensor(
        _trace_xy_to_twin_helper_order(-trace72.detach().cpu().numpy()),
        dtype=source.dtype,
        device=source.device,
    )
    eye_norm = common._eye_deg_to_norm(torch.fliplr(eye), float(ppd), tuple(source.shape))
    return common._shift_movie_with_eye(
        stack,
        eye_norm,
        out_size=(OUT_SIZE, OUT_SIZE),
        center=(0.0, 0.0),
        scale_factor=1.0,
        mode="bilinear",
    )


def extract_patches(movie: torch.Tensor, centers: list[tuple[int, int]], patch_size: int) -> torch.Tensor:
    half = int(patch_size) // 2
    if int(patch_size) % 2 != 1:
        raise ValueError("Patch size must be odd")
    return torch.stack(
        [movie[:, y - half : y + half + 1, x - half : x + half + 1] for y, x in centers],
        dim=0,
    )


def spatial_bin_lookup(size: int, *, ppd: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    fy = np.fft.fftshift(np.fft.fftfreq(int(size), d=1.0 / float(ppd)))
    fx = np.fft.fftshift(np.fft.fftfreq(int(size), d=1.0 / float(ppd)))
    radial = np.hypot(fx[None, :], fy[:, None])
    orientation = np.mod(np.degrees(np.arctan2(fy[:, None], fx[None, :])), 180.0)
    sf_bin = np.digitize(radial.ravel(), SF_EDGES_CPD, right=False) - 1
    ori_bin = np.digitize(orientation.ravel(), ORIENTATION_EDGES_DEG, right=False) - 1
    sf_bin = np.clip(sf_bin, 0, len(SF_EDGES_CPD) - 2)
    ori_bin = np.clip(ori_bin, 0, len(ORIENTATION_EDGES_DEG) - 2)
    joint = sf_bin * (len(ORIENTATION_EDGES_DEG) - 1) + ori_bin
    fitted = (radial.ravel() >= SF_FIT_MIN_CPD) & (radial.ravel() <= SF_FIT_MAX_CPD)
    return (
        torch.as_tensor(joint, dtype=torch.long, device=device),
        torch.as_tensor(fitted, dtype=torch.bool, device=device),
    )


def gaussian_window(size: int, sigma: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    coordinate = torch.arange(int(size), dtype=dtype, device=device) - (int(size) - 1) / 2.0
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    return torch.exp(-0.5 * (xx.square() + yy.square()) / float(sigma) ** 2)


def hann_window(size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.hann_window(int(size), periodic=False, dtype=dtype, device=device)


def binned_power(
    patches: torch.Tensor,
    *,
    spatial_window: torch.Tensor,
    temporal_window: torch.Tensor,
    ppd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return B x positive-TF x SF x orientation power."""
    if patches.ndim != 4:
        raise ValueError(f"Expected B x T x H x W patches, got {tuple(patches.shape)}")
    batch, n_time, height, width = patches.shape
    if height != width or tuple(spatial_window.shape) != (height, width):
        raise ValueError("Spatial window shape mismatch")
    if tuple(temporal_window.shape) != (n_time,):
        raise ValueError("Temporal window shape mismatch")
    residual = patches - patches.mean(dim=1, keepdim=True)
    tapered = residual * temporal_window[None, :, None, None] * spatial_window[None, None]
    temporal_fft = torch.fft.rfft(tapered, dim=1)
    spectrum = torch.fft.fftshift(torch.fft.fft2(temporal_fft, dim=(-2, -1)), dim=(-2, -1))
    power = spectrum.real.square() + spectrum.imag.square()
    temporal_weights = torch.ones(power.shape[1], dtype=power.dtype, device=power.device)
    if power.shape[1] > 2:
        temporal_weights[1:-1] = 2.0
    power = power * temporal_weights[None, :, None, None]
    tf_hz = torch.fft.rfftfreq(int(n_time), d=1.0 / FRAME_RATE_HZ, device=power.device)
    positive = tf_hz > 0
    power = power[:, positive].reshape(batch, int(positive.sum()), height * width)
    positive_tf = tf_hz[positive]
    joint, _ = spatial_bin_lookup(height, ppd=float(ppd), device=power.device)
    n_sf = len(SF_EDGES_CPD) - 1
    n_ori = len(ORIENTATION_EDGES_DEG) - 1
    index = joint[None, None].expand(batch, power.shape[1], -1)
    output = torch.zeros((batch, power.shape[1], n_sf * n_ori), dtype=power.dtype, device=power.device)
    output.scatter_add_(2, index, power)
    return output.reshape(batch, power.shape[1], n_sf, n_ori), positive_tf


def selected_power(power: torch.Tensor, tf_hz: torch.Tensor) -> torch.Tensor:
    sf_centers = torch.as_tensor(
        0.5 * (SF_EDGES_CPD[:-1] + SF_EDGES_CPD[1:]),
        dtype=power.dtype,
        device=power.device,
    )
    sf = (sf_centers >= SF_FIT_MIN_CPD) & (sf_centers <= SF_FIT_MAX_CPD)
    tf = (tf_hz > 0) & (tf_hz <= TF_CORE_MAX_HZ)
    return power[:, tf][:, :, sf, :]


def power_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    amplitude_predicted = torch.sqrt(torch.clamp_min(predicted, 0.0) + EPS)
    amplitude_target = torch.sqrt(torch.clamp_min(target, 0.0) + EPS)
    floor = 0.02 * amplitude_target.amax(dim=tuple(range(1, amplitude_target.ndim)), keepdim=True)
    relative = (amplitude_predicted - amplitude_target) / (amplitude_target + floor + EPS)
    total_ratio = predicted.sum(dim=tuple(range(1, predicted.ndim))) / (
        target.sum(dim=tuple(range(1, target.ndim))) + EPS
    )
    return relative.square().mean() + 0.2 * torch.log(total_ratio + EPS).square().mean()


def tensor_power_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    a = target.detach().cpu().numpy().astype(np.float64).ravel()
    b = predicted.detach().cpu().numpy().astype(np.float64).ravel()
    denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPS)
    positive = a > 1e-8 * max(float(a.max()), EPS)
    return {
        "power_cosine": float(np.dot(a, b) / denominator),
        "power_ratio": float(b.sum() / max(float(a.sum()), EPS)),
        "power_relative_l2_error": float(np.linalg.norm(b - a) / max(float(np.linalg.norm(a)), EPS)),
        "median_absolute_log_power_ratio_supported_bins": float(
            np.median(np.abs(np.log((b[positive] + EPS) / (a[positive] + EPS)))) if np.any(positive) else np.nan
        ),
    }


def build_history_movies(movie72: torch.Tensor, frame_indices: tuple[int, ...] = HISTORY_FRAMES) -> torch.Tensor:
    return torch.stack(
        [
            torch.stack([movie72[N_HISTORY + frame - lag] for lag in range(N_HISTORY)], dim=0)
            for frame in frame_indices
        ],
        dim=0,
    )


def local_movie_power(
    movie: torch.Tensor,
    centers: list[tuple[int, int]],
    *,
    patch_size: int,
    sigma: float,
    temporal_window: torch.Tensor,
    ppd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    patches = extract_patches(movie, centers, int(patch_size))
    spatial = gaussian_window(
        int(patch_size), float(sigma), dtype=patches.dtype, device=patches.device
    )
    return binned_power(patches, spatial_window=spatial, temporal_window=temporal_window, ppd=float(ppd))


def history_local_power(
    movie72: torch.Tensor,
    centers: list[tuple[int, int]],
    *,
    patch_size: int,
    sigma: float,
    temporal_window: torch.Tensor,
    ppd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    histories = build_history_movies(movie72)
    patches = torch.cat(
        [extract_patches(histories[index], centers, int(patch_size)) for index in range(histories.shape[0])],
        dim=0,
    )
    spatial = gaussian_window(
        int(patch_size), float(sigma), dtype=patches.dtype, device=patches.device
    )
    return binned_power(patches, spatial_window=spatial, temporal_window=temporal_window, ppd=float(ppd))


def global_movie_power(movie: torch.Tensor, *, ppd: float) -> tuple[torch.Tensor, torch.Tensor]:
    spatial = torch.outer(
        hann_window(movie.shape[-2], dtype=movie.dtype, device=movie.device),
        hann_window(movie.shape[-1], dtype=movie.dtype, device=movie.device),
    )
    temporal = hann_window(movie.shape[0], dtype=movie.dtype, device=movie.device)
    return binned_power(movie[None], spatial_window=spatial, temporal_window=temporal, ppd=float(ppd))


def intensity_loss(source: torch.Tensor, target_source: torch.Tensor, movie: torch.Tensor, target_movie: torch.Tensor, roi: tuple[slice, slice]) -> torch.Tensor:
    source_roi = source[roi]
    target_roi = target_source[roi]
    terms = []
    for predicted, target in ((source_roi, target_roi), (movie, target_movie)):
        target_mean = target.mean()
        target_std = target.std(unbiased=False)
        terms.append(((predicted.mean() - target_mean) / (target_std + EPS)).square())
        terms.append(torch.log((predicted.std(unbiased=False) + EPS) / (target_std + EPS)).square())
    return torch.stack(terms).mean()


def phase_relation_audit_2d(reference: np.ndarray, control: np.ndarray) -> dict[str, float]:
    before = np.fft.fft2(np.asarray(reference, dtype=np.float64) - float(np.mean(reference)))
    after = np.fft.fft2(np.asarray(control, dtype=np.float64) - float(np.mean(control)))
    index = (slice(0, before.shape[0] // 2 + 1), slice(0, before.shape[1] // 2 + 1))
    before = before[index]
    after = after[index]
    support = (
        (np.abs(before) > 1e-6 * max(float(np.abs(before).max()), EPS))
        & (np.abs(after) > 1e-6 * max(float(np.abs(after).max()), EPS))
    )
    support[0, 0] = False
    delta = np.angle(after * np.conj(before))
    result = {
        "fourier_phase_retention_coherence": float(np.abs(np.mean(np.exp(1j * delta[support]))))
    }
    for axis, name in ((0, "vertical"), (1, "horizontal")):
        low = [slice(None), slice(None)]
        high = [slice(None), slice(None)]
        low[axis] = slice(0, -1)
        high[axis] = slice(1, None)
        low_t, high_t = tuple(low), tuple(high)
        relation_before = np.angle(before[high_t] * np.conj(before[low_t]))
        relation_after = np.angle(after[high_t] * np.conj(after[low_t]))
        valid = support[high_t] & support[low_t]
        result[f"adjacent_{name}_frequency_phase_relation_retention_coherence"] = float(
            np.abs(np.mean(np.exp(1j * (relation_after[valid] - relation_before[valid]))))
        )
    return result


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    # Copy before centering: callers commonly pass ROI/slice views whose source
    # arrays must remain unchanged for later histogram and range diagnostics.
    x = np.asarray(a, dtype=np.float64).ravel().copy()
    y = np.asarray(b, dtype=np.float64).ravel().copy()
    x -= x.mean()
    y -= y.mean()
    return float(np.dot(x, y) / max(float(np.linalg.norm(x) * np.linalg.norm(y)), EPS))


def max_shift_correlation(reference: np.ndarray, control: np.ndarray, *, max_shift: int, edge: bool) -> float:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(control, dtype=np.float64)
    if edge:
        a = np.hypot(sobel(a, axis=0), sobel(a, axis=1))
        b = np.hypot(sobel(b, axis=0), sobel(b, axis=1))
    best = -1.0
    for dy in range(-int(max_shift), int(max_shift) + 1):
        for dx in range(-int(max_shift), int(max_shift) + 1):
            y0a, y1a = max(0, dy), min(a.shape[0], a.shape[0] + dy)
            x0a, x1a = max(0, dx), min(a.shape[1], a.shape[1] + dx)
            y0b, y1b = max(0, -dy), min(b.shape[0], b.shape[0] - dy)
            x0b, x1b = max(0, -dx), min(b.shape[1], b.shape[1] - dx)
            best = max(best, correlation(a[y0a:y1a, x0a:x1a], b[y0b:y1b, x0b:x1b]))
    return float(best)


def source_audit(reference: np.ndarray, control: np.ndarray, roi: tuple[slice, slice]) -> dict[str, float]:
    a = np.asarray(reference[roi], dtype=np.float64)
    b = np.asarray(control[roi], dtype=np.float64)
    data_range = max(float(a.max() - a.min()), 1.0)
    return {
        **phase_relation_audit_2d(a, b),
        **histogram_audit(a, b),
        "pixel_correlation": correlation(a, b),
        "maximum_translation_aligned_pixel_correlation_shift12": max_shift_correlation(a, b, max_shift=12, edge=False),
        "maximum_translation_aligned_edge_correlation_shift12": max_shift_correlation(a, b, max_shift=12, edge=True),
        "structural_similarity": float(structural_similarity(a, b, data_range=data_range)),
        "minimum": float(b.min()),
        "maximum": float(b.max()),
        "mean": float(b.mean()),
        "std": float(b.std()),
    }


def optimize_source(
    common: Any,
    intact_source: np.ndarray,
    initial_source: np.ndarray,
    trace72_np: np.ndarray,
    lag_window_np: np.ndarray,
    roi: tuple[slice, slice],
    center_batches: list[list[tuple[int, int]]],
    validation_centers: list[tuple[int, int]],
    *,
    args: argparse.Namespace,
) -> tuple[np.ndarray, pd.DataFrame, dict[int, np.ndarray], dict[str, Any]]:
    device = torch.device(str(args.device))
    dtype = torch.float32
    intact = torch.as_tensor(intact_source, dtype=dtype, device=device)
    trace72 = torch.as_tensor(trace72_np, dtype=dtype, device=device)
    roi_mask = torch.zeros_like(intact)
    roi_mask[roi] = 1.0
    score_window = hann_window(N_SCORE, dtype=dtype, device=device)
    history_window = torch.as_tensor(lag_window_np, dtype=dtype, device=device)

    with torch.no_grad():
        target72 = render_source(common, intact, trace72, ppd=float(args.ppd)) / 255.0
        target_score = target72[N_HISTORY:]
        global_target_raw, global_tf = global_movie_power(target_score, ppd=float(args.ppd))
        global_target = selected_power(global_target_raw, global_tf)
        validation_target_raw, validation_tf = local_movie_power(
            target_score,
            validation_centers,
            patch_size=int(args.patch_size),
            sigma=float(args.median_sigma_px),
            temporal_window=score_window,
            ppd=float(args.ppd),
        )
        validation_target = selected_power(validation_target_raw, validation_tf)

    parameter = torch.nn.Parameter(torch.as_tensor(initial_source, dtype=dtype, device=device).clone())
    optimizer = torch.optim.Adam([parameter], lr=float(args.learning_rate))
    rows: list[dict[str, Any]] = []
    snapshots: dict[int, np.ndarray] = {}
    best_loss = float("inf")
    best_iteration = 0

    for iteration in range(int(args.iterations) + 1):
        train_centers = center_batches[iteration]
        fraction = iteration / max(int(args.iterations), 1)
        learning_rate = float(args.learning_rate) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * fraction)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        movie72 = render_source(common, parameter, trace72, ppd=float(args.ppd)) / 255.0
        score = movie72[N_HISTORY:]
        with torch.no_grad():
            local_target_raw, local_tf = local_movie_power(
                target_score,
                train_centers,
                patch_size=int(args.patch_size),
                sigma=float(args.median_sigma_px),
                temporal_window=score_window,
                ppd=float(args.ppd),
            )
            history_target_raw, history_tf = history_local_power(
                target72,
                train_centers,
                patch_size=int(args.patch_size),
                sigma=float(args.median_sigma_px),
                temporal_window=history_window,
                ppd=float(args.ppd),
            )
            local_target = selected_power(local_target_raw, local_tf)
            history_target = selected_power(history_target_raw, history_tf)
        local_raw, _ = local_movie_power(
            score,
            train_centers,
            patch_size=int(args.patch_size),
            sigma=float(args.median_sigma_px),
            temporal_window=score_window,
            ppd=float(args.ppd),
        )
        history_raw, _ = history_local_power(
            movie72,
            train_centers,
            patch_size=int(args.patch_size),
            sigma=float(args.median_sigma_px),
            temporal_window=history_window,
            ppd=float(args.ppd),
        )
        global_raw, _ = global_movie_power(score, ppd=float(args.ppd))
        local_selected = selected_power(local_raw, local_tf)
        history_selected = selected_power(history_raw, history_tf)
        global_selected = selected_power(global_raw, global_tf)
        local_value = power_loss(local_selected, local_target)
        history_value = power_loss(history_selected, history_target)
        global_value = power_loss(global_selected, global_target)
        intensity_value = intensity_loss(parameter, intact, score, target_score, roi)
        loss = (
            float(args.weight_local) * local_value
            + float(args.weight_history) * history_value
            + float(args.weight_global) * global_value
            + float(args.weight_intensity) * intensity_value
        )

        local_metrics = tensor_power_metrics(local_selected, local_target)
        history_metrics = tensor_power_metrics(history_selected, history_target)
        global_metrics = tensor_power_metrics(global_selected, global_target)
        record = {
            "iteration": int(iteration),
            "learning_rate": learning_rate,
            "total_loss": float(loss.detach().cpu()),
            "local_loss": float(local_value.detach().cpu()),
            "history_loss": float(history_value.detach().cpu()),
            "global_loss": float(global_value.detach().cpu()),
            "intensity_loss": float(intensity_value.detach().cpu()),
            **{f"local_{key}": value for key, value in local_metrics.items()},
            **{f"history_{key}": value for key, value in history_metrics.items()},
            **{f"global_{key}": value for key, value in global_metrics.items()},
            "source_min": float(parameter.detach()[roi].min().cpu()),
            "source_max": float(parameter.detach()[roi].max().cpu()),
            "source_mean": float(parameter.detach()[roi].mean().cpu()),
            "source_std": float(parameter.detach()[roi].std(unbiased=False).cpu()),
        }
        if iteration % int(args.checkpoint_every) == 0 or iteration == int(args.iterations):
            source_np = parameter.detach().cpu().numpy().astype(np.float32)
            phase = phase_relation_audit_2d(intact_source[roi], source_np[roi])
            record.update({f"source_{key}": value for key, value in phase.items()})
            with torch.no_grad():
                validation_raw, _ = local_movie_power(
                    score,
                    validation_centers,
                    patch_size=int(args.patch_size),
                    sigma=float(args.median_sigma_px),
                    temporal_window=score_window,
                    ppd=float(args.ppd),
                )
                validation_selected = selected_power(validation_raw, validation_tf)
                validation_metrics = tensor_power_metrics(validation_selected, validation_target)
            record.update({f"validation_{key}": value for key, value in validation_metrics.items()})
            snapshots[int(iteration)] = source_np.copy()
            print(
                f"iter {iteration:04d}: loss={record['total_loss']:.4f}; "
                f"local cos={record['local_power_cosine']:.4f} ratio={record['local_power_ratio']:.3f}; "
                f"history cos={record['history_power_cosine']:.4f} ratio={record['history_power_ratio']:.3f}; "
                f"global cos={record['global_power_cosine']:.4f} ratio={record['global_power_ratio']:.3f}; "
                f"valid cos={record['validation_power_cosine']:.4f} ratio={record['validation_power_ratio']:.3f}; "
                f"phase={record['source_fourier_phase_retention_coherence']:.4f}",
                flush=True,
            )
        rows.append(record)
        detached_loss = float(loss.detach().cpu())
        if detached_loss < best_loss:
            best_loss = detached_loss
            best_iteration = int(iteration)
        if iteration == int(args.iterations):
            break
        loss.backward()
        if parameter.grad is None:
            raise RuntimeError("Source parameter received no gradient")
        parameter.grad.mul_(roi_mask)
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(0.0, 255.0)
            parameter.mul_(roi_mask).add_(intact * (1.0 - roi_mask))

    final_source = parameter.detach().cpu().numpy().astype(np.float32).copy()
    targets = {
        "target72": target72.detach().cpu().numpy().astype(np.float32) * 255.0,
        "global_target": global_target.detach().cpu().numpy().astype(np.float32),
        "best_loss": best_loss,
        "best_iteration": best_iteration,
    }
    return final_source, pd.DataFrame(rows), snapshots, targets


def audit_local_scales(
    intact72: np.ndarray,
    surrogate72: np.ndarray,
    *,
    train_centers: list[tuple[int, int]],
    offset_centers: list[tuple[int, int]],
    patch_size: int,
    ppd: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    device = torch.device("cpu")
    intact = torch.as_tensor(intact72 / 255.0, dtype=torch.float32, device=device)
    surrogate = torch.as_tensor(surrogate72 / 255.0, dtype=torch.float32, device=device)
    score_window = hann_window(N_SCORE, dtype=torch.float32, device=device)
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for grid_name, centers in (("stochastic_training_pool_grid", train_centers), ("heldout_offset_grid", offset_centers)):
            for sigma in AUDIT_SIGMAS_PX:
                intact_raw, tf = local_movie_power(
                    intact[N_HISTORY:], centers, patch_size=int(patch_size), sigma=float(sigma),
                    temporal_window=score_window, ppd=float(ppd)
                )
                surrogate_raw, _ = local_movie_power(
                    surrogate[N_HISTORY:], centers, patch_size=int(patch_size), sigma=float(sigma),
                    temporal_window=score_window, ppd=float(ppd)
                )
                intact_selected = selected_power(intact_raw, tf)
                surrogate_selected = selected_power(surrogate_raw, tf)
                metrics = tensor_power_metrics(surrogate_selected, intact_selected)
                rows.append(
                    {
                        "grid": grid_name,
                        "n_centers": len(centers),
                        "sigma_px": float(sigma),
                        "is_optimized_scale": bool(np.isclose(sigma, MEDIAN_SIGMA_PX, atol=1e-3)),
                        **metrics,
                    }
                )
                token = str(sigma).replace(".", "p")
                arrays[f"{grid_name}_sigma_{token}_intact"] = intact_selected.cpu().numpy().astype(np.float32)
                arrays[f"{grid_name}_sigma_{token}_surrogate"] = surrogate_selected.cpu().numpy().astype(np.float32)
    return pd.DataFrame(rows), arrays


def plot_checkpoint(
    intact_source: np.ndarray,
    initial_source: np.ndarray,
    surrogate_source: np.ndarray,
    intact72: np.ndarray,
    surrogate72: np.ndarray,
    optimization: pd.DataFrame,
    local_audit: pd.DataFrame,
    source_rows: pd.DataFrame,
    radial_intact: np.ndarray,
    radial_surrogate: np.ndarray,
    *,
    roi: tuple[slice, slice],
    path: Path,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(17.5, 12.3), constrained_layout=True)
    source_panels = (
        (intact_source[roi], "Intact source ROI"),
        (initial_source[roi], "Random-phase initialization"),
        (surrogate_source[roi], "Optimized median-RF surrogate"),
        (surrogate_source[roi] - intact_source[roi], "Surrogate − intact source"),
    )
    difference_limit = float(np.quantile(np.abs(source_panels[-1][0]), 0.99))
    for column, (values, title) in enumerate(source_panels):
        if column == 3:
            image = axes[0, column].imshow(values, cmap="coolwarm", vmin=-difference_limit, vmax=difference_limit, origin="lower")
            fig.colorbar(image, ax=axes[0, column], fraction=0.046)
        else:
            axes[0, column].imshow(values, cmap="gray", vmin=0, vmax=255, origin="lower")
        axes[0, column].set_title(title, fontsize=10)
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])

    frame_indices = (32, 51, 71)
    axes[1, 0].imshow(np.concatenate([intact72[i] for i in frame_indices], axis=1), cmap="gray", vmin=0, vmax=255, origin="lower")
    axes[1, 0].set_title("Intact FEM frames 0, 19, 39")
    axes[1, 1].imshow(np.concatenate([surrogate72[i] for i in frame_indices], axis=1), cmap="gray", vmin=0, vmax=255, origin="lower")
    axes[1, 1].set_title("Surrogate FEM frames 0, 19, 39")
    difference = np.concatenate([surrogate72[i] - intact72[i] for i in frame_indices], axis=1)
    limit = float(np.quantile(np.abs(difference), 0.99))
    axes[1, 2].imshow(difference, cmap="coolwarm", vmin=-limit, vmax=limit, origin="lower")
    axes[1, 2].set_title(f"Retinal difference (±{limit:.1f})")
    for ax in axes[1, :3]:
        ax.set_xticks([])
        ax.set_yticks([])
    delta_db = relative_db(radial_surrogate) - relative_db(radial_intact)
    sf_centers = 0.5 * (SF_EDGES_CPD[:-1] + SF_EDGES_CPD[1:])
    tf_hz = np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)[1:]
    mesh = axes[1, 3].pcolormesh(sf_centers, tf_hz, delta_db, cmap="coolwarm", vmin=-12, vmax=12, shading="nearest")
    axes[1, 3].set(xscale="log", xlabel="SF (cpd)", ylabel="TF (Hz)", title="Canonical scored power ΔdB")
    fig.colorbar(mesh, ax=axes[1, 3], fraction=0.046)

    axes[2, 0].plot(optimization.iteration, optimization.total_loss, label="total", color="black")
    axes[2, 0].plot(optimization.iteration, optimization.local_loss, label="local", color="#0072B2")
    axes[2, 0].plot(optimization.iteration, optimization.history_loss, label="history", color="#D55E00")
    axes[2, 0].plot(optimization.iteration, optimization.global_loss, label="global", color="#009E73")
    axes[2, 0].set(xlabel="iteration", ylabel="loss", yscale="log", title="Optimization trace")
    axes[2, 0].legend(frameon=False, fontsize=8)

    for grid, style in (("stochastic_training_pool_grid", "-o"), ("heldout_offset_grid", "--s")):
        work = local_audit.loc[local_audit.grid.eq(grid)].sort_values("sigma_px")
        axes[2, 1].plot(work.sigma_px, work.power_cosine, style, label=grid.replace("_", " "))
    axes[2, 1].axvline(MEDIAN_SIGMA_PX, color="black", ls=":", lw=1)
    axes[2, 1].set(xlabel="Gaussian sigma (px)", ylabel="power cosine", ylim=(0, 1.005), title="Held-out RF scales and locations")
    axes[2, 1].legend(frameon=False, fontsize=7)

    axes[2, 2].plot(optimization.iteration, optimization.local_power_ratio, label="local", color="#0072B2")
    axes[2, 2].plot(optimization.iteration, optimization.history_power_ratio, label="history", color="#D55E00")
    axes[2, 2].plot(optimization.iteration, optimization.global_power_ratio, label="global", color="#009E73")
    axes[2, 2].axhline(1.0, color="black", ls="--", lw=1)
    axes[2, 2].set(xlabel="iteration", ylabel="surrogate / intact power", title="Power totals")
    axes[2, 2].legend(frameon=False, fontsize=8)

    checkpoint_rows = optimization.dropna(subset=["source_fourier_phase_retention_coherence"])
    axes[2, 3].plot(
        checkpoint_rows.iteration,
        checkpoint_rows.source_fourier_phase_retention_coherence,
        "-o",
        label="direct Fourier phase",
    )
    axes[2, 3].plot(
        checkpoint_rows.iteration,
        checkpoint_rows.source_adjacent_horizontal_frequency_phase_relation_retention_coherence,
        "-s",
        label="adjacent horizontal relation",
    )
    axes[2, 3].plot(
        checkpoint_rows.iteration,
        checkpoint_rows.source_adjacent_vertical_frequency_phase_relation_retention_coherence,
        "-^",
        label="adjacent vertical relation",
    )
    axes[2, 3].set(xlabel="iteration", ylabel="coherence with intact", ylim=(0, 1), title="Phase reconstruction audit")
    axes[2, 3].legend(frameon=False, fontsize=7)
    for ax in axes[2]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    final = source_rows.loc[source_rows.control.eq("optimized")].iloc[0]
    fig.suptitle(
        "Checkpoint 49: one-scale RF-local power-matched phase surrogate\n"
        f"Median sigma={MEDIAN_SIGMA_PX:.2f}px constrained; phase coherence={final.fourier_phase_retention_coherence:.3f}; "
        f"edge correlation={final.maximum_translation_aligned_edge_correlation_shift12:.3f}; no twin scoring",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(path, dpi=185)
    plt.close(fig)


def video_frames(
    intact_fem: np.ndarray,
    surrogate_fem: np.ndarray,
    intact_stable: np.ndarray,
    surrogate_stable: np.ndarray,
    *,
    title: str,
) -> Iterable[Image.Image]:
    scale = 2
    panel = OUT_SIZE * scale
    margin, gap = 14, 16
    width = 2 * margin + 2 * panel + gap
    height = 104 + 2 * panel + gap
    title_font = _font(20)
    label_font = _font(15)
    arrays = (intact_fem, surrogate_fem, intact_stable, surrogate_stable)
    labels = ("Intact + FEM", "Surrogate + same FEM", "Intact stabilized", "Surrogate stabilized")
    positions = (
        (margin, 82),
        (margin + panel + gap, 82),
        (margin, 82 + panel + gap),
        (margin + panel + gap, 82 + panel + gap),
    )
    for frame_index in range(N_HISTORY + N_SCORE):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        phase = "history" if frame_index < N_HISTORY else "scored"
        draw.text((margin, 5), title, fill="black", font=title_font)
        draw.text((margin, 36), f"frame {frame_index:02d}/71 ({phase})", fill="black", font=label_font)
        for array, label, (x, y) in zip(arrays, labels, positions, strict=True):
            canvas.paste(_gray(array[frame_index], scale), (x, y))
            draw.text((x, y - 22), label, fill="black", font=label_font)
        yield canvas


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {out_dir}")
    (out_dir / "movies").mkdir(parents=True)
    (out_dir / "data").mkdir()
    torch.set_num_threads(max(1, int(args.threads)))

    with np.load(args.source.resolve(), allow_pickle=False) as data:
        intact_source = np.asarray(data["source_intact"], dtype=np.float32)
        raw_phase_source = np.asarray(data["source_global_phase_scrambled"], dtype=np.float32)
        history_trace = np.asarray(data["history_xy_deg"], dtype=np.float32)
        score_trace = np.asarray(data["score_xy_deg"], dtype=np.float32)
        image_index = int(data["image_index"].item())
        trace_index = int(data["trace_index"].item())
        scramble_seed = int(data["scramble_seed"].item())
    trace72 = np.concatenate([history_trace, score_trace], axis=0)
    roi = source_roi_slices(trace72, ppd=float(args.ppd), source_shape=tuple(intact_source.shape))
    initial_source = intact_source.copy()
    initial_source[roi] = rank_histogram_match(intact_source[roi], raw_phase_source[roi])

    lag_profiles = pd.read_csv(args.erf_lag_profiles.resolve())
    lag_energy = lag_profiles.groupby("lag_frames_ago").lag_energy_fraction.mean().sort_index().to_numpy(np.float64)
    if lag_energy.shape != (N_HISTORY,):
        raise ValueError(f"Expected a 32-lag ERF profile, got {lag_energy.shape}")
    lag_window = np.sqrt(np.maximum(lag_energy, 0.0))
    lag_window /= max(float(lag_window.max()), EPS)
    train_centers = grid_centers(OUT_SIZE, int(args.train_grid_size), offset=False)
    offset_centers = grid_centers(OUT_SIZE, int(args.train_grid_size), offset=True)
    offset_coordinates = sorted({value for center in offset_centers for value in center})
    validation_coordinates = [
        offset_coordinates[index]
        for index in np.rint(np.linspace(0, len(offset_coordinates) - 1, min(5, len(offset_coordinates)))).astype(int)
    ]
    validation_centers = [(y, x) for y in validation_coordinates for x in validation_coordinates]
    center_batches = stochastic_center_batches(
        train_centers,
        n_batches=int(args.iterations) + 1,
        batch_size=int(args.center_batch_size),
        seed=int(args.spatial_seed),
    )
    common = _load_twin_common()

    surrogate_source, optimization, snapshots, target_payload = optimize_source(
        common,
        intact_source,
        initial_source,
        trace72,
        lag_window,
        roi,
        center_batches,
        validation_centers,
        args=args,
    )
    device = torch.device(str(args.device))
    with torch.no_grad():
        trace_tensor = torch.as_tensor(trace72, dtype=torch.float32, device=device)
        zero_trace = torch.zeros_like(trace_tensor)
        surrogate72 = render_source(
            common, torch.as_tensor(surrogate_source, dtype=torch.float32, device=device), trace_tensor, ppd=float(args.ppd)
        ).cpu().numpy().astype(np.float32)
        intact_stable72 = render_source(
            common, torch.as_tensor(intact_source, dtype=torch.float32, device=device), zero_trace, ppd=float(args.ppd)
        ).cpu().numpy().astype(np.float32)
        surrogate_stable72 = render_source(
            common, torch.as_tensor(surrogate_source, dtype=torch.float32, device=device), zero_trace, ppd=float(args.ppd)
        ).cpu().numpy().astype(np.float32)
    intact72 = np.asarray(target_payload["target72"], dtype=np.float32)

    local_audit, local_arrays = audit_local_scales(
        intact72,
        surrogate72,
        train_centers=train_centers,
        offset_centers=offset_centers,
        patch_size=int(args.patch_size),
        ppd=float(args.ppd),
    )
    global_metrics, radial_surrogate = power_audit(
        intact72[N_HISTORY:], surrogate72[N_HISTORY:], ppd=float(args.ppd)
    )
    _, radial_intact = power_audit(intact72[N_HISTORY:], intact72[N_HISTORY:], ppd=float(args.ppd))
    # power_audit returns the intact radial map as its second item when both inputs are intact.
    source_rows = pd.DataFrame(
        [
            {"control": "random_phase_histogram_initialization", **source_audit(intact_source, initial_source, roi)},
            {"control": "optimized", **source_audit(intact_source, surrogate_source, roi)},
        ]
    )
    movie_metrics = movie_audit(intact72, surrogate72)

    optimization.to_csv(out_dir / "optimization_trace.csv", index=False)
    local_audit.to_csv(out_dir / "local_power_scale_and_grid_audit.csv", index=False)
    source_rows.to_csv(out_dir / "source_phase_contrast_audit.csv", index=False)
    pd.DataFrame([global_metrics | movie_metrics]).to_csv(out_dir / "global_movie_audit.csv", index=False)
    np.savez_compressed(
        out_dir / "data" / "optimized_source_and_four_condition_movies.npz",
        source_intact=intact_source,
        source_initial_random_phase_histogram=initial_source,
        source_optimized=surrogate_source,
        movie_intact_fem_full72=intact72,
        movie_surrogate_fem_full72=surrogate72,
        movie_intact_stabilized_full72=intact_stable72,
        movie_surrogate_stabilized_full72=surrogate_stable72,
        history_xy_deg=history_trace,
        score_xy_deg=score_trace,
        source_roi_bounds_yx=np.asarray([roi[0].start, roi[0].stop, roi[1].start, roi[1].stop], dtype=np.int64),
        train_centers_yx=np.asarray(train_centers, dtype=np.int64),
        offset_centers_yx=np.asarray(offset_centers, dtype=np.int64),
        median_sigma_px=np.asarray(float(args.median_sigma_px)),
        audit_sigmas_px=np.asarray(AUDIT_SIGMAS_PX, dtype=np.float64),
        lag_window=lag_window.astype(np.float32),
    )
    np.savez_compressed(out_dir / "data" / "local_power_audit_arrays.npz", **local_arrays)
    np.savez_compressed(
        out_dir / "data" / "optimization_source_snapshots.npz",
        **{f"iteration_{iteration:04d}": values for iteration, values in snapshots.items()},
    )

    plot_checkpoint(
        intact_source,
        initial_source,
        surrogate_source,
        intact72,
        surrogate72,
        optimization,
        local_audit,
        source_rows,
        radial_intact,
        radial_surrogate,
        roi=roi,
        path=out_dir / "rf_local_power_matched_phase_surrogate_checkpoint.png",
    )
    _write_mp4(
        video_frames(
            intact72,
            surrogate72,
            intact_stable72,
            surrogate_stable72,
            title=f"Image {image_index}, trace {trace_index}: one-scale RF-local power phase control",
        ),
        out_dir / "movies" / "image_068_trace_561_four_condition_input_movie.mp4",
        fps=int(args.fps),
    )

    optimized_source_audit = source_rows.loc[source_rows.control.eq("optimized")].iloc[0]
    median_train = local_audit.loc[
        local_audit.grid.eq("stochastic_training_pool_grid")
        & np.isclose(local_audit.sigma_px, float(args.median_sigma_px), atol=1e-3)
    ].iloc[0]
    median_offset = local_audit.loc[
        local_audit.grid.eq("heldout_offset_grid")
        & np.isclose(local_audit.sigma_px, float(args.median_sigma_px), atol=1e-3)
    ].iloc[0]
    manifest = {
        "analysis": "rr100_one_scale_rf_local_power_matched_phase_surrogate_input_checkpoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "input_only_human_checkpoint_no_neural_model_no_ssi",
        "image_index": image_index,
        "trace_index": trace_index,
        "source_phase_seed": scramble_seed,
        "construction": (
            "one static source image initialized by a histogram-matched global random-phase ROI is optimized "
            "through the exact differentiable 72-frame 151x151 renderer; only sigma=2.97px is constrained; "
            "deterministic stochastic center batches cover an 11x11 spatial pool"
        ),
        "optimizer": {
            "iterations": int(args.iterations),
            "learning_rate": float(args.learning_rate),
            "train_grid_size": int(args.train_grid_size),
            "n_train_pool_centers": len(train_centers),
            "center_batch_size": int(args.center_batch_size),
            "spatial_seed": int(args.spatial_seed),
            "n_validation_centers": len(validation_centers),
            "patch_size_px": int(args.patch_size),
            "optimized_sigma_px": float(args.median_sigma_px),
            "heldout_audit_sigmas_px": list(AUDIT_SIGMAS_PX),
            "history_frames": list(HISTORY_FRAMES),
            "loss_weights": {
                "local": float(args.weight_local),
                "history": float(args.weight_history),
                "global": float(args.weight_global),
                "intensity": float(args.weight_intensity),
            },
        },
        "minimum_observed_minibatch_total_loss": float(target_payload["best_loss"]),
        "minimum_minibatch_loss_iteration_not_used_for_source_selection": int(target_payload["best_iteration"]),
        "canonical_scored_movie_power": global_metrics,
        "median_scale_stochastic_training_pool_grid": median_train.to_dict(),
        "constrained_median_scale_heldout_offset_grid": median_offset.to_dict(),
        "optimized_source_audit": optimized_source_audit.to_dict(),
        "full72_movie_audit": movie_metrics,
        "input_range_valid": bool(surrogate_source.min() >= 0.0 and surrogate_source.max() <= 255.0),
        "critical_scope_limit": (
            "This is a one-image, one-trace feasibility test. A successful input audit would authorize only "
            "the targeted activation-map checkpoint, not a population conclusion."
        ),
        "inputs": {
            "source": file_identity(args.source.resolve()),
            "erf_lag_profiles": file_identity(args.erf_lag_profiles.resolve()),
        },
        "outputs": {
            name: file_identity(out_dir / name)
            for name in (
                "optimization_trace.csv",
                "local_power_scale_and_grid_audit.csv",
                "source_phase_contrast_audit.csv",
                "global_movie_audit.csv",
                "rf_local_power_matched_phase_surrogate_checkpoint.png",
                "movies/image_068_trace_561_four_condition_input_movie.mp4",
                "data/optimized_source_and_four_condition_movies.npz",
                "data/local_power_audit_arrays.npz",
                "data/optimization_source_snapshots.npz",
            )
        },
        "next_checkpoint_if_approved": (
            "targeted four-condition activation maps for audibly selected units; no population summary"
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(json_ready(manifest), indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Checkpoint 49: one-scale RF-local power-matched phase surrogate\n\n"
        "This input-only checkpoint optimizes one source image through the genuine history+score retinal "
        "renderer. It constrains only the population-median composite RF scale (sigma 2.97 px) using "
        "deterministic stochastic 25-center batches from an 11x11 pool. Four other RF-scale quantiles and "
        "a disjoint midpoint spatial grid are held out. The output must be "
        "judged from the saved movie, phase/edge audit, contrast audit, and local/global power residuals "
        "before any digital-twin response or SSI is calculated.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(manifest), indent=2))


if __name__ == "__main__":
    main()
