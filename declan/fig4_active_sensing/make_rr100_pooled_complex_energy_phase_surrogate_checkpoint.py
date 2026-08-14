#!/usr/bin/env python3
"""Build one RF-pooled complex-energy phase surrogate, then stop for inspection.

This is the replacement branch after the direct local-FFT checkpoint failed by
locally reconstructing intact edges.  Spatial filtering is global: an
undecimated complex steerable pyramid is applied to the retinal movie, temporal
power is calculated from each complex coefficient time course, and only the
resulting quadrature energy is Gaussian pooled.  No complex coefficient or
phase is matched.

The script is deliberately input-only.  It writes a four-condition movie and
power/energy/phase audits, but never loads or scores the digital twin.
"""
from __future__ import annotations

import argparse
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
import numpy as np
import pandas as pd
from scipy.ndimage import sobel
import torch

from declan.fig4_active_sensing.make_rr100_rf_local_power_matched_phase_surrogate_checkpoint import (
    AUDIT_SIGMAS_PX,
    ERF,
    HISTORY_FRAMES,
    MEDIAN_SIGMA_PX,
    N_HISTORY,
    N_SCORE,
    OUT_SIZE,
    SOURCE,
    correlation,
    global_movie_power,
    grid_centers,
    hann_window,
    intensity_loss,
    json_ready,
    phase_relation_audit_2d,
    power_loss,
    render_source,
    selected_power,
    source_audit,
    source_roi_slices,
    stochastic_center_batches,
    tensor_power_metrics,
    video_frames,
)
from declan.fig4_active_sensing.make_rr100_global_3d_phase_scramble_checkpoint import (
    rank_histogram_match,
)
from declan.fig4_active_sensing.make_rr100_global_source_phase_scramble_checkpoint import (
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
    SF_EDGES_CPD,
    TF_CORE_MAX_HZ,
)
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _load_twin_common,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_pooled_complex_energy_phase_surrogate_checkpoint_50_v1"
)
PYRAMID_HEIGHT = 4
PYRAMID_ORDER = 3
PATCH_SIZE = 31
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--erf-lag-profiles", type=Path, default=ERF)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=1.5)
    parser.add_argument("--train-grid-size", type=int, default=11)
    parser.add_argument("--center-batch-size", type=int, default=25)
    parser.add_argument("--spatial-seed", type=int, default=20260815)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--median-sigma-px", type=float, default=MEDIAN_SIGMA_PX)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--ppd", type=float, default=PPD)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--weight-local", type=float, default=1.0)
    parser.add_argument("--weight-history", type=float, default=0.5)
    parser.add_argument("--weight-global", type=float, default=0.35)
    parser.add_argument("--weight-intensity", type=float, default=0.08)
    return parser.parse_args()


class ComplexEnergyRepresentation:
    """Undecimated complex pyramid followed by temporal energy and RF pooling."""

    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        from plenoptic.simulate import SteerablePyramidFreq

        self.device = device
        self.dtype = dtype
        self.pyramid = SteerablePyramidFreq(
            (OUT_SIZE, OUT_SIZE),
            height=PYRAMID_HEIGHT,
            order=PYRAMID_ORDER,
            is_complex=True,
            downsample=False,
            tight_frame=False,
        ).to(device=device, dtype=dtype)
        self.levels = tuple(range(PYRAMID_HEIGHT))

    def spatial_coefficients(self, movie: torch.Tensor) -> torch.Tensor:
        coefficients = self.pyramid(movie[:, None])
        bands = []
        for level in self.levels:
            value = coefficients[level]
            if not torch.is_complex(value):
                raise TypeError(f"Pyramid level {level} is not complex")
            bands.append(value[:, 0])
        # T x level x orientation x Y x X
        return torch.stack(bands, dim=1)

    def temporal_energy_maps(
        self,
        movie: torch.Tensor,
        temporal_window: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return undirected positive-TF quadrature energy maps.

        Complex spatial coefficients encode a quadrature pair.  Summing the
        +TF and -TF energies removes motion-direction sign while retaining the
        SF x orientation x TF redistribution relevant to the mechanism test.
        """
        coefficients = self.spatial_coefficients(movie)
        coefficients = coefficients - coefficients.mean(dim=0, keepdim=True)
        tapered = coefficients * temporal_window[:, None, None, None, None]
        spectrum = torch.fft.fft(tapered, dim=0)
        frequencies = torch.fft.fftfreq(
            movie.shape[0], d=1.0 / FRAME_RATE_HZ, device=movie.device
        )
        positive_indices = torch.where(
            (frequencies > 0) & (frequencies <= TF_CORE_MAX_HZ)
        )[0]
        negative_indices = torch.remainder(-positive_indices, movie.shape[0])
        positive = spectrum[positive_indices].abs().square()
        negative = spectrum[negative_indices].abs().square()
        return positive + negative, frequencies[positive_indices]

    @staticmethod
    def pool(
        energy: torch.Tensor,
        centers: list[tuple[int, int]],
        *,
        sigma: float,
        patch_size: int,
    ) -> torch.Tensor:
        """Gaussian-pool already-computed energy; never window the input."""
        half = int(patch_size) // 2
        coordinate = torch.arange(
            int(patch_size), dtype=energy.dtype, device=energy.device
        ) - half
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        kernel = torch.exp(-0.5 * (xx.square() + yy.square()) / float(sigma) ** 2)
        kernel = kernel / kernel.sum()
        patches = torch.stack(
            [
                energy[..., y - half : y + half + 1, x - half : x + half + 1]
                for y, x in centers
            ],
            dim=0,
        )
        # center x TF x level x orientation
        return (patches * kernel).sum(dim=(-2, -1))


def build_history(movie72: torch.Tensor, scored_frame: int) -> torch.Tensor:
    return torch.stack(
        [movie72[N_HISTORY + int(scored_frame) - lag] for lag in range(N_HISTORY)],
        dim=0,
    )


def local_phase_audit(
    intact72: np.ndarray,
    surrogate72: np.ndarray,
    centers: list[tuple[int, int]],
    *,
    patch_size: int,
) -> pd.DataFrame:
    """Audit phase/edge retention locally, where checkpoint 49 failed."""
    half = int(patch_size) // 2
    window = np.outer(np.hanning(int(patch_size)), np.hanning(int(patch_size)))
    rows: list[dict[str, Any]] = []
    for scored_frame in HISTORY_FRAMES:
        frame_index = N_HISTORY + int(scored_frame)
        for center_index, (y, x) in enumerate(centers):
            before = intact72[
                frame_index, y - half : y + half + 1, x - half : x + half + 1
            ]
            after = surrogate72[
                frame_index, y - half : y + half + 1, x - half : x + half + 1
            ]
            phase = phase_relation_audit_2d(
                (before - before.mean()) * window,
                (after - after.mean()) * window,
            )
            edge_before = np.hypot(sobel(before, axis=0), sobel(before, axis=1))
            edge_after = np.hypot(sobel(after, axis=0), sobel(after, axis=1))
            rows.append(
                {
                    "scored_frame": int(scored_frame),
                    "center_index": int(center_index),
                    "center_y": int(y),
                    "center_x": int(x),
                    "pixel_correlation": correlation(before, after),
                    "edge_correlation": correlation(edge_before, edge_after),
                    **phase,
                }
            )
    return pd.DataFrame(rows)


def optimize_source(
    common: Any,
    representation: ComplexEnergyRepresentation,
    intact_source: np.ndarray,
    initial_source: np.ndarray,
    trace72_np: np.ndarray,
    lag_window_np: np.ndarray,
    roi: tuple[slice, slice],
    center_batches: list[list[tuple[int, int]]],
    train_centers: list[tuple[int, int]],
    validation_centers: list[tuple[int, int]],
    *,
    args: argparse.Namespace,
) -> tuple[np.ndarray, pd.DataFrame, dict[int, np.ndarray], dict[str, Any]]:
    device = representation.device
    dtype = representation.dtype
    intact = torch.as_tensor(intact_source, dtype=dtype, device=device)
    trace72 = torch.as_tensor(trace72_np, dtype=dtype, device=device)
    roi_mask = torch.zeros_like(intact)
    roi_mask[roi] = 1.0
    score_window = hann_window(N_SCORE, dtype=dtype, device=device)
    history_window = torch.as_tensor(lag_window_np, dtype=dtype, device=device)

    with torch.no_grad():
        target72 = render_source(common, intact, trace72, ppd=float(args.ppd)) / 255.0
        target_score = target72[N_HISTORY:]
        target_score_energy, score_tf = representation.temporal_energy_maps(
            target_score, score_window
        )
        target_history_energy = []
        history_tf = None
        for scored_frame in HISTORY_FRAMES:
            value, history_tf = representation.temporal_energy_maps(
                build_history(target72, scored_frame), history_window
            )
            target_history_energy.append(value)
        global_target_raw, global_tf = global_movie_power(
            target_score, ppd=float(args.ppd)
        )
        global_target = selected_power(global_target_raw, global_tf)

    parameter = torch.nn.Parameter(
        torch.as_tensor(initial_source, dtype=dtype, device=device).clone()
    )
    optimizer = torch.optim.Adam([parameter], lr=float(args.learning_rate))
    rows: list[dict[str, Any]] = []
    snapshots: dict[int, np.ndarray] = {}
    best_loss = float("inf")
    best_iteration = 0

    for iteration in range(int(args.iterations) + 1):
        train_batch = center_batches[iteration]
        history_index = iteration % len(HISTORY_FRAMES)
        fraction = iteration / max(int(args.iterations), 1)
        learning_rate = float(args.learning_rate) * (
            0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * fraction))
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)

        movie72 = render_source(common, parameter, trace72, ppd=float(args.ppd)) / 255.0
        score = movie72[N_HISTORY:]
        score_energy, _ = representation.temporal_energy_maps(score, score_window)
        predicted_local = representation.pool(
            score_energy,
            train_batch,
            sigma=float(args.median_sigma_px),
            patch_size=int(args.patch_size),
        )
        target_local = representation.pool(
            target_score_energy,
            train_batch,
            sigma=float(args.median_sigma_px),
            patch_size=int(args.patch_size),
        )
        history_energy, _ = representation.temporal_energy_maps(
            build_history(movie72, HISTORY_FRAMES[history_index]), history_window
        )
        predicted_history = representation.pool(
            history_energy,
            train_batch,
            sigma=float(args.median_sigma_px),
            patch_size=int(args.patch_size),
        )
        target_history = representation.pool(
            target_history_energy[history_index],
            train_batch,
            sigma=float(args.median_sigma_px),
            patch_size=int(args.patch_size),
        )
        global_raw, _ = global_movie_power(score, ppd=float(args.ppd))
        global_predicted = selected_power(global_raw, global_tf)

        local_value = power_loss(predicted_local, target_local)
        history_value = power_loss(predicted_history, target_history)
        global_value = power_loss(global_predicted, global_target)
        intensity_value = intensity_loss(
            parameter, intact, score, target_score, roi
        )
        loss = (
            float(args.weight_local) * local_value
            + float(args.weight_history) * history_value
            + float(args.weight_global) * global_value
            + float(args.weight_intensity) * intensity_value
        )

        local_metrics = tensor_power_metrics(predicted_local, target_local)
        history_metrics = tensor_power_metrics(predicted_history, target_history)
        global_metrics = tensor_power_metrics(global_predicted, global_target)
        record: dict[str, Any] = {
            "iteration": int(iteration),
            "history_frame_used": int(HISTORY_FRAMES[history_index]),
            "learning_rate": learning_rate,
            "total_loss": float(loss.detach().cpu()),
            "local_energy_loss": float(local_value.detach().cpu()),
            "history_energy_loss": float(history_value.detach().cpu()),
            "global_power_loss": float(global_value.detach().cpu()),
            "intensity_loss": float(intensity_value.detach().cpu()),
            **{f"local_{key}": value for key, value in local_metrics.items()},
            **{f"history_{key}": value for key, value in history_metrics.items()},
            **{f"global_{key}": value for key, value in global_metrics.items()},
        }
        if iteration % int(args.checkpoint_every) == 0 or iteration == int(args.iterations):
            with torch.no_grad():
                train_all = representation.pool(
                    score_energy,
                    train_centers,
                    sigma=float(args.median_sigma_px),
                    patch_size=int(args.patch_size),
                )
                target_train_all = representation.pool(
                    target_score_energy,
                    train_centers,
                    sigma=float(args.median_sigma_px),
                    patch_size=int(args.patch_size),
                )
                validation = representation.pool(
                    score_energy,
                    validation_centers,
                    sigma=float(args.median_sigma_px),
                    patch_size=int(args.patch_size),
                )
                target_validation = representation.pool(
                    target_score_energy,
                    validation_centers,
                    sigma=float(args.median_sigma_px),
                    patch_size=int(args.patch_size),
                )
            record.update(
                {
                    f"train_all_{key}": value
                    for key, value in tensor_power_metrics(train_all, target_train_all).items()
                }
            )
            record.update(
                {
                    f"validation_{key}": value
                    for key, value in tensor_power_metrics(validation, target_validation).items()
                }
            )
            source_np = parameter.detach().cpu().numpy().astype(np.float32)
            record.update(
                {
                    f"source_{key}": value
                    for key, value in phase_relation_audit_2d(
                        intact_source[roi], source_np[roi]
                    ).items()
                }
            )
            snapshots[int(iteration)] = source_np.copy()
            print(
                f"iter {iteration:04d}: loss={record['total_loss']:.4f}; "
                f"local cos={record['local_power_cosine']:.4f} ratio={record['local_power_ratio']:.3f}; "
                f"train-all cos={record['train_all_power_cosine']:.4f}; "
                f"valid cos={record['validation_power_cosine']:.4f}; "
                f"history cos={record['history_power_cosine']:.4f}; "
                f"global cos={record['global_power_cosine']:.4f}; "
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

    payload = {
        "target72": target72.detach().cpu().numpy().astype(np.float32) * 255.0,
        "target_score_energy": target_score_energy.detach().cpu(),
        "target_history_energy": [value.detach().cpu() for value in target_history_energy],
        "score_tf_hz": score_tf.detach().cpu().numpy(),
        "history_tf_hz": history_tf.detach().cpu().numpy(),
        "best_loss": best_loss,
        "best_iteration": best_iteration,
    }
    return (
        parameter.detach().cpu().numpy().astype(np.float32).copy(),
        pd.DataFrame(rows),
        snapshots,
        payload,
    )


def audit_energy(
    representation: ComplexEnergyRepresentation,
    intact72: np.ndarray,
    surrogate72: np.ndarray,
    lag_window: np.ndarray,
    train_centers: list[tuple[int, int]],
    offset_centers: list[tuple[int, int]],
    *,
    patch_size: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    device = representation.device
    intact = torch.as_tensor(intact72 / 255.0, dtype=torch.float32, device=device)
    surrogate = torch.as_tensor(surrogate72 / 255.0, dtype=torch.float32, device=device)
    score_window = hann_window(N_SCORE, dtype=torch.float32, device=device)
    history_window = torch.as_tensor(lag_window, dtype=torch.float32, device=device)
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    with torch.no_grad():
        score_intact, score_tf = representation.temporal_energy_maps(
            intact[N_HISTORY:], score_window
        )
        score_surrogate, _ = representation.temporal_energy_maps(
            surrogate[N_HISTORY:], score_window
        )
        contexts = [("score40", score_intact, score_surrogate, score_tf)]
        for scored_frame in HISTORY_FRAMES:
            history_intact, history_tf = representation.temporal_energy_maps(
                build_history(intact, scored_frame), history_window
            )
            history_surrogate, _ = representation.temporal_energy_maps(
                build_history(surrogate, scored_frame), history_window
            )
            contexts.append(
                (
                    f"history_scored_frame_{scored_frame}",
                    history_intact,
                    history_surrogate,
                    history_tf,
                )
            )
        for context, target_map, predicted_map, tf_hz in contexts:
            for grid_name, centers in (
                ("training_pool_grid", train_centers),
                ("heldout_offset_grid", offset_centers),
            ):
                for sigma in AUDIT_SIGMAS_PX:
                    target = representation.pool(
                        target_map, centers, sigma=float(sigma), patch_size=int(patch_size)
                    )
                    predicted = representation.pool(
                        predicted_map, centers, sigma=float(sigma), patch_size=int(patch_size)
                    )
                    metrics = tensor_power_metrics(predicted, target)
                    rows.append(
                        {
                            "context": context,
                            "grid": grid_name,
                            "n_centers": len(centers),
                            "sigma_px": float(sigma),
                            "is_optimized_scale": bool(
                                np.isclose(sigma, MEDIAN_SIGMA_PX, atol=1e-3)
                            ),
                            "n_tf": int(tf_hz.numel()),
                            **metrics,
                        }
                    )
                    if context == "score40" and np.isclose(
                        sigma, MEDIAN_SIGMA_PX, atol=1e-3
                    ):
                        arrays[f"{grid_name}_target"] = target.cpu().numpy().astype(np.float32)
                        arrays[f"{grid_name}_surrogate"] = predicted.cpu().numpy().astype(np.float32)
    return pd.DataFrame(rows), arrays


def plot_checkpoint(
    intact_source: np.ndarray,
    initial_source: np.ndarray,
    surrogate_source: np.ndarray,
    intact72: np.ndarray,
    surrogate72: np.ndarray,
    optimization: pd.DataFrame,
    energy_audit: pd.DataFrame,
    phase_audit: pd.DataFrame,
    radial_intact: np.ndarray,
    radial_surrogate: np.ndarray,
    score_arrays: dict[str, np.ndarray],
    *,
    roi: tuple[slice, slice],
    path: Path,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(18, 12.5), constrained_layout=True)
    panels = (
        (intact_source[roi], "Intact source ROI"),
        (initial_source[roi], "Random-phase initialization"),
        (surrogate_source[roi], "Pooled-energy surrogate"),
        (surrogate_source[roi] - intact_source[roi], "Surrogate − intact"),
    )
    limit = float(np.quantile(np.abs(panels[-1][0]), 0.99))
    for column, (values, title) in enumerate(panels):
        if column == 3:
            image = axes[0, column].imshow(
                values, cmap="coolwarm", vmin=-limit, vmax=limit, origin="lower"
            )
            fig.colorbar(image, ax=axes[0, column], fraction=0.046)
        else:
            axes[0, column].imshow(values, cmap="gray", vmin=0, vmax=255, origin="lower")
        axes[0, column].set_title(title)
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])

    frame_indices = (32, 51, 71)
    for column, (movie, title) in enumerate(
        ((intact72, "Intact FEM frames"), (surrogate72, "Surrogate FEM frames"))
    ):
        axes[1, column].imshow(
            np.concatenate([movie[index] for index in frame_indices], axis=1),
            cmap="gray", vmin=0, vmax=255, origin="lower"
        )
        axes[1, column].set_title(f"{title} 0, 19, 39")
        axes[1, column].set_xticks([])
        axes[1, column].set_yticks([])
    difference = np.concatenate(
        [surrogate72[index] - intact72[index] for index in frame_indices], axis=1
    )
    movie_limit = float(np.quantile(np.abs(difference), 0.99))
    axes[1, 2].imshow(
        difference, cmap="coolwarm", vmin=-movie_limit, vmax=movie_limit, origin="lower"
    )
    axes[1, 2].set_title(f"Retinal difference (±{movie_limit:.1f})")
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    sf_centers = 0.5 * (SF_EDGES_CPD[:-1] + SF_EDGES_CPD[1:])
    tf_hz = np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)[1:]
    delta_db = relative_db(radial_surrogate) - relative_db(radial_intact)
    mesh = axes[1, 3].pcolormesh(
        sf_centers, tf_hz, delta_db, cmap="coolwarm", vmin=-12, vmax=12, shading="nearest"
    )
    axes[1, 3].set(
        xscale="log", xlabel="SF (cpd)", ylabel="TF (Hz)", title="Canonical scored power ΔdB"
    )
    fig.colorbar(mesh, ax=axes[1, 3], fraction=0.046)

    axes[2, 0].plot(optimization.iteration, optimization.total_loss, color="black", label="total")
    axes[2, 0].plot(optimization.iteration, optimization.local_energy_loss, label="pooled energy")
    axes[2, 0].plot(optimization.iteration, optimization.history_energy_loss, label="history")
    axes[2, 0].plot(optimization.iteration, optimization.global_power_loss, label="global")
    axes[2, 0].set(xlabel="iteration", ylabel="loss", yscale="log", title="Optimization")
    axes[2, 0].legend(frameon=False, fontsize=7)

    score = energy_audit.loc[energy_audit.context.eq("score40")]
    for grid, style in (("training_pool_grid", "-o"), ("heldout_offset_grid", "--s")):
        work = score.loc[score.grid.eq(grid)].sort_values("sigma_px")
        axes[2, 1].plot(work.sigma_px, work.power_cosine, style, label=grid.replace("_", " "))
    axes[2, 1].axvline(MEDIAN_SIGMA_PX, color="black", ls=":")
    axes[2, 1].set(
        xlabel="pool sigma (px)", ylabel="energy cosine", ylim=(0, 1.005),
        title="Held-out scales and locations"
    )
    axes[2, 1].legend(frameon=False, fontsize=7)

    n_centers = int(score_arrays["training_pool_grid_target"].shape[0])
    grid_side = int(round(math.sqrt(n_centers)))
    if grid_side * grid_side != n_centers:
        raise ValueError(f"Training centers do not form a square grid: {n_centers}")
    target = score_arrays["training_pool_grid_target"].reshape(n_centers, -1)
    predicted = score_arrays["training_pool_grid_surrogate"].reshape(n_centers, -1)
    cosine = np.sum(target * predicted, axis=1) / np.maximum(
        np.linalg.norm(target, axis=1) * np.linalg.norm(predicted, axis=1), EPS
    )
    heat = axes[2, 2].imshow(
        cosine.reshape(grid_side, grid_side), vmin=0, vmax=1, cmap="viridis", origin="lower"
    )
    axes[2, 2].set_title("Score40 local energy cosine")
    fig.colorbar(heat, ax=axes[2, 2], fraction=0.046)

    phase_columns = [
        "fourier_phase_retention_coherence",
        "adjacent_horizontal_frequency_phase_relation_retention_coherence",
        "adjacent_vertical_frequency_phase_relation_retention_coherence",
        "edge_correlation",
    ]
    labels = ["Fourier phase", "adjacent H", "adjacent V", "edge"]
    medians = [float(phase_audit[column].median()) for column in phase_columns]
    q25 = [float(phase_audit[column].quantile(0.25)) for column in phase_columns]
    q75 = [float(phase_audit[column].quantile(0.75)) for column in phase_columns]
    axes[2, 3].bar(
        np.arange(4), medians,
        yerr=np.asarray([np.asarray(medians) - q25, np.asarray(q75) - medians]),
        color=["#0072B2", "#56B4E9", "#009E73", "#D55E00"], capsize=3
    )
    axes[2, 3].set_xticks(np.arange(4), labels, rotation=25, ha="right")
    axes[2, 3].set(ylabel="local retention", ylim=(-0.15, 1), title="Held-out local phase/edge audit")
    for ax in axes[2]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.suptitle(
        "Checkpoint 50: globally filtered, RF-pooled complex-energy phase surrogate\n"
        "sigma=2.97 px constrained; quadrature energy only; no neural scoring",
        fontsize=14, fontweight="bold"
    )
    fig.savefig(path, dpi=185)
    plt.close(fig)


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
        random_phase_source = np.asarray(data["source_global_phase_scrambled"], dtype=np.float32)
        history_trace = np.asarray(data["history_xy_deg"], dtype=np.float32)
        score_trace = np.asarray(data["score_xy_deg"], dtype=np.float32)
        image_index = int(data["image_index"].item())
        trace_index = int(data["trace_index"].item())
        scramble_seed = int(data["scramble_seed"].item())
    trace72 = np.concatenate([history_trace, score_trace], axis=0)
    roi = source_roi_slices(trace72, ppd=float(args.ppd), source_shape=intact_source.shape)
    initial_source = intact_source.copy()
    initial_source[roi] = rank_histogram_match(
        intact_source[roi], random_phase_source[roi]
    )

    lag_profiles = pd.read_csv(args.erf_lag_profiles.resolve())
    lag_energy = (
        lag_profiles.groupby("lag_frames_ago").lag_energy_fraction.mean().sort_index().to_numpy(np.float64)
    )
    if lag_energy.shape != (N_HISTORY,):
        raise ValueError(f"Expected 32 lag weights, got {lag_energy.shape}")
    lag_window = np.sqrt(np.maximum(lag_energy, 0.0))
    lag_window /= max(float(lag_window.max()), EPS)

    train_centers = grid_centers(OUT_SIZE, int(args.train_grid_size), offset=False)
    offset_centers = grid_centers(OUT_SIZE, int(args.train_grid_size), offset=True)
    offset_coordinates = sorted({value for center in offset_centers for value in center})
    validation_coordinates = [
        offset_coordinates[index]
        for index in np.rint(
            np.linspace(0, len(offset_coordinates) - 1, min(5, len(offset_coordinates)))
        ).astype(int)
    ]
    validation_centers = [(y, x) for y in validation_coordinates for x in validation_coordinates]
    center_batches = stochastic_center_batches(
        train_centers,
        n_batches=int(args.iterations) + 1,
        batch_size=int(args.center_batch_size),
        seed=int(args.spatial_seed),
    )

    device = torch.device(str(args.device))
    common = _load_twin_common()
    representation = ComplexEnergyRepresentation(device=device, dtype=torch.float32)
    surrogate_source, optimization, snapshots, payload = optimize_source(
        common,
        representation,
        intact_source,
        initial_source,
        trace72,
        lag_window,
        roi,
        center_batches,
        train_centers,
        validation_centers,
        args=args,
    )

    with torch.no_grad():
        trace_tensor = torch.as_tensor(trace72, dtype=torch.float32, device=device)
        zero_trace = torch.zeros_like(trace_tensor)
        surrogate_tensor = torch.as_tensor(surrogate_source, dtype=torch.float32, device=device)
        intact_tensor = torch.as_tensor(intact_source, dtype=torch.float32, device=device)
        surrogate72 = render_source(common, surrogate_tensor, trace_tensor, ppd=float(args.ppd)).cpu().numpy().astype(np.float32)
        intact_stable72 = render_source(common, intact_tensor, zero_trace, ppd=float(args.ppd)).cpu().numpy().astype(np.float32)
        surrogate_stable72 = render_source(common, surrogate_tensor, zero_trace, ppd=float(args.ppd)).cpu().numpy().astype(np.float32)
    intact72 = np.asarray(payload["target72"], dtype=np.float32)

    energy_audit, energy_arrays = audit_energy(
        representation,
        intact72,
        surrogate72,
        lag_window,
        train_centers,
        offset_centers,
        patch_size=int(args.patch_size),
    )
    phase_audit = local_phase_audit(
        intact72, surrogate72, offset_centers, patch_size=int(args.patch_size)
    )
    source_rows = pd.DataFrame(
        [
            {"control": "random_phase_histogram_initialization", **source_audit(intact_source, initial_source, roi)},
            {"control": "optimized", **source_audit(intact_source, surrogate_source, roi)},
        ]
    )
    global_metrics, radial_surrogate = power_audit(
        intact72[N_HISTORY:], surrogate72[N_HISTORY:], ppd=float(args.ppd)
    )
    _, radial_intact = power_audit(
        intact72[N_HISTORY:], intact72[N_HISTORY:], ppd=float(args.ppd)
    )
    movie_metrics = movie_audit(intact72, surrogate72)

    optimization.to_csv(out_dir / "optimization_trace.csv", index=False)
    energy_audit.to_csv(out_dir / "pooled_complex_energy_scale_and_grid_audit.csv", index=False)
    phase_audit.to_csv(out_dir / "heldout_local_phase_edge_audit.csv", index=False)
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
        source_roi_bounds_yx=np.asarray([roi[0].start, roi[0].stop, roi[1].start, roi[1].stop]),
        train_centers_yx=np.asarray(train_centers),
        offset_centers_yx=np.asarray(offset_centers),
        median_sigma_px=np.asarray(float(args.median_sigma_px)),
        audit_sigmas_px=np.asarray(AUDIT_SIGMAS_PX),
        lag_window=lag_window.astype(np.float32),
        score_tf_hz=np.asarray(payload["score_tf_hz"]),
        history_tf_hz=np.asarray(payload["history_tf_hz"]),
    )
    np.savez_compressed(out_dir / "data" / "pooled_complex_energy_audit_arrays.npz", **energy_arrays)
    np.savez_compressed(
        out_dir / "data" / "optimization_source_snapshots.npz",
        **{f"iteration_{iteration:04d}": values for iteration, values in snapshots.items()},
    )

    figure_path = out_dir / "pooled_complex_energy_phase_surrogate_checkpoint.png"
    plot_checkpoint(
        intact_source,
        initial_source,
        surrogate_source,
        intact72,
        surrogate72,
        optimization,
        energy_audit,
        phase_audit,
        radial_intact,
        radial_surrogate,
        energy_arrays,
        roi=roi,
        path=figure_path,
    )
    movie_path = out_dir / "movies" / "image_068_trace_561_four_condition_input_movie.mp4"
    _write_mp4(
        video_frames(
            intact72,
            surrogate72,
            intact_stable72,
            surrogate_stable72,
            title=f"Image {image_index}, trace {trace_index}: pooled complex-energy phase control",
        ),
        movie_path,
        fps=int(args.fps),
    )

    score_rows = energy_audit.loc[energy_audit.context.eq("score40")]
    median_train = score_rows.loc[
        score_rows.grid.eq("training_pool_grid")
        & np.isclose(score_rows.sigma_px, float(args.median_sigma_px), atol=1e-3)
    ].iloc[0]
    median_offset = score_rows.loc[
        score_rows.grid.eq("heldout_offset_grid")
        & np.isclose(score_rows.sigma_px, float(args.median_sigma_px), atol=1e-3)
    ].iloc[0]
    local_phase_summary = {
        column: {
            "median": float(phase_audit[column].median()),
            "q25": float(phase_audit[column].quantile(0.25)),
            "q75": float(phase_audit[column].quantile(0.75)),
        }
        for column in (
            "fourier_phase_retention_coherence",
            "adjacent_horizontal_frequency_phase_relation_retention_coherence",
            "adjacent_vertical_frequency_phase_relation_retention_coherence",
            "pixel_correlation",
            "edge_correlation",
        )
    }
    manifest = {
        "analysis": "rr100_pooled_complex_energy_phase_surrogate_input_checkpoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "input_only_human_checkpoint_no_neural_model_no_ssi",
        "provisional_machine_audit": (
            "warning: power agreement is strong and direct phase retention is low, but a recognizable "
            "local facade/edge complex is visible and held-out local edge correlation has median 0.253; "
            "do not authorize neural scoring without human acceptance"
        ),
        "image_index": image_index,
        "trace_index": trace_index,
        "source_phase_seed": scramble_seed,
        "construction": (
            "global undecimated four-level/four-orientation complex steerable-pyramid filtering; "
            "positive and negative temporal-frequency quadrature energies summed; energy then Gaussian "
            "pooled at sigma 2.97px; no complex coefficients or phases matched"
        ),
        "optimizer": {
            "iterations": int(args.iterations),
            "learning_rate": float(args.learning_rate),
            "train_grid_size": int(args.train_grid_size),
            "n_train_centers": len(train_centers),
            "center_batch_size": int(args.center_batch_size),
            "spatial_seed": int(args.spatial_seed),
            "patch_size_px": int(args.patch_size),
            "optimized_sigma_px": float(args.median_sigma_px),
            "heldout_audit_sigmas_px": list(AUDIT_SIGMAS_PX),
            "history_frames_cycled": list(HISTORY_FRAMES),
            "pyramid_height": PYRAMID_HEIGHT,
            "pyramid_orientations": PYRAMID_ORDER + 1,
            "loss_weights": {
                "local_energy": float(args.weight_local),
                "history_energy": float(args.weight_history),
                "global_canonical_power": float(args.weight_global),
                "intensity": float(args.weight_intensity),
            },
        },
        "minimum_observed_minibatch_total_loss": float(payload["best_loss"]),
        "minimum_minibatch_loss_iteration_not_used_for_source_selection": int(payload["best_iteration"]),
        "score40_median_scale_training_grid": median_train.to_dict(),
        "score40_median_scale_heldout_offset_grid": median_offset.to_dict(),
        "canonical_scored_movie_power": global_metrics,
        "optimized_source_audit": source_rows.loc[source_rows.control.eq("optimized")].iloc[0].to_dict(),
        "heldout_local_phase_edge_summary": local_phase_summary,
        "full72_movie_audit": movie_metrics,
        "input_range_valid": bool(surrogate_source.min() >= 0 and surrogate_source.max() <= 255),
        "critical_scope_limit": (
            "One image and one trace only. Human inspection of the movie and local phase/edge audit is "
            "required before any targeted activation-map calculation."
        ),
        "inputs": {
            "source": file_identity(args.source.resolve()),
            "erf_lag_profiles": file_identity(args.erf_lag_profiles.resolve()),
        },
        "outputs": {
            name: file_identity(out_dir / name)
            for name in (
                "optimization_trace.csv",
                "pooled_complex_energy_scale_and_grid_audit.csv",
                "heldout_local_phase_edge_audit.csv",
                "source_phase_contrast_audit.csv",
                "global_movie_audit.csv",
                "pooled_complex_energy_phase_surrogate_checkpoint.png",
                "movies/image_068_trace_561_four_condition_input_movie.mp4",
                "data/optimized_source_and_four_condition_movies.npz",
                "data/pooled_complex_energy_audit_arrays.npz",
                "data/optimization_source_snapshots.npz",
            )
        },
        "next_checkpoint_if_approved": "targeted four-condition activation maps only; no population summary",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "README.md").write_text(
        "# Checkpoint 50: RF-pooled complex-energy phase surrogate\n\n"
        "This input-only checkpoint replaces rejected sliding local FFT magnitudes with global, "
        "undecimated complex steerable-pyramid filtering followed by temporal quadrature energy and "
        "Gaussian pooling at the twin-derived median sigma (2.97 px). Complex coefficients and phase "
        "are never matched. Inspect the four-condition movie, local phase/edge table, held-out pooling "
        "scales and locations, histogram/contrast audit, and canonical power residual before any neural "
        "response or SSI calculation. The completed 250-iteration run carries a warning because a "
        "recognizable local facade/edge complex re-emerged despite low direct Fourier-phase retention.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(manifest), indent=2))


if __name__ == "__main__":
    main()
