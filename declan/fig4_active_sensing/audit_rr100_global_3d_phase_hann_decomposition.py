#!/usr/bin/env python3
"""Decompose the scored-window Hann-power mismatch in the phase ensemble."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig4_active_sensing.make_rr100_global_source_phase_scramble_checkpoint import file_identity
from declan.fig4_active_sensing.run_interim_input_spectral_cache import (
    FRAME_RATE_HZ,
    SF_EDGES_CPD,
    spatial_lookup,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/fig4_active_sensing/rr100_global_3d_phase_ensemble_movie_checkpoint_46_v3"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_global_3d_phase_hann_decomposition_checkpoint_47_v1"
PPD = 37.5
N_HISTORY = 32


def spectral_sums(
    movie: np.ndarray,
    *,
    temporal_hann: bool,
    spatial_hann: bool,
) -> tuple[float, float]:
    values = np.asarray(movie, dtype=np.float64)
    n_frames, height, width = values.shape
    if height != width:
        raise ValueError(f"Expected a square movie, got {values.shape}")
    residual = values - values.mean(axis=0, keepdims=True)
    if temporal_hann:
        residual *= np.hanning(n_frames)[:, None, None]
    if spatial_hann:
        residual *= np.outer(np.hanning(height), np.hanning(width))[None]
    temporal_fft = np.fft.rfft(residual, axis=0)
    spectrum = np.fft.fftshift(np.fft.fft2(temporal_fft, axes=(1, 2)), axes=(1, 2))
    power = np.abs(spectrum) ** 2
    temporal_weights = np.ones(power.shape[0], dtype=np.float64)
    if n_frames % 2 == 0:
        temporal_weights[1:-1] = 2.0
    else:
        temporal_weights[1:] = 2.0
    power *= temporal_weights[:, None, None]
    tf_hz = np.fft.rfftfreq(n_frames, d=1.0 / FRAME_RATE_HZ)
    positive = tf_hz > 0
    sf_bin, _, _ = spatial_lookup(PPD, size=height)
    sf_centers = 0.5 * (SF_EDGES_CPD[:-1] + SF_EDGES_CPD[1:])
    radial = np.stack(
        [
            np.bincount(sf_bin, weights=power[index].ravel(), minlength=len(sf_centers))
            for index in np.flatnonzero(positive)
        ]
    )
    positive_tf = tf_hz[positive]
    supported = (
        (positive_tf[:, None] <= 56.0)
        & (sf_centers[None, :] >= 1.0)
        & (sf_centers[None, :] <= 11.3137085)
    )
    return float(radial.sum()), float(radial[supported].sum())


def plot_decomposition(rows: pd.DataFrame, pixel: pd.DataFrame, path: Path) -> None:
    order = [
        "full72_rectangular",
        "score40_rectangular",
        "score40_temporal_hann",
        "score40_spatial_hann",
        "score40_full_hann",
    ]
    labels = [
        "Full 72\nrectangular",
        "Score 40\nrectangular",
        "Score 40\ntemporal Hann",
        "Score 40\nspatial Hann",
        "Score 40\nfull Hann",
    ]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.7), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.9, rows.seed.nunique()))
    for color, (seed, group) in zip(colors, rows.groupby("seed", sort=True), strict=True):
        indexed = group.set_index("condition").loc[order]
        axes[0].plot(x, indexed.total_positive_tf_ratio, marker="o", color=color, alpha=0.85, label=str(seed))
        axes[1].plot(x, indexed.supported_sf_tf_ratio, marker="o", color=color, alpha=0.85)
    for ax, title, ylabel in (
        (axes[0], "All positive-TF modulation power", "surrogate / intact"),
        (axes[1], "Supported SF×TF modulation power", "surrogate / intact"),
    ):
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    axes[0].legend(title="seed", fontsize=7, title_fontsize=8)
    pixel_metrics = [
        "score40_mean_luminance_ratio",
        "score40_second_moment_ratio",
        "score40_global_variance_ratio",
        "score40_framewise_spatial_variance_ratio",
        "score40_pixelwise_temporal_variance_ratio",
    ]
    pixel_labels = ["mean", "second\nmoment", "global\nvariance", "spatial\nvariance", "temporal\nvariance"]
    for index, metric in enumerate(pixel_metrics):
        values = pixel[metric].to_numpy(float)
        axes[2].scatter(np.full_like(values, index, dtype=float), values, c=colors, s=35)
        axes[2].plot([index - 0.18, index + 0.18], [np.median(values)] * 2, color="black", linewidth=2)
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_xticks(np.arange(len(pixel_metrics)), pixel_labels)
    axes[2].set_ylabel("surrogate / intact")
    axes[2].set_title("Direct scored-window pixel quantities")
    axes[2].grid(axis="y", alpha=0.22)
    fig.suptitle(
        "Where exact global power ceases to match after localization\n"
        "Five predeclared phase realizations; image 68; no neural scoring",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {OUT}")
    OUT.mkdir(parents=True)
    arrays_path = SOURCE / "phase_ensemble_movies.npz"
    with np.load(arrays_path, allow_pickle=False) as data:
        intact = np.asarray(data["movie_intact_full72"], dtype=np.float64)
        surrogates = np.asarray(data["movie_phase_ensemble_full72"], dtype=np.float64)
        seeds = np.asarray(data["seeds"], dtype=np.int64)
        image_index = int(data["image_index"].item())
        trace_index = int(data["trace_index"].item())
    conditions = (
        ("full72_rectangular", slice(None), False, False),
        ("score40_rectangular", slice(N_HISTORY, None), False, False),
        ("score40_temporal_hann", slice(N_HISTORY, None), True, False),
        ("score40_spatial_hann", slice(N_HISTORY, None), False, True),
        ("score40_full_hann", slice(N_HISTORY, None), True, True),
    )
    rows = []
    for condition, selection, temporal_hann, spatial_hann in conditions:
        intact_total, intact_supported = spectral_sums(
            intact[selection], temporal_hann=temporal_hann, spatial_hann=spatial_hann
        )
        for seed, surrogate in zip(seeds, surrogates, strict=True):
            control_total, control_supported = spectral_sums(
                surrogate[selection], temporal_hann=temporal_hann, spatial_hann=spatial_hann
            )
            rows.append(
                {
                    "image_index": image_index,
                    "trace_index": trace_index,
                    "seed": int(seed),
                    "condition": condition,
                    "temporal_selection": "full72" if selection == slice(None) else "scored40",
                    "temporal_hann": temporal_hann,
                    "spatial_hann": spatial_hann,
                    "total_positive_tf_ratio": control_total / intact_total,
                    "supported_sf_tf_ratio": control_supported / intact_supported,
                }
            )
    decomposition = pd.DataFrame(rows)
    pixel_rows = []
    intact_score = intact[N_HISTORY:]
    for seed, surrogate in zip(seeds, surrogates, strict=True):
        control = surrogate[N_HISTORY:]
        pixel_rows.append(
            {
                "image_index": image_index,
                "trace_index": trace_index,
                "seed": int(seed),
                "score40_mean_luminance_ratio": float(control.mean() / intact_score.mean()),
                "score40_second_moment_ratio": float(np.mean(control**2) / np.mean(intact_score**2)),
                "score40_global_variance_ratio": float(control.var() / intact_score.var()),
                "score40_framewise_spatial_variance_ratio": float(
                    np.mean(np.var(control, axis=(1, 2))) / np.mean(np.var(intact_score, axis=(1, 2)))
                ),
                "score40_pixelwise_temporal_variance_ratio": float(
                    np.mean(np.var(control, axis=0)) / np.mean(np.var(intact_score, axis=0))
                ),
            }
        )
    pixel = pd.DataFrame(pixel_rows)
    decomposition_path = OUT / "hann_decomposition_by_seed.csv"
    pixel_path = OUT / "scored_window_pixel_quantity_ratios.csv"
    figure_path = OUT / "hann_power_localization_decomposition.png"
    decomposition.to_csv(decomposition_path, index=False)
    pixel.to_csv(pixel_path, index=False)
    plot_decomposition(decomposition, pixel, figure_path)
    grouped = decomposition.groupby("condition").agg(
        total_ratio_min=("total_positive_tf_ratio", "min"),
        total_ratio_median=("total_positive_tf_ratio", "median"),
        total_ratio_max=("total_positive_tf_ratio", "max"),
        supported_ratio_min=("supported_sf_tf_ratio", "min"),
        supported_ratio_median=("supported_sf_tf_ratio", "median"),
        supported_ratio_max=("supported_sf_tf_ratio", "max"),
    ).reset_index()
    grouped_path = OUT / "hann_decomposition_summary.csv"
    grouped.to_csv(grouped_path, index=False)
    manifest = {
        "analysis": "rr100_global_3d_phase_hann_localization_decomposition",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "input_only_localization_audit_no_neural_scoring",
        "interpretation": (
            "the scored temporal crop creates a temporal-modulation mismatch; temporal Hann attenuates it; "
            "central spatial Hann substantially increases it; mean luminance, overall variance, second moment, "
            "and framewise spatial contrast remain near matched"
        ),
        "input": file_identity(arrays_path),
        "outputs": {
            "by_seed": file_identity(decomposition_path),
            "summary": file_identity(grouped_path),
            "pixel_quantities": file_identity(pixel_path),
            "figure": file_identity(figure_path),
        },
        "not_run": "No neural response, activation map, or SSI was computed.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(grouped.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
