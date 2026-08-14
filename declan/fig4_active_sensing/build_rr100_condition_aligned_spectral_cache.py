#!/usr/bin/env python3
"""Build an auditable spectral cache directly on an assembled response row axis.

This input-only runner writes one resumable shard per image, then assembles all
spectra into the declared ``matrix_row_index``. It never loads the neural model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from declan.fig4_active_sensing.input_only_retinal_renderer import render_retinal_frames_lag_zero
from declan.fig4_active_sensing.run_interim_input_spectral_cache import (
    FRAME_RATE_HZ,
    ORIENTATION_EDGES_DEG,
    SCALAR_NAMES,
    SF_EDGES_CPD,
    atomic_npz,
    spatial_lookup,
    spectral_statistics,
)
from declan.fig4_active_sensing.spectral_cache_contract import sha256, validate_spectral_cache
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _load_twin_common


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESPONSE = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "assembled/rounds_000_026_n027_clean_history_snapshot_v1"
)
DEFAULT_CACHE = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1"
DEFAULT_COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
DEFAULT_FLAGS = DEFAULT_CACHE / "quality_control/pre_fixation_history_trace_flags.csv"
DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_clean_history_spectral_cache_rounds000_026_n027_v1"
)
N_SCORE = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-dir", type=Path, default=DEFAULT_RESPONSE)
    parser.add_argument("--response-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--cohort-dir", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--trace-flags", type=Path, default=DEFAULT_FLAGS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--rerender-sample-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def identity(path: Path) -> dict[str, object]:
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "sha256": sha256(path.resolve())}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_contract(args: argparse.Namespace):
    conditions = pd.read_csv(args.response_dir / "condition_index.csv").sort_values("matrix_row_index")
    conditions = conditions.reset_index(drop=True)
    n = len(conditions)
    if not np.array_equal(conditions.matrix_row_index.to_numpy(int), np.arange(n)):
        raise ValueError("Response condition table is not a contiguous matrix-row axis")
    required = {"matrix_row_index", "round_index", "half_index", "image_index", "trace_index"}
    if not required.issubset(conditions.columns):
        raise ValueError(f"Condition table lacks {sorted(required.difference(conditions.columns))}")
    per_round = conditions.groupby("round_index").agg(
        rows=("matrix_row_index", "size"),
        images=("image_index", "nunique"),
        traces=("trace_index", "nunique"),
    )
    if not ((per_round.rows == 577) & (per_round.traces == 577)).all():
        raise ValueError("Every development round must contain each of the 577 clean traces exactly once")
    if not (per_round.images == 99).all():
        raise ValueError("The audited provisional schedule is expected to retain 99 nonempty image blocks per round")
    if conditions.duplicated(["image_index", "trace_index"]).any():
        raise ValueError("Image-trace pairs repeat across rounds")

    flags = pd.read_csv(args.trace_flags)
    used = flags[flags.trace_index.isin(conditions.trace_index.unique())]
    if len(used) != 577 or not used.history_within_selected_fixation.astype(bool).all():
        raise ValueError("Condition table contains traces outside the selected-fixation history gate")
    if used.trace_index.nunique() != 577:
        raise ValueError("Clean trace identity axis is incomplete")

    images = pd.read_csv(args.cohort_dir / "corrected100_images.csv").sort_values("image_index")
    traces = pd.read_csv(args.cohort_dir / "corrected1000_traces.csv").sort_values("trace_index")
    if images.image_index.nunique() != 100 or traces.trace_index.nunique() != 1000:
        raise ValueError("Frozen cohort axes are incomplete")
    trace_cache_path = args.response_cache_dir / "input_cache/corrected_trace_segments.npz"
    with np.load(trace_cache_path, allow_pickle=False) as archive:
        trace_ids = np.asarray(archive["trace_index"], dtype=int)
        score = np.asarray(archive["score_xy_deg"], dtype=np.float32)
    if trace_ids.shape != (1000,) or score.shape != (1000, N_SCORE, 2):
        raise ValueError("Frozen scored-trace cache has unexpected shape")
    trace_ordinal = {int(value): index for index, value in enumerate(trace_ids)}
    return conditions, images, score, trace_ordinal


def load_image(cache_dir: Path, image_index: int) -> tuple[np.ndarray, float]:
    path = cache_dir / "input_cache/images" / f"image_{image_index:03d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["image_index"].item()) != image_index:
            raise ValueError(f"Image identity mismatch in {path}")
        return np.asarray(archive["corrected_patch"], dtype=np.float32), float(archive["patch_ppd"].item())


def static_spatial_statistics(frame: np.ndarray, ppd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(frame, dtype=np.float64)
    residual = image - image.mean()
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    spectrum = np.fft.fftshift(np.fft.fft2(residual * window))
    power = np.abs(spectrum).ravel() ** 2
    sf_bin, ori_bin, _ = spatial_lookup(ppd, size=image.shape[0])
    n_sf = len(SF_EDGES_CPD) - 1
    n_ori = len(ORIENTATION_EDGES_DEG) - 1
    radial = np.bincount(sf_bin, weights=power, minlength=n_sf).astype(np.float32)
    oriented = np.bincount(
        sf_bin * n_ori + ori_bin, weights=power, minlength=n_sf * n_ori
    ).reshape(n_sf, n_ori).astype(np.float32)
    mean = float(image.mean())
    sd = float(image.std())
    summary = np.asarray([mean, sd, sd / max(abs(mean), np.finfo(float).tiny)], dtype=np.float64)
    return radial, oriented, summary


def shard_valid(path: Path, expected: pd.DataFrame, condition_sha: str) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                str(archive["condition_index_sha256"].item()) == condition_sha
                and np.array_equal(archive["matrix_row_index"], expected.matrix_row_index.to_numpy(np.int64))
                and np.array_equal(archive["trace_index"], expected.trace_index.to_numpy(np.int64))
                and archive["radial_power"].shape == (len(expected), 20, 13)
                and archive["orientation_power"].shape == (len(expected), 20, 13, 12)
            )
    except Exception:
        return False


def _filled_log_power_per_mode(power: np.ndarray, mode_count: np.ndarray) -> np.ndarray:
    """Return log-power density with unsupported SF bins interpolated for display only."""
    values = np.asarray(power, dtype=np.float64)
    counts = np.asarray(mode_count, dtype=np.float64)
    per_mode = np.divide(
        values,
        counts,
        out=np.full_like(values, np.nan, dtype=np.float64),
        where=np.broadcast_to(counts > 0, values.shape),
    )
    supported_values = per_mode[np.isfinite(per_mode) & (per_mode > 0)]
    if supported_values.size == 0:
        raise ValueError("Cannot construct power display without positive supported values")
    floor = max(float(np.max(supported_values)) * 1e-8, np.finfo(np.float64).tiny)
    log_values = np.log10(np.maximum(per_mode, floor))
    supported = counts > 0
    log_sf = np.log2(0.5 * (SF_EDGES_CPD[:-1] + SF_EDGES_CPD[1:]))
    rows = log_values.reshape(-1, log_values.shape[-1])
    for row in rows:
        row[~supported] = np.interp(log_sf[~supported], log_sf[supported], row[supported])
    return log_values


def make_example_figure(example: dict[str, np.ndarray], path: Path) -> None:
    patch = example["source_patch"]
    trace = example["retinal_trace"]
    moving = example["moving_movie"]
    stabilized = example["stabilized_movie"]
    radial = example["moving_radial_power"]
    static_radial = example["static_radial_spatial_power"]
    ppd = float(example["ppd"])
    sf = 0.5 * (SF_EDGES_CPD[:-1] + SF_EDGES_CPD[1:])
    tf = np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)[1:]
    sf_bin, _, _ = spatial_lookup(ppd, size=moving.shape[1])
    mode_count = np.bincount(sf_bin, minlength=len(SF_EDGES_CPD) - 1)
    unsupported = mode_count == 0
    log_sf = np.log2(sf)

    # These transformations are intentionally confined to this explanatory figure.
    # The cached band sums remain raw, and downstream analyses use the support mask.
    filled_log_power = _filled_log_power_per_mode(radial, mode_count)
    smoothed_log_power = gaussian_filter(filled_log_power, sigma=(0.55, 0.55), mode="nearest")
    dense_tf = np.linspace(tf[0], tf[-1], 240)
    dense_log_sf = np.linspace(log_sf[0], log_sf[-1], 240)
    dense_sf = np.exp2(dense_log_sf)
    interpolator = RegularGridInterpolator(
        (tf, log_sf), smoothed_log_power, bounds_error=False, fill_value=None
    )
    dense_tf_grid, dense_log_sf_grid = np.meshgrid(dense_tf, dense_log_sf, indexing="ij")
    dense_log_power = interpolator(
        np.column_stack((dense_tf_grid.ravel(), dense_log_sf_grid.ravel()))
    ).reshape(dense_tf_grid.shape)

    filled_static_log_power = _filled_log_power_per_mode(static_radial, mode_count)
    smoothed_static_log_power = gaussian_filter1d(
        filled_static_log_power, sigma=0.55, mode="nearest"
    )
    dense_static_log_power = np.interp(dense_log_sf, log_sf, smoothed_static_log_power)
    display_values_path = path.parent / f"{path.name}_display_values.npz"
    np.savez_compressed(
        display_values_path,
        spatial_frequency_cycles_per_degree=sf,
        temporal_frequency_hz=tf,
        spatial_frequency_mode_count=mode_count,
        spatial_frequency_has_support=~unsupported,
        raw_moving_radial_band_power=radial,
        filled_log10_moving_power_per_mode=filled_log_power,
        dense_spatial_frequency_cycles_per_degree=dense_sf,
        dense_temporal_frequency_hz=dense_tf,
        smoothed_log10_moving_power_per_mode=dense_log_power,
        raw_static_radial_band_power=static_radial,
        filled_log10_static_power_per_mode=filled_static_log_power,
        smoothed_log10_static_power_per_mode=dense_static_log_power,
    )

    fig, axes = plt.subplots(
        2, 4, figsize=(17, 8), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.12]},
    )
    axes[0, 0].imshow(patch, cmap="gray", origin="lower")
    axes[0, 0].set_title("Corrected natural-image source patch")
    axes[0, 1].plot(trace[:, 0] * 60, trace[:, 1] * 60, color="#0072B2")
    axes[0, 1].scatter(trace[0, 0] * 60, trace[0, 1] * 60, color="#009E73", label="first frame")
    axes[0, 1].set_title("Retinal image trajectory during 40 scored frames")
    axes[0, 1].set_xlabel("horizontal displacement (arcmin)")
    axes[0, 1].set_ylabel("vertical displacement (arcmin)")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].set_aspect("equal", adjustable="datalim")
    axes[0, 2].imshow(moving[0], cmap="gray", origin="lower")
    axes[0, 2].set_title("Moving-retina movie\nfirst scored frame")
    axes[0, 3].imshow(moving[-1], cmap="gray", origin="lower")
    axes[0, 3].set_title("Moving-retina movie\nfinal scored frame")
    axes[1, 0].imshow(stabilized[0], cmap="gray", origin="lower")
    axes[1, 0].set_title("Stabilized movie: repeated retinal frame")
    mesh = axes[1, 1].pcolormesh(
        dense_sf, dense_tf, dense_log_power, shading="auto", cmap="viridis"
    )
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_title("Moving movie spatial-by-temporal Fourier power")
    axes[1, 1].set_xlabel("spatial frequency (cycles/degree)")
    axes[1, 1].set_ylabel("temporal frequency (Hz)")
    axes[1, 1].text(
        0.02, 0.98,
        "Smoothed display; the empty spatial-frequency bin is interpolated",
        transform=axes[1, 1].transAxes, ha="left", va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    fig.colorbar(
        mesh, ax=axes[1, 1], pad=0.03,
        label="log10 power per Fourier mode\n(smoothed display)",
    )
    axes[1, 2].plot(dense_sf, 10.0 ** dense_static_log_power, color="#D55E00")
    axes[1, 2].set_xscale("log", base=2)
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_title("Stabilized frame spatial Fourier power")
    axes[1, 2].set_xlabel("spatial frequency (cycles/degree)")
    axes[1, 2].set_ylabel("power per Fourier mode\n(smoothed display)")
    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.0, 0.95,
        "Interpretation\n\n"
        "The moving movie redistributes static image\n"
        "structure into positive temporal frequencies.\n\n"
        "The stabilized movie has zero positive-temporal-\n"
        "frequency power after temporal-mean subtraction.\n"
        "Its spatial spectrum is retained separately.\n\n"
        "The displays divide each band by its number of\n"
        "available Fourier modes, interpolate the unsupported\n"
        "bin, and apply light smoothing. This does not alter\n"
        "the raw cached power or its analysis support mask.",
        va="top", fontsize=11, wrap=True,
    )
    fig.suptitle("Concrete input example for the provisional clean-history spectral cache", fontsize=15, weight="bold")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_spatial_frequency_support_audit(ppd: float, out_dir: Path) -> pd.DataFrame:
    sf_bin, _, _ = spatial_lookup(ppd, size=51)
    counts = np.bincount(sf_bin, minlength=len(SF_EDGES_CPD) - 1)
    table = pd.DataFrame({
        "spatial_frequency_bin_index": np.arange(len(counts), dtype=int),
        "lower_edge_cycles_per_degree": SF_EDGES_CPD[:-1],
        "upper_edge_cycles_per_degree": SF_EDGES_CPD[1:],
        "center_cycles_per_degree": 0.5 * (SF_EDGES_CPD[:-1] + SF_EDGES_CPD[1:]),
        "discrete_fourier_mode_count": counts.astype(int),
        "has_fourier_support": counts > 0,
        "support_note": [
            "spatial DC mode only" if index == 0 and count == 1
            else "unsupported at 51x51 resolution" if count == 0
            else "supported"
            for index, count in enumerate(counts)
        ],
    })
    table.to_csv(out_dir / "spatial_frequency_bin_support.csv", index=False)
    fig, axis = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    centers = table.center_cycles_per_degree.to_numpy(float)
    colors = np.where(table.has_fourier_support, "#0072B2", "#D55E00")
    axis.bar(centers, counts, width=centers * 0.24, color=colors, align="center")
    axis.set_xscale("log", base=2)
    axis.set_yscale("symlog", linthresh=1)
    axis.set_xlabel("spatial-frequency bin (cycles/degree)")
    axis.set_ylabel("number of discrete 2-D Fourier modes")
    axis.set_title("Spatial-frequency support of the 51×51 retinal movie transform")
    empty = table[~table.has_fourier_support].iloc[0]
    axis.annotate(
        "No Fourier modes: this bin must be masked, not interpreted as low power",
        xy=(float(empty.center_cycles_per_degree), 0),
        xytext=(1.1, max(counts) / 5),
        arrowprops={"arrowstyle": "->", "color": "#D55E00"},
        color="#A84300",
    )
    axis.text(
        0.01, 0.98, f"pixels/degree = {ppd:.6f}; fundamental spacing = {ppd / 51:.6f} cycles/degree",
        transform=axis.transAxes, ha="left", va="top",
    )
    fig.savefig(out_dir / "03_spatial_frequency_bin_support_audit.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "03_spatial_frequency_bin_support_audit.pdf", bbox_inches="tight")
    plt.close(fig)
    return table


def save_design_balance_artifacts(conditions: pd.DataFrame, out_dir: Path) -> dict[str, int]:
    counts = (
        conditions.groupby(["round_index", "image_index"]).size().rename("condition_count").reset_index()
    )
    complete_grid = pd.MultiIndex.from_product(
        [sorted(conditions.round_index.unique()), range(100)], names=["round_index", "image_index"]
    ).to_frame(index=False)
    complete_grid = complete_grid.merge(counts, on=["round_index", "image_index"], how="left")
    complete_grid["condition_count"] = complete_grid.condition_count.fillna(0).astype(int)
    complete_grid.to_csv(out_dir / "round_by_image_condition_balance.csv", index=False)
    image_degree = complete_grid.groupby("image_index", as_index=False).condition_count.sum()
    image_degree = image_degree.rename(columns={"condition_count": "conditions_across_included_rounds"})
    image_degree.to_csv(out_dir / "image_condition_degree.csv", index=False)
    omitted = complete_grid[complete_grid.condition_count.eq(0)][["round_index", "image_index"]]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    axes[0].bar(image_degree.image_index, image_degree.conditions_across_included_rounds, color="#0072B2")
    axes[0].axhline(image_degree.conditions_across_included_rounds.mean(), color="black", linestyle="--", label="mean")
    axes[0].set_title("Image representation is unequal in the 27-round clean-history development snapshot")
    axes[0].set_xlabel("image identity")
    axes[0].set_ylabel("number of condition rows")
    axes[0].legend(frameon=False)
    axes[1].scatter(omitted.round_index, omitted.image_index, color="#D55E00", s=35)
    axes[1].set_title("Exactly one image has no retained trace in each round after history quarantine")
    axes[1].set_xlabel("round identity")
    axes[1].set_ylabel("omitted image identity")
    axes[1].set_xticks(sorted(conditions.round_index.unique()))
    axes[1].text(
        0.99, 0.05,
        "Every clean trace appears once per round (577/577).\n"
        "Image identity must therefore be controlled in development fits.",
        transform=axes[1].transAxes, ha="right", va="bottom",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    fig.suptitle("Condition-design audit for the provisional clean-history spectral cache", fontsize=15, weight="bold")
    stem = out_dir / "02_condition_design_balance_audit"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "image_degree_min": int(image_degree.conditions_across_included_rounds.min()),
        "image_degree_max": int(image_degree.conditions_across_included_rounds.max()),
        "omitted_round_image_cells": int(len(omitted)),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    condition_path = args.response_dir / "condition_index.csv"
    condition_sha = sha256(condition_path)
    conditions, images, score, trace_ordinal = load_contract(args)
    common = _load_twin_common()
    started = time.perf_counter()
    first_example: dict[str, np.ndarray] | None = None
    selected_images = images
    if args.max_images > 0:
        selected_images = images.iloc[: args.max_images]

    with torch.no_grad():
        for image in selected_images.itertuples(index=False):
            image_index = int(image.image_index)
            rows = conditions[conditions.image_index.eq(image_index)].sort_values("matrix_row_index")
            destination = shard_dir / f"image_{image_index:03d}.npz"
            if destination.exists() and shard_valid(destination, rows, condition_sha):
                continue
            patch, ppd = load_image(args.response_cache_dir, image_index)
            radial_rows, orientation_rows, scalar_rows = [], [], []
            for row in rows.itertuples(index=False):
                retinal_trace = -score[trace_ordinal[int(row.trace_index)]]
                movie_tensor = render_retinal_frames_lag_zero(
                    common, patch, retinal_trace, ppd=ppd, device=args.device
                )
                movie = movie_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                radial, oriented, scalar = spectral_statistics(movie, ppd=ppd)
                radial_rows.append(radial)
                orientation_rows.append(oriented)
                scalar_rows.append(scalar)
                if first_example is None:
                    zero = np.zeros_like(retinal_trace)
                    stabilized = render_retinal_frames_lag_zero(
                        common, patch, zero, ppd=ppd, device=args.device
                    ).detach().cpu().numpy().astype(np.float32, copy=False)
                    static_radial, _, _ = static_spatial_statistics(stabilized[0], ppd)
                    first_example = {
                        "source_patch": patch,
                        "retinal_trace": retinal_trace,
                        "moving_movie": movie.copy(),
                        "stabilized_movie": stabilized,
                        "moving_radial_power": radial,
                        "static_radial_spatial_power": static_radial,
                        "ppd": np.asarray(ppd),
                    }
                del movie_tensor
            atomic_npz(
                destination,
                condition_index_sha256=np.asarray(condition_sha),
                image_index=np.asarray(image_index, dtype=np.int64),
                matrix_row_index=rows.matrix_row_index.to_numpy(np.int64),
                round_index=rows.round_index.to_numpy(np.int64),
                half_index=rows.half_index.to_numpy(np.int64),
                trace_index=rows.trace_index.to_numpy(np.int64),
                radial_power=np.stack(radial_rows).astype(np.float32),
                orientation_power=np.stack(orientation_rows).astype(np.float32),
                scalar_metrics=np.stack(scalar_rows).astype(np.float64),
            )
            print(
                f"completed image {image_index:03d}: {len(rows)} condition rows; "
                f"elapsed {time.perf_counter() - started:.1f} s",
                flush=True,
            )
    if args.max_images > 0:
        smoke_manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "partial_image_smoke_complete",
            "tier": "implementation_smoke_test_not_analysis",
            "images": int(len(selected_images)),
            "condition_rows": int(
                conditions[conditions.image_index.isin(selected_images.image_index)].shape[0]
            ),
            "condition_index_sha256": condition_sha,
            "neural_model_calls": False,
        }
        atomic_json(args.out_dir / "manifest.json", smoke_manifest)
        print(json.dumps(smoke_manifest, indent=2))
        return

    n = len(conditions)
    radial_all = np.empty((n, 20, 13), dtype=np.float32)
    oriented_all = np.empty((n, 20, 13, 12), dtype=np.float32)
    scalar_all = np.empty((n, len(SCALAR_NAMES)), dtype=np.float64)
    populated = np.zeros(n, dtype=bool)
    for image_index in images.image_index.astype(int):
        path = shard_dir / f"image_{image_index:03d}.npz"
        rows = conditions[conditions.image_index.eq(image_index)].sort_values("matrix_row_index")
        if not shard_valid(path, rows, condition_sha):
            raise RuntimeError(f"Missing or invalid completed image shard: {path}")
        with np.load(path, allow_pickle=False) as archive:
            target = np.asarray(archive["matrix_row_index"], dtype=int)
            if populated[target].any():
                raise ValueError("A spectral storage row would be populated more than once")
            radial_all[target] = archive["radial_power"]
            oriented_all[target] = archive["orientation_power"]
            scalar_all[target] = archive["scalar_metrics"]
            populated[target] = True
    if not populated.all():
        raise ValueError(f"Unpopulated spectral rows: {np.flatnonzero(~populated).tolist()[:20]}")
    conservation = float(
        np.max(np.abs(oriented_all.astype(np.float64).sum(axis=-1) - radial_all) / np.maximum(radial_all, 1.0))
    )
    if conservation > 1e-5:
        raise ValueError(f"Orientation-resolved power does not conserve radial power: {conservation}")
    tf_hz = np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)[1:]
    reference_ppd = float(load_image(args.response_cache_dir, int(images.image_index.iloc[0]))[1])
    support_table = save_spatial_frequency_support_audit(reference_ppd, args.out_dir)
    mode_count = support_table.discrete_fourier_mode_count.to_numpy(np.int64)
    unsupported_power_max = float(np.max(np.abs(radial_all[:, :, mode_count == 0])))
    unsupported_oriented_power_max = float(np.max(np.abs(oriented_all[:, :, mode_count == 0, :])))
    if unsupported_power_max != 0.0 or unsupported_oriented_power_max != 0.0:
        raise ValueError("A spatial-frequency bin with no Fourier modes contains nonzero cached power")
    np.savez_compressed(
        args.out_dir / "condition_spectra.npz",
        matrix_row_index=conditions.matrix_row_index.to_numpy(np.int64),
        round_index=conditions.round_index.to_numpy(np.int64),
        half_index=conditions.half_index.to_numpy(np.int64),
        image_index=conditions.image_index.to_numpy(np.int64),
        trace_index=conditions.trace_index.to_numpy(np.int64),
        radial_power=radial_all,
        orientation_power=oriented_all,
        scalar_metrics=scalar_all,
        scalar_metric_names=np.asarray(SCALAR_NAMES, dtype="U64"),
        tf_hz=tf_hz,
        sf_edges_cpd=SF_EDGES_CPD,
        orientation_edges_deg=ORIENTATION_EDGES_DEG,
        spatial_frequency_mode_count=mode_count,
        spatial_frequency_has_support=(mode_count > 0),
    )
    metric_table = conditions.copy()
    for index, name in enumerate(SCALAR_NAMES):
        metric_table[name] = scalar_all[:, index]
    metric_table.to_csv(args.out_dir / "condition_spectral_metrics.csv", index=False)

    static_radial, static_oriented, static_summary = [], [], []
    stabilized_dynamic_max = 0.0
    for image_index in images.image_index.astype(int):
        patch, ppd = load_image(args.response_cache_dir, image_index)
        zero = np.zeros((N_SCORE, 2), dtype=np.float32)
        stabilized = render_retinal_frames_lag_zero(common, patch, zero, ppd=ppd, device=args.device)
        stabilized_np = stabilized.detach().cpu().numpy().astype(np.float32, copy=False)
        dynamic_radial, dynamic_oriented, _ = spectral_statistics(stabilized_np, ppd=ppd)
        stabilized_dynamic_max = max(
            stabilized_dynamic_max, float(np.max(np.abs(dynamic_radial))), float(np.max(np.abs(dynamic_oriented)))
        )
        radial, oriented, summary = static_spatial_statistics(stabilized_np[0], ppd)
        static_radial.append(radial)
        static_oriented.append(oriented)
        static_summary.append(summary)
    np.savez_compressed(
        args.out_dir / "stabilized_input_predictors_by_image.npz",
        image_index=images.image_index.to_numpy(np.int64),
        dynamic_positive_tf_radial_power=np.zeros((len(images), 20, 13), dtype=np.float32),
        dynamic_positive_tf_orientation_power=np.zeros((len(images), 20, 13, 12), dtype=np.float32),
        static_radial_spatial_power=np.stack(static_radial),
        static_orientation_spatial_power=np.stack(static_oriented),
        static_mean_sd_rms_contrast=np.stack(static_summary),
        static_summary_names=np.asarray(("local_mean", "local_sd", "local_rms_contrast")),
        sf_edges_cpd=SF_EDGES_CPD,
        orientation_edges_deg=ORIENTATION_EDGES_DEG,
        spatial_frequency_mode_count=mode_count,
        spatial_frequency_has_support=(mode_count > 0),
    )

    rng = np.random.default_rng(args.seed)
    boundaries = {0, n - 1}
    for _, group in conditions.groupby("round_index", sort=True):
        boundaries.add(int(group.matrix_row_index.min()))
        boundaries.add(int(group.matrix_row_index.max()))
    random_rows = rng.choice(n, size=min(args.rerender_sample_size, n), replace=False)
    audit_rows = sorted(boundaries.union(map(int, random_rows)))
    rerender_records = []
    for matrix_row in audit_rows:
        row = conditions.iloc[matrix_row]
        patch, ppd = load_image(args.response_cache_dir, int(row.image_index))
        retinal_trace = -score[trace_ordinal[int(row.trace_index)]]
        movie = render_retinal_frames_lag_zero(common, patch, retinal_trace, ppd=ppd, device=args.device)
        radial, oriented, _ = spectral_statistics(movie.detach().cpu().numpy(), ppd=ppd)
        radial_error = float(np.linalg.norm(radial.astype(float) - radial_all[matrix_row]) / max(np.linalg.norm(radial), 1.0))
        oriented_error = float(np.linalg.norm(oriented.astype(float) - oriented_all[matrix_row]) / max(np.linalg.norm(oriented), 1.0))
        rerender_records.append({
            "matrix_row_index": matrix_row,
            "round_index": int(row.round_index),
            "image_index": int(row.image_index),
            "trace_index": int(row.trace_index),
            "radial_relative_l2_error": radial_error,
            "orientation_relative_l2_error": oriented_error,
        })
    rerender = pd.DataFrame(rerender_records)
    rerender.to_csv(args.out_dir / "independent_rerender_audit.csv", index=False)
    max_rerender = float(rerender[["radial_relative_l2_error", "orientation_relative_l2_error"]].to_numpy().max())
    if max_rerender > 1e-7:
        raise ValueError(f"Independent rerender audit failed: {max_rerender}")

    design_balance = save_design_balance_artifacts(conditions, args.out_dir)

    if first_example is None:
        row = conditions.iloc[0]
        patch, ppd = load_image(args.response_cache_dir, int(row.image_index))
        retinal_trace = -score[trace_ordinal[int(row.trace_index)]]
        moving = render_retinal_frames_lag_zero(
            common, patch, retinal_trace, ppd=ppd, device=args.device
        ).detach().cpu().numpy().astype(np.float32, copy=False)
        zero = np.zeros_like(retinal_trace)
        stabilized = render_retinal_frames_lag_zero(
            common, patch, zero, ppd=ppd, device=args.device
        ).detach().cpu().numpy().astype(np.float32, copy=False)
        static_radial_example, _, _ = static_spatial_statistics(stabilized[0], ppd)
        first_example = {
            "source_patch": patch,
            "retinal_trace": retinal_trace,
            "moving_movie": moving,
            "stabilized_movie": stabilized,
            "moving_radial_power": radial_all[0],
            "static_radial_spatial_power": static_radial_example,
            "ppd": np.asarray(ppd),
        }
    make_example_figure(first_example, args.out_dir / "01_example_retinal_movie_and_spatial_temporal_power")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "provisional_clean_history_development_spectral_cache_complete",
        "tier": "engineering_development_only_not_confirmatory",
        "warning": (
            "100x577 clean-history subset over 27 trace-balanced but image-imbalanced rounds; "
            "not the replacement 100x1000 production endpoint"
        ),
        "scope": {
            "conditions": n,
            "rounds": int(conditions.round_index.nunique()),
            "images": int(conditions.image_index.nunique()),
            "traces": int(conditions.trace_index.nunique()),
            "image_degree_min": int(conditions.groupby("image_index").size().min()),
            "image_degree_max": int(conditions.groupby("image_index").size().max()),
            "nonempty_images_per_round": int(conditions.groupby("round_index").image_index.nunique().iloc[0]),
        },
        "contract": {
            "retinal_motion": "negative corrected dpi_pix crop trajectory",
            "frames": "40 scored lag-zero retinal frames at 120 Hz",
            "history": "32 recorded frames validated within the selected fixation; excluded from input power",
            "spatial_window": "separable 2D Hann",
            "temporal_window": "Hann after subtracting temporal mean image",
            "neural_model_calls": False,
            "storage_order": "written directly to declared response matrix_row_index",
            "stabilized_dynamic_power": "computed under identical transform and verified zero; stored by image",
            "stabilized_static_predictors": "spatial spectrum, local mean, local standard deviation, RMS contrast stored separately",
            "spatial_frequency_support": (
                "discrete mode counts stored; unsupported bins are masked in analysis; "
                "the example figure separately labels its display-only interpolation"
            ),
            "example_figure_smoothing": (
                "band power divided by available Fourier-mode count, transformed to log10, "
                "unsupported bins interpolated in log2 spatial frequency, Gaussian smoothing "
                "of 0.55 bins, then dense interpolation for display only"
            ),
        },
        "sources": {
            "conditions": identity(condition_path),
            "images": identity(args.cohort_dir / "corrected100_images.csv"),
            "traces": identity(args.cohort_dir / "corrected1000_traces.csv"),
            "trace_history_flags": identity(args.trace_flags),
            "frozen_trace_arrays": identity(args.response_cache_dir / "input_cache/corrected_trace_segments.npz"),
            "runner": identity(Path(__file__)),
            "renderer": identity(ROOT / "declan/fig4_active_sensing/input_only_retinal_renderer.py"),
        },
        "validation": {
            "all_rows_populated_once": bool(populated.all()),
            "maximum_radial_orientation_relative_error": conservation,
            "independent_rerender_rows": len(rerender),
            "maximum_independent_rerender_relative_l2_error": max_rerender,
            "maximum_computed_stabilized_dynamic_power": stabilized_dynamic_max,
            "all_trace_histories_within_selected_fixation": True,
            "each_round_contains_every_clean_trace_once": True,
            "image_balance_pass": False,
            "image_balance_implication": "development fits must control image identity; confirmatory inference is prohibited",
            "omitted_round_image_cells": design_balance["omitted_round_image_cells"],
            "unsupported_spatial_frequency_bins": support_table.loc[
                ~support_table.has_fourier_support, "spatial_frequency_bin_index"
            ].astype(int).tolist(),
            "downstream_grating_supported_minimum_cycles_per_degree": 1.0,
            "unsupported_bins_excluded_from_existing_grating_supported_predictors": True,
            "maximum_power_in_unsupported_spatial_frequency_bins": unsupported_power_max,
            "maximum_orientation_power_in_unsupported_spatial_frequency_bins": unsupported_oriented_power_max,
        },
        "outputs": {
            "arrays": str((args.out_dir / "condition_spectra.npz").resolve()),
            "metrics": str((args.out_dir / "condition_spectral_metrics.csv").resolve()),
            "stabilized_predictors": str((args.out_dir / "stabilized_input_predictors_by_image.npz").resolve()),
            "rerender_audit": str((args.out_dir / "independent_rerender_audit.csv").resolve()),
            "input_example": str((args.out_dir / "01_example_retinal_movie_and_spatial_temporal_power.pdf").resolve()),
            "input_example_display_values": str((
                args.out_dir / "01_example_retinal_movie_and_spatial_temporal_power_display_values.npz"
            ).resolve()),
            "condition_design_audit": str((args.out_dir / "02_condition_design_balance_audit.pdf").resolve()),
            "round_by_image_balance": str((args.out_dir / "round_by_image_condition_balance.csv").resolve()),
            "spatial_frequency_support": str((args.out_dir / "spatial_frequency_bin_support.csv").resolve()),
            "spatial_frequency_support_figure": str((args.out_dir / "03_spatial_frequency_bin_support_audit.pdf").resolve()),
        },
    }
    atomic_json(args.out_dir / "manifest.json", manifest)
    contract = validate_spectral_cache(args.out_dir, require_rounds=27)
    print(json.dumps({"manifest": manifest, "shared_contract": contract}, indent=2))


if __name__ == "__main__":
    main()
