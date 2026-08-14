#!/usr/bin/env python3
"""Calibrate local-metamer pooling scales to digital-twin spatial readouts.

This is an input-design checkpoint, not a neural surrogate test.  It extracts
the learned 14 x 14 Gaussian spatial mask for every canonical unit, identifies
the fixed RR100 movie-medoid units, and reports mask-derived spatial scales in
core-grid cells, stimulus pixels, degrees, and arcminutes.

The reported stimulus conversion uses the measured two-pixel spatial jump of
the twin core.  These are *readout pooling* scales, not full input-space
effective receptive fields: the learned convolutional and recurrent core adds
spatial support.  The distinction is explicit in every output artifact so this
table cannot silently be used as a claim about the composite model RF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from declan.redundancy_resolved_v1_population import (  # noqa: E402
    load_canonical_twin_bundle,
    load_population_view,
)


RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_spatial_filter_pooling_scale_checkpoint_41_v1"
PPD = 37.50476617
MASK_SIZE = 14
FRACTIONS = (0.50, 0.80, 0.90, 0.95)
EPS = np.finfo(np.float64).tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rr100-version", type=str, default=RR100_VERSION)
    parser.add_argument("--ppd", type=float, default=PPD)
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


def mask_metrics(mask: np.ndarray) -> dict[str, float]:
    """Return mass- and energy-based geometry of one realized readout mask."""
    weight = np.asarray(mask, dtype=np.float64)
    weight = np.maximum(weight, 0.0)
    weight /= max(float(weight.sum()), EPS)
    yy, xx = np.mgrid[: weight.shape[0], : weight.shape[1]]
    cy = float(np.sum(weight * yy))
    cx = float(np.sum(weight * xx))
    dy = yy - cy
    dx = xx - cx
    covariance = np.array(
        [
            [np.sum(weight * dy * dy), np.sum(weight * dy * dx)],
            [np.sum(weight * dy * dx), np.sum(weight * dx * dx)],
        ],
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    sigma_major, sigma_minor = np.sqrt(eigenvalues)
    major_y, major_x = eigenvectors[:, 0]
    angle_deg = float(np.degrees(np.arctan2(major_y, major_x)))
    radius = np.hypot(dy, dx)

    energy = weight**2
    energy /= max(float(energy.sum()), EPS)
    flat_order = np.argsort(radius.ravel())
    sorted_radius = radius.ravel()[flat_order]
    mass_cdf = np.cumsum(weight.ravel()[flat_order])
    energy_cdf = np.cumsum(energy.ravel()[flat_order])
    out: dict[str, float] = {
        "mask_center_y_core": cy,
        "mask_center_x_core": cx,
        "mask_sigma_major_core": float(sigma_major),
        "mask_sigma_minor_core": float(sigma_minor),
        "mask_sigma_geomean_core": float(np.sqrt(max(sigma_major * sigma_minor, 0.0))),
        "mask_axis_ratio": float(sigma_major / max(sigma_minor, EPS)),
        "mask_principal_angle_deg": angle_deg,
        "mask_edge_mass": float(
            weight[0, :].sum()
            + weight[-1, :].sum()
            + weight[1:-1, 0].sum()
            + weight[1:-1, -1].sum()
        ),
    }
    for fraction in FRACTIONS:
        tag = str(int(round(100 * fraction)))
        mass_idx = min(int(np.searchsorted(mass_cdf, fraction)), sorted_radius.size - 1)
        energy_idx = min(int(np.searchsorted(energy_cdf, fraction)), sorted_radius.size - 1)
        out[f"mask_mass_radius_{tag}_core"] = float(sorted_radius[mass_idx])
        out[f"mask_energy_radius_{tag}_core"] = float(sorted_radius[energy_idx])
    return out


def add_physical_units(row: dict[str, Any], *, jump_px: float, ppd: float) -> None:
    core_scale_keys = [
        key
        for key in list(row)
        if key.startswith("mask_")
        and key.endswith("_core")
        and ("sigma" in key or "radius" in key)
    ]
    for key in core_scale_keys:
        stem = key[: -len("_core")]
        px = float(row[key]) * float(jump_px)
        row[f"{stem}_stimulus_px"] = px
        row[f"{stem}_deg"] = px / float(ppd)
        row[f"{stem}_arcmin"] = 60.0 * px / float(ppd)


def measure_core_jump(model: Any, *, device: str) -> tuple[float, dict[str, int]]:
    """Measure the input-pixel jump between adjacent core-map positions."""
    sizes = (51, 53)
    outputs: list[int] = []
    with torch.inference_mode():
        for size in sizes:
            x = torch.zeros((1, 1, 32, size, size), dtype=torch.float32, device=device)
            y = model.model.core_forward(x, None)
            outputs.append(int(y.shape[-1]))
    output_delta = outputs[1] - outputs[0]
    if output_delta <= 0:
        raise RuntimeError(f"Could not infer core jump from input/output sizes: {sizes} -> {outputs}")
    jump = float((sizes[1] - sizes[0]) / output_delta)
    return jump, {
        "probe_input_size_1": sizes[0],
        "probe_input_size_2": sizes[1],
        "probe_core_size_1": outputs[0],
        "probe_core_size_2": outputs[1],
    }


def quantile_summary(frame: pd.DataFrame, population: str) -> list[dict[str, Any]]:
    metrics = [
        "raw_std_y_core",
        "raw_std_x_core",
        "mask_sigma_major_stimulus_px",
        "mask_sigma_minor_stimulus_px",
        "mask_sigma_geomean_stimulus_px",
        "mask_mass_radius_90_stimulus_px",
        "mask_energy_radius_90_stimulus_px",
        "mask_axis_ratio",
        "mask_edge_mass",
    ]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = frame[metric].to_numpy(np.float64)
        for quantile in (0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0):
            rows.append(
                {
                    "population": population,
                    "n_units": int(values.size),
                    "metric": metric,
                    "quantile": quantile,
                    "value": float(np.quantile(values, quantile)),
                }
            )
    return rows


def plot_checkpoint(frame: pd.DataFrame, out_path: Path) -> list[int]:
    full = frame.copy()
    rr = frame.loc[frame.is_rr100].copy().sort_values("rr100_index")
    metric = "mask_sigma_geomean_stimulus_px"
    example_quantiles = (0.05, 0.25, 0.50, 0.75, 0.95)
    example_rows = []
    used: set[int] = set()
    for quantile in example_quantiles:
        target = float(np.quantile(rr[metric], quantile))
        candidates = rr.loc[~rr.canonical_channel.isin(used)].copy()
        index = (candidates[metric] - target).abs().idxmin()
        example = rr.loc[index]
        example_rows.append(example)
        used.add(int(example.canonical_channel))

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 5, height_ratios=(1.35, 1.0))
    ax_hist = fig.add_subplot(grid[0, :3])
    ax_cdf = fig.add_subplot(grid[0, 3:])
    bins = np.linspace(
        0,
        float(max(full[metric].max(), rr[metric].max())) * 1.04,
        32,
    )
    ax_hist.hist(full[metric], bins=bins, density=True, alpha=0.45, color="0.55", label="canonical 756")
    ax_hist.hist(rr[metric], bins=bins, density=True, histtype="step", linewidth=2.5, color="#b2182b", label="RR100 medoids")
    for quantile, linestyle in ((0.05, ":"), (0.50, "-"), (0.95, ":")):
        ax_hist.axvline(np.quantile(rr[metric], quantile), color="#b2182b", linestyle=linestyle, alpha=0.85)
    ax_hist.set(
        xlabel="realized readout-mask geometric-mean sigma (stimulus pixels)",
        ylabel="density",
        title="Learned spatial pooling-width distribution",
    )
    ax_hist.legend(frameon=False)

    for data, color, label in ((full, "0.45", "canonical 756"), (rr, "#b2182b", "RR100 medoids")):
        values = np.sort(data[metric].to_numpy(np.float64))
        ax_cdf.plot(values, np.arange(1, values.size + 1) / values.size, color=color, linewidth=2, label=label)
    ax_cdf.set(xlabel="geometric-mean sigma (stimulus pixels)", ylabel="cumulative fraction", title="Distribution used for scale matching", ylim=(0, 1.01))
    ax_cdf.grid(alpha=0.2)

    for column, (quantile, row) in enumerate(zip(example_quantiles, example_rows)):
        ax = fig.add_subplot(grid[1, column])
        mask = np.asarray(row["_mask"], dtype=np.float64)
        ax.imshow(mask, cmap="magma", interpolation="nearest")
        ax.set_title(
            f"RR100 {int(row.rr100_index):02d}; q{int(100*quantile):02d}\n"
            f"sigma={row[metric]:.2f} px; ratio={row.mask_axis_ratio:.2f}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Digital-twin spatial-filter checkpoint: match a distribution, not one pooling radius\n"
        "Masks are learned 14x14 readouts; stimulus conversion uses measured 2-pixel core jump",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return [int(row.canonical_channel) for row in example_rows]


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_canonical_twin_bundle(device=str(args.device), mode="standard")
    view = load_population_view(version_name=str(args.rr100_version))
    if view.membership is None or int(view.n_units) != 100:
        raise ValueError(f"Expected one-hot RR100 population, got n_units={view.n_units}")
    rr_channels = np.argmax(view.membership, axis=1).astype(int)
    if not np.allclose(view.membership[np.arange(view.n_units), rr_channels], 1.0):
        raise ValueError("RR100 population is not a one-hot medoid view")
    if np.unique(rr_channels).size != rr_channels.size:
        raise ValueError("RR100 medoid channels are not unique")
    rr_lookup = {int(channel): int(index) for index, channel in enumerate(rr_channels)}

    jump_px, jump_probe = measure_core_jump(bundle.model, device=str(args.device))
    rows: list[dict[str, Any]] = []
    for canonical_channel, unit in enumerate(bundle.unit_rows):
        readout_index = int(unit["model_readout_index"])
        source_unit_index = int(unit["source_unit_index"])
        readout = bundle.model.model.readouts[readout_index]
        with torch.inference_mode():
            mask = (
                readout.compute_gaussian_mask(MASK_SIZE, MASK_SIZE, torch.device(args.device))[source_unit_index]
                .detach()
                .cpu()
                .numpy()
            )
        raw_std = readout.std[source_unit_index].detach().cpu().numpy()
        raw_mean = readout.mean[source_unit_index].detach().cpu().numpy()
        raw_theta = float(readout.theta[source_unit_index].detach().cpu())
        row: dict[str, Any] = {
            "canonical_channel": int(canonical_channel),
            "is_rr100": int(canonical_channel) in rr_lookup,
            "rr100_index": rr_lookup.get(int(canonical_channel), -1),
            "session": str(unit["session"]),
            "source_unit_index": source_unit_index,
            "ccnorm": float(unit["ccnorm"]),
            "model_readout_index": readout_index,
            "raw_mean_y_core": float(raw_mean[0]),
            "raw_mean_x_core": float(raw_mean[1]),
            "raw_std_y_core": float(raw_std[0]),
            "raw_std_x_core": float(raw_std[1]),
            "raw_theta_rad": raw_theta,
            "raw_theta_deg": float(np.degrees(raw_theta)),
            "core_spatial_jump_stimulus_px": jump_px,
            "scope": "learned_readout_mask_only_not_composite_input_effective_rf",
            "_mask": mask,
        }
        row.update(mask_metrics(mask))
        add_physical_units(row, jump_px=jump_px, ppd=float(args.ppd))
        rows.append(row)

    frame = pd.DataFrame(rows)
    rr_frame = frame.loc[frame.is_rr100].copy()
    if len(frame) != 756 or len(rr_frame) != 100:
        raise ValueError(f"Unexpected population sizes: canonical={len(frame)}, RR100={len(rr_frame)}")

    figure_path = out_dir / "rr100_spatial_filter_pooling_scale_checkpoint.png"
    example_channels = plot_checkpoint(frame, figure_path)
    table = frame.drop(columns=["_mask"]).sort_values("canonical_channel")
    table_path = out_dir / "canonical_spatial_readout_scales.csv"
    table.to_csv(table_path, index=False)
    summary = pd.DataFrame(
        quantile_summary(frame, "canonical_756")
        + quantile_summary(rr_frame, "rr100_movie_medoids")
    )
    summary_path = out_dir / "spatial_readout_scale_quantiles.csv"
    summary.to_csv(summary_path, index=False)

    rr_sigma = rr_frame["mask_sigma_geomean_stimulus_px"].to_numpy(np.float64)
    manifest = {
        "analysis": "digital_twin_spatial_readout_pooling_scale_checkpoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_mode": "standard",
        "model_checkpoint": str(getattr(bundle.model, "checkpoint_path", "self_reported_by_loader")),
        "rr100_version": str(args.rr100_version),
        "n_canonical_units": int(len(frame)),
        "n_rr100_units": int(len(rr_frame)),
        "mask_size_core_cells": MASK_SIZE,
        "stimulus_ppd": float(args.ppd),
        "measured_core_spatial_jump_stimulus_px": jump_px,
        "core_jump_probe": jump_probe,
        "rr100_geomean_sigma_stimulus_px_quantiles": {
            str(q): float(np.quantile(rr_sigma, q)) for q in (0.05, 0.25, 0.50, 0.75, 0.95)
        },
        "recommended_surrogate_parameterization": (
            "sample paired major/minor Gaussian sigmas from RR100 mask-derived rows; "
            "do not substitute a single median scalar radius"
        ),
        "critical_scope_limit": (
            "These widths describe the learned Gaussian readout masks after the twin core. "
            "They exclude spatial support added by the learned convolutional and recurrent core, "
            "so they are not yet composite input-space effective receptive-field radii."
        ),
        "example_canonical_channels": example_channels,
        "outputs": {
            "unit_table": file_identity(table_path),
            "quantiles": file_identity(summary_path),
            "figure": file_identity(figure_path),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    readme = out_dir / "README.md"
    readme.write_text(
        "# RR100 spatial-filter pooling-scale checkpoint\n\n"
        "This checkpoint turns the learned digital-twin spatial readouts into an auditable "
        "distribution of candidate local-metamer pooling scales. The RR100 control should "
        "sample paired major/minor widths from the RR100 rows; it should not use one arbitrary "
        "pooling radius.\n\n"
        "The figure and table concern the learned 14 x 14 Gaussian readout masks. They are a "
        "necessary model-derived calibration, but not the final input-space RF measurement. "
        "The shared convolutional and recurrent core adds support. Before surrogate synthesis, "
        "the next map-first checkpoint should backpropagate representative RR100 outputs to "
        "movie pixels and compare composite enclosed-gradient radii with these readout-only "
        "values.\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
