#!/usr/bin/env python3
"""Cartographer Stage 3: targeted spatial-coordinate and aperture checkpoint."""
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
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import torch

from declan.fig4_active_sensing.make_rr100_recorded_grating_three_way_response_checkpoint import (
    backproject_readout_footprint,
)
from declan.fig4_active_sensing.run_rr100_direct_fem_orientation_checkpoint import (
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.redundancy_resolved_v1_population import (
    load_canonical_twin_bundle,
    load_population_view,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TUNING = ROOT / "outputs/fig4_active_sensing/rr100_grating_only_orientation_tuning_v1"
DEFAULT_ASSIGNMENTS = ROOT / (
    "outputs/fig4_active_sensing/backimage_real_trace_sf_halves_recorded_validated_r0p5_v1/"
    "sf_half_recorded_validated_unit_assignments.csv"
)
DEFAULT_READOUT = ROOT / (
    "outputs/fig4_active_sensing/rr100_spatial_filter_pooling_scale_checkpoint_41_v1/"
    "canonical_spatial_readout_scales.csv"
)
DEFAULT_OUT = ROOT / "outputs/fig4_active_sensing/rr100_spatial_coordinate_contract_stage3_v1"

N_HISTORY = 32
NATIVE_SIZE = 51
LARGE_SIZE = 151
MAP_SIZE = 51
FRAME_RATE_HZ = 120.0
PPD = 37.50476617
PROBE_SIGMA_PX = 7.0
CONTRAST = 0.35
PHASE_RAD = math.pi / 7.0
TRANSLATIONS_PX = (
    (0, 0),
    (0, 2), (0, -2), (2, 0), (-2, 0),
    (0, 8), (0, -8), (8, 0), (-8, 0),
    (8, 8),
    (0, 40), (0, -40), (40, 0), (-40, 0),
)
EPS = np.finfo(np.float64).tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning-dir", type=Path, default=DEFAULT_TUNING)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--readout-audit", type=Path, default=DEFAULT_READOUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": digest.hexdigest()}


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


def orientation_strength(tuning_dir: Path) -> pd.DataFrame:
    with np.load(tuning_dir / "grating_only_orientation_tuning.npz", allow_pickle=False) as archive:
        units = np.asarray(archive["rr100_index"], dtype=int)
        orientations = np.asarray(archive["measured_grating_orientation_deg"], dtype=float)
        positive = np.asarray(archive["measured_positive_f0_hz"], dtype=float)
    marginal = positive.mean(axis=(1, 2))
    vector = np.exp(2j * np.deg2rad(orientations))
    strength = np.abs(marginal @ vector) / np.maximum(marginal.sum(axis=1), 1e-12)
    return pd.DataFrame({"rr100_index": units, "grating_orientation_vector_strength": strength})


def select_units(assignments_path: Path, readout_path: Path, tuning_dir: Path) -> pd.DataFrame:
    assignments = pd.read_csv(assignments_path)
    readout = pd.read_csv(readout_path)
    readout = readout.loc[readout.is_rr100.astype(bool)].copy()
    eligible = (
        assignments.loc[assignments.recorded_validation_pass.astype(bool)]
        .merge(
            readout[
                [
                    "rr100_index", "canonical_channel", "mask_energy_radius_90_stimulus_px",
                    "mask_sigma_major_stimulus_px", "mask_sigma_minor_stimulus_px",
                    "mask_axis_ratio", "mask_principal_angle_deg",
                ]
            ],
            on="rr100_index",
            validate="one_to_one",
        )
        .merge(orientation_strength(tuning_dir), on="rr100_index", validate="one_to_one")
        .sort_values("rr100_index")
    )
    selected: list[pd.Series] = []
    used: set[int] = set()

    def add(role: str, metric: str, target: float, criterion: str) -> None:
        available = eligible.loc[~eligible.rr100_index.astype(int).isin(used)].copy()
        index = (available[metric].astype(float) - float(target)).abs().idxmin()
        row = eligible.loc[index].copy()
        row["selection_role"] = role
        row["selection_criterion"] = criterion
        row["selection_metric"] = metric
        row["selection_target_value"] = float(target)
        row["selection_value"] = float(row[metric])
        row["selection_is_algorithmic_and_pre_response"] = True
        selected.append(row)
        used.add(int(row.rr100_index))

    radius = eligible.mask_energy_radius_90_stimulus_px.astype(float)
    add("small learned-readout support", "mask_energy_radius_90_stimulus_px", float(radius.min()),
        "minimum learned-readout 90-percent energy radius among validated unused units")
    add("median learned-readout support", "mask_energy_radius_90_stimulus_px", float(radius.median()),
        "closest to median learned-readout 90-percent energy radius among validated unused units")
    add("large learned-readout support", "mask_energy_radius_90_stimulus_px", float(radius.max()),
        "maximum learned-readout 90-percent energy radius among validated unused units")
    add("low spatial-frequency control", "preferred_sf_cpd", float(eligible.preferred_sf_cpd.min()),
        "minimum recorded-validated preferred spatial frequency among unused units")
    add("high spatial-frequency control", "preferred_sf_cpd", float(eligible.preferred_sf_cpd.max()),
        "maximum recorded-validated preferred spatial frequency among unused units")
    add(
        "strong orientation-selectivity control", "grating_orientation_vector_strength",
        float(eligible.grating_orientation_vector_strength.max()),
        "maximum fixed-retina grating orientation-vector strength among unused units",
    )
    return pd.DataFrame(selected).reset_index(drop=True)


def attach_preferred_probes(units: pd.DataFrame, tuning_dir: Path) -> pd.DataFrame:
    with np.load(tuning_dir / "grating_only_orientation_tuning.npz", allow_pickle=False) as archive:
        archive_units = np.asarray(archive["rr100_index"], dtype=int)
        sf = np.asarray(archive["measured_sf_cpd"], dtype=float)
        tf = np.asarray(archive["measured_tf_hz"], dtype=float)
        orientation = np.asarray(archive["measured_grating_orientation_deg"], dtype=float)
        positive = np.asarray(archive["measured_positive_f0_hz"], dtype=float)
    lookup = {int(unit): index for index, unit in enumerate(archive_units)}
    rows = []
    for row in units.itertuples(index=False):
        unit = int(row.rr100_index)
        response = positive[lookup[unit]]
        sf_index, tf_index, orientation_index = np.unravel_index(int(np.argmax(response)), response.shape)
        values = dict(row._asdict())
        values.update(
            {
                "probe_spatial_frequency_cpd": float(sf[sf_index]),
                "probe_temporal_frequency_hz": float(tf[tf_index]),
                "probe_orientation_deg": float(orientation[orientation_index]),
                "probe_measured_positive_f0_hz": float(response[sf_index, tf_index, orientation_index]),
                "probe_selection_criterion": "maximum measured positive fixed-retina grating response for this unit",
                "probe_phase_rad": PHASE_RAD,
                "probe_gaussian_sigma_px": PROBE_SIGMA_PX,
                "probe_normalized_contrast": CONTRAST,
            }
        )
        rows.append(values)
    return pd.DataFrame(rows)


def make_probe_cube(size: int, *, sf: float, tf: float, orientation_deg: float) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    center = 0.5 * (size - 1)
    x_deg = (xx - center) / PPD
    y_deg = (yy - center) / PPD
    theta = np.deg2rad(float(orientation_deg))
    normal = -np.sin(theta) * x_deg + np.cos(theta) * y_deg
    window = np.exp(-0.5 * ((xx - center) ** 2 + (yy - center) ** 2) / PROBE_SIGMA_PX**2)
    lag = np.arange(N_HISTORY, dtype=np.float64)
    time_seconds = -lag / FRAME_RATE_HZ
    phase = (
        2.0 * np.pi * float(sf) * normal[None]
        - 2.0 * np.pi * float(tf) * time_seconds[:, None, None]
        + PHASE_RAD
    )
    return (CONTRAST * np.sin(phase) * window[None]).astype(np.float32)


def embed_native(native: np.ndarray, dy: int, dx: int) -> np.ndarray:
    output = np.zeros((N_HISTORY, LARGE_SIZE, LARGE_SIZE), dtype=np.float32)
    start = (LARGE_SIZE - NATIVE_SIZE) // 2
    y0, x0 = start + int(dy), start + int(dx)
    if y0 < 0 or x0 < 0 or y0 + NATIVE_SIZE > LARGE_SIZE or x0 + NATIVE_SIZE > LARGE_SIZE:
        raise ValueError(f"Translation {(dy, dx)} moves native probe outside the large canvas")
    output[:, y0:y0 + NATIVE_SIZE, x0:x0 + NATIVE_SIZE] = native
    return output


def forward_selected_maps(
    bundle: Any, cubes: np.ndarray, canonical_channels: np.ndarray, *, batch_size: int
) -> np.ndarray:
    device = torch.device(bundle.device)
    dtype = next(bundle.model.model.parameters()).dtype
    outputs = []
    bundle.model.model.eval()
    bundle.readout.eval()
    with torch.no_grad():
        for start in range(0, len(cubes), int(batch_size)):
            x = torch.as_tensor(cubes[start:start + int(batch_size), None], device=device, dtype=dtype)
            core = bundle.model.model.core_forward(x, None)
            rates = bundle.model.model.activation(bundle.readout(core[:, :, -1]))
            outputs.append(rates[:, canonical_channels].detach().cpu().numpy().astype(np.float32))
            del x, core, rates
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    return np.concatenate(outputs, axis=0)


def shifted_overlap_metrics(reference: np.ndarray, observed: np.ndarray, shift_y: int, shift_x: int) -> dict[str, float]:
    height, width = reference.shape
    ref_y = slice(0, height - shift_y) if shift_y >= 0 else slice(-shift_y, height)
    obs_y = slice(shift_y, height) if shift_y >= 0 else slice(0, height + shift_y)
    ref_x = slice(0, width - shift_x) if shift_x >= 0 else slice(-shift_x, width)
    obs_x = slice(shift_x, width) if shift_x >= 0 else slice(0, width + shift_x)
    left = np.asarray(reference[ref_y, ref_x], dtype=np.float64).ravel()
    right = np.asarray(observed[obs_y, obs_x], dtype=np.float64).ravel()
    difference = right - left
    correlation = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 1e-12 and np.std(right) > 1e-12 else float("nan")
    return {
        "overlap_bin_count": int(left.size),
        "translation_map_pearson_r": correlation,
        "translation_map_root_mean_square_error_hz": float(np.sqrt(np.mean(difference**2))),
        "translation_map_normalized_root_mean_square_error": float(
            np.sqrt(np.mean(difference**2)) / max(np.sqrt(np.mean(left**2)), 1e-12)
        ),
        "translation_map_maximum_absolute_error_hz": float(np.max(np.abs(difference))),
    }


def weighted_geometry(weight: np.ndarray, *, coordinate_step_px: float) -> dict[str, float]:
    values = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    values /= max(float(values.sum()), EPS)
    yy, xx = np.mgrid[: values.shape[0], : values.shape[1]].astype(np.float64)
    center_y = 0.5 * (values.shape[0] - 1)
    center_x = 0.5 * (values.shape[1] - 1)
    y = (yy - center_y) * float(coordinate_step_px)
    x = (xx - center_x) * float(coordinate_step_px)
    cy, cx = float(np.sum(values * y)), float(np.sum(values * x))
    dy, dx = y - cy, x - cx
    covariance = np.asarray(
        [[np.sum(values * dy * dy), np.sum(values * dy * dx)],
         [np.sum(values * dy * dx), np.sum(values * dx * dx)]],
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    vector = eigenvectors[:, order[0]]
    radius = np.hypot(dy, dx)
    flat_order = np.argsort(radius.ravel())
    cumulative = np.cumsum(values.ravel()[flat_order])
    radius90 = radius.ravel()[flat_order][min(int(np.searchsorted(cumulative, 0.9)), radius.size - 1)]
    return {
        "center_y_relative_input_px": cy,
        "center_x_relative_input_px": cx,
        "center_offset_input_px": float(np.hypot(cy, cx)),
        "sigma_major_input_px": float(np.sqrt(eigenvalues[0])),
        "sigma_minor_input_px": float(np.sqrt(eigenvalues[1])),
        "axis_ratio": float(np.sqrt(eigenvalues[0]) / max(np.sqrt(eigenvalues[1]), EPS)),
        "principal_angle_deg": float(np.degrees(np.arctan2(vector[0], vector[1]))),
        "energy_radius_90_input_px": float(radius90),
    }


def gradient_energy(bundle: Any, cube: np.ndarray, canonical_channel: int) -> tuple[np.ndarray, float]:
    device = torch.device(bundle.device)
    dtype = next(bundle.model.model.parameters()).dtype
    for parameter in bundle.model.model.parameters():
        parameter.requires_grad_(False)
    for parameter in bundle.readout.parameters():
        parameter.requires_grad_(False)
    x = torch.as_tensor(cube[None, None], device=device, dtype=dtype).clone().requires_grad_(True)
    core = bundle.model.model.core_forward(x, None)
    rates = bundle.model.model.activation(bundle.readout(core[:, :, -1]))
    if rates.shape[-2:] != (1, 1):
        raise ValueError(f"Native input did not produce a scalar rate map: {tuple(rates.shape)}")
    target = rates[0, int(canonical_channel), 0, 0]
    gradient = torch.autograd.grad(target, x, create_graph=False, retain_graph=False)[0][0, 0]
    energy = np.square(gradient.detach().cpu().numpy().astype(np.float64)).sum(axis=0)
    rate = float(target.detach().cpu())
    del x, core, rates, target, gradient
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return energy.astype(np.float32), rate


def plot_probe_design(units: pd.DataFrame, native: np.ndarray, extended: np.ndarray, embedded: np.ndarray, path: Path, dpi: int) -> None:
    figure, axes = plt.subplots(len(units), 3, figsize=(12, 3.1 * len(units)), constrained_layout=True)
    for row, unit in units.iterrows():
        panels = (
            (native[row, 0], "native 51×51 probe"),
            (embedded[row, 0], "same 51×51 probe embedded in 151×151"),
            (extended[row, 0], "same analytic probe extended over 151×151"),
        )
        for column, (values, label) in enumerate(panels):
            axes[row, column].imshow(values, cmap="gray", vmin=-CONTRAST, vmax=CONTRAST, origin="lower")
            axes[row, column].set_title(label, fontsize=9)
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(
            f"RR100 unit {int(unit.rr100_index)}\n{unit.selection_role}\n"
            f"{unit.probe_spatial_frequency_cpd:g} cycles/degree, "
            f"{unit.probe_temporal_frequency_hz:g} Hz, {unit.probe_orientation_deg:g}°",
            fontsize=8,
        )
    figure.suptitle(
        "Cartographer Stage 3 input checkpoint: identical preferred-grating probes across native and large canvases\n"
        "Displayed image is the current lag; every response uses the complete 32-frame history",
        fontsize=14, weight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_equivalence(metrics: pd.DataFrame, path: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, column, title in (
        (axes[0], "large_embedded_center_rate_hz", "Central response to exactly embedded probe"),
        (axes[1], "large_extended_center_rate_hz", "Central response to analytically extended probe"),
    ):
        axis.scatter(metrics.native_rate_hz, metrics[column], s=28, alpha=0.75)
        low = float(min(metrics.native_rate_hz.min(), metrics[column].min()))
        high = float(max(metrics.native_rate_hz.max(), metrics[column].max()))
        axis.plot([low, high], [low, high], color="0.35", ls="--")
        axis.set(xlabel="native 51×51 response (Hz)", ylabel="large-canvas central response (Hz)", title=title)
    axes[2].scatter(
        metrics.native_rate_hz,
        np.maximum(metrics.embedded_absolute_error_hz, metrics.extended_absolute_error_hz),
        s=28, alpha=0.75, color="#D55E00",
    )
    axes[2].set_yscale("symlog", linthresh=1e-10)
    axes[2].set(
        xlabel="native 51×51 response (Hz)", ylabel="maximum absolute equivalence error (Hz)",
        title="Numerical equivalence error for all selected unit–probe pairs",
    )
    figure.suptitle(
        "Stage 3 response-scale test: a central large-canvas readout reproduces the native scalar pathway",
        fontsize=14, weight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_translation_maps(
    units: pd.DataFrame, maps: np.ndarray, blank: np.ndarray, key_lookup: dict[tuple[int, str, int, int], int],
    path: Path, dpi: int,
) -> None:
    figure, axes = plt.subplots(len(units), 4, figsize=(15, 3.2 * len(units)), constrained_layout=True)
    for row, unit in units.iterrows():
        probe = int(row)
        target_column = int(row)
        center = maps[key_lookup[(probe, "embedded", 0, 0)], target_column] - blank[target_column]
        shifted = maps[key_lookup[(probe, "embedded", 0, 8)], target_column] - blank[target_column]
        predicted = np.zeros_like(center)
        predicted[:, 4:] = center[:, :-4]
        difference = shifted - predicted
        limit = max(float(np.quantile(np.abs(np.stack([center, shifted])), 0.995)), 1e-8)
        diff_limit = max(float(np.quantile(np.abs(difference), 0.995)), 1e-8)
        for column, (values, title) in enumerate(
            ((center, "centered probe response minus blank"),
             (shifted, "probe translated +8 input pixels horizontally"),
             (predicted, "centered response translated +4 map bins"))
        ):
            axes[row, column].imshow(values, cmap="RdBu_r", origin="lower", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
            axes[row, column].set_title(title, fontsize=8)
        axes[row, 3].imshow(difference, cmap="RdBu_r", origin="lower", norm=TwoSlopeNorm(vmin=-diff_limit, vcenter=0, vmax=diff_limit))
        axes[row, 3].set_title("observed minus translated prediction", fontsize=8)
        for axis in axes[row]:
            axis.set_xticks([]); axis.set_yticks([])
        axes[row, 0].set_ylabel(f"RR100 unit {int(unit.rr100_index)}\n{unit.selection_role}", fontsize=8)
    figure.suptitle(
        "Stage 3 raw translation maps: two input pixels correspond to one activation-map bin\n"
        "Each row uses that unit's strongest measured grating; maps show signed firing-rate modulation from blank",
        fontsize=14, weight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_apertures(
    units: pd.DataFrame, readout_maps: np.ndarray, gradient_maps: np.ndarray, envelope_maps: np.ndarray,
    candidates: pd.DataFrame, path: Path, dpi: int,
) -> None:
    figure, axes = plt.subplots(len(units), 3, figsize=(12, 3.2 * len(units)), constrained_layout=True)
    labels = (
        "learned readout back-projected to input pixels",
        "squared input gradient for the native grating response",
        "squared response modulation while translating the localized probe",
    )
    arrays = (readout_maps, gradient_maps, envelope_maps)
    kinds = ("learned readout back-projection", "grating input-gradient energy", "translated-probe response envelope")
    for row, unit in units.iterrows():
        for column, (collection, label, kind) in enumerate(zip(arrays, labels, kinds, strict=True)):
            values = np.asarray(collection[row], dtype=np.float64)
            axes[row, column].imshow(values / max(float(values.max()), EPS), cmap="magma", origin="lower")
            record = candidates.loc[
                candidates.rr100_index.eq(int(unit.rr100_index)) & candidates.candidate_aperture.eq(kind)
            ].iloc[0]
            axes[row, column].set_title(
                f"{label}\nmajor/minor width={record.sigma_major_input_px:.2f}/{record.sigma_minor_input_px:.2f} input pixels; "
                f"90% radius={record.energy_radius_90_input_px:.2f}",
                fontsize=7.5,
            )
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(f"RR100 unit {int(unit.rr100_index)}\n{unit.selection_role}", fontsize=8)
    figure.suptitle(
        "Stage 3 candidate spatial apertures: architectural support, local gradient sensitivity, and translated-probe response\n"
        "Every panel is independently normalized to expose shape; numeric widths remain in common input-pixel units",
        fontsize=14, weight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(f"Completed checkpoint already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    units = attach_preferred_probes(select_units(args.assignments, args.readout_audit, args.tuning_dir), args.tuning_dir)
    units.to_csv(out_dir / "selected_units_and_preferred_probes.csv", index=False)
    native = np.stack(
        [
            make_probe_cube(
                NATIVE_SIZE, sf=row.probe_spatial_frequency_cpd,
                tf=row.probe_temporal_frequency_hz, orientation_deg=row.probe_orientation_deg,
            )
            for row in units.itertuples(index=False)
        ]
    )
    extended = np.stack(
        [
            make_probe_cube(
                LARGE_SIZE, sf=row.probe_spatial_frequency_cpd,
                tf=row.probe_temporal_frequency_hz, orientation_deg=row.probe_orientation_deg,
            )
            for row in units.itertuples(index=False)
        ]
    )
    embedded_center = np.stack([embed_native(cube, 0, 0) for cube in native])
    center = (LARGE_SIZE - NATIVE_SIZE) // 2
    maximum_central_crop_error = float(
        np.max(np.abs(extended[:, :, center:center + NATIVE_SIZE, center:center + NATIVE_SIZE] - native))
    )
    if maximum_central_crop_error > 1e-6:
        raise ValueError(f"Analytic large probe does not reproduce native center: {maximum_central_crop_error}")

    plot_probe_design(units, native, extended, embedded_center, out_dir / "01_probe_and_coordinate_design", int(args.dpi))
    np.savez_compressed(
        out_dir / "native_probe_histories_and_translation_design.npz",
        native_probe_histories=native,
        translation_input_pixels=np.asarray(TRANSLATIONS_PX, dtype=np.int64),
        rr100_index=units.rr100_index.to_numpy(int),
        canonical_channel=units.canonical_channel.to_numpy(int),
        pixels_per_degree=np.asarray(PPD),
        probe_gaussian_sigma_px=np.asarray(PROBE_SIGMA_PX),
        normalized_contrast=np.asarray(CONTRAST),
        phase_rad=np.asarray(PHASE_RAD),
    )

    large_keys: list[tuple[int, str, int, int]] = []
    large_cubes: list[np.ndarray] = []
    for probe in range(len(units)):
        large_keys.append((probe, "extended", 0, 0)); large_cubes.append(extended[probe])
        for dy, dx in TRANSLATIONS_PX:
            large_keys.append((probe, "embedded", int(dy), int(dx)))
            large_cubes.append(embed_native(native[probe], int(dy), int(dx)))
    large_stack = np.stack(large_cubes)
    del large_cubes

    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    if view.membership is None or view.membership.shape != (100, 756):
        raise ValueError("Stage 3 requires the one-hot RR100 movie-medoid population view")
    mapped_channels = np.argmax(view.membership, axis=1).astype(int)
    if not np.allclose(view.membership[np.arange(100), mapped_channels], 1.0):
        raise ValueError("RR100 population view is not one-hot")
    canonical_channels = units.canonical_channel.to_numpy(int)
    expected_channels = mapped_channels[units.rr100_index.to_numpy(int)]
    if not np.array_equal(canonical_channels, expected_channels):
        raise ValueError("Selected-unit canonical channels disagree with the frozen RR100 view")

    bundle = load_canonical_twin_bundle(device=str(args.device), mode="standard")
    native_maps = forward_selected_maps(bundle, native, canonical_channels, batch_size=int(args.batch_size))
    large_maps = forward_selected_maps(bundle, large_stack, canonical_channels, batch_size=int(args.batch_size))
    blank_large = forward_selected_maps(
        bundle, np.zeros((1, N_HISTORY, LARGE_SIZE, LARGE_SIZE), dtype=np.float32),
        canonical_channels, batch_size=1,
    )[0]
    if native_maps.shape[-2:] != (1, 1) or large_maps.shape[-2:] != (MAP_SIZE, MAP_SIZE):
        raise ValueError(f"Unexpected native/large map shapes {native_maps.shape}/{large_maps.shape}")
    key_lookup = {key: index for index, key in enumerate(large_keys)}
    center_map = MAP_SIZE // 2

    equivalence_rows = []
    for probe, probe_row in units.iterrows():
        embedded_index = key_lookup[(probe, "embedded", 0, 0)]
        extended_index = key_lookup[(probe, "extended", 0, 0)]
        for selected_column, response_row in units.iterrows():
            native_rate = float(native_maps[probe, selected_column, 0, 0])
            embedded_rate = float(large_maps[embedded_index, selected_column, center_map, center_map])
            extended_rate = float(large_maps[extended_index, selected_column, center_map, center_map])
            equivalence_rows.append(
                {
                    "probe_owner_rr100_index": int(probe_row.rr100_index),
                    "response_rr100_index": int(response_row.rr100_index),
                    "response_selection_role": response_row.selection_role,
                    "probe_spatial_frequency_cpd": float(probe_row.probe_spatial_frequency_cpd),
                    "probe_temporal_frequency_hz": float(probe_row.probe_temporal_frequency_hz),
                    "probe_orientation_deg": float(probe_row.probe_orientation_deg),
                    "native_rate_hz": native_rate,
                    "large_embedded_center_rate_hz": embedded_rate,
                    "large_extended_center_rate_hz": extended_rate,
                    "embedded_absolute_error_hz": abs(embedded_rate - native_rate),
                    "extended_absolute_error_hz": abs(extended_rate - native_rate),
                }
            )
    equivalence = pd.DataFrame(equivalence_rows)
    equivalence.to_csv(out_dir / "native_to_large_canvas_response_equivalence.csv", index=False)

    translation_rows = []
    for probe, probe_row in units.iterrows():
        reference_index = key_lookup[(probe, "embedded", 0, 0)]
        for selected_column, response_row in units.iterrows():
            reference = large_maps[reference_index, selected_column] - blank_large[selected_column]
            for dy, dx in TRANSLATIONS_PX:
                observed = large_maps[key_lookup[(probe, "embedded", int(dy), int(dx))], selected_column] - blank_large[selected_column]
                if dy % 2 or dx % 2:
                    raise ValueError("All Stage 3 translations must be divisible by the measured two-pixel jump")
                metrics = shifted_overlap_metrics(reference, observed, dy // 2, dx // 2)
                translation_rows.append(
                    {
                        "probe_owner_rr100_index": int(probe_row.rr100_index),
                        "response_rr100_index": int(response_row.rr100_index),
                        "response_selection_role": response_row.selection_role,
                        "translation_y_input_px": int(dy), "translation_x_input_px": int(dx),
                        "expected_translation_y_map_bins": int(dy // 2),
                        "expected_translation_x_map_bins": int(dx // 2),
                        **metrics,
                    }
                )
    translation = pd.DataFrame(translation_rows)
    translation.to_csv(out_dir / "translation_equivariance_metrics.csv", index=False)

    readout_maps = []
    gradient_maps = []
    envelope_maps = []
    candidate_rows = []
    for ordinal, unit in units.iterrows():
        canonical = int(unit.canonical_channel)
        mask = bundle.readout.space_weights[canonical, 0].detach().cpu().numpy()
        footprint, _ = backproject_readout_footprint(mask)
        gradient, gradient_rate = gradient_energy(bundle, native[ordinal], canonical)
        centered = large_maps[key_lookup[(ordinal, "embedded", 0, 0)], ordinal] - blank_large[ordinal]
        envelope = np.square(centered, dtype=np.float32)
        readout_maps.append(footprint)
        gradient_maps.append(gradient)
        envelope_maps.append(envelope)
        for name, values, step in (
            ("learned readout back-projection", footprint, 1.0),
            ("grating input-gradient energy", gradient, 1.0),
            ("translated-probe response envelope", envelope, 2.0),
        ):
            candidate_rows.append(
                {
                    "rr100_index": int(unit.rr100_index), "canonical_channel": canonical,
                    "selection_role": unit.selection_role, "candidate_aperture": name,
                    "coordinate_step_input_px": step, "native_gradient_rate_hz": gradient_rate,
                    **weighted_geometry(values, coordinate_step_px=step),
                }
            )
    readout_maps_array = np.stack(readout_maps).astype(np.float32)
    gradient_maps_array = np.stack(gradient_maps).astype(np.float32)
    envelope_maps_array = np.stack(envelope_maps).astype(np.float32)
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(out_dir / "candidate_aperture_geometry.csv", index=False)
    np.savez_compressed(
        out_dir / "selected_unit_spatial_contract_maps.npz",
        rr100_index=units.rr100_index.to_numpy(int),
        large_condition_keys=np.asarray([json.dumps(key) for key in large_keys], dtype="U80"),
        selected_unit_large_rate_maps=large_maps,
        selected_unit_blank_large_rate_maps=blank_large,
        learned_readout_backprojected_footprints=readout_maps_array,
        native_grating_input_gradient_energy=gradient_maps_array,
        translated_probe_response_envelopes=envelope_maps_array,
    )

    plot_equivalence(equivalence, out_dir / "02_native_to_large_canvas_response_equivalence", int(args.dpi))
    plot_translation_maps(units, large_maps, blank_large, key_lookup, out_dir / "03_translation_equivariance_raw_maps", int(args.dpi))
    plot_apertures(
        units, readout_maps_array, gradient_maps_array, envelope_maps_array, candidates,
        out_dir / "04_candidate_spatial_apertures", int(args.dpi),
    )

    nonzero_translation = translation.loc[
        translation.translation_x_input_px.ne(0) | translation.translation_y_input_px.ne(0)
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "cartographer_rr100_spatial_coordinate_contract_stage3",
        "status": "targeted_spatial_contract_checkpoint_complete_awaiting_human_review",
        "scope": {
            "n_selected_units": int(len(units)),
            "n_unit_probe_equivalence_pairs": int(len(equivalence)),
            "n_translation_measurements": int(len(translation)),
            "translations_input_pixels": list(TRANSLATIONS_PX),
            "reserved_final_test_identities_opened": False,
        },
        "contracts": {
            "unit_selection": "algorithmic from validated grating and learned-readout metadata before Stage 3 responses",
            "probe_selection": "each selected unit's maximum measured positive fixed-retina grating response",
            "input": "normalized 32-lag Gaussian-windowed drifting grating; native 51x51 and large 151x151 canvases",
            "coordinate_hypothesis": "two input pixels correspond to one activation-map bin",
            "response": "post-output-activation firing rate in hertz",
            "aperture_candidates": [
                "learned readout back-projection", "grating input-gradient energy", "translated-probe response envelope"
            ],
            "checkpoint": "stop before Stage 4 local power-map construction",
        },
        "validation": {
            "maximum_analytic_large_center_crop_input_error": maximum_central_crop_error,
            "maximum_embedded_native_response_error_hz": float(equivalence.embedded_absolute_error_hz.max()),
            "maximum_extended_native_response_error_hz": float(equivalence.extended_absolute_error_hz.max()),
            "minimum_nonzero_translation_map_pearson_r": float(nonzero_translation.translation_map_pearson_r.min()),
            "maximum_nonzero_translation_normalized_rmse": float(
                nonzero_translation.translation_map_normalized_root_mean_square_error.max()
            ),
        },
        "decision_gate": (
            "inspect exact response equivalence, raw translated maps, edge offsets, and candidate aperture shapes; "
            "do not choose an aperture or begin Stage 4 until human review"
        ),
        "sources": {
            "tuning": file_identity(args.tuning_dir / "grating_only_orientation_tuning.npz"),
            "validated_assignments": file_identity(args.assignments),
            "readout_audit": file_identity(args.readout_audit),
            "runner": file_identity(Path(__file__)),
        },
        "outputs": {
            name: file_identity(out_dir / name)
            for name in (
                "selected_units_and_preferred_probes.csv",
                "native_probe_histories_and_translation_design.npz",
                "native_to_large_canvas_response_equivalence.csv",
                "translation_equivariance_metrics.csv",
                "candidate_aperture_geometry.csv",
                "selected_unit_spatial_contract_maps.npz",
                "01_probe_and_coordinate_design.png",
                "02_native_to_large_canvas_response_equivalence.png",
                "03_translation_equivariance_raw_maps.png",
                "04_candidate_spatial_apertures.png",
            )
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(json_ready(manifest), indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Cartographer Stage 3 spatial-coordinate checkpoint\n\n"
        "This targeted checkpoint tests native-to-large-canvas response equivalence, the two-input-pixel "
        "activation-map stride, edge translations, and three candidate spatial-aperture constructions for six "
        "units selected before Stage 3 responses. It uses only synthetic preferred-grating probes and opens no "
        "reserved natural-image or trace identities. Stop for human review before Stage 4.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(manifest), indent=2))


if __name__ == "__main__":
    main()
