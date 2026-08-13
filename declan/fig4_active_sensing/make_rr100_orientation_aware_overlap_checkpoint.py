#!/usr/bin/env python3
"""Checkpoint 03: orientation-aware retinal-power overlap for selected RR100 units.

The prior SF-by-TF proxy radialized retinal power. Here each Fourier mode is
mapped to the corresponding grating-bar orientation and weighted by the unit's
measured four-point positive-F0 orientation curve. A 90-degree rotation of that
curve is the explicit orthogonal control. Smoothing is display-only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from declan.active_sensing_movie_information.plot_rr100_kuang_input_power_checkpoint import spectral_decomposition
from declan.fig4_active_sensing.make_rr100_kuang_unit_overlap_checkpoint import surface
from declan.fig4_active_sensing.polish_rr100_kuang_unit_overlap_checkpoint import kuang_colormap


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "outputs/fig4_active_sensing/rr100_kuang_input_power_checkpoint_01_v1"
MODEL_DIR = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1"
F0_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_f0_factorization_v1"
SELECTION_DIR = ROOT / "outputs/fig4_active_sensing/rr100_kuang_unit_overlap_checkpoint_02_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_orientation_aware_overlap_checkpoint_03_v1"
FRAME_RATE_HZ = 120.0
SF_MIN, SF_MAX = 1.0, float(8.0 * np.sqrt(2.0))
TF_MIN, TF_MAX = 0.5, 32.0
DISPLAY_FLOOR_DB = -30.0
SMOOTH_SF_OCTAVES = 0.10
SMOOTH_TF_OCTAVES = 0.12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--f0-dir", type=Path, default=F0_DIR)
    parser.add_argument("--selection-dir", type=Path, default=SELECTION_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--dpi", type=int, default=230)
    return parser.parse_args()


def axial_delta_deg(a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
    return np.abs((np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 90.0) % 180.0 - 90.0)


def orientation_factor(table: pd.DataFrame, angles_deg: np.ndarray, shift_deg: float = 0.0) -> np.ndarray:
    frame = table.sort_values("orientation_deg")
    sampled_angle = frame["orientation_deg"].to_numpy(float)
    sampled_value = frame["mean_positive_f0_hz"].to_numpy(float, copy=True)
    sampled_value /= max(float(sampled_value.max()), 1e-15)
    extended_angle = np.r_[sampled_angle - 180.0, sampled_angle, sampled_angle + 180.0]
    extended_value = np.tile(sampled_value, 3)
    query = (np.asarray(angles_deg, dtype=float) - float(shift_deg)) % 180.0
    return np.interp(query, extended_angle, extended_value)


def validate_orientation_convention(ppd: float, frame_size: int = 51, spatial_cpd: float = 4.0) -> pd.DataFrame:
    coordinate = np.arange(frame_size, dtype=float) - (frame_size - 1.0) / 2.0
    xx, yy = np.meshgrid(coordinate, coordinate)
    fy = np.fft.fftshift(np.fft.fftfreq(frame_size, d=1.0 / ppd))
    fx = np.fft.fftshift(np.fft.fftfreq(frame_size, d=1.0 / ppd))
    fxx, fyy = np.meshgrid(fx, fy)
    radial = np.hypot(fxx, fyy)
    window = np.outer(np.hanning(frame_size), np.hanning(frame_size))
    rows = []
    for orientation in (0.0, 45.0, 90.0, 135.0):
        radians = np.deg2rad(orientation)
        # Matches the native renderer: bar orientation o has normal
        # sin(o)*x + cos(o)*y in image-array coordinates.
        normal_px = np.sin(radians) * xx + np.cos(radians) * yy
        grating = np.cos(2.0 * np.pi * spatial_cpd * normal_px / ppd)
        power = np.abs(np.fft.fftshift(np.fft.fft2(grating * window))) ** 2
        candidate = power.copy()
        candidate[radial < spatial_cpd / 2.0] = 0.0
        iy, ix = np.unravel_index(int(np.argmax(candidate)), candidate.shape)
        normal_angle = math.degrees(math.atan2(float(fyy[iy, ix]), float(fxx[iy, ix])))
        recovered = (90.0 - normal_angle) % 180.0
        rows.append({
            "input_grating_bar_orientation_deg": orientation,
            "peak_fourier_fx_cpd": float(fxx[iy, ix]),
            "peak_fourier_fy_cpd": float(fyy[iy, ix]),
            "recovered_grating_bar_orientation_deg": recovered,
            "axial_error_deg": float(axial_delta_deg(recovered, orientation)),
        })
    return pd.DataFrame(rows)


def bin_power_by_sf(dynamic_power: np.ndarray, radial_sf: np.ndarray, sf_edges: np.ndarray,
                    mode_weight: np.ndarray) -> np.ndarray:
    rows = []
    for lo, hi in zip(sf_edges[:-1], sf_edges[1:], strict=True):
        mask = (radial_sf >= lo) & (radial_sf < hi)
        rows.append(np.sum(dynamic_power[:, mask] * mode_weight[mask][None, :], axis=1))
    return np.asarray(rows, dtype=float)


def interpolate_and_smooth(sf: np.ndarray, tf: np.ndarray, values: np.ndarray,
                           dense_sf: np.ndarray, dense_tf: np.ndarray) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (np.log2(sf), np.log2(tf)), np.asarray(values, dtype=float), bounds_error=False, fill_value=None,
    )
    grid_sf, grid_tf = np.meshgrid(np.log2(dense_sf), np.log2(dense_tf), indexing="ij")
    dense = interpolator(np.column_stack([grid_sf.ravel(), grid_tf.ravel()])).reshape(grid_sf.shape)
    dense = np.maximum(dense, 0.0)
    sf_step = float(np.mean(np.diff(np.log2(dense_sf))))
    tf_step = float(np.mean(np.diff(np.log2(dense_tf))))
    sigma = (SMOOTH_SF_OCTAVES / sf_step, SMOOTH_TF_OCTAVES / tf_step)
    return gaussian_filter(dense, sigma=sigma, mode="nearest")


def edges(values: np.ndarray) -> np.ndarray:
    middle = 0.5 * (values[:-1] + values[1:])
    return np.r_[values[0] - (middle[0] - values[0]), middle, values[-1] + (values[-1] - middle[-1])]


def setup_map_axis(axis: plt.Axes, show_y: bool) -> None:
    sf_ticks = np.asarray([1, 2, 4, 8], dtype=float)
    tf_ticks = np.asarray([1, 2, 4, 8, 16, 32], dtype=float)
    axis.set_xticks(np.log2(sf_ticks), [f"{value:g}" for value in sf_ticks])
    axis.set_yticks(np.log2(tf_ticks), [f"{value:g}" for value in tf_ticks] if show_y else [])
    axis.set_xlabel("SF (cpd)", fontsize=7.5)
    if show_y:
        axis.set_ylabel("TF (Hz)", fontsize=7.5)
    axis.tick_params(labelsize=7, length=2)


def db(values: np.ndarray, reference: float) -> np.ndarray:
    relative = np.maximum(np.asarray(values, dtype=float) / max(float(reference), 1e-30), 10.0 ** (DISPLAY_FLOOR_DB / 10.0))
    return 10.0 * np.log10(relative)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    input_manifest = json.loads((args.input_dir / "manifest.json").read_text())
    ppd = float(input_manifest["model_ppd"])
    archive = np.load(args.input_dir / "checkpoint_01_retinal_movies_and_power.npz")
    movie = archive["real_fem_movie"].astype(np.float64)
    decomp = spectral_decomposition(movie, ppd=ppd, frame_rate_hz=FRAME_RATE_HZ)
    dynamic = decomp["dynamic_power_tf_y_x"]
    tf_all = decomp["temporal_frequency_hz"]
    radial_sf = decomp["radial_sf_cpd"]
    frame_size = int(movie.shape[-1])
    sf_edges = np.geomspace(ppd / frame_size, ppd / 2.0, 14)
    sf_centers = np.sqrt(sf_edges[:-1] * sf_edges[1:])
    sf_mask = (sf_centers >= SF_MIN) & (sf_centers <= SF_MAX)
    tf_mask = (tf_all >= TF_MIN) & (tf_all <= TF_MAX)
    sf = sf_centers[sf_mask]
    tf = tf_all[tf_mask]
    dynamic_support = dynamic[tf_mask]

    spatial_axis = np.fft.fftshift(np.fft.fftfreq(frame_size, d=1.0 / ppd))
    fx, fy = np.meshgrid(spatial_axis, spatial_axis)
    normal_angle_deg = np.degrees(np.arctan2(fy, fx))
    grating_bar_orientation_deg = (90.0 - normal_angle_deg) % 180.0

    radial_binned_all = bin_power_by_sf(dynamic_support, radial_sf, sf_edges, np.ones_like(radial_sf))
    radial_binned = radial_binned_all[sf_mask]
    saved_radial = archive["real_fem_dynamic_radial_power"][sf_mask][:, tf_mask]
    power_long = pd.read_csv(args.input_dir / "checkpoint_01_sf_tf_power_long.csv")
    mode_count = (
        power_long.loc[power_long["condition"].eq("real_fem")]
        .groupby("sf_bin_center_cpd")["spatial_mode_count"].first().reindex(sf_centers).to_numpy(float)
    )
    saved_annular = saved_radial * mode_count[sf_mask, None]
    radial_reproduction_relative_error = float(
        np.max(np.abs(radial_binned - saved_annular)) / max(float(np.max(saved_annular)), 1e-30)
    )

    models = pd.read_csv(args.model_dir / "rr100_sf_tf_parametric_models.csv").set_index("rr100_index")
    orientation_scores = pd.read_csv(args.f0_dir / "f0_orientation_scores.csv")
    selected = pd.read_csv(args.selection_dir / "selected_unit_roles.csv").sort_values("display_order")
    validation = validate_orientation_convention(ppd, frame_size=frame_size)
    validation.to_csv(args.out_dir / "fourier_grating_orientation_convention_audit.csv", index=False)

    dense_sf = np.geomspace(sf.min(), sf.max(), 181)
    dense_tf = np.geomspace(tf.min(), tf.max(), 241)
    dense_log_sf_edges = edges(np.log2(dense_sf))
    dense_log_tf_edges = edges(np.log2(dense_tf))
    radial_dense = interpolate_and_smooth(sf, tf, radial_binned, dense_sf, dense_tf)
    global_power_reference = float(radial_dense.max())
    cmap = kuang_colormap()
    power_norm = Normalize(vmin=DISPLAY_FLOOR_DB, vmax=0.0)

    records = []
    dense_maps: dict[str, np.ndarray] = {"spatial_cpd": dense_sf, "temporal_hz": dense_tf,
                                        "radial_power_display": radial_dense}
    unit_payload: dict[int, dict[str, np.ndarray | float]] = {}
    for _, choice in selected.iterrows():
        unit = int(choice["rr100_index"])
        unit_orientation = orientation_scores.loc[orientation_scores["rr100_index"].eq(unit)].copy()
        preferred_mode_weight = orientation_factor(unit_orientation, grating_bar_orientation_deg) ** 2
        orthogonal_mode_weight = orientation_factor(unit_orientation, grating_bar_orientation_deg, shift_deg=90.0) ** 2
        preferred_power = bin_power_by_sf(dynamic_support, radial_sf, sf_edges, preferred_mode_weight)[sf_mask]
        orthogonal_power = bin_power_by_sf(dynamic_support, radial_sf, sf_edges, orthogonal_mode_weight)[sf_mask]
        model = models.loc[unit]
        gain = surface(model, sf, tf)
        numerator_radial = float(np.sum(radial_binned * gain**2))
        numerator_preferred = float(np.sum(preferred_power * gain**2))
        numerator_orthogonal = float(np.sum(orthogonal_power * gain**2))
        denominator = float(np.sum(radial_binned))
        preferred_throughput = float(np.sum(preferred_power) / denominator)
        orthogonal_throughput = float(np.sum(orthogonal_power) / denominator)
        metrics = {
            "rr100_index": unit,
            "selection_role": choice["selection_role"],
            "preferred_orientation_deg": float(model["preferred_orientation_deg"]),
            "radial_sf_tf_overlap": numerator_radial / denominator,
            "preferred_orientation_aware_overlap": numerator_preferred / denominator,
            "orthogonal_orientation_control_overlap": numerator_orthogonal / denominator,
            "preferred_minus_orthogonal_overlap": (numerator_preferred - numerator_orthogonal) / denominator,
            "preferred_to_radial_overlap_fraction": numerator_preferred / max(numerator_radial, 1e-30),
            "preferred_to_orthogonal_overlap_ratio": numerator_preferred / max(numerator_orthogonal, 1e-30),
            "preferred_orientation_power_throughput": preferred_throughput,
            "orthogonal_orientation_power_throughput": orthogonal_throughput,
            "conditional_sf_tf_overlap_given_preferred_orientation": numerator_preferred / max(float(np.sum(preferred_power)), 1e-30),
        }
        records.append(metrics)

        preferred_dense = interpolate_and_smooth(sf, tf, preferred_power, dense_sf, dense_tf)
        orthogonal_dense = interpolate_and_smooth(sf, tf, orthogonal_power, dense_sf, dense_tf)
        dense_gain = surface(model, dense_sf, dense_tf)
        dense_gain /= max(float(dense_gain.max()), 1e-15)
        radial_overlap_dense = radial_dense * dense_gain**2
        preferred_overlap_dense = preferred_dense * dense_gain**2
        unit_payload[unit] = {
            "preferred_power": preferred_dense,
            "orthogonal_power": orthogonal_dense,
            "difference_power": preferred_dense - orthogonal_dense,
            "radial_overlap": radial_overlap_dense,
            "preferred_overlap": preferred_overlap_dense,
            "gain_power": dense_gain**2,
            **metrics,
        }
        for name, values in (
            ("preferred_orientation_power", preferred_power),
            ("orthogonal_orientation_power", orthogonal_power),
            ("radial_overlap", radial_binned * gain**2),
            ("preferred_orientation_aware_overlap", preferred_power * gain**2),
        ):
            for i, sf_value in enumerate(sf):
                for j, tf_value in enumerate(tf):
                    records_value = {
                        "rr100_index": unit, "selection_role": choice["selection_role"], "map_kind": name,
                        "spatial_cpd": float(sf_value), "temporal_hz": float(tf_value), "power": float(values[i, j]),
                    }
                    # Stored separately below to avoid mixing unit metrics with map values.
                    dense_maps.setdefault("_long_records", []).append(records_value)
        for key in ("preferred_power", "orthogonal_power", "difference_power", "radial_overlap", "preferred_overlap", "gain_power"):
            dense_maps[f"rr100_{unit:03d}_{key}"] = np.asarray(unit_payload[unit][key])

    metrics_table = pd.DataFrame(records)
    metrics_table.to_csv(args.out_dir / "selected_unit_orientation_overlap_metrics.csv", index=False)
    long_records = dense_maps.pop("_long_records")
    pd.DataFrame(long_records).to_csv(args.out_dir / "selected_unit_orientation_power_maps_long.csv", index=False)
    np.savez_compressed(args.out_dir / "display_smoothed_orientation_maps.npz", **dense_maps)

    # Figure 1: measured orientation curve and preferred-versus-orthogonal power.
    differences = [np.asarray(unit_payload[int(unit)]["difference_power"]) for unit in selected["rr100_index"]]
    difference_limit = max(float(np.max(np.abs(value))) for value in differences) / global_power_reference
    fig, axes = plt.subplots(len(selected), 4, figsize=(12.6, 10.0), constrained_layout=True)
    power_image = None
    diff_image = None
    angle_dense = np.linspace(0.0, 180.0, 361)
    for row_index, (_, choice) in enumerate(selected.iterrows()):
        unit = int(choice["rr100_index"])
        payload = unit_payload[unit]
        ori = orientation_scores.loc[orientation_scores["rr100_index"].eq(unit)].sort_values("orientation_deg")
        curve = orientation_factor(ori, angle_dense)
        ax = axes[row_index, 0]
        ax.plot(angle_dense, curve, color="#A71930", lw=2)
        ax.scatter(ori["orientation_deg"], ori["mean_positive_f0_hz"] / ori["mean_positive_f0_hz"].max(),
                   color="#17175f", s=27, zorder=3)
        preferred_orientation = float(payload["preferred_orientation_deg"])
        ax.axvline(preferred_orientation, color="#111111", lw=1)
        ax.axvline((preferred_orientation + 90.0) % 180.0, color="#777777", lw=1, ls="--")
        ax.set(xlim=(0, 180), ylim=(0, 1.05), xticks=[0, 45, 90, 135, 180],
               xlabel="grating-bar orientation (deg)", ylabel="normalized positive F0")
        ax.tick_params(labelsize=7)
        ax.set_title(f"RR100 {unit}: {choice['selection_role']}", fontsize=8.5, loc="left")

        for column, key, title in (
            (1, "preferred_power", f"preferred-axis power\nthroughput={payload['preferred_orientation_power_throughput']:.2f}"),
            (2, "orthogonal_power", f"90° control power\nthroughput={payload['orthogonal_orientation_power_throughput']:.2f}"),
        ):
            map_axis = axes[row_index, column]
            power_image = map_axis.pcolormesh(
                dense_log_sf_edges, dense_log_tf_edges, db(payload[key], global_power_reference).T,
                cmap=cmap, norm=power_norm, shading="flat", rasterized=True,
            )
            setup_map_axis(map_axis, show_y=(column == 1))
            map_axis.set_title(title, fontsize=8.5)
        ax = axes[row_index, 3]
        difference_relative = np.asarray(payload["difference_power"]) / global_power_reference
        diff_image = ax.pcolormesh(
            dense_log_sf_edges, dense_log_tf_edges, difference_relative.T, cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-difference_limit, vcenter=0.0, vmax=difference_limit), shading="flat",
            rasterized=True,
        )
        setup_map_axis(ax, show_y=False)
        ax.set_title(
            f"preferred − 90° control\nΔ overlap={payload['preferred_minus_orthogonal_overlap']:+.3f}", fontsize=8.5
        )
    for column, title in enumerate(("A  Measured orientation factor", "B  Unit-matched retinal power",
                                    "C  Rotated-orientation control", "D  Direct power difference")):
        axes[0, column].text(0.0, 1.22, title, transform=axes[0, column].transAxes, fontsize=10, fontweight="bold")
    cbar = fig.colorbar(power_image, ax=axes[:, 1:3], shrink=0.75, pad=0.01)
    cbar.set_label("dB relative to peak radial FEM power", fontsize=8)
    cbar = fig.colorbar(diff_image, ax=axes[:, 3], shrink=0.75, pad=0.01)
    cbar.set_label("preferred − control / peak radial power", fontsize=8)
    fig.suptitle("Checkpoint 03: orientation content can support or oppose radial SF–TF overlap", fontsize=13)
    mechanism_path = args.out_dir / "checkpoint_03_orientation_power_mechanism"
    fig.savefig(mechanism_path.with_suffix(".png"), dpi=args.dpi)
    fig.savefig(mechanism_path.with_suffix(".pdf"))
    plt.close(fig)

    # Figure 2: direct radial-versus-orientation-aware overlap comparison.
    fig, axes = plt.subplots(2, len(selected), figsize=(12.8, 6.0), constrained_layout=True)
    overlap_image = None
    for column, (_, choice) in enumerate(selected.iterrows()):
        unit = int(choice["rr100_index"])
        payload = unit_payload[unit]
        for row_index, key, prefix in ((0, "radial_overlap", "radial"), (1, "preferred_overlap", "orientation-aware")):
            ax = axes[row_index, column]
            overlap_image = ax.pcolormesh(
                dense_log_sf_edges, dense_log_tf_edges, db(payload[key], global_power_reference).T,
                cmap=cmap, norm=power_norm, shading="flat", rasterized=True,
            )
            ax.contour(np.log2(dense_sf), np.log2(dense_tf), np.asarray(payload["gain_power"]).T,
                       levels=[10.0 ** (-6.0 / 10.0)], colors="white", linestyles="--", linewidths=0.8)
            setup_map_axis(ax, show_y=(column == 0))
            score = payload["radial_sf_tf_overlap"] if row_index == 0 else payload["preferred_orientation_aware_overlap"]
            if row_index == 0:
                ax.set_title(
                    f"RR100 {unit}: {str(choice['selection_role']).replace(' positive overlap', '')}\n"
                    f"{prefix} overlap={score:.3f}", fontsize=8.5,
                )
            else:
                fraction = payload["preferred_to_radial_overlap_fraction"]
                ax.set_title(f"{prefix} overlap={score:.3f}\nretains {fraction:.0%} of radial proxy", fontsize=8.5)
    axes[0, 0].text(-0.28, 0.5, "RADIAL\n(previous)", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=9, fontweight="bold")
    axes[1, 0].text(-0.28, 0.5, "ORIENTATION-AWARE\n(new)", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=9, fontweight="bold")
    cbar = fig.colorbar(overlap_image, ax=axes, shrink=0.83, pad=0.012)
    cbar.set_label("overlap power, dB relative to peak radial FEM power", fontsize=8)
    fig.suptitle("Radial SF–TF overlap versus unit-orientation-aware overlap", fontsize=13)
    comparison_path = args.out_dir / "checkpoint_03_radial_vs_orientation_aware_overlap"
    fig.savefig(comparison_path.with_suffix(".png"), dpi=args.dpi)
    fig.savefig(comparison_path.with_suffix(".pdf"))
    plt.close(fig)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "selected-unit orientation-aware FEM power overlap checkpoint",
        "scope": "same image, trace, and four audibly selected RR100 units as checkpoints 01 and 02",
        "orientation_contract": {
            "fourier_to_grating": "bar_orientation_deg = (90 - atan2(fy, fx) degrees) mod 180 in image-array coordinates",
            "unit_factor": "circular linear interpolation of four measured mean-positive-F0 values at 0,45,90,135 degrees; normalized by sampled maximum",
            "power_weight": "orientation amplitude factor squared",
            "orthogonal_control": "same measured orientation factor rotated by 90 degrees",
            "separability_assumption": "orientation factor multiplies the preferred-orientation SFxTF parametric amplitude factor",
        },
        "checks": {
            "maximum_synthetic_grating_orientation_error_deg": float(validation["axial_error_deg"].max()),
            "radial_binning_reproduction_relative_max_error": radial_reproduction_relative_error,
        },
        "display_policy": {
            "smoothing": "visual only; linear power interpolated and Gaussian-smoothed on log-frequency axes",
            "smoothing_sigma_octaves": {"sf": SMOOTH_SF_OCTAVES, "tf": SMOOTH_TF_OCTAVES},
            "scores": "computed from unsmoothed native Fourier modes and bins",
        },
        "status": "orientation_mechanism_checkpoint_complete_stop_before_direct_model_response",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Checkpoint 03: orientation-aware FEM power overlap\n\n"
        "This targeted checkpoint asks whether the radial SF-by-TF overlap survives when retinal Fourier modes "
        "are weighted by each selected unit's measured orientation curve. The 90-degree-rotated curve is the "
        "matched control. All reported scores use unsmoothed native bins; smoothing is used only in the figures. "
        "The orientation factor is assumed separable from SF and TF and is sampled at four orientations, so this "
        "remains a mechanistic proxy rather than a calibrated response prediction.\n"
    )
    print(metrics_table.to_string(index=False))
    print(json.dumps(manifest["checks"], indent=2))


if __name__ == "__main__":
    main()
