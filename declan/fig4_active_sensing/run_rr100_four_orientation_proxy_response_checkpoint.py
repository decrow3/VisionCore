#!/usr/bin/env python3
"""Checkpoint 05: four-orientation spectral proxy versus frozen-RR100 response.

The selected natural-image patch and its eye trace are jointly rotated through
0, 45, 90, and 135 degrees before canonical 51x51 retinal rendering.  Each
orientation has its own true-zero-gaze baseline.  Orientation-aware SFxTF
spectral-overlap predictions are saved before the frozen model is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
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
import numpy as np
import pandas as pd
from scipy.ndimage import rotate as ndimage_rotate
from scipy.stats import pearsonr, spearmanr
import torch

from declan.active_sensing_movie_information.plot_rr100_kuang_input_power_checkpoint import (
    FRAME_RATE_HZ,
    render_retinal_movie,
    spectral_decomposition,
    support_summary,
)
from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import source_row_by_id
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import load_source_rows
from declan.active_sensing_movie_information.run_backimage_rr100_frequency_tuning_probe import embed_time_lags_local
from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.fig4_active_sensing.make_rr100_kuang_unit_overlap_checkpoint import surface
from declan.fig4_active_sensing.make_rr100_orientation_aware_overlap_checkpoint import (
    SF_MAX,
    SF_MIN,
    TF_MAX,
    TF_MIN,
    bin_power_by_sf,
    orientation_factor,
)
from declan.fig4_active_sensing.run_rr100_direct_fem_orientation_checkpoint import run_condition
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import CanonicalTwinScorer
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.redundancy_resolved_v1_population import load_population_view


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "outputs/fig4_active_sensing/rr100_kuang_input_power_checkpoint_01_v1"
MODEL_DIR = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1"
F0_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_f0_factorization_v1"
MAPPING = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_recorded_twin_gratings_check_v1/rr100_unit_mapping.csv"
DIRECT_04 = ROOT / "outputs/fig4_active_sensing/rr100_direct_fem_orientation_checkpoint_04_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_four_orientation_proxy_response_checkpoint_05_v2"
ANGLES = np.asarray([0.0, 45.0, 90.0, 135.0])
UNITS = np.asarray([3, 1, 19, 81], dtype=np.int64)
CASES = {
    3: "broad spectral match",
    1: "orientation-specific match",
    19: "orientation mismatch",
    81: "low-overlap control",
}
ANGLE_COLORS = {0.0: "#B2182B", 45.0: "#EF8A62", 90.0: "#2166AC", 135.0: "#67A9CF"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--f0-dir", type=Path, default=F0_DIR)
    parser.add_argument("--mapping-csv", type=Path, default=MAPPING)
    parser.add_argument("--direct-04-dir", type=Path, default=DIRECT_04)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dpi", type=int, default=230)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
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


def axial_delta_deg(a: float, b: float) -> float:
    return float(abs((float(a) - float(b) + 90.0) % 180.0 - 90.0))


def rotate_trace(trace_xy: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate the helper's eye trace with the image in its native x/y convention."""
    radians = math.radians(float(angle_deg))
    rotation = np.asarray(
        [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]],
        dtype=np.float32,
    )
    return np.asarray(trace_xy, dtype=np.float32) @ rotation.T


def rotate_patch(patch: np.ndarray, angle_deg: float) -> np.ndarray:
    quarter_turn = float(angle_deg) / 90.0
    if abs(quarter_turn - round(quarter_turn)) < 1e-12:
        return np.rot90(np.asarray(patch), k=int(round(quarter_turn))).copy()
    return ndimage_rotate(
        np.asarray(patch), float(angle_deg), reshape=False, order=1,
        mode="reflect", prefilter=False,
    ).astype(np.float32, copy=False)


def reconstruct_rotated_movies(input_dir: Path) -> tuple[dict[float, dict[str, np.ndarray]], dict[str, object]]:
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    archive = np.load(input_dir / "checkpoint_01_retinal_movies_and_power.npz")
    source_rows = load_source_rows(Path(manifest["inputs"]["source_windows"]["path"]))
    source_row_id = int(manifest["selected_image"]["source_row"])
    source_row = source_row_by_id(source_rows, source_row_id)
    patch, _ = _extract_patch(source_row, canvas_cache={}, patch_size_px=540)
    trace = archive["centered_real_trace_xy_deg"].astype(np.float32)
    ppd = float(manifest["model_ppd"])
    movies: dict[float, dict[str, np.ndarray]] = {}
    audit_rows = []
    for angle in ANGLES:
        transformed_patch = rotate_patch(patch, float(angle))
        transformed_trace = rotate_trace(trace, float(angle))
        zero = render_retinal_movie(transformed_patch, np.zeros_like(transformed_trace), ppd=ppd)
        fem = render_retinal_movie(transformed_patch, transformed_trace, ppd=ppd)
        movies[float(angle)] = {"zero": zero, "fem": fem}
        decomp = spectral_decomposition(fem, ppd=ppd, frame_rate_hz=FRAME_RATE_HZ)
        support = support_summary(decomp)
        audit_rows.append({
            "rotation_deg": float(angle),
            "patch_mean": float(np.mean(transformed_patch)),
            "patch_sd": float(np.std(transformed_patch)),
            "fem_movie_mean": float(np.mean(fem)),
            "fem_movie_sd": float(np.std(fem)),
            "zero_gaze_max_frame_difference": float(np.max(np.abs(zero - zero[:1]))),
            **support,
        })
    angle0_fem_error = float(np.max(np.abs(movies[0.0]["fem"] - archive["real_fem_movie"])))
    angle0_zero_error = float(np.max(np.abs(movies[0.0]["zero"] - archive["zero_gaze_movie"])))
    angle90_fem_error = float(np.max(np.abs(movies[90.0]["fem"] - np.rot90(archive["real_fem_movie"], axes=(1, 2)))))
    angle90_zero_error = float(np.max(np.abs(movies[90.0]["zero"] - np.rot90(archive["zero_gaze_movie"], axes=(1, 2)))))
    checks = {
        "source_row": source_row_id,
        "patch_shape": list(np.asarray(patch).shape),
        "angle0_saved_fem_max_abs_pixel_error": angle0_fem_error,
        "angle0_saved_zero_max_abs_pixel_error": angle0_zero_error,
        "angle90_prior_array_rotation_fem_max_abs_pixel_error": angle90_fem_error,
        "angle90_prior_array_rotation_zero_max_abs_pixel_error": angle90_zero_error,
        "audit_table": pd.DataFrame(audit_rows),
        "input_manifest": manifest,
    }
    if angle0_fem_error != 0.0 or angle0_zero_error != 0.0:
        raise ValueError(f"0-degree source reconstruction did not reproduce checkpoint 01: {checks}")
    if max(angle90_fem_error, angle90_zero_error) > 0.01:
        raise ValueError(f"90-degree source reconstruction did not reproduce prior rotation: {checks}")
    return movies, checks


def predicted_overlap_table(
    movies: dict[float, dict[str, np.ndarray]], ppd: float, models: pd.DataFrame,
    orientation_scores: pd.DataFrame, units: np.ndarray = UNITS,
) -> pd.DataFrame:
    records = []
    frame_size = int(next(iter(movies.values()))["fem"].shape[-1])
    sf_edges = np.geomspace(ppd / frame_size, ppd / 2.0, 14)
    sf_centers = np.sqrt(sf_edges[:-1] * sf_edges[1:])
    sf_mask = (sf_centers >= SF_MIN) & (sf_centers <= SF_MAX)
    spatial_axis = np.fft.fftshift(np.fft.fftfreq(frame_size, d=1.0 / ppd))
    fx, fy = np.meshgrid(spatial_axis, spatial_axis)
    mode_orientation = (90.0 - np.degrees(np.arctan2(fy, fx))) % 180.0
    for angle in ANGLES:
        decomp = spectral_decomposition(
            movies[float(angle)]["fem"], ppd=ppd, frame_rate_hz=FRAME_RATE_HZ
        )
        dynamic = decomp["dynamic_power_tf_y_x"]
        tf_all = decomp["temporal_frequency_hz"]
        tf_mask = (tf_all >= TF_MIN) & (tf_all <= TF_MAX)
        tf = tf_all[tf_mask]
        dynamic_support = dynamic[tf_mask]
        radial_sf = decomp["radial_sf_cpd"]
        radial_binned = bin_power_by_sf(
            dynamic_support, radial_sf, sf_edges, np.ones_like(radial_sf)
        )[sf_mask]
        denominator = float(np.sum(radial_binned))
        sf = sf_centers[sf_mask]
        for unit in np.asarray(units, dtype=np.int64):
            orientation_table = orientation_scores.loc[orientation_scores["rr100_index"].eq(int(unit))]
            mode_weight = orientation_factor(orientation_table, mode_orientation) ** 2
            matched_power = bin_power_by_sf(
                dynamic_support, radial_sf, sf_edges, mode_weight
            )[sf_mask]
            gain = surface(models.loc[int(unit)], sf, tf)
            raw_overlap = float(np.sum(matched_power * gain**2))
            records.append({
                "rr100_index": int(unit),
                "case": CASES.get(int(unit), "RR100 population unit"),
                "rotation_deg": float(angle),
                "predicted_overlap_raw_arbitrary": raw_overlap,
                "predicted_overlap_per_total_supported_power": raw_overlap / max(denominator, 1e-30),
                "supported_dynamic_power": denominator,
                "orientation_weighted_supported_power": float(np.sum(matched_power)),
            })
    table = pd.DataFrame(records)
    table["predicted_overlap_raw_fraction_of_unit_max"] = table.groupby("rr100_index")[
        "predicted_overlap_raw_arbitrary"
    ].transform(lambda values: values / max(float(values.max()), 1e-30))
    table["predicted_overlap_fraction_of_unit_max"] = table.groupby("rr100_index")[
        "predicted_overlap_per_total_supported_power"
    ].transform(lambda values: values / max(float(values.max()), 1e-30))
    return table


def predeclare_predictions(proxy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for unit, frame in proxy.groupby("rr100_index", sort=False):
        ordered = frame.sort_values("predicted_overlap_per_total_supported_power", ascending=False)
        rows.append({
            "rr100_index": int(unit),
            "case": CASES[int(unit)],
            "prediction_basis": "orientation-aware SFxTF overlap divided by total dynamic power in fitted support for each exact rendered movie",
            "predicted_peak_rotation_deg": float(ordered.iloc[0]["rotation_deg"]),
            "predicted_trough_rotation_deg": float(ordered.iloc[-1]["rotation_deg"]),
            "predicted_rotation_order_high_to_low": ">".join(
                f"{value:g}" for value in ordered["rotation_deg"].to_numpy(float)
            ),
            "primary_observed_metric": "temporal SD of paired framewise FEM-minus-matched-zero response",
            "evaluation": "four-point rank association and peak-angle agreement; descriptive, not inferential",
        })
    return pd.DataFrame(rows)


def summarize_responses(
    responses: dict[float, dict[str, np.ndarray]], proxy: pd.DataFrame,
    source_frame_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = []
    time_records = []
    for unit_position, unit in enumerate(UNITS):
        for angle in ANGLES:
            zero_map = responses[float(angle)]["zero"][:, unit_position]
            fem_map = responses[float(angle)]["fem"][:, unit_position]
            center_y = int(zero_map.shape[-2] // 2)
            center_x = int(zero_map.shape[-1] // 2)
            zero = zero_map[:, center_y, center_x].astype(float)
            fem = fem_map[:, center_y, center_x].astype(float)
            delta = fem - zero
            records.append({
                "rr100_index": int(unit),
                "case": CASES[int(unit)],
                "rotation_deg": float(angle),
                "zero_mean_rate_hz": float(np.mean(zero)),
                "fem_mean_rate_hz": float(np.mean(fem)),
                "fem_minus_zero_mean_hz": float(np.mean(delta)),
                "fem_delta_temporal_sd_hz": float(np.std(delta)),
                "fem_delta_rms_hz": float(np.sqrt(np.mean(delta**2))),
                "fraction_delta_positive": float(np.mean(delta > 0)),
                "center_y": center_y,
                "center_x": center_x,
            })
            for time_index, source_frame in enumerate(source_frame_indices):
                time_records.append({
                    "rr100_index": int(unit),
                    "case": CASES[int(unit)],
                    "rotation_deg": float(angle),
                    "response_frame_index": int(time_index),
                    "source_movie_frame_index": int(source_frame),
                    "time_from_movie_start_ms": float(source_frame * 1000.0 / FRAME_RATE_HZ),
                    "zero_rate_hz": float(zero[time_index]),
                    "fem_rate_hz": float(fem[time_index]),
                    "fem_minus_zero_hz": float(delta[time_index]),
                })
    orientation = pd.DataFrame(records).merge(
        proxy, on=["rr100_index", "case", "rotation_deg"], validate="one_to_one"
    )
    orientation["observed_modulation_fraction_of_unit_max"] = orientation.groupby("rr100_index")[
        "fem_delta_temporal_sd_hz"
    ].transform(lambda values: values / max(float(values.max()), 1e-30))
    unit_rows = []
    for unit, frame in orientation.groupby("rr100_index", sort=False):
        frame = frame.sort_values("rotation_deg")
        predicted = frame["predicted_overlap_per_total_supported_power"].to_numpy(float)
        predicted_raw = frame["predicted_overlap_raw_arbitrary"].to_numpy(float)
        observed = frame["fem_delta_temporal_sd_hz"].to_numpy(float)
        predicted_peak = float(frame.iloc[int(np.argmax(predicted))]["rotation_deg"])
        observed_peak = float(frame.iloc[int(np.argmax(observed))]["rotation_deg"])
        pearson = pearsonr(predicted, observed)
        spearman = spearmanr(predicted, observed)
        pearson_raw = pearsonr(predicted_raw, observed)
        spearman_raw = spearmanr(predicted_raw, observed)
        unit_rows.append({
            "rr100_index": int(unit),
            "case": CASES[int(unit)],
            "predicted_peak_rotation_deg": predicted_peak,
            "observed_peak_rotation_deg": observed_peak,
            "peak_rotation_axial_error_deg": axial_delta_deg(predicted_peak, observed_peak),
            "four_point_pearson_r": float(pearson.statistic),
            "four_point_spearman_rho": float(spearman.statistic),
            "raw_numerator_four_point_pearson_r": float(pearson_raw.statistic),
            "raw_numerator_four_point_spearman_rho": float(spearman_raw.statistic),
            "observed_peak_modulation_sd_hz": float(np.max(observed)),
            "observed_minimum_modulation_sd_hz": float(np.min(observed)),
            "observed_peak_to_trough_ratio": float(np.max(observed) / max(np.min(observed), 1e-30)),
            "interpretation_scope": "descriptive four-orientation selected-unit diagnostic",
        })
    return orientation, pd.DataFrame(unit_rows), pd.DataFrame(time_records)


def montage(movie: np.ndarray, indices: np.ndarray) -> np.ndarray:
    separator = np.full((movie.shape[1], 2), np.nan)
    pieces = []
    for position, index in enumerate(indices):
        if position:
            pieces.append(separator)
        pieces.append(movie[int(index)])
    return np.concatenate(pieces, axis=1)


def plot_proxy_response(
    movies: dict[float, dict[str, np.ndarray]], orientation: pd.DataFrame,
    unit_summary: pd.DataFrame, out_base: Path, dpi: int,
) -> None:
    fig = plt.figure(figsize=(16.0, 13.0), constrained_layout=False)
    grid = fig.add_gridspec(
        5, 4, height_ratios=[0.62, 1, 1, 1, 1], left=0.055, right=0.975,
        top=0.925, bottom=0.06, hspace=0.62, wspace=0.34,
    )
    frame_indices = np.asarray([31, 63, 95, 127])
    all_frames = np.concatenate([movies[float(angle)]["fem"][frame_indices] for angle in ANGLES])
    vmin, vmax = np.percentile(all_frames, [1, 99])
    for column, angle in enumerate(ANGLES):
        axis = fig.add_subplot(grid[0, column])
        axis.imshow(montage(movies[float(angle)]["fem"], frame_indices), cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(f"{angle:g}° joint image + trace rotation", fontweight="bold", fontsize=10)
        axis.set_xticks([]); axis.set_yticks([])
    summary_index = unit_summary.set_index("rr100_index")
    for row, unit in enumerate(UNITS, start=1):
        frame = orientation.loc[orientation["rr100_index"].eq(int(unit))].sort_values("rotation_deg")
        angles = frame["rotation_deg"].to_numpy(float)
        predicted = frame["predicted_overlap_per_total_supported_power"].to_numpy(float)
        observed = frame["fem_delta_temporal_sd_hz"].to_numpy(float)
        predicted_norm = frame["predicted_overlap_fraction_of_unit_max"].to_numpy(float)
        observed_norm = frame["observed_modulation_fraction_of_unit_max"].to_numpy(float)
        summary = summary_index.loc[int(unit)]

        axis = fig.add_subplot(grid[row, 0])
        axis.plot(angles, predicted, color="#7B3294", lw=1.7, marker="o")
        axis.set_title(f"RR100 {unit}: {CASES[int(unit)]}\npredicted spectral overlap", fontsize=10)
        axis.set(xlabel="rotation (°)", ylabel="overlap / supported power")
        axis.set_xticks(ANGLES); axis.grid(color="0.91")

        axis = fig.add_subplot(grid[row, 1])
        axis.plot(angles, observed, color="#008837", lw=1.7, marker="o")
        axis.set_title("measured FEM modulation", fontsize=10)
        axis.set(xlabel="rotation (°)", ylabel="temporal SD (Hz)")
        axis.set_xticks(ANGLES); axis.grid(color="0.91")

        axis = fig.add_subplot(grid[row, 2])
        axis.plot(angles, predicted_norm, color="#7B3294", lw=1.6, marker="o", label="spectral proxy")
        axis.plot(angles, observed_norm, color="#008837", lw=1.6, marker="o", label="model response")
        axis.set_title(
            f"shape comparison\npredicted peak {summary['predicted_peak_rotation_deg']:.0f}°; "
            f"observed {summary['observed_peak_rotation_deg']:.0f}°",
            fontsize=9.5,
        )
        axis.set(xlabel="rotation (°)", ylabel="fraction of own maximum")
        axis.set_xticks(ANGLES); axis.set_ylim(0, 1.12); axis.grid(color="0.91")
        axis.legend(frameon=False, fontsize=7.5)

        axis = fig.add_subplot(grid[row, 3])
        axis.scatter(predicted_norm, observed_norm, s=46, c=[ANGLE_COLORS[float(a)] for a in angles])
        for x_value, y_value, angle in zip(predicted_norm, observed_norm, angles, strict=True):
            axis.annotate(f"{angle:g}°", (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
        axis.plot([0, 1], [0, 1], color="0.75", lw=0.8, ls="--")
        axis.set_xlim(0, 1.08); axis.set_ylim(0, 1.08)
        axis.set(xlabel="predicted fraction of max", ylabel="observed fraction of max")
        axis.set_title(
            f"four-point agreement\nSpearman ρ={summary['four_point_spearman_rho']:+.2f}; "
            f"peak error={summary['peak_rotation_axial_error_deg']:.0f}°",
            fontsize=9.5,
        )
        axis.grid(color="0.93")
    fig.suptitle(
        "Checkpoint 05: does orientation-aware spectral overlap predict the frozen model's rotation tuning?\n"
        "Same natural retinal movie at four orientations; no population inference",
        fontsize=14,
    )
    fig.text(
        0.055, 0.018,
        "Purple is derived from exact-movie SF×TF×orientation power and grating fits. Green is measured from the frozen model as the temporal SD of FEM minus its matched zero-gaze response.",
        fontsize=8, color="0.3",
    )
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_timecourses(timecourses: pd.DataFrame, orientation: pd.DataFrame, out_base: Path, dpi: int) -> None:
    fig, axes = plt.subplots(len(UNITS), len(ANGLES), figsize=(16.0, 10.5), constrained_layout=True)
    orientation_index = orientation.set_index(["rr100_index", "rotation_deg"])
    for row, unit in enumerate(UNITS):
        unit_data = timecourses.loc[timecourses["rr100_index"].eq(int(unit))]
        unit_values = np.concatenate([unit_data["zero_rate_hz"], unit_data["fem_rate_hz"]])
        low = max(0.0, float(np.min(unit_values)) - 0.05 * max(float(np.ptp(unit_values)), 1e-6))
        high = float(np.max(unit_values)) + 0.08 * max(float(np.ptp(unit_values)), 0.05)
        for column, angle in enumerate(ANGLES):
            axis = axes[row, column]
            frame = unit_data.loc[unit_data["rotation_deg"].eq(float(angle))]
            summary = orientation_index.loc[(int(unit), float(angle))]
            axis.plot(frame["time_from_movie_start_ms"], frame["fem_rate_hz"],
                      color=ANGLE_COLORS[float(angle)], lw=1.25, label="real FEM")
            axis.plot(frame["time_from_movie_start_ms"], frame["zero_rate_hz"],
                      color=ANGLE_COLORS[float(angle)], lw=0.95, ls="--", label="zero gaze")
            axis.set_ylim(low, high); axis.grid(color="0.92", lw=0.6)
            axis.set_title(
                f"RR100 {unit} · {angle:g}°\nmodulation SD={summary['fem_delta_temporal_sd_hz']:.3f} Hz",
                fontsize=9,
            )
            if row == len(UNITS) - 1:
                axis.set_xlabel("time from movie start (ms)")
            if column == 0:
                axis.set_ylabel(f"{CASES[int(unit)]}\nresponse (Hz)")
            if row == 0 and column == len(ANGLES) - 1:
                axis.legend(frameon=False, fontsize=7.5)
    fig.suptitle(
        "Raw frozen-RR100 response traces across the four matched orientation controls",
        fontsize=14,
    )
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if (args.out_dir / "manifest.json").exists():
        raise FileExistsError(f"Completed output already exists: {args.out_dir}")
    # Permit resumption after a pre-response failure. Any partial stimulus and
    # hypothesis files are deterministically regenerated before model loading.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    movies, construction = reconstruct_rotated_movies(args.input_dir)
    input_manifest = construction.pop("input_manifest")
    audit_table = construction.pop("audit_table")
    audit_table.to_csv(args.out_dir / "stimulus_rotation_audit.csv", index=False)
    np.savez_compressed(
        args.out_dir / "four_orientation_retinal_movies.npz",
        rotation_deg=ANGLES,
        **{
            f"angle_{int(angle):03d}_{condition}_movie": movies[float(angle)][condition]
            for angle in ANGLES for condition in ("zero", "fem")
        },
    )

    models_path = args.model_dir / "rr100_sf_tf_parametric_models.csv"
    orientation_path = args.f0_dir / "f0_orientation_scores.csv"
    models = pd.read_csv(models_path).set_index("rr100_index")
    orientation_scores = pd.read_csv(orientation_path)
    proxy = predicted_overlap_table(
        movies, float(input_manifest["model_ppd"]), models, orientation_scores
    )
    proxy.to_csv(args.out_dir / "four_orientation_predicted_overlap.csv", index=False)
    hypotheses = predeclare_predictions(proxy)
    # This file is intentionally written before any frozen-model responses are evaluated.
    hypotheses.to_csv(args.out_dir / "predeclared_four_orientation_predictions.csv", index=False)

    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    mapping = pd.read_csv(args.mapping_csv).sort_values("rr100_index")
    selected_channels = np.argmax(view.membership, axis=1)
    if not np.array_equal(selected_channels, mapping["canonical_channel"].to_numpy(int)):
        raise ValueError("RR100 movie-medoid view does not match the grating-fit unit mapping")
    scorer = CanonicalTwinScorer(device=args.device, batch_size=args.batch_size, empty_cache_every_batch=True)
    n_lags = int(scorer.common.N_LAGS)
    responses: dict[float, dict[str, np.ndarray]] = {}
    for angle in ANGLES:
        responses[float(angle)] = {}
        for condition in ("zero", "fem"):
            print(f"running angle={angle:g} condition={condition}", flush=True)
            responses[float(angle)][condition] = run_condition(
                scorer, view, movies[float(angle)][condition], UNITS, n_lags
            )
    shapes = {
        f"angle_{int(angle):03d}_{condition}": list(responses[float(angle)][condition].shape)
        for angle in ANGLES for condition in ("zero", "fem")
    }
    if len({tuple(shape) for shape in shapes.values()}) != 1:
        raise ValueError(f"Response shapes differ: {shapes}")
    n_response_frames = next(iter(responses.values()))["fem"].shape[0]
    source_frame_indices = np.arange(n_lags - 1, n_lags - 1 + n_response_frames)
    orientation, unit_summary, timecourses = summarize_responses(
        responses, proxy, source_frame_indices
    )
    orientation.to_csv(args.out_dir / "four_orientation_proxy_response_values.csv", index=False)
    unit_summary.to_csv(args.out_dir / "selected_unit_four_orientation_agreement.csv", index=False)
    timecourses.to_csv(args.out_dir / "four_orientation_response_timecourses.csv", index=False)
    np.savez_compressed(
        args.out_dir / "four_orientation_response_maps.npz",
        rr100_indices=UNITS,
        rotation_deg=ANGLES,
        source_movie_frame_indices=source_frame_indices,
        **{
            f"angle_{int(angle):03d}_{condition}_response": responses[float(angle)][condition]
            for angle in ANGLES for condition in ("zero", "fem")
        },
    )

    plot_proxy_response(
        movies, orientation, unit_summary,
        args.out_dir / "checkpoint_05_four_orientation_proxy_vs_response", args.dpi,
    )
    plot_timecourses(
        timecourses, orientation,
        args.out_dir / "checkpoint_05_four_orientation_raw_response_timecourses", args.dpi,
    )

    direct04_path = args.direct_04_dir / "selected_unit_direct_response_summary.csv"
    direct04 = pd.read_csv(direct04_path).set_index("rr100_index")
    angle0 = orientation.loc[orientation["rotation_deg"].eq(0.0)].set_index("rr100_index")
    angle90 = orientation.loc[orientation["rotation_deg"].eq(90.0)].set_index("rr100_index")
    direct_reproduction = {
        "angle0_max_abs_modulation_sd_error_hz": float(np.max(np.abs(
            angle0.loc[UNITS, "fem_delta_temporal_sd_hz"].to_numpy()
            - direct04.loc[UNITS, "original_fem_delta_temporal_sd_hz"].to_numpy()
        ))),
        "angle90_max_abs_modulation_sd_error_hz": float(np.max(np.abs(
            angle90.loc[UNITS, "fem_delta_temporal_sd_hz"].to_numpy()
            - direct04.loc[UNITS, "rotated_fem_delta_temporal_sd_hz"].to_numpy()
        ))),
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "selected-unit four-orientation spectral-proxy versus direct frozen-model response",
        "status": "four_orientation_selected_unit_checkpoint_complete_stop_before_population_summary",
        "scope": "same four preselected mechanism/control units and one natural-image/eye-trace pair",
        "stimulus_construction": {
            "angles_deg": ANGLES.tolist(),
            "method": "jointly rotate the 540x540 source patch and centered eye trace, then render each through the canonical 51x51 retinal helper",
            "oblique_patch_interpolation": "bilinear scipy.ndimage.rotate, reflect boundary on the large source patch; the central 51x51 retinal aperture never uses a post-render padding operation",
            "right_angle_rotation": "exact np.rot90 source-patch rotation",
            "baseline": "separate true-zero-gaze movie rendered from every rotated source patch",
        },
        "prediction_contract": {
            "saved_before_model_evaluation": True,
            "proxy": "exact rendered movie dynamic power weighted by separable grating-derived SFxTF gain squared and four-point orientation amplitude squared, divided by total dynamic power in fitted support; raw numerator retained as an audit diagnostic",
            "primary_response_metric": "temporal SD of paired framewise FEM-minus-matched-zero response",
            "agreement": "descriptive four-point Pearson/Spearman association and peak-angle error",
        },
        "checks": {
            **construction,
            "response_shapes": shapes,
            "maximum_zero_gaze_frame_difference": float(audit_table["zero_gaze_max_frame_difference"].max()),
            "direct_checkpoint_04_reproduction": direct_reproduction,
        },
        "inputs": {
            "checkpoint_01_manifest": file_identity(args.input_dir / "manifest.json"),
            "checkpoint_01_movies": file_identity(args.input_dir / "checkpoint_01_retinal_movies_and_power.npz"),
            "parametric_models": file_identity(models_path),
            "orientation_scores": file_identity(orientation_path),
            "rr100_mapping": file_identity(args.mapping_csv),
            "direct_checkpoint_04_summary": file_identity(direct04_path),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Checkpoint 05: four-orientation proxy versus frozen-model response\n\n"
        "This targeted checkpoint expands the prior 0/90-degree response test to 0, 45, 90, and 135 degrees. "
        "The source image patch and eye trace are jointly rotated before canonical retinal rendering, and every "
        "orientation receives its own true-zero-gaze baseline. Predictions were saved before model evaluation. "
        "The four-point correlations are descriptive selected-unit diagnostics, not population statistics.\n"
    )
    print(unit_summary.to_string(index=False))
    print(json.dumps(manifest["checks"], indent=2))


if __name__ == "__main__":
    main()
