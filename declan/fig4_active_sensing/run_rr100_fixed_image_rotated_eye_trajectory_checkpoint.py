#!/usr/bin/env python3
"""Checkpoint 06: fixed natural image with only the eye trajectory rotated.

The exact same unrotated source image is used at 0, 45, 90, and 135 degrees.
Only the centered eye trace rotates. Each resulting retinal movie is compared
with the same true-zero-gaze baseline, and its orientation-aware SFxTF proxy is
saved before evaluation of the frozen RR100 model.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.plot_rr100_kuang_input_power_checkpoint import (
    FRAME_RATE_HZ,
    render_retinal_movie,
    spectral_decomposition,
    support_summary,
)
from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import source_row_by_id
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import load_source_rows
from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.fig4_active_sensing.run_rr100_direct_fem_orientation_checkpoint import run_condition
from declan.fig4_active_sensing.run_rr100_four_orientation_proxy_response_checkpoint import (
    ANGLES,
    ANGLE_COLORS,
    CASES,
    UNITS,
    file_identity,
    montage,
    predeclare_predictions,
    predicted_overlap_table,
    rotate_trace,
    summarize_responses,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import CanonicalTwinScorer
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.redundancy_resolved_v1_population import load_population_view


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "outputs/fig4_active_sensing/rr100_kuang_input_power_checkpoint_01_v1"
MODEL_DIR = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1"
F0_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_f0_factorization_v1"
MAPPING = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_recorded_twin_gratings_check_v1/rr100_unit_mapping.csv"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_fixed_image_rotated_eye_trajectory_checkpoint_06_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--f0-dir", type=Path, default=F0_DIR)
    parser.add_argument("--mapping-csv", type=Path, default=MAPPING)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dpi", type=int, default=230)
    return parser.parse_args()


def construct_fixed_image_movies(
    input_dir: Path,
) -> tuple[np.ndarray, dict[float, np.ndarray], dict[float, dict[str, np.ndarray]], pd.DataFrame, dict[str, object]]:
    manifest = json.loads((input_dir / "manifest.json").read_text())
    archive = np.load(input_dir / "checkpoint_01_retinal_movies_and_power.npz")
    source_rows = load_source_rows(Path(manifest["inputs"]["source_windows"]["path"]))
    source_row_id = int(manifest["selected_image"]["source_row"])
    source_row = source_row_by_id(source_rows, source_row_id)
    patch, _ = _extract_patch(source_row, canvas_cache={}, patch_size_px=540)
    patch = np.asarray(patch, dtype=np.float32)
    original_trace = archive["centered_real_trace_xy_deg"].astype(np.float32)
    ppd = float(manifest["model_ppd"])
    fixed_zero = render_retinal_movie(patch, np.zeros_like(original_trace), ppd=ppd)
    traces: dict[float, np.ndarray] = {}
    movies: dict[float, dict[str, np.ndarray]] = {}
    audit_rows = []
    trace_rows = []
    for angle in ANGLES:
        trace = rotate_trace(original_trace, float(angle))
        zero = render_retinal_movie(patch, np.zeros_like(trace), ppd=ppd)
        fem = render_retinal_movie(patch, trace, ppd=ppd)
        traces[float(angle)] = trace
        movies[float(angle)] = {"zero": zero, "fem": fem}
        decomp = spectral_decomposition(fem, ppd=ppd, frame_rate_hz=FRAME_RATE_HZ)
        audit_rows.append({
            "trajectory_rotation_deg": float(angle),
            "image_rotation_deg": 0.0,
            "image_sha256_source_identity": file_identity(input_dir / "checkpoint_01_retinal_movies_and_power.npz")["sha256"],
            "trace_rms_radius_deg": float(np.sqrt(np.mean(np.sum(trace**2, axis=1)))),
            "trace_step_rms_deg": float(np.sqrt(np.mean(np.sum(np.diff(trace, axis=0) ** 2, axis=1)))),
            "zero_gaze_difference_from_fixed_max_abs_pixel": float(np.max(np.abs(zero - fixed_zero))),
            "zero_gaze_max_frame_difference": float(np.max(np.abs(zero - zero[:1]))),
            **support_summary(decomp),
        })
        for time_index, (x_deg, y_deg) in enumerate(trace):
            trace_rows.append({
                "trajectory_rotation_deg": float(angle),
                "image_rotation_deg": 0.0,
                "time_index": int(time_index),
                "time_ms": float(time_index * 1000.0 / FRAME_RATE_HZ),
                "eye_x_deg": float(x_deg),
                "eye_y_deg": float(y_deg),
            })
    angle0_error = float(np.max(np.abs(movies[0.0]["fem"] - archive["real_fem_movie"])))
    zero_saved_error = float(np.max(np.abs(fixed_zero - archive["zero_gaze_movie"])))
    if angle0_error != 0.0 or zero_saved_error != 0.0:
        raise ValueError("Fixed-image reconstruction did not exactly reproduce checkpoint 01")
    checks = {
        "source_row": source_row_id,
        "source_patch_shape": list(patch.shape),
        "image_rotation_deg_at_every_condition": 0.0,
        "angle0_saved_fem_max_abs_pixel_error": angle0_error,
        "saved_zero_gaze_max_abs_pixel_error": zero_saved_error,
        "maximum_zero_gaze_difference_across_trajectory_conditions": float(
            max(np.max(np.abs(payload["zero"] - fixed_zero)) for payload in movies.values())
        ),
        "input_manifest": manifest,
    }
    return patch, traces, movies, pd.DataFrame(audit_rows), {**checks, "trace_table": pd.DataFrame(trace_rows)}


def draw_trace_on_fixed_frame(axis: plt.Axes, fixed_frame: np.ndarray, trace: np.ndarray, ppd: float) -> None:
    center = (fixed_frame.shape[-1] - 1.0) / 2.0
    x_px = center + trace[:, 0] * ppd
    y_px = center - trace[:, 1] * ppd
    axis.imshow(fixed_frame, cmap="gray", vmin=np.percentile(fixed_frame, 1), vmax=np.percentile(fixed_frame, 99))
    axis.plot(x_px, y_px, color="#F0E442", lw=1.4, alpha=0.95)
    axis.scatter([x_px[0]], [y_px[0]], s=24, color="#009E73", edgecolor="black", lw=0.4, zorder=4)
    axis.scatter([x_px[-1]], [y_px[-1]], s=28, color="#D55E00", marker="X", edgecolor="black", lw=0.4, zorder=4)
    axis.set_xlim(center - 12, center + 12)
    axis.set_ylim(center + 12, center - 12)
    axis.set_xticks([]); axis.set_yticks([])


def plot_input_construction(
    movies: dict[float, dict[str, np.ndarray]], traces: dict[float, np.ndarray], ppd: float,
    audit: pd.DataFrame, out_base: Path, dpi: int,
) -> None:
    fig, axes = plt.subplots(2, len(ANGLES), figsize=(16.0, 7.2), constrained_layout=False)
    fig.subplots_adjust(left=0.045, right=0.98, top=0.83, bottom=0.08, hspace=0.30, wspace=0.16)
    fixed_frame = movies[0.0]["zero"][0]
    frame_indices = np.asarray([31, 63, 95, 127])
    all_frames = np.concatenate([movies[float(angle)]["fem"][frame_indices] for angle in ANGLES])
    vmin, vmax = np.percentile(all_frames, [1, 99])
    audit_index = audit.set_index("trajectory_rotation_deg")
    for column, angle in enumerate(ANGLES):
        draw_trace_on_fixed_frame(axes[0, column], fixed_frame, traces[float(angle)], ppd)
        axes[0, column].set_title(
            f"IMAGE FIXED AT 0°\nonly eye trajectory rotates to {angle:g}°",
            fontsize=10.5, fontweight="bold",
        )
        axes[1, column].imshow(
            montage(movies[float(angle)]["fem"], frame_indices), cmap="gray", vmin=vmin, vmax=vmax
        )
        axes[1, column].set_xticks([]); axes[1, column].set_yticks([])
        power_fraction = audit_index.loc[float(angle), "fraction_dynamic_power_in_joint_fitted_support"]
        axes[1, column].set_title(
            f"retinal movie from {angle:g}° trajectory\n{power_fraction:.1%} of dynamic power in fitted SF×TF support",
            fontsize=9.3,
        )
    fig.suptitle(
        "Checkpoint 06 manipulation: the natural image never rotates — only the eye trajectory does\n"
        "Yellow: eye path on the identical zero-gaze retinal image · green: start · orange: end",
        fontsize=14,
    )
    fig.text(
        0.5, 0.02,
        "The apparent image orientation is identical in all columns. Different retinal frames arise only because the rotated eye path samples translations in different directions.",
        ha="center", fontsize=8.5, color="0.25",
    )
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_proxy_response(
    movies: dict[float, dict[str, np.ndarray]], traces: dict[float, np.ndarray], ppd: float,
    orientation: pd.DataFrame, unit_summary: pd.DataFrame, out_base: Path, dpi: int,
) -> None:
    fig = plt.figure(figsize=(16.0, 13.1), constrained_layout=False)
    grid = fig.add_gridspec(
        5, 4, height_ratios=[0.72, 1, 1, 1, 1], left=0.055, right=0.975,
        top=0.90, bottom=0.06, hspace=0.62, wspace=0.34,
    )
    fixed_frame = movies[0.0]["zero"][0]
    for column, angle in enumerate(ANGLES):
        axis = fig.add_subplot(grid[0, column])
        draw_trace_on_fixed_frame(axis, fixed_frame, traces[float(angle)], ppd)
        axis.set_title(f"IMAGE FIXED · trajectory {angle:g}°", fontweight="bold", fontsize=10)
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
        axis.set_title(f"RR100 {unit}: {CASES[int(unit)]}\nprediction from fixed-image retinal power", fontsize=9.7)
        axis.set(xlabel="eye-trajectory rotation (°)", ylabel="overlap / supported power")
        axis.set_xticks(ANGLES); axis.grid(color="0.91")

        axis = fig.add_subplot(grid[row, 1])
        axis.plot(angles, observed, color="#008837", lw=1.7, marker="o")
        axis.set_title("measured response to rotated trajectory", fontsize=9.7)
        axis.set(xlabel="eye-trajectory rotation (°)", ylabel="FEM modulation SD (Hz)")
        axis.set_xticks(ANGLES); axis.grid(color="0.91")

        axis = fig.add_subplot(grid[row, 2])
        axis.plot(angles, predicted_norm, color="#7B3294", lw=1.6, marker="o", label="spectral proxy")
        axis.plot(angles, observed_norm, color="#008837", lw=1.6, marker="o", label="model response")
        axis.set_title(
            f"trajectory tuning shapes\npredicted peak {summary['predicted_peak_rotation_deg']:.0f}°; "
            f"observed {summary['observed_peak_rotation_deg']:.0f}°",
            fontsize=9.3,
        )
        axis.set(xlabel="eye-trajectory rotation (°)", ylabel="fraction of own maximum")
        axis.set_xticks(ANGLES); axis.set_ylim(0, 1.12); axis.grid(color="0.91")
        axis.legend(frameon=False, fontsize=7.5)

        axis = fig.add_subplot(grid[row, 3])
        axis.scatter(predicted_norm, observed_norm, s=46, c=[ANGLE_COLORS[float(a)] for a in angles])
        axis.plot([0, 1], [0, 1], color="0.75", lw=0.8, ls="--")
        axis.set_xlim(0, 1.08); axis.set_ylim(0, 1.08)
        axis.set(xlabel="predicted fraction of max", ylabel="observed fraction of max")
        axis.set_title(
            f"four trajectory angles\nSpearman ρ={summary['four_point_spearman_rho']:+.2f}; "
            f"peak error={summary['peak_rotation_axial_error_deg']:.0f}°",
            fontsize=9.3,
        )
        axis.grid(color="0.93")
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="none", markersize=5.5,
                   markerfacecolor=ANGLE_COLORS[float(angle)], markeredgecolor="none",
                   label=f"eye trace {angle:g}°")
            for angle in ANGLES
        ]
        axis.legend(handles=legend_handles, frameon=False, fontsize=6.5, ncol=2,
                    loc="lower right", columnspacing=0.7, handletextpad=0.3)
    fig.suptitle(
        "Checkpoint 06: can retinal-power overlap predict responses when only eye direction changes?\n"
        "NATURAL IMAGE FIXED AT 0° IN EVERY CONDITION · ONLY THE EYE TRAJECTORY ROTATES",
        fontsize=14,
    )
    fig.text(
        0.055, 0.018,
        "Purple: derived from each exact fixed-image retinal movie. Green: frozen-model temporal SD of FEM minus the identical zero-gaze response. Four selected units; no population inference.",
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
        values = np.concatenate([unit_data["zero_rate_hz"], unit_data["fem_rate_hz"]])
        low = max(0.0, float(np.min(values)) - 0.05 * max(float(np.ptp(values)), 1e-6))
        high = float(np.max(values)) + 0.08 * max(float(np.ptp(values)), 0.05)
        for column, angle in enumerate(ANGLES):
            axis = axes[row, column]
            frame = unit_data.loc[unit_data["rotation_deg"].eq(float(angle))]
            summary = orientation_index.loc[(int(unit), float(angle))]
            axis.plot(frame["time_from_movie_start_ms"], frame["fem_rate_hz"],
                      color=ANGLE_COLORS[float(angle)], lw=1.25, label="rotated eye trajectory")
            axis.plot(frame["time_from_movie_start_ms"], frame["zero_rate_hz"],
                      color="0.35", lw=0.95, ls="--", label="fixed-image zero gaze")
            axis.set_ylim(low, high); axis.grid(color="0.92", lw=0.6)
            axis.set_title(
                f"IMAGE 0° FIXED · EYE TRACE {angle:g}°\nmodulation SD={summary['fem_delta_temporal_sd_hz']:.3f} Hz",
                fontsize=8.7,
            )
            if row == len(UNITS) - 1:
                axis.set_xlabel("time from movie start (ms)")
            if column == 0:
                axis.set_ylabel(f"RR100 {unit}: {CASES[int(unit)]}\nresponse (Hz)")
            if row == 0 and column == len(ANGLES) - 1:
                axis.legend(frameon=False, fontsize=7.2)
    fig.suptitle(
        "Raw frozen-RR100 responses: identical natural image and zero-gaze baseline; only eye trajectory rotates",
        fontsize=14,
    )
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    patch, traces, movies, audit, construction = construct_fixed_image_movies(args.input_dir)
    input_manifest = construction.pop("input_manifest")
    trace_table = construction.pop("trace_table")
    audit.to_csv(args.out_dir / "fixed_image_rotated_trajectory_stimulus_audit.csv", index=False)
    trace_table.to_csv(args.out_dir / "rotated_eye_trajectories.csv", index=False)
    np.savez_compressed(
        args.out_dir / "fixed_image_rotated_trajectory_retinal_movies.npz",
        rotation_deg=ANGLES,
        fixed_source_patch=patch,
        **{
            f"trace_{int(angle):03d}_xy_deg": traces[float(angle)] for angle in ANGLES
        },
        **{
            f"trace_{int(angle):03d}_{condition}_movie": movies[float(angle)][condition]
            for angle in ANGLES for condition in ("zero", "fem")
        },
    )
    ppd = float(input_manifest["model_ppd"])
    plot_input_construction(
        movies, traces, ppd, audit,
        args.out_dir / "checkpoint_06_fixed_image_rotated_trajectory_inputs", args.dpi,
    )

    models_path = args.model_dir / "rr100_sf_tf_parametric_models.csv"
    orientation_path = args.f0_dir / "f0_orientation_scores.csv"
    models = pd.read_csv(models_path).set_index("rr100_index")
    orientation_scores = pd.read_csv(orientation_path)
    proxy = predicted_overlap_table(movies, ppd, models, orientation_scores)
    proxy = proxy.rename(columns={"rotation_deg": "trajectory_rotation_deg"})
    proxy["image_rotation_deg"] = 0.0
    proxy.to_csv(args.out_dir / "fixed_image_trajectory_predicted_overlap.csv", index=False)
    proxy_for_common = proxy.rename(columns={"trajectory_rotation_deg": "rotation_deg"})
    hypotheses = predeclare_predictions(proxy_for_common)
    hypotheses = hypotheses.rename(columns={
        "predicted_peak_rotation_deg": "predicted_peak_trajectory_rotation_deg",
        "predicted_trough_rotation_deg": "predicted_trough_trajectory_rotation_deg",
        "predicted_rotation_order_high_to_low": "predicted_trajectory_rotation_order_high_to_low",
    })
    hypotheses["image_rotation_deg"] = 0.0
    hypotheses.to_csv(args.out_dir / "predeclared_fixed_image_trajectory_predictions.csv", index=False)

    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    mapping = pd.read_csv(args.mapping_csv).sort_values("rr100_index")
    if not np.array_equal(np.argmax(view.membership, axis=1), mapping["canonical_channel"].to_numpy(int)):
        raise ValueError("RR100 movie-medoid view does not match grating-fit mapping")
    scorer = CanonicalTwinScorer(device=args.device, batch_size=args.batch_size, empty_cache_every_batch=True)
    n_lags = int(scorer.common.N_LAGS)
    responses: dict[float, dict[str, np.ndarray]] = {}
    for angle in ANGLES:
        responses[float(angle)] = {}
        for condition in ("zero", "fem"):
            print(f"running FIXED IMAGE 0 deg; eye trajectory {angle:g} deg; {condition}", flush=True)
            responses[float(angle)][condition] = run_condition(
                scorer, view, movies[float(angle)][condition], UNITS, n_lags
            )
    shapes = {
        f"trace_{int(angle):03d}_{condition}": list(responses[float(angle)][condition].shape)
        for angle in ANGLES for condition in ("zero", "fem")
    }
    if len({tuple(value) for value in shapes.values()}) != 1:
        raise ValueError(f"Response shapes differ: {shapes}")
    n_frames = next(iter(responses.values()))["fem"].shape[0]
    source_frames = np.arange(n_lags - 1, n_lags - 1 + n_frames)
    orientation, unit_summary, timecourses = summarize_responses(
        responses, proxy_for_common, source_frames
    )
    orientation = orientation.rename(columns={"rotation_deg": "trajectory_rotation_deg"})
    orientation["image_rotation_deg"] = 0.0
    unit_summary = unit_summary.rename(columns={
        "predicted_peak_rotation_deg": "predicted_peak_trajectory_rotation_deg",
        "observed_peak_rotation_deg": "observed_peak_trajectory_rotation_deg",
        "peak_rotation_axial_error_deg": "peak_trajectory_rotation_axial_error_deg",
    })
    unit_summary["image_rotation_deg"] = 0.0
    timecourses = timecourses.rename(columns={"rotation_deg": "trajectory_rotation_deg"})
    timecourses["image_rotation_deg"] = 0.0
    orientation.to_csv(args.out_dir / "fixed_image_trajectory_proxy_response_values.csv", index=False)
    unit_summary.to_csv(args.out_dir / "selected_unit_fixed_image_trajectory_agreement.csv", index=False)
    timecourses.to_csv(args.out_dir / "fixed_image_trajectory_response_timecourses.csv", index=False)
    np.savez_compressed(
        args.out_dir / "fixed_image_trajectory_response_maps.npz",
        rr100_indices=UNITS, trajectory_rotation_deg=ANGLES, source_movie_frame_indices=source_frames,
        **{
            f"trace_{int(angle):03d}_{condition}_response": responses[float(angle)][condition]
            for angle in ANGLES for condition in ("zero", "fem")
        },
    )

    plot_orientation = orientation.rename(columns={"trajectory_rotation_deg": "rotation_deg"})
    plot_summary = unit_summary.rename(columns={
        "predicted_peak_trajectory_rotation_deg": "predicted_peak_rotation_deg",
        "observed_peak_trajectory_rotation_deg": "observed_peak_rotation_deg",
        "peak_trajectory_rotation_axial_error_deg": "peak_rotation_axial_error_deg",
    })
    plot_time = timecourses.rename(columns={"trajectory_rotation_deg": "rotation_deg"})
    plot_proxy_response(
        movies, traces, ppd, plot_orientation, plot_summary,
        args.out_dir / "checkpoint_06_fixed_image_trajectory_proxy_vs_response_clear", args.dpi,
    )
    plot_timecourses(
        plot_time, plot_orientation,
        args.out_dir / "checkpoint_06_fixed_image_trajectory_raw_response_timecourses", args.dpi,
    )

    zero_responses = [responses[float(angle)]["zero"] for angle in ANGLES]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "selected-unit fixed-image rotated-eye-trajectory spectral proxy versus frozen response",
        "status": "fixed_image_rotated_trajectory_selected_unit_checkpoint_complete_stop_before_population_summary",
        "manipulation_contract": {
            "image": "same unrotated 540x540 source patch and same zero-gaze 51x51 retinal image at every condition",
            "image_rotation_deg": 0.0,
            "eye_trajectory_rotations_deg": ANGLES.tolist(),
            "baseline": "identical true-zero-gaze movie for every trajectory rotation",
            "question": "does changing eye-movement direction alone alter power overlap and direct frozen-model modulation as predicted?",
        },
        "prediction_contract": {
            "saved_before_model_evaluation": True,
            "proxy": "each exact fixed-image retinal movie's orientation-aware SFxTF overlap divided by total dynamic power in fitted support",
            "primary_response_metric": "temporal SD of paired FEM-minus-identical-zero-gaze response",
            "agreement": "descriptive four-point association and peak trajectory-angle error",
        },
        "checks": {
            **construction,
            "response_shapes": shapes,
            "maximum_zero_gaze_response_difference_across_conditions_hz": float(
                max(np.max(np.abs(value - zero_responses[0])) for value in zero_responses)
            ),
            "trace_rms_radius_relative_range": float(
                np.ptp(audit["trace_rms_radius_deg"]) / max(float(audit["trace_rms_radius_deg"].mean()), 1e-30)
            ),
            "trace_step_rms_relative_range": float(
                np.ptp(audit["trace_step_rms_deg"]) / max(float(audit["trace_step_rms_deg"].mean()), 1e-30)
            ),
        },
        "inputs": {
            "checkpoint_01_manifest": file_identity(args.input_dir / "manifest.json"),
            "checkpoint_01_movies": file_identity(args.input_dir / "checkpoint_01_retinal_movies_and_power.npz"),
            "parametric_models": file_identity(models_path),
            "orientation_scores": file_identity(orientation_path),
            "rr100_mapping": file_identity(args.mapping_csv),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Checkpoint 06: fixed image, rotated eye trajectory\n\n"
        "The natural image is unrotated and identical in all four conditions. Only the eye trajectory rotates "
        "through 0, 45, 90, and 135 degrees. Each model response is paired with the same true-zero-gaze baseline. "
        "The figures label the fixed image and rotated trajectory explicitly. Predictions were saved before frozen-model "
        "evaluation; four-point agreement is descriptive and no population conclusion is made.\n"
    )
    print(unit_summary.to_string(index=False))
    print(json.dumps(manifest["checks"], indent=2))


if __name__ == "__main__":
    main()
