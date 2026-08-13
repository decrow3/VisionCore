#!/usr/bin/env python3
"""Checkpoint 07: rotate only the image and build a consolidated RR100 cache.

The original eye trajectory remains fixed at 0 degrees while the source image
rotates through 0, 45, 90, and 135 degrees. In the same frozen-model run, all
100 RR100 outputs are retained for every unique movie from the joint-rotation,
trajectory-only, and image-only factorial arms completed so far.
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
    rotate_patch,
    summarize_responses,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import CanonicalTwinScorer
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.redundancy_resolved_v1_population import load_population_view


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "outputs/fig4_active_sensing/rr100_kuang_input_power_checkpoint_01_v1"
JOINT_DIR = ROOT / "outputs/fig4_active_sensing/rr100_four_orientation_proxy_response_checkpoint_05_v2"
TRACE_DIR = ROOT / "outputs/fig4_active_sensing/rr100_fixed_image_rotated_eye_trajectory_checkpoint_06_v1"
MODEL_DIR = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1"
F0_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_f0_factorization_v1"
MAPPING = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_recorded_twin_gratings_check_v1/rr100_unit_mapping.csv"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_image_only_rotation_checkpoint_07_v2"
ALL_UNITS = np.arange(100, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--joint-dir", type=Path, default=JOINT_DIR)
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--f0-dir", type=Path, default=F0_DIR)
    parser.add_argument("--mapping-csv", type=Path, default=MAPPING)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dpi", type=int, default=230)
    return parser.parse_args()


def construct_image_only_movies(
    input_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[float, dict[str, np.ndarray]], pd.DataFrame, dict[str, object]]:
    manifest = json.loads((input_dir / "manifest.json").read_text())
    archive = np.load(input_dir / "checkpoint_01_retinal_movies_and_power.npz")
    source_rows = load_source_rows(Path(manifest["inputs"]["source_windows"]["path"]))
    source_row_id = int(manifest["selected_image"]["source_row"])
    source_row = source_row_by_id(source_rows, source_row_id)
    patch, _ = _extract_patch(source_row, canvas_cache={}, patch_size_px=540)
    patch = np.asarray(patch, dtype=np.float32)
    trace = archive["centered_real_trace_xy_deg"].astype(np.float32)
    ppd = float(manifest["model_ppd"])
    movies: dict[float, dict[str, np.ndarray]] = {}
    audit_rows = []
    for angle in ANGLES:
        transformed_patch = rotate_patch(patch, float(angle))
        zero = render_retinal_movie(transformed_patch, np.zeros_like(trace), ppd=ppd)
        fem = render_retinal_movie(transformed_patch, trace, ppd=ppd)
        movies[float(angle)] = {"zero": zero, "fem": fem}
        decomp = spectral_decomposition(fem, ppd=ppd, frame_rate_hz=FRAME_RATE_HZ)
        audit_rows.append({
            "image_rotation_deg": float(angle),
            "trajectory_rotation_deg": 0.0,
            "source_patch_sd": float(np.std(transformed_patch)),
            "eye_trace_max_abs_difference_from_original_deg": 0.0,
            "zero_gaze_max_frame_difference": float(np.max(np.abs(zero - zero[:1]))),
            **support_summary(decomp),
        })
    if float(np.max(np.abs(movies[0.0]["fem"] - archive["real_fem_movie"]))) != 0.0:
        raise ValueError("0-degree FEM reconstruction does not reproduce checkpoint 01")
    if float(np.max(np.abs(movies[0.0]["zero"] - archive["zero_gaze_movie"]))) != 0.0:
        raise ValueError("0-degree zero reconstruction does not reproduce checkpoint 01")
    return patch, trace, movies, pd.DataFrame(audit_rows), {
        "source_row": source_row_id,
        "source_patch_shape": list(patch.shape),
        "trajectory_rotation_deg_at_every_condition": 0.0,
        "angle0_saved_fem_max_abs_pixel_error": 0.0,
        "angle0_saved_zero_max_abs_pixel_error": 0.0,
        "input_manifest": manifest,
    }


def load_prior_movies(joint_dir: Path, trace_dir: Path) -> tuple[dict[float, dict[str, np.ndarray]], dict[float, dict[str, np.ndarray]]]:
    joint_archive = np.load(joint_dir / "four_orientation_retinal_movies.npz")
    trace_archive = np.load(trace_dir / "fixed_image_rotated_trajectory_retinal_movies.npz")
    joint = {
        float(angle): {
            condition: joint_archive[f"angle_{int(angle):03d}_{condition}_movie"].astype(np.float32)
            for condition in ("zero", "fem")
        }
        for angle in ANGLES
    }
    trace = {
        float(angle): {
            condition: trace_archive[f"trace_{int(angle):03d}_{condition}_movie"].astype(np.float32)
            for condition in ("zero", "fem")
        }
        for angle in ANGLES
    }
    return joint, trace


def unique_movie_bank(
    joint: dict[float, dict[str, np.ndarray]], trace: dict[float, dict[str, np.ndarray]],
    image: dict[float, dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, float]]:
    unique: dict[str, np.ndarray] = {}
    rows = []
    for angle in ANGLES:
        unique[f"zero_image_{int(angle):03d}"] = image[float(angle)]["zero"]
    unique["fem_base_000"] = image[0.0]["fem"]
    for angle in ANGLES[1:]:
        unique[f"fem_joint_{int(angle):03d}"] = joint[float(angle)]["fem"]
        unique[f"fem_trace_only_{int(angle):03d}"] = trace[float(angle)]["fem"]
        unique[f"fem_image_only_{int(angle):03d}"] = image[float(angle)]["fem"]
    for manipulation in ("joint_image_and_trajectory", "trajectory_only", "image_only"):
        for angle in ANGLES:
            if manipulation == "trajectory_only":
                zero_id = "zero_image_000"
            else:
                zero_id = f"zero_image_{int(angle):03d}"
            if angle == 0.0:
                fem_id = "fem_base_000"
            elif manipulation == "joint_image_and_trajectory":
                fem_id = f"fem_joint_{int(angle):03d}"
            elif manipulation == "trajectory_only":
                fem_id = f"fem_trace_only_{int(angle):03d}"
            else:
                fem_id = f"fem_image_only_{int(angle):03d}"
            rows.extend([
                {"manipulation": manipulation, "angle_deg": float(angle), "condition": "zero", "unique_movie_id": zero_id},
                {"manipulation": manipulation, "angle_deg": float(angle), "condition": "fem", "unique_movie_id": fem_id},
            ])
    checks = {
        "angle0_joint_vs_trace_fem_max_abs_pixel_error": float(np.max(np.abs(joint[0.0]["fem"] - trace[0.0]["fem"]))),
        "angle0_joint_vs_image_fem_max_abs_pixel_error": float(np.max(np.abs(joint[0.0]["fem"] - image[0.0]["fem"]))),
        "trajectory_only_zero_across_angles_max_abs_pixel_error": float(
            max(np.max(np.abs(trace[float(angle)]["zero"] - trace[0.0]["zero"])) for angle in ANGLES)
        ),
        "joint_vs_image_zero_max_abs_pixel_error": float(
            max(np.max(np.abs(joint[float(angle)]["zero"] - image[float(angle)]["zero"])) for angle in ANGLES)
        ),
    }
    if max(checks.values()) > 0.01:
        raise ValueError(f"Movie deduplication checks failed: {checks}")
    return unique, pd.DataFrame(rows), checks


def draw_fixed_trace(axis: plt.Axes, frame: np.ndarray, trace: np.ndarray, ppd: float) -> None:
    center = (frame.shape[-1] - 1.0) / 2.0
    x_px = center + trace[:, 0] * ppd
    y_px = center - trace[:, 1] * ppd
    axis.imshow(frame, cmap="gray", vmin=np.percentile(frame, 1), vmax=np.percentile(frame, 99))
    axis.plot(x_px, y_px, color="#F0E442", lw=1.4)
    axis.scatter([x_px[0]], [y_px[0]], s=24, color="#009E73", edgecolor="black", lw=0.4)
    axis.scatter([x_px[-1]], [y_px[-1]], s=28, color="#D55E00", marker="X", edgecolor="black", lw=0.4)
    axis.set_xlim(center - 12, center + 12); axis.set_ylim(center + 12, center - 12)
    axis.set_xticks([]); axis.set_yticks([])


def plot_inputs(movies: dict[float, dict[str, np.ndarray]], trace: np.ndarray, ppd: float,
                audit: pd.DataFrame, out_base: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.2), constrained_layout=False)
    fig.subplots_adjust(left=0.045, right=0.98, top=0.83, bottom=0.08, hspace=0.30, wspace=0.16)
    frames = np.asarray([31, 63, 95, 127])
    all_frames = np.concatenate([movies[float(angle)]["fem"][frames] for angle in ANGLES])
    vmin, vmax = np.percentile(all_frames, [1, 99])
    audit_index = audit.set_index("image_rotation_deg")
    for column, angle in enumerate(ANGLES):
        draw_fixed_trace(axes[0, column], movies[float(angle)]["zero"][0], trace, ppd)
        axes[0, column].set_title(
            f"EYE TRAJECTORY FIXED AT 0°\nonly image rotates to {angle:g}°", fontweight="bold", fontsize=10.5
        )
        axes[1, column].imshow(montage(movies[float(angle)]["fem"], frames), cmap="gray", vmin=vmin, vmax=vmax)
        axes[1, column].set_xticks([]); axes[1, column].set_yticks([])
        support = audit_index.loc[float(angle), "fraction_dynamic_power_in_joint_fitted_support"]
        axes[1, column].set_title(
            f"retinal movie from image {angle:g}°\n{support:.1%} of dynamic power in fitted SF×TF support", fontsize=9.3
        )
    fig.suptitle(
        "Checkpoint 07 manipulation: the eye trajectory never rotates — only the natural image does\n"
        "Yellow: identical eye path · green: start · orange: end", fontsize=14,
    )
    fig.text(0.5, 0.02, "Every column uses the same eye coordinates; changes in retinal movies arise only from rotating image content.",
             ha="center", fontsize=8.5, color="0.25")
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi); fig.savefig(out_base.with_suffix(".pdf")); plt.close(fig)


def plot_proxy_response(orientation: pd.DataFrame, summary: pd.DataFrame, out_base: Path, dpi: int) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(16.0, 11.8), constrained_layout=True)
    summary_index = summary.set_index("rr100_index")
    for row, unit in enumerate(UNITS):
        frame = orientation.loc[orientation["rr100_index"].eq(int(unit))].sort_values("rotation_deg")
        angles = frame["rotation_deg"].to_numpy(float)
        predicted = frame["predicted_overlap_per_total_supported_power"].to_numpy(float)
        observed = frame["fem_delta_temporal_sd_hz"].to_numpy(float)
        pn = frame["predicted_overlap_fraction_of_unit_max"].to_numpy(float)
        on = frame["observed_modulation_fraction_of_unit_max"].to_numpy(float)
        info = summary_index.loc[int(unit)]
        axes[row, 0].plot(angles, predicted, color="#7B3294", marker="o")
        axes[row, 0].set_title(f"RR100 {unit}: {CASES[int(unit)]}\nprediction from image-only retinal power", fontsize=9.5)
        axes[row, 0].set_ylabel("overlap / supported power")
        axes[row, 1].plot(angles, observed, color="#008837", marker="o")
        axes[row, 1].set_title("measured response; eye path fixed", fontsize=9.5)
        axes[row, 1].set_ylabel("FEM modulation SD (Hz)")
        axes[row, 2].plot(angles, pn, color="#7B3294", marker="o", label="spectral proxy")
        axes[row, 2].plot(angles, on, color="#008837", marker="o", label="model response")
        axes[row, 2].set_ylim(0, 1.12)
        axes[row, 2].set_title(
            f"image tuning shapes\npredicted peak {info['predicted_peak_rotation_deg']:.0f}°; observed {info['observed_peak_rotation_deg']:.0f}°",
            fontsize=9.2,
        )
        if row == 0: axes[row, 2].legend(frameon=False, fontsize=7)
        axes[row, 3].scatter(pn, on, s=46, c=[ANGLE_COLORS[float(a)] for a in angles])
        axes[row, 3].plot([0, 1], [0, 1], color="0.75", ls="--", lw=0.8)
        axes[row, 3].set_xlim(0, 1.08); axes[row, 3].set_ylim(0, 1.08)
        axes[row, 3].set_title(
            f"four image angles\nSpearman ρ={info['four_point_spearman_rho']:+.2f}; peak error={info['peak_rotation_axial_error_deg']:.0f}°",
            fontsize=9.2,
        )
        handles = [Line2D([0], [0], marker="o", ls="none", markersize=5, color=ANGLE_COLORS[float(a)], label=f"image {a:g}°") for a in ANGLES]
        axes[row, 3].legend(handles=handles, frameon=False, fontsize=6.4, ncol=2, loc="lower right")
        for column in range(4):
            axes[row, column].set_xticks(ANGLES if column < 3 else [0, 0.5, 1])
            axes[row, column].grid(color="0.92", lw=0.6)
            if column < 3: axes[row, column].set_xlabel("image rotation (°); eye trace fixed 0°")
        axes[row, 3].set_xlabel("predicted fraction of max"); axes[row, 3].set_ylabel("observed fraction of max")
    fig.suptitle(
        "Checkpoint 07: IMAGE ROTATES while EYE TRAJECTORY REMAINS FIXED AT 0°\n"
        "Does exact retinal-power overlap predict the frozen model's image-orientation dependence?", fontsize=14,
    )
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi); fig.savefig(out_base.with_suffix(".pdf")); plt.close(fig)


def plot_timecourses(timecourses: pd.DataFrame, orientation: pd.DataFrame, out_base: Path, dpi: int) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(16.0, 10.5), constrained_layout=True)
    idx = orientation.set_index(["rr100_index", "rotation_deg"])
    for row, unit in enumerate(UNITS):
        unit_data = timecourses.loc[timecourses["rr100_index"].eq(int(unit))]
        values = np.concatenate([unit_data["zero_rate_hz"], unit_data["fem_rate_hz"]])
        low = max(0.0, float(values.min()) - 0.05 * max(float(np.ptp(values)), 1e-6))
        high = float(values.max()) + 0.08 * max(float(np.ptp(values)), 0.05)
        for column, angle in enumerate(ANGLES):
            axis = axes[row, column]
            frame = unit_data.loc[unit_data["rotation_deg"].eq(float(angle))]
            modulation = idx.loc[(int(unit), float(angle)), "fem_delta_temporal_sd_hz"]
            axis.plot(frame["time_from_movie_start_ms"], frame["fem_rate_hz"], color=ANGLE_COLORS[float(angle)], lw=1.25)
            axis.plot(frame["time_from_movie_start_ms"], frame["zero_rate_hz"], color="0.35", lw=0.95, ls="--")
            axis.set_ylim(low, high); axis.grid(color="0.92", lw=0.6)
            axis.set_title(f"IMAGE {angle:g}° · EYE TRACE 0° FIXED\nmodulation SD={modulation:.3f} Hz", fontsize=8.7)
            if row == 3: axis.set_xlabel("time from movie start (ms)")
            if column == 0: axis.set_ylabel(f"RR100 {unit}: {CASES[int(unit)]}\nresponse (Hz)")
    fig.suptitle("Raw frozen-RR100 responses: only image orientation changes; eye trajectory remains identical", fontsize=14)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi); fig.savefig(out_base.with_suffix(".pdf")); plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    patch, fixed_trace, image_movies, audit, construction = construct_image_only_movies(args.input_dir)
    input_manifest = construction.pop("input_manifest")
    ppd = float(input_manifest["model_ppd"])
    audit.to_csv(args.out_dir / "image_only_stimulus_audit.csv", index=False)
    np.savez_compressed(
        args.out_dir / "image_only_retinal_movies.npz", rotation_deg=ANGLES, fixed_trace_xy_deg=fixed_trace,
        fixed_source_patch=patch,
        **{f"image_{int(a):03d}_{c}_movie": image_movies[float(a)][c] for a in ANGLES for c in ("zero", "fem")},
    )
    plot_inputs(image_movies, fixed_trace, ppd, audit, args.out_dir / "checkpoint_07_image_rotates_eye_fixed_inputs", args.dpi)

    models_path = args.model_dir / "rr100_sf_tf_parametric_models.csv"
    orientation_path = args.f0_dir / "f0_orientation_scores.csv"
    models = pd.read_csv(models_path).set_index("rr100_index")
    orientation_scores = pd.read_csv(orientation_path)
    proxy_all = predicted_overlap_table(image_movies, ppd, models, orientation_scores, units=ALL_UNITS)
    proxy_all.to_csv(args.out_dir / "image_only_predicted_overlap_all_rr100.csv", index=False)
    proxy_selected = proxy_all.loc[proxy_all["rr100_index"].isin(UNITS)].copy()
    hypotheses = predeclare_predictions(proxy_selected)
    hypotheses["trajectory_rotation_deg"] = 0.0
    hypotheses.to_csv(args.out_dir / "predeclared_image_only_predictions.csv", index=False)

    joint_movies, trace_movies = load_prior_movies(args.joint_dir, args.trace_dir)
    unique_movies, condition_map, dedup_checks = unique_movie_bank(joint_movies, trace_movies, image_movies)
    condition_map.to_csv(args.out_dir / "consolidated_condition_to_unique_movie.csv", index=False)
    np.savez_compressed(args.out_dir / "consolidated_unique_retinal_movies.npz", **unique_movies)

    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    mapping = pd.read_csv(args.mapping_csv).sort_values("rr100_index")
    if not np.array_equal(np.argmax(view.membership, axis=1), mapping["canonical_channel"].to_numpy(int)):
        raise ValueError("RR100 movie-medoid view does not match grating-fit mapping")
    scorer = CanonicalTwinScorer(device=args.device, batch_size=args.batch_size, empty_cache_every_batch=True)
    n_lags = int(scorer.common.N_LAGS)
    responses = {}
    for movie_id, movie in unique_movies.items():
        print(f"running all RR100: {movie_id}", flush=True)
        responses[movie_id] = run_condition(scorer, view, movie, ALL_UNITS, n_lags)
    shapes = {key: list(value.shape) for key, value in responses.items()}
    if len({tuple(value) for value in shapes.values()}) != 1:
        raise ValueError(f"Response shapes differ: {shapes}")
    source_frames = np.arange(n_lags - 1, n_lags - 1 + next(iter(responses.values())).shape[0])
    np.savez_compressed(
        args.out_dir / "consolidated_all_rr100_response_cache.npz", rr100_indices=ALL_UNITS,
        source_movie_frame_indices=source_frames, **responses,
    )

    selected_responses = {
        float(angle): {
            "zero": responses[f"zero_image_{int(angle):03d}"][:, UNITS],
            "fem": responses["fem_base_000" if angle == 0.0 else f"fem_image_only_{int(angle):03d}"][:, UNITS],
        }
        for angle in ANGLES
    }
    orientation, summary, timecourses = summarize_responses(selected_responses, proxy_selected, source_frames)
    orientation.to_csv(args.out_dir / "selected_unit_image_only_proxy_response_values.csv", index=False)
    summary.to_csv(args.out_dir / "selected_unit_image_only_agreement.csv", index=False)
    timecourses.to_csv(args.out_dir / "selected_unit_image_only_response_timecourses.csv", index=False)
    plot_proxy_response(orientation, summary, args.out_dir / "checkpoint_07_image_only_proxy_vs_response", args.dpi)
    plot_timecourses(timecourses, orientation, args.out_dir / "checkpoint_07_image_only_raw_response_timecourses", args.dpi)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "image-only rotation selected-unit checkpoint plus consolidated all-RR100 response cache",
        "status": "checkpoint_07_complete_ready_for_generalization_and_population",
        "manipulation_contract": {
            "image_rotations_deg": ANGLES.tolist(), "eye_trajectory_rotation_deg": 0.0,
            "baseline": "separate zero-gaze movie for each rotated image",
        },
        "cache_contract": {
            "rr100_units": 100, "nominal_conditions": int(condition_map.shape[0]),
            "unique_movies_inferred": len(unique_movies),
            "manipulations": sorted(condition_map["manipulation"].unique().tolist()),
        },
        "checks": {**construction, **dedup_checks, "response_shapes": shapes},
        "inputs": {
            "checkpoint_01_manifest": file_identity(args.input_dir / "manifest.json"),
            "joint_movies": file_identity(args.joint_dir / "four_orientation_retinal_movies.npz"),
            "trajectory_only_movies": file_identity(args.trace_dir / "fixed_image_rotated_trajectory_retinal_movies.npz"),
            "parametric_models": file_identity(models_path), "orientation_scores": file_identity(orientation_path),
            "mapping": file_identity(args.mapping_csv),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Checkpoint 07: image-only rotation and consolidated RR100 cache\n\n"
        "The eye trajectory remains fixed while the source image rotates. Predictions were saved before model evaluation. "
        "The same run retains all 100 RR100 responses for 14 deduplicated movies spanning joint, trajectory-only, and "
        "image-only rotations. Selected-unit correlations are descriptive; population analysis is deferred.\n"
    )
    print(summary.to_string(index=False)); print(json.dumps(manifest["cache_contract"], indent=2))


if __name__ == "__main__":
    main()
