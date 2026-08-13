#!/usr/bin/env python3
"""Checkpoint 08: preregistered multi-image trajectory-direction generalization.

Six natural image/eye-trace pairs are chosen without neural responses by a
feature-space maximin rule seeded by the original checkpoint image. For each
fixed image, only its eye trajectory rotates through 0, 45, 90, and 135
degrees. Exact-movie spectral predictions are saved before all 100 frozen RR100
responses are evaluated.
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from declan.active_sensing_movie_information.plot_rr100_kuang_input_power_checkpoint import (
    FRAME_RATE_HZ,
    render_retinal_movie,
    spectral_decomposition,
    support_summary,
)
from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import (
    one_trace_from_source,
    source_row_by_id,
)
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
    predicted_overlap_table,
    rotate_trace,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import CanonicalTwinScorer
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.redundancy_resolved_v1_population import load_population_view


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "outputs/active_sensing_movie_information/temporal_remapping/backimage_rr100_retiming_medium_figure4pool_n16_t32_fullgrid_cuda0_v1"
SOURCE_CSV = ROOT / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
MODEL_DIR = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1"
F0_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_f0_factorization_v1"
MAPPING = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_recorded_twin_gratings_check_v1/rr100_unit_mapping.csv"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_multiimage_trajectory_generalization_checkpoint_08_v1"
ALL_UNITS = np.arange(100, dtype=np.int64)
FEATURES = (
    "image_patch_rms_contrast",
    "image_gradient_energy",
    "image_orientation_coherence",
    "image_high_freq_power_fraction",
    "anisotropy",
    "speed_mean_deg_s",
)
N_IMAGES = 6
ANCHOR_IMAGE_INDEX = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--f0-dir", type=Path, default=F0_DIR)
    parser.add_argument("--mapping-csv", type=Path, default=MAPPING)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--n-images", type=int, default=N_IMAGES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def select_images(feature_table: pd.DataFrame, n_images: int) -> pd.DataFrame:
    eligible = feature_table.loc[
        feature_table["image_feature_ok"].astype(bool)
        & feature_table["image_patch_fraction_inside_image"].ge(0.99)
    ].copy().reset_index(drop=True)
    transformed = eligible.loc[:, FEATURES].astype(float).copy()
    transformed["image_gradient_energy"] = np.log1p(transformed["image_gradient_energy"])
    z = (transformed - transformed.mean()) / transformed.std(ddof=0)
    anchor_positions = np.flatnonzero(eligible["image_index"].to_numpy(int) == ANCHOR_IMAGE_INDEX)
    if len(anchor_positions) != 1:
        raise ValueError("Expected exactly one anchor image")
    chosen = [int(anchor_positions[0])]
    min_distances = [np.nan]
    while len(chosen) < int(n_images):
        distance = np.sqrt(
            np.sum((z.to_numpy()[:, None, :] - z.to_numpy()[np.asarray(chosen)][None, :, :]) ** 2, axis=2)
        )
        minimum = distance.min(axis=1)
        minimum[chosen] = -np.inf
        position = int(np.argmax(minimum))
        chosen.append(position)
        min_distances.append(float(minimum[position]))
    selected = eligible.iloc[chosen].copy()
    selected.insert(0, "selection_order", np.arange(1, len(selected) + 1))
    selected.insert(1, "selection_role", ["original_checkpoint_anchor"] + ["feature_space_maximin"] * (len(selected) - 1))
    selected.insert(2, "minimum_z_feature_distance_at_selection", min_distances)
    selected["selection_rule"] = (
        "anchor image_index=9, then greedy maximin in z-scored contrast, log-gradient, orientation coherence, "
        "high-frequency power, trace anisotropy, and mean speed"
    )
    return selected


def build_movies(
    selected: pd.DataFrame, source_rows: pd.DataFrame, ppd: float,
) -> tuple[dict[int, dict[float, dict[str, np.ndarray]]], dict[int, np.ndarray], dict[int, np.ndarray], pd.DataFrame]:
    movies = {}
    patches = {}
    original_traces = {}
    audit_rows = []
    for _, image_row in selected.iterrows():
        image_index = int(image_row["image_index"])
        source_row_id = int(image_row["source_row"])
        source_row = source_row_by_id(source_rows, source_row_id)
        patch, _ = _extract_patch(source_row, canvas_cache={}, patch_size_px=540)
        patch = np.asarray(patch, dtype=np.float32)
        trace = one_trace_from_source(
            source_rows, source_row_id, n_timepoints=128, bin_seconds=1.0 / FRAME_RATE_HZ
        ).astype(np.float32)
        trace -= np.mean(trace, axis=0, keepdims=True)
        zero = render_retinal_movie(patch, np.zeros_like(trace), ppd=ppd)
        patches[image_index] = patch
        original_traces[image_index] = trace
        movies[image_index] = {}
        for angle in ANGLES:
            rotated = rotate_trace(trace, float(angle))
            fem = render_retinal_movie(patch, rotated, ppd=ppd)
            movies[image_index][float(angle)] = {"zero": zero, "fem": fem}
            decomp = spectral_decomposition(fem, ppd=ppd, frame_rate_hz=FRAME_RATE_HZ)
            audit_rows.append({
                "image_index": image_index,
                "source_row": source_row_id,
                "session": str(image_row["session"]),
                "image_rotation_deg": 0.0,
                "trajectory_rotation_deg": float(angle),
                "trace_rms_radius_deg": float(np.sqrt(np.mean(np.sum(rotated**2, axis=1)))),
                "trace_step_rms_deg": float(np.sqrt(np.mean(np.sum(np.diff(rotated, axis=0)**2, axis=1)))),
                "zero_gaze_max_frame_difference": float(np.max(np.abs(zero - zero[:1]))),
                **support_summary(decomp),
            })
    return movies, patches, original_traces, pd.DataFrame(audit_rows)


def plot_inputs(
    selected: pd.DataFrame, movies: dict[int, dict[float, dict[str, np.ndarray]]],
    traces: dict[int, np.ndarray], ppd: float, out_base: Path, dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.8), constrained_layout=True)
    for axis, (_, row) in zip(axes.ravel(), selected.iterrows(), strict=True):
        image_index = int(row["image_index"])
        frame = movies[image_index][0.0]["zero"][0]
        trace = traces[image_index]
        center = (frame.shape[-1] - 1) / 2
        x = center + trace[:, 0] * ppd; y = center - trace[:, 1] * ppd
        axis.imshow(frame, cmap="gray", vmin=np.percentile(frame, 1), vmax=np.percentile(frame, 99))
        axis.plot(x, y, color="#F0E442", lw=1.25)
        axis.scatter([x[0]], [y[0]], color="#009E73", s=22, edgecolor="black", lw=0.4)
        axis.scatter([x[-1]], [y[-1]], color="#D55E00", marker="X", s=26, edgecolor="black", lw=0.4)
        axis.set_xlim(center - 17, center + 17); axis.set_ylim(center + 17, center - 17)
        axis.set_xticks([]); axis.set_yticks([])
        axis.set_title(
            f"selection {int(row['selection_order'])}: image {image_index} · {row['session']}\n"
            f"image fixed; yellow eye trace rotates in the test", fontsize=9.5,
        )
    fig.suptitle(
        "Checkpoint 08 preregistered generalization set\n"
        "Six fixed natural images selected without neural responses by feature-space maximin sampling",
        fontsize=14,
    )
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi); fig.savefig(out_base.with_suffix(".pdf")); plt.close(fig)


def plot_retinal_movies(
    selected: pd.DataFrame, movies: dict[int, dict[float, dict[str, np.ndarray]]], out_base: Path, dpi: int,
) -> None:
    fig, axes = plt.subplots(len(selected), len(ANGLES), figsize=(15.5, 12.5), constrained_layout=True)
    frames = np.asarray([31, 63, 95, 127])
    for row_index, (_, row) in enumerate(selected.iterrows()):
        image_index = int(row["image_index"])
        all_frames = np.concatenate([movies[image_index][float(a)]["fem"][frames] for a in ANGLES])
        vmin, vmax = np.percentile(all_frames, [1, 99])
        for column, angle in enumerate(ANGLES):
            axis = axes[row_index, column]
            axis.imshow(montage(movies[image_index][float(angle)]["fem"], frames), cmap="gray", vmin=vmin, vmax=vmax)
            axis.set_xticks([]); axis.set_yticks([])
            axis.set_title(f"image {image_index} FIXED · eye trajectory {angle:g}°", fontsize=8.5)
    fig.suptitle("Exact retinal movies: image content stays fixed within each row; only eye-trajectory direction changes", fontsize=13)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi); fig.savefig(out_base.with_suffix(".pdf")); plt.close(fig)


def response_tables(
    responses: dict[str, np.ndarray], movies: dict[int, dict[float, dict[str, np.ndarray]]],
    proxy: pd.DataFrame, source_frames: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    times = []
    for image_index in movies:
        zero = responses[f"image_{image_index:02d}_zero"][:, :, 0, 0].astype(float)
        for angle in ANGLES:
            fem = responses[f"image_{image_index:02d}_trace_{int(angle):03d}_fem"][:, :, 0, 0].astype(float)
            delta = fem - zero
            for unit in ALL_UNITS:
                values = delta[:, int(unit)]
                rows.append({
                    "image_index": int(image_index), "rr100_index": int(unit),
                    "image_rotation_deg": 0.0, "trajectory_rotation_deg": float(angle),
                    "zero_mean_rate_hz": float(np.mean(zero[:, int(unit)])),
                    "fem_mean_rate_hz": float(np.mean(fem[:, int(unit)])),
                    "fem_minus_zero_mean_hz": float(np.mean(values)),
                    "fem_delta_temporal_sd_hz": float(np.std(values)),
                    "fem_delta_rms_hz": float(np.sqrt(np.mean(values**2))),
                })
            for time_index, source_frame in enumerate(source_frames):
                for unit in UNITS:
                    times.append({
                        "image_index": int(image_index), "rr100_index": int(unit),
                        "trajectory_rotation_deg": float(angle), "response_frame_index": int(time_index),
                        "source_movie_frame_index": int(source_frame),
                        "time_from_movie_start_ms": float(source_frame * 1000.0 / FRAME_RATE_HZ),
                        "zero_rate_hz": float(zero[time_index, int(unit)]),
                        "fem_rate_hz": float(fem[time_index, int(unit)]),
                        "fem_minus_zero_hz": float(delta[time_index, int(unit)]),
                    })
    values = pd.DataFrame(rows).merge(
        proxy, on=["image_index", "rr100_index", "trajectory_rotation_deg"], validate="one_to_one"
    )
    values["predicted_fraction_of_image_unit_max"] = values.groupby(["image_index", "rr100_index"])[
        "predicted_overlap_per_total_supported_power"
    ].transform(lambda x: x / max(float(x.max()), 1e-30))
    values["observed_fraction_of_image_unit_max"] = values.groupby(["image_index", "rr100_index"])[
        "fem_delta_temporal_sd_hz"
    ].transform(lambda x: x / max(float(x.max()), 1e-30))
    agreement_rows = []
    for (image_index, unit), frame in values.groupby(["image_index", "rr100_index"]):
        frame = frame.sort_values("trajectory_rotation_deg")
        p = frame["predicted_overlap_per_total_supported_power"].to_numpy(float)
        o = frame["fem_delta_temporal_sd_hz"].to_numpy(float)
        p_peak = float(frame.iloc[int(np.argmax(p))]["trajectory_rotation_deg"])
        o_peak = float(frame.iloc[int(np.argmax(o))]["trajectory_rotation_deg"])
        agreement_rows.append({
            "image_index": int(image_index), "rr100_index": int(unit),
            "four_point_pearson_r": float(pearsonr(p, o).statistic),
            "four_point_spearman_rho": float(spearmanr(p, o).statistic),
            "predicted_peak_trajectory_rotation_deg": p_peak,
            "observed_peak_trajectory_rotation_deg": o_peak,
            "peak_trajectory_rotation_axial_error_deg": float(abs((p_peak - o_peak + 90) % 180 - 90)),
            "observed_peak_modulation_sd_hz": float(o.max()),
            "observed_peak_to_trough_ratio": float(o.max() / max(o.min(), 1e-30)),
        })
    return values, pd.DataFrame(agreement_rows), pd.DataFrame(times)


def plot_selected_summary(
    selected: pd.DataFrame, values: pd.DataFrame, agreement: pd.DataFrame, out_base: Path, dpi: int,
) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(16.0, 12.0), constrained_layout=True)
    images = selected["image_index"].to_numpy(int)
    image_labels = [str(value) for value in images]
    cmap = plt.get_cmap("tab10")
    for row, unit in enumerate(UNITS):
        frame = values.loc[values["rr100_index"].eq(int(unit))]
        predicted = frame.pivot(index="image_index", columns="trajectory_rotation_deg", values="predicted_overlap_per_total_supported_power").reindex(images)
        observed = frame.pivot(index="image_index", columns="trajectory_rotation_deg", values="fem_delta_temporal_sd_hz").reindex(images)
        im = axes[row, 0].imshow(predicted, aspect="auto", cmap="magma")
        axes[row, 0].set_title(f"RR100 {unit}: {CASES[int(unit)]}\npredicted overlap", fontsize=9.5)
        fig.colorbar(im, ax=axes[row, 0], fraction=0.046, pad=0.03)
        im = axes[row, 1].imshow(observed, aspect="auto", cmap="viridis")
        axes[row, 1].set_title("measured modulation SD (Hz)", fontsize=9.5)
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.03)
        for col in (0, 1):
            axes[row, col].set_xticks(range(4), [f"{a:g}°" for a in ANGLES])
            axes[row, col].set_yticks(range(len(images)), image_labels)
            axes[row, col].set_xlabel("eye-trajectory rotation; image fixed")
            axes[row, col].set_ylabel("image index")
        for image_position, image_index in enumerate(images):
            group = frame.loc[frame["image_index"].eq(int(image_index))]
            axes[row, 2].scatter(
                group["predicted_fraction_of_image_unit_max"], group["observed_fraction_of_image_unit_max"],
                s=28, color=cmap(image_position), alpha=0.85,
            )
        rho = spearmanr(frame["predicted_fraction_of_image_unit_max"], frame["observed_fraction_of_image_unit_max"]).statistic
        axes[row, 2].plot([0, 1], [0, 1], color="0.75", ls="--", lw=0.8)
        axes[row, 2].set(xlim=(0, 1.05), ylim=(0, 1.05), xlabel="predicted fraction of image maximum",
                         ylabel="observed fraction of image maximum")
        axes[row, 2].set_title(f"all 24 image×trajectory points\nSpearman ρ={rho:+.2f}", fontsize=9.5)
        axes[row, 2].grid(color="0.93")
        ag = agreement.loc[agreement["rr100_index"].eq(int(unit))].set_index("image_index").reindex(images)
        axes[row, 3].bar(np.arange(len(images)), ag["four_point_spearman_rho"], color=[cmap(i) for i in range(len(images))])
        axes[row, 3].axhline(0, color="0.4", lw=0.8)
        axes[row, 3].set_xticks(range(len(images)), image_labels)
        axes[row, 3].set_ylim(-1.08, 1.08)
        axes[row, 3].set(xlabel="image index", ylabel="within-image Spearman ρ")
        axes[row, 3].set_title(f"generalization across images\nmedian ρ={ag['four_point_spearman_rho'].median():+.2f}", fontsize=9.5)
        axes[row, 3].grid(axis="y", color="0.92")
    handles = [Line2D([0], [0], marker="o", ls="none", color=cmap(i), label=f"image {idx}") for i, idx in enumerate(images)]
    axes[0, 2].legend(handles=handles, frameon=False, fontsize=6.5, ncol=2, loc="lower right")
    fig.suptitle(
        "Checkpoint 08: fixed-image eye-direction predictions across six preregistered natural image/trace pairs\n"
        "Selected examples only; all 100 RR100 responses are cached for population analysis",
        fontsize=14,
    )
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi); fig.savefig(out_base.with_suffix(".pdf")); plt.close(fig)


def plot_raw_selected_timecourses(
    selected: pd.DataFrame, timecourses: pd.DataFrame, values: pd.DataFrame, out_pdf: Path,
) -> None:
    value_index = values.set_index(["image_index", "rr100_index", "trajectory_rotation_deg"])
    with PdfPages(out_pdf) as pdf:
        for _, image_row in selected.iterrows():
            image_index = int(image_row["image_index"])
            fig, axes = plt.subplots(4, 4, figsize=(15.5, 10.0), constrained_layout=True)
            for row, unit in enumerate(UNITS):
                unit_data = timecourses.loc[
                    timecourses["image_index"].eq(image_index) & timecourses["rr100_index"].eq(int(unit))
                ]
                raw = np.concatenate([unit_data["zero_rate_hz"], unit_data["fem_rate_hz"]])
                low = max(0.0, float(raw.min()) - 0.05 * max(float(np.ptp(raw)), 1e-6))
                high = float(raw.max()) + 0.08 * max(float(np.ptp(raw)), 0.05)
                for column, angle in enumerate(ANGLES):
                    axis = axes[row, column]
                    frame = unit_data.loc[unit_data["trajectory_rotation_deg"].eq(float(angle))]
                    modulation = value_index.loc[(image_index, int(unit), float(angle)), "fem_delta_temporal_sd_hz"]
                    axis.plot(frame["time_from_movie_start_ms"], frame["fem_rate_hz"], color=ANGLE_COLORS[float(angle)], lw=1.2)
                    axis.plot(frame["time_from_movie_start_ms"], frame["zero_rate_hz"], color="0.35", ls="--", lw=0.9)
                    axis.set_ylim(low, high); axis.grid(color="0.92", lw=0.6)
                    axis.set_title(f"IMAGE {image_index} FIXED · EYE TRACE {angle:g}°\nSD={modulation:.3f} Hz", fontsize=8.2)
                    if row == 3: axis.set_xlabel("time (ms)")
                    if column == 0: axis.set_ylabel(f"RR100 {unit}\nresponse (Hz)")
            fig.suptitle(f"Raw selected-unit response traces — fixed image {image_index}; only eye direction rotates", fontsize=13)
            pdf.savefig(fig); plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    feature_path = args.run_dir / "image_feature_table.csv"
    feature_table = pd.read_csv(feature_path)
    selected = select_images(feature_table, args.n_images)
    selection_columns = [
        "selection_order", "selection_role", "minimum_z_feature_distance_at_selection", "selection_rule",
        "image_index", "source_row", "session", *FEATURES,
    ]
    selected[selection_columns].to_csv(args.out_dir / "predeclared_generalization_image_selection.csv", index=False)
    source_rows = load_source_rows(args.source_csv)
    from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import _load_twin_common
    ppd = float(_load_twin_common().PPD)
    movies, patches, traces, audit = build_movies(selected, source_rows, ppd)
    audit.to_csv(args.out_dir / "multiimage_trajectory_stimulus_audit.csv", index=False)
    plot_inputs(selected, movies, traces, ppd, args.out_dir / "checkpoint_08_preregistered_images_and_traces", args.dpi)
    plot_retinal_movies(selected, movies, args.out_dir / "checkpoint_08_exact_retinal_movies", args.dpi)

    movie_payload = {}
    for image_index in movies:
        movie_payload[f"image_{image_index:02d}_zero"] = movies[image_index][0.0]["zero"]
        for angle in ANGLES:
            movie_payload[f"image_{image_index:02d}_trace_{int(angle):03d}_fem"] = movies[image_index][float(angle)]["fem"]
    np.savez_compressed(args.out_dir / "multiimage_trajectory_retinal_movies.npz", **movie_payload)

    models_path = args.model_dir / "rr100_sf_tf_parametric_models.csv"
    orientation_path = args.f0_dir / "f0_orientation_scores.csv"
    models = pd.read_csv(models_path).set_index("rr100_index")
    orientation_scores = pd.read_csv(orientation_path)
    proxy_frames = []
    for image_index in movies:
        proxy = predicted_overlap_table(movies[image_index], ppd, models, orientation_scores, units=ALL_UNITS)
        proxy["image_index"] = int(image_index)
        proxy = proxy.rename(columns={"rotation_deg": "trajectory_rotation_deg"})
        proxy_frames.append(proxy)
    proxy_all = pd.concat(proxy_frames, ignore_index=True)
    proxy_all.to_csv(args.out_dir / "predeclared_multiimage_trajectory_predictions_all_rr100.csv", index=False)

    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    mapping = pd.read_csv(args.mapping_csv).sort_values("rr100_index")
    if not np.array_equal(np.argmax(view.membership, axis=1), mapping["canonical_channel"].to_numpy(int)):
        raise ValueError("RR100 movie-medoid view does not match grating-fit mapping")
    scorer = CanonicalTwinScorer(device=args.device, batch_size=args.batch_size, empty_cache_every_batch=True)
    n_lags = int(scorer.common.N_LAGS)
    responses = {}
    for movie_id, movie in movie_payload.items():
        print(f"running all RR100: {movie_id}", flush=True)
        responses[movie_id] = run_condition(scorer, view, movie, ALL_UNITS, n_lags)
    shapes = {key: list(value.shape) for key, value in responses.items()}
    if len({tuple(value) for value in shapes.values()}) != 1:
        raise ValueError(f"Response shapes differ: {shapes}")
    source_frames = np.arange(n_lags - 1, n_lags - 1 + next(iter(responses.values())).shape[0])
    np.savez_compressed(
        args.out_dir / "multiimage_all_rr100_response_cache.npz", rr100_indices=ALL_UNITS,
        source_movie_frame_indices=source_frames, **responses,
    )
    values, agreement, timecourses = response_tables(responses, movies, proxy_all, source_frames)
    values.to_csv(args.out_dir / "multiimage_trajectory_proxy_response_values_all_rr100.csv", index=False)
    agreement.to_csv(args.out_dir / "multiimage_unit_image_four_angle_agreement_all_rr100.csv", index=False)
    timecourses.to_csv(args.out_dir / "multiimage_selected_unit_response_timecourses.csv", index=False)
    selected_values = values.loc[values["rr100_index"].isin(UNITS)]
    selected_agreement = agreement.loc[agreement["rr100_index"].isin(UNITS)]
    plot_selected_summary(
        selected, selected_values, selected_agreement,
        args.out_dir / "checkpoint_08_selected_unit_generalization", args.dpi,
    )
    plot_raw_selected_timecourses(
        selected, timecourses, selected_values,
        args.out_dir / "checkpoint_08_selected_unit_raw_response_timecourses.pdf",
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "preregistered six-image fixed-image rotated-eye-trajectory generalization",
        "status": "checkpoint_08_complete_ready_for_population_summary",
        "selection_contract": {
            "selected_before_neural_responses": True, "n_images": int(len(selected)),
            "anchor_image_index": ANCHOR_IMAGE_INDEX, "features": list(FEATURES),
            "algorithm": str(selected["selection_rule"].iloc[0]),
            "selected_image_indices": selected["image_index"].astype(int).tolist(),
        },
        "manipulation_contract": {
            "image_fixed_within_pair": True, "image_rotation_deg": 0.0,
            "trajectory_rotations_deg": ANGLES.tolist(), "one_identical_zero_gaze_baseline_per_image": True,
        },
        "cache_contract": {"rr100_units": 100, "unique_movies": len(movie_payload), "response_shapes": shapes},
        "checks": {
            "maximum_zero_gaze_frame_difference": float(audit["zero_gaze_max_frame_difference"].max()),
            "maximum_within_image_trace_rms_radius_range_deg": float(audit.groupby("image_index")["trace_rms_radius_deg"].agg(np.ptp).max()),
            "maximum_within_image_trace_step_rms_range_deg": float(audit.groupby("image_index")["trace_step_rms_deg"].agg(np.ptp).max()),
        },
        "inputs": {
            "image_feature_table": file_identity(feature_path), "source_windows": file_identity(args.source_csv),
            "parametric_models": file_identity(models_path), "orientation_scores": file_identity(orientation_path),
            "mapping": file_identity(args.mapping_csv),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Checkpoint 08: multi-image eye-trajectory generalization\n\n"
        "Six image/trace pairs were selected before model evaluation by feature-space maximin sampling seeded with the "
        "original checkpoint image. Each image remains fixed while its eye trajectory rotates to four angles. Exact-movie "
        "predictions were saved before all 100 RR100 responses were evaluated. This checkpoint preserves selected-unit raw "
        "traces and defers population inference to checkpoint 09.\n"
    )
    print(selected[["selection_order", "image_index", "source_row", "session"]].to_string(index=False))
    print(selected_agreement.groupby("rr100_index")["four_point_spearman_rho"].agg(["median", "mean", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
