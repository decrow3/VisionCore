#!/usr/bin/env python3
"""Checkpoint 10: integrated first-pass Kuang-style RR100 figure.

Combines fixed-retina grating sensitivity, FEM-created retinal power, their
predicted overlap, raw success/failure examples, and population validation. The
figure distinguishes prediction of broad modulation scale across images from
the failed prediction of preferred eye-movement direction within an image.
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
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from declan.active_sensing_movie_information.plot_rr100_kuang_input_power_checkpoint import (
    FRAME_RATE_HZ,
    radialize_power,
    spectral_decomposition,
)
from declan.fig4_active_sensing.make_rr100_kuang_unit_overlap_checkpoint import surface
from declan.fig4_active_sensing.make_rr100_orientation_aware_overlap_checkpoint import interpolate_and_smooth
from declan.fig4_active_sensing.polish_rr100_kuang_unit_overlap_checkpoint import kuang_colormap
from declan.fig4_active_sensing.run_rr100_four_orientation_proxy_response_checkpoint import file_identity, montage


ROOT = Path(__file__).resolve().parents[2]
SOURCE08 = ROOT / "outputs/fig4_active_sensing/rr100_multiimage_trajectory_generalization_checkpoint_08_v1"
SOURCE09 = ROOT / "outputs/fig4_active_sensing/rr100_population_proxy_validation_checkpoint_09_v1"
MODEL_DIR = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_integrated_kuang_first_pass_checkpoint_10_v1"
SF_MIN, SF_MAX = 1.0, float(8.0 * np.sqrt(2.0))
TF_MIN, TF_MAX = 0.5, 32.0
DISPLAY_FLOOR_DB = -20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source08-dir", type=Path, default=SOURCE08)
    parser.add_argument("--source09-dir", type=Path, default=SOURCE09)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def db(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    normalized = values / max(float(np.max(values)), 1e-30)
    return 10.0 * np.log10(np.maximum(normalized, 10 ** (DISPLAY_FLOOR_DB / 10)))


def setup_map_axis(axis: plt.Axes, show_y: bool = True) -> None:
    sf_ticks = np.asarray([1, 2, 4, 8], dtype=float)
    tf_ticks = np.asarray([1, 2, 4, 8, 16, 32], dtype=float)
    axis.set_xticks(np.log2(sf_ticks), [f"{value:g}" for value in sf_ticks])
    axis.set_yticks(np.log2(tf_ticks), [f"{value:g}" for value in tf_ticks] if show_y else [])
    axis.set_xlabel("spatial frequency (cpd)", fontsize=8)
    if show_y: axis.set_ylabel("temporal frequency (Hz)", fontsize=8)
    axis.tick_params(labelsize=7)


def population_maps(
    movies: np.lib.npyio.NpzFile, selected: pd.DataFrame, models: pd.DataFrame,
    cohort_units: np.ndarray, ppd: float,
) -> dict[str, np.ndarray]:
    power_maps = []
    sf = tf = None
    for image_index in selected["image_index"].astype(int):
        movie = movies[f"image_{image_index:02d}_trace_000_fem"].astype(float)
        decomp = spectral_decomposition(movie, ppd=ppd, frame_rate_hz=FRAME_RATE_HZ)
        radial = radialize_power(decomp, ppd=ppd, frame_size=movie.shape[-1])
        sf_all = radial["sf_centers_cpd"]
        tf_all = decomp["temporal_frequency_hz"]
        sf_mask = (sf_all >= SF_MIN) & (sf_all <= SF_MAX)
        tf_mask = (tf_all >= TF_MIN) & (tf_all <= TF_MAX)
        sf = sf_all[sf_mask]; tf = tf_all[tf_mask]
        annular = radial["dynamic_radial_power"][np.ix_(sf_mask, tf_mask)] * radial["spatial_mode_count"][sf_mask, None]
        power_maps.append(annular / max(float(annular.sum()), 1e-30))
    mean_power = np.mean(power_maps, axis=0)
    sensitivities = []
    model_index = models.set_index("rr100_index")
    for unit in cohort_units:
        gain2 = surface(model_index.loc[int(unit)], sf, tf) ** 2
        sensitivities.append(gain2 / max(float(gain2.max()), 1e-30))
    static_sensitivity = np.mean(sensitivities, axis=0)
    predicted_overlap = static_sensitivity * mean_power
    dense_sf = np.geomspace(float(sf.min()), float(sf.max()), 181)
    dense_tf = np.geomspace(float(tf.min()), float(tf.max()), 241)
    return {
        "sf": dense_sf, "tf": dense_tf,
        "static_sensitivity": interpolate_and_smooth(sf, tf, static_sensitivity, dense_sf, dense_tf),
        "mean_fem_power": interpolate_and_smooth(sf, tf, mean_power, dense_sf, dense_tf),
        "predicted_overlap": interpolate_and_smooth(sf, tf, predicted_overlap, dense_sf, dense_tf),
        "native_sf": sf, "native_tf": tf, "native_static_sensitivity": static_sensitivity,
        "native_mean_fem_power": mean_power, "native_predicted_overlap": predicted_overlap,
    }


def paired_heatmap(values: pd.DataFrame, unit: int, images: list[int]) -> np.ndarray:
    frame = values.loc[values["rr100_index"].eq(int(unit))]
    predicted = frame.pivot(index="image_index", columns="trajectory_rotation_deg", values="predicted_fraction_of_image_unit_max").reindex(images)
    observed = frame.pivot(index="image_index", columns="trajectory_rotation_deg", values="observed_fraction_of_image_unit_max").reindex(images)
    return np.hstack([predicted.to_numpy(float), np.full((len(images), 1), np.nan), observed.to_numpy(float)])


def plot_main(
    movies: np.lib.npyio.NpzFile, selected: pd.DataFrame, maps: dict[str, np.ndarray],
    unit_summary: pd.DataFrame, agreement: pd.DataFrame, values: pd.DataFrame,
    examples: pd.DataFrame, out_base: Path, dpi: int,
) -> dict[str, float]:
    cohort = unit_summary.loc[unit_summary["passes_primary_fit_quality"]]
    cohort_units = cohort["rr100_index"]
    pair = agreement.loc[agreement["rr100_index"].isin(cohort_units)]
    cohort_values = values.loc[values["rr100_index"].isin(cohort_units)]
    global_amp_rho = float(spearmanr(
        cohort_values["predicted_response_sd_proxy_arbitrary"], cohort_values["fem_delta_temporal_sd_hz"]
    ).statistic)
    median_within_unit_amp = float(cohort["all_24_amplitude_proxy_spearman_rho"].median())
    global_direction_rho = float(spearmanr(
        cohort_values["predicted_fraction_of_image_unit_max"], cohort_values["observed_fraction_of_image_unit_max"]
    ).statistic)
    median_unit_direction = float(cohort["median_within_image_spearman_rho"].median())

    fig = plt.figure(figsize=(16.5, 13.2), constrained_layout=False)
    grid = fig.add_gridspec(3, 4, left=0.055, right=0.975, top=0.91, bottom=0.055, hspace=0.46, wspace=0.34)
    cmap = kuang_colormap(); norm = Normalize(vmin=DISPLAY_FLOOR_DB, vmax=0)
    dense_sf = maps["sf"]; dense_tf = maps["tf"]
    sf_edges = np.r_[np.log2(dense_sf[0]), (np.log2(dense_sf[:-1]) + np.log2(dense_sf[1:])) / 2, np.log2(dense_sf[-1])]
    tf_edges = np.r_[np.log2(dense_tf[0]), (np.log2(dense_tf[:-1]) + np.log2(dense_tf[1:])) / 2, np.log2(dense_tf[-1])]

    axis = fig.add_subplot(grid[0, 0])
    image_index = int(selected.iloc[0]["image_index"])
    zero = movies[f"image_{image_index:02d}_zero"]
    fem = movies[f"image_{image_index:02d}_trace_000_fem"]
    frames = np.asarray([31, 63, 95, 127])
    combined = np.concatenate([zero[0], np.full((zero.shape[1], 3), np.nan), montage(fem, frames)], axis=1)
    axis.imshow(combined, cmap="gray", vmin=np.percentile(fem, 1), vmax=np.percentile(fem, 99))
    axis.set_xticks([]); axis.set_yticks([])
    axis.set_title("A  Static natural image + FEM\nleft: zero gaze · right: translated retinal frames", loc="left", fontweight="bold", fontsize=9.5)
    axis.text(0.5, -0.12, "Only translations create nonzero temporal power", transform=axis.transAxes, ha="center", fontsize=8)

    map_axes = []
    for column, key, title in (
        (1, "static_sensitivity", "B  Fixed-retina grating sensitivity\nnonzero TF comes from drifting gratings"),
        (2, "mean_fem_power", "C  FEM-created retinal power\nmean of six preregistered images"),
        (3, "predicted_overlap", "D  Kuang-style predicted drive\nsensitivity × FEM power"),
    ):
        axis = fig.add_subplot(grid[0, column]); map_axes.append(axis)
        axis.pcolormesh(sf_edges, tf_edges, db(maps[key]).T, cmap=cmap, norm=norm, shading="auto")
        setup_map_axis(axis, show_y=(column == 1))
        axis.set_title(title, loc="left", fontweight="bold", fontsize=9.5)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=map_axes, orientation="horizontal", fraction=0.045, pad=0.14)
    cbar.set_label("dB relative to each panel maximum (display-smoothed)", fontsize=8)

    axis = fig.add_subplot(grid[1, 0])
    static_sf = maps["static_sensitivity"].sum(axis=1); overlap_sf = maps["predicted_overlap"].sum(axis=1)
    axis.plot(dense_sf, static_sf / static_sf.max(), color="black", lw=1.6, label="fixed-retina sensitivity")
    axis.plot(dense_sf, overlap_sf / overlap_sf.max(), color="#D55E00", lw=1.6, label="FEM-weighted prediction")
    axis.set_xscale("log", base=2); axis.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    axis.set(title="E  Population SF marginal", xlabel="spatial frequency (cpd)", ylabel="fraction of maximum")
    axis.legend(frameon=False, fontsize=7.5); axis.grid(color="0.92")

    axis = fig.add_subplot(grid[1, 1])
    static_tf = maps["static_sensitivity"].sum(axis=0); overlap_tf = maps["predicted_overlap"].sum(axis=0)
    axis.plot(dense_tf, static_tf / static_tf.max(), color="black", lw=1.6, label="fixed-retina sensitivity")
    axis.plot(dense_tf, overlap_tf / overlap_tf.max(), color="#D55E00", lw=1.6, label="FEM-weighted prediction")
    axis.set_xscale("log", base=2); axis.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    axis.set(title="F  Population TF marginal", xlabel="temporal frequency (Hz)", ylabel="fraction of maximum")
    axis.grid(color="0.92")

    images = selected["image_index"].astype(int).tolist()
    best_unit = int(examples.loc[examples["selection_role"].eq("best proxy generalizer"), "rr100_index"].iloc[0])
    worst_unit = int(examples.loc[examples["selection_role"].eq("worst proxy generalizer"), "rr100_index"].iloc[0])
    for column, unit, label in ((2, best_unit, "G  A genuine success"), (3, worst_unit, "H  A systematic failure")):
        axis = fig.add_subplot(grid[1, column])
        matrix = paired_heatmap(values, unit, images)
        axis.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="viridis")
        axis.set_xticks([0, 1, 2, 3, 5, 6, 7, 8], ["0", "45", "90", "135"] * 2, fontsize=7)
        axis.set_yticks(range(len(images)), images, fontsize=7)
        axis.axvline(4, color="white", lw=2)
        rho = float(unit_summary.set_index("rr100_index").loc[unit, "median_within_image_spearman_rho"])
        axis.set_title(
            f"{label}: RR100 {unit}\npredicted (left) | observed (right)\nmedian ρ={rho:+.2f}",
            fontsize=9.0, pad=4,
        )
        axis.set(xlabel="eye direction (°); image fixed", ylabel="image index")

    axis = fig.add_subplot(grid[2, 0])
    positive = (cohort_values["predicted_response_sd_proxy_arbitrary"] > 0) & (cohort_values["fem_delta_temporal_sd_hz"] > 0)
    axis.scatter(cohort_values.loc[positive, "predicted_response_sd_proxy_arbitrary"],
                 cohort_values.loc[positive, "fem_delta_temporal_sd_hz"], s=6, alpha=0.12, color="#008837", rasterized=True)
    axis.set_xscale("log"); axis.set_yscale("log")
    axis.set(title=f"I  What works: modulation scale\nglobal ρ={global_amp_rho:+.2f}; median within-unit ρ={median_within_unit_amp:+.2f}",
             xlabel="gain × √(raw overlap), arbitrary", ylabel="measured modulation SD (Hz)")
    axis.grid(color="0.93")

    axis = fig.add_subplot(grid[2, 1])
    axis.hist(cohort["median_within_image_spearman_rho"], bins=np.linspace(-1.05, 1.05, 16), color="#4C78A8", edgecolor="white")
    axis.axvline(median_unit_direction, color="#B2182B", lw=1.5)
    axis.set(title=f"J  What fails: preferred eye direction\nmedian unit ρ={median_unit_direction:+.2f}; global ρ={global_direction_rho:+.2f}",
             xlabel="unit median within-image Spearman ρ", ylabel="RR100 units")
    axis.grid(color="0.93")

    axis = fig.add_subplot(grid[2, 2])
    fractions = [float(np.mean(pair["peak_trajectory_rotation_axial_error_deg"] == value)) for value in (0, 45, 90)]
    axis.bar(range(3), fractions, color=["#009E73", "#E69F00", "#D55E00"])
    axis.set_xticks(range(3), ["exact", "45° off", "90° off"])
    axis.set_ylim(0, 1)
    axis.set(title="K  Peak eye-direction prediction", ylabel="fraction of 396 unit×image tests")
    axis.grid(axis="y", color="0.93")

    axis = fig.add_subplot(grid[2, 3]); axis.axis("off")
    axis.text(0.02, 0.93, "L  First-pass conclusion", fontsize=11, fontweight="bold", transform=axis.transAxes)
    axis.text(
        0.02, 0.78,
        "FEM power × grating sensitivity\n"
        "captures which image/unit combinations\n"
        "produce larger temporal modulation.\n\n"
        "It does not reliably capture which\n"
        "eye-movement direction is optimal\n"
        "within a fixed image.\n\n"
        "Therefore the Kuang-style overlap is\n"
        "a useful broad drive proxy, not a\n"
        "calibrated direction-selective response model.",
        fontsize=10, va="top", linespacing=1.35, transform=axis.transAxes,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F4F4F4", "edgecolor": "0.75"},
    )
    fig.suptitle(
        "RR100 active-sensing Figure 4 first pass: retinal power predicts modulation scale, not preferred eye direction\n"
        "66 high-quality fixed-retina SF×TF fits · six preregistered natural images · frozen-model validation",
        fontsize=14,
    )
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi); fig.savefig(out_base.with_suffix(".pdf")); plt.close(fig)
    return {
        "n_quality_units": int(len(cohort)), "n_unit_image_direction_tests": int(len(pair)),
        "global_amplitude_proxy_spearman_rho": global_amp_rho,
        "median_within_unit_amplitude_proxy_spearman_rho": median_within_unit_amp,
        "global_normalized_direction_spearman_rho": global_direction_rho,
        "median_unit_within_image_direction_spearman_rho": median_unit_direction,
        "exact_peak_fraction": fractions[0], "peak_45deg_off_fraction": fractions[1], "peak_90deg_off_fraction": fractions[2],
        "best_example_rr100": best_unit, "worst_example_rr100": worst_unit,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    movies_path = args.source08_dir / "multiimage_trajectory_retinal_movies.npz"
    selection_path = args.source08_dir / "predeclared_generalization_image_selection.csv"
    values_path = args.source09_dir / "rr100_population_proxy_response_values_with_amplitude_proxy.csv"
    unit_path = args.source09_dir / "rr100_population_proxy_validation_by_unit.csv"
    examples_path = args.source09_dir / "population_example_selection.csv"
    agreement_path = args.source08_dir / "multiimage_unit_image_four_angle_agreement_all_rr100.csv"
    models_path = args.model_dir / "rr100_sf_tf_parametric_models.csv"
    movies = np.load(movies_path)
    selected = pd.read_csv(selection_path).sort_values("selection_order")
    values = pd.read_csv(values_path)
    unit_summary = pd.read_csv(unit_path)
    examples = pd.read_csv(examples_path)
    agreement = pd.read_csv(agreement_path)
    models = pd.read_csv(models_path)
    cohort_units = unit_summary.loc[unit_summary["passes_primary_fit_quality"], "rr100_index"].to_numpy(int)
    from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import _load_twin_common
    ppd = float(_load_twin_common().PPD)
    maps = population_maps(movies, selected, models, cohort_units, ppd)
    np.savez_compressed(args.out_dir / "integrated_population_maps_native_and_display.npz", **maps)
    results = plot_main(
        movies, selected, maps, unit_summary, agreement, values, examples,
        args.out_dir / "checkpoint_10_integrated_kuang_first_pass", args.dpi,
    )
    pd.DataFrame([results]).to_csv(args.out_dir / "integrated_figure_summary_values.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "integrated Kuang-style RR100 active-sensing first-pass figure",
        "status": "checkpoint_10_first_pass_complete",
        "map_contract": {
            "static_sensitivity": "mean per-unit maximum-normalized squared SFxTF sensitivity among 66 high-quality fits",
            "fem_power": "mean per-image supported-power-normalized annular SFxTF power for original eye trajectories across six preregistered images",
            "predicted_drive": "product of population static sensitivity and mean FEM power",
            "smoothing": "display only on log SF and TF axes; saved native values drive calculations",
        },
        "validation_results": results,
        "interpretation": "power overlap predicts broad modulation scale across images/units but fails to predict preferred eye direction within fixed images",
        "inputs": {
            "movies": file_identity(movies_path), "image_selection": file_identity(selection_path),
            "population_values": file_identity(values_path), "unit_summary": file_identity(unit_path),
            "examples": file_identity(examples_path), "agreement": file_identity(agreement_path),
            "models": file_identity(models_path),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Checkpoint 10: integrated Kuang-style RR100 first pass\n\n"
        "This figure combines fixed-retina SF×TF grating sensitivity, FEM-created natural-image power, their predicted "
        "overlap, auditable success/failure units, and frozen-model population validation. The broad gain-scaled overlap "
        "predicts response magnitude across images and units, while the orientation-aware extension fails to predict the "
        "preferred eye direction within fixed images. Display maps are smoothed; all reported statistics use native values.\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
