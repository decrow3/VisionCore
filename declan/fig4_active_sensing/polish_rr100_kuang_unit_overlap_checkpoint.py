#!/usr/bin/env python3
"""Create a visually simplified Kuang-style view of checkpoint-02 maps.

Only the displayed FEM power is lightly smoothed after interpolation onto a
uniform log-SF/log-TF grid. All selection, overlap scores, and saved primary
statistics remain based on the original unsmoothed Fourier bins.
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from declan.fig4_active_sensing.make_rr100_kuang_unit_overlap_checkpoint import (
    MODEL_DIR,
    POWER_DIR,
    ROOT,
    load_power,
    surface,
)


SELECTION_DIR = ROOT / "outputs/fig4_active_sensing/rr100_kuang_unit_overlap_checkpoint_02_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_kuang_unit_overlap_checkpoint_02_visual_revision_v1"
DISPLAY_FLOOR_DB = -30.0
SMOOTH_SF_OCTAVES = 0.10
SMOOTH_TF_OCTAVES = 0.12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--power-dir", type=Path, default=POWER_DIR)
    parser.add_argument("--selection-dir", type=Path, default=SELECTION_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--smooth-sf-octaves", type=float, default=SMOOTH_SF_OCTAVES)
    parser.add_argument("--smooth-tf-octaves", type=float, default=SMOOTH_TF_OCTAVES)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def kuang_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "kuang_cool_hot",
        ["#17175f", "#174aa5", "#168fc1", "#46bd8a", "#d7db43", "#f28a2d", "#a71930"],
        N=256,
    )


def dense_display_power(sf: np.ndarray, tf: np.ndarray, power: np.ndarray,
                        smooth_sf_octaves: float, smooth_tf_octaves: float
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    dense_log_sf = np.linspace(np.log2(sf.min()), np.log2(sf.max()), 181)
    dense_log_tf = np.linspace(np.log2(tf.min()), np.log2(tf.max()), 241)
    interpolator = RegularGridInterpolator(
        (np.log2(sf), np.log2(tf)), power / power.max(), bounds_error=False, fill_value=None,
    )
    grid_sf, grid_tf = np.meshgrid(dense_log_sf, dense_log_tf, indexing="ij")
    interpolated = interpolator(np.column_stack([grid_sf.ravel(), grid_tf.ravel()])).reshape(grid_sf.shape)
    interpolated = np.maximum(interpolated, 0.0)
    sf_step = float(np.mean(np.diff(dense_log_sf)))
    tf_step = float(np.mean(np.diff(dense_log_tf)))
    sigma = (smooth_sf_octaves / sf_step, smooth_tf_octaves / tf_step)
    smoothed = gaussian_filter(interpolated, sigma=sigma, mode="nearest")
    smoothed /= max(float(smoothed.max()), 1e-15)
    return 2.0**dense_log_sf, 2.0**dense_log_tf, smoothed, {
        "sf_octaves": smooth_sf_octaves,
        "tf_octaves": smooth_tf_octaves,
        "sf_dense_pixel_sigma": float(sigma[0]),
        "tf_dense_pixel_sigma": float(sigma[1]),
    }


def edges(values: np.ndarray) -> np.ndarray:
    middle = 0.5 * (values[:-1] + values[1:])
    return np.r_[values[0] - (middle[0] - values[0]), middle, values[-1] + (values[-1] - middle[-1])]


def to_db_power(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(values, 10.0 ** (DISPLAY_FLOOR_DB / 10.0)))


def setup_axis(axis: plt.Axes, show_y: bool = True) -> None:
    sf_ticks = np.asarray([1, 2, 4, 8], dtype=float)
    # The 128-frame Fourier record resolves positive TF from 0.9375 Hz;
    # do not visually imply a measured 0.5-Hz power bin.
    tf_ticks = np.asarray([1, 2, 4, 8, 16, 32], dtype=float)
    axis.set_xticks(np.log2(sf_ticks), [f"{value:g}" for value in sf_ticks])
    axis.set_yticks(np.log2(tf_ticks), [f"{value:g}" for value in tf_ticks] if show_y else [])
    axis.set_xlabel("spatial frequency (cpd)", fontsize=8)
    if show_y:
        axis.set_ylabel("temporal frequency (Hz)", fontsize=8)
    axis.tick_params(length=2, labelsize=7)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    models = pd.read_csv(args.model_dir / "rr100_sf_tf_parametric_models.csv").set_index("rr100_index")
    selected = pd.read_csv(args.selection_dir / "selected_unit_roles.csv").sort_values("display_order")
    sf_raw, tf_raw, power_raw, image_selection = load_power(args.power_dir)
    sf, tf, power_display, smoothing = dense_display_power(
        sf_raw, tf_raw, power_raw, args.smooth_sf_octaves, args.smooth_tf_octaves
    )

    log_sf_edges = edges(np.log2(sf))
    log_tf_edges = edges(np.log2(tf))
    cmap = kuang_colormap()
    norm = Normalize(vmin=DISPLAY_FLOOR_DB, vmax=0.0)
    power_db = to_db_power(power_display)

    fig, axes = plt.subplots(2, 5, figsize=(13.2, 6.0), constrained_layout=True)
    input_axis = axes[0, 0]
    image = input_axis.pcolormesh(log_sf_edges, log_tf_edges, power_db.T, cmap=cmap, norm=norm, shading="flat")
    setup_axis(input_axis)
    input_axis.set_title("A  Exact-movie FEM power\n(display-smoothed)", fontsize=9)

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.05, 0.76,
        "In dB:\n\noverlap = FEM power\n+ 2 × sensitivity amplitude\n\n"
        "Smoothing affects display only.\nScores use unsmoothed bins.",
        transform=axes[1, 0].transAxes, va="top", fontsize=9, linespacing=1.35,
    )

    archive: dict[str, np.ndarray] = {"spatial_cpd": sf, "temporal_hz": tf,
                                     "display_smoothed_power_linear": power_display,
                                     "display_smoothed_power_db": power_db}
    for column, (_, choice) in enumerate(selected.iterrows(), start=1):
        unit = int(choice["rr100_index"])
        model = models.loc[unit]
        sensitivity = surface(model, sf, tf)
        sensitivity /= max(float(sensitivity.max()), 1e-15)
        sensitivity_power = sensitivity**2
        tuning_db = to_db_power(sensitivity_power)
        overlap_display = power_display * sensitivity_power
        overlap_db = to_db_power(overlap_display)

        top = axes[0, column]
        top.pcolormesh(log_sf_edges, log_tf_edges, tuning_db.T, cmap=cmap, norm=norm, shading="flat")
        setup_axis(top, show_y=False)
        short_role = str(choice["selection_role"]).replace(" positive overlap", "").replace("strong-unit ", "")
        top.set_title(
            f"{chr(65 + column)}  RR100 {unit}: {short_role}\n"
            f"pref. {choice['preferred_sf_cpd']:.2f} cpd, {choice['preferred_tf_hz']:.1f} Hz",
            fontsize=8.5,
        )

        bottom = axes[1, column]
        bottom.pcolormesh(log_sf_edges, log_tf_edges, overlap_db.T, cmap=cmap, norm=norm, shading="flat")
        bottom.contour(np.log2(sf), np.log2(tf), tuning_db.T, levels=[-6.0], colors="white",
                       linewidths=0.85, linestyles="--")
        setup_axis(bottom, show_y=(column == 1))
        bottom.set_title(
            f"FEM-weighted overlap = {choice['normalized_power_overlap']:.3f}\n"
            "white contour: unit −6 dB passband",
            fontsize=8.5,
        )
        archive[f"rr100_{unit:03d}_normalized_sensitivity_power"] = sensitivity_power
        archive[f"rr100_{unit:03d}_display_overlap_linear"] = overlap_display
        archive[f"rr100_{unit:03d}_display_overlap_db"] = overlap_db

    cbar = fig.colorbar(image, ax=axes, location="right", shrink=0.82, pad=0.012)
    cbar.set_label("relative power / sensitivity power (dB)", fontsize=8)
    cbar.set_ticks([-30, -20, -10, 0])
    fig.suptitle(
        "Kuang-style view: fixed-eye unit tuning reshapes FEM-induced retinal power",
        fontsize=13,
    )
    figure_path = args.out_dir / "checkpoint_02_kuang_style_smoothed_overlap"
    fig.savefig(figure_path.with_suffix(".png"), dpi=args.dpi)
    fig.savefig(figure_path.with_suffix(".pdf"))
    plt.close(fig)
    np.savez_compressed(args.out_dir / "display_smoothed_maps.npz", **archive)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "visual-only Kuang-style smoothing of checkpoint-02 power and overlap maps",
        "source_checkpoint": str(args.selection_dir.resolve()),
        "selected_rr100_indices": selected["rr100_index"].astype(int).tolist(),
        "display": {
            "frequency_coordinates": "uniform log2 SF and log2 TF",
            "interpolation": "linear interpolation of raw linear annular power",
            "gaussian_smoothing_sigma_octaves": smoothing,
            "colormap": "custom Kuang-like cool-to-hot sequential map",
            "floor_db": DISPLAY_FLOOR_DB,
            "unit_tuning_db": "10 log10(normalized sensitivity squared)",
            "overlap_db": "10 log10(display-smoothed normalized FEM power * normalized sensitivity squared)",
        },
        "quantification_policy": "all unit selection and overlap scores remain from unsmoothed checkpoint-02 bins",
        "image_selection": image_selection,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Checkpoint 02 visual revision\n\n"
        "This presentation view uses light Gaussian smoothing in log-frequency coordinates and a Kuang-like "
        "cool-to-hot colormap. Smoothing is visual only: the selected units and annotated overlap scores are "
        "copied unchanged from the unsmoothed checkpoint. The original measured-versus-fitted audit sheet remains "
        "the quantitative reference.\n"
    )
    print(json.dumps({"output": str(args.out_dir), "smoothing": smoothing,
                      "units": selected["rr100_index"].astype(int).tolist()}, indent=2))


if __name__ == "__main__":
    main()
