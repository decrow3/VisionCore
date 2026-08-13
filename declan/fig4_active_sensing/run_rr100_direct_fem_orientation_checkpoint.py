#!/usr/bin/env python3
"""Checkpoint 04: direct frozen-RR100 responses to FEM and orientation controls.

The exact 51x51 movies from checkpoint 01 are embedded into the model's native
32-frame history. The four conditions form a matched 2x2 design:

    original content: zero gaze, real FEM
    90-degree-rotated retinal content: zero gaze, real FEM

The primary readout is the same center pixel of the post-activation RR100 rate
map used for the grating tuning. This targeted render evaluates the three
mechanistic examples plus the saved low-overlap control; it is not a population
summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import torch

from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.active_sensing_movie_information.run_backimage_rr100_frequency_tuning_probe import (
    embed_time_lags_local,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import CanonicalTwinScorer
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "outputs/fig4_active_sensing/rr100_kuang_input_power_checkpoint_01_v1"
ORIENTATION_DIR = ROOT / "outputs/fig4_active_sensing/rr100_orientation_aware_overlap_checkpoint_03_v1"
MODELS = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/rr100_sf_tf_parametric_models.csv"
MAPPING = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_recorded_twin_gratings_check_v1/rr100_unit_mapping.csv"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_direct_fem_orientation_checkpoint_04_v1"
FRAME_RATE_HZ = 120.0
UNITS = (3, 1, 19, 81)
CONDITIONS = ("original_zero_gaze", "original_real_fem", "rotated_zero_gaze", "rotated_real_fem")
COLORS = {"original": "#B2182B", "rotated": "#2166AC"}


HYPOTHESES = {
    3: {
        "case": "broad spectral match",
        "prediction": "original and rotated FEM modulation should be similar",
        "predicted_direction": "similar",
        "basis": "preferred and rotated spectral-overlap scores differ by less than 5%",
    },
    1: {
        "case": "orientation-specific match",
        "prediction": "original FEM modulation should exceed rotated FEM modulation",
        "predicted_direction": "original_gt_rotated",
        "basis": "preferred-orientation spectral overlap is 2.4x the rotated control",
    },
    19: {
        "case": "orientation mismatch",
        "prediction": "rotated FEM modulation should exceed original FEM modulation",
        "predicted_direction": "rotated_gt_original",
        "basis": "the rotated spectral-overlap control is 3.2x the preferred-orientation score",
    },
    81: {
        "case": "low-overlap control",
        "prediction": "absolute FEM modulation should be weak for both orientations",
        "predicted_direction": "weak_both",
        "basis": "both orientation-aware overlap scores are below 0.011",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--orientation-dir", type=Path, default=ORIENTATION_DIR)
    parser.add_argument("--models-csv", type=Path, default=MODELS)
    parser.add_argument("--mapping-csv", type=Path, default=MAPPING)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", type=str, default="cuda:0")
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
    return {"path": str(resolved), "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def make_conditions(archive: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    zero = archive["zero_gaze_movie"].astype(np.float32)
    fem = archive["real_fem_movie"].astype(np.float32)
    if zero.shape != fem.shape or zero.ndim != 3:
        raise ValueError(f"Unexpected retinal movie shapes: zero={zero.shape}, fem={fem.shape}")
    return {
        "original_zero_gaze": zero,
        "original_real_fem": fem,
        "rotated_zero_gaze": np.rot90(zero, k=1, axes=(1, 2)).copy(),
        "rotated_real_fem": np.rot90(fem, k=1, axes=(1, 2)).copy(),
    }


def movie_spectrum_power(movie: np.ndarray) -> np.ndarray:
    centered = np.asarray(movie, dtype=np.float64) - np.mean(movie, axis=0, keepdims=True)
    spectrum = np.fft.fftn(centered, axes=(0, 1, 2))
    return np.abs(spectrum) ** 2


def validate_saved_movie_history(input_manifest_path: Path, saved_movie: np.ndarray,
                                 n_lags: int) -> dict[str, object]:
    """Reconstruct the native helper tensor and compare the valid saved-movie history."""
    from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import (
        one_trace_from_source,
        source_row_by_id,
    )
    from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import load_source_rows
    from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
    from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import _load_twin_common
    from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
        _standardize_uint_like,
        _trace_xy_to_twin_helper_order,
    )

    manifest = json.loads(input_manifest_path.read_text())
    source_rows = load_source_rows(Path(manifest["inputs"]["source_windows"]["path"]))
    source_row_id = int(manifest["selected_image"]["source_row"])
    source_row = source_row_by_id(source_rows, source_row_id)
    patch, _ = _extract_patch(source_row, canvas_cache={}, patch_size_px=540)
    trace = one_trace_from_source(
        source_rows, source_row_id, n_timepoints=int(saved_movie.shape[0]), bin_seconds=1.0 / FRAME_RATE_HZ
    ).astype(np.float32)
    trace -= np.mean(trace, axis=0, keepdims=True)
    common = _load_twin_common()
    image = _standardize_uint_like(patch)
    full_stack = np.broadcast_to(
        image[None, :, :], (trace.shape[0] + int(n_lags) + 1, *image.shape)
    ).copy()
    eye = torch.from_numpy(_trace_xy_to_twin_helper_order(trace))
    native = common.make_counterfactual_stim(
        full_stack, eye, ppd=float(common.PPD), scale_factor=1.0,
        n_lags=int(n_lags), out_size=(int(saved_movie.shape[1]), int(saved_movie.shape[2])),
    ).detach().cpu()
    embedded = embed_time_lags_local(torch.from_numpy(np.asarray(saved_movie, dtype=np.float32)), n_lags=n_lags)
    native_start = int(n_lags)
    native_valid = native[native_start : native_start + int(embedded.shape[0])]
    history_error = torch.abs(native_valid - embedded)
    lag_zero_error = torch.abs(
        native[1 : 1 + int(saved_movie.shape[0]), 0, 0] - torch.from_numpy(np.asarray(saved_movie, dtype=np.float32))
    )
    return {
        "native_tensor_shape": list(native.shape),
        "embedded_saved_movie_shape": list(embedded.shape),
        "native_valid_start_index": native_start,
        "history_max_abs_pixel_error": float(history_error.max()),
        "history_mean_abs_pixel_error": float(history_error.mean()),
        "saved_lag_zero_max_abs_pixel_error": float(lag_zero_error.max()),
        "contract": "native[n_lags:n_lags+T_valid] equals time-lag embedding of saved lag-zero retinal movie",
    }


def run_condition(scorer: CanonicalTwinScorer, view: object, movie: np.ndarray,
                  units: np.ndarray, n_lags: int) -> np.ndarray:
    normalized = (np.asarray(movie, dtype=np.float32) - 127.0) / 255.0
    stim = embed_time_lags_local(torch.from_numpy(normalized), n_lags=n_lags)
    full = scorer._compute_rate_map_batched(stim)
    full_np = full.detach().cpu().numpy().astype(np.float32, copy=False)
    rr100 = apply_population_view(full_np, view).astype(np.float32, copy=False)
    selected = np.maximum(rr100[:, units], 0.0)
    del stim, full, full_np, rr100
    if str(scorer.device).startswith("cuda") and scorer.torch.cuda.is_available():
        scorer.torch.cuda.empty_cache()
    return selected


def make_hypothesis_table(overlap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = overlap.set_index("rr100_index")
    for order, unit in enumerate(UNITS, start=1):
        spec = HYPOTHESES[unit]
        source = indexed.loc[unit]
        rows.append({
            "display_order": order,
            "rr100_index": unit,
            **spec,
            "preferred_orientation_overlap": float(source["preferred_orientation_aware_overlap"]),
            "rotated_orientation_overlap": float(source["orthogonal_orientation_control_overlap"]),
            "proxy_preferred_minus_rotated": float(source["preferred_minus_orthogonal_overlap"]),
        })
    return pd.DataFrame(rows)


def summarize_responses(maps: dict[str, np.ndarray], hypotheses: pd.DataFrame,
                        source_frame_indices: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    time_rows = []
    center_y = int(next(iter(maps.values())).shape[-2] // 2)
    center_x = int(next(iter(maps.values())).shape[-1] // 2)
    hypothesis_index = hypotheses.set_index("rr100_index")
    for position, unit in enumerate(UNITS):
        original_zero = maps["original_zero_gaze"][:, position, center_y, center_x].astype(float)
        original_fem = maps["original_real_fem"][:, position, center_y, center_x].astype(float)
        rotated_zero = maps["rotated_zero_gaze"][:, position, center_y, center_x].astype(float)
        rotated_fem = maps["rotated_real_fem"][:, position, center_y, center_x].astype(float)
        original_delta = original_fem - original_zero
        rotated_delta = rotated_fem - rotated_zero
        original_rms = float(np.sqrt(np.mean(original_delta**2)))
        rotated_rms = float(np.sqrt(np.mean(rotated_delta**2)))
        original_modulation_sd = float(np.std(original_delta))
        rotated_modulation_sd = float(np.std(rotated_delta))
        original_mean_delta = float(np.mean(original_delta))
        rotated_mean_delta = float(np.mean(rotated_delta))
        predicted = str(hypothesis_index.loc[unit, "predicted_direction"])
        tolerance = 0.10 * max(original_modulation_sd, rotated_modulation_sd, 1e-12)
        if abs(original_modulation_sd - rotated_modulation_sd) <= tolerance:
            observed_direction = "similar"
        elif original_modulation_sd > rotated_modulation_sd:
            observed_direction = "original_gt_rotated"
        else:
            observed_direction = "rotated_gt_original"
        if predicted == "weak_both":
            qualitative_match = "control_requires_cross_unit_scale"
        else:
            qualitative_match = str(observed_direction == predicted)
        rows.append({
            "rr100_index": unit,
            "case": hypothesis_index.loc[unit, "case"],
            "center_y": center_y,
            "center_x": center_x,
            "n_response_frames": len(original_fem),
            "original_zero_mean_rate_hz": float(np.mean(original_zero)),
            "original_fem_mean_rate_hz": float(np.mean(original_fem)),
            "rotated_zero_mean_rate_hz": float(np.mean(rotated_zero)),
            "rotated_fem_mean_rate_hz": float(np.mean(rotated_fem)),
            "original_fem_minus_zero_mean_hz": original_mean_delta,
            "rotated_fem_minus_zero_mean_hz": rotated_mean_delta,
            "original_fem_delta_rms_hz": original_rms,
            "rotated_fem_delta_rms_hz": rotated_rms,
            "original_minus_rotated_delta_rms_hz": original_rms - rotated_rms,
            "original_to_rotated_delta_rms_ratio": original_rms / max(rotated_rms, 1e-12),
            "original_fem_delta_temporal_sd_hz": original_modulation_sd,
            "rotated_fem_delta_temporal_sd_hz": rotated_modulation_sd,
            "original_minus_rotated_delta_temporal_sd_hz": original_modulation_sd - rotated_modulation_sd,
            "original_to_rotated_delta_temporal_sd_ratio": original_modulation_sd / max(rotated_modulation_sd, 1e-12),
            "original_fem_rate_std_hz": float(np.std(original_fem)),
            "rotated_fem_rate_std_hz": float(np.std(rotated_fem)),
            "original_fraction_delta_positive": float(np.mean(original_delta > 0)),
            "rotated_fraction_delta_positive": float(np.mean(rotated_delta > 0)),
            "predicted_direction": predicted,
            "observed_direction_10pct_similarity_tolerance": observed_direction,
            "qualitative_prediction_match": qualitative_match,
        })
        for time_index, source_frame in enumerate(source_frame_indices):
            time_rows.append({
                "rr100_index": unit,
                "response_frame_index": time_index,
                "source_movie_frame_index": int(source_frame),
                "time_from_movie_start_ms": float(source_frame * 1000.0 / FRAME_RATE_HZ),
                "original_zero_rate_hz": float(original_zero[time_index]),
                "original_fem_rate_hz": float(original_fem[time_index]),
                "original_fem_minus_zero_hz": float(original_delta[time_index]),
                "rotated_zero_rate_hz": float(rotated_zero[time_index]),
                "rotated_fem_rate_hz": float(rotated_fem[time_index]),
                "rotated_fem_minus_zero_hz": float(rotated_delta[time_index]),
            })
    summary = pd.DataFrame(rows)
    # The low-overlap control is supported only on an absolute selected-unit
    # scale, not by its original-versus-rotated direction.
    control = summary["rr100_index"].eq(81)
    examples = ~control
    control_peak = float(summary.loc[control, ["original_fem_delta_temporal_sd_hz", "rotated_fem_delta_temporal_sd_hz"]].max(axis=1).iloc[0])
    example_floor = float(summary.loc[examples, ["original_fem_delta_temporal_sd_hz", "rotated_fem_delta_temporal_sd_hz"]].min(axis=1).min())
    control_supported = bool(control_peak < example_floor)
    summary.loc[control, "observed_direction_10pct_similarity_tolerance"] = "weak_both" if control_supported else "not_weak"
    summary.loc[control, "qualitative_prediction_match"] = str(control_supported)
    summary.loc[control, "control_peak_modulation_sd_hz"] = control_peak
    summary.loc[control, "noncontrol_minimum_modulation_sd_hz"] = example_floor
    return summary, pd.DataFrame(time_rows)


def montage(movie: np.ndarray, indices: np.ndarray) -> np.ndarray:
    separator = np.full((movie.shape[1], 2), np.nan)
    pieces = []
    for number, index in enumerate(indices):
        if number:
            pieces.append(separator)
        pieces.append(movie[int(index)])
    return np.concatenate(pieces, axis=1)


def plot_response_summary(conditions: dict[str, np.ndarray], summaries: pd.DataFrame,
                          timecourses: pd.DataFrame, hypotheses: pd.DataFrame,
                          out_base: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(14.5, 12.0), constrained_layout=False)
    grid = fig.add_gridspec(5, 3, height_ratios=[0.72, 1, 1, 1, 1],
                           left=0.07, right=0.97, top=0.92, bottom=0.065, hspace=0.58, wspace=0.34)
    input_frames = np.asarray([31, 63, 95, 127], dtype=int)
    input_axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    all_images = np.concatenate([conditions[name][input_frames] for name in CONDITIONS], axis=0)
    vmin, vmax = np.percentile(all_images, [1, 99])
    input_axes[0].imshow(montage(conditions["original_real_fem"], input_frames), cmap="gray", vmin=vmin, vmax=vmax)
    input_axes[0].set_title("A  Original FEM retinal movie", loc="left", fontweight="bold")
    input_axes[1].imshow(montage(conditions["rotated_real_fem"], input_frames), cmap="gray", vmin=vmin, vmax=vmax)
    input_axes[1].set_title("B  Same retinal movie rotated 90°", loc="left", fontweight="bold")
    input_axes[2].axis("off")
    input_axes[2].text(0.0, 0.88, "Matched 2×2 control", fontweight="bold", fontsize=11,
                       transform=input_axes[2].transAxes)
    input_axes[2].text(0.0, 0.60, "original:  zero gaze  →  real FEM\nrotated:   zero gaze  →  real FEM",
                       fontsize=10, transform=input_axes[2].transAxes, va="top")
    for axis in input_axes[:2]:
        axis.set_xticks([]); axis.set_yticks([])

    summary_index = summaries.set_index("rr100_index")
    hypothesis_index = hypotheses.set_index("rr100_index")
    for row_index, unit in enumerate(UNITS, start=1):
        data = timecourses.loc[timecourses["rr100_index"].eq(unit)]
        summary = summary_index.loc[unit]
        hypothesis = hypothesis_index.loc[unit]
        time_ms = data["time_from_movie_start_ms"].to_numpy(float)
        raw_values = np.concatenate([
            data["original_zero_rate_hz"], data["original_fem_rate_hz"],
            data["rotated_zero_rate_hz"], data["rotated_fem_rate_hz"],
        ])
        low = max(0.0, float(np.min(raw_values)) - 0.06 * float(np.ptp(raw_values)))
        high = float(np.max(raw_values)) + max(0.06 * float(np.ptp(raw_values)), 0.1)

        ax = fig.add_subplot(grid[row_index, 0])
        ax.plot(time_ms, data["original_fem_rate_hz"], color=COLORS["original"], lw=1.35, label="real FEM")
        ax.plot(time_ms, data["original_zero_rate_hz"], color=COLORS["original"], lw=1.0, ls="--", label="zero gaze")
        ax.set_ylim(low, high)
        ax.set(title=f"RR100 {unit}: {hypothesis['case']}\noriginal content",
               xlabel="time from movie start (ms)", ylabel="center response (Hz)")
        ax.grid(color="0.91", lw=0.6); ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

        ax = fig.add_subplot(grid[row_index, 1])
        ax.plot(time_ms, data["rotated_fem_rate_hz"], color=COLORS["rotated"], lw=1.35, label="real FEM")
        ax.plot(time_ms, data["rotated_zero_rate_hz"], color=COLORS["rotated"], lw=1.0, ls="--", label="zero gaze")
        ax.set_ylim(low, high)
        ax.set(title="90°-rotated content", xlabel="time from movie start (ms)", ylabel="center response (Hz)")
        ax.grid(color="0.91", lw=0.6); ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

        ax = fig.add_subplot(grid[row_index, 2])
        modulation = [summary["original_fem_delta_temporal_sd_hz"], summary["rotated_fem_delta_temporal_sd_hz"]]
        bars = ax.bar([0, 1], modulation, color=[COLORS["original"], COLORS["rotated"]], width=0.62)
        ax.set_xticks([0, 1], ["original", "rotated 90°"])
        ax.set_ylabel("FEM modulation SD (Hz)")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, modulation):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035 * max(modulation), f"{value:.3f}", ha="center", fontsize=8)
        ax.set_ylim(0, max(modulation) * 1.25 if max(modulation) > 0 else 1)
        predicted_labels = {
            "similar": "similar for original and rotated",
            "original_gt_rotated": "original > rotated",
            "rotated_gt_original": "rotated > original",
            "weak_both": "weak in both",
        }
        observed_direction = str(summary["observed_direction_10pct_similarity_tolerance"])
        rotated_over_original = float(modulation[1] / max(modulation[0], 1e-12))
        if observed_direction == "weak_both":
            observed_label = "weak in both"
        elif observed_direction == "similar":
            observed_label = f"similar (rotated = {rotated_over_original:.2f}× original)"
        elif observed_direction == "rotated_gt_original":
            observed_label = f"rotated = {rotated_over_original:.2f}× original"
        else:
            observed_label = f"original = {1.0 / max(rotated_over_original, 1e-12):.2f}× rotated"
        verdict = "supported" if str(summary["qualitative_prediction_match"]) == "True" else "not supported"
        ax.set_title(
            f"Predicted: {predicted_labels[str(hypothesis['predicted_direction'])]}\n"
            f"Observed: {observed_label} — {verdict}",
            fontsize=9, loc="left",
        )
        ax.text(0.0, -0.26,
                f"mean FEM effect: {summary['original_fem_minus_zero_mean_hz']:+.2f} / "
                f"{summary['rotated_fem_minus_zero_mean_hz']:+.2f} Hz",
                transform=ax.transAxes, fontsize=8, color="0.30")

    fig.suptitle("Checkpoint 04: direct frozen-RR100 responses test the spectral-overlap predictions", fontsize=14)
    fig.text(0.07, 0.018,
             "Solid: real FEM. Dashed: matched true-zero-gaze movie. Temporal SD is computed after removing the mean from the paired "
             "FEM-minus-zero trace; raw rates and signed mean effects are retained beside it.", fontsize=8, color="0.30")
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_delta_map_sheet(maps: dict[str, np.ndarray], summaries: pd.DataFrame,
                         out_base: Path, dpi: int) -> None:
    response_indices = np.asarray([0, 48, 96], dtype=int)
    source_indices = response_indices + 31
    fig, axes = plt.subplots(len(UNITS), 6, figsize=(13.8, 8.8), constrained_layout=True)
    summary_index = summaries.set_index("rr100_index")
    for row, unit in enumerate(UNITS):
        position = list(UNITS).index(unit)
        original_delta = maps["original_real_fem"][:, position] - maps["original_zero_gaze"][:, position]
        rotated_delta = maps["rotated_real_fem"][:, position] - maps["rotated_zero_gaze"][:, position]
        selected_values = np.concatenate([original_delta[response_indices].ravel(), rotated_delta[response_indices].ravel()])
        limit = max(float(np.percentile(np.abs(selected_values), 99)), 1e-6)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = None
        for condition_index, (label, delta) in enumerate((("original", original_delta), ("rotated 90°", rotated_delta))):
            for time_position, (response_index, source_index) in enumerate(zip(response_indices, source_indices, strict=True)):
                column = condition_index * 3 + time_position
                axis = axes[row, column]
                image = axis.imshow(delta[response_index], cmap="RdBu_r", norm=norm, interpolation="nearest")
                center_y = int(delta.shape[-2] // 2); center_x = int(delta.shape[-1] // 2)
                axis.scatter([center_x], [center_y], s=14, facecolors="none", edgecolors="black", lw=0.6)
                center_delta = float(delta[response_index, center_y, center_x])
                axis.set_title(f"{label}; {source_index / FRAME_RATE_HZ * 1000:.0f} ms\ncenter Δ={center_delta:+.2f} Hz", fontsize=7.5)
                axis.set_xticks([]); axis.set_yticks([])
        axes[row, 0].set_ylabel(f"RR100 {unit}\n{summary_index.loc[unit, 'case']}", fontsize=8.5)
        cbar = fig.colorbar(image, ax=axes[row], shrink=0.78, pad=0.008)
        cbar.set_label("FEM − matched zero gaze (Hz)", fontsize=7)
        cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Direct response-map differences at three fixed movie times\n(per-unit symmetric color scale; circle marks center readout)", fontsize=13)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    input_npz_path = args.input_dir / "checkpoint_01_retinal_movies_and_power.npz"
    input_manifest_path = args.input_dir / "manifest.json"
    overlap_path = args.orientation_dir / "selected_unit_orientation_overlap_metrics.csv"
    archive = np.load(input_npz_path)
    conditions = make_conditions(archive)
    overlap = pd.read_csv(overlap_path)
    hypotheses = make_hypothesis_table(overlap)
    hypotheses.to_csv(args.out_dir / "predeclared_example_hypotheses.csv", index=False)

    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    mapping = pd.read_csv(args.mapping_csv).sort_values("rr100_index")
    selected_channels = np.argmax(view.membership, axis=1)
    if not np.array_equal(selected_channels, mapping["canonical_channel"].to_numpy(int)):
        raise ValueError("RR100 movie-medoid view does not match the grating-fit unit mapping")
    if int(view.n_units) != 100:
        raise ValueError(f"Expected RR100 view, got {view.n_units} units")

    scorer = CanonicalTwinScorer(device=args.device, batch_size=args.batch_size, empty_cache_every_batch=True)
    n_lags = int(scorer.common.N_LAGS)
    history_check = validate_saved_movie_history(
        input_manifest_path, conditions["original_real_fem"], n_lags=n_lags
    )
    if float(history_check["history_max_abs_pixel_error"]) != 0.0:
        raise ValueError(f"Saved-movie history does not reproduce native helper tensor: {history_check}")
    (args.out_dir / "native_history_equivalence_check.json").write_text(
        json.dumps(history_check, indent=2) + "\n"
    )
    units = np.asarray(UNITS, dtype=np.int64)
    maps = {}
    for condition in CONDITIONS:
        print(f"running {condition}", flush=True)
        maps[condition] = run_condition(scorer, view, conditions[condition], units, n_lags)
    shapes = {name: value.shape for name, value in maps.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Response map shapes differ: {shapes}")
    n_response_frames = next(iter(maps.values())).shape[0]
    source_frame_indices = np.arange(n_lags - 1, n_lags - 1 + n_response_frames, dtype=int)
    if source_frame_indices[-1] != conditions["original_real_fem"].shape[0] - 1:
        raise ValueError("Lag embedding did not align to the final retinal movie frame")

    summaries, timecourses = summarize_responses(maps, hypotheses, source_frame_indices)
    summaries.to_csv(args.out_dir / "selected_unit_direct_response_summary.csv", index=False)
    timecourses.to_csv(args.out_dir / "selected_unit_direct_response_timecourses.csv", index=False)
    np.savez_compressed(
        args.out_dir / "selected_unit_direct_response_maps.npz",
        rr100_indices=units,
        source_movie_frame_indices=source_frame_indices,
        **{name: value for name, value in maps.items()},
    )

    plot_response_summary(conditions, summaries, timecourses, hypotheses,
                          args.out_dir / "checkpoint_04_direct_response_timecourses", args.dpi)
    # The exact 51x51 aperture produces a 1x1 rate-map readout. Spatial delta
    # maps would therefore be single pixels and are intentionally not plotted.

    zero_original_dynamic = float(np.max(np.abs(conditions["original_zero_gaze"] - conditions["original_zero_gaze"][0])))
    zero_rotated_dynamic = float(np.max(np.abs(conditions["rotated_zero_gaze"] - conditions["rotated_zero_gaze"][0])))
    rotation_exact = float(np.max(np.abs(conditions["rotated_real_fem"] - np.rot90(conditions["original_real_fem"], k=1, axes=(1, 2)))))
    original_spectrum = movie_spectrum_power(conditions["original_real_fem"])
    rotated_spectrum = movie_spectrum_power(conditions["rotated_real_fem"])
    spectrum_total_relative_error = float(abs(original_spectrum.sum() - rotated_spectrum.sum()) / max(original_spectrum.sum(), 1e-30))
    maximum_static_response_time_range = max(
        float(np.max(np.ptp(maps["original_zero_gaze"], axis=0))),
        float(np.max(np.ptp(maps["rotated_zero_gaze"], axis=0))),
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "targeted direct frozen-RR100 FEM orientation checkpoint",
        "status": "direct_selected_unit_checkpoint_complete_stop_before_population_summary",
        "design": "matched 2x2: original/rotated retinal content x true-zero-gaze/real-FEM",
        "rotation_control": "np.rot90 applied identically to every already-rendered 51x51 retinal frame; this preserves time ordering and total 3D Fourier power while rotating spatial orientation",
        "response_contract": {
            "model": scorer.model_family,
            "rr100_version": RR100_MOVIE_MEDOID_VERSION,
            "mapping_exact_to_grating_fit": True,
            "history_frames": n_lags,
            "history_embedding": "frame t at lag0, prior retinal frames at increasing lag; first 31 movie frames supply history",
            "stimulus_normalization": "(retinal_movie_float_0_to_255 - 127) / 255",
            "scalar_readout": "center pixel of post-activation RR100 rate map",
            "primary_dynamic_metric": "temporal standard deviation of framewise (real_FEM_rate - matched_zero_gaze_rate), equivalent to demeaned modulation RMS",
            "secondary_metrics": "raw mean rates, signed mean FEM-minus-zero effect, and non-demeaned RMS relative to zero gaze",
            "spatial_map_policy": "exact 51x51 aperture yields 1x1 post-activation readout; no meaningless single-pixel map sheet is plotted",
        },
        "checks": {
            "response_shapes": {name: list(shape) for name, shape in shapes.items()},
            "zero_gaze_original_max_frame_difference": zero_original_dynamic,
            "zero_gaze_rotated_max_frame_difference": zero_rotated_dynamic,
            "rotation_exact_max_abs_pixel_error": rotation_exact,
            "rotation_total_3d_fourier_power_relative_error": spectrum_total_relative_error,
            "maximum_static_response_time_range_hz": maximum_static_response_time_range,
            "native_history_equivalence": history_check,
        },
        "inputs": {
            "retinal_movies": file_identity(input_npz_path),
            "input_manifest": file_identity(input_manifest_path),
            "orientation_overlap_metrics": file_identity(overlap_path),
            "models": file_identity(args.models_csv),
            "mapping": file_identity(args.mapping_csv),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Checkpoint 04: direct FEM orientation response test\n\n"
        "This targeted frozen-model run tests the three checkpoint-03 mechanisms plus RR100 81 as the low-overlap "
        "control. Original and 90-degree-rotated retinal content each receive their own true-zero-gaze baseline, "
        "preventing static orientation preference from being mistaken for a FEM effect. The primary metric is the "
        "temporal standard deviation of the paired framewise response difference; the signed mean effect is retained "
        "separately. The exact 51x51 aperture produces a 1x1 cell-like readout, so no spatial map sheet is shown. "
        "No population conclusion is made at this checkpoint.\n"
    )
    print(summaries.to_string(index=False))
    print(json.dumps(manifest["checks"], indent=2))


if __name__ == "__main__":
    main()
