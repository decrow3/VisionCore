#!/usr/bin/env python3
"""Matched-pathway Stage 3 checkpoint for native tuning and large-canvas maps.

This rerenders predeclared empirical SF-by-TF-by-direction cells using the exact
33-frame native stimulus construction and learned session adapter.  The same
adapted history is sent either through the native 51-by-51 scalar pathway or
embedded in a 151-by-151 canvas and scored with the same unit readout translated
over space.  Outputs are expected firing rates in Hz (model counts/frame times
120), and every grating response is blank-subtracted within its own pathway.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataYatesV1.exp.gratings import GratingsTrial
from DataYatesV1.utils.io import get_session

from declan.run_rr100_zero_gaze_separable_sf_tf_input_checkpoint import (
    FRAME_RATE_HZ,
    derive_zero_gaze_roi,
)
from declan.run_rr100_zero_gaze_separable_sf_tf_native_smoke import (
    build_movie,
    embed_native_history,
)
from scripts.utils import get_model_and_dataset_configs


DESIGN = ROOT / "outputs/fig4_active_sensing/rr100_stage3_matched_tuning_transfer_design_v2"
NATIVE = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_native_production_v1"
DEFAULT_OUT = ROOT / "outputs/fig4_active_sensing/rr100_stage3_matched_tuning_transfer_v2"
NATIVE_SIZE = 51
LARGE_SIZE = 151
MAP_SIZE = 51
MAP_CENTER = MAP_SIZE // 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def circular_difference(left: float, right: float) -> float:
    return float(abs((float(left) - float(right) + 180.0) % 360.0 - 180.0))


def selected_readout_parameters(model, dataset_idx: int, source_unit_index: int, device: torch.device):
    readout = model.model.readouts[int(dataset_idx)]
    feature = readout.features.weight[int(source_unit_index) : int(source_unit_index) + 1]
    bias = readout.bias[int(source_unit_index) : int(source_unit_index) + 1]
    mask = readout.compute_gaussian_mask(14, 14, device)[int(source_unit_index) : int(source_unit_index) + 1]
    baseline = None
    if model.model.baseline_enabled:
        baseline = model.model.baseline_activation(model.model.baselines[int(dataset_idx)])[int(source_unit_index)]
    return feature, bias, mask[:, None], baseline


def score_histories(
    model,
    stimulus: torch.Tensor,
    dataset_idx: int,
    source_unit_index: int,
    *,
    device: torch.device,
    batch_size: int,
    verify_direct: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return native rates [T], large maps [T,51,51], and direct-path error."""
    native_chunks: list[np.ndarray] = []
    map_chunks: list[np.ndarray] = []
    direct_error = 0.0
    feature, bias, mask, baseline = selected_readout_parameters(
        model, dataset_idx, source_unit_index, device
    )
    insert = (LARGE_SIZE - NATIVE_SIZE) // 2
    dtype = next(model.model.parameters()).dtype
    with torch.inference_mode():
        for start in range(0, len(stimulus), int(batch_size)):
            raw = stimulus[start : start + int(batch_size)].to(device=device, dtype=dtype)
            adapted = model.model.adapters[int(dataset_idx)](raw)

            native_core = model.model.core_forward(adapted, None)
            # Use the complete ordinary session readout here, rather than a
            # mathematically equivalent selected-unit convolution, so the
            # native rerender follows exactly the same GPU accumulation order
            # as model.model(...).  Select the requested unit only afterward.
            native_all = model.model.activation(
                model.model.readouts[int(dataset_idx)](native_core)
            )
            if model.model.baseline_enabled:
                native_all = native_all + model.model.baseline_activation(
                    model.model.baselines[int(dataset_idx)]
                )
            native_count = native_all[:, int(source_unit_index)]

            if verify_direct and start == 0:
                direct = model.model(raw, int(dataset_idx), None)[:, int(source_unit_index)]
                direct_error = float(torch.max(torch.abs(direct - native_count)).item())
                if direct_error > 2e-6:
                    raise AssertionError(
                        f"Manual selected readout differs from ordinary native forward by {direct_error:g} counts/frame"
                    )

            large = torch.zeros(
                (len(raw), adapted.shape[1], adapted.shape[2], LARGE_SIZE, LARGE_SIZE),
                device=device,
                dtype=dtype,
            )
            large[:, :, :, insert : insert + NATIVE_SIZE, insert : insert + NATIVE_SIZE] = adapted
            large_core = model.model.core_forward(large, None)
            large_feature = F.conv2d(large_core[:, :, -1], feature)
            large_logit = F.conv2d(large_feature, mask, bias=bias)
            large_count = model.model.activation(large_logit).squeeze(1)
            if baseline is not None:
                large_count = large_count + baseline

            native_chunks.append((native_count * FRAME_RATE_HZ).cpu().numpy().astype(np.float32))
            map_chunks.append((large_count * FRAME_RATE_HZ).cpu().numpy().astype(np.float32))
            del raw, adapted, native_core, native_all, native_count
            del large, large_core, large_feature, large_logit, large_count
    native = np.concatenate(native_chunks, axis=0)
    maps = np.concatenate(map_chunks, axis=0)
    if maps.shape[1:] != (MAP_SIZE, MAP_SIZE):
        raise AssertionError(f"Expected 51-by-51 maps, received {maps.shape}")
    return native, maps, direct_error


def session_renderer(session: str):
    subject, date = session.split("_", maxsplit=1)
    sess = get_session(subject, date)
    dataset = sess.get_dataset("gratings", strict=True)
    roi, roi_audit = derive_zero_gaze_roi(dataset, session)
    trial_index = int(np.asarray(dataset["trial_inds"])[0])
    trial = GratingsTrial(sess.exp["D"][trial_index], sess.exp["S"])
    return trial, roi, roi_audit


def condition_cache_path(out_dir: Path, rr100_index: int, condition_id: int) -> Path:
    return out_dir / "phase_condition_maps" / f"unit_{rr100_index:03d}_condition_{condition_id:04d}.npz"


def evaluate(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    schedule = pd.read_csv(DESIGN / "phase_specific_render_schedule.csv")
    cached_native = pd.read_csv(NATIVE / "native_condition_unit_summary.csv")
    conditions = pd.read_csv(NATIVE / "condition_table.csv").set_index("condition_id")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "phase_condition_maps").mkdir(exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable for requested device {device}")
    load_start = time.perf_counter()
    model, _ = get_model_and_dataset_configs(mode="standard")
    model = model.to(device)
    model.model.eval()
    print(f"Loaded model on {device} in {time.perf_counter() - load_start:.1f} seconds", flush=True)

    rows: list[dict[str, object]] = []
    mean_maps: dict[str, np.ndarray] = {}
    roi_rows: list[dict[str, object]] = []
    overall_start = time.perf_counter()
    completed = 0
    total = len(schedule)
    for session, session_schedule in schedule.groupby("session", sort=False):
        trial, roi, roi_audit = session_renderer(str(session))
        roi_rows.append(dict(roi_audit))
        dataset_idx = list(model.names).index(str(session))
        session_units = session_schedule[["rr100_index", "source_unit_index"]].drop_duplicates()
        blank_condition = conditions.loc[0]
        blank_results: dict[int, tuple[float, np.ndarray, float]] = {}
        for unit_row in session_units.itertuples(index=False):
            unit = int(unit_row.rr100_index)
            source = int(unit_row.source_unit_index)
            blank_path = args.out_dir / "phase_condition_maps" / f"unit_{unit:03d}_blank.npz"
            if blank_path.exists():
                payload = np.load(blank_path)
                blank_results[unit] = (
                    float(payload["native_mean_hz"]),
                    payload["large_mean_map_hz"].astype(np.float32),
                    float(payload["direct_forward_maximum_error_counts_per_frame"]),
                )
            else:
                movie = build_movie(blank_condition, trial, roi, int(blank_condition.n_valid_response_frames))
                stimulus = embed_native_history(movie)
                native, maps, direct_error = score_histories(
                    model,
                    stimulus,
                    dataset_idx,
                    source,
                    device=device,
                    batch_size=args.batch_size,
                    verify_direct=True,
                )
                blank_results[unit] = (float(native.mean()), maps.mean(axis=0), direct_error)
                np.savez_compressed(
                    blank_path,
                    native_mean_hz=np.float64(native.mean()),
                    large_mean_map_hz=maps.mean(axis=0).astype(np.float32),
                    direct_forward_maximum_error_counts_per_frame=np.float64(direct_error),
                )

        for scheduled in session_schedule.itertuples(index=False):
            unit = int(scheduled.rr100_index)
            source = int(scheduled.source_unit_index)
            condition_id = int(scheduled.native_condition_id)
            output_path = condition_cache_path(args.out_dir, unit, condition_id)
            start = time.perf_counter()
            if output_path.exists():
                payload = np.load(output_path)
                native_mean = float(payload["native_mean_hz"])
                large_mean = payload["large_mean_map_hz"].astype(np.float32)
                direct_error = float(payload["direct_forward_maximum_error_counts_per_frame"])
                status = "resumed from saved condition"
            else:
                condition = conditions.loc[condition_id]
                movie = build_movie(condition, trial, roi, int(scheduled.n_valid_response_frames))
                stimulus = embed_native_history(movie)
                native, maps, direct_error = score_histories(
                    model,
                    stimulus,
                    dataset_idx,
                    source,
                    device=device,
                    batch_size=args.batch_size,
                    verify_direct=False,
                )
                native_mean = float(native.mean())
                large_mean = maps.mean(axis=0).astype(np.float32)
                np.savez_compressed(
                    output_path,
                    native_mean_hz=np.float64(native_mean),
                    large_mean_map_hz=large_mean,
                    direct_forward_maximum_error_counts_per_frame=np.float64(direct_error),
                )
                status = "freshly evaluated"

            blank_native, blank_map, blank_direct_error = blank_results[unit]
            cache_match = cached_native.loc[
                cached_native.rr100_index.eq(unit)
                & cached_native.condition_id.eq(condition_id)
                & cached_native.session.eq(str(session))
            ]
            if len(cache_match) != 1:
                raise AssertionError(f"Expected one cached native row for unit {unit}, condition {condition_id}")
            cache_match = cache_match.iloc[0]
            map_modulation = large_mean - blank_map
            key = f"unit_{unit:03d}_condition_{condition_id:04d}"
            mean_maps[key] = large_mean
            rows.append(
                {
                    **scheduled._asdict(),
                    "native_rerender_mean_rate_hz": native_mean,
                    "native_rerender_blank_rate_hz": blank_native,
                    "native_rerender_rate_above_blank_hz": native_mean - blank_native,
                    "cached_native_mean_rate_hz": float(cache_match.mean_rate_hz),
                    "cached_native_rate_above_blank_hz": float(cache_match.mean_rate_above_blank_hz),
                    "native_rerender_minus_cached_mean_rate_hz": native_mean - float(cache_match.mean_rate_hz),
                    "large_canvas_center_mean_rate_hz": float(large_mean[MAP_CENTER, MAP_CENTER]),
                    "large_canvas_center_blank_rate_hz": float(blank_map[MAP_CENTER, MAP_CENTER]),
                    "large_canvas_center_rate_above_blank_hz": float(map_modulation[MAP_CENTER, MAP_CENTER]),
                    "large_canvas_spatial_mean_rate_above_blank_hz": float(map_modulation.mean()),
                    "large_canvas_spatial_rms_rate_above_blank_hz": float(np.sqrt(np.mean(map_modulation**2))),
                    "ordinary_vs_manual_native_maximum_error_counts_per_frame": float(max(direct_error, blank_direct_error)),
                    "evaluation_status": status,
                    "evaluation_seconds": float(time.perf_counter() - start),
                }
            )
            completed += 1
            elapsed = time.perf_counter() - overall_start
            print(
                f"[{completed}/{total}] unit {unit}, condition {condition_id}, "
                f"SF {scheduled.spatial_frequency_cpd:g} cycles/degree, "
                f"TF {scheduled.temporal_frequency_magnitude_hz:g} Hz, "
                f"motion {scheduled.motion_direction_image_deg:g} degrees: "
                f"{status}; elapsed {elapsed / 60:.1f} minutes",
                flush=True,
            )
            atomic_csv(pd.DataFrame(rows), args.out_dir / "phase_specific_pathway_comparison.csv")

    return pd.DataFrame(rows), mean_maps, pd.DataFrame(roi_rows)


def aggregate_phases(
    phase: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray]]:
    blank_maps: dict[int, np.ndarray] = {}
    for unit in phase.rr100_index.unique():
        payload = np.load(out_dir / "phase_condition_maps" / f"unit_{int(unit):03d}_blank.npz")
        blank_maps[int(unit)] = payload["large_mean_map_hz"].astype(np.float32)
    cell_rows: list[dict[str, object]] = []
    cell_maps: dict[int, np.ndarray] = {}
    for routing_cell_id, group in phase.groupby("routing_cell_id", sort=True):
        unit = int(group.rr100_index.iloc[0])
        maps = [np.load(condition_cache_path(out_dir, unit, int(cid)))["large_mean_map_hz"] for cid in group.native_condition_id]
        mean_map = np.mean(maps, axis=0).astype(np.float32)
        modulation_map = mean_map - blank_maps[unit]
        cell_maps[int(routing_cell_id)] = modulation_map
        first = group.iloc[0]
        cell_rows.append(
            {
                **{column: first[column] for column in [
                    "routing_cell_id", "rr100_index", "selection_role", "slice_role", "sf_index", "tf_index",
                    "spatial_frequency_cpd", "temporal_frequency_magnitude_hz", "direction_index",
                    "motion_direction_image_deg", "bar_orientation_image_deg", "native_renderer_orientation_deg", "drift_sign",
                    "signed_temporal_frequency_hz", "native_tensor_signed_f0_hz", "native_tensor_positive_f0_hz",
                    "session", "source_unit_index", "canonical_channel"
                ]},
                "phase_count": int(len(group)),
                "native_rerender_phase_averaged_rate_above_blank_hz": float(group.native_rerender_rate_above_blank_hz.mean()),
                "central_map_phase_averaged_rate_above_blank_hz": float(group.large_canvas_center_rate_above_blank_hz.mean()),
                "map_spatial_mean_phase_averaged_rate_above_blank_hz": float(modulation_map.mean()),
                "map_spatial_rms_phase_averaged_rate_above_blank_hz": float(np.sqrt(np.mean(modulation_map**2))),
            }
        )
    return pd.DataFrame(cell_rows), cell_maps, blank_maps


def slice_metrics(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in cells.groupby(["rr100_index", "slice_role", "sf_index", "tf_index"], sort=False):
        group = group.sort_values("direction_index")
        native = group.native_rerender_phase_averaged_rate_above_blank_hz.to_numpy(float)
        center = group.central_map_phase_averaged_rate_above_blank_hz.to_numpy(float)
        tensor = group.native_tensor_signed_f0_hz.to_numpy(float)
        directions = group.motion_direction_image_deg.to_numpy(float)
        preferred_native = float(directions[np.argmax(native)])
        preferred_center = float(directions[np.argmax(center)])
        coefficients = np.polyfit(native, center, 1)
        fitted = np.polyval(coefficients, native)
        ss_total = float(np.sum((center - center.mean()) ** 2))
        r2 = float(1 - np.sum((center - fitted) ** 2) / ss_total) if ss_total > 1e-12 else float("nan")
        sign_mask = np.abs(native) > 1e-6
        sign_fraction = float(np.mean(np.sign(native[sign_mask]) == np.sign(center[sign_mask]))) if sign_mask.any() else float("nan")
        first = group.iloc[0]
        rows.append(
            {
                "rr100_index": int(keys[0]),
                "selection_role": first.selection_role,
                "slice_role": keys[1],
                "spatial_frequency_cpd": float(first.spatial_frequency_cpd),
                "temporal_frequency_magnitude_hz": float(first.temporal_frequency_magnitude_hz),
                "cached_tensor_vs_native_rerender_pearson_r": pearson(tensor, native),
                "native_rerender_vs_central_map_pearson_r": pearson(native, center),
                "native_rerender_vs_central_map_affine_r_squared_descriptive": r2,
                "central_map_per_native_slope": float(coefficients[0]),
                "central_map_intercept_hz": float(coefficients[1]),
                "nonzero_native_sign_preservation_fraction": sign_fraction,
                "native_rerender_preferred_motion_direction_deg": preferred_native,
                "central_map_preferred_motion_direction_deg": preferred_center,
                "preferred_direction_difference_deg": circular_difference(preferred_native, preferred_center),
            }
        )
    return pd.DataFrame(rows)


def plot_profiles(cells: pd.DataFrame, path: Path, dpi: int) -> None:
    units = cells.rr100_index.drop_duplicates().tolist()
    figure, axes = plt.subplots(len(units), 2, figsize=(17, 4.3 * len(units)), constrained_layout=True)
    for row, unit in enumerate(units):
        unit_cells = cells.loc[cells.rr100_index.eq(unit)]
        for column, (_, group) in enumerate(unit_cells.groupby("slice_role", sort=False)):
            group = group.sort_values("direction_index")
            direction = group.motion_direction_image_deg
            axes[row, column].plot(direction, group.native_tensor_signed_f0_hz, "o-", label="previous native tuning tensor")
            axes[row, column].plot(direction, group.native_rerender_phase_averaged_rate_above_blank_hz, "s--", label="exact native-path rerender")
            axes[row, column].plot(direction, group.central_map_phase_averaged_rate_above_blank_hz, "^--", label="center of large-canvas activation map")
            axes[row, column].axhline(0, color="0.35", linewidth=0.8)
            axes[row, column].set_xticks(np.arange(0, 360, 45))
            axes[row, column].set(
                xlabel="motion direction in image coordinates (degrees)",
                ylabel="phase-averaged firing-rate modulation above matched blank (Hz)",
                title=(
                    f"RR100 unit {unit}: {group.selection_role.iloc[0].replace('_', ' ')}\n"
                    f"{group.slice_role.iloc[0]}\n"
                    f"spatial frequency {group.spatial_frequency_cpd.iloc[0]:g} cycles/degree; "
                    f"temporal frequency {group.temporal_frequency_magnitude_hz.iloc[0]:g} Hz"
                ),
            )
            axes[row, column].title.set_fontsize(10)
            axes[row, column].yaxis.labelpad = 8
            axes[row, column].grid(alpha=0.2)
            if row == 0 and column == 0:
                axes[row, column].legend(fontsize=8)
    figure.suptitle(
        "Matched-pathway Stage 3: does native empirical direction tuning transfer to the center of the large-canvas activation map?",
        fontsize=14,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_maps(cells: pd.DataFrame, cell_maps: dict[int, np.ndarray], path: Path, dpi: int) -> None:
    units = cells.rr100_index.drop_duplicates().tolist()
    figure, axes = plt.subplots(len(units), 6, figsize=(20, 3.4 * len(units)), constrained_layout=True)
    for row, unit in enumerate(units):
        unit_cells = cells.loc[cells.rr100_index.eq(unit)]
        for slice_column, (_, group) in enumerate(unit_cells.groupby("slice_role", sort=False)):
            preferred = group.iloc[int(np.argmax(group.native_tensor_signed_f0_hz.to_numpy(float)))]
            opposite_direction = (float(preferred.motion_direction_image_deg) + 180.0) % 360.0
            opposite = group.iloc[int(np.argmin(np.abs(((group.motion_direction_image_deg - opposite_direction + 180) % 360) - 180)))]
            preferred_map = cell_maps[int(preferred.routing_cell_id)]
            opposite_map = cell_maps[int(opposite.routing_cell_id)]
            difference = preferred_map - opposite_map
            shared = max(float(np.max(np.abs(preferred_map))), float(np.max(np.abs(opposite_map))), 1e-6)
            diff_limit = max(float(np.max(np.abs(difference))), 1e-6)
            panels = [
                (preferred_map, shared, f"native-preferred motion {preferred.motion_direction_image_deg:g}°\nbar {preferred.bar_orientation_image_deg:g}°, drift {preferred.drift_sign}"),
                (opposite_map, shared, f"opposite motion {opposite.motion_direction_image_deg:g}°\nbar {opposite.bar_orientation_image_deg:g}°, drift {opposite.drift_sign}"),
                (difference, diff_limit, "preferred minus opposite\nactivation-map modulation"),
            ]
            for offset, (array, limit, title) in enumerate(panels):
                column = slice_column * 3 + offset
                image = axes[row, column].imshow(array, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="upper")
                axes[row, column].plot(MAP_CENTER, MAP_CENTER, marker="+", color="black", markersize=7)
                axes[row, column].set_title(
                    f"{title}\nSF {group.spatial_frequency_cpd.iloc[0]:g} cycles/degree, TF {group.temporal_frequency_magnitude_hz.iloc[0]:g} Hz",
                    fontsize=8,
                )
                axes[row, column].set_xlabel("activation-map horizontal bin")
                axes[row, column].set_ylabel("activation-map vertical bin")
                figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.02, label="firing-rate modulation above blank (Hz)")
        axes[row, 0].text(-0.34, 0.5, f"RR100 unit {unit}\n{unit_cells.selection_role.iloc[0].replace('_', ' ')}", transform=axes[row, 0].transAxes, rotation=90, va="center", ha="center", fontsize=9)
    figure.suptitle(
        "Matched-pathway Stage 3 raw activation maps: native-preferred and opposite motion directions for both predeclared frequency slices",
        fontsize=14,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    phase, mean_maps, roi_audit = evaluate(args)
    atomic_csv(phase, args.out_dir / "phase_specific_pathway_comparison.csv")
    atomic_csv(roi_audit, args.out_dir / "zero_gaze_roi_audit.csv")
    np.savez_compressed(args.out_dir / "phase_specific_mean_activation_maps_hz.npz", **mean_maps)
    cells, cell_maps, blank_maps = aggregate_phases(phase, args.out_dir)
    atomic_csv(cells, args.out_dir / "phase_averaged_directional_transfer.csv")
    metrics = slice_metrics(cells)
    atomic_csv(metrics, args.out_dir / "sf_tf_slice_transfer_metrics.csv")
    np.savez_compressed(
        args.out_dir / "phase_averaged_blank_subtracted_activation_maps_hz.npz",
        **{f"routing_cell_{key:03d}": value for key, value in cell_maps.items()},
        **{f"unit_{key:03d}_blank": value for key, value in blank_maps.items()},
    )
    plot_profiles(cells, args.out_dir / "01_native_and_large_canvas_directional_profiles.png", args.dpi)
    plot_maps(cells, cell_maps, args.out_dir / "02_raw_activation_maps_for_preferred_and_opposite_motion.png", args.dpi)

    maximum_cache_error = float(np.max(np.abs(phase.native_rerender_minus_cached_mean_rate_hz)))
    maximum_direct_error = float(np.max(phase.ordinary_vs_manual_native_maximum_error_counts_per_frame))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "targeted multi-map checkpoint complete; stop for visual inspection before aperture calibration or population propagation",
        "scientific_question": "Does empirical native SF-by-TF-by-direction tuning transfer to the center and spatial pattern of a 151-by-151 activation map when stimulus history, session adapter, readout, averaging, and rate units are matched?",
        "response_units": "expected firing rate in Hz; model expected counts per 1/120-second frame multiplied by 120",
        "history_contract": "33 frames ordered current, t-1, ..., t-32; 240 scored frames after history warmup",
        "canvas_contract": "apply the learned fixed-output 51-by-51 session adapter once, then branch the identical adapted history to the native core or embed it centrally in a zero-normalized 151-by-151 canvas",
        "readout_contract": "same exact session unit feature weight, learned 14-by-14 Gaussian mask, bias, output activation, and optional baseline; valid readout convolution yields a 51-by-51 map",
        "blank_contract": "subtract a separately scored gray blank within native and large-canvas pathways before phase averaging",
        "maximum_native_rerender_minus_cached_mean_rate_hz": maximum_cache_error,
        "maximum_ordinary_vs_manual_native_error_counts_per_frame": maximum_direct_error,
        "unit_count": int(phase.rr100_index.nunique()),
        "sf_tf_slice_count": int(metrics.shape[0]),
        "directional_routing_cell_count": int(cells.shape[0]),
        "phase_specific_condition_count": int(phase.shape[0]),
        "interpretation_limit": "This is a targeted five-unit, ten-slice map-first transfer checkpoint. Parametric fits are descriptive diagnostics, not routing weights. Do not freeze an aperture or propagate these results into later panels before inspecting the raw maps.",
        "artifacts": {
            "phase_comparison": "phase_specific_pathway_comparison.csv",
            "directional_transfer": "phase_averaged_directional_transfer.csv",
            "slice_metrics": "sf_tf_slice_transfer_metrics.csv",
            "all_phase_maps": "phase_specific_mean_activation_maps_hz.npz",
            "phase_averaged_maps": "phase_averaged_blank_subtracted_activation_maps_hz.npz",
            "directional_profiles": "01_native_and_large_canvas_directional_profiles.png",
            "raw_maps": "02_raw_activation_maps_for_preferred_and_opposite_motion.png",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        "# RR100 Stage 3 matched tuning-transfer checkpoint\n\n"
        "This directory contains the adapter-aware rerender comparing the ordinary native 51-by-51 unit response "
        "with the center and full spatial map obtained after embedding the identical adapted 33-frame history in "
        "a 151-by-151 canvas. All response values and map colorbars are expected firing rates in Hz.\n\n"
        "The checkpoint deliberately stops at five audibly selected units and ten predeclared SF-by-TF slices. "
        "Inspect the directional profiles and raw activation maps before choosing an aperture or using tuning values "
        "as local spectral routing weights.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
