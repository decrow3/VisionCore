#!/usr/bin/env python3
"""Targeted corrected short/long-path response maps for u054 and u018."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from declan.fig4_active_sensing.run_rr100_corrected_production_cache import (
    DT,
    MAPPING,
    N_HISTORY,
    N_SCORE,
    RR100_MOVIE_MEDOID_VERSION,
    file_identity,
    render_scored_embedding,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import CanonicalTwinScorer
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1"
ASSEMBLED = CACHE / "assembled/rounds_000_011_n012_quartile_snapshot_v1"
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
ASSIGNMENTS = ROOT / (
    "outputs/fig/ssi_figure_v2/corrected_sf_quartiles_rounds000_011_v2/"
    "ssi_figure_v4_corrected_cache_sf_quartiles_no_bottom_row_rounds000_011_v2_unit_assignments.csv"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_corrected_outlier_short_long_maps_v1"
TARGETS = (54, 18)
EPS = 1e-10
CONDITION_COLORS = {"stabilized": "#777777", "short": "#0072B2", "long": "#D55E00"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=210)
    return parser.parse_args()


def within_image_slope(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, float) - np.mean(x)
    yy = np.asarray(y, float) - np.mean(y)
    return float(np.dot(xx, yy) / np.dot(xx, xx))


def choose_trace(frame: pd.DataFrame, quantile: float) -> pd.Series:
    target = float(frame.corrected_dpi_crop120_path_length_arcmin.quantile(quantile))
    return frame.iloc[int(np.argmin(np.abs(frame.corrected_dpi_crop120_path_length_arcmin.to_numpy(float) - target)))]


def select_examples(
    condition: pd.DataFrame,
    images: pd.DataFrame,
    moving_ssi: np.ndarray,
    baseline_ssi: np.ndarray,
) -> pd.DataFrame:
    strong_images = images.loc[
        images.corrected_reconstruction_orientation_coherence.ge(0.20), "image_index"
    ].to_numpy(int)
    selected: list[dict[str, object]] = []
    for unit in TARGETS:
        slopes = []
        for image in strong_images:
            frame = condition[condition.image_index.eq(image)]
            rows = frame.matrix_row_index.to_numpy(int)
            delta = np.asarray(moving_ssi[rows, unit], float) - float(baseline_ssi[image, unit])
            slopes.append({"image_index": image, "image_unit_path_slope": within_image_slope(
                frame.corrected_dpi_crop120_path_length_arcmin.to_numpy(float), delta
            )})
        slopes_frame = pd.DataFrame(slopes).sort_values("image_unit_path_slope")
        median_slope = float(slopes_frame.image_unit_path_slope.median())
        representative = slopes_frame.iloc[int(np.argmin(np.abs(slopes_frame.image_unit_path_slope - median_slope)))]
        counterexample = slopes_frame.iloc[-1]
        for role, chosen, criterion in (
            ("representative_negative_image", representative, "closest image-specific slope to strong-image median"),
            ("positive_counterexample_image", counterexample, "maximum image-specific slope among strong images"),
        ):
            image = int(chosen.image_index)
            frame = condition[condition.image_index.eq(image)].copy()
            short = choose_trace(frame, 0.10)
            long = choose_trace(frame, 0.90)
            record: dict[str, object] = {
                "rr100_index": unit,
                "selection_role": role,
                "image_selection_criterion": criterion,
                "selection_is_algorithmic": True,
                "image_index": image,
                "image_unit_path_slope": float(chosen.image_unit_path_slope),
                "strong_image_median_unit_path_slope": median_slope,
            }
            for label, trace_row in (("short", short), ("long", long)):
                matrix_row = int(trace_row.matrix_row_index)
                record[f"{label}_trace_index"] = int(trace_row.trace_index)
                record[f"{label}_matrix_row_index"] = matrix_row
                record[f"{label}_path_arcmin"] = float(trace_row.corrected_dpi_crop120_path_length_arcmin)
                record[f"{label}_cached_ssi"] = float(moving_ssi[matrix_row, unit])
                record[f"{label}_cached_delta_ssi"] = float(moving_ssi[matrix_row, unit] - baseline_ssi[image, unit])
            record["stabilized_cached_ssi"] = float(baseline_ssi[image, unit])
            selected.append(record)
    return pd.DataFrame(selected)


def score_maps(scorer, view, stim, selected_units: np.ndarray) -> np.ndarray:
    full = scorer._compute_rate_map_batched(stim)
    rr100 = apply_population_view(full, view).clamp_min(0.0)
    maps = rr100[:, selected_units].detach().cpu().numpy().astype(np.float32)
    del full, rr100
    if scorer.torch.cuda.is_available():
        scorer.torch.cuda.empty_cache()
    if maps.shape[:2] != (N_SCORE, len(selected_units)):
        raise ValueError(f"Unexpected selected response-map shape {maps.shape}")
    return maps


def map_metrics(maps: np.ndarray) -> dict[str, np.ndarray]:
    flat = np.asarray(maps, float).reshape(N_SCORE, maps.shape[1], -1)
    rate = flat.mean(axis=2)
    gain = flat / np.maximum(rate[..., None], EPS)
    ssi = np.mean(gain * np.log2(np.maximum(gain, EPS)), axis=2)
    expected_t = rate * DT
    numerator_t = ssi * expected_t
    expected = expected_t.sum(axis=0)
    return {
        "instantaneous_ssi": ssi,
        "mean_rate_t": rate,
        "information_numerator_t": numerator_t,
        "expected_spikes": expected,
        "movie_ssi": numerator_t.sum(axis=0) / np.maximum(expected, EPS),
        "mean_rate": rate.mean(axis=0),
    }


def load_patch(image_index: int) -> tuple[np.ndarray, float]:
    path = CACHE / "input_cache/images" / f"image_{image_index:03d}.npz"
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["corrected_patch"], np.float32), float(data["patch_ppd"].item())


def draw_inputs(selected: pd.DataFrame, traces: pd.DataFrame, history: np.ndarray, score: np.ndarray, out: Path, dpi: int) -> None:
    trace_ordinal = {int(v): i for i, v in enumerate(traces.trace_index.to_numpy(int))}
    fig, axes = plt.subplots(len(selected), 3, figsize=(11.8, 2.6 * len(selected)), constrained_layout=True)
    for row_index, row in enumerate(selected.itertuples(index=False)):
        patch, _ = load_patch(int(row.image_index))
        center = np.asarray(patch.shape) // 2
        half = min(95, int(min(patch.shape) // 2))
        view = patch[center[0] - half:center[0] + half, center[1] - half:center[1] + half]
        axes[row_index, 0].imshow(view, cmap="gray", vmin=np.percentile(view, 1), vmax=np.percentile(view, 99))
        axes[row_index, 0].set_title(f"image {int(row.image_index)} · strong contour\nimage slope {row.image_unit_path_slope:+.2e}")
        axes[row_index, 0].set_xticks([]); axes[row_index, 0].set_yticks([])
        for label in ("short", "long"):
            trace_index = int(getattr(row, f"{label}_trace_index"))
            ordinal = trace_ordinal[trace_index]
            xy = score[ordinal] * 60.0
            axes[row_index, 1].plot(xy[:, 0], xy[:, 1], color=CONDITION_COLORS[label], lw=1.6, label=f"{label}: {getattr(row, f'{label}_path_arcmin'):.1f}′")
            full = np.concatenate([history[ordinal], score[ordinal]], axis=0) * 60.0
            speed = np.linalg.norm(np.diff(full, axis=0), axis=1)
            axes[row_index, 2].plot(np.arange(1, len(full)), speed, color=CONDITION_COLORS[label], lw=1.4, label=label)
        axes[row_index, 1].set_aspect("equal", adjustable="datalim")
        axes[row_index, 1].set(xlabel="horizontal (arcmin)", ylabel="vertical (arcmin)", title="40-frame scored paths")
        axes[row_index, 1].legend(frameon=False, fontsize=8)
        axes[row_index, 2].axvline(N_HISTORY, color="0.5", ls="--", lw=0.9)
        axes[row_index, 2].set(xlabel="explicit-history + scored frame", ylabel="step (arcmin)", title="recorded prehistory and scored motion")
        axes[row_index, 2].legend(frameon=False, fontsize=8)
        axes[row_index, 0].set_ylabel(f"u{int(row.rr100_index):03d}\n{row.selection_role.replace('_', ' ')}")
    fig.suptitle("Corrected outlier-map inputs · response-based image roles, input-based path quantiles", fontsize=14, weight="bold")
    fig.savefig(out / "selected_short_long_inputs.png", dpi=dpi, facecolor="white")
    fig.savefig(out / "selected_short_long_inputs.pdf", facecolor="white")
    plt.close(fig)


def draw_compact_maps(selected: pd.DataFrame, maps: dict[tuple[int, int, str], np.ndarray], metrics: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(len(selected), 6, figsize=(16.2, 2.45 * len(selected)), constrained_layout=True)
    for row_index, row in enumerate(selected.itertuples(index=False)):
        key = (int(row.rr100_index), int(row.image_index))
        stabilized = maps[(*key, "stabilized")][:, 0]
        short = maps[(*key, "short")][:, 0]
        long = maps[(*key, "long")][:, 0]
        frame = int(np.argmax(np.mean(np.abs(long - short), axis=(1, 2))))
        rates = [stabilized[frame], short[frame], long[frame]]
        differences = [short[frame] - stabilized[frame], long[frame] - stabilized[frame], long[frame] - short[frame]]
        vmax = max(float(np.percentile(np.concatenate([v.ravel() for v in rates]), 99.5)), 1e-5)
        limit = max(float(np.percentile(np.abs(np.concatenate([v.ravel() for v in differences])), 99.5)), 1e-5)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        rate_image = diff_image = None
        for col, (label, value) in enumerate(zip(("stabilized", "short", "long"), rates, strict=True)):
            rate_image = axes[row_index, col].imshow(value, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            metric = metrics[(metrics.rr100_index == row.rr100_index) & (metrics.image_index == row.image_index) & (metrics.condition == label)].iloc[0]
            inst = json.loads(metric.instantaneous_ssi_json)[frame]
            rate = json.loads(metric.mean_rate_timecourse_json)[frame]
            axes[row_index, col].set_title(f"{label} · frame {frame}\nSSI {inst:.3f}; rate {rate:.3f} Hz", fontsize=8)
        for col, (label, value) in enumerate(zip(("short − stabilized", "long − stabilized", "long − short"), differences, strict=True), start=3):
            diff_image = axes[row_index, col].imshow(value, cmap="RdBu_r", norm=norm, interpolation="nearest")
            axes[row_index, col].set_title(label, fontsize=8)
        for col in range(6):
            axes[row_index, col].set_xticks([]); axes[row_index, col].set_yticks([])
        axes[row_index, 0].set_ylabel(f"u{int(row.rr100_index):03d} · image {int(row.image_index)}\n{row.selection_role.replace('_', ' ')}", fontsize=8)
        fig.colorbar(rate_image, ax=axes[row_index, :3], shrink=0.56, pad=0.004, label="rate (Hz)")
        fig.colorbar(diff_image, ax=axes[row_index, 3:], shrink=0.56, pad=0.004, label="Δ rate (Hz)")
    fig.suptitle("Targeted corrected response maps · shared condition scale per unit/image", fontsize=14, weight="bold")
    fig.savefig(out / "compact_short_long_response_maps.png", dpi=dpi, facecolor="white")
    fig.savefig(out / "compact_short_long_response_maps.pdf", facecolor="white")
    plt.close(fig)


def draw_timecourses(selected: pd.DataFrame, metrics: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(len(selected), 3, figsize=(13.5, 2.5 * len(selected)), constrained_layout=True)
    for row_index, row in enumerate(selected.itertuples(index=False)):
        frame = metrics[(metrics.rr100_index == row.rr100_index) & (metrics.image_index == row.image_index)]
        for condition in ("stabilized", "short", "long"):
            rec = frame[frame.condition.eq(condition)].iloc[0]
            ssi = np.asarray(json.loads(rec.instantaneous_ssi_json), float)
            rate = np.asarray(json.loads(rec.mean_rate_timecourse_json), float)
            cumulative = np.cumsum(ssi * rate * DT) / np.maximum(np.cumsum(rate * DT), EPS)
            axes[row_index, 0].plot(ssi, color=CONDITION_COLORS[condition], label=condition)
            axes[row_index, 1].plot(rate, color=CONDITION_COLORS[condition], label=condition)
            axes[row_index, 2].plot(cumulative, color=CONDITION_COLORS[condition], label=condition)
        axes[row_index, 0].set_ylabel(f"u{int(row.rr100_index):03d} · image {int(row.image_index)}\ninstantaneous SSI")
        axes[row_index, 1].set_ylabel("mean rate (Hz)")
        axes[row_index, 2].set_ylabel("cumulative weighted SSI")
        for col in range(3):
            axes[row_index, col].set_xlabel("scored frame")
            axes[row_index, col].spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=3)
    fig.suptitle("SSI and rate timecourses underlying the selected maps", fontsize=14, weight="bold")
    fig.savefig(out / "short_long_ssi_rate_timecourses.png", dpi=dpi, facecolor="white")
    fig.savefig(out / "short_long_ssi_rate_timecourses.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if (args.out_dir / "manifest.json").exists():
        raise FileExistsError(f"Completed targeted checkpoint exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    condition = pd.read_csv(ASSEMBLED / "condition_index.csv").merge(
        pd.read_csv(COHORT / "corrected1000_traces.csv")[["trace_index", "corrected_dpi_crop120_path_length_arcmin"]],
        on="trace_index", validate="many_to_one",
    )
    images = pd.read_csv(COHORT / "corrected100_images.csv").sort_values("image_index")
    traces = pd.read_csv(COHORT / "corrected1000_traces.csv").sort_values("trace_index").reset_index(drop=True)
    assignments = pd.read_csv(ASSIGNMENTS)
    moving_ssi = np.load(ASSEMBLED / "moving_movie_ssi_bits_per_spike.npy", mmap_mode="r")
    moving_expected = np.load(ASSEMBLED / "moving_expected_spikes.npy", mmap_mode="r")
    with np.load(ASSEMBLED / "stabilized_by_image_sufficient_statistics.npz") as archive:
        baseline_ssi = np.asarray(archive["movie_ssi_bits_per_spike"], float)
        baseline_expected = np.asarray(archive["expected_spikes"], float)
    selected = select_examples(condition, images, moving_ssi, baseline_ssi).merge(
        assignments[["rr100_index", "sf_quartile", "preferred_sf_cpd", "preferred_tf_hz"]],
        on="rr100_index", validate="many_to_one",
    )
    selected.to_csv(args.out_dir / "selected_unit_image_trace_roles.csv", index=False)
    with np.load(CACHE / "input_cache/corrected_trace_segments.npz", allow_pickle=False) as archive:
        history = np.asarray(archive["history_xy_deg"], np.float32)
        score = np.asarray(archive["score_xy_deg"], np.float32)
        frozen_trace_ids = np.asarray(archive["trace_index"], int)
    if not np.array_equal(frozen_trace_ids, traces.trace_index.to_numpy(int)):
        raise ValueError("Frozen trace order does not match cohort table")
    draw_inputs(selected, traces, history, score, args.out_dir, args.dpi)

    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    mapping = pd.read_csv(MAPPING).sort_values("rr100_index")
    if not np.array_equal(np.argmax(view.membership, axis=1), mapping.canonical_channel.to_numpy(int)):
        raise ValueError("RR100 mapping mismatch")
    scorer = CanonicalTwinScorer(device=args.device, batch_size=args.batch_size, empty_cache_every_batch=True)
    trace_ordinal = {int(v): i for i, v in enumerate(frozen_trace_ids)}
    map_cache: dict[tuple[int, int, str], np.ndarray] = {}
    metric_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        unit = int(row.rr100_index)
        image = int(row.image_index)
        patch, ppd = load_patch(image)
        unit_position = 0
        for condition_name in ("stabilized", "short", "long"):
            if condition_name == "stabilized":
                trace72 = np.zeros((N_HISTORY + N_SCORE, 2), np.float32)
                trace_index = -1
                cached_ssi = float(baseline_ssi[image, unit])
                cached_expected = float(baseline_expected[image, unit])
            else:
                trace_index = int(getattr(row, f"{condition_name}_trace_index"))
                ordinal = trace_ordinal[trace_index]
                trace72 = np.concatenate([history[ordinal], score[ordinal]], axis=0)
                matrix_row = int(getattr(row, f"{condition_name}_matrix_row_index"))
                cached_ssi = float(moving_ssi[matrix_row, unit])
                cached_expected = float(moving_expected[matrix_row, unit])
            print(f"scoring u{unit:03d} image {image:03d} {condition_name}", flush=True)
            stim = render_scored_embedding(scorer.common, scorer.torch, patch, trace72, ppd)
            maps = score_maps(scorer, view, stim, np.asarray([unit], int))
            metric = map_metrics(maps)
            map_cache[(unit, image, condition_name)] = maps
            observed_ssi = float(metric["movie_ssi"][unit_position])
            observed_expected = float(metric["expected_spikes"][unit_position])
            metric_rows.append({
                "rr100_index": unit, "image_index": image, "selection_role": row.selection_role,
                "condition": condition_name, "trace_index": trace_index,
                "movie_ssi_bits_per_spike": observed_ssi,
                "mean_rate_hz": float(metric["mean_rate"][unit_position]),
                "expected_spikes": observed_expected,
                "instantaneous_ssi_json": json.dumps(metric["instantaneous_ssi"][:, unit_position].tolist()),
                "mean_rate_timecourse_json": json.dumps(metric["mean_rate_t"][:, unit_position].tolist()),
            })
            validation_rows.append({
                "rr100_index": unit, "image_index": image, "condition": condition_name,
                "trace_index": trace_index, "cached_ssi": cached_ssi, "rerendered_ssi": observed_ssi,
                "absolute_ssi_difference": abs(cached_ssi - observed_ssi),
                "cached_expected_spikes": cached_expected, "rerendered_expected_spikes": observed_expected,
                "absolute_expected_spikes_difference": abs(cached_expected - observed_expected),
            })
    metrics = pd.DataFrame(metric_rows)
    validation = pd.DataFrame(validation_rows)
    metrics.to_csv(args.out_dir / "targeted_map_metrics.csv", index=False)
    validation.to_csv(args.out_dir / "cached_vs_rerendered_validation.csv", index=False)
    max_ssi_error = float(validation.absolute_ssi_difference.max())
    max_expected_error = float(validation.absolute_expected_spikes_difference.max())
    if max_ssi_error > 2e-6 or max_expected_error > 2e-6:
        raise ValueError(f"Targeted rerender failed cache validation: SSI={max_ssi_error}, spikes={max_expected_error}")
    draw_compact_maps(selected, map_cache, metrics, args.out_dir, args.dpi)
    draw_timecourses(selected, metrics, args.out_dir, args.dpi)
    np.savez_compressed(
        args.out_dir / "selected_response_maps.npz",
        **{f"u{unit:03d}_image{image:03d}_{condition}": value for (unit, image, condition), value in map_cache.items()},
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "targeted_corrected_multi_map_checkpoint_complete_stop_before_all_frame_sheets",
        "scope": "u054 and u018; representative-negative and positive-counterexample strong-contour images; 10th/90th path-quantile traces",
        "render_contract": "frozen corrected production patches and 32 recorded history + 40 scored frames; moving and zero-motion stabilized history scored identically",
        "selection_contract": "image roles selected from 12-round cached unit slopes; traces selected only by within-image path quantile",
        "map_scaling": "shared rate and symmetric difference scales within each unit/image row",
        "validation": {"max_absolute_ssi_error": max_ssi_error, "max_absolute_expected_spikes_error": max_expected_error},
        "sources": {
            "assembled_manifest": file_identity(ASSEMBLED / "manifest.json"),
            "trace_cache": file_identity(CACHE / "input_cache/corrected_trace_segments.npz"),
            "assignments": file_identity(ASSIGNMENTS),
        },
        "next_checkpoint": "human inspection before optional full 40-frame sheets or population-figure revision",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(selected.to_string(index=False), flush=True)
    print(validation.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
