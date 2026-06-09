#!/usr/bin/env python3
"""Pre-model retinal movie transform QC for the active-sensing figure.

This companion audit reconstructs the exact 151x151 retinal movies for the
selected image/crop/trace pairs and computes stimulus-side diagnostics before
the V1 model is involved.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jake.twininfo.common import DT, extract_fixrsvp_eye_traces, load_digital_twin
from jake.twininfo.image_selection import PYRAMID_HEIGHT, PYRAMID_ORDER, SF_BANDS_4
from jake.twininfo.pipeline import (
    _canonical_condition_name,
    _example_seed,
    _trajectory_for_condition,
)
from jake.twininfo.retinal_examples import (
    TraceExample,
    pyramid_local_image_controls,
    retinal_movie_from_image_trace,
    select_trace_examples,
)
from jake.twininfo.stimuli import load_natural_images


DEFAULT_RUN = ROOT / "outputs" / "twininfo" / "active-sensing-all-images-1crop-2fix2ms-16units-gpu"
DEFAULT_OUT = ROOT / "outputs" / "active_sensing_movie_information" / "active_sensing_movie_information_figure"

CONDITION_ORDER = (
    "real",
    "stabilized",
    "random_amp",
    "random_amp_cloud_matched",
    "sf_low",
    "stabilized_sf_low",
    "sf_mid_low",
    "stabilized_sf_mid_low",
    "sf_mid_high",
    "stabilized_sf_mid_high",
    "sf_high",
    "stabilized_sf_high",
    "pyramid_phase_scrambled",
    "stabilized_pyramid_phase_scrambled",
)

COND_LABELS = {
    "real": "real FEM",
    "stabilized": "stabilized",
    "random_amp": "amp-matched random dirs",
    "random_amp_cloud_matched": "amp+cloud matched dirs",
    "sf_low": "lowpass",
    "stabilized_sf_low": "lowpass stabilized",
    "sf_mid_low": "mid-low SF",
    "stabilized_sf_mid_low": "mid-low SF stabilized",
    "sf_mid_high": "mid-high SF",
    "stabilized_sf_mid_high": "mid-high SF stabilized",
    "sf_high": "highpass",
    "stabilized_sf_high": "highpass stabilized",
    "pyramid_phase_scrambled": "phase scramble",
    "stabilized_pyramid_phase_scrambled": "phase scramble stabilized",
}

COND_COLORS = {
    "real": "#1f77b4",
    "stabilized": "#2ca02c",
    "random_amp": "#7f7f7f",
    "random_amp_cloud_matched": "#525252",
    "sf_low": "#9467bd",
    "stabilized_sf_low": "#c7a9dd",
    "sf_mid_low": "#8c564b",
    "stabilized_sf_mid_low": "#d7b5d8",
    "sf_mid_high": "#ff7f0e",
    "stabilized_sf_mid_high": "#ffbb78",
    "sf_high": "#4c78a8",
    "stabilized_sf_high": "#9ecae1",
    "pyramid_phase_scrambled": "#d62728",
    "stabilized_pyramid_phase_scrambled": "#f2a4a4",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "metadata" / "run_config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _selected_examples(run_dir: Path, config: dict[str, Any]) -> list[TraceExample]:
    model, _info, _device = load_digital_twin()
    eye_traces, durations = extract_fixrsvp_eye_traces(model, min_fix_dur=int(config["t_max"]))
    examples = select_trace_examples(
        eye_traces,
        durations,
        t_max=int(config["t_max"]),
        n_each=int(config["n_examples_per_kind"]),
        seed=int(config["seed"]),
        stride=int(config["stride"]),
    )
    used_rows = read_csv_rows(run_dir / "metadata" / "01_trace_examples_used.csv")
    used_ids = [str(row["example_id"]) for row in used_rows]
    by_id = {example.example_id: example for example in examples}
    return [by_id[example_id] for example_id in used_ids]


def _image_by_index(crop_rows: list[dict[str, str]]) -> dict[int, np.ndarray]:
    indices = tuple(sorted({int(row["image_index"]) for row in crop_rows}))
    loaded = load_natural_images(max(indices) + 1, indices=indices)
    return {int(spec.image_index): image for spec, image in loaded}


def _stable_trace(trace: np.ndarray, t_max: int) -> np.ndarray:
    return np.repeat(np.mean(trace[:t_max], axis=0, keepdims=True), t_max, axis=0).astype(np.float32)


def _movie_for_condition(
    *,
    image: np.ndarray,
    trace: np.ndarray,
    condition: str,
    t_max: int,
    seed: int,
    crop_offset: tuple[float, float],
    control_images: dict[str, np.ndarray],
) -> np.ndarray:
    condition = _canonical_condition_name(condition)
    if condition in {"real", "stabilized", "random_amp", "random_amp_cloud_matched"}:
        tr, _desc = _trajectory_for_condition(trace, condition, t_max=t_max, seed=seed)
        return retinal_movie_from_image_trace(image, tr, t_max=t_max, crop_center_offset_px=crop_offset)
    if condition in control_images:
        return retinal_movie_from_image_trace(
            control_images[condition],
            trace,
            t_max=t_max,
            crop_center_offset_px=crop_offset,
        )
    if condition.startswith("stabilized_"):
        visual_condition = condition.removeprefix("stabilized_")
        if visual_condition not in control_images:
            raise ValueError(f"Missing control image for {condition}")
        return retinal_movie_from_image_trace(
            control_images[visual_condition],
            _stable_trace(trace, t_max),
            t_max=t_max,
            crop_center_offset_px=crop_offset,
        )
    raise ValueError(f"Unsupported retinal QC condition: {condition}")


def _temporal_power_bands(movie: np.ndarray) -> dict[str, float]:
    arr = np.asarray(movie, dtype=np.float32)
    pixels = arr.reshape(arr.shape[0], -1)
    stride = max(1, pixels.shape[1] // 2048)
    pixels = pixels[:, ::stride]
    pixels = pixels - np.mean(pixels, axis=0, keepdims=True)
    spec = np.fft.rfft(pixels, axis=0)
    freq = np.fft.rfftfreq(arr.shape[0], d=DT)
    power = np.mean(np.abs(spec) ** 2, axis=1)

    def band(lo: float, hi: float) -> float:
        mask = (freq >= lo) & (freq < hi)
        return float(np.sum(power[mask]))

    return {
        "temporal_power_0p5_4hz": band(0.5, 4.0),
        "temporal_power_4_15hz": band(4.0, 15.0),
        "temporal_power_15_60hz": band(15.0, 60.0),
    }


def movie_metrics(movie: np.ndarray, stable_reference: np.ndarray) -> dict[str, float]:
    arr = np.asarray(movie, dtype=np.float32)
    diff = np.diff(arr, axis=0)
    temporal_rms = np.sqrt(np.mean(diff * diff, axis=(1, 2))) if diff.size else np.zeros((0,), dtype=np.float32)
    gy, gx = np.gradient(arr, axis=(1, 2))
    grad = np.sqrt(gx * gx + gy * gy)
    metrics = {
        "temporal_contrast_rms_mean": float(np.mean(temporal_rms)) if temporal_rms.size else 0.0,
        "temporal_contrast_rms_p95": float(np.percentile(temporal_rms, 95.0)) if temporal_rms.size else 0.0,
        "movie_power_mean": float(np.mean(arr * arr)),
        "motion_power_vs_matched_stabilized_mean": float(np.mean((arr - stable_reference) ** 2)),
        "gradient_magnitude_mean": float(np.mean(grad)),
        "gradient_magnitude_p95": float(np.percentile(grad, 95.0)),
    }
    metrics.update(_temporal_power_bands(arr))
    return metrics


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "temporal_contrast_rms_mean",
        "temporal_contrast_rms_p95",
        "movie_power_mean",
        "motion_power_vs_matched_stabilized_mean",
        "gradient_magnitude_mean",
        "temporal_power_0p5_4hz",
        "temporal_power_4_15hz",
        "temporal_power_15_60hz",
    ]
    out = []
    for condition in CONDITION_ORDER:
        cr = [row for row in rows if row["condition"] == condition]
        if not cr:
            continue
        for metric in metrics:
            vals = np.asarray([float(row[metric]) for row in cr], dtype=np.float64)
            out.append({
                "condition": condition,
                "metric": metric,
                "mean": float(np.mean(vals)),
                "ci95_low": float(np.quantile(vals, 0.025)),
                "ci95_high": float(np.quantile(vals, 0.975)),
                "n": int(vals.size),
            })
    return out


def plot_qc(summary_rows: list[dict[str, Any]], path: Path) -> None:
    panels = [
        ("temporal_contrast_rms_mean", "temporal contrast RMS"),
        ("motion_power_vs_matched_stabilized_mean", "motion power vs stabilized"),
        ("gradient_magnitude_mean", "gradient magnitude"),
        ("temporal_power_15_60hz", "15-60 Hz temporal power"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(13.0, 8.2))
    axs = axs.ravel()
    for ax, (metric, ylabel) in zip(axs, panels, strict=True):
        rows = [row for row in summary_rows if row["metric"] == metric]
        labels = [COND_LABELS[str(row["condition"])] for row in rows]
        means = np.asarray([float(row["mean"]) for row in rows], dtype=np.float64)
        lows = np.asarray([float(row["ci95_low"]) for row in rows], dtype=np.float64)
        highs = np.asarray([float(row["ci95_high"]) for row in rows], dtype=np.float64)
        x = np.arange(len(rows), dtype=np.float64)
        ax.bar(
            x,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            capsize=2,
            color=[COND_COLORS[str(row["condition"])] for row in rows],
            alpha=0.88,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(metric.replace("_", " "), loc="left", fontweight="bold")
        ax.grid(axis="y", color="0.9", lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Retinal movie transform QC: stimulus-side effects before the model", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_qc(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    config = _load_run_config(run_dir)
    t_max = int(config["t_max"])
    examples = _selected_examples(run_dir, config)
    crop_rows = read_csv_rows(run_dir / "metadata" / "02_image_crop_hotspots.csv")
    images = _image_by_index(crop_rows)
    by_example = {example.example_id: example for example in examples}

    rows: list[dict[str, Any]] = []
    for crop in crop_rows:
        image_index = int(crop["image_index"])
        crop_rank = int(crop["crop_rank"])
        crop_offset = (float(crop["offset_x_px"]), float(crop["offset_y_px"]))
        image = images[image_index]
        for example_id, example in by_example.items():
            seed = _example_seed(int(config["seed"]), example_id, image_index, crop_rank)
            control_images, _audits = pyramid_local_image_controls(
                image,
                example.trace,
                np.random.default_rng(seed),
                crop_center_offset_px=crop_offset,
                height=PYRAMID_HEIGHT,
                order=PYRAMID_ORDER,
                sf_bands=SF_BANDS_4,
            )
            stable_refs = {
                "natural": _movie_for_condition(
                    image=image,
                    trace=example.trace,
                    condition="stabilized",
                    t_max=t_max,
                    seed=seed,
                    crop_offset=crop_offset,
                    control_images=control_images,
                ),
                "sf_low": _movie_for_condition(
                    image=image,
                    trace=example.trace,
                    condition="stabilized_sf_low",
                    t_max=t_max,
                    seed=seed,
                    crop_offset=crop_offset,
                    control_images=control_images,
                ),
                "sf_mid_low": _movie_for_condition(
                    image=image,
                    trace=example.trace,
                    condition="stabilized_sf_mid_low",
                    t_max=t_max,
                    seed=seed,
                    crop_offset=crop_offset,
                    control_images=control_images,
                ),
                "sf_mid_high": _movie_for_condition(
                    image=image,
                    trace=example.trace,
                    condition="stabilized_sf_mid_high",
                    t_max=t_max,
                    seed=seed,
                    crop_offset=crop_offset,
                    control_images=control_images,
                ),
                "sf_high": _movie_for_condition(
                    image=image,
                    trace=example.trace,
                    condition="stabilized_sf_high",
                    t_max=t_max,
                    seed=seed,
                    crop_offset=crop_offset,
                    control_images=control_images,
                ),
                "pyramid_phase_scrambled": _movie_for_condition(
                    image=image,
                    trace=example.trace,
                    condition="stabilized_pyramid_phase_scrambled",
                    t_max=t_max,
                    seed=seed,
                    crop_offset=crop_offset,
                    control_images=control_images,
                ),
            }
            for condition in CONDITION_ORDER:
                stable_key = "natural"
                if "sf_low" in condition:
                    stable_key = "sf_low"
                elif "sf_mid_low" in condition:
                    stable_key = "sf_mid_low"
                elif "sf_mid_high" in condition:
                    stable_key = "sf_mid_high"
                elif "sf_high" in condition:
                    stable_key = "sf_high"
                elif "pyramid_phase_scrambled" in condition:
                    stable_key = "pyramid_phase_scrambled"
                movie = _movie_for_condition(
                    image=image,
                    trace=example.trace,
                    condition=condition,
                    t_max=t_max,
                    seed=seed,
                    crop_offset=crop_offset,
                    control_images=control_images,
                )
                row = {
                    "example_id": example.example_id,
                    "kind": example.kind,
                    "image_index": image_index,
                    "crop_rank": crop_rank,
                    "condition": condition,
                }
                row.update(movie_metrics(movie, stable_refs[stable_key]))
                rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    detailed = out_dir / "retinal_movie_transform_qc.csv"
    summary = out_dir / "retinal_movie_transform_qc_summary.csv"
    figure = out_dir / "retinal_movie_transform_qc.pdf"
    write_csv_rows(rows, detailed)
    summary_rows = _summary(rows)
    write_csv_rows(summary_rows, summary)
    plot_qc(summary_rows, figure)
    manifest = {
        "source_run": str(run_dir),
        "n_rows": len(rows),
        "conditions": list(CONDITION_ORDER),
        "detailed_csv": str(detailed),
        "summary_csv": str(summary),
        "figure_pdf": str(figure),
        "figure_png": str(figure.with_suffix(".png")),
    }
    with (out_dir / "retinal_movie_transform_qc_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_qc(args.run_dir, args.out_dir)
    print(f"Wrote {manifest['detailed_csv']}")
    print(f"Wrote {manifest['summary_csv']}")
    print(f"Wrote {manifest['figure_pdf']}")


if __name__ == "__main__":
    main()
