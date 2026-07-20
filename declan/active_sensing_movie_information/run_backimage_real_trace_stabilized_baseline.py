#!/usr/bin/env python3
"""Score zero-motion stabilized SSI baselines for a real-trace SSI matrix.

The real-trace matrix is image x trace.  A truly stabilized counterfactual is
trace-independent, so this runner scores one zero-motion 40-bin movie per
selected image and saves image-indexed baseline matrices.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import (
    score_traces_for_patch,
)
from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import (
    _extract_patch,
)
from declan.redundancy_resolved_v1_population import load_population_view
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
)


DEFAULT_MATRIX_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged"
)

OUTPUT_FILES = {
    "ssi": "stabilized_ssi_by_image.npy",
    "expected": "stabilized_expected_spikes_by_image.npy",
    "mean_rate": "stabilized_mean_rate_by_image.npy",
    "population": "stabilized_population_ssi_by_image.npy",
    "table": "stabilized_movie_feature_table.csv",
    "summary": "stabilized_baseline_summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for stabilized_* outputs. Defaults to --matrix-dir.",
    )
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[backimage-stabilized-baseline] {message}", flush=True)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def existing_outputs(out_dir: Path) -> list[Path]:
    return [out_dir / name for name in OUTPUT_FILES.values() if (out_dir / name).exists()]


def read_merged_summary_defaults(matrix_dir: Path) -> dict[str, Any]:
    path = matrix_dir / "summary.json"
    if not path.exists():
        return {}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    shard_summaries = summary.get("shard_summaries")
    if isinstance(shard_summaries, list) and shard_summaries:
        first = shard_summaries[0]
        if isinstance(first, dict):
            return first
    return summary if isinstance(summary, dict) else {}


def main() -> None:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else matrix_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    present = existing_outputs(out_dir)
    if present and not bool(args.force):
        names = ", ".join(path.name for path in present)
        raise FileExistsError(f"Baseline outputs already exist in {out_dir}: {names}. Pass --force to overwrite.")

    summary_defaults = read_merged_summary_defaults(matrix_dir)
    n_timepoints = int(summary_defaults.get("n_timepoints", args.n_timepoints))
    bin_seconds = float(summary_defaults.get("bin_seconds", args.bin_seconds))
    patch_size_px = int(summary_defaults.get("patch_size_px", args.patch_size_px))
    rr100_version = str(summary_defaults.get("rr100_version", args.rr100_version))

    image_path = matrix_dir / "image_feature_table.csv"
    unit_path = matrix_dir / "unit_feature_table.csv"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing selected image table: {image_path}")
    if not unit_path.exists():
        raise FileNotFoundError(f"Missing unit feature table: {unit_path}")
    images = pd.read_csv(image_path)
    units = pd.read_csv(unit_path)
    if "image_index" not in images.columns:
        raise ValueError(f"{image_path} must contain image_index.")

    population_view = load_population_view(version_name=rr100_version)
    if int(population_view.n_units) != int(units.shape[0]):
        raise ValueError(
            f"RR100 population has {int(population_view.n_units)} units but unit_feature_table has {units.shape[0]} rows."
        )

    scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(args.frame_batch_size), empty_cache_every_batch=False)
    zero_trace = np.zeros((n_timepoints, 2), dtype=np.float32)
    n_images = int(images.shape[0])
    n_units = int(population_view.n_units)
    ssi = np.zeros((n_images, n_units), dtype=np.float32)
    expected = np.zeros((n_images, n_units), dtype=np.float32)
    mean_rate = np.zeros((n_images, n_units), dtype=np.float32)
    population = np.zeros((n_images,), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}

    started = time.perf_counter()
    for baseline_row_index, (_, image_row) in enumerate(images.iterrows()):
        patch, patch_meta = _extract_patch(
            image_row,
            canvas_cache=canvas_cache,
            patch_size_px=patch_size_px,
        )
        image_ssi, image_expected, image_mean_rate, image_population = score_traces_for_patch(
            scorer,
            population_view,
            patch,
            [zero_trace],
            trace_batch_size=1,
            frame_batch_size=int(args.frame_batch_size),
            n_timepoints=n_timepoints,
            bin_seconds=bin_seconds,
        )
        ssi[baseline_row_index] = image_ssi[0]
        expected[baseline_row_index] = image_expected[0]
        mean_rate[baseline_row_index] = image_mean_rate[0]
        population[baseline_row_index] = image_population[0]
        rows.append(
            {
                "baseline_row_index": int(baseline_row_index),
                "image_index": int(image_row["image_index"]),
                "condition_id": "counterfactual_stabilized_zero_motion",
                "n_timepoints": int(n_timepoints),
                "bin_seconds": float(bin_seconds),
                "zero_trace_path_length_arcmin": 0.0,
                "stabilized_population_ssi": float(image_population[0]),
                "stabilized_total_expected_spikes": float(np.sum(image_expected[0], dtype=np.float64)),
                **patch_meta,
            }
        )
        progress(f"scored stabilized image {baseline_row_index + 1}/{n_images} (image_index={int(image_row['image_index'])})")

    elapsed = time.perf_counter() - started
    np.save(out_dir / OUTPUT_FILES["ssi"], ssi)
    np.save(out_dir / OUTPUT_FILES["expected"], expected)
    np.save(out_dir / OUTPUT_FILES["mean_rate"], mean_rate)
    np.save(out_dir / OUTPUT_FILES["population"], population)
    pd.DataFrame(rows).to_csv(out_dir / OUTPUT_FILES["table"], index=False)

    payload = {
        "analysis": "backimage_real_trace_stabilized_baseline",
        "matrix_dir": matrix_dir,
        "out_dir": out_dir,
        "rr100_version": rr100_version,
        "n_images": n_images,
        "n_units": n_units,
        "n_timepoints": n_timepoints,
        "bin_seconds": bin_seconds,
        "patch_size_px": patch_size_px,
        "device": str(args.device),
        "frame_batch_size": int(args.frame_batch_size),
        "elapsed_s": float(elapsed),
        "images_per_s": float(n_images / elapsed) if elapsed > 0.0 else float("nan"),
        "outputs": {key: out_dir / name for key, name in OUTPUT_FILES.items()},
        "contract": (
            "Rows are selected images in image_feature_table order. Each row is a counterfactually stabilized "
            "zero-motion movie: the same static BackImage patch rendered for n_timepoints with a zero displacement "
            "trace, scored with the same corrected time-resolved spatial SSI calculation and RR100 population view "
            "as the real-trace image x trace matrix."
        ),
    }
    write_json(out_dir / OUTPUT_FILES["summary"], payload)
    progress(f"wrote stabilized baseline to {out_dir}")
    print(json.dumps(json_ready(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
