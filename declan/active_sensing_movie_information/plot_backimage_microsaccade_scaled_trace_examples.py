#!/usr/bin/env python3
"""Plot example microsaccade traces after event-only scaling."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.plot_backimage_microsaccade_bimodal_instantaneous_maps import (
    condition_specs_from_run,
    source_args_from_run,
)
from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    combined_axis_trace,
    rotated_axis_deg,
    rotate_trace_xy,
    select_source_trials,
    trial_event_scale_mask,
)


DEFAULT_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_microsaccade_snippet_rr100_spatial_ssi_isotropic_padded_event_scaled_full_amp1sd_n40_v1"
)
DEFAULT_SCALES = "0,0.25,0.5,1,2,3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--scales", type=str, default=DEFAULT_SCALES)
    parser.add_argument("--movie-indices", type=str, default="")
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one scale is required.")
    return values


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


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


def default_movie_indices(run_dir: Path, n_examples: int) -> list[int]:
    selected = run_dir / "bimodal_unit_curve_groups" / "example_instantaneous_maps" / "selected_example_units_with_representative_movies.csv"
    if selected.exists():
        rows = pd.read_csv(selected)
        if "representative_movie_index" in rows.columns:
            out: list[int] = []
            for value in rows["representative_movie_index"].astype(int).to_list():
                if int(value) not in out:
                    out.append(int(value))
                if len(out) >= int(n_examples):
                    return out
    with np.load(run_dir / "cache" / "backimage_contour_axis_rr100_spatial_ssi_cache.npz", allow_pickle=False) as data:
        n_movies = int(np.asarray(data["movie_source_row"]).shape[0])
    if n_movies <= 0:
        raise ValueError("No movies available in run cache.")
    return [int(v) for v in np.linspace(0, n_movies - 1, min(int(n_examples), n_movies), dtype=int)]


def axis_components(trace: np.ndarray, axis_deg: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(trace, dtype=np.float64)
    theta = np.radians(float(axis_deg))
    along_u = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)
    across_u = np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    return arr @ along_u, arr @ across_u


def step_arcmin(trace: np.ndarray) -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.shape[0] < 2:
        return np.zeros((arr.shape[0],), dtype=np.float64)
    out = np.zeros((arr.shape[0],), dtype=np.float64)
    out[1:] = 60.0 * np.linalg.norm(np.diff(arr, axis=0), axis=1)
    return out


def trace_for_spec(trial: pd.Series, spec: dict[str, Any], *, axis_column: str, rotation_deg: int) -> tuple[np.ndarray, dict[str, Any], np.ndarray, float]:
    source_trace = np.asarray(trial["source_trace"], dtype=np.float32)
    if rotation_deg:
        source_trace = rotate_trace_xy(source_trace, rotation_deg)
    event_mask = trial_event_scale_mask(trial, int(source_trace.shape[0]))
    axis_deg = rotated_axis_deg(float(trial[str(axis_column)]), rotation_deg)
    trace, meta = combined_axis_trace(
        source_trace,
        axis_deg=axis_deg,
        along_scale=float(spec["along_scale"]),
        across_scale=float(spec["across_scale"]),
        event_scale_mask=event_mask,
    )
    if event_mask is None:
        event_mask = np.zeros((source_trace.shape[0],), dtype=bool)
    return trace, meta, np.asarray(event_mask, dtype=bool), axis_deg


def plot_examples(
    out_dir: Path,
    run_dir: Path,
    trials: pd.DataFrame,
    specs: list[dict[str, Any]],
    movie_indices: list[int],
    *,
    axis_column: str,
    rotation_deg: int,
    dpi: int,
) -> tuple[Path, Path, Path]:
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(specs)))
    labels = [str(spec["condition_label"]) for spec in specs]
    n_rows = len(movie_indices)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13.6, max(4.8, 2.75 * n_rows)), constrained_layout=True)
    if n_rows == 1:
        axes = np.asarray([axes])
    summary_rows: list[dict[str, Any]] = []
    for row_idx, movie_idx in enumerate(movie_indices):
        trial = trials.iloc[int(movie_idx)]
        rendered = [
            trace_for_spec(trial, spec, axis_column=axis_column, rotation_deg=rotation_deg)
            for spec in specs
        ]
        event_mask = rendered[0][2]
        event_frames = np.flatnonzero(event_mask)
        axis_deg = float(rendered[0][3])
        ax_path, ax_comp, ax_step = axes[row_idx]
        for spec, (trace, meta, _mask, _axis), color, label in zip(specs, rendered, colors, labels, strict=True):
            xy = 60.0 * np.asarray(trace, dtype=np.float64)
            along, across = axis_components(trace, axis_deg)
            step = step_arcmin(trace)
            motion_scale = float(spec.get("motion_scale", spec.get("across_scale", 0.0)))
            ax_path.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.25, alpha=0.92, label=label)
            ax_path.scatter(xy[0, 0], xy[0, 1], color=color, s=8, marker="o")
            ax_path.scatter(xy[-1, 0], xy[-1, 1], color=color, s=11, marker="x")
            if event_frames.size:
                ax_path.plot(xy[event_frames, 0], xy[event_frames, 1], color=color, linewidth=3.0, alpha=0.45)
            ax_comp.plot(60.0 * along, color=color, linewidth=1.2, alpha=0.95)
            ax_comp.plot(60.0 * across, color=color, linewidth=1.0, linestyle="--", alpha=0.85)
            ax_step.plot(step, color=color, linewidth=1.2, alpha=0.95)
            summary_rows.append(
                {
                    "movie_index": int(movie_idx),
                    "source_row": int(trial["source_row"]),
                    "condition_label": label,
                    "motion_scale": motion_scale,
                    "axis_deg": axis_deg,
                    "event_sample_count": int(np.sum(event_mask)),
                    "event_step_count": int(np.sum(event_mask[1:])) if event_mask.size > 1 else 0,
                    **{k: v for k, v in meta.items() if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_))},
                }
            )
        if event_frames.size:
            for ax in (ax_comp, ax_step):
                ax.axvspan(float(event_frames[0]) - 0.5, float(event_frames[-1]) + 0.5, color="#f2c94c", alpha=0.16, linewidth=0)
        ax_path.axhline(0.0, color="0.88", linewidth=0.7)
        ax_path.axvline(0.0, color="0.88", linewidth=0.7)
        ax_path.set_aspect("equal", adjustable="datalim")
        ax_path.set_title(f"movie {movie_idx} / source {int(trial['source_row'])}: x-y path", fontsize=9)
        ax_path.set_xlabel("x position (arcmin)")
        ax_path.set_ylabel("y position (arcmin)")
        ax_path.grid(True, color="0.92", linewidth=0.6)
        ax_comp.set_title("axis components", fontsize=9)
        ax_comp.set_xlabel("time bin")
        ax_comp.set_ylabel("arcmin")
        ax_comp.grid(True, color="0.92", linewidth=0.6)
        ax_step.set_title("step size into sample", fontsize=9)
        ax_step.set_xlabel("time bin")
        ax_step.set_ylabel("arcmin/bin")
        ax_step.grid(True, color="0.92", linewidth=0.6)
        if row_idx == 0:
            ax_path.legend(title="scale", frameon=False, fontsize=7, title_fontsize=7, loc="best")
            ax_comp.text(0.02, 0.96, "solid=along, dashed=across", transform=ax_comp.transAxes, ha="left", va="top", fontsize=7)
    fig.suptitle(
        "Example real microsaccade snippets with detected-event increments scaled\n"
        "Traces are the mean-centered condition traces sent to the twin; yellow band marks the resampled event mask",
        fontsize=12,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "backimage_microsaccade_scaled_trace_examples.png"
    pdf = out_dir / "backimage_microsaccade_scaled_trace_examples.pdf"
    csv_path = out_dir / "backimage_microsaccade_scaled_trace_examples_summary.csv"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    return png, pdf, csv_path


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir / "microsaccade_scaled_trace_examples"
    source_args = source_args_from_run(run_dir)
    trials, source_meta = select_source_trials(source_args)
    scales = parse_float_list(str(args.scales))
    specs = condition_specs_from_run(run_dir, scales)
    movie_indices = parse_int_list(str(args.movie_indices)) if str(args.movie_indices).strip() else default_movie_indices(run_dir, int(args.n_examples))
    identity = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))["identity"]
    png, pdf, csv_path = plot_examples(
        out_dir,
        run_dir,
        trials,
        specs,
        movie_indices,
        axis_column=str(identity.get("axis_column", "image_edge_axis_deg")),
        rotation_deg=int(identity.get("stimulus_rotation_deg", 0)),
        dpi=int(args.dpi),
    )
    write_json(
        out_dir / "backimage_microsaccade_scaled_trace_examples_metadata.json",
        {
            "analysis": "backimage_microsaccade_scaled_trace_examples",
            "run_dir": run_dir,
            "movie_indices": movie_indices,
            "scales": scales,
            "source_meta": source_meta,
            "png": png,
            "pdf": pdf,
            "summary_csv": csv_path,
            "trace_contract": (
                "Plots show the mean-centered condition traces sent to the twin. In event-scaled "
                "full-snippet mode, displacement steps into event-mask samples are scaled by condition; "
                "outside-event displacement increments are retained at 1x before final mean-centering."
            ),
        },
    )
    print(f"Wrote {png}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
