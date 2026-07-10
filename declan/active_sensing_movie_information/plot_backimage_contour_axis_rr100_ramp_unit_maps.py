#!/usr/bin/env python3
"""Rerender image-by-condition activation-map sheets for selected RR100 units."""

from __future__ import annotations

import argparse
import csv
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

from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    DEFAULT_AXIS_RUN_DIR,
    RR100_MOVIE_MEDOID_VERSION,
    combined_axis_trace,
    condition_specs,
    parse_float_list,
    rate_map_for_trace,
    select_source_trials,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
    _align_response_to_trace,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


DEFAULT_RUN_DIR = Path(
    "outputs/active_sensing_movie_information/backimage_contour_axis_rr100_spatial_ssi_n128_across_sweep_v1"
)
DEFAULT_ACROSS_SCALES = "0,0.125,0.25,0.5,0.75,1,1.5,2,3"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--axis-run-dir", type=Path, default=DEFAULT_AXIS_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--across-scales", type=str, default=DEFAULT_ACROSS_SCALES)
    parser.add_argument("--along-scale", type=float, default=1.0)
    parser.add_argument("--source-trace-scale", type=float, default=1.0)
    parser.add_argument("--source-trace-prior-family", type=str, default="axis_edge_parallel")
    parser.add_argument("--axis-column", type=str, default="image_edge_axis_deg")
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--top-units", type=int, default=6)
    parser.add_argument("--images-per-unit", type=int, default=6)
    parser.add_argument("--min-unit-std", type=float, default=1e-4)
    parser.add_argument(
        "--selection-mode",
        choices=("positive-ramp", "drop-1-to-3"),
        default="positive-ramp",
        help="Unit/image selection rule. positive-ramp uses across=0 to 3; drop-1-to-3 uses across=1 to 3.",
    )
    parser.add_argument("--include-static-baseline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--map-vmin-percentile", type=float, default=1.0)
    parser.add_argument("--map-vmax-percentile", type=float, default=99.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def identity_text(identity: dict[str, Any]) -> str:
    return json.dumps(json_ready(identity), sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_slug(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text).strip("_") or "unnamed"


def selection_slug(selection_mode: str) -> str:
    return safe_slug(selection_mode.replace("-", "_"))


def selection_label(selection_mode: str) -> str:
    if selection_mode == "positive-ramp":
        return "positive SSI ramps"
    if selection_mode == "drop-1-to-3":
        return "SSI drop-offs from 1x to 3x"
    return selection_mode


def selection_score_label(selection_mode: str) -> str:
    if selection_mode == "positive-ramp":
        return "ramp"
    if selection_mode == "drop-1-to-3":
        return "drop"
    return "score"


def image_scale(images: list[np.ndarray], vmin_percentile: float, vmax_percentile: float) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(img, dtype=np.float32).ravel() for img in images])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, float(vmin_percentile)))
    vmax = float(np.nanpercentile(finite, float(vmax_percentile)))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def load_production_cache(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "cache" / "backimage_contour_axis_rr100_spatial_ssi_cache.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def select_units(
    run_dir: Path,
    *,
    top_units: int,
    min_unit_std: float,
    selection_mode: str,
) -> tuple[list[int], pd.DataFrame]:
    path = run_dir / "backimage_contour_axis_rr100_unit_zscore_curves.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing z-score curve table: {path}. Run plot_backimage_contour_axis_rr100_zscore_curves.py first."
        )
    rows = pd.read_csv(path)
    rows = rows[np.isfinite(rows["z_slope_vs_across_scale"])].copy()
    rows = rows[rows["absolute_curve_std"].astype(float) >= float(min_unit_std)]
    if selection_mode == "positive-ramp":
        rows = rows[rows["z_slope_vs_across_scale"].astype(float) > 0.0]
        rows["positive_ramp_score"] = rows["z_slope_vs_across_scale"].astype(float) * rows[
            "absolute_dynamic_range"
        ].astype(float)
        rows = rows.sort_values(["positive_ramp_score", "absolute_dynamic_range"], ascending=False)
    elif selection_mode == "drop-1-to-3":
        rows["drop_1_to_3"] = rows["ssi_at_across_1p0"].astype(float) - rows["ssi_at_across_3p0"].astype(float)
        rows = rows[rows["drop_1_to_3"].astype(float) > 0.0]
        rows = rows.sort_values(["drop_1_to_3", "absolute_dynamic_range"], ascending=False)
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")
    if rows.empty:
        raise ValueError(f"No units matched selection mode {selection_mode!r}.")
    selected = rows.head(max(1, int(top_units))).copy()
    return [int(v) for v in selected["unit_index"].to_list()], selected


def condition_index_for_across_scale(cache: dict[str, Any], scale: float) -> int:
    is_sweep = np.asarray(cache["is_across_sweep"], dtype=bool)
    across = np.asarray(cache["across_scale"], dtype=np.float64)
    candidates = np.flatnonzero(is_sweep & np.isclose(across, float(scale), rtol=0.0, atol=1e-6))
    if candidates.size != 1:
        raise ValueError(f"Expected exactly one across-scale condition for {scale}, found {candidates.size}.")
    return int(candidates[0])


def selected_image_rows_for_units(
    cache: dict[str, Any],
    units: list[int],
    *,
    images_per_unit: int,
    selection_mode: str,
) -> tuple[dict[int, list[int]], list[dict[str, Any]]]:
    condition_id = np.asarray(cache["condition_id"]).astype(str)
    is_sweep = np.asarray(cache["is_across_sweep"], dtype=bool)
    across = np.asarray(cache["across_scale"], dtype=np.float64)
    if selection_mode == "positive-ramp":
        sweep_idx = np.flatnonzero(is_sweep)
        start_idx = int(sweep_idx[np.nanargmin(across[sweep_idx])])
        end_idx = int(sweep_idx[np.nanargmax(across[sweep_idx])])
        direction = 1.0
    elif selection_mode == "drop-1-to-3":
        start_idx = condition_index_for_across_scale(cache, 1.0)
        end_idx = condition_index_for_across_scale(cache, 3.0)
        direction = -1.0
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")
    unit_bits = np.asarray(cache["unit_bits_per_movie"], dtype=np.float64)
    source_rows = np.asarray(cache["movie_source_row"], dtype=int)
    trial_ids = np.asarray(cache["movie_trial_id"], dtype=int)
    by_unit: dict[int, list[int]] = {}
    rows: list[dict[str, Any]] = []
    for unit in units:
        delta = unit_bits[end_idx, :, int(unit)] - unit_bits[start_idx, :, int(unit)]
        score = direction * delta
        order = np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1]
        keep = [int(idx) for idx in order[: max(1, int(images_per_unit))]]
        by_unit[int(unit)] = keep
        for rank, movie_idx in enumerate(keep):
            rows.append(
                {
                    "unit_index": int(unit),
                    "unit_label": f"u{int(unit):03d}",
                    "image_rank_for_unit": int(rank + 1),
                    "movie_index": int(movie_idx),
                    "trial_id": int(trial_ids[movie_idx]),
                    "source_row": int(source_rows[movie_idx]),
                    "selection_mode": str(selection_mode),
                    "selection_score_label": selection_score_label(selection_mode),
                    "selection_start_condition_id": str(condition_id[start_idx]),
                    "selection_end_condition_id": str(condition_id[end_idx]),
                    "ssi_start": float(unit_bits[start_idx, movie_idx, int(unit)]),
                    "ssi_end": float(unit_bits[end_idx, movie_idx, int(unit)]),
                    "ssi_end_minus_start": float(delta[movie_idx]),
                    "selection_score": float(score[movie_idx]),
                }
            )
    return by_unit, rows


def cache_path(out_dir: Path) -> Path:
    return out_dir / "cache" / "backimage_contour_axis_rr100_selected_unit_image_maps.npz"


def load_map_cache(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            observed = str(np.asarray(data["cache_identity_json"]).ravel()[0])
            if observed != identity_text(identity):
                return None
            return {key: np.asarray(data[key]) for key in data.files if key != "cache_identity_json"}
    except Exception:
        return None


def save_map_cache(path: Path, payload: dict[str, Any], identity: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload, cache_identity_json=np.asarray([identity_text(identity)]))


def compute_maps(
    args: argparse.Namespace,
    *,
    trials: pd.DataFrame,
    movie_indices: list[int],
    specs: list[dict[str, Any]],
    units: list[int],
) -> dict[str, Any]:
    view = load_population_view(version_name=str(args.rr100_version))
    scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(args.batch_size), empty_cache_every_batch=True)
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    maps = None
    rows: list[dict[str, Any]] = []
    total = len(movie_indices) * len(specs)
    done = 0
    for movie_pos, movie_idx in enumerate(movie_indices):
        trial = trials.iloc[int(movie_idx)]
        patch, _patch_meta = _extract_patch(
            trial,
            canvas_cache=canvas_cache,
            patch_size_px=int(args.patch_size_px),
        )
        source_trace = np.asarray(trial["source_trace"], dtype=np.float32)
        axis_deg = float(trial[str(args.axis_column)])
        for condition_pos, spec in enumerate(specs):
            done += 1
            if bool(spec["is_static_baseline"]):
                trace = np.zeros_like(source_trace, dtype=np.float32)
            else:
                trace, _trace_meta = combined_axis_trace(
                    source_trace,
                    axis_deg=axis_deg,
                    along_scale=float(spec["along_scale"]),
                    across_scale=float(spec["across_scale"]),
                )
            print(
                f"[backimage-ramp-unit-maps] {done}/{total} movie={movie_idx} "
                f"source_row={int(trial['source_row'])} condition={spec['condition_id']}",
                flush=True,
            )
            full_map = rate_map_for_trace(scorer, patch, trace)
            full_map = _align_response_to_trace(full_map, int(args.n_timepoints))
            rr_map = apply_population_view(full_map, view).astype(np.float32, copy=False)
            mean_map = np.mean(rr_map[:, units], axis=0).astype(np.float32, copy=False)
            if maps is None:
                maps = np.zeros(
                    (len(movie_indices), len(specs), len(units), mean_map.shape[-2], mean_map.shape[-1]),
                    dtype=np.float32,
                )
            maps[movie_pos, condition_pos] = mean_map
            rows.append(
                {
                    "movie_index": int(movie_idx),
                    "movie_pos": int(movie_pos),
                    "condition_index": int(condition_pos),
                    "condition_id": str(spec["condition_id"]),
                    "condition_label": str(spec["condition_label"]),
                    "source_row": int(trial["source_row"]),
                    "trial_id": int(trial["trial_id"]),
                    "session": str(trial["session"]),
                    "trial_idx": int(trial["trial_idx"]),
                }
            )
            del full_map, rr_map, mean_map
    if maps is None:
        raise ValueError("No maps computed.")
    return {
        "unit_maps": maps,
        "movie_indices": np.asarray(movie_indices, dtype=np.int32),
        "unit_indices": np.asarray(units, dtype=np.int32),
        "condition_id": np.asarray([str(spec["condition_id"]) for spec in specs]),
        "condition_label": np.asarray([str(spec["condition_label"]) for spec in specs]),
        "condition_across_scale": np.asarray([float(spec["across_scale"]) for spec in specs], dtype=np.float32),
        "render_rows_json": np.asarray([json.dumps(rows, sort_keys=True)]),
    }


def plot_unit_sheets(
    out_dir: Path,
    maps_payload: dict[str, Any],
    *,
    selected_by_unit: dict[int, list[int]],
    selected_rows: list[dict[str, Any]],
    selection_mode: str,
    map_vmin_percentile: float,
    map_vmax_percentile: float,
    dpi: int,
) -> list[Path]:
    unit_maps = np.asarray(maps_payload["unit_maps"], dtype=np.float32)
    movie_indices = [int(v) for v in np.asarray(maps_payload["movie_indices"], dtype=int)]
    unit_indices = [int(v) for v in np.asarray(maps_payload["unit_indices"], dtype=int)]
    labels = np.asarray(maps_payload["condition_label"]).astype(str)
    condition_id = np.asarray(maps_payload["condition_id"]).astype(str)
    movie_pos_by_index = {int(movie_idx): pos for pos, movie_idx in enumerate(movie_indices)}
    unit_pos_by_index = {int(unit): pos for pos, unit in enumerate(unit_indices)}
    selected_row_lookup = {
        (int(row["unit_index"]), int(row["movie_index"])): row
        for row in selected_rows
    }
    paths: list[Path] = []
    slug = selection_slug(selection_mode)
    sheet_dir = out_dir / f"{slug}_unit_image_condition_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for unit in unit_indices:
        rows_for_unit = selected_by_unit[int(unit)]
        n_rows = len(rows_for_unit)
        n_cols = len(labels)
        fig_w = max(10.0, 1.18 * n_cols + 2.2)
        fig_h = max(4.0, 1.18 * n_rows + 1.4)
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(
            nrows=n_rows + 1,
            ncols=n_cols + 1,
            width_ratios=[1.65, *([1.0] * n_cols)],
            height_ratios=[0.28, *([1.0] * n_rows)],
            hspace=0.045,
            wspace=0.045,
        )
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.axis("off")
        ax0.text(1.0, 0.2, "image", ha="right", va="center", fontsize=7, color="#555555")
        for c, label in enumerate(labels, start=1):
            ax = fig.add_subplot(gs[0, c])
            ax.axis("off")
            ax.text(0.5, 0.2, str(label), ha="center", va="center", fontsize=7, color="#555555")
        unit_pos = unit_pos_by_index[int(unit)]
        row_images = [
            unit_maps[movie_pos_by_index[int(movie_idx)], :, unit_pos]
            for movie_idx in rows_for_unit
        ]
        vmin, vmax = image_scale(
            [img for row_stack in row_images for img in row_stack],
            float(map_vmin_percentile),
            float(map_vmax_percentile),
        )
        for r, movie_idx in enumerate(rows_for_unit, start=1):
            label_ax = fig.add_subplot(gs[r, 0])
            label_ax.axis("off")
            meta = selected_row_lookup.get((int(unit), int(movie_idx)), {})
            score_name = str(meta.get("selection_score_label", selection_score_label(selection_mode)))
            label = (
                f"rank {int(meta.get('image_rank_for_unit', r))}\n"
                f"movie {int(movie_idx)} src {int(meta.get('source_row', -1))}\n"
                f"{score_name} {float(meta.get('selection_score', np.nan)):.3g}"
            )
            label_ax.text(1.0, 0.5, label, ha="right", va="center", fontsize=6.4, color="#444444")
            movie_pos = movie_pos_by_index[int(movie_idx)]
            for c in range(n_cols):
                ax = fig.add_subplot(gs[r, c + 1])
                ax.imshow(unit_maps[movie_pos, c, unit_pos], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.55)
                    spine.set_edgecolor("#666666")
        fig.suptitle(
            f"BackImage RR100 unit u{int(unit):03d}: strongest per-image {selection_label(selection_mode)}\n"
            "columns are across-contour scale conditions with along=1; grayscale shared within sheet",
            fontsize=10.5,
            y=0.995,
        )
        png = sheet_dir / f"backimage_rr100_unit_u{int(unit):03d}_{slug}_image_condition_maps.png"
        pdf = sheet_dir / f"backimage_rr100_unit_u{int(unit):03d}_{slug}_image_condition_maps.pdf"
        fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        paths.append(png)
        paths.append(pdf)
    return paths


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = load_production_cache(run_dir)
    selection_mode = str(args.selection_mode)
    units, unit_selection = select_units(
        run_dir,
        top_units=int(args.top_units),
        min_unit_std=float(args.min_unit_std),
        selection_mode=selection_mode,
    )
    selected_by_unit, image_rows = selected_image_rows_for_units(
        cache,
        units,
        images_per_unit=int(args.images_per_unit),
        selection_mode=selection_mode,
    )
    unique_movies = sorted({int(movie_idx) for rows in selected_by_unit.values() for movie_idx in rows})
    across_scales = parse_float_list(str(args.across_scales))
    specs = condition_specs(
        across_scales,
        along_scale=float(args.along_scale),
        include_static_baseline=bool(args.include_static_baseline),
    )
    identity = {
        "analysis": "backimage_contour_axis_rr100_selected_unit_image_maps",
        "selection_mode": selection_mode,
        "run_dir": str(run_dir.resolve()),
        "axis_run_dir": str(Path(args.axis_run_dir).resolve()),
        "rr100_version": str(args.rr100_version),
        "units": units,
        "selected_by_unit": selected_by_unit,
        "unique_movies": unique_movies,
        "condition_ids": [str(spec["condition_id"]) for spec in specs],
        "n_timepoints": int(args.n_timepoints),
        "patch_size_px": int(args.patch_size_px),
    }
    slug = selection_slug(selection_mode)
    unit_selection.to_csv(out_dir / f"{slug}_unit_selection.csv", index=False)
    write_csv_rows(out_dir / f"{slug}_unit_selected_images.csv", image_rows)
    write_json(out_dir / f"{slug}_unit_image_map_request.json", identity)

    print(f"Selected units: {', '.join(f'u{u:03d}' for u in units)}")
    print(f"Unique movie indices to render: {len(unique_movies)}")
    if bool(args.dry_run):
        print("Dry run only; no model maps rendered.")
        return

    map_cache_path = cache_path(out_dir)
    maps_payload = None if bool(args.force) else load_map_cache(map_cache_path, identity)
    if maps_payload is None:
        select_args = argparse.Namespace(
            axis_run_dir=Path(args.axis_run_dir),
            source_trace_scale=float(args.source_trace_scale),
            source_trace_prior_family=str(args.source_trace_prior_family),
            axis_column=str(args.axis_column),
            max_trials=0,
            trial_start=0,
            n_timepoints=int(args.n_timepoints),
        )
        trials, _meta = select_source_trials(select_args)
        expected = np.asarray(cache["movie_trial_id"], dtype=int)
        observed = trials["trial_id"].to_numpy(dtype=int)
        if expected.shape[0] != observed.shape[0] or not np.array_equal(expected, observed):
            raise ValueError("Reconstructed trial order does not match production cache movie_trial_id.")
        maps_payload = compute_maps(
            args,
            trials=trials,
            movie_indices=unique_movies,
            specs=specs,
            units=units,
        )
        save_map_cache(map_cache_path, maps_payload, identity)
        print(f"Saved map cache: {map_cache_path}")
    else:
        print(f"Loaded map cache: {map_cache_path}")

    paths = plot_unit_sheets(
        out_dir,
        maps_payload,
        selected_by_unit=selected_by_unit,
        selected_rows=image_rows,
        selection_mode=selection_mode,
        map_vmin_percentile=float(args.map_vmin_percentile),
        map_vmax_percentile=float(args.map_vmax_percentile),
        dpi=int(args.dpi),
    )
    print(f"Wrote {len(paths)} unit sheet files under {out_dir / f'{slug}_unit_image_condition_sheets'}")


if __name__ == "__main__":
    main()
