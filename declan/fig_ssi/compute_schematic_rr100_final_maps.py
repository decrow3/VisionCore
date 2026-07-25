#!/usr/bin/env python3
"""Compute RR100 final activation maps for the SSI schematic trace pair."""

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
    RR100_MOVIE_MEDOID_VERSION,
    STIMULUS_NORMALIZATION,
    rate_map_for_trace,
)
from declan.fig_ssi.make_ssi_contour_schematic import (
    MODEL_INPUT_N_LAGS,
    MODEL_SOURCE_PATCH_SIZE_PX,
    SCHEMATIC_NEW_BANK_IMAGE_INDEX,
    SCHEMATIC_PATCH_SIZE_PX,
    endpoint_stabilized_trace,
    lag_trace,
    load_new_bank_stimulus_patch,
)
from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
    _align_response_to_trace,
)
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _clip_patch
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


OUT_DIR = ROOT / "outputs" / "fig_ssi" / "rr100_schematic_endpoint_final_maps"
RUN_DIR = ROOT / "outputs" / "active_sensing_movie_information" / "backimage_rr100_instantaneous_unit_maps_latest_v1"
SF_GROUP_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
    "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)
ORIENTATION_GROUP_CSV = RUN_DIR / "orientation_tuning_groups.csv"
EPS = 1e-12
CONDITION_IDS = np.asarray(["real_trace_final", "endpoint_stabilized_final"])
CONDITION_LABELS = np.asarray(["real FEM final", "endpoint-stabilized final"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--force", action="store_true")
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def cache_path(out_dir: Path) -> Path:
    return out_dir / "cache" / "schematic_rr100_final_maps.npz"


def raw_source_patch(stimulus: dict[str, Any]) -> tuple[np.ndarray, float, tuple[int, int]]:
    row = stimulus["row"]
    canvas, ppd, screen_shape = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
    center = (float(row["image_patch_center_x_px"]), float(row["image_patch_center_y_px"]))
    return _clip_patch(canvas, center, MODEL_SOURCE_PATCH_SIZE_PX), float(ppd), screen_shape


def spatial_ssi_single_map(image: np.ndarray) -> float:
    rate = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    mean_rate = float(np.nanmean(rate))
    if not np.isfinite(mean_rate) or mean_rate <= EPS:
        return 0.0
    gain = rate / mean_rate
    return float(np.nanmean(gain * np.log2(gain + EPS)))


def map_axis_deg(image: np.ndarray) -> tuple[float, float, float, float]:
    arr = np.asarray(image, dtype=np.float64)
    baseline = float(np.nanpercentile(arr, 20.0))
    weights = np.clip(arr - baseline, 0.0, None)
    total = float(np.nansum(weights))
    if not np.isfinite(total) or total <= EPS:
        return float("nan"), 0.0, float(np.nanmean(arr)), 0.0
    yy, xx = np.mgrid[: arr.shape[0], : arr.shape[1]]
    cx = float(np.nansum(weights * xx) / total)
    cy = float(np.nansum(weights * yy) / total)
    dx = xx - cx
    dy = yy - cy
    cov_xx = float(np.nansum(weights * dx * dx) / total)
    cov_yy = float(np.nansum(weights * dy * dy) / total)
    cov_xy = float(np.nansum(weights * dx * dy) / total)
    theta = 0.5 * math.degrees(math.atan2(2.0 * cov_xy, cov_xx - cov_yy))
    vals = np.linalg.eigvalsh(np.asarray([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64))
    denom = float(np.sum(vals))
    anisotropy = float((vals[-1] - vals[0]) / denom) if denom > EPS else 0.0
    mean_rate = float(np.nanmean(arr))
    return theta, anisotropy, mean_rate, spatial_ssi_single_map(arr)


def axis_distance_deg(a_deg: float, b_deg: float) -> float:
    return float(abs(((float(a_deg) - float(b_deg) + 90.0) % 180.0) - 90.0))


def load_unit_annotations() -> pd.DataFrame:
    rows = pd.DataFrame({"unit_index": np.arange(100, dtype=int), "unit_label": [f"u{i:03d}" for i in range(100)]})
    if SF_GROUP_CSV.exists():
        sf = pd.read_csv(SF_GROUP_CSV)
        keep = ["unit_index", "unit_label", "sf_group", "sf_split_metric", "sf_rank_low_to_high"]
        rows = rows.merge(sf[[c for c in keep if c in sf.columns]], on=["unit_index", "unit_label"], how="left")
    if ORIENTATION_GROUP_CSV.exists():
        ori = pd.read_csv(ORIENTATION_GROUP_CSV)
        keep = [
            "unit_index",
            "unit_label",
            "orientation_group",
            "orientation_group_label",
            "preferred_orientation_deg",
            "preferred_delta_from_contour_deg",
            "preferred_delta_from_across_deg",
            "orientation_selectivity_index",
        ]
        rows = rows.merge(ori[[c for c in keep if c in ori.columns]], on=["unit_index", "unit_label"], how="left")
    return rows


def unit_metric_rows(final_maps: np.ndarray, contour_axis_image_deg: float) -> list[dict[str, Any]]:
    annotations = load_unit_annotations().set_index("unit_index", drop=False)
    real_maps = np.maximum(np.asarray(final_maps[0], dtype=np.float64), 0.0)
    stable_maps = np.maximum(np.asarray(final_maps[1], dtype=np.float64), 0.0)
    rows: list[dict[str, Any]] = []
    for unit in range(real_maps.shape[0]):
        real_axis, real_anis, real_mean, real_ssi = map_axis_deg(real_maps[unit])
        stable_axis, stable_anis, stable_mean, stable_ssi = map_axis_deg(stable_maps[unit])
        delta_ssi = float(real_ssi - stable_ssi)
        delta_mean = float(real_mean - stable_mean)
        if np.isfinite(real_axis):
            contour_delta = axis_distance_deg(real_axis, contour_axis_image_deg)
        else:
            contour_delta = float("nan")
        figure_candidate_score = (
            max(delta_ssi, 0.0)
            * max(real_anis, 0.0)
            * math.sqrt(max(real_mean, 0.0) + EPS)
            * math.exp(-0.5 * (min(contour_delta, abs(contour_delta - 90.0)) / 45.0) ** 2)
            if np.isfinite(contour_delta)
            else 0.0
        )
        annotated = annotations.loc[unit].to_dict() if unit in annotations.index else {}
        row = {
            "unit_index": int(unit),
            "unit_label": f"u{unit:03d}",
            "real_final_mean_rate": float(real_mean),
            "stable_final_mean_rate": float(stable_mean),
            "real_minus_stable_mean_rate": delta_mean,
            "real_final_map_ssi_bits_per_spike": float(real_ssi),
            "stable_final_map_ssi_bits_per_spike": float(stable_ssi),
            "real_minus_stable_map_ssi": delta_ssi,
            "real_final_axis_deg_image": float(real_axis),
            "stable_final_axis_deg_image": float(stable_axis),
            "real_final_axis_anisotropy": float(real_anis),
            "stable_final_axis_anisotropy": float(stable_anis),
            "real_final_axis_abs_delta_from_contour_deg": float(contour_delta),
            "figure_candidate_score": float(figure_candidate_score),
        }
        for key, value in annotated.items():
            if key not in row:
                row[key] = value
        rows.append(row)
    return rows


def image_limits(images: list[np.ndarray]) -> tuple[float, float]:
    flat = np.concatenate([np.maximum(np.asarray(img, dtype=np.float64), 0.0).ravel() for img in images])
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0
    vmax = float(np.nanpercentile(flat, 99.2))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(flat))
    return 0.0, max(vmax, EPS)


def save_contact_sheet(out_dir: Path, final_maps: np.ndarray, metrics: pd.DataFrame, *, top_n: int, dpi: int) -> Path:
    top = metrics.sort_values("figure_candidate_score", ascending=False).head(int(top_n)).copy()
    n = int(top.shape[0])
    if n == 0:
        raise ValueError("No units available for contact sheet.")
    n_cols = 6
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(2.05 * n_cols, 1.42 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for pos, row in enumerate(top.itertuples(index=False)):
        unit = int(row.unit_index)
        r = pos // n_cols
        c = (pos % n_cols) * 2
        vmin, vmax = image_limits([final_maps[0, unit], final_maps[1, unit]])
        for j, title in enumerate(["real", "stable"]):
            ax = axes[r, c + j]
            ax.imshow(final_maps[j, unit], cmap="bone_r", vmin=vmin, vmax=vmax, interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_edgecolor("#444")
            if j == 0:
                sf_group = getattr(row, "sf_group", "")
                ax.set_title(f"u{unit:03d} {sf_group}\nΔSSI {float(row.real_minus_stable_map_ssi):.3f}", fontsize=6.2)
            else:
                ax.set_title(title, fontsize=6.2)
    fig.tight_layout(pad=0.45, w_pad=0.04, h_pad=0.35)
    png = out_dir / "schematic_rr100_final_map_candidate_sheet.png"
    pdf = out_dir / "schematic_rr100_final_map_candidate_sheet.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, dpi=int(dpi), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png


def compute_maps(args: argparse.Namespace) -> dict[str, Any]:
    stimulus = load_new_bank_stimulus_patch(SCHEMATIC_NEW_BANK_IMAGE_INDEX)
    if stimulus is None:
        raise RuntimeError("Could not load schematic BackImage stimulus.")
    real_trace = lag_trace(stimulus.get("real_trace_lag32"), MODEL_INPUT_N_LAGS)
    if real_trace is None:
        real_trace = lag_trace(stimulus.get("real_trace_center40"), MODEL_INPUT_N_LAGS)
    if real_trace is None:
        raise RuntimeError("Could not load the schematic real trace.")
    stable_trace = endpoint_stabilized_trace(real_trace)
    if stable_trace is None:
        raise RuntimeError("Could not construct the endpoint-stabilized trace.")

    patch, ppd, screen_shape = raw_source_patch(stimulus)
    view = load_population_view(version_name=str(args.rr100_version))
    scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(args.batch_size), empty_cache_every_batch=True)

    final_maps = []
    aligned_lengths = []
    for condition_id, trace in zip(CONDITION_IDS, [real_trace, stable_trace]):
        print(f"[schematic-rr100] computing {condition_id} on {args.device}", flush=True)
        full_map = rate_map_for_trace(scorer, patch, trace)
        aligned = _align_response_to_trace(full_map, int(trace.shape[0]))
        rr100 = apply_population_view(aligned, view).astype(np.float32, copy=False)
        final_maps.append(rr100[-1].astype(np.float32, copy=False))
        aligned_lengths.append(int(rr100.shape[0]))
        del full_map, aligned, rr100

    final_maps_arr = np.stack(final_maps, axis=0).astype(np.float32)
    row = stimulus["row"]
    meta = {
        "analysis": "fig_ssi_schematic_rr100_endpoint_final_maps",
        "image_index": int(stimulus["image_index"]),
        "source_row": int(stimulus["source_row"]),
        "session": str(row["session"]),
        "trial_idx": int(row["trial_idx"]),
        "trace_samples": int(real_trace.shape[0]),
        "model_input_n_lags": int(MODEL_INPUT_N_LAGS),
        "model_crop_size_px": int(SCHEMATIC_PATCH_SIZE_PX),
        "source_patch_size_px": int(MODEL_SOURCE_PATCH_SIZE_PX),
        "ppd": float(ppd),
        "screen_shape": [int(screen_shape[0]), int(screen_shape[1])],
        "rr100_version": str(view.name),
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "aligned_response_lengths": aligned_lengths,
        "condition_ids": CONDITION_IDS.tolist(),
    }
    metrics = unit_metric_rows(final_maps_arr, float(stimulus["contour_axis_image_deg"]))
    return {
        "final_maps": final_maps_arr,
        "condition_traces": np.stack([real_trace, stable_trace], axis=0).astype(np.float32),
        "condition_id": CONDITION_IDS,
        "condition_label": CONDITION_LABELS,
        "rr100_version": np.asarray([str(view.name)]),
        "stimulus_normalization": np.asarray([STIMULUS_NORMALIZATION]),
        "image_index": np.asarray([int(stimulus["image_index"])], dtype=np.int32),
        "source_row": np.asarray([int(stimulus["source_row"])], dtype=np.int32),
        "contour_axis_deg": np.asarray([float(stimulus["contour_axis_deg"])], dtype=np.float32),
        "contour_axis_image_deg": np.asarray([float(stimulus["contour_axis_image_deg"])], dtype=np.float32),
        "patch_meta_json": np.asarray([json.dumps(json_ready(meta), sort_keys=True)]),
        "meta": meta,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    path = cache_path(out_dir)
    metrics_path = out_dir / "schematic_rr100_final_map_unit_metrics.csv"
    summary_path = out_dir / "summary.json"
    if path.exists() and metrics_path.exists() and not bool(args.force):
        print(f"[schematic-rr100] using existing cache: {path}", flush=True)
        return

    payload = compute_maps(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        final_maps=payload["final_maps"],
        condition_traces=payload["condition_traces"],
        condition_id=payload["condition_id"],
        condition_label=payload["condition_label"],
        rr100_version=payload["rr100_version"],
        stimulus_normalization=payload["stimulus_normalization"],
        image_index=payload["image_index"],
        source_row=payload["source_row"],
        contour_axis_deg=payload["contour_axis_deg"],
        contour_axis_image_deg=payload["contour_axis_image_deg"],
        patch_meta_json=payload["patch_meta_json"],
    )
    write_csv(metrics_path, payload["metrics"])
    metrics_df = pd.DataFrame(payload["metrics"])
    sheet = save_contact_sheet(out_dir, payload["final_maps"], metrics_df, top_n=int(args.top_n), dpi=int(args.dpi))
    top = metrics_df.sort_values("figure_candidate_score", ascending=False).head(10)
    write_json(
        summary_path,
        {
            **payload["meta"],
            "cache_npz": path,
            "unit_metrics_csv": metrics_path,
            "candidate_sheet_png": sheet,
            "top_units": top[
                [
                    "unit_index",
                    "unit_label",
                    "sf_group",
                    "orientation_group",
                    "real_minus_stable_map_ssi",
                    "real_final_map_ssi_bits_per_spike",
                    "stable_final_map_ssi_bits_per_spike",
                    "figure_candidate_score",
                ]
            ].to_dict(orient="records"),
        },
    )
    print(f"[schematic-rr100] wrote {path}", flush=True)
    print(f"[schematic-rr100] wrote {metrics_path}", flush=True)
    print(f"[schematic-rr100] wrote {sheet}", flush=True)


if __name__ == "__main__":
    main()
