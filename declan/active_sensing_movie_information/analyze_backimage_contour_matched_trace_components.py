#!/usr/bin/env python3
"""Condition contour-matched BackImage SSI on eye-motion components.

This is a post hoc analysis of the 100k image x real-trace SSI matrix.  It
keeps the unit-image contour match fixed, decomposes each trace into motion
components parallel/perpendicular to the local image contour axis, and asks
whether SSI changes with those components.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.analyze_backimage_real_trace_ssi_matrix_phase1_phase2 import (
    DEFAULT_DENSE_TUNING_CSV,
    DEFAULT_DROP_CSV,
    DEFAULT_MATRIX_DIR,
    DEFAULT_ORIENTATION_GROUPS_CSV,
    DEFAULT_STRONG_RAMP_CSV,
    DEFAULT_TOP_DELTA_CSV,
    DEFAULT_TRACE_PATH_CONTEXT_REFERENCE_CSV,
    EPS,
    HIGH_RED,
    LOW_BLUE,
    add_trace_path_context_bands,
    axis_delta_deg,
    baseline_rows_for_image_indices,
    color_for_label,
    contour_matched_definition_text,
    enrich_unit_table,
    load_stabilized_baseline,
    load_trace_path_context_reference,
    parse_csv_list,
    save_json,
    sem,
)


DEFAULT_OUT_DIR = DEFAULT_MATRIX_DIR / "phase1_phase2_conditioning_v1" / "trace_component_conditioning_v1"
COMPONENT_SPECS = [
    ("across_path_arcmin", "across_path_bin", "Across-contour path", "across-contour path bin median (arcmin)", "arcmin"),
    ("along_path_arcmin", "along_path_bin", "Along-contour path", "along-contour path bin median (arcmin)", "arcmin"),
    ("across_rms_arcmin", "across_rms_bin", "Across-contour RMS excursion", "across-contour RMS bin median (arcmin)", "arcmin"),
    ("across_path_fraction", "across_fraction_bin", "Across-contour path fraction", "across-contour path fraction bin median", "fraction"),
]
RELATION_LABELS = {
    "matched": "matched contours",
    "all": "all contours",
    "orthogonal": "orthogonal contours",
}
RELATION_ORDER = ["matched", "all", "orthogonal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dense-tuning-csv", type=Path, default=DEFAULT_DENSE_TUNING_CSV)
    parser.add_argument("--strong-ramp-csv", type=Path, default=DEFAULT_STRONG_RAMP_CSV)
    parser.add_argument("--drop-unit-csv", type=Path, default=DEFAULT_DROP_CSV)
    parser.add_argument("--top-delta-csv", type=Path, default=DEFAULT_TOP_DELTA_CSV)
    parser.add_argument("--orientation-groups-csv", type=Path, default=DEFAULT_ORIENTATION_GROUPS_CSV)
    parser.add_argument("--trace-path-context-reference-csv", type=Path, default=DEFAULT_TRACE_PATH_CONTEXT_REFERENCE_CSV)
    parser.add_argument("--image-axis-col", type=str, default="image_edge_axis_deg")
    parser.add_argument("--sf-groups", type=str, default="low_sf,high_sf")
    parser.add_argument("--match-max-deg", type=float, default=22.5)
    parser.add_argument("--orthogonal-min-deg", type=float, default=None)
    parser.add_argument("--relation-modes", type=str, default="matched,all,orthogonal")
    parser.add_argument("--min-osi", type=float, default=0.05)
    parser.add_argument("--min-matched-images-per-unit", type=int, default=1)
    parser.add_argument("--n-component-bins", type=int, default=6)
    parser.add_argument("--n-total-path-bins", type=int, default=6)
    parser.add_argument("--n-across-fraction-control-bins", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[backimage-contour-trace-components] {message}", flush=True)


def load_inputs(matrix_dir: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    ssi = np.load(matrix_dir / "ssi_matrix.npy", mmap_mode="r")
    expected_path = matrix_dir / "expected_spikes_matrix.npy"
    mean_rate_path = matrix_dir / "mean_rate_matrix.npy"
    expected = np.load(expected_path, mmap_mode="r") if expected_path.exists() else None
    mean_rate = np.load(mean_rate_path, mmap_mode="r") if mean_rate_path.exists() else None
    movie = pd.read_csv(matrix_dir / "movie_feature_table.csv")
    trace = pd.read_csv(matrix_dir / "trace_feature_table.csv")
    unit = pd.read_csv(matrix_dir / "unit_feature_table.csv")
    trace_xy = np.load(matrix_dir / "trace_xy.npy", mmap_mode="r")
    if ssi.shape[0] != movie.shape[0]:
        raise ValueError(f"SSI rows {ssi.shape[0]} do not match movie rows {movie.shape[0]}.")
    if ssi.shape[1] != unit.shape[0]:
        raise ValueError(f"SSI columns {ssi.shape[1]} do not match unit rows {unit.shape[0]}.")
    if trace_xy.shape[0] != trace.shape[0]:
        raise ValueError(f"trace_xy rows {trace_xy.shape[0]} do not match trace rows {trace.shape[0]}.")
    return ssi, expected, mean_rate, movie, trace, unit, trace_xy


def microsaccade_count(frame: pd.DataFrame) -> pd.Series:
    for col in ("rendered_n_microsaccade_events", "n_microsaccade_events", "source_n_microsaccade_events"):
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").fillna(0).clip(lower=0).astype(int)
    return pd.Series(np.zeros(frame.shape[0], dtype=int), index=frame.index)


def coalesce_numeric(frame: pd.DataFrame, columns: list[str], default: float = float("nan")) -> pd.Series:
    out = pd.Series(default, index=frame.index, dtype=float)
    for col in columns:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        out = out.where(out.notna(), values)
    return out


def add_quantile_bin(frame: pd.DataFrame, source_col: str, out_col: str, *, n_bins: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = frame.copy()
    out[out_col] = pd.NA
    values = pd.to_numeric(out[source_col], errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return out, []
    q = min(int(n_bins), int(finite.nunique(dropna=True)))
    if q <= 1:
        out.loc[finite.index, out_col] = "q01"
    else:
        codes = pd.qcut(finite, q=q, labels=False, duplicates="drop")
        out.loc[finite.index, out_col] = [f"q{int(code) + 1:02d}" if pd.notna(code) else pd.NA for code in codes]
    rows: list[dict[str, Any]] = []
    for order, (label, sub) in enumerate(out[out[out_col].notna()].groupby(out_col, sort=True), start=1):
        vals = pd.to_numeric(sub[source_col], errors="coerce").dropna()
        rows.append(
            {
                "metric": source_col,
                "bin_col": out_col,
                "bin_label": str(label),
                "bin_order": int(order),
                "n": int(vals.shape[0]),
                "low": float(vals.min()),
                "q25": float(vals.quantile(0.25)),
                "median": float(vals.median()),
                "q75": float(vals.quantile(0.75)),
                "high": float(vals.max()),
            }
        )
    return out, rows


def ordered_relation_modes(raw_modes: list[str]) -> list[str]:
    cleaned = []
    for mode in raw_modes:
        key = str(mode).strip().lower()
        if not key:
            continue
        if key not in RELATION_LABELS:
            raise ValueError(f"Unknown relation mode {mode!r}; expected one of {sorted(RELATION_LABELS)}.")
        if key not in cleaned:
            cleaned.append(key)
    return [mode for mode in RELATION_ORDER if mode in cleaned] + [mode for mode in cleaned if mode not in RELATION_ORDER]


def relation_mask(deltas_deg: np.ndarray, relation: str, *, match_max_deg: float, orthogonal_min_deg: float) -> np.ndarray:
    finite = np.isfinite(deltas_deg)
    if relation == "all":
        return finite
    if relation == "matched":
        return finite & (deltas_deg <= float(match_max_deg))
    if relation == "orthogonal":
        return finite & (deltas_deg >= float(orthogonal_min_deg))
    raise ValueError(f"Unknown relation mode {relation!r}.")


def relation_threshold_text(relation: str, *, match_max_deg: float, orthogonal_min_deg: float) -> str:
    if relation == "matched":
        return f"align <= {match_max_deg:g} deg"
    if relation == "orthogonal":
        return f"delta >= {orthogonal_min_deg:g} deg"
    if relation == "all":
        return "all contour deltas"
    return str(relation)


def component_reference_rows(metrics: pd.DataFrame, metric_cols: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    has_ms = metrics["has_microsaccade"].fillna(False).astype(bool)
    for metric in metric_cols:
        if metric not in metrics.columns:
            continue
        for mask, label, display in [(~has_ms, "no_microsaccade", "drift-only"), (has_ms, "microsaccade", "microsaccade")]:
            vals = pd.to_numeric(metrics.loc[mask, metric], errors="coerce").dropna()
            if vals.empty:
                continue
            rows.append(
                {
                    "metric": metric,
                    "trace_component_context": label,
                    "display_label": display,
                    "has_microsaccade": bool(label == "microsaccade"),
                    "n_movie_rows": int(vals.shape[0]),
                    "low": float(vals.min()),
                    "q25": float(vals.quantile(0.25)),
                    "q40": float(vals.quantile(0.40)),
                    "median": float(vals.median()),
                    "q60": float(vals.quantile(0.60)),
                    "q75": float(vals.quantile(0.75)),
                    "high": float(vals.max()),
                }
            )
    return pd.DataFrame(rows)


def add_component_reference_strips(ax: Any, reference: pd.DataFrame, metric: str, *, include_legend: bool = True) -> None:
    if reference.empty:
        return
    sub = reference[reference["metric"].astype(str).eq(str(metric))].copy()
    if sub.empty:
        return
    styles = {
        "no_microsaccade": {"color": "#8c8c8c", "alpha": 0.24, "line_alpha": 0.62, "linestyle": "-", "y0": 0.940, "y1": 0.982},
        "microsaccade": {"color": "#5f5f5f", "alpha": 0.20, "line_alpha": 0.70, "linestyle": "--", "y0": 0.888, "y1": 0.930},
    }
    trans = ax.get_xaxis_transform()
    for row in sub.sort_values("median").itertuples(index=False):
        key = str(getattr(row, "trace_component_context", ""))
        style = styles.get(key, styles["no_microsaccade"])
        low = float(getattr(row, "q25", np.nan))
        high = float(getattr(row, "q75", np.nan))
        median = float(getattr(row, "median", np.nan))
        if not (math.isfinite(low) and math.isfinite(high) and high > low):
            continue
        display = str(getattr(row, "display_label", key))
        rect = patches.Rectangle(
            (low, style["y0"]),
            high - low,
            style["y1"] - style["y0"],
            transform=trans,
            facecolor=style["color"],
            edgecolor="none",
            alpha=style["alpha"],
            zorder=3,
            label=f"{display} ref IQR" if include_legend else None,
        )
        ax.add_patch(rect)
        if math.isfinite(median):
            ax.plot(
                [median, median],
                [style["y0"], style["y1"]],
                transform=trans,
                color=style["color"],
                alpha=style["line_alpha"],
                linestyle=style["linestyle"],
                linewidth=1.1,
                zorder=4,
            )


def compute_component_movie_metrics(
    movie: pd.DataFrame,
    trace: pd.DataFrame,
    trace_xy: np.ndarray,
    *,
    image_axis_col: str,
) -> pd.DataFrame:
    required = {"movie_index", "image_index", "trace_index", image_axis_col}
    missing = sorted(required.difference(movie.columns))
    if missing:
        raise ValueError(f"Movie table is missing required columns for component metrics: {missing}")
    n_time = int(trace_xy.shape[1])
    trace_index = movie["trace_index"].astype(int).to_numpy()
    if np.any((trace_index < 0) | (trace_index >= trace_xy.shape[0])):
        raise ValueError("movie.trace_index contains values outside trace_xy.")
    axes = pd.to_numeric(movie[image_axis_col], errors="coerce").to_numpy(dtype=float)
    theta = np.radians(axes)
    cos_t = np.cos(theta).astype(np.float32)
    sin_t = np.sin(theta).astype(np.float32)

    xy = np.asarray(trace_xy, dtype=np.float32)
    steps = np.diff(xy, axis=1)
    selected_steps = steps[trace_index]
    along_step = selected_steps[:, :, 0] * cos_t[:, None] + selected_steps[:, :, 1] * sin_t[:, None]
    across_step = -selected_steps[:, :, 0] * sin_t[:, None] + selected_steps[:, :, 1] * cos_t[:, None]
    total_step = np.sqrt(np.sum(selected_steps.astype(np.float64) ** 2, axis=2))

    selected_xy = xy[trace_index]
    centered_xy = selected_xy - np.nanmean(selected_xy, axis=1, keepdims=True)
    along_pos = centered_xy[:, :, 0] * cos_t[:, None] + centered_xy[:, :, 1] * sin_t[:, None]
    across_pos = -centered_xy[:, :, 0] * sin_t[:, None] + centered_xy[:, :, 1] * cos_t[:, None]

    durations = coalesce_numeric(trace, ["snippet_duration_s", "rendered_duration_s", "duration_s"], default=np.nan).to_numpy(dtype=float)
    if not np.isfinite(durations).any():
        durations = np.full(trace.shape[0], (n_time - 1) / 120.0, dtype=float)
    median_duration = float(np.nanmedian(durations[np.isfinite(durations)]))
    durations = np.where(np.isfinite(durations) & (durations > 0), durations, median_duration)
    dt = durations / max(n_time - 1, 1)
    threshold = coalesce_numeric(
        trace,
        ["rendered_microsaccade_threshold_dps", "microsaccade_threshold_dps", "source_microsaccade_threshold_dps"],
        default=np.nan,
    ).to_numpy(dtype=float)
    finite_threshold = threshold[np.isfinite(threshold) & (threshold > 0)]
    fallback_threshold = float(np.nanmedian(finite_threshold)) if finite_threshold.size else float("inf")
    threshold = np.where(np.isfinite(threshold) & (threshold > 0), threshold, fallback_threshold)
    speed = np.sqrt(np.sum(steps.astype(np.float64) ** 2, axis=2)) / np.maximum(dt[:, None], EPS)
    trace_ms_step = speed >= threshold[:, None]
    selected_ms_step = trace_ms_step[trace_index]

    along_abs = np.abs(along_step)
    across_abs = np.abs(across_step)
    total_path_arcmin = np.sum(total_step, axis=1) * 60.0
    along_path_arcmin = np.sum(along_abs, axis=1) * 60.0
    across_path_arcmin = np.sum(across_abs, axis=1) * 60.0
    component_l1_arcmin = along_path_arcmin + across_path_arcmin
    with np.errstate(divide="ignore", invalid="ignore"):
        across_path_fraction = np.divide(across_path_arcmin, component_l1_arcmin)
        across_to_along_path_ratio = np.divide(across_path_arcmin, along_path_arcmin)

    along_ms_path = np.sum(np.where(selected_ms_step, along_abs, 0.0), axis=1) * 60.0
    across_ms_path = np.sum(np.where(selected_ms_step, across_abs, 0.0), axis=1) * 60.0
    along_drift_path = np.sum(np.where(~selected_ms_step, along_abs, 0.0), axis=1) * 60.0
    across_drift_path = np.sum(np.where(~selected_ms_step, across_abs, 0.0), axis=1) * 60.0

    trace_has_ms = microsaccade_count(trace) > 0
    out = pd.DataFrame(
        {
            "movie_index": movie["movie_index"].astype(int).to_numpy(),
            "image_index": movie["image_index"].astype(int).to_numpy(),
            "trace_index": trace_index,
            image_axis_col: axes,
            "trace_path_length_bin": movie.get("trace_path_length_bin", pd.Series(pd.NA, index=movie.index)).astype(object).to_numpy(),
            "has_microsaccade": trace_has_ms.iloc[trace_index].astype(bool).to_numpy(),
            "rendered_n_microsaccade_events": microsaccade_count(trace).iloc[trace_index].astype(int).to_numpy(),
            "rendered_path_length_arcmin": pd.to_numeric(movie["rendered_path_length_arcmin"], errors="coerce").to_numpy(dtype=float),
            "component_total_path_arcmin": total_path_arcmin,
            "component_l1_path_arcmin": component_l1_arcmin,
            "along_path_arcmin": along_path_arcmin,
            "across_path_arcmin": across_path_arcmin,
            "across_path_fraction": across_path_fraction,
            "across_to_along_path_ratio": across_to_along_path_ratio,
            "along_rms_arcmin": np.sqrt(np.nanmean(along_pos.astype(np.float64) ** 2, axis=1)) * 60.0,
            "across_rms_arcmin": np.sqrt(np.nanmean(across_pos.astype(np.float64) ** 2, axis=1)) * 60.0,
            "along_net_arcmin": np.abs(np.sum(along_step, axis=1)) * 60.0,
            "across_net_arcmin": np.abs(np.sum(across_step, axis=1)) * 60.0,
            "along_drift_path_arcmin": along_drift_path,
            "across_drift_path_arcmin": across_drift_path,
            "along_microsaccade_path_arcmin": along_ms_path,
            "across_microsaccade_path_arcmin": across_ms_path,
            "microsaccade_step_fraction": selected_ms_step.mean(axis=1),
        }
    )
    return out


def build_component_unit_curves(
    metrics: pd.DataFrame,
    ssi: np.ndarray,
    mean_rate: np.ndarray | None,
    unit: pd.DataFrame,
    *,
    image_axis_col: str,
    sf_groups: list[str],
    match_max_deg: float,
    orthogonal_min_deg: float,
    min_osi: float,
    min_matched_images_per_unit: int,
    relation_modes: list[str],
    component_specs: list[tuple[str, str, str, str, str]],
    stabilized_baseline: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image_axis = (
        metrics[["image_index", image_axis_col]]
        .drop_duplicates("image_index")
        .sort_values("image_index")
        .reset_index(drop=True)
    )
    image_axis[image_axis_col] = pd.to_numeric(image_axis[image_axis_col], errors="coerce")
    movie_images = metrics["image_index"].astype(int).to_numpy()
    row_index = np.arange(metrics.shape[0], dtype=int)
    baseline_ssi = stabilized_baseline["ssi"] if stabilized_baseline is not None else None
    baseline_rate = stabilized_baseline.get("mean_rate") if stabilized_baseline is not None else None
    selection_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    use_units = unit[unit["sf_group"].astype(str).isin(sf_groups)].copy()
    use_units = use_units.sort_values(["sf_group", "unit_index"], kind="mergesort")
    for unit_row in use_units.itertuples(index=False):
        unit_index = int(unit_row.unit_index)
        if unit_index < 0 or unit_index >= ssi.shape[1]:
            continue
        sf_group = str(unit_row.sf_group)
        pref = float(unit_row.analysis_preferred_orientation_deg)
        osi = float(unit_row.prior_orientation_selectivity_index)
        orientation_ok = math.isfinite(pref) and math.isfinite(osi) and osi >= float(min_osi)
        if orientation_ok:
            deltas = axis_delta_deg(image_axis[image_axis_col].to_numpy(dtype=float), pref)
        else:
            deltas = np.full(image_axis.shape[0], np.nan)
        for relation in relation_modes:
            selected_mask = (
                relation_mask(deltas, relation, match_max_deg=float(match_max_deg), orthogonal_min_deg=float(orthogonal_min_deg))
                if orientation_ok
                else np.zeros(image_axis.shape[0], dtype=bool)
            )
            selected_images = image_axis.loc[selected_mask, "image_index"].astype(int).to_numpy()
            selection_rows.append(
                {
                    "relation": relation,
                    "relation_label": RELATION_LABELS.get(relation, relation),
                    "relation_definition": relation_threshold_text(
                        relation,
                        match_max_deg=float(match_max_deg),
                        orthogonal_min_deg=float(orthogonal_min_deg),
                    ),
                    "unit_index": unit_index,
                    "unit_label": str(unit_row.unit_label),
                    "sf_group": sf_group,
                    "sf_group_label": str(getattr(unit_row, "sf_group_label", sf_group)),
                    "preferred_orientation_deg": pref,
                    "prior_orientation_selectivity_index": osi,
                    "passes_orientation_selectivity": bool(orientation_ok),
                    "n_relation_images": int(selected_images.size),
                    "n_matched_images": int(selected_images.size),
                    "fraction_relation_images": float(selected_images.size / image_axis.shape[0]) if image_axis.shape[0] else float("nan"),
                    "fraction_matched_images": float(selected_images.size / image_axis.shape[0]) if image_axis.shape[0] else float("nan"),
                    "mean_delta_from_contour_deg": float(np.nanmean(deltas[selected_mask])) if np.any(selected_mask) else float("nan"),
                    "median_delta_from_contour_deg": float(np.nanmedian(deltas[selected_mask])) if np.any(selected_mask) else float("nan"),
                    "relation_image_indices": " ".join(str(int(idx)) for idx in selected_images),
                    "matched_image_indices": " ".join(str(int(idx)) for idx in selected_images),
                }
            )
            if selected_images.size < int(min_matched_images_per_unit):
                continue
            image_ok = np.isin(movie_images, selected_images)
            if stabilized_baseline is not None:
                baseline_rows = baseline_rows_for_image_indices(selected_images, stabilized_baseline)
                baseline_values = np.asarray(baseline_ssi[baseline_rows, unit_index], dtype=float)
                baseline_values = baseline_values[np.isfinite(baseline_values)]
                stabilized_ssi_reference = float(np.nanmean(baseline_values)) if baseline_values.size else float("nan")
                if baseline_rate is not None:
                    rate_values = np.asarray(baseline_rate[baseline_rows, unit_index], dtype=float)
                    rate_values = rate_values[np.isfinite(rate_values)]
                    stabilized_rate_reference = float(np.nanmean(rate_values)) if rate_values.size else float("nan")
                else:
                    stabilized_rate_reference = float("nan")
            else:
                stabilized_ssi_reference = float("nan")
                stabilized_rate_reference = float("nan")
            for metric_col, bin_col, metric_label, _xlabel, metric_unit in component_specs:
                if bin_col not in metrics.columns:
                    continue
                bin_medians = (
                    metrics[[bin_col, metric_col]]
                    .dropna(subset=[bin_col, metric_col])
                    .groupby(bin_col, sort=True)[metric_col]
                    .median()
                    .sort_values()
                )
                if bin_medians.empty:
                    continue
                reference_bin = str(bin_medians.index[0])
                per_bin_ssi: dict[str, float] = {}
                for bin_label, bin_median in bin_medians.items():
                    bin_key = str(bin_label)
                    rows = row_index[image_ok & metrics[bin_col].astype(str).eq(bin_key).to_numpy()]
                    ssi_values = np.asarray(ssi[rows, unit_index], dtype=float) if rows.size else np.asarray([], dtype=float)
                    ssi_values = ssi_values[np.isfinite(ssi_values)]
                    mean_rate_values = (
                        np.asarray(mean_rate[rows, unit_index], dtype=float)
                        if mean_rate is not None and rows.size
                        else np.asarray([], dtype=float)
                    )
                    mean_rate_values = mean_rate_values[np.isfinite(mean_rate_values)]
                    ssi_mean = float(np.nanmean(ssi_values)) if ssi_values.size else float("nan")
                    rate_mean = float(np.nanmean(mean_rate_values)) if mean_rate_values.size else float("nan")
                    per_bin_ssi[bin_key] = ssi_mean
                    curve_rows.append(
                        {
                            "relation": relation,
                            "relation_label": RELATION_LABELS.get(relation, relation),
                            "relation_definition": relation_threshold_text(
                                relation,
                                match_max_deg=float(match_max_deg),
                                orthogonal_min_deg=float(orthogonal_min_deg),
                            ),
                            "unit_index": unit_index,
                            "unit_label": str(unit_row.unit_label),
                            "sf_group": sf_group,
                            "sf_group_label": str(getattr(unit_row, "sf_group_label", sf_group)),
                            "sf_group_definition": str(getattr(unit_row, "sf_group_definition", "")),
                            "sf_split_metric": float(getattr(unit_row, "sf_split_metric", float("nan"))),
                            "preferred_orientation_deg": pref,
                            "prior_orientation_selectivity_index": osi,
                            "n_relation_images": int(selected_images.size),
                            "n_matched_images": int(selected_images.size),
                            "mean_delta_from_contour_deg": selection_rows[-1]["mean_delta_from_contour_deg"],
                            "median_delta_from_contour_deg": selection_rows[-1]["median_delta_from_contour_deg"],
                            "component_metric": metric_col,
                            "component_metric_label": metric_label,
                            "component_metric_unit": metric_unit,
                            "component_bin_col": bin_col,
                            "component_bin": bin_key,
                            "component_bin_median": float(bin_median),
                            "reference_component_bin": reference_bin,
                            "n_unit_window_samples": int(ssi_values.size),
                            "unit_relation_stabilized_ssi_bits_per_spike": stabilized_ssi_reference,
                            "unit_relation_stabilized_mean_rate": stabilized_rate_reference,
                            "unit_relation_ssi_bits_per_spike": ssi_mean,
                            "unit_relation_mean_rate": rate_mean,
                            "unit_contour_matched_stabilized_ssi_bits_per_spike": stabilized_ssi_reference,
                            "unit_contour_matched_stabilized_mean_rate": stabilized_rate_reference,
                            "unit_contour_matched_ssi_bits_per_spike": ssi_mean,
                            "unit_contour_matched_mean_rate": rate_mean,
                        }
                    )
                reference = float(per_bin_ssi.get(reference_bin, float("nan")))
                for row in curve_rows:
                    if row["unit_index"] != unit_index or row["component_metric"] != metric_col or row["relation"] != relation:
                        continue
                    value = float(row["unit_relation_ssi_bits_per_spike"])
                    rate_value = float(row["unit_relation_mean_rate"])
                    row["unit_relation_ssi_at_reference"] = reference
                    row["unit_relation_ssi_delta_vs_reference"] = (
                        value - reference if math.isfinite(value) and math.isfinite(reference) else float("nan")
                    )
                    row["unit_relation_ssi_delta_vs_stabilized"] = (
                        value - stabilized_ssi_reference
                        if math.isfinite(value) and math.isfinite(stabilized_ssi_reference)
                        else float("nan")
                    )
                    row["unit_relation_mean_rate_delta_vs_stabilized"] = (
                        rate_value - stabilized_rate_reference
                        if math.isfinite(rate_value) and math.isfinite(stabilized_rate_reference)
                        else float("nan")
                    )
                    row["unit_contour_matched_ssi_at_reference"] = row["unit_relation_ssi_at_reference"]
                    row["unit_contour_matched_ssi_delta_vs_reference"] = row["unit_relation_ssi_delta_vs_reference"]
                    row["unit_contour_matched_ssi_delta_vs_stabilized"] = row["unit_relation_ssi_delta_vs_stabilized"]
                    row["unit_contour_matched_mean_rate_delta_vs_stabilized"] = row[
                        "unit_relation_mean_rate_delta_vs_stabilized"
                    ]

    selection = pd.DataFrame(selection_rows)
    curves = pd.DataFrame(curve_rows)
    if curves.empty:
        raise ValueError("No contour-matched component curves survived the support filters.")

    value_cols = [
        "unit_relation_ssi_bits_per_spike",
        "unit_relation_ssi_delta_vs_reference",
        "unit_relation_ssi_delta_vs_stabilized",
        "unit_relation_mean_rate",
        "unit_relation_mean_rate_delta_vs_stabilized",
    ]
    summary_rows: list[dict[str, Any]] = []
    for (relation, metric_col, sf_group, bin_label), sub in curves.groupby(
        ["relation", "component_metric", "sf_group", "component_bin"],
        sort=False,
    ):
        for value_col in value_cols:
            values = pd.to_numeric(sub[value_col], errors="coerce")
            finite_values = values.to_numpy(dtype=float)
            finite = np.isfinite(finite_values)
            valid_values = finite_values[finite]
            summary_rows.append(
                {
                    "relation": str(relation),
                    "relation_label": str(sub["relation_label"].iloc[0]),
                    "relation_definition": str(sub["relation_definition"].iloc[0]),
                    "component_metric": str(metric_col),
                    "component_metric_label": str(sub["component_metric_label"].iloc[0]),
                    "component_metric_unit": str(sub["component_metric_unit"].iloc[0]),
                    "component_bin_col": str(sub["component_bin_col"].iloc[0]),
                    "component_bin": str(bin_label),
                    "component_bin_median": float(sub["component_bin_median"].iloc[0]),
                    "reference_component_bin": str(sub["reference_component_bin"].iloc[0]),
                    "sf_group": str(sf_group),
                    "sf_group_label": str(sub["sf_group_label"].iloc[0]),
                    "value_name": value_col,
                    "n_units": int(sub.loc[finite, "unit_index"].nunique()),
                    "n_finite": int(finite.sum()),
                    "mean": float(np.nanmean(valid_values)) if valid_values.size else float("nan"),
                    "sem": sem(valid_values),
                    "median": float(np.nanmedian(valid_values)) if valid_values.size else float("nan"),
                    "mean_relation_images_per_unit": float(np.nanmean(sub["n_relation_images"].to_numpy(dtype=float))),
                    "mean_matched_images_per_unit": float(np.nanmean(sub["n_matched_images"].to_numpy(dtype=float))),
                    "mean_unit_window_samples": float(np.nanmean(sub["n_unit_window_samples"].to_numpy(dtype=float))),
                }
            )
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["relation", "component_metric", "value_name", "component_bin_median", "sf_group"])
        .reset_index(drop=True)
    )
    return selection, curves, summary


def plot_component_curve(
    summary: pd.DataFrame,
    unit: pd.DataFrame,
    fig_dir: Path,
    *,
    metric_col: str,
    relation: str,
    xlabel: str,
    sf_groups: list[str],
    match_max_deg: float,
    orthogonal_min_deg: float,
    min_osi: float,
    min_matched_images_per_unit: int,
    image_axis_col: str,
    component_reference: pd.DataFrame,
) -> dict[str, str]:
    use = summary[
        summary["component_metric"].astype(str).eq(metric_col)
        & summary["relation"].astype(str).eq(str(relation))
    ].copy()
    if use.empty:
        raise ValueError(f"No summary rows for component metric {metric_col!r} and relation {relation!r}.")
    relation_label = RELATION_LABELS.get(relation, relation).replace("_", " ")
    panels = [
        ("unit_relation_ssi_bits_per_spike", "SSI (bits/spike)", f"Absolute SSI on {relation_label}"),
        ("unit_relation_ssi_delta_vs_stabilized", "SSI minus stabilized baseline (bits/spike)", "Movement modulation"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharex=True)
    for ax, (value_name, ylabel, title) in zip(axes, panels, strict=True):
        add_component_reference_strips(ax, component_reference, metric_col)
        for sf_group in sf_groups:
            sub = use[use["value_name"].astype(str).eq(value_name) & use["sf_group"].astype(str).eq(str(sf_group))]
            sub = sub.sort_values("component_bin_median")
            if sub.empty:
                continue
            x = sub["component_bin_median"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            e = sub["sem"].to_numpy(dtype=float)
            color = color_for_label(sf_group)
            label = f"{str(sf_group).replace('_', ' ')} (n={int(sub['n_units'].iloc[0])})"
            ax.plot(x, y, marker="o", linewidth=2.3, markersize=4.8, color=color, label=label, zorder=5)
            ax.fill_between(x, y - e, y + e, color=color, alpha=0.16, linewidth=0, zorder=2)
        if "delta" in value_name:
            ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.9", linewidth=0.8)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    definition_text = contour_matched_definition_text(unit, sf_groups)
    fig.suptitle(
        f"{relation_label.capitalize()} unit-window pairs conditioned on motion component\n"
        f"{definition_text}; axis={image_axis_col}; "
        f"{relation_threshold_text(relation, match_max_deg=match_max_deg, orthogonal_min_deg=orthogonal_min_deg)}; "
        f"OSI >= {min_osi:g}; min images/unit = {min_matched_images_per_unit}",
        fontsize=11.3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    safe = metric_col.replace("_arcmin", "").replace("_", "-")
    suffix = "matched" if relation == "matched" else relation
    return save_figure(fig, fig_dir, f"phase2_contour_{suffix}_{safe}_low_high_scale_curves")


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"png": out_dir / f"{stem}.png", "pdf": out_dir / f"{stem}.pdf"}
    fig.savefig(paths["png"], dpi=220, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)
    return {key: str(path) for key, path in paths.items()}


def plot_across_vs_along_overlay(
    summary: pd.DataFrame,
    unit: pd.DataFrame,
    fig_dir: Path,
    *,
    relation: str,
    sf_groups: list[str],
    match_max_deg: float,
    orthogonal_min_deg: float,
    min_osi: float,
    min_matched_images_per_unit: int,
    image_axis_col: str,
) -> dict[str, str]:
    use = summary[summary["relation"].astype(str).eq(str(relation))].copy()
    if use.empty:
        raise ValueError(f"No summary rows for relation {relation!r}.")
    panels = [
        ("unit_relation_ssi_bits_per_spike", "SSI (bits/spike)", "Absolute SSI"),
        ("unit_relation_ssi_delta_vs_stabilized", "SSI minus stabilized baseline (bits/spike)", "Movement modulation"),
    ]
    metric_specs = [
        ("across_path_arcmin", "across", "-", "o"),
        ("along_path_arcmin", "along", "--", "s"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, (value_name, ylabel, title) in zip(axes, panels, strict=True):
        for metric_col, metric_label, linestyle, marker in metric_specs:
            for sf_group in sf_groups:
                sub = summary[
                    summary["component_metric"].astype(str).eq(metric_col)
                    & summary["value_name"].astype(str).eq(value_name)
                    & summary["sf_group"].astype(str).eq(str(sf_group))
                    & summary["relation"].astype(str).eq(str(relation))
                ].sort_values("component_bin_median")
                if sub.empty:
                    continue
                x = sub["component_bin_median"].to_numpy(dtype=float)
                y = sub["mean"].to_numpy(dtype=float)
                e = sub["sem"].to_numpy(dtype=float)
                color = color_for_label(sf_group)
                ax.plot(
                    x,
                    y,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2.3,
                    markersize=4.8,
                    color=color,
                    zorder=5,
                )
                ax.fill_between(
                    x,
                    y - e,
                    y + e,
                    color=color,
                    alpha=0.11 if metric_col == "across_path_arcmin" else 0.07,
                    linewidth=0,
                    zorder=2,
                )
        if "delta" in value_name:
            ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("component path bin median (arcmin)")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    color_handles = [
        Line2D([0], [0], color=color_for_label(sf_group), lw=2.5, label=str(sf_group).replace("_", " "))
        for sf_group in sf_groups
    ]
    component_handles = [
        Line2D([0], [0], color="0.25", lw=2.2, linestyle="-", marker="o", markersize=4.5, label="across contour"),
        Line2D([0], [0], color="0.25", lw=2.2, linestyle="--", marker="s", markersize=4.5, label="along contour"),
    ]
    fig.legend(handles=color_handles + component_handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.02))
    definition_text = contour_matched_definition_text(unit, sf_groups)
    relation_label = RELATION_LABELS.get(relation, relation).replace("_", " ")
    fig.suptitle(
        f"{relation_label.capitalize()} unit-window pairs: across vs along motion components\n"
        f"{definition_text}; axis={image_axis_col}; "
        f"{relation_threshold_text(relation, match_max_deg=match_max_deg, orthogonal_min_deg=orthogonal_min_deg)}; "
        f"OSI >= {min_osi:g}; min images/unit = {min_matched_images_per_unit}",
        fontsize=11.3,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.84))
    suffix = "matched" if relation == "matched" else relation
    return save_figure(fig, fig_dir, f"phase2_contour_{suffix}_across_vs_along_path_low_high_overlay")


def plot_relation_across_vs_along_grid(
    summary: pd.DataFrame,
    unit: pd.DataFrame,
    fig_dir: Path,
    *,
    relation_modes: list[str],
    sf_groups: list[str],
    match_max_deg: float,
    orthogonal_min_deg: float,
    min_osi: float,
    min_matched_images_per_unit: int,
    image_axis_col: str,
) -> dict[str, str]:
    available = set(summary["relation"].astype(str).unique()) if "relation" in summary.columns else set()
    relations = [relation for relation in relation_modes if relation in available]
    if not relations:
        raise ValueError("No relation-conditioned summary rows available for comparison plot.")
    panels = [
        ("unit_relation_ssi_bits_per_spike", "SSI (bits/spike)", "Absolute SSI"),
        ("unit_relation_ssi_delta_vs_stabilized", "SSI minus stabilized baseline (bits/spike)", "Movement modulation"),
    ]
    metric_specs = [
        ("across_path_arcmin", "across contour", "-", "o"),
        ("along_path_arcmin", "along contour", "--", "s"),
    ]
    fig, axes = plt.subplots(len(relations), 2, figsize=(11.6, 2.85 * len(relations) + 1.15), sharex="col", sharey="col")
    if len(relations) == 1:
        axes = np.asarray([axes])
    for row_i, relation in enumerate(relations):
        relation_label = RELATION_LABELS.get(relation, relation).replace("_", " ")
        for col_i, (value_name, ylabel, title) in enumerate(panels):
            ax = axes[row_i, col_i]
            for metric_col, _metric_label, linestyle, marker in metric_specs:
                for sf_group in sf_groups:
                    sub = summary[
                        summary["relation"].astype(str).eq(relation)
                        & summary["component_metric"].astype(str).eq(metric_col)
                        & summary["value_name"].astype(str).eq(value_name)
                        & summary["sf_group"].astype(str).eq(str(sf_group))
                    ].sort_values("component_bin_median")
                    if sub.empty:
                        continue
                    x = sub["component_bin_median"].to_numpy(dtype=float)
                    y = sub["mean"].to_numpy(dtype=float)
                    e = sub["sem"].to_numpy(dtype=float)
                    color = color_for_label(sf_group)
                    ax.plot(
                        x,
                        y,
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=2.0,
                        markersize=4.2,
                        color=color,
                        zorder=5,
                    )
                    ax.fill_between(
                        x,
                        y - e,
                        y + e,
                        color=color,
                        alpha=0.10 if metric_col == "across_path_arcmin" else 0.06,
                        linewidth=0,
                        zorder=2,
                    )
            if "delta" in value_name:
                ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
            if row_i == 0:
                ax.set_title(title)
            if row_i == len(relations) - 1:
                ax.set_xlabel("component path bin median (arcmin)")
            if col_i == 0:
                ax.set_ylabel("SSI (bits/spike)")
                ax.text(
                    0.015,
                    0.94,
                    relation_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                    fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
                )
            else:
                ax.set_ylabel("SSI - stabilized\n(bits/spike)")
            ax.grid(True, color="0.9", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
    color_handles = [
        Line2D([0], [0], color=color_for_label(sf_group), lw=2.5, label=str(sf_group).replace("_", " "))
        for sf_group in sf_groups
    ]
    component_handles = [
        Line2D([0], [0], color="0.25", lw=2.1, linestyle="-", marker="o", markersize=4.2, label="across contour"),
        Line2D([0], [0], color="0.25", lw=2.1, linestyle="--", marker="s", markersize=4.2, label="along contour"),
    ]
    definition_text = contour_matched_definition_text(unit, sf_groups)
    fig.legend(handles=color_handles + component_handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle(
        "Across vs along motion components across unit-contour relations\n"
        f"{definition_text}; axis={image_axis_col}; matched <= {match_max_deg:g} deg; "
        f"orthogonal >= {orthogonal_min_deg:g} deg; OSI >= {min_osi:g}; min images/unit = {min_matched_images_per_unit}",
        fontsize=11.0,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.925))
    return save_figure(fig, fig_dir, "phase2_contour_relation_across_vs_along_path_low_high_grid")


def build_unit_slope_robustness(
    curves: pd.DataFrame,
    *,
    relation_modes: list[str],
    sf_groups: list[str],
    component_metrics: list[str],
    value_col: str = "unit_relation_ssi_delta_vs_stabilized",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    for relation in relation_modes:
        for metric in component_metrics:
            metric_rows = curves[
                curves["relation"].astype(str).eq(relation)
                & curves["component_metric"].astype(str).eq(metric)
            ].copy()
            if metric_rows.empty:
                continue
            bin_order = (
                metric_rows[["component_bin", "component_bin_median"]]
                .drop_duplicates()
                .sort_values("component_bin_median")
            )
            if bin_order.empty:
                continue
            first_bin = str(bin_order.iloc[0]["component_bin"])
            last_bin = str(bin_order.iloc[-1]["component_bin"])
            first_median = float(bin_order.iloc[0]["component_bin_median"])
            last_median = float(bin_order.iloc[-1]["component_bin_median"])
            for sf_group in sf_groups:
                sub = metric_rows[metric_rows["sf_group"].astype(str).eq(str(sf_group))].copy()
                if sub.empty:
                    continue
                wide = sub.pivot_table(index=["unit_index", "unit_label"], columns="component_bin", values=value_col, aggfunc="mean")
                if first_bin not in wide.columns or last_bin not in wide.columns:
                    continue
                slopes = (wide[last_bin] - wide[first_bin]).dropna().astype(float)
                if slopes.empty:
                    continue
                slope_vals = slopes.to_numpy(dtype=float)
                mean_slope = float(np.mean(slope_vals))
                median_slope = float(np.median(slope_vals))
                q25, q75 = np.quantile(slope_vals, [0.25, 0.75])
                pos = int(np.sum(slope_vals > 0))
                neg = int(np.sum(slope_vals < 0))
                loo = []
                for idx in range(slope_vals.size):
                    remain = np.delete(slope_vals, idx)
                    loo.append(float(np.mean(remain)) if remain.size else float("nan"))
                loo_arr = np.asarray(loo, dtype=float)
                influence = mean_slope - loo_arr
                finite_loo = loo_arr[np.isfinite(loo_arr)]
                finite_influence = influence[np.isfinite(influence)]
                if finite_influence.size:
                    max_idx = int(np.nanargmax(np.abs(influence)))
                    most_influential_unit_index = int(slopes.index[max_idx][0])
                    most_influential_unit_label = str(slopes.index[max_idx][1])
                    max_shift = float(np.nanmax(np.abs(finite_influence)))
                else:
                    most_influential_unit_index = -1
                    most_influential_unit_label = ""
                    max_shift = float("nan")
                if finite_loo.size and abs(mean_slope) > 0:
                    loo_sign_stable = bool(np.all(np.sign(finite_loo) == np.sign(mean_slope)))
                else:
                    loo_sign_stable = False
                abs_vals = np.abs(slope_vals)
                total_abs = float(np.sum(abs_vals))
                sorted_abs = np.sort(abs_vals)[::-1]
                top1_share = float(sorted_abs[0] / total_abs) if total_abs > 0 else float("nan")
                top3_share = float(np.sum(sorted_abs[:3]) / total_abs) if total_abs > 0 else float("nan")
                abs_wide = sub.pivot_table(
                    index=["unit_index", "unit_label"],
                    columns="component_bin",
                    values="unit_relation_ssi_bits_per_spike",
                    aggfunc="mean",
                )
                mean_abs_ssi = abs_wide.mean(axis=1).reindex(slopes.index)
                corr = (
                    float(np.corrcoef(mean_abs_ssi.to_numpy(dtype=float), slope_vals)[0, 1])
                    if slopes.size > 2 and np.isfinite(mean_abs_ssi.to_numpy(dtype=float)).all()
                    else float("nan")
                )
                summary_rows.append(
                    {
                        "relation": relation,
                        "relation_label": RELATION_LABELS.get(relation, relation),
                        "component_metric": metric,
                        "sf_group": str(sf_group),
                        "n_units": int(slopes.size),
                        "first_bin": first_bin,
                        "last_bin": last_bin,
                        "first_bin_median": first_median,
                        "last_bin_median": last_median,
                        "mean_last_minus_first": mean_slope,
                        "median_last_minus_first": median_slope,
                        "q25_last_minus_first": float(q25),
                        "q75_last_minus_first": float(q75),
                        "n_positive_units": pos,
                        "n_negative_units": neg,
                        "fraction_positive_units": float(pos / slopes.size),
                        "leave_one_out_min_mean": float(np.nanmin(finite_loo)) if finite_loo.size else float("nan"),
                        "leave_one_out_max_mean": float(np.nanmax(finite_loo)) if finite_loo.size else float("nan"),
                        "leave_one_out_sign_stable": loo_sign_stable,
                        "max_leave_one_out_shift": max_shift,
                        "most_influential_unit_index": most_influential_unit_index,
                        "most_influential_unit_label": most_influential_unit_label,
                        "top1_abs_slope_share": top1_share,
                        "top3_abs_slope_share": top3_share,
                        "corr_unit_abs_ssi_with_slope": corr,
                    }
                )
                for (unit_index, unit_label), slope in slopes.items():
                    unit_rows.append(
                        {
                            "relation": relation,
                            "relation_label": RELATION_LABELS.get(relation, relation),
                            "component_metric": metric,
                            "sf_group": str(sf_group),
                            "unit_index": int(unit_index),
                            "unit_label": str(unit_label),
                            "last_minus_first_delta": float(slope),
                            "mean_absolute_ssi": (
                                float(mean_abs_ssi.loc[(unit_index, unit_label)])
                                if (unit_index, unit_label) in mean_abs_ssi.index
                                else float("nan")
                            ),
                        }
                    )
    robustness = pd.DataFrame(summary_rows)
    unit_slopes = pd.DataFrame(unit_rows)
    if not robustness.empty:
        robustness = robustness.sort_values(["component_metric", "relation", "sf_group"]).reset_index(drop=True)
    if not unit_slopes.empty:
        unit_slopes = unit_slopes.sort_values(["component_metric", "relation", "sf_group", "last_minus_first_delta"]).reset_index(drop=True)
    return robustness, unit_slopes


def plot_unit_slope_distributions(
    unit_slopes: pd.DataFrame,
    out_dir: Path,
    *,
    relation_modes: list[str],
    sf_groups: list[str],
) -> dict[str, str]:
    metrics = ["across_path_arcmin", "along_path_arcmin"]
    metric_titles = {
        "across_path_arcmin": "Across-contour path",
        "along_path_arcmin": "Along-contour path",
    }
    if unit_slopes.empty:
        raise ValueError("No unit slopes available for robustness plot.")
    fig, axes = plt.subplots(len(metrics), len(relation_modes), figsize=(4.0 * len(relation_modes), 6.0), sharey=True)
    if len(metrics) == 1:
        axes = np.asarray([axes])
    if len(relation_modes) == 1:
        axes = np.asarray([[ax] for ax in axes.ravel()])
    colors = {"low_sf": LOW_BLUE, "high_sf": HIGH_RED}
    for row_i, metric in enumerate(metrics):
        for col_i, relation in enumerate(relation_modes):
            ax = axes[row_i, col_i]
            for x_pos, sf_group in enumerate(sf_groups):
                vals = unit_slopes[
                    unit_slopes["component_metric"].astype(str).eq(metric)
                    & unit_slopes["relation"].astype(str).eq(relation)
                    & unit_slopes["sf_group"].astype(str).eq(str(sf_group))
                ]["last_minus_first_delta"].dropna().to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                jitter = np.linspace(-0.055, 0.055, vals.size) if vals.size > 1 else np.asarray([0.0])
                ax.scatter(
                    np.full(vals.size, x_pos, dtype=float) + jitter,
                    vals,
                    s=16,
                    color=colors.get(str(sf_group), "0.4"),
                    alpha=0.45,
                    linewidth=0,
                )
                median = float(np.median(vals))
                q25, q75 = np.quantile(vals, [0.25, 0.75])
                mean = float(np.mean(vals))
                color = colors.get(str(sf_group), "0.4")
                ax.plot([x_pos - 0.24, x_pos + 0.24], [median, median], color=color, linewidth=2.2)
                ax.plot([x_pos, x_pos], [q25, q75], color=color, linewidth=4.0, alpha=0.45)
                ax.plot([x_pos - 0.18, x_pos + 0.18], [mean, mean], color="black", linewidth=1.2, alpha=0.75)
            ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
            ax.set_xticks(np.arange(len(sf_groups)), [str(group).replace("_", " ").replace("sf", "SF") for group in sf_groups])
            if row_i == 0:
                ax.set_title(RELATION_LABELS.get(relation, relation).replace("_", " "))
            if col_i == 0:
                ax.set_ylabel(f"{metric_titles.get(metric, metric)}\nunit slope (last - first)")
            ax.grid(True, color="0.9", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Per-unit movement-modulation slope robustness\npoints=units, thick bar=IQR, colored line=median, black line=mean",
        fontsize=11.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return save_figure(fig, out_dir, "relation_component_unit_slope_distributions")


def build_across_fraction_control(
    metrics: pd.DataFrame,
    ssi: np.ndarray,
    unit: pd.DataFrame,
    *,
    image_axis_col: str,
    sf_groups: list[str],
    match_max_deg: float,
    min_osi: float,
    stabilized_baseline: dict[str, Any] | None,
    n_total_path_bins: int,
    n_fraction_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = metrics.copy()
    total_bin_col = "control_total_path_length_bin"
    work, _total_bin_rows = add_quantile_bin(
        work,
        "rendered_path_length_arcmin",
        total_bin_col,
        n_bins=int(n_total_path_bins),
    )
    work["across_fraction_within_total_bin"] = pd.NA
    labels = ["low_across_fraction", "middle_across_fraction", "high_across_fraction"]
    bin_rows: list[dict[str, Any]] = []
    for total_bin, sub in work[work[total_bin_col].notna()].groupby(total_bin_col, sort=True):
        vals = pd.to_numeric(sub["across_path_fraction"], errors="coerce")
        finite = vals[np.isfinite(vals)]
        if finite.empty:
            continue
        q = min(int(n_fraction_bins), int(finite.nunique(dropna=True)))
        if q <= 1:
            assigned = pd.Series(["low_across_fraction"] * finite.shape[0], index=finite.index)
        else:
            codes = pd.qcut(finite, q=q, labels=False, duplicates="drop")
            assigned = pd.Series([labels[min(int(code), len(labels) - 1)] if pd.notna(code) else pd.NA for code in codes], index=finite.index)
        work.loc[assigned.index, "across_fraction_within_total_bin"] = assigned
        for label, vals_sub in vals.loc[assigned.index].groupby(assigned, sort=True):
            bin_rows.append(
                {
                    "trace_path_length_bin": str(total_bin),
                    "across_fraction_bin": str(label),
                    "n_movie_rows": int(vals_sub.dropna().shape[0]),
                    "rendered_path_length_median_arcmin": float(pd.to_numeric(sub.loc[vals_sub.index, "rendered_path_length_arcmin"], errors="coerce").median()),
                    "median_across_fraction": float(vals_sub.median()),
                }
            )

    image_axis = work[["image_index", image_axis_col]].drop_duplicates("image_index").sort_values("image_index").reset_index(drop=True)
    movie_images = work["image_index"].astype(int).to_numpy()
    row_index = np.arange(work.shape[0], dtype=int)
    total_bin_values = work[total_bin_col].astype(str).to_numpy()
    fraction_bin_values = work["across_fraction_within_total_bin"].astype(str).to_numpy()
    baseline_ssi = stabilized_baseline["ssi"] if stabilized_baseline is not None else None
    curve_rows: list[dict[str, Any]] = []
    use_units = unit[unit["sf_group"].astype(str).isin(sf_groups)].copy()
    for unit_row in use_units.itertuples(index=False):
        unit_index = int(unit_row.unit_index)
        pref = float(unit_row.analysis_preferred_orientation_deg)
        osi = float(unit_row.prior_orientation_selectivity_index)
        if not (math.isfinite(pref) and math.isfinite(osi) and osi >= float(min_osi)):
            continue
        deltas = axis_delta_deg(image_axis[image_axis_col].to_numpy(dtype=float), pref)
        matched_images = image_axis.loc[np.isfinite(deltas) & (deltas <= float(match_max_deg)), "image_index"].astype(int).to_numpy()
        if matched_images.size == 0:
            continue
        image_ok = np.isin(movie_images, matched_images)
        if stabilized_baseline is not None:
            baseline_rows = baseline_rows_for_image_indices(matched_images, stabilized_baseline)
            baseline_values = np.asarray(baseline_ssi[baseline_rows, unit_index], dtype=float)
            stabilized_reference = float(np.nanmean(baseline_values[np.isfinite(baseline_values)]))
        else:
            stabilized_reference = float("nan")
        grouped = work[work[total_bin_col].notna() & work["across_fraction_within_total_bin"].notna()]
        for (total_bin, fraction_bin), sub in grouped.groupby([total_bin_col, "across_fraction_within_total_bin"], sort=True):
            rows = row_index[image_ok & (total_bin_values == str(total_bin)) & (fraction_bin_values == str(fraction_bin))]
            vals = np.asarray(ssi[rows, unit_index], dtype=float) if rows.size else np.asarray([], dtype=float)
            vals = vals[np.isfinite(vals)]
            value = float(np.nanmean(vals)) if vals.size else float("nan")
            total_path_median = float(pd.to_numeric(sub["rendered_path_length_arcmin"], errors="coerce").median())
            fraction_median = float(pd.to_numeric(sub["across_path_fraction"], errors="coerce").median())
            curve_rows.append(
                {
                    "unit_index": unit_index,
                    "unit_label": str(unit_row.unit_label),
                    "sf_group": str(unit_row.sf_group),
                    "trace_path_length_bin": str(total_bin),
                    "trace_path_length_bin_median_arcmin": total_path_median,
                    "across_fraction_bin": str(fraction_bin),
                    "across_fraction_bin_median": fraction_median,
                    "n_matched_images": int(matched_images.size),
                    "n_unit_window_samples": int(vals.size),
                    "unit_contour_matched_stabilized_ssi_bits_per_spike": stabilized_reference,
                    "unit_contour_matched_ssi_bits_per_spike": value,
                    "unit_contour_matched_ssi_delta_vs_stabilized": (
                        value - stabilized_reference if math.isfinite(value) and math.isfinite(stabilized_reference) else float("nan")
                    ),
                }
            )
    curves = pd.DataFrame(curve_rows)
    summary_rows: list[dict[str, Any]] = []
    if not curves.empty:
        for (sf_group, total_bin, fraction_bin), sub in curves.groupby(["sf_group", "trace_path_length_bin", "across_fraction_bin"], sort=False):
            vals = pd.to_numeric(sub["unit_contour_matched_ssi_delta_vs_stabilized"], errors="coerce")
            finite = np.isfinite(vals.to_numpy(dtype=float))
            summary_rows.append(
                {
                    "sf_group": str(sf_group),
                    "trace_path_length_bin": str(total_bin),
                    "trace_path_length_bin_median_arcmin": float(sub["trace_path_length_bin_median_arcmin"].iloc[0]),
                    "across_fraction_bin": str(fraction_bin),
                    "across_fraction_bin_median": float(sub["across_fraction_bin_median"].iloc[0]),
                    "n_units": int(sub.loc[finite, "unit_index"].nunique()),
                    "mean_delta_vs_stabilized": float(np.nanmean(vals)),
                    "sem_delta_vs_stabilized": sem(vals),
                    "median_delta_vs_stabilized": float(np.nanmedian(vals)),
                }
            )
    return curves, pd.DataFrame(summary_rows)


def plot_across_fraction_control(summary: pd.DataFrame, fig_dir: Path) -> dict[str, str]:
    if summary.empty:
        raise ValueError("No rows available for across-fraction control plot.")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True, sharey=True)
    sf_groups = ["low_sf", "high_sf"]
    fraction_order = ["low_across_fraction", "middle_across_fraction", "high_across_fraction"]
    colors = {"low_across_fraction": "#4c78a8", "middle_across_fraction": "#8f8f8f", "high_across_fraction": "#d62728"}
    for ax, sf_group in zip(axes, sf_groups, strict=True):
        for fraction_bin in fraction_order:
            sub = summary[summary["sf_group"].astype(str).eq(sf_group) & summary["across_fraction_bin"].astype(str).eq(fraction_bin)]
            sub = sub.sort_values("trace_path_length_bin_median_arcmin")
            if sub.empty:
                continue
            x = sub["trace_path_length_bin_median_arcmin"].to_numpy(dtype=float)
            y = sub["mean_delta_vs_stabilized"].to_numpy(dtype=float)
            e = sub["sem_delta_vs_stabilized"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=2.0, color=colors[fraction_bin], label=fraction_bin.replace("_", " "))
            ax.fill_between(x, y - e, y + e, color=colors[fraction_bin], alpha=0.12, linewidth=0)
        ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
        ax.set_title(sf_group.replace("_", " "))
        ax.set_xlabel("total path length bin median (arcmin)")
        ax.grid(True, color="0.9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("SSI minus stabilized baseline (bits/spike)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Across-contour fraction within total path-length bins", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return save_figure(fig, fig_dir, "phase2_contour_matched_across_fraction_within_total_path_control")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output directory already exists and is non-empty: {out_dir}. Use --force to overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    matrix_dir = Path(args.matrix_dir)
    progress(f"Loading matrix dataset from {matrix_dir}")
    ssi, _expected, mean_rate, movie, trace, unit_raw, trace_xy = load_inputs(matrix_dir)
    image = pd.read_csv(matrix_dir / "image_feature_table.csv")
    stabilized_baseline = load_stabilized_baseline(matrix_dir, image, unit_raw)
    progress(f"Loaded SSI matrix {ssi.shape}, movies={movie.shape[0]}, traces={trace.shape[0]}, units={unit_raw.shape[0]}")

    progress("Enriching unit table and computing contour-relative motion components")
    unit, external_sources, unit_bin_rows = enrich_unit_table(unit_raw, args)
    metrics = compute_component_movie_metrics(movie, trace, trace_xy, image_axis_col=str(args.image_axis_col))
    bin_rows: list[dict[str, Any]] = []
    for metric_col, bin_col, _label, _xlabel, _unit in COMPONENT_SPECS:
        metrics, rows = add_quantile_bin(metrics, metric_col, bin_col, n_bins=int(args.n_component_bins))
        bin_rows.extend(rows)

    metrics_csv = out_dir / "phase2_contour_relative_trace_component_movie_metrics.csv"
    bins_csv = out_dir / "phase2_contour_relative_trace_component_bin_definitions.csv"
    ref_csv = out_dir / "phase2_contour_relative_trace_component_reference_windows.csv"
    metrics.to_csv(metrics_csv, index=False)
    pd.DataFrame(bin_rows).to_csv(bins_csv, index=False)
    component_reference = component_reference_rows(metrics, [spec[0] for spec in COMPONENT_SPECS])
    component_reference.to_csv(ref_csv, index=False)

    progress("Building unit-first relation-conditioned SSI summaries by component")
    sf_groups = parse_csv_list(str(args.sf_groups))
    relation_modes = ordered_relation_modes(parse_csv_list(str(args.relation_modes)))
    orthogonal_min_deg = float(args.orthogonal_min_deg) if args.orthogonal_min_deg is not None else 90.0 - float(args.match_max_deg)
    selection, curves, summary = build_component_unit_curves(
        metrics,
        ssi,
        mean_rate,
        unit,
        image_axis_col=str(args.image_axis_col),
        sf_groups=sf_groups,
        match_max_deg=float(args.match_max_deg),
        orthogonal_min_deg=orthogonal_min_deg,
        min_osi=float(args.min_osi),
        min_matched_images_per_unit=int(args.min_matched_images_per_unit),
        relation_modes=relation_modes,
        component_specs=COMPONENT_SPECS,
        stabilized_baseline=stabilized_baseline,
    )
    relation_selection_csv = out_dir / "phase2_contour_relation_trace_component_unit_selection.csv"
    relation_curves_csv = out_dir / "phase2_contour_relation_trace_component_unit_curves.csv"
    relation_summary_csv = out_dir / "phase2_contour_relation_trace_component_summary.csv"
    selection.to_csv(relation_selection_csv, index=False)
    curves.to_csv(relation_curves_csv, index=False)
    summary.to_csv(relation_summary_csv, index=False)
    selection_csv = out_dir / "phase2_contour_matched_trace_component_unit_selection.csv"
    curves_csv = out_dir / "phase2_contour_matched_trace_component_unit_curves.csv"
    summary_csv = out_dir / "phase2_contour_matched_trace_component_summary.csv"
    selection[selection["relation"].astype(str).eq("matched")].to_csv(selection_csv, index=False)
    curves[curves["relation"].astype(str).eq("matched")].to_csv(curves_csv, index=False)
    summary[summary["relation"].astype(str).eq("matched")].to_csv(summary_csv, index=False)

    progress("Building within-total-path across-fraction control")
    control_curves, control_summary = build_across_fraction_control(
        metrics,
        ssi,
        unit,
        image_axis_col=str(args.image_axis_col),
        sf_groups=sf_groups,
        match_max_deg=float(args.match_max_deg),
        min_osi=float(args.min_osi),
        stabilized_baseline=stabilized_baseline,
        n_total_path_bins=int(args.n_total_path_bins),
        n_fraction_bins=int(args.n_across_fraction_control_bins),
    )
    control_curves_csv = out_dir / "phase2_contour_matched_across_fraction_within_total_path_unit_curves.csv"
    control_summary_csv = out_dir / "phase2_contour_matched_across_fraction_within_total_path_summary.csv"
    control_curves.to_csv(control_curves_csv, index=False)
    control_summary.to_csv(control_summary_csv, index=False)

    progress("Writing figures")
    figures: dict[str, dict[str, str]] = {}
    for relation in relation_modes:
        for metric_col, _bin_col, _label, xlabel, _unit in COMPONENT_SPECS:
            figures[f"{relation}_{metric_col}_curve"] = plot_component_curve(
                summary,
                unit,
                fig_dir,
                metric_col=metric_col,
                relation=relation,
                xlabel=xlabel,
                sf_groups=sf_groups,
                match_max_deg=float(args.match_max_deg),
                orthogonal_min_deg=orthogonal_min_deg,
                min_osi=float(args.min_osi),
                min_matched_images_per_unit=int(args.min_matched_images_per_unit),
                image_axis_col=str(args.image_axis_col),
                component_reference=component_reference,
            )
        figures[f"{relation}_across_vs_along_overlay"] = plot_across_vs_along_overlay(
            summary,
            unit,
            fig_dir,
            relation=relation,
            sf_groups=sf_groups,
            match_max_deg=float(args.match_max_deg),
            orthogonal_min_deg=orthogonal_min_deg,
            min_osi=float(args.min_osi),
            min_matched_images_per_unit=int(args.min_matched_images_per_unit),
            image_axis_col=str(args.image_axis_col),
        )
    figures["relation_across_vs_along_grid"] = plot_relation_across_vs_along_grid(
        summary,
        unit,
        fig_dir,
        relation_modes=relation_modes,
        sf_groups=sf_groups,
        match_max_deg=float(args.match_max_deg),
        orthogonal_min_deg=orthogonal_min_deg,
        min_osi=float(args.min_osi),
        min_matched_images_per_unit=int(args.min_matched_images_per_unit),
        image_axis_col=str(args.image_axis_col),
    )
    if not control_summary.empty:
        figures["across_fraction_within_total_path_control"] = plot_across_fraction_control(control_summary, fig_dir)

    key_rows = []
    for metric_col in ["across_path_arcmin", "along_path_arcmin", "across_rms_arcmin", "across_path_fraction"]:
        sub = summary[
            summary["component_metric"].astype(str).eq(metric_col)
            & summary["value_name"].astype(str).eq("unit_relation_ssi_delta_vs_stabilized")
        ]
        for (relation, sf_group), sf_sub in sub.groupby(["relation", "sf_group"], sort=False):
            ordered = sf_sub.sort_values("component_bin_median")
            if ordered.empty:
                continue
            key_rows.append(
                {
                    "relation": str(relation),
                    "relation_label": str(ordered["relation_label"].iloc[0]),
                    "component_metric": metric_col,
                    "sf_group": str(sf_group),
                    "n_units": int(ordered["n_units"].iloc[0]),
                    "first_bin_median": float(ordered["component_bin_median"].iloc[0]),
                    "first_bin_delta_mean": float(ordered["mean"].iloc[0]),
                    "last_bin_median": float(ordered["component_bin_median"].iloc[-1]),
                    "last_bin_delta_mean": float(ordered["mean"].iloc[-1]),
                    "last_minus_first_delta_mean": float(ordered["mean"].iloc[-1] - ordered["mean"].iloc[0]),
                }
            )
    key_effects = pd.DataFrame(key_rows)
    key_effects_csv = out_dir / "phase2_contour_relation_trace_component_key_effects.csv"
    key_effects.to_csv(key_effects_csv, index=False)
    matched_key_effects_csv = out_dir / "phase2_contour_matched_trace_component_key_effects.csv"
    key_effects[key_effects["relation"].astype(str).eq("matched")].to_csv(matched_key_effects_csv, index=False)

    progress("Writing unit-level robustness diagnostics")
    robustness_dir = out_dir / "robustness_diagnostics_v1"
    robustness_dir.mkdir(parents=True, exist_ok=True)
    robustness_summary, unit_slopes = build_unit_slope_robustness(
        curves,
        relation_modes=relation_modes,
        sf_groups=sf_groups,
        component_metrics=[spec[0] for spec in COMPONENT_SPECS],
    )
    robustness_summary_csv = robustness_dir / "relation_component_unit_slope_robustness_summary.csv"
    unit_slopes_csv = robustness_dir / "relation_component_unit_slopes.csv"
    robustness_summary.to_csv(robustness_summary_csv, index=False)
    unit_slopes.to_csv(unit_slopes_csv, index=False)
    figures["unit_slope_robustness"] = plot_unit_slope_distributions(
        unit_slopes,
        robustness_dir,
        relation_modes=relation_modes,
        sf_groups=sf_groups,
    )

    summary_payload = {
        "matrix_dir": matrix_dir,
        "out_dir": out_dir,
        "fig_dir": fig_dir,
        "robustness_dir": robustness_dir,
        "image_axis_col": str(args.image_axis_col),
        "sf_groups": sf_groups,
        "relation_modes": relation_modes,
        "match_max_deg": float(args.match_max_deg),
        "orthogonal_min_deg": orthogonal_min_deg,
        "min_osi": float(args.min_osi),
        "min_matched_images_per_unit": int(args.min_matched_images_per_unit),
        "n_movies": int(movie.shape[0]),
        "n_traces": int(trace.shape[0]),
        "n_units": int(unit.shape[0]),
        "n_component_summary_rows": int(summary.shape[0]),
        "component_metrics_csv": metrics_csv,
        "component_bin_definitions_csv": bins_csv,
        "component_reference_windows_csv": ref_csv,
        "relation_selection_csv": relation_selection_csv,
        "relation_curves_csv": relation_curves_csv,
        "relation_summary_csv": relation_summary_csv,
        "matched_selection_csv": selection_csv,
        "matched_curves_csv": curves_csv,
        "matched_summary_csv": summary_csv,
        "control_curves_csv": control_curves_csv,
        "control_summary_csv": control_summary_csv,
        "key_effects_csv": key_effects_csv,
        "matched_key_effects_csv": matched_key_effects_csv,
        "robustness_summary_csv": robustness_summary_csv,
        "unit_slopes_csv": unit_slopes_csv,
        "key_effects": key_rows,
        "figures": figures,
        "external_unit_label_sources": external_sources,
        "unit_bin_rows": unit_bin_rows,
        "notes": [
            "Components are computed in image coordinates using image_edge_axis_deg: along is parallel to the contour axis and across is perpendicular.",
            "Path component metrics sum absolute projected frame-to-frame displacement, in arcmin.",
            "Relation curves are unit-first: each unit is averaged over its selected image windows within each component bin before SF-group means are computed.",
            "The all relation uses all contour-bearing image windows for the same OSI-filtered low/high SF units.",
            "Movement modulation subtracts the zero-motion stabilized baseline matched by image_index and selected relation image set.",
            "Across-fraction control remains matched-contour only and bins across_path_fraction within each total path-length bin.",
        ],
    }
    save_json(out_dir / "phase2_contour_relation_trace_component_analysis_summary.json", summary_payload)
    save_json(out_dir / "phase2_contour_matched_trace_component_analysis_summary.json", summary_payload)
    progress(f"Done. Wrote {out_dir}")
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in summary_payload.items() if k not in {"external_unit_label_sources", "unit_bin_rows"}}, indent=2))


if __name__ == "__main__":
    main()
