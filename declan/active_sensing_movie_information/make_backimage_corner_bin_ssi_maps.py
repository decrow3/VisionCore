#!/usr/bin/env python3
"""Inspect unit/image SSI maps for two 2D component path-length surface corners."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from declan.active_sensing_movie_information.make_backimage_component_2d_surface_diagnostic import (
    MATRIX_DIR,
    MIN_OSI,
    N_BINS,
    ORTHOGONAL_MIN_DEG,
    OUT_DIR,
    SF_GROUP,
    _assign_bins,
    _compute_component_metrics,
    _finite_ratio,
    _json_ready,
    _pct_delta,
    _quantile_edges,
)
from declan.active_sensing_movie_information.plot_backimage_real_trace_unit_first_and_population_schematics import (
    accumulate_population,
    accumulate_population_movie_rows,
    baseline_rows_by_image,
    build_movie_row_grid,
    load_dataset,
    unit_image_selection,
)


OUT_STEM = "backimage_real_trace_corner_bin_ssi_maps"
RELATION = "contour_matched"
RELATION_LABEL = "high-SF units matched to strong contour"
MATCH_MAX_DEG = 22.5
EPS = 1e-12
CELLS = [
    {
        "cell_key": "top_left",
        "title": "Top-left surface cell",
        "short_label": "Q8 across / Q1 along",
        "across_bin": 8,
        "along_bin": 1,
        "why": "high across-contour path, low along-contour path",
    },
    {
        "cell_key": "bottom_right",
        "title": "Bottom-right surface cell",
        "short_label": "Q1 across / Q8 along",
        "across_bin": 1,
        "along_bin": 8,
        "why": "low across-contour path, high along-contour path",
    },
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _axis_delta_deg(a_deg: Any, b_deg: Any) -> np.ndarray:
    a = np.asarray(a_deg, dtype=np.float64)
    b = np.asarray(b_deg, dtype=np.float64)
    return np.abs(0.5 * np.degrees(np.angle(np.exp(2j * np.radians(a - b)))))


def _pct(value: float, baseline: float) -> float:
    if not (math.isfinite(value) and math.isfinite(baseline) and abs(baseline) > EPS):
        return float("nan")
    return 100.0 * (value - baseline) / baseline


def _weighted_ratio(value: np.ndarray, weight: np.ndarray) -> float:
    val = np.asarray(value, dtype=np.float64)
    wt = np.asarray(weight, dtype=np.float64)
    ok = np.isfinite(val) & np.isfinite(wt)
    if not np.any(ok):
        return float("nan")
    num = float(np.nansum(val[ok] * wt[ok]))
    den = float(np.nansum(wt[ok]))
    return _finite_ratio(num, den)


def _prepare_selection(data: dict[str, Any]) -> dict[int, np.ndarray]:
    selections = unit_image_selection(
        data["unit"],
        data["image"],
        relation=RELATION,
        sf_groups=[SF_GROUP],
        min_osi=MIN_OSI,
        match_max_deg=MATCH_MAX_DEG,
        orthogonal_min_deg=ORTHOGONAL_MIN_DEG,
        image_axis_col="image_edge_axis_deg",
    )
    return selections[SF_GROUP]


def _component_bin_masks(metrics: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drift_global = metrics[metrics["context"].astype(str).eq("drift_only")]
    across_edges = _quantile_edges(drift_global["across_path_arcmin"].to_numpy(dtype=float), N_BINS)
    along_edges = _quantile_edges(drift_global["along_path_arcmin"].to_numpy(dtype=float), N_BINS)
    across_bins = _assign_bins(metrics["across_path_arcmin"].to_numpy(dtype=float), across_edges)
    along_bins = _assign_bins(metrics["along_path_arcmin"].to_numpy(dtype=float), along_edges)
    drift_mask = metrics["context"].astype(str).eq("drift_only").to_numpy(dtype=bool)
    return across_edges, along_edges, across_bins, along_bins, drift_mask


def _cell_population_summary(
    *,
    data: dict[str, Any],
    metrics: pd.DataFrame,
    unit_to_images: dict[int, np.ndarray],
    row_mask: np.ndarray,
    baseline_lookup: dict[int, int],
    row_grid: np.ndarray,
    n_images: int,
) -> dict[str, float]:
    baseline = accumulate_population(
        ssi=data["ssi"],
        expected=data["expected"],
        stabilized_ssi=data["stabilized_ssi"],
        stabilized_expected=data["stabilized_expected"],
        row_grid=row_grid,
        baseline_lookup=baseline_lookup,
        unit_to_images=unit_to_images,
        trace_indices=None,
        n_images=n_images,
    )
    pop = accumulate_population_movie_rows(
        ssi=data["ssi"],
        expected=data["expected"],
        row_image_index=metrics["image_index"].astype(int).to_numpy(),
        row_mask=row_mask,
        unit_to_images=unit_to_images,
        n_images=n_images,
    )
    baseline_ssi = float(baseline["population_ssi_bits_per_spike"])
    baseline_info = _finite_ratio(float(baseline["information_numerator_bits"]), float(baseline["n_movie_samples"]))
    baseline_spikes = _finite_ratio(float(baseline["expected_spikes"]), float(baseline["n_movie_samples"]))
    moving_ssi = float(pop["population_ssi_bits_per_spike"])
    moving_info = _finite_ratio(float(pop["information_numerator_bits"]), float(pop["n_movie_samples"]))
    moving_spikes = _finite_ratio(float(pop["expected_spikes"]), float(pop["n_movie_samples"]))
    return {
        "baseline_population_ssi_bits_per_spike": baseline_ssi,
        "population_ssi_bits_per_spike": moving_ssi,
        "population_ssi_percent_vs_stabilized": _pct_delta(moving_ssi, baseline_ssi),
        "baseline_information_bits_per_sample": baseline_info,
        "information_bits_per_sample": moving_info,
        "information_bits_per_sample_percent_vs_stabilized": _pct_delta(moving_info, baseline_info),
        "baseline_expected_spikes_per_sample": baseline_spikes,
        "expected_spikes_per_sample": moving_spikes,
        "expected_spikes_per_sample_percent_vs_stabilized": _pct_delta(moving_spikes, baseline_spikes),
        "n_movie_samples": int(pop["n_movie_samples"]),
        "n_images_contributing": int(pop["n_images_contributing"]),
    }


def _unit_image_tables_for_cell(
    *,
    data: dict[str, Any],
    metrics: pd.DataFrame,
    unit_to_images: dict[int, np.ndarray],
    row_mask: np.ndarray,
    baseline_lookup: dict[int, int],
    cell: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ssi = data["ssi"]
    expected = data["expected"]
    stabilized_ssi = data["stabilized_ssi"]
    stabilized_expected = data["stabilized_expected"]
    unit = data["unit"].set_index("unit_index", drop=False)
    image = data["image"].set_index("image_index", drop=False)

    candidate_rows = np.flatnonzero(np.asarray(row_mask, dtype=bool))
    candidate_images = metrics["image_index"].astype(int).to_numpy()[candidate_rows]
    rows_by_image: dict[int, np.ndarray] = {}
    for image_index in sorted(set(int(v) for v in candidate_images.tolist())):
        rows_by_image[image_index] = candidate_rows[candidate_images == image_index]

    records: list[dict[str, Any]] = []
    for unit_index, image_indices in unit_to_images.items():
        unit_row = unit.loc[int(unit_index)]
        pref = float(unit_row["prior_preferred_orientation_deg"])
        for image_index in np.asarray(image_indices, dtype=int).tolist():
            rows = rows_by_image.get(int(image_index))
            if rows is None or rows.size == 0:
                continue
            image_row = image.loc[int(image_index)]
            baseline_row = int(baseline_lookup[int(image_index)])
            values = np.asarray(ssi[rows, int(unit_index)], dtype=np.float64)
            weights = np.asarray(expected[rows, int(unit_index)], dtype=np.float64)
            ok = np.isfinite(values) & np.isfinite(weights)
            if not np.any(ok):
                continue
            values = values[ok]
            weights = weights[ok]
            n = int(values.size)
            numerator = float(np.nansum(values * weights))
            spikes = float(np.nansum(weights))
            moving_ssi = _finite_ratio(numerator, spikes)
            moving_info = _finite_ratio(numerator, float(n))
            moving_spikes = _finite_ratio(spikes, float(n))
            base_ssi = float(stabilized_ssi[baseline_row, int(unit_index)])
            base_spikes = float(stabilized_expected[baseline_row, int(unit_index)])
            base_info = base_ssi * base_spikes
            records.append(
                {
                    "cell_key": str(cell["cell_key"]),
                    "cell_label": str(cell["short_label"]),
                    "across_bin": int(cell["across_bin"]),
                    "along_bin": int(cell["along_bin"]),
                    "unit_index": int(unit_index),
                    "unit_label": str(unit_row["unit_label"]),
                    "unit_preferred_orientation_deg": pref,
                    "unit_osi": float(unit_row["prior_orientation_selectivity_index"]),
                    "unit_sf_cpd": float(unit_row["dynamic_log_gaussian_marginal_sf_cpd"]),
                    "image_index": int(image_index),
                    "image_source_row": int(image_row.get("source_row", -1)),
                    "image_session": str(image_row["session"]),
                    "image_trial_idx": int(image_row["trial_idx"]),
                    "image_edge_axis_deg": float(image_row["image_edge_axis_deg"]),
                    "image_orientation_coherence": float(image_row["image_orientation_coherence"]),
                    "image_patch_rms_contrast": float(image_row["image_patch_rms_contrast"]),
                    "image_power_8plus_cpd_fraction": float(image_row["image_power_8plus_cpd_fraction"]),
                    "unit_contour_delta_deg": float(_axis_delta_deg(pref, float(image_row["image_edge_axis_deg"]))),
                    "moving_ssi_bits_per_spike": moving_ssi,
                    "stabilized_ssi_bits_per_spike": base_ssi,
                    "ssi_percent_vs_stabilized": _pct(moving_ssi, base_ssi),
                    "moving_information_bits_per_sample": moving_info,
                    "stabilized_information_bits_per_sample": base_info,
                    "information_percent_vs_stabilized": _pct(moving_info, base_info),
                    "moving_expected_spikes_per_sample": moving_spikes,
                    "stabilized_expected_spikes_per_sample": base_spikes,
                    "expected_spikes_percent_vs_stabilized": _pct(moving_spikes, base_spikes),
                    "information_delta_bits_per_sample": moving_info - base_info,
                    "expected_spikes_delta_per_sample": moving_spikes - base_spikes,
                    "n_movie_samples": n,
                }
            )
    unit_image = pd.DataFrame(records)
    image_summary = _image_summary(unit_image, cell=cell)
    return unit_image, image_summary


def _image_summary(unit_image: pd.DataFrame, *, cell: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if unit_image.empty:
        return pd.DataFrame(rows)
    for image_index, group in unit_image.groupby("image_index", sort=False):
        numerator = float(np.nansum(group["moving_information_bits_per_sample"].to_numpy(dtype=float) * group["n_movie_samples"].to_numpy(dtype=float)))
        n_samples = int(np.nansum(group["n_movie_samples"].to_numpy(dtype=float)))
        moving_info = _finite_ratio(numerator, float(n_samples))
        moving_spikes = _finite_ratio(
            float(np.nansum(group["moving_expected_spikes_per_sample"].to_numpy(dtype=float) * group["n_movie_samples"].to_numpy(dtype=float))),
            float(n_samples),
        )
        moving_num = float(np.nansum(group["moving_ssi_bits_per_spike"].to_numpy(dtype=float) * group["moving_expected_spikes_per_sample"].to_numpy(dtype=float) * group["n_movie_samples"].to_numpy(dtype=float)))
        moving_den = float(np.nansum(group["moving_expected_spikes_per_sample"].to_numpy(dtype=float) * group["n_movie_samples"].to_numpy(dtype=float)))
        moving_ssi = _finite_ratio(moving_num, moving_den)
        base_info = float(np.nanmean(group["stabilized_information_bits_per_sample"].to_numpy(dtype=float)))
        base_spikes = float(np.nanmean(group["stabilized_expected_spikes_per_sample"].to_numpy(dtype=float)))
        base_num = float(np.nansum(group["stabilized_ssi_bits_per_spike"].to_numpy(dtype=float) * group["stabilized_expected_spikes_per_sample"].to_numpy(dtype=float)))
        base_den = float(np.nansum(group["stabilized_expected_spikes_per_sample"].to_numpy(dtype=float)))
        base_ssi = _finite_ratio(base_num, base_den)
        first = group.iloc[0]
        rows.append(
            {
                "cell_key": str(cell["cell_key"]),
                "cell_label": str(cell["short_label"]),
                "image_index": int(image_index),
                "image_source_row": int(first["image_source_row"]),
                "image_session": str(first["image_session"]),
                "image_trial_idx": int(first["image_trial_idx"]),
                "image_edge_axis_deg": float(first["image_edge_axis_deg"]),
                "image_orientation_coherence": float(first["image_orientation_coherence"]),
                "image_patch_rms_contrast": float(first["image_patch_rms_contrast"]),
                "image_power_8plus_cpd_fraction": float(first["image_power_8plus_cpd_fraction"]),
                "n_units": int(group["unit_index"].nunique()),
                "n_unit_image_pairs": int(group.shape[0]),
                "n_movie_samples": n_samples,
                "moving_population_ssi_bits_per_spike": moving_ssi,
                "stabilized_population_ssi_bits_per_spike": base_ssi,
                "population_ssi_percent_vs_stabilized": _pct(moving_ssi, base_ssi),
                "moving_information_bits_per_sample": moving_info,
                "stabilized_information_bits_per_sample": base_info,
                "information_percent_vs_stabilized": _pct(moving_info, base_info),
                "moving_expected_spikes_per_sample": moving_spikes,
                "stabilized_expected_spikes_per_sample": base_spikes,
                "expected_spikes_percent_vs_stabilized": _pct(moving_spikes, base_spikes),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_population_ssi_percent_vs_stabilized"] = out["population_ssi_percent_vs_stabilized"].abs()
    return out.sort_values("abs_population_ssi_percent_vs_stabilized", ascending=False).reset_index(drop=True)


def _composition_matched_summary(unit_image: pd.DataFrame) -> dict[str, float]:
    if unit_image.empty:
        return {
            "cell_matched_baseline_population_ssi_bits_per_spike": float("nan"),
            "cell_matched_population_ssi_percent_vs_stabilized": float("nan"),
            "cell_matched_baseline_information_bits_per_sample": float("nan"),
            "cell_matched_information_bits_per_sample_percent_vs_stabilized": float("nan"),
            "cell_matched_baseline_expected_spikes_per_sample": float("nan"),
            "cell_matched_expected_spikes_per_sample_percent_vs_stabilized": float("nan"),
        }
    n = unit_image["n_movie_samples"].to_numpy(dtype=float)
    moving_num = float(np.nansum(unit_image["moving_information_bits_per_sample"].to_numpy(dtype=float) * n))
    moving_den = float(np.nansum(unit_image["moving_expected_spikes_per_sample"].to_numpy(dtype=float) * n))
    baseline_num = float(np.nansum(unit_image["stabilized_information_bits_per_sample"].to_numpy(dtype=float) * n))
    baseline_den = float(np.nansum(unit_image["stabilized_expected_spikes_per_sample"].to_numpy(dtype=float) * n))
    n_total = float(np.nansum(n))
    moving_ssi = _finite_ratio(moving_num, moving_den)
    baseline_ssi = _finite_ratio(baseline_num, baseline_den)
    moving_info = _finite_ratio(moving_num, n_total)
    baseline_info = _finite_ratio(baseline_num, n_total)
    moving_spikes = _finite_ratio(moving_den, n_total)
    baseline_spikes = _finite_ratio(baseline_den, n_total)
    return {
        "cell_matched_baseline_population_ssi_bits_per_spike": baseline_ssi,
        "cell_matched_population_ssi_percent_vs_stabilized": _pct(moving_ssi, baseline_ssi),
        "cell_matched_baseline_information_bits_per_sample": baseline_info,
        "cell_matched_information_bits_per_sample_percent_vs_stabilized": _pct(moving_info, baseline_info),
        "cell_matched_baseline_expected_spikes_per_sample": baseline_spikes,
        "cell_matched_expected_spikes_per_sample_percent_vs_stabilized": _pct(moving_spikes, baseline_spikes),
    }


def _matrix(unit_image: pd.DataFrame, units: pd.DataFrame, images: pd.DataFrame, value_col: str) -> np.ndarray:
    unit_pos = {int(v): idx for idx, v in enumerate(units["unit_index"].astype(int).tolist())}
    image_pos = {int(v): idx for idx, v in enumerate(images["image_index"].astype(int).tolist())}
    arr = np.full((len(unit_pos), len(image_pos)), np.nan, dtype=np.float64)
    for row in unit_image.itertuples(index=False):
        arr[unit_pos[int(row.unit_index)], image_pos[int(row.image_index)]] = float(getattr(row, value_col))
    return arr


def _symmetric_percentile_limit(values: list[np.ndarray], pct: float = 97.0, floor: float = 1.0) -> tuple[float, float]:
    finite = np.concatenate([arr[np.isfinite(arr)].reshape(-1) for arr in values if np.any(np.isfinite(arr))])
    if finite.size == 0:
        return -floor, floor
    vmax = float(np.nanpercentile(np.abs(finite), float(pct)))
    vmax = max(vmax, floor)
    return -vmax, vmax


def _sparse_image_labels(images: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    n = int(images.shape[0])
    if n == 0:
        return np.asarray([], dtype=int), []
    step = max(1, math.ceil(n / 12))
    ticks = np.arange(0, n, step, dtype=int)
    labels = [f"{int(images.iloc[idx]['image_index'])}\n{float(images.iloc[idx]['image_edge_axis_deg']):.0f}" for idx in ticks]
    return ticks, labels


def _plot_heatmap_page(
    *,
    unit_images: dict[str, pd.DataFrame],
    image_summaries: dict[str, pd.DataFrame],
    cell_summaries: dict[str, dict[str, float]],
) -> plt.Figure:
    all_units = (
        pd.concat([frame[["unit_index", "unit_label", "unit_preferred_orientation_deg"]] for frame in unit_images.values()], ignore_index=True)
        .drop_duplicates("unit_index")
        .sort_values("unit_preferred_orientation_deg")
        .reset_index(drop=True)
    )
    all_images = (
        pd.concat(
            [
                frame[
                    [
                        "image_index",
                        "image_edge_axis_deg",
                        "image_orientation_coherence",
                        "image_patch_rms_contrast",
                    ]
                ]
                for frame in unit_images.values()
            ],
            ignore_index=True,
        )
        .drop_duplicates("image_index")
        .sort_values(["image_edge_axis_deg", "image_index"])
        .reset_index(drop=True)
    )
    ssi_mats = [_matrix(frame, all_units, all_images, "ssi_percent_vs_stabilized") for frame in unit_images.values()]
    info_mats = [_matrix(frame, all_units, all_images, "information_percent_vs_stabilized") for frame in unit_images.values()]
    ssi_vmin, ssi_vmax = _symmetric_percentile_limit(ssi_mats, pct=96.0, floor=25.0)
    info_vmin, info_vmax = _symmetric_percentile_limit(info_mats, pct=96.0, floor=25.0)

    fig, axes = plt.subplots(3, 2, figsize=(15.2, 10.4), constrained_layout=False)
    fig.suptitle(
        "SSI maps for two component path-length surface corners",
        fontsize=14.0,
        y=0.992,
    )
    unit_labels = [
        f"{row.unit_label}\n{float(row.unit_preferred_orientation_deg):.0f}"
        for row in all_units.itertuples(index=False)
    ]
    xticks, xlabels = _sparse_image_labels(all_images)
    for col, cell in enumerate(CELLS):
        key = str(cell["cell_key"])
        unit_image = unit_images[key]
        image_summary = image_summaries[key]
        summary = cell_summaries[key]
        title = (
            f"{cell['title']}: {cell['short_label']}\n"
            f"global-base SSI {summary['population_ssi_percent_vs_stabilized']:+.1f}%, "
            f"cell-base {summary['cell_matched_population_ssi_percent_vs_stabilized']:+.1f}%, "
            f"info {summary['information_bits_per_sample_percent_vs_stabilized']:+.1f}%, "
            f"n={summary['n_movie_samples']}"
        )
        ax = axes[0, col]
        im = ax.imshow(
            _matrix(unit_image, all_units, all_images, "ssi_percent_vs_stabilized"),
            cmap="RdBu_r",
            vmin=ssi_vmin,
            vmax=ssi_vmax,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=10.0)
        ax.set_ylabel("unit; pref ori deg" if col == 0 else "")
        ax.set_yticks(np.arange(all_units.shape[0]), unit_labels if col == 0 else [])
        ax.set_xticks(xticks, [])
        ax.tick_params(labelsize=6.2)
        cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.015)
        cbar.set_label("unit-image SSI % vs stabilized", fontsize=7.0)
        cbar.ax.tick_params(labelsize=6.5)

        ax = axes[1, col]
        im = ax.imshow(
            _matrix(unit_image, all_units, all_images, "information_percent_vs_stabilized"),
            cmap="RdBu_r",
            vmin=info_vmin,
            vmax=info_vmax,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_ylabel("unit; pref ori deg" if col == 0 else "")
        ax.set_yticks(np.arange(all_units.shape[0]), unit_labels if col == 0 else [])
        ax.set_xticks(xticks, xlabels)
        ax.set_xlabel("image index; contour axis deg")
        ax.tick_params(labelsize=6.2)
        cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.015)
        cbar.set_label("unit-image info % vs stabilized", fontsize=7.0)
        cbar.ax.tick_params(labelsize=6.5)

        ax = axes[2, col]
        if image_summary.empty:
            ax.axis("off")
            continue
        plot_rows = image_summary.sort_values("image_edge_axis_deg").reset_index(drop=True)
        x = np.arange(plot_rows.shape[0])
        colors = np.where(plot_rows["population_ssi_percent_vs_stabilized"].to_numpy(dtype=float) >= 0.0, "#a83f2d", "#2f6f9f")
        ax.bar(x, plot_rows["population_ssi_percent_vs_stabilized"].to_numpy(dtype=float), color=colors, width=0.85, alpha=0.9)
        ax.axhline(0.0, color="0.25", linewidth=0.8)
        ax.set_ylabel("image-level population\nSSI % vs stabilized")
        ax.set_xlabel("matched images sorted by contour axis")
        ax.set_title(
            f"Image spread: median {float(np.nanmedian(plot_rows['population_ssi_percent_vs_stabilized'])):+.1f}%, "
            f"range {float(np.nanmin(plot_rows['population_ssi_percent_vs_stabilized'])):+.0f} to "
            f"{float(np.nanmax(plot_rows['population_ssi_percent_vs_stabilized'])):+.0f}%",
            fontsize=9.0,
        )
        ax.tick_params(labelsize=7.0)
        ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        0.016,
        (
            "Rows are high-SF units sorted by preferred orientation; columns are strong-contour images sorted by contour axis. "
            "Blank cells are unit-image pairs outside the <=22.5 deg contour match. Top-left is high-across/low-along; bottom-right is low-across/high-along."
        ),
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="0.30",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.08, hspace=0.38, wspace=0.16)
    return fig


def _plot_image_feature_page(image_summaries: dict[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), constrained_layout=False)
    fig.suptitle("Image-feature locations of the two corner cells", fontsize=13.0, y=0.985)
    all_vals = pd.concat(
        [frame["population_ssi_percent_vs_stabilized"] for frame in image_summaries.values() if not frame.empty],
        ignore_index=True,
    ).to_numpy(dtype=float)
    finite = all_vals[np.isfinite(all_vals)]
    vmax = max(float(np.nanpercentile(np.abs(finite), 96.0)) if finite.size else 1.0, 25.0)
    for col, cell in enumerate(CELLS):
        key = str(cell["cell_key"])
        frame = image_summaries[key].copy()
        for row_idx, (x_col, y_col, xlabel, ylabel) in enumerate(
            [
                ("image_edge_axis_deg", "image_orientation_coherence", "contour axis deg", "orientation coherence"),
                ("image_patch_rms_contrast", "image_power_8plus_cpd_fraction", "patch RMS contrast", "8+ cpd power fraction"),
            ]
        ):
            ax = axes[row_idx, col]
            if frame.empty:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
                continue
            size = 20.0 + 85.0 * np.clip(frame["n_units"].to_numpy(dtype=float) / max(float(frame["n_units"].max()), 1.0), 0.0, 1.0)
            im = ax.scatter(
                frame[x_col].to_numpy(dtype=float),
                frame[y_col].to_numpy(dtype=float),
                c=frame["population_ssi_percent_vs_stabilized"].to_numpy(dtype=float),
                s=size,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                edgecolor="0.22",
                linewidth=0.25,
                alpha=0.88,
            )
            ax.axhline(0.0, color="0.84", linewidth=0.7, zorder=0)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(f"{cell['short_label']}: {cell['why']}", fontsize=9.2)
            ax.tick_params(labelsize=7.0)
            ax.spines[["top", "right"]].set_visible(False)
            cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.018)
            cbar.set_label("image-level SSI %", fontsize=7.0)
            cbar.ax.tick_params(labelsize=6.5)
    fig.text(
        0.5,
        0.02,
        "Each point is one image contributing to the matched high-SF population in that component-path cell; point size tracks matched unit count.",
        ha="center",
        fontsize=8.0,
        color="0.30",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.09, hspace=0.34, wspace=0.28)
    return fig


def _attach_patch_centers(image_summary: pd.DataFrame, image_table: pd.DataFrame) -> pd.DataFrame:
    if image_summary.empty:
        return image_summary
    cols = ["image_index", "image_patch_center_x_px", "image_patch_center_y_px"]
    return image_summary.merge(image_table[cols], on="image_index", how="left", validate="one_to_one")


def main() -> None:
    data = load_dataset(MATRIX_DIR)
    metrics = _compute_component_metrics(data)
    unit_to_images = _prepare_selection(data)
    baseline_lookup = baseline_rows_by_image(data["image"], data["baseline_table"])
    row_grid = build_movie_row_grid(data["movie"])
    n_images = int(data["stabilized_ssi"].shape[0])
    across_edges, along_edges, across_bins, along_bins, drift_mask = _component_bin_masks(metrics)

    unit_image_frames: list[pd.DataFrame] = []
    image_summary_frames: list[pd.DataFrame] = []
    cell_summaries: dict[str, dict[str, float]] = {}
    row_masks: dict[str, np.ndarray] = {}
    for cell in CELLS:
        row_mask = (
            drift_mask
            & (across_bins == int(cell["across_bin"]) - 1)
            & (along_bins == int(cell["along_bin"]) - 1)
        )
        row_masks[str(cell["cell_key"])] = row_mask
        cell_summaries[str(cell["cell_key"])] = _cell_population_summary(
            data=data,
            metrics=metrics,
            unit_to_images=unit_to_images,
            row_mask=row_mask,
            baseline_lookup=baseline_lookup,
            row_grid=row_grid,
            n_images=n_images,
        )
        unit_image, image_summary = _unit_image_tables_for_cell(
            data=data,
            metrics=metrics,
            unit_to_images=unit_to_images,
            row_mask=row_mask,
            baseline_lookup=baseline_lookup,
            cell=cell,
        )
        cell_summaries[str(cell["cell_key"])].update(_composition_matched_summary(unit_image))
        unit_image_frames.append(unit_image)
        image_summary_frames.append(_attach_patch_centers(image_summary, data["image"]))

    unit_image_all = pd.concat(unit_image_frames, ignore_index=True)
    image_summary_all = pd.concat(image_summary_frames, ignore_index=True)
    unit_images = {str(cell["cell_key"]): frame for cell, frame in zip(CELLS, unit_image_frames, strict=True)}
    image_summaries = {str(cell["cell_key"]): frame for cell, frame in zip(CELLS, image_summary_frames, strict=True)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unit_image_csv = OUT_DIR / f"{OUT_STEM}_unit_image_values.csv"
    image_summary_csv = OUT_DIR / f"{OUT_STEM}_image_summary.csv"
    summary_json = OUT_DIR / f"{OUT_STEM}_summary.json"
    pdf = OUT_DIR / f"{OUT_STEM}.pdf"
    png = OUT_DIR / f"{OUT_STEM}_heatmaps.png"
    unit_image_all.to_csv(unit_image_csv, index=False)
    image_summary_all.to_csv(image_summary_csv, index=False)

    heatmap_fig = _plot_heatmap_page(
        unit_images=unit_images,
        image_summaries=image_summaries,
        cell_summaries=cell_summaries,
    )
    heatmap_fig.savefig(png, dpi=240, bbox_inches="tight")
    with PdfPages(pdf) as pages:
        pages.savefig(heatmap_fig, bbox_inches="tight")
        plt.close(heatmap_fig)
        feature_fig = _plot_image_feature_page(image_summaries)
        pages.savefig(feature_fig, bbox_inches="tight")
        plt.close(feature_fig)

    _write_json(
        summary_json,
        {
            "analysis": "backimage_real_trace_corner_bin_ssi_maps",
            "matrix_dir": MATRIX_DIR,
            "out_dir": OUT_DIR,
            "outputs": {
                "pdf": pdf,
                "heatmap_png": png,
                "unit_image_csv": unit_image_csv,
                "image_summary_csv": image_summary_csv,
                "summary_json": summary_json,
            },
            "selection": {
                "relation": RELATION,
                "relation_label": RELATION_LABEL,
                "sf_group": SF_GROUP,
                "min_osi": MIN_OSI,
                "match_max_deg": MATCH_MAX_DEG,
                "n_units": int(len(unit_to_images)),
                "n_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
            },
            "component_bins": {
                "n_bins": N_BINS,
                "across_edges_arcmin": across_edges,
                "along_edges_arcmin": along_edges,
                "cells": CELLS,
            },
            "cell_summaries": cell_summaries,
            "note": (
                "The maps show directly observed scalar SSI from the real-trace matrix, aggregated over movie rows "
                "inside each component path-length cell. They are not raw model activation maps; image thumbnails are "
                "omitted here to avoid slow BackImage session loading."
            ),
        },
    )
    print(pdf)
    print(png)
    print(unit_image_csv)
    print(image_summary_csv)
    print(summary_json)


if __name__ == "__main__":
    main()
