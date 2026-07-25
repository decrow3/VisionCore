#!/usr/bin/env python3
"""Reconstruct the full drift path curve from a 2D RMS-excursion surface.

This is a quantitative version of the "hidden cancellation" check.  For the
high-SF units aligned to strong image contours, estimate response surfaces over
(across-contour RMS, along-contour RMS), then predict each total-path bin by
reweighting those surfaces with the empirical joint RMS-cell mix in that path
bin.
"""

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

from declan.active_sensing_movie_information.make_backimage_component_rms_excursion_diagnostic import (
    _compute_rms_metrics,
    _finite_ratio,
    _pct_delta,
    _write_json,
)
from declan.active_sensing_movie_information.plot_backimage_real_trace_unit_first_and_population_schematics import (
    add_equal_count_trace_bins,
    accumulate_population,
    accumulate_population_movie_rows,
    baseline_rows_by_image,
    build_movie_row_grid,
    load_dataset,
    ratio_delta_stats,
    unit_image_selection,
)


MATRIX_DIR = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged"
)
CONDITION_DIR = MATRIX_DIR / "phase1_phase2_conditioning_v1"
OUT_DIR = CONDITION_DIR / "plot_collections"
OUT_STEM = "backimage_real_trace_rms_reconstruction_diagnostic"

SF_GROUP = "high_sf"
RELATION = "contour_matched"
RELATION_LABEL = "Aligned high-SF units on strong contours"
MIN_OSI = 0.05
MATCH_MAX_DEG = 22.5
ORTHOGONAL_MIN_DEG = 67.5
N_PATH_BINS = 8
N_MICROSACCADE_BINS = 5
N_RMS_BINS = 4
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 47
EPS = 1e-12

OUTCOMES = [
    (
        "population_ssi_percent_vs_stabilized",
        "SSI bits/spike\nchange (%)",
        "SSI efficiency",
        "RdBu_r",
    ),
    (
        "information_bits_per_sample_percent_vs_stabilized",
        "information bits/window\nchange (%)",
        "Information",
        "RdBu_r",
    ),
    (
        "expected_spikes_per_sample_percent_vs_stabilized",
        "expected spikes/window\nchange (%)",
        "Expected spikes",
        "RdBu_r",
    ),
]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(val) for val in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot create RMS bins without finite values.")
    edges = np.quantile(finite, np.linspace(0.0, 1.0, int(n_bins) + 1))
    # Expand the end points slightly so values exactly on the empirical min/max
    # are included after digitization.
    span = max(float(edges[-1] - edges[0]), 1e-6)
    edges[0] -= 1e-6 * span
    edges[-1] += 1e-6 * span
    return edges


def _assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, -1, dtype=int)
    ok = np.isfinite(arr)
    out[ok] = np.searchsorted(edges[1:-1], arr[ok], side="right")
    out[(out < 0) | (out >= len(edges) - 1)] = -1
    return out


def _selected_sample_count(
    *,
    row_image_index: np.ndarray,
    row_mask: np.ndarray,
    unit_to_images: dict[int, np.ndarray],
    n_images: int,
) -> int:
    candidate_rows = np.flatnonzero(np.asarray(row_mask, dtype=bool))
    if candidate_rows.size == 0:
        return 0
    candidate_images = np.asarray(row_image_index[candidate_rows], dtype=int)
    total = 0
    for image_indices in unit_to_images.values():
        images = np.asarray(image_indices, dtype=int)
        if images.size == 0:
            continue
        selected = np.zeros(n_images, dtype=bool)
        selected[images] = True
        total += int(np.count_nonzero(selected[candidate_images]))
    return total


def _population_row(
    *,
    pop: dict[str, Any],
    baseline_ssi: float,
    baseline_info_per_sample: float,
    baseline_spikes_per_sample: float,
) -> dict[str, float]:
    ssi = float(pop["population_ssi_bits_per_spike"])
    info_per_sample = _finite_ratio(float(pop["information_numerator_bits"]), float(pop["n_movie_samples"]))
    spikes_per_sample = _finite_ratio(float(pop["expected_spikes"]), float(pop["n_movie_samples"]))
    return {
        "population_ssi_bits_per_spike": ssi,
        "population_ssi_percent_vs_stabilized": _pct_delta(ssi, baseline_ssi),
        "information_bits_per_sample": info_per_sample,
        "information_bits_per_sample_percent_vs_stabilized": _pct_delta(info_per_sample, baseline_info_per_sample),
        "expected_spikes_per_sample": spikes_per_sample,
        "expected_spikes_per_sample_percent_vs_stabilized": _pct_delta(
            spikes_per_sample,
            baseline_spikes_per_sample,
        ),
        "information_numerator_bits": float(pop["information_numerator_bits"]),
        "expected_spikes": float(pop["expected_spikes"]),
    }


def _prepare_data() -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    dict[int, np.ndarray],
    dict[str, Any],
    float,
    float,
    float,
]:
    data = load_dataset(MATRIX_DIR)
    component_metrics, _bins_by_metric = _compute_rms_metrics(data)
    trace_with_bins, path_bins = add_equal_count_trace_bins(
        data["trace"],
        n_drift_bins=N_PATH_BINS,
        n_microsaccade_bins=N_MICROSACCADE_BINS,
    )
    path_bin_by_trace = trace_with_bins["path_bin"].astype(object).to_numpy()
    path_order_by_trace = pd.to_numeric(trace_with_bins["path_bin_order"], errors="coerce").to_numpy(dtype=float)
    trace_index = component_metrics["trace_index"].astype(int).to_numpy()
    component_metrics = component_metrics.copy()
    component_metrics["path_bin"] = path_bin_by_trace[trace_index]
    component_metrics["path_bin_order"] = path_order_by_trace[trace_index]

    row_grid = build_movie_row_grid(data["movie"])
    baseline_lookup = baseline_rows_by_image(data["image"], data["baseline_table"])
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
    unit_to_images = selections[SF_GROUP]
    n_images = int(data["stabilized_ssi"].shape[0])
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
    baseline_ssi = float(baseline["population_ssi_bits_per_spike"])
    baseline_info = _finite_ratio(float(baseline["information_numerator_bits"]), float(baseline["n_movie_samples"]))
    baseline_spikes = _finite_ratio(float(baseline["expected_spikes"]), float(baseline["n_movie_samples"]))
    return data, component_metrics, path_bins, unit_to_images, baseline, baseline_ssi, baseline_info, baseline_spikes


def _build_cell_surface(
    *,
    data: dict[str, Any],
    component_metrics: pd.DataFrame,
    unit_to_images: dict[int, np.ndarray],
    baseline_ssi: float,
    baseline_info_per_sample: float,
    baseline_spikes_per_sample: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    drift = component_metrics[component_metrics["context"].astype(str).eq("drift_only")]
    across_edges = _quantile_edges(drift["across_rms_arcmin"].to_numpy(dtype=float), N_RMS_BINS)
    along_edges = _quantile_edges(drift["along_rms_arcmin"].to_numpy(dtype=float), N_RMS_BINS)
    component_metrics["across_rms_bin"] = _assign_bins(
        component_metrics["across_rms_arcmin"].to_numpy(dtype=float),
        across_edges,
    )
    component_metrics["along_rms_bin"] = _assign_bins(
        component_metrics["along_rms_arcmin"].to_numpy(dtype=float),
        along_edges,
    )

    row_image_index = component_metrics["image_index"].astype(int).to_numpy()
    n_images = int(data["stabilized_ssi"].shape[0])
    rows: list[dict[str, Any]] = []
    for across_bin in range(N_RMS_BINS):
        for along_bin in range(N_RMS_BINS):
            row_mask = (
                component_metrics["context"].astype(str).eq("drift_only").to_numpy(dtype=bool)
                & component_metrics["across_rms_bin"].eq(across_bin).to_numpy(dtype=bool)
                & component_metrics["along_rms_bin"].eq(along_bin).to_numpy(dtype=bool)
            )
            pop = accumulate_population_movie_rows(
                ssi=data["ssi"],
                expected=data["expected"],
                row_image_index=row_image_index,
                row_mask=row_mask,
                unit_to_images=unit_to_images,
                n_images=n_images,
            )
            out = _population_row(
                pop=pop,
                baseline_ssi=baseline_ssi,
                baseline_info_per_sample=baseline_info_per_sample,
                baseline_spikes_per_sample=baseline_spikes_per_sample,
            )
            global_rows = component_metrics[row_mask]
            rows.append(
                {
                    "across_rms_bin": int(across_bin),
                    "along_rms_bin": int(along_bin),
                    "across_rms_min_arcmin": float(across_edges[across_bin]),
                    "across_rms_max_arcmin": float(across_edges[across_bin + 1]),
                    "along_rms_min_arcmin": float(along_edges[along_bin]),
                    "along_rms_max_arcmin": float(along_edges[along_bin + 1]),
                    "across_rms_median_arcmin": float(np.nanmedian(global_rows["across_rms_arcmin"])),
                    "along_rms_median_arcmin": float(np.nanmedian(global_rows["along_rms_arcmin"])),
                    "n_movie_rows_global": int(np.count_nonzero(row_mask)),
                    "n_selected_movie_samples": int(pop["n_movie_samples"]),
                    **out,
                }
            )
    return pd.DataFrame(rows), across_edges, along_edges


def _path_reconstruction(
    *,
    data: dict[str, Any],
    component_metrics: pd.DataFrame,
    path_bins: pd.DataFrame,
    cell_surface: pd.DataFrame,
    unit_to_images: dict[int, np.ndarray],
    baseline: dict[str, Any],
    baseline_ssi: float,
    baseline_info_per_sample: float,
    baseline_spikes_per_sample: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    row_image_index = component_metrics["image_index"].astype(int).to_numpy()
    n_images = int(data["stabilized_ssi"].shape[0])
    rows: list[dict[str, Any]] = []
    for path_row in path_bins[path_bins["context"].astype(str).eq("drift_only")].itertuples(index=False):
        path_bin = str(path_row.path_bin)
        path_mask = component_metrics["path_bin"].astype(str).eq(path_bin).to_numpy(dtype=bool)
        observed = accumulate_population_movie_rows(
            ssi=data["ssi"],
            expected=data["expected"],
            row_image_index=row_image_index,
            row_mask=path_mask,
            unit_to_images=unit_to_images,
            n_images=n_images,
        )
        observed_row = _population_row(
            pop=observed,
            baseline_ssi=baseline_ssi,
            baseline_info_per_sample=baseline_info_per_sample,
            baseline_spikes_per_sample=baseline_spikes_per_sample,
        )
        delta_stats = ratio_delta_stats(
            observed["per_image_num"],
            observed["per_image_den"],
            baseline["per_image_num"],
            baseline["per_image_den"],
            n_resamples=N_BOOTSTRAP,
            rng=rng,
        )

        weighted_info = 0.0
        weighted_spikes = 0.0
        weighted_ssi_percent = 0.0
        selected_sample_count = 0
        cell_count_rows: list[dict[str, Any]] = []
        for cell in cell_surface.itertuples(index=False):
            cell_mask = (
                path_mask
                & component_metrics["across_rms_bin"].eq(int(cell.across_rms_bin)).to_numpy(dtype=bool)
                & component_metrics["along_rms_bin"].eq(int(cell.along_rms_bin)).to_numpy(dtype=bool)
            )
            count = _selected_sample_count(
                row_image_index=row_image_index,
                row_mask=cell_mask,
                unit_to_images=unit_to_images,
                n_images=n_images,
            )
            selected_sample_count += count
            cell_count_rows.append({"cell": cell, "count": count})
        for item in cell_count_rows:
            count = int(item["count"])
            if count <= 0 or selected_sample_count <= 0:
                continue
            cell = item["cell"]
            weight = count / float(selected_sample_count)
            weighted_info += weight * float(cell.information_bits_per_sample)
            weighted_spikes += weight * float(cell.expected_spikes_per_sample)
            weighted_ssi_percent += weight * float(cell.population_ssi_percent_vs_stabilized)
        predicted_ssi = _finite_ratio(weighted_info, weighted_spikes)
        predicted_ssi_percent = _pct_delta(predicted_ssi, baseline_ssi)
        rows.append(
            {
                "path_bin": path_bin,
                "path_bin_order": int(path_row.path_bin_order),
                "path_median_arcmin": float(path_row.median_path_arcmin),
                "path_q25_arcmin": float(path_row.q25_path_arcmin),
                "path_q75_arcmin": float(path_row.q75_path_arcmin),
                "n_traces": int(path_row.n_traces),
                "n_selected_movie_samples": int(observed["n_movie_samples"]),
                "n_selected_movie_samples_reconstruction": int(selected_sample_count),
                "observed_population_ssi_bits_per_spike": observed_row["population_ssi_bits_per_spike"],
                "observed_population_ssi_percent_vs_stabilized": observed_row[
                    "population_ssi_percent_vs_stabilized"
                ],
                "observed_information_bits_per_sample": observed_row["information_bits_per_sample"],
                "observed_information_bits_per_sample_percent_vs_stabilized": observed_row[
                    "information_bits_per_sample_percent_vs_stabilized"
                ],
                "observed_expected_spikes_per_sample": observed_row["expected_spikes_per_sample"],
                "observed_expected_spikes_per_sample_percent_vs_stabilized": observed_row[
                    "expected_spikes_per_sample_percent_vs_stabilized"
                ],
                "observed_population_delta_p_image_bootstrap_sign": float(
                    delta_stats["population_delta_p_image_bootstrap_sign"]
                ),
                "predicted_population_ssi_bits_per_spike": predicted_ssi,
                "predicted_population_ssi_percent_vs_stabilized": predicted_ssi_percent,
                "predicted_population_ssi_percent_direct_cell_average": weighted_ssi_percent,
                "predicted_information_bits_per_sample": weighted_info,
                "predicted_information_bits_per_sample_percent_vs_stabilized": _pct_delta(
                    weighted_info,
                    baseline_info_per_sample,
                ),
                "predicted_expected_spikes_per_sample": weighted_spikes,
                "predicted_expected_spikes_per_sample_percent_vs_stabilized": _pct_delta(
                    weighted_spikes,
                    baseline_spikes_per_sample,
                ),
                "residual_population_ssi_percent": observed_row["population_ssi_percent_vs_stabilized"]
                - predicted_ssi_percent,
                "residual_information_bits_per_sample_percent": observed_row[
                    "information_bits_per_sample_percent_vs_stabilized"
                ]
                - _pct_delta(weighted_info, baseline_info_per_sample),
                "residual_expected_spikes_per_sample_percent": observed_row[
                    "expected_spikes_per_sample_percent_vs_stabilized"
                ]
                - _pct_delta(weighted_spikes, baseline_spikes_per_sample),
            }
        )
    return pd.DataFrame(rows)


def _sym_limits(values: pd.Series, observed: pd.Series | None = None, predicted: pd.Series | None = None) -> tuple[float, float]:
    parts = [pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)]
    if observed is not None:
        parts.append(pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float))
    if predicted is not None:
        parts.append(pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float))
    arr = np.concatenate(parts)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0
    mag = max(abs(float(np.nanmin(arr))), abs(float(np.nanmax(arr))), 1.0)
    return -1.10 * mag, 1.10 * mag


def _plot(
    *,
    path_summary: pd.DataFrame,
    cell_surface: pd.DataFrame,
    across_edges: np.ndarray,
    along_edges: np.ndarray,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, len(OUTCOMES), figsize=(12.4, 7.25))
    x = path_summary["path_median_arcmin"].to_numpy(dtype=float)
    along_tick_labels = []
    across_tick_labels = []
    for bin_idx in range(N_RMS_BINS):
        along_vals = cell_surface[cell_surface["along_rms_bin"].eq(bin_idx)]["along_rms_median_arcmin"]
        across_vals = cell_surface[cell_surface["across_rms_bin"].eq(bin_idx)]["across_rms_median_arcmin"]
        along_tick_labels.append(f"Q{bin_idx + 1}\n{float(np.nanmedian(along_vals)):.1f}")
        across_tick_labels.append(f"Q{bin_idx + 1}\n{float(np.nanmedian(across_vals)):.1f}")
    for col_idx, (field, ylabel, title, cmap) in enumerate(OUTCOMES):
        ax = axes[0, col_idx]
        obs_col = f"observed_{field}"
        pred_col = f"predicted_{field}"
        observed = path_summary[obs_col].to_numpy(dtype=float)
        predicted = path_summary[pred_col].to_numpy(dtype=float)
        ax.plot(x, observed, color="0.1", marker="o", markersize=4.5, linewidth=1.8, label="observed total-path curve")
        ax.plot(
            x,
            predicted,
            color="#CC79A7",
            marker="s",
            markersize=4.0,
            linewidth=1.7,
            label="2D RMS reconstruction",
        )
        ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.85)
        ax.grid(True, color="0.90", linewidth=0.75)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("total drift path length (arcmin)", fontsize=8.6)
        ax.set_ylabel(ylabel, fontsize=8.7)
        ax.set_title(title, fontsize=10.2)
        ymin = min(np.nanmin(observed), np.nanmin(predicted), 0.0)
        ymax = max(np.nanmax(observed), np.nanmax(predicted), 0.0)
        span = max(ymax - ymin, 1.0)
        ax.set_ylim(ymin - 0.16 * span, ymax + 0.18 * span)
        ax.tick_params(labelsize=8.0)
        if col_idx == 0:
            first = path_summary.sort_values("path_bin_order").iloc[0]
            ax.text(
                0.04,
                0.08,
                f"first-bin p={first['observed_population_delta_p_image_bootstrap_sign']:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.6,
                color="0.30",
            )
            ax.legend(loc="upper right", frameon=False, fontsize=7.7)

        heat_ax = axes[1, col_idx]
        z = np.full((N_RMS_BINS, N_RMS_BINS), np.nan, dtype=float)
        for cell in cell_surface.itertuples(index=False):
            z[int(cell.across_rms_bin), int(cell.along_rms_bin)] = float(getattr(cell, field))
        vmin, vmax = _sym_limits(cell_surface[field], path_summary[obs_col], path_summary[pred_col])
        mesh = heat_ax.imshow(z, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        heat_ax.set_xticks(np.arange(N_RMS_BINS), along_tick_labels)
        heat_ax.set_yticks(np.arange(N_RMS_BINS), across_tick_labels)
        heat_ax.set_xlabel(r"$\sigma_{\mathrm{along}}$ quantile; median arcmin", fontsize=8.8)
        heat_ax.set_ylabel(r"$\sigma_{\mathrm{across}}$ quantile; median arcmin", fontsize=8.8)
        heat_ax.set_title(f"2D RMS surface: {title}", fontsize=9.8)
        heat_ax.tick_params(labelsize=8.0)
        heat_ax.spines[["top", "right"]].set_visible(False)
        threshold = 0.52 * max(abs(vmin), abs(vmax))
        for across_bin in range(N_RMS_BINS):
            for along_bin in range(N_RMS_BINS):
                value = z[across_bin, along_bin]
                if not math.isfinite(value):
                    continue
                heat_ax.text(
                    along_bin,
                    across_bin,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.4,
                    color="white" if abs(value) >= threshold else "0.18",
                )
        cbar = fig.colorbar(mesh, ax=heat_ax, fraction=0.046, pad=0.025)
        cbar.set_label("% vs stabilized", fontsize=7.7)
        cbar.ax.tick_params(labelsize=7.3)
    fig.suptitle(
        "Can the flat total-path result be reconstructed from the joint RMS-excursion distribution?",
        fontsize=13.0,
        y=0.982,
    )
    fig.text(
        0.5,
        0.018,
        "Reconstruction weights each RMS cell by its selected unit-image sample count inside the total-path bin. "
        "RMS heatmaps use quantile bins; tick numbers are medians in arcmin. "
        "A residual means RMS bins are too coarse, the effect is nonstationary within cells, or another variable carries the marginal.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="0.30",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.105, wspace=0.34, hspace=0.43)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{OUT_STEM}.png"
    pdf = OUT_DIR / f"{OUT_STEM}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    (
        data,
        component_metrics,
        path_bins,
        unit_to_images,
        baseline,
        baseline_ssi,
        baseline_info,
        baseline_spikes,
    ) = _prepare_data()
    cell_surface, across_edges, along_edges = _build_cell_surface(
        data=data,
        component_metrics=component_metrics,
        unit_to_images=unit_to_images,
        baseline_ssi=baseline_ssi,
        baseline_info_per_sample=baseline_info,
        baseline_spikes_per_sample=baseline_spikes,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    path_summary = _path_reconstruction(
        data=data,
        component_metrics=component_metrics,
        path_bins=path_bins,
        cell_surface=cell_surface,
        unit_to_images=unit_to_images,
        baseline=baseline,
        baseline_ssi=baseline_ssi,
        baseline_info_per_sample=baseline_info,
        baseline_spikes_per_sample=baseline_spikes,
        rng=rng,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path_csv = OUT_DIR / f"{OUT_STEM}_path_reconstruction.csv"
    cell_csv = OUT_DIR / f"{OUT_STEM}_rms_cell_surface.csv"
    json_path = OUT_DIR / f"{OUT_STEM}_summary.json"
    path_summary.to_csv(path_csv, index=False)
    cell_surface.to_csv(cell_csv, index=False)
    png, pdf = _plot(
        path_summary=path_summary,
        cell_surface=cell_surface,
        across_edges=across_edges,
        along_edges=along_edges,
    )
    residual_summary = {
        "ssi_percent_abs_residual_median": float(np.nanmedian(np.abs(path_summary["residual_population_ssi_percent"]))),
        "ssi_percent_abs_residual_max": float(np.nanmax(np.abs(path_summary["residual_population_ssi_percent"]))),
        "information_percent_abs_residual_median": float(
            np.nanmedian(np.abs(path_summary["residual_information_bits_per_sample_percent"]))
        ),
        "expected_spikes_percent_abs_residual_median": float(
            np.nanmedian(np.abs(path_summary["residual_expected_spikes_per_sample_percent"]))
        ),
    }
    _write_json(
        json_path,
        _json_ready(
            {
                "analysis": "backimage_real_trace_rms_reconstruction_diagnostic",
                "matrix_dir": MATRIX_DIR,
                "out_dir": OUT_DIR,
                "outputs": {
                    "png": png,
                    "pdf": pdf,
                    "path_reconstruction_csv": path_csv,
                    "rms_cell_surface_csv": cell_csv,
                    "summary_json": json_path,
                },
                "selection": {
                    "relation": RELATION,
                    "relation_label": RELATION_LABEL,
                    "sf_group": SF_GROUP,
                    "min_osi": MIN_OSI,
                    "match_max_deg": MATCH_MAX_DEG,
                    "orthogonal_min_deg": ORTHOGONAL_MIN_DEG,
                    "n_units": len(unit_to_images),
                    "n_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
                },
                "rms_bins": {
                    "n_bins_per_axis": N_RMS_BINS,
                    "across_edges_arcmin": across_edges,
                    "along_edges_arcmin": along_edges,
                },
                "baseline": {
                    "population_ssi_bits_per_spike": baseline_ssi,
                    "information_bits_per_sample": baseline_info,
                    "expected_spikes_per_sample": baseline_spikes,
                },
                "residual_summary": residual_summary,
                "n_bootstrap": N_BOOTSTRAP,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "note": (
                    "This is an in-sample coarsened reconstruction. It tests whether the total-path marginal "
                    "is explainable by the empirical distribution over RMS-excursion cells, not whether RMS "
                    "excursion is causally identified."
                ),
            }
        ),
    )
    print(png)
    print(pdf)
    print(path_csv)
    print(cell_csv)
    print(json_path)


if __name__ == "__main__":
    main()
