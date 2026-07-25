#!/usr/bin/env python3
"""2D component-path residual surfaces across high-SF thresholds."""

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
from matplotlib.colors import TwoSlopeNorm

from declan.active_sensing_movie_information.make_backimage_component_2d_surface_diagnostic import (
    MATRIX_DIR,
    OUT_DIR,
    _assign_bins,
    _compute_component_metrics,
    _format_count,
    _json_ready,
    _quantile_edges,
)
from declan.active_sensing_movie_information.make_backimage_component_path_baseline_decomposition_surface import (
    _cell_matched_baseline,
    _population_values,
)
from declan.active_sensing_movie_information.plot_backimage_real_trace_unit_first_and_population_schematics import (
    accumulate_population_movie_rows,
    baseline_rows_by_image,
    load_dataset,
    unit_image_selection,
)


OUT_STEM = "backimage_real_trace_component_path_sf_threshold_surface_sweep_coh020_n8"
SF_METRIC_COL = "sf_split_metric"
SF_GROUP = "high_sf"
SF_MIN_CPDS = (0.50, 0.625, 0.75, 0.875, 1.00)
CONTOUR_COHERENCE_MIN = 0.20
RELATION = "contour_matched"
MIN_OSI = 0.05
MATCH_MAX_DEG = 22.5
ORTHOGONAL_MIN_DEG = 67.5
N_BINS = 8
EPS = 1e-12

OUTCOMES = [
    ("ssi_percent_vs_cell_baseline", "SSI bits/spike residual", "% vs cell-stabilized baseline"),
    ("information_percent_vs_cell_baseline", "Information residual", "% vs cell-stabilized baseline"),
    ("spikes_percent_vs_cell_baseline", "Expected-spike residual", "% vs cell-stabilized baseline"),
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_ratio(num: float, den: float) -> float:
    return float(num / den) if math.isfinite(num) and math.isfinite(den) and den > EPS else float("nan")


def _pct_delta(value: float, baseline: float) -> float:
    if not (math.isfinite(value) and math.isfinite(baseline) and abs(baseline) > EPS):
        return float("nan")
    return 100.0 * (value - baseline) / baseline


def _thresholded_data(data: dict[str, Any], *, sf_min_cpd: float) -> dict[str, Any]:
    out = dict(data)

    image = data["image"].copy()
    coherence = pd.to_numeric(image["image_orientation_coherence"], errors="coerce").to_numpy(dtype=float)
    axis = pd.to_numeric(image["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=float)
    image["image_contour_strong"] = np.isfinite(axis) & np.isfinite(coherence) & (
        coherence >= float(CONTOUR_COHERENCE_MIN)
    )
    out["image"] = image

    unit = data["unit"].copy()
    sf = pd.to_numeric(unit[SF_METRIC_COL], errors="coerce").to_numpy(dtype=float)
    unit["sf_group_original"] = unit["sf_group"].astype(str)
    unit["sf_group"] = np.where(np.isfinite(sf) & (sf >= float(sf_min_cpd)), SF_GROUP, "below_sf_threshold")
    unit["sf_group_label"] = np.where(
        unit["sf_group"].eq(SF_GROUP),
        f"SF >= {float(sf_min_cpd):.3g} cpd",
        f"SF < {float(sf_min_cpd):.3g} cpd",
    )
    out["unit"] = unit
    return out


def _surface_values(
    data: dict[str, Any],
    metrics: pd.DataFrame,
    *,
    sf_min_cpd: float,
    across_edges: np.ndarray,
    along_edges: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    threshold_data = _thresholded_data(data, sf_min_cpd=sf_min_cpd)
    selections = unit_image_selection(
        threshold_data["unit"],
        threshold_data["image"],
        relation=RELATION,
        sf_groups=[SF_GROUP],
        min_osi=MIN_OSI,
        match_max_deg=MATCH_MAX_DEG,
        orthogonal_min_deg=ORTHOGONAL_MIN_DEG,
        image_axis_col="image_edge_axis_deg",
    )
    unit_to_images = selections[SF_GROUP]
    baseline_lookup = baseline_rows_by_image(threshold_data["image"], threshold_data["baseline_table"])
    n_images = int(threshold_data["stabilized_ssi"].shape[0])
    row_image_index = metrics["image_index"].astype(int).to_numpy()

    drift_mask = metrics["context"].astype(str).eq("drift_only").to_numpy(dtype=bool)
    across_bins = _assign_bins(metrics["across_path_arcmin"].to_numpy(dtype=float), across_edges)
    along_bins = _assign_bins(metrics["along_path_arcmin"].to_numpy(dtype=float), along_edges)

    rows: list[dict[str, Any]] = []
    for across_bin in range(N_BINS):
        for along_bin in range(N_BINS):
            row_mask = drift_mask & (across_bins == across_bin) & (along_bins == along_bin)
            moving_pop = accumulate_population_movie_rows(
                ssi=threshold_data["ssi"],
                expected=threshold_data["expected"],
                row_image_index=row_image_index,
                row_mask=row_mask,
                unit_to_images=unit_to_images,
                n_images=n_images,
            )
            cell_pop = _cell_matched_baseline(
                stabilized_ssi=threshold_data["stabilized_ssi"],
                stabilized_expected=threshold_data["stabilized_expected"],
                row_image_index=row_image_index,
                row_mask=row_mask,
                baseline_lookup=baseline_lookup,
                unit_to_images=unit_to_images,
                n_images=n_images,
            )
            moving = _population_values(moving_pop)
            cell = _population_values(cell_pop)
            global_rows = metrics[row_mask]
            rows.append(
                {
                    "sf_min_cpd": float(sf_min_cpd),
                    "sf_metric_col": SF_METRIC_COL,
                    "contour_coherence_min": float(CONTOUR_COHERENCE_MIN),
                    "relation": RELATION,
                    "across_bin": int(across_bin + 1),
                    "along_bin": int(along_bin + 1),
                    "across_min_arcmin": float(across_edges[across_bin]),
                    "across_max_arcmin": float(across_edges[across_bin + 1]),
                    "along_min_arcmin": float(along_edges[along_bin]),
                    "along_max_arcmin": float(along_edges[along_bin + 1]),
                    "across_median_arcmin": float(np.nanmedian(global_rows["across_path_arcmin"])),
                    "along_median_arcmin": float(np.nanmedian(global_rows["along_path_arcmin"])),
                    "n_movie_rows_global": int(np.count_nonzero(row_mask)),
                    "n_movie_samples": int(moving_pop["n_movie_samples"]),
                    "n_images_contributing": int(moving_pop["n_images_contributing"]),
                    "moving_population_ssi_bits_per_spike": moving["population_ssi_bits_per_spike"],
                    "cell_baseline_population_ssi_bits_per_spike": cell["population_ssi_bits_per_spike"],
                    "ssi_percent_vs_cell_baseline": _pct_delta(
                        moving["population_ssi_bits_per_spike"],
                        cell["population_ssi_bits_per_spike"],
                    ),
                    "moving_information_bits_per_sample": moving["information_bits_per_sample"],
                    "cell_baseline_information_bits_per_sample": cell["information_bits_per_sample"],
                    "information_percent_vs_cell_baseline": _pct_delta(
                        moving["information_bits_per_sample"],
                        cell["information_bits_per_sample"],
                    ),
                    "moving_expected_spikes_per_sample": moving["expected_spikes_per_sample"],
                    "cell_baseline_expected_spikes_per_sample": cell["expected_spikes_per_sample"],
                    "spikes_percent_vs_cell_baseline": _pct_delta(
                        moving["expected_spikes_per_sample"],
                        cell["expected_spikes_per_sample"],
                    ),
                }
            )

    sf = pd.to_numeric(data["unit"][SF_METRIC_COL], errors="coerce")
    strong = threshold_data["image"]["image_contour_strong"].astype(bool)
    metadata = {
        "sf_min_cpd": float(sf_min_cpd),
        "sf_metric_col": SF_METRIC_COL,
        "contour_coherence_min": float(CONTOUR_COHERENCE_MIN),
        "n_sf_candidate_units": int(np.count_nonzero(sf >= float(sf_min_cpd))),
        "n_contour_images": int(strong.sum()),
        "n_selected_units": int(len(unit_to_images)),
        "n_selected_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
    }
    surface = pd.DataFrame(rows)
    for key, value in metadata.items():
        if key not in surface.columns:
            surface[key] = value
    return surface, metadata


def _compute_sweep(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metrics = _compute_component_metrics(data)
    drift = metrics[metrics["context"].astype(str).eq("drift_only")]
    across_edges = _quantile_edges(drift["across_path_arcmin"].to_numpy(dtype=float), N_BINS)
    along_edges = _quantile_edges(drift["along_path_arcmin"].to_numpy(dtype=float), N_BINS)

    surfaces: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {
        "across_edges_arcmin": across_edges,
        "along_edges_arcmin": along_edges,
        "sf_thresholds": SF_MIN_CPDS,
        "contour_coherence_min": CONTOUR_COHERENCE_MIN,
        "n_bins": N_BINS,
        "thresholds": {},
    }
    for sf_min in SF_MIN_CPDS:
        surface, threshold_metadata = _surface_values(
            data,
            metrics,
            sf_min_cpd=float(sf_min),
            across_edges=across_edges,
            along_edges=along_edges,
        )
        surfaces.append(surface)
        metadata["thresholds"][f"{float(sf_min):.3g}"] = threshold_metadata

    summary = pd.concat(surfaces, ignore_index=True)
    return summary, _conditional_summary(summary), metadata


def _grid(surface: pd.DataFrame, value_col: str) -> np.ndarray:
    arr = np.full((N_BINS, N_BINS), np.nan, dtype=float)
    for row in surface.itertuples(index=False):
        arr[int(row.across_bin) - 1, int(row.along_bin) - 1] = float(getattr(row, value_col))
    return arr


def _labels(surface: pd.DataFrame, axis: str) -> list[str]:
    labels: list[str] = []
    for idx in range(1, N_BINS + 1):
        if axis == "across":
            values = surface[surface["across_bin"].eq(idx)]["across_median_arcmin"].to_numpy(dtype=float)
        else:
            values = surface[surface["along_bin"].eq(idx)]["along_median_arcmin"].to_numpy(dtype=float)
        labels.append(f"Q{idx}\n{float(np.nanmedian(values)):.0f}")
    return labels


def _common_limits(summary: pd.DataFrame, value_col: str, *, floor: float = 2.5) -> tuple[float, float]:
    values = pd.to_numeric(summary[value_col], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -floor, floor
    lo = float(np.nanpercentile(values, 2.0))
    hi = float(np.nanpercentile(values, 98.0))
    span = max(abs(lo), abs(hi), floor)
    return -span, span


def _annotate_cells(ax: plt.Axes, values: np.ndarray, *, vmin: float, vmax: float) -> None:
    threshold = 0.55 * max(abs(vmin), abs(vmax))
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if math.isfinite(value):
                ax.text(
                    col,
                    row,
                    f"{value:+.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="white" if abs(value) >= threshold else "0.18",
                )


def _plot_outcome_surfaces(summary: pd.DataFrame, value_col: str, title: str, color_label: str) -> plt.Figure:
    fig, axes = plt.subplots(1, len(SF_MIN_CPDS), figsize=(22.0, 5.1), constrained_layout=False)
    fig.suptitle(
        f"{title} across high-SF thresholds, aligned units on contours",
        fontsize=13.3,
        y=0.985,
    )
    axes_arr = np.atleast_1d(axes)
    vmin, vmax = _common_limits(summary, value_col)
    image = None
    for ax, sf_min in zip(axes_arr, SF_MIN_CPDS, strict=True):
        surface = summary[summary["sf_min_cpd"].eq(float(sf_min))].copy()
        values = _grid(surface, value_col)
        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
        )
        n_units = int(surface["n_selected_units"].iloc[0])
        n_pairs = int(surface["n_selected_unit_image_pairs"].iloc[0])
        n_candidates = int(surface["n_sf_candidate_units"].iloc[0])
        ax.set_title(f"SF >= {sf_min:.3g}\n{n_units}/{n_candidates} units, {n_pairs} pairs", fontsize=9.0)
        ax.set_xticks(np.arange(N_BINS), _labels(surface, "along"))
        ax.set_yticks(np.arange(N_BINS), _labels(surface, "across") if ax is axes_arr[0] else [])
        ax.set_xlabel("along path bin; median arcmin", fontsize=7.7)
        if ax is axes_arr[0]:
            ax.set_ylabel("across path bin; median arcmin", fontsize=7.7)
        ax.tick_params(labelsize=6.2)
        _annotate_cells(ax, values, vmin=vmin, vmax=vmax)
        ax.spines[["top", "right"]].set_visible(False)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes_arr.tolist(), fraction=0.015, pad=0.016)
        cbar.set_label(color_label, fontsize=7.0)
        cbar.ax.tick_params(labelsize=6.7)
    fig.text(
        0.5,
        0.025,
        (
            f"Rows/columns use fixed drift-bank quantile edges for all SF thresholds; contour coherence >= "
            f"{CONTOUR_COHERENCE_MIN:.2f}. Values are moving-vs-cell-stabilized residuals."
        ),
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="0.30",
    )
    fig.subplots_adjust(left=0.045, right=0.895, top=0.79, bottom=0.19, wspace=0.23)
    return fig


def _plot_counts(summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, len(SF_MIN_CPDS), figsize=(22.0, 5.0), constrained_layout=False)
    fig.suptitle("Selected sample counts across high-SF thresholds", fontsize=13.2, y=0.985)
    axes_arr = np.atleast_1d(axes)
    counts_all = np.log10(np.maximum(pd.to_numeric(summary["n_movie_samples"], errors="coerce").to_numpy(dtype=float), 1.0))
    vmin = float(np.nanmin(counts_all))
    vmax = float(np.nanmax(counts_all))
    image = None
    for ax, sf_min in zip(axes_arr, SF_MIN_CPDS, strict=True):
        surface = summary[summary["sf_min_cpd"].eq(float(sf_min))]
        counts = _grid(surface, "n_movie_samples")
        image = ax.imshow(np.log10(np.maximum(counts, 1.0)), origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"SF >= {sf_min:.3g}", fontsize=9.0)
        ax.set_xticks(np.arange(N_BINS), _labels(surface, "along"))
        ax.set_yticks(np.arange(N_BINS), _labels(surface, "across") if ax is axes_arr[0] else [])
        ax.set_xlabel("along path bin; median arcmin", fontsize=7.7)
        if ax is axes_arr[0]:
            ax.set_ylabel("across path bin; median arcmin", fontsize=7.7)
        ax.tick_params(labelsize=6.2)
        median_count = float(np.nanmedian(counts))
        for row in range(counts.shape[0]):
            for col in range(counts.shape[1]):
                ax.text(
                    col,
                    row,
                    _format_count(counts[row, col]),
                    ha="center",
                    va="center",
                    fontsize=5.7,
                    color="white" if counts[row, col] >= median_count else "0.15",
                )
        ax.spines[["top", "right"]].set_visible(False)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes_arr.tolist(), fraction=0.015, pad=0.016)
        cbar.set_label("log10 selected samples", fontsize=7.0)
        cbar.ax.tick_params(labelsize=6.7)
    fig.subplots_adjust(left=0.045, right=0.895, top=0.79, bottom=0.17, wspace=0.23)
    return fig


def _block_mean(values: np.ndarray, row_slice: slice, col_slice: slice) -> float:
    block = values[row_slice, col_slice]
    finite = block[np.isfinite(block)]
    return float(np.nanmean(finite)) if finite.size else float("nan")


def _conditional_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    low = slice(0, 2)
    high = slice(N_BINS - 2, N_BINS)
    for sf_min, surface in summary.groupby("sf_min_cpd", sort=True):
        row: dict[str, Any] = {
            "sf_min_cpd": float(sf_min),
            "n_sf_candidate_units": int(surface["n_sf_candidate_units"].iloc[0]),
            "n_selected_units": int(surface["n_selected_units"].iloc[0]),
            "n_selected_unit_image_pairs": int(surface["n_selected_unit_image_pairs"].iloc[0]),
            "n_contour_images": int(surface["n_contour_images"].iloc[0]),
        }
        for value_col, _title, _label in OUTCOMES:
            key = value_col.replace("_percent_vs_cell_baseline", "")
            values = _grid(surface, value_col)
            across_effects = []
            along_effects = []
            for col in range(N_BINS):
                across_effects.append(float(np.nanmean(values[high, col]) - np.nanmean(values[low, col])))
            for row_idx in range(N_BINS):
                along_effects.append(float(np.nanmean(values[row_idx, high]) - np.nanmean(values[row_idx, low])))
            row[f"{key}_mean_across_high_minus_low_pct_points"] = float(np.nanmean(across_effects))
            row[f"{key}_mean_along_high_minus_low_pct_points"] = float(np.nanmean(along_effects))
            row[f"{key}_low_across_high_along_mean_pct"] = _block_mean(values, low, high)
            row[f"{key}_high_across_low_along_mean_pct"] = _block_mean(values, high, low)
            row[f"{key}_low_across_low_along_mean_pct"] = _block_mean(values, low, low)
            row[f"{key}_high_across_high_along_mean_pct"] = _block_mean(values, high, high)
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_conditional_contrasts(conditional: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), constrained_layout=False)
    x = conditional["sf_min_cpd"].to_numpy(dtype=float)
    ax = axes[0]
    ax.plot(
        x,
        conditional["ssi_mean_across_high_minus_low_pct_points"],
        marker="o",
        linewidth=1.9,
        label="high-low across, along held",
    )
    ax.plot(
        x,
        conditional["ssi_mean_along_high_minus_low_pct_points"],
        marker="s",
        linewidth=1.9,
        label="high-low along, across held",
    )
    ax.axhline(0.0, color="0.40", linewidth=0.9)
    ax.set_xlabel("SF threshold (cpd)")
    ax.set_ylabel("SSI conditional contrast\n(percentage points)")
    ax.set_title("Conditional surface contrasts", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.0)
    ax.grid(True, color="0.90", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.plot(x, conditional["n_selected_units"], marker="o", linewidth=1.8, label="units")
    ax2 = ax.twinx()
    ax2.plot(x, conditional["n_selected_unit_image_pairs"], marker="s", linewidth=1.8, color="#D55E00", label="unit-image pairs")
    ax.set_xlabel("SF threshold (cpd)")
    ax.set_ylabel("selected units")
    ax2.set_ylabel("selected unit-image pairs")
    ax.set_title("Selection size", fontsize=10.5)
    ax.grid(True, color="0.90", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, frameon=False, fontsize=8.0, loc="best")

    fig.suptitle("How the across-vs-along surface changes with SF threshold", fontsize=13.0, y=0.98)
    fig.text(
        0.5,
        0.02,
        "High-low contrasts compare the mean of Q7-Q8 to Q1-Q2 along one axis while averaging over bins on the other axis.",
        ha="center",
        fontsize=8.0,
        color="0.30",
    )
    fig.subplots_adjust(left=0.09, right=0.91, top=0.80, bottom=0.20, wspace=0.34)
    return fig


def main() -> None:
    data = load_dataset(MATRIX_DIR)
    summary, conditional, metadata = _compute_sweep(data)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{OUT_STEM}_values.csv"
    conditional_csv_path = OUT_DIR / f"{OUT_STEM}_conditional_summary.csv"
    json_path = OUT_DIR / f"{OUT_STEM}_summary.json"
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    count_png_path = OUT_DIR / f"{OUT_STEM}_counts.png"
    contrast_png_path = OUT_DIR / f"{OUT_STEM}_conditional_contrasts.png"
    outcome_pngs: list[Path] = []

    summary.to_csv(csv_path, index=False)
    conditional.to_csv(conditional_csv_path, index=False)
    with PdfPages(pdf_path) as pages:
        for value_col, title, color_label in OUTCOMES:
            fig = _plot_outcome_surfaces(summary, value_col, title, color_label)
            pages.savefig(fig, bbox_inches="tight")
            png_path = OUT_DIR / f"{OUT_STEM}_{value_col.replace('_percent_vs_cell_baseline', '')}.png"
            fig.savefig(png_path, dpi=240, bbox_inches="tight")
            outcome_pngs.append(png_path)
            plt.close(fig)
        fig = _plot_counts(summary)
        pages.savefig(fig, bbox_inches="tight")
        fig.savefig(count_png_path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        fig = _plot_conditional_contrasts(conditional)
        pages.savefig(fig, bbox_inches="tight")
        fig.savefig(contrast_png_path, dpi=240, bbox_inches="tight")
        plt.close(fig)

    _write_json(
        json_path,
        {
            "analysis": "backimage_real_trace_component_path_sf_threshold_surface_sweep",
            "matrix_dir": MATRIX_DIR,
            "out_dir": OUT_DIR,
            "outputs": {
                "pdf": pdf_path,
                "csv": csv_path,
                "conditional_summary_csv": conditional_csv_path,
                "summary_json": json_path,
                "outcome_pngs": outcome_pngs,
                "count_png": count_png_path,
                "conditional_contrasts_png": contrast_png_path,
            },
            "metadata": metadata,
            "note": (
                "SF threshold is applied by rewriting unit.sf_group to high_sf before the standard contour_matched "
                "unit-image selection. Contour gate is coherence >= 0.20. Surfaces use drift-only rows and fixed "
                "global component-path quantile edges."
            ),
        },
    )

    print(pdf_path)
    for path in outcome_pngs:
        print(path)
    print(count_png_path)
    print(contrast_png_path)
    print(csv_path)
    print(conditional_csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
