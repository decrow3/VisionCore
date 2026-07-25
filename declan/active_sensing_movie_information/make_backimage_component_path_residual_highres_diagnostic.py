#!/usr/bin/env python3
"""High-resolution residual-only component path diagnostic."""

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
    ORTHOGONAL_MIN_DEG,
    OUT_DIR,
    SF_GROUP,
    _assign_bins,
    _compute_component_metrics,
    _finite_ratio,
    _format_count,
    _json_ready,
    _pct_delta,
    _quantile_edges,
)
from declan.active_sensing_movie_information.make_backimage_component_path_baseline_decomposition_surface import (
    MATCH_MAX_DEG,
    RELATION,
    RELATION_LABEL,
    _cell_matched_baseline,
    _population_values,
    _scaled_colormap,
)
from declan.active_sensing_movie_information.plot_backimage_real_trace_unit_first_and_population_schematics import (
    accumulate_population,
    accumulate_population_movie_rows,
    baseline_rows_by_image,
    build_movie_row_grid,
    load_dataset,
    unit_image_selection,
)


OUT_STEM = "backimage_real_trace_component_path_residual_highres_n16"
N_BINS_HIGH = 16
EPS = 1e-12


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compute_highres(data: dict[str, Any], metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    row_image_index = metrics["image_index"].astype(int).to_numpy()

    global_baseline = accumulate_population(
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
    global_values = _population_values(global_baseline)

    drift_global = metrics[metrics["context"].astype(str).eq("drift_only")]
    across_edges = _quantile_edges(drift_global["across_path_arcmin"].to_numpy(dtype=float), N_BINS_HIGH)
    along_edges = _quantile_edges(drift_global["along_path_arcmin"].to_numpy(dtype=float), N_BINS_HIGH)
    across_bins = _assign_bins(metrics["across_path_arcmin"].to_numpy(dtype=float), across_edges)
    along_bins = _assign_bins(metrics["along_path_arcmin"].to_numpy(dtype=float), along_edges)
    drift_mask = metrics["context"].astype(str).eq("drift_only").to_numpy(dtype=bool)

    rows: list[dict[str, Any]] = []
    for across_bin in range(N_BINS_HIGH):
        for along_bin in range(N_BINS_HIGH):
            row_mask = drift_mask & (across_bins == across_bin) & (along_bins == along_bin)
            moving_pop = accumulate_population_movie_rows(
                ssi=data["ssi"],
                expected=data["expected"],
                row_image_index=row_image_index,
                row_mask=row_mask,
                unit_to_images=unit_to_images,
                n_images=n_images,
            )
            cell_pop = _cell_matched_baseline(
                stabilized_ssi=data["stabilized_ssi"],
                stabilized_expected=data["stabilized_expected"],
                row_image_index=row_image_index,
                row_mask=row_mask,
                baseline_lookup=baseline_lookup,
                unit_to_images=unit_to_images,
                n_images=n_images,
            )
            moving = _population_values(moving_pop)
            cell = _population_values(cell_pop)
            global_rows = metrics[row_mask]
            moving_ssi = moving["population_ssi_bits_per_spike"]
            cell_ssi = cell["population_ssi_bits_per_spike"]
            moving_info = moving["information_bits_per_sample"]
            cell_info = cell["information_bits_per_sample"]
            moving_spikes = moving["expected_spikes_per_sample"]
            cell_spikes = cell["expected_spikes_per_sample"]
            rows.append(
                {
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
                    "moving_population_ssi_bits_per_spike": moving_ssi,
                    "cell_baseline_population_ssi_bits_per_spike": cell_ssi,
                    "ssi_motion_percent_vs_cell_baseline": _pct_delta(moving_ssi, cell_ssi),
                    "moving_information_bits_per_sample": moving_info,
                    "cell_baseline_information_bits_per_sample": cell_info,
                    "information_motion_percent_vs_cell_baseline": _pct_delta(moving_info, cell_info),
                    "moving_expected_spikes_per_sample": moving_spikes,
                    "cell_baseline_expected_spikes_per_sample": cell_spikes,
                    "spikes_motion_percent_vs_cell_baseline": _pct_delta(moving_spikes, cell_spikes),
                    "global_baseline_population_ssi_bits_per_spike": global_values["population_ssi_bits_per_spike"],
                    "global_baseline_information_bits_per_sample": global_values["information_bits_per_sample"],
                    "global_baseline_expected_spikes_per_sample": global_values["expected_spikes_per_sample"],
                }
            )

    metadata = {
        "n_units": int(len(unit_to_images)),
        "n_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
        "global_baseline": global_values,
        "across_edges_arcmin": across_edges,
        "along_edges_arcmin": along_edges,
    }
    return pd.DataFrame(rows), metadata


def _grid(surface: pd.DataFrame, value_col: str) -> np.ndarray:
    arr = np.full((N_BINS_HIGH, N_BINS_HIGH), np.nan, dtype=float)
    for row in surface.itertuples(index=False):
        arr[int(row.across_bin) - 1, int(row.along_bin) - 1] = float(getattr(row, value_col))
    return arr


def _labels(surface: pd.DataFrame, axis: str) -> list[str]:
    labels = []
    for idx in range(1, N_BINS_HIGH + 1):
        if axis == "across":
            values = surface[surface["across_bin"].eq(idx)]["across_median_arcmin"].to_numpy(dtype=float)
        else:
            values = surface[surface["along_bin"].eq(idx)]["along_median_arcmin"].to_numpy(dtype=float)
        labels.append(f"Q{idx}\n{float(np.nanmedian(values)):.0f}")
    return labels


def _annotate_sparse(ax: plt.Axes, values: np.ndarray, *, vmin: float, vmax: float) -> None:
    threshold = 0.55 * max(abs(vmin), abs(vmax))
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if math.isfinite(value):
                ax.text(
                    x,
                    y,
                    f"{value:+.0f}",
                    ha="center",
                    va="center",
                    fontsize=4.8,
                    color="white" if abs(value) >= threshold else "0.18",
                )


def _plot_surface(surface: pd.DataFrame) -> plt.Figure:
    panels = [
        ("ssi_motion_percent_vs_cell_baseline", "SSI residual\n% vs cell baseline"),
        ("information_motion_percent_vs_cell_baseline", "Information residual\n% vs cell baseline"),
        ("spikes_motion_percent_vs_cell_baseline", "Expected-spike residual\n% vs cell baseline"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16.4, 5.2), constrained_layout=False)
    fig.suptitle("16x16 component path residual surface: moving vs cell-stabilized baseline", fontsize=13.2, y=0.985)
    xlabels = _labels(surface, "along")
    ylabels = _labels(surface, "across")
    for idx, (value_col, title) in enumerate(panels):
        ax = axes[idx]
        values = _grid(surface, value_col)
        kwargs, vmin, vmax = _scaled_colormap(values)
        image = ax.imshow(values, origin="lower", aspect="auto", **kwargs)
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks(np.arange(N_BINS_HIGH), xlabels)
        ax.set_yticks(np.arange(N_BINS_HIGH), ylabels)
        ax.set_xlabel("along bin; median arcmin", fontsize=8.0)
        if idx == 0:
            ax.set_ylabel("across bin; median arcmin", fontsize=8.0)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=5.7)
        _annotate_sparse(ax, values, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(labelsize=6.2)

    count_ax = axes[3]
    counts = _grid(surface, "n_movie_samples")
    count_image = count_ax.imshow(np.log10(np.maximum(counts, 1.0)), origin="lower", cmap="viridis", aspect="auto")
    count_ax.set_title("Selected samples", fontsize=9.5)
    count_ax.set_xticks(np.arange(N_BINS_HIGH), xlabels)
    count_ax.set_yticks(np.arange(N_BINS_HIGH), [])
    count_ax.set_xlabel("along bin; median arcmin", fontsize=8.0)
    count_ax.tick_params(labelsize=5.7)
    median_count = float(np.nanmedian(counts))
    for y in range(counts.shape[0]):
        for x in range(counts.shape[1]):
            count_ax.text(
                x,
                y,
                _format_count(counts[y, x]),
                ha="center",
                va="center",
                fontsize=4.6,
                color="white" if counts[y, x] >= median_count else "0.15",
            )
    cbar = fig.colorbar(count_image, ax=count_ax, fraction=0.04, pad=0.02)
    cbar.set_label("log10 samples", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6.2)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.02,
        "Higher-resolution bins test whether the low-across/high-along SSI residual lobe is hidden inside the wide 8-bin tail.",
        ha="center",
        fontsize=8.0,
        color="0.30",
    )
    fig.subplots_adjust(left=0.055, right=0.992, top=0.82, bottom=0.18, wspace=0.28)
    return fig


def _plot_low_across_profile(surface: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=False)
    colors = {
        "ssi_motion_percent_vs_cell_baseline": "#333333",
        "information_motion_percent_vs_cell_baseline": "#C44E52",
        "spikes_motion_percent_vs_cell_baseline": "#4C72B0",
    }
    labels = {
        "ssi_motion_percent_vs_cell_baseline": "SSI",
        "information_motion_percent_vs_cell_baseline": "information",
        "spikes_motion_percent_vs_cell_baseline": "spikes",
    }
    for ax_idx, across_bin in enumerate([1, 2]):
        ax = axes[ax_idx]
        row = surface[surface["across_bin"].eq(across_bin)].sort_values("along_bin")
        x = row["along_median_arcmin"].to_numpy(dtype=float)
        for col, color in colors.items():
            ax.plot(x, row[col].to_numpy(dtype=float), marker="o", linewidth=1.7, markersize=4.2, color=color, label=labels[col])
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.set_title(
            f"Across Q{across_bin}: median {float(np.nanmedian(row['across_median_arcmin'])):.1f} arcmin",
            fontsize=10.0,
        )
        ax.set_xlabel("along-contour component path median arcmin")
        ax.set_ylabel("% vs cell-stabilized baseline" if ax_idx == 0 else "")
        ax.tick_params(labelsize=8.0)
        ax.spines[["top", "right"]].set_visible(False)
        if ax_idx == 0:
            ax.legend(frameon=False, fontsize=8.0, loc="best")
        for x_val, y_val, count in zip(x, row["ssi_motion_percent_vs_cell_baseline"], row["n_movie_samples"], strict=True):
            ax.text(float(x_val), float(y_val), _format_count(float(count)), ha="center", va="bottom", fontsize=6.0, color="0.35")
    fig.suptitle("Low-across rows: residual motion profile across along-path bins", fontsize=13.0, y=0.98)
    fig.text(0.5, 0.02, "Numbers near SSI points are selected sample counts.", ha="center", fontsize=8.0, color="0.30")
    fig.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.17, wspace=0.22)
    return fig


def _tail_extract(surface: pd.DataFrame) -> pd.DataFrame:
    keep = surface["across_bin"].isin([1, 2]) & surface["along_bin"].ge(10)
    cols = [
        "across_bin",
        "along_bin",
        "across_median_arcmin",
        "along_min_arcmin",
        "along_max_arcmin",
        "along_median_arcmin",
        "ssi_motion_percent_vs_cell_baseline",
        "information_motion_percent_vs_cell_baseline",
        "spikes_motion_percent_vs_cell_baseline",
        "n_movie_samples",
        "n_images_contributing",
    ]
    return surface.loc[keep, cols].sort_values(["across_bin", "along_bin"]).reset_index(drop=True)


def main() -> None:
    data = load_dataset(MATRIX_DIR)
    metrics = _compute_component_metrics(data)
    summary, metadata = _compute_highres(data, metrics)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / f"{OUT_STEM}_values.csv"
    tail_csv = OUT_DIR / f"{OUT_STEM}_low_across_tail_extract.csv"
    json_path = OUT_DIR / f"{OUT_STEM}_summary.json"
    pdf = OUT_DIR / f"{OUT_STEM}.pdf"
    surface_png = OUT_DIR / f"{OUT_STEM}_surface.png"
    profile_png = OUT_DIR / f"{OUT_STEM}_low_across_profile.png"
    summary.to_csv(csv, index=False)
    tail = _tail_extract(summary)
    tail.to_csv(tail_csv, index=False)
    with PdfPages(pdf) as pages:
        fig = _plot_surface(summary)
        pages.savefig(fig, bbox_inches="tight")
        fig.savefig(surface_png, dpi=240, bbox_inches="tight")
        plt.close(fig)
        fig = _plot_low_across_profile(summary)
        pages.savefig(fig, bbox_inches="tight")
        fig.savefig(profile_png, dpi=240, bbox_inches="tight")
        plt.close(fig)
    _write_json(
        json_path,
        {
            "analysis": "backimage_real_trace_component_path_residual_highres_diagnostic",
            "matrix_dir": MATRIX_DIR,
            "out_dir": OUT_DIR,
            "outputs": {
                "pdf": pdf,
                "surface_png": surface_png,
                "profile_png": profile_png,
                "csv": csv,
                "low_across_tail_extract_csv": tail_csv,
                "summary_json": json_path,
            },
            "selection": {
                "relation": RELATION,
                "relation_label": RELATION_LABEL,
                "sf_group": SF_GROUP,
                "min_osi": MIN_OSI,
                "match_max_deg": MATCH_MAX_DEG,
                **metadata,
            },
            "n_bins": N_BINS_HIGH,
            "note": "Residuals compare moving real-trace cells to cell-matched stabilized unit-image composition.",
        },
    )
    print(pdf)
    print(surface_png)
    print(profile_png)
    print(csv)
    print(tail_csv)
    print(json_path)


if __name__ == "__main__":
    main()
