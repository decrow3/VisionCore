#!/usr/bin/env python3
"""High-resolution component path residual surfaces across contour thresholds."""

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
    _compute_component_metrics,
    _format_count,
    _json_ready,
)
from declan.active_sensing_movie_information.make_backimage_component_path_baseline_decomposition_surface import (
    MATCH_MAX_DEG,
)
from declan.active_sensing_movie_information.make_backimage_component_path_residual_highres_diagnostic import (
    N_BINS_HIGH,
    _compute_highres,
    _grid,
    _labels,
)
from declan.active_sensing_movie_information.plot_backimage_real_trace_unit_first_and_population_schematics import (
    load_dataset,
)


OUT_STEM = "backimage_real_trace_component_path_coherence_sweep_residual_highres_n16"
CONTOUR_COHERENCE_MINS = (0.20, 0.35, 0.50, 0.65)
LOW_ACROSS_BINS = (1, 2)
HIGH_ALONG_MIN_BIN = 9
TAIL_ALONG_MIN_BIN = 13

OUTCOMES = [
    (
        "ssi_motion_percent_vs_cell_baseline",
        "SSI bits/spike residual",
        "% vs cell-stabilized baseline",
    ),
    (
        "information_motion_percent_vs_cell_baseline",
        "Information residual",
        "% vs cell-stabilized baseline",
    ),
    (
        "spikes_motion_percent_vs_cell_baseline",
        "Expected-spike residual",
        "% vs cell-stabilized baseline",
    ),
]


def _slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _thresholded_data(data: dict[str, Any], threshold: float) -> dict[str, Any]:
    out = dict(data)
    image = data["image"].copy()
    coherence = pd.to_numeric(image["image_orientation_coherence"], errors="coerce").to_numpy(dtype=float)
    axis = pd.to_numeric(image["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=float)
    image["image_contour_strong"] = np.isfinite(axis) & np.isfinite(coherence) & (coherence >= float(threshold))
    out["image"] = image
    return out


def _compute_sweep(data: dict[str, Any], metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {}
    for threshold in CONTOUR_COHERENCE_MINS:
        threshold_data = _thresholded_data(data, threshold)
        surface, surface_metadata = _compute_highres(threshold_data, metrics)
        coherence = pd.to_numeric(threshold_data["image"]["image_orientation_coherence"], errors="coerce")
        strong = threshold_data["image"]["image_contour_strong"].astype(bool)
        surface = surface.copy()
        surface.insert(0, "contour_coherence_min", float(threshold))
        surface.insert(1, "match_max_deg", float(MATCH_MAX_DEG))
        surface.insert(2, "n_contour_images", int(strong.sum()))
        surface.insert(3, "n_selected_units", int(surface_metadata["n_units"]))
        surface.insert(4, "n_selected_unit_image_pairs", int(surface_metadata["n_unit_image_pairs"]))
        rows.append(surface)
        metadata[_slug(threshold)] = {
            **surface_metadata,
            "contour_coherence_min": float(threshold),
            "n_contour_images": int(strong.sum()),
            "contour_coherence_min_observed": float(coherence[strong].min()) if bool(strong.any()) else float("nan"),
            "contour_coherence_median_observed": float(coherence[strong].median()) if bool(strong.any()) else float("nan"),
            "contour_coherence_max_observed": float(coherence[strong].max()) if bool(strong.any()) else float("nan"),
        }

    summary = pd.concat(rows, ignore_index=True)
    return summary, _lobe_summary(summary), metadata


def _lobe_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold, frame in summary.groupby("contour_coherence_min", sort=True):
        low_across_high_along = frame[
            frame["across_bin"].isin(LOW_ACROSS_BINS) & frame["along_bin"].ge(HIGH_ALONG_MIN_BIN)
        ].copy()
        low_across_tail = frame[
            frame["across_bin"].isin(LOW_ACROSS_BINS) & frame["along_bin"].ge(TAIL_ALONG_MIN_BIN)
        ].copy()
        extreme = frame[frame["across_bin"].eq(1) & frame["along_bin"].eq(N_BINS_HIGH)]
        peak = low_across_high_along.sort_values("ssi_motion_percent_vs_cell_baseline", ascending=False).head(1)
        peak_row = peak.iloc[0] if not peak.empty else None
        extreme_row = extreme.iloc[0] if not extreme.empty else None
        rows.append(
            {
                "contour_coherence_min": float(threshold),
                "match_max_deg": float(frame["match_max_deg"].iloc[0]),
                "n_contour_images": int(frame["n_contour_images"].iloc[0]),
                "n_selected_units": int(frame["n_selected_units"].iloc[0]),
                "n_selected_unit_image_pairs": int(frame["n_selected_unit_image_pairs"].iloc[0]),
                "low_across_high_along_peak_across_bin": int(peak_row["across_bin"]) if peak_row is not None else None,
                "low_across_high_along_peak_along_bin": int(peak_row["along_bin"]) if peak_row is not None else None,
                "low_across_high_along_peak_across_median_arcmin": (
                    float(peak_row["across_median_arcmin"]) if peak_row is not None else float("nan")
                ),
                "low_across_high_along_peak_along_median_arcmin": (
                    float(peak_row["along_median_arcmin"]) if peak_row is not None else float("nan")
                ),
                "low_across_high_along_peak_ssi_percent_vs_cell": (
                    float(peak_row["ssi_motion_percent_vs_cell_baseline"]) if peak_row is not None else float("nan")
                ),
                "low_across_high_along_peak_info_percent_vs_cell": (
                    float(peak_row["information_motion_percent_vs_cell_baseline"]) if peak_row is not None else float("nan")
                ),
                "low_across_high_along_peak_spikes_percent_vs_cell": (
                    float(peak_row["spikes_motion_percent_vs_cell_baseline"]) if peak_row is not None else float("nan")
                ),
                "tail_q13_q16_mean_ssi_percent_vs_cell": float(
                    low_across_tail["ssi_motion_percent_vs_cell_baseline"].mean()
                ),
                "tail_q13_q16_max_ssi_percent_vs_cell": float(
                    low_across_tail["ssi_motion_percent_vs_cell_baseline"].max()
                ),
                "extreme_q1_across_q16_along_ssi_percent_vs_cell": (
                    float(extreme_row["ssi_motion_percent_vs_cell_baseline"]) if extreme_row is not None else float("nan")
                ),
                "extreme_q1_across_q16_along_n_movie_samples": (
                    int(extreme_row["n_movie_samples"]) if extreme_row is not None else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def _common_limits(summary: pd.DataFrame, value_col: str) -> tuple[float, float]:
    finite = pd.to_numeric(summary[value_col], errors="coerce").to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.nanpercentile(finite, 2))
    hi = float(np.nanpercentile(finite, 98))
    span = max(abs(lo), abs(hi), 1e-6)
    return -span, span


def _imshow_kwargs(vmin: float, vmax: float) -> dict[str, Any]:
    return {"cmap": "RdBu_r", "norm": TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)}


def _plot_threshold_surfaces(summary: pd.DataFrame, value_col: str, title: str, color_label: str) -> plt.Figure:
    fig, axes = plt.subplots(1, len(CONTOUR_COHERENCE_MINS), figsize=(21.8, 5.25), constrained_layout=False)
    fig.suptitle(f"{title} by contour-coherence threshold", fontsize=13.0, y=0.985)
    axes_arr = np.atleast_1d(axes)
    vmin, vmax = _common_limits(summary, value_col)
    threshold_for_color = 0.55 * max(abs(vmin), abs(vmax))
    image = None
    for ax, threshold in zip(axes_arr, CONTOUR_COHERENCE_MINS, strict=True):
        surface = summary[summary["contour_coherence_min"].eq(float(threshold))]
        values = _grid(surface, value_col)
        image = ax.imshow(values, origin="lower", aspect="auto", **_imshow_kwargs(vmin, vmax))
        xlabels = _labels(surface, "along")
        ylabels = _labels(surface, "across")
        n_images = int(surface["n_contour_images"].iloc[0])
        n_pairs = int(surface["n_selected_unit_image_pairs"].iloc[0])
        ax.set_title(f"coherence >= {threshold:.2f}\n{n_images} images, {n_pairs} pairs", fontsize=9.5)
        ax.set_xticks(np.arange(N_BINS_HIGH), xlabels)
        ax.set_yticks(np.arange(N_BINS_HIGH), ylabels if ax is axes_arr[0] else [])
        ax.set_xlabel("along path bin; median arcmin", fontsize=8.0)
        if ax is axes_arr[0]:
            ax.set_ylabel("across path bin; median arcmin", fontsize=8.0)
        ax.tick_params(labelsize=5.8)
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
                        fontsize=4.3,
                        color="white" if abs(value) >= threshold_for_color else "0.18",
                    )
        ax.spines[["top", "right"]].set_visible(False)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes_arr.tolist(), fraction=0.016, pad=0.018)
        cbar.set_label(color_label, fontsize=7.0)
        cbar.ax.tick_params(labelsize=6.7)
    fig.text(
        0.5,
        0.025,
        (
            "Rows are across-contour component path bins; columns are along-contour component path bins. "
            "All panels use the same trace-bin edges and moving-vs-cell-stabilized residual."
        ),
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="0.30",
    )
    fig.subplots_adjust(left=0.045, right=0.890, top=0.80, bottom=0.18, wspace=0.22)
    return fig


def _plot_low_across_profile(summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, len(LOW_ACROSS_BINS), figsize=(12.6, 4.7), constrained_layout=False)
    colors = {0.35: "#777777", 0.50: "#222222", 0.65: "#D55E00"}
    for ax, across_bin in zip(np.atleast_1d(axes), LOW_ACROSS_BINS, strict=True):
        for threshold in CONTOUR_COHERENCE_MINS:
            frame = summary[
                summary["contour_coherence_min"].eq(float(threshold)) & summary["across_bin"].eq(int(across_bin))
            ].sort_values("along_bin")
            ax.plot(
                frame["along_median_arcmin"].to_numpy(dtype=float),
                frame["ssi_motion_percent_vs_cell_baseline"].to_numpy(dtype=float),
                marker="o",
                linewidth=1.6,
                markersize=4.0,
                color=colors.get(float(threshold), None),
                label=f"coh >= {threshold:.2f}",
            )
        ax.axhline(0.0, color="0.38", linewidth=0.8)
        ax.axvspan(
            float(
                summary[
                    summary["contour_coherence_min"].eq(float(CONTOUR_COHERENCE_MINS[0]))
                    & summary["along_bin"].eq(TAIL_ALONG_MIN_BIN)
                ]["along_min_arcmin"].median()
            ),
            float(
                summary[
                    summary["contour_coherence_min"].eq(float(CONTOUR_COHERENCE_MINS[0]))
                    & summary["along_bin"].eq(N_BINS_HIGH)
                ]["along_max_arcmin"].median()
            ),
            color="0.90",
            alpha=0.8,
            zorder=0,
        )
        across_median = float(
            summary[summary["across_bin"].eq(int(across_bin))]["across_median_arcmin"].median()
        )
        ax.set_title(f"Across Q{across_bin}: median {across_median:.1f} arcmin", fontsize=10.0)
        ax.set_xlabel("along-contour component path median arcmin")
        ax.set_ylabel("SSI residual, % vs cell baseline" if ax is np.atleast_1d(axes)[0] else "")
        ax.tick_params(labelsize=8.0)
        ax.spines[["top", "right"]].set_visible(False)
    axes_arr = np.atleast_1d(axes)
    axes_arr[0].legend(frameon=False, fontsize=8.0, loc="best")
    fig.suptitle("Low-across rows: does the bottom-right SSI lobe survive contour thresholding?", fontsize=13.0, y=0.98)
    fig.text(
        0.5,
        0.02,
        "Gray band marks the extreme high-along tail, where the coarse 8-bin bottom-right corner was dominated.",
        ha="center",
        fontsize=8.0,
        color="0.30",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.17, wspace=0.22)
    return fig


def _plot_counts(summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, len(CONTOUR_COHERENCE_MINS), figsize=(21.8, 5.05), constrained_layout=False)
    fig.suptitle("Selected sample counts by contour-coherence threshold", fontsize=13.0, y=0.985)
    axes_arr = np.atleast_1d(axes)
    all_counts = np.log10(np.maximum(pd.to_numeric(summary["n_movie_samples"], errors="coerce").to_numpy(dtype=float), 1.0))
    vmin = float(np.nanmin(all_counts))
    vmax = float(np.nanmax(all_counts))
    image = None
    for ax, threshold in zip(axes_arr, CONTOUR_COHERENCE_MINS, strict=True):
        surface = summary[summary["contour_coherence_min"].eq(float(threshold))]
        counts = _grid(surface, "n_movie_samples")
        image = ax.imshow(np.log10(np.maximum(counts, 1.0)), origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        xlabels = _labels(surface, "along")
        ylabels = _labels(surface, "across")
        ax.set_title(f"coherence >= {threshold:.2f}", fontsize=9.5)
        ax.set_xticks(np.arange(N_BINS_HIGH), xlabels)
        ax.set_yticks(np.arange(N_BINS_HIGH), ylabels if ax is axes_arr[0] else [])
        ax.set_xlabel("along path bin; median arcmin", fontsize=8.0)
        if ax is axes_arr[0]:
            ax.set_ylabel("across path bin; median arcmin", fontsize=8.0)
        ax.tick_params(labelsize=5.8)
        median_count = float(np.nanmedian(counts))
        for y in range(counts.shape[0]):
            for x in range(counts.shape[1]):
                ax.text(
                    x,
                    y,
                    _format_count(counts[y, x]),
                    ha="center",
                    va="center",
                    fontsize=4.3,
                    color="white" if counts[y, x] >= median_count else "0.15",
                )
        ax.spines[["top", "right"]].set_visible(False)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes_arr.tolist(), fraction=0.016, pad=0.018)
        cbar.set_label("log10 selected samples", fontsize=7.0)
        cbar.ax.tick_params(labelsize=6.7)
    fig.subplots_adjust(left=0.045, right=0.890, top=0.80, bottom=0.16, wspace=0.22)
    return fig


def main() -> None:
    data = load_dataset(MATRIX_DIR)
    metrics = _compute_component_metrics(data)
    summary, lobe, metadata = _compute_sweep(data, metrics)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{OUT_STEM}_values.csv"
    lobe_csv_path = OUT_DIR / f"{OUT_STEM}_lobe_summary.csv"
    json_path = OUT_DIR / f"{OUT_STEM}_summary.json"
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    profile_png = OUT_DIR / f"{OUT_STEM}_low_across_profile.png"
    count_png = OUT_DIR / f"{OUT_STEM}_counts.png"
    outcome_pngs: list[Path] = []

    summary.to_csv(csv_path, index=False)
    lobe.to_csv(lobe_csv_path, index=False)
    with PdfPages(pdf_path) as pages:
        for value_col, title, color_label in OUTCOMES:
            fig = _plot_threshold_surfaces(summary, value_col, title, color_label)
            pages.savefig(fig, bbox_inches="tight")
            png_path = OUT_DIR / f"{OUT_STEM}_{value_col.replace('_motion_percent_vs_cell_baseline', '')}.png"
            fig.savefig(png_path, dpi=240, bbox_inches="tight")
            outcome_pngs.append(png_path)
            plt.close(fig)
        fig = _plot_low_across_profile(summary)
        pages.savefig(fig, bbox_inches="tight")
        fig.savefig(profile_png, dpi=240, bbox_inches="tight")
        plt.close(fig)
        fig = _plot_counts(summary)
        pages.savefig(fig, bbox_inches="tight")
        fig.savefig(count_png, dpi=240, bbox_inches="tight")
        plt.close(fig)

    _write_json(
        json_path,
        {
            "analysis": "backimage_real_trace_component_path_coherence_sweep_residual_highres",
            "matrix_dir": MATRIX_DIR,
            "out_dir": OUT_DIR,
            "outputs": {
                "pdf": pdf_path,
                "csv": csv_path,
                "lobe_summary_csv": lobe_csv_path,
                "summary_json": json_path,
                "profile_png": profile_png,
                "count_png": count_png,
                "outcome_pngs": outcome_pngs,
            },
            "sweep": {
                "contour_coherence_mins": CONTOUR_COHERENCE_MINS,
                "n_bins": N_BINS_HIGH,
                "match_max_deg": MATCH_MAX_DEG,
                "low_across_bins_for_lobe_summary": LOW_ACROSS_BINS,
                "high_along_min_bin_for_lobe_summary": HIGH_ALONG_MIN_BIN,
                "tail_along_min_bin_for_lobe_summary": TAIL_ALONG_MIN_BIN,
            },
            "metadata": metadata,
            "note": "The only intended change across panels is the image_orientation_coherence threshold used to overwrite image_contour_strong before running the existing high-resolution residual computation.",
        },
    )

    print(pdf_path)
    for path in outcome_pngs:
        print(path)
    print(profile_png)
    print(count_png)
    print(csv_path)
    print(lobe_csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
