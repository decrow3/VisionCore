#!/usr/bin/env python3
"""Line-plot marginals from the high-resolution component path surfaces."""

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

from declan.active_sensing_movie_information.make_backimage_component_2d_surface_diagnostic import OUT_DIR, _json_ready
from declan.active_sensing_movie_information.make_backimage_component_path_coherence_sweep_residual_highres import (
    OUT_STEM as SURFACE_STEM,
)


OUT_STEM = "backimage_real_trace_component_path_axis_marginal_lineplots"
SURFACE_CSV = OUT_DIR / f"{SURFACE_STEM}_values.csv"
MIDDLE_BIN_MIN = 5
MIDDLE_BIN_MAX = 12

THRESHOLD_COLORS = {
    0.20: "#0072B2",
    0.35: "#777777",
    0.50: "#222222",
    0.65: "#D55E00",
}
AXIS_SPECS = [
    ("across", "Across-contour component path", "across component path length (arcmin)"),
    ("along", "Along-contour component path", "along component path length (arcmin)"),
]
OUTCOME_SPECS = [
    ("ssi", "SSI bits/spike", "SSI residual\n(% vs cell baseline)", "#222222"),
    ("information", "Information bits/window", "Information residual\n(% vs cell baseline)", "#C44E52"),
    ("spikes", "Expected spikes/window", "Expected-spike residual\n(% vs cell baseline)", "#4C72B0"),
]
EPS = 1e-12


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_ratio(num: float, den: float) -> float:
    return float(num / den) if math.isfinite(num) and math.isfinite(den) and den > EPS else float("nan")


def _pct_delta(value: float, baseline: float) -> float:
    if not (math.isfinite(value) and math.isfinite(baseline) and abs(baseline) > EPS):
        return float("nan")
    return 100.0 * (value - baseline) / baseline


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(arr) & np.isfinite(w) & (w > 0)
    if not bool(np.any(ok)):
        return float("nan")
    return float(np.average(arr[ok], weights=w[ok]))


def _weighted_slope_per_10_arcmin(frame: pd.DataFrame, value_col: str) -> float:
    x = pd.to_numeric(frame["component_median_arcmin"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype=float)
    w = np.sqrt(np.maximum(pd.to_numeric(frame["n_movie_samples"], errors="coerce").to_numpy(dtype=float), 0.0))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if np.count_nonzero(ok) < 2:
        return float("nan")
    slope = np.polyfit(x[ok], y[ok], deg=1, w=w[ok])[0]
    return float(10.0 * slope)


def _aggregate_axis(surface: pd.DataFrame, axis: str) -> pd.DataFrame:
    bin_col = f"{axis}_bin"
    min_col = f"{axis}_min_arcmin"
    max_col = f"{axis}_max_arcmin"
    median_col = f"{axis}_median_arcmin"
    rows: list[dict[str, Any]] = []

    for (threshold, bin_index), group in surface.groupby(["contour_coherence_min", bin_col], sort=True):
        n = pd.to_numeric(group["n_movie_samples"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        n_total = float(np.nansum(n))
        moving_info = float(np.nansum(pd.to_numeric(group["moving_information_bits_per_sample"], errors="coerce") * n))
        cell_info = float(np.nansum(pd.to_numeric(group["cell_baseline_information_bits_per_sample"], errors="coerce") * n))
        moving_spikes = float(np.nansum(pd.to_numeric(group["moving_expected_spikes_per_sample"], errors="coerce") * n))
        cell_spikes = float(np.nansum(pd.to_numeric(group["cell_baseline_expected_spikes_per_sample"], errors="coerce") * n))

        moving_ssi = _finite_ratio(moving_info, moving_spikes)
        cell_ssi = _finite_ratio(cell_info, cell_spikes)
        moving_info_per_sample = _finite_ratio(moving_info, n_total)
        cell_info_per_sample = _finite_ratio(cell_info, n_total)
        moving_spikes_per_sample = _finite_ratio(moving_spikes, n_total)
        cell_spikes_per_sample = _finite_ratio(cell_spikes, n_total)

        rows.append(
            {
                "contour_coherence_min": float(threshold),
                "component_axis": axis,
                "component_bin": int(bin_index),
                "component_min_arcmin": float(pd.to_numeric(group[min_col], errors="coerce").min()),
                "component_max_arcmin": float(pd.to_numeric(group[max_col], errors="coerce").max()),
                "component_median_arcmin": _weighted_mean(group[median_col], group["n_movie_samples"]),
                "n_movie_samples": int(round(n_total)),
                "n_contour_images": int(group["n_contour_images"].iloc[0]),
                "n_selected_units": int(group["n_selected_units"].iloc[0]),
                "n_selected_unit_image_pairs": int(group["n_selected_unit_image_pairs"].iloc[0]),
                "moving_population_ssi_bits_per_spike": moving_ssi,
                "cell_baseline_population_ssi_bits_per_spike": cell_ssi,
                "ssi_percent_vs_cell_baseline": _pct_delta(moving_ssi, cell_ssi),
                "moving_information_bits_per_sample": moving_info_per_sample,
                "cell_baseline_information_bits_per_sample": cell_info_per_sample,
                "information_percent_vs_cell_baseline": _pct_delta(moving_info_per_sample, cell_info_per_sample),
                "moving_expected_spikes_per_sample": moving_spikes_per_sample,
                "cell_baseline_expected_spikes_per_sample": cell_spikes_per_sample,
                "spikes_percent_vs_cell_baseline": _pct_delta(moving_spikes_per_sample, cell_spikes_per_sample),
            }
        )
    return pd.DataFrame(rows)


def _make_marginals(surface: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    marginal = pd.concat([_aggregate_axis(surface, axis) for axis, _title, _x in AXIS_SPECS], ignore_index=True)
    rows: list[dict[str, Any]] = []
    for (threshold, axis), group in marginal.groupby(["contour_coherence_min", "component_axis"], sort=True):
        row: dict[str, Any] = {
            "contour_coherence_min": float(threshold),
            "component_axis": str(axis),
            "n_contour_images": int(group["n_contour_images"].iloc[0]),
            "n_selected_unit_image_pairs": int(group["n_selected_unit_image_pairs"].iloc[0]),
        }
        for key, _title, _ylabel, _color in OUTCOME_SPECS:
            col = f"{key}_percent_vs_cell_baseline"
            finite = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            row[f"{key}_range_percent_points"] = float(np.nanmax(finite) - np.nanmin(finite)) if finite.size else float("nan")
            row[f"{key}_min_percent_vs_cell"] = float(np.nanmin(finite)) if finite.size else float("nan")
            row[f"{key}_max_percent_vs_cell"] = float(np.nanmax(finite)) if finite.size else float("nan")
            row[f"{key}_slope_percent_per_10_arcmin"] = _weighted_slope_per_10_arcmin(group, col)
        rows.append(row)
    return marginal, pd.DataFrame(rows)


def _middle_band(marginal: pd.DataFrame, axis: str) -> tuple[float, float] | None:
    frame = marginal[marginal["component_axis"].eq(axis)]
    middle = frame[frame["component_bin"].between(MIDDLE_BIN_MIN, MIDDLE_BIN_MAX)]
    if middle.empty:
        return None
    low = float(middle["component_min_arcmin"].min())
    high = float(middle["component_max_arcmin"].max())
    if not (math.isfinite(low) and math.isfinite(high) and high > low):
        return None
    return low, high


def _format_axes(ax: plt.Axes, *, axis: str, ylabel: str, xlabel: str, add_band_label: bool = False) -> None:
    band = _middle_band(_CURRENT_MARGINAL, axis)
    if band is not None:
        ax.axvspan(band[0], band[1], color="0.92", alpha=0.85, zorder=0)
        if add_band_label:
            ax.text(
                0.5 * (band[0] + band[1]),
                0.965,
                "middle 50% of component paths",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.2,
                color="0.35",
            )
    ax.axhline(0.0, color="0.35", linewidth=0.85)
    ax.grid(True, color="0.90", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(labelsize=8.0)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_focus(marginal: pd.DataFrame) -> plt.Figure:
    global _CURRENT_MARGINAL
    _CURRENT_MARGINAL = marginal
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), constrained_layout=False)
    fig.suptitle("Component path line marginals: SSI residual", fontsize=13.0, y=0.98)
    for ax, (axis, title, xlabel) in zip(axes, AXIS_SPECS, strict=True):
        frame = marginal[marginal["component_axis"].eq(axis)]
        for threshold, group in frame.groupby("contour_coherence_min", sort=True):
            group = group.sort_values("component_bin")
            ax.plot(
                group["component_median_arcmin"],
                group["ssi_percent_vs_cell_baseline"],
                marker="o",
                linewidth=1.8,
                markersize=4.2,
                color=THRESHOLD_COLORS.get(float(threshold), None),
                label=f"coh >= {float(threshold):.2f}",
            )
        ax.set_title(title, fontsize=10.5)
        _format_axes(
            ax,
            axis=axis,
            ylabel="SSI residual (% vs cell baseline)" if axis == "across" else "",
            xlabel=xlabel,
            add_band_label=axis == "across",
        )
    axes[0].legend(frameon=False, fontsize=8.2, loc="best")
    fig.text(
        0.5,
        0.025,
        "Marginals are reconstructed by summing information and expected spikes across the other 2D component axis.",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="0.30",
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.81, bottom=0.17, wspace=0.22)
    return fig


def _plot_outcomes(marginal: pd.DataFrame) -> plt.Figure:
    global _CURRENT_MARGINAL
    _CURRENT_MARGINAL = marginal
    fig, axes = plt.subplots(len(OUTCOME_SPECS), 2, figsize=(12.8, 9.2), constrained_layout=False)
    fig.suptitle("Component path line marginals from the 16x16 residual surface", fontsize=13.0, y=0.99)
    for row_idx, (key, title, ylabel, _color) in enumerate(OUTCOME_SPECS):
        value_col = f"{key}_percent_vs_cell_baseline"
        for col_idx, (axis, axis_title, xlabel) in enumerate(AXIS_SPECS):
            ax = axes[row_idx, col_idx]
            frame = marginal[marginal["component_axis"].eq(axis)]
            for threshold, group in frame.groupby("contour_coherence_min", sort=True):
                group = group.sort_values("component_bin")
                ax.plot(
                    group["component_median_arcmin"],
                    group[value_col],
                    marker="o",
                    linewidth=1.55,
                    markersize=3.8,
                    color=THRESHOLD_COLORS.get(float(threshold), None),
                    label=f"coh >= {float(threshold):.2f}",
                )
            if row_idx == 0:
                ax.set_title(axis_title, fontsize=10.2)
            _format_axes(
                ax,
                axis=axis,
                ylabel=ylabel if col_idx == 0 else "",
                xlabel=xlabel if row_idx == len(OUTCOME_SPECS) - 1 else "",
                add_band_label=row_idx == 0 and col_idx == 0,
            )
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=7.6, loc="best")
            ax.text(
                0.02,
                0.93,
                title,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.2,
                color="0.20",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.8},
            )
    fig.text(
        0.5,
        0.018,
        "Each point collapses the 2D cells along the unplotted component axis using information/spike totals.",
        ha="center",
        fontsize=8.0,
        color="0.30",
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.93, bottom=0.085, hspace=0.34, wspace=0.20)
    return fig


_CURRENT_MARGINAL = pd.DataFrame()


def main() -> None:
    if not SURFACE_CSV.exists():
        raise FileNotFoundError(f"Expected surface CSV at {SURFACE_CSV}")
    surface = pd.read_csv(SURFACE_CSV)
    marginal, axis_summary = _make_marginals(surface)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    values_csv = OUT_DIR / f"{OUT_STEM}_values.csv"
    axis_summary_csv = OUT_DIR / f"{OUT_STEM}_axis_summary.csv"
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    focus_png = OUT_DIR / f"{OUT_STEM}_ssi_focus.png"
    outcomes_png = OUT_DIR / f"{OUT_STEM}_outcomes.png"
    json_path = OUT_DIR / f"{OUT_STEM}_summary.json"

    marginal.to_csv(values_csv, index=False)
    axis_summary.to_csv(axis_summary_csv, index=False)
    with PdfPages(pdf_path) as pages:
        fig = _plot_focus(marginal)
        pages.savefig(fig, bbox_inches="tight")
        fig.savefig(focus_png, dpi=240, bbox_inches="tight")
        plt.close(fig)
        fig = _plot_outcomes(marginal)
        pages.savefig(fig, bbox_inches="tight")
        fig.savefig(outcomes_png, dpi=240, bbox_inches="tight")
        plt.close(fig)

    _write_json(
        json_path,
        {
            "analysis": "backimage_real_trace_component_path_axis_marginal_lineplots",
            "surface_csv": SURFACE_CSV,
            "out_dir": OUT_DIR,
            "outputs": {
                "pdf": pdf_path,
                "ssi_focus_png": focus_png,
                "outcomes_png": outcomes_png,
                "values_csv": values_csv,
                "axis_summary_csv": axis_summary_csv,
                "summary_json": json_path,
            },
            "contract": {
                "marginalization": "For each component axis bin, sums moving/cell information and expected-spike totals across all bins of the other component axis, then recomputes SSI.",
                "baseline": "Residuals remain moving real traces vs cell-matched stabilized unit-image composition.",
                "middle_path_band": f"Gray band spans bins {MIDDLE_BIN_MIN}-{MIDDLE_BIN_MAX}, the middle 50% of the component-path quantile bins.",
            },
        },
    )
    print(pdf_path)
    print(focus_png)
    print(outcomes_png)
    print(values_csv)
    print(axis_summary_csv)
    print(json_path)


if __name__ == "__main__":
    main()
