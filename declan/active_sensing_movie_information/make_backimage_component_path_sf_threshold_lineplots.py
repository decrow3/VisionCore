#!/usr/bin/env python3
"""Component path line marginals across stricter high-SF unit thresholds."""

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
from matplotlib.lines import Line2D

from declan.active_sensing_movie_information.make_backimage_component_2d_surface_diagnostic import (
    MATRIX_DIR,
    OUT_DIR,
    _compute_component_metrics,
    _json_ready,
)
from declan.active_sensing_movie_information.make_backimage_component_path_baseline_decomposition_surface import (
    MATCH_MAX_DEG,
)
from declan.active_sensing_movie_information.make_backimage_component_path_residual_highres_diagnostic import (
    _compute_highres,
)
from declan.active_sensing_movie_information.plot_backimage_real_trace_unit_first_and_population_schematics import (
    load_dataset,
)


OUT_STEM = "backimage_real_trace_component_path_sf_threshold_lineplots"
SF_METRIC_COL = "sf_split_metric"
SF_MIN_CPDS = (0.50, 0.75, 1.00)
CONTOUR_COHERENCE_MINS = (0.20, 0.50)
CANONICAL_CONTOUR_COHERENCE_MIN = 0.50
PRESENTATION_SF_MIN_CPD = 0.75
BROKEN_MIN_POS = 45.0
BROKEN_MAX_POS = 180.0
BROKEN_TICKS = (0.0, 50.0, 65.0, 90.0, 120.0, 160.0)
MIDDLE_BIN_MIN = 5
MIDDLE_BIN_MAX = 12
EPS = 1e-12

SF_COLORS = {
    0.50: "#777777",
    0.75: "#222222",
    1.00: "#D55E00",
}
AXIS_SPECS = [
    ("across", "Across-contour component path", "across component path length (arcmin)"),
    ("along", "Along-contour component path", "along component path length (arcmin)"),
]
OUTCOME_SPECS = [
    ("ssi", "SSI bits/spike", "SSI residual\n(% vs cell baseline)"),
    ("information", "Information bits/window", "Information residual\n(% vs cell baseline)"),
    ("spikes", "Expected spikes/window", "Expected-spike residual\n(% vs cell baseline)"),
]


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


def _x_broken_log(values: np.ndarray | pd.Series | list[float], *, min_pos: float, max_pos: float) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    mapped = np.zeros_like(x, dtype=float)
    positive = x > 0
    span = 5.1
    mapped[positive] = 1.0 + span * np.log(x[positive] / min_pos) / np.log(max_pos / min_pos)
    return mapped


def _thresholded_data(data: dict[str, Any], *, contour_min: float, sf_min_cpd: float) -> dict[str, Any]:
    out = dict(data)

    image = data["image"].copy()
    coherence = pd.to_numeric(image["image_orientation_coherence"], errors="coerce").to_numpy(dtype=float)
    axis = pd.to_numeric(image["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=float)
    image["image_contour_strong"] = np.isfinite(axis) & np.isfinite(coherence) & (coherence >= float(contour_min))
    out["image"] = image

    unit = data["unit"].copy()
    sf = pd.to_numeric(unit[SF_METRIC_COL], errors="coerce").to_numpy(dtype=float)
    unit["sf_group_original"] = unit["sf_group"].astype(str)
    unit["sf_group"] = np.where(np.isfinite(sf) & (sf >= float(sf_min_cpd)), "high_sf", "below_sf_threshold")
    unit["sf_group_label"] = np.where(
        unit["sf_group"].eq("high_sf"),
        f"SF >= {float(sf_min_cpd):.2f} cpd",
        f"SF < {float(sf_min_cpd):.2f} cpd",
    )
    out["unit"] = unit
    return out


def _compute_surface_sweep(data: dict[str, Any], metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {}
    base_sf = pd.to_numeric(data["unit"][SF_METRIC_COL], errors="coerce")
    for contour_min in CONTOUR_COHERENCE_MINS:
        for sf_min in SF_MIN_CPDS:
            threshold_data = _thresholded_data(data, contour_min=float(contour_min), sf_min_cpd=float(sf_min))
            surface, surface_metadata = _compute_highres(threshold_data, metrics)
            strong = threshold_data["image"]["image_contour_strong"].astype(bool)
            surface = surface.copy()
            surface.insert(0, "contour_coherence_min", float(contour_min))
            surface.insert(1, "sf_min_cpd", float(sf_min))
            surface.insert(2, "sf_metric_col", SF_METRIC_COL)
            surface.insert(3, "n_sf_candidate_units", int(np.count_nonzero(base_sf >= float(sf_min))))
            surface.insert(4, "n_contour_images", int(strong.sum()))
            surface.insert(5, "n_selected_units", int(surface_metadata["n_units"]))
            surface.insert(6, "n_selected_unit_image_pairs", int(surface_metadata["n_unit_image_pairs"]))
            rows.append(surface)
            metadata[f"coh{float(contour_min):.2f}_sf{float(sf_min):.2f}"] = {
                **surface_metadata,
                "contour_coherence_min": float(contour_min),
                "sf_min_cpd": float(sf_min),
                "sf_metric_col": SF_METRIC_COL,
                "n_sf_candidate_units": int(np.count_nonzero(base_sf >= float(sf_min))),
                "n_contour_images": int(strong.sum()),
            }
    return pd.concat(rows, ignore_index=True), metadata


def _aggregate_axis(surface: pd.DataFrame, axis: str) -> pd.DataFrame:
    bin_col = f"{axis}_bin"
    min_col = f"{axis}_min_arcmin"
    max_col = f"{axis}_max_arcmin"
    median_col = f"{axis}_median_arcmin"
    rows: list[dict[str, Any]] = []

    group_cols = ["contour_coherence_min", "sf_min_cpd", bin_col]
    for keys, group in surface.groupby(group_cols, sort=True):
        contour_min, sf_min, bin_index = keys
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
                "contour_coherence_min": float(contour_min),
                "sf_min_cpd": float(sf_min),
                "component_axis": axis,
                "component_bin": int(bin_index),
                "component_min_arcmin": float(pd.to_numeric(group[min_col], errors="coerce").min()),
                "component_max_arcmin": float(pd.to_numeric(group[max_col], errors="coerce").max()),
                "component_median_arcmin": _weighted_mean(group[median_col], group["n_movie_samples"]),
                "n_movie_samples": int(round(n_total)),
                "n_sf_candidate_units": int(group["n_sf_candidate_units"].iloc[0]),
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
    group_cols = ["contour_coherence_min", "sf_min_cpd", "component_axis"]
    for keys, group in marginal.groupby(group_cols, sort=True):
        contour_min, sf_min, axis = keys
        row: dict[str, Any] = {
            "contour_coherence_min": float(contour_min),
            "sf_min_cpd": float(sf_min),
            "component_axis": str(axis),
            "n_sf_candidate_units": int(group["n_sf_candidate_units"].iloc[0]),
            "n_contour_images": int(group["n_contour_images"].iloc[0]),
            "n_selected_units": int(group["n_selected_units"].iloc[0]),
            "n_selected_unit_image_pairs": int(group["n_selected_unit_image_pairs"].iloc[0]),
        }
        for key, _title, _ylabel in OUTCOME_SPECS:
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


def _format_axes(ax: plt.Axes, *, marginal: pd.DataFrame, axis: str, ylabel: str, xlabel: str) -> None:
    band = _middle_band(marginal, axis)
    if band is not None:
        ax.axvspan(band[0], band[1], color="0.92", alpha=0.85, zorder=0)
    ax.axhline(0.0, color="0.35", linewidth=0.85)
    ax.grid(True, color="0.90", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(labelsize=8.0)
    ax.spines[["top", "right"]].set_visible(False)


def _format_broken_axis(
    ax: plt.Axes,
    *,
    marginal: pd.DataFrame,
    axis: str,
    ylabel: str,
    xlabel: str,
    show_xlabel: bool,
) -> None:
    band = _middle_band(marginal, axis)
    if band is not None:
        x0, x1 = _x_broken_log([band[0], band[1]], min_pos=BROKEN_MIN_POS, max_pos=BROKEN_MAX_POS)
        ax.axvspan(float(x0), float(x1), color="0.92", alpha=0.85, zorder=0)
    ax.axhline(0.0, color="0.35", linewidth=0.9)
    ax.grid(True, color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.4)
    ax.set_xlim(-0.12, 5.35)
    ax.set_xticks(_x_broken_log(list(BROKEN_TICKS), min_pos=BROKEN_MIN_POS, max_pos=BROKEN_MAX_POS))
    ax.set_xticklabels([str(int(tick)) for tick in BROKEN_TICKS])
    if show_xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel(ylabel)
    ax.text(
        0.52,
        -0.075,
        "//",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        rotation=-20,
        clip_on=False,
    )


def _plot_focus(marginal: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(len(CONTOUR_COHERENCE_MINS), 2, figsize=(12.7, 7.6), constrained_layout=False)
    fig.suptitle("Stricter high-SF unit thresholds: SSI component-path marginals", fontsize=13.2, y=0.985)
    axes_arr = np.asarray(axes)
    for row_idx, contour_min in enumerate(CONTOUR_COHERENCE_MINS):
        for col_idx, (axis, title, xlabel) in enumerate(AXIS_SPECS):
            ax = axes_arr[row_idx, col_idx]
            frame = marginal[
                marginal["contour_coherence_min"].eq(float(contour_min)) & marginal["component_axis"].eq(axis)
            ]
            for sf_min, group in frame.groupby("sf_min_cpd", sort=True):
                group = group.sort_values("component_bin")
                n_units = int(group["n_selected_units"].iloc[0])
                ax.plot(
                    group["component_median_arcmin"],
                    group["ssi_percent_vs_cell_baseline"],
                    marker="o",
                    linewidth=1.8,
                    markersize=4.2,
                    color=SF_COLORS.get(float(sf_min), None),
                    label=f"SF >= {float(sf_min):.2f} cpd (n={n_units})",
                )
            if row_idx == 0:
                ax.set_title(title, fontsize=10.5)
            ax.text(
                0.02,
                0.93,
                f"contour coherence >= {float(contour_min):.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.0,
                color="0.20",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.8},
            )
            _format_axes(
                ax,
                marginal=marginal,
                axis=axis,
                ylabel="SSI residual\n(% vs cell baseline)" if col_idx == 0 else "",
                xlabel=xlabel if row_idx == len(CONTOUR_COHERENCE_MINS) - 1 else "",
            )
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=7.9, loc="best")
    fig.text(
        0.5,
        0.018,
        "Default high SF is >=0.50 cpd. Marginals recompute SSI after summing information/spikes over the unplotted component axis.",
        ha="center",
        fontsize=8.0,
        color="0.30",
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.09, hspace=0.30, wspace=0.18)
    return fig


def _plot_outcomes(marginal: pd.DataFrame) -> plt.Figure:
    canonical = marginal[marginal["contour_coherence_min"].eq(float(CANONICAL_CONTOUR_COHERENCE_MIN))]
    fig, axes = plt.subplots(len(OUTCOME_SPECS), 2, figsize=(12.8, 9.2), constrained_layout=False)
    fig.suptitle(
        f"Stricter high-SF thresholds at contour coherence >= {CANONICAL_CONTOUR_COHERENCE_MIN:.2f}",
        fontsize=13.0,
        y=0.99,
    )
    for row_idx, (key, outcome_title, ylabel) in enumerate(OUTCOME_SPECS):
        value_col = f"{key}_percent_vs_cell_baseline"
        for col_idx, (axis, axis_title, xlabel) in enumerate(AXIS_SPECS):
            ax = axes[row_idx, col_idx]
            frame = canonical[canonical["component_axis"].eq(axis)]
            for sf_min, group in frame.groupby("sf_min_cpd", sort=True):
                group = group.sort_values("component_bin")
                n_units = int(group["n_selected_units"].iloc[0])
                ax.plot(
                    group["component_median_arcmin"],
                    group[value_col],
                    marker="o",
                    linewidth=1.55,
                    markersize=3.8,
                    color=SF_COLORS.get(float(sf_min), None),
                    label=f"SF >= {float(sf_min):.2f} cpd (n={n_units})",
                )
            if row_idx == 0:
                ax.set_title(axis_title, fontsize=10.2)
            _format_axes(
                ax,
                marginal=canonical,
                axis=axis,
                ylabel=ylabel if col_idx == 0 else "",
                xlabel=xlabel if row_idx == len(OUTCOME_SPECS) - 1 else "",
            )
            ax.text(
                0.02,
                0.93,
                outcome_title,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.2,
                color="0.20",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.8},
            )
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=7.7, loc="best")
    fig.text(
        0.5,
        0.018,
        "Cell baseline, contour alignment, and 16x16 component-path bins match the previous residual diagnostic.",
        ha="center",
        fontsize=8.0,
        color="0.30",
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.93, bottom=0.085, hspace=0.34, wspace=0.20)
    return fig


def _plot_fixed_sf_broken_log(marginal: pd.DataFrame) -> plt.Figure:
    fixed = marginal[marginal["sf_min_cpd"].eq(float(PRESENTATION_SF_MIN_CPD))].copy()
    fig, axes = plt.subplots(len(CONTOUR_COHERENCE_MINS), 2, figsize=(12.2, 7.2), constrained_layout=False)
    fig.suptitle(
        (
            f"SF >= {PRESENTATION_SF_MIN_CPD:.2f} cpd contour-aligned units: "
            "rows differ only by contour-coherence gate"
        ),
        fontsize=13.2,
        y=0.985,
    )
    axes_arr = np.asarray(axes)
    finite_y = fixed["ssi_percent_vs_cell_baseline"].to_numpy(dtype=float)
    finite_y = finite_y[np.isfinite(finite_y)]
    if finite_y.size:
        lo = min(0.0, float(np.nanmin(finite_y)))
        hi = max(0.0, float(np.nanmax(finite_y)))
        span = max(hi - lo, 1.0)
        ylim = (lo - 0.12 * span, hi + 0.15 * span)
    else:
        ylim = (-1.0, 1.0)
    curve_color = SF_COLORS[PRESENTATION_SF_MIN_CPD]
    row_names = {
        0.20: "Reliable-contour gate",
        0.50: "Strong-contour gate",
    }

    for row_idx, contour_min in enumerate(CONTOUR_COHERENCE_MINS):
        row_frame = fixed[fixed["contour_coherence_min"].eq(float(contour_min))]
        n_images = int(row_frame["n_contour_images"].iloc[0]) if not row_frame.empty else 0
        n_units = int(row_frame["n_selected_units"].iloc[0]) if not row_frame.empty else 0
        n_pairs = int(row_frame["n_selected_unit_image_pairs"].iloc[0]) if not row_frame.empty else 0
        row_label = (
            f"{row_names.get(float(contour_min), 'Contour gate')}: "
            f"image coherence >= {float(contour_min):.2f}\n"
            f"{n_images} images, {n_units} units, {n_pairs} unit-image pairs"
        )
        for col_idx, (axis, axis_title, xlabel) in enumerate(AXIS_SPECS):
            ax = axes_arr[row_idx, col_idx]
            frame = row_frame[row_frame["component_axis"].eq(axis)].sort_values("component_bin")
            x = _x_broken_log(
                frame["component_median_arcmin"],
                min_pos=BROKEN_MIN_POS,
                max_pos=BROKEN_MAX_POS,
            )
            y = frame["ssi_percent_vs_cell_baseline"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                color=curve_color,
                linewidth=2.0,
                marker="o",
                markersize=4.6,
                markerfacecolor="white",
                markeredgewidth=1.25,
                zorder=3,
            )
            ax.scatter(
                [0.0],
                [0.0],
                marker="o",
                s=34,
                facecolors="white",
                edgecolors=curve_color,
                linewidths=1.35,
                zorder=5,
            )
            if row_idx == 0:
                ax.set_title(axis_title, fontsize=11.0, pad=8)
            if col_idx == 0:
                ax.text(
                    0.02,
                    0.93,
                    row_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.4,
                    color="0.18",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 2.0},
                )
            _format_broken_axis(
                ax,
                marginal=fixed,
                axis=axis,
                ylabel="SSI residual\n(% vs cell baseline)" if col_idx == 0 else "",
                xlabel=xlabel,
                show_xlabel=row_idx == len(CONTOUR_COHERENCE_MINS) - 1,
            )
            ax.set_ylim(*ylim)

    handles = [
        Line2D(
            [0],
            [0],
            color=curve_color,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.25,
            linewidth=2.0,
            label=f"drift-only component bins; SF >= {PRESENTATION_SF_MIN_CPD:.2f} cpd",
        ),
        Line2D(
            [0],
            [0],
            color=curve_color,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.35,
            linewidth=0.0,
            label="cell-stabilized baseline at 0 path",
        ),
    ]
    axes_arr[0, 1].legend(handles=handles, frameon=False, fontsize=8.3, loc="best")
    fig.text(
        0.5,
        0.022,
        (
            "Positive component path lengths use the same broken log mapping as the reordered story figure; "
            "the 0-path baseline point is intentionally disconnected from the drift curve."
        ),
        ha="center",
        fontsize=8.0,
        color="0.30",
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.89, bottom=0.105, hspace=0.26, wspace=0.18)
    return fig


def main() -> None:
    data = load_dataset(MATRIX_DIR)
    metrics = _compute_component_metrics(data)
    surface, metadata = _compute_surface_sweep(data, metrics)
    marginal, axis_summary = _make_marginals(surface)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    surface_csv = OUT_DIR / f"{OUT_STEM}_surface_values.csv"
    values_csv = OUT_DIR / f"{OUT_STEM}_values.csv"
    axis_summary_csv = OUT_DIR / f"{OUT_STEM}_axis_summary.csv"
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    focus_png = OUT_DIR / f"{OUT_STEM}_ssi_focus.png"
    outcomes_png = OUT_DIR / f"{OUT_STEM}_outcomes_contour_ge_{CANONICAL_CONTOUR_COHERENCE_MIN:g}.png"
    fixed_sf_broken_png = OUT_DIR / f"{OUT_STEM}_sf_ge_{PRESENTATION_SF_MIN_CPD:g}_broken_log_ssi.png"
    fixed_sf_broken_pdf = OUT_DIR / f"{OUT_STEM}_sf_ge_{PRESENTATION_SF_MIN_CPD:g}_broken_log_ssi.pdf"
    json_path = OUT_DIR / f"{OUT_STEM}_summary.json"

    surface.to_csv(surface_csv, index=False)
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
        fig = _plot_fixed_sf_broken_log(marginal)
        pages.savefig(fig, bbox_inches="tight")
        fig.savefig(fixed_sf_broken_png, dpi=240, bbox_inches="tight")
        plt.close(fig)
    with PdfPages(fixed_sf_broken_pdf) as pages:
        fig = _plot_fixed_sf_broken_log(marginal)
        pages.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    _write_json(
        json_path,
        {
            "analysis": "backimage_real_trace_component_path_sf_threshold_lineplots",
            "matrix_dir": MATRIX_DIR,
            "out_dir": OUT_DIR,
            "outputs": {
                "pdf": pdf_path,
                "ssi_focus_png": focus_png,
                "outcomes_png": outcomes_png,
                "fixed_sf_broken_log_pdf": fixed_sf_broken_pdf,
                "fixed_sf_broken_log_png": fixed_sf_broken_png,
                "surface_values_csv": surface_csv,
                "values_csv": values_csv,
                "axis_summary_csv": axis_summary_csv,
                "summary_json": json_path,
            },
            "sweep": {
                "sf_metric_col": SF_METRIC_COL,
                "sf_min_cpds": SF_MIN_CPDS,
                "presentation_sf_min_cpd": PRESENTATION_SF_MIN_CPD,
                "contour_coherence_mins": CONTOUR_COHERENCE_MINS,
                "match_max_deg": MATCH_MAX_DEG,
                "canonical_contour_coherence_min": CANONICAL_CONTOUR_COHERENCE_MIN,
                "broken_log": {
                    "min_pos": BROKEN_MIN_POS,
                    "max_pos": BROKEN_MAX_POS,
                    "ticks": BROKEN_TICKS,
                },
            },
            "metadata": metadata,
            "contract": {
                "unit_selection": "The unit table is copied per sweep cell and sf_group is overwritten to high_sf only when sf_split_metric >= sf_min_cpd.",
                "marginalization": "For each component axis bin, sums moving/cell information and expected-spike totals across all bins of the other component axis, then recomputes SSI.",
                "baseline": "Residuals compare moving real traces to the cell-matched stabilized unit-image composition.",
            },
        },
    )

    print(pdf_path)
    print(focus_png)
    print(outcomes_png)
    print(fixed_sf_broken_pdf)
    print(fixed_sf_broken_png)
    print(surface_csv)
    print(values_csv)
    print(axis_summary_csv)
    print(json_path)


if __name__ == "__main__":
    main()
