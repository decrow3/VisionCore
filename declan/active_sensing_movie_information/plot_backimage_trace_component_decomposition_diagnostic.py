#!/usr/bin/env python3
"""Plot a compact component-decomposition diagnostic for real-trace BackImage SSI.

This is the component-path counterpart to the older trace covariance major-axis
diagnostic.  The older plot grouped movies by the major axis of the trace
covariance ellipse relative to the local image contour.  Here we instead ask how
SSI changes as the actual path component across or along the contour changes.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

from declan.active_sensing_movie_information.plot_backimage_real_trace_unit_first_and_population_schematics import (
    SF_COLORS,
    SF_LABELS,
    SF_ORDER,
    add_break_marks,
    component_log_ticks,
    y_limits_for_plot,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged/phase1_phase2_conditioning_v1/"
    "schematic_pathlength_summary_v1/unit_first_and_population_v1"
)

RELATION_TITLES = {
    "all_images_no_osi": "all image windows, all units",
    "strong_contours_no_osi": "strong contour images, all units",
    "contour_matched": "strong contour images, orientation-aligned units",
    "contour_intermediate": "strong contour images, intermediate-orientation units",
    "contour_orthogonal": "strong contour images, orientation-orthogonal units",
}
COMPONENT_LABELS = {
    "across_path_arcmin": "across contour",
    "along_path_arcmin": "along contour",
}
COMPONENT_STYLES = {
    "across_path_arcmin": {"linestyle": "-", "marker": "o", "label": "across-contour component"},
    "along_path_arcmin": {"linestyle": "--", "marker": "s", "label": "along-contour component"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument(
        "--relations",
        default="strong_contours_no_osi,all_images_no_osi",
        help="Comma-separated relation names from spike_weighted_population_component_summary.csv.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def ordered_rows(summary: pd.DataFrame, sf_group: str, component_metric: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = summary[
        summary["sf_group"].astype(str).eq(str(sf_group))
        & summary["component_metric"].astype(str).eq(str(component_metric))
    ].copy()
    zero = sub[sub["context"].astype(str).eq("stabilized")].sort_values("component_bin_order").head(1)
    drift = sub[sub["context"].astype(str).eq("drift_only")].sort_values("component_median_arcmin")
    microsaccade = sub[sub["context"].astype(str).eq("microsaccade")].sort_values("component_median_arcmin")
    return zero, drift, microsaccade


def finite_range(values: list[np.ndarray], pad_frac: float = 0.08) -> tuple[float, float]:
    merged = np.concatenate([arr[np.isfinite(arr)] for arr in values if arr.size])
    if merged.size == 0:
        return 0.0, 1.0
    lo = float(np.nanmin(merged))
    hi = float(np.nanmax(merged))
    if lo <= 0.0 <= hi:
        span = max(hi - lo, 0.01)
    else:
        span = max(hi - lo, max(abs(lo), abs(hi), 0.01) * 0.12)
    return lo - pad_frac * span, hi + pad_frac * span


def panel_y_limits(summary: pd.DataFrame, metric_col: str, low_col: str, high_col: str) -> tuple[float, float]:
    if low_col and high_col:
        return y_limits_for_plot(summary, metric_col, low_col, high_col)
    vals = [pd.to_numeric(summary[metric_col], errors="coerce").to_numpy(dtype=float)]
    if metric_col.endswith("delta_vs_stabilized"):
        vals.append(np.asarray([0.0], dtype=float))
    return finite_range(vals)


def setup_broken_axes(
    fig: Any,
    gs: Any,
    row_idx: int,
    col_left: int,
    col_right: int,
    positives: pd.Series,
    y_limits: tuple[float, float],
    *,
    show_x: bool,
) -> tuple[Any, Any]:
    ax_left = fig.add_subplot(gs[row_idx, col_left])
    ax_right = fig.add_subplot(gs[row_idx, col_right])
    for ax in (ax_left, ax_right):
        ax.set_ylim(*y_limits)
        ax.grid(True, color="0.9", linewidth=0.8)
        ax.spines[["top"]].set_visible(False)
        ax.tick_params(labelsize=8.8)
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.yaxis.set_visible(False)
    ax_left.set_xlim(-0.18, 0.18)
    ax_left.set_xticks([0.0])
    ax_left.set_xticklabels(["0"])
    ax_right.set_xscale("log")
    pos = pd.to_numeric(positives, errors="coerce")
    pos = pos[np.isfinite(pos) & (pos > 0)]
    if pos.empty:
        ax_right.set_xlim(1.0, 2.0)
        ticks = [1.0]
    else:
        ax_right.set_xlim(float(pos.min()) * 0.94, float(pos.max()) * 1.05)
        ticks = component_log_ticks(pos)
    ax_right.xaxis.set_major_locator(FixedLocator(ticks))
    ax_right.xaxis.set_major_formatter(FixedFormatter([str(int(tick)) for tick in ticks]))
    ax_right.xaxis.set_minor_formatter(NullFormatter())
    if not show_x:
        ax_left.tick_params(axis="x", labelbottom=False)
        ax_right.tick_params(axis="x", labelbottom=False)
    add_break_marks(ax_left, ax_right)
    return ax_left, ax_right


def plot_relation_component_decomposition(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    relation: str,
    dpi: int,
) -> dict[str, str]:
    rel = summary[summary["relation"].astype(str).eq(str(relation))].copy()
    if rel.empty:
        raise ValueError(f"No rows found for relation={relation!r}")

    sf_groups = [sf for sf in SF_ORDER if sf in set(rel["sf_group"].astype(str))]
    fig = plt.figure(figsize=(11.8, 7.2), constrained_layout=False)
    gs = fig.add_gridspec(
        len(sf_groups),
        5,
        width_ratios=[0.34, 3.05, 0.58, 0.34, 3.05],
        wspace=0.045,
        hspace=0.30,
    )
    axes_by_panel: dict[tuple[int, int], tuple[Any, Any]] = {}

    for row_idx, sf_group in enumerate(sf_groups):
        sf_summary = rel[rel["sf_group"].astype(str).eq(sf_group)].copy()
        sf_color = SF_COLORS.get(sf_group, "0.2")
        positives = sf_summary.loc[sf_summary["component_median_arcmin"] > 0, "component_median_arcmin"]
        panel_specs = [
            (
                "population_ssi_bits_per_spike",
                "population_ci95_low_image_boot",
                "population_ci95_high_image_boot",
                "Population SSI\n(bits/spike)",
                "Absolute SSI",
                0,
                1,
            ),
            (
                "population_ssi_delta_vs_stabilized",
                "population_delta_ci95_low_image_boot",
                "population_delta_ci95_high_image_boot",
                "Population SSI - stabilized\n(bits/spike)",
                "Movement modulation",
                3,
                4,
            ),
        ]
        for col_idx, (metric_col, low_col, high_col, ylabel, panel_title, left_col, right_col) in enumerate(panel_specs):
            limits = panel_y_limits(sf_summary, metric_col, low_col, high_col)
            ax_left, ax_right = setup_broken_axes(
                fig,
                gs,
                row_idx,
                left_col,
                right_col,
                positives,
                limits,
                show_x=row_idx == len(sf_groups) - 1,
            )
            axes_by_panel[(row_idx, col_idx)] = (ax_left, ax_right)
            if metric_col.endswith("delta_vs_stabilized"):
                ax_left.axhline(0.0, color="0.35", lw=1.0, ls=":")
                ax_right.axhline(0.0, color="0.35", lw=1.0, ls=":")
            if row_idx == len(sf_groups) // 2:
                ax_left.set_ylabel(ylabel, fontsize=10)
            if row_idx == 0:
                ax_right.set_title(panel_title, fontsize=12, pad=8)
            if col_idx == 0:
                ax_right.text(
                    0.02,
                    0.84,
                    SF_LABELS.get(sf_group, sf_group),
                    color=sf_color,
                    fontsize=10.5,
                    fontweight="bold",
                    va="center",
                    ha="left",
                    transform=ax_right.transAxes,
                )

            zero = sf_summary[sf_summary["context"].astype(str).eq("stabilized")].sort_values("component_bin_order").head(1)
            if not zero.empty:
                z = zero.iloc[0]
                y = float(z[metric_col])
                ax_left.plot(
                    [0.0],
                    [y],
                    marker="o",
                    markersize=5.4,
                    color=sf_color,
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                    lw=0,
                    alpha=0.95,
                )

            for component_metric in ("across_path_arcmin", "along_path_arcmin"):
                style = COMPONENT_STYLES[component_metric]
                _zero, drift, microsaccade = ordered_rows(sf_summary, sf_group, component_metric)
                for rows, context in ((drift, "drift_only"), (microsaccade, "microsaccade")):
                    if rows.empty:
                        continue
                    x = rows["component_median_arcmin"].to_numpy(dtype=float)
                    y = rows[metric_col].to_numpy(dtype=float)
                    if low_col in rows.columns and high_col in rows.columns:
                        lo = rows[low_col].to_numpy(dtype=float)
                        hi = rows[high_col].to_numpy(dtype=float)
                        ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
                        if np.any(ok):
                            yerr = np.vstack([np.maximum(y[ok] - lo[ok], 0.0), np.maximum(hi[ok] - y[ok], 0.0)])
                            ax_right.errorbar(
                                x[ok],
                                y[ok],
                                yerr=yerr,
                                color=sf_color,
                                lw=1.05 if metric_col.endswith("delta_vs_stabilized") else 1.25,
                                capsize=0,
                                alpha=0.42 if metric_col.endswith("delta_vs_stabilized") else 0.55,
                                zorder=2,
                            )
                    marker_face = "white" if context == "drift_only" else sf_color
                    ax_right.plot(
                        x,
                        y,
                        color=sf_color,
                        linestyle=str(style["linestyle"]),
                        lw=1.75,
                        marker=str(style["marker"]),
                        markersize=4.6,
                        markerfacecolor=marker_face,
                        markeredgewidth=1.45,
                        zorder=4,
                    )

    handles = [
        Line2D([0], [0], color="0.20", marker="o", markerfacecolor="white", markeredgewidth=1.45, lw=1.8, ls="-", label="across-contour component"),
        Line2D([0], [0], color="0.20", marker="s", markerfacecolor="white", markeredgewidth=1.45, lw=1.8, ls="--", label="along-contour component"),
        Line2D([0], [0], color="0.25", marker="o", markerfacecolor="white", markeredgewidth=1.45, lw=0, label="no detected microsaccade"),
        Line2D([0], [0], color="0.25", marker="o", markerfacecolor="0.25", markeredgewidth=1.45, lw=0, label=">=1 detected microsaccade"),
        Line2D([0], [0], color="0.45", lw=1.3, alpha=0.55, label="95% image bootstrap CI"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.8, bbox_to_anchor=(0.54, 0.055))
    title = f"Trace component path relative to image contour axis - {RELATION_TITLES.get(relation, relation)}"
    fig.suptitle(title, fontsize=14, y=0.982)
    fig.supxlabel("Eye movement component size (component path length, arcmin; log scale after break)", y=0.018, fontsize=11)
    fig.subplots_adjust(left=0.078, right=0.985, bottom=0.14, top=0.90)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"phase2_trace_component_decomposition_{relation}_by_sf"
    paths = {"png": out_dir / f"{stem}.png", "pdf": out_dir / f"{stem}.pdf"}
    fig.savefig(paths["png"], dpi=int(dpi), bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)
    return {key: str(path) for key, path in paths.items()}


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    summary_csv = summary_dir / "spike_weighted_population_component_summary.csv"
    summary = pd.read_csv(summary_csv)
    fig_dir = summary_dir / "figures"
    outputs: dict[str, dict[str, str]] = {}
    for relation in parse_csv_list(args.relations):
        outputs[relation] = plot_relation_component_decomposition(summary, fig_dir, relation=relation, dpi=int(args.dpi))
    for relation, paths in outputs.items():
        print(f"{relation}:")
        for kind, path in paths.items():
            print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
