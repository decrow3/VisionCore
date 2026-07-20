#!/usr/bin/env python3
"""Make a lead-in panel for coarse motion direction relative to contours.

These panels are deliberately narrower and clearer than the older diagnostic named
"Trace covariance axis relative to image contour axis".  It fixes the local
image contour as the reference frame, pools units across their tuning
orientation relative to that contour, and varies the direction in which the real
fixational trace is elongated.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


MATRIX_DIR = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged"
)
CONDITION_DIR = MATRIX_DIR / "phase1_phase2_conditioning_v1"
SUMMARY_DIR = CONDITION_DIR / "schematic_pathlength_summary_v1" / "unit_first_and_population_v1"
OUT_DIR = SUMMARY_DIR / "figures"
MOVIE_TABLE = CONDITION_DIR / "phase1_movie_analysis_table.csv"

SF_GROUPS = {
    "low_sf": {
        "label": "Low SF",
        "color": "#0072B2",
        "baseline_col": "sf_low_sf_weighted_stabilized_ssi",
        "delta_col": "sf_low_sf_weighted_ssi_delta_vs_stabilized",
    },
    "middle_sf": {
        "label": "Middle SF",
        "color": "#009E73",
        "baseline_col": "sf_middle_sf_weighted_stabilized_ssi",
        "delta_col": "sf_middle_sf_weighted_ssi_delta_vs_stabilized",
    },
    "high_sf": {
        "label": "High SF",
        "color": "#D55E00",
        "baseline_col": "sf_high_sf_weighted_stabilized_ssi",
        "delta_col": "sf_high_sf_weighted_ssi_delta_vs_stabilized",
    },
}
ALL_UNITS_GROUP = {
    "label": "All SF groups",
    "color": "0.25",
    "baseline_col": "stabilized_population_ssi",
    "delta_col": "population_ssi_delta_vs_stabilized",
}
GROUP_SPECS = {**SF_GROUPS, "all_units": ALL_UNITS_GROUP}
AXIS_CLASS_STYLES = {
    "across_contour_axis": {
        "label": "across contour",
        "color": "0.18",
        "linestyle": "-",
        "marker": "o",
        "zorder": 4,
    },
    "along_contour_axis": {
        "label": "along contour",
        "color": "0.18",
        "linestyle": "--",
        "marker": "s",
        "zorder": 4,
    },
    "oblique": {
        "label": "oblique",
        "color": "0.45",
        "linestyle": "-.",
        "marker": "D",
        "zorder": 3,
    },
    "low_anisotropy": {
        "label": "weak directional bias",
        "color": "0.68",
        "linestyle": ":",
        "marker": "^",
        "zorder": 2,
    },
}


def _sem(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def _x_broken_log(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    mapped = np.zeros_like(x, dtype=float)
    positive = x > 0
    min_pos = 88.0
    max_pos = 180.0
    span = 5.1
    mapped[positive] = 1.0 + span * np.log(x[positive] / min_pos) / np.log(max_pos / min_pos)
    return mapped


def _format_movement_axis(ax: plt.Axes) -> None:
    ticks = [0, 90, 105, 120, 150, 175]
    ax.set_xlim(-0.12, 5.35)
    ax.set_xticks(_x_broken_log(ticks))
    ax.set_xticklabels([str(tick) for tick in ticks])
    ax.text(
        0.52,
        -0.075,
        "//",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        rotation=-20,
        clip_on=False,
    )


def _load_summary() -> pd.DataFrame:
    usecols = [
        "image_contour_strong",
        "trace_image_axis_class",
        "trace_path_length_bin",
        "rendered_path_length_arcmin",
    ]
    for spec in GROUP_SPECS.values():
        usecols.extend([spec["baseline_col"], spec["delta_col"]])
    movie = pd.read_csv(MOVIE_TABLE, usecols=usecols)
    movie = movie[movie["image_contour_strong"].astype(bool)].copy()
    rows = []
    for group_name, spec in GROUP_SPECS.items():
        percent_col = f"{group_name}_percent_vs_stabilized"
        movie[percent_col] = (
            100.0
            * pd.to_numeric(movie[spec["delta_col"]], errors="coerce")
            / pd.to_numeric(movie[spec["baseline_col"]], errors="coerce")
        )
        for (axis_class, trace_bin), group in movie.groupby(["trace_image_axis_class", "trace_path_length_bin"], sort=True):
            values = group[percent_col]
            rows.append(
                {
                    "sf_group": group_name,
                    "sf_label": spec["label"],
                    "trace_image_axis_class": axis_class,
                    "trace_path_length_bin": trace_bin,
                    "path_median_arcmin": float(np.nanmedian(group["rendered_path_length_arcmin"])),
                    "mean_percent": float(np.nanmean(values)),
                    "sem_percent": _sem(values),
                    "n_movies": int(group.shape[0]),
                    "n_traces": int(group["rendered_path_length_arcmin"].shape[0]),
                }
            )
    return pd.DataFrame(rows)


def _plot_sf_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    sf_group: str,
    show_legend: bool,
    show_xlabel: bool,
    show_ylabel: bool = True,
) -> None:
    spec = GROUP_SPECS[sf_group]
    ax.axhline(0, color="0.35", lw=1.0, ls=":")
    ax.scatter(
        [0.0],
        [0.0],
        s=48,
        facecolors="white",
        edgecolors=spec["color"],
        linewidths=1.8,
        zorder=6,
        label="no-motion baseline",
    )

    for axis_class, style in AXIS_CLASS_STYLES.items():
        rows = summary[
            summary["sf_group"].eq(sf_group) & summary["trace_image_axis_class"].eq(axis_class)
        ].sort_values("path_median_arcmin")
        if rows.empty:
            continue
        x = _x_broken_log(rows["path_median_arcmin"])
        y = rows["mean_percent"].to_numpy(dtype=float)
        err = rows["sem_percent"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=err,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=5.2,
            markerfacecolor="white",
            markeredgewidth=1.4,
            linewidth=1.9,
            capsize=2,
            alpha=0.95,
            zorder=style["zorder"],
            label=style["label"],
        )

    ax.text(
        0.02,
        0.88,
        spec["label"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=spec["color"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
    )
    ax.set_ylabel("SSI change from no-motion movie (%)" if show_ylabel else "")
    ax.set_xlabel("trajectory path length (arcmin; log scale after break)" if show_xlabel else "")
    _format_movement_axis(ax)
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)
    ax.grid(True, color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        ax.legend(
            title="Trajectory relative to contour",
            frameon=True,
            facecolor="white",
            edgecolor="white",
            framealpha=0.88,
            fontsize=8.4,
            title_fontsize=8.8,
            loc="lower right",
            ncols=1,
        )


def make_high_sf_panel(summary: pd.DataFrame) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    _plot_sf_panel(ax, summary, sf_group="high_sf", show_legend=True, show_xlabel=True)
    ax.set_title("Information gain varies with gaze trajectory relative to local contours", fontsize=14, pad=12)
    ax.set_ylim(-2.5, 22.0)
    fig.tight_layout()

    png = OUT_DIR / "motion_direction_relative_to_contour_high_sf_leadin_panel.png"
    pdf = OUT_DIR / "motion_direction_relative_to_contour_high_sf_leadin_panel.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def make_all_sf_panel(summary: pd.DataFrame) -> tuple[Path, Path]:
    fig, axes = plt.subplots(3, 1, figsize=(8.7, 8.2), sharex=False)
    for idx, sf_group in enumerate(SF_GROUPS.keys()):
        _plot_sf_panel(
            axes[idx],
            summary,
            sf_group=sf_group,
            show_legend=False,
            show_xlabel=idx == len(SF_GROUPS) - 1,
            show_ylabel=False,
        )
        sub = summary[summary["sf_group"].eq(sf_group)]
        y_vals = sub["mean_percent"].to_numpy(dtype=float)
        y_err = sub["sem_percent"].to_numpy(dtype=float)
        ok = np.isfinite(y_vals) & np.isfinite(y_err)
        if np.any(ok):
            y_min = min(0.0, float(np.nanmin(y_vals[ok] - y_err[ok])))
            y_max = max(0.0, float(np.nanmax(y_vals[ok] + y_err[ok])))
            span = max(y_max - y_min, 1.0)
            axes[idx].set_ylim(y_min - 0.08 * span, y_max + 0.08 * span)
    axes[0].set_title("Information gain varies with gaze trajectory relative to local contours", fontsize=14, pad=12)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="0.25",
            markeredgewidth=1.8,
            linestyle="none",
            label="no-motion baseline",
        )
    ]
    for axis_class, style in AXIS_CLASS_STYLES.items():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=5.5,
                markerfacecolor="white",
                markeredgewidth=1.4,
                linewidth=1.9,
                label=style["label"],
            )
        )
    fig.supylabel("SSI change from no-motion movie (%)", x=0.025, fontsize=11)
    fig.legend(
        handles=legend_handles,
        title="Trajectory relative to contour",
        frameon=False,
        loc="lower center",
        ncols=5,
        fontsize=8.4,
        title_fontsize=8.8,
        bbox_to_anchor=(0.52, 0.035),
    )
    fig.tight_layout(rect=(0.055, 0.085, 0.995, 0.985), h_pad=0.25)

    png = OUT_DIR / "motion_direction_relative_to_contour_all_sf_leadin_panel.png"
    pdf = OUT_DIR / "motion_direction_relative_to_contour_all_sf_leadin_panel.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def make_all_units_panel(summary: pd.DataFrame) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    _plot_sf_panel(ax, summary, sf_group="all_units", show_legend=True, show_xlabel=True)
    sub = summary[summary["sf_group"].eq("all_units")]
    y_vals = sub["mean_percent"].to_numpy(dtype=float)
    y_err = sub["sem_percent"].to_numpy(dtype=float)
    ok = np.isfinite(y_vals) & np.isfinite(y_err)
    if np.any(ok):
        y_min = min(0.0, float(np.nanmin(y_vals[ok] - y_err[ok])))
        y_max = max(0.0, float(np.nanmax(y_vals[ok] + y_err[ok])))
        span = max(y_max - y_min, 1.0)
        ax.set_ylim(y_min - 0.08 * span, y_max + 0.08 * span)
    ax.set_title("Information gain varies with gaze trajectory relative to local contours", fontsize=14, pad=12)
    fig.tight_layout()

    png = OUT_DIR / "motion_direction_relative_to_contour_all_units_leadin_panel.png"
    pdf = OUT_DIR / "motion_direction_relative_to_contour_all_units_leadin_panel.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def make_panels() -> tuple[Path, ...]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load_summary()
    summary_csv = OUT_DIR / "motion_direction_relative_to_contour_all_sf_leadin_summary.csv"
    summary.to_csv(summary_csv, index=False)
    high_png, high_pdf = make_high_sf_panel(summary)
    all_png, all_pdf = make_all_sf_panel(summary)
    all_units_png, all_units_pdf = make_all_units_panel(summary)
    return all_units_png, all_units_pdf, all_png, all_pdf, high_png, high_pdf, summary_csv


def main() -> None:
    for path in make_panels():
        print(path)


if __name__ == "__main__":
    main()
