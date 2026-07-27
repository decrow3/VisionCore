#!/usr/bin/env python3
"""Methods/results story figure for the zero-gap Vernier contour test."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from declan.vernier_active_sensing.make_rr100_single_contour_story_line_profiles import (
    COMPONENT_SPECS,
    RELATION_ORDER,
    RELATION_TITLES,
    SF_COLOR,
    _axis_bounds,
    _draw_native_ticks,
    _find_reference_surface,
    _format_broken_log_axis,
    _plot_component_line,
    _style_axis,
    load_native_references,
    summarize_line_profiles,
)


DEFAULT_SUMMARY = Path(
    "outputs/notebook_vernier_walkthrough/"
    "rr100_single_contour_panel_c_random_ori_blocks4_n20/"
    "rr100_single_contour_panel_c_high_sf_arcmin_binned_n8_from_grid4_random_ori_4blocks_"
    "arcmin_binned_component_surfaces_n8_summary.csv"
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.13,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15.0,
        fontweight="bold",
    )


def _draw_segment(ax: plt.Axes, center: tuple[float, float], angle_deg: float, length: float, **kwargs: Any) -> None:
    theta = math.radians(float(angle_deg))
    dx = 0.5 * length * math.sin(theta)
    dy = 0.5 * length * math.cos(theta)
    cx, cy = center
    ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], **kwargs)


def _draw_stimulus_panel(ax: plt.Axes) -> None:
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.12")
        spine.set_linewidth(0.9)

    ax.plot([0.0, 0.0], [-0.36, 0.36], color="0.45", lw=10.0, alpha=0.50, solid_capstyle="butt")
    ax.plot([0.0, 0.0], [-0.34, 0.34], color="white", lw=5.6, solid_capstyle="butt")

    t = np.linspace(0.0, 1.0, 140)
    trace_x = 0.13 * np.sin(7.0 * np.pi * t + 0.3) + 0.17 * (t - 0.50)
    trace_y = 0.20 * np.sin(4.2 * np.pi * t + 0.9)
    ax.plot(trace_x, trace_y, color="#56B4E9", lw=1.55, alpha=0.95)
    ax.scatter(trace_x[::24], trace_y[::24], s=9, color="#56B4E9", zorder=3)

    ax.annotate(
        "",
        xy=(0.48, 0.62),
        xytext=(0.48, 0.28),
        arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "white"},
    )
    ax.annotate(
        "",
        xy=(0.68, 0.45),
        xytext=(0.34, 0.45),
        arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "white"},
    )
    ax.text(0.52, 0.63, "along", color="white", fontsize=7.4, va="bottom")
    ax.text(0.70, 0.45, "across", color="white", fontsize=7.4, va="center", ha="left")
    ax.set_title("Zero-gap contour stimulus", fontsize=10.8, pad=7)
    ax.text(
        0.02,
        -0.12,
        "single continuous contour; retinal trace projected onto contour axes",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="0.25",
    )


def _draw_design_panel(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("Randomized axes and high-SF unit groups", fontsize=10.8, pad=7)
    ax.text(
        0.5,
        0.89,
        "4 contour rotations x 20 eye traces",
        ha="center",
        va="center",
        fontsize=9.1,
        color="0.20",
    )
    ax.text(
        0.5,
        0.80,
        "moving responses vs static-center baseline",
        ha="center",
        va="center",
        fontsize=7.6,
        color="0.34",
    )
    specs = [
        (0.20, "aligned", 0.0),
        (0.50, "oblique", 45.0),
        (0.80, "orthogonal", 90.0),
    ]
    for x, label, angle in specs:
        ax.plot([x, x], [0.31, 0.61], color="0.72", lw=3.0, solid_capstyle="butt")
        _draw_segment(
            ax,
            (x, 0.46),
            angle,
            0.38,
            color=SF_COLOR,
            lw=3.0,
            solid_capstyle="round",
        )
        ax.text(x, 0.22, label, ha="center", va="center", fontsize=8.7, color="0.10")
    ax.text(
        0.5,
        0.08,
        "unit orientation relative to the contour",
        ha="center",
        va="center",
        fontsize=7.7,
        color="0.33",
    )


def _surface_grid(surface: pd.DataFrame, value_col: str) -> np.ndarray:
    n_bins = int(pd.to_numeric(surface["arcmin_bin_count"], errors="coerce").max())
    grid = np.full((n_bins, n_bins), np.nan, dtype=np.float64)
    for row in surface.itertuples(index=False):
        across_idx = int(getattr(row, "across_bin")) - 1
        along_idx = int(getattr(row, "along_bin")) - 1
        grid[across_idx, along_idx] = float(getattr(row, value_col))
    return grid


def _surface_tick_labels(surface: pd.DataFrame, axis: str) -> tuple[list[int], list[str]]:
    n_bins = int(pd.to_numeric(surface["arcmin_bin_count"], errors="coerce").max())
    positions = sorted(set([0, max(0, n_bins // 2 - 1), n_bins - 1]))
    bin_col = f"{axis}_bin"
    med_col = f"{axis}_median_arcmin"
    labels: list[str] = []
    for pos in positions:
        rows = surface[pd.to_numeric(surface[bin_col], errors="coerce").eq(pos + 1)]
        median = float(pd.to_numeric(rows[med_col], errors="coerce").median()) if not rows.empty else float("nan")
        labels.append(f"Q{pos + 1}\n{median:.0f}" if math.isfinite(median) else f"Q{pos + 1}")
    return positions, labels


def _symmetric_limits(values: np.ndarray, *, floor: float = 1.0) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -floor, floor
    lim = max(floor, float(np.nanmax(np.abs(finite))))
    return -lim, lim


def _draw_surface_panel(fig: plt.Figure, ax: plt.Axes, summary: pd.DataFrame) -> None:
    surface = summary[
        summary["metric_family"].astype(str).eq("path")
        & summary["relation"].astype(str).eq("contour_matched")
    ].copy()
    all_path = summary[summary["metric_family"].astype(str).eq("path")]
    grid = _surface_grid(surface, "population_ssi_percent_vs_static")
    vmin, vmax = _symmetric_limits(
        pd.to_numeric(all_path["population_ssi_percent_vs_static"], errors="coerce").to_numpy(dtype=np.float64),
        floor=10.0,
    )
    image = ax.imshow(grid, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    x_pos, x_labels = _surface_tick_labels(surface, "along")
    y_pos, y_labels = _surface_tick_labels(surface, "across")
    ax.set_xticks(x_pos, x_labels)
    ax.set_yticks(y_pos, y_labels)
    ax.tick_params(labelsize=7.4)
    ax.set_xlabel("along bin; median arcmin", fontsize=8.1)
    ax.set_ylabel("across bin; median arcmin", fontsize=8.1)
    ax.set_title("Aligned high-SF surface", fontsize=10.8, pad=7)
    ax.spines[["top", "right"]].set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    cbar.ax.tick_params(labelsize=7.0)
    cbar.set_label("SSI change vs static (%)", fontsize=7.3)


def _draw_line_panels(
    axes: list[plt.Axes],
    line_summary: pd.DataFrame,
    native_refs: pd.DataFrame,
) -> None:
    min_pos, max_pos, ticks = _axis_bounds(line_summary, native_refs)
    y_values = pd.to_numeric(line_summary["population_ssi_percent_vs_static"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    finite_y = y_values[np.isfinite(y_values)]
    if finite_y.size:
        low = min(0.0, float(np.min(finite_y)))
        high = max(0.0, float(np.max(finite_y)))
        span = max(high - low, 1.0)
        ylim = (low - 0.14 * span, high + 0.16 * span)
    else:
        ylim = (-1.0, 1.0)
    for idx, relation in enumerate(RELATION_ORDER):
        ax = axes[idx]
        rel = line_summary[line_summary["relation"].astype(str).eq(relation)].copy()
        ax.axhline(0, color="0.35", lw=0.9, ls=":")
        for component in ("across", "along"):
            rows = rel[rel["component"].astype(str).eq(component)]
            if not rows.empty:
                _plot_component_line(ax, rows, component=component, min_pos=min_pos, max_pos=max_pos)
        _draw_native_ticks(ax, native_refs, relation=relation, min_pos=min_pos, max_pos=max_pos)
        _format_broken_log_axis(
            ax,
            ticks=ticks,
            min_pos=min_pos,
            max_pos=max_pos,
            xlabel="path length bin median (arcmin)" if idx == 1 else "",
        )
        _style_axis(ax)
        ax.set_ylim(*ylim)
        ax.set_title(RELATION_TITLES.get(relation, relation.replace("_", " ")), fontsize=10.8, pad=7)
        ax.set_ylabel("SSI change vs static (%)" if idx == 0 else "")


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=SF_COLOR,
            linestyle=COMPONENT_SPECS["across"]["linestyle"],
            marker=COMPONENT_SPECS["across"]["marker"],
            markerfacecolor="white",
            markeredgewidth=1.25,
            linewidth=2.1,
            label=COMPONENT_SPECS["across"]["label"],
        ),
        Line2D(
            [0],
            [0],
            color=SF_COLOR,
            linestyle=COMPONENT_SPECS["along"]["linestyle"],
            marker=COMPONENT_SPECS["along"]["marker"],
            markerfacecolor="white",
            markeredgewidth=1.25,
            linewidth=2.1,
            label=COMPONENT_SPECS["along"]["label"],
        ),
        Line2D(
            [0],
            [0],
            color="0.22",
            linewidth=1.2,
            label="x-axis ticks: native 1x randomized trace projection",
        ),
    ]


def make_figure(
    *,
    summary: pd.DataFrame,
    line_summary: pd.DataFrame,
    native_refs: pd.DataFrame,
    out_dir: Path,
    out_stem: str,
    dpi: int,
) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(11.4, 8.0), dpi=int(dpi))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.060,
        right=0.985,
        top=0.885,
        bottom=0.105,
        hspace=0.48,
        wspace=0.34,
        height_ratios=(0.92, 1.08),
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    bottom_axes = [fig.add_subplot(grid[1, idx]) for idx in range(3)]

    _draw_stimulus_panel(ax_a)
    _draw_design_panel(ax_b)
    _draw_surface_panel(fig, ax_c, summary)
    _draw_line_panels(bottom_axes, line_summary, native_refs)

    for label, ax in zip(["A", "B", "C", "D", "E", "F"], [ax_a, ax_b, ax_c] + bottom_axes):
        _panel_label(ax, label)

    fig.suptitle(
        "Zero-gap Vernier contour tests: contour-relative motion and high-SF SSI",
        fontsize=15.0,
        y=0.970,
    )
    fig.legend(
        handles=_legend_handles(),
        frameon=False,
        fontsize=8.0,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.515),
        ncol=3,
        columnspacing=1.35,
        handlelength=2.8,
    )
    fig.text(
        0.5,
        0.030,
        "Surfaces use measured contour-relative component bins from generated moving traces; line points pool "
        "information numerator and expected spikes before computing spike-weighted population SSI.",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="0.30",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{out_stem}.png"
    pdf = out_dir / f"{out_stem}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--reference-surface-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--out-stem", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=230)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    summary = pd.read_csv(summary_csv)
    reference_surface = Path(args.reference_surface_csv) if args.reference_surface_csv else _find_reference_surface(summary_csv)
    line_summary = summarize_line_profiles(summary, metric_family="path")
    native_refs = load_native_references(reference_surface, metric_family="path")
    out_dir = Path(args.out_dir) if args.out_dir else summary_csv.parent
    default_stem = summary_csv.stem.replace("_summary", "") + "_methods_results_story_figure"
    out_stem = str(args.out_stem or default_stem)
    png, pdf = make_figure(
        summary=summary,
        line_summary=line_summary,
        native_refs=native_refs,
        out_dir=out_dir,
        out_stem=out_stem,
        dpi=int(args.dpi),
    )
    line_csv = out_dir / f"{out_stem}_line_profile_summary.csv"
    line_summary.to_csv(line_csv, index=False)
    manifest = out_dir / f"{out_stem}_manifest.json"
    _write_json(
        manifest,
        {
            "summary_csv": summary_csv,
            "reference_surface_csv": reference_surface,
            "line_profile_summary_csv": line_csv,
            "png": png,
            "pdf": pdf,
        },
    )
    print(line_csv)
    print(png)
    print(pdf)
    print(manifest)


if __name__ == "__main__":
    main()
