#!/usr/bin/env python3
"""Pure across/along SSI tuning split by SF group and contour relation."""

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
from matplotlib import transforms
from matplotlib.lines import Line2D


DEFAULT_OUT_DIR = Path("outputs/notebook_vernier_walkthrough/rr100_single_contour_panel_c_random_ori_blocks4_n20")
DEFAULT_HIGH_SUMMARY = (
    DEFAULT_OUT_DIR / "rr100_single_contour_panel_c_high_sf_orientation_relation_split_random_ori_4blocks_summary.csv"
)
DEFAULT_LOW_SUMMARY = (
    DEFAULT_OUT_DIR
    / "rr100_single_contour_panel_c_orientation_tuned_low_sf_random_ori_4blocks_orientation_relation_split_summary.csv"
)

SF_ORDER = ("low_sf", "high_sf")
SF_TITLES = {"low_sf": "Low-SF units", "high_sf": "High-SF units"}
SF_COLORS = {"low_sf": "#0072B2", "high_sf": "#D55E00"}
RELATION_ORDER = ("contour_matched", "contour_intermediate", "contour_orthogonal")
RELATION_TITLES = {
    "contour_matched": "Contour-aligned",
    "contour_intermediate": "Oblique",
    "contour_orthogonal": "Contour-orthogonal",
}
MOVEMENT_STYLES = {
    "across": {
        "label": "pure across-contour motion",
        "marker": "o",
        "linestyle": "-",
    },
    "along": {
        "label": "pure along-contour motion",
        "marker": "s",
        "linestyle": (0, (4.2, 2.0)),
    },
}
TICKS = [0.0, 25.0, 50.0, 65.0, 90.0, 120.0, 160.0, 240.0, 360.0]


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


def _x_broken_log(values: np.ndarray | pd.Series | list[float], *, min_pos: float, max_pos: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    mapped = np.zeros_like(x, dtype=np.float64)
    positive = x > 0.0
    if max_pos <= min_pos:
        max_pos = min_pos * 2.0
    mapped[positive] = 1.0 + 5.1 * np.log(x[positive] / min_pos) / np.log(max_pos / min_pos)
    return mapped


def _axis_bounds(summary: pd.DataFrame) -> tuple[float, float, list[float]]:
    values = pd.to_numeric(summary["path_median_arcmin"], errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1.0, 2.0, [0.0, 1.0, 2.0]
    min_pos = float(np.nanmin(values)) * 0.94
    max_pos = float(np.nanmax(values)) * 1.04
    ticks = [tick for tick in TICKS if tick == 0.0 or (tick >= min_pos * 0.90 and tick <= max_pos * 1.02)]
    return min_pos, max_pos, ticks


def _format_axis(ax: plt.Axes, *, min_pos: float, max_pos: float, ticks: list[float]) -> None:
    ax.set_xlim(-0.12, float(_x_broken_log([max(max(ticks), max_pos)], min_pos=min_pos, max_pos=max_pos)[0]) + 0.25)
    ax.set_xticks(_x_broken_log(ticks, min_pos=min_pos, max_pos=max_pos))
    ax.set_xticklabels([str(int(tick)) if abs(tick - round(tick)) < 1e-9 else f"{tick:g}" for tick in ticks])
    ax.text(
        0.52,
        -0.075,
        "//",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        rotation=-20,
        clip_on=False,
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.4)


def _draw_native_projection_ticks(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    color: str,
    min_pos: float,
    max_pos: float,
) -> None:
    transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for family, style in MOVEMENT_STYLES.items():
        sub = rows[
            rows["movement_family"].astype(str).eq(family)
            & np.isclose(pd.to_numeric(rows["movement_scale"], errors="coerce"), 1.0)
        ]
        if sub.empty:
            continue
        x_val = float(pd.to_numeric(sub["path_median_arcmin"], errors="coerce").iloc[0])
        if not (math.isfinite(x_val) and x_val > 0.0):
            continue
        x = float(_x_broken_log([x_val], min_pos=min_pos, max_pos=max_pos)[0])
        ax.plot(
            [x, x],
            [-0.018, 0.060],
            transform=transform,
            color=color,
            linestyle=":",
            linewidth=1.4,
            alpha=0.85,
            clip_on=False,
            zorder=6,
        )


def _plot_one(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    color: str,
    min_pos: float,
    max_pos: float,
    ticks: list[float],
) -> None:
    moving = rows[~rows["is_static_baseline"].astype(bool)].copy()
    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    for family, style in MOVEMENT_STYLES.items():
        sub = moving[moving["movement_family"].astype(str).eq(family)].sort_values("movement_scale")
        if sub.empty:
            continue
        x = _x_broken_log(sub["path_median_arcmin"], min_pos=min_pos, max_pos=max_pos)
        y = pd.to_numeric(sub["population_ssi_percent_vs_static"], errors="coerce").to_numpy(dtype=np.float64)
        lo = pd.to_numeric(sub["population_delta_percent_ci95_low_trace_boot"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        hi = pd.to_numeric(sub["population_delta_percent_ci95_high_trace_boot"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        yerr = np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])
        ax.plot(
            x,
            y,
            color=color,
            linestyle=style["linestyle"],
            linewidth=2.0,
            marker=str(style["marker"]),
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            zorder=3,
        )
        ax.errorbar(x, y, yerr=yerr, color=color, linestyle="none", elinewidth=1.0, capsize=2.0, zorder=2)
        ax.scatter(
            [0.0],
            [0.0],
            marker=str(style["marker"]),
            s=28,
            facecolors="white",
            edgecolors=color,
            linewidths=1.2,
            zorder=5,
        )
    _draw_native_projection_ticks(ax, rows, color=color, min_pos=min_pos, max_pos=max_pos)
    _format_axis(ax, min_pos=min_pos, max_pos=max_pos, ticks=ticks)
    _style_axis(ax)


def _load_summary(path: Path, sf_group: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.copy()
    data["sf_group"] = str(sf_group)
    return data


def make_figure(
    summary: pd.DataFrame,
    *,
    out_dir: Path,
    out_stem: str,
    dpi: int,
) -> tuple[Path, Path]:
    min_pos, max_pos, ticks = _axis_bounds(summary)
    y_cols = [
        "population_ssi_percent_vs_static",
        "population_delta_percent_ci95_low_trace_boot",
        "population_delta_percent_ci95_high_trace_boot",
    ]
    y_values: list[float] = [0.0]
    for col in y_cols:
        arr = pd.to_numeric(summary[col], errors="coerce").to_numpy(dtype=np.float64)
        y_values.extend(arr[np.isfinite(arr)].tolist())
    finite = np.asarray(y_values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    y_min = min(-5.0, float(np.nanmin(finite)) if finite.size else -5.0)
    y_max = max(5.0, float(np.nanmax(finite)) if finite.size else 5.0)
    span = max(y_max - y_min, 1.0)
    ylim = (y_min - 0.10 * span, y_max + 0.14 * span)

    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.2), dpi=int(dpi), sharex=True, sharey=True)
    for row_idx, sf_group in enumerate(SF_ORDER):
        for col_idx, relation in enumerate(RELATION_ORDER):
            ax = axes[row_idx, col_idx]
            rows = summary[
                summary["sf_group"].astype(str).eq(sf_group)
                & summary["relation"].astype(str).eq(relation)
            ].copy()
            _plot_one(ax, rows, color=SF_COLORS[sf_group], min_pos=min_pos, max_pos=max_pos, ticks=ticks)
            ax.set_ylim(*ylim)
            if row_idx == 0:
                ax.set_title(RELATION_TITLES[relation], fontsize=10.8, pad=7)
            if col_idx == 0:
                ax.set_ylabel(f"{SF_TITLES[sf_group]}\nSSI change vs static (%)", fontsize=9.4, color=SF_COLORS[sf_group])
            if row_idx == 1 and col_idx == 1:
                ax.set_xlabel("component path length (arcmin; log scale after break)", fontsize=9.4)

    style_handles = [
        Line2D(
            [0],
            [0],
            color="0.18",
            marker=str(style["marker"]),
            markerfacecolor="white",
            markeredgewidth=1.2,
            linestyle=style["linestyle"],
            linewidth=2.0,
            label=str(style["label"]),
        )
        for style in MOVEMENT_STYLES.values()
    ]
    color_handles = [
        Line2D([0], [0], color=SF_COLORS["low_sf"], linewidth=2.3, label="low-SF population"),
        Line2D([0], [0], color=SF_COLORS["high_sf"], linewidth=2.3, label="high-SF population"),
    ]
    tick_handle = Line2D([0], [0], color="0.35", linestyle=":", linewidth=1.3, label="dotted ticks: native 1x projection")
    fig.legend(
        handles=color_handles + style_handles + [tick_handle],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        fontsize=8.2,
        handlelength=2.7,
        columnspacing=1.15,
    )
    fig.suptitle(
        "Zero-gap Vernier contour: pure-direction SSI tuning by SF population",
        fontsize=14.2,
        y=0.982,
    )
    fig.text(
        0.5,
        0.025,
        "Trajectories are projected onto contour-relative axes; one component is varied while the orthogonal component is held at the trace mean. "
        "Values are spike-weighted population SSI changes from the static-center baseline.",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="0.30",
    )
    fig.tight_layout(rect=(0.035, 0.065, 0.995, 0.885), h_pad=1.1, w_pad=1.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{out_stem}.png"
    pdf = out_dir / f"{out_stem}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-summary", type=Path, default=DEFAULT_LOW_SUMMARY)
    parser.add_argument("--high-summary", type=Path, default=DEFAULT_HIGH_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-stem", type=str, default="rr100_single_contour_low_high_sf_pure_direction_tuning")
    parser.add_argument("--dpi", type=int, default=230)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = pd.concat(
        [
            _load_summary(Path(args.low_summary), "low_sf"),
            _load_summary(Path(args.high_summary), "high_sf"),
        ],
        ignore_index=True,
    )
    png, pdf = make_figure(
        summary,
        out_dir=Path(args.out_dir),
        out_stem=str(args.out_stem),
        dpi=int(args.dpi),
    )
    combined_csv = Path(args.out_dir) / f"{args.out_stem}_summary.csv"
    summary.to_csv(combined_csv, index=False)
    manifest = Path(args.out_dir) / f"{args.out_stem}_manifest.json"
    _write_json(
        manifest,
        {
            "low_summary": Path(args.low_summary),
            "high_summary": Path(args.high_summary),
            "combined_summary": combined_csv,
            "png": png,
            "pdf": pdf,
            "movement_contract": "pure across or pure along contour-relative eye motion; other component held at trace mean",
            "information_metric": "single-spike information, spike-weighted population percent change vs static center",
        },
    )
    print(combined_csv)
    print(png)
    print(pdf)
    print(manifest)


if __name__ == "__main__":
    main()
