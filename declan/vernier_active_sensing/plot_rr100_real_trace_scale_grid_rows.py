"""Plot RR100 Vernier real-trace scale-grid rows as line curves.

This is a lightweight plotting companion for
``run_rr100_real_trace_scale_grid.py``. It reads the completed summary CSV and
shows across-contour scale on the x-axis, with one curve per selected
along-contour scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path("outputs/notebook_vernier_walkthrough/rr100_real_trace_scale_grid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_RUN_DIR / "rr100_real_trace_scale_grid_summary.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--drop-along-scale",
        type=float,
        default=3.0,
        help="Along-contour row to omit after stride selection.",
    )
    parser.add_argument(
        "--along-scales",
        type=str,
        default="0,0.5,1,2",
        help="Comma-separated along-contour scales to plot. Use empty string for every-second selection.",
    )
    parser.add_argument(
        "--baseline",
        choices=["grid_0_0", "static_center"],
        default="grid_0_0",
        help="Normalization reference for the y-axis.",
    )
    parser.add_argument(
        "--figure-title",
        type=str,
        default="RR100 Vernier real-trace scale grid: absolute loss and incremental motion gain",
    )
    parser.add_argument("--file-prefix", type=str, default="rr100_real_trace_scale_grid")
    parser.add_argument(
        "--ssi-scale",
        choices=["ratio", "absolute"],
        default="ratio",
        help="Use baseline-normalized SSI or absolute SSI bits/spike in the General SSI column.",
    )
    parser.add_argument("--row-title-a", type=str, default="")
    parser.add_argument("--row-title-b", type=str, default="")
    parser.add_argument(
        "--write-absolute-metrics",
        action="store_true",
        help="Also write a one-row figure with raw Fisher and raw SSI values, not baseline ratios.",
    )
    return parser.parse_args()


def _finite_sorted(values: pd.Series) -> list[float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return [float(v) for v in sorted(np.unique(arr))]


def _parse_scale_list(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _select_every_second_along(along_scales: list[float], drop_scale: float) -> list[float]:
    selected = [scale for idx, scale in enumerate(along_scales) if idx % 2 == 0]
    selected = [scale for scale in selected if not np.isclose(scale, drop_scale)]
    if along_scales and not any(np.isclose(scale, 0.0) for scale in selected):
        selected.insert(0, 0.0)
    return selected


def _value_at(df: pd.DataFrame, across: float, along: float, column: str) -> float:
    rows = df[
        np.isclose(pd.to_numeric(df["across_scale"], errors="coerce"), across)
        & np.isclose(pd.to_numeric(df["along_scale"], errors="coerce"), along)
    ]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0][column])


def _grid_values(df: pd.DataFrame, across_scales: list[float], along_scales: list[float], column: str) -> np.ndarray:
    values = np.full((len(along_scales), len(across_scales)), np.nan, dtype=np.float64)
    for y, along in enumerate(along_scales):
        for x, across in enumerate(across_scales):
            values[y, x] = _value_at(df, across, along, column)
    return values


def _annotate_heatmap(ax: Any, values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    span = float(np.nanmax(finite) - np.nanmin(finite))
    cutoff = float(np.nanmin(finite) + 0.38 * span)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if not np.isfinite(value):
                continue
            text_color = "white" if span > 0 and value < cutoff else "black"
            ax.text(x, y, f"{value:.2g}", ha="center", va="center", fontsize=5.6, color=text_color)


def _format_scale(scale: float) -> str:
    return f"{scale:g}x"


def _baseline_row(summary: pd.DataFrame, baseline: str) -> pd.Series:
    if baseline == "static_center":
        rows = summary[summary["condition"].eq("static_center")]
    elif baseline == "grid_0_0":
        rows = summary[
            np.isclose(pd.to_numeric(summary["across_scale"], errors="coerce"), 0.0)
            & np.isclose(pd.to_numeric(summary["along_scale"], errors="coerce"), 0.0)
        ]
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    if rows.empty:
        raise ValueError(f"Could not find baseline row: {baseline}")
    return rows.iloc[0]


def _baseline_label(baseline: str) -> str:
    if baseline == "grid_0_0":
        return "trace-mean static catalog"
    if baseline == "static_center":
        return "centered static oracle"
    return baseline


def _baseline_suffix(baseline: str) -> str:
    return "vs_grid0x0" if baseline == "grid_0_0" else "vs_static_center"


def _add_baseline_ratios(summary: pd.DataFrame, baseline: str) -> pd.DataFrame:
    row = _baseline_row(summary, baseline)
    out = summary.copy()
    for raw_col, ratio_col in [
        ("pose_aware_fisher_mean", "pose_aware_fisher_vs_baseline"),
        ("pose_hidden_fisher", "pose_hidden_fisher_vs_baseline"),
        ("ssi_bits_per_spike_mean", "ssi_bits_per_spike_vs_baseline"),
    ]:
        denom = float(row[raw_col])
        out[ratio_col] = out[raw_col] / denom if denom > 0 else np.nan
    return out


def _metric_specs() -> list[tuple[str, str, str, str]]:
    return [
        (
            "pose_aware_fisher_vs_baseline",
            "Known-trace Fisher",
            "known-trace Fisher / baseline",
            "rr100_known_trace_fisher_baseline_rows_by_along_scale_subset",
        ),
        (
            "pose_hidden_fisher_vs_baseline",
            "Hidden-trace Fisher",
            "hidden-trace Fisher / baseline",
            "rr100_hidden_trace_fisher_baseline_rows_by_along_scale_subset",
        ),
        (
            "ssi_bits_per_spike_vs_baseline",
            "General SSI",
            "SSI / baseline",
            "rr100_ssi_baseline_rows_by_along_scale_subset",
        ),
    ]


def _absolute_metric_specs() -> list[tuple[str, str, str, str]]:
    return [
        (
            "pose_aware_fisher_mean",
            "Known-trace Fisher",
            "known-trace Fisher",
            "rr100_known_trace_fisher_absolute_rows_by_along_scale_subset",
        ),
        (
            "pose_hidden_fisher",
            "Hidden-trace Fisher",
            "hidden-trace Fisher",
            "rr100_hidden_trace_fisher_absolute_rows_by_along_scale_subset",
        ),
        (
            "ssi_bits_per_spike_mean",
            "General SSI",
            "general SSI (bits/spike)",
            "rr100_ssi_absolute_rows_by_along_scale_subset",
        ),
    ]


def _plot_metric_rows(
    ax: Any,
    df: pd.DataFrame,
    *,
    across_scales: list[float],
    along_scales: list[float],
    column: str,
    title: str,
    ylabel: str,
    baseline_label: str,
    baseline_y: float = 1.0,
) -> None:
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(along_scales), 2)))
    x = np.asarray(across_scales, dtype=float)
    for color, along in zip(colors, along_scales, strict=False):
        y = [_value_at(df, across, along, column) for across in across_scales]
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.0,
            linewidth=1.8,
            color=color,
            label=f"along {_format_scale(along)}",
        )
    ax.axhline(float(baseline_y), color="0.45", linestyle="--", linewidth=1.0, label=baseline_label)
    if any(np.isclose(across, 1.0) for across in across_scales):
        ax.axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
    major_ticks = [scale for scale in [0.0, 0.25, 0.5, 1.0, 2.0, 3.0] if any(np.isclose(scale, x))]
    minor_ticks = [scale for scale in across_scales if not any(np.isclose(scale, major) for major in major_ticks)]
    ax.set_xlim(min(across_scales) - 0.05, max(across_scales) + 0.08)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticklabels([f"{scale:g}" for scale in major_ticks])
    ax.tick_params(axis="x", which="minor", length=3, labelbottom=False)
    ax.set_xlabel("across-contour motion scale")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", color="0.88", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _prepared_grid_and_scales(
    summary: pd.DataFrame,
    *,
    baseline: str,
    along_scales: list[float] | None = None,
    drop_along_scale: float = 3.0,
) -> tuple[pd.DataFrame, list[float], list[float]]:
    summary = _add_baseline_ratios(summary, baseline)
    grid = summary[~summary["is_static_baseline"].astype(bool)].copy()
    across_scales = _finite_sorted(grid["across_scale"])
    available_along_scales = _finite_sorted(grid["along_scale"])
    if along_scales is None:
        selected_along_scales = _select_every_second_along(available_along_scales, drop_scale=drop_along_scale)
    else:
        missing = [
            scale
            for scale in along_scales
            if not any(np.isclose(scale, available) for available in available_along_scales)
        ]
        if missing:
            raise ValueError(f"Requested along scales are missing from summary CSV: {missing}")
        selected_along_scales = along_scales
    return grid, across_scales, selected_along_scales


def write_metric_row_figures(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    along_scales: list[float] | None = None,
    drop_along_scale: float = 3.0,
    baseline: str = "grid_0_0",
) -> list[Path]:
    grid, across_scales, along_scales = _prepared_grid_and_scales(
        summary,
        baseline=baseline,
        along_scales=along_scales,
        drop_along_scale=drop_along_scale,
    )
    metric_specs = _metric_specs()
    baseline_label = _baseline_label(baseline)
    suffix = _baseline_suffix(baseline)

    saved: list[Path] = []
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), dpi=220, constrained_layout=True)
    for ax, (column, title, ylabel, _stem) in zip(axes, metric_specs, strict=True):
        _plot_metric_rows(
            ax,
            grid,
            across_scales=across_scales,
            along_scales=along_scales,
            column=column,
            title=title,
            ylabel=ylabel,
            baseline_label=baseline_label,
        )
    axes[-1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "RR100 Vernier real-trace scale grid rows\n"
        f"each curve is an along-contour scale; dashed line is the {baseline_label}",
        fontsize=11,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_png = out_dir / f"rr100_real_trace_scale_grid_rows_subset_by_metric_{suffix}.png"
    panel_pdf = out_dir / f"rr100_real_trace_scale_grid_rows_subset_by_metric_{suffix}.pdf"
    fig.savefig(panel_png, bbox_inches="tight")
    fig.savefig(panel_pdf, bbox_inches="tight")
    plt.close(fig)
    saved.extend([panel_png, panel_pdf])

    for column, title, ylabel, stem in metric_specs:
        fig_single, ax = plt.subplots(figsize=(6.2, 4.2), dpi=220, constrained_layout=True)
        _plot_metric_rows(
            ax,
            grid,
            across_scales=across_scales,
            along_scales=along_scales,
            column=column,
            title=title,
            ylabel=ylabel,
            baseline_label=baseline_label,
        )
        ax.legend(frameon=False, fontsize=8, loc="best")
        png = out_dir / f"{stem}_{suffix}.png"
        pdf = out_dir / f"{stem}_{suffix}.pdf"
        fig_single.savefig(png, bbox_inches="tight")
        fig_single.savefig(pdf, bbox_inches="tight")
        plt.close(fig_single)
        saved.extend([png, pdf])

    return saved


def write_metric_heatmaps(summary: pd.DataFrame, out_dir: Path, *, baseline: str = "grid_0_0") -> list[Path]:
    summary = _add_baseline_ratios(summary, baseline)
    grid = summary[~summary["is_static_baseline"].astype(bool)].copy()
    across_scales = _finite_sorted(grid["across_scale"])
    along_scales = _finite_sorted(grid["along_scale"])
    metric_specs = [
        ("pose_aware_fisher_vs_baseline", "Known-trace Fisher / baseline"),
        ("pose_hidden_fisher_vs_baseline", "Hidden-trace Fisher / baseline"),
        ("ssi_bits_per_spike_vs_baseline", "SSI / baseline"),
    ]
    baseline_label = _baseline_label(baseline)
    suffix = _baseline_suffix(baseline)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), dpi=220, constrained_layout=True)
    for ax, (column, title) in zip(axes, metric_specs, strict=True):
        values = _grid_values(grid, across_scales, along_scales, column)
        im = ax.imshow(values, origin="lower", interpolation="nearest", cmap="viridis")
        _annotate_heatmap(ax, values)
        if 1.0 in across_scales and 1.0 in along_scales:
            ax.scatter(
                [across_scales.index(1.0)],
                [along_scales.index(1.0)],
                marker="x",
                s=36,
                color="white",
                linewidths=1.4,
            )
        ax.set_xticks(np.arange(len(across_scales)))
        ax.set_yticks(np.arange(len(along_scales)))
        ax.set_xticklabels([f"{scale:g}" for scale in across_scales], rotation=45, ha="right")
        ax.set_yticklabels([f"{scale:g}" for scale in along_scales])
        ax.set_xlabel("across-contour scale")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("along-contour scale")
    fig.suptitle(
        f"RR100 real-trace scale grid, relative to {baseline_label}",
        y=1.04,
        fontsize=11,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"rr100_real_trace_scale_grid_heatmaps_{suffix}.png"
    pdf = out_dir / f"rr100_real_trace_scale_grid_heatmaps_{suffix}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _heatmap_limits(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    vmin = min(float(np.nanmin(finite)), 1.0)
    vmax = max(float(np.nanmax(finite)), 1.0)
    if vmin >= 0.0:
        vmin = 0.0 if vmax <= 1.0 else vmin
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def write_two_baseline_row_figure(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    along_scales: list[float] | None = None,
    drop_along_scale: float = 3.0,
    figure_title: str = "RR100 Vernier real-trace scale grid: absolute loss and incremental motion gain",
    file_prefix: str = "rr100_real_trace_scale_grid",
    row_titles: tuple[str, str] | None = None,
    ssi_scale: str = "ratio",
) -> list[Path]:
    titles = row_titles or (
        "A. Absolute information scale",
        "B. Incremental motion gain within same mean-position catalog",
    )
    baselines = [("static_center", titles[0]), ("grid_0_0", titles[1])]
    metric_specs = _metric_specs()
    if str(ssi_scale) == "absolute":
        metric_specs = [
            metric_specs[0],
            metric_specs[1],
            (
                "ssi_bits_per_spike_mean",
                "General SSI (absolute)",
                "general SSI (bits/spike)",
                "rr100_ssi_absolute_rows_by_along_scale_subset",
            ),
        ]
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.1), dpi=220, constrained_layout=True)
    for row_idx, (baseline, row_title) in enumerate(baselines):
        grid, across_scales, selected_along_scales = _prepared_grid_and_scales(
            summary,
            baseline=baseline,
            along_scales=along_scales,
            drop_along_scale=drop_along_scale,
        )
        baseline_label = _baseline_label(baseline)
        for col_idx, (column, title, _ylabel, _stem) in enumerate(metric_specs):
            if str(ssi_scale) == "absolute" and column == "ssi_bits_per_spike_mean":
                ylabel = _ylabel
                baseline_y = float(_baseline_row(summary, baseline)["ssi_bits_per_spike_mean"])
            else:
                ylabel = f"{title.lower()} / {baseline_label}"
                baseline_y = 1.0
            ax = axes[row_idx, col_idx]
            _plot_metric_rows(
                ax,
                grid,
                across_scales=across_scales,
                along_scales=selected_along_scales,
                column=column,
                title=title if row_idx == 0 else "",
                ylabel=ylabel,
                baseline_label=baseline_label,
                baseline_y=baseline_y,
            )
            if col_idx == 0:
                ax.text(
                    -0.12,
                    1.08,
                    row_title,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
            if row_idx == 0 and col_idx == 1:
                ax.set_ylim(bottom=0.0, top=max(1.05, ax.get_ylim()[1]))
            if row_idx == 1 and col_idx == 1:
                ax.set_ylim(bottom=0.0)
    axes[0, -1].legend(frameon=False, fontsize=8, loc="best")
    axes[1, -1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(figure_title, fontsize=12)
    out_dir.mkdir(parents=True, exist_ok=True)
    ssi_suffix = "_absolute_ssi" if str(ssi_scale) == "absolute" else ""
    png = out_dir / f"{file_prefix}{ssi_suffix}_rows_two_baselines.png"
    pdf = out_dir / f"{file_prefix}{ssi_suffix}_rows_two_baselines.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def write_absolute_metric_row_figure(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    along_scales: list[float] | None = None,
    drop_along_scale: float = 3.0,
    figure_title: str = "RR100 Vernier scale grid: absolute metrics",
    file_prefix: str = "rr100_real_trace_scale_grid",
    baseline: str = "static_center",
) -> list[Path]:
    grid, across_scales, selected_along_scales = _prepared_grid_and_scales(
        summary,
        baseline=baseline,
        along_scales=along_scales,
        drop_along_scale=drop_along_scale,
    )
    baseline_label = _baseline_label(baseline)
    baseline_row = _baseline_row(summary, baseline)
    metric_specs = _absolute_metric_specs()
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.35), dpi=220, constrained_layout=True)
    for ax, (column, title, ylabel, _stem) in zip(axes, metric_specs, strict=True):
        _plot_metric_rows(
            ax,
            grid,
            across_scales=across_scales,
            along_scales=selected_along_scales,
            column=column,
            title=title,
            ylabel=ylabel,
            baseline_label=baseline_label,
            baseline_y=float(baseline_row[column]),
        )
    axes[-1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(figure_title, fontsize=12)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{file_prefix}_absolute_metrics_rows.png"
    pdf = out_dir / f"{file_prefix}_absolute_metrics_rows.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def write_two_baseline_heatmap_figure(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    baselines = [
        ("static_center", "A. Relative to centered static oracle"),
        ("grid_0_0", "B. Relative to trace-mean static catalog"),
    ]
    metric_specs = [
        ("pose_aware_fisher_vs_baseline", "Known-trace Fisher"),
        ("pose_hidden_fisher_vs_baseline", "Hidden-trace Fisher"),
        ("ssi_bits_per_spike_vs_baseline", "General SSI"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.4), dpi=220, constrained_layout=True)
    for row_idx, (baseline, row_title) in enumerate(baselines):
        ratio_summary = _add_baseline_ratios(summary, baseline)
        grid = ratio_summary[~ratio_summary["is_static_baseline"].astype(bool)].copy()
        across_scales = _finite_sorted(grid["across_scale"])
        along_scales = _finite_sorted(grid["along_scale"])
        baseline_label = _baseline_label(baseline)
        for col_idx, (column, metric_title) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            values = _grid_values(grid, across_scales, along_scales, column)
            vmin, vmax = _heatmap_limits(values)
            im = ax.imshow(values, origin="lower", interpolation="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
            _annotate_heatmap(ax, values)
            if 1.0 in across_scales and 1.0 in along_scales:
                ax.scatter(
                    [across_scales.index(1.0)],
                    [along_scales.index(1.0)],
                    marker="x",
                    s=36,
                    color="white",
                    linewidths=1.4,
                )
            ax.set_xticks(np.arange(len(across_scales)))
            ax.set_yticks(np.arange(len(along_scales)))
            ax.set_xticklabels([f"{scale:g}" for scale in across_scales], rotation=45, ha="right")
            ax.set_yticklabels([f"{scale:g}" for scale in along_scales])
            ax.set_xlabel("across-contour scale")
            if col_idx == 0:
                ax.set_ylabel("along-contour scale")
                ax.text(
                    -0.18,
                    1.08,
                    row_title,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
            ax.set_title(f"{metric_title} / {baseline_label}", fontsize=9.5)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        "RR100 Vernier real-trace scale grid: two complementary normalizations",
        fontsize=12,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "rr100_real_trace_scale_grid_heatmaps_two_baselines.png"
    pdf = out_dir / "rr100_real_trace_scale_grid_heatmaps_two_baselines.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)
    along_scales = _parse_scale_list(args.along_scales) if str(args.along_scales).strip() else None
    saved = write_metric_row_figures(
        summary,
        args.out_dir,
        along_scales=along_scales,
        drop_along_scale=float(args.drop_along_scale),
        baseline=str(args.baseline),
    )
    saved.extend(write_metric_heatmaps(summary, args.out_dir, baseline=str(args.baseline)))
    saved.extend(
        write_two_baseline_row_figure(
            summary,
            args.out_dir,
            along_scales=along_scales,
            drop_along_scale=float(args.drop_along_scale),
            figure_title=str(args.figure_title),
            file_prefix=str(args.file_prefix),
            row_titles=(
                str(args.row_title_a),
                str(args.row_title_b),
            )
            if str(args.row_title_a).strip() or str(args.row_title_b).strip()
            else None,
            ssi_scale=str(args.ssi_scale),
        )
    )
    if bool(args.write_absolute_metrics):
        saved.extend(
            write_absolute_metric_row_figure(
                summary,
                args.out_dir,
                along_scales=along_scales,
                drop_along_scale=float(args.drop_along_scale),
                figure_title=f"{str(args.figure_title)}: absolute metrics",
                file_prefix=str(args.file_prefix),
                baseline="static_center",
            )
        )
    saved.extend(write_two_baseline_heatmap_figure(summary, args.out_dir))
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
