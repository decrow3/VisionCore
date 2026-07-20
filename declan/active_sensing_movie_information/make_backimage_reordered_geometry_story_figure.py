#!/usr/bin/env python3
"""Assemble the reordered BackImage real-trace contour-geometry story figure."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib import patches, transforms


MATRIX_DIR = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged"
)
SUMMARY_DIR = (
    MATRIX_DIR
    / "phase1_phase2_conditioning_v1"
    / "schematic_pathlength_summary_v1"
    / "unit_first_and_population_v1"
)
CONDITION_DIR = MATRIX_DIR / "phase1_phase2_conditioning_v1"
FIG_DIR = SUMMARY_DIR / "figures"
OUT_DIR = MATRIX_DIR / "phase1_phase2_conditioning_v1" / "plot_collections"
MOVIE_TABLE = CONDITION_DIR / "phase1_movie_analysis_table.csv"
TRACE_CONTEXT_REFERENCE = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_trace_bank_diffusion_large_fixation_sample_n5000_n40_v1/"
    "filtered_path_length_le350arcmin/trace_bank_metadata_filtered.csv"
)

SF_ORDER = ("low_sf", "middle_sf", "high_sf")
SF_COLORS = {"low_sf": "#0072B2", "middle_sf": "#009E73", "high_sf": "#D55E00"}
SF_LABELS = {"low_sf": "Low SF", "middle_sf": "Middle SF", "high_sf": "High SF"}
AXIS_CLASS_STYLES = {
    "across_contour_axis": ("across contour", "0.18", "-", "o", 4),
    "along_contour_axis": ("along contour", "0.18", "--", "s", 4),
    "oblique": ("oblique", "0.45", "-.", "D", 3),
    "low_anisotropy": ("weak directional bias", "0.68", ":", "^", 2),
}
AXIS_CLASS_ORDER = ("along_contour_axis", "oblique", "across_contour_axis", "low_anisotropy")
COMPONENT_STYLES = {
    "across_path_arcmin": ("across contour", "-", "o"),
    "along_path_arcmin": ("along contour", "--", "s"),
}


def _add_percent_change(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    group_cols = ["relation", "sf_group"]
    if "component_metric" in df.columns:
        group_cols.append("component_metric")
    for _, group in df.groupby(group_cols, sort=False):
        baseline = float(group.loc[group["context"].eq("stabilized"), "population_ssi_bits_per_spike"].iloc[0])
        copy = group.copy()
        copy["population_ssi_percent_vs_stabilized"] = 100.0 * copy["population_ssi_delta_vs_stabilized"] / baseline
        out.append(copy)
    return pd.concat(out, ignore_index=True)


def _sem(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def _load_axis_summary() -> pd.DataFrame:
    movie = pd.read_csv(
        MOVIE_TABLE,
        usecols=[
            "trace_image_axis_class",
            "trace_path_length_bin",
            "rendered_path_length_arcmin",
            "population_ssi",
        ],
    ).dropna(subset=["trace_image_axis_class", "trace_path_length_bin"])
    rows = []
    for (axis_class, trace_bin), group in movie.groupby(
        ["trace_image_axis_class", "trace_path_length_bin"], sort=True
    ):
        rows.append(
            {
                "trace_image_axis_class": str(axis_class),
                "trace_path_length_bin": str(trace_bin),
                "path_median_arcmin": float(np.nanmedian(group["rendered_path_length_arcmin"])),
                "mean_ssi": float(np.nanmean(group["population_ssi"])),
                "sem_ssi": _sem(group["population_ssi"]),
                "n_movies": int(group.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _load_trace_context() -> pd.DataFrame:
    if not TRACE_CONTEXT_REFERENCE.exists():
        return pd.DataFrame()
    reference = pd.read_csv(TRACE_CONTEXT_REFERENCE)
    if "rendered_path_length_arcmin" not in reference.columns:
        return pd.DataFrame()
    if "has_microsaccade" in reference.columns:
        has_ms = reference["has_microsaccade"].fillna(False).astype(bool)
    elif "rendered_n_microsaccade_events" in reference.columns:
        has_ms = pd.to_numeric(reference["rendered_n_microsaccade_events"], errors="coerce").fillna(0).gt(0)
    elif "n_microsaccade_events" in reference.columns:
        has_ms = pd.to_numeric(reference["n_microsaccade_events"], errors="coerce").fillna(0).gt(0)
    else:
        has_ms = pd.Series(False, index=reference.index)
    work = pd.DataFrame(
        {
            "rendered_path_length_arcmin": pd.to_numeric(
                reference["rendered_path_length_arcmin"], errors="coerce"
            ),
            "has_microsaccade": has_ms,
        }
    ).dropna(subset=["rendered_path_length_arcmin"])
    rows = []
    for has_microsaccade, label, display_label in [
        (False, "no_microsaccade", "drift-only"),
        (True, "microsaccade", "microsaccade"),
    ]:
        values = work.loc[work["has_microsaccade"].eq(has_microsaccade), "rendered_path_length_arcmin"].to_numpy(
            dtype=float
        )
        if values.size == 0:
            continue
        rows.append(
            {
                "trace_path_context": label,
                "display_label": display_label,
                "n_traces": int(values.size),
                "q25_arcmin": float(np.nanpercentile(values, 25.0)),
                "median_arcmin": float(np.nanmedian(values)),
                "q75_arcmin": float(np.nanpercentile(values, 75.0)),
            }
        )
    return pd.DataFrame(rows)


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = _add_percent_change(pd.read_csv(SUMMARY_DIR / "spike_weighted_population_summary.csv"))
    comp = _add_percent_change(pd.read_csv(SUMMARY_DIR / "spike_weighted_population_component_summary.csv"))
    axis_summary = _load_axis_summary()
    trace_context = _load_trace_context()
    return total, comp, axis_summary, trace_context


def _x_broken_log(values: np.ndarray | pd.Series | list[float], *, min_pos: float, max_pos: float) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    mapped = np.zeros_like(x, dtype=float)
    positive = x > 0
    span = 5.1
    mapped[positive] = 1.0 + span * np.log(x[positive] / min_pos) / np.log(max_pos / min_pos)
    return mapped


def _format_axis(
    ax: plt.Axes,
    *,
    ticks: list[float],
    min_pos: float,
    max_pos: float,
    show_xlabel: bool = True,
) -> None:
    ax.set_xlim(-0.12, 5.35)
    ax.set_xticks(_x_broken_log(ticks, min_pos=min_pos, max_pos=max_pos))
    ax.set_xticklabels([str(int(tick)) for tick in ticks])
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)
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


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.8)


def _add_trace_context_bands(ax: plt.Axes, context: pd.DataFrame, *, include_legend: bool = True) -> None:
    if context.empty:
        return
    styles = {
        "no_microsaccade": {
            "color": "#8c8c8c",
            "alpha": 0.24,
            "line_alpha": 0.62,
            "linestyle": "-",
            "y0": 0.940,
            "y1": 0.982,
        },
        "microsaccade": {
            "color": "#5f5f5f",
            "alpha": 0.20,
            "line_alpha": 0.70,
            "linestyle": "--",
            "y0": 0.888,
            "y1": 0.930,
        },
    }
    strip_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    legend_handles: list[patches.Patch] = []
    for row in context.sort_values("median_arcmin").itertuples(index=False):
        key = str(getattr(row, "trace_path_context", "trace_path"))
        style = styles.get(key, styles["no_microsaccade"])
        low = float(getattr(row, "q25_arcmin", np.nan))
        high = float(getattr(row, "q75_arcmin", np.nan))
        median = float(getattr(row, "median_arcmin", np.nan))
        if not (math.isfinite(low) and math.isfinite(high) and high > low):
            continue
        y0 = float(style["y0"])
        y1 = float(style["y1"])
        ax.add_patch(
            patches.Rectangle(
                (low, y0),
                high - low,
                y1 - y0,
                transform=strip_transform,
                facecolor=str(style["color"]),
                edgecolor="none",
                alpha=float(style["alpha"]),
                zorder=0,
            )
        )
        if math.isfinite(median):
            ax.plot(
                [median, median],
                [y0, y1],
                transform=strip_transform,
                color=str(style["color"]),
                alpha=float(style["line_alpha"]),
                linestyle=str(style["linestyle"]),
                linewidth=0.9,
                zorder=1,
            )
        if include_legend:
            display = str(getattr(row, "display_label", key)).replace("_", " ")
            n_traces = int(getattr(row, "n_traces", 0))
            legend_handles.append(
                patches.Patch(
                    facecolor=str(style["color"]),
                    alpha=float(style["alpha"]),
                    edgecolor="none",
                    label=f"{display} q25-q75 (n={n_traces})",
                )
            )
    if include_legend and legend_handles:
        existing_handles, existing_labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=legend_handles + existing_handles,
            labels=[h.get_label() for h in legend_handles] + existing_labels,
            frameon=False,
            fontsize=7.7,
            loc="lower right",
        )


def _plot_path_series(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    x_col: str,
    color: str,
    min_pos: float,
    max_pos: float,
    label: str | None = None,
    linestyle: str = "-",
    marker: str = "o",
    linewidth: float = 1.9,
    include_microsaccade: bool = True,
) -> None:
    zero = rows[rows["context"].eq("stabilized")]
    drift = rows[rows["context"].eq("drift_only")].sort_values(x_col)
    ms = rows[rows["context"].eq("microsaccade")].sort_values(x_col)
    if not zero.empty and not drift.empty:
        joined = pd.concat([zero.iloc[:1], drift], ignore_index=True)
        ax.plot(
            _x_broken_log(joined[x_col], min_pos=min_pos, max_pos=max_pos),
            joined["population_ssi_percent_vs_stabilized"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
    elif not drift.empty:
        ax.plot(
            _x_broken_log(drift[x_col], min_pos=min_pos, max_pos=max_pos),
            drift["population_ssi_percent_vs_stabilized"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
    groups = [(drift, False)]
    if include_microsaccade:
        groups.append((ms, True))
    for plot_rows, filled in groups:
        if plot_rows.empty:
            continue
        ax.plot(
            _x_broken_log(plot_rows[x_col], min_pos=min_pos, max_pos=max_pos),
            plot_rows["population_ssi_percent_vs_stabilized"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        ax.scatter(
            _x_broken_log(plot_rows[x_col], min_pos=min_pos, max_pos=max_pos),
            plot_rows["population_ssi_percent_vs_stabilized"],
            marker=marker,
            s=26,
            facecolors=color if filled else "white",
            edgecolors=color,
            linewidths=1.35,
            zorder=4,
        )
    if not zero.empty:
        ax.scatter(
            [0.0],
            [0.0],
            marker=marker,
            s=28,
            facecolors="white",
            edgecolors=color,
            linewidths=1.35,
            zorder=5,
        )


def _panel_a(ax: plt.Axes, axis_summary: pd.DataFrame, trace_context: pd.DataFrame) -> None:
    _add_trace_context_bands(ax, trace_context, include_legend=False)
    for axis_class in AXIS_CLASS_ORDER:
        label, color, linestyle, marker, zorder = AXIS_CLASS_STYLES[axis_class]
        rows = axis_summary[axis_summary["trace_image_axis_class"].eq(axis_class)].sort_values("path_median_arcmin")
        if rows.empty:
            continue
        x = rows["path_median_arcmin"].to_numpy(dtype=float)
        y = rows["mean_ssi"].to_numpy(dtype=float)
        err = rows["sem_ssi"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=err,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.25,
            linewidth=1.8,
            capsize=2,
            alpha=0.95,
            zorder=zorder,
            label=label,
        )
    _style_axis(ax)
    ax.set_title("Information gain varies with gaze trajectory relative to local contours", fontsize=11.3, pad=8)
    ax.set_ylabel("population SSI (bits/spike)")
    ax.set_xlabel("trajectory path length bin median (arcmin)")
    ax.set_xlim(86, 171)
    ax.set_xticks([90, 105, 113, 123, 137, 163])
    y = axis_summary["mean_ssi"].to_numpy(dtype=float)
    err = axis_summary["sem_ssi"].to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(err)
    if np.any(ok):
        lo = float(np.nanmin(y[ok] - err[ok]))
        hi = float(np.nanmax(y[ok] + err[ok]))
        span = max(hi - lo, 0.001)
        ax.set_ylim(lo - 0.10 * span, hi + 0.20 * span)
    ref_handles = [
        patches.Patch(facecolor="#8c8c8c", alpha=0.24, edgecolor="none", label="drift-only q25-q75"),
        patches.Patch(facecolor="#5f5f5f", alpha=0.20, edgecolor="none", label="microsaccade q25-q75"),
    ]
    line_handles, line_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=ref_handles + line_handles,
        labels=[h.get_label() for h in ref_handles] + line_labels,
        title="trajectory relative to contour",
        frameon=False,
        fontsize=7.4,
        title_fontsize=7.8,
        loc="lower right",
    )


def _panel_b(fig: plt.Figure, subspec: object, total: pd.DataFrame) -> list[plt.Axes]:
    sub = subspec.subgridspec(3, 2, hspace=0.18, wspace=0.12)
    axes = [[fig.add_subplot(sub[i, j]) for j in range(2)] for i in range(3)]
    column_specs = [
        ("strong_contours_no_osi", "all units"),
        ("contour_matched", "orientation-aligned units"),
    ]
    for idx, sf_group in enumerate(SF_ORDER):
        row_sources = [
            total[(total["relation"].eq(relation)) & (total["sf_group"].eq(sf_group))].copy()
            for relation, _title in column_specs
        ]
        y = np.concatenate(
            [
                rows["population_ssi_percent_vs_stabilized"].to_numpy(dtype=float)
                for rows in row_sources
                if not rows.empty
            ]
        )
        finite = y[np.isfinite(y)] if y.size else np.asarray([], dtype=float)
        row_ylim: tuple[float, float] | None = None
        if finite.size:
            lo = min(0.0, float(finite.min()))
            hi = max(0.0, float(finite.max()))
            span = max(hi - lo, 1.0)
            row_ylim = (lo - 0.10 * span, hi + 0.14 * span)
        for col_idx, ((relation, column_title), rows) in enumerate(zip(column_specs, row_sources, strict=True)):
            ax = axes[idx][col_idx]
            _plot_path_series(
                ax,
                rows,
                x_col="path_median_arcmin",
                color=SF_COLORS[sf_group],
                min_pos=88.0,
                max_pos=180.0,
                linewidth=1.65,
                include_microsaccade=True,
            )
            ax.axhline(0, color="0.35", lw=0.9, ls=":")
            _format_axis(
                ax,
                ticks=[0, 90, 105, 120, 150, 175],
                min_pos=88.0,
                max_pos=180.0,
                show_xlabel=idx == 2,
            )
            _style_axis(ax)
            if row_ylim is not None:
                ax.set_ylim(*row_ylim)
            if col_idx == 1:
                ax.yaxis.set_visible(False)
                ax.spines["left"].set_visible(False)
            if idx == 0:
                ax.set_title(column_title, fontsize=9.5, pad=6)
            if col_idx == 0:
                label_y = 0.22 if sf_group == "high_sf" else 0.80
                ax.text(
                    0.03,
                    label_y,
                    SF_LABELS[sf_group],
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                    color=SF_COLORS[sf_group],
                    fontsize=8.8,
                    fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.0},
                )
            if idx < 2:
                ax.set_xlabel("")
            elif col_idx == 0:
                ax.set_xlabel("trajectory path length (arcmin)")
            else:
                ax.set_xlabel("")
    axes[0][0].text(
        1.05,
        1.20,
        "Strong contour images: SF bands",
        transform=axes[0][0].transAxes,
        ha="center",
        va="bottom",
        fontsize=11.3,
    )
    axes[1][0].set_ylabel("SSI change (%)")
    return [ax for row in axes for ax in row]


def _panel_component(ax: plt.Axes, comp: pd.DataFrame, *, relation: str, title: str) -> None:
    source = comp[(comp["relation"].eq(relation)) & (comp["sf_group"].eq("high_sf"))].copy()
    ax.axhline(0, color="0.35", lw=0.9, ls=":")
    for metric, (label, linestyle, marker) in COMPONENT_STYLES.items():
        rows = source[source["component_metric"].eq(metric)]
        _plot_path_series(
            ax,
            rows,
            x_col="component_median_arcmin",
            color=SF_COLORS["high_sf"],
            min_pos=45.0,
            max_pos=125.0,
            label=label,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.9,
            include_microsaccade=False,
        )
    _format_axis(ax, ticks=[0, 50, 65, 80, 105, 120], min_pos=45.0, max_pos=125.0)
    _style_axis(ax)
    ax.set_title(title, fontsize=11.3, pad=8)
    ax.set_ylabel("SSI change (%)")
    ax.set_xlabel("component path length (arcmin; log scale after break)")
    y = source["population_ssi_percent_vs_stabilized"].to_numpy(dtype=float)
    finite = y[np.isfinite(y)]
    if finite.size:
        lo = min(0.0, float(finite.min()))
        hi = max(0.0, float(finite.max()))
        span = max(hi - lo, 1.0)
        ax.set_ylim(lo - 0.12 * span, hi + 0.14 * span)
    ax.legend(frameon=False, fontsize=8.0, loc="lower left")


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.09,
        1.10,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
    )


def main() -> None:
    total, comp, axis_summary, trace_context = _load_tables()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16.2, 8.4))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.985,
        top=0.885,
        bottom=0.13,
        hspace=0.46,
        wspace=0.24,
        height_ratios=(1.1, 1.0),
        width_ratios=(1.0, 1.0),
    )
    ax_a = fig.add_subplot(gs[0, 0])
    axes_b = _panel_b(fig, gs[0, 1], total)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _panel_a(ax_a, axis_summary, trace_context)
    _panel_component(ax_c, comp, relation="contour_matched", title="Aligned high-SF units")
    _panel_component(ax_d, comp, relation="contour_orthogonal", title="Orthogonal high-SF units")

    _panel_label(ax_a, "A")
    _panel_label(axes_b[0], "B")
    _panel_label(ax_c, "C")
    _panel_label(ax_d, "D")

    fig.suptitle("Real fixational motion changes SSI according to local contour geometry", fontsize=15.5, y=0.972)
    fig.text(
        0.5,
        0.045,
        "A pools all units and SF groups and plots absolute population SSI by trajectory direction relative to the contour; "
        "B separates SF groups; C-D fix high-SF unit-contour alignment. "
        "Open points are drift-only snippets; filled points show snippets with detected microsaccades where present.",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="0.25",
    )
    png = OUT_DIR / "backimage_real_trace_geometry_reordered_story_figure.png"
    pdf = OUT_DIR / "backimage_real_trace_geometry_reordered_story_figure.pdf"
    fig.savefig(png, dpi=230, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
