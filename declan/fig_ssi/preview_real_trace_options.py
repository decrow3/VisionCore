#!/usr/bin/env python3
"""Render real drift trace options for the SSI Panel A schematic."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from declan.fig_ssi.preview_panel_a_luminance_traces import (
    BLUE,
    CYAN,
    GRATING_ANGLE_DEG,
    GRAY,
    HIGH_SF_CPD,
    INK,
    LOW_SF_CPD,
    MODEL_PPD,
    RED,
    TRACE_COMPONENT_METRICS_CSV,
    TRACE_XY_NPY,
    grating_normal,
    grating_patch,
    sampled_luminance_trace,
    sf_cpd_to_wavelength_px,
    temporal_integrate_trace,
)


OUT_BASE = ROOT / "outputs" / "fig_ssi" / "real_trace_pathlength_options"
IMAGE_INDEX = 86
IMAGE_FEATURE_TABLE = TRACE_XY_NPY.parent / "image_feature_table.csv"
SMALL_QUANTILES = [0.05, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15, 0.18]
LONG_QUANTILES = [0.82, 0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.95]
GRATING_SIZE_PX = 220
GRATING_VIEW_HALF_PX = 18.0
SCALE_BAR_ARCMIN = 10.0


def arcmin_to_px(value_arcmin: float) -> float:
    return float(value_arcmin) / 60.0 * MODEL_PPD


def orientation_axis_180(angle_deg: float) -> float:
    return float(float(angle_deg) % 180.0)


def load_panel_a_contour_axis_image_deg(image_index: int = IMAGE_INDEX) -> float:
    if not IMAGE_FEATURE_TABLE.exists():
        return float(GRATING_ANGLE_DEG)
    try:
        images = pd.read_csv(IMAGE_FEATURE_TABLE)
    except Exception:
        return float(GRATING_ANGLE_DEG)
    rows = images[images["image_index"].eq(int(image_index))]
    if rows.empty:
        return float(GRATING_ANGLE_DEG)
    row = rows.iloc[0]
    if "image_edge_axis_array_deg" in row and np.isfinite(float(row["image_edge_axis_array_deg"])):
        return orientation_axis_180(float(row["image_edge_axis_array_deg"]))
    if "image_edge_axis_deg" in row and np.isfinite(float(row["image_edge_axis_deg"])):
        return orientation_axis_180(-float(row["image_edge_axis_deg"]))
    return float(GRATING_ANGLE_DEG)


def load_drift_only_metrics(image_index: int = IMAGE_INDEX) -> pd.DataFrame:
    cols = [
        "image_index",
        "trace_index",
        "has_microsaccade",
        "rendered_path_length_arcmin",
        "across_path_arcmin",
        "along_path_arcmin",
    ]
    metrics = pd.read_csv(TRACE_COMPONENT_METRICS_CSV, usecols=cols)
    sub = metrics[
        metrics["image_index"].eq(int(image_index))
        & ~metrics["has_microsaccade"].astype(bool)
    ].copy()
    sub = sub.drop_duplicates("trace_index").sort_values("rendered_path_length_arcmin")
    return sub.reset_index(drop=True)


def choose_quantile_rows(
    metrics: pd.DataFrame,
    quantiles: list[float],
    *,
    group_label: str,
) -> pd.DataFrame:
    used: set[int] = set()
    rows = []
    metric = metrics["rendered_path_length_arcmin"].to_numpy(dtype=np.float64)
    for q in quantiles:
        target = float(np.nanquantile(metric, float(q)))
        ranked = metrics.assign(_distance=(metrics["rendered_path_length_arcmin"] - target).abs())
        for _, row in ranked.sort_values(["_distance", "trace_index"]).iterrows():
            trace_index = int(row["trace_index"])
            if trace_index not in used:
                used.add(trace_index)
                out = row.drop(labels=["_distance"]).to_dict()
                out["group"] = group_label
                out["quantile"] = float(q)
                rows.append(out)
                break
    return pd.DataFrame(rows)


def trace_to_image_px(trace_xy_deg: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace_xy_deg, dtype=np.float64)
    centered = trace - np.nanmean(trace, axis=0, keepdims=True)
    # Retinal-image displacement convention used by the model-input rendering.
    return np.column_stack([-centered[:, 1], centered[:, 0]]) * MODEL_PPD


def trace_metrics_px(trace_xy_px: np.ndarray) -> dict[str, float]:
    trace = np.asarray(trace_xy_px, dtype=np.float64)
    steps = np.linalg.norm(np.diff(trace, axis=0), axis=1)
    radius = np.linalg.norm(trace, axis=1)
    return {
        "median_step_px": float(np.nanmedian(steps)),
        "rms_step_px": float(np.sqrt(np.nanmean(steps**2))),
        "max_radius_px": float(np.nanmax(radius)),
        "range_x_px": float(np.ptp(trace[:, 0])),
        "range_y_px": float(np.ptp(trace[:, 1])),
    }


def luminance_for_trace(trace_xy_px: np.ndarray, sf_cpd: float, contour_axis_image_deg: float) -> np.ndarray:
    across_px = np.asarray(trace_xy_px, dtype=np.float64) @ grating_normal(contour_axis_image_deg)
    raw = sampled_luminance_trace(across_px, sf_cpd_to_wavelength_px(sf_cpd))
    return temporal_integrate_trace(raw)


def add_trace_path(ax, center: np.ndarray, trace_xy_px: np.ndarray, color: str) -> None:
    points = center[None, :] + np.asarray(trace_xy_px, dtype=np.float64)
    ax.plot(
        points[:, 0],
        points[:, 1],
        color="white",
        lw=4.0,
        alpha=0.90,
        solid_capstyle="round",
        zorder=3,
    )
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        lw=2.3,
        alpha=0.96,
        solid_capstyle="round",
        zorder=4,
    )
    ax.plot(points[0, 0], points[0, 1], marker="o", ms=4.0, mfc="white", mec=color, mew=1.2, zorder=5)
    ax.plot(points[-1, 0], points[-1, 1], marker="o", ms=4.0, mfc=color, mec="white", mew=0.8, zorder=5)


def add_scale_bar(ax, center: np.ndarray, view_half: float) -> None:
    bar_px = arcmin_to_px(SCALE_BAR_ARCMIN)
    x0 = center[0] - 0.82 * view_half
    y0 = center[1] + 0.77 * view_half
    line = ax.plot([x0, x0 + bar_px], [y0, y0], color=INK, lw=1.2, zorder=6)[0]
    line.set_path_effects([patheffects.withStroke(linewidth=2.6, foreground="white")])
    ax.text(
        x0 + 0.5 * bar_px,
        y0 - 0.07 * view_half,
        f"{SCALE_BAR_ARCMIN:g}'",
        fontsize=7.0,
        ha="center",
        va="bottom",
        color=INK,
        path_effects=[patheffects.withStroke(linewidth=2.4, foreground="white")],
        zorder=6,
    )


def add_option_axis(
    ax,
    trace_xy_px: np.ndarray,
    row: pd.Series,
    *,
    color: str,
    contour_axis_image_deg: float,
    view_half: float,
) -> float:
    image = grating_patch(
        GRATING_SIZE_PX,
        sf_cpd_to_wavelength_px(HIGH_SF_CPD),
        angle_deg=contour_axis_image_deg,
    )
    center = np.asarray([0.5 * (GRATING_SIZE_PX - 1), 0.5 * (GRATING_SIZE_PX - 1)], dtype=np.float64)

    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="bicubic")
    theta = np.deg2rad(float(contour_axis_image_deg))
    tangent = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)
    line_half = 0.90 * view_half
    ax.plot(
        [center[0] - tangent[0] * line_half, center[0] + tangent[0] * line_half],
        [center[1] - tangent[1] * line_half, center[1] + tangent[1] * line_half],
        color=CYAN,
        lw=1.8,
        alpha=0.82,
        solid_capstyle="round",
        zorder=2,
    )
    add_trace_path(ax, center, trace_xy_px, color)
    add_scale_bar(ax, center, view_half)

    ax.set_xlim(center[0] - view_half, center[0] + view_half)
    ax.set_ylim(center[1] + view_half, center[1] - view_half)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
        spine.set_linewidth(0.8)

    ax.set_title(
        f"q{float(row['quantile']):.2f}  trace {int(row['trace_index'])}\n"
        f"path {float(row['rendered_path_length_arcmin']):.0f}'",
        fontsize=8.1,
        pad=4,
    )
    return view_half


def add_luminance_axis(ax, values: np.ndarray, *, label: str, color: str) -> None:
    values = np.asarray(values, dtype=np.float64)
    t = np.linspace(0.0, 1.0, values.size)
    ax.axhline(0.50, color="#d4d7dc", lw=0.7, zorder=0)
    ax.plot(t, values, color=color, lw=1.4)
    ax.text(
        -0.03,
        0.50,
        label,
        transform=ax.transAxes,
        fontsize=6.8,
        color=GRAY,
        ha="right",
        va="center",
        clip_on=False,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def main() -> None:
    if not TRACE_XY_NPY.exists() or not TRACE_COMPONENT_METRICS_CSV.exists():
        raise FileNotFoundError("Missing real trace bank outputs.")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    contour_axis_image_deg = load_panel_a_contour_axis_image_deg()
    metrics = load_drift_only_metrics()
    selected = pd.concat(
        [
            choose_quantile_rows(metrics, SMALL_QUANTILES, group_label="small_pathlength"),
            choose_quantile_rows(metrics, LONG_QUANTILES, group_label="long_pathlength"),
        ],
        ignore_index=True,
    )

    trace_xy = np.load(TRACE_XY_NPY, mmap_mode="r")
    traces_px = []
    extra_metrics = []
    for row in selected.itertuples(index=False):
        xy_px = trace_to_image_px(trace_xy[int(row.trace_index)])
        traces_px.append(xy_px)
        high_lum = luminance_for_trace(xy_px, HIGH_SF_CPD, contour_axis_image_deg)
        low_lum = luminance_for_trace(xy_px, LOW_SF_CPD, contour_axis_image_deg)
        metrics_out = trace_metrics_px(xy_px)
        metrics_out.update(
            {
                "contour_axis_image_deg": float(contour_axis_image_deg),
                "high_sf_cpd": float(HIGH_SF_CPD),
                "low_sf_cpd": float(LOW_SF_CPD),
                "high_sf_luminance_range": float(np.nanmax(high_lum) - np.nanmin(high_lum)),
                "low_sf_luminance_range": float(np.nanmax(low_lum) - np.nanmin(low_lum)),
            }
        )
        extra_metrics.append(metrics_out)
    selected = pd.concat([selected, pd.DataFrame(extra_metrics)], axis=1)

    n_cols = max(len(SMALL_QUANTILES), len(LONG_QUANTILES))
    max_trace_extent = max(float(np.nanmax(np.abs(trace_px))) for trace_px in traces_px)
    shared_view_half = min(
        0.49 * GRATING_SIZE_PX,
        max(float(GRATING_VIEW_HALF_PX), max_trace_extent + 4.0),
    )

    fig = plt.figure(figsize=(17.2, 7.2), facecolor="white")
    fig.patch.set_facecolor("white")
    fig.text(0.018, 0.970, "Real drift trace options from bank", fontsize=15.0, weight="bold", ha="left", va="top")
    fig.text(
        0.018,
        0.925,
        f"Drift-only traces, image {IMAGE_INDEX}; candidates chosen by total rendered pathlength. "
        f"All paths share one {shared_view_half / MODEL_PPD * 60.0:.0f}' half-width view, "
        f"matched to Panel A contour axis ({contour_axis_image_deg:.1f} deg). "
        f"Strips show sampled luminance at {HIGH_SF_CPD:g} and {LOW_SF_CPD:g} cpd.",
        fontsize=9.2,
        color="#5f6368",
        ha="left",
        va="top",
    )

    group_labels = {
        "small_pathlength": ("lower quartile", RED),
        "long_pathlength": ("upper quartile", BLUE),
    }
    for row_i, group in enumerate(["small_pathlength", "long_pathlength"]):
        group_rows = selected[selected["group"].eq(group)].reset_index(drop=True)
        label, color = group_labels[group]
        fig.text(0.023, 0.690 - row_i * 0.432, label, fontsize=11.0, weight="bold", color=color, rotation=90, ha="center", va="center")
        for col_i, row in group_rows.iterrows():
            selected_index = selected.index[selected["trace_index"].eq(row["trace_index"]) & selected["group"].eq(group)][0]
            trace_px = traces_px[selected_index]
            left0 = 0.050
            right = 0.992
            col_gap = 0.010
            cell_w = (right - left0 - col_gap * (n_cols - 1.0)) / float(n_cols)
            cell_left = left0 + col_i * (cell_w + col_gap)
            image_w = min(0.082, cell_w)
            image_h = image_w * fig.get_figwidth() / fig.get_figheight()
            image_left = cell_left + 0.5 * (cell_w - image_w)
            row_top = 0.855 - row_i * 0.432
            image_bottom = row_top - image_h
            high_bottom = image_bottom - 0.056
            low_bottom = image_bottom - 0.102
            trace_h = 0.030

            image_ax = fig.add_axes([image_left, image_bottom, image_w, image_h])
            add_option_axis(
                image_ax,
                trace_px,
                row,
                color=color,
                contour_axis_image_deg=contour_axis_image_deg,
                view_half=shared_view_half,
            )
            high_ax = fig.add_axes([cell_left + 0.006, high_bottom, cell_w - 0.012, trace_h])
            low_ax = fig.add_axes([cell_left + 0.006, low_bottom, cell_w - 0.012, trace_h])
            add_luminance_axis(
                high_ax,
                luminance_for_trace(trace_px, HIGH_SF_CPD, contour_axis_image_deg),
                label=f"{HIGH_SF_CPD:g} cpd",
                color=color,
            )
            add_luminance_axis(
                low_ax,
                luminance_for_trace(trace_px, LOW_SF_CPD, contour_axis_image_deg),
                label=f"{LOW_SF_CPD:g} cpd",
                color=color,
            )
            if row_i == 1:
                low_ax.annotate(
                    "",
                    xy=(1.02, -0.22),
                    xytext=(0.0, -0.22),
                    xycoords=("axes fraction", "axes fraction"),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK),
                    clip_on=False,
                )
                low_ax.text(0.50, -0.62, "time", transform=low_ax.transAxes, fontsize=6.8, ha="center", va="top", clip_on=False)

    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(f"{OUT_BASE}.csv", index=False)
    fig.savefig(f"{OUT_BASE}.png", bbox_inches="tight", pad_inches=0.05, dpi=220)
    fig.savefig(f"{OUT_BASE}.pdf", bbox_inches="tight", pad_inches=0.05, dpi=300)
    fig.savefig(f"{OUT_BASE}.svg", bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)
    print(f"{OUT_BASE}.png")
    print(f"{OUT_BASE}.csv")


if __name__ == "__main__":
    main()
