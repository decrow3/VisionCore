#!/usr/bin/env python3
"""Double-check across-vs-along real-trace geometry effects.

This diagnostic contrasts three views of the same aligned high-SF population:

1. The marginal component-path curves used in the story figure.
2. A joint across x along component-path surface.
3. Dominant trace-axis classes matched by total trajectory path.

The purpose is to check whether the weak marginal across-vs-along separation is
real, or whether it is a consequence of marginalizing over the other component.
"""

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
from matplotlib.lines import Line2D

from declan.active_sensing_movie_information.make_backimage_component_2d_surface_diagnostic import (
    MATRIX_DIR,
    OUT_DIR,
    _assign_bins,
    _compute_component_metrics,
    _json_ready,
    _quantile_edges,
)
from declan.active_sensing_movie_information.make_backimage_reordered_geometry_story_figure_cell_baseline_sf075_coh020_cde8bins import (
    _cell_residual_values,
    _selected_unit_images,
)
from declan.active_sensing_movie_information.plot_backimage_real_trace_unit_first_and_population_schematics import (
    axis_delta_deg,
    baseline_rows_by_image,
    load_dataset,
)


OUT_STEM = "backimage_real_trace_across_along_doublecheck_coh020"
STORY_STEM = "backimage_real_trace_geometry_reordered_story_figure_cell_baseline_sf075_coh020_cde8bins"
RELATION = "contour_matched"
SF_GROUP = "high_ge0p75"
N_JOINT_BINS = 4
N_PATH_BINS = 4
TRACE_ANISOTROPY_MIN = 0.50
TRACE_MATCH_MAX_DEG = 22.5
TRACE_ORTHOGONAL_MIN_DEG = 67.5
EPS = 1e-12

ORANGE = "#D55E00"
BLUE = "#0072B2"
GRAY = "0.35"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_range(values: np.ndarray, *, floor: float = 1.0) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -floor, floor
    vmax = max(float(np.nanmax(np.abs(finite))), floor)
    return -vmax, vmax


def _path_quantile_bins(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = _quantile_edges(values, n_bins)
    bins = _assign_bins(values, edges)
    return edges, bins


def _trace_axis_classes(data: dict[str, Any], metrics: pd.DataFrame) -> np.ndarray:
    movie = data["movie"]
    trace = data["trace"].set_index("trace_bank_index")
    trace_index = movie["trace_index"].astype(int)
    image_axis = pd.to_numeric(movie["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=float)
    trace_axis = pd.to_numeric(trace.loc[trace_index, "rendered_cov_orientation_deg"], errors="coerce").to_numpy(dtype=float)
    anis = pd.to_numeric(trace.loc[trace_index, "rendered_cov_anisotropy"], errors="coerce").to_numpy(dtype=float)
    delta = axis_delta_deg(trace_axis, image_axis)

    labels = np.full(metrics.shape[0], "oblique_or_low_anis", dtype=object)
    usable = np.isfinite(delta) & np.isfinite(anis) & (anis >= TRACE_ANISOTROPY_MIN)
    labels[usable & (delta <= TRACE_MATCH_MAX_DEG)] = "along_axis"
    labels[usable & (delta >= TRACE_ORTHOGONAL_MIN_DEG)] = "across_axis"
    labels[usable & (delta > TRACE_MATCH_MAX_DEG) & (delta < TRACE_ORTHOGONAL_MIN_DEG)] = "oblique_axis"
    return labels


def _compute_summary(data: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    values_csv = OUT_DIR / f"{STORY_STEM}_values.csv"
    if values_csv.exists():
        marginal = pd.read_csv(values_csv)
        marginal = marginal[
            (marginal["panel"].astype(str) != "B")
            & marginal["relation"].astype(str).eq(RELATION)
            & marginal["context"].astype(str).eq("drift_only")
        ].copy()
    else:
        marginal = pd.DataFrame()

    metrics = _compute_component_metrics(data)
    drift_mask = metrics["context"].astype(str).eq("drift_only").to_numpy(dtype=bool)
    row_image_index = metrics["image_index"].astype(int).to_numpy()
    baseline_lookup = baseline_rows_by_image(data["image"], data["baseline_table"])
    n_images = int(data["stabilized_ssi"].shape[0])
    unit_to_images = _selected_unit_images(data["unit"], data["image"], sf_group=SF_GROUP, relation=RELATION)

    across_edges, _ = _path_quantile_bins(
        metrics.loc[drift_mask, "across_path_arcmin"].to_numpy(dtype=float),
        N_JOINT_BINS,
    )
    along_edges, _ = _path_quantile_bins(
        metrics.loc[drift_mask, "along_path_arcmin"].to_numpy(dtype=float),
        N_JOINT_BINS,
    )
    across_bins = _assign_bins(metrics["across_path_arcmin"].to_numpy(dtype=float), across_edges)
    along_bins = _assign_bins(metrics["along_path_arcmin"].to_numpy(dtype=float), along_edges)

    rows: list[dict[str, Any]] = []
    joint = np.full((N_JOINT_BINS, N_JOINT_BINS), np.nan, dtype=float)
    joint_counts = np.zeros((N_JOINT_BINS, N_JOINT_BINS), dtype=int)
    for across_bin in range(N_JOINT_BINS):
        for along_bin in range(N_JOINT_BINS):
            row_mask = drift_mask & (across_bins == across_bin) & (along_bins == along_bin)
            values = _cell_residual_values(
                data,
                row_image_index=row_image_index,
                row_mask=row_mask,
                baseline_lookup=baseline_lookup,
                unit_to_images=unit_to_images,
                n_images=n_images,
            )
            joint[across_bin, along_bin] = float(values["ssi_percent_vs_cell_baseline"])
            joint_counts[across_bin, along_bin] = int(np.count_nonzero(row_mask))
            global_rows = metrics[row_mask]
            rows.append(
                {
                    "view": "joint_component_surface",
                    "across_bin": across_bin + 1,
                    "along_bin": along_bin + 1,
                    "n_movie_rows_global": int(np.count_nonzero(row_mask)),
                    "across_median_arcmin": float(np.nanmedian(global_rows["across_path_arcmin"])),
                    "along_median_arcmin": float(np.nanmedian(global_rows["along_path_arcmin"])),
                    **values,
                }
            )

    trace_labels = _trace_axis_classes(data, metrics)
    metrics = metrics.copy()
    metrics["trace_axis_class"] = trace_labels
    total_edges, _ = _path_quantile_bins(
        metrics.loc[drift_mask, "rendered_path_length_arcmin"].to_numpy(dtype=float),
        N_PATH_BINS,
    )
    total_bins = _assign_bins(metrics["rendered_path_length_arcmin"].to_numpy(dtype=float), total_edges)
    metrics["total_path_bin"] = total_bins

    axis_classes = [
        ("along_axis", "along-axis traces"),
        ("across_axis", "across-axis traces"),
        ("oblique_axis", "oblique-axis traces"),
        ("oblique_or_low_anis", "low-anis/other traces"),
    ]
    for axis_class, label in axis_classes:
        row_mask = drift_mask & (trace_labels == axis_class)
        if not np.any(row_mask):
            continue
        values = _cell_residual_values(
            data,
            row_image_index=row_image_index,
            row_mask=row_mask,
            baseline_lookup=baseline_lookup,
            unit_to_images=unit_to_images,
            n_images=n_images,
        )
        global_rows = metrics[row_mask]
        rows.append(
            {
                "view": "trace_axis_aggregate",
                "trace_axis_class": axis_class,
                "trace_axis_label": label,
                "n_movie_rows_global": int(np.count_nonzero(row_mask)),
                "path_median_arcmin": float(np.nanmedian(global_rows["rendered_path_length_arcmin"])),
                "across_median_arcmin": float(np.nanmedian(global_rows["across_path_arcmin"])),
                "along_median_arcmin": float(np.nanmedian(global_rows["along_path_arcmin"])),
                **values,
            }
        )
        for total_bin in range(N_PATH_BINS):
            bin_mask = row_mask & (total_bins == total_bin)
            if np.count_nonzero(bin_mask) == 0:
                continue
            values = _cell_residual_values(
                data,
                row_image_index=row_image_index,
                row_mask=bin_mask,
                baseline_lookup=baseline_lookup,
                unit_to_images=unit_to_images,
                n_images=n_images,
            )
            global_rows = metrics[bin_mask]
            rows.append(
                {
                    "view": "trace_axis_by_total_path",
                    "trace_axis_class": axis_class,
                    "trace_axis_label": label,
                    "total_path_bin": total_bin + 1,
                    "n_movie_rows_global": int(np.count_nonzero(bin_mask)),
                    "path_median_arcmin": float(np.nanmedian(global_rows["rendered_path_length_arcmin"])),
                    "across_median_arcmin": float(np.nanmedian(global_rows["across_path_arcmin"])),
                    "along_median_arcmin": float(np.nanmedian(global_rows["along_path_arcmin"])),
                    **values,
                }
            )

    drift = metrics.loc[drift_mask].copy()
    frac = drift["across_path_arcmin"].to_numpy(dtype=float) / (
        drift["across_path_arcmin"].to_numpy(dtype=float) + drift["along_path_arcmin"].to_numpy(dtype=float) + EPS
    )
    marginal_pivot = pd.DataFrame()
    if not marginal.empty:
        marginal_pivot = marginal.pivot(
            index="component_bin_order",
            columns="component_metric",
            values="ssi_percent_vs_cell_baseline",
        )
    across_conditional = joint[-1, :] - joint[0, :]
    along_conditional = joint[:, -1] - joint[:, 0]
    metadata = {
        "selection": {
            "n_units": int(len(unit_to_images)),
            "n_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
            "sf_group": SF_GROUP,
            "relation": RELATION,
            "contour_coherence_min": 0.20,
            "trace_anisotropy_min": TRACE_ANISOTROPY_MIN,
            "trace_match_max_deg": TRACE_MATCH_MAX_DEG,
            "trace_orthogonal_min_deg": TRACE_ORTHOGONAL_MIN_DEG,
        },
        "joint_edges": {
            "across_path_arcmin": across_edges,
            "along_path_arcmin": along_edges,
        },
        "total_path_edges_arcmin": total_edges,
        "joint_counts": joint_counts,
        "joint_ssi_percent_vs_cell_baseline": joint,
        "conditional_effects_pct_points": {
            "across_q4_minus_q1_within_along_bins": across_conditional,
            "along_q4_minus_q1_within_across_bins": along_conditional,
            "mean_across_effect": float(np.nanmean(across_conditional)),
            "mean_along_effect": float(np.nanmean(along_conditional)),
        },
        "marginal_effects_pct_points": {
            "mean_across_minus_along": float(
                np.nanmean(marginal_pivot["across_path_arcmin"] - marginal_pivot["along_path_arcmin"])
            )
            if {"across_path_arcmin", "along_path_arcmin"}.issubset(marginal_pivot.columns)
            else float("nan"),
            "last_bin_across_minus_along": float(
                (marginal_pivot["across_path_arcmin"] - marginal_pivot["along_path_arcmin"]).iloc[-1]
            )
            if {"across_path_arcmin", "along_path_arcmin"}.issubset(marginal_pivot.columns)
            else float("nan"),
        },
        "drift_component_covariation": {
            "pearson_across_along": float(
                pd.DataFrame(
                    {
                        "across": drift["across_path_arcmin"].to_numpy(dtype=float),
                        "along": drift["along_path_arcmin"].to_numpy(dtype=float),
                    }
                ).corr().iloc[0, 1]
            ),
            "pearson_across_total": float(
                pd.DataFrame(
                    {
                        "across": drift["across_path_arcmin"].to_numpy(dtype=float),
                        "total": drift["rendered_path_length_arcmin"].to_numpy(dtype=float),
                    }
                ).corr().iloc[0, 1]
            ),
            "pearson_along_total": float(
                pd.DataFrame(
                    {
                        "along": drift["along_path_arcmin"].to_numpy(dtype=float),
                        "total": drift["rendered_path_length_arcmin"].to_numpy(dtype=float),
                    }
                ).corr().iloc[0, 1]
            ),
            "across_fraction_quantiles": {
                str(q): float(np.nanquantile(frac, q)) for q in (0.05, 0.25, 0.50, 0.75, 0.95)
            },
        },
    }
    return pd.concat([marginal.assign(view="marginal_component_curves"), pd.DataFrame(rows)], ignore_index=True, sort=False), metadata


def _plot_marginal(ax: plt.Axes, summary: pd.DataFrame) -> None:
    frame = summary[summary["view"].astype(str).eq("marginal_component_curves")].copy()
    styles = {
        "across_path_arcmin": ("across component", "-", "o"),
        "along_path_arcmin": ("along component", (0, (4.2, 2.0)), "s"),
    }
    for metric, (label, ls, marker) in styles.items():
        sub = frame[frame["component_metric"].astype(str).eq(metric)].sort_values("component_bin_order")
        ax.plot(
            sub["component_median_arcmin"],
            sub["ssi_percent_vs_cell_baseline"],
            color=ORANGE,
            linestyle=ls,
            marker=marker,
            linewidth=1.7,
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.1,
            label=label,
        )
    ax.axhline(0.0, color="0.45", linestyle=":", linewidth=0.9)
    ax.set_title("Marginal component bins", fontsize=10.5)
    ax.set_xlabel("component path (arcmin)")
    ax.set_ylabel("SSI residual (% vs cell baseline)")
    ax.grid(True, color="0.9", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)


def _plot_joint(ax: plt.Axes, metadata: dict[str, Any]) -> None:
    grid = np.asarray(metadata["joint_ssi_percent_vs_cell_baseline"], dtype=float)
    vmin, vmax = _finite_range(grid, floor=5.0)
    image = ax.imshow(
        grid,
        origin="lower",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax),
        aspect="auto",
    )
    ax.set_title("Joint component surface", fontsize=10.5)
    ax.set_xlabel("along path quantile")
    ax.set_ylabel("across path quantile")
    ax.set_xticks(range(N_JOINT_BINS), [f"Q{i}" for i in range(1, N_JOINT_BINS + 1)])
    ax.set_yticks(range(N_JOINT_BINS), [f"Q{i}" for i in range(1, N_JOINT_BINS + 1)])
    for row in range(N_JOINT_BINS):
        for col in range(N_JOINT_BINS):
            value = grid[row, col]
            color = "white" if abs(value) > 0.55 * max(abs(vmin), abs(vmax)) else "0.15"
            ax.text(col, row, f"{value:+.1f}", ha="center", va="center", fontsize=8, color=color)
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_label("% vs cell baseline", fontsize=8)


def _plot_trace_axis(ax: plt.Axes, summary: pd.DataFrame) -> None:
    frame = summary[summary["view"].astype(str).eq("trace_axis_by_total_path")].copy()
    plot_specs = [
        ("along_axis", "along-axis traces", BLUE, "^"),
        ("across_axis", "across-axis traces", ORANGE, "o"),
    ]
    for axis_class, label, color, marker in plot_specs:
        sub = frame[frame["trace_axis_class"].astype(str).eq(axis_class)].sort_values("total_path_bin")
        ax.plot(
            sub["path_median_arcmin"],
            sub["ssi_percent_vs_cell_baseline"],
            color=color,
            marker=marker,
            linewidth=1.9,
            markersize=4.6,
            markerfacecolor="white",
            markeredgewidth=1.15,
            label=label,
        )
    ax.axhline(0.0, color="0.45", linestyle=":", linewidth=0.9)
    ax.set_title("Dominant trace axis, total path matched", fontsize=10.5)
    ax.set_xlabel("trajectory path quartile median (arcmin)")
    ax.set_ylabel("SSI residual (% vs cell baseline)")
    ax.grid(True, color="0.9", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)


def _plot_fraction_hist(ax: plt.Axes, data: dict[str, Any]) -> None:
    metrics = _compute_component_metrics(data)
    drift = metrics[metrics["context"].astype(str).eq("drift_only")].copy()
    frac = drift["across_path_arcmin"].to_numpy(dtype=float) / (
        drift["across_path_arcmin"].to_numpy(dtype=float) + drift["along_path_arcmin"].to_numpy(dtype=float) + EPS
    )
    ax.hist(frac[np.isfinite(frac)], bins=np.linspace(0.30, 0.70, 25), color="0.45", alpha=0.82)
    ax.axvline(0.5, color="white", linewidth=1.2)
    ax.set_title("Component balance in drift bank", fontsize=10.5)
    ax.set_xlabel("across / (across + along) path")
    ax.set_ylabel("drift rows")
    ax.grid(True, axis="y", color="0.9", linewidth=0.8)


def _make_figure(data: dict[str, Any], summary: pd.DataFrame, metadata: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.3), constrained_layout=False)
    _plot_marginal(axes[0, 0], summary)
    _plot_joint(axes[0, 1], metadata)
    _plot_trace_axis(axes[1, 0], summary)
    _plot_fraction_hist(axes[1, 1], data)
    fig.subplots_adjust(left=0.095, right=0.94, top=0.875, bottom=0.145, wspace=0.28, hspace=0.40)
    fig.suptitle("Across-vs-along double-check, aligned SF >= 0.75 units", fontsize=14.5)
    fig.text(
        0.5,
        0.035,
        (
            "Contour coherence >= 0.20. Marginal component bins can look similar because both components covary with total path; "
            "the joint surface and trace-axis split recover the across-axis penalty."
        ),
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="0.25",
    )
    return fig


def main() -> None:
    data = load_dataset(MATRIX_DIR)
    summary, metadata = _compute_summary(data)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{OUT_STEM}_values.csv"
    json_path = OUT_DIR / f"{OUT_STEM}_summary.json"
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    summary.to_csv(csv_path, index=False)
    _write_json(json_path, metadata)
    fig = _make_figure(data, summary, metadata)
    fig.savefig(png_path, dpi=230, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    with PdfPages(OUT_DIR / f"{OUT_STEM}_multipage.pdf") as pages:
        pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(png_path)
    print(pdf_path)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
