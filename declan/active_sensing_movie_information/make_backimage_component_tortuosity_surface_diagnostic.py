#!/usr/bin/env python3
"""Component surfaces stratified by path/excursion ratio.

This tests whether the path-length and RMS-excursion surfaces disagree because
large path length can mean either sustained translation or high-tortuosity
jitter with little net excursion.  The third axis is along-contour
path/RMS ratio.
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

from declan.active_sensing_movie_information.make_backimage_component_2d_surface_diagnostic import (
    FAMILIES,
    MATCH_MAX_DEG,
    MIN_OSI,
    ORTHOGONAL_MIN_DEG,
    RELATION,
    RELATION_LABEL,
    SF_GROUP,
    _assign_bins,
    _compute_component_metrics,
    _finite_ratio,
    _pct_delta,
    _quantile_edges,
    _write_json,
)
from declan.active_sensing_movie_information.plot_backimage_real_trace_unit_first_and_population_schematics import (
    accumulate_population,
    accumulate_population_movie_rows,
    baseline_rows_by_image,
    build_movie_row_grid,
    load_dataset,
    unit_image_selection,
)


MATRIX_DIR = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged"
)
CONDITION_DIR = MATRIX_DIR / "phase1_phase2_conditioning_v1"
OUT_DIR = CONDITION_DIR / "plot_collections"
OUT_STEM = "backimage_real_trace_component_tortuosity_surface_diagnostic"

N_COMPONENT_BINS = 4
N_TORTUOSITY_BINS = 3
EPS = 1e-12
TORTUOSITY_COL = "along_path_over_rms"
TORTUOSITY_LABELS = {
    1: "low along tortuosity",
    2: "mid along tortuosity",
    3: "high along tortuosity",
}
OUTCOME_SPECS = [
    (
        "population_ssi_percent_vs_stabilized",
        "SSI bits/spike % vs stabilized",
        "SSI Efficiency",
        "RdBu_r",
        "symmetric",
    ),
    (
        "information_bits_per_sample_percent_vs_stabilized",
        "information bits/window % vs stabilized",
        "Information",
        "RdBu_r",
        "symmetric",
    ),
    (
        "expected_spikes_per_sample_percent_vs_stabilized",
        "expected spikes/window % vs stabilized",
        "Expected Spikes",
        "RdBu_r",
        "symmetric",
    ),
    (
        "n_movie_samples",
        "selected unit-image movie samples",
        "Selected Samples",
        "viridis",
        "log_count",
    ),
]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(val) for val in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _population_metrics(
    pop: dict[str, Any],
    *,
    baseline_ssi: float,
    baseline_information_bits_per_sample: float,
    baseline_expected_spikes_per_sample: float,
) -> dict[str, float]:
    ssi = float(pop["population_ssi_bits_per_spike"])
    info = _finite_ratio(float(pop["information_numerator_bits"]), float(pop["n_movie_samples"]))
    spikes = _finite_ratio(float(pop["expected_spikes"]), float(pop["n_movie_samples"]))
    return {
        "population_ssi_bits_per_spike": ssi,
        "population_ssi_percent_vs_stabilized": _pct_delta(ssi, baseline_ssi),
        "information_bits_per_sample": info,
        "information_bits_per_sample_percent_vs_stabilized": _pct_delta(
            info,
            baseline_information_bits_per_sample,
        ),
        "expected_spikes_per_sample": spikes,
        "expected_spikes_per_sample_percent_vs_stabilized": _pct_delta(
            spikes,
            baseline_expected_spikes_per_sample,
        ),
        "information_numerator_bits": float(pop["information_numerator_bits"]),
        "expected_spikes": float(pop["expected_spikes"]),
    }


def _add_tortuosity(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    out["along_path_over_rms"] = out["along_path_arcmin"] / out["along_rms_arcmin"].where(out["along_rms_arcmin"].abs() > EPS)
    out["across_path_over_rms"] = out["across_path_arcmin"] / out["across_rms_arcmin"].where(out["across_rms_arcmin"].abs() > EPS)
    rms_radius = np.sqrt(
        out["along_rms_arcmin"].to_numpy(dtype=float) ** 2
        + out["across_rms_arcmin"].to_numpy(dtype=float) ** 2
    )
    out["l1_path_over_rms_radius"] = (
        out["along_path_arcmin"].to_numpy(dtype=float) + out["across_path_arcmin"].to_numpy(dtype=float)
    ) / np.where(rms_radius > EPS, rms_radius, np.nan)
    return out


def _make_summary(data: dict[str, Any], metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    row_grid = build_movie_row_grid(data["movie"])
    baseline_lookup = baseline_rows_by_image(data["image"], data["baseline_table"])
    selections = unit_image_selection(
        data["unit"],
        data["image"],
        relation=RELATION,
        sf_groups=[SF_GROUP],
        min_osi=MIN_OSI,
        match_max_deg=MATCH_MAX_DEG,
        orthogonal_min_deg=ORTHOGONAL_MIN_DEG,
        image_axis_col="image_edge_axis_deg",
    )
    unit_to_images = selections[SF_GROUP]
    n_images = int(data["stabilized_ssi"].shape[0])
    baseline = accumulate_population(
        ssi=data["ssi"],
        expected=data["expected"],
        stabilized_ssi=data["stabilized_ssi"],
        stabilized_expected=data["stabilized_expected"],
        row_grid=row_grid,
        baseline_lookup=baseline_lookup,
        unit_to_images=unit_to_images,
        trace_indices=None,
        n_images=n_images,
    )
    baseline_ssi = float(baseline["population_ssi_bits_per_spike"])
    baseline_info = _finite_ratio(float(baseline["information_numerator_bits"]), float(baseline["n_movie_samples"]))
    baseline_spikes = _finite_ratio(float(baseline["expected_spikes"]), float(baseline["n_movie_samples"]))

    drift_global = metrics[metrics["context"].astype(str).eq("drift_only")]
    tort_edges = _quantile_edges(drift_global[TORTUOSITY_COL].to_numpy(dtype=float), N_TORTUOSITY_BINS)
    tort_bins = _assign_bins(metrics[TORTUOSITY_COL].to_numpy(dtype=float), tort_edges)
    drift_mask = metrics["context"].astype(str).eq("drift_only").to_numpy(dtype=bool)
    row_image_index = metrics["image_index"].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        across_col = str(family["across"])
        along_col = str(family["along"])
        across_edges = _quantile_edges(drift_global[across_col].to_numpy(dtype=float), N_COMPONENT_BINS)
        along_edges = _quantile_edges(drift_global[along_col].to_numpy(dtype=float), N_COMPONENT_BINS)
        across_bins = _assign_bins(metrics[across_col].to_numpy(dtype=float), across_edges)
        along_bins = _assign_bins(metrics[along_col].to_numpy(dtype=float), along_edges)
        for tort_bin in range(N_TORTUOSITY_BINS):
            for across_bin in range(N_COMPONENT_BINS):
                for along_bin in range(N_COMPONENT_BINS):
                    row_mask = (
                        drift_mask
                        & (tort_bins == tort_bin)
                        & (across_bins == across_bin)
                        & (along_bins == along_bin)
                    )
                    pop = accumulate_population_movie_rows(
                        ssi=data["ssi"],
                        expected=data["expected"],
                        row_image_index=row_image_index,
                        row_mask=row_mask,
                        unit_to_images=unit_to_images,
                        n_images=n_images,
                    )
                    global_rows = metrics[row_mask]
                    rows.append(
                        {
                            "metric_family": str(family["key"]),
                            "metric_family_title": str(family["title"]),
                            "relation": RELATION,
                            "relation_label": RELATION_LABEL,
                            "sf_group": SF_GROUP,
                            "tortuosity_metric": TORTUOSITY_COL,
                            "tortuosity_bin": int(tort_bin + 1),
                            "tortuosity_label": TORTUOSITY_LABELS[tort_bin + 1],
                            "tortuosity_min": float(tort_edges[tort_bin]),
                            "tortuosity_max": float(tort_edges[tort_bin + 1]),
                            "tortuosity_median": float(np.nanmedian(global_rows[TORTUOSITY_COL])),
                            "across_bin": int(across_bin + 1),
                            "along_bin": int(along_bin + 1),
                            "across_median_arcmin": float(np.nanmedian(global_rows[across_col])),
                            "along_median_arcmin": float(np.nanmedian(global_rows[along_col])),
                            "median_across_path_arcmin": float(np.nanmedian(global_rows["across_path_arcmin"])),
                            "median_along_path_arcmin": float(np.nanmedian(global_rows["along_path_arcmin"])),
                            "median_across_rms_arcmin": float(np.nanmedian(global_rows["across_rms_arcmin"])),
                            "median_along_rms_arcmin": float(np.nanmedian(global_rows["along_rms_arcmin"])),
                            "median_across_path_over_rms": float(np.nanmedian(global_rows["across_path_over_rms"])),
                            "median_along_path_over_rms": float(np.nanmedian(global_rows["along_path_over_rms"])),
                            "median_l1_path_over_rms_radius": float(np.nanmedian(global_rows["l1_path_over_rms_radius"])),
                            "n_movie_rows_global": int(np.count_nonzero(row_mask)),
                            "n_movie_samples": int(pop["n_movie_samples"]),
                            "n_images_contributing": int(pop["n_images_contributing"]),
                            "baseline_population_ssi_bits_per_spike": baseline_ssi,
                            "baseline_information_bits_per_sample": baseline_info,
                            "baseline_expected_spikes_per_sample": baseline_spikes,
                            **_population_metrics(
                                pop,
                                baseline_ssi=baseline_ssi,
                                baseline_information_bits_per_sample=baseline_info,
                                baseline_expected_spikes_per_sample=baseline_spikes,
                            ),
                        }
                    )
    metadata = {
        "n_units": int(len(unit_to_images)),
        "n_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
        "baseline": {
            "population_ssi_bits_per_spike": baseline_ssi,
            "information_bits_per_sample": baseline_info,
            "expected_spikes_per_sample": baseline_spikes,
        },
        "tortuosity_edges": tort_edges,
    }
    return pd.DataFrame(rows), metadata


def _grid(frame: pd.DataFrame, value_col: str) -> np.ndarray:
    out = np.full((N_COMPONENT_BINS, N_COMPONENT_BINS), np.nan, dtype=float)
    for row in frame.itertuples(index=False):
        out[int(row.across_bin) - 1, int(row.along_bin) - 1] = float(getattr(row, value_col))
    return out


def _axis_labels(frame: pd.DataFrame, axis: str) -> list[str]:
    labels = []
    for bin_idx in range(1, N_COMPONENT_BINS + 1):
        if axis == "across":
            values = frame[frame["across_bin"].eq(bin_idx)]["across_median_arcmin"].to_numpy(dtype=float)
        else:
            values = frame[frame["along_bin"].eq(bin_idx)]["along_median_arcmin"].to_numpy(dtype=float)
        labels.append(f"Q{bin_idx}\n{float(np.nanmedian(values)):.1f}")
    return labels


def _format_count(value: float) -> str:
    if not math.isfinite(value):
        return ""
    if value >= 1000:
        return f"{value / 1000.0:.1f}k"
    return f"{int(round(value))}"


def _limits(values: np.ndarray, mode: str) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    if mode == "log_count":
        return float(np.nanmin(finite)), float(np.nanmax(finite))
    mag = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), 1.0)
    return -1.05 * mag, 1.05 * mag


def _plot_page(summary: pd.DataFrame, outcome: tuple[str, str, str, str, str]) -> plt.Figure:
    value_col, color_label, title, cmap, scale_mode = outcome
    fig, axes = plt.subplots(2, N_TORTUOSITY_BINS, figsize=(12.0, 7.35), constrained_layout=False)
    all_values = summary[value_col].to_numpy(dtype=float)
    if scale_mode == "log_count":
        plotted_values = np.log10(np.maximum(all_values, 1.0))
        vmin, vmax = _limits(plotted_values, scale_mode)
    else:
        vmin, vmax = _limits(all_values, scale_mode)
    for row_idx, family in enumerate(FAMILIES):
        family_rows = summary[summary["metric_family"].eq(str(family["key"]))].copy()
        xlabels = _axis_labels(family_rows, "along")
        ylabels = _axis_labels(family_rows, "across")
        for tort_bin in range(1, N_TORTUOSITY_BINS + 1):
            ax = axes[row_idx, tort_bin - 1]
            cell = family_rows[family_rows["tortuosity_bin"].eq(tort_bin)].copy()
            values = _grid(cell, value_col)
            display = np.log10(np.maximum(values, 1.0)) if scale_mode == "log_count" else values
            image = ax.imshow(display, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            tort_med = float(np.nanmedian(cell["tortuosity_median"].to_numpy(dtype=float)))
            ax.set_title(f"{TORTUOSITY_LABELS[tort_bin]}\nmedian path/RMS={tort_med:.1f}", fontsize=9.2)
            ax.set_xticks(np.arange(N_COMPONENT_BINS), xlabels)
            ax.set_yticks(np.arange(N_COMPONENT_BINS), ylabels)
            if row_idx == len(FAMILIES) - 1:
                ax.set_xlabel("along bin; median arcmin", fontsize=8.0)
            if tort_bin == 1:
                ax.set_ylabel(f"{family['title']}\nacross bin; median arcmin", fontsize=8.0)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7.0)
            threshold = 0.52 * max(abs(vmin), abs(vmax))
            for across_bin in range(N_COMPONENT_BINS):
                for along_bin in range(N_COMPONENT_BINS):
                    raw = values[across_bin, along_bin]
                    shown = display[across_bin, along_bin]
                    if not (math.isfinite(raw) and math.isfinite(shown)):
                        continue
                    label = _format_count(raw) if scale_mode == "log_count" else f"{raw:+.1f}"
                    text_color = "white" if (scale_mode == "log_count" and shown > np.nanmedian(display)) or (
                        scale_mode != "log_count" and abs(shown) >= threshold
                    ) else "0.14"
                    ax.text(
                        along_bin,
                        across_bin,
                        label,
                        ha="center",
                        va="center",
                        fontsize=7.0,
                        color=text_color,
                    )
            ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.075, right=0.875, top=0.86, bottom=0.13, wspace=0.22, hspace=0.34)
    cax = fig.add_axes([0.895, 0.22, 0.018, 0.53])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("log10 samples" if scale_mode == "log_count" else color_label, fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.2)
    fig.suptitle(
        f"{title} by component size and along-contour tortuosity",
        fontsize=13.0,
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "Third axis is along-contour path/RMS ratio: low ~= sustained translation, high ~= jittery/tortuous along-contour motion. "
        "All cells use drift-only rows and the aligned high-SF contour-matched spike-weighted population estimand.",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="0.30",
    )
    return fig


def _plot(summary: pd.DataFrame) -> tuple[Path, list[Path]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{OUT_STEM}.pdf"
    pngs: list[Path] = []
    with PdfPages(pdf) as pages:
        for outcome in OUTCOME_SPECS:
            fig = _plot_page(summary, outcome)
            pages.savefig(fig, bbox_inches="tight")
            png = OUT_DIR / f"{OUT_STEM}_{outcome[2].lower().replace(' ', '_')}.png"
            fig.savefig(png, dpi=240, bbox_inches="tight")
            pngs.append(png)
            plt.close(fig)
    return pdf, pngs


def main() -> None:
    data = load_dataset(MATRIX_DIR)
    metrics = _add_tortuosity(_compute_component_metrics(data))
    summary, metadata = _make_summary(data, metrics)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / f"{OUT_STEM}_values.csv"
    json_path = OUT_DIR / f"{OUT_STEM}_summary.json"
    summary.to_csv(csv, index=False)
    pdf, pngs = _plot(summary)
    _write_json(
        json_path,
        {
            "analysis": "backimage_real_trace_component_tortuosity_surface_diagnostic",
            "matrix_dir": MATRIX_DIR,
            "out_dir": OUT_DIR,
            "outputs": {"pdf": pdf, "pngs": pngs, "csv": csv, "summary_json": json_path},
            "selection": {
                "relation": RELATION,
                "relation_label": RELATION_LABEL,
                "sf_group": SF_GROUP,
                "min_osi": MIN_OSI,
                "match_max_deg": MATCH_MAX_DEG,
                "orthogonal_min_deg": ORTHOGONAL_MIN_DEG,
                **metadata,
            },
            "n_component_bins": N_COMPONENT_BINS,
            "n_tortuosity_bins": N_TORTUOSITY_BINS,
            "tortuosity_metric": {
                "column": TORTUOSITY_COL,
                "meaning": "along_path_arcmin / along_rms_arcmin; high means lots of path for little excursion",
            },
            "families": FAMILIES,
        },
    )
    print(pdf)
    for png in pngs:
        print(png)
    print(csv)
    print(json_path)


if __name__ == "__main__":
    main()
