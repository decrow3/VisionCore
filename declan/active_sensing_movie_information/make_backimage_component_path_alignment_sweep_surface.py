#!/usr/bin/env python3
"""2D component path-length surfaces across unit-contour alignment thresholds."""

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
    MATRIX_DIR,
    MIN_OSI,
    N_BINS,
    ORTHOGONAL_MIN_DEG,
    OUT_DIR,
    SF_GROUP,
    _assign_bins,
    _compute_component_metrics,
    _finite_ratio,
    _format_count,
    _json_ready,
    _pct_delta,
    _quantile_edges,
    _symmetric_limits,
)
from declan.active_sensing_movie_information.plot_backimage_real_trace_unit_first_and_population_schematics import (
    accumulate_population,
    accumulate_population_movie_rows,
    baseline_rows_by_image,
    build_movie_row_grid,
    load_dataset,
    unit_image_selection,
)


OUT_STEM = "backimage_real_trace_component_path_alignment_sweep_surface"
MATCH_MAX_DEG_VALUES = (22.5, 15.0, 10.0, 5.0)
RELATION = "contour_matched"
RELATION_LABEL = "High-SF units matched to strong-contour axis"
PATH_FAMILY = {
    "key": "path",
    "title": "Component path length",
    "across": "across_path_arcmin",
    "along": "along_path_arcmin",
    "unit": "arcmin",
    "description": "sum absolute projected frame-to-frame displacement",
}
OUTCOMES = [
    (
        "population_ssi_percent_vs_stabilized",
        "SSI bits/spike\n% vs stabilized",
        "SSI efficiency",
        "RdBu_r",
    ),
    (
        "information_bits_per_sample_percent_vs_stabilized",
        "information bits/window\n% vs stabilized",
        "Information",
        "RdBu_r",
    ),
    (
        "expected_spikes_per_sample_percent_vs_stabilized",
        "expected spikes/window\n% vs stabilized",
        "Expected spikes",
        "RdBu_r",
    ),
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _make_surfaces_for_thresholds(data: dict[str, Any], metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    row_grid = build_movie_row_grid(data["movie"])
    baseline_lookup = baseline_rows_by_image(data["image"], data["baseline_table"])
    row_image_index = metrics["image_index"].astype(int).to_numpy()
    n_images = int(data["stabilized_ssi"].shape[0])

    drift_global = metrics[metrics["context"].astype(str).eq("drift_only")]
    across_col = str(PATH_FAMILY["across"])
    along_col = str(PATH_FAMILY["along"])
    across_edges = _quantile_edges(drift_global[across_col].to_numpy(dtype=float), N_BINS)
    along_edges = _quantile_edges(drift_global[along_col].to_numpy(dtype=float), N_BINS)
    across_bins = _assign_bins(metrics[across_col].to_numpy(dtype=float), across_edges)
    along_bins = _assign_bins(metrics[along_col].to_numpy(dtype=float), along_edges)
    drift_mask = metrics["context"].astype(str).eq("drift_only").to_numpy(dtype=bool)

    rows: list[dict[str, Any]] = []
    threshold_metadata: list[dict[str, Any]] = []
    for match_max_deg in MATCH_MAX_DEG_VALUES:
        selections = unit_image_selection(
            data["unit"],
            data["image"],
            relation=RELATION,
            sf_groups=[SF_GROUP],
            min_osi=MIN_OSI,
            match_max_deg=float(match_max_deg),
            orthogonal_min_deg=ORTHOGONAL_MIN_DEG,
            image_axis_col="image_edge_axis_deg",
        )
        unit_to_images = selections[SF_GROUP]
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
        n_pairs = int(sum(len(images) for images in unit_to_images.values()))
        threshold_metadata.append(
            {
                "match_max_deg": float(match_max_deg),
                "n_units": int(len(unit_to_images)),
                "n_unit_image_pairs": n_pairs,
                "baseline_population_ssi_bits_per_spike": baseline_ssi,
                "baseline_information_bits_per_sample": baseline_info,
                "baseline_expected_spikes_per_sample": baseline_spikes,
            }
        )

        for across_bin in range(N_BINS):
            for along_bin in range(N_BINS):
                row_mask = drift_mask & (across_bins == across_bin) & (along_bins == along_bin)
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
                        "metric_family": PATH_FAMILY["key"],
                        "metric_family_title": PATH_FAMILY["title"],
                        "metric_family_description": PATH_FAMILY["description"],
                        "relation": RELATION,
                        "relation_label": RELATION_LABEL,
                        "sf_group": SF_GROUP,
                        "match_max_deg": float(match_max_deg),
                        "n_units": int(len(unit_to_images)),
                        "n_unit_image_pairs": n_pairs,
                        "across_bin": int(across_bin + 1),
                        "along_bin": int(along_bin + 1),
                        "across_bin_label": f"Q{across_bin + 1}",
                        "along_bin_label": f"Q{along_bin + 1}",
                        "across_min_arcmin": float(across_edges[across_bin]),
                        "across_max_arcmin": float(across_edges[across_bin + 1]),
                        "along_min_arcmin": float(along_edges[along_bin]),
                        "along_max_arcmin": float(along_edges[along_bin + 1]),
                        "across_median_arcmin": float(np.nanmedian(global_rows[across_col])),
                        "along_median_arcmin": float(np.nanmedian(global_rows[along_col])),
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
        "thresholds": threshold_metadata,
        "across_edges_arcmin": across_edges,
        "along_edges_arcmin": along_edges,
    }
    return pd.DataFrame(rows), metadata


def _grid(surface: pd.DataFrame, value_col: str) -> np.ndarray:
    arr = np.full((N_BINS, N_BINS), np.nan, dtype=float)
    for row in surface.itertuples(index=False):
        arr[int(row.across_bin) - 1, int(row.along_bin) - 1] = float(getattr(row, value_col))
    return arr


def _axis_labels(surface: pd.DataFrame, axis: str) -> list[str]:
    labels = []
    for idx in range(1, N_BINS + 1):
        if axis == "across":
            values = surface[surface["across_bin"].eq(idx)]["across_median_arcmin"].to_numpy(dtype=float)
        else:
            values = surface[surface["along_bin"].eq(idx)]["along_median_arcmin"].to_numpy(dtype=float)
        labels.append(f"Q{idx}\n{float(np.nanmedian(values)):.1f}")
    return labels


def _plot_sweep(summary: pd.DataFrame) -> plt.Figure:
    n_rows = len(MATCH_MAX_DEG_VALUES)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14.8, 13.4), constrained_layout=False)
    fig.suptitle(
        "Component path-length surface by unit-contour orientation match",
        fontsize=14.0,
        y=0.992,
    )
    thresholds = list(MATCH_MAX_DEG_VALUES)
    xlabels = _axis_labels(summary[summary["match_max_deg"].eq(thresholds[0])], "along")
    ylabels = _axis_labels(summary[summary["match_max_deg"].eq(thresholds[0])], "across")
    value_limits = {
        value_col: _symmetric_limits(summary[value_col].to_numpy(dtype=float))
        for value_col, _, _, _ in OUTCOMES
    }
    count_values = summary["n_movie_samples"].to_numpy(dtype=float)
    count_vmin = float(np.nanmin(np.log10(np.maximum(count_values, 1.0))))
    count_vmax = float(np.nanmax(np.log10(np.maximum(count_values, 1.0))))

    for row_idx, match_max_deg in enumerate(thresholds):
        surface = summary[summary["match_max_deg"].eq(float(match_max_deg))].copy()
        row_label = (
            f"unit-contour\n<= {match_max_deg:g} deg\n"
            f"{int(surface['n_units'].iloc[0])} units, "
            f"{int(surface['n_unit_image_pairs'].iloc[0])} unit/images"
        )
        for col_idx, (value_col, color_label, title, cmap) in enumerate(OUTCOMES):
            ax = axes[row_idx, col_idx]
            values = _grid(surface, value_col)
            vmin, vmax = value_limits[value_col]
            image = ax.imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            if row_idx == 0:
                ax.set_title(title, fontsize=10.0)
            ax.set_xticks(np.arange(N_BINS), xlabels if row_idx == n_rows - 1 else [])
            ax.set_yticks(np.arange(N_BINS), ylabels)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\n\nacross bin; median arcmin", fontsize=8.1)
            else:
                ax.set_yticklabels([])
            if row_idx == n_rows - 1:
                ax.set_xlabel("along bin; median arcmin", fontsize=8.5)
            ax.tick_params(labelsize=7.0)
            threshold = 0.52 * max(abs(vmin), abs(vmax))
            for across_bin in range(N_BINS):
                for along_bin in range(N_BINS):
                    value = values[across_bin, along_bin]
                    if math.isfinite(value):
                        ax.text(
                            along_bin,
                            across_bin,
                            f"{value:+.0f}",
                            ha="center",
                            va="center",
                            fontsize=6.1,
                            color="white" if abs(value) >= threshold else "0.16",
                        )
            if row_idx == n_rows - 1:
                cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
                cbar.set_label(color_label, fontsize=7.0)
                cbar.ax.tick_params(labelsize=6.7)

        count_ax = axes[row_idx, 3]
        counts = _grid(surface, "n_movie_samples")
        count_image = count_ax.imshow(
            np.log10(np.maximum(counts, 1.0)),
            origin="lower",
            cmap="viridis",
            vmin=count_vmin,
            vmax=count_vmax,
            aspect="auto",
        )
        if row_idx == 0:
            count_ax.set_title("Selected samples", fontsize=10.0)
        count_ax.set_xticks(np.arange(N_BINS), xlabels if row_idx == n_rows - 1 else [])
        count_ax.set_yticks(np.arange(N_BINS), [])
        if row_idx == n_rows - 1:
            count_ax.set_xlabel("along bin; median arcmin", fontsize=8.5)
        count_ax.tick_params(labelsize=7.0)
        median_count = float(np.nanmedian(counts))
        for across_bin in range(N_BINS):
            for along_bin in range(N_BINS):
                count_ax.text(
                    along_bin,
                    across_bin,
                    _format_count(counts[across_bin, along_bin]),
                    ha="center",
                    va="center",
                    fontsize=5.9,
                    color="white" if counts[across_bin, along_bin] >= median_count else "0.12",
                )
        if row_idx == n_rows - 1:
            cbar = fig.colorbar(count_image, ax=count_ax, fraction=0.046, pad=0.025)
            cbar.set_label("log10 samples", fontsize=7.0)
            cbar.ax.tick_params(labelsize=6.7)

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)

    fig.text(
        0.5,
        0.017,
        (
            "Rows increase across-contour unsigned component path; columns increase along-contour unsigned component path. "
            "Bins are fixed global drift-only component quantiles; only unit-image alignment tolerance changes."
        ),
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="0.30",
    )
    fig.subplots_adjust(left=0.105, right=0.992, top=0.955, bottom=0.07, hspace=0.34, wspace=0.28)
    return fig


def _corner_extract(summary: pd.DataFrame) -> pd.DataFrame:
    weird_region = (
        (summary["across_bin"].eq(8) & summary["along_bin"].between(1, 4))
        | (summary["along_bin"].eq(8) & summary["across_bin"].between(1, 4))
        | (summary["across_bin"].eq(1) & summary["along_bin"].between(6, 8))
    )
    cols = [
        "match_max_deg",
        "n_units",
        "n_unit_image_pairs",
        "across_bin",
        "along_bin",
        "across_median_arcmin",
        "along_median_arcmin",
        "population_ssi_percent_vs_stabilized",
        "information_bits_per_sample_percent_vs_stabilized",
        "expected_spikes_per_sample_percent_vs_stabilized",
        "n_movie_samples",
    ]
    return summary.loc[weird_region, cols].sort_values(["match_max_deg", "across_bin", "along_bin"])


def main() -> None:
    data = load_dataset(MATRIX_DIR)
    metrics = _compute_component_metrics(data)
    summary, metadata = _make_surfaces_for_thresholds(data, metrics)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / f"{OUT_STEM}_values.csv"
    corner_csv = OUT_DIR / f"{OUT_STEM}_corner_extract.csv"
    json_path = OUT_DIR / f"{OUT_STEM}_summary.json"
    pdf = OUT_DIR / f"{OUT_STEM}.pdf"
    png = OUT_DIR / f"{OUT_STEM}.png"

    summary.to_csv(csv, index=False)
    corners = _corner_extract(summary)
    corners.to_csv(corner_csv, index=False)
    fig = _plot_sweep(summary)
    with PdfPages(pdf) as pages:
        pages.savefig(fig, bbox_inches="tight")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)

    _write_json(
        json_path,
        {
            "analysis": "backimage_real_trace_component_path_alignment_sweep_surface",
            "matrix_dir": MATRIX_DIR,
            "out_dir": OUT_DIR,
            "outputs": {
                "pdf": pdf,
                "png": png,
                "csv": csv,
                "corner_extract_csv": corner_csv,
                "summary_json": json_path,
            },
            "selection": {
                "relation": RELATION,
                "relation_label": RELATION_LABEL,
                "sf_group": SF_GROUP,
                "min_osi": MIN_OSI,
                "match_max_deg_values": MATCH_MAX_DEG_VALUES,
                "orthogonal_min_deg": ORTHOGONAL_MIN_DEG,
                **metadata,
            },
            "n_bins": N_BINS,
            "metric_family": PATH_FAMILY,
            "note": (
                "This sweep holds the component path-length bins fixed and recomputes the high-SF "
                "strong-contour unit-image selection at each unit-contour axial alignment tolerance."
            ),
        },
    )

    print(pdf)
    print(png)
    print(csv)
    print(corner_csv)
    print(json_path)


if __name__ == "__main__":
    main()
