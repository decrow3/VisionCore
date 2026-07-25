#!/usr/bin/env python3
"""Sensitivity sheet for the high-SF contour-aligned drift-axis split.

The plotted effect depends on three thresholds:

1. how coherent an image window must be to count as a contour,
2. how closely a high-SF unit must align with that contour,
3. how tightly a drift trace must align with the contour axis to be called
   along/oblique/across.

This script sweeps those three knobs. Each PDF page fixes the contour coherence
threshold; rows vary unit-contour alignment tolerance; columns vary trace-axis
grouping tolerance. Each small panel overlays across/oblique/along trace-axis
classes using the same total drift path-length x-axis.
"""

from __future__ import annotations

import json
import math
import os
import sys
import textwrap
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

from declan.active_sensing_movie_information.plot_backimage_real_trace_unit_first_and_population_schematics import (
    add_equal_count_trace_bins,
    accumulate_population,
    accumulate_population_movie_rows,
    axis_delta_deg,
    baseline_rows_by_image,
    build_movie_row_grid,
    load_dataset,
)


MATRIX_DIR = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged"
)
CONDITION_DIR = MATRIX_DIR / "phase1_phase2_conditioning_v1"
PHASE1_MOVIE_TABLE = CONDITION_DIR / "phase1_movie_analysis_table.csv"
TRACE_PATH_CONTEXT_REFERENCE = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_trace_bank_diffusion_large_fixation_sample_n5000_n40_v1/"
    "filtered_path_length_le350arcmin/trace_bank_metadata_filtered.csv"
)
OUT_DIR = CONDITION_DIR / "plot_collections"
OUT_STEM = "backimage_real_trace_b4_trace_axis_dependency_sweep"

SF_GROUP = "high_sf"
MIN_OSI = 0.05
TRACE_AXIS_MIN_ANISOTROPY = 0.5
N_DRIFT_BINS = 8
N_MICROSACCADE_BINS = 5

CONTOUR_COHERENCE_MINS = [0.35, 0.50, 0.65]
UNIT_ALIGN_MAX_DEGS = [15.0, 22.5, 30.0]
TRACE_AXIS_TOLERANCE_DEGS = [15.0, 22.5, 30.0]

CLASS_SPECS = [
    ("across_contour_axis", "across", "#D55E00", "o"),
    ("oblique", "oblique", "#6A51A3", "s"),
    ("along_contour_axis", "along", "#0072B2", "^"),
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def _load_drift_path_context() -> dict[str, float] | None:
    if not TRACE_PATH_CONTEXT_REFERENCE.exists():
        return None
    reference = pd.read_csv(TRACE_PATH_CONTEXT_REFERENCE)
    if "rendered_path_length_arcmin" not in reference.columns:
        return None
    if "has_microsaccade" in reference.columns:
        has_ms = reference["has_microsaccade"].fillna(False).astype(bool)
    elif "rendered_n_microsaccade_events" in reference.columns:
        has_ms = pd.to_numeric(reference["rendered_n_microsaccade_events"], errors="coerce").fillna(0).gt(0)
    elif "n_microsaccade_events" in reference.columns:
        has_ms = pd.to_numeric(reference["n_microsaccade_events"], errors="coerce").fillna(0).gt(0)
    else:
        has_ms = pd.Series(False, index=reference.index)
    values = pd.to_numeric(reference.loc[~has_ms, "rendered_path_length_arcmin"], errors="coerce").dropna()
    if values.empty:
        return None
    arr = values.to_numpy(dtype=float)
    return {
        "n_traces": int(arr.size),
        "q25_arcmin": float(np.nanpercentile(arr, 25.0)),
        "median_arcmin": float(np.nanmedian(arr)),
        "q75_arcmin": float(np.nanpercentile(arr, 75.0)),
    }


def _add_drift_regime_band(ax: plt.Axes, context: dict[str, float] | None) -> None:
    if not context:
        return
    low = float(context["q25_arcmin"])
    high = float(context["q75_arcmin"])
    median = float(context["median_arcmin"])
    if not (math.isfinite(low) and math.isfinite(high) and high > low and math.isfinite(median)):
        return
    ax.axvspan(low, high, ymin=0.932, ymax=0.978, facecolor="#8c8c8c", edgecolor="none", alpha=0.22, zorder=0)
    ax.plot(
        [median, median],
        [0.932, 0.978],
        transform=ax.get_xaxis_transform(),
        color="#666666",
        alpha=0.72,
        linewidth=0.8,
        zorder=1,
    )


def _movie_with_trace_bins() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_dataset(MATRIX_DIR)
    trace, trace_bins = add_equal_count_trace_bins(
        data["trace"],
        n_drift_bins=N_DRIFT_BINS,
        n_microsaccade_bins=N_MICROSACCADE_BINS,
    )
    phase = pd.read_csv(
        PHASE1_MOVIE_TABLE,
        usecols=["movie_index", "trace_image_axis_delta_deg", "rendered_cov_anisotropy"],
    ).set_index("movie_index")
    movie = data["movie"][["movie_index", "image_index", "trace_index"]].copy()
    movie_index = movie["movie_index"].astype(int)
    missing = ~movie_index.isin(phase.index)
    if bool(missing.any()):
        raise ValueError("Missing phase-1 trace-axis metadata for at least one movie row.")
    movie["trace_image_axis_delta_deg"] = phase.loc[movie_index, "trace_image_axis_delta_deg"].to_numpy(dtype=float)
    movie["rendered_cov_anisotropy"] = phase.loc[movie_index, "rendered_cov_anisotropy"].to_numpy(dtype=float)

    trace_lookup = trace.set_index("trace_bank_index")
    trace_index = movie["trace_index"].astype(int)
    movie["context"] = trace_lookup.loc[trace_index, "context"].to_numpy()
    movie["path_bin"] = trace_lookup.loc[trace_index, "path_bin"].to_numpy()
    movie["path_bin_order"] = trace_lookup.loc[trace_index, "path_bin_order"].to_numpy()
    return movie, trace_bins


def _selected_unit_images(
    *,
    data: dict[str, Any],
    contour_coherence_min: float,
    unit_align_max_deg: float,
) -> dict[int, np.ndarray]:
    image = data["image"]
    unit = data["unit"]
    image_indices = image["image_index"].astype(int).to_numpy()
    image_axis = pd.to_numeric(image["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=float)
    coherence = pd.to_numeric(image["image_orientation_coherence"], errors="coerce").to_numpy(dtype=float)
    contour_mask = np.isfinite(image_axis) & np.isfinite(coherence) & (coherence >= float(contour_coherence_min))

    selected: dict[int, np.ndarray] = {}
    unit_rows = unit[unit["sf_group"].astype(str).eq(SF_GROUP)].copy()
    for row in unit_rows.itertuples(index=False):
        unit_index = int(row.unit_index)
        pref = float(row.prior_preferred_orientation_deg)
        osi = float(row.prior_orientation_selectivity_index)
        if not (math.isfinite(pref) and math.isfinite(osi) and osi >= MIN_OSI):
            continue
        delta = axis_delta_deg(image_axis, pref)
        keep = contour_mask & np.isfinite(delta) & (delta <= float(unit_align_max_deg))
        image_subset = image_indices[keep]
        if image_subset.size:
            selected[unit_index] = image_subset.astype(int)
    return selected


def _trace_class_masks(movie: pd.DataFrame, trace_axis_tolerance_deg: float) -> dict[str, np.ndarray]:
    delta = pd.to_numeric(movie["trace_image_axis_delta_deg"], errors="coerce").to_numpy(dtype=float)
    anis = pd.to_numeric(movie["rendered_cov_anisotropy"], errors="coerce").to_numpy(dtype=float)
    usable = np.isfinite(delta) & np.isfinite(anis) & (anis >= TRACE_AXIS_MIN_ANISOTROPY)
    tol = float(trace_axis_tolerance_deg)
    return {
        "along_contour_axis": usable & (delta <= tol),
        "across_contour_axis": usable & (delta >= 90.0 - tol),
        "oblique": usable & (delta > tol) & (delta < 90.0 - tol),
    }


def _compute_sweep() -> tuple[pd.DataFrame, dict[str, Any]]:
    data = load_dataset(MATRIX_DIR)
    movie, trace_bins = _movie_with_trace_bins()
    row_grid = build_movie_row_grid(data["movie"])
    baseline_lookup = baseline_rows_by_image(data["image"], data["baseline_table"])
    row_image_index = movie["image_index"].astype(int).to_numpy()
    row_context = movie["context"].astype(str).to_numpy()
    row_path_bin = movie["path_bin"].astype(str).to_numpy()
    n_images = int(data["stabilized_ssi"].shape[0])
    drift_bins = trace_bins[trace_bins["context"].astype(str).eq("drift_only")].sort_values("path_bin_order")

    rows: list[dict[str, Any]] = []
    selection_cache: dict[tuple[float, float], dict[int, np.ndarray]] = {}
    baseline_cache: dict[tuple[float, float], dict[str, Any]] = {}
    trace_mask_cache = {
        float(trace_tol): _trace_class_masks(movie, float(trace_tol))
        for trace_tol in TRACE_AXIS_TOLERANCE_DEGS
    }

    for contour_min in CONTOUR_COHERENCE_MINS:
        n_contour_images = int(
            np.count_nonzero(
                pd.to_numeric(data["image"]["image_orientation_coherence"], errors="coerce").to_numpy(dtype=float)
                >= float(contour_min)
            )
        )
        for unit_tol in UNIT_ALIGN_MAX_DEGS:
            cache_key = (float(contour_min), float(unit_tol))
            unit_to_images = _selected_unit_images(
                data=data,
                contour_coherence_min=float(contour_min),
                unit_align_max_deg=float(unit_tol),
            )
            selection_cache[cache_key] = unit_to_images
            baseline_pop = accumulate_population(
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
            baseline_cache[cache_key] = baseline_pop
            baseline_value = float(baseline_pop["population_ssi_bits_per_spike"])
            n_unit_image_pairs = int(sum(len(images) for images in unit_to_images.values()))
            unique_images = np.unique(np.concatenate(list(unit_to_images.values()))) if unit_to_images else np.asarray([])

            for trace_tol in TRACE_AXIS_TOLERANCE_DEGS:
                class_masks = trace_mask_cache[float(trace_tol)]
                for class_key, class_label, _color, _marker in CLASS_SPECS:
                    class_mask = class_masks[class_key]
                    for bin_row in drift_bins.itertuples(index=False):
                        path_bin = str(bin_row.path_bin)
                        row_mask = (row_context == "drift_only") & (row_path_bin == path_bin) & class_mask
                        pop = accumulate_population_movie_rows(
                            ssi=data["ssi"],
                            expected=data["expected"],
                            row_image_index=row_image_index,
                            row_mask=row_mask,
                            unit_to_images=unit_to_images,
                            n_images=n_images,
                        )
                        value = float(pop["population_ssi_bits_per_spike"])
                        delta = value - baseline_value if math.isfinite(baseline_value) else float("nan")
                        rows.append(
                            {
                                "contour_coherence_min": float(contour_min),
                                "unit_align_max_deg": float(unit_tol),
                                "trace_axis_tolerance_deg": float(trace_tol),
                                "trace_axis_orthogonal_min_deg": float(90.0 - float(trace_tol)),
                                "trace_axis_class": class_key,
                                "trace_axis_label": class_label,
                                "path_bin": path_bin,
                                "path_bin_order": int(bin_row.path_bin_order),
                                "path_median_arcmin": float(bin_row.median_path_arcmin),
                                "n_contour_images": n_contour_images,
                                "n_selected_units": int(len(unit_to_images)),
                                "n_selected_unique_images": int(unique_images.size),
                                "n_selected_unit_image_pairs": n_unit_image_pairs,
                                "n_axis_movie_rows_global": int(np.count_nonzero(row_mask)),
                                "n_unique_traces_axis_bin": int(movie.loc[row_mask, "trace_index"].nunique()),
                                "n_images_contributing": int(pop["n_images_contributing"]),
                                "n_movie_samples": int(pop["n_movie_samples"]),
                                "baseline_population_ssi_bits_per_spike": baseline_value,
                                "population_ssi_bits_per_spike": value,
                                "population_ssi_delta_vs_stabilized": delta,
                                "population_ssi_percent_vs_stabilized": (
                                    100.0 * delta / baseline_value
                                    if math.isfinite(delta) and math.isfinite(baseline_value) and baseline_value != 0.0
                                    else float("nan")
                                ),
                                "information_numerator_bits": float(pop["information_numerator_bits"]),
                                "expected_spikes": float(pop["expected_spikes"]),
                            }
                        )
    metadata = {
        "n_rows": len(rows),
        "n_drift_bins": int(drift_bins.shape[0]),
        "selection_cache_size": len(selection_cache),
        "baseline_cache_size": len(baseline_cache),
    }
    return pd.DataFrame(rows), metadata


def _limits(summary: pd.DataFrame) -> tuple[float, float]:
    vals = pd.to_numeric(summary["population_ssi_percent_vs_stabilized"], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -20.0, 20.0
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    span = max(hi - lo, 1.0)
    return lo - 0.10 * span, hi + 0.14 * span


def _plot_page(
    summary: pd.DataFrame,
    *,
    contour_min: float,
    drift_context: dict[str, float] | None,
    ylim: tuple[float, float],
    png_path: Path | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(
        len(UNIT_ALIGN_MAX_DEGS),
        len(TRACE_AXIS_TOLERANCE_DEGS),
        figsize=(10.6, 8.0),
        sharex=True,
        sharey=True,
    )
    xlim = (82.0, 166.0)
    for row_idx, unit_tol in enumerate(UNIT_ALIGN_MAX_DEGS):
        for col_idx, trace_tol in enumerate(TRACE_AXIS_TOLERANCE_DEGS):
            ax = axes[row_idx, col_idx]
            _add_drift_regime_band(ax, drift_context)
            cell = summary[
                summary["contour_coherence_min"].eq(float(contour_min))
                & summary["unit_align_max_deg"].eq(float(unit_tol))
                & summary["trace_axis_tolerance_deg"].eq(float(trace_tol))
            ].copy()
            for class_key, class_label, color, marker in CLASS_SPECS:
                rows = cell[cell["trace_axis_class"].eq(class_key)].sort_values("path_bin_order")
                if rows.empty:
                    continue
                x = rows["path_median_arcmin"].to_numpy(dtype=float)
                y = rows["population_ssi_percent_vs_stabilized"].to_numpy(dtype=float)
                ax.plot(
                    x,
                    y,
                    color=color,
                    marker=marker,
                    markersize=3.0,
                    linewidth=1.25,
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    label=class_label,
                    zorder=3,
                )
            ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.grid(True, color="0.91", linewidth=0.65)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7.4)
            if row_idx == 0:
                ax.set_title(f"trace tol <= {trace_tol:g} deg\nacross >= {90.0 - trace_tol:g} deg", fontsize=8.8)
            if col_idx == 0:
                ax.set_ylabel(f"unit align\n<= {unit_tol:g} deg\nSSI change (%)", fontsize=8.2)
            if row_idx == len(UNIT_ALIGN_MAX_DEGS) - 1:
                ax.set_xlabel("total drift path length (arcmin)", fontsize=8.0)
            first = cell.iloc[0] if not cell.empty else None
            if first is not None:
                ax.text(
                    0.03,
                    0.94,
                    f"u={int(first['n_selected_units'])}, img={int(first['n_selected_unique_images'])}, pairs={int(first['n_selected_unit_image_pairs'])}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.5,
                    color="0.28",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.62, "pad": 0.6},
                )
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.53, 0.035))
    drift_text = ""
    if drift_context:
        drift_text = (
            f"; gray bar: normal drift q25-q75 {drift_context['q25_arcmin']:.1f}-"
            f"{drift_context['q75_arcmin']:.1f} arcmin"
        )
    fig.suptitle(
        "Sensitivity of high-SF contour-aligned trace-axis split\n"
        f"page: image contour coherence >= {contour_min:g}; OSI >= {MIN_OSI:g}; trace anisotropy >= {TRACE_AXIS_MIN_ANISOTROPY:g}{drift_text}",
        fontsize=12.2,
        y=0.982,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.875, bottom=0.10, wspace=0.14, hspace=0.22)
    if png_path is not None:
        fig.savefig(png_path, dpi=230, bbox_inches="tight")
    return fig


def _wrapped_lines(text: str, width: int = 116) -> str:
    return "\n".join(textwrap.fill(part, width=width) for part in text.splitlines())


def _plot_explanation_page(drift_context: dict[str, float] | None) -> plt.Figure:
    fig = plt.figure(figsize=(10.6, 8.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    drift_text = "not available"
    if drift_context:
        drift_text = (
            f"q25-q75 {drift_context['q25_arcmin']:.1f}-{drift_context['q75_arcmin']:.1f} arcmin, "
            f"median {drift_context['median_arcmin']:.1f} arcmin, n={int(drift_context['n_traces'])}"
        )

    sections = [
        (
            "Purpose",
            "This sheet audits whether the high-SF contour-aligned drift-axis split depends on three choices: "
            "how strict the contour image definition is, how close a unit's preferred orientation must be to the "
            "local contour, and how close a drift trace axis must be to along/across/oblique relative to that contour.",
        ),
        (
            "Grid Layout",
            "Each data page fixes the image contour coherence threshold. Rows vary the unit-contour alignment "
            "threshold. Columns vary the trace-axis tolerance. Within each small panel, orange is across-contour, "
            "purple is oblique, and blue is along-contour drift.",
        ),
        (
            "Contour Definition",
            "A contour image window means image_orientation_coherence >= the page threshold and finite "
            "image_edge_axis_deg. This sweep deliberately avoids treating the old image_contour_strong boolean as "
            "a fixed hidden choice.",
        ),
        (
            "Unit Selection",
            f"Units are restricted to high_sf with prior_orientation_selectivity_index >= {MIN_OSI:g}. A unit-image "
            "pair is included when the axial distance between that unit's prior preferred orientation and that "
            "image's contour axis is <= the row threshold. The small label u/img/pairs is selected units, unique "
            "selected image windows, and selected unit-image pairs for that cell.",
        ),
        (
            "Trace-Axis Classes",
            f"Trace classes are assigned per image x trace movie row, because the same trace has a different "
            "relationship to different image contour axes. Only drift-only rows with trace covariance anisotropy "
            f">= {TRACE_AXIS_MIN_ANISOTROPY:g} enter the colored curves. For column tolerance theta: along means "
            "delta <= theta, across means delta >= 90 - theta, and oblique means theta < delta < 90 - theta. "
            "Low-anisotropy or unavailable rows are excluded from the colored curves.",
        ),
        (
            "Axes And Baseline",
            "The x-axis is total rendered drift path-length bin median in arcmin. It is not across/along projected "
            "component path length. The gray top bar is the normal drift-only path-length regime from the larger "
            f"reference trace bank: {drift_text}. The y-axis is spike-weighted population SSI percent change from "
            "the cell's own non-directional stabilized zero-motion baseline.",
        ),
        (
            "Important Caveats",
            "Changing contour and unit-alignment thresholds changes the selected unit/image set, so baselines and "
            "sample counts can change across rows and pages. These panels do not show bootstrap error bars; use "
            "them as a robustness/sign-pattern sheet, and use the CSV for exact counts and values. Sparse cells can "
            "show large excursions, especially in the across-contour curve.",
        ),
    ]

    fig.text(
        0.055,
        0.945,
        "How to read the high-SF contour-aligned trace-axis dependency sweep",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
    )
    y = 0.875
    for title, body in sections:
        fig.text(0.065, y, title, ha="left", va="top", fontsize=11.5, fontweight="bold", color="0.16")
        fig.text(
            0.065,
            y - 0.027,
            _wrapped_lines(body),
            ha="left",
            va="top",
            fontsize=9.4,
            color="0.24",
            linespacing=1.28,
        )
        y -= 0.113 if title not in {"Trace-Axis Classes", "Important Caveats"} else 0.135

    fig.text(
        0.065,
        0.055,
        "Computation: spike-weighted population SSI = sum(unit SSI bits/spike * expected spikes) / sum(expected spikes).",
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="0.28",
    )
    return fig


def _plot_outputs(summary: pd.DataFrame) -> tuple[Path, list[Path]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    drift_context = _load_drift_path_context()
    ylim = _limits(summary)
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    png_paths: list[Path] = []
    with PdfPages(pdf_path) as pdf:
        fig = _plot_explanation_page(drift_context)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        for contour_min in CONTOUR_COHERENCE_MINS:
            png_path = OUT_DIR / f"{OUT_STEM}_contour_coherence_ge_{_slug(contour_min)}.png"
            fig = _plot_page(
                summary,
                contour_min=float(contour_min),
                drift_context=drift_context,
                ylim=ylim,
                png_path=png_path,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            png_paths.append(png_path)
    return pdf_path, png_paths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    values, metadata = _compute_sweep()
    csv_path = OUT_DIR / f"{OUT_STEM}_values.csv"
    values.to_csv(csv_path, index=False)
    pdf_path, png_paths = _plot_outputs(values)
    json_path = OUT_DIR / f"{OUT_STEM}_summary.json"
    _write_json(
        json_path,
        {
            "analysis": "backimage_real_trace_b4_trace_axis_dependency_sweep",
            "matrix_dir": MATRIX_DIR,
            "phase1_movie_table": PHASE1_MOVIE_TABLE,
            "trace_path_context_reference": TRACE_PATH_CONTEXT_REFERENCE,
            "out_dir": OUT_DIR,
            "outputs": {"csv": csv_path, "pdf": pdf_path, "png_pages": png_paths, "summary_json": json_path},
            "sweep": {
                "contour_coherence_mins": CONTOUR_COHERENCE_MINS,
                "unit_align_max_degs": UNIT_ALIGN_MAX_DEGS,
                "trace_axis_tolerance_degs": TRACE_AXIS_TOLERANCE_DEGS,
                "trace_axis_orthogonal_min_degs": [90.0 - value for value in TRACE_AXIS_TOLERANCE_DEGS],
                "sf_group": SF_GROUP,
                "min_osi": MIN_OSI,
                "trace_axis_min_anisotropy": TRACE_AXIS_MIN_ANISOTROPY,
            },
            "contracts": {
                "contour_definition": "A contour image is image_orientation_coherence >= contour_coherence_min with finite image_edge_axis_deg.",
                "unit_alignment": "Selected unit-image pairs are high-SF units with OSI >= min_osi and axial distance from unit preferred orientation to image contour axis <= unit_align_max_deg.",
                "trace_axis_classes": "For finite, sufficiently anisotropic drift traces, along is trace-image axis delta <= trace_axis_tolerance_deg; across is delta >= 90 - trace_axis_tolerance_deg; oblique is between.",
                "baseline": "Each cell uses its own non-directional stabilized zero-motion baseline for the selected unit-image set.",
                "x_axis": "All panels use total rendered drift path length, not projected component path length.",
            },
            "metadata": metadata,
        },
    )
    print(pdf_path)
    for path in png_paths:
        print(path)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
