#!/usr/bin/env python3
"""Plot the high-SF contour-aligned real-trace effect split by trace axis.

This companion figure keeps the Panel-B4 estimand and total path-length x-axis,
then stratifies drift-only image x trace rows by the dominant trace axis relative
to the local image contour. It is meant to audit whether the apparent high-SF
aligned decline is driven by non-across-contour drift directions.
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

from declan.active_sensing_movie_information.plot_backimage_real_trace_unit_first_and_population_schematics import (
    add_equal_count_trace_bins,
    accumulate_population,
    accumulate_population_movie_rows,
    baseline_rows_by_image,
    build_movie_row_grid,
    load_dataset,
    ratio_delta_stats,
    unit_image_selection,
)


MATRIX_DIR = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged"
)
TRACE_PATH_CONTEXT_REFERENCE = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_trace_bank_diffusion_large_fixation_sample_n5000_n40_v1/"
    "filtered_path_length_le350arcmin/trace_bank_metadata_filtered.csv"
)
CONDITION_DIR = MATRIX_DIR / "phase1_phase2_conditioning_v1"
SUMMARY_DIR = CONDITION_DIR / "schematic_pathlength_summary_v1" / "unit_first_and_population_v1"
PHASE1_MOVIE_TABLE = CONDITION_DIR / "phase1_movie_analysis_table.csv"
OUT_DIR = CONDITION_DIR / "plot_collections"
OUT_STEM = "backimage_real_trace_b4_trace_axis_stratified_drift_only"
OVERLAY_STEM = "backimage_real_trace_b4_trace_axis_stratified_drift_only_overlay"

SF_GROUP = "high_sf"
RELATION = "contour_matched"
MIN_OSI = 0.05
MATCH_MAX_DEG = 22.5
ORTHOGONAL_MIN_DEG = 67.5
N_DRIFT_BINS = 8
N_MICROSACCADE_BINS = 5
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 47

X_MIN_POS = 88.0
X_MAX_POS = 180.0
X_TICKS = [0, 90, 105, 120, 150, 175]

PANEL_SPECS = [
    {
        "key": "all_drift",
        "axis_class": None,
        "title": "All drift directions\n(existing B4)",
        "color": "0.36",
        "marker": "D",
    },
    {
        "key": "across_contour_axis",
        "axis_class": "across_contour_axis",
        "title": "Across-contour\ntrace axis",
        "color": "#D55E00",
        "marker": "o",
    },
    {
        "key": "oblique",
        "axis_class": "oblique",
        "title": "Oblique\ntrace axis",
        "color": "#6A51A3",
        "marker": "s",
    },
    {
        "key": "along_contour_axis",
        "axis_class": "along_contour_axis",
        "title": "Along-contour\ntrace axis",
        "color": "#0072B2",
        "marker": "^",
    },
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


def _x_broken_log(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    mapped = np.zeros_like(x, dtype=float)
    positive = x > 0
    span = 5.1
    mapped[positive] = 1.0 + span * np.log(x[positive] / X_MIN_POS) / np.log(X_MAX_POS / X_MIN_POS)
    return mapped


def _format_axis(ax: plt.Axes) -> None:
    ax.set_xlim(-0.12, 5.50)
    ax.set_xticks(_x_broken_log(X_TICKS))
    ax.set_xticklabels([str(int(tick)) for tick in X_TICKS])
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
    ax.grid(True, color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.6)


def _add_percent_change(total: pd.DataFrame) -> pd.DataFrame:
    selected = total[total["relation"].eq(RELATION) & total["sf_group"].eq(SF_GROUP)].copy()
    baseline = float(
        selected.loc[selected["context"].eq("stabilized"), "population_ssi_bits_per_spike"].iloc[0]
    )
    selected["baseline_population_ssi_bits_per_spike"] = baseline
    selected["population_ssi_percent_vs_stabilized"] = (
        100.0 * selected["population_ssi_delta_vs_stabilized"] / baseline
    )
    selected["population_delta_percent_ci95_low_image_boot"] = (
        100.0 * selected["population_delta_ci95_low_image_boot"] / baseline
    )
    selected["population_delta_percent_ci95_high_image_boot"] = (
        100.0 * selected["population_delta_ci95_high_image_boot"] / baseline
    )
    return selected


def _load_existing_b4_drift_rows() -> pd.DataFrame:
    total = _add_percent_change(pd.read_csv(SUMMARY_DIR / "spike_weighted_population_summary.csv"))
    drift = total[total["context"].eq("drift_only")].copy()
    rows = []
    for row in drift.sort_values("path_bin_order").itertuples(index=False):
        rows.append(
            {
                "panel_key": "all_drift",
                "trace_image_axis_class": "all_drift",
                "trace_image_axis_label": "all drift directions",
                "path_bin": str(row.path_bin),
                "path_bin_order": int(row.path_bin_order),
                "path_median_arcmin": float(row.path_median_arcmin),
                "n_trace_bin_traces": int(row.n_traces),
                "n_axis_movie_rows": int(row.n_traces) * 100,
                "n_unique_traces_axis_bin": int(row.n_traces),
                "n_units": int(row.n_units),
                "n_images_contributing": int(row.n_images_contributing),
                "n_movie_samples": int(row.n_movie_samples),
                "population_ssi_bits_per_spike": float(row.population_ssi_bits_per_spike),
                "population_ssi_delta_vs_stabilized": float(row.population_ssi_delta_vs_stabilized),
                "population_ssi_percent_vs_stabilized": float(row.population_ssi_percent_vs_stabilized),
                "population_delta_percent_ci95_low_image_boot": float(
                    row.population_delta_percent_ci95_low_image_boot
                ),
                "population_delta_percent_ci95_high_image_boot": float(
                    row.population_delta_percent_ci95_high_image_boot
                ),
                "population_delta_p_image_bootstrap_sign": float(row.population_delta_p_image_bootstrap_sign),
                "baseline_population_ssi_bits_per_spike": float(row.baseline_population_ssi_bits_per_spike),
                "source": "existing_spike_weighted_population_summary",
            }
        )
    return pd.DataFrame(rows)


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


def _add_drift_regime_band(ax: plt.Axes, context: dict[str, float] | None, *, label: bool = True) -> None:
    if not context:
        return
    low = float(context["q25_arcmin"])
    high = float(context["q75_arcmin"])
    median = float(context["median_arcmin"])
    if not (math.isfinite(low) and math.isfinite(high) and high > low):
        return
    x_low, x_high = _x_broken_log([low, high])
    x_median = float(_x_broken_log([median])[0])
    ax.axvspan(
        float(x_low),
        float(x_high),
        ymin=0.942,
        ymax=0.985,
        facecolor="#8c8c8c",
        edgecolor="none",
        alpha=0.26,
        zorder=0,
    )
    ax.plot(
        [x_median, x_median],
        [0.942, 0.985],
        transform=ax.get_xaxis_transform(),
        color="#666666",
        alpha=0.76,
        linewidth=1.0,
        zorder=1,
    )
    if label:
        ax.text(
            0.985,
            0.988,
            f"normal drift q25-q75 (n={int(context['n_traces'])})",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.3,
            color="0.34",
        )


def _movie_with_trace_bins_and_axis_class(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trace, trace_bins = add_equal_count_trace_bins(
        data["trace"],
        n_drift_bins=N_DRIFT_BINS,
        n_microsaccade_bins=N_MICROSACCADE_BINS,
    )
    axis = pd.read_csv(PHASE1_MOVIE_TABLE, usecols=["movie_index", "trace_image_axis_class"])
    if axis["movie_index"].duplicated().any():
        raise ValueError(f"{PHASE1_MOVIE_TABLE} has duplicate movie_index rows.")
    axis = axis.set_index("movie_index")

    movie = data["movie"][["movie_index", "image_index", "trace_index", "rendered_path_length_arcmin"]].copy()
    movie_index = movie["movie_index"].astype(int)
    missing_axis = ~movie_index.isin(axis.index)
    if bool(missing_axis.any()):
        raise ValueError("Missing trace_image_axis_class for at least one movie row.")
    movie["trace_image_axis_class"] = axis.loc[movie_index, "trace_image_axis_class"].to_numpy()

    trace_lookup = trace.set_index("trace_bank_index")
    trace_index = movie["trace_index"].astype(int)
    movie["context"] = trace_lookup.loc[trace_index, "context"].to_numpy()
    movie["path_bin"] = trace_lookup.loc[trace_index, "path_bin"].to_numpy()
    movie["path_bin_order"] = trace_lookup.loc[trace_index, "path_bin_order"].to_numpy()
    return movie, trace_bins


def _compute_axis_stratified_rows() -> pd.DataFrame:
    data = load_dataset(MATRIX_DIR)
    movie, trace_bins = _movie_with_trace_bins_and_axis_class(data)
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
    baseline_value = float(baseline_pop["population_ssi_bits_per_spike"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    row_image_index = movie["image_index"].astype(int).to_numpy()
    row_context = movie["context"].astype(str).to_numpy()
    row_path_bin = movie["path_bin"].astype(str).to_numpy()
    row_axis_class = movie["trace_image_axis_class"].astype(str).to_numpy()

    rows = []
    drift_bins = trace_bins[trace_bins["context"].astype(str).eq("drift_only")].sort_values("path_bin_order")
    for spec in PANEL_SPECS[1:]:
        axis_class = str(spec["axis_class"])
        for bin_row in drift_bins.itertuples(index=False):
            path_bin = str(bin_row.path_bin)
            row_mask = (
                (row_context == "drift_only")
                & (row_path_bin == path_bin)
                & (row_axis_class == axis_class)
            )
            pop = accumulate_population_movie_rows(
                ssi=data["ssi"],
                expected=data["expected"],
                row_image_index=row_image_index,
                row_mask=row_mask,
                unit_to_images=unit_to_images,
                n_images=n_images,
            )
            value = float(pop["population_ssi_bits_per_spike"])
            delta = value - baseline_value
            delta_stats = ratio_delta_stats(
                pop["per_image_num"],
                pop["per_image_den"],
                baseline_pop["per_image_num"],
                baseline_pop["per_image_den"],
                n_resamples=N_BOOTSTRAP,
                rng=rng,
            )
            finite_delta = math.isfinite(delta)
            rows.append(
                {
                    "panel_key": str(spec["key"]),
                    "trace_image_axis_class": axis_class,
                    "trace_image_axis_label": str(spec["title"]).replace("\n", " "),
                    "path_bin": path_bin,
                    "path_bin_order": int(bin_row.path_bin_order),
                    "path_median_arcmin": float(bin_row.median_path_arcmin),
                    "n_trace_bin_traces": int(bin_row.n_traces),
                    "n_axis_movie_rows": int(np.count_nonzero(row_mask)),
                    "n_unique_traces_axis_bin": int(movie.loc[row_mask, "trace_index"].nunique()),
                    "n_units": int(len(unit_to_images)),
                    "n_images_contributing": int(pop["n_images_contributing"]),
                    "n_movie_samples": int(pop["n_movie_samples"]),
                    "population_ssi_bits_per_spike": value,
                    "population_ssi_delta_vs_stabilized": delta,
                    "population_ssi_percent_vs_stabilized": (
                        100.0 * delta / baseline_value if finite_delta else float("nan")
                    ),
                    "population_delta_percent_ci95_low_image_boot": (
                        100.0 * float(delta_stats["population_delta_ci95_low_image_boot"]) / baseline_value
                    ),
                    "population_delta_percent_ci95_high_image_boot": (
                        100.0 * float(delta_stats["population_delta_ci95_high_image_boot"]) / baseline_value
                    ),
                    "population_delta_p_image_bootstrap_sign": float(
                        delta_stats["population_delta_p_image_bootstrap_sign"]
                    ),
                    "baseline_population_ssi_bits_per_spike": baseline_value,
                    "source": "recomputed_axis_class_row_mask",
                }
            )
    return pd.DataFrame(rows)


def _panel_count_label(rows: pd.DataFrame) -> str:
    counts = pd.to_numeric(rows["n_axis_movie_rows"], errors="coerce").dropna().astype(int)
    if counts.empty:
        return "image-trace/bin n/a"
    if int(counts.min()) == int(counts.max()):
        return f"image-trace/bin {int(counts.iloc[0])}"
    return f"image-trace/bin {int(counts.min())}-{int(counts.max())}"


def _plot(summary: pd.DataFrame) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.5), sharey=True)
    drift_context = _load_drift_path_context()
    for ax, spec in zip(axes, PANEL_SPECS, strict=True):
        _add_drift_regime_band(ax, drift_context, label=bool(spec["key"] == "all_drift"))
        rows = summary[summary["panel_key"].eq(spec["key"])].sort_values("path_bin_order")
        color = str(spec["color"])
        x = _x_broken_log(rows["path_median_arcmin"])
        y = rows["population_ssi_percent_vs_stabilized"].to_numpy(dtype=float)
        ci_low = rows["population_delta_percent_ci95_low_image_boot"].to_numpy(dtype=float)
        ci_high = rows["population_delta_percent_ci95_high_image_boot"].to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(y - ci_low, 0.0), np.maximum(ci_high - y, 0.0)])
        ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            linestyle="-",
            linewidth=1.9,
            marker=str(spec["marker"]),
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.25,
            elinewidth=1.05,
            capsize=2.0,
            zorder=4,
        )
        _format_axis(ax)
        ax.set_title(str(spec["title"]), fontsize=10.8, pad=8, color=color)
        ax.text(
            0.04,
            0.93,
            _panel_count_label(rows),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.4,
            color="0.32",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
        )
        ax.set_xlabel("total drift path length (arcmin)")
    axes[0].set_ylabel("SSI change from shared\nzero-motion baseline (%)")

    vals = []
    for col in [
        "population_ssi_percent_vs_stabilized",
        "population_delta_percent_ci95_low_image_boot",
        "population_delta_percent_ci95_high_image_boot",
    ]:
        arr = pd.to_numeric(summary[col], errors="coerce").to_numpy(dtype=float)
        vals.extend(arr[np.isfinite(arr)].tolist())
    if vals:
        lo = min(0.0, min(vals))
        hi = max(0.0, max(vals))
        span = max(hi - lo, 1.0)
        axes[0].set_ylim(lo - 0.12 * span, hi + 0.18 * span)

    fig.suptitle(
        "Same high-SF contour-aligned units, drift traces split by movement-axis alignment",
        fontsize=14.0,
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "Same strong-contour high-SF unit/image selection as B4; x is total rendered drift path length. "
        "Across/oblique/along classes use the trace covariance axis relative to the local contour; "
        "low-anisotropy traces are included only in the all-directions reference.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="0.25",
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.78, bottom=0.24, wspace=0.22)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{OUT_STEM}.png"
    pdf = OUT_DIR / f"{OUT_STEM}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _plot_overlay(summary: pd.DataFrame) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    drift_context = _load_drift_path_context()
    _add_drift_regime_band(ax, drift_context, label=True)
    reference = summary[summary["panel_key"].eq("all_drift")].sort_values("path_bin_order")
    if not reference.empty:
        ax.plot(
            _x_broken_log(reference["path_median_arcmin"]),
            reference["population_ssi_percent_vs_stabilized"],
            color="0.55",
            linestyle=":",
            linewidth=1.7,
            marker="D",
            markersize=3.9,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label="all drift directions",
            zorder=1,
        )

    offsets = {
        "across_contour_axis": -0.045,
        "oblique": 0.0,
        "along_contour_axis": 0.045,
    }
    labels = {
        "across_contour_axis": "across-contour trace axis",
        "oblique": "oblique trace axis",
        "along_contour_axis": "along-contour trace axis",
    }
    for spec in PANEL_SPECS[1:]:
        rows = summary[summary["panel_key"].eq(spec["key"])].sort_values("path_bin_order")
        color = str(spec["color"])
        x = _x_broken_log(rows["path_median_arcmin"]) + float(offsets[str(spec["key"])])
        y = rows["population_ssi_percent_vs_stabilized"].to_numpy(dtype=float)
        ci_low = rows["population_delta_percent_ci95_low_image_boot"].to_numpy(dtype=float)
        ci_high = rows["population_delta_percent_ci95_high_image_boot"].to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(y - ci_low, 0.0), np.maximum(ci_high - y, 0.0)])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            linestyle="-",
            linewidth=1.95,
            marker=str(spec["marker"]),
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.25,
            elinewidth=1.0,
            capsize=2.0,
            label=labels[str(spec["key"])],
            zorder=4,
        )

    ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
    _format_axis(ax)
    ax.set_ylabel("SSI change from shared\nzero-motion baseline (%)")
    ax.set_xlabel("total drift path length (arcmin)")
    vals = []
    for col in [
        "population_ssi_percent_vs_stabilized",
        "population_delta_percent_ci95_low_image_boot",
        "population_delta_percent_ci95_high_image_boot",
    ]:
        arr = pd.to_numeric(summary[col], errors="coerce").to_numpy(dtype=float)
        vals.extend(arr[np.isfinite(arr)].tolist())
    if vals:
        lo = min(0.0, min(vals))
        hi = max(0.0, max(vals))
        span = max(hi - lo, 1.0)
        ax.set_ylim(lo - 0.12 * span, hi + 0.18 * span)
    ax.set_title(
        "Same high-SF contour-aligned units:\ndrift traces split by movement-axis alignment",
        fontsize=12.4,
        pad=10,
    )
    ax.legend(frameon=False, fontsize=8.0, loc="upper left", ncol=1)
    ax.text(
        0.02,
        0.02,
        "Thin gray: all drift directions. Colored points are horizontally nudged only for readability.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="0.30",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
    )
    fig.tight_layout()
    png = OUT_DIR / f"{OVERLAY_STEM}.png"
    pdf = OUT_DIR / f"{OVERLAY_STEM}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    existing = _load_existing_b4_drift_rows()
    stratified = _compute_axis_stratified_rows()
    summary = pd.concat([existing, stratified], ignore_index=True, sort=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / f"{OUT_STEM}_values.csv"
    json_path = OUT_DIR / f"{OUT_STEM}_summary.json"
    summary.to_csv(csv, index=False)
    png, pdf = _plot(summary)
    overlay_png, overlay_pdf = _plot_overlay(summary)
    _write_json(
        json_path,
        {
        "analysis": "backimage_real_trace_b4_trace_axis_stratified_drift_only",
        "matrix_dir": MATRIX_DIR,
        "phase1_movie_table": PHASE1_MOVIE_TABLE,
        "trace_path_context_reference": TRACE_PATH_CONTEXT_REFERENCE,
        "summary_dir": SUMMARY_DIR,
            "out_dir": OUT_DIR,
            "outputs": {
                "csv": csv,
                "png": png,
                "pdf": pdf,
                "overlay_png": overlay_png,
                "overlay_pdf": overlay_pdf,
                "summary_json": json_path,
            },
            "selection": {
                "relation": RELATION,
                "sf_group": SF_GROUP,
                "min_osi": MIN_OSI,
                "match_max_deg": MATCH_MAX_DEG,
                "orthogonal_min_deg": ORTHOGONAL_MIN_DEG,
            },
            "contracts": {
                "existing_reference": "All-directions panel reuses the drift-only rows from the existing B4 spike-weighted population summary.",
                "axis_panels": "Across/oblique/along panels recompute the same high-SF contour-aligned spike-weighted population after masking image x trace rows by trace_image_axis_class.",
                "x_axis": "All panels use the same global drift-only total path-length bins from trace_path_bin_definitions_8_drift_5_ms.csv.",
                "baseline": "All y-values are percent change from the same non-directional stabilized zero-motion baseline for the selected unit/image set.",
            },
        },
    )
    print(png)
    print(pdf)
    print(overlay_png)
    print(overlay_pdf)
    print(csv)
    print(json_path)


if __name__ == "__main__":
    main()
