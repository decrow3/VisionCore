#!/usr/bin/env python3
"""Component RMS-excursion diagnostic for BackImage real-trace SSI.

This is the RMS-excursion counterpart to the earlier C-E component path-length
decomposition. The x-axis is contour-relative displacement spread in arcmin,
not distance traveled. That makes the dose closer to the behavioral
contour-relative RMS variable and avoids treating tortuous path length as the
primary mechanism.
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
    add_equal_count_component_bins,
    accumulate_population,
    accumulate_population_movie_rows,
    baseline_rows_by_image,
    bootstrap_ratio_ci,
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
CONDITION_DIR = MATRIX_DIR / "phase1_phase2_conditioning_v1"
OUT_DIR = CONDITION_DIR / "plot_collections"
OUT_STEM = "backimage_real_trace_component_rms_excursion_diagnostic"

SF_GROUP = "high_sf"
MIN_OSI = 0.05
MATCH_MAX_DEG = 22.5
ORTHOGONAL_MIN_DEG = 67.5
N_DRIFT_BINS = 8
N_MICROSACCADE_BINS = 5
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 47

RELATION_SPECS = [
    ("contour_matched", "Aligned high-SF units"),
    ("contour_intermediate", "Oblique high-SF units"),
    ("contour_orthogonal", "Orthogonal high-SF units"),
]
RMS_SPECS = [
    ("across_rms_arcmin", "across-contour RMS", "#D55E00", "o"),
    ("along_rms_arcmin", "along-contour RMS", "#0072B2", "^"),
]
OUTCOME_SPECS = [
    (
        "population_ssi_percent_vs_stabilized",
        "SSI bits/spike\nchange (%)",
        "efficiency",
    ),
    (
        "information_bits_per_sample_percent_vs_stabilized",
        "information bits/window\nchange (%)",
        "information",
    ),
    (
        "expected_spikes_per_sample_percent_vs_stabilized",
        "expected spikes/window\nchange (%)",
        "rate",
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_ratio(num: float, den: float) -> float:
    return float(num / den) if math.isfinite(num) and math.isfinite(den) and den > 1e-12 else float("nan")


def _pct_delta(value: float, baseline: float) -> float:
    if not (math.isfinite(value) and math.isfinite(baseline) and baseline != 0.0):
        return float("nan")
    return 100.0 * (value - baseline) / baseline


def _compute_rms_metrics(data: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    movie = data["movie"]
    trace_xy = np.asarray(data["trace_xy"], dtype=np.float32)
    trace_index = movie["trace_index"].astype(int).to_numpy()
    axes = pd.to_numeric(movie["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=np.float64)
    theta = np.radians(axes)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xy = trace_xy[trace_index]
    centered = xy - np.nanmean(xy, axis=1, keepdims=True)
    along = centered[:, :, 0] * cos_t[:, None] + centered[:, :, 1] * sin_t[:, None]
    across = -centered[:, :, 0] * sin_t[:, None] + centered[:, :, 1] * cos_t[:, None]
    invalid = ~np.isfinite(axes)

    across_rms = np.sqrt(np.nanmean(across * across, axis=1)) * 60.0
    along_rms = np.sqrt(np.nanmean(along * along, axis=1)) * 60.0
    across_ptp = (np.nanmax(across, axis=1) - np.nanmin(across, axis=1)) * 60.0
    along_ptp = (np.nanmax(along, axis=1) - np.nanmin(along, axis=1)) * 60.0
    for arr in (across_rms, along_rms, across_ptp, along_ptp):
        arr[invalid] = np.nan

    if "rendered_n_microsaccade_events" in movie.columns:
        has_ms = pd.to_numeric(movie["rendered_n_microsaccade_events"], errors="coerce").fillna(0).gt(0)
    else:
        has_ms = pd.Series(False, index=movie.index)
    context = np.where(has_ms.to_numpy(dtype=bool), "microsaccade", "drift_only")
    metrics = pd.DataFrame(
        {
            "movie_index": movie["movie_index"].astype(int).to_numpy(),
            "image_index": movie["image_index"].astype(int).to_numpy(),
            "trace_index": trace_index,
            "has_microsaccade": has_ms.to_numpy(dtype=bool),
            "context": context,
            "context_label": np.where(context == "drift_only", "no detected microsaccade", ">=1 detected microsaccade"),
            "across_rms_arcmin": across_rms,
            "along_rms_arcmin": along_rms,
            "across_peak_to_peak_arcmin": across_ptp,
            "along_peak_to_peak_arcmin": along_ptp,
        }
    )

    bins_by_metric: dict[str, pd.DataFrame] = {}
    work = metrics
    for metric_col, label, _color, _marker in RMS_SPECS:
        work, bins = add_equal_count_component_bins(
            work,
            metric_col=metric_col,
            metric_label=label,
            n_drift_bins=N_DRIFT_BINS,
            n_microsaccade_bins=N_MICROSACCADE_BINS,
        )
        bins_by_metric[metric_col] = bins
    return work, bins_by_metric


def _population_summary_for_relation(
    *,
    relation: str,
    relation_label: str,
    data: dict[str, Any],
    component_metrics: pd.DataFrame,
    bins_by_metric: dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> pd.DataFrame:
    row_grid = build_movie_row_grid(data["movie"])
    baseline_lookup = baseline_rows_by_image(data["image"], data["baseline_table"])
    selections = unit_image_selection(
        data["unit"],
        data["image"],
        relation=relation,
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
    baseline_info_per_sample = _finite_ratio(float(baseline["information_numerator_bits"]), float(baseline["n_movie_samples"]))
    baseline_spikes_per_sample = _finite_ratio(float(baseline["expected_spikes"]), float(baseline["n_movie_samples"]))
    unique_images = np.unique(np.concatenate(list(unit_to_images.values()))) if unit_to_images else np.asarray([])
    row_image_index = component_metrics["image_index"].astype(int).to_numpy()

    rows: list[dict[str, Any]] = []
    for metric_col, metric_label, _color, _marker in RMS_SPECS:
        bin_col = f"{metric_col}_bin"
        bin_defs = bins_by_metric[metric_col]
        for bin_row in bin_defs[bin_defs["context"].astype(str).eq("drift_only")].itertuples(index=False):
            component_bin = str(bin_row.component_bin)
            row_mask = component_metrics[bin_col].astype(str).eq(component_bin).to_numpy(dtype=bool)
            pop = accumulate_population_movie_rows(
                ssi=data["ssi"],
                expected=data["expected"],
                row_image_index=row_image_index,
                row_mask=row_mask,
                unit_to_images=unit_to_images,
                n_images=n_images,
            )
            ci = bootstrap_ratio_ci(
                pop["per_image_num"],
                pop["per_image_den"],
                n_bootstrap=N_BOOTSTRAP,
                rng=rng,
            )
            delta_stats = ratio_delta_stats(
                pop["per_image_num"],
                pop["per_image_den"],
                baseline["per_image_num"],
                baseline["per_image_den"],
                n_resamples=N_BOOTSTRAP,
                rng=rng,
            )
            ssi = float(pop["population_ssi_bits_per_spike"])
            info_per_sample = _finite_ratio(float(pop["information_numerator_bits"]), float(pop["n_movie_samples"]))
            spikes_per_sample = _finite_ratio(float(pop["expected_spikes"]), float(pop["n_movie_samples"]))
            rows.append(
                {
                    "relation": relation,
                    "relation_label": relation_label,
                    "sf_group": SF_GROUP,
                    "component_metric": metric_col,
                    "component_metric_label": metric_label,
                    "component_bin": component_bin,
                    "component_bin_order": int(bin_row.component_bin_order),
                    "component_median_arcmin": float(bin_row.median_component_arcmin),
                    "component_q25_arcmin": float(bin_row.q25_component_arcmin),
                    "component_q75_arcmin": float(bin_row.q75_component_arcmin),
                    "n_units": int(len(unit_to_images)),
                    "n_selected_unique_images": int(unique_images.size),
                    "n_selected_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
                    "n_movie_rows_global": int(bin_row.n_movie_rows_global),
                    "n_unique_traces_global": int(bin_row.n_unique_traces_global),
                    "n_images_contributing": int(pop["n_images_contributing"]),
                    "n_movie_samples": int(pop["n_movie_samples"]),
                    "baseline_population_ssi_bits_per_spike": baseline_ssi,
                    "population_ssi_bits_per_spike": ssi,
                    "population_ssi_percent_vs_stabilized": _pct_delta(ssi, baseline_ssi),
                    "population_ci95_low_image_boot": float(ci[0]),
                    "population_ci95_high_image_boot": float(ci[1]),
                    "population_delta_ci95_low_image_boot": float(delta_stats["population_delta_ci95_low_image_boot"]),
                    "population_delta_ci95_high_image_boot": float(delta_stats["population_delta_ci95_high_image_boot"]),
                    "population_delta_percent_ci95_low_image_boot": (
                        100.0 * float(delta_stats["population_delta_ci95_low_image_boot"]) / baseline_ssi
                    ),
                    "population_delta_percent_ci95_high_image_boot": (
                        100.0 * float(delta_stats["population_delta_ci95_high_image_boot"]) / baseline_ssi
                    ),
                    "population_delta_p_image_bootstrap_sign": float(delta_stats["population_delta_p_image_bootstrap_sign"]),
                    "baseline_information_bits_per_sample": baseline_info_per_sample,
                    "information_bits_per_sample": info_per_sample,
                    "information_bits_per_sample_percent_vs_stabilized": _pct_delta(info_per_sample, baseline_info_per_sample),
                    "baseline_expected_spikes_per_sample": baseline_spikes_per_sample,
                    "expected_spikes_per_sample": spikes_per_sample,
                    "expected_spikes_per_sample_percent_vs_stabilized": _pct_delta(
                        spikes_per_sample,
                        baseline_spikes_per_sample,
                    ),
                    "information_numerator_bits": float(pop["information_numerator_bits"]),
                    "expected_spikes": float(pop["expected_spikes"]),
                }
            )
    return pd.DataFrame(rows)


def _normal_rms_quantiles(component_metrics: pd.DataFrame) -> dict[str, dict[str, float]]:
    drift = component_metrics[component_metrics["context"].astype(str).eq("drift_only")]
    quantiles: dict[str, dict[str, float]] = {}
    for metric_col, _metric_label, _color, _marker in RMS_SPECS:
        values = pd.to_numeric(drift[metric_col], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
        quantiles[metric_col] = {
            "q25_arcmin": float(q25),
            "median_arcmin": float(median),
            "q75_arcmin": float(q75),
            "n_movie_rows": int(values.size),
        }
    return quantiles


def _limits(summary: pd.DataFrame, value_col: str) -> tuple[float, float]:
    vals = pd.to_numeric(summary[value_col], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -10.0, 10.0
    lo = min(0.0, float(np.nanmin(vals)))
    hi = max(0.0, float(np.nanmax(vals)))
    span = max(hi - lo, 1.0)
    return lo - 0.12 * span, hi + 0.16 * span


def _x_limits(summary: pd.DataFrame, normal_quantiles: dict[str, dict[str, float]]) -> tuple[float, float]:
    values = pd.to_numeric(summary["component_median_arcmin"], errors="coerce").to_numpy(dtype=float)
    xvals = [float(value) for value in values if math.isfinite(value)]
    for quantile_summary in normal_quantiles.values():
        xvals.extend(
            value
            for key in ("q25_arcmin", "q75_arcmin")
            if math.isfinite(value := float(quantile_summary.get(key, float("nan"))))
        )
    if not xvals:
        return 0.5, 3.5
    lo = min(xvals)
    hi = max(xvals)
    span = max(hi - lo, 1.0)
    return max(0.0, lo - 0.10 * span), hi + 0.08 * span


def _plot(summary: pd.DataFrame, normal_quantiles: dict[str, dict[str, float]]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(
        len(OUTCOME_SPECS),
        len(RELATION_SPECS),
        figsize=(11.0, 8.55),
        sharex=True,
    )
    xlim = _x_limits(summary, normal_quantiles)
    for row_idx, (value_col, ylabel, _key) in enumerate(OUTCOME_SPECS):
        ylim = _limits(summary, value_col)
        for col_idx, (relation, relation_label) in enumerate(RELATION_SPECS):
            ax = axes[row_idx, col_idx]
            cell = summary[summary["relation"].eq(relation)].copy()
            for metric_col, metric_label, color, marker in RMS_SPECS:
                rows = cell[cell["component_metric"].eq(metric_col)].sort_values("component_bin_order")
                x = rows["component_median_arcmin"].to_numpy(dtype=float)
                y = rows[value_col].to_numpy(dtype=float)
                ax.plot(
                    x,
                    y,
                    color=color,
                    marker=marker,
                    markersize=4.2,
                    linewidth=1.85,
                    markerfacecolor="white",
                    markeredgewidth=1.1,
                    label=metric_label,
                    zorder=3,
                )
                if value_col == "population_ssi_percent_vs_stabilized":
                    ci_low = rows["population_delta_percent_ci95_low_image_boot"].to_numpy(dtype=float)
                    ci_high = rows["population_delta_percent_ci95_high_image_boot"].to_numpy(dtype=float)
                    yerr = np.vstack([np.maximum(y - ci_low, 0.0), np.maximum(ci_high - y, 0.0)])
                    ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        color=color,
                        linestyle="none",
                        linewidth=1.0,
                        elinewidth=0.95,
                        capsize=2.0,
                        zorder=2,
                    )
            ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.85)
            ax.grid(True, color="0.90", linewidth=0.75)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8.0)
            ax.set_ylim(*ylim)
            ax.set_xlim(*xlim)
            if row_idx == 0:
                transform = ax.get_xaxis_transform()
                for band_idx, (metric_col, _metric_label, color, _marker) in enumerate(RMS_SPECS):
                    normal = normal_quantiles.get(metric_col)
                    if not normal:
                        continue
                    y = 0.985 - 0.035 * band_idx
                    x0 = normal["q25_arcmin"]
                    x1 = normal["q75_arcmin"]
                    ax.plot(
                        [x0, x1],
                        [y, y],
                        color=color,
                        linewidth=4.6,
                        solid_capstyle="butt",
                        alpha=0.58,
                        transform=transform,
                        zorder=1,
                    )
            if row_idx == 0:
                ax.set_title(relation_label, fontsize=10.5)
                first = cell.iloc[0]
                ax.text(
                    0.03,
                    0.90,
                    f"u={int(first['n_units'])}, img={int(first['n_selected_unique_images'])}, pairs={int(first['n_selected_unit_image_pairs'])}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7.1,
                    color="0.28",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 0.8},
                )
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=9.0)
            if row_idx == len(OUTCOME_SPECS) - 1:
                ax.set_xlabel("component RMS excursion (arcmin)", fontsize=8.8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9.0, bbox_to_anchor=(0.53, 0.082))
    fig.suptitle(
        "Marginal real-drift component RMS curves: high-SF contour geometry diagnostic",
        fontsize=13.2,
        y=0.982,
    )
    fig.text(
        0.5,
        0.015,
        "Each curve bins one RMS component while averaging over the other, so these are marginal dose curves, not controlled across-vs-along contrasts. "
        "SSI row includes 95% image-bootstrap CIs; lower rows are point estimates. Top bars mark drift-only q25-q75 RMS.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="0.28",
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.16, wspace=0.20, hspace=0.32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{OUT_STEM}.png"
    pdf = OUT_DIR / f"{OUT_STEM}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    data = load_dataset(MATRIX_DIR)
    component_metrics, bins_by_metric = _compute_rms_metrics(data)
    normal_quantiles = _normal_rms_quantiles(component_metrics)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = pd.concat(
        [
            _population_summary_for_relation(
                relation=relation,
                relation_label=relation_label,
                data=data,
                component_metrics=component_metrics,
                bins_by_metric=bins_by_metric,
                rng=rng,
            )
            for relation, relation_label in RELATION_SPECS
        ],
        ignore_index=True,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / f"{OUT_STEM}_values.csv"
    json_path = OUT_DIR / f"{OUT_STEM}_summary.json"
    summary.to_csv(csv, index=False)
    png, pdf = _plot(summary, normal_quantiles)
    _write_json(
        json_path,
        {
            "analysis": "backimage_real_trace_component_rms_excursion_diagnostic",
            "matrix_dir": MATRIX_DIR,
            "out_dir": OUT_DIR,
            "outputs": {"csv": csv, "png": png, "pdf": pdf, "summary_json": json_path},
            "selection": {
                "sf_group": SF_GROUP,
                "min_osi": MIN_OSI,
                "match_max_deg": MATCH_MAX_DEG,
                "orthogonal_min_deg": ORTHOGONAL_MIN_DEG,
                "relations": [relation for relation, _label in RELATION_SPECS],
            },
            "metrics": {
                "component_rms": "RMS of centered trace positions projected along/across image_edge_axis_deg, in arcmin.",
                "bits_per_window": "information_numerator_bits / n_movie_samples for selected unit-image movie samples.",
                "expected_spikes_per_window": "expected_spikes / n_movie_samples for selected unit-image movie samples.",
                "baseline": "Each relation uses its own non-directional stabilized zero-motion baseline for the selected unit-image set.",
            },
            "normal_drift_rms_quantiles": normal_quantiles,
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    )
    print(png)
    print(pdf)
    print(csv)
    print(json_path)


if __name__ == "__main__":
    main()
