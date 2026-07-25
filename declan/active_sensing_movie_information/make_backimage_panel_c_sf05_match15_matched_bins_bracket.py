#!/usr/bin/env python3
"""Panel C with shared component-path bins and a last-bin across/along bracket."""

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

from declan.active_sensing_movie_information import make_backimage_panel_c_sf05_cell_baseline_errorbars as panel_c
from declan.active_sensing_movie_information.make_backimage_component_2d_surface_diagnostic import (
    _assign_bins,
    _compute_component_metrics,
)
from declan.active_sensing_movie_information.make_backimage_component_path_baseline_decomposition_surface import (
    _cell_matched_baseline,
)
from declan.active_sensing_movie_information.make_backimage_panel_c_across_along_tail_contrast import (
    _bootstrap_residual_difference,
    _population_for_mask,
)
from declan.active_sensing_movie_information.plot_backimage_real_trace_unit_first_and_population_schematics import (
    accumulate_population_movie_rows,
    baseline_rows_by_image,
    finite_ratio,
    ratio_delta_stats,
)


MATCH_MAX_DEG = 15.0
BODY_QUANTILES = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
TAIL_QUANTILE = 0.95
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 47
OUT_STEM = "backimage_real_trace_panel_c_aligned_sf_ge_0p5_match15_matched_bins_bracket"
ORANGE = "#D55E00"
EPS = 1e-12


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(panel_c._json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _shared_component_edges(metrics: pd.DataFrame, drift_mask: np.ndarray) -> np.ndarray:
    across = pd.to_numeric(metrics.loc[drift_mask, "across_path_arcmin"], errors="coerce").to_numpy(dtype=float)
    along = pd.to_numeric(metrics.loc[drift_mask, "along_path_arcmin"], errors="coerce").to_numpy(dtype=float)
    across = across[np.isfinite(across) & (across > 0)]
    along = along[np.isfinite(along) & (along > 0)]
    pooled = np.concatenate([across, along])
    body_edges = np.quantile(pooled, BODY_QUANTILES)
    tail_low = max(float(np.quantile(across, TAIL_QUANTILE)), float(np.quantile(along, TAIL_QUANTILE)))
    tail_high = min(float(np.max(across)), float(np.max(along)))
    edges = np.asarray([*body_edges, tail_low, tail_high], dtype=np.float64)
    edges = np.maximum.accumulate(edges)
    if np.any(np.diff(edges) <= 0):
        raise ValueError(f"Matched component edges are not strictly increasing: {edges}")
    span = max(float(edges[-1] - edges[0]), 1e-6)
    edges[0] -= 1e-6 * span
    edges[-1] += 1e-6 * span
    return edges


def _pooled_bin_medians(metrics: pd.DataFrame, drift_mask: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.concatenate(
        [
            pd.to_numeric(metrics.loc[drift_mask, "across_path_arcmin"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(metrics.loc[drift_mask, "along_path_arcmin"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    bins = _assign_bins(values, edges)
    medians = np.full(len(edges) - 1, np.nan, dtype=np.float64)
    for bin_index in range(len(edges) - 1):
        vals = values[bins == bin_index]
        medians[bin_index] = float(np.nanmedian(vals)) if vals.size else float("nan")
    return medians


def _component_reference_context(metrics: pd.DataFrame, drift_mask: np.ndarray) -> dict[str, Any]:
    values = np.concatenate(
        [
            pd.to_numeric(metrics.loc[drift_mask, "across_path_arcmin"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(metrics.loc[drift_mask, "along_path_arcmin"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return {}
    return {
        "definition": "Drift-only BackImage movie-row component paths, pooled across/along and ignoring trace-contour alignment classes.",
        "n_component_values": int(values.size),
        "q25_arcmin": float(np.nanpercentile(values, 25.0)),
        "median_arcmin": float(np.nanmedian(values)),
        "q75_arcmin": float(np.nanpercentile(values, 75.0)),
    }


def _compute_panel(data: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
    metrics = _compute_component_metrics(data)
    drift_mask = metrics["context"].astype(str).eq("drift_only").to_numpy(dtype=bool)
    edges = _shared_component_edges(metrics, drift_mask)
    pooled_medians = _pooled_bin_medians(metrics, drift_mask, edges)
    reference_context = _component_reference_context(metrics, drift_mask)
    row_image_index = metrics["image_index"].astype(int).to_numpy()
    baseline_lookup = baseline_rows_by_image(data["image"], data["baseline_table"])
    n_images = int(data["stabilized_ssi"].shape[0])
    unit_to_images = panel_c._selected_unit_images(data["unit"], data["image"], match_max_deg=MATCH_MAX_DEG)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "component_metric": "matched_shared_bins",
            "component_metric_label": "matched stabilized",
            "context": "stabilized",
            "component_bin": "stabilized_zero_motion",
            "component_bin_order": 0,
            "plot_median_arcmin": 0.0,
            "component_median_arcmin": 0.0,
            "ssi_percent_vs_cell_baseline": 0.0,
        }
    )

    last_populations: dict[str, dict[str, Any]] = {}
    for metric_col, (metric_label, _linestyle, _marker) in panel_c.COMPONENT_STYLES.items():
        bins = _assign_bins(metrics[metric_col].to_numpy(dtype=float), edges)
        for bin_index in range(len(edges) - 1):
            row_mask = drift_mask & (bins == bin_index)
            moving_pop = accumulate_population_movie_rows(
                ssi=data["ssi"],
                expected=data["expected"],
                row_image_index=row_image_index,
                row_mask=row_mask,
                unit_to_images=unit_to_images,
                n_images=n_images,
            )
            cell_pop = _cell_matched_baseline(
                stabilized_ssi=data["stabilized_ssi"],
                stabilized_expected=data["stabilized_expected"],
                row_image_index=row_image_index,
                row_mask=row_mask,
                baseline_lookup=baseline_lookup,
                unit_to_images=unit_to_images,
                n_images=n_images,
            )
            moving_ssi = finite_ratio(float(moving_pop["information_numerator_bits"]), float(moving_pop["expected_spikes"]))
            cell_ssi = finite_ratio(float(cell_pop["information_numerator_bits"]), float(cell_pop["expected_spikes"]))
            delta_stats = ratio_delta_stats(
                moving_pop["per_image_num"],
                moving_pop["per_image_den"],
                cell_pop["per_image_num"],
                cell_pop["per_image_den"],
                n_resamples=N_BOOTSTRAP,
                rng=rng,
            )
            delta_low = float(delta_stats["population_delta_ci95_low_image_boot"])
            delta_high = float(delta_stats["population_delta_ci95_high_image_boot"])
            values = pd.to_numeric(metrics.loc[row_mask, metric_col], errors="coerce").to_numpy(dtype=float)
            row = {
                "component_metric": metric_col,
                "component_metric_label": metric_label,
                "context": "drift_only",
                "component_bin": f"matched_path_q{bin_index + 1:02d}",
                "component_bin_order": int(bin_index + 1),
                "component_min_arcmin": float(edges[bin_index]),
                "component_max_arcmin": float(edges[bin_index + 1]),
                "plot_median_arcmin": float(pooled_medians[bin_index]),
                "component_median_arcmin": float(np.nanmedian(values)) if values.size else float("nan"),
                "n_movie_rows_global": int(np.count_nonzero(row_mask)),
                "n_movie_samples": int(moving_pop["n_movie_samples"]),
                "n_images_contributing": int(moving_pop["n_images_contributing"]),
                "moving_population_ssi_bits_per_spike": moving_ssi,
                "cell_baseline_population_ssi_bits_per_spike": cell_ssi,
                "ssi_percent_vs_cell_baseline": panel_c._pct_delta(moving_ssi, cell_ssi),
                "population_delta_ci95_low_image_boot": delta_low,
                "population_delta_ci95_high_image_boot": delta_high,
                "population_delta_percent_ci95_low_image_boot": 100.0 * delta_low / cell_ssi
                if math.isfinite(delta_low) and math.isfinite(cell_ssi) and abs(cell_ssi) > EPS
                else float("nan"),
                "population_delta_percent_ci95_high_image_boot": 100.0 * delta_high / cell_ssi
                if math.isfinite(delta_high) and math.isfinite(cell_ssi) and abs(cell_ssi) > EPS
                else float("nan"),
                "population_delta_p_image_bootstrap_sign": float(delta_stats["population_delta_p_image_bootstrap_sign"]),
            }
            rows.append(row)
            if bin_index == len(edges) - 2:
                key = "across" if metric_col == "across_path_arcmin" else "along"
                last_populations[key] = {
                    "moving": moving_pop,
                    "cell": cell_pop,
                    "moving_ssi": moving_ssi,
                    "cell_ssi": cell_ssi,
                    "ssi_percent_vs_cell_baseline": row["ssi_percent_vs_cell_baseline"],
                }

    # Recompute the contrast with a fresh RNG so it is independent of the one-sample CI draw order.
    contrast = _bootstrap_residual_difference(
        last_populations["across"],
        last_populations["along"],
        rng=np.random.default_rng(BOOTSTRAP_SEED),
    )
    metadata = {
        "n_selected_units": int(len(unit_to_images)),
        "n_selected_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
        "matched_component_edges_arcmin": [float(v) for v in edges],
        "last_bin_range_arcmin": [float(edges[-2]), float(edges[-1])],
        "standard_drift_component_path_context": reference_context,
    }
    return pd.DataFrame(rows), metadata, contrast


def _add_vertical_bracket(
    ax: plt.Axes,
    *,
    x: float,
    y0: float,
    y1: float,
    label: str,
    color: str,
) -> None:
    low, high = sorted([float(y0), float(y1)])
    tick = 0.13
    ax.plot([x, x], [low, high], color=color, lw=1.35, clip_on=False, zorder=6)
    ax.plot([x - tick, x], [low, low], color=color, lw=1.35, clip_on=False, zorder=6)
    ax.plot([x - tick, x], [high, high], color=color, lw=1.35, clip_on=False, zorder=6)
    ax.text(
        x + 0.08,
        0.5 * (low + high),
        label,
        ha="left",
        va="center",
        fontsize=8.6,
        color=color,
        zorder=7,
    )


def _add_component_reference_bar(ax: plt.Axes, context: dict[str, Any] | None) -> None:
    if not context:
        return
    low = float(context["q25_arcmin"])
    high = float(context["q75_arcmin"])
    if not (math.isfinite(low) and math.isfinite(high) and high > low):
        return
    x_low, x_high = panel_c._x_broken_log(
        [low, high],
        min_pos=panel_c.LOWER_MIN_POS,
        max_pos=panel_c.LOWER_MAX_POS,
    )
    ax.axvspan(
        float(x_low),
        float(x_high),
        ymin=0.0,
        ymax=1.0,
        facecolor="#7c7c7c",
        edgecolor="none",
        alpha=0.13,
        zorder=0,
    )


def _plot_panel(summary: pd.DataFrame, metadata: dict[str, Any], contrast: dict[str, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.45, 4.45))
    vals = [0.0]
    for col in [
        "ssi_percent_vs_cell_baseline",
        "population_delta_percent_ci95_low_image_boot",
        "population_delta_percent_ci95_high_image_boot",
    ]:
        if col not in summary:
            continue
        arr = pd.to_numeric(summary[col], errors="coerce").to_numpy(dtype=float)
        vals.extend(arr[np.isfinite(arr)].tolist())
    vals.extend(
        [
            float(contrast["contrast_ci95_low_image_boot"]),
            float(contrast["contrast_ci95_high_image_boot"]),
        ]
    )
    lo = min(vals)
    hi = max(vals)
    span = max(hi - lo, 1.0)
    ax.set_ylim(lo - 0.13 * span, hi + 0.34 * span)

    first_rows: dict[str, pd.Series] = {}
    last_rows: dict[str, pd.Series] = {}
    for metric_col, (label, linestyle, marker) in panel_c.COMPONENT_STYLES.items():
        drift = summary[
            summary["component_metric"].eq(metric_col) & summary["context"].eq("drift_only")
        ].sort_values("component_bin_order")
        x = panel_c._x_broken_log(
            drift["plot_median_arcmin"],
            min_pos=panel_c.LOWER_MIN_POS,
            max_pos=panel_c.LOWER_MAX_POS,
        )
        y = drift["ssi_percent_vs_cell_baseline"].to_numpy(dtype=float)
        ci_low = drift["population_delta_percent_ci95_low_image_boot"].to_numpy(dtype=float)
        ci_high = drift["population_delta_percent_ci95_high_image_boot"].to_numpy(dtype=float)
        yerr = np.vstack([y - ci_low, ci_high - y])
        ax.plot(x, y, color=ORANGE, linestyle=linestyle, linewidth=2.1, label=label, zorder=3)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=ORANGE,
            linestyle="none",
            marker=marker,
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.25,
            linewidth=1.6,
            elinewidth=1.2,
            capsize=2.0,
            zorder=4,
        )
        ax.scatter(
            [0.0],
            [0.0],
            marker=marker,
            s=32,
            facecolors="white",
            edgecolors=ORANGE,
            linewidths=1.35,
            zorder=5,
        )
        first_rows[metric_col] = drift.iloc[0]
        last_rows[metric_col] = drift.iloc[-1]

    panel_c._format_axis(
        ax,
        ticks=panel_c.LOWER_TICKS,
        min_pos=panel_c.LOWER_MIN_POS,
        max_pos=panel_c.LOWER_MAX_POS,
    )
    _add_component_reference_bar(ax, metadata.get("standard_drift_component_path_context"))
    ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
    ax.set_title(
        f"SF >= {panel_c.SF_MIN_CPD:.2f}; unit-contour match <= {MATCH_MAX_DEG:g} deg\n"
        f"coh >= {panel_c.CONTOUR_COHERENCE_MIN:.2f}; shared component-path bins; "
        f"{metadata['n_selected_units']} units, {metadata['n_selected_unit_image_pairs']} pairs",
        fontsize=11.5,
        pad=8,
    )
    ax.set_ylabel("SSI residual\n(% vs cell-matched stabilized)")
    ax.set_xlabel("component path length (arcmin; shared bins, log scale after break)")
    ax.legend(frameon=False, fontsize=8.3, loc="lower left")

    across_first = first_rows["across_path_arcmin"]
    along_first = first_rows["along_path_arcmin"]
    x1_raw = max(float(across_first["plot_median_arcmin"]), float(along_first["plot_median_arcmin"]))
    x1 = float(panel_c._x_broken_log([x1_raw], min_pos=panel_c.LOWER_MIN_POS, max_pos=panel_c.LOWER_MAX_POS)[0])
    y_lo, y_top = ax.get_ylim()
    y_span = max(y_top - y_lo, 1.0)
    label = (
        "across "
        + panel_c._format_p_label(float(across_first["population_delta_p_image_bootstrap_sign"]))
        + "\nalong "
        + panel_c._format_p_label(float(along_first["population_delta_p_image_bootstrap_sign"]))
    )
    panel_c._add_bracket(
        ax,
        x0=0.0,
        x1=x1,
        y=y_top - 0.105 * y_span,
        text=label,
        color=ORANGE,
        linestyle="-",
        text_x=x1 + 0.10,
        text_ha="left",
    )

    across_last = last_rows["across_path_arcmin"]
    along_last = last_rows["along_path_arcmin"]
    x_last = float(
        panel_c._x_broken_log(
            [float(across_last["plot_median_arcmin"])],
            min_pos=panel_c.LOWER_MIN_POS,
            max_pos=panel_c.LOWER_MAX_POS,
        )[0]
    )
    diff = float(contrast["across_minus_along_percent_point"])
    p_label = panel_c._format_p_label(float(contrast["contrast_p_image_bootstrap_sign"]))
    _add_vertical_bracket(
        ax,
        x=x_last + 0.30,
        y0=float(across_last["ssi_percent_vs_cell_baseline"]),
        y1=float(along_last["ssi_percent_vs_cell_baseline"]),
        label=f"across - along\n{diff:+.1f} pp, {p_label}",
        color=ORANGE,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    data = panel_c.load_dataset(panel_c.MATRIX_DIR)
    summary, metadata, contrast = _compute_panel(data)
    panel_c.OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = panel_c.OUT_DIR / f"{OUT_STEM}_values.csv"
    contrast_csv = panel_c.OUT_DIR / f"{OUT_STEM}_last_bin_contrast.csv"
    json_path = panel_c.OUT_DIR / f"{OUT_STEM}_summary.json"
    png_path = panel_c.OUT_DIR / f"{OUT_STEM}.png"
    pdf_path = panel_c.OUT_DIR / f"{OUT_STEM}.pdf"
    summary.to_csv(csv_path, index=False)
    pd.DataFrame([{**contrast, **{f"last_bin_{k}": v for k, v in metadata.items()}}]).to_csv(contrast_csv, index=False)

    fig = _plot_panel(summary, metadata, contrast)
    fig.savefig(png_path, dpi=230, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    _write_json(
        json_path,
        {
            "analysis": OUT_STEM,
            "matrix_dir": panel_c.MATRIX_DIR,
            "out_dir": panel_c.OUT_DIR,
            "outputs": {
                "png": png_path,
                "pdf": pdf_path,
                "values_csv": csv_path,
                "last_bin_contrast_csv": contrast_csv,
                "summary_json": json_path,
            },
            "selection": {
                "sf_metric_col": panel_c.SF_METRIC_COL,
                "sf_min_cpd": panel_c.SF_MIN_CPD,
                "contour_coherence_min": panel_c.CONTOUR_COHERENCE_MIN,
                "min_osi": panel_c.MIN_OSI,
                "match_max_deg": MATCH_MAX_DEG,
                "n_selected_units": metadata["n_selected_units"],
                "n_selected_unit_image_pairs": metadata["n_selected_unit_image_pairs"],
            },
            "binning": {
                "body_quantiles_from_pooled_across_and_along": BODY_QUANTILES,
                "tail_quantile": TAIL_QUANTILE,
                "matched_component_edges_arcmin": metadata["matched_component_edges_arcmin"],
                "last_bin_range_arcmin": metadata["last_bin_range_arcmin"],
                "plot_x": "Pooled median component path length within each shared absolute bin.",
                "standard_drift_component_path_context": metadata["standard_drift_component_path_context"],
            },
            "last_bin_across_along_contrast": contrast,
            "bootstrap": {
                "n_bootstrap": N_BOOTSTRAP,
                "seed": BOOTSTRAP_SEED,
                "unit": "paired image bootstrap; one-sample CIs are moving-vs-cell baseline, bracket is across-minus-along residual percent",
            },
        },
    )
    print(png_path)
    print(pdf_path)
    print(csv_path)
    print(contrast_csv)
    print(json_path)


if __name__ == "__main__":
    main()
