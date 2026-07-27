#!/usr/bin/env python3
"""Diagnostic: matched-bin component paths for high-SF unit-contour relations."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[4]
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
)
from declan.active_sensing_movie_information.make_backimage_panel_c_sf05_match15_matched_bins_bracket import (
    _component_reference_context,
    _pooled_bin_medians,
)
from declan.active_sensing_movie_information.plot_backimage_real_trace_unit_first_and_population_schematics import (
    accumulate_population_movie_rows,
    axis_delta_deg,
    baseline_rows_by_image,
    finite_ratio,
    ratio_delta_stats,
)


OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels"
OUT_STEM = "panel_g_relation_sweep_matched_bins"
SF_MIN_CPD = 0.50
CONTOUR_COHERENCE_MIN = 0.20
MIN_OSI = 0.05
MATCH_MAX_DEG = 15.0
ORTHOGONAL_MIN_DEG = 67.5
BODY_QUANTILES = (0.0, 0.03125, 0.0625, 0.09375, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
TAIL_QUANTILES = (0.95, 0.975)
PLOT_MIN_POS = 40.0
PLOT_MAX_POS = panel_c.LOWER_MAX_POS
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 47
ORANGE = "#D55E00"
EPS = 1e-12

RELATION_SPECS = [
    ("all", "All high SF", "all contour relations"),
    ("aligned", "Aligned", "unit-contour <=15 deg"),
    ("oblique", "Oblique", "15-67.5 deg"),
    ("orthogonal", "Orthogonal", ">=67.5 deg"),
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(panel_c._json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _shared_component_edges(metrics: pd.DataFrame, drift_mask: np.ndarray) -> np.ndarray:
    across = pd.to_numeric(metrics.loc[drift_mask, "across_path_arcmin"], errors="coerce").to_numpy(dtype=float)
    along = pd.to_numeric(metrics.loc[drift_mask, "along_path_arcmin"], errors="coerce").to_numpy(dtype=float)
    across = across[np.isfinite(across) & (across > 0)]
    along = along[np.isfinite(along) & (along > 0)]
    pooled = np.concatenate([across, along])
    body_edges = np.quantile(pooled, BODY_QUANTILES)
    tail_edges = [
        max(float(np.quantile(across, tail_quantile)), float(np.quantile(along, tail_quantile)))
        for tail_quantile in TAIL_QUANTILES
    ]
    tail_high = min(float(np.max(across)), float(np.max(along)))
    edges = np.asarray([*body_edges, *tail_edges, tail_high], dtype=np.float64)
    edges = np.maximum.accumulate(edges)
    if np.any(np.diff(edges) <= 0):
        raise ValueError(f"Matched component edges are not strictly increasing: {edges}")
    span = max(float(edges[-1] - edges[0]), 1e-6)
    edges[0] -= 1e-6 * span
    edges[-1] += 1e-6 * span
    return edges


def _selected_unit_images(data: dict[str, Any], relation: str) -> dict[int, np.ndarray]:
    unit = data["unit"]
    image = data["image"]
    sf = pd.to_numeric(unit[panel_c.SF_METRIC_COL], errors="coerce").to_numpy(dtype=float)
    pref = pd.to_numeric(unit["prior_preferred_orientation_deg"], errors="coerce").to_numpy(dtype=float)
    osi = pd.to_numeric(unit["prior_orientation_selectivity_index"], errors="coerce").to_numpy(dtype=float)
    unit_index = unit["unit_index"].astype(int).to_numpy()

    image_indices = image["image_index"].astype(int).to_numpy()
    image_axis = pd.to_numeric(image["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=float)
    coherence = pd.to_numeric(image["image_orientation_coherence"], errors="coerce").to_numpy(dtype=float)
    contour_mask = np.isfinite(image_axis) & np.isfinite(coherence) & (coherence >= CONTOUR_COHERENCE_MIN)

    selected: dict[int, np.ndarray] = {}
    for idx, unit_id in enumerate(unit_index):
        if not (
            math.isfinite(float(sf[idx]))
            and float(sf[idx]) >= SF_MIN_CPD
            and math.isfinite(float(pref[idx]))
            and math.isfinite(float(osi[idx]))
            and float(osi[idx]) >= MIN_OSI
        ):
            continue
        delta = axis_delta_deg(image_axis, float(pref[idx]))
        if relation == "all":
            keep = contour_mask & np.isfinite(delta)
        elif relation == "aligned":
            keep = contour_mask & np.isfinite(delta) & (delta <= MATCH_MAX_DEG)
        elif relation == "oblique":
            keep = contour_mask & np.isfinite(delta) & (delta > MATCH_MAX_DEG) & (delta < ORTHOGONAL_MIN_DEG)
        elif relation == "orthogonal":
            keep = contour_mask & np.isfinite(delta) & (delta >= ORTHOGONAL_MIN_DEG)
        else:
            raise ValueError(f"unknown relation {relation!r}")
        images = image_indices[keep]
        if images.size:
            selected[int(unit_id)] = images.astype(int)
    return selected


def _population_for_mask(
    data: dict[str, Any],
    *,
    row_image_index: np.ndarray,
    row_mask: np.ndarray,
    baseline_lookup: dict[int, int],
    unit_to_images: dict[int, np.ndarray],
    n_images: int,
) -> dict[str, Any]:
    moving = accumulate_population_movie_rows(
        ssi=data["ssi"],
        expected=data["expected"],
        row_image_index=row_image_index,
        row_mask=row_mask,
        unit_to_images=unit_to_images,
        n_images=n_images,
    )
    cell = _cell_matched_baseline(
        stabilized_ssi=data["stabilized_ssi"],
        stabilized_expected=data["stabilized_expected"],
        row_image_index=row_image_index,
        row_mask=row_mask,
        baseline_lookup=baseline_lookup,
        unit_to_images=unit_to_images,
        n_images=n_images,
    )
    moving_ssi = finite_ratio(float(moving["information_numerator_bits"]), float(moving["expected_spikes"]))
    cell_ssi = finite_ratio(float(cell["information_numerator_bits"]), float(cell["expected_spikes"]))
    return {
        "moving": moving,
        "cell": cell,
        "moving_ssi": moving_ssi,
        "cell_ssi": cell_ssi,
        "ssi_percent_vs_cell_baseline": panel_c._pct_delta(moving_ssi, cell_ssi),
    }


def _compute_relation(
    data: dict[str, Any],
    *,
    metrics: pd.DataFrame,
    drift_mask: np.ndarray,
    edges: np.ndarray,
    pooled_medians: np.ndarray,
    relation: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
    row_image_index = metrics["image_index"].astype(int).to_numpy()
    baseline_lookup = baseline_rows_by_image(data["image"], data["baseline_table"])
    n_images = int(data["stabilized_ssi"].shape[0])
    unit_to_images = _selected_unit_images(data, relation)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    last_populations: dict[str, dict[str, Any]] = {}

    for metric_col, (metric_label, _linestyle, _marker) in panel_c.COMPONENT_STYLES.items():
        bins = _assign_bins(metrics[metric_col].to_numpy(dtype=float), edges)
        for bin_index in range(len(edges) - 1):
            row_mask = drift_mask & (bins == bin_index)
            pop = _population_for_mask(
                data,
                row_image_index=row_image_index,
                row_mask=row_mask,
                baseline_lookup=baseline_lookup,
                unit_to_images=unit_to_images,
                n_images=n_images,
            )
            delta_stats = ratio_delta_stats(
                pop["moving"]["per_image_num"],
                pop["moving"]["per_image_den"],
                pop["cell"]["per_image_num"],
                pop["cell"]["per_image_den"],
                n_resamples=N_BOOTSTRAP,
                rng=rng,
            )
            delta_low = float(delta_stats["population_delta_ci95_low_image_boot"])
            delta_high = float(delta_stats["population_delta_ci95_high_image_boot"])
            values = pd.to_numeric(metrics.loc[row_mask, metric_col], errors="coerce").to_numpy(dtype=float)
            row = {
                "relation": relation,
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
                "n_movie_samples": int(pop["moving"]["n_movie_samples"]),
                "n_images_contributing": int(pop["moving"]["n_images_contributing"]),
                "moving_population_ssi_bits_per_spike": float(pop["moving_ssi"]),
                "cell_baseline_population_ssi_bits_per_spike": float(pop["cell_ssi"]),
                "ssi_percent_vs_cell_baseline": float(pop["ssi_percent_vs_cell_baseline"]),
                "population_delta_percent_ci95_low_image_boot": 100.0 * delta_low / float(pop["cell_ssi"])
                if math.isfinite(delta_low) and math.isfinite(float(pop["cell_ssi"])) and abs(float(pop["cell_ssi"])) > EPS
                else float("nan"),
                "population_delta_percent_ci95_high_image_boot": 100.0 * delta_high / float(pop["cell_ssi"])
                if math.isfinite(delta_high) and math.isfinite(float(pop["cell_ssi"])) and abs(float(pop["cell_ssi"])) > EPS
                else float("nan"),
                "population_delta_p_image_bootstrap_sign": float(delta_stats["population_delta_p_image_bootstrap_sign"]),
            }
            rows.append(row)
            if bin_index == len(edges) - 2:
                key = "across" if metric_col == "across_path_arcmin" else "along"
                last_populations[key] = pop

    contrast = _bootstrap_residual_difference(
        last_populations["across"],
        last_populations["along"],
        rng=np.random.default_rng(BOOTSTRAP_SEED),
    )
    metadata = {
        "relation": relation,
        "n_selected_units": int(len(unit_to_images)),
        "n_selected_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
    }
    return pd.DataFrame(rows), metadata, contrast


def _set_shared_ylim(axes: np.ndarray, values: pd.DataFrame) -> None:
    vals = [0.0]
    for col in [
        "ssi_percent_vs_cell_baseline",
        "population_delta_percent_ci95_low_image_boot",
        "population_delta_percent_ci95_high_image_boot",
    ]:
        arr = pd.to_numeric(values[col], errors="coerce").to_numpy(dtype=float)
        vals.extend(arr[np.isfinite(arr)].tolist())
    lo = min(vals)
    hi = max(vals)
    span = max(hi - lo, 1.0)
    for ax in axes:
        ax.set_ylim(lo - 0.13 * span, hi + 0.28 * span)


def _add_vertical_bracket(
    ax: plt.Axes,
    *,
    x: float,
    y0: float,
    y1: float,
    label: str,
) -> None:
    low, high = sorted([float(y0), float(y1)])
    tick = 0.10
    ax.plot([x, x], [low, high], color=ORANGE, lw=1.15, clip_on=False, zorder=7)
    ax.plot([x - tick, x], [low, low], color=ORANGE, lw=1.15, clip_on=False, zorder=7)
    ax.plot([x - tick, x], [high, high], color=ORANGE, lw=1.15, clip_on=False, zorder=7)
    ax.text(x + 0.045, 0.5 * (low + high), label, ha="left", va="center", fontsize=7.0, color=ORANGE, zorder=8)


def _add_component_reference_bar(ax: plt.Axes, context: dict[str, Any] | None) -> None:
    if not context:
        return
    low = float(context["q25_arcmin"])
    high = float(context["q75_arcmin"])
    if not (math.isfinite(low) and math.isfinite(high) and high > low):
        return
    x_low, x_high = panel_c._x_broken_log([low, high], min_pos=PLOT_MIN_POS, max_pos=PLOT_MAX_POS)
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


def _plot_sheet(values: pd.DataFrame, metadata: pd.DataFrame, contrasts: pd.DataFrame, reference: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(14.6, 3.9), sharey=True)
    _set_shared_ylim(axes, values)
    for ax, (relation, title, subtitle) in zip(axes, RELATION_SPECS, strict=True):
        frame = values[values["relation"].eq(relation)].copy()
        first_rows: dict[str, pd.Series] = {}
        last_rows: dict[str, pd.Series] = {}
        for metric_col, (series_label, linestyle, marker) in panel_c.COMPONENT_STYLES.items():
            drift = frame[frame["component_metric"].eq(metric_col)].sort_values("component_bin_order")
            x = panel_c._x_broken_log(drift["plot_median_arcmin"], min_pos=PLOT_MIN_POS, max_pos=PLOT_MAX_POS)
            y = drift["ssi_percent_vs_cell_baseline"].to_numpy(dtype=float)
            ci_low = drift["population_delta_percent_ci95_low_image_boot"].to_numpy(dtype=float)
            ci_high = drift["population_delta_percent_ci95_high_image_boot"].to_numpy(dtype=float)
            yerr = np.vstack([y - ci_low, ci_high - y])
            ax.plot(x, y, color=ORANGE, linestyle=linestyle, linewidth=1.9, label=series_label, zorder=3)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=ORANGE,
                linestyle="none",
                marker=marker,
                markersize=4.4,
                markerfacecolor="white",
                markeredgewidth=1.1,
                elinewidth=1.0,
                capsize=2.0,
                zorder=4,
            )
            ax.scatter([0.0], [0.0], marker=marker, s=28, facecolors="white", edgecolors=ORANGE, linewidths=1.2, zorder=5)
            first_rows[metric_col] = drift.iloc[0]
            last_rows[metric_col] = drift.iloc[-1]
        panel_c._format_axis(ax, ticks=panel_c.LOWER_TICKS, min_pos=PLOT_MIN_POS, max_pos=PLOT_MAX_POS)
        _add_component_reference_bar(ax, reference)
        ax.axhline(0.0, color="0.35", lw=0.85, ls=":")
        meta = metadata[metadata["relation"].eq(relation)].iloc[0]
        ax.set_title(
            f"{title}\n{subtitle}; {int(meta.n_selected_units)} units, {int(meta.n_selected_unit_image_pairs)} pairs",
            fontsize=10.0,
            pad=6,
        )
        ax.set_xlabel("component path (arcmin)")
        if ax is axes[0]:
            ax.set_ylabel("SSI residual (% vs cell baseline)")
            ax.legend(frameon=False, fontsize=7.4, loc="lower left")
        else:
            ax.spines["left"].set_visible(False)
        contrast = contrasts[contrasts["relation"].eq(relation)].iloc[0]
        x_last = float(
            panel_c._x_broken_log(
                [float(last_rows["across_path_arcmin"]["plot_median_arcmin"])],
                min_pos=PLOT_MIN_POS,
                max_pos=PLOT_MAX_POS,
            )[0]
        )
        _add_vertical_bracket(
            ax,
            x=x_last + 0.22,
            y0=float(last_rows["across_path_arcmin"]["ssi_percent_vs_cell_baseline"]),
            y1=float(last_rows["along_path_arcmin"]["ssi_percent_vs_cell_baseline"]),
            label=(
                f"{float(contrast.across_minus_along_percent_point):+.1f} pp\n"
                f"{panel_c._format_p_label(float(contrast.contrast_p_image_bootstrap_sign))}"
            ),
        )
    fig.suptitle("High-SF component-path effects across unit-contour tuning relations", fontsize=13.0, y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return fig


def build(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    data = panel_c.load_dataset(panel_c.MATRIX_DIR)
    metrics = _compute_component_metrics(data)
    drift_mask = metrics["context"].astype(str).eq("drift_only").to_numpy(dtype=bool)
    edges = _shared_component_edges(metrics, drift_mask)
    pooled_medians = _pooled_bin_medians(metrics, drift_mask, edges)
    reference = _component_reference_context(metrics, drift_mask)
    values_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for relation, _title, _subtitle in RELATION_SPECS:
        frame, meta, contrast = _compute_relation(
            data,
            metrics=metrics,
            drift_mask=drift_mask,
            edges=edges,
            pooled_medians=pooled_medians,
            relation=relation,
        )
        values_frames.append(frame)
        metadata_rows.append(meta)
        contrast_rows.append({"relation": relation, **contrast})

    values = pd.concat(values_frames, ignore_index=True, sort=False)
    metadata = pd.DataFrame(metadata_rows)
    contrasts = pd.DataFrame(contrast_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    values_csv = out_dir / f"{OUT_STEM}_values.csv"
    metadata_csv = out_dir / f"{OUT_STEM}_selection.csv"
    contrast_csv = out_dir / f"{OUT_STEM}_last_bin_contrasts.csv"
    json_path = out_dir / f"{OUT_STEM}_provenance.json"
    png = out_dir / f"{OUT_STEM}.png"
    pdf = out_dir / f"{OUT_STEM}.pdf"
    svg = out_dir / f"{OUT_STEM}.svg"
    values.to_csv(values_csv, index=False)
    metadata.to_csv(metadata_csv, index=False)
    contrasts.to_csv(contrast_csv, index=False)
    _write_json(
        json_path,
        {
            "analysis": OUT_STEM,
            "selection": {
                "sf_min_cpd": SF_MIN_CPD,
                "contour_coherence_min": CONTOUR_COHERENCE_MIN,
                "min_osi": MIN_OSI,
                "match_max_deg": MATCH_MAX_DEG,
                "orthogonal_min_deg": ORTHOGONAL_MIN_DEG,
            },
            "binning": {
                "body_quantiles": [float(v) for v in BODY_QUANTILES],
                "tail_quantiles": [float(v) for v in TAIL_QUANTILES],
                "matched_component_edges_arcmin": [float(v) for v in edges],
                "standard_drift_component_path_context": reference,
                "plot_min_pos_arcmin": float(PLOT_MIN_POS),
                "plot_max_pos_arcmin": float(PLOT_MAX_POS),
            },
            "outputs": {
                "png": png,
                "pdf": pdf,
                "svg": svg,
                "values_csv": values_csv,
                "selection_csv": metadata_csv,
                "last_bin_contrasts_csv": contrast_csv,
                "provenance_json": json_path,
            },
            "bootstrap": {
                "n_bootstrap": N_BOOTSTRAP,
                "seed": BOOTSTRAP_SEED,
                "unit": "paired image bootstrap; error bars are moving-vs-cell baseline; brackets are across-minus-along",
            },
        },
    )
    fig = _plot_sheet(values, metadata, contrasts, reference)
    fig.savefig(png, dpi=230, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {
        "png": png,
        "pdf": pdf,
        "svg": svg,
        "values_csv": values_csv,
        "selection_csv": metadata_csv,
        "last_bin_contrasts_csv": contrast_csv,
        "provenance_json": json_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    paths = build(args.out_dir)
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
