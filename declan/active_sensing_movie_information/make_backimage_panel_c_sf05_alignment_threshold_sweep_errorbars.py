#!/usr/bin/env python3
"""Panel C sweep over unit-contour orientation-match thresholds."""

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

from declan.active_sensing_movie_information import make_backimage_panel_c_sf05_cell_baseline_errorbars as panel_c


MATCH_THRESHOLDS_DEG = (22.5, 15.0, 10.0, 5.0)
OUT_STEM = "backimage_real_trace_panel_c_aligned_sf_ge_0p5_orientation_match_sweep_errorbars"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _threshold_slug(threshold_deg: float) -> str:
    return f"{threshold_deg:g}".replace(".", "p")


def _panel_title(metadata: dict[str, Any], threshold_deg: float) -> str:
    return (
        f"SF >= {panel_c.SF_MIN_CPD:.2f}; unit-contour match <= {threshold_deg:g} deg\n"
        f"coh >= {panel_c.CONTOUR_COHERENCE_MIN:.2f}; "
        f"{metadata['n_selected_units']} units, {metadata['n_selected_unit_image_pairs']} pairs"
    )


def _shared_ylim(summaries: list[pd.DataFrame]) -> tuple[float, float]:
    vals: list[float] = [0.0]
    cols = [
        "ssi_percent_vs_cell_baseline",
        "population_delta_percent_ci95_low_image_boot",
        "population_delta_percent_ci95_high_image_boot",
    ]
    for summary in summaries:
        for col in cols:
            arr = pd.to_numeric(summary[col], errors="coerce").to_numpy(dtype=float)
            vals.extend(arr[np.isfinite(arr)].tolist())
    lo = min(vals)
    hi = max(vals)
    span = max(hi - lo, 1.0)
    return (lo - 0.13 * span, hi + 0.32 * span)


def _first_bin_table(summary: pd.DataFrame, threshold_deg: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_col, (metric_label, _linestyle, _marker) in panel_c.COMPONENT_STYLES.items():
        drift = summary[
            summary["component_metric"].eq(metric_col) & summary["context"].eq("drift_only")
        ].sort_values("component_median_arcmin")
        if drift.empty:
            continue
        first = drift.iloc[0]
        rows.append(
            {
                "match_max_deg": float(threshold_deg),
                "component_metric": metric_col,
                "component_metric_label": metric_label,
                "first_bin_median_arcmin": float(first["component_median_arcmin"]),
                "first_bin_ssi_percent_vs_cell_baseline": float(first["ssi_percent_vs_cell_baseline"]),
                "first_bin_ci95_low_percent": float(first["population_delta_percent_ci95_low_image_boot"]),
                "first_bin_ci95_high_percent": float(first["population_delta_percent_ci95_high_image_boot"]),
                "first_bin_p_image_bootstrap_sign": float(first["population_delta_p_image_bootstrap_sign"]),
            }
        )
    return rows


def main() -> None:
    data = panel_c.load_dataset(panel_c.MATRIX_DIR)
    panel_c.OUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[pd.DataFrame] = []
    metadata_by_threshold: dict[str, dict[str, Any]] = {}
    first_bin_rows: list[dict[str, Any]] = []

    for threshold_deg in MATCH_THRESHOLDS_DEG:
        summary, metadata = panel_c._compute_panel(data, match_max_deg=threshold_deg)
        summary = summary.copy()
        summary.insert(0, "match_max_deg", float(threshold_deg))
        summary.insert(1, "match_threshold_label", f"<={threshold_deg:g} deg")
        summaries.append(summary)
        slug = _threshold_slug(threshold_deg)
        metadata_by_threshold[slug] = metadata
        first_bin_rows.extend(_first_bin_table(summary, threshold_deg))

    shared_ylim = _shared_ylim(summaries)
    combined = pd.concat(summaries, ignore_index=True)
    combined_csv = panel_c.OUT_DIR / f"{OUT_STEM}_values.csv"
    first_bin_csv = panel_c.OUT_DIR / f"{OUT_STEM}_first_bin_tests.csv"
    summary_json = panel_c.OUT_DIR / f"{OUT_STEM}_summary.json"
    multipage_pdf = panel_c.OUT_DIR / f"{OUT_STEM}.pdf"
    grid_png = panel_c.OUT_DIR / f"{OUT_STEM}_sheet.png"
    grid_pdf = panel_c.OUT_DIR / f"{OUT_STEM}_sheet.pdf"

    combined.to_csv(combined_csv, index=False)
    pd.DataFrame(first_bin_rows).to_csv(first_bin_csv, index=False)

    individual_outputs: dict[str, dict[str, Path]] = {}
    with PdfPages(multipage_pdf) as pdf:
        for threshold_deg, summary in zip(MATCH_THRESHOLDS_DEG, summaries, strict=True):
            slug = _threshold_slug(threshold_deg)
            metadata = metadata_by_threshold[slug]
            title = _panel_title(metadata, threshold_deg)
            fig = panel_c._plot_panel(summary, metadata, title=title, ylim=shared_ylim)
            png_path = panel_c.OUT_DIR / f"{OUT_STEM}_match_le_{slug}deg.png"
            pdf_path = panel_c.OUT_DIR / f"{OUT_STEM}_match_le_{slug}deg.pdf"
            fig.savefig(png_path, dpi=230, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            individual_outputs[slug] = {"png": png_path, "pdf": pdf_path}

    grid_fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.35), sharex=False, sharey=True)
    for ax, threshold_deg, summary in zip(axes.ravel(), MATCH_THRESHOLDS_DEG, summaries, strict=True):
        slug = _threshold_slug(threshold_deg)
        metadata = metadata_by_threshold[slug]
        panel_c._plot_panel(summary, metadata, ax=ax, title=_panel_title(metadata, threshold_deg), ylim=shared_ylim)
    grid_fig.suptitle(
        "Component path-length Panel C across unit-contour orientation-match thresholds",
        fontsize=13.0,
        y=0.996,
    )
    grid_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    grid_fig.savefig(grid_png, dpi=230, bbox_inches="tight")
    grid_fig.savefig(grid_pdf, bbox_inches="tight")
    plt.close(grid_fig)

    _write_json(
        summary_json,
        {
            "analysis": "backimage_real_trace_panel_c_sf05_alignment_threshold_sweep_errorbars",
            "matrix_dir": panel_c.MATRIX_DIR,
            "out_dir": panel_c.OUT_DIR,
            "outputs": {
                "multipage_pdf": multipage_pdf,
                "sheet_png": grid_png,
                "sheet_pdf": grid_pdf,
                "values_csv": combined_csv,
                "first_bin_tests_csv": first_bin_csv,
                "individual_panels": individual_outputs,
                "summary_json": summary_json,
            },
            "selection": {
                "sf_metric_col": panel_c.SF_METRIC_COL,
                "sf_min_cpd": panel_c.SF_MIN_CPD,
                "contour_coherence_min": panel_c.CONTOUR_COHERENCE_MIN,
                "min_osi": panel_c.MIN_OSI,
                "match_thresholds_deg": MATCH_THRESHOLDS_DEG,
            },
            "binning": {"component_drift_bins": panel_c.N_COMPONENT_BINS},
            "bootstrap": {
                "n_bootstrap": panel_c.N_BOOTSTRAP,
                "seed": panel_c.BOOTSTRAP_SEED,
                "unit": "paired image bootstrap of moving-vs-cell-matched-stabilized ratio delta",
            },
            "metadata_by_threshold": metadata_by_threshold,
            "shared_ylim_percent": shared_ylim,
        },
    )

    print(multipage_pdf)
    print(grid_png)
    print(grid_pdf)
    print(combined_csv)
    print(first_bin_csv)
    print(summary_json)


if __name__ == "__main__":
    main()
