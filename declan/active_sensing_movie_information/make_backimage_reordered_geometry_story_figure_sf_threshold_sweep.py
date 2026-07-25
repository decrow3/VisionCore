#!/usr/bin/env python3
"""B-E story-figure sweep across high-SF thresholds."""

from __future__ import annotations

import json
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
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from declan.active_sensing_movie_information import (
    make_backimage_reordered_geometry_story_figure_cell_baseline_sf075_coh020_cde8bins as story,
)


OUT_STEM = "backimage_real_trace_geometry_reordered_story_figure_sf_threshold_sweep_coh020_cde8bins"
SF_MIN_CPDS = (0.50, 0.625, 0.75, 0.875, 1.00)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(story._json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sf_label(threshold: float) -> str:
    return f"{float(threshold):.3g}"


def _sf_slug(threshold: float) -> str:
    return _sf_label(threshold).replace(".", "p").replace("-", "m")


def _configure_story_globals(threshold: float) -> None:
    label = _sf_label(threshold)
    story.HIGH_SF_MIN_CPD = float(threshold)
    story.SF_GROUPS["high_ge0p75"]["label"] = f"SF >= {label}"
    story.SF_GROUPS["high_ge0p75"]["title"] = f"SF >= {label}"
    story.B_PANEL_SPECS = [
        ("low_lt0p5", "strong_contours_no_osi", "SF < 0.50\nall contour units"),
        ("low_lt0p5", "contour_matched", "SF < 0.50\norientation-aligned"),
        ("high_ge0p75", "strong_contours_no_osi", f"SF >= {label}\nall contour units"),
        ("high_ge0p75", "contour_matched", f"SF >= {label}\norientation-aligned"),
    ]
    story.LOWER_PANEL_SPECS = [
        ("contour_matched", f"Aligned SF >= {label} units"),
        ("contour_intermediate", f"Oblique SF >= {label} units"),
        ("contour_orthogonal", f"Orthogonal SF >= {label} units"),
    ]


def _retitle_figure(fig: plt.Figure, threshold: float) -> None:
    label = _sf_label(threshold)
    if fig._suptitle is not None:
        fig._suptitle.set_text(f"Cell-matched real fixational motion effects by contour geometry: SF >= {label}")
    footer = (
        f"B uses contour image windows (coherence >= {story.CONTOUR_COHERENCE_MIN:.2f}), "
        f"SF < {story.LOW_SF_MAX_CPD:.2f} as low, and SF >= {label} as high; "
        f"units with {story.LOW_SF_MAX_CPD:.2f} <= SF < {label} are excluded. "
        f"C-E use SF >= {label}, {story.N_COMPONENT_BINS} wider component bins, and omit the full-trajectory curve."
    )
    for text in fig.texts:
        if "B uses contour image windows" in text.get_text():
            text.set_text(footer)


def _run_threshold(data: dict[str, Any], threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
    _configure_story_globals(threshold)
    panel_b, b_selection = story._compute_panel_b(data)
    component, cde_selection = story._compute_component_panel(data)
    for frame in (panel_b, component):
        frame.insert(0, "high_sf_min_cpd", float(threshold))
        frame.insert(1, "high_sf_label", f"SF >= {_sf_label(threshold)}")
    selection = pd.concat([b_selection, cde_selection], ignore_index=True, sort=False)
    selection.insert(0, "high_sf_min_cpd", float(threshold))
    selection.insert(1, "high_sf_label", f"SF >= {_sf_label(threshold)}")
    fig = story._plot_figure(panel_b, component)
    _retitle_figure(fig, threshold)
    values = pd.concat([panel_b, component], ignore_index=True, sort=False)
    return values, selection, fig


def main() -> None:
    data = story.load_dataset(story.MATRIX_DIR)
    story.OUT_DIR.mkdir(parents=True, exist_ok=True)

    values_path = story.OUT_DIR / f"{OUT_STEM}_values.csv"
    selection_path = story.OUT_DIR / f"{OUT_STEM}_selection_summary.csv"
    json_path = story.OUT_DIR / f"{OUT_STEM}_summary.json"
    pdf_path = story.OUT_DIR / f"{OUT_STEM}.pdf"

    all_values: list[pd.DataFrame] = []
    all_selection: list[pd.DataFrame] = []
    png_paths: list[Path] = []

    with PdfPages(pdf_path) as pages:
        for threshold in SF_MIN_CPDS:
            values, selection, fig = _run_threshold(data, float(threshold))
            all_values.append(values)
            all_selection.append(selection)
            png_path = story.OUT_DIR / f"{OUT_STEM}_sf_ge_{_sf_slug(float(threshold))}.png"
            fig.savefig(png_path, dpi=230, bbox_inches="tight")
            pages.savefig(fig, bbox_inches="tight")
            png_paths.append(png_path)
            plt.close(fig)

    values = pd.concat(all_values, ignore_index=True, sort=False)
    selection = pd.concat(all_selection, ignore_index=True, sort=False)
    values.to_csv(values_path, index=False)
    selection.to_csv(selection_path, index=False)

    _write_json(
        json_path,
        {
            "analysis": "backimage_real_trace_geometry_reordered_story_figure_sf_threshold_sweep",
            "matrix_dir": story.MATRIX_DIR,
            "out_dir": story.OUT_DIR,
            "outputs": {
                "pdf": pdf_path,
                "pngs": png_paths,
                "values_csv": values_path,
                "selection_summary_csv": selection_path,
                "summary_json": json_path,
            },
            "sweep": {
                "sf_metric_col": story.SF_METRIC_COL,
                "low_sf": f"{story.SF_METRIC_COL} < {story.LOW_SF_MAX_CPD}",
                "high_sf_thresholds": SF_MIN_CPDS,
                "contour_coherence_min": story.CONTOUR_COHERENCE_MIN,
                "min_osi": story.MIN_OSI,
                "match_max_deg": story.MATCH_MAX_DEG,
                "orthogonal_min_deg": story.ORTHOGONAL_MIN_DEG,
                "panel_b_drift_bins": story.N_DRIFT_BINS,
                "panel_b_microsaccade_bins": story.N_MICROSACCADE_BINS,
                "component_drift_bins": story.N_COMPONENT_BINS,
            },
            "baseline": "Each plotted nonzero bin is compared with a cell-matched stabilized baseline weighted by that bin's image composition.",
            "note": "Each page repeats panels B-E while changing only the high-SF cutoff; low SF remains <0.50 cpd.",
        },
    )

    print(pdf_path)
    for path in png_paths:
        print(path)
    print(values_path)
    print(selection_path)
    print(json_path)


if __name__ == "__main__":
    main()
