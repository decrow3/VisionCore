#!/usr/bin/env python3
"""Panel C with a split high component-path tail bin."""

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

from declan.active_sensing_movie_information import make_backimage_panel_c_sf05_cell_baseline_errorbars as panel_c


MATCH_MAX_DEG = 15.0
BIN_QUANTILES = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.95, 1.0)
OUT_STEM = "backimage_real_trace_panel_c_aligned_sf_ge_0p5_match15_tail_bin_errorbars"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(panel_c._json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    data = panel_c.load_dataset(panel_c.MATRIX_DIR)
    summary, metadata = panel_c._compute_panel(
        data,
        match_max_deg=MATCH_MAX_DEG,
        component_bin_quantiles=BIN_QUANTILES,
    )
    panel_c.OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = panel_c.OUT_DIR / f"{OUT_STEM}_values.csv"
    json_path = panel_c.OUT_DIR / f"{OUT_STEM}_summary.json"
    png_path = panel_c.OUT_DIR / f"{OUT_STEM}.png"
    pdf_path = panel_c.OUT_DIR / f"{OUT_STEM}.pdf"

    summary.to_csv(csv_path, index=False)
    title = (
        f"SF >= {panel_c.SF_MIN_CPD:.2f}; unit-contour match <= {MATCH_MAX_DEG:g} deg\n"
        f"coh >= {panel_c.CONTOUR_COHERENCE_MIN:.2f}; top component-path bin split; "
        f"{metadata['n_selected_units']} units, {metadata['n_selected_unit_image_pairs']} pairs"
    )
    fig = panel_c._plot_panel(summary, metadata, title=title)
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
                "summary_json": json_path,
            },
            "selection": {
                **metadata,
                "sf_metric_col": panel_c.SF_METRIC_COL,
                "sf_min_cpd": panel_c.SF_MIN_CPD,
                "contour_coherence_min": panel_c.CONTOUR_COHERENCE_MIN,
                "min_osi": panel_c.MIN_OSI,
                "match_max_deg": MATCH_MAX_DEG,
            },
            "binning": {
                "component_bin_quantiles": BIN_QUANTILES,
                "note": "The original q87.5-q100 top bin is split into q87.5-q95 and q95-q100 to expose a higher component-path point.",
            },
            "bootstrap": {
                "n_bootstrap": panel_c.N_BOOTSTRAP,
                "seed": panel_c.BOOTSTRAP_SEED,
                "unit": "paired image bootstrap of moving-vs-cell-matched-stabilized ratio delta",
            },
        },
    )

    print(png_path)
    print(pdf_path)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
