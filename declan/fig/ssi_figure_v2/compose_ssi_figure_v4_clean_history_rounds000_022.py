#!/usr/bin/env python3
"""Compose non-overwriting Figure 4 from the 23-round clean-history snapshot."""
from pathlib import Path

from declan.fig.ssi_figure_v2 import compose_ssi_figure_v4_corrected_sf_quartiles as figure


ROOT = Path(__file__).resolve().parents[3]
figure.ASSEMBLED = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "assembled/rounds_000_022_n023_clean_history_snapshot_v1"
)
figure.OUT = ROOT / "outputs/fig/ssi_figure_v2/corrected_sf_quartiles_clean_history_rounds000_022_v4"
figure.PANELS = figure.OUT / "panels"
figure.STEM = "ssi_figure_v4_corrected_cache_sf_quartiles_clean_history_no_bottom_row_rounds000_022_v4"
figure.FIGURE_SCOPE_LABEL = "clean histories · 23 balanced rounds · interim"


if __name__ == "__main__":
    figure.main()
