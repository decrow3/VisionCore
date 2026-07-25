#!/usr/bin/env python3
"""Panel B with SF>=0.5 and a 15 degree unit-contour match threshold."""

from __future__ import annotations

from declan.active_sensing_movie_information import make_backimage_panel_b_orientation_match_15deg as panel_b


panel_b.HIGH_SF_MIN_CPD = 0.50
panel_b.OUT_STEM = "backimage_real_trace_panel_b_cell_baseline_sf05_coh020_match15"


if __name__ == "__main__":
    panel_b.main()
