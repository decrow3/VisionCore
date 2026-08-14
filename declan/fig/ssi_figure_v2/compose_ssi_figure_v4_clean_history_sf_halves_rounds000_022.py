#!/usr/bin/env python3
"""Compose a non-overwriting corrected-cache Figure 4 using SF halves."""
from pathlib import Path

import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2 import compose_ssi_figure_v4_corrected_sf_quartiles as figure


ROOT = Path(__file__).resolve().parents[3]
GROUPS = ("sf_low_half", "sf_high_half")
LABELS = {"sf_low_half": "low-SF half", "sf_high_half": "high-SF half"}
COLORS = {"sf_low_half": "#0072B2", "sf_high_half": "#D55E00"}


def assign_validated_halves(assignments: pd.DataFrame) -> pd.DataFrame:
    audit = assignments.copy()
    required = {"rr100_index", "preferred_sf_cpd", "recorded_validation_pass"}
    missing = required.difference(audit.columns)
    if missing:
        raise ValueError(f"Assignment table lacks {sorted(missing)}")
    valid = audit[audit.recorded_validation_pass.astype(bool)].sort_values(
        ["preferred_sf_cpd", "rr100_index"]
    )
    if len(valid) != 61:
        raise ValueError(f"Expected 61 recorded-validated units, found {len(valid)}")
    low_index, high_index = np.array_split(valid.index.to_numpy(), 2)
    if (len(low_index), len(high_index)) != (31, 30):
        raise AssertionError("Unexpected half sizes")
    low_max = float(audit.loc[low_index, "preferred_sf_cpd"].max())
    high_min = float(audit.loc[high_index, "preferred_sf_cpd"].min())
    if not low_max < high_min:
        raise ValueError("Median split would divide an exact preferred-SF tie")
    audit["sf_quartile"] = "excluded"
    audit.loc[low_index, "sf_quartile"] = GROUPS[0]
    audit.loc[high_index, "sf_quartile"] = GROUPS[1]
    audit["sf_quartile_label"] = audit.sf_quartile.map(LABELS).fillna("excluded by fit gate")
    audit["quartile_definition"] = (
        "median halves after model-valid and recorded SF-curve r >= 0.5 gate; "
        f"low <= {low_max:.6g} cpd, high >= {high_min:.6g} cpd"
    )
    return audit


figure.ASSEMBLED = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "assembled/rounds_000_022_n023_clean_history_snapshot_v1"
)
figure.OUT = ROOT / "outputs/fig/ssi_figure_v2/corrected_sf_halves_clean_history_rounds000_022_v5"
figure.PANELS = figure.OUT / "panels"
figure.STEM = "ssi_figure_v4_corrected_cache_sf_halves_clean_history_no_bottom_row_rounds000_022_v5"
figure.FIGURE_SCOPE_LABEL = "clean histories · 23 balanced rounds · interim"
figure.GROUPS = GROUPS
figure.LABELS = LABELS
figure.COLORS = COLORS
figure.COMPONENT_GROUP = GROUPS[1]
figure.GROUPING_DESCRIPTION = "61 recorded-validated units split into 31 low-SF and 30 high-SF units"
figure.PATH_PANEL_TITLE = "Path-length effects in\npreferred-SF halves"
figure.MATCH_PANEL_TITLE = "Contour-matched effects in\npreferred-SF halves"
figure.COMPONENT_PANEL_TITLE = "Across/along spread in the\nhigh-SF half"
figure.assign_validated_quartiles = assign_validated_halves


if __name__ == "__main__":
    figure.main()
