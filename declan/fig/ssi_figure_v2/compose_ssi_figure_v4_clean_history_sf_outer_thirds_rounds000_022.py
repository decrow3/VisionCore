#!/usr/bin/env python3
"""Compose corrected-cache Figure 4 using validated SF outer thirds."""
from pathlib import Path

import pandas as pd

from declan.fig.ssi_figure_v2 import compose_ssi_figure_v4_corrected_sf_quartiles as figure


ROOT = Path(__file__).resolve().parents[3]
GROUPS = ("sf_bottom_third", "sf_top_third")
LABELS = {"sf_bottom_third": "bottom-SF third", "sf_top_third": "top-SF third"}
COLORS = {"sf_bottom_third": "#0072B2", "sf_top_third": "#D55E00"}


def assign_validated_outer_thirds(assignments: pd.DataFrame) -> pd.DataFrame:
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
    n_outer = len(valid) // 3
    bottom_index = valid.index.to_numpy()[:n_outer]
    middle_index = valid.index.to_numpy()[n_outer:-n_outer]
    top_index = valid.index.to_numpy()[-n_outer:]
    if (len(bottom_index), len(middle_index), len(top_index)) != (20, 21, 20):
        raise AssertionError("Unexpected outer-third sizes")
    bottom_max = float(audit.loc[bottom_index, "preferred_sf_cpd"].max())
    middle_min = float(audit.loc[middle_index, "preferred_sf_cpd"].min())
    middle_max = float(audit.loc[middle_index, "preferred_sf_cpd"].max())
    top_min = float(audit.loc[top_index, "preferred_sf_cpd"].min())
    if not (bottom_max < middle_min and middle_max < top_min):
        raise ValueError("An outer-third boundary would divide an exact preferred-SF tie")
    audit["sf_quartile"] = "excluded"
    audit.loc[middle_index, "sf_quartile"] = "sf_middle_third_excluded"
    audit.loc[bottom_index, "sf_quartile"] = GROUPS[0]
    audit.loc[top_index, "sf_quartile"] = GROUPS[1]
    audit["sf_quartile_label"] = audit.sf_quartile.map(LABELS).fillna("excluded")
    audit["quartile_definition"] = (
        "outer thirds after model-valid and recorded SF-curve r >= 0.5 gate; "
        f"bottom <= {bottom_max:.6g} cpd, middle excluded, top >= {top_min:.6g} cpd"
    )
    return audit


figure.ASSEMBLED = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "assembled/rounds_000_022_n023_clean_history_snapshot_v1"
)
figure.OUT = ROOT / "outputs/fig/ssi_figure_v2/corrected_sf_outer_thirds_clean_history_rounds000_022_v1"
figure.PANELS = figure.OUT / "panels"
figure.STEM = "ssi_figure_v4_corrected_cache_sf_outer_thirds_clean_history_no_bottom_row_rounds000_022_v1"
figure.FIGURE_SCOPE_LABEL = "clean histories · 23 balanced rounds · interim"
figure.GROUPS = GROUPS
figure.LABELS = LABELS
figure.COLORS = COLORS
figure.COMPONENT_GROUP = GROUPS[1]
figure.GROUPING_DESCRIPTION = (
    "61 recorded-validated units split into 20 bottom-SF, 21 excluded middle-SF, and 20 top-SF units"
)
figure.PATH_PANEL_TITLE = "Path-length effects in\npreferred-SF outer thirds"
figure.MATCH_PANEL_TITLE = "Contour-matched effects in\npreferred-SF outer thirds"
figure.COMPONENT_PANEL_TITLE = "Across/along spread in the\ntop-SF third"
figure.assign_validated_quartiles = assign_validated_outer_thirds


if __name__ == "__main__":
    figure.main()
