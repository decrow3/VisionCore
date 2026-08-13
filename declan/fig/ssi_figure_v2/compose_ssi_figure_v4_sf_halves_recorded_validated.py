#!/usr/bin/env python3
"""Build a recorded-validated median-half SSI Figure 4 comparison.

Apply model-valid + recorded SF-curve r >= 0.5 before taking the median split.
The resulting 61 units are divided into 31 low-SF and 30 high-SF units.  This
two-row comparison writes to a new root and preserves all earlier variants.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2 import compose_ssi_figure_v4_sf_outer_thirds as outer
from declan.fig.ssi_figure_v2 import compose_ssi_figure_v4_sf_outer_thirds_recorded_validated as runner


ROOT = outer.ROOT
FIT_CSV = runner.FIT_CSV
ANALYSIS_ROOT = ROOT / "outputs/fig4_active_sensing/backimage_real_trace_sf_halves_recorded_validated_r0p5_v1"
OUT_DIR = ROOT / "outputs/fig/ssi_figure_v2/sf_halves_recorded_validated_r0p5_v1"
PANELS_DIR = OUT_DIR / "panels"
ASSIGNMENTS_OUT = ANALYSIS_ROOT / "sf_half_recorded_validated_unit_assignments.csv"
OUTPUT_STEM = "ssi_figure_v4_sf_halves_recorded_validated_r0p5_no_bottom_row_v1"
GROUPS = ("sf_low_half", "sf_high_half")
LABELS = {"sf_low_half": "low-SF half", "sf_high_half": "high-SF half"}


def prepare_recorded_validated_halves(
    unit: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    fit = pd.read_csv(FIT_CSV)[
        [
            "rr100_index", "recorded_sf_curve_r_full_support",
            "recorded_sf_curve_nrmse_full_support", "recorded_sf_peak_cpd", "sf_fit_r2",
        ]
    ]
    assigned = assignments[
        [
            "rr100_index", "model_valid", "preferred_sf_cpd", "preferred_tf_hz",
            "joint_parametric_surface_r2",
        ]
    ].merge(fit, on="rr100_index", how="left", validate="one_to_one")
    assigned["recorded_validation_pass"] = (
        assigned["model_valid"].fillna(False).astype(bool)
        & assigned["recorded_sf_curve_r_full_support"].ge(runner.RECORDED_R_MIN)
    )
    valid = assigned[assigned["recorded_validation_pass"]].sort_values(
        ["preferred_sf_cpd", "rr100_index"]
    ).reset_index(drop=True)
    if len(valid) != 61:
        raise ValueError(f"Expected 61 recorded-validated units, found {len(valid)}")
    threshold = float(valid["preferred_sf_cpd"].median())
    low = valid[valid["preferred_sf_cpd"] <= threshold]
    high = valid[valid["preferred_sf_cpd"] > threshold]
    if (len(low), len(high)) != (31, 30):
        raise ValueError(f"Unexpected validated half counts: {len(low)}/{len(high)}")
    low_max = float(low["preferred_sf_cpd"].max())
    high_min = float(high["preferred_sf_cpd"].min())
    if not low_max < high_min:
        raise ValueError("Validated median split would divide an exact preferred-SF tie")

    assigned["sf_outer_third"] = "excluded_invalid_parametric_model"
    assigned.loc[
        assigned["model_valid"].fillna(False).astype(bool) & ~assigned["recorded_validation_pass"],
        "sf_outer_third",
    ] = "excluded_failed_recorded_validation"
    assigned.loc[assigned["rr100_index"].isin(low["rr100_index"]), "sf_outer_third"] = GROUPS[0]
    assigned.loc[assigned["rr100_index"].isin(high["rr100_index"]), "sf_outer_third"] = GROUPS[1]
    assigned["sf_outer_third_label"] = assigned["sf_outer_third"].map(
        {
            **LABELS,
            "excluded_failed_recorded_validation": "excluded: recorded SF validation r < 0.5",
            "excluded_invalid_parametric_model": "excluded: invalid parametric model",
        }
    )
    contract: dict[str, float | int | str] = {
        "valid_fits_before_recorded_gate": int(assigned["model_valid"].fillna(False).sum()),
        "recorded_validated_fits": int(len(valid)),
        "valid_fits": int(len(valid)),
        "low_n": int(len(low)),
        "high_n": int(len(high)),
        "recorded_curve_r_min": runner.RECORDED_R_MIN,
        "recorded_validation_gate": "recorded_sf_curve_r_full_support >= 0.5",
        "median_threshold_cpd": threshold,
        "low_half_max_cpd": low_max,
        "high_half_min_cpd": high_min,
    }
    for key, value in contract.items():
        assigned[key] = value
    assigned["rank_definition"] = (
        "model-valid units with recorded_sf_curve_r_full_support >= 0.5; low <= validated "
        "preferred-SF median, high > median"
    )

    selected = unit.merge(assigned, left_on="unit_index", right_on="rr100_index", how="left", validate="one_to_one")
    selected = selected[selected["sf_outer_third"].isin(GROUPS)].copy()
    selected["historical_sf_group"] = selected["sf_group"]
    selected["sf_group"] = selected["sf_outer_third"]
    selected["sf_group_label"] = selected["sf_outer_third_label"]
    selected["sf_group_definition"] = (
        "median halves of model-valid, recorded-validated parametric preferred SF"
    )
    selected["sf_split_metric"] = selected["preferred_sf_cpd"]
    return selected, assigned, contract


def selected_high_half_aligned_images(data: dict) -> dict[int, np.ndarray]:
    selection = pd.read_csv(ANALYSIS_ROOT / "contour_matched/unit_image_selection.csv")
    selection = selection[selection["sf_group"].eq(GROUPS[1])]
    available = set(data["unit"]["unit_index"].astype(int))
    output: dict[int, np.ndarray] = {}
    for row in selection.itertuples(index=False):
        unit_index = int(row.unit_index)
        if unit_index not in available:
            raise ValueError(f"Unknown high-half unit: {unit_index}")
        text = str(row.selected_image_indices).strip()
        images = np.asarray([int(value) for value in text.split()], dtype=int) if text else np.asarray([], dtype=int)
        if images.size:
            output[unit_index] = images
    return output


def compute_rms_values() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = outer.rms_analysis.panel_c.load_dataset(outer.rms_analysis.panel_c.MATRIX_DIR)
    metrics = outer.rms_analysis._compute_extended_component_metrics(data)
    reference_map = outer.rms_analysis._reference_context_by_family(metrics)
    family = next(item for item in outer.rms_analysis.FAMILIES if item["key"] == "component_rms")
    population = {
        "key": outer.dose_plot.POPULATION_KEY,
        "title": "Aligned Recorded-Validated High-SF Half",
        "subtitle": "upper half after recorded SF-curve r >= 0.5 gate",
        "sf_group": "high_half", "relation": "aligned", "requires_orientation_tuning": True,
    }
    unit_to_images = selected_high_half_aligned_images(data)
    values, meta = outer.rms_analysis._compute_family(
        data, metrics, family, population=population, population_index=1, family_index=1,
        unit_to_images=unit_to_images,
    )
    contrasts = pd.DataFrame(meta["contrast"])
    populations = pd.DataFrame([{
        "population_key": outer.dose_plot.POPULATION_KEY,
        "population_title": population["title"],
        "population_subtitle": population["subtitle"],
        "sf_group": GROUPS[1], "relation": "aligned", "requires_orientation_tuning": True,
        "n_selected_units": len(unit_to_images),
        "n_selected_unit_image_pairs": sum(len(images) for images in unit_to_images.values()),
    }])
    reference = pd.DataFrame([{"metric_family": key, **context} for key, context in reference_map.items()])
    return values, contrasts, populations, reference


def configure() -> None:
    outer.GROUPS = GROUPS
    outer.LABELS = LABELS
    outer.ANALYSIS_ROOT = ANALYSIS_ROOT
    outer.OUT_DIR = OUT_DIR
    outer.PANELS_DIR = PANELS_DIR
    outer.ASSIGNMENTS_OUT = ASSIGNMENTS_OUT
    outer.OUTPUT_STEM = OUTPUT_STEM
    outer.prepare_outer_thirds = prepare_recorded_validated_halves
    outer.compute_rms_values = compute_rms_values
    runner.OUT_DIR = OUT_DIR
    runner.PANELS_DIR = PANELS_DIR
    runner.ANALYSIS_ROOT = ANALYSIS_ROOT
    runner.ASSIGNMENTS_OUT = ASSIGNMENTS_OUT
    runner.OUTPUT_STEM = OUTPUT_STEM
    runner.prepare_recorded_validated_outer_thirds = prepare_recorded_validated_halves
    runner.LOW_GROUP_DISPLAY = "low-SF"
    runner.HIGH_GROUP_DISPLAY = "high-SF"
    runner.PATH_PANEL_TITLE = "Path length separates low- and\nhigh-SF halves"
    runner.ALIGN_PANEL_TITLE = "Contour alignment exposes a\nhigh-SF limit"
    runner.RMS_PANEL_TITLE = "Across-contour spread limits\nhigh-SF benefit"


def run_stage(stage: str) -> None:
    configure()
    runner.run_stage(stage)
    if stage == "assemble":
        provenance_path = OUT_DIR / f"{OUTPUT_STEM}_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["split"] = "median halves after recorded-validation gate"
        provenance["sf_half_contract"] = provenance.pop("sf_contract")
        contract = provenance["sf_half_contract"]
        for stale_key in ("bottom_n", "middle_excluded_n", "top_n"):
            contract.pop(stale_key, None)
        contract["rule"] = "low <= validated median; high > validated median"
        contract["tie_check"] = "validated median boundary does not split an exact preferred-SF tie"
        provenance["previous_half_split_untouched"] = str(
            ROOT / "outputs/fig/ssi_figure_v2/sf_halves_v1/ssi_figure_v4_sf_halves_v1.pdf"
        )
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("analysis", "a", "d", "bc", "ef", "g", "assemble"))
    args = parser.parse_args()
    if args.stage:
        run_stage(args.stage)
        return
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    module = "declan.fig.ssi_figure_v2.compose_ssi_figure_v4_sf_halves_recorded_validated"
    for stage in ("analysis", "a", "d", "bc", "ef", "g", "assemble"):
        subprocess.run([sys.executable, "-m", module, "--stage", stage], check=True, env=env)


if __name__ == "__main__":
    main()
