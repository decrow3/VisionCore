#!/usr/bin/env python3
"""Build a recorded-validated outer-third SSI Figure 4 variant.

The recorded validation gate is applied before ranking preferred SF.  Only
units with a valid parametric model and recorded-versus-parametric SF curve
Pearson r >= 0.5 enter the ranked population.  Bottom and top thirds are then
recomputed within that validated population; the middle third is excluded.

This configures the existing outer-third compositor to write to a distinct
analysis and figure root, leaving every earlier figure untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from declan.fig.ssi_figure_v2 import compose_ssi_figure_v4_sf_outer_thirds as outer


ROOT = outer.ROOT
FIT_CSV = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/rr100_sf_tf_parametric_models.csv"
ANALYSIS_ROOT = ROOT / (
    "outputs/fig4_active_sensing/"
    "backimage_real_trace_sf_outer_thirds_recorded_validated_r0p5_v1"
)
OUT_DIR = ROOT / "outputs/fig/ssi_figure_v2/sf_outer_thirds_recorded_validated_r0p5_v1"
PANELS_DIR = OUT_DIR / "panels"
ASSIGNMENTS_OUT = ANALYSIS_ROOT / "sf_outer_third_recorded_validated_unit_assignments.csv"
OUTPUT_STEM = "ssi_figure_v4_sf_outer_thirds_recorded_validated_r0p5_no_bottom_row_v1"
RECORDED_R_MIN = 0.5
LOW_GROUP_DISPLAY = "bottom-SF"
HIGH_GROUP_DISPLAY = "top-SF"
PATH_PANEL_TITLE = "Path length separates bottom- and\ntop-SF thirds"
ALIGN_PANEL_TITLE = "Contour alignment exposes a\ntop-SF limit"
RMS_PANEL_TITLE = "Across-contour spread limits\ntop-SF benefit"


def prepare_recorded_validated_outer_thirds(
    unit: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    fit = pd.read_csv(FIT_CSV)[
        [
            "rr100_index",
            "recorded_sf_curve_r_full_support",
            "recorded_sf_curve_nrmse_full_support",
            "recorded_sf_peak_cpd",
            "sf_fit_r2",
        ]
    ]
    columns = [
        "rr100_index",
        "model_valid",
        "preferred_sf_cpd",
        "preferred_tf_hz",
        "joint_parametric_surface_r2",
    ]
    audit = assignments[columns].merge(fit, on="rr100_index", how="left", validate="one_to_one")
    audit["recorded_validation_pass"] = (
        audit["model_valid"].fillna(False).astype(bool)
        & audit["recorded_sf_curve_r_full_support"].ge(RECORDED_R_MIN)
    )
    validated = audit[audit["recorded_validation_pass"]].copy()
    validated = validated.sort_values(["preferred_sf_cpd", "rr100_index"]).reset_index(drop=True)
    if len(validated) != 61:
        raise ValueError(f"Expected 61 recorded-validated fits, found {len(validated)}")
    n_outer = len(validated) // 3
    bottom = validated.iloc[:n_outer]
    middle = validated.iloc[n_outer : len(validated) - n_outer]
    top = validated.iloc[len(validated) - n_outer :]
    boundaries: dict[str, float | int | str] = {
        "valid_fits_before_recorded_gate": int(audit["model_valid"].fillna(False).sum()),
        "recorded_validated_fits": int(len(validated)),
        "valid_fits": int(len(validated)),
        "bottom_n": int(len(bottom)),
        "middle_excluded_n": int(len(middle)),
        "top_n": int(len(top)),
        "recorded_curve_r_min": RECORDED_R_MIN,
        "recorded_validation_gate": "recorded_sf_curve_r_full_support >= 0.5",
        "bottom_max_cpd": float(bottom["preferred_sf_cpd"].max()),
        "middle_min_cpd": float(middle["preferred_sf_cpd"].min()),
        "middle_max_cpd": float(middle["preferred_sf_cpd"].max()),
        "top_min_cpd": float(top["preferred_sf_cpd"].min()),
    }
    if not (
        boundaries["bottom_max_cpd"] < boundaries["middle_min_cpd"]
        and boundaries["middle_max_cpd"] < boundaries["top_min_cpd"]
    ):
        raise ValueError(f"A recorded-validated outer-third boundary splits a tie: {boundaries}")

    audit["sf_outer_third"] = "excluded_invalid_parametric_model"
    audit.loc[
        audit["model_valid"].fillna(False).astype(bool) & ~audit["recorded_validation_pass"],
        "sf_outer_third",
    ] = "excluded_failed_recorded_validation"
    audit.loc[audit["rr100_index"].isin(bottom["rr100_index"]), "sf_outer_third"] = outer.GROUPS[0]
    audit.loc[audit["rr100_index"].isin(middle["rr100_index"]), "sf_outer_third"] = "sf_middle_third"
    audit.loc[audit["rr100_index"].isin(top["rr100_index"]), "sf_outer_third"] = outer.GROUPS[1]
    audit["sf_outer_third_label"] = audit["sf_outer_third"].map(
        {
            **outer.LABELS,
            "sf_middle_third": "validated middle SF third",
            "excluded_failed_recorded_validation": "excluded: recorded SF validation r < 0.5",
            "excluded_invalid_parametric_model": "excluded: invalid parametric model",
        }
    )
    for key, value in boundaries.items():
        audit[key] = value
    audit["rank_definition"] = (
        "model-valid units with recorded_sf_curve_r_full_support >= 0.5, sorted by "
        "preferred_sf_cpd then rr100_index; 20 bottom, 21 middle, 20 top"
    )

    selected = unit.merge(audit, left_on="unit_index", right_on="rr100_index", how="left", validate="one_to_one")
    selected = selected[selected["sf_outer_third"].isin(outer.GROUPS)].copy()
    selected["historical_sf_group"] = selected["sf_group"]
    selected["sf_group"] = selected["sf_outer_third"]
    selected["sf_group_label"] = selected["sf_outer_third_label"]
    selected["sf_group_definition"] = (
        "outer thirds of model-valid, recorded-validated parametric preferred SF; middle third excluded"
    )
    selected["sf_split_metric"] = selected["preferred_sf_cpd"]
    counts = selected["sf_group"].value_counts().reindex(outer.GROUPS, fill_value=0).to_dict()
    if counts != {outer.GROUPS[0]: 20, outer.GROUPS[1]: 20}:
        raise ValueError(f"Unexpected recorded-validated outer-third counts: {counts}")
    return selected, audit, boundaries


def configure_variant() -> None:
    outer.ANALYSIS_ROOT = ANALYSIS_ROOT
    outer.OUT_DIR = OUT_DIR
    outer.PANELS_DIR = PANELS_DIR
    outer.ASSIGNMENTS_OUT = ASSIGNMENTS_OUT
    outer.OUTPUT_STEM = OUTPUT_STEM
    outer.prepare_outer_thirds = prepare_recorded_validated_outer_thirds


def stage_path(name: str) -> Path:
    return OUT_DIR / f".stage_{name}.csv"


def run_stage(stage: str) -> None:
    configure_variant()
    if stage == "analysis":
        outer.build_analysis()
        return
    outer.configure_base()
    outer.base.RMS_PANEL_TITLE = RMS_PANEL_TITLE
    outer.base.GROUP_LABELS = {
        outer.GROUPS[0]: f"recorded-validated {LOW_GROUP_DISPLAY}",
        outer.GROUPS[1]: f"recorded-validated {HIGH_GROUP_DISPLAY}",
    }
    outer.base.path_panel._support_note_label = lambda label: {
        "B": f"validated {LOW_GROUP_DISPLAY}",
        "C": f"validated {HIGH_GROUP_DISPLAY}",
        "E": f"validated {LOW_GROUP_DISPLAY}",
        "F": f"validated {HIGH_GROUP_DISPLAY}",
    }.get(label, label)
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    if stage == "a":
        outer.base.panel_a_motion_schematic.build_panel(
            figsize=outer.base.PLACEMENT_BOXES["A"][2:4], out_dir=PANELS_DIR,
            panel_label="A", panel_title="FEMs sharpen spatial coding",
        )
        return
    if stage == "d":
        outer.base.panel_d_contour_relative_stimulus.build_panel(
            figsize=outer.base.PLACEMENT_BOXES["D"][2:4], out_dir=PANELS_DIR,
            panel_label="C", panel_title="Local contours define the\nrelevant image axis",
        )
        return
    if stage == "bc":
        _path, table = outer.base._build_path_pair_panel(
            "strong_contours_no_osi", labels=("B", "C"),
            figsize=outer.base.PLACEMENT_BOXES["BC"][2:4], panel_label="B",
            panel_title=PATH_PANEL_TITLE,
            axes_box=outer.base.path_panel.TOP_ROW_PAIR_AXES_BOX, separate_header=True,
        )
        table.to_csv(stage_path("panel_b"), index=False)
        return
    if stage == "ef":
        _path, table = outer.base._build_path_pair_panel(
            "contour_matched", labels=("E", "F"),
            figsize=outer.base.PLACEMENT_BOXES["EF"][2:4], panel_label="D",
            panel_title=ALIGN_PANEL_TITLE,
            axes_box=outer.base.panel_header.MIDDLE_ROW_AXES_BOX, separate_header=False,
            xlabel="path length (arcmin; irrespective of\nspatial footprint)",
        )
        table.to_csv(stage_path("panel_d"), index=False)
        return
    if stage == "g":
        result = outer.base._build_rms_panel()
        _path, values, contrasts, populations, reference = result
        values.to_csv(stage_path("panel_e"), index=False)
        contrasts.to_csv(stage_path("panel_e_contrasts"), index=False)
        populations.to_csv(stage_path("panel_e_populations"), index=False)
        reference.to_csv(stage_path("panel_e_reference"), index=False)
        return
    if stage != "assemble":
        raise ValueError(f"Unknown stage: {stage}")

    def staged_panel_set() -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
        paths = {
            "A": PANELS_DIR / "panel_a.pdf",
            "D": PANELS_DIR / "panel_d.pdf",
            "BC": PANELS_DIR / "panel_b_sf_halves.pdf",
            "EF": PANELS_DIR / "panel_d_sf_halves.pdf",
            "G": PANELS_DIR / "panel_e_rms_sf_high_half.pdf",
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing staged panels: {missing}")
        tables = {
            "panel_b": pd.read_csv(stage_path("panel_b")),
            "panel_d": pd.read_csv(stage_path("panel_d")),
            "panel_e": pd.read_csv(stage_path("panel_e")),
            "panel_e_contrasts": pd.read_csv(stage_path("panel_e_contrasts")),
            "panel_e_populations": pd.read_csv(stage_path("panel_e_populations")),
            "panel_e_reference": pd.read_csv(stage_path("panel_e_reference")),
        }
        return paths, tables

    outer.build_panel_set = staged_panel_set
    paths = outer.compose()
    provenance_path = paths["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["validation_note"] = (
        "Recorded-validation r >= 0.5 was applied before preferred-SF ranking and outer-third assignment."
    )
    provenance["previous_outer_thirds_variant_untouched"] = str(
        ROOT / "outputs/fig/ssi_figure_v2/sf_outer_thirds_v1/ssi_figure_v4_sf_outer_thirds_no_bottom_row_v1.pdf"
    )
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in paths.values():
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("analysis", "a", "d", "bc", "ef", "g", "assemble"), default=None
    )
    args = parser.parse_args()
    if args.stage is not None:
        run_stage(args.stage)
        return
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    module = "declan.fig.ssi_figure_v2.compose_ssi_figure_v4_sf_outer_thirds_recorded_validated"
    for stage in ("analysis", "a", "d", "bc", "ef", "g", "assemble"):
        subprocess.run([sys.executable, "-m", module, "--stage", stage], check=True, env=env)


if __name__ == "__main__":
    main()
