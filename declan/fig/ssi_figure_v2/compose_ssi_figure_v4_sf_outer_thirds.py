#!/usr/bin/env python3
"""Build a two-row SSI Figure 4 using the bottom and top SF thirds.

The 85 valid joint-parametric SF/TF fits are ordered by preferred SF. The
lowest 28 and highest 28 units are retained; the middle 29 are excluded. The
figure contains displayed panels A-E only and writes to a new output root.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pypdf import PdfWriter

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from declan.active_sensing_movie_information import (
    plot_backimage_real_trace_unit_first_and_population_schematics as schematic,
)
from declan.fig.ssi_figure_v2 import compose_ssi_figure_v4_sf_halves as base
from declan.fig.ssi_figure_v2.panels import panel_g_alternative_x_axes_diagnostic as rms_analysis
from declan.fig.ssi_figure_v2.panels import panel_g_option_sheet as dose_plot
from declan.fig4_active_sensing.rerun_backimage_all_images_population_sf_quartiles import (
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    N_DRIFT_BINS,
    N_MICROSACCADE_BINS,
)
from declan.fig4_active_sensing.rerun_backimage_real_trace_contour_matched_sf_quartiles import (
    DEFAULT_ASSIGNMENTS,
    DEFAULT_MATRIX_DIR,
)
from declan.fig4_active_sensing.rerun_backimage_strong_contours_population_sf_quartiles import (
    build_summary,
)


ROOT = base.ROOT
ANALYSIS_ROOT = ROOT / "outputs" / "fig4_active_sensing" / "backimage_real_trace_sf_outer_thirds_v1"
OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "sf_outer_thirds_v1"
PANELS_DIR = OUT_DIR / "panels"
ASSIGNMENTS_OUT = ANALYSIS_ROOT / "sf_outer_third_unit_assignments.csv"
OUTPUT_STEM = "ssi_figure_v4_sf_outer_thirds_no_bottom_row_v1"

GROUPS = ("sf_bottom_third", "sf_top_third")
LABELS = {"sf_bottom_third": "bottom-SF third", "sf_top_third": "top-SF third"}
COLORS = {"sf_bottom_third": base.path_panel.BLUE, "sf_top_third": base.path_panel.ORANGE}
RELATIONS = {
    "strong_contours_no_osi": "strong contour image windows; no OSI gate",
    "contour_matched": "strong contour image windows; orientation aligned",
}


def prepare_outer_thirds(unit: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    columns = [
        "rr100_index", "model_valid", "preferred_sf_cpd", "preferred_tf_hz",
        "joint_parametric_surface_r2",
    ]
    audit = assignments[columns].copy()
    valid = audit[audit["model_valid"].fillna(False).astype(bool)].copy()
    if len(valid) != 85:
        raise ValueError(f"Expected 85 valid parametric fits, found {len(valid)}")
    valid = valid.sort_values(["preferred_sf_cpd", "rr100_index"]).reset_index(drop=True)
    n_outer = len(valid) // 3
    if n_outer != 28:
        raise ValueError(f"Expected 28 units per outer third, found {n_outer}")
    bottom = valid.iloc[:n_outer]
    middle = valid.iloc[n_outer : len(valid) - n_outer]
    top = valid.iloc[len(valid) - n_outer :]
    boundaries = {
        "bottom_max_cpd": float(bottom["preferred_sf_cpd"].max()),
        "middle_min_cpd": float(middle["preferred_sf_cpd"].min()),
        "middle_max_cpd": float(middle["preferred_sf_cpd"].max()),
        "top_min_cpd": float(top["preferred_sf_cpd"].min()),
    }
    if not (boundaries["bottom_max_cpd"] < boundaries["middle_min_cpd"]
            and boundaries["middle_max_cpd"] < boundaries["top_min_cpd"]):
        raise ValueError(f"An outer-third boundary splits an exact preferred-SF tie: {boundaries}")

    audit["sf_outer_third"] = "invalid_model"
    audit.loc[audit["rr100_index"].isin(bottom["rr100_index"]), "sf_outer_third"] = GROUPS[0]
    audit.loc[audit["rr100_index"].isin(middle["rr100_index"]), "sf_outer_third"] = "sf_middle_third"
    audit.loc[audit["rr100_index"].isin(top["rr100_index"]), "sf_outer_third"] = GROUPS[1]
    audit["sf_outer_third_label"] = audit["sf_outer_third"].map({**LABELS, "sf_middle_third": "middle SF third"})
    for key, value in boundaries.items():
        audit[key] = value
    audit["rank_definition"] = "valid fits sorted by preferred_sf_cpd then rr100_index; 28 bottom, 29 middle, 28 top"

    selected = unit.merge(audit, left_on="unit_index", right_on="rr100_index", how="left", validate="one_to_one")
    selected = selected[selected["sf_outer_third"].isin(GROUPS)].copy()
    selected["historical_sf_group"] = selected["sf_group"]
    selected["sf_group"] = selected["sf_outer_third"]
    selected["sf_group_label"] = selected["sf_outer_third_label"]
    selected["sf_group_definition"] = "outer thirds of valid joint-parametric preferred SF; middle third excluded"
    selected["sf_split_metric"] = selected["preferred_sf_cpd"]
    counts = selected["sf_group"].value_counts().reindex(GROUPS, fill_value=0).to_dict()
    if counts != {GROUPS[0]: 28, GROUPS[1]: 28}:
        raise ValueError(f"Unexpected outer-third counts: {counts}")
    return selected, audit, boundaries


def build_analysis() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    data = schematic.load_dataset(DEFAULT_MATRIX_DIR)
    assignments = pd.read_csv(DEFAULT_ASSIGNMENTS)
    unit_outer, assignment_audit, boundaries = prepare_outer_thirds(data["unit"], assignments)
    trace, trace_bins = schematic.add_equal_count_trace_bins(
        data["trace"], n_drift_bins=N_DRIFT_BINS, n_microsaccade_bins=N_MICROSACCADE_BINS,
    )
    row_grid = schematic.build_movie_row_grid(data["movie"])
    baseline_lookup = schematic.baseline_rows_by_image(data["image"], data["baseline_table"])
    outputs: dict[str, pd.DataFrame] = {}
    for relation, relation_label in RELATIONS.items():
        relation_dir = ANALYSIS_ROOT / relation
        relation_dir.mkdir(parents=True, exist_ok=True)
        selection, _curves, unit_summary, population = build_summary(
            data, unit_outer, list(GROUPS), trace, trace_bins, row_grid, baseline_lookup,
            n_bootstrap=N_BOOTSTRAP, relation=relation, relation_label=relation_label,
        )
        selection.to_csv(relation_dir / "unit_image_selection.csv", index=False)
        unit_summary.to_csv(relation_dir / "unit_first_summary.csv", index=False)
        population.to_csv(relation_dir / "spike_weighted_population_summary.csv", index=False)
        outputs[relation] = population
    assignment_audit.to_csv(ASSIGNMENTS_OUT, index=False)
    trace_bins.to_csv(ANALYSIS_ROOT / "trace_path_bin_definitions.csv", index=False)
    return outputs, {"data": data, "boundaries": boundaries, "assignment_audit": assignment_audit}


def selected_top_third_aligned_images(data: dict[str, Any]) -> dict[int, np.ndarray]:
    selection = pd.read_csv(ANALYSIS_ROOT / "contour_matched" / "unit_image_selection.csv")
    selection = selection[selection["sf_group"].eq("sf_top_third")]
    available_units = set(data["unit"]["unit_index"].astype(int))
    selected: dict[int, np.ndarray] = {}
    for row in selection.itertuples(index=False):
        unit_index = int(row.unit_index)
        if unit_index not in available_units:
            raise ValueError(f"Unknown unit in top-third contour selection: {unit_index}")
        text = str(row.selected_image_indices).strip()
        images = np.asarray([int(value) for value in text.split()], dtype=int) if text else np.asarray([], dtype=int)
        if images.size:
            selected[unit_index] = images
    return selected


def compute_rms_values() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = rms_analysis.panel_c.load_dataset(rms_analysis.panel_c.MATRIX_DIR)
    metrics = rms_analysis._compute_extended_component_metrics(data)
    reference_map = rms_analysis._reference_context_by_family(metrics)
    family = next(item for item in rms_analysis.FAMILIES if item["key"] == "component_rms")
    population = {
        "key": dose_plot.POPULATION_KEY,
        "title": "Aligned Top-SF Third",
        "subtitle": "top third of valid parametric preferred SF; audited contour-matched selection",
        "sf_group": "top_third", "relation": "aligned", "requires_orientation_tuning": True,
    }
    unit_to_images = selected_top_third_aligned_images(data)
    values, meta = rms_analysis._compute_family(
        data, metrics, family, population=population, population_index=2, family_index=1,
        unit_to_images=unit_to_images,
    )
    contrasts = pd.DataFrame(meta["contrast"])
    populations = pd.DataFrame([{
        "population_key": dose_plot.POPULATION_KEY,
        "population_title": population["title"],
        "population_subtitle": population["subtitle"],
        "sf_group": "sf_top_third", "relation": "aligned", "requires_orientation_tuning": True,
        "n_selected_units": len(unit_to_images),
        "n_selected_unit_image_pairs": sum(len(images) for images in unit_to_images.values()),
    }])
    reference = pd.DataFrame([{"metric_family": key, **context} for key, context in reference_map.items()])
    return values, contrasts, populations, reference


def configure_base() -> None:
    base.HALF_ROOT = ANALYSIS_ROOT
    base.ASSIGNMENTS_CSV = ASSIGNMENTS_OUT
    base.OUT_DIR = OUT_DIR
    base.PANELS_DIR = PANELS_DIR
    base.OUTPUT_STEM = OUTPUT_STEM
    base.GROUPS = GROUPS
    base.GROUP_LABELS = {GROUPS[0]: "bottom-SF", GROUPS[1]: "top-SF"}
    base.GROUP_TO_LEGACY_KEY = {GROUPS[0]: "low_lt0p5", GROUPS[1]: "high_ge0p75"}
    base.COLORS = COLORS
    base.RMS_PANEL_TITLE = "Across-contour spread limits\ntop-SF benefit"
    base._compute_rms_values = compute_rms_values
    base.path_panel._support_note_label = lambda label: {
        "B": "bottom-SF", "C": "top-SF", "E": "bottom-SF", "F": "top-SF",
    }.get(label, label)


def build_panel_set() -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    tables: dict[str, pd.DataFrame] = {}
    paths["A"] = base.panel_a_motion_schematic.build_panel(
        figsize=base.PLACEMENT_BOXES["A"][2:4], out_dir=PANELS_DIR,
        panel_label="A", panel_title="FEMs sharpen spatial coding",
    )
    paths["D"] = base.panel_d_contour_relative_stimulus.build_panel(
        figsize=base.PLACEMENT_BOXES["D"][2:4], out_dir=PANELS_DIR,
        panel_label="C", panel_title="Local contours define the\nrelevant image axis",
    )
    paths["BC"], tables["panel_b"] = base._build_path_pair_panel(
        "strong_contours_no_osi", labels=("B", "C"), figsize=base.PLACEMENT_BOXES["BC"][2:4],
        panel_label="B", panel_title="Path length separates bottom- and\ntop-SF thirds",
        axes_box=base.path_panel.TOP_ROW_PAIR_AXES_BOX, separate_header=True,
    )
    paths["EF"], tables["panel_d"] = base._build_path_pair_panel(
        "contour_matched", labels=("E", "F"), figsize=base.PLACEMENT_BOXES["EF"][2:4],
        panel_label="D", panel_title="Contour alignment exposes a\ntop-SF limit",
        axes_box=base.panel_header.MIDDLE_ROW_AXES_BOX, separate_header=False,
        xlabel="path length (arcmin; irrespective of\nspatial footprint)",
    )
    paths["G"], tables["panel_e"], tables["panel_e_contrasts"], tables["panel_e_populations"], tables["panel_e_reference"] = base._build_rms_panel()
    return paths, tables


def compose() -> dict[str, Path]:
    populations, context = build_analysis()
    configure_base()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_paths, tables = build_panel_set()

    page_w_in = base.layout.PAGE_SIZE_IN[0]
    page_h_in = 7.85
    page_w_pt, page_h_pt = page_w_in * 72.0, page_h_in * 72.0
    writer = PdfWriter()
    writer.add_blank_page(width=page_w_pt, height=page_h_pt)
    page = writer.pages[0]
    for key in ("A", "BC", "D", "EF", "G"):
        x_in, y_in, _width, _height = base.PLACEMENT_BOXES[key]
        base._place(writer, page, panel_paths[key], x_in, y_in, page_h_pt)
    pdf = OUT_DIR / f"{OUTPUT_STEM}.pdf"
    with pdf.open("wb") as handle:
        writer.write(handle)

    table_paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = OUT_DIR / f"{OUTPUT_STEM}_{name}_values.csv"
        table.to_csv(path, index=False)
        table_paths[name] = path
    endpoint_rows = []
    for relation, population in populations.items():
        for group in GROUPS:
            for context_name in ("drift_only", "microsaccade"):
                row = population[population["sf_group"].eq(group) & population["context"].eq(context_name)].sort_values("path_bin_order").iloc[0]
                endpoint_rows.append({
                    "relation": relation, "sf_group": group, "context": context_name,
                    "path_median_arcmin": float(row.path_median_arcmin),
                    "delta_bits_per_spike": float(row.population_ssi_delta_vs_stabilized),
                    "ci95_low": float(row.population_delta_ci95_low_image_boot),
                    "ci95_high": float(row.population_delta_ci95_high_image_boot),
                })
    endpoint_path = OUT_DIR / f"{OUTPUT_STEM}_endpoint_audit.csv"
    pd.DataFrame(endpoint_rows).to_csv(endpoint_path, index=False)

    weighting_rows = []
    for relation, population in populations.items():
        unit_summary = pd.read_csv(ANALYSIS_ROOT / relation / "unit_first_summary.csv")
        for group in GROUPS:
            pop_group = population[population["sf_group"].eq(group) & population["context"].eq("drift_only")].sort_values("path_bin_order")
            unit_group = unit_summary[unit_summary["sf_group"].eq(group) & unit_summary["context"].eq("drift_only")].sort_values("path_bin_order")
            for endpoint, index in (("first_drift_bin", 0), ("last_drift_bin", -1)):
                pop_row = pop_group.iloc[index]
                unit_row = unit_group.iloc[index]
                pop_delta = float(pop_row.population_ssi_delta_vs_stabilized)
                unit_delta = float(unit_row.mean_unit_ssi_delta_vs_stabilized)
                weighting_rows.append({
                    "relation": relation,
                    "sf_group": group,
                    "endpoint": endpoint,
                    "path_median_arcmin": float(pop_row.path_median_arcmin),
                    "spike_weighted_delta_bits_per_spike": pop_delta,
                    "spike_weighted_ci95_low": float(pop_row.population_delta_ci95_low_image_boot),
                    "spike_weighted_ci95_high": float(pop_row.population_delta_ci95_high_image_boot),
                    "equal_unit_mean_delta_bits_per_spike": unit_delta,
                    "equal_unit_sem": float(unit_row.sem_unit_ssi_delta_vs_stabilized),
                    "equal_unit_median_delta_bits_per_spike": float(unit_row.median_unit_ssi_delta_vs_stabilized),
                    "population_equal_unit_sign_agreement": bool(np.sign(pop_delta) == np.sign(unit_delta)),
                })
    weighting_path = OUT_DIR / f"{OUTPUT_STEM}_weighting_audit.csv"
    pd.DataFrame(weighting_rows).to_csv(weighting_path, index=False)

    provenance_path = OUT_DIR / f"{OUTPUT_STEM}_provenance.json"
    provenance = {
        "figure": OUTPUT_STEM,
        "layout": "two rows only; original bottom row F-H omitted",
        "page_size_in": [page_w_in, page_h_in],
        "sf_contract": {
            "metric": "preferred_sf_cpd from joint parametric SF/TF fit",
            "valid_fits": 85, "bottom_n": 28, "middle_excluded_n": 29, "top_n": 28,
            **context["boundaries"],
            "tie_check": "neither outer-third boundary splits an exact preferred-SF tie",
        },
        "updated_panels": {"B": "outer-thirds total path", "D": "aligned outer-thirds total path", "E": "aligned top-third component RMS"},
        "unchanged_panels": ["A", "C"],
        "omitted_panels": ["F", "G", "H"],
        "analysis_root": str(ANALYSIS_ROOT.relative_to(ROOT)),
        "panel_paths": {key: str(path) for key, path in panel_paths.items()},
        "table_paths": {key: str(path) for key, path in table_paths.items()},
        "endpoint_audit": str(endpoint_path),
        "weighting_audit": str(weighting_path),
        "output_pdf": str(pdf),
        "bootstrap": {"n_image_resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED},
    }
    provenance_path.write_text(json.dumps(base._json_ready(provenance), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "pdf": pdf, "provenance": provenance_path, "endpoint_audit": endpoint_path,
        "weighting_audit": weighting_path, **table_paths,
    }


def main() -> None:
    for path in compose().values():
        print(path)


if __name__ == "__main__":
    main()
