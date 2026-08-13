#!/usr/bin/env python3
"""Checkpoints 15-17: finish historical pages 24-26 with new SF quartiles."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from declan.active_sensing_movie_information import (
    plot_backimage_real_trace_unit_first_and_population_schematics as schematic,
)
from declan.fig4_active_sensing.rerun_backimage_all_images_component_sf_quartiles import (
    build_component_summary,
    endpoint_audit as component_endpoint_audit,
    make_direct_overlay,
    prepare_component_inputs,
    validate_historical as validate_component_historical,
)
from declan.fig4_active_sensing.rerun_backimage_all_images_population_sf_quartiles import (
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    N_DRIFT_BINS,
    N_MICROSACCADE_BINS,
    endpoint_audit,
    make_unit_weighting_figure,
    prepare_quartile_units,
)
from declan.fig4_active_sensing.rerun_backimage_real_trace_contour_matched_sf_quartiles import (
    COLORS,
    DEFAULT_ASSIGNMENTS,
    DEFAULT_MATRIX_DIR,
    GROUPS,
    LABELS,
    ROOT,
    configure_matplotlib,
    file_identity,
)
from declan.fig4_active_sensing.rerun_backimage_strong_contours_population_sf_quartiles import (
    build_summary,
    validate_historical as validate_population_historical,
)


BASE_OUT = ROOT / "outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1"
DEFAULT_OUT_15 = BASE_OUT / "checkpoint_15_contour_orthogonal_population_sf_quartiles"
DEFAULT_OUT_16 = BASE_OUT / "checkpoint_16_contour_orthogonal_component_sf_quartiles"
DEFAULT_OUT_17 = BASE_OUT / "checkpoint_17_mixed_context_sf_quartiles"
DEFAULT_ALL_IMAGES = BASE_OUT / "checkpoint_07_all_images_population_sf_quartiles/spike_weighted_population_summary.csv"
DEFAULT_CONTOUR_MATCHED = BASE_OUT / "checkpoint_11_contour_matched_population_sf_quartiles/spike_weighted_population_summary.csv"
RELATION = "contour_orthogonal"
RELATION_LABEL = "strong contour image windows; orientation orthogonal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--assignments-csv", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--out-dir-15", type=Path, default=DEFAULT_OUT_15)
    parser.add_argument("--out-dir-16", type=Path, default=DEFAULT_OUT_16)
    parser.add_argument("--out-dir-17", type=Path, default=DEFAULT_OUT_17)
    parser.add_argument("--all-images-summary", type=Path, default=DEFAULT_ALL_IMAGES)
    parser.add_argument("--contour-matched-summary", type=Path, default=DEFAULT_CONTOUR_MATCHED)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for out_dir in (args.out_dir_15, args.out_dir_16, args.out_dir_17):
        out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    matrix_dir = Path(args.matrix_dir)
    data = schematic.load_dataset(matrix_dir)
    assignments = pd.read_csv(args.assignments_csv)
    unit_new = prepare_quartile_units(data["unit"], assignments)
    row_grid = schematic.build_movie_row_grid(data["movie"])
    baseline_lookup = schematic.baseline_rows_by_image(data["image"], data["baseline_table"])

    # Checkpoint 15: total trace path, orientation-orthogonal pairs.
    trace, trace_bins = schematic.add_equal_count_trace_bins(
        data["trace"], n_drift_bins=N_DRIFT_BINS,
        n_microsaccade_bins=N_MICROSACCADE_BINS,
    )
    population_validation = validate_population_historical(
        data, trace, trace_bins, row_grid, baseline_lookup, matrix_dir,
        n_bootstrap=int(args.n_bootstrap), relation=RELATION,
        relation_label=RELATION_LABEL,
    )
    if not population_validation["passed"]:
        raise ValueError(f"Historical page-24 reconstruction failed: {population_validation}")
    selection, unit_curves, unit_summary, population = build_summary(
        data, unit_new, list(GROUPS), trace, trace_bins, row_grid,
        baseline_lookup, n_bootstrap=int(args.n_bootstrap), relation=RELATION,
        relation_label=RELATION_LABEL,
    )
    population_endpoint = endpoint_audit(population, unit_summary)
    schematic.SF_COLORS.update(COLORS)
    schematic.SF_LABELS.update(LABELS)
    schematic.plot_sf_rows_population_panel(
        population, args.out_dir_15,
        stem="024_contour_orthogonal_valid_fit_sf_quartiles_population_absolute_delta",
        title="Spike-weighted population SSI - strong contours, orientation-orthogonal units",
        subtitle=(
            "new SF quartiles; OSI >= 0.05 and unit-contour mismatch >= 67.5 deg; "
            f"{N_DRIFT_BINS} drift-only / {N_MICROSACCADE_BINS} microsaccade bins"
        ),
        sf_groups=list(GROUPS), dpi=220,
    )
    make_unit_weighting_figure(
        population, unit_summary, args.out_dir_15,
        title="Population weighting audit: strong contours and orientation-orthogonal units",
        stem="checkpoint_15_population_vs_equal_unit_audit",
    )
    trace_bins.to_csv(args.out_dir_15 / "trace_path_bin_definitions.csv", index=False)
    selection.to_csv(args.out_dir_15 / "unit_image_selection.csv", index=False)
    unit_curves.to_csv(args.out_dir_15 / "unit_first_curves.csv", index=False)
    unit_summary.to_csv(args.out_dir_15 / "unit_first_summary.csv", index=False)
    population.to_csv(args.out_dir_15 / "spike_weighted_population_summary.csv", index=False)
    population_endpoint.to_csv(args.out_dir_15 / "population_vs_unit_first_endpoint_audit.csv", index=False)
    save_json(args.out_dir_15 / "historical_contract_validation.json", population_validation)

    # Checkpoint 16: across/along component path, same orthogonal pairs.
    component_metrics, bins_by_metric, bins_table = prepare_component_inputs(data)
    component_validation = validate_component_historical(
        data, component_metrics, bins_by_metric, row_grid, baseline_lookup,
        matrix_dir, n_bootstrap=int(args.n_bootstrap), relation=RELATION,
        relation_label=RELATION_LABEL,
    )
    if not component_validation["passed"]:
        raise ValueError(f"Historical page-25 reconstruction failed: {component_validation}")
    component_selection, component_summary = build_component_summary(
        data, unit_new, list(GROUPS), component_metrics, bins_by_metric,
        row_grid, baseline_lookup, n_bootstrap=int(args.n_bootstrap),
        relation=RELATION, relation_label=RELATION_LABEL,
    )
    component_endpoint = component_endpoint_audit(component_summary)
    schematic.plot_component_population_12_panel(
        component_summary, args.out_dir_16,
        stem="025_contour_orthogonal_valid_fit_sf_quartiles_across_along_components",
        title="Spike-weighted population SSI - orientation-orthogonal: across and along components",
        sf_groups=list(GROUPS), dpi=220,
    )
    make_direct_overlay(
        component_summary, args.out_dir_16,
        stem="checkpoint_16_across_vs_along_direct_overlay",
        title="Orientation-orthogonal units: across- versus along-contour population modulation",
    )
    bins_table.to_csv(args.out_dir_16 / "component_path_bin_definitions.csv", index=False)
    component_selection.to_csv(args.out_dir_16 / "unit_image_selection.csv", index=False)
    component_summary.to_csv(args.out_dir_16 / "spike_weighted_population_component_summary.csv", index=False)
    component_endpoint.to_csv(args.out_dir_16 / "component_endpoint_audit.csv", index=False)
    save_json(args.out_dir_16 / "historical_contract_validation.json", component_validation)

    # Checkpoint 17: presentation-only mixed-context view from validated checkpoints 7 and 11.
    all_images = pd.read_csv(args.all_images_summary)
    contour_matched = pd.read_csv(args.contour_matched_summary)
    expected_groups = set(GROUPS)
    if set(all_images["sf_group"].unique()) != expected_groups:
        raise ValueError("all-images summary does not contain exactly the four new SF quartiles")
    if set(contour_matched["sf_group"].unique()) != expected_groups:
        raise ValueError("contour-matched summary does not contain exactly the four new SF quartiles")
    schematic.plot_sf_rows_population_panel(
        contour_matched, args.out_dir_17,
        stem="026_all_images_absolute_contour_matched_delta_valid_fit_sf_quartiles",
        title="SF quartiles: spike-weighted population SSI",
        subtitle=(
            "left: all image windows, no OSI gate; right: strong-contour, orientation-aligned unit-image pairs; "
            f"{N_DRIFT_BINS} drift-only / {N_MICROSACCADE_BINS} microsaccade bins"
        ),
        sf_groups=list(GROUPS), dpi=220,
        absolute_summary=all_images,
        modulation_summary=contour_matched,
        absolute_column_title="Absolute SSI\nall images, no OSI gate",
        modulation_column_title="Movement modulation\nstrong contours, orientation-aligned",
    )
    all_images.assign(panel_source="all_images_absolute").to_csv(
        args.out_dir_17 / "all_images_absolute_summary.csv", index=False,
    )
    contour_matched.assign(panel_source="contour_matched_modulation").to_csv(
        args.out_dir_17 / "contour_matched_modulation_summary.csv", index=False,
    )

    counts = selection.groupby("sf_group")["unit_index"].nunique().reindex(GROUPS, fill_value=0)
    common = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "replacement_scope": "new tie-aware SF quartiles among 85 valid parametric fits",
        "historical_baseline_contract": "trial-mean stabilized reference",
        "bootstrap": {"n_image_resamples": int(args.n_bootstrap), "seed": BOOTSTRAP_SEED},
        "unit_counts": {str(k): int(v) for k, v in counts.items()},
        "sources": {
            "assignments": file_identity(Path(args.assignments_csv)),
            "ssi_matrix": file_identity(matrix_dir / "ssi_matrix.npy"),
            "expected_spikes_matrix": file_identity(matrix_dir / "expected_spikes_matrix.npy"),
        },
    }
    save_json(args.out_dir_15 / "manifest.json", {
        **common, "status": "checkpoint_15_complete",
        "source_test": "PDF page 25 / figure label 024: orientation-orthogonal population SSI",
        "selection_contract": "strong contour images; OSI >= 0.05; axial mismatch >= 67.5 degrees",
        "historical_contract_validation": population_validation,
    })
    save_json(args.out_dir_16 / "manifest.json", {
        **common, "status": "checkpoint_16_complete",
        "source_test": "PDF page 26 / figure label 025: orientation-orthogonal component SSI",
        "selection_contract": "strong contour images; OSI >= 0.05; axial mismatch >= 67.5 degrees",
        "historical_contract_validation": component_validation,
    })
    save_json(args.out_dir_17 / "manifest.json", {
        "created_utc": common["created_utc"],
        "status": "checkpoint_17_complete",
        "source_test": "PDF page 27 / figure label 026: mixed-context comparison",
        "panel_contract": "left and right intentionally use different selection rules; qualitative presentation comparison only",
        "sources": {
            "all_images_summary": file_identity(Path(args.all_images_summary)),
            "contour_matched_summary": file_identity(Path(args.contour_matched_summary)),
        },
    })

    first_pop = population_endpoint[population_endpoint["endpoint"].eq("smallest_drift")]
    first_comp = component_endpoint[component_endpoint["endpoint"].eq("smallest_drift")]
    print(f"Wrote {args.out_dir_15.resolve()}")
    print(first_pop.to_string(index=False))
    print(f"Wrote {args.out_dir_16.resolve()}")
    print(first_comp.to_string(index=False))
    print(f"Wrote {args.out_dir_17.resolve()}")
    print(json.dumps({"population": population_validation, "component": component_validation}, indent=2))


if __name__ == "__main__":
    main()
