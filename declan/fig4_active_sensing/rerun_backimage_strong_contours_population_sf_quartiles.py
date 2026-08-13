#!/usr/bin/env python3
"""Checkpoint 9: page-18 strong-contour population SSI with new SF quartiles."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from declan.active_sensing_movie_information import (
    plot_backimage_real_trace_unit_first_and_population_schematics as schematic,
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


DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/"
    "checkpoint_09_strong_contours_population_sf_quartiles"
)
RELATION = "strong_contours_no_osi"
RELATION_LABEL = "strong contour image windows; no OSI gate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--assignments-csv", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    return parser.parse_args()


def build_summary(
    data: dict,
    unit: pd.DataFrame,
    groups: list[str],
    trace: pd.DataFrame,
    trace_bins: pd.DataFrame,
    row_grid: np.ndarray,
    baseline_lookup: dict[int, int],
    *,
    n_bootstrap: int,
    relation: str = RELATION,
    relation_label: str = RELATION_LABEL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections = schematic.unit_image_selection(
        unit, data["image"], relation=relation, sf_groups=groups,
        min_osi=0.05, match_max_deg=22.5, orthogonal_min_deg=67.5,
        image_axis_col="image_edge_axis_deg",
    )
    return schematic.build_curves_for_relation(
        relation=relation, relation_label=relation_label,
        selections=selections, sf_groups=groups,
        trace=trace, trace_bins=trace_bins, row_grid=row_grid,
        baseline_lookup=baseline_lookup, ssi=data["ssi"], expected=data["expected"],
        stabilized_ssi=data["stabilized_ssi"], stabilized_expected=data["stabilized_expected"],
        unit=unit, rng=np.random.default_rng(BOOTSTRAP_SEED), n_bootstrap=int(n_bootstrap),
    )


def validate_historical(
    data: dict,
    trace: pd.DataFrame,
    trace_bins: pd.DataFrame,
    row_grid: np.ndarray,
    baseline_lookup: dict[int, int],
    matrix_dir: Path,
    *,
    n_bootstrap: int,
    relation: str = RELATION,
    relation_label: str = RELATION_LABEL,
) -> dict:
    _, _, unit_summary, population = build_summary(
        data, data["unit"], ["low_sf", "middle_sf", "high_sf"],
        trace, trace_bins, row_grid, baseline_lookup, n_bootstrap=n_bootstrap,
        relation=relation, relation_label=relation_label,
    )
    old_root = matrix_dir / (
        "phase1_phase2_conditioning_v1/schematic_pathlength_summary_v1/unit_first_and_population_v1"
    )
    saved_pop_path = old_root / "spike_weighted_population_summary.csv"
    saved_unit_path = old_root / "unit_first_summary.csv"
    saved_pop = pd.read_csv(saved_pop_path)
    saved_pop = saved_pop[saved_pop["relation"].eq(relation)].copy()
    saved_unit = pd.read_csv(saved_unit_path)
    saved_unit = saved_unit[saved_unit["relation"].eq(relation)].copy()
    keys = ["sf_group", "context", "path_bin"]
    pop_cols = [
        "population_ssi_bits_per_spike", "population_ssi_delta_vs_stabilized",
        "information_numerator_bits", "expected_spikes",
    ]
    unit_cols = [
        "mean_unit_ssi_bits_per_spike", "sem_unit_ssi_bits_per_spike",
        "mean_unit_ssi_delta_vs_stabilized", "sem_unit_ssi_delta_vs_stabilized",
    ]
    pop_compare = population.merge(
        saved_pop[keys + pop_cols], on=keys, suffixes=("_new", "_saved"), validate="one_to_one",
    )
    unit_compare = unit_summary.merge(
        saved_unit[keys + unit_cols], on=keys, suffixes=("_new", "_saved"), validate="one_to_one",
    )
    pop_diff = max(
        float(np.max(np.abs(pop_compare[f"{col}_new"] - pop_compare[f"{col}_saved"])))
        for col in pop_cols
    )
    unit_diff = max(
        float(np.max(np.abs(unit_compare[f"{col}_new"] - unit_compare[f"{col}_saved"])))
        for col in unit_cols
    )
    passed = (
        len(pop_compare) == len(saved_pop) == len(population)
        and len(unit_compare) == len(saved_unit) == len(unit_summary)
        and pop_diff < 1e-8 and unit_diff < 1e-8
    )
    return {
        "saved_population_summary": file_identity(saved_pop_path),
        "saved_unit_first_summary": file_identity(saved_unit_path),
        "n_population_rows_matched": int(len(pop_compare)),
        "n_unit_first_rows_matched": int(len(unit_compare)),
        "max_absolute_population_core_difference": pop_diff,
        "max_absolute_unit_first_difference": unit_diff,
        "passed": bool(passed),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    matrix_dir = Path(args.matrix_dir)
    data = schematic.load_dataset(matrix_dir)
    trace, trace_bins = schematic.add_equal_count_trace_bins(
        data["trace"], n_drift_bins=N_DRIFT_BINS, n_microsaccade_bins=N_MICROSACCADE_BINS,
    )
    row_grid = schematic.build_movie_row_grid(data["movie"])
    baseline_lookup = schematic.baseline_rows_by_image(data["image"], data["baseline_table"])
    validation = validate_historical(
        data, trace, trace_bins, row_grid, baseline_lookup, matrix_dir,
        n_bootstrap=int(args.n_bootstrap),
    )
    if not validation["passed"]:
        raise ValueError(f"Historical page-18 reconstruction failed: {validation}")

    assignments = pd.read_csv(args.assignments_csv)
    unit_new = prepare_quartile_units(data["unit"], assignments)
    selection, unit_curves, unit_summary, population = build_summary(
        data, unit_new, list(GROUPS), trace, trace_bins, row_grid, baseline_lookup,
        n_bootstrap=int(args.n_bootstrap),
    )
    endpoint = endpoint_audit(population, unit_summary)

    schematic.SF_COLORS.update(COLORS)
    schematic.SF_LABELS.update(LABELS)
    schematic.plot_sf_rows_population_panel(
        population, args.out_dir,
        stem="018_strong_contours_valid_fit_sf_quartiles_population_absolute_delta",
        title="Spike-weighted population SSI - strong contour images, valid-fit units",
        subtitle=(
            f"new SF quartiles; {N_DRIFT_BINS} drift-only / "
            f"{N_MICROSACCADE_BINS} microsaccade trace bins"
        ),
        sf_groups=list(GROUPS), dpi=220,
    )
    make_unit_weighting_figure(
        population, unit_summary, args.out_dir,
        title="Population weighting audit: strong-contour images and valid-fit units",
        stem="checkpoint_09_population_vs_equal_unit_audit",
    )

    trace_bins.to_csv(args.out_dir / "trace_path_bin_definitions.csv", index=False)
    selection.to_csv(args.out_dir / "unit_image_selection.csv", index=False)
    unit_curves.to_csv(args.out_dir / "unit_first_curves.csv", index=False)
    unit_summary.to_csv(args.out_dir / "unit_first_summary.csv", index=False)
    population.to_csv(args.out_dir / "spike_weighted_population_summary.csv", index=False)
    endpoint.to_csv(args.out_dir / "population_vs_unit_first_endpoint_audit.csv", index=False)
    (args.out_dir / "historical_contract_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    counts = selection.groupby("sf_group")["unit_index"].nunique().reindex(GROUPS, fill_value=0).to_dict()
    n_images = int(selection["n_selected_images"].iloc[0])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "checkpoint_09_complete",
        "source_test": "PDF page 18: strong-contour images, all-units population SSI",
        "replacement_scope": "all 85 valid-fit units in tie-aware SF quartiles; no OSI gate",
        "image_selection": f"image_contour_strong == True ({n_images} images per unit)",
        "aggregation_contract": "spike-weighted population SSI with equal-unit audit",
        "baseline_contract": "trial-mean stabilized reference restricted to the same strong-contour images",
        "bootstrap": {"n_image_resamples": int(args.n_bootstrap), "seed": BOOTSTRAP_SEED},
        "unit_counts": counts,
        "sources": {
            "assignments": file_identity(Path(args.assignments_csv)),
            "ssi_matrix": file_identity(matrix_dir / "ssi_matrix.npy"),
            "expected_spikes_matrix": file_identity(matrix_dir / "expected_spikes_matrix.npy"),
        },
        "historical_contract_validation": validation,
        "not_run": "No strong-contour component decomposition (page 19) or orientation-conditioned test was regenerated.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    first = endpoint[endpoint["endpoint"].eq("smallest_drift")]
    lines = []
    for row in first.itertuples(index=False):
        lines.append(
            f"- {LABELS[row.sf_quartile]}: population {row.population_delta_vs_stabilized:+.4f} "
            f"[{row.population_delta_ci95_low:+.4f}, {row.population_delta_ci95_high:+.4f}]; "
            f"equal-unit {row.mean_unit_delta_vs_stabilized:+.4f} +/- {row.sem_unit_delta_vs_stabilized:.4f}."
        )
    readme = f"""# Checkpoint 9: strong-contour population SF quartiles

This regenerates page 18 on the {n_images} strong-contour image windows while
retaining all 85 valid-fit units in tie-aware SF quartiles and applying no OSI
gate. The stabilized baseline uses the same selected images.

## Smallest drift-only bin versus stabilization

{chr(10).join(lines)}

The historical low/middle/high core values are exactly reproduced. No page-19
component decomposition was run here.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {args.out_dir.resolve()}")
    print(first.to_string(index=False))
    print(endpoint.to_string(index=False))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
