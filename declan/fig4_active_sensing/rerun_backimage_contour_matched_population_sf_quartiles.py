#!/usr/bin/env python3
"""Checkpoint 11: page-20 orientation-aligned population SSI with new SF quartiles."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

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
from declan.fig4_active_sensing.rerun_backimage_strong_contours_population_sf_quartiles import (
    build_summary,
    validate_historical,
)


DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/"
    "checkpoint_11_contour_matched_population_sf_quartiles"
)
RELATION = "contour_matched"
RELATION_LABEL = "strong contour image windows; orientation aligned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--assignments-csv", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    matrix_dir = Path(args.matrix_dir)

    data = schematic.load_dataset(matrix_dir)
    trace, trace_bins = schematic.add_equal_count_trace_bins(
        data["trace"],
        n_drift_bins=N_DRIFT_BINS,
        n_microsaccade_bins=N_MICROSACCADE_BINS,
    )
    row_grid = schematic.build_movie_row_grid(data["movie"])
    baseline_lookup = schematic.baseline_rows_by_image(data["image"], data["baseline_table"])

    validation = validate_historical(
        data,
        trace,
        trace_bins,
        row_grid,
        baseline_lookup,
        matrix_dir,
        n_bootstrap=int(args.n_bootstrap),
        relation=RELATION,
        relation_label=RELATION_LABEL,
    )
    if not validation["passed"]:
        raise ValueError(f"Historical page-20 reconstruction failed: {validation}")

    assignments = pd.read_csv(args.assignments_csv)
    unit_new = prepare_quartile_units(data["unit"], assignments)
    selection, unit_curves, unit_summary, population = build_summary(
        data,
        unit_new,
        list(GROUPS),
        trace,
        trace_bins,
        row_grid,
        baseline_lookup,
        n_bootstrap=int(args.n_bootstrap),
        relation=RELATION,
        relation_label=RELATION_LABEL,
    )
    endpoint = endpoint_audit(population, unit_summary)

    schematic.SF_COLORS.update(COLORS)
    schematic.SF_LABELS.update(LABELS)
    schematic.plot_sf_rows_population_panel(
        population,
        args.out_dir,
        stem="020_contour_matched_valid_fit_sf_quartiles_population_absolute_delta",
        title="Spike-weighted population SSI - strong contours, orientation-aligned units",
        subtitle=(
            f"new SF quartiles; OSI >= 0.05 and unit-contour mismatch <= 22.5 deg; "
            f"{N_DRIFT_BINS} drift-only / {N_MICROSACCADE_BINS} microsaccade bins"
        ),
        sf_groups=list(GROUPS),
        dpi=220,
    )
    make_unit_weighting_figure(
        population,
        unit_summary,
        args.out_dir,
        title="Population weighting audit: strong contours and orientation-aligned units",
        stem="checkpoint_11_population_vs_equal_unit_audit",
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

    counts = selection.groupby("sf_group")["unit_index"].nunique().reindex(GROUPS, fill_value=0)
    pair_counts = selection.groupby("sf_group")["n_selected_images"].sum().reindex(GROUPS, fill_value=0)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "checkpoint_11_complete",
        "source_test": "PDF page 20 / figure label 020: strong-contour orientation-aligned population SSI",
        "replacement_scope": "new tie-aware SF quartiles among 85 valid parametric fits; orientation relation retained from the historical analysis",
        "selection_contract": "image_contour_strong; unit OSI >= 0.05; axial unit-contour mismatch <= 22.5 degrees",
        "aggregation_contract": "spike-weighted population SSI with equal-unit audit",
        "baseline_contract": "historical trial-mean stabilized reference restricted to each unit's selected images",
        "downstream_revision_boundary": "later Figure 4 analyses use a cell-matched baseline, SF thresholds, contour coherence >= 0.20, and eight component bins; those changes are not silently mixed into this bridge checkpoint",
        "bootstrap": {"n_image_resamples": int(args.n_bootstrap), "seed": BOOTSTRAP_SEED},
        "unit_counts": {str(k): int(v) for k, v in counts.items()},
        "selected_unit_image_pair_counts": {str(k): int(v) for k, v in pair_counts.items()},
        "sources": {
            "assignments": file_identity(Path(args.assignments_csv)),
            "ssi_matrix": file_identity(matrix_dir / "ssi_matrix.npy"),
            "expected_spikes_matrix": file_identity(matrix_dir / "expected_spikes_matrix.npy"),
        },
        "historical_contract_validation": validation,
        "not_run": "No orientation-aligned component decomposition (page 21) or later revised Figure 4 panel was regenerated.",
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    first = endpoint[endpoint["endpoint"].eq("smallest_drift")]
    lines = []
    for row in first.itertuples(index=False):
        lines.append(
            f"- {LABELS[row.sf_quartile]}: population {row.population_delta_vs_stabilized:+.4f} "
            f"[{row.population_delta_ci95_low:+.4f}, {row.population_delta_ci95_high:+.4f}]; "
            f"equal-unit {row.mean_unit_delta_vs_stabilized:+.4f} +/- "
            f"{row.sem_unit_delta_vs_stabilized:.4f}."
        )
    readme = f"""# Checkpoint 11: orientation-aligned population SF quartiles

This regenerates historical figure label 020 with the new preferred-SF
quartiles. It retains the historical orientation gate and trial-mean stabilized
baseline so the effect of changing the SF assignments remains isolated.

## Smallest drift-only bin versus stabilization

{chr(10).join(lines)}

The historical low/middle/high core values are exactly reproduced. Later
cell-matched Figure 4 revisions remain a separate downstream checkpoint.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote {args.out_dir.resolve()}")
    print(first.to_string(index=False))
    print(endpoint.to_string(index=False))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
