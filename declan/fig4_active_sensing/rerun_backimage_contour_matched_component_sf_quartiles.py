#!/usr/bin/env python3
"""Checkpoint 12: page-21 aligned-unit component SSI with new SF quartiles."""

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
    endpoint_audit,
    make_direct_overlay,
    prepare_component_inputs,
    validate_historical,
)
from declan.fig4_active_sensing.rerun_backimage_all_images_population_sf_quartiles import (
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
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
    "checkpoint_12_contour_matched_component_sf_quartiles"
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
    component_metrics, bins_by_metric, bins_table = prepare_component_inputs(data)
    row_grid = schematic.build_movie_row_grid(data["movie"])
    baseline_lookup = schematic.baseline_rows_by_image(data["image"], data["baseline_table"])

    validation = validate_historical(
        data,
        component_metrics,
        bins_by_metric,
        row_grid,
        baseline_lookup,
        matrix_dir,
        n_bootstrap=int(args.n_bootstrap),
        relation=RELATION,
        relation_label=RELATION_LABEL,
    )
    if not validation["passed"]:
        raise ValueError(f"Historical page-21 reconstruction failed: {validation}")

    assignments = pd.read_csv(args.assignments_csv)
    unit_new = prepare_quartile_units(data["unit"], assignments)
    selection, summary = build_component_summary(
        data,
        unit_new,
        list(GROUPS),
        component_metrics,
        bins_by_metric,
        row_grid,
        baseline_lookup,
        n_bootstrap=int(args.n_bootstrap),
        relation=RELATION,
        relation_label=RELATION_LABEL,
    )
    endpoint = endpoint_audit(summary)

    schematic.SF_COLORS.update(COLORS)
    schematic.SF_LABELS.update(LABELS)
    schematic.plot_component_population_12_panel(
        summary,
        args.out_dir,
        stem="021_contour_matched_valid_fit_sf_quartiles_across_along_components",
        title="Spike-weighted population SSI - orientation-aligned: across and along components",
        sf_groups=list(GROUPS),
        dpi=220,
    )
    make_direct_overlay(
        summary,
        args.out_dir,
        stem="checkpoint_12_across_vs_along_direct_overlay",
        title="Orientation-aligned units: across- versus along-contour population modulation",
    )

    bins_table.to_csv(args.out_dir / "component_path_bin_definitions.csv", index=False)
    selection.to_csv(args.out_dir / "unit_image_selection.csv", index=False)
    summary.to_csv(args.out_dir / "spike_weighted_population_component_summary.csv", index=False)
    endpoint.to_csv(args.out_dir / "component_endpoint_audit.csv", index=False)
    (args.out_dir / "historical_contract_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    counts = selection.groupby("sf_group")["unit_index"].nunique().reindex(GROUPS, fill_value=0)
    image_counts = selection.groupby("sf_group")["n_selected_images"].sum().reindex(GROUPS, fill_value=0)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "checkpoint_12_complete",
        "source_test": "PDF page 21 / figure label 021: orientation-aligned across/along component-path population SSI",
        "replacement_scope": "new tie-aware SF quartiles among 85 valid parametric fits; historical orientation gate retained",
        "selection_contract": "strong contour images; unit OSI >= 0.05; axial unit-contour mismatch <= 22.5 degrees",
        "component_contract": "trace steps projected onto each image contour axis; independently binned across- and along-contour component path lengths",
        "aggregation_contract": "spike-weighted population SSI with trial-mean stabilized baseline",
        "downstream_revision_boundary": "later Figure 4 analyses use cell-matched baselines, fixed SF thresholds, contour coherence >= 0.20, and eight component bins; those changes are not mixed into this checkpoint",
        "bootstrap": {"n_image_resamples": int(args.n_bootstrap), "seed": BOOTSTRAP_SEED},
        "unit_counts": {str(k): int(v) for k, v in counts.items()},
        "selected_unit_image_pair_counts": {str(k): int(v) for k, v in image_counts.items()},
        "sources": {
            "assignments": file_identity(Path(args.assignments_csv)),
            "trace_xy": file_identity(matrix_dir / "trace_xy.npy"),
            "ssi_matrix": file_identity(matrix_dir / "ssi_matrix.npy"),
            "expected_spikes_matrix": file_identity(matrix_dir / "expected_spikes_matrix.npy"),
        },
        "historical_contract_validation": validation,
        "not_run": "No intermediate-orientation population analysis (page 22) or later revised Figure 4 panel was regenerated.",
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    first = endpoint[endpoint["endpoint"].eq("smallest_drift")]
    lines = []
    for group in GROUPS:
        sub = first[first["sf_quartile"].eq(group)].set_index("component")
        lines.append(
            f"- {LABELS[group]}: across "
            f"{sub.loc['across_path_arcmin', 'population_delta_vs_stabilized']:+.4f}; "
            f"along {sub.loc['along_path_arcmin', 'population_delta_vs_stabilized']:+.4f}; "
            f"across-minus-along {sub['across_minus_along_delta'].iloc[0]:+.4f} bits/spike."
        )
    readme = f"""# Checkpoint 12: orientation-aligned component SF quartiles

This regenerates historical figure label 021 with the new preferred-SF
quartiles. It retains the historical orientation gate and trial-mean stabilized
baseline to isolate the changed SF assignments.

## Smallest drift-only component bins

{chr(10).join(lines)}

The historical low/middle/high core values are exactly reproduced. The
across-minus-along values are descriptive contrasts; this checkpoint does not
bootstrap the paired difference directly. Later cell-matched revisions remain
a separate downstream analysis.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote {args.out_dir.resolve()}")
    print(first.to_string(index=False))
    print(endpoint.to_string(index=False))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
