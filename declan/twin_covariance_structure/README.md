# Twin Covariance Structure Module

This module implements digital-twin analyses of reafferent covariance structure.

## Scope

These scripts are intentionally restricted to covariance geometry:

- reafferent covariance estimation
- eigenspectra and participation ratio
- subspace overlap and principal angles
- translation-tangent alignment
- image specificity
- occupancy controls
- single-unit to population bridge

The module does not implement decoding accuracy, information, bits/spike, or ideal-observer analyses.

## Files

- `run_twin_covariance_structure.py`: CLI runner for A1-A6 outputs
- `run_a2_audit.py`: dedicated A2 control-construction and trace-count audit runner
- `run_a3_audit.py`: dedicated A3 image-specificity trace-count audit runner
- `run_a3_fixrsvp_audit.py`: dedicated fixRSVP image-specificity audit runner using empirical image windows
- `covariance_core.py`: covariance primitives
- `subspace_metrics.py`: overlap/capture/angle utilities
- `eye_controls.py`: occupancy and control transforms
- `plotting.py`: figure builders for A1-A6

## Usage

```bash
python declan/twin_covariance_structure/run_twin_covariance_structure.py
```

Example with explicit controls:

```bash
python declan/twin_covariance_structure/run_twin_covariance_structure.py \
  --logmar -0.30 \
  --conditions real,matched_null,stabilized,fixed_center,scaled_0.5,scaled_2.0 \
  --k-list 1,2,3,5,10 \
  --max-trials 120
```

## Outputs

The runner writes to `outputs/twin_covariance_structure/`:

- `config.json`
- `summary.csv`
- `per_image_metrics.csv`
- `per_condition_metrics.csv`
- `cfem_cache.pkl`
- `figures/A1_signal_alignment.(png|pdf|svg)`
- `figures/A2_rank_mechanism.(png|pdf|svg)`
- `figures/A3_image_specificity.(png|pdf|svg)`
- `figures/A4_translation_tangent_alignment.(png|pdf|svg)`
- `figures/A5_occupancy_vs_dynamics.(png|pdf|svg)`
- `figures/A6_single_unit_to_population_bridge.(png|pdf|svg)`
- `README.md` (run summary)

The A3 audit runner writes to `outputs/twin_covariance_structure/a3_audit/`:

- `a3_tracecount_metrics.csv`
- `a3_splithalf_repeats.csv`
- `a3_overlap_matrix_long.csv`
- `a3_audit_metadata.json`
- `figures/a3_delta_vs_n_traces.png`
- `figures/a3_within_cross_vs_n_traces.png`
- `figures/a3_overlap_heatmaps_selected_counts.png`

The fixRSVP A3 audit runner writes session-specific outputs to `outputs/twin_covariance_structure/a3_fixrsvp_audit/<session_name>/`:

- `a3_image_support.csv`
- `a3_tracecount_metrics.csv`
- `a3_splithalf_repeats.csv`
- `a3_overlap_matrix_long.csv`
- `a3_audit_metadata.json`
- `figures/a3_delta_vs_n_samples.png`
- `figures/a3_within_cross_vs_n_samples.png`
- `figures/a3_overlap_heatmaps_selected_counts.png`
