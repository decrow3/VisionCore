# Shared Transformation Geometry (STG)

This module implements the early STG pipeline described in the handoff plan:

- Stage 0 support census across Allen/Logan fixRSVP sessions
- Stage 1 signed tangent-map analysis (per-session, per-source; fixed-n capable, with image-similarity controls)
- Stage 1b twin-template tangent confirmation (recorded vs twin template)
- Stage 2 residual RDM geometry (diagnostic-only standalone runner)
- Stage 2+3 residual geometry + image-similarity controls (canonical inferential path)
- Stage 5 cross-session aggregation over controlled tangent and template scalars

Output root:

- `outputs/twin_covariance_structure/shared_transformation_geometry/`

## Stage 0: Support Census

Script:

- `declan/shared_transformation_geometry/run_stg_support_census.py`

Example:

```bash
python declan/shared_transformation_geometry/run_stg_support_census.py \
  --dataset-configs-path experiments/dataset_configs/multi_basic_240_rsvp.yaml \
  --use-cached-data
```

Outputs:

- `stg_support_census.csv`
- `stg_support_census_summary.json`

## Stage 1: Signed Tangent-Map Analysis

Script:

- `declan/shared_transformation_geometry/run_stg_tangent_stage1.py`

Recorded example:

```bash
python declan/shared_transformation_geometry/run_stg_tangent_stage1.py \
  --subject Allen --date 2022-02-16 \
  --dataset-configs-path experiments/dataset_configs/multi_basic_240_rsvp.yaml \
  --source recorded --sample-mode fixed_n --n-samples-threshold 320 \
  --min-samples 320 --recorded-nuisance time_global --recorded-axis-projection both \
  --n-nulls 200 --bootstrap-repeats 2000 --use-cached-data
```

Twin example:

```bash
python declan/shared_transformation_geometry/run_stg_tangent_stage1.py \
  --subject Allen --date 2022-02-16 \
  --dataset-configs-path experiments/dataset_configs/multi_basic_240_rsvp.yaml \
  --source twin --sample-mode fixed_n --n-samples-threshold 320 \
  --min-samples 320 --n-nulls 200 --bootstrap-repeats 2000 \
  --predict-batch-size 64 --model-device cuda --use-cached-data
```

Per-session outputs:

- `stg_tangent_maps.pkl`
- `stg_tangent_map_image_metrics.csv`
- `stg_tangent_map_alignment.csv`
- `stg_tangent_summary.csv`
- `stg_tangent_metadata.json`

Notes:

- Stage 1 summary inference is null-relative and image-bootstrap based (`bootstrap_unit=image`).
- `analysis_representation=raw_samples` is written to Stage 1 CSV outputs.
- Stage 1 alignment now includes pairwise image similarity metrics (`pixel_correlation`, `rms_contrast_difference`, `fourier_amplitude_similarity`).
- Stage 1 summary now reports controlled real-minus-null effects and low-similarity-pair effects; `interpretation_label` is gated on low-similarity controlled effects for `pixel_correlation`.
- If the primary image-similarity control is not numerically evaluable (NaN/insufficient pairs), Stage 1 writes `interpretation_label=control_not_evaluable` (not `not_supported`).
- For recorded fits, `--recorded-nuisance time_global` regresses out `time_bin`, `time_bin^2`, per-timepoint global mean rate, and a shared temporal mode score before tangent fitting.
- Stricter recorded control: `--recorded-axis-projection {global_rate|pc1|both}` projects responses orthogonal to those global population axes before nuisance regression and dx/dy tangent fitting.
- Stage 1 image metrics include per-image `align_bx_global_rate_axis` and `align_bx_global_pc1_axis` to test whether displacement maps align with global population modes.

Shared-mode sweep:

- `declan/shared_transformation_geometry/run_stg_tangent_stage1.py --recorded-shared-mode-projection-k 0,1,2,3,5,10` delegates recorded fits to `declan/shared_transformation_geometry/run_stg_shared_mode_projection_sweep.py`.
- The sweep writes projection-specific outputs under `source_recorded/projection_k*/` and a session-level `stg_shared_mode_projection_sweep.csv`.
- The sweep keeps the strict recorded preprocessing (`--recorded-axis-projection both`, `--recorded-nuisance time_global`) and adds `shared_mode_basis_source=global_response_pca` plus `variance_explained_by_projected_modes`.
- Drift-only option: add `--drift-only` to Stage 1 (or sweep directly) to compute eye velocity, detect high-velocity events, exclude peri-event windows (`--drift-exclusion-pre-ms`, `--drift-exclusion-post-ms`), and rerun tangent fits on retained drift samples.
- Drift-only support reporting is written into Stage 1 and sweep summaries via `n_valid_samples_before_exclusion`, `n_valid_samples_excluded`, `n_valid_samples_after_exclusion`, `n_images_with_samples_before_exclusion`, and `n_images_with_samples_after_exclusion`.
- Direct recorded-vs-twin comparison is written by `declan/shared_transformation_geometry/run_stg_direct_recorded_twin_tangent_match.py`; it reports `not_run_dimension_mismatch` instead of falling back to Gram matching when unit dimensions differ.

## Stage 1b: Twin-Template Tangent Confirmation

Script:

- `declan/shared_transformation_geometry/run_stg_tangent_template_confirmation.py`

Example:

```bash
python declan/shared_transformation_geometry/run_stg_tangent_template_confirmation.py \
  --subject Allen --date 2022-02-16 \
  --n-nulls 200 --bootstrap-repeats 2000
```

Outputs (session root):

- `stg_tangent_template_match.csv`
- `stg_tangent_template_summary.csv`
- `stg_tangent_template_metadata.json`

Notes:

- Uses `source_twin/stg_tangent_maps.pkl` and `source_recorded/stg_tangent_maps.pkl` from the same session.
- Nulls: eye-label shuffle, image-label shuffle, random-map.
- This runner uses `template_feature_type=gram_JtJ` and reports `template_match_semantics=unit_count_invariant_tangent_metric_match`.

## Stage 2: Residual RDM Geometry (Diagnostic)

Script:

- `declan/shared_transformation_geometry/run_stg_residual_rdm_stage2.py`

Example:

```bash
python declan/shared_transformation_geometry/run_stg_residual_rdm_stage2.py \
  --subject Allen --date 2022-02-16 \
  --dataset-configs-path experiments/dataset_configs/multi_basic_240_rsvp.yaml \
  --source recorded --n-samples 320 --min-images 8 --use-cached-data
```

Outputs:

- `stg2_diagnostic_image_smoothness_metrics.csv`
- `stg2_diagnostic_cross_image_residual_rdm.csv`
- `stg2_diagnostic_session_rdm_summary.csv`

Notes:

- This standalone runner is now diagnostic-only (`analysis_role=diagnostic_only`).
- Use Stage 2+3 below for inferential outputs and controls.

## Stage 2+3 Combined Runner

Script:

- `declan/shared_transformation_geometry/run_stg_residual_rdm_stage23.py`

Example:

```bash
python declan/shared_transformation_geometry/run_stg_residual_rdm_stage23.py \
  --subject Allen --date 2022-02-16 \
  --dataset-configs-path experiments/dataset_configs/multi_basic_240_rsvp.yaml \
  --source recorded --min-samples 320 --use-cached-data
```

Additional output:

- `stg_image_similarity_controls.csv`
- `stg_stage23_binning_diagnostics.csv`
- `stg_stage23_metadata.json`

Notes:

- Stage 2+3 summary CIs use image-bootstrap (`bootstrap_unit=image`).
- Control regression uses centered similarity predictor; intercept is reported as `adjusted_mean_metric_at_centered_similarity`.
- `analysis_representation=eye_bin_averages` is written to Stage 2+3 CSV outputs.
- Stage 2+3 binning diagnostics include occupied/retained bins, bin-count distribution, centered eye range, and assumed eye units.

## Stage 5: Cross-Session Aggregation

Script:

- `declan/shared_transformation_geometry/run_stg_aggregate_stage5.py`

Example:

```bash
python declan/shared_transformation_geometry/run_stg_aggregate_stage5.py \
  --dataset-configs-path experiments/dataset_configs/multi_basic_240_rsvp.yaml \
  --image-sim-control pixel_correlation
```

Outputs (global STG output root):

- `stg_stage5_tangent_controlled_aggregation.csv`
- `stg_stage5_template_match_aggregation.csv`
- `stg_stage5_summary.json`

Notes:

- Stage 5 now reads the strict recorded row from `stg_shared_mode_projection_sweep.csv` when available and aggregates only `recorded_shared_mode_projection_k=0` rows.
- Stage 5 summary reports `n_sessions_run`, `n_sessions_evaluable`, and Allen/Logan splits, while controlled inference excludes `control_not_evaluable` rows.

## Notes

- Undirected anisotropy stage (Stage 4) is not yet implemented.
