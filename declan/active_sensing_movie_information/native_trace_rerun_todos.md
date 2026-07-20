# Native Trace Rerun TODOs

Date started: 2026-07-17

## Current Policy

- Use native `n_timepoints=40` center-cropped BackImage trace snippets.
- Do not use temporally compressed 128-sample source windows for scale-1 SSI.
- Treat `source/rendered diffusion delta = 0` as the scale-1 native-snippet invariant.
- Keep path length, path speed, RMS/BCEA, diffusion constant, covariance anisotropy/axis ratio/orientation, lag-1 autocorrelation, and microsaccade contamination metrics in trace-bank outputs.

## Phase 1: Freeze Trace Policy

Status: done for forward code paths.

- Contour selected-window reconstruction now uses `center_cropped_native_selected_window_trace_n_timepoints`.
- Shared aggregate trace-bank rendering now defaults to `center_crop_native`.
- Legacy full-window compression is explicit only through `--trace-window-policy resample_full_window`.
- New aggregate outputs expose contract columns such as `trace_render_contract`, `source_window_n_samples`, `rendered_trace_n_samples`, `rendered_trace_source_offset`, and `source_to_render_time_compression`.

## Phase 2: Build Filtered Source Manifests

Status: done.

Trace-bank filter source:

```text
outputs/active_sensing_movie_information/
  backimage_trace_bank_aligned_contours_balanced_source_windows_n576_n40_v1/
    filtered_path_length_le350arcmin/trace_bank_metadata_filtered.csv
```

Validation:

```text
unique source rows: 572
max native path_length_arcmin: 256.4887520016042
snippet policy: center_crop_native_n_timepoints
```

Filtered selected-window manifests:

```text
outputs/active_sensing_movie_information/
  backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_v1/
    balanced_source_windows_pathle350arcmin/selected_windows.csv

outputs/active_sensing_movie_information/
  backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_rot90_v1/
    balanced_source_windows_pathle350arcmin/selected_windows.csv
```

Each filtered manifest has 572 data rows.

For sample-by-scale trace-bank runs, keep:

```text
--trace-bank-max-path-length-deg 5.8333333333
```

## Phase 3: Rerun Primary SSI Caches

Status: done for the older aligned-contour original and rot90 analyses.

Dry-run inventories:

```text
outputs/active_sensing_movie_information/
  backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_v1/
    contour_rr100_spatial_ssi_pairs27_native_n40_pathle350_dryrun/

outputs/active_sensing_movie_information/
  backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_rot90_v1/
    contour_rr100_spatial_ssi_pairs27_native_n40_pathle350_rot90_dryrun/
```

Dry-run validation:

```text
identity.schema_version: 3
trace_source_contracts: center_cropped_native_selected_window_trace_n_timepoints
n_selected_trials: 572
movies x conditions: 572 x 27
movie_condition_inventory rows: 15444
max source_trace_path_length_arcmin: 256.4887520016042
```

Production output directories:

```text
outputs/active_sensing_movie_information/
  backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_v1/
    contour_rr100_spatial_ssi_pairs27_native_n40_pathle350_v1/

outputs/active_sensing_movie_information/
  backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_rot90_v1/
    contour_rr100_spatial_ssi_pairs27_native_n40_pathle350_rot90_v1/
```

Production validation:

```text
identity.schema_version: 3
trace_source_contracts: center_cropped_native_selected_window_trace_n_timepoints
n_selected_trials: 572
movies x conditions: 572 x 27
movie_condition_inventory rows: 15444
max source_trace_path_length_arcmin: 256.4887520016042
original peak condition: along2_across1, population_ssi=0.07350180201375714
rot90 peak condition: along2_across1, population_ssi=0.07602547916381822
```

## Phase 4: Regenerate Downstream Figures

Status: done for the corrected aligned-contour original and rot90 rerun products.

Regenerated only from corrected cache dirs:

- population aligned-vs-orthogonal and SF low/high summaries:
  `population_across_sweep_along0_native_n40_pathle350_v1/`,
  `population_across_sweep_along1_native_n40_pathle350_v1/`,
  `population_along_sweep_across0_native_n40_pathle350_v1/`,
  `population_along_sweep_across1_native_n40_pathle350_v1/`
  in both original and rot90 roots.
- orientation-stratified summaries:
  `orientation_stratified_population_native_n40_pathle350_v1/`
  in both original and rot90 roots.
- original vs rot90 / rotation-crossover diagnostics:
  `rotation_crossover_diagnostics_native_n40_pathle350_v1/`
  under the rot90 root.
- unit-first contour-matched scale curves:
  `unit_first_primary_results_native_n40_pathle350_v1/`
  under the rot90 root.
- example-unit and high-SF contribution diagnostics:
  `backimage_contour_axis_rr100_sf_contour_alignment_example_units_dynamic_log_gaussian_marginal_low0p05_high0p5_native_n40_pathle350_v1/`.

Acceptance checks before interpreting regenerated plots:

```text
identity.schema_version = 3
source_meta.source_trace_contract = center_cropped_native_selected_window_trace_n_timepoints
identity.trace_source_contracts includes only center_cropped_native_selected_window_trace_n_timepoints
max source_trace_path_length_arcmin <= 350
```

For `--sweep-mode trace_bank` runs, also require:

```text
trace_bank_metric_summary.csv exists
trace_bank_metadata.csv exists
trace_bank_assignment_manifest.csv exists
```

Old caches may remain as historical/bug-note artifacts, but they should not feed current interpretation.
