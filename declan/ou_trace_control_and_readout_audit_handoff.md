# Coding-Agent Handoff: OU Trace Control And Readout Audit

Status: handoff for a coding agent
Date: 2026-06-21

## Purpose

Audit whether the OU control used in the aggregate BackImage FEM feature
analysis is a valid matched trajectory control, and determine which response
readouts are principled enough to use in a Panel-B-style aggregate figure.

Do not rewrite the narrative, priority checklist, manifest, or figure-story docs
as part of this audit. Produce audit outputs first. Interpretation and document
promotion come after review.

## Scientific Question

The aggregate FEM analysis is now split into two distinct claims:

1. Absolute feature readout:

```text
Does adding a motion-derived response summary improve feature decoding beyond
the static mean V1-twin response?
```

2. Motion-family specificity:

```text
Does recorded empirical FEM-like motion have feature-relevant temporal structure
that differs from OU, Brownian, and rotated controls?
```

Order-blind readouts such as `mean` and `delta_mean` answer the first question
more cleanly. Order-sensitive readouts such as `temporal_pca`,
`temporal_delta_pca`, `temporal_dct`, and `temporal_dct_delta` may be essential
for the second question, because they preserve trajectory ordering.

The OU control currently behaves strangely in order-sensitive readouts. That
does not by itself invalidate temporal PCA. It means OU and the decoder/readout
contract need an explicit audit.

## Important Constraint

Work only in new audit output folders and, if needed, new audit scripts. Do not
edit broad synthesis docs while the audit is unresolved.

Allowed:

- new audit script(s)
- new CSV/JSON/PNG/PDF outputs
- a short audit report inside the new output directory
- narrowly scoped bug fixes if the audit finds a concrete implementation bug

Avoid:

- changing Figure 4 claims
- changing priority checklist language
- changing manuscript-facing docs
- rerunning the expensive V1-twin response cache unless the audit proves cached
  traces cannot answer the question

## Key Inputs

Aggregate response cache:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/
```

Important files in that folder:

```text
run_metadata.json
analysis_images.csv
trace_bank_metadata.csv
aggregate_motion_metadata.csv
aggregate_motion_summary.csv
response_summary_arrays.npz
latent_feature_arrays.npz
```

Current all-readout posthoc:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/
    incremental_staticmean_plus_motion_allreadouts_v1/
```

Existing all-readout figures:

```text
.../incremental_staticmean_plus_motion_allreadouts_v1/readout_atlas_figures/
  readout_atlas_gain_over_static_mean.png
  readout_atlas_empirical_minus_controls.png
  readout_atlas_primary_scale_score_table.csv
  temporal_alpha_sensitivity_primary_scales.csv
  temporal_alpha_sensitivity_primary_scales.png
  nested_alpha_primary_scale_diagnostic.csv
  nested_alpha_primary_scale_diagnostic.png
```

Relevant code:

```text
declan/fixation_statistics_by_stimulus/run_backimage_aggregate_fem_information.py
declan/fixation_statistics_by_stimulus/summarize_backimage_aggregate_incremental_motion.py
declan/fixation_statistics_by_stimulus/run_backimage_latent_information_screen.py
```

Trace generation functions to inspect/reuse:

```text
_prepare_windows
_session_dataset_cache
_build_trace_bank
_eligible_trace_bank_indices
_family_raw_trace
_scale_family_raw_trace
_ou_trace
_brownian_trace
_rotated_trace
```

Decoder detail:

`_cross_validated_decode` standardizes X and Z within each outer fold, so this
is not simply raw feature magnitude leakage. It supports `alpha_mode="fixed"`
and `alpha_mode="nested_per_candidate"`.

## Current Clues

From the corrected static-mean all-readout pass:

- `mean` and `delta_mean` are the only readouts with positive absolute
  empirical gain over static mean at natural scale.
- Temporal readouts are negative versus static mean under the fixed-alpha
  `10.0` pass.
- Temporal readouts strongly separate empirical from OU.
- Alpha sensitivity showed temporal absolute gains can flip sign with fixed
  alpha, but low-alpha positives also make the static baseline bad.
- Nested-alpha diagnostic chose:
  - static/mean-style readouts: alpha around `100`
  - temporal readouts: alpha around `1000`
  - temporal readouts still lose to static mean
  - temporal readouts still beat OU strongly

This suggests the final answer may be:

```text
mean/delta_mean: best for absolute "adds beyond static" readout
temporal PCA/DCT: best for order-sensitive empirical-vs-control diagnostic
OU: not trustworthy as a headline control until trajectory metrics are audited
```

But do not lock this until the trace audit is complete.

## Required Output Folder

Create:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_ou_trace_control_audit_n384_power_v1/
```

Write all new audit files there.

## Recommended Script

Implement a cache-first audit script, for example:

```text
declan/fixation_statistics_by_stimulus/audit_backimage_ou_trace_controls.py
```

Suggested CLI:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python \
  -m declan.fixation_statistics_by_stimulus.audit_backimage_ou_trace_controls \
  --run-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1 \
  --readout-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/incremental_staticmean_plus_motion_allreadouts_v1 \
  --out-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_ou_trace_control_audit_n384_power_v1 \
  --primary-scales rel_0p25x,rel_0p5x,rel_1x \
  --n-bootstrap 10000 \
  --seed 0
```

## Exact-Trace Reconstruction

The current aggregate cache stores `aggregate_motion_metadata.csv`, but not the
full generated OU/Brownian trajectories. For time-series metrics such as PSD,
velocity autocorrelation, and endpoint displacement, reconstruct the generated
traces by replaying the original aggregate trace generation without running the
V1 twin.

Use the original run config from `run_metadata.json`:

- input CSV
- filters
- max_images
- seed
- `reuse_trace_sources_across_scales=true`
- motion families
- scales
- trace samples per condition
- source trace filters
- n_timepoints
- max_rms_deg

Replay the same loop structure in
`run_backimage_aggregate_fem_information.py`:

1. Load and filter windows with `_prepare_windows`.
2. Load eye-position arrays with `_session_dataset_cache`.
3. Build the trace bank with `_build_trace_bank`.
4. For each image row:
   - sample reusable source trace indices per `(family, sample_index)`
   - generate reusable raw traces with `_family_raw_trace`
   - scale each reusable raw trace to each requested scale with
     `_scale_family_raw_trace`
5. Record one audit row per non-static generated trace.

Verification is mandatory. Join reconstructed rows to
`aggregate_motion_metadata.csv` on:

```text
image_index, source_row, family, scale_id, sample_index
```

Then verify the reconstructed values match cached metadata:

```text
trace_bank_index
effective_rms_deg
path_length_deg
generated_lag1_autocorr
```

Write:

```text
trace_replay_validation.csv
```

Hard fail if:

- any `trace_bank_index` mismatch exists
- median absolute `effective_rms_deg` mismatch exceeds `1e-8`
- median absolute `path_length_deg` mismatch exceeds `1e-6`
- median absolute `generated_lag1_autocorr` mismatch exceeds `1e-6`

If exact replay fails, write `trace_replay_failure_report.md` and stop before
interpreting time-series metrics.

## Trace Metrics To Compute

For each generated trace row, compute these metrics.

### Scale And Amplitude

```text
requested_rms_deg
effective_rms_deg
effective_to_requested_rms
max_radius_deg
endpoint_displacement_deg
radial_distance_mean_deg
radial_distance_p95_deg
bounding_box_x_deg
bounding_box_y_deg
```

### Path And Speed

```text
path_length_deg
path_to_target_ratio
tortuosity = path_length / max(endpoint_displacement, eps)
step_length_mean_deg
step_length_median_deg
step_length_p95_deg
speed_mean_deg_s
speed_median_deg_s
speed_p95_deg_s
```

### Covariance Geometry

```text
trace_cov_eig1
trace_cov_eig2
trace_cov_anisotropy
trace_cov_axis_deg
source_cov_axis_deg
axis_delta_to_source_deg
axis_delta_to_image_edge_deg, if image_edge_axis_deg is available
```

### Temporal Correlation

For position and velocity/increment traces:

```text
position_autocorr_lag_1_to_20
velocity_autocorr_lag_1_to_20
radial_autocorr_lag_1_to_20
```

Use available lags only if `n_timepoints` is too short. Current n_timepoints is
40, so lags 1 to 20 are reasonable.

### Spectrum

Use `dt = 1 / 120` unless the run config implies otherwise.

For x/y position and x/y velocity:

```text
position_psd_lowfreq_fraction
position_psd_highfreq_fraction
velocity_psd_lowfreq_fraction
velocity_psd_highfreq_fraction
velocity_psd_centroid_hz
velocity_psd_slope_loglog
```

Also save family-average PSD curves:

```text
position_psd_by_family_scale.csv
velocity_psd_by_family_scale.csv
```

### Endpoint And Centering Behavior

```text
start_radius_deg
end_radius_deg
start_end_distance_deg
mean_return_to_center_slope
fraction_samples_inside_25pct_rms_radius
fraction_samples_outside_2x_rms_radius
```

This is useful because an OU position process can be too center-pulling or too
confined even when RMS and lag-1 are matched.

## Grouped Summaries

Write a trace-level table:

```text
trace_control_metrics_by_generated_trace.csv
```

Then write summaries:

```text
trace_control_metric_summary_by_family_scale.csv
trace_control_metric_pairwise_empirical_minus_control.csv
trace_control_metric_session_bootstrap.csv
```

Summaries should include:

```text
family
scale_id
metric
mean
median
iqr
ci95_low
ci95_high
n_traces
n_images
n_sessions
```

Pairwise contrasts should include:

```text
empirical - ou
empirical - brownian
empirical - rotated
ou - brownian
```

Use session bootstrap when possible. If a metric is per generated trace rather
than per image, aggregate to image first, then session, before bootstrapping.

## Figures To Produce

Minimum figures:

```text
fig_trace_qc_rms_path_lag1.png
fig_trace_qc_speed_tortuosity.png
fig_position_autocorr_by_family.png
fig_velocity_autocorr_by_family.png
fig_position_psd_by_family.png
fig_velocity_psd_by_family.png
fig_covariance_anisotropy_axis_by_family.png
fig_endpoint_centering_by_family.png
```

Each figure should show families:

```text
empirical
ou
brownian
rotated
```

Use the primary scales first:

```text
rel_0p25x
rel_0p5x
rel_1x
```

Include `1.5x` and `2x` as sentinel/supplement if cheap.

## Response-Space Metrics

In the same audit folder, summarize response-summary arrays from:

```text
response_summary_arrays.npz
```

For each readout, family, and scale:

```text
feature_norm_mean
feature_norm_median
feature_norm_iqr
feature_variance_trace
effective_rank
top_pc_variance_fraction
static_relative_norm_mean
condition_number_approx
```

Readouts:

```text
mean
delta_mean
temporal_pca
temporal_delta_pca
temporal_dct
temporal_dct_delta
```

Write:

```text
response_readout_geometry_summary.csv
fig_response_readout_norms_by_family.png
fig_response_readout_effective_rank_by_family.png
```

Purpose:

```text
Determine whether OU is weird in trace space, response space, or only decoder
space.
```

## Decoder/Readout Audit

Use existing tables first:

```text
incremental_staticmean_plus_motion_allreadouts_v1/
readout_atlas_figures/readout_atlas_primary_scale_score_table.csv
readout_atlas_figures/temporal_alpha_sensitivity_primary_scales.csv
readout_atlas_figures/nested_alpha_primary_scale_diagnostic.csv
```

Write a compact table:

```text
readout_decision_matrix.csv
```

Rows should be readout options. Columns:

```text
readout
preserves_trajectory_order
subtracts_static_response
basis_type
absolute_empirical_gain_primary_mean
absolute_empirical_gain_ci_pass_n
empirical_minus_ou_primary_mean
empirical_minus_ou_ci_pass_n
empirical_minus_brownian_primary_mean
empirical_minus_rotated_primary_mean
nested_alpha_absolute_gain_primary_mean
nested_alpha_empirical_minus_ou_primary_mean
interpretation
recommended_role
```

Use these role labels:

```text
primary_absolute_candidate
order_sensitive_specificity_candidate
diagnostic_control
not_recommended
```

## Principled Readout Options To Evaluate

The audit report should explicitly discuss these options.

### `mean`

Order-blind. Uses the mean response over the trajectory. It can detect feature
relevance carried by the distribution of retinal positions visited, but it
obscures the ordering of those positions. It is currently the strongest
absolute aggregate readout under nested alpha.

### `delta_mean`

Order-blind, static-subtracted. This is the cleanest "motion-induced response
change" readout and aligns naturally with the local `I_z` pairing branch. It
may miss temporal ordering, but it is biologically interpretable and robust for
local mechanistic sensitivity.

### `temporal_pca`

Order-sensitive and data-adaptive. Useful for detecting temporal structure in
empirical FEM responses. Because the temporal basis is learned from response
movies, it should be treated as an order-sensitive diagnostic unless it survives
fair static baselines and nested regularization.

### `temporal_delta_pca`

Order-sensitive and static-subtracted. This may be the most natural PCA version
if the goal is dynamic motion-induced response structure, but it is still
data-adaptive.

### `temporal_dct`

Order-sensitive and fixed-basis. More predeclared than temporal PCA. If the
paper needs an order-sensitive readout, DCT may be easier to defend because the
basis is not fit to the response cache.

### `temporal_dct_delta`

Order-sensitive, fixed-basis, static-subtracted. This is arguably the most
principled order-sensitive diagnostic if the purpose is to test temporal
trajectory ordering rather than static image content.

## OU-Specific Questions

Answer these explicitly:

1. Does OU match empirical traces in RMS, covariance anisotropy, lag-1
   autocorrelation, and path length as intended?
2. Does OU differ from empirical traces in velocity autocorrelation or temporal
   spectrum despite matching position lag-1?
3. Is OU too center-pulling or too confined relative to empirical drift?
4. Is OU's response-space norm/effective-rank abnormal for temporal readouts?
5. Does OU remain pathological when the decoder uses nested alpha?
6. If OU is pathological, is Brownian or rotated a cleaner primary control for
   Panel B?
7. Should we add a better control, such as:
   - phase-randomized empirical traces preserving spectrum and RMS
   - velocity-AR surrogate matched to empirical increment autocorrelation
   - shuffled empirical increments with matched endpoint/RMS
   - time-reversed empirical traces
   - circularly shifted empirical traces
   - spectrum-matched Gaussian surrogate

## Decision Criteria

Classify OU as one of:

```text
ou_valid_primary_control
ou_valid_diagnostic_only
ou_invalid_until_regenerated
ou_inconclusive_needs_new_control
```

Suggested criteria:

### OU Valid Primary Control

Use only if:

- exact replay validates generated traces;
- OU matches intended RMS, path, lag-1, and covariance metrics;
- velocity spectrum is not grossly outside empirical range;
- response-space norms/effective ranks are not pathological;
- nested-alpha decoder does not show a pure overfitting artifact.

### OU Valid Diagnostic Only

Use if:

- trace generation is reproducible and mostly matched;
- OU differs in a scientifically interpretable way, such as smoother or more
  center-pulling temporal structure;
- temporal readouts show empirical-minus-OU signal, but OU is not a fair
  headline negative control for "biological motion helps."

### OU Invalid Until Regenerated

Use if:

- exact replay fails;
- OU does not match its advertised metrics;
- OU has severe spectrum/velocity/path artifacts not implied by the control
  definition;
- OU response summaries are dominated by numerical or dimensional artifacts.

## Expected Final Report

Write:

```text
ou_trace_control_audit_report.md
```

It should include:

1. One-paragraph conclusion.
2. Exact replay validation status.
3. Metric summary table for empirical, OU, Brownian, rotated.
4. OU verdict using the labels above.
5. Readout decision matrix summary.
6. Recommended Panel-B candidates:
   - absolute readout candidate
   - order-sensitive specificity candidate
   - controls to show in main panel
   - controls to route to supplement
7. Recommended next run, if needed.

Keep the wording claim-scoped. Example:

```text
Mean/delta-mean support absolute feature-decodability beyond static at natural
scale. Temporal PCA/DCT preserve trajectory ordering and reveal strong
empirical-vs-OU specificity, but they should not be used as the absolute
gain-over-static headline unless that survives the fair decoder/readout gates.
```

## Completion Criteria

This handoff is complete when:

- exact trace replay is validated or a replay failure report is written;
- all required CSVs and figures exist;
- `ou_trace_control_audit_report.md` gives an OU verdict;
- `readout_decision_matrix.csv` recommends roles for all six readouts;
- no broad narrative docs were edited during the audit.

