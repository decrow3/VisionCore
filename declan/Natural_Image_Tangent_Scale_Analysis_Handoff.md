# Natural Image Tangent Scale Analysis Handoff

## Purpose

Implement a tightly scoped deterministic geometry analysis that uses natural-image histories from the existing TFTS pipeline to test whether the scale at which local retinal-translation tangents break down depends on natural-image structure.

This is intended as a possible ecological anchor for the FEM covariance and TFTS story. It should not become a new decoding, performance, or optimality project.

## Scientific question

The minimal hypothesis is:

> In the canonical twin, local retinal-translation tangents remain predictive over larger retinal displacements for smooth or coarse image regions, but break down at smaller displacements for fine, high-gradient, or high-spatial-frequency image regions.

This analysis asks whether the response geometry is set by image structure, not merely by the eye-movement range on which the twin was trained.

## Why this matters

The current TFTS result shows that small retinal translations generate image/history-specific tangents that occupy a compact, image-generalizing subspace. This explains how FEM covariance can remain structured across content without requiring a universal signed x/y displacement axis.

The new tangent-scale analysis asks a complementary question:

> How far can the local tangent description be trusted before the finite-displacement response leaves the locally linear regime?

If the breakdown scale depends on natural-image structure, then the tangent geometry is tied to image/RF curvature rather than being merely a consequence of the model being trained on FEM-scale jitter.

## Critical guardrail

The twin was trained on retinal inputs jittered by real eye movements. Therefore, an absolute match between the model's tangent breakdown scale and the empirical FEM amplitude distribution could be circular.

The non-circular test is the image-scale gate:

> Breakdown scale must depend systematically on image structure.

Only if this gate passes should the analysis compare breakdown scales to empirical FEM amplitudes.

If breakdown scale is flat across image structure and simply sits near the training displacement range, do not make an ecological claim. Report the scale gate as failed and stop.

## What not to do

Do not add:

- classifiers
- percent-correct metrics
- decoding accuracy
- new observer models
- E-optotypes
- RF or STA reconstruction
- layerwise decomposition
- random-feature nulls
- recorded-basis extensions
- claims of optimality

This is a deterministic model-geometry assay only.

## Existing inputs

Use the existing TFTS production output as the starting point:

```bash
outputs/twin_feature_tangent_structure_prod_limited_synth
```

Important existing files:

```bash
twin_feature_tangent_summary.json
tangent_maps/twin_tangent_maps.pkl
tangent_maps/twin_tangent_object_metrics.csv
sampled_object_stats.csv
input_shape_audit.csv
prediction_path_validation.csv
delta_sensitivity_summary.csv
canonical_unit_manifest.csv
```

The existing TFTS run used:

```text
subject/date: Allen 2022-02-16
canonical cells: 756
history length: 32 frames
object type: full history object
valid objects for union/basis: 63
primary tangent deltas: 0.125, 0.25, 0.5 arcmin
primary TFTS delta: 0.25 arcmin
```

Use the same canonical model-loading and prediction path as:

```bash
declan/twin_feature_tangent_structure/run_twin_feature_tangent_structure.py
```

## New module

Create a new module rather than overloading the TFTS runner:

```bash
declan/natural_image_tangent_scale/
```

Suggested runner:

```bash
declan/natural_image_tangent_scale/run_natural_image_tangent_scale.py
```

Suggested output root:

```bash
outputs/natural_image_tangent_scale/
```

The runner should support reusing saved TFTS histories/tangents:

```bash
python -m declan.natural_image_tangent_scale.run_natural_image_tangent_scale \
  --tfts-root outputs/twin_feature_tangent_structure_prod_limited_synth \
  --dataset-configs-path experiments/dataset_configs/multi_basic_240_rsvp.yaml \
  --subject Allen \
  --date 2022-02-16 \
  --j-delta-arcmin 0.25 \
  --sensitivity-j-delta-arcmin 0.125,0.5 \
  --displacement-magnitudes-arcmin 0.125,0.25,0.5,1.0,2.0,4.0 \
  --directions cardinal \
  --model-device cuda \
  --use-cached-data \
  --output-root outputs/natural_image_tangent_scale
```

## Analysis object

The analysis unit is a full history object:

```text
object_id = image_id / trial_index / time_index
```

Do not average over image ID before computing tangent prediction metrics.

For each object:

```text
r0 = model response to baseline history
J = [b_x, b_y], local tangent matrix
r_delta = model response to the same history shifted by finite displacement Delta
r_hat_delta = r0 + J Delta
```

Use `J` from the saved TFTS tangent maps when possible.

Primary `J`:

```text
j_delta_arcmin = 0.25
```

Sensitivity:

```text
j_delta_arcmin = 0.125
j_delta_arcmin = 0.5
```

## Displacement grid

Evaluate finite displacements at:

```text
0.125, 0.25, 0.5, 1.0, 2.0, 4.0 arcmin
```

If 4.0 arcmin is not supported by the renderer or is outside a safe model regime, mark it as:

```text
not_run_out_of_range
```

Minimum displacement directions:

```text
+x
-x
+y
-y
```

Optional directions:

```text
+x+y
+x-y
-x+y
-x-y
```

For v1, cardinal directions are enough.

## Finite-displacement prediction metrics

For each object, `j_delta_arcmin`, displacement magnitude, and direction:

```text
dr = r_delta - r0
dr_hat = J Delta
```

Compute:

```text
true_response_norm = ||dr||
predicted_response_norm = ||dr_hat||
cosine_alignment = cos(dr, dr_hat)
variance_explained = 1 - ||dr - dr_hat||^2 / ||dr||^2
relative_error = ||dr - dr_hat|| / ||dr||
magnitude_ratio = ||dr_hat|| / ||dr||
```

Use robust numerical guards for zero or tiny response changes.

## Small-signal guard

Metrics are unstable when the true finite-displacement response is tiny.

Flag rows as low-signal if either:

```text
true_response_norm < absolute_epsilon
```

or

```text
true_response_norm is in the bottom 5 percent of all finite rows
```

Use an absolute epsilon appropriate for model rates. Start with:

```text
absolute_epsilon = 1e-8
```

Required behavior:

- Do not silently drop low-signal rows.
- Write all rows to disk.
- Add `metric_status = ok` or `metric_status = low_signal`.
- Primary summaries should exclude low-signal rows.
- Sensitivity summaries may include them.

## Natural-image structure metrics

For each object, compute local image-structure predictors from the central frame of the full history. If easy, also compute the same metrics on the history mean, but central-frame metrics are required.

Required metrics:

```text
rms_contrast
gradient_rms
gradient_anisotropy
high_frequency_energy
low_frequency_energy
hf_lf_ratio
```

Optional but useful:

```text
autocorrelation_length
edge_density
dominant_orientation
```

Suggested definitions:

### RMS contrast

```text
std(pixel values)
```

### Gradient RMS

Use finite differences:

```text
gx = horizontal image gradient
gy = vertical image gradient
gradient_rms = sqrt(mean(gx^2 + gy^2))
```

### Gradient anisotropy

Use a simple structure-tensor measure:

```text
Jxx = mean(gx^2)
Jyy = mean(gy^2)
Jxy = mean(gx * gy)
lambda1, lambda2 = eigenvalues([[Jxx, Jxy], [Jxy, Jyy]])
gradient_anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)
```

### Fourier energy

Compute 2D FFT power of the central frame.

Use a fixed radial frequency cutoff, for example:

```text
low_frequency_energy: radial frequency <= 0.25 * Nyquist
high_frequency_energy: radial frequency > 0.25 * Nyquist
hf_lf_ratio = high_frequency_energy / (low_frequency_energy + eps)
```

The exact cutoff should be written to config and metadata.

### Autocorrelation length

If implemented, compute the normalized autocorrelation and report the first radius where it falls below `1/e`. If this is not quick, set:

```text
autocorrelation_length = NaN
image_scale_status = autocorr_not_run
```

## Required output files

Create:

```bash
outputs/natural_image_tangent_scale/config.json
outputs/natural_image_tangent_scale/natural_image_scale_metrics.csv
outputs/natural_image_tangent_scale/natural_image_tangent_prediction_metrics.csv
outputs/natural_image_tangent_scale/natural_image_tangent_breakdown_by_object.csv
outputs/natural_image_tangent_scale/natural_image_scale_gate_summary.csv
outputs/natural_image_tangent_scale/fem_amplitude_vs_breakdown_summary.csv
outputs/natural_image_tangent_scale/README.md
outputs/natural_image_tangent_scale/figures/
```

Only create `fem_amplitude_vs_breakdown_summary.csv` and the FEM overlay figure if the scale gate passes.

## natural_image_scale_metrics.csv schema

Required columns:

```text
object_id
image_id
trial_index
time_index
central_frame_index
rms_contrast
gradient_rms
gradient_anisotropy
high_frequency_energy
low_frequency_energy
hf_lf_ratio
autocorrelation_length
image_scale_status
```

## natural_image_tangent_prediction_metrics.csv schema

Required columns:

```text
object_id
image_id
trial_index
time_index
j_delta_arcmin
displacement_magnitude_arcmin
direction_label
dx_arcmin
dy_arcmin
true_response_norm
predicted_response_norm
cosine_alignment
variance_explained
relative_error
magnitude_ratio
metric_status
low_signal_reason
rms_contrast
gradient_rms
gradient_anisotropy
high_frequency_energy
hf_lf_ratio
autocorrelation_length
```

## Breakdown scale definitions

For each object and criterion, define breakdown scale as:

```text
smallest displacement magnitude where criterion is crossed
```

Required criteria:

```text
cosine_alignment < 0.8
cosine_alignment < 0.6
variance_explained < 0.5
relative_error > 0.5
relative_error > 1.0
```

Breakdown status:

```text
ok
not_reached
not_run_low_signal
not_run_insufficient_displacements
```

If the criterion is never crossed:

```text
breakdown_status = not_reached
breakdown_scale_arcmin = NaN
breakdown_scale_label = >max_tested
```

If all relevant rows are low-signal:

```text
breakdown_status = not_run_low_signal
```

## natural_image_tangent_breakdown_by_object.csv schema

Required columns:

```text
object_id
image_id
trial_index
time_index
j_delta_arcmin
criterion
breakdown_scale_arcmin
breakdown_scale_label
breakdown_status
n_valid_displacements
n_low_signal_rows
rms_contrast
gradient_rms
gradient_anisotropy
high_frequency_energy
hf_lf_ratio
autocorrelation_length
```

## Scale gate

The scale gate decides whether it is legitimate to compare breakdown scale to empirical FEM amplitudes.

Primary test:

```text
Does breakdown scale decrease as local image structure increases?
```

Expected directions:

```text
gradient_rms higher -> smaller breakdown_scale
hf_lf_ratio higher -> smaller breakdown_scale
autocorrelation_length longer -> larger breakdown_scale
```

Compute Spearman correlations across objects for each:

```text
criterion
j_delta_arcmin
image_structure_predictor
```

Use bootstrap over objects for confidence intervals.

Primary predictors:

```text
gradient_rms
hf_lf_ratio
```

Optional:

```text
autocorrelation_length
```

Primary criteria:

```text
cosine_alignment < 0.8
variance_explained < 0.5
relative_error > 0.5
```

The gate passes only if:

1. At least one primary image-structure predictor shows the expected sign for at least two criteria.
2. One of those criteria must be cosine-based or variance-explained.
3. One of those criteria must be relative-error or variance-explained.
4. The result is not driven only by low-signal rows.
5. The trend is qualitatively consistent for the primary tangent delta, 0.25 arcmin.

Use labels:

```text
scale_dependence_supported
scale_dependence_mixed
scale_dependence_not_supported
```

## natural_image_scale_gate_summary.csv schema

Required columns:

```text
j_delta_arcmin
criterion
predictor
n_objects
n_ok_objects
spearman_r
spearman_p
bootstrap_ci_low
bootstrap_ci_high
expected_direction
direction_pass
effect_label
gate_status
```

## Binned summaries

Also produce binned summaries for figure generation.

Bin objects into tertiles by:

```text
gradient_rms
hf_lf_ratio
```

For each bin, compute prediction quality versus displacement:

```text
median cosine_alignment
median variance_explained
median relative_error
median magnitude_ratio
bootstrap CI
```

Output:

```bash
outputs/natural_image_tangent_scale/natural_image_scale_binned_prediction_summary.csv
```

Required columns:

```text
j_delta_arcmin
predictor
bin_label
bin_low
bin_high
displacement_magnitude_arcmin
n_rows
n_objects
median_cosine_alignment
cosine_ci_low
cosine_ci_high
median_variance_explained
ve_ci_low
ve_ci_high
median_relative_error
relerr_ci_low
relerr_ci_high
median_magnitude_ratio
magratio_ci_low
magratio_ci_high
```

## FEM amplitude overlay, conditional

Only if the scale gate passes, compute empirical FEM amplitude distributions.

Use eye traces from the same dataset where possible.

Compute amplitude distributions for:

```text
drift-only if drift mask exists
all valid FEM samples
microsaccade-excluded if event mask exists
```

Use displacement windows:

```text
1 frame
2 frames
4 frames
8 frames
16 frames
32 frames
```

If masks are unavailable, report:

```text
drift_only_status = not_run_missing_event_mask
```

Required percentiles:

```text
p05
p25
p50
p75
p95
```

Compare these to breakdown-scale distributions.

Do not report a single RMS as the main result. Use distributions.

## fem_amplitude_vs_breakdown_summary.csv schema

Required columns:

```text
eye_distribution
time_window_frames
n_samples
amplitude_p05
amplitude_p25
amplitude_p50
amplitude_p75
amplitude_p95
j_delta_arcmin
breakdown_criterion
breakdown_p25
breakdown_p50
breakdown_p75
overlap_fraction
interpretation_label
```

Allowed labels:

```text
fem_overlaps_breakdown_transition
fem_below_breakdown_transition
fem_above_breakdown_transition
not_run_scale_gate_failed
not_run_missing_eye_data
```

Required caveat in README if this section runs:

```text
Because the twin was trained on real-eye jitter, the absolute displacement range of reliable model behavior may partly reflect the training distribution. The non-circular evidence is the dependence of breakdown scale on natural-image structure.
```

## Figures

Generate:

```bash
figures/prediction_quality_vs_displacement_by_gradient_rms.png
figures/prediction_quality_vs_displacement_by_hf_lf_ratio.png
figures/breakdown_scale_vs_gradient_rms.png
figures/breakdown_scale_vs_hf_lf_ratio.png
```

Only if gate passes:

```bash
figures/fem_amplitude_overlay.png
```

Figure requirements:

- No claims of optimality.
- Use "transition" or "breakdown" language, not "optimal scale."
- Show multiple metrics or thresholds if possible.
- Clearly mark low-signal exclusions in captions or README.

## README content

The README should include:

1. Analysis purpose.
2. Exact input root and model population.
3. Object count and exclusion count.
4. Displacement grid and directions.
5. Tangent delta(s) used.
6. Small-signal guard definition.
7. Image-structure predictor definitions.
8. Scale gate result.
9. Whether FEM overlay was run.
10. Final recommendation.

Final recommendation should be one of:

```text
include_as_supplemental_ecological_anchor
include_only_as_exploratory_supplement
drop_panel_scale_gate_failed
drop_panel_metric_unstable
```

## Stop rules

Stop after scale gate if:

```text
scale_dependence_not_supported
```

Do not produce FEM amplitude overlay except as a file with status:

```text
not_run_scale_gate_failed
```

Stop and report metric instability if:

```text
more than 50 percent of rows are low-signal
```

or

```text
breakdown scales are not reached for more than 75 percent of objects across all criteria
```

Do not expand scope to rescue weak results.

## Minimal status report after running

Report:

```text
1. Code compile status.
2. Number of objects analyzed.
3. Number and fraction of low-signal rows.
4. Scale gate status.
5. Breakdown-scale correlations with gradient_rms and hf_lf_ratio.
6. Whether FEM overlay was run.
7. Final recommendation.
```

## Interpretation guide

If scale gate passes:

> Natural-image structure predicts the displacement scale over which the local translation tangent remains valid. This supports an ecological interpretation of TFTS: natural FEMs sample a response regime where V1 translation geometry is locally structured but finite-displacement curvature depends on image scale.

If scale gate fails:

> The analysis does not support an image-structure-dependent breakdown scale. Do not compare breakdown scale to FEM amplitudes, because such a comparison could be circular given the model's eye-jitter training distribution.

## Do not overclaim

Do not write:

```text
FEMs are optimal.
FEMs are tuned to the breakdown scale.
This proves ecological matching.
This proves the brain exploits the tangent structure.
```

Acceptable language:

```text
FEM amplitudes overlap the transition regime.
The result is consistent with natural FEMs sampling a locally structured but curved response manifold.
The absolute displacement range may partly reflect the model training distribution.
The image-scale dependence is the non-circular evidence.
```
