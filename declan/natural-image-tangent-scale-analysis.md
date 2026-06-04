Please implement a tightly scoped natural-image tangent-scale analysis as a deterministic geometry add-on to TFTS. Do not add decoding, observers, optotypes, layerwise decomposition, or random-feature nulls.

## Scientific goal

We want to test one minimal hypothesis:

> In the canonical twin, the displacement scale over which local retinal-translation tangents remain predictive depends on natural-image structure. Fine/high-gradient image regions should break local linearity at smaller retinal displacements than smooth/coarse regions.

This is intended as a biological/ecological anchor for the TFTS result. TFTS showed that local retinal-translation tangents are compact and generalize across held-out image identities. This analysis asks how far those local tangents remain valid as finite retinal displacements grow, and whether that validity scale is set by image structure.

Important framing:

* This is a deterministic geometry analysis.
* Do not compute classifier accuracy or percent correct.
* Do not use E-optotypes.
* Do not claim optimality.
* Do not claim FEMs are tuned to an optimum.
* The strongest allowed claim, if supported, is: natural FEMs operate near the displacement scale where V1 translation geometry transitions from locally linear to curved, especially for fine image structure.
* The empirical FEM overlay is only legitimate if breakdown scale varies with image structure. If breakdown scale is flat across image scale, drop the FEM overlay because it may reflect the model’s training displacement range rather than intrinsic image/RF geometry.

## Input data

Reuse existing TFTS outputs and machinery where possible.

Primary root:

```bash
outputs/twin_feature_tangent_structure_prod_limited_synth
```

Use the saved TFTS history objects/tangent payload if possible:

```bash
tangent_maps/twin_tangent_maps.pkl
tangent_maps/twin_tangent_object_metrics.csv
sampled_object_stats.csv
input_shape_audit.csv
prediction_path_validation.csv
```

Use the same canonical twin population:

```text
n_canonical_cells = 756
subject/date = Allen 2022-02-16
history_length_frames = 32
full history object is the analysis unit
```

Do not rerun heavy model prediction unless the saved histories are insufficient. If finite-shift responses are not saved, run only the additional deterministic forward passes needed for the selected objects and displacement grid.

## Primary analysis object

Each object is a full history object:

```text
object_id = image_id / trial_index / time_index
```

Do not average over repeated images before computing metrics.

For each object, use:

```text
r0 = model response at baseline retinal position
J = [b_x, b_y] = local tangent matrix estimated from small finite differences
r_delta = true model response to a finite retinal displacement Δ
r_hat_delta = r0 + J Δ
```

Use the existing primary tangent scale if available:

```text
J from delta = 0.25 arcmin
```

Also include sensitivity using:

```text
J from delta = 0.125 arcmin
J from delta = 0.5 arcmin
```

But keep 0.25 arcmin as the primary.

## Displacement grid

Evaluate finite-displacement prediction across multiple displacement magnitudes:

```text
displacement_magnitudes_arcmin = 0.125, 0.25, 0.5, 1.0, 2.0, 4.0
```

If 4.0 arcmin is too expensive or outside stable renderer/model support, make it optional and report not_run_out_of_range.

Use multiple displacement directions:

```text
directions = +x, -x, +y, -y, and optionally diagonals
```

Minimum required directions:

```text
+x, -x, +y, -y
```

For each object, direction, and magnitude, compute the true shifted response and tangent-predicted response.

## Prediction metrics

For each object, displacement direction, and displacement magnitude, compute:

```text
dr = r_delta - r0
dr_hat = J Δ
```

Required metrics:

1. Response norm:

```text
true_response_norm = ||dr||
predicted_response_norm = ||dr_hat||
```

2. Cosine alignment:

```text
cosine_alignment = cos(dr, dr_hat)
```

3. Variance explained:

```text
variance_explained = 1 - ||dr - dr_hat||^2 / ||dr||^2
```

4. Relative error:

```text
relative_error = ||dr - dr_hat|| / ||dr||
```

5. Magnitude ratio:

```text
magnitude_ratio = ||dr_hat|| / ||dr||
```

## Small-signal guards

These metrics are unstable when the true response change is tiny.

For each object/direction/magnitude, mark rows as low-signal if:

```text
true_response_norm < percentile threshold
```

Use both:

```text
absolute threshold: true_response_norm < 1e-8 or numerical epsilon appropriate to rates
relative threshold: bottom 5% of true_response_norm values across all rows
```

Do not drop low-signal rows silently. Write them and set:

```text
metric_status = low_signal
```

Primary summaries should exclude low-signal rows, with a sensitivity summary including them.

## Image structure predictors

For each object, compute natural-image structure from the central frame and, if cheap, from the full history average.

Required predictors:

```text
rms_contrast
gradient_rms
gradient_anisotropy
high_frequency_energy
low_frequency_energy
hf_lf_ratio
```

Preferred additional predictor if easy:

```text
autocorrelation_length
```

Keep this simple. We do not need a perfect spatial-frequency analysis.

Definitions can be pragmatic:

* `gradient_rms`: RMS magnitude of finite-difference image gradients.
* `gradient_anisotropy`: structure-tensor anisotropy.
* `high_frequency_energy`: Fourier power above a fixed fraction of Nyquist, e.g. > 0.25 cycles/pixel.
* `low_frequency_energy`: Fourier power below that boundary.
* `hf_lf_ratio`: high_frequency_energy / low_frequency_energy.
* `autocorrelation_length`: first radius where normalized autocorrelation falls below 1/e, if easy.

Write:

```text
natural_image_scale_metrics.csv
```

Required columns:

```text
object_id
image_id
trial_index
time_index
rms_contrast
gradient_rms
gradient_anisotropy
high_frequency_energy
low_frequency_energy
hf_lf_ratio
autocorrelation_length
image_scale_status
```

## Breakdown scale

For each object, define breakdown scale Δ* under several criteria.

Required criteria:

```text
cosine_alignment < 0.8
cosine_alignment < 0.6
variance_explained < 0.5
relative_error > 0.5
relative_error > 1.0
```

For each object and criterion:

```text
breakdown_scale = smallest displacement magnitude at which criterion is crossed
```

If criterion is never crossed:

```text
breakdown_scale = > max_tested
breakdown_status = not_reached
```

If all valid rows are low-signal:

```text
breakdown_status = not_run_low_signal
```

Write:

```text
natural_image_tangent_breakdown_by_object.csv
```

Required columns:

```text
object_id
image_id
trial_index
time_index
j_delta_arcmin
criterion
breakdown_scale_arcmin
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

## Gate test: image-scale dependence

This is the load-bearing test.

Before any FEM overlay, test whether breakdown scale depends on image structure.

Primary tests:

```text
Spearman correlation between breakdown_scale and gradient_rms
Spearman correlation between breakdown_scale and hf_lf_ratio
Spearman correlation between breakdown_scale and autocorrelation_length, if available
```

Expected direction:

```text
higher gradient_rms or hf_lf_ratio -> smaller breakdown_scale
longer autocorrelation_length -> larger breakdown_scale
```

Also run binned summaries:

```text
bin objects into low / medium / high image structure by gradient_rms or hf_lf_ratio
plot tangent prediction quality vs displacement for each bin
```

Write:

```text
natural_image_scale_gate_summary.csv
```

Required columns:

```text
j_delta_arcmin
criterion
predictor
n_objects
spearman_r
spearman_p
bootstrap_ci_low
bootstrap_ci_high
expected_direction
direction_pass
effect_label
```

Use bootstrap over objects for CIs.

Decision labels:

```text
scale_dependence_supported
scale_dependence_mixed
scale_dependence_not_supported
```

## Strict stop rule

This analysis should have a hard gate.

If breakdown scale does not depend on image structure in the expected direction for at least two robust metrics, stop and report:

```text
status = scale_gate_failed
```

Do not make an FEM-amplitude overlay claim if the scale gate fails.

Minimum robust metrics:

```text
one cosine-based criterion
one error/variance-based criterion
```

Example pass condition:

```text
gradient_rms or hf_lf_ratio significantly predicts smaller breakdown scale
AND
the same trend appears for both cosine_alignment and relative_error or variance_explained
```

If the gate fails, final interpretation is:

> No reliable image-scale dependence was detected; do not interpret breakdown scale relative to FEM amplitudes because it may reflect model training range or numerical factors.

## FEM amplitude overlay, only if gate passes

If the scale gate passes, compare breakdown scales to empirical FEM displacement amplitudes.

Use empirical eye traces from the same dataset where possible.

Compute distributions, not just RMS:

```text
drift-only displacement amplitude over 1 frame
drift-only displacement amplitude over integration windows matching history or TFTS time scale
all valid FEM displacement amplitude
optional microsaccade-excluded distribution
```

Report percentiles:

```text
5, 25, 50, 75, 95 percentiles
```

Overlay these on the breakdown-scale distribution.

Write:

```text
fem_amplitude_vs_breakdown_summary.csv
```

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
breakdown_criterion
breakdown_p25
breakdown_p50
breakdown_p75
overlap_fraction
interpretation_label
```

Allowed interpretation if gate passes:

> Empirical FEM amplitudes overlap the transition regime where local translation tangents begin to fail for high-structure natural image regions.

Do not say:

```text
FEMs are optimal
FEMs are tuned to the breakdown scale
the model proves ecological matching
```

Add explicit caveat:

```text
Because the twin was trained on real-eye jitter, the absolute displacement range of reliable model behavior may partly reflect the training distribution. The non-circular evidence is the dependence of breakdown scale on image structure.
```

## Outputs

Create a new output root:

```bash
outputs/natural_image_tangent_scale/
```

Required files:

```text
config.json
natural_image_scale_metrics.csv
natural_image_tangent_prediction_metrics.csv
natural_image_tangent_breakdown_by_object.csv
natural_image_scale_gate_summary.csv
fem_amplitude_vs_breakdown_summary.csv   # only if gate passes
README.md
figures/
```

Figures:

```text
figures/prediction_quality_vs_displacement_by_image_scale.png
figures/breakdown_scale_vs_gradient_rms.png
figures/breakdown_scale_vs_hf_lf_ratio.png
figures/fem_amplitude_overlay.png        # only if gate passes
```

## Minimal report

After the run, report:

1. Number of objects analyzed.
2. Number of objects excluded by low-signal guard.
3. Prediction quality vs displacement for low/medium/high image-structure bins.
4. Breakdown-scale correlations with image structure.
5. Gate status.
6. If gate passed, FEM amplitude distribution overlay.
7. Final recommendation:

   * include as supplemental ecological anchor
   * include only as exploratory supplement
   * drop because scale gate failed

## Implementation advice

Reuse TFTS code and saved tangents wherever possible.

Do not compute decoders.

Do not use E-optotypes.

Do not add model variants.

Do not run a heavy production job unless absolutely necessary.

Time-box this to a quick deterministic geometry assay.
