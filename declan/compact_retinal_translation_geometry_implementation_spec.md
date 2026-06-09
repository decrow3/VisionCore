# Compact Retinal-Translation Geometry Required Analyses: Implementation Specification

## Purpose

This file is a coding-agent handoff for the analyses behind the compact reafferent-geometry / hidden-coordinate result.

The intended scientific claim is:

> FEM-linked shared variability is not arbitrary motion noise. It is carried by a compact, image-generalizing retinal-translation geometry that predicts recorded FEM covariance, survives meaningful nulls, and may contain a readable signal about retinal displacement.

The figure should work in either manuscript framing:

- Hidden-coordinate framing: this analysis block is the core coordinate-like population-geometry result.
- Active-sensing framing: this analysis block is the mechanistic population-format result between the digital twin mechanism and the active-sensing information figure.

## Main Guardrail

Do not let the figure overclaim.

Supported if analyses succeed:

- Small retinal translations produce image-dependent population response changes.
- Those local translation tangents occupy a compact population subspace.
- The compact basis generalizes across held-out images.
- Full finite-difference and compact-restricted translation sources predict recorded FEM-linked covariance.
- The covariance bridge survives unit-shuffle and RF/readout-preserving nulls.
- If displacement decoding succeeds, the compact subspace carries a readable retinal-displacement signal.

Not supported by the compact-geometry analyses alone:

- The animal behaviorally uses this coordinate system.
- V1 fully separates image content and retinal pose.
- There is a literal universal 2D eye-position map in V1.
- FEMs improve perception or are optimally chosen.
- All V1 shared variability is explained by FEMs.

## Expected Compact-Geometry Panel Logic

Recommended final panel structure:

```text
Compact retinal-translation geometry. FEM-linked shared variability lies in a compact retinal-translation geometry

A. Image-dependent local translation charts
B. Compact tangent spectrum
C. Cross-image tangent generalization
D. Hidden-coordinate metric validation
E. Recorded covariance closure: full finite-difference versus compact k=10
F. Variability budget and optional same-image relative-displacement decoding bridge
```

If space is tight, panels B and C can be visually compressed, and the decoding bridge can move to supplement. The metric-validation panel should be treated as the main hidden-coordinate test: compactness says the response changes are low-dimensional, while the metric test asks whether distances and local geometry in that compact space behave like retinal displacement. If same-image relative decoding is strong, it becomes the clearest recorded-data "readability" bridge to active sensing, but it should not replace the metric validation.

## Inputs

The implementation should reuse existing caches and scripts where possible.

Likely relevant workspace files:

```text
run_finite_difference_closure.py
summarize_finite_difference_results.py
figure4_coding_agent_handoff.md
rf_readout_preserving_null_prescription.md
direct_recorded_derivative_twin_alignment_prescription.md
matched_twin_covariance_closure_rf_null_step025_rfbacked_v2/
direct_recorded_derivative_twin_alignment/
rendered_figure4_v3/
rendered_figure4_v5/
```

Required data objects, regardless of actual cache names:

```text
recorded responses in matched recorded/twin unit order
recorded FEM-linked covariance target, Sigma_FEM
recorded total/shared covariance denominators
recorded residual covariance after FEM conditioning if available
recorded stimulus-condition labels: image/time/window/history
recorded trial/repeat labels
measured eye position or eye offset for each response sample
fitted-twin base responses r0(I)
fitted-twin finite-difference translation Jacobians J(I), shape [n_units, 2]
finite-difference translated responses for +/- dx and +/- dy if available
matched unit mask and unit order metadata
RF/readout center metadata if available
unit response scale / mean rate
model quality metric such as CCnorm or CCmax if available
```

Fail loudly if matched unit order cannot be verified across recorded responses, covariance targets, finite-difference sources, compact bases, and unit metadata.

## Shared Conventions

Use one manifest field for every analysis:

```text
finite_difference_step_arcmin
response_window_ms
eye_history_window_ms
eye_neural_latency_ms
eye_coordinate_units
eye_coordinate_sign_convention
sessions_included
matched_units_per_session
context_definition
projection_controls
random_seed
number_of_null_draws
code_version_or_git_commit
input_cache_paths
```

Coordinate convention must be explicit:

- Is eye x/y in degrees, arcmin, pixels, or screen units?
- Is positive y screen-up or image-row-down?
- Does a positive retinal image shift correspond to positive eye displacement or the opposite?
- Are eye offsets centered on fixation center, trial mean, or condition mean?

For subspace/covariance metrics, exact signed convention is less critical. For signed displacement decoding and opposite-shift tests, it is critical.

## Projection Controls

Run key analyses under the same projection controls used in the covariance-closure work:

```text
none
global-rate removed
target PC1 removed
global-rate + target PC1 removed
```

Primary conservative condition:

```text
global-rate + target PC1 removed
```

Projection must be applied consistently:

```text
projected tangent vectors:       b_projected = P @ b
projected derivative matrices:   B_projected = P @ B
projected covariance sources:    Sigma_src_projected = P @ Sigma_src @ P.T
projected covariance targets:    Sigma_tgt_projected = P @ Sigma_tgt @ P.T
projected bases:                 U_projected = orth(P @ U)
```

Record how `P` is constructed. For target PC1 removal, compute the PC from the corresponding target covariance on the training/analysis subset only, not from test data if cross-fitting requires separation.

## Nulls Used Across Analyses

Use a hierarchy of nulls. Each result should show at least the nulls relevant to its claim.

### Null 1: Random subspace

Draw random orthonormal k-dimensional bases in matched unit space.

Purpose:

- sanity check for compact basis capture/readout;
- not sufficient as the only null.

### Null 2: Unconstrained unit shuffle

Permute unit rows of tangents, bases, finite-difference predicted increments, or source covariances.

Purpose:

- continuity with earlier analyses;
- simple reviewer-readable null;
- may be too easy.

### Null 3: RF/readout-preserving fixed row permutation

Permute units only within adaptive bins matched for available metadata:

```text
session
RF/readout x/y location
tangent norm or derivative norm
mean response scale / variance
model quality if available
```

Use fixed within-bin permutations for recorded covariance closure and decoding source-to-target identity tests.

Purpose:

- preserve broad retinotopic/readout layout while breaking exact unit-level pairing between fitted-twin translation geometry and recorded V1 covariance or responses.

### Null 4: RF/readout-preserving samplewise permutation

Permute units within adaptive metadata bins independently per image/history sample.

Use for compact tangent spectrum and cross-sample compactness tests. A fixed row permutation does not change the eigenspectrum of `B B.T`, so it is invalid as a compactness-spectrum null.

### Null 5: Within-condition eye-label shuffle

Shuffle eye positions or eye-position differences within stimulus condition.

Use for same-image relative displacement decoding and direct recorded derivative validation.

Purpose:

- test whether decoding is tied to true eye-response coupling rather than image condition, global variance, or pair reuse.

### Null 6: Context/image shuffle

Train with mismatched image/context labels or test the compact basis on unrelated contexts.

Use as a diagnostic for image-specific versus image-generalizing structure.

## Adaptive RF/Readout Binning

Use this binning for RF/readout-preserving nulls.

Default:

1. Split RF/readout x into quantile bins.
2. Split RF/readout y into quantile bins.
3. Within each spatial bin, split by tangent norm if enough units remain.
4. Optionally split by mean rate or model quality if enough units remain.

Recommended constants:

```text
initial_xy_bins = 3 x 3
min_bin_units = 6
minimum_session_units = 50
n_null_draws_main = 200 if compute is cheap, otherwise at least 100
```

If spatial metadata are unavailable, fall back in this order:

1. model readout spatial position,
2. fitted RF center,
3. unit metadata from model training cache,
4. tangent-profile summary bins only.

If no spatial metadata are recovered, do not call the null RF/readout-preserving. Call it response/tangent-norm constrained.

Save a bin audit:

```text
unit_id
session_id
bin_id
rf_x
rf_y
tangent_norm
mean_rate
model_quality
```

Warn if more than 30% of units fall into one bin or if any bin is below the minimum after merging.

## Analysis 1: Image-Dependent Local Translation Charts

### Question

Do local retinal translations define image-dependent response directions rather than one universal signed x/y axis?

### Inputs

For each image/history object `I`:

```text
r0(I): fitted-twin base response, shape [n_units]
J(I): finite-difference translation Jacobian, shape [n_units, 2]
b_x(I) = J(I)[:, 0]
b_y(I) = J(I)[:, 1]
```

### Computation

1. Choose a sparse representative subset of image/history objects, e.g. 20-40.
2. Build a 2D display space.
   - Preferred: PCA of base responses `r0(I)` plus tangent endpoints.
   - Alternative: PCA of tangent vectors if base-response geometry is too spread out.
3. Plot each selected `r0(I)` as a small point.
4. Draw short arrows or local glyphs for projected `b_x(I)` and `b_y(I)`.
5. Normalize arrow scale only for display; do not use display-normalized tangents for quantitative analyses.

### Required Outputs

```text
local_translation_charts.csv
local_translation_selected_contexts.csv
local_translation_projection_basis.npy
figures/local_translation_charts.png
figures/local_translation_charts.pdf
```

### Acceptance Criteria

- The panel visually shows multiple local translation charts attached to different image/history states.
- It does not imply the x/y axes are globally fixed.
- Nulls are not plotted here; null comparisons belong in later panels.

### Caption Language

> Local finite-difference translations in the fitted twin defined response-change directions attached to each image/history state. The projected tangent glyphs vary across states, showing that the relevant object is not a single universal horizontal or vertical response axis.

## Analysis 2: Compact Tangent Spectrum

### Question

Do the pooled local translation tangents occupy a compact population subspace?

### Inputs

For each session or pooled matched unit set:

```text
B = [b_x(I_1), b_y(I_1), ..., b_x(I_N), b_y(I_N)]
shape: [n_units, 2 * n_contexts]
```

Use finite-difference step:

```text
primary step = 0.25 arcmin unless the existing cache specifies otherwise
```

### Computation

1. Optionally apply projection control `P`.
2. Form tangent covariance:

```text
C_tan = B @ B.T / n_tangents
```

3. Eigendecompose:

```text
lambda_1 >= lambda_2 >= ...
cumulative_variance(k) = sum_{i <= k} lambda_i / sum_i lambda_i
participation_ratio = (sum_i lambda_i)^2 / sum_i lambda_i^2
```

4. Compute null curves:
   - unit-shuffle samplewise null;
   - RF/readout-preserving samplewise null if metadata are available;
   - random subspace reference only as a sanity check.

5. Repeat for displacement step sweep if available:

```text
0.125, 0.25, 0.5, 1.0 arcmin
```

### Required Outputs

```text
tangent_spectrum.csv
tangent_participation_ratio_summary.csv
tangent_null_spectra.csv
figures/compact_tangent_spectrum.png
figures/compact_tangent_spectrum.pdf
```

CSV columns:

```text
session_id
projection_control
finite_difference_step_arcmin
null_type
null_draw
rank
eigenvalue
cumulative_variance
participation_ratio
n_units
n_contexts
```

### Primary Result To Check

Existing draft target:

```text
participation ratio around 9 at 0.25 arcmin
unit-shuffle null around 31
```

Do not hardcode these numbers. Load from analysis outputs.

### Acceptance Criteria

- Observed cumulative spectrum rises faster than valid samplewise null spectra.
- Participation ratio is consistently below unit-shuffle and RF/readout-preserving samplewise nulls.
- Results are not only present before projection controls.

### Caption Language

> Pooling horizontal and vertical finite-difference tangents across image/history states produced a rapidly decaying tangent spectrum. Samplewise unit-shuffle and RF/readout-preserving nulls disrupted the stable unit-specific tangent structure and yielded less compact spectra.

## Analysis 3: Cross-Image Tangent Generalization

### Question

Does a compact basis learned from one set of images capture local translation tangents from held-out images?

### Inputs

Same tangent stack `B`, plus image/context labels:

```text
image_id
stimulus_time_index
history_id if available
trial/repeat ID if needed
```

### Splits

Primary split:

```text
image-disjoint folds
```

Fallback split:

```text
context-disjoint folds
```

If image IDs are unavailable, record the fallback explicitly and do not call the result image-disjoint.

### Computation

For each fold:

1. Train compact basis on training tangents:

```text
U_train,k = top k eigenvectors of B_train @ B_train.T
```

2. Evaluate held-out tangent capture:

```text
capture_test(k) = || U_train,k.T @ B_test ||_F^2 / || B_test ||_F^2
```

3. Use `k_list`:

```text
k = 1, 2, 5, 10, 20, 30
```

4. Compare to:
   - random k-dimensional bases;
   - unit-shuffled train bases;
   - RF/readout-preserving samplewise null bases;
   - optionally context-shuffled bases.

### Required Outputs

```text
cross_image_generalization.csv
cross_image_generalization_summary.csv
figures/cross_image_generalization.png
figures/cross_image_generalization.pdf
```

CSV columns:

```text
session_id
fold_id
projection_control
finite_difference_step_arcmin
k
capture_observed
null_type
null_draw
capture_null
n_train_contexts
n_test_contexts
split_mode
```

### Primary Result To Check

Existing draft target:

```text
k=10 basis captures about 0.50 held-out translation-tangent variance
unit-shuffle null around 0.11
```

Do not hardcode these numbers.

### Acceptance Criteria

- Held-out tangent capture exceeds nulls at k=10.
- k-sweep is plausible and monotonic.
- Image-disjoint split is used if labels allow.
- No test image/context contributes to the basis used to evaluate it.

### Caption Language

> A compact basis learned from finite-difference tangents in one set of images captured held-out translation-tangent variance from other images, indicating that the compact geometry is not a visualization of a single image set.

## Analysis 4: Recorded Covariance Closure

### Question

Does the fitted-twin finite-difference translation source predict recorded FEM-linked covariance in matched recorded/twin units?

### Inputs

For each session:

```text
J_i: fitted-twin finite-difference translation Jacobian, [n_samples, n_units, 2]
e_i: measured eye offset or displacement, [n_samples, 2]
Sigma_target: recorded FEM-linked covariance, [n_units, n_units]
```

Source response increments:

```text
delta_r_i = J_i @ e_i
Sigma_src = cov_i(delta_r_i)
```

Use existing source variants if present:

```text
fd_sample_eye_trace_cov
fd_sample_eye_trace_xfit_compact_k10_cov
```

### Target Variants

Report both:

```text
raw target covariance
PSD-clipped target covariance
```

Use PSD target as headline only if raw target has negative eigenspectrum mass or estimator instability, and show raw in supplement/main audit.

### Metric

For top source eigenspace `U_src,k`:

```text
capture(k) = tr(U_src,k.T @ Sigma_target @ U_src,k) / tr(Sigma_target)
```

Primary k:

```text
k = 2 for finite-difference source closure
```

Also report:

```text
k sweep = 1, 2, 5, 10, 20
```

### Compact-Restricted Source

Build a compact-restricted source:

```text
U_compact_10 = cross-fit compact basis from Analysis 3
delta_r_compact_i = U_compact_10 @ U_compact_10.T @ delta_r_i
Sigma_src_compact = cov_i(delta_r_compact_i)
```

Compare full versus compact:

```text
capture_full
capture_compact_k10
compact_to_full_ratio = capture_compact_k10 / capture_full
```

### Nulls

Required:

- unconstrained unit shuffle;
- RF/readout-preserving fixed row permutation.

Optional diagnostics:

- RF/readout-preserving within-bin random rotation;
- samplewise permutation for source compactness only, not as the main covariance-closure null.

### Required Outputs

```text
covariance_closure_metrics.csv
covariance_closure_bootstrap_summary.csv
covariance_closure_k_sweep.csv
covariance_closure_raw_vs_psd.csv
figures/covariance_closure_full_vs_compact.png
figures/covariance_closure_full_vs_compact.pdf
```

CSV columns:

```text
session_id
target_variant
projection_control
source_variant
k
capture_observed
null_type
null_draw
capture_null
excess_over_null
trace_target
trace_source
n_units
n_samples
```

### Session-Level Inference

For each session:

```text
effect_session = capture_observed - median(capture_null)
```

Across sessions:

```text
mean effect
median effect
bootstrap 95% CI over sessions
sign count
exact sign-test p-value
```

Do not use sample-level bootstrap as the headline inference.

### Primary Result To Check

Existing draft target under PSD target and `global-rate + target PC1`:

```text
full finite-difference source capture around 0.216
full source unit-shuffle excess around +0.172 [0.142, 0.206]
full source RF/readout-null excess around +0.158 [0.125, 0.193]

compact k=10 source capture around 0.217
compact source unit-shuffle excess around +0.174 [0.143, 0.208]
compact source RF/readout-null excess around +0.161 [0.128, 0.196]
```

Do not hardcode these numbers. Recompute or load from trusted analysis outputs.

### Acceptance Criteria

- Full finite-difference source captures target covariance above unit-shuffle.
- Full finite-difference source remains above RF/readout-preserving fixed-permutation null.
- Compact k=10 restricted source is comparable to full source.
- Raw and PSD target variants have consistent sign, or discrepancy is disclosed.
- Results survive the conservative projection control.

### Caption Language

> Finite-difference translation sources from the fitted twin predicted recorded FEM-linked covariance in matched recorded/twin units. Restricting the source to a cross-fit compact k=10 tangent basis retained the closure, and capture remained above both unit-shuffle and RF/readout-preserving nulls.

## Analysis 5: Variability Budget

### Question

How large is the compact translation component relative to full FEM covariance, total reliable shared covariance, and total trial-to-trial variance?

This is necessary because null-adjusted covariance capture values are hard to interpret without denominators.

### Required Denominators

Compute as many as the data support:

```text
non_global_FEM_target_trace
full_FEM_linked_covariance_trace
positive_shared_covariance_trace
total_reliable_shared_covariance_trace
total_trial_to_trial_covariance_trace
split_half_reliability_ceiling_for_FEM_covariance
```

Define each denominator precisely in the manifest and README.

### Recommended Budget Ladder

For each session and projection control:

```text
total trial-to-trial variance
total reliable shared covariance
full FEM-linked covariance
non-global FEM target after projection
full finite-difference captured covariance
compact k=10 captured covariance
compact k=10 excess over RF/readout null
```

### Metrics

For each source:

```text
absolute_capture_trace = tr(U_src,k.T @ Sigma_target @ U_src,k)
fraction_of_non_global_target = absolute_capture_trace / tr(non_global_target)
fraction_of_full_FEM = absolute_capture_trace / tr(full_FEM_covariance)
fraction_of_reliable_shared = absolute_capture_trace / tr(reliable_shared_covariance)
fraction_of_total_trial_variance = absolute_capture_trace / tr(total_trial_covariance)
ceiling_normalized_capture = capture / split_half_reliability_ceiling
null_adjusted_fraction = (capture_observed - median(capture_null)) * tr(target) / denominator
```

If target covariance has negative eigenvalue mass, report:

```text
raw trace
PSD positive trace
negative eigenvalue mass
reason for headline denominator choice
```

### Required Outputs

```text
variability_budget.csv
variability_budget_summary.csv
variability_budget_reliability_ceiling.csv
figures/variability_budget.png
figures/variability_budget.pdf
```

CSV columns:

```text
session_id
projection_control
budget_level
denominator_name
denominator_trace
source_variant
k
absolute_capture_trace
fraction_of_denominator
null_type
null_adjusted_fraction
reliability_ceiling
ceiling_normalized_capture
```

### Acceptance Criteria

- The panel makes clear what `0.216 capture` or `+0.16 excess` means biologically.
- Compact contribution is shown relative to at least full FEM-linked covariance and non-global target.
- Reliability ceiling is shown if split-half covariance estimates are available.
- Negative/PSD target treatment is transparent.

### Caption Language

> A variance-budget view placed the compact finite-difference closure in context. We report compact absolute capture and RF/readout-null-adjusted capture relative to the non-global FEM target, the full FEM-linked covariance, and reliable shared covariance denominators.

## Analysis 6: Coordinate-Like Metric Structure

### Question

Does compact-space displacement behave like a local coordinate for retinal translation?

This is the primary validation analysis for the hidden-coordinate idea.

Compactness alone only shows that translation-induced response changes are low-dimensional. A coordinate-like claim needs a metric test: nearby retinal displacements should have predictable distances, directions, and local composition rules in the compact response space. The central object is the image-local metric induced by the compact translation basis.

If this analysis fails, the manuscript should avoid strong "coordinate system" language even if compactness and covariance closure succeed. The result would instead be a compact covariance-predictive reafferent geometry.

### Inputs

For each image/history object `I`, evaluate the fitted twin at small translations:

```text
r(I, 0)
r(I, +dx)
r(I, -dx)
r(I, +dy)
r(I, -dy)
possibly larger displacements and diagonal combinations
```

Use cross-fit compact basis `U_k`, preferably k=10.

Compact coordinates:

```text
z(I, delta) = U_k.T @ (r(I, delta) - r(I, 0))
```

Compact finite-difference Jacobian:

```text
J_k(I) = U_k.T @ J(I)
shape: [k, 2]
```

Image-local pullback metric:

```text
G_k(I) = J_k(I).T @ J_k(I)
shape: [2, 2]
```

Optional noise-aware version:

```text
G_k_noise(I) = J(I).T @ U_k @ inv_or_pinv(Sigma_resid_k) @ U_k.T @ J(I)
```

Use the unwhitened `G_k` as the primary metric unless the residual covariance estimate is reliable and the noise-aware version is clearly specified as a supplement. Do not mix the two in one headline.

### Tests

#### Test 6.0: Local metric validity and conditioning

For each image/history object, compute:

```text
eigvals(G_k(I))
trace(G_k(I))
det(G_k(I))
condition_number(G_k(I))
anisotropy = (lambda_max - lambda_min) / (lambda_max + lambda_min)
```

Report the fraction of contexts with two nonzero metric dimensions above numerical threshold. A coordinate-like 2D retinal-displacement metric requires rank 2 for a meaningful fraction of contexts. A mostly rank-1 result can still support a local displacement-sensitivity axis, but not a full 2D local coordinate.

Compare metric rank, trace, and condition number to:

- random k-dimensional bases;
- unit-shuffled compact bases;
- RF/readout-preserving compact-basis nulls;
- full-population finite-difference metric.

#### Test 6.0b: Quadratic distance prediction

This is the most important metric test.

Use the local metric estimated from small finite differences to predict compact-space distances for held-out finite displacements:

```text
d_compact_actual^2(I, delta) = || z(I, delta) ||^2
d_metric_pred^2(I, delta) = delta.T @ G_k(I) @ delta
```

For pairs of displacements:

```text
d_compact_actual^2(I, delta_a, delta_b) = || z(I, delta_a) - z(I, delta_b) ||^2
d_metric_pred^2(I, delta_a, delta_b) = (delta_a - delta_b).T @ G_k(I) @ (delta_a - delta_b)
```

Evaluation:

- fit or estimate `G_k(I)` using cardinal small shifts only;
- test on held-out diagonal shifts and larger-but-still-local displacements;
- report correlation, slope, R2, and calibration error between actual and predicted squared distances;
- repeat for natural FEM displacement samples, not only grid displacements, if available.

Success means that a metric inferred from local compact tangents predicts the geometry of nearby translated responses. This is stronger than showing low-dimensionality.

#### Test 6.1: Opposite shifts are opposite

For each axis:

```text
opposition_x = cos(z(+dx), -z(-dx))
opposition_y = cos(z(+dy), -z(-dy))
```

Compare to random, unit-shuffle, and RF/readout-preserving null bases.

#### Test 6.2: Magnitude scales with displacement

For displacement magnitudes `a`:

```text
norm_z_axis(a) = || z(a * axis) ||
```

Fit:

```text
norm_z_axis(a) ~ a
```

Report slope, R2, monotonicity, and saturation/nonlinearity.

Also report whether `norm_z_axis(a)^2` is predicted by the corresponding metric element:

```text
norm_z_x(a)^2 approx a^2 * G_k(I)[0, 0]
norm_z_y(a)^2 approx a^2 * G_k(I)[1, 1]
```

#### Test 6.3: Local composition

For small displacements `a` and `b`:

```text
composition_error = || z(a + b) - z(a) - z(b) || / || z(a + b) ||
```

For diagonal:

```text
z(dx + dy) approx z(dx) + z(dy)
```

#### Test 6.4: Directional geometry

Check whether compact-space displacement angle tracks retinal displacement direction:

```text
cos_between_compact_and_physical_direction
or decoder-based angular error
```

Do not overinterpret if image-specific tangents rotate substantially.

#### Test 6.5: Metric-normalized coordinate recovery

This is a bridge between metric validation and decoding, but it uses the model's local metric rather than a trained black-box decoder.

For each context, recover retinal displacement from compact displacement using the local compact Jacobian:

```text
delta_hat = inv_or_pinv(J_k(I).T @ J_k(I) + lambda I) @ J_k(I).T @ z(I, delta)
```

Evaluate on held-out finite translations:

```text
R2(delta_hat, delta)
angular_error(delta_hat, delta)
magnitude_error(||delta_hat||, ||delta||)
```

This asks whether the compact geometry is locally invertible as a coordinate chart. It is model-side, so it should not be interpreted as a recorded downstream readout. Same-image relative displacement decoding remains the separate recorded-data readability test.

#### Test 6.6: Cross-image metric regularity

Because the draft explicitly allows image-specific local charts, do not require one identical metric tensor across images. Instead report whether the metrics are regular enough to support a coordinate-like interpretation.

Compute:

```text
G_norm(I) = G_k(I) / trace(G_k(I))
```

Then summarize:

- distribution of metric anisotropy;
- distribution of principal-axis orientations;
- split-half reliability of `G_norm(I)` across image/context folds if repeated contexts allow it;
- whether metric statistics are stable across held-out image sets and sessions.

Interpretation:

- stable metric statistics support a reusable population format for local retinal translation;
- highly irregular or rank-deficient metrics argue for compact sensitivity without a strong coordinate-system interpretation.

### Required Outputs

```text
metric_structure_local_metric.csv
metric_structure_quadratic_prediction.csv
metric_structure_opposition.csv
metric_structure_scaling.csv
metric_structure_composition.csv
metric_structure_coordinate_recovery.csv
metric_structure_cross_image_regularities.csv
metric_structure_summary.csv
figures/metric_structure_summary.png
figures/metric_structure_summary.pdf
```

### Acceptance Criteria

- Local compact metrics are rank 2 in a meaningful fraction of reliable contexts.
- Local metric tensors predict held-out compact-space squared distances above random, unit-shuffle, and RF/readout-preserving nulls.
- Opposite shifts are more opposite in the true compact basis than in null bases.
- Compact displacement norm increases with physical displacement at least over the natural FEM range.
- Local composition errors are small for drift-scale displacements and increase for larger displacements, consistent with a local chart.
- Metric-normalized coordinate recovery works for small held-out model translations, at least over the natural FEM displacement range.

### Caption Language

> We treated the compact tangent basis as a candidate local coordinate chart and computed the pullback metric from retinal displacement into compact population space. This local metric predicted held-out compact-space distances for nearby translations, opposite shifts produced approximately opposite compact displacements, displacement norm scaled with physical shift size over the FEM range, and nearby shifts composed approximately linearly. These tests support a coordinate-like interpretation while preserving the image-dependent chart framing.

## Analysis 7: Same-Image Relative Displacement Decoding

### Question

After controlling for image content, can recorded V1 response differences decode the relative retinal displacement between two presentations of the same image/time condition?

This is the preferred recorded-data readability test. It should not be framed as an image-independent eye-position code. It asks whether active retinal sampling leaves a readable sensory trace of how two retinal samples of the same image differed.

The bridge to the active-sensing story is:

> FEMs create different retinal samples of the same screen image. If those samples are useful, the visual system needs some way, explicit or implicit, to register how the samples differ. Same-image relative decoding tests whether recorded V1 activity contains a compact sensory trace of that relative retinal displacement.

Allowed narrow claim if successful:

> Recorded V1 contains a compact, image-conditioned signal about relative retinal displacement.

Avoid:

> V1 contains an image-independent eye-position code.

### Primary Decoding Test

Use same-condition pairwise response differences.

For stimulus/history condition `c` and repeats `a`, `b`:

```text
Delta_y_ab = y(c, a) - y(c, b)
Delta_e_ab = e(c, a) - e(c, b)
```

Because the two repeats share the same image/time condition, the image-locked mean cancels:

```text
Delta_y_ab approx B_c Delta_e_ab + noise difference
```

Train a simple decoder:

```text
Delta_e_hat = W.T @ features(Delta_y)
```

Feature spaces:

```text
full population:             features = Delta_y
compact U_k:                 features = U_k.T @ Delta_y
orthogonal complement:       features = U_perp.T @ Delta_y
random k-dimensional basis:  features = U_rand.T @ Delta_y
RF/readout null basis:       features = U_rf_null.T @ Delta_y
global/top-PC controls:      features from removed global/PC modes
```

Primary compact basis:

```text
U_k = cross-fit compact twin translation basis, k=10
```

### Decoder

Use ridge regression as the primary decoder.

```text
W = argmin_W || Delta_E_train - X_train W ||_F^2 + lambda ||W||_F^2
```

Select lambda by nested cross-validation on training data only. Do not choose lambda based on test performance or basis comparison.

Recommended lambda grid:

```text
lambda_grid = logspace(-4, 4, 17) * trace(X_train.T @ X_train) / n_features
```

### Cross-Validation

Primary split:

```text
train on pairs from some image/time conditions
test on pairs from held-out image/time conditions
```

Leakage prevention:

- No image/time condition appears in both train and test.
- If a repeat/trial appears in a training pair for a condition, it must not appear in a test pair for the same condition.
- Prefer disjoint conditions, which mostly avoids repeat reuse across train/test.
- Pair generation should be balanced so conditions with many repeats do not dominate.

Pair subsampling:

```text
max_pairs_per_condition = choose a fixed value such as 100 or less if needed
pair_sampling_seed recorded
```

### Metrics

Primary continuous metrics:

```text
R2_x
R2_y
R2_mean = mean(R2_x, R2_y)
Pearson_r_x
Pearson_r_y
RMSE_x
RMSE_y
2D_vector_correlation
```

Direction/magnitude diagnostics:

```text
signed direction classification: right/left/up/down or quadrant
angular error for Delta_e direction
magnitude correlation: corr(||Delta_e_hat||, ||Delta_e||)
```

Report signed direction and magnitude separately.

Reliability ceiling:

```text
split-half decoding ceiling
repeat-pair resampling ceiling
or within-condition reproducibility ceiling
```

The ceiling is important because recorded spike data may have low recoverable displacement signal even when the biological effect is meaningful. A modest raw `R2` can still be useful if it is a substantial fraction of the ceiling.

Interpretation:

- Signed Delta x/y decoding across held-out image/time conditions supports a readable relative retinal-displacement trace.
- Magnitude-only decoding supports a general displacement-amount signal but not a signed relative-displacement vector.
- Full-population decoding without compact-subspace enrichment supports a recorded displacement trace, but not the compact-geometry readout claim.
- No above-null decoding means the compact geometry predicts covariance structure but does not provide a simple readable displacement signal in recorded V1.

### Nulls

Required:

1. within-condition eye-difference label shuffle;
2. response-pair shuffle;
3. random k-dimensional bases;
4. unit-shuffled compact basis;
5. RF/readout-preserving compact-basis row permutation;
6. global-rate/top-PC-only controls;
7. orthogonal complement to the compact basis.

Most important comparisons:

- compact `U_k` versus orthogonal complement;
- compact `U_k` versus RF/readout-preserving null subspace;
- compact `U_k` versus within-condition eye-label shuffle.

If compact and orthogonal complement decode equally well, the readable signal is not specific to the compact geometry. If the RF/readout-preserving null decodes equally well, the signal may reflect generic retinotopic/readout organization rather than the fitted compact translation geometry.

### Active-Sensing Bridge Tests

Run these only after the primary relative decoder is implemented and audited. They connect the recorded displacement signal to the model-side active-sampling information result.

#### Bridge Test 7.1: Spectral bridge

Run the same relative decoder separately by image spectral condition or by image bins ranked by high-spatial-frequency content.

Preferred groups if available:

```text
lowpass
intact
highpass
```

Fallback:

```text
tertiles or quartiles of high-spatial-frequency energy
```

Prediction under the active-sensing story:

```text
high-spatial-frequency images should show stronger relative-displacement decoding
or stronger compact-subspace decoding
```

This should be compared to the Figure 5 / active-sensing model result, e.g. real-minus-stabilized information-efficiency gains by spectral condition.

#### Bridge Test 7.2: Model information-gain bridge

For each image or image/history condition, compute:

```text
recorded_relative_decoding_accuracy(condition)
model_real_minus_stabilized_information_gain(condition)
```

Then test:

```text
corr(recorded_relative_decoding_accuracy, model_information_gain)
```

Use cross-validation or split-half aggregation so the same condition-level noise does not inflate both variables. Report Pearson and Spearman correlations, confidence intervals, and permutation controls over condition labels.

Interpretation if positive:

> The conditions in which FEMs are most information-enhancing in the model are also the conditions in which recorded V1 carries the clearest relative-displacement signal.

#### Bridge Test 7.3: Compact bridge

Ask whether compact-subspace decoding predicts model information gain better than:

```text
full-population decoding
orthogonal-complement decoding
global/top-PC decoding
RF/readout-null compact decoding
```

Primary metric:

```text
corr(compact_decoding_accuracy(condition), model_information_gain(condition))
```

Optional regression:

```text
model_information_gain ~ compact_decoding + orthogonal_decoding + high_frequency_energy
```

Use this only as a bridge analysis, not as proof of behavioral use.

### Required Outputs

```text
displacement_decoding_metrics.csv
displacement_decoding_bootstrap_summary.csv
displacement_decoding_nulls.csv
displacement_decoding_pair_inventory.csv
displacement_decoding_reliability_ceiling.csv
displacement_decoding_spectral_bridge.csv
displacement_decoding_information_gain_bridge.csv
figures/displacement_decoding.png
figures/displacement_decoding_spectral_bridge.png
figures/displacement_decoding_information_gain_bridge.png
figures/displacement_decoding.pdf
```

CSV columns:

```text
session_id
fold_id
feature_space
k
projection_control
decoder
lambda_selected
split_mode
n_train_conditions
n_test_conditions
n_train_pairs
n_test_pairs
metric_name
metric_value
null_type
null_draw
metric_null
spectral_group
image_hsf_energy
model_information_gain
reliability_ceiling
fraction_of_ceiling
```

### Session-Level Inference

For each session:

```text
effect_session = metric_observed - median(metric_null)
```

Across sessions:

```text
mean effect
median effect
bootstrap 95% CI over sessions
sign count
exact sign-test p-value
```

### Acceptance Criteria For Main-Figure Promotion

Promote to the main compact-geometry figure only if:

- decoding generalizes to held-out image/time conditions;
- compact k=10 captures a substantial fraction of full-population decoding;
- compact k=10 exceeds random, unit-shuffle, RF/readout-preserving, and eye-label shuffle nulls;
- orthogonal complement is weaker than compact space;
- result survives conservative projection control;
- no leakage audit fails.

Promote as an active-sensing bridge panel only if at least one bridge test succeeds:

- relative decoding is stronger for high-spatial-frequency image content;
- relative decoding strength predicts model real-minus-stabilized information gain;
- compact-subspace decoding predicts model information gain better than orthogonal or null subspaces.

Keep as supplement/diagnostic if:

- full population decodes but compact does not;
- compact equals random subspaces;
- only within-image or non-cross-image decoding works;
- only magnitude decodes but signed Delta x/y does not.
- spectral/information-gain links are weak even though primary relative decoding works.

### Caption Language If Successful

> We trained linear decoders to predict relative eye-position differences from response differences between repeats of the same image/time condition. Because the image/time condition was matched within each pair, the stimulus-locked response largely canceled. Decoding generalized to held-out image conditions and was retained in the compact translation basis, exceeding random, unit-shuffled, RF/readout-preserving, orthogonal-complement, and eye-label shuffle controls. Thus, recorded V1 contains a compact, image-conditioned signal about relative retinal displacement.

### Caption Language If Active-Sensing Bridge Succeeds

> The recorded relative-displacement signal was strongest in image regimes where real FEM retinal movies produced the largest model information-efficiency gains. Thus, the same self-generated retinal motion that explains shared variability and improves model information leaves a compact sensory trace of relative sample displacement in recorded V1.

### Caption Language If Mixed

> Response differences carried some information about relative eye-position differences, but this signal was not selectively concentrated in the compact translation basis, did not generalize robustly across held-out images, or did not track model information gain. We therefore treat the compact geometry primarily as a covariance-predictive reafferent structure rather than a demonstrated compact readout of relative retinal displacement.

## Analysis 8: Direct Recorded Derivative Alignment

This is a supportive analysis, not required for the main figure unless unexpectedly clean.

Detailed prescription already exists:

```text
direct_recorded_derivative_twin_alignment_prescription.md
```

Primary tier:

```text
Estimate recorded eye-position derivatives B_rec,c within condition.
Measure energy capture in cross-fit compact twin basis U_twin,k.
capture_rec,c(k) = || U_twin,k.T @ B_rec,c ||_F^2 / || B_rec,c ||_F^2
```

Use ridge estimation, reliability gates, and split-half derivative reliability. Do not select contexts based on twin alignment.

Recommended role:

- Supplemental support if reliable.
- Diagnostic only if context support is weak.
- Do not let this supersede covariance closure.

## Analysis 9: Robustness Sweeps

Run these if compute permits. At minimum, produce a supplement-ready table.

### Displacement Step Sweep

For tangent compactness, generalization, metric structure, and closure:

```text
0.125, 0.25, 0.5, 1.0 arcmin
```

Expected:

- local compactness strongest at small steps;
- closure stable around natural FEM range;
- nonlinear deviations at larger steps are acceptable and may support "local chart" language.

### k Sweep

For cross-image capture, covariance closure, and decoding:

```text
k = 1, 2, 5, 10, 20, 30
```

Expected:

- k=10 is a continuity point, not a cherry-picked optimum.
- compact k=10 should retain most of full-source closure if current draft numbers hold.

### Projection Sweep

Required:

```text
none
global-rate
target PC1
global-rate + target PC1
```

### Session/Subject Robustness

Report:

```text
per-session effects
per-subject effects if multiple subjects
leave-one-session-out summary
```

### Raw Versus PSD Target

For covariance closure and budget:

```text
raw target
PSD-clipped target
negative eigenvalue mass
trace differences
```

## Output Directory Structure

Create a single consolidated output directory:

```text
outputs/compact_retinal_translation_geometry/
```

Recommended structure:

```text
outputs/compact_retinal_translation_geometry/
  README.md
  manifest.json
  audit.json
  tables/
    local_translation_charts.csv
    tangent_spectrum.csv
    tangent_participation_ratio_summary.csv
    cross_image_generalization.csv
    metric_structure_local_metric.csv
    metric_structure_quadratic_prediction.csv
    metric_structure_summary.csv
    variability_budget.csv
    variability_budget_reliability_ceiling.csv
    covariance_closure_metrics.csv
    covariance_closure_bootstrap_summary.csv
    displacement_decoding_metrics.csv
    displacement_decoding_bootstrap_summary.csv
    displacement_decoding_nulls.csv
    displacement_decoding_pair_inventory.csv
    displacement_decoding_reliability_ceiling.csv
    displacement_decoding_spectral_bridge.csv
    displacement_decoding_information_gain_bridge.csv
    rf_readout_unit_bins.csv
    session_summary.csv
  arrays/
    compact_basis_k10_by_session.npz
    tangent_eigensystems_by_session.npz
    projection_controls_by_session.npz
  figures/
    local_translation_charts.png
    compact_tangent_spectrum.png
    cross_image_generalization.png
    metric_structure_summary.png
    variability_budget.png
    covariance_closure_full_vs_compact.png
    displacement_decoding.png
    displacement_decoding_spectral_bridge.png
    displacement_decoding_information_gain_bridge.png
    compact_geometry_draft_composite.png
```

## Manifest Requirements

`manifest.json` must include:

```text
code_version_or_git_commit
run_datetime
host
input_cache_paths
model_checkpoint_or_config
sessions_included
sessions_excluded_with_reasons
matched_units_per_session
finite_difference_step_arcmin
response_window_ms
eye_window_ms
eye_neural_latency_ms
context_definition
projection_controls
target_covariance_variants
compact_basis_split_mode
decoding_split_mode
ridge_lambda_grid
null_types
n_null_draws
rf_readout_binning_features
min_bin_units
random_seeds
```

## Audit Checks

The run should fail if:

- matched unit order cannot be verified;
- fewer than the predeclared minimum number of sessions are valid;
- any main session has fewer than 50 matched units;
- compact-basis train/test splits leak image IDs when image-disjoint labels are available;
- metric validation uses the same finite translations to estimate and test the local metric without labeling this as in-sample;
- decoding train/test splits share image/time conditions;
- pairwise decoding reuses the same repeat/trial across train and test for the same condition;
- eye labels for decoding are not centered/differenced correctly;
- same-image relative decoding is described as absolute eye-position decoding or as image-independent eye-position coding;
- information-gain bridge correlations use the same data to select conditions and report the final correlation without labeling this as exploratory;
- projection controls are applied to source but not target, or vice versa;
- RF/readout null is labeled RF/readout-preserving without spatial metadata;
- PSD target is used without reporting raw target diagnostics;
- random seeds are missing.

The run should warn if:

- raw and PSD target results differ in sign;
- most recorded decoding power is in global/top-PC controls;
- compact decoding does not exceed random k-dimensional subspaces;
- compact decoding does not exceed the orthogonal complement;
- same-image relative decoding works but spectral or model-information-gain bridge tests are weak;
- local compact metrics are mostly rank 1 or ill-conditioned;
- local metric tensors do not predict held-out compact-space distances above nulls;
- RF/readout-preserving null absorbs most of the covariance-closure effect;
- direct recorded derivative reliability is near eye-shuffle;
- many contexts have rank-1 eye-position variation.

## Statistical Reporting Standard

Use session-level inference for main claims.

For every headline metric:

```text
effect_session = observed_session - median(null_session)
```

Report across sessions:

```text
mean effect
median effect
bootstrap 95% CI over sessions
sign count
exact sign-test p-value
n_sessions
```

Within-session bootstraps over contexts/samples can be reported as diagnostics but should not replace session-level inference.

## Implementation Order

Recommended coding order:

1. Build a loader/audit layer that verifies unit order and exposes common objects.
2. Implement shared projection controls and null generators.
3. Recompute or load finite-difference tangents and compact bases.
4. Run Analysis 2: compact tangent spectrum.
5. Run Analysis 3: cross-image tangent generalization.
6. Run Analysis 6: metric-structure tests.
7. Run Analysis 4: recorded covariance closure, full and compact.
8. Run Analysis 5: variability budget and reliability ceiling.
9. Run Analysis 1: local chart visualization once quantitative bases are stable.
10. Run Analysis 7: same-image relative displacement decoding and active-sensing bridge tests.
11. Optionally run Analysis 8: direct recorded derivative alignment.
12. Generate draft panels and a composite figure.
13. Write README, manifest, and audit summary.

Do not begin with plotting scripts. First make the analysis outputs deterministic and inspectable.

## Suggested README Summary Template

The output README should include:

```text
# Compact Retinal-Translation Geometry Required Analyses

## Run Summary
- Date:
- Code version:
- Sessions:
- Matched units:
- Finite-difference step:
- Projection controls:
- Null draws:

## Headline Results
- Compact tangent PR:
- Cross-image k=10 capture:
- Local metric rank/conditioning:
- Quadratic distance prediction:
- Metric-normalized coordinate recovery:
- Covariance closure full source:
- Covariance closure compact k=10:
- RF/readout-null excess:
- Variability-budget fractions:
- Same-image relative displacement decoding result:
- Spectral bridge result:
- Model information-gain bridge result:

## Warnings / Limitations
- Raw vs PSD target:
- Missing metadata:
- Session exclusions:
- Any failed acceptance criteria:

## Files
- Tables:
- Arrays:
- Figures:
```

## Manuscript Interpretation Decision Tree

### Strongest outcome

Criteria:

- compact tangent spectrum beats samplewise nulls;
- cross-image k=10 capture beats nulls;
- local compact metrics are rank 2 in reliable contexts;
- local metric tensors predict held-out compact-space distances above nulls;
- metric-normalized coordinate recovery works over the FEM displacement range;
- full and compact sources predict recorded FEM covariance;
- RF/readout-preserving null remains below observed;
- variability budget shows nontrivial contribution relative to full FEM/reliable shared covariance;
- same-image relative displacement decoding from compact space generalizes to held-out image/time conditions and beats nulls;
- relative decoding is strongest in high-spatial-frequency regimes or predicts model real-minus-stabilized information gain.

Allowed claim:

> FEMs reveal a compact, metric-validated retinal-translation geometry in foveal V1 whose relative displacement signal is readable in recorded V1 and linked to active-sampling information gain.

### Good outcome

Criteria:

- compactness, cross-image capture, metric validation, covariance closure, and RF/readout null succeed;
- displacement decoding is weak or mixed.

Allowed claim:

> FEM-linked shared variability is carried by a compact coordinate-like retinal-translation geometry that predicts recorded covariance.

Avoid:

> readable coordinate system

or qualify as:

> coordinate-like geometry

### Compact but not metric-positive outcome

Criteria:

- compact tangent spectrum and cross-image capture succeed;
- covariance closure succeeds;
- local metric tensors are rank-deficient, ill-conditioned, or fail to predict held-out compact-space distances.

Allowed claim:

> FEM-linked covariance is carried by a compact retinal-translation subspace that predicts recorded covariance.

Avoid:

> hidden coordinate system

or use only with explicit qualification:

> coordinate-like compact geometry was not supported by metric validation.

### Mixed RF/readout-null outcome

Criteria:

- unit-shuffle effect is strong;
- RF/readout null absorbs much of the closure.

Allowed claim:

> FEM covariance is consistent with compact retinotopically organized reafferent translation geometry.

Avoid:

> feature-specific learned V1 coordinate system beyond retinotopy

### Positive relative-decoding bridge outcome

Criteria:

- same-image relative displacement decoding generalizes to held-out image/time conditions;
- compact k=10 retains most of the full-population decodable signal;
- compact decoding exceeds orthogonal-complement, random, RF/readout-preserving, and eye-label shuffle controls;
- decoding is strongest in high-spatial-frequency regimes or predicts model real-minus-stabilized information gain.

Allowed claim:

> The same FEMs that generate apparent shared variability and improve model information leave a compact, readable trace of relative retinal displacement in recorded V1.

Avoid:

> V1 encodes absolute eye position independently of image content.

### Negative relative-decoding outcome

Criteria:

- covariance closure works;
- same-image relative displacement decoding fails across held-out images or is no better than random/RF null bases.

Allowed claim:

> compact covariance-predictive geometry

Avoid:

> readable relative retinal-displacement signal

## Minimal Main-Figure Deliverable

If time is limited, the minimum required deliverable for a credible compact-geometry figure is:

1. Panel A: local translation charts.
2. Panel B: compact tangent spectrum with valid samplewise null.
3. Panel C: cross-image k-sweep generalization.
4. Panel D: hidden-coordinate metric validation.
5. Panel E: full versus compact covariance closure with unit-shuffle and RF/readout-preserving nulls.
6. Panel F or inset: variability budget relative to full FEM and non-global target.

The same-image relative-displacement decoding analysis can become an added panel, replace the budget inset, or move to supplement depending on strength. Promote it only if it clears the acceptance criteria above. If the spectral or model-information-gain bridge works, it can also serve as a bridge panel into the active-sensing figure.
