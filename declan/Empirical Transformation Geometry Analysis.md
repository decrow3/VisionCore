## Working goal

We are now aiming for the harder, biology-first version of the Figure 4 story:

> **FEM-driven variability in recorded V1 reflects sampling of local, image-specific retinal-translation geometry.**

The model-only geometry is not enough for the paper unless it connects convincingly to recorded V1. The model should be used as a mechanistic scaffold and source of predicted transformation geometry, but the load-bearing claim should be empirical:

> **The same images predicted by the model to have stronger or differently organized FEM-induced transformation sensitivity are the images whose recorded V1 responses show reliable FEM-dependent modulation.**

Short-term implementation still starts with the same pieces: lock Step 2A, run the model translation-grid chart test, and then connect those charts to real FEM-driven neural variability. But the standard for inclusion is higher than a model-geometry figure.

---

## Current state of evidence

### Strong / near locked: Step 01 local generator result

The Step 01 result supports the model-internal claim that natural-image retinal translations have local population generators.

Current interpretation:

- Local midpoint image-translation Jacobians predict model response increments at 0–1 px pairwise separations.
- Prediction quality degrades with displacement, as expected for a local tangent on a curved response manifold.
- The matched Jacobian column space captures FEM-induced model covariance above image-shuffled controls across displacement bins.
- This validates the use of Jacobian-derived local translation geometry in the model.

Use this as a foundation, not as the whole Figure 4.

### Promising: Step 2A scalar empirical bridge

The updated Step 2A result is now a real signal after split-half correction of the empirical target.

Current cross-session diagnostic:

| Session | N images | b_split_med | b_rel_frac | e_snr_frac | r(E,E_cv) | r(tr_model,E) | r(tr_model,E_cv) | r(eye_amp,E_cv) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 02-16 | 20 | 0.196 | 0.60 | 0.76 | 0.546 | 0.215 | 0.695 | −0.258 |
| 02-24 | 14 | 0.511 | 0.68 | 0.68 | 0.808 | 0.182 | 0.783 | −0.260 |
| 03-04 | 32 | 0.134 | 0.63 | 0.87 | 0.663 | 0.383 | 0.531 | +0.093 |
| 04-08 | 34 | 0.336 | 0.77 | 0.87 | 0.606 | 0.394 | 0.286 | −0.268 |

Interpretation:

- `E_FEM` is not pure noise. It is self-consistent under split-half correction.
- The nonlinear model FEM trace predicts the reliable empirical FEM drive in all four sessions in the raw analysis, but the strength is heterogeneous and close to absent in 03-04.
- The residualized scalar bridge is positive in three sessions and negative in 03-04, so the scalar bridge should be treated as supportive but not uniformly clean.
- Eye amplitude does not explain the cross-validated empirical target.
- This is a scalar bridge, not yet an empirical geometry bridge.

### Reopened: empirical transformation geometry

The original negative ceiling read was too strong because it relied on a split-label shuffle that turned out to be an overly conservative diagnostic, not the right biological null. Using eye-position permutation as the primary eye-response decoupling null, empirical split-half geometry is above baseline in all four rerun Allen sessions:

| Session | Emp 2D | Eye-perm 2D | Delta 2D | Emp top-1 | Eye-perm top-1 | Delta top-1 | Positive paired windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| 02-16 | 0.241 | 0.044 | 0.196 | 0.231 | 0.015 | 0.217 | 15 / 16 |
| 02-24 | 0.338 | 0.039 | 0.299 | 0.536 | 0.012 | 0.522 | 12 / 12 |
| 03-04 | 0.203 | 0.030 | 0.172 | 0.086 | 0.008 | 0.079 | 22 / 22 |
| 04-08 | 0.221 | 0.038 | 0.185 | 0.190 | 0.013 | 0.175 | 32 / 32 |

Interpretation:

- Recorded V1 contains recoverable eye-linked response geometry above an eye-response decoupling null.
- The effect is modest and session-dependent, but the 2D empirical geometry survives in all four sessions and is the strongest biology-facing result in the current pipeline.
- Some sessions support top-1 structure more strongly than full 2D geometry, so top-1 and 2D should continue to be reported separately.
- The split-label shuffle remains useful as a diagnostic of partition sensitivity, but it should not be used as the decision null for empirical geometry.

This clears the empirical-geometry gate:

> There is reliable empirical eye-dependent geometry to compare against the model.

What remains unproven is the load-bearing Figure 4 claim:

> Does the model-predicted translation geometry align with this empirical geometry above image-shuffled controls?

### Promising but modest: model translation charts and FEM chart sampling

The hardened chart outputs now support a narrower and cleaner claim than the initial chart language.

| Session | Chart matched | Chart shuffled | Chart delta | FEM capture matched | FEM capture shuffled | FEM capture delta |
|---|---:|---:|---:|---:|---:|---:|
| 02-16 | 0.159 | 0.066 | 0.108 | 0.265 | 0.232 | 0.091 |
| 02-24 | -0.242 | -0.196 | 0.018 | 0.400 | 0.374 | 0.061 |
| 03-04 | 0.428 | 0.251 | 0.212 | 0.312 | 0.223 | 0.031 |
| 04-08 | 0.421 | 0.136 | 0.301 | 0.444 | 0.379 | 0.077 |

Interpretation:

- Matched model translation charts beat image-shuffled charts in all four sessions, so image-specific chart structure is present in the model.
- The absolute coordinate recovery is not uniformly strong across sessions; 02-24 is only barely above shuffled and remains negative in absolute coordinate $R^2$.
- Real FEM sampling is captured modestly but consistently better by matched than shuffled charts across sessions.

This supports chart language of the form:

> Model translation charts contain image-specific coordinate structure above shuffled controls, and real FEM offsets sample those charts modestly but consistently better than shuffled charts.

It does not yet support stronger wording like:

> model translation geometry predicts empirical eye-linked geometry.

### New: direct model-to-empirical geometry alignment

The direct Phase 3 geometry bridge is now complete across the four-session Allen panel. The main result is positive but object-dependent:

| Session | B_model 2D delta | B_model 2D CI | B_model top-1 delta | B_model top-1 CI | FEM_PCs 2D delta | FEM_PCs 2D CI | FEM_PCs top-1 delta | FEM_PCs top-1 CI | J_local 2D delta | J_local 2D CI | J_local top-1 delta | J_local top-1 CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 02-16 | 0.040 | [-0.011, 0.096] | 0.011 | [-0.023, 0.211] | 0.042 | [0.022, 0.066] | 0.041 | [-0.019, 0.233] | 0.015 | [-0.056, 0.088] | 0.010 | [-0.035, 0.079] |
| 02-24 | 0.049 | [0.012, 0.069] | 0.034 | [-0.025, 0.126] | 0.030 | [-0.014, 0.048] | -0.005 | [-0.221, 0.113] | 0.005 | [-0.033, 0.033] | 0.002 | [-0.033, 0.072] |
| 03-04 | 0.035 | [-0.015, 0.054] | 0.003 | [-0.009, 0.011] | 0.026 | [0.000, 0.057] | 0.009 | [-0.009, 0.024] | 0.014 | [-0.030, 0.028] | 0.001 | [-0.021, 0.035] |
| 04-08 | 0.018 | [0.001, 0.032] | 0.022 | [0.001, 0.054] | 0.011 | [0.001, 0.025] | -0.013 | [-0.138, 0.056] | 0.024 | [-0.031, 0.042] | 0.012 | [-0.014, 0.065] |

Interpretation:

- Matched finite-displacement model geometries beat image-shuffled controls in multiple sessions.
- `FEM_PCs` are the most consistent object: positive 2D deltas with nonnegative or clearly positive paired CIs in 02-16, 03-04, and 04-08.
- `B_model` is clearly positive in 02-24 and 04-08, and trends positive in 02-16 and 03-04.
- `J_local` does not carry the empirical bridge; its paired 2D confidence interval crosses zero in all four sessions.
- The bridge is therefore real, but it is carried by finite-displacement response geometry objects rather than by the raw local Jacobian alone.

---

## Strategic framing

### Hard but desired paper claim

The version we want to make work:

> FEM-driven shared variability in recorded V1 reflects the same image-specific translation geometry predicted by the digital twin.

This requires more than model-internal geometry. It requires at least one empirical result showing that model-predicted transformation geometry aligns with reliable eye-dependent structure in recorded V1.

### Current honest claim boundary

The current pipeline now establishes the empirical bridge, but in a qualified form:

1. empirical eye-linked geometry exists above an eye-position permutation null,
2. model image-specific translation charts exist above shuffled controls,
3. real FEM sampling is better captured by matched than shuffled model charts,
4. the scalar bridge between nonlinear model FEM trace and empirical FEM sensitivity is supportive but session-dependent,
5. direct matched model-to-empirical geometry alignment is positive in multiple sessions for finite-displacement model objects.

The remaining limitation is not absence of a geometry bridge, but object-specificity: the bridge is strongest for `FEM_PCs`, positive for `B_model` in a subset of sessions, and not carried by `J_local`.

### Conservative fallback

If the model-to-empirical geometry alignment fails even though the empirical ceiling survives:

> The model captures image-wise variation in empirical FEM sensitivity, and empirical eye-linked geometry exists, but the model-to-biology geometry bridge remains unproven.

This is still weaker than the biology-first Figure 4 endpoint, but it is no longer the same as saying empirical geometry is absent.

### What not to claim yet

Avoid claiming:

- recorded V1 has a fully established fiber-bundle structure,
- downstream circuits read out these coordinates,
- a universal FEM plane exists across images,
- transformations are encoded only in correlations,
- model-only geometry proves empirical biology.

### Updated Figure 4 spine

The current biology-first Figure 4 can now be framed as:

- A. Empirical geometry exists: split-half empirical geometry exceeds an eye-permutation null.
- B. Model translation charts exist: matched model translation charts outperform image-shuffled controls.
- C. FEMs sample model charts: real FEM offsets are better captured by matched than shuffled charts.
- D. Scalar bridge: nonlinear model FEM trace predicts reliable empirical FEM sensitivity in most sessions.
- E. Direct geometry bridge: matched finite-displacement model geometries align with empirical eye-linked geometry above image-shuffled controls in multiple sessions.

Panel E no longer functions as a missing test; it now sets the honest claim boundary. The strongest current wording is that the model-to-biology geometry bridge is present for finite-displacement response-space objects, with `FEM_PCs` the most consistent carrier and `J_local` insufficient on its own.

---

## Analysis roadmap

## Phase 0. Lock Step 2A scalar bridge

### Purpose

Make the scalar empirical bridge robust enough to serve as biological support.

### Existing script

Current script:

```text
scripts/jacobian_predictive_framework/run_fixrsvp_step2.py

or current working version equivalent to uploaded run_fixrsvp_step2(2).py.

Required additions / checks
0.1 Repeated trial-level split-half

Current split-half is one trial-alternating split. Replace or supplement with repeated random trial-level splits.

For each image:

Split trials into A/B at the trial level.
Fit B_emp_A and B_emp_B.
Compute symmetric cross-validated drive:
E_FEM_cv = 0.5 * [tr(B_A Σ_eye B_Bᵀ) + tr(B_B Σ_eye B_Aᵀ)]
Repeat 50–100 times.
Save mean, median, SD/CI for E_FEM_cv.
Save mean/median split-half correlation of flattened B_A and B_B.

CSV additions:

e_fem_cv_mean
e_fem_cv_median
e_fem_cv_std
e_fem_cv_ci_low
e_fem_cv_ci_high
b_split_corr_mean
b_split_corr_median
b_split_corr_ci_low
b_split_corr_ci_high
n_split_repeats_valid
0.2 Reliable-image subset

Compute Step 2A correlations using subsets:

all images
B_split_corr_median > 0
E_FEM > E_FEM_shuffle_median
both criteria
top 75% by n_samples
top 75% by b_split reliability

Important output:

r(trace_cov_model_fem, e_fem_cv_mean)
r(trace_cov_model_fem, e_fem_cv_median)

for each subset and session.

0.3 Residualized / partial correlation controls

The core confound controls:

eye_amplitude_px2
mean_emp_rate
mean_model_rate
stim_rms
stim_grad_energy
n_samples
stimulus_locked_variance (empirical PSTH variance — required, not optional; see decisive test below)
model_baseline_response_norm

**Decisive test: stimulus-locked variance.** The most plausible benign explanation for the scalar bridge is that high-contrast, high-gradient images drive both larger model Jacobians and larger empirical response variance for reasons unrelated to translation geometry. Residualize both `trace_cov_model_fem` and `e_fem_cv` against stimulus-locked variance (and both mean rates). If the partial correlation survives, the bridge is real. If it does not survive this specific control, the bridge is a contrast confound. This is the control a reviewer will raise; treat it as the load-bearing test, not one of several.

For each session, run residualized Spearman or rank-linear residualization:

Rank-transform target and predictors.
Regress e_fem_cv on nuisance variables.
Regress trace_cov_model_fem on nuisance variables.
Correlate residuals.

Save:

rho_trace_vs_Ecv_raw
rho_trace_vs_Ecv_resid_basic
rho_trace_vs_Ecv_resid_full
rho_eye_amp_vs_Ecv
rho_rate_vs_Ecv
rho_grad_vs_Ecv
0.4 Decision gate for Step 2A

Step 2A is considered paper-usable if:

r(trace_cov_model_fem, E_FEM_cv) is positive in 4/4 sessions.
It is clearly positive in at least 3/4 sessions after repeated split-half.
The relationship survives reliable-image filtering.
Residualized correlations remain positive in most sessions.
Eye amplitude alone does not explain the effect.

Wording if passed:

The nonlinear model-predicted FEM trace captures reliable image-to-image variation in empirical FEM sensitivity.

Do not phrase as full covariance-geometry prediction.

---

## Phase 0 (parallel). Empirical split-half geometry feasibility check

### Purpose

Determine — before investing in the model-to-empirical alignment machinery — whether empirical eye-dependent response geometry is recoverable from the data at all. This gate is now resolved positively: `B_emp` has reliable split-half structure above an eye-position permutation null in all four tested Allen sessions. The remaining question is not whether empirical geometry exists, but whether model-predicted geometry aligns with it.

This uses machinery already built for Phase 0.1: the repeated split-half `B_emp_A` / `B_emp_B` fits require no additional data collection.

### Computation

For each image/window, using the repeated split-half `B_emp` fits from Phase 0.1:

- Compute `align(B_emp_A, B_emp_B)`: subspace alignment of the 2D column spaces
- Compute principal angles between `B_emp_A` and `B_emp_B`
- Compute variance capture of `B_emp_B` by `B_emp_A`

Compare to:

- Shuffled-trial control: shuffle trial labels before fitting `B_emp_A` and `B_emp_B`
- Random 2D subspace baseline

### Metrics

Distribution of `align(B_emp_A, B_emp_B)` across images and sessions.
Fraction of images where empirical alignment exceeds shuffle-null 95th percentile.
Session-level median and bootstrap CI over images.

### Decision gate

The biology-first path (Phase 3) is viable if:

- Median empirical split-half alignment exceeds shuffled-trial control in most sessions.
- The alignment distribution is meaningfully above the random-subspace floor.

This gate is now passed for the current four-session Allen panel, so the next decisive analysis is Phase 3 rather than further re-litigation of whether empirical geometry exists.

If a future broader rerun ever places the empirical geometry ceiling at the noise floor across all sessions:

- Phase 3 would no longer be viable with current data.
- The paper would revert to the scalar-bridge version (Step 2A + model-internal Phases 1–2).
- Report this diagnostic honestly. Scalar bridge with model geometry would still be a real result.
- Skip further Phase 3 development and invest only in the model-internal and scalar-bridge endpoints.

---

Phase 1. Model translation-grid chart test
Purpose

Establish that translated versions of the same image form a low-dimensional, image-specific translation chart in the model, and that this chart preserves true retinal displacement coordinates.

This is the key model-geometry test. It should be run before more complex empirical geometry work.

Core question

Does each image define a local translation chart whose coordinates correspond to true retinal Δx, Δy?

Analysis units

Use natural-image fixRSVP stimulus states that match the Step 2A windows as closely as possible.

Preferred hierarchy:

exact lagged stimulus-history window,
image ID plus phase/time-bin if exact history is too sparse,
image identity only as fallback.

Record which level was used.

Translation grid

For each image/window baseline stimulus:

local grid:        dx,dy in [-1, -0.5, 0, +0.5, +1] px
intermediate grid: dx,dy in [-3, -2, -1, 0, +1, +2, +3] px
broad grid:        optionally include ±5, ±8 px

Recommended first pass:

grid values = [-4, -3, -2, -1, -0.5, 0, +0.5, +1, +2, +3, +4]

This gives distance-dependent curves without exploding compute.

For each offset:

r_grid(dx,dy) = model_response(shifted_baseline_stim(dx,dy))
delta_r_grid = r_grid(dx,dy) - r_grid(0,0)
Candidate bases / charts

Compute these bases for each image/window:

1. Matched image Jacobian basis
J_I = [dr/dx, dr/dy]
U_J = orthonormal_basis(J_I)

This is the mechanistic chart.

2. Same-image PCA basis

Top two PCs of delta_r_grid.

Use as descriptive upper bound, not the mechanistic result.

3. Image-shuffled Jacobian bases

Use Jacobian bases from other images matched on:

jacobian_fro_norm
stim_rms
stim_grad_energy
mean_model_rate
eye_amplitude_px2 if using FEM windows

This is the critical null.

4. Random 2D subspaces

Random orthonormal 2D planes in neuron space, optionally energy matched.

5. Global pooled FEM / global translation basis

One pooled basis across images. Tests whether a universal FEM plane suffices.

Primary metric: coordinate recovery

This is the key endpoint.

For each basis U, project responses:

z(dx,dy) = Uᵀ delta_r_grid(dx,dy)

Then quantify how well z recovers true displacement.

Recommended metrics:

Forward coordinate map

Fit:

z = A * [dx,dy] + b

Then evaluate:

R2_z
residual norm
local linearity

But this can favor arbitrary bases.

Inverse coordinate recovery, preferred

Fit:

[dx,dy] = W * z + b

Evaluate cross-validated:

R2_dx
R2_dy
R2_total
angular error of recovered displacement
radial error
correlation of true vs recovered displacement magnitude

Use leave-quadrant-out as the primary CV scheme: hold out one quadrant of the displacement plane (e.g. dx>0, dy>0) as the test set and fit the recovery map on the remaining three quadrants. This forces extrapolation across displacement space and prevents leakage between correlated neighboring offsets in the structured grid. Report leave-one-radius-out as secondary only.

**Pre-specified statistic for matched vs. shuffled:** Wilcoxon signed-rank test on matched-minus-shuffled coordinate-recovery R² across images, computed within each session. Cross-session test: sign consistency of the session-level effect across all four sessions. Do not use grid-point-level replicates as the unit; image-level R² values are the replicates throughout.

Primary comparison:

matched Jacobian chart coordinate recovery
>
image-shuffled Jacobian chart coordinate recovery
>
random/global controls

Same-image PCA may outperform matched Jacobian, but it is descriptive. The important result is matched Jacobian beating image-shuffled.

Distance-dependent analysis

Compute coordinate recovery as a function of radius:

≤1 px
1–2 px
2–4 px
4–8 px

Expected result:

strong recovery locally,
smooth degradation with radius,
degradation is interpreted as curvature, not failure.

This should tie directly to Step 01.

Chart smoothness

Quantify whether nearby retinal offsets map to nearby neural chart coordinates.

Metrics:

Spearman corr(pairwise retinal distance, pairwise chart distance)
local neighbor preservation
trustworthiness / continuity if available
mean angular error between displacement vector and chart vector

Compare matched vs shuffled/random/global.

Chart dimensionality

Compute, but do not make it primary:

top-2 PCA variance explained
participation ratio
Jacobian basis variance capture

Reason: low dimensionality alone is not enough. Coordinate recovery is the main test.

Outputs

Per image/window CSV:

session
image_id
window_id / phase / history_id
n_neurons
baseline_model_rate
stim_rms
stim_grad_energy
jacobian_fro_norm
jacobian_rank_ratio

chart_basis
radius_bin
coord_R2_dx
coord_R2_dy
coord_R2_total
angular_error_deg
radial_error_px
smoothness_rho
variance_capture
n_grid_points

Per session summary JSON:

median_coord_R2_total_matched
median_coord_R2_total_shuffled
delta_coord_R2_matched_minus_shuffled
fraction_images_matched_gt_shuffled
median_angular_error_matched
median_angular_error_shuffled
distance_breakdown

Figures:

Example image translation chart:
true grid colored by dx/dy,
matched chart coordinates,
recovered coordinate overlay.
Cross-image boxplots:
matched vs shuffled vs random vs global coordinate recovery.
Distance-dependent breakdown:
coordinate recovery vs radius.
Relationship to Step 2A:
chart quality vs trace_cov_model_fem,
chart quality vs E_FEM_cv.
Phase 1 decision gate

Strong support for translation-chart language requires:

matched chart coordinate recovery beats image-shuffled in most sessions/images,
effect strongest locally and degrades with distance,
same-image PCA upper bound shows the chart exists,
global pooled basis is insufficient compared with image-specific matched chart.

If matched does not beat image-shuffled, downgrade the Figure 4 framework. Keep local generator result, but avoid “image-specific translation chart/fiber” language.

Phase 2. Real FEM sampling of the model chart
Purpose

Show that real FEM traces sample the same model translation charts revealed by controlled grid translations.

Inputs

From Phase 1:

U_J(image/window)
U_PCA_grid(image/window)
translation grid chart statistics

From Step 2A:

observed eye offsets per image/window
model responses under observed offsets
trace_cov_model_fem
E_FEM_cv
Analysis

For each image/window:

Generate model responses under observed eye offsets.
Compute residuals relative to baseline.
Project residuals into the matched chart:
z_fem(t) = Uᵀ [r(I_p(t)) - r(I_0)]
Compute covariance of z_fem and full response covariance.
Compute covariance capture by:
matched chart,
image-shuffled charts,
random charts,
global pooled chart.

Primary metric:

V_FEM_chart = trace(U Uᵀ C_FEM_model) / trace(C_FEM_model)
delta_V = V_matched - median(V_shuffled)

Also compute coordinate correspondence:

recovered eye offset from z_fem using Phase 1 coordinate map
corr(recovered dx, true dx)
corr(recovered dy, true dy)
angular error of recovered FEM displacement

This is still model-internal, but it connects real behavior to the chart.

Link to empirical Step 2A

For each image/window, ask:

Does chart quality predict E_FEM_cv?
Does V_FEM_chart or delta_V predict E_FEM_cv?
Does trace_cov_model_fem still predict E_FEM_cv after adding chart quality?

This links model chart geometry to empirical modulation strength.

Outputs

CSV additions:

fem_chart_capture_matched
fem_chart_capture_shuffled_median
fem_chart_capture_delta
fem_coord_R2_dx
fem_coord_R2_dy
fem_coord_R2_total
fem_coord_angular_error

Session summaries:

median_delta_V
fraction_images_delta_V_positive
rho(delta_V, E_FEM_cv)
rho(fem_coord_R2_total, E_FEM_cv)
Phase 2 decision gate

Strong support requires:

real FEM model responses are captured by matched charts above shuffled controls,
recovered FEM coordinates correspond to true eye offsets better in matched than shuffled charts,
chart-based metrics relate positively to reliable empirical FEM sensitivity.

If only the first two pass, the result supports model geometry but not empirical biology. If the third also passes, it supports a biology-first Figure 4.

Phase 3. Empirical geometry bridge
Purpose

Move beyond scalar E_FEM_cv and ask whether model-predicted translation geometry aligns with empirical eye-dependent response directions.

This is the high-risk, high-bar result needed for a true biology-first transformation-geometry claim.

Empirical object

For each image/window, use split-half empirical eye-sensitivity matrices:

B_emp_A, B_emp_B: NC x 2

These map eye displacement to empirical response modulation.

Treat the column space of B_emp as empirical FEM/translation sensitivity geometry.

Model objects to compare

For the same image/window:

local Jacobian J_I,
nonlinear model empirical-equivalent sensitivity B_model, fitted by regressing model responses under observed offsets onto eye offsets,
top PCs of nonlinear model FEM covariance.

B_model may be the fairest comparison to B_emp because both are regression-based eye-sensitivity matrices.

Primary comparisons
3.1 Empirical split-half ceiling

Before comparing model to data, compute empirical reliability ceiling:

align(B_emp_A, B_emp_B)

Metrics:

subspace alignment
principal angles
variance capture of B_B by B_A
column-wise x/y alignment if coordinate convention is stable

This determines whether empirical geometry is measurable.

3.2 Model-to-empirical alignment

Compare:

align(B_model, B_emp_cv)
align(J_I, B_emp_cv)
align(model_FEM_PCs, B_emp_cv)

where B_emp_cv can be represented by one half tested against model predictions from the other half, or by an average if reliability is acceptable.

Primary metric:

A_model_emp = trace(P_model C_emp_cv) / trace(C_emp_cv)

or subspace alignment between 2D spaces.

Use split-half cross-validation:

A_model_to_BA
A_model_to_BB
mean/symmetric score

**Report alignment relative to empirical ceiling.** Express model-to-empirical alignment as a fraction of the empirical split-half ceiling from Phase 0 (parallel):

```
A_relative = A_model_emp / median(align(B_emp_A, B_emp_B))
```

This is the honest way to report how much of the recoverable geometry the model captures. It prevents a reviewer from comparing an absolute alignment of (e.g.) 0.4 to a theoretical maximum of 1.0 when the empirical ceiling itself is 0.5.

3.3 Image-shuffled null

For each image, compare model geometry to empirical geometry using:

matched model geometry
image-shuffled model geometry, matched on norm/rate/stim stats
random 2D geometry
global pooled geometry

Primary test:

matched model-to-empirical alignment > image-shuffled alignment

This is the load-bearing empirical geometry test.

Outputs

Per image/window:

emp_split_alignment
model_B_alignment_to_emp
model_J_alignment_to_emp
model_PCA_alignment_to_emp
matched_minus_shuffled_alignment
n_valid_neurons
b_split_corr
e_fem_cv
trace_cov_model_fem

Session summary:

median_emp_split_alignment
median_model_emp_alignment
median_shuffled_alignment
delta_model_emp
fraction_images_model_gt_shuffled
correlation(delta_model_emp, E_FEM_cv)
Phase 3 decision gate

To claim empirical transformation geometry:

empirical split-half geometry must be measurable above noise,
model geometry must align with empirical geometry above image-shuffled controls,
effect should be positive in most sessions,
alignment should not be explained by eye amplitude, mean rate, or gradient energy.

If empirical split-half geometry is unreliable, do not claim geometry-level biology. Fall back to Step 2A scalar bridge.

Phase 4. Identity/transformation consequence
Purpose

Give the framework an encoding consequence.

Start with E-optotypes or the cleanest controlled stimulus class.

Analysis

For stimulus pair A/B:

task_signal = μ_B - μ_A
translation_plane_A = span(J_A)
mimicry_A_to_B = ||P_A task_signal||² / ||task_signal||²

Then relate mimicry to:

translation-induced confusability
real FEM vs stabilized decoder performance
orientation/identity decoder errors
threshold changes
Decision gate

Use this panel only if signal-tangent overlap predicts a downstream consequence clearly.

If not, keep it as supplement or discussion.

Recommended final Figure 4 if all goes well
A. Schematic

Repeated responses are not a noise cloud. They sample image-specific translation charts.

B. Model translation-grid chart

Matched image chart recovers true retinal Δx, Δy better than image-shuffled/random/global charts.

C. Local generator

Local Jacobian predicts small response increments; accuracy degrades with distance, showing a curved local manifold.

D. Empirical bridge

Nonlinear model FEM trace predicts reliable empirical E_FEM_cv, and, if Phase 3 passes, model geometry aligns with empirical eye-dependent response geometry above shuffled controls.

E. Encoding consequence

Identity differences can align with or lie transverse to translation directions; signal-tangent overlap predicts mimicry/confusability.

Immediate coding tasks
Task 0 (parallel with Task 1): Empirical split-half geometry feasibility check

Reuse the repeated split-half B_emp fits produced by the Task 1 script. Add a post-processing step or standalone analysis script:

```text
scripts/jacobian_predictive_framework/run_fixrsvp_empirical_geometry_ceiling.py
```

Outputs:

```text
empirical_geometry_ceiling.csv
empirical_geometry_ceiling_summary.json
figures/empirical_geometry_ceiling_distribution.png
```

Run immediately after Task 1 produces B_emp_A / B_emp_B splits. Gate Phase 3 development on the result.

Task 1: Update Step 2A robustness

File:

scripts/jacobian_predictive_framework/run_fixrsvp_step2.py

Add:

--n-split-repeats 100
--reliable-subset-summary
--residualized-controls

Outputs:

step2_image_windows.csv
step2_summary.json
step2_residualized_summary.json
step2_reliable_subset_summary.json
figures/step2_cv_bridge.png
Task 2: New script for Phase 1 chart test

Create:

scripts/jacobian_predictive_framework/run_fixrsvp_translation_chart.py

CLI:

--subject
--date
--dataset-configs-path
--checkpoint-path
--dataset-idx
--output-dir
--grid-px "-4,-3,-2,-1,-0.5,0,0.5,1,2,3,4"
--min-samples 50
--n-shuffle-matches 10
--n-random-subspaces 100
--basis-types jacobian,pca,shuffled,random,global
--model-device cuda

Outputs:

translation_chart_image_windows.csv
translation_chart_pairwise_grid.csv
translation_chart_summary.json
figures/chart_example_images.png
figures/chart_coordinate_recovery_summary.png
figures/chart_distance_breakdown.png
Task 3: Add Phase 2 FEM sampling to same or separate script

Preferred: same script with flag:

--run-fem-sampling

Outputs:

fem_chart_sampling.csv
fem_chart_sampling_summary.json
figures/fem_trajectory_examples.png
figures/fem_chart_capture_summary.png
Task 4: Only after Phase 1/2, add empirical geometry bridge

Create:

scripts/jacobian_predictive_framework/run_fixrsvp_empirical_geometry_bridge.py

Outputs:

empirical_geometry_bridge.csv
empirical_geometry_summary.json
figures/empirical_model_alignment_summary.png
Suggested function-level design
Shared helpers

Move/reuse from Step 01/Step 2:

_normalize_stim_dims
_shift_stimulus_batch
_predict_responses
_compute_jacobian
_choose_baseline
_resolve_pixels_per_degree
_collect_image_windows
_matched_candidate_ids
orthonormalize_basis
variance_capture
subspace_alignment
New helpers
def generate_translation_grid(grid_px: list[float]) -> np.ndarray:
    """Return (M,2) offsets for all dx,dy grid combinations."""

def compute_grid_responses(model, baseline_stim, offsets_px, dataset_idx) -> np.ndarray:
    """Return model responses to translated baseline stimulus."""

def compute_chart_bases(delta_r_grid, J, candidate_Js, n_random) -> dict:
    """Return matched J, PCA, shuffled J, random, and global bases."""

def coordinate_recovery_metrics(z, offsets_px, cv_mode="leave_radius_out") -> dict:
    """Fit offsets from chart coordinates and return R2/angle/radial metrics."""

def chart_smoothness_metrics(z, offsets_px) -> dict:
    """Distance preservation and neighborhood continuity metrics."""

def fem_sampling_metrics(responses_fem, offsets_fem, basis, coord_map) -> dict:
    """Covariance capture and coordinate recovery for observed FEM offsets."""

def subspace_alignment(U, V) -> float:
    """Mean squared cosine or projection overlap between two bases."""
Statistical reporting

For each session:

median across images
bootstrap CI over images
fraction images matched > shuffled
Wilcoxon signed-rank matched vs shuffled

Across sessions:

mean/median session effect
4-session sign consistency
session-level paired comparison

Avoid treating image × bin or grid points as independent biological replicates.

Critical failure modes and interpretations
Failure mode 1: matched chart does not beat image-shuffled chart

Interpretation:

The model has local Jacobians, but the broader chart/fiber language is not supported.
Figure 4 should shrink to Step 01 + scalar bridge.
Do not claim image-specific translation charts.
Failure mode 2: chart works in model but FEM sampling does not

Interpretation:

Controlled translations reveal local geometry, but real FEM distributions may be too broad, noisy, or mismatched.
Need distance-restricted FEM analysis or exact history windows.
Failure mode 3: Step 2A bridge fails after residualization

Interpretation:

Model geometry remains model-internal.
Biological bridge weakens substantially.
Use only conservative model-based claims.
Failure mode 4: empirical geometry split-half is unreliable

Interpretation:

Cannot test biology-first geometry with current data.
Scalar Step 2A may still be valid if E_FEM_cv is reliable.
Full empirical geometry becomes future work.

**This failure mode is now caught at Phase 0 (parallel), before Phases 1–2 are fully built.** If the empirical ceiling is at the noise floor, stop Phase 3 development immediately and invest only in the model-internal results (Phases 1–2) and the scalar bridge.
Failure mode 5: empirical geometry reliable but model does not align above shuffled

Interpretation:

Real V1 eye-dependent modulation exists but is not captured by current retinal reafference model.
Could reflect nonretinal behavior signals, calibration/model mismatch, or missing temporal history.
Final intended claim if all passes

Strong version:

Recorded V1 FEM-driven variability reflects sampling of image-specific retinal-translation geometry. The digital twin predicts both the strength and geometry of empirical eye-dependent modulation: translated versions of each image form local transformation charts in model population space, real FEM traces sample those charts, and model-predicted chart structure aligns with reliable empirical FEM sensitivity above image-shuffled controls.

Moderate version:

The digital twin reveals that FEMs sample image-specific translation charts in V1 population space, and the predicted strength of this sampling explains reliable image-wise empirical FEM sensitivity.

Conservative version:

Local image-translation Jacobians explain model FEM covariance geometry, and nonlinear model FEM sensitivity predicts reliable empirical FEM drive, suggesting that at least part of recorded V1 shared variability reflects retinal reafference.

Priority order
Step 2A robustness lock (Task 1) **in parallel with** empirical geometry feasibility check (Task 0)
Phase 1 translation-grid coordinate recovery — only after Task 0 and Task 1 are complete
Phase 2 real FEM sampling of model chart
Phase 3 empirical geometry bridge — only if Task 0 shows empirical ceiling is above noise floor
Phase 4 E-optotype or controlled mimicry consequence
Optional only: Δz decoding, synthetic rotation/scale

Do not run Δz decoding before Phase 1. Do not run rotation/scale for this paper.
Do not develop Phase 3 script if Task 0 shows empirical geometry is unrecoverable.
"""