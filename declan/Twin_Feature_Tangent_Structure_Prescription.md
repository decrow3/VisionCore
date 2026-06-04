# Prescription: Twin feature-tangent analyses of the first-order FEM rate map

**Working title.** Twin feature-tangent structure: a first-order mechanism for reafferent covariance

**Scope.** This document prescribes a set of deterministic digital-twin analyses to explain how fixational eye movements (FEMs) generate structured V1 population variability through the first-order rate map. The prior covariance prescription asked why `Σ_FEM` is low-rank, signal-aligned, and stimulus-dependent. This document moves one level upstream: it asks whether those covariance properties are the second-order consequence of a lawful first-order map,

```text
r_I(e) = model-predicted population response to image I at retinal eye position e
```

and its local translation tangents,

```text
b_x(I) = ∂r_I(e)/∂x
b_y(I) = ∂r_I(e)/∂y
```

The analysis is restricted to the canonical shared twin population. It does **not** require estimating receptive fields, STAs, simple/complex status, orientation tuning, or spatial-frequency tuning for the recorded units. In the twin, the model itself is the image-computable feature operator.

The goal is not to show that the same signed x/y tangent direction is conserved across images. That test is too strict and has already suggested that signed directions are strongly image-specific. The goal is to test a subtler and more plausible claim:

> Different images generate different translation tangents, but those tangents are produced by a shared feature operator and may live in a compact, feature-defined subspace.

This reconciles two observations that otherwise appear in tension:

1. `Σ_FEM` is low-rank and aligned with the stimulus-driven subspace.
2. Cross-image signed tangent alignment can be near zero, because the tangent direction depends on local image content.

The conserved object may be the **operator**, **subspace**, or **metric law**, not a single image-independent signed direction.

---

## 1. Why this pivot is justified

The digital-twin figure already supports the mechanism. The model includes an extraretinal eye-position/velocity input, but ablating that input has essentially no effect on trial-level prediction: the manuscript draft reports a median `Δr² = -0.003`, with the ablation cost flat against FEM modulation (`ρ = 0.07`, n.s.). Therefore, in the twin, FEM-driven variability is carried almost entirely by the moving retinal image, not by a separate gain field or extraretinal eye-position route.

That makes the twin the right tool for this question. The twin is the learned function

```text
retinal image history → canonical population response
```

so it directly implements the hypothesized feature operator. For the canonical twin cells, we can compute the response derivative with respect to retinal translation without first estimating RFs or cell classes.

This also connects directly to the covariance results. The recorded/twin paper already reports that FEM covariance is low-dimensional and aligned with stimulus-tuned structure. The first-order tangent analyses below ask **why**: are these low-rank covariance effects produced because image translations, though content-dependent, pass through a compact shared feature operator?

---

## 2. Boundary conditions

Do not frame these analyses as performance or decoding claims.

Do **not** claim:

```text
FEMs improve coding
the brain decodes this structure
the shared tangent subspace is behaviorally optimal
the tangent structure is information-limiting in magnitude
```

Do claim only structural statements:

```text
The first-order rate map has a compact tangent structure.
The tangent family generalizes across images.
The local translation metric is lawfully related to image structure.
The first-order tangent approximation explains a measurable fraction of the full FEM covariance.
```

The correct conceptual hierarchy is:

```text
First-order object:
    r_I(e)

Local tangent structure:
    J_I = [b_x(I), b_y(I)]

Second-order covariance:
    C_FEM(I) = Cov_e[r_I(e)]

Small-displacement approximation:
    C_FEM(I) ≈ J_I Cov(e) J_Iᵀ
```

The covariance is the shadow cast by the rate map under the eye-position distribution.

---

## 3. Implementation scope

Create a new module, separate from the STG recorded-analysis code:

```text
declan/twin_feature_tangent_structure/
```

Suggested runner:

```text
run_twin_feature_tangent_structure.py
```

Suggested output root:

```text
outputs/twin_feature_tangent_structure/
```

Inputs:

```text
canonical shared twin model/readout
fixRSVP or natural-image bank used in the current twin analyses
canonical twin cell responses
eye-position cloud or synthetic local displacement grid
```

Core output files:

```text
twin_tangent_maps.pkl
twin_tangent_union_spectrum.csv
twin_tangent_train_test_basis.csv
twin_tangent_metric_by_image.csv
twin_tangent_metric_image_law.csv
twin_linear_covariance_approx.csv
twin_feature_tangent_summary.json
README.md
figures/
```

Use finite differences first. Autograd can be added later if convenient, but finite differences are sufficient and easier to audit.

Recommended finite-difference displacement:

```text
delta = small retinal offset in model input coordinates
```

Use several values:

```text
delta ∈ {0.25, 0.5, 1.0} arcmin, or nearest model-pixel equivalents
```

The exact coordinate conversion must be recorded in metadata.

For each image `I`, compute:

```text
r0      = response(image I at baseline retinal position)
r_xplus = response(image I shifted +delta in x)
r_xminus= response(image I shifted -delta in x)
r_yplus = response(image I shifted +delta in y)
r_yminus= response(image I shifted -delta in y)

b_x(I) = (r_xplus - r_xminus) / (2 * delta)
b_y(I) = (r_yplus - r_yminus) / (2 * delta)
J_I    = [b_x(I), b_y(I)]
```

Also compute a finite-cloud version using the actual eye-position samples:

```text
R_I = responses for image I under sampled eye positions
C_FEM(I) = Cov_e(R_I)
```

---

# Analysis 1: Tangent union compactness

## Aim

Test whether image-specific translation tangents live in a compact shared subspace across images.

This is the core test for the claim:

> The shared structure is not a universal signed x/y axis, but a compact feature-defined tangent subspace.

## What it computes

Stack tangents across images:

```text
B = [b_x(I_1), b_y(I_1), b_x(I_2), b_y(I_2), ..., b_x(I_N), b_y(I_N)]
```

where each column is a vector over canonical twin cells.

Compute:

```text
eigenvalues of B Bᵀ
cumulative variance explained
participation ratio
rank needed for 50%, 75%, 90%, 95% variance
separate spectra for b_x only, b_y only, and combined [b_x,b_y]
```

## Required outputs

```text
twin_tangent_union_spectrum.csv
```

Columns:

```text
delta
image_set
n_images
n_cells
tangent_set              # bx / by / combined
component_index
eigenvalue
fraction_variance
cumulative_fraction_variance
participation_ratio
rank_50
rank_75
rank_90
rank_95
```

Figure:

```text
figures/twin_tangent_union_spectrum.png/pdf/svg
```

Plot cumulative variance and eigenvalue spectrum for `bx`, `by`, and combined tangents.

## Nulls

Use at least:

```text
unit-shuffled tangents:
    shuffle cell identities independently for each image tangent

image-sign-shuffled tangents:
    random sign flips of tangent vectors, preserving norms

random orthogonal basis null:
    random vectors matched for norm and number of images

phase-scrambled image tangents, if easy:
    recompute tangents on phase-scrambled versions of images
```

The most important null is unit-shuffle across images, because it preserves each image's tangent magnitude distribution but destroys cross-image population structure.

## Interpretation

```text
Compact observed spectrum << unit-shuffled/null:
    evidence for a shared tangent subspace.

Observed spectrum comparable to null:
    tangents are image-specific without detectable shared subspace.

Very low dimension, near 2:
    unexpectedly strong shared x/y-like tangent subspace.

Moderate dimension:
    shared feature-defined subspace, not a simple two-axis translation code.
```

## Relationship to prior covariance work

This is the first-order counterpart of low-rank `Σ_FEM`. If `Σ_FEM` has PR ≈ 2–3 but signed cross-image tangent alignment is low, this analysis tests whether the *union* of many image-specific tangent directions is nevertheless compact.

## Priority

Essential. Run first.

---

# Analysis 2: Train/test shared tangent basis

## Aim

Test whether a tangent basis learned from one set of images generalizes to held-out images.

This is stronger than compactness alone because it asks whether the shared subspace is not just low-dimensional within a sampled image set, but predictive across new image content.

## What it computes

Split images into train/test sets.

From training images:

```text
B_train = [b_x, b_y] for training images
U_train = top-k PCs of B_train
```

For held-out images:

```text
B_test = [b_x, b_y] for held-out images
variance_capture(k) = ||U_trainᵀ B_test||²_F / ||B_test||²_F
```

Repeat across folds.

Evaluate for:

```text
k = 1, 2, 3, 5, 10, 20, 50
```

and for both:

```text
combined tangents [b_x,b_y]
bx only
by only
```

## Required outputs

```text
twin_tangent_train_test_basis.csv
```

Columns:

```text
delta
fold
train_n_images
test_n_images
tangent_set
basis_rank_k
test_variance_captured
null_mean
null_ci_low
null_ci_high
effect_minus_null
interpretation_label
```

Figure:

```text
figures/twin_tangent_train_test_basis.png/pdf/svg
```

Plot held-out variance captured versus basis rank, with null bands.

## Nulls

Use:

```text
random basis matched for k
unit-shuffled training tangents
image-label-shuffled tangents
```

For the unit-shuffle null, shuffle cell identities in `B_train` before learning the basis, then evaluate on unshuffled `B_test`.

## Interpretation

```text
High held-out variance capture above null:
    tangent maps across images share a generalizable feature-defined subspace.

Low held-out variance capture:
    compactness, if present, may be image-set-specific or driven by a few images.

Held-out capture requiring high k:
    shared structure exists but is not very compact.
```

## Relationship to manuscript

This is the cleanest way to say:

> The conserved object is a shared tangent subspace that generalizes across images, not a single signed x/y direction.

## Priority

Essential. Run after Analysis 1.

---

# Analysis 3: Feature-gradient law inside the model

## Aim

Test whether image-specific tangents are generated by the model's feature sensitivity acting on image gradients.

This is the precise twin version of:

```text
fixed V1 feature operator × image gradient → translation tangent
```

It replaces the RF/STA problem in recorded data with a model-native derivative analysis.

## What it computes

The mathematical identity is:

```text
∂r/∂x = (∂r/∂stimulus) · (∂stimulus/∂x)
∂r/∂y = (∂r/∂stimulus) · (∂stimulus/∂y)
```

There are two implementation levels.

## Level 1: finite-difference feature-gradient prediction

For each image:

1. Compute output tangents `b_x`, `b_y` from shifted images.
2. Compute image gradients `∂I/∂x`, `∂I/∂y`.
3. Compute simple image-gradient statistics:
   - RMS gradient magnitude
   - gradient anisotropy
   - dominant orientation
   - Fourier amplitude / orientation energy
4. Relate these image statistics to tangent norms and tangent metrics.

This is easy and provides the bridge to Analysis 4.

## Level 2: layerwise derivative propagation

For each selected model layer `L`:

```text
h_L(I) = feature representation at layer L
∂h_L/∂x = [h_L(I shifted +dx) - h_L(I shifted -dx)] / 2dx
∂h_L/∂y = [h_L(I shifted +dy) - h_L(I shifted -dy)] / 2dy
```

Then ask:

```text
At which layer does the tangent union become compact?
How much output tangent variance is explained by layer-L tangent features?
Does compactness emerge early or after nonlinear/recurrent processing?
```

If layerwise readout weights are accessible, decompose output tangents into feature-channel contributions.

## Required outputs

```text
twin_feature_gradient_law.csv
twin_layerwise_tangent_compactness.csv
```

Columns for `twin_feature_gradient_law.csv`:

```text
delta
image_id
gradient_rms_x
gradient_rms_y
gradient_rms_total
gradient_orientation_anisotropy
dominant_gradient_orientation
norm_bx
norm_by
norm_combined
metric_gxx
metric_gyy
metric_gxy
metric_anisotropy
metric_dominant_axis_angle
```

Columns for `twin_layerwise_tangent_compactness.csv`:

```text
delta
layer_name
n_features
n_images
tangent_set
participation_ratio
rank_50
rank_75
rank_90
rank_95
train_test_basis_rank
heldout_variance_captured
null_mean
null_ci_low
null_ci_high
```

Figures:

```text
figures/twin_gradient_stats_vs_tangent_norm.png/pdf/svg
figures/twin_layerwise_tangent_compactness.png/pdf/svg
```

## Optional autograd version

If convenient, compute exact derivatives using autograd through the differentiable shift/rendering operation. If not, finite differences are acceptable and preferable for auditability.

## Interpretation

```text
Tangent magnitude/metric predicted by image-gradient statistics:
    translation response geometry is lawfully tied to image structure.

Layerwise compactness emerges early:
    shared structure is mostly feedforward/local-feature driven.

Layerwise compactness emerges late:
    nonlinear/recurrent model stages organize translation tangents into a shared subspace.

No relationship to gradient structure:
    tangent compactness may reflect model/readout structure rather than image-gradient law.
```

## Priority

Strong support. Start with Level 1, then add Level 2 if feasible.

---

# Analysis 4: Translation metric law, not signed direction

## Aim

Characterize the law governing the 2D local translation metric for each image.

Signed cross-image tangent alignment can be low because image-specific tangents point in different population directions. The local metric,

```text
G_I = J_Iᵀ J_I
```

is basis-invariant within the population and captures how strongly the image drives responses for different displacement directions.

## What it computes

For each image:

```text
J_I = [b_x, b_y]
G_I = J_Iᵀ J_I =
    [[||b_x||², b_x · b_y],
     [b_x · b_y, ||b_y||²]]
```

From `G_I`, compute:

```text
gxx = ||b_x||²
gyy = ||b_y||²
gxy = b_x · b_y
trace = gxx + gyy
determinant
anisotropy = (lambda_max - lambda_min) / (lambda_max + lambda_min)
dominant_displacement_axis_angle
condition_number
```

Relate these metrics to image structure:

```text
RMS contrast
gradient magnitude
gradient anisotropy
dominant orientation energy
Fourier orientation/SF energy
```

## Required outputs

```text
twin_tangent_metric_by_image.csv
twin_tangent_metric_image_law.csv
```

Columns for `twin_tangent_metric_by_image.csv`:

```text
delta
image_id
gxx
gyy
gxy
metric_trace
metric_det
metric_lambda1
metric_lambda2
metric_anisotropy
metric_dominant_axis_angle
norm_bx
norm_by
cos_bx_by
```

Columns for `twin_tangent_metric_image_law.csv`:

```text
delta
predictor_set
target_metric
model_type              # linear / ridge / random_forest_optional
crossval_r2
crossval_corr
null_mean
null_ci_low
null_ci_high
effect_minus_null
interpretation_label
```

Figures:

```text
figures/twin_metric_anisotropy_vs_image_orientation.png/pdf/svg
figures/twin_metric_prediction_cv.png/pdf/svg
```

## Nulls

Use:

```text
image-label shuffle
phase-scrambled images
rotation-shuffled gradients
random image features matched for contrast
```

## Interpretation

```text
Metric law predicted by image structure:
    the local geometry of translation is image-dependent but lawful.

No metric prediction:
    shared tangent compactness, if present, may not be explained by simple image statistics.

Metric law strong while signed alignment weak:
    conserved structure lives at metric/geometry level, not signed population direction.
```

## Priority

Strong support. This is the right successor to the discarded Gram-template comparison: not as a recorded-twin signed match, but as a model-internal characterization of local translation geometry.

---

# Analysis 5: First-order tangent approximation to full FEM covariance

## Aim

Test how much of the full FEM covariance is explained by the local first-order rate map.

This links the new first-order analysis back to the paper's covariance backbone.

## What it computes

For each image, compute the full sampled model covariance:

```text
C_full(I) = Cov_e[r_I(e)]
```

using the real eye-position samples or a fixed synthetic eye cloud.

Compute the local linear approximation:

```text
C_lin(I) = J_I Cov(e) J_Iᵀ
```

where `J_I = [b_x,b_y]`.

Compare `C_lin` and `C_full` by structure, not raw magnitude only.

Metrics:

```text
subspace overlap between top eigenvectors of C_lin and C_full
trace ratio tr(C_lin) / tr(C_full)
Frobenius correlation between C_lin and C_full
fraction of C_full variance captured by C_lin subspace
participation ratio of C_lin vs C_full
```

Run this across eye-cloud radii:

```text
cloud_scale = 0.25x, 0.5x, 1x, 2x real FEM cloud
```

The expectation is:

```text
small cloud:
    local tangent approximation should work well

large cloud:
    magnitude may fail because of curvature/nonlinearity,
    but subspace may remain aligned
```

This preserves the guardrail from the older Jacobian analysis: do not overclaim a magnitude identity at the real cloud scale.

## Required outputs

```text
twin_linear_covariance_approx.csv
```

Columns:

```text
delta
image_id
cloud_source              # real_eye / synthetic_gaussian
cloud_scale
n_eye_samples
trace_c_full
trace_c_lin
trace_ratio_lin_full
frobenius_corr_lin_full
subspace_overlap_k1
subspace_overlap_k2
subspace_overlap_k3
full_pr
lin_pr
fraction_full_variance_in_lin_subspace_k1
fraction_full_variance_in_lin_subspace_k2
fraction_full_variance_in_lin_subspace_k3
interpretation_label
```

Figures:

```text
figures/twin_linear_covariance_approx_vs_cloud_scale.png/pdf/svg
figures/twin_linear_vs_full_covariance_subspace.png/pdf/svg
```

## Interpretation

```text
C_lin aligns with C_full at small clouds:
    FEM covariance is locally generated by first-order translation tangents.

Alignment persists but magnitude fails at real cloud:
    direction is first-order, magnitude reflects finite-cloud curvature/nonlinearity.

C_lin fails even at small cloud:
    tangent estimation or coordinate handling is likely wrong.
```

## Relationship to old failed Jacobian work

This explicitly avoids repeating the failed claim:

```text
C_FEM ≈ J Cov(e) Jᵀ at full real cloud scale
```

as a magnitude identity. Instead, it tests where the approximation holds and emphasizes subspace alignment and cloud-scale dependence.

## Priority

Essential bridge back to covariance. Run after Analyses 1–2.

---

## 4. Required metadata and guardrails

Every run should write:

```text
twin_feature_tangent_summary.json
```

with:

```json
{
  "analysis_name": "twin_feature_tangent_structure",
  "model_name": "...",
  "readout_name": "...",
  "n_canonical_cells": 756,
  "image_set": "...",
  "n_images": "...",
  "delta_values": "...",
  "eye_cloud_source": "...",
  "coordinate_frame": "...",
  "finite_difference_units": "...",
  "performance_claim": false,
  "requires_recorded_rf_estimation": false,
  "uses_recorded_units": false,
  "guardrails": [
    "Do not claim behavioral benefit or decoding performance.",
    "Do not compare canonical twin tangent vectors directly to recorded tangent vectors.",
    "Do not interpret low signed cross-image alignment as absence of shared structure.",
    "The conserved object may be a subspace, metric, or operator rather than a signed x/y direction.",
    "For C_lin versus C_full, report subspace and scale dependence; do not assert a full-cloud magnitude identity."
  ]
}
```

---

## 5. Recommended execution order

1. **Tangent maps**: compute `b_x`, `b_y` for all selected images and deltas.
2. **Analysis 1**: union compactness.
3. **Analysis 2**: train/test shared tangent basis.
4. **Analysis 5**: first-order approximation to full FEM covariance.
5. **Analysis 4**: metric law by image statistics.
6. **Analysis 3**: layerwise / feature-gradient law, starting with simple image-gradient statistics and adding layerwise features if feasible.

This order gives an early answer to the main paper question before deeper model introspection.

---

## 6. Success criteria

A strong structural result would look like:

```text
1. The union of image-specific tangents is compact relative to nulls.
2. A basis learned from training images captures held-out tangent variance.
3. The first-order tangent approximation captures the dominant subspace of full FEM covariance.
4. The 2D translation metric is lawfully related to image gradient/orientation statistics.
5. Layerwise analysis shows where compactness emerges in the model.
```

The manuscript claim would be:

> FEM-induced covariance arises from a lawful first-order rate map: retinal translations pass image gradients through a fixed feature-tuned V1 operator, producing image-specific tangent maps that nevertheless occupy a compact, shared feature-defined subspace. Thus, the conserved object is not a universal signed displacement axis, but the operator/subspace family that generates content-dependent translation responses.

A weak or negative result would be:

```text
Tangents are high-dimensional, do not generalize across images, and do not explain the full FEM covariance subspace.
```

In that case, the paper should keep the twin covariance results descriptive and not pursue the feature-tangent structure claim.

---

## 7. Minimal command sketch

The exact command will depend on model-loading conventions, but the intended interface is:

```bash
python -m declan.twin_feature_tangent_structure.run_twin_feature_tangent_structure \
  --dataset-configs-path experiments/dataset_configs/multi_basic_240_rsvp.yaml \
  --model-path <canonical_model_checkpoint> \
  --readout-name canonical_fovea \
  --image-source fixrsvp \
  --n-images 64 \
  --delta-arcmin 0.25,0.5,1.0 \
  --eye-cloud-source real_eye \
  --cloud-scales 0.25,0.5,1.0,2.0 \
  --device cuda \
  --batch-size 64 \
  --out-dir outputs/twin_feature_tangent_structure
```

If model path/readout loading is already handled by existing A3/twin covariance code, reuse that loader rather than creating a second model-loading path.

---

## 8. Stop rule

Run the first pass on the canonical shared twin cells only.

Do not add recorded RF/STAs, recorded unit matching, or decoder/performance analyses until the twin-only structure is clear.

If Analyses 1, 2, and 5 do not support compact/generalizable tangent structure, stop. Do not chase layerwise explanations.

If Analyses 1, 2, and 5 support the structure, then add metric-law and layerwise analyses to explain what image properties and model stages generate it.
