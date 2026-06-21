# Handoff: Shared Transformation Geometry Analysis

## Working name

**Shared Transformation Geometry (STG)**

Avoid calling this "A3b" in code, filenames, figure titles, or summaries. The analysis is conceptually adjacent to A3, but the aim is broader:

> Test whether FEM-induced population structure contains a shared retinal-transformation geometry across image contents, beyond trivial displacement magnitude and beyond image similarity.

Suggested module name:

```text
shared_transformation_geometry
```

Suggested output root:

```text
outputs/twin_covariance_structure/shared_transformation_geometry/
```

Suggested short label in tables and figures:

```text
STG
```

---

## Scientific motivation

The main paper result is that a substantial component of foveal V1 shared variability during fixation is reafferent: it is attributable to measured eye-position-dependent modulation of the visual response. The current twin and recorded analyses support a structured covariance story:

- FEM covariance is signal-aligned.
- FEM covariance aligns with retinal-translation tangent directions.
- FEM covariance is low-dimensional at adequate sampling, though its rank is better explained by finite sampling of a curved response manifold than by literal eye-motion degrees of freedom.
- FEM covariance is shaped by eye-position occupancy.
- In high-support fixRSVP images, both recorded V1 and the deterministic twin show a modest content-specific component, with within-image covariance-subspace overlap exceeding cross-image overlap.

However, the high-support A3 result also showed that cross-image covariance overlap remains large. This suggests that FEM covariance may contain two components:

```text
shared transformation backbone
+
image-specific warping
```

The Shared Transformation Geometry analysis asks what the shared backbone is.

The strong version of the question is:

> Does recorded V1 contain a conserved retinal-transformation geometry across image contents, expressed through FEM-linked covariance, after controlling for trivial eye-displacement magnitude and low-level image similarity?

This is a structural question, not a performance question. It does not ask whether FEMs improve coding, whether velocity is decoded, whether downstream circuits use the signal, or whether eye movements are optimized. It asks whether the covariance geometry induced by measured retinal translation preserves a shared transformation structure across images.

If the result works in recorded V1 across sessions and animals, it could become a headline-level synthesis:

> Foveal V1 does not separate content and transformation into independent channels. Instead, self-generated retinal translations impose a shared covariance geometry that is expressed through, and warped by, image content.

---

## Guardrails

Do not implement or report:

```text
velocity decoding
trajectory-order decoding
behavioral benefit
information gain
ideal-observer performance
nonlinear decoder rescue
active-sensing optimization
```

Do not report raw cross-image response-RDM similarity as evidence of nontrivial shared transformation geometry. Raw RDM similarity can be trivially high because all images share the same eye-position distance matrix.

The primary evidence must survive controls for:

```text
1. eye-displacement magnitude and axis-specific displacement terms
2. low-level image similarity
3. matched dimensionality / eigenspectrum nulls for directional or topology tests
4. session-level replication
```

---

## Key terminology

### RDM

RDM means **representational dissimilarity matrix**.

For image I, and eye-position samples a,b:

```text
D_I(a,b) = distance(r_I(e_a), r_I(e_b))
```

Each entry is the distance between two population response patterns evoked by the same image at two eye positions.

The corresponding eye-position RDM is:

```text
D_eye(a,b) = ||e_a - e_b||
```

Raw D_I will usually correlate with D_eye, because larger retinal shifts tend to cause larger response changes. That is expected and not sufficient.

### Nontrivial shared transformation geometry

A nontrivial shared transformation geometry is present only if cross-image response geometry remains similar **after removing** the geometry expected from eye-displacement magnitude and image similarity.

---

## Analysis overview

Implement STG in two stages.

### Stage 0: support census

Before running the analysis, report the available data across all candidate sessions.

For each session:

```text
subject
date
source availability: recorded, twin, or both
n_units
n_unique_fixRSVP_images
image support distribution
n_images >= 80 samples
n_images >= 160 samples
n_images >= 320 samples
n_images >= 640 samples, if available
eye trace availability
twin response availability or cache status
```

Save:

```text
stg_support_census.csv
stg_support_census_summary.json
```

Do not proceed to a pooled analysis until the support census is written.

Recommended inclusion criteria:

```text
preferred:
  >=8 images with >=320 samples

minimum:
  >=8 images with >=160 samples

exclude:
  sessions with fewer than 8 usable images at the chosen threshold
```

Use session-level statistics. Do not pool raw neurons across sessions.

---

## Stage 1: residual shared response geometry

### Goal

Test whether response geometry is shared across images after removing trivial displacement geometry.

### Inputs

For each included session, source, image set, and image:

```text
R_I: response samples
E: corresponding eye-position samples
image_id
source: recorded or twin
session_id
subject
date
```

Use high-support fixRSVP images first.

Recommended first run:

```text
n_samples = 320
image_set = direct_surviving and matched_common
sources = recorded and twin, where available
```

If 320 is too restrictive across sessions, also run 160, but keep 320 as the preferred claim-supporting threshold.

### Response representation

Use the response representation closest to the existing A3 covariance audit.

Allowed options:

```text
full_response:
  population response vector per eye sample

fem_pca_k2:
  response projected into image-specific top-2 FEM covariance subspace

fem_pca_k3:
  response projected into image-specific top-3 FEM covariance subspace
```

Start with `full_response`. Add FEM-PCA spaces only after the full-response analysis is stable.

For recorded data, retain the exact unit subset used in the core analysis. Record n_units.

For twin data, record whether responses are deterministic/noiseless.

### Response distance metrics

Run at least:

```text
euclidean_zscored:
  Euclidean distance after within-image unit z-scoring across samples

correlation_distance:
  1 - Pearson correlation between response vectors
```

Also optionally include:

```text
euclidean_centered:
  Euclidean distance after subtracting within-image mean response, without unit z-scoring
```

Normalization matters. Report it explicitly.

### Build RDMs

For each image I:

1. Select the same n eye samples used in the high-support A3 audit.
2. Compute response RDM D_I.
3. Compute eye geometry predictors for all sample pairs:

```text
radial_distance = sqrt(dx^2 + dy^2)
abs_dx = abs(dx)
abs_dy = abs(dy)
dx2 = dx^2
dy2 = dy^2
abs_dxdy = abs(dx * dy)
angle = atan2(dy, dx)
```

Vectorize the upper triangle of each RDM and predictor matrix.

---

## Stage 1A: raw smoothness baseline

For each image, compute:

```text
corr(vec(D_I), radial_distance)
corr(vec(D_I), abs_dx)
corr(vec(D_I), abs_dy)
```

These are smoothness baselines, not evidence of nontrivial shared transformation geometry.

Save:

```text
stg_image_smoothness_metrics.csv
```

Columns:

```text
session_id
subject
date
source
image_set
image_id
n_samples
n_units
distance_metric
response_space
corr_response_radial_pearson
corr_response_radial_spearman
corr_response_absdx_pearson
corr_response_absdy_pearson
response_rdm_mean
response_rdm_std
eye_rdm_mean
eye_rdm_std
```

---

## Stage 1B: residual RDM after eye-geometry nuisance regression

This is the primary STG analysis.

For each image, fit:

```text
vec(D_I) =
  beta_0
  + beta_1 * radial
  + beta_2 * abs(dx)
  + beta_3 * abs(dy)
  + beta_4 * dx^2
  + beta_5 * dy^2
  + beta_6 * abs(dx * dy)
  + epsilon_I
```

Also run a simpler radial-only model:

```text
vec(D_I) = beta_0 + beta_1 * radial + epsilon_I
```

Report both:

```text
residual_model = radial_only
residual_model = expanded_eye_geometry
```

The expanded model is the primary nontrivial test.

After residualization, verify:

```text
corr(epsilon_I, radial_distance) is approximately 0
corr(epsilon_I, abs_dx) is reduced
corr(epsilon_I, abs_dy) is reduced
```

Then compute cross-image residual similarity:

```text
corr(epsilon_I, epsilon_J)
```

Use both Pearson and Spearman where feasible.

Save:

```text
stg_cross_image_residual_rdm.csv
```

Columns:

```text
session_id
subject
date
source
image_set
image_i
image_j
n_samples
n_units
distance_metric
response_space
residual_model
raw_rdm_corr_pearson
raw_rdm_corr_spearman
residual_rdm_corr_pearson
residual_rdm_corr_spearman
image_similarity
image_similarity_metric
```

Aggregate per session:

```text
stg_session_summary.csv
```

Columns:

```text
session_id
subject
date
source
image_set
n_images
n_samples
n_units
distance_metric
response_space
residual_model
mean_raw_cross_image_rdm_corr
mean_residual_cross_image_rdm_corr
bootstrap_ci_low_residual_corr
bootstrap_ci_high_residual_corr
bootstrap_p_residual_corr_le_0
interpretation_label
```

Bootstrap over images, not RDM entries. RDM entries are not independent.

---

## Stage 1C: control for low-level image similarity

A positive residual RDM correlation may reflect natural images having similar low-level statistics, not a content-invariant transformation geometry. Add image-similarity controls.

Compute image similarity metrics for each image pair:

```text
pixel_correlation, after resizing/cropping to common stimulus region
RMS contrast difference
orientation/SF energy similarity, if available
Gabor/filter-bank energy similarity, if available
CNN early-layer feature similarity, optional
```

Minimum required:

```text
pixel_correlation
RMS contrast difference
orientation/SF energy similarity OR simple Fourier amplitude spectrum similarity
```

Then test whether residual geometry survives after controlling image similarity.

Two acceptable approaches:

### Regression approach

Across image pairs within session:

```text
residual_rdm_corr_IJ =
  alpha_0
  + alpha_1 * image_similarity_IJ
  + eta_IJ
```

Report the intercept and residual mean.

### Stratified approach

Split image pairs into low-similarity and high-similarity bins. The strong result is residual geometry above zero even for low-similarity image pairs.

Save:

```text
stg_image_similarity_controls.csv
```

Columns:

```text
session_id
subject
date
source
image_set
image_i
image_j
image_similarity_metric
image_similarity_value
residual_rdm_corr
low_similarity_bin
```

Primary interpretive threshold:

```text
residual shared geometry should remain positive in low-image-similarity pairs,
or remain positive after regression on image similarity.
```

---

## Stage 2: twin-template confirmation in recorded data

### Goal

Use the noiseless twin to define a predicted shared transformation geometry, then ask whether recorded V1 expresses that geometry above null.

This is more powerful than asking recorded data to estimate the residual geometry from scratch.

### Construct twin template

Within each session or matched image set, using twin responses:

1. Build residual RDMs epsilon_I_twin after expanded eye-geometry nuisance regression.
2. Average across images:

```text
T_twin = mean over images of zscore(epsilon_I_twin)
```

Optionally compute leave-one-image-out templates:

```text
T_twin_minus_I
```

Save:

```text
stg_twin_templates.pkl
stg_twin_template_summary.csv
```

### Test recorded alignment

For each recorded image I:

1. Build recorded residual RDM epsilon_I_rec.
2. Correlate it with the twin template:

```text
template_match_I = corr(epsilon_I_rec, T_twin_minus_I)
```

Use leave-one-image-out if the same image appears in the twin template.

Also compute session-level mean template match.

Save:

```text
stg_recorded_twin_template_match.csv
```

Columns:

```text
session_id
subject
date
image_id
n_samples
n_units
distance_metric
response_space
residual_model
template_match
template_match_spearman
image_similarity_to_template_mean
```

Aggregate per session and subject:

```text
stg_template_match_summary.csv
```

Columns:

```text
session_id
subject
date
n_images
n_samples
n_units
distance_metric
response_space
residual_model
mean_template_match
bootstrap_ci_low
bootstrap_ci_high
bootstrap_p_match_le_0
interpretation_label
```

### Nulls for template match

Implement at least:

```text
eye-label shuffle:
  permute eye-position labels before building recorded residual RDMs

image-label shuffle:
  match recorded image residuals to twin templates from mismatched images, if applicable

matched-rank random template:
  create random residual templates with same length and approximate covariance/eigenspectrum, if feasible
```

At minimum, template match must exceed eye-label shuffle and image-label shuffle.

---

## Stage 3: direction/topology preservation

### Goal

Test whether displacement direction has conserved organization across images, beyond magnitude.

This is the most direct version of the shared transformation coordinate idea.

### Method

For each pair of eye samples:

```text
dx = x_b - x_a
dy = y_b - y_a
radial = sqrt(dx^2 + dy^2)
angle = atan2(dy, dx)
```

To avoid magnitude confounds, restrict to a distance band:

```text
primary band:
  radial percentile 40-60

secondary bands:
  25-50
  50-75
```

Bin angles:

```text
8 bins over [-pi, pi)
```

For each image, compute mean residual response distance by angle bin:

```text
M_I[angle_bin] = mean residual_D_I(a,b) for pairs in bin
```

Use residual distances after expanded eye-geometry regression where possible. If using raw distances, distance-band matching is mandatory.

Normalize angle profiles within image:

```text
M_I_z = zscore(M_I)
```

Compute across-image similarity:

```text
zero_shift_corr = corr(M_I_z, M_J_z)
best_circular_shift_corr = max_shift corr(roll(M_I_z, shift), M_J_z)
best_shift_bins
```

Interpretation:

```text
zero_shift_corr > null:
  shared absolute directional anisotropy

best_shift_corr > null but zero_shift weak:
  shared directional topology up to image-specific rotation

neither > null:
  no directional/topological preservation
```

Save:

```text
stg_direction_topology.csv
stg_direction_topology_summary.csv
```

Columns for pairwise file:

```text
session_id
subject
date
source
image_set
image_i
image_j
n_samples
distance_metric
response_space
distance_band
angle_bins
zero_shift_angle_profile_corr
best_shift_angle_profile_corr
best_shift_bins
```

### Nulls

Use at least:

```text
angle-label shuffle:
  shuffle displacement-angle labels within distance band

eye-label permutation:
  permute eye-position labels before building angle profiles

matched-dimension smooth null:
  optional but recommended if topology results are strong
```

---

## Cross-session inference

Do not pool raw response vectors across sessions. Pool summary statistics.

For each metric, report:

```text
per-session effect
per-subject mean effect
grand mean across sessions
sign consistency across sessions
bootstrap over sessions, if enough sessions
```

Save:

```text
stg_cross_session_summary.csv
```

Columns:

```text
metric_name
source
n_sessions
n_subjects
session_effects_json
mean_effect
median_effect
n_positive_sessions
sign_test_p
bootstrap_ci_low
bootstrap_ci_high
interpretation_label
```

If there are too few sessions for formal inference, report descriptive session-level consistency only.

---

## Interpretation labels

Use conservative labels.

```text
raw_only:
  raw RDM similarity is high, but residual geometry is not above null.

residual_shared_geometry:
  residual geometry survives eye-distance and image-similarity controls.

recorded_template_supported:
  recorded residual geometry matches the twin-derived template above null.

direction_topology_supported:
  displacement-direction profiles are shared above matched nulls.

twin_only_prediction:
  twin shows residual/directional geometry, recorded does not resolve it.

not_supported:
  residual and directional metrics do not exceed nulls.

underpowered:
  image support or session count insufficient.
```

---

## Criteria for headline-level result

This analysis becomes headline-level only if the recorded data meet a high bar.

Minimum headline criteria:

```text
1. Recorded residual geometry is positive after expanded eye-geometry nuisance regression.
2. Effect remains after controlling for low-level image similarity.
3. Effect is consistent across multiple sessions and both animals, or at least not driven by one session.
4. Recorded residual geometry aligns with the twin-derived shared transformation template above null.
```

Stronger headline criteria:

```text
5. Direction/topology preservation survives matched nulls.
```

If criteria 1-4 pass:

> V1 reafferent covariance contains a shared retinal-transformation geometry across image contents.

If criteria 1-5 pass:

> FEM-linked shared variability reveals a conserved retinal-translation geometry in foveal V1.

If only the twin passes:

> The deterministic twin predicts shared transformation geometry, but the recorded data do not resolve it at current support.

If only raw RDM similarity passes:

> The shared covariance backbone is best explained by generic displacement smoothness, not a nontrivial transformation geometry.

---

## Recommended execution order

1. Support census across Allen and Logan sessions.
2. Twin-only STG on high-support images to define expected geometry.
3. Recorded-only residual geometry per session.
4. Image-similarity control.
5. Twin-template match in recorded data.
6. Direction/topology analysis only if residual/template results are promising.
7. Cross-session summary.
8. Stop.

---

## Stop rule

This analysis gets one disciplined pass.

Do not branch into:

```text
velocity decoding
nonlinear decoder rescue
flow-field dynamics
behavioral/performance analysis
ideal observers
large hyperparameter search
```

If residual or template effects do not survive controls, report that the shared A3 backbone is mostly generic displacement smoothness or unresolved in recorded data.

If the recorded data are underpowered, report the twin prediction and the recorded limitation. Do not keep changing estimators until a desired result appears.

---

## Manuscript language if successful

Use this only if the recorded criteria pass:

```text
Across images, FEM-induced response geometry shared more than the trivial dependence on displacement magnitude. After controlling for eye-displacement geometry and low-level image similarity, recorded V1 retained a shared residual structure that aligned with the deterministic twin's predicted retinal-translation template. Thus, the shared component of FEM-linked covariance is not simply generic smoothness or global gain. It reflects a conserved retinal-transformation geometry that is expressed through, and warped by, image content.
```

## Manuscript language if mixed

```text
The deterministic twin predicted a shared residual transformation geometry beyond displacement magnitude, but the recorded data provided only limited support at current sample sizes. Thus, the shared cross-image covariance backbone may reflect a conserved retinal-translation structure, but the present recordings do not resolve this component cleanly beyond generic displacement smoothness and common low-level image statistics.
```

## Manuscript language if negative

```text
After controlling for eye-displacement magnitude and low-level image similarity, cross-image response-geometry similarity was not reliably above null. Thus, the shared covariance backbone is best interpreted conservatively as generic smoothness of the response with retinal displacement, plus common low-level image statistics, while image-specific deviations account for the content-dependent component observed in A3.
```
