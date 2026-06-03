# Handoff: Shared Transformation Geometry (STG)

## Working name

**Shared Transformation Geometry (STG)**

Do **not** call this analysis “A3b” in code, filenames, figure titles, or summaries. It is conceptually downstream of A3, but it asks a broader and more ambitious question:

> Does FEM-induced population structure contain a shared retinal-transformation geometry across image contents, beyond trivial displacement magnitude and beyond image similarity?

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

The main paper result is that a substantial component of foveal V1 shared variability during fixation is reafferent: it is attributable to measured eye-position-dependent modulation of the visual response. Current twin and recorded analyses support a structured covariance story:

- FEM covariance is signal-aligned.
- FEM covariance aligns with retinal-translation tangent directions.
- FEM covariance is low-dimensional at adequate sampling, though its rank is better explained by finite sampling of a curved response manifold than by literal eye-motion degrees of freedom.
- FEM covariance is shaped by eye-position occupancy.
- In high-support fixRSVP images, both recorded V1 and the deterministic twin show a modest content-specific component, with within-image covariance-subspace overlap exceeding cross-image overlap.
- Cross-image overlap remains high, implying a large shared covariance backbone beneath the image-specific increment.

STG asks what that shared backbone is.

The strong version of the question is:

> Does recorded V1 contain a conserved retinal-transformation geometry across image contents, expressed through FEM-linked covariance, after controlling for trivial eye-displacement magnitude and low-level image similarity?

This is a structural question, not a performance question. It does not ask whether FEMs improve coding, whether velocity is decoded, whether downstream circuits use the signal, or whether eye movements are optimized. It asks whether the covariance geometry induced by measured retinal translation preserves a shared transformation structure across images.

If the result works in recorded V1 across sessions and animals, it could become a headline-level synthesis:

> Foveal V1 does not separate content and transformation into independent channels. Instead, self-generated retinal translations impose a shared covariance geometry that is expressed through, and warped by, image content.

---

## Core guardrails

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
3. eye-label and random-map nulls
4. matched dimensionality / eigenspectrum nulls for directional or topology tests
5. session-level replication
```

---

## Key methodological correction

The first draft put too much weight on a symmetric response-RDM residual. That is useful, but it should **not** be the primary signed transformation test.

An RDM is a **representational dissimilarity matrix**:

```text
D_I(a,b) = distance(r_I(e_a), r_I(e_b))
```

It is symmetric:

```text
D_I(a,b) = D_I(b,a)
```

Therefore, it cannot distinguish a displacement from its reverse. Eastward and westward displacements of the same magnitude produce the same RDM entry if they produce the same response-distance magnitude. An RDM can test conserved magnitude, anisotropy, or undirected orientation structure, but it cannot by itself establish a signed transformation coordinate.

The direct signed test should be a **cross-image tangent-map comparison**:

```text
retinal displacement -> signed response-change vector
```

This is a cross-image extension of A4. It tests whether the signed map from x/y retinal displacement to population-response change is conserved across images.

Thus, STG has two complementary components:

```text
Primary:
  cross-image tangent-map alignment
  recorded-to-twin tangent-template match

Secondary:
  residual RDM geometry after removing eye-distance magnitude and image similarity
  undirected anisotropy/topology of response distances
```

---

# Stage 0: support census across sessions

Before running the analysis, report available data across all candidate sessions.

Include all available Allen and Logan fixRSVP sessions.

For each session, write:

```text
subject
date
session_id
source availability: recorded, twin, or both
validated_twin_available: true/false
n_units
n_unique_fixRSVP_images
image support distribution
n_images >= 80 samples
n_images >= 160 samples
n_images >= 320 samples
n_images >= 640 samples, if available
eye trace availability
twin response cache status
```

Save:

```text
stg_support_census.csv
stg_support_census_summary.json
```

Do not proceed to pooled analysis until this census is written.

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

## Multi-session design and the data-limit problem

More sessions help, but only if the analysis is designed correctly.

### What multiple sessions cannot do

Multiple sessions cannot make any single image’s response geometry better estimated within a session. Recorded populations are different cells with different tuning, and the twin is a session-specific model of those cells. Do not pool raw neurons across sessions.

If every session is below the per-image support floor, stacking sessions gives many noisy per-image estimates, not one good estimate.

The full-RDM residual is the statistic least helped by more sessions because it requires a reliable per-image distance matrix.

### What multiple sessions can do

Power comes from the number of independent within-session scalar effects that can be averaged:

```text
per-image template-match scores
per-image-pair tangent alignments
per-session mean residual correlations
per-session mean tangent-template matches
```

Multiple sessions also provide replication across animals.

### Design implication

Choose within-session statistics that are stable at low support and aggregate those across sessions.

The two most important statistics are:

```text
1. twin-template match
2. cross-image tangent-map alignment
```

Both reduce each session to low-dimensional, support-robust numbers.

The full-RDM residual should be reported, but should not be the primary multi-session claim.

### Control before aggregation

Aggregating across sessions amplifies shared confounds as well as signal. Therefore, apply all controls **inside each session** before aggregation:

```text
eye-displacement regression
image-similarity control
eye-shuffle null
matched-random map null
```

Never aggregate raw correlations and then control afterward.

### Inference unit

Session is the random effect.

Report:

```text
per-session effects
per-subject means
sign consistency across sessions
session-level bootstrap or sign test
```

Do not treat image pairs within a session as independent. They share images.

With few sessions, a sign test over sessions plus descriptive consistency is the honest ceiling.

---

# Stage 1: primary signed tangent-map analysis

## Goal

Test whether the signed map from retinal displacement to population response change is conserved across images.

This is the primary STG analysis because it tests the signed transformation coordinate directly and is more feasible at low support than a full pairwise RDM residual.

## Per-image tangent-map fit

For each image \(I\), session, and source, regress the population response on centered eye displacement.

For each unit \(u\):

```text
r_{I,u}(e) ≈ a_u + b^x_{I,u} * dx + b^y_{I,u} * dy
```

where:

```text
dx = x - mean(x) within the sampled eye cloud
dy = y - mean(y) within the sampled eye cloud
```

Define:

```text
J_I = [b^x_I | b^y_I]
```

where `J_I` has shape:

```text
n_units x 2
```

The two columns are the signed response-space directions driven by x- and y-displacement.

## Outputs per image

Save:

```text
stg_tangent_maps.pkl
stg_tangent_map_image_metrics.csv
```

Per-image metrics:

```text
session_id
subject
date
source
image_set
image_id
n_samples
n_units
r2_mean
r2_median
norm_bx
norm_by
angle_between_bx_by
condition_number_J
```

Use ridge regression if needed for stability. Record the ridge parameter.

## Within-session cross-image tangent alignment

Only compare tangent maps directly **within session**, because neurons are shared only within session.

For image pair \(I,J\):

### Signed column alignment

```text
cos_bx = cos(bx_I, bx_J)
cos_by = cos(by_I, by_J)
```

High signed column alignment means:

```text
x-displacement drives the same neurons in the same signed direction across images
y-displacement drives the same neurons in the same signed direction across images
```

This is the strongest signed transformation-map evidence.

### Subspace alignment

Compute principal angles between:

```text
col(J_I)
col(J_J)
```

Report:

```text
subspace_overlap_k2 = mean(cos^2 principal_angles)
```

This is weaker because it ignores signed axis identity and within-plane rotations.

Save:

```text
stg_tangent_map_alignment.csv
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
cos_bx
cos_by
mean_signed_column_alignment
subspace_overlap_k2
null_eyeshuffle_mean
null_eyeshuffle_ci_low
null_eyeshuffle_ci_high
null_random_mean
null_random_ci_low
null_random_ci_high
exceeds_eyeshuffle_null
exceeds_random_null
image_similarity
image_similarity_metric
```

## Required nulls

### Eye-label shuffle

For each image, permute `(dx, dy)` labels before fitting `J_I`. This kills the displacement-to-response map.

### Matched-random map

Generate random `n_units x 2` maps matched for:

```text
column norms
optional column covariance/eigenstructure
```

Any two 2D subspaces in a high-dimensional space can have nonzero overlap, so random-map nulls are mandatory.

Real cross-image tangent alignment must exceed both nulls.

## Twin-template version

Use the twin to define the predicted shared tangent map, then ask whether recorded V1 expresses that map above null.

For each session:

1. Fit twin tangent maps `J_I_twin`.
2. Build a template:

```text
J_twin_template = mean over images of sign-aligned J_I_twin
```

Sign alignment must be explicit. If using columns bx/by, signs should be defined by positive dx and positive dy in eye coordinates.

3. For each recorded image:

```text
cos_to_twin_bx = cos(bx_rec_I, bx_template)
cos_to_twin_by = cos(by_rec_I, by_template)
subspace_overlap_to_twin = overlap(col(J_rec_I), col(J_twin_template))
```

Save:

```text
stg_tangent_template_match.csv
```

Columns:

```text
session_id
subject
date
source
image_id
n_samples
n_units
cos_to_twin_bx
cos_to_twin_by
mean_signed_template_match
subspace_overlap_to_twin
null_eyeshuffle_mean
null_random_mean
exceeds_null
```

Aggregate:

```text
stg_tangent_summary.csv
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
mean_cos_bx
mean_cos_by
mean_signed_column_alignment
mean_subspace_overlap_k2
bootstrap_ci_low_signed_alignment
bootstrap_ci_high_signed_alignment
bootstrap_p_signed_alignment_le_null
interpretation_label
```

---

# Stage 2: residual RDM geometry

## Goal

Test whether cross-image response geometry remains shared after removing trivial displacement magnitude.

This is secondary to the tangent-map test because RDMs are symmetric and cannot carry signed direction. However, RDMs remain useful for testing shared smoothness, anisotropy, and residual warping.

## Response representation

Use the response representation closest to the existing A3 covariance audit.

Allowed:

```text
full_response
fem_pca_k2
fem_pca_k3
```

Start with `full_response`.

For each image:

```text
D_I(a,b) = distance(r_I(e_a), r_I(e_b))
```

Run at least:

```text
euclidean_zscored
correlation_distance
```

Optionally:

```text
euclidean_centered
```

## Stage 2A: raw smoothness baseline

For each image, compute:

```text
corr(vec(D_I), radial_distance)
corr(vec(D_I), abs_dx)
corr(vec(D_I), abs_dy)
```

Save:

```text
stg_image_smoothness_metrics.csv
```

Raw smoothness is descriptive only. Do not use it as evidence of nontrivial shared transformation geometry.

## Stage 2B: residual RDM after displacement regression

Run two residual models.

### Model 1: radial-only residual

```text
vec(D_I) = beta_0 + beta_1 * radial + epsilon_I
```

Interpretation:

```text
shared geometry beyond displacement magnitude
```

This retains x/y anisotropy and is closer to the conceptual claim.

### Model 2: expanded eye-geometry residual

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

Interpretation:

```text
shared geometry beyond any smooth displacement function in this basis
```

This is conservative. A null under the expanded model does **not** rule out a shared transformation geometry because the conserved transformation mapping may live in those displacement terms.

After residualization, compute cross-image residual similarity:

```text
corr(epsilon_I, epsilon_J)
```

Save:

```text
stg_cross_image_residual_rdm.csv
stg_session_rdm_summary.csv
```

Important interpretation:

```text
radial-only positive:
  shared geometry beyond magnitude

expanded positive:
  shared residual warping beyond the displacement basis

expanded null:
  no evidence for structure beyond smooth displacement function,
  but not a rejection of transformation geometry
```

---

# Stage 3: image-similarity controls

A positive residual or tangent result may reflect natural images sharing low-level statistics rather than V1 carrying a content-invariant transformation geometry.

Compute image-pair similarity metrics:

```text
pixel_correlation after common crop/resize
RMS contrast difference
Fourier amplitude spectrum similarity
orientation/SF energy similarity, preferred if available
Gabor/filter-bank energy similarity, optional
CNN early-layer feature similarity, optional
```

Minimum required:

```text
pixel_correlation
RMS contrast difference
Fourier amplitude spectrum similarity OR orientation/SF energy similarity
```

Controls must be applied inside session before aggregation.

For tangent-map and RDM pairwise metrics, run:

```text
metric_IJ = alpha_0 + alpha_1 * image_similarity_IJ + residual_IJ
```

Also run stratified analysis:

```text
low-image-similarity pairs
high-image-similarity pairs
```

The strong result is positive tangent alignment or residual RDM similarity in low-image-similarity pairs.

Save:

```text
stg_image_similarity_controls.csv
```

---

# Stage 4: undirected anisotropy/topology from RDMs

This is secondary.

Because RDMs are symmetric, use undirected orientation bins:

```text
theta modulo pi
4 bins over [0, pi)
```

Do not use signed 8-bin direction over `[-pi, pi)` as a signed direction claim.

To avoid magnitude confounds, restrict to a distance band:

```text
primary: radial percentile 40-60
secondary: 25-50 and 50-75
```

For each image:

```text
M_I[orientation_bin] = mean residual_D_I(a,b) for pairs in bin
```

Normalize within image:

```text
zscore(M_I)
```

Compute cross-image similarity and nulls.

Save:

```text
stg_undirected_anisotropy.csv
stg_undirected_anisotropy_summary.csv
```

Interpretation:

```text
anisotropy_supported:
  displacement axes differ in response strength in a conserved way across images

not:
  no conserved undirected anisotropy beyond null
```

This supports but does not replace the signed tangent-map test.

---

# Stage 5: cross-session inference

Do not pool raw response vectors across sessions.

Pool summary statistics:

```text
per-session tangent alignment
per-session recorded-to-twin template match
per-session radial-only residual RDM similarity
per-session expanded residual RDM similarity
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

Report separately:

```text
Allen sessions
Logan sessions
all sessions
```

If there are too few sessions for formal inference, report descriptive consistency only.

---

# Revised execution order

1. Stage 0 support census across all Allen and Logan sessions, including twin-availability column.
2. Twin-only tangent maps and residual RDMs on high-support images, per session.
3. Recorded tangent maps per session.
4. Within-session controls: eye-shuffle, random-map nulls, image-similarity controls.
5. Recorded-to-twin tangent-template match.
6. Residual RDM analyses as corroboration.
7. Undirected anisotropy/topology only if primary tests are promising.
8. Cross-session summary.
9. Stop.

---

# Interpretation labels

Use conservative labels:

```text
tangent_template_supported:
  recorded tangent maps align to twin-derived template above null

tangent_shared_geometry:
  recorded cross-image tangent maps align above eye-shuffle and random-map nulls

residual_shared_geometry:
  residual RDM geometry survives radial-only or expanded controls

anisotropy_supported:
  undirected displacement-axis profile is shared above null

raw_only:
  raw RDM similarity is high, but residual/tangent tests fail

twin_only_prediction:
  twin shows shared tangent/residual geometry, recorded does not resolve it

not_supported:
  residual and tangent metrics do not exceed nulls

underpowered:
  image support or session count insufficient
```

---

# Criteria for headline-level result

This analysis becomes headline-level only if the recorded data meet a high bar.

Minimum headline criteria:

```text
1. Recorded cross-image tangent-map alignment exceeds eye-shuffle and random-map nulls.
2. Recorded-to-twin tangent-template match exceeds nulls.
3. The effect survives low-level image-similarity controls.
4. The effect is sign-consistent across multiple sessions and both animals, or at least not driven by one session.
```

Stronger headline criterion:

```text
5. Undirected anisotropy/topology or residual RDM structure also survives matched nulls.
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

# Stop rule

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

If the recorded data are underpowered, report the twin prediction and recorded limitation. Do not keep changing estimators until a desired result appears.

---

# Manuscript language templates

## If successful

```text
Across images and sessions, the signed map from retinal displacement to V1 population response was conserved beyond null expectations and beyond low-level image similarity. Recorded tangent maps aligned with the deterministic twin's predicted retinal-translation template, indicating that the shared component of FEM-linked covariance is not simply generic smoothness or global gain. It reflects a conserved retinal-transformation geometry that is expressed through, and warped by, image content.
```

## If mixed

```text
The deterministic twin predicted a shared retinal-translation tangent map across image contents, but the recorded data provided only limited support at current sample sizes. Thus, the shared cross-image covariance backbone may reflect a conserved transformation structure, but the present recordings do not resolve this component cleanly beyond generic displacement smoothness and common low-level image statistics.
```

## If negative

```text
After controlling for eye-displacement magnitude and low-level image similarity, recorded cross-image tangent alignment and residual response-geometry similarity were not reliably above null. Thus, the shared covariance backbone is best interpreted conservatively as generic smoothness of the response with retinal displacement, plus common low-level image statistics, while image-specific deviations account for the content-dependent component observed in A3.
```

---

# Final one-line summary

Lead with the cross-image tangent-map comparison because it is signed, A4-consistent, first-order, and more feasible in recorded data. Use residual RDM and undirected anisotropy analyses as corroboration. Aggregate controlled within-session scalar effects across Allen and Logan sessions. Do not let raw RDM similarity become the headline.
