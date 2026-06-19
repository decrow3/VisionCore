# BackImage Trajectory-Table Observer Plan

Last curated: 2026-06-19.

## Goal

Test whether natural-image structure fixes the failure mode exposed by the
Vernier trajectory-table observer.

The Vernier result is now interpretable: known-eye responses contain the task
information, but pose-free marginalization over a finite trajectory catalog
washes the evidence out because the trajectory posterior stays nearly uniform.
That makes Vernier a good coordinate-frame diagnostic, but probably a poor
stimulus for a Wu-style joint observer.

This BackImage branch asks the more appropriate natural-image question:

```text
Given a V1-twin response to a natural patch under unknown fixational motion,
can the observer identify the image patch while marginalizing trajectory as a
nuisance variable?
```

The core likelihood is:

```text
log p(y | I)
  = log sum_tau p(y | I, tau) p(tau)
```

The desired latent is image identity, local feature identity, or image-feature
class. The trajectory is nuisance state.

## Current Analysis Status

The strongest current result is not an axis-direction claim by itself. It is
that exact trajectory marginalization gives a robust pose-uncertainty rescue:
joint-eye image identity and posterior-weighted feature recovery beat the
zero-eye observer across the clean shared-source axis-conditioned hard-negative
scale sweep.

The completed hard-negative `n128/c4/k16` scale sweep covers `0.5x`, `1.0x`,
and `2.0x` motion scales for `axis_edge_parallel` and
`axis_edge_orthogonal`, with shared source catalogs for every paired axis
comparison. At likelihood scale `1.0`, joint-eye image identity remains far
above zero-eye at every motion scale. The feature-posterior posthoc on the same
cache also shows strongly positive joint-minus-zero Gabor/pyramid recovery for
all axis/scale/feature/k rows.

Axis direction is still claim-bounded. Feature recovery trends edge-parallel at
`0.5x` and `1.0x`, trends edge-orthogonal at `2.0x`, and the paired
parallel-minus-orthogonal feature contrasts are not individually significant.
This should be read as a scale-dependent reconciliation problem rather than a
failed result: orthogonal motion can be a strong hard-negative discriminator,
while along-contour motion may still be better for feature-preserving recovery
in some natural-scale settings.

Compact tangent geometry is now a central mechanism question. The n128
hard-negative compact-mechanism posthoc is complete for the image-identity
observer: compact-only preserves much of the exact joint-eye rescue, and
compact removal collapses it. The fully unified compact-by-feature-posterior
analysis is not wired yet, but it should also be a cache-only extension: reuse
cached response tensors, apply the same compact-only/compact-removed/log-rate
projections, emit compact-variant score vectors, and feed those into the
feature-posterior bridge.

## Scope Boundary: Exact Tables, Not Compact Geometry

This BackImage observer does not use the compact translation geometry from the
Vernier/local-chart branch. It uses exact cached V1-twin responses for each
finite image candidate and each finite trajectory candidate:

```text
lambda_counts[I_i, tau_k, time, unit]
```

Then it marginalizes over the trajectory catalog:

```text
log p(y | I_i) = log sum_k p(y | I_i, tau_k) p(tau_k)
```

The image candidates are hypotheses in a finite choice set, with an implicit
uniform candidate-image prior unless explicitly reweighted. The named
`empirical`, `OU`, or matched-null prior is the nuisance trajectory prior
`p(tau_k)`, not a learned natural-image prior.

This means the current branch is:

```text
finite natural-image candidates
+ finite trajectory catalog
+ exact full-population twin response table
+ trajectory marginalization
```

It is not:

```text
compact translation basis
instantaneous local tangent chart
lag-linear geometry approximation
continuous natural-image prior
full Wu-style image reconstruction
```

The current result validates the trajectory-marginalized decision logic for
natural-image candidates. The compact-mechanism posthoc now asks whether a
low-dimensional or lagged translation approximation can reproduce the same
joint-eye rescue without evaluating the full exact response table.

## Compact-Mechanism Post-Hoc Analysis

The compact-mechanism analysis is a cache-only diagnostic for the exact
BackImage response tables. It does not rerun the V1 twin and it does not replace
the primary exact-table observer. Instead, it asks whether the motion-induced
response component that drove the exact joint-eye rescue is carried by a compact
translation subspace.

For each cached table:

```text
lambda_full[I, tau, t, u]
lambda_zero[I, t, u]
Delta[I, tau, t, u] = lambda_full - lambda_zero
```

the analyzer projects `Delta` into a supplied unit-space basis `U` and scores
the same known-eye, zero-eye, and joint-eye observers under these response
variants:

```text
full_exact:
  lambda_full

zero_static:
  lambda_zero

compact_only:
  lambda_zero + P_U Delta

compact_removed:
  lambda_zero + (I - P_U) Delta

log_compact_only:
  lambda_zero * exp(P_U log(lambda_full / lambda_zero))

log_compact_removed:
  lambda_zero * exp((I - P_U) log(lambda_full / lambda_zero))

random_k:
  compact_only with matched random subspaces

unit_shuffle_compact:
  compact_only after permuting unit identities in U

gain_only:
  compact_only with the all-ones gain axis

static_pc_k:
  compact_only with PCs of stabilized/static responses
```

The first implemented analyzer is:

```text
declan/backimage_trajectory_observer/analyze_compact_mechanism.py
```

Supporting follow-up utilities:

```text
declan/backimage_trajectory_observer/build_image_disjoint_compact_basis.py
declan/backimage_trajectory_observer/summarize_compact_mechanism_followups.py
```

It writes:

```text
compact_mechanism_trials.csv
compact_mechanism_summary.csv
compact_mechanism_by_variant.csv
compact_mechanism_random_null_summary.csv
compact_mechanism_posterior_summary.csv
compact_mechanism_rate_clipping_audit.csv
compact_mechanism_reconstruction_checks.csv
compact_mechanism_report.md
compact_mechanism_run_metadata.json
```

The basis must match the response-table unit count and is orthonormality-audited
before use. Projected rates are clipped only at likelihood-scoring time, and the
pre-clipping negative/clipped-rate audit is recorded. A global basis smoke is
useful for debugging, but claim-level compact-geometry results should use an
image-disjoint basis or be explicitly labeled as `basis_mode=global`.

`basis_mode=image_disjoint` is validated rather than treated as a cosmetic
label. The analyzer requires the basis file to declare image-disjoint provenance
through metadata such as `image_disjoint=True` or a provenance string containing
`image_disjoint`. If provenance was verified outside the file, the override flag
`--allow-unverified-image-disjoint-basis` must be supplied explicitly.

For current response caches, nearest-trajectory distance is recovered from the
response cache if present, otherwise from `response_cache_manifest.csv`, and
finally from `observer_trials.csv` for older runs. Future response-cache files
written by `run_backimage_trajectory_table_observer.py` include
`nearest_trajectory_distance` in both the `.npz` table and manifest.

### Compact-Mechanism Promotion Gates

The compact-mechanism result should not be promoted from diagnostic to a strong
mechanistic claim until the following checks are satisfied, with
`matched_static_response` as the primary candidate-set mode:

```text
compact_only > random_k
compact_only > unit_shuffle_compact
compact_only > gain_only
compact_only >= or > static_pc_k
compact_removed loses the exact-table rescue
compact-only clipping remains low
compact-removed loss is not explained by invalid projected negative rates
the qualitative pattern survives an image-disjoint compact basis
```

The first full image-disjoint run passes the central qualitative gate:
compact-only preserves much of the exact-table true-score rescue and beats
random, unit-shuffle, and gain controls. Static-response PCs remain the serious
open control because they recover substantial rescue too. The image-disjoint
result should therefore be read as compact-geometry sufficiency above
random/unit-shuffle/gain controls, not yet as unique superiority over every
generic low-dimensional static-response subspace.

The n128 shared-source hard-negative scale sweep extends that conclusion to the
axis-conditioned cache. The image-disjoint compact basis again carries most of
the full exact true-score rescue, compact removal collapses the rescue, and
`log_compact_removed` shows the same qualitative loss without negative-rate
clipping. At likelihood scale `1.0`, `compact_only` preserves roughly
`46-82%` of the full-minus-zero accuracy rescue depending on axis, scale, and
`k`, and roughly `0.84-0.90` of the median true-score rescue for the strongest
compact settings. `compact_removed` usually falls to zero-static or worse.

Static-response PCs remain a fair but subtle control. They are fair because
they ask whether any low-dimensional unit-space basis drawn from the static
response ensemble can carry the same observer rescue. They are subtle because
they are not designed to exclude compact translation geometry: if the compact
translation directions align with high-variance stabilized/static response
axes, static PCs can inherit part of the compact geometry. Strong static-PC
performance therefore weakens a uniqueness claim, but it does not by itself
falsify compact geometry as the mechanism. The stronger next control is an
overlap and residualization analysis:

```text
compact vs static-PC subspace overlap
compact-only after removing static-PC overlap
static-PC-only after removing compact overlap
feature-posterior compact-only / compact-removed scoring
```

`log_compact_only` and `log_compact_removed` provide that clipping-safe
companion diagnostic. They project the log-rate ratio
`log(lambda_full / lambda_zero)` and exponentiate back to rates, so the
reconstructed response tables are positive by construction. These log-rate
variants are not the same additive delta decomposition as `compact_only` and
`compact_removed`; they are a robustness check for whether the necessity
conclusion survives without projected negative rates. In the first
image-disjoint run, `log_compact_removed` has zero clipping and removes most of
the true-score rescue, reducing the concern that the linear compact-removal
failure was only a negative-rate artifact.

The strategic bridge being tested is:

```text
compact FEM geometry
  -> carries motion-dependent likelihood structure
  -> supports trajectory-marginalized natural-image inference
```

For the current axis-conditioned feature-posterior branch, compact geometry has
two statuses:

```text
image-identity compact mechanism:
  possible now as a cache-only posthoc on the response tables

feature-posterior compact mechanism:
  requires a small analyzer extension, but not a new V1 forward rerun
```

## Relationship To Existing BackImage Aggregate Work

This is a sibling analysis, not a replacement for the current BackImage
aggregate FEM-information branch.

Existing aggregate branch:

```text
Does a motion distribution improve decodability of natural-image features from
V1-twin response summaries?
```

New trajectory-observer branch:

```text
Does natural-image structure let a likelihood observer localize nuisance eye
trajectory enough for pose-free image/feature inference?
```

The new branch should reuse the same BackImage window manifests, patch
extraction code, canonical twin scorer, and motion-family bookkeeping where
possible, but its primary metrics are likelihood decisions and trajectory
posterior diagnostics rather than ridge-decoding scores.

Detailed implementation prescription:

```text
declan/backimage_trajectory_observer/backimage_trajectory_table_observer_prescription.md
```

Implementation pre-mortem:

```text
declan/backimage_trajectory_observer/implementation_premortem.md
```

Results log:

```text
declan/backimage_trajectory_observer/results_log.md
```

## Primary Inputs

Use the reviewed BackImage window table as the first manifest:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_image_structure_reviewed_v2_screenfiltered_yfix/
    backimage_image_fem_windows.csv
```

Reuse utilities from:

```text
declan/fixation_statistics_by_stimulus/run_backimage_aggregate_fem_information.py
declan/fixation_statistics_by_stimulus/run_backimage_latent_information_screen.py
declan/vernier_active_sensing/trajectory_table_observer.py
```

The first implementation should create a new runner rather than bending the
aggregate decoder runner:

```text
declan/fixation_statistics_by_stimulus/run_backimage_trajectory_table_observer.py
declan/backimage_trajectory_observer/
```

## Reuse Existing Infrastructure, Not Aggregate Likelihood Caches

This branch should heavily reuse existing BackImage infrastructure:

- window filtering and selected-window provenance;
- BackImage canvas loading, gaze-to-pixel conversion, and patch extraction;
- empirical trace-bank construction and motion-family controls;
- `CanonicalTwinScorer.responses(patch, traces)`;
- response alignment utilities;
- existing static/summary caches for candidate preselection and hard-negative
  controls.

However, existing aggregate response caches should not be used as the primary
observer likelihood table. Files such as `response_summary_arrays.npz` and
`response_feature_arrays.npz` contain compressed summaries like mean responses,
temporal PCA features, or motion-averaged response features. They generally do
not preserve the raw table needed here:

```text
lambda_counts[candidate, trajectory, time, unit]
```

Use existing aggregate caches only for support tasks:

- `analysis_images.csv` or selected-window manifests for provenance;
- `aggregate_motion_metadata.csv` for trajectory/source bookkeeping;
- `response_summary_arrays.npz` or `response_feature_arrays.npz` for
  `matched_static_response` or other hard-negative candidate preselection.

The observer itself needs a new raw response-table cache emitted before the
aggregate runner's summary compression step.

## Observer Definitions

For image patch `I_i`, trajectory `tau_k`, and cached expected-count response:

```text
R_ik = F_twin(I_i, tau_k)
```

For an observed response generated by `(I_true, tau_true)`, score candidate
images under three observers:

```text
known-eye:
  log p(y_obs | I_i, tau_true)

zero-eye:
  log p(y_obs | I_i, tau = 0)

joint-eye:
  log mean_tau p(y_obs | I_i, tau)
```

Image identity is only one endpoint. The implemented feature-posterior bridge
attaches a feature vector to every candidate image and decodes the
posterior-weighted feature estimate:

```text
z_i = phi(I_i)
p_mode(I_i | y_obs) = softmax(score_mode(I_i))
z_hat_mode = sum_i p_mode(I_i | y_obs) z_i
feature_error_mode = ||z_hat_mode - z_true||^2
```

Run this for `known-eye`, `zero-eye`, `joint-eye`, and the `best_single_tau`
diagnostic. The first implemented posthoc runner also writes a `motion_delta`
diagnostic using:

```text
Delta S_i = S_joint(I_i) - S_zero(I_i)
z_hat_delta = sum_i softmax(Delta S_i) z_i
```

`motion_delta` is a candidate-wise log-likelihood-ratio diagnostic, not an
independent generative likelihood. It is useful for isolating the motion-added
score component, but it should not be read as a calibrated standalone observer.

The primary implemented feature targets match the BackImage positive branches:

```text
gabor_local_field
pyramid_local_field
```

The runner is:

```text
declan/backimage_trajectory_observer/analyze_feature_posterior.py
```

It is cache-first: it consumes existing exact response-table runs and does not
rerun the V1 twin unless it needs to compute image feature vectors from patches.
It writes:

```text
feature_posterior_trials.csv
feature_posterior_summary.csv
feature_axis_contrasts.csv
feature_motion_evidence_contrasts.csv
feature_posterior_uncertainty.csv
feature_posterior_qc.csv
feature_posterior_report.md
feature_posterior_metadata.json
```

Guardrails:

- candidate IDs from `candidate_sets.csv` are checked against response-table
  candidate IDs before candidate feature vectors are aligned;
- external feature NPZ files must validate all shared row identities
  (`source_row`, `image_index`) or be passed with an explicit
  `--trust-feature-row-order` override;
- dry-run or cache-skipped manifests fail with a clear no-response-cache error;
- `feature_axis_contrasts.csv` only includes parallel-vs-orthogonal rows whose
  source trajectory catalogs were shared (`axis_shared_source_catalog=True`).

The first completed feature-posterior run is:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_matched_static_feature_posterior_gabor_pyramid_k4_8_v1/
```

It scored the clean shared-source matched-static axis cache:

```text
base run = backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1
n_windows = 64
response tables = 128
candidate_set_mode = matched_static_response
priors = axis_edge_parallel, axis_edge_orthogonal
latents = gabor_local_field, pyramid_local_field
k = 4, 8
likelihood_scale = 1.0
```

The initial `k=4,8` PCA feature sweep was deliberately conservative rather
than definitive. It matched the low-dimensional Gabor/pyramid feature
decomposition branch, kept the posterior-MSE readout interpretable with only
four image candidates per trial, and avoided pushing too hard on a PCA fit over
the selected-window feature rows (`128 x 4608` for Gabor and `128 x 3072` for
pyramid in the n128 scale sweep). The follow-up question is whether the
natural-scale axis interaction survives or sharpens in a richer feature
subspace. A principled next sweep is:

```text
k = 2, 4, 8, 16, 32
```

Interpret the sweep as:

```text
k2/k4:
  coarse dominant feature recovery

k8:
  richer but still compact feature recovery; current most sensitive readout,
  with a small exploratory preference for the natural 1x scale in the
  parallel/axis-by-scale feature-recovery contrast

k16/k32:
  broader feature reconstruction, useful only if the axis-by-scale interaction
  survives without being dominated by noisy or idiosyncratic dimensions
```

Headline readout:

```text
joint feature recovery > zero-eye feature recovery in all 8 summary rows
parallel joint feature recovery > orthogonal joint feature recovery in all 4
  feature/k axis contrasts
```

Mean joint MSE reduction relative to zero-eye:

```text
orthogonal gabor k4:    9.3%
orthogonal gabor k8:   14.0%
orthogonal pyramid k4: 10.1%
orthogonal pyramid k8:  8.7%
parallel gabor k4:     11.2%
parallel gabor k8:     18.4%
parallel pyramid k4:   14.8%
parallel pyramid k8:   14.3%
```

Parallel-minus-orthogonal joint feature recovery:

```text
gabor k4:    mean +1.19, median +0.37, parallel wins 57.8% of trials
gabor k8:    mean +3.43, median +0.91, parallel wins 56.2% of trials
pyramid k4:  mean +1.88, median +0.61, parallel wins 53.1% of trials
pyramid k8:  mean +2.37, median +0.11, parallel wins 51.6% of trials
```

Interpretation:

```text
The exact joint trajectory-table observer carries recoverable image-feature
information beyond zero-eye. In the clean matched-static axis cache, the
edge-parallel prior recovers more Gabor/pyramid feature structure than the
edge-orthogonal prior, even though the effect is heterogeneous across trials.
Known-eye remains well above joint-eye, so this is a partial pose-uncertainty
rescue rather than complete trajectory recovery.
```

The first pass should use the same deterministic expected-count Poisson score
used by the Vernier trajectory-table observer:

```text
lambda_safe = max(lambda_i,k[t,u], eps)
sum_t,u y_obs[t,u] log(lambda_safe) - lambda_safe
```

This is a likelihood-ratio style score for deterministic expected counts, up to
the observation-only log-factorial constant.

Required guardrails:

- use the same observation convention for known-eye, zero-eye, and joint-eye;
- record `eps`, `bin_seconds`, likelihood family, and any response clipping in
  `run_metadata.json`;
- report whether scores are deterministic expected-count scores or sampled
  spike-count likelihoods;
- expose and record `likelihood_scale`;
- apply likelihood scale consistently across known-eye, zero-eye, joint-eye,
  posterior diagnostics, and best-single-trajectory diagnostics;
- add a deviance-style diagnostic later if low-rate units appear to dominate.

Do not use candidate-dependent Gaussian log determinants as the default for
deterministic expected-count observations.

Suggested first likelihood-scale sensitivity grid:

```text
0.25, 0.5, 1.0, 2.0
```

This is a sensitivity parameter, not a final-accuracy tuning knob. Either
predefine the grid and report all values, or calibrate on a separate validation
split.

## Zero/Reference Pose

The primary zero-eye baseline should be:

```text
static image rendered at the same fixation/window patch center with
tau(t) = 0 for all t
```

This means zero-eye uses the same selected BackImage crop/patch location as the
observed trial, but removes within-window retinal motion.

Secondary controls should be explicitly labeled:

- `screen_center_static`: screen-centered or image-centered patch, if available;
- `trace_mean_static`: static image at the mean position of the observed trace;
- `initial_pose_static`: static image at the first valid trace sample;
- `per_candidate_static`: static rendering for each candidate patch center.

The primary joint-vs-zero comparison should use one fixed zero convention, not
mix reference definitions across conditions.

## Primary Endpoint

Start with image identity among a finite set of natural patches:

```text
argmax_i log p(y_obs | I_i)
```

Candidate-set construction matters. A naive all-vs-all image-identity task may
mostly measure trivial image separability rather than whether trajectory
marginalization helps. Every run should record a `candidate_set_mode`.

Candidate-set modes:

- `self_lookup`: exact true patch with trivial distractors; sanity check only.
- `random_global`: random patches from the manifest; useful only as an easy
  sanity check.
- `same_session_region`: candidate patches from the same session and similar
  screen/image region.
- `matched_structure_bins`: candidates matched by contrast, gradient energy,
  orientation coherence, and spatial-frequency power bins.
- `nearby_patch`: candidates from nearby image locations when multiple windows
  are available.
- `hard_negative_structure`: nearest neighbors in low-level image-feature
  space, excluding the true patch.
- `matched_static_response`: hard negatives matched by stabilized twin
  response, mean-rate distance, or covariance-whitened static response distance.

Primary scientific claims should use matched or hard-negative candidate sets.
`self_lookup` and `random_global` are smoke tests, not headline endpoints.

Candidate-set guardrails:

- true patch appears exactly once;
- exact duplicate windows are excluded unless intentionally testing
  `self_lookup`;
- near-duplicate patches or movies are detected and reported;
- all candidate ids are recorded per trial;
- nearest-distractor contrast, mean-rate, and static-response distances are
  recorded when available.

Report:

- `known_eye_accuracy`
- `zero_eye_accuracy`
- `joint_eye_accuracy`
- `joint_minus_zero_accuracy`
- `known_minus_zero_accuracy`
- candidate-rank and margin summaries

Success pattern:

```text
known-eye high
zero-eye lower
joint-eye improves over zero-eye
joint-eye remains below or near known-eye
```

Failure pattern:

```text
known-eye high
joint-eye approximately zero-eye
trajectory posterior N_eff approximately N_catalog
```

## Critical Posterior Diagnostics

The main reason to run this analysis is not only image-identity accuracy. It is
to test whether natural images collapse the nuisance trajectory posterior.

For each observed trial and candidate image, compute:

```text
p(tau | y_obs, I_i)
N_eff = 1 / sum_tau p(tau | y_obs, I_i)^2
nearest_trajectory_rank
nearest_trajectory_distance
true_trajectory_rank
best_single_tau_score
best_single_tau_accuracy
joint_vs_best_single_tau_gap
```

Use nearest-trajectory rank as the primary pose-localization diagnostic. Exact
trajectory rank can be misleading when the catalog contains near-duplicate
paths or when a held-out path has a close surrogate. Nearest trajectory should
be computed in at least one task-relevant metric:

- retinal-image displacement distance over time;
- rendered patch/movie distance if cheap enough;
- response-space distance under the known image;
- raw eye-coordinate distance as a secondary bookkeeping metric.

Interpretation:

- Low `N_eff` plus improved joint accuracy means natural-image structure
  supplies pose constraints.
- High `N_eff` plus known-eye success means the likelihood does not localize
  pose, as in Vernier.
- Strong best-single-trajectory score but weak marginal observer means evidence
  dilution.
- Weak best-single-trajectory score means finite catalog support is inadequate or
  the response/noise model is wrong.

The best-single-trajectory diagnostic is not a plausible observer and is not a
strict oracle upper bound on image-identity accuracy. It is a failure
localization diagnostic that asks whether any single retained trajectory gives a
strong fit for each candidate image:

```text
best_single_tau_score = max_tau log p(y_obs | I, tau)
joint_vs_best_single_tau_gap = best_single_tau_score - joint_score
```

Older result files may use `best_trajectory_oracle_*` and
`joint_vs_best_dilution_gap` for the same quantity. Treat those as legacy names:
because single-trajectory maximization can overfit distractor images, its
accuracy can be lower than the marginalized joint observer.

Interpretation matrix:

```text
known high, best single-tau high, joint low, N_eff high:
  useful trajectory-specific evidence exists, but marginalization dilutes it.

known high, best single-tau low, joint low:
  trajectory support or likelihood model is inadequate.

known high, joint high, N_eff low, nearest trajectory rank good:
  natural-image structure supports useful pose marginalization.

known high, joint high, N_eff low, nearest trajectory rank poor:
  posterior concentrates, but on the wrong motion; inspect gain/image confounds.
```

## Motion Families

Match the aggregate FEM branch as closely as practical:

1. `static`
2. `empirical`
3. `scaled_empirical`: initially `0.25x`, `0.5x`, `1x`
4. `ou_matched`
5. `brownian_matched`
6. `rotated_empirical`
7. Optional later: time-shuffled increments or phase-randomized empirical

Every row must record:

- nominal scale
- effective RMS
- path length
- speed summary
- autocorrelation summary
- clipping fraction
- trajectory source provenance

## Image-Structure Stratification

Stratify posterior localization and image-identity accuracy by existing
BackImage columns:

- `image_patch_rms_contrast`
- `image_gradient_energy`
- `image_edge_density`
- `image_orientation_coherence`
- `image_spectrum_anisotropy`
- `image_high_freq_power_fraction`
- `image_power_0_2_cpd_fraction`
- `image_power_2_4_cpd_fraction`
- `image_power_4_8_cpd_fraction`
- `image_power_8plus_cpd_fraction`

Treat these raw image metrics as exploratory stratification variables, not as
the final definition of pose-identifiability. A patch can have high contrast
but be repetitive or aperture-limited, while a moderate-contrast patch can have
highly diagnostic local structure.

Add likelihood-derived structure metrics as primary mechanistic diagnostics:

- trajectory identifiability for the true image: how separable are retained
  trajectory likelihoods under the true candidate?
- image identifiability under motion: how separable is the true image from
  distractors after trajectory marginalization?
- static-response nearest-distractor distance under stabilized twin responses;
- motion-induced response diversity across trajectories;
- posterior entropy or `N_eff / K` under the true image.

The preferred mechanistic signature is:

```text
lower posterior trajectory entropy -> larger joint advantage over zero-eye
```

Raw structure metrics should help generate hypotheses about which patches are
pose-identifiable, but likelihood-derived metrics should carry the main
interpretation.

## Staged Runs

### Stage 0: Cache/Manifest Dry Run

Purpose: verify patch extraction, trajectory generation, and tensor shapes
without expensive inference.

Suggested size:

```text
n_images = 8
k_trajectories = 4
families = empirical, ou_matched
scales = 0.5x
device = cpu or cuda smoke
```

Expected outputs:

- selected window manifest
- generated motion metadata
- cache manifest
- no scientific claim

### Stage 1: Tiny Twin Smoke

Purpose: validate exact response cache and observer logic.

Suggested size:

```text
n_images = 16
k_trajectories = 4
families = static, empirical, ou_matched
scales = 0.5x
```

Required checks:

- known-eye should exceed zero-eye unless the candidate set is too hard or
  noise model is broken.
- self-lookup sanity rows should classify correctly.
- leave-one-trajectory-out rows should stay interpretable.
- leave-one-image-out candidate-set rows should not leak the true patch through
  duplicate entries.
- posterior metrics should be finite.
- static observer should be internally deterministic.

### Stage 2: First Interpretive Pilot

Purpose: answer whether BackImage natural patches reduce trajectory posterior
entropy.

Suggested size:

```text
n_images = 32 or 64
k_trajectories = 8 or 16
families = static, empirical, ou_matched
scales = 0.5x, 1x, 2.0x
candidate_set_modes = random_global, matched_structure_bins, hard_negative_structure
```

Primary readout:

```text
known-eye > zero-eye
joint-eye > zero-eye
median N_eff / N_catalog decreases for joint candidate images
nearest trajectory rank improves over chance
```

Keep Stage 2 deliberately narrow. If it fails, we should know whether the
failure is in the cache/likelihood/candidate-set/posterior machinery before
adding Brownian, rotated, shuffled, and wider scale sweeps.

Keep the `2.0x` scale as the above-natural sentinel, paired with `0.5x` for a
half/natural/double sweep. Treat it as a check against the trivial interpretation
that more motion is simply better or simply more damaging to zero-eye observers,
and audit effective RMS and clipping.

`random_global` is a positive control only. The primary Stage 2 readout should
come from at least one of:

```text
matched_structure_bins
hard_negative_structure
matched_static_response
```

### Stage 3: Scaled Aggregate Pilot

Only run this if Stage 2 shows posterior localization.

Suggested size:

```text
n_images = 128
k_trajectories = 16 or 32
families = static, empirical, ou_matched, brownian_matched, rotated_empirical
scales = 0.25x, 0.5x, 1x, 2.0x
```

Add:

- grouped bootstrap by image
- stratified summaries by image structure
- leave-one-image-family or leave-session diagnostics if practical

## Output Layout

Use a new output namespace:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_<tag>/
```

Suggested files:

```text
selected_windows.csv
candidate_sets.csv
motion_catalog.csv
response_cache_manifest.json
observer_trials.csv
observer_summary.csv
posterior_summary_by_image.csv
posterior_summary_by_structure_bin.csv
run_metadata.json
analysis_report.md
```

Trial rows should carry separate observation and prior provenance:

```text
observation_condition
observation_family
observation_scale
prior_condition
prior_family
prior_scale
trajectory_prior_mode
```

Required `trajectory_prior_mode` values:

```text
include_self
leave_one_out
```

`include_self` is a table-lookup diagnostic. Interpretive claims should rely on
`leave_one_out` or another non-leaky prior mode.

The runner should support a small cross-prior matrix from the beginning:

```text
observed empirical x prior empirical
observed empirical x prior OU
observed empirical x prior shuffled-position or matched null
observed OU x prior empirical
```

### Response Cache Shape

Recommended per-trial cache arrays:

```text
lambda_counts: [n_candidates, n_trajectories, n_timebins, n_units]
y_obs_counts: [n_timebins, n_units]
candidate_ids: [n_candidates]
trajectory_ids: [n_trajectories]
true_candidate_index: int
true_trajectory_index: int or -1 if held out
zero_trajectory_index: int
trajectory_prior_mode: string
```

Use `nan` for unavailable diagnostics in CSV outputs, not empty strings.

## Likelihood API Sketch

Keep likelihood and observer scoring independent of the runner:

```text
poisson_expected_count_loglik(y_obs, lambda_counts, eps)
logmeanexp(values, axis)
score_known_eye(loglik_table, true_trajectory_index)
score_zero_eye(loglik_table, zero_trajectory_index)
score_joint_eye(loglik_table, log_trajectory_prior=None, temperature=1.0)
rank_of_true(scores, true_candidate_index)
posterior_from_scores(scores, temperature=1.0)
posterior_weighted_feature(scores, candidate_features)
feature_recovery_metrics(z_hat, z_true)
```

Shape convention:

```text
loglik_table: [n_candidates, n_trajectories]
known_score: [n_candidates]
zero_score: [n_candidates]
joint_score: [n_candidates]
candidate_features: [n_candidates, n_features]
z_hat: [n_features]
```

## Proposed Module Layout

Keep this separated from the aggregate BackImage ridge/Fisher analyses:

```text
declan/backimage_trajectory_observer/
  __init__.py
  likelihood.py
  trajectory_catalog.py
  candidate_sets.py
  response_table.py
  observer.py
  diagnostics.py

declan/fixation_statistics_by_stimulus/
  run_backimage_trajectory_table_observer.py
```

The runner can import existing BackImage patch extraction, trajectory
generation, and `CanonicalTwinScorer` utilities, but should not turn the
aggregate decoder runner into this observer.

## Acceptance Criteria For Promotion

Do not promote this to a larger run unless Stage 2 shows at least one of:

- `joint_eye_accuracy` clearly exceeds `zero_eye_accuracy`;
- median `N_eff / N_catalog` drops well below the Vernier near-uniform regime;
- true or nearest trajectory rank is near the top for the true image;
- best-single-trajectory score is high and the joint observer loses only moderately
  to marginalization.

If known-eye is high but `joint ~= zero` and `N_eff ~= N_catalog`, then the
problem is not just Vernier impoverishment. Next suspects would be:

- deterministic expected-count likelihood mismatch;
- finite trajectory catalog support;
- missing continuous trajectory prior;
- response summary too lossy or too high-dimensional;
- model state/history mismatch.

## Immediate Implementation Checklist

1. Add GPL-clean likelihood utilities with Poisson expected-count scoring,
   clipping, stable `logsumexp`/`logmeanexp`, and likelihood-scale provenance.
2. Add candidate-set construction with `self_lookup`, `random_global`,
   `matched_structure_bins`, `hard_negative_structure`, and eventually
   `matched_static_response`.
3. Add trajectory catalog metadata with observation/prior family, scale,
   `include_self`, and `leave_one_out`.
4. Add response-table cache writer with `[candidate, trajectory, time, unit]`
   shape and full metadata.
5. Add known-eye, zero-eye, joint-eye, and best-single-trajectory scoring.
6. Add posterior diagnostics: `N_eff`, entropy, max posterior mass,
   nearest-trajectory rank, and dilution gap.
7. Add trial-level and summary CSV writers with separate observation/prior
   provenance fields.
8. Run Stage 0 dry run.
9. Run Stage 1 tiny twin smoke on an idle GPU.
10. Review known/zero/joint/posterior metrics before any expensive run.

## Code-Review Checklist

Reviewers should specifically check:

- known-eye, zero-eye, and joint-eye use the same `y_obs`, candidates, units,
  binning, likelihood, `eps`, and likelihood scale;
- zero-eye uses the fixed primary zero convention unless explicitly labeled;
- joint-eye is a true log-sum/log-mean over trajectories, not a best trajectory;
- trajectory prior weights are included exactly once;
- observation and prior conditions are recorded separately;
- `include_self` and `leave_one_out` are not conflated;
- response tables are indexed `[candidate, trajectory, time, unit]`
  consistently;
- the true image appears exactly once in every candidate set;
- duplicate and near-duplicate patches are reported;
- posterior concentration is checked near the true or nearest trajectory, not
  merely somewhere in the catalog;
- Stage 1 and Stage 2 outputs are deterministic under a fixed seed.

## Claim Boundary

This branch is a V1-twin likelihood diagnostic. It should be described as:

```text
Natural-image structure can or cannot help a V1-twin likelihood observer
marginalize unknown fixational motion.
```

It should not yet be described as full image reconstruction, literal mutual
information, or a claim about biological optimality.

Implementation note: this plan was written independently from the provided
specification; no GPL-covered source code was copied.
