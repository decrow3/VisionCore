# BackImage Aggregate FEM Information Plan

Last curated: 2026-06-16.

## Goal

Test a broader active-sensing hypothesis than the per-fixation local `I_z`
axis-matching screen:

```text
Across natural-image samples I ~ p(I), does the empirical FEM distribution
q_real(tau) produce a better V1-twin representation of image information than
matched non-biological motion controls q_control(tau)?
```

The target is not whether fixation `i` uses the exact locally optimal drift
axis for patch `I_i`. The target is whether empirical fixational eye-movement
statistics are useful at the ensemble level for natural-image representation.

## Framing

For image patch `I`, trajectory `tau`, and canonical 756-unit digital-twin
response `R = F_theta(I, tau)`, define an aggregate utility:

```text
U(q) = I(z(I); R(I, tau)),    I ~ p(I), tau ~ q(tau)
```

where `z(I)` is an external image latent, feature vector, or image identity
surrogate, and `q(tau)` is a motion distribution.

A positive result should not be merely "any motion beats static." The stronger
result is that empirical FEMs either outperform matched synthetic controls at
comparable motion energy, or lie near a useful information/cost frontier.

Language guardrail: deterministic ridge-decoding scores from twin rates are
linear decodability or information proxies, not literal mutual information. The
Vernier branch did not depend on stochastic spike draws, but its headline
Fisher quantities did depend on explicit readout/noise assumptions, especially
the pose-aware diagonal-Poisson observer. This aggregate branch can start from
deterministic rates, but any paper-level "information" statement needs either
proxy wording or an added fixed noise/logdet formulation.

## Why This Branch Exists

The corrected local BackImage `I_z` branch is informative but heterogeneous:
small-scale Gabor/pyramid real-vs-random effects survive in some settings, while
real-vs-edge and `1x` effects remain mixed. That may be the wrong biological
level of description. FEM statistics may be adapted to the ensemble of natural
images rather than to every exact local patch/axis pairing.

This branch is therefore the figure-level candidate if local per-fixation axis
matching remains exploratory.

## Primary Inputs

- Canonical 756-unit V1 digital twin.
- Many BackImage/natural-image patches or crops.
- Empirical centered FEM traces sampled from fixation windows.
- Optional pairing modes:
  - `paired/original`: original trace with original image patch.
  - `unpaired/ensemble`: traces sampled independently from image patches.

The main aggregate hypothesis should use the unpaired ensemble mode; paired mode
is a diagnostic for local policy matching.

## Motion Families

Priority order:

1. `static`
   - `tau(t) = 0`.
2. `empirical`
   - measured FEM traces, centered within fixation windows.
3. `scaled_empirical`
   - scales `0`, `0.125x`, `0.25x`, `0.5x`, `1x`, optionally `1.5x`.
   - Avoid treating `2x` as biological unless explicitly labeled over-large or
     capped.
4. `ou_matched`
   - Ornstein-Uhlenbeck motion matched to RMS, autocorrelation, and confinement.
   - This is the primary synthetic null. If real ties OU but beats Brownian,
     the conclusion is FEM-like confinement/autocorrelation, not exact trace
     specialness.
5. `brownian_matched`
   - Brownian motion matched to effective RMS or diffusion.
   - This is a secondary generic-diffusion null.
6. `shuffled_empirical`
   - at least one of time-shuffled increments, trace-shuffled image pairing,
     rotated traces, or phase-randomized traces with matched temporal spectrum.

Every generated trajectory must record nominal scale, effective RMS, path
length, velocity distribution, autocorrelation, duration, and clipping fraction.
Main comparisons must be matched or stratified by effective motion energy.

## Primary Metrics

### 1. Ensemble Image-Feature Decoding

Decode image latents from response summaries:

```text
z(I) ~ R(I, tau)
```

Candidate latents:

- Gabor or steerable-pyramid features.
- DCT coefficients.
- Spatial-frequency band energies.
- Low/mid/high spatial-frequency grouped features.

Response summaries:

- temporal PCs, primary;
- compact multi-bin response features if dimensionality permits;
- delta from static, secondary;
- mean response over trajectory, diagnostic only.

Primary contrasts:

```text
real - static
real - OU matched
real - Brownian matched
real - shuffled empirical
scaled empirical - real
large motion - real
```

Use shared or fixed ridge regularization for figure-level results. Candidate-
specific regularization is allowed only as a secondary permissive estimate.

### 2. Signal Versus Motion-Nuisance Covariance

For image `i` and motion sample `k`, compute:

```text
R_ik = response_summary(F_theta(I_i, tau_k))
mu_i = E_k[R_ik]
Sigma_signal = Cov_i(mu_i)
Sigma_motion = E_i[Cov_k(R_ik)]
```

Report:

- `trace(Sigma_signal)`;
- `trace(Sigma_motion)`;
- signal/motion ratio;
- signal-motion subspace overlap, promoted as a primary paper-coherent metric;
- top-k signal and motion variance;
- participation ratio;
- optional logdet score using `Sigma_motion + lambda I`,
  `diag(Sigma_motion) + lambda I`, or a Poisson/mean-rate diagonal proxy.

This asks whether a motion family improves image-driven structure more than it
adds motion-induced nuisance variation.

### 3. Information/Cost Frontier

For each motion family and scale, plot representation utility against:

- effective RMS;
- path length;
- velocity;
- motion-nuisance covariance.

Empirical FEMs are interesting if they sit near a Pareto-efficient regime, even
if they do not maximize a single scalar objective.

## Diagnostics and Guardrails

Every nominal-scale summary must also report:

- fraction clipped;
- median and IQR effective RMS;
- effective/target RMS;
- effective motion-energy differences between families.

Run subsampling or bootstraps over image samples, trace samples, and sessions.
Include leave-session-out summaries where feasible.

Do not claim real FEMs are globally optimal unless they beat matched motion
controls and do not simply track motion amplitude. If all motion families
improve similarly and the largest motion always wins, the metric is measuring
generic modulation rather than active sensing.

Claim boundary: this branch is twin-scoped unless later connected to recorded
data. Preferred wording is "empirical FEM statistics improve the V1-twin
representation of natural-image structure," not "empirical FEMs improve foveal
V1 representation."

## Suggested Implementation

New runner:

```text
declan/fixation_statistics_by_stimulus/run_backimage_aggregate_fem_information.py
```

Companion posthoc:

```text
declan/fixation_statistics_by_stimulus/summarize_backimage_aggregate_fem_information.py
```

Output root:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n<...>_<tag>/
```

## Cache-First Path

Before running new twin inference, reuse the caches from the corrected
BackImage branch.

Primary existing cache:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_latent_information_scalesweep_n256_rel0125-2_rand8_delta/
```

Useful files:

- `analysis_windows.csv`: the locked n=256 image/window manifest with session,
  phase, real drift axis, edge axis, coherence, anisotropy, and observed RMS.
- `latent_feature_arrays.npz`: fixed local-field latents,
  `gabor_local_field` `(256, 4608)` and `pyramid_local_field` `(256, 3072)`.
- `response_feature_arrays.npz`: canonical 756-unit
  `pose_blind_delta_mean` response summaries for scales
  `0.125x`, `0.25x`, `0.5x`, `1x`, and `2x`, with static, real, edge,
  edge-orthogonal, and rand8 axis candidates.
- `candidate_motion_metadata.csv`: nominal/effective RMS, path length, clipping,
  duration, response length, and candidate provenance.

Cache-only proxy script:

```text
declan/fixation_statistics_by_stimulus/summarize_backimage_aggregate_cache_proxy.py
```

This script loads the saved latents/responses and computes cheap aggregate
decoding and response-covariance summaries without touching the twin. It is a
debugging and prioritization step, not the full aggregate motion-distribution
test.

Current focused proxy output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_latent_information_scalesweep_n256_rel0125-2_rand8_delta/
    aggregate_cache_proxy_full_postfix_nested/
```

Status: post-fix full cached proxy, all cached scales, nested alpha, rand8
baseline computed as mean decoded random-axis score.

Post-fix cache-proxy readout:

- Therefore the cache proxy supports using cached arrays to debug scoring and
  regularization, but it does not replace the OU/Brownian/unpaired empirical
  aggregate run.
- `random_axis_mean` now means the mean of decoded random-axis scores, not a
  decoder fit to the averaged random-axis response vector.
- Raw random axes are retained internally for covariance diagnostics even when
  the requested contrast name is `random_axis_mean`.
- Motion-versus-static is strongly positive and grows with scale, consistent
  with generic motion-induced feature modulation.
- Real-vs-random specificity is narrow: Gabor `k=4`, `0.25x`
  `+3.480`, CI `[+0.642, +7.030]`; pyramid `k=8`, `0.25x`
  `+2.185`, CI `[+0.617, +4.251]`. Other scales mostly have CIs crossing zero.

Second useful cache:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_twin_stability_metric_audit/
```

Useful files:

- `twin_endpoint_tail_vectors.npz`: endpoint response cache for static,
  edge-parallel, and edge-orthogonal endpoint motions.
- `endpoint_feature_preservation_static_decoder/feature_latents.npz`: matching
  Gabor/pyramid latents.
- `endpoint_feature_preservation_static_decoder/*`: existing static-trained
  feature-preservation decoder summaries.

This cache is best for preservation controls and edge-parallel/orthogonal
sanity checks, not for full trajectory-distribution comparisons.

What still needs new inference:

- true empirical trace ensembles, especially unpaired image-trace sampling;
- Brownian matched controls;
- OU matched controls;
- time-shuffled, rotated, or phase-randomized empirical traces;
- response summaries beyond the cached `pose_blind_delta_mean` axis endpoints.

## Minimum Viable Run

```text
images: 256 or 512 patches
traces: 256 empirical traces
population: canonical 756 units
pairing:
  unpaired ensemble primary
  paired/original diagnostic
motion families:
  static
  empirical 0.25x
  empirical 0.5x
  empirical 1x
  OU matched effective RMS + autocorrelation/confinement
  Brownian matched RMS, secondary
  shuffled or rotated empirical trajectory control
features:
  Gabor k=4 or grouped Gabor features
  pyramid k=8 or grouped pyramid features
response summaries:
  temporal PCs, primary
  compact multi-bin summary, if cheap
  delta from static, secondary
  mean response over trajectory, integrated-readout diagnostic
  delta_mean, motion-induced remapping readout
metrics:
  feature decoding with shared or fixed alpha
  signal/motion covariance ratio
  signal-motion subspace overlap
controls:
  clipping/effective-RMS report
```

MVP decision question:

```text
Does the empirical FEM distribution improve aggregate natural-image feature
representation compared with static and matched synthetic motion, and does the
useful scale sit near biological/sub-biological motion rather than the largest
tested motion?
```

Predefined pilot outcomes:

```text
real > OU > Brownian/static:
  empirical trajectory statistics add something beyond generic confined motion.

real ~= OU > Brownian/static:
  FEM-like confinement/autocorrelation is sufficient; exact biological trace
  identity is not uniquely required.

real ~= OU ~= Brownian > static, and score rises with motion scale:
  likely generic motion modulation unless signal-motion overlap shows a useful
  frontier.
```

## Current Pathfinder Result

The cleaned pathfinder:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_pathfinder_n64_k2_drift_only_common_unclipped_rel025-2_not_final
```

used the canonical `756`-unit twin, `64` images, `K=2` trace samples,
grouped-by-image CV, a drift-only/common-unclipped trace bank, scales
`0.25x`, `0.5x`, `1x`, `1.5x`, and `2x`, and empirical, OU, Brownian, and
rotated controls.

Sanity checks were clean:

```text
accepted drift-only trace sources: 40 / 64
same source traces reused across all scales: yes
median effective/requested RMS: 1.0
clipped fraction: 0.0
```

Interpretation:

1. The pathfinder does not show a simple "more motion is better" artifact. For
   temporal PCA/DCT summaries, empirical incremental gain over static becomes
   more negative as scale increases.
2. Empirical trajectories still outperform OU in several motion-derived
   contrasts, especially Gabor temporal PCA/DCT and pyramid at lower scales.
   This suggests a drift-like trajectory-statistics effect rather than a pure
   null.
3. Rotated empirical traces are competitive with empirical traces. Therefore
   the effect is not clearly specific to original trajectory orientation
   relative to the image; it is more consistent with useful real-trace
   kinematics in an unpaired ensemble.
4. The response summary decides the conclusion. Temporal PCA/DCT test a
   temporal-code enhancement story and are negative relative to static in this
   pathfinder. `mean` tests the time-averaged moving response. `delta_mean`
   tests whether the average motion-induced response change carries feature
   information, and remains strongly positive.

This pathfinder recommendation is now superseded by the substantial patched run
below. At the time, the proposed adjudicating run was:

```text
n=128
K=4
drift-only/common-unclipped trace bank through 2x
grouped-by-image CV
families: empirical, OU, rotated, static
features: Gabor k=4, pyramid k=8
summaries: temporal PCA, temporal DCT, mean, delta_mean
required reports: predictor dimensionality, fixed alpha, effective RMS,
clipping, source reuse, empirical-OU, empirical-rotated, empirical-static
```

Decision question:

```text
Do temporal PCA/DCT remain negative while delta_mean remains positive?
```

The substantial run did show positive temporal-PC incremental gains with better
sampling, so temporal summaries should no longer be treated as pathfinder-only
negatives. The remaining caveat is control specificity: empirical beats OU
robustly, while Brownian and rotated controls narrow the advantage at larger
scales.

## Current Substantial Result

The patched larger run is:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched
```

Corrected incremental static-plus-motion posthoc:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
    incremental_static_plus_motion_relids
```

The first automatic posthoc folder, `incremental_static_plus_motion`, used
old-style scale IDs in the launch command and produced empty gain tables. Use
the corrected `incremental_static_plus_motion_relids` folder for all incremental
claims.

Run configuration:

```text
images: 256
trace samples per family/scale/image: K = 4
population: canonical 756-unit twin
families: empirical, OU, Brownian, rotated
scales: 0.25x, 0.5x, 1x, 1.5x, 2x
features: gabor_local_field, pyramid_local_field
feature ranks: k = 4, 8
response summaries: temporal_pca, temporal_delta_pca, temporal_dct,
  temporal_dct_delta, mean, delta_mean
CV: grouped by image, 5 outer folds
trace policy: drift-only, common-unclipped source pool, same raw trace reused
  across scales for each family/sample
```

Motion bookkeeping was clean:

```text
accepted drift-only trace sources: 151 / 256
median effective/requested RMS: 1.0 for every family/scale
clipped fraction: 0.0 for every family/scale
```

This removes the major scale confound from the pathfinder. The above-`1x`
conditions are over-large relative to observed drift, but they are not capped or
clipping-driven in this cleaned source bank.

Primary temporal-PCA incremental result:

```text
static + empirical temporal_pca versus static alone

Gabor k=4:
  0.25x  +14.31, CI [+7.45, +21.79]
  0.5x   +13.04, CI [+6.81, +20.89]
  1x     +9.10,  CI [+3.73, +14.86]
  1.5x   +9.98,  CI [+5.36, +15.87]
  2x     +9.07,  CI [+3.87, +15.73]

Pyramid k=8:
  0.25x  +5.20, CI [+3.02, +7.68]
  0.5x   +4.89, CI [+2.88, +7.07]
  1x     +3.93, CI [+1.93, +5.86]
  1.5x   +4.44, CI [+2.34, +6.64]
  2x     +4.21, CI [+2.38, +6.23]
```

Empirical temporal-PCA incremental gain beat matched controls most cleanly at
small scales:

```text
Gabor k=4, empirical incremental-gain advantage:

0.25x: vs OU +21.24, vs Brownian +10.52, vs rotated +15.27
0.5x:  vs OU +19.59, vs Brownian +7.89,  vs rotated +11.21
1x:    vs OU +17.16, vs Brownian +0.51,  vs rotated +5.63
1.5x:  vs OU +18.69, vs Brownian +0.15,  vs rotated +8.58
2x:    vs OU +18.03, vs Brownian -0.60, vs rotated +7.55
```

Main read:

1. Empirical drift-like motion adds feature-decoding signal beyond the full
   static response for temporal-PC summaries.
2. Empirical beats OU robustly across scale, feature family, and rank.
3. Empirical beats Brownian and rotated most cleanly at `0.25x-0.5x`. Brownian
   becomes competitive at `1x-2x`, so the high-scale claim must be guarded.
4. The curve does not show a simple "more motion is better" failure mode.
   Gabor temporal-PCA gain is largest at `0.25x-0.5x`, then plateaus/lower
   through `2x`.
5. The result is stronger than the n=64 pathfinder, but it still does not prove
   exact biological trajectory optimality. The supported claim is
   distributional and twin-scoped.

Figure-relevant wording:

```text
In a cleaned BackImage aggregate run, empirical drift-like motion adds
feature-decoding signal beyond static V1-twin responses and outperforms
OU-like confined controls across scale. The advantage over Brownian/generic
motion is strongest at small biologically plausible scales and narrows at
larger scales, arguing against both a pure null and a simple more-motion-is-best
interpretation.
```

## Production Run

If the MVP lands:

```text
images: 1024+
traces: 512+
motion families:
  static
  empirical scales 0.125, 0.25, 0.5, 1
  OU matched
  Brownian matched
  shuffled empirical
  phase-randomized if cheap
  over-large empirical scale as a diagnostic
features:
  Gabor grouped by orientation/SF
  steerable pyramid grouped by scale/orientation
  DCT grouped by frequency band
response summaries:
  temporal PCs
  compact multi-bin
  delta from static
  mean diagnostic
metrics:
  decoding
  linear-Gaussian/logdet information
  signal-versus-motion covariance decomposition
  signal-motion subspace overlap
  information/cost Pareto frontier
```

## Figure Target

A successful aggregate analysis could support a Figure 4/5 panel sequence:

```text
A. Natural images x FEM distribution -> V1 response movies.
B. Motion families: static, empirical FEM, OU, Brownian, scaled empirical.
C. Aggregate image-feature information across scale/family.
D. Signal versus motion-nuisance covariance decomposition.
E. Feature breakdown by spatial-frequency band or latent family.
```

Possible claim:

```text
Empirical fixational drift statistics improve the V1-twin representation of
natural-image feature structure over matched OU-like confined motion, and add
feature-decoding signal beyond static responses. The advantage over generic
Brownian/rotated motion is strongest at small drift scales, so the claim is a
scale- and readout-dependent active-sensing result rather than proof of exact
trajectory optimality.
```
