# BackImage Aggregate FEM Information Plan

Last curated: 2026-06-15.

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
  mean response over trajectory, diagnostic only
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
Fixational eye movements improve the ensemble representation of natural-image
structure in foveal V1, placing empirical drift near a useful information/cost
regime rather than simply maximizing retinal motion.
```
