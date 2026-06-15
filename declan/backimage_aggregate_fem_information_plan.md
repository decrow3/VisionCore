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
4. `brownian_matched`
   - Brownian motion matched to effective RMS or diffusion.
5. `ou_matched`
   - Ornstein-Uhlenbeck motion matched to RMS, autocorrelation, and confinement.
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

- mean response over trajectory;
- delta from static;
- temporal PCs;
- compact multi-bin response features if dimensionality permits.

Primary contrasts:

```text
real - static
real - Brownian matched
real - OU matched
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
- top-k signal and motion variance;
- participation ratio;
- signal-motion subspace overlap;
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

## Minimum Viable Run

```text
images: 256 or 512 patches
traces: 256 empirical traces
population: canonical 756 units
motion families:
  static
  empirical 0.25x
  empirical 0.5x
  empirical 1x
  Brownian matched RMS
  OU matched RMS
features:
  Gabor k=4 or grouped Gabor features
  pyramid k=8 or grouped pyramid features
response summaries:
  mean response over trajectory
  delta from static
metrics:
  feature decoding
  signal/motion covariance ratio
controls:
  shared or fixed-alpha ridge
  clipping/effective-RMS report
```

MVP decision question:

```text
Does the empirical FEM distribution improve aggregate natural-image feature
representation compared with static and matched synthetic motion, and does the
useful scale sit near biological/sub-biological motion rather than the largest
tested motion?
```

## Production Run

If the MVP lands:

```text
images: 1024+
traces: 512+
motion families:
  static
  empirical scales 0.125, 0.25, 0.5, 1
  Brownian matched
  OU matched
  shuffled empirical
  phase-randomized if cheap
features:
  Gabor grouped by orientation/SF
  steerable pyramid grouped by scale/orientation
  DCT grouped by frequency band
response summaries:
  mean
  temporal PCs
  delta from static
metrics:
  decoding
  linear-Gaussian/logdet information
  signal-versus-motion covariance decomposition
  information/cost Pareto frontier
```

## Figure Target

A successful aggregate analysis could support a Figure 4/5 panel sequence:

```text
A. Natural images x FEM distribution -> V1 response movies.
B. Motion families: static, empirical FEM, Brownian/OU, scaled empirical.
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
