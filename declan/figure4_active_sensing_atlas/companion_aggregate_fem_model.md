# Companion: Aggregate FEM Information Model

Date: 2026-06-21
Status: provisional methods/logic companion for Figure 4B

## Top-Line Logic

Figure 4B is there to show one result:

```text
Motion enhances feature encoding, but only when eye position is known.
```

This is the organizing claim for the panel and for this companion. "Motion
enhances feature encoding" refers to the known-eye / exact-trajectory result:
the model is handed the retinal trajectory when rendering the response movie,
and the motion summary improves decoding of local image features relative to the
static response at the larger motion scales. "But only when eye position is
known" refers to the hidden-pose contrast: when the decoder is not given the eye
position or trajectory, the same motion becomes a nuisance source and the
pose-unaware proxy falls below the static baseline.

That framing helps rather than hurts the story. Panel B is the pose-known /
exact-trajectory upper bound: retinal motion can expose feature-decodable
structure when retinal pose is available. The pose-unaware trace makes the
condition explicit rather than burying it. Panel C then asks the separate
latent-eye question: whether an observer can recover some known-eye benefit
without being given the true trajectory.

Everything below is included to support, qualify, or protect that result:

- Motivation: why motion should be tested as a feature-encoding operation rather
  than treated only as nuisance displacement.
- Decisions and assumptions: why the current panel is known-eye, drift-scoped,
  and targeted to coarse local pyramid features.
- Methods and results: how the decoding score, target, source traces, and
  current production values instantiate the claim.
- Controls and caveats: why generic motion, hidden pose, OU behavior, readout
  choice, block averaging, and microsaccade scope constrain the interpretation.

The specific pose-unaware contrast from the older covariance-aware decoder notes
should stay visible here. In those notes, "pose-unaware" was called
`pose_blind`: the response is decoded without being told the eye position or
trajectory that generated each response sample. In probability notation, the
observer no longer conditions on the trajectory:

```text
known-eye / pose-aware:
  p(y | I, tau)

pose-unaware / pose-blind:
  p(y | I) = integral p(y | I, tau) p(tau) d tau
```

In plain English, the model response still contains motion-driven structure, but
the decoder cannot re-index that structure by eye position. The unknown motion
therefore enters the observation model as nuisance covariance. The covariance-
aware Fisher version used:

```text
mu(t) ~= mu_bar + J delta_e(t)
Sigma_FEM(D) ~= J Sigma_e(D) J.T

F_pose_aware = J.T inv(Sigma_poisson + Sigma_residual) J
F_pose_blind = J.T inv(Sigma_poisson + Sigma_residual + Sigma_FEM(D)) J
```

where `D` is movement scale and `Sigma_FEM(D)` is the movement-induced response
covariance. This is the mathematical reason a below-zero pose-unaware trace
would not undermine 4B: it would show that the same motion can be useful when
pose is known and costly when pose is hidden. It is also a warning about axes:
this covariance-aware Fisher/readout penalty is not identical to the current 4B
incremental negative-MSE score unless we explicitly build a 4B posthoc that
implements the same latent-pose or covariance-penalty assumption.

The current 4B production surface is drift-scoped. The n384 power rerun used a
source-trace filter with `max_trace_source_microsaccade_events = 0`, giving `241
/ 384` eligible source traces under the configured RMS, radius, path-length,
speed, and source-microsaccade filters. There is no dedicated
drift-versus-microsaccade breakdown for this panel. Microsaccades are therefore
out of scope for the current 4B claim; the separate covariance-aware Figure 5
work is where fixation versus microsaccade windows were explicitly compared.

This companion keeps the older motivation and pilot lineage because it explains
why the current modest, known-eye, drift-only claim is the right one:

- The original aggregate plan reframed the local `I_z` axis screen into an
  ensemble question about empirical FEM statistics over natural images.
- The n64 pathfinder established the drift-only/common-unclipped trace policy
  and showed that response summary choice mattered.
- The n256 run made the first strong aggregate case but used a temporal-PCA
  absolute-gain framing that was later superseded.
- The n384 power rerun plus static-mean correction moved the claim to a readout
  role split: `mean`/`delta_mean` for absolute aggregate gain, temporal PCA/DCT
  for order-sensitive empirical-vs-control diagnostics.

## Summary

The aggregate FEM model supports the Panel B claim that motion enhances feature
encoding, but only when eye position is known. It asks whether the distribution
of biological-like fixational motion adds feature-decodable structure to
V1-twin responses beyond the static image response. The simplifying assumption
it breaks is that a single static response is the relevant sensory
representation during fixation. In these analyses the input is a retinal movie,
and the response summary can carry information about image features through the
interaction of image structure and eye trajectory. It is not a hidden-eye
decoder: the eye trajectory is known to the rendering/model side of the
analysis.

A pose-unaware extension would be a new observer assumption, not a relabeling of
the existing temporal-PCA guardrail. It would either marginalize the trajectory,
`p(y | I) = integral p(y | I, tau) p(tau) d tau`, or add the movement-induced
covariance term `Sigma_FEM(D)` to the readout noise. The current panel has the
known-eye upper-bound interpretation; the pose-unaware line would be a companion
contrast showing the cost of withholding the trajectory.

The supported claim is distributional, readout-scoped, and eye-position
conditional. After the 2026-06-21 static-baseline correction, mean and
`delta_mean` are the absolute feature-gain candidates, while temporal PCA/DCT
variants are order-sensitive empirical-vs-control diagnostics. The model does
not prove that the exact recorded trajectory for an image is optimal, nor that
the animal is optimizing the tested decoder. It also does not yet address
microsaccade windows.

## Motivation

The motivation is to test whether fixational motion can be a feature-encoding
operation when the sensory system knows where the eye is. If FEMs were only
nuisance displacement, then adding a motion summary to a static response should
either do nothing useful or behave like generic motion energy. The aggregate
model turns that into a controlled comparison. It holds the image set and V1
twin fixed, draws trajectories from empirical and matched control families, and
asks whether known-eye response movies improve recovery of local image features
under grouped-by-image cross-validation.

This is the ensemble-level companion to the local pairing and joint-posterior
models. It answers "does the biological-like trajectory distribution create
useful response structure?" rather than "is this exact trajectory paired with
this exact image better than alternatives?"

## Notation And Estimator Contract

Shared notation:

```text
I: image or image window
tau: known eye trajectory used to render the retinal movie
y = f_theta(I, tau): response movie from the V1 twin
phi(I): image feature target
s(y): response summary
D(s(y), phi(I)): cross-validated feature-decoding score
```

The response summaries used here include `temporal_pca`, `temporal_delta_pca`,
`temporal_dct`, `temporal_dct_delta`, `mean`, and `delta_mean`. The current
production target is a role split:

```text
absolute aggregate candidates:
  s(y) = mean(y), delta_mean(y)
local mechanistic bridge:
  s(y) = delta_mean(y)
order-sensitive diagnostics:
  s(y) = temporal_pca(y), temporal_dct(y), and static-subtracted variants
feature target:
  phi(I) = pyramid_local_field(I), k = 16
```

The aggregate static-plus-motion contract is:

```text
R_static(I) = mean(f_theta(I, tau_0))
R_motion(I, tau) = s(f_theta(I, tau)) or a motion-induced component

G(F, a; phi, s) =
  CV_D([R_static(I), R_motion(I, tau ~ F_a)], phi(I))
  - CV_D(R_static(I), phi(I))
```

where `F` is a trajectory family, `a` is the relative RMS scale, and `CV_D` is a
grouped-by-image ridge decoding score reported as incremental negative-MSE
gain. Family specificity is measured by:

```text
C(F1, F2, a; phi, s) = G(F1, a; phi, s) - G(F2, a; phi, s)
```

The estimator contract is therefore not "mutual information" in a literal
noise-model sense. It is a deterministic feature-decodability proxy with a fixed
split convention and explicit motion-family controls.

## Feature Target Interpretation

The `pyramid_local_field` target is not an image reconstruction target. It is a
coarse retinotopic map of local oriented pyramid features. The current feature
constructor computes steerable-pyramid coefficient maps and stores, for each
scale/orientation, 8x8 block means of:

```text
real coefficient
imaginary coefficient
magnitude
```

This target is therefore not an energy-only view of V1. V1 retains substantial
phase and position information, and the signed `real`/`imag` channels keep a
coarse phase-sensitive component alongside the phase-insensitive magnitude
channel. A pure magnitude-only target would be a cleaner complex-cell/energy
summary, but it would be an incomplete target for V1-like coding.

The important limitation is the block averaging. We are not literally averaging
an angle-valued phase variable, but averaging signed real and imaginary
coefficient maps is equivalent to averaging complex coefficients. Within a
block, spatially changing phase can cancel. Thus the signed channels preserve
phase/position information only at the scale that is coherent inside the block;
they discard finer within-block spatial phase layout. The magnitude channel
continues to report local oriented energy even when signed components cancel.

The scientific question is therefore:

```text
Given the known retinal trajectory, how well can the V1-twin response summary
recover a coarse map of local oriented image structure, including both
phase-sensitive signed components and phase-insensitive magnitude?
```

This is a principled local-feature aggregation target, not a lossless pyramid
or pixel-level reconstruction. If a motion-scale curve peaks away from 1x, one
possible interpretation is that that scale improves the predictability or
coherence of these coarse signed/magnitude feature summaries from the response
movie. It should not be described as evidence that V1 is only encoding energy,
nor as evidence that the full image is reconstructable.

## Assumptions

A0. The central interpretation is conditional: motion enhances feature encoding
only under a known-eye / exact-trajectory rendering and decoding assumption.
Hidden eye position is a different observer problem and is represented here by
the pose-unaware proxy.

A1. The V1 twin response movie is a meaningful proxy for the response changes
that structured retinal image motion would induce in the analyzed population
when the retinal trajectory is known.

A2. Grouped-by-image cross-validation prevents the decoder from using the same
image identity in train and test.

A3. The feature targets `gabor_local_field` and `pyramid_local_field` capture
image structure that is relevant to the local retinal movie, even though they
are not a complete description of natural-image content.

A4. The response summary `s(y)` is part of the scientific hypothesis. A temporal
summary can support an ensemble motion code even when `delta_mean` or other
local summaries behave differently.

A5. Matched motion families make the right null comparisons: a positive result
against static is not enough unless generic diffusion, confined drift, path
rotation, RMS mismatch, and clipping artifacts are also checked.

A6. The current panel is drift-scoped. Source traces with detected microsaccade
events are excluded by the production filter; a microsaccade-specific extension
would require a separate stratified run.

A7. The current 4B pose-unaware line is a hidden-sample proxy: train the
static-plus-motion decoder on the known-eye image-mean motion summary and test
held-out images on individual trajectory samples without the trajectory label.
It is useful for showing hidden-pose cost in the same `-MSE` units, but it is
not the old covariance-aware Fisher observer. A true covariance-aware
pose-unaware 4B calculation still needs per-trace or raw-response covariance
and should be labeled separately.

A8. The `pyramid_local_field` target keeps coarse signed phase-sensitive
structure, but not full within-block phase layout. Signed coefficient averaging
can cancel when phase varies across an 8x8 block; this is a known property of
the target rather than an energy-only assumption.

## Controls

The controls are organized around the same top-line claim. They ask whether the
known-eye gain is really a motion-driven feature-encoding benefit, whether it is
specific to empirical drift-like motion rather than generic displacement, and
whether the benefit depends on eye position being known.

Static baseline:

```text
R_static alone. Tests whether the movie summary adds anything beyond the
stabilized response.
```

OU-like control:

```text
Confined drift with matched scale. Tests whether the empirical result is merely
small bounded displacement.
```

Brownian control:

```text
Generic diffusion with matched RMS. Tests whether any motion of the same scale
is enough.
```

Rotated control:

```text
The same or matched path magnitude with altered image-relative direction. Tests
whether direction relative to image structure matters.
```

Motion QC:

```text
Effective/requested RMS, path length, accepted trace sources, and clipping.
The current n384 production surface uses a drift-only source pool by filtering
source traces to zero detected microsaccade events, plus RMS/radius/path/speed
limits. Under those filters, 241 / 384 source traces are eligible; the rendered
family/scale rows have median effective/requested RMS 1.0 and clipped fraction
0.0.
```

## Existing Evidence

The current evidence has two linked pieces: a known-eye gain trace above static
at larger motion scales, and a pose-unaware proxy below static when eye position
is hidden. Together they motivate the figure title rather than two independent
claims.

Primary production-candidate source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/
    incremental_staticmean_plus_motion_tworeadout_v2/
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/
    incremental_staticmean_plus_motion_allreadouts_v1/readout_atlas_figures/
```

Run scope:

```text
384 images
30 sessions
canonical 756-unit V1 twin
K = 8 trace samples per image/family/scale
families = empirical, OU, Brownian, rotated
scales = 0.25x, 0.5x, 1x, 1.5x, 2x
CV grouped by image
known-eye rendering: yes
source trace policy: drift-only source pool, no microsaccade condition
displayed main trace set: known-eye empirical, Brownian, rotated, plus
  pose-unaware empirical hidden-sample proxy
OU display policy: audit-only until the trace/readout behavior is resolved
```

Corrected primary-scale read:

```text
pyramid_local_field k16 delta_mean, n=384:
  empirical - static_mean:
    0.25x -0.65 CI [-2.17, +0.93]
    0.5x  +0.16 CI [-1.35, +1.77]
    1x    +1.76 CI [+0.58, +3.05]
    1.5x  +2.15 CI [+0.94, +3.31]
    2x    +2.23 CI [+0.98, +3.40]

pose-unaware empirical hidden-sample proxy:
  0.25x -5.29 CI [-7.01, -3.50]
  0.5x  -4.41 CI [-6.17, -2.62]
  1x    -2.74 CI [-4.09, -1.27]
  1.5x  -1.69 CI [-3.00, -0.39]
  2x    -1.79 CI [-3.12, -0.51]

temporal_pca:
  empirical - static_mean is negative at 0.25x, 0.5x, and 1x.
  empirical - OU remains strongly positive at those scales.
```

Superseded v5 power-rerun temporal-PCA values:

```text
Historical for absolute gain claims. These used the old temporal-PCA static
baseline and should not be used for Panel B headline text.

pyramid_local_field k16, temporal_pca:
  empirical - static:
    0.25x +8.82, CI [+5.90, +12.09]
    0.5x  +7.84, CI [+5.19, +10.75]
    1x    +7.79, CI [+5.27, +10.73]

  empirical - OU:
    0.25x +10.76, CI [+8.00, +13.95]
    0.5x   +9.74, CI [+7.43, +12.49]
    1x     +9.14, CI [+6.74, +11.98]

  empirical - Brownian:
    0.25x +2.77, CI [+1.44, +4.13]
    0.5x  +0.70, CI [-0.75, +1.99]
    1x    +0.85, CI [-0.30, +1.99]

  empirical - rotated:
    0.25x +1.66, CI [+0.39, +3.08]
    0.5x  +0.77, CI [-0.91, +2.26]
    1x    +1.12, CI [-0.15, +2.47]
```

The v5 temporal-PCA absolute-gain interpretation is superseded. Its static
baseline was a near-zero static temporal-PC summary rather than the static mean
response. The corrected v6 static-mean posthoc and all-readout/nested-alpha
audit now support a role split:

```text
mean: strongest absolute aggregate candidate under nested alpha
delta_mean: static-subtracted motion-induced/local-pairing bridge
temporal PCA/DCT: order-sensitive empirical-vs-control diagnostics
pose-unaware hidden-sample proxy: companion trace showing hidden-pose cost
OU: audit-pending and removed from the main 4B trace set
```

## Diagnostics And Failure Modes

The main anticipated failure modes are the ways the top-line claim could be
overstated or misassigned:

```text
"More motion is better" rather than empirical FEM-like motion.
Feature/readout choice is post hoc.
Signed coefficient block-averaging is mistaken for full phase reconstruction.
RMS, path length, or clipping mismatch explains the effect.
Grouped CV or cache construction leaks image identity.
A single seed, trace cohort, or K sample drives the effect size.
Brownian/rotated controls erase empirical specificity at the chosen k16 readout.
Microsaccade contamination is mistaken for a drift result.
```

Current handling:

```text
Show static gain and control contrasts together.
Add the pose-unaware empirical hidden-sample proxy as a distinct below-zero
trace, labeled as a proxy rather than a full covariance-aware observer.
Keep the Brownian/rotated narrowing visible.
Keep OU out of the main trace set and route it to the audit/supplement because
its below-static temporal behavior can reflect trace-generation or analysis
mismatch.
Call the score feature-decodable structure or a decoding proxy.
State the known-eye condition in the panel title/caption.
State that the current run is drift-only; microsaccades require a separate
stratified analysis.
Do not label the negative temporal-PCA static-mean guardrail as pose-unaware; it
is a response-summary/readout diagnostic, not the covariance-aware pose-blind
observer from the older notes.
```

## Current Claim Boundary

Supported:

```text
Motion enhances feature encoding, but only when eye position is known: in the
cleaned aggregate BackImage analysis, empirical drift-like motion can add
feature-decodable response structure beyond static V1-twin responses when the
V1 twin is given the exact retinal trajectory, while the current pose-unaware
proxy is below static.
```

Not yet supported:

```text
The exact recorded trajectory is optimal for its image.
The animal optimizes the feature decoder.
The effect is literal mutual information under a full observation-noise model.
Empirical motion uniquely beats every generic-motion control at every scale.
This result applies to microsaccade windows.
```

## Production Rerun Implications

Production figures and captions should keep the full sentence visible: motion
enhances feature encoding, but only when eye position is known. The plotted
trace set, caption, and supplement routing should make the known-eye assumption
and hidden-pose cost explicit.

The final production panel should use the completed power-rerun surface:

```text
absolute aggregate candidates: pyramid_local_field k16 mean, delta_mean
local mechanistic bridge: pyramid_local_field k16 delta_mean
order-sensitive diagnostics: pyramid_local_field k16 temporal_pca / temporal_dct variants
output target:
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1
corrected posthoc:
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/
    incremental_staticmean_plus_motion_tworeadout_v2/
all-readout audit:
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/
    incremental_staticmean_plus_motion_allreadouts_v1/readout_atlas_figures/
```

The production figure pack should report:

```text
known-eye / exact-trajectory rendering assumption
split convention
trace source policy, including drift-only source filtering
effective/requested RMS and clipping
empirical gain over static
empirical-minus-Brownian/rotated contrasts
pose-unaware empirical hidden-sample proxy trace
absolute gain guardrail showing when generic motion catches up
OU contrast only as an audit-pending temporal diagnostic, not in the main
displayed trace set
older pathfinder/n256 motivation only as lineage, not as current headline values
full covariance-aware pose-unaware trace only after a dedicated latent-pose or
covariance-aware posthoc with per-trace response covariance
```
