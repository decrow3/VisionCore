# Companion: Aggregate FEM Information Model

Date: 2026-06-21
Status: provisional methods/logic companion for Figure 4B

## Panel Claim Under Test

```text
Motion enhances feature encoding.
```

This is the result Panel 4B is there to show if the evidence supports it. In
the current figure the claim is explicitly scoped: recorded trajectories are
used to render V1-twin response movies, and aggregate ridge readouts then test
whether the static-plus-motion response summaries carry more feature evidence
than stabilized/static responses. The trajectory is part of response rendering,
not an explicit aggregate ridge input. A same-axis pose-unaware hidden-sample
proxy tests the hidden-trajectory cost, while a full covariance-aware pose-blind
observer remains future work. Everything below is organized to decide how strongly
that panel claim can be made: the motivation, estimator choice, assumptions,
methods, results, controls, and caveats are all evidence for or against this
sentence.

## Top-Line Logic

Figure 4B is there to show one result:

```text
Motion-rendered responses carry additional feature evidence over static
responses.
```

This is the organizing claim for the panel and for this companion. "Motion-
rendered responses" means that recorded or control trajectories are used to
render the response movie from the same image; the aggregate ridge decoder then
receives response summaries, not the trajectory itself. The current promoted
readout is a diagonal Gaussian decoder-information increment over the stabilized
baseline, expressed in bits, rather than the legacy negative-MSE axis.

That framing helps rather than hurts the story. Panel B is a trajectory-
conditioned rendering analysis: retinal motion can expose feature-decodable
structure in the response movie without making the aggregate decoder a
trajectory-aware observer. The pose-unaware trace remains valuable as a
separate hidden-pose diagnostic, and it is now recomputed on the same
source-trial grouped information axis as the promoted trace. Panel C then asks
the separate latent-eye question: whether an observer can recover feature
evidence without being given the true trajectory.

Everything below is included to support, qualify, or protect that result:

- Motivation: why motion should be tested as a feature-encoding operation rather
  than treated only as nuisance displacement.
- Decisions and assumptions: why the current panel is trajectory-conditioned,
  drift-scoped, and targeted to coarse local pyramid features.
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
decoder-information axis unless we explicitly build a 4B posthoc that
implements the same latent-pose or covariance-penalty assumption.

The current 4B production surface is drift-scoped. The n384 power rerun used a
source-trace filter with `max_trace_source_microsaccade_events = 0`, giving `241
/ 384` eligible source traces under the configured RMS, radius, path-length,
speed, and source-microsaccade filters. There is no dedicated
drift-versus-microsaccade breakdown for this panel. Microsaccades are therefore
out of scope for the current 4B claim; the separate covariance-aware Figure 5
work is where fixation versus microsaccade windows were explicitly compared.

This companion keeps the older motivation and pilot lineage because it explains
why the current modest, trajectory-conditioned, drift-only claim is the right
one:

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

The aggregate FEM model supports the Panel B claim that motion-rendered
responses carry more feature evidence than stabilized/static responses. It asks
whether the distribution of biological-like fixational motion adds feature-
decodable structure to V1-twin responses beyond the static image response. The
simplifying assumption it breaks is that a single static response is the
relevant sensory representation during fixation. In these analyses the V1-twin
input is a retinal movie, and the response summary can carry information about
image features through the interaction of image structure and eye trajectory. It
is not a hidden-eye decoder and not an aggregate decoder that is handed the
trajectory: the eye trajectory is used to render the response movie.

A pose-unaware extension would be a new observer assumption, not a relabeling of
the existing temporal-PCA guardrail. It would either marginalize the trajectory,
`p(y | I) = integral p(y | I, tau) p(tau) d tau`, or add the movement-induced
covariance term `Sigma_FEM(D)` to the readout noise. The current panel has a
trajectory-conditioned rendering interpretation; the pose-unaware line is a
same-axis companion contrast showing the cost of withholding the trajectory
sample. A full covariance-aware pose-blind observer remains a separate future
analysis.

The supported claim is distributional, readout-scoped, and response-rendering
conditional. After the 2026-06-21 static-baseline correction, mean and
`delta_mean` are the absolute feature-gain candidates, while temporal PCA/DCT
variants are order-sensitive empirical-vs-control diagnostics. The model does
not prove that the exact recorded trajectory for an image is optimal, nor that
the animal is optimizing the tested decoder. It also does not yet address
microsaccade windows.

## Motivation

The motivation is to test whether fixational motion can be a feature-encoding
operation in rendered response movies. If FEMs were only nuisance displacement,
then adding a motion summary to a static response should either do nothing
useful or behave like generic motion energy. The aggregate model turns that
into a controlled comparison. It holds the image set and V1 twin fixed, draws
trajectories from empirical and matched control families, and asks whether
motion-rendered response summaries improve recovery of local image features
under grouped cross-validation.

This is the ensemble-level companion to the local pairing and joint-posterior
models. It answers "does the biological-like trajectory distribution create
useful response structure?" rather than "is this exact trajectory paired with
this exact image better than alternatives?"

## Notation And Estimator Contract

Shared notation:

```text
I: image or image window
tau: eye trajectory used to render the retinal movie
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
grouped ridge decoding score reported for the promoted panel as a diagonal
Gaussian decoder-information increment in bits. Family specificity is measured
by:

```text
C(F1, F2, a; phi, s) = G(F1, a; phi, s) - G(F2, a; phi, s)
```

The estimator contract is therefore not "mutual information" in a literal
noise-model sense. It is a deterministic feature-decodability proxy with a fixed
split convention and explicit motion-family controls.

## Plain-English Methods

The aggregate 4B analysis asks whether response movies contain more recoverable
image-feature information than a static response, when the model is told the
true retinal trajectory.

The analysis starts with a reviewed set of BackImage image windows and recorded
fixation traces. For each image window, the static condition renders the image
at one reference eye position. The motion condition renders a short retinal
movie by shifting the same image according to a sampled eye trace. The V1 twin
is then run on each rendered movie to produce a response movie.

The current production run is drift-scoped. Traces with detected microsaccade
events are filtered out before sampling source traces. Additional filters keep
the source traces within the configured RMS, radius, path-length, and speed
limits. Under these filters, 241 of 384 source traces are eligible. The rendered
movies were checked for effective/requested RMS and clipping; the inspected
production rows have median RMS ratio 1.0 and clipped fraction 0.0.

Each response movie is turned into a simpler response summary before decoding.
The important summaries are `mean`, which averages the response over the movie,
and `delta_mean`, which measures the motion-induced change relative to the
static response. Temporal PCA and temporal DCT summaries are also computed, but
they are now treated as order-sensitive diagnostics rather than the main
absolute-gain result.

The target to be decoded is a coarse local feature map of the image. For the
main panel, this is the `pyramid_local_field` target with `k = 16`. It is built
from local oriented pyramid coefficients and includes signed real/imaginary
components plus magnitude. It is not a full image reconstruction target.

The decoder is trained with grouped cross-validation. The historical default is
grouped by rendered image window, so all examples from a held-out image window
stay out of the training set when that window is tested. Because several crops
can share a source trial, the promoted run also has a strict
`source_trial`-grouped cache. That stricter grouping materially changed the
scale curve, so the source-trial grouped value is now the primary result and
the image-group result is retained as optimistic provenance context.

The promoted score is the diagonal Gaussian decoder-information increment in
bits when the motion response summary is added to the static response summary.
A positive value means the motion response summary helped recover the image
feature target. Legacy negative-MSE summaries remain archive/QC context and
should not be mixed with the information-axis panel.

The main comparison uses trajectories to render each response movie and then
decodes from aggregate response summaries. This is why the result should be
called trajectory-conditioned or motion-rendered, not an exact-trajectory ridge
decoder. The pose-unaware proxy is a separate check: it trains on image-mean
motion summaries but tests on individual hidden trajectory samples without
telling the decoder which trajectory generated them. This proxy is not a full
Bayesian observer, but it is now commensurate with the promoted
decoder-information units.

The motion controls are rendered and scored the same way. Empirical traces are
compared with Brownian, rotated, and OU-like trace families. Brownian asks
whether generic motion of the same scale is enough. Rotated asks whether the
image-relative direction matters. OU-like confined motion remains audit-only
because its below-static behavior may reflect a trace-generation or readout
mismatch.

## SSI Adjudication Test

The older spatial-SSI analysis should not be dismissed as only a "where"
measure. The activation map contains many differently tuned units, so map
peakiness can in principle carry image-content information. The clean test is
therefore not an argument about whether SSI or the decoder is better in the
abstract. The clean test is to feed SSI-derived features into the same held-out
feature decoder used for 4B.

The implemented adjudication keeps the downstream endpoint fixed:

```text
target: pyramid_local_field image feature target
decoder: grouped-by-image ridge decoder
folds and ridge policy: same as the aggregate 4B decoder
score: held-out feature prediction / decoder-information increment
```

Only the input representation changes:

```text
ordinary response summary:
  X = mean, delta_mean, temporal PCA/DCT, etc.

SSI representation:
  X = spatial-SSI-derived features computed from the full readout map before
      spatial max-pooling

incremental representation:
  X = [ordinary response summary, SSI representation]
```

The aggregate runner now exposes this with:

```text
--compute-ssi-features
--ssi-summary-names
--ssi-incremental-base-summaries
```

The main SSI summary is `ssi_itn`, the flattened per-time, per-unit
Skaggs-style SSI quantity. Smaller or more generous variants are also
available: `ssi_unit_mean`, `ssi_unit_spike_weighted_mean`, `ssi_rbar_itn`,
`ssi_itn_plus_rbar`, and `ssi_population_time`. Incremental summaries such as
`delta_mean_plus_ssi_itn` ask whether SSI features add held-out feature
prediction beyond an ordinary response summary.

Interpretation is endpoint-specific:

```text
If SSI features improve held-out feature prediction beyond the response
summary, then the response summary is leaving content-relevant map structure on
the table.

If SSI features do not improve held-out feature prediction, then any real SSI
gain is not adding recoverable image-feature information for the current 4B
endpoint.
```

This test does not make SSI the gold standard. It uses the feature decoder as a
common measuring stick so SSI and ordinary response summaries answer the same
question: how much do they predict about image content on held-out images?

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
Given a trajectory-rendered response movie, how well can the V1-twin response
summary recover a coarse map of local oriented image structure, including both
phase-sensitive signed components and phase-insensitive magnitude?
```

This is a principled local-feature aggregation target, not a lossless pyramid
or pixel-level reconstruction. If a motion-scale curve peaks away from 1x, one
possible interpretation is that that scale improves the predictability or
coherence of these coarse signed/magnitude feature summaries from the response
movie. It should not be described as evidence that V1 is only encoding energy,
nor as evidence that the full image is reconstructable.

## Assumptions

A0. The central interpretation is trajectory-conditioned rendering: motion is
used to generate the response movie, and the aggregate ridge decoder reads only
the resulting static-plus-motion response summaries.

A1. The V1 twin response movie is a meaningful proxy for the response changes
that structured retinal image motion would induce in the analyzed population.

A2. Grouped-by-image cross-validation prevents exact image-window reuse in train
and test, but crops can still share a source trial. Strict source-trial grouping
is therefore a required robustness check before promotion.

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
static-plus-motion decoder on the image-mean motion summary and test held-out
images on individual trajectory samples without the trajectory label. It is
useful for showing hidden-pose cost and is now on the promoted
decoder-information axis. It is still not the old covariance-aware Fisher
observer. A true covariance-aware pose-unaware 4B calculation still needs
per-trace or raw-response covariance and should be labeled separately.

A8. The `pyramid_local_field` target keeps coarse signed phase-sensitive
structure, but not full within-block phase layout. Signed coefficient averaging
can cancel when phase varies across an 8x8 block; this is a known property of
the target rather than an energy-only assumption.

## Controls

The controls are organized around the same top-line claim. They ask whether the
motion-rendered gain is really a response-summary feature-encoding benefit,
whether it is specific to empirical drift-like motion rather than generic
displacement, and whether hidden pose changes the sign or magnitude of the
effect.

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

The current evidence has two linked pieces: a source-trial grouped motion-
rendered information-gain trace above static across the tested motion scales,
and a same-axis pose-unaware hidden-sample proxy with negative point estimates
when the trajectory sample is hidden. The promoted figure now uses only
same-axis decoder-information rows.

Primary production-candidate source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/
    incremental_staticmean_plus_motion_info_decode_bootstrap_b50_validated_20260630/
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
CV grouped by strict source trial for the promoted cache
trajectory-conditioned rendering: yes
source trace policy: drift-only source pool, no microsaccade condition
displayed main trace set: motion-rendered empirical, Brownian, rotated, pose-unaware empirical hidden-sample proxy
pose-unaware empirical hidden-sample proxy: same-axis source-trial grouped information proxy
OU display policy: audit-only until the trace/readout behavior is resolved
```

Strict source-trial grouped primary information-axis read:

```text
pyramid_local_field k16 delta_mean, n=384, information gain bits:
  empirical - static_mean:
    0.25x +1.09 CI [+0.22, +2.36]
    0.5x  +1.15 CI [+0.25, +1.74]
    1x    +0.98 CI [+0.33, +1.84]
    1.5x  +0.69 CI [-0.01, +1.60]
    2x    +0.72 CI [+0.07, +1.53]
```

Same-axis pose-unaware hidden-sample proxy:

```text
pyramid_local_field k16 delta_mean, n=384, information gain bits:
  pose-unaware hidden-sample - static_mean:
    0.25x -0.53 CI [-1.41, +0.36]
    0.5x  -0.40 CI [-1.22, +0.57]
    1x    -0.60 CI [-1.38, +0.34]
    1.5x  -0.62 CI [-1.44, +0.35]
    2x    -0.49 CI [-1.44, +0.35]
```

Image-grouped comparison, retained as optimistic provenance context:

```text
pyramid_local_field k16 delta_mean, n=384, information gain bits:
  empirical - static_mean:
    0.25x -0.28 CI [-1.21, +0.99]
    0.5x  +0.26 CI [-0.62, +0.97]
    1x    +0.90 CI [+0.10, +1.62]
    1.5x  +0.98 CI [+0.32, +1.49]
    2x    +0.98 CI [+0.42, +1.71]
```

Legacy corrected negative-MSE lineage read, not the promoted information axis:

```text
pyramid_local_field k16 delta_mean, n=384, legacy `-MSE` units:
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
pose-unaware hidden-sample proxy: same-axis companion trace showing hidden-sample cost
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
Keep the pose-unaware empirical hidden-sample proxy visible as a same-axis
hidden-sample cost trace.
Keep the Brownian/rotated narrowing visible.
Keep OU out of the main trace set and route it to the audit/supplement because
its below-static temporal behavior can reflect trace-generation or analysis
mismatch.
Call the score a decoder-information feature-evidence increment.
State that the trajectory renders the response movie and is not an explicit
aggregate ridge input.
Require point estimates and point-centered decode-bootstrap CIs to come from the
same validated cache; fail the panel build if any information point estimate
lies outside its reported CI.
Use strict source-trial grouping as the promoted split; report the image-group
cache as the optimistic/provenance comparison.
State that the current run is drift-only; microsaccades require a separate
stratified analysis.
Do not label the negative temporal-PCA static-mean guardrail as pose-unaware; it
is a response-summary/readout diagnostic, not the covariance-aware pose-blind
observer from the older notes.
```

## Current Claim Boundary

Supported:

```text
In the cleaned aggregate BackImage analysis, empirical drift-like trajectories
render V1-twin response movies whose static-plus-motion summaries carry more
diagonal Gaussian decoder information about local pyramid features than the
stabilized/static baseline.
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

Production figures and captions should keep the scope visible: trajectories
render the response movie; they are not an explicit aggregate ridge-decoder
input. The plotted trace set, caption, and supplement routing should separate
the same-axis hidden-sample proxy from a still-future full covariance-aware
pose-blind observer.

The final production panel should use the completed power-rerun surface:

```text
absolute aggregate candidates: pyramid_local_field k16 mean, delta_mean
local mechanistic bridge: pyramid_local_field k16 delta_mean
order-sensitive diagnostics: pyramid_local_field k16 temporal_pca / temporal_dct variants
output target:
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1
primary corrected posthoc:
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/
    incremental_staticmean_plus_motion_info_decode_bootstrap_b50_source_trial_validated_20260630/
image-group provenance comparison:
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/
    incremental_staticmean_plus_motion_info_decode_bootstrap_b50_validated_20260630/
same-axis pose-unaware proxy:
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_pose_unaware_production_n384_empirical_k8_seed0/
    pose_unaware_staticmean_plus_motion_info_source_trial_b50_20260630/
all-readout audit:
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/
    incremental_staticmean_plus_motion_allreadouts_v1/readout_atlas_figures/
```

The production figure pack should report:

```text
trajectory-conditioned response-rendering assumption
split convention
strict source-trial grouping primary result plus image-group comparison
trace source policy, including drift-only source filtering
effective/requested RMS and clipping
empirical information gain over static
empirical-minus-Brownian/rotated contrasts
same-axis pose-unaware empirical hidden-sample proxy
absolute gain guardrail showing when generic motion catches up
OU contrast only as an audit-pending temporal diagnostic, not in the main
displayed trace set
older pathfinder/n256 motivation only as lineage, not as current headline values
full covariance-aware pose-unaware trace only after a dedicated latent-pose or
covariance-aware posthoc with per-trace response covariance
```
