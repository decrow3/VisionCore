# Figure 5 Reafferent Covariance Plan

This is the current north-star plan after the natural-image and e-optotype
Checks 5-9 audit.

## Reframed Claim

Do not make real-trajectory optimality the load-bearing claim. The current
random-motion controls make that too fragile.

Use this stronger and safer frame instead:

> Fixational eye movements generate a structured reafferent covariance
> component in V1. The key question is whether this component accounts for
> reliable shared variability, whether it is benign or limiting for population
> coding, and whether pose-aware or compact-geometry-aware readouts can recover
> information that pose-blind analyses treat as noise.

The natural-image movie-information result remains useful context: real
retinal motion improves deterministic V1-model spatial information efficiency
over stabilization, especially for mid/high spatial-frequency content. It is
not, by itself, enough to claim that measured FEM trajectories are optimal.

## What Is Already Done

Production natural-image twininfo run:

```text
outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu/
```

Main model-information endpoint:

```text
metadata/05_lagcube_information_summary.csv
final_cumulative_spatial_ssi_bits_per_spike
```

Current read:

- real beats stabilization on bits per expected spike;
- gains are strongest for mid/high spatial-frequency image content;
- matched random-motion controls match or exceed real, so trajectory
  optimality is not supported.

Natural-image center-channel Checks 5-9:

```text
outputs/active_sensing_movie_information/figure5_natural_image_population_checks_5_to_9/
```

Current read:

- 16-channel center response cache is internally consistent;
- Check 6 gives real higher covariance efficiency than stabilized
  (`eta` 1.499 versus 1.117);
- Check 5 alignment and Check 7 remove-out do not show a real-specific
  advantage;
- the result is not directly comparable to the e-optotype scaffold because the
  response dimension, class count, repeat count, and stabilization definition
  are different.

Historical e-optotype scaffold:

```text
outputs/active_sensing_movie_information/figure5_cached_rate_checks_5_to_9_fixed_lm-020/
outputs/active_sensing_movie_information/figure5_cached_rate_checks_5_to_9_check8_tfts_delta025_lm-020/
```

Current status:

- useful for debugging the Check 5-9 machinery;
- not Figure 5 evidence;
- not a matched natural-image comparison.

## Priority 1: Variance Accounting Denominator

Goal:

> Quantify what fraction of reliable shared variability is FEM-linked or
> reafferent.

This is the most direct route to the claim that FEMs explain cortical
variability.

Recommended denominator:

```text
reliable shared covariance after removing measurement noise / trial-private noise
```

Candidate numerator:

```text
FEM-linked covariance component explained by eye position, retinal pose,
translation-tangent predictors, or the compact Figure 4 subspace
```

Report:

```text
fraction_explained = tr(C_reaff_explained) / tr(C_reliable_shared)
```

Include controls:

- global-rate residualization;
- target PC1 residualization;
- shuffled eye traces;
- stimulus-label or time-bin shuffle;
- split-half reliability of the numerator and denominator.

Decision:

- high reliable fraction: strong support for "reafference explains cortical
  shared variability";
- low reliable fraction: keep reafference as a structured component, but do not
  make it the dominant variability explanation.

Implementation status, 2026-06-09:

```text
declan/active_sensing_movie_information/summarize_reafferent_variance_accounting.py
outputs/active_sensing_movie_information/reafferent_variance_accounting/
```

This first implementation summarizes existing Phase 1, direct derivative, and
finite-difference outputs. It is a denominator dashboard, not a raw covariance
recomputation.

Current aggregate read:

```text
aggregation true-minus-eye-shuffle fraction: 0.848 +/- 0.040 across 4 sessions
noise-correlation reduction fraction:        0.333 +/- 0.037 across 4 sessions
model-alignment excess / reliability:        0.083 +/- 0.025 across 4 sessions
direct derivative compact capture:           0.484 +/- 0.030 across 13 sessions
finite-difference tangent capture:           0.436 in 1 session
```

Added trace-unit closure from the saved finite-difference run:

```text
outputs/active_sensing_movie_information/reafferent_variance_accounting/
variance_accounting_trace_closure.csv
variance_accounting_trace_closure_summary.csv
```

This converts the finite-difference capture fractions into matched target
covariance trace units. In the one current finite-difference session
(`Allen_2022-02-16`), the sample eye-trace covariance captures:

```text
no residualization:              0.602, 0.747, 0.819 at k = 2, 10, 20
global-rate residualization:     0.402, 0.614, 0.724 at k = 2, 10, 20
global-rate + target-PC1:        0.188, 0.426, 0.584 at k = 2, 10, 20
```

Guardrail:

> These rows do not all share the same denominator. The aggregation and
> noise-correlation rows are denominator-like evidence; compact capture and
> finite-difference capture are numerator candidates relative to their own
> target covariance. A load-bearing claim still needs a raw
> reliable-shared-covariance trace denominator.

Current missing artifact:

> No saved full reliable-shared covariance matrix or raw trace denominator was
> found in the existing Phase 1 / derivative / finite-difference output
> folders. To finish Priority 1 as a load-bearing denominator claim, the
> producer should save either the relevant covariance matrices or at least
> matched traces for `C_reliable_shared`, `C_reaff_explained`, and control
> covariance terms.

## Priority 2: Constrained Population Coding

Goal:

> Test whether FEM-linked covariance is benign, limiting, or useful under a
> covariance-aware population metric.

Primary metrics:

```text
dprime_pop^2   = dmu.T inv(Sigma) dmu
dprime_indep^2 = dmu.T inv(diag(Sigma)) dmu
eta            = dprime_pop^2 / max(dprime_indep^2, eps)
```

or the equivalent Fisher form:

```text
eta = J_pop / J_indep
```

Run this in a matched response space whenever possible:

- recorded V1 response/covariance space;
- canonical 756 response-channel twin space;
- or a deliberately downsampled control where natural-image and e-optotype
  runs use the same response dimension and repeat count.

Interpretation:

- `eta_real > eta_stabilized`: reafferent covariance is relatively benign or
  helpful;
- `eta_real < eta_stabilized`: reafferent covariance is limiting for a
  pose-blind decoder;
- random controls similar to real: generic retinal motion is sufficient;
- real above matched random controls: measured FEM statistics carry additional
  structure.

Implementation status, 2026-06-09:

```text
declan/active_sensing_movie_information/summarize_constrained_population_coding.py
outputs/active_sensing_movie_information/constrained_population_coding/
```

This summarizes the saved natural-image Check 6 pairwise rows:

```text
input rows:      5616
conditions:      16
paired contrasts: 5
```

Current paired real-minus-control read:

```text
real - stabilized:
  delta dprime2_pop = -1.548
  delta eta         = +0.382

real - random_cov:
  delta dprime2_pop = -0.270
  delta eta         = -0.035

real - random_amp_cloud_matched:
  delta dprime2_pop = -0.414
  delta eta         = -0.004

real - trajectory_order_shuffle:
  delta dprime2_pop = +0.143
  delta eta         = +0.194
```

Interpretation:

- real retinal motion has higher `eta` than stabilization, so covariance is
  more benign by this ratio;
- real also has lower absolute full-covariance image-identity `dprime2_pop`
  than stabilization, so this is not a simple information-win;
- matched/random motion controls are similar to or above real on `eta`, so
  this still does not support real-trajectory optimality;
- the canonical 756-channel natural-image Check 6 remains tabled for the
  proper e-optotype comparison.

## Priority 3: Pose-Aware Recoverability

Goal:

> Show that a pose-aware or tangent-aware readout recovers information that a
> pose-blind readout treats as noise.

Compare held-out decoders:

```text
pose-blind:      response-only stimulus/readout
pose-aware:      response + eye position / retinal pose / trace coefficients
tangent-aware:   response after fitting/removing compact reafferent terms
```

Guardrails:

- fit pose/tangent terms on training folds only;
- evaluate stimulus readout on held-out trials;
- report both accuracy/log-likelihood and residual covariance;
- include shuffled-pose and shuffled-trace controls.

Decision:

- pose-aware recovery supports active-sensing computation language;
- no recovery means the covariance may be structured without being
  functionally recoverable.

Implementation status, 2026-06-09:

The natural-image Checks 5-9 runner now has an explicit pose-covariate export
hook:

```bash
.venv/bin/python declan/active_sensing_movie_information/run_figure5_natural_image_population_checks_5_to_9.py \
  --out-dir outputs/active_sensing_movie_information/figure5_natural_image_population_checks_5_to_9 \
  --export-pose-covariates
```

Expected outputs:

```text
natural_image_condition_pose_summary.csv
natural_image_condition_pose_frames.csv
```

This is not yet the pose-aware decoder. It is the missing design-matrix export
needed for one: per-record condition traces are reconstructed with the same
trajectory-control generator used for the model movies, then written as
summary and per-frame pose covariates aligned to the cached response records.
The normal cached Checks 5-9 path was rerun without export and still completed
cleanly.

## Priority 4: Compact Addback / Removeout

Goal:

> Test whether the compact Figure 4 geometry carries the covariance and/or
> information effects.

Addback construction:

```text
delta_r          = r_real - r_stabilized
delta_compact    = P_U delta_r
delta_orthogonal = (I - P_U) delta_r

r_compact_addback = r_stabilized + delta_compact
r_orth_addback    = r_stabilized + delta_orthogonal
```

Removeout:

```text
r_removed = r - P_U r
```

Success criterion:

- compact addback restores the covariance or constrained-coding effect;
- compact removeout reduces or kills the effect;
- orthogonal addback fails or is clearly weaker.

Current blocker:

The completed natural-image center-channel run uses 16 response channels, so
the prior 756-unit Figure 4/TFTS basis is not compatible. A fair compact
addback/removeout test needs a matched response space.

## Tabled But Important: Canonical Natural-Image Checks 5-9

Do not lose this thread.

The completed natural-image Checks 5-9 used only the 16 biological twin
channels at the center readout location. That was a tractable natural-image
sanity bridge, but it is not the proper comparison to the e-optotype scaffold
or the Figure 4/TFTS compact basis.

The tabled heavier run is:

> Recompute natural-image Checks 5-9 in the canonical 756-response-channel
> space, or in whatever exact response space is used by the compact Figure 4
> basis.

Why it matters:

- it removes the biggest response-space mismatch between natural images and
  the e-optotype scaffold;
- it makes compact addback/removeout dimension-compatible;
- it avoids the 16D alignment-ceiling problem seen in
  `check5_natural_image_covariance_spectrum_diagnostics.csv`;
- it gives a fairer test of whether the e-optotype alignment/recoverability
  result transfers to natural images.

Why it is tabled for now:

- it will be computationally heavier than the 16-channel cache;
- full covariance and cross-validation need enough repeats or careful
  regularization;
- it should be scheduled after the variance-accounting and constrained-coding
  priorities are stable.

Minimum guardrails for that run:

- record the exact response-channel identity and basis compatibility in the
  manifest;
- keep natural images, e-optotype controls, and Figure 4 basis in matched
  response coordinates;
- report covariance spectrum/effective-rank diagnostics before interpreting
  alignment;
- do a small `--max-images` or reduced-condition smoke first;
- launch production with `nohup` only when GPU occupancy is acceptable.

## Optional Final Step: Amplitude / Diffusion Sweep

This should be optional and downstream, not a core deliverable.

Run only after the constrained-coding and recoverability metrics are stable.

Candidate sweep:

```text
s = 0, 0.25, 0.5, 1, 1.5, 2, 3
```

For each scale or diffusion constant, report:

```text
bits per expected spike
raw bits
expected spikes
dprime_pop or J_pop
dprime_indep or J_indep
eta
compact recruitment
pose-aware recovery
```

Separate:

- intact natural images;
- spatial-frequency bands;
- fixation-only windows;
- one-microsaccade windows.

Why optional:

- it mostly asks about dose-response and operating point;
- it does not by itself establish that reafference explains recorded cortical
  variability;
- it can become expensive quickly if run in the full canonical response space.

## Draft Language

Use:

> FEMs generate a structured reafferent covariance component in V1. We quantify
> how much reliable shared variability this component explains and test whether
> it is limiting, benign, or recoverable under covariance-aware and pose-aware
> readouts.

Avoid:

> Real FEM trajectories are optimal.

Use only with strong new evidence:

> Measured FEM statistics provide trajectory-specific benefits beyond matched
> random retinal motion.
