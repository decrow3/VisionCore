# Analysis plan: RF-local power-matched phase surrogates for the FEM–SSI effect

Date: 2026-08-13

Status: planned map-first analysis; input-level controls must pass before neural scoring

Scope: corrected natural-image retinal movies, genuine 32-frame model history,
40 scored frames, RR100 digital-twin activation maps, firing rates, and spatial
selectivity index (SSI)

## 1. Scientific objective

The current FEM-versus-power analyses indicate that fixational eye movements
redistribute retinal image power across spatial and temporal frequencies. This
power redistribution explains part of the shared response modulation, but the
current power predictors do not explain the population differences in spatial
SSI sharpening.

The missing experiment asks whether the unexplained SSI effect depends on phase
structure:

> If receptive-field-local spatiotemporal power is held fixed while phase
> structure is destroyed, does FEM-induced SSI sharpening persist?

The control must not introduce a change in local temporal power that could itself
explain the SSI result. This is especially important because temporal-power
redistribution is the mechanism under test, not a nuisance variable that can be
absorbed into a generic definition of phase scrambling.

The primary hypotheses are:

1. **Power-sufficient hypothesis:** a phase-destroyed surrogate that matches
   response-relevant local SF×orientation×TF power retains the FEM-induced SSI
   change.
2. **Additional phase-sensitive hypothesis:** the matched surrogate loses or
   substantially reduces the FEM-induced SSI change, despite matching the power
   representation and intensity statistics.

A failed power match, reconstructed phase structure, or an uncontrolled
contrast difference makes the experiment non-diagnostic rather than evidence
for either hypothesis.

## 2. Decisions from the preliminary controls

Several simpler manipulations have already clarified the design.

### 2.1 Global and all-pass controls are diagnostics, not the primary test

- A coherent all-pass manipulation preserved too much cross-frequency phase
  organization and visually resembled small translations of intact edges.
- Correct global source-image Fourier scrambling destroyed phase, but applying
  the same FEM trace did not preserve the retinal movie's relevant power.
- Correct global 3-D phase scrambling exactly matched the unwindowed power of
  the complete 72-frame movie, but the 40 scored frames contained approximately
  1.8–1.9 times the intact movie's local temporal variance/power.
- The mismatch was systematic across independently sampled phase seeds. It was
  not merely an unlucky realization.
- Restricting or Hann-windowing the scored interval is phase-sensitive. Exact
  full-block power equality therefore does not imply equality of the
  response-window power used by the mechanistic analysis.

Consequently, the global 3-D scramble can remain an upper-bound or
out-of-distribution diagnostic, but it cannot provide the decisive phase test.
Seed-searching for a global scramble with favorable scored-window power is also
not the primary solution because it conditions the null on the local statistic
whose mechanistic role is being tested.

### 2.2 Existing local pyramid scrambling is not yet a matched-power control

The existing complex steerable-pyramid manipulation genuinely destroys phase
relationships at the coefficient level, but its reconstruction visibly blurs
the movie and reduces positive-temporal-frequency power to only a few percent
of the intact value. It is useful evidence that coefficient phase was altered,
but it cannot isolate phase from power or contrast.

### 2.3 Use the metamer principle, not the published human pooling geometry

The foveated-metamer work in
[eLife reviewed preprint 90554v2](https://elifesciences.org/reviewed-preprints/90554v2)
provides a useful conceptual template: start from noise and optimize a stimulus
to match locally pooled complex-filter energy without matching phase. Its
eccentricity-dependent human pooling windows are not directly applicable here.
The pooling distribution for this analysis must instead come from the digital
twin's effective spatial filters, and the matched representation must include
temporal frequency and model history.

## 3. Primary experimental construction

Optimize one static surrogate source image and render it through the exact
retinal-movie pipeline. Do not optimize an arbitrary collection of unrelated
movie frames in the first implementation.

For one source image and one eye trace, construct four paired conditions:

| Source image | Eye condition | Purpose |
|---|---|---|
| Intact | Stabilized | Intact baseline |
| Intact | FEM | Original FEM-induced SSI effect |
| Phase-destroyed surrogate | Stabilized | Surrogate baseline |
| Phase-destroyed surrogate | Same FEM | Matched-power phase test |

This construction has three advantages:

1. Every movie remains a physically valid translation of one source image.
2. The same FEM trace is used for the intact and surrogate conditions.
3. Each source has its own stabilized baseline, so the inference does not
   compare a scrambled moving movie against an unrelated intact static image.

The primary neural contrast is

\[
\Delta SSI_{\mathrm{intact}}
=SSI_{\mathrm{intact,FEM}}-SSI_{\mathrm{intact,stabilized}}
\]

versus

\[
\Delta SSI_{\mathrm{surrogate}}
=SSI_{\mathrm{surrogate,FEM}}-SSI_{\mathrm{surrogate,stabilized}}.
\]

Use paired differences rather than ratios because stabilized SSI can approach
zero. Keep the exact existing definition of `stabilized`; do not silently
replace trial-mean stabilization with a static-center oracle.

## 4. Stage 1 — measure composite model pooling scales

### Objective

Measure the distribution of full input-space effective receptive fields used by
the RR100 digital twin.

### Motivation

The existing spatial-scale checkpoint measures learned 14×14 Gaussian readout
masks after the twin core. It explicitly excludes support added by the learned
convolutional and recurrent core. It is therefore a necessary calibration but
not the final distribution of source-image pooling radii.

### Method

- Backpropagate representative RR100 outputs to the exact movie pixels that
  drive them.
- Use natural operating points rather than an arbitrary blank input.
- Retain the gradient by lag before forming the spatial summary.
- Form a 2-D spatial energy map from squared gradients, with a separately saved
  lag-energy profile.
- Measure per unit and operating point:
  - center of gradient energy;
  - major and minor covariance widths;
  - principal-axis angle;
  - 50%, 80%, 90%, and 95% enclosed-gradient radii;
  - edge energy and truncation diagnostics;
  - lag-wise energy and effective temporal support.
- Repeat across representative images and scored frames so stimulus-dependent
  variation is visible rather than hidden in one linearization.
- Compare composite estimates against the readout-only quantities already saved
  by `audit_rr100_spatial_filter_pooling_scales.py`.

The surrogate should match a distribution of paired major/minor widths. It
should not substitute one median circular radius for the population.

### Required artifacts

- effective-RF map sheet for multiple units, images, frames, and lags;
- per-measurement and per-unit CSV tables;
- distribution and quantile figure in stimulus pixels, degrees, and arcminutes;
- comparison with readout-only widths;
- manifest recording model checkpoint, source inputs, frames, and gradient
  construction.

### Human checkpoint 1

Before surrogate synthesis, decide whether the composite maps are spatially
localized and stable enough to define pooling windows. Inspect anisotropy,
stimulus dependence, lag dependence, and edge truncation. If the effective RFs
are not well summarized by Gaussian ellipses, retain the measured normalized
energy maps as pooling kernels rather than forcing an elliptical approximation.

## 5. Stage 2 — implement the simplest RF-local power optimizer

### Optimization variable and initialization

- Optimize the static source image, not the rendered frames.
- Initialize it with a correctly Hermitian global random-phase version of the
  intact source crop.
- Render the full genuine 72-frame, 151×151 movie after every relevant update:
  32 frames of history plus 40 scored frames.
- Construct the same lagged model input used in production.
- Bound pixels to the model's valid input range by parameterization or a smooth
  range penalty, not by an unaudited final clipping step.

### First-pass power representation

Start with a direct differentiable extension of the current production
SF×orientation×TF calculation rather than immediately building a large new
filter bank:

- retain the same mean removal, temporal Hann window, spatial Hann convention,
  frame rate, SF bins, orientation bins, TF bins, and fitted-support limits;
- calculate the representation within Gaussian or measured-RF spatial pooling
  windows;
- sample window shapes from the composite RR100 distribution;
- evaluate at spatial locations that cover the model-relevant retinal field;
- weight the contribution of pooling scales according to the measured RR100
  distribution rather than treating each arbitrary scale equally.

The optimization should match both:

1. the canonical 40-frame scored-movie power statistic used by the existing
   mechanism analysis; and
2. a response-history guardrail calculated from the genuine 32-frame histories,
   weighted by the measured lag sensitivity.

Match the supported SF×orientation×TF grid and audit out-of-support power
separately. High-TF energy must not be allowed to move into an unreported region
simply because it falls outside the fitted grating support.

### Loss terms

Use separately reported loss components:

\[
L = L_{\mathrm{local\ power}}
  + \lambda_h L_{\mathrm{history\ power}}
  + \lambda_s L_{\mathrm{global\ scored\ power}}
  + \lambda_i L_{\mathrm{intensity}}
  + \lambda_r L_{\mathrm{range}}.
\]

- `local power`: RF-windowed SF×orientation×TF energy mismatch;
- `history power`: mismatch over response-relevant lagged inputs;
- `global scored power`: the existing canonical summary as a guardrail;
- `intensity`: source and retinal mean, RMS contrast, and selected quantiles;
- `range`: out-of-training-range pixel penalty.

Do not match complex coefficients, local signed filter outputs, or phase. Begin
without an explicit intact-phase repulsion term. Audit whether optimization from
random phase preserves phase destruction. Add an explicit phase-decorrelation
constraint only if the power constraints consistently reconstruct intact phase,
and label that as a separate implementation decision.

### Why IAAFT is not the first-line solution

IAAFT alternates spectrum and histogram projections. It can improve a global
histogram/spectrum compromise, but it does not guarantee preservation of the
RF-local response-window power objective and may restore structured phase
relationships. For the first pass:

- bound the optimized pixels;
- match mean and RMS contrast;
- use a small differentiable quantile or histogram loss if needed;
- audit the full histogram afterward.

Do not use final rank-histogram matching without rerunning every power and phase
diagnostic. If a residual contrast mismatch remains scientifically meaningful,
add the double-dissociation control described below.

## 6. Stage 3 — input and mechanism checkpoint

Run only one median-complexity image and one representative trace initially.
Use a small predeclared set of independent random initializations. Do not select
seeds using neural responses.

### Provisional engineering acceptance gates

Freeze final tolerances after measuring numerical variability but before neural
scoring. Initial targets are:

- canonical supported SF×TF cosine similarity at least 0.995;
- total supported-power ratio within 0.98–1.02;
- median RF-local power-bin error below 5%;
- no systematic residual in a particular SF, orientation, TF, or pooling-radius
  stratum;
- comparable agreement across the composite-RF scale distribution;
- source and retinal mean/RMS contrast within approximately 1–2%;
- no out-of-range pixels;
- phase-retention statistics within an independent random-phase null;
- visibly lower translation-aligned edge similarity than intact-versus-small-
  translation controls.

These are engineering gates for determining whether the manipulation is usable,
not thresholds for the biological conclusion.

### Required visualizations

- side-by-side intact and surrogate source images;
- intact and surrogate FEM movies under one fixed display scale;
- corresponding stabilized movies;
- intact-minus-surrogate difference movie;
- eye trace and framewise displacement;
- temporal-variance maps;
- global scored SF×TF maps and differences;
- RF-local SF×orientation×TF residuals for small, median, and large composite
  RF scales;
- intensity histograms and per-frame contrast;
- phase-retention diagnostics:
  - global Fourier phase-vector coherence;
  - adjacent-frequency phase-difference coherence;
  - complex-pyramid within-band and cross-scale phase relations;
  - maximum translation-aligned image, edge, and SSIM similarity.

### Human checkpoint 2

Verify by eye that the surrogate does not merely translate, blur, or weakly
jitter intact edges. Confirm that the power agreement is local, response-window
specific, and present throughout the model's RF distribution. Neural scoring is
not authorized by completion of the optimizer alone.

Failure branches:

- **Power matches but intact edges reappear:** the constraints overdetermine
  phase at the chosen pooling density. Reduce redundant window-center density
  or test whether the model-derived power representation is intrinsically close
  to phase-determining; do not call the output phase-scrambled.
- **Phase is destroyed but power does not match:** the control remains
  non-diagnostic. Improve the optimizer or representation before neural use.
- **Contrast or histogram remains different:** add a phase-preserving contrast
  control or a differentiable histogram term, then repeat the complete audit.
- **Direct local FFT causes seams or unstable gradients:** advance to the
  complex-filter implementation below.

## 7. Stage 4 — sophisticated local representation if required

If the direct localized-FFT version fails for representational rather than
optimization reasons, replace it with a complex spatial-orientation pyramid
crossed with a temporal-frequency filter bank:

- use multiple spatial scales and orientations;
- retain complex quadrature pairs;
- pool squared magnitudes/energy with the measured composite-RF kernels;
- filter the full history-plus-score sequence temporally;
- match pooled energy but not complex phase;
- reconstruct or directly optimize the source image through the differentiable
  renderer.

This is the closest adaptation of the foveated-metamer method. The essential
changes are model-derived pooling, explicit temporal filters, genuine FEM
translation, and exact production history. Re-run all Stage-3 gates; the more
sophisticated representation does not inherit validity automatically.

## 8. Stage 5 — targeted activation-map checkpoint

Only after the input control passes, run a targeted neural render for a small,
audibly selected unit set. Define selection roles before inspecting surrogate
responses.

Suggested roles are:

- strong intact SSI sharpening unexplained by the current power model;
- power-consistent SSI or activation-map example;
- high predicted power shift with weak SSI change;
- weak predicted power shift with strong SSI change;
- weak-effect or negative control;
- units spanning small, median, and large effective-RF scales.

Save `selected_units.csv` with unit identity, role, criterion, criterion value,
reference condition/frame, RF scale, tuning metadata, mean rate, and intact SSI
effect. Clearly distinguish algorithmic roles from any user-requested examples.

### Required maps and quantities

- raw activation maps for all four conditions at multiple frames;
- FEM-minus-stabilized difference maps within each source;
- intact-effect minus surrogate-effect difference maps;
- full time-resolved map sheets for selected units;
- instantaneous SSI beside the map from which it is calculated;
- expected-spike-weighted mean instantaneous SSI;
- separately labelled SSI of the trajectory-averaged map;
- mean rate, expected spikes, raw information numerator, bits per second where
  defined, and bits per spike;
- model response and SSI time courses.

Use matched spatial extents and comparable color scales. If a per-unit scale is
needed to see structure, state this and retain a shared-scale companion panel.

### Human checkpoint 3

Decide from the raw maps whether the phase surrogate preserves, reduces, or
qualitatively changes the FEM-induced sharpening. Inspect dissociations and
negative controls rather than advancing based on one attractive unit.

## 9. Contrast double dissociation

If the accepted surrogate retains a meaningful contrast or histogram mismatch,
add a fifth source condition that preserves intact phase structure while matching
the surrogate's contrast statistics. Render both its stabilized and FEM movies.

The intended dissociation is:

1. phase destroyed, contrast controlled as closely as possible;
2. phase preserved, contrast reduced to the surrogate level.

Interpret phase sensitivity only if the loss of SSI follows phase destruction
rather than the matched contrast reduction. A direct contrast sweep can be added
as a secondary diagnostic because SSI may be relatively invariant to modulation
amplitude, but this should be demonstrated rather than assumed.

## 10. Stage 6 — scaling and population inference

Proceed only if the accepted input manipulation is interpretable and the
targeted maps establish what the comparison means.

Scale in the following order:

1. three images spanning low, median, and high structural coherence;
2. several traces spanning low, median, and high supported dynamic power;
3. multiple independently initialized surrogate sources per image–trace pair;
4. all RR100 units for the accepted stimulus set;
5. a broader balanced image–trace sample only after runtime and failure rates
   are known.

Keep surrogate seed as repeated-stimulus uncertainty rather than choosing the
seed with the preferred neural result. Report optimization failures and rejected
inputs with their predeclared reasons.

The primary population endpoint is the paired difference

\[
\Delta SSI_{\mathrm{phase\ residual}}
=\Delta SSI_{\mathrm{intact}}
-\operatorname{mean}_{s}\Delta SSI_{\mathrm{surrogate},s},
\]

with image, trace, unit, and surrogate-seed structure preserved. Also report
absolute quantities and paired differences for rate and the raw information
terms. Do not reduce the analysis to bits-per-spike SSI alone.

Every population result must trace back to:

- the accepted input audit and synthesis configuration;
- the effective-RF scale artifact;
- the exact renderer and trace provenance;
- the selected-unit table and map-level checkpoints;
- the individual surrogate seeds and their power/phase/contrast diagnostics.

## 11. Interpretation table

| Input outcome | Neural outcome | Interpretation |
|---|---|---|
| Power matched, phase destroyed | SSI sharpening persists | Consistent with RF-local power being sufficient for the measured effect |
| Power matched, phase destroyed | SSI sharpening is reduced or lost | Supports an additional phase-sensitive mechanism, subject to contrast control |
| Power mismatched | Any response result | Non-diagnostic because the temporal-power mechanism was not controlled |
| Phase relationships reconstructed | SSI sharpening persists | Non-diagnostic because the surrogate no longer isolates phase |
| Phase result follows contrast control | SSI changes with contrast in both branches | Contrast, not phase, is the parsimonious explanation |
| Different units show different outcomes | Structured dissociation by tuning/RF scale | Evidence for heterogeneous mechanisms; preserve rather than collapse it |

## 12. Immediate next action

Implement Stage 1 only: the composite input-space effective-RF audit. Present
the maps, scale distribution, surprises, and recommended pooling-kernel
parameterization at Human checkpoint 1. After approval, synthesize one
median-image/representative-trace surrogate and save the Stage-3 eye-check movie.
Do not score digital-twin responses before that movie passes the input-level
checkpoint.

## 13. Relevant existing implementations and artifacts

- Current scored-movie spectral statistic:
  `declan/fig4_active_sensing/run_interim_input_spectral_cache.py`
- Exact input-only retinal renderer:
  `declan/fig4_active_sensing/input_only_retinal_renderer.py`
- Readout-only pooling-scale audit:
  `declan/fig4_active_sensing/audit_rr100_spatial_filter_pooling_scales.py`
- Existing global 3-D explicit-history diagnostic:
  `declan/fig4_active_sensing/make_rr100_global_3d_phase_scramble_explicit_history_checkpoint.py`
- Global phase-ensemble eye-check movie generator:
  `declan/fig4_active_sensing/make_rr100_global_3d_phase_ensemble_movie.py`
- Hann/localization decomposition:
  `declan/fig4_active_sensing/audit_rr100_global_3d_phase_hann_decomposition.py`
- Existing local steerable-pyramid diagnostic:
  `declan/fig4_active_sensing/make_rr100_local_pyramid_phase_scramble_checkpoint.py`
- Broader FEM power-routing protocol:
  `declan/fig4_active_sensing/FEM_POWER_ROUTING_ANALYSIS_PLAN.md`

