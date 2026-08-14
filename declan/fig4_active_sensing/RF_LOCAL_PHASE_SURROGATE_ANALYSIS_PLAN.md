# Conditional analysis plan: RF-local phase surrogates after the map-support test

Date: 2026-08-13; reframed as conditional follow-up 2026-08-14

Status: conditional follow-up; checkpoints 49 and 50 rejected for the primary
question; the map-support amplitude-by-phase factorial in
`FEM_POWER_ROUTING_ANALYSIS_PLAN.md` now runs first

Scope: corrected natural-image retinal movies, genuine 32-frame model history,
40 scored frames, RR100 digital-twin activation maps, firing rates, and spatial
selectivity index (SSI)

## 0. Reframing decision

The original plan matched power inside windows approximating the effective RF
of one translated readout position. That was a defensible construction for a
perceptual metamer or a within-position sufficiency test, but it was not the
smallest justified support for the measured outcome. The digital twin produces
each unit's activation map by convolving that unit's readout across the full
core feature map. SSI is then calculated across those translated positions.
The input object driving one scored map is therefore the complete 32-frame by
151-by-151 history cube and the union of spatial support across map positions.

The primary phase test is now Stage 2A of
`FEM_POWER_ROUTING_ANALYSIS_PLAN.md`: a map-support amplitude-by-phase
factorial. It preserves or replaces the complete raw history-cube Fourier
magnitude and phase, uses one random phase field shared across paired power
conditions, and includes the complementary FEM-phase-preserved/stabilized-power
arm. This directly tests cross-position phase organization and avoids
renderer-in-the-loop metamer optimization.

Everything below is retained as a documented conditional branch. Run it only
if the map-support factorial shows that global power is insufficient and the
remaining scientific question specifically requires distinguishing local-power
redistribution among tiles from nonlinear phase sensitivity within a tile. Do
not interpret the completed RF-scale audit as a requirement that every
whole-map phase control match power within 2.97-pixel windows.

## 1. Scientific objective

The current FEM-versus-power analyses indicate that fixational eye movements
redistribute retinal image power across spatial and temporal frequencies. This
power redistribution explains part of the shared response modulation, but the
current power predictors do not explain the population differences in spatial
SSI sharpening.

The missing experiment asks whether the unexplained SSI effect depends on phase
structure:

> After a map-support control shows that global power is insufficient, does the
> residual remain when receptive-field-local spatiotemporal power is also held
> fixed while phase structure is destroyed?

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

### 2.1 Earlier source/movie-global controls did not use the map-input support

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

Consequently, the earlier 72-frame global 3-D scramble remains an upper-bound
or out-of-distribution diagnostic rather than the decisive phase test.
Seed-searching for a favorable scored-window realization is also rejected
because it conditions the null on a different, phase-sensitive support.

This failure does not apply to the new Stage-2A map-support construction. The
new transform and the exact power claim both refer to the same raw 32-frame
history cube consumed by one activation-map computation. No subsequent Hann
window or 40-frame restriction defines its primary equality contract.

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

## 3. Conditional RF-local experimental construction

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

## 9. Complementary power-by-phase dissociation

The complementary arm is now primary rather than a reactive contrast control.
Follow the Stage-2A factorial in `FEM_POWER_ROUTING_ANALYSIS_PLAN.md`:

1. FEM power with FEM phase;
2. FEM power with shared random phase;
3. stabilized power with FEM phase;
4. stabilized power with the same shared random phase.

This design separately measures phase effects at each power level, power effects
at each phase level, and their interaction. Generic contrast reduction or
spectral whitening is not an interchangeable definition of "power removed."
The promoted power-reduced arm replaces the complete FEM history-cube amplitude
spectrum with the stabilized amplitude spectrum while retaining FEM phase.

If a later RF-local surrogate changes contrast or histogram despite its local
power objective, add a phase-preserving contrast-matched sensitivity condition,
but keep it distinct from the primary amplitude-by-phase factorial. Do not force
factorial terms into percentages that sum to the original SSI effect when the
twin response is nonlinear.

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

This is a conditional RF-local extension of the authoritative Stage 2A table in
`FEM_POWER_ROUTING_ANALYSIS_PLAN.md`. If the two documents differ, use the
primary plan's Stage 2A logic; in particular, its symmetric phase-support and
structural out-of-distribution gates apply before any row below is interpreted.

| Input outcome | Neural outcome | Interpretation |
|---|---|---|
| Global map-input power matched, phase destroyed | SSI sharpening persists | Strong evidence that global map-input power is sufficient for the example |
| Global map-input power matched, phase destroyed | SSI sharpening is reduced or lost | Phase-dependent localization beyond global power matters; within-RF phase sensitivity remains unresolved |
| Stabilized power, FEM phase retained | SSI follows the power-reduced arm | Power reduction is sufficient to reproduce the change under preserved FEM phase |
| Factorial phase effect remains at both power levels | SSI follows phase rather than power | Evidence for a phase contribution, with the interaction reported separately |
| RF-local power also matched, phase destroyed | Residual SSI change remains | Conditional evidence for within-position phase sensitivity |
| Power mismatched | Any response result | Non-diagnostic because the temporal-power mechanism was not controlled |
| Phase relationships reconstructed | SSI sharpening persists | Non-diagnostic because the surrogate no longer isolates phase |
| Phase result follows contrast control | SSI changes with contrast in both branches | Contrast, not phase, is the parsimonious explanation |
| Different units show different outcomes | Structured dissociation by tuning/RF scale | Evidence for heterogeneous mechanisms; preserve rather than collapse it |

## 12. Current checkpoint status and immediate next action

Stage 1 is complete. The composite input-space RF audit supports a first-pass
median circular Gaussian sigma of 2.97 pixels; the other four measured scale
quantiles remain held-out audits.

The first Stage-2 implementation, based on Gaussian-windowed local FFT
magnitudes, was rejected at checkpoint 49. It matched the canonical global
statistic well but generalized poorly to offset pooling locations and visibly
reconstructed a local building/edge complex. This is consistent with dense
overlapping STFT magnitudes acting as a phase-retrieval constraint.

Checkpoint 50 therefore uses global, undecimated complex steerable-pyramid
filtering followed by temporal quadrature energy and only then Gaussian spatial
pooling. For the one predeclared image/trace run:

- score-40 pooled-energy cosine is 0.943 on the training grid and 0.925 on the
  held-out offset grid;
- canonical supported-power cosine is 0.997, with a power ratio of 0.977;
- full supported positive-TF power is not treated as a consequence of phase
  destruction: its ratio is explicitly constrained/audited and equals 1.061
  when all positive-TF power is included;
- source Fourier phase-retention coherence is 0.003;
- the held-out local Fourier-phase coherence median is 0.103, but adjacent
  frequency-relation coherences are about 0.37;
- the held-out local edge-correlation median is 0.253, and a recognizable local
  facade/edge complex is visible near the center of the surrogate.

Checkpoint 50 is rejected for the primary whole-map phase question. It remains
useful provenance showing that dense RF-local energy constraints can recreate
recognizable structure. Do not tune its window density further before the
simpler map-support experiment.

The immediate next action is Stage 2A of
`FEM_POWER_ROUTING_ANALYSIS_PLAN.md`: synthesize the exact history-cube
amplitude-by-phase factorial for one development image, one clean-history trace,
one scored frame, several predeclared shared-phase seeds, and a small unit set;
then stop after the input and raw activation-map checkpoint. Return to this
RF-local branch only if that factorial produces an informative loss of SSI that
cannot be interpreted without controlling translated local power.

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
