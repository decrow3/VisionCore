# declan Analysis Narrative

Last curated: 2026-06-09.

Companion to `MANIFEST.md`. The manifest answers "where is it?" This file
answers "why did we do it, what happened, and how did later work change the
interpretation?"

This is a living synthesis from the markdown plans, handoffs, READMEs, and
result notes in `declan/`, plus the run summaries under `outputs/` when they
closed a thread. It deliberately keeps both the at-the-time interpretation and
the later revision when a control or follow-up narrowed the claim.

## Reading Rules

- `Closed` means there is enough result evidence to treat the thread as
  resolved for current purposes.
- `Promoted` means the thread has a result that can plausibly carry a figure or
  manuscript claim, with the stated guardrails.
- `Supportive` means useful evidence, but not a standalone headline.
- `Historical` means useful context or machinery, but superseded by a later
  framing or control.
- `Open` means a plan/spec exists, and sometimes code exists, but the analysis
  is not yet fully interpreted.

## Current Synthesis

The story has moved through three broad phases:

1. Early FEM population-coding work found a real E-optotype crossover and a
   tempting temporal/covariance story, but later controls narrowed the mechanism
   to first-order spatial sampling in the mean-rate code.
2. The Jacobian/translation-covariance work rescued the structural part of the
   idea: image translation directions robustly organize FEM-linked covariance,
   even when magnitude identities and temporal-code interpretations fail.
3. The newest Figure 4/Figure 5 split is cleaner: Figure 4 is about compact
   reafferent retinal-translation geometry and recorded covariance closure;
   Figure 5 is about active-sensing movie information efficiency, with explicit
   limits on claims of real-trajectory optimality.

## 2026-06-09: Active-Sensing Movie Information / Figure 5

Status: `Promoted`, with strong claim discipline.

Primary docs and outputs:

- `active_sensing_movie_information/README.md`
- `active_sensing_movie_information/active_sensing_movie_information_plan.md`
- `active_sensing_movie_information/figure5_additional_checks_prep.md`
- `Figure5_active_sensing_triage_plan.md`
- `outputs/active_sensing_movie_information/active_sensing_movie_information_figure/active_sensing_movie_information_figure.{png,pdf,svg}`
- `outputs/active_sensing_movie_information/active_sensing_movie_information_figure/active_sensing_movie_information_figure_caption.md`
- `outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu/`

Motivation:

After the E-optotype/covariance path became too easy to overstate, this thread
reframed active sensing around natural-image retinal movies. The central
question became: do measured FEM-like retinal movies improve a deterministic V1
twin's spatial information efficiency relative to stabilization, and what image
statistics explain that gain?

At-the-time plan:

- Use Jake's `jake.twininfo` production pipeline as the source of truth.
- Treat cumulative spatial SSI bits per expected spike as the primary endpoint.
- Use raw bits, bits/sec, expected spikes, Fisher, and retinal transform QC as
  companions, not as the primary claim.
- Separate three claims that had been blurring together:
  measured FEMs explain recorded V1 shared variability; retinal motion improves
  a model information proxy; and the animal's exact trajectories are uniquely
  useful.

Main outcome:

- Real FEM retinal motion improved final spatial information efficiency over
  stabilization by about `+0.035` bits/expected spike.
- The saved caption reports real increasing the endpoint from `0.110` to
  `0.145` bits/expected spike, with 95% CI `[0.026, 0.045]`.
- The real-minus-stabilized cumulative curve stayed positive at all sampled
  time points in the figure summary.
- Raw information and expected spike count also increased, but the bits/spike
  endpoint survived spike-count normalization.
- Spatial-frequency controls showed a graded mechanism: lowpass produced a
  small gain, mid/high SF bands produced larger gains.

Important later interpretation:

- Random trajectory controls matter. `random_amp`, `random_cov`, and especially
  `random_amp_cloud_matched` equaled or exceeded real FEMs on the current
  bits/expected-spike endpoint.
- The current safe claim is: real retinal motion improves model spatial
  information efficiency over stabilization through a spectral-temporal
  mechanism.
- The current unsafe claim is: real FEM trajectories are optimal.
- Natural-image-only Checks 5-9 supersede the old cached e-optotype scaffolds
  for Figure 5 evidence.

Historical scaffold:

- `outputs/active_sensing_movie_information/figure5_cached_rate_checks_5_to_9_fixed_lm-020/`
  and the Check 8 add-back run are useful debugging context.
- The cached e-optotype scaffold found real residual structure more aligned
  with stimulus axes than stabilized, higher covariance-efficiency ratio `eta`,
  and positive remove-out effects. But matched/null controls also improved in
  places, and the stimulus was synthetic E-optotype rather than natural image.
- Do not promote those e-optotype checks as Figure 5 evidence.

Open follow-ups:

- Natural-image population Checks 5-9 are the current route for constrained
  population coding, pose-aware recoverability, and amplitude/diffusion sweeps.
- Compact add-back/remove-out should wait until the compact basis is
  dimension-compatible with the natural-image center-channel response space.

## 2026-06-09 / 2026-06-08: Compact Retinal-Translation Geometry

Status: `Promoted`, with some panels still carrying explicit caveats.

Primary docs and outputs:

- `compact_retinal_translation_geometry/README.md`
- `compact_retinal_translation_geometry_implementation_spec.md`
- `outputs/compact_retinal_translation_geometry/`
- `outputs/compact_retinal_translation_geometry/tables/acceptance_matrix.csv`
- `outputs/compact_retinal_translation_geometry/figures/panelA_local_translation_charts.{png,pdf}`
- `outputs/compact_retinal_translation_geometry/figures/panelB_compact_tangent_spectrum.{png,pdf}`
- `outputs/compact_retinal_translation_geometry/figures/panelC_cross_image_generalization.{png,pdf}`
- `outputs/compact_retinal_translation_geometry/figures/panelE_covariance_closure_full_vs_compact.{png,pdf}`
- `outputs/compact_retinal_translation_geometry/figures/metric_structure_summary.{png,pdf}`

Motivation:

This was created to turn the Figure 4 tangent/covariance material into a
coherent hidden-coordinate-style result: small retinal translations produce
image-dependent response changes, but those changes live in a compact,
image-generalizing population geometry that predicts recorded FEM covariance.

Panel logic:

- A: image-dependent local translation charts.
- B: compact tangent spectrum.
- C: cross-image tangent generalization.
- D: variability budget / denominator context.
- E: recorded covariance closure, full finite-difference source versus compact
  k=10 source.
- Metric validation: the coordinate-like hidden-geometry test.
- F / decoding bridge: optional, promote only if recorded displacement decoding
  survives leakage and null checks.

Main outcomes:

- Panel B compactness passed: observed participation ratio was about `9.04` at
  `0.25` arcmin, far below the unit-shuffle samplewise null around `31.0`.
- Panel C generalization passed: an image-disjoint compact basis at `k=10`
  captured about `0.525` held-out tangent variance versus null around `0.122`.
- Panel E covariance closure passed: full finite-difference translation sources
  predicted recorded FEM covariance above unit-shuffle and RF/readout nulls.
- Compact k=10 retained the closure. The compact-to-full capture ratio at k=2
  was about `1.005`, so restricting to the compact source did not cost the
  closure effect.
- Under the conservative `global_rate+target_pc1` projection, PSD full
  finite-difference source at `k=10` captured about `0.535`; the compact
  cross-fit source captured about `0.536`, with positive effects over RF/readout
  fixed-permutation nulls in 24/24 sessions.
- Metric structure has partial support: rank-2 local compact metrics pass, and
  displacement scaling is strong (`R2 ~0.995` for norm/metric scaling), but
  coordinate recovery and diagonal composition are not fully landed because the
  current cache only has cardinal `+/-x` and `+/-y` translations.

Important caveats:

- Do not claim a universal literal 2D eye-position map in V1.
- Do not claim behavior or perceptual optimality from these panels.
- Do not claim the compact spectrum survives RF/readout-preserving samplewise
  null or projection-control spectrum until those are explicitly run.
- Do not promote recorded displacement decoding yet; the acceptance matrix
  marks Panel F decoding as not run for promotion, despite smoke/prod machinery
  existing.

Later interpretation:

This is the current structural spine for Figure 4. It absorbed the safer parts
of the older TFTS, covariance closure, and recorded-derivative branches, while
keeping performance/active-sensing claims separated into Figure 5.

## 2026-06-08: Direct Recorded Derivative / Twin Tangent Alignment

Status: `Supportive`.

Primary docs and outputs:

- `direct_recorded_derivative_twin_alignment/README.md`
- `direct_recorded_derivative_twin_alignment_prescription.md`
- `outputs/direct_recorded_derivative_twin_alignment_prod/README.md`
- `outputs/direct_recorded_derivative_twin_alignment_prod/tier1_compact_basis_bootstrap_summary.csv`

Motivation:

The covariance-closure result asks whether fitted-twin finite-difference
translation covariances predict recorded `Sigma_FEM`. This branch asked a more
direct but noisier question: if we estimate eye-position derivatives directly
from recorded V1 repeats, do those derivatives lie in the compact fitted-twin
translation geometry?

At-the-time guardrail:

Do not try to resurrect a clean image-specific signed `x/y` derivative match
between recording and twin. Older STG work showed signed/context-specific
derivative recovery was fragile. The primary claim should be compact-subspace
enrichment, not signed-axis recovery.

Outcome:

- Tier 1 survived the conservative control in the eligible-session set.
- Primary condition: `target_variant=psd`,
  `projection_control=global_rate+target_pc1`,
  `context_subset=reliability_qualified`, `k=10`.
- Capture mean was `0.386`.
- Effect over RF/readout null was `+0.210`, CI `[0.178, 0.246]`.
- Effect over unit-shuffle null was `+0.288`.
- Effect over random-subspace null was `+0.284`.
- Sign consistency was 13/13 eligible sessions, sign-test p `0.000244`.

Interpretation:

This is a supportive direct recorded-data bridge: recorded eye-position
sensitivity is enriched in the compact twin tangent subspace. It strengthens
the compact-geometry story but does not supersede covariance closure and should
not be phrased as signed horizontal/vertical axis recovery.

## 2026-06-07 / 2026-06-08: Matched Twin Covariance Closure

Status: `Promoted`.

Primary docs and outputs:

- `matched_twin_covariance_closure/README.md`
- `matched_twin_covariance_closure/rf_readout_preserving_null_prescription.md`
- `outputs/matched_twin_covariance_closure_finite_difference/`
- `outputs/matched_twin_covariance_closure_rf_null_step025_rfbacked_v2/`

Motivation:

This thread asked whether recorded FEM covariance in Ryan's matched
recorded/twin unit space is captured by fitted-twin eye-position structure and,
more strictly, by fitted-twin finite-difference retinal translation tangents.

At-the-time path:

- Start with cache-only eye-position regression because it was the strongest
  analysis possible from Ryan's Fig2/Fig3 caches alone.
- Replace that proxy with true finite-difference fitted-twin retinal
  translation tangents once model reconstruction was stable.
- Add projection controls and nulls: random subspace, unit shuffle, and later
  RF/readout-preserving nulls.

Outcome:

- The 24-session finite-difference sweep ran successfully.
- PSD `fd_sample_eye_trace_cov`, `k=2`, no projection: mean capture `0.531`,
  mean effect over unit shuffle `0.368`, positive in 24/24 sessions.
- With `global_rate+target_pc1` projection: mean capture `0.220`, mean effect
  `0.177`, positive in 24/24 sessions.
- Bootstrap CIs stayed positive; for PSD samplewise k=2,
  `global_rate+target_pc1` effect was `0.177`, CI `[0.144, 0.212]`.
- Raw target variants were also positive, though PSD is cleaner for
  variance-capture summaries.
- Step-size sensitivity on Allen was stable from 0.25 to 1.0 px.
- The RF/readout-preserving null extension became the stronger reviewer-facing
  version and feeds the compact geometry Panel E.

Interpretation:

This supports a substantial first-order retinal-translation component of
recorded `Sigma_FEM` geometry in matched recorded/twin unit space. It is not a
complete explanation of all FEM covariance.

Later refinement:

The strictest useful wording is now: finite-difference fitted-twin retinal
translation sources, including compact-restricted k=10 sources, predict a
reliable component of recorded FEM-linked covariance above unit and RF/readout
preserving nulls. Avoid "the twin fully reproduces recorded covariance."

## 2026-06-07: Figure 4 Covariance / TFTS Figure Work

Status: `Historical -> Integrated`.

Primary docs and outputs:

- `fig4_cov_TFTS/update.md`
- `fig4_cov_TFTS/covTFTS_figure_panel_prescription.md`
- `fig4_cov_TFTS/covTFTS_figure_data_forward_prescription.md`
- `fig4_cov_TFTS/figure4_panelF_natural_structure_coda_plan.md`
- `outputs/twin_feature_tangent_structure_prod_v2/MANUSCRIPT_REPORT.md`
- `outputs/compact_retinal_translation_geometry/`

Motivation:

This was the first attempt to make a clean Figure 4 out of recorded
reafferent covariance, local translation charts, tangent compactness,
cross-image generalization, and partial covariance bridging.

At-the-time figure claim:

The figure should communicate that recorded V1 shared variability is
reafferent, local retinal translations define image-specific response tangents,
and those tangents form a compact, image-generalizing structure. It should not
claim behavioral benefit or hard-code unfinished Panel E/F interpretations.

Outcomes:

- The tangent-family structural result landed in
  `outputs/twin_feature_tangent_structure_prod_v2/`.
- Production report status: `core_structural_result_passed`.
- At `0.25` arcmin, union compactness PR was `9.04` versus null mean `31.03`.
- Train/test basis at `0.25` arcmin and `k=10` captured median held-out
  variance `0.552` versus null median `0.118`.
- Local first-order covariance approximation showed locality dependence: the
  tangent approximation was more sensible at smaller cloud scales and
  over/under-scaled as the finite cloud grew.

Later interpretation:

The material was too broad as a single ad hoc figure workspace. Its stable
parts were promoted into `compact_retinal_translation_geometry/`. The proposed
Panel F natural-image coda remains conceptually useful but optional: it should
only enter the main figure if high-structure natural patches preferentially
route drift-scale response changes through the compact tangent basis above
matched controls. Otherwise, keep it as supplement or cut it.

## 2026-06-04: Natural Image Tangent Scale

Status: `Open`.

Primary docs:

- `Natural_Image_Tangent_Scale_Analysis_Handoff.md`
- `natural_image_tangent_scale/run_natural_image_tangent_scale.py`

Motivation:

TFTS showed that small retinal translations produce compact,
image-generalizing tangent structure. This follow-up asks how far the local
tangent description remains valid before finite displacement leaves the local
linear regime, and whether that breakdown scale depends on natural-image
structure.

Key guardrail:

Because the twin was trained with FEM-jittered retinal inputs, an absolute
match between tangent breakdown scale and FEM amplitude could be circular. The
non-circular gate is image-structure dependence: breakdown scale must vary
systematically with natural-image structure before making an ecological claim.

Outcome so far:

The module and runner exist, but this thread is not yet summarized as a closed
result in the `declan/` docs. Treat as an open ecological-anchor analysis.

Interpretation if it lands:

- If breakdown scale depends on gradients/SF/structure, the local compact
  geometry is tied to image curvature rather than only to the model's training
  eye-jitter distribution.
- If breakdown scale is flat and merely near FEM amplitude, report the scale
  gate as failed and do not compare to empirical FEM amplitudes.

## 2026-06-04 / 2026-06-03: Twin Feature Tangent Structure

Status: `Promoted`, now largely folded into compact geometry.

Primary docs and outputs:

- `Twin_Feature_Tangent_Structure_Prescription.md`
- `twin_feature_tangent_structure/run_twin_feature_tangent_structure.py`
- `outputs/twin_feature_tangent_structure_prod_v2/MANUSCRIPT_REPORT.md`

Motivation:

Earlier signed cross-image tangent alignment was too strict and could be near
zero, because translation tangents are image-specific. This pivot asked whether
the conserved object is not a signed universal `x/y` axis, but a compact
feature-defined tangent subspace, metric law, or operator family.

At-the-time claim:

Different images generate different translation tangents, but those tangents
are produced by a shared feature operator and may live in a compact,
image-generalizing subspace.

Outcome:

- Core structural stop rule passed.
- Union compactness was well above null at all tested deltas.
- Train/test generalization passed across folds and k values.
- The output report labels the claim state as
  `core_structural_result_passed`.

Later interpretation:

This is the first-order mechanism behind the compact retinal-translation
geometry. It should be described structurally: first-order tangents occupy a
compact shared subspace and generalize across images. It should not be turned
into a behavioral, optimality, or decoder claim.

## 2026-06-03: Shared Transformation Geometry

Status: `Historical / partially superseded`.

Primary docs and outputs:

- `shared_transformation_geometry_handoff.md`
- `shared_transformation_geometry_handoff_v2.md`
- `shared_transformation_geometry/README.md`
- `outputs/twin_covariance_structure/shared_transformation_geometry/`

Motivation:

STG asked whether recorded V1 contains a conserved retinal-transformation
geometry across images beyond trivial displacement magnitude and image
similarity. It was an ambitious recorded/twin bridge for signed tangent maps,
twin-template matching, and residual RDM geometry.

At-the-time correction:

The early RDM framing was demoted because an RDM is symmetric and cannot
distinguish signed displacement direction. Signed tangent-map comparison became
the primary signed test; residual RDM geometry became secondary/diagnostic.

Outcome / lessons:

- The infrastructure produced support census, tangent-map, template-match,
  residual RDM, and aggregation runners.
- It established useful patterns: session-level inference, support census
  first, image-similarity controls, drift-only masking, and explicit
  `control_not_evaluable` labels.
- It also exposed fragility: clean signed image-specific recorded derivative
  manifolds and exact recorded/twin signed-axis matches were not robust enough
  to headline.

Later interpretation:

STG became a reference layer rather than the final claim vehicle. Its safer
ideas were absorbed into direct recorded derivative Tier 1 and compact
covariance closure. Do not resurrect the strong signed-axis STG claim without
new evidence.

## 2026-06-03: Twin Covariance Structure

Status: `Supportive / framing pivot`.

Primary docs and outputs:

- `Twin_Covariance_Structure_Prescription.md`
- `twin_covariance_analysis_plan.md`
- `twin_covariance_structure/README.md`
- `outputs/twin_covariance_structure/`

Motivation:

This prescription separated what the deterministic twin can answer from what
requires a noise model. The twin is a good instrument for structure: low rank,
signal alignment, image specificity, occupancy dependence, translation tangent
alignment, and single-neuron-to-population bridges. It is not a good standalone
instrument for whether FEM covariance helps or hurts coding.

At-the-time interpretation:

The recording proves reafferent covariance exists and dominates positive noise
correlations; the twin explains why that reafferent covariance has its
structure.

Core conceptual outcomes:

- Signal alignment is not automatically a catastrophe; moving the image and
  changing the image drive should overlap in response space.
- Low rank should be tied to 2D translation plus finite-cloud curvature.
- Image specificity distinguishes reafference from global state.
- Occupancy, not trajectory order, governs second-moment covariance structure.

Later interpretation:

This prescription was important because it stopped the twin from being treated
as a contested performance oracle. Its structural pieces feed Figure 4 and
compact geometry. Functional/information claims are kept separate and require
explicit noise/readout assumptions.

## 2026-06-01: Keystone / Geometry-Crossover Link

Status: `Open / adjudication plan`.

Primary docs:

- `Keystone_Geometry_Crossover_handoff_v2.md`
- `Keystone_Geometry_Crossover_handoff_v3.md`
- `bigpicture_fem_v1_high_impact_analysis_plan_v2.md`
- `bigpicture_phase1_fem_v1_coding_agent_plan_v2.md`

Motivation:

After the E-optotype crossover and translation geometry were both in hand, the
keystone thread asked for the missing link: does a decoder-free geometry
quantity predict the sign, transition LogMAR, and magnitude of the FEM accuracy
advantage?

At-the-time design:

- Tier 1: cloud-separability gain `G_sep`, computed from deterministic mean
  responses over real versus stabilized position clouds.
- Tier 2: Jacobian/tangent mechanism `DeltaM`, asking whether translation
  mimicry/tangent escape tracks the same transition.
- The firewall: geometry observables cannot use decoder outputs or noise models,
  otherwise the test is circular.

Current interpretation:

This is an adjudication plan, not a closed result in `declan/`. It is valuable
because it formalized a clean distinction:

- `geometry_predicts_global_crossover`: geometry predicts and explains the
  functional crossover via tangent mechanism.
- `geometry_predicts_crossover_via_sampling_not_tangent`: cloud sampling
  predicts the crossover, but the equivariant tangent story does not.
- `geometry_tracks_difficulty_not_mechanism`: geometry is just difficulty.

Given later Figure 4/Figure 5 separation, this plan is less central than it was
when Figure 4 was trying to carry the active-sensing crossover.

## 2026-05-29: Jacobian Audit / Predictive Framework

Status: `Historical -> partially rescued as structure`.

Primary docs and outputs:

- `jacobian_results/results_and_interpretation.md`
- `jacobian_predictive_framework_progress_summary.md`
- `jacobian_predictive_framework_handoff_revised.md`
- `eoptotype_jacobian_field_smoothness_handoff.md`
- `fem_path_integrated_separability_handoff.md`
- `outputs/stats/eoptotype_jacobian_field_*`
- `outputs/stats/fem_step_jacobian_*`

Motivation:

The original Jacobian hypothesis was that FEM-induced covariance might be
predicted by a first-order pushforward:

```text
C_FEM ~= J Sigma_eye J.T
```

The appeal was strong: it would connect eye motion, local image translation,
and population covariance in one equation.

Outcome:

- Direction worked robustly. The image-translation Jacobian captured the
  leading FEM covariance subspace with alignment roughly `0.40-0.60`, 2-4x
  above null.
- Magnitude was fragile. Naive `J_static x Sigma_frame` overpredicted by
  `6-490x`; `J_eff x Sigma_trial` underpredicted by `0.003-0.053x`.
- Position-histogram integrated `J_int x Sigma_total` got near scale agreement
  at `lm=-0.20` for three of four orientations, but remained off at
  `lm=-0.40`, consistent with grid resolution being too coarse for tiny E
  strokes.
- Representational intervention with stimulus-specific J could raise decoding
  to 100%, but the class-specific nature of that manipulation made it too easy
  to overinterpret. Pooled-J controls were safer and less dramatic.

Later interpretation:

The magnitude identity is a closed/failed branch at full cloud scale. The
directional/subspace result survived and became the right way to use Jacobians:
translation tangents define the geometry, but do not by themselves provide a
full covariance magnitude identity.

The smoothness work also changed the story: the issue was not a wildly rough
Jacobian field. Instead, finite-cloud scale, phase, curvature, and resolution
explain much of the magnitude mismatch.

## 2026-05-26: E-Optotype Hyperacuity, Crossover, and Covariance Ablations

Status: `Closed`, with a narrowed mechanism.

Primary docs and outputs:

- `revised_analysis_plan.md`
- `FEM_population_coding_writeup.md`
- `fem_eoptotype_hyperacuity_results.md`
- `fem_covariance_geometry.py`
- `fem_global_intervention.py`
- `fem_differential_intervention.py`
- `eoptotype_continuous_pass.py`
- `declan/fem_covariance_geometry_results/`
- `declan/fem_global_intervention_results/`
- `declan/fem_differential_intervention_results/`
- `declan/continuous_pass_results/`
- `declan/gru_passthrough_figures/`

Motivation:

This was the first functional active-sensing arc: real FEMs appeared to hurt
orientation decoding at larger E sizes but help near the hyperacuity regime.
The exciting hypothesis was that information might migrate from mean rates into
FEM covariance geometry or temporal trajectory structure.

At-the-time established facts:

- The twin has real temporal processing within its window, and model
  correlations are purely reafferent by architecture.
- D1 time-averaged rate showed a real-vs-stabilized crossover around
  `LogMAR ~ -0.32`.
- Real FEM hurt at `-0.20/-0.25`, became roughly neutral near `-0.30`, and
  helped around `-0.35` to `-0.40` under the windowed pipeline.
- Spatial SSI on E-optotype at `-0.20` increased under real FEM despite
  stabilized outperforming real in orientation decoding, implying a readout/task
  distinction.

Closed outcomes:

- Covariance-code migration was false. FEM subspaces did not rotate with E
  orientation; off-diagonal overlap was ~1.0. Covariance decoders were near
  chance, and combined covariance features added essentially nothing over D1.
- Alignment transition was real: alpha was higher near `-0.20` than `-0.40` in
  real FEM, with stabilized showing the opposite ordering.
- Signal geometry likely moved: `C_signal` eigenvalues were larger at `-0.40`,
  and overlap with translation nuisance directions fell.
- Pooled FEM-subspace ablation improved real at `-0.20` and was null at
  `-0.40`, matching the alpha pattern, but the stabilized control also
  improved. Therefore it removed a generic positional nuisance, not a uniquely
  dynamic-FEM covariance component.
- Differential `C_real - C_stabilized` ablation also failed to isolate a
  real-specific causal covariance mode.
- Temporal coding remained null. D3/temporal residual features did not rescue
  orientation information; continuous forward pass degraded performance and
  stayed below the windowed pipeline.
- Fixed-center was exposed as a deterministic oracle, not a biological static
  baseline.
- Among nonzero FEM amplitudes, larger movements reduced E-orientation decoding
  in this model/readout; no inverted-U or optimal biological amplitude emerged.
- `-0.40/-0.45/-0.50` formed a model-native retinal saturation plateau, so the
  smallest nominal sizes are not independent hyperacuity measurements.

Final interpretation:

The E-optotype crossover is real in the windowed pipeline but the mechanism is
not a temporal code and not a covariance-code migration. It is best read as
first-order spatial sampling in the time-averaged rate code, relative to
trial-mean stabilization. Dynamic FEM can help near the model's resolution
limit by sampling useful nearby retinal phases, but it does not beat a
deterministic fixed-position oracle and should not be framed as optimal active
trajectory selection.

## 2026-05-21: Early FEM / Temporal Decoding / COM Dynamics

Status: `Historical`, with some durable findings.

Primary docs and outputs:

- `results_summary.md`
- `temporal_decoding_analysis_plan_consolidated_v2.md`
- `temporal_decoding_analysis_implementation_plan.md`
- `temporal_decoding_diagnostic_plan.md`
- `com_dynamics.py`
- `transformation_dynamics.py`
- `displacement_decoding.py`
- `eoptotype_continuous_pass.py`
- `translation_covariance.py`
- `declan/displacement_decoding_figures/`
- `declan/transformation_dynamics_figures/`
- `declan/transformation_dynamics_figures/com/`

Motivation:

This was the broad exploration phase: try temporal decoding, velocity
readouts, displacement decoding, COM/spatial moments, transformation dynamics,
and translation covariance to see what FEM-driven population dynamics encode.

Durable outcomes:

- Temporal residual features did not improve over time-averaged rates in the
  orientation task.
- Velocity/transformation variables were not decodable from the tested latent
  or spatial-moment representations under independent-window processing.
- Within-image displacement decoding was near-perfect (`R2 ~0.998-0.999`).
- Cross-image displacement decoding failed badly (`R2 ~ -1.3`), which was
  initially a null for universal displacement decoding but later became a
  positive control for image-specific reafferent geometry.
- CoM/moment features did not beat scalar rates for small displacement
  decoding.

Later interpretation:

The early "temporal dynamics encode transformation" branch did not survive as
an active mechanism. But the displacement-decoding result became crucial: V1
encodes retinal displacement exquisitely within an image, and that code is
content-specific rather than universal. That fact directly feeds the later
TFTS/compact-geometry story.

## 2026-01 to 2026-04: Backimage, Translation Covariance, and Generated Diagnostics

Status: `Historical / artifact base`.

Primary artifacts:

- `translation_covariance/`
- `overnight_backimage_sweeps/`
- `overnight_backimage_long_sweeps_20s/`
- `overnight_backimage_long_sweeps_20s_re/`
- `test_sweeps/`
- `E_diagnostics_human_240ppd/`
- `E_diagnostics_model_37ppd_resnet_none_convgru/`
- `backimage_*`, `hybrid_eye_trace_*`, `fixrsvp_*`, and `spatial_info_*`
  caches.

Motivation:

These were the early data/caching/sweep artifacts that made later work
possible: backimage fixation pools, hybrid eye traces, natural-image sweeps,
E-optotype retinal/model diagnostics, and January translation-covariance
products.

Outcome:

They produced many useful caches and figures, but most are generated artifacts
rather than current analysis entry points.

Later interpretation:

Keep them as provenance and source material. Do not use them as current
manuscript claims without checking which later plan or README superseded their
interpretation.

## Open Claim Boundaries

Current safe claims:

- Recorded FEM-linked covariance is a major, low-dimensional reafferent
  component of V1 shared variability.
- The deterministic twin is useful for explaining the structure of that
  covariance, not for proving perception or optimality.
- Image translation tangents are image-specific but compact and
  image-generalizing as a family.
- Finite-difference fitted-twin translation sources, including compact
  restricted sources, predict a reliable component of recorded FEM covariance
  above strong nulls.
- Real retinal motion improves a V1-model natural-image spatial-information
  efficiency endpoint relative to stabilization.

Claims to avoid unless new evidence lands:

- Real FEM trajectories are optimal.
- V1 has a literal universal 2D eye-position coordinate map.
- FEM covariance fully explains all recorded shared variability.
- The E-optotype crossover is caused by a temporal code or covariance-code
  migration.
- The deterministic twin alone proves whether FEM covariance helps or hurts
  biological visual coding.

## Fast Resume Pointers

If resuming Figure 4 compact geometry:

- Read `compact_retinal_translation_geometry/README.md`.
- Then read `outputs/compact_retinal_translation_geometry/tables/acceptance_matrix.csv`.
- Then check whether `relative_displacement_decoding_prod_gpu1` has completed
  and whether its decoding results pass leakage and null checks.

If resuming Figure 5 active sensing:

- Read `active_sensing_movie_information/README.md`.
- Then read `active_sensing_movie_information/figure5_additional_checks_prep.md`.
- Treat the natural-image-only population checks as current; treat cached
  e-optotype checks as historical scaffolding.

If resuming recorded derivative alignment:

- Read `outputs/direct_recorded_derivative_twin_alignment_prod/README.md`.
- Keep Tier 1 as compact-basis enrichment only; do not headline signed axes.

If resuming the old E-optotype crossover:

- Read `fem_eoptotype_hyperacuity_results.md` before `revised_analysis_plan.md`.
- Assume the final mechanism is mean-rate spatial sampling unless you are
  explicitly testing a new control.
