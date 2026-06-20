# Figure 4 Active Sensing Panel Text Atlas

Status: generated panel-reading document, cache-first.

Use this as the human-facing contact sheet before composing the final Figure 4.
Each entry lists the current rendered subpanel, the intended read, and the
claim boundary or active flags that should travel with it.

For LLM-first review, start with `panel_text_atlas_compact.md`. It embeds one
composite per module plus a main-spine composite, avoiding the many-small-image
parse problem in this detailed contact sheet.

## Main Spine Candidate

Current best compressed story:

```text
A: FEMs convert a fixed screen image into a retinal movie.
B: Empirical drift-like movies add feature-decodable response structure.
C: A joint image-and-eye observer recovers information lost by ignoring motion.
D: Local image geometry defines useful motion axes, with guardrails.
E: Measured free-viewing FEM axes align with local image geometry.
```

Recommended main candidates:

```text
A1/A2/A4, B3/B4, C2/C3, D1/D4, E2/E3
```

Likely supplement or inset candidates:

```text
A5, B2/B5/B6, C4/C5/C6, D2/D3/D5, E1/E4/E5/E6/E7/E8
```

## Expanded Results, Interpretation, Caveats, And Flags

This section is the more verbose scientific reading of the panel set. The
later sections remain a visual contact sheet. The goal here is to separate four
things that can otherwise get tangled together:

```text
1. Result: what the existing cache/result table actually shows.
2. Interpretation: what the result makes plausible in the Figure 4 story.
3. Caveat: what the result does not establish.
4. Flag: the local bookkeeping tag that should follow the claim.
```

### Module A Expanded Read

Result:

```text
Panel A establishes the sensory transformation. During fixation, the screen
image can be fixed while the retinal crop moves. In the frozen retinal-movie
QC pack, real FEM movies have temporal contrast and motion power that are zero
in the matched stabilized counterfactual, while static movie power remains
matched. The BackImage bridge records that the downstream results use 256
images, 29 sessions, 151 drift-only trace sources, and the canonical 756-unit
V1 twin.
```

Interpretation:

```text
This licenses the opening premise of Figure 4: FEM-linked variability should
not be treated only as nuisance noise. It is also a reafferent sensory
transformation, because each small eye displacement samples the same screen
image at a different retinal location. A3 supplies the local linear intuition:
small image translations interact with local gradients, so the consequence of
motion depends on image geometry.
```

Caveats and flags:

```text
F002 remains active because the subpanels are not yet assembled into a final
main-figure composite. A5 is a bridge to earlier covariance evidence, not a
new load-bearing endpoint for this atlas. The covariance bars in A5 use mixed
denominators, so they should stay supplement/bridge unless that caveat is made
central in the caption.
```

### Module B Expanded Read

Result:

```text
In the cleaned BackImage aggregate run, empirical drift-like response movies
add feature-decodable signal beyond static responses. For the temporal-PCA
readout, empirical static-plus-motion gains over static are positive across
the tested scales:

Gabor k=4:
  0.25x +14.31, CI [+7.45, +21.79]
  0.5x  +13.04, CI [+6.81, +20.89]
  1x     +9.10, CI [+3.73, +14.86]

Pyramid k=8:
  0.25x  +5.20, CI [+3.02, +7.68]
  0.5x   +4.89, CI [+2.88, +7.07]
  1x     +3.93, CI [+1.93, +5.86]
```

Interpretation:

```text
This is the first functional result in the atlas. It says that adding
FEM-like temporal samples to a static V1-twin response can improve recovery of
natural-image feature structure. In story terms, the retinal movie is not just
a nuisance perturbation; the response movie carries recoverable image
structure that is absent or weaker in a static-only response summary.
```

Control interpretation:

```text
Empirical drift beats the OU-like confined control robustly. For Gabor k=4,
the empirical-minus-OU contrast is +21.24 at 0.25x, +19.59 at 0.5x, and
+17.16 at 1x. This makes the result stronger than a generic "any tiny OU-like
jitter helps" account.
```

Caveats and flags:

```text
F003: the plotted endpoint is a deterministic feature-decoding proxy in -MSE
units. Do not call it literal mutual information unless a noise/logdet model is
added.

F004: empirical specificity is not uniform across all generic-motion controls.
The empirical-minus-Brownian contrast is strong at 0.25x (+10.52) and 0.5x
(+7.89), but narrows by 1x (+0.51) and is slightly negative at 2x (-0.60).
This argues for scale/readout-specific wording.

F005: the local exact image-trace pairing branch is not ready to carry the
headline. The main B claim is distributional over image/trace families.
```

### Module C Expanded Read

Result:

```text
The exact finite trajectory-table observer shows the expected ordering:
known-eye is highest, zero-eye drops when motion matters, and joint-eye
inference recovers much of the lost image identity by marginalizing over
plausible trajectories. In the matched-static-response condition at 1.0x:

known-eye = 1.000
zero-eye = 0.328
joint-eye empirical prior = 0.766
joint-eye OU prior = 0.797
```

Interpretation:

```text
This is the conceptual center of the atlas. Module B says the response movie
contains feature structure. Module C asks whether that structure can still be
used when the observer is not handed the true eye trace. The answer is yes in
the finite-cache observer: marginalizing over possible trajectories recovers a
large fraction of the known-eye minus zero-eye gap. The matched-static
distractor condition is especially important because it reduces the chance
that the observer is winning from trivial static-response differences.
```

Posterior interpretation:

```text
The posterior does not need to identify one exact trajectory. Median N_eff/K
in the matched-static 1.0x condition is about 0.364 for the empirical prior
and 0.400 for the OU prior. This supports a partial latent-pose constraint:
the image and response movie narrow the trajectory possibilities without
requiring perfect trajectory decoding.
```

Mechanism interpretation:

```text
The compact-mechanism panel says compact translation geometry can carry much
of the trajectory-dependent likelihood structure. Compact-only image-disjoint
projections recover high true-score rescue relative to random, unit-shuffle,
and gain controls. But static-PC subspaces remain close controls at some
dimensions, so compact geometry is a sufficiency/mechanism bridge, not unique
mechanism proof.
```

Caveats and flags:

```text
F011: C subpanels are generated but not integrated into the final Figure 4
style. The observer is exact-cache and finite-candidate scoped.

F006: compact geometry should be framed as sufficient evidence, not as a
unique or necessary mechanism.
```

### Module D Expanded Read

Result:

```text
Axis-conditioned trajectory priors rescue image identity relative to zero-eye,
but the preferred axis is not universal. At matched-static 0.5x, edge-parallel
beats edge-orthogonal by +0.031. At hard-negative 0.5x, the same +0.031
parallel advantage appears. At hard-negative 1.0x, the sign is approximately
flat/slightly orthogonal (-0.008), and at 2.0x orthogonal is ahead (-0.063).
```

Interpretation:

```text
This moves the story from "motion can help" to "which motion directions help
for which images and objectives?" The clean point is image dependence: useful
motion axes are defined relative to local image geometry. The result should
not be compressed into a universal law that animals should always move
parallel to edges or always move orthogonal to edges.
```

Preservation interpretation:

```text
The strongest D panel is the edge-parallel preservation audit. Matched
edge-parallel displacements disrupt pixels and V1-twin responses less than
matched orthogonal displacements:

pixel advantage = 300.54, CI [172.789, 408.961], 26/29 sessions positive
twin advantage = 0.000454, CI [0.000371, 0.000537], 29/29 sessions positive
```

Objective interpretation:

```text
D5 keeps the objective story honest. Current response-objective models do not
yet beat raw local edge geometry as a behavioral alignment baseline. Pixel
controls can be positive, but the response objectives are not yet clean
behavioral predictors.
```

Caveats and flags:

```text
F007: axis preference is candidate-set and scale dependent. Use D2/D3 as
evidence for image-conditioned axis structure, not as a universal parallel
policy.

F008: edge-parallel preservation is a local stability/preservation result. It
does not by itself define the full active-sensing objective.

F009: model-objective adjudication remains open because current V1-twin
response objectives do not cleanly outperform raw edge geometry.
```

### Module E Expanded Read

Result:

```text
Free-viewing FEM axes are modestly but reliably aligned with local edge
geometry. Using the unweighted session-mean atlas convention:

all windows: mean cos2 = 0.105, CI [0.067, 0.145], n = 11749
reliable axes: mean cos2 = 0.140, CI [0.089, 0.188], n = 6242
high confidence: mean cos2 = 0.269, CI [0.138, 0.396], n = 1045
```

Endpoint-zone interpretation:

```text
The zone analysis gives a more intuitive angle read. The parallel <=15 deg
zone is enriched relative to a uniform axial expectation:

all windows = 1.304x expected
reliable axes = 1.427x expected
high confidence = 2.124x expected

Orthogonal and mid-angle zones are at or below uniform expectation in the same
summaries. This supports the qualitative statement that measured FEM axes are
biased toward local edge-parallel geometry, especially when the local edge
estimate is reliable.
```

Interpretation:

```text
This is the behavioral payoff for the figure. After A-D establish the retinal
movie premise, model-side usefulness, latent-pose observer rescue, and local
geometry predictions, E asks whether animals' measured free-viewing FEM axes
are related to image geometry. The answer is positive, modest, and reliability
dependent. The effect is not huge, but it is consistent across sessions and
stronger when the local image-axis estimate is cleaner.
```

Caveats and flags:

```text
F009: behavior aligns with raw image geometry better than current model
objectives. The behavioral claim should be "image-geometry alignment," not
"animals optimize this V1-twin response objective."

F012: the metric convention matters. Weighted headline-style means are larger
than unweighted session means:
  all windows: 0.181 weighted versus 0.105 unweighted
  reliable axes: 0.201 weighted versus 0.140 unweighted

Use the unweighted session-mean convention for atlas prose unless directly
referencing the old rendered headline figure.
```

### Cross-Module Interpretation

The cleanest complete Figure 4 argument is layered rather than single-step.
Panel A establishes the retinal movie transformation. Panel B shows that
motion-linked response movies contain decodable image-feature structure. Panel
C shows that this structure can remain usable when eye position is latent.
Panel D explains why useful motion directions should depend on local image
geometry rather than following one universal axis rule. Panel E then tests the
behavioral prediction and finds that free-viewing FEM axes are biased toward
local edge geometry.

The strongest current main-figure spine is therefore:

```text
A1/A2/A4: physical premise and canonical pipeline
B3/B4: empirical drift adds feature-decodable structure and beats OU controls
C2/C3: joint observer recovers information lost by zero-eye assumptions
D1/D4: local geometry defines axes; edge-parallel motion preserves structure
E2/E3: measured free-viewing FEM axes align with local image geometry
```

The most important tone constraint is that the figure should not imply a
settled optimal-control story. It supports an active-sensing interpretation of
FEM-linked reafference, with clear boundaries around proxy metrics, controls,
mechanism uniqueness, and behavioral objective adjudication.

## Module A: Retinal Movie Premise

### A1: Fixed Screen Image Becomes Moving Retinal Crop

![A1 fixed screen to retinal crop](figures/panel_A/A1_retinal_movie_transform.png)

Read:

```text
During fixation, the screen image is fixed but the retinal sample is not fixed:
the eye trace creates a sequence of shifted retinal crops.
```

Use: main setup.

Boundary: schematic premise panel; no result dependency.

Flags: `F002`.

### A2: Stabilized Versus FEM Movie QC

![A2 retinal-motion transform QC](figures/panel_A/A2_movie_transform_qc.png)

Read:

```text
FEM movies contain temporal contrast and motion power relative to stabilized
counterfactuals, while static movie power remains matched.
```

Key values:

```text
temporal contrast RMS mean: FEM 11.245, stabilized 0.000
motion power vs stabilized: FEM 1462.431, stabilized 0.000
movie power mean: FEM 15178.177, stabilized 15185.182
```

Use: main setup or supplement QC.

Boundary: older movie-information QC; use as transformation evidence, not the
main canonical 756-unit BackImage endpoint.

Flags: `F002`.

### A3: Translation Samples Local Gradients

![A3 local-gradient cartoon](figures/panel_A/A3_gradient_sampling_cartoon.png)

Read:

```text
Small retinal translations sample local image gradients. The same displacement
can be informative or preserving depending on local geometry.
```

Use: main conceptual bridge into D, or an inset.

Boundary: cartoon only.

Flags: `F002`.

### A4: V1-Twin Retinal Movie Pipeline

![A4 BackImage pipeline](figures/panel_A/A4_backimage_pipeline_bridge.png)

Read:

```text
The downstream B-E analyses use the canonical BackImage/V1-twin pathway:
screen image + eye trace -> retinal movie -> 756-unit V1 twin -> response movie.
```

Key provenance:

```text
256 images; 29 sessions; 151 drift-only trace sources
canonical 756-unit V1 twin; grouped-by-image CV
RMS ratio = 1.0; clipping = 0.0
```

Use: main setup.

Boundary: provenance bridge rather than a separate result.

Flags: `F002`.

### A5: Covariance Bridge To Sigma_FEM

![A5 covariance bridge](figures/panel_A/A5_covariance_bridge_guardrail.png)

Read:

```text
Existing covariance analyses support a reafferent bridge, but the rows use
mixed denominators and should be routed as context or supplement.
```

Use: supplement or bridge.

Boundary: do not make this the main Figure 4 endpoint unless the denominator
caveat is central and explicit.

Flags: `F002`.

## Module B: FEM Movies Add Feature-Decodable Structure

### B1: Feature-Decoding Task Schematic

![B1 task schematic](figures/panel_B/B1_task_schematic.png)

Read:

```text
The response movie is summarized over time and used to decode natural-image
feature targets beyond a static response baseline.
```

Use: main setup if A4 does not already carry the pipeline.

Boundary: schematic; pair with B3 for the actual result.

Flags: `F003`.

### B2: Motion Family QC

![B2 motion family QC](figures/panel_B/B2_motion_family_qc.png)

Read:

```text
Empirical, OU-like, Brownian, and rotated motion families are RMS-matched with
no clipping in the cleaned aggregate run.
```

Use: supplement or compact QC inset.

Boundary: QC panel, not a benefit result.

Flags: `F003`, `F004`.

### B3: Static-Plus-Motion Gain Over Static

![B3 gain over static](figures/panel_B/B3_empirical_gain_vs_static.png)

Read:

```text
Empirical drift-like temporal response movies add feature-decodable structure
above static V1-twin responses for both Gabor and pyramid feature targets.
```

Use: main.

Boundary: deterministic feature-decoding proxy in `-MSE` units, not literal
mutual information.

Flags: `F003`.

### B4: Empirical Minus Controls

![B4 empirical minus controls](figures/panel_B/B4_empirical_minus_controls.png)

Read:

```text
Empirical drift robustly beats OU-like confined controls; the advantage over
Brownian and rotated controls is clearest at smaller scales.
```

Use: main or main inset.

Boundary: specificity narrows at larger scales.

Flags: `F003`, `F004`.

### B5: Absolute Gain Guardrail

![B5 absolute gain guardrail](figures/panel_B/B5_absolute_gain_guardrail.png)

Read:

```text
Generic motion controls can catch up at larger scales, so the result is
distributional and scale/readout scoped.
```

Use: supplement or guardrail inset.

Boundary: avoids a "more motion is always better" interpretation.

Flags: `F003`, `F004`.

### B6: Local Exact Image-Trace Pairing

Rendered asset: none in the atlas panel set.

Read:

```text
The local exact-pairing branch remains unresolved and should not carry the
headline until rechecked.
```

Use: supplement only.

Flags: `F005`.

## Module C: Joint Image-And-Eye Observer

### C1/C2: Observer Schematic And Equations

![C1 observer schematic](figures/panel_C/C1_observer_schematic.png)

Read:

```text
The observer compares known-eye, zero-eye, and joint-eye inference, where the
joint observer marginalizes over a finite trajectory catalog.
```

Use: main setup.

Boundary: schematic/methods bridge.

Flags: `F011`.

### C3: Accuracy Ordering Across Candidate Sets

![C2 accuracy ordering](figures/panel_C/C2_accuracy_ordering.png)

Read:

```text
Known-eye is the ceiling, zero-eye fails when motion matters, and joint-eye
recovers substantial image identity across candidate sets.
```

Use: main.

Boundary: exact finite cache observer; not a biological decoder claim by
itself.

Flags: `F011`.

### C4: Matched-Static Distractor Control

![C3 matched static rescue](figures/panel_C/C3_matched_static_rescue.png)

Read:

```text
In matched-static distractors at 1.0x, joint-eye inference recovers much of
the known-zero gap even when static responses are matched.
```

Key values:

```text
known = 1.000
zero = 0.328
joint empirical = 0.766
joint OU = 0.797
```

Use: main.

Boundary: strongest C result; still exact-cache scoped.

Flags: `F011`.

### C5: Posterior Concentration

![C4 posterior concentration](figures/panel_C/C4_posterior_concentration.png)

Read:

```text
The joint observer concentrates over plausible trajectories without needing
exact trajectory recovery.
```

Use: supplement or small inset.

Boundary: supports partial latent-pose constraint, not perfect pose decoding.

Flags: `F011`.

### C6: Scale Rescue Guardrail

![C5 scale guardrail](figures/panel_C/C5_scale_gap_guardrail.png)

Read:

```text
Larger rescue at 1.0x partly reflects larger zero-eye failure, so scale should
be discussed with the zero-eye baseline.
```

Use: supplement or guardrail inset.

Flags: `F011`.

### C7: Compact Mechanism Guardrail

![C6 compact mechanism guardrail](figures/panel_C/C6_compact_mechanism_guardrail.png)

Read:

```text
Compact translation geometry is sufficient for much of the trajectory rescue,
but static-PC controls remain a close low-dimensional alternative.
```

Use: supplement or mechanism inset.

Boundary: do not claim uniqueness.

Flags: `F006`.

## Module D: Image-Dependent Useful Motion Directions

### D1: Local Edge/Gradient/Motion Axes

![D1 local axis schematic](figures/panel_D/D1_local_axis_schematic.png)

Read:

```text
Local image geometry defines edge-parallel and edge-orthogonal directions,
which can support different objectives.
```

Use: main setup.

Boundary: schematic.

Flags: `F007`.

### D2: Axis-Conditioned Observer

![D2 axis-conditioned accuracy](figures/panel_D/D2_axis_conditioned_accuracy.png)

Read:

```text
Axis-conditioned trajectory priors rescue image identity above zero-eye, but
both axes can help.
```

Use: supplement or main support.

Boundary: axis preference is condition-dependent.

Flags: `F007`.

### D3: Axis Preference Guardrail

![D3 axis preference guardrail](figures/panel_D/D3_axis_preference_guardrail.png)

Read:

```text
Parallel preference can flip toward orthogonal at larger hard-negative scales.
```

Use: supplement/guardrail.

Boundary: do not claim one universal motion policy.

Flags: `F007`.

### D4: Edge-Parallel Preservation Audit

![D4 edge-parallel stability](figures/panel_D/D4_edge_parallel_stability.png)

Read:

```text
Edge-parallel displacement disrupts local pixels and V1-twin responses less
than matched orthogonal displacement.
```

Key values:

```text
pixel advantage: 300.54, CI [172.789, 408.961], 26/29 sessions positive
twin advantage: 0.000454, CI [0.000371, 0.000537], 29/29 sessions positive
```

Use: main.

Boundary: local preservation audit, not a full policy objective.

Flags: `F008`.

### D5: Objective Alignment Guardrail

![D5 objective alignment guardrail](figures/panel_D/D5_objective_alignment_guardrail.png)

Read:

```text
Current response-objective models do not yet beat raw edge geometry as a
behavioral alignment baseline.
```

Use: supplement/guardrail.

Boundary: response-objective adjudication remains open.

Flags: `F009`.

## Module E: Free-Viewing FEMs Follow Image Geometry

### E1: Behavior Setup Example

![E1 behavior setup](figures/panel_E/E1_behavior_setup_example.png)

Read:

```text
A representative high-confidence free-viewing window shows the measured FEM
axis against the local edge axis.
```

Use: setup or supplement.

Boundary: selected example, not a distribution result.

Flags: `F010`.

### E2: Behavioral Edge-Alignment Strength

![E2 behavior alignment strength](figures/panel_E/E2_behavior_alignment_strength.png)

Read:

```text
Measured free-viewing FEM axes align modestly but reliably with local edge
geometry, especially when local axes are reliable.
```

Key values:

```text
all windows: mean session cos2 = 0.105, CI [0.067, 0.145]
reliable axes: mean session cos2 = 0.140, CI [0.089, 0.188]
high confidence: mean session cos2 = 0.269, CI [0.138, 0.396]
```

Use: main.

Boundary: use the unweighted session-mean convention in atlas prose.

Flags: `F009`, `F012`.

### E3: Endpoint-Zone Enrichment

![E3 endpoint enrichment](figures/panel_E/E3_parallel_zone_enrichment.png)

Read:

```text
Parallel endpoint-zone occupancy is enriched relative to a uniform angular
expectation, especially in high-confidence windows.
```

Key values:

```text
parallel <=15 deg enrichment:
  all windows = 1.304x
  reliable axes = 1.427x
  high confidence = 2.124x
```

Use: main or main inset.

Flags: `F012`.

Provenance: compact atlas redraw from
`backimage_edge_alignment_distribution_inspection/endpoint_zone_enrichment_summary.csv`.
Pair with E6-E8 when reviewing the behavior metric because those copied source
diagnostics show the full distribution, confidence dependence, and null shape
behind this summary.

### E6: Full Drift-Edge Distribution And Session Diagnostic

![E6 full distribution diagnostic](figures/panel_E/E6_full_distribution_session_diagnostic.png)

Read:

```text
The positive edge-parallel bias is visible in the full window distribution,
session-level means, and cumulative parallel-preference curve, with the
high-confidence subset showing the strongest effect.
```

Use: supplement/provenance for E2/E3; include when explaining the behavioral
contour-following metric.

Source:
`outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_edge_alignment_distribution_inspection/edge_alignment_window_and_session_distributions.png`.

Flags: `F012`.

### E7: Confidence And Signed-Delta Diagnostic

![E7 confidence signed delta diagnostic](figures/panel_E/E7_confidence_signed_delta_diagnostic.png)

Read:

```text
Drift-edge alignment strengthens as image orientation coherence and FEM
anisotropy increase, and the signed reliable-axis delta distribution clusters
around the edge-parallel direction.
```

Use: supplement/provenance for the high-confidence behavioral read.

Source:
`outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_edge_alignment_distribution_inspection/edge_alignment_confidence_and_signed_delta.png`.

Flags: `F012`.

### E8: Endpoint/Null Diagnostic

![E8 endpoint null diagnostic](figures/panel_E/E8_endpoint_null_diagnostic.png)

Read:

```text
The raw cos2 histogram has endpoint-heavy mass even under a uniform axial-angle
null; after that correction, the excess is concentrated near edge-parallel
directions rather than edge-orthogonal directions.
```

Use: supplement/provenance for E3; important for explaining why endpoint-zone
enrichment is the clean behavior metric.

Source:
`outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_edge_alignment_distribution_inspection/edge_alignment_endpoint_null_diagnostic.png`.

Flags: `F012`.

### E4: Metric Convention Guardrail

![E4 metric convention guardrail](figures/panel_E/E4_metric_convention_guardrail.png)

Read:

```text
Weighted headline-style summaries are larger than unweighted session-mean
summaries, so the metric convention must be stated explicitly.
```

Use: supplement/guardrail.

Flags: `F012`.

### E5: Supported Versus Not-Yet-Supported Summary

![E5 scope summary](figures/panel_E/E5_scope_summary.png)

Read:

```text
Behavioral image-geometry alignment is positive; specific V1-twin response
objective adjudication remains open.
```

Use: supplement/claim-boundary box.

Flags: `F009`, `F012`.

## Remaining Global Flags

```text
F001: external literature citations still need verification
  Meaning:
    The local atlas intentionally avoids leaning on external citation claims
    until the source papers/pages are checked. This matters mostly for the
    Results lead and any comparison to recent retinal trajectory-inference
    literature.
  Handling:
    Keep citation-specific claims out of final prose until verified.

F002: Module A subpanels exist, but final composite selection remains open
  Meaning:
    A1-A5 are generated, but the main Figure 4 layout has not been composed.
    A5 is especially sensitive because it bridges to covariance evidence with
    mixed denominators.
  Handling:
    Use A1/A2/A4 as the likely main setup; route A5 to supplement unless the
    covariance caveat is made explicit.

F003: aggregate FEM information is a deterministic decoding proxy
  Meaning:
    B3/B4 report feature-decoding gain in -MSE units from deterministic
    V1-twin responses. They are not literal mutual information.
  Handling:
    Use "feature-decodable structure", "decoding gain", or "information
    proxy"; avoid unqualified "mutual information".

F004: Brownian/rotated control specificity narrows at larger scales
  Meaning:
    Empirical drift robustly beats OU, but Brownian and rotated controls become
    competitive at larger scale. This keeps the B claim scale/readout scoped.
  Handling:
    Emphasize strongest specificity at 0.25x-0.5x and avoid claiming empirical
    motion uniquely beats all generic controls at all scales.

F005: local exact image-trace pairing remains unresolved
  Meaning:
    The exact local image-trace pairing branch has not been promoted to a
    figure-ready result. It could support a narrower claim after rechecking.
  Handling:
    Keep B's main claim distributional and route local pairing to supplement.

F006: compact mechanism is sufficient but not unique
  Meaning:
    Compact translation geometry carries much of the trajectory-rescue signal,
    but static-PC controls remain close at some dimensions.
  Handling:
    Present compact geometry as mechanism support/sufficiency, not unique
    necessity.

F007: D axis preference is candidate-set and scale dependent
  Meaning:
    Axis-conditioned priors help, but the preferred axis changes with the
    task/candidate set/scale. Parallel is not always better.
  Handling:
    Use D2/D3 to argue for image-conditioned motion axes, not a universal
    biological edge-parallel law.

F008: edge-parallel preservation is not a full motion policy
  Meaning:
    D4 cleanly shows local preservation for pixels and V1-twin responses, but
    preservation is only one possible active-sensing objective.
  Handling:
    Use D4 as a mechanistic/explanatory panel and keep policy language narrow.

F009: model objectives do not yet beat raw edge geometry for behavior
  Meaning:
    The behavioral result is positive for raw image-geometry alignment, but
    current V1-twin response objectives are not yet better behavioral
    predictors than raw edge axes.
  Handling:
    Claim "measured FEMs align with image geometry", not "animals optimize the
    tested response objective."

F010: final atlas composites are not built yet
  Meaning:
    A-E subpanels exist, but no full main figure or supplement sheets have
    been composed with unified typography, panel letters, or final sizes.
  Handling:
    Do a selection pass before building the composite.

F011: C subpanels are generated but not integrated
  Meaning:
    The joint observer is conceptually central, but it is absent from the old
    rendered headline figure and still needs final style integration.
  Handling:
    Promote C2/C3 into the main Figure 4 candidate and route C4-C6 based on
    space.

F012: E metric convention must be stated explicitly
  Meaning:
    Weighted headline-style behavior summaries are larger than unweighted
    session-mean summaries. Both are valid, but they answer slightly different
    summary questions.
  Handling:
    Use unweighted session means in atlas prose; use weighted values only when
    referencing the old headline figure or stats manifest.
```
