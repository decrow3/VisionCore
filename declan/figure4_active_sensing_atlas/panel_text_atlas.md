# Figure 4 Active Sensing Panel Text Atlas

Status: generated panel-reading document, cache-first.

2026-06-21 correction note:

```text
Panel B temporal-PCA absolute-gain prose and candidate images in this generated
contact sheet are superseded. The corrected aggregate posthoc uses static mean
responses as the baseline and turns the feature target into a role split:
mean/delta_mean are absolute aggregate candidates, delta_mean remains the local
mechanistic bridge, and temporal PCA/DCT variants are order-sensitive
empirical-vs-control diagnostics. Candidate 3 has been redrawn from
incremental_staticmean_plus_motion_tworeadout_v2 and the all-readout atlas
lives under incremental_staticmean_plus_motion_allreadouts_v1. Regenerate this
contact sheet before promotion if every embedded old Panel B note must be
removed.
```

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
B: Empirical drift-like movies produce feature-relevant response changes.
C: A joint image-and-eye observer preserves image-feature information when
   eye position is latent.
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
Empirical drift beats Brownian and rotated controls most cleanly at small
scales. OU is no longer promoted here: its absolute gain falls below static,
which is a trace-generation or analysis audit trigger rather than a clean
negative-control result.

Current 4B implementation promotes a recomputed source-trial grouped
information axis: diagonal Gaussian decoder lower-bound gain in bits over
stabilized/static with point-centered decode-bootstrap CIs. The
trajectory renders the response movie but is not an explicit aggregate
ridge-decoder input. The pose-unaware hidden-sample proxy has the same
source-trial grouped information-axis recompute and has negative point estimates
across scales.
```

Caveats and flags:

```text
F003: the promoted plotted endpoint is a decoder-information increment in bits,
not an absolute mutual-information estimate. The old -MSE endpoint is archived
as QC/provenance.

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
The promoted continuous no-anchor joint observer should be read first through
posterior feature recovery rather than exact image identity. In the full
scale-prior run, the posterior expected feature vector reaches mean cosine
0.9378 to the true image feature, while exact image accuracy is 0.7083. The
split-heldout promotion gate is 0.9371. The scale-conditioned analyzer
temperatures are 0.125 at 0.5x, 0.125 at 1.0x, and 0.5 at 2.0x; the trajectory
prior is AR(1) at 0.5x/1.0x and matched-Brownian scale 8 at 2.0x.

The older exact finite trajectory-table observer remains useful as a context
and stress-test result: known-eye is highest, zero-eye drops when motion
matters, and joint-eye inference recovers much of the lost image identity by
marginalizing over plausible trajectories. In the matched-static-response
condition at 1.0x:

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
the promoted continuous observer under the feature-recovery readout: the joint
posterior can preserve much of the true image-feature direction even when exact
catalog identity remains a stricter and more brittle endpoint. The finite-cache
matched-static distractor condition is still important because it reduces the
chance that the observer is winning from trivial static-response differences.
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
style. The promoted continuous observer has a verified full-cache artifact and
manifest, but the rendered C contact-sheet panels still mostly reflect the
older exact finite-cache image-identity observer.

F006: compact geometry should be framed as sufficient evidence, not as a
unique or necessary mechanism.
```

### Module D Expanded Read

Result:

```text
Axis-conditioned trajectory priors help hidden-eye readout, but the panel
should be about readout performance, not raw response-change magnitude. In the
matched-static 0.5x feature-posterior branch, along-edge priors recover more
pyramid k8 feature signal than across-edge priors:

along-edge joint-zero gain = +6.052 [-MSE]
across-edge joint-zero gain = +3.684 [-MSE]
paired along-minus-across = +2.368, CI [+0.392, +4.589], p = 0.0257

The clean image-identity observer shows the same weak matched-static direction:
edge-parallel accuracy 0.859 versus edge-orthogonal 0.828, delta +0.031.
```

Interpretation:

```text
This moves the story from "motion can help" to "which motion directions help
for which images and objectives?" The clean positive read is that, in the
matched-static hidden-eye feature decoder, the along-contour prior is the more
useful local image-axis prior. The result should not be compressed into a
universal law that animals should always move parallel to edges.
```

Caveat:

```text
The hard-negative branch remains a guardrail. In the n64 feature-posterior
posthoc, hard negatives recover features above zero-eye for both axes, but the
parallel-minus-orthogonal feature gain is -0.745 with CI [-3.147, +1.631].
The hard-negative image-identity observer also weakly favors orthogonal at
0.5x (0.891 versus 0.844), with McNemar p ~= 0.51. The safe claim is therefore
matched-static along-contour utility plus image-axis dependence, not a settled
biological edge-axis law.
```

Preservation audit:

```text
The edge-parallel preservation audit is supporting context, not the promoted D
story. Matched edge-parallel displacements disrupt pixels and V1-twin responses
less than matched orthogonal displacements:

pixel advantage = 300.54, CI [172.789, 408.961], 26/29 sessions positive
twin advantage = 0.000454, CI [0.000371, 0.000537], 29/29 sessions positive
```

The newer twin-stability metric audit preserves this first-order message:
signed response-normalized, per-rate, covariance-whitened, and unit-subset
metrics are all positive at the tested endpoint displacement. This remains a
mechanistic support result, even though it should not be used to say the main
effect is that responses change more across than along contours.

Objective interpretation:

```text
D5 keeps the objective story honest. Current response-objective models do not
yet beat raw local edge geometry as a behavioral alignment baseline:
optimized response-stability and response-refresh axes are negative relative
to the raw edge axis, while the pixel-isophote axis is approximately flat.
Pixel controls can be positive, but the response objectives are not yet clean
behavioral predictors.
```

Caveats and flags:

```text
F007: axis preference is candidate-set and scale dependent. Use D2/D3 as
evidence for image-conditioned axis structure, not as a universal parallel
policy.

F008: edge-parallel preservation is a supporting local stability result. It
does not by itself define the promoted D readout result or the full
active-sensing objective.

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
B3/B4: motion-rendered empirical drift adds feature-decodable structure; the
  same-axis pose-unaware proxy has negative point estimates while OU stays audit-only
C2/C3: joint observer preserves feature information lost by zero-eye assumptions
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

Boundary: promoted 4B should use the residual-variance decoder information
increment in bits. Legacy `-MSE` panels are archive/QC context, not the current
axis.

Flags: `F003`.

### B4: Empirical Minus Controls

![B4 empirical minus controls](figures/panel_B/B4_empirical_minus_controls.png)

Read:

```text
Empirical drift has the clearest advantage over Brownian and rotated controls
at smaller scales. OU should remain diagnostic-only until the below-static
absolute-gain anomaly is audited.
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

Current promoted readout note:

```text
For the current continuous no-anchor observer, treat posterior expected
feature recovery as the primary diagnostic and exact image identity as a hard
secondary endpoint. This is a feature reconstruction-quality readout, not an
absolute mutual-information or bits estimate. The verified full strict
scale-prior artifact gives emitted mean feature cosine 0.9378,
split-heldout feature gate 0.9371, and exact image accuracy 0.7083.
Provenance lives in
figures/panel_C/diagnostics/continuous_joint/continuous_joint_promoted_observer_manifest.json.
The newest representation diagnostic separates the known-trace representation control
from the latent-eye recovery claim: at 1x, 0x stabilized feature cosine is
0.6678, hidden-eye joint motion is 0.8721, and known-trace motion is 0.9358.
Thus measured motion gives a deterministic known-trace gain of +0.2680 over the stabilized
counterfactual, and the hidden-eye joint observer preserves +0.2043 of that
gain.
The guarded affine-quadratic lead improves the split-swapped feature gate to
0.9374, but lowers hard image accuracy to 0.6927, so it remains a
diagnostic rather than the promoted endpoint. A full-cache
direct intercept ablation lowers heldout feature cosine to 0.9184, below the
origin-constrained observer, so the affine feature lead is intercept-dependent
and not a clean encoder promotion.
The cleaner known-start candidate is known-start quadratic inference: it improves
the full-cache heldout feature gate from 0.9343 to 0.9361 with essentially flat
image accuracy, and improves trajectory correlation, but uses the first
measured eye-position sample. The calibrated production artifact with the fixed
scale-temperature schedule reaches emitted feature cosine 0.9374.
```

### C1/C2: Observer Schematic And Equations

![C1 observer schematic](figures/panel_C/C1_observer_schematic.png)

Read:

```text
The observer compares known-eye, zero-eye, and joint-eye inference. In the
promoted continuous readout, the joint posterior is scored by recovered image
feature direction first, with exact image identity retained as a stricter
stress test.
```

Use: main setup.

Boundary: schematic/methods bridge; rendered schematic still reflects the
finite-cache lineage.

Flags: `F011`.

### C3: Accuracy Ordering Across Candidate Sets

![C2 accuracy ordering](figures/panel_C/C2_accuracy_ordering.png)

Read:

```text
Known-trace is the deterministic control, zero-eye fails when motion matters,
and joint-eye recovers substantial signal across candidate sets. For the
promoted continuous observer, read this as feature recovery first and image
identity second; do not treat the known-trace row as an independent response
target.
```

Use: main.

Boundary: exact finite-cache image identity is historical/contextual here; the
promoted continuous result is the verified feature-posterior artifact, not a
biological decoder claim by itself.

Flags: `F011`.

### C4: Matched-Static Distractor Control

![C3 matched static rescue](figures/panel_C/C3_matched_static_rescue.png)

Read:

```text
In matched-static distractors at 1.0x, joint-eye inference recovers much of
the known-zero gap even when static responses are matched. This remains the
best exact-cache stress test supporting the continuous feature-recovery story.
```

Key values:

```text
known = 1.000
zero = 0.328
joint empirical = 0.766
joint OU = 0.797
```

Use: main context or supplement, depending on whether the continuous
feature-recovery panel is rendered for the final composite.

Boundary: strongest older C image-identity result; still exact-cache scoped.

Flags: `F011`.

### C5: Posterior Concentration

![C4 posterior concentration](figures/panel_C/C4_posterior_concentration.png)

Read:

```text
The joint observer concentrates over plausible trajectories without needing
exact trajectory recovery. This supports the softer feature-recovery readout:
the posterior need not put all mass on one exact image/trace pair to preserve
the correct feature direction.
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

### D2: Axis-Conditioned Feature Readout

![D2 axis-conditioned feature recovery](figures/panel_D/D2_axis_feature_recovery.png)

Read:

```text
Along-edge trajectory priors recover more matched-static feature signal than
across-edge priors when the eye trajectory is latent.
```

Key values:

```text
matched-static, 0.5x, pyramid k8 feature posterior:
  along-edge joint-zero gain = +6.052 [-MSE]
  across-edge joint-zero gain = +3.684 [-MSE]
  paired along-minus-across = +2.368, CI [+0.392, +4.589], p = 0.0257
```

Use: main.

Boundary: matched-static feature-posterior readout; hard negatives remain a guardrail.
The promoted strict continuous joint estimator weakens this as a cross-panel
claim: its all-scale along-minus-across feature-cosine contrast is only
`+0.0011`, with intervals crossing zero. At 1x it is `0.9407` along versus
`0.9366` across with identical image accuracy (`0.7031`). So D2 remains an
axis-conditioned matched-static readout, not a guaranteed property of the
strict no-start joint estimator.

Matched-static versus strict-continuous distinction: D2 uses matched-static
response candidates and scores joint-minus-zero gain in pyramid feature
`-MSE`. The promoted 4C diagnostic uses the hard-negative continuous-joint
cache, a no-start latent trajectory estimator, and posterior feature cosine.
Thus D2 supports "axis priors can help this matched-static feature decoder,"
whereas the promoted 4C check asks the harder inheritance question, "does the
strict continuous observer itself prefer along-contour traces?" That latter
answer is currently weak/null.

Known-axis guardrail: a direct rotated-trace diagnostic asks the simpler
question with the trajectory index known. In the same matched-static 0.5x,
`pyramid_local_field` k8 cache, across-contour feature cosine is `0.8834` and
along-contour is `0.8758`; along-minus-across is `-0.0076`, CI
`[-0.0096, -0.0057]`, p `0.0010`. So D2 should not be interpreted as a direct
known-trace feature-alignment advantage for along-contour motion.

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

Use: support/supplement.

Boundary: local preservation audit, not the promoted readout story or a full policy objective.

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

F003: aggregate FEM information uses a decoder lower-bound increment
  Meaning:
    The promoted B3/B4-style recompute reports feature-information gain in bits
    from held-out residual variances. Legacy B3/B4 report feature-decoding gain
    in -MSE units and are retained only as archive/QC.
  Handling:
    Use "decoder information increment" or "decoder lower-bound gain"; avoid
    unqualified absolute mutual information.

F004: Brownian/rotated control specificity narrows at larger scales
  Meaning:
    Brownian and rotated controls become competitive at larger scale. The OU
    family is currently an audit trigger because its absolute gain falls below
    static. This keeps the B claim scale/readout scoped.
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
    Axis-conditioned priors help, and the matched-static feature-posterior
    readout favors along-edge priors. The preferred axis changes with the
    task/candidate set/scale, and hard-negative controls do not support a clean
    universal parallel advantage.
  Handling:
    Use D2/D3 to argue for matched-static along-contour readout utility and
    image-conditioned motion axes, not a universal biological edge-parallel law.

F008: edge-parallel preservation is supporting evidence, not the D headline
  Meaning:
    D4 cleanly shows local preservation for pixels and V1-twin responses, but
    the promoted D story is better feature recovery along than across contours,
    not simply larger response change across contours.
  Handling:
    Use D4 as a mechanistic/explanatory support panel and keep policy language
    narrow.

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
    The joint observer is conceptually central. The verified promoted result is
    now the continuous no-anchor feature-recovery observer, but the old
    rendered headline/contact-sheet panels mostly show the finite-cache
    image-identity lineage and still need final style integration.
  Handling:
    Promote the continuous feature-recovery readout into the main Figure 4
    candidate, keep finite-cache image-identity panels as context/guardrails,
    and route C4-C6 based on space.

F012: E metric convention must be stated explicitly
  Meaning:
    Weighted headline-style behavior summaries are larger than unweighted
    session-mean summaries. Both are valid, but they answer slightly different
    summary questions.
  Handling:
    Use unweighted session means in atlas prose; use weighted values only when
    referencing the old headline figure or stats manifest.
```
