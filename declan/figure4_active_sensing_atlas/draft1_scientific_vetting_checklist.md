# Draft1 Scientific Vetting Checklist

Status: working checklist, 2026-07-05.

Companion methods reset:

```text
declan/figure4_active_sensing_atlas/literature_methods_reset.md
```

Companion design decision:

```text
declan/figure4_active_sensing_atlas/unified_feature_observer_design_decision.md
```

Companion score decision:

```text
declan/figure4_active_sensing_atlas/unified_feature_observer_score_decision.md
```

Source draft:

```text
/home/declan/.codex/attachments/9a4419c7-f442-448b-bf67-9712d6bc2571/draft1.docx
```

Purpose: vet the argument in `draft1.docx` step by step without expanding into
every adjacent active-sensing branch. This checklist is scoped to claims that
appear in the draft and to tests needed to make those claims scientifically
rigorous. Earlier RR100/Vernier/behavior steps are included as prerequisite
gates, but the main focus is the feature-decoder sequence and its assumptions.

## Scope Lock

In scope:

- RR100 as the reduced V1 representation introduced in the draft.
- Vernier as a pose-confusion demonstration.
- Behavior panels used only to motivate what animals track.
- Natural-image feature target definition.
- Unified feature-recovery observer conditions: stabilized, known trajectory,
  zero-eye baseline, hidden/marginal trajectory baseline, and joint latent
  trajectory if robust.
- Local image/trajectory pairing model as a supplemental specificity test.
- Compact-geometry link as an intervention inside the same observer family.
- Separation between the primary 4C active-sensing observer and the
  compact-dependent forward-model audit branch.

Out of scope unless a draft claim explicitly depends on it:

- Reopening the old E-optotype covariance-code or temporal-code story.
- Exhaustive behavior-objective optimization claims.
- New compact local-derivative channel analyses.
- New long canonical production runs.
- Full exhaustive literature review; method-alignment reading is tracked in the
  companion reset note.
- Any claim that exact real FEM trajectories are optimal.

## Design Lock

The main Figure 4 feature-decoder sequence should not be written as aggregate
decoder, local decoder, and joint decoder. It should be written as one
feature-recovery observer family with one feature target, one primary score
axis, and several eye-information conditions:

```text
static_matched_mean
known trajectory / motion-rendered empirical trajectory
zero-eye trajectory baseline
hidden/marginal trajectory baseline
joint latent-eye trajectory, if robust
```

Static/motion labels must obey the coordinate contract:

```text
static_matched_mean  = static at each movie/window mean fixation position
motion_mean_centered = within-movie residual motion around that mean
zero_eye_on_motion   = observer/readout assumes no residual motion for a
                       motion-rendered response
static_global_zero   = optional oracle/control, not the BackImage default
```

For BackImage, cached `zero_lambda_counts` / legacy `zero_static` rows are
only acceptable manuscript evidence if labeled as `static_matched_mean` or
static at the crop/mean fixation position. Do not call this a global zero-eye
stimulus.

The main 4C observer should not require compact geometry. Use response-space
or static-PC feature recovery first, then ask whether compact-only,
compact-removed, compact-addback, or static-PC interventions explain the
recovery. Compact-forward tau inference is a useful diagnostic branch, but it
is entangled with the compact observation model and should not be the sole
paper-facing 4C result.

The primary score axis should be held-out normalized MSE, reported as `R2_cv`,
computed as pooled out-of-fold SSE/SST in the locked, train-normalized feature
space. The target is the pooled complex steerable-pyramid-like magnitude target
unless a run literally uses Plenoptic. Feature cosine is a secondary robustness
score. Gaussian likelihood / diagonal information is a supplemental theory
score.

Linear-Gaussian feature observers should use source-balanced training weights
unless explicitly labeled as row-unweighted diagnostics. Candidate and
trajectory tables reuse sources unevenly, so unweighted rows answer a
candidate/trajectory-frequency-weighted question rather than a source-uniform
one.

Do not use the pooled-prior continuous feature observer to make along/across
axis claims. Its current contract fits across prior-family labels and evaluates
the same observed response under both axis labels, so any parallel-minus-
orthogonal contrast from that runner is a bookkeeping artifact. Along/across
requires a dedicated per-axis observer.

Local exact image/trajectory pairing is not load-bearing. It remains a
specificity control unless the corrected exact-pairing contrast stabilizes.

## Claim 1: RR100 Is The Working Reduced V1 Representation

Draft role:

```text
The 756-channel canonical V1 digital twin is reduced to the RR100 movie-medoid
population by removing likely duplicate or near-duplicate channels while
preserving the convolutional spatial readout grid.
```

Hidden assumptions:

- Channel redundancy is real and biologically relevant, not merely a plotting
  artifact.
- Reducing channel redundancy improves biological interpretability without
  destroying the response geometry needed for the draft claims.
- RR100 should be treated as a representation choice, not only as a convenience
  or speed optimization.

Already checked:

- [x] RR100 construction preserves the spatial readout grid and only pools over
      channel identity.
- [x] RR100 uses movie medoids, so representatives are actual modeled units
      rather than averaged activation maps.
- [x] Construction used multi-stimulus validation rather than a single
      BackImage-only fingerprint.
- [x] RR100 QC includes group quality, reconstruction checks, activation maps,
      center-pixel traces, and Vernier/SSI downstream sanity checks.
- [x] RR100 retains substantial per-spike SSI but less total response rate than
      the 756-channel population.
- [x] Matched-total anisotropic Brownian RR100 check was basically null, which
      weakens a simple intrinsic along-motion claim.

Still needed for draft rigor:

- [ ] Reword RR100 as a biologically motivated reduced representation candidate,
      not just "useful for decoder development and sanity checks."
- [ ] For every later figure/result, state whether the analysis used RR100,
      RR265/RR192, or the canonical 756-channel population.
- [ ] Do not transfer a result across populations without labeling it as a
      representation-dependence assumption.
- [ ] For every Figure 4 static/motion comparison, state whether the static
      reference is `static_matched_mean` or `static_global_zero`.

Allowed wording:

```text
RR100 is a QC-backed, biologically motivated reduced V1-twin representation
that removes likely redundant model channels while retaining each unit's spatial
readout. It is not lossless, so population-dependent claims should state which
population was used.
```

Avoid:

```text
RR100 is only infrastructure.
RR100 and 756 are interchangeable.
Full 756 is automatically the more appropriate biological representation.
```

## Claim 2: Vernier Demonstrates The Pose-Confusion Problem

Draft role:

```text
Vernier provides a clean task variable. Motion can increase signal when the
trajectory is known, but it can become nuisance covariance when trajectory is
unknown.
```

Hidden assumptions:

- The synthetic Vernier task is a valid pedagogical bridge to natural-image
  feature recovery.
- The known/unknown/joint distinction is mathematically correct and not
  overread as a claim about animal behavior.
- The pose-robust decoder figure is not mistaken for a true eye-trace estimator.

Already checked:

- [x] Vernier task variable is explicit: `theta in {+delta, -delta}`.
- [x] Pose-known Fisher, pose-unknown Fisher, and pose-robust decoder are
      separated conceptually.
- [x] Pose-unknown Fisher collapse demonstrates trajectory-induced nuisance
      covariance.
- [x] Pose-robust decoder diagnostic recovers some performance without explicit
      eye-trace estimation.
- [x] RR100 Vernier checks preserve a weaker but qualitatively similar
      pose-aware modulation.
- [x] Old Vernier/E-optotype covariance-code and temporal-code interpretations
      were tested and narrowed historically.

Still needed for draft rigor:

- [ ] Label the pose-robust decoder as a diagnostic that does not estimate the
      eye trace.
- [ ] State whether each Vernier figure uses RR100 or full 756.
- [ ] State the likelihood/noise approximation used for `d'^2` or Fisher
      curves.
- [ ] Avoid implying Vernier proves active sensing in natural images.

Allowed wording:

```text
Vernier is a controlled pose-confusion demonstration: the same motion can help
under known-pose decoding and hurt under pose-hidden decoding. It motivates
natural-image joint inference but does not by itself prove biological
trajectory inference.
```

Avoid:

```text
The pose-robust Vernier decoder infers the eye trajectory.
Vernier proves FEMs are optimized.
```

## Claim 3: Behavior Motivates Local Image Geometry, Not Model Optimality

Draft role:

```text
Behavior asks what animals actually track; BackImage drift scale and
contour-relative motion differ from simpler tasks.
```

Hidden assumptions:

- The behavior panels are being used as geometric motivation, not as proof of
  a V1-twin objective.
- Contour alignment can reflect reduced across-edge motion, enhanced along-edge
  motion, or both.

Already checked:

- [x] FixRSVP and BackImage motion-scale diagnostics exist.
- [x] Drift/fixation-cloud axes align modestly with local image geometry.
- [x] Alignment strengthens with local orientation coherence.
- [x] Raw/local edge geometry currently beats or matches model-objective axes
      as the behavior baseline.
- [x] Scalar local-image features do not robustly explain all eye metrics.
- [x] Contour-relative behavior plots suggest along/across structure but do not
      isolate mechanism.

Still needed for draft rigor:

- [ ] Replace placeholder prose ("what do animals actually track?") with a
      specific claim.
- [ ] State whether behavior evidence is weighted or unweighted, and what the
      inference unit is.
- [ ] Separate "drift follows clear edges" from "animals optimize a V1-twin
      objective."
- [ ] For along/across language, explicitly keep open whether the key component
      is increased along motion, reduced across motion, or anisotropy at fixed
      total RMS.

Allowed wording:

```text
Measured drift geometry is modestly contour-following, especially when the
local image supplies a reliable edge axis. This behavior result motivates
edge-relative model tests but does not prove a model objective.
```

Avoid:

```text
Animals optimize the feature decoder.
Animals choose along-edge motion because the model says it is optimal.
```

## Claim 4: Natural Images Require An Explicit Feature Target

Draft role:

```text
Vernier has a scalar task variable; natural images require an explicit feature
target Phi = phi(I).
```

Hidden assumptions:

- The chosen feature target is scientifically meaningful.
- The target is not silently treated as a pixel reconstruction.
- The target preprocessing does not leak test information.
- The diagonal information score is a valid lower-bound-style diagnostic, not
  literal total population mutual information.

Already checked:

- [x] The target is defined as block-pooled pyramid features with orientation,
      scale, signed quadrature, and local energy components.
- [x] The target is explicitly not a pixel reconstruction.
- [x] Train-fold z-scoring and target PCA were added to avoid raw heterogeneous
      feature-scale artifacts.
- [x] The diagonal residual-variance score is documented as a decoder
      information increment, not absolute mutual information.
- [x] Full-covariance/log-det versions are recognized as supplementary or
      limited by sample size.

Still needed for draft rigor:

- [ ] Define the exact feature family used in each plotted result
      (`pyramid_local_field`, Gabor, orientation-covariance, etc.).
- [ ] State PCA dimension `k` and whether PCA/normalization is train-fold local.
- [ ] State whether residual covariance is diagonal, shrinkage/full, or
      negative-MSE.
- [ ] Make clear that `Delta I_diag` is a difference between estimated
      lower-bound-style decoder scores; it is not guaranteed to be a lower bound
      on true information gain.
- [ ] State whether the readout is meant as a biological decoder, a linear
      probe, or a controlled measurement device.

Allowed wording:

```text
The feature decoder estimates whether motion-rendered responses reduce
held-out uncertainty about a specified local feature target under a stated
linear-Gaussian/diagonal residual approximation.
```

Avoid:

```text
The decoder measures total information in V1.
The feature target is natural-image perception.
Motion increases information in an unconditional sense.
```

## Claim 5: Aggregate Feature Decoder Tests Distributional Motion Utility

Draft role:

```text
The aggregate feature decoder pools image windows and trajectories. It tests
whether measured-FEM-like motion statistics improve feature recovery relative
to stabilized responses, not whether exact trajectories are paired with exact
patches.
```

Hidden assumptions:

- Train/test splits and baselines are strict enough to support held-out feature
  recovery.
- The motion summary used in the decoder is not leaking the answer through a
  stale or non-comparable static baseline.
- The result is not driven by generic motion scale, clipping, path length, or a
  fragile readout choice.

Already checked:

- [x] Old temporal-PCA static baseline issue was found and demoted.
- [x] Corrected static-mean baseline is now the primary baseline.
- [x] Strict source-trial grouped n384 information-axis rerun is complete.
- [x] Same-axis pose-unaware hidden-sample proxy is complete and negative.
- [x] Image/window grouped result is retained as optimistic provenance context.
- [x] Fixed/shared ridge alpha reduces differential-regularization concerns.
- [x] Brownian and rotated controls show empirical specificity narrows at
      larger scales.
- [x] Temporal PCA/DCT are demoted to order-sensitive diagnostics rather than
      absolute-gain headlines.

Still needed for draft rigor:

- [ ] Decide whether the main draft sentence should use `mean`, `delta_mean`,
      or both, and state the role split.
- [ ] Complete or summarize the all-readout/nested-alpha review before a
      write-lock claim.
- [ ] Complete the OU / synthetic confined prior audit before using OU as a
      headline negative control.
- [ ] State motion families, scales, source-trial grouping, and trace source
      policy in the caption or methods.
- [ ] State that the trajectory renders the response movie but is not an
      explicit aggregate ridge-decoder input.
- [ ] Do not claim exact trajectory/image optimality from the aggregate result.

Allowed wording:

```text
In the aggregate analysis, empirical drift-like trajectories render response
movies whose static-plus-motion summaries carry more held-out feature evidence
than stabilized/static responses under strict source-trial grouping.
```

Avoid:

```text
The aggregate decoder proves the animal uses the exact trajectory.
The aggregate decoder is a full pose-aware observer.
Empirical motion uniquely beats every generic-motion control.
```

## Claim 6: Local Pairing Tests Exact Image/Trajectory Specificity

Draft role:

```text
The local pairing model asks whether real image-trajectory pairings outperform
matched trajectory swaps.
```

Hidden assumptions:

- The local result is not just an artifact of repeated source trials, static
  zero baselines, or readout mismatch.
- A positive pairing contrast would mean image-specific coupling, not just
  marginal FEM distributional utility.

Already checked:

- [x] Older local table had inherited grouping and baseline issues.
- [x] Corrected posthoc uses source-trial grouped decoding and static mean as
      the stabilized baseline.
- [x] Corrected exact-pairing contrasts weakened; CIs cross zero in the
      stricter rescore.
- [x] Power-seed local `delta_mean` summaries still provide mechanistic
      sensitivity evidence, but not a promoted exact-pairing claim.

Still needed for draft rigor:

- [ ] Replace "Limited small signal of this" with a precise status sentence.
- [ ] Separate local motion-induced feature sensitivity from exact
      image-trace pairing.
- [ ] Do not use local pairing as the main 4B proof unless the corrected
      source-trial grouped exact-pairing contrast is stable.
- [ ] If included as a supplement, show actual-minus-matched, actual-minus-
      rotated, actual-minus-Brownian/OU, and motion QC together.

Allowed wording:

```text
The local branch is a specificity test. Current corrected results support local
motion-delta feature sensitivity, but they do not yet support a strong exact
image-trace pairing claim.
```

Avoid:

```text
Real trajectories are specially matched to their viewed patches.
Local pairing explains the aggregate result.
```

## Claim 7: Joint Feature-Eye Decoder Tests Latent-Pose Recovery

Draft role:

```text
The joint feature-eye decoder asks whether feature recovery can be rescued when
eye trajectory is not handed to the observer.
```

Hidden assumptions:

- Joint recovery is not merely static response strength.
- Posterior feature recovery is not mixed with information-in-bits claims from
  4B unless both are recomputed on a shared target/score contract.
- Candidate-table or response-table leakage is not driving the result.
- Finite image-catalog search is not being used as the main endpoint when a
  continuous candidate-free feature observer is available.
- The continuous no-anchor observer is distinguished from candidate-posterior
  and candidate-free feature-embedding diagnostics.
- The current compact-forward candidate-free branch is compact-dependent, so a
  failure there can reflect compact observation-model misspecification rather
  than a general failure of latent-eye feature recovery.

Already checked:

- [x] Matched-static image-identity observer historically showed zero-eye
      failure and joint-eye rescue.
- [x] Promoted endpoint is now posterior expected feature recovery, not image
      identity accuracy.
- [x] Continuous no-anchor observer has a verified full-cache artifact.
- [x] Promoted calibration gate uses source-row-heldout feature cosine.
- [x] Inherited-decoder audit has no failures: posterior math, source identity,
      source-row folds, and validated point/CI contrast files pass.
- [x] Candidate-free linear feature-embedding branch exists but does not yet
      replace the promoted candidate-posterior endpoint.
- [x] MLP/residual-tau branches are upper-bound diagnostics, not downstream
      linear-readout claims.

Still needed for draft rigor:

- [ ] State which joint result the draft is relying on:
      candidate-posterior compact intervention, continuous no-anchor observer,
      or candidate-free embedding diagnostic.
- [ ] Prefer continuous / candidate-free latent-eye feature recovery for the
      main 4C endpoint; use finite candidate-image catalog search only as a
      diagnostic, legacy comparison, or fallback if the continuous observer
      fails.
- [ ] Make the primary 4C endpoint geometry-uncommitted: full response or
      static-PC response movie to feature target before compact interventions.
- [ ] Treat compact-forward latent-tau inference as a compact-dependent audit
      branch, not as the main active-sensing consequence.
- [ ] If candidate/source tables are used to fit response geometry, label that
      as source-disjoint geometry calibration, not as a catalog-search feature
      endpoint.
- [ ] Re-score the promoted 4B/4C feature-recovery claims on the shared primary
      axis: pooled multi-output held-out `R2_cv` / normalized MSE relative to
      the train-fold feature mean.
- [ ] Keep feature cosine as a secondary robustness score rather than the final
      primary figure axis.
- [ ] Keep diagonal information / Gaussian likelihood as supplemental theory
      scores unless their covariance contract is locked.
- [ ] State whether the observer estimates continuous `tau`, marginalizes a
      catalog, or uses posterior over candidate images.
- [ ] Do not imply the posterior identifies the animal's true eye trajectory.
- [ ] If using the newer synthetic empirical-confined prior, state that it is a
      synthetic prior calibrated from aggregate FEM statistics, not empirical
      trace replay.

Allowed wording:

```text
The joint observer shows that latent-eye feature recovery can be rescued above
zero-eye baselines, and the promoted continuous no-anchor calibration should be
read as provisional feature recovery rather than exact image or trajectory
identification. Strong main-text wording requires a geometry-uncommitted
response-space or static-PC observer to show the same ordering on held-out
feature R2: known-trajectory > joint > declared baseline, where the baseline is
explicitly labeled as zero-eye or hidden/marginal.
```

Avoid:

```text
The joint decoder recovers the true eye trace.
The joint decoder proves animals use this posterior.
Feature cosine and information bits are directly comparable.
High feature cosine alone proves calibrated feature recovery.
The joint result is a finite image-catalog search, unless that is explicitly
the diagnostic being discussed.
```

## Claim 8: Compact Geometry Supports Joint Recovery

Draft role:

```text
If eye-position-dependent response changes lie in compact geometry, the
observer can infer position along a low-dimensional motion-induced manifold.
```

Hidden assumptions:

- Compact geometry is not merely a trivial consequence of translating images
  through a smooth model.
- Compact geometry provides functional support for joint feature recovery.
- Static-response PCs and ordinary image-response manifolds are not being
  ignored.
- The compact-specific observer is not being used circularly as the only
  evidence for the active-sensing consequence.

Already checked:

- [x] Translation/Jacobian work shows image-specific but compact response
      tangents.
- [x] Matched twin covariance closure is promoted: finite-difference retinal
      translation sources predict a reliable component of recorded FEM-linked
      covariance above strong nulls.
- [x] Feature-space compact-only / compact-removed / compact-addback
      decomposition is complete for the promoted 4C metric.
- [x] Compact-only retains much of full joint feature recovery.
- [x] Compact removal collapses recovery toward zero-eye.
- [x] Compact addback reconstructs full joint to numerical tolerance.
- [x] Static-response-PC controls are nearly as strong as compact on several
      endpoints.

Still needed for draft rigor:

- [ ] Phrase compact as useful/shared manifold structure, not a unique compact
      eye-movement code.
- [ ] State explicitly that compact uniqueness over static PCs is not
      supported.
- [ ] If the draft claims compact geometry is "functionally relevant," specify
      the intervention evidence: compact-only, compact-removed, addback.
- [ ] Do not imply the joint decoder explicitly uses the compact basis as an
      eye-trajectory prior unless that is true for the specific analysis.
- [ ] Do not make compact geometry necessary for 4C to succeed. Compact should
      explain or localize recovery after the primary observer is established.
- [ ] Reserve the "trivial consequence of translation" question for the compact
      vs static-PC / residualized compact / covariance-closure controls already
      tied to this claim.

Allowed wording:

```text
Compact translation geometry is functionally important in the current
projection intervention: compact-only preserves much of the latent-eye feature
recovery and compact removal collapses recovery toward zero-eye. However, much
of this structure overlaps the ordinary static image-response manifold.
```

Avoid:

```text
Compact geometry is a unique eye-position code.
Compact geometry by itself proves active sensing.
```

## Claim 9: Along/Across Motion Should Be Framed As A Mechanism Question

Draft role:

```text
The draft currently shows behavior contour alignment and earlier Vernier
along/across diagnostics. The downstream feature-decoder story should not
collapse this into "along motion is always useful."
```

Hidden assumptions:

- Contour following could mean enhanced along-edge sampling, suppressed
  across-edge sampling, fixed-total anisotropy, or a combination.
- Along/across readouts may depend on candidate set, scale, readout, and
  whether the trajectory is known or latent.

Already checked:

- [x] Matched-static hidden-eye feature-posterior branch favors along-edge.
- [x] Hard-negative branch weakens or reverses a universal along-edge claim.
- [x] Known-axis feature diagnostic favors across-contour feature recovery.
- [x] Edge-parallel preservation of pixels and V1-twin responses is strong.
- [x] RR100 matched-total anisotropic Brownian check is basically null.
- [x] Behavior is contour-following but raw edge remains the baseline.

Still needed for draft rigor:

- [ ] Avoid saying "along motion helps" without specifying the observer and
      control condition.
- [ ] Add a decision note: the theoretical alternative is "reduced across-edge
      displacement may be useful" rather than "extra along-edge displacement is
      useful."
- [ ] If the draft uses along/across behavior figures, state whether they show
      contour-parallel RMS increase, contour-normal RMS reduction, or a ratio.
- [ ] Do not connect behavior alignment to model objective optimization without
      raw-edge residual adjudication.

Allowed wording:

```text
The current evidence supports an edge-relative mechanism question: local
geometry can define motion axes, but existing diagnostics do not show a
universal along-edge policy. A key unresolved distinction is whether useful FEM
structure reflects enhanced along-edge sampling or reduced damaging
across-edge displacement.
```

Avoid:

```text
Along-edge motion is intrinsically optimal.
Behavior confirms the model's along-edge objective.
```

## Draft Revision Checklist

Before turning `draft1.docx` into a Results-style section:

- [ ] Add section headings that match the argument steps:
      RR100 representation, Vernier pose-confusion, behavior geometry,
      feature target, aggregate decoder, local pairing, joint decoder, compact
      mechanism.
- [ ] For every figure, add a one-sentence claim and a one-sentence boundary.
- [ ] For every decoder result, state population, split, target, response
      summary, score, and baseline.
- [ ] Replace placeholder text with explicit scientific claims.
- [ ] Re-express 4B and 4C feature-observer results on the same primary metric
      where possible: pooled held-out `R2_cv` / normalized MSE for the locked
      steerable-pyramid-like magnitude target. Implementation is in place and
      smoke-verified; production artifacts still need rerun or postprocessing.
- [ ] Keep 4B diagonal information, 4C feature cosine / posterior recovery,
      image identity, and Vernier Fisher/discriminability labeled as separate
      secondary or diagnostic quantities unless explicitly recomputed onto the
      shared score.
- [ ] Mark whether each claim is promoted, supportive, diagnostic, historical,
      or open.
- [ ] Do not let any paragraph imply exact trajectory optimality.
- [ ] Do not let compact geometry imply uniqueness over static PCs.
- [ ] Do not let behavior imply model-objective optimization.
- [ ] Include the deeper theoretical alternatives in the Discussion:
      motion-prior choice, feature-decoder model choice, compact-vs-static
      manifold, and along-helpful versus across-suppressed mechanisms.

## Minimal Promotion Table

| Draft step | Current status | Main gate before strong wording |
| --- | --- | --- |
| RR100 representation | Supportive / representation candidate | Label population used in each downstream result |
| Vernier pose confusion | Supportive diagnostic | Do not call pose-robust decoder true trajectory inference |
| Behavior geometry | Supportive bridge | Keep raw edge as baseline; avoid objective optimization |
| Feature target | Supportive methods contract | Specify target family, PCA, covariance approximation |
| Aggregate feature decoder | Provisional 4B; R2_cv smoke verified | Production re-score as unified feature `R2_cv`; all-readout review and OU/synthetic prior audit for strong specificity |
| Local pairing | Diagnostic | Corrected source-trial exact-pairing stability |
| Joint feature-eye decoder | Repaired diagnostic, not promoted | Primary response-space/static-PC observer must pass known > joint > baseline on shared `R2_cv`; compact-forward branch is only an audit |
| Compact mechanism | Promoted with specificity caveats | State non-uniqueness over static PCs and do not make compact required for 4C |
| Along/across mechanism | Diagnostic / scoped 4D | Separate along benefit from across suppression |
