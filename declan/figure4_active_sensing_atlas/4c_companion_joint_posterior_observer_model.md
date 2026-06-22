# Companion: Joint Posterior / Trajectory-Table Observer Model

Date: 2026-06-22
Status: provisional methods/logic companion for Figure 4C

## Panel Claim Under Test

```text
Joint eye/image decoding can recover feature information when eye position is
hidden, and the compact translation component is important for that recovery.
```

This is the result Panel 4C is there to show if the evidence supports it. The
claim has two parts: joint image/trajectory inference can recover feature
information when eye position is latent, and compact response structure carries
enough of that recovery that compact removal should damage the endpoint. The
analysis motivations, assumptions, estimator contracts, controls, results, and
caveats below are all there to decide whether that sentence is strong,
qualified, or should be softened.

The current evidence supports that claim, but with an important boundary. It
does not show that the compact translation component is the unique response
structure that makes the observer work. A matched control based on static image
responses is nearly as strong.

## Summary

The current 4C claim is no longer only the older matched-static image-identity
rescue. It now asks whether the compact translation-response component is
important for the feature-posterior endpoint used in the panel. The simplifying
assumption it breaks is still the zero-eye approximation: treating the response
to a moving retinal image as if the image stayed fixed on the retina. But the
promoted result adds a mechanism test: compact-only recovery should retain much
of the joint feature recovery, while compact removal should push recovery back
toward the zero-eye curve.

The older matched-static image-identity observer remains important historical
support because it established that marginalizing over possible trajectories
can rescue image information when eye position is latent. The current panel,
however, is supported by the newer feature-posterior compact-only /
compact-removed / compact-addback analysis. That is the analysis that matches
the title.

The newest static-response-PC tests change the mechanistic interpretation. In
plain language: the compact translation component is real and useful, but it
mostly overlaps with the main response patterns the model already uses for
static images. This means the safe claim is not "there is a special compact
translation-only channel." The safer and better claim is "eye movements push V1
responses along a compact part of the normal image-response manifold."

## Motivation

The aggregate model establishes that response movies can carry feature
structure. The joint observer asks a different question: can image information
remain usable when the observer does not know which eye trajectory generated
the response? This matters because biological downstream circuits rarely get a
perfect external label for retinal pose. Motion is part of the inference
problem, not merely a nuisance to average away.

The trajectory-table observer is intentionally explicit. It builds a finite
catalog of response movies for candidate images and candidate trajectories,
then compares observers or interventions: one that knows the true trajectory,
one that assumes zero motion, one that marginalizes over a trajectory prior,
and compact-subspace variants that keep, remove, or add back the compact
translation component. This makes the compact claim testable rather than a
visual interpretation of a latent space.

## Notation And Estimator Contract

Shared notation:

```text
I: candidate image/window
tau: candidate eye trajectory
y = f_theta(I, tau): V1-twin response movie
p(tau): trajectory prior
p(I): image prior over the candidate set
```

The exact table stores:

```text
T[I, tau] = f_theta(I, tau)
```

For an observed response movie `y_obs = f_theta(I*, tau*)`, the observer scores
each candidate pair with a likelihood:

```text
L(I, tau | y_obs) = p(y_obs | I, tau)
```

The three observer contracts are:

```text
known-eye:
  I_hat = argmax_I L(I, tau* | y_obs)

zero-eye:
  I_hat = argmax_I L(I, tau_0 | y_obs)

joint-eye:
  I_hat = argmax_I sum_tau L(I, tau | y_obs) p(tau)
```

Equivalently, the joint observer forms:

```text
p(I, tau | y_obs) proportional to p(y_obs | I, tau) p(tau) p(I)
p(I | y_obs) = sum_tau p(I, tau | y_obs)
```

The older observer outcome is image-identification accuracy over candidate
sets, with posterior concentration diagnostics such as `N_eff / K` used as
guardrails. The current promoted 4C outcome is feature-posterior recovery,
reported as cosine similarity between recovered and target local feature
vectors.

For the compact-subspace intervention, the contract is:

```text
y_full = f_theta(I, tau)
y_zero = f_theta(I, tau_0)
y_compact_only = project_compact(y_full - y_zero) + y_zero
y_compact_removed = y_full - project_compact(y_full - y_zero)
y_compact_addback = y_compact_removed + project_compact(y_full - y_zero)
```

The claim is supported if:

```text
feature_recovery(compact_only) remains close to feature_recovery(full_joint)
feature_recovery(compact_removed) moves toward feature_recovery(zero_eye)
feature_recovery(compact_addback) reconstructs feature_recovery(full_joint)
```

## Plain-English Methods

The 4C analysis asks what an observer can recover when it sees a response movie
but does not know which eye trajectory produced it. The implementation makes
this inference problem explicit by building a table of model responses for many
candidate images and candidate trajectories.

For each candidate image, the V1 twin is run on several rendered retinal
movies. Each movie uses the same image but a different sampled eye trajectory.
Together these responses form a lookup table: if the image were this one and
the trajectory were that one, this is the response movie the model would
produce.

The observed response is then held out and compared with entries in the table.
The known-eye observer is allowed to compare only against the true trajectory.
This is the ceiling for the finite candidate set. The zero-eye observer ignores
motion and compares the observed response with the static response. The joint
observer does not know the true trajectory, so it averages or sums evidence
over the possible trajectories in the prior catalog.

The older observer analyses scored image identification: did the observer pick
the right image from a candidate set? The current promoted panel scores feature
recovery instead. After the joint observer forms a posterior over candidate
images and trajectories, that posterior is used to recover the local image
feature vector. The panel reports how close the recovered feature vector is to
the true feature vector.

The compact intervention is applied to the motion-induced part of the response.
First, subtract the static response from the moving response. Then project that
motion-induced response onto the compact translation subspace, remove that
component, or add it back. The `compact_only` variant keeps only the compact
motion component plus the static response. The `compact_removed` variant removes
that component from the full moving response. The `compact_addback` variant
checks that adding the removed component back reconstructs the full result.

The main comparison is therefore not just "joint is better than zero-eye." It
asks whether the compact translation component carries much of the useful
feature-recovery signal. The expected pattern is: known-eye is highest,
full-joint and compact-addback are close, compact-only retains much of the full
joint result, and compact-removed falls toward the zero-eye result.

Candidate sets and priors are guarded so the test is not too easy. Historical
matched-static observers selected distractor images with similar stabilized
responses, which reduces the chance that static response strength alone solves
the task. The current compact-removal endpoint uses hard-negative feature
posterior recovery. Leave-one-out trajectory priors prevent the exact observed
trace from being reused as an unfair template.

Uncertainty and interpretation are kept at the endpoint level. The posterior is
used as a tool for recovering image features; it is not interpreted as proof
that the animal, or the observer, exactly identified the true eye trace.

## Assumptions

A1. The finite trajectory table samples enough of the relevant response
variation to make marginalization meaningful.

A2. The likelihood scale is calibrated well enough that accuracy ordering is
not a trivial temperature artifact.

A3. Candidate sets are hard enough to prevent static-response shortcuts. The
matched-static image-identity observer is the historical control for this
assumption; the current compact-removal panel uses hard-negative feature
posterior recovery and should be read with that endpoint in mind.

A4. Leave-one-out trajectory priors prevent the observer from using the exact
observed trace as an unfair template.

A5. The posterior is used for image or feature recovery, not as a claim that
the true trajectory is exactly identified.

A6. Compact-subspace projection is a mechanistic intervention on response
components. It can show importance and sufficiency for this endpoint, but not
uniqueness over all useful low-dimensional components. The current matched
static-PC controls do not fail; they are nearly as strong as compact.

## Controls

Known-eye observer:

```text
Upper reference for image information when the true trajectory is specified.
```

Zero-eye observer:

```text
Failure mode for treating motion as if the image were stabilized.
```

Joint-eye observer:

```text
Tests whether marginalizing over plausible trajectories recovers pose-lost
image information.
```

Matched-static distractors:

```text
Candidate images are selected to have similar stabilized V1-twin responses.
This blocks the trivial explanation that joint-eye wins only through static
response strength.
```

Hard-negative candidate sets:

```text
Stress-test image discrimination against structurally similar alternatives.
```

Empirical and OU priors:

```text
Check whether rescue depends on the exact empirical trajectory prior or on a
broader plausible confined-motion prior.
```

Posterior `N_eff / K`:

```text
Checks whether the posterior is diffuse, concentrated, or collapsing to a
trivial single trajectory.
```

Compact-only projection:

```text
Keeps only the compact translation component of the motion-induced response.
Tests whether that component is sufficient to retain feature recovery.
```

Compact-removed projection:

```text
Removes the compact translation component from the response. Tests whether the
feature-posterior endpoint collapses toward zero-eye when that component is
absent.
```

Compact-addback reconstruction:

```text
Adds the compact component back after removal. This is the algebraic/QC control
that the intervention is targeting the intended response component.
```

Static-PC, gain, random, and unit-shuffle controls:

```text
Compact beats random, unit-shuffled, and gain-only alternatives, while
static-response PCs remain a serious matched contender. The current static-PC
tests are strong enough that compact should be framed as important, not unique.
```

## Existing Evidence

Primary current 4C source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1/
    feature_compact_mechanism_summary.csv
    feature_compact_mechanism_uncertainty.csv
    feature_compact_mechanism_qc.csv
```

Run scope for the promoted endpoint:

```text
candidate mode = hard_negative_structure
feature target = pyramid_local_field
feature PCA k = 8
compact k = 10
motion scales = 0.5x, 1x, 2x
variants = full_exact, zero_static, compact_only, compact_removed, compact_addback
score = feature recovery cosine
```

Current panel values:

```text
zeroed-eye feature recovery:
  0.5x = 0.765
  1.0x = 0.668
  2.0x = 0.576

compact-only feature recovery:
  0.5x = 0.850
  1.0x = 0.838
  2.0x = 0.826

compact-removed feature recovery:
  0.5x = 0.759
  1.0x = 0.635
  2.0x = 0.537

full joint / compact-addback feature recovery:
  0.5x = 0.872
  1.0x = 0.872
  2.0x = 0.871

known-eye ceiling:
  0.5x = 0.927
  1.0x = 0.936
  2.0x = 0.949
```

This is the analysis that directly matches the panel title: compact-only
retains much of full joint feature recovery, while compact removal falls toward
or below the zero-eye curve.

Historical observer source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    observer_summary.csv
```

Run scope:

```text
n_images = 64
n_candidates = 8
candidate modes = hard_negative_structure, matched_static_response
observation family = empirical
prior families = empirical, OU
scales = 0.5x, 1.0x
n_prior_trajectories = 8
trajectory prior mode = leave_one_out
likelihood scales = 0.5, 1.0
```

Accuracy ordering:

```text
hard_negative_structure, 0.5x:
  zero 0.578, joint 0.781-0.844

hard_negative_structure, 1.0x:
  zero 0.312, joint 0.734-0.875

matched_static_response, 0.5x:
  zero 0.578, joint 0.750-0.828

matched_static_response, 1.0x:
  zero 0.328, joint 0.672-0.797
```

Matched-static rescue at 1.0x, likelihood scale 1.0:

```text
empirical prior:
  known-eye = 1.000
  zero-eye = 0.328
  joint-eye = 0.766
  median N_eff / K ~= 0.364

OU prior:
  known-eye = 1.000
  zero-eye = 0.328
  joint-eye = 0.797
  median N_eff / K ~= 0.400
```

The matched-static result remains one of the cleanest historical observer
findings because the zero-eye failure cannot be dismissed as an easy static
response mismatch. It supports the joint eye/image decoding half of the title,
but it is no longer the exact endpoint plotted in current 4C.

Static-response-PC specificity audit:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_symmetric_subspace_removal_prod_v1/

outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_feature_symmetric_subspace_removal_prod_v1/

outputs/matched_twin_covariance_closure_static_pc_predictor_prod_v1/
```

Here "static-response PCs" means the main population response patterns seen
when the model views many static image windows. They are not eye-movement
patterns by construction. They are a strong control because small eye movements
move an image locally, and local image changes naturally follow directions that
already live in the static image-response space.

The new tests ask whether compact translation geometry still looks special
after comparing it with those static-image directions.

In the image-identity decoder, removing compact directions and removing
static-response-PC directions hurt by almost the same amount:

```text
full-minus-removed accuracy loss:
  compact removed   = 0.337
  static-PC removed = 0.315
```

In the feature-recovery endpoint used by Panel 4C, compact removal hurts
somewhat more, but static-PC removal is still nearly as damaging:

```text
full-minus-removed feature neg-MSE loss:
  compact removed   = 45.042
  static-PC removed = 38.900
```

When the shared overlap between compact and static PCs is removed first, very
little unique structure remains:

```text
feature neg-MSE loss after removing only residualized pieces:
  compact residual after static PCs = 3.670
  static-PC residual after compact  = 0.963
```

The covariance predictor test is the strongest check because it is not just a
"how much tangent variance can this basis capture?" test. It asks which basis
actually predicts recorded FEM-linked covariance. In the current full-sample
Allen `2022-02-16` run, the static-PC predictor matches or slightly beats the
compact predictor:

```text
k=10 covariance capture:
  no projection:
    compact predictor   = 0.742
    static-PC predictor = 0.748

  after global_rate+target_pc1 projection:
    compact predictor   = 0.417
    static-PC predictor = 0.425
```

These gaps are small and change sign across endpoints: compact removal is a bit
worse in the ablation tests, while static PCs are a bit higher in the covariance
prediction test. That is the pattern expected if there is no reliable
compact-specific advantage over static PCs.

Plain-English conclusion:

```text
The compact component matters. But what matters is mostly the part it shares
with the ordinary image-response manifold. Current results support a shared
manifold story, not a compact-only mechanism.
```

## Mechanistic Posture

Figure 4C can be written as a mechanistic bridge, but not yet as a definitive
mechanism proof.

The current evidence supports the claim that structured motion-induced response
variation in the V1 twin can be used by a joint image-and-eye observer. The
compact-geometry analyses sharpen that statement by showing that an
image-disjoint compact translation subspace is a plausible carrier of much of
the motion-dependent likelihood structure. This is stronger than a purely
descriptive observer result, because it identifies a concrete response
component that can preserve the latent-eye rescue.

The stricter feature-space compact-removal test is now complete for the current
promoted endpoint. The newer static-PC tests show that the compact component is
not cleanly separable from the response directions already used for static
images. The safe language is:

```text
compact geometry supports or carries much of the joint-observer rescue;
compact-only is a sufficiency/mechanism guardrail;
compact-removed loss is evidence that this response component is important;
static-response PCs are a matched control and perform nearly as well;
compact geometry is not the currently supported unique mechanism.
```

Avoid language implying that the decoder explicitly uses a compact trajectory
prior, that the compact basis uniquely explains the rescue, or that the
posterior identifies the animal's true eye trajectory.

The best current mechanistic wording is:

```text
Eye movements push the V1 population along a compact set of response directions.
Those directions are largely shared with the main response patterns evoked by
static images. This shared structure can support latent-eye feature recovery,
but it should not be described as a dedicated compact-only eye-movement code.
```

## Diagnostics And Failure Modes

The main failure modes are:

```text
joint-eye wins by static response strength rather than motion-aware inference;
the scale effect is only zero-eye failure;
the posterior collapses to a meaningless or trivial trajectory;
the prior catalog leaks the observed trace;
candidate hardness or source imbalance drives the effect;
compact-mechanism projections are overread as unique mechanisms;
static PCs explain comparable useful structure and are not cleanly beaten.
```

Current handling:

```text
Keep compact-only, compact-removed, compact-addback, zero-eye, and known-eye in
the main panel.
Route matched-static image identity and posterior concentration to historical
support or supplement.
Use compact projection as importance/sufficiency evidence, not a
unique-mechanism proof over static PCs.
Mention that static-PC controls are nearly matched whenever the compact
mechanism is interpreted.
```

## Current Claim Boundary

Supported:

```text
When retinal pose is latent, joint image/trajectory inference recovers local
feature information above a zero-eye observer, and the compact translation
component is important for that recovery: compact-only retains much of the full
joint feature recovery, while compact removal collapses recovery toward the
zero-eye curve.
```

Also supported:

```text
The useful compact component is mostly shared with static image-response
directions. This means Panel 4C can say that a shared compact image-response
structure supports the latent-eye recovery, not that compact geometry is unique.
```

Not yet supported:

```text
The posterior exactly identifies the animal's true eye trajectory.
Empirical FEMs are proven optimal relative to all plausible priors.
The compact translation basis is the unique mechanism of the rescue relative to
static response PCs or every other low-dimensional response component.
The decoder uses compact geometry as an explicit eye-trajectory prior.
Axis-specific edge-parallel behavior follows directly from this observer.
```

Now disfavored by the completed static-PC tests:

```text
The finite-difference compact basis is privileged over the static image-response
manifold for explaining FEM-linked covariance.
The compact translation subspace should be presented as a dedicated
eye-movement-only code.
```

## Production Rerun Implications

The final Figure 4C panel should preserve four contracts:

```text
known-eye ceiling above latent-eye recovery
compact-only near full joint recovery
compact-removed near zero-eye recovery
static-PC controls described as nearly matched, not just as minor guardrails
```

If the optional prior-depth target is run:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_n128_c4_k16_rel0p25_prior32_power_v1
```

then it should be treated as a power and prior-depth extension for the
trajectory-table observer. It should not replace the current 4C
feature-posterior compact-removal endpoint unless it includes the same compact
projection intervention and the same feature-space score.
