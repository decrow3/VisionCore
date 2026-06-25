# Companion: Along-Edge Model Feature Encoding

Date: 2026-06-22
Status: provisional methods/logic companion for Figure 4D

## Panel Claim Under Test

```text
Along-edge motion benefits model feature encoding.
```

This is the result Panel 4D is there to show if the evidence supports it. The
claim is model-scoped: in the matched-static hidden-eye feature decoder,
trajectory priors aligned with local edges should recover more feature signal
than matched across-edge priors. Everything below is organized to decide how
strongly that sentence can be made. The companion should also prevent the
stronger, not-yet-supported reading that animals universally choose
edge-parallel motion or optimize this exact model objective.

## Summary

Panel 4D is the bridge between the compact latent-eye decoder in 4C and the
behavioral edge-following result in 4E. It asks whether local image geometry
defines a useful motion axis in the model. The current promoted readout is a
matched-static feature-posterior contrast: along-edge trajectory priors recover
more feature signal than across-edge priors at the scoped 0.5x condition.

The result is positive but bounded. The matched-static branch supports the
along-edge feature-encoding claim; hard-negative controls and raw-edge
guardrails prevent a universal policy claim. Edge-parallel pixel and V1-twin
preservation remain supporting mechanism evidence rather than the main D
endpoint. This division matches the analysis narrative: the axis-conditioned
BackImage observer branch supports trajectory-aware feature recovery, but not
yet a clean along-contour mechanism across every candidate set.

## Motivation

If compact joint decoding works because motion samples a structured response
subspace, the next question is which local motion directions are useful. Natural
images are anisotropic: along an edge, small translations can preserve local
structure, while across-edge translations can change local contrast and feature
identity more abruptly. Panel 4D tests whether that geometric intuition appears
in the feature-posterior observer.

The point is not to prove that measured drift always runs along edges. That is
the behavior-side question in 4E. The model-side question is narrower: when
the observer is forced to use local edge-aligned or edge-orthogonal trajectory
priors, does along-edge motion better support feature recovery?

## Notation And Estimator Contract

For an image/window `I` with local edge axis `e`:

```text
tau_parallel: trajectory prior aligned with e
tau_orthogonal: trajectory prior orthogonal to e
y = f_theta(I, tau): V1-twin response movie
phi(I): local image feature target
D(y, phi): feature-posterior score, reported as negative MSE or cosine
```

The core contrast is:

```text
G_parallel = D_joint(I, tau_parallel, phi) - D_zero(I, phi)
G_orthogonal = D_joint(I, tau_orthogonal, phi) - D_zero(I, phi)
axis benefit = G_parallel - G_orthogonal
```

The panel claim is supported when `axis benefit > 0` in the matched-static
feature-posterior setting, with uncertainty and guardrails reported.

## Plain-English Methods

The 4D analysis asks whether the direction of local motion matters for feature
recovery in the model. Each image window is assigned a local edge axis. Motion
along that axis is called along-edge or parallel motion. Motion rotated by 90
degrees is called across-edge or orthogonal motion.

The image axis is estimated from the local image patch before looking at the
model result. This patch-average axis is a useful simple summary, but it can
miss cases where a window contains more than one salient contour. That is why
winner-take-all or salient-contour axis variants remain sensitivity checks
rather than hidden assumptions.

For each image window, two matched trajectory-prior catalogs are built. One
catalog contains trajectories aligned with the local edge axis. The other
contains trajectories aligned across the edge. The two catalogs are intended to
match in basic motion scale and construction, so the main difference is their
direction relative to the local image structure.

Implementation note: the along/across manipulation applies to the latent
trajectory prior catalogs. In the saved response-table caches, `known-eye`
means the candidate response under the true empirical observed trace, while
`zero-eye` means the candidate response under the static trace. Those two
control tables are shared across the parallel and orthogonal prior-family
files. Thus known-eye feature cosine is expected to be identical for
`axis_edge_parallel` and `axis_edge_orthogonal`; it is a ceiling/control, not a
test of two rotated known trajectories.

The V1 twin is run on retinal movies generated from these priors. The observer
then performs the same kind of latent-eye feature recovery used in the joint
posterior analyses: it does not get the true trajectory label, so it must use
the candidate trajectory prior to recover the local image feature target from
the response movie.

The main score is the feature-posterior gain above the zero-eye observer. In
plain English, we ask how much better the moving-prior observer recovers the
feature than an observer that pretends the eye never moved. We compute that
gain for the along-edge prior and for the across-edge prior, then subtract
across-edge from along-edge.

The current promoted result uses the matched-static candidate set at the 0.5x
scale with the `pyramid_local_field` k8 feature target. Matched-static
candidates make the task harder by reducing easy static-response shortcuts. A
hard-negative branch is kept as a guardrail because it does not show the same
positive along-edge pattern. That means the panel can claim a scoped
matched-static model result, not a universal edge-parallel policy.

A newer known-axis diagnostic asks a simpler different question: if the rotated
along- or across-axis trace is known, which response movie gives better feature
posterior recovery? In the same matched-static 0.5x, `pyramid_local_field` k8
cache, this direct test favors across-contour motion:

```text
across-contour known-axis feature cosine = 0.8834
along-contour known-axis feature cosine  = 0.8758
along-minus-across                       = -0.0076
CI                                       = [-0.0096, -0.0057]
p                                        = 0.0010
```

This does not invalidate the hidden-eye D2 result, but it narrows the claim.
D2 should be read as "along-edge priors helped the latent-eye matched-static
observer," not as "known along-edge motion is intrinsically the best feature
alignment direction."

The preservation analyses are separate support. They move image patches along
and across local edges and measure how much the pixels and V1-twin responses
change. These tests support the intuition that along-edge motion is locally
stable, but they are not the main feature-recovery endpoint.

## Assumptions

A1. The local image-axis estimator is a meaningful summary of the patch
geometry for the tested window.

A2. Matched-static candidate sets remove the easiest static-response shortcut,
so the axis contrast is about latent-eye feature recovery rather than trivial
image separability.

A3. Along-edge and across-edge trajectory priors are matched except for their
axis relation to the local image structure.

A4. The feature-posterior endpoint is the relevant promoted D readout; image
identity and preservation audits are supporting context.

A5. A scoped positive model contrast is not equivalent to an animal policy
claim.

## Controls

Matched-static candidate set:

```text
Forces the observer to solve a harder latent-eye feature recovery problem, not
an easy static-response discrimination.
```

Hard-negative guardrail:

```text
Tests whether the along-edge advantage generalizes under a more difficult
candidate set. The current hard-negative branch weakens the universal axis
claim and should remain visible.
```

Image-identity observer:

```text
Useful supporting context, but not the promoted feature-posterior endpoint.
```

Edge-parallel preservation audit:

```text
Shows that along-edge displacement preserves pixels and V1-twin responses
relative to orthogonal displacement. This supports the mechanism but does not
replace the feature-recovery contrast.
```

Raw-edge and objective guardrails:

```text
Prevent the panel from being read as evidence that model-derived objective axes
explain behavior beyond image geometry.
```

## Existing Evidence

Current selected D source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_matched_static_feature_posterior_gabor_pyramid_k4_8_uncertainty_v2/
    feature_posterior_summary.csv
```

Historical axis-conditioned source lineage:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n64_c4_k16_v1/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
```

These runs established the panel-order correction: both clean n64 observer
branches showed joint-eye rescue above zero-eye, but the preferred axis depends
on the candidate set. Matched-static weakly favors edge-parallel; hard-negative
weakly favors edge-orthogonal. The older stronger `target128` orthogonal result
is diagnostic-only because it predates the unmatched-catalog fix.

Selected v5 value table:

```text
declan/figure4_active_sensing_atlas/figures/composites/
  figure4_selected_v5_panel_d_values.csv
```

Matched-static 0.5x, `pyramid_local_field` k8 feature posterior:

```text
along-edge joint-zero feature gain = +6.052 [-MSE]
across-edge joint-zero feature gain = +3.684 [-MSE]
paired along-minus-across = +2.368
CI = [+0.392, +4.589]
p = 0.0257
```

Matched-static image-identity support:

```text
edge-parallel joint accuracy = 0.859
edge-orthogonal joint accuracy = 0.828
parallel-minus-orthogonal = +0.031
```

Known-axis feature-alignment diagnostic:

```text
script:
  declan/figure4_active_sensing_atlas/scripts/build_panel_d_known_axis_feature_alignment.py

outputs:
  declan/figure4_active_sensing_atlas/figures/panel_D/diagnostics/
    known_axis_feature_alignment/

read:
  Uses the saved rotated axis-conditioned response tables as synthetic
  observations with trajectory index known. Across-contour feature cosine is
  higher than along-contour feature cosine, so this is a guardrail against
  interpreting D2 as a direct known-trace along-motion advantage.
```

Hard-negative guardrail:

```text
n64 feature posterior parallel-minus-orthogonal = -0.745
CI = [-3.147, +1.631]
n64 image-identity observer weakly favors orthogonal, 0.891 versus 0.844
```

Historical narrative read:

```text
The safest current interpretation is trajectory-aware feature recovery, not a
clean universal along-contour mechanism split. The matched-static branch gives
the positive 4D panel claim; the hard-negative branch keeps the claim scoped.
```

Mechanism support from preservation:

```text
pixel preservation edge-parallel advantage = 300.54
CI = [172.789, 408.961]
positive sessions = 26 / 29

V1-twin preservation advantage = 0.000454497
CI = [0.000371047, 0.000536519]
positive sessions = 29 / 29
```

## Diagnostics And Failure Modes

The D claim can be overstated in several ways:

```text
the matched-static positive contrast may not generalize to hard negatives;
the local image-axis estimator may average over multiple salient contours;
feature recovery, image identity, and preservation endpoints may disagree;
raw edge geometry may explain behavior without any model-objective residual;
the result may depend on scale, candidate construction, or axis catalog.
older target128 axis results can be mistaken for current shared-source evidence.
```

Current handling:

```text
Use the matched-static feature-posterior contrast as the main D readout.
Keep hard-negative and raw-edge guardrails in caption/supplement.
Treat preservation as mechanism support.
Avoid language that says along-edge motion is a universal policy or behavioral
objective.
```

## Current Claim Boundary

Supported:

```text
In the scoped matched-static hidden-eye feature-posterior observer,
along-edge trajectory priors recover more local feature signal than matched
across-edge priors.
```

Not yet supported:

```text
Along-edge motion is universally optimal across candidate sets and scales.
Animals choose drift axes by optimizing this observer.
Model-derived axes explain behavior beyond raw local edge geometry.
Patch-average edge orientation is the only behaviorally relevant axis estimate.
```

## Production Rerun Implications

Before promoting a stronger version of the D claim, the production package
should report:

```text
matched-static along/across feature-posterior contrast;
hard-negative along/across guardrail;
image-identity support as secondary context;
edge-parallel preservation as mechanism support;
axis-estimator sensitivity, including salient/WTA contour catalogs if promoted;
raw-edge residual behavior tests before any behavioral objective claim.
```
