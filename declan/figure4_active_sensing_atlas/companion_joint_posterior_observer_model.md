# Companion: Joint Posterior / Trajectory-Table Observer Model

Date: 2026-06-21
Status: provisional methods/logic companion for Figure 4C

## Summary

The joint posterior observer asks what an idealized decoder can recover when
the retinal trajectory is latent. The simplifying assumption it breaks is the
zero-eye approximation: treating the response to a moving retinal image as if
the image stayed fixed on the retina. If motion meaningfully changes the
response, a zero-eye observer should lose image information. A joint observer
can recover some of that loss by marginalizing over plausible trajectories.

The clean current result is the matched-static rescue: when candidate images
are selected to have similar stabilized V1-twin responses, known-eye decoding
is high, zero-eye decoding collapses, and joint image/trajectory marginalization
rescues much of the gap.

## Motivation

The aggregate model establishes that response movies can carry feature
structure. The joint observer asks a different question: can image information
remain usable when the observer does not know which eye trajectory generated
the response? This matters because biological downstream circuits rarely get a
perfect external label for retinal pose. Motion is part of the inference
problem, not merely a nuisance to average away.

The trajectory-table observer is intentionally explicit. It builds a finite
catalog of response movies for candidate images and candidate trajectories,
then compares three observers: one that knows the true trajectory, one that
assumes zero motion, and one that marginalizes over a trajectory prior.

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

The outcome is image-identification accuracy over candidate sets, with
posterior concentration diagnostics such as `N_eff / K` used as guardrails.

## Assumptions

A1. The finite trajectory table samples enough of the relevant response
variation to make marginalization meaningful.

A2. The likelihood scale is calibrated well enough that accuracy ordering is
not a trivial temperature artifact.

A3. Candidate sets are hard enough to prevent static-response shortcuts. The
matched-static condition is the key control for this assumption.

A4. Leave-one-out trajectory priors prevent the observer from using the exact
observed trace as an unfair template.

A5. The posterior is used for image recovery, not as a claim that the true
trajectory is exactly identified.

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

## Existing Evidence

Primary source:

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

The matched-static result is one of the clearest current model-objective
findings because the zero-eye failure cannot be dismissed as an easy static
response mismatch.

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

The stricter compact-specific mechanism test remains pending. The final
mechanistic claim should wait for the adjudication suite that separates compact
translation geometry from other useful low-dimensional response axes, especially
static-response PCs, gain axes, unit-shuffled compact bases, and
entropy-matched compact-aware trajectory weighting. Until that is complete, the
safe language is:

```text
compact geometry supports or carries much of the joint-observer rescue;
compact-only is a sufficiency/mechanism guardrail;
compact-removed loss is evidence that the compact channel is important;
compact geometry is not yet proven to be the unique mechanism.
```

Avoid language implying that the decoder explicitly uses a compact trajectory
prior, that the compact basis uniquely explains the rescue, or that the
posterior identifies the animal's true eye trajectory.

## Diagnostics And Failure Modes

The main failure modes are:

```text
joint-eye wins by static response strength rather than motion-aware inference;
the scale effect is only zero-eye failure;
the posterior collapses to a meaningless or trivial trajectory;
the prior catalog leaks the observed trace;
candidate hardness or source imbalance drives the effect;
compact-mechanism projections are overread as unique mechanisms.
```

Current handling:

```text
Keep matched-static rescue in the main panel.
Pair accuracy with scale and posterior concentration diagnostics.
Use compact projection only as a sufficiency or mechanism guardrail, not a
unique-mechanism proof.
Label the stricter compact-specific mechanistic test as pending.
```

## Current Claim Boundary

Supported:

```text
When retinal pose is latent, joint image/trajectory marginalization over exact
V1-twin response tables recovers much of the image information lost by a
zero-eye observer, including for matched-static distractors.
```

Not yet supported:

```text
The posterior exactly identifies the animal's true eye trajectory.
Empirical FEMs are proven optimal relative to all plausible priors.
The compact translation basis is the unique mechanism of the rescue.
The decoder uses compact geometry as an explicit eye-trajectory prior.
Axis-specific edge-parallel behavior follows directly from this observer.
```

## Production Rerun Implications

The final Figure 4 observer panel should preserve three contracts:

```text
known-eye > joint-eye > zero-eye ordering
matched-static distractor rescue
posterior concentration as a guardrail
```

If the optional prior-depth target is run:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_n128_c4_k16_rel0p25_prior32_power_v1
```

then it should be treated as a power and prior-depth extension. It should not
replace the matched-static rescue unless it carries an equally clear static
shortcut control.
