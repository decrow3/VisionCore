# Prescription: GPL-Clean Geometry-Aware Vernier Joint Observer

## Purpose

Implement an independently written, geometry-aware joint observer for the Vernier active-sensing pilot. The goal is to estimate whether compact, eye-movement-induced response geometry improves Vernier discrimination over a zero-eye baseline, while keeping the implementation clean of GPL-covered source code.

This applies published joint-inference ideas only at the level of Bayesian nuisance marginalization. The Vernier sign is the desired latent variable, and eye trajectory is a nuisance variable. Unlike high-dimensional image reconstruction, this pilot should not treat eye-path reconstruction as the primary objective.

This document is the implementation source of truth. Do not copy code, comments, file structure, control flow, tests, or naming conventions from any GPL-licensed repository. Treat any GPL repository only as conceptual background, or preferably do not inspect it at all while implementing this work.

## Licensing Guardrail

The project may be conceptually inspired by published joint eye-movement inference ideas, but this implementation must be original.

Requirements:

- Do not paste, translate, mechanically rewrite, or adapt GPL-covered code.
- Do not copy function bodies, class layouts, comments, tests, file organization, variable names, or exact control flow from the reference repo.
- Implement from the mathematical and behavioral specification below.
- In the final coding-agent summary, state: "Implemented independently from the provided specification; no GPL-covered source code was copied."
- If adding a provenance note to the repo, use neutral wording such as: "This module implements a particle-based joint observer from first principles for the Vernier pilot."

## High-Level Objective

The current pilot has a compact joint observer based on fitting latent eye pose. The desired change is to make the joint observer evaluate Vernier hypotheses by marginalizing over plausible eye trajectories rather than selecting a single best MAP trajectory. The target decision statistic is the Vernier likelihood ratio:

```text
LLR =
  log sum_w p(response | theta = +delta, w) p(w | condition)
  - log sum_w p(response | theta = -delta, w) p(w | condition)
```

Pose posterior quality can be reported as a secondary diagnostic, but the primary scientific question is whether the nuisance-marginalized Vernier decision improves over zero-eye and approaches known-eye.

Replace or supplement the current "best pose fit per Vernier candidate" score with a particle posterior over eye trajectories:

- Vernier hypothesis: `theta in {-delta, +delta}`.
- Eye trajectory: `x_1:T`, where each `x_t` is a 2D translation/state.
- Motion prior: local, discrete, approximately Gaussian step prior over `x_t - x_{t-1}`.
- Neural likelihood: compact residual likelihood under the local geometry chart for each `theta`.
- Hypothesis score: marginal log evidence `log p(y_1:T | theta)` approximated by particles, not the minimum loss trajectory.

The intended comparison remains:

- `zero_eye`: assumes no eye movement.
- `known_eye`: uses the true/simulated eye trajectory.
- `joint_eye`: infers/marginalizes eye trajectory from responses.
- Controls: correct geometry, wrong geometry, random geometry, gain-only/no-translation geometry.

## Conceptual Model

For each Vernier candidate `theta`, each time bin `t`, and candidate eye state `x_t`, predict the compact response:

```text
z_t(theta, x_t) = U(theta)^T [mu(theta, x_t) - mu0(theta)]
```

where:

- `mu(theta, x_t)` is the population response predicted for Vernier offset `theta` under eye displacement `x_t`.
- `mu0(theta)` is the response at zero eye displacement.
- `U(theta)` is a compact geometry basis, usually derived from the local translation Jacobian around `theta`.

For observed compact response `z_obs_t`, use a Gaussian likelihood:

```text
log p(z_obs_t | theta, x_t)
  = -0.5 * (z_obs_t - z_t(theta, x_t))^T Sigma_z^-1 (z_obs_t - z_t(theta, x_t))
    -0.5 * logdet(Sigma_z)
    + constant
```

The constant may be omitted if all hypotheses are scored with identical dimensionality and covariance. Be consistent across `zero_eye`, `known_eye`, and `joint_eye`.

Important: after projection into the compact basis, the noise covariance is generally not diagonal. If the original response noise is diagonal with covariance `Sigma_y`, the compact covariance is:

```text
Sigma_z(theta) = U(theta)^T Sigma_y U(theta)
```

If the current code assumes isotropic or diagonal compact noise, keep that mode available for compatibility, but add a full-covariance option and make it the preferred/default path for the geometry-aware observer.

## Core Algorithm

Implement a sequential particle approximation for each Vernier hypothesis.

Inputs:

- `theta`: Vernier hypothesis.
- `z_obs[time, compact_dim]`: observed compact residuals.
- `chart`: geometry mapping from 2D eye displacement to compact response.
- `step_prior`: discrete distribution over 2D eye steps.
- `num_particles`.
- `resample_threshold` or equivalent effective sample size threshold.
- `likelihood_scale` or `temperature`, optional.

State per particle:

- Current eye position `x_t`.
- Cumulative log weight.
- Optional full trajectory if needed for diagnostics.

Initialization:

- Start all particles at zero eye position, unless the experiment config specifies an initial distribution.
- Initial log weights are uniform.

At each time bin:

1. For each particle, sample a candidate 2D step from the motion prior.
2. Update position: `x_t = x_{t-1} + step`.
3. Compute compact prediction under `theta` and `x_t`.
4. Add scaled log likelihood for `z_obs_t`.
5. Normalize particle log weights with log-sum-exp.
6. Optionally resample if effective sample size falls below a threshold.
7. Optionally merge or trim duplicate/low-weight trajectories for speed, but correctness matters more than clever pruning.

Hypothesis evidence:

Approximate the marginal log evidence incrementally. A standard approach is:

```text
logZ_t = logsumexp(previous_log_weights + transition_log_prob + log_likelihood_t)
```

Then normalize weights after each observation. The final hypothesis score is the accumulated log evidence across time.

If implementing a bootstrap particle filter where particles are sampled from the transition prior, the transition probability is represented by the sampling distribution and does not need to be explicitly added to the importance weight. If enumerating all possible steps instead of sampling, include the log step prior in the step score.

Choose one of these two approaches and document it in code:

- Bootstrap SIR: sample from prior, weight by likelihood.
- Enumerated step update: branch over discrete steps, weight by `log_prior_step + log_likelihood`.

For this pilot, an enumerated step update can be more stable and deterministic if the step grid is small.

## Recommended Initial Implementation

Prefer the deterministic enumerated-step version first. It is easier to test and less noisy.

For each time bin and each current weighted state:

1. Enumerate allowed steps, for example:

```text
dx, dy in {-max_step, ..., +max_step}
```

2. Assign each step a discrete Gaussian prior:

```text
log p(step) proportional to -0.5 * (dx^2 + dy^2) / step_sigma^2
```

3. Form candidate next states.
4. Score each candidate by:

```text
candidate_log_weight =
  previous_log_weight
  + log_step_prior
  + compact_log_likelihood(theta, candidate_state, z_obs_t)
```

5. The evidence increment for this time bin is:

```text
logZ_increment = logsumexp(candidate_log_weight)
```

assuming previous weights are normalized log posterior weights.

6. Normalize:

```text
new_log_weight = candidate_log_weight - logZ_increment
```

7. If the number of candidates exceeds `max_particles`, keep the top `max_particles` by posterior weight. Track the amount of discarded posterior mass as a diagnostic.

This is technically a beam-filtered hidden Markov model approximation. That is acceptable for the pilot and avoids stochastic test failures.

## Scoring Definitions

All score families must use the same likelihood/objective units.

Do not compare:

- joint posterior score including motion prior penalties

against:

- zero/known residual-only score

unless the labels clearly distinguish them.

Preferred outputs:

```text
zero_log_evidence[theta]
known_log_evidence[theta]
joint_log_evidence[theta]
```

Then classify by:

```text
predicted_theta = argmax_theta log_evidence[theta]
```

For zero-eye:

```text
x_t = (0, 0) for all t
zero_log_evidence(theta) = sum_t log p(z_obs_t | theta, x_t = 0)
```

For known-eye:

```text
x_t = true trajectory
known_log_evidence(theta) = sum_t log p(z_obs_t | theta, x_t = true_x_t)
```

For joint-eye:

```text
joint_log_evidence(theta) = log sum over trajectories p(z_obs_1:T | theta, x_1:T) p(x_1:T)
```

If reporting losses rather than log evidence, convert consistently:

```text
loss = -log_evidence
```

Avoid mixing residual sum of squares, posterior losses, and marginal log evidence in one closure metric.

## Gap-Closure Metric

If the existing code reports something like:

```text
gap_closure_vs_zero_known =
  (joint_score - zero_score) / (known_score - zero_score)
```

revise it so all three scores are the same type and orientation.

For log evidence, higher is better:

```text
gap_closure =
  (joint_log_evidence_true - zero_log_evidence_true)
  / (known_log_evidence_true - zero_log_evidence_true)
```

For loss, lower is better:

```text
gap_closure =
  (zero_loss_true - joint_loss_true)
  / (zero_loss_true - known_loss_true)
```

Guard against tiny denominators:

```text
if abs(denominator) < eps:
    gap_closure = nan
```

Also report classification accuracy separately from gap closure.

## Geometry Controls

Implement or preserve these geometry modes:

### Correct Geometry

Use the correct local translation chart for each Vernier hypothesis.

Expected behavior:

- Joint observer should improve over zero-eye when eye movements create informative compact residuals.
- Joint observer should approach known-eye when the trajectory posterior is well identified.

### Wrong-Theta Geometry

Use the geometry chart from the opposite Vernier hypothesis while scoring a candidate.

Expected behavior:

- Should reduce performance if hypothesis-specific geometry matters.

### Random-Basis Geometry

Use a random orthonormal compact basis with the same dimensionality as the correct basis.

Expected behavior:

- Should preserve dimensionality and noise level but remove meaningful translation geometry.

### Gain-Only Geometry

Use a control chart that captures global gain or low-rank response changes but not spatial translation.

Expected behavior:

- Helps test whether performance comes from true reafferent translation structure rather than generic modulation.

Important: the `known_eye` upper-bound score should always use the correct known-eye prediction if it is intended as an upper bound. If the user asks to evaluate known-eye under a wrong/random control chart, label it explicitly as a control score, not as the upper bound.

## Suggested File-Level Integration

Adapt names to the actual repository style. Do not create large abstractions if the repo is currently simple.

Likely targets:

- `joint_observer.py`
- `run_vernier_active_sensing.py`
- existing test/smoke-test files, or a new focused test file
- README or experiment notes, if the repo already documents CLI flags

Recommended additions:

```text
build_discrete_gaussian_step_prior(max_step, sigma)
compact_log_likelihood(...)
score_zero_eye_evidence(...)
score_known_eye_evidence(...)
score_joint_eye_evidence_enumerated(...)
class JointObserverResult or dict-like result object
```

Keep public outputs explicit:

```text
{
  "theta_values": ...,
  "zero_log_evidence": ...,
  "known_log_evidence": ...,
  "joint_log_evidence": ...,
  "pred_zero": ...,
  "pred_known": ...,
  "pred_joint": ...,
  "gap_closure": ...,
  "posterior_diagnostics": ...
}
```

Recommended diagnostics:

- number of retained particles/states per time bin
- effective sample size, if using sampled particles
- retained posterior mass, if using top-k pruning
- posterior mean eye position per time bin
- posterior covariance or spread per time bin
- trajectory entropy or final-state entropy

## CLI / Configuration

Expose small, explicit controls:

```text
--joint-observer {map,enumerated,particle}
--geometry-control {correct,wrong_theta,random_basis,gain_only}
--compact-k INT
--eye-step-max INT
--eye-step-sigma FLOAT
--max-particles INT
--likelihood-scale FLOAT
--full-compact-covariance / --diagonal-compact-covariance
--random-seed INT
```

Defaults should favor deterministic reproducibility:

```text
--joint-observer enumerated
--geometry-control correct
--eye-step-max 1 or 2
--eye-step-sigma around 1.0
--max-particles 1000 to 5000
--full-compact-covariance true
```

Retain the old MAP observer behind `--joint-observer map` if it is useful for comparison, but do not make it the main reported joint result.

## Numerical Stability Requirements

Use log-space for all posterior/evidence calculations.

Implement or use existing helpers for:

```text
logsumexp
stable covariance inverse or Cholesky solve
safe log determinant
normalizing log weights
```

Covariance handling:

- Add small diagonal jitter before Cholesky decomposition, for example `1e-6` times the average variance.
- If Cholesky fails, increase jitter in a bounded way and report a warning/diagnostic.
- Do not silently fall back to a different scoring model.

## Tests

Add focused tests that can run quickly.

### Test 1: Step Prior Normalization

Verify:

- probabilities are nonnegative
- probabilities sum to one
- zero step has the highest probability
- opposite steps have equal probabilities

### Test 2: Log Evidence Prefers True Hypothesis With Known Eye

Create a tiny synthetic fixture:

- two Vernier hypotheses
- short trajectory
- compact observations generated from the true hypothesis and true trajectory
- low or zero noise

Expected:

- `known_eye` assigns higher log evidence to the true hypothesis.

### Test 3: Joint Marginalization Beats Zero Eye In Geometry-Informative Case

Use the same synthetic fixture with nonzero eye trajectory.

Expected:

- `joint_eye` true-hypothesis evidence exceeds `zero_eye` true-hypothesis evidence.
- true hypothesis is classified correctly by `joint_eye`.

### Test 4: No Mixed Score Units

Assert that gap closure uses either all log evidence or all loss values.

This can be a direct unit test for the metric function:

- Feed known values.
- Check orientation.
- Check zero denominator returns `nan` or documented sentinel.

### Test 5: Wrong/Random Geometry Does Not Masquerade As Known-Eye Upper Bound

Verify:

- the normal `known_eye` score uses correct geometry
- control-specific known scores, if present, are named differently

### Test 6: Reproducibility

For stochastic particle mode, fixed seed should produce identical outputs.

For enumerated mode, outputs should be deterministic independent of seed.

## Smoke Experiment

Add or update a lightweight smoke command that completes in under a minute.

It should run:

```text
correct geometry
wrong-theta geometry
random-basis geometry
```

and print a compact table:

```text
mode             zero_acc   joint_acc   known_acc   gap_closure   notes
correct          ...
wrong_theta      ...
random_basis     ...
```

The smoke test does not need to prove the scientific claim, but it should catch broken scoring, shape errors, unstable covariance, and non-reproducible defaults.

## Acceptance Criteria

The implementation is acceptable when:

- No GPL-covered code is copied or mechanically adapted.
- The main joint observer score is marginal evidence over eye trajectories, not only the best MAP trajectory.
- `zero_eye`, `known_eye`, and `joint_eye` scores use consistent units.
- The known-eye upper bound is not polluted by wrong/random/gain geometry controls.
- Full compact covariance is supported and used by default where feasible.
- Geometry-control modes are available and clearly labeled.
- Tests cover prior normalization, evidence orientation, gap closure, and at least one synthetic joint-vs-zero case.
- A smoke run produces deterministic, inspectable output.
- The final agent summary states what changed, what tests ran, and confirms no GPL source was copied.

## Explicit Non-Goals

Do not implement a full retinal image reconstruction pipeline.

Do not add a natural-image prior, denoiser, or HQS-style image optimizer.

Do not optimize for large-scale runtime before the small deterministic pilot is correct.

Do not introduce heavy dependencies unless the repo already uses them.

Do not make broad architectural rewrites unrelated to the observer scoring problem.

## Suggested Final Agent Report

The coding agent should report something like:

```text
Implemented an independently written geometry-aware Vernier joint observer using enumerated trajectory marginalization. Added consistent log-evidence scoring for zero-eye, known-eye, and joint-eye observers; corrected gap-closure orientation; added geometry controls; and added focused synthetic tests.

No GPL-covered source code was copied or adapted. Implementation was written from the provided clean-room specification.

Tests run:
- ...

Key files changed:
- ...
```
