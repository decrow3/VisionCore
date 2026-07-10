# Candidate-Free Observer Recovery Protocol

Status: working validation ladder, not a paper-facing result.

The current candidate-free branch is still the most principled route toward a
latent-eye observer claim, but the current implementation has not established a
valid known-trajectory ceiling under the locked pooled `R2_cv` feature-recovery
score.  The branch should be labeled:

`validity_unresolved: true_tau_ceiling_not_established_under_observation_model`

This is deliberately weaker than calling the latent a model-error variable.  The
optimized trajectory fitting the compact response better than recorded tau is a
serious warning, not a final identity claim.

## Non-Negotiable Gate

Do not promote the candidate-free branch unless the same target, folds,
normalization, and pooled `R2_cv` score satisfy at least:

`S_known > S_zero` and `S_joint > S_zero`

The preferred promotion pattern remains:

`S_known > S_joint > S_zero`

Do not report gap recovered when `S_known - S_zero` is non-positive or not
meaningfully above uncertainty.

## Recovery Ladder

1. Self-consistency positive control

Generate synthetic responses from the same observation model used by the
observer:

`r = F(z, tau_true) + noise`

Then run zero, joint, and known observers.  The expected result is:

`S_known > S_joint > S_zero`

If this fails, the issue is in scoring, folds, normalization, inversion, or
observer implementation rather than biology.

2. Direct known-tau feature decoder

Build the known reference as direct feature recovery:

`z_hat_known = g(r, tau_true)`

The conservative implementation is:

`z_hat_known = z_hat_response + alpha h(r, tau_true)`

with `alpha = 0` available and selected by inner-fold pooled `R2_cv`.  Record
internal first-pass metrics so the response-only fallback is auditable on
identical rows.

3. Recorded-tau alignment audit

Before interpreting optimized tau as model-error correction, run a predeclared
coordinate audit:

- x/y swap
- sign flip
- one- or two-bin lag
- scale convention
- retinal-versus-eye coordinate sign
- interpolation or sampling-rate mismatch

These are coordinate-system checks, not metric fishing.

4. Eye trajectory versus nuisance slack

If optimized tau keeps beating recorded tau, test whether a small non-eye
nuisance term explains the mismatch:

`response_t = A e_t + B q_t + noise_t`

Compare recorded eye only, optimized eye only, recorded eye plus optimized
nuisance, and optimized eye plus optimized nuisance.  The nuisance term must be
low-dimensional and explicitly labeled as model-error slack.

5. Failure decomposition

Split the gate by motion scale and feature family before making broad claims:

- 0.5x, 1x, 2x
- real part, imaginary part, magnitude
- orientation
- spatial block
- pyramid scale

If true tau only fails at large motion scale or only for signed phase channels,
that implies a local-model or target-contract problem rather than global failure
of candidate-free latent-eye decoding.

## Current Interpretation

The current v4 biological result says:

The candidate-free implementation has failed the paper gate because the
known-tau ceiling is not valid under the current observation model.

It does not say:

Candidate-free joint decoding is a dead end.

Until the recovery ladder is complete, optimized-tau advantages should be
reported as observation-model validity warnings, not as successful inferred
retinal trajectory recovery.

## Completed Recovery Checks

1. Self-consistency control

Output directories:

- `outputs/figure4_candidate_free_self_consistency_v1`
- `outputs/figure4_candidate_free_self_consistency_iter6_v1`
- `outputs/figure4_candidate_free_self_consistency_iter12_v1`

The deterministic exact-model control establishes a real known-tau ceiling:

`S_known = 0.6457 > S_zero = 0.4676`

With the original two joint iterations, hidden joint is below zero:

`S_joint = 0.4408 < S_zero = 0.4676`

With six joint iterations, the overall synthetic gate passes:

`S_known = 0.6457 > S_joint = 0.4824 > S_zero = 0.4676`

However, the 2x scale remains below zero even with 6 and 12 iterations.  This
points to a joint-inference/prior or large-motion-regime issue, not a total
failure of the compact known-tau observation model.

2. Direct recorded-tau decoder gate

Output directories:

- `outputs/figure4_joint_decoder_known_residual_r2cal_affine_v4/gates_direct_true_tau`
- `outputs/figure4_joint_decoder_known_residual_r2cal_affine_v4/gates_direct_true_tau_cv_gain_calibrated`

The direct recorded-tau interaction decoder has the desired uncalibrated point
ordering:

`S_known = -2.1420 > S_joint = -2.3593 > S_zero = -2.5709`

This ordering does not survive cross-fold scalar gain calibration, so it is not
paper-promotable yet.  It does show that the known-tau signal is not globally
absent.

3. Recorded-tau alignment audit

Output directory:

- `outputs/figure4_recorded_tau_alignment_v1`

The best direct known-tau coordinate variant is `scale_0p5`:

`R2_cv = -1.7903`

Identity recorded tau is:

`R2_cv = -2.1420`

Both beat zero-static in the uncalibrated locked score, but 2x remains worse
than zero.  Sign and axis swaps are not decisive in this direct linear decoder,
because train/test coefficients can absorb those transformations.  Lag and
scale variants are more informative and should be followed up only as
predeclared coordinate-convention checks.

4. 2x self-consistency sweep

Output directories:

- `outputs/figure4_candidate_free_2x_self_consistency_sweep_v1`
- `outputs/figure4_candidate_free_self_consistency_2x_first64_best_v1`

The full 2x synthetic sweep tested 24 settings crossing joint iterations,
Brownian prior scale, feature prior precision, and process variance.  Twelve
settings pass `known > joint > zero`.

The decisive knob is `forward_model_z_prior_precision`:

- `1.0`: all 12 tested settings pass.
- `0.1`: all 12 tested settings fail.

The best full-2x setting is:

`iter6_bcov1_zprior1_pvar0p001_ovarauto`

with:

`S_known = 0.6248 > S_joint = 0.5667 > S_zero = 0.5590`

This means 2x is not globally impossible for the candidate-free synthetic
self-consistency control.  However, the first-64 2x subset still fails under the
same setting:

`S_known = 0.6676 > S_zero = 0.6300 > S_joint = 0.6059`

So 2x recovery is subset-sensitive and should not yet be treated as a stable
paper-facing regime.

5. Biological iter-6 rerun

Output directory:

- `outputs/figure4_joint_decoder_known_residual_r2cal_affine_iter6_v1`

Using six hidden-joint iterations and `forward_model_z_prior_precision=1.0`
does not rescue the biological 4C gate.

Direct true-tau gate, uncalibrated:

`S_known = -2.1420 > S_joint = -2.4318 > S_zero = -2.5709`

The joint point estimate still beats zero, but less strongly than v4:

`joint - zero = 0.1390`

versus v4:

`joint - zero = 0.2115`

After cross-fold scalar gain calibration, zero still wins:

`S_zero = 0.1049 > S_joint = 0.0586`

By scale, 0.5x and 1x keep the desired direct-known ordering, but 2x remains
below zero because the direct recorded-tau known reference itself is below zero
at 2x.  This points back to the 2x feature target / local model / coordinate
contract rather than merely hidden-joint iteration count.
