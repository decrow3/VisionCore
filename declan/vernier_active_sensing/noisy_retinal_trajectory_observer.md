# Noisy Retinal-Trajectory Vernier Observer

## Motivation

The Vernier analysis needs three observer assumptions that are easy to teach and
defensible in manuscript language:

1. Retinal trajectory known: the observer is given the eye-generated retinal
   trajectory for the trial.
2. Retinal trajectory unknown: the observer must marginalize over the empirical
   trajectory catalog.
3. Retinal trajectory cue available with finite precision: the observer receives
   a noisy trajectory cue and marginalizes over trajectories consistent with that
   cue. Whether this interpolates to the trajectory-known observer depends on
   whether the true trajectory is included in, or well approximated by, the
   candidate trajectory support.

The third condition is the replacement for the earlier phrase "pose-robust
Fisher." The old finite-sigma Fisher diagnostic in `metrics.py` is useful as a
local sensitivity analysis, but it is not a likelihood observer. It averages
signals and pose-induced variances inside a diagonal Fisher denominator. The
method here instead performs the standard ideal-observer operation: marginalize
over a nuisance variable in the likelihood.

Terminology should use retinal position or retinal trajectory rather than pose.
The nuisance variable is the retinal displacement time series induced by the eye
movement. It changes the retinal movie and therefore the population response,
but it is not the Vernier sign the observer is asked to report.

## Observer Model

Let:

- `s in {+, -}` be the Vernier sign.
- `tau_j = (e_j(1), ..., e_j(T))` be candidate retinal trajectory `j`.
- `hat_tau_i` be the trajectory cue available on observed trial `i`.
- `r_i` be the population response on observed trial `i`.

The exact trajectory table stores predicted responses for paired Vernier signs
and paired trajectories:

```text
mu_plus[j, t, u]  = expected response of unit u at time t for +delta and tau_j
mu_minus[j, t, u] = expected response of unit u at time t for -delta and tau_j
```

For a Poisson count model, the response likelihood for one sign and one
trajectory is:

```text
log p(r_i | s, tau_j)
  = sum_t,u r_i[t,u] log mu_s[j,t,u] - mu_s[j,t,u] + constant(r_i).
```

The response-only constant cancels in the Vernier likelihood ratio, so the code
omits it.

With a noisy retinal-trajectory cue, the sign evidence is:

```text
log p(r_i | s, hat_tau_i)
  = log sum_j exp(log p(r_i | s, tau_j) + log p(tau_j | hat_tau_i)).
```

The current finite-catalog prior is a Gaussian over whole-path distance:

```text
log p(tau_j | hat_tau_i)
  = -0.5 / sigma_e^2 * mean_t ||e_j(t) - hat_e_i(t)||^2 - log Z_i.
```

Thus `sigma_e` is an RMS whole-trajectory cue width in arcmin. If we want a
literal independent Gaussian measurement at every time bin, the exponent should
use `sum_t` instead of `mean_t`, or equivalently the reported `sigma_e` should be
rescaled by `sqrt(T)`. The current choice is intentional for the pilot because it
keeps the uncertainty axis interpretable as an average retinal-position error
over the trajectory.

The normalization `log Z_i` is computed after any include-self or leave-one-out
masking. This matters: a finite trajectory prior should be a probability
distribution over the actually retained candidate catalog.

The decision variable is the Vernier likelihood ratio:

```text
Lambda_i = log p(r_i | +, hat_tau_i) - log p(r_i | -, hat_tau_i).
```

The decoder predicts `+` if `Lambda_i >= 0`, otherwise `-`.

The `best_trajectory_*` diagnostic is also evaluated under the same trajectory
cue. It is the MAP/profile counterpart to the marginal decision:

```text
max_j [log p(r_i | s, tau_j) + log p(tau_j | hat_tau_i)].
```

This is a diagnostic for posterior concentration, not the primary Vernier
decision rule. The primary rule remains the marginal likelihood above.

## Endpoint Checks

The include-self catalog has two useful limits:

- `sigma_e = 0`: all prior mass goes to the true/anchored trajectory when it is
  retained. This recovers the trajectory-known observer.
- `sigma_e = infinity`: all retained trajectories receive equal prior weight.
  This recovers the trajectory-unknown empirical-catalog marginal observer.

In a leave-one-trajectory-out catalog, the true trajectory is deliberately
excluded. The limits are therefore different:

- `sigma_e = 0`: all prior mass goes to the nearest retained held-out
  trajectory. This is not trajectory-known.
- finite `sigma_e`: a local catalog marginal around the trajectory cue.
- `sigma_e = infinity`: a uniform marginal over the retained held-out catalog.

Thus the held-out sigma sweep should not be described as an interpolation from
trajectory-known to trajectory-unknown. It interpolates from nearest held-out
catalog trajectory to uniform held-out catalog marginal. The trajectory-known
observer is a separate oracle reference, reported through the `known_*` columns.

## Practical Implementation In This Repository

The shared implementation hook is
`declan/vernier_active_sensing/trajectory_table_observer.py`.

New helper:

```text
trajectory_gaussian_log_weights(cue_pose_arcmin, candidate_poses_arcmin,
                                sigma_arcmin, mask, anchor_index)
```

This returns normalized log weights over the retained candidate trajectories and
the mean squared path distances in arcmin^2.

The existing scorer now accepts:

```text
joint_log_trajectory_weights
trajectory_prior_label
trajectory_weight_sigma_arcmin
trajectory_mean_dist2_arcmin2
```

If no weights are supplied, behavior is unchanged: the observer uses the old
uniform empirical catalog. If weights are supplied, the uniform marginal

```text
log mean_j p(r | s, tau_j)
```

becomes the weighted marginal

```text
log sum_j p(r | s, tau_j) p(tau_j | hat_tau_i).
```

The RR100 pilot runner is
`declan/vernier_active_sensing/run_rr100_noisy_trajectory_observer.py`. It is a
cache-only postprocess for
`outputs/notebook_vernier_walkthrough/rr100_real_trace_scale_grid/cache`.

Important pilot choices:

- It uses the saved real scaled trajectories from the RR100 grid caches.
- It does not generate synthetic trajectories.
- It pairs `+delta` and `-delta` by the same trajectory index.
- It treats the saved rates as deterministic expected counts after multiplying
  by the bin duration.
- It uses the same-condition trajectory catalog as the nuisance prior for the
  first pilot.
- It includes self by default, so `sigma_e = 0` is an endpoint sanity check and
  an optimistic upper bound. A leave-one-out run is the next stricter check.

The stricter RR100 runner is
`declan/vernier_active_sensing/run_rr100_heldout_trajectory_observer.py`. It
uses the same exact response tables but splits the real trajectory indices into
disjoint observation and nuisance-prior sets. This is the preferred stepping
stone when the goal is to avoid fitting a tiny trajectory catalog:

```text
observed response:       r_i from tau_i in held-out observation set
pose-aware endpoint:     log p(r_i | s, tau_i)
sigma=0 catalog limit:   log p(r_i | s, tau_NN(i))
finite-sigma marginal:   log sum_{j in C, j != i} p(r_i | s, tau_j) p(tau_j | hat_tau_i)
sigma=inf marginal:      log mean_{j in C, j != i} p(r_i | s, tau_j)
```

where every nuisance trajectory `tau_j` comes from the disjoint prior set `C`.
In this held-out version, `sigma_e = 0` means nearest retained empirical
trajectory, not the true trajectory. Therefore the exact trajectory-known
endpoint is reported separately in the `known_*` columns and plotted as the
pose-aware reference curve. `sigma_e = infinity` remains the pose-unaware
empirical Monte Carlo marginal over the retained prior set.

The intended pilot cache is an along-fixed real-trajectory sweep:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_rr100_real_trace_scale_grid \
  --out-dir outputs/notebook_vernier_walkthrough/rr100_real_trace_along1_mc \
  --across-scales 0,0.125,0.25,0.5,0.75,1,1.5,2,3 \
  --along-scales 1 \
  --n-traces 160 \
  --max-frames 60 \
  --fd-step-arcmin 0.25 \
  --device cuda:1 \
  --batch-size 64 \
  --force
```

Then the held-out observer is:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_rr100_heldout_trajectory_observer \
  --source-dir outputs/notebook_vernier_walkthrough/rr100_real_trace_along1_mc \
  --out-dir outputs/notebook_vernier_walkthrough/rr100_heldout_trajectory_observer_along1 \
  --trajectory-sigmas-arcmin 0,0.25,0.5,1,2,4,8,inf \
  --prior-k-list 32,64,128 \
  --n-observation-traces 32 \
  --n-prior-traces 128 \
  --split-seed 0
```

The convergence axis is `K`, the number of held-out nuisance trajectories used
in the prior. The main plot fixes along-contour motion at `1x`, sweeps
across-contour scale, overlays the separate pose-aware endpoint, and reports the
finite-sigma and pose-unaware marginal curves against the static condition.

Before interpreting held-out accuracy, run the catalog-density diagnostic:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_rr100_catalog_mismatch_diagnostic \
  --source-dir outputs/notebook_vernier_walkthrough/rr100_real_trace_along1_mc \
  --out-dir outputs/notebook_vernier_walkthrough/rr100_catalog_mismatch_diagnostic_along1 \
  --n-observation-traces 32 \
  --n-prior-traces 128 \
  --split-seed 0
```

It compares:

```text
D_traj(i) = ||mu(s_i, tau_i) - mu(s_i, tau_NN(i))||^2_Sigma^-1
D_sign(i) = ||mu(+, tau_i) - mu(-, tau_i)||^2_Sigma^-1
```

using a diagonal Poisson count metric. If `D_traj >> D_sign`, the held-out
catalog approximation is too sparse relative to the response precision: the
decoder is comparing two poor trajectory matches before it can compare Vernier
signs.

Outputs:

```text
outputs/notebook_vernier_walkthrough/rr100_noisy_trajectory_observer/
  rr100_noisy_trajectory_observer_summary.csv
  rr100_noisy_trajectory_observer_trials.csv
  rr100_noisy_trajectory_observer_manifest.json
  rr100_noisy_trajectory_observer_sigma_sweep.png
  rr100_noisy_trajectory_observer_closure_heatmaps.png
  rr100_noisy_trajectory_observer_static_relative_heatmaps.png
```

The most important diagnostic columns are:

- `joint_accuracy`: Vernier sign accuracy after noisy-trajectory marginalization.
- `known_accuracy`: endpoint accuracy with the trajectory supplied.
- `zero_accuracy`: static-center reference.
- `mean_trajectory_weight_neff`: effective number of trajectories in the prior.
- `mean_posterior_neff_true`: effective number of trajectories after combining
  the prior and response likelihood under the true sign.
- `mean_true_trajectory_rank_true`: rank of the true trajectory after weighting.
- `mean_margin_gap_closure_vs_zero_known`: useful but can be unstable when the
  known-minus-zero denominator is small.

## Pilot Interpretation

The first include-self RR100 pilot should be treated as a methods check, not a
final claim. It shows the desired endpoint behavior only because the true
trajectory is retained in the catalog:

- `sigma_e = 0` gives known-trajectory behavior.
- `sigma_e = infinity` gives a diffuse trajectory prior with
  `mean_trajectory_weight_neff` near the full 16-trace catalog.
- Intermediate values reveal how quickly the observer loses the known-trajectory
  advantage as trajectory uncertainty broadens within that retained catalog.

Because the first pilot is include-self and uses deterministic expected counts,
it is optimistic at small `sigma_e`. The held-out catalog run is more
conservative, but it is no longer a calibrated continuous trajectory-uncertainty
observer: it is a finite-catalog stress test. If nearby held-out trajectories
evoke responses that differ more than the Vernier sign does, failure is expected
and should be interpreted as catalog sparsity, not as evidence against a true
continuous trajectory marginal.

Also, margin closure is not the first teaching plot. It can exceed 1 or become
undefined when the zero-known denominator is small. For the tutorial, lead with
accuracy, trajectory/posterior `N_eff`, and the mean Vernier LLR margin relative
to the static condition; use closure only as a secondary continuity diagnostic.

## Literature Basis

This observer is not claiming a novel biological mechanism. It combines standard
pieces:

- Ideal-observer analysis treats perceptual tasks probabilistically and computes
  the optimal decision statistic under specified noise and uncertainty
  assumptions. See Knill and Richards, *Perception as Bayesian Inference* (1996)
  and Geisler, "Sequential ideal-observer analysis of visual discriminations"
  (Psychological Review, 1989).
- Nuisance marginalization is the standard Bayesian operation when an unknown
  variable affects the data but is not the task variable. Here the task variable
  is Vernier sign and the nuisance variable is retinal trajectory.
- The finite-catalog calculation is the discrete version of the same integral.
  It is also the static-table analogue of the forward likelihood in a hidden
  Markov model. Rabiner's HMM tutorial is the canonical algorithmic reference
  for summing over hidden state sequences with a forward recursion:
  https://doi.org/10.1109/5.18626.
- The FEM motivation comes from the literature showing that fixational retinal
  motion is not merely nuisance noise. Relevant references include Rucci,
  Iovin, Poletti, and Santini, "Miniature eye movements enhance fine spatial
  detail" (Nature, 2007; https://doi.org/10.1038/nature05866), Kuang, Poletti,
  Victor, and Rucci, "Temporal encoding of spatial information during active
  visual fixation" (Current Biology, 2012; https://doi.org/10.1016/j.cub.2012.01.050),
  Rucci and Victor, "The unsteady eye: an information-processing stage, not a
  bug" (Trends in Neurosciences, 2015; https://doi.org/10.1016/j.tins.2015.01.005),
  and Rucci and Poletti, "Control and functions of fixational eye movements"
  (Annual Review of Vision Science, 2015).
- A full continuous trajectory observer would move toward state-space or HMM
  inference over retinal position. The finite-catalog observer here is the
  simpler, auditable version that uses exact cached twin responses rather than a
  fitted continuous response surface.

## Recommended Tutorial Treatment

Teach the likelihood operation, but do not promote the leave-one-trajectory-out
catalog observer as the solution for spanning trajectory-known to
trajectory-unknown. It is best treated as a principled negative control showing
that sparse whole-trajectory lookup is inadequate for Vernier. Keep the
likelihood observer separate from Fisher:

```text
Known trajectory:
  log p(r | s, tau_i)

Include-self noisy trajectory cue:
  log sum_j p(r | s, tau_j) p(tau_j | hat_tau_i)
  with tau_i retained, so sigma=0 recovers known trajectory

Leave-one-trajectory-out catalog cue:
  log sum_{j in C, j != i} p(r | s, tau_j) p(tau_j | hat_tau_i)
  with sigma=0 equal to nearest retained catalog trajectory

Unknown trajectory:
  log mean_j p(r | s, tau_j)
```

The tutorial should explicitly say that no discriminative classifier is trained.
The decoder is generative: it uses the cached twin response table, a response
noise model, and a trajectory uncertainty prior. In held-out catalog runs,
`sigma_e` is a catalog weighting scale, not a calibrated bridge to the
pose-aware endpoint.

For the Vernier tutorial, the recommended stopping point is:

1. Report pose-aware and pose-unaware Fisher for the real-trace scale sweep.
2. Emphasize that the known-trace benefit comes mainly from reducing
   across-contour motion.
3. Include the leave-one-out catalog observer only as a negative control and
   catalog-density diagnostic.
4. Defer robust joint inference to the feature-decoder or natural-image branch.
