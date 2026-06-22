# Compact-Aware Joint Prior Test Plan

Last curated: 2026-06-21.

## Purpose

The current BackImage joint observer is intentionally finite and explicit:

```text
candidate images x candidate trajectories -> exact V1-twin response table
joint score = logsumexp over the finite trajectory catalog
```

This is useful because it is auditable, but it also makes the "prior" mostly a
choice of trajectory catalog. Wu et al. had a stronger inference setup: a
continuous image prior and a continuous eye-motion model. The question here is
whether the compact translation geometry lets us add an analytic, compact-aware
latent-eye model while reusing the existing response-table infrastructure.

The goal is not to claim that compact geometry is already proven unique. The
goal is to test that claim directly:

```text
Does knowing the compact translation geometry improve latent-eye image/feature
recovery beyond uniform finite-catalog marginalization, and does that advantage
disappear when the compact component is removed?
```

There is one attribution issue that must be handled from the beginning. A
compact-aware trajectory prior can show that compact geometry helps the
pose-marginalization step, but it does not by itself show that compact geometry
improves the represented sensory signal. To separate these, every compact
response intervention must be crossed with both:

```text
known-eye: true trajectory supplied
joint-eye: trajectory latent and marginalized
```

If compact structure helps only for `joint-eye`, the result is a decoding or
pose-marginalization convenience. If it also helps for `known-eye`, the result
is a signal-level benefit in the represented response itself, closer to the
central attribution logic in Wu et al.

## Existing Infrastructure To Reuse

Core finite observer:

```text
declan/backimage_trajectory_observer/observer.py
declan/backimage_trajectory_observer/likelihood.py
```

Main response-table runner:

```text
declan/fixation_statistics_by_stimulus/run_backimage_trajectory_table_observer.py
```

Feature-posterior bridge:

```text
declan/backimage_trajectory_observer/analyze_feature_posterior.py
declan/backimage_trajectory_observer/analyze_feature_posterior_compact_mechanism.py
```

Compact subspace machinery:

```text
declan/backimage_trajectory_observer/analyze_compact_mechanism.py
declan/backimage_trajectory_observer/build_image_disjoint_compact_basis.py
declan/backimage_trajectory_observer/summarize_compact_mechanism_followups.py
```

Axis-conditioned trajectory catalogs:

```text
declan/axis_conditioned_backimage_trajectory_observer/axis_conditioned_traces.py
declan/axis_conditioned_backimage_trajectory_observer/summarize_axis_conditioned_run.py
```

Trace reconstruction helpers already exist in plotting/geometry utilities,
especially:

```text
declan/backimage_trajectory_observer/plot_global_fixation_fourier_component_flow.py
```

Canonical wrapper/config pattern:

```text
declan/canonical_active_sensing/run_joint_observer.py
declan/canonical_active_sensing/analyze_joint_posterior.py
declan/canonical_active_sensing/configs/joint_posterior_k16_v1.json
```

## Current Limitation

`score_image_identity_score_vectors(...)` already accepts
`log_trajectory_prior`, but only as one shared vector of shape `(K,)`.

For compact-aware catalog reweighting, one natural but claim-sensitive object is
candidate-conditioned:

```text
log w(tau_k | I_i)
```

because whether a trajectory is compact-consistent depends on the candidate
image's translation geometry. The first code change should therefore be a small,
well-tested extension allowing `log_trajectory_prior` to be either:

```text
None
(K,)
(n_candidates, K)
```

with normalization along the trajectory axis.

This extension must be interpreted carefully. In the current finite-table
implementation, an image-independent `(K,)` compact weight is not yet a
universal or biological eye-motion prior. It is a leave-one-table-out
catalog-statistic trajectory reweighting in the current catalog's trajectory
slot/order. A candidate-conditioned `(n_candidates, K)` weight is even more
local, because it uses the candidate image's forward-model geometry. It is
better described as a geometry-aware proposal or geometry-aware marginalization
weight. The first analysis must therefore compare:

```text
image_independent_compact_prior: one `(K,)` weight averaged across candidates
candidate_conditioned_compact_weight: one `(C,K)` weight per candidate image
```

A win by the candidate-conditioned version alone should be treated as a
potential leakage/alignment result, not as evidence for a biological
content-independent eye-motion prior.

## Stage 0: Baseline Reproduction

Before adding new priors, reproduce the existing exact-table and feature
posterior endpoints from cache.

Inputs:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
```

Hard-negative caches are the strongest current power target, but the primary
Wu-faithful zero-eye baseline should use `matched_static_response` candidate
sets whenever possible. Wu's zero observer is misspecified on a moving stimulus:
the stimulus moved, but the decoder assumed zero eye motion. That makes
joint-vs-zero a pose-compensation comparison rather than just "motion adds
information." `matched_static_response` is the local analog because it reduces
static-response shortcuts.

and the matched feature arrays used by:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1/
```

Acceptance checks:

```text
known-eye > joint-eye > zero-eye for image identity
joint feature recovery > zero feature recovery
known-eye feature recovery is reported for every compact response variant
compact_only, compact_removed, compact_addback reproduce the current reports
compact_addback reconstruction error remains near numerical zero
```

This stage should not rerun the V1 twin.

Add two attribution diagnostics at this stage:

```text
trajectory correctness: nearest/true trajectory rank or RMSE where traces exist
rate/gain control: score identity/feature recovery after global-rate + PC1 projection
```

`N_eff / K` and posterior entropy measure sharpness, not correctness. The
trajectory-correctness readout is needed to distinguish better pose inference
from better sensory encoding. The rate/gain projection keeps the readout
commensurable with the covariance-closure controls and guards against a pure
more-spikes/global-rate explanation.

## Stage 1: Cache-Only Compact-Aware Catalog Reweighting

This is the fastest test. Keep the existing finite trajectory catalog and exact
response tables, but replace uniform trajectory weights with compact-aware
weights.

For each response cache:

```text
lambda_full[I, tau, t, u]
lambda_zero[I, t, u]
Delta[I, tau, t, u] = lambda_full[I, tau, t, u] - lambda_zero[I, t, u]
U = image-disjoint compact translation basis
P_U Delta = projection of Delta into U
R_U Delta = Delta - P_U Delta
```

Define a candidate-conditioned compact leakage statistic:

```text
rho_noncompact[I, tau] =
  ||R_U Delta[I, tau]||^2 / (||Delta[I, tau]||^2 + eps)
```

From this, build two distinct objects:

```text
image-independent compact prior:
  log p_compact(tau)
    = log p_base(tau)
      - beta * zscore_over_tau(mean_over_training_images rho_noncompact[I, tau])

candidate-conditioned compact weight:
  log w_compact(tau | I)
    = log p_base(tau)
      - beta * zscore_within_candidate(rho_noncompact[I, tau])
```

`p_base` should be the existing finite-catalog prior. In current runs that is
usually uniform over the retained empirical, OU, edge-parallel, or
edge-orthogonal catalog. The image-independent prior is the primary Wu-style
comparison because it is shared across candidate images. The
candidate-conditioned weight is useful, but it should be labeled as a
geometry-aware proposal/marginalization weight rather than a content-independent
eye-motion prior. For per-candidate axis catalogs, normalize within each
candidate row and include a same-catalog image-independent summary where
possible.

Primary variants:

```text
uniform_base
image_independent_compact_prior
candidate_conditioned_compact_weight
random_subspace_aware
unit_shuffle_compact_aware
gain_axis_aware
static_pc_aware
inverse_compact_control
```

Entropy matching:

Tune `beta` so that the compact-aware and control priors have matched average
trajectory entropy or matched average `N_eff / K` within each:

```text
candidate_set_mode x prior_family x motion_scale x likelihood_scale
```

This prevents a sharper prior from winning merely because it is sharper.

Implementation route:

1. Add a small helper in `observer.py` or a new utility module:

```text
normalize_log_trajectory_prior(log_prior, n_candidates, n_trajectories)
```

2. Add tests covering `None`, `(K,)`, and `(n_candidates, K)` prior weights.

3. Add a cache-only analyzer:

```text
declan/backimage_trajectory_observer/analyze_compact_aware_prior.py
```

4. Reuse `_load_basis`, `_project_delta`, `_static_pc_basis`,
   `_random_basis`, and `_unit_shuffle_basis` from `analyze_compact_mechanism.py`
   where possible. If import coupling gets ugly, factor these into a small
   `compact_projection_utils.py` module.

5. Reuse feature loading/alignment from
   `analyze_feature_posterior_compact_mechanism.py`.

Outputs:

```text
compact_aware_prior_trials.csv
compact_aware_prior_summary.csv
compact_aware_prior_contrasts.csv
compact_aware_prior_qc.csv
compact_aware_prior_report.md
compact_aware_prior_metadata.json
```

QC must include the shared-prior source. For `image_independent_*` families,
the analyzer should report `raw_weight_source =
selected_manifest_stable_trajectory_leave_one_table_out`, high
`shared_prior_matched_slots`, and low `shared_prior_fallback_fraction` or
`shared_prior_nonmatching_fallback_slots`. High fallback means the selected
tables do not share enough stable trajectory identities, so the clean prior
partly collapses toward the outside-table pooled fallback rather than using
current-table leakage.

Primary readouts:

```text
image-identity accuracy
true-candidate score margin
posterior-weighted feature MSE / neg-MSE
candidate posterior entropy
trajectory posterior N_eff / K
joint-minus-zero feature recovery
known-minus-joint pose cost
```

Success pattern:

```text
image_independent_compact_prior improves or stabilizes feature recovery over uniform_base
image_independent_compact_prior beats entropy-matched random/gain/unit-shuffle controls
candidate_conditioned_compact_weight adds little beyond the image-independent prior
static_pc_aware is either weaker or explicitly explained by compact overlap
```

Failure patterns:

```text
all entropy-matched priors perform similarly
static_pc_aware matches image_independent_compact_prior
candidate_conditioned_compact_weight is the only winning compact variant
compact-aware weighting improves image identity but not feature recovery
compact-aware weighting only wins by lowering posterior entropy without improving truth
```

Magnitude diagnostic:

```text
compact-aware-over-uniform gap small:
  consistent with the current finite catalog already sampling the relevant
  motion support well.

compact-aware-over-uniform gap large:
  either foveal/V1 compact geometry makes pose inference unusually useful, or
  the finite trajectory catalog is impoverished and any informative proposal
  helps. Check trajectory correctness, prior entropy, and catalog support before
  interpreting it as compact-specific prior knowledge.
```

## Stage 2: Symmetric Subspace-Removal Ablation

The current feature-space compact-removal result asks whether compact removal
hurts. The next test must ask whether compact removal hurts more than removing
other similarly useful subspaces.

Use the same response-table construction as `analyze_compact_mechanism.py`, but
make removal symmetric:

```text
full_exact
zero_static
compact_only
compact_removed
log_compact_removed
static_pc_only
static_pc_removed
gain_only
gain_removed
random_only
random_removed
unit_shuffle_only
unit_shuffle_removed
compact_residualized_against_static_pc_only
compact_residualized_against_static_pc_removed
static_pc_residualized_against_compact_only
static_pc_residualized_against_compact_removed
```

Run each response variant under:

```text
uniform_base prior
image_independent_compact_prior
candidate_conditioned_compact_weight
static_pc_aware prior
random_subspace_aware prior
gain_axis_aware prior
```

This separates five claims:

1. `compact_only` sufficiency.
2. `compact_removed` necessity.
3. specificity versus matched high-value subspaces.
4. image-independent compact-prior usefulness versus candidate-conditioned
   compact-weighting usefulness.
5. signal-level benefit versus pose-marginalization benefit.

Key comparisons:

```text
known-eye compact_only vs known-eye full_exact
known-eye compact_removed vs known-eye full_exact
joint-eye compact_only vs joint-eye full_exact
joint-eye compact_removed vs joint-eye full_exact
compact_removed vs static_pc_removed
compact_removed vs gain_removed
compact_removed vs random_removed
compact_residualized_removed vs static_pc_residualized_removed
image_independent_compact_prior_on_full vs uniform_base_on_full
image_independent_compact_prior_on_compact_removed vs uniform_base_on_compact_removed
candidate_conditioned_compact_weight vs image_independent_compact_prior
static_pc_aware_on_full vs image_independent_compact_prior_on_full
```

Required QC:

```text
projection basis provenance
image-disjoint basis verification
subspace overlap matrix, especially compact vs static_pc
rate clipping / negative-rate fractions
log-rate removal companion for any linear removal claim
addback reconstruction error for every subspace family
entropy / N_eff matching for every prior family
```

Claim gate:

Do not claim compact-specific prior knowledge unless compact removal produces a
larger loss than matched static-PC/gain/random removals and the compact-aware
prior beats entropy-matched control priors. Do not claim a signal-level compact
coding benefit unless the known-eye compact cross supports it; a joint-eye-only
effect is still useful, but it is specifically a pose-marginalization result.

## Stage 3: Analytic Compact Linear-Gaussian Observer

The catalog reweighting test is still finite-catalog marginalization. The more
interesting trick is to use compact geometry to integrate over continuous eye
paths.

For a candidate image `I_i`, define compact coordinates:

```text
z_i(t) = U^T [y_obs(t) - lambda_zero(I_i, t)]
```

Fit or derive an image-specific compact routing matrix:

```text
z_i(t) ~= A_i tau(t) + noise
```

where `tau(t)` is a continuous 2D retinal displacement.

Use an analytic eye-motion prior:

```text
tau(t) = alpha tau(t - 1) + eta(t)
eta(t) ~ N(0, Q)
tau(0) ~ N(0, S0)
```

Then compute:

```text
log p(z_i(1:T) | I_i)
```

with a Kalman filter. This gives a continuous latent-eye joint score without
summing over the finite trajectory catalog.

Validity domain:

The Kalman observer is justified only inside the regime where the compact
linearization is accurate:

```text
lambda(I, tau) - lambda(I, zero) ~= U A_i tau
```

Therefore every Kalman-vs-table comparison must report the linearization
residual as a function of motion scale. The primary comparison should be
restricted to scales where that residual is small, with larger scales treated as
stress tests rather than clean evidence for the analytic observer.

### Stage 3A: Cache-Fit `A_i`

Use existing response caches as design points:

```text
Z[I_i, tau_k, t] = U^T [lambda_full(I_i, tau_k, t) - lambda_zero(I_i, t)]
```

Recover the corresponding trajectory coordinates from:

```text
prior_trajectory_ids in response caches
trace_bank.csv
axis_trajectory_catalog.csv
run_metadata.json
selected_windows.csv
```

Existing reconstruction helpers in
`plot_global_fixation_fourier_component_flow.py` can be adapted, but per-candidate
axis catalogs may need a cleaner trace-loading utility.

Fit `A_i` by ridge regression:

```text
min_A sum_{k,t} ||Z[I_i, tau_k, t] - A_i tau_k(t)||^2 + ridge ||A_i||^2
```

Estimate compact observation noise from cross-validated residuals:

```text
R_i = diag_or_lowrank_cov(residuals)
```

`R_i` is load-bearing. The primary version should use full or low-rank
covariance in compact coordinates, with diagonal `R_i` as a control. Diagonal
noise discards correlated response structure, which both Wu's
coupled-vs-uncoupled result and the current covariance-closure work suggest can
matter.

Acceptance checks:

```text
held-out trajectory compact-coordinate R2 > random/unit-shuffle controls
A_i fit is not dominated by gain axis
linearization residual is small in the claimed motion-scale regime
Kalman log-likelihood ranks true image above zero-eye/static baseline
```

### Stage 3B: Chart-Fit `A_i`

Once Stage 3A works, stop fitting `A_i` from the same trajectory catalog and use
finite-difference/local-chart geometry instead.

Possible sources:

```text
declan/backimage_trajectory_observer/build_image_disjoint_compact_basis.py
declan/backimage_trajectory_observer/run_cardinal_tangent_chart.py
outputs/active_sensing_movie_information/compact_basis_exports/
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/
```

This is the cleaner compact-geometry test:

```text
U and A_i come from image-disjoint translation geometry
trajectory marginalization is analytic
feature recovery is evaluated on held-out response tables
```

### Stage 3C: Control Analytic Observers

Build identical analytic observers with:

```text
U_full_linear
U_random
U_unit_shuffle
U_gain
U_static_pc
U_compact_residualized_against_static_pc
U_static_pc_residualized_against_compact
```

All should use the same Kalman transition parameters, same likelihood
temperature, same feature endpoint, and same candidate sets.

`U_full_linear` is the full-space or high-rank linear-Gaussian control. If it
reproduces the finite-table rescue as well as the compact observer, then the
result is about first-order linearity rather than compactness. The compact claim
requires compact Kalman performance to be close to full linear performance and
better than matched low-dimensional controls.

## Stage 4: Optional Image Prior Extension

The analytic compact observer still uses finite candidate images. A full Wu-like
continuous image prior is a larger project and should not be mixed into the
first compact-prior test.

If needed later, the safer intermediate step is not full pixel reconstruction.
Use a low-dimensional image-feature prior over the existing candidate features:

```text
p(feature) from Gabor/pyramid/ImageNet patch feature density
```

Then test whether feature-prior weighting changes the compact-aware observer's
conclusions. This should be a separate branch after the compact eye-prior test.

## Recommended First Implementation

Start with the cache-only prior-weighting analysis:

```text
declan/backimage_trajectory_observer/analyze_compact_aware_prior.py
```

Command shape:

```bash
.venv/bin/python -m declan.backimage_trajectory_observer.analyze_compact_aware_prior \
  --run-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1 \
  --out-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_compact_aware_prior_hardneg_n128_k10_v1 \
  --feature-npz outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/feature_latent_arrays.npz \
  --compact-basis-path outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz \
  --basis-mode image_disjoint \
  --k-dims 10 \
  --candidate-set-modes hard_negative_structure \
  --priors axis_edge_parallel,axis_edge_orthogonal \
  --motion-scales 0.5,1.0,2.0 \
  --likelihood-scales 1.0 \
  --latent-names pyramid_local_field,gabor_local_field \
  --pca-k-list 8,16 \
  --prior-families uniform_base,image_independent_compact_prior,candidate_conditioned_compact_weight,random_subspace_aware,unit_shuffle_compact_aware,gain_axis_aware,static_pc_aware,inverse_compact_control \
  --entropy-match-target image_independent_compact_prior \
  --n-random 8 \
  --progress-every 16
```

Smoke command should use:

```text
--max-tables 8
--n-random 2
```

## Tests To Add

Observer prior weights:

```text
tests/test_backimage_trajectory_observer.py
  - shared `(K,)` prior reproduces old behavior
  - candidate-conditioned `(C,K)` prior changes only the intended candidate rows
  - invalid prior shapes fail loudly
  - normalized priors have logsumexp zero along trajectory axis
```

Compact-aware prior analyzer:

```text
tests/test_compact_aware_prior.py
  - compact leakage statistic is zero for perfectly compact deltas
  - image-independent compact prior has shape `(K,)`
  - candidate-conditioned compact weight has shape `(C,K)`
  - candidate-conditioned compact weight is labeled as proposal/marginalization weight, not pure generative prior
  - random/gain/static-PC controls use matched dimensions
  - entropy matching returns comparable N_eff/K
  - compact prior/weight construction does not use `y_obs_counts`
  - known-eye and joint-eye are both emitted for every response variant
  - compact addback reconstructs full response
```

Analytic Kalman observer:

```text
tests/test_compact_kalman_observer.py
  - known linear-Gaussian synthetic model recovers correct image
  - larger Q approaches broad-motion marginalization
  - Q -> 0 approaches a fixed-pose observer
  - random subspace control fails on synthetic compact data
```

## Decision Rules

Strong positive:

```text
leave-one-table-out compact catalog-statistic prior improves feature recovery over uniform
leave-one-table-out compact catalog-statistic prior beats entropy-matched random/gain/unit-shuffle/static-PC priors
known-eye compact cross shows signal-level compact benefit, or the claim is
  explicitly limited to pose marginalization
advantage disappears under compact removal
analytic compact Kalman observer reproduces or improves the finite-table rescue
```

Medium positive:

```text
leave-one-table-out compact catalog-statistic prior is not better than uniform, but analytic
compact Kalman observer matches finite-table joint recovery with fewer
trajectory samples
```

Bounded/null:

```text
static-PC-aware prior matches the compact catalog-statistic prior
compact removal is no worse than static-PC/gain removal
compact catalog-statistic prior only sharpens the posterior without improving feature truth
candidate-conditioned compact weight is the only winning compact-weighting variant
full-space linear-Gaussian observer matches compact Kalman without compact specificity
```

In the bounded/null case, the correct interpretation is:

```text
compact geometry is an important response subspace, but the current observer
does not isolate compact-specific prior knowledge as the source of the rescue.
The finite-table result should be described as response-space trajectory
reweighting until a compact-specific analytic eye-motion prior passes matched
controls.
```

## Claim Boundary

Even a positive result would not show that the animal implements this observer.
It would show that compact translation geometry is sufficient to support a
more analytic latent-eye inference model than the finite trajectory catalog,
and that this model explains feature recovery better than matched control
subspaces.

Do not claim:

```text
the posterior identifies the true animal eye trace exactly
empirical FEMs are uniquely optimal
compact geometry is the only useful low-dimensional response structure
full Wu-style natural-image reconstruction has been implemented
```

Allowed if the strong positive gate passes:

```text
Knowing the compact translation geometry gives an idealized observer useful
prior/latent-state structure for recovering image features when eye position is
unknown.
```
