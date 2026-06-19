# BackImage Trajectory-Table Observer Results Log

Last updated: 2026-06-19.

## Scope Boundary

These BackImage trajectory-table results use exact cached full-population
V1-twin response tables:

```text
lambda_counts[I_i, tau_k, time, unit]
```

They do not use compact translation geometry, instantaneous local charts, or a
lag-linear approximation. The named `empirical` and `OU` priors are trajectory
priors over the nuisance catalog `p(tau_k)`, not learned natural-image priors.
The image patches are finite candidate hypotheses with an implicit uniform
candidate-image prior.

Current claim:

```text
Trajectory marginalization over exact natural-image response tables can rescue
image identity from pose uncertainty.
```

Not yet claimed:

```text
compact geometry is the mechanism
real FEM statistics are uniquely optimal
full Wu-style natural-image reconstruction works here
```

## Directional Pilot: n16/k4/0.5x/LOO

Output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_cuda0_pilot_n16_k4_loo_v1/
```

Configuration:

```text
n_images = 16
n_candidates = 4
candidate_set_mode = hard_negative_structure
observation_family = empirical
prior_families = empirical, ou
scale = 0.5
n_prior_trajectories = 4
trajectory_prior_mode = leave_one_out
likelihood_scales = 0.5, 1.0
```

Runtime:

```text
~15m 50s on CUDA
```

Summary:

```text
prior     likelihood   known   zero    joint   best_single_tau   median_Neff/K   median_nearest_rank
empirical 0.5          1.000   0.750   1.000   1.000         0.745           1
empirical 1.0          1.000   0.750   1.000   1.000         0.724           1
ou        0.5          1.000   0.750   0.875   0.938         0.725           2
ou        1.0          1.000   0.750   0.938   0.938         0.562           2
```

Interpretation:

This is the first result in the BackImage trajectory-observer branch that should
be treated as directionally positive rather than merely interpretable. In this
small hard-negative natural-image pilot, trajectory-marginalized image
identification improves over the zero-eye observer in leave-one-out mode.

The qualitative pattern is:

```text
known-eye >= joint-eye > zero-eye
```

The empirical prior is stronger than the OU prior in this pilot. With empirical
prior support, joint-eye reaches `16/16` even though the exact observed
trajectory is held out. With OU support, joint-eye remains above zero-eye but
drops to `14/16` or `15/16`, depending on likelihood scale.

This differs from the Vernier trajectory-table result. Vernier showed that
known-eye could expose task information while nuisance marginalization washed
out Vernier evidence. Here, natural-image structure appears to provide enough
constraints for pose-free marginalization to preserve image identity.

Important caveats:

- `n=16` is small; `1.00` versus `0.75` is four extra correct trials.
- `zero-eye = 0.75`, so image identity is already fairly accessible from
  static responses.
- The posterior does not collapse to one trajectory. Median `N_eff / K` is
  often around `0.55-0.75`, so this is partial pose localization, not perfect
  trajectory inference.
- At the time of this pilot, `matched_static_response` was not yet implemented
  as a cache-backed candidate mode, so this hard-negative condition matched
  image structure but not stabilized twin responses. Later matched-static runs
  below supersede this caveat.

Claim boundary:

```text
In a small hard-negative natural-image pilot, trajectory-marginalized image
identification improves over a zero-eye observer, especially with an empirical
FEM prior. This suggests that natural-image structure can support pose-free
recovery in a way the Vernier stimulus did not.
```

## Confirmation Run: Option C n64/k8/scale sweep

Output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_cuda0_optionC_n64_k8_scales_v1/
```

Configuration:

```text
n_images = 64
n_candidates = 4
candidate_set_mode = hard_negative_structure
observation_family = empirical
prior_families = empirical, ou
scales = 0.25, 0.5, 1.0
n_prior_trajectories = 8
trajectory_prior_mode = leave_one_out
likelihood_scales = 0.5, 1.0
```

Runtime:

```text
~3h 54m on CUDA.
384 response tables.
768 observer rows.
```

Summary:

```text
scale  prior      likelihood   known   zero    joint   best_single_tau   joint-zero   median_Neff/K
0.25   empirical  0.5          1.000   0.797   0.844   0.891         +0.047       0.842
0.25   empirical  1.0          1.000   0.797   0.875   0.891         +0.078       0.782
0.25   ou         0.5          1.000   0.797   0.859   0.906         +0.062       0.828
0.25   ou         1.0          1.000   0.797   0.844   0.906         +0.047       0.769

0.50   empirical  0.5          1.000   0.719   0.797   0.828         +0.078       0.710
0.50   empirical  1.0          1.000   0.719   0.797   0.828         +0.078       0.622
0.50   ou         0.5          1.000   0.719   0.844   0.922         +0.125       0.727
0.50   ou         1.0          1.000   0.719   0.844   0.922         +0.125       0.613

1.00   empirical  0.5          1.000   0.484   0.781   0.812         +0.297       0.639
1.00   empirical  1.0          1.000   0.484   0.797   0.812         +0.312       0.455
1.00   ou         0.5          1.000   0.484   0.797   0.766         +0.312       0.611
1.00   ou         1.0          1.000   0.484   0.812   0.766         +0.328       0.464
```

Interpretation:

This run confirms the main positive pattern from the n16 pilot at a more useful
scale. Known-eye remains perfect, showing that the response tables contain
image-identity information when the trajectory is specified. Zero-eye falls
sharply relative to the known-eye ceiling and the joint observer as motion scale
increases. At `1.0x`, the key contrast is `known = 1.000`,
`zero = 0.484`, and `joint = 0.781-0.812`. The trajectory-marginalized
observer stays substantially above zero-eye in every condition.

The most important scale trend is:

```text
0.25x: joint-zero = +0.047 to +0.078
0.50x: joint-zero = +0.078 to +0.125
1.00x: joint-zero = +0.297 to +0.328
```

That is the expected qualitative behavior if unknown FEMs make a static/zero
observer increasingly wrong, while natural-image structure still supports
nuisance-marginalized image identification.

The empirical-vs-OU contrast is weaker than in the n16 pilot. OU is often equal
or slightly better in accuracy, so this run should not be used to claim that the
empirical motion prior is specifically superior. The stronger claim is that
trajectory marginalization itself is helping under hard-negative natural-image
candidate sets.

Posterior concentration is partial, not complete. Median `N_eff / K` decreases
with motion scale and likelihood scale, reaching roughly `0.45-0.46` in the
strongest `1.0x` / likelihood-scale `1.0` cases. That is much more localized
than the nearly uniform Vernier trajectory posterior, but it is not exact
trajectory recovery. Median nearest-trajectory rank is `2.0` at `0.25x` and
`0.5x`, and `2.5` at `1.0x`.

One diagnostic caveat: in the OU `1.0x` rows, joint accuracy slightly exceeds
the legacy `best_trajectory_oracle` field. Inspection confirms that this field
is not a true oracle upper bound; it is a best-single-trajectory/MAP-nuisance
diagnostic that chooses the single trajectory giving the largest likelihood for
each candidate. Because this can overfit distractor images, marginalization can
beat it in accuracy. Future outputs add clearer `best_single_tau_*` aliases;
treat the legacy `best_trajectory_oracle_*` names as deprecated wording.

Updated claim boundary:

```text
In a 64-image hard-negative BackImage pilot with leave-one-trajectory-out
trajectory catalogs, trajectory-marginalized image identification consistently
outperforms the zero-eye observer. The improvement is largest when motion scale
is large enough that the zero-eye observer falls sharply relative to the
known-eye ceiling, while known-eye remains perfect. This supports the idea that
natural-image structure can provide pose constraints for a Wu-style
nuisance-marginalized observer, unlike the impoverished Vernier stimulus. It
does not yet support a claim that empirical FEMs are uniquely optimal relative
to OU motion.
```

## Option C Post-Hoc Image/Condition Analysis

Output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_cuda0_optionC_n64_k8_scales_v1/
    posthoc_image_condition_analysis/
```

Files:

```text
condition_summary.csv
feature_bin_summary.csv
feature_correlation_summary.csv
error_case_summary.csv
trial_feature_join.csv
accuracy_vs_scale.png
posterior_diagnostics_vs_scale.png
joint_zero_score_gain_vs_image_features_scale1_like1.png
neff_fraction_vs_joint_zero_score_gain_scale1_like1.png
image_condition_analysis_report.md
```

Main read:

This is a pose-uncertainty rescue result, not an active-sensing optimality
result. Joint accuracy is not higher at `1.0x` than at `0.25x`. Instead, the
`1.0x` gain is driven primarily by zero-eye falling sharply while joint-eye
stays comparatively robust. Joint accuracy also declines modestly relative to
`0.25x`, so the effect is not pure zero collapse, but the widened zero-to-joint
gap is the dominant pattern.

```text
0.25x: zero 0.797, joint 0.844-0.875, joint-zero +0.047 to +0.078
0.50x: zero 0.719, joint 0.797-0.844, joint-zero +0.078 to +0.125
1.00x: zero 0.484, joint 0.781-0.812, joint-zero +0.297 to +0.328
```

The raw image-feature dependence is exploratory and not yet a clean monotonic
story. In the primary `1.0x`, likelihood-scale `1.0` slice, the strongest
feature correlation with joint-minus-zero true-score gain is nearest-distractor
contrast distance:

```text
empirical prior: rho ~= 0.271
OU prior:        rho ~= 0.243
```

Most raw structure metrics have weaker correlations with score gain:

```text
contrast:                rho ~= 0.070 to 0.099
edge density:            rho ~= -0.050 to -0.038
orientation coherence:   rho ~= -0.072 to -0.048
high-frequency power:    rho ~= 0.057 to 0.058
8+ cpd power:            rho ~= 0.064 to 0.077
```

The cleaner mechanistic signal is posterior concentration versus evidence
gain. This should be foregrounded over the raw feature correlations. At `1.0x`,
likelihood-scale `1.0`, lower `N_eff / K` is moderately associated with larger
joint-minus-zero true-score gain:

```text
empirical prior: rho(N_eff/K, score gain) ~= -0.464
OU prior:        rho(N_eff/K, score gain) ~= -0.390
```

The same relationship is weaker for discrete correct/incorrect rescue, so this
should be treated as evidence-shape support rather than a fully explained
accuracy mechanism.

Unavailable diagnostic:

```text
static_response_distance_to_nearest_distractor = all nan
mean_rate_distance_to_nearest_distractor = all nan
```

That is expected because this run used `hard_negative_structure`, not
`matched_static_response`. The matched-static-response control remains the key
addition before making the result hard to dismiss as residual static image
identity.

Current best wording:

```text
In natural-image patches, trajectory marginalization recovers much of the
image-identity information lost by a zero-eye observer, especially when motion
is large enough to strongly invalidate the zero-eye assumption. The recovery is
linked to partial concentration of the trajectory posterior, suggesting that
natural-image responses provide useful constraints on latent eye trajectory.
Raw image-structure metrics do not yet explain which patches benefit, and
empirical trajectory priors do not clearly outperform OU, so the current claim
is trajectory-marginalized robustness under natural-image structure, not
real-FEM optimality.
```

Next targeted confirmatory run:

```text
n_images = 64 or 128
n_candidates = 8
K = 8 or 16
candidate modes = hard_negative_structure, matched_static_response
priors = empirical, OU, optional shuffled-position
scales = 0.5x, 1.0x
likelihood_scale = 0.5, 1.0
primary mode = leave-one-out
```

Key success pattern:

```text
known-eye high
zero-eye impaired at larger motion
joint-eye > zero-eye
joint advantage associated with lower N_eff/K
effect survives matched_static_response
```

## Confirmatory Matched-Static Run: n64/c8/k8

Output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
```

Configuration:

```text
n_images = 64
n_candidates = 8
candidate_set_modes = hard_negative_structure, matched_static_response
observation_family = empirical
prior_families = empirical, ou
scales = 0.5, 1.0
n_prior_trajectories = 8
trajectory_prior_mode = leave_one_out
likelihood_scales = 0.5, 1.0
```

Status:

```text
Completed.
Runtime ~= 10h 12m including stabilized static-response prepass.
512 response tables.
1024 observer rows.
```

Summary:

```text
hard_negative_structure, 0.5x:
  zero 0.578, joint 0.781-0.844, joint-zero +0.203 to +0.266

hard_negative_structure, 1.0x:
  zero 0.312, joint 0.734-0.875, joint-zero +0.422 to +0.562

matched_static_response, 0.5x:
  zero 0.578, joint 0.750-0.828, joint-zero +0.172 to +0.250

matched_static_response, 1.0x:
  zero 0.328, joint 0.672-0.797, joint-zero +0.344 to +0.469
```

Interpretation:

The joint-eye advantage survives `matched_static_response`. This is the
strongest version of the result so far: even when distractors are matched by
stabilized/static twin responses, trajectory marginalization recovers a large
fraction of the known-eye gap lost by the zero-eye observer.

At `matched_static_response`, `1.0x`, likelihood scale `1.0`:

```text
empirical trajectory prior:
  zero = 0.328
  joint = 0.766
  recovery of known-zero gap ~= 65%
  median N_eff / K ~= 0.364

OU trajectory prior:
  zero = 0.328
  joint = 0.797
  recovery of known-zero gap ~= 70%
  median N_eff / K ~= 0.400
```

This still does not imply compact geometry is the mechanism. The result was
obtained with exact cached full-response tables, not a compact translation
approximation. The mechanistic next step is to test whether a compact or lagged
geometry surrogate can reproduce this same pose-uncertainty rescue.

## Compact-Mechanism Smoke: global basis, two cached tables

Output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    compact_mechanism_smoke/
```

Configuration:

```text
base run = backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1
candidate_set_mode = matched_static_response
motion_scale = 1.0
prior = empirical
likelihood_scale = 1.0
max_tables = 2
k_dims = 2,10
basis_path = outputs/active_sensing_movie_information/
  compact_basis_exports/figure4_tfts_compact_basis_delta025.npz
basis_key = basis
basis_mode = global
variants = full_exact, zero_static, compact_only, compact_removed,
  random_k, unit_shuffle_compact, gain_only, static_pc_k
```

Status:

```text
Completed.
Cache-only; no V1 twin rerun and no GPU required.
```

Outputs:

```text
compact_mechanism_trials.csv
compact_mechanism_summary.csv
compact_mechanism_by_variant.csv
compact_mechanism_random_null_summary.csv
compact_mechanism_posterior_summary.csv
compact_mechanism_rate_clipping_audit.csv
compact_mechanism_reconstruction_checks.csv
compact_mechanism_report.md
compact_mechanism_run_metadata.json
```

Sanity checks:

```text
max prior delta reconstruction error ~= 8.7e-19
max known delta reconstruction error ~= 8.7e-19
basis shape = 756 x 126
orthonormal error after audit ~= 2.2e-14
nearest_tau_distance finite rows = 30 / 30
```

Tiny-smoke readout:

```text
full_exact:
  zero = 0.5
  joint = 0.5
  median joint-zero true-score gain ~= 10.27
  negative-rate fraction = 0

compact_only k=2:
  joint = 0.5
  median joint-zero true-score gain ~= 8.74
  median compact sufficiency by true-score ~= 1.01
  median negative-rate fraction ~= 0.0012

compact_only k=10:
  joint = 0.5
  median joint-zero true-score gain ~= 9.30
  median compact sufficiency by true-score ~= 1.04
  median negative-rate fraction ~= 0.0007

compact_removed k=2:
  joint = 0.0
  median joint-zero true-score gain ~= -3.03
  median compact necessity by true-score ~= 0.37
  median negative-rate fraction ~= 0.043

compact_removed k=10:
  joint = 0.0
  median joint-zero true-score gain ~= -4.82
  median compact necessity by true-score ~= 0.24
  median negative-rate fraction ~= 0.041
```

Interpretation:

This smoke validates the mechanics of the post-hoc projection analysis, not the
scientific claim. The projection decomposition reconstructs the exact
motion-induced delta to numerical precision, the exact and zero variants are
scored through the same deterministic Poisson expected-count convention, and
projected negative-rate artifacts are recorded explicitly.

The two-table readout is suggestive but intentionally underpowered: the compact
projection preserves most of the full exact true-score gain on these two tables,
while the compact-removed and random controls perform worse and introduce much
larger negative-rate clipping. Because the smoke uses only two tables and a
global basis, it should be treated as a debugging pass. A claim-level mechanism
test needs the full matched-static run, more random nulls, and ideally an
image-disjoint compact basis.

Follow-up fix after review:

```text
- random_k with --n-random 0 no longer consumes a missing random basis.
- nearest_tau_distance is recovered from observer_trials.csv for old caches.
- future response caches record nearest_trajectory_distance in the .npz table
  and response_cache_manifest.csv.
- basis_mode=image_disjoint now requires basis-file provenance or an explicit
  --allow-unverified-image-disjoint-basis override.
- compact scoring now validates positive eps, positive likelihood_scale,
  nonnegative observations, candidate-id length, and table shapes.
```

## Compact-Mechanism Full Global Run: n512/k2-20/random8

Output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    compact_mechanism_full_global_n512_k2_5_10_20_rand8_v1/
```

Configuration:

```text
base run = backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1
selected response tables = 512
candidate_set_modes = hard_negative_structure, matched_static_response
motion_scales = 0.5, 1.0
priors = empirical, OU
likelihood_scales = 0.5, 1.0
k_dims = 2,5,10,20
n_random = 8
basis_path = outputs/active_sensing_movie_information/
  compact_basis_exports/figure4_tfts_compact_basis_delta025.npz
basis_key = basis
basis_mode = global
```

Status:

```text
Completed.
Cache-only; no V1 twin rerun and no GPU required.
```

Primary matched-static result:

At `matched_static_response`, `1.0x`, likelihood scale `1.0`:

```text
empirical full_exact:
  known = 1.000
  zero = 0.328
  joint = 0.766
  median N_eff/K = 0.364

empirical compact_only:
  k=2  joint = 0.563, true-score rescue ~= 0.794
  k=5  joint = 0.609, true-score rescue ~= 0.825
  k=10 joint = 0.609, true-score rescue ~= 0.880
  k=20 joint = 0.641, true-score rescue ~= 0.861

empirical compact_removed:
  k=2-20 joint = 0.313-0.359
  true-score rescue = negative
  median clipped/negative-rate fraction ~= 0.038-0.051
```

```text
OU full_exact:
  known = 1.000
  zero = 0.328
  joint = 0.797
  median N_eff/K = 0.400

OU compact_only:
  k=2  joint = 0.516, true-score rescue ~= 0.817
  k=5  joint = 0.516, true-score rescue ~= 0.827
  k=10 joint = 0.578, true-score rescue ~= 0.876
  k=20 joint = 0.578, true-score rescue ~= 0.855

OU compact_removed:
  k=2-20 joint = 0.297-0.391
  true-score rescue = negative
  median clipped/negative-rate fraction ~= 0.040-0.056
```

Controls at `matched_static_response`, `1.0x`, likelihood scale `1.0`:

```text
empirical:
  compact_only k=10:
    joint = 0.609
    true-score rescue ~= 0.880
    clipped fraction ~= 0.0012

  random_k k=10:
    joint = 0.340
    true-score rescue ~= -0.229
    clipped fraction ~= 0.0425

  unit_shuffle_compact k=10:
    joint = 0.422
    true-score rescue ~= 0.289
    clipped fraction ~= 0.0022

  gain_only:
    joint = 0.469
    true-score rescue ~= 0.527
    clipped fraction = 0

  static_pc_k k=10:
    joint = 0.547
    true-score rescue ~= 0.772
    clipped fraction ~= 0.0004
```

```text
OU:
  compact_only k=10:
    joint = 0.578
    true-score rescue ~= 0.876
    clipped fraction ~= 0.0015

  random_k k=10:
    joint = 0.330
    true-score rescue ~= -0.289
    clipped fraction ~= 0.0493

  unit_shuffle_compact k=10:
    joint = 0.422
    true-score rescue ~= 0.273
    clipped fraction ~= 0.0034

  gain_only:
    joint = 0.531
    true-score rescue ~= 0.556
    clipped fraction = 0

  static_pc_k k=10:
    joint = 0.578
    true-score rescue ~= 0.771
    clipped fraction ~= 0.0007
```

Interpretation:

This is the first compact-mechanism result that plausibly bridges the two
stories:

```text
compact FEM geometry
  -> carries motion-dependent likelihood structure
  -> supports trajectory-marginalized natural-image inference
```

The sufficiency side is the strongest part. On the harder
`matched_static_response` setting, compact-only projections recover a large
fraction of the exact-table true-score gain with very low clipping
(`~0.001-0.002`). Accuracy is coarser at `n=64`, but it still rises well above
zero-eye and well above random subspaces.

Specificity is mixed but encouraging. Compact-only clearly beats random
subspaces and unit-shuffled compact bases. It also beats gain-only. Static-PC
subspaces are a tougher control: they recover substantial true-score gain and
sometimes approach compact-only accuracy. Therefore, the current result supports
compact-subspace sufficiency relative to random/unit-shuffle/gain controls, but
static-response PCs remain an important alternative low-dimensional-response
control.

The necessity side is suggestive but not yet clean. Removing the compact
component collapses joint accuracy near zero/static performance and produces
negative true-score rescue. However, compact-removed also creates many more
negative projected rates than compact-only, with median clipped/negative-rate
fractions around `0.04-0.06` in the key matched-static rows. This means the
compact-removed collapse should be treated as partly confounded by projected
rate invalidity until a nonnegative reconstruction or safer likelihood
parameterization is tested.

Current claim:

```text
Using a global compact translation basis, the compact projection of
motion-induced BackImage response deltas preserves much of the exact
trajectory-marginalized image-identity rescue in the matched-static setting.
The effect is not explained by random subspaces, unit identity shuffling, or a
single gain axis. Static-response PC controls remain close enough to require
additional follow-up. Necessity evidence from compact removal is encouraging
but clipping-confounded. Image-disjoint basis validation is still the promotion
gate.
```

Next checks before promotion:

```text
1. Image-disjoint basis:
   repeat the full analysis with a cross-fit/image-disjoint compact basis.
   Qualitative success pattern:
     compact_only > random/unit-shuffle/gain/static-PC controls
     compact_removed loses rescue
     compact-only clipping remains low

2. Static-PC control:
   inspect whether static PCs overlap with the compact translation basis and
   whether they are carrying translation-like axes or generic image identity.

3. Clipping-safe necessity:
   rerun compact_removed with a safer nonnegative response reconstruction or
   alternate scoring convention so loss of rescue is not driven by invalid
   projected rates.

4. True-score rescue curves:
   make k-sweep plots for true-score rescue and accuracy because true-score
   rescue is smoother than n=64 image-identity accuracy.

5. Stay primary on matched_static_response:
   this is the harder and more convincing candidate set.
```

## Compact-Mechanism Follow-Up Implementation

Implemented the next-step diagnostic machinery:

```text
declan/backimage_trajectory_observer/build_image_disjoint_compact_basis.py
declan/backimage_trajectory_observer/summarize_compact_mechanism_followups.py
```

New response variants in `analyze_compact_mechanism.py`:

```text
log_compact_only:
  lambda_zero * exp(P_U log(lambda_full / lambda_zero))

log_compact_removed:
  lambda_zero * exp((I - P_U) log(lambda_full / lambda_zero))
```

These are positivity-preserving companion diagnostics. They do not replace the
linear additive `compact_only` / `compact_removed` decomposition, but they test
whether compact-removal conclusions survive without creating negative projected
rates.

Global-run follow-up outputs:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    compact_mechanism_full_global_n512_k2_5_10_20_rand8_v1/
      followup_summary/
```

Files:

```text
compact_mechanism_promotion_gates.csv
compact_staticpc_basis_overlap.csv
compact_mechanism_followup_metadata.json
figures/
```

Image-disjoint basis export:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_image_disjoint_compact_basis_delta025_v1/
    image_disjoint_compact_basis_delta0p25_fold0of2.npz
```

Export summary:

```text
source = outputs/twin_feature_tangent_structure_prod_limited_synth/
split_by = image_id
n_folds = 2
heldout_fold = 0
centering = centered_across_tangents_per_unit
basis shape = 756 x 50
n_train_objects = 25
n_train_split_groups = 7
top-k tangent variance fraction:
  k=2  0.410
  k=5  0.622
  k=10 0.786
  k=20 0.918
```

Image-disjoint smoke:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    compact_mechanism_image_disjoint_smoke/
```

The smoke verified:

```text
basis_mode = image_disjoint
image_disjoint_basis_verified = true
log_compact_removed clipped fraction = 0 in smoke rows
```

Full image-disjoint run launched:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    compact_mechanism_image_disjoint_fold0_n512_k2_5_10_20_rand8_log_v1/
```

Configuration:

```text
selected response tables = 512
basis_mode = image_disjoint
k_dims = 2,5,10,20
n_random = 8
likelihood_scales = 0.5,1.0
variants = full_exact, zero_static, compact_only, compact_removed,
  log_compact_only, log_compact_removed, random_k, unit_shuffle_compact,
  gain_only, static_pc_k
PID = 1750828
```

## Compact-Mechanism Image-Disjoint Result: fold0/n512/log variants

Output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    compact_mechanism_image_disjoint_fold0_n512_k2_5_10_20_rand8_log_v1/
```

Status:

```text
Completed.
Cache-only; no V1 twin rerun and no GPU required.
```

Follow-up summary and plots:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    compact_mechanism_image_disjoint_fold0_n512_k2_5_10_20_rand8_log_v1/
      followup_summary/
```

Primary matched-static result:

At `matched_static_response`, `1.0x`, likelihood scale `1.0`:

```text
empirical full_exact:
  known = 1.000
  zero = 0.328
  joint = 0.766

empirical compact_only, image-disjoint:
  k=2  joint = 0.563, true-score rescue = 0.784
  k=5  joint = 0.578, true-score rescue = 0.848
  k=10 joint = 0.547, true-score rescue = 0.804
  k=20 joint = 0.609, true-score rescue = 0.836
```

```text
OU full_exact:
  known = 1.000
  zero = 0.328
  joint = 0.797

OU compact_only, image-disjoint:
  k=2  joint = 0.531, true-score rescue = 0.811
  k=5  joint = 0.531, true-score rescue = 0.854
  k=10 joint = 0.531, true-score rescue = 0.790
  k=20 joint = 0.563, true-score rescue = 0.840
```

Controls:

At `matched_static_response`, `1.0x`, likelihood scale `1.0`, `k=20`:

```text
empirical:
  compact_only true-score rescue = 0.836
  random_k true-score rescue = -0.338
  unit_shuffle true-score rescue = 0.247
  gain_only true-score rescue = 0.527
  static_pc true-score rescue = 0.811
  compact_removed true-score rescue = -0.345, clipped fraction = 0.032
  log_compact_removed true-score rescue = -0.048, clipped fraction = 0.000

OU:
  compact_only true-score rescue = 0.840
  random_k true-score rescue = -0.406
  unit_shuffle true-score rescue = 0.230
  gain_only true-score rescue = 0.556
  static_pc true-score rescue = 0.815
  compact_removed true-score rescue = -0.467, clipped fraction = 0.034
  log_compact_removed true-score rescue = -0.084, clipped fraction = 0.000
```

Static-PC overlap:

```text
compact_vs_static_pc mean cos^2:
  k=2  0.344
  k=5  0.310
  k=10 0.292
  k=20 0.276

compact_vs_random mean cos^2:
  k=2  0.005
  k=5  0.006
  k=10 0.015
  k=20 0.026
```

Interpretation:

The image-disjoint basis passes the main promotion gate qualitatively.
Compact-only preserves a large fraction of exact-table true-score rescue in the
hard matched-static setting, and it clearly beats random subspaces,
unit-shuffled compact bases, and gain-only controls. This materially strengthens
the bridge:

```text
compact FEM geometry
  -> carries motion-dependent likelihood structure
  -> supports trajectory-marginalized natural-image inference
```

The biggest remaining caveat is static-PC specificity. Static PCs are not random
and are not equivalent to the compact basis, but they recover a similar amount
of true-score rescue at `k=10/20`. The overlap audit shows moderate but far from
complete overlap between compact and static-PC subspaces. Therefore the current
strong claim should be that an image-disjoint compact translation basis is
sufficient to preserve the rescue above random/unit-shuffle/gain controls; it is
not yet uniquely better than all generic low-dimensional static-response
subspaces.

The clipping-safe necessity check is encouraging. Linear `compact_removed`
still loses rescue but creates `~0.02-0.03` clipped rates in the key rows.
`log_compact_removed` has zero clipping and still removes most true-score
rescue, with true-score rescue near zero or negative at higher k. This reduces
the concern that compact-removal failure was purely a negative-rate artifact.

Current best wording:

```text
In the BackImage exact-cache observer, an image-disjoint compact translation
basis preserves much of the trajectory-marginalized natural-image rescue in the
matched-static condition. The effect survives random, unit-shuffle, and gain
controls, and a clipping-safe log-rate removal diagnostic still removes most
of the rescue. Static-response PCs remain a close low-dimensional control, so
the current result supports a real bridge from compact translation geometry to
Wu-style trajectory marginalization, but not yet a claim that compact geometry
is uniquely superior to every generic low-dimensional static-response subspace.
```

## Axis-Conditioned Shared-Source Matched-Static Pilot

Output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1/
```

Configuration:

```text
n_images = 64
n_candidates = 4
candidate_set_mode = matched_static_response
observation_family = empirical
prior_families = axis_edge_parallel, axis_edge_orthogonal
axis_catalog_mode = per_candidate
scale = 0.5
n_prior_trajectories = 16
trajectory_prior_mode = leave_one_out
likelihood_scale = 1.0
```

This is the first completed axis-conditioned run after the shared-source fix.
The audit verifies that the comparison is a clean same-source axis comparison:

```text
axis_shared_source_catalog_fraction = 1.0
source Jaccard = 1.0
paired prior rows = 4096
parallel/orthogonal motion-stat deltas = 0 across audited metrics
```

Primary readout:

```text
known-eye = 64/64 = 1.000
zero-eye  = 41/64 = 0.641

axis_edge_parallel:
  joint = 55/64 = 0.859
  joint-zero = +0.219
  median N_eff/K = 0.471
  median nearest trajectory rank = 3
  median joint true margin = 0.265

axis_edge_orthogonal:
  joint = 53/64 = 0.828
  joint-zero = +0.188
  median N_eff/K = 0.370
  median nearest trajectory rank = 3
  median joint true margin = 0.198
```

Paired-trial posthoc:

```text
parallel-only correct = 6
orthogonal-only correct = 4
both correct = 49
both wrong = 5

median parallel-minus-orthogonal margin delta = +0.021
median parallel-minus-orthogonal true-score delta = +0.225
median parallel-minus-orthogonal N_eff/K delta = +0.013
```

The posthoc output is:

```text
axis_conditioned_posthoc/
  axis_posthoc_report.md
  axis_observer_summary.csv
  axis_pair_summary.csv
  axis_paired_trial_feature_table.csv
  axis_case_summary.csv
  axis_feature_correlation_summary.csv
  axis_feature_bin_summary.csv
```

The strongest exploratory feature relationships are modest. Drift-edge
parallelism correlates positively with parallel-minus-orthogonal true-score
delta (`rho = 0.220`) and margin delta (`rho = 0.200`), while the equivalent
drift-gradient parallelism correlations have opposite sign. Parallel-only
correct trials have a larger median margin delta (`+0.269`) than
orthogonal-only correct trials (`-0.067`). This is directionally aligned with
the along-contour story, but it is still an n64 exploratory pilot.

Important caveat:

```text
The clean shared-source matched-static run weakly favors edge-parallel, but only
by 2 trials. Treat this as a positive pilot and a rationale for replication, not
as claim-level evidence.
```

## Feature-Posterior Axis Matched-Static Run: Gabor/Pyramid k4/k8

Output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_matched_static_feature_posterior_gabor_pyramid_k4_8_v1/
```

Base exact-cache run:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1/
```

Configuration:

```text
candidate_set_mode = matched_static_response
observation_family = empirical
prior_families = axis_edge_parallel, axis_edge_orthogonal
axis_catalog_mode = per_candidate
axis_shared_source_catalog = True
scale = 0.5
n_images = 64
n_candidates = 4
n_prior_trajectories = 16
likelihood_scale = 1.0
latents = gabor_local_field, pyramid_local_field
feature PCA k = 4, 8
feature source = computed BackImage patch latents
```

Runtime:

```text
feature extraction = 64/64 windows in ~8m26s
cache scoring = 128/128 response tables in ~5s
```

Outputs:

```text
feature_posterior_trials.csv
feature_posterior_summary.csv
feature_axis_contrasts.csv
feature_motion_evidence_contrasts.csv
feature_posterior_qc.csv
feature_posterior_report.md
feature_posterior_metadata.json
feature_latent_arrays.npz
```

QC:

```text
response cache rows scored = 128
manifest rows without response cache = 0
gabor raw feature dim = 4608
pyramid raw feature dim = 3072
nonfinite feature count = 0

PCA variance fraction:
  gabor k4 = 0.178
  gabor k8 = 0.306
  pyramid k4 = 0.208
  pyramid k8 = 0.329
```

Summary table, mean feature MSE reduction relative to zero-eye:

```text
orthogonal gabor k4:
  zero MSE = 61.10
  joint MSE = 55.43
  reduction = 9.3%

orthogonal gabor k8:
  zero MSE = 77.38
  joint MSE = 66.57
  reduction = 14.0%

orthogonal pyramid k4:
  zero MSE = 40.37
  joint MSE = 36.28
  reduction = 10.1%

orthogonal pyramid k8:
  zero MSE = 42.21
  joint MSE = 38.53
  reduction = 8.7%

parallel gabor k4:
  zero MSE = 61.10
  joint MSE = 54.24
  reduction = 11.2%

parallel gabor k8:
  zero MSE = 77.38
  joint MSE = 63.14
  reduction = 18.4%

parallel pyramid k4:
  zero MSE = 40.37
  joint MSE = 34.40
  reduction = 14.8%

parallel pyramid k8:
  zero MSE = 42.21
  joint MSE = 36.16
  reduction = 14.3%
```

Axis feature-recovery contrasts:

```text
parallel - orthogonal, joint feature recovery:

gabor k4:
  mean = +1.19
  median = +0.37
  parallel wins per trial = 57.8%

gabor k8:
  mean = +3.43
  median = +0.91
  parallel wins per trial = 56.2%

pyramid k4:
  mean = +1.88
  median = +0.61
  parallel wins per trial = 53.1%

pyramid k8:
  mean = +2.37
  median = +0.11
  parallel wins per trial = 51.6%
```

Posterior diagnostics:

```text
zero-eye true candidate mass ~= 0.371
orthogonal joint true candidate mass ~= 0.383
parallel joint true candidate mass ~= 0.393
known-eye true candidate mass ~= 0.461

zero-eye median candidate N_eff fraction ~= 0.801
orthogonal joint median candidate N_eff fraction ~= 0.929
parallel joint median candidate N_eff fraction ~= 0.922
```

Known-eye remains substantially better than joint-eye. Known beats joint on
roughly `86-92%` of per-trial feature rows depending on latent, k, and prior.
This is a useful ceiling check: trajectory marginalization recovers feature
information above zero-eye, but does not close the gap to a known trajectory.

`best_single_tau` is close to joint. The gap is usually small relative to the
zero-eye MSE scale (`~0.3-2%`), so the feature-posterior result is not dominated
by a catastrophic dilution penalty from marginalizing over trajectories.

The `motion_delta` rows should be read carefully:

```text
motion_delta = joint candidate score - zero candidate score
```

This is a candidate-wise log-likelihood-ratio diagnostic, not a calibrated
generative posterior. It is useful for isolating the motion-added score
component, but as a standalone softmax posterior it produces much worse feature
MSE than zero/joint. Do not interpret that as motion evidence being useless.

Interpretation:

This is the first successful feature-posterior unification of the exact
trajectory-table observer and the Gabor/pyramid feature-decomposition branches.
The main result is:

```text
joint-eye feature recovery > zero-eye feature recovery
parallel-axis joint feature recovery > orthogonal-axis joint feature recovery
```

The parallel-over-orthogonal effect is directionally consistent across all four
feature/k settings, but heterogeneous across trials. It is strongest in the
mean for `gabor_local_field, k=8`, while per-trial win rates are modest
(`~52-58%`). This should be read as a positive feature-preservation signal, not
as a saturated per-trial classifier.

Current best wording:

```text
The exact joint trajectory-table observer carries recoverable image-feature
information beyond zero-eye. In the clean matched-static axis cache,
edge-parallel priors recover more Gabor/pyramid feature structure than
edge-orthogonal priors, even though the effect is heterogeneous and known-eye
still has a large ceiling advantage. This supports the along-contour
feature-preservation interpretation more directly than finite-candidate image
identity accuracy alone.
```

Immediate follow-ups:

```text
1. Add bootstrap or permutation confidence intervals for joint-minus-zero
   feature gain and parallel-minus-orthogonal feature gain.
2. Run the same feature-posterior endpoint on the non-axis empirical-vs-OU
   matched-static exact cache.
3. Reuse the saved feature_latent_arrays.npz when running additional
   likelihood-scale or subset analyses to avoid recomputing Gabor/pyramid
   features.
```

## Axis-Conditioned Shared-Source Hard-Negative Replacement

Output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n64_c4_k16_v1/
```

Configuration:

```text
n_images = 64
n_candidates = 4
candidate_set_mode = hard_negative_structure
observation_family = empirical
prior_families = axis_edge_parallel, axis_edge_orthogonal
axis_catalog_mode = per_candidate
scale = 0.5
n_prior_trajectories = 16
trajectory_prior_mode = leave_one_out
likelihood_scale = 1.0
```

Runtime:

```text
2h19m on CUDA GPU1.
128 response tables.
128 observer rows.
```

The audit verifies that this is a clean same-source axis comparison:

```text
axis_shared_source_catalog_fraction = 1.0
source Jaccard = 1.0
paired prior rows = 4096
parallel/orthogonal motion-stat deltas = 0 across audited metrics
```

Primary readout:

```text
known-eye = 64/64 = 1.000
zero-eye = 41/64 = 0.641

axis_edge_parallel:
  joint = 54/64 = 0.844
  joint-zero = +0.203
  median N_eff/K = 0.506
  median nearest trajectory rank = 3
  median joint true margin = 0.360

axis_edge_orthogonal:
  joint = 57/64 = 0.891
  joint-zero = +0.250
  median N_eff/K = 0.431
  median nearest trajectory rank = 3
  median joint true margin = 0.355
```

Paired-trial posthoc:

```text
parallel-only correct = 3
orthogonal-only correct = 6
both correct = 51
both wrong = 4

median parallel-minus-orthogonal margin delta = -0.014
median parallel-minus-orthogonal true-score delta = +0.259
median parallel-minus-orthogonal N_eff/K delta = +0.019
```

The posthoc output is:

```text
axis_conditioned_posthoc/
  axis_posthoc_report.md
  axis_observer_summary.csv
  axis_pair_summary.csv
  axis_paired_trial_feature_table.csv
  axis_case_summary.csv
  axis_feature_correlation_summary.csv
  axis_feature_bin_summary.csv
```

Interpretation:

This clean hard-negative replacement resolves the old catalog-mismatch caveat
for the hard-negative setting. The general trajectory-marginalization result is
robust: both edge-parallel and edge-orthogonal axis-conditioned priors rescue
image identity above zero-eye. The axis-specific result is mixed. Accuracy
favors edge-orthogonal by three trials, while the paired true-score delta and
some feature correlations retain an edge-parallel signal. In particular,
drift-edge parallelism correlates positively with parallel-minus-orthogonal
true-score delta (`rho = 0.341`) and margin delta (`rho = 0.301`), but this is
exploratory at `n=64`.

Current axis claim boundary:

```text
Clean shared-source axis-conditioned observers rescue image identity over
zero-eye, but the direction of the edge-parallel versus edge-orthogonal effect
is not yet stable across candidate modes. Matched-static weakly favors
edge-parallel; hard-negative favors edge-orthogonal in accuracy while retaining
some edge-parallel score diagnostics. Treat the axis-specific biology as
unresolved pending larger shared-source replications.
```

## Axis-Conditioned Shared-Source Hard-Negative n128 Scale Sweep

Source exact-cache output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
```

Feature-posterior posthoc output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k4_8_uncertainty_v1/
```

Configuration:

```text
candidate_set_mode = hard_negative_structure
observation_family = empirical
prior_families = axis_edge_parallel, axis_edge_orthogonal
axis_catalog_mode = per_candidate
n_images = 128
n_candidates = 4
n_prior_trajectories = 16
scales = 0.5, 1.0, 2.0
trajectory_prior_mode = leave_one_out
likelihood_scale = 1.0 primary
axis_shared_source_catalog = True
```

Runtime and cache:

```text
observer run = 14h07m26s on CUDA GPU1
selected windows = 128
response caches = 768/768
observer rows = 1536
feature-posterior response tables scored = 768
feature-posterior bootstrap/permutation resamples = 10000 / 10000
```

Primary image-identity readout at likelihood scale `1.0`:

```text
scale 0.5x:
  known-eye = 1.000
  zero-eye = 0.609
  axis_edge_parallel joint = 0.8125
  axis_edge_orthogonal joint = 0.7813
  paired discordance: parallel-only 13, orthogonal-only 9, p = 0.523

scale 1.0x:
  known-eye = 1.000
  zero-eye = 0.391
  axis_edge_parallel joint = 0.7969
  axis_edge_orthogonal joint = 0.8047
  paired discordance: parallel-only 12, orthogonal-only 13, p = 1.000

scale 2.0x:
  known-eye = 1.000
  zero-eye = 0.336
  axis_edge_parallel joint = 0.6797
  axis_edge_orthogonal joint = 0.7422
  paired discordance: parallel-only 13, orthogonal-only 21, p = 0.229
```

Feature-posterior posthoc:

```text
latents = gabor_local_field, pyramid_local_field
feature PCA k = 4, 8
feature source = computed BackImage patch latents
axis contrast gating = axis_shared_source_catalog must be true
motion_delta = joint candidate score - zero candidate score
```

K-choice note:

The `k=4,8` feature PCA setting was chosen as a conservative bridge to the
existing low-dimensional Gabor/pyramid decomposition branch. It keeps the
posterior-MSE readout interpretable with only four image candidates per trial
and with PCA fit over selected-window feature rows. It should not be read as a
final rejection of richer feature spaces. The next principled sensitivity run
is:

```text
k = 2, 4, 8, 16, 32
```

The key question is whether the 1x parallel feature-recovery advantage is
strongest at `k8`, survives at `k16/k32`, or washes out when broader
fine-grained feature reconstruction is included.

Current small-signal readout:

```text
k8 looks like the more scale-sensitive current setting. The axis-pooled
absolute joint MSE is still mostly flat, so this is not an overall 1x optimum.
The promising signal is narrower: the parallel-vs-orthogonal feature-recovery
contrast is most positive near 1x and is easier to see in the richer k8 readout
than in the coarser k4 readout, especially for Gabor features.
```

Joint-minus-zero feature recovery is the durable result. It is positive for
every scale, axis, latent, and k row, with sign-flip permutation
`p ~= 0.0001` throughout.

Mean joint MSE reduction relative to zero-eye:

```text
0.5x: 46-55% across axis/latent/k rows
1.0x: 62-70% across axis/latent/k rows
2.0x: 71-78% across axis/latent/k rows
```

Parallel-minus-orthogonal joint feature recovery:

```text
0.5x:
  gabor k4:    +3.36, CI [-4.16, 13.01], p = 0.536
  gabor k8:    +0.71, CI [-4.03,  6.68], p = 0.831
  pyramid k4:  +1.68, CI [-2.29,  6.34], p = 0.471
  pyramid k8:  +1.74, CI [-1.30,  5.41], p = 0.341

1.0x:
  gabor k4:    +7.20, CI [-2.85, 20.71], p = 0.283
  gabor k8:    +3.22, CI [-3.62, 11.78], p = 0.470
  pyramid k4:  +2.91, CI [-2.60,  9.53], p = 0.366
  pyramid k8:  +2.34, CI [-1.95,  7.43], p = 0.355

2.0x:
  gabor k4:    -0.02, CI [-13.40, 12.43], p = 0.998
  gabor k8:    -4.60, CI [-14.41,  4.66], p = 0.356
  pyramid k4:  -1.56, CI [-10.41,  5.86], p = 0.745
  pyramid k8:  -3.39, CI [-10.70,  2.93], p = 0.353
```

Interpretation:

The n128 scale sweep upgrades the feature-posterior bridge from a narrow pilot
to a substantial clean-cache result. The robust claim is that joint trajectory
marginalization rescues image identity and Gabor/pyramid feature recovery
relative to zero-eye under hard negatives. The axis-specific claim remains
bounded. Parallel trends positive for feature recovery at `0.5x` and `1.0x`,
but the paired uncertainty intervals are broad; at `2.0x`, the direction
reverses toward orthogonal. Image-identity accuracy also does not give a stable
parallel advantage under hard negatives. The current best reading is a
scale- and endpoint-dependent reconciliation: edge-orthogonal motion can be a
better discriminating probe for hard-negative identity, while edge-parallel
motion may be feature-preserving at natural scales, but the unified model has
not yet made that mechanism claim decisive.

Compact tangent geometry status:

```text
image-identity compact mechanism:
  possible now as a cache-only posthoc on this response cache using
  analyze_compact_mechanism.py

feature-posterior compact mechanism:
  not yet wired; requires emitting compact-variant candidate score vectors
  and feeding them into analyze_feature_posterior.py

V1 forward rerun:
  not required for the initial compact mechanism tests
```

## Compact-Mechanism Image-Disjoint n128 Hard-Negative Scale Sweep

Output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
    compact_mechanism_image_disjoint_fold0_n768_k2_5_10_20_rand8_log_v1/
```

Configuration:

```text
source exact cache = backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1
selected response tables = 768/768
basis_mode = image_disjoint
basis shape = 756 x 50
basis source = backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz
variants = full_exact, zero_static, compact_only, compact_removed,
           log_compact_only, log_compact_removed, random_k,
           unit_shuffle_compact, gain_only, static_pc_k
k = 2, 5, 10, 20
n_random = 8
likelihood_scales = 0.5, 1.0
```

QC:

```text
image_disjoint_basis_verified = true
max prior reconstruction error ~= 1.7e-18
max known reconstruction error ~= 1.7e-18
trial rows = 90,624
summary rows = 372
```

Primary likelihood-scale `1.0` result:

```text
compact_only:
  preserves substantial full-minus-zero accuracy rescue at every scale
  k5 is usually the best compact-only setting by accuracy
  median true-score rescue is typically ~0.84-0.90 of full exact

compact_removed:
  falls to zero-static or below in most rows
  median true-score rescue is negative

log_compact_removed:
  has zero clipping
  remains near zero-static
  supports the compact-removal result without the linear negative-rate caveat

random_k:
  weak or below zero-static

unit_shuffle_compact:
  partial but much weaker than compact_only

gain_only:
  partial but not sufficient

static_pc_k:
  competitive with compact_only and sometimes better in accuracy
```

Selected accuracy rows at likelihood scale `1.0`:

```text
0.5x orthogonal:
  zero = 0.609
  full_exact = 0.781
  compact_only k20 = 0.734
  compact_removed k20 = 0.586
  static_pc k20 = 0.750

0.5x parallel:
  zero = 0.609
  full_exact = 0.812
  compact_only k20 = 0.703
  compact_removed k20 = 0.578
  static_pc k20 = 0.766

1.0x orthogonal:
  zero = 0.391
  full_exact = 0.805
  compact_only k20 = 0.648
  compact_removed k20 = 0.375
  static_pc k20 = 0.703

1.0x parallel:
  zero = 0.391
  full_exact = 0.797
  compact_only k20 = 0.641
  compact_removed k20 = 0.445
  static_pc k20 = 0.641

2.0x orthogonal:
  zero = 0.336
  full_exact = 0.742
  compact_only k20 = 0.586
  compact_removed k20 = 0.305
  static_pc k20 = 0.578

2.0x parallel:
  zero = 0.336
  full_exact = 0.680
  compact_only k20 = 0.555
  compact_removed k20 = 0.305
  static_pc k20 = 0.547
```

Best compact-only `k` by accuracy:

```text
0.5x orthogonal: k5,  joint = 0.750
0.5x parallel:   k2,  joint = 0.773
1.0x orthogonal: k5,  joint = 0.680
1.0x parallel:   k5,  joint = 0.656
2.0x orthogonal: k5,  joint = 0.617
2.0x parallel:   k10, joint = 0.570
```

Interpretation:

This is a positive compact-mechanism result for image-identity joint decoding.
An image-disjoint compact tangent basis is sufficient for much of the exact
joint-eye rescue, and removing that compact component destroys most of the
rescue. The clipping-safe log-removal variants make the necessity result more
credible than the earlier linear-only removal result.

The result does not yet prove compact geometry is unique. Static-response PCs
are a serious control and remain competitive. This is a fair dimensionality
control because it asks whether a generic low-dimensional static-response
subspace can carry the same likelihood structure. However, static PCs are not a
compact-excluding negative control. If compact translation directions are
embedded in high-variance static-response axes, static PCs can inherit part of
the compact geometry. Therefore strong static-PC performance weakens the claim
that the compact basis is uniquely special, but it does not falsify the compact
geometry mechanism.

Current best wording:

```text
In the hard-negative n128 axis-conditioned cache, an image-disjoint compact
translation basis carries a large fraction of the exact joint-eye likelihood
rescue, and compact removal collapses the rescue. This links compact
translation geometry to trajectory-marginalized image inference. Static PCs
remain competitive, so the current claim is compact sufficiency/necessity
relative to random, unit-shuffle, gain, and removal controls, not uniqueness
over all low-dimensional response subspaces.
```

Next compact controls:

```text
1. Measure compact-vs-static-PC subspace overlap for this n128 hard-negative
   cache.
2. Residualize compact against static PCs and static PCs against compact to ask
   which component uniquely carries rescue.
3. Add compact-only / compact-removed score variants to the feature-posterior
   bridge, especially for the k8, 1x, parallel-vs-orthogonal feature signal.
```

### Pre-Fix Axis Runs

The following axis-conditioned runs were generated before the shared-source
catalog fix and should be treated as diagnostics only:

```text
backimage_axis_conditioned_trajectory_observer_percandidate_gpu1_pilot32_c4_k8
backimage_axis_conditioned_trajectory_observer_percandidate_gpu1_pilot64_c4_k16
backimage_axis_conditioned_trajectory_observer_percandidate_gpu1_target128_c4_k32
```

The completed target128 hard-negative run has:

```text
zero-eye = 0.617
axis_edge_parallel joint = 0.766
axis_edge_orthogonal joint = 0.875
median source Jaccard = 0.143
```

Because parallel and orthogonal used different retained source catalogs, this
orthogonal advantage is not a clean biological result. The clean shared-source
hard-negative replacement above supersedes these pre-fix runs for interpretation
of hard-negative axis effects.

Implementation note: this results log was written independently from the
provided specification. No GPL-covered source code was copied or adapted.
