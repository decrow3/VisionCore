# BackImage Trajectory-Table Observer Results Log

Last updated: 2026-06-17.

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
- `matched_static_response` is not yet implemented as a cache-backed candidate
  mode, so the current hard-negative condition matches image structure but not
  stabilized twin responses.

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

Implementation note: this results log was written independently from the
provided specification. No GPL-covered source code was copied or adapted.
