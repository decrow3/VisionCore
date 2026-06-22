# Panel C Feature-Space Compact-Removed Handoff

Date: 2026-06-21
Status: implementation handoff for a coding agent

## Bottom Line

Panel C now needs a compact-removed control in the same feature-recovery
metric used by the promoted C panel. The existing compact-removal result is
useful implementation evidence, but it is an image-identification accuracy
audit, not the feature-posterior cosine endpoint shown in Figure 4C.

The coding task is to run the compact-subspace intervention through the
feature-posterior scorer and produce feature-recovery cosine curves for:

```text
zero eye
full joint
compact only
compact removed
compact addback / reconstruction sanity check
known eye
```

The promoted Panel C should only claim compact-subspace necessity if
`compact_removed` collapses toward the zero-eye feature-recovery curve while
`compact_only` retains much of the full joint feature recovery.

## Current Figure Context

Current selected Figure 4 composite:

```text
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v5.png
```

Current Panel C source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/
    feature_posterior_summary.csv
```

Current Panel C builder:

```text
declan/figure4_active_sensing_atlas/scripts/build_selected_figure4_v4_design.py
declan/figure4_active_sensing_atlas/scripts/build_joint_feature_posterior_panel.py
declan/figure4_active_sensing_atlas/scripts/build_panel_c_feature_recovery_options.py
```

Current C5 values for the selected feature-posterior endpoint:

```text
latent = pyramid_local_field
requested_k = 8
candidate_set_mode = hard_negative_structure
observation_scale = 0.5, 1.0, 2.0
likelihood_scale = 1.0
feature metric = feature_cosine / mean feature recovery
```

The currently promoted C visual shows:

```text
zeroed eye
compact subspace
known eye ceiling
```

It does not yet show the feature-space compact-removed control.

## Important Terminology

Use `compact subspace`, `compact only`, and `compact removed`.

Avoid calling this `tangent`, `normal`, or `geometry` in the implementation
outputs or panel labels. Those words collide with actual image geometry and
parallel/orthogonal axis labels elsewhere in Figure 4.

## Existing Result To Reuse As Implementation Reference Only

The current compact-mechanism audit is here:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
    compact_mechanism_image_disjoint_fold0_n768_k2_5_10_20_rand8_log_v1/
```

Useful files:

```text
compact_mechanism_summary.csv
compact_mechanism_trials.csv
compact_mechanism_run_metadata.json
```

This run used an image-disjoint compact basis:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_image_disjoint_compact_basis_delta025_v1/
    image_disjoint_compact_basis_delta0p25_fold0of2.npz
```

with:

```text
basis_key = basis
basis_mode = image_disjoint
basis_shape = [756, 50]
verified image_disjoint = true
```

Do not use `compact_mechanism_summary.csv` as the final Panel C evidence,
because it scores joint image-decoding accuracy. Reuse its projection and
response-variant logic.

Relevant implementation file:

```text
declan/backimage_trajectory_observer/analyze_compact_mechanism.py
```

Key logic to reuse:

```text
_load_basis(...)
_project_delta(...)
_variant_tables(...)
```

Existing response variants:

```text
full_exact
zero_static
compact_only
compact_removed
log_compact_only
log_compact_removed
random_k
unit_shuffle_compact
gain_only
static_pc_k
```

For the Figure 4C feature-space control, the required core variants are
`full_exact`, `zero_static`, `compact_only`, and `compact_removed`. Add an
explicit `compact_addback` or reconstruction QC row if useful; it should
reconstruct `full_exact` from compact and residual components within numerical
tolerance and should be treated as a sanity check, not a new biological claim.
In figure-facing text, `full_exact` can be described as the full joint observer.

## Feature-Posterior Code Path

The existing feature-posterior scorer is:

```text
declan/backimage_trajectory_observer/analyze_feature_posterior.py
```

Current observer modes:

```text
known
zero
joint
best_single_tau
motion_delta
```

Relevant scoring flow:

```text
score_image_identity_score_vectors(...)
score_by_mode["known"]
score_by_mode["zero"]
score_by_mode["joint"]
_mode_row(... feature_cosine ...)
_summary_rows(...)
```

Implementation options:

1. Extend `analyze_feature_posterior.py` with compact-variant arguments.
2. Or create a sibling script, for example:

```text
declan/backimage_trajectory_observer/analyze_feature_posterior_compact_mechanism.py
```

The sibling-script route is probably cleaner because it avoids disturbing the
already-used feature-posterior endpoint.

## Required Computation

For each selected response table:

1. Load the original observed response, candidate response table, known-eye
   response table, and zero/static response table exactly as
   `analyze_feature_posterior.py` does.
2. Load the image-disjoint compact basis above.
3. Construct variant response tables using the same delta projection contract
   as `analyze_compact_mechanism.py`:

```text
delta = response - zero
compact_delta = P_compact(delta)
residual_delta = delta - compact_delta

full_exact      = zero + delta
zero_static     = zero
compact_only    = zero + compact_delta
compact_removed = zero + residual_delta
compact_addback = zero + residual_delta + compact_delta
```

4. Score each variant through the same posterior-to-feature pipeline used by
   `analyze_feature_posterior.py`.
5. For every row, output feature-posterior metrics including:

```text
feature_neg_mse
feature_rmse
feature_cosine
candidate_posterior_true_mass
candidate_posterior_N_eff_fraction
score_true_rank
score_true_margin
```

6. Summarize by:

```text
candidate_set_mode
observation_scale
prior_family
likelihood_scale
latent
requested_k
response_variant
k_dim
basis_mode
basis_type
```

The headline summary should report at least:

```text
mean_feature_cosine
median_feature_cosine
mean_feature_neg_mse
mean_candidate_true_mass
median_candidate_N_eff_fraction
n_trials
```

and paired uncertainty for:

```text
full_exact - zero_eye
compact_only - zero_eye
compact_removed - zero_eye
compact_only - compact_removed
full_exact - compact_removed
known_eye - full_exact
```

Use paired trial-level bootstrap and sign-flip permutation, matching the
existing feature-posterior uncertainty convention where practical.

## Scope For The First Production Run

Use the same primary C conditions:

```text
run_dir:
  outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
    backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1

feature_npz:
  outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
    backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/
      feature_latent_arrays.npz

candidate_set_modes:
  hard_negative_structure

priors:
  axis_edge_parallel,axis_edge_orthogonal

motion_scales:
  0.5,1.0,2.0

likelihood_scales:
  1.0

latent_names:
  pyramid_local_field

pca_k_list:
  8

compact k_dims:
  start with 10 for the figure-facing result
  optionally include 2,5,10,20 for a diagnostic table
```

Suggested output directory:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1/
```

Suggested output files:

```text
feature_compact_mechanism_trials.csv
feature_compact_mechanism_summary.csv
feature_compact_mechanism_uncertainty.csv
feature_compact_mechanism_qc.csv
feature_compact_mechanism_report.md
feature_compact_mechanism_metadata.json
```

## Suggested Command Shape

If implemented as a sibling script:

```bash
.venv/bin/python -m declan.backimage_trajectory_observer.analyze_feature_posterior_compact_mechanism \
  --run-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1 \
  --out-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1 \
  --feature-npz outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/feature_latent_arrays.npz \
  --compact-basis-path outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz \
  --basis-mode image_disjoint \
  --basis-key auto \
  --candidate-set-modes hard_negative_structure \
  --priors axis_edge_parallel,axis_edge_orthogonal \
  --motion-scales 0.5,1.0,2.0 \
  --likelihood-scales 1.0 \
  --latent-names pyramid_local_field \
  --pca-k-list 8 \
  --k-dims 10 \
  --variants full_exact,zero_static,compact_only,compact_removed,compact_addback \
  --n-bootstrap 10000 \
  --n-permutations 10000 \
  --uncertainty-seed 17 \
  --progress-every 16
```

Add a smoke command with `--max-tables 8`, `--n-bootstrap 100`, and
`--n-permutations 100` before running the full output.

## Validation Gates

The coding agent should not update the main Figure 4C claim until these pass:

1. `full_exact` and `zero_static` reproduce the existing feature-posterior
   `joint` and `zero` curves within expected numerical tolerance when no
   compact projection is applied.
2. `compact_addback` reconstructs `full_exact` scores or responses within a
   documented tolerance.
3. Rate clipping/negative-rate QC is reported for every projected variant.
4. The compact basis is confirmed image-disjoint from the scored fold/source,
   or the report clearly flags any provenance caveat.
5. The result table includes both compact-source priors
   `axis_edge_parallel` and `axis_edge_orthogonal`; the main panel may plot the
   selected compact source, but the table/report must show both.
6. The feature-space result supports the intended story:

```text
compact_only feature recovery > zero_eye
compact_removed feature recovery near zero_eye
compact_only much closer to full_exact than compact_removed
known_eye remains the ceiling
```

If this pattern does not hold, keep Panel C as compact-subspace sufficiency
only and route compact removal to the caveat/supplement.

## Figure Update After The Metric Exists

If the validation gates pass:

1. Update the selected Panel C plot in:

```text
declan/figure4_active_sensing_atlas/scripts/build_selected_figure4_v4_design.py
```

2. Update the standalone C option sheet in:

```text
declan/figure4_active_sensing_atlas/scripts/build_panel_c_feature_recovery_options.py
```

3. Preferred main-panel visual:

```text
zeroed eye
compact subspace
compact removed
known eye ceiling
```

with x-axis:

```text
motion scale
```

and y-axis:

```text
feature recovery (cosine)
```

4. Regenerate:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_panel_c_feature_recovery_options.py
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_selected_figure4_v5_compact_layout.py
```

5. Update these docs:

```text
declan/figure4_active_sensing_atlas/provisional_figure4_v0.md
declan/figure4_active_sensing_atlas/provisional_panel_contract_v0.csv
declan/figure4_active_sensing_atlas/incomplete_results_flags.md
declan/figure4_active_sensing_atlas/figure_build_log.md
```

Remove the current caveat only if the new feature-space compact-removed result
passes the validation gates.

## Tests To Add

Add focused tests under:

```text
declan/backimage_trajectory_observer/tests/
```

Recommended tests:

```text
test_compact_variant_addback_reconstructs_full_response
test_feature_compact_full_and_zero_match_existing_observer_modes
test_feature_compact_removed_uses_residual_not_projection
test_feature_compact_summary_has_required_contrasts
```

Keep tests small and synthetic; do not require the full production cache.

## Claim Boundary For The Report

If successful, the result supports:

```text
The compact subspace carries much of the feature information that lets the
joint decoder recover features when eye trajectory is latent.
```

It does not by itself prove:

```text
the animal computes this posterior;
the posterior identifies the true eye trajectory;
compact subspace is the only useful response structure;
behavior optimizes this model objective.
```

Keep those boundaries in the generated report and figure caption.
