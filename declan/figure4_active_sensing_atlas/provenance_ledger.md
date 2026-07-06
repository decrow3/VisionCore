# Provenance Ledger

This file records where each atlas claim comes from. Add to this before moving
numbers into the working atlas document.

## Source Brief

The starting brief was the pasted Figure 4 atlas note attached in the Codex
thread on 2026-06-19. It proposed five expanded modules:

```text
A. FEMs turn a static image into a retinal movie.
B. FEM movies can improve visual encoding.
C. A joint image-and-eye observer recovers information lost by ignoring motion.
D. The model predicts image-dependent useful motion directions.
E. Free-viewing FEMs follow image geometry.
```

External citations in that pasted brief have not been re-verified in this local
folder. Verify them before manuscript use.

## Existing Figure 4 Active-Sensing Workspace

Code:

```text
declan/fig4_active_sensing/
  README.md
  generate_fig4_active_sensing.py
  run_fig4_active_sensing_sanity_checks.py
```

Default output:

```text
outputs/fig4_active_sensing/active_sensing_headline_figure/
```

Relevant note:

```text
The current figure generator is already cache-first and uses the canonical
756-unit V1 twin. Its README warns that deterministic static-plus-motion
feature-decoding gains are proxy scores, not literal mutual information.
```

Use:

```text
Good source for current compressed-headline figure conventions and for deciding
which panels are already figure-ready.
```

## Active-Sensing Movie-Information Workspace

Code and notes:

```text
declan/active_sensing_movie_information/
  README.md
  data_and_code_inventory.md
  generate_active_sensing_movie_information_figure.py
  generate_retinal_movie_transform_qc.py
```

Useful outputs:

```text
outputs/active_sensing_movie_information/
  active_sensing_movie_information_figure/
  active_sensing_movie_information_figure_frozen_20260615_pre_backimage_collab_pack/
  reafferent_variance_accounting/
  constrained_population_coding/
  information_accumulation/
  input_whitening_primary_psd/
  compact_basis_exports/
```

Claim boundary:

```text
Useful for retinal rendering/QC, the broad movie-information framing, and
covariance bridge notes. Do not use its older 16-channel natural-image endpoint
as the main Figure 4 evidence when the claim needs the canonical 756-unit twin.
```

## Module A: Retinal Movie Premise And QC

Primary QC source:

```text
outputs/active_sensing_movie_information/
  active_sensing_movie_information_figure_frozen_20260615_pre_backimage_collab_pack/
    retinal_movie_transform_qc_summary.csv
```

QC read:

```text
real FEM movies:
  temporal contrast RMS mean = 11.245
  motion power vs matched stabilized mean = 1462.431
  movie power mean = 15178.177
  n = 108 image/trace movies

stabilized movies:
  temporal contrast RMS mean = 0.000
  motion power vs matched stabilized mean = 0.000
  movie power mean = 15185.182
  n = 108 image/trace movies
```

Canonical downstream source:

```text
outputs/fig4_active_sensing/active_sensing_headline_figure/
  fig4_active_sensing_headline_stats.json
```

BackImage/V1-twin provenance:

```text
n_images = 256
n_sessions = 29
trace_samples_per_condition = 4
n_trace_sources = 151
population = canonical 756-unit V1 twin
CV = grouped by image, 5 outer folds
median effective/requested RMS = 1.0
max clipped fraction = 0.0
```

Covariance bridge source:

```text
outputs/active_sensing_movie_information/reafferent_variance_accounting/
  variance_accounting_aggregate_summary.csv
```

Bridge read:

```text
compact derivative capture:
  session mean fraction = 0.484, SEM = 0.030, sessions = 13

finite-difference tangent capture:
  session mean fraction = 0.436, sessions = 1

noise-correlation eye-correction proxy:
  session mean fraction = 0.333, SEM = 0.037, sessions = 4

reliable-shared denominator proxy:
  session mean fraction = 0.848, SEM = 0.040, sessions = 4
```

Generated atlas assets:

```text
declan/figure4_active_sensing_atlas/scripts/plot_panel_a_subpanels.py
declan/figure4_active_sensing_atlas/figures/panel_A/
  A1_retinal_movie_transform.png
  A2_movie_transform_qc.png
  A3_gradient_sampling_cartoon.png
  A4_backimage_pipeline_bridge.png
  A5_covariance_bridge_guardrail.png
  panel_A_retinal_movie_transform_values.csv
  panel_A_movie_transform_qc_values.csv
  panel_A_gradient_sampling_values.csv
  panel_A_backimage_pipeline_values.csv
  panel_A_covariance_bridge_values.csv
  panel_A_subpanels_caption.md
```

Safe claim:

```text
During fixation, eye movements convert a fixed screen image into shifted
retinal samples. This motivates treating retinal motion as part of the sensory
input and provides the setup for the downstream BackImage analyses. The
covariance bridge is corroborating context, not the main Figure 4 endpoint.
```

## Module B: Aggregate FEM Information

2026-06-21 corrected static-mean power-rerun integration:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/
    incremental_staticmean_plus_motion_tworeadout_v2/

outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_local_pairing_Iz_power_pyramid_k16_rel025_0p5_1_seed7_k64_v1/
  backimage_local_pairing_Iz_power_pyramid_k16_rel025_0p5_1_seed11_k64_v1/

outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_feature_decomposition_adjudication_v6_staticmean_corrected_power_rerun_primary_scales/

outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/
    incremental_staticmean_plus_motion_allreadouts_v1/readout_atlas_figures/
```

Correction:

```text
The previous v5 aggregate temporal-PCA posthoc used static temporal PCs as the
static baseline. For a static trace that baseline is nearly zero, so the
temporal-PCA "gain beyond static" claim was not a valid static-response
comparison. The v6 posthoc uses the static mean response as the baseline for
all motion summaries.
```

Corrected power read:

```text
pyramid_local_field k16 delta_mean, n=384:
  empirical - static_mean:
    0.25x -0.65 CI [-2.17, +0.93]
    0.5x  +0.16 CI [-1.35, +1.77]
    1x    +1.76 CI [+0.58, +3.05]

  empirical - OU:
    0.25x -0.20 CI [-1.04, +0.77]
    0.5x  -1.29 CI [-2.14, -0.47]
    1x    -0.11 CI [-0.86, +0.67]

temporal_pca secondary diagnostic:
  empirical - static_mean is negative at 0.25x, 0.5x, and 1x.
  empirical - OU remains strongly positive at 0.25x, 0.5x, and 1x.
```

Adjudication read:

```text
delta_mean is the corrected primary feature readout:
  score_with_joint_axis_term = 1.912
  score_without_joint_axis_term = 1.971
  aggregate_score = 0.332
  local_Iz_score = 2.139

temporal_pca is demoted to secondary diagnostic:
  score_with_joint_axis_term = 0.608
  score_without_joint_axis_term = 0.667
  aggregate_score = 0.593
  local_Iz_score = 0.575
```

Use:

```text
Use the v6 values and all-readout atlas as the current production candidate
for Panel B text. The selected v5 composite has been redrawn once from the
corrected static-mean posthoc, but manuscript promotion should wait for the
Panel-B-style all-readout review and OU trace-control verdict. The old v5
temporal-PCA absolute-gain text is superseded.
```

All-readout/nested-alpha read:

```text
mean: strongest absolute aggregate candidate under nested alpha
delta_mean: static-subtracted motion-induced/local-pairing bridge
temporal PCA/DCT: order-sensitive empirical-vs-control diagnostics
OU: audit-pending, not yet a headline negative control
```

Planning note:

```text
declan/backimage_aggregate_fem_information_plan.md
```

Code:

```text
declan/fixation_statistics_by_stimulus/run_backimage_aggregate_fem_information.py
declan/fixation_statistics_by_stimulus/summarize_backimage_aggregate_cache_proxy.py
declan/fixation_statistics_by_stimulus/summarize_backimage_aggregate_incremental_motion.py
```

Primary run:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
```

Use this posthoc folder for incremental claims:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
    incremental_static_plus_motion_relids/
```

Do not use for incremental claims:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
    incremental_static_plus_motion/
```

Reason:

```text
The first automatic posthoc folder used old-style scale IDs and produced empty
gain tables.
```

Run scope:

```text
images = 256
trace samples per family/scale/image = K=4
population = canonical 756-unit V1 twin
families = empirical, OU, Brownian, rotated
scales = 0.25x, 0.5x, 1x, 1.5x, 2x
features = gabor_local_field, pyramid_local_field
feature ranks = 4, 8
response summaries = temporal_pca, temporal_delta_pca, temporal_dct,
  temporal_dct_delta, mean, delta_mean
CV = grouped by image, 5 outer folds
trace policy = drift-only, common-unclipped source pool
```

Motion QC:

```text
accepted drift-only trace sources = 151 / 256
median effective/requested RMS = 1.0 for every family/scale
clipped fraction = 0.0 for every family/scale
```

2026-06-21 supersession note:

```text
The temporal-PCA absolute-gain numbers below are historical for figure claims.
They used the older aggregate posthoc contract and are superseded by the n384
k16 static-mean correction plus all-readout/nested-alpha audit. Temporal
PCA/DCT should now be read as order-sensitive empirical-vs-control diagnostics,
not as the absolute gain-over-static headline.
```

Primary temporal-PCA incremental result:

```text
static + empirical temporal_pca versus static alone

Gabor k=4:
  0.25x  +14.31, CI [+7.45, +21.79]
  0.5x   +13.04, CI [+6.81, +20.89]
  1x     +9.10,  CI [+3.73, +14.86]
  1.5x   +9.98,  CI [+5.36, +15.87]
  2x     +9.07,  CI [+3.87, +15.73]

Pyramid k=8:
  0.25x  +5.20, CI [+3.02, +7.68]
  0.5x   +4.89, CI [+2.88, +7.07]
  1x     +3.93, CI [+1.93, +5.86]
  1.5x   +4.44, CI [+2.34, +6.64]
  2x     +4.21, CI [+2.38, +6.23]
```

Control read:

```text
Empirical beats OU robustly across scales. Brownian and rotated are most
clearly below empirical at 0.25x-0.5x; Brownian becomes competitive at larger
scales.
```

Generated atlas assets:

```text
declan/figure4_active_sensing_atlas/scripts/plot_panel_b_subpanels.py
declan/figure4_active_sensing_atlas/figures/panel_B/
  B1_task_schematic.png
  B2_motion_family_qc.png
  B3_empirical_gain_vs_static.png
  B4_empirical_minus_controls.png
  B5_absolute_gain_guardrail.png
  panel_B_motion_qc_values.csv
  panel_B_gain_vs_static_values.csv
  panel_B_control_contrast_values.csv
  panel_B_absolute_gain_guardrail_values.csv
  panel_B_subpanels_caption.md
```

Safe claim:

```text
Empirical drift-like motion adds feature-decodable structure beyond static
V1-twin responses. The result is distributional, scale/readout scoped, and
twin-scoped; it is not exact-trajectory optimality.
```

## Module C: Exact Trajectory-Table Observer

Results log:

```text
declan/backimage_trajectory_observer/results_log.md
```

Core code:

```text
declan/backimage_trajectory_observer/observer.py
declan/backimage_trajectory_observer/likelihood.py
declan/backimage_trajectory_observer/candidate_sets.py
declan/fixation_statistics_by_stimulus/run_backimage_trajectory_table_observer.py
```

Primary matched-static run:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
```

Run scope:

```text
n_images = 64
n_candidates = 8
candidate_set_modes = hard_negative_structure, matched_static_response
observation_family = empirical
prior_families = empirical, OU
scales = 0.5, 1.0
n_prior_trajectories = 8
trajectory_prior_mode = leave_one_out
likelihood_scales = 0.5, 1.0
```

Key summary:

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

At `matched_static_response`, `1.0x`, likelihood scale `1.0`:

```text
empirical prior:
  zero = 0.328
  joint = 0.766
  recovery of known-zero gap ~= 65%
  median N_eff / K ~= 0.364

OU prior:
  zero = 0.328
  joint = 0.797
  recovery of known-zero gap ~= 70%
  median N_eff / K ~= 0.400
```

Generated atlas assets:

```text
declan/figure4_active_sensing_atlas/scripts/plot_panel_c_subpanels.py
declan/figure4_active_sensing_atlas/figures/panel_C/
  C1_observer_schematic.png
  C2_accuracy_ordering.png
  C3_matched_static_rescue.png
  C4_posterior_concentration.png
  C5_scale_gap_guardrail.png
  C6_compact_mechanism_guardrail.png
  panel_C_accuracy_ordering_values.csv
  panel_C_matched_static_rescue_values.csv
  panel_C_posterior_concentration_values.csv
  panel_C_scale_gap_guardrail_values.csv
  panel_C_compact_guardrail_values.csv
  panel_C_subpanels_caption.md
```

Safe claim:

```text
Trajectory marginalization over exact natural-image response tables can rescue
image identity from pose uncertainty. This result does not by itself establish
compact geometry as the mechanism or empirical FEM optimality.
```

## Module C Mechanism: Compact Projection

Code:

```text
declan/backimage_trajectory_observer/analyze_compact_mechanism.py
declan/backimage_trajectory_observer/build_image_disjoint_compact_basis.py
declan/backimage_trajectory_observer/summarize_compact_mechanism_followups.py
```

Image-disjoint run:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    compact_mechanism_image_disjoint_fold0_n512_k2_5_10_20_rand8_log_v1/
```

Basis export:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_image_disjoint_compact_basis_delta025_v1/
    image_disjoint_compact_basis_delta0p25_fold0of2.npz
```

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

Specificity caveat:

```text
Compact-only beats random/unit-shuffle/gain controls, but static-PC subspaces
recover similar true-score rescue at k=10/20. Use as sufficiency evidence, not
unique-mechanism proof.
```

## Module C Diagnostic: Continuous Feature Embedding

Code:

```text
declan/figure4_active_sensing_atlas/scripts/
  build_panel_c_continuous_feature_embedding_reconstruction.py
```

Output:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/
  continuous_feature_embedding/
    continuous_feature_embedding_reconstruction_manifest.json
    continuous_feature_embedding_reconstruction_summary.csv
    continuous_feature_embedding_reconstruction_contrasts.csv
    continuous_feature_embedding_reconstruction.png
```

Run scope:

```text
response cache = backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1
feature target = 32D whitened PCA of pyramid_local_field
response target = 20D image-disjoint compact basis, flattened across 40 time bins
folding = source-row cross-fit
endpoint = cosine(z_hat, z_true)
```

Full-cache read:

```text
known-trace feature cosine:             0.1261
hidden-eye tau-nuisance feature cosine: 0.1036
zero-eye model on motion:               0.0471
0x stabilized feature cosine:           0.1860

known-trace - hidden:                   +0.0225
hidden - zero-eye model:                +0.0565
known-trace motion - 0x stabilized:     -0.0599
hidden motion - 0x stabilized:          -0.0824
```

Boundary:

```text
This is the first candidate-free continuous feature-embedding branch. It
supports the feasibility of continuous z inference and shows hidden-eye gain
over the zero-eye model, but it does not replace the promoted 4C
candidate-posterior endpoint because the stabilized 0x counterfactual remains
higher in this compact-basis linear-Gaussian readout.
```

## Module D: Axis-Conditioned Observer

Code:

```text
declan/axis_conditioned_backimage_trajectory_observer/
declan/axis_conditioned_backimage_trajectory_observer/axis_conditioned_traces.py
declan/axis_conditioned_backimage_trajectory_observer/summarize_axis_conditioned_run.py
declan/fixation_statistics_by_stimulus/run_backimage_trajectory_table_observer.py
```

Clean matched-static pilot:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1/
```

At `matched_static_response`, `0.5x`, likelihood scale `1.0`:

```text
known-eye = 1.000
zero-eye = 0.641
axis_edge_parallel joint = 0.859
axis_edge_orthogonal joint = 0.828
```

Clean hard-negative scale sweep:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
```

At `hard_negative_structure`, likelihood scale `1.0`:

```text
0.5x:
  zero-eye = 0.609
  parallel joint = 0.813
  orthogonal joint = 0.781

1.0x:
  zero-eye = 0.391
  parallel joint = 0.797
  orthogonal joint = 0.805

2.0x:
  zero-eye = 0.336
  parallel joint = 0.680
  orthogonal joint = 0.742
```

Safe claim:

```text
Axis-conditioned priors rescue image identity above zero-eye. Directional
preference is mixed and depends on candidate set/scale, so use this branch as
evidence for image-conditioned axis structure, not a universal parallel rule.
```

Generated atlas assets:

```text
declan/figure4_active_sensing_atlas/scripts/plot_panel_d_subpanels.py
declan/figure4_active_sensing_atlas/figures/panel_D/
  D1_local_axis_schematic.png
  D2_axis_conditioned_accuracy.png
  D3_axis_preference_guardrail.png
  D4_edge_parallel_stability.png
  D5_objective_alignment_guardrail.png
  panel_D_axis_conditioned_values.csv
  panel_D_axis_preference_values.csv
  panel_D_edge_stability_values.csv
  panel_D_objective_guardrail_values.csv
  panel_D_subpanels_caption.md
```

## Module D: Edge-Parallel Preservation

Code:

```text
declan/fixation_statistics_by_stimulus/run_backimage_edge_parallel_stability_screen.py
```

Primary output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_edge_parallel_stability_screen_yfix_n256_pop256/
```

Input:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_image_structure_reviewed_v2_screenfiltered_yfix/
    backimage_image_fem_windows.csv
```

Run scope:

```text
windows = 256
displacement = 0.125 deg
twin = True
population = 256
hold frames = 40
shuffle nulls = 5000
session bootstraps = 5000
```

Result:

```text
pixel:
  session mean advantage = 300.54
  CI = [172.789, 408.961]
  positive sessions = 26 / 29

twin:
  session mean advantage = 0.000454497
  CI = [0.000371047, 0.000536519]
  positive sessions = 29 / 29
```

Safe claim:

```text
Edge-parallel motion disrupts pixels and V1-twin responses less than
edge-orthogonal motion in this local preservation audit.
```

## Module E: Behavioral Drift-Edge Alignment

Code:

```text
declan/fixation_statistics_by_stimulus/posthoc_backimage_edge_alignment_distribution_inspection.py
declan/fixation_statistics_by_stimulus/run_backimage_image_structure_analysis.py
```

Primary output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_edge_alignment_distribution_inspection/
```

Source window table:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_image_structure_reviewed_v2_screenfiltered_yfix/
    backimage_image_fem_windows.csv
```

Alignment metric:

```text
cos(2 * drift-edge delta)
+1 = edge-parallel
0 = 45 deg from edge
-1 = edge-orthogonal
```

Distribution summary:

```text
All windows:
  n_windows = 11749
  n_sessions = 30
  mean session cos2 = 0.105
  CI = [0.067, 0.145]
  median abs delta = 39.0 deg

Reliable axes:
  n_windows = 6242
  n_sessions = 30
  mean session cos2 = 0.140
  CI = [0.089, 0.188]
  median abs delta = 36.4 deg

High confidence:
  n_windows = 1045
  n_sessions = 30
  mean session cos2 = 0.269
  CI = [0.138, 0.396]
  median abs delta = 25.6 deg
```

Endpoint-zone enrichment:

```text
Observed / uniform expected fraction in the parallel <=15 deg zone:
  all windows = 1.304
  reliable axes = 1.427
  high confidence = 2.124

Observed / uniform expected fraction in the orthogonal >=75 deg zone:
  all windows = 0.906
  reliable axes = 0.851
  high confidence = 0.833
```

Metric-convention guardrail:

```text
All windows:
  unweighted session mean cos2 = 0.105
  weighted headline-style cos2 = 0.181

Reliable axes:
  unweighted session mean cos2 = 0.140
  weighted headline-style cos2 = 0.201
```

Generated atlas assets:

```text
declan/figure4_active_sensing_atlas/scripts/plot_panel_e_subpanels.py
declan/figure4_active_sensing_atlas/figures/panel_E/
  E1_behavior_setup_example.png
  E2_behavior_alignment_strength.png
  E3_parallel_zone_enrichment.png
  E6_full_distribution_session_diagnostic.png
  E7_confidence_signed_delta_diagnostic.png
  E8_endpoint_null_diagnostic.png
  E4_metric_convention_guardrail.png
  E5_scope_summary.png
  panel_E_behavior_example_values.csv
  panel_E_alignment_strength_values.csv
  panel_E_endpoint_enrichment_values.csv
  panel_E_contour_following_source_panels.csv
  panel_E_metric_convention_values.csv
  panel_E_scope_summary_values.csv
  panel_E_subpanels_caption.md
```

Source-diagnostic copies:

```text
E6 <- backimage_edge_alignment_distribution_inspection/
      edge_alignment_window_and_session_distributions.png
E7 <- backimage_edge_alignment_distribution_inspection/
      edge_alignment_confidence_and_signed_delta.png
E8 <- backimage_edge_alignment_distribution_inspection/
      edge_alignment_endpoint_null_diagnostic.png
```

Safe claim:

```text
During free viewing of natural images, drift/fixation-cloud orientation is
modestly but reliably aligned with local image geometry, especially when the
local axis estimate is reliable.
```

## Module E: Model Objectives Versus Raw Edge

Output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_conditional_fixation_objectives_twin_axis_only_n256/
    alignment_by_objective_summary.csv
```

Raw edge row:

```text
raw_edge_axis:
  n_windows = 256
  n_sessions = 29
  mean session cos2 = 0.182
  weighted session cos2 = 0.218
  positive sessions = 23 / 29
```

Model-objective caveat:

```text
The current V1-twin pose-aware/pose-blind/Pareto objectives do not cleanly
outperform raw edge orientation. Raw image geometry is the baseline to beat.
```

## 4C Continuous Feature-Embedding Target-Space Sweep

Output:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/
  continuous_feature_embedding/
    continuous_feature_embedding_reconstruction_summary.csv
    continuous_feature_embedding_reconstruction_contrasts.csv
    continuous_feature_embedding_reconstruction_manifest.json
```

Implemented options:

```text
global_centered_whitened_pca
fold_centered_whitened_pca
fold_zscore_whitened_pca
fold_zscore_pca
```

All-scale feature cosine:

```text
feature-space mode              known   hidden  zero-model  0x stabilized
global centered whitened PCA    0.1261  0.1036  0.0471      0.1860
fold centered whitened PCA      0.1203  0.0994  0.0567      0.1896
fold z-scored whitened PCA      0.1505  0.1271  0.0618      0.2330
fold z-scored PCA               0.1372  0.1346  0.1368      0.3218
```

Status:

```text
Best Gaussian-prior variant so far: fold_zscore_whitened_pca.
Still diagnostic: no target-space option beats the 0x stabilized
counterfactual with the current compact-basis linear-Gaussian readout.
```

## 4C Continuous Feature-Embedding MLP Upper Bound

Output:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/
  continuous_feature_embedding_mlp_hammer/
    continuous_feature_embedding_reconstruction_summary.csv
    continuous_feature_embedding_reconstruction_contrasts.csv
    continuous_feature_embedding_reconstruction_manifest.json
```

Configuration:

```text
decoder_mode = mlp
target = fold_zscore_whitened_pca, 32D pyramid_local_field
response = 20D image-disjoint compact response movie
MLP = 4 layers, hidden_dim 512, dropout 0, AdamW lr 1e-3, weight_decay 1e-5
split = source-row cross-fit, source-group validation within training folds
```

All-scale feature cosine:

```text
known-eye moving response:           0.2329
hidden/tau-nuisance moving response: 0.3482
zero-eye model on moving response:   0.1824
0x stabilized response:              0.2819
```

Paired contrasts:

```text
hidden - zero-eye model:       +0.1657  CI [+0.1422, +0.1912]
hidden motion - 0x stabilized: +0.0662  CI [+0.0450, +0.0887]
known motion - 0x stabilized:  -0.0491  CI [-0.0684, -0.0302]
```

Safe claim:

```text
With a strong nonlinear readout, compact moving responses contain recoverable
feature information above both the zero-eye model and the 0x stabilized
counterfactual. This is an information upper-bound result, not a claim that V1
features are linearly decodable downstream.
```

## 4C Continuous-Tau MLP Feature Decoder

Output:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/
  continuous_tau_mlp_feature_decoder_residual/
    continuous_tau_mlp_feature_decoder_summary.csv
    continuous_tau_mlp_feature_decoder_contrasts.csv
    continuous_tau_mlp_feature_decoder_models.csv
    continuous_tau_mlp_feature_decoder_manifest.json
```

Configuration:

```text
script = declan/figure4_active_sensing_atlas/scripts/build_panel_c_continuous_tau_mlp_feature_decoder.py
target = fold_zscore_whitened_pca, 32D pyramid_local_field
response = 20D image-disjoint compact response movie
tau = continuous tau_hat from promoted continuous-joint scorer
decoder = MLP, 4 layers, hidden_dim 512
split = source-row cross-fit, source-group validation within training folds
endpoint = direct feature embedding recovery, not image-candidate choice
```

All-scale feature cosine:

```text
augmented compact-only hidden readout: 0.3695
augmented true-tau residual readout:   0.3683
augmented true-tau raw readout:        0.3521
augmented true-tau interactions:       0.3499
augmented tau_hat residual readout:    0.3440
augmented tau_hat raw readout:         0.3140
augmented tau_hat interactions:        0.3078
augmented 0x stabilized readout:       0.3243
augmented known-eye readout:           0.3196
```

Paired contrasts:

```text
augmented compact-only - augmented 0x: +0.0452  CI [+0.0292, +0.0618]
true-tau residual - augmented 0x:      +0.0440  CI [+0.0276, +0.0610]
tau_hat residual - augmented 0x:       +0.0196  CI [+0.0029, +0.0374]
true-tau residual - compact-only:      -0.0012  CI [-0.0183, +0.0154]
true-tau residual - raw true-tau:      +0.0162  CI [-0.0005, +0.0333]
```

Safe claim:

```text
The candidate-free nonlinear feature decoder beats the fair augmented static
baseline through nuisance-augmented moving compact responses. Raw tau
concatenation is not the right recorded-eye interface. A nested residual
eye-trace decoder behaves like a guarded diagnostic: true-tau residual matches
the compact-only readout and beats augmented static, while estimated tau_hat
residual also beats augmented static. This is not a physical known-pose
ceiling.
```

## 4C Continuous-Tau MLP Along/Across Contrast

Output:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/
  continuous_tau_mlp_feature_decoder_residual/
    continuous_tau_mlp_axis_summary.csv
    continuous_tau_mlp_axis_contrasts.csv
    continuous_tau_mlp_axis_contrast_README.md
```

Configuration:

```text
script = declan/figure4_active_sensing_atlas/scripts/summarize_panel_c_continuous_tau_mlp_axis_contrast.py
source = continuous_tau_mlp_feature_decoder_residual/continuous_tau_mlp_feature_decoder_trials.csv
contrast = axis_edge_parallel - axis_edge_orthogonal
paired by = trial_id, prior_scale, true_source_row, input_mode
```

All-scale feature-cosine contrast:

```text
augmented compact-only:        +0.00000  CI [+0.00000, +0.00000]
true-tau residual:            +0.00000  CI [+0.00000, +0.00000]
tau_hat residual:             +0.00013  CI [-0.00514, +0.00620]
raw tau_hat concatenation:    -0.01095  CI [-0.02414, +0.00218]
```

Safe claim:

```text
The candidate-free residual MLP method does not show a meaningful along-contour
advantage. In this paired setup the observed response and true trace are the
same across axis labels; only the axis-conditioned inferred tau_hat can differ,
and its residual-decoder contrast is effectively zero.
```

## 4C Direct Axis MLP Feature Decoder

Output:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/
  direct_axis_mlp_feature_decoder/
    direct_axis_mlp_feature_decoder_trials.csv
    direct_axis_mlp_feature_decoder_summary.csv
    direct_axis_mlp_feature_decoder_contrasts.csv
    direct_axis_mlp_feature_decoder_models.csv
    direct_axis_mlp_feature_decoder_manifest.json
    direct_axis_mlp_feature_decoder_README.md
```

Configuration:

```text
script = declan/figure4_active_sensing_atlas/scripts/build_panel_c_direct_axis_mlp_feature_decoder.py
source response tables = promoted axis_edge_parallel / axis_edge_orthogonal continuous-joint cache
test observation = true candidate's sampled prior response movie for each trajectory sample
target = fold_zscore_whitened_pca, 32D pyramid_local_field
decoder = source-disjoint MLP; includes response-only, raw true-tau, residual true-tau, static control
contrast = axis_edge_parallel - axis_edge_orthogonal
CI = bootstrap over trial/scale cluster means; row-level trajectory-sample CI retained in CSV
```

All-scale feature-cosine contrast:

```text
compact response-only: -0.00960  CI [-0.01556, -0.00380]
true-tau residual:     -0.00557  CI [-0.01093, +0.00019]
raw true-tau concat:   -0.00120  CI [-0.00800, +0.00569]
0x static control:     +0.00000  CI [+0.00000, +0.00000]
```

Safe claim:

```text
The direct candidate-free MLP test does not show a reliable along-contour
advantage. A second source-fold seed (`20260625`) changes compact response-only
from `-0.00960` to `+0.00597`, with a cluster CI crossing zero, while true-eye
residual stays near null. The direct MLP axis effect should therefore be treated
as null/split-sensitive, not as a stable across-axis advantage. This should keep
the Figure 4D along/across story separate from the candidate-free 4C MLP
feature-decoder result.
```

Matched-static rerun:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/diagnostics/
  direct_axis_mlp_feature_decoder_matched_static/
    direct_axis_mlp_feature_decoder_trials.csv
    direct_axis_mlp_feature_decoder_summary.csv
    direct_axis_mlp_feature_decoder_contrasts.csv
    direct_axis_mlp_feature_decoder_models.csv
    direct_axis_mlp_feature_decoder_manifest.json
```

Configuration difference:

```text
source response tables = backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1
feature npz = backimage_axis_conditioned_matched_static_feature_posterior_gabor_pyramid_k4_8_uncertainty_v2
available modes = compact response-only, 0x static control
```

All-scale feature-cosine contrast:

```text
compact response-only: -0.00541  CI [-0.01492, +0.00426]
0x static control:     +0.00000  CI [+0.00000, +0.00000]
```

Matched-static safe claim:

```text
The matched-static direct candidate-free MLP test also does not show a reliable
along-contour advantage. The first source split is `-0.00541`, CI
`[-0.01492, +0.00426]`; the second source split is `+0.00191`, CI
`[-0.00668, +0.01086]`. The MLP axis result is null/split-sensitive, even
though the older known-axis posterior diagnostic is separately across-favoring.
```
