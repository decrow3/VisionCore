# Feature Decomposition Adjudication Handoff

Last updated: 2026-06-20.

## Purpose

This handoff defines the priority that should happen before any "canonical"
large BackImage active-sensing figure run:

```text
Find the feature/readout decomposition in which empirical or contour-aligned
FEM utility is most stable, robust, and biologically relevant.
```

The working prior is that the animal may be doing something close to useful or
optimal FEM sampling. Therefore it is reasonable to search for the model feature
space where that utility is expressed most clearly. The guardrail is that the
chosen feature specification must be locked before the canonical large run or
main figure claim.

This replaces a premature "run bigger with the current default Gabor/pyramid
k=4/8" plan. The immediate goal is not maximum effect size; it is a stable,
interpretable feature target that makes contour-following predictions
reproducible across the existing aggregate FEM, local `I_z`, and joint-posterior
branches.

## Core Question

```text
Which feature target and response summary make edge-parallel or empirical FEM
utility stable across scale, seed/control family, and analysis family?
```

The desired pattern is:

```text
edge-parallel / empirical motion helps recover local natural-image structure
or reduces pose/feature cost
```

not merely:

```text
one feature/k combination gives the largest isolated positive.
```

## Scientific Claim Target

If successful, this branch supports the statement:

```text
The model predicts that contour-aligned or empirical drift-like motion is useful
for preserving/recovering a specific natural-image feature decomposition, and
the animal's spontaneous BackImage drift is biased toward that same
contour-aligned geometry.
```

This is the closest non-behavioral evidence available without an explicit
perceptual task. It is a model-behavior alignment claim, not proof that the
animal behaviorally uses the decoded information.

## Source Notes To Read First

Read:

```text
declan/fem_v1_maximal_story_priority_checklist.md
declan/active_sensing_roadmap_after_vernier_fixation_image_structure.md
declan/backimage_aggregate_fem_information_plan.md
declan/backimage_local_pairing_Iz_revisit_plan.md
declan/axis_conditioned_backimage_trajectory_observer_plan.md
declan/raw_edge_roadblock_handoff.md
declan/aggregate_fem_figure_robustness_handoff.md
```

Then inspect the code knobs in:

```text
declan/fixation_statistics_by_stimulus/run_backimage_aggregate_fem_information.py
declan/fixation_statistics_by_stimulus/summarize_backimage_aggregate_incremental_motion.py
declan/fixation_statistics_by_stimulus/run_backimage_local_pairing_Iz_revisit.py
declan/backimage_trajectory_observer/analyze_feature_posterior.py
declan/fixation_statistics_by_stimulus/run_backimage_latent_information_screen.py
```

Important existing knobs:

```text
--latent-names
--pca-k-list
--latent-crop-px
--center-crop-px
--local-field-grid
--summaries
--ridge-alpha-mode
--fixed-ridge-alpha
```

Existing summaries include:

```text
temporal_pca
temporal_delta_pca
temporal_dct
temporal_dct_delta
mean
delta_mean
```

## Existing Evidence To Respect

### Aggregate FEM

Primary run:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
```

Use the corrected incremental folder:

```text
incremental_static_plus_motion_relids/
```

Current pattern:

- empirical temporal-PCA feature gain beyond static is positive across tested
  scales;
- empirical beats OU robustly;
- empirical advantage over Brownian/rotated is strongest at `0.25x-0.5x` and
  narrows near `1x-2x`;
- the run used Gabor/pyramid local fields and `k=4,8`;
- this is currently the strongest distributional BackImage active-sensing
  positive, but it is not axis-explicit.

### Local `I_z` Pairing

Clean result:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_gabor_pyramid_rel025_1_seed7_v1/
```

Current pattern:

- actual paired empirical traces beat matched unpaired empirical traces for
  `delta_mean` feature-response gains in both Gabor and pyramid local fields;
- temporal PCA/DCT summaries are weak or negative;
- rotated actual controls remain competitive;
- this branch is local image-contingent evidence, not yet unique-axis optimum
  evidence.

Earlier seed/pathfinder outputs exist, including seed0/seed1 k8 and cleaner
K32 variants. Treat them as stability evidence only after checking feature
geometry and matched-control logic.

### Joint Posterior

Current shared-source n128 run:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
```

Latest feature posterior:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k4_8_16_uncertainty_v1/
```

Current pattern:

- joint-minus-zero feature recovery is strong across axis/scale/latent/k;
- parallel-minus-orthogonal feature recovery remains non-significant;
- `0.5x` and `1.0x` often trend parallel-positive;
- `2.0x` trends zero or orthogonal-positive, especially for richer `k`;
- k16 already ran, so do not describe k16 as still pending.

## Feature/Readout Search Space

### Latent Families

Primary:

```text
gabor_local_field
pyramid_local_field
```

Secondary/sensitivity:

```text
dct_local_field
gabor_center
pyramid_center
dct_center
```

Interpretation preference:

- local-field features are more relevant to contour-following and natural-image
  local structure;
- center features are useful sensitivity checks but should not become the
  primary target unless local-field features fail;
- DCT is a broad frequency-control feature, not the main biological contour
  representation unless it clearly outperforms and remains interpretable.

### Feature Dimension

Evaluate:

```text
k = 2, 4, 8, 16, 32
```

Use current `selected_windows_zscore_pca` feature space as the first pass. Add
feature-space variants only if needed:

```text
zscore PCA
whitened PCA scores
band-balanced PCA or per-band normalization
variance-fraction matched PCA
```

Selection should prefer a plateau or stable range over a sharp one-off peak.
For example, `k=8/16` with consistent signs is preferable to `k=32` if k32 is
larger but fragile.

### Response Summaries

Evaluate:

```text
delta_mean
temporal_pca
temporal_delta_pca
temporal_dct
temporal_dct_delta
mean
```

Interpretation preference:

- `delta_mean`: best current local-pairing signal; interpretable as integrated
  motion-induced feature response change.
- `temporal_pca`: strongest current aggregate branch; interpretable as a
  compact movie-response summary.
- `temporal_dct`: useful temporal-order control.
- `mean`: diagnostic, not primary unless it cleanly carries the result.
- `temporal_delta_pca` / `temporal_dct_delta`: useful for separating static
  plus motion from absolute response effects.

### Scales

Primary:

```text
0.25x
0.5x
1.0x
```

Sentinel:

```text
2.0x
```

Optional:

```text
1.5x in aggregate FEM
```

Desired contour-following-compatible pattern:

```text
small/natural scales show positive empirical or edge-parallel utility;
2x does not become the only positive condition.
```

### Controls

Aggregate:

```text
empirical vs OU
empirical vs Brownian
empirical vs rotated
empirical vs static
```

Local `I_z`:

```text
actual_paired_empirical vs matched_unpaired_empirical
actual_paired_empirical vs rotated_actual_90
actual_paired_empirical vs ou_matched_actual
actual_paired_empirical vs brownian_matched_actual
edge_axis vs edge_orthogonal where available
```

Joint posterior:

```text
axis_edge_parallel vs axis_edge_orthogonal
joint vs zero
known-minus-joint pose cost
motion_delta as diagnostic only
```

## Proposed New Script

Create:

```text
declan/fixation_statistics_by_stimulus/analyze_backimage_feature_decomposition_adjudication.py
```

Default output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_feature_decomposition_adjudication_v1/
```

This should be cache-first. The first implementation should not run new V1
forward passes.

Suggested CLI:

```text
--aggregate-run-dir
--aggregate-incremental-dir
--local-pairing-dirs
--joint-feature-dirs
--joint-run-dir
--out-dir
--latent-names gabor_local_field,pyramid_local_field
--k-list 2,4,8,16,32
--summaries delta_mean,temporal_pca,temporal_delta_pca,temporal_dct,temporal_dct_delta,mean
--primary-scales rel_0p25x,rel_0p5x,rel_1x
--sentinel-scales rel_2x
--session-bootstrap-n 10000
--random-seed 0
```

## Implementation Plan

### Stage 1: Inventory Existing Feature Results

Build a long-form result index from:

```text
aggregate:
  incremental_static_plus_motion_relids/incremental_gain_vs_static.csv
  incremental_static_plus_motion_relids/incremental_gain_contrasts.csv
  decode_summary.csv
  covariance_summary.csv

local I_z:
  decode_summary.csv
  decode_contrasts.csv
  incremental_decode_summary.csv
  incremental_gain_vs_static.csv
  incremental_gain_contrasts.csv

joint posterior:
  feature_posterior_summary.csv
  feature_axis_contrasts.csv
  feature_motion_evidence_contrasts.csv
  feature_posterior_uncertainty.csv
```

Write:

```text
existing_feature_result_inventory.csv
existing_feature_result_inventory.md
```

Each row should include:

```text
branch
source_dir
latent
k
response_summary or posterior_mode
scale
control_contrast
estimate
ci_low
ci_high
p_value if available
n_images or n_trials
n_sessions if available
claim_role
known_caveat
```

### Stage 2: Fill Cache-Only Gaps

Before any new forward run, fill gaps using saved arrays.

Aggregate:

```text
rerun summarize_backimage_aggregate_incremental_motion.py on the n256 patched
run with k=2,4,8,16,32 and all available summaries.
```

Use an output like:

```text
backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
  incremental_static_plus_motion_feature_adjudication_k2_4_8_16_32_v1/
```

Local `I_z`:

If no cache-only local posthoc exists, add one rather than rerendering the twin.
It should load:

```text
latent_feature_arrays.npz
response_summary_arrays.npz
analysis_images.csv
local_pairing_motion_metadata.csv
```

and rerun the same decode/incremental logic across:

```text
k=2,4,8,16,32
summaries=delta_mean,temporal_pca,temporal_delta_pca,temporal_dct,temporal_dct_delta,mean
```

Joint posterior:

Rerun `analyze_feature_posterior.py` on compatible shared-source caches with:

```text
--pca-k-list 2,4,8,16,32
```

Use the existing feature NPZ where possible and validate row identity. Do not
use `--trust-feature-row-order` unless the feature source has no manifest and
the row identity has been audited.

Priority joint caches:

```text
hard-negative n128 scale sweep
matched-static n64 shared-source cache
hard-negative n64 shared-source cache
```

Write all new posthoc locations into:

```text
posthoc_completion_manifest.csv
```

### Stage 3: Define Stability Scores

For each feature specification:

```text
feature_spec = latent + k + response_summary + feature_space + scale policy
```

compute stability across branches.

Recommended metrics:

```text
aggregate_empirical_minus_ou
aggregate_empirical_minus_brownian_small_scale
aggregate_empirical_minus_rotated_small_scale
aggregate_empirical_minus_static
local_actual_minus_matched_unpaired
local_actual_minus_rotated
joint_parallel_minus_orthogonal
joint_minus_zero_feature_gain
joint_known_minus_joint_pose_cost_axis_delta
```

Convert each metric to standardized evidence:

```text
sign = sign(estimate)
ci_pass = CI excludes zero in expected direction
effect_z = estimate / max(SE, eps) when SE is recoverable
stability_weight = min(abs(effect_z), cap) or sign-only fallback
```

Use expected directions:

```text
aggregate: empirical > OU/Brownian/rotated/static
local: actual paired > matched unpaired and preferably > rotated
joint: edge-parallel > edge-orthogonal for feature recovery or lower pose cost
```

Because the hard-negative joint endpoint can reward across-edge discrimination,
do not require image-identity accuracy to favor edge-parallel. Feature recovery
and pose/preservation cost are more relevant to contour following.

Write:

```text
feature_spec_branch_metrics.csv
feature_spec_stability_scores.csv
```

### Stage 4: Rank Candidate Feature Specs

Suggested composite score:

```text
score =
  + aggregate empirical-vs-OU stability
  + aggregate small-scale Brownian/rotated advantage
  + local actual-vs-unpaired delta_mean stability
  + joint parallel-vs-orthogonal feature/pose signal at 0.5x/1x
  - penalty for 2x-only effects
  - penalty for sign reversals across seeds/branches
  - penalty for uninterpretable feature spaces
```

Do not let one branch dominate. Report scores both with and without the joint
axis term because the joint hard-negative candidate set has a known pressure
toward edge-orthogonal discrimination.

Write:

```text
feature_spec_ranking.csv
feature_spec_ranking_report.md
```

The report should nominate:

```text
primary_locked_feature_spec
secondary_sensitivity_feature_spec
negative_control_feature_spec
```

Example format:

```text
primary_locked_feature_spec:
  latent = pyramid_local_field
  k = 8 or 16
  response_summary = delta_mean for local / temporal_pca for aggregate
  feature_space = selected_windows_zscore_pca
  primary_scales = 0.25x, 0.5x, 1x
```

Do not fill the exact values until the adjudication is run.

### Stage 5: Lock Before Canonical Run

Once the feature spec is chosen, write:

```text
locked_feature_decomposition_spec.md
locked_feature_decomposition_spec.json
```

Required fields:

```text
latent_name
latent_crop_px
center_crop_px
local_field_grid
feature_space
pca_k
response_summary_primary
response_summary_secondary
ridge_alpha_policy
primary_scales
sentinel_scales
primary_control_contrasts
reason_for_locking
known_failure_modes
```

After this file exists, canonical large runs should not change the feature
target except through an explicit new version:

```text
locked_feature_decomposition_spec_v2.md
```

## Output Contract

Write:

```text
run_metadata.json
existing_feature_result_inventory.csv
existing_feature_result_inventory.md
posthoc_completion_manifest.csv
feature_spec_branch_metrics.csv
feature_spec_stability_scores.csv
feature_spec_ranking.csv
feature_spec_ranking_report.md
locked_feature_decomposition_spec.md
locked_feature_decomposition_spec.json
```

Recommended figures:

```text
fig_feature_spec_heatmap_by_branch.png
fig_k_stability_curves.png
fig_latent_family_comparison.png
fig_response_summary_comparison.png
fig_small_scale_vs_2x_sentinel.png
fig_top_feature_spec_evidence_panel.png
```

## Acceptance Criteria

This priority is complete when:

- existing aggregate, local `I_z`, and joint-posterior feature results are
  inventoried in one table;
- cache-only k/summaries gaps are filled or explicitly marked impossible;
- `k=2,4,8,16,32` are evaluated where cached features allow it;
- Gabor and pyramid local fields are both evaluated;
- `delta_mean` and `temporal_pca` are both evaluated in their relevant branches;
- the chosen feature spec is stable across at least two branches or has a clear
  reason to be branch-specific;
- `2x` sentinel behavior is reported and does not become the sole source of the
  claimed effect;
- a locked spec is written before any canonical large run.

## Decision Rule

Promote a feature spec for the canonical run if:

- it gives stable empirical/contour-aligned utility at small or natural scales;
- it survives the strongest relevant controls for its branch;
- its effect direction is compatible with contour following or local structure
  preservation;
- it is interpretable enough to explain in the manuscript;
- it does not rely on a single seed, single k, or single high-scale sentinel.

Keep two feature specs if:

- one spec is clearly best for aggregate distributional FEM utility and another
  is clearly best for local contour-aligned pairing;
- both are interpretable and the distinction can be stated simply.

Demote a spec if:

- the effect is only positive at `2x`;
- it wins by raw magnitude but reverses direction across seeds/branches;
- it only helps hard-negative image identity by favoring edge-orthogonal
  across-contour discrimination;
- it is too opaque to connect to natural-image contour structure.

## Practical First Pass

1. Build the result inventory from existing CSVs.
2. Rerun aggregate incremental posthoc cache-only with:

```text
--pca-k-list 2,4,8,16,32
--summaries temporal_pca,temporal_delta_pca,temporal_dct,temporal_dct_delta,mean,delta_mean
```

3. Rerun joint feature posterior on the n128 hard-negative cache with k32 added
   if feasible.
4. Add or run a local `I_z` cache-only decode posthoc for k2/k16/k32 if missing.
5. Produce `feature_spec_ranking_report.md`.
6. Only then proceed to figure pack or canonical large run.

