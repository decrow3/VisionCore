# Raw Edge Roadblock Handoff

Last updated: 2026-06-20.

## Purpose

This handoff turns the current top FEM-V1 active-sensing roadblock into a
coding-agent plan.

The roadblock is:

```text
Observed BackImage drift axes are reliably biased toward local edge-parallel
geometry, but current model-derived observer or V1-twin objective metrics have
not yet beaten raw edge geometry as an explanation of those axes.
```

The next coding task is not another broad pathfinder. It is a cache-first
adjudication:

```text
Do model-derived variables explain residual drift-axis variation after raw edge
geometry and simple image-confidence variables have already been accounted for?
```

If yes, the model branch can earn mechanistic status. If no, the main story
should keep the local BackImage result as raw image geometry plus
local-preservation, and keep the joint-posterior observer as trajectory-aware
feature recovery rather than an explanation of along-contour behavior.

## Current Scientific Boundary

Safe claim:

```text
Real BackImage drift is modestly but reliably edge-parallel. Edge-parallel
endpoint perturbations preserve local pixels and V1-twin responses better than
edge-orthogonal perturbations. Trajectory-aware observers recover identity and
features lost by a zero-eye observer.
```

Not yet safe:

```text
The V1 twin explains why biological drift chooses those axes beyond raw local
edge geometry.
```

Do not promote:

- exact trajectory optimality;
- unsigned circular resultant alone as the biological axis metric;
- pre-fix unmatched-catalog orthogonal advantages;
- any global all-window axis predictor as a local image mechanism;
- a model objective that only reproduces edge-parallel structure without
  explaining residuals beyond raw edge alignment.

## Source Notes To Read First

Read these before editing code:

```text
declan/fem_v1_maximal_story_priority_checklist.md
  sections:
    Current Anchors
    Raw Edge Geometry Roadblock Investigation
    Priority 1: Wu-Style Axis-Conditioned Along-Contour Audit
    Main-Paper Decision Rule

declan/active_sensing_roadmap_after_vernier_fixation_image_structure.md
  sections:
    Core Update
    Local Image Geometry, Not Scalar Feature Magnitude
    Conditional Fixation Objective
    Practical Next Steps

declan/ANALYSIS_NARRATIVE.md
  sections:
    Current Synthesis
    Practical next gates under the 2026-06-13 active-sensing roadmap
```

Note: the checklist was edited before the later `k16` feature-posterior
posthoc finished. That newer output reinforces the checklist's caution:
joint-minus-zero feature recovery is strong, but parallel-minus-orthogonal
feature recovery remains non-significant.

## Model Groupings

Keep the three active model families separate.

| Family | Current scale | Axis status | Use for this roadblock |
| --- | --- | --- | --- |
| aggregate FEM | `256` images, motion families x scales x samples/image | no explicit axes | Context only. It supports distributional empirical-drift utility, but it does not solve raw edge axis prediction. |
| local `I_z` pairing | `128` images, actual paired trace plus matched unpaired/rotated/OU/Brownian controls | axes only in edge-control runs | Secondary. Use edge-control and paired-vs-unpaired deltas as local image-contingent support if compatible window metadata exists. |
| joint posterior | `128` windows, `16` prior trajectories per candidate/prior/scale | yes, edge-parallel vs edge-orthogonal per candidate | Primary cache-first route for observer-derived residual predictors. |

The raw-edge roadblock itself is cross-cutting. The headline target is always
the signed drift-edge alignment:

```text
drift_edge_cos2 = cos(2 * (drift_axis - image_edge_axis))
```

Use this signed axial metric because it distinguishes edge-parallel from
edge-orthogonal preference.

## Primary Inputs

### Raw Edge Baseline

Use:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_image_structure_reviewed_v2_screenfiltered_yfix/
    backimage_image_fem_windows.csv
```

Important columns:

```text
session
trial_idx
global_start
global_stop
local_start
local_stop
duration_s
rms_radius_deg
anisotropy
path_length_deg
image_patch_center_x_px
image_patch_center_y_px
image_patch_distance_to_image_border_px
image_patch_rms_contrast
image_gradient_energy
image_edge_density
image_orientation_coherence
image_gradient_axis_deg
image_edge_axis_deg
image_spectrum_anisotropy
image_high_freq_power_fraction
drift_orientation_deg
drift_edge_delta_deg
drift_edge_cos2
```

Known baseline sizes from the checklist:

```text
all windows:
  n_windows = 11749
  n_sessions = 30
  session mean drift_edge_cos2 ~= +0.105
  bootstrap CI ~= [+0.067, +0.145]

reliable-axis subset:
  image_orientation_coherence >= 0.20
  drift anisotropy >= 0.20
  session mean ~= +0.140

high-confidence subset:
  image_orientation_coherence >= 0.50
  drift anisotropy >= 0.50
  session mean ~= +0.269
```

### Pixel And Twin Preservation Baseline

Use:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_twin_stability_metric_audit/
    twin_stability_metric_by_window.csv

outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_twin_stability_metric_audit/
    endpoint_feature_preservation_static_decoder/
      feature_preservation_by_window.csv
```

Important columns:

```text
window_row
window_id
session_id
image_id
edge_axis_deg
real_drift_axis_deg
drift_edge_align_signed
image_orientation_coherence
drift_anisotropy
pixel_stability_advantage
twin_stability_advantage
raw_mse_stability_advantage
response_norm_mse_stability_advantage
per_rate_mse_stability_advantage
diag_whitened_mse_stability_advantage
full_cov_whitened_mse_stability_advantage
edge_parallel_preservation_minus_orthogonal
```

Treat these as image-geometry and preservation baselines, not as a successful
model-specific explanation by themselves. They are the nearest non-neural and
V1-twin local-preservation controls that any observer-derived metric must beat.

### Joint Posterior Axis Cache

Use the completed shared-source n128 hard-negative scale sweep:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
    observer_trials.csv
    observer_summary.csv
    motion_catalog.csv
    axis_trajectory_catalog.csv
    response_cache_manifest.csv
```

Important `observer_trials.csv` columns:

```text
trial_id
observation_source_row
candidate_set_mode
observation_scale
prior_family
prior_scale
axis_shared_source_catalog
likelihood_scale
n_candidates
n_trajectories
posterior_N_eff_true_image
N_eff_true_image_fraction
posterior_entropy_true_image
max_tau_posterior_true_image
joint_minus_zero_true_score
known_minus_zero_true_score
joint_correct
joint_true_rank
joint_true_margin
joint_true_score
zero_correct
zero_true_score
known_correct
known_true_score
```

Only compare `axis_edge_parallel` and `axis_edge_orthogonal` rows where
`axis_shared_source_catalog` is true. The source catalog fix is mandatory.

### Joint Feature Posterior

Use the latest k-expanded posthoc:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k4_8_16_uncertainty_v1/
    feature_posterior_trials.csv
    feature_posterior_summary.csv
    feature_axis_contrasts.csv
    feature_motion_evidence_contrasts.csv
    feature_posterior_uncertainty.csv
    feature_posterior_qc.csv
```

Current result to preserve in the report:

```text
joint-minus-zero feature recovery is positive across axis, scale, latent, and k.
parallel-minus-orthogonal feature recovery remains non-significant.
At 0.5x and 1.0x the mean axis contrast is often parallel-positive.
At 2.0x it trends zero or orthogonal-positive, especially for richer k.
```

This means the feature posterior is a good source of observer-derived
predictors, but not a completed along-contour mechanism.

### Local `I_z` Pairing

Use only as a secondary branch after the primary raw-edge residual table exists:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_gabor_pyramid_rel025_1_seed7_v1/
```

The safe local claim is:

```text
actual image-trace pairings beat matched unpaired empirical trace swaps for
delta_mean feature-response gains in both Gabor and pyramid local fields.
```

This does not yet prove a unique axis/local optimum because rotated actual
controls remain competitive and temporal PCA/DCT summaries are weak.

## Proposed New Script

Create:

```text
declan/fixation_statistics_by_stimulus/analyze_backimage_raw_edge_roadblock.py
```

Default output root:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_raw_edge_roadblock_residual_adjudication_v1/
```

Recommended CLI flags:

```text
--windows-csv
--stability-window-csv
--feature-preservation-window-csv
--observer-trials-csv
--feature-posterior-trials-csv
--feature-axis-contrasts-csv
--out-dir
--primary-scale 1.0
--include-scales 0.5,1.0,2.0
--primary-likelihood-scale 1.0
--session-bootstrap-n 10000
--random-seed 0
```

Keep implementation cache-first. Do not trigger new V1 forward passes.

## Implementation Plan

### Stage 1: Build A Window-Level Raw-Edge Table

Start from `backimage_image_fem_windows.csv`.

Add a stable row id:

```text
raw_window_row = file row index
raw_window_key = session, trial_idx, global_start, global_stop, local_start,
                 local_stop, image_patch_center_x_px, image_patch_center_y_px
```

Do not rely only on row order when joining external tables. Use available keys
such as session, trial index, global/local start/stop, and patch center. If an
external table only has `window_row`, verify that row indices match by auditing
at least session, trial, edge axis, and drift-edge alignment.

Primary target:

```text
y = drift_edge_cos2
```

Baseline raw-image predictors:

```text
image_orientation_coherence
image_edge_density
image_patch_rms_contrast
image_gradient_energy
image_spectrum_anisotropy
image_high_freq_power_fraction
image_patch_distance_to_image_border_px
rms_radius_deg
anisotropy
duration_s
path_length_deg
phase
```

Derived binary subsets:

```text
reliable_axis = image_orientation_coherence >= 0.20 and anisotropy >= 0.20
high_confidence = image_orientation_coherence >= 0.50 and anisotropy >= 0.50
```

Write:

```text
joined_raw_edge_baseline_table.csv
join_qc.csv
join_qc.md
```

### Stage 2: Add Preservation Predictors

Join the stability outputs at window level.

Candidate preservation predictors:

```text
pixel_stability_advantage
twin_stability_advantage
raw_mse_stability_advantage
response_norm_mse_stability_advantage
per_rate_mse_stability_advantage
diag_whitened_mse_stability_advantage
full_cov_whitened_mse_stability_advantage
edge_parallel_preservation_minus_orthogonal
```

Treat these as the first model-like block after raw edge confidence variables.
They ask whether windows where edge-parallel motion is especially preserving
are also the windows where real drift is more edge-parallel.

Write:

```text
joined_preservation_table.csv
preservation_predictor_correlations.csv
```

### Stage 3: Add Joint-Posterior Axis Predictors

From `observer_trials.csv`, build paired parallel-minus-orthogonal deltas within
each:

```text
trial_id
observation_source_row
observation_scale
likelihood_scale
candidate_set_mode
```

Only include rows where:

```text
prior_family in {axis_edge_parallel, axis_edge_orthogonal}
axis_shared_source_catalog == true
```

Candidate observer predictors:

```text
joint_correct_parallel_minus_orthogonal
joint_true_score_parallel_minus_orthogonal
joint_true_margin_parallel_minus_orthogonal
joint_minus_zero_true_score_parallel_minus_orthogonal
N_eff_true_image_fraction_parallel_minus_orthogonal
posterior_entropy_true_image_parallel_minus_orthogonal
max_tau_posterior_true_image_parallel_minus_orthogonal
known_minus_joint_pose_cost_parallel_minus_orthogonal
```

Define:

```text
known_minus_joint_pose_cost = known_true_score - joint_true_score
```

Interpretation:

- positive joint score or feature gain delta means edge-parallel prior helps;
- lower pose cost for edge-parallel may be useful even if hard-negative image
  identity accuracy favors edge-orthogonal;
- posterior concentration should not be treated as good by itself unless it is
  associated with true-score or feature recovery.

Write:

```text
observer_axis_delta_by_window.csv
observer_axis_delta_qc.csv
```

### Stage 4: Add Feature-Posterior Axis Predictors

From `feature_posterior_trials.csv`, build paired parallel-minus-orthogonal
deltas by:

```text
observation_source_row or trial_id
observation_scale
latent
requested_k
likelihood_scale
```

Primary feature-posterior predictors:

```text
joint_minus_zero_feature_gain_parallel_minus_orthogonal
joint_feature_recovery_parallel_minus_orthogonal
known_minus_joint_pose_cost_parallel_minus_orthogonal
motion_delta_minus_zero_feature_gain_parallel_minus_orthogonal
```

Use `motion_delta` only as a diagnostic. It is a contrast diagnostic
constructed from `joint - zero`, not an independent generative posterior.

Primary feature rows for first pass:

```text
scale = 1.0
latent in {gabor_local_field, pyramid_local_field}
k in {8, 16}
```

Then repeat sensitivity:

```text
scale in {0.5, 1.0, 2.0}
k in {4, 8, 16}
```

Write:

```text
feature_posterior_axis_delta_by_window.csv
feature_posterior_axis_delta_qc.csv
```

### Stage 5: Residual Adjudication

Use two complementary analyses.

First, session-demeaned regression:

```text
y = drift_edge_cos2
standardize predictors within session
include session demeaning or session fixed effects
```

Model blocks:

```text
M0: intercept/session only
M1: raw image confidence + FEM reliability variables
M2: M1 + pixel/twin preservation variables
M3: M2 + joint-posterior axis variables
M4: M3 + feature-posterior axis variables
```

Second, residual regression:

```text
fit M1
residual_y = y - yhat_M1
test preservation and observer variables against residual_y
```

Report for each block:

```text
in-sample R2
grouped/session cross-validated R2 if practical
incremental Delta R2 versus previous block
session bootstrap CI for Delta R2
session sign count for predictor direction
standardized coefficient and CI
Spearman rho and CI as a nonparametric check
```

Bootstrap unit should be session, not window.

Primary success statistic:

```text
Delta R2(M3 or M4 over M1/M2) > 0 with session-bootstrap CI excluding zero
```

Stratify:

```text
all reviewed BackImage windows
reliable_axis subset
high_confidence subset
scale 0.5x / 1.0x / 2.0x for observer predictors
```

### Stage 6: Robustness And Negative Controls

Required controls:

- within-session demeaning or session bootstrap;
- nuisance control for global/screen-axis predictors;
- compare against raw `image_edge_axis_deg` confidence variables first;
- exclude or flag windows near image borders;
- report whether effects survive high-confidence subset;
- confirm no result depends on pre-fix unmatched axis catalogs;
- confirm observer predictors are not just candidate hardness:

```text
static_response_distance_to_nearest_distractor
mean_rate_distance_to_nearest_distractor
contrast_distance_to_nearest_distractor
n_matched_distractors
random_fallback_used
```

If candidate-hardness metadata is only available in `observer_trials.csv`, add
it as a nuisance block in observer-only regressions.

## Output Contract

Write these files:

```text
run_metadata.json
joined_raw_edge_baseline_table.csv
joined_preservation_table.csv
observer_axis_delta_by_window.csv
feature_posterior_axis_delta_by_window.csv
raw_edge_residual_master_table.csv
join_qc.csv
join_qc.md
predictor_dictionary.csv
model_block_summary.csv
incremental_r2_session_bootstrap.csv
standardized_coefficients.csv
spearman_predictor_summary.csv
stratified_model_summary.csv
raw_edge_roadblock_report.md
```

Recommended figures:

```text
fig_raw_edge_alignment_by_confidence.png
fig_preservation_predicts_residual_alignment.png
fig_observer_axis_delta_predicts_residual_alignment.png
fig_incremental_r2_by_block.png
fig_session_delta_r2_signs.png
```

## Report Template

The final report should answer these questions in order:

1. Are the joins clean enough to interpret?
2. Does raw edge confidence reproduce the known signed alignment result?
3. Do pixel/twin preservation variables explain additional drift-axis residuals?
4. Do joint-posterior axis variables explain residuals beyond raw edge and
   preservation?
5. Do feature-posterior axis variables add anything beyond identity/posterior
   variables?
6. Are results stable across all, reliable-axis, and high-confidence windows?
7. Is the conclusion main-paper mechanism, separate observer module, or demote?

Use this conclusion language:

```text
If positive:
  Model-derived observer/preservation variables explain session-robust residual
  variation in observed drift-edge alignment beyond raw local edge geometry.
  This supports a mechanistic bridge between BackImage along-contour drift and
  trajectory-aware V1-twin sampling.

If negative:
  Raw edge geometry and local preservation remain the best explanation of
  observed BackImage drift axes. The joint-posterior observer is still evidence
  for trajectory-aware feature recovery, but not yet an explanation of the
  biological along-contour drift bias.
```

## Promotion Gates

Promote the raw-edge roadblock branch only if all are true:

- model-derived predictors add positive `Delta R2` beyond raw edge confidence
  and preservation baselines;
- session-bootstrap CI for the key `Delta R2` excludes zero or is very clearly
  one-sided with strong session sign support;
- effect survives reliable-axis or high-confidence subsets;
- effect is not explained by candidate hardness, border distance, or global
  screen-axis artifacts;
- direction is biologically interpretable, for example edge-parallel observer
  advantage or lower pose cost predicts more edge-parallel real drift.

Keep as supportive but not mechanistic if:

- preservation variables explain residuals, but joint-posterior variables do
  not;
- observer variables are directionally positive but fragile across scale/k;
- only aggregate FEM or local `I_z` positives remain, with no axis-residual
  prediction.

Demote as a mechanism if:

- raw edge confidence variables absorb the effect;
- observer variables fail to add residual explanation;
- edge-orthogonal advantages dominate the preservation or robustness metrics;
- any apparent predictor is a global-axis or candidate-hardness artifact.

## Practical First Pass

A good first coding pass is:

1. Build `raw_edge_residual_master_table.csv` for the intersection of
   `backimage_image_fem_windows.csv` and `twin_stability_metric_by_window.csv`.
2. Reproduce known signed alignment summaries for all/reliable/high-confidence
   subsets.
3. Test whether `pixel_stability_advantage` and
   `full_cov_whitened_mse_stability_advantage` predict residual
   `drift_edge_cos2` beyond image orientation coherence and drift anisotropy.
4. Add the n128 joint-posterior scale `1.0` parallel-minus-orthogonal deltas.
5. Add feature-posterior `gabor k8/k16` and `pyramid k8/k16` deltas.
6. Write the report before adding local `I_z` or compact variants.

This first pass should be possible without new GPU work.

