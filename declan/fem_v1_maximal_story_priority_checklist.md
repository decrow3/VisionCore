# FEM-V1 Maximal Story Priority Checklist

Last updated: 2026-06-20

## Purpose

This checklist turns the current FEM-V1 roadmap into a priority-ordered set of
gates for deciding whether the compact reafferent geometry plus Wu-style
trajectory observer belongs in the main manuscript or should become a separate
functional paper.

The maximal story is:

```text
recorded V1 covariance
-> model-derived reafferent geometry
-> trajectory-aware natural-image inference
-> observed FEM behavior along local image structure
```

The current recommended posture is modular: write the main paper so it stands on
recorded reafferent covariance and BackImage local image geometry, then fold in
the compact/Wu-style observer only if it explains the along-contour behavior
cleanly and compact geometry carries the functional rescue.

## Current Anchors

- [x] Main paper can stand without compact geometry as a required pillar:
  recorded FEM-linked covariance, reafference, low-dimensional structure,
  twin translation bridge, and BackImage local image geometry.
- [x] BackImage exact-cache trajectory observer is directionally positive.
- [x] Confirmatory matched-static BackImage observer run completed:
  `matched_static_response`, `n64/c8/k8`, `0.5x` and `1.0x`,
  leave-one-out trajectory priors.
- [x] Joint-eye advantage survives matched-static-response distractors.
- [x] Axis-conditioned trace utilities exist and are wired into the current
  BackImage trajectory-table runner as `axis_edge_parallel` and
  `axis_edge_orthogonal`.
- [x] First clean shared-source axis-conditioned matched-static run completed.
  It weakly favors edge-parallel over edge-orthogonal, but only by 2 trials
  (`55/64` vs `53/64`), so this is a positive pilot rather than claim-level
  evidence.
- [x] First clean shared-source axis-conditioned hard-negative replacement
  completed. Both axis priors rescue over zero-eye, but accuracy favors
  edge-orthogonal (`57/64` vs `54/64`) while paired true-score diagnostics keep
  some edge-parallel signal.
- [x] A full global-basis compact-mechanism run now exists on the completed
  matched-static observer cache.
- [x] Feature-posterior joint-decoding bridge is implemented as a cache-first
  posthoc over exact BackImage response tables.
- [x] Feature-posterior uncertainty reruns completed for the clean matched-static
  and hard-negative `n64` axis caches. Joint feature recovery beats zero-eye
  robustly, especially in hard-negative, but the axis-specific feature result is
  now claim-bounded: matched-static keeps a parallel-positive signal strongest
  for `pyramid k8`, while hard-negative trends orthogonal and has no significant
  parallel-minus-orthogonal rows.
- [x] The clean shared-source hard-negative `n128`, `c4`, `k16` scale sweep
  completed for `0.5x`, `1.0x`, and `2.0x`, and its feature-posterior
  uncertainty posthoc completed on all 768 response tables.
- [ ] Compact geometry has not yet been shown at claim level to be the mechanism
  for the current axis-conditioned feature-posterior rescue. The image-identity
  compact mechanism can be tested cache-only now, and previous image-disjoint
  matched-static compact results are positive, but static-PC controls remain
  nontrivial and compact feature-posterior variants are not yet wired.
- [ ] Axis-conditioned image-identity accuracy has not shown that edge-parallel
  or real-like trajectories preserve image identity better than edge-orthogonal
  trajectories at claim level. The feature-posterior endpoint remains useful,
  but the uncertainty reruns show it is not yet a clean along-contour mechanism
  result: matched-static trends parallel, hard-negative trends depend on scale,
  and none of the n128 parallel-minus-orthogonal feature rows are individually
  significant.
- [ ] Model-derived observer or geometry metrics have not yet beaten raw edge
  geometry as an explanation of observed drift axes.
- [x] Claim-critical diagnostics are now consolidated in
  `declan/figure4_active_sensing_atlas/claim_critical_diagnostics_queue.md`.
  Use that document as the current gatekeeping index before promoting main
  claims or treating long canonical runs as final.
- [x] Canonical production surfaces now exist:
  `declan/canonical_active_sensing/` for aggregate/local/joint/adjudication and
  active-sensing figure-pack jobs, and `declan/canonical_geometry/` for raw-edge
  residual adjudication plus geometry figure-pack jobs.
- [x] Current feature target is a two-readout candidate rather than a final
  lock: aggregate/ensemble `pyramid_local_field k16 temporal_pca`, local
  mechanistic sensitivity `pyramid_local_field k16 delta_mean`.

## Priority 0: Claim-Critical Diagnostics And Canonical Preflight

Goal:

```text
Make sure every claim-critical active-sensing result has an explicit failure
mode diagnostic before it is promoted or used to justify a long canonical run.
```

Current source of truth:

```text
declan/figure4_active_sensing_atlas/claim_critical_diagnostics_queue.md
```

Required gates:

- [ ] `canonical_active_sensing.validate_configs` and
  `canonical_geometry.validate_configs` pass before production runs.
- [ ] Use `--print-command` for every long canonical wrapper launch.
- [ ] Do not overwrite non-empty production output folders unless the refresh
  is intentional and documented.
- [ ] Keep the two-readout feature target provisional until joint `rel_0p25x`
  completion and final adjudication review are closed.
- [ ] Treat model-objective panels as diagnostic/deep-dive triggers unless they
  explain residual behavior beyond raw edge geometry on a shared window table.

Model-objective deep-dive gates:

- [ ] Build a same-window objective-vs-raw master table with raw edge,
  behavior, objective axes, confidence, border distance, candidate hardness,
  and source flags.
- [ ] Run within-session residual tests after raw edge confidence and drift
  anisotropy, reporting `Delta R2`, session-bootstrap CI, and sign count.
- [ ] Audit global/screen-axis nuisance predictors and all-zero/all-90-degree
  predicted-axis artifacts.
- [ ] Audit shared-source overlap and candidate hardness before interpreting
  any objective-axis advantage.
- [ ] Check population/readout sensitivity before mixing sampled 64/256-unit
  diagnostics with canonical 756-unit claims.

## Priority 1A: Feature-Posterior Joint-Decoding Bridge

Goal:

```text
Does trajectory marginalization preserve or recover local image features under
latent eye motion, rather than merely identify an exact image patch among
finite distractors?
```

Why this bridge matters:

- [x] Edge-parallel motion is behaviorally and geometrically plausible: real
  BackImage drift axes are edge-aligned, and pixel/V1-twin perturbation audits
  show edge-parallel motion disrupts local structure less than edge-orthogonal
  motion.
- [x] The Gabor/pyramid local and aggregate branches show that empirical motion
  can add feature-decodable signal beyond static responses, especially for
  `delta_mean` local-pairing and temporal-PCA aggregate readouts.
- [x] The exact-cache joint observer rescues image identity above zero-eye, so
  the likelihood machinery is useful.
- [x] The current image-identity endpoint may reward across-edge, high-modulation
  trajectories because those trajectories separate hard-negative image patches.
  That endpoint can disagree with the along-contour / feature-preservation
  mechanism we care about.

Primary implementation:

```text
For each trial and observer mode:
  compute image posterior over candidate patches from known/zero/joint scores
  attach feature vector z_i = phi(I_i) to every candidate image
  estimate z_hat = sum_i p(I_i | y) z_i
  score feature recovery against z_true
```

Use the same feature targets as the positive decomposition branches first:

```text
gabor_local_field
pyramid_local_field
delta_mean or static-plus-motion feature deltas
temporal_pca / temporal_dct summaries as secondary checks
```

Required outputs:

```text
feature_posterior_trials.csv
feature_posterior_summary.csv
feature_axis_contrasts.csv
feature_motion_evidence_contrasts.csv
feature_posterior_uncertainty.csv
feature_posterior_qc.csv
```

Implemented runner:

```text
declan/backimage_trajectory_observer/analyze_feature_posterior.py
```

First completed output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_matched_static_feature_posterior_gabor_pyramid_k4_8_v1/
```

Uncertainty rerun outputs:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_matched_static_feature_posterior_gabor_pyramid_k4_8_uncertainty_v2/
  backimage_axis_conditioned_hard_negative_feature_posterior_gabor_pyramid_k4_8_uncertainty_v1/
```

Configuration:

```text
base run = backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1
candidate_set_mode = matched_static_response
axis_shared_source_catalog = True
n_windows = 64
response tables = 128
latents = gabor_local_field, pyramid_local_field
k = 4, 8
likelihood_scale = 1.0
```

Initial no-uncertainty readout:

```text
joint feature recovery > zero-eye in all 8 summary rows
parallel joint feature recovery > orthogonal joint feature recovery in all
  4 feature/k contrasts
```

Uncertainty-bounded update:

```text
Matched-static parallel-minus-orthogonal joint feature recovery:
  gabor k4:    +1.19, CI [-1.72, 3.99], p=0.443
  gabor k8:    +3.43, CI [-1.03, 7.59], p=0.125
  pyramid k4:  +1.88, CI [-0.03, 3.87], p=0.066
  pyramid k8:  +2.37, CI [ 0.38, 4.62], p=0.025

Hard-negative parallel-minus-orthogonal joint feature recovery:
  gabor k4:    -2.84, CI [-7.74, 1.70], p=0.256
  gabor k8:    -0.98, CI [-5.21, 3.19], p=0.664
  pyramid k4:  -1.98, CI [-4.71, 0.35], p=0.132
  pyramid k8:  -0.75, CI [-3.18, 1.70], p=0.558
```

Interpretation:

- [x] Joint-minus-zero feature recovery is the durable feature-posterior result,
  especially in the hard-negative cache where all bootstrap CIs are positive and
  sign-flip support is present in 7/8 rows.
- [x] Matched-static keeps a parallel-positive feature-recovery hint, strongest
  for `pyramid k8`.
- [ ] The hoped-for mechanism split is not yet supported: in the hard-negative
  cache, feature recovery trends orthogonal, the same direction as image
  identity, with no significant axis contrasts.
- [ ] `motion_delta` remains a useful diagnostic but should not be interpreted
  as a standalone posterior because its zero-relative feature gain is negative
  and its axis contrasts are not significant.

Mean joint MSE reduction versus zero-eye:

```text
orthogonal gabor k4:    9.3%
orthogonal gabor k8:   14.0%
orthogonal pyramid k4: 10.1%
orthogonal pyramid k8:  8.7%
parallel gabor k4:     11.2%
parallel gabor k8:     18.4%
parallel pyramid k4:   14.8%
parallel pyramid k8:   14.3%
```

Parallel-minus-orthogonal joint feature recovery:

```text
gabor k4:    mean +1.19, median +0.37, parallel wins 57.8%
gabor k8:    mean +3.43, median +0.91, parallel wins 56.2%
pyramid k4:  mean +1.88, median +0.61, parallel wins 53.1%
pyramid k8:  mean +2.37, median +0.11, parallel wins 51.6%
```

N128 hard-negative scale-sweep update:

```text
source exact cache:
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/

feature posterior:
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k4_8_uncertainty_v1/

response tables scored = 768
axis_shared_source_catalog = True
bootstrap/permutation resamples = 10000 / 10000
current feature PCA k = 4, 8
```

K-choice note:

```text
k=4/8 was a conservative first-pass bridge to the existing feature-decomposition
branch, not a final model-selection claim. In the n128 run, PCA is fit over
selected-window feature rows, so higher k is possible but should be treated as
a feature-space sensitivity analysis.

next k sweep = 2, 4, 8, 16, 32
primary question = does the 1x parallel advantage persist in richer feature
subspaces, especially for k16/k32?
current hint = k8 shows the clearest small 1x preference so far, but only as an
axis-by-scale feature-recovery interaction rather than a broad unconditioned
absolute-MSE optimum.
```

Image-identity readout at likelihood scale `1.0`:

```text
0.5x:
  zero = 0.609
  parallel joint = 0.8125
  orthogonal joint = 0.7813
  paired discordance p = 0.523

1.0x:
  zero = 0.391
  parallel joint = 0.7969
  orthogonal joint = 0.8047
  paired discordance p = 1.000

2.0x:
  zero = 0.336
  parallel joint = 0.6797
  orthogonal joint = 0.7422
  paired discordance p = 0.229
```

N128 feature-posterior readout:

```text
joint-minus-zero feature recovery:
  positive for every axis/scale/latent/k row
  sign-flip permutation p ~= 0.0001 throughout

mean joint MSE reduction versus zero-eye:
  0.5x: 46-55%
  1.0x: 62-70%
  2.0x: 71-78%

parallel-minus-orthogonal joint feature recovery:
  0.5x: all four rows positive, all CIs include zero
  1.0x: all four rows positive, all CIs include zero
  2.0x: all four rows approximately zero or orthogonal-positive
```

N128 interpretation:

- [x] The core joint-decoding result is stronger: joint-eye rescues
  hard-negative image identity and feature recovery relative to zero-eye across
  all tested scales.
- [x] The compact tangent geometry subspace has now been tested for the
  image-identity endpoint on the n128 hard-negative scale-sweep cache. Compact
  only preserves much of the full exact rescue and compact removal collapses the
  rescue.
- [ ] The compact tangent geometry subspace has not yet been tested inside the
  feature-posterior endpoint. That is possible cache-only, but requires a new
  compact-aware score-vector bridge.
- [ ] The axis mechanism split is not claim-ready. The trend is compatible with
  parallel feature preservation at natural scales and orthogonal discrimination
  in hard negatives or above-natural motion, but current paired uncertainty does
  not make the split decisive.

N128 compact-mechanism readout:

```text
source:
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
    compact_mechanism_image_disjoint_fold0_n768_k2_5_10_20_rand8_log_v1/

basis_mode = image_disjoint
response tables = 768
image_disjoint_basis_verified = true
compact_only true-score rescue ~= 0.84-0.90 in strong rows
compact_removed ~= zero-static or worse
log_compact_removed ~= zero-static with zero clipping
random/unit-shuffle/gain controls weaker than compact_only
static_pc_k competitive with compact_only
```

Compact-mechanism interpretation:

- [x] Compact sufficiency is supported for image identity: a low-dimensional
  image-disjoint compact tangent basis carries much of the exact-cache
  joint-eye rescue.
- [x] Compact necessity is supported by compact removal and by clipping-safe
  `log_compact_removed`.
- [ ] Compact uniqueness is not established. Static-response PCs are a fair
  low-dimensional-response control but may include the compact translation
  geometry if those directions overlap high-variance static response axes.
  Treat static-PC competitiveness as a uniqueness caveat, not as a falsification
  of compact geometry.

Primary contrasts:

```text
parallel joint feature recovery - orthogonal joint feature recovery
parallel joint-minus-zero feature gain - orthogonal joint-minus-zero feature gain
parallel known-minus-joint pose cost - orthogonal known-minus-joint pose cost
parallel motion-added feature evidence - orthogonal motion-added feature evidence
```

Motion-added evidence should mirror the aggregate/local incremental logic:

```text
Delta S_i = S_joint(I_i) - S_zero(I_i)
z_hat_delta = sum_i softmax(Delta S_i) z_i
```

Interpretation gates:

- [ ] If edge-parallel wins feature recovery or has lower pose cost while
  edge-orthogonal wins hard-negative image identity, interpret this as a
  mechanism split: orthogonal motion is a discriminating probe, while parallel
  motion is feature-preserving / pose-tolerant. The uncertainty reruns do not
  yet satisfy this gate because hard-negative feature recovery trends
  orthogonal.
- [x] If edge-orthogonal also wins feature recovery, the along-contour story is
  weakened and should retreat to the pixel/twin perturbation result. The current
  hard-negative cache points in this direction, though the axis contrasts are
  not significant.
- [x] If neither axis wins but both beat zero-eye, keep the Wu-style observer as
  evidence for trajectory marginalization, not as an explanation of
  along-contour behavior. This remains the safest hard-negative interpretation
  after the n128 scale sweep, with the added nuance that parallel feature
  recovery trends are positive at `0.5x` and `1.0x` but reverse at `2.0x`.
- [x] Treat this as a posthoc on existing exact-cache tables before requesting
  new twin forward passes.

Immediate next checks:

- [x] Add bootstrap/permutation uncertainty for `joint_minus_zero_feature_gain`
  and `parallel_minus_orthogonal` feature gain.
- [x] Run the same feature-posterior endpoint on the clean hard-negative `n64`
  shared-source cache.
- [x] Apply the same feature-posterior uncertainty posthoc to the hard-negative
  `n128`, `0.5x/1x/2x` shared-source run.
- [x] Run the existing compact-mechanism posthoc on the n128 hard-negative
  scale-sweep cache to test compact-only and compact-removed image-identity
  rescue across axis and scale.
- [ ] Measure compact-vs-static-PC subspace overlap for the n128 hard-negative
  compact run.
- [ ] Residualize compact against static PCs, and static PCs against compact, to
  ask which component uniquely carries the image-identity rescue.
- [ ] Add a compact-aware feature-posterior posthoc that reuses the cached
  response tensors, applies compact-only/compact-removed/log-rate variants, and
  scores Gabor/pyramid posterior recovery without a new V1 forward run.
- [ ] Re-run the n128 feature-posterior bridge with `k = 2,4,8,16,32` to test
  whether the apparent 1x axis-by-scale interaction is concentrated in the
  dominant feature axes or survives richer Gabor/pyramid feature subspaces.
- [ ] Run the same feature-posterior endpoint on the non-axis empirical-vs-OU
  matched-static exact cache if that cache has compatible candidate metadata.
- [ ] Decide whether to add response-summary targets (`delta_mean`,
  `temporal_pca`, `temporal_dct`) after the image-feature-vector endpoint is
  summarized.

## Recent Results Check

### Compact-mechanism full global run

Output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    compact_mechanism_full_global_n512_k2_5_10_20_rand8_v1/
```

Key `matched_static_response`, `1.0x`, empirical prior, likelihood scale `1.0`
readout:

```text
zero_static:
  joint = 0.328

full_exact:
  joint = 0.766
  joint-zero = +0.438
  median N_eff/K = 0.364

compact_only:
  k=2   joint = 0.563
  k=5   joint = 0.609
  k=10  joint = 0.609
  k=20  joint = 0.641

compact_removed:
  k=2   joint = 0.313
  k=5   joint = 0.344
  k=10  joint = 0.359
  k=20  joint = 0.328

random_k:
  k=2   joint = 0.293
  k=5   joint = 0.291
  k=10  joint = 0.340
  k=20  joint = 0.348

unit_shuffle_compact:
  k=2   joint = 0.359
  k=5   joint = 0.438
  k=10  joint = 0.422
  k=20  joint = 0.453

gain_only:
  joint = 0.469

static_pc_k:
  k=2   joint = 0.531
  k=5   joint = 0.594
  k=10  joint = 0.547
  k=20  joint = 0.609
```

Interpretation:

- [x] Compact-only preserves a meaningful fraction of the exact-cache rescue.
- [x] Compact-removed collapses to near zero/static.
- [x] Random subspaces are weak.
- [x] Unit-shuffled compact and gain-only controls are much weaker than
  compact-only.
- [ ] Static PCs are still competitive at higher `k`, so the next compact
  adjudication should foreground compact-vs-static-PC and image-disjoint basis
  controls.
- [ ] This is not yet a claim-level compact mechanism result because it uses a
  global basis, not an image-disjoint basis.

### Axis-conditioned per-candidate pilots

Completed outputs:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_trajectory_observer_percandidate_gpu1_pilot32_c4_k8/
  backimage_axis_conditioned_trajectory_observer_percandidate_gpu1_pilot64_c4_k16/
```

Shared configuration:

```text
candidate_set_mode = hard_negative_structure
observation_family = empirical
prior_families = axis_edge_parallel, axis_edge_orthogonal
axis_catalog_mode = per_candidate
scale = 0.5
likelihood_scale = 1.0
```

Primary readout:

```text
n32:
  zero = 0.625
  axis_edge_parallel joint = 0.8125
  axis_edge_orthogonal joint = 0.875

n64:
  zero = 0.640625
  axis_edge_parallel joint = 0.8125
  axis_edge_orthogonal joint = 0.875
```

Interpretation:

- [x] Axis-conditioned observer machinery runs beyond dry-run.
- [x] Both axis priors rescue above zero-eye.
- [ ] The result is not an along-contour positive: edge-orthogonal is currently
  better than edge-parallel in both completed pilots, but those pilots used
  unmatched parallel/orthogonal source catalogs and should be treated as pre-fix
  diagnostics only.
- [ ] Need a completed larger axis-conditioned result and matched-static-response
  axis run with `axis_shared_source_catalog_fraction = 1.0` and source Jaccard
  `1.0` before treating this as a stable biological conclusion.

Completed larger pre-fix output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_trajectory_observer_percandidate_gpu1_target128_c4_k32/
```

```text
zero-eye = 0.617
axis_edge_parallel joint = 0.766
axis_edge_orthogonal joint = 0.875
median source Jaccard = 0.143
```

This run completed, but it also predates the shared-source fix. Treat it as
pre-fix diagnostic evidence only, not as evidence for or against edge-parallel
utility.

### Clean shared-source axis-conditioned matched-static pilot

Output:

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
scale = 0.5
n_prior_trajectories = 16
likelihood_scale = 1.0
```

Audit:

```text
axis_shared_source_catalog_fraction = 1.0
source Jaccard = 1.0
paired prior rows = 4096
parallel/orthogonal motion-stat deltas = 0
```

Primary readout:

```text
known-eye = 1.000
zero-eye  = 0.641
axis_edge_parallel joint = 55/64 = 0.859
axis_edge_orthogonal joint = 53/64 = 0.828
```

Paired posthoc:

```text
parallel-only correct = 6
orthogonal-only correct = 4
both correct = 49
both wrong = 5

median parallel-minus-orthogonal margin delta = +0.021
median parallel-minus-orthogonal true-score delta = +0.225
```

Interpretation:

- [x] The first clean shared-source matched-static axis run weakly favors
  edge-parallel.
- [x] The posthoc is directionally consistent: parallel-only correct trials have
  positive margin deltas, and drift-edge parallelism has a small positive
  correlation with parallel-minus-orthogonal true-score and margin deltas.
- [ ] The effect is small (`+2/64` trials), so it needs replication before it can
  be used as claim-level evidence.

### Clean shared-source axis-conditioned hard-negative replacement

Output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n64_c4_k16_v1/
```

Configuration:

```text
candidate_set_mode = hard_negative_structure
observation_family = empirical
prior_families = axis_edge_parallel, axis_edge_orthogonal
axis_catalog_mode = per_candidate
scale = 0.5
n_prior_trajectories = 16
likelihood_scale = 1.0
```

Audit:

```text
axis_shared_source_catalog_fraction = 1.0
source Jaccard = 1.0
paired prior rows = 4096
parallel/orthogonal motion-stat deltas = 0
```

Primary readout:

```text
known-eye = 1.000
zero-eye = 0.641
axis_edge_parallel joint = 54/64 = 0.844
axis_edge_orthogonal joint = 57/64 = 0.891
```

Paired posthoc:

```text
parallel-only correct = 3
orthogonal-only correct = 6
both correct = 51
both wrong = 4

median parallel-minus-orthogonal margin delta = -0.014
median parallel-minus-orthogonal true-score delta = +0.259
median parallel-minus-orthogonal N_eff/K delta = +0.019
```

Interpretation:

- [x] The clean hard-negative replacement passes the same-source audit and
  supersedes the old pre-fix hard-negative runs for interpretation.
- [x] Both axis-conditioned priors rescue image identity over zero-eye.
- [ ] The axis direction is mixed: accuracy favors edge-orthogonal by
  `+3/64`, while true-score and drift-edge-parallelism diagnostics retain some
  edge-parallel signal. This is not yet claim-level along-contour evidence.

### Local BackImage `I_z` pairing revisit

Current adjudication caches:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_gabor_pyramid_rel025_0p5_1_seed7_v1/
  backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_gabor_pyramid_rel2_seed7_v1/
  backimage_feature_decomposition_adjudication_v3_local_rel05_rel2_filled/
```

Configuration:

```text
n_images = 128
unpaired samples per image = 32
latents = gabor_local_field, pyramid_local_field
families = actual_paired_empirical, matched_unpaired_empirical,
           rotated_actual_90, ou_matched_actual, brownian_matched_actual
scales = 0.25x, 0.5x, 1.0x plus 2.0x sentinel
trace pool = full strict filtered pool, 3013 rows
matched controls = sampled K=32, zero same-trial matches
feature geometry = gabor (128, 4608), pyramid (128, 3072)
```

Current adjudication interpretation:

- [x] `pyramid_local_field k16 temporal_pca` is the top aggregate/ensemble
  candidate after cache-filled adjudication.
- [x] `pyramid_local_field k16 delta_mean` remains the local/mechanistic
  sensitivity readout because it better captures paired-trace
  feature-response changes.
- [ ] This is a two-readout candidate, not a final lock, until joint
  `rel_0p25x` completion and final adjudication review are closed.

Representative earlier clean-run incremental contrasts:

```text
delta_mean, gabor k=4, 0.25x:
  actual - matched_unpaired = +9.95, CI [+0.73, +20.62]

delta_mean, gabor k=4, 1x:
  actual - matched_unpaired = +8.27, CI [+2.70, +14.79]

delta_mean, gabor k=8, 0.25x:
  actual - matched_unpaired = +6.33, CI [+1.27, +11.90]

delta_mean, gabor k=8, 1x:
  actual - matched_unpaired = +6.51, CI [+2.77, +11.07]

delta_mean, pyramid k=8, 0.25x:
  actual - matched_unpaired = +6.09, CI [+1.51, +10.53]

delta_mean, pyramid k=8, 1x:
  actual - matched_unpaired = +3.79, CI [+1.46, +6.28]
```

Interpretation:

- [x] There is a clean paired-vs-matched-unpaired signal for `delta_mean` in
  both Gabor and pyramid local-field features.
- [x] The effect survives corrected feature geometry, full-pool matched
  controls, sampled K-unpaired controls, grouped-by-image decoding, and zero
  same-trial matches.
- [ ] The effect is not yet a clean temporal-code result: temporal PCA/DCT
  summaries are weak or negative.
- [ ] The effect is not yet unique-axis/local-optimum evidence: rotated actual
  controls remain competitive.
- [ ] Current safe claim: actual local fixation traces carry extra
  feature-relevant response delta beyond matched aggregate empirical FEM
  statistics.

## Raw Edge Geometry Roadblock Investigation

Potential roadblock:

```text
Model-derived observer or geometry metrics have not yet beaten raw edge
geometry as an explanation of observed drift axes.
```

Recommended primary metric:

```text
edge_alignment_index = mean cos(2 * (drift_axis - image_edge_axis))
```

This is a signed axial orientation-selectivity/alignment index:

```text
+1 = drift axis exactly edge-parallel / along-contour
 0 = no axial preference relative to local edge
-1 = drift axis exactly edge-orthogonal / across-contour
```

The unsigned circular resultant,

```text
R = |mean exp(2i * (drift_axis - image_edge_axis))|
```

is useful as a consistency/OSI-like magnitude, but it is not enough by itself
because it does not distinguish edge-parallel from edge-orthogonal preference.
For the biological claim, use the signed `edge_alignment_index` as the headline.

### Raw FEM along-contour preference size

Source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_image_structure_reviewed_v2_screenfiltered_yfix/
    backimage_image_fem_windows.csv
```

All reviewed BackImage windows:

```text
n_windows = 11749
n_sessions = 30
window mean edge_alignment_index = +0.097
weighted mean = +0.181
session mean = +0.105
session bootstrap 95% CI = [+0.067, +0.145]
positive sessions = 25 / 30
median |drift-edge delta| = 39.0 deg
fraction within 30 deg of edge-parallel = 0.402
fraction within 30 deg of edge-orthogonal = 0.296
```

Reliable-axis subset, `image_orientation_coherence >= 0.20` and
`drift anisotropy >= 0.20`:

```text
n_windows = 6242
n_sessions = 30
window mean edge_alignment_index = +0.135
weighted mean = +0.201
session mean = +0.140
session bootstrap 95% CI = [+0.090, +0.188]
positive sessions = 26 / 30
median |drift-edge delta| = 36.4 deg
fraction within 30 deg of edge-parallel = 0.433
fraction within 30 deg of edge-orthogonal = 0.286
```

High-confidence subset, `image_orientation_coherence >= 0.50` and
`drift anisotropy >= 0.50`:

```text
n_windows = 1045
n_sessions = 30
window mean edge_alignment_index = +0.289
weighted mean = +0.313
session mean = +0.269
session bootstrap 95% CI = [+0.131, +0.395]
positive sessions = 24 / 30
median |drift-edge delta| = 25.6 deg
fraction within 30 deg of edge-parallel = 0.556
fraction within 30 deg of edge-orthogonal = 0.238
```

Interpretation:

- [x] The along-contour preference is modest in the full window population.
- [x] The preference is reliable across sessions.
- [x] The preference becomes medium-sized in windows with strong image
  orientation and reliable drift anisotropy.
- [x] Distribution inspection shows the visually bimodal
  `edge_alignment_index` histogram must be read against the transformed
  uniform-angle null: `cos(2 * delta)` has endpoint-heavy bin mass even when
  `delta` is uniform. After this correction, the real excess is concentrated at
  the edge-parallel endpoint, not the edge-orthogonal endpoint.
- [ ] The strict within-session image-edge shuffle in the existing
  `orientation_alignment_summary.csv` is conservative and partly absorbs global
  axis biases. Report both absolute signed alignment versus zero and
  pair-specific residual alignment versus shuffled image-edge pairings.

Distribution inspection artifacts:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_edge_alignment_distribution_inspection/
    edge_alignment_window_and_session_distributions.png
    edge_alignment_confidence_and_signed_delta.png
    edge_alignment_endpoint_null_diagnostic.png
    endpoint_zone_enrichment_summary.csv
```

Atlas provenance:

```text
declan/figure4_active_sensing_atlas/figures/panel_E/
  E3_parallel_zone_enrichment.png
  E6_full_distribution_session_diagnostic.png
  E7_confidence_signed_delta_diagnostic.png
  E8_endpoint_null_diagnostic.png
declan/figure4_active_sensing_atlas/figures/composites/
  module_E_contour_following_diagnostics.png
```

Interpretation for figure routing:

- [x] E3 is the compact endpoint-zone summary.
- [x] E6/E7/E8 are the provenance panels that should travel with E3 when the
  behavior metric is explained: full distribution/session scatter,
  confidence/signed-delta structure, and the endpoint/null diagnostic.
- [x] E8 is especially important because it shows that `cos(2 * delta)` endpoint
  bins are endpoint-heavy even under a uniform axial-angle null; the real claim
  is the edge-parallel excess after that null is considered.

Endpoint enrichment versus a uniform axial-angle null:

```text
All windows:
  parallel <=15 deg = 21.7% observed vs 16.7% null, 1.30x expected
  orthogonal >=75 deg = 15.1% observed vs 16.7% null, 0.91x expected

Reliable axes:
  parallel <=15 deg = 23.8% observed vs 16.7% null, 1.43x expected
  orthogonal >=75 deg = 14.2% observed vs 16.7% null, 0.85x expected

High confidence:
  parallel <=15 deg = 35.4% observed vs 16.7% null, 2.12x expected
  orthogonal >=75 deg = 13.9% observed vs 16.7% null, 0.83x expected
```

### Existing model-derived axis results

Current `n=256`, `64`-unit sampled twin axis-only results:

```text
raw_edge_axis:
  session mean cos2 ~= +0.182
  positive sessions = 23 / 29

optimized_PA:
  delta vs raw edge ~= -0.190

optimized_PB / optimized_response_stability:
  delta vs raw edge ~= -0.201

optimized_pixel_isophote:
  delta vs raw edge ~= +0.018
  CI includes zero

optimized_response_refresh_lambda_0.25 / 0.5:
  delta vs raw edge ~= -0.24
```

An apparent positive:

```text
optimized_refresh_only / optimized_pixel_refresh_lambda_0:
  delta vs raw edge ~= +0.168
```

should not be treated as a solution yet. In the current candidate grid it chooses
`predicted_axis_deg = 0` for all `256` windows, so the effect is a global/screen
axis artifact or trajectory-grid artifact rather than a local image, compact,
or V1 observer explanation.

Current conclusion:

- [x] Raw local edge geometry is a real, useful baseline.
- [x] Current V1 response-stability and PA/PB/Pareto objectives do not beat raw
  edge geometry.
- [x] Pixel isophote is approximately tied with raw edge, not better.
- [ ] Any model-derived result must explain residual drift-axis variation beyond
  `edge_alignment_index`, not merely reproduce edge-parallel structure.

Next investigation:

- [ ] Use
  `declan/figure4_active_sensing_atlas/claim_critical_diagnostics_queue.md` as
  the claim-critical gate list for this branch.
- [ ] Regress `drift_edge_cos2` or signed residual alignment on raw edge
  confidence variables first: `image_orientation_coherence`, drift anisotropy,
  edge/pixel stability advantage.
- [ ] Add observer-derived variables second: axis-conditioned joint advantage,
  posterior concentration difference, compact-only sufficiency, pose-cost
  advantage.
- [ ] Use within-session demeaning or session bootstrap.
- [ ] Report incremental `Delta R2`, session-bootstrap CI, and sign count.
- [ ] Treat any all-windows/global-axis predictor as a nuisance control, not a
  biological local-image mechanism.

## Priority 1: Wu-Style Axis-Conditioned Along-Contour Audit

Goal:

```text
Does the Wu-style trajectory observer predict along-contour / edge-parallel
utility, or does it favor across-edge motion because across-edge motion is more
informative for pose localization?
```

Why this is the top gate:

- [x] Observed BackImage FEM axes show a real edge-parallel concentration,
  especially in high-confidence windows.
- [x] Pixel and V1-twin endpoint audits show edge-parallel motion disrupts local
  structure less than edge-orthogonal motion.
- [ ] The Wu-style joint-encoding objective has clean shared-source pilots, but
  has not yet explained this result at claim level. Matched-static weakly
  favors edge-parallel, while the clean hard-negative replacement favors
  edge-orthogonal in accuracy and retains only score-level edge-parallel
  diagnostics.

Primary comparison:

```text
axis_edge_parallel vs axis_edge_orthogonal
```

Use full twin forward responses first. The promoted compact tangent-map cache
currently supports cardinal `+/-x` and `+/-y` grids, not arbitrary edge-conditioned
directions, so compact projections should be secondary diagnostics after the
full-response result is interpretable.

Pre-fix hard-negative pilot readout:

```text
n32:
  zero-eye = 0.625
  joint-eye(axis_edge_parallel) = 0.8125
  joint-eye(axis_edge_orthogonal) = 0.875

n64:
  zero-eye = 0.640625
  joint-eye(axis_edge_parallel) = 0.8125
  joint-eye(axis_edge_orthogonal) = 0.875
```

Interpretation:

- [x] Axis-conditioned trace utilities implemented.
- [x] Runner supports `axis_edge_parallel` and `axis_edge_orthogonal`.
- [x] Run a dry-run or tiny smoke with `per_candidate` axis catalog mode.
- [x] Run `n=32` and `n=64` diagnostics with `hard_negative_structure`.
- [x] Both axis priors rescue above zero-eye.
- [x] Preflight catalog audit completed for the n32 and n64 hard-negative pilots,
  and rerun after fixing the audit pairing key. The catalogs are well matched in
  distribution for rendered RMS, path length, speed, duration, and clipping.
  Median clipping is `0`. Source-matched complete parallel/orthogonal pairs have
  zero metric deltas across the audited motion statistics.
- [x] The n64 audit confirms the current orthogonal advantage is not an obvious
  motion-statistic imbalance:

```text
n64 family balance:
  effective RMS median:
    parallel   = 0.026404 deg
    orthogonal = 0.026404 deg
  path length median:
    parallel   = 0.391553 deg
    orthogonal = 0.395793 deg
  speed mean median:
    parallel   = 1.204778 deg/s
    orthogonal = 1.217823 deg/s
  speed p95 median:
    parallel   = 2.994394 deg/s
    orthogonal = 3.012515 deg/s
  clipping median:
    parallel   = 0
    orthogonal = 0
```

- [x] Upstream table construction has been fixed so per-candidate
  `axis_edge_parallel` and `axis_edge_orthogonal` priors can share the same
  retained source rows per trial/candidate/scale/sample index. A dry-run smoke
  now records `axis_shared_source_catalog=True` for both families and reaches
  source Jaccard `1.0`.
- [x] Treat all pre-fix unmatched-catalog orthogonal advantages as
  diagnostic-only. Any axis-family effect from a run lacking
  `axis_shared_source_catalog=True` can reflect catalog composition rather than
  the biological/image-structure question.
- [x] First clean shared-source `matched_static_response` run completed. It has
  `axis_shared_source_catalog_fraction = 1.0`, source Jaccard `1.0`, and zero
  audited motion-stat deltas across `4096` paired prior rows.
- [x] The clean matched-static run weakly favors edge-parallel:

```text
known-eye = 1.000
zero-eye = 0.641
axis_edge_parallel joint = 0.859
axis_edge_orthogonal joint = 0.828
parallel-only correct = 6
orthogonal-only correct = 4
```

- [ ] The clean matched-static result is not yet claim-level because the
  advantage is only `+2/64` trials.
- [x] First clean shared-source `hard_negative_structure` replacement completed.
  It has `axis_shared_source_catalog_fraction = 1.0`, source Jaccard `1.0`, and
  zero audited motion-stat deltas across `4096` paired prior rows.
- [x] The clean hard-negative replacement confirms joint-eye rescue over
  zero-eye for both axis priors:

```text
known-eye = 1.000
zero-eye = 0.641
axis_edge_parallel joint = 0.844
axis_edge_orthogonal joint = 0.891
parallel-only correct = 3
orthogonal-only correct = 6
```

- [ ] The clean hard-negative replacement is mixed for the axis claim: accuracy
  favors edge-orthogonal by `+3/64`, while paired true-score delta favors
  edge-parallel. Replicate across seeds, candidate modes, and larger n before
  making a biological along-contour claim.
- [ ] The completed n32/n64 pilots below are pre-fix and remain
  distribution-matched rather than strictly same-source paired across axis
  families. Source overlap is partial:

```text
n64:
  source-matched paired prior rows              = 1056
  median parallel sources per trial/candidate   = 16
  median orthogonal sources per trial/candidate = 16
  median shared sources                         = 4
  median source Jaccard                         = 0.143

n32:
  source-matched paired prior rows              = 340
  median parallel sources per trial/candidate   = 8
  median orthogonal sources per trial/candidate = 8
  median shared sources                         = 3
  median source Jaccard                         = 0.231
```

- [x] The audit now annotates trial-level parallel-minus-orthogonal observer
  deltas with source-overlap diagnostics, handles empty/dry-run CSV outputs, and
  defaults missing shared-mode `axis_catalog_mode` metadata to `shared`. It also
  reports whether manifest rows used a shared axis source catalog.
- [ ] The completed `target128_c4_k32` hard-negative run is also pre-fix. It
  favors edge-orthogonal (`0.875` vs `0.766`) but has source Jaccard `0.143`;
  do not treat it as biological evidence.
- [x] Repeat with `matched_static_response`.
- [x] Verify matched RMS, path length, duration, time bins, speed, and clipping
  across axis families for the clean shared-source n64 runs.
- [ ] Include at least one above-natural scale sentinel in the clean
  axis-conditioned comparison. Use `2.0x` so the scale sweep has a symmetric
  half/natural/double structure, and audit effective RMS and clipping.
- [x] Run first posthoc join of observer outcomes to image features, candidate
  hardness, and drift-axis alignment for the clean matched-static pilot.
- [ ] Extend the posthoc to pixel perturbation, V1 response perturbation, and
  endpoint-zone enrichment.
- [ ] Separate observation family from prior family in all summaries.

Success pattern:

```text
known-eye high
zero-eye impaired
joint-eye(axis_edge_parallel) > joint-eye(axis_edge_orthogonal)
edge-parallel has lower pose cost or better image preservation
posterior concentration or margin diagnostics explain trial-level gains
axis utility predicts observed drift alignment beyond raw edge geometry
```

Critical baseline:

- [ ] Raw edge orientation must be the baseline to beat.
- [ ] Pixel edge-parallel stability must be included as a non-neural baseline.
- [ ] A model-derived objective only earns main-paper status if it explains
  residual behavior beyond raw edge geometry or identifies image-dependent
  cases where edge-parallel motion should help most.
- [ ] If future clean shared-source replications favor edge-orthogonal over
  edge-parallel, inspect whether the observer objective is measuring useful pose
  localization/modulation rather than safe identity preservation.
- [ ] Add an explicit tradeoff metric if needed:

```text
joint_eye_gain - lambda * pose_or_response_disruption
```

## Priority 2: Compact Mechanism On The Completed Matched-Static Run

Goal:

```text
Does compact reafferent geometry carry the exact-cache joint-eye rescue?
```

Use the completed run:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
```

Required variants:

- [x] `full_exact`
- [x] `zero_static`
- [x] `compact_only`
- [x] `compact_removed`
- [x] `random_k`
- [x] `unit_shuffle_compact`
- [x] `gain_only`
- [x] `static_pc_k`

Claim-level requirements:

- [x] Run all matched-static tables, not only the two-table smoke.
- [ ] Use enough random nulls for stable summaries. The current full run used
  `rand8`, which is useful but still light for final null quantification.
- [ ] Prefer an image-disjoint compact basis.
- [x] Clearly label the current result as `basis_mode=global`.
- [x] Audit negative-rate clipping and report it by variant.
- [x] Report accuracy and true-score rescue, not only one metric.
- [x] Report posterior concentration preservation in `compact_only`.

Success pattern:

```text
compact_only preserves much of full_exact joint rescue
compact_removed loses much of the rescue
compact_only beats random/gain/static-PC controls
posterior concentration is preserved in compact_only
```

Decision:

- [x] Current global-basis result makes compact geometry a plausible mechanism
  for the exact-cache observer.
- [ ] If negative, keep the exact-cache observer as a functional result but do
  not claim compact geometry explains the rescue.
- [ ] Promote to claim-level mechanism only after image-disjoint/static-PC
  adjudication.

## Priority 3: Harder Exact-Cache Observer Confirmation

Goal:

```text
Does trajectory marginalization remain useful after further scale-up?
```

Current completed result:

```text
n_images = 64
n_candidates = 8
K = 8
candidate modes = hard_negative_structure, matched_static_response
scales = 0.5x, 1.0x
priors = empirical, OU
trajectory_prior_mode = leave_one_out
```

Newest completed axis-conditioned hard-negative scale sweep:

```text
n_images = 128
n_candidates = 4
K = 16
candidate mode = hard_negative_structure
scales = 0.5x, 1.0x, 2.0x
priors = axis_edge_parallel, axis_edge_orthogonal
trajectory_prior_mode = leave_one_out
axis_shared_source_catalog = True
```

Next run targets:

- [x] `n_images = 128`
- [ ] `n_candidates = 8` or `16`
- [x] `K = 16` or `32`
- [ ] `candidate modes = matched_static_response, hard_negative_structure`
- [ ] `priors = empirical, OU, Brownian, shuffled-position, rotated`
- [x] `scales = 0.5x, 1.0x, 2.0x`
- [ ] Audit the `2.0x` sentinel for clipping, effective RMS, and any runaway
  monotonic more-motion-is-better pattern. The endpoint trend does not show a
  simple identity-accuracy improvement at `2.0x`, but the full motion-quality
  audit is still pending.
- [ ] `likelihood_scale = 0.5, 1.0`

Success pattern:

```text
known-eye high
zero-eye impaired at larger motion
joint-eye > zero-eye
joint advantage associated with lower N_eff / K
nearest-trajectory rank better than chance
effect survives matched_static_response
above-1x sentinel does not turn the result into a trivial more-motion-is-better curve
```

Claim boundary:

- [ ] Treat this as pose-uncertainty rescue, not active-sensing optimality.
- [ ] Do not claim empirical FEM priors are uniquely optimal unless they beat
  OU/Brownian/rotated/shuffled controls consistently.
- [ ] If accuracy or score gain keeps improving monotonically above `1.0x`
  without image-structure or prior-family specificity, frame the result as a
  scale/pose-uncertainty effect rather than an FEM-optimality result.

## Priority 4: Static / Stabilized Benefit Comparison

Goal:

```text
Does moving joint-eye performance exceed an appropriate static or stabilized
observer, or does joint-eye mainly rescue damage caused by wrong pose?
```

Compare:

- [ ] static/stabilized observer
- [ ] moving known-eye
- [ ] moving joint-eye
- [ ] moving zero-eye

Interpretation:

```text
moving joint > static:
  active-sensing benefit

moving joint ~= static and zero-eye << static:
  motion is tolerable if modeled, but not necessarily beneficial

moving known > static and joint ~= static:
  motion creates useful samples, but pose uncertainty consumes the benefit
```

Keep this separate from the joint-eye rescue claim.

## Priority 5: Noise-Model Robustness

Goal:

```text
Are observer conclusions robust to reasonable observation/noise assumptions?
```

Checklist:

- [ ] Likelihood-scale grid beyond the current primary values if needed.
- [ ] Sampled spike-count observations.
- [ ] Diagonal Poisson expected-count likelihood.
- [ ] Overdispersed Poisson or negative-binomial-like sensitivity.
- [ ] Diagonal Gaussian approximation.
- [ ] Empirical residual covariance or low-rank residual perturbation where
  available.
- [ ] Low-rate unit dominance diagnostics.
- [ ] Unit subset robustness.

Claim boundary:

- [ ] If results require a narrow likelihood scale, call them fragile.
- [ ] If results survive sampled spikes and broad likelihood scales, promote
  them as robust observer evidence.

## Branch-Specific Motion-Prior Interpretation

Keep these two statements separate:

- Exact-cache trajectory observer:
  empirical priors do not clearly beat OU priors yet. The strong current claim
  is trajectory marginalization under natural-image structure, not empirical
  FEM optimality.
- Aggregate BackImage feature-decoding branch:
  empirical drift-like motion currently beats OU-like confined motion in the
  cleaned `n=256`, `K=4` temporal-PCA result, especially at `0.25x-0.5x`. This
  supports a distributional, readout- and scale-dependent claim, not exact
  trajectory-order optimality.

## Main-Paper Decision Rule

Include the compact/Wu-style module in the main paper only if all are true:

- [ ] BackImage joint-eye rescue survives hard controls.
- [ ] Compact-only preserves the joint-eye rescue and compact-removed weakens it.
- [ ] Axis-conditioned observer predicts edge-parallel or real-like utility.
- [ ] Observer-derived or compact-derived metrics add explanatory power beyond
  raw edge geometry.
- [ ] The result can be explained in one concise Results section.

Keep it separate if any are true:

- [ ] The observer result is robust but does not explain along-contour behavior
  beyond raw edge geometry.
- [ ] The compact mechanism is positive but too technically distinct from the
  recorded covariance/reafference manuscript.
- [ ] The main paper becomes clearer when focused on recorded shared
  variability, reafference, and natural-image local geometry.

Demote it if any are true:

- [ ] Joint-eye rescue collapses under harder matched-static or larger-catalog
  controls.
- [ ] Posterior concentration remains diffuse or unrelated to score gain.
- [ ] Compact-only fails to preserve the exact-cache rescue.
- [ ] Clean shared-source edge-orthogonal trajectories outperform edge-parallel
  trajectories in the preservation or robustness metrics.
- [ ] Model-derived utility fails to beat raw edge geometry and does not explain
  residual behavior.

## Safest Current Wording

```text
FEM-linked V1 variability is structured reafference rather than arbitrary
internal noise. In natural images, trajectory-aware observers can recover image
identity lost by a pose-blind zero-eye observer, and this recovery is linked to
partial trajectory-posterior concentration. Real BackImage drift is aligned with
local oriented image structure, and edge-parallel motion preserves local pixels
and V1-twin responses better than edge-orthogonal motion. A first full
global-basis compact projection suggests that the compact translation channel
can carry much of the exact-cache observer rescue, but claim-level mechanism
status still requires image-disjoint/static-PC adjudication. Clean
shared-source axis-conditioned pilots show robust joint-eye rescue over
zero-eye, but the edge-parallel versus edge-orthogonal direction is not yet
stable: matched-static weakly favors edge-parallel, while hard-negative favors
edge-orthogonal in accuracy and retains only score-level edge-parallel
diagnostics. The axis-conditioned observer is therefore promising but still
needs larger shared-source replication before it can explain along-contour
behavior at claim level.
```
