# Two-Readout Feature Lock And Canonical Run Preflight

Last updated: 2026-06-20.

## Purpose

This note is the landing pad for the feature-decomposition closure pass before
the next canonical BackImage active-sensing figure run.

The working candidate is now a two-readout spec:

```text
primary aggregate / ensemble readout:
  latent = pyramid_local_field
  k = 16
  response_summary = temporal_pca

local mechanistic sensitivity readout:
  latent = pyramid_local_field
  k = 16
  response_summary = delta_mean
```

This should not become a final lock until the joint `rel_0p25x` completion run
lands and the final adjudication has zero missing cache cells.

## Active Completion Run

The only remaining v3 gap was joint posterior `rel_0p25x`.

Observer run:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scale_0p25_v1/
```

Observer log:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scale_0p25_v1/
    background_logs/observer_rel0p25.log
```

Detached watcher log:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scale_0p25_v1/
    background_logs/joint_rel0p25_posthoc_and_adjudication.log
```

The watcher should produce:

```text
joint rel0.25 posterior:
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_n128_rel0p25_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/

final adjudication:
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_feature_decomposition_adjudication_v4_joint_rel0p25_complete/
```

## Current Baseline Before v4

Latest completed adjudication with local `rel_0p5x` and local `rel_2x` filled:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_feature_decomposition_adjudication_v3_local_rel05_rel2_filled/
```

Top candidate:

```text
pyramid_local_field, k=16, temporal_pca
score_with_joint_axis_term = 3.0906
score_without_joint_axis_term = 2.5656
aggregate_score = 2.7386
local_Iz_score = 0.0770
joint_axis_score = 0.5250
joint_generic_score = 3.0
```

Local sensitivity candidate:

```text
pyramid_local_field, k=16, delta_mean
rank = 6
score_with_joint_axis_term = 2.4716
score_without_joint_axis_term = 1.9466
aggregate_score = 1.1364
local_Iz_score = 1.8102
joint_axis_score = 0.5250
sign_reversal_penalty = 1.0
```

Remaining v3 missing cache cells:

```text
10 total, all joint_posterior rel_0p25x
```

## v4 Acceptance Criteria

Treat the feature-decomposition search as closed if all of the following hold
in `backimage_feature_decomposition_adjudication_v4_joint_rel0p25_complete/`:

```text
posthoc_completion_manifest.csv:
  missing_cache_only_gap_count == 0

feature_spec_ranking.csv:
  top row is pyramid_local_field, k=16, temporal_pca
  pyramid_local_field, k=16, delta_mean remains the strongest local-Iz readout
  no new sentinel/sign-reversal penalty flips the interpretation

provisional_feature_decomposition_spec.md:
  canonical_run_allowed is true only if the script was invoked with lock/write-lock
  otherwise manually record "two-readout candidate, cache-complete"
```

If v4 changes the top row but preserves the same two-readout structure, inspect
the score deltas before reopening the full search. If v4 makes joint-axis
evidence strongly negative or changes the preferred latent/k, pause before any
canonical run.

## Canonical Run Command Scaffold

The canonical aggregate run should be launched only after v4 is reviewed. The
intended run is not another feature search; it is a production run around the
candidate feature target.

Recommended output root:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_pyramid_k16_tworeadout_rel025-2_canonical_v1/
```

Full response/decode run scaffold:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python -m declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information \
  --input outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv \
  --out-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_pyramid_k16_tworeadout_rel025-2_canonical_v1 \
  --max-images 256 \
  --trace-samples-per-condition 4 \
  --motion-families empirical,ou,brownian,rotated \
  --observed-rms-scales 0.25,0.5,1.0,1.5,2.0 \
  --patch-size-px 540 \
  --min-patch-image-margin-px 270 \
  --latent-names pyramid_local_field \
  --pca-k-list 16 \
  --latent-crop-px 151 \
  --center-crop-px 41 \
  --local-field-grid 8 \
  --n-timepoints 40 \
  --temporal-pc-components 4 \
  --fixed-ridge-alpha 10.0 \
  --decode-group-mode image \
  --outer-folds 5 \
  --inner-folds 3 \
  --n-bootstrap 10000 \
  --reliable-image-coherence-min 0.20 \
  --reliable-drift-anisotropy-min 0.20 \
  --min-duration-s 0.10 \
  --max-rms-deg 0.12 \
  --max-trace-source-rms-deg 0.06 \
  --max-trace-source-radius-deg 0.2 \
  --max-trace-source-speed-p95-deg-s 20.0 \
  --max-trace-source-microsaccade-events 0 \
  --max-rendered-trace-path-length-deg 1.5 \
  --reuse-trace-sources-across-scales \
  --twin-batch-size 48 \
  --twin-trace-batch-size 2 \
  --device cuda:0 \
  --seed 0 \
  --progress-every 4
```

Required cache-only incremental posthoc after the full run:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python -m declan.fixation_statistics_by_stimulus.summarize_backimage_aggregate_incremental_motion \
  --run-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_pyramid_k16_tworeadout_rel025-2_canonical_v1 \
  --out-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_pyramid_k16_tworeadout_rel025-2_canonical_v1/incremental_static_plus_motion_tworeadout_v1 \
  --summaries temporal_pca,delta_mean \
  --families empirical,ou,brownian,rotated \
  --contrast-pairs empirical:ou,empirical:brownian,empirical:rotated,empirical:static \
  --scale-ids all \
  --latent-names pyramid_local_field \
  --pca-k-list 16 \
  --fixed-ridge-alpha 10.0 \
  --outer-folds 5 \
  --inner-folds 3 \
  --decode-group-mode image \
  --n-bootstrap 10000 \
  --seed 0
```

Notes:

- `temporal_pca` is the canonical aggregate readout.
- `delta_mean` is included as a sensitivity readout, not as the main aggregate
  score.
- The current adjudication used existing aggregate/local/joint caches; this
  command is the production aggregate rerun around the selected target.
- If we want a visible Gabor comparison in figures, add `gabor_local_field` to
  `--latent-names`, but do not let that reopen the feature-target decision.

## Figure-Code Audit

### Production-nearest aggregate figure pack

```text
declan/fixation_statistics_by_stimulus/make_backimage_aggregate_fem_figure_pack.py
```

Status:

- Uses the correct n256 patched aggregate run by default.
- Uses `incremental_static_plus_motion_relids/`, not the stale folder.
- Now accepts a configured incremental directory, expected scales, primary
  summary, primary gain rows, and primary contrast latent/k.
- The canonical wrapper config points the figure pack at
  `pyramid_local_field k16 temporal_pca`.
- The default remains backward-compatible with the current n256 patched cached
  figure pack.

### Figure 4 atlas panel B

```text
declan/figure4_active_sensing_atlas/scripts/plot_panel_b_subpanels.py
```

Status:

- Reads the current n256 aggregate/incremental folders.
- Hardcodes `motion_summary == "temporal_pca"`.
- Hardcodes visual rows to `gabor_local_field k=4` and
  `pyramid_local_field k=8`.
- Should be refactored to accept the same primary latent/k/summary arguments
  as the aggregate figure pack.

### Older collaborator figure script

```text
declan/fixation_statistics_by_stimulus/make_backimage_active_sensing_collab_figures.py
```

Status:

- Useful for provenance and old panel logic only.
- Points at older n128/pathfinder outputs in several places.
- Do not use as the production figure source without rewriting path constants
  and removing stale k8 assumptions.

### Missing production panel

There is not yet a compact production panel dedicated to the local
`delta_mean` sensitivity result. Add one rather than folding local `delta_mean`
into the aggregate `temporal_pca` panel. The panel should read:

```text
local posthoc dirs:
  backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_gabor_pyramid_rel025_0p5_1_seed7_v1/
    incremental_static_plus_motion_feature_adjudication_k2_4_8_16_32_v1/
  backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_gabor_pyramid_rel2_seed7_v1/
    incremental_static_plus_motion_feature_adjudication_k2_4_8_16_32_v1/

filter:
  latent = pyramid_local_field
  k = 16
  motion_summary = delta_mean
```

## Post-Run Validation Checklist

Run this checklist after v4 lands and again after the canonical aggregate run:

```text
[ ] final feature adjudication has missing_cache_only_gap_count == 0
[ ] top aggregate candidate remains pyramid_local_field k16 temporal_pca
[ ] local sensitivity candidate remains pyramid_local_field k16 delta_mean
[ ] joint rel0.25 posthoc validates feature identity, not row-order trust
[ ] aggregate canonical run metadata matches the locked latent/k/scale/seed policy
[ ] aggregate canonical run uses common-unclipped drift-only trace sources
[ ] effective/requested RMS is near 1.0 for every non-static family and scale
[ ] clipped fraction is 0.0 for every non-static family and scale
[ ] incremental posthoc uses fixed/shared ridge alpha 10.0
[ ] no stale `incremental_static_plus_motion/` folder is used for figure claims
[ ] figure pack reports temporal_pca as primary aggregate readout
[ ] local panel reports delta_mean as local/mechanistic sensitivity
[ ] claim text distinguishes model feature utility from behavioral proof
```

## Claim Wording

Preferred short version:

```text
Across BackImage natural-image windows, empirical drift-like motion improves a
V1-twin representation of local image structure beyond the static response
under a pyramid local-field temporal-PCA readout. A complementary paired-trace
local analysis shows that the same feature family has its clearest
image-contingent response-change signal under a delta-mean readout. Together,
these results support a model-behavior alignment claim: spontaneous drift is
biased toward a geometry that the model predicts can preserve or recover local
natural-image structure.
```

Guardrails:

- Do not claim the animal behaviorally optimizes the readout.
- Do not claim the joint axis term is the primary evidence.
- Do not claim `delta_mean` is the canonical aggregate readout.
- Do not treat `rel_2x` sentinel behavior as a source of the main positive.
- Do not present Gabor/k4 or pyramid/k8 rows as the final target once the k16
  two-readout spec is accepted.

## Immediate Next Actions

1. Wait for `backimage_feature_decomposition_adjudication_v4_joint_rel0p25_complete/`.
2. Record v4 scores into this note or a final lock note.
3. Parameterize the figure pack and atlas panel B away from hardcoded k4/k8.
4. Add the local `delta_mean` sensitivity panel.
5. Launch the canonical aggregate run only after the v4 acceptance criteria pass.
