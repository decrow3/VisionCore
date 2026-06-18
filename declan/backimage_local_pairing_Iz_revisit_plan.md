# Local BackImage I_z Pairing Revisit

Last curated: 2026-06-18.

## Purpose

Reopen the local BackImage `I_z` branch after the positive aggregate FEM result,
but with a sharper question than the original fixed-axis screen:

```text
Does the actual image-trace pairing provide local feature-decoding benefit
beyond the aggregate usefulness of empirical drift statistics?
```

The aggregate result asks:

```text
I ~ p(I), tau ~ q_empirical(tau)
```

This local revisit asks:

```text
I_i paired with its own tau_i
```

A null local result would not undermine the aggregate result. It would say that
empirical drift statistics are useful distributionally, while exact local
image-trace matching remains weak or unresolved under this readout.

## Aggregate Lessons To Inherit

The local revisit should deliberately reuse the aggregate analysis conventions
that made the later result interpretable:

```text
strict drift-only trace-bank filtering
source traces reused across scales
effective RMS and clipping/capping audit for every rendered trace
common-unclipped policy for above-1x scale claims
grouped-by-image/source_row cross-validation
static-plus-motion incremental score as the primary readout
fixed/shared ridge alpha for primary family contrasts
Brownian and rotated controls as large-scale/generic-motion guardrails
scale IDs in the corrected rel_0p25x / rel_0p5x / rel_1x convention
```

The output contract should also match the aggregate runner wherever possible:

```text
analysis_images.csv
trace_bank_metadata.csv
local_pairing_motion_metadata.csv
latent_feature_arrays.npz
response_summary_arrays.npz
decode_summary.csv
decode_contrasts.csv
covariance_summary.csv
```

The local runner should write the primary incremental static-plus-motion tables
directly, with the contrast table configured for local families such as:

```text
actual_paired_empirical:matched_unpaired_empirical
actual_paired_empirical:rotated_actual_90
actual_paired_empirical:ou_matched_actual
actual_paired_empirical:brownian_matched_actual
edge_axis:edge_orthogonal
```

The local-specific part is candidate construction. The aggregate runner samples
unpaired traces from `q(tau)`; the local runner must additionally render the
actual paired trace `tau_i` and matched alternatives for the same image `I_i`.

## Primary Question

For each BackImage fixation window `i`, image patch `I_i`, measured drift trace
`tau_i`, and image latent `z_i = phi(I_i)`:

```text
Does F_theta(I_i, tau_i) decode z_i better than F_theta(I_i, tau_j),
where tau_j is a matched empirical drift trace from another fixation?
```

Primary contrast:

```text
actual paired empirical trace - matched unpaired empirical trace
```

Secondary contrasts:

```text
actual paired empirical trace - rotated actual trace
actual paired empirical trace - OU matched trace
actual paired empirical trace - Brownian matched trace
actual paired empirical trace - raw edge-axis trace
actual paired empirical trace - edge-orthogonal trace
edge-axis trace - edge-orthogonal trace
```

All contrasts should be computed as incremental gains beyond static where
possible:

```text
gain_family = score(R_static + R_motion_family) - score(R_static)
paired_advantage = gain_actual_paired - mean(gain_matched_unpaired)
```

## Inputs And Unit Space

Use the canonical `756`-unit V1 twin for any interpretable run. Smaller
populations are smoke/pathfinder only.

Use a fixed BackImage manifest for all candidate families. Preferred manifest:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_latent_information_scalesweep_n256_rel0125-2_rand8_delta/
    analysis_windows.csv
```

The runner should use `--window-manifest` replay and must not resample windows
when controls change.

## Trace Policy

Use the aggregate runner's trace-bank machinery and Jake microsaccade detector.
The primary analysis should be strict drift-only:

```text
--max-trace-source-microsaccade-events 0
--max-trace-source-rms-deg 0.06
--max-trace-source-radius-deg 0.2
--max-rendered-trace-path-length-deg 1.5
--max-trace-source-speed-p95-deg-s 20
```

The actual paired trace must pass the same criteria for the strict primary
subset. If too many actual traces fail, also report a flagged natural-fixation
subset, but do not mix it into the primary result.

Required trace QC fields:

```text
source_row
trace_bank_index
n_microsaccade_events
fraction_microsaccade_samples
peak_microsaccade_speed_dps
observed_rms_deg
effective_rms_deg
effective/requested_rms
max_radius_deg
rendered_path_length_deg
source_path_length_deg
speed_p95_deg_s
clipping/capping flag
```

## Candidate Families

For each image/window `i`, construct:

1. `actual_paired_empirical`: `I_i + tau_i`.
2. `matched_unpaired_empirical`: `I_i + tau_j`, `j != i`, excluding the same
   session/trial by default and matched on rendered effective RMS, rendered path
   length, source/rendered anisotropy, lag-1 autocorrelation, source radius, and
   drift-only status. Use `K_unpaired = 4` minimum, `8` preferred.
3. `rotated_actual_90`: actual trace rotated by 90 degrees.
4. `rotated_actual_random`: actual trace randomly rotated while preserving
   radius/time structure.
5. `ou_matched`: OU matched to effective RMS, duration, lag-1 autocorrelation,
   and covariance shape where available.
6. `brownian_matched`: Brownian matched to effective RMS and duration.
7. `edge_axis`: axis-constrained local trace along the raw edge axis.
8. `edge_orthogonal`: axis-constrained local trace along the edge-orthogonal
   axis.
9. `static`: patch-centered static baseline.

Use the same raw trace source across scales for every candidate where
applicable. Record effective RMS and clipping for every rendered trajectory.
All candidate traces are rendered to a fixed `n_timepoints`; source duration is
recorded for audit/source filtering but is not part of the primary rendered
matching distance.
For edge-axis controls, preserve the actual trace's dominant 1D temporal
waveform and place that waveform on the edge or edge-orthogonal axis. Do not use
the older sinusoidal line template for primary edge contrasts, because it
confounds axis geometry with path length and temporal profile.

## Scales

Primary:

```text
0.25x, 0.5x, 1x
```

Optional diagnostic:

```text
1.5x, 2x
```

Large-scale local claims must be effective-RMS-aware because earlier local
screens showed substantial clipping at large nominal scales.

## Features And Response Summaries

Primary feature latents:

```text
gabor_local_field, k=4
pyramid_local_field, k=8
```

Secondary if cheap:

```text
gabor_local_field, k=8
pyramid_local_field, k=4
dct_local_field, k=8
```

Feature extraction must match the corrected post-fix implementation:

```text
Gabor local fields include even, odd, and amplitude maps.
Pyramid local fields use the expanded local grid.
Feature PCA/standardization is fit within training folds.
```

Response summaries should match the aggregate analysis:

```text
temporal_pca       primary temporal-code summary
temporal_dct       secondary temporal-code summary
delta_mean         motion-induced feature-remapping summary
mean               integrated response summary
```

Do not collapse readouts unless explicitly labeled.

## Decoder And Scoring

Use grouped cross-validation by image/window:

```text
decode_group_mode = image or source_row
all rows from the same image/window stay in the same fold
```

Primary score:

```text
held-out negative MSE for z decoding; higher is better
```

Regularization policies:

```text
primary: fixed/shared alpha per latent/readout/scale across families
sensitivity: nested alpha with identical fold structure
```

Candidate-specific alpha is allowed only as a sensitivity analysis.

Save fold assignments, group IDs, selected alpha, feature dimensions, and
per-window held-out scores.

## Regime Tests

Predefine a small set of strata:

```text
high vs low image_orientation_coherence
high vs low drift anisotropy
high vs low real-edge alignment
high vs low edge-parallel pixel stability advantage
```

Optional:

```text
observed RMS
edge density
V1-twin edge-parallel stability advantage
```

For each stratum, report:

```text
paired_advantage
actual_minus_rotated
actual_minus_OU
edge_minus_edge_orthogonal
n windows
session distribution
```

Use session-clustered bootstrap where feasible.

## Minimum Pathfinder

```text
n = 128 windows
K_unpaired = 4
population = canonical 756
scales = 0.25x, 0.5x, 1x
latents = gabor_local_field k=4, pyramid_local_field k=8
summary = temporal_pca
families = actual_paired, matched_unpaired, rotated_actual_90,
  ou_matched, edge_axis, edge_orthogonal
CV = grouped by image/source_row
trace bank = strict drift-only
alpha = fixed/shared or nested with identical folds
```

For the local runner, `--max-images` means the target number of accepted paired
windows after strict actual-trace filtering. The trace bank and matched-unpaired
pool should be built from the larger filtered source table, then the accepted
paired analysis set should be sampled.

Practical smoke order:

```text
1. Gabor-only dry run, to validate strict filtering and trace metadata.
2. Gabor-only GPU smoke, to validate response arrays and incremental decode.
3. Add pyramid_local_field once the cache path is proven.
```

Runner command pattern:

```bash
python -m declan.fixation_statistics_by_stimulus.run_backimage_local_pairing_Iz_revisit \
  --max-images 128 \
  --latent-names gabor_local_field \
  --pca-k-list 4 \
  --observed-rms-scales 0.25,0.5,1.0 \
  --unpaired-samples-per-image 4 \
  --max-trace-source-rms-deg 0.06 \
  --max-trace-source-radius-deg 0.2 \
  --max-rendered-trace-path-length-deg 1.5 \
  --max-trace-source-speed-p95-deg-s 20 \
  --max-trace-source-microsaccade-events 0
```

The runner writes local incremental summaries directly:

```text
incremental_decode_summary.csv
incremental_gain_vs_static.csv
incremental_gain_contrasts.csv
```

For `matched_unpaired_empirical`, the primary gain estimator is mean-over-K
unpaired sample gains, not decoding after averaging the K response vectors.

Decision criteria:

```text
actual_paired > matched_unpaired:
  promote local pairing as a candidate behavioral-payoff panel.

actual_paired ~= matched_unpaired but edge_axis > edge_orthogonal:
  keep local result as image-stability constraint; aggregate remains main
  functional panel.

actual_paired < matched_unpaired or rotated_actual > actual_paired:
  do not claim image-specific trace matching.
```

## Current Clean Status

The first local-pairing pathfinders remain diagnostic only. They produced an
encouraging fixed-manifest `K_unpaired=32` pyramid result at `0.25x`, but a code
review found two implementation mismatches:

```text
1. With --window-manifest, the runner built the matched-unpaired trace bank
   from the same 128 analysis windows. The K=32 control was therefore a
   within-manifest permutation baseline, not the intended full strict
   drift-only trace-bank baseline.

2. The local runner used reduced feature geometry defaults:
   patch_size_px=160, latent_crop_px=96, local_field_grid=4.
   This yielded gabor_local_field (N, 1152) and pyramid_local_field (N, 768),
   rather than the corrected aggregate/local-screen convention:
   patch_size_px=540, latent_crop_px=151, local_field_grid=8,
   gabor_local_field (N, 4608), pyramid_local_field (N, 3072).
```

Therefore the current local result should be described as a useful pathfinder:

```text
On one fixed window set, actual image-trace pairing beat a high-K matched
within-manifest empirical baseline for small-scale pyramid temporal summaries.
```

It should not be described as:

```text
Actual image-trace pairing has beaten the intended full matched-unpaired
empirical control.
```

The corrected runner now freezes the actual analysis windows while building the
matched-unpaired trace bank from the full filtered source pool, and uses the
aggregate feature geometry defaults listed above.

A subsequent review of the corrected full-pool run found additional analysis
bookkeeping caveats:

```text
1. decode_contrasts.csv is diagnostic only for matched-unpaired local claims.
   It compares actual paired responses against the averaged K-trace
   matched_unpaired_empirical response condition. The claim-relevant local
   contrast is incremental_gain_contrasts.csv, which uses mean-over-sample
   matched-unpaired gains.

2. The original K_unpaired implementation selected deterministic nearest-K
   matched traces; seed changes did not sample new matched controls for a fixed
   manifest. The runner should instead sample without replacement from a
   near-match candidate pool and write match distance/rank diagnostics.

3. Runs without --window-manifest sample a new accepted-window set after strict
   filtering. They are pathfinders, not seed replications. Use the saved
   analysis_images.csv as a fixed manifest for control-draw or rerun
   comparisons.

4. Absolute temporal summaries such as temporal_pca and temporal_dct encode
   R_static + R_motion_absolute in the incremental model. They should be read as
   preservation/absolute-response diagnostics. Motion-contribution claims should
   prioritize delta summaries or an explicit static-plus-delta feature contract.
```

Completed clean runs:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_pyramid_rel025_1_v1

outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_gabor_pyramid_rel025_1_seed7_v1
```

Both runs used the same fixed `128`-image manifest, the full strict filtered
trace pool (`3013` trace-bank rows), sampled matched-unpaired controls
(`K_unpaired=32`), grouped-by-image decoding, the canonical `756`-unit twin, and
scales `0.25x` and `1x`. Motion QC was clean: `0` same-trial matched controls,
`0` clipping, and median effective/requested RMS `1.0` for every
family/scale. The second run confirmed corrected feature dimensions:

```text
gabor_local_field   (128, 4608)
pyramid_local_field (128, 3072)
```

The claim-relevant local-pairing file is:

```text
incremental_gain_contrasts.csv
```

Do not use `decode_contrasts.csv` as the local-pairing headline because its
matched-unpaired condition decodes the averaged K-trace response vector. The
local-pairing contrast uses mean-over-sample gains.

Clean actual-paired minus matched-unpaired result, seed 7:

```text
delta_mean, gabor k=4,   0.25x: +9.95, CI [+0.73, +20.62]
delta_mean, gabor k=4,   1x:    +8.27, CI [+2.70, +14.79]
delta_mean, gabor k=8,   0.25x: +6.33, CI [+1.27, +11.90]
delta_mean, gabor k=8,   1x:    +6.51, CI [+2.77, +11.07]

delta_mean, pyramid k=4, 0.25x: +6.89, CI [+1.66, +12.17]
delta_mean, pyramid k=4, 1x:    +2.63, CI [-1.26, +6.55]
delta_mean, pyramid k=8, 0.25x: +6.09, CI [+1.51, +10.53]
delta_mean, pyramid k=8, 1x:    +3.79, CI [+1.46, +6.28]
```

This is the current positive local result:

```text
Actual local fixation traces beat matched empirical trace swaps for
motion-induced feature-response deltas, and this survives in both Gabor and
pyramid local-field features.
```

The result does not yet support a broad temporal-code claim. For
`temporal_pca` and `temporal_dct`, actual-minus-matched mostly crosses zero or
goes negative, especially for Gabor. Actual paired traces also do not cleanly
beat the `rotated_actual_90` control. Therefore the current claim boundary is:

```text
Local image-trace pairing carries extra feature-relevant response delta beyond
matched aggregate empirical FEM statistics, but this has not yet become a
general temporal-code result or a unique-axis-optimality result.
```

Relationship to the aggregate result:

```text
The aggregate BackImage FEM analysis shows that empirical drift statistics are
useful distributionally. The local-pairing result adds a narrower possibility:
the specific trace paired with its local image/fixation context can carry extra
feature-relevant delta signal beyond matched empirical trace swaps. This is an
additional local-image-contingent benefit, not a replacement for the aggregate
distributional claim.
```

## Full Run

If the pathfinder is promising:

```text
n = 256 windows
K_unpaired = 8
population = canonical 756
scales = 0.25x, 0.5x, 1x, optional 1.5x
latents = gabor_local_field k=4, pyramid_local_field k=8
summaries = temporal_pca, temporal_dct, delta_mean
families = actual_paired, matched_unpaired, rotated_actual_90,
  rotated_actual_random, ou_matched, brownian_matched, edge_axis,
  edge_orthogonal
```

## Required Outputs

Create:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_local_pairing_Iz_revisit_<tag>/
```

Required files:

```text
run_metadata.json
analysis_images.csv
trace_bank_metadata.csv
actual_trace_filter_qc.csv
local_pairing_motion_metadata.csv
local_pairing_motion_summary.csv
latent_feature_arrays.npz
response_summary_arrays.npz
decode_summary.csv
decode_contrasts.csv
covariance_summary.csv
incremental_decode_summary.csv
incremental_gain_vs_static.csv
incremental_gain_contrasts.csv
summary_report.md
```

## Suggested Implementation

New runner:

```text
declan/fixation_statistics_by_stimulus/run_backimage_local_pairing_Iz_revisit.py
```

The aggregate-compatible response arrays can still be re-summarized with:

```text
declan/fixation_statistics_by_stimulus/summarize_backimage_aggregate_incremental_motion.py
```

But the claim-relevant local primary uses the runner's direct incremental files,
because those preserve per-sample matched-unpaired gains.

Reuse from existing code:

```text
run_backimage_aggregate_fem_information.py:
  _prepare_windows
  _session_dataset_cache
  _build_trace_bank
  _eligible_trace_bank_indices
  _family_raw_trace / _family_trace / _scale_family_raw_trace
  _extract_requested_latents
  temporal response summaries

run_backimage_latent_information_screen.py:
  CanonicalTwinScorer
  _cross_validated_decode
  _static_trace
  _align_response_to_trace
  _trace_rms

run_backimage_trajectory_table_observer.py:
  leave-one-out empirical prior sampling and trace duplicate checks

audit_backimage_latent_real_random.py:
  effective-scale, leave-session-out, and regime summaries
```

## Claim Boundary

Safe positive wording:

```text
Actual image-trace pairings provide additional feature-decodable temporal
structure beyond matched unpaired empirical traces under a V1-twin readout.
```

Unsafe wording:

```text
The animal selects optimal FEM trajectories for each image.
```
