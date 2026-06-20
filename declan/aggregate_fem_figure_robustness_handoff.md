# Aggregate FEM Figure And Robustness Handoff

Last updated: 2026-06-20.

## Purpose

This handoff turns the strongest current BackImage active-sensing positive into
a concrete coding-agent task:

```text
Build figure-ready summaries and targeted robustness checks for the cleaned
aggregate natural-image FEM information result.
```

This is the next manuscript-leverage priority while the raw-edge roadblock is
being adjudicated. It is not a new discovery pathfinder. The main run already
landed and should now be turned into a compact, auditable figure package.

## Scientific Claim Boundary

Current safe claim:

```text
In a canonical 756-unit V1 twin, empirical drift-like motion adds
feature-decodable signal beyond static natural-image responses and robustly
beats OU-like confined controls. The advantage over Brownian and rotated
empirical controls is strongest at small scales, especially 0.25x-0.5x, and
narrows at 1x-2x.
```

Do not claim:

- exact biological trace order is uniquely optimal;
- empirical FEMs globally maximize a scalar objective;
- larger motion is generally better;
- this is recorded V1 evidence rather than a deterministic V1-twin result;
- temporal PCA/DCT positives prove a biological temporal code without noise and
  readout robustness.

Preferred manuscript wording:

```text
empirical FEM statistics improve a V1-twin representation of natural-image
structure under this readout
```

not:

```text
FEMs improve biological V1 information.
```

## Main Result To Use

Primary completed run:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
```

Configuration:

```text
images = 256
canonical twin units = 756
trace samples per family/scale/image = 4
families = empirical, ou, brownian, rotated
scales = rel_0p25x, rel_0p5x, rel_1x, rel_1p5x, rel_2x
latents = gabor_local_field, pyramid_local_field
k = 4, 8
primary response summary = temporal_pca
cross-validation = grouped by image
trace policy = drift-only, common-unclipped through 2x
```

Important source files:

```text
analysis_images.csv
trace_bank_metadata.csv
aggregate_motion_metadata.csv
aggregate_motion_summary.csv
latent_feature_arrays.npz
response_summary_arrays.npz
decode_summary.csv
decode_contrasts.csv
covariance_summary.csv
run_metadata.json
summary_report.md
```

Critical incremental folder:

```text
incremental_static_plus_motion_relids/
```

Use this folder for static-plus-motion claims:

```text
incremental_static_plus_motion_relids/incremental_gain_vs_static.csv
incremental_static_plus_motion_relids/incremental_gain_contrasts.csv
incremental_static_plus_motion_relids/incremental_decode_summary.csv
incremental_static_plus_motion_relids/summary_report.md
```

Do not use the older `incremental_static_plus_motion/` folder for figure-level
claims unless it has been explicitly repaired. The `relids` folder was created
because the earlier automatic launch used stale scale IDs and produced empty
gain tables.

## Key Numbers To Preserve

Primary temporal-PCA empirical static-plus-motion gain:

```text
Gabor k=4:
  0.25x  +14.31, CI [+7.45, +21.79]
  0.5x   +13.04, CI [+6.81, +20.89]
  1x      +9.10, CI [+3.73, +14.86]
  1.5x    +9.98, CI [+5.36, +15.87]
  2x      +9.07, CI [+3.87, +15.73]

Pyramid k=8:
  0.25x   +5.20, CI [+3.02, +7.68]
  0.5x    +4.89, CI [+2.88, +7.07]
  1x      +3.93, CI [+1.93, +5.86]
  1.5x    +4.44, CI [+2.34, +6.64]
  2x      +4.21, CI [+2.38, +6.23]
```

Primary empirical control contrast for Gabor k=4, temporal PCA:

```text
empirical incremental gain advantage

0.25x: vs OU +21.24, vs Brownian +10.52, vs rotated +15.27
0.5x:  vs OU +19.59, vs Brownian  +7.89, vs rotated +11.21
1x:    vs OU +17.16, vs Brownian  +0.51, vs rotated  +5.63
1.5x:  vs OU +18.69, vs Brownian  +0.15, vs rotated  +8.58
2x:    vs OU +18.03, vs Brownian  -0.60, vs rotated  +7.55
```

Motion-quality sanity:

```text
accepted drift-only trace sources = 151 / 256
rows per family/scale = 1024
median effective/requested RMS = 1.0 for every family and scale
clipped fraction = 0.0 for every family and scale
same raw source traces reused across scales = yes
```

This is the central guardrail against the simple "more motion or clipping made
the result" explanation.

## Existing Code To Reuse

Aggregate runner:

```text
declan/fixation_statistics_by_stimulus/run_backimage_aggregate_fem_information.py
```

Incremental static-plus-motion posthoc:

```text
declan/fixation_statistics_by_stimulus/summarize_backimage_aggregate_incremental_motion.py
```

Cache proxy, useful only for debugging older fixed-axis screens:

```text
declan/fixation_statistics_by_stimulus/summarize_backimage_aggregate_cache_proxy.py
```

Existing collaborator figure pack:

```text
declan/fixation_statistics_by_stimulus/make_backimage_active_sensing_collab_figures.py
```

Important warning:

```text
make_backimage_active_sensing_collab_figures.py currently points at some older
aggregate paths. For this task, either add an n256-specific mode or create a
new figure script rather than silently plotting the older n128/pathfinder
outputs.
```

Recommended new script:

```text
declan/fixation_statistics_by_stimulus/make_backimage_aggregate_fem_figure_pack.py
```

Recommended output root:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
    figure_pack_v1/
```

## Figure Package Goals

Build a figure package that can stand on its own for collaborators and later
feed a manuscript panel.

### Panel A: Design And Motion QC

Show:

- families: empirical, OU, Brownian, rotated;
- scales: 0.25x, 0.5x, 1x, 1.5x, 2x;
- effective RMS versus requested scale;
- clipping fraction;
- path length or speed as a reminder that Brownian is motion-energy matched
  differently from empirical/OU/rotated.

Inputs:

```text
aggregate_motion_summary.csv
aggregate_motion_metadata.csv
trace_bank_metadata.csv
```

Expected conclusion:

```text
Motion bookkeeping is clean: effective RMS is matched to requested scale and
clipping is zero across families/scales.
```

### Panel B: Static-Plus-Motion Feature Gain

Plot empirical incremental gain versus static across scales.

Primary rows:

```text
motion_summary = temporal_pca
family = empirical
latent/k = gabor_local_field k=4
latent/k = pyramid_local_field k=8
```

Inputs:

```text
incremental_static_plus_motion_relids/incremental_gain_vs_static.csv
```

Expected conclusion:

```text
Empirical motion adds feature-decodable signal beyond static responses at every
tested scale, with strongest gain at small scales.
```

### Panel C: Empirical Versus Controls

Plot empirical-minus-control incremental gain across scales.

Primary rows:

```text
motion_summary = temporal_pca
latent = gabor_local_field
k = 4
lhs_family = empirical
rhs_family in {ou, brownian, rotated}
```

Inputs:

```text
incremental_static_plus_motion_relids/incremental_gain_contrasts.csv
```

Expected conclusion:

```text
Empirical robustly beats OU-like confined motion. The advantage over Brownian
and rotated controls is strongest at small scales and narrows near/above 1x.
```

### Panel D: Covariance / Signal-Motion Tradeoff

Use the existing covariance summary to show whether the response summary carries
image signal versus motion nuisance.

Inputs:

```text
covariance_summary.csv
```

Plot candidates:

```text
signal_cov_trace
motion_cov_trace
signal_motion_trace_ratio
signal_motion_subspace_overlap
```

Primary goal:

```text
Show the readout is not only a scalar decode score. Tie the aggregate result
back to the broader covariance/reafference story.
```

If the covariance table is too broad for a single main panel, make this a
supplement-style panel and keep a concise signal/motion ratio plot in the main
figure pack.

### Panel E: Claim Boundary / Negative Controls

Optional but useful collaborator panel:

- compare temporal PCA against temporal DCT and mean/delta summaries if present;
- show that not every response summary makes the same claim;
- show Brownian/rotated narrowing at high scales.

This panel is a guardrail, not a headline.

## Targeted Robustness Checks

Do these before requesting another broad forward run.

### 1. Fixed/Shared Alpha Sensitivity

Goal:

```text
Confirm the empirical-minus-control conclusions are not an artifact of
family-specific ridge regularization.
```

Check existing columns in:

```text
decode_summary.csv
incremental_static_plus_motion_relids/incremental_decode_summary.csv
```

Look for:

```text
ridge_alpha_mode
fixed_ridge_alpha
chosen_alpha_median
```

If the current incremental posthoc already used a shared/fixed alpha, document
that in the figure report. If not, rerun only the posthoc with shared/fixed
alpha; do not rerun the twin.

Output:

```text
robustness_fixed_alpha_summary.csv
robustness_fixed_alpha_report.md
```

### 2. Seed/Source Resampling From Existing Arrays

Goal:

```text
Assess whether the n=256 result depends on the sampled images or the four trace
samples per family/image.
```

Cache-first approach:

- use `response_summary_arrays.npz` and `latent_feature_arrays.npz`;
- bootstrap images grouped by session or image;
- if sample-index metadata is recoverable, resample trace samples within
  family/scale/image;
- preserve grouped-by-image CV for decode estimates where practical.

Minimum acceptable check:

```text
session bootstrap / image bootstrap on incremental_gain_vs_static and
empirical-minus-control contrasts
```

Preferred check:

```text
repeat the decoder over 20-100 resampled image/source subsets with fixed alpha
and report sign stability for empirical-minus-OU/Brownian/rotated.
```

Output:

```text
robustness_resampling_summary.csv
robustness_resampling_report.md
```

### 3. Concise Scale-Curve Audit

Goal:

```text
Make the scale story visually impossible to overstate.
```

Report:

- empirical gain versus static by scale;
- empirical-minus-OU by scale;
- empirical-minus-Brownian by scale;
- empirical-minus-rotated by scale;
- motion QC under the same x-axis.

Expected conclusion:

```text
The result is not a monotonic more-motion-is-better artifact. Empirical beats
OU across scales, while Brownian/rotated advantages are strongest at small
scales and narrow at larger scales.
```

### 4. Signal-Motion Covariance Panels

Goal:

```text
Tie the aggregate information result to the compact/reafference covariance
story without claiming a recorded-cortex bridge.
```

Use:

```text
covariance_summary.csv
```

Make a table and figure for:

```text
signal_cov_trace
motion_cov_trace
signal_motion_trace_ratio
signal_motion_subspace_overlap
```

If needed, restrict first to:

```text
summary = temporal_pca
latent = gabor_local_field
k = 4
```

then add pyramid k8 sensitivity.

### 5. Sanity Check The Incremental Folder

Goal:

```text
Prevent accidental reuse of stale empty-scale outputs.
```

The figure script should explicitly check:

```text
incremental_static_plus_motion_relids/incremental_gain_contrasts.csv exists
scale_id values are rel_0p25x, rel_0p5x, rel_1x, rel_1p5x, rel_2x
n_images == 256
n_sessions == 29
```

If this fails, stop with a clear error.

## Output Contract

Write these files under `figure_pack_v1/`:

```text
figure_pack_metadata.json
figure_pack_report.md
figure_source_tables/
  motion_qc_table.csv
  empirical_gain_curve.csv
  empirical_control_contrasts.csv
  covariance_panel_table.csv
  key_numbers_for_caption.csv
figures/
  aggregate_motion_qc.png
  aggregate_motion_qc.pdf
  empirical_static_plus_motion_gain.png
  empirical_static_plus_motion_gain.pdf
  empirical_vs_controls_scale_curve.png
  empirical_vs_controls_scale_curve.pdf
  signal_motion_covariance_summary.png
  signal_motion_covariance_summary.pdf
  aggregate_fem_summary_multipanel.png
  aggregate_fem_summary_multipanel.pdf
robustness/
  robustness_fixed_alpha_summary.csv
  robustness_resampling_summary.csv
  robustness_report.md
```

Also write a concise provenance table:

```text
panel_provenance.csv
```

Columns:

```text
panel
claim
source_file
filters
metric
n_images
n_sessions
n_trace_samples
unit_space
caveat
```

## Implementation Notes

Use `matplotlib` with `Agg` backend and `MPLCONFIGDIR=/tmp/matplotlib-cache`,
matching local plotting scripts.

Suggested CLI:

```text
python -m declan.fixation_statistics_by_stimulus.make_backimage_aggregate_fem_figure_pack \
  --run-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched \
  --incremental-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/incremental_static_plus_motion_relids \
  --out-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/figure_pack_v1
```

Use existing visual conventions from:

```text
declan/fixation_statistics_by_stimulus/make_backimage_active_sensing_collab_figures.py
```

but avoid modifying that script in-place unless you preserve backward
compatibility for older collaborator figures.

## Acceptance Criteria

The handoff is complete when:

- the figure pack uses the n256 patched run, not older n128/pathfinder outputs;
- static-plus-motion panels use `incremental_static_plus_motion_relids`;
- motion QC confirms effective RMS matching and zero clipping;
- the main empirical gain curve and empirical-minus-control curve reproduce the
  key numbers above;
- Brownian/rotated narrowing at high scales is visible and described;
- a covariance/signal-motion panel or table is included;
- the report clearly states the twin-scoped claim and guardrails;
- no new V1 forward pass was required for the first figure pack.

## Decision Rule

Promote to main active-sensing figure material if:

- empirical static-plus-motion gain remains positive under the selected
  figure/posthoc settings;
- empirical-minus-OU remains robust and visually clean;
- small-scale empirical advantages over Brownian/rotated are preserved;
- motion QC rules out clipping/effective-RMS confounds;
- the result can be explained in one compact panel plus one guardrail panel.

Keep as supplement or robustness material if:

- empirical beats OU but Brownian/rotated are too competitive for a clean main
  panel;
- the result depends strongly on ridge-alpha choice;
- covariance panels do not add interpretive clarity.

Demote if:

- corrected fixed-alpha or resampling checks erase empirical-minus-OU;
- all families collapse to the same scale curve;
- the apparent positive is driven by stale incremental outputs or mismatched
  scale IDs.

