# Main Figure Compression v0

Status: first compression pass from the atlas.

This is the current best main Figure 4 if we compress from the atlas today. It
prioritizes the broad active-sensing story and uses compact geometry as a
mechanism candidate rather than the figure headline.

## Main Claim

```text
FEMs convert static natural images into retinal movies that add
feature-decodable structure to V1-twin responses. That structure remains usable
when retinal pose is latent, and measured drift directions are aligned with
local image geometry.
```

## Proposed Main Panels

### 4A: Retinal Movie Premise

Content:

```text
Fixed image + eye trace -> moving retinal crop -> V1 response movie.
```

Source status:

```text
Atlas subpanels generated. Full figure composite is not built.
```

Candidate assets:

```text
declan/figure4_active_sensing_atlas/figures/panel_A/A1_retinal_movie_transform.png
declan/figure4_active_sensing_atlas/figures/panel_A/A2_movie_transform_qc.png
declan/figure4_active_sensing_atlas/figures/panel_A/A3_gradient_sampling_cartoon.png
declan/figure4_active_sensing_atlas/figures/panel_A/A4_backimage_pipeline_bridge.png
outputs/active_sensing_movie_information/active_sensing_movie_information_figure_frozen_20260615_pre_backimage_collab_pack/retinal_movie_transform_qc.png
outputs/fig4_active_sensing/active_sensing_headline_figure/fig4_active_sensing_headline.png
```

Flags:

```text
F002
```

### 4B: FEM-Like Motion Adds Feature-Decodable Structure

Content:

```text
Static-plus-motion feature decoding gain over static for empirical drift-like
motion, with Gabor k=4 and/or pyramid k=8 temporal-PCA readouts.
```

Primary numbers:

```text
Gabor k=4 temporal PCA, empirical gain over static:
  0.25x +14.31, CI [+7.45, +21.79]
  0.5x  +13.04, CI [+6.81, +20.89]
  1x     +9.10, CI [+3.73, +14.86]

Pyramid k=8 temporal PCA, empirical gain over static:
  0.25x +5.20, CI [+3.02, +7.68]
  0.5x  +4.89, CI [+2.88, +7.07]
  1x    +3.93, CI [+1.93, +5.86]
```

Candidate asset:

```text
outputs/fig4_active_sensing/active_sensing_headline_figure/fig4_active_sensing_headline.png
```

Primary source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
    incremental_static_plus_motion_relids/incremental_gain_vs_static.csv
```

Flags:

```text
F003
```

### 4C: Biological-Like Motion Benefit Is Scale And Control Dependent

Content:

```text
Empirical-minus-control contrasts, with OU as strong positive contrast and
Brownian/rotated narrowing at larger scales.
```

Primary numbers:

```text
Gabor k=4 temporal PCA, empirical-minus-control:
  0.25x: OU +21.24, Brownian +10.52, rotated +15.27
  0.5x:  OU +19.59, Brownian  +7.89, rotated +11.21
  1x:    OU +17.16, Brownian  +0.51, rotated  +5.63
```

Primary source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
    incremental_static_plus_motion_relids/incremental_gain_contrasts.csv
```

Flags:

```text
F003
F004
```

### 4D: Joint Image-And-Eye Observer Recovers Pose-Lost Information

Content:

```text
Known-eye, zero-eye, and joint-eye accuracy for hard-negative and
matched-static candidate sets. This is the new conceptual center that the
existing headline figure does not yet include.
```

Primary numbers:

```text
matched_static_response, 1.0x:
  known = 1.000
  zero = 0.328
  joint = 0.766 empirical prior
  joint = 0.797 OU prior
```

Primary source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    observer_summary.csv
```

Generated atlas asset:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/C2_accuracy_ordering.png
declan/figure4_active_sensing_atlas/figures/panel_C/C3_matched_static_rescue.png
declan/figure4_active_sensing_atlas/figures/panel_C/C4_posterior_concentration.png
```

Flags:

```text
F011
```

### 4E: Image Geometry Defines Useful Motion Axes

Content:

```text
Local edge geometry defines parallel and orthogonal axes; edge-parallel motion
preserves pixels and V1-twin responses relative to orthogonal motion.
```

Primary numbers:

```text
pixel preservation advantage:
  session mean = 300.54
  CI = [172.789, 408.961]
  positive sessions = 26 / 29

twin preservation advantage:
  session mean = 0.000454497
  CI = [0.000371047, 0.000536519]
  positive sessions = 29 / 29
```

Candidate assets:

```text
declan/figure4_active_sensing_atlas/figures/panel_D/D1_local_axis_schematic.png
declan/figure4_active_sensing_atlas/figures/panel_D/D2_axis_conditioned_accuracy.png
declan/figure4_active_sensing_atlas/figures/panel_D/D4_edge_parallel_stability.png
```

Flags:

```text
F008
```

### 4F: Measured FEMs Align With Local Image Geometry

Content:

```text
Behavioral drift-edge alignment distribution or edge-parallel enrichment.
```

Primary metric options:

```text
Option 1: weighted all-window alignment from current headline figure
  all-window weighted cos2 = 0.181
  session-bootstrap CI = [0.124, 0.241]
  reliable weighted cos2 = 0.201

Option 2: unweighted session-mean alignment from distribution inspection
  all windows mean session cos2 = 0.105, CI [0.067, 0.145]
  reliable axes mean session cos2 = 0.140, CI [0.089, 0.188]
  high confidence mean session cos2 = 0.269, CI [0.138, 0.396]

Endpoint-zone enrichment:
  all windows parallel <=15 deg = 1.304x uniform expectation
  reliable axes parallel <=15 deg = 1.427x uniform expectation
  high confidence parallel <=15 deg = 2.124x uniform expectation
```

Recommendation:

```text
Use Option 2 for the atlas text because it is easiest to describe as a
session-level behavioral estimate. Keep Option 1 in the figure stats manifest
or use it only if the plotted panel is the existing headline panel.
```

Primary sources:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_edge_alignment_distribution_inspection/edge_alignment_distribution_summary.csv

outputs/fig4_active_sensing/active_sensing_headline_figure/
  fig4_active_sensing_headline_stats.json
```

Generated atlas assets:

```text
declan/figure4_active_sensing_atlas/figures/panel_E/E2_behavior_alignment_strength.png
declan/figure4_active_sensing_atlas/figures/panel_E/E3_parallel_zone_enrichment.png
declan/figure4_active_sensing_atlas/figures/panel_E/E4_metric_convention_guardrail.png
declan/figure4_active_sensing_atlas/figures/panel_E/E5_scope_summary.png
```

Flags:

```text
F009
F012
```

## What Drops To Supplement

- Retinal rendering QC and movie transform details.
- Motion QC and absolute-gain sanity checks.
- Local exact image-trace pairing branch.
- Posterior concentration and image-condition diagnostics.
- Compact projection mechanism tests.
- Axis-conditioned observer details.
- Model objective comparison versus raw edge geometry.
- Endpoint-zone enrichment if 4F uses the distribution plot instead.

## Immediate Build Gap

The current existing headline figure covers parts of A/B/C-as-control/D/E in
its older numbering, but it does not include the joint observer. A cache-only
joint-observer panel now exists in this folder; the next step is integrating it
with the rest of the figure style.
