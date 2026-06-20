# Working Results Draft

Status: v0.1 build draft, cache-first.

This document starts assembling the Figure 4 atlas as a readable Results module.
Inline `[FLAG]` notes point to `incomplete_results_flags.md`.

## Results Lead

Fixational eye movements are usually small enough to be easy to treat as a
nuisance, but they continuously transform a fixed screen image into a retinal
movie. That movie is not arbitrary motion blur. Because a small translation
samples image structure through local gradients, the response changes induced
by drift and microsaccades depend jointly on the eye trajectory and on the
spatial content of the image. We therefore asked whether FEM-linked response
variability can be treated as an active-sensing signal: first by testing whether
FEM-like movies add recoverable natural-image feature structure to V1-twin
responses, then by asking whether that information remains usable when the eye
trajectory is latent, and finally by comparing predicted useful motion axes
with the directions animals actually move during natural viewing.

[FLAG F001] External literature references from the source brief still need
verification before this paragraph becomes manuscript prose.

## Figure 4A: FEMs Turn Static Images Into Retinal Movies

Panel goal:

```text
Teach the physical premise before introducing decoders or compact geometry.
```

Proposed panel sequence:

- A1: show a fixed natural-image patch on the screen, a measured fixation trace,
  and three time points along the trace.
- A2: compare a stabilized movie row with an FEM movie row.
- A3: cartoon a small translation across an oriented edge or textured patch.
- A4: show the analysis pipeline: image + eye trace -> retinal movie -> V1 twin
  -> population response movie.
- A5: bridge to earlier covariance figures:
  `retinal motion -> response modulation -> Sigma_FEM`.

Draft result text:

```text
During fixation, a single screen image is not a single retinal input. Even when
the observer maintains fixation, drift and microsaccades translate the image
across the retina, so V1 receives a short movie of shifted samples. The
response changes produced by that movie are image-dependent: the same
displacement has different consequences depending on local gradients, edges,
and texture. This physical transformation motivates treating FEM-linked
population variability as reafferent sensory structure rather than merely as
noise to subtract.
```

Source assets:

```text
declan/fig4_active_sensing/
declan/active_sensing_movie_information/
outputs/active_sensing_movie_information/active_sensing_movie_information_figure_frozen_20260615_pre_backimage_collab_pack/
outputs/fig4_active_sensing/active_sensing_headline_figure/
```

Generated Panel A read:

```text
FEM retinal movies have temporal contrast and motion power relative to matched
stabilized movies, while static movie power remains matched:
  temporal contrast RMS mean: real = 11.245, stabilized = 0.000
  motion power vs stabilized: real = 1462.431, stabilized = 0.000
  movie power mean: real = 15178.177, stabilized = 15185.182

Downstream BackImage provenance:
  256 images
  29 sessions
  151 drift-only trace sources
  canonical 756-unit V1 twin
  RMS ratio = 1.0; clipping = 0.0
```

Build status:

```text
Panel A premise/QC subpanels generated in
declan/figure4_active_sensing_atlas/figures/panel_A/. Full figure composition
not attempted yet. The covariance bridge is supplemental because its evidence
classes use mixed denominators.
```

[FLAG F002] Module A needs final composite panels.

## Figure 4B: FEM Movies Add Feature-Decodable Structure

Panel goal:

```text
Show that FEM-like retinal motion can add feature-decodable structure beyond a
static V1-twin response.
```

Primary source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
    incremental_static_plus_motion_relids/
```

Verified files:

```text
incremental_gain_vs_static.csv
incremental_gain_contrasts.csv
incremental_decode_summary.csv
summary_report.md
run_metadata.json
```

Run scope:

```text
256 images
canonical 756-unit V1 twin
K=4 trace samples per image/family/scale
families: empirical, OU, Brownian, rotated
scales: 0.25x, 0.5x, 1x, 1.5x, 2x
CV: grouped by image
trace policy: drift-only, common-unclipped source pool
```

Motion QC:

```text
accepted drift-only sources: 151 / 256
median effective/requested RMS: 1.0 for every family/scale
clipped fraction: 0.0 for every family/scale
```

Panel B4: static-plus-motion gain.

Verified temporal-PCA incremental gain over static:

```text
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

Panel B5: matched-control contrast.

Verified Gabor k=4 temporal-PCA empirical advantage:

```text
0.25x: vs OU +21.24, vs Brownian +10.52, vs rotated +15.27
0.5x:  vs OU +19.59, vs Brownian +7.89,  vs rotated +11.21
1x:    vs OU +17.16, vs Brownian +0.51,  vs rotated +5.63
1.5x:  vs OU +18.69, vs Brownian +0.15,  vs rotated +8.58
2x:    vs OU +18.03, vs Brownian -0.60, vs rotated +7.55
```

Draft result text:

```text
In the cleaned aggregate BackImage analysis, empirical drift-like motion added
feature-decodable structure to the V1-twin response beyond a static response.
The effect was robust for temporal-PCA response summaries across Gabor and
pyramid feature targets. Empirical trajectories also outperformed matched
OU-like controls across the tested scales. The advantage over Brownian and
rotated controls was strongest at small, biologically plausible scales and
narrowed at larger scales, arguing against both a pure null and a simple
"more motion is always better" account.
```

Build status:

```text
Panel B subpanels generated from existing CSVs in
declan/figure4_active_sensing_atlas/figures/panel_B/. Full figure composition
not attempted yet.
```

[FLAG F003] Decoding proxy language required.

[FLAG F004] Brownian/rotated specificity narrows at larger scales.

[FLAG F005] Local exact image-trace pairing remains supplemental/unresolved.

## Figure 4C: Joint Image-And-Eye Observer

Panel goal:

```text
Show that an observer can recover image information under latent eye position by
marginalizing over plausible trajectories.
```

Primary source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
```

Run scope:

```text
n_images = 64
n_candidates = 8
candidate modes = hard_negative_structure, matched_static_response
observation family = empirical
prior families = empirical, OU
scales = 0.5, 1.0
n_prior_trajectories = 8
trajectory prior mode = leave_one_out
likelihood scales = 0.5, 1.0
```

Panel C2/C3: accuracy ordering and matched-static rescue.

Verified observer results:

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

Matched-static, 1.0x, likelihood scale 1.0:

```text
empirical prior:
  known = 1.000
  zero = 0.328
  joint = 0.766
  median N_eff / K = 0.364

OU prior:
  known = 1.000
  zero = 0.328
  joint = 0.797
  median N_eff / K = 0.400
```

Draft result text:

```text
The exact trajectory-table observer showed the expected ordering. When the
true trajectory was specified, image identity was recoverable. When the
observer incorrectly assumed zero eye motion, accuracy dropped strongly,
especially at larger motion scale. A joint observer that marginalized over a
finite catalog of plausible trajectories recovered much of that lost
performance, including in the matched-static-response condition where
distractors were selected to have similar stabilized V1-twin responses. The
posterior did not collapse onto a single trajectory, but it concentrated over a
small subset of plausible trajectories, consistent with partial latent-pose
constraint from natural-image response structure.
```

Panel C6: compact mechanism guardrail.

Primary compact source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    compact_mechanism_image_disjoint_fold0_n512_k2_5_10_20_rand8_log_v1/
```

At matched_static_response, 1.0x, likelihood scale 1.0:

```text
empirical compact_only, image-disjoint:
  k=2  joint = 0.563, true-score rescue = 0.784
  k=5  joint = 0.578, true-score rescue = 0.848
  k=10 joint = 0.547, true-score rescue = 0.804
  k=20 joint = 0.609, true-score rescue = 0.836

OU compact_only, image-disjoint:
  k=2  joint = 0.531, true-score rescue = 0.811
  k=5  joint = 0.531, true-score rescue = 0.854
  k=10 joint = 0.531, true-score rescue = 0.790
  k=20 joint = 0.563, true-score rescue = 0.840
```

Draft mechanism text:

```text
As a mechanism test, projecting motion-induced response deltas into an
image-disjoint compact translation basis preserved a large fraction of the
exact-table trajectory rescue and outperformed random, unit-shuffled, and
gain-only controls. This supports compact translation geometry as a sufficient
carrier of much of the motion-dependent likelihood structure. It does not yet
prove that compact geometry is the unique mechanism, because static-response PC
subspaces remain a close low-dimensional control.
```

Build status:

```text
Panel C subpanels generated from existing observer and compact-mechanism CSVs
in declan/figure4_active_sensing_atlas/figures/panel_C/. Full figure
composition not attempted yet. Compact mechanism best treated as mechanism
guardrail or supplement.
```

[FLAG F006] Compact mechanism is sufficient but not unique.

## Figure 4D: Image-Dependent Useful Motion Directions

Panel goal:

```text
Move from "motion helps" to "which directions of motion help for which images?"
```

Panel D1-D3 should define local axes and explain that different objectives can
prefer different directions. For a simple edge, normal motion creates large
luminance changes, while parallel motion preserves local structure. The atlas
should therefore avoid a one-line rule like "parallel is always best."

Axis-conditioned observer sources:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1/

outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
```

Verified axis-conditioned results:

```text
Matched-static n64, 0.5x, likelihood scale 1.0:
  known = 1.000
  zero = 0.641
  edge-parallel joint = 0.859
  edge-orthogonal joint = 0.828

Hard-negative n128, likelihood scale 1.0:
  0.5x: zero 0.609, parallel 0.813, orthogonal 0.781
  1.0x: zero 0.391, parallel 0.797, orthogonal 0.805
  2.0x: zero 0.336, parallel 0.680, orthogonal 0.742
```

Edge-parallel preservation source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_edge_parallel_stability_screen_yfix_n256_pop256/
```

Verified preservation result:

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

Draft result text:

```text
Useful motion axes were image-dependent rather than universal. In the
axis-conditioned observer, both edge-parallel and edge-orthogonal trajectory
priors rescued image identity relative to a zero-eye observer, but the preferred
axis changed with candidate set and scale. This makes the observer result a
clean argument for image-conditioned trajectory priors, not for a single
parallel-versus-orthogonal law. A separate preservation audit supplied the
clearest local geometric result: displacements parallel to local contours
disrupted both pixels and V1-twin responses less than matched orthogonal
displacements.
```

Build status:

```text
Panel D subpanels generated from existing axis-observer, edge-stability, and
objective-alignment CSVs in declan/figure4_active_sensing_atlas/figures/panel_D/.
Full figure composition not attempted yet. Use as a prediction/mechanism module
with explicit guardrails.
```

[FLAG F007] Axis preference is candidate-set and scale dependent.

[FLAG F008] Edge-parallel preservation is clean but not a full policy.

## Figure 4E: Free-Viewing FEMs Follow Image Geometry

Panel goal:

```text
Test whether measured FEM directions during natural viewing relate to local
image geometry.
```

Primary behavior source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_edge_alignment_distribution_inspection/
```

Metric:

```text
cos(2 * drift-edge delta)
+1 = edge-parallel
0 = 45 degrees from edge
-1 = edge-orthogonal
```

Verified behavior summary:

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

The current rendered headline figure uses a related weighted all-window
summary:

```text
all-window weighted edge-axis cos2 = 0.181
session-bootstrap CI = [0.124, 0.241]
reliable-axis weighted edge-axis cos2 = 0.201
```

Raw edge baseline source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_conditional_fixation_objectives_twin_axis_only_n256/
    alignment_by_objective_summary.csv
```

Verified raw-edge row:

```text
raw_edge_axis:
  n_windows = 256
  n_sessions = 29
  mean session cos2 = 0.182
  weighted session cos2 = 0.218
  positive sessions = 23 / 29
```

Draft result text:

```text
Measured FEM direction was not random with respect to local image structure.
Across free-viewing windows, drift/fixation-cloud axes were modestly but
reliably aligned with local edge geometry, and the alignment strengthened in
subsets with more reliable local axis estimates. This supports the broad
active-sensing prediction that fixational motion is image-contingent. At the
same time, the strongest current behavioral baseline is raw image geometry:
the tested V1-twin pose-aware, pose-blind, and Pareto objectives do not yet
cleanly outperform the raw edge axis.
```

Build status:

```text
Panel E subpanels generated from existing image-geometry and drift-edge
alignment CSVs in declan/figure4_active_sensing_atlas/figures/panel_E/. Full
figure composition not attempted yet. Objective adjudication remains unresolved.
```

[FLAG F009] Behavior aligns with raw image geometry better than current model
objectives.

[FLAG F012] Behavioral alignment metric convention needs to be chosen.

## Current Main-Figure Compression

Working main figure:

- 4A: fixed screen image -> retinal movie -> V1 response movie.
- 4B: empirical drift-like motion adds feature-decodable structure beyond
  static and beats OU, with small-scale Brownian/rotated guardrail.
- 4C: known-eye > joint-eye > zero-eye, including matched-static distractors.
- 4D: local image geometry defines useful axes; preservation is clean, axis
  preference is objective-dependent.
- 4E: measured free-viewing FEM axes align modestly but reliably with local
  image geometry.

Working supplement routing:

- retinal rendering QC;
- motion-family matching and effective-RMS checks;
- response summary comparisons;
- local exact-pairing branch;
- Vernier failure versus natural-image success;
- posterior concentration and image-condition diagnostics;
- compact projection mechanism tests;
- full axis-conditioned observer tables;
- model-objective versus raw-edge behavioral comparisons.

[FLAG F010] Final atlas figure composites are not built yet.
