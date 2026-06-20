# Incomplete Results Flags

Status: running list, started 2026-06-19.

Use this file to keep the atlas honest while building. A flag does not mean the
story fails; it means the result, source file, figure asset, or claim boundary
needs care before promotion.

## Active Flags

### F001: External Literature Citations Not Verified Locally

Modules: A, C.

The source brief cited external PubMed/Nature references for the active-sensing
frame and recent retinal trajectory-inference work. Those links were not
verified while creating this local workspace.

Action before manuscript use:

```text
Verify the citations, exact titles, dates, and claims against source pages or
PDFs before including them in Results prose.
```

Current handling:

```text
Keep external citations out of the local atlas draft except as placeholders.
```

### F002: Module A Needs Final Composite Selection

Module: A.

The retinal-movie premise is scientifically straightforward and cache-only A1-A5
subpanels now exist, but the atlas does not yet have a final composed Figure 4A
layout or final decision about whether A5 stays in the main figure or
supplement.

Candidate sources:

```text
declan/figure4_active_sensing_atlas/figures/panel_A/
outputs/active_sensing_movie_information/active_sensing_movie_information_figure_frozen_20260615_pre_backimage_collab_pack/
outputs/fig4_active_sensing/active_sensing_headline_figure/
outputs/covTFTS_figure_frozen_20260615_pre_backimage_active_sensing_collab_pack/
```

Action:

```text
Choose which generated A subpanels survive into the compressed main figure.
Keep A5 as bridge/supplement unless the covariance denominator caveat is made
central and explicit.
```

### F003: Aggregate FEM Information Is A Decoding Proxy

Module: B.

The cleaned BackImage aggregate result is strong for deterministic
static-plus-motion feature-decoding gain, but it is not literal mutual
information.

Primary source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
    incremental_static_plus_motion_relids/
```

Current handling:

```text
Use "feature-decodable structure", "decoding gain", or "information proxy".
Avoid unqualified "mutual information" unless a fixed noise/logdet model is
added.
```

### F004: Brownian/Rotated Control Specificity Narrows At Larger Scales

Module: B.

Empirical motion beats OU robustly, but the empirical advantage over Brownian
and rotated controls is clearest at small scales and narrows at 1x-2x.

Verified examples:

```text
Gabor k=4, temporal PCA, empirical - Brownian:
  0.25x +10.52, CI [+5.09, +16.17]
  0.5x   +7.89, CI [+2.56, +12.79]
  1x     +0.51, CI [-4.32, +5.33]
  2x     -0.60, CI [-7.09, +5.36]
```

Current handling:

```text
Phrase the benefit as small-scale and distributional. Do not claim empirical
motion uniquely beats every generic-motion control across all scales.
```

### F005: Local Exact Image-Trace Pairing Is Supplemental/Unresolved

Module: B.

The local pairing branch supports a narrower result than the aggregate branch.
It should not carry the headline unless rechecked and made visually coherent.

Candidate sources:

```text
declan/backimage_local_pairing_Iz_revisit_plan.md
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_gabor_pyramid_rel025_1_seed7_v1/
```

Current handling:

```text
Keep as optional supplement. The main Module B claim is distributional.
```

### F006: Compact Mechanism Is Sufficient But Not Unique

Module: C.

The image-disjoint compact projection preserves much of the exact-table
trajectory rescue and beats random/unit-shuffle/gain controls, but static-PC
subspaces recover similar true-score rescue at k=10/20.

Primary source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    compact_mechanism_image_disjoint_fold0_n512_k2_5_10_20_rand8_log_v1/
```

Static-PC overlap source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    compact_mechanism_image_disjoint_fold0_n512_k2_5_10_20_rand8_log_v1/
      followup_summary/compact_staticpc_basis_overlap.csv
```

Current handling:

```text
Use compact geometry as a mechanism test or supplement. Claim compact
translation subspace sufficiency above random/unit-shuffle/gain controls, not
unique necessity.
```

### F007: Axis Preference Is Candidate-Set And Scale Dependent

Module: D.

Axis-conditioned priors rescue image identity over zero-eye, but the preferred
axis is mixed.

Verified examples:

```text
Matched-static n64, 0.5x:
  zero = 0.641
  parallel joint = 0.859
  orthogonal joint = 0.828

Hard-negative n128, likelihood scale 1.0:
  0.5x: zero 0.609, parallel 0.813, orthogonal 0.781
  1.0x: zero 0.391, parallel 0.797, orthogonal 0.805
  2.0x: zero 0.336, parallel 0.680, orthogonal 0.742
```

Current handling:

```text
Use this as evidence for image-conditioned axis priors and objective dependence.
Do not frame it as a universal edge-parallel result.
```

### F008: Edge-Parallel Preservation Is Clean But Not A Full Policy

Module: D.

The edge-parallel preservation audit is strong, but it addresses local
structure preservation, not the full active-sensing objective.

Primary source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_edge_parallel_stability_screen_yfix_n256_pop256/
```

Current handling:

```text
Use as a mechanistic/explanatory panel: sliding along contours preserves local
pixels and V1-twin responses. Do not equate preservation with all useful
motion.
```

### F009: Behavior Aligns With Raw Image Geometry Better Than Current Model Objectives

Module: E.

Behavioral drift-edge alignment is positive, but current V1-twin
pose-aware/pose-blind/Pareto objectives do not clearly outperform raw edge
geometry.

Primary sources:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_edge_alignment_distribution_inspection/
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_conditional_fixation_objectives_twin_axis_only_n256/
    alignment_by_objective_summary.csv
```

Current handling:

```text
State the behavioral result as local image-geometry alignment. Treat specific
V1-twin objective adjudication as unresolved. Panel E now includes a generated
scope-summary guardrail rather than promoting a specific response objective.
```

### F010: Final Atlas Figure Composites Are Not Built Yet

Modules: A-E.

This workspace currently contains a document draft, provenance notes, and
cache-only subpanel sets for A-E. It does not yet contain final atlas panel
composites.

Action:

```text
After the draft stabilizes, compose selected subpanels into the full Figure 4
layout.
```

### F011: Joint-Observer Subpanels Are Generated But Not Yet Integrated

Module: C.

The exact trajectory-table observer has strong matched-static results, but the
current rendered headline figure does not include this conceptual center. A
cache-only atlas subpanel set has now been generated locally, but it still
needs final style integration with the full Figure 4 composite.

Primary source:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    observer_summary.csv
```

Action:

```text
Choose which Panel C subpanels enter the compressed figure style and decide
whether posterior N_eff and compact-mechanism panels stay main or supplement.
```

Generated assets:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/
declan/figure4_active_sensing_atlas/figures/panel_C/C1_observer_schematic.png
declan/figure4_active_sensing_atlas/figures/panel_C/C2_accuracy_ordering.png
declan/figure4_active_sensing_atlas/figures/panel_C/C3_matched_static_rescue.png
declan/figure4_active_sensing_atlas/figures/panel_C/C4_posterior_concentration.png
declan/figure4_active_sensing_atlas/figures/panel_C/C5_scale_gap_guardrail.png
declan/figure4_active_sensing_atlas/figures/panel_C/C6_compact_mechanism_guardrail.png
declan/figure4_active_sensing_atlas/figures/panel_C_joint_observer_accuracy.png
declan/figure4_active_sensing_atlas/figures/panel_C_joint_observer_accuracy.pdf
declan/figure4_active_sensing_atlas/figures/panel_C_joint_observer_values.csv
declan/figure4_active_sensing_atlas/figures/panel_C_joint_observer_accuracy_caption.md
```

### F012: Behavioral Alignment Metric Convention Needs To Be Chosen

Module: E.

Two valid behavior summaries are currently in play:

```text
1. Current headline figure metric:
   weighted all-window edge-axis cos2 = 0.181
   session-bootstrap CI = [0.124, 0.241]

2. Distribution-inspection metric:
   unweighted all-window session mean cos2 = 0.105
   CI = [0.067, 0.145]
   reliable-axis session mean cos2 = 0.140
   CI = [0.089, 0.188]
```

They come from the same broad behavior family but use different weighting and
summary conventions. This is not a fatal inconsistency, but it needs to be made
explicit.

Recommended handling:

```text
Use the unweighted session-mean distribution-inspection metric in atlas prose
because it is easiest to describe as a behavioral estimate. Use the weighted
headline metric only when directly referencing the existing rendered headline
figure or its stats manifest.
```

Generated guardrail assets:

```text
declan/figure4_active_sensing_atlas/figures/panel_E/E4_metric_convention_guardrail.png
declan/figure4_active_sensing_atlas/figures/panel_E/panel_E_metric_convention_values.csv
```
