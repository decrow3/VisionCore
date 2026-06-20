# Supplement Routing v0

Status: first supplement routing pass.

## Supplement S4.1: Retinal Rendering And Movie QC

Purpose:

```text
Show that the retinal movie construction is well defined and matched to model
inputs.
```

Sources:

```text
declan/figure4_active_sensing_atlas/figures/panel_A/
outputs/active_sensing_movie_information/active_sensing_movie_information_figure_frozen_20260615_pre_backimage_collab_pack/retinal_movie_transform_qc.png
outputs/active_sensing_movie_information/active_sensing_movie_information_figure_frozen_20260615_pre_backimage_collab_pack/retinal_movie_transform_qc.csv
outputs/active_sensing_movie_information/active_sensing_movie_information_figure_frozen_20260615_pre_backimage_collab_pack/retinal_movie_transform_qc_manifest.json
outputs/active_sensing_movie_information/reafferent_variance_accounting/variance_accounting_aggregate_summary.csv
```

Flags:

```text
F002
```

## Supplement S4.2: Motion Family Matching And Decoder Sanity

Purpose:

```text
Document effective RMS, clipping, trace reuse, absolute gains, and fixed-alpha
decoder settings for the aggregate FEM-information result.
```

Sources:

```text
outputs/fig4_active_sensing/sanity_checks/fig4_active_sensing_sanity_report.md
outputs/fig4_active_sensing/sanity_checks/motion_metric_sanity_panels.png
outputs/fig4_active_sensing/sanity_checks/absolute_gains_gabor_local_field_k4_temporal_pca.png
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/aggregate_motion_metadata.csv
```

Flags:

```text
F003
F004
```

## Supplement S4.3: Local Exact-Pairing Revisit

Purpose:

```text
Keep exact image-trace pairing separate from the distributional aggregate
result.
```

Sources:

```text
declan/backimage_local_pairing_Iz_revisit_plan.md
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_gabor_pyramid_rel025_1_seed7_v1/
```

Flags:

```text
F005
```

## Supplement S4.4: Observer Mechanics And Posterior Diagnostics

Purpose:

```text
Explain the exact finite trajectory-table observer, posterior concentration,
and image-condition diagnostics.
```

Sources:

```text
declan/backimage_trajectory_observer/results_log.md
declan/backimage_trajectory_observer/backimage_trajectory_table_observer_prescription.md
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/posthoc_image_condition_analysis/
```

Flags:

```text
F011
```

## Supplement S4.5: Compact Mechanism Bridge

Purpose:

```text
Test whether compact translation geometry carries the motion-dependent
likelihood structure in the exact observer.
```

Sources:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/compact_mechanism_image_disjoint_fold0_n512_k2_5_10_20_rand8_log_v1/
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/compact_mechanism_image_disjoint_fold0_n512_k2_5_10_20_rand8_log_v1/followup_summary/compact_mechanism_promotion_gates.csv
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/compact_mechanism_image_disjoint_fold0_n512_k2_5_10_20_rand8_log_v1/followup_summary/compact_staticpc_basis_overlap.csv
```

Flags:

```text
F006
```

## Supplement S4.6: Axis-Conditioned Observer

Purpose:

```text
Show that axis-conditioned trajectory priors rescue image identity, while axis
preference itself is mixed.
```

Sources:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1/observer_summary.csv
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/observer_summary.csv
declan/axis_conditioned_backimage_trajectory_observer/README.md
```

Flags:

```text
F007
```

## Supplement S4.7: Behavior Controls And Objective Comparison

Purpose:

```text
Show the full behavioral drift-edge distribution, endpoint-zone enrichment,
random/null controls, and model objective comparison.
```

Sources:

```text
declan/figure4_active_sensing_atlas/figures/panel_E/
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_edge_alignment_distribution_inspection/
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_conditional_fixation_objectives_twin_axis_only_n256/alignment_by_objective_summary.csv
```

Flags:

```text
F009
F012
```
