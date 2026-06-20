# Figure Build Log

## 2026-06-19: Panel A Subpanels

Built cache-only Panel A premise/QC subpanels from existing retinal-movie QC,
Figure 4 headline stats, and reafferent-variance accounting tables.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/plot_panel_a_subpanels.py
```

Inputs:

```text
outputs/active_sensing_movie_information/
  active_sensing_movie_information_figure_frozen_20260615_pre_backimage_collab_pack/
    retinal_movie_transform_qc_summary.csv
outputs/fig4_active_sensing/active_sensing_headline_figure/
  fig4_active_sensing_headline_stats.json
outputs/active_sensing_movie_information/reafferent_variance_accounting/
  variance_accounting_aggregate_summary.csv
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_A/A1_retinal_movie_transform.png
declan/figure4_active_sensing_atlas/figures/panel_A/A2_movie_transform_qc.png
declan/figure4_active_sensing_atlas/figures/panel_A/A3_gradient_sampling_cartoon.png
declan/figure4_active_sensing_atlas/figures/panel_A/A4_backimage_pipeline_bridge.png
declan/figure4_active_sensing_atlas/figures/panel_A/A5_covariance_bridge_guardrail.png
declan/figure4_active_sensing_atlas/figures/panel_A/panel_A_*_values.csv
declan/figure4_active_sensing_atlas/figures/panel_A/panel_A_subpanels_caption.md
```

Read:

```text
Panel A now teaches the physical premise and source provenance: a fixed screen
image becomes shifted retinal samples under FEMs; FEM movies add temporal
contrast and motion power relative to stabilization; and downstream B-E panels
use the canonical BackImage 756-unit V1-twin pathway.
```

Remaining work:

```text
Keep F002 visible until a full Figure 4 composite exists. A5 is a covariance
bridge only: the reafferent-variance accounting rows use mixed denominators
and should not be promoted as the main Figure 4 endpoint.
```

## 2026-06-19: Panel E Subpanels

Built cache-only Panel E subpanels from free-viewing BackImage image-geometry
and drift-edge alignment outputs.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/plot_panel_e_subpanels.py
```

Inputs:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_image_structure_reviewed_v2_screenfiltered_yfix/
    backimage_image_fem_windows.csv
    orientation_alignment_summary.csv
  backimage_edge_alignment_distribution_inspection/
    edge_alignment_distribution_summary.csv
    endpoint_zone_enrichment_summary.csv
    edge_alignment_window_and_session_distributions.png
    edge_alignment_confidence_and_signed_delta.png
    edge_alignment_endpoint_null_diagnostic.png
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_E/E1_behavior_setup_example.png
declan/figure4_active_sensing_atlas/figures/panel_E/E2_behavior_alignment_strength.png
declan/figure4_active_sensing_atlas/figures/panel_E/E3_parallel_zone_enrichment.png
declan/figure4_active_sensing_atlas/figures/panel_E/E6_full_distribution_session_diagnostic.png
declan/figure4_active_sensing_atlas/figures/panel_E/E7_confidence_signed_delta_diagnostic.png
declan/figure4_active_sensing_atlas/figures/panel_E/E8_endpoint_null_diagnostic.png
declan/figure4_active_sensing_atlas/figures/panel_E/E4_metric_convention_guardrail.png
declan/figure4_active_sensing_atlas/figures/panel_E/E5_scope_summary.png
declan/figure4_active_sensing_atlas/figures/panel_E/panel_E_*_values.csv
declan/figure4_active_sensing_atlas/figures/panel_E/panel_E_contour_following_source_panels.csv
declan/figure4_active_sensing_atlas/figures/panel_E/panel_E_subpanels_caption.md
```

Read:

```text
Free-viewing drift/FEM axes align modestly but reliably with local edge
geometry: all-window session mean cos2 = 0.105, reliable-axis mean = 0.140,
and high-confidence mean = 0.269. The parallel endpoint zone is enriched
relative to a uniform angular expectation, especially in high-confidence
windows.

2026-06-20 update: E3 is now explicitly treated as a compact redraw of the
endpoint summary. E6-E8 copy the original distribution, confidence, and
endpoint/null diagnostics into the atlas so the contour-following behavior
metric can be reviewed with its provenance visible.
```

Remaining work:

```text
Keep F009/F012 visible. The behavioral result supports image-contingent FEM
geometry, but weighted and unweighted summaries differ and current V1-twin
response-objective models do not yet beat raw edge geometry.
```

## 2026-06-19: Panel D Subpanels

Built cache-only Panel D subpanels from axis-conditioned observer,
edge-parallel stability, and objective-alignment outputs.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/plot_panel_d_subpanels.py
```

Inputs:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1/
    observer_summary.csv
  backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/
    observer_summary.csv
  backimage_edge_parallel_stability_screen_yfix_n256_pop256/
    stability_summary.csv
  backimage_conditional_fixation_objectives_twin_axis_only_n256/
    paired_session_deltas_vs_raw_edge.csv
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_D/D1_local_axis_schematic.png
declan/figure4_active_sensing_atlas/figures/panel_D/D2_axis_conditioned_accuracy.png
declan/figure4_active_sensing_atlas/figures/panel_D/D3_axis_preference_guardrail.png
declan/figure4_active_sensing_atlas/figures/panel_D/D4_edge_parallel_stability.png
declan/figure4_active_sensing_atlas/figures/panel_D/D5_objective_alignment_guardrail.png
declan/figure4_active_sensing_atlas/figures/panel_D/panel_D_*_values.csv
declan/figure4_active_sensing_atlas/figures/panel_D/panel_D_subpanels_caption.md
```

Read:

```text
Axis-conditioned priors rescue image identity above zero-eye, but the preferred
axis is condition-dependent: parallel beats orthogonal by +0.031 at matched
static 0.5x and hard-negative 0.5x, while orthogonal is ahead at hard-negative
1.0x and 2.0x. Separately, edge-parallel displacement reduces pixel and V1-twin
costs relative to matched orthogonal displacement.
```

Remaining work:

```text
Keep F007/F008/F009 visible. Panel D supports image-conditioned axis structure
and local edge-parallel preservation, not a universal motion policy or a
settled V1-twin objective that beats raw image geometry.
```

## 2026-06-19: Panel C Subpanels

Built cache-only Panel C subpanels from the matched-static BackImage
trajectory-table observer output and compact-mechanism followup.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/plot_panel_c_subpanels.py
```

Inputs:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    observer_summary.csv
    compact_mechanism_image_disjoint_fold0_n512_k2_5_10_20_rand8_log_v1/
      followup_summary/compact_mechanism_promotion_gates.csv
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/C1_observer_schematic.png
declan/figure4_active_sensing_atlas/figures/panel_C/C2_accuracy_ordering.png
declan/figure4_active_sensing_atlas/figures/panel_C/C3_matched_static_rescue.png
declan/figure4_active_sensing_atlas/figures/panel_C/C4_posterior_concentration.png
declan/figure4_active_sensing_atlas/figures/panel_C/C5_scale_gap_guardrail.png
declan/figure4_active_sensing_atlas/figures/panel_C/C6_compact_mechanism_guardrail.png
declan/figure4_active_sensing_atlas/figures/panel_C/panel_C_*_values.csv
declan/figure4_active_sensing_atlas/figures/panel_C/panel_C_subpanels_caption.md
```

Read:

```text
At matched-static 1.0x and likelihood scale 1.0, zero-eye accuracy was 0.328,
joint-eye accuracy was 0.766 with the empirical prior and 0.797 with the OU
prior, and known-eye accuracy was 1.000. The joint observer recovered about
65%-70% of the known-zero gap. Median N_eff / K was 0.364-0.400, consistent
with partial posterior concentration rather than exact trajectory recovery.
```

Remaining work:

```text
Choose which C subpanels belong in the compressed main figure. Compact
projection should stay framed as a mechanism guardrail rather than unique
mechanism proof.
```

## 2026-06-19: Panel B Subpanels

Built cache-only Panel B subpanels from the cleaned BackImage aggregate
FEM-information output.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/plot_panel_b_subpanels.py
```

Inputs:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
    aggregate_motion_summary.csv
    incremental_static_plus_motion_relids/
      incremental_gain_vs_static.csv
      incremental_gain_contrasts.csv
      incremental_decode_summary.csv
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_B/B1_task_schematic.png
declan/figure4_active_sensing_atlas/figures/panel_B/B2_motion_family_qc.png
declan/figure4_active_sensing_atlas/figures/panel_B/B3_empirical_gain_vs_static.png
declan/figure4_active_sensing_atlas/figures/panel_B/B4_empirical_minus_controls.png
declan/figure4_active_sensing_atlas/figures/panel_B/B5_absolute_gain_guardrail.png
declan/figure4_active_sensing_atlas/figures/panel_B/panel_B_*_values.csv
declan/figure4_active_sensing_atlas/figures/panel_B/panel_B_subpanels_caption.md
```

Read:

```text
Empirical temporal-PCA response movies add feature-decoding gain over static
responses for both Gabor k=4 and pyramid k=8 targets. Empirical beats the
OU-like control robustly; the Brownian and rotated specificity is clearest at
0.25x-0.5x and narrows at larger scales.
```

Remaining work:

```text
Integrate the chosen B subpanels into the eventual compressed figure style.
Keep F003/F004 visible because these are deterministic -MSE decoding proxies
and Brownian/rotated controls catch up at larger scales.
```

## 2026-06-19: Panel C Joint Observer

Built a cache-only joint-observer accuracy panel from the matched-static
BackImage trajectory-table observer output.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/plot_joint_observer_panel.py
```

Input:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
    observer_summary.csv
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_C_joint_observer_accuracy.png
declan/figure4_active_sensing_atlas/figures/panel_C_joint_observer_accuracy.pdf
declan/figure4_active_sensing_atlas/figures/panel_C_joint_observer_values.csv
declan/figure4_active_sensing_atlas/figures/panel_C_joint_observer_accuracy_caption.md
```

Read:

```text
At matched-static 1.0x, known-eye accuracy was 1.000, zero-eye was 0.328,
joint-eye was 0.766 with the empirical prior and 0.797 with the OU prior.
```

Remaining work:

```text
Integrate the panel into the full Figure 4 visual style and decide whether
posterior N_eff belongs as a main inset or supplement.
```
