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

## 2026-06-21: Provisional Figure 4 Contract And Companion Documents

Built a cache-first provisional Figure 4 package from the existing atlas
documents, provenance ledger, diagnostics queue, and current canonical
active-sensing status note. The companion documents follow the explanatory
methods-note style of `declan/inhomogenous stimuli writeup.pdf`: each starts
from the broken simplifying assumption, defines notation and an estimator
contract, names assumptions and controls, then separates supported claims from
attractive but unsupported interpretations.

Outputs:

```text
declan/figure4_active_sensing_atlas/provisional_panel_contract_v0.csv
declan/figure4_active_sensing_atlas/provisional_figure4_v0.md
declan/figure4_active_sensing_atlas/companion_aggregate_fem_model.md
declan/figure4_active_sensing_atlas/companion_local_Iz_pairing_model.md
declan/figure4_active_sensing_atlas/companion_joint_posterior_observer_model.md
declan/figure4_active_sensing_atlas/companion_behavior_geometry_bridge.md
```

Read:

```text
The provisional main figure should emphasize aggregate feature-decodable
motion structure, matched-static joint-observer rescue, edge-parallel
preservation, and measured contour-following behavior. It should not claim
that measured FEMs optimize the tested model objective. The local I_z branch is
best treated as mechanistic sensitivity: the current pyramid k16 delta_mean
rows are useful, but matched-unpaired and rotated controls keep the exact
pairing claim provisional.
```

Remaining work:

```text
Promote or replace provisional values after the guarded canonical active-sensing
power reruns and canonical raw-edge residual adjudication close.
Compose final visual panels from the selected atlas subpanels.
```

## 2026-06-21: Figure 4A Single-Panel Promotion Candidates

Built a new review surface for choosing one promoted 4A panel. This supersedes
the earlier combined-layout option-sheet interpretation for 4A selection: the
current question is which single panel to promote, with A1 as the leading
design.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_panel_a_single_panel_candidates.py
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_A/promotion_candidates/
  4A_candidate_0_current_A1_reference.png
  4A_candidate_1_real_backimage_a1_proportions.png
  4A_candidate_2_real_backimage_context.png
  4A_candidate_3_real_high_contrast_positive.png
  4A_single_panel_candidate_sheet.png
  4A_single_panel_candidate_values.csv
  README.md
```

Read:

```text
Selected provisional 4A is candidate 3:
4A_candidate_3_real_high_contrast_positive.png. Candidate 1 kept the original
A1 proportions but was centered on a dark patch. Candidate 3 uses a real
BackImage canvas crop and recorded fixation trace, has clearer high-contrast
retinal samples, and has positive drift-edge alignment metadata.
```

Provenance:

```text
Real variants call declan.fixation_statistics_by_stimulus.image_features._backimage_canvas
and use recorded eyepos slices from each session's backimage.dset, indexed by
global_start:global_stop in backimage_image_fem_windows.csv.
```

## 2026-06-21: Figure 4B Single-Panel Promotion Candidates

Built a review surface for choosing one promoted aggregate-FEM panel.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_panel_b_single_panel_candidates.py
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_B/promotion_candidates/
  4B_candidate_1_gain_over_static_audited.png
  4B_candidate_2_empirical_minus_controls.png
  4B_candidate_3_absolute_gain_guardrail.png
  4B_candidate_4_k16_tworeadout_preview.png
  4B_single_panel_candidate_sheet.png
  4B_single_panel_candidate_values.csv
  README.md
```

Read:

```text
Candidate 3 is now the selected provisional Panel B: a B5-style absolute-family
guardrail with OU omitted. It keeps the empirical positive-gain result visible,
shows Brownian catching up at larger scales, and preserves the suggestive
rotated bump near 1x as a possible local alignment bonus rather than a global
contour-alignment scale optimum. The original OU condition is excluded because
its below-static absolute gain is a red flag for trace generation or analysis
bookkeeping, not a promoted scientific read. Candidate 1 remains a clean
gain-over-static headline. Candidate 4 best matches the current provisional
aggregate target, pyramid_local_field k16 temporal_pca, but should remain a
preview until the guarded production rerun is promoted.
```

## 2026-06-21: Figure 4C Single-Panel Promotion Candidates

Built a review surface for choosing one promoted joint-observer panel.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_panel_c_single_panel_candidates.py
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/
  4C_candidate_1_matched_static_rescue_current.png
  4C_candidate_2_empirical_prior_rescue_clean.png
  4C_candidate_3_accuracy_ordering_context.png
  4C_candidate_4_scale_gap_guardrail.png
  4C_single_panel_candidate_sheet.png
  4C_single_panel_candidate_values.csv
  README.md
```

Read:

```text
Candidate 2 is selected provisionally for Panel C: under matched-static
distractors, zero-eye accuracy is 0.328, empirical-prior joint-eye accuracy is
0.766, and known-eye accuracy is 1.000, so the joint observer recovers 65% of
the known-minus-zero gap. Candidate 1 is more faithful to the full current
contract because it also shows the OU-prior robustness check at 0.797; keep
that value in the caption, supplement, or companion prose. Candidate 3 gives
broader hard-negative and matched-static context, while candidate 4 is a
scale/zero-eye-failure guardrail.
```

## 2026-06-21: Figure 4D Single-Panel Promotion Candidates

Built a review surface for choosing one promoted local image-axis mechanism
panel.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_panel_d_single_panel_candidates.py
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_D/promotion_candidates/
  4D_candidate_1_edge_parallel_preservation.png
  4D_candidate_2_axis_conditioned_rescue.png
  4D_candidate_3_axis_preference_guardrail.png
  4D_candidate_4_raw_edge_objective_guardrail.png
  4D_single_panel_candidate_sheet.png
  4D_single_panel_candidate_values.csv
  README.md
```

Read:

```text
Candidate 1 is selected provisionally for Panel D: edge-parallel
displacement has lower local pixel and V1-twin cost than matched orthogonal
displacement. Pixel preservation advantage is 300.54 with CI
[172.789, 408.961] and 26/29 positive sessions; twin preservation advantage is
0.000454497 with CI [0.000371047, 0.000536519] and 29/29 positive sessions.
Candidate 2 connects local image axes to observer rescue but has mixed
parallel-versus-orthogonal ordering. Candidate 3 foregrounds that mixed axis
preference; candidate 4 foregrounds the raw-edge/objective guardrail.
```

## 2026-06-21: Figure 4E Single-Panel Promotion Candidates

Built a review surface for choosing one promoted behavior-geometry bridge
panel.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_panel_e_single_panel_candidates.py
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_E/promotion_candidates/
  4E_candidate_1_alignment_strength.png
  4E_candidate_2_parallel_zone_enrichment.png
  4E_candidate_3a_image_coherence_focus.png
  4E_candidate_3b_fem_anisotropy_focus.png
  4E_candidate_3c_polar_alignment_rose.png
  4E_candidate_5_confidence_dependence_full.png
  4E_candidate_6_endpoint_null_diagnostic.png
  4E_single_panel_candidate_sheet.png
  4E_single_panel_candidate_values.csv
  README.md
```

Read:

```text
Candidate 1 is the clean statistical headline: free-viewing FEM axes align
modestly but reliably with local edges, and alignment strengthens in
high-confidence windows. Candidate 2 is the more intuitive endpoint-zone view:
parallel endpoint directions are enriched and orthogonal directions are
depleted. After review, candidate 3A and 3B were split out from the dense E7
diagnostic. Candidate 3A carries the image-orientation-coherence message most
directly: FEM-edge alignment rises as the local image axis becomes coherent.
Candidate 3B is the paired FEM-anisotropy reliability view. Candidate 3C is a
polar/rose option showing the high-confidence concentration around the
edge-parallel axis relative to a uniform axial baseline. Candidates 5-6 are
dense confidence/null diagnostics.
```

## 2026-06-21: Selected Provisional Figure 4 Composite

Built the first selected A-E Figure 4 composite from the promoted single-panel
candidates, with Panel E set to candidate 3A.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_selected_figure4.py
```

Inputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_A/promotion_candidates/4A_candidate_3_real_high_contrast_positive.png
declan/figure4_active_sensing_atlas/figures/panel_B/promotion_candidates/4B_candidate_3_absolute_gain_guardrail.png
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/4C_candidate_2_empirical_prior_rescue_clean.png
declan/figure4_active_sensing_atlas/figures/panel_D/promotion_candidates/4D_candidate_1_edge_parallel_preservation.png
declan/figure4_active_sensing_atlas/figures/panel_E/promotion_candidates/4E_candidate_3a_image_coherence_focus.png
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v0.png
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v0.pdf
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v0_manifest.csv
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v0_caption.md
```

Read:

```text
The selected composite is A: real BackImage retinal-movie premise, B: no-OU
absolute gain guardrail, C: empirical-prior matched-static rescue, D:
edge-parallel preservation, and E: image-orientation-coherence behavior
alignment. It is a provisional layout draft; canonical reruns and raw-edge
residual adjudication can still replace values before manuscript promotion.
```

## 2026-06-21: Selected Figure 4 Composite v1, Tangent-Geometry D

Revised the selected composite after review to include the key D message that
local tangent/axis geometry rescues joint image decoding without known eye
position. The previous v0 composite used the edge-parallel preservation audit
as Panel D; v1 promotes the axis-conditioned rescue panel and keeps
edge-parallel preservation as support/caption material.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_panel_d_single_panel_candidates.py
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_selected_figure4.py
```

Inputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_A/promotion_candidates/4A_candidate_3_real_high_contrast_positive.png
declan/figure4_active_sensing_atlas/figures/panel_B/promotion_candidates/4B_candidate_3_absolute_gain_guardrail.png
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/4C_candidate_2_empirical_prior_rescue_clean.png
declan/figure4_active_sensing_atlas/figures/panel_D/promotion_candidates/4D_candidate_2_axis_conditioned_rescue.png
declan/figure4_active_sensing_atlas/figures/panel_E/promotion_candidates/4E_candidate_3a_image_coherence_focus.png
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v1.png
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v1.pdf
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v1_manifest.csv
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v1_caption.md
```

Read:

```text
Panel D now shows that axis-conditioned tangent/normal geometry priors rescue
image-identification accuracy above the zero-eye observer without known eye
position. The tangent/edge-parallel prior reaches 0.859 versus zero-eye 0.641
on matched-static 0.5x, and 0.813 versus zero-eye 0.609 on hard negatives 0.5x.
At larger hard-negative scales both axis priors remain above zero-eye, but the
parallel-versus-orthogonal ordering flips, so the promoted claim is
tangent/axis-geometry rescue rather than universal edge-parallel optimality.
```

## 2026-06-21: Selected Figure 4 Composite v2, Feature-Posterior Joint Model

Checked the newer joint-model analysis and revised the selected composite away
from image-identity accuracy. Panel C now promotes the feature-posterior
endpoint: joint posterior feature recovery above zero-eye without known eye
position. Panel D returns to the edge-parallel preservation mechanism support.

Commands:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_joint_feature_posterior_panel.py
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_panel_c_single_panel_candidates.py
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_panel_d_single_panel_candidates.py
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_selected_figure4.py
```

Inputs:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/
    feature_posterior_summary.csv

outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_feature_decomposition_adjudication_joint_k2_4_8_16_32_v1/
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/4C_candidate_5_joint_feature_posterior_recovery.png
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/4C_candidate_5_joint_feature_posterior_recovery_values.csv
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v2.png
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v2.pdf
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v2_manifest.csv
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v2_caption.md
```

Read:

```text
The primary promoted C panel uses hard-negative n128 feature posterior rows for
pyramid_local_field k=8. Joint-minus-zero feature recovery is positive with
positive CIs for both edge-parallel and edge-orthogonal priors at 0.5x, 1x, and
2x. Edge-parallel is slightly higher at 0.5x and 1x, while edge-orthogonal is
higher at 2x; axis ordering is therefore not promoted as a universal
edge-parallel win. The robust claim is feature recovery by the latent-eye joint
posterior.
```

## 2026-06-21: Selected Figure 4 Composite v3, Story-First Communication Pass

Rebuilt the selected composite to put the panel take-away ahead of the
implementation details. The data choices are unchanged from v2, but the panel
titles, selected plot titles, and axes now use reader-facing language:
one image becomes a movie, motion adds feature information, features survive
hidden eye position, along-edge motion preserves structure, and real drift
follows coherent edges.

Commands:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_panel_a_single_panel_candidates.py
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_panel_b_single_panel_candidates.py
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_joint_feature_posterior_panel.py
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/plot_panel_d_subpanels.py
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_panel_d_single_panel_candidates.py
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_panel_e_single_panel_candidates.py
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_selected_figure4.py
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v3.png
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v3.pdf
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v3_manifest.csv
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v3_caption.md
```

Read:

```text
The v3 composite is a communication draft, not a new analysis revision. It
keeps the feature-posterior C panel and preservation-mechanism D panel from v2,
but reduces acronym-heavy titles and method-facing axis labels so the figure
arc is visible before the reader studies the implementation details.
```

## 2026-06-21: Selected Figure 4 Composite v4, Design-First Layout

Built a design-first composite instead of pasting the standalone promotion
candidate PNGs. The analysis choices remain A3, B3, C5, D1, and E3A, but B-E
are redrawn from source tables in one shared visual grammar and A is cropped as
the opening movie schematic. This version tests whether the layout itself can
serve the story rather than merely arranging finished panels.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_selected_figure4_v4_design.py
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v4.png
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v4.pdf
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v4_manifest.csv
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v4_caption.md
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v4_panel_b_values.csv
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v4_panel_c_values.csv
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v4_panel_d_values.csv
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v4_panel_e_values.csv
```

Read:

```text
The v4 composite is the current design draft. It preserves the v3 story but
reduces duplicated titles, harmonizes plot typography and colors, gives the
movie schematic a dedicated opening band, and pairs B/C as model evidence and
D/E as mechanism-to-behavior closure.
```

## 2026-06-21: Selected Figure 4 Composite v5, Compact A/B Top Row

Moved Panel B into the top row with Panel A so the opening read is immediately
retinal movie -> feature gain. Compressed C, D, and E into a lower row with
shorter compact headers: hidden-eye recovery, along-edge preservation, and drift
following clear edges. The analysis values are unchanged from v4.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_selected_figure4_v5_compact_layout.py
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v5.png
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v5.pdf
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v5_manifest.csv
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v5_caption.md
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v5_panel_b_values.csv
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v5_panel_c_values.csv
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v5_panel_d_values.csv
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v5_panel_e_values.csv
```

Read:

```text
The v5 compact layout is the current selected design draft. It keeps the
story-first v4 visual grammar, but uses horizontal space more aggressively:
A+B form the first-row premise/payoff, and C/D/E close the figure in a compact
inference-mechanism-behavior row. Later polish passes tightened row spacing,
quieted Panel E's secondary window-count axis, and cropped Panel A more tightly
so the retinal-movie elements fill the opening panel.
```

## 2026-06-21: Panel C Reframed To Absolute Feature Recovery

Replaced the C-panel gain-over-zero plot with absolute feature recovery cosine
for zeroed-eye and compact-subspace inference across the three motion scales. The
previous joint-minus-zero feature gain made the rising curve look like a
"larger motion is better" result, even though the zeroed-eye baseline is being
increasingly disrupted by motion scale. The new plot makes the intended read
explicit: zeroed-eye recovery falls with motion scale, while compact-subspace
recovery stays stable around 0.87.

Commands:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_joint_feature_posterior_panel.py
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_panel_c_single_panel_candidates.py
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_selected_figure4_v5_compact_layout.py
```

Read:

```text
Pyramid local-field k=8, hard-negative feature posterior:
zeroed-eye cosine falls from 0.765 at 0.5x to 0.576 at 2x, while
compact-subspace joint cosine stays approximately flat at 0.872, 0.872, and
0.871. The two compact sources are no longer the promoted visual contrast;
they are retained only as the small spread around the compact-subspace line.
```

## 2026-06-21: Panel C Feature-Recovery Option Sheet

Built a focused option sheet for the revised Panel C, using the newer
feature-posterior endpoint rather than the older image-identity observer. The
sheet now separates the clean main-panel read from the compact-mechanism
necessity audit.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_panel_c_feature_recovery_options.py
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/feature_recovery_options/4C_feature_recovery_option_sheet.png
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/feature_recovery_options/4C_option_1_zeroed_vs_compact_subspace.png
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/feature_recovery_options/4C_option_2_compact_sources_explicit.png
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/feature_recovery_options/4C_option_3_observer_scale_heatmap.png
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/feature_recovery_options/4C_option_4_scale_robustness.png
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/feature_recovery_options/4C_option_5_compact_subspace_rescue.png
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/feature_recovery_options/4C_option_6_compact_necessity_audit.png
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/feature_recovery_options/4C_feature_recovery_option_values.csv
```

Read:

```text
Options 1 and 5 are the cleanest feature-posterior main-panel candidates.
Option 1 promotes the guarded result that the compact subspace stabilizes
latent-eye feature recovery; Option 5 promotes the compact-subspace rescue
directly while omitting the second compact-source comparator. Option 6 uses the
newer compact-mechanism joint-decoding audit: compact-only k=10 recovers about
60%, 64%, and 69% of the full joint-minus-zero accuracy gain at 0.5x, 1x, and
2x, while compact-removed recovers approximately -25%, 3%, and -1%, collapsing
to the zero-eye baseline.

Important boundary: Option 6 uses joint image-decoding accuracy, not the
feature-recovery cosine endpoint used by Options 1-5 and the current Panel C
candidate. A feature-space compact-only / compact-removed / addback
decomposition is still required before claiming that the compact subspace is
necessary for the promoted feature-posterior recovery result.
```

## 2026-06-21: Panel C Option 5 Installed In Composite

Updated the selected v5 composite so Panel C uses option 5 from the
feature-recovery option sheet: zeroed eye, compact subspace, and known-eye
ceiling. The second compact-source comparator is omitted from the main panel,
and the compact-removal audit remains a pending feature-space follow-up rather
than a main-panel claim.

Command:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_selected_figure4_v5_compact_layout.py
```

Outputs:

```text
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v5.png
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v5.pdf
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v5_panel_c_values.csv
```

Read:

```text
Panel C now reports zeroed-eye feature recovery cosine 0.765, 0.668, and
0.576 across 0.5x, 1x, and 2x; compact-subspace recovery 0.877, 0.877, and
0.871; and known-eye ceiling 0.927, 0.936, and 0.949. This is the option 5
main-panel read, not the averaged compact-source line or the joint-decoding
compact-removal audit.
```

## 2026-06-21: Panel B Title Clarified

Updated the selected v5 composite Panel B header to:

```text
Motion supports feature encoding
when exact eye-trajectory is known to the model
```

This keeps Panel B as the exact-eye-trajectory feature-encoding result and
reserves Panel C for hidden-eye inference.

## 2026-06-21: Panel B Power Rerun Integrated

Panel B is now redrawn from:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/
    incremental_static_plus_motion_tworeadout_v1/
```

The selected aggregate readout is `pyramid_local_field k16 temporal_pca`.
Recorded drift is above static across the displayed scale sweep, OU is below
static, and Brownian/rotated controls remain above static enough that
empirical-motion specificity should be stated as strongest at small scale
rather than universal. The selected standalone Panel B asset is now:

```text
declan/figure4_active_sensing_atlas/figures/panel_B/promotion_candidates/4B_candidate_3_power_rerun_absolute_gain.png
```

Correction, later on 2026-06-21:

```text
This entry is superseded for current Panel B interpretation. The corrected
static-mean posthoc is incremental_staticmean_plus_motion_tworeadout_v2, and
the all-readout audit lives under
incremental_staticmean_plus_motion_allreadouts_v1/readout_atlas_figures/.
Current roles: mean/delta_mean for absolute aggregate gain, delta_mean for
local mechanistic sensitivity, and temporal PCA/DCT for order-sensitive
empirical-vs-control diagnostics. OU is audit-pending before headline use.
```

## 2026-06-21: Panel C Feature-Space Compact Removal Installed

Ran the compact-subspace intervention through the promoted feature-posterior
metric for hard-negative n128, `pyramid_local_field` PCA k=8, compact k=10.
This supersedes the earlier Panel C caveat that compact-removal evidence was
only available from image-identity accuracy.

Analysis command:

```bash
.venv/bin/python -m declan.backimage_trajectory_observer.analyze_feature_posterior_compact_mechanism \
  --run-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1 \
  --out-dir outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1 \
  --feature-npz outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/feature_latent_arrays.npz \
  --compact-basis-path outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz \
  --basis-mode image_disjoint \
  --candidate-set-modes hard_negative_structure \
  --priors axis_edge_parallel,axis_edge_orthogonal \
  --motion-scales 0.5,1.0,2.0 \
  --likelihood-scales 1.0 \
  --latent-names pyramid_local_field \
  --pca-k-list 8 \
  --k-dims 10 \
  --variants full_exact,zero_static,compact_only,compact_removed,compact_addback \
  --reference-feature-summary outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/feature_posterior_summary.csv \
  --n-bootstrap 10000 \
  --n-permutations 10000 \
  --uncertainty-seed 17 \
  --progress-every 16
```

Regenerated:

```bash
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_joint_feature_posterior_panel.py
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_panel_c_feature_recovery_options.py
.venv/bin/python declan/figure4_active_sensing_atlas/scripts/build_selected_figure4_v5_compact_layout.py
```

Outputs:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1/
    feature_compact_mechanism_summary.csv
    feature_compact_mechanism_uncertainty.csv
    feature_compact_mechanism_qc.csv
    feature_compact_mechanism_report.md
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/4C_candidate_5_joint_feature_posterior_recovery.png
declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/feature_recovery_options/4C_option_6_compact_necessity_audit.png
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v5.png
declan/figure4_active_sensing_atlas/figures/composites/figure4_selected_v5_panel_c_values.csv
```

Read:

```text
Feature recovery cosine averaged across the two compact-source priors:
zeroed eye = 0.765, 0.668, 0.576 at 0.5x, 1x, 2x.
compact-only = 0.850, 0.838, 0.826.
compact-removed = 0.759, 0.635, 0.537.
full joint / compact-addback = 0.872, 0.872, 0.871.
known-eye ceiling = 0.927, 0.936, 0.949.

Validation: compact-addback reconstructs full responses with max error
3.47e-18, and full/zero/known summaries match the prior feature-posterior
endpoint with max summary delta 3.55e-15.
```

Claim boundary:

```text
The selected Panel C can now say that compact-only retains much of full joint
feature recovery and compact removal collapses recovery toward zeroed-eye in
the promoted feature-posterior metric. This does not prove that animals compute
the posterior, that the compact subspace is unique, or that behavior optimizes
this model objective.
```
