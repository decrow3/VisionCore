# Panel Source Map

This map links each proposed atlas module to existing repo assets. It is a
working map, not a final manuscript claim list.

| Module | Panel Role | Existing Code Or Notes | Existing Results Or Caches | Current Use |
| --- | --- | --- | --- | --- |
| A | Retinal movie premise, stabilized versus FEM movie, rendering QC | `declan/fig4_active_sensing/`, `declan/active_sensing_movie_information/`, `declan/active_sensing_movie_information/generate_retinal_movie_transform_qc.py` | `outputs/active_sensing_movie_information/active_sensing_movie_information_figure_frozen_20260615_pre_backimage_collab_pack/retinal_movie_transform_qc.*`, `outputs/fig4_active_sensing/active_sensing_headline_figure/` | Main explanatory/cartoon/QC source |
| A | Generated atlas subpanels | `declan/figure4_active_sensing_atlas/scripts/plot_panel_a_subpanels.py` | `declan/figure4_active_sensing_atlas/figures/panel_A/` | Cache-only Panel A premise/QC assets |
| A | Covariance bridge | `declan/fig4_cov_TFTS/`, `declan/active_sensing_movie_information/summarize_reafferent_variance_accounting.py` | `outputs/covTFTS_figure_frozen_20260615_pre_backimage_active_sensing_collab_pack/`, `outputs/active_sensing_movie_information/reafferent_variance_accounting/` | Bridge or supplement |
| B | Aggregate motion benefit over static and controls | `declan/fixation_statistics_by_stimulus/run_backimage_aggregate_fem_information.py`, `declan/backimage_aggregate_fem_information_plan.md` | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/incremental_static_plus_motion_relids/` | Main-ready evidence |
| B | Generated atlas subpanels | `declan/figure4_active_sensing_atlas/scripts/plot_panel_b_subpanels.py` | `declan/figure4_active_sensing_atlas/figures/panel_B/` | Cache-only Panel B subpanel assets |
| B | Motion QC and trace policy | Same aggregate runner; `aggregate_motion_metadata.csv`, `trace_bank_metadata.csv` | Same run root as B primary | Main/supplement guardrail |
| B | Local actual-pairing diagnostic | `declan/backimage_local_pairing_Iz_revisit_plan.md`, `declan/fixation_statistics_by_stimulus/run_backimage_local_pairing_Iz_revisit.py` | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_gabor_pyramid_rel025_1_seed7_v1/` and related local-pairing runs | Supplement unless cleaner |
| C | Exact trajectory-table observer | `declan/backimage_trajectory_observer/observer.py`, `declan/backimage_trajectory_observer/likelihood.py`, `declan/fixation_statistics_by_stimulus/run_backimage_trajectory_table_observer.py`, `declan/backimage_trajectory_observer/results_log.md` | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/` | Main-ready evidence |
| C | Generated atlas subpanels | `declan/figure4_active_sensing_atlas/scripts/plot_panel_c_subpanels.py` | `declan/figure4_active_sensing_atlas/figures/panel_C/` | Cache-only Panel C subpanel assets |
| C | Option C scale sweep | Same observer code/log | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_trajectory_table_observer_cuda0_optionC_n64_k8_scales_v1/` | Main or supplement scale panel |
| C | Posterior concentration and image-condition diagnostics | `declan/backimage_trajectory_observer/analyze_feature_posterior.py`, posthoc image-condition analysis | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/posthoc_image_condition_analysis/` | Supplement or small main inset |
| C | Compact-mechanism projection | `declan/backimage_trajectory_observer/analyze_compact_mechanism.py`, `build_image_disjoint_compact_basis.py`, `summarize_compact_mechanism_followups.py` | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/compact_mechanism_image_disjoint_fold0_n512_k2_5_10_20_rand8_log_v1/` | Mechanism panel/supplement |
| D | Axis-conditioned observer | `declan/axis_conditioned_backimage_trajectory_observer/`, `declan/fixation_statistics_by_stimulus/run_backimage_trajectory_table_observer.py` | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1/`, `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/` | Shows axis priors help; axis preference unresolved |
| D | Generated atlas subpanels | `declan/figure4_active_sensing_atlas/scripts/plot_panel_d_subpanels.py` | `declan/figure4_active_sensing_atlas/figures/panel_D/` | Cache-only Panel D subpanel assets |
| D | Edge-parallel preservation audit | `declan/fixation_statistics_by_stimulus/run_backimage_edge_parallel_stability_screen.py` | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_edge_parallel_stability_screen_yfix_n256_pop256/` | Strong explanatory panel |
| D/E | Candidate image-conditioned objectives | `declan/fixation_statistics_by_stimulus/run_backimage_twin_drift_geometry.py`, `summarize_backimage_twin_drift_geometry.py`, conditional objective outputs | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_conditional_fixation_objectives_twin_axis_only_n256/` | Compare objectives to raw edge |
| E | Behavioral drift-edge alignment | `declan/fixation_statistics_by_stimulus/posthoc_backimage_edge_alignment_distribution_inspection.py` | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_edge_alignment_distribution_inspection/` | Main-ready behavior evidence |
| E | Generated atlas subpanels | `declan/figure4_active_sensing_atlas/scripts/plot_panel_e_subpanels.py` | `declan/figure4_active_sensing_atlas/figures/panel_E/` | Cache-only Panel E subpanel assets |
| E | Image structure and fixation windows | `declan/fixation_statistics_by_stimulus/run_backimage_image_structure_analysis.py` | `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv` | Behavior source table |

## Result Strength Tags

Main-ready:

- B aggregate FEM information, cleaned n256/k4-8 run.
- C matched-static trajectory-table observer.
- E behavioral drift-edge alignment.

Main or supplement:

- A retinal movie QC and existing headline figure assets.
- D edge-parallel preservation audit.
- D axis-conditioned observer, if presented as objective/candidate-set
  dependent.

Supplement/method:

- Local exact-pairing branch.
- Vernier failure versus natural-image success.
- Compact-mechanism projection until static-PC specificity is resolved.
- Motion-family matching and posterior diagnostics.
