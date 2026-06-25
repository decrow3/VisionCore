# Panel C Joint-Decoder Diagnostic Checks

Cache-only diagnostics for the Figure 4C joint observer / joint decoder result.

## Inputs

- `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1/feature_compact_mechanism_summary.csv`
- `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1/feature_compact_mechanism_uncertainty.csv`
- `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1/feature_compact_mechanism_qc.csv`
- `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/observer_summary.csv`

## Outputs

- `panel_C_joint_decoder_check_sheet.png`: six-panel check sheet for feature recovery, compact-removal contrasts, posterior concentration, addback/clipping QC, older image-identity observer accuracy, and axis-prior detail.
- `panel_C_joint_decoder_axis_detail.png`: split axis-prior feature-recovery curves.
- `panel_C_joint_decoder_feature_summary.csv`
- `panel_C_joint_decoder_contrasts.csv`
- `panel_C_joint_decoder_observer_accuracy.csv`
- `panel_C_joint_decoder_feature_rows.csv`

These figures are diagnostics, not replacement promotion candidates.
