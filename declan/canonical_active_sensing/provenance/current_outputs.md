# Canonical Active-Sensing Output Provenance

Last updated: 2026-06-21.

## Feature Target

Current corrected candidate:

```text
Absolute aggregate candidates: pyramid_local_field k16 mean, delta_mean
Local mechanistic sensitivity: pyramid_local_field k16 delta_mean
Order-sensitive diagnostics: pyramid_local_field k16 temporal_pca / temporal_dct variants
```

The prior v5 power-rerun adjudication is superseded. Its aggregate
`temporal_pca` posthoc compared motion temporal PCs against the static
temporal-PC summary, which is nearly zero for a static trace. The corrected
cache-only posthoc uses the static mean response as the static baseline for all
motion summaries, matching the intended question:

```text
z ~ R_static_mean
z ~ R_static_mean + R_motion_summary
```

The latest corrected adjudication is still provisional because it was run
without `--write-lock` and because the OU/readout audit is not closed.

Latest corrected power-rerun adjudication:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_feature_decomposition_adjudication_v6_staticmean_corrected_power_rerun_primary_scales
```

Key v6 scores:

```text
pyramid_local_field k16 delta_mean:
  score_with_joint_axis_term = 1.912
  score_without_joint_axis_term = 1.971
  aggregate_score = 0.332
  local_Iz_score = 2.139
  joint_axis_score = -0.059
  joint_generic_score = 3.000

pyramid_local_field k16 temporal_pca:
  score_with_joint_axis_term = 0.608
  score_without_joint_axis_term = 0.667
  aggregate_score = 0.593
  local_Iz_score = 0.575
  joint_axis_score = -0.059
  joint_generic_score = 3.000
```

Interpretation:

- `delta_mean` is the current primary feature readout. It keeps the cleanest
  local mechanistic signal and is the only member of the two-readout pair with
  positive static-mean aggregate support at `1x`.
- The all-readout nested-alpha audit makes this a role split rather than a
  single winner: `mean` is the strongest absolute aggregate candidate under the
  nested-alpha diagnostic, while `delta_mean` is the more interpretable
  static-subtracted motion-induced response readout and aligns with local
  pairing.
- `temporal_pca` remains useful as a relative control diagnostic: empirical
  temporal PCs beat OU strongly, but adding temporal PCs to the static mean
  baseline worsens absolute feature decoding for all motion families. It should
  not be used as the primary "motion adds beyond static" aggregate claim.
- Temporal DCT variants should travel with temporal PCA in the audit because
  they preserve trajectory order with a fixed basis rather than a fitted PCA
  basis.
- OU is audit-pending. Strong empirical-minus-OU temporal-readout separation is
  not yet a headline control result until trace replay, PSD/autocorrelation,
  centering, response geometry, and nested-alpha behavior are checked.
- Generic joint-minus-zero feature recovery remains strong; the axis-specific
  parallel-over-orthogonal term remains weak/slightly negative.
- The v6 pass has no primary-scale cache gaps for the requested
  `pyramid_local_field k16` comparison.

All-readout audit posthoc:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/incremental_staticmean_plus_motion_allreadouts_v1
```

All-readout figures/tables:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/incremental_staticmean_plus_motion_allreadouts_v1/readout_atlas_figures/
  readout_atlas_gain_over_static_mean.png
  readout_atlas_empirical_minus_controls.png
  readout_atlas_primary_scale_score_table.csv
  temporal_alpha_sensitivity_primary_scales.csv
  nested_alpha_primary_scale_diagnostic.csv
```

OU/readout audit handoff:

```text
declan/ou_trace_control_and_readout_audit_handoff.md
```

## Current Evidence Caches

Aggregate feature adjudication cache:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/incremental_static_plus_motion_feature_adjudication_k2_4_8_16_32_v1
```

Local `I_z` primary and sentinel caches:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_gabor_pyramid_rel025_0p5_1_seed7_v1/incremental_static_plus_motion_feature_adjudication_k2_4_8_16_32_v1
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_gabor_pyramid_rel2_seed7_v1/incremental_static_plus_motion_feature_adjudication_k2_4_8_16_32_v1
```

Joint posterior completed scale-sweep cache:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1
```

Joint posterior `rel_0p25x` completion target:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_rel0p25_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1
```

Feature adjudication v4 target:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_feature_decomposition_adjudication_v4_joint_rel0p25_complete
```

## Canonical Production Targets

Aggregate production rerun:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_pyramid_k16_tworeadout_rel025-2_canonical_v1
```

Aggregate production posthoc:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_pyramid_k16_tworeadout_rel025-2_canonical_v1/incremental_staticmean_plus_motion_tworeadout_v2
```

Aggregate production figure pack:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_pyramid_k16_tworeadout_rel025-2_canonical_v1/figure_pack_staticmean_delta_v2
```

Local production reruns:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_local_pairing_Iz_canonical_pyramid_k16_rel025_0p5_1_seed7_v1
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_local_pairing_Iz_canonical_pyramid_k16_rel2_seed7_v1
```

## Figure 4 Power Rerun Outputs

The higher-power rerun surface lives in:

```text
declan/canonical_active_sensing/configs/figure4_power_rerun_v1.json
```

Primary aggregate power output:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1
```

Primary aggregate power posthoc:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1/incremental_staticmean_plus_motion_tworeadout_v2
```

Local pairing power targets:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_local_pairing_Iz_power_pyramid_k16_rel025_0p5_1_seed7_k64_v1
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_local_pairing_Iz_power_pyramid_k16_rel025_0p5_1_seed11_k64_v1
```

Local pairing power posthocs:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_local_pairing_Iz_power_pyramid_k16_rel025_0p5_1_seed7_k64_v1/incremental_static_plus_motion_tworeadout_v1
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_local_pairing_Iz_power_pyramid_k16_rel025_0p5_1_seed11_k64_v1/incremental_static_plus_motion_tworeadout_v1
```

Aggregate seed replicate, not yet required by the current audit decision:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_pyramid_k16_tworeadout_rel025-2_power_seed11_k8_v1
```

Optional joint prior-depth target:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_c4_k16_rel0p25_prior32_power_v1
```

## Validation Commands

```bash
.venv/bin/python -m declan.canonical_active_sensing.validate_configs
.venv/bin/python -m declan.canonical_active_sensing.validate_configs --check-output-freshness
.venv/bin/python -m declan.canonical_active_sensing.run_aggregate_fem --print-command
.venv/bin/python -m declan.canonical_active_sensing.run_incremental_posthoc --print-command
.venv/bin/python -m declan.canonical_active_sensing.run_local_pairing --section local_pairing_sentinel --print-command
.venv/bin/python -m declan.canonical_active_sensing.run_incremental_posthoc --config declan/canonical_active_sensing/configs/local_pairing_k16_v1.json --section local_incremental_primary --print-command
.venv/bin/python -m declan.canonical_active_sensing.run_incremental_posthoc --config declan/canonical_active_sensing/configs/local_pairing_k16_v1.json --section local_incremental_sentinel --print-command
.venv/bin/python -m declan.canonical_active_sensing.adjudicate_feature_spec --print-command --skip-figures
.venv/bin/python -m declan.canonical_active_sensing.make_active_sensing_figure_pack --print-command
```
