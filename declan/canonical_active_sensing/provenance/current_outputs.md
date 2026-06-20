# Canonical Active-Sensing Output Provenance

Last updated: 2026-06-20.

## Feature Target

Current two-readout candidate:

```text
Primary aggregate readout: pyramid_local_field k16 temporal_pca
Local mechanistic sensitivity: pyramid_local_field k16 delta_mean
```

This target remains provisional until the joint `rel_0p25x` completion pass
lands and the v4 adjudication is reviewed.

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
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_pyramid_k16_tworeadout_rel025-2_canonical_v1/incremental_static_plus_motion_tworeadout_v1
```

Aggregate production figure pack:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n256_pyramid_k16_tworeadout_rel025-2_canonical_v1/figure_pack_tworeadout_v1
```

Local production reruns:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_local_pairing_Iz_canonical_pyramid_k16_rel025_0p5_1_seed7_v1
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_local_pairing_Iz_canonical_pyramid_k16_rel2_seed7_v1
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
