# Canonical Geometry Output Provenance

Last updated: 2026-06-20.

## Current Raw-Edge Inputs

BackImage image/window manifest:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv
```

Edge-parallel stability input:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_twin_stability_metric_audit/twin_stability_metric_by_window.csv
```

Feature-preservation input:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_twin_stability_metric_audit/endpoint_feature_preservation_static_decoder/feature_preservation_by_window.csv
```

Joint observer and posterior inputs:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/observer_trials.csv
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/feature_posterior_trials.csv
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/feature_axis_contrasts.csv
```

## Canonical Raw-Edge Output

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_raw_edge_roadblock_residual_adjudication_canonical_v1
```

## Canonical Geometry Figure Pack

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_geometry_figure_pack_canonical_v1
```

This folder contains:

```text
panel_D/
panel_E/
raw_edge_audit/
figure_source_tables/
panel_provenance.csv
geometry_figure_pack_report.md
figure_pack_metadata.json
```

## Validation Commands

```bash
.venv/bin/python -m declan.canonical_geometry.validate_configs
.venv/bin/python -m declan.canonical_geometry.run_raw_edge_audit --print-command
.venv/bin/python -m declan.canonical_geometry.make_geometry_figure_pack --print-command
.venv/bin/python -m declan.canonical_geometry.make_geometry_figure_pack --validate-only
```

The geometry figure-pack wrapper now regenerates panel D/E assets from the
atlas panel scripts, copies raw-edge residual-adjudication artifacts, and writes
an explicit provenance/report contract.
