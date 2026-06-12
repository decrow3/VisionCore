# Vernier Active-Sensing Analysis

This package implements the first-pass Vernier active-sensing analysis described
in `declan/vernier_active_sensing_analysis_plan.md`.

The runner first writes rendering/provenance audits, then optionally runs the
digital twin on finite-difference Vernier pairs with paired trajectories.

## Render-only smoke

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
  --skip-model \
  --fd-steps-arcmin 0.25,0.5 \
  --out-dir outputs/vernier_active_sensing_smoke
```

## Tiny model smoke

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
  --out-dir outputs/vernier_active_sensing_model_smoke \
  --n-traces 1 \
  --max-frames 3 \
  --fd-steps-arcmin 0.5 \
  --conditions static_center \
  --device cpu \
  --batch-size 2
```

## Larger first pass

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
  --out-dir outputs/vernier_active_sensing_first_pass \
  --n-traces 16 \
  --max-frames 60 \
  --fd-steps-arcmin 0.25,0.5 \
  --conditions static_center,static_repeated_phase,static_phase_cloud_single,static_phase_cloud_matched_positions,real_fem,order_shuffled_positions,axis_horizontal,axis_vertical,scaled_real_0.5,scaled_real_1.5 \
  --device cuda:0 \
  --batch-size 16
```

## Drift/microsaccade component scale pass

Component conditions use a per-trace velocity-threshold detector. The default
threshold is `30 deg/s` with `1` frame of padding on each side.

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
  --out-dir outputs/vernier_active_sensing_component_scale \
  --n-traces 16 \
  --max-frames 60 \
  --fd-steps-arcmin 0.25,0.5 \
  --conditions static_center,static_phase_cloud_matched_positions,real_fem,drift_only_scaled_0.5,drift_only_scaled_1.0,drift_only_scaled_1.5,microsaccade_only_scaled_0.5,microsaccade_only_scaled_1.0,microsaccade_only_scaled_1.5,drift_scaled_0.5,drift_scaled_1.5,microsaccade_scaled_0.5,microsaccade_scaled_1.5 \
  --device cuda:0 \
  --batch-size 16 \
  --microsaccade-speed-threshold-dps 30 \
  --microsaccade-pad-frames 1
```

## Summarize and plot

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.summarize_vernier_active_sensing \
  --run-dir outputs/vernier_active_sensing_first_pass
```

Outputs:

- `render_audit/pixel_audit.json`
- `render_audit/pixel_audit_fd_rows.csv`
- `render_audit/*.png`
- `cache/rates_<condition>_fd<step>arcmin.npz`
- `information_summary.csv`
- `condition_reliability_summary.csv`
- `paired_baseline_contrasts.csv`
- `paired_baseline_contrast_summary.csv`
- `motion_inventory.csv`
- `vernier_active_sensing_manifest.json`
- `figures/*.png`
- `figures/source_tables/*.csv`
