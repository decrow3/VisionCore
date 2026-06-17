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

The current joint-geometry observer is an instantaneous local-chart pilot. The
Wu-style interpretation here is Bayesian nuisance marginalization, not joint
image/eye reconstruction: Vernier sign is the desired latent, and eye trajectory
is nuisance state. The primary joint observer score is therefore a Vernier
likelihood ratio, approximating `log sum_w p(response | theta, w) p(w)` for
`theta in {+delta, -delta}`. Pose recovery metrics are diagnostics, not the
success criterion.

The `residual` likelihood mode reports Mahalanobis residual scores for
deterministic expected-count comparisons, not normalized log probabilities. In
residual mode, the raw true-hypothesis gap closure is suppressed and the
margin-based closure is reported separately. The lag-aware diagnostic path in
`run_lag_geometry_diagnostic.py` estimates temporal lag-plane translation
kernels and compares them with exact known-trajectory responses before those
kernels are promoted into the production joint filter.

The lag diagnostic writes three row families in `lag_geometry_diagnostic.csv`:
`rate_fidelity` for motion-induced response deltas, `likelihood_fidelity` for
exact-vs-approximate known-trajectory residual scores, and
`decision_fidelity` for whether the approximation preserves the known-eye
Vernier decision margins. Use those diagnostics to choose a history length
before running or implementing an expensive trajectory-marginalized observer.

## Joint geometry observer smoke

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
  --out-dir outputs/vernier_joint_geometry_smoke \
  --n-traces 2 \
  --max-frames 5 \
  --fd-steps-arcmin 0.5 \
  --conditions real_fem,static_center,order_shuffled_positions \
  --run-joint-geometry-observer \
  --joint-observer enumerated \
  --joint-compact-k-list 2 \
  --joint-eye-step-max-arcmin 1 \
  --joint-eye-step-sigma-arcmin 1 \
  --joint-eye-step-arcmin 1 \
  --joint-max-particles 3000 \
  --joint-likelihood-normalization residual \
  --device cpu \
  --batch-size 2
```

## Exact trajectory-table observer

This cache-only observer reads saved `rates_*.npz` files and evaluates the
Vernier likelihood ratio after marginalizing over the empirical trajectory
catalog. By default it uses a Poisson count likelihood and includes
the observed trajectory in the empirical prior. Pass `--leave-one-out` for a
generalization diagnostic that excludes the true trace from the nuisance
catalog.

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_trajectory_table_observer \
  --source-dir outputs/vernier_joint_geometry_enumerated_gpu0_fixed \
  --out-dir outputs/vernier_trajectory_table_observer \
  --conditions real_fem,static_center \
  --prior-conditions real_fem,static_center \
  --fd-steps-arcmin 0.25,0.5 \
  --reference-condition static_center \
  --likelihood-normalization poisson
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

Component conditions use Jake's `jake.twininfo.eye_controls.detect_microsaccade_events`
labeling path. By default, the per-trace speed threshold is the robust MAD
threshold (`z=6`) with `1` frame of padding on each side. Pass
`--microsaccade-speed-threshold-dps 30` to force the earlier fixed threshold.

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
  --out-dir outputs/vernier_active_sensing_component_scale \
  --n-traces 16 \
  --max-frames 60 \
  --fd-steps-arcmin 0.25,0.5 \
  --conditions static_center,static_phase_cloud_matched_positions,real_fem,drift_only_scaled_0.5,drift_only_scaled_1.0,drift_only_scaled_1.5,microsaccade_only_scaled_0.5,microsaccade_only_scaled_1.0,microsaccade_only_scaled_1.5,drift_scaled_0.5,drift_scaled_1.5,microsaccade_scaled_0.5,microsaccade_scaled_1.5 \
  --device cuda:0 \
  --batch-size 16 \
  --microsaccade-threshold-z 6 \
  --microsaccade-pad-frames 1
```

## Next-pass scale and pose-readout sweep

Scale-specific controls use aliases such as `static_phase_cloud_matched_scaled_0.5`
and `order_shuffled_scaled_0.5`, so reduced-amplitude motion is compared against
reduced-amplitude phase clouds rather than the full real-FEM cloud.
When `--full-cov-max-units` is smaller than the readout dimensionality, full-covariance
rows are labeled as unit-subset diagnostics.

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
  --out-dir outputs/vernier_active_sensing_scale_pose_sweep \
  --n-traces 16 \
  --max-frames 60 \
  --fd-steps-arcmin 0.125,0.25,0.5,1.0 \
  --conditions static_center,real_fem,scaled_real_0,scaled_real_0.125,scaled_real_0.25,scaled_real_0.5,scaled_real_0.75,scaled_real_1.5,scaled_real_2,scaled_real_3,static_phase_cloud_matched_scaled_0,static_phase_cloud_matched_scaled_0.125,static_phase_cloud_matched_scaled_0.25,static_phase_cloud_matched_scaled_0.5,static_phase_cloud_matched_scaled_0.75,static_phase_cloud_matched_scaled_1,static_phase_cloud_matched_scaled_1.5,static_phase_cloud_matched_scaled_2,static_phase_cloud_matched_scaled_3,order_shuffled_scaled_0,order_shuffled_scaled_0.125,order_shuffled_scaled_0.25,order_shuffled_scaled_0.5,order_shuffled_scaled_0.75,order_shuffled_scaled_1,order_shuffled_scaled_1.5,order_shuffled_scaled_2,order_shuffled_scaled_3,axis_horizontal,axis_vertical \
  --pose-sigmas-arcmin 0,0.25,0.5,1,2,4 \
  --run-full-cov-pose-blind \
  --run-compact-aware-pose-blind \
  --compact-k-list 1,2,5,10 \
  --compact-alphas 0,0.25,0.5,0.75,1 \
  --full-cov-max-units 256 \
  --device cuda:0 \
  --batch-size 16
```

Summarize the run with the same directory as both the source and figure target:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.summarize_vernier_active_sensing \
  --run-dir outputs/vernier_active_sensing_scale_pose_sweep \
  --out-dir outputs/vernier_active_sensing_scale_pose_sweep
```

For the rotated-stimulus axis control, rerun the same axis conditions with:

```bash
--stimulus-orientation-deg 90
```

## Scale-specific phase-cloud controls

Use these conditions to test whether a scaled-real advantage survives baselines
matched to the scaled retinal positions, rather than to the full real-FEM cloud:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
  --out-dir outputs/vernier_active_sensing_scale_sweep \
  --n-traces 16 \
  --max-frames 60 \
  --fd-steps-arcmin 0.125,0.25,0.5,1.0 \
  --conditions static_center,real_fem,static_phase_cloud_matched_positions,order_shuffled_positions,scaled_real_0,scaled_phase_cloud_matched_positions_0,scaled_order_shuffled_positions_0,scaled_real_0.125,scaled_phase_cloud_matched_positions_0.125,scaled_order_shuffled_positions_0.125,scaled_real_0.25,scaled_phase_cloud_matched_positions_0.25,scaled_order_shuffled_positions_0.25,scaled_real_0.5,scaled_phase_cloud_matched_positions_0.5,scaled_order_shuffled_positions_0.5,scaled_real_0.75,scaled_phase_cloud_matched_positions_0.75,scaled_order_shuffled_positions_0.75,scaled_real_1.5,scaled_phase_cloud_matched_positions_1.5,scaled_order_shuffled_positions_1.5,scaled_real_2,scaled_phase_cloud_matched_positions_2,scaled_order_shuffled_positions_2,scaled_real_3,scaled_phase_cloud_matched_positions_3,scaled_order_shuffled_positions_3 \
  --device cuda:0 \
  --batch-size 16
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
