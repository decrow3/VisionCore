# Figure 4 geometry-bridge audit

## Scope
- Implements the canonical y-flip audit for model stimulus shifts.
- Keeps the B_model regressor in empirical eye coordinates.
- Runs the full sign/axis grid as a confirmation sweep.
- Regenerates empirical split-half bundles inside the audit using the same predictor frame used for the local empirical checks.
- Computes ceiling-normalized summaries.

## Current limitations
- Window ladder, mixture-matched model objects, and residual analyses are not implemented yet.
- Figure generation is not implemented yet in this first executable slice.

## Guardrails
- `x_negy` is the principled transform from the resampler convention; the 8-mode sweep is confirmatory, not a free selection step.
- If a non-canonical transform outperforms `x_negy`, treat that as a metadata or calibration red flag rather than a mode to adopt.
- `J_local` is structurally insensitive to eye-cloud-dependent mixture and window-pooling manipulations because it only sees the local baseline image tangent.
- `mean` should be interpreted as a mean-nearest baseline, not an exact rendered mean-eye retinal state.

## Command
- subject/date: Allen 2022-02-16
- dataset_configs_path: experiments/dataset_configs/multi_basic_120_long_legacy.yaml
- checkpoint_path: /mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints/learned_resnet_none_convgru_gaussian_ddp_bs128_ds30_lr1e-3_wd1e-4_corelrscale.5_warmup5/last.ckpt
- split_bundle_path: none (provenance only; local empirical bundles are regenerated in this audit)
- transform_modes: xy,x_negy
- centering_modes: current,baseline
- basis_names: B_model,FEM_PCs,J_local

## Canonical verification
- verification_window_id: image_6
- empirical_bundle_vs_model_x_corr_current: -0.1507
- empirical_bundle_vs_model_x_corr_canonical: -0.2352
- empirical_bundle_vs_model_y_corr_current: 0.6932
- empirical_bundle_vs_model_y_corr_canonical: -0.6940
- warning: canonical y correlation did not improve on the representative window; inspect eye-trace metadata before treating any alternative transform as valid.

## Session summary
- B_model / baseline: delta_2d=0.0960, delta_top1=0.0835, delta_over_ceiling_2d=0.2206
- B_model / current: delta_2d=0.0971, delta_top1=0.0687, delta_over_ceiling_2d=0.2278
- FEM_PCs / baseline: delta_2d=0.0521, delta_top1=-0.3026, delta_over_ceiling_2d=0.1166
- FEM_PCs / current: delta_2d=0.0550, delta_top1=-0.3188, delta_over_ceiling_2d=0.1327
- J_local / baseline: delta_2d=-0.0708, delta_top1=-0.0687, delta_over_ceiling_2d=-0.1777
- J_local / current: delta_2d=-0.0773, delta_top1=-0.0799, delta_over_ceiling_2d=-0.2029
