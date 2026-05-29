# Figure 4 geometry-bridge audit

## Scope
- Implements the canonical y-flip audit for model stimulus shifts.
- Keeps the B_model regressor in empirical eye coordinates.
- Runs the full sign/axis grid as a confirmation sweep.
- Reports model-side centering variants against fixed empirical split bundles.
- Computes ceiling-normalized summaries.

## Current limitations
- Empirical split-half bundles are treated as fixed upstream targets in this first pass.
- Window ladder, mixture-matched model objects, and residual analyses are not implemented yet.
- Figure generation is not implemented yet in this first executable slice.

## Command
- subject/date: Allen 2022-02-16
- dataset_configs_path: experiments/dataset_configs/multi_basic_120_long_legacy.yaml
- checkpoint_path: /mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints/learned_resnet_none_convgru_gaussian_ddp_bs128_ds30_lr1e-3_wd1e-4_corelrscale.5_warmup5/last.ckpt
- split_bundle_path: outputs/jacobian_predictive_framework/allen_2022_02_16_step2_iter1/step2_split_half_bundles.npz
- transform_modes: xy,x_negy
- centering_modes: current
- basis_names: B_model,FEM_PCs,J_local

## Canonical verification
- verification_window_id: none

## Session summary
