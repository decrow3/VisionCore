# Figure 4 geometry-bridge audit

## Scope
- Implements the canonical y-flip audit for model stimulus shifts.
- Separates model stimulus-shift transforms from empirical predictor-frame transforms.
- Runs the full sign/axis grid as a confirmation sweep.
- Regenerates empirical split-half bundles inside the audit using the same predictor frame used for the local empirical checks.
- Computes ceiling-normalized summaries.

## Current limitations
- Mixture-matched model objects and residual analyses are not implemented yet.
- Window ladder currently includes image_id, previous/next image context, and exact lagged-stimulus history hash levels, but not time-bin or longer local-context levels.
- Figure generation is not implemented yet in this first executable slice.

## Guardrails
- `x_negy` is the principled transform from the resampler convention; the 8-mode sweep is confirmatory, not a free selection step.
- If a non-canonical transform outperforms `x_negy`, treat that as a metadata or calibration red flag rather than a mode to adopt.
- `J_local` is structurally insensitive to eye-cloud-dependent mixture and window-pooling manipulations because it only sees the local baseline image tangent.
- `mean` should be interpreted as a mean-nearest baseline, not an exact rendered mean-eye retinal state.
- `current` currently means median-nearest baseline stimulus with a mean-centered eye predictor, not purely baseline-centered eye offsets.

## Command
- subject/date: Allen 2022-02-16
- dataset_configs_path: experiments/dataset_configs/multi_basic_120_long_legacy.yaml
- checkpoint_path: /mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints/learned_resnet_none_convgru_gaussian_ddp_bs128_ds30_lr1e-3_wd1e-4_corelrscale.5_warmup5/last.ckpt
- split_bundle_path: none (provenance only; local empirical bundles are regenerated in this audit)
- transform_modes: xy,x_negy
- predictor_modes: emp_xy
- centering_modes: current,baseline
- basis_names: B_model,FEM_PCs,J_local
- window_levels: image_id,prev_next,history_hash

## Canonical verification
- verification_window_id: image_6
- predictor_mode: emp_xy
- empirical_bundle_vs_model_x_corr_current: -0.1614
- empirical_bundle_vs_model_x_corr_canonical: -0.2235
- empirical_bundle_vs_model_y_corr_current: 0.6911
- empirical_bundle_vs_model_y_corr_canonical: -0.6829
- warning: canonical y correlation did not improve on the representative window; inspect eye-trace metadata before treating any alternative transform as valid.

## Session summary
- image_id / B_model / baseline / emp_xy: delta_2d=0.0238, delta_top1=0.0342, delta_over_ceiling_2d=0.3173
- image_id / B_model / current / emp_xy: delta_2d=0.0566, delta_top1=0.0194, delta_over_ceiling_2d=0.3302
- image_id / FEM_PCs / baseline / emp_xy: delta_2d=0.0423, delta_top1=0.0129, delta_over_ceiling_2d=0.1134
- image_id / FEM_PCs / current / emp_xy: delta_2d=0.0141, delta_top1=-0.0000, delta_over_ceiling_2d=0.0528
- image_id / J_local / baseline / emp_xy: delta_2d=0.0066, delta_top1=0.0056, delta_over_ceiling_2d=-0.0908
- image_id / J_local / current / emp_xy: delta_2d=0.0108, delta_top1=0.0107, delta_over_ceiling_2d=0.0325
- prev_next / B_model / baseline / emp_xy: delta_2d=0.0205, delta_top1=-0.0133, delta_over_ceiling_2d=0.0886
- prev_next / B_model / current / emp_xy: delta_2d=0.0384, delta_top1=-0.0147, delta_over_ceiling_2d=0.1895
- prev_next / FEM_PCs / baseline / emp_xy: delta_2d=0.0130, delta_top1=-0.0108, delta_over_ceiling_2d=0.0560
- prev_next / FEM_PCs / current / emp_xy: delta_2d=0.0230, delta_top1=0.0534, delta_over_ceiling_2d=0.1016
- prev_next / J_local / baseline / emp_xy: delta_2d=0.0086, delta_top1=0.0088, delta_over_ceiling_2d=0.0404
- prev_next / J_local / current / emp_xy: delta_2d=0.0110, delta_top1=-0.0137, delta_over_ceiling_2d=0.0658

## Window ladder
- B_model / baseline / emp_xy: image_id delta_2d=0.0238, prev_next delta_2d=0.0205, history_hash delta_2d=nan
- B_model / current / emp_xy: image_id delta_2d=0.0566, prev_next delta_2d=0.0384, history_hash delta_2d=nan
- FEM_PCs / baseline / emp_xy: image_id delta_2d=0.0423, prev_next delta_2d=0.0130, history_hash delta_2d=nan
- FEM_PCs / current / emp_xy: image_id delta_2d=0.0141, prev_next delta_2d=0.0230, history_hash delta_2d=nan
- J_local / baseline / emp_xy: image_id delta_2d=0.0066, prev_next delta_2d=0.0086, history_hash delta_2d=nan
- J_local / current / emp_xy: image_id delta_2d=0.0108, prev_next delta_2d=0.0110, history_hash delta_2d=nan

## Render sanity check
- render_sanity_check_png: results/figure4_geometry_bridge_audit_allen_2022_02_16_medium_prevnext/figures/render_sanity_check.png
- render_sanity_check_txt: results/figure4_geometry_bridge_audit_allen_2022_02_16_medium_prevnext/render_sanity_check.txt

## Tensor equivalence
- tensor_equivalence_csv: results/figure4_geometry_bridge_audit_allen_2022_02_16_medium_prevnext/tensor_equivalence_summary.csv
- provenance_summary_txt: results/figure4_geometry_bridge_audit_allen_2022_02_16_medium_prevnext/provenance_summary.txt
