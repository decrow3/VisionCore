# Figure 4 geometry-bridge audit

## Scope
- Implements the canonical y-flip audit for model stimulus shifts.
- Separates model stimulus-shift transforms from empirical predictor-frame transforms.
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
- predictor_modes: emp_xy,emp_x_negy
- centering_modes: current,baseline
- basis_names: B_model,FEM_PCs,J_local

## Canonical verification
- verification_window_id: image_6
- predictor_mode: emp_xy
- empirical_bundle_vs_model_x_corr_current: -0.1538
- empirical_bundle_vs_model_x_corr_canonical: -0.2152
- empirical_bundle_vs_model_y_corr_current: 0.6955
- empirical_bundle_vs_model_y_corr_canonical: -0.6911
- warning: canonical y correlation did not improve on the representative window; inspect eye-trace metadata before treating any alternative transform as valid.

## Session summary
- B_model / baseline / emp_x_negy: delta_2d=0.0981, delta_top1=0.0796, delta_over_ceiling_2d=0.2239
- B_model / baseline / emp_xy: delta_2d=0.0935, delta_top1=0.0788, delta_over_ceiling_2d=0.2206
- B_model / current / emp_x_negy: delta_2d=0.1018, delta_top1=0.0339, delta_over_ceiling_2d=0.2363
- B_model / current / emp_xy: delta_2d=0.1026, delta_top1=0.0319, delta_over_ceiling_2d=0.2368
- FEM_PCs / baseline / emp_x_negy: delta_2d=0.0561, delta_top1=-0.3117, delta_over_ceiling_2d=0.1286
- FEM_PCs / baseline / emp_xy: delta_2d=0.0559, delta_top1=-0.3022, delta_over_ceiling_2d=0.1320
- FEM_PCs / current / emp_x_negy: delta_2d=0.0290, delta_top1=-0.2307, delta_over_ceiling_2d=0.0660
- FEM_PCs / current / emp_xy: delta_2d=0.0305, delta_top1=-0.2235, delta_over_ceiling_2d=0.0695
- J_local / baseline / emp_x_negy: delta_2d=-0.0719, delta_top1=-0.0759, delta_over_ceiling_2d=-0.1826
- J_local / baseline / emp_xy: delta_2d=-0.0737, delta_top1=-0.0793, delta_over_ceiling_2d=-0.1902
- J_local / current / emp_x_negy: delta_2d=-0.0727, delta_top1=-0.0753, delta_over_ceiling_2d=-0.1844
- J_local / current / emp_xy: delta_2d=-0.0718, delta_top1=-0.0770, delta_over_ceiling_2d=-0.1812

## Render sanity check
- render_sanity_check_png: results/figure4_geometry_bridge_audit_allen_2022_02_16_2x2_cap4_helper/figures/render_sanity_check.png
- render_sanity_check_txt: results/figure4_geometry_bridge_audit_allen_2022_02_16_2x2_cap4_helper/render_sanity_check.txt

## Tensor equivalence
- tensor_equivalence_csv: results/figure4_geometry_bridge_audit_allen_2022_02_16_2x2_cap4_helper/tensor_equivalence_summary.csv
- provenance_summary_txt: results/figure4_geometry_bridge_audit_allen_2022_02_16_2x2_cap4_helper/provenance_summary.txt
