# Panel B Generated Assets

- `B1_task_schematic.png`
- `B1_task_schematic.pdf`
- `B2_motion_family_qc.png`
- `B2_motion_family_qc.pdf`
- `B3_empirical_gain_vs_static.png`
- `B3_empirical_gain_vs_static.pdf`
- `B4_empirical_minus_controls.png`
- `B4_empirical_minus_controls.pdf`
- `B5_absolute_gain_guardrail.png`
- `B5_absolute_gain_guardrail.pdf`
- `panel_B_motion_qc_values.csv`
- `panel_B_gain_vs_static_values.csv`
- `panel_B_control_contrast_values.csv`
- `panel_B_absolute_gain_guardrail_values.csv`
- `panel_B_subpanels_caption.md`
- `promotion_candidates/`
  - Current review surface for choosing one promoted 4B panel.
  - Candidate 3 has been redrawn from the corrected static-mean posthoc:
    `incremental_staticmean_plus_motion_tworeadout_v2`.
  - Treat this as a provisional single-readout panel. Current readout roles are:
    `mean`/`delta_mean` for absolute aggregate gain, `delta_mean` for local
    mechanistic sensitivity, and temporal PCA/DCT for order-sensitive
    empirical-vs-control diagnostics.
  - The all-readout audit lives in the n384 aggregate output under
    `incremental_staticmean_plus_motion_allreadouts_v1/readout_atlas_figures/`.
  - Do not use OU as the headline negative control until the OU trace-control
    audit is closed.
