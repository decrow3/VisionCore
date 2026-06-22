# Figure 4D Story Options

Status: exploratory option sheet for revising Panel D.

![Story option sheet](/home/declan/VisionCore/declan/figure4_active_sensing_atlas/figures/panel_D/story_options/4D_story_option_sheet.png)

## Main Story

Panel D should explain why local image geometry is a useful coordinate system for the active-sensing story. The clean main-figure claim is: local edges define matched along-edge and across-edge directions, and along-edge shifts preserve V1-twin responses better at the tested displacement. Pixel preservation remains a sanity check, but the promoted panel should use model responses because that is the link to Panels B/C.

## Current Recommendation

Option D1 is the best starting point for the main composite. It makes the along/across comparison visually legible before showing the absolute model-response disruption costs. D3 is useful if we need to foreground robustness to model-response metric choice. D5 should remain a guardrail or supplement unless the figure needs to emphasize what we are not claiming.

## Planned B-Consistent Rerun

The current promoted feature-recovery result is still the n64 matched-static
run. Before write-lock, rerun the matched-static axis-conditioned observer on a
larger n128 manifest while keeping the model approximately aligned with Panel
4B. The intended rerun should keep the production local image axis as
`image_edge_axis_deg`, but switch the feature-posterior endpoint to
`pyramid_local_field` with k16 as the 4B-compatible headline dimension and k8
as a bridge to the existing 4C/4D feature-posterior runs. Use 0.5x as the
primary scale for continuity with the current D panel, and retain 1x/2x scale
context if runtime allows.

Queue script:

```text
declan/figure4_active_sensing_atlas/scripts/run_panel_d_matched_static_n128_b_consistent_gpu_queue.sh
```

This script uses the 4B-style drift/source restrictions where practical
(`min_patch_image_margin_px=270`, no microsaccade trace sources,
`max_trace_source_rms_deg=0.06`, `max_trace_source_radius_deg=0.20`,
`max_trace_source_speed_p95_deg_s=20.0`) while preserving the latent-eye,
axis-conditioned 4D observer structure.

## Axis Estimator Caveat

The local image axis may depend on the estimator. A patch-level average orientation-energy estimate can differ from a prominent orientation feature that a winner-take-all readout might select. The row-17/18 rail crop is the current reference example: raw BackImage rows 17/18 from `Allen_2022-02-16`, trial `184`, have stored aggregate `image_edge_axis_deg = -31.4 deg`, while a visible bright-rail fit gives `-37.6 deg`.

![Row-17/18 visible rail fit](/home/declan/VisionCore/declan/figure4_active_sensing_atlas/figures/panel_D/story_options/4D_row17_row18_visible_rail_fit_orientation.png)

Use this as a provenance example for the open axis-estimator question. It is not a correction to the quantitative Panel D readout.

Update 2026-06-22: a connected salient-contour pilot was run as a second
axis-estimator option, alongside the patch-average axis and the orientation-bin
WTA diagnostic. The pilot is useful for the axis-estimator caveat but should
not replace the current quantitative axis. On the row-17/18 rail thumbnail, the
connected-contour axis closely tracks the visible rail fit (`1.57 deg` error
versus `9.55 deg` for the patch-average axis). On the tighter analysis patch,
however, the patch-average axis is closer to the same Hough-style rail fit
(`4.21 deg` versus `7.53 deg` for the connected-contour axis). Across the
selected n=64 Panel D manifest, the connected-contour axis was available for
all windows but often disagreed with both average and WTA axes
(`median |salient - average| = 26.33 deg`,
`median |salient - WTA| = 26.50 deg`) and had low median component coherence
(`0.19`).

Decision for the current Figure 4D: keep `image_edge_axis_deg`, the
patch-level average orientation-energy axis, as the production quantitative
axis for the feature-recovery readout. Treat WTA and connected-contour axes as
diagnostics/provenance for the unresolved question of whether a behaviorally
meaningful local axis should reflect average orientation energy or a prominent
connected contour.

## Real-Patch Provenance

- `source`: `real_backimage`
- `session`: `Allen_2022-03-02`
- `trial_idx`: `477`
- `window_row`: `17`
- `source_window_id`: `252`
- `selection_note`: `preferred corrected source-row exemplar`
- `plot_axis_screen_deg`: `-4.813505179367278`
- `stability_edge_axis_gaze_deg`: `-4.813505179367278`
- `image_orientation_coherence`: `0.6800731251333298`
- `pixel_stability_advantage`: `1541.051025390625`
- `twin_stability_advantage`: `0.0001480113714933`

## Files

- `4D_story_option_1_axes_plus_preservation.png`
- `4D_story_option_1_axes_plus_preservation.pdf`
- `4D_story_option_2_along_across_costs.png`
- `4D_story_option_2_along_across_costs.pdf`
- `4D_story_option_3_metric_robustness.png`
- `4D_story_option_3_metric_robustness.pdf`
- `4D_story_option_4_axes_to_behavior_bridge.png`
- `4D_story_option_4_axes_to_behavior_bridge.pdf`
- `4D_story_option_5_objective_guardrail.png`
- `4D_story_option_5_objective_guardrail.pdf`
- `4D_story_option_6_minimal_mechanism.png`
- `4D_story_option_6_minimal_mechanism.pdf`
- `4D_story_option_sheet.png`
- `4D_story_option_values.csv`
- `4D_row17_row18_visible_rail_fit_orientation.png`
- `4D_row17_row18_visible_rail_fit_orientation_values.csv`
