# Figure 4A Single-Panel Promotion Candidates

Status: draft candidates for choosing one promoted 4A panel.

![Candidate sheet](/home/declan/VisionCore/declan/figure4_active_sensing_atlas/figures/panel_A/promotion_candidates/4A_single_panel_candidate_sheet.png)

## Recommendation

Selected provisional 4A: candidate 3, `4A_candidate_3_real_high_contrast_positive.png`.

Rationale: candidate 1 preserves A1's proportions but is centered on a dark patch. Candidate 3 keeps the single-panel A1 grammar, uses a real BackImage canvas crop and recorded fixation trace, and has a clearer high-contrast retinal sample with positive drift-edge alignment metadata. Additional image/fixation pairs can be screened later.

## Files

- `4A_candidate_0_current_A1_reference.png`
- `4A_candidate_1_real_backimage_a1_proportions.png`
- `4A_candidate_2_real_backimage_context.png`
- `4A_candidate_3_real_high_contrast_positive.png`
- `4A_single_panel_candidate_sheet.png`
- `4A_single_panel_candidate_values.csv`

## Real-Data Provenance

The real candidates call `_backimage_canvas(session, trial_idx)` and use the recorded `backimage.dset` eyepos slice indexed by `global_start:global_stop` from `backimage_image_fem_windows.csv`.
