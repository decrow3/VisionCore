# Figure 4B Single-Panel Promotion Candidates

Status: draft candidates for choosing one promoted aggregate-FEM panel.

![Candidate sheet](/home/declan/VisionCore/declan/figure4_active_sensing_atlas/figures/panel_B/promotion_candidates/4B_single_panel_candidate_sheet.png)

## Recommendation

Candidate 3 is the promoted strict source-trial grouped information-axis target after the incremental static-plus-motion summaries are recomputed. The plotted quantity is diagonal Gaussian decoder information gain over the stabilized/static baseline in bits, with point-centered decode-bootstrap CIs. The pose-unaware hidden-sample proxy is now plotted on the same information axis.

## Files

- `4B_candidate_3_power_rerun_absolute_gain.png`
- `4B_candidate_4_k16_tworeadout_preview.png`
- `4B_single_panel_candidate_sheet.png`
- `4B_single_panel_candidate_values.csv`

## Claim Boundary

The promoted axis is a Gaussian variational decoder lower-bound increment, not an absolute mutual-information estimate. Headline panels use the diagonal residual-variance form in bits; full-covariance Ledoit-Wolf log-det values are supplemental robustness. Legacy `-MSE` candidates remain archive/QC only and require `PANEL_B_ALLOW_LEGACY_MSE=1` to render from old tables.
