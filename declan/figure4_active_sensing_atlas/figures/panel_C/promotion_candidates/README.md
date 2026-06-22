# Figure 4C Single-Panel Promotion Candidates

Status: candidate 5 selected provisionally for the promoted joint feature-posterior panel.

![Candidate sheet](/home/declan/VisionCore/declan/figure4_active_sensing_atlas/figures/panel_C/promotion_candidates/4C_single_panel_candidate_sheet.png)

## Recommendation

Selected provisional 4C: candidate 5, `4C_candidate_5_joint_feature_posterior_recovery.png`. It reflects the newer joint-model endpoint: zero-eye feature recovery falls as motion scale grows, while latent-eye joint inference remains stable without being given the measured eye trace. Candidate 2 remains useful historical image-identity context, and candidates 1, 3, and 4 remain guardrails/supporting views.

## Files

- `4C_candidate_1_matched_static_rescue_current.png`
- `4C_candidate_2_empirical_prior_rescue_clean.png`
- `4C_candidate_3_accuracy_ordering_context.png`
- `4C_candidate_4_scale_gap_guardrail.png`
- `4C_candidate_5_joint_feature_posterior_recovery.png`
- `4C_single_panel_candidate_sheet.png`
- `4C_single_panel_candidate_values.csv`

## Claim Boundary

These panels use an exact finite trajectory-table/posterior observer. The promoted 4C endpoint is absolute feature recovery under latent eye position, not image-identity accuracy or a gain normalized to a moving zero-eye baseline. Zero-eye scores the moved observation under a zero-eye-motion assumption; latent-eye joint hides the measured eye trace and marginalizes over candidate trajectories. It does not show that the animal computes this posterior or that the posterior identifies the true eye trajectory.
