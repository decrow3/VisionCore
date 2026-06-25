# Catalog-Residual Refinement Diagnostics

Catalog-residual scoring improves over the finite best-trajectory observer mostly by rescuing finite-catalog misses.
Best-single -> catalog-residual rescued: 73
Best-single -> catalog-residual lost: 22

Aggregation/calibration also matters: all-anchor log-mean is weaker than top-2, and a small shrink toward all-anchor improves full-cache accuracy.

Primary files:
- `catalog_residual_refinement_lift_trials.csv`
- `catalog_residual_aggregation_calibration_summary.csv`
- `catalog_residual_aggregation_transition_counts.csv`
- `catalog_residual_refinement_diagnostics.png`
