# BackImage Axis Residual Adjudication: Archive Note

Archived on 2026-06-19.

## Why Archived

`posthoc_backimage_axis_residual_adjudication.py` was a useful scratch
prototype, but it duplicates several existing BackImage analyses:

- `summarize_backimage_latent_information_screen.py` already writes
  `posthoc_residual_prediction_summary.csv`, which tests whether model scores
  predict within-session `drift_edge_cos2` beyond image orientation coherence
  and drift anisotropy.
- `run_backimage_edge_parallel_stability_screen.py` already writes
  `alignment_strength_prediction_summary.csv`, which tests pixel/twin
  edge-parallel preservation beyond edge coherence.
- `summarize_backimage_twin_drift_geometry.py` already adjudicates fixed
  V1-twin PA/PB/Pareto axis selectors against raw edge geometry.

The prototype's main distinct idea was joining newer axis-conditioned observer
and feature-posterior trial caches into the residual-prediction question.
However, the preservation-audit window set overlaps sparsely with those newer
caches, so a single wide table creates a misleading impression of one unified
analysis when the fitted rows differ by cache family.

## Best Path Forward

1. Keep `summarize_backimage_latent_information_screen.py` as the canonical
   residual-prediction home.
2. Add a small loader there, or a sibling posthoc module, that converts
   axis-conditioned observer and feature-posterior trial caches into the same
   tidy predictor schema used by `posthoc_residual_prediction_summary.csv`.
3. Fit one predictor family at a time on its own valid window set, with explicit
   `n_windows`, `n_sessions`, and cache-source columns.
4. Compare against the same baseline controls already used in the latent
   summary: within-session `image_orientation_coherence` and `drift_anisotropy`.
5. Add pixel/twin preservation controls only for rows that actually overlap the
   preservation audit, and report that as a separate overlap sensitivity rather
   than the headline model.
6. Put any final cross-family table in a tidy long form:
   `source_family`, `cache_name`, `predictor`, `coef`, `incremental_r2`,
   `control_r2`, `full_r2`, `n_windows`, `n_sessions`, and optional
   session-bootstrap/sign-count columns.

## Claim Boundary

The current durable claim remains:

Observed BackImage drift is robustly related to raw local edge geometry. Existing
latent-information residual summaries show small predictor-specific incremental
R2s beyond edge coherence and drift anisotropy, while the fixed V1-twin
PA/PB/Pareto axis-selection analysis does not beat raw edge geometry. The newer
axis-conditioned feature-posterior branch is promising for feature recovery, but
it still needs a tidy, cache-family-aware residual-prediction add-on before it
can be claimed as explaining drift-axis behavior.
