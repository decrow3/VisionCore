# Predictive Jacobian framework scaffold

This directory holds the first runnable slice of the fixRSVP Jacobian generalization pipeline.

Current scope:

- reuses `eval.fixrsvp.get_fixrsvp_data()` for trial-aligned fixRSVP collation
- computes empirical eye-position radius and frame-to-frame displacement summaries for Step 0 gating
- defines a primary analysis unit as pooled image identity with a minimum valid-sample threshold
- writes manifest outputs to `outputs/jacobian_predictive_framework/`

Entry point:

```bash
python scripts/jacobian_predictive_framework/run_fixrsvp_steps01.py \
  --subject Luke \
  --date 2026-03-16 \
  --dataset-configs-path experiments/dataset_configs/single_Luke_2026-03-16_left_V1_rowley.yaml \
  --use-cached-data
```

Current deliberate boundary:

The script does not yet compute model-response finite differences or local Jacobians. That backend is isolated behind `Step01Backend` so the next iteration can stay local and consistent with the same analysis-unit manifest.

Recommended next implementation step:

1. Attach a dataset-stimulus replay backend that uses the collated fixRSVP stimulus snippets already aligned by `get_fixrsvp_data()`.
2. For each retained unit, compute actual response changes under small synthetic translations and compare them to first-order Jacobian predictions.
3. Reuse `alignment_score()` and `capture_fraction()` from `declan/jacobian_test3.py` once unit-level FEM covariance estimates are available.