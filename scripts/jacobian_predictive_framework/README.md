# Predictive Jacobian framework scaffold

This directory holds the first runnable slice of the fixRSVP Jacobian generalization pipeline.

Current scope:

- reuses `eval.fixrsvp.get_fixrsvp_data()` for trial-aligned fixRSVP collation
- computes empirical eye-position radius and frame-to-frame displacement summaries for Step 0 gating
- defines a primary analysis unit as pooled image identity with a minimum valid-sample threshold
- optionally loads a trained model, chooses one representative baseline lagged stimulus per unit, and estimates a common-translation Jacobian by finite differences on that baseline
- computes model FEM covariance from controlled shifted copies of the same baseline stimulus using empirical centered eye displacements converted from degrees to model pixels
- computes a matched image-shuffled null by comparing each unit against nearest baseline-matched units using Jacobian norm, stimulus energy, response level, and eye-radius summaries
- evaluates Step 0 linearization on both explicit CLI displacement magnitudes and empirical eye-step percentile magnitudes converted into model pixels
- writes manifest outputs to `outputs/jacobian_predictive_framework/`

Entry point:

```bash
python scripts/jacobian_predictive_framework/run_fixrsvp_steps01.py \
  --subject Allen \
  --date 2022-02-24 \
  --dataset-configs-path experiments/dataset_configs/multi_basic_120_long_legacy.yaml \
  --use-cached-data
```

Optional model-backed run:

```bash
python scripts/jacobian_predictive_framework/run_fixrsvp_steps01.py \
  --subject Allen \
  --date 2022-02-24 \
  --dataset-configs-path experiments/dataset_configs/multi_basic_120_long_legacy.yaml \
  --use-cached-data \
  --model-type resnet_none_convgru \
  --checkpoint-dir /mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints \
  --dataset-idx 10 \
  --empirical-displacement-percentiles 50,75,90,95 \
  --n-image-shuffle-matches 8 \
  --max-units 8 \
  --max-samples-per-unit 128
```

Recent validation:

- A smoke run on `Allen_2022-02-24` with `dataset_idx = 10` loaded cached fixRSVP data, loaded the best `resnet_none_convgru` checkpoint from the legacy 120-long stack, and wrote backend outputs successfully.
- The backend now fills `--max-units` with the first usable units rather than truncating early if initial pooled image units contain unusable samples.

Recommended Allen target:

- The strongest repo-local default is `Allen_2022-02-24`, because multiple fixRSVP model-check scripts reuse `dataset_idx = 10` against the same `multi_basic_120_long_legacy.yaml` stack, and that index resolves to `Allen_2022-02-24`.
- A secondary Allen target is `Allen_2022-03-04`, which appears explicitly in several direct loading and McFarland preparation scripts.

Current deliberate boundary:

The implemented backend works directly on the saved lagged stimulus tensor rather than reconstructing full world images. That keeps the first pass local and consistent with the dataset-aligned model input, but it should still be labeled as a lagged-input common-translation Jacobian until it is cross-checked against a full stimulus replay pipeline.

Recommended next implementation step:

1. Cross-check the lagged-input common-translation Jacobian against a replayed fixRSVP stack built from image frames plus eye traces.
2. Extend the summary outputs to the required per-step markdown format in the handoff document.
3. Tighten the image-shuffled null further if needed with explicit norm bins or additional image-state matching variables.