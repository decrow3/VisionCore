# Continuous-Tau MLP Feature Decoder

Hybrid diagnostic: continuous-eye-trace inference supplies `tau_hat`, then
a Tejas-style MLP decodes directly to a compact image-feature embedding.
The endpoint is continuous feature recovery, not image-candidate choice.
Augmented modes train on the continuous response-bank rows
`(prior response, trajectory) -> phi(source image)` and test on held-out
observed responses with continuous `tau_hat`.

Primary input:

```text
compact observed response movie
+ continuous tau_hat features
-> z_hat ~= PCA(phi(image))
```

All-scale feature cosine:

```text
observed compact only:                 0.2007
augmented compact only:                0.5199
augmented continuous tau:              0.3962
augmented continuous tau + interactions: 0.2416
augmented true tau + interactions:     0.5448
augmented 0x stabilized response:      0.5774
augmented known-eye model response:     0.5430
true-row 0x stabilized response:       0.5070
true-row known-eye model response:      0.4287
```

Primary paired contrast:

```text
continuous tau - augmented 0x stabilized: -0.1812  CI [-0.2344, -0.0457]
```

Interpretation boundary: the continuous trajectory estimate is recovered
using the true-image branch of the continuous-joint model for each held-out
supervised row. This avoids trajectory-candidate selection and image
candidate readout at the endpoint, but it is still a known-image
trajectory-conditioning upper-bound diagnostic.

Outputs:

- `continuous_tau_mlp_feature_decoder_trials.csv`
- `continuous_tau_mlp_feature_decoder_summary.csv`
- `continuous_tau_mlp_feature_decoder_contrasts.csv`
- `continuous_tau_mlp_feature_decoder_models.csv`
- `continuous_tau_mlp_feature_decoder_dataset.npz`
- `continuous_tau_mlp_feature_decoder_manifest.json`
