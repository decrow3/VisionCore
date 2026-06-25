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
observed compact only:                 0.1967
augmented compact only:                0.3695
augmented continuous tau:              0.3140
augmented continuous tau + interactions: 0.3078
augmented true tau:                    0.3521
augmented true tau + interactions:     0.3499
augmented 0x stabilized response:      0.3243
augmented known-eye model response:     0.3196
true-row 0x stabilized response:       0.2146
true-row known-eye model response:      0.1467
```

Primary paired contrast:

```text
continuous tau - augmented 0x stabilized: -0.0104  CI [-0.0307, +0.0084]
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
