# Continuous Feature-Embedding Reconstruction

This diagnostic is the first 4C branch that infers a continuous compact
feature embedding instead of selecting or posterior-averaging over the
candidate image list.

Model:

```text
response_features = A z + noise
z ~ N(0, I)
z_hat = E[z | response]
```

The feature target is a whitened PCA embedding of the existing local
`pyramid_local_field` feature array. The response target is
the image-disjoint compact response basis used by the promoted 4C
continuous observer. Cross-fitting is by source image: no response sample
whose target source row is in the held-out fold is used to fit that fold.

At the 1x scale:

```text
known eye feature cosine:          0.2908
hidden eye feature cosine:         0.0750
zero-eye model on motion:          0.1306
0x stabilized feature cosine:      0.2117
```

All-scale paired contrasts:

```text
known - hidden:                    0.1931
hidden - zero-eye model:           -0.0593
known motion - 0x stabilized:      0.0469
hidden motion - 0x stabilized:     -0.1463
```

Interpretation boundary: this is a continuous feature posterior, not a
pixel MAP reconstruction and not a candidate posterior. The finite image
set is still used to fit the empirical feature prior/encoder and to score
held-out source rows.

Outputs:

- `continuous_feature_embedding_reconstruction_trials.csv`
- `continuous_feature_embedding_reconstruction_summary.csv`
- `continuous_feature_embedding_reconstruction_contrasts.csv`
- `continuous_feature_embedding_reconstruction_models.csv`
- `continuous_feature_embedding_reconstruction_manifest.json`
- `continuous_feature_embedding_reconstruction.png`
