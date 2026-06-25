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

Decoder mode: `mlp`

The feature target is a compact PCA-space embedding of the existing local
`pyramid_local_field` feature array. The plotted feature-space
option is `fold_zscore_whitened_pca`. The response target is
the image-disjoint compact response basis used by the promoted 4C
continuous observer. Cross-fitting is by source image: no response sample
whose target source row is in the held-out fold is used to fit that fold.

At the 1x scale:

```text
known eye feature cosine:          -0.0143
hidden eye feature cosine:         -0.0751
zero-eye model on motion:          -0.0017
0x stabilized feature cosine:      0.4895
```

All-scale paired contrasts:

```text
known - hidden:                    0.0471
hidden - zero-eye model:           -0.0738
known motion - 0x stabilized:      -0.3856
hidden motion - 0x stabilized:     -0.4327
```

All-scale option means:

```csv
decoder_mode,feature_space_mode,known,hidden,zero_eye_model,zero_static
mlp,fold_zscore_whitened_pca,0.1572,0.1101,0.1839,0.5428
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
