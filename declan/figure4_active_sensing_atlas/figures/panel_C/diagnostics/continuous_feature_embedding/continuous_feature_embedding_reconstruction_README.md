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

Decoder mode: `linear_gaussian`
Linear-Gaussian decoder with a compact feature prior.

The feature target is a compact PCA-space embedding of the existing local
`pyramid_local_field` feature array. The plotted feature-space
option is `fold_zscore_whitened_pca`. The response target is
the image-disjoint compact response basis used by the promoted 4C
continuous observer. Cross-fitting is by source image: no response sample
whose target source row is in the held-out fold is used to fit that fold.

At the 1x scale:

```text
known eye feature cosine:          0.1700
hidden eye feature cosine:         0.1419
zero-eye model on motion:          0.0509
0x stabilized feature cosine:      0.2330
```

All-scale paired contrasts:

```text
known - hidden:                    0.0234
hidden - zero-eye model:           0.0653
known motion - 0x stabilized:      -0.0825
hidden motion - 0x stabilized:     -0.1059
```

All-scale option means:

```csv
decoder_mode,feature_space_mode,known,hidden,zero_eye_model,zero_static
linear_gaussian,fold_centered_whitened_pca,0.1203,0.0994,0.0567,0.1896
linear_gaussian,fold_zscore_pca,0.1372,0.1346,0.1368,0.3218
linear_gaussian,fold_zscore_whitened_pca,0.1505,0.1271,0.0618,0.2330
linear_gaussian,global_centered_whitened_pca,0.1261,0.1036,0.0471,0.1860
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
