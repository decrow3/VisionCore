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

The feature target is a compact PCA-space embedding of the existing local
`pyramid_local_field` feature array. The plotted feature-space
option is `fold_zscore_whitened_pca`. The response target is
the image-disjoint compact response basis used by the promoted 4C
continuous observer. Cross-fitting is by source image: no response sample
whose target source row is in the held-out fold is used to fit that fold.

At the 1x scale:

```text
known eye feature cosine:          0.7241
hidden eye feature cosine:         0.6005
zero-eye model on motion:          -0.1155
0x stabilized feature cosine:      0.6123
```

All-scale paired contrasts:

```text
known - hidden:                    0.1277
hidden - zero-eye model:           0.2719
known motion - 0x stabilized:      -0.2264
hidden motion - 0x stabilized:     -0.3541
```

All-scale option means:

```csv
feature_space_mode,known,hidden,zero_eye_model,zero_static
fold_centered_whitened_pca,0.3661,0.1329,0.0967,0.7078
fold_zscore_pca,0.3471,0.0588,-0.0338,0.5953
fold_zscore_whitened_pca,0.3705,0.2428,-0.0291,0.5969
global_centered_whitened_pca,0.4278,0.2057,0.2683,0.6481
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
