# Continuous-Joint Feature-Recovery Diagnostics

This diagnostic asks whether each observer recovers the true candidate's local image feature vector, even when the discrete image-identification decision is wrong.

Feature source: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/feature_latent_arrays.npz`.
Primary latent for the plot: `pyramid_local_field`.

Best continuous-joint mean feature cosine: 0.9358 (No-anchor quadratic scale-calibrated).

Posterior feature cosine is the cosine between the true feature vector and the posterior-weighted candidate feature vector. MAP feature cosine is also written to the trial CSV so near-miss wrong-image choices can be inspected.

Model-development ranking should use posterior feature cosine first. Hard-negative image accuracy is retained as the stricter top-1 identity endpoint. See `continuous_joint_endpoint_metric_comparison.csv` and `continuous_joint_endpoint_metric_comparison.png` for the explicit rank comparison.

Production analyzer runs can emit calibrated continuous-joint posterior scores with `--continuous-posterior-temperature-by-scale 0.5:0.125,1.0:0.125,2.0:0.5`; raw scores remain available as `candidate_score_raw`.

In these diagnostics, `posterior_temperature` is the additional posthoc scorer temperature. `analyzer_posterior_temperature` preserves the temperature already emitted by analyzer rows.

Full calibrated analyzer artifact: `continuous_joint_quadratic_poisson_scale_conditioned_calibrated_full`; summary CSV: `continuous_joint_quadratic_scale_conditioned_calibrated_full_summary.csv`; all-scale feature cosine 0.93584 at unchanged image accuracy 0.7083.

Primary all-scale mean feature cosine:

| run_slug | run_label | best_single_tau | continuous_joint | joint | known | zero |
| --- | --- | --- | --- | --- | --- | --- |
| catalog_residual_all | Catalog residual all anchors | 0.9267 | 0.9300 | 0.9265 | 0.9593 | 0.8265 |
| catalog_residual_ctf_keep8 | Catalog residual CTF keep 8 | 0.9267 | 0.9299 | 0.9265 | 0.9593 | 0.8265 |
| catalog_residual_smooth6 | Catalog residual smoothed anchor | 0.9267 | 0.9301 | 0.9265 | 0.9593 | 0.8265 |
| catalog_residual_top2_shrink | Catalog residual top-2 shrink | 0.9267 | 0.9300 | 0.9265 | 0.9593 | 0.8265 |
| noanchor_ar1 | No-anchor AR(1) | 0.9267 | 0.8712 | 0.9265 | 0.9593 | 0.8265 |
| noanchor_axis_interleaved | No-anchor axis basis | 0.9267 | 0.8695 | 0.9265 | 0.9593 | 0.8265 |
| noanchor_brownian_ctf | No-anchor Brownian CTF | 0.9267 | 0.8711 | 0.9265 | 0.9593 | 0.8265 |
| noanchor_dct_ctf | No-anchor DCT CTF | 0.9267 | 0.8704 | 0.9265 | 0.9593 | 0.8265 |
| noanchor_quadratic_poisson | No-anchor quadratic | 0.9267 | 0.9098 | 0.9265 | 0.9593 | 0.8265 |
| noanchor_quadratic_scale_conditioned | No-anchor quadratic scale-conditioned | 0.9267 | 0.9108 | 0.9265 | 0.9593 | 0.8265 |
| noanchor_quadratic_scale_conditioned_calibrated | No-anchor quadratic scale-calibrated | 0.9267 | 0.9358 | 0.9265 | 0.9593 | 0.8265 |
| noanchor_quadratic_scale_conditioned_iter160 | No-anchor quadratic scale-conditioned iter160 | 0.9267 | 0.9107 | 0.9265 | 0.9593 | 0.8265 |
| noanchor_residual_ctf | No-anchor residual CTF | 0.9267 | 0.8700 | 0.9265 | 0.9593 | 0.8265 |
| poisson_k10 | Poisson k=10 | 0.9267 | 0.8425 | 0.9265 | 0.9593 | 0.8265 |
| poisson_k10_matched_brownian | Poisson k=10 Brownian | 0.9267 | 0.8659 | 0.9265 | 0.9593 | 0.8265 |
| poisson_k10_timevary | Poisson k=10 A(t) | 0.9267 | 0.8639 | 0.9265 | 0.9593 | 0.8265 |
| poisson_k10_timevary_smooth | Poisson k=10 smooth A(t) | 0.9267 | 0.8713 | 0.9265 | 0.9593 | 0.8265 |
| poisson_k20 | Poisson k=20 | 0.9267 | 0.8405 | 0.9265 | 0.9593 | 0.8265 |
| poisson_k20_timevary | Poisson k=20 A(t) | 0.9267 | 0.8622 | 0.9265 | 0.9593 | 0.8265 |
| poisson_k5 | Poisson k=5 | 0.9267 | 0.8420 | 0.9265 | 0.9593 | 0.8265 |
