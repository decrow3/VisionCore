# Quadratic Joint Feature Diagnostic

No-anchor diagnostic using an origin-constrained quadratic compact response map and feature cosine endpoint.

Feature source: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/feature_latent_arrays.npz`
Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 10
Manifest rows: 4
Ridge: 0.01
Initial position mode: `inferred`

Overall:

 index     observer_mode prior_scale  n  image_accuracy  mean_feature_cosine  median_feature_cosine  mean_map_feature_cosine  mean_true_mass  median_N_eff_fraction
     0              zero         all  4            0.00             0.739197               0.739197                 0.576918    1.144600e-01               0.769204
     1             joint         all  4            0.75             0.907276               0.907276                 0.935340    3.677940e-01               0.815407
     2   best_single_tau         all  4            1.00             0.915527               0.915527                 1.000000    3.935059e-01               0.809177
     3 linear_continuous         all  4            0.00             0.643797               0.643797                 0.643226    2.950137e-08               0.251581
     4 quadratic_profile         all  4            0.25             0.806526               0.806526                 0.756953    3.275782e-01               0.379998
     5 quadratic_poisson         all  4            0.50             0.869801               0.869801                 0.772546    2.903248e-01               0.862270
     6             known         all  4            1.00             0.947133               0.947133                 1.000000    5.054196e-01               0.662747
