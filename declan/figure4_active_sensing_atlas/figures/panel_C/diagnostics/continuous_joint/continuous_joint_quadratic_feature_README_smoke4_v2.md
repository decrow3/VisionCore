# Quadratic Joint Feature Diagnostic

No-anchor diagnostic using an origin-constrained quadratic compact response map and feature cosine endpoint.

Feature source: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/feature_latent_arrays.npz`
Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 10
Manifest rows: 4
Ridge: 0.01
Initial position mode: `inferred`

Overall:

    observer_mode prior_scale  n  image_accuracy  mean_feature_cosine  median_feature_cosine  mean_map_feature_cosine  mean_true_mass  median_N_eff_fraction
             zero         all  4            0.00             0.739197               0.739197                 0.576918        0.114460               0.769204
            joint         all  4            0.75             0.907276               0.907276                 0.935340        0.367794               0.815407
  best_single_tau         all  4            1.00             0.915527               0.915527                 1.000000        0.393506               0.809177
linear_continuous         all  4            0.00             0.649499               0.649499                 0.643226        0.004138               0.259328
quadratic_profile         all  4            0.25             0.806526               0.806526                 0.756953        0.327578               0.379998
quadratic_poisson         all  4            0.50             0.869801               0.869801                 0.772546        0.290325               0.862270
            known         all  4            1.00             0.947133               0.947133                 1.000000        0.505420               0.662747
