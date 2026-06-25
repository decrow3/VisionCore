# Quadratic Joint Feature Diagnostic

No-anchor diagnostic using an origin-constrained quadratic compact response map and feature cosine endpoint.

Feature source: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/feature_latent_arrays.npz`
Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 10
Manifest rows: 4
Skip tables: 0
Prior family filter: ``
Scale filter: `1.0`
Ridge: 0.01
Initial position mode: `inferred`
Quadratic continuation scales: `1`
Observation continuation scales: `1`

Overall:

    observer_mode prior_scale  n  image_accuracy  mean_feature_cosine  median_feature_cosine  mean_map_feature_cosine  mean_true_mass  median_N_eff_fraction
             zero         all  4            0.00             0.730310               0.730310                 0.585684    1.064919e-02               0.575779
            joint         all  4            0.25             0.912013               0.908712                 0.847208    3.241787e-01               0.783001
  best_single_tau         all  4            0.75             0.923642               0.919982                 0.955934    3.660789e-01               0.791124
linear_continuous         all  4            0.00             0.655630               0.659211                 0.651992    5.462633e-10               0.250058
quadratic_profile         all  4            0.00             0.745910               0.790397                 0.717270    1.576790e-02               0.381374
quadratic_poisson         all  4            0.00             0.879841               0.882281                 0.684414    2.547579e-01               0.844380
            known         all  4            1.00             0.953565               0.953565                 1.000000    4.938962e-01               0.685012
