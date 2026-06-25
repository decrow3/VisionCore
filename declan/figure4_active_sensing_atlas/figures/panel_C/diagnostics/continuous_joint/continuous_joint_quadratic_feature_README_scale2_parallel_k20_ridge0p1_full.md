# Quadratic Joint Feature Diagnostic

No-anchor diagnostic using an origin-constrained quadratic compact response map and feature cosine endpoint.

Feature source: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/feature_latent_arrays.npz`
Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 20
Manifest rows: 128
Skip tables: 0
Prior family filter: `axis_edge_parallel`
Scale filter: `2.0`
Ridge: 0.1
Initial position mode: `inferred`
Quadratic continuation scales: `1`
Observation continuation scales: `1`

Overall:

    observer_mode prior_scale   n  image_accuracy  mean_feature_cosine  median_feature_cosine  mean_map_feature_cosine  mean_true_mass  median_N_eff_fraction
             zero         all 128        0.335938             0.778714               0.816105                 0.736833        0.307724               0.373064
            joint         all 128        0.679688             0.918361               0.946745                 0.916514        0.459675               0.708746
  best_single_tau         all 128        0.664062             0.917340               0.940098                 0.905059        0.453877               0.742456
linear_continuous         all 128        0.242188             0.701623               0.606558                 0.701414        0.220281               0.250000
quadratic_profile         all 128        0.328125             0.779062               0.791374                 0.762413        0.323388               0.250036
quadratic_poisson         all 128        0.617188             0.904574               0.919303                 0.867703        0.413884               0.804182
            known         all 128        1.000000             0.966812               0.973478                 1.000000        0.609514               0.567595
