# Quadratic Joint Feature Diagnostic

No-anchor diagnostic using an origin-constrained quadratic compact response map and feature cosine endpoint.

Feature source: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/feature_latent_arrays.npz`
Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 10
Manifest rows: 96
Ridge: 0.01
Initial position mode: `inferred`
Quadratic continuation scales: `0.5,1`
Observation continuation scales: `1,1`

Overall:

    observer_mode prior_scale  n  image_accuracy  mean_feature_cosine  median_feature_cosine  mean_map_feature_cosine  mean_true_mass  median_N_eff_fraction
             zero         all 96        0.645833             0.891924               0.995838                 0.868872        0.500170               0.482196
            joint         all 96        0.864583             0.947419               0.948625                 0.977172        0.536094               0.591596
  best_single_tau         all 96        0.885417             0.944377               0.945151                 0.977870        0.525150               0.606503
linear_continuous         all 96        0.489583             0.824878               0.758892                 0.824535        0.444865               0.250001
quadratic_profile         all 96        0.604167             0.893307               0.997616                 0.886127        0.551180               0.252068
quadratic_poisson         all 96        0.812500             0.936487               0.937069                 0.954454        0.508226               0.581212
            known         all 96        1.000000             0.973354               0.973463                 1.000000        0.640763               0.514596
