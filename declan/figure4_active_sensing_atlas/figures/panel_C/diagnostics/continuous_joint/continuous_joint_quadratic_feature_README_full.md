# Quadratic Joint Feature Diagnostic

No-anchor diagnostic using an origin-constrained quadratic compact response map and feature cosine endpoint.

Feature source: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/feature_latent_arrays.npz`
Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 10
Manifest rows: 768
Skip tables: 0
Prior family filter: ``
Scale filter: ``
Ridge: 0.01
Initial position mode: `inferred`
Quadratic continuation scales: `1`
Observation continuation scales: `1`

Overall:

    observer_mode prior_scale   n  image_accuracy  mean_feature_cosine  median_feature_cosine  mean_map_feature_cosine  mean_true_mass  median_N_eff_fraction
             zero         all 768        0.445312             0.826462               0.886181                 0.790122        0.348009               0.506095
            joint         all 768        0.769531             0.926532               0.945896                 0.946103        0.466446               0.760208
  best_single_tau         all 768        0.783854             0.926677               0.945151                 0.949350        0.463242               0.773699
linear_continuous         all 768        0.335938             0.761355               0.674620                 0.753816        0.316193               0.250046
quadratic_profile         all 768        0.514323             0.857114               0.992243                 0.843656        0.483567               0.263568
quadratic_poisson         all 768        0.691406             0.909775               0.919664                 0.916701        0.415564               0.839086
            known         all 768        1.000000             0.959315               0.970460                 1.000000        0.574005               0.613555
