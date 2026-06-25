# Quadratic Joint Feature Diagnostic

No-anchor diagnostic using an origin-constrained quadratic compact response map and feature cosine endpoint.

Feature source: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k2_4_8_16_32_uncertainty_v1/feature_latent_arrays.npz`
Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 20
Manifest rows: 64
Skip tables: 0
Prior family filter: `axis_edge_parallel`
Scale filter: `2.0`
Ridge: 0.1
Initial position mode: `inferred`
Quadratic continuation scales: `1`
Observation continuation scales: `1`

Overall:

    observer_mode prior_scale  n  image_accuracy  mean_feature_cosine  median_feature_cosine  mean_map_feature_cosine  mean_true_mass  median_N_eff_fraction
             zero         all 64        0.437500             0.792432               0.859879                 0.775159        0.344348               0.405978
            joint         all 64        0.703125             0.915047               0.944404                 0.924666        0.474907               0.692780
  best_single_tau         all 64        0.703125             0.913789               0.938495                 0.907658        0.469869               0.668689
linear_continuous         all 64        0.234375             0.695212               0.594481                 0.695628        0.227511               0.250000
quadratic_profile         all 64        0.359375             0.782951               0.842765                 0.771043        0.350044               0.250022
quadratic_poisson         all 64        0.671875             0.899268               0.909852                 0.891540        0.422933               0.810033
            known         all 64        1.000000             0.966413               0.974923                 1.000000        0.635410               0.531331
