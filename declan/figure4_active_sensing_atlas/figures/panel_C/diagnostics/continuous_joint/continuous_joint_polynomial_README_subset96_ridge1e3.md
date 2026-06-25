# Polynomial Observation Diagnostic

This diagnostic compares origin-constrained linear/quadratic/cubic eye-position maps against affine variants.
Fits are evaluated with trajectory-held-out folds within each candidate image.

Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 10
Manifest rows: 96
Best model by mean CV R2: affine_quadratic (0.659512)

Overall:

      poly_model  degree  include_intercept  n_tables  median_coef_s2_over_s1  median_coef_s3_over_s1  mean_cv_r2_energy  mean_train_r2_energy
          linear       1              False        96                0.000009                     NaN          -0.001894              0.003827
       quadratic       2              False        96                0.005560            5.896590e-08           0.187478              0.344117
           cubic       3              False        96                0.073437            3.427947e-03           0.137833              0.374242
   affine_linear       1               True        96                0.207508            3.334572e-06           0.603129              0.637929
affine_quadratic       2               True        96                0.018976            7.022388e-03           0.659512              0.708631
