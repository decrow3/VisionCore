# Polynomial Observation Diagnostic

This diagnostic compares origin-constrained linear/quadratic/cubic eye-position maps against affine variants.
Fits are evaluated with trajectory-held-out folds within each candidate image.

Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 10
Manifest rows: 96
Best model by mean CV R2: affine_quadratic (0.648777)

Overall:

      poly_model  degree  include_intercept  n_tables  median_coef_s2_over_s1  median_coef_s3_over_s1  mean_cv_r2_energy  mean_train_r2_energy
          linear       1              False        96                0.000092                     NaN          -0.001901              0.003827
       quadratic       2              False        96                0.005518            5.805448e-07           0.145909              0.345102
           cubic       3              False        96                0.076124            2.291553e-03          -0.219027              0.381887
   affine_linear       1               True        96                0.207501            3.334069e-05           0.603123              0.637929
affine_quadratic       2               True        96                0.018645            6.851448e-03           0.648777              0.709237
