# Polynomial Observation Diagnostic

This diagnostic compares origin-constrained linear/quadratic/cubic eye-position maps against affine variants.
Fits are evaluated with trajectory-held-out folds within each candidate image.

Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 10
Manifest rows: 96
Best model by mean CV R2: affine_quadratic (0.654455)

Overall:

      poly_model  degree  include_intercept  n_tables  median_coef_s2_over_s1  median_coef_s3_over_s1  mean_cv_r2_energy  mean_train_r2_energy
          linear       1              False        96            9.233141e-07                     NaN          -0.001832              0.003827
       quadratic       2              False        96            6.708629e-03            6.879029e-09           0.220880              0.320055
           cubic       3              False        96            2.111225e-02            4.095710e-03           0.227911              0.338559
   affine_linear       1               True        96            2.075813e-01            3.339586e-07           0.603193              0.637928
affine_quadratic       2               True        96            2.194940e-02            9.024762e-03           0.654455              0.697085
