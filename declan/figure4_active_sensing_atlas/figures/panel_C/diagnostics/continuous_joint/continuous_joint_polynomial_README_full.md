# Polynomial Observation Diagnostic

This diagnostic compares origin-constrained linear/quadratic/cubic eye-position maps against affine variants.
Fits are evaluated with trajectory-held-out folds within each candidate image.

Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 10
Manifest rows: 768
Best model by mean CV R2: affine_quadratic (0.618513)

Overall:

      poly_model  degree  include_intercept  n_tables  median_coef_s2_over_s1  median_coef_s3_over_s1  mean_cv_r2_energy  mean_train_r2_energy
          linear       1              False       768                0.006039                     NaN          -0.000718              0.005063
       quadratic       2              False       768                0.005975                0.000041           0.082344              0.320806
           cubic       3              False       768                0.041081                0.001860          -0.552220              0.374048
   affine_linear       1               True       768                0.205268                0.002043           0.602140              0.637164
affine_quadratic       2               True       768                0.016708                0.007267           0.618513              0.697092
