# Polynomial Observation Diagnostic

This diagnostic compares origin-constrained linear/quadratic/cubic eye-position maps against affine variants.
Fits are evaluated with trajectory-held-out folds within each candidate image.

Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 10
Manifest rows: 768
Best model by mean CV R2: affine_quadratic (0.643358)

Overall:

      poly_model  degree  include_intercept  n_tables  median_coef_s2_over_s1  median_coef_s3_over_s1  mean_cv_r2_energy  mean_train_r2_energy
          linear       1              False       768            6.059612e-07                     NaN          -0.000626              0.005062
       quadratic       2              False       768            7.051859e-03            4.759133e-09           0.204048              0.296819
           cubic       3              False       768            2.019901e-02            4.601594e-03           0.211382              0.318353
   affine_linear       1               True       768            2.069615e-01            2.048985e-07           0.602233              0.637164
affine_quadratic       2               True       768            2.059754e-02            9.357341e-03           0.643358              0.687562
