# Lagged Observation Diagnostic

This diagnostic compares the current instantaneous compact response model against causal lagged eye-position designs.
Fits are evaluated with trajectory-held-out folds within each candidate image.

Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 10
Manifest rows: 768

Overall:

   lag_model      lags  n_lags  n_tables  median_coef_s2_over_s1  median_coef_s3_over_s1  mean_cv_r2_energy  mean_train_r2_energy
     instant         0       1       768                0.006039                     NaN          -0.000718              0.005063
      lag0_1       0,1       2       768                0.180275                0.002441          -0.004773              0.009754
    lag0_1_2     0,1,2       3       768                0.164400                0.078241          -0.011654              0.015658
  lag0_1_2_4   0,1,2,4       4       768                0.279199                0.091848          -0.025961              0.029915
lag0_1_2_4_8 0,1,2,4,8       5       768                0.259945                0.119696          -0.062055              0.064360
