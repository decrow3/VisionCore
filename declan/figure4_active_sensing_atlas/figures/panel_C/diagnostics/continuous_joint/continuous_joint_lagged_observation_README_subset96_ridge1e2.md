# Lagged Observation Diagnostic

This diagnostic compares the current instantaneous compact response model against causal lagged eye-position designs.
Fits are evaluated with trajectory-held-out folds within each candidate image.

Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 10
Manifest rows: 96

Overall:

   lag_model      lags  n_lags  n_tables  median_coef_s2_over_s1  median_coef_s3_over_s1  mean_cv_r2_energy  mean_train_r2_energy
     instant         0       1        96            9.233141e-07                     NaN          -0.001832              0.003827
      lag0_1       0,1       2        96            1.651239e-01            4.884528e-07          -0.006286              0.006897
    lag0_1_2     0,1,2       3        96            1.774402e-01            7.897077e-02          -0.014461              0.010726
  lag0_1_2_4   0,1,2,4       4        96            3.106224e-01            9.831708e-02          -0.032550              0.020299
lag0_1_2_4_8 0,1,2,4,8       5        96            2.749672e-01            1.373763e-01          -0.055518              0.049198
