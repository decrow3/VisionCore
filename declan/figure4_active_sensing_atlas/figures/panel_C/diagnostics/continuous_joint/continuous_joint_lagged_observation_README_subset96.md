# Lagged Observation Diagnostic

This diagnostic compares the current instantaneous compact response model against causal lagged eye-position designs.
Fits are evaluated with trajectory-held-out folds within each candidate image.

Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 10
Manifest rows: 96

Overall:

   lag_model      lags  n_lags  n_tables  median_coef_s2_over_s1  median_coef_s3_over_s1  mean_cv_r2_energy  mean_train_r2_energy
     instant         0       1        96                0.009199                     NaN          -0.001902              0.003827
      lag0_1       0,1       2        96                0.164258                0.004815          -0.007005              0.006906
    lag0_1_2     0,1,2       3        96                0.178371                0.079374          -0.016156              0.010735
  lag0_1_2_4   0,1,2,4       4        96                0.316348                0.098212          -0.034937              0.020309
lag0_1_2_4_8 0,1,2,4,8       5        96                0.276484                0.139443          -0.057529              0.049210
