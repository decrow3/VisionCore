# Constrained Temporal-Filter Observation Diagnostic

This diagnostic compares instantaneous eye position against causal delay, boxcar, and EMA filters that keep the eye regressor two-dimensional.
Fits are evaluated with trajectory-held-out folds within each candidate image.

Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 10
Manifest rows: 768
Best filter by mean CV R2: instant (-0.000718393)

Overall:

filter_model filter_family  filter_parameter  n_tables  median_coef_s2_over_s1  mean_cv_r2_energy  mean_train_r2_energy
     instant         delay              0.00       768                0.006039          -0.000718              0.005063
      delay1         delay              1.00       768                0.004952          -0.001827              0.007833
      delay2         delay              2.00       768                0.004203          -0.003840              0.011133
      delay3         delay              3.00       768                0.004057          -0.006659              0.015036
      delay4         delay              4.00       768                0.004057          -0.010233              0.020107
      delay6         delay              6.00       768                0.003654          -0.021878              0.032697
      delay8         delay              8.00       768                0.003682          -0.037140              0.046705
        box2        boxcar              2.00       768                0.004717          -0.001303              0.006672
        box4        boxcar              4.00       768                0.003483          -0.003319              0.010220
        box8        boxcar              8.00       768                0.002366          -0.010742              0.020146
     ema0p25           ema              0.25       768                0.004878          -0.001074              0.006134
     ema0p50           ema              0.50       768                0.003826          -0.002123              0.008424
     ema0p75           ema              0.75       768                0.002985          -0.008459              0.017947
     ema0p90           ema              0.90       768                0.001780          -0.046131              0.059776
