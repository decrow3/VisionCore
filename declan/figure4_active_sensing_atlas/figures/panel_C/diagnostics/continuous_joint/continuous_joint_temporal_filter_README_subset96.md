# Constrained Temporal-Filter Observation Diagnostic

This diagnostic compares instantaneous eye position against causal delay, boxcar, and EMA filters that keep the eye regressor two-dimensional.
Fits are evaluated with trajectory-held-out folds within each candidate image.

Basis: `/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_disjoint_compact_basis_delta025_v1/image_disjoint_compact_basis_delta0p25_fold0of2.npz`
Basis dim: 10
Manifest rows: 96
Best filter by mean CV R2: instant (-0.00190151)

Overall:

filter_model filter_family  filter_parameter  n_tables  median_coef_s2_over_s1  mean_cv_r2_energy  mean_train_r2_energy
     instant         delay              0.00        96                0.009199          -0.001902              0.003827
      delay1         delay              1.00        96                0.005543          -0.004286              0.005559
      delay2         delay              2.00        96                0.005694          -0.007628              0.007640
      delay3         delay              3.00        96                0.005146          -0.011928              0.010132
      delay4         delay              4.00        96                0.005495          -0.016346              0.013548
      delay6         delay              6.00        96                0.004469          -0.026407              0.022604
      delay8         delay              8.00        96                0.004395          -0.038381              0.033853
        box2        boxcar              2.00        96                0.006831          -0.002924              0.004913
        box4        boxcar              4.00        96                0.004335          -0.005896              0.007252
        box8        boxcar              8.00        96                0.003749          -0.014208              0.014024
     ema0p25           ema              0.25        96                0.007279          -0.002500              0.004560
     ema0p50           ema              0.50        96                0.004894          -0.004107              0.006059
     ema0p75           ema              0.75        96                0.003784          -0.011673              0.012471
     ema0p90           ema              0.90        96                0.002981          -0.046402              0.045506
