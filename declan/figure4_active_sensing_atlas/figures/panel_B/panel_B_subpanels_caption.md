# Panel B Subpanels

Generated cache-only from the cleaned BackImage aggregate FEM-information run:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched/
```

Subpanels:

- `B1_task_schematic`: analysis schematic for image features decoded from
  static-plus-motion response summaries.
- `B2_motion_family_qc`: RMS matching and path-length summaries for empirical,
  OU-like, Brownian, and rotated motion families.
- `B3_empirical_gain_vs_static`: empirical temporal-PCA feature-decoding gain
  over the static-only response.
- `B4_empirical_minus_controls`: empirical-minus-control incremental gain
  contrasts for Gabor k=4 temporal-PCA summaries.
- `B5_absolute_gain_guardrail`: absolute gains for all motion families, showing
  why Brownian/rotated caveats matter at larger scales.

Claim boundary:

```text
Legacy B3-B5 panels in this directory used deterministic V1-twin
feature-decoding gain in -MSE units. Those files are now archive/QC context.
The promoted 4B recompute uses the diagonal Gaussian decoder lower-bound
increment in bits over stabilized/static:
0.5 * sum_j log(var_static_j / var_condition_j) / log(2).
Full-covariance Ledoit-Wolf log-det values are supplemental robustness, not the
headline axis.
```
