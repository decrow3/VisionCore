# Endpoint-History Objective Completion Audit

Status date: 2026-07-06

## Objective

Find an implementation where:

1. Motion beats static.
2. Known beats unknown.
3. Jointly decoding / marginalizing trajectory helps.
4. The implementation remains scientifically rigorous and grounded in established methods.

## Promoted implementation

Output:

`outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_primary_beta_scale1_cached_v1`

Gate tables:

- Zero-history unknown baseline:
  `outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_primary_beta_scale1_cached_v1/gates_known_joint_zero_static/unified_feature_observer_gate_table.csv`
- Response-only unknown baseline:
  `outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_primary_beta_scale1_cached_v1/gates_known_joint_responseonly_static/unified_feature_observer_gate_table.csv`

Configuration:

- Endpoint-aligned 32-frame history assay.
- Terminal-frame readout only.
- Primary test history: empirical endpoint history.
- Training histories: empirical, OU, Brownian endpoint histories.
- Response population: RR100.
- Response basis: full units.
- Feature target: `pyramid_local_field`.
- Feature coordinate: fold-fit `fold_zscore_whitened_pca`.
- Feature dimension: 4.
- History coordinate: train-fold PCA.
- History dimension: 8.
- Score: pooled multi-output `R2_cv` in the locked, train-normalized feature space.

Workhorse file map:

`declan/figure4_active_sensing_atlas/endpoint_history_workhorse_files.md`

Primary figure:

`outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_primary_beta_scale1_cached_v1/main_results_figures/endpoint_history_main_results.png`

## Gate Evidence

All scores below use source-cluster bootstrap intervals from the gate-table builder.

| Requirement | Evidence | Status |
|---|---:|---|
| Motion beats static | Joint - static = 0.6543, CI [0.2059, 1.1246] | Pass |
| Known beats zero-history unknown | Known - zero = 0.9879, CI [0.7752, 1.2774] | Pass |
| Joint helps vs zero-history unknown | Joint - zero = 0.9879, CI [0.7752, 1.2774] | Pass |
| Known beats response-only unknown | Known - response-only = 0.8675, CI [0.6688, 1.1165] | Pass |
| Joint helps vs response-only unknown | Joint - response-only = 0.8675, CI [0.6688, 1.1165] | Pass |

The corresponding all-row pooled scores are:

| Observer | Mode | R2_cv |
|---|---:|---:|
| Known history | `known_history_generative` | -1.4739 |
| Joint history | `joint_history_generative` | -1.4739 |
| Zero history | `zero_history_generative_on_motion` | -2.4618 |
| Response-only hidden history | `joint_history_response_only` | -2.3414 |
| Static history | `static_history` | -2.1283 |

## Rigour Checks

- Feature transforms are fold-fit and source-disjoint.
- Scoring uses pooled multi-output `R2_cv` from out-of-fold SSE/SST, not unweighted fold means.
- Endpoint alignment removes current-position confound: all trajectories end at zero displacement.
- Readout uses terminal response only, preventing full-movie target mismatch.
- Multi-history training gives repeated endpoint histories per source, separating image identity from history condition.
- Gate intervals are source-cluster bootstrapped.
- Cached dataset reuse is guarded by condition availability and endpoint-history contract checks.

## Important Caveat

This completes the stated gate objective, but it does not establish a strict known trajectory ceiling over the joint latent observer:

`known_history_generative = joint_history_generative`

The known observer selects the beta=0 fallback, so the current paper-facing claim should be phrased as:

> accounting for endpoint-aligned history, either known or latent/joint, improves feature recovery relative to zero-history and response-only unknown-history baselines, and the motion/history observer beats the static endpoint-history control.

It should not be phrased as:

> known trajectory strictly exceeds the latent joint observer.

Additional diagnostics tested and failed to rescue strict known > joint:

- primary-history-aware beta selection,
- source-centered repeated-measures history adjustment,
- correlated feature-history Gaussian prior,
- alternate OU and Brownian primary histories.

## Family and axis follow-ups

The empirical/OU/Brownian comparison is useful for checking whether the effect
is specific to recorded-like endpoint bridges:

`outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_primary_beta_scale1_cached_v1/main_results_figures/endpoint_history_family_comparisons.png`

Brownian and empirical primary histories pass the weaker endpoint-history gates.
OU gives a positive joint-static point estimate but its interval crosses zero.
None of the three families rescues a strict `known > joint` ceiling.

The true edge-parallel versus edge-orthogonal endpoint-history control is:

`outputs/figure4_endpoint_history_feature_readout_rr100_n128_axis_parallel_orthogonal_fdim4_hpc8_scale1_v1/main_results_figures/endpoint_history_axis_edge_comparison.png`

This control confirms that both edge-parallel and edge-orthogonal endpoint
histories support the history-use gates. The direct orthogonal-minus-parallel
joint contrast is small and not outside its bootstrap interval:

```text
joint edge-orthogonal - edge-parallel = -0.017, CI [-0.262, 0.245]
```

Therefore the current endpoint-history branch should not be used to claim a
reliable along-edge advantage. It supports a history-accounting effect, not a
settled along/across mechanism.

This is not the same contract as the older Panel D axis result. Panel D used a
candidate-posterior observer with `axis_catalog_mode=per_candidate`, so every
candidate patch was evaluated with trajectories rendered parallel or normal to
that candidate's own edge. The endpoint-history control evaluates
true-source endpoint features under matched source-axis trajectories and lacks
that candidate-conditioned axis layer. This distinction is claim-critical when
comparing the old along/across result with the new decoder.
