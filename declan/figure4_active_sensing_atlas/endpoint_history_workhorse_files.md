# Endpoint-History Workhorse Files

Status date: 2026-07-06

## Core analysis contract

The endpoint-history branch tests feature recovery from the final model state
after 32-frame retinal histories that all end at the same endpoint. The readout
uses the terminal response only, so the target feature is the endpoint image
feature rather than a shifted full-movie target.

Current paper-facing result:

```text
known history = joint latent history > zero/unknown history
known history = joint latent history > response-only hidden history
joint latent history > static history
```

This is a contrastive endpoint-history result. It is not a strict
`known > joint` ceiling result.

## Main scripts

| Role | File |
|---|---|
| Core endpoint-history runner | `declan/figure4_active_sensing_atlas/scripts/build_panel_c_endpoint_history_feature_readout.py` |
| Shared gate-table builder | `declan/figure4_active_sensing_atlas/scripts/build_unified_feature_observer_gate_table.py` |
| Main result figure | `declan/figure4_active_sensing_atlas/scripts/plot_endpoint_history_main_results.py` |
| Empirical/OU/Brownian comparison figure | `declan/figure4_active_sensing_atlas/scripts/plot_endpoint_history_family_comparisons.py` |
| True edge-parallel versus edge-orthogonal comparison figure | `declan/figure4_active_sensing_atlas/scripts/plot_endpoint_history_axis_edge_comparison.py` |
| Feature-dimension sweep figure | `declan/figure4_active_sensing_atlas/scripts/plot_endpoint_history_feature_dim_sweep.py` |
| Endpoint-history/helper tests | `declan/figure4_active_sensing_atlas/tests/test_panel_c_continuous_feature_embedding_reconstruction.py` |

## Canonical outputs

Primary endpoint-history run:

```text
outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_primary_beta_scale1_cached_v1
```

Primary gate table:

```text
outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_primary_beta_scale1_cached_v1/gates_known_joint_zero_static/unified_feature_observer_gate_table.csv
```

Primary main-results figure:

```text
outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_primary_beta_scale1_cached_v1/main_results_figures/endpoint_history_main_results.png
```

Empirical/OU/Brownian comparison figure:

```text
outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_primary_beta_scale1_cached_v1/main_results_figures/endpoint_history_family_comparisons.png
```

True edge-parallel versus edge-orthogonal output:

```text
outputs/figure4_endpoint_history_feature_readout_rr100_n128_axis_parallel_orthogonal_fdim4_hpc8_scale1_v1
```

True edge-parallel versus edge-orthogonal figure:

```text
outputs/figure4_endpoint_history_feature_readout_rr100_n128_axis_parallel_orthogonal_fdim4_hpc8_scale1_v1/main_results_figures/endpoint_history_axis_edge_comparison.png
```

## Current numeric anchors

Primary empirical endpoint-history scores use RR100, full-unit responses,
`pyramid_local_field`, `fold_zscore_whitened_pca`, feature dimension 4, history
PCA dimension 8, and pooled multi-output `R2_cv`.

| Observer | R2_cv |
|---|---:|
| Known history | -1.4739 |
| Joint history | -1.4739 |
| Zero-history generative on motion | -2.4618 |
| Response-only hidden history | -2.3414 |
| Static history | -2.1283 |

Primary contrasts:

| Contrast | Delta R2_cv | 95% CI |
|---|---:|---:|
| Joint - static | 0.6543 | [0.2059, 1.1246] |
| Joint - zero | 0.9879 | [0.7752, 1.2774] |
| Known - zero | 0.9879 | [0.7752, 1.2774] |
| Joint - response-only | 0.8675 | [0.6688, 1.1165] |
| Known - joint | 0.0000 | [0.0000, 0.0000] |

The edge-axis endpoint-history control gives a directionally edge-parallel
advantage but not a reliable one:

| Direct comparison | Delta R2_cv | 95% CI |
|---|---:|---:|
| Joint, edge-orthogonal - edge-parallel | -0.017 | [-0.262, 0.245] |
| Zero-history, edge-orthogonal - edge-parallel | -0.068 | [-0.408, 0.302] |

Both edge-parallel and edge-orthogonal histories pass the history-use gates, so
the safe interpretation is that the endpoint-history benefit is not specific to
edge-parallel motion in the current RR100 run.

Do not read this edge-axis control as a direct replacement for the older Panel D
axis-conditioned feature posterior. The older result used
`axis_catalog_mode=per_candidate`, rendering parallel/orthogonal prior
trajectories relative to each candidate patch's own edge. The endpoint-history
axis control uses matched source-axis trajectories for a true-source endpoint
feature readout and does not include the candidate-conditioned axis hypothesis
layer.
