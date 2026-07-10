# Endpoint-History Candidate Gate Status

Status date: 2026-07-06

## Candidate configuration

The current strongest endpoint-history candidate is the n=128
primary-history-aware beta-selection run:

- Script: `declan.figure4_active_sensing_atlas.scripts.build_panel_c_endpoint_history_feature_readout`
- Output: `outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_primary_beta_scale1_cached_v1`
- Response population: RR100
- Response basis: full units
- Assay: 32-frame endpoint-aligned histories, terminal frame readout only
- Primary test history: empirical endpoint history
- Multi-history training bank: empirical, OU, Brownian endpoint histories
- Feature target: `pyramid_local_field`
- Feature coordinate: `fold_zscore_whitened_pca`
- Feature dimension: 4
- History coordinate: train-fold PCA
- History dimension: 8
- Score: pooled multi-output `R2_cv` in the locked, train-normalized feature space

## Primary n=128 R2 gate table

Gate table:

`outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_primary_beta_scale1_cached_v1/gates_known_joint_zero_static/unified_feature_observer_gate_table.csv`

All-row pooled scores:

| Observer | Mode | R2_cv |
|---|---:|---:|
| Known history | `known_history_generative` | -1.4739 |
| Joint history | `joint_history_generative` | -1.4739 |
| Zero history | `zero_history_generative_on_motion` | -2.4618 |
| Response-only hidden history | `joint_history_response_only` | -2.3414 |
| Static history | `static_history` | -2.1283 |

Contrasts with source-cluster bootstrap intervals:

| Contrast | Delta R2_cv | 95% CI |
|---|---:|---:|
| Known - zero | 0.9879 | [0.7752, 1.2774] |
| Joint - zero | 0.9879 | [0.7752, 1.2774] |
| Joint - response-only | 0.8675 | [0.6688, 1.1165] |
| Joint - static | 0.6543 | [0.2059, 1.1246] |
| Known - joint | 0.0000 | [0.0000, 0.0000] |

## Earlier n=64 pilot

The n=64 pilot found the same qualitative pattern:

`outputs/figure4_endpoint_history_feature_readout_rr100_n64_multi_history_fdim4_hpc8_scale1_cached_v1`

| Contrast | Delta R2_cv | 95% CI |
|---|---:|---:|
| Known - zero | 1.3834 | [0.9039, 1.9677] |
| Joint - zero | 1.3834 | [0.9039, 1.9677] |
| Joint - static | 0.7914 | [0.0776, 1.5679] |
| Known - joint | 0.0000 | [0.0000, 0.0000] |

The n=128 run strengthens the contrastive endpoint-history result and makes the
motion-versus-static gate cleaner. It also confirms that the strict
known-ceiling problem is not obviously an n=64 power issue: the known-history
generative observer again selected beta=0 in every outer fold.

## Larger n=128 replication before primary-beta correction

The same assay was also run with 128 endpoint images before the
primary-history-aware beta-selection correction:

`outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_scale1_v1`

`outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_scale1_v1/gates_known_joint_zero_static/unified_feature_observer_gate_table.csv`

All-row pooled scores:

| Observer | Mode | R2_cv |
|---|---:|---:|
| Known history | `known_history_generative` | -1.4739 |
| Joint history | `joint_history_generative` | -1.4739 |
| Zero history | `zero_history_generative_on_motion` | -2.4618 |
| Static history | `static_history` | -2.1283 |

Contrasts with source-cluster bootstrap intervals:

| Contrast | Delta R2_cv | 95% CI |
|---|---:|---:|
| Known - zero | 0.9879 | [0.7752, 1.2774] |
| Joint - zero | 0.9879 | [0.7752, 1.2774] |
| Joint - static | 0.6543 | [0.2059, 1.1246] |
| Known - joint | 0.0000 | [0.0000, 0.0000] |

A source-centered repeated-measures history-adjustment observer was also tested
at n=128. It did not rescue the strict known ceiling; its gamma was 0 in every
outer fold and it underperformed the joint generative observer.

## Additional strict-ceiling diagnostics

Two additional diagnostics were run after the n=128 replication.

### Primary-history-aware beta selection

The known-history generative observer originally selected the shrinkage beta on
the full multi-history training bank. This was changed so beta is selected only
on primary-history rows from the outer training sources, matching the empirical
test contract more closely.

Output:

`outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_primary_beta_scale1_cached_v1`

Result: beta remained 0 in every outer fold. The strict known > joint ceiling
was not rescued.

### Correlated feature-history prior

A linear-Gaussian observer with a joint Gaussian prior over feature and history
coordinates was added. The known observer uses the conditional prior
`p(z | tau_true)`, while the joint observer marginalizes latent history.

Output:

`outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_correlated_prior_scale1_cached_v1`

R2 gate table:

`outputs/figure4_endpoint_history_feature_readout_rr100_n128_multi_history_fdim4_hpc8_correlated_prior_scale1_cached_v1/gates_correlated_known_joint_zero_static/unified_feature_observer_gate_table.csv`

All-row pooled scores:

| Observer | Mode | R2_cv |
|---|---:|---:|
| Known correlated | `known_history_correlated_generative` | -2.4431 |
| Joint correlated | `joint_history_correlated_generative` | -1.4609 |
| Zero correlated | `zero_history_correlated_generative_on_motion` | -2.4123 |
| Static history | `static_history` | -2.1283 |

This model improved the joint observer slightly but made the known observer
worse. It therefore argues against the simple explanation that the known ceiling
failed only because the previous observer ignored feature-history covariance.

### Alternate primary histories

The n=128 cache was reanalyzed with OU and Brownian histories as the primary
test condition. Both retained the `known = joint` pattern for the independent
generative observer.

| Primary history | Known - zero | Joint - zero | Joint - static | Known - joint |
|---|---:|---:|---:|---:|
| empirical | 0.9879 | 0.9879 | 0.6543 | 0.0000 |
| OU | 1.6320 | 1.6320 | 0.3057 | 0.0000 |
| Brownian | 1.4964 | 1.4964 | 0.5874 | 0.0000 |

The Brownian and empirical primary-history runs pass the weaker endpoint-history
gates; OU does not have a positive lower bootstrap interval for joint - static.
None of the primary-history choices produce a strict known > joint result.

## True edge-parallel versus edge-orthogonal endpoint histories

The earlier family figure used trajectory-intrinsic principal-axis geometry,
not an image-edge along/across test. A separate endpoint-history run now
generates matched histories aligned to each image's local edge axis:

`outputs/figure4_endpoint_history_feature_readout_rr100_n128_axis_parallel_orthogonal_fdim4_hpc8_scale1_v1`

Companion figure:

`outputs/figure4_endpoint_history_feature_readout_rr100_n128_axis_parallel_orthogonal_fdim4_hpc8_scale1_v1/main_results_figures/endpoint_history_axis_edge_comparison.png`

Both edge-axis conditions pass the history-use gates:

| Primary history | Joint - static | Joint - zero |
|---|---:|---:|
| edge-parallel | 0.573, CI [0.067, 1.079] | 0.907, CI [0.656, 1.248] |
| edge-orthogonal | 0.556, CI [0.035, 1.069] | 0.958, CI [0.727, 1.262] |

The direct edge-axis comparison is directionally parallel-favoring but not
reliable:

| Direct comparison | Delta R2_cv | 95% CI |
|---|---:|---:|
| Joint, orthogonal - parallel | -0.017 | [-0.262, 0.245] |
| Known, orthogonal - parallel | -0.017 | [-0.262, 0.245] |
| Zero-history, orthogonal - parallel | -0.068 | [-0.408, 0.302] |
| Response-only, orthogonal - parallel | -0.082 | [-0.406, 0.255] |

Safe interpretation: the endpoint-history benefit is not specific to
edge-parallel motion in the current RR100 run. The along/across question remains
open, with only a small directionally edge-parallel point estimate here.

Important methodological distinction from the older Panel D result: the older
axis-conditioned feature-posterior observer used `axis_catalog_mode=per_candidate`.
Each candidate patch therefore received its own contour axis, and the
parallel/orthogonal prior response tables were rendered relative to that
candidate's edge. The endpoint-history edge-axis control above is a
true-source endpoint-feature readout with matched trajectories aligned to the
source image's local edge axis; it does not include this candidate-conditioned
axis hypothesis layer. The two results should therefore be treated as related
but not interchangeable tests of the along/across mechanism.

## Interpretation

These candidates support the endpoint-history gates:

- Motion-history observer beats static in pooled `R2_cv`.
- Known-history observer beats zero/unknown-history observer.
- Joint latent-history observer beats zero/unknown-history observer.

The result is not a strict known-ceiling result, because the known-history generative observer selects the beta=0 fallback and is numerically identical to the joint-history observer. Therefore the current evidence supports:

`known = joint > zero / response-only / static`

not:

`known > joint > zero`.

## Scientific caveats

Absolute `R2_cv` remains negative in the whitened feature coordinate system. The positive evidence is contrastive, not an absolute claim that the observer recovers endpoint features better than the train-fold mean in every scored coordinate. An unwhitened fold-PCA feature space gives positive absolute scores, but the motion/history gates fail there.

The candidate should be treated as a promising endpoint-history contrast, not yet as a final Figure 4C result, unless the paper-facing claim is explicitly phrased as known/joint history accounting beats zero-history and static controls. If the claim requires a strict known trajectory ceiling above the joint latent observer, that requirement remains open.
