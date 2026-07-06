# Compact Translation Channel Local-Derivative Analysis Plan

Date: 2026-07-01

## Purpose

Test whether the compact retinal-translation channel carries local
image-derivative information: the signed feature changes revealed by small
retinal translations. The target computational story is not just that response
deltas predict feature deltas. In the fitted twin, full response deltas should
predict many image-derived feature deltas almost by construction. The scientific
question is whether this predictive structure is concentrated in the compact
translation channel, and whether it survives the strongest low-dimensional
manifold controls.

## Primary Hypothesis

For a small fixed-magnitude retinal shift, compact-channel response derivatives
predict signed local feature derivatives on held-out images, especially
phase-sensitive and high-spatial-frequency targets.

Pre-committed primary contrast:

```text
compact translation basis, k=10
  versus
fold/image-disjoint static-response PCs, k=10

target:
signed local phase or signed steerable-pyramid/Gabor coefficient derivatives

metric:
held-out image-disjoint derivative prediction
```

Full response-space prediction is a sanity check only. It should not be the
headline.

## Core Objects

For image/window/context `I`, unit retinal direction `tau_hat`, and small shift
size `epsilon`:

```text
central response derivative:
dr(I, tau_hat; epsilon)
  = [r(I, +epsilon tau_hat) - r(I, -epsilon tau_hat)] / (2 epsilon)

central feature derivative:
df(I, tau_hat; epsilon)
  = [f(I shifted +epsilon tau_hat) - f(I shifted -epsilon tau_hat)] / (2 epsilon)
```

Basis-projected response derivative:

```text
dz_basis(I, tau_hat; epsilon) = U_basis.T dr(I, tau_hat; epsilon)
```

Primary decoder:

```text
dz_basis(I, tau_hat; epsilon) -> df(I, tau_hat; epsilon)
```

This central-difference formulation removes the trivial displacement-magnitude
confound and makes the target explicitly signed.

## Provenance Gate

Do not interpret any result unless these checks pass first.

1. Recompute shifted images, twin shifted responses, and feature targets in the
   same run or from a manifest-verified fresh cache.
2. Record retinal shift units, sign conventions, pixel/arcmin conversion, image
   crop boundaries, interpolation mode, and model checkpoint.
3. Compute `df_direct = f(I + epsilon tau_hat) - f(I - epsilon tau_hat)` and
   `df_jacobian = J_f(I) tau_hat` through independent code paths. Their
   agreement is an audit output, not a hidden assumption.
4. Confirm response derivative sign convention with a small known synthetic
   image or simple edge/stripe stimulus.
5. Verify no image/window/context used to fit a basis, feature normalizer, or
   decoder appears in the corresponding test fold.
6. Emit a run manifest with git commit, input cache paths, feature config,
   image IDs, fold assignments, `epsilon` values, directions, basis definitions,
   and random seeds.

## Feature Targets

Primary targets:

- signed steerable-pyramid coefficients by spatial-frequency band;
- signed Gabor even/odd coefficients;
- local phase advance represented as sine/cosine phase-vector derivatives;
- signed high-spatial-frequency bandpass derivatives.

Secondary targets:

- Gabor amplitude or energy derivatives;
- pyramid magnitude derivatives;
- raw image gradient and high-spatial-frequency gradient energy;
- edge-normal and edge-tangent derivative components;
- local contrast-normalized derivative targets.

Use amplitude/energy targets mainly as controls and secondary context. The
sharpest discriminator is signed phase/coefficient structure.

## Basis And Control Ladder

All bases must be fold-trained where applicable and dimensionality matched.
The primary dimensionality is `k=10`, with a secondary k sweep:

```text
k = 2, 5, 10, 20, 30
```

Basis set:

1. `full_response`: full response derivative, sanity check only.
2. `compact`: image-disjoint compact translation basis.
3. `static_pc`: fold/image-disjoint static-response PCs from `r0(I)`.
4. `compact_resid_static`: compact basis residualized against static PCs.
5. `static_resid_compact`: static PCs residualized against compact.
6. `noncompact_complement`: response derivative projected outside compact.
7. `global_rate`: global-rate axis or low-rank gain axis.
8. `target_pc1`: dominant response/tangent PC control.
9. `random`: random orthonormal basis with matched k.
10. `unit_shuffle_compact`: compact basis after unit-order shuffle.
11. `rf_readout_permuted_compact`: RF/readout-preserving compact control if
    metadata are available.

Static PCs are not a throwaway nuisance control. They are the primary manifold
competitor.

## Split Design

Primary split:

```text
image-disjoint folds
```

Secondary splits:

- image-family or spectral-bin held out;
- source-window held out;
- session held out if enough support exists;
- trial/repeat held out for any recorded extension.

The compact and static-PC bases, decoder weights, feature normalizers, and
target PCA/whitening transforms must all be fit only on training images.

## Decoder

Use ridge regression as the primary simple decoder:

```text
df_hat = W dz_basis + b
```

Hyperparameters:

- selected by inner cross-validation inside the training split;
- fixed across bases where feasible, or selected with identical grids;
- no tuning on compact-minus-static outcome.

Secondary decoders:

- PLS with matched latent dimensionality;
- linear CCA-like readout for phase-vector targets;
- no nonlinear decoder in the primary analysis.

## Metrics

Primary metrics:

- held-out `R2` for signed derivative targets;
- feature cosine between predicted and true derivative vectors;
- negative MSE after target standardization;
- paired compact-minus-static-PC effect;
- clustered bootstrap CI over held-out images/windows.

Report performance within fixed `epsilon` and direction strata. Do not pool
variable `|tau|` values unless derivatives have been normalized by shift size.

## Local Linear Consistency Tests

These tests are required because generic feature decodability is not enough.
Run them on true targets and decoded targets.

Antisymmetry:

```text
df_hat(+x) + df_hat(-x) ~= 0
df_hat(+y) + df_hat(-y) ~= 0
```

Scaling:

```text
df_hat(2 epsilon, tau_hat) ~= 2 df_hat(epsilon, tau_hat)
```

Additivity:

```text
df_hat(x + y) ~= df_hat(x) + df_hat(y)
```

Directional edge structure:

```text
edge-normal derivative signal > edge-tangent derivative signal
```

Quantify with normalized residuals, for example:

```text
antisymmetry_residual =
  ||df_hat(+x) + df_hat(-x)|| /
  mean(||df_hat(+x)||, ||df_hat(-x)||)
```

Compare signed phase/coefficient targets against amplitude/energy targets as a
negative-control family. Energy targets should not satisfy signed antisymmetry
in the same way.

## Scale Sweep

Run at two regimes:

```text
epsilon_small: finite-difference linear regime
epsilon_FEM: FEM-relevant retinal displacement scale
```

Recommended initial sweep:

```text
epsilon_arcmin = 0.125, 0.25, 0.5, 1.0
```

Report:

- derivative-decoding performance by scale;
- linearity residuals by scale;
- scale at which signed derivative consistency breaks down;
- whether the FEM-relevant scale lies inside or outside the linear regime.

This separates a true local differential result from a broader motion/feature
gain that may simply increase with displacement size.

## Bridge To Recorded Covariance Geometry

Add an explicit bridge so the derivative result connects to the existing
compact covariance object.

Required comparisons:

1. Run the primary derivative decoder using the exact cross-fit compact basis
   used for covariance closure when possible.
2. Compute principal angles between the derivative-predictive compact subspace
   and the compact/covariance-closure basis.
3. Test whether derivative-predictive directions capture recorded FEM covariance
   above random, unit-shuffle, and RF/readout controls.
4. Test whether covariance-closure compact directions retain derivative readout
   performance relative to a freshly learned derivative basis.

If this bridge fails, the derivative result may still be true but should be
described as a separate twin-internal local feature-derivative finding rather
than as an explanation of the recorded shared-variability object.

## Feature-Recovery Utility Test

After the derivative readout is established, ask whether the derivative signal
improves recovery of shifted-image features.

Target:

```text
f(I shifted by epsilon tau_hat)
```

Readout variants:

```text
static only:             r0 -> f_shifted
static + compact:        [r0, U_compact.T dr] -> f_shifted
static + static_pc:      [r0, U_static.T dr] -> f_shifted
static + compact_resid:  [r0, U_compact_resid.T dr] -> f_shifted
static + random:         [r0, U_random.T dr] -> f_shifted
static + noncompact:     [r0, U_noncompact.T dr] -> f_shifted
```

All variants must add the same number of derivative dimensions. Otherwise the
gain can reflect parameter count rather than useful compact structure.

Primary utility metric:

```text
Delta utility = performance(static + basis) - performance(static only)
```

Report the paired compact-minus-static-PC utility contrast.

## Output Tables

Suggested output root:

```text
outputs/compact_retinal_translation_geometry/local_derivative_channel_v1/
```

Required files:

```text
run_manifest.json
provenance_audit.json
image_fold_assignments.csv
feature_target_inventory.csv
basis_inventory.csv
derivative_prediction_metrics.csv
derivative_prediction_bootstrap.csv
compact_minus_static_primary.csv
linearity_consistency_metrics.csv
scale_sweep_summary.csv
subspace_bridge_principal_angles.csv
covariance_bridge_metrics.csv
feature_recovery_utility.csv
decision_table.json
README.md
```

Suggested figures:

```text
figures/signed_phase_compact_vs_static.png
figures/k_sweep_derivative_prediction.png
figures/linearity_residuals_by_scale.png
figures/subspace_bridge_principal_angles.png
figures/feature_recovery_utility.png
```

## Decision Table

| Outcome | Interpretation |
|---|---|
| Compact beats static PCs on signed phase/coefficient derivatives, compact residual remains positive, and linearity passes | Strong compact-specific local derivative result. |
| Compact and static PCs both beat random/gain/unit-shuffle, but compact does not beat static PCs | Strong manifold-tangent derivative result; useful, but not compact-specific. |
| Full response predicts derivatives, compact/static do not | Local derivative information exists but is not captured by the compact object. |
| Performance disappears after fixed-magnitude or derivative normalization | Previous effect was displacement-magnitude confounded. |
| Amplitude/energy targets succeed but signed phase/coefficient targets fail | Feature-energy readout only; not a signed local derivative result. |
| Small `epsilon` succeeds but FEM-scale shifts fail linearity | Local differential geometry exists, but FEM-scale usefulness is limited or nonlinear. |
| Derivative subspace does not overlap covariance-closure compact basis | Derivative finding is not the same object as the recorded FEM covariance geometry. |
| Compact derivative readout predicts recorded covariance above controls | Derivative structure plausibly explains part of the recorded compact covariance object. |

## Safe Interpretation Language

Strong positive:

```text
The compact translation channel carries signed local feature-derivative
structure for held-out images. This signal is strongest for phase-sensitive
high-spatial-frequency targets, obeys local linear derivative constraints, and
overlaps the compact covariance-closure geometry.
```

Manifold-tangent positive:

```text
Signed local feature-derivative information is concentrated in the compact /
static image-response manifold overlap. The compact channel is therefore useful
as a low-dimensional manifold-tangent carrier of translation-linked feature
changes, but the current result does not isolate a compact-specific component
beyond static-response PCs.
```

Negative or bounded:

```text
The fitted twin contains local feature-derivative information in full response
space, but the current compact translation channel does not concentrate that
information beyond matched low-dimensional controls.
```

## Minimal First Pass

The smallest useful run should include:

1. fresh provenance-checked shifted images and twin response derivatives;
2. signed Gabor even/odd and signed pyramid coefficient derivative targets;
3. `epsilon_arcmin = 0.125, 0.25, 0.5`;
4. directions `+x`, `-x`, `+y`, `-y`, and diagonals if available;
5. image-disjoint folds;
6. bases `compact`, `static_pc`, `compact_resid_static`, `random`,
   `unit_shuffle_compact`, `global_rate`, and `full_response`;
7. fixed-magnitude derivative prediction;
8. antisymmetry and scaling residuals;
9. compact-versus-static paired bootstrap;
10. subspace overlap with the covariance-closure compact basis.

