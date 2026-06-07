# Matched Twin Covariance Closure

Cache-based first pass for testing whether recorded FEM covariance in Ryan's
matched recorded/twin unit space is captured by fitted-twin eye-position
structure.

## Inputs

The default runner now reads Declan-side copies of Ryan's caches:

- `/home/declan/VisionCore/outputs/cache/fig3_digitaltwin.pkl`
- `/home/declan/VisionCore/outputs/cache/fig2_decomposition_ryan.pkl`

They were copied byte-for-byte from:

- `/home/ryanress/v1-fovea/VisionCore/outputs/cache/fig3_digitaltwin.pkl`
- `/home/ryanress/v1-fovea/VisionCore/outputs/cache/fig2_decomposition.pkl`

The exact Fig3 model provenance copied locally is:

- checkpoint: `/home/declan/VisionCore/outputs/cache/fig3_digitaltwin_best.ckpt`
- model config: `/home/declan/VisionCore/outputs/cache/fig3_digitaltwin_model_config.yaml`
- multi-session data config: `/home/declan/VisionCore/outputs/cache/fig3_digitaltwin_multi_basic_120_long.yaml`
- Allen session config: `/home/declan/VisionCore/outputs/cache/Allen_2022-02-16.session.yaml`

Fig3 provides matched included units with `robs_used`, `rhat_used`,
`dfs_used`, `eyepos_used`, and `neuron_mask`. Fig2 provides the recorded
covariance decomposition, including `mats[window_idx]["FEM"]`, in a larger
unit space with its own `neuron_mask`. The runner intersects unit masks before
computing any comparison.

## Basis Sources

- `eye_regression_matrix`: regress fitted-twin time-residual responses
  (`rhat_used` after valid-sample time mean removal) on measured eye position.
  The resulting unit-by-2 matrix is the cache-only proxy for the retinal
  translation tangent plane.
- `eye_regression_cov`: covariance implied by the same regression matrix and
  empirical eye-position covariance.
- `model_residual_cov`: covariance of fitted-twin time-residual responses.
  This is broader than the two-dimensional eye-position proxy and is useful as
  a matched-unit closure check.

This is not yet the finite-difference image-translation tangent analysis. It is
the strongest analysis possible from the Ryan matched caches alone.

## Targets And Controls

The target is recorded Fig2 `FEM` covariance after unit intersection and
finite-unit filtering. Empirical `FEM` estimates are sometimes mildly non-PSD,
so the runner reports both:

- `raw`: the symmetrized recorded target.
- `psd`: eigenvalue-clipped target for variance-capture summaries.

Projection controls are applied to both target and source bases:

- `none`
- `global_rate`
- `target_pc1`
- `global_rate+target_pc1`

Each capture is compared to two nulls:

- random subspace with the same dimensionality.
- unit-shuffled source basis, preserving source structure but breaking unit
  identity.

## Run

```bash
.venv/bin/python -m declan.matched_twin_covariance_closure.run_cache_closure \
  --n-nulls 200 \
  --output-root outputs/matched_twin_covariance_closure
```

Main outputs:

- `session_inventory.csv`
- `closure_session_summary.csv`
- `closure_capture_metrics.csv`
- `closure_metric_summary.csv`
- `run_manifest.json`

## Finite-Difference Tangent Run

`run_finite_difference_closure.py` is the stricter model-based version. It
loads the copied Fig3 checkpoint, reconstructs each matched fixRSVP sample with
Ryan's trial/bin assembly, shifts the lagged stimulus by central finite
differences, and compares the resulting fitted-twin retinal translation tangent
covariances to recorded Fig2 `FEM`.

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
.venv/bin/python -m declan.matched_twin_covariance_closure.run_finite_difference_closure \
  --sessions "" \
  --max-samples 512 \
  --n-nulls 100 \
  --batch-size 128 \
  --output-root outputs/matched_twin_covariance_closure_finite_difference
```

Because the Codex sandbox does not expose `/dev/nvidia*`, this command needs to
run outside the sandbox to use CUDA. The manifest records `device: cuda:0` when
that works.

Finite-difference source bases:

- `fd_mean_tangent_matrix`: rank-2 mean translation tangent plane.
- `fd_mean_tangent_cov`: covariance implied by the mean tangent plane and the
  empirical eye-position covariance.
- `fd_sample_eye_trace_cov`: covariance of samplewise linear responses
  `J_i e_i`, using each sample's finite-difference Jacobian and measured eye
  offset.
- `fd_tangent_gram_cov`: average `J_i Sigma_eye J_i.T` over sample-specific
  finite-difference Jacobians.

Current 24-session finite-difference sweep:

- 24/24 sessions ran successfully on CUDA.
- Up to 512 valid matched samples were used per session.
- Finite-difference step: 0.5 px.
- Source responses were multiplied by the fitted affine gain used to put model
  rates in recorded spike-count units.
- `window_idx=1`.

PSD `fd_sample_eye_trace_cov`, `k=2`:

- no projection: mean capture = 0.531; mean effect over unit-shuffle = 0.368;
  positive in 24/24 sessions.
- `global_rate`: mean capture = 0.382; mean effect = 0.346; positive in 24/24.
- `target_pc1`: mean capture = 0.220; mean effect = 0.180; positive in 24/24.
- `global_rate+target_pc1`: mean capture = 0.220; mean effect = 0.177;
  positive in 24/24.

PSD `fd_tangent_gram_cov`, `k=2` is essentially the same:

- no projection: mean capture = 0.528; mean effect = 0.364.
- `global_rate`: mean capture = 0.384; mean effect = 0.349.
- `target_pc1`: mean capture = 0.220; mean effect = 0.180.
- `global_rate+target_pc1`: mean capture = 0.221; mean effect = 0.177.

The strict rank-2 mean tangent plane is also supportive, though weaker:

- PSD `fd_mean_tangent_matrix`, `k=2`, no projection:
  mean capture = 0.368; mean effect = 0.245; positive in 23/24.
- `global_rate`: mean capture = 0.271; mean effect = 0.232; positive in 24/24.
- `target_pc1`: mean capture = 0.135; mean effect = 0.093; positive in 23/24.

Post-hoc audit and bootstrap/sign-test summaries are generated by:

```bash
.venv/bin/python -m declan.matched_twin_covariance_closure.summarize_finite_difference_results \
  --root outputs/matched_twin_covariance_closure_finite_difference \
  --n-boot 10000
```

Additional outputs:

- `finite_difference_provenance_audit.json`
- `finite_difference_bootstrap_summary.csv`
- `finite_difference_headline_raw_psd_bootstrap.csv`

Audit highlights:

- manifest status: `ok`
- device: `cuda:0`
- sessions: 24/24 `ok`
- metric rows: 4608, matching manifest
- rescale status: 24/24 `affine`
- target variants present: `raw`, `psd`
- projection controls present: `none`, `global_rate`, `target_pc1`,
  `global_rate+target_pc1`
- raw target total trace = 254.998; PSD target total trace = 307.642; total
  negative eigenvalue mass clipped from raw targets = 52.644.

Session-bootstrap 95% CIs for PSD `fd_sample_eye_trace_cov`, `k=2`:

- no projection: effect over unit-shuffle = 0.368, CI [0.324, 0.411],
  sign-test 24/24, p = 1.19e-7.
- `global_rate`: effect = 0.346, CI [0.297, 0.394],
  sign-test 24/24, p = 1.19e-7.
- `target_pc1`: effect = 0.180, CI [0.144, 0.218],
  sign-test 24/24, p = 1.19e-7.
- `global_rate+target_pc1`: effect = 0.177, CI [0.144, 0.212],
  sign-test 24/24, p = 1.19e-7.

Raw target summaries are also positive for `fd_sample_eye_trace_cov`, `k=2`:

- no projection: capture = 0.676; effect = 0.480; sign-test 24/24.
- `global_rate`: capture = 0.528; effect = 0.498; sign-test 24/24.
- `target_pc1`: capture = 0.319; effect = 0.298; sign-test 22/24.
- `global_rate+target_pc1`: capture = 0.318; effect = 0.278;
  sign-test 22/24.

Allen step-size sensitivity was stable for the samplewise basis:

- 0.25 px, PSD `fd_sample_eye_trace_cov`, `k=2`: capture 0.602, global-rate
  capture 0.402, target-PC1 capture 0.186.
- 0.5 px: capture 0.600, global-rate capture 0.403, target-PC1 capture 0.190.
- 1.0 px: capture 0.592, global-rate capture 0.392, target-PC1 capture 0.192.

Interpretation: this supports a substantial first-order retinal-translation
component of recorded `Sigma_FEM` geometry in the matched recorded/twin unit
space. It should not be worded as a complete explanation of all FEM covariance.

Null caveat: the current run includes random-subspace and unit-shuffle nulls.
The unit-shuffle null breaks recorded/twin unit identity while preserving source
loading structure. A stricter readout-geometry null, such as a readout-location
or RF-center preserving shuffle, is still a useful reviewer-facing extension if
we want to separate retinal-translation geometry from any generic co-located
readout/RF geometry.

## Current Full-Run Readout

Using 24 matched sessions, `window_idx=1`, and 200 null draws:

- PSD `eye_regression_matrix`, `k=2`, no projection:
  mean capture = 0.436; mean effect over unit-shuffle = 0.286; positive in
  24/24 sessions.
- PSD `eye_regression_matrix`, `k=2`, `global_rate` projection:
  mean capture = 0.299; mean effect over unit-shuffle = 0.260; positive in
  24/24 sessions.
- PSD `eye_regression_matrix`, `k=2`, `target_pc1` projection:
  mean capture = 0.132; mean effect over unit-shuffle = 0.088; positive in
  24/24 sessions.
- PSD `eye_regression_matrix`, `k=2`, `global_rate+target_pc1` projection:
  mean capture = 0.137; mean effect over unit-shuffle = 0.093; positive in
  23/24 sessions.

The cache-only evidence is therefore supportive, not null. The next stricter
step is to replace the eye-position regression proxy with true finite-difference
fitted-twin retinal translation tangents in the same matched unit space.
