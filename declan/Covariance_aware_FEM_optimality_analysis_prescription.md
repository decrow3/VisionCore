# Covariance-Aware FEM Optimality Analysis Prescription

## Goal

Build a Figure 5 extension that tests whether empirical fixational eye-movement
statistics occupy a useful operating regime after population covariance is
included in the readout.

The current movie-information result asks whether retinal motion increases a
pose-aware, conditionally independent spatial-information proxy. The new
analysis should ask:

> Does the information gain from retinal motion survive when the response
> covariance induced by that same motion is treated as redundancy or nuisance
> for a pose-blind decoder, and does the covariance-aware efficiency curve peak
> or shoulder near empirical FEM scale?

The target result is not "prove global optimality." The clean target result is
a pose-aware versus pose-blind comparison over movement scale:

1. independent/pose-aware information efficiency;
2. covariance-aware/pose-aware information efficiency as a consistency control;
3. covariance-aware/pose-blind information efficiency.

Overlay the empirical FEM scale. Treat the pose-aware minus pose-blind gap as
the primary interpretable result: it estimates the value of conditioning on
retinal position. Treat any peak near scale 1.0 as an operating-regime
diagnostic, not as design optimality, unless the additional non-tautological
and gain-sensitivity checks below pass.

Critical caveat: the displacement derivative `J = dmu/d[x,y]` and the
movement-induced covariance `Sigma_FEM(D)` are not independent objects. To
first order,

```text
mu(t) ~= mu_bar + J delta_e(t)
Sigma_FEM(D) ~= J Sigma_e(D) J.T
```

so a pose-blind covariance penalty along the displacement axis is partly
structural. It is valid evidence for the pose-aware/pose-blind distinction, but
not by itself evidence that empirical FEM amplitude is optimal.

## Repo Context To Reuse

Use the production `jake.twininfo` path as the source of truth.

Relevant files from `repo_bundle.txt`:

- `jake/twininfo/pipeline.py`
  - production pipeline;
  - `PipelineConfig`;
  - trajectory conditions;
  - `_trajectory_for_condition`;
  - `_condition_blocks`;
  - `run_pipeline`;
  - writes `metadata/05_lagcube_information_summary.csv`;
  - writes `cache/cumulative_information_series.npz`.
- `jake/twininfo/lagcube_information.py`
  - `run_lag_cube_rates`;
  - `run_shifted_lag_cube_rates`;
  - `run_shifted_lag_cube_rate_maps`;
  - `finite_difference_shift_set`;
  - `finite_difference_derivatives`;
  - `fisher_by_time`;
  - `cumulative_pattern_fisher`;
  - `cumulative_spatial_ssi`.
- `jake/twininfo/information.py`
  - Poisson Fisher helpers;
  - event-code information helpers;
  - spatial SSI helpers.
- `declan/active_sensing_movie_information/summarize_figure5_additional_checks.py`
  - paired final and time-series summaries;
  - condition auditing;
  - trajectory QC correlation summaries.
- `declan/active_sensing_movie_information/generate_active_sensing_movie_information_figure.py`
  - final Figure 5 plotting style and metric conventions.
- `VisionCore/covariance.py`
  - `project_to_psd`;
  - `participation_ratio`;
  - `directional_variance_capture`;
  - `cov_to_corr`;
  - recorded-data covariance utilities.
- Existing denominator/accounting dashboard:
  - `declan/active_sensing_movie_information/summarize_reafferent_variance_accounting.py`;
  - `outputs/active_sensing_movie_information/reafferent_variance_accounting/`.

Do not build this on the exploratory
`declan/active_sensing_movie_information/run_active_sensing_movie_information.py`
unless the production `jake.twininfo` path is unavailable. The README says that
script is not the final source of truth.

## Scientific Definitions

### Movement scale

For each real trace `e(t)`, define its mean-centered trace:

```text
e_c(t) = e(t) - mean_t e(t)
```

and scaled traces:

```text
e_D(t) = mean_t e(t) + D * e_c(t)
```

Use at least:

```text
D in [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
```

where:

- `D = 0` is stabilized at the trace mean;
- `D = 1` is empirical FEM amplitude;
- `D > 1` tests whether larger motion keeps helping or rolls over.

Also include control families at matched scale:

- `random_amp_scaled_D`: measured step amplitudes scaled by `D`, randomized
  directions;
- `random_amp_cloud_matched_scaled_D`: scaled step amplitudes plus occupancy
  match, extending the existing `random_amp_cloud_matched`;
- optionally `trajectory_order_shuffle_scaled_D`, preserving visited positions
  after scaling but shuffling time order.

### Three readout regimes

For each image/crop/trace/condition/scale, compute a local displacement Fisher
metric with response mean `mu` and derivative `J = dmu/d[x,y]`.

1. Independent pose-aware:

```text
F_ind = J.T diag(1 / mu) J
```

This is the existing optimistic Poisson endpoint. It should match the spirit of
`cumulative_pattern_fisher` / `cumulative_spatial_ssi`.

2. Covariance-aware pose-aware:

```text
F_cov_aware = J.T inv(Sigma_poisson + Sigma_residual) J
```

In the first pass, set:

```text
Sigma_residual = 0
```

or use a diagonal/low-rank residual calibrated from recorded FEM-corrected
covariance only after the twin-only version works. With `Sigma_residual = 0`
and `Sigma_poisson = diag(mu)`, this metric is exactly the independent metric.
Even with recorded calibration it should remain close to independent if the
FEM-corrected correlations are near zero. Therefore this curve is a consistency
check, not a separate source of evidence.

3. Covariance-aware pose-blind:

```text
F_cov_blind = J.T inv(Sigma_poisson + Sigma_residual + Sigma_FEM(D)) J
```

where `Sigma_FEM(D)` is the covariance of the model response driven by the
movement cloud at scale `D`, computed across time samples, trace examples, and
possibly image/crop examples depending on the analysis level.

Efficiency normalization:

```text
eta_* = trace(F_*) / expected_spikes
```

or use the geometric mean / determinant scalar as a sensitivity check. The
primary scalar should be `trace(F)` because it is stable and already used in the
repo's Fisher summaries.

### Gain and noise sensitivity

The location of any pose-blind peak depends on where reafferent covariance
crosses the observation-noise floor:

```text
D^2 trace(J Sigma_e J.T) ~= trace(diag(mu))
```

This makes the peak sensitive to model gain and to the assumed intrinsic noise
floor. Before reporting "peak near empirical scale," run sensitivity sweeps:

```text
rate_gain in [0.5, 1.0, 2.0]
noise_floor_multiplier in [0.5, 1.0, 2.0]
```

Implement this as:

```text
mu_eff = rate_gain * mu
J_eff = rate_gain * J
Sigma_poisson_eff = noise_floor_multiplier * diag(mu_eff)
```

and recompute `D_peak`, empirical fraction of peak, and pose gap. If `D_peak`
slides strongly across this grid, demote peak language and report the result as
an operating-regime or pose-gap analysis.

### Linear-validity range

The covariance-aware Fisher uses a local finite-difference derivative `J`, while
large movement scales produce finite-displacement curvature. Do not interpret
large-D rolloff or recovery unless the local tangent still captures the
movement-induced covariance.

Use the existing tangent-capture logic, or compute within this analysis:

```text
linear_capture(D) = trace(U_J,k.T Sigma_FEM(D) U_J,k) / trace(Sigma_FEM(D))
```

with `k = 2, 5, 10`. Define a primary valid range, for example:

```text
linear_capture_k10(D) >= 0.7
```

or a threshold justified by the existing Figure 4 capture-vs-displacement
curve. Optimality/operating-regime language should be restricted to that valid
range.

## Implementation Strategy

### Preferred design

Add a new analysis module rather than overloading the existing Figure 5 summary:

```text
jake/twininfo/covariance_optimality.py
```

and a command-line runner:

```text
jake/twininfo/run_covariance_optimality.py
```

The runner should read or reuse a production `jake.twininfo.pipeline` run:

```text
outputs/twininfo/<run-name>/
```

It should either:

1. reuse existing selected images, crops, trace examples, and population; or
2. call the same pipeline helpers to build them when `--from-run-dir` is not
   provided.

The safest first pass is `--from-run-dir`, because it guarantees paired keys
match Figure 5.

### Do not recompute what is already cached

Use existing run metadata:

```text
metadata/run_config.json
metadata/00_population_units.csv
metadata/01_trace_examples_used.csv or metadata/01_trace_examples.csv
metadata/02_image_crop_hotspots.csv
metadata/03_trajectory_control_qc.csv
metadata/05_lagcube_information_summary.csv
cache/cumulative_information_series.npz
```

For covariance-aware Fisher, new forward passes are needed for scaled traces and
finite-difference shifts. Cache them in a separate output tree:

```text
outputs/twininfo/<run-name>/covariance_optimality/
```

Do not mutate the existing `05_lagcube_information_summary.csv` unless adding a
deliberate `--augment-existing-covariance-optimality` mode later.

## Core Algorithm

### Step 1: Add scaled trajectory generation

Extend or wrap `jake.twininfo.pipeline._trajectory_for_condition`.

New condition parser:

```text
scaled_real_D0p50
scaled_real_D1p00
scaled_real_D2p00
random_amp_scaled_D1p00
random_amp_cloud_matched_scaled_D1p00
trajectory_order_shuffle_scaled_D1p00
```

Use a parser that converts `D0p50` to `0.50`.

Pseudo-code:

```python
def scale_trace(trace, D, t_max):
    tr = np.asarray(trace[:t_max], dtype=np.float32)
    center = np.mean(tr, axis=0, keepdims=True)
    return (center + D * (tr - center)).astype(np.float32)
```

For random controls:

1. first generate the unscaled control using the existing helper on the real
   trace;
2. mean-center it;
3. multiply by `D`;
4. add the real trace center.

This keeps the random-control construction close to existing tested behavior.

Acceptance tests:

- `D=0` equals stabilized trace within tolerance;
- `D=1` equals the original condition for real traces;
- path length and RMS displacement scale approximately linearly with `D`;
- covariance scales approximately as `D^2`;
- all controls remain deterministic for a fixed seed.

### Step 2: Render lag cubes for each scale and condition

Reuse:

```python
model_lag_cubes_from_image_trace(...)
block_endpoint_lag_cubes(...)
```

from `jake/twininfo/pipeline.py` and `retinal_examples.py`.

For each paired item:

```text
example_id, kind, image_index, crop_rank, condition, scale_D
```

render:

```text
base_cubes[D, condition] = lag cubes at the condition trace
```

Cache:

```text
cache/covopt_lagcube_index.csv
cache/covopt_lagcubes/<series_id>.npz
```

where `series_id` is stable and contains the paired key plus condition/scale.

If disk use is too high, cache response rates rather than lag cubes.

### Step 3: Run model rates and finite-difference rates

For each base cube, compute:

```python
rates_center, rate_map_center = run_lag_cube_rates(..., return_rate_map=True)
```

Then compute central finite-difference rates for local image displacement:

```text
shifts = [(+h, 0), (-h, 0), (0, +h), (0, -h)]
h = config.fisher_step_arcmin / 60
```

Use the existing helper if available:

```python
run_shifted_lag_cube_rates(...)
run_shifted_lag_cube_rate_maps(...)
finite_difference_derivatives(...)
```

Primary response space:

- start with sampled population rates `(T, N)` for covariance-aware Fisher,
  because full rate maps make `Sigma` enormous;
- retain existing spatial SSI / full-rate-map metrics as the independent
  pose-aware comparison.

For covariance-aware Fisher, define:

```text
mu_tn = expected counts = rates_tn * DT
J_tnd = d(expected counts) / d[x,y]
```

Derivative units should be documented as per degree. If plotting against
arcmin, scale only the axis labels, not the stored derivative.

Cache:

```text
cache/covopt_rates_center.npz
cache/covopt_rates_fd.npz
metadata/covopt_rate_records.csv
```

Required arrays:

```text
mu: rows x T x N
J: rows x T x N x 2
expected_spikes_t: rows x T
```

### Step 4: Compute movement-induced covariance

For each grouping level, compute covariance of mean-centered model responses.

Primary grouping:

```text
group = condition family + scale_D + kind
samples = all image/crop/example/time rows
features = N sampled population channels
```

For a row-specific local covariance, also compute:

```text
group = image_index + crop_rank + example_id + condition + scale_D
samples = time
features = N
```

The primary plot should use group-level covariance because per-row time-only
covariance is noisy.

Definition:

```python
X = mu_tn.reshape(n_samples, n_units)
Xc = X - X.mean(axis=0, keepdims=True)
Sigma_fem = (Xc.T @ Xc) / max(n_samples - 1, 1)
```

Important: for `D=0`, `Sigma_fem` should be near zero for the scaled real
family, modulo model temporal history from static images. If it is not near
zero, inspect whether the response mean includes image/crop variation. The
pose-blind covariance should be computed from movement-induced variation at a
fixed image/crop when possible, then averaged; otherwise image identity
variance will contaminate `Sigma_FEM`.

Recommended two versions:

1. `Sigma_FEM_within_pair`: compute covariance over time within each
   image/crop/example, then average PSD covariances across pairs.
2. `Sigma_FEM_pooled_residual`: subtract each pair's mean response before
   pooling all time samples.

Use `Sigma_FEM_pooled_residual` as primary if stable; report both as an audit.

PSD projection:

```python
from VisionCore.covariance import project_to_psd, participation_ratio
Sigma_fem = project_to_psd(Sigma_fem, eps=0.0)
```

Cache:

```text
results/covopt_covariance_spectra.csv
cache/covopt_covariances.npz
```

Diagnostics:

- `trace`;
- top eigenvalues;
- participation ratio;
- fraction of covariance in top 2, 5, 10 modes;
- `trace(Sigma_FEM(D)) / trace(Sigma_FEM(D=1))`;
- deviation from expected `D^2` scaling.

### Step 5: Compute covariance-aware Fisher

For each row and scale, flatten over time:

```text
mu_flat: (T*N,)
J_flat:  (T*N, 2)
```

Independent Poisson:

```python
F_ind = J_flat.T @ (J_flat / np.clip(mu_flat, eps, None)[:, None])
```

Covariance-aware:

Use a block-diagonal approximation over time for the first pass:

```text
Sigma_obs_t = diag(mu_t) + Sigma_extra
```

where `Sigma_extra` is `N x N` and shared across time for a group. Then:

```python
F_cov = sum_t J_t.T @ inv(Sigma_obs_t) @ J_t
```

Do not build a full `(T*N) x (T*N)` covariance initially. It is expensive and
confounds temporal autocorrelation with population covariance. Add temporal
covariance only as a later sensitivity check.

Implement:

```python
def covariance_fisher_by_time(mu_tn, J_tnd, Sigma_extra, ridge_frac=1e-4):
    for t:
        Sigma_t = diag(max(mu_t, eps)) + Sigma_extra
        Sigma_t = 0.5 * (Sigma_t + Sigma_t.T)
        ridge = ridge_frac * median(diag(Sigma_t))
        inv = np.linalg.pinv(Sigma_t + ridge * I)
        F_t = J_t.T @ inv @ J_t
```

Outputs per row:

```text
cumulative_fisher_ind_trace
cumulative_fisher_cov_pose_aware_trace
cumulative_fisher_cov_pose_blind_trace
cumulative_fisher_ind_per_spike
cumulative_fisher_cov_pose_aware_per_spike
cumulative_fisher_cov_pose_blind_per_spike
expected_spikes
```

Primary scalars:

```text
final_fisher_ind_trace
final_fisher_ind_per_spike
final_fisher_cov_aware_trace
final_fisher_cov_aware_per_spike
final_fisher_cov_blind_trace
final_fisher_cov_blind_per_spike
```

### Step 6: Alignment diagnostics

For each group covariance and scale, compute how much FEM covariance lies along
the coding directions. Split this into two different questions:

1. displacement-axis alignment, which is expected because `Sigma_FEM` is
   generated by displacement through `J`;
2. stimulus-signal alignment, which is the non-tautological question.

Let:

```text
G = sum over rows and time of J_t @ J_t.T
```

or use the top right singular subspace of `J` in neuron space.

Metrics:

```text
coding_variance_fem = trace(U_J.T Sigma_FEM U_J) / trace(Sigma_FEM)
fem_variance_coding = trace(U_FEM.T G U_FEM) / trace(G)
muprime_fem_muprime = mean_d J_d.T Sigma_FEM J_d / J_d.T J_d
```

These displacement-axis metrics are useful diagnostics, but they should not be
used as evidence that the empirical trajectory is specially matched or optimal.
They mainly verify that the pose-blind penalty is working as expected.

Add a stimulus-identity/signal-subspace alignment analysis:

```text
C_signal = covariance of mean responses across image/crop identities
```

Compute `C_signal` using the same sampled population channels used for
covariance Fisher. Use pair-mean responses so this captures across-image or
across-stimulus variation rather than FEM variance:

```python
R_pair = mean_t mu_tn for each image/crop/example under a reference condition
C_signal = covariance(R_pair across image/crop identities)
```

Then compute:

```text
signal_variance_fem_k = trace(U_signal,k.T Sigma_FEM U_signal,k) / trace(Sigma_FEM)
fem_variance_signal_k = trace(U_FEM,k.T C_signal U_FEM,k) / trace(C_signal)
```

for `k = 2, 5, 10`.

This is the alignment fork that matters scientifically:

- high displacement alignment alone is expected;
- high signal-subspace alignment means FEM covariance corrupts or modulates the
  same population dimensions used for stimulus identity/content;
- low signal-subspace alignment means the pose-blind penalty is mostly about
  displacement uncertainty rather than stimulus-coding corruption.

For a simple first pass:

1. compute the neuron-space coding covariance:

```python
G = sum_t J_t @ J_t.T
```

2. take top `k = 2, 5, 10` eigenvectors of `G`;
3. compute `directional_variance_capture(Sigma_FEM, U_Gk)`.

Outputs:

```text
results/covopt_alignment_diagnostics.csv
```

Acceptance:

- alignment should be finite and monotonic-ish with `k`;
- `D=0` should have tiny `Sigma_FEM` trace;
- pose-blind penalty should grow with covariance trace and displacement-axis
  alignment;
- stimulus-signal alignment should be reported separately and interpreted as
  the non-tautological covariance result.

### Step 7: Summaries and figures

Add a summarizer:

```text
declan/active_sensing_movie_information/summarize_covariance_optimality.py
```

Inputs:

```text
outputs/twininfo/<run-name>/covariance_optimality/results/covopt_row_metrics.csv
outputs/twininfo/<run-name>/covariance_optimality/results/covopt_covariance_spectra.csv
outputs/twininfo/<run-name>/covariance_optimality/results/covopt_alignment_diagnostics.csv
metadata/03_trajectory_control_qc.csv
metadata/05_lagcube_information_summary.csv
```

Outputs:

```text
outputs/active_sensing_movie_information/covariance_optimality/
covopt_scale_summary.csv
covopt_paired_contrasts.csv
covopt_alignment_summary.csv
covopt_decision_table.csv
covopt_summary.md
figures/covopt_scale_curves.pdf
figures/covopt_pose_gap.pdf
figures/covopt_covariance_spectra.pdf
figures/covopt_alignment.pdf
manifest.json
```

Primary summaries:

1. Scale curves:

```text
x = scale_D
y = mean final metric across paired rows
curves = independent, cov-aware pose-aware, cov-aware pose-blind
```

2. Efficiency curves:

same as above but divided by expected spikes.

3. Pose gap:

```text
pose_gap(D) = cov_aware_pose_aware_per_spike(D)
              - cov_aware_pose_blind_per_spike(D)
```

4. Independent optimism gap:

```text
optimism_gap(D) = independent_per_spike(D)
                  - cov_aware_pose_blind_per_spike(D)
```

5. Optimality/operating-regime table:

```text
condition_family
kind
metric
D_empirical = 1.0 value
D_peak
peak_value
empirical_fraction_of_peak
empirical_on_80pct_plateau
curve_shape_label
```

Curve shape labels:

- `peak_near_empirical`: peak scale between 0.75 and 1.5;
- `empirical_on_plateau`: empirical value >= 0.8 * peak and peak is broad;
- `monotonic_increasing`;
- `monotonic_decreasing`;
- `flat_or_unresolved`;
- `random_control_dominates`.

## Statistical Design

Use paired hierarchy:

```text
image_index -> crop_rank -> trace example -> condition/scale
```

For scale curves, bootstrap over image/crop/trace paired units. Do not treat
time bins as independent samples for CIs.

Implement a lightweight hierarchical bootstrap:

1. sample image indices with replacement;
2. within each sampled image, sample crop ranks with replacement;
3. within each crop, sample trace examples with replacement;
4. recompute mean metric and paired contrasts.

At minimum, use paired row bootstrap over the unique key:

```text
example_id, kind, image_index, crop_rank
```

and label it as row-paired bootstrap.

## Recommended Production Scope

Smoke test:

```text
3 images
1 crop per image
1 fixation + 1 microsaccade example
population_size = 16
D = [0, 0.5, 1.0, 2.0]
conditions = scaled_real only
```

Pilot:

```text
all current Figure 5 images
1 crop per image
2 fixation + 2 microsaccade examples
population_size = 16 or 32
D = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]
conditions = scaled_real, random_amp_scaled, random_amp_cloud_matched_scaled
```

Production:

```text
all Figure 5 images/crops/traces from the production run
population_size = 100 if compute permits
full D grid
all primary condition families
```

Do not use the 16-unit smoke or pilot runs to make claims about covariance
geometry. They are for code, cache, and gross metric validation only. The
compact tangent geometry is already around participation ratio 9 in a much
larger population; with 16 sampled channels there is little room to see the
nontrivial covariance structure. Treat covariance conclusions as requiring
approximately `population_size >= 100`, or explicitly label smaller runs as
underpowered.

Current response-space note, 2026-06-13:

- The active-sensing production source run is a 16 biological-unit run with a
  full spatial grid in metadata. The current covariance-optimality implementation
  defaults to `--population-mode sampled_units`, selecting one center/sample
  readout per biological unit, so `covopt_full_gpu1` is a 16-channel
  covariance-optimality result.
- The compact Figure 4/TFTS geometry basis used in historical Check 8 is a
  756-response-channel canonical shared readout. It is not the geometry-aware
  comparator for the 16-channel `covopt_full_gpu1` curves.
- The intended hypothesis is
  `cov_pose_aware >= cov_geometry_aware >= cov_pose_blind`: full pose knowledge
  should recover at least as much information as a compact-geometry-aware
  observer, which should recover at least as much as a pose-blind observer.
  Test this only after all three terms are implemented in the same response
  coordinates.
- Implementation update: `run_covariance_optimality.py` now has an analysis
  population path (`--population-source analysis --analysis-population-size N`)
  and emits `cov_geometry_aware_k*` rows. The first-pass geometry-aware observer
  removes/conditions on the top-k movement-covariance eigenspace and computes
  Fisher with the residual movement covariance. Treat this as the first
  concrete middle-observer hierarchy test; the preferred production target is
  `N=256` center-grid sampled canonical readout, followed by full `N=756` if
  compute permits.

Interpretation update, 2026-06-14:

- The completed `covopt_geometry_hierarchy_n256` hierarchy demonstrates strong
  low-rank covariance rescue: at empirical `D=1`, `k=2` closes roughly
  `96-98%` of the pose-aware versus pose-blind gap, and `k=20` closes roughly
  `99.7-99.9%`.
- This is **not yet** a compact translation-tangent specificity result. In the
  current implementation, `cov_geometry_aware_k` is the residual after removing
  the top eigenspace of the same movement covariance being corrected. It is
  therefore an oracle top-movement-PC observer, and `cov_topPC_aware_k` is
  identical to `cov_geometry_aware_k` for this run.
- The current safe claim is: pose-induced covariance is highly structured and a
  low-dimensional covariance-aware observer can recover most of the pose-aware
  information. Do not claim that the independently discovered compact
  translation-tangent geometry is the functional rescue basis until
  `cov_tangent_geometry_aware_k` is implemented with a basis learned outside the
  movement covariance being corrected.
- Signal preservation is mixed: the removed low-rank movement subspace also
  contains substantial coding/signal variance, especially for larger `k`. The
  result should be phrased as covariance accounting in an overlapping
  signal/nuisance subspace, not as removing pure nuisance while leaving signal
  untouched.
- Next specificity test: compare oracle top movement-PC, independently learned
  translation-tangent basis, random subspaces, and unit-shuffled tangent bases
  in the same response space:
  `closure_tangent_k - closure_random_k`,
  `closure_tangent_k - closure_unitshuffled_k`, and
  `closure_tangent_k - closure_topPC_k`.

Pathfinding update, 2026-06-14:

- Implemented an exact D=1 partial-overlap tangent pilot in
  `declan/active_sensing_movie_information/run_tangent_geometry_pathfinding.py`.
  It reuses the completed `covopt_geometry_hierarchy_n256` `mu/J` cache,
  intersects it with the canonical 756-channel Figure 4/TFTS tangent manifest,
  and recomputes covariance-aware Fisher scores on the 116 overlapping units.
- Output:
  `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/tangent_geometry_pathfinding_d1/`.
  Scope: `k = 2, 5, 10, 20`, 5 random subspaces, 5 unit-shuffled tangent
  controls, 20,736 row scores, 32 decision rows.
- Result: the independent tangent basis is non-random but not sufficient to
  explain the hierarchy rescue. At `k=2`, tangent gap closure is roughly
  `0.18-0.25`, random is near `0`, unit-shuffled tangent is roughly
  `0.10-0.19`, and oracle top-PC is roughly `0.95-0.98`. At `k=20`, tangent
  closure is roughly `0.33-0.41`, random is roughly `0.02-0.05`,
  unit-shuffled tangent is roughly `0.27-0.32`, and oracle top-PC is roughly
  `0.998-0.999`.
- Interpretation: this supports a modest translation-tangent-specific component
  of the covariance penalty, but the strong rescue in `covopt_geometry_hierarchy_n256`
  is still primarily an oracle low-rank movement-covariance/top-PC result.
  Keep the final claim scoped to "low-rank covariance rescue" until the
  covariance and tangent bases are evaluated in a fully matched response space,
  preferably the full canonical 756-channel model when the question is
  population-level rather than empirical-neuron matched.
- Cleanup sanity check:
  `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/tangent_geometry_pathfinding_d1_cache_tangent/`.
  This recomputed a tangent basis directly from cached Jacobian columns in the
  same 116-unit subset. Because the basis is learned from the evaluation cache,
  it is not an independent mechanistic test. It also does not rescue the
  functional interpretation: mean same-cache tangent closure is roughly
  `0.04`, `0.08`, `0.09`, and `0.09` for `k = 2, 5, 10, 20`, far below oracle
  top-PC closure near `0.97-1.00`.
- Paused status labels after removal-semantics concern:
  the previous "compact functional branch = useful negative" label should not
  be treated as settled until the corrected noise-side-only audit is complete.
  The compact structural branch remains promotable as a structural mechanism
  for FEM-linked covariance, but not as a demonstrated functional readout
  mechanism.

Removal-semantics audit, 2026-06-14:

- Implemented
  `declan/active_sensing_movie_information/run_noise_side_closure_audit.py`.
  It applies the matched noise-side-only definition for every basis:
  `Sigma_k = R Sigma_FEM R.T`, with task derivatives, responses, and signal
  terms untouched.
- Added `covariance_residual_noise_side` in
  `jake/twininfo/covariance_optimality.py`. The older
  `covariance_residual_after_subspace` helper used `cov - U(U.T cov U)U.T`
  plus PSD projection, which is not equivalent for arbitrary non-eigenvector
  bases and can confound tangent-vs-oracle interpretation.
- Output:
  `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/noise_side_closure_audit_sampled/`.
  Scope: `Sigma_FEM_pooled_residual`, D scales `0` and `1`, `k = 2, 5, 10, 20`,
  116 matched units, 12 rows per group, one random and one unit-shuffled draw.
- Required sanity checks pass. `D=0` has zero covariance trace and
  `F_PA ~= F_PB ~= F_k` to about `1.6e-14` Fisher trace per expected spike.
  Synthetic `Sigma_FEM = J Sigma_e J.T` gives residual trace fraction
  `8.2e-17` and closure `1.0`.
- Corrected sampled D=1 result: oracle top-PC still closes nearly all of the
  gap (`0.967`, `0.990`, `0.996`, `0.999` mean closure for
  `k = 2, 5, 10, 20`). Manifest tangent is materially stronger under corrected
  semantics (`0.078`, `0.225`, `0.337`, `0.507`). Cache tangent closure remains
  modest (`0.204`, `0.244`, `0.244`, `0.240`) and its residual trace remains
  high (`0.658`, `0.537`, `0.468`, `0.414`).
- Current interpretation: the previous useful-negative tangent result was at
  least partly a removal-semantics artifact. The corrected sampled audit points
  to `J`/`Sigma_FEM` provenance mismatch for cache tangents, because cache
  tangent does not strongly reduce the pooled-residual covariance trace. Do not
  call this a true negative for translation-tangent bases until a full all-row
  corrected audit, preferably including `within_pair`, confirms it.

Covariance-target provenance audit, 2026-06-14:

- Implemented
  `declan/active_sensing_movie_information/run_covariance_target_provenance_audit.py`.
  This is a trace-capture provenance audit, not a new closure variant. It
  compares the exact reconstructed `J Sigma_e J.T` target from cached Jacobian
  columns, the within-pair movement covariance, and the pooled-residual
  covariance used by the corrected closure audit.
- Outputs:
  `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/covariance_target_provenance_d1/`
  and
  `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/covariance_target_provenance_d1_highk/`.
  Scope: D scale `1`, 116 matched units, all rows, no additional closure
  variants.
- Mean trace-capture result at `k = 2, 5, 10, 20`:
  exact `J Sigma_e J.T` cache tangent `0.273`, `0.423`, `0.534`, `0.632`;
  pooled-residual cache tangent `0.360`, `0.478`, `0.546`, `0.593`;
  within-pair cache tangent is effectively the same as pooled residual. Oracle
  top-PC captures much more trace for all three targets; for pooled residual it
  captures `0.767`, `0.874`, `0.931`, and `0.968`.
- High-k result: at `k=116`, the manifest tangent basis spans the full 116D
  response space and captures all target trace, but the current cache-tangent
  construction captures only about `0.744` of the exact `J Sigma_e J.T` target
  and `0.638` of the pooled/within-pair target. This means the current cache
  tangent basis is not behaving as a full spanning basis for the exact
  reconstructed cached-J covariance target.
- Working diagnosis: within-pair versus pooled-residual target definition does
  not by itself explain the mismatch. The next issue to decompose is the cache
  tangent basis definition, especially unit-centering or rank handling that may
  remove common-rate directions present in `Sigma_FEM`.
- Updated status label: compact covariance functional branch is paused pending
  covariance-provenance resolution. The corrected noise-side-only audit
  validates the scoring semantics. Oracle top-PC closure remains near complete,
  but cache tangents do not span the pooled-residual covariance target, and
  under the current basis construction they also do not fully span the exact
  reconstructed `J Sigma_e J.T` target. The previous useful-negative label
  should not be interpreted until the covariance target and cache-basis
  definition are decomposed.

Projection debug, 2026-06-14:

- Implemented
  `declan/active_sensing_movie_information/run_covariance_projection_debug.py`.
  Output:
  `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/projection_debug_d1/`.
- The projection math passes. Identity projection captures all trace for exact
  `J Sigma_e J.T`, pooled-residual, and within-pair targets. A basis built as
  `orth(J_exact)` inside the same code path that reconstructs `Sigma_J` has
  rank `116`, trace capture `1.0`, residual trace about `7e-31`, direct J
  residual about `7e-31`, and closure `1.0`.
- The existing cache tangent basis is not `orth(J_exact)`. It is the
  unit-centered-J subspace: at `k=116`, rank is `115`, `||U U.T - I||_F = 1`,
  direct centered-J residual is about `8e-31`, and direct uncentered-J residual
  is about `0.257`. This explains why its trace capture is about `0.744` for
  exact `J Sigma_e J.T` and `0.638` for pooled/within-pair covariance.
- Basis and covariance unit hashes match (`baac97be3d382ca4`) in the 116-unit
  response space. The current issue is therefore a centering/response-space
  definition mismatch, not a unit ordering or source-file mismatch.

Uncentered exact-J closure follow-up, 2026-06-14:

- Implemented
  `declan/active_sensing_movie_information/run_uncentered_j_tangent_closure_audit.py`.
  Output:
  `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/uncentered_j_tangent_closure_d1_sample6_k2_20_116/`.
  Scope: D=1, 116 matched units, six sampled rows per family/kind group,
  `k = 2, 20, 116`, one random and one unit-shuffled draw. This uses the
  corrected noise-side-only residual for every basis. A full all-row closure
  pass was too slow interactively; all-row trace/residual provenance is in the
  trace-only audit.
- Exact `J Sigma_e J.T`: uncentered exact-J nearly matches oracle top-PC. Mean
  trace capture/closure are `0.517/0.892` at `k=2` for uncentered exact-J versus
  `0.531/0.893` for oracle; at `k=20`, `0.899/0.993` versus `0.915/0.994`; at
  `k=116`, both close exactly.
- Pooled-residual covariance: uncentered exact-J is also close to oracle in the
  sampled pass. Mean trace capture/closure are `0.694/0.947` at `k=2` for
  uncentered exact-J versus `0.767/0.967` for oracle; at `k=20`,
  `0.943/0.998` versus `0.968/0.999`. Within-pair covariance is effectively
  identical to pooled residual.
- Centered exact-J remains poor: pooled/within closure is about `0.21` at
  `k=2`, `0.24` at `k=20`, and `0.23` at `k=116`, because the centered basis
  omits the common-mode direction. Manifest tangent is a separate weaker
  independent-basis result, with pooled closure about `0.077` at `k=2` and
  `0.502` at `k=20`, similar to unit-shuffled manifest at `k=20`.
- Updated status: paused, semantics fixed, basis-centering mismatch identified.
  The previous tangent-vs-oracle negative is not interpretable. Under matched
  noise-side-only semantics, uncentered same-cache exact-J tangents strongly
  close the exact linear target and most of the sampled pooled/within movement
  covariance. The local-linear tangent mechanism is back in play as a same-cache
  explanation, but independent manifest-basis specificity remains unresolved.
- Current compact covariance branch status: positive matched-cache mechanism.
  Uncentered local translation sensitivity explains the movement-covariance
  rescue in the matched cache. The independent canonical/manifest tangent basis
  remains partial and is not sufficient for the full rescue.

Final focused k=20 summary:

- Implemented
  `declan/active_sensing_movie_information/summarize_uncentered_j_tangent_closure.py`.
  Output:
  `outputs/active_sensing_movie_information/covariance_optimality/covopt_geometry_hierarchy_n256/uncentered_j_tangent_closure_d1_sample6_k20_summary/`.
  This reads the sampled closure audit and recomputes only signal fraction for
  the same sampled row/basis regime.
- At `k=20`, closure fractions are:
  exact `J Sigma_e J.T`: oracle `0.994`, uncentered exact-J `0.993`, centered
  exact-J `0.402`, manifest `0.469`, random `0.312`;
  pooled residual: oracle `0.999`, uncentered exact-J `0.998`, centered exact-J
  `0.241`, manifest `0.502`, random `0.344`;
  within-pair: oracle `0.999`, uncentered exact-J `0.998`, centered exact-J
  `0.241`, manifest `0.502`, random `0.295`.
- Residual trace at `k=20` is low for uncentered exact-J (`0.101` exact-J,
  `0.057` pooled/within) and high for manifest (`0.793` exact-J, `0.772`
  pooled/within). Signal fraction is high for uncentered exact-J (`0.904`) and
  low for manifest (`0.210`). The common-mode uncentered tangent is therefore
  essential for the same-cache closure result.

Canonical manifest convention check:

- Implemented
  `declan/active_sensing_movie_information/rebuild_manifest_tangent_basis_conventions.py`.
  Output:
  `outputs/active_sensing_movie_information/compact_basis_exports/manifest_tangent_basis_conventions/`.
- The saved `twin_tangent_maps.pkl` cache still contains raw per-object `bx/by`
  tangent vectors, so the canonical manifest basis can be rebuilt without
  rerunning the digital twin. The diagnostic exports both the raw uncentered
  convention and the historical convention.
- Important nuance: the historical manifest export centered across tangent
  samples per unit, i.e. `stack bx/by rows -> subtract per-unit mean tangent
  across rows -> SVD`. This is not the same operation as the exact-J cache
  failure mode, which removed the response-space common-mode direction within
  each tangent vector.
- At the default `0.25` arcmin delta, the uncentered and historical centered
  manifest bases are almost the same top-k subspace: overlap is `0.999` at
  `k=2`, `0.999` at `k=20`, and `0.998` at `k=50`. The historical basis is
  exactly reconstructed from the cache. Therefore rebuilding the manifest basis
  in this uncentered source convention is feasible, but is unlikely by itself
  to make the independent manifest basis close like the same-cache uncentered
  exact-J basis.

## Suggested Commands

Reuse an existing run:

```bash
conda run --no-capture-output -n yatesfv python -m jake.twininfo.run_covariance_optimality \
  --from-run-dir outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu \
  --run-name covopt_smoke \
  --scales 0,0.5,1,2 \
  --condition-families scaled_real \
  --max-pairs 6 \
  --population-mode sampled_units \
  --recompute
```

Pilot:

```bash
conda run --no-capture-output -n yatesfv python -m jake.twininfo.run_covariance_optimality \
  --from-run-dir outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu \
  --run-name covopt_pilot \
  --scales 0,0.25,0.5,0.75,1,1.5,2,3 \
  --condition-families scaled_real,random_amp_scaled,random_amp_cloud_matched_scaled \
  --ridge-frac 1e-4 \
  --recompute
```

Summarize:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python \
  declan/active_sensing_movie_information/summarize_covariance_optimality.py \
  --covopt-dir outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu/covariance_optimality/covopt_pilot \
  --figure5-run-dir outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu \
  --out-dir outputs/active_sensing_movie_information/covariance_optimality/covopt_pilot
```

Adjust environment names to the machine. Existing docs use `conda ... -n
yatesfv` for the production model path and `.venv/bin/python` for lightweight
summaries.

## Tests

Add tests under:

```text
jake/twininfo/tests/test_covariance_optimality.py
```

Required tests:

1. Scaled trace invariants:
   - D=0 equals stabilized;
   - D=1 equals real;
   - RMS scales linearly;
   - covariance trace scales quadratically for simple traces.

2. Covariance Fisher sanity:
   - with `Sigma_extra = 0`, covariance-aware Fisher equals independent Fisher
     when `Sigma_poisson = diag(mu)`;
   - adding PSD `Sigma_extra` never increases Fisher beyond numerical tolerance;
   - zero derivative gives zero Fisher;
   - scaling all rates and derivatives consistently behaves as expected.

3. Gain/noise sensitivity:
   - changing `rate_gain` and `noise_floor_multiplier` changes `D_peak` in the
     expected direction;
   - summary code records the sensitivity grid and does not silently overwrite
     the primary `D_peak`.

4. Pose-blind penalty:
   - construct synthetic `J` aligned with covariance top eigenvector;
   - `F_cov_blind < F_cov_aware`;
   - penalty is larger when covariance aligns with `J` than when it is
     orthogonal.

5. Alignment interpretation:
   - synthetic `Sigma_FEM = J Sigma_e J.T` gives high displacement-axis
     alignment;
   - the same synthetic covariance does not necessarily align with an
     independently chosen `C_signal`;
   - signal-subspace alignment code handles low-rank `C_signal` robustly.

6. Cache/records integrity:
   - row count matches arrays;
   - all paired keys have all requested scales;
   - no duplicate `(example_id, image_index, crop_rank, family, D)`.

## Interpretation Rules

### Clean pose-gap result

```text
independent and covariance-aware pose-aware curves agree
pose-blind curve diverges from pose-aware as D grows
pose gap is nonzero near D=1
```

Claim:

> FEMs are beneficial to a pose-aware reader but costly to a pose-blind reader;
> the gap quantifies the value of conditioning on retinal position.

This is the primary clean interpretation. It does not require proving that real
FEM trajectories are optimal.

### Strong operating-regime result

```text
independent per-spike rises with D or peaks above empirical D
covariance-aware pose-blind per-spike peaks or shoulders near D=1
random controls do not exceed empirical scaled_real after covariance penalty
stimulus-signal-subspace alignment is nontrivial
gain/noise sensitivity keeps D=1 near the peak or high plateau
the peak lies inside the linear-validity range
```

Claim:

> Empirical FEM amplitudes lie in a covariance-aware operating regime in the
> model; the independent metric overestimates the value of larger or random
> motion because it ignores FEM-induced population covariance.

Use "optimum" only if the sensitivity and control checks are stable. Otherwise
use "operating regime" or "high plateau."

### Conservative result

```text
random controls still match/exceed real under covariance-aware metrics
```

Claim:

> Covariance-aware readout supports a generic FEM-like motion benefit, but not
> trajectory-specific optimality.

### Null result

```text
covariance-aware and independent curves tell the same story
```

Claim:

> The current optimality question is dominated by signal-side blur/spike cost,
> not population covariance. Keep Figure 5 framed as spectral-temporal motion
> benefit rather than covariance-limited optimality.

## Common Pitfalls

- Do not let image identity covariance become `Sigma_FEM`. Subtract pair means
  or compute within-pair covariance before pooling.
- Do not treat high `Sigma_FEM` alignment with the displacement derivative `J`
  as a discovery. To first order it is built in. The non-tautological alignment
  check is against `C_signal`, the across-image/stimulus signal covariance.
- Do not interpret `D_peak` without gain/noise sensitivity. The peak can move
  when model rates or the assumed Poisson/intrinsic noise floor are rescaled.
- Do not interpret large-D behavior outside the local linear-validity range of
  the finite-difference derivative.
- Do not invert unregularized covariance matrices. Always PSD-project and use a
  ridge or pseudo-inverse.
- Do not treat time bins as independent for uncertainty.
- Do not use full spatial rate maps for covariance inversion in the first pass.
  Use sampled population channels.
- Do not claim biological mutual information. This is a deterministic twin
  plus assumed observation covariance.
- Do not call the result "optimal" unless the empirical scale is near a peak or
  high plateau under the covariance-aware metric, controls are fair,
  stimulus-signal alignment is nontrivial, gain/noise sensitivity is stable,
  and the peak is inside the linear-validity range.

## Non-Circular Companion Analyses

The covariance-aware sweep is useful, but the peak/optimality interpretation is
vulnerable because `Sigma_FEM` and the displacement derivative `J` share the
same first-order operator. Add two companion analyses whose answers come from
outside that circularity.

### Companion 1: Input-whitening optimum

This is the cleanest optimality-style test because the predicted optimum is
computed from image statistics and eye-motion kinematics only. It should not use
the fitted twin response model.

Question:

> Does the biological drift amplitude sit near the diffusion scale that whitens
> natural retinal input over an independently chosen V1-sensitive passband?

Analysis:

1. Use the same natural images/crops and measured trace pool used by
   `jake.twininfo`, but do not run the model.
2. Estimate biological drift scale from drift-only windows:

```text
MSD(tau) = E[||e(t + tau) - e(t)||^2] ~= 4 D_eye tau
```

for short lags before confinement dominates. Save `D_eye` in deg^2/s and
arcmin^2/s.

3. Generate retinal movies under:

```text
stabilized
scaled measured drift, D_scale in [0, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]
synthetic Brownian/O-U drift with matched diffusion constants
```

4. For each movie, compute retinal temporal power spectra from luminance or
contrast after applying a fixed spatial passband. Use at least:

```text
temporal_psd(f)
loglog_slope over temporal passband
spectral_entropy over temporal passband
flatness = geometric_mean_power / arithmetic_mean_power
autocorrelation_time
```

5. Define the whitening optimum as:

```text
D_whiten_slope = argmin_D abs(loglog_slope(D))
D_whiten_entropy = argmax_D spectral_entropy(D)
D_whiten_flatness = argmax_D flatness(D)
```

6. Overlay `D_eye` and report whether biological drift lies near the optimum or
on a high plateau.

Passband:

- must be fixed before looking at the optimum;
- should be independently motivated by recorded RF sizes, known V1 temporal
  sensitivity, or a conservative literature passband;
- run a sensitivity grid over plausible passbands.

Outputs:

```text
outputs/active_sensing_movie_information/input_whitening/
whitening_movie_manifest.csv
drift_diffusion_estimates.csv
retinal_temporal_psd_by_movie.csv
whitening_scale_summary.csv
whitening_passband_sensitivity.csv
figures/whitening_scale_curves.pdf
figures/retinal_temporal_psd_examples.pdf
whitening_summary.md
```

Optimistic claim:

> Biological drift amplitude lies near the scale that decorrelates natural
> retinal input across a V1-sensitive band, an input-level efficient-coding
> signature independent of the fitted twin.

Caveat:

> Whitening is necessary but not sufficient for cortical utility, and the
> numerical optimum depends on the assumed passband.

Implementation note:

This analysis can reuse `jake/twininfo/retinal_examples.py`,
`jake/twininfo/retinal_movies.py`, and the image/trace selections from an
existing `jake.twininfo` run. It should be a lightweight image/movie analysis,
not a model-forward job.

### Companion 2: Recorded-cortex pose-aware information

This is the direct V1 anchor. It asks whether accounting for measured eye
position increases recoverable stimulus information in recorded population
responses.

Question:

> In recorded V1, does a decoder/observer that conditions on eye position
> recover more stimulus information than a pose-blind observer?

Primary comparison:

```text
pose_blind:  p(y | stimulus)
pose_aware:  p(y | stimulus, eye_position_or_recent_eye_history)
```

Use cross-validated held-out likelihood, stimulus decoding accuracy, or a
bounded mutual-information estimate. The pose-aware model must be evaluated on
held-out trials/time bins so the eye covariates cannot simply overfit noise.

Recommended first implementation:

1. Use the existing Fig 2/Fig 4 aligned spike-count and eye-position windows.
2. Define a conservative stimulus label:

```text
image identity, stimulus frame/time bin, or coarse natural-image segment
```

depending on which has enough repeats.

3. Fit paired decoders:

```text
blind decoder: spikes -> stimulus label
aware decoder: spikes + measured eye state -> stimulus label
```

or a generative version:

```text
blind:  log p(y | s)
aware:  log p(y | s, e)
```

4. Evaluate:

```text
cross_validated_log_likelihood
balanced_accuracy
confusion_MI_bits
information_per_spike
```

5. Report paired deltas:

```text
aware - blind
```

by session and subject.

Important guardrail:

The cleanest generative readout is not "remove eye variance and decode the
residual." It is "decode stimulus while conditioning on observed pose." The
former can throw away useful reafferent signal; the latter asks whether the
same spikes become more interpretable when pose is known.

Outputs:

```text
outputs/active_sensing_movie_information/recorded_pose_information/
recorded_pose_info_session_metrics.csv
recorded_pose_info_paired_contrasts.csv
recorded_pose_info_decoder_qc.csv
figures/recorded_pose_info_session_pairs.pdf
recorded_pose_info_summary.md
```

Optimistic claim:

> In recorded V1, accounting for self-generated retinal motion increased the
> stimulus information recoverable from the population, so eye-linked
> variability is usable pose-conditioned signal rather than only nuisance noise.

Caveat:

> This shows information is recoverable given pose; it does not show that
> downstream areas actually recover pose.

### Companion 3: SF-localized benefit

This is already close to the current Figure 5 spatial-frequency result.

Reframe:

> The motion benefit is concentrated at high spatial frequencies, where the
> instantaneous/static code is most resolution-limited, and is weak at low
> spatial frequencies where the static code is already adequate.

Use paired FEM-minus-stabilized gains for:

```text
sf_low
sf_mid_low
sf_mid_high
sf_high
```

and keep the phase-scramble result as the caveat:

> Phase-scrambled gains similar to intact gains bound the result to
> spectral-temporal content, not natural phase structure.

### Companion 4: Information accumulation slope

Reframe the cumulative information curves by their slope over fixation time.

Question:

> Does retinal motion sustain information accumulation while stabilized input
> saturates early?

Metrics:

```text
early_slope
late_slope
late_minus_early_slope
time_to_80pct_final_information
real_minus_stabilized cumulative gain over time
```

Optimistic claim:

> Self-motion sustains information accumulation across the fixation while the
> stabilized code saturates earlier, consistent with motion refreshing the
> cortical representation rather than rereading a static image.

Caveat:

> This remains a pose-aware/upper-bound metric, justified by the near-zero
> conditional correlations in the recorded covariance analysis.

## Minimum Deliverable For The Coding Agent

The first complete deliverable should include:

```text
jake/twininfo/covariance_optimality.py
jake/twininfo/run_covariance_optimality.py
jake/twininfo/tests/test_covariance_optimality.py
declan/active_sensing_movie_information/summarize_covariance_optimality.py
outputs/active_sensing_movie_information/covariance_optimality/<run>/covopt_summary.md
outputs/active_sensing_movie_information/covariance_optimality/<run>/figures/covopt_scale_curves.pdf
```

The `covopt_summary.md` should answer these questions explicitly:

1. Does the independent metric keep favoring larger motion?
2. Does covariance-aware pose-blind information peak or roll off?
3. Is empirical `D=1` near the peak or on a high plateau?
4. How large is the pose-aware minus pose-blind gap?
5. Do random matched controls still equal or exceed real?
6. Is the covariance penalty explained by covariance trace, covariance
   dimensionality, displacement-axis alignment, or stimulus-signal-subspace
   alignment?
7. Does the apparent peak survive rate-gain and noise-floor sensitivity?
8. Is the interpreted scale range inside the finite-difference linear-validity
   range?
