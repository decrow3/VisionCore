# FEM path-integrated separability analysis handoff

## Purpose

This document specifies the next analysis after the expanded FEM step-Jacobian regime result.

The step-regime result now supports the local field framework:

- Drift-sized steps (`<= 1 arcmin`) are locally predictable by midpoint Jacobians.
- Intermediate steps (`<= 1.5 arcmin`) preserve tangent direction while magnitude prediction collapses.
- Large steps (`>= 2 arcmin`) fail both direction and magnitude prediction.
- Therefore, V1 response geometry is locally differentiable over drift-scale displacements but globally curved over larger stimulus-synchronous transitions.

The next question is functional:

> Does sampling multiple phases along a FEM trajectory make stimulus identity more separable than it is at a single stabilized phase, after controlling for sample count?

This analysis should test whether FEMs provide **complementary identity information across phases**, rather than merely more repeated samples.

---

# Part 1. Quick QA/debugging on the completed step-regime result

Before starting the separability analysis, please do a short QA pass on the already completed regime matrix.

## 1. Resolve large-step cosine median/CI mismatch

The headline table reported something like:

```text
large >= 2 arcmin: median cosine = -0.03, 95% CI = [+0.01, +0.04]
```

A CI entirely on the opposite side of zero from the median is a reporting inconsistency unless the point estimate and CI refer to different statistics.

Most likely explanation:

```text
point estimate = pooled-step median
CI = trace-level bootstrap statistic
```

or:

```text
point estimate = median of pooled raw steps
CI = bootstrap over trace-level medians / mean of trace medians
```

Please inspect:

```text
outputs/stats/fem_step_jacobian_regime_matrix_full/summary/figure_panel_regime_summary.csv
outputs/stats/fem_step_jacobian_regime_matrix_full/summary/comparison_summary_by_regime_bootstrap.csv
```

Write:

```text
outputs/stats/fem_step_jacobian_regime_matrix_full/summary/QA_NOTES.md
```

with:

- the exact source of the mismatch;
- whether the issue is pooled-step median vs trace-level bootstrap;
- the corrected manuscript-facing large-step cosine statistic;
- which statistic should be used consistently for all rows.

Preferred reporting convention:

```text
trace-level statistic with trace-level bootstrap CI
```

because traces, not individual steps, are the appropriate unit of independence.

If useful, report both:

```text
pooled_step_median_cosine
trace_level_median_cosine
trace_bootstrap_ci_low
trace_bootstrap_ci_high
```

but do not mix pooled medians with trace-level CIs in the manuscript-facing table.

## 2. Confirm extreme predicted-fraction plotting

Predicted fractions for all/large-step regimes are extremely negative, around `-21` and `-42`.

Please verify that:

- the predicted-fraction figure is readable;
- the drift and intermediate regimes are not visually flattened by the large negative values;
- the figure clearly communicates that large/all regimes are far below zero.

Add a clipped figure if not already present:

```text
figures/regime_predicted_fraction_summary_clipped.png
```

Suggested y-axis:

```text
[-2, 1]
```

Make clear in the caption or table that all-step and large-step values extend below the clipped range.

## 3. Confirm low-count bin handling

For step-size bin plots, bins with low sample counts should not drive interpretation.

Use:

```text
headline_bin_inclusion = n_steps >= 5
```

Add or verify column:

```text
bin_included_in_headline
```

in the binned CSV.

In figures, either omit bins with `n_steps < 5` or plot them as faint/descriptive points.

## 4. Confirm completed matrix scope

Add to `QA_NOTES.md`:

```text
n_logmars
n_orientations
n_conditions
n_traces
n_steps per regime
step filters per regime
bootstrap unit
bootstrap_samples
```

The manuscript-facing result should be based on the expanded matrix, not the earlier capped runs.

---

# Part 2. New analysis: path-integrated identity separability

## New script

Create:

```text
scripts/fem_path_integrated_separability.py
```

Output root:

```text
outputs/stats/fem_path_integrated_separability/
```

## Scientific goal

Test whether FEM phase sampling provides complementary identity information across phases.

The key claim to test is:

> FEM trajectories sample multiple local identity/translation decompositions, and those samples can provide complementary identity information under a fixed readout model.

Do **not** frame this as proving that FEMs are optimized or tuned.

---

# Conceptual setup

For each optotype identity/orientation `a` at retinal phase `p`:

```text
r_a(p) = model response to identity a at phase p
J_a(p) = local image-translation Jacobian for identity a at phase p
T_a(p) = span(J_a(p))                       # local translation tangent
P_T_a(p) = projector onto T_a(p)
P_perp_a(p) = I - P_T_a(p)                  # translation-discounted complement
```

For identity pair `(a, b)`:

```text
delta_mu_ab(p) = r_b(p) - r_a(p)
delta_mu_ab_perp(p) = P_perp_a(p) @ delta_mu_ab(p)
delta_mu_ab_tangent(p) = P_T_a(p) @ delta_mu_ab(p)
```

The previous mimicry analysis corresponds to the tangent component. This analysis asks whether the orthogonal/translationally-discounted components from multiple phases are complementary.

---

# Primary analyses

## Q1. Local identity/translation separability

At each phase:

```text
mimicry_fraction = ||P_T delta_mu||^2 / ||delta_mu||^2
orthogonal_identity_fraction = ||P_perp delta_mu||^2 / ||delta_mu||^2
```

This reports how much identity difference is locally aligned with translation versus locally separable from translation.

## Q2. Orthogonal-complement complementarity across phases

For a path or phase set `{p_1, ..., p_T}`, compute:

```text
f_t = delta_mu_ab_perp(p_t)
```

Measure whether the set `{f_t}` is redundant or complementary.

Metrics:

```text
mean_pairwise_cosine_delta_perp
median_pairwise_cosine_delta_perp
orthogonal_union_rank_80
orthogonal_union_rank_90
orthogonal_participation_ratio
cumulative_identity_energy_explained
```

Interpretation:

- high pairwise cosine / low rank: redundant identity information across phases;
- lower cosine / higher rank: complementary identity information across phases.

## Q3. Path-integrated separability

For identity pair `(a,b)` and phase samples `{p_t}_{t=1..T}`, compute the integrated identity feature:

```text
F_path = sum_t f_t
```

where:

```text
f_t = delta_mu_ab_perp(p_t)
```

Primary readout metric with fixed independent readout noise:

```text
dprime2_orthogonal_path = ||F_path||^2 / (T * sigma0^2)
```

Also compute raw and tangent versions:

```text
dprime2_raw_path = ||sum_t delta_mu_ab(p_t)||^2 / (T * sigma0^2)
dprime2_tangent_path = ||sum_t delta_mu_ab_tangent(p_t)||^2 / (T * sigma0^2)
```

Record `sigma0` in `run_config.json`.

If there is an existing fixed-noise E-optotype decoding implementation, reuse that convention.

---

# Primary complementarity ratio

Add a primary metric that distinguishes complementary information from redundant repeated sampling.

For a single phase baseline:

```text
dprime2_single = ||f_0||^2 / sigma0^2
```

For `T` repeated samples of the same phase, expected sample-count scaling is:

```text
T * dprime2_single
```

Define:

```text
complementarity_ratio = dprime2_orthogonal_path / (T * dprime2_orthogonal_single_phase)
```

Interpretation:

- `~1`: path samples add identity information as well as repeated independent samples at the single-phase baseline, or better depending on chosen baseline;
- `<<1`: phase samples cancel or provide redundant/opposing identity vectors;
- `>1`: path samples are more identity-informative than the chosen single-phase baseline after sample-count normalization.

Also compute a version relative to stabilized repeated:

```text
gain_vs_stabilized_repeated = dprime2_orthogonal_path / dprime2_orthogonal_stabilized_repeated
```

This is the most important headline comparison.

The central question is:

```text
real_fem_path > stabilized_repeated with same T?
```

If yes, FEM phase diversity provides identity information beyond repeated sampling at one phase.

---

# Sampling conditions and priority order

Use the same number of samples `T` for all conditions.

The priority order is:

## Primary comparison

1. `stabilized_repeated`
2. `real_fem_path`

This is the core test:

```text
Does real FEM phase sampling beat repeated sampling at one stabilized phase, controlling for sample count?
```

## Secondary comparison

3. `phase_shuffled_path`

This tests whether temporal order matters.

If:

```text
real_fem_path > stabilized_repeated
real_fem_path ~= phase_shuffled_path
```

then phase diversity helps, but there is no evidence that trajectory order itself is special.

If:

```text
real_fem_path > phase_shuffled_path
```

then the real path/order adds information beyond phase coverage.

## Optional controls

4. `single_phase`
5. `random_matched_steps`
6. `random_matched_phase_cloud`

These are valuable but secondary. Do not let them delay the primary comparisons.

---

# Step-regime stratification is co-primary

The step-regime result is the foundation for this analysis. Do not treat step-regime stratification as a minor control.

Run path-integrated separability separately for:

```text
drift_only_paths
all_step_paths
large_step_containing_paths
```

At minimum, for every path record:

```text
fraction_steps_lte_1arcmin
fraction_steps_1_to_1p5arcmin
fraction_steps_gte_2arcmin
mean_step_arcmin
step_rms_arcmin
step_p90_arcmin
```

Framework prediction:

```text
drift-only paths should show the strongest complementarity / separability gain
large-step-containing paths should show weaker or absent gain
```

This is the analysis that links the step-Jacobian result to the functional separability result.

---

# Identity-relevant subspace option

Be explicit about whether the orthogonal complement is computed in the full population space or after restricting to identity-relevant directions.

## First pass: full population orthogonal complement

This is simpler:

```text
delta_mu_perp = P_perp @ delta_mu
```

## Preferred final/headline variant: identity-relevant restriction

Because `P_perp` is `(N-2)` dimensional, most of population space remains after removing the translation tangent. To avoid measuring arbitrary high-dimensional residual energy, also compute an identity-relevant version.

Define an identity subspace for each LogMAR using all pairwise identity differences across orientations and phases:

```text
U_id = orthonormal basis of span({delta_mu_ab(p)})
```

Then compute:

```text
delta_mu_id = U_id U_id^T delta_mu
delta_mu_id_perp = U_id U_id^T P_perp delta_mu
```

or equivalently project after removing the tangent and then restrict to identity space.

Report both:

```text
orthogonal_identity_fraction_full
orthogonal_identity_fraction_idspace
dprime2_orthogonal_path_full
dprime2_orthogonal_path_idspace
complementarity_ratio_full
complementarity_ratio_idspace
```

If time is limited, implement full-space first but design the code so the identity-subspace restriction can be added cleanly.

---

# Initial scope

Start with E-optotype only.

Recommended full matrix:

```text
LogMARs: -0.35, -0.30, -0.20, 0.00, +0.20, +0.40, +0.60
Orientations: 0, 90, 180, 270
Identity pairs: all pairwise orientation contrasts
Traces: 10
T: fixed sample count, e.g. 60, or min valid across traces
```

Fast first pass:

```text
LogMARs: -0.20, +0.20
Orientations: 0, 90, 180, 270
Identity pairs: all pairwise orientation contrasts
Traces: 5
T: 60
Sampling conditions: stabilized_repeated, real_fem_path, phase_shuffled_path
Step regimes: drift_only, all_steps
```

---

# Practical implementation details

## Phase samples

Use the same renderer-verified convention and phase/position representation as the step-Jacobian analysis.

For each trace:

```text
real_fem_path: use positions along the trace
stabilized_repeated: use same center/stabilized position repeated T times
phase_shuffled_path: same set of real positions, shuffled in time
```

If `T` valid samples are not available for a trace/condition, either skip the trace or subsample to the minimum common `T`. Record the rule in `run_config.json`.

## Drift-only path construction

For drift-only path segments, use only adjacent steps satisfying:

```text
step_norm_arcmin <= 1.0
```

Need to be careful: if selecting isolated drift steps, the resulting phase samples may no longer form a continuous path. Two possible modes:

```text
drift_step_pairs: treat each valid drift step independently
drift_contiguous_segments: require contiguous runs of drift steps
```

Start with `drift_step_pairs` for simplicity. If positive, consider `drift_contiguous_segments`.

Record path mode:

```text
path_mode = drift_step_pairs | drift_contiguous_segments | all_steps
```

## Noise model

Use fixed isotropic readout noise:

```text
sigma0
```

Record in `run_config.json`.

Do not treat deterministic model samples as independent biological trials. This is a fixed-noise readout-energy analysis.

## Bootstrap

Use trace-level or trace × identity-pair bootstrap.

Do not treat time samples as independent.

---

# Required outputs

Output root:

```text
outputs/stats/fem_path_integrated_separability/
```

Files:

```text
run_config.json
local_separability_by_phase.csv
orthogonal_complement_diversity.csv
path_separability_by_pair.csv
path_separability_summary.csv
README.md
figures/
  dprime2_real_vs_stabilized.png
  dprime2_real_vs_shuffled.png
  complementarity_ratio_by_regime.png
  orthogonal_union_rank_by_logmar.png
  path_gain_by_step_regime.png
```

---

# CSV schema: local_separability_by_phase.csv

One row per LogMAR × source orientation × target orientation × phase.

Columns:

```text
logmar
source_orientation
target_orientation
phase_x_px
phase_y_px
phase_x_arcmin
phase_y_arcmin
delta_mu_norm
tangent_component_norm
orthogonal_component_norm
mimicry_fraction
orthogonal_identity_fraction
orthogonal_identity_fraction_idspace
jacobian_condition_number
jacobian_singular_value_1
jacobian_singular_value_2
```

---

# CSV schema: orthogonal_complement_diversity.csv

One row per LogMAR × identity pair × trajectory/phase set.

Columns:

```text
logmar
source_orientation
target_orientation
trace_id
sampling_condition
path_mode
T
n_valid_samples
mean_step_arcmin
step_rms_arcmin
step_p90_arcmin
fraction_steps_lte_1arcmin
fraction_steps_1_to_1p5arcmin
fraction_steps_gte_2arcmin
orthogonal_union_rank_80
orthogonal_union_rank_90
orthogonal_participation_ratio
mean_pairwise_cosine_delta_perp
median_pairwise_cosine_delta_perp
```

---

# CSV schema: path_separability_by_pair.csv

One row per LogMAR × identity pair × trace × sampling condition × path mode.

Columns:

```text
logmar
source_orientation
target_orientation
trace_id
sampling_condition
path_mode
T
n_valid_samples
sigma0
phase_source

fraction_steps_lte_1arcmin
fraction_steps_1_to_1p5arcmin
fraction_steps_gte_2arcmin
mean_step_arcmin
step_rms_arcmin
step_p90_arcmin

dprime2_raw_path
dprime2_tangent_path
dprime2_orthogonal_path
dprime2_orthogonal_path_idspace

dprime2_raw_single_phase
dprime2_orthogonal_single_phase
dprime2_orthogonal_single_phase_idspace

dprime2_raw_stabilized_repeated
dprime2_orthogonal_stabilized_repeated
dprime2_orthogonal_stabilized_repeated_idspace

gain_vs_stabilized_repeated
gain_vs_stabilized_repeated_idspace

complementarity_ratio
complementarity_ratio_idspace

mean_local_mimicry_fraction
mean_local_orthogonal_identity_fraction
mean_local_orthogonal_identity_fraction_idspace

orthogonal_union_rank_90
orthogonal_participation_ratio
mean_pairwise_cosine_delta_perp
```

---

# CSV schema: path_separability_summary.csv

Aggregate over traces and identity pairs.

Columns:

```text
logmar
sampling_condition
path_mode
n_pairs
n_traces
n_rows

median_dprime2_raw_path
ci_low_dprime2_raw_path
ci_high_dprime2_raw_path

median_dprime2_orthogonal_path
ci_low_dprime2_orthogonal_path
ci_high_dprime2_orthogonal_path

median_dprime2_orthogonal_path_idspace
ci_low_dprime2_orthogonal_path_idspace
ci_high_dprime2_orthogonal_path_idspace

median_gain_vs_stabilized_repeated
ci_low_gain_vs_stabilized_repeated
ci_high_gain_vs_stabilized_repeated

median_complementarity_ratio
ci_low_complementarity_ratio
ci_high_complementarity_ratio

median_orthogonal_union_rank_90
median_orthogonal_participation_ratio
median_mean_pairwise_cosine_delta_perp
```

---

# Interpretation rules

## Strong support for path-integrated separability

If:

```text
real_fem_path > stabilized_repeated
```

for orthogonal `dprime2` or `gain_vs_stabilized_repeated > 1`, with same sample count, and orthogonal complement diversity is high, then:

> FEM phase sampling provides complementary identity information beyond repeated sampling at one phase.

## Stronger active-trajectory result

If:

```text
real_fem_path > phase_shuffled_path
```

then:

> The specific real FEM temporal order adds information beyond phase coverage.

If:

```text
real_fem_path ~= phase_shuffled_path
```

then:

> Phase diversity helps, but the specific order is not special under this readout.

## Step-regime functional result

If drift-only paths show larger complementarity gain than large-step/all-step paths:

> The same regime where local Jacobians predict response increments is also the regime where path integration produces useful identity separability.

This is the strongest link between the step-Jacobian result and the separability result.

## Null result

If:

```text
real_fem_path ~= stabilized_repeated
```

then:

> The Jacobian field is locally predictive, but this readout does not show evidence that FEM phase traversal improves identity separability.

This is still useful and should be reported honestly.

---

# README requirements

Write a README with:

```markdown
# FEM path-integrated separability

## Scope
- Model:
- LogMARs:
- Orientations:
- Identity pairs:
- Traces:
- T:
- Sampling conditions:
- Path modes:
- sigma0:
- Orthogonal mode: full / identity-subspace / both

## Primary results
- real FEM vs stabilized repeated:
- real FEM vs phase shuffled:
- drift-only vs all-step:
- complementarity ratio:
- orthogonal union rank / participation ratio:

## Interpretation
- Does phase diversity improve identity separability?
- Does temporal order matter?
- Does the gain depend on drift-sized local predictability?

## Caveats
- deterministic model + fixed-noise readout:
- no claim of optimality:
- high-dimensional orthogonal complement:
- trace-level bootstrap:
```

---

# Guardrails

Do not claim:

```text
FEMs are optimized
FEMs are tuned
V1 fully separates identity and transformation
```

unless the corresponding controls explicitly support those stronger claims.

Preferred language:

```text
FEM trajectories can provide complementary identity information across local identity/translation decompositions.
```

or:

```text
The same drift-scale regime in which local Jacobians predict response increments also supports path-integrated identity separability under a fixed readout model.
```
