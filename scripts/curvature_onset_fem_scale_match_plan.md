# Curvature-onset versus FEM-scale analysis plan

## Purpose

This analysis tests a sharper version of the finite-local trajectory hypothesis:

> The displacement scale at which the V1 population response surface stops being well approximated by the local image-translation Jacobian is comparable to the natural displacement scale of real fixational eye movements (FEMs).

This is intended as a compact, falsifiable analysis that can strengthen the current manuscript without committing to a full alpha-sweep or broad optimality claim.

The core question is:

```text
Does real FEM amplitude lie near the boundary between local linear translation geometry
and finite-displacement curvature in the V1 population response surface?
```

If yes, the result supports the idea that FEMs sample a useful finite-local regime: large enough to generate substantial response trajectories, but small enough to remain organized by local translation geometry.

For the E-optotype analysis, the strongest version of the claim is not a global one-number match across all LogMAR values. The sharper question is whether the fixed FEM sampling scale intersects the stimulus-dependent curvature-onset scale near a behaviorally relevant stimulus size.

If no, the Jacobian/trajectory story can still stand as a geometric observation, but the stronger scale-matching interpretation should be dropped or moved to future work.

---

## Scientific claim this analysis can support

Strong supported claim, if the result works:

> The natural scale of fixational eye movements is matched, within a factor of order one, to the displacement scale at which local V1 translation geometry transitions from tangent-like to curved finite-displacement trajectories.

Do **not** claim:

```text
FEMs are optimized for V1.
FEMs create the Jacobian landscape.
The oculomotor system is tuned by this analysis alone.
```

Safer language:

```text
matched to
consistent with
falls near
operates close to
finite-local regime
```

---

## High-level logic

For each stimulus condition or image/window:

1. Compute model responses to small retinal translations.
2. Use local or midpoint Jacobians to predict response differences.
3. Quantify the error of the local linear prediction as a function of physical displacement.
4. Define a curvature-onset scale, `delta_star`, where the local linear approximation breaks down.
5. Compute the real FEM displacement scale, `fem_rms`, in the same physical units.
6. Compare `fem_rms` to `delta_star`.
7. For E-optotypes, trace `delta_star` as a function of stimulus size and identify the stimulus scale where the fixed FEM band intersects the curvature-onset curve.

For E-optotypes, the headline result is the crossing point between the stimulus-dependent `delta_star` curve and the approximately fixed `fem_rms` band:

```text
crossing stimulus size = size where delta_star(size) ~= fem_rms
crossing LogMAR = LogMAR where delta_star(LogMAR) ~= fem_rms
```

This asks whether real FEM scale falls near the linear-to-curved transition at a behaviorally relevant stimulus scale.

The scale ratio remains a secondary summary statistic:

```text
scale_ratio = fem_rms / delta_star
```

A ratio near 1 means real FEM scale is close to the curvature-onset scale at that condition.

---

# Definitions

## Response difference

For two retinal positions `p0` and `p1`:

```text
delta_p = p1 - p0
delta_r_true = r(I at p1) - r(I at p0)
```

## Local Jacobian prediction

Using a Jacobian evaluated at `p0`:

```text
delta_r_pred = J(p0) @ delta_p
```

or, preferably if already implemented:

```text
delta_r_pred = J(midpoint(p0, p1)) @ delta_p
```

Midpoint Jacobians are preferred because they better isolate curvature from trivial first-order expansion error.

## Normalized prediction error

Primary error metric:

```text
err_norm = ||delta_r_true - delta_r_pred|| / (||delta_r_true|| + eps)
```

Recommended epsilon:

```python
eps = 1e-12
```

Also compute optional explained variance / cosine metrics:

```text
cosine_true_pred = cos(delta_r_true, delta_r_pred)
frac_residual_energy = ||delta_r_true - delta_r_pred||^2 / (||delta_r_true||^2 + eps)
predicted_fraction = 1 - frac_residual_energy
```

Use `err_norm` as the primary metric.

## Curvature-onset scale

For each stimulus/window, bin pairwise displacements by magnitude:

```text
d = ||delta_p||
```

Compute median `err_norm` in displacement bins.

Define `delta_star_tau` by interpolating the threshold crossing of the binned error curve rather than snapping to the next tested bin.

Recommended implementation:

```text
1. Compute median err_norm at each bin center.
2. Find the first adjacent bin pair whose medians straddle tau.
3. Interpolate the crossing displacement between those two bin centers.
```

Primary interpolation:

```text
linear interpolation in displacement versus median err_norm
```

Optional sensitivity check:

```text
log-displacement interpolation if the curve is sampled on a multiplicative grid
```

Report the interpolated crossing value as `delta_star_tau`.

Primary threshold:

```text
tau = 0.5
```

Sensitivity thresholds:

```text
tau = 0.25
tau = 0.75
```

If the curve never crosses the threshold within tested displacements, set:

```text
delta_star_tau = NaN
delta_star_status = "not_crossed"
```

If the curve starts above threshold in the smallest bin, set:

```text
delta_star_status = "below_resolution"
```

and report the smallest tested bin center as an upper bound.

Also record whether `delta_star_tau` is interpolated, below resolution, or not crossed.

---

# FEM displacement scale

Compute real FEM scale for the matching condition, window, or trace.

Recommended primary measure:

```text
fem_rms = sqrt(trace(cov(eye_position_px)))
```

This is the radial RMS scale of the eye-position cloud and is the correct primary quantity because the claim is about spatial sampling extent, not instantaneous velocity.

Also compute:

```text
fem_median_radius = median(||eye_position_px - center||)
fem_p75_radius
fem_p90_radius
fem_step_rms = sqrt(mean(||eye_t+1 - eye_t||^2))
```

Use `fem_rms` as the primary scale for comparison to `delta_star`.

Treat `fem_step_rms` as a secondary diagnostic only. It indexes frame-to-frame motion scale, not the spatial extent of the sampled retinal position cloud.

Units:

- Store all results in model pixels.
- Also convert to degrees and arcmin where possible.

```text
delta_star_deg = delta_star_px / pixels_per_degree
delta_star_arcmin = 60 * delta_star_deg
fem_rms_deg = fem_rms_px / pixels_per_degree
fem_rms_arcmin = 60 * fem_rms_deg
```

---

# Data targets

## Preferred first target: E-optotype model analysis

Use the E-optotype stimuli first because:

- stimulus identity is controlled;
- LogMAR provides a natural spatial-scale axis;
- existing E-optotype results already include real-FEM versus stabilized comparisons;
- the result can be connected to mimicry and phase landscapes.

Recommended conditions:

```text
LogMAR values: -0.20, -0.30, -0.35
Orientations: 0, 90, 180, 270 degrees
Eye traces: real FEM traces used in existing E-optotype analyses
```

Optional:

```text
Include -0.25 if it is already part of the key crossover.
Include -0.40 only as model-native saturation/control, not as an independent biological scale.
```

## Secondary target: natural-image fixRSVP windows

Use only after the E-optotype implementation is working.

Natural-image version:

- compute `delta_star` per image/window using controlled translations;
- compute `fem_rms` from the observed eye samples in that window;
- compare across windows and sessions.

This can serve as a biological/natural-image anchor, but it is noisier than the E-optotype version.

---

# Required outputs

Suggested output directory:

```text
results/fem_curvature_scale_match/
```

Required files:

```text
run_config.json
curvature_by_pair.csv
curvature_by_condition.csv
curvature_scale_match_summary.csv
fem_scale_by_trace.csv
README.md
figures/
  jacobian_error_vs_displacement_by_logmar.png
  delta_star_and_fem_vs_stimulus_size.png
  delta_star_vs_fem_rms.png
  scale_ratio_by_logmar.png
  curvature_scale_match_overview.png
```

---

# Output schemas

## `curvature_by_pair.csv`

One row per pair of translated stimulus positions.

Required columns:

```text
dataset
session
stimulus_family
stimulus_id
logmar
orientation
image_id
window_key
center_position_x_px
center_position_y_px
p0_x_px
p0_y_px
p1_x_px
p1_y_px
delta_x_px
delta_y_px
displacement_px
displacement_deg
displacement_arcmin
jacobian_mode
response_norm
prediction_norm
residual_norm
err_norm
frac_residual_energy
predicted_fraction
cosine_true_pred
valid
failure_reason
```

Notes:

- For E-optotype, `image_id` and `window_key` can be blank.
- For fixRSVP, `logmar` and `orientation` can be blank.

## `curvature_by_condition.csv`

One row per condition × displacement bin.

Required columns:

```text
dataset
session
stimulus_family
logmar
orientation
image_id
window_key
jacobian_mode
bin_index
bin_low_px
bin_high_px
bin_center_px
bin_center_deg
bin_center_arcmin
n_pairs
median_err_norm
mean_err_norm
median_frac_residual_energy
median_predicted_fraction
median_cosine_true_pred
```

## `curvature_scale_match_summary.csv`

One row per condition/window.

Required columns:

```text
dataset
session
stimulus_family
logmar
orientation
image_id
window_key
jacobian_mode
pixels_per_degree
n_pairs
n_bins
delta_star_025_px
delta_star_050_px
delta_star_075_px
delta_star_025_interp_method
delta_star_050_interp_method
delta_star_075_interp_method
delta_star_025_arcmin
delta_star_050_arcmin
delta_star_075_arcmin
delta_star_025_status
delta_star_050_status
delta_star_075_status
stimulus_size_arcmin
crossing_logmar_est
crossing_stimulus_size_arcmin_est
fem_rms_px
fem_rms_arcmin
fem_median_radius_px
fem_median_radius_arcmin
fem_p75_radius_px
fem_p75_radius_arcmin
fem_p90_radius_px
fem_p90_radius_arcmin
fem_step_rms_px
fem_step_rms_arcmin
scale_ratio_050
scale_ratio_025
scale_ratio_075
within_factor2_050
within_factor3_050
```

## `fem_scale_by_trace.csv`

One row per trace.

Required columns:

```text
dataset
session
trace_id
condition
logmar
orientation
n_frames
pixels_per_degree
center_mode
fem_rms_px
fem_rms_deg
fem_rms_arcmin
fem_median_radius_px
fem_p75_radius_px
fem_p90_radius_px
fem_step_rms_px
fem_step_rms_deg
fem_step_rms_arcmin
valid_fraction
edge_valid_fraction
```

---

# Implementation details

## Pair generation

Use an existing controlled translation grid if available.

For each stimulus/condition, generate a set of retinal positions around the center:

```text
positions = grid of offsets in pixels
```

Use displacements spanning below and above the expected FEM scale.

Recommended grid for first pass:

```python
offsets_px = np.linspace(-8, 8, 33)  # if model resolution supports this
```

or reuse the existing Step 01 translation grid.

Construct pairs:

```text
all pairs within a max distance
or pairs grouped by radial displacement bins
```

To reduce compute, prefer radial offsets around the center for first pass:

```text
directions = 8 or 16 angles
radii_px = [0.25, 0.5, 1, 2, 4, 6, 8]
```

For each radius/direction:

```text
p0 = center
p1 = center + radius * direction
```

If midpoint Jacobian is implemented, use:

```text
p_mid = (p0 + p1) / 2
J = J(p_mid)
delta_r_pred = J @ (p1 - p0)
```

This avoids requiring all pairwise combinations for the first pass.

## Displacement bins

If using all pairwise combinations, bin displacements.

Recommended bins:

```python
bin_edges_px = np.array([0, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 16])
```

Adjust to the model's pixel resolution and actual FEM range.

The bins are used to stabilize the curve, not to quantize `delta_star`. The reported `delta_star_tau` should always come from interpolation across the threshold-crossing bins when a crossing exists.

## Jacobian modes

Implement at least:

```text
center_jacobian
midpoint_jacobian
```

If compute is limited, use `midpoint_jacobian` only if already available.

Interpretation:

- `center_jacobian` asks how far a single local tangent remains valid.
- `midpoint_jacobian` asks whether the response surface is locally smooth at the displacement scale.

The curvature-onset claim should be based on a predefined mode, preferably `center_jacobian` if the question is “how far does one local chart extend,” or `midpoint_jacobian` if the question is “at what scale does local linearity fail anywhere along the path.”

Recommended primary:

```text
center_jacobian for chart-radius interpretation
midpoint_jacobian as sensitivity check
```

## Edge support

Large translations can move the stimulus out of valid support.

For every condition and displacement, report:

```text
edge_valid_fraction
padding_fraction
```

Exclude pairs where too much of the stimulus is padding/invalid.

Recommended threshold:

```text
edge_valid_fraction >= 0.95
```

For E-optotypes, ensure that edge effects do not explain the apparent curvature onset.

## Feasibility gate before full run

Before launching the full analysis, run a short feasibility check on the finest planned E-optotype condition:

```text
LogMAR = -0.35
max displacement = 8 px
all planned orientations
edge_valid_fraction and padding_fraction summary
```

Goal:

```text
verify that large displacements still retain adequate valid stimulus support
at the condition where edge confounds are most likely
```

If large-displacement samples lose substantial support, do one of the following before the full run:

```text
increase the stimulus canvas for the curvature analysis only
reduce the maximum displacement range
accept not_crossed outcomes for affected conditions rather than forcing extrapolation
```

---

# Primary hypotheses

## H1. Transition-regime crossing at behaviorally relevant scale

For E-optotypes, the primary hypothesis is that the stimulus-dependent curvature-onset curve crosses the approximately fixed FEM RMS band near a behaviorally relevant fine-scale stimulus regime.

Operationally:

```text
delta_star decreases as stimuli get finer
fem_rms is approximately fixed across E-optotype conditions
the crossing delta_star(size) ~= fem_rms occurs near a threshold-relevant stimulus size
```

Primary readout:

```text
crossing LogMAR
crossing stimulus size in arcmin
whether the crossing lies near the behaviorally relevant range
```

This is the main scientific test because a fixed FEM scale should not be expected to match `delta_star` at every LogMAR.

## H2. Finite-local scale match at the crossing point

Real FEM RMS is comparable to the curvature-onset scale:

```text
scale_ratio_050 = fem_rms_px / delta_star_050_px
```

Primary heuristic criterion:

```text
0.5 <= scale_ratio_050 <= 2.0
```

This is a factor-of-two match, not a formal p-value.

Use this as a local sanity check at or near the inferred crossing point, not as the main claim across all conditions.

## H3. Stimulus-scale dependence

Curvature-onset scale should vary with stimulus scale.

For E-optotypes:

```text
finer / high-resolution stimuli should have smaller delta_star
coarser stimuli should have larger delta_star
```

This is more informative than a global match.

This provides the mechanistic basis for H1.

---

# Pass/fail interpretation

## Strong support

```text
delta_star shifts systematically with LogMAR / stimulus scale
the delta_star curve crosses the FEM RMS band near a threshold-relevant stimulus scale
delta_star_050 is within a factor of 2 of fem_rms near that crossing
curvature-onset occurs before edge artifacts dominate
```

Interpretation:

> Real FEMs operate near the finite-local boundary of the V1 translation geometry for behaviorally relevant fine spatial structure.

## Partial support

```text
delta_star and fem_rms are in the same order of magnitude
but the crossing is ambiguous, off-axis, or outside the most relevant stimulus range
or midpoint_jacobian supports local smoothness but center_jacobian breaks down earlier
```

Interpretation:

> FEM scale is compatible with finite-local sampling, but the match is stimulus-dependent and not a global optimum.

## No support

```text
delta_star is consistently >10x fem_rms
or delta_star is consistently <0.1x fem_rms
or no systematic displacement-dependent error curve is observed
or no plausible crossing occurs within or near the relevant stimulus range
```

Interpretation:

> Real FEM amplitude is not obviously matched to the curvature scale of the model response surface. Keep the Jacobian/trajectory geometry claim, but drop scale-matching language.

---

# Figures

## Figure 1. Jacobian error versus displacement

Plot:

```text
x-axis: displacement arcmin
y-axis: median err_norm
curves: LogMAR or stimulus condition
vertical band: real FEM RMS / IQR
horizontal line: tau = 0.5
```

Purpose:

Show where the local linear approximation breaks down and where real FEM scale lies.

## Figure 2. Delta-star and FEM RMS versus stimulus size

Plot:

```text
x-axis: stimulus size arcmin
y-axis: scale arcmin
curve: delta_star_050_arcmin
horizontal band: fem_rms_arcmin
marker or annotation: inferred crossing point
```

Purpose:

Show directly whether the curvature-onset curve intersects the real FEM scale at a behaviorally meaningful stimulus size.

Use stimulus size in arcmin on the x-axis rather than LogMAR so the comparison is dimensionally direct. Convert to LogMAR in the caption or panel annotation.

## Figure 3. Delta-star versus FEM RMS

Plot:

```text
x-axis: delta_star_050_arcmin
y-axis: fem_rms_arcmin
identity line
factor-of-two bounds
points: stimulus conditions or image/windows
```

Purpose:

Show per-condition scale matching directly as a secondary summary.

## Figure 4. Scale ratio by stimulus scale

Plot:

```text
x-axis: LogMAR or stimulus size
y-axis: scale_ratio_050
horizontal line: 1
shaded region: 0.5 to 2
```

Purpose:

Summarize where FEM lies below, near, or above curvature onset across stimulus scale.

## Figure 5. Overview schematic / summary

Optional.

Show:

```text
too small: little response trajectory
finite-local: organized response trajectory near curvature onset
too large: local chart breaks down
```

---

# Statistical reporting

Use bootstrap confidence intervals over:

```text
stimulus phases
orientations
traces
image/windows
```

depending on the analysis.

For each summary:

```text
median delta_star
median fem_rms
crossing LogMAR or crossing stimulus size
median scale_ratio
bootstrap CI for median scale_ratio
fraction within factor 2
fraction within factor 3
```

Do not overemphasize p-values. The main claim is quantitative scale agreement.

For the E-optotype version, bootstrap the inferred crossing point over orientations, phases, and traces when feasible.

---

# Sanity checks

## Existing E-optotype anchors

If this analysis uses E-optotype stimuli, verify:

```text
alpha = 0 or center condition matches existing stabilized/fixed-center response outputs
real FEM scale matches previous real-FEM trace statistics
LogMAR conditions match previous mimicry/phase landscape runs
```

The curvature analysis should not silently use a different rendering path or eye convention.

## Coordinate convention

Use the renderer-verified convention for eye-to-retina shifts.

Record:

```text
eye_convention_helper
renderer_path
pixels_per_degree
grid_sample align_corners
padding_mode
```

Do not choose a reflected transform based on empirical B alignment.

## Units

Every plot must state whether displacement is in:

```text
model pixels
degrees
arcmin
```

Prefer arcmin for biological interpretation.

## Edge artifacts

Confirm curvature onset is not simply where the stimulus hits padding or crop boundaries.

The feasibility gate above is required before committing to the full E-optotype sweep.

---

# Minimal first-pass command target

Implement first on E-optotype only:

```text
LogMAR: -0.20, -0.30, -0.35
Orientations: all four
Radii: [0.25, 0.5, 1, 2, 4, 6, 8] model pixels
Directions: 8 directions
Jacobian mode: center_jacobian primary
Thresholds: 0.25, 0.5, 0.75
```

Do not implement temporal shuffles or SF stratification until this result is reviewed.

---

# Suggested README template

Each run should write a `README.md`:

```markdown
# Curvature-onset versus FEM-scale analysis

## Scope
- Dataset/stimulus family:
- Model checkpoint:
- LogMARs/images:
- Translation radii:
- Jacobian mode:
- FEM trace source:
- Eye convention:

## Primary result
- Crossing LogMAR:
- Crossing stimulus size:
- Median delta_star_050:
- Median FEM RMS:
- Median scale_ratio_050:
- Fraction within factor 2:
- Fraction within factor 3:

## Stimulus-scale dependence
- Coarse stimulus delta_star:
- Fine stimulus delta_star:
- Does the delta_star curve cross the FEM RMS band?
- Does delta_star shift with stimulus scale?

## Interpretation
- Strong / partial / no support:
- Does real FEM scale lie near the finite-local transition regime?
- Are results limited by edge artifacts or sampling range?

## Caveats
- Model-internal analysis:
- Directionality of tuning not proven:
- Biological anchor:
- Any excluded conditions:
```

---

# Final guidance

This is not an optimization analysis. It is a scale-match analysis.

Do not search for a threshold that makes the result look good. Use `tau = 0.5` as primary and report `tau = 0.25` and `tau = 0.75` as sensitivity checks.

For E-optotypes, prioritize the transition-regime crossing question over a global ratio summary. The central test is whether the fixed FEM sampling scale intersects the stimulus-dependent curvature-onset scale near a behaviorally meaningful stimulus size.

A clean null is useful. If FEM scale does not match curvature onset, that tells us not to make the finite-local scale-matching claim in the current paper.
