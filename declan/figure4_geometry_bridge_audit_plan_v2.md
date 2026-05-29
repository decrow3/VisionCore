# Figure 4 geometry-bridge audit: implementation plan

## Purpose

This audit tests whether the current Figure 4 model-to-empirical geometry bridge is genuinely modest, or whether it is being artificially suppressed by implementation choices.

The current result is scientifically interesting but unsatisfying: empirical eye-linked geometry is robust, while model-to-empirical alignment is small. Before interpreting this as biology, we need to rule out three concrete suppressors:

1. **Eye-position / retinal-displacement frame mismatch**
2. **Inconsistent centering between empirical and model regressions**
3. **Window pooling across distinct lagged stimulus histories**

A fourth issue, the **empirical reliability ceiling**, is not a bug but must be included in the reporting.

The goal is not to make the result work at all costs. The goal is to determine whether the bridge strengthens when model and empirical geometry are computed in the same coordinate frame and over the same stimulus-state support.

---

## Scientific question

Does FEM-linked V1 geometry align with model-predicted retinal-translation geometry more strongly when:

- eye-position and retinal-image displacement conventions are matched,
- empirical and model regressions use the same centering convention,
- empirical windows are not pooled across incompatible lagged stimulus histories,
- model alignment is interpreted relative to the empirical split-half ceiling?

---

## Starting hypothesis

The current Figure 4 bridge may be artificially suppressed because empirical `B_emp` is fit over broad image-ID windows, while model objects such as `J_local`, `B_model`, and `FEM_PCs` are computed from one selected baseline stimulus state.

This can produce a reliable empirical geometry, because the pooled mixture is stable across split halves, but only weak model alignment, because the model predicts one state while the empirical target is a mixture over many states.

---

## Relevant code targets

Search the repository for these functions and scripts before reimplementing anything:

- `run_fixrsvp_step2.py`
- empirical geometry bridge / Phase 3 script
- `_collect_image_windows`
- `_choose_baseline`
- `_fit_empirical_B`
- `_empirical_fem_drive`
- `_basis_from_model_fem`
- `compute_grid_responses`
- `_shift_stimulus_batch`
- any McFarland / covariance / retinal rendering path that converts `eyepos_deg` to stimulus-grid shifts

Expected current behavior:

- `_collect_image_windows` groups by image identity only.
- `_choose_baseline` chooses the sample nearest median eye position.
- empirical `B_emp` regresses recorded responses on mean-centered eye position.
- model `B_model` shifts the baseline stimulus by `offsets_px` and regresses model response deltas on mean-centered `offsets_px`.
- current reporting compares model alignment against shuffle but not systematically against empirical split-half reliability ceiling.

---

# Audit overview

Implement a new audit script rather than modifying the original analysis in-place.

Suggested file:

```text
scripts/figure4_geometry_bridge_audit.py
```

Suggested output directory:

```text
results/figure4_geometry_bridge_audit/
```

The script should run all audit variants and save:

```text
results/figure4_geometry_bridge_audit/
  audit_summary.csv
  audit_by_window.csv
  audit_by_session.csv
  sign_axis_grid_summary.csv
  window_definition_ladder_summary.csv
  ceiling_normalized_summary.csv
  figures/
    sign_axis_grid_<session>.png
    centering_comparison_<session>.png
    window_ladder_<session>.png
    ceiling_normalized_bridge_<session>.png
  README.md
```

Do not overwrite existing Figure 4 outputs.

---

# Analysis governance

This audit is diagnostic, not a configuration search.

Lock the following distinctions before looking at outcomes:

1. The canonical frame fix is known from the resampler: use `x_negy` for the **stimulus shift only**, while keeping the regression predictor in empirical eye coordinates.
2. The full 8-mode sign/axis grid is a **confirmation sweep**, not a selection procedure.
3. The unit of inference is the **window nested within session**. Do not pool windows across sessions as if they were independent replicates.
4. For the paper-facing reruns, fix the principled frame first, then fix the principled centering choice, then run the window ladder. Do not jointly maximize over frame × centering × window level and report the best combination.

Cross-session language should therefore mean: report within-session paired results first, then summarize whether the effect has the same sign across the four sessions. With four sessions, `4/4` same-sign is the strongest simple cross-session consistency criterion available.

---

---

# Canonical coordinate convention update: confirmed y-axis bug

## Summary

A follow-up code read identified the canonical gaze-contingent resampling convention used elsewhere in the codebase. The relevant convention is:

1. Eye position is expressed in degrees/pixels with **y positive up**.
2. `grid_sample` uses image/grid coordinates with **y positive down**.
3. The canonical conversion therefore flips y when converting eye coordinates into normalized grid coordinates.
4. The gaze-contingent sample is then taken using:

```python
grid = base_grid - eye_xy_norm
```

This means that, per eye-position pixel:

```text
canonical grid offset x = -2 * eye_x_px / (W - 1)
canonical grid offset y = +2 * eye_y_px / (H - 1)
```

Equivalently, if the existing `_shift_stimulus_batch` applies:

```python
base_grid[..., 0] -= 2.0 * d[:, 0] / (W - 1)
base_grid[..., 1] -= 2.0 * d[:, 1] / (H - 1)
```

then the `d[:, 1]` passed into `_shift_stimulus_batch` must be the **negative** of the empirical eye-y coordinate if `d` is interpreted as retinal stimulus displacement.

## Current suspected bug

The current alignment script appears to pass:

```python
offsets_px = (eyepos_deg - baseline_eye_deg) * ppd
```

directly into `_shift_stimulus_batch`.

That is correct for x but wrong for y under the canonical convention.

Current script behavior:

```text
x: grid offset = -2 * eye_x_px / (W - 1)   correct
y: grid offset = -2 * eye_y_px / (H - 1)   wrong sign
```

Canonical behavior:

```text
x: grid offset = -2 * eye_x_px / (W - 1)
y: grid offset = +2 * eye_y_px / (H - 1)
```

Thus, the model-side vertical retinal displacement is reflected relative to the empirical eye-position coordinate frame.

## Minimal corrected convention

Use Option A as the default corrected implementation.

### Option A, preferred: convert only the stimulus-shift offsets

Keep `_shift_stimulus_batch` as a generic image-shift primitive. Before calling it, convert empirical eye-position offsets into the displacement argument expected by `_shift_stimulus_batch`:

```python
def eye_offsets_to_shift_offsets_px(eye_offsets_px):
    '''
    Convert empirical eye offsets, x right and y up, into the pixel displacement
    argument expected by _shift_stimulus_batch.

    _shift_stimulus_batch applies:
        grid_x -= 2 * d_x / (W - 1)
        grid_y -= 2 * d_y / (H - 1)

    The canonical gaze-contingent convention requires:
        grid_x offset = -2 * eye_x / (W - 1)
        grid_y offset = +2 * eye_y / (H - 1)

    Therefore:
        d_x =  eye_x
        d_y = -eye_y
    '''
    d = np.asarray(eye_offsets_px, dtype=float).copy()
    d[:, 1] *= -1.0
    return d
```

Use this conversion for model stimulus generation:

```python
eye_offsets_px = (win.eyepos_deg - center_eye_deg[None, :]) * pixels_per_degree
shift_offsets_px = eye_offsets_to_shift_offsets_px(eye_offsets_px)

model_responses = compute_grid_responses(
    model,
    baseline_stim,
    shift_offsets_px,
)
```

But keep the regression predictor in empirical eye coordinates:

```python
eye_c = eye_offsets_px - eye_offsets_px.mean(axis=0, keepdims=True)
delta_model_c = delta_model - delta_model.mean(axis=0, keepdims=True)
B_model = np.linalg.lstsq(eye_c, delta_model_c, rcond=None)[0].T
```

This is crucial: **flip y for the model stimulus shift, not for the B_model regressor**. The returned `B_model` should map empirical eye position, x right and y up, into response modulation, because that is the same coordinate frame used by `B_emp`.

### Option B: fix inside `_shift_stimulus_batch`

Only use this if `_shift_stimulus_batch` is meant specifically to accept empirical eye-position offsets rather than generic image-shift offsets.

Change:

```python
base_grid[..., 1] -= 2.0 * d[:, 1, None, None] / max(height - 1, 1)
```

to:

```python
base_grid[..., 1] += 2.0 * d[:, 1, None, None] / max(height - 1, 1)
```

This is more invasive because it changes the meaning of the shift primitive. Prefer Option A unless all callers are audited.

## Required change to Jacobian computation

If Option A is used, the finite-difference Jacobian must also be expressed in empirical eye-position coordinates.

For an eye-coordinate finite difference:

```python
eye_plus_x  = [[+h, 0]]
eye_minus_x = [[-h, 0]]
eye_plus_y  = [[0, +h]]
eye_minus_y = [[0, -h]]
```

convert each eye offset to shift offsets before calling the model:

```python
shift_plus_x  = eye_offsets_to_shift_offsets_px(eye_plus_x)
shift_minus_x = eye_offsets_to_shift_offsets_px(eye_minus_x)
shift_plus_y  = eye_offsets_to_shift_offsets_px(eye_plus_y)
shift_minus_y = eye_offsets_to_shift_offsets_px(eye_minus_y)
```

Then compute:

```python
J_x = (r(shift_plus_x) - r(shift_minus_x)) / (2 * h)
J_y = (r(shift_plus_y) - r(shift_minus_y)) / (2 * h)
J_eye = np.column_stack([J_x, J_y])
```

The resulting Jacobian columns are derivatives with respect to empirical eye-position x and y, not generic image-shift x and y. This makes `J_local`, `B_model`, `FEM_PCs`, and `B_emp` comparable in the same coordinate frame.

## Required change to FEM_PC model responses

For FEM covariance PCs, use the corrected shift offsets to generate model responses:

```python
eye_offsets_px = (win.eyepos_deg - center_eye_deg[None, :]) * pixels_per_degree
shift_offsets_px = eye_offsets_to_shift_offsets_px(eye_offsets_px)

R_model = compute_grid_responses(model, baseline_stim, shift_offsets_px)
```

The covariance PCs themselves are response-space objects, so they do not have eye-coordinate columns. However, the response cloud must be generated using the corrected canonical shift convention.

## Thirty-second confirmation test

Before rerunning the full pipeline, implement a single-window diagnostic.

For one representative window:

1. compute `B_emp`;
2. compute current/buggy `B_model`;
3. compute corrected y-flipped `B_model`;
4. compute current and corrected `J_local`;
5. compute coordinate recovery for controlled displacements.

Required outputs:

```text
window_id
n_samples
current_B_model_2d_alignment
corrected_B_model_2d_alignment
current_B_model_top1_alignment
corrected_B_model_top1_alignment
current_dx_R2
current_dy_R2
corrected_dx_R2
corrected_dy_R2
```

Predicted signature of the bug:

```text
current dx recovery: reasonable
current dy recovery: poor, near zero, or negative
corrected dy recovery: improves substantially
corrected top-1 alignment: improves
2D alignment: improves modestly or remains similar
```

If this signature holds, the y-axis convention bug is confirmed.

## Update to Audit 1

Audit 1 should still include the full sign/axis grid, but the canonical corrected mode should be explicitly included and treated as the primary candidate:

```text
canonical_yflip: shift_x = eye_x, shift_y = -eye_y, regressor = (eye_x, eye_y)
```

The audit grid should distinguish between:

1. **stimulus-shift transform**, used to generate model responses;
2. **regressor transform**, used to express B_model columns.

For the main corrected mode, only the stimulus-shift transform changes. The regressor remains empirical eye position.

Do not apply the y flip to both the model shift and the regression predictor, because that would hide the bug while leaving `B_model` in the wrong coordinate frame relative to `B_emp`.

## Expected impact

If the y-axis convention bug is a major suppressor:

- B_model top-1 alignment should increase and become more consistent across sessions.
- Coordinate recovery should improve especially for vertical displacement.
- J_local may improve and should be re-evaluated before concluding that local Jacobians fail empirically.
- 2D alignment may increase modestly, because 2D subspace metrics are partly invariant to reflection.
- FEM_PC response clouds may align better if the previous vertical response trajectory was reflected relative to empirical geometry.

This should be run before interpreting the residual as non-retinal biology.

## Required empirical eye-frame verification

Before treating the canonical y flip as fixed, verify that the empirical eye traces are stored in the expected eye-coordinate convention.

Required check:

1. choose one representative high-sample window per session;
2. compute `B_emp` in the recorded empirical eye coordinates;
3. compute corrected `B_model` using canonical `x_negy` stimulus shifts and the unchanged eye-coordinate regressor;
4. compare `B_emp` and `B_model` on the unambiguous x axis first;
5. if x agrees but y disagrees systematically, treat that as evidence that the stored empirical y convention may be flipped relative to the simulation pipeline.

This check is not a replacement for the canonical derivation. It is a guard against an unverified data-loader convention. If the canonical shift fix fails this verification, investigate the eye-trace metadata and calibration path before adopting any empirically better-looking alternative.

# Audit 1. Eye-position / retinal-displacement frame convention

## Problem

The empirical regression uses eye position as the predictor:

```python
eye_c = eyepos_px - eyepos_px.mean(axis=0)
R ≈ eye_c @ B_emp.T
```

The model side appears to use the same offsets both as eye-position coordinates and as stimulus-shift coordinates:

```python
model_responses = compute_grid_responses(model, baseline_stim, offsets_px)
B_model = OLS(delta_model_response ~ offsets_px_centered)
```

But physical eye movement and retinal image displacement have opposite signs under the usual convention: when the eye moves right, the retinal image shifts left. The model shift convention may also be affected by `grid_sample` sign conventions. A sign or axis mismatch can attenuate top-1 and finite-displacement alignment.

## Required implementation

Treat the canonical corrected mode as known from first principles:

```text
canonical mode = x_negy for the stimulus shift
regressor mode = empirical eye coordinates
```

Run the full 8-mode grid only as a confirmation that the principled mode is also the empirical winner. Do not use the grid as a free selection step.

Add a configurable eye-to-retina transform:

```python
def transform_eye_to_retinal_offsets(eye_offsets_px, mode):
    x = eye_offsets_px[:, 0]
    y = eye_offsets_px[:, 1]

    if mode == "xy":
        out = np.column_stack([ x,  y])
    elif mode == "negx_y":
        out = np.column_stack([-x,  y])
    elif mode == "x_negy":
        out = np.column_stack([ x, -y])
    elif mode == "negx_negy":
        out = np.column_stack([-x, -y])
    elif mode == "yx":
        out = np.column_stack([ y,  x])
    elif mode == "negy_x":
        out = np.column_stack([-y,  x])
    elif mode == "y_negx":
        out = np.column_stack([ y, -x])
    elif mode == "negy_negx":
        out = np.column_stack([-y, -x])
    else:
        raise ValueError(mode)

    return out
```

Use:

- empirical predictor = `eye_offsets_px` in the empirical eye-position coordinate convention;
- model stimulus shift = `retinal_offsets_px = transform_eye_to_retinal_offsets(eye_offsets_px, mode)`;
- model regression predictor = the empirical eye-position variable, not the retinal-shift variable.

That is:

```python
delta_model = r_model(stim shifted by retinal_offsets_px) - r_model(reference)
B_model = OLS(delta_model ~ eye_offsets_px_centered)
```

This distinction is essential. The model should be driven by retinal displacement but reported as an eye-sensitivity matrix in the same coordinate convention as empirical `B_emp`.

## Metrics

For each transform mode, compute:

- matched model-to-empirical 2D alignment
- image-shuffled 2D alignment
- paired 2D delta
- matched top-1 alignment
- image-shuffled top-1 alignment
- paired top-1 delta
- FEM_PC matched-minus-shuffled 2D delta
- B_model matched-minus-shuffled 2D delta
- J_local matched-minus-shuffled 2D delta, if applicable

## Primary diagnostic

A frame/sign bug is likely if:

- the canonical `x_negy` shift mode increases top-1 alignment within sessions relative to the current mode;
- the canonical `x_negy` shift mode increases B_model or FEM_PC paired delta across sessions;
- 2D alignment changes less than top-1, but top-1 becomes less erratic.

## Required plots

For each session:

1. bar plot of paired 2D delta by transform mode;
2. bar plot of paired top-1 delta by transform mode;
3. matched vs shuffled distributions for the canonical and current modes;
4. session-level table with canonical-mode rank and empirical winner, if different.

## Decision rule

If the canonical `x_negy` shift mode beats the current mode consistently across sessions, update the main analysis to use the canonical convention.

If a non-canonical mode appears to beat the canonical mode, treat that as a red flag, not as a license to switch conventions. Inspect source rendering code, eye-trace metadata, and per-session calibration before changing anything.

---

# Audit 2. Centering and baseline consistency

## Problem

The current code chooses a baseline stimulus near the median eye position, but empirical and model regressions are mean-centered. On a curved response surface, this means the model response surface is anchored at one point while the regression is centered at another.

## Required variants

Implement at least three centering modes.

### Mode A: current

Keep current behavior for comparison.

```python
baseline_eye = sample_nearest_median_eye
regression_center = mean_eye
```

### Mode B: mean-centered baseline

Use the sample nearest mean eye position as the model baseline and mean-center both empirical and model regressions.

```python
center_eye = nanmean(win.eyepos_deg, axis=0)
baseline_idx = sample_nearest(center_eye)
eye_offsets = eyepos_deg - center_eye
```

### Mode C: baseline-centered

Use the chosen baseline eye position as the center for both empirical and model regressions.

```python
center_eye = baseline_eye_deg
eye_offsets = eyepos_deg - center_eye
```

For each mode, make sure:

- the same `center_eye` is used to compute empirical predictor coordinates;
- model retinal shifts are computed relative to the same center;
- model response deltas are computed relative to the response at the same baseline/center stimulus;
- B_model is regressed on the empirical eye-position coordinate after the same centering.

## Mechanistic prediction

State the expected pattern before running the comparison:

- Mode C should help `J_local` most, because the Jacobian is a tangent object defined at the baseline point.
- Modes B and C should be closer for `B_model` and `FEM_PCs`, which average over the finite eye cloud and are less sensitive to the exact anchor.
- If `J_local` improves strongly under Mode C while `B_model` changes little, that specifically implicates centering inconsistency rather than a general frame bug.

## Metrics

For each centering mode:

- empirical split-half ceiling
- B_model matched and shuffled alignment
- B_model paired delta
- FEM_PC matched and shuffled alignment
- FEM_PC paired delta
- J_local matched and shuffled alignment
- J_local paired delta

## Required plots

For each session:

1. paired B_model delta by centering mode;
2. paired FEM_PC delta by centering mode;
3. empirical ceiling by centering mode;
4. scatter of current vs corrected per-window deltas.

## Decision rule

For the paper-facing analysis, treat Mode C as the principled default because it matches the point at which `J_local` is computed. Use the comparison against Modes A and B diagnostically.

If Mode C improves `J_local` without degrading empirical reliability, keep it as the default centering. If Modes B and C are effectively tied for `B_model` and `FEM_PCs`, prefer the simpler interpretation that remains consistent with the Jacobian anchor.

If improvements are small, retain the simplest convention but document it explicitly.

---

# Audit 3. Window-definition ladder

## Problem

The current window definition pools all samples with the same image ID. But the model input contains a lagged stimulus history. Therefore image ID alone may mix distinct retinal/model states.

This can make empirical `B_emp` a stable average over several true geometries, while model objects are computed from a single baseline state.

## Required window definitions

Implement a ladder of increasingly specific analysis units.

### Level 0: image ID only

Current behavior. Use as baseline.

```text
window_key = image_id
```

### Level 1: image ID × time bin

Split windows by image identity and time index within trial or RSVP presentation.

```text
window_key = (image_id, time_bin)
```

Use the existing `time_indices` if available.

### Level 2: image ID × local stimulus context

Split by image identity plus nearby stimulus context, for example preceding and following image IDs or a short RSVP context hash.

Example:

```text
window_key = (image_id[t], image_id[t-1], image_id[t+1])
```

or:

```text
window_key = hash(image_id[t-L:t+1])
```

where `L` should match the model lag horizon as closely as sample counts allow.

### Level 3: exact or approximate lagged-stimulus-history hash

If feasible, compute a hash of the actual model input tensor or a reduced representation of the lagged stimulus stack.

Examples:

```python
history_key = hash(model_input_lag_stack.cpu().numpy().tobytes())
```

or, for memory efficiency:

```python
history_key = hash(tuple(image_ids[t-L:t+1]))
```

Only use this level if there are enough repeated samples per key.

## Sample count gates

For each level, record the sample-count distribution.

Required minimums:

```python
min_samples_total = 20
min_samples_per_split = 8
min_unique_eye_positions = 8
```

Do not interpret a window if it fails these gates.

Also record:

- number of valid windows
- median samples/window
- 10th and 90th percentile samples/window
- fraction of windows passing gates
- empirical split-half ceiling at that level

## Model matching rule

For each empirical window key, compute the model baseline and model objects from the matching stimulus state.

Do not compute a model object from an arbitrary baseline sample if the empirical window has a more specific stimulus-history key.

For each window:

1. collect all samples matching the key;
2. choose center eye according to the selected centering mode;
3. choose baseline stimulus sample nearest the center eye within that same key;
4. compute model geometry from that baseline stimulus or, for the mixture variant below, from each sample state.

## Required metrics

For each window-definition level and session:

- N valid windows
- median samples/window
- empirical split-half 2D alignment
- empirical split-half top-1 alignment
- empirical split-half matched-minus-eye-permutation delta
- B_model matched-minus-image-shuffled 2D delta
- FEM_PC matched-minus-image-shuffled 2D delta
- J_local matched-minus-image-shuffled 2D delta
- ceiling-normalized B_model delta
- ceiling-normalized FEM_PC delta
- ceiling-normalized J_local delta

## Required plots

For each session:

1. model bridge delta vs window-definition level;
2. empirical ceiling vs window-definition level;
3. ceiling-normalized delta vs window-definition level;
4. sample-count distribution by level;
5. scatter of per-window delta vs sample count.

## Decision rule

Run this ladder only after the frame convention and centering convention are fixed on principled grounds. Do not use the ladder to reopen those earlier choices.

If tighter window definitions increase model bridge or ceiling-normalized model bridge, the current image-only analysis was suppressing the result.

If tighter definitions reduce sample counts and empirical reliability collapses, report this as a data limitation: exact-state matching is desirable but underpowered.

If model bridge does not improve despite adequate sample counts and stricter matching, the remaining gap is more likely biological or model-related.

---

# Audit 4. Mixture-matched model object

## Rationale

If the empirical `B_emp` must pool across a mixture of stimulus histories for sample-count reasons, then the model object should be computed over the same mixture.

Instead of comparing pooled empirical `B_emp` to a single-baseline model object, compute a model eye-sensitivity object over the same sample mixture.

## Required implementation

For each broad empirical window:

1. For each sample `s` in the window:
   - retrieve the actual model input / stimulus history for sample `s`;
   - compute the eye offset relative to the chosen center;
   - transform eye offset into retinal stimulus displacement using the selected sign/axis convention;
   - generate model response for that sample's stimulus history under that displacement.
2. Fit:

```python
B_model_mix = OLS(delta_model_sample ~ eye_offset_sample)
```

where `delta_model_sample` should be defined consistently.

Possible delta definitions:

### Option A: subtract model response at center/baseline for each sample history

```python
delta_model_s = r_model(I_s shifted by retinal_offset_s) - r_model(I_s at center)
```

This is the more local finite-displacement construction, and it is useful as a supplementary comparison.

### Option B: subtract the window mean model response

```python
delta_model_s = r_model(I_s shifted by retinal_offset_s) - mean_s(r_model)
```

This is the principled default for comparing against `B_emp`, because it matches the empirical estimator's intercept-centering more closely.

Run Option B by default. Run Option A as a supplementary sensitivity check if feasible.

## Compare against

- current single-baseline `B_model`
- image-shuffled `B_model_mix`
- empirical split-half ceiling

## Metrics

- B_model_mix matched alignment
- B_model_mix shuffled alignment
- paired delta
- ceiling-normalized delta
- comparison with single-baseline B_model

## Required plot

For each session:

```text
single-baseline B_model vs mixture-matched B_model_mix
```

Use paired per-window deltas.

## Decision rule

If `B_model_mix` substantially improves alignment, then the current analysis is limited by stimulus-history mixture mismatch.

If it does not improve, the remaining mismatch likely reflects non-retinal eye signals, model inadequacy, or empirical noise.

---

# Audit 5. Empirical linearity check

## Problem

`B_emp` is linear in eye position. If true eye-linked response modulation is nonlinear over the eye cloud, linear `B_emp` captures only part of the empirical geometry. This may explain why nonlinear FEM covariance PCs bridge better than linear `B_model`.

## Required implementation

For each empirical window with sufficient samples, fit cross-validated models:

### Linear

```text
x, y
```

### Quadratic

```text
x, y, x^2, y^2, x*y
```

### Optional radial / direction model

```text
x, y, radius, radius^2, angle_sin, angle_cos
```

Use ridge regression if sample counts are modest.

## Cross-validation

Use split-half or K-fold cross-validation, keeping trial structure intact if possible.

Report:

- cross-validated variance explained by linear eye model;
- cross-validated variance explained by quadratic eye model;
- incremental variance explained by nonlinear terms;
- whether nonlinear terms improve reproducibly across windows.

## Metrics

For each session:

- median linear CV-R2
- median quadratic CV-R2
- median incremental nonlinear CV-R2
- fraction of windows with nonlinear improvement > 0
- relationship between nonlinear improvement and FEM_PC > B_model bridge advantage

## Decision rule

If nonlinear terms add substantial cross-validated explanatory power, then `B_emp` is not the complete empirical FEM geometry target. In that case, Figure 4 should emphasize finite-displacement covariance objects more than linear B_model.

---

# Audit 6. Reliability-ceiling normalization

## Problem

Model bridge effects are currently interpreted against an implicit maximum of 1.0. But empirical geometry is only recoverable to the extent that split-half empirical estimates align.

If empirical split-half alignment is 0.2 to 0.3, then a model bridge of 0.03 to 0.05 may represent a meaningful fraction of the recoverable geometry.

## Required calculations

For each window and session, compute:

```python
emp_ceiling_2d = align_2d(B_emp_A, B_emp_B)
emp_ceiling_top1 = align_top1(B_emp_A, B_emp_B)

model_matched_2d = align_2d(B_emp_full_or_split, model_basis)
model_shuffled_2d = align_2d(B_emp_full_or_split, shuffled_model_basis)

model_delta_2d = model_matched_2d - model_shuffled_2d

relative_delta_2d = model_delta_2d / emp_delta_2d
relative_matched_2d = model_matched_2d / emp_ceiling_2d
```

Be careful with denominators:

- do not compute ratios when empirical ceiling is near zero;
- define a minimum denominator, e.g. `emp_ceiling_2d > 0.05`;
- report both raw deltas and ceiling-normalized ratios.

## Required session summary

For each session and model object:

```text
object
N windows
empirical ceiling median
matched alignment median
shuffled alignment median
paired delta median
paired delta CI
median delta / empirical ceiling
median matched / empirical ceiling
```

## Required plot

For each session:

1. raw model delta;
2. empirical ceiling;
3. ceiling-normalized delta;
4. model object comparison: J_local, B_model, FEM_PCs, B_model_mix.

## Decision rule

If ceiling-normalized effects are substantial, report the bridge as a fraction of recoverable empirical geometry.

Suggested interpretation scale:

- <5% of ceiling: weak
- 5 to 15%: modest
- 15 to 30%: meaningful
- >30%: strong

These are heuristic labels, not statistical thresholds.

---

# Audit 7. Residual structure after retinal-translation prediction

## Purpose

If the model bridge remains incomplete after fixing bugs and matching windows, test whether the residual empirical eye-linked geometry looks like non-retinal eye signals.

Candidate residual sources:

- eye-position gain fields;
- microsaccade-triggered transients;
- drift velocity or temporal eye dynamics;
- pupil/state-linked modulation;
- calibration or tracking artifacts.

## Required implementation

After selecting the best retinal-translation model object:

```python
B_residual = B_emp - projection_of_B_emp_onto_model_basis
```

or compute residual response variance after regressing out model-predicted retinal-translation component.

Then test whether residual geometry correlates with:

1. absolute eye position rather than baseline-relative retinal displacement;
2. microsaccade occurrence or microsaccade rate;
3. drift velocity;
4. pupil size or pupil derivative, if available;
5. trial time / arousal proxies.

## Required outputs

- residual magnitude per window;
- residual alignment across split halves;
- residual correlation with absolute eye position;
- residual correlation with microsaccade features;
- residual correlation with pupil/state features if present.

## Interpretation

If residuals are reliable and correlate with absolute eye position, this supports gain-field or eye-in-orbit modulation.

If residuals are locked to microsaccade timing, this supports transient eye-movement responses not captured by static retinal translation.

If residuals are unreliable, the apparent gap may be mostly empirical noise.

---

# Statistical requirements

## Paired tests

Use paired-window statistics wherever possible.

Inference protocol:

- Treat the window as the unit of analysis within each session.
- Do not pool windows across sessions as independent observations.
- For each session, report the paired matched-minus-shuffled comparison across windows using a Wilcoxon signed-rank test alongside the paired bootstrap confidence interval.
- Use the four-session sign pattern as the cross-session summary. In practice, `4/4` same sign is the main bar for calling an effect cross-session consistent.

For each comparison:

```text
matched alignment - shuffled alignment
corrected mode - current mode
fine window - image-only window
mixture model - single-baseline model
```

Use bootstrap percentile confidence intervals over windows:

```python
n_boot = 10000
rng_seed = fixed integer, not Python hash()
```

Do not use Python's built-in `hash()` for RNG seeds unless `PYTHONHASHSEED` is fixed. Prefer stable hashes:

```python
import hashlib

def stable_int_hash(s, modulo=2**31):
    h = hashlib.sha256(str(s).encode("utf-8")).hexdigest()
    return int(h[:16], 16) % modulo
```

## Nulls

Primary null:

- image-shuffled model geometry, preferably matched on broad stimulus/model statistics where available.

Secondary nulls:

- eye-permuted empirical geometry;
- random 2D subspace;
- isotropic eye covariance;
- phase/history-shuffled model geometry.

Do not rely on random subspace as the primary null. It tests dimensionality, not image-specific retinal-translation geometry.

---

# Summary tables

## `audit_by_session.csv`

Required columns:

```text
session
model_object
transform_mode
centering_mode
window_level
mixture_mode
n_windows
median_samples_per_window
emp_ceiling_2d
emp_ceiling_top1
matched_2d
shuffled_2d
delta_2d
delta_2d_ci_low
delta_2d_ci_high
matched_top1
shuffled_top1
delta_top1
delta_top1_ci_low
delta_top1_ci_high
delta_over_ceiling_2d
matched_over_ceiling_2d
notes
```

## `audit_by_window.csv`

Required columns:

```text
session
window_key
image_id
time_bin
history_key
n_samples
model_object
transform_mode
centering_mode
window_level
mixture_mode
emp_ceiling_2d
emp_ceiling_top1
matched_2d
shuffled_2d
delta_2d
matched_top1
shuffled_top1
delta_top1
mean_eye_x
mean_eye_y
cov_eye_xx
cov_eye_xy
cov_eye_yy
model_j_norm
model_fem_trace
psth_amp
mean_rate
valid
failure_reason
```

## `README.md`

For each run, document:

1. git commit or repo state;
2. input files and sessions;
3. model checkpoint(s);
4. exact script command;
5. transform modes tested;
6. centering modes tested;
7. window levels tested;
8. sample gates;
9. primary null;
10. interpretation of pass/fail outcomes.

---

# Pass/fail interpretation

## Strong rescue

Criteria:

- the canonical frame fix improves B_model or FEM_PC bridge relative to the current mode;
- tighter window matching increases model alignment or ceiling-normalized alignment;
- bridge becomes a substantial fraction of empirical ceiling;
- image-shuffled controls remain lower.

Interpretation:

> The previous Figure 4 underestimated the retinal-translation bridge. When coordinate conventions and stimulus-state support are matched, model-predicted retinal-translation geometry explains a substantial fraction of recoverable FEM-linked V1 geometry.

## Partial rescue

Criteria:

- one or two fixes improve the bridge, but effects remain modest;
- ceiling-normalized ratios are meaningful but not large;
- residual empirical geometry remains reliable.

Interpretation:

> Retinal translation accounts for a measurable and meaningful fraction of recoverable FEM-linked geometry, but additional eye-linked signals contribute.

## No rescue

Criteria:

- sign/axis modes do not materially change the result;
- tighter window matching does not improve alignment when adequately powered;
- mixture-matched model objects do not improve the bridge;
- ceiling-normalized effects remain small.

Interpretation:

> The modest bridge is not an implementation artifact. FEM-linked V1 geometry only partly reflects retinal translation, with the remainder likely due to gain fields, microsaccade transients, velocity/state modulation, or model mismatch.

---

# Recommended implementation order

1. **Make a non-destructive audit script** that reproduces the current analysis exactly.
2. **Add the canonical frame fix plus an empirical eye-frame verification check** and rerun current image-ID windows.
3. **Add the confirmatory sign/axis transform grid** and verify that canonical `x_negy` is the empirical winner.
4. **Add centering modes** and rerun with the canonical sign/axis mode.
5. **Lock the principled centering choice** before broadening the window definition.
6. **Add window-definition ladder** and sample-count reporting.
7. **Add ceiling-normalized summary tables.**
8. **Add mixture-matched B_model** if window mixing appears to suppress bridge.
9. **Add empirical linearity check.**
10. **Add residual structure analysis** only after selecting the best corrected retinal-translation model.

---

# Minimal first-pass deliverable

If time is limited, implement only the following:

1. reproduce current result;
2. run the canonical frame fix plus the empirical eye-frame verification;
3. run 8 sign/axis transform modes as a confirmation sweep;
4. run 3 centering modes;
5. compute empirical ceiling-normalized bridge;
6. write `audit_by_session.csv` and one markdown summary.

This minimal pass should already reveal whether the bridge is being suppressed by coordinate/centering bugs.

---

# Expected final markdown summary format

For each session:

```markdown
## Session YYYY-MM-DD

### Current result
- empirical ceiling:
- B_model delta:
- FEM_PC delta:
- J_local delta:

### Best sign/axis mode
- mode:
- B_model delta:
- FEM_PC delta:
- top-1 improvement:
- interpretation:

### Best centering mode
- mode:
- B_model delta:
- FEM_PC delta:
- interpretation:

### Window-definition ladder
| level | n windows | median samples | empirical ceiling | B_model delta | FEM_PC delta | delta/ceiling |
|---|---:|---:|---:|---:|---:|---:|

### Conclusion
- rescue status:
- likely remaining limitation:
- recommended paper placement:
```

---

# Final note for the coding agent

Do not optimize the analysis after looking at outcomes. The audit is not a hyperparameter search for the best story. It is a falsification-oriented test of specific suspected suppressors.

Always report the current/original mode next to corrected modes. The value of the audit is knowing whether the original conclusion was robust, underestimated, or invalidated.
