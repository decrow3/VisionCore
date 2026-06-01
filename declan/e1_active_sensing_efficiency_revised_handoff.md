# E1 Active-Sensing Efficiency Analysis: Revised Coding-Agent Handoff

## Purpose

This document supersedes the earlier spatial-efficiency-only handoff.

The previous plan was methodologically disciplined, but it primarily tested:

> Do FEMs increase spatial-position information per expected spike on fixRSVP / natural-image stimuli?

That is useful, but it does **not** directly settle Hypothesis E:

> Do real FEM trajectories improve identity readout near the acuity limit?

The revised plan keeps the spatial-information efficiency test, but adds the missing co-primary axis:

1. **Spatial SSI efficiency**: bits per expected spike from `spatial_ssi_population`.
2. **E-optotype identity efficiency**: orientation identity information / decoder performance per expected spike across a LogMAR ladder.

The E-optotype LogMAR ladder is the load-bearing addition. Without it, a positive result would support generic spatial active sensing but not the acuity-limit identity claim.

This remains a model-side analysis. Do not claim biological active-sensing benefit from model-only data.

---

## Core Scientific Questions

### Q1. Spatial-information efficiency

Do real FEMs increase spatial information per expected spike relative to:

- fixed center,
- trial-mean stabilized,
- matched-amplitude random dither,
- matched-covariance random dither?

This is a Rucci-adjacent spatial information question.

### Q2. Acuity-limit identity efficiency

Do real FEMs improve E-orientation identity readout per expected spike near the acuity transition?

This is the direct Hypothesis E test.

### Q3. Specificity of real FEM statistics

If motion helps, are real FEM statistics special, or is matched random dither sufficient?

---

## Relevant Existing Files

### Primary spatial SSI starting point

```text
check_fixrsvp_model_spatialinfo.py
```

This already:

- loads the `multidataset_120_long` model,
- builds a pooled spatial readout from `mcfarland_outputs.pkl`,
- reconstructs counterfactual stimuli from real and null eye traces,
- computes model rate maps,
- calls `spatial_ssi_population(y)` for real and null traces.

### Metric implementation

```text
spatial_info.py
```

Relevant functions:

```python
spatial_ssi_population(y, dt=1.0, eps=1e-8, log_base=2.0, spike_weighted=True)
make_stimulus_stack(...)
make_counterfactual_stim(...)
get_spatial_readout(...)
compute_rate_map(...)
compute_rate_map_batched(...)
```

### E-optotype stimulus machinery

Use the existing high-resolution E-optotype / temporal-decoding machinery, especially:

```text
scripts/temporal_decoding/stimulus_hires.py
```

and any existing functions used by the Step 1.5 / E-optotype generalized-geometry scripts.

If the relevant helper is named differently, locate it with:

```bash
grep -R "hires_counterfactual" -n scripts
grep -R "Eoptotype\|E_optotype\|logmar\|LogMAR" -n scripts
grep -R "stimulus_hires" -n scripts
```

### Related secondary file

```text
check_fixrsvp_model_fisherinfo.py
```

This contains Fisher-information code and optimized-trajectory probes. Use only as secondary or sanity-check machinery. Do not build the primary result around optimized traces.

---

## Primary Script to Create

Create one reproducible batch script with two stimulus modes.

Suggested path:

```text
scripts/jacobian_predictive_framework/run_active_sensing_efficiency.py
```

The script must support:

```text
--stim-modes fixrsvp eoptotype
```

The two modes may share rate computation, trajectory-control generation, summaries, and decision logic, but they should write separate metric tables and a combined decision table.

---

## Required Output Directory

Use run-label-based output paths, not hard-coded dates.

Example:

```text
outputs/jacobian_predictive_framework/active_sensing_efficiency_<run_label>/
```

Required subdirectories:

```text
fixrsvp_spatial_ssi/
eoptotype_identity/
figures/
logs/
```

---

## Required CLI

Example command:

```bash
python scripts/jacobian_predictive_framework/run_active_sensing_efficiency.py \
  --checkpoint-dir /mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints \
  --model-type resnet_none_convgru \
  --model-index 0 \
  --mcfarland-outputs mcfarland_outputs.pkl \
  --dataset-idx 10 \
  --stim-modes fixrsvp eoptotype \
  --out-dir outputs/jacobian_predictive_framework/active_sensing_efficiency_20260531 \
  --run-label 20260531 \
  --device cuda \
  --pilot-trials 20 \
  --max-trials 100 \
  --min-fix-dur 60 \
  --n-lags 32 \
  --out-size 151 151 \
  --dt auto \
  --n-random-controls 20 \
  --random-seed 0
```

Required optional arguments:

```text
--stim-modes fixrsvp eoptotype
--frames-per-im 2 6 12 30 60
--logmar-values -0.30 -0.20 -0.10 0.00 0.10
--orientations 0 90 180 270
--trajectory-controls real_FEM fixed_center stabilized random_amp random_cov scaled_FEM
--scaled-fem-values 0.5 2.0
--readouts linear energy multinomial
--sanity-check
--pilot-only
--skip-fixrsvp
--skip-eoptotype
```

---

## Required Analysis Structure

The script should run in this order:

1. **Load model and readout.**
2. **Extract real eye traces from fixRSVP trials.**
3. **Run sanity checks.**
4. **Run pilot analysis on ~20 trials.**
5. **Estimate CI width and decide whether full run is adequately powered.**
6. **Run full analysis if pilot passes.**
7. **Write tables, figures, and readme.**
8. **Write final decision table.**

---

## Step 0: Load Model, Readout, and Eye Traces

Reuse the old loading logic from `check_fixrsvp_model_spatialinfo.py`.

### Load model

```python
model, model_info = load_model(
    model_type=args.model_type,
    model_index=args.model_index,
    checkpoint_dir=args.checkpoint_dir,
    device="cpu",
)
model.model.eval()
model.model.convnet.use_checkpointing = True
model = model.to(device)
```

### Load pooled spatial readout

```python
with open(args.mcfarland_outputs, "rb") as f:
    outputs = dill.load(f)

readout = get_spatial_readout(model, outputs).to(device)
```

Report:

```text
n_sessions_in_readout
n_units_total
ccnorm_threshold
model_type
checkpoint_dir
dataset_idx
```

### Extract real eye traces

Use fixRSVP trial indices:

```python
train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)

inds = torch.concatenate([
    train_data.get_dataset_inds("fixrsvp"),
    val_data.get_dataset_inds("fixrsvp")
], dim=0)
```

Apply fixation and duration gates:

```python
fixation = hypot(eyepos_x, eyepos_y) < 1
good_trials = fix_dur > min_fix_dur
```

Save eye-trace metadata:

```text
trial_index
fix_dur
n_valid_frames
mean_eye_x
mean_eye_y
eye_rms_deg
eye_path_length_deg
eye_cov_xx
eye_cov_xy
eye_cov_yy
```

---

## Step 1: Verify dt Convention

This is required.

The old spatial helper uses `dt=1.0` by default. Bits per expected spike is insensitive to a constant dt at each time point, but bits/sec, total bits, and expected spike counts are not.

The model output convention must be checked before running.

### Implement `verify_dt_convention`

For one or more representative trials:

1. Compute model outputs `y`.
2. Estimate the mean response magnitude.
3. Compare total expected spikes under candidate dt values:

```python
dt_candidates = [1.0, 1/120]
```

4. If known empirical rates are available, compare to expected firing rates.

At minimum write:

```text
dt_convention_check.csv
```

Required columns:

```text
candidate_dt
mean_model_output
total_expected_spikes
mean_expected_rate_hz_if_dt_used
plausibility_status
selected_dt
selection_reason
```

If the convention is ambiguous, continue but label the rate/information-rate metrics as tentative. Bits per expected spike remains interpretable if the same dt is used across conditions.

---

## Step 2: Trajectory Controls

For each real eye trace, generate the following trajectories.

Let:

```python
eye_real = eyepos_trial  # shape [T, 2], in degrees
center = eye_real.mean(axis=0)
centered = eye_real - center
```

### 1. `real_FEM`

```python
eye_real
```

### 2. `fixed_center`

```python
zeros_like(eye_real)
```

This is strict no-FEM, centered at stimulus center.

### 3. `stabilized`

```python
center + zeros_like(centered)
```

This preserves trial-mean gaze offset but removes motion.

### 4. `random_amp`

Matched RMS amplitude and approximate temporal smoothness.

Implementation requirement:

- Generate random smooth 2D trace.
- Match real trace RMS.
- Match trace length.
- Center at trial mean gaze.
- Use smoothing scale corresponding to approximately 20–30 ms drift autocorrelation, unless measured autocorrelation is available.

For 120 Hz data:

```text
20–30 ms ≈ 2–4 frames
```

Use a default Gaussian smoothing sigma:

```python
sigma_frames = 3
```

### 5. `random_cov`

Matched 2D covariance and approximate temporal smoothness.

Implementation requirement:

- Generate smooth Gaussian trace.
- Whiten it.
- Recolor with real 2D covariance.
- Center at trial mean gaze.
- Regularize covariance if singular:

```python
cov += 1e-6 * eye(2)
```

### 6. `scaled_FEM`

Optional but useful:

```python
center + scale * centered
```

with:

```text
scale = 0.5
scale = 2.0
```

### Random-control repeats

Use:

```text
n_random_controls = 20
```

For trial-level comparisons, average the 20 random controls **within trial** before computing paired contrasts:

```text
real_i - mean_random_i
```

Also report across-random-control variability.

Do not treat 20 random repeats as 20 independent trials.

---

## Step 3: Sanity Checks

Before the full run, implement:

```bash
--sanity-check
```

### Sanity check A: SSI changes with stimulus update rate

For fixRSVP mode, run a small set of trials with:

```text
frames_per_im = 2
frames_per_im = 60
```

Expected qualitative behavior:

- slow/stable image conditions should produce stronger FEM-dependent spatial modulation than fast flashing conditions, or at least a measurable difference.

If SSI is identical across frame-rate conditions, stop and inspect stimulus construction.

### Sanity check B: trajectory controls have matched statistics

Write:

```text
trajectory_control_qc.csv
```

Required columns:

```text
trial_index
condition
random_repeat
eye_rms_deg
eye_path_length_deg
cov_xx
cov_xy
cov_yy
acf_lag1
acf_lag2
acf_lag4
matched_rms_error
matched_cov_error
```

Controls pass only if:

```text
random_amp RMS error < 10–15%
random_cov covariance error < 10–20%
```

These thresholds can be logged as warnings rather than hard failures.

### Sanity check C: rates are nonnegative and finite

For all rate maps:

```python
assert torch.isfinite(y).all()
```

If small negative rates occur:

```python
y = clamp(y, min=0)
```

Report `clamped_negative_rates=True`.

---

## Step 4: Pilot Run and Power Check

Run a pilot on:

```text
pilot_trials = 20
```

Compute paired bootstrap CIs for primary contrasts:

```text
real_FEM - fixed_center
real_FEM - stabilized
real_FEM - random_cov
```

Primary metric:

```text
cumulative_bits_per_expected_spike
```

Write:

```text
pilot_power_summary.csv
```

Required columns:

```text
metric
contrast
n_trials
median_delta
ci_low
ci_high
ci_width
abs_effect_size
ci_width_over_abs_effect
pilot_decision
```

Decision logic:

```text
if ci_width_over_abs_effect > 2 and abs_effect_size is nontrivial:
    pilot_decision = underpowered_add_trials
elif all deltas near zero and CIs tight:
    pilot_decision = likely_null_continue_or_stop
else:
    pilot_decision = proceed_full
```

The agent should report if `max_trials=100` is unlikely to resolve the effect. Do not hide underpowered results under `inconclusive_low_reliability`.

---

# Co-Primary Analysis 1: FixRSVP Spatial SSI Efficiency

## Question

Do real FEMs increase spatial information per expected spike on fixRSVP / natural-image-like repeated stimuli?

This supports a general active-sensing spatial-efficiency claim, but does not directly prove acuity-limit identity benefit.

## Stimuli

Use:

```python
make_stimulus_stack(type="fixrsvp", frame=None, frames_per_im=frames_per_im)
```

Primary stable-image setting:

```text
frames_per_im = 60
```

Optional temporal sweep:

```text
frames_per_im = 2, 6, 12, 30, 60
```

Interpretation:

- `frames_per_im=60` is the closest to static/stable-image sampling.
- The sweep asks whether FEM benefit depends on stimulus temporal update rate.
- This is not the acuity axis.

## Computation

For each trial × trajectory condition × frame-rate setting:

```python
eye_stim = make_counterfactual_stim(
    full_stack,
    eye_trace,
    ppd=ppd,
    scale_factor=scale,
    n_lags=n_lags,
    out_size=out_size,
)

y = compute_rate_map_batched(model, readout, eye_stim)

ispike_t, irate_t, I_tn = spatial_ssi_population(
    y,
    dt=selected_dt,
    spike_weighted=True
)
```

## Metrics

Compute:

```text
mean_bits_per_expected_spike
median_bits_per_expected_spike
cumulative_bits_per_expected_spike
mean_bits_per_sec
total_bits
mean_expected_spikes_per_bin
total_expected_spikes
```

### Cumulative bits per expected spike

Use cumulative numerator and denominator, not the mean of ratios.

If needed, wrap `spatial_ssi_population` or reimplement the summary to get:

```python
bits_t = sum_n(spikes_tn * I_tn)
spikes_t = sum_n(spikes_tn)
cumulative_bits_per_expected_spike = sum(bits_t) / sum(spikes_t)
```

---

# Co-Primary Analysis 2: E-optotype Identity Efficiency

## Question

Do real FEMs improve identity readout per expected spike near the acuity limit?

This directly tests Hypothesis E.

## Stimulus mode

Add:

```text
--stim-mode eoptotype
```

Use existing high-resolution E-optotype machinery.

Required LogMAR sweep:

```text
-0.30
-0.20
-0.10
0.00
+0.10
```

This must span the acuity transition more directly than the previous fine-only sweep.

Required orientations:

```text
0
90
180
270
```

Required orientation pairs:

```text
0_vs_90
0_vs_180
0_vs_270
90_vs_180
90_vs_270
180_vs_270
```

At minimum, include:

```text
0_vs_180   # energy-resolvable/control-like pair
0_vs_90
90_vs_180
90_vs_270
```

## Trajectory conditions

Use the same trajectory controls:

```text
real_FEM
fixed_center
stabilized
random_amp
random_cov
scaled_FEM_0.5
scaled_FEM_2.0
```

## Rate computation

For each orientation × LogMAR × trajectory condition × trial / trajectory repeat:

1. Render E stimulus at the specified orientation and LogMAR.
2. Sample it under the trajectory.
3. Run the model and spatial readout.
4. Summarize population responses for identity decoding.

Response features may include:

```text
time-averaged population rate vector
time-concatenated population vector
energy/rectified pooled vector
late-window average
```

Use one primary feature representation and at most one secondary.

Keep this simple and predefined.

## Required identity readouts

### 1. Linear pairwise decoder

For each orientation pair:

- train linear logistic regression or ridge/LDA-style classifier,
- use cross-validation across trials/trajectory samples,
- report held-out accuracy.

### 2. Confusion-matrix mutual information

For the four-way orientation task or pairwise task:

```text
MI(identity; decoded_identity)
```

For pairwise balanced classes, max MI is 1 bit.

### 3. Identity information per expected spike

For each condition:

```text
identity_bits_per_expected_spike = decoder_MI_bits / mean_total_expected_spikes
```

or for pairwise:

```text
pairwise_identity_bits_per_expected_spike
```

### 4. Fixed-spike-budget decoder

If feasible, normalize/subsample rates or Poisson-sample spikes to a matched expected spike budget across conditions.

Report:

```text
fixed_spike_budget_accuracy
```

This helps separate efficiency from rate increases.

## Important distinction

Do not mix spatial SSI bits with identity bits.

Use separate column names:

```text
spatial_bits_per_expected_spike
identity_bits_per_expected_spike
```

---

## Required Output Tables

### 1. `fixrsvp_spatial_trial_metrics.csv`

One row per trial × trajectory condition × stimulus setting.

Required columns:

```text
run_label
session
dataset_idx
trial_index
condition
random_repeat
stim_type
frame
frames_per_im
n_lags
out_h
out_w
dt_selected
n_time_bins
n_units
fix_dur
mean_eye_x
mean_eye_y
eye_rms_deg
eye_path_length_deg
eye_cov_xx
eye_cov_xy
eye_cov_yy
mean_spatial_bits_per_expected_spike
median_spatial_bits_per_expected_spike
cumulative_spatial_bits_per_expected_spike
mean_bits_per_sec
total_bits
mean_expected_spikes_per_bin
total_expected_spikes
clamped_negative_rates
status
```

### 2. `fixrsvp_spatial_summary_by_condition.csv`

Group by:

```text
session × condition × frames_per_im
```

Required columns:

```text
session
dataset_idx
condition
frames_per_im
n_trials
median_cumulative_spatial_bits_per_expected_spike
mean_cumulative_spatial_bits_per_expected_spike
sem_cumulative_spatial_bits_per_expected_spike
median_bits_per_sec
median_total_bits
median_total_expected_spikes
```

### 3. `eoptotype_identity_trial_metrics.csv`

One row per trial/trajectory sample × condition × LogMAR × orientation.

Required columns:

```text
run_label
session
dataset_idx
trial_index
condition
random_repeat
logmar
orientation
n_lags
out_h
out_w
dt_selected
n_time_bins
n_units
total_expected_spikes
mean_expected_spikes_per_bin
feature_representation
status
```

Include feature vectors in an `.npz` file if too large for CSV:

```text
eoptotype_identity_features.npz
```

### 4. `eoptotype_identity_decoder_metrics.csv`

One row per condition × LogMAR × orientation pair × readout.

Required columns:

```text
condition
logmar
orientation_pair
readout_type
feature_representation
n_train
n_test
n_splits
heldout_accuracy
heldout_balanced_accuracy
confusion_mi_bits
identity_bits_per_expected_spike
fixed_spike_budget_accuracy
mean_total_expected_spikes
real_minus_fixed_identity_bits_per_expected_spike
real_minus_stabilized_identity_bits_per_expected_spike
real_minus_random_amp_identity_bits_per_expected_spike
real_minus_random_cov_identity_bits_per_expected_spike
decision_status
```

### 5. `active_sensing_efficiency_contrast_table.csv`

Pairwise contrasts for both spatial SSI and identity metrics.

Required columns:

```text
analysis_mode
session
dataset_idx
stimulus_axis
logmar
frames_per_im
orientation_pair
contrast
metric
median_delta
mean_delta
bootstrap_ci_low
bootstrap_ci_high
n_trials_or_splits
p_sign
effect_status
```

### 6. `active_sensing_efficiency_decision_table.csv`

One row per mode plus grand decision.

Required rows:

```text
fixrsvp_spatial_ssi
eoptotype_identity
combined
```

Required columns:

```text
analysis_mode
primary_metric
n_trials
n_logmar
n_orientation_pairs
real_minus_fixed
real_minus_fixed_ci
real_minus_stabilized
real_minus_stabilized_ci
real_minus_random_amp
real_minus_random_amp_ci
real_minus_random_cov
real_minus_random_cov_ci
scale_dependence_status
random_control_status
rate_confound_status
decision_label
controls_passed
manuscript_implication
next_action
```

### 7. `active_sensing_efficiency_readme.md`

Must answer:

1. Does real FEM beat fixed center in spatial bits per expected spike?
2. Does real FEM beat random dither in spatial bits per expected spike?
3. Does real FEM beat fixed center in identity bits per expected spike near acuity?
4. Does real FEM beat random dither in identity bits per expected spike near acuity?
5. Does the benefit peak near the acuity transition or only monotonically track motion/rate?
6. Are benefits efficiency-specific or only bits/sec / total bits?
7. What is the final decision label?

---

## Required Figures

### Figure 1: FixRSVP spatial efficiency by condition

```text
x = condition
y = cumulative_spatial_bits_per_expected_spike
```

Use paired trial lines if legible.

### Figure 2: FixRSVP real-minus-control contrasts

```text
real - fixed_center
real - stabilized
real - random_amp
real - random_cov
```

Metric:

```text
cumulative_spatial_bits_per_expected_spike
```

### Figure 3: Frames-per-image dependence

If sweep is run:

```text
x = frames_per_im or update rate
y = real_minus_fixed spatial bits per expected spike
```

### Figure 4: E-optotype identity efficiency across LogMAR

```text
x = LogMAR
y = identity_bits_per_expected_spike
color = condition
facet = readout_type or orientation_pair
```

### Figure 5: Real-minus-control identity benefit across LogMAR

```text
x = LogMAR
y = real_minus_random_cov identity bits per expected spike
```

This is the load-bearing Hypothesis E figure.

### Figure 6: Identity readout vs expected spikes

Scatter:

```text
x = total_expected_spikes
y = identity_bits_per_expected_spike
color = condition
```

Checks whether apparent benefit is just rate-related.

---

## Bootstrap / Statistics

Use paired bootstrap where possible.

### FixRSVP spatial SSI

Pair by trial.

For random controls:

```text
random_metric_i = mean over random repeats within trial
delta_i = real_i - random_metric_i
```

Bootstrap trials.

### E-optotype identity

Use cross-validation splits and/or trial-level bootstrap.

For each LogMAR × pair × condition:

- compute held-out decoder metrics,
- bootstrap over trials / trajectories if features are trial-indexed,
- otherwise bootstrap CV splits with caution and label accordingly.

Report if identity decoder statistics are split-level rather than trial-level.

---

## Decision Logic

### `model_active_sensing_efficiency_supported`

Criteria:

```text
E-optotype identity:
real_FEM > fixed_center in identity_bits_per_expected_spike
AND real_FEM > random_cov in identity_bits_per_expected_spike
AND benefit is strongest near acuity transition or high-frequency/fine-scale regime
AND benefit is not explained solely by increased total expected spikes
```

Manuscript implication:

```text
Can add model-side active-sensing efficiency panel to Figure 4.
```

### `generic_dither_efficiency_supported`

Criteria:

```text
real_FEM > fixed_center
BUT random_cov ≈ real_FEM
```

Manuscript implication:

```text
Motion/phase sampling helps, but real FEM statistics are not uniquely privileged.
```

### `spatial_efficiency_only`

Criteria:

```text
fixRSVP spatial SSI positive
BUT E-optotype identity efficiency absent or inconclusive
```

Manuscript implication:

```text
Use as supporting model evidence only, not a Hypothesis E headline.
```

### `rate_or_total_information_only`

Criteria:

```text
real_FEM increases bits/sec / total bits
BUT not bits per expected spike
```

Manuscript implication:

```text
Do not claim efficiency.
```

### `no_efficiency_benefit`

Criteria:

```text
tight CIs around zero for real-minus-control bits/spike contrasts
```

Manuscript implication:

```text
Keep Figure 4 focused on aggregate covariance and identity/transformation geometry.
```

### `fem_efficiency_cost`

Criteria:

```text
real_FEM < fixed_center or stabilized in bits per expected spike
```

Manuscript implication:

```text
FEMs may diversify or structure responses while reducing efficiency under this metric.
```

### `underpowered`

Criteria:

```text
CIs are too wide to adjudicate plausible effect sizes
```

Do not call this a biological or model null.

### `inconclusive_mixed`

Criteria:

```text
effects are sign-inconsistent across LogMAR, orientation pairs, or controls
```

Manuscript implication:

```text
Do not widen the slice. Summarize as mixed and stop.
```

---

## Cross-Axis Interpretation Rules

### FixRSVP positive, E-optotype identity null

Interpretation:

```text
FEMs may improve spatial-position information on natural/fixRSVP stimuli, but this does not establish acuity-limit identity benefit.
```

Manuscript role:

```text
supporting only
```

### FixRSVP null, E-optotype identity positive

Interpretation:

```text
FEM benefits may be specific to fine controlled stimuli near acuity, not broad natural-image spatial SSI.
```

Manuscript role:

```text
potential active-sensing mechanism, model-side
```

### Both positive

Interpretation:

```text
strongest model-side active-sensing efficiency support
```

Still require:

```text
real_FEM > random_cov
```

before saying real FEM statistics matter.

### Both null

Interpretation:

```text
do not include active-sensing efficiency as main result
```

### Identity improves but bits/spike does not

Interpretation:

```text
readout benefit may be rate/drive mediated rather than efficiency mediated
```

Do not claim efficiency.

---

## Implementation Guardrails

1. Do not use orientation/image shuffle as random dither.
2. Do not headline total bits or bits/sec.
3. Do not call spatial SSI identity decoding.
4. Do not claim real FEM optimization unless real_FEM beats matched random-dither controls.
5. Do not treat random-control repeats as independent trials.
6. Do not bury underpowered outcomes under generic inconclusive labels.
7. Do not run new exploratory variants after this analysis unless a specific implementation failure is found.
8. Keep model-only and neural-data claims separate.
9. Keep fixed-center and stabilized distinct.
10. Use run-label paths, not hard-coded dates.

---

## Minimal Acceptance Criteria

The run is complete only if it produces:

```text
dt_convention_check.csv
trajectory_control_qc.csv
pilot_power_summary.csv
fixrsvp_spatial_trial_metrics.csv
fixrsvp_spatial_summary_by_condition.csv
eoptotype_identity_trial_metrics.csv
eoptotype_identity_decoder_metrics.csv
active_sensing_efficiency_contrast_table.csv
active_sensing_efficiency_decision_table.csv
active_sensing_efficiency_readme.md
at least 5 figures
```

The readme must end with:

```text
Final E1 status:
- model_active_sensing_efficiency_supported
- generic_dither_efficiency_supported
- spatial_efficiency_only
- rate_or_total_information_only
- no_efficiency_benefit
- fem_efficiency_cost
- underpowered
- inconclusive_mixed
```

and explicitly state whether this changes Figure 4.

---

## Final Stop Rule

After this analysis, stop.

This is the adjudicating E1 test. If it is mixed, summarize it as mixed. Do not expand into another search tree.

The only exception is an implementation failure identified by:

- failed dt convention check,
- failed trajectory-control QC,
- broken stimulus rendering,
- nonfinite or degenerate model outputs,
- decoder cannot learn even in an easy positive-control condition.

In that case, fix the implementation and rerun the predefined analysis only.
