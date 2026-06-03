# Coding-agent handoff: Figure 4 reconciliation and validated geometry rerun

## Purpose

Resolve the remaining blockers for the Figure 4 draft.

The manuscript now uses the framing:

> Fixational eye movements drive a first-order discrimination benefit whose second moment is reafferent V1 shared variability.

Two technical issues must be resolved before the figure and Results text can be finalized:

1. **Canonical effect-size reconciliation:** current analyses report two different real-FEM minus stabilized optotype-discrimination advantages at the finest LogMAR values, approximately 9 percentage points in the keystone mono run and approximately 5 percentage points in the neurometric / all-hires sweep. These cannot both be reported as the same Model A effect.

2. **Validated geometry rerun:** mimicry and phase-landscape results must be recomputed on the same validated mono Model A population used for the main Figure 4 discrimination analysis before they appear in Figure S4 or a main-panel schematic.

This handoff is not asking for new exploratory analyses. It is a reconciliation and regeneration task with predefined outputs and stop rules.

---

## Reviewer edits (what changed in this revision)

These were added because the original plan under-specified the single most important decision and omitted one consistency check that is the actual source of the prose discrepancy:

1. **Canonical anchor made explicit (Part 2, Part 3).** The two prior estimates are *both* nominally "756-unit mono Model A, 4-way, time-mean," yet they differ in *both* absolute accuracy (keystone real ≈ 0.94 vs neurometric real ≈ 0.89 at −0.40) and delta (+0.093 vs +0.050). The canonical pipeline is now defined as the one whose plotted absolute-accuracy curve appears in Figure 4B — the cited delta and the plotted accuracies must come from the **same run**. The original multipanel showed real ≈ 0.89 (the neurometric / `dual_regime` / `decoding.py` path), so that is the presumptive canonical unless reconciliation overturns it.
2. **Reconcile absolute accuracies, not just deltas (Part 3).** The absolutes differ too, and matching them localizes the cause faster than matching deltas alone.
3. **First lead to check (Part 3).** The keystone built its own features via `build_keystone_mono_cache_from_temporal_rates.py`, re-deriving from rate caches; the most likely cause of the divergence is that this conversion altered feature construction / normalization relative to the canonical `decoding.py` path. Check this before the broader perturbation sweep.
4. **Integration/size consistency check added (Part 2).** The window-60 point of the integration sweep must equal the size-sweep delta at the primary LogMAR. The 9-vs-5 prose discrepancy was exactly this: a size-sweep number and an integration-endpoint number that should be identical but weren't.
5. **Mechanism validation must use the canonical population (Part 4).** `mean_only` sufficiency was demonstrated in the keystone-mono pipeline; if that is not the canonical pipeline, re-confirm it on the canonical one, or the effect size and the mechanism claim rest on different populations.
6. **Mimicry guardrail sharpened (Part 5).** Mimicry is a local first-order direction-alignment; do not reconstruct `Σ_FEM ≈ JΣ_eyeJᵀ` (that bridge failed at 90,000×). The Jacobian here supplies only a local translation direction for the alignment.

---

## Scientific questions

### Q1. What is the canonical real-FEM discrimination effect?

For validated mono Model A, using the manuscript-intended decoder, trace set, population, LogMAR grid, and window length:

* What is the real-FEM minus stabilized four-way orientation accuracy at each LogMAR?
* What is the effect at the finest supported LogMAR values?
* Does the effect decay toward zero at coarser sizes?
* Is there any statistically reliable above-threshold negative limb, or should the manuscript retain the “fine-scale benefit that decays above threshold” framing?

### Q2. Why did previous pipelines report different effect sizes?

Compare the two prior effect-size sources:

* Keystone mono run, approximately +0.093 / +0.094 at −0.40 / −0.35 (absolute real ≈ 0.94).
* Neurometric or all-hires sweep, approximately +0.050 / +0.052 at the longest window (absolute real ≈ 0.89).

Note that the **absolute** accuracies differ as well, not only the deltas. Reconcile the absolute real and stabilized accuracies first — matching the absolutes will localize the cause faster than matching the deltas, and the canonical run must match the absolute-accuracy curve shown in Figure 4B. First lead to check: the keystone path re-derived features through `build_keystone_mono_cache_from_temporal_rates.py`; verify whether that conversion changed feature construction, rate normalization, or windowing relative to the canonical `decoding.py` path, since that is the most probable single cause of both the accuracy and delta divergence.

Determine whether the discrepancy is due to:

* different decoder type,
* different feature representation,
* different trial or eye-trace subset,
* different population or unit set,
* different stimulus rendering,
* different LogMAR grid,
* different window length,
* different regularization / cross-validation,
* accuracy column provenance mismatch,
* rate normalization or spike-count convention,
* random seed or split policy,
* bug or stale-cache use.

### Q3. What are the validated mono-population mimicry and phase-landscape results?

Using the exact same validated mono Model A population and stimulus/rendering conventions as the canonical discrimination analysis:

* Recompute pair-specific translation mimicry.
* Recompute phase-resolved mimicry landscapes.
* Compute occupancy-weighted or phase-averaged mimicry under real FEM traces where possible.
* Output plots and tables suitable for Figure S4.
* Do not use older cached mimicry values unless they are verified to be from the same population and rendering path.

---

## Required constraints

### Manuscript guardrails

The coding agent must preserve these interpretation boundaries:

1. Do not claim covariance geometry causes the discrimination benefit.
2. The discrimination benefit is first-order if the time-averaged mean-rate observer is sufficient.
3. Explicit temporal-trajectory features can be reported as a null if already run cleanly.
4. Do not claim eye-state conditioning did not help unless that exact experiment has been run on the validated population.
5. Eye-aware or joint readout belongs to the reafferent-covariance recoverability story, not the optotype-benefit mechanism.
6. Do not claim second-order covariance decoders are a clean null if the estimator is p≫n-unreliable.
7. Do not claim a help-to-hurt crossover unless the negative limb is reliable on the canonical pipeline.
8. Do not include mimicry or phase-landscape numbers in the manuscript unless recomputed or verified on the validated mono Model A population.

---

## Definitions

### Validated mono Model A

Use the same population and model configuration intended for the main Figure 4 discrimination analysis.

**Caution:** "mono Model A" is currently a label shared by at least two pipelines (the keystone-mono converted-cache path and the neurometric / `dual_regime` path) that produce *different* absolute accuracies. Do not assume it denotes one thing. The manifest below must pin the specific checkpoint, feature path, and trace set actually used, and Part 1/Part 3 must record whether the two pipelines are the same population or not — that ambiguity may itself be the source of the effect-size discrepancy.

The coding agent must identify and write to disk:

```text
model_name
checkpoint_path
model_epoch
model_type
population_name
n_units
unit_indices_or_mask
stimulus_ppd
retina_ppd
input_preprocessing
readout_name
readout_units
dataset_or_trace_source
eye_trace_count
random_seed
```

Save as:

```text
figure4_reconciliation/model_population_manifest.json
figure4_reconciliation/model_population_manifest.csv
```

### Conditions

Primary conditions:

```text
real_FEM
stabilized
```

Optional controls only if already available or cheap:

```text
fixed_center
random_cov
random_amp
scaled_FEM_0.5
scaled_FEM_2.0
```

Do not let optional controls block the required reconciliation.

### Primary LogMAR policy

Use the manuscript policy:

```text
primary: -0.20, -0.25, -0.30, -0.35
saturation / render-limit control: -0.40
omit by policy unless explicitly requested: -0.45, -0.50, -0.55
```

If a prior pipeline used a different grid, record it in the provenance table.

### Primary orientation task

Four-way Tumbling-E orientation discrimination:

```text
orientations = 0, 90, 180, 270
```

Pairwise analyses are secondary.

### Primary window

Use the manuscript-intended longest integration window, currently 60 frames, unless the existing canonical code defines a different value.

Also run the integration-window sweep if available:

```text
frames / windows: 1, 5, 10, 20, 30, 60
```

The exact values should match prior code when possible.

---

## Part 1: Pipeline inventory and provenance audit

### Goal

Identify every existing result file that has been used to cite the real-FEM benefit.

### Required search targets

Search output directories for:

```text
eoptotype_identity_decoder_metrics.csv
active_sensing_efficiency_decision_table.csv
active_sensing_efficiency_contrast_table.csv
keystone_readme.md
function_curve.csv
dprime_geometry_vs_D1_crosswalk.csv
geometry_dprime_decision_table.csv
*D1*
*Model_A*
*real_minus_stabilized*
*real_minus_stab*
*rate_normalized_decoder_accuracy*
*time_mean*
```

### Required output

Create:

```text
figure4_reconciliation/effect_size_source_inventory.csv
```

Required columns:

```text
source_id
file_path
run_label
date_created
analysis_name
model_name
population_name
n_units
n_traces
condition_real_label
condition_stabilized_label
logmar_values
window_values
primary_window
decoder_type
feature_representation
accuracy_column
accuracy_column_description
is_window_specific
is_aggregate
cross_validation_policy
regularization_policy
random_seed
reported_delta_at_minus_0p40
reported_delta_at_minus_0p35
reported_delta_at_minus_0p30
reported_delta_at_minus_0p25
notes
status
```

`status` must be one of:

```text
candidate_canonical
legacy_reference
stale_or_mismatched_population
wrong_accuracy_column
aggregate_not_window_specific
unusable_missing_metadata
```

### Required readme section

In:

```text
figure4_reconciliation/effect_size_reconciliation_readme.md
```

include a subsection:

```text
## Existing effect-size sources
```

Summarize which files produced the 9 percentage-point estimate and which produced the 5 percentage-point estimate.

---

## Part 2: Canonical discrimination rerun

### Goal

Run one clean canonical analysis that produces the effect size to be used in the manuscript.

### Required script

Create or adapt:

```text
scripts/figure4/run_canonical_eoptotype_discrimination.py
```

If this already exists under another name, add a wrapper with this name that calls the existing code and writes the standardized outputs below.

### Required CLI

Example:

```bash
python scripts/figure4/run_canonical_eoptotype_discrimination.py \
  --checkpoint-dir <checkpoint_dir> \
  --model-type <model_type> \
  --model-index <model_index> \
  --population validated_mono_modelA \
  --eye-traces <trace_source> \
  --conditions real_FEM stabilized \
  --logmar-values -0.40 -0.35 -0.30 -0.25 -0.20 \
  --orientations 0 90 180 270 \
  --windows 1 5 10 20 30 60 \
  --primary-window 60 \
  --feature time_mean_rate \
  --decoder <canonical_decoder> \
  --n-splits <n_splits> \
  --n-bootstrap 1000 \
  --random-seed 0 \
  --out-dir outputs/figure4_reconciliation/canonical_discrimination
```

Use repo conventions if paths or argument names differ, but the output tables must match this spec.

### Required methodological choices

**Canonical anchor:** the decoder, feature representation, rendering, and trace set must be the ones that produce the absolute-accuracy curve plotted in Figure 4B. The cited real−stabilized delta and the plotted absolute accuracies must come from this single run, so they cannot disagree. If the keystone-mono and neurometric pipelines produce different absolute accuracies (≈0.94 vs ≈0.89), only the one consistent with the figure is canonical; the other is a cross-check, not a source of headline numbers.

The script must explicitly log:

```text
decoder_type
feature_representation
regularization
cross_validation_split_policy
trial_grouping_policy
random_seed
trace_count
whether random-control repeats are treated as repeated controls or independent trials
```

### Required outputs

#### 1. Trial / feature metadata

```text
canonical_discrimination/eoptotype_trial_manifest.csv
```

Required columns:

```text
trial_index
trace_id
condition
logmar
orientation
window
n_frames
valid
mean_eye_x
mean_eye_y
eye_rms
eye_path_length
status
```

#### 2. Decoder metrics

```text
canonical_discrimination/canonical_decoder_metrics.csv
```

Required columns:

```text
run_label
condition
logmar
window
orientation_task
decoder_type
feature_representation
n_units
n_traces
n_splits
heldout_accuracy
heldout_balanced_accuracy
accuracy_ci_low
accuracy_ci_high
confusion_mi_bits
mean_total_expected_spikes
status
```

#### 3. Real-minus-stabilized contrasts

```text
canonical_discrimination/canonical_real_minus_stabilized.csv
```

Required columns:

```text
run_label
logmar
window
orientation_task
decoder_type
feature_representation
real_accuracy
stabilized_accuracy
delta_accuracy
delta_ci_low
delta_ci_high
p_sign
n_bootstrap
effect_status
```

`effect_status` must be one of:

```text
reliable_positive
near_zero
reliable_negative
wide_ci
render_limit_control
```

#### 4. Integration sweep

```text
canonical_discrimination/integration_window_sweep.csv
```

Required columns:

```text
logmar
window
real_accuracy
stabilized_accuracy
delta_accuracy
delta_ci_low
delta_ci_high
effect_status
```

**Required consistency check.** The row at `window = 60` (the primary window) and a given LogMAR must report the *same* `delta_accuracy` as the corresponding row in `canonical_real_minus_stabilized.csv` at that LogMAR. They are the same quantity viewed two ways. If they differ, the size sweep and the integration sweep are using different code paths and the discrepancy must be resolved before either number is cited — this exact mismatch (a size-sweep delta vs an integration-endpoint delta that should have been identical) is what produced the 9-vs-5 inconsistency in the prose. Write the check result to the readme.

#### 5. Confusion matrices

```text
canonical_discrimination/confusion_matrices.npz
```

Save confusion matrices by:

```text
condition × logmar × window × split
```

#### 6. Figures

Save under:

```text
canonical_discrimination/figures/
```

Required figures:

```text
fig4B_canonical_accuracy_vs_logmar.png
fig4B_canonical_delta_vs_logmar.png
fig4C_integration_window_dependence.png
fig4D_decoder_confusion_or_summary.png
```

### Required readme

Create:

```text
canonical_discrimination/canonical_discrimination_readme.md
```

Must answer:

1. What is the canonical real-minus-stabilized effect at each LogMAR?
2. What is the canonical effect at the finest non-render-limit LogMAR?
3. Does the effect decay toward zero at coarser sizes?
4. Is there a reliable negative limb?
5. Which pipeline generated the prior 9 percentage-point estimate?
6. Which pipeline generated the prior 5 percentage-point estimate?
7. Why do they differ?
8. Which number should be used in the manuscript?
9. Does this require changing the current Figure 4 prose?

---

## Part 3: Reconcile 9 percentage points versus 5 percentage points

### Goal

Determine why previous pipelines diverged.

### Required comparison table

Create:

```text
figure4_reconciliation/pipeline_difference_audit.csv
```

Rows should include at least:

```text
population
n_units
trace_source
n_traces
logmar_grid
condition_labels
window_length
feature_representation
decoder_type
regularization
split_policy
accuracy_column
rate_normalization
stimulus_rendering_path
random_seed
cache_source
```

Columns:

```text
keystone_mono_run
allhires_or_neurometric_run
canonical_rerun
match_status
difference_likely_explains_delta
notes
```

`match_status` values:

```text
same
different
unknown
not_applicable
```

`difference_likely_explains_delta` values:

```text
yes
no
possible
unknown
```

### Required minimal perturbation reruns

If feasible, rerun or recompute enough variants to isolate the discrepancy.

Suggested order:

1. Same canonical population, compare decoder types.
2. Same canonical decoder, compare trace subsets.
3. Same canonical decoder and traces, compare feature representation.
4. Same canonical decoder and traces, compare accuracy column / aggregation policy.
5. Same canonical settings, compare old cached features versus freshly rendered features.

Do not expand beyond these unless a specific implementation failure is found.

### Required terminal labels

The reconciliation readme must end with one label:

```text
effect_size_reconciled_decoder_difference
effect_size_reconciled_trace_subset_difference
effect_size_reconciled_population_difference
effect_size_reconciled_accuracy_column_error
effect_size_reconciled_stale_cache
effect_size_reconciled_other_documented
effect_size_unresolved_blocking
```

If the label is `effect_size_unresolved_blocking`, do not report a hard effect size in the manuscript.

---

## Part 4: Observer-claim validation table

### Goal

Support the narrowed mechanism claim:

> A time-averaged mean-rate observer was sufficient to reproduce the discrimination benefit, while explicit temporal-trajectory features did not carry the relevant orientation information.

Do not include eye-state conditioning, nonlinear readout, or covariance observers in the main mechanism sentence unless cleanly validated.

**Population constraint.** All observers in this table must be run on the **canonical population from Part 2**. `mean_only` sufficiency and the temporal null were established in the keystone-mono pipeline; if reconciliation (Part 3) shows that is not the canonical pipeline, re-confirm both on the canonical population. Otherwise the headline effect size and the first-order mechanism claim rest on different populations, which is the same class of mismatch this whole reconciliation exists to remove.

### Required output

Create:

```text
canonical_discrimination/observer_claim_validation.csv
```

Required columns:

```text
observer_name
question_tested
population
n_units
n_traces
feature_representation
decoder_type
logmar_values
window_values
primary_metric
canonical_delta_at_primary_logmar
status
manuscript_allowed_claim
notes
```

Rows should include:

```text
time_mean_rate_observer
temporal_trajectory_feature_observer
eye_state_conditioned_observer
nonlinear_observer
second_order_covariance_observer
```

`status` values:

```text
validated_supports_claim
validated_null
not_run
unreliable_p_gt_gt_n
mismatched_population
do_not_claim
```

For current manuscript, expected allowed claims are:

* time-mean-rate observer: sufficient to reproduce benefit
* temporal trajectory observer: did not carry relevant orientation information, if validated
* eye-state conditioning: do not claim as optotype mechanism unless run
* nonlinear observer: do not claim unless run
* covariance observer: do not claim as clean null if p≫n-unreliable

---

## Part 5: Validated mono-population mimicry and phase-landscape rerun

### Goal

Regenerate Figure S4 model geometry using the same validated mono Model A population as the canonical discrimination analysis.

**Scope guardrail.** Mimicry here is a *local, first-order direction alignment* — the projection of the orientation identity-difference direction onto the local translation direction. It is well-conditioned and does not require the population covariance. Do **not** use this rerun to reconstruct `Σ_FEM ≈ JΣ_eyeJᵀ`; that bridge failed by ~5 orders of magnitude (local tangent extrapolated over the full eye-position cloud) and is explicitly out of scope. The finite-difference translation direction supplies only the axis for the alignment, not a covariance estimate.

### Required script

Create or adapt:

```text
scripts/figure4/run_validated_mono_mimicry_phase_landscape.py
```

### Required CLI

Example:

```bash
python scripts/figure4/run_validated_mono_mimicry_phase_landscape.py \
  --checkpoint-dir <checkpoint_dir> \
  --model-type <model_type> \
  --model-index <model_index> \
  --population validated_mono_modelA \
  --logmar-values -0.35 -0.30 -0.25 -0.20 \
  --saturation-logmar -0.40 \
  --orientations 0 90 180 270 \
  --phase-grid-size 33 \
  --phase-range-arcmin <range> \
  --finite-difference-step <validated_step> \
  --eye-traces <trace_source> \
  --occupancy-weighted \
  --random-seed 0 \
  --out-dir outputs/figure4_reconciliation/validated_mimicry
```

### Required computations

#### A. Mean responses by phase

For each:

```text
logmar × orientation × phase_x × phase_y
```

compute:

```text
mu[unit]
```

Save to:

```text
validated_mimicry/phase_response_means.npz
```

and metadata to:

```text
validated_mimicry/phase_response_manifest.csv
```

#### B. Translation directions

Compute local translation directions using the validated finite-difference convention.

Save:

```text
validated_mimicry/translation_direction_metrics.csv
```

Required columns:

```text
logmar
orientation
phase_x
phase_y
j_norm_x
j_norm_y
j_rank_or_condition
fd_step
status
```

#### C. Pair-specific mimicry

For each source-target orientation pair:

```text
source_orientation
target_orientation
logmar
phase_x
phase_y
```

compute:

```text
identity_difference_norm
translation_projection_norm
mimicry_fraction
mimicry_status
```

Save:

```text
validated_mimicry/pairwise_mimicry_by_phase.csv
```

#### D. Phase-resolved landscape summaries

For each LogMAR and pair:

```text
mimicry_mean
mimicry_median
mimicry_min
mimicry_max
mimicry_p10
mimicry_p90
center_phase_mimicry
```

Save:

```text
validated_mimicry/phase_landscape_summary.csv
```

#### E. Occupancy-weighted mimicry

Using real FEM eye-position occupancy from the canonical trace set, compute weighted averages over the phase grid.

Save:

```text
validated_mimicry/occupancy_weighted_mimicry.csv
```

Required columns:

```text
logmar
orientation_pair
condition_or_trace_set
weighted_mimicry_mean
weighted_mimicry_median
unweighted_mimicry_mean
center_mimicry
n_traces
occupancy_grid_status
status
```

### Required figures

Save under:

```text
validated_mimicry/figures/
```

Required:

```text
figS4A_finite_response_neighborhood_example.png
figS4B_pairwise_mimicry_matrix_by_logmar.png
figS4C_phase_resolved_mimicry_landscape.png
figS4D_occupancy_weighted_vs_center_mimicry.png
figS4E_recoverability_schematic_inputs.png
```

The schematic itself can be made later, but the inputs should be saved.

### Required readme

Create:

```text
validated_mimicry/validated_mimicry_readme.md
```

Must answer:

1. Were all analyses run on the validated mono Model A population?
2. What unit count and trace set were used?
3. What finite-difference step was used?
4. Do mimicry values materially differ from the older cached results?
5. Is mimicry pair-specific?
6. Is mimicry phase-dependent?
7. How does occupancy-weighted mimicry compare to center-phase mimicry?
8. Are these results safe for Figure S4?
9. Do these results change the main Figure 4 claim? Expected answer: no, they characterize recoverability/confusability of reafferent covariance, not the first-order discrimination mechanism.

---

## Part 6: Final manuscript update bundle

### Goal

Produce one folder with the exact numbers, figure panels, and text snippets needed for the manuscript.

Create:

```text
outputs/figure4_reconciliation/manuscript_bundle/
```

Required files:

```text
figure4_numbers_for_text.csv
figure4_panel_file_manifest.csv
figure4_claim_checklist.csv
figure4_reconciled_results_summary.md
figure4_methods_snippet.md
figureS4_methods_snippet.md
```

### `figure4_numbers_for_text.csv`

Required rows:

```text
real_minus_stabilized_delta_primary_logmar
real_minus_stabilized_delta_minus_0p35
real_minus_stabilized_delta_minus_0p30
real_minus_stabilized_delta_minus_0p25
single_frame_delta_primary_logmar
long_window_delta_primary_logmar
noise_corr_raw_median_by_session
noise_corr_corrected_median_by_session
noise_corr_delta_by_session
fem_covariance_participation_ratio
fem_covariance_signal_alignment
eye_position_decoding_metric
```

Columns:

```text
quantity
value
ci_low
ci_high
source_file
source_row_or_filter
status
manuscript_sentence
```

### `figure4_claim_checklist.csv`

Rows:

```text
fine_scale_benefit
decays_above_threshold_not_crossover
integration_dependence
mean_rate_sufficiency
temporal_trajectory_null
first_moment_second_moment_bridge
noise_corr_reduction
low_dim_signal_aligned_covariance
eye_position_decodable
mimicry_recomputed_validated_population
```

Columns:

```text
claim
supported
source_file
allowed_wording
disallowed_wording
notes
```

### Required final readme

Create:

```text
outputs/figure4_reconciliation/final_reconciliation_readme.md
```

It must end with:

```text
Final status:
- ready_for_manuscript
- ready_except_mimicry_supplement
- ready_except_effect_size_reconciliation
- not_ready
```

If not ready, state exactly what remains.

---

## Stop rules

### Stop and report `effect_size_unresolved_blocking` if:

* the canonical rerun cannot reproduce either previous estimate,
* the provenance of one or both prior estimates cannot be determined,
* the canonical settings cannot be identified,
* the difference remains unexplained after the minimal perturbation reruns.

### Stop and report `mimicry_not_ready_for_supplement` if:

* the validated mono Model A population cannot be loaded,
* phase responses cannot be computed for the required LogMAR values,
* finite-difference translation directions are degenerate,
* mimicry output is based on an old or mismatched population.

### Do not launch new analyses if:

* the canonical effect is smaller than expected,
* the mimicry result is less visually striking than older cached plots,
* the negative limb remains absent,
* eye-state conditioning is not run,
* covariance observers remain p≫n-unreliable.

Report the result and update manuscript wording accordingly.

---

## Expected manuscript impact

### If canonical effect is reconciled

Use the reconciled effect size throughout Figure 4 and the Results text.

Allowed wording:

```text
Real FEMs improved four-way orientation discrimination below the resolution limit, with the largest effect at [LogMAR] of [Δ accuracy, CI]. The advantage decayed toward zero at coarser sizes.
```

Disallowed wording until resolved:

```text
The benefit was approximately 5 percentage points.
The benefit was approximately 9 percentage points.
```

unless one value has been identified as canonical.

### If mimicry validates on the mono population

Use Figure S4 as a model-side characterization of reafferent covariance recoverability.

Allowed wording:

```text
Translation mimicry varied across stimulus pairs and retinal phase, indicating that reafferent covariance should be recoverable when pose and identity directions are separable, but potentially confounding when retinal translation mimics identity change.
```

Disallowed wording:

```text
Mimicry explains the discrimination benefit.
The covariance geometry causes the FEM advantage.
Recorded V1 contains the same image-specific mimicry landscape.
```

### If mimicry does not validate

Keep only a schematic recoverability panel and move detailed mimicry claims out of the manuscript.

Allowed wording:

```text
The model geometry provides a framework for interpreting when reafferent covariance should be reducible versus confounding, but detailed mimicry analyses were not included because they did not survive population reconciliation.
```
