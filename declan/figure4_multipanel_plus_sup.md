# Coding-agent handoff: Generate Figure 4 multipanel figure

## Purpose

Generate the final multipanel Figure 4 for the manuscript section currently framed as:

> Fixational eye movements drive a first-order discrimination benefit whose second moment is reafferent V1 shared variability.

The figure should show that:

1. Real fixational eye movements (FEMs) alter retinal input relative to stabilization.
2. In the digital twin, real FEMs improve fine-scale optotype discrimination after sufficient integration.
3. The benefit is first-order: it is carried by time-averaged mean rates, not by an explicit temporal-trajectory code.
4. The same eye-conditioned response modulation appears across repeated trials as FEM-linked covariance.
5. In recorded V1, removing the FEM-linked component reduces positive noise correlations and exposes a reafferent component of shared variability.
6. Model-side geometry illustrates when this reafferent covariance is recoverable pose information versus potentially confounding identity/pose overlap.

This is a figure-generation task, not a new analysis task, except where specific missing covariance-branch values must be pulled from existing outputs.

---

## Figure title

Use one of the following:

### Main figure title

```text
Fixational eye movements drive a first-order discrimination benefit whose second moment is reafferent V1 shared variability
```

### Shorter display title, if needed

```text
FEMs improve fine discrimination and appear as reafferent shared variability
```

---

## Required output directory

Create:

```text
outputs/figure4_reconciliation/final_figure4/
```

Subdirectories:

```text
outputs/figure4_reconciliation/final_figure4/
  panels/
  source_tables/
  qc/
  exports/
```

---

## Required figure export files

Save the assembled figure as:

```text
outputs/figure4_reconciliation/final_figure4/exports/Figure4_multipanel.pdf
outputs/figure4_reconciliation/final_figure4/exports/Figure4_multipanel.svg
outputs/figure4_reconciliation/final_figure4/exports/Figure4_multipanel.png
```

Use high-resolution PNG export:

```text
dpi = 600
```

Also save individual panels:

```text
outputs/figure4_reconciliation/final_figure4/panels/Fig4A_counterfactual_input.png
outputs/figure4_reconciliation/final_figure4/panels/Fig4B_accuracy_vs_logmar.png
outputs/figure4_reconciliation/final_figure4/panels/Fig4C_integration_dependence.png
outputs/figure4_reconciliation/final_figure4/panels/Fig4D_observer_mechanism.png
outputs/figure4_reconciliation/final_figure4/panels/Fig4E_moments_bridge.png
outputs/figure4_reconciliation/final_figure4/panels/Fig4F_reafferent_covariance_recorded_v1.png
outputs/figure4_reconciliation/final_figure4/panels/Fig4G_recoverability_geometry.png
```

Save each panel as both `.png` and `.svg` where possible.

---

## Required source data

### Canonical discrimination outputs

Use:

```text
outputs/figure4_reconciliation/canonical_discrimination/canonical_decoder_metrics.csv
outputs/figure4_reconciliation/canonical_discrimination/canonical_real_minus_stabilized.csv
outputs/figure4_reconciliation/canonical_discrimination/integration_window_sweep.csv
outputs/figure4_reconciliation/canonical_discrimination/observer_claim_validation.csv
outputs/figure4_reconciliation/manuscript_bundle/figure4_numbers_for_text.csv
outputs/figure4_reconciliation/manuscript_bundle/figure4_claim_checklist.csv
```

### Validated mimicry outputs

Use:

```text
outputs/figure4_reconciliation/validated_mimicry/phase_landscape_summary.csv
outputs/figure4_reconciliation/validated_mimicry/pairwise_mimicry_by_phase.csv
outputs/figure4_reconciliation/validated_mimicry/occupancy_weighted_mimicry.csv
outputs/figure4_reconciliation/validated_mimicry/translation_direction_metrics.csv
```

### Dedicated covariance-branch outputs

The current `figure4_numbers_for_text.csv` has missing entries for the covariance branch. The plotting script must either locate these outputs or fail gracefully with a clear message.

Required quantities:

```text
noise_corr_raw_median_by_session
noise_corr_corrected_median_by_session
noise_corr_delta_by_session
fem_covariance_participation_ratio
fem_covariance_signal_alignment
eye_position_decoding_metric
```

Expected source candidates include Phase 1 / covariance outputs, for example:

```text
outputs/phase1_fem_covariance/summaries/phase1_master_summary.csv
outputs/phase1_fem_covariance/noise_correlations/noise_correlation_session_metrics.csv
outputs/phase1_fem_covariance/covariance_geometry/covariance_geometry_session_metrics.csv
outputs/phase1_fem_covariance/covariance_geometry/subspace_alignment_metrics.csv
outputs/phase1_fem_covariance/summaries/phase1_decision_table.csv
```

If the script cannot find these, write:

```text
outputs/figure4_reconciliation/final_figure4/qc/missing_covariance_branch_values.md
```

and still generate panels A to E plus a placeholder panel F labeled:

```text
Covariance-branch values pending
```

Do not silently substitute old or hard-coded numbers.

---

## Global visual style

Use a clean manuscript style.

Suggested defaults:

```python
font_family = "Arial"
font_size_axis = 7
font_size_label = 8
font_size_panel_letter = 10
line_width = 1.0
marker_size = 3.5
```

Figure dimensions:

```text
single-column draft: 7.2 in wide × 8.5 in tall
final multipanel: 7.2 in wide × 9.0 in tall if Panel G is included
```

Suggested layout:

```text
Row 1: A | B
Row 2: C | D
Row 3: E | F
Row 4: G full-width or G + optional inset
```

Alternative compact layout:

```text
A B C
D E F
G full-width
```

Use the same condition colors across all panels:

```text
real_FEM: dark blue
stabilized: dark gray
delta / real-minus-stabilized: black
positive effect shading: light blue
negative effect shading: light red or gray
```

If using colorblind-safe palette, prefer:

```text
real_FEM: #0072B2
stabilized: #4D4D4D
fixed_center: #999999
random_cov: #009E73
negative_delta: #D55E00
```

Make sure colors are defined in one shared style file or function.

---

## Panel A: Counterfactual retinal input

### Goal

Show the conceptual manipulation: the same optotype is sampled under real FEMs or held stabilized at the trial-mean eye position.

### Required content

Panel A should include:

1. A Tumbling-E stimulus patch.
2. A real FEM trajectory overlaid on the stimulus or retinal coordinate frame.
3. A stabilized condition shown as a fixed point at the trial-mean gaze position.
4. Labels:

```text
Real FEM: retinal image moves across sampled phases
Stabilized: motion removed, trial-mean gaze preserved
```

### Input data

Use one representative eye trace from:

```text
canonical_discrimination/eoptotype_trial_manifest.csv
```

Select a trial satisfying:

```text
condition == real_FEM
window == 60
logmar == -0.35
valid == true
moderate eye_rms, not an outlier
```

If stimulus rendering functions are available, render the actual optotype at LogMAR −0.35. Otherwise, draw a schematic E with accurate condition labels.

### Plot details

Suggested subpanels within A:

```text
A1: real FEM path over stimulus
A2: stabilized point
```

Use identical axes and scale.

Add a small scale bar if possible:

```text
1 arcmin
```

### Allowed interpretation

```text
Exact stabilization is a model-side counterfactual that removes retinal motion while preserving trial-mean gaze.
```

### Disallowed interpretation

```text
The animal experienced the stabilized condition.
```

---

## Panel B: Discrimination across optotype size

### Goal

Show canonical four-way orientation accuracy for real FEM and stabilized conditions across LogMAR.

### Required input

Use:

```text
canonical_discrimination/canonical_decoder_metrics.csv
canonical_discrimination/canonical_real_minus_stabilized.csv
```

Primary window:

```text
window = 60
```

Primary task:

```text
orientation_task = four_way
```

Conditions:

```text
real_FEM
stabilized
```

Note that in the actual CSV the condition labels may be:

```text
real
stabilized
```

Map labels consistently.

### Required plotted values

From the reconciled canonical table:

```text
LogMAR −0.35:
delta_accuracy = +0.050955414012738856
95% CI = [0.02971072186836518, 0.0727176220806794]

LogMAR −0.30:
delta_accuracy = −0.03662420382165605
95% CI = [−0.059447983014861996, −0.014331210191082803]

LogMAR −0.25:
delta_accuracy = −0.1040339702760085
95% CI = [−0.1321656050955414, −0.07534501061571128]
```

Also include LogMAR −0.40 as render-limit or saturation control if present, but visually distinguish it from the primary grid.

### Recommended plot

Main axis:

```text
x = LogMAR
y = four-way held-out accuracy
lines = real_FEM, stabilized
points = mean or heldout accuracy
error bars = 95% CI if available
```

Inset or lower axis:

```text
x = LogMAR
y = real_FEM − stabilized accuracy
horizontal line at zero
```

If space is tight, use only the delta plot in panel B and show absolute accuracy lightly in the background.

### Required annotation

At LogMAR −0.35:

```text
Δ = +0.051 [0.030, 0.073]
```

At coarser sizes, annotate or mark negative deltas:

```text
−0.30: Δ = −0.037
−0.25: Δ = −0.104
```

### Interpretation

The panel should support the sentence:

```text
Real FEMs improved four-way orientation discrimination at the finest non-render-limit scale tested, but impaired discrimination at coarser sizes where stabilized samples were already informative.
```

### Important language update

Do not use:

```text
benefit decays toward zero
```

Use:

```text
scale-dependent sign-changing effect
```

or:

```text
fine-scale benefit and coarser-scale cost
```

because the canonical values are reliably negative at −0.30 and −0.25.

---

## Panel C: Integration dependence

### Goal

Show that the fine-scale benefit emerges with temporal integration.

### Required input

Use:

```text
canonical_discrimination/integration_window_sweep.csv
```

Primary LogMAR:

```text
logmar = -0.35
```

Plot:

```text
x = integration window
y = real_FEM − stabilized accuracy
```

### Required plotted values

From the manuscript bundle:

```text
window = 1:
delta_accuracy = −0.45116772823779194
95% CI = [−0.4755971337579618, −0.42461518046709135]

window = 60:
delta_accuracy = +0.050955414012738856
95% CI = [0.02971072186836518, 0.0727176220806794]
```

Other window values should be read directly from:

```text
integration_window_sweep.csv
```

### Plot details

Use a line plot with markers and vertical CI bars.

Add:

```text
horizontal zero line
```

Consider using a log-scaled x-axis if windows are irregular, but ordinary categorical spacing is acceptable.

### Required annotation

Add text or arrows:

```text
single frame: real FEM worse
60 frames: real FEM better
```

or:

```text
benefit emerges with integration
```

### Interpretation

The panel should support:

```text
The functional advantage is not instantaneous. It emerges from accumulation across multiple retinal samples.
```

### Disallowed interpretation

Do not say:

```text
temporal trajectory code explains the benefit
```

The current mechanism claim is first-order time-averaged mean-rate sufficiency, not temporal-code sufficiency.

---

## Panel D: First-order observer mechanism

### Goal

Show that the discrimination benefit is attributable to time-averaged class means, while explicit temporal-trajectory features do not carry the relevant orientation information.

### Required input

Use:

```text
canonical_discrimination/observer_claim_validation.csv
```

This table should include rows for:

```text
time_mean_rate_observer
temporal_trajectory_feature_observer
eye_state_conditioned_observer
nonlinear_observer
second_order_covariance_observer
```

### Required interpretation of statuses

Expected allowed claims:

```text
time_mean_rate_observer:
  status = validated_supports_claim
  allowed claim = sufficient to reproduce benefit

temporal_trajectory_feature_observer:
  status = validated_null or validated according to table
  allowed claim = explicit temporal-trajectory features did not carry relevant orientation information, if validated

eye_state_conditioned_observer:
  status = not_run or do_not_claim
  allowed claim = no optotype-mechanism claim

nonlinear_observer:
  status = not_run or do_not_claim
  allowed claim = no optotype-mechanism claim

second_order_covariance_observer:
  status = unreliable_p_gt_gt_n or do_not_claim
  allowed claim = no clean null; mean-rate sufficiency makes covariance unnecessary
```

### Recommended plot

Use a compact status matrix or bar summary.

Option 1: status matrix

```text
rows = observer classes
columns = status / allowed manuscript claim
icons = check, not run, unreliable
```

Option 2: performance bars

Only if comparable metrics exist on the canonical population:

```text
x = observer
y = Δ accuracy at LogMAR −0.35, window 60
```

Include only observers with validated comparable results.

### Preferred panel D if not all observer metrics are numeric

A schematic table is acceptable and may be clearer:

```text
Observer                         Figure 4 claim
Time-averaged mean rate           sufficient
Temporal trajectory features      no relevant orientation information
Eye-state-conditioned readout      not claimed here
Nonlinear observer                 not claimed
Second-order covariance observer   p≫n-unreliable, not a clean null
```

### Required caption language

```text
A time-averaged mean-rate observer is sufficient to reproduce the effect. Explicit temporal-trajectory features do not explain the benefit. Eye-aware recoverability is treated separately as a property of reafferent covariance, not as the optotype mechanism.
```

---

## Panel E: First moment / second moment bridge

### Goal

Provide the conceptual bridge between the optotype functional result and the noise-correlation / covariance result.

### Required content

Draw a schematic or equation panel with the eye-conditioned response:

```text
R(t,e) = E[Y | t,e]
```

Show two branches:

```text
First moment:
within-trial average over eye trajectory
→ shifted class means
→ optotype discrimination

Second moment:
across-trial covariance over eye states
→ Σ_FEM
→ reafferent shared variability / noise correlations
```

Use the equation:

```text
Σ_FEM = E_t[Cov_e(R(t,e) | t)]
```

Make sure symbols render correctly in vector output.

### Recommended visual structure

```text
center: R(t,e)
left branch: mean over e along trajectory
right branch: covariance over e across trials
```

Use labels:

```text
within-trial first moment
across-trial second moment
```

### Caption language

```text
The discrimination benefit and the reafferent covariance are different statistical moments of the same eye-conditioned response modulation.
```

### Disallowed language

Do not state:

```text
Σ_FEM causes the discrimination benefit.
```

Use:

```text
linked but mechanistically distinct
```

---

## Panel F: Reafferent shared variability in recorded V1

### Goal

Show that FEM correction reduces positive noise correlations and that the removed covariance has the expected reafferent structure.

### Required input

Use dedicated covariance-branch outputs.

The script should try to locate:

```text
outputs/phase1_fem_covariance/noise_correlations/noise_correlation_session_metrics.csv
outputs/phase1_fem_covariance/covariance_geometry/covariance_geometry_session_metrics.csv
outputs/phase1_fem_covariance/covariance_geometry/subspace_alignment_metrics.csv
outputs/phase1_fem_covariance/summaries/phase1_master_summary.csv
```

### Required data values

At minimum:

```text
session
raw_noise_corr_median
eye_corrected_corr_median
noise_corr_delta
```

Preferably also:

```text
participation_ratio
top2_variance_fraction
stimulus_subspace_captures_FEM_variance
FEM_subspace_captures_stimulus_variance
eye_position_decoding_metric
```

### Recommended plot

Use two subcomponents:

#### F1: Noise correlation reduction

Plot paired points by session:

```text
x = raw, eye-corrected
y = median pairwise noise correlation
line = session
```

Add a zero line.

If exact values are available from the Phase 1 interim summary, expected values are approximately:

```text
2022-02-16: raw 0.0315, corrected 0.0240
2022-02-24: raw 0.0267, corrected 0.0159
2022-03-04: raw 0.0197, corrected 0.0122
2022-04-08: raw 0.0350, corrected 0.0241
```

But do not hard-code these unless loaded from a source table.

#### F2: Covariance geometry summary

Plot one or more of:

```text
participation ratio of Σ_FEM
top-2 variance fraction
signal/FEM subspace overlap
eye-position decoding metric
```

If space is tight, show these as small annotated summary boxes.

### Required interpretation

```text
FEM correction reduces positive shared variability. The removed component is low-dimensional and signal-aligned, consistent with reafferent covariance.
```

### Important wording

Do not claim:

```text
FEM correction reveals negative residual correlations
```

unless the canonical covariance branch supports it. Current framing should be:

```text
corrected correlations approach zero rather than revealing a robust negative residual
```

---

## Panel G: Geometry of recoverability

### Goal

Show how reafferent covariance can be recoverable pose variation or confounding identity/pose overlap depending on identity/translation geometry.

This is a model-side interpretive panel. It should not be presented as the mechanism of the optotype benefit.

### Required input

Use validated mono-population outputs:

```text
validated_mimicry/phase_landscape_summary.csv
validated_mimicry/pairwise_mimicry_by_phase.csv
validated_mimicry/occupancy_weighted_mimicry.csv
validated_mimicry/translation_direction_metrics.csv
```

### Required checks before plotting

Confirm:

```text
all analyses run on validated mono Model A population
mimicry_fraction finite
mimicry values in [0,1]
phase grid complete for requested LogMAR values
occupancy-weighted outputs available
```

Write results to:

```text
final_figure4/qc/panelG_mimicry_qc.md
```

### Recommended plot options

#### Option 1: schematic plus small matrix

Main schematic:

```text
low mimicry:
translation direction orthogonal or separable from identity direction
→ recoverable pose covariance

high mimicry:
translation direction aligned with identity difference
→ confounding for pose-ignorant readout
```

Add a small mimicry matrix for one representative LogMAR, preferably −0.35 or −0.30.

#### Option 2: occupancy-weighted mimicry summary

Plot:

```text
x = orientation pair
y = occupancy-weighted mimicry
points or bars = LogMAR
```

Compare:

```text
center_mimicry
occupancy_weighted_mimicry
```

#### Option 3: phase landscape inset

Show one phase-resolved heatmap from:

```text
phase_landscape_summary.csv
or reconstructed grid from pairwise_mimicry_by_phase.csv
```

Use only if visually clear.

### Recommended main Figure 4 version

For the main figure, use a schematic with one small quantitative inset. Put detailed phase landscapes in Figure S4.

### Required caption language

```text
Translation mimicry characterizes when reafferent covariance is recoverable pose variation versus potentially confounding identity/pose overlap. These model-side analyses characterize the coding consequences of the covariance removed by the LOTC decomposition; they do not explain the first-order discrimination benefit.
```

---

## Suggested Figure S4

Generate a supplementary figure from the validated mimicry outputs.

### Figure S4 title

```text
Model geometry of reafferent covariance recoverability
```

### Suggested panels

#### S4A: Finite response neighborhood

Show a low-dimensional projection of responses to translated versions of one optotype.

Input:

```text
phase_response_means.npz
```

Plot:

```text
PCA or two-dimensional projection of μ(orientation, phase)
```

#### S4B: Pair-specific mimicry matrix

Input:

```text
phase_landscape_summary.csv
```

Plot:

```text
rows = source orientation
columns = target orientation
value = mean or occupancy-weighted mimicry
```

#### S4C: Phase-resolved mimicry landscape

Input:

```text
pairwise_mimicry_by_phase.csv
```

Plot heatmap:

```text
x = phase_x
y = phase_y
color = mimicry_fraction
```

Use a representative orientation pair and LogMAR.

#### S4D: Occupancy-weighted versus center mimicry

Input:

```text
occupancy_weighted_mimicry.csv
```

Plot:

```text
x = center_mimicry
y = occupancy_weighted_mimicry
point = orientation pair × LogMAR
line y=x
```

#### S4E: Recoverability schematic

Schematic only.

---

## Figure-level QC

Before final export, run these checks and write:

```text
outputs/figure4_reconciliation/final_figure4/qc/figure4_qc_report.md
```

### Required checks

#### Data availability

```text
canonical_decoder_metrics.csv loaded
canonical_real_minus_stabilized.csv loaded
integration_window_sweep.csv loaded
observer_claim_validation.csv loaded
covariance branch table loaded or placeholder used
validated_mimicry tables loaded
```

#### Canonical consistency

Check that:

```text
canonical_real_minus_stabilized.csv and integration_window_sweep.csv agree exactly for shared logmar × window rows
```

This was previously verified, but recheck inside the plotting script.

#### Effect-size checks

Verify that the following values are present:

```text
LogMAR −0.35, window 60:
delta = +0.050955414012738856
CI = [0.02971072186836518, 0.0727176220806794]

LogMAR −0.30, window 60:
delta = −0.03662420382165605
CI = [−0.059447983014861996, −0.014331210191082803]

LogMAR −0.25, window 60:
delta = −0.1040339702760085
CI = [−0.1321656050955414, −0.07534501061571128]

LogMAR −0.35, window 1:
delta = −0.45116772823779194
CI = [−0.4755971337579618, −0.42461518046709135]
```

If any value differs beyond rounding tolerance, stop and write:

```text
effect_size_mismatch
```

#### Claim validation

Check:

```text
time_mean_rate_observer status allows sufficiency claim
temporal_trajectory_feature_observer status allows null claim, if plotted
eye_state_conditioned_observer not used for optotype mechanism claim unless validated
second_order_covariance_observer not described as clean null if status is unreliable_p_gt_gt_n
```

#### Mimicry validation

Check:

```text
mimicry tables generated on validated mono Model A population
mimicry_fraction in [0,1]
no NaN or Inf in plotted columns
```

#### Covariance branch

Check whether all required covariance values are present.

If missing, panel F must display placeholder and QC report must state:

```text
Panel F requires dedicated covariance branch outputs before manuscript submission.
```

---

## Required plotting script

Create:

```text
scripts/figure4/plot_figure4_multipanel.py
```

### Required CLI

```bash
python scripts/figure4/plot_figure4_multipanel.py \
  --reconciliation-dir outputs/figure4_reconciliation \
  --covariance-dir outputs/phase1_fem_covariance \
  --out-dir outputs/figure4_reconciliation/final_figure4 \
  --include-panel-g \
  --export-pdf \
  --export-svg \
  --export-png \
  --dpi 600
```

Optional flags:

```text
--skip-panel-g
--panel-f-placeholder-ok
--make-supplement-s4
--style manuscript
--style talk
```

### Fail behavior

The script should fail if canonical discrimination inputs are missing or inconsistent.

The script may continue with a placeholder if covariance branch values are missing only when:

```text
--panel-f-placeholder-ok
```

is passed.

---

## Methods snippets to write

Save:

```text
outputs/figure4_reconciliation/final_figure4/qc/figure4_methods_snippet.md
```

Include concise methods for:

1. Counterfactual E-optotype rendering.
2. Real-FEM and stabilized conditions.
3. Four-way decoder.
4. Integration-window analysis.
5. Observer comparison.
6. LOTC / FEM covariance definition.
7. Noise-correlation correction.
8. Mimicry / recoverability analysis.

### Required methods language for discrimination

```text
We rendered four Tumbling-E orientations across a LogMAR ladder and passed each stimulus through the frozen V1 digital twin under measured real-FEM trajectories or a stabilized counterfactual that held gaze at the trial mean. Four-way orientation accuracy was estimated from held-out trials using the canonical time-averaged mean-rate feature representation. Real-minus-stabilized contrasts were bootstrapped over trials or splits using the same cross-validation policy as the decoder.
```

### Required methods language for moments bridge

```text
We define the eye-conditioned mean population response as R(t,e)=E[Y|t,e]. The first-order discrimination analysis depends on the within-trial average of R(t,e) over the sampled eye trajectory. The FEM-linked covariance is the second moment of this same modulation, Σ_FEM=E_t[Cov_e(R(t,e)|t)], and corresponds to the component of shared variability removed by conditioning on eye position.
```

### Required methods language for mimicry

```text
Translation mimicry was computed as a first-order alignment between orientation identity-difference directions and local retinal-translation directions in the validated mono Model A population. This analysis was used to characterize when reafferent covariance should be recoverable as pose variation versus confounding for pose-ignorant readouts. It was not used to estimate Σ_FEM via a local JΣ_eyeJᵀ approximation.
```

---

## Final figure caption draft

Generate a draft caption file:

```text
outputs/figure4_reconciliation/final_figure4/qc/figure4_caption_draft.md
```

Use this starting caption:

```text
Figure 4. Fixational eye movements drive a first-order discrimination benefit whose second moment is reafferent V1 shared variability.

(A) Counterfactual digital-twin manipulation. Under real FEMs, fine optotypes sweep across retinal positions during fixation. Under stabilization, retinal motion is removed while preserving trial-mean gaze position.

(B) Four-way Tumbling-E orientation accuracy across LogMAR. Real FEMs improved discrimination at the finest non-render-limit scale tested (LogMAR −0.35; Δ accuracy = +0.051 [0.030, 0.073]) but impaired performance at coarser sizes (−0.30 and −0.25), revealing a scale-dependent sign-changing effect.

(C) The fine-scale benefit emerged with integration. At LogMAR −0.35, real FEMs were strongly worse than stabilization at a single frame (Δ = −0.451 [−0.476, −0.425]) but better after 60-frame integration (Δ = +0.051 [0.030, 0.073]).

(D) The benefit was first-order. A time-averaged mean-rate observer reproduced the discrimination effect, whereas explicit temporal-trajectory features did not carry the relevant orientation information.

(E) The first-moment / second-moment relationship. The within-trial average of the eye-conditioned response R(t,e) shifts class means and supports discrimination. Across trials, the variance of the same eye-conditioned modulation is Σ_FEM, the reafferent covariance component.

(F) In recorded foveal V1, removing the FEM-linked component reduces positive noise correlations and identifies a low-dimensional, signal-aligned component of shared variability. Exact covariance-branch values should be inserted from the dedicated covariance outputs.

(G) Model-side translation mimicry illustrates the recoverability of reafferent covariance. When retinal-pose directions are separable from identity directions, FEM-linked covariance is potentially reducible. When retinal translation mimics identity change, the same covariance can be confounding for pose-ignorant readouts.
```

---

## Final manuscript claims allowed from Figure 4

Allowed:

```text
Real FEMs improved four-way orientation discrimination at LogMAR −0.35 after 60-frame integration.
The effect was scale-dependent and sign-changing, with coarser-scale costs at −0.30 and −0.25.
The fine-scale benefit emerged with integration and was not present at a single frame.
The benefit was first-order and captured by a time-averaged mean-rate observer.
The same eye-conditioned modulation has a second moment, Σ_FEM, that appears as reafferent shared variability.
FEM-linked covariance should be interpreted as reafferent latent-variable covariance, not internal noise by default.
Mimicry characterizes recoverability/confusability of reafferent covariance, not the cause of the discrimination benefit.
```

Disallowed:

```text
FEMs generally improve discrimination at all fine scales.
The benefit merely decays to zero above threshold.
Covariance geometry causes the discrimination benefit.
Eye-state conditioning does not help the optotype mechanism, unless run and validated.
Second-order covariance observers are a clean null, if p≫n-unreliable.
FEM correction reveals robust negative residual noise correlations, unless the covariance branch supports it.
Recorded V1 contains the same image-specific mimicry landscape as the model.
```

---

## Final stop condition

The figure is ready for manuscript review only if:

```text
Panel A generated
Panel B generated from canonical discrimination outputs
Panel C generated from canonical integration sweep
Panel D generated from observer_claim_validation.csv without overclaiming
Panel E generated with correct equations
Panel F generated from dedicated covariance outputs, or explicitly marked placeholder
Panel G generated from validated mimicry outputs or simplified schematic
QC report written
caption draft written
all exports saved as PDF, SVG, and PNG
```

If Panel F remains placeholder, final status should be:

```text
ready_except_covariance_branch_values
```

If Panel G is omitted or schematic only, final status can still be:

```text
ready_for_main_figure
```

provided Figure S4 contains the validated mimicry analysis or the manuscript does not require the detailed mimicry claim.



# Addendum: Canonical observer validation for the sign-changing Figure 4 effect

## Why this addendum is required

In the canonical validated mono Model A analysis, real FEMs produced a scale-dependent sign-changing effect. At the finest non-render-limit scale tested (LogMAR −0.35), real FEMs improved four-way orientation accuracy after 60-frame integration by 0.051 [0.030, 0.073] relative to stabilization. At coarser sizes, the effect reversed: real-minus-stabilized accuracy was −0.037 [−0.059, −0.014] at LogMAR −0.30 and −0.104 [−0.132, −0.075] at LogMAR −0.25.

This creates a new blocking requirement. The original “mean-only” / first-order decomposition was established on the keystone-mono pipeline, which is now non-canonical and did not reproduce the negative limb. Therefore, the current Panel D claim,

```text
A time-averaged mean-rate observer reproduces the benefit; the effect is first-order.
```

is not yet validated for the canonical sign-changing curve.

Panel D must not claim a first-order mechanism for the whole Figure 4B effect until the observer decomposition has been rerun on the canonical population and reproduces both:

1. the positive fine-scale benefit at LogMAR −0.35, and
2. the negative coarser-scale costs at LogMAR −0.30 and −0.25.

---

## Required new analysis task

Create or adapt:

```text
scripts/figure4/run_canonical_observer_decomposition.py
```

This script must use the exact canonical inputs from:

```text
outputs/figure4_reconciliation/canonical_discrimination/
```

and must match the population, traces, rendering path, decoder conventions, LogMAR grid, and windows used to generate:

```text
canonical_real_minus_stabilized.csv
integration_window_sweep.csv
```

Do not reuse keystone-mono observer-decomposition outputs unless they are shown to be bitwise or numerically identical to the canonical population and feature path.

---

## Primary question

Does a time-averaged mean-rate observer reproduce the **full canonical sign-changing real-FEM effect**?

Specifically, for window 60:

```text
LogMAR −0.35: positive benefit
LogMAR −0.30: negative cost
LogMAR −0.25: negative cost
```

The observer-decomposition result must be compared against the canonical full model / canonical decoder curve at the same LogMARs.

---

## Required observer families

At minimum, run:

```text
canonical_full_observer
mean_only_observer
temporal_trajectory_feature_observer
```

Optional, but only if already implemented cleanly:

```text
eye_state_conditioned_observer
nonlinear_observer
second_order_covariance_observer
```

However, optional observers must not delay the required mean-only validation.

---

## Required outputs

Create:

```text
outputs/figure4_reconciliation/canonical_discrimination/canonical_observer_decomposition.csv
```

Required columns:

```text
observer_name
population
feature_path
logmar
window
condition
accuracy
accuracy_ci_low
accuracy_ci_high
real_minus_stabilized_delta
delta_ci_low
delta_ci_high
n_units
n_traces
n_splits
status
notes
```

Create a contrast table:

```text
outputs/figure4_reconciliation/canonical_discrimination/canonical_observer_decomposition_contrasts.csv
```

Required columns:

```text
observer_name
logmar
window
canonical_delta
observer_delta
observer_minus_canonical_delta
canonical_delta_ci_low
canonical_delta_ci_high
observer_delta_ci_low
observer_delta_ci_high
sign_matches_canonical
magnitude_error
effect_class
status
```

Where:

```text
effect_class =
  positive_benefit
  near_zero
  negative_cost
  wide_ci
```

---

## Pass / fail criteria for Panel D

### `mean_only_reproduces_full_sign_changing_curve`

Panel D may claim a first-order mechanism for the full Figure 4B effect only if:

```text
At LogMAR −0.35:
  mean_only_delta > 0
  CI excludes or is clearly consistent with positive canonical effect
  sign_matches_canonical = true

At LogMAR −0.30:
  mean_only_delta < 0
  sign_matches_canonical = true

At LogMAR −0.25:
  mean_only_delta < 0
  sign_matches_canonical = true
```

and the magnitudes are reasonably close to the canonical curve. Suggested tolerance:

```text
abs(mean_only_delta − canonical_delta) <= 0.025
```

or a normalized criterion:

```text
magnitude_error <= 50% of abs(canonical_delta)
```

The readme should report both raw and normalized errors.

### `mean_only_reproduces_benefit_only`

If mean-only reproduces the +0.051 benefit at −0.35 but not the −0.30 / −0.25 costs, then Panel D must be weakened.

Allowed wording:

```text
The fine-scale benefit is captured by the time-averaged population mean, but the coarser-scale cost is not fully explained by the mean-only observer.
```

Disallowed wording:

```text
The sign-changing effect is first-order.
```

### `mean_only_fails_canonical_curve`

If mean-only fails to reproduce either the benefit or the cost on the canonical population, Panel D should not make a first-order mechanism claim. It should instead become a neutral observer-comparison or be moved to supplement.

Allowed wording:

```text
Observer decompositions did not establish a single first-order explanation for the canonical sign-changing curve.
```

### `temporal_features_null_validated`

The temporal-feature null may be included only if run on the canonical population.

Allowed wording:

```text
Explicit temporal-trajectory features did not improve orientation readout on the canonical population.
```

If the temporal observer is inherited only from keystone, mark:

```text
mismatched_population
```

and do not use it in Panel D.

---

## Updated Panel D requirements

Panel D should not be generated from the old `observer_claim_validation.csv` unless that table has been regenerated on the canonical population.

Required source:

```text
canonical_observer_decomposition.csv
canonical_observer_decomposition_contrasts.csv
observer_claim_validation.csv
```

The new `observer_claim_validation.csv` must include:

```text
observer_name
population
feature_path
canonical_population_match
tested_full_sign_changing_curve
status
allowed_panel_D_claim
disallowed_panel_D_claim
```

Expected statuses:

```text
mean_only_reproduces_full_sign_changing_curve
mean_only_reproduces_benefit_only
mean_only_fails_canonical_curve
temporal_features_null_validated
mismatched_population
not_run
unreliable_p_gt_gt_n
```

---

## Updated Panel D display options

### Preferred if mean-only passes

Plot canonical versus mean-only real-minus-stabilized deltas across LogMAR:

```text
x = LogMAR
y = real − stabilized accuracy
lines = canonical full observer, mean-only observer
horizontal line at zero
```

The panel should show that mean-only captures:

```text
+ effect at −0.35
− effect at −0.30
− effect at −0.25
```

Caption:

```text
The time-averaged population mean reproduced the canonical sign-changing effect, indicating that both the fine-scale benefit and coarser-scale cost arise from first-order changes in class means.
```

### Preferred if mean-only only reproduces benefit

Plot only the −0.35 observer comparison, and explicitly avoid claiming the cost.

Caption:

```text
The fine-scale benefit was captured by the time-averaged population mean. The coarser-scale cost was not fully explained by the mean-only observer and is not assigned to the first-order mechanism here.
```

### Preferred if mean-only fails

Replace Panel D with a status matrix and move mechanism decomposition to supplement.

Caption:

```text
Observer decompositions did not establish a single mechanism for the canonical sign-changing curve.
```

---

## Single-frame below-chance QC check

The canonical bundle reports a large single-frame deficit at LogMAR −0.35:

```text
window = 1:
real − stabilized = −0.451168 [−0.475597, −0.424615]
```

Before plotting this value in Panel C, confirm that the underlying real-FEM and stabilized single-frame accuracies are plausible.

Required QC:

```text
outputs/figure4_reconciliation/canonical_discrimination/single_frame_qc.csv
```

Required columns:

```text
logmar
window
real_accuracy
stabilized_accuracy
real_balanced_accuracy
stabilized_balanced_accuracy
chance_level
real_below_chance
stabilized_below_chance
label_balance_ok
confusion_matrix_status
qc_status
notes
```

For four-way orientation discrimination:

```text
chance_level = 0.25
```

### Stop / warning rule

If:

```text
real_accuracy < 0.25
```

or:

```text
real_balanced_accuracy < 0.25
```

then flag:

```text
single_frame_below_chance_warning
```

Do not interpret the single-frame deficit as a biological or model deficit until label ordering, decoder target labels, and confusion matrices are checked.

Allowed wording if below chance is detected:

```text
At a single frame, the canonical decoder showed a large real-FEM deficit; because this approached or fell below chance, we treat the single-frame endpoint as a diagnostic of insufficient single-sample information rather than a mechanistic claim.
```

Allowed wording if above chance:

```text
At a single frame, real-FEM samples were less informative than stabilized samples but remained above chance, supporting the interpretation that the FEM advantage emerges only after integration.
```

---

## Reconciliation-readme terminal label check

Confirm that:

```text
outputs/figure4_reconciliation/final_reconciliation_readme.md
```

or the corresponding reconciliation readme contains a terminal label specifying why the keystone pipeline diverged.

Expected label examples:

```text
effect_size_reconciled_stale_cache
effect_size_reconciled_feature_conversion_difference
effect_size_reconciled_accuracy_column_error
effect_size_reconciled_other_documented
```

If the readme only selects the neurometric pipeline without explaining why the keystone pipeline diverged, add a section:

```text
## Cause of keystone / neurometric divergence
```

This section should specifically assess:

```text
build_keystone_mono_cache_from_temporal_rates.py
converted-cache feature path
feature normalization
windowing
absolute accuracy mismatch
delta mismatch
missing negative limb
```

The readme should state whether the converted-cache path is confirmed, likely, possible, or ruled out as the source.

---

## Prose-draft conflict warning

The current prose draft must be updated because it still says:

```text
we frame the reliable effect as a fine-scale benefit that decays above threshold, rather than as a sign-changing crossover
```

This is now inconsistent with the canonical results.

Replace with:

```text
In the canonical validated mono Model A analysis, real FEMs produced a scale-dependent sign-changing effect. At the finest non-render-limit scale tested (LogMAR −0.35), real FEMs improved four-way orientation accuracy after 60-frame integration by 0.051 [0.030, 0.073] relative to stabilization. At coarser sizes, the effect reversed: real-minus-stabilized accuracy was −0.037 [−0.059, −0.014] at LogMAR −0.30 and −0.104 [−0.132, −0.075] at LogMAR −0.25.
```

Do not circulate the old prose and the new Figure 4 handoff together.

---

## Updated figure readiness status

The figure is not ready for manuscript review until:

```text
canonical effect-size reconciliation complete
Panel B/C generated from canonical neurometric pipeline
single-frame below-chance QC complete
observer decomposition rerun on canonical population
mean-only result adjudicated for full sign-changing curve
Panel D updated according to observer-decomposition outcome
covariance branch values inserted or Panel F marked placeholder
validated mimicry outputs passed QC
```

If all are complete except covariance branch values:

```text
status = ready_except_covariance_branch_values
```

If observer decomposition is not rerun on the canonical population:

```text
status = not_ready_panel_D_blocking
```
