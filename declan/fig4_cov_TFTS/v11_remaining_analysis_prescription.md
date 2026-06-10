# V11 Remaining Analysis Prescription

## Purpose

This note updates the remaining analysis plan after the current v11 manuscript framing:

```text
Fixational eye movements reveal a compact reafferent geometry underlying V1 shared variability
```

The paper no longer needs to argue that V1 contains a coordinate or metric for retinal position. The central question is narrower and stronger:

> Does the reafferent retinal-translation geometry make a nontrivial prediction about recorded V1, and does that prediction bear out?

Current answer:

> Yes, at the covariance/source-geometry level. Fitted-twin finite-difference retinal-translation covariances predict a reliable component of recorded FEM-linked covariance in matched V1 units, and the prediction is retained when the source is forced through the compact k=10 tangent geometry.

What should be avoided:

- hidden coordinate system;
- readable compact displacement coordinate;
- V1 encodes absolute eye position;
- FEMs are behaviorally optimal;
- all V1 shared variability is explained by the compact tangent geometry.

The remaining work should make the successful prediction quantitatively interpretable, reviewer-proof, and cleanly bounded.

## Current Manuscript Anchor

The current v11 Figure 4 story is:

1. Recorded FEM-linked covariance is low-dimensional after eye-position conditioning.
2. Digital-twin local retinal-translation tangents are image/history-specific.
3. Pooled tangents are compact relative to unit-shuffle controls.
4. Compact tangent basis generalizes across images.
5. Compact basis captures a meaningful fraction of FEM-related local displacement Fisher sensitivity.
6. Finite-difference translation covariance predicts recorded FEM covariance; compact k=10 source retains this prediction.

This is already the right spine. The remaining analyses should not expand the paper into a full active-sensing optimality/readout paper.

## Tier 1: Submission-Critical Analyses

These are the analyses still needed before the Figure 4 claim is quantitatively locked.

### 1. Variability Budget Denominators

#### Motivation

The manuscript currently reports strong covariance-capture effects but still contains TODOs like:

```text
X% of FEM covariance
X% of reliable shared covariance
X% of total covariance
reliability-adjusted X% of explainable FEM covariance
orthogonal covariance partition
```

Without these denominators, readers cannot tell how large the compact translation-predicted component is biologically.

#### Required denominators

Compute and report, per session and in session-level summaries:

```text
matched conservative projected PSD FEM target trace
matched full PSD FEM target trace
matched raw FEM target trace
total reliable shared covariance trace
total trial-to-trial covariance trace, optional/supplement
split-half reliability ceiling for the recorded FEM target
```

#### Required fractions

For both full finite-difference source and compact k=10 source:

```text
absolute captured trace
fraction of conservative projected FEM target
fraction of full matched FEM covariance
fraction of reliable shared covariance
fraction of total trial-to-trial covariance, optional
reliability-ceiling-normalized capture
RF/readout-null-adjusted fraction
```

#### Important details

- Report both session-unweighted and trace-weighted summaries.
- Keep raw and PSD targets side by side.
- If PSD is the headline, explicitly report negative eigenvalue mass and why PSD is used.
- Do not collapse all denominators into one number; the paper needs a small budget table or inset.

#### Output files

```text
variance_budget_denominators.csv
variance_budget_capture_fractions.csv
variance_budget_reliability_ceiling.csv
variance_budget_summary.json
figures/variance_budget_compact_translation_component.png
```

#### Main-text use

This should replace the current TODO sentence in the "Accounting for the FEM-linked covariance" section.

Safe wording:

> The compact translation-predicted component accounted for X of the conservative non-global FEM target, Y of the full matched FEM covariance, and Z of the reliable shared covariance denominator, with ceiling-normalized capture of C.

### 2. Orthogonal Covariance Partition

#### Motivation

The manuscript currently mentions global-rate, target-PC1, compact tangent, and residual components, but the order of removing components can change apparent fractions. A reviewer may ask whether compact capture remains meaningful after global/state components are partitioned fairly.

#### Required analysis

Construct an order-independent or explicitly order-averaged partition among:

```text
global-rate mode
target PC1 mode
compact translation source subspace
remaining finite-difference source subspace
residual FEM covariance target
```

Two acceptable implementations:

1. **Orthogonalized fixed-order partition with multiple order checks**
   - Use a declared primary order.
   - Repeat over alternative orders.
   - Report range/order sensitivity.

2. **Shapley-style order-averaged partition**
   - Compute incremental captured trace for all or sampled permutations of component groups.
   - Report mean contribution and CI.

#### Outputs

```text
orthogonal_covariance_partition.csv
orthogonal_covariance_partition_order_sensitivity.csv
figures/orthogonal_covariance_partition.png
```

#### Main-text use

This does not need to be a main panel unless it is visually clean. It should be available as a supplement or audit table.

### 3. RF/Readout-Preserving Null Integration

#### Motivation

The strongest current recorded-covariance statement uses the RF/readout-preserving null, but v11 Figure 4F caption still foregrounds unit-shuffle in the visible y-axis language. The main figure/caption should make the RF/readout-preserving null visible because it is the reviewer-facing specificity control.

#### Required actions

- Ensure Figure 4F or its inset shows both:
  - unit-shuffle reference;
  - RF/readout-preserving null reference.
- Report the headline conservative row:

```text
PSD target
global-rate + target-PC1 removed
source eigenspace k=2
full finite-difference source
compact k=10 restricted source
RF/readout-preserving null excess
raw target sign check
```

#### Existing target numbers to verify

Current notes report approximately:

```text
full source capture = 0.216
full RF/readout-null excess = +0.158 [0.125, 0.193]
compact source capture = 0.217
compact RF/readout-null excess = +0.161 [0.128, 0.196]
compact/full ratio about 1.01
```

Do not hardcode; regenerate or load from audited production outputs.

#### Outputs

```text
rf_readout_null_headline.csv
rf_readout_null_session_effects.csv
figures/covariance_closure_unit_shuffle_and_rf_null.png
```

### 4. Figure 4 Bookkeeping and Window Harmonization

#### Motivation

v11 methods notes contain different session counts/windows for different panels:

```text
Figure 4A anchor: 8 sessions, 33.333 ms window
Figure 4F closure: 24 sessions, target window index 1
max samples per session: 512
raw/PSD target variants
```

This may be correct, but it needs to be explicit.

#### Required output

Create a small bookkeeping table:

```text
figure4_panel_bookkeeping.csv
```

Columns:

```text
panel
analysis_name
sessions
units
stimulus regime
response window
eye window
latency convention
sample cap
target/source cache
projection controls
included/excluded criteria
```

#### Main-text use

This can be a Methods table or supplement. The caption should not imply all panels use the same sessions/units if they do not.

## Tier 2: Mechanistic Support Analyses

These analyses strengthen the interpretation but should not replace the covariance-prediction spine.

### 5. Curvature / Amplitude Law

#### Motivation

This is the highest-value remaining mechanistic test. It directly supports the tempered interpretation:

> The compact geometry is a local first-order reafferent approximation, not a global coordinate/readout.

It also addresses the v11 residual-covariance language:

```text
Some residual may reflect finite-displacement curvature, because the recorded FEM cloud samples a nonlinear response surface rather than a single infinitesimal derivative.
```

#### Required analysis

For fitted-twin samples or controlled translated responses, bin by retinal displacement amplitude:

```text
drift-scale
intermediate
microsaccade-scale / larger finite offsets
```

For each amplitude bin:

```text
actual finite response change: Delta r_actual
linear tangent prediction: J delta_e
compact linear prediction: U_k U_k.T J delta_e
residual: Delta r_actual - J delta_e
```

Metrics:

```text
pointwise R2 / prediction quality
tangent-subspace capture fraction
compact source covariance capture
residual norm fraction
residual enrichment in curvature/Hessian directions if available
```

Prediction:

```text
linear tangent prediction is strongest at drift scale
prediction declines with displacement amplitude
compact covariance geometry remains more stable than pointwise Taylor prediction
optional Hessian/curvature terms recover residual at larger displacement scales
```

#### Required controls

- Unit-shuffle tangent basis.
- Random k-dimensional basis.
- RF/readout-preserving null if feasible.
- Separate pointwise prediction from covariance capture; they answer different questions.

#### Outputs

```text
curvature_amplitude_law_metrics.csv
curvature_amplitude_law_covariance_capture.csv
curvature_residual_enrichment.csv
figures/curvature_amplitude_law.png
```

#### Figure use

This can be a supplement or a small main-panel/inset if the curve is clean. It is more valuable to the paper than another decoder panel because it clarifies what the compact geometry is.

Safe wording:

> Tangent predictions were most accurate at small drift-scale displacements and degraded for larger offsets, consistent with a local first-order approximation to a curved, image-conditioned response surface.

### 6. Residual FEM Covariance Characterization

#### Motivation

After the compact translation-predicted component is accounted for, the remaining FEM-linked covariance should not be called irreducible noise. It may contain finite-displacement curvature, non-translation oculomotor variables, velocity/history, or state.

#### Required analysis

Using the residual target after removing compact translation-predicted components, test enrichment for:

```text
finite-displacement curvature residuals
eye velocity
microsaccade indicator or event phase
retinal displacement magnitude
temporal eye-history terms
global-rate/state modes
```

This can be lightweight. The goal is not to fully explain residual covariance, but to show that residual structure has plausible sources.

#### Outputs

```text
residual_fem_covariance_diagnostics.csv
figures/residual_fem_covariance_diagnostics.png
```

#### Main-text use

Likely one paragraph in Discussion or supplement. Do not overbuild unless results are clean.

## Tier 3: Diagnostic / Boundary Analyses

These are useful, but should not be load-bearing.

### 7. Relative Displacement Decoding Audit

#### Current status

Preliminary production result is mixed/negative for the compact subspace:

```text
full_population R2_mean under global_rate+target_pc1: +0.0143
global_top_pc_modes: +0.0542
compact k10: -0.0210
compact loses to full population in 0/13 sessions
compact loses to orthogonal complement on average
compact beats random/unit-shuffle/RF-null subspaces in most sessions
0 split leakage failures
```

Interpretation if code review passes:

> Recorded V1 contains a weak same-image relative-displacement signal, but this signal is not primarily carried by the compact twin-tangent subspace under conservative controls.

#### Required cleanup

Before using this result:

1. Merge split production outputs into a first-class combined artifact.
2. Code-review context labels:
   - Production used `time_bin`.
   - Confirm this really defines matched image/time/history condition.
3. Audit projection versus feature-space logic:
   - Explain why `global_top_pc_modes` can decode after `global_rate+target_pc1` projection.
   - Ensure removed modes are not reintroduced.
4. Add dimension-matched orthogonal-subspace control if not already present.
5. Verify R2 baseline and test-fold centering.
6. Keep train/test condition-disjoint leakage audit.

#### Outputs

```text
relative_displacement_decoding_combined/
  decoder_metrics.csv
  feature_space_comparison.csv
  decoder_nulls.csv
  split_leakage_audit.csv
  audit.json
  README.md
```

#### Manuscript use

Use as a supplement or a brief caveat only. Do not promote to main Figure 4.

Safe wording:

> A stricter same-image relative-displacement decoder revealed weak recorded displacement information, but this signal was not preferentially carried by the compact tangent subspace after conservative global/PC controls. We therefore interpret the compact geometry as covariance-predictive rather than as a demonstrated trial-wise displacement readout.

### 8. Metric Validation Result

#### Current status

Metric validation was mixed:

```text
local compact metrics rank 2: pass
median condition: good
norm scaling: strong
per-object squared-distance scaling: strong
pooled quadratic prediction: weak
opposite-shift test: weak/mixed
coordinate recovery: weak
diagonal/arbitrary translations not available in current cache
```

#### Interpretation

This is exactly why the paper should avoid coordinate-system language.

Use as an internal audit or supplemental boundary:

> Local charts have metric-like behavior within objects, but the compact geometry does not support a stable pooled coordinate metric or robust coordinate recovery.

Do not spend more time here unless the paper returns to coordinate language.

### 9. Windowed / Siamese MLP Decoder

#### Status

Potential high-upside extension inspired by Tejas's windowed eye-position MLP, but not submission-critical.

#### Recommendation

Do not prioritize above the curvature law or variance budget.

If implemented, use a strict pairwise formulation:

```text
encoder(response window a, same condition context)
encoder(response window b, same condition context)
prediction = head(encoder_a - encoder_b)
target = eye_a - eye_b
```

Required controls:

```text
context-only decoder
eye-prior / autoregressive baseline
neural-shuffle decoder
compact vs orthogonal vs RF/readout-null features
trial-disjoint split
swap antisymmetry check
```

Interpret only as a readout bridge if it beats these controls.

## Tier 4: Figure Polish and Reviewer-Proofing

These are not conceptual analyses, but they matter for clarity.

### 10. Panel B: Local Translation Charts

Implement or preserve the redesigned local tangent glyph panel:

```text
base response points r0(I)
projected local b_x(I), b_y(I) arrows/crosses
20-40 representative objects
optional cos(b_x, b_y) inset
```

Goal:

> Show image-specific local translation charts at a glance.

### 11. Panel C: Null Spectrum Display

If actual unit-shuffle null spectra are available:

```text
show gray null band or faint null curves
```

If only PR reference is available:

```text
label honestly as Unit-shuffle PR reference
```

Avoid a schematic-looking diagonal unless it is clearly labeled.

### 12. Panel D: Cross-Image Generalization Labeling

Use:

```text
Cross-image generalization
Held-out translation-tangent variance captured
```

Avoid:

```text
held-out translation variance
```

because the panel is about tangent variance, not finite response variance.

### 13. Panel E: Fisher Sensitivity Guardrails

Keep the claim narrow:

> The compact tangent basis captures part of local spatial-displacement Fisher sensitivity induced by real FEM histories.

Avoid:

> FEMs are optimal
> animal uses this information behaviorally
> compact basis captures all active-sensing information
```

If possible, include:

```text
orthogonal complement partition
unit-shuffle and random nulls
Poisson/Fisher assumption note
```

### 14. Natural-Image Regime Validation

This is later validation / reviewer response, not a blocker.

Run only if ecological active-sensing language becomes central:

```text
compact tangent spectrum outside FixRSVP
cross-image generalization outside FixRSVP
optional tangent-subspace sensitivity outside FixRSVP
```

## Recommended Work Order

1. **Variance budget denominators and reliability ceiling.**
2. **RF/readout-preserving null integration into the visible Figure 4F/caption.**
3. **Orthogonal covariance partition.**
4. **Figure 4 bookkeeping table for sessions/windows/caches.**
5. **Curvature/amplitude law.**
6. **Relative displacement decoder code review and combined artifact.**
7. **Residual FEM covariance diagnostics.**
8. **Panel B/C/D/E figure polish.**
9. Optional: natural-image regime validation.
10. Optional: windowed/Siamese MLP decoder.

## Stop Rules

Do not chase a stronger coordinate/readout story unless:

- metric validation becomes positive under held-out directions;
- relative displacement decoding is compact-specific under conservative controls;
- decoder controls rule out context priors, global modes, and trajectory priors.

Do not chase active-sensing optimality unless:

- Figure 5 / model information-efficiency results survive matched-motion controls;
- recorded-data bridge tests align with model information gain;
- behavioral or task evidence exists.

## Final Target Claim

The final paper should be able to say:

> A compact retinal-translation geometry derived from an image-computable V1 model predicts a reliable component of FEM-linked shared covariance in recorded foveal V1. This geometry is image-specific but image-generalizing, captures part of local FEM-related displacement sensitivity, and remains predictive after RF/readout-preserving controls. It is best interpreted as a covariance-predictive footprint of retinal reafference, not as an explicit coordinate system or trial-wise position code.

