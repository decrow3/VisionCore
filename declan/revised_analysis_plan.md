# FEM Population Coding in the V1 Digital Twin: Revised Analysis Plan

## Status as of April 2026

---

## 1. What the Model Is

The digital twin (learned_resnet_none_convgru_gaussian, epoch 147, val BPS = 0.5702) has the following architecture:

```
Stimulus (1, 1, T, H, W)
  → LearnedTemporalConv frontend (kernel_size=16, 4 channels)   T → T-15
  → 3D ResNet convnet (256 channels)                             T-15 → ~T/4
  → ConvGRU (128 hidden channels, 8 recurrent steps)             T preserved
  → Per-neuron readout: 1×1 feature conv × Gaussian spatial mask → scalar rate
```

**Temporal processing is real.** The GRU passthrough test shows median R² = −50.5 between dynamic (real history) and static (current-frame-only) predictions. The GRU processes 8 temporal steps with genuine recurrence and substantially reshapes the convnet representation (RSA = 0.82). The model is not a spatial encoder — history strongly affects the output.

**Population correlations are purely reafferent.** All neurons read from a single shared feature map produced by the shared core (frontend → convnet → GRU). There is no neuron-neuron recurrence, no independent noise, no shared modulatory state. Every correlation in the model arises because neurons at different spatial positions sample a shared temporally evolving representation driven by the same eye movement over the same image. The model isolates the retinal-reafference component of V1 population correlations.

**Training regime constrains temporal horizon.** The model was trained on short windows (~25–32 frames) with fresh GRU hidden state per window. Each analysis time point in the current pipeline is processed independently — the GRU builds its representation from scratch in 8 steps per window. There is no inter-window temporal memory.

---

## 2. What Is Established

### 2.1 FEM improves orientation discrimination in the hyperacuity regime

The D1 decoder (time-averaged rate, no PCA, no temporal features) shows a clear crossover:

| LogMAR | D1 stabilized (W=60) | D1 real FEM (W=60) | FEM advantage |
|--------|----------------------|--------------------|---------------|
| −0.20  | 0.769                | 0.646              | −0.123 (FEM hurts) |
| −0.25  | 0.773                | 0.654              | −0.119 (FEM hurts) |
| −0.30  | 0.875                | 0.828              | −0.047 (FEM hurts) |
| −0.35  | 0.838                | 0.891              | **+0.053 (FEM helps)** |
| −0.40  | 0.841                | 0.894              | **+0.053 (FEM helps)** |
| −0.45  | 0.840                | 0.894              | **+0.054 (FEM helps)** |
| −0.50  | 0.841                | 0.895              | **+0.054 (FEM helps)** |

The crossover occurs at approximately LogMAR −0.32. Below this, FEM with temporal accumulation exceeds stabilized by ~5 points. The benefit is through the simplest possible readout — no temporal features (D3 ≈ chance), no eye-state conditioning (D2a ≈ D1), no nonlinear decoder (MLP hurts). Both conditions collapse to chance by −0.55 (rendering/discretization limit).

### 2.2 The response manifold is smooth, image-specific, and locally linear

Within-image displacement decoding achieves R² ≈ 0.998 from scalar rate differences across a grid of 121 positions spanning ±0.05°. Cross-image displacement decoding yields R² ≈ −1.3 (anti-generalizes). The displacement code is high-fidelity within each image but the direction in population space that encodes displacement depends entirely on image content.

### 2.3 FEM covariance is rank-2 and image-specific

Top 2 PCs capture ~97% of FEM-induced variance, consistent with 2-DOF translation. Cross-stimulus PCA alignment is 0.37 — the translation-induced covariance directions differ across images.

### 2.4 Spatial information increases under FEM

Population SSI on the E-optotype at LogMAR −0.20 shows cumulative bits: 0.543 (real) vs 0.463 (stabilized), a +17% gain. This is consistent across all four orientations, indicating position diversity rather than orientation-specific information per se.

### 2.5 Temporal trajectory features do not help orientation decoding (under current pipeline)

Model C (temporal residual PCA) ≈ Model A (time-averaged rate) across all tested conditions. D3 (supervised mean-trajectory subspace) is near chance. However, these were tested with independently processed windows (fresh GRU state), not with continuous temporal integration. The null is established for the current pipeline but may not hold for the continuous forward pass.

### 2.6 Velocity and transformation variables are not decodable from scalar or spatial-moment representations

Δz ≈ 0 (PCA latent dynamics → velocity). ΔCoM R² < 0 (spatial map dynamics → velocity). These were tested under independent-window processing and likely reflect the lack of inter-window temporal continuity rather than a fundamental absence of transformation encoding in the model.

---

## 3. The Central Hypothesis

**FEM-driven reafferent correlations transition from information-limiting to information-expanding at the resolution limit.**

Above threshold (~LogMAR −0.20), the population can resolve orientation from a single fixation position. FEM-induced correlations add variability along the orientation signal direction, degrading discrimination. Below threshold (~LogMAR −0.40), no single position provides sufficient orientation information. FEM-induced correlations spread the response across positions that sample different parts of the E, spanning new dimensions that carry orientation evidence unavailable from any single position.

The crossover at −0.32 marks the transition. The geometric mechanism is a change in the alignment between the FEM covariance subspace and the orientation signal subspace.

**Status: the crossover is established; the geometric mechanism is hypothesized and directly testable.**

---

## 4. Priority Experiments

### Priority 1: Per-orientation FEM subspace geometry and second-order decoding

**Question:** Does the FEM covariance subspace rotate with orientation, and does this rotation become the primary carrier of orientation information at hyperacuity?

**Motivation:** The displacement code is image-specific (cross-image R² = −1.3, cross-stimulus PCA alignment = 0.37). The four E orientations are four different images. Therefore, the direction in population space that encodes "the image shifted right" almost certainly differs across orientations. Averaging C_FEM across orientations (as originally planned) would collapse exactly the structure that makes FEM correlations potentially information-expanding. The right procedure is per-orientation.

If the FEM subspace rotates with orientation, that rotation *is* orientation information — encoded in second-order statistics (covariance geometry) rather than first-order statistics (mean rate). This leads to a strong prediction about where orientation information lives at each regime.

**Method:**

At LogMAR −0.20 (FEM hurts) and −0.40 (FEM helps), using cached rates from the D1 analysis:

**Step 1 — Per-orientation FEM subspaces:**
For each orientation k ∈ {1, 2, 3, 4}, compute C_FEM^k = Cov_traces[r̄_k], the covariance of the time-averaged rate vector across eye traces. Extract U_FEM^k = top-2 eigenvectors of C_FEM^k.

**Step 2 — Subspace rotation check:**
Compute all 6 pairwise principal angles between U_FEM^1 … U_FEM^4. Report the mean subspace overlap (squared cosines). If FEM subspaces are distinct across orientations, orientation is encoded in covariance geometry. If they are aligned, orientation information from FEM structure is limited.

**Step 3 — Signal alignment per orientation:**
For each orientation k, compute the orientation signal covariance C_signal = (1/K) Σ_k (μ_k − μ)(μ_k − μ)ᵀ (pooled across orientations as before). Compute the per-orientation alignment fraction α^k = tr(U_FEM^k ᵀ C_signal U_FEM^k) / tr(C_signal). This measures how much each orientation's FEM noise falls along the global signal directions. Report mean and spread across orientations, at both LogMARs.

**Step 4 — Second-order decoder:**
Classify orientation using covariance geometry alone. For each held-out trial (identity unknown), project its time-averaged rate onto *all four* U_FEM^k and assign to the orientation whose subspace captures the most variance (smallest residual ‖δr − U_FEM^k (U_FEM^k)ᵀ δr‖²), where δr = r̄ − μ̄ and μ̄ is the grand mean across all orientations and traces. Compare covariance decoder accuracy to D1 (mean rate) at both LogMARs.

**Implementation note — mean subtraction is required.** Without subtracting μ̄ first, the residual ‖r̄ − U_FEM^k (U_FEM^k)ᵀ r̄‖² is dominated by how well each FEM subspace captures the *mean response* μ_k, not the FEM variability pattern. Since mean responses differ across orientations, the decoder would be doing first-order (D1-style) classification through the back door — whichever orientation's FEM subspace happens to align with that orientation's mean direction wins. Subtracting the grand mean before projecting isolates the second-order signal: the decoder operates on the deviation from the population average and classifies by which subspace best captures the *shape of variability around the mean*, not the mean itself. A positive Step 4 result without this correction would be uninterpretable.

Note: Step 4 is a decoder — the orientation identity is unknown at test time. The within-orientation projection in Priority 2 is a separate operation (causal ablation, orientation known). Do not conflate them in implementation.

Step 4 is potentially confounded even with mean subtraction if the FEM subspace directions happen to correlate with residual mean-response structure within δr. Step 5 (combined decoder) sidesteps this entirely and is the load-bearing test; Step 4 is the more interpretable companion.

**Step 5 — Combined decoder (D1 + covariance geometry):**
Concatenate each trial's mean rate vector with its 4×2 = 8 FEM-subspace projection magnitudes (‖(U_FEM^k)ᵀ r̄‖ for each k), and rerun the D1 classifier on this combined feature vector. Test at −0.40.

This is the critical test for whether FEM-induced structure carries orientation information *beyond* what the time-averaged rate already provides. D3 and Model C showed no temporal trajectory benefit over D1 — but that was testing temporal sequence features. This tests second-order statistics (covariance geometry), which D3/C never touched. If the combined decoder exceeds D1 alone at −0.40, it is the first positive evidence of a temporal correlation code — just not in the form the original plan expected.

**Predictions:**
- U_FEM^k rotates with orientation (pairwise overlaps < 1): the FEM subspace is an orientation fingerprint.
- At −0.20: D1 >> covariance decoder. Mean-rate signal is intact; FEM subspace rotation adds noise on top of a strong first-order signal. Combined decoder ≈ D1 (covariance features don't add).
- At −0.40: D1 ≈ or < covariance decoder. Mean rates compress toward chance as the E is barely resolved, but FEM subspaces still rotate because image-specificity persists below the resolution limit. Orientation information has migrated from mean rate to covariance geometry. Combined decoder > D1 (covariance features genuinely add).
- α^k is high at −0.20 (FEM noise aligned with signal, information-limiting) and lower at −0.40 (FEM noise less aligned with compressed signal, consistent with information-expanding).

**Computational cost:** Minutes. Uses cached rate matrices; all steps are linear algebra.

**What it would show:**
- Subspace rotation confirmed + covariance decoder > D1 at −0.40 + combined decoder > D1: the strongest possible result. Orientation information migrates from first-order (mean rate) to second-order (FEM covariance geometry) at hyperacuity. FEM correlations don't just stop limiting information — they become an independent information carrier.
- Subspace rotation confirmed + combined decoder > D1 at −0.40 but covariance decoder alone is weak: FEM geometry adds information beyond D1 but can't carry the signal on its own. Information is distributed across first- and second-order statistics.
- Subspace rotation confirmed + covariance decoder ≈ D1 at both LogMARs: FEM subspaces are orientation-specific but migration is not complete. Intermediate result.
- No subspace rotation: orientation-invariant FEM noise. Falls back to the alignment-fraction story.
- α(−0.20) > α(−0.40) regardless: original geometric mechanism for the crossover holds.

**RESULTS — April 2026** (script: `declan/fem_covariance_geometry.py`, M=471 trials/ori, N=756)

| Metric | −0.20 real | −0.40 real | −0.20 stab | −0.40 stab |
|--------|-----------|-----------|-----------|-----------|
| Off-diagonal overlap | 0.9998 | 0.9995 | 0.9395 | 0.9951 |
| Mean α | 0.720 | 0.558 | 0.607 | 0.665 |
| D1 accuracy | 0.746 | 0.936 | 0.766 | 0.840 |
| Covariance decoder | 0.263 | 0.333 | 0.288 | 0.317 |
| Combined − D1 | +0.001 | +0.000 | +0.000 | +0.001 |

**Subspace rotation: NULL.** Real-FEM subspaces are orientation-invariant (overlap ~1.0) at both LogMARs — more uniform than the stabilized baseline at −0.20. Covariance decoder is near chance at both LogMARs. Combined decoder adds nothing over D1.

**Why the rotation is absent:** FEM is 2D image translation. The FEM subspace represents displacement directions in response space — a property of the spatial encoder's RF structure, not of stimulus identity. Translating a 0° E and a 90° E by the same ±0.05° displacements produces nearly identical directions in population space because the translational response is dominated by the shared letter envelope, not by the orientation-specific gap positions. This is consistent with cross-stimulus PCA alignment being 0.37 for different natural images (genuinely different spatial structure) but ~1.0 for four rotations of the same letter at the same scale. The image-specificity of the displacement code does not extend to stimuli that share global spatial structure.

**Alignment transition: CONFIRMED — with an important caveat.** α is higher at −0.20 (0.720) than at −0.40 (0.558) in the real condition. The stabilized condition shows the *opposite* ordering (0.607 at −0.20, 0.665 at −0.40). This reversal is striking: under real FEM, the noise becomes *less* aligned with the signal at hyperacuity; under stabilization, it becomes *more* aligned. That is not just "FEM noise decreases at hyperacuity" — it is a qualitative reversal of the regime dependence between conditions, which demands an explanation and is consistent with the crossover.

**Caveat — signal geometry vs noise geometry:** The α transition has two possible interpretations. (1) The FEM noise subspace stays approximately fixed while the orientation signal subspace contracts at hyperacuity and moves away from the displacement directions — producing lower α at −0.40 because the *signal* moved, not the *noise*. (2) The FEM noise subspace shifts relative to the signal as the regime changes — the *noise* moves. These have different mechanistic implications. At −0.20, the E is large enough that its orientation-specific features (gap positions) span many RF positions, overlapping with the global envelope that drives FEM variability. At −0.40, the E is so small that only a few central RFs see the orientation-specific features, while the displacement directions still span the full envelope. So the α decrease may partly or entirely reflect the signal subspace shrinking and shifting, not the noise subspace changing. **Computing the signal covariance eigenspectrum (C_signal) at both LogMARs would disambiguate** — if the orientation signal directions rotate or contract between −0.20 and −0.40, that is part of the explanation for α.

**What this rules out:** Information migration from mean rate to covariance geometry does not occur in this model. FEM covariance does not carry orientation identity at either LogMAR.

**What this confirms and what remains open:** The crossover mechanism is spatial sampling via mean-rate accumulation. The α pattern and its reversal between real and stabilized conditions are consistent with the alignment story, but the causal direction (noise moving, signal moving, or both) is not yet established. Priority 2 (global FEM subspace ablation) is the next step and takes minutes. The C_signal eigenspectrum comparison is an additional cheap diagnostic that should accompany Priority 2.

### Priority 2: Global FEM subspace intervention (causal test, revised)

**Revision note:** The original Priority 2 used per-orientation U_FEM^k for the ablation. Priority 1 established that U_FEM^k is orientation-invariant (overlap ~1.0), so per-orientation ablation collapses to a global ablation. The revised test uses the shared FEM subspace U_FEM (fit on all orientations pooled, or equivalently any single orientation's U_FEM^k since they are nearly identical).

**Question:** Does removing the shared FEM covariance subspace causally change D1 accuracy, and in the predicted direction at each regime?

**Method:**

1. Fit U_FEM (top-2 eigenvectors of the pooled within-orientation covariance C_FEM) from training traces.

2. Project out U_FEM from each trial's time-averaged rate: r̄_corrected = r̄ − U_FEM U_FEM^T r̄.

3. Rerun D1 decoding on corrected rates at −0.20 and −0.40.

**Predictions (from α pattern in Priority 1):**
- At −0.20: α = 0.720 (FEM subspace strongly aligned with signal). Removing it should *improve* D1 — the FEM variability falls in the signal direction and is contaminating the classifier.
- At −0.40: α = 0.558 (less aligned). Effect should be smaller. The qualitative prediction (removal hurts less or not at all) is robust; the sign depends on whether FEM variability at −0.40 helps or merely doesn't hurt.

The α numbers make a quantitative prediction about the *size* of the D1 accuracy change: if α tracks how much the FEM subspace contaminates the classifier, the improvement at −0.20 should be larger than any change at −0.40. If the D1 accuracy change tracks α in the right direction, the geometry is causally connected to the behavior.

**Additional diagnostic — C_signal eigenspectrum at both LogMARs:**
Compute the signal covariance C_signal at −0.20 and −0.40 and compare the top eigenvectors and eigenvalues. This disambiguates the α caveat from Priority 1: if the signal subspace contracts or rotates substantially between LogMARs, the α decrease reflects signal geometry changing, not FEM noise geometry. If the signal subspace is stable across LogMARs but α still decreases, the FEM noise direction is genuinely moving relative to the signal. This is a cheap computation using the same cached rates.

**Computational cost:** Minutes. Same cached rates, one projection step before decoding, one eigendecomposition for the diagnostic.

**Note on existing result:** The existing `intervention.pkl` in `scripts/temporal_decoding/data/results/` ran this analysis at a single LogMAR using the pooled C_FEM. Check whether that result already covers −0.20 and −0.40 before rerunning.

**RESULTS — April 2026** (script: `declan/fem_global_intervention.py`, `--rate_file_tag allhires_fresh`, M=471, N=756)

Real condition:

| LogMAR | α | D1 original | D1 cleaned | Δ |
|--------|---|-------------|------------|---|
| −0.20 | 0.689 | 0.747 | 0.773 | **+0.027** |
| −0.40 | 0.559 | 0.936 | 0.936 | **+0.000** |

Stabilized control:

| LogMAR | α | D1 original | D1 cleaned | Δ |
|--------|---|-------------|------------|---|
| −0.20 | 0.052 | 0.774 | 0.803 | **+0.029** |
| −0.40 | 0.666 | 0.840 | 0.855 | **+0.015** |

**Finding:** The real-condition ordering is as predicted from α: removing the pooled subspace helps at −0.20 and is null at −0.40. But the stabilized control matters: the same intervention also improves decoding there, by a similar amount at −0.20 and modestly at −0.40. So this ablation is not isolating a dynamic FEM-specific nuisance mode. It is removing a broader low-rank translation/displacement nuisance present even when traces are "stabilized" to each trial's mean position.

**Why the control changes the interpretation:** In this pipeline, `stabilized` does not collapse all trace-to-trace variability to zero. It holds each trial at its own mean eye position, so across trials there is still a low-rank retinal translation distribution, and the readout remains position-sensitive. Projecting out the pooled within-orientation covariance therefore removes a generic positional nuisance subspace in both conditions, not a variance component unique to dynamic FEMs.

**C_signal diagnostic:** Top signal eigenvalues are [3.9e-05, 6.0e-06] at −0.20 and [2.24e-04, 1.50e-05] at −0.40 in the real condition. At −0.40 the class means are more separated overall, but their overlap with the pooled nuisance/translation subspace is lower (smaller α and null Δ). This still supports the "signal moved" interpretation of the α transition, but not a strong claim that dynamic FEM covariance is itself the causal limiter.

**Revised conclusion:** Priority 2 partially supports the geometry story but does not cleanly prove that dynamic FEM correlations are the mechanism behind the crossover. What it shows is narrower: removing the dominant pooled translation-like covariance subspace improves decoding near −0.20, while having little or no effect on the real condition at −0.40. Because the same operation also helps in the stabilized control, the removed subspace is better interpreted as a shared positional nuisance mode than as FEM-specific covariance.

**Differential-ablation follow-up — April 2026** (script: `declan/fem_differential_intervention.py`, `--rate_file_tag allhires_fresh`, M=471, N=756)

To isolate the variance component unique to dynamic FEMs, we fit the positive eigenspace of `C_real - C_stabilized` inside each CV fold and projected out only that differential subspace before rerunning D1.

| LogMAR | mean positive eigvals(`C_real - C_stabilized`) | real Δ | stabilized Δ |
|--------|-----------------------------------------------|--------|---------------|
| −0.20 | [0.003668, 0.001265] | **+0.016** | **+0.017** |
| −0.40 | [0.000859, 0.000293] | **+0.002** | **−0.002** |

**Finding:** This did not isolate a real-only causal component. At −0.20, ablating the real-minus-stabilized subspace improves real and stabilized by essentially the same amount. At −0.40, the differential effect is null in both directions. So even after subtracting the stabilized covariance, the top positive differential directions still behave like a shared nuisance axis under the current decoder/representation, not a uniquely dynamic-FEM correlation mode that explains the crossover.

**Updated conclusion:** The covariance-ablation route has now been tested in two forms: pooled `C_FEM` and differential `C_real - C_stabilized`. Neither yields a clean real-specific causal effect. The crossover still appears to be dominated by first-order spatial sampling in the mean-rate code, while covariance structure is at most a weak byproduct.

**Clear next step:** If we want to stay on mechanism rather than broaden into exploration, the right next control is a *true fixed-position stabilization baseline* (same retinal position across all traces, not per-trace mean position) and then rerun the same covariance comparisons. If we want to move to the next big question instead, Priority 3 (continuous forward pass) is now the more informative branch.

### Priority 3: Continuous forward pass (exploratory)

**Question:** Does the model produce qualitatively different representations when processing a continuous FEM-driven movie without window resets?

**Method:**

Feed the full trial as a single sequence through the model:
- Frontend and convnet can be chunked temporally (feedforward with finite receptive fields — overlap chunks by the temporal receptive field width, concatenate outputs).
- GRU processes the full convnet output sequence in one pass, carrying hidden state naturally across all ~476 steps.
- Read out rates at every time step.

Run on the E-optotype at LogMAR −0.40 under real FEM and stabilized.

**Tests to run on continuous responses:**

1. **D1 retest:** Does time-averaged rate from the continuous pass still show FEM > stabilized? (Confirms the crossover isn't an artifact of windowing.)

2. **D3 retest:** Does the supervised temporal trajectory now carry orientation information beyond D1? (Tests whether continuous GRU dynamics produce orientation-informative temporal structure that independent windows couldn't.)

3. **Correlation geometry evolution:** Compute population covariance in early (frames 1–50), middle (200–250), and late (400–450) windows within the trial. Does the alignment between FEM covariance and orientation signal change over the course of a fixation? (Tests whether temporal integration progressively reshapes the correlation geometry.)

4. **Velocity tracking:** Retest ΔCoM and Δz on continuous responses. Does the continuous GRU hidden state now track eye velocity? (Revisits the transformation dynamics question with genuine temporal continuity.)

**Framing:** This is an out-of-training-regime probe. The model was trained on short windows; the continuous pass characterizes what the learned weights do on long sequences. Results are properties of the trained model, not claims about biological V1 temporal integration. If interesting dynamics emerge, they reflect structure in the learned weights that the training regime didn't require but also didn't prevent.

**Computational cost:** Significant — one full forward pass per (stimulus × condition × trace). Memory is the bottleneck. The convnet can be chunked; the GRU is lightweight. Estimate ~30 minutes per condition if chunked properly.

**PRELIMINARY RESULTS — April 2026** (script: `declan/eoptotype_continuous_pass.py`, LogMAR −0.40, 32-trace pilot)

Continuous-pass decoding on a deterministic 32-trace subset gave:

| Condition | D1 W=1 | D1 W=24 | D1 W=60 | D3 W=60 |
|-----------|--------|---------|---------|---------|
| real | 0.380 ± 0.062 | 0.355 ± 0.093 | 0.414 ± 0.057 | 0.264 ± 0.039 |
| stabilized | 0.415 ± 0.058 | 0.445 ± 0.042 | 0.400 ± 0.051 | 0.302 ± 0.052 |

For comparison, the **standard cached-window pipeline on the exact same 32 traces** still shows the expected crossover at W=60:

| Condition | cached D1 W=1 | cached D1 W=24 | cached D1 W=60 | cached D3 W=60 |
|-----------|---------------|----------------|----------------|----------------|
| real | 0.345 ± 0.033 | 0.538 ± 0.057 | 0.611 ± 0.082 | 0.311 ± 0.037 |
| stabilized | 0.500 ± 0.064 | 0.467 ± 0.076 | 0.461 ± 0.035 | 0.320 ± 0.038 |

**Interpretation:** On this pilot subset, the continuous forward pass does **not** reveal a richer temporal code. D3 remains below D1, and the real-FEM advantage seen in the standard windowed pipeline is largely lost. The simplest reading is that the long continuous pass pushes the model out of its training regime and degrades orientation readout rather than uncovering useful temporal integration. This is therefore a negative result for the strong "figuring space by time" story, at least in the current implementation.

**Caution:** This is still a subset-level pilot, not a full rerun. But it is already informative because the within-subset comparison is apples-to-apples: the same 32 traces still show the crossover under the standard cached-window evaluation, so the continuous-pass drop is not a trace-selection artifact.

---

## 5. Secondary Experiments (after Priorities 1–3)

### 5.1 Phase 3 of displacement decoding at hyperacuity LogMAR

Rerun the FEM vs. static displacement comparison (the Ahissar test) at LogMAR −0.40 instead of the natural-image setting. Does FEM temporal integration improve displacement encoding at the scale where the orientation crossover happens?

### 5.2 FEM amplitude scaling at hyperacuity

The existing plan proposed scaled FEM conditions (0.5×, 2.0×). Run these at −0.40 to test whether the biological FEM amplitude is near-optimal in the hyperacuity regime. If 1.0× is better than 0.5× and 2.0×, the biological amplitude sits at or near the peak of the information-vs-amplitude curve.

### 5.3 Eigenspectrum dimensionality expansion

Compute effective dimensionality (participation ratio) of the signal covariance for instantaneous vs. time-averaged responses under FEM. Does time-averaging under FEM increase the effective dimensionality of the orientation-discriminative signal? This is the dimensionality expansion prediction from the original eigenspectrum plan — now specifically targeted at the hyperacuity regime where FEM helps.

### 5.4 GRU passthrough test at hyperacuity LogMAR

Run the passthrough test on the E-optotype at −0.40 (not just natural images). Does temporal history matter more or less for the E at hyperacuity than for a natural image? If the GRU's temporal processing is more impactful for stimuli near the resolution limit, that would connect the temporal architecture to the crossover.

---

## 6. What Each Outcome Would Mean

### Priority 1 result (April 2026):

Subspace rotation: **NULL**. FEM subspaces are orientation-invariant. Covariance decoder: near chance. Combined decoder: no gain over D1. The information-migration hypothesis is false.

Alignment transition: **CONFIRMED**. α = 0.720 at −0.20, α = 0.558 at −0.40 (real condition). FEM noise is more aligned with the orientation signal direction above threshold than at hyperacuity. This is the supported form of the geometric mechanism — weaker than subspace rotation would have been, but data-consistent.

The story: FEM helps at hyperacuity through spatial sampling (mean-rate accumulation). The correlation geometry is consistent with FEM being more information-limiting when the signal is strong and less limiting at hyperacuity, but the FEM covariance subspace does not carry orientation identity in any decodable form.

### If Priority 2 (global FEM ablation) shows the predicted causal pattern:

Removing the FEM subspace improves D1 at −0.20 (correlations were contaminating the signal direction, α = 0.720) and has a smaller or null effect at −0.40 (α = 0.558, less aligned). This converts the descriptive α finding into a causal claim: the alignment difference translates into a performance difference when the FEM subspace is removed. That is the mechanistic form of the alignment story.

### If Priority 3 shows continuous GRU dynamics add orientation information (D3 > D1):

"Figuring space by time" has legs in this model. Temporal integration produces orientation-informative structure beyond spatial sampling. This would motivate the full temporal coding story from the original plan.

### If Priority 3 shows no additional benefit (D3 ≈ D1 even with continuous pass):

The mechanism is purely spatial sampling via time-averaged rate, and the GRU's temporal processing (while real and substantial) serves spike-rate prediction rather than orientation discrimination. The story is about sampling, not temporal coding. Still publishable — it identifies the mechanism cleanly.

### If Priorities 1–2 fail (alignment doesn't change, intervention is ambiguous):

The crossover is real but the geometric explanation is wrong. Alternative explanations: nonlinear manifold curvature, effective sample size effects, or rendering artifacts at the pipeline boundary. The non-monotonicity between −0.25 and −0.30 (likely reflecting the lo-res/hi-res pipeline transition) would need to be resolved before trusting the exact crossover location.

---

## 7. Relationship to the Grant

**Aim 1 (shared variability = active sensing or noise?):** The model establishes that FEM-driven reafferent correlations are structured and low-rank. The alignment/intervention tests determine whether these correlations are information-limiting or information-expanding — distinguishing "noise" from "signal" at the population geometry level.

**Aim 2 (when and how do FEMs enhance visual information?):** The crossover at −0.32 answers "when" — in the hyperacuity regime. The alignment test answers "how" — by reshaping the correlation geometry. The continuous forward pass tests whether temporal integration adds to the spatial sampling mechanism.

**Aim 3 (invariance across the hierarchy?):** The image-specificity of the displacement code (cross-image R² = −1.3) establishes what V1 provides to downstream areas: a rich, image-specific spatial code in which content and displacement are entangled. Downstream areas must disentangle them. The binding framework's Prediction 2 (separable subspaces under multiple causes) remains the natural Aim 3 test.

---

## 8. Honest Assessment of What Is Established vs. Hypothesized

| Claim | Status | Evidence |
|-------|--------|----------|
| FEMs improve orientation discrimination in hyperacuity | **Established** | D1 crossover at −0.35 through −0.50, ~5-point FEM advantage |
| The benefit is through time-averaged rate (spatial sampling) | **Established** | D3 ≈ chance, D2a ≈ D1, MLP ≈ D1; D1 is sufficient |
| The response manifold is smooth, image-specific, locally linear | **Established** | Displacement R² ≈ 0.998 within-image, −1.3 cross-image |
| FEM covariance is rank-2, image-specific | **Established** | PCA capture ~97%, cross-stimulus alignment 0.37 |
| Model has meaningful temporal processing within its window | **Established** | GRU passthrough R² = −50.5, RSA = 0.82 |
| All model correlations are purely reafferent | **Established** | Architecture: shared core, no neuron-neuron interaction |
| FEM correlations transition from info-limiting to info-expanding | **Partially confirmed** | α higher at −0.20 (0.720) than −0.40 (0.558); causal intervention (Priority 2) not yet run |
| FEM covariance subspace rotates with E orientation | **FALSE** | Off-diagonal overlap ~1.0 at both LogMARs; FEM subspace is orientation-invariant |
| Orientation info migrates from mean rate to FEM covariance geometry at hyperacuity | **FALSE** | Covariance decoder near chance; combined decoder no gain over D1 |
| Temporal integration (beyond averaging) helps in hyperacuity | **Unknown** | D3 null under independent windows; continuous pass not yet run |
| The crossover location (−0.32) is precise | **Uncertain** | Non-monotonicity at −0.25/−0.30 may reflect pipeline boundary |
| Velocity/transformation is decodable with continuous GRU | **Unknown** | Previous nulls may reflect independent-window limitation |

---

## 9. Non-Monotonicity — RESOLVED (April 2026)

The all-hires rerun (`neurometric_allhires_fresh.npz`, `hires_threshold=2.0`) confirms the non-monotonicity is **not a pipeline artifact**:

| LogMAR | stabilized A (old mixed) | stabilized A (all hires) |
|--------|--------------------------|--------------------------|
| −0.15  | 0.948                    | 0.955                    |
| −0.20  | 0.758                    | 0.774                    |
| −0.25  | 0.761                    | 0.777                    |
| −0.30  | 0.865                    | 0.882                    |
| −0.35  | 0.837                    | 0.837                    |

The dip at −0.20/−0.25 persists with a uniform hi-res pipeline. It likely reflects E-optotype discretization at a spatial scale where letter features are near the pixel grid's Nyquist limit. The crossover and FEM advantage are unchanged: FEM hurts at −0.20 (−0.107) and helps at −0.35 through −0.50 (+0.051). The −0.20 and −0.40 operating points used in Priority 1 are valid.

**ΔLogMAR = 0.002** in the all-hires run is misleading — the threshold-fitting routine expects a monotone psychometric curve. The non-monotonic stabilized curve breaks the fit. Do not report ΔLogMAR as the primary metric; use the table directly.
