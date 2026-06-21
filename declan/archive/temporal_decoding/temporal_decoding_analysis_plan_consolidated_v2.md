# How Eye Movements Shape the Neural Code: A Consolidated Analysis Plan

## Using the Digital Twin of Marmoset Foveal V1 to Test Whether Fixational Eye Movements Create a Useful Temporal Population Code

---

## 0. Preamble: What This Document Is

This document consolidates and revises three earlier analysis plans in light of extensive critical review. The earlier plans proposed five metrics for quantifying how fixational eye movements (FEMs) transform spatial information into temporal neural code structure. Those plans were critiqued for three main weaknesses:

1. **Conflating "temporal code" with "temporal correlation code."** Concatenating rate vectors across time and showing improved decoding establishes that a temporally extended response is useful — not that second-order correlation structure beyond the mean trajectory is useful. The stronger claim requires an explicit ablation.

2. **Treating descriptive geometry as mechanistic explanation.** Eigenspectrum alignment, dimensionality expansion, and coding efficiency describe population structure but do not, by themselves, determine whether that structure is useful for downstream computation. Only task-level performance adjudicates utility.

3. **Presenting co-equal analyses without hierarchy.** The analyses have different evidential weight and different assumption loads. They should be organized as: primary task metric → mechanistic companion → secondary/exploratory.

This revised plan addresses all three weaknesses. It is organized as a strict hierarchy:

- **Tier 1 (Primary):** Task-level evidence that FEMs improve a temporally extended population code for fine spatial discrimination.
- **Tier 2 (Mechanistic):** Population-geometric and biophysical analyses that explain *how* the temporal code works.
- **Tier 3 (Secondary):** Local parametric sensitivity analyses for continuous stimulus parameters.
- **Tier 4 (Exploratory):** Near-optimality tests and further extensions.

Throughout, we are explicit about what each analysis can and cannot prove.

---

# TIER 1: PRIMARY TASK-LEVEL ANALYSES

## 1. Temporal Decoding of E Orientation with Ablation Ladder

### 1.1 Scientific question

Do fixational eye movements improve the discriminability of fine spatial features in V1 population responses, and if so, does the improvement come from temporally extended mean trajectories, from second-order temporal structure beyond the mean, or from both?

### 1.2 Why this is primary

This analysis directly tests whether a realistic downstream readout can exploit FEM-induced temporal structure for a concrete perceptual task. It produces interpretable outputs (accuracy, mutual information, threshold shifts in LogMAR) that connect to clinical acuity and psychophysical measurements. And it includes the critical ablation that separates "useful temporal code" from "useful temporal correlation code."

### 1.3 Stimulus and task

**Tumbling E orientation discrimination (4AFC).** Classify the orientation of a Tumbling E (0°, 90°, 180°, 270°) at each LogMAR value. The `DifferentiableStimulus` class already parameterizes the E with continuous control over position, orientation, and LogMAR size.

**LogMAR sweep:** LogMAR ∈ {1.0, 0.8, 0.6, 0.4, 0.2, 0.0, −0.1, −0.2, −0.3}. At LogMAR 0.0 the critical gap subtends 1 arcmin.

**Eye movement conditions:**

| Condition | Description | Purpose |
|-----------|-------------|---------|
| Real FEM | Measured fixational eye traces | Biological condition |
| Stabilized | Fixed at trial-mean position (scale = 0) | Null: no temporal modulation |
| Half-amplitude | Scale = 0.5 | Parametric control |
| Double-amplitude | Scale = 2.0 | Tests saturation / degradation |
| Shuffled | Real traces assigned to wrong trials | Breaks stimulus-trace coupling |
| Matched-budget null | Synthetic traces matching RMS displacement, path length, and velocity PSD of real traces | Tests whether biological FEM *structure* matters beyond kinematics |

### 1.4 The ablation ladder (critical component)

This is the single most important addition relative to the original plans. For each condition, train and evaluate four nested decoders:

**Model A — Rate only.** The decoder receives the time-averaged rate vector **r̄** ∈ ℝ^N for each single trial. This tests what the spatial code alone provides. It is the "camera model" baseline.

**Model B — Temporal mean trajectory subspace.** First, compute the class-mean temporal trajectory for each stimulus condition (averaged across all eye traces for that condition). Perform PCA on these K class-mean trajectories to obtain a d_B-dimensional subspace U_B that captures the systematic, stimulus-driven temporal dynamics. Then, for each *single trial*, project its full temporal trajectory onto U_B:

**x_B** = U_Bᵀ **r_trial** ∈ ℝ^{d_B}

The decoder receives x_B. Crucially, U_B is learned from class means only (fit on training data), so it captures systematic temporal dynamics but discards trace-specific variation. Each trial is still decoded individually — this is not a template matcher.

**Model C — Full single-trial temporal trajectory.** The decoder receives the full single-trial rate trajectory [**r**(1); ...; **r**(T_int)] ∈ ℝ^{NT}, optionally projected onto a broader subspace (e.g., top K PCs of the total temporal covariance, K > d_B) that preserves both mean dynamics and trace-specific structure. The difference between C and B is that C retains trace-specific temporal variation that B projects out.

**Model D — Residual covariance features.** After projecting out the B-subspace from each single-trial trajectory (removing the systematic temporal mean dynamics), compute second-order features of the residual:

- **Primary D representation:** compute the low-rank lagged cross-covariance matrix of the residual trajectory (residual = r_trial − U_B U_Bᵀ r_trial), vectorize the upper triangle, and reduce via PCA to a manageable dimension. The decoder receives Model B features + these residual covariance features.
- **Sensitivity D2:** within-neuron lagged autocovariance only (no cross-neuron terms). This tests whether the correlation-code contribution, if present, requires population-level temporal interactions or is carried by single-neuron temporal structure.

The key comparisons:

| Comparison | What it tests |
|------------|--------------|
| B > A | Systematic temporal dynamics help, even without trial-specific variation |
| C > B | Trial-specific temporal patterns add information beyond mean dynamics |
| C > A | Temporally extended code is useful (temporal code claim) |
| D > B (marginal gain of covariance features) | Second-order temporal structure carries discriminative information beyond the mean trajectory (correlation code claim) |

**Only if D shows significant marginal gain over B, and that gain is stable across LogMAR values and robust to matched-budget null traces, can we claim a "temporal correlation code."** A marginal D gain at a single condition is suggestive but not sufficient. Otherwise, the correct claim is that FEMs improve a temporally extended population code through systematic mean trajectory modulation.

**Critical implementation details for the ladder:**
- U_B must be fit on training-fold data only and applied to held-out data. The same U_B is used across FEM conditions within each fold to ensure fair comparison.
- **Trace distribution equalization:** when computing class means for U_B, verify that eye trace statistics (RMS, path length, velocity distribution) are identical across stimulus conditions, or resample to equalize. If trace distributions differ across classes (even slightly due to dataset structure), U_B can encode trace-distribution information rather than purely stimulus-driven dynamics, contaminating the B→C comparison.
- All four models decode single trials. The only difference is which statistics of those trials are available to the decoder.
- Regularization (L2 penalty) is chosen by inner CV within each training fold, separately for each model.
- **Model D negative control:** shuffle time indices independently within each neuron (destroying temporal correlations while preserving per-neuron marginal rate distributions) and recompute Model D features. This provides a baseline for how much apparent D gain can arise from finite-sample structure or feature construction artifacts alone. If the real D gain does not significantly exceed the shuffled-time D gain, the correlation-code claim is not supported.

**Narrative discipline:** the paper should present C > A as the headline ("temporal code exists"), B > A vs. C > B as the decomposition ("mean trajectory vs. trace-specific gain"), and D > B as a conditional sub-claim, visually and textually separated. Do not let the narrative center on the correlation-code story unless D is strong, stable, and survives the negative control.

### 1.5 Decoder specification

**Primary:** L2-regularized logistic regression (ridge). Regularization chosen by inner cross-validation. Linear decoders connect to the geometric analyses in Tier 2 and are biologically plausible as models of downstream readout.

**Control:** Small MLP (1 hidden layer, 64 units, ReLU, dropout). If the MLP substantially outperforms the linear decoder under FEMs but not under stabilization, the temporal code has nonlinear structure that the linear geometric analyses miss.

### 1.6 Cross-validation protocol

**Eye-trace-level grouped cross-validation (5-fold).** Split the M measured eye traces into 5 non-overlapping folds. Train on 4 folds, test on the held-out fold. This ensures the decoder generalizes across eye movements and cannot memorize specific temporal patterns of specific traces.

**Poisson noise injection.** For each eye trace and stimulus, generate multiple Poisson spike count realizations from the twin's predicted rates. This provides more training samples and ensures robustness to intrinsic neural variability. This is the primary noise model for the decoder results.

**Noise sensitivity check.** Repeat the primary decoder analysis with additional injected low-rank residual covariance (matched to Aim 1 residual structure when available) to test whether the main conclusions survive a richer noise model. Report whether the A/B/C/D ordering and the crossover point are robust.

**Analytical companions (separate from decoder results).** For analytical metrics (d', J_pop) computed as companions, evaluate under: (i) Poisson-only intrinsic noise, (ii) Poisson + FEM covariance, (iii) Poisson + FEM + scaled empirical residual from Aim 1 data. Report sensitivity to noise model choice. The primary claim rests on the decoder results, not on these analytical companions.

### 1.7 Output metrics

**Cross-validated accuracy** and **mutual information** I(S; Ŝ) from the confusion matrix. MI is preferred for cross-condition comparison because it accounts for error patterns.

**Analytical d'** as a companion: d'² = Δμᵀ Σ⁻¹ Δμ, computed with the noise model caveats above. Used for decomposition (which neurons contribute, which signal directions matter) but not as the headline result.

### 1.8 Neurometric curves

For each FEM condition and each model (A–D), plot decoding accuracy vs. LogMAR. Define "neural acuity threshold" as the LogMAR where accuracy reaches a criterion (e.g., 62.5% = halfway between chance and ceiling for 4AFC).

The headline result: **ΔLogMAR** = threshold(stabilized, Model A) − threshold(real FEM, Model C). This is the total acuity gain from FEMs, directly comparable to psychophysical measurements (Rucci et al. 2007, *Nature*).

The ablation result: decompose ΔLogMAR into contributions from rate modulation (A→B), temporal mean dynamics (B→C), and correlation structure (C→D, if present).

---

## 2. Decoding vs. Integration Time

### 2.1 Scientific question

At what temporal integration timescale does the FEM-driven code become useful, and does this match the timescale of fixational drift?

### 2.2 The integration-time curve

For each FEM condition, train a causal sliding-window decoder (Model C features, window W) and evaluate accuracy:

W ∈ {1, 3, 6, 12, 24, 36, 48, 60} frames (8.3 ms to 500 ms at 120 Hz)

Plot accuracy(W) for each condition. Key predictions:

- **Short W (1–3 frames):** Stabilized may slightly outperform real FEMs, because FEMs add single-frame variance without temporal context.
- **Moderate W (12–30 frames, 100–250 ms):** Real FEMs overtake stabilization as the temporal code becomes readable.
- **Long W (48–60 frames):** Performance saturates. The saturation timescale should match FEM drift dynamics.

The **crossover point** — the W at which real FEMs first outperform stabilization — is a vivid demonstration that eye movements hurt at snapshot timescales but help at biologically relevant integration times. This directly challenges the camera model.

**Alternative outcome to anticipate:** the curves may not cross. Instead, real FEMs may outperform stabilization at all W, but with a steeper slope (faster accumulation). This would be a weaker but still valid active-sensing signature — FEMs consistently improve accumulation efficiency rather than creating a qualitative regime change. Plan to present both possible shapes without forcing one interpretation.

**Implementation note (April 2026):** be explicit about *how* the W-frame window is featurized.
- `time_mean` (accumulation-aligned): average the last W frames to an N-dim vector, then decode.
- `flat_pca` (legacy): flatten the last W frames to $W \times N$, then PCA, then decode.

In the hyperacuity regime, `flat_pca` can artificially wash out real signal and produce a misleading flat/near-chance integration-time curve; use `time_mean` as a primary sanity check when results look suspicious.

### 2.3 Sequential entropy reduction (ideal observer)

As an alternative or supplement to the sliding-window decoder, implement a sequential ideal observer:

For the 4AFC E task, start with uniform prior p(S) = 1/4. At each time step t, update:

p(S | r_{1:t}) ∝ p(r_t | S, r_{1:t-1}) · p(S | r_{1:t-1})

Under a Gaussian approximation, this is tractable with ~130 neurons and 4 classes. Compute the posterior entropy:

H_t = H(S | r_{1:t})

Summary metrics:
- **Entropy reduction rate:** ΔH/Δt in bits per 100 ms
- **Time to criterion confidence:** the time t at which H_t drops below a threshold (e.g., 0.5 bits, corresponding to ~85% confidence in one orientation)
- **Entropy reduction per spike:** ΔH / (total spikes in window)

This is more principled than the fixed-window decoder because it captures the *dynamics* of evidence accumulation. Under the active sensing hypothesis, real FEMs should show faster entropy reduction than stabilization, with the rate peaking at the drift timescale.

### 2.4 Post-saccadic information ramp (ecological extension — conditional)

**Include only if the twin handles post-saccadic dynamics reasonably and the figure is simple enough not to distract from the core E-decoding story. Otherwise defer to later work.**

If the twin generalizes to the saccade-to-fixation transition (it was trained on free-viewing data including both):

1. Run continuous traces containing saccade → fixation (FEM) → saccade sequences.
2. Apply the sliding-window decoder continuously, aligned to saccade offset (t = 0 when the eye lands).
3. Plot decoding accuracy vs. time since saccade offset.

The prediction: accuracy ramps from chance at t = 0 (saccadic blur/suppression) to asymptote over 100–300 ms as FEMs progressively accumulate spatial evidence. This provides a direct ecological demonstration of active sensing: FEMs are the process by which the visual system builds high-acuity representations between gaze shifts.

---

## 3. Spatial Frequency Decomposition

### 3.1 Scientific question

Which spatial frequency components of the stimulus benefit from the temporal code?

### 3.2 Method

Bandpass filter the E stimulus into spatial frequency bands before computing rates:
- Low SF: 0–2 cpd (overall envelope)
- Medium SF: 2–8 cpd (stroke structure)
- High SF: 8–20 cpd (edges, gaps)

For each band, compute decoding accuracy (Model C vs. Model A) under real FEMs and stabilization.

### 3.3 Frequency-resolved gain function

For finer resolution, compute a continuous transfer function:

G(SF) = accuracy_FEM(SF) − accuracy_stabilized(SF)

Overlay this with the predicted resonance curve from the spatiotemporal resonance analysis (Tier 2, Section 6). If G(SF) peaks where the FEM velocity spectrum converts the stimulus SF into the temporal passband of V1 neurons, you have a mechanistic explanation for the spatial-frequency structure of the temporal code benefit.

---

# TIER 2: MECHANISTIC COMPANION ANALYSES

## 4. Signal-Noise Subspace Geometry

### 4.1 Scientific question

How does FEM-induced variability relate geometrically to the stimulus-discriminative subspace, and does that geometry explain the decoding results from Tier 1?

### 4.2 What this analysis can and cannot prove

**Can prove:** the geometric relationship between FEM noise and stimulus signal in population space. Whether FEM variability concentrates in, avoids, or is orthogonal to the signal subspace.

**Cannot prove by itself:** whether that geometry is beneficial or harmful. Alignment can mean "FEMs are converting spatial structure into temporal modulations along coding axes" (good) or "FEMs are generating information-limiting correlations" (bad). Only the task metrics in Tier 1 adjudicate utility. Geometry *explains* the task result; it does not *replace* it.

### 4.3 Covariance decomposition

**Signal covariance C_signal.** For K stimulus conditions, compute condition means averaged across M eye traces, then compute the covariance of those means:

C_signal = (1/K) Σ_k (μ_k − μ)(μ_k − μ)ᵀ

**FEM noise covariance C_FEM.** For each stimulus condition, compute the within-condition covariance across eye traces, then average across conditions:

C_FEM = (1/K) Σ_k Cov_traces[r_k]

**Intrinsic noise.** C_Poisson = diag(⟨λ⟩). Add analytically.

Compute these for both instantaneous (time-averaged) and temporal (vectorized trajectory) representations.

### 4.4 Corrected null for temporal dimensions

The original plan proposed lifting the instantaneous subspace into temporal space and identifying orthogonal eigenvectors as "new temporal dimensions." This is wrong — even stabilized responses have temporal dynamics (onset transients, adaptation, temporal filtering).

**Corrected null:** the signal subspace computed *under stabilization in the temporal representation*. Only dimensions present under real FEMs but absent under stabilized temporal responses are genuinely FEM-created.

### 4.5 Task-relevant subspace SNR (replacing participation ratio)

Rather than reporting generic effective dimensionality (participation ratio), define the task-relevant subspace directly:

- For E discrimination: the LDA discriminant directions (3 directions for 4 classes).
- For continuous parameters: the span of Fisher derivative vectors {f'_x, f'_y, f'_θ, f'_logmar}.

Then compute:

TR-SNR = tr(P_task C_signal P_task) / tr(P_task C_noise P_task)

where P_task projects onto the task-defined subspace and C_noise = C_FEM + C_Poisson.

TR-SNR directly measures whether the signal-to-noise ratio in the *relevant* coding dimensions improves with FEMs. It avoids the inflation and ambiguity of generic dimensionality measures.

### 4.6 Alignment analysis

**Alignment fraction α (descriptive).** Project C_FEM onto the top d signal eigenvectors:

α = tr(U_sᵀ C_FEM U_s) / tr(C_FEM)

Compare to chance level d/N. Compare across conditions (real, shuffled, scaled). α is a geometric descriptor, not a value judgment — it tells you *where* the noise lives, not whether it helps or hurts.

**Interpretation:** High α under real FEMs could indicate:
- *Information-limiting correlations* (bad): random, unpredictable FEM variability along signal axes.
- *Systematic signal-bearing modulations* (good): predictable, stimulus-specific temporal trajectories that traverse the signal subspace.

The distinction between these cases cannot be resolved by α alone. It is resolved by the representational intervention below.

**Descriptive bridge to Tier 1:** Plot ΔAccuracy (from Tier 1 decoding) vs. α across FEM conditions. Suggestive but not decisive.

**Representational intervention (the decisive mechanistic test — primary within Tier 2):**

This is the strongest analysis in Tier 2 and should be the main panel in the geometry figure.

1. Identify the FEM-aligned temporal subspace: the top principal components of C_FEM that overlap most with the signal subspace (the components with smallest principal angles to U_s).
2. Project out this aligned subspace from the single-trial temporal trajectories used in Model C.
3. Rerun the Tier 1 decoder on the residual trajectories.

Interpretation:
- **Accuracy collapses:** the aligned subspace was carrying useful stimulus information. FEM variability along signal axes is signal-bearing, not information-limiting. **This is the decisive result for the mechanistic story.**
- **Accuracy improves:** the aligned variability was nuisance. Removing it helps the decoder by cleaning the signal subspace.
- **Accuracy unchanged:** the aligned subspace contributed redundant information that was also available from other coding dimensions.

This is a stronger mechanistic test than any descriptive scatterplot or eigenspectrum summary because it establishes a representational dependence, not just a correlation. It directly answers: "does this subspace matter for the code?"

### 4.7 Principal angles and per-dimension contamination

Compute canonical angles between signal and noise subspaces. Compute per-signal-dimension contamination ν_j = u_jᵀ C_FEM u_j and SNR_j = σ²_j / ν_j.

Examine whether the most important signal dimensions (largest σ²_j) are the best protected (highest SNR_j) or the most contaminated (lowest SNR_j).

### 4.8 Cross-kinematic generalization

Test whether the temporal code is robust across different eye traces:

1. Train the decoder on responses to FEM Set A.
2. Test on responses to FEM Set B (different traces, same biological statistics).

High generalization confirms that the temporal code is an invariant of the eye-movement statistics, not specific to particular trajectories. This is already implicit in the grouped cross-validation (Section 1.6), but making it explicit and reporting the cross-trace generalization accuracy is valuable.

Additionally: test whether a decoder trained on FEM responses can decode stabilized responses (and vice versa). If cross-regime generalization is poor, the two conditions use fundamentally different coding strategies — a strong result.

---

## 5. Trajectory Sensitivity and Reproducibility

### 5.1 Scientific question

How sensitive are the twin's population responses to small perturbations in eye trajectory, and does that sensitivity concentrate in task-relevant or task-irrelevant subspaces?

### 5.2 Why this matters

This analysis resolves the alignment ambiguity from Section 4.6. If FEM covariance aligns with the signal subspace, we need to know whether that alignment represents reproducible, stimulus-specific temporal modulations (signal) or chaotic sensitivity to small trajectory differences (noise).

### 5.3 The twin triviality and the real question

Since the twin is deterministic, exact reproducibility given the same eye trace is trivially perfect (ρ = 1). The naive version of the reproducibility analysis ("do similar traces give similar responses?") reduces to "is the deterministic model deterministic?" — which is uninformative.

The non-trivial twin-side question is about **local Lipschitz structure**: how rapidly do predicted response trajectories diverge as eye traces diverge, and does that divergence occur in task-relevant or task-irrelevant subspaces?

### 5.4 Method: trajectory divergence analysis

1. For a fixed stimulus, select pairs of eye traces at varying distances in trajectory space (e.g., L2 distance on trajectory PCA projections, or DTW distance).
2. Compute the corresponding response divergence: ||r(trace_a) − r(trace_b)||² in the full (NT)-dimensional space.
3. Decompose response divergence into task-relevant and task-irrelevant components by projecting onto the signal subspace (U_s) and its complement:
   - Task-relevant divergence: ||U_sᵀ(r_a − r_b)||²
   - Task-irrelevant divergence: ||(I − U_s U_sᵀ)(r_a − r_b)||²
4. **Normalize by signal variance in each subspace** to obtain sensitivity rather than raw scale:
   - Task-relevant sensitivity: ||U_sᵀ(r_a − r_b)||² / tr(U_sᵀ C_signal U_s)
   - Task-irrelevant sensitivity: ||(I − U_s U_sᵀ)(r_a − r_b)||² / tr((I − U_s U_sᵀ) C_total (I − U_s U_sᵀ))
5. Plot normalized sensitivity vs. eye-trace distance.

Without this normalization, large divergence in the task-relevant subspace could simply reflect that the signal itself is large there. The normalized version asks: relative to how much signal lives in each subspace, how much does eye-trace perturbation affect it?

If normalized task-relevant sensitivity is high and task-irrelevant sensitivity is low, the twin amplifies eye-movement differences specifically into stimulus-relevant modulations — the "reformatting" mechanism.

### 5.5 Extension to real neural data (Aim 1)

The full reproducibility analysis — do similar eye traces produce similar neural responses? — becomes non-trivial with real data, where intrinsic noise adds genuine unpredictability. For real data from Aim 1:

1. Find trial pairs with similar eye trajectories (nearest neighbors in eye-trace space).
2. Compute within-pair response similarity vs. overall across-trial variability.
3. Ask whether this ratio predicts decoding performance across stimulus conditions.

### 5.6 Ablation: does the predictable FEM component carry stimulus information?

1. Compute eye-state-conditioned response templates (predicted rate for each eye position/velocity from the twin).
2. Subtract the template from single-trial responses to get residuals.
3. Decode stimulus from (a) the templates alone and (b) the residuals alone.

If template-based decoding is strong, the predictable FEM component is signal-bearing. If residual-based decoding is strong, there is information beyond what eye-state conditioning captures.

---

## 6. Spatiotemporal Resonance

**Status: conditional — include only if the analysis comes out clean. This is the most speculative mechanistic section.**

### 6.1 Scientific question

Is there a biophysical mechanism that explains *why* FEM drift helps at the single-neuron level?

### 6.2 The physics

V1 neurons are spatiotemporal bandpass filters. Eye movement with velocity v converts a spatial frequency f_s into a temporal frequency f_t:

f_t = f_s × v

If the FEM velocity distribution shifts the stimulus's spatial frequencies into the temporal passband of foveal V1 neurons, FEMs act as a physical heterodyne: downconverting high-resolution spatial information into the cortical temporal passband.

### 6.3 Caveats

This analysis is attractive but carries real risks:
- Many twin units may not have cleanly separable SF/TF preferences (the twin's spatiotemporal tuning may not factorize into independent SF and TF components).
- The effective temporal passband under naturalistic input may differ from grating-derived TF tuning.
- The image PSD and eye-velocity PSD may interact in a less factorizable way than the simple resonance integral assumes.

If these issues make the resonance scores noisy or uncorrelated with decoding weights, the analysis should be reported as inconclusive rather than forced.

### 6.4 Analysis

1. **Characterize each neuron's spatiotemporal tuning.** From the twin's learned filters (or from grating responses computed through the twin), estimate preferred spatial frequency (SF_pref) and preferred temporal frequency (TF_pref) for each neuron.

2. **Compute the FEM velocity spectrum.** From measured eye traces, compute the distribution of instantaneous velocities and the power spectral density PSD_eye(f).

3. **Predict resonance.** For each neuron i, compute the overlap between its temporal tuning curve and the FEM-induced temporal frequency:

resonance_i = ∫ TF_tuning_i(f_t) × PSD_image(f_t / v) × PSD_eye(v) dv df_t

In simpler terms: does the FEM velocity shift the stimulus's peak spatial frequencies into the neuron's preferred temporal frequency?

4. **Test the prediction.** Compare resonance_i to each neuron's contribution to temporal decoding (e.g., the absolute decoder weight for neuron i in Model C). If neurons with higher resonance contribute more to decoding, the heterodyne mechanism explains the temporal code.

5. **Overlay with spatial frequency gain.** Compare the predicted resonance curve G_predicted(SF) to the empirical gain function G(SF) = accuracy_FEM(SF) − accuracy_stabilized(SF) from Section 3. If they match, the biophysics explains the spatial-frequency structure of the temporal code benefit.

### 6.5 What this proves (if it works)

This analysis provides a *mechanistic explanation* at the single-neuron level that complements the population-level geometric analysis in Section 4. It connects the biophysics of spatiotemporal filtering to the perceptual prediction: the SF band that benefits most from FEMs should be the one where FEM velocities shift stimulus SFs into the temporal passband of foveal V1. If the resonance scores are noisy or unpredictive, the analysis is inconclusive about this particular mechanism — but the population-level results from Tiers 1–2 still stand.

---

# TIER 3: SECONDARY LOCAL-SENSITIVITY ANALYSES

## 7. Fisher Information for Continuous Parameters

### 7.1 Scope

This analysis is restricted to **local parametric sensitivity** around threshold stimulus values. It uses the forward-AD infrastructure already implemented in `check_fixrsvp_model_fisherinfo.py`. It is secondary because Fisher information is local (infinitesimal perturbations), assumption-dependent (requires a noise model for Σ), and does not directly establish task utility.

### 7.2 What Fisher adds that decoding cannot

Fisher information provides:
- Sensitivity to *continuous* parameters (position, orientation, size) rather than discrete categories.
- The full **Fisher information matrix** J ∈ ℝ^{K×K} across multiple parameters simultaneously, revealing estimation trade-offs and parameter coupling.
- Direct connection to the Cramér-Rao bound on estimation variance.

### 7.3 The upgrade from J_indep to J_pop

Replace the current implementation (summing single-neuron Fisher info):

J_indep = Σ_i (∂λ_i/∂θ)² / λ_i

with the population Fisher information:

J_pop = f'ᵀ Σ⁻¹ f'

where f' = ∂λ/∂θ is computed via forward-AD (already implemented) and Σ = Σ_FEM + Σ_Poisson.

Compute η = J_pop / J_indep as a **comparative diagnostic** across FEM conditions, not as a headline metric. η tells you whether FEM-induced correlations help or hurt for that specific parameter, but its absolute value depends on the noise model.

### 7.4 Stress-test across noise models

Compute J_pop under three noise models:
1. Σ = diag(λ) (Poisson only, reduces to J_indep)
2. Σ = Σ_FEM + diag(λ) (FEM covariance + Poisson)
3. Σ = Σ_FEM + diag(λ) + Σ_residual (adding a scaled version of the residual internal covariance from Aim 1)

Report sensitivity of η to noise model choice. If η is robust, the result is trustworthy. If η is sensitive, report the range and interpret cautiously.

### 7.5 Fisher information matrix: parameter coupling geometry

Compute the full 4×4 Fisher matrix for (x, y, orientation, LogMAR):

J_jk = f'_jᵀ Σ⁻¹ f'_k

This reveals:
- **Diagonal elements:** precision for individual parameters.
- **Off-diagonal elements:** statistical coupling. Do FEMs create confounds between position and orientation? Between size and position?
- **Condition number κ(J):** is the code isotropic (all parameters estimated equally well) or anisotropic (some much better than others)?
- **How FEMs change the coupling structure:** do they make the code more isotropic, or more specialized for particular parameters?

This is the primary reason to keep Fisher: no other analysis provides the multi-parameter estimation geometry.

---

# TIER 4: EXPLORATORY ANALYSES

## 8. Matched-Budget Near-Optimality Test

### 8.1 Motivation

Are biological FEMs better than random movements with the same motor budget?

### 8.2 Why this replaces the "frontier"

The earlier plan proposed plotting J_pop vs. tr(Σ) across FEM amplitude scales and looking for an "elbow." This was criticized because total neural variance is not the relevant cost, and the elbow interpretation requires an independently justified cost function.

Matched-budget null traces solve this by asking a simpler, more direct question: among traces with the *same oculomotor budget*, is the real trace especially good?

### 8.3 Method

For each real eye trace, generate matched null traces that preserve:
- RMS displacement
- Path length
- Velocity power spectral density
- Microsaccade count and amplitude distribution

But have different specific trajectories (different random seeds, or phase-randomized versions of the real trace).

Compare decoding accuracy (Tier 1 Model C) for real traces vs. matched nulls. If real traces consistently outperform their matched nulls, biological FEM statistics are near-optimal for the temporal code — and the conclusion does not depend on any assumed cost function.

**Report the full distribution, not just the mean.** For each real trace, generate multiple matched nulls and report the distribution of null accuracies alongside the real accuracy. If the real trace's accuracy exceeds the 95th percentile of its matched-null distribution (not just the mean), the result is much harder to dismiss as a statistical fluctuation.

### 8.4 Implementation

- **Phase randomization:** FFT the eye trace, randomize phases while preserving the amplitude spectrum, inverse FFT. This preserves the velocity PSD exactly while destroying the specific trajectory.
- **Bootstrap matching:** from the library of measured traces, find pairs with similar path length / RMS but different trajectories. Use one as the "real" and the other as the "matched null."
- **Gaussian process nulls:** fit a GP to the eye trace statistics (autocorrelation, velocity distribution) and sample new traces from the posterior.

### 8.5 Trace-budget stratification within real traces

As a cheaper complement to the matched-budget nulls, stratify the real eye trace library by natural variation in oculomotor budget:

- **Low budget:** bottom tercile of RMS displacement / path length
- **Medium budget:** middle tercile
- **High budget:** top tercile

Ask whether the temporal decoding gain (Model C accuracy − Model A accuracy) scales monotonically with budget, saturates, or peaks at intermediate values. This reveals whether biological FEM benefits are concentrated in a particular part of trace space and whether there are diminishing returns to movement — without requiring any synthetic null generation.

If the gain peaks at intermediate budget, this provides natural evidence for an optimal FEM amplitude that is more grounded than the earlier "frontier elbow" proposal.

---

## 9. Optimal Eye Trajectory via Gradient Descent

### 9.1 Scope

Using the differentiable pipeline (DifferentiableStimulus → DifferentiableRetina → twin → readout), optimize eye trajectories to maximize decoding performance or Fisher information for E discrimination at threshold LogMAR, subject to realistic kinematic constraints.

### 9.2 What it adds

If optimized trajectories have statistics similar to measured FEMs (drift amplitude, velocity spectrum, directional structure), this supports the near-optimality claim from a complementary direction. If they differ systematically, the differences reveal what aspects of FEM statistics are suboptimal or optimized for other objectives.

### 9.3 Caveats

This analysis is computationally expensive, high-dimensional, and the result depends on the choice of constraints and objective. It should be treated as exploratory and supportive, not as primary evidence.

---

# CROSS-CUTTING CONSIDERATIONS

## 10. The Noise Model Problem

### 10.1 Statement of the problem

The digital twin is deterministic. To compute any metric involving noise (d', J_pop, decoding with added noise), we must construct a noise model. The results are only as trustworthy as the noise model.

### 10.2 The recommended noise model

Σ_total = Σ_FEM + Σ_Poisson + (optionally) Σ_residual

- **Σ_FEM:** across-eye-trace covariance of the twin's rate predictions for a fixed stimulus. This is the component of neural variability driven by eye movement variability. It is low-rank (eye movements are 2D) and structured.
- **Σ_Poisson:** diag(⟨λ⟩). Intrinsic Poisson spiking variability. Not a ridge penalty — it follows from the law of total covariance for a Poisson spiking model with deterministic rates.
- **Σ_residual:** internal neural variability not explained by stimulus or eye state. This is estimated from real neural data in Aim 1. Its magnitude and structure matter. If it is large and structured, it could change the conclusions.

### 10.3 Implicit assumption: Poisson noise is independent across neurons

The Poisson component assumes independent spiking noise across neurons. In reality, residual noise (after conditioning on stimulus and eye state) almost certainly has cross-neuron structure — shared gain fluctuations, correlated variability from recurrent circuits, etc. This means:

- **Poisson-only results are a lower bound on the impact of noise correlations.** The Poisson model underestimates the true noise covariance and may overstate d', J_pop, and decoding performance.
- **Structured residual noise (Σ_residual) can interact with FEM-induced structure in nontrivial ways.** It could amplify or suppress the temporal coding benefit depending on its alignment with the signal and FEM subspaces.

This is why Σ_residual from Aim 1 data matters and why early results should be explicitly flagged as "Poisson-only lower bound" until the residual structure is incorporated.

### 10.4 Sensitivity analysis

Every Σ⁻¹-based result (analytical d', J_pop, η) should be reported under at least two noise models (Poisson-only and Poisson + FEM) and ideally three (adding Σ_residual from Aim 1 when available). The primary results (Tier 1 decoding) should be reported under both Poisson-only noise injection and Poisson + FEM covariance noise injection, and the conclusions should be robust to the choice.

---

## 11. What Each Analysis Can Claim

| Analysis | Can claim | Cannot claim without further evidence |
|----------|-----------|---------------------------------------|
| Tier 1 decoding (Model C > A) | FEMs improve a temporally extended population code | That the improvement is specifically in correlation structure |
| Tier 1 ablation (Model D > B, stable across LogMAR and nulls) | Second-order temporal structure carries discriminative information beyond the mean trajectory (correlation code claim) | That this structure is the dominant contribution (could be small relative to B→C gain) |
| Tier 1 integration time crossover | Temporal code requires biologically relevant integration windows | That the code is optimal |
| Tier 2 alignment + intervention (accuracy collapses when aligned subspace removed) | FEM variability along signal axes is signal-bearing | This from alignment fraction α alone (α is descriptive, not signed) |
| Tier 2 spatiotemporal resonance (if clean) | Biophysical mechanism linking FEM velocity to V1 temporal tuning | That this mechanism fully explains the population-level benefit |
| Tier 3 Fisher matrix | Local parametric sensitivity and parameter coupling | Broad perceptual claims or natural-image coding conclusions |
| Tier 4 matched-budget optimality | Whether biological FEM statistics outperform kinematics-matched nulls | That FEMs are globally optimal |

---

## 12. Suggested Figure Structure

### Figure 1 (Primary result): Neurometric curves and integration time
- **A:** Decoding accuracy vs. LogMAR for real FEM, stabilized, and matched-budget nulls, using Model C (full temporal) and Model A (rate-only).
- **B:** Integration time curve: accuracy vs. window W for real FEM and stabilized at threshold LogMAR. Highlight crossover.
- **C:** Sequential entropy reduction: H(S|r_{1:t}) vs. t for real FEM and stabilized.
- **D:** Neural acuity threshold (LogMAR at criterion) for each condition.

### Figure 2 (Ablation): Where does the temporal gain live?
- **A:** Decoding accuracy for Models A, B, C, D at threshold LogMAR, with error bars from cross-validation.
- **B:** Marginal gain at each ablation step (B−A, C−B, D−B). Significance bars. D−B is the critical comparison for the correlation-code claim.
- **C:** Stability of the D−B gain across LogMAR values.
- **D:** If D > B: visualization of the covariance features that contribute (which neuron pairs, which lags).

### Figure 3 (Mechanism — geometry): Signal-noise subspace structure
- **A:** Eigenvalue spectra of C_signal (instantaneous vs. temporal under real FEM vs. stabilized). Use stabilized temporal as the null, not lifted instantaneous.
- **B:** Task-relevant subspace SNR across FEM conditions.
- **C:** Representational intervention: decoding accuracy before and after removing FEM-aligned signal subspace. If accuracy collapses → aligned variability is signal-bearing.

### Figure 4 (Mechanism — biophysics): Spatiotemporal resonance *(conditional — include only if resonance scores are predictive)*
- **A:** Joint distribution of SF_pref and TF_pref for the V1 population.
- **B:** FEM velocity distribution and its implied SF-to-TF conversion.
- **C:** Predicted resonance vs. empirical spatial-frequency gain G(SF).
- **D:** Neuron-level: decoder weight vs. resonance score.

### Figure 5 (Ecological context): Post-saccadic information ramp *(conditional — include only if twin handles post-saccadic dynamics well)*
- **A:** Continuous decoding accuracy aligned to saccade offset.
- **B:** Rate of evidence accumulation vs. time post-saccade.
- **C:** Comparison to stabilized (flat) and saccade-only (reset) controls.

### Figure 6 (Secondary / supplement): Fisher matrix geometry
- **A:** J_pop for orientation, position, and size across FEM conditions under multiple noise models.
- **B:** Fisher matrix heatmap showing parameter coupling under real FEM vs. stabilized.
- **C:** Cramér-Rao bounds on position and orientation estimation variance.

---

## 13. Implementation Priority

### Must-have (the paper cannot exist without these)
1. Generate E optotypes at LogMAR grid (DifferentiableStimulus — exists).
2. Compute counterfactual stimuli under eye trace conditions (make_counterfactual_stim — exists).
3. Compute population rates (compute_rate_map_batched — exists).
4. **Implement the decoding pipeline with grouped CV and the A/B/C/D ablation ladder.** This is the single most important new code.
5. Implement integration time sweep (sliding-window decoder).
6. Generate matched-budget null traces (phase randomization at minimum).

### Strong companion (should be in the paper)
7. Compute C_signal and C_FEM; task-relevant subspace SNR.
8. Representational intervention: project out FEM-aligned subspace, rerun decoding.
9. Implement sequential entropy reduction (ideal observer).
10. Trace-budget stratification analysis (bin real traces by RMS, check gain vs. budget).

### Useful secondary (include if time permits)
11. Upgrade Fisher computation from J_indep to J_pop with multiple noise models.
12. Fisher information matrix for parameter coupling geometry.
13. Characterize spatiotemporal tuning; compute resonance scores.

### Conditional (include only if results are clean and don't distract)
14. Spatiotemporal resonance overlay with spatial-frequency gain function.
15. Post-saccadic information ramp.
16. Gradient-based optimal trajectory search.

---

## 14. References

### Active sensing and FEMs
- Rucci, Iovin, Poletti & Santini (2007). Miniature eye movements enhance fine spatial detail. *Nature*, 447, 852–855.
- Ratnam, Dobie, Engbert & Rucci (2017). Benefits of retinal image motion at the limits of spatial vision. *Journal of Vision*, 17(1), 30.
- Ahissar & Arieli (2001). Figuring space by time. *Neuron*, 32(2), 185–201.
- Gruber et al. (2021). Oculo-retinal dynamics can explain the perception of minimal recognizable configurations. *PNAS*, 118(34), e2022792118.
- Ahissar et al. (2025). Closed-loop perception: gaps between artificial intelligence and biology. *Current Opinion in Behavioral Sciences*, 65, 101572.

### Neural correlates of FEMs
- McFarland, Cui & Butts (2016). Variability and correlations in primary visual cortical neurons driven by fixational eye movements. *Journal of Neuroscience*, 36(23), 6225–6241.
- Baudot et al. (2013). Animation of natural scene by virtual eye-movements evokes high precision and low noise in V1 neurons. *Frontiers in Neural Circuits*, 7, 206.
- Gur & Snodderly (2006). High response reliability of neurons in primary visual cortex (V1) of alert, trained monkeys. *Cerebral Cortex*, 16(6), 888–895.

### Information-limiting correlations and population geometry
- Moreno-Bote, Beck, Kanitscheider, Pitkow, Latham & Pouget (2014). Information-limiting correlations. *Nature Neuroscience*, 17(10), 1410–1417.
- Kanitscheider, Coen-Cagli & Pouget (2015). Origin of information-limiting noise correlations. *PNAS*, 112(50), E6973–E6982.
- Pospisil & Pillow (2025). Revisiting the high-dimensional geometry of population responses in the visual cortex. *PNAS*, 122(45), e2506535122.
- Averbeck, Latham & Pouget (2006). Neural correlations, population coding and computation. *Nature Reviews Neuroscience*, 7(5), 358–366.

### Population dimensionality
- Stringer, Pachitariu, Steinmetz, Carandini & Harris (2019). High-dimensional geometry of population responses in visual cortex. *Nature*, 571, 361–365.
- Gao & Ganguli (2015). On simplicity and complexity in the brave new world of large-scale neuroscience. *Current Opinion in Neurobiology*, 32, 148–155.

### Decoding and readout
- Graf, Kohn, Jazayeri & Movshon (2011). Decoding the activity of neuronal populations in macaque primary visual cortex. *Nature Neuroscience*, 14(2), 239–245.
- Pillow, Shlens, Paninski, Sher, Litke, Chichilnisky & Simoncelli (2008). Spatio-temporal correlations and visual signalling in a complete neuronal population. *Nature*, 454, 995–999.
- Sahani & Linden (2003). How linear are auditory cortical responses? *Advances in Neural Information Processing Systems*, 15, 301–308.

### Hyperacuity
- Westheimer (1975). Visual acuity and hyperacuity. *Investigative Ophthalmology & Visual Science*, 14(8), 570–572.

### Digital twins
- Lurz et al. (2020). Generalization in data-driven models of primary visual cortex. *BioRxiv*.
- Wang et al. (2025). Foundation model of neural activity predicts response to new stimulus types. *Nature*, 640, 470–477.

### Variational inference and neural noise
- Vafaii, Galor & Yates (2025). Brain-like variational inference via iPVAE. *Preprint*.
