# FEM Population Coding in the V1 Digital Twin: Full Results Write-Up

*Compiled May 2026. Based on analyses in the VisionCore repository.*

---

## 1. Background and Setup

### The digital twin

The model is a learned, image-computable digital twin of foveal V1 in the marmoset, trained on 20 simultaneous recording sessions using static natural images with free-viewing eye movements (backimage protocol). Architecture: a learned temporal-convolutional frontend (16-frame kernel, 4 channels) feeds a 3D ResNet convnet (256 channels), which feeds a ConvGRU (128 hidden channels, 8 recurrent steps), whose output is read out through per-neuron Gaussian spatial masks.

Validation performance: BPS = 0.5702 (epoch 147). The model generalises from natural images to Gabor stimuli, capturing spatial RF structure but with known limitations in temporal dynamics (filters skew toward lag-0 under natural image statistics) and gain calibration across stimulus types (cross-condition affine correction improves bits/spike substantially).

Two critical architectural facts govern all downstream analyses:

**Population correlations in the model are purely reafferent.** All neurons read from a single shared feature map. There is no neuron-neuron recurrence and no independent noise. Every correlation arises because different neurons sample a shared spatiotemporal representation driven by the same eye movement over the same image. The model isolates the retinal-reafference component of V1 population correlations.

**Temporal processing is windowed, not continuous.** The GRU resets its hidden state for each 32-frame window. Each analysis time point is processed independently. Analyses assuming continuous temporal memory — velocity tracking across a fixation, long-timescale trajectory coding — are not valid under this architecture. Null results for such analyses are inconclusive rather than negative.

### Datasets

**Yates lab (Allen, Logan):** 14 Allen sessions + 16 Logan sessions; marmoset V1; fixRSVP protocol (static natural images, free-viewing fixation epochs). Recordings range from ~6 to ~125 simultaneously recorded foveal neurons per session. Used for the covariance decomposition analyses (Fig 2) and for training the digital twin.

**Rowley lab (Luke):** Multiple sessions from 2025–2026; dual Neuropixels probes recording simultaneous V1 + V2; binocular DPI eye tracking. Currently loaded and the McFarland covariance pipeline is running. A separate digital twin for this dataset is in early training. These data are the substrate for future V1→V2 hierarchy analyses.

---

## 2. Real-Data Analysis: FEM Covariance Decomposition (Fig 2)

### What was done

Across the 8 analyzed Yates V1 sessions (Allen/Logan animals), the within-fixation population covariance was decomposed into a FEM-driven component and a residual intrinsic noise component, using the McFarland (2016) framework. FEM-driven rate covariance C_rate is estimated by regressing spike count covariance against eye-position distance within each fixation epoch. The residual C_noiseC = C_total − C_rate gives the covariance unexplained by eye movements.

### Results

**FEM explains a large fraction of shared variability.** The (1 − α) metric — the fraction of total covariance attributable to FEM-driven modulation — sits at 0.67–0.78 across integration windows of 8–67 ms, well above the shuffle null (permuting spike-eye coupling within time bins collapses the estimated FEM covariance to near zero).

**FEM-corrected noise correlations are negative.** After removing the FEM-driven component, residual intrinsic noise correlations are anticorrelated across neuron pairs. This is consistent with competitive normalization or lateral inhibition structure in V1 being unmasked once shared excitatory FEM drive is removed. It should be treated as a mechanistic hypothesis rather than a direct circuit identification.

**The negative corrected correlations are real.** A systematic six-hypothesis bias diagnosis (synthetic ground-truth validation, intercept-fitting method comparisons, PSD projection asymmetry, mean-subtraction effects, Jensen's inequality in normalisation, shuffle null baselines) found no methodological artifact large enough to explain the observed Δz. Approximately 93% of the measured effect survives all bias corrections. This figure is an internal diagnostic, not manuscript-ready precision, but the qualitative finding is robust.

**Time course within fixation:** Uncorrected noise correlations decline monotonically within the fixation epoch (Spearman ρ = −0.87 across 55 time bins, p < 1e-14), dropping ~47% from early to late fixation. FEM-corrected noise correlations are flat (ρ = −0.10, ns). This confirms that FEM is the dominant source of the temporal structure in uncorrected correlations, and that the intrinsic noise structure does not vary meaningfully within a fixation.

**Multi-session replication:** The uncorrected decline is significant in 11/11 Allen sessions and 8/10 Logan sessions. The FEM correction sign-reversal (corrected correlations becoming flat or positive relative to the uncorrected negative trend) holds in 9/11 Allen sessions. Logan sessions show more variability, likely reflecting smaller population sizes (6–32 neurons vs 39–125 in Allen).

**DPI vs pupil eye tracking:** For the Rowley dataset, DPI-derived traces give substantially higher and more stable FEM fraction estimates in the near-zero distance diagnostics (~0.13–0.19 higher FEM fraction) compared to affine-transformed pupil traces. The coarse production summary makes DPI and pupil look similar; the adaptive near-zero intercept diagnostics separate them. DPI is the more reliable estimate where available.

**Subspace geometry:** FEM-induced covariance is low-dimensional (participation ratio ~2.0–2.9, consistent with 2-DOF translation). The leading FEM subspace overlaps substantially with the stimulus-driven (PSTH) subspace: X ≈ 0.692 (fraction of PSTH variance captured by the FEM subspace), Y ≈ 0.642 (fraction of FEM variance captured by the PSTH subspace). This asymmetry is consistent with FEM driving a modulation of existing stimulus-driven population patterns rather than introducing orthogonal modes.

### Interpretation

FEM creates structured, low-dimensional shared variability that dominates the positive noise correlations observed in standard analyses. Once this component is removed, the remaining correlations suggest that the intrinsic V1 circuit is anticorrelated — which makes functional sense if competitive normalization prevents neurons from co-firing in the absence of shared input. The positive noise correlations typically reported in V1 may be largely a consequence of shared FEM-driven retinal input rather than intrinsic circuit coupling.

Note on the model's role: the digital twin, whose correlations are purely reafferent by construction (shared feature map, no neuron-neuron recurrence), can account for how retinal motion generates low-dimensional shared drive. It cannot adjudicate the residual anticorrelations in real data, which require intrinsic circuit mechanisms the model does not contain. The real-data decomposition and the model analyses are therefore complementary, not interchangeable.

---

## 3. The Image-Translation Jacobian

### What was done

The Jacobian J is defined as the matrix of partial derivatives of population firing rates with respect to 2D image translation: J_ij = ∂r_i/∂x_j. It is computed via finite differencing (or forward-mode AD where implemented) at each retinal position. The central hypothesis is that FEM-induced population covariance can be predicted by a pushforward: C_FEM ≈ J · Σ_eye · J^T, where Σ_eye is the covariance of retinal displacement across traces.

Four model variants were tested, varying how J is computed (at the mean position, averaged across traces, or averaged weighted by the position histogram) and what Σ_eye is used (frame-to-frame variance vs trial-level variance vs total budget).

### Results

**Test 3 — Direction (robust, positive).**
The leading FEM covariance subspace is substantially predicted by the image-translation Jacobian direction. Alignment scores of 0.40–0.60, across all letter sizes and orientations, sit 2–4× above a noise-corrected null (null p95 ≈ 0.007). The direction result is the most reliable finding from the Jacobian analyses: the top FEM covariance mode lies approximately in the 2D subspace spanned by ∂r/∂x and ∂r/∂y. Remaining variance (~40–60% of the leading subspace) is structured, low-rank, and orthogonal to J — possibly reflecting temporal integration, nonlinearities, or finite-grid/position-averaging effects, but a direct decomposition establishing the specific source has not been run.

**Test 4 variants — Scale (conditional, regime-dependent).**

- *J_static × Σ_frame*: overpredicts by 6–490×. Uses the most sensitive position and counts all within-trial variance. Both errors compound.
- *J_eff × Σ_trial*: underpredicts by 0.003–0.05×. Averaging J over 471 actual eye traces collapses the effective gradient toward zero because off-peak image positions dominate the average.
- *J_int × Σ_total (21×21 grid)*: near-exact scale agreement at lm=−0.20 for three of four orientations (EV-ratios 1.05×, 1.85×, 1.56×). This is the best-principled version: it weights J by the actual eye position histogram, uses static traces at each grid point to isolate settled spatial gradients free from GRU transients, and accounts for the full variance budget (between-trial drift + within-trial fluctuations). At lm=−0.40 it still overpredicts by 4–11×, quantitatively consistent with the 21×21 grid being too coarse to resolve the 1–2 arcmin stroke structure of the smaller letter.
- *Two-component model*: combining within-trial and between-trial terms with a fitted scalar weight still underpredicts empirical C_FEM by 1.7–3× (EV-ratio 0.32–0.60×), with the within-trial term contributing 83–99% of the prediction. The Jacobian story is not quantitatively closed at the hyperacuity letter size.

**The 180° anomaly:** At 180°, lm=−0.20, the fine 21×21 grid undershoots (EV-ratio 0.089) while the coarser 7×7 grid overshoots. This happens because the spatial gradient alternates sign at sub-bin scale for this configuration, causing histogram-weighted averaging to cancel large Jacobians that point in opposite directions. It is mechanistically explained rather than anomalous.

**Test 6 — Functional consequence (positive, with caveats).**
Subtracting the stimulus-specific Jacobian direction from each trial's population response, before decoding orientation, raises accuracy to ~100% at both letter sizes (from 74.7% and 93.6% baseline). Subtracting a pooled (non-orientation-specific) Jacobian leaves decoding essentially unchanged. This demonstrates strong entanglement between stimulus-specific position sensitivity and orientation readout. However, class-specific Jacobian subtraction may partially create an analysis-induced class transformation. The pooled-subspace control is the more interpretable causal test: the fact that it does not rescue decoding suggests the Jacobian subspace is a generic positional nuisance rather than a uniquely orientation-informative direction.

### Interpretation

The Jacobian pushforward model works in direction, robustly and across conditions. Scale is recoverable in the easier size regime with careful position-averaged estimation, but remains unresolved at hyperacuity scale and depends on finer spatial sampling than current grid resolution provides. The functional consequence — that Jacobian-like position sensitivity is entangled with identity readout — is supported by Test 6 and by the displacement decoding results (see Section 5). The two-component scale underprediction suggests additional structure beyond the first-order spatial Jacobian: likely the GRU temporal integration that redistributes the stimulus-driven signal across the 32-frame context.

---

## 4. Orientation Decoding and the FEM Crossover

### What was done

The digital twin was used to generate population responses to the Tumbling E optotype at a range of LogMAR sizes (lm = +1.00 to −0.55), under two conditions: real fixational eye movements (drawn from empirical eye traces) and stabilized (each trial held at its own mean eye position). Four decoders were compared: Model A (time-averaged population rate), Model C (Model A plus temporal residual PCA features), D2a (eye-state conditioned), and D3 (supervised temporal trajectory). A high-resolution E rendering pipeline was validated and used throughout (hires_threshold=2.0).

### Results

**The crossover is real and survives all-hires rerun.**

| LogMAR | Stabilized A | Real A | FEM effect |
|--------|-------------|--------|-----------|
| −0.20 | 0.774 | 0.658 | −0.116 (FEM hurts) |
| −0.25 | 0.777 | 0.658 | −0.119 (FEM hurts) |
| −0.30 | 0.882 | 0.813 | −0.069 (FEM hurts) |
| −0.35 | 0.837 | 0.890 | +0.053 (FEM helps) |
| −0.40 | 0.841 | 0.892 | +0.051 (FEM helps) |
| −0.45 | 0.840 | 0.892 | +0.052 (FEM helps) |
| −0.50 | 0.841 | 0.892 | +0.051 (FEM helps) |

The crossover occurs at approximately lm = −0.32. Above this, a single fixation position is sufficient to resolve orientation; FEM-induced position variability contaminates an otherwise discriminable signal. Below it, evidence at any single stabilized position is weaker or less uniformly reliable, and sampling multiple nearby retinal positions improves the time-averaged rate evidence.

Both conditions approach chance by lm = −0.55. This is a rendering/discretization floor — the E becomes indistinguishable at this scale under the current pixel grid without anti-aliased subpixel rendering.

**The non-monotonicity at lm = −0.20/−0.25 is real.** Performance dips then recovers before the crossover. The all-hires rerun confirms this is not a mixed-pipeline artifact; it likely reflects E-optotype discretization near the Nyquist limit of the pixel grid. The fitted ΔLogMAR = 0.002 from the threshold-fitting routine is misleading because the fitting assumes a monotone psychometric curve. The per-operating-point table is the correct summary.

**The benefit comes through the simplest readout.** Model A (time-averaged rate) is sufficient to show the crossover. Model C (temporal residuals), D2a (eye-state conditioning), and MLP decoders do not improve on Model A. The FEM advantage at hyperacuity is captured by time-averaged rates alone — no temporal sequence features, no knowledge of eye position at readout.

**The integration-time sweep confirms accumulation, not temporal structure.** At lm = −0.20, stabilized accuracy is flat across integration windows (~0.77 at W=1 through W=60), while real FEM recovers from near-chance at W=1 (0.296) to 0.646 at W=60. This pattern is consistent with FEM sampling diverse retinal positions — each contributing independent orientation evidence that accumulates — rather than with temporal structure within the trace carrying additional information.

### Interpretation

FEM helps at hyperacuity by expanding the effective spatial sample from a single fixation point to a distribution of nearby positions. At each position the signal is weak; integrated over the distribution it becomes sufficient. This is a first-order, mean-rate account. The gain does not require, and does not demonstrate, any temporal correlation code.

---

## 5. Displacement Decoding

### What was done

Population responses were computed on an 11×11 grid of eye positions spanning ±0.05° (121 positions, 6 natural images). Ridge regression was trained to decode retinal displacement (δx, δy) from response differences Δr = r(p2) − r(p1), using four feature sets: scalar rates, CoM features, width features, and combined moments. Two cross-validation schemes: within-image (5-fold) and leave-one-image-out.

### Results

**Within-image: near-perfect decoding.**

| Feature set | Mean R² |
|---|---|
| Scalar rates | 0.9975 |
| CoM features | 0.9984 |
| Width features | 0.9988 |
| Moments combined | 0.9991 |

Every small retinal shift (down to 0.01°) produces a discriminable population response change. Scalar rates perform as well as spatial moments — CoM features add nothing over simple rate differences at this scale.

**Cross-image: catastrophic failure.** Leave-one-image-out R² ≈ −1.3 (mean across held-out images and feature sets). The decoder trained on five images actively anti-generalises to the sixth. Spatial moments (CoM) generalise *worse* than scalar rates, suggesting they are more entangled with image-specific content.

**Displacement magnitude sweep:** Decoding is reliable across the full FEM range (±0.05°), with a modest dip at 0.02° (likely a transition artifact rather than real non-monotonicity).

### Interpretation

The population encodes retinal displacement at very high fidelity within any given image. But the direction in population space that encodes displacement is determined entirely by which neurons are activated and how their receptive fields interact with local image structure. There is no image-invariant displacement code. A downstream decoder cannot read out absolute or relative eye position without image-specific calibration or a nonlinear readout.

This finding is the displacement-decoding face of the same phenomenon the Jacobian captures: the sensitivity of the population to retinal translation is a local, image-specific property. It also explains why the FEM covariance subspace does not rotate with E orientation (see Section 6) — four rotations of the same letter at the same scale share so much global spatial structure that their displacement responses are nearly identical.

---

## 6. FEM Covariance Geometry (Priority 1)

### What was done

For each E orientation (0°, 90°, 180°, 270°), the FEM covariance matrix C_FEM^k was computed from the spread of time-averaged rates across 471 eye traces. Top-2 eigenvectors U_FEM^k were extracted. Three tests: (1) pairwise subspace overlaps between all four U_FEM^k (does the FEM subspace rotate with orientation?); (2) signal alignment α^k = tr(U_FEM^k^T C_signal U_FEM^k) / tr(C_signal) (does FEM noise fall in the orientation signal direction?); (3) second-order decoder classifying orientation by which FEM subspace best captures each trial's residual response.

Tested at lm = −0.20 (FEM hurts D1) and lm = −0.40 (FEM helps D1). M = 471 trials/orientation, N = 756 neurons.

### Results

| Metric | −0.20 real | −0.40 real | −0.20 stab | −0.40 stab |
|--------|-----------|-----------|-----------|-----------|
| Off-diagonal overlap | 0.9998 | 0.9995 | 0.9395 | 0.9951 |
| Mean α | 0.720 | 0.558 | 0.607 | 0.665 |
| D1 accuracy | 0.746 | 0.936 | 0.766 | 0.840 |
| Covariance decoder | 0.263 | 0.333 | 0.288 | 0.317 |
| Combined − D1 | +0.001 | +0.000 | +0.000 | +0.001 |

**Subspace rotation: no support.** Real-FEM subspaces are nearly orientation-invariant (off-diagonal overlap ~1.0) at both LogMARs — more uniform than the stabilized baseline at −0.20. The FEM subspace represents displacement directions in response space, a property of RF geometry relative to the letter envelope rather than letter identity. Four rotations of the same E at the same scale share enough global spatial structure that their displacement responses are essentially identical. Covariance decoder is near chance at both LogMARs.

**Alignment transition: confirmed.** α is higher at lm = −0.20 (0.720) than at lm = −0.40 (0.558) in the real condition. The stabilized condition shows the *opposite* ordering (0.607 at −0.20, 0.665 at −0.40). This reversal is important: under real FEM, the noise becomes less aligned with the signal at hyperacuity; under stabilization, it becomes more aligned. This is not a simple "FEM noise decreases at hyperacuity" story — it is a qualitative reversal in the regime dependence between conditions.

**C_signal diagnostic.** The orientation signal eigenvalues are *larger* at lm = −0.40 [2.24e−04, 1.50e−05] than at lm = −0.20 [3.9e−05, 6.0e−06]. The class means are more separated at hyperacuity, even though α is lower. This supports "signal subspace moved away from the FEM translation directions" as the mechanism for the α decrease — the orientation-discriminative signal expands relative to the nuisance direction at hyperacuity — rather than "FEM covariance became informative."

**What this rules out.** Orientation information does not migrate from mean rate to FEM covariance geometry at hyperacuity. The covariance-code hypothesis for the crossover is not supported.

### Interpretation

The α reversal is the most striking observation from this analysis and still lacks a fully closed explanation. The C_signal diagnostic supports the signal-moved interpretation, but this does not by itself explain why the stabilized α moves in the opposite direction. That asymmetry may require decomposing how the across-trial position distribution changes with LogMAR differently in the real vs stabilized conditions. This is a tractable computation but has not yet been done.

---

## 7. Global FEM Subspace Ablation (Priority 2)

### What was done

The pooled top-2 FEM covariance subspace U_FEM (fit across all orientations) was projected out from each trial's time-averaged rate vector before rerunning D1 decoding. Tested at lm = −0.20 and lm = −0.40, for both real and stabilized conditions. M = 471, N = 756. Rate file tag: allhires_fresh.

### Results

Real condition:

| LogMAR | α | D1 original | D1 cleaned | Δ |
|--------|---|-------------|------------|---|
| −0.20 | 0.689 | 0.747 | 0.773 | +0.027 |
| −0.40 | 0.559 | 0.936 | 0.936 | +0.000 |

Stabilized control:

| LogMAR | α | D1 original | D1 cleaned | Δ |
|--------|---|-------------|------------|---|
| −0.20 | 0.052 | 0.774 | 0.803 | +0.029 |
| −0.40 | 0.666 | 0.840 | 0.855 | +0.015 |

*(Note: the α values here are the pooled-ablation alignment metric computed from the pooled within-orientation covariance, and are not directly comparable to the per-orientation α values reported in Section 6, which use a different pooling and normalisation.)*

**Finding.** The real-condition ordering matches the α prediction: removing U_FEM helps at −0.20 and is null at −0.40. But the stabilized control improves by a similar amount at −0.20 (+0.029), and by a smaller but non-zero amount at −0.40 (+0.015). The ablation is removing a shared positional nuisance present in both conditions, not a covariance component unique to dynamic FEMs. The stabilized condition in this pipeline does not fully flatten retinal position — each trial is held at its own mean eye position, so across trials there is still a low-rank retinal translation distribution.

### Interpretation

The pooled ablation result is consistent with the alignment story but does not prove causality. What it establishes is that the shared translation-like nuisance subspace is information-limiting at −0.20 and not at −0.40, which matches the α values and the C_signal eigenspectrum. It does not establish that dynamic FEM correlations specifically, as opposed to static positional diversity, are the causal mechanism.

The clean causal test requires a true fixed-position stabilization baseline — every trial rendered at the same single retinal position, removing across-trial positional variance in the stabilized condition. This would allow the ablation to isolate dynamic FEM covariance from static position dispersion.

---

## 8. Differential Covariance Ablation

### What was done

To isolate variance unique to dynamic FEMs, the positive eigenspace of (C_real − C_stabilized) was computed within each CV fold and projected out before rerunning D1. This targets only the covariance that exists in real FEM traces but not in stabilized traces.

### Results

| LogMAR | Mean positive eigvals | Real Δ | Stabilized Δ |
|---|---|---|---|
| −0.20 | [0.003668, 0.001265] | +0.016 | +0.017 |
| −0.40 | [0.000859, 0.000293] | +0.002 | −0.002 |

**Finding.** At −0.20, ablating the differential subspace improves both real and stabilized by essentially equal amounts. At −0.40, the effect is null in both directions. Even after subtracting the stabilized covariance, the top differential directions behave like a shared nuisance axis, not a uniquely dynamic-FEM correlation mode.

### Interpretation

The covariance-ablation route has been tested in two forms — pooled C_FEM and differential C_real − C_stabilized — and neither yields a clean real-FEM-specific causal component. The crossover is dominated by first-order spatial sampling in the mean-rate code. Covariance structure is at most a byproduct of the same position-sensitivity that drives the sampling benefit.

---

## 9. Temporal Residual Coding

### 9a. Model C vs Model A

Across the full LogMAR sweep, Model C (temporal residual PCA features) does not improve over Model A (time-averaged rates) at any operating point. At lm = −0.20, temporal features actively hurt (Model C ≈ 0.617 vs Model A ≈ 0.658 for real FEM). At large LogMAR (easy regime), both are near ceiling and equivalent.

### 9b. Motion-magnitude binning

Stimuli were binned by retinal motion RMS and FEM gain (acc(C) − acc(A)) was computed per bin:

| Motion bin | RMS (mean) | n stimuli | acc(A) | acc(C) | Gain |
|---|---|---|---|---|---|
| High | 0.222° | 7 | 1.000 | 0.500 | −0.500 |
| Medium | 0.139° | 6 | 0.875 | 0.656 | −0.219 |
| Low | 0.057° | 7 | 0.938 | 0.594 | −0.344 |

Gain is negative at all motion levels. The high-motion bin is particularly informative: RMS = 0.222°, which is large enough that lack of eye movement cannot explain the null — acc(A) = 1.000, yet Model C scores only 0.500, near chance for a 4-way task. Temporal residual features do not just fail to help; they actively hurt decoding regardless of how large the eye movements are.

### 9c. Continuous forward pass (preliminary)

The model was run in a single continuous forward pass (GRU state carried across the full ~476-frame trial), removing the window-reset limitation of the standard pipeline. 32-trace pilot at lm = −0.40.

| Condition | D1 W=1 | D1 W=24 | D1 W=60 | D3 W=60 |
|---|---|---|---|---|
| Real (continuous) | 0.380 | 0.355 | 0.414 | 0.264 |
| Stabilized (continuous) | 0.415 | 0.445 | 0.400 | 0.302 |

For comparison, the same 32 traces run through the standard windowed pipeline still show the expected crossover (real D1 W=60 = 0.611, stabilized = 0.461). The continuous pass degrades both conditions toward ~0.40 and leaves D3 below D1. This is consistent with out-of-training-distribution degradation rather than hidden temporal integration.

**Important caveat:** The temporal null result across all three tests (Model C, motion binning, continuous pass) shares a methodological risk identified in the temporal decoding diagnostic plan. The model returns a spatial rate map (B, N, H, W) per neuron, and the current pipeline collapses this with `amax` before decoding. If FEM expresses temporal information as movement of a response hotspot across the spatial readout map, `amax` collapse would destroy this signal before any decoder sees it. This concern has not been resolved by a direct diagnostic. The step-shift sensitivity test, spatial map visualisation, and collapse-mode comparison recommended in the diagnostic plan have not yet been run.

### Interpretation

The current scalar-rate temporal residual story looks bad: three independent negative results with different methodological leverage. But the spatial-map temporal code — where FEM shifts hotspots rather than modulating peak rates — has not been directly tested. The current C ≈ A conclusion should be described as "strongly constrained under the current collapsed-rate representation" rather than "temporal coding is absent." The next step is a direct spatial-map audit, not another decoder variant on the same collapsed features.

---

## 10. Transformation Dynamics

### What was done

A linear dynamical system z(t+1) = A·z(t) + B·v(t) + c was fit to the 2D PCA-projected population latent state z, where v(t) is eye velocity. B was tested for explanatory power. Separately, eye velocity was decoded directly from Δz (change in latent state) and from mean rates.

### Results

Across 6 natural images: A (autoregressive term) explains ~92% of latent variance; B (eye velocity input) explains < 0.05% — less than 1/5000th of total variance. Spectral radius of A ≈ 0.96, indicating slow attractor dynamics with time constant ~25 frames. Velocity is not decodable from Δz or from mean rates (R² ≈ 0 in both directions). Mean rates perfectly decode stimulus identity (R² = 1.0); Δz decodes identity at R² = 0.22.

These nulls should be treated as inconclusive given the windowed architecture: the 2D PCA latent space discards most spatial structure, and the GRU has no inter-window memory. A continuous-state model or spatial readout would need to be tested before claiming biological absence of velocity coding.

---

## 11. Where Things Stand: Integrated Status Table

| Claim | Status | Key evidence |
|---|---|---|
| FEMs improve orientation discrimination at hyperacuity | **Established** | D1 crossover at −0.35 to −0.50, ~5-point advantage, replicated in all-hires rerun |
| The benefit is through time-averaged rate (spatial sampling) | **Established** | D3 ≈ chance, D2a ≈ D1, MLP ≈ D1; D1 with longer accumulation windows is sufficient |
| Response manifold is smooth, image-specific, locally linear | **Established** | Displacement R² ≈ 0.998 within-image, −1.3 cross-image |
| FEM-induced variability is dominant low-dimensional, translation-like | **Established** | Participation ratio ~2.0–2.9; Jacobian direction alignment 0.40–0.60, 2–4× above null |
| Jacobian predicts leading FEM covariance direction | **Established** | Robust across letter sizes and orientations |
| Jacobian predicts FEM covariance scale | **Partial** | Near-exact at lm=−0.20 with J_int (21×21 grid); unresolved at lm=−0.40 |
| FEM explains dominant fraction of shared V1 variability (real data) | **Established** | (1−α) ≈ 0.67–0.78, shuffle null confirmed, bias diagnosis passed |
| FEM-corrected noise correlations are negative | **Established** | Systematic in Allen sessions; bias diagnosis rules out artifacts |
| All model correlations are purely reafferent | **Established** | Architecture: shared core, no neuron-neuron interaction |
| FEM covariance limits decoding at −0.20, neutral at −0.40 | **Confirmed descriptively** | α = 0.720 vs 0.558; ablation helps at −0.20 only in real condition |
| Ablation effect is specific to dynamic FEM correlations | **Not established** | Stabilized control improves similarly; differential ablation also non-specific |
| FEM covariance subspace rotates with E orientation | **No support in current tests** | Off-diagonal overlap ~1.0 at both LogMARs; covariance decoder near chance |
| Orientation info migrates from mean rate to covariance geometry | **Not supported** | Covariance decoder near chance; combined decoder no gain over D1 |
| Temporal residual features help orientation decoding | **Not supported (with caveat)** | D3 null, motion binning null, continuous pass pilot null; spatial-map collapse not yet audited |
| GRU temporal processing is orientation-task relevant | **Unknown** | Architecture has within-window state; decoders have not shown useful temporal residuals |
| Velocity/transformation decodable with continuous GRU | **Unknown** | Previous nulls reflect windowed architecture; continuous pass pending full run |
| True fixed-position stabilization isolates dynamic FEM | **Not yet tested** | Current stabilized condition still has across-trial position variability |
| Spatial-map temporal code (hotspot movement) | **Not tested** | Spatial collapse diagnostic plan identified but not yet executed |

---

## 12. Next Steps

### Highest priority: two tractable experiments that resolve the main remaining ambiguities

**A. True fixed-position stabilization baseline**

Replace the current stabilized condition — each trial at its own mean position, so there is still across-trial position variation — with a single fixed retinal position across all trials. This removes the residual positional variance that currently makes the stabilized control indistinguishable from real FEM under covariance ablation. With a true fixed-position baseline:

- The pooled and differential ablations can be rerun; if real condition still improves and stabilized does not, the effect is genuinely dynamic-FEM-specific.
- The α comparison becomes cleaner: stabilized α would reflect purely within-trial variability rather than a mix of within- and across-trial position distributions.
- The α reversal between conditions (real drops at hyperacuity, stabilized currently rises) may be resolved: some or all of the stabilized α increase at −0.40 could reflect the wider across-trial position distribution at that LogMAR.

This is a rendering change, not a new experiment. The differentiable stimulus pipeline can produce this directly.

**B. Spatial-map audit and collapse-mode comparison**

Before drawing any further conclusions from C ≈ A, verify that the collapsed-rate representation is capable of seeing what it claims to test. Concretely:

1. *Hotspot visualisation.* Run the model on one trial under real FEM and stabilized. Plot the spatial rate map (H, W) for each neuron at 5–10 timepoints. Does the peak of the response map move over time in the real condition but not in stabilized?

2. *Step-shift sensitivity test.* Present a static stimulus, then apply a single-step shift of ±1 pixel at time t=60. Measure population response at the raw map level and after `amax` collapse. If the shift is visible in raw maps but disappears after collapse, `amax` is the bottleneck.

3. *Collapse-mode comparison.* Re-run a small decoding experiment (e.g., 30 trials per orientation, lm = −0.40) with four collapse modes: `max`, `mean`, `flat` (spatially concatenated), and CoM-like spatial moments. If C ≈ A persists across all modes, the null becomes much more credible. If `flat` or CoM rescues Model C, the original null was a representation artifact.

This audit is a measurement verification exercise, not a new scientific question. The code infrastructure already exists.

### Secondary: completing planned analyses

**C. α reversal mechanism.** Compute the signal covariance C_signal at a range of LogMARs (not just −0.20 and −0.40) to track how the orientation-discriminative subspace moves relative to the FEM translation subspace as the letter shrinks. Does the signal subspace drift away from the translation direction monotonically? Does the reversal in the stabilized condition track a different trajectory? This is a cheap computation on cached rates.

**D. FEM amplitude scaling at hyperacuity.** Test 0.5× and 2.0× amplitude FEM conditions at lm = −0.40. Does D1 accuracy form an inverted-U in FEM amplitude, with 1.0× near the peak? This would connect the biological FEM amplitude to an optimality claim.

**E. GRU passthrough at hyperacuity LogMAR.** The existing GRU passthrough test (median R² dynamic vs static ≈ −50.5, RSA = 0.82) was run on natural images. Running it on the E-optotype at lm = −0.40 tests whether temporal history matters more or less for a near-threshold stimulus. If the GRU's temporal processing is more impactful near the resolution limit, that would connect the temporal architecture to the crossover even if it does not support a trajectory code.

**F. Continuous forward pass full run.** The 32-trace pilot is suggestive but not definitive. A full continuous-pass run at lm = −0.40 (all traces, not just 32) would establish whether the pilot's negative result holds at scale or was a sampling artifact.

**G. V1→V2 hierarchy (Rowley data).** Once a digital twin is fit to the Rowley dataset, the covariance decomposition, Jacobian, and alignment analyses can be run on V2. A key hypothesis is that FEM-induced covariance becomes more orthogonal to stimulus tuning higher in the hierarchy — position and identity separating progressively. The Rowley dataset enables this test directly.

---

## 13. Narrative Summary

FEMs push foveal V1 populations along an image-specific, low-dimensional translation manifold. The image-translation Jacobian predicts the direction of this manifold reliably, and scale is recoverable with the right position-weighted estimator for stimuli at the larger letter size. Within any given image, every small retinal shift produces a near-perfectly discriminable population response change; across images, the direction of this displacement code is entirely image-dependent.

This displacement mode is large enough to shape orientation readout. Above threshold (lm = −0.20), the FEM covariance subspace overlaps substantially with the orientation-discriminative signal subspace (α = 0.720), and removing the shared translation nuisance improves decoding by ~3 percentage points. At hyperacuity (lm = −0.40), the orientation signal expands and moves away from the FEM translation subspace (α = 0.558, larger C_signal eigenvalues), so FEM no longer limits decoding and instead helps by expanding the retinal sample — accumulating independent orientation evidence across the fixation.

Attempts to push the story further — into covariance geometry as an independent orientation code, or into temporal correlation structure carrying information beyond mean rates — have mostly returned negative or non-specific results. FEM covariance subspaces do not rotate meaningfully with E orientation. Covariance decoders perform near chance. Pooled and differential ablations improve decoding without isolating a real-FEM-specific causal component. Temporal residual features actively hurt decoding, and the continuous-pass pilot does not recover them. The cleanest current account of the crossover is first-order spatial sampling plus an alignment transition: the FEM translation nuisance is more damaging when the signal it overlaps with is already strong and concentrated, and less damaging when the signal has expanded and moved.

The main open questions are: (1) whether the temporal coding null is genuine or a representation-pipeline artifact from spatial-map collapse, which requires a direct audit; (2) whether the crossover can be turned into a cleaner causal story with a true fixed-position stabilization baseline; and (3) what happens to all of this in V2, where a key hypothesis is that FEM-induced variability becomes more orthogonal to stimulus tuning.
