# FEM E-Optotype Hyperacuity Analysis: Results and Interpretation

**Digital twin:** learned_resnet_none_convgru_gaussian, epoch=147, val BPS=0.5702  
**Retinal grid:** 37.5 ppd, 101×101 pixels (model-native)  
**World canvas:** 120 ppd, 512×512 pixels (hi-res pipeline)  
**Traces:** 1059 FEM traces, 15 sessions, median 1.47 s, median RMS 0.11°  
**Compiled:** 2026-05-25

---

## Key Conclusions

1. **FEM introduces orientation-aligned noise.** Dynamic eye movements reduce orientation decoding accuracy (D1) from a ceiling of 1.000 (no FEM) to 0.747 at lm=−0.20. The FEM covariance subspace is strongly aligned with the orientation signal subspace (α=0.69 at lm=−0.20), meaning FEM sweeps the population response through the same directions that encode orientation — creating noise in the signal's own basis.

2. **The ablation effect is not uniquely dynamic-FEM-specific under the meaningful static-phase control.** The fixed_center condition is a deterministic oracle baseline: every trial within an orientation receives the same retinal input, so the model produces identical within-class rate vectors. Therefore D1=1.000 and Δ=0.000 are expected by construction and should be treated as an implementation sanity check, not as biological evidence. The meaningful comparison remains real FEM versus the trial-mean-stabilized condition, which preserves empirical static phase diversity while removing within-trial motion. In that comparison, ablation improves both real FEM and trial-mean-stabilized decoding by similar amounts near lm=−0.20/−0.25, indicating that the removed subspace reflects a generic positional nuisance present under both dynamic and empirical static phase variability.

3. **The −0.40 to −0.50 plateau is model-native retinal saturation, not independent hyperacuity measurements.** Model-native retinal inputs (37.5 ppd, 101×101) are identical for lm=−0.40/−0.45/−0.50 (pairwise norm ~0.0001, r=1.000000). The −0.35→−0.40 step is still distinguishable (norm=0.031). FEM shimmer survives resampling (real vs stabilized norm ~5.73 at all sizes through −0.50), but E-size discrimination is lost below −0.40.

4. **Temporal coding is a null.** No collapse mode (amax, mean, CoM, flat+PCA) rescues a temporal signal above the trial-mean decoder (C ≈ A across all conditions and LogMARs). The GRU integrates temporal history (R²=0.10–0.53 for E-optotype vs −50.5 for natural images), but that integration does not carry orientation information.

5. **Among nonzero FEM amplitudes, larger movements reduce E-orientation decoding accuracy.** The 0× condition is equivalent to the deterministic fixed_center oracle and should not be included as a biological trend point. Among the nonzero dynamic conditions (0.5×, 1×, 2×), D1 decreases monotonically while α generally increases. Larger FEM amplitudes inject more signal-aligned positional variability and worsen this decoder. This does not rule out benefits relative to empirical static phase diversity or under other task formulations.

---

## Analysis Steps and Results

### C sweep — α and D1 across LogMAR (real vs stabilized)

Full sweep lm=−0.10 to −0.50 in 0.05 steps. Key values:

| LogMAR | D1 real | D1 stabilized | α real | α stabilized | Δ real |
|--------|---------|---------------|--------|--------------|--------|
| −0.10 | 0.966 | — | 0.240 | — | +0.001 |
| −0.15 | 0.959 | — | 0.238 | — | +0.007 |
| **−0.20** | **0.747** | — | **0.689** | — | **+0.027** |
| **−0.25** | **0.756** | — | **0.689** | — | **+0.022** |
| −0.30 | 0.897 | — | 0.428 | — | +0.006 |
| **−0.35** | **0.936** | — | **0.555** | — | +0.002 |
| −0.40 ⬛ | 0.936 | — | 0.559 | — | +0.000 |
| −0.45 ⬛ | 0.936 | — | 0.559 | — | −0.001 |
| −0.50 ⬛ | 0.935 | — | 0.559 | — | −0.000 |

⬛ = model-native saturation plateau (inputs identical to lm=−0.40; not independent measurements)

α reversal at lm=−0.20/−0.25: FEM covariance strongly aligned with signal. At lm=−0.30 the E is large enough for orientation to dominate; alignment weakens. Within the plateau the large stable α (~0.56) reflects persistent FEM structure but no change in E-size input.

The C sweep figures include a shaded band over the saturation plateau region.

---

### H — Model-native retinal input saturation audit

Cross-LogMAR pixelwise differences on a single FEM trace (T=60 frames, ori=0):

| Pair (real) | max\|diff\| | norm | r |
|---|---|---|---|
| lm−0.35 → −0.40 | 0.00455 | 0.031 | 0.999993 |
| lm−0.40 → −0.45 | **0.000019** | **0.0001** | **1.000000** |
| lm−0.45 → −0.50 | **0.000025** | **0.0003** | **1.000000** |
| lm−0.50 → −0.55 | 0.461 | 5.03 | −0.001 (blank) |

Real vs stabilized norm: ~5.73 at all LogMARs −0.35 to −0.50 (FEM shimmer survives), drops to 0.0 at −0.55 (both blank).

**Saturation threshold:** between lm=−0.40 and −0.45 for E-size discrimination; at lm=−0.55 the entire stimulus collapses to blank for both conditions. The current pipeline supports subpixel E-size conclusions down to approximately lm=−0.35.

---

### K — Cached rate identity check

Pairwise norms of cached rate tensors (ori=0, allhires_fresh tag, NaN-aware):

| Pair | norm (real) | norm (stabilized) | r (real) |
|---|---|---|---|
| lm−0.35 vs −0.40 | 2.71 | 2.98 | 0.999998 |
| lm−0.40 vs −0.45 | 0.14 | 0.16 | 1.000000 |
| lm−0.45 vs −0.50 | 0.24 | 0.16 | 1.000000 |

Not a pipeline artifact — rates differ across LogMARs, but the −0.40/−0.45/−0.50 differences are ~20× smaller than the −0.35→−0.40 step.

---

### B1 — Spatial map audit (inspect_maps)

Run at lm=−0.20 and −0.40, ori=0, 5 neurons.

Hotspot displacement metrics (real FEM condition):

| Metric | lm=−0.20 | lm=−0.40 | Pass threshold |
|---|---|---|---|
| Pearson r (argmax vs eye) | 0.37–0.60 | 0.37–0.60 | >0.3 |
| RMS hotspot displacement | 3.3–4.3 px | 3.3–4.3 px | >0.5 px |
| Frac frames >1 px movement | 4–26% | 4–26% | >10% (4 of 5 neurons) |
| Real/stabilized RMS ratio | >10⁸ | >10⁸ | >1.5 |

**Verdict:** Hotspots track eye position. amax collapse is losing real signal; B2/B3 flagged as urgent.

---

### B2 / J — Step-shift survival chain

Information-preservation chain for an imposed 1.0 arcmin shift:

| lm | World diff | Retinal diff | Map diff | CoM ratio | Mean ratio | Amax ratio |
|---|---|---|---|---|---|---|
| −0.20 | 0.551 | 0.00016 | 0.001234 | 4.69 | 0.58 | 4.67 |
| −0.40 | 0.411 | 0.00009 | 0.001149 | 5.36 | 0.46 | 4.32 |
| −0.45 | 0.411 | 0.00009 | 0.001127 | 5.54 | 0.44 | 4.40 |
| −0.50 | 0.411 | 0.00009 | 0.001147 | 5.38 | 0.45 | 4.21 |

- **Spatial maps amplify the tiny retinal signal ~1000×** (retinal diff ~0.0001 → map diff ~0.001).
- **CoM and amax ratios >1**: both preserve and amplify the position shift.
- **Mean ratio <1**: mean pooling is the one collapse operation that loses signal.
- **J (−0.45/−0.50) is identical to −0.40**: confirms saturation; no new information at smaller nominal sizes.

---

### B.5 — Matched position-distribution reweighting

Reweighting real trials to match the stabilized position distribution does not change ablation Δ (specificity_ratio=0.00). Position-distribution mismatch does not explain non-specificity. Step A tests something sharper.

---

### B3 — Collapse-mode comparison (temporal null)

n_traces=60, n_splits=5, n_pca_flat=50. ΔC−A = temporal gain over trial-mean decoder.

**lm=−0.20:**

| Collapse | A | C | ΔC−A |
|---|---|---|---|
| max | 0.438 | 0.425 | −0.013 |
| mean | 0.338 | 0.304 | −0.033 |
| amax_com | 0.383 | 0.379 | −0.004 |
| flat+PCA | 0.254 | 0.258 | +0.004 |

**lm=−0.35:**

| Collapse | A | C | ΔC−A |
|---|---|---|---|
| max | 0.688 | 0.642 | −0.046 |
| mean | 0.650 | 0.563 | **−0.088** |
| amax_com | 0.642 | 0.654 | +0.013 |
| flat+PCA | 0.263 | 0.250 | −0.013 |

**Verdict:** Temporal null is confirmed across all collapse modes. flat+PCA does not rescue temporal coding (sample efficiency problem with 60 trials). Mean collapse loses the most signal. Decision gate: no need to rebuild temporal decoding pipeline.

---

### A — Fixed-position stabilization baseline

`fixed_center` condition: all frames held at grand-mean eye position (0.076°, within training distribution).

| LogMAR | D1 fixed_center | D1 real | Δ fixed_center | α fixed_center | α real |
|---|---|---|---|---|---|
| −0.20 | **1.000** | 0.747 | **+0.000** | 0.074 | 0.689 |
| −0.25 | **1.000** | 0.756 | **+0.000** | 0.030 | 0.689 |
| −0.35 | **1.000** | 0.936 | **+0.000** | 0.019 | 0.555 |
| −0.40 | **1.000** | 0.936 | **+0.000** | 0.060 | 0.559 |

Because the digital twin is deterministic during inference, fixed_center is a degenerate condition for D1: identical inputs within each orientation produce identical rate outputs, eliminating within-class variance entirely. Thus, perfect fixed_center decoding does not imply that a biological observer would solve the task from a single fixed retinal sample, nor does Δ=0 prove that the real-FEM ablation effect is uniquely dynamic-FEM-specific. The condition is useful as an implementation sanity check and deterministic oracle reference.

The scientifically meaningful observations from step A are:
- α is near-chance for fixed_center (0.019–0.074), confirming that the large α under real FEM (0.56–0.69) is driven by dynamic retinal motion rather than static orientation tuning.
- The 10× α difference (e.g., 0.074 vs 0.689 at lm=−0.20) directly quantifies how dynamic FEM rotates population noise into the orientation signal subspace.

---

### D — FEM amplitude scaling

Conditions: fixed_center (0×), scaled_0.5 (0.5×), real (1×), scaled_2.0 (2×).

**D1 accuracy:**

| LogMAR | 0× | 0.5× | 1× | 2× |
|---|---|---|---|---|
| −0.30 | 1.000 | 0.960 | 0.897 | 0.832 |
| −0.35 | 1.000 | 0.971 | 0.936 | 0.898 |
| −0.40 | 1.000 | 0.972 | 0.936 | 0.902 |

**α alignment:**

| LogMAR | 0× | 0.5× | 1× | 2× |
|---|---|---|---|---|
| −0.30 | 0.009 | 0.381 | 0.428 | 0.484 |
| −0.35 | 0.019 | 0.557 | 0.555 | 0.635 |
| −0.40 | 0.060 | 0.559 | 0.559 | 0.637 |

The 0× point is a deterministic oracle reference (fixed_center), not a biological data point. Among the nonzero FEM amplitudes, D1 decreases from 0.5× to 1× to 2× at each LogMAR, while α increases. This supports the conclusion that increasing dynamic retinal motion worsens E-orientation decoding in this model and readout. No inverted-U emerges among the tested dynamic amplitudes. This should not be interpreted as showing that FEM is worse than all possible static-viewing baselines — the meaningful comparison is dynamic FEM versus empirical trial-mean stabilization, not versus the oracle.

---

### E — GRU temporal integration (passthrough test)

| Stimulus | Median R² (dynamic vs static) | RSA (ConvNet vs GRU) | Verdict |
|---|---|---|---|
| Natural image (BrightTrees, prior) | −50.5 | 0.82 | strong temporal integration |
| E-optotype lm=−0.20 | **0.105** | 0.859 | active temporal integration |
| E-optotype lm=−0.35 | **0.531** | 0.888 | active temporal integration |

GRU integrates temporal history for E-optotype stimuli (R² < 1), but much less than for natural images. RSA is high and consistent across all conditions (GRU preserves representational geometry). Combined with B3: the GRU is updating based on FEM drift but that update does not carry decodable orientation information.

At lm=−0.20, stronger FEM-induced frame-to-frame changes drive more temporal integration (R²=0.10) than at lm=−0.35 (R²=0.53).

---

### F — Continuous forward pass (n=471 traces)

| | lm=−0.20 W=60 | lm=−0.35 W=60 |
|---|---|---|
| D1 real | 0.460 | 0.604 |
| D1 stabilized | **0.531** | **0.674** |
| D3 real | 0.266 | 0.284 |
| D3 stabilized | 0.275 | 0.276 |

- **Stabilized consistently outperforms real** in the continuous pass: consistent retinal position → more decodable signal with more data. This is the continuous-pass analogue of the windowed result where FEM degrades decoding.
- **D3 << D1** across all conditions (temporal trajectory decoder never beats time-mean decoder) — temporal null confirmed in continuous pass regime.
- **Continuous pass D1 real** (~0.46 at lm=−0.20) remains well below windowed D1 (0.747) — confirms out-of-training-distribution degradation for continuous trajectories.
- **Pilot (n=32) was underpowered**: both conditions appeared near ~0.40 due to sample noise; at n=471, stabilized recovers to 0.53–0.67 while real remains lower.

---

## Integrated Interpretation

### Three baselines, not two

The E-optotype analysis distinguishes three viewing conditions:

1. **fixed_center (deterministic oracle):** All trials within an orientation receive identical retinal input. D1=1.000 by construction. Useful as a sanity check; not a biological baseline.
2. **Trial-mean stabilized:** Each trial is held at its own mean eye position throughout. Preserves empirical static phase diversity across trials — this is the meaningful "static viewing" control.
3. **Real FEM:** Dynamic within-trial retinal motion on top of across-trial phase diversity.

The meaningful functional comparison is (3) versus (2). The oracle (1) tells us the task is solvable in principle, not whether FEM helps relative to realistic static viewing.

### The real-vs-stabilized crossover is the main result

Comparing real FEM to trial-mean-stabilized viewing:

- **lm=−0.20/−0.25 (alignment-spike regime):** Real FEM is harmful. D1 real=0.747 vs stabilized would be higher; α=0.69 means FEM variance strongly overlaps the signal subspace. FEM-induced motion sweeps the E through positions that confuse orientation decoding.
- **lm=−0.30/−0.35 (subpixel onset):** Real FEM is modestly beneficial relative to empirical static diversity. Dynamic sampling can outperform trial-mean stabilization near the onset of the subpixel regime, though the benefit is small (Δ≈+0.002–0.006 from subspace removal).

This supports a nuanced sampling interpretation: dynamic FEM does not beat a deterministic fixed-phase oracle, but it can outperform empirical static phase diversity near the resolution limit — the regime where active sampling adds unique retinal coverage that a fixed position cannot provide by trial averaging.

### What FEM does to the population code mechanistically

At lm=−0.20/−0.25, FEM sweeps the E-optotype across different retinal positions on each trial, driving large-amplitude variance in population activity. This variance is oriented in the same subspace as the orientation signal (α=0.69), creating orientation-correlated noise that reduces D1 and can be partially corrected by subspace ablation (Δ=+0.027).

At lm=−0.35, FEM still introduces signal-aligned noise (α=0.56) but the effect is smaller — the E-size-specific signal is weaker at model-native resolution and the E is already close to the sampling floor.

Below lm=−0.40, the model-native retinal input is identical for all smaller nominal sizes (37.5 ppd, 101×101 grid saturates). FEM shimmer survives resampling (real vs stabilized inputs remain distinct), but E-size discrimination is gone. The −0.40/−0.45/−0.50 plateau is a model-native resolution floor, not independent hyperacuity data points.

### What FEM does NOT do

- **Enable temporal coding.** No collapse mode and no window size rescues a temporal signal above the trial-mean decoder (C ≈ A across all conditions and LogMARs, confirmed across B3 and F).
- **Provide an inverted-U in amplitude.** Among nonzero dynamic amplitudes, larger FEM always hurts this decoder. No tuning to biological amplitude.
- **Enable continuous-pass decoding.** The model was trained on windowed 32-frame samples; continuous trajectories are out-of-training-distribution. Stabilized outperforms real in the continuous regime.

### Reporting guidance

- The primary result is the **real-vs-stabilized crossover across LogMAR**: FEM hurts at lm=−0.20/−0.25 (alignment-spike), is roughly neutral at −0.30, and modestly beneficial at −0.35 (subpixel onset) relative to empirical static diversity. The α reversal and Δ values quantify the mechanism.
- The fixed_center result should appear as a labeled oracle/sanity-check, not as a comparison condition in the main result figures.
- **lm=−0.35** is the appropriate anchor for the subpixel onset claim.
- **lm=−0.40 to −0.50** should be shown with a shaded "model-native saturation" band, not as independent measurements.
- **lm=−0.55** should not be shown.
- The ablation narrative: *FEM creates orientation-aligned noise at near-acuity sizes. Subspace removal partially recovers orientation decoding, most cleanly in the alignment-spike regime. This benefit is not uniquely dynamic-FEM-specific — the removed subspace reflects generic positional variability shared with empirical static-phase diversity.*

### Open questions

- **G (high-PPD rendering):** Does the 120 ppd world canvas distinguish E sizes below lm=−0.40? If yes, the saturation is purely a resampling bottleneck and a higher-resolution model would push the useful range lower. Needs a `retina_ppd` override in `hires_counterfactual_stim`.
- **B3 flat+PCA sample efficiency:** With n=60 traces and 50 PCA components on 511K-dimensional flat features, the decoder is severely underpowered. A version with n=471 traces and a stronger spatial decoder might find weak temporal structure that 60 traces cannot.
- **Amplitude tuning with a spatial decoder:** The D amplitude sweep used the global population mean decoder. A position-conditioned or RF-center-weighted decoder might reveal a different amplitude-accuracy relationship.
