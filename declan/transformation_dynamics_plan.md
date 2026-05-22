# Transformation Dynamics in the V1 Digital Twin: Refined Analysis Plan

## How FEM-Induced Population Trajectories Encode Retinal Translation

---

## 0. The Central Claim and What It Takes to Support It

### What we want to show

Fixational eye movements induce structured, low-dimensional population dynamics in V1 whose temporal evolution (Δz) makes transformation variables (retinal translation) more accessible than static population state does, while static features make content variables more accessible than dynamics do. This would establish that content and transformation are dissociated across different statistics of population activity — not that either is exclusively "encoded" in one feature class, but that each is disproportionately readable from a specific representational format.

**Scope of this plan:** this analysis establishes the representational substrate — that transformation variables are structured and accessible in population dynamics. A later analysis will test the downstream functional consequence: whether this transformation information supports grouping, binding, or segmentation of shared causes.

### What we must rule out

The analysis must go beyond showing that "a system driven by u contains information about u." Any deterministic system driven by smooth input will produce outputs from which the input can be recovered — that is not a representational claim. To establish a genuine representational dissociation, we need:

1. **Specificity**: Δz predicts transformation variables but not content variables, and vice versa for static state features.
2. **Non-triviality**: the result does not hold for arbitrary projections of the population response — the specific low-dimensional subspace matters.
3. **Generality**: the transformation readout works across stimuli, not just within a single image.
4. **Necessity of dynamics**: temporal structure (not just instantaneous rates at each time step) carries the transformation information.

### What we already know from existing analyses

From `translation_covariance.py` and the temporal decoding pipeline:

- FEM-induced covariance is extremely low-rank: top 2 PCs capture ~97% of FEM variance (consistent with 2 DOF translation)
- The eigenvalue spectra match the prediction: approximately rank-2 with sharp falloff
- Within-stimulus PCA stability is high (mean 0.942)
- Cross-stimulus PCA alignment is moderate (off-diagonal mean cos² = 0.369) — the subspace is substantially stimulus-dependent
- Regression from filtered eye position captures the subspace well (best_reg scores 0.5–0.99)
- C ≈ A for orientation decoding: temporal residuals do not add to content discrimination
- Spatial rate maps move under real FEM (CoM variance is large)

---

## 1. Analysis Architecture

Three analyses, ordered by evidential strength (run in this order):

| Analysis | What it tests | If it fails |
|----------|--------------|-------------|
| **A. Δz readout with controls** | Do population dynamics encode transformation? Is this non-trivial and specific? | Stop: no dynamical transformation code |
| **B. Content-transformation dissociation** | Are content and transformation separable across statistics? | Weaker claim: both in first-order statistics |
| **C. Flow fields and Model A** | What are the dynamics? Linear? Recurrent? Input-driven? | Descriptive, not fatal |

Analysis A is the killer test. If it fails decisively, B and C become descriptive exercises. If A passes, B provides the dissociation, and C provides mechanistic interpretation.

---

## 2. Analysis A: Δz Readout with Decisive Controls

### 2.1 Core test

**Target**: u(t) = [vel_x(t), vel_y(t)] — instantaneous eye velocity in degrees/frame.

**Predictors** (compared head-to-head):
- **z(t)**: 2D projection of δr(t) onto U_s — the static state in the translation subspace
- **Δz(t)**: z(t+1) − z(t) — the state increment
- **[z(t), Δz(t)]**: combined static + dynamic features

**Basis choice**: U_pca2 (top 2 eigenvectors of Σ_FEM) as the primary basis. This is the non-circular choice — PCA does not use eye position labels, so decoding velocity from PCA-derived Δz is a genuine test. U_grad2 (regression-derived) is a ceiling/consistency check only, reported separately and explicitly flagged as partially circular.

**Decoding model**: ridge regression for continuous u(t); multinomial logistic for direction bins (8 bins).

**Cross-validation**: by trace (train on traces 1..k, test on k+1..M for the same stimulus).

### 2.2 Control 1: Time-shuffle (necessity of temporal order)

Shuffle time indices within each trace independently per neuron. This preserves the marginal distribution of z but destroys temporal order. Δz readout should collapse to chance. If it doesn't, the readout is exploiting distributional features, not dynamics.

### 2.3 Control 2: Random projection baseline (non-triviality)

**This is the most important missing control in the original plan.**

Instead of projecting onto U_pca2, project δr(t) onto random 2D subspaces:
- Sample Q ∈ ℝ^{N×2} with orthonormal columns uniformly from the Stiefel manifold (QR decomposition of random Gaussian matrix)
- Compute z_rand(t) = Qᵀ δr(t) and Δz_rand(t) = z_rand(t+1) − z_rand(t)
- Decode u(t) from Δz_rand
- Repeat for 100–200 random projections

**The test**: does the U_pca2 Δz readout significantly outperform the distribution of random-projection Δz readouts? If the R² from U_pca2 falls within the distribution of random R² values, then ANY 2D projection of the population carries transformation information in its dynamics, and the specific subspace doesn't matter. That would mean the result is a trivial property of the driven system, not a representational finding.

If U_pca2 Δz readout significantly exceeds the random-projection distribution (e.g., above the 95th percentile), then the specific FEM-aligned subspace carries disproportionate transformation information — upgrading the result from "trivial driven-system property" to "structured low-dimensional transformation-sensitive subspace." Note: this does not yet establish a "canonical transformation code" — it establishes that the FEM covariance-aligned directions are special relative to arbitrary projections, which may partly reflect variance-maximization rather than a dedicated representational axis. The cross-stimulus generalization test (Section 2.6) further adjudicates this.

**Report**: R² for U_pca2 vs. distribution of R² for random projections (histogram + percentile).

### 2.4 Control 3: Matched dimensionality from shuffled covariance (PCA-specific non-triviality)

A subtler version of Control 2: compute PCA on a shuffled version of δr (shuffle neurons independently across time to destroy population structure while preserving per-neuron variance). The top 2 PCs of this shuffled covariance give a 2D subspace that captures variance but not structured FEM covariance. Compare Δz readout from shuffled-PCA basis vs. real-PCA basis.

### 2.5 Control 4: Specificity (Δz does NOT predict unrelated variables)

Decode stimulus identity (which image?) from Δz. The prediction: Δz should be poor at content decoding (content is in z or mean rate, not dynamics). If Δz also predicts content well, the dissociation is weaker.

Additionally, decode a well-matched nuisance target from Δz — a variable that shares the scale, smoothness, and autocorrelation of velocity but is irrelevant to the transformation hypothesis:
- **Phase-randomized velocity**: FFT the real velocity trace, randomize phases, inverse FFT. This produces a surrogate with the same power spectrum but no relationship to the actual eye movement.
- **Velocity from a mismatched trace**: use the velocity from a different trial (breaking the z-u coupling while preserving velocity statistics).
- **Orthogonalized surrogate**: regress out the real velocity from a synthetic smooth 2D signal with matched autocorrelation, leaving only the component orthogonal to the true transformation.

Avoid time-within-trial as a nuisance target — too many things co-vary with it in recurrent models, making it uninformative as a specificity control.

### 2.6 Cross-stimulus generalization (generality)

**This should be primary, not optional.**

- **Within-stimulus**: train Δz readout on traces from stimulus A, test on held-out traces from stimulus A. This works by construction if the system is driven.
- **Across-stimulus with stimulus-specific basis**: train on stimulus A using U_pca2(A), test on stimulus B using U_pca2(B). This tests whether the readout *rule* generalizes even though the subspace is stimulus-specific.
- **Across-stimulus with shared basis (dimensionality curve)**: learn a single canonical subspace of dimensionality d from a training set of stimuli. Apply it to held-out stimuli. Sweep d ∈ {2, 4, 6, 8, 10, 15, 20} and plot cross-stimulus R² as a function of d.

The existing data shows off-diagonal PCA alignment ~0.37 for 2D subspaces. A 2D shared basis will likely fail — different images recruit different neurons, so the per-stimulus translation generators point in different directions. The informative question is not "does a 2D universal basis work?" (probably not) but "how many shared dimensions are needed before cross-stimulus generalization stabilizes?" That dimensionality curve is the real result:

- If generalization stabilizes at d ≈ 4–6: a modest shared subspace captures the universal transformation structure, with per-stimulus variation confined to a low-dimensional complement.
- If generalization keeps improving up to d ≈ 20–30: the transformation code is high-dimensional and stimulus-entangled, weakening the "low-dimensional shared cause" interpretation.
- If generalization never exceeds chance: the transformation encoding is entirely stimulus-specific. Still consistent with "binding by shared transformation" (neurons responding to the same image show coherent modulation), but the universality claim is weakened.

Report the curve, not just a pass/fail at one dimensionality.

### 2.7 Summary of what constitutes a strong result for Analysis A

| Test | Strong result | Weak/ambiguous result |
|------|--------------|----------------------|
| Δz > z for velocity | Δz R² >> z R² | Both similar |
| Time-shuffle | Δz collapses to ~0 | Δz partially survives |
| Random projection | U_pca2 >> 95th percentile of random | U_pca2 within random distribution |
| Specificity | Δz poor for content, good for velocity; poor for surrogate velocity | Δz decodes everything |
| Cross-stimulus (dim curve) | Generalization stabilizes at d ≈ 4–6 | Generalization never exceeds chance, or requires d > 20 |

All five should point the same way. If the random-projection control fails (U_pca2 is not special), the rest of the analysis becomes descriptive rather than representational. If cross-stimulus generalization requires very high dimensionality, the "low-dimensional shared cause" interpretation is weakened but the per-stimulus transformation-sensitivity finding still stands.

---

## 3. Analysis B: Content-Transformation Dissociation

### 3.1 The double dissociation

This is the clean test of the binding/transformation framing from the motivation document. Define:

**Content labels**: stimulus identity (which image) or, for E optotypes, orientation (0°/90°/180°/270°)

**Transformation labels**: eye velocity direction (8 bins) or continuous [vel_x, vel_y]

**Feature sets (staged presentation)**:

*Primary dissociation (present first):*
- **First-order static**: time-averaged δr (or time-averaged z) — the "rate code"
- **First-order dynamic**: Δz(t) — instantaneous state changes

*Extension (present second, only if clean):*
- **Second-order temporal**: lagged cross-covariance of δr within a time window (the genuine correlation-structure test)

The prediction for the primary dissociation:

| | Content | Transformation |
|---|---------|---------------|
| Static (mean rate) | **High** | Low |
| Dynamic (Δz) | Low | **High** |

Present this clean 2×2 first. The asymmetry is the result.

The second-order extension asks whether lagged covariance structure predicts transformation variables beyond what Δz captures. This is the only test that directly addresses the "correlation code" question. But if the covariance features are weak, noisy, or sample-hungry (which is likely), present them as a targeted extension rather than a co-equal entry in the main dissociation. Do not let a noisy second-order cell dominate attention or blur the primary first-order dissociation.

### 3.2 Why Δz is not truly second-order

An important conceptual clarification that both critiques raised: Δz = z(t+1) − z(t) is a finite difference of a first-order statistic (a linear projection of rates). It is NOT a second-order/correlation feature. It's a temporal derivative of a first-order feature.

This matters because the binding PDF's claim is that "transformation variables should be disproportionately represented in second-order and temporal statistics." Δz tests the temporal part but not the second-order part.

To genuinely test whether second-order structure carries transformation information, we need:
- Compute the lagged cross-covariance matrix of δr(t) within a sliding window (e.g., 10 frames)
- Vectorize the upper triangle
- Decode velocity from these covariance features
- Compare to Δz (first-order temporal) decoding

If covariance features add nothing beyond Δz, then transformations are in the temporal derivatives of first-order statistics. If covariance features outperform Δz, then genuine correlation structure matters. Both are interesting findings; they just support different versions of the claim.

### 3.3 Implementation

This reuses the Models A/B/C/D infrastructure from the temporal decoding pipeline, but with different labels:

- Model A features (time-averaged rate) → decode content AND transformation
- Model C features (temporal trajectory) → decode content AND transformation  
- Δz features → decode content AND transformation
- Covariance features (Model D-style) → decode content AND transformation

Report a 4×2 accuracy matrix (4 feature sets × 2 tasks) with the dissociation as the main result.

---

## 4. Analysis C: Flow Fields and Linear Dynamics (Model A)

### 4.1 Revised scope

This analysis is interpretive and descriptive, not evidential. It explains the dynamics characterized in Analysis A but does not establish new claims. Downweight relative to A and B.

### 4.2 The linear dynamical model

Fit: z(t+1) = A z(t) + B u(t) + c + ε(t)

where z ∈ ℝ², u ∈ ℝ² (eye velocity), A ∈ ℝ^{2×2}, B ∈ ℝ^{2×2}, c ∈ ℝ².

**Critical baseline (added)**: also fit the B-only model:

z(t+1) = B u(t) + c + ε(t)

Compare R² of the full model vs. B-only. If B-only is nearly as good, the autoregressive component (A) is not doing meaningful work — the dynamics are purely input-driven with no interesting recurrence. If A substantially improves R², the ConvGRU's recurrence contributes to the translation dynamics (integration, filtering, or rotation).

**Interpretability of A** (heavily caveated): 
- Eigenvalues of A: magnitude < 1 → stable/decaying; complex → rotation
- **BUT**: with autocorrelated u(t), A absorbs input temporal structure, making eigenvalue interpretation fragile. Apparent integration, rotation, or recurrence are too easy to misread without explicit lagged-input comparisons.
- The B-only baseline partially deconfounds: if R² barely changes when A is removed, then A is not doing meaningful work regardless of its eigenvalues.
- **Recommendation**: report A eigenvalues as descriptive summaries only. Do not claim "the ConvGRU implements leaky integration" from eigenvalue structure alone. That claim requires fitting lagged-u variants (u(t-1), u(t-2)) and showing that adding lags does not absorb the apparent A dynamics. If that comparison is not done, state the eigenvalue results as "consistent with" rather than "demonstrating" a particular dynamical regime.

### 4.3 Flow fields

**Input-conditioned flow fields (primary)**: bin timepoints by eye velocity direction (8 bins) and compute mean Δz per z-space grid cell for each velocity condition. Show 2–4 direction conditions as separate quiver plots. This directly visualizes how the B matrix maps input directions to state-space dynamics.

**Unconditioned flow field (secondary)**: compute mean Δz per grid cell without conditioning on u. This approximately visualizes (A−I)z + c (the autonomous dynamics). If A is stable, arrows point inward; if unstable, outward. Less informative than the conditioned fields unless A has interesting structure (complex eigenvalues, anisotropy).

**Model-vs-empirical comparison**: overlay Model A's predicted Δz field on the empirical field. Compute per-cell cosine similarity. Report the fraction of populated cells where agreement exceeds 0.8.

### 4.4 What this adds

If Analysis A establishes that Δz encodes velocity (non-trivially, specifically, and across stimuli), then Analysis C tells you the mechanism:
- Is the encoding a simple passthrough (B dominates, A ≈ 0)?
- Is there temporal integration (A ≈ αI, α < 1)?
- Is there rotation or oscillation (A has complex eigenvalues)?
- How does the mechanism vary across stimuli (per-stimulus A, B)?

These are interesting interpretive questions, but they don't strengthen or weaken the core claim from Analysis A.

---

## 5. Connections to Existing Results and the Binding Framework

### 5.1 How this completes the temporal decoding story

The temporal decoding pipeline (Tier 1) established:
- C ≈ A for orientation: temporal residuals don't help content decoding
- FEMs induce real spatial map dynamics (CoM movement)
- Real FEM > matched_null for Model A: trajectory structure matters for rate-code quality

The transformation dynamics analysis (this plan) completes the picture:
- If Analysis A succeeds: temporal dynamics encode transformations, establishing a content/transformation dissociation
- If Analysis B succeeds: the dissociation is clean — content in static features, transformation in dynamics
- Together: FEMs serve a dual role — modulating mean rates for content AND creating temporal dynamics that encode the transformation

### 5.2 Connection to the binding/shared-causes framework

The motivation PDF proposes three tests:

1. **Dimensionality matches DOF**: ✅ already established — top 2 PCs capture ~97% of FEM variance, consistent with 2 DOF translation

2. **Subspace separability under multiple causes**: partially testable with existing infrastructure (e.g., simulate an object moving against a static background and check whether FEM covariance factorizes). Not addressed in this plan — flag as future work.

3. **Content-transformation dissociation across statistics**: directly tested by Analysis B.

### 5.3 What this does NOT test

This analysis uses a deterministic digital twin. It cannot test:
- Whether real V1 neurons show the same transformation encoding
- Whether the temporal structure is actually read out by downstream areas
- Whether multiple latent causes (multiple objects) produce separable subspaces

These are future directions for the real neural data (Aim 1) and the visual hierarchy recordings (Aim 3).

---

## 6. Implementation

### 6.1 Existing code to reuse

| Component | Source | What it provides |
|-----------|--------|-----------------|
| δr computation | `translation_covariance.py` | `get_trial_stim_and_rates`, `compute_delta_y`, `compress_to_population_vec` |
| U_pca2 per stimulus | `all_cov_results.pkl` | Pre-computed FEM covariance eigenvectors |
| Eye traces | `backimage_fixation_results.pkl` | Per-stimulus eye position traces |
| Model + readout | `utils.py`, `mcfarland_outputs_mono.pkl` | Digital twin infrastructure |
| Decoding infrastructure | `temporal_decoding/decoding.py` | `GroupKFold`, `LogisticRegression`, `StandardScaler` |

### 6.2 New code to write

**File: `declan/transformation_dynamics.py`**

```
# Core functions needed:

def compute_z_trajectory(delta_r, U_s):
    """Project δr(t) onto 2D basis. Returns z: (T, 2)"""

def compute_dz(z):
    """Δz(t) = z(t+1) - z(t). Returns (T-1, 2)"""

def compute_velocity(eyepos):
    """vel(t) = pos(t) - pos(t-1). Returns (T-1, 2)"""

def readout_decoding(features, targets, trace_ids, n_splits=5):
    """Ridge regression CV by trace. Returns R², per-fold R²"""

def random_projection_baseline(delta_r, targets, n_projections=200, d=2):
    """Random dD projections → Δz_rand → decode. Returns R² distribution"""

def shuffled_pca_baseline(delta_r, targets, n_shuffles=100):
    """PCA on neuron-shuffled δr → Δz → decode. Returns R² distribution"""

def cross_stimulus_generalization_curve(all_stimuli_data, d_values=[2,4,6,8,10,15,20]):
    """Sweep shared basis dimensionality, report cross-stimulus R² vs d"""

def generate_surrogate_velocity(velocity, method='phase_randomize'):
    """Generate matched nuisance target: same PSD, no real coupling"""

def content_transformation_dissociation(features_dict, content_labels, transform_labels, 
                                         surrogate_labels=None):
    """2×2 (or 3×2) accuracy matrix for the dissociation, with surrogate specificity"""

def fit_linear_dynamics(z, u, fit_B_only=True):
    """Fit z(t+1) = Az(t) + Bu(t) + c. Also fit B-only baseline. Report both R²."""

def compute_flow_field(z, dz, u, grid_res=20, direction_bins=8):
    """Empirical and input-conditioned flow fields"""
```

### 6.3 Task breakdown for agents

**Task 1: Core z/Δz computation and readout** (Phase 1 — killer test)
- Load existing `all_cov_results.pkl` for U_pca2 per stimulus
- Compute z(t), Δz(t), vel(t) for all traces × stimuli
- Implement ridge regression CV by trace
- Compare z vs Δz vs [z,Δz] for velocity decoding
- Add time-shuffle control
- **Acceptance**: Δz > z for velocity, time-shuffle collapses Δz

**Task 2: Random projection baseline** (Phase 1 — non-triviality)
- Implement random 2D projections of δr
- Compute Δz_rand → velocity decoding for 200 random projections
- Report percentile of U_pca2 R² within random distribution
- **Acceptance**: U_pca2 R² above 95th percentile of random distribution

**Task 3: Cross-stimulus generalization as dimensionality curve** (Phase 1 — generality)
- Leave-one-stimulus-out: train Δz readout on N−1 stimuli, test on held-out
- Sweep shared basis dimensionality d ∈ {2, 4, 6, 8, 10, 15, 20}
- Plot cross-stimulus R² vs d — the curve shape is the result
- **Acceptance**: curve shows clear elbow or stabilization; report d* (minimal effective dimensionality)

**Task 4: Content-transformation dissociation** (Phase 2)
- Compute the primary 2×2 matrix (static vs dynamic × content vs transformation)
- Include surrogate velocity targets (phase-randomized or trace-mismatched) as specificity controls
- Optionally add second-order covariance features as a third row if sample size supports stable estimation
- **Acceptance**: clear asymmetry — static features better for content, dynamic features better for transformation

**Task 5: Flow fields and Model A** (Phase 3 — descriptive)
- Fit A, B, c per stimulus; also fit B-only baseline
- Compute input-conditioned flow fields
- Visualize for 2–3 exemplar stimuli
- **Acceptance**: interpretable flow fields; Model A R² > B-only R² (if recurrence matters)

### 6.4 Execution order

```
Task 1 (Δz readout + time-shuffle) ──► decision point
    │
    ├── If Δz ≈ z: STOP. No dynamical encoding advantage.
    │   Report as "transformation info in static state, not dynamics."
    │
    └── If Δz > z: continue
            │
            ├── Task 2 (random projection) ──► decision point
            │       │
            │       ├── If U_pca2 within random dist: STOP.
            │       │   Report as "any subspace works — trivial driven-system property."
            │       │
            │       └── If U_pca2 special: continue
            │               │
            │               ├── Task 3 (cross-stimulus) ──► strength of claim
            │               │
            │               └── Task 4 (dissociation) ──► full story
            │
            └── Task 5 (flow fields) ── always run, descriptive
```

---

## 7. Figure Plan

### Figure 1: Δz encodes transformation (Analysis A core result)
- **A**: Exemplar stimulus + eye trajectory + z(t) trajectory in 2D (time-colored)
- **B**: Bar plot: R² for velocity decoding from z vs Δz vs [z,Δz]. Error bars from CV.
- **C**: Time-shuffle control: Δz R² collapses. Bar plot with shuffled vs real.

### Figure 2: The encoding is non-trivial and general (Analysis A controls)
- **A**: Histogram of R² from 200 random projections, with U_pca2 R² marked. Shows specificity of the FEM-aligned subspace.
- **B**: Cross-stimulus generalization: R² for within-stimulus vs across-stimulus readout. Shows generality.
- **C**: Specificity: R² for Δz predicting velocity (high) vs content (low). Shows selectivity.

### Figure 3: Content-transformation dissociation (Analysis B)
- **A**: The primary 2×2 heatmap: static vs dynamic features × content vs transformation tasks. Clean asymmetry = dissociation.
- **B**: Extension (if clean): second-order covariance features added as a third row. Does covariance add to transformation readout beyond Δz?

### Figure 2 addendum: Cross-stimulus generalization curve
- **D** (add to Figure 2): R² as a function of shared-basis dimensionality d. Mark the "elbow" where generalization stabilizes. Report d* (the minimal dimensionality for robust cross-stimulus readout).

### Figure 4: Dynamics and mechanism (Analysis C)
- **A**: Input-conditioned flow fields for 2 direction conditions, exemplar stimulus.
- **B**: Model A vs B-only R² comparison across stimuli. 
- **C**: A eigenvalues across stimuli (spectral radius plot).
- **D**: Model-predicted vs empirical flow field agreement (cosine similarity).

---

## 8. What Each Analysis Can Claim

| Analysis | Can claim | Cannot claim |
|----------|-----------|-------------|
| A (Δz > z, passes random-projection test) | Transformation variables are more accessible from FEM-aligned population dynamics than from static state, and this is not a trivial property of arbitrary projections | That dynamics are the *only* or *canonical* format for transformation encoding; that this constitutes a "code" rather than a structured driven response |
| A (cross-stimulus generalization curve) | The transformation-sensitive subspace generalizes across image content at a specific dimensionality | That the subspace is universal in 2D (it likely requires higher dimensionality) |
| B (double dissociation) | Content and transformation are disproportionately accessible from different statistics of population activity | That this dissociation is absolute or unique to FEM-driven systems |
| C (Model A with interpretable A, B) | The dynamics can be approximated as linear with specific filtering properties (descriptive) | That the eigenvalue structure reflects neural recurrence rather than input autocorrelation (without lagged-u comparison) |

---

## 9. Risks and Decision Points

**Risk 1: Δz ≈ z.** Both predict velocity equally well. This would mean transformation info is in the static state, not specifically in dynamics. The claim becomes "z tracks eye position" which is weaker but still publishable (it's the existing regression result reframed).

**Risk 2: Random projections work as well as U_pca2.** Any 2D slice of the population carries velocity info in its dynamics. This means the result is a property of the driven system's dimensionality and smoothness, not of the specific FEM-aligned subspace. The finding becomes: "the V1 population response is rich enough that any projection preserves transformation information." Interesting but different.

**Risk 3: Cross-stimulus generalization fails even at higher dimensionality.** The transformation encoding is fully stimulus-specific. This is consistent with "different images activate different neurons, and each subpopulation encodes translation independently." It weakens the "universal transformation code" story but is consistent with a "binding by shared transformation" account (neurons responding to the SAME image show coherent modulation).

**Risk 4: The double dissociation is messy.** Both feature sets predict both tasks to some degree. This is the most likely outcome — perfect dissociations are rare. Report the *relative* advantage: how much better is Δz for transformation relative to content, and vice versa for mean rate? An asymmetry is still informative even if not perfectly clean.

---

## 10. Relationship to the Original Temporal Decoding Plan

This analysis represents a **pivot** from the original plan, not an extension of it. The original plan asked:

> "Do FEMs create a temporal code that improves content discrimination?"

The answer from the data: **no, for orientation at these spatial scales.** Content discrimination is captured by time-averaged rates.

The new plan asks:

> "Do FEMs create population dynamics that encode the transformation, not the content?"

This is a different and arguably more interesting question, but it must be presented honestly as a pivot motivated by the C ≈ A null result, not as the original hypothesis. The strongest framing:

1. We set out to test whether temporal structure improves content decoding.
2. It does not (C ≈ A). Content is accessible primarily from first-order static statistics.
3. But FEMs induce real, structured, low-dimensional population dynamics (existing covariance analysis).
4. We therefore asked: what are these dynamics about, if not content?
5. Transformation variables (retinal slip / eye velocity) are disproportionately accessible from population dynamics (Δz) rather than static state (z).
6. Content and transformation are dissociated across different statistics of population activity.
7. This is consistent with a binding-via-shared-causes framework, where transformations are embedded in correlation geometry rather than in explicit rate codes.
8. The downstream functional consequence — whether this transformation information supports grouping, binding, or segmentation — remains to be tested.

That narrative is honest, well-motivated, and tells a complete story without overclaiming.
