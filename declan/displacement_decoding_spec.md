# Figuring Space by Time: Displacement Decoding in the V1 Digital Twin

## Companion Analysis to the CoM Dynamics Spec

---

## 1. Theoretical Motivation

Ahissar & Arieli (2001) proposed that sensory organ movements convert spatial structure into temporal structure: spatial offsets between image features become temporal delays between neural responses as RFs sweep across the image. This "figuring space by time" framework makes a specific prediction that our prior analyses did not test.

**The category mismatch.** Our C ≈ A null result (temporal features don't improve orientation decoding) is predicted by this framework. Ahissar's temporal code is *differential* — it encodes relative spatial offsets (Δx between features), not absolute pattern identity (which orientation is the E?). We tested a differential code against an absolute classification task. The code may be present but aligned with a different variable.

**The prediction.** The most immediate variable exposed by FEM-driven population dynamics should be *relative displacement* — how the retinal image has shifted, not what the image contains. This is testable: decode small image displacements from population responses, comparing FEM vs. stabilized conditions.

**Relationship to other analyses:**
- **CoM dynamics spec** (running): tests whether spatial map features track eye velocity — the *transformation* variable
- **This analysis**: tests whether population responses encode *relative image displacement* — the *spatial offset* variable that Ahissar argues is converted to temporal structure
- **Temporal decoding pipeline** (completed): tested content decoding (orientation) — the *identity* variable → null result

Together, these three analyses triangulate the question: what variables do FEM-driven dynamics encode?

---

## 2. Core Analysis: Displacement Decoding

### 2.1 Setup

For a fixed natural image, generate population responses at a grid of controlled retinal positions, then test whether the displacement between positions is decodable from the response difference.

**Stimuli:** use the same backimage stimuli from `translation_covariance.py` (6 natural images, already cached in `backimage_fixation_results.pkl`).

**Displacement grid:** for each image, compute responses at a grid of eye positions:
- Center position: (0, 0) — the mean fixation point
- Displacements: (δx, δy) on a grid spanning ±0.05° in steps of 0.01° (11 × 11 = 121 positions)
- This covers the typical FEM range with sub-FEM resolution

**Responses:** at each position, compute the full population response using the existing pipeline. Two versions:
- **Scalar rates:** `amax`-collapsed, giving (N,) per position — what Model A uses
- **Spatial moments:** (CoM_x, CoM_y, σ_x, σ_y) per neuron, giving (4N,) per position — from Phase 0 infrastructure

### 2.2 Feature construction

For each pair of positions (p₁, p₂), the displacement target is:

```
(δx, δy) = p₂ - p₁
```

The features are response differences:

```
Δr = r(p₂) - r(p₁)
```

This directly implements Ahissar's logic: the spatial offset (δx, δy) should be encoded in the response difference between the two positions. The response difference is the population-level analog of "temporal delay between burst onsets of adjacent neurons" — just expressed in rate-code form rather than spike timing.

**Feature sets to compare:**

| Feature | Description | Dimension | What it tests |
|---------|-------------|-----------|---------------|
| Δr_scalar | Difference of amax-collapsed rates | N | Does the scalar rate code carry displacement? |
| Δr_moments | Difference of spatial moments | 4N | Do spatial map features carry displacement better? |
| ΔCoM only | Difference of CoM features only | 2N | Is displacement in map position or map shape? |
| Δwidth only | Difference of width features only | 2N | Is displacement in map deformation? |

### 2.3 Decoding

Train a ridge regression decoder to predict (δx, δy) from each feature set.

**Cross-validation:** leave-one-image-out. Train on displacement pairs from 5 images, test on the held-out 6th. This tests *generalization across image content* — if the displacement code is truly about the transformation (how much the image shifted) rather than the content (what the image is), it should transfer across images.

**Within-image cross-validation** (secondary): for each image, split displacement pairs into train/test. This is easier and should work if the code exists at all.

**Report:** R² for (δx, δy) prediction, separately for each feature set, for both within-image and cross-image conditions.

### 2.4 Key comparison: static vs. FEM-driven

The above uses static (instantaneous) responses at each position. This already tests whether the population response difference encodes displacement. But Ahissar's specific claim is that *movement* improves this encoding.

To test this, compare displacement encoding under three conditions:

1. **Static snapshots:** responses at fixed eye positions (from the displacement grid above). The response difference Δr encodes displacement purely through the spatial code — different positions activate neurons differently.

2. **FEM-driven temporal trajectory:** responses under real eye traces. At each time point t, the population has seen a *trajectory* of positions, not just the current one. Extract features from the temporal trajectory (mean, PCA, or moments over a time window) and decode the *current displacement* from the fixation center.

3. **Stabilized:** responses with eye fixed at center. No displacement information should be present (the input never moves).

**The decisive comparison:** does the FEM-driven trajectory representation decode displacement better than the static snapshot at the same position? If yes, temporal integration of the movement-induced responses adds displacement information beyond what a single-frame spatial code provides. That's the Ahissar prediction: movement *improves* the encoding of spatial offsets by converting them into temporal structure.

---

## 3. Supportive Analysis: Cross-Neuron Lag Structure

### 3.1 Motivation

Ahissar's specific mechanism is that spatial offsets become temporal delays between neurons whose RFs are separated along the drift direction: Δt = Δx / V_eye. In a rate-based model, this should appear as lagged cross-correlations between neurons that depend on their RF separation and the drift direction.

### 3.2 Implementation

For each pair of neurons (i, j) under real FEM:

1. Compute the cross-correlation function between r_i(t) and r_j(t) over a ±10 frame lag window.
2. Find the lag of peak correlation: τ_ij.
3. Compute the RF separation vector between neurons i and j (from their spatial readout weights or from the CoM at the mean fixation point).
4. Compute the eye drift direction at each time point.

**The prediction:** for neuron pairs whose RF separation is aligned with the instantaneous drift direction, the peak-correlation lag τ_ij should be proportional to their RF separation divided by drift velocity:

```
τ_ij ≈ |RF_sep_ij| / V_drift (when RF_sep aligned with drift)
```

For pairs whose separation is orthogonal to drift, τ_ij should be near zero.

### 3.3 What to report

- Scatter plot: τ_ij vs. projected RF separation (along drift direction), colored by drift speed
- Correlation between τ_ij and predicted delay (RF_sep / V_drift)
- Compare real FEM vs. stabilized: lag structure should vanish under stabilization

### 3.4 Caveats

This is the most fragile of the three analyses. In a rate-based model with learned ConvGRU filters, the lag structure may be:
- Smeared by temporal integration (GRU time constant ~25 frames >> expected delays of ~1-2 frames)
- Confounded by shared input dynamics
- Weak relative to internal recurrent dynamics

A null here does NOT kill the broader displacement-decoding hypothesis. A positive here strengthens it considerably by providing a mechanistic link to Ahissar's specific prediction. Treat as supportive, not decisive.

---

## 4. Connection to Existing Results

### 4.1 What we already know

| Finding | Implication for displacement decoding |
|---------|--------------------------------------|
| C ≈ A for orientation | Temporal structure doesn't help identity → category mismatch, not absence of code |
| FEM covariance is rank-2 | Population variability under FEM is ~2-dimensional → consistent with 2 DOF translation |
| Maps move under FEM | Spatial activation shifts with eye position → displacement signal exists in spatial maps |
| B ≈ 0 in scalar z-space | Velocity doesn't drive scalar latent dynamics → displacement signal destroyed by spatial collapse |
| Phase 0 passed (CoM tracks shifts) | CoM reliably encodes position with slopes 0.97–2.84 px/deg → spatial moment features are viable |

### 4.2 How this analysis completes the picture

The CoM analysis (running) asks: does the *rate of change* of spatial features track eye *velocity*?

This analysis asks: does the *difference* in population responses between positions encode *displacement*?

These are complementary:
- Velocity = d(position)/dt — a temporal derivative
- Displacement = position₂ - position₁ — a spatial difference

If both work, the population encodes both the static displacement (from response differences) and the dynamic transformation (from temporal evolution of spatial features). If only one works, we learn which representation — spatial difference or temporal dynamics — is the primary carrier.

---

## 5. Implementation

### 5.1 Code location

**File:** `declan/displacement_decoding.py`

Reuses:
- `translation_covariance.py`: model/readout loading, `get_trial_stim_and_rates`, `compress_to_population_vec`
- `com_dynamics.py`: `compute_spatial_moments` (from Phase 0 infrastructure)
- `backimage_fixation_results.pkl`: stimulus images and eye traces

### 5.2 Key functions

```python
def compute_displacement_grid(model, readout, image, shifts_deg,
                               ppd=37.5, n_lags=32, out_size=(101,101)):
    """Compute population responses at a grid of controlled eye positions.
    Returns: 
        scalar_rates: (n_positions, N) — amax-collapsed
        spatial_moments: (n_positions, N, 4) — CoM + width per neuron
        positions: (n_positions, 2) — (x, y) in degrees
    """

def build_displacement_pairs(responses, positions, max_displacement_deg=0.05):
    """Generate all pairs of positions within max displacement.
    Returns:
        features: (n_pairs, feature_dim) — response differences
        targets: (n_pairs, 2) — (δx, δy) displacement vectors
        pair_indices: (n_pairs, 2) — which positions were paired
    """

def decode_displacement(features, targets, n_splits=5):
    """Ridge regression CV for displacement prediction.
    Returns: R² for δx, δy, and combined.
    """

def cross_image_displacement_decoding(all_images_data, n_folds=6):
    """Leave-one-image-out: train on 5 images, test on 6th.
    Tests whether displacement code generalizes across image content.
    """

def compute_pairwise_lags(rates_by_neuron, rf_positions, eye_velocity,
                           max_lag=10, n_pairs=500):
    """Cross-neuron lag analysis.
    Returns: lag_matrix, rf_separations, predicted_lags
    """

def fem_vs_static_displacement(model, readout, image, eye_traces,
                                 displacement_grid_responses):
    """Compare displacement decoding from static snapshots vs FEM trajectories.
    The key Ahissar test: does movement improve displacement encoding?
    """
```

### 5.3 Computational cost

- **Displacement grid:** 121 positions × 6 images × ~5 sec/position ≈ 1 hour. Can reuse Phase 0 shift data (already computed for 400 positions on 1 image — extend to other images).
- **Pair construction and decoding:** minutes (combinatorics on cached responses).
- **Cross-neuron lags:** ~30 minutes per stimulus (pairwise correlations over ~130 neurons).
- **FEM vs static comparison:** uses existing FEM rate data from `translation_covariance.py` + the displacement grid. No new forward passes needed.

**Total: ~2 hours new compute + existing cached data.**

---

## 6. Decision Tree

```
Displacement grid computation (6 images × 121 positions)
    │
    Phase 1: Within-image displacement decoding
    │
    ├── Δr_scalar R² >> 0: scalar rate code carries displacement
    │   (expected — different positions produce different rates)
    │
    ├── Δr_moments R² >> Δr_scalar R²: spatial moments carry 
    │   displacement better than scalar rates
    │   → spatial map features add value
    │
    └── Both R² ≈ 0: positions don't produce discriminable responses
        → problem with the grid or the pipeline (unlikely given Phase 0)
        │
    Phase 2: Cross-image generalization
    │
    ├── Cross-image R² >> 0: displacement code is image-independent
    │   → genuine transformation encoding (strong result)
    │
    └── Cross-image R² ≈ 0: displacement code is image-specific
        → each image produces its own displacement signature
        │   (weaker but still meaningful)
        │
    Phase 3: FEM vs static comparison (the Ahissar test)
    │
    ├── FEM trajectory > static snapshot for displacement:
    │   → movement improves displacement encoding
    │   → direct support for "figuring space by time"
    │
    ├── FEM ≈ static: movement doesn't add to displacement encoding
    │   → displacement info is in the instantaneous spatial code
    │   → Ahissar's specific temporal mechanism not supported
    │
    └── FEM < static: movement degrades displacement encoding
        → FEM adds noise relative to static snapshots
        │
    Phase 4 (supportive): Cross-neuron lag analysis
    │
    ├── Lags correlate with RF_sep / V_drift:
    │   → mechanistic support for Ahissar's specific Δt = Δx/V
    │
    └── No lag structure:
        → doesn't kill displacement result, but weakens mechanistic link
```

---

## 7. Success Criteria (Pre-Declared)

| Test | Strong positive | Ambiguous | Negative |
|------|----------------|-----------|----------|
| Within-image displacement R² | > 0.3 for moments, > scalar | 0.05–0.3 | < 0.05 |
| Cross-image generalization | Cross-image R² > 50% of within-image R² | 20–50% | < 20% |
| FEM > static for displacement | FEM R² > static R² by > 0.05 | Within 0.05 | FEM < static |
| Cross-neuron lags | r(τ_ij, predicted_lag) > 0.3 | 0.1–0.3 | < 0.1 |

**The hierarchy:**
1. Within-image displacement decoding — must work or the whole approach fails
2. Cross-image generalization — determines whether the code is about transformation or content
3. FEM vs. static — the specific Ahissar test
4. Cross-neuron lags — supportive mechanistic evidence

---

## 8. What Each Outcome Means for the Project Narrative

### If displacement decoding works AND generalizes across images:

The population response to a natural image changes systematically with small retinal shifts, and this change is consistent across different images. FEMs continuously sample these displacement-dependent response changes. The C ≈ A null for orientation was a category mismatch: the population dynamics encode *where the image has shifted*, not *what the image contains*.

Combined with CoM results (if positive): transformations are accessible from both spatial map dynamics (CoM) and population response differences (displacement code), providing convergent evidence for representation-dependent transformation encoding.

### If displacement decoding works but is image-specific:

The population encodes displacement, but the *direction in population space* that encodes displacement depends on the image content (which neurons are activated, and how their RFs are arranged relative to local image structure). This is still interesting — it means displacement information exists but is entangled with content. A downstream decoder would need to know what image is being viewed to extract the displacement, or would need to learn a nonlinear readout.

### If FEM trajectories improve displacement over static snapshots:

Direct evidence for Ahissar's "figuring space by time": temporal integration of movement-induced responses enhances displacement encoding beyond what a single-frame spatial code provides. This would be the strongest possible result connecting FEMs to improved spatial information.

### If only static displacement works (FEM doesn't help):

The population has a fine spatial code for displacement (different positions produce discriminable responses), but temporal integration doesn't improve it. FEMs are sampling this code over time but not adding information through temporal structure. The "active sensing" benefit, if any, is in coverage (sampling more positions) rather than in temporal enhancement.

### If nothing works:

In this digital twin, small retinal displacements (at FEM scale) do not produce linearly discriminable changes in population responses. This is a genuine constraint on the model — possibly reflecting insufficient spatial resolution in the readout, or learned response properties that are robust to small shifts rather than sensitive to them. Combined with the existing B ≈ 0 and C ≈ A results, this would establish that the twin represents content but not transformation at FEM scales.

---

## 9. Relationship to the Full Analysis Program

This document is one of four complementary specs:

| Document | Question | Status |
|----------|----------|--------|
| `analysis_plan_consolidated_v2.md` | Does temporal structure improve content (orientation) decoding? | Complete — C ≈ A (null) |
| `transformation_dynamics_plan.md` | Do population dynamics encode transformation (velocity) in PCA-projected scalar space? | Complete — R² ≈ 0 (null) |
| `com_dynamics_spec.md` | Do spatial map features (CoM, width) track eye velocity? | Phase 0 passed, Phase 1 running |
| **This document** | Do population response differences encode relative displacement? | Ready to run |

The four documents test four different variables (content, velocity, spatial map dynamics, displacement) in four different representations (scalar rates, PCA latent space, spatial moments, response differences). Together they answer: *what do FEM-driven population dynamics encode, and in which representational format?*

The emerging answer: content is in first-order rate statistics; transformation/displacement information, if present, is in representations that preserve spatial structure (CoM, response differences) rather than in scalar or latent-space summaries.
