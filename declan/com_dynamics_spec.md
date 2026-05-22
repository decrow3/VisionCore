# Spatial Map Dynamics: CoM-Based Transformation Analysis

## Implementation Spec for Claude Code Agents

---

## Motivation (one paragraph)

All previous analyses collapsed the readout's spatial rate maps to scalars before asking temporal questions. The one robust positive finding — that spatial maps move under real FEM — lives in the spatial structure that was discarded. This analysis asks whether the *rate of change of spatial map position* tracks eye velocity, testing the hypothesis that transformation information lives in the spatial organization of responses rather than in scalar population dynamics or temporal residuals.

---

## Phase 0: Sanity Check (run first, stop if it fails)

### Static-shift linearity test

Before using real FEM traces, verify that the readout's spatial maps respond linearly to controlled image translations.

**Procedure:**
1. Take a single natural image (e.g., the most-used backimage stimulus).
2. Generate a series of synthetic eye positions: a grid of offsets from (0,0) in steps of 0.01 degrees, spanning ±0.1 degrees (covering the FEM range). Use ~20 positions along each axis = 400 total positions.
3. For each position, compute the full spatial rate map y_i(x,y) for every neuron i via the existing `compute_rate_map` pipeline (do NOT collapse spatially).
4. For each neuron, compute the center of mass (CoM) of its spatial rate map at each eye position.
5. Plot CoM_x vs. eye_position_x and CoM_y vs. eye_position_y for a sample of 20 neurons.

**Implementation:**
```python
def compute_com(rate_map):
    """
    Compute center of mass of a 2D rate map.
    rate_map: (H, W) non-negative array
    Returns: (com_x, com_y) in pixel coordinates
    """
    total = rate_map.sum()
    if total < 1e-10:
        return np.nan, np.nan
    H, W = rate_map.shape
    yy, xx = np.mgrid[0:H, 0:W]
    com_x = (xx * rate_map).sum() / total
    com_y = (yy * rate_map).sum() / total
    return float(com_x), float(com_y)

def compute_spatial_moments(rate_map):
    """
    Compute first and second spatial moments of a 2D rate map.
    Returns: (com_x, com_y, sigma_x, sigma_y) — center of mass + spatial width.
    sigma_x/y are the standard deviations of the rate-weighted spatial distribution.
    """
    total = rate_map.sum()
    if total < 1e-10:
        return np.nan, np.nan, np.nan, np.nan
    H, W = rate_map.shape
    yy, xx = np.mgrid[0:H, 0:W]
    com_x = (xx * rate_map).sum() / total
    com_y = (yy * rate_map).sum() / total
    sigma_x = np.sqrt(((xx - com_x)**2 * rate_map).sum() / total)
    sigma_y = np.sqrt(((yy - com_y)**2 * rate_map).sum() / total)
    return float(com_x), float(com_y), float(sigma_x), float(sigma_y)
```

This gives 4 features per neuron per time step (CoM_x, CoM_y, width_x, width_y) rather than just 2. The widths capture map deformation — if the map broadens, narrows, or changes shape with translation, the second moments will pick that up even if CoM is stable. Feature vector dimension becomes 4N instead of 2N.

**What to check:**
- Is the CoM-vs-eye-position relationship **locally monotonic and reliable** over the FEM range actually occupied by the traces (typically ±0.05°)?
- Global linearity over the full ±0.1° range is less important than local reliability near zero.
- Compute:
  - Local slope of CoM vs. shift at the origin (central difference over ±0.01°)
  - Trial-to-trial reliability of that slope (if applicable — here the model is deterministic, so reliability is perfect, but slope magnitude and consistency of sign across neurons matters)
  - Fraction of neurons with consistent sign (CoM_x increases when eye moves right)
- Also check whether sigma_x or sigma_y change systematically with shift — this indicates map deformation.
- Verify that the spatial maps are in a common coordinate frame: check for padding/convolution edge effects by inspecting maps near the edges of the output grid. If CoM drifts as an artifact of the readout support rather than genuine spatial shift, this will show up as edge-dependent bias in Phase 0.

**Decision:**
- If >50% of neurons show locally monotonic, consistent-sign CoM response AND the local slope is non-negligible (> 0.1 pixels per 0.01° shift): proceed to Phase 1 with CoM features.
- If CoM is unreliable but sigma_x/sigma_y show systematic shift-dependence: proceed with width features only (map deformation encodes translation even if position doesn't).
- If neither first nor second spatial moments respond reliably: the readout's spatial maps are too noisy or nonlinear for moment-based summaries. Report as constraint; consider full-map analysis (e.g., flattened map PCA) as a last resort, but note the much higher dimensionality.

**Key detail:** the rate map must NOT be collapsed. Use the full output of `readout(feats_last)` → `model.model.activation(y)`, which gives (B, N, H_out, W_out). Check what H_out and W_out actually are — if they're 1×1, the spatial map has already been collapsed by the readout architecture and this entire analysis is impossible. **This is the first thing to verify.**

---

## Phase 1: Spatial Moment Trajectory Computation

### For each stimulus × trace combination:

1. Run the existing rate computation pipeline but **retain the full spatial rate maps** (N, H_out, W_out) at each time step. Do NOT call `_collapse_spatial`.
2. At each time step t, for each neuron i, compute spatial moments: CoM_i(t) = (com_x, com_y) and width_i(t) = (sigma_x, sigma_y).
3. This gives a feature vector of dimension 4N at each time step: moments(t) ∈ ℝ^{4N}.
4. Compute Δmoments(t) = moments(t+1) − moments(t) ∈ ℝ^{4N}.
5. Separately extract CoM-only features (2N) and width-only features (2N) for ablation.
6. Compute eye velocity: vel(t) = eyepos(t+1) − eyepos(t) ∈ ℝ².

**Storage:** for M traces × T time steps × N neurons × 4 (com_x, com_y, sigma_x, sigma_y), the moments array is (M, T, 4N) float32. With M=50, T=200, N=130: ~20 MB per stimulus. Manageable.

**Stimulus set:** use the same backimage stimuli from the translation_covariance analysis (6 images, 50 traces each). Reuse cached rate maps if available; otherwise recompute but save the full spatial maps this time.

**Optimization:** compute spatial moments on-the-fly during the forward pass — compute rate map → compute moments → discard rate map → store only the 4-element moment vector per neuron. This reduces storage from (M, T, N, H, W) to (M, T, N, 4).

---

## Phase 2: Velocity Decoding from CoM Features

### 2.1 Core comparison

Decode eye velocity vel(t) from multiple feature sets, using ridge regression with trace-level grouped CV:

| Feature | Description | Dimension |
|---------|-------------|-----------|
| **CoM(t)** | Static spatial map positions | 2N |
| **ΔCoM(t)** | Rate of change of map positions | 2N |
| **width(t)** | Spatial map widths (second moments) | 2N |
| **Δwidth(t)** | Rate of change of map widths | 2N |
| **all_moments(t)** | CoM + width combined | 4N |
| **Δall_moments(t)** | Rate of change of all moments | 4N |
| **[CoM(t), ΔCoM(t)]** | Combined static + dynamic position | 4N |

**Also run for comparison (from existing analysis):**

| Feature | Description | Dimension |
|---------|-------------|-----------|
| **z(t)** | PCA-projected scalar population state | 2 |
| **Δz(t)** | Scalar state increment | 2 |
| **r(t)** | Full scalar rate vector (amax-collapsed) | N |

Report R² for each, per stimulus and averaged across stimuli. The CoM-only vs. width-only comparison reveals whether translation is encoded primarily in map position (first moment) or map deformation (second moment).

### 2.2 Controls

**Time-shuffle:** shuffle time indices within each trace independently per neuron. ΔCoM readout should collapse.

**Neuron-shuffle:** shuffle neuron identities at each time step (preserving temporal structure but destroying population spatial organization). Tests whether the readout requires the specific neuron-to-RF mapping.

**Stimulus specificity:** decode stimulus identity (which image?) from ΔCoM. Should be poor (content is in mean CoM or mean rate, not in the dynamics of map position).

### 2.3 Cross-stimulus generalization

Train the ΔCoM → velocity decoder on 5 stimuli, test on the held-out 6th. Repeat for all 6 leave-one-out folds. Since CoM features are neuron-specific (not PCA-dependent), there's no basis-choice issue — the same neuron's CoM is used across stimuli. The decoder weights should transfer if the transformation encoding is stimulus-independent.

---

## Phase 3: Content-Transformation Dissociation (only if Phase 2 shows ΔCoM > 0)

If ΔCoM successfully decodes velocity, run the 2×2 dissociation:

| | Content (which image?) | Transformation (velocity) |
|---|---------|---------------|
| Mean rate (time-averaged scalar) | ? | ? |
| ΔCoM (spatial map dynamics) | ? | ? |

**Prediction for a clean dissociation:** mean rate decodes content well and velocity poorly; ΔCoM decodes velocity well and content poorly.

---

## Phase 4: Comparison to Existing Results

If ΔCoM works, directly compare:

1. **ΔCoM R²** (this analysis) vs. **Δz R²** (existing, ≈ 0) — quantifies how much signal was lost by spatial collapse.
2. **CoM(t) R²** vs. **z(t) R²** — is static spatial position more informative than the scalar latent state for transformation?
3. **Mean rate content accuracy** vs. **ΔCoM content accuracy** — does the dissociation hold?

---

## Implementation Details

### Where to put the code

**File:** `declan/com_dynamics.py`

This is a standalone analysis script, not integrated into the `temporal_decoding/` pipeline (which uses scalar rates). It reuses:
- Model/readout loading from `translation_covariance.py`
- Stimulus/trace infrastructure from `translation_covariance.py`
- Ridge regression CV from `transformation_dynamics.py`

### Key functions

```python
def check_spatial_output_size(model, readout):
    """FIRST THING TO RUN: verify H_out, W_out > 1.
    Also check for padding/edge artifacts in the spatial maps."""

def static_shift_sanity_check(model, readout, image, shifts_deg, ppd=37.5):
    """Phase 0: verify CoM and width respond reliably to controlled image shifts.
    Returns: per-neuron moments vs shift, local slope at origin, sign consistency."""
    
def compute_moment_trajectories(model, readout, stim_stack, eye_traces, durations,
                                 n_lags=32, out_size=(101,101), batch_size=16):
    """Phase 1: compute full spatial rate maps and extract moments per neuron per frame.
    Computes on-the-fly: rate map → (com_x, com_y, sigma_x, sigma_y) → discard map.
    Returns: list of (T_m, N, 4) moment arrays."""
    
def moment_velocity_decoding(moment_trajectories, eye_traces, trace_ids, 
                              feature_sets=['com', 'dcom', 'width', 'dwidth', 'all', 'dall'],
                              n_splits=5):
    """Phase 2: decode velocity from various moment features.
    Ridge regression with GroupKFold by trace.
    Returns: R² for each feature set."""

def cross_stimulus_moment_decoding(all_stimuli_moments, all_stimuli_vel, n_folds=6):
    """Leave-one-stimulus-out cross-validation using CoM features (neuron-indexed, no basis)."""

def content_transform_dissociation(mean_rates, moment_trajectories, 
                                     content_labels, velocity_targets):
    """Phase 3: 2×2 dissociation matrix using mean rate vs Δmoments."""
```

### Computational cost

- **Phase 0:** ~400 forward passes (one per shift position) × ~5 sec each = ~30 minutes. Run once.
- **Phase 1:** 6 stimuli × 50 traces × ~200 frames = 60,000 forward passes. But we need full spatial maps, which means NOT collapsing — same compute cost as `compute_rate_map_batched` but storing more data. ~8 hours.
- **Phase 2–4:** minutes (just linear algebra on cached CoM features).

**Optimization:** if full-map caching is too large, compute CoM on-the-fly during the forward pass (compute rate map → compute CoM → discard rate map → store only CoM). This reduces storage from (M, T, N, H, W) to (M, T, N, 2).

---

## Decision Tree

```
Phase 0: check spatial output size
    │
    ├── H_out = W_out = 1: STOP. No spatial structure in readout.
    │   Conclusion: readout architecture collapses space; spatial moment features undefined.
    │
    └── H_out, W_out > 1: continue
            │
            Phase 0: static-shift sanity (local monotonicity + sign consistency)
            │
            ├── Neither CoM nor width respond reliably: STOP (or try full-map PCA as last resort).
            │
            ├── CoM reliable, width also informative: proceed with full 4N features
            │
            └── CoM unreliable but width responds: proceed with width-only features
                    │
                    Phase 2: Δmoments velocity decoding
                    │
                    ├── R² ≈ 0 for all moment features: STOP.
                    │   Twin does not encode FEM velocity in any tested scalar,
                    │   latent, or spatial-moment representation.
                    │   (Does NOT rule out higher-order spatial features.)
                    │
                    └── R² >> 0 for ΔCoM and/or Δwidth: continue
                            │
                            ├── Phase 2 controls (time-shuffle, neuron-shuffle, cross-stim)
                            │
                            └── Phase 3: dissociation
                                    │
                                    └── Write up positive result
```

---

## What Each Outcome Means

| Outcome | Interpretation | Publishability |
|---------|---------------|----------------|
| H_out = 1 | Readout has no spatial structure; spatial features undefined | Report as architecture constraint |
| CoM not reliable with shift | RF structure too complex for CoM summary; other spatial features (width, ellipse) may still work | Weak negative — try second-moment features before concluding |
| ΔCoM and Δwidth R² ≈ 0 | Twin does not encode FEM velocity in any tested scalar, latent, or spatial-moment representation | Strong negative — clean constraint on the model. Note: does not rule out encoding in higher-order spatial features (e.g., full map shape, nonlinear spatial interactions) |
| ΔCoM R² >> 0, controls pass | Transformation more accessible from spatial map dynamics than from scalar collapse | Strong positive — best possible outcome |
| ΔCoM R² >> 0, cross-stim fails | Transformation readout is stimulus-dependent (CoM response depends on local image structure) | Moderate positive — weaker generality, expected given stimulus-dependent RF engagement |
| Dissociation holds | Content more accessible from mean rates, transformation more accessible from spatial dynamics | Strongest positive — completes the story |

---

## Estimated Timeline

| Phase | Time | Blocking? |
|-------|------|-----------|
| Check H_out, W_out | 5 minutes | YES — if 1×1, everything stops |
| Phase 0 sanity check | 30 minutes | YES — if CoM not linear, analysis changes |
| Phase 1 CoM computation | 8 hours (can run overnight) | YES for Phase 2 |
| Phase 2 decoding | 10 minutes | NO |
| Phase 3 dissociation | 10 minutes | NO |
| Total if positive | ~9 hours | |
| Total if stopped at Phase 0 | 35 minutes | |

**Start with the 5-minute H_out check. Everything depends on that.**
