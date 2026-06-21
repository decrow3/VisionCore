# Analysis Plan: The Image-Specific Jacobian Hypothesis (v3)

## What changed from v2

1. **Factorized Jacobian section added** (after Mechanistic model): makes explicit that
   J = J_model · J_image and C_FEM = J_model · M_image · J_modelᵀ, where M_image is
   the image structure tensor. Explains rank-2, image-specificity, and cross-image
   failure as consequences of the factorization, not separate findings.
2. **Matched-energy null added to Test 3**: row-shuffled J_null preserves column norms
   but destroys image/model alignment. Stronger than the random rank-2 null because it
   controls for energy; U_jac must outperform it to be meaningful.
3. **IC-E added** (ensemble directional amplification): a minimal 5-minute probe
   asking whether C_FEM averaged over images has stable structure, and whether biological
   Σ_eye selectively amplifies specific population directions. Scoped explicitly as
   descriptive only — does not derail the core tests.

---

## Mechanistic model

Under the Jacobian hypothesis, the FEM-induced covariance at a fixed image and eye
position **p₀** is:

```
C_FEM(p₀) ≈ J(p₀) · Σ_eye · J(p₀)ᵀ
```

where **J(p₀) ∈ ℝ^{N×2}** = [∂λ/∂x, ∂λ/∂y] evaluated at p₀, and **Σ_eye ∈ ℝ^{2×2}**
is the effective eye-position covariance seen by the model within the decoding window.

This model competes with a second explanation:

**View A — local differential geometry:** C_FEM is a passive pushforward of eye-motion
variability through the static representation. The Jacobian model is sufficient.

**View B — shared temporal-core dynamics:** C_FEM also reflects GRU mixing of recent
positions. The shared feature map creates low-rank, image-specific covariance *beyond*
what a point Jacobian predicts, especially when the GRU integrates over multiple frames.

Tests 3 and 4 put these two views into direct competition.

### Factorized Jacobian

The full Jacobian J = ∂λ/∂p chain-rule decomposes as:

```
J = (∂λ/∂I) · (∂I/∂p) = J_model · J_image
```

so that the FEM covariance becomes:

```
C_FEM ≈ J_model · M_image · J_model ᵀ
```

where:
- **J_image ∈ ℝ^{P×2}** = [∇_x I, ∇_y I] — the spatial gradient of the retinal image
  under translation. This encodes pure image content; it is entirely determined by the
  scene and the rendering pipeline.
- **M_image = J_image Σ_eye J_imageᵀ ∈ ℝ^{P×P}** — the image-domain covariance induced
  by eye movements. Under isotropic Σ_eye, this reduces to ∇I ∇Iᵀ, which is the
  classical **structure tensor** of the image.
- **J_model ∈ ℝ^{N×P}** = ∂λ/∂I — the model's pixel-to-rate sensitivity. This encodes
  the effective receptive field geometry of each neuron. For the current architecture it
  implicitly includes temporal filtering by the GRU (see View B caveat below).

This factorization separates three independent sources:

| Factor | Source | What it explains |
|---|---|---|
| J_image | Scene statistics | Why covariance is image-specific |
| Σ_eye | Eye movement statistics | Why covariance amplitude scales with FEM |
| J_model | Model / RF geometry | How image-domain perturbations project into population space |

**Rank-2 is a constraint, not a discovery.** Because J_image ∈ ℝ^{P×2}, C_FEM has rank
≤ 2 regardless of image content or model architecture. This makes the PR ≈ 2 finding an
expected consequence of 2D translation, not a finding that requires further explanation.

**Cross-image generalization failure is a prediction.** Different images → different
∇I → different J_image → different column space for J → different covariance geometry.
No decoder without image information can generalize because the signal direction in
population space is image-conditioned.

**View B caveat on J_model.** The true model computes λ(t) = f({I(p(t−τ)), ..., I(p(t))})
over a GRU window. J_model therefore implicitly includes temporal filtering of position
history, not just instantaneous pixel sensitivity. Discrepancies between the analytic
factorization and the forward-AD J from Test 3 should be interpreted as View B effects
(temporal mixing), not measurement failures.

Tests 3 and 4 evaluate whether C_FEM is fully explained by this factorization (View A)
or requires additional temporal-core dynamics beyond what J_model · M_image · J_modelᵀ
predicts (View B).

---

## Decision criteria (stated in advance)

| Test 3 result | Test 6 result | Interpretation |
|---|---|---|
| Alignment > 0.90 | Subtract helps at −0.20, neutral at −0.40 | **Jacobian model confirmed; FEM variability = nuisance above threshold, irrelevant at hyperacuity** |
| Alignment > 0.90 | Subtract hurts at −0.40 | **Jacobian-aligned variability is signal-bearing at hyperacuity** |
| Alignment 0.70–0.90 | Any | Jacobian is partial explanation; GRU/curvature adds residual structure |
| Alignment < 0.70 | Any | Jacobian model inadequate; investigate architectural causes (see below) |

If Test 3 fails cleanly (alignment < 0.70), the architectural causes to examine first are:
- GRU temporal mixing: the shared feature map is not a static translation of the world,
  so the Jacobian of the *static* map is the wrong derivative
- Operating point: J(0,0) may not represent the effective operating point during natural
  FEM traces (which have nonzero mean position and velocity)
- Curvature: the FEM amplitude range is large enough that J is not constant (Test 5)

These are analytic/architectural issues, not biological noise sources. The model has no
internal noise beyond Poisson and eye-trace variability by construction.

---

## Test 3 — Analytic Jacobian for E-optotype via forward-AD *(core mechanistic test)*

**Priority: 1. Run this first.**

**What it tests:** Whether the FEM covariance subspace for the E equals the subspace
spanned by the analytic translation Jacobian, per orientation, per LogMAR. This is the
cleanest test of View A vs View B.

**Steps:**

1. For each E orientation (0°, 90°, 180°, 270°) and both LogMARs (−0.20, −0.40), render
   at mean eye position p₀ = (0, 0) using `DifferentiableStimulus`.

2. Compute ∂λ/∂x and ∂λ/∂y via forward-AD using the existing `fwAD` infrastructure:

   ```python
   import torch.autograd.forward_ad as fwAD

   J_cols = []
   for param_idx in [0, 1]:           # x=0, y=1
       with fwAD.dual_level():
           tangent = torch.zeros_like(base_theta)
           tangent[:, param_idx] = 1.0
           dual_theta = fwAD.make_dual(base_theta, tangent)
           world = stim_gen(dual_theta, logmar=logmar)
           movie = retina(world, mean_trace)
           rates = compute_rate_map(model, readout, movie)
           J_cols.append(fwAD.unpack_dual(rates).tangent.detach())  # [N]

   J = np.stack(J_cols, axis=1)      # [N, 2]
   U_jac, _ = np.linalg.qr(J)        # [N, 2], orthonormal
   ```

3. Load U_pca2 from the per-orientation FEM covariance cached in
   `temporal_decoding/data/` (Priority 1). Compute per orientation per LogMAR:
   - `alignment_score(U_jac, U_pca2)` — cos² of principal angles
   - `capture_fraction(U_jac, C_FEM)` — fraction of FEM variance in the Jacobian span

4. **Null controls:** Two controls, in increasing stringency:

   - **Random rank-2 null:** Generate 500 random rank-2 subspaces (QR of random N×2
     Gaussian matrices). With N ≈ 130 and rank 2, a random subspace captures ~1.5% of
     variance; the Jacobian must substantially exceed this floor.

   - **Matched-energy null (stronger):** Construct J_null by shuffling neuron identities
     in J (permute rows) while preserving column norms. This gives a rank-2 subspace with
     identical scale and energy distribution to U_jac, but with image/model alignment
     destroyed. Compute `alignment_score(U_null, U_pca2)` and
     `capture_fraction(U_null, C_FEM)` and compare to U_jac. U_jac must outperform
     U_null; if it does not, the alignment result is explained by energy matching alone,
     not by genuine image/model structure.

5. **Orientation-invariance check:** Compute `alignment_score(U_jac_k1, U_jac_k2)` for
   all orientation pairs. If the Jacobian subspace also does not rotate with orientation
   (like U_pca2), this explains the Priority 1 finding mechanistically: the E's
   translation sensitivity is orientation-invariant at this scale.

**Runtime:** ~10 minutes (8 forward-AD passes + cheap numpy steps).

---

## Test 6 — Representational intervention via U_jac *(core causal test)*

**Priority: 1. Run this in parallel with Test 3.**

**What it tests:** Whether the Jacobian-aligned subspace helps or hurts orientation
decoding, with the sign expected to differ across LogMAR regimes.

**Two intervention variants** (both required):

### 6a — Subtraction (nuisance test)

Project out U_jac per orientation and rerun the D1 decoder:

```python
def project_out(ravg, U):
    return ravg - ravg @ U @ U.T     # [M, N]

ravg_clean_k = project_out(ravg_k, U_jac_k)
```

Report Δacc = acc(clean) − acc(original) at both LogMARs.

### 6b — Isolation (signal test)

Project each orientation's responses *onto* U_jac (keep only the 2D Jacobian component)
and decode from that 2D subspace alone:

```python
def project_onto(ravg, U):
    return ravg @ U @ U.T            # [M, N], in Jacobian span only

ravg_jac_k = project_onto(ravg_k, U_jac_k)
```

Also run the decoder on the complement:

```python
ravg_perp_k = project_out(ravg_k, U_jac_k)   # = ravg - ravg_jac
```

Report acc(U_jac), acc(U_⊥), and acc(full) at both LogMARs.

### Interpretation table

| Δacc (subtract) | acc(U_jac) | acc(U_⊥) | Interpretation |
|---|---|---|---|
| > 0 at −0.20 | Low | ≈ Full | Jacobian subspace is nuisance above threshold |
| ≈ 0 at −0.40 | Low | ≈ Full | Jacobian subspace irrelevant at hyperacuity |
| < 0 at −0.40 | Moderate | < Full | Jacobian subspace is signal-bearing at hyperacuity |
| ≈ 0 both | Low | ≈ Full | Jacobian subspace geometrically aligned but functionally redundant |

### 6c — Pooled intervention (consistency check)

Repeat 6a and 6b using a *pooled* U_jac estimated from all four orientations jointly
(QR of the concatenated [J_0°, J_90°, J_180°, J_270°], keeping top 2 singular vectors).
If orientation-specific and pooled interventions give the same Δacc, the result is not
an artifact of class-conditional geometric manipulation.

**Note:** The orientation-specific projection (6a/6b) changes each class in its own
basis, which could create classifier artifacts. The pooled version (6c) avoids this.
Both should point the same direction if the Jacobian model is correct.

---

## Test 4 — Full mechanistic model: C_FEM ≈ J Σ_eye Jᵀ *(quantitative test)*

**Priority: 2.**

**What it tests:** Whether the Jacobian model predicts not just the subspace but the
magnitude and shape of C_FEM.

### Tightened Σ_eye definition

The original plan used the covariance of trial-mean positions, which is too crude. The
model integrates over 8 GRU steps within a window, so the effective perturbation it sees
is a temporally filtered version of the eye trace, not the trial mean. Use instead:

```python
# eye_traces: [M, T, 2] — full traces
# For each trace m, compute the per-frame position deviations around the centroid
centroids = eye_traces.mean(axis=1, keepdims=True)   # [M, 1, 2]
deviations = eye_traces - centroids                   # [M, T, 2]

# Pool all (M*T) per-frame deviations to estimate effective position covariance
deviations_flat = deviations.reshape(-1, 2)          # [M*T, 2]
Sigma_eye_frame = np.cov(deviations_flat.T)          # [2, 2]

# Alternative: use only the within-window temporal covariance (8-frame GRU window)
# Weight frames by GRU temporal decay if known
```

Report results under both the trial-mean and per-frame Σ_eye definitions. If they differ,
the per-frame version is more mechanistically appropriate because the GRU weights recent
frames — the effective Σ_eye is the frame-level, not trace-mean, covariance.

**Steps:**

1. For each image/orientation, compute J from Test 3.

2. Compute predicted covariance:
   ```python
   C_predicted = J @ Sigma_eye @ J.T              # [N, N]
   ```

3. Compare C_predicted to C_FEM (empirical):
   - **Subspace alignment**: `alignment_score(U_pred, U_pca2_empirical)`
   - **Eigenvalue ratio**: `lambda_pred[0] / lambda_empirical[0]` — is the scale right?
   - **Residual fraction**: `||C_FEM - C_predicted||_F / ||C_FEM||_F`

4. Decompose the residual: what is the rank of (C_FEM − C_predicted)? If it is > 2,
   there is shared variance beyond the Jacobian model. That residual is the View B
   signature — GRU temporal mixing creating additional low-rank structure.

---

## Test 1 — Capture matrix from existing outputs *(triage, not evidence)*

**Priority: 3. Run after Tests 3 and 6 are done.**

**Status:** This uses U_grad2, which is a grid-averaged regression Jacobian, not the
analytic point Jacobian. Its results are informative as a sanity check but are not
evidentially decisive for or against the Jacobian hypothesis. A good Test 1 result is
encouraging; a mediocre result does not falsify the hypothesis — it may only mean the
regression approximation is poor.

**Steps:** Load `all_cov_results.pkl`, extract the capture matrix diagonal vs off-diagonal,
and compare to what Tests 3 and 4 find with the analytic J.

---

## Test 5 — Jacobian curvature *(only if Tests 3–4 show discrepancies)*

**Priority: 4.**

Compute J(p) at 5 positions spanning the FEM range (−0.04° to +0.04°). Check whether
`alignment_score(U_jac(p1), U_jac(p2))` degrades with |p1 − p2|. If the subspace is
stable (alignment > 0.95 across positions), the linear model is adequate. If it rotates,
the linearisation fails at the scale of natural FEM amplitudes, and the discrepancy in
Tests 3–4 is explained by curvature rather than by View B dynamics.

---

## Test 7 — Cross-image transfer via Jacobian projection *(secondary)*

**Priority: 5.**

Project response differences onto U_jac(image) for cross-image displacement decoding.
If R²_projected ≫ R²_raw (−1.3 → positive), the Jacobian functions as the per-image
decoder key and the image-specificity of the code is fully explained by the
image-specificity of J.

---

## Information-consequences module *(subordinate; run after Tests 3 and 6)*

This module interprets the functional meaning of the Jacobian structure without becoming
a second project. The four pieces below are companions to Test 6, not standalone claims.

### IC-A — Signal geometry (descriptive)

Compute C_signal (across-orientation mean covariance) and its overlap with U_jac:

```python
# class means at each LogMAR
mu_k = ravg_k.mean(axis=0)        # [N] per orientation k
mu_grand = np.mean([mu_0, mu_90, mu_180, mu_270], axis=0)
C_signal = np.cov(np.stack([mu_0, mu_90, mu_180, mu_270] - mu_grand).T)
alpha_signal = capture_fraction(U_jac, C_signal)
```

Report `alpha_signal` (overlap of Jacobian subspace with signal covariance) alongside
the Test 6 Δacc values. This is the geometric explanation for why subtraction helps,
hurts, or does nothing: high alpha_signal = Jacobian subspace overlaps the signal
direction = intervention effect will be large.

**Use as interpretation only, not as a primary result.**

### IC-B — Orthogonal-complement decoding (from Test 6)

Test 6 already computes acc(U_jac) and acc(U_⊥). Present them together as:

| Component | acc at −0.20 | acc at −0.40 |
|---|---|---|
| Full | baseline | baseline |
| U_jac (Jacobian only) | ? | ? |
| U_⊥ (complement) | ? | ? |

This is the cleanest information-consequence analysis: it directly shows where task
information lives without requiring a noise model.

### IC-C — Task-relevant SNR (simple form)

For each LogMAR, compute the signal-to-noise ratio in the Jacobian subspace:

```python
# Project class means and FEM variability onto U_jac
signal_var_jac = np.var([mu_k @ U_jac for k in orientations])  # across classes
noise_var_jac  = np.mean([capture_fraction(U_jac, C_FEM_k) * np.trace(C_FEM_k)
                          for k in orientations])               # within class
SNR_jac = signal_var_jac / (noise_var_jac + epsilon)
```

Repeat for U_⊥. Compare SNR_jac vs SNR_perp at −0.20 and −0.40. This is the signed
geometric version of "help vs hurt" without requiring Σ⁻¹ inversion.

### IC-D — Noise model sensitivity (last)

Only after IC-A through IC-C, repeat the key metrics under:
1. Poisson-only (lower bound on noise)
2. Poisson + C_FEM (primary)
3. Poisson + C_FEM + scaled residual from Aim 1 data (when available)

Report whether the sign of the conclusions (improvement vs impairment in Test 6) is
stable across noise models. If it is, the conclusions are robust. If it flips, flag
explicitly.

### IC-E — Ensemble directional amplification *(minimal; no new infrastructure)*

**Question:** When C_FEM is averaged over many images, do any stable population
directions emerge — or does image-specificity cause everything to average out?

```python
# C_FEM per image already computed in translation_covariance outputs
C_FEM_avg = np.mean([C_FEM_i for i in images], axis=0)   # [N, N]
w, V = np.linalg.eigh(C_FEM_avg)
U_avg = V[:, np.argsort(w)[::-1][:2]]                    # top-2 eigenvectors
```

Compare under three conditions:
1. **Biological Σ_eye** (measured traces) — the real ensemble
2. **Isotropic / shuffled Σ_eye** — matched amplitude, no structure
3. **Stabilized** (Σ_eye → 0) — baseline with no FEM

Questions to answer:
- Does C_FEM_avg have rank > 2? If yes, there is stable cross-image shared structure that
  the per-image rank-2 Jacobian model cannot explain.
- Do the top eigenvectors of C_FEM_avg show interpretable structure (e.g. systematic
  orientation or spatial-frequency preferences across the neuron population)?
- Are those eigenvectors different under biological vs. isotropic Σ_eye? If yes, the
  biological FEM statistics selectively amplify specific population directions that
  isotropic motion does not.

**Scope:** This is a descriptive probe, not a hypothesis test. It takes ~5 minutes to run
(numpy only, using existing outputs). A positive result (stable structure in C_FEM_avg,
altered by eye movement statistics) motivates future work on optimal eye movement
policies. A null result (C_FEM_avg ≈ 0 or isotropic) confirms that image-specificity
fully dominates and that no population direction is systematically amplified.

**Do not over-interpret.** This is not a claim about whitening, optimality, or active
sensing policy. It is a sanity check on whether there is any stable structure to study.

---

## Final ordering

```
WEEK 1
  Test 3 — Analytic Jacobian (E, forward-AD)          ~10 min compute
  Test 6 — Representational intervention               ~30 min compute
      (runs in parallel with Test 3 once U_jac exists)

WEEK 2
  Test 4 — Full mechanistic model C_FEM ≈ J Σ_eye Jᵀ  ~30 min compute
  IC module — Signal geometry, SNR, orthogonal decomp  ~1 hr analysis

WEEK 3 (if needed)
  Test 1 — Capture matrix sanity check                 <1 hr (load existing)
  Test 5 — Curvature (only if Tests 3–4 fail)
  Test 7 — Cross-image transfer
```

---

## What would stop the analysis

**Stop saying "Jacobian" in a strong mechanistic sense if:**
- Test 3 alignment < 0.70 AND
- Test 4 residual > 40% of empirical variance AND
- Test 5 shows substantial curvature across the FEM amplitude range

In that case, the covariance is consistent with View B (GRU temporal mixing) rather than
View A (static Jacobian pushforward), and the description should shift accordingly.

**Strongest possible outcome:**
- Test 3: alignment > 0.90, U_jac is orientation-invariant (explains Priority 1)
- Test 4: C_predicted ≈ C_FEM up to ~10% residual
- Test 6: Δacc positive at −0.20 (nuisance) and near-zero at −0.40 (irrelevant at
  hyperacuity), acc(U_⊥) ≈ acc(full) confirming task information lives in the complement
- IC-A: alpha_signal matches the Δacc sign
- IC-C: SNR_jac < SNR_perp at both LogMARs (Jacobian subspace is the noise direction)

That would be a fully specified, mechanistically grounded, causally tested account of why
FEM-driven correlations behave differently above and below the resolution limit.
