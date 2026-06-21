# Jacobian Identity/Transformation Geometry: Implementation Handoff

**Project scope:** deterministic encoding geometry of FEM-induced variability in a V1 digital twin  
**Primary goal:** show that the image-translation Jacobian identifies local transformation tangent spaces in V1 population activity, and use those tangent spaces to quantify when retinal translation can mimic or obscure stimulus identity.  
**Do not include for this phase:** noisy observer models, V2/Rowley hierarchy analyses, behavioral hyperacuity claims, trajectory optimization.

---

## 0. Conceptual framing for the coding agent

The digital twin is deterministic during inference. It should be treated as a **sensory transducer**, not as a behavioral observer. Any analysis that gives identical retinal input for every trial within a class will produce identical rate vectors and trivially perfect decoding. Do not interpret those fixed-input conditions as biological observer baselines.

The scientifically meaningful object here is the **geometry of the deterministic rate manifold**:

\[
r = f(I)
\]

where \(I\) is a retinal image or movie, and \(r \in \mathbb{R}^N\) is the population rate vector.

For small retinal translations, responses can be locally approximated by:

\[
r(I(x+\Delta x, y+\Delta y)) \approx r(I) + J_I \Delta p,
\]

where:

\[
J_I =
\left[
\frac{\partial r}{\partial x},
\frac{\partial r}{\partial y}
\right],
\quad
\Delta p =
\begin{bmatrix}
\Delta x \\
\Delta y
\end{bmatrix}.
\]

The columns of \(J_I\) define the **local translation tangent plane**: the population directions corresponding to “the same stimulus, slightly translated.” This is the core object.

The new analyses below ask:

1. How well does the Jacobian tangent plane explain FEM-induced covariance?
2. When can translation of one identity mimic another identity?
3. How does this identity/transformation confusability vary with LogMAR and retinal phase?
4. Which FEM components contribute to the translation tangent geometry?

The main contribution should be framed as:

> The image-translation Jacobian identifies local transformation tangent spaces in V1 population activity. FEM-induced variability reveals these tangent spaces during normal fixation. The overlap between transformation tangent directions and identity-signal directions determines when self-induced retinal motion will mimic, obscure, or spare identity information.

---

## 1. Existing results and constraints that should guide implementation

### 1.1 Results to preserve

The current project already has several stable results:

- Real-data covariance decomposition: FEM explains a large fraction of shared V1 variability.
- Digital twin: model correlations are reafferent by construction, with no independent neural noise.
- Jacobian direction result: leading FEM covariance directions align with image-translation Jacobian directions.
- Displacement decoding: within-image displacement decoding is near-perfect, but cross-image generalization fails, implying image-specific translation manifolds.
- E-optotype geometry: real vs trial-mean stabilized comparisons show regime-dependent class-separation changes.
- Temporal residual features: no evidence for useful orientation information beyond time-averaged rate geometry.
- Model-native saturation: nominal LogMAR values below approximately −0.40 are not independent E-size conditions on the model-native 37.5 ppd grid.

### 1.2 Results to avoid over-interpreting

- `fixed_center` D1 = 1.000 is degenerate in a deterministic model. Use it only as a deterministic oracle / sanity check.
- `0×` amplitude is also degenerate and should not be included in biological amplitude trends.
- Ablation effects are not uniquely dynamic-FEM-specific if old trial-mean stabilized conditions show similar gains.
- Avoid claiming that FEM “helps behavior” or “hurts behavior” in general. Use language like “deterministic class separability” or “rate-geometry separability.”

### 1.3 Key model-native rendering caveat

Treat lm = −0.40, −0.45, and −0.50 as a saturation plateau unless specifically auditing high-PPD rendering. The model-native retinal inputs are effectively identical across −0.40 to −0.50. Use −0.35 as the primary interpretable onset-of-subpixel condition.

Recommended LogMARs for new analyses:

```text
primary:  -0.20, -0.25, -0.30, -0.35
optional plateau control: -0.40
avoid as independent points: -0.45, -0.50
omit: -0.55
```

---

## 2. Analysis A: Translation mimicry

### 2.1 Motivation

This is the highest-priority new analysis.

The goal is to quantify whether the neural difference between two identities can be explained as a small translation of one identity.

For two E orientations \(a\) and \(b\):

\[
d_{a\rightarrow b} = \mu_b - \mu_a
\]

where \(\mu_a\) and \(\mu_b\) are deterministic class mean population rate vectors.

Given the translation Jacobian for identity \(a\), \(J_a \in \mathbb{R}^{N \times 2}\), solve:

\[
\Delta p^\ast =
\arg\min_{\Delta p}
\left\|
\mu_b - (\mu_a + J_a \Delta p)
\right\|_2^2.
\]

Equivalently:

\[
\Delta p^\ast =
\arg\min_{\Delta p}
\left\|
d_{a\rightarrow b} - J_a \Delta p
\right\|_2^2.
\]

If \(J_a \Delta p^\ast\) explains much of \(d_{a\rightarrow b}\), then the identity difference between \(a\) and \(b\) is locally confusable with a translation of \(a\).

This directly tests identity/transformation separation.

### 2.2 Inputs

Use existing cached rates and Jacobian files where possible.

Required inputs per LogMAR:

- Orientation labels: `0`, `90`, `180`, `270`
- Population rate vectors for each orientation, preferably time-averaged rates:
  - shape expected: `(n_orientations, n_trials, n_neurons)` or equivalent
  - for deterministic class means, compute \(\mu_k = \text{mean over trials}\)
- Translation Jacobian for each orientation:
  - shape expected: `(n_orientations, n_neurons, 2)`
  - if multiple positions/traces are available, use the same Jacobian convention as the previous successful Jacobian direction analysis:
    - preferred: position/histogram-weighted integrated Jacobian if available
    - fallback: central finite-difference Jacobian at the matched retinal phase

Recommended conditions:

- `real`
- old `stabilized` / trial-mean stabilized if rates/Jacobians exist
- optionally `fixed_center` only as a deterministic reference, not as main baseline

Recommended LogMARs:

```text
-0.20, -0.25, -0.30, -0.35
```

Add `-0.40` only as a model-native saturation control.

### 2.3 Core metrics

For each ordered pair \(a \rightarrow b\), \(a \neq b\):

#### 2.3.1 Least-squares translation

Use ridge-stabilized least squares:

\[
\Delta p^\ast =
(J_a^\top J_a + \lambda I)^{-1}
J_a^\top d_{a\rightarrow b}.
\]

Use a small ridge:

```python
ridge = 1e-6 * trace(J.T @ J) / 2
```

or similar scale-normalized ridge.

Save:

```text
dx_star
dy_star
translation_magnitude_deg
translation_magnitude_arcmin
translation_angle_rad
```

#### 2.3.2 Mimicry fraction (primary metric: orthonormal projection)

The primary mimicry metric uses the orthonormal basis of \(J_a\), computed via SVD, not least squares:

1. Compute \(J_a = U S V^\top\) via thin SVD.
2. Keep columns of \(U\) with \(S_i > \epsilon \cdot S_0\) (non-negligible singular values).
3. Report:

\[
M_{a\rightarrow b}
=
\frac{
\|U U^\top d_{a\rightarrow b}\|_2^2
}{
\|d_{a\rightarrow b}\|_2^2 + \epsilon
}.
\]

This directly answers "how much of the identity vector lies in the translation tangent subspace?" and is numerically stable even when \(J\) is ill-conditioned, because it never inverts \(J^\top J\).

The least-squares translation \(\Delta p^\ast\) is a **secondary, interpretive** quantity used for the constrained mimicry scores and the translation-vector figures (Section 2.7 Figure C). Compute it in addition to the projection metric but report the projection metric as the headline mimicry score. The two should agree closely when \(J\) is well-conditioned; disagreement at saturation LogMARs is a diagnostic signal.

#### 2.3.3 Residual identity fraction

\[
R_{a\rightarrow b}
=
\frac{
\|d_{a\rightarrow b} - J_a \Delta p^\ast\|_2^2
}{
\|d_{a\rightarrow b}\|_2^2 + \epsilon
}.
\]

For a pure orthogonal projection:

\[
R \approx 1 - M.
\]

#### 2.3.4 Cosine alignment

\[
\cos\theta_{a\rightarrow b}
=
\frac{
\|P_{J_a} d_{a\rightarrow b}\|_2
}{
\|d_{a\rightarrow b}\|_2 + \epsilon
}.
\]

This is \(\sqrt{M}\). Report either this or principal-angle-equivalent values if useful.

#### 2.3.5 FEM-constrained mimicry

The unconstrained least-squares solution may require a translation outside the biological FEM range. Compute a constrained version for each limit:

```text
r_max_arcmin ∈ {0.5, 1.0, 2.0}
```

or use empirical FEM displacement percentiles (median RMS, 95th percentile).

**Exact constrained LS in 2D:** if the unconstrained \(\|\Delta p^\ast\| \le r_\text{max}\), use it directly. Otherwise, the true optimum under \(\|\Delta p\| \le r_\text{max}\) lies on the boundary circle. Find it by sampling 720 angles uniformly on the circle and taking the angle that minimises \(\|d - J_a \Delta p\|_2^2\). For a 2D parameter space this is exact to within angular resolution and cheap.

```python
def constrained_mimicry(d, Ja, r_max_deg, n_angles=720):
    if np.linalg.norm(delta_unconstrained) <= r_max_deg:
        delta_c = delta_unconstrained
    else:
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        best_M, best_delta = -np.inf, None
        for ang in angles:
            dc = r_max_deg * np.array([np.cos(ang), np.sin(ang)])
            pred = Ja @ dc
            m = float(pred @ pred)
            if m > best_M:
                best_M, best_delta = m, dc
        delta_c = best_delta
    pred_c = Ja @ delta_c
    return float(pred_c @ pred_c) / (float(d @ d) + 1e-12)
```

Label this "constrained mimicry" in outputs, **not** "projection fraction."

Save both:

```text
mimicry_unconstrained (primary: orthonormal projection)
mimicry_constrained_0p5_arcmin (exact 2D constrained LS)
mimicry_constrained_1p0_arcmin
mimicry_constrained_2p0_arcmin
```

### 2.4 Symmetry and pair structure

Compute ordered pairs, not just unordered pairs:

```text
0→90, 90→0, 0→180, 180→0, ...
```

Because:

\[
J_a \neq J_b
\]

in general.

Save both ordered and symmetrized summaries:

```python
M_sym[a,b] = 0.5 * (M[a,b] + M[b,a])
M_max[a,b] = max(M[a,b], M[b,a])
M_min[a,b] = min(M[a,b], M[b,a])
```

### 2.5 Expected patterns

Predictions:

- Mimicry should be high at LogMARs where translation and identity are confounded.
- Mimicry should correlate with the existing α metric, but may be sharper because it works on pairwise class means.
- Mimicry may be high around −0.20/−0.25 where α spikes.
- Mimicry should decrease or change structure near −0.30/−0.35 if identity separates from translation tangent directions.
- Values at −0.40 should be interpreted as plateau/saturation controls.

### 2.6 Output files

Create a results directory, for example:

```text
declan/results/translation_mimicry/
```

Save:

```text
translation_mimicry_summary.csv
translation_mimicry_by_logmar.npz
translation_mimicry_config.json
```

CSV columns:

```text
logmar
condition
orientation_a
orientation_b
mimicry_unconstrained
residual_unconstrained
cosine_alignment
dx_star_deg
dy_star_deg
translation_mag_deg
translation_mag_arcmin
translation_angle_rad
mimicry_constrained_0p5_arcmin
mimicry_constrained_1p0_arcmin
mimicry_constrained_2p0_arcmin
identity_norm
jacobian_norm_x
jacobian_norm_y
jacobian_condition_number
ridge_lambda
```

Also save full matrices in `.npz`:

```python
{
  "logmars": ...,
  "orientations": ...,
  "mimicry": ...,        # shape (L, C, 4, 4), diagonal NaN
  "residual": ...,
  "dx_star": ...,
  "dy_star": ...,
  "translation_mag": ...,
  "mimicry_constrained": ...,
  "identity_norm": ...,
}
```

### 2.7 Figures

#### Figure A: mimicry vs LogMAR

Plot mean pairwise mimicry across LogMAR:

- y-axis: mean \(M_{a\rightarrow b}\), excluding diagonal
- x-axis: LogMAR
- show unconstrained and 1 arcmin constrained versions
- shade model-native saturation plateau at −0.40 and smaller
- optionally overlay α on secondary y-axis

#### Figure B: 4×4 mimicry matrix per LogMAR

For each LogMAR, plot an ordered-pair heatmap:

```text
rows = source orientation a
columns = target orientation b
value = M_{a→b}
diagonal = NaN or black
```

Use the same color scale across LogMARs.

#### Figure C: optimal translation vectors

For each ordered pair \(a\rightarrow b\), plot \(\Delta p^\ast\) as arrows in a small 2D diagram.

- Color by target orientation.
- Separate panels per source orientation or per LogMAR.
- Mark 1 arcmin circle and 2 arcmin circle to show whether solutions are biologically/local plausible.

#### Figure D: mimicry predicts class-separation change

Scatter:

```text
x = mean mimicry at LogMAR
y = real-vs-stabilized D1 difference or class-separation difference
```

Use old trial-mean stabilized comparison, not fixed_center.

Goal: show whether higher mimicry predicts regimes where dynamic translation obscures or reshapes identity separability.

### 2.8 Pseudocode

```python
import numpy as np

def orthonormal_basis(J, eps=1e-9):
    # J: (N, 2)
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    keep = S > eps * S[0]
    return U[:, keep], S

def compute_mimicry(mu, J_by_ori, orientations, ridge_scale=1e-6, arcmin_limits=(0.5, 1.0, 2.0),
                    svd_eps=1e-9, n_constrained_angles=720):
    # mu: dict ori -> (N,)
    # J_by_ori: dict ori -> (N, 2), units should be response per degree or per same unit as delta
    rows = []
    for a in orientations:
        Ja = J_by_ori[a]

        # Primary: SVD basis for robust projection metric
        U, S, _ = np.linalg.svd(Ja, full_matrices=False)
        keep = S > svd_eps * S[0]
        Ua = U[:, keep]

        # Secondary: ridge least squares for delta_star and constrained variants
        JTJ = Ja.T @ Ja
        ridge = ridge_scale * np.trace(JTJ) / max(1, JTJ.shape[0])
        A = JTJ + ridge * np.eye(2)
        cond = float(np.linalg.cond(A))

        for b in orientations:
            if a == b:
                continue
            d = mu[b] - mu[a]
            norm_d2 = float(d @ d) + 1e-12

            # Primary mimicry: orthonormal projection
            proj = Ua @ (Ua.T @ d)
            M_proj = float(proj @ proj) / norm_d2
            R_proj = float((d - proj) @ (d - proj)) / norm_d2

            # Secondary: least-squares delta_star for interpretive outputs
            delta = np.linalg.solve(A, Ja.T @ d)
            mag_deg = float(np.linalg.norm(delta))
            mag_arcmin = mag_deg * 60.0

            row = dict(
                orientation_a=a,
                orientation_b=b,
                mimicry_unconstrained=M_proj,       # PRIMARY: SVD projection
                residual_unconstrained=R_proj,
                cosine_alignment=float(np.sqrt(max(0.0, M_proj))),
                dx_star_deg=float(delta[0]),
                dy_star_deg=float(delta[1]),
                translation_mag_deg=mag_deg,
                translation_mag_arcmin=mag_arcmin,
                translation_angle_rad=float(np.arctan2(delta[1], delta[0])),
                identity_norm=float(np.sqrt(norm_d2)),
                jacobian_norm_x=float(np.linalg.norm(Ja[:, 0])),
                jacobian_norm_y=float(np.linalg.norm(Ja[:, 1])),
                jacobian_condition_number=cond,
                jacobian_rank=int(keep.sum()),
                ridge_lambda=float(ridge),
            )

            # Constrained mimicry: exact 2D boundary search
            for lim in arcmin_limits:
                lim_deg = lim / 60.0
                if mag_deg <= lim_deg:
                    delta_c = delta
                else:
                    angles = np.linspace(0, 2 * np.pi, n_constrained_angles, endpoint=False)
                    candidates = lim_deg * np.stack([np.cos(angles), np.sin(angles)], axis=1)
                    preds = (Ja @ candidates.T)  # (N, n_angles)
                    scores = np.einsum('ni,ni->i', preds, preds)
                    delta_c = candidates[np.argmax(scores)]
                pred_c = Ja @ delta_c
                M_c = float(pred_c @ pred_c) / norm_d2
                row[f"mimicry_constrained_{str(lim).replace('.', 'p')}_arcmin"] = M_c

            rows.append(row)

    return rows
```

### 2.9 Validation checks

Before trusting results:

- Confirm Jacobian units match translation units.
- Confirm \(J\Delta p\) has the same scale as \(\mu_b-\mu_a\).
- Check condition number of \(J^\top J\); if high, use orthonormal projection instead of raw least squares. Runs with cond > 1e6 should be flagged as suspect. At saturation LogMARs, a specific failure mode: both Jacobian columns may go to near-zero simultaneously (the model output is flat and insensitive to translation), making \(J^\top J\) near-singular regardless of ridge. Inspect and flag these separately — they indicate retinal input saturation, not a solver issue.
- Check optimal translations are in plausible local range.
- Check results are stable to ridge size.
- Compare projection-based mimicry with least-squares mimicry.
- Verify pairwise identity norms are nonzero and not saturated/blank.
- Exclude or flag LogMARs in model-native saturation plateau.
- **Normalization sensitivity analysis:** run all mimicry metrics in raw-rate Euclidean space as the primary result. Also run once with neuron-wise z-scoring (zero mean, unit variance across the trial-mean rate vectors) as a sensitivity check. The qualitative LogMAR trend should be robust; if raw and normalized results diverge substantially, inspect which neurons dominate the projection — a small number of high-rate units could be driving the result. Report both in supplementary if they disagree.

---

## 3. Analysis B: Phase landscape

### 3.1 Motivation

The old trial-mean stabilized condition already samples empirical static retinal phases, while `fixed_center` is a deterministic single-phase oracle. We need to map the deterministic response geometry over local subpixel phase to understand where these conditions lie.

This is an encoding-geometry analysis, not a noisy decoder.

Questions:

1. Is `fixed_center` a lucky phase, an average phase, or simply a degenerate deterministic anchor?
2. How does class separation vary over subpixel phase?
3. How does the translation tangent plane vary over phase?
4. Do real FEM trajectories sample phases with different identity/transformation geometry than trial means?

### 3.2 Inputs

Use the existing E-optotype stimulus generation and model inference paths.

Required:

- E-optotype renderer
- model checkpoint: `learned_resnet_none_convgru_gaussian`, epoch 147
- retinal grid parameters matching the existing model-native analyses
- orientations: `0`, `90`, `180`, `270`
- LogMARs: `-0.20`, `-0.25`, `-0.30`, `-0.35`, optional `-0.40`

### 3.3 Phase grid

Define a local retinal offset grid centered around the grand-mean position or nominal fixation center.

Recommended units:

- Use degrees internally.
- Define grid in model pixels and convert to degrees using retina ppd.

Model-native ppd:

```text
retina_ppd = 37.50476617  # or whatever exact value is used in existing rates
pixel_deg = 1 / retina_ppd
```

Recommended grid:

```text
range: ±3 model pixels (~±4.8 arcmin at 37.5 ppd)
resolution: 25×25 or 33×33
```

This range is necessary to contain the empirical FEM trajectory cloud. Real FEMs have RMS amplitude around 6.6 arcmin, so a ±1 pixel (±1.6 arcmin) grid covers less than one FEM standard deviation and the empirical eye positions will mostly fall outside it. ±3 pixels ensures the trial-mean stabilized position cloud and the real FEM positions are contextualized within the landscape.

If runtime is a concern, ±2 pixels at 17×17 is the minimum that still meaningfully includes the bulk of empirical positions — do not go narrower.

Examples:

```python
offset_pix = np.linspace(-3.0, 3.0, 25)
offset_deg = offset_pix / retina_ppd
```

**Recommended two-pass strategy to manage runtime:**

- **Coarse pass:** 17×17 over ±3 pixels for all LogMARs and orientations. This is the default run.
- **Fine pass:** 33×33 over ±3 pixels for selected LogMARs only — recommended −0.20 (highest mimicry regime) and −0.35 (subpixel onset). Run fine only after the coarse pass confirms the grid range is appropriate.

Do not attempt 33×33 across all LogMARs in a single run; the coarse pass is sufficient for the completion criteria.

### 3.4 Response computation

For each LogMAR, orientation, and phase offset:

1. Render a static E at that fixed retinal offset.
2. Run deterministic model inference.
3. Compute time-averaged population rate vector or the same readout representation used in previous D1 analyses.
4. Store:
   - `rates[logmar, orientation, x_idx, y_idx, neuron]`
   - optionally map-level features if available.

Important: keep the input deterministic, but do not interpret classification accuracy as behavioral performance. Use it as class geometry.

### 3.5 Metrics

For each phase offset \((x,y)\), compute:

#### 3.5.1 Class means

Since deterministic rates per phase are single vectors:

\[
\mu_k(x,y) = r_k(x,y)
\]

for orientation \(k\).

#### 3.5.2 Signal covariance

\[
C_\text{signal}(x,y)
=
\text{Cov}_k(\mu_k(x,y)).
\]

Save:

- trace of \(C_\text{signal}\)
- top eigenvalues
- participation ratio
- top signal subspace

#### 3.5.3 Pairwise class separation

For each pair \(a,b\):

\[
D_{ab}(x,y) = \|\mu_a(x,y) - \mu_b(x,y)\|_2
\]

Optionally normalize by mean rate norm.

Save:

- mean pairwise separation
- minimum pairwise separation
- pairwise matrices

#### 3.5.4 Jacobian norm and anisotropy

Compute \(J_k(x,y)\) at each phase or use finite differences over the phase grid.

Metrics:

```text
||J_x||
||J_y||
sqrt(trace(J^T J))
condition number of J^T J
anisotropy ratio = singular_value_1 / singular_value_2
```

#### 3.5.5 Translation/signal alignment

At each phase, compute the alignment between the translation tangent subspace and signal covariance:

\[
\alpha(x,y)
=
\frac{
\text{tr}(U_J^\top C_\text{signal} U_J)
}{
\text{tr}(C_\text{signal}) + \epsilon
}.
\]

There are two possible choices for \(U_J\):

1. orientation-specific \(U_{J_k}\), then average over \(k\)
2. pooled \(U_J\) from concatenating Jacobians across orientations

Save both if feasible:

```text
alpha_orientation_mean
alpha_pooled
principal_angles
```

#### 3.5.6 Translation mimicry over phase

For each phase, run Analysis A using phase-specific class means and Jacobians.

**This is recommended, not optional.** It is one of the most informative metrics in the landscape: it tells you whether mimicry varies systematically with phase or is roughly constant. If mimicry varies substantially across the grid, FEM-driven confusability is phase-dependent — which matters for interpreting α reversals across LogMAR. If mimicry is roughly constant, confusability is a stimulus-scale phenomenon rather than a phase phenomenon. Either result is interpretable and important.

Save:

```text
mean_mimicry(x,y)
max_mimicry(x,y)
pairwise_mimicry(x,y,a,b)
```

### 3.6 Overlay empirical eye positions

Use existing eye traces to extract:

- grand mean eye position
- trial mean positions
- real FEM frame positions
- old stabilized trial means

Project them into the same phase-offset coordinates used for the grid.

Overlay on phase maps:

- fixed_center marker
- trial-mean stabilized position cloud
- real FEM position cloud or density contours

### 3.7 Output files

Directory:

```text
declan/results/phase_landscape/
```

Save:

```text
phase_landscape_rates.npz
phase_landscape_metrics.npz
phase_landscape_summary.csv
phase_landscape_config.json
```

Suggested `.npz` keys:

```python
{
  "logmars": ...,
  "orientations": ...,
  "offset_x_pix": ...,
  "offset_y_pix": ...,
  "offset_x_deg": ...,
  "offset_y_deg": ...,
  "rates": ...,                    # (L, O, X, Y, N)
  "signal_trace": ...,             # (L, X, Y)
  "signal_top_eigs": ...,          # (L, X, Y, K)
  "mean_pairwise_sep": ...,        # (L, X, Y)
  "min_pairwise_sep": ...,         # (L, X, Y)
  "jacobian_norm": ...,            # (L, O or pooled, X, Y)
  "alpha_pooled": ...,             # (L, X, Y)
  "alpha_orientation_mean": ...,   # (L, X, Y)
  "mean_mimicry": ...,             # optional
  "eye_trial_mean_xy": ...,
  "eye_frame_xy": ...,
}
```

### 3.8 Figures

#### Figure A: class separation landscape

Heatmap over phase offsets:

```text
value = mean pairwise class separation or min pairwise separation
```

Separate panels for LogMAR.

Overlay:

- fixed_center marker
- trial-mean stabilized cloud
- real FEM density contour

#### Figure B: α landscape

Heatmap:

```text
value = alpha_pooled or alpha_orientation_mean
```

Shows where translation tangent directions overlap with identity signal.

#### Figure C: mimicry landscape

Heatmap:

```text
value = mean translation mimicry
```

High values indicate phases where identity differences are most translation-confusable.

#### Figure D: fixed_center contextualization

Histogram of class separation over phase grid. Add vertical lines:

- fixed_center
- median trial-mean-stabilized position
- average over empirical trial means
- average over real FEM frame positions

This figure answers:

> Was fixed_center a privileged phase or just one point in the landscape?

### 3.9 Validation checks

- Confirm phase offsets produce distinct model-native retinal inputs at each LogMAR.
- Exclude/flag −0.40 if landscape becomes saturated.
- Verify finite-difference Jacobians over the phase grid match previously computed Jacobians at center.
- Make sure orientation-specific rates are not accidentally reused across phase offsets.
- Check for edge/cropping artifacts at ±1 pixel offsets.

---

## 4. Analysis C: FEM component decomposition

> **Priority: secondary.** The must-have analyses are translation mimicry (Section 2), phase landscape (Section 3), and the central Jacobian figure (Section 5). FEM component decomposition is biologically valuable and well-specified, but it is less central to the identity/transformation framing and can expand scope if undertaken before the first three are stable. Implement Section 4 only after Sections 2, 3, and 5 are complete and passing their validation checks.

### 4.1 Motivation

The current analyses treat FEM as a unitary trace. Biologically, FEMs include drift, microsaccades, tremor, and slow offsets. We want to know which components generate which parts of the translation geometry.

This remains deterministic and encoding-side.

Questions:

1. Which eye-movement components dominate FEM-induced covariance?
2. Which components align most strongly with the translation Jacobian?
3. Which components overlap with identity-signal directions?
4. Which components drive translation mimicry/confusability?

### 4.2 Conditions to implement

Use existing eye traces and derive component-specific counterfactual traces.

Minimum set:

```text
real
drift_only
microsaccade_only
drift_without_microsaccades
lowpass_drift
highpass_jitter
trial_mean_stabilized
```

Optional:

```text
microsaccade_times_shuffled
microsaccade_directions_shuffled
drift_amplitude_scaled_0.5
drift_amplitude_scaled_2.0
microsaccade_amplitude_scaled_0.5
microsaccade_amplitude_scaled_2.0
```

### 4.3 Component extraction

If microsaccade labels already exist, use them. At the start of the script, add an explicit check that logs and saves to metadata which microsaccade detection method is being used (pre-existing labels vs. velocity threshold detection, including threshold value and any merge parameters). Different detection methods give different segmentations; reproducibility requires recording which was used.

Basic detector:

1. Smooth eye position lightly if needed.
2. Compute velocity:
   \[
   v_t = \|\Delta p_t\|/\Delta t
   \]
3. Detect candidate microsaccades where velocity exceeds threshold:
   - threshold could be percentile-based, e.g. 99th or 99.5th percentile
   - or absolute threshold if existing convention exists
4. Merge events within short gaps.
5. Store onset, offset, peak time, amplitude vector.

Construct traces:

#### drift_without_microsaccades

Replace microsaccade intervals with interpolation between pre- and post-event positions.

#### microsaccade_only

Start from trial mean or zero trace, insert microsaccade displacement transients, otherwise hold position constant.

#### lowpass_drift

Low-pass filter eye trace below microsaccade frequency band.

#### highpass_jitter

Subtract lowpass trace from real trace, optionally excluding microsaccades.

### 4.4 Rate caching

For each component condition and LogMAR:

Recommended LogMARs:

```text
-0.20, -0.25, -0.30, -0.35
```

Optional plateau control:

```text
-0.40
```

Generate model rates using the same pipeline as existing E-optotype cached rates.

Save with explicit condition names:

```text
rates_eoptotype_logmar_m0p20_condition_drift_only.npz
...
```

Ensure metadata stores:

```text
condition
trace_source
component_extraction_params
microsaccade_threshold
filter_cutoffs
retina_ppd
world_ppd
model_checkpoint
```

### 4.5 Metrics

For each condition and LogMAR:

#### 4.5.1 FEM covariance

Compute covariance over trial/time-averaged rate vectors:

\[
C_\text{component}
\]

Save:

- trace
- top eigenvalues
- participation ratio
- top-2 subspace

#### 4.5.2 Jacobian alignment

Align top covariance subspace with translation Jacobian subspace:

\[
\text{align}(U_C, U_J)
\]

Use the same metric as previous Jacobian direction analysis.

#### 4.5.3 Signal alignment α

Compute:

\[
\alpha =
\frac{\text{tr}(U_C^\top C_\text{signal} U_C)}
{\text{tr}(C_\text{signal})}.
\]

This says how much component-induced variability overlaps identity signal.

#### 4.5.4 Class-separation summary

Use deterministic class geometry metrics, not behavioral accuracy:

- pairwise class separation
- class separation after projecting out component covariance subspace
- change in separation after ablation

If D1 is used, label it explicitly as deterministic class separability under rate variation, not behavioral accuracy.

#### 4.5.5 Translation mimicry

Run Analysis A using component-specific class means/covariance if appropriate.

For component decomposition, the most useful thing may be:

- compare component covariance subspace to pairwise identity vectors
- compare component covariance subspace to Jacobian tangent plane
- compare component amplitude to mimicry scores

### 4.6 Figures

#### Figure A: covariance magnitude by component

Bar plot:

```text
component vs trace(C_component)
```

Separate panels per LogMAR.

#### Figure B: subspace alignment with Jacobian

Bar plot:

```text
component vs alignment(U_component, U_J)
```

#### Figure C: signal alignment α by component

Bar plot or line plot:

```text
component vs α
```

This identifies which components are most identity-confounding.

#### Figure D: component covariance ellipses in Jacobian plane

Project component-induced population responses into the \(J_x,J_y\) plane and show covariance ellipses.

This figure will make clear whether drift and microsaccades occupy the same or different translation directions.

### 4.7 Validation checks

- Confirm component traces reconstruct real trace approximately:
  \[
  \text{drift_without_microsaccades} + \text{microsaccade_only} \approx \text{real}
  \]
  where appropriate.
- Confirm component RMS values are plausible.
- Confirm no interpolation creates discontinuities worse than original microsaccades.
- Inspect example traces visually.
- Compare retinal input norm for each component condition.
- Ensure component conditions do not accidentally collapse to fixed_center.
- **Drift reconstruction caveat:** The `drift_without_microsaccades` trace interpolates across removed microsaccade intervals, but microsaccades cover ground — pre- and post-event positions may differ by 10–30 arcmin. Interpolating across that gap introduces a slow ramp that was not in the original drift. Mitigations: (a) concatenate inter-microsaccade segments, eliminating the gaps entirely and shortening the trace, then run analysis on the concatenated segments separately; or (b) keep interpolation but report explicitly which proportion of frames were reconstructed, and flag any traces where reconstructed frames dominate the duration. Whichever approach is used, document it in the config metadata.

---

## 5. Central Jacobian demonstration figure

This should be the main conceptual figure of the paper.

### 5.1 Goal

Make the Jacobian tangible:

> The Jacobian maps tiny eye movements into population-rate changes. It defines the local tangent plane of the same-stimulus-under-translation manifold. FEM-induced covariance is the eye-position covariance pushed through this image-specific tangent plane.

### 5.2 Suggested panels

#### Panel A: image translation derivatives

Show one stimulus image or E-optotype at center, plus tiny +x and +y translations.

Label:

\[
J_x = \partial r/\partial x,\quad J_y = \partial r/\partial y.
\]

#### Panel B: population tangent plane

Schematic or PCA projection:

- point \(r(I)\)
- vector \(J_x\)
- vector \(J_y\)
- plane spanned by \(J_x,J_y\)

Label as:

```text
same image, different retinal positions
```

#### Panel C: FEM trajectory in Jacobian coordinates

For real FEM responses, project population response differences into the orthonormal basis of \(J\):

\[
z_t = U_J^\top (r_t - \bar r)
\]

Plot:

- trajectory over time
- scatter/cloud over many traces
- optionally overlay actual eye trace scaled into same coordinate system

Expected: FEM-driven response cloud should lie strongly in the Jacobian plane.

#### Panel D: empirical vs predicted covariance ellipse

In the \(J\)-plane, plot:

- empirical FEM response covariance ellipse
- predicted covariance ellipse from:
  \[
  J\Sigma_\text{eye}J^\top
  \]

This is the visual “pushforward” demonstration.

#### Panel E: real-data bridge — empirical V1 FEM covariance in J-subspace (conditional)

**If a matched real-data/model mapping exists for the same stimulus set** (aligned stimuli, units, eye traces, and a reliable correspondence between model neurons and recorded neurons), project empirical V1 FEM covariance into the model-predicted J-subspace and report the variance fraction captured:

\[
\text{bridge fraction} = \frac{\text{tr}(U_J^\top C_\text{real} U_J)}{\text{tr}(C_\text{real}) + \epsilon}
\]

If this is feasible, include it in the main figure — it would be the strongest mechanistic claim. If this panel cannot fit in the main figure, it must be the immediately following figure, not supplementary.

**If a matched mapping does not exist** (e.g., the Fig 2 dataset uses natural images while the Jacobian analyses use E-optotypes, or there is no unit-level model-to-recording alignment), do not attempt the projection. Instead, present the existing real-data covariance decomposition result as the empirical motivation panel — it motivates why the Jacobian is the right object without requiring a joint projection that may be ill-posed.

#### Panel F: translation mimicry across LogMAR

Plot mean pairwise translation mimicry across LogMAR.

This connects the Jacobian tangent plane to identity/transformation confusability.

Shade saturation plateau at −0.40 and smaller.

### 5.3 Supplementary figure: image specificity

A supplementary figure should show that the tangent plane is image-specific:

- compare two natural images or two E phases
- show their \(J\) planes differ
- show within-image displacement decoding succeeds, cross-image fails

This was previously designated as an optional companion figure. It has been moved to supplementary to make room for the real-data bridge (Panel E) in the main figure, which is the stronger empirical claim.

---

## 6. Recommended code organization

### 6.1 New scripts

Create three new scripts:

```text
declan/translation_mimicry.py
declan/phase_landscape.py
declan/fem_component_decomposition.py
```

Optional figure script:

```text
declan/figure_jacobian_identity_geometry.py
```

### 6.2 Shared utilities

If not already available, create or extend utilities:

```text
declan/geometry_utils.py
```

Functions:

```python
orthonormal_basis(J)
subspace_overlap(U, V)
principal_angles(U, V)
project_onto_subspace(x, U)
compute_signal_covariance(class_means)
compute_alpha(U, C_signal)
compute_translation_mimicry(mu_a, mu_b, J_a, ridge_scale=1e-6)
ellipse_from_covariance(C2d)
load_eoptotype_rates(...)
load_eoptotype_jacobian(...)
```

### 6.3 Metadata discipline

Every output should save a config JSON with:

```text
script name
git commit if available
model checkpoint
epoch
rate file tags
conditions
logmars
orientations
retina_ppd
world_ppd
retina_size
world_size
date/time
analysis parameters
```

---

## 7. Decision rules for completion

After these analyses, the project should be considered complete enough for the narrowed deterministic encoding-geometry write-up if:

1. Translation mimicry runs and produces stable values across LogMAR.
2. Mimicry and α have been compared across LogMARs and the comparison is interpretable. Strong correlation would confirm they track the same underlying geometry; disagreement may also be scientifically meaningful (α uses signal covariance structure, mimicry uses pairwise class mean vectors with orientation-specific Jacobians — they can legitimately differ). If the comparison looks pathological (random, sign-flipped, or scale-discordant), check Jacobian units, normalization convention, and whether α is pooled or orientation-specific before interpreting. This is a diagnostic comparison, not a hard pass/fail threshold.
3. Phase landscape contextualizes fixed_center and trial-mean stabilized positions.
4. (Secondary) FEM component decomposition identifies whether drift or microsaccade components dominate the translation geometry, or else shows that real FEM geometry is distributed across components. This criterion is not blocking — the write-up can proceed without it if the first three criteria are met.
5. The central Jacobian figure can visually show the pushforward from eye-position covariance to population covariance.
6. No new noisy-observer or V2 claims are introduced.

---

## 8. Suggested final narrative after implementation

The intended final story should be:

> Fixational eye movements generate structured, low-dimensional population variability in V1. In a deterministic digital twin, this variability is explained by the local image-translation Jacobian: small retinal translations move activity along image-specific tangent planes in population space. These tangent planes provide a formal encoding-side substrate for the long-standing problem of separating stimulus identity from transformations applied to the same stimulus — making explicit when the V1 representation locally factorizes identity from translation and when it does not. For E-optotypes, translation mimicry quantifies when the neural difference between two orientations can be explained as a small translation of one orientation; this confusability varies with stimulus scale and retinal phase, explaining when FEM-induced variation aligns with or spares identity-relevant dimensions. Thus, FEM-induced variability is not undifferentiated noise; it is structured movement along image-specific transformation manifolds that downstream circuits could, in principle, discount for identity or exploit for localization.

---

## 9. Non-goals for this handoff

Do **not** add the following unless explicitly requested later:

- Poisson or noisy observer decoding
- V2/Rowley analyses
- behavioral hyperacuity claims
- optimized eye trajectories
- claims that fixed_center is a biological baseline
- claims that 0× amplitude is part of an amplitude tuning curve
- claims that \(J^\perp\) is literally identity space

Use careful language:

- “translation tangent space”
- “response variation not locally explained by translation”
- “candidate substrate for translation-tolerant identity readout”
- “deterministic class geometry”
- “identity/transformation confusability”
