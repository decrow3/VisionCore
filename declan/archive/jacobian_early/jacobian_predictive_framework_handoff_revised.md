# Predictive Jacobian framework analyses: handoff plan

## Purpose

This document specifies a gated set of analyses designed to test whether the image-translation Jacobian is more than an interesting model geometry. The goal is to determine whether the Jacobian can act as a **predictive object for active vision**, linking image content and eye-movement statistics to V1 population covariance and to task-dependent consequences of fixational eye movements (FEMs).

The analyses should be run as a sequence with explicit go/no-go criteria. The goal is not to make every analysis work at all costs. The goal is to determine which results are clean enough to promote to the main paper, which belong in supplement, and which should remain future directions.

## Top-level scientific questions

1. **Generality:** Does the Jacobian account generalize beyond E-optotypes to the natural-image/fixRSVP stimulus regime used in the main paper?
2. **Biological relevance:** Do model-derived Jacobian quantities predict aspects of real V1 FEM-dependent variance or covariance?
3. **Active-sensing prediction:** Can the Jacobian predict which eye-movement directions should help or hurt for a given image/task?

## Most important implementation warnings

Read these before implementing any step.

1. **Linearization validity is the hard prerequisite.** The Jacobian is local. For natural images, finite-difference step size and empirical displacement size matter more than they did for E-optotypes. If linearization is poor over realistic FEM amplitudes, all downstream claims must be restricted to small displacements or reframed as local tangent predictions.

2. **Use image-shuffled Jacobian nulls, not only random-subspace nulls.** Random subspaces test dimensionality; image-shuffled Jacobians test image specificity. The latter is the load-bearing null for the framework.

3. **Match time/frame binning between predicted and empirical quantities.** For the real-data bridge, predicted Jacobian drive and empirical FEM-dependent variance must be computed over the same stimulus/time windows. Otherwise the comparison is confounded.

4. **Control for stimulus drive and mean rate in real-data correlations.** A correlation between predicted Jacobian drive and empirical variance is only meaningful if it survives controls for mean predicted rate, observed PSTH amplitude, and/or stimulus-locked variance.

5. **Do scalar real-data bridge before pairwise bridge.** Pairwise covariance prediction requires clean model-unit to recorded-neuron correspondence. If the digital twin was not trained on the same recorded units, do not attempt a pairwise neural covariance comparison as a main result.

6. **Do not optimize full eye trajectories.** For the active-sensing prediction, restrict to 2D covariance ellipses under a fixed movement budget. Do not introduce full trajectory optimization, learned policies, or additional objective functions during this pass.

7. **Report failures as scope constraints, not project failures.** Each analysis has independent value. A weak result in one branch should not block publication of clean results from another.

---

# Step 0. Linearization validity check

## Goal

Before using the Jacobian to explain fixRSVP FEM covariance, test whether the first-order approximation is valid at realistic FEM displacement magnitudes for the stimulus regime of interest.

For a frame/image \(I\), response vector \(r(I)\), local displacement \(\Delta p\), and image-translation Jacobian \(J_I\), compare the actual model response change:

\[
\Delta r_{actual}(\Delta p) = r(I_{\Delta p}) - r(I)
\]

with the linearized prediction:

\[
\Delta r_{lin}(\Delta p) = J_I\Delta p.
\]

## Inputs

- fixRSVP/natural-image frames or short windows used in the main paper
- empirical FEM displacement distribution for the same trials/session where possible
- trained image-computable model
- model response function for shifted retinal inputs

## Finite-difference step sizes

Test several finite-difference steps for computing \(J_I\), for example:

- \(h = 0.125\) model pixels
- \(h = 0.25\) model pixels
- \(h = 0.5\) model pixels

Do not choose \(h\) only by numerical smoothness. Choose the main \(h\) based on both:

1. stability of the estimated Jacobian across neighboring \(h\) values, and
2. relevance to the empirical FEM displacement distribution.

If \(J_I\) changes substantially between \(h=0.125\) and \(h=0.5\), the tangent geometry is very local. That should be reported and downstream analyses should restrict to the corresponding displacement scale.

## Displacement grid for validation

Evaluate actual vs predicted response changes for displacements spanning the empirical FEM range, for example:

- 0.125, 0.25, 0.5, 1.0 model pixels
- empirical median displacement
- empirical 75th, 90th, and 95th percentile displacement magnitudes

Use multiple directions:

- x
- y
- diagonals
- random directions sampled from empirical eye displacement vectors

## Metrics

For each image/frame and displacement:

\[
R^2_{lin} = 1 - \frac{\|\Delta r_{actual} - \Delta r_{lin}\|^2}{\|\Delta r_{actual}\|^2}
\]

and

\[
\epsilon_{lin} = \frac{\|\Delta r_{actual} - \Delta r_{lin}\|}{\|\Delta r_{actual}\|}.
\]

Also compute cosine similarity:

\[
\cos(\Delta r_{actual}, \Delta r_{lin}).
\]

## Required plots

1. \(R^2_{lin}\) vs displacement magnitude
2. relative residual \(\epsilon_{lin}\) vs displacement magnitude
3. cosine similarity vs displacement magnitude
4. histograms of empirical FEM displacement magnitudes with tested ranges overlaid
5. example scatter plot of actual vs predicted response change magnitude

## Decision gate

Proceed to strong Step 1 claims only if linearization is reasonable for the central mass of empirical FEM displacements.

Suggested practical rule:

- **Good:** median \(R^2_{lin} > 0.5\) for displacements covering the median to 75th percentile of empirical FEM samples.
- **Conditional:** linearization only good below some displacement threshold. Restrict subsequent analyses to small-displacement windows or explicitly label results as local predictions.
- **Poor:** residuals exceed ~50% of actual response magnitude over typical FEM displacements. Do not present the Jacobian as explaining full FEM covariance; instead present it only as a small-displacement tangent analysis.

These thresholds are heuristics. Use plots and judgment.

---

# Step 1. fixRSVP Jacobian generalization

## Goal

Test whether the Jacobian account generalizes from E-optotypes to the natural-image/fixRSVP stimulus regime used in the main paper.

Primary question:

> Does the dominant model FEM covariance for fixRSVP/natural-image stimuli lie in the local image-translation Jacobian plane?

## Key point

Clarify the analysis unit before implementation. The phrase “frame” can mean several things in fixRSVP:

1. a single static image frame sampled over many empirical eye positions,
2. a short within-trial time window during one RSVP image presentation,
3. a pooled image identity across repeated trials,
4. a sliding window over stimulus and response time.

Choose one primary unit and document it. The covariance estimate must have enough samples to be stable.

Recommended first pass:

- Use repeated presentations or pooled time samples for a given image/frame identity where possible.
- If individual RSVP frames are too short for stable covariance, pool across matched image/frame instances or use short windows aligned to the same stimulus state.
- Keep the chosen binning consistent with Step 2.

## Inputs

- trained image-computable model
- fixRSVP stimulus frames or images
- empirical eye traces aligned to those frames
- code for rendering shifted retinal inputs
- model response extraction for the appropriate time window

## Procedure

For each selected image/frame/window \(I_k\):

1. Compute the baseline model response \(r(I_k)\).
2. Compute \(J_{I_k}\), the two-column image-translation Jacobian.
3. Generate model responses under empirical FEM samples for that same stimulus/window.
4. Compute model FEM covariance:

\[
C^{model}_{FEM,k} = \mathrm{Cov}_{s}(r(I_{k, \Delta p_s}))
\]

where \(s\) indexes empirical eye samples or matched simulated samples for the same stimulus unit.

5. Compute top two eigenvectors of \(C^{model}_{FEM,k}\).
6. Compare the Jacobian plane to the FEM covariance subspace.

## Metrics

Let \(U_J\) be an orthonormal basis for the column space of \(J_I\). Let \(U_{FEM}\) be the top-two eigenspace of \(C^{model}_{FEM}\).

Subspace alignment:

\[
A_J = \frac{1}{2}\|U_J^T U_{FEM}\|_F^2.
\]

FEM variance captured by the Jacobian plane:

\[
V_J = \frac{\mathrm{tr}(U_J^T C^{model}_{FEM} U_J)}{\mathrm{tr}(C^{model}_{FEM})}.
\]

Also report:

- total FEM covariance magnitude: \(\mathrm{tr}(C^{model}_{FEM})\)
- Jacobian norm: \(\|J_I\|_F\)
- condition number of \(J_I\)
- linearization validity for this frame/window from Step 0

## Mandatory nulls

### 1. Image-shuffled Jacobian null

Use \(J\) from a different image/frame/window while preserving the same \(C^{model}_{FEM}\). This tests image specificity.

Important refinement: construct image-shuffled controls that are matched, when possible, on broad stimulus/Jacobian statistics such as:

- \(\|J_I\|_F\)
- image contrast or gradient energy
- mean model response magnitude

The goal is to isolate geometric image-specificity from generic image energy.

### 2. Random 2D subspace null

Sample random two-dimensional subspaces in the model population space. This tests whether the alignment is meaningful relative to dimensionality alone.

### 3. Eye-shuffled or isotropic-eye null, optional

Keep the image fixed but shuffle/replace eye covariance to ask whether empirical FEM statistics matter beyond the local image geometry.

## Required plots

1. Distribution of \(A_J\) across fixRSVP frames/windows, with image-shuffled and random-subspace nulls.
2. Distribution of \(V_J\) across frames/windows.
3. Scatter plot of \(\|J_I\|_F\) or predicted FEM drive vs actual model FEM covariance magnitude.
4. Example stimulus image with schematic eye covariance ellipse and predicted neural covariance/tangent plane.
5. Linearization validity overlays for the same frames/windows, to show that alignment is interpreted in the valid local regime.

## Decision gate

Strong main-text result if:

- image-matched \(A_J\) reliably exceeds image-shuffled \(A_J\), and
- the result holds for frames/windows with acceptable Step 0 linearization, and
- random-subspace nulls are clearly lower.

Useful supplement result if:

- alignment is above random subspace but not clearly above image-shuffled controls.

Weak result / future direction if:

- alignment is not above image-shuffled controls, or
- linearization is poor for the displacement regime needed to estimate covariance.

---

# Step 1.5. Pipeline consistency check on E-optotypes

## Goal

Verify that the generalized Jacobian pipeline reproduces the already established E-optotype findings when applied to the optotype stimulus regime.

This step guards against a mismatch where fixRSVP analyses and E-optotype analyses use subtly different definitions of response windows, covariance, projection, or normalization.

## Procedure

Using the same code path developed for Step 1:

1. Apply the Jacobian + FEM covariance pipeline to the E-optotype stimuli.
2. Recompute:
   - Jacobian/FEM subspace alignment
   - variance captured by \(J\)
   - image-shuffled or orientation-shuffled nulls
3. Confirm that results are consistent with prior E-optotype Jacobian/mimicry outputs.

## Decision gate

This is not meant as a new result unless discrepancies arise. If discrepancies appear, resolve the pipeline definitions before interpreting Step 1.

---

# Step 2. Real-data scalar bridge

## Goal

Test whether model-derived Jacobian quantities predict aspects of real V1 FEM-dependent variance/covariance.

Primary question:

> Do images/time windows predicted by the model to be highly translation-sensitive show larger empirically estimated FEM-dependent variability in recorded V1?

## First-pass bridge: scalar predicted drive

For each stimulus/time window \(k\), compute model-predicted FEM drive:

\[
g_k = \mathrm{tr}(J_{I_k}\Sigma_{eye,k}J_{I_k}^T).
\]

Here \(\Sigma_{eye,k}\) must be computed over the same eye samples/time window used to estimate empirical FEM-dependent variance.

## Empirical target quantities

Use one or more empirical V1 measures computed over matched windows:

- eye-dependent rate variance
- magnitude of estimated \(C_{FEM}\)
- variance reduction after eye correction
- covariance reduction after eye correction
- McFarland-style eye component magnitude if available

## Critical binning requirement

Predicted and empirical quantities must use the same analysis unit:

- same stimulus/frame/window definition
- same time lag convention if neural responses are delayed relative to stimulus/eye position
- same trial inclusion criteria
- same eye-validity masks
- same pooling across repeated image presentations

Document this explicitly in the output README.

## Controls for stimulus drive

A naive correlation between \(g_k\) and empirical variance may be confounded by stimulus drive. Strong images could produce both larger model Jacobians and larger empirical variance.

Therefore compute:

1. raw correlation:

\[
\mathrm{corr}(g_k, v^{emp}_k)
\]

2. partial correlation controlling for:
   - mean predicted model rate
   - observed PSTH amplitude or mean empirical rate
   - stimulus-locked variance / response magnitude per frame
   - image contrast or gradient energy if available

3. image-shuffled control:
   - replace \(J_{I_k}\) with matched-shuffled \(J\)
   - recompute \(g_k^{shuffle}\)
   - compare matched vs shuffled correlations

## Required success criteria

Do not rely only on an absolute correlation threshold.

A clean positive result requires:

1. image-matched correlation is reliably greater than image-shuffled correlation, using permutation or bootstrap confidence intervals;
2. matched advantage persists after controlling for mean rate / PSTH amplitude / stimulus-locked variance;
3. the absolute effect is nontrivial, with a heuristic target of Spearman or Pearson \(r \gtrsim 0.3\) when estimates are not too noisy.

If the matched-vs-shuffled advantage is strong but absolute \(r\) is below 0.3 due to noisy empirical estimates, report effect size and confidence intervals rather than treating this as automatic failure.

## Required plots

1. Scatter: predicted drive \(g_k\) vs empirical FEM-dependent variance.
2. Matched vs image-shuffled correlation distribution.
3. Partial correlation summary with controls.
4. Binned plot: empirical variance for low/mid/high predicted-drive windows.
5. Optional: example high-drive and low-drive images/windows.

## Decision gate

Main-text candidate if:

- matched \(g_k\) predicts empirical FEM-dependent variance beyond controls and above image-shuffled nulls.

Supplement candidate if:

- matched effect exists but is weak/noisy.

Future direction if:

- no matched advantage over image-shuffled controls, especially after rate/stimulus controls.

---

# Step 3. Constrained eye-movement prediction

## Goal

Use the Jacobian to predict which eye-movement directions should help or hurt for a given stimulus/task.

This is the active-sensing payoff. It should be framed as a local prediction, not as a claim that real FEMs are globally optimal.

## Scope lock

Do only the constrained covariance-ellipse analysis.

Do not:

- optimize full trajectories
- add learned policies
- introduce extra objective functions after seeing results
- claim real FEMs are optimal unless the evidence is exceptionally clean
- introduce arbitrary combined objectives with free parameters as main results

## Candidate eye covariances

Compare:

1. empirical \(\Sigma_{eye}\)
2. isotropic covariance with matched trace:

\[
\Sigma_{iso} = \frac{\mathrm{tr}(\Sigma_{eye})}{2}I
\]

3. covariance aligned to helpful direction
4. covariance aligned to harmful direction
5. random covariance directions under same trace

Represent rank-1 directional covariances as:

\[
\Sigma(v) = c vv^T, \quad \|v\|=1
\]

where \(c = \mathrm{tr}(\Sigma_{eye})\).

## Objective 1: identity-confounding cost

For an orientation identity task, define signal subspace \(U_{signal}\). The identity-confounding cost of an eye covariance is:

\[
H(\Sigma) = \mathrm{tr}(U_{signal}^T J_I \Sigma J_I^T U_{signal}).
\]

High \(H\) means the eye movement injects variation along identity-signal dimensions, likely confounding identity readout.

For rank-1 covariance \(\Sigma(v)=cvv^T\):

\[
H(v) = c\, v^T J_I^T U_{signal}U_{signal}^T J_I v.
\]

The maximally harmful direction is the top eigenvector of:

\[
J_I^T U_{signal}U_{signal}^T J_I.
\]

The minimally harmful direction is the bottom eigenvector.

## Objective 2: spatial-sampling / sensitivity benefit

Define a simple sampling sensitivity objective:

\[
S(\Sigma) = \mathrm{tr}(J_I \Sigma J_I^T).
\]

For rank-1 covariance:

\[
S(v) = c\, v^T J_I^T J_I v.
\]

The maximally sensitive direction is the top eigenvector of:

\[
J_I^T J_I.
\]

The least sensitive direction is the bottom eigenvector.

## Do not use a free-parameter combined objective in the main analysis

Do not make

\[
B = S - \lambda H
\]

a main result unless a principled value of \(\lambda\) is defined before looking at results.

If useful, discuss the tradeoff qualitatively or report \(S\) and \(H\) as a two-dimensional Pareto-style plot.

## Primary reporting unit: percentile over directions

Angles to optimal/worst directions can be misleading if the objective landscape is flat. Therefore report percentile scores over all possible 2D directions.

For each stimulus/phase:

1. sample many directions \(v_\theta = (\cos\theta, \sin\theta)\), e.g. 720 angles;
2. compute \(S(v_\theta)\) and \(H(v_\theta)\);
3. compute where empirical \(\Sigma_{eye}\) falls in this distribution.

Report:

- percentile of empirical FEM in spatial-sensitivity objective \(S\)
- percentile of empirical FEM in identity-confounding cost \(H\)
- same percentiles for isotropic covariance where applicable
- distance/angle to max/min directions as secondary metrics

## Required plots

1. Polar plots of \(S(v)\) and \(H(v)\) for example phases/images.
2. Empirical FEM covariance ellipse overlaid with helpful/harmful axes.
3. Distribution of empirical FEM percentiles across phases/images.
4. Scatter or Pareto plot of spatial-sensitivity percentile vs identity-confounding percentile.
5. If E-optotype: show how these predictions relate to translation mimicry crossover or phase landscape.

## Decision gate

Main-text candidate if:

- the Jacobian yields clear, interpretable helpful/harmful axes;
- empirical FEM is systematically biased relative to isotropic or random directions;
- the result explains or predicts an existing observed pattern, such as real/stabilized crossover or spatial-information changes.

Supplement candidate if:

- objective landscapes are meaningful but empirical FEM is intermediate.

Discussion-only if:

- results depend strongly on objective choice or are too heterogeneous for a concise claim.

---

# Step 4. Pairwise real-data bridge, only if warranted

## Goal

Predict the pairwise structure of empirical FEM-driven covariance using model-derived Jacobian sensitivities.

This is the strongest biological bridge but the highest-risk analysis.

## Attempt only if all conditions are met

1. Step 2 scalar bridge gives a positive result.
2. The digital twin was trained on the same recorded units/session being analyzed.
3. Model outputs map cleanly to recorded neurons.
4. Empirical pairwise FEM covariance estimates are stable.
5. Eye and stimulus alignment are reliable enough at the relevant time scale.

If the model was trained on a different population/session or does not have one-to-one model-unit correspondence with the recorded neurons, do not attempt this as a main result.

## Prediction

For neurons \(i,j\), with row vectors \(J_i\) and \(J_j\):

\[
C^{pred}_{ij} = J_i \Sigma_{eye} J_j^T.
\]

Compare \(C^{pred}_{ij}\) to empirical FEM-driven pairwise covariance or covariance reduction after eye correction.

## Controls

- image-shuffled \(J\)
- trial-shuffled eye traces
- mean-rate / PSTH covariance controls
- distance or RF-overlap controls if available

## Required plots

1. predicted vs empirical pairwise FEM covariance scatter
2. matched vs image-shuffled prediction performance
3. binned empirical covariance by predicted covariance quantile
4. example neuron pairs with high vs low predicted FEM covariance

## Decision gate

Main text only if the result is very clean. Otherwise supplement or future direction.

---

# Expected repository hooks

Based on the current repo snapshot, useful starting points include:

- `README.md`: installation and data-package setup for VisionCore and sibling repositories.
- `V2tracetestingRowley1.py`: example code loading a Rowley fixRSVP dataset through `DictDataset.load`, loading session data from `DataRowleyV1V2`, extracting `FixRsvpStim` trials, aligning `NoiseHistory`, eye traces, and spikes, and building arrays such as `binned_spikes`, `eyepos`, `image_id`, and trial-level metadata.
- Existing Jacobian/mimicry/phase-landscape outputs and scripts from the E-optotype analysis.
- Existing McFarland/covariance decomposition code used in the main FEM V1 analysis.

The coding agent should search the repo for existing helper functions before reimplementing:

- model loading and response prediction
- shifted stimulus rendering / gaze-contingent rendering
- Jacobian finite differences
- covariance estimation
- eye-trace alignment
- spatial information / translation covariance utilities
- figure helpers

## Suggested output location

Create a new directory, for example:

```text
results/jacobian_predictive_framework/
```

or, if repo conventions prefer scripts:

```text
scripts/jacobian_predictive_framework/
```

Save:

- `.npz` numeric outputs for each step
- `.csv` summary tables
- `.md` result summaries
- `.png` and `.pdf` figures
- a README documenting exact inputs and commit/state

---

# Required result summary format

For each step, write a short markdown summary with:

1. question
2. dataset/session/stimulus regime
3. analysis unit definition
4. key metrics
5. nulls/controls
6. pass/fail gate outcome
7. figures generated
8. interpretation
9. caveats
10. recommended paper placement: main text, supplement, discussion, or omit

---

# Framework-level outcomes

## If Step 1 works

Claim:

> The Jacobian framework generalizes beyond optotypes and predicts FEM covariance geometry for natural-image/fixRSVP stimuli.

## If Step 2 works

Claim:

> Model-derived Jacobian quantities predict aspects of empirically observed V1 FEM-dependent variability.

## If Step 3 works

Claim:

> The Jacobian framework predicts which eye-movement directions should be helpful or harmful for a given image/task.

## If all three work

Strong framework claim:

> The image-translation Jacobian is a predictive object for active vision: it links image content and eye-movement statistics to V1 population covariance and to task-dependent representational consequences.

## If only Step 1 works

Narrow but valuable claim:

> The Jacobian provides a general model-based mechanism for low-dimensional FEM covariance across controlled and natural-image stimuli.

## If Step 1 fails

Restrict claim:

> The E-optotype Jacobian result remains a controlled demonstration of identity/transformation geometry, but natural-image generality is not established.

---

# Final scope-control rule

Attempt all steps, but promote only clean results.

Do not let an ambiguous Step 2 or Step 3 result delay the paper indefinitely. A clean Step 1 alone is already valuable. A clean Step 1 plus either Step 2 or Step 3 is enough to justify the Jacobian framework as a main-text contribution.
